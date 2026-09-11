// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals3.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

struct OneToNVMIGroupStoreOpPattern
    : OneToNOpConversionPattern<VMIGroupStoreOp> {
  using OneToNOpConversionPattern<VMIGroupStoreOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerAlignedPackedByteStore(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, Value destination, Value offset,
      int64_t numGroups) const {
    MLIRContext *ctx = rewriter.getContext();
    auto ui16 = IntegerType::get(
        ctx, 16, IntegerType::SignednessSemantics::Unsigned);
    auto ui8 = IntegerType::get(
        ctx, 8, IntegerType::SignednessSemantics::Unsigned);
    auto packed16Type = VRegType::get(ctx, 128, ui16);
    auto packed8Type = VRegType::get(ctx, 256, ui8);
    Value packed16 =
        rewriter
            .create<VpackOp>(op.getLoc(), packed16Type, valueParts.front(),
                             rewriter.getStringAttr("LOWER"))
            .getResult();
    Value packed8 =
        rewriter
            .create<VpackOp>(op.getLoc(), packed8Type, packed16,
                             rewriter.getStringAttr("LOWER"))
            .getResult();
    FailureOr<MaskType> packedMaskType =
        getMaskTypeForVReg(packed8Type, ctx);
    if (failed(packedMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed byte group_store mask type");
    }
    FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *packedMaskType, numGroups, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed byte group_store mask");
    }
    rewriter.create<VstsOp>(op.getLoc(), Type{}, packed8, destination, offset,
                            rewriter.getStringAttr("NORM_B8"), *storeMask);
    return success();
  }

  LogicalResult emitPackedByteStoreStream(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value destination,
      Value offset, ArrayRef<Value> values, ArrayRef<int64_t> advances) const {
    Type destinationElementType = getMemoryElementType(destination.getType());
    Value storeBase = materializeBufferPointer(
        destination, destinationElementType,
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!storeBase) {
      return rewriter.notifyMatchFailure(
          op, "unaligned packed byte group_store requires a ptr-compatible destination");
    }
    storeBase = rewriter
                    .create<AddPtrOp>(op.getLoc(), storeBase.getType(),
                                      storeBase, offset)
                    .getResult();
    return emitStatefulStoreStream(op, storeBase, values, advances, rewriter);
  }

  FailureOr<Value> materializePackedSlots1Group(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      int64_t group, VRegType firstType, MaskType maskType, Value allMask,
      Value packed) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType || vregType != firstType) {
      return rewriter.notifyMatchFailure(
          op, "packed group_store requires uniform vreg parts");
    }
    Value splat = rewriter
                      .create<VdupOp>(op.getLoc(), firstType, value, allMask,
                                      rewriter.getStringAttr("LOWEST"))
                      .getResult();
    FailureOr<Value> laneMask =
        createLaneRangeMask(op.getLoc(), maskType, group, group + 1, rewriter);
    if (failed(laneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store lane mask");
    }
    return rewriter
        .create<VselOp>(op.getLoc(), firstType, splat, packed, *laneMask)
        .getResult();
  }

  FailureOr<Value> buildPackedSlots1Value(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMILayoutAttr layout, VRegType firstType,
      MaskType maskType, Value allMask) const {
    Value packed = rewriter
                       .create<VdupOp>(op.getLoc(), firstType,
                                       valueParts.front(), allMask,
                                       rewriter.getStringAttr("LOWEST"))
                       .getResult();
    for (int64_t group = 1; group < layout.getNumGroups(); ++group) {
      FailureOr<Value> nextPacked = materializePackedSlots1Group(
          op, rewriter, valueParts[group], group, firstType, maskType, allMask,
          packed);
      if (failed(nextPacked)) {
        return failure();
      }
      packed = *nextPacked;
    }
    return packed;
  }

  LogicalResult lowerSlots1PackedUnitStride(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMIVRegType valueVMIType, VMILayoutAttr layout,
      Value destination, Value offset) const {
    auto firstType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!firstType) {
      return rewriter.notifyMatchFailure(op, "group_store value must be vreg");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(firstType, rewriter.getContext());
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), firstType, rewriter);
    bool unsupportedMasks = failed(maskType) || failed(allMask);
    if (unsupportedMasks) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for packed group_store mask");
    }
    FailureOr<Value> packed = buildPackedSlots1Value(
        op, rewriter, valueParts, layout, firstType, *maskType, *allMask);
    if (failed(packed)) {
      return failure();
    }
    if (isKnownAddressAligned(destination, offset,
                               valueVMIType.getElementType(), 32)) {
      FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
          op.getLoc(), *maskType, layout.getNumGroups(), rewriter);
      if (failed(storeMask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create packed group_store store mask");
      }
      rewriter.create<VstsOp>(op.getLoc(), Type{}, *packed, destination, offset,
                              nullptr, *storeMask);
    } else if (failed(emitPackedSlots1StoreStream(
                   op, rewriter, destination, offset, valueVMIType, *packed,
                   layout.getNumGroups()))) {
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitPackedSlots1StoreStream(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value destination,
      Value offset, VMIVRegType valueVMIType, Value packed,
      int64_t numGroups) const {
    Value storeBase = materializeBufferPointer(
        destination, valueVMIType.getElementType(),
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!storeBase) {
      return rewriter.notifyMatchFailure(
          op, "packed unaligned group_store requires a ptr-compatible destination");
    }
    storeBase = rewriter
                    .create<AddPtrOp>(op.getLoc(), storeBase.getType(), storeBase,
                                      offset)
                    .getResult();
    SmallVector<Value> streamValues{packed};
    SmallVector<int64_t> streamAdvances{numGroups};
    return emitStatefulStoreStream(op, storeBase, streamValues, streamAdvances,
                                   rewriter);
  }

  LogicalResult lowerSlots1PointStores(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMIVRegType valueVMIType, Value destination,
      Value offset, Value rowStride) const {
    std::optional<std::string> pointDist =
        getPointStoreDistToken(valueVMIType.getElementType());
    if (!pointDist) {
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_store requires 1PT_B8/B16/B32 store support");
    }
    for (auto [group, value] : llvm::enumerate(valueParts)) {
      if (failed(emitSlots1PointStore(op, value, group, destination, offset,
                                      rowStride,
                                      *pointDist, rewriter))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitSlots1PointStore(
      VMIGroupStoreOp op, Value value, int64_t group, Value destination,
      Value offset, Value rowStride, StringRef pointDist,
      OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(op, "group_store value must be vreg");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(vregType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_store mask");
    }
    FailureOr<Value> mask =
        createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create slots=1 group_store mask");
    }
    Value groupOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, 0, rewriter);
    rewriter.create<VstsOp>(op.getLoc(), Type{}, value, destination, groupOffset,
                            rewriter.getStringAttr(pointDist), *mask);
    return success();
  }

  LogicalResult lowerSlots1(VMIGroupStoreOp op, OpAdaptor adaptor,
                            OneToNPatternRewriter &rewriter,
                            VMIVRegType valueVMIType, VMILayoutAttr layout,
                            Value destination, Value offset,
                            Value rowStride) const {
    ValueRange valueParts = adaptor.getValue();
    bool hasExpectedArity =
        static_cast<int64_t>(valueParts.size()) == layout.getNumGroups();
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(op,
                                         "slots=1 group_store arity mismatch");
    }
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(valueVMIType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_store requires supported element width");
    }
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (constantRowStride && *constantRowStride == 1 &&
        succeeded(lanesPerPart) && layout.getNumGroups() <= *lanesPerPart) {
      return lowerSlots1PackedUnitStride(
          op, rewriter, valueParts, valueVMIType, layout, destination, offset);
    }
    if (constantRowStride && *constantRowStride <= 0) {
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_store requires positive row_stride when row_stride is constant");
    }
    return lowerSlots1PointStores(op, rewriter, valueParts, valueVMIType,
                                  destination, offset, rowStride);
  }

  LogicalResult lowerScalarGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      Value destination, Value offset) const {
    ValueRange valueParts = adaptor.getValue();
    bool invalidValueArity = valueParts.size() != 1;
    if (invalidValueArity) {
      return rewriter.notifyMatchFailure(
          op, "scalar group_store requires one physical value part");
    }
    auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(
          op, "scalar group_store value must be vreg");
    }
    std::optional<std::string> pointDist =
        getPointStoreDistToken(valueVMIType.getElementType());
    if (!pointDist) {
      return rewriter.notifyMatchFailure(
          op, "scalar group_store requires point-store support");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(valueType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for scalar group_store mask");
    }
    FailureOr<Value> mask =
        createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create scalar group_store mask");
    }
    rewriter.create<VstsOp>(op.getLoc(), Type{}, valueParts.front(),
                            destination, offset,
                            rewriter.getStringAttr(*pointDist), *mask);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerDeinterleaved2GroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, const VMIGroupStoreLayoutFact &fact,
      Value destination, Value offset, Value rowStride) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    std::string reason;
    if (failed(checkDeinterleaved2GroupStoreChunkShape(
            valueVMIType, fact.groupSize, &lanesPerPart, &groupCount,
            &chunksPerGroup, &reason))) {
      return failure();
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "group_store requires vstsx2 INTLV element support");
    }
    ValueRange valueParts = adaptor.getValue();
    int64_t chunksPerPart = groupCount * chunksPerGroup;
    bool hasExpectedArity =
        static_cast<int64_t>(valueParts.size()) == 2 * chunksPerPart;
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_store arity mismatch");
    }
    for (int64_t group = 0; group < groupCount; ++group) {
      for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
        int64_t lowIndex = group * chunksPerGroup + chunk;
        int64_t highIndex = chunksPerPart + lowIndex;
        if (failed(emitDeinterleaved2GroupStorePair(
                op, valueParts[lowIndex], valueParts[highIndex], group, chunk,
                lanesPerPart, destination, offset, rowStride, *dist,
                rewriter))) {
          return failure();
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitDeinterleaved2GroupStorePair(
      VMIGroupStoreOp op, Value low, Value high, int64_t group, int64_t chunk,
      int64_t lanesPerPart, Value destination, Value offset, Value rowStride,
      StringRef dist, OneToNPatternRewriter &rewriter) const {
    bool mismatchedTypes = low.getType() != high.getType();
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "vstsx2 group_store requires matching low/high types");
    }
    auto vregType = dyn_cast<VRegType>(low.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(op, "group_store value must be vreg");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_store mask");
    }
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, chunk * 2 * lanesPerPart,
        rewriter);
    rewriter.create<Vstsx2Op>(op.getLoc(), low, high, destination, chunkOffset,
                              rewriter.getStringAttr(dist), *mask);
    return success();
  }

  LogicalResult emitContiguousGroupStorePart(
      VMIGroupStoreOp op, Value value, int64_t index, int64_t chunksPerGroup,
      int64_t lanesPerPart, Value destination, Value offset, Value rowStride,
      OneToNPatternRewriter &rewriter) const {
    if (chunksPerGroup <= 0) {
      return rewriter.notifyMatchFailure(
          op, "group_store requires positive chunks per group");
    }
    int64_t safeChunksPerGroup = chunksPerGroup;
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(op,
                                         "group_store value must be vreg");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_store mask");
    }
    int64_t group = index / safeChunksPerGroup;
    int64_t chunkInGroup = index % safeChunksPerGroup;
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, chunkInGroup * lanesPerPart,
        rewriter);
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                            destination, chunkOffset, /*dist=*/nullptr, *mask);
    return success();
  }

  LogicalResult lowerContiguousGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, const VMIGroupStoreLayoutFact &fact,
      Value destination, Value offset, Value rowStride) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(
            op, valueVMIType, fact.groupSize, &lanesPerPart, &groupCount,
            &chunksPerGroup, rewriter))) {
      return failure();
    }
    ValueRange valueParts = adaptor.getValue();
    bool hasExpectedArity = static_cast<int64_t>(valueParts.size()) ==
                            groupCount * chunksPerGroup;
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(op, "group_store arity mismatch");
    }
    for (auto [index, value] : llvm::enumerate(valueParts)) {
      if (failed(emitContiguousGroupStorePart(
              op, value, index, chunksPerGroup, lanesPerPart, destination,
              offset, rowStride, rewriter))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitAlignedSlots8Contiguous(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ArrayRef<Value> groupOffsets, Value destination,
      int64_t numGroups) const {
    for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        return rewriter.notifyMatchFailure(op,
                                           "group_store value must be vreg");
      }
      FailureOr<MaskType> maskType =
          getMaskTypeForVReg(vregType, rewriter.getContext());
      if (failed(maskType)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_store mask");
      }
      int64_t activeGroups = std::min<int64_t>(8, numGroups - slotBlock * 8);
      FailureOr<Value> mask = createPrefixMaskForActiveLanes(
          op.getLoc(), *maskType, activeGroups, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create slots=8 group_store mask");
      }
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, groupOffsets[slotBlock],
                              /*dist=*/nullptr, *mask);
    }
    return success();
  }

  LogicalResult lowerSlots8Contiguous(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      Value destination, Value offset, Value rowStride,
      int64_t numGroups) const {
    ValueRange valueParts = adaptor.getValue();
    FailureOr<std::pair<SmallVector<Value>, bool>> offsetPlan =
        buildSlots8GroupOffsets(op, valueParts, destination, offset, rowStride,
                                "", rewriter);
    if (failed(offsetPlan)) {
      return failure();
    }
    SmallVector<Value> groupOffsets = std::move(offsetPlan->first);
    bool useDirectAccess = offsetPlan->second;

    if (!useDirectAccess) {
      SmallVector<int64_t> advances =
          buildSlots8StreamAdvances(numGroups, valueParts.size());
      if (failed(emitGroupStoreStream(op, destination, offset, valueParts,
                                      advances, rewriter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    if (failed(emitAlignedSlots8Contiguous(
            op, rewriter, valueParts, groupOffsets, destination, numGroups))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitOneBlockGroupStorePart(
      VMIGroupStoreOp op, Value value, int64_t part,
      VMIVRegType valueVMIType, const OneBlockGroupStorePlan &plan,
      Value destination, Value offset, Value rowStride, Value blockStride,
      Value repeatStride, OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "one-block group_store value must be vreg");
    }
    FailureOr<Value> mask = createContiguousStoreMask(
        op.getLoc(), valueVMIType, part, vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create one-block group_store mask");
    }
    Value partOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, part * plan.groupsPerPart,
        /*inGroupLaneOffset=*/0, rewriter);
    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), destination.getType(),
                                       destination, partOffset)
                     .getResult();
    rewriter.create<VsstbOp>(op.getLoc(), /*updated_base=*/Type{}, value, base,
                             blockStride, repeatStride, *mask);
    return success();
  }

  LogicalResult lowerOneBlockGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, const VMIGroupStoreLayoutFact &fact,
      Value destination, Value offset, Value rowStride) const {
    FailureOr<OneBlockGroupStorePlan> plan =
        getOneBlockGroupStorePlan(op, valueVMIType, fact, nullptr);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build one-block group_store vsstb plan");
    }
    ValueRange valueParts = adaptor.getValue();
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    bool hasExpectedArity =
        static_cast<int64_t>(valueParts.size()) ==
        ceilDivNonNegative(numGroups, plan->groupsPerPart);
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(
          op, "one-block group_store physical arity mismatch");
    }
    Value blockStride =
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), plan->blockStride, 16);
    Value repeatStride = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), 0, 16);
    for (auto [part, value] : llvm::enumerate(valueParts)) {
      if (failed(emitOneBlockGroupStorePart(
              op, value, part, valueVMIType, *plan, destination, offset,
              rowStride, blockStride, repeatStride, rewriter))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitAlignedSlots8LaneStride(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ArrayRef<Value> groupOffsets, Value destination,
      MaskType maskType, int64_t numGroups, StringRef dist) const {
    for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
      if (!isa<VRegType>(value.getType())) {
        return rewriter.notifyMatchFailure(op,
                                           "group_store value must be vreg");
      }
      int64_t activeGroups = std::min<int64_t>(8, numGroups - slotBlock * 8);
      FailureOr<Value> mask = createPrefixMaskForActiveLanes(
          op.getLoc(), maskType, activeGroups, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create packed slots=8 group_store mask");
      }
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, groupOffsets[slotBlock],
                              rewriter.getStringAttr(dist), *mask);
    }
    return success();
  }

  FailureOr<std::pair<SmallVector<Value>, bool>> buildSlots8GroupOffsets(
      VMIGroupStoreOp op, ValueRange valueParts, Value destination,
      Value offset, Value rowStride, StringRef dist,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> groupOffsets;
    bool useDirectAccess = true;
    for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        (void)rewriter.notifyMatchFailure(op, "group_store value must be vreg");
        return failure();
      }
      Value groupOffset = createGroupChunkOffset(
          op.getLoc(), offset, rowStride, slotBlock * 8, 0, rewriter);
      groupOffsets.push_back(groupOffset);
      useDirectAccess &= isDirectMemoryDistAddressLegal(
          op.getDestination(), groupOffset,
          getMemoryElementType(destination.getType()), vregType,
          VPTOMemoryOpFamily::Store, dist);
    }
    return std::make_pair(std::move(groupOffsets), useDirectAccess);
  }

  FailureOr<SmallVector<Value>> materializeLaneStrideStreamValues(
      VMIGroupStoreOp op, ValueRange valueParts, VMIVRegType valueVMIType,
      VMILayoutAttr layout, OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr compactLayout = VMILayoutAttr::getGroupSlots(
        rewriter.getContext(), layout.getNumGroups(), layout.getSlots());
    auto compactType = VMIVRegType::get(
        rewriter.getContext(), valueVMIType.getElementCount(),
        valueVMIType.getElementType(), compactLayout);
    FailureOr<SmallVector<Value>> compactValues = materializeEnsureLayoutConversion(
        op, valueParts, valueVMIType, compactType,
        *this->getTypeConverter(), rewriter);
    bool invalidCompactValues = failed(compactValues) ||
                                compactValues->size() != valueParts.size();
    if (invalidCompactValues) {
      return rewriter.notifyMatchFailure(
          op, "failed to compact unaligned slots=8 group_store");
    }
    return *compactValues;
  }

  SmallVector<int64_t> buildSlots8StreamAdvances(
      int64_t numGroups, size_t valueCount) const {
    SmallVector<int64_t> advances;
    advances.reserve(valueCount);
    for (size_t slotBlock = 0; slotBlock < valueCount; ++slotBlock) {
      advances.push_back(std::min<int64_t>(8, numGroups - slotBlock * 8));
    }
    return advances;
  }

  LogicalResult lowerSlots8LaneStride(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, VMILayoutAttr layout, Value destination,
      Value offset, Value rowStride, int64_t numGroups) const {
    std::optional<std::string> dist =
        getLaneStrideStoreDistToken(layout, valueVMIType.getElementType());
    std::optional<StringRef> maskGranularity =
        getLaneStrideStoreMaskGranularity(layout,
                                          valueVMIType.getElementType());
    if (!dist || !maskGranularity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported slots=8 lane_stride group_store packing");
    }
    ValueRange valueParts = adaptor.getValue();
    auto maskType = MaskType::get(rewriter.getContext(), *maskGranularity);
    FailureOr<std::pair<SmallVector<Value>, bool>> offsetPlan =
        buildSlots8GroupOffsets(op, valueParts, destination, offset, rowStride,
                                *dist, rewriter);
    if (failed(offsetPlan)) {
      return failure();
    }
    SmallVector<Value> groupOffsets = std::move(offsetPlan->first);
    bool useDirectAccess = offsetPlan->second;
    if (!useDirectAccess) {
      FailureOr<SmallVector<Value>> compactValues =
          materializeLaneStrideStreamValues(op, valueParts, valueVMIType,
                                            layout, rewriter);
      if (failed(compactValues)) {
        return failure();
      }
      SmallVector<int64_t> advances =
          buildSlots8StreamAdvances(numGroups, compactValues->size());
      if (failed(emitGroupStoreStream(op, destination, offset, *compactValues,
                                      advances, rewriter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }
    if (failed(emitAlignedSlots8LaneStride(
            op, rewriter, valueParts, groupOffsets, destination, maskType,
            numGroups, *dist))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<Value> materializeCompactSmallGroupValue(
      VMIGroupStoreOp op, ValueRange valueParts, VMIVRegType valueVMIType,
      VMILayoutAttr layout, OneToNPatternRewriter &rewriter) const {
    Value compactValue = valueParts.front();
    bool alreadyCompact = layout.getLaneStride() == 1;
    if (alreadyCompact) {
      return compactValue;
    }
    VMILayoutAttr compactLayout = VMILayoutAttr::getGroupSlots(
        rewriter.getContext(), layout.getNumGroups(), layout.getSlots());
    auto compactVMIType = VMIVRegType::get(
        rewriter.getContext(), valueVMIType.getElementCount(),
        valueVMIType.getElementType(), compactLayout);
    FailureOr<SmallVector<Value>> packed = materializeEnsureLayoutConversion(
        op, valueParts, valueVMIType, compactVMIType,
        *this->getTypeConverter(), rewriter);
    bool invalidPacked = failed(packed) || packed->size() != 1;
    if (invalidPacked) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize compact group_store layout");
    }
    return packed->front();
  }

  LogicalResult emitAlignedCompactSmallGroupStore(
      VMIGroupStoreOp op, Value compactValue, VMIVRegType valueVMIType,
      Value destination, Value offset,
      OneToNPatternRewriter &rewriter) const {
    auto compactType = dyn_cast<VRegType>(compactValue.getType());
    std::optional<std::string> normalDist =
        getX2MemoryDistToken(valueVMIType.getElementType(), "NORM");
    if (!compactType || !normalDist) {
      return rewriter.notifyMatchFailure(
          op, "aligned compact group_store requires a supported vreg element type");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(compactType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive aligned compact group_store mask type");
    }
    FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *maskType, valueVMIType.getElementCount(), rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create aligned compact group_store mask");
    }
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, compactValue,
                            destination, offset,
                            rewriter.getStringAttr(*normalDist), *storeMask);
    return success();
  }

  LogicalResult emitUnalignedCompactSmallGroupStore(
      VMIGroupStoreOp op, Value compactValue, VMIVRegType valueVMIType,
      Value destination, Value offset,
      OneToNPatternRewriter &rewriter) const {
    Value elementBase =
        rewriter.create<AddPtrOp>(op.getLoc(), destination.getType(),
                                  destination, offset)
            .getResult();
    SmallVector<Value> streamValues{compactValue};
    SmallVector<int64_t> streamAdvances{valueVMIType.getElementCount()};
    return emitStatefulStoreStream(op, elementBase, streamValues, streamAdvances,
                                   rewriter);
  }

  LogicalResult lowerCompactSmallGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      VMILayoutAttr layout, Value destination, Value offset) const {
    ValueRange valueParts = adaptor.getValue();
    bool invalidValueArity = valueParts.size() != 1;
    if (invalidValueArity) {
      return rewriter.notifyMatchFailure(
          op, "compact small group_store requires one physical value part");
    }
    auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!valueType || !isa<PtrType>(destination.getType())) {
      return rewriter.notifyMatchFailure(
          op, "compact small group_store requires vreg and ptr operands");
    }

    FailureOr<Value> compactValue = materializeCompactSmallGroupValue(
        op, valueParts, valueVMIType, layout, rewriter);
    if (failed(compactValue)) {
      return failure();
    }

    if (isKnownAddressAligned(destination, offset,
                              valueVMIType.getElementType(), 32)) {
      if (failed(emitAlignedCompactSmallGroupStore(
              op, *compactValue, valueVMIType, destination, offset, rewriter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    if (failed(emitUnalignedCompactSmallGroupStore(
            op, *compactValue, valueVMIType, destination, offset, rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<std::tuple<Value, Value, Value>> buildPackedByteStoreBlock(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts,
      VRegType firstVRegType, MaskType maskType, Value slotIndex,
      Value destination, Value offset, Value rowStride, int64_t numGroups,
      int64_t blockStart) const {
    FailureOr<Value> zero =
        createZeroVector(op.getLoc(), firstVRegType, rewriter);
    if (failed(zero)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store accumulator");
    }
    Value merged = *zero;
    for (int64_t localPart = 0; localPart < 4; ++localPart) {
      int64_t partIndex = blockStart / 8 + localPart;
      if (partIndex >= static_cast<int64_t>(valueParts.size())) {
        break;
      }
      int64_t activeGroups = std::min<int64_t>(8, numGroups - partIndex * 8);
      if (activeGroups <= 0) {
        break;
      }
      FailureOr<Value> nextMerged = mergePackedByteStoreBlockPart(
          op, rewriter, valueParts[partIndex], slotIndex, merged,
          firstVRegType, maskType, localPart * 8, activeGroups);
      if (failed(nextMerged)) {
        return failure();
      }
      merged = *nextMerged;
    }
    int64_t activeGroups = std::min<int64_t>(32, numGroups - blockStart);
    FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store store mask");
    }
    Value groupOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, blockStart, 0, rewriter);
    return std::make_tuple(merged, *storeMask, groupOffset);
  }

  FailureOr<Value> mergePackedByteStoreBlockPart(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      Value slotIndex, Value merged, VRegType valueType, MaskType maskType,
      int64_t laneStart, int64_t activeGroups) const {
    Value selected = rewriter
                         .create<VselrOp>(op.getLoc(), valueType, value, slotIndex)
                         .getResult();
    FailureOr<Value> laneMask = createLaneRangeMask(
        op.getLoc(), maskType, laneStart, laneStart + activeGroups, rewriter);
    if (failed(laneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store lane mask");
    }
    return rewriter
        .create<VselOp>(op.getLoc(), valueType, selected, merged, *laneMask)
        .getResult();
  }

  FailureOr<Value> buildPackedByteStatefulValue(
      VMIGroupStoreOp op, Value merged,
      OneToNPatternRewriter &rewriter) const {
    MLIRContext *ctx = rewriter.getContext();
    auto ui16 = IntegerType::get(
        ctx, 16, IntegerType::SignednessSemantics::Unsigned);
    auto ui8 = IntegerType::get(
        ctx, 8, IntegerType::SignednessSemantics::Unsigned);
    auto packed16Type = VRegType::get(ctx, 128, ui16);
    auto packed8Type = VRegType::get(ctx, 256, ui8);
    Value packed16 = rewriter
                         .create<VpackOp>(op.getLoc(), packed16Type, merged,
                                          rewriter.getStringAttr("LOWER"))
                         .getResult();
    return rewriter
        .create<VpackOp>(op.getLoc(), packed8Type, packed16,
                         rewriter.getStringAttr("LOWER"))
        .getResult();
  }

  void emitPackedByteDirectStore(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value merged,
      Value destination, Value groupOffset, Value storeMask) const {
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, merged,
                            destination, groupOffset,
                            rewriter.getStringAttr("PK4_B32"), storeMask);
  }

  LogicalResult emitPackedByteStoreBlocks(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VRegType firstVRegType, MaskType maskType,
      Value slotIndex, Value destination, Value offset, Value rowStride,
      int64_t numGroups, bool useDirectPack4) const {
    SmallVector<Value> statefulValues;
    SmallVector<int64_t> statefulAdvances;
    for (int64_t blockStart = 0; blockStart < numGroups; blockStart += 32) {
      FailureOr<std::tuple<Value, Value, Value>> block =
          buildPackedByteStoreBlock(
              op, rewriter, valueParts, firstVRegType, maskType, slotIndex,
              destination, offset, rowStride, numGroups, blockStart);
      if (failed(block)) {
        return failure();
      }
      Value merged = std::get<0>(*block);
      Value storeMask = std::get<1>(*block);
      Value groupOffset = std::get<2>(*block);
      if (useDirectPack4) {
        emitPackedByteDirectStore(op, rewriter, merged, destination,
                                  groupOffset, storeMask);
        continue;
      }
      FailureOr<Value> statefulValue =
          buildPackedByteStatefulValue(op, merged, rewriter);
      if (failed(statefulValue)) {
        return failure();
      }
      statefulValues.push_back(*statefulValue);
      statefulAdvances.push_back(
          std::min<int64_t>(32, numGroups - blockStart));
    }
    if (!useDirectPack4 &&
        failed(emitPackedByteStoreStream(op, rewriter, destination, offset,
                                         statefulValues, statefulAdvances))) {
      return failure();
    }
    return success();
  }

  FailureOr<std::pair<MaskType, Value>> buildPackedByteSelectors(
      VMIGroupStoreOp op, VRegType firstVRegType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(firstVRegType, rewriter.getContext());
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), firstVRegType, rewriter);
    bool failedMasks = failed(maskType) || failed(allMask);
    if (failedMasks) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported element type for packed group_store mask");
      return failure();
    }
    auto indexElementType = IntegerType::get(
        rewriter.getContext(),
        pto::getPTOStorageElemBitWidth(firstVRegType.getElementType()));
    auto indexType = VRegType::get(rewriter.getContext(),
                                   firstVRegType.getElementCount(),
                                   indexElementType);
    FailureOr<Value> slotIndex = createGroupSlotIndexVector(
        op.getLoc(), indexType, /*groupSize=*/8, /*baseGroupSlot=*/0,
        rewriter);
    if (failed(slotIndex)) {
      (void)rewriter.notifyMatchFailure(
          op, "failed to create packed group_store lane selector");
      return failure();
    }
    return std::make_pair(*maskType, *slotIndex);
  }

  LogicalResult lowerPackedByteSlots8(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMIVRegType valueVMIType, VMILayoutAttr layout,
      Value destination, Value offset, Value rowStride, int64_t numGroups,
      VRegType firstVRegType) const {
    bool laneStrided = layout.hasLaneStride();
    for (Value value : valueParts) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType || vregType != firstVRegType) {
        return rewriter.notifyMatchFailure(
            op, "packed slots=8 group_store requires uniform vreg parts");
      }
    }
    bool alignedSinglePart =
        !laneStrided && numGroups == 8 && valueParts.size() == 1 &&
        isKnownAddressAligned(destination, offset,
                              valueVMIType.getElementType(), 32);
    if (alignedSinglePart) {
      if (failed(lowerAlignedPackedByteStore(
              op, rewriter, valueParts, destination, offset, numGroups))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<std::pair<MaskType, Value>> selectors =
        buildPackedByteSelectors(op, firstVRegType, rewriter);
    if (failed(selectors)) {
      return failure();
    }
    bool useDirectPack4 = isDirectMemoryDistAddressLegal(
        op.getDestination(), op.getOffset(),
        getMemoryElementType(op.getDestination().getType()), firstVRegType,
        VPTOMemoryOpFamily::Store, "PK4_B32");
    if (failed(emitPackedByteStoreBlocks(
            op, rewriter, valueParts, firstVRegType, selectors->first,
            selectors->second,
            destination, offset, rowStride, numGroups, useDirectPack4))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<std::optional<VRegType>> getSlots8FirstVRegType(
      VMIGroupStoreOp op, OpAdaptor adaptor, VMILayoutAttr layout,
      OneToNPatternRewriter &rewriter) const {
    ValueRange valueParts = adaptor.getValue();
    bool hasExpectedArity = static_cast<int64_t>(valueParts.size()) ==
                            ceilDivNonNegative(layout.getNumGroups(), 8);
    if (!hasExpectedArity) {
      (void)rewriter.notifyMatchFailure(op,
                                        "slots=8 group_store arity mismatch");
      return failure();
    }
    if (valueParts.empty()) {
      return std::optional<VRegType>();
    }
    auto firstVRegType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!firstVRegType) {
      (void)rewriter.notifyMatchFailure(op,
                                        "group_store value must be vreg");
      return failure();
    }
    return std::optional<VRegType>(firstVRegType);
  }

  LogicalResult lowerSlots8Dispatch(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, VMILayoutAttr layout, Value destination,
      Value offset, Value rowStride) const {
    int64_t numGroups = layout.getNumGroups();
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    bool hasUnitRowStride = constantRowStride && *constantRowStride == 1;
    if (!hasUnitRowStride) {
      return rewriter.notifyMatchFailure(
          op, "slots=8 group_store requires constant unit row_stride");
    }
    FailureOr<std::optional<VRegType>> firstVRegType =
        getSlots8FirstVRegType(op, adaptor, layout, rewriter);
    if (failed(firstVRegType)) {
      return failure();
    }
    if (*firstVRegType && isPackedByteGroupStore(
                               op.getDestination().getType(), **firstVRegType)) {
      return lowerPackedByteSlots8(
            op, rewriter, adaptor.getValue(), valueVMIType, layout, destination,
            offset, rowStride, numGroups, **firstVRegType);
    }
    if (layout.hasLaneStride()) {
      return lowerSlots8LaneStride(op, adaptor, rewriter, valueVMIType, layout,
                                   destination, offset, rowStride, numGroups);
    }
    return lowerSlots8Contiguous(op, adaptor, rewriter, destination, offset,
                                 rowStride, numGroups);
  }

  enum class GroupStoreLayoutKind { Scalar, Compact, Slots1, Slots8, General };

  GroupStoreLayoutKind classifyGroupStoreLayout(
      VMIGroupStoreOp op, VMIVRegType valueVMIType, VMILayoutAttr layout) const {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    bool scalar = numGroups == 1 && valueVMIType.getElementCount() == 1;
    if (scalar) {
      return GroupStoreLayoutKind::Scalar;
    }
    bool compact = isCompactSmallGroupStore(
        layout, valueVMIType, numGroups, getConstantIndexValue(op.getRowStride()));
    if (compact) {
      return GroupStoreLayoutKind::Compact;
    }
    bool slots1 = layout && layout.isGroupSlots() && layout.getSlots() == 1 &&
                  layout.getNumGroups() == numGroups;
    if (slots1) {
      return GroupStoreLayoutKind::Slots1;
    }
    bool slots8 = layout && layout.isGroupSlots() && layout.getSlots() == 8 &&
                  layout.getNumGroups() == numGroups;
    return slots8 ? GroupStoreLayoutKind::Slots8 : GroupStoreLayoutKind::General;
  }

  LogicalResult lowerGeneralGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      Value destination, Value offset, Value rowStride) const {
    VMILayoutSupport supports;
    FailureOr<VMIGroupStoreLayoutFact> fact =
        supports.getGroupStoreLayoutFact(op, valueVMIType);
    if (failed(fact)) {
      return rewriter.notifyMatchFailure(
          op, "group_store layout does not match the support table");
    }
    if (fact->blockClass == VMIGroupBlockClass::OneBlock) {
      return lowerOneBlockGroupStore(
          op, adaptor, rewriter, valueVMIType, *fact, destination, offset,
          rowStride);
    }
    int64_t d2LanesPerPart = 0;
    int64_t d2GroupCount = 0;
    int64_t d2ChunksPerGroupPerPart = 0;
    std::string d2Reason;
    bool hasDeinterleaved2Shape = succeeded(checkDeinterleaved2GroupStoreChunkShape(
        valueVMIType, fact->groupSize, &d2LanesPerPart, &d2GroupCount,
        &d2ChunksPerGroupPerPart, &d2Reason));
    if (hasDeinterleaved2Shape) {
      return lowerDeinterleaved2GroupStore(
          op, adaptor, rewriter, valueVMIType, *fact, destination, offset,
          rowStride);
    }
    return lowerContiguousGroupStore(
        op, adaptor, rewriter, valueVMIType, *fact, destination, offset,
        rowStride);
  }

  LogicalResult lowerByLayout(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      VMILayoutAttr layout, Value destination, Value offset,
      Value rowStride) const {
    GroupStoreLayoutKind layoutKind =
        classifyGroupStoreLayout(op, valueVMIType, layout);
    if (layoutKind == GroupStoreLayoutKind::Scalar) {
      return lowerScalarGroupStore(op, adaptor, rewriter, valueVMIType,
                                   destination, offset);
    }
    if (layoutKind == GroupStoreLayoutKind::Compact) {
      return lowerCompactSmallGroupStore(op, adaptor, rewriter, valueVMIType,
                                         layout, destination, offset);
    }

    if (layoutKind == GroupStoreLayoutKind::Slots1) {
      return lowerSlots1(op, adaptor, rewriter, valueVMIType, layout,
                         destination, offset, rowStride);
    }
    if (layoutKind == GroupStoreLayoutKind::Slots8) {
      return lowerSlots8Dispatch(op, adaptor, rewriter, valueVMIType, layout,
                                 destination, offset, rowStride);
    }

    return lowerGeneralGroupStore(op, adaptor, rewriter, valueVMIType,
                                  destination, offset, rowStride);
  }

public:
  LogicalResult
  matchAndRewrite(VMIGroupStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    VMILayoutAttr layout = valueVMIType.getLayoutAttr();

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "group_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "group_store offset must convert to one value",
        rewriter);
    FailureOr<Value> rowStride = getSingleValue(
        op, adaptor.getRowStride(),
        "group_store row_stride must convert to one value", rewriter);
    bool operandsConverted = succeeded(destination) && succeeded(offset) &&
                             succeeded(rowStride);
    if (!operandsConverted) {
      return failure();
    }

    // Unified scalar vstore is lowered to group_store(num_groups=1) before
    // layout assignment.  The producer may therefore carry the slots=8
    // layout selected by group_slot_load, even though the logical operation
    // still writes one scalar.  Preserve the scalar memory semantics here;
    // a masked ordinary vsts would require a 32-byte-aligned destination.
    return lowerByLayout(op, adaptor, rewriter, valueVMIType, layout,
                         *destination, *offset, *rowStride);
  }
};

struct OneToNVMIMaskedStoreOpPattern
    : OneToNOpConversionPattern<VMIMaskedStoreOp> {
  using OneToNOpConversionPattern<VMIMaskedStoreOp>::OneToNOpConversionPattern;

private:
  FailureOr<int64_t> emitLaneStrideMaskedStorePart(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      Value mask, VMIVRegType valueVMIType, Value destination, Value offset,
      StringRef dist, StringRef maskGranularity, int64_t index,
      int64_t semanticOffset) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    bool invalidTypes = !vregType || !isa<MaskType>(mask.getType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "lane_stride masked_store parts must be vreg/mask");
    }
    FailureOr<int64_t> activeLanes =
        getActiveDataLanesInPhysicalChunk(valueVMIType, index);
    if (failed(activeLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute lane_stride masked_store active lanes");
    }
    if (*activeLanes == 0) {
      return 0;
    }
    FailureOr<Value> storeMask = createDenseLaneStrideStorePredicate(
        op.getLoc(), valueVMIType, index, mask, maskGranularity, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compact lane_stride masked_store predicate");
    }
    Value chunkOffset =
        createChunkOffset(op.getLoc(), offset, semanticOffset, rewriter);
    bool illegalAddress = !isDirectMemoryDistAddressLegal(
        destination, chunkOffset, valueVMIType.getElementType(), vregType,
        VPTOMemoryOpFamily::Store, dist);
    if (illegalAddress) {
      return rewriter.notifyMatchFailure(
          op, "lane_stride masked_store requires a proven target alignment "
              "for every physical store chunk");
    }
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                            destination, chunkOffset,
                            rewriter.getStringAttr(dist), *storeMask);
    return *activeLanes;
  }

  LogicalResult lowerLaneStride(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType, Value destination, Value offset,
      StringRef dist, StringRef maskGranularity) const {
    VMILayoutAttr valueLayout = valueVMIType.getLayoutAttr();
    VMILayoutAttr maskLayout = maskVMIType.getLayoutAttr();
    if (!valueLayout || !maskLayout || valueLayout != maskLayout) {
      return rewriter.notifyMatchFailure(
          op, "lane_stride masked_store requires matching value/mask layouts");
    }
    int64_t semanticOffset = 0;
    for (auto [index, valueAndMask] :
         llvm::enumerate(llvm::zip_equal(valueParts, maskParts))) {
      auto [value, mask] = valueAndMask;
      FailureOr<int64_t> activeLanes = emitLaneStrideMaskedStorePart(
          op, rewriter, value, mask, valueVMIType, destination, offset, dist,
          maskGranularity, index, semanticOffset);
      if (failed(activeLanes)) {
        return failure();
      }
      semanticOffset += *activeLanes;
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitContiguousMaskedStorePart(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      Value mask, VMIVRegType valueVMIType, Value destination, Value offset,
      int64_t index, int64_t lanesPerPart) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    bool invalidTypes = !vregType || !isa<MaskType>(mask.getType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "masked_store converted parts must be vreg/mask");
    }
    FailureOr<int64_t> activeLanes =
        getContiguousActiveDataLanes(valueVMIType, index);
    if (failed(activeLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute masked_store active lanes");
    }
    if (*activeLanes == 0) {
      return success();
    }
    FailureOr<Value> storeMask = createMaskedStorePredicate(
        op.getLoc(), valueVMIType, index, mask, vregType, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize masked_store predicate");
    }
    Value chunkOffset =
        createChunkOffset(op.getLoc(), offset, index * lanesPerPart, rewriter);
    bool illegalAddress = !isDirectMemoryDistAddressLegal(
        destination, chunkOffset, valueVMIType.getElementType(), vregType,
        VPTOMemoryOpFamily::Store, /*dist=*/{});
    if (illegalAddress) {
      return rewriter.notifyMatchFailure(
          op, "masked_store requires a proven target alignment for every "
              "physical store chunk");
    }
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                            destination, chunkOffset, /*dist=*/nullptr,
                            *storeMask);
    return success();
  }

  FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
  materializeContiguousMaskedStoreParts(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType) const {
    SmallVector<Type> contiguousValueTypes;
    contiguousValueTypes.reserve(valueParts.size());
    for (Value value : valueParts) {
      contiguousValueTypes.push_back(value.getType());
    }
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousValueTypes, valueVMIType.getLayoutAttr(),
        contiguousLayout, valueVMIType.getElementType(), rewriter);
    if (failed(storeParts)) {
      return failure();
    }
    SmallVector<Type> contiguousMaskTypes;
    contiguousMaskTypes.reserve(maskParts.size());
    for (Value mask : maskParts) {
      contiguousMaskTypes.push_back(mask.getType());
    }
    FailureOr<SmallVector<Value>> storeMasks = materializeMaskLayoutConversion(
        op, maskParts, contiguousMaskTypes, maskVMIType.getLayoutAttr(),
        contiguousLayout, rewriter);
    if (failed(storeMasks)) {
      return failure();
    }
    bool mismatchedArity = storeParts->size() != storeMasks->size();
    if (mismatchedArity) {
      return rewriter.notifyMatchFailure(
          op, "masked_store converted value/mask arity mismatch");
    }
    return std::make_pair(std::move(*storeParts), std::move(*storeMasks));
  }

  LogicalResult lowerContiguous(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType, Value destination, Value offset,
      int64_t lanesPerPart) const {
    FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>> converted =
        materializeContiguousMaskedStoreParts(op, rewriter, valueParts,
                                              maskParts, valueVMIType,
                                              maskVMIType);
    if (failed(converted)) {
      return failure();
    }

    for (auto [index, valueAndMask] :
         llvm::enumerate(llvm::zip_equal(converted->first,
                                         converted->second))) {
      auto [value, mask] = valueAndMask;
      if (failed(emitContiguousMaskedStorePart(
              op, rewriter, value, mask, valueVMIType, destination, offset,
              index, lanesPerPart))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerByLayout(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType, Value destination, Value offset,
      int64_t lanesPerPart) const {
    std::optional<std::string> dist =
        getDenseLaneStrideStoreDistToken(valueVMIType);
    if (dist) {
      std::optional<StringRef> maskGranularity =
          getDenseLaneStrideMaskedStoreMaskGranularity(valueVMIType);
      if (maskGranularity) {
        return lowerLaneStride(op, rewriter, valueParts, maskParts,
                               valueVMIType, maskVMIType, destination, offset,
                               *dist, *maskGranularity);
      }
    }
    return lowerContiguous(op, rewriter, valueParts, maskParts, valueVMIType,
                           maskVMIType, destination, offset, lanesPerPart);
  }

public:

  LogicalResult
  matchAndRewrite(VMIMaskedStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "masked_store requires known physical lanes per part");
    }

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "masked_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "masked_store offset must convert to one value", rewriter);
    bool invalidAddressOperands = failed(destination) || failed(offset);
    if (invalidAddressOperands) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    bool arityMismatch = valueParts.size() != maskParts.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "masked_store value/mask physical arity mismatch");
    }

    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    return lowerByLayout(op, rewriter, valueParts, maskParts, valueVMIType,
                         maskVMIType, *destination, *offset, *lanesPerPart);
  }
};

struct OneToNVMIGroupBroadcastLoadOpPattern
    : OneToNOpConversionPattern<VMIGroupBroadcastLoadOp> {
  using OneToNOpConversionPattern<VMIGroupBroadcastLoadOp>::OneToNOpConversionPattern;

private:
  LogicalResult validateDirectE2BBasicContract(
      VMIGroupBroadcastLoadOp op, Value source, int64_t numGroups,
      unsigned elementBits, VMILayoutAttr layout,
      OneToNPatternRewriter &rewriter) const {
    bool contiguousPacketLayout = layout && layout.isContiguous();
    bool splitPacketLayout = layout && layout.isDeinterleaved() &&
                             (layout.getFactor() == 2 ||
                              layout.getFactor() == 4) &&
                             layout.getLaneStride() == 1;
    if (!contiguousPacketLayout && !splitPacketLayout) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires contiguous result "
              "layout for direct group size or deinterleaved=2/4 result "
              "layout for split group size");
    }
    if (elementBits != 16 && elementBits != 32) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires b16 or b32 element type");
    }
    std::optional<int64_t> stride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (!stride || *stride != 1) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires constant unit source_group_stride");
    }
    if (!isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires !pto.ptr source");
    }
    if (numGroups != 8) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires num_groups = 8");
    }
    return success();
  }

  FailureOr<Value> emitE2BPacket(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Type packetType, int64_t chunk,
      StringRef e2bDist) const {
    if (!isa<VRegType>(packetType)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load result must be vreg");
    }
    Value packetOffset =
        createChunkOffset(op.getLoc(), offset, chunk * 8, rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), packetType, Type{}, source, packetOffset,
                        rewriter.getStringAttr(e2bDist))
        .getResult();
  }

  FailureOr<SmallVector<Value>> emitE2BPackets(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, ArrayRef<Type> resultTypes,
      int64_t chunksPerPart, StringRef e2bDist) const {
    SmallVector<Value> packets;
    packets.reserve(chunksPerPart);
    for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
      FailureOr<Value> packet = emitE2BPacket(
          op, rewriter, source, offset, resultTypes[chunk], chunk, e2bDist);
      if (failed(packet)) {
        return failure();
      }
      packets.push_back(*packet);
    }
    return packets;
  }

  FailureOr<SmallVector<Value>> buildE2BResults(
      VMIGroupBroadcastLoadOp op, ArrayRef<Value> packets,
      ArrayRef<Type> resultTypes, int64_t factor, int64_t chunksPerPart,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        int64_t flatIndex = part * chunksPerPart + chunk;
        if (resultTypes[flatIndex] != resultTypes[chunk]) {
          return rewriter.notifyMatchFailure(
              op, "group_broadcast_load E2B reused packet type mismatch");
        }
        results.push_back(packets[chunk]);
      }
    }
    return results;
  }

  FailureOr<std::tuple<StringRef, int64_t, int64_t>> validateDirectE2BShape(
      VMIGroupBroadcastLoadOp op, Value source,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      int64_t numGroups, unsigned elementBits, VMILayoutAttr layout,
      OneToNPatternRewriter &rewriter) const {
    if (failed(validateDirectE2BBasicContract(op, source, numGroups,
                                              elementBits, layout, rewriter))) {
      return failure();
    }
    FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
    bool invalidChunks = failed(chunksPerPart) || *chunksPerPart <= 0;
    if (invalidChunks) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load requires known chunks per part");
    }
    int64_t factor = layout.getFactor();
    FailureOr<int64_t> uniformChunks = validateDirectE2BChunks(
        op, resultVMIType, factor, *chunksPerPart, rewriter);
    if (failed(uniformChunks)) {
      return failure();
    }
    bool invalidArity =
        static_cast<int64_t>(resultTypes.size()) != factor * *chunksPerPart;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load physical arity mismatch");
    }
    if (*chunksPerPart != 1) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load expected one E2B packet in each part");
    }
    StringRef e2bDist = elementBits == 16 ? "E2B_B16" : "E2B_B32";
    return std::make_tuple(e2bDist, factor, *chunksPerPart);
  }

  FailureOr<int64_t> validateDirectE2BChunks(
      VMIGroupBroadcastLoadOp op, VMIVRegType resultVMIType, int64_t factor,
      int64_t chunksPerPart, OneToNPatternRewriter &rewriter) const {
    for (int64_t part = 1; part < factor; ++part) {
      FailureOr<int64_t> currentChunks =
          getDataChunksInPart(resultVMIType, part);
      bool nonUniformChunks =
          failed(currentChunks) || *currentChunks != chunksPerPart;
      if (nonUniformChunks) {
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load requires uniform chunks per part");
      }
    }
    return chunksPerPart;
  }

  LogicalResult lowerDirectE2B(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes, int64_t numGroups, unsigned elementBits,
      VMILayoutAttr layout) const {
    FailureOr<std::tuple<StringRef, int64_t, int64_t>> shape =
        validateDirectE2BShape(op, source, resultVMIType, resultTypes,
                               numGroups, elementBits, layout, rewriter);
    if (failed(shape)) {
      return failure();
    }
    StringRef e2bDist = std::get<0>(*shape);
    int64_t factor = std::get<1>(*shape);
    int64_t chunksPerPart = std::get<2>(*shape);
    FailureOr<SmallVector<Value>> packets = emitE2BPackets(
        op, rewriter, source, offset, resultTypes, chunksPerPart, e2bDist);
    if (failed(packets)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = buildE2BResults(
        op, *packets, resultTypes, factor, chunksPerPart, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildDirectBRCResult(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride, Type resultType,
      int64_t group, StringRef brcDist) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC result must be vreg");
    }
    Value groupOffset = createGroupChunkOffset(
        op.getLoc(), offset, sourceGroupStride, group, 0, rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, groupOffset,
                        rewriter.getStringAttr(brcDist))
        .getResult();
  }

  FailureOr<int64_t> validateDirectBRCShape(
      VMIGroupBroadcastLoadOp op, Value source, ArrayRef<Type> resultTypes,
      int64_t numGroups, OneToNPatternRewriter &rewriter) const {
    if (numGroups <= 0) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC requires positive num_groups");
    }
    int64_t safeNumGroups = numGroups;
    bool invalidArity =
        static_cast<int64_t>(resultTypes.size()) % safeNumGroups != 0;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC result arity is not divisible by num_groups");
    }
    if (!isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC lowering requires !pto.ptr source");
    }
    int64_t chunksPerGroup =
        static_cast<int64_t>(resultTypes.size()) / safeNumGroups;
    bool invalidChunkArity =
        chunksPerGroup <= 0 ||
        static_cast<int64_t>(resultTypes.size()) !=
            safeNumGroups * chunksPerGroup;
    if (invalidChunkArity) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC physical arity mismatch");
    }
    return chunksPerGroup;
  }

  FailureOr<std::pair<VMIVRegType, SmallVector<Type>>>
  buildGroupBroadcastFallbackSourcePlan(
      VMIGroupBroadcastLoadOp op, VMIVRegType resultVMIType,
      int64_t numGroups, OneToNPatternRewriter &rewriter) const {
    std::optional<int64_t> stride =
        getConstantIndexValue(op.getSourceGroupStride());
    int64_t slots = (stride && *stride == 1) ? 8 : 1;
    auto sourceVMIType = VMIVRegType::get(
        rewriter.getContext(), numGroups, resultVMIType.getElementType(),
        VMILayoutAttr::getGroupSlots(rewriter.getContext(), numGroups, slots));
    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceVMIType);
    if (failed(sourceArity)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load fallback cannot derive physical types");
    }
    Type sourceElementType = getVMIPhysicalDataElementType(sourceVMIType);
    FailureOr<int64_t> sourceLanesPerPart =
        getDataLanesPerPart(sourceElementType);
    if (failed(sourceLanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load fallback cannot derive source lanes");
    }
    SmallVector<Type> sourceTypes;
    sourceTypes.reserve(*sourceArity);
    for (int64_t i = 0; i < *sourceArity; ++i) {
      sourceTypes.push_back(VRegType::get(
          rewriter.getContext(), *sourceLanesPerPart, sourceElementType));
    }
    return std::make_pair(sourceVMIType, std::move(sourceTypes));
  }

  LogicalResult lowerDirectBRC(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      ArrayRef<Type> resultTypes, int64_t numGroups,
      StringRef brcDist) const {
    FailureOr<int64_t> chunksPerGroup = validateDirectBRCShape(
        op, source, resultTypes, numGroups, rewriter);
    if (failed(chunksPerGroup)) {
      return failure();
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      int64_t group = static_cast<int64_t>(index) / *chunksPerGroup;
      FailureOr<Value> result = buildDirectBRCResult(
          op, rewriter, source, offset, sourceGroupStride, resultType, group,
          brcDist);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerGroupSlotFallback(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      int64_t numGroups) const {
    FailureOr<std::pair<VMIVRegType, SmallVector<Type>>> sourcePlan =
        buildGroupBroadcastFallbackSourcePlan(op, resultVMIType, numGroups,
                                              rewriter);
    if (failed(sourcePlan)) {
      return failure();
    }
    SmallVector<Value> sourceParts;
    if (failed(lowerGroupSlotLoadParts(
            op, source, offset, sourceGroupStride, sourcePlan->first,
            sourcePlan->second, numGroups, rewriter, sourceParts))) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerGroupBroadcastParts(
            op, sourceParts, sourcePlan->first, resultVMIType, resultTypes,
            numGroups, rewriter, results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<bool> tryLowerDirectBRC(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes, int64_t numGroups,
      const FailureOr<VMIGroupBroadcastLoadDirectFact> &directFact) const {
    bool candidate = succeeded(directFact) &&
                     directFact->kind == VMIGroupBroadcastLoadDirectKind::BRC &&
                     !resultTypes.empty();
    if (!candidate) {
      return false;
    }
    unsigned bits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    std::optional<StringRef> dist;
    if (bits == 8) {
      dist = StringRef("BRC_B8");
    } else if (bits == 16) {
      dist = StringRef("BRC_B16");
    } else if (bits == 32) {
      dist = StringRef("BRC_B32");
    }
    auto firstType = dyn_cast<VRegType>(resultTypes.front());
    bool legal = dist && firstType && isDirectMemoryDistAddressLegal(
                                  op.getSource(), op.getOffset(),
                                  resultVMIType.getElementType(), firstType,
                                  VPTOMemoryOpFamily::Load, *dist);
    if (!legal) {
      return false;
    }
    if (failed(lowerDirectBRC(op, rewriter, source, offset, sourceGroupStride,
                              resultTypes, numGroups, *dist))) {
      return failure();
    }
    return true;
  }

  FailureOr<bool> tryLowerDirectE2B(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes, int64_t numGroups,
      const FailureOr<VMIGroupBroadcastLoadDirectFact> &directFact) const {
    bool candidate = succeeded(directFact) &&
                     directFact->kind == VMIGroupBroadcastLoadDirectKind::E2B &&
                     !resultTypes.empty();
    if (!candidate) {
      return false;
    }
    unsigned bits = directFact->layout.elementBits;
    StringRef dist = bits == 16 ? StringRef("E2B_B16") : StringRef("E2B_B32");
    auto firstType = dyn_cast<VRegType>(resultTypes.front());
    bool legal = (bits == 16 || bits == 32) && firstType &&
                 isDirectMemoryDistAddressLegal(
                     op.getSource(), op.getOffset(),
                     resultVMIType.getElementType(), firstType,
                     VPTOMemoryOpFamily::Load, dist);
    if (!legal) {
      return false;
    }
    // E2B materializes one packet per physical result chunk.  A contiguous
    // result spanning multiple chunks cannot reuse one packet for every
    // chunk; route it through the group-slot fallback instead.
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
    bool requiresGroupSlotFallback =
        resultLayout && resultLayout.isContiguous() &&
        (failed(chunksPerPart) || *chunksPerPart != 1);
    if (requiresGroupSlotFallback) {
      return false;
    }
    if (failed(lowerDirectE2B(op, rewriter, source, offset, resultVMIType,
                              resultTypes, numGroups, bits,
                              resultVMIType.getLayoutAttr()))) {
      return failure();
    }
    return true;
  }

  LogicalResult lowerDirectOrFallback(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes, int64_t numGroups,
      const FailureOr<VMIGroupBroadcastLoadDirectFact> &directFact) const {
    FailureOr<bool> loweredBRC = tryLowerDirectBRC(
        op, rewriter, source, offset, sourceGroupStride, resultVMIType,
        resultTypes, numGroups, directFact);
    if (failed(loweredBRC)) {
      return failure();
    }
    if (*loweredBRC) {
      return success();
    }

    FailureOr<bool> loweredE2B = tryLowerDirectE2B(
        op, rewriter, source, offset, resultVMIType, resultTypes, numGroups,
        directFact);
    if (failed(loweredE2B)) {
      return failure();
    }
    if (*loweredE2B) {
      return success();
    }
    return lowerGroupSlotFallback(op, rewriter, source, offset,
                                  sourceGroupStride, resultVMIType, resultTypes,
                                  numGroups);
  }

public:
  LogicalResult
  matchAndRewrite(VMIGroupBroadcastLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "group_broadcast_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "group_broadcast_load offset must convert to one value", rewriter);
    FailureOr<Value> sourceGroupStride = getSingleValue(
        op, adaptor.getSourceGroupStride(),
        "group_broadcast_load source_group_stride must convert to one value",
        rewriter);
    bool invalidOperands =
        failed(source) || failed(offset) || failed(sourceGroupStride);
    if (invalidOperands) {
      return failure();
    }

    VMILayoutSupport supports;
    std::string supportReason;
    FailureOr<VMIGroupBroadcastLoadLayoutFact> loadFact =
        supports.getGroupBroadcastLoadLayoutFact(op, &supportReason);
    if (failed(loadFact)) {
      return rewriter.notifyMatchFailure(
          op, Twine("group_broadcast_load has no registered support: ") +
                  supportReason);
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<VMIGroupBroadcastLoadDirectFact> directFact =
        supports.getGroupBroadcastLoadDirectFact(op);
    return lowerDirectOrFallback(op, rewriter, *source, *offset,
                                 *sourceGroupStride, resultVMIType, resultTypes,
                                 numGroups, directFact);
  }
};


