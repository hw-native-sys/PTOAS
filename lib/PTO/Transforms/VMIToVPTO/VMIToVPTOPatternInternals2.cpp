// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals2.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

struct OneToNVMIGroupSlotLoadOpPattern
    : OneToNOpConversionPattern<VMIGroupSlotLoadOp> {
  using OneToNOpConversionPattern<
      VMIGroupSlotLoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupSlotLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    bool invalidLayout =
        !layout || !layout.isGroupSlots() || layout.getSlots() <= 0;
    if (invalidLayout) {
      return rewriter.notifyMatchFailure(
          op, "group_slot_load requires explicit group_slots layout");
    }

    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "group_slot_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "group_slot_load offset must convert to one value", rewriter);
    FailureOr<Value> sourceGroupStride = getSingleValue(
        op, adaptor.getSourceGroupStride(),
        "group_slot_load source_group_stride must convert to one value",
        rewriter);
    bool invalidOperands =
        failed(source) || failed(offset) || failed(sourceGroupStride);
    if (invalidOperands) {
      return failure();
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    int64_t numGroups = op.getNumGroupsAttr().getInt();

    SmallVector<Value> results;
    if (failed(lowerGroupSlotLoadParts(op, *source, *offset, *sourceGroupStride,
                                       resultVMIType, resultTypes, numGroups,
                                       rewriter, results))) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIMaskedLoadOpPattern
    : OneToNOpConversionPattern<VMIMaskedLoadOp> {
  using OneToNOpConversionPattern<VMIMaskedLoadOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> materializeMaskedLoadPart(
      VMIMaskedLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value mask, Value passthru, Type resultType,
      int64_t index, int64_t lanesPerPart) const {
    bool invalidPartTypes = !isa<MaskType>(mask.getType()) ||
                            passthru.getType() != resultType ||
                            !isa<VRegType>(resultType);
    if (invalidPartTypes) {
      return rewriter.notifyMatchFailure(
          op, "masked_load physical part type mismatch");
    }
    Value chunkOffset = createChunkOffset(
        op.getLoc(), offset, index * lanesPerPart, rewriter);
    Value loaded = rewriter
                       .create<VldsOp>(op.getLoc(), resultType, Type{}, source,
                                       chunkOffset, nullptr)
                       .getResult();
    return rewriter
        .create<VselOp>(op.getLoc(), resultType, loaded, passthru, mask)
        .getResult();
  }

  LogicalResult lowerPhysicalParts(
      VMIMaskedLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart) const {
    bool arityMismatch = maskParts.size() != passthruParts.size() ||
                         passthruParts.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op,
                                         "masked_load physical arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "masked_load physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return materializeMaskedLoadPart(op, rewriter, source, offset,
                                           maskParts[index],
                                           passthruParts[index], resultType,
                                           index, lanesPerPart);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIMaskedLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "masked_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "masked_load offset must convert to one value",
        rewriter);
    bool failedOperands = failed(source) || failed(offset);
    if (failedOperands) {
      return failure();
    }

    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource(), op.getOffset(), rewriter);
    if (failed(lanesPerPart)) {
      return failure();
    }

    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerPhysicalParts(op, rewriter, *source, *offset, maskParts,
                              passthruParts, resultTypes, *lanesPerPart);
  }
};

struct OneToNVMIGatherOpPattern : OneToNOpConversionPattern<VMIGatherOp> {
  using OneToNOpConversionPattern<VMIGatherOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> materializeGatherPart(
      VMIGatherOp op, OneToNPatternRewriter &rewriter, Value source,
      Value indices, Value mask, Value passthru, Type resultType,
      bool allActive) const {
    bool invalidPartTypes = !isa<VRegType>(indices.getType()) ||
                            !isa<MaskType>(mask.getType()) ||
                            passthru.getType() != resultType ||
                            !isa<VRegType>(resultType);
    if (invalidPartTypes) {
      return rewriter.notifyMatchFailure(
          op, "gather physical part type mismatch");
    }
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        cast<VRegType>(resultType).getElementType());
    Value gathered = resultBits == 16
                         ? rewriter.create<Vgather2Op>(
                               op.getLoc(), resultType, source, indices, mask)
                               .getResult()
                         : rewriter.create<Vgather2BcOp>(
                               op.getLoc(), resultType, source, indices, mask)
                               .getResult();
    if (allActive) {
      return gathered;
    }
    return rewriter
        .create<VselOp>(op.getLoc(), resultType, gathered, passthru, mask)
        .getResult();
  }

  LogicalResult lowerPhysicalParts(
      VMIGatherOp op, OneToNPatternRewriter &rewriter, Value source,
      ValueRange indicesParts, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes, bool allActive) const {
    bool arityMismatch = indicesParts.size() != maskParts.size() ||
                         indicesParts.size() != passthruParts.size() ||
                         indicesParts.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op, "gather physical arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "gather physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return materializeGatherPart(op, rewriter, source,
                                       indicesParts[index], maskParts[index],
                                       passthruParts[index], resultType,
                                       allActive);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIGatherOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "gather source must convert to one value", rewriter);
    if (failed(source)) {
      return failure();
    }

    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    // Static all-active masks select gathered[0] for every lane, so the
    // trailing vsel is a semantic no-op. Skip it and keep gathered directly.
    // Non-static masks still take the original gather + vsel path.
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    bool allActive = isStaticAllActiveMask(op.getMask(),
                                           resultVMIType.getElementCount());


    return lowerPhysicalParts(op, rewriter, *source, indicesParts, maskParts,
                              passthruParts, resultTypes, allActive);
  }
};

struct OneToNVMIExpandLoadOpPattern
    : OneToNOpConversionPattern<VMIExpandLoadOp> {
  using OneToNOpConversionPattern<VMIExpandLoadOp>::OneToNOpConversionPattern;

private:
  struct RuntimeExpandLoadPlan {
    VRegType resultType;
    Value gatherBase;
    Value mask;
    Value passthru;
  };

  FailureOr<RuntimeExpandLoadPlan> buildRuntimeExpandLoadPlan(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes) const {
    bool invalidRuntimeArity = resultTypes.size() != 1 || maskParts.size() != 1 ||
                               passthruParts.size() != 1;
    if (invalidRuntimeArity) {
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load supports only one physical chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    bool invalidRuntimeTypes =
        !resultType || !maskType || passthruParts.front().getType() != resultType;
    if (invalidRuntimeTypes) {
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load requires physical result/passthru/mask");
    }
    if (!isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(op, "runtime expand_load requires ptr");
    }
    Value gatherBase = rewriter
                           .create<AddPtrOp>(op.getLoc(), source.getType(), source,
                                             offset)
                           .getResult();
    return RuntimeExpandLoadPlan{resultType, gatherBase, maskParts.front(),
                                 passthruParts.front()};
  }

  FailureOr<Value> materializeRuntimeExpandLoad(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter,
      const RuntimeExpandLoadPlan &plan) const {
    auto indexType = VRegType::get(rewriter.getContext(),
                                   plan.resultType.getElementCount(),
                                   rewriter.getI32Type());
    FailureOr<Value> indexSeedMask =
        createAllTrueMaskForVReg(op.getLoc(), indexType, rewriter);
    if (failed(indexSeedMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create runtime expand_load index seed mask");
    }
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    Value carrier = rewriter
                        .create<VdupOp>(op.getLoc(), indexType, zero,
                                       *indexSeedMask, /*position=*/nullptr)
                        .getResult();
    Value indices = rewriter
                        .create<VusqzOp>(op.getLoc(), indexType, carrier,
                                        plan.mask)
                        .getResult();
    Value gathered = rewriter
                         .create<Vgather2BcOp>(op.getLoc(), plan.resultType,
                                               plan.gatherBase, indices,
                                               plan.mask)
                         .getResult();
    return rewriter
        .create<VselOp>(op.getLoc(), plan.resultType, gathered, plan.passthru,
                        plan.mask)
        .getResult();
  }

  FailureOr<Value> materializeStaticExpandLoadPart(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Type resultType, int64_t index, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "expand_load result must be vreg");
    }
    Value chunkOffset = createChunkOffset(
        op.getLoc(), offset, index * lanesPerPart, rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, chunkOffset,
                        nullptr)
        .getResult();
  }

  LogicalResult lowerRuntimeExpandLoad(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes) const {
    FailureOr<RuntimeExpandLoadPlan> plan = buildRuntimeExpandLoadPlan(
        op, rewriter, source, offset, maskParts, passthruParts, resultTypes);
    if (failed(plan)) {
      return failure();
    }
    FailureOr<Value> result = materializeRuntimeExpandLoad(op, rewriter, *plan);
    if (failed(result)) {
      return failure();
    }
    return replaceSinglePhysicalResult(rewriter, op, *result,
                                       *this->getTypeConverter());
  }

  LogicalResult lowerExpandLoadParts(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    return resultTypes.size() == 1
               ? lowerRuntimeExpandLoad(op, rewriter, source, offset, maskParts,
                                       passthruParts, resultTypes)
               : lowerStaticExpandLoad(op, rewriter, source, offset,
                                       sourceVMIType, resultTypes);
  }

  LogicalResult lowerStaticExpandLoad(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, VMIVRegType resultVMIType, ArrayRef<Type> resultTypes) const {
    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource(), op.getOffset(), rewriter);
    if (failed(lanesPerPart)) {
      return failure();
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "expand_load physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return materializeStaticExpandLoadPart(op, rewriter, source, offset,
                                                 resultType, index,
                                                 *lanesPerPart);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIExpandLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "expand_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "expand_load offset must convert to one value",
        rewriter);
    bool operandsConverted = succeeded(source) && succeeded(offset);
    if (!operandsConverted) {
      return failure();
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (isStaticAllActiveMask(op.getMask(), resultVMIType.getElementCount())) {
      return lowerStaticExpandLoad(op, rewriter, *source, *offset,
                                   resultVMIType, resultTypes);
    }

    return lowerRuntimeExpandLoad(op, rewriter, *source, *offset,
                                  adaptor.getMask(), adaptor.getPassthru(),
                                  resultTypes);
  }
};

struct OneToNVMIStoreOpPattern : OneToNOpConversionPattern<VMIStoreOp> {
  using OneToNOpConversionPattern<VMIStoreOp>::OneToNOpConversionPattern;

private:
  struct StorePhysicalPlan {
    SmallVector<Type> contiguousTypes;
    VMILayoutAttr contiguousLayout;
    int64_t lanesPerPart;
    bool fullPhysicalChunks;
    bool noWiderThanContiguous;
  };

  struct StoreLoweringInput {
    Value destination;
    Value offset;
    ValueRange valueParts;
    VMIVRegType valueVMIType;
    StorePhysicalPlan plan;
  };

  FailureOr<StorePhysicalPlan> buildPhysicalPlan(
      VMIStoreOp op, ValueRange valueParts, VMIVRegType valueVMIType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "store requires known physical lanes per part");
    }
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> contiguousTypes =
        getContiguousStoreTypes(op, valueVMIType, rewriter);
    if (failed(contiguousTypes)) {
      return failure();
    }
    SmallVector<Type> valuePartTypes;
    valuePartTypes.reserve(valueParts.size());
    for (Value value : valueParts) {
      valuePartTypes.push_back(value.getType());
    }
    FailureOr<bool> noWiderThanContiguous =
        hasNoWiderFootprintThanContiguous(valuePartTypes, *contiguousTypes);
    if (failed(noWiderThanContiguous)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compare store physical footprint");
    }
    return StorePhysicalPlan{std::move(*contiguousTypes), contiguousLayout,
                             *lanesPerPart,
                             succeeded(checkFullDataPhysicalChunks(
                                 valueVMIType, nullptr)),
                             *noWiderThanContiguous};
  }

  FailureOr<StoreLoweringInput> getLoweringInput(
      VMIStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "store offset must convert to one value",
        rewriter);
    bool invalidAddressOperands = failed(destination) || failed(offset);
    if (invalidAddressOperands) {
      return failure();
    }
    ValueRange valueParts = adaptor.getValue();
    FailureOr<StorePhysicalPlan> plan =
        buildPhysicalPlan(op, valueParts, valueVMIType, rewriter);
    if (failed(plan)) {
      return failure();
    }
    return StoreLoweringInput{*destination, *offset, valueParts, valueVMIType,
                              std::move(*plan)};
  }

  FailureOr<SmallVector<Type>> getContiguousStoreTypes(
      VMIStoreOp op, VMIVRegType valueVMIType,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> contiguousTypes =
        getConvertedVRegTypesWithLayout(valueVMIType, contiguousLayout,
                                        *this->getTypeConverter());
    if (failed(contiguousTypes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute contiguous store footprint");
    }
    return std::move(*contiguousTypes);
  }

  FailureOr<bool> tryLowerLaneStrideStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      VMIVRegType valueVMIType, OneToNPatternRewriter &rewriter) const {
    std::optional<std::string> dist =
        getDenseLaneStrideStoreDistToken(valueVMIType);
    auto valueType = valueParts.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(valueParts.front().getType());
    bool canUseDist =
        dist && valueType && isDirectMemoryDistAddressLegal(
                                  op.getDestination(), op.getOffset(),
                                  valueVMIType.getElementType(), valueType,
                                  VPTOMemoryOpFamily::Store, *dist);
    if (!canUseDist) {
      return false;
    }
    std::optional<StringRef> maskGranularity =
        getDenseLaneStrideStoreMaskGranularity(valueVMIType);
    if (!maskGranularity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported lane_stride store mask granularity");
    }
    if (failed(emitLaneStrideStore(op, destination, offset, valueParts,
                                   valueVMIType, *dist, *maskGranularity,
                                   rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return true;
  }

  LogicalResult emitAlignedContiguousStoreParts(
      VMIStoreOp op, Value destination, Value offset, ValueRange storeParts,
      VMIVRegType valueVMIType, int64_t lanesPerPart, bool fullPhysicalChunks,
      OneToNPatternRewriter &rewriter) const {
    for (auto [index, value] : llvm::enumerate(storeParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      }
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> activeLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(activeLanes)) {
          return rewriter.notifyMatchFailure(
              op, "failed to compute store active lanes");
        }
        if (*activeLanes == 0) {
          continue;
        }
      }
      FailureOr<Value> mask =
          fullPhysicalChunks
              ? createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter)
              : createContiguousStoreMask(op.getLoc(), valueVMIType, index,
                                          vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for store mask");
      }
      Value chunkOffset = createChunkOffset(op.getLoc(), offset,
                                            index * lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, chunkOffset, /*dist=*/nullptr,
                              *mask);
    }
    return success();
  }

  FailureOr<SmallVector<Value>> collectUnalignedStoreValues(
      VMIStoreOp op, ValueRange storeParts, VMIVRegType valueVMIType,
      int64_t lanesPerPart, bool fullPhysicalChunks,
      SmallVectorImpl<int64_t> &advances,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> values;
    for (auto [index, value] : llvm::enumerate(storeParts)) {
      if (!isa<VRegType>(value.getType())) {
        (void)rewriter.notifyMatchFailure(op, "store value must be vreg");
        return failure();
      }
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> maybeActiveLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(maybeActiveLanes)) {
          return rewriter.notifyMatchFailure(
              op, "failed to compute unaligned store active lanes");
        }
        if (*maybeActiveLanes == 0) {
          continue;
        }
        values.push_back(value);
        advances.push_back(*maybeActiveLanes);
        continue;
      }
      values.push_back(value);
      advances.push_back(lanesPerPart);
    }
    return values;
  }

  FailureOr<Value> materializeUnalignedStoreBase(
      VMIStoreOp op, Value destination, Value offset,
      VMIVRegType valueVMIType, OneToNPatternRewriter &rewriter) const {
    Value storeBase = materializeBufferPointer(
        destination, valueVMIType.getElementType(),
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!storeBase) {
      return rewriter.notifyMatchFailure(
          op, "continuous unaligned store requires a ptr-compatible destination");
    }
    return rewriter
        .create<AddPtrOp>(op.getLoc(), storeBase.getType(), storeBase, offset)
        .getResult();
  }

  LogicalResult lowerContiguousStoreParts(
      VMIStoreOp op, Value destination, Value offset, ValueRange storeParts,
      VMIVRegType valueVMIType, int64_t lanesPerPart, bool fullPhysicalChunks,
      OneToNPatternRewriter &rewriter) const {
    auto firstStoreType =
        storeParts.empty() ? VRegType{}
                           : dyn_cast<VRegType>(storeParts.front().getType());
    bool useAlignedAccess =
        firstStoreType &&
        isDirectMemoryDistAddressLegal(
            op.getDestination(), op.getOffset(), valueVMIType.getElementType(),
            firstStoreType, VPTOMemoryOpFamily::Store, "");
    if (useAlignedAccess) {
      return emitAlignedContiguousStoreParts(
          op, destination, offset, storeParts, valueVMIType, lanesPerPart,
          fullPhysicalChunks, rewriter);
    }

    FailureOr<Value> storeBase = materializeUnalignedStoreBase(
        op, destination, offset, valueVMIType, rewriter);
    if (failed(storeBase)) {
      return failure();
    }
    SmallVector<int64_t> advances;
    FailureOr<SmallVector<Value>> values = collectUnalignedStoreValues(
        op, storeParts, valueVMIType, lanesPerPart, fullPhysicalChunks, advances,
        rewriter);
    if (failed(values)) {
      return failure();
    }
    if (failed(emitStatefulStoreStream(op, *storeBase, *values, advances,
                                       rewriter))) {
      return failure();
    }
    return success();
  }

  LogicalResult lowerByPhysicalPlan(
      VMIStoreOp op, const StoreLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<bool> laneStrideStore = tryLowerLaneStrideStore(
        op, input.destination, input.offset, input.valueParts,
        input.valueVMIType, rewriter);
    if (failed(laneStrideStore)) {
      return failure();
    }
    if (*laneStrideStore) {
      return success();
    }
    FailureOr<bool> deinterleavedStore = tryLowerDeinterleavedStore(
        op, input.destination, input.offset, input.valueParts,
        input.valueVMIType, input.plan.lanesPerPart,
        input.plan.fullPhysicalChunks, input.plan.noWiderThanContiguous,
        rewriter);
    if (failed(deinterleavedStore)) {
      return failure();
    }
    if (*deinterleavedStore) {
      return success();
    }
    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, input.valueParts, input.plan.contiguousTypes,
        input.valueVMIType.getLayoutAttr(), input.plan.contiguousLayout,
        input.valueVMIType.getElementType(), rewriter);
    if (failed(storeParts)) {
      return failure();
    }
    if (failed(lowerContiguousStoreParts(
            op, input.destination, input.offset, *storeParts,
            input.valueVMIType, input.plan.lanesPerPart,
            input.plan.fullPhysicalChunks, rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

public:

  static LogicalResult emitLaneStrideStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      VMIVRegType valueVMIType, StringRef laneStrideDist,
      StringRef maskGranularity, OneToNPatternRewriter &rewriter) {
    int64_t semanticOffset = 0;
    for (auto [index, value] : llvm::enumerate(valueParts)) {
      (void)index;
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      }
      FailureOr<int64_t> activeLanes =
          getActiveDataLanesInPhysicalChunk(valueVMIType, index);
      if (failed(activeLanes)) {
        return rewriter.notifyMatchFailure(
            op, "failed to compute lane_stride store active lanes");
      }
      if (*activeLanes == 0) {
        continue;
      }
      auto maskType = MaskType::get(rewriter.getContext(), maskGranularity);
      FailureOr<Value> mask = createPrefixMaskForActiveLanes(
          op.getLoc(), maskType, *activeLanes, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create lane_stride store mask");
      }
      Value chunkOffset =
          createChunkOffset(op.getLoc(), offset, semanticOffset, rewriter);
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, chunkOffset,
                              rewriter.getStringAttr(laneStrideDist), *mask);
      semanticOffset += *activeLanes;
    }
    return success();
  }

  static LogicalResult emitDeinterleavedStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      int64_t lanesPerPart, StringRef dist,
      OneToNPatternRewriter &rewriter) {
    bool oddValuePartCount = valueParts.size() % 2 != 0;
    if (oddValuePartCount) {
      return failure();
    }
    int64_t groups = valueParts.size() / 2;
    for (int64_t group = 0; group < groups; ++group) {
      Value low = valueParts[group];
      Value high = valueParts[groups + group];
      bool mismatchedTypes = low.getType() != high.getType();
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "vstsx2 requires matching low/high value types");
      }
      auto vregType = dyn_cast<VRegType>(low.getType());
      bool invalidValueType = !vregType;
      if (invalidValueType) {
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      bool maskFailed = failed(mask);
      if (maskFailed) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for store mask");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, group * 2 * lanesPerPart, rewriter);
      rewriter.create<Vstsx2Op>(op.getLoc(), low, high, destination, chunkOffset,
                                rewriter.getStringAttr(dist), *mask);
    }
    return success();
  }

  FailureOr<bool> tryLowerDeinterleavedStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      VMIVRegType valueVMIType, int64_t lanesPerPart,
      bool fullPhysicalChunks, bool noWiderThanContiguous,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutSupport supports;
    FailureOr<VMIStoreLayoutFact> storeFact =
        supports.getStoreLayoutFact(valueVMIType);
    bool candidate = succeeded(storeFact) &&
                     storeFact->valueLayout.isDeinterleaved() &&
                     storeFact->valueLayout.getFactor() == 2 &&
                     fullPhysicalChunks && noWiderThanContiguous;
    if (!candidate) {
      return false;
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
    auto firstType = valueParts.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(valueParts.front().getType());
    bool canUseDist = dist && firstType &&
                      isDirectMemoryDistAddressLegal(
                          op.getDestination(), op.getOffset(),
                          valueVMIType.getElementType(), firstType,
                          VPTOMemoryOpFamily::StoreX2, *dist);
    bool evenValuePartCount = valueParts.size() % 2 == 0;
    if (!canUseDist || !evenValuePartCount) {
      return false;
    }
    if (failed(emitDeinterleavedStore(
            op, destination, offset, valueParts, lanesPerPart, *dist,
            rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return true;
  }

  LogicalResult
  matchAndRewrite(VMIStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<StoreLoweringInput> input = getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    return lowerByPhysicalPlan(op, *input, rewriter);
  }
};

struct OneToNVMIInterleaveStoreOpPattern
    : OneToNOpConversionPattern<VMIInterleaveStoreOp> {
  using OneToNOpConversionPattern<
      VMIInterleaveStoreOp>::OneToNOpConversionPattern;

private:
  LogicalResult emitInterleaveStoreChunk(
      VMIInterleaveStoreOp op, Value low, Value high, size_t index,
      int64_t lanesPerPart, Value destination, Value offset, StringRef dist,
      bool useDirectAccess, SmallVectorImpl<Value> &streamValues,
      SmallVectorImpl<int64_t> &streamAdvances,
      OneToNPatternRewriter &rewriter) const {
    bool mismatchedTypes = low.getType() != high.getType();
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires matching low/high physical types");
    }
    auto vregType = dyn_cast<VRegType>(low.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store value must be vreg");
    }
    if (useDirectAccess) {
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for interleave_store mask");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, static_cast<int64_t>(index) * 2 * lanesPerPart,
          rewriter);
      rewriter.create<Vstsx2Op>(op.getLoc(), low, high, destination, chunkOffset,
                                rewriter.getStringAttr(dist), *mask);
      return success();
    }
    auto packets =
        rewriter.create<VintlvOp>(op.getLoc(), vregType, vregType, low, high);
    streamValues.push_back(packets.getLow());
    streamValues.push_back(packets.getHigh());
    streamAdvances.push_back(lanesPerPart);
    streamAdvances.push_back(lanesPerPart);
    return success();
  }

  FailureOr<bool> canUseDirectAccess(
      VMIInterleaveStoreOp op, ValueRange lowParts,
      VMIVRegType lowVMIType, StringRef dist) const {
    auto firstType = lowParts.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(lowParts.front().getType());
    if (!firstType) {
      return false;
    }
    return isDirectMemoryDistAddressLegal(
        op.getDestination(), op.getOffset(), lowVMIType.getElementType(),
        firstType, VPTOMemoryOpFamily::StoreX2, dist);
  }

  FailureOr<Value> getUnalignedBase(
      VMIInterleaveStoreOp op, Value destination, Value offset,
      VMIVRegType lowVMIType, OneToNPatternRewriter &rewriter) const {
    Value streamBase = materializeBufferPointer(
        destination, lowVMIType.getElementType(),
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!streamBase) {
      return rewriter.notifyMatchFailure(
          op, "unaligned interleave_store requires a ptr-compatible "
              "destination");
    }
    return rewriter
        .create<AddPtrOp>(op.getLoc(), streamBase.getType(), streamBase,
                          offset)
        .getResult();
  }

  struct InterleaveStoreAddressPlan {
    Value destination;
    Value offset;
    Value streamBase;
    bool useDirectAccess;
  };

  struct InterleaveStoreLoweringInput {
    Value destination;
    Value offset;
    ValueRange lowParts;
    ValueRange highParts;
    VMIVRegType lowVMIType;
    int64_t lanesPerPart;
    std::string dist;
  };

  FailureOr<InterleaveStoreLoweringInput> getLoweringInput(
      VMIInterleaveStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto lowVMIType = cast<VMIVRegType>(op.getLow().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(lowVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires known physical lanes per part");
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(lowVMIType.getElementType(), "INTLV");
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires vstsx2 INTLV element support");
    }
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "interleave_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "interleave_store offset must convert to one value", rewriter);
    bool invalidAddressOperands = failed(destination) || failed(offset);
    if (invalidAddressOperands) {
      return failure();
    }
    ValueRange lowParts = adaptor.getLow();
    ValueRange highParts = adaptor.getHigh();
    bool arityMismatch = lowParts.size() != highParts.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires matching low/high physical arity");
    }
    return InterleaveStoreLoweringInput{*destination, *offset, lowParts,
                                        highParts, lowVMIType, *lanesPerPart,
                                        *dist};
  }

  FailureOr<InterleaveStoreAddressPlan> buildAddressPlan(
      VMIInterleaveStoreOp op, Value destination, Value offset,
      ValueRange lowParts, VMIVRegType lowVMIType, StringRef dist,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<bool> directAccess =
        canUseDirectAccess(op, lowParts, lowVMIType, dist);
    if (failed(directAccess)) {
      return failure();
    }
    if (*directAccess) {
      return InterleaveStoreAddressPlan{destination, offset, Value{}, true};
    }
    FailureOr<Value> base =
        getUnalignedBase(op, destination, offset, lowVMIType, rewriter);
    if (failed(base)) {
      return failure();
    }
    return InterleaveStoreAddressPlan{destination, offset, *base, false};
  }

  LogicalResult lowerByAddressPlan(
      VMIInterleaveStoreOp op, const InterleaveStoreLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<InterleaveStoreAddressPlan> addressPlan = buildAddressPlan(
        op, input.destination, input.offset, input.lowParts, input.lowVMIType,
        input.dist, rewriter);
    if (failed(addressPlan)) {
      return failure();
    }
    SmallVector<Value> streamValues;
    SmallVector<int64_t> streamAdvances;
    for (size_t index = 0, e = input.lowParts.size(); index < e; ++index) {
      if (failed(emitInterleaveStoreChunk(
              op, input.lowParts[index], input.highParts[index], index,
              input.lanesPerPart, input.destination, input.offset, input.dist,
              addressPlan->useDirectAccess, streamValues, streamAdvances,
              rewriter))) {
        return failure();
      }
    }
    if (!addressPlan->useDirectAccess &&
        failed(emitStatefulStoreStream(op, addressPlan->streamBase,
                                       streamValues, streamAdvances, rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIInterleaveStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<InterleaveStoreLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    return lowerByAddressPlan(op, *input, rewriter);
  }
};


