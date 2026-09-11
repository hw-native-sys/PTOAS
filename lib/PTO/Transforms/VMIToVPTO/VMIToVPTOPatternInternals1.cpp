// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals1.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

struct OneToNVMILoadOpPattern : OneToNOpConversionPattern<VMILoadOp> {
  using OneToNOpConversionPattern<VMILoadOp>::OneToNOpConversionPattern;

private:
  struct LoadPhysicalPlan {
    Value source;
    Value offset;
    SmallVector<Type> resultTypes;
    SmallVector<Type> contiguousTypes;
    VMILayoutAttr resultLayout;
    int64_t lanesPerPart;
    bool noWiderThanContiguous;
  };

  FailureOr<SmallVector<Type>> getLoadResultTypes(
      VMILoadOp op) const {
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    return std::move(*resultTypes);
  }

  FailureOr<SmallVector<Type>> getContiguousLoadTypes(
      VMILoadOp op, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> contiguousTypes =
        getConvertedVRegTypesWithLayout(resultVMIType, contiguousLayout,
                                        *this->getTypeConverter());
    if (failed(contiguousTypes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute contiguous load footprint");
    }
    return std::move(*contiguousTypes);
  }

  FailureOr<LoadPhysicalPlan> buildPhysicalPlan(
      VMILoadOp op, Value source, Value offset,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> resultTypes =
        getLoadResultTypes(op);
    if (failed(resultTypes)) {
      return failure();
    }
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource(), op.getOffset(), rewriter);
    if (failed(lanesPerPart)) {
      return failure();
    }
    FailureOr<SmallVector<Type>> contiguousTypes =
        getContiguousLoadTypes(op, resultVMIType, rewriter);
    if (failed(contiguousTypes)) {
      return failure();
    }
    FailureOr<bool> noWiderThanContiguous =
        hasNoWiderFootprintThanContiguous(*resultTypes, *contiguousTypes);
    if (failed(noWiderThanContiguous)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compare load physical footprint");
    }
    return LoadPhysicalPlan{source,
                           offset,
                           std::move(*resultTypes),
                           std::move(*contiguousTypes),
                           resultVMIType.getLayoutAttr(),
                           *lanesPerPart,
                           *noWiderThanContiguous};
  }

  FailureOr<SmallVector<Value>> materializeLaneStrideParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      StringRef dist) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    int64_t semanticOffset = 0;
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "load result must be vreg");
      }
      Value chunkOffset =
          createChunkOffset(op.getLoc(), offset, semanticOffset, rewriter);
      results.push_back(rewriter
                            .create<VldsOp>(op.getLoc(), resultType,
                                            /*updated_base=*/Type{}, source,
                                            chunkOffset,
                                            rewriter.getStringAttr(dist))
                            .getResult());
      FailureOr<int64_t> activeLanes =
          getActiveDataLanesInPhysicalChunk(resultVMIType, index);
      if (failed(activeLanes)) {
        return rewriter.notifyMatchFailure(
            op, "failed to compute lane_stride load active lanes");
      }
      semanticOffset += *activeLanes;
    }
    return results;
  }

  LogicalResult lowerLaneStride(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      StringRef dist) const {
    FailureOr<SmallVector<Value>> results = materializeLaneStrideParts(
        op, rewriter, source, offset, resultVMIType, resultTypes, dist);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerDeinterleaved2(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart, StringRef dist) const {
    bool invalidFactor2Arity = resultTypes.size() % 2 != 0;
    if (invalidFactor2Arity) {
      return rewriter.notifyMatchFailure(
          op, "vldsx2 deinterleaved=2 load requires even physical arity");
    }
    int64_t groups = resultTypes.size() / 2;
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(groups);
    highs.reserve(groups);
    for (int64_t group = 0; group < groups; ++group) {
      Type lowType = resultTypes[group];
      Type highType = resultTypes[groups + group];
      if (lowType != highType) {
        return rewriter.notifyMatchFailure(
            op, "vldsx2 requires matching low/high result types");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, group * 2 * lanesPerPart, rewriter);
      auto load = rewriter.create<Vldsx2Op>(
          op.getLoc(), lowType, highType, Type{}, source, chunkOffset,
          rewriter.getStringAttr(dist));
      lows.push_back(load.getLow());
      highs.push_back(load.getHigh());
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<std::array<Value, 4>> materializeDeinterleaved4LoadGroup(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ArrayRef<Type> resultTypes, int64_t groups,
      int64_t group, int64_t lanesPerPart, StringRef dist) const {
    Type types[4] = {resultTypes[group], resultTypes[groups + group],
                     resultTypes[2 * groups + group],
                     resultTypes[3 * groups + group]};
    bool mismatchedTypes = types[0] != types[1] || types[0] != types[2] ||
                           types[0] != types[3];
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "vldsx2 deinterleaved=4 load requires matching part types");
    }
    Value firstOffset = createChunkOffset(
        op.getLoc(), offset, group * 4 * lanesPerPart, rewriter);
    Value secondOffset = createChunkOffset(
        op.getLoc(), offset, (group * 4 + 2) * lanesPerPart, rewriter);
    auto first = rewriter.create<Vldsx2Op>(
        op.getLoc(), types[0], types[1], Type{}, source, firstOffset,
        rewriter.getStringAttr(dist));
    auto second = rewriter.create<Vldsx2Op>(
        op.getLoc(), types[2], types[3], Type{}, source, secondOffset,
        rewriter.getStringAttr(dist));
    auto even = rewriter.create<VdintlvOp>(
        op.getLoc(), types[0], types[2], first.getLow(), second.getLow());
    auto odd = rewriter.create<VdintlvOp>(
        op.getLoc(), types[1], types[3], first.getHigh(), second.getHigh());
    return std::array<Value, 4>{even.getLow(), odd.getLow(), even.getHigh(),
                                odd.getHigh()};
  }

  LogicalResult lowerDeinterleaved4(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart, StringRef dist) const {
    bool invalidFactor4Arity = resultTypes.size() % 4 != 0;
    if (invalidFactor4Arity) {
      return rewriter.notifyMatchFailure(
          op, "vldsx2 deinterleaved=4 load requires physical arity divisible by 4");
    }
    int64_t groups = resultTypes.size() / 4;
    SmallVector<Value> parts[4];
    for (auto &part : parts) {
      part.reserve(groups);
    }
    for (int64_t group = 0; group < groups; ++group) {
      FailureOr<std::array<Value, 4>> groupValues =
          materializeDeinterleaved4LoadGroup(
              op, rewriter, source, offset, resultTypes, groups, group,
              lanesPerPart, dist);
      if (failed(groupValues)) {
        return failure();
      }
      for (size_t part = 0; part < 4; ++part) {
        parts[part].push_back((*groupValues)[part]);
      }
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto &part : parts) {
      results.append(part);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<SmallVector<Value>> materializeAlignedContiguousParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ArrayRef<Type> contiguousTypes, int64_t lanesPerPart) const {
    SmallVector<Value> parts;
    parts.reserve(contiguousTypes.size());
    for (auto [index, resultType] : llvm::enumerate(contiguousTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "load result must be vreg");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, index * lanesPerPart, rewriter);
      parts.push_back(rewriter
                          .create<VldsOp>(op.getLoc(), resultType,
                                          /*updated_base=*/Type{}, source,
                                          chunkOffset, /*dist=*/nullptr)
                          .getResult());
    }
    return parts;
  }

  struct UnalignedLoadPart {
    Value result;
    Value base;
    Value align;
  };

  FailureOr<SmallVector<Value>> materializeUnalignedContiguousParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      ArrayRef<Type> contiguousTypes, int64_t lanesPerPart) const {
    Value unalignedBase = materializeBufferPointer(
        source, getMemoryElementType(source.getType()),
        getMemorySpace(source.getType()), rewriter, op.getLoc());
    if (!unalignedBase) {
      return rewriter.notifyMatchFailure(
          op, "continuous unaligned load requires a ptr-compatible source");
    }
    unalignedBase = rewriter
                        .create<AddPtrOp>(op.getLoc(), unalignedBase.getType(),
                                          unalignedBase, offset)
                        .getResult();
    Value unalignedAlign = rewriter
                               .create<VldasOp>(
                                   op.getLoc(),
                                   AlignType::get(rewriter.getContext()),
                                   unalignedBase)
                               .getResult();
    SmallVector<Value> parts;
    parts.reserve(contiguousTypes.size());
    for (Type resultType : contiguousTypes) {
      FailureOr<UnalignedLoadPart> updatedState = emitUnalignedLoadPart(
          op, rewriter, unalignedBase, unalignedAlign, resultType,
          lanesPerPart);
      if (failed(updatedState)) {
        return failure();
      }
      parts.push_back(updatedState->result);
      unalignedBase = updatedState->base;
      unalignedAlign = updatedState->align;
    }
    return parts;
  }

  FailureOr<UnalignedLoadPart> emitUnalignedLoadPart(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value base, Value align,
      Type resultType, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "load result must be vreg");
    }
    Value increment =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lanesPerPart);
    auto load = rewriter.create<VldusOp>(
        op.getLoc(), resultType, align.getType(), base.getType(), base, align,
        increment);
    return UnalignedLoadPart{load.getResult(), load.getUpdatedBase(),
                             load.getUpdatedAlign()};
  }

  FailureOr<SmallVector<Value>> materializeContiguousLoadParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, ArrayRef<Type> contiguousTypes,
      int64_t lanesPerPart) const {
    auto firstType = contiguousTypes.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(contiguousTypes.front());
    bool useAlignedAccess =
        firstType && isDirectMemoryDistAddressLegal(
                          op.getSource(), op.getOffset(),
                          resultVMIType.getElementType(), firstType,
                          VPTOMemoryOpFamily::Load, "NORM");
    if (useAlignedAccess) {
      return materializeAlignedContiguousParts(
          op, rewriter, source, offset, contiguousTypes, lanesPerPart);
    }
    return materializeUnalignedContiguousParts(
        op, rewriter, source, offset, contiguousTypes, lanesPerPart);
  }

  LogicalResult lowerContiguous(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      ArrayRef<Type> contiguousTypes, int64_t lanesPerPart,
      VMILayoutAttr contiguousLayout) const {
    FailureOr<SmallVector<Value>> contiguousParts =
        materializeContiguousLoadParts(op, rewriter, source, offset,
                                       resultVMIType, contiguousTypes,
                                       lanesPerPart);
    if (failed(contiguousParts)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, *contiguousParts, resultTypes, contiguousLayout,
        resultVMIType.getLayoutAttr(), resultVMIType.getElementType(),
        rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  std::optional<LogicalResult> lowerDirectDeinterleaved(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, VMILayoutAttr resultLayout,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart,
      bool noWiderThanContiguous) const {
    bool unsupportedLayout = !resultLayout || !resultLayout.isDeinterleaved();
    if (unsupportedLayout || !noWiderThanContiguous) {
      return std::nullopt;
    }
    int64_t factor = resultLayout.getFactor();
    bool supportedFactor = factor == 2 || factor == 4;
    if (!supportedFactor) {
      return std::nullopt;
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
    auto firstType =
        resultTypes.empty() ? VRegType{} : dyn_cast<VRegType>(resultTypes.front());
    bool canUseDist =
        dist && firstType &&
        isDirectMemoryDistAddressLegal(
            op.getSource(), op.getOffset(), resultVMIType.getElementType(),
            firstType, VPTOMemoryOpFamily::LoadX2, *dist);
    bool validArity = resultTypes.size() % static_cast<size_t>(factor) == 0;
    if (!canUseDist || !validArity) {
      return std::nullopt;
    }
    if (factor == 2) {
      return lowerDeinterleaved2(op, rewriter, source, offset, resultTypes,
                                 lanesPerPart, *dist);
    }
    return lowerDeinterleaved4(op, rewriter, source, offset, resultTypes,
                               lanesPerPart, *dist);
  }

  std::optional<std::string> getLoadLaneStrideDist(
      VMILoadOp op, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes) const {
    std::optional<std::string> dist =
        getDenseLaneStrideLoadDistToken(resultVMIType);
    auto resultType =
        resultTypes.empty() ? VRegType{} : dyn_cast<VRegType>(resultTypes.front());
    if (!dist || !resultType ||
        !isDirectMemoryDistAddressLegal(
            op.getSource(), op.getOffset(), resultVMIType.getElementType(),
            resultType, VPTOMemoryOpFamily::Load, *dist)) {
      return std::nullopt;
    }
    return dist;
  }

public:

  LogicalResult
  matchAndRewrite(VMILoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "load offset must convert to one value",
        rewriter);
    bool failedOperands = failed(source) || failed(offset);
    if (failedOperands) {
      return failure();
    }
    FailureOr<SmallVector<Type>> maybe_resultTypes = getLoadResultTypes(op);
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    // Contiguous lane_stride loads read packed data straight from memory with
    // a dist-bearing vlds; resolve them before the full-chunk verification so
    // the strided footprint is not mistaken for an incomplete chunk.
    std::optional<std::string> laneStrideDist =
        getLoadLaneStrideDist(op, resultVMIType, resultTypes);
    if (laneStrideDist) {
      return lowerLaneStride(op, rewriter, *source, *offset, resultVMIType,
                             resultTypes, *laneStrideDist);
    }

    FailureOr<LoadPhysicalPlan> plan =
        buildPhysicalPlan(op, *source, *offset, rewriter);
    if (failed(plan)) {
      return failure();
    }

    std::optional<LogicalResult> deinterleavedResult =
        lowerDirectDeinterleaved(op, rewriter, plan->source, plan->offset,
                                 resultVMIType, plan->resultLayout,
                                 plan->resultTypes, plan->lanesPerPart,
                                 plan->noWiderThanContiguous);
    if (deinterleavedResult) {
      return *deinterleavedResult;
    }

    return lowerContiguous(op, rewriter, plan->source, plan->offset,
                            resultVMIType, plan->resultTypes,
                            plan->contiguousTypes, plan->lanesPerPart,
                            VMILayoutAttr::getContiguous(
                                rewriter.getContext()));
  }
};

struct OneToNVMIDeinterleaveLoadOpPattern
    : OneToNOpConversionPattern<VMIDeinterleaveLoadOp> {
  using OneToNOpConversionPattern<
      VMIDeinterleaveLoadOp>::OneToNOpConversionPattern;

private:
  struct UnalignedDeinterleaveLoadPair {
    Value low;
    Value high;
    Value updatedBase;
    Value updatedAlign;
  };

  struct DeinterleaveLoadLoweringInput {
    Value source;
    Value offset;
    SmallVector<Type> lowTypes;
    SmallVector<Type> highTypes;
    VMIVRegType lowVMIType;
    int64_t lanesPerPart;
    std::string dist;
  };

  FailureOr<UnalignedDeinterleaveLoadPair> materializeUnalignedLoadPair(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter,
      Value streamBase, Value streamAlign, Type lowType, Type highType,
      Value increment) const {
    if (lowType != highType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires matching low/high physical types");
    }
    auto first = rewriter.create<VldusOp>(
        op.getLoc(), lowType, streamAlign.getType(), streamBase.getType(),
        streamBase, streamAlign, increment);
    auto second = rewriter.create<VldusOp>(
        op.getLoc(), highType, first.getUpdatedAlign().getType(),
        first.getUpdatedBase().getType(), first.getUpdatedBase(),
        first.getUpdatedAlign(), increment);
    auto deinterleaved = rewriter.create<VdintlvOp>(
        op.getLoc(), lowType, highType, first.getResult(), second.getResult());
    return UnalignedDeinterleaveLoadPair{deinterleaved.getLow(),
                                         deinterleaved.getHigh(),
                                         second.getUpdatedBase(),
                                         second.getUpdatedAlign()};
  }

  LogicalResult lowerDirect(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes, int64_t lanesPerPart, StringRef dist) const {
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(lowTypes.size());
    highs.reserve(highTypes.size());
    for (size_t index = 0; index < lowTypes.size(); ++index) {
      if (lowTypes[index] != highTypes[index]) {
        return rewriter.notifyMatchFailure(
            op, "deinterleave_load requires matching low/high physical types");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, static_cast<int64_t>(index) * 2 * lanesPerPart,
          rewriter);
      auto load = rewriter.create<Vldsx2Op>(
          op.getLoc(), lowTypes[index], highTypes[index], Type{}, source,
          chunkOffset, rewriter.getStringAttr(dist));
      lows.push_back(load.getLow());
      highs.push_back(load.getHigh());
    }
    SmallVector<Value> results;
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerUnaligned(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes, int64_t lanesPerPart) const {
    Value streamBase = materializeBufferPointer(
        source, getMemoryElementType(source.getType()),
        getMemorySpace(source.getType()), rewriter, op.getLoc());
    if (!streamBase) {
      return rewriter.notifyMatchFailure(
          op, "unaligned deinterleave_load requires a ptr-compatible source");
    }
    streamBase = rewriter
                     .create<AddPtrOp>(op.getLoc(), streamBase.getType(),
                                       streamBase, offset)
                     .getResult();
    Value streamAlign = rewriter
                            .create<VldasOp>(
                                op.getLoc(), AlignType::get(rewriter.getContext()),
                                streamBase)
                            .getResult();
    Value increment =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lanesPerPart);
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(lowTypes.size());
    highs.reserve(highTypes.size());
    for (size_t index = 0; index < lowTypes.size(); ++index) {
      FailureOr<UnalignedDeinterleaveLoadPair> pair =
          materializeUnalignedLoadPair(op, rewriter, streamBase, streamAlign,
                                       lowTypes[index], highTypes[index],
                                       increment);
      if (failed(pair)) {
        return failure();
      }
      lows.push_back(pair->low);
      highs.push_back(pair->high);
      streamBase = pair->updatedBase;
      streamAlign = pair->updatedAlign;
    }
    SmallVector<Value> results;
    results.reserve(lows.size() + highs.size());
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>>
  getDeinterleaveLoadResultTypes(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> lowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> highTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    bool failedTypeConversion = failed(lowTypes) || failed(highTypes);
    if (failedTypeConversion) {
      return failure();
    }
    bool mismatchedArity = lowTypes->size() != highTypes->size();
    if (mismatchedArity) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires matching low/high physical arity");
    }
    return std::make_pair(std::move(*lowTypes), std::move(*highTypes));
  }

  FailureOr<DeinterleaveLoadLoweringInput> getLoweringInput(
      VMIDeinterleaveLoadOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto lowVMIType = cast<VMIVRegType>(op.getLow().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "deinterleave_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "deinterleave_load offset must convert to one value", rewriter);
    bool invalidOperands = failed(source) || failed(offset);
    if (invalidOperands) {
      return failure();
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(lowVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires known physical lanes per part");
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(lowVMIType.getElementType(), "DINTLV");
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires vldsx2 DINTLV element support");
    }
    FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>> resultTypes =
        getDeinterleaveLoadResultTypes(op, rewriter);
    if (failed(resultTypes)) {
      return failure();
    }
    return DeinterleaveLoadLoweringInput{
        *source, *offset, std::move(resultTypes->first),
        std::move(resultTypes->second), lowVMIType, *lanesPerPart, *dist};
  }

  LogicalResult lowerByAddressPlan(
      VMIDeinterleaveLoadOp op, const DeinterleaveLoadLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    auto firstType = input.lowTypes.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(input.lowTypes.front());
    bool useDirectAccess =
        firstType && isDirectMemoryDistAddressLegal(
                         op.getSource(), op.getOffset(),
                         input.lowVMIType.getElementType(), firstType,
                         VPTOMemoryOpFamily::LoadX2, input.dist);
    if (!useDirectAccess) {
      return lowerUnaligned(op, rewriter, input.source, input.offset,
                            input.lowTypes, input.highTypes,
                            input.lanesPerPart);
    }
    return lowerDirect(op, rewriter, input.source, input.offset, input.lowTypes,
                       input.highTypes, input.lanesPerPart, input.dist);
  }


public:

  LogicalResult
  matchAndRewrite(VMIDeinterleaveLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<DeinterleaveLoadLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    return lowerByAddressPlan(op, *input, rewriter);
  }
};

struct OneToNVMIGroupLoadOpPattern : OneToNOpConversionPattern<VMIGroupLoadOp> {
  using OneToNOpConversionPattern<VMIGroupLoadOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Type>> getResultTypes(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    return std::move(*resultTypes);
  }

  LogicalResult lowerContiguousPath(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout) const {
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    }
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    FailureOr<SmallVector<Type>> resultTypes = getResultTypes(op, rewriter);
    if (failed(resultTypes)) {
      return failure();
    }
    bool unitStride = constantRowStride && *constantRowStride == *groupSize;
    if (unitStride && resultLayout && resultLayout.isContiguous()) {
      return lowerContiguousUnitStride(op, rewriter, source, offset,
                                       resultVMIType, *resultTypes);
    }
    return lowerContiguousChunks(op, rewriter, source, offset, rowStride,
                                 resultVMIType, *resultTypes);
  }

  LogicalResult lowerContiguousUnitStride(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, VMIVRegType resultVMIType, ArrayRef<Type> resultTypes) const {
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "contiguous group_load requires known physical lanes");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> result = materializeContiguousUnitStrideChunk(
          op, rewriter, source, offset, resultType, index, *lanesPerPart);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<Value> materializeContiguousUnitStrideChunk(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Type resultType, size_t index, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(
          op, "contiguous group_load result must be vreg");
    }
    Value chunkOffset = createChunkOffset(
        op.getLoc(), offset, static_cast<int64_t>(index) * lanesPerPart,
        rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, chunkOffset,
                        nullptr)
        .getResult();
  }

  FailureOr<Value> materializeBlockDeinterleavedChunk(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, Type resultType, int64_t part,
      int64_t chunk, int64_t blockElems, int64_t constantRowStride) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "block_deinterleaved group_load result must be vreg");
    }
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(allMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create block_deinterleaved group_load mask");
    }
    Value blockStride = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), constantRowStride / 8, 16);
    Value zeroI16 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 16);
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, chunk * 8, part * blockElems,
        rewriter);
    Value chunkBase = rewriter
                          .create<AddPtrOp>(op.getLoc(), source.getType(), source,
                                            chunkOffset)
                          .getResult();
    return rewriter
        .create<VsldbOp>(op.getLoc(), vregType, Type{}, chunkBase, blockStride,
                         zeroI16, *allMask)
        .getResult();
  }

  LogicalResult lowerBlockDeinterleaved(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes, VMILayoutAttr resultLayout,
      int64_t factor, int64_t blockElems, int64_t chunksPerPart,
      int64_t constantRowStride) const {
    bool invalidResultArity =
        static_cast<int64_t>(resultTypes.size()) != factor * chunksPerPart;
    if (invalidResultArity) {
      return rewriter.notifyMatchFailure(
          op, "block_deinterleaved group_load arity mismatch");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        int64_t flatIndex = part * chunksPerPart + chunk;
        FailureOr<Value> result = materializeBlockDeinterleavedChunk(
            op, rewriter, source, offset, rowStride, resultTypes[flatIndex],
            part, chunk, blockElems, constantRowStride);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<Value> materializeContiguousGroupLoadChunk(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, Type resultType, int64_t group,
      int64_t chunkInGroup, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "group_load result must be vreg");
    }
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, chunkInGroup * lanesPerPart,
        rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, chunkOffset,
                        nullptr)
        .getResult();
  }

  LogicalResult lowerContiguousChunks(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    }
    if (failed(checkContiguousFullGroupChunks(
            op, resultVMIType, *groupSize, &lanesPerPart, &groupCount,
            &chunksPerGroup, rewriter))) {
      return failure();
    }
    bool invalidArity = static_cast<int64_t>(resultTypes.size()) !=
                        groupCount * chunksPerGroup;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "group_load arity mismatch");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      int64_t group = index / chunksPerGroup;
      int64_t chunkInGroup = index % chunksPerGroup;
      FailureOr<Value> result = materializeContiguousGroupLoadChunk(
          op, rewriter, source, offset, rowStride, resultType, group,
          chunkInGroup, lanesPerPart);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<std::pair<int64_t, int64_t>> validateBlockF32Shape(
      VMIGroupLoadOp op, Value source, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    }
    bool validFactorShape =
        (*groupSize == 16 && resultLayout.getFactor() == 2) ||
        (*groupSize == 32 && resultLayout.getFactor() == 4);
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    bool validRowStride = constantRowStride && *constantRowStride > 0 &&
                          *constantRowStride % 8 == 0;
    bool validGroupCount = op.getNumGroupsAttr().getInt() % 8 == 0;
    if (!validFactorShape || !validGroupCount || !validRowStride ||
        !isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(
          op, !validFactorShape
                  ? "block_deinterleaved group_load requires S=16/factor=2 or S=32/factor=4"
                  : !validGroupCount
                        ? "block_deinterleaved group_load requires num_groups multiple of 8"
                        : !validRowStride
                              ? "block_deinterleaved group_load requires constant positive "
                                "row_stride divisible by 8 f32 elements"
                              : "block_deinterleaved group_load requires !pto.ptr source");
    }
    return std::make_pair(*constantRowStride, resultLayout.getFactor());
  }

  LogicalResult lowerBlockF32(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout) const {
    FailureOr<std::pair<int64_t, int64_t>> shape =
        validateBlockF32Shape(op, source, resultVMIType, resultLayout,
                              rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t constantRowStride = shape->first;
    int64_t factor = shape->second;
    FailureOr<SmallVector<Type>> maybeResultTypes = getResultTypes(op, rewriter);
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    FailureOr<int64_t> blockElems = getVMILayoutBlockElems(resultVMIType);
    FailureOr<int64_t> chunksPerPart =
        getDataChunksInPart(resultVMIType, 0);
    bool invalidChunks = failed(blockElems) || failed(chunksPerPart) ||
                         *chunksPerPart <= 0;
    if (invalidChunks) {
      return rewriter.notifyMatchFailure(
          op, "block_deinterleaved group_load requires known block and "
              "chunks per part");
    }
    for (int64_t part = 1; part < factor; ++part) {
      FailureOr<int64_t> currentChunks =
          getDataChunksInPart(resultVMIType, part);
      bool nonUniformChunks =
          failed(currentChunks) || *currentChunks != *chunksPerPart;
      if (nonUniformChunks) {
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires uniform chunks per "
                "part");
      }
    }
    return lowerBlockDeinterleaved(
        op, rewriter, source, offset, rowStride, resultVMIType, resultTypes,
        resultLayout, factor, *blockElems, *chunksPerPart, constantRowStride);
  }

  LogicalResult lowerByLayout(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout) const {
    bool isBlockF32 = resultLayout && resultLayout.isBlockDeinterleaved() &&
                      resultVMIType.getElementType().isF32();
    if (isBlockF32) {
      return lowerBlockF32(op, rewriter, source, offset, rowStride,
                           resultVMIType, resultLayout);
    }
    return lowerContiguousPath(op, rewriter, source, offset, rowStride,
                               resultVMIType, resultLayout);
  }

public:

  LogicalResult
  matchAndRewrite(VMIGroupLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "group_load source must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "group_load offset must convert to one value", rewriter);
    FailureOr<Value> rowStride = getSingleValue(
        op, adaptor.getRowStride(),
        "group_load row_stride must convert to one value", rewriter);
    bool invalidOperands =
        failed(source) || failed(offset) || failed(rowStride);
    if (invalidOperands) {
      return failure();
    }

    return lowerByLayout(op, rewriter, *source, *offset, *rowStride,
                         resultVMIType, resultVMIType.getLayoutAttr());
  }
};

struct GroupSlotLoadResultPart {
  VRegType valueType;
  MaskType maskType;
};

static FailureOr<GroupSlotLoadResultPart> getGroupSlotLoadResultPart(
    Operation *op, Type resultType, OneToNPatternRewriter &rewriter) {
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    (void)rewriter.notifyMatchFailure(op,
                                      "group_slot_load result must be vreg");
    return failure();
  }
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  if (failed(maskType)) {
    (void)rewriter.notifyMatchFailure(
        op, "unsupported element type for group_slot_load mask");
    return failure();
  }
  return GroupSlotLoadResultPart{vregType, *maskType};
}

static LogicalResult lowerSingleGroupSlotLoad(
    Operation *op, Value source, Value offset, VMIVRegType resultVMIType,
    TypeRange resultTypes, OneToNPatternRewriter &rewriter,
    SmallVectorImpl<Value> &results) {
  std::optional<std::string> dist =
      getScalarBroadcastLoadDistToken(resultVMIType.getElementType());
  if (!dist) {
    return rewriter.notifyMatchFailure(
        op, "single-slot group_slot_load requires supported BRC load element width");
  }
  auto vregType = dyn_cast<VRegType>(resultTypes.front());
  if (!vregType) {
    return rewriter.notifyMatchFailure(
        op, "single-slot group_slot_load result must be vreg");
  }
  results.push_back(rewriter
                       .create<VldsOp>(op->getLoc(), vregType,
                                       /*updated_base=*/Type{}, source, offset,
                                       rewriter.getStringAttr(*dist))
                       .getResult());
  return success();
}

static LogicalResult emitGroupSlotLoadSlots8Chunk(
    Operation *op, Value source, Value offset, Type resultType, int64_t chunk,
    int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<GroupSlotLoadResultPart> resultPart =
      getGroupSlotLoadResultPart(op, resultType, rewriter);
  if (failed(resultPart)) {
    return failure();
  }
  int64_t groupBegin = chunk * 8;
  int64_t activeGroups = std::min<int64_t>(8, numGroups - groupBegin);
  if (activeGroups <= 0) {
    return rewriter.notifyMatchFailure(
        op, "slots=8 group_slot_load has no active groups for chunk");
  }
  std::string pattern = (Twine("PAT_VL") + Twine(activeGroups)).str();
  FailureOr<Value> slotMask = createPrefixMask(
      op->getLoc(), resultPart->maskType, pattern, rewriter);
  if (failed(slotMask)) {
    return rewriter.notifyMatchFailure(
        op, "failed to create slots=8 group_slot_load mask");
  }
  Value groupOffset =
      createChunkOffset(op->getLoc(), offset, groupBegin, rewriter);
  Value slotBase = rewriter
                       .create<AddPtrOp>(op->getLoc(), source.getType(), source,
                                         groupOffset)
                       .getResult();
  auto zeroI16 = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 16);
  results.push_back(
      rewriter
          .create<VsldbOp>(op->getLoc(), resultPart->valueType,
                          /*updated_base=*/Type{}, slotBase, zeroI16, zeroI16,
                          *slotMask)
          .getResult());
  return success();
}

static LogicalResult lowerGroupSlotLoadSlots8(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  std::optional<int64_t> stride = getConstantIndexValue(sourceGroupStride);
  if (!stride || *stride != 1) {
    return rewriter.notifyMatchFailure(
        op, "slots=8 group_slot_load requires constant unit stride");
  }
  if (numGroups == 1) {
    return lowerSingleGroupSlotLoad(op, source, offset, resultVMIType,
                                    resultTypes, rewriter, results);
  }
  for (auto [chunk, resultType] : llvm::enumerate(resultTypes)) {
    if (failed(emitGroupSlotLoadSlots8Chunk(
            op, source, offset, resultType,
            static_cast<int64_t>(chunk), numGroups, rewriter, results))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult emitGroupSlotLoadSlots1Chunk(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    Type resultType, int64_t group, OneToNPatternRewriter &rewriter,
    SmallVectorImpl<Value> &results) {
  FailureOr<GroupSlotLoadResultPart> resultPart =
      getGroupSlotLoadResultPart(op, resultType, rewriter);
  if (failed(resultPart)) {
    return failure();
  }
  FailureOr<Value> oneBlockMask = createPrefixMask(
      op->getLoc(), resultPart->maskType, "PAT_VL1", rewriter);
  if (failed(oneBlockMask)) {
    return rewriter.notifyMatchFailure(op, "failed to create group_slot_load mask");
  }
  Value groupOffset = offset;
  if (group != 0) {
    Value groupIndex =
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), group);
    Value rowOffset = rewriter
                          .create<arith::MulIOp>(op->getLoc(),
                                                 sourceGroupStride, groupIndex)
                          .getResult();
    groupOffset = rewriter
                      .create<arith::AddIOp>(op->getLoc(), groupOffset,
                                             rowOffset)
                      .getResult();
  }
  Value slotBase = rewriter
                       .create<AddPtrOp>(op->getLoc(), source.getType(), source,
                                         groupOffset)
                       .getResult();
  auto zeroI16 = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 16);
  results.push_back(
      rewriter
          .create<VsldbOp>(op->getLoc(), resultPart->valueType,
                          /*updated_base=*/Type{}, slotBase, zeroI16, zeroI16,
                          *oneBlockMask)
          .getResult());
  return success();
}

static LogicalResult lowerGroupSlotLoadSlots1(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
  if (elementBits == 0 || 256 % elementBits != 0) {
    return rewriter.notifyMatchFailure(
        op, "slots=1 group_slot_load requires supported element width");
  }
  int64_t alignedStrideElems = 256 / elementBits;
  std::optional<int64_t> constantStride =
      getConstantIndexValue(sourceGroupStride);
  if (!constantStride || *constantStride <= 0 ||
      *constantStride % alignedStrideElems != 0) {
    return rewriter.notifyMatchFailure(
        op, Twine("slots=1 group_slot_load requires constant positive "
                  "source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B lane-0 vsldb alignment");
  }
  for (auto [group, resultType] : llvm::enumerate(resultTypes)) {
    if (failed(emitGroupSlotLoadSlots1Chunk(
            op, source, offset, sourceGroupStride, resultType,
            static_cast<int64_t>(group), rewriter, results))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<int64_t> getGroupSlotLoadSlots(
    Operation *op, Value source, VMIVRegType resultVMIType,
    TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter) {
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  bool invalidLayout = !layout || !layout.isGroupSlots() || layout.getSlots() <= 0;
  if (invalidLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "group_slot_load requires explicit group_slots layout");
    return failure();
  }
  if (!isa<PtrType>(source.getType())) {
    (void)rewriter.notifyMatchFailure(
        op, "group_slot_load requires !pto.ptr source");
    return failure();
  }
  int64_t slots = layout.getSlots();
  int64_t expectedArity = ceilDivNonNegative(numGroups, slots);
  bool arityMismatch =
      static_cast<int64_t>(resultTypes.size()) != expectedArity;
  if (arityMismatch) {
    (void)rewriter.notifyMatchFailure(op, "group_slot_load arity mismatch");
    return failure();
  }
  if (slots != 8 && slots != 1) {
    (void)rewriter.notifyMatchFailure(
        op, "group_slot_load supports only slots=8 or slots=1");
    return failure();
  }
  return slots;
}

static LogicalResult lowerGroupSlotLoadParts(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<int64_t> maybeSlots = getGroupSlotLoadSlots(
      op, source, resultVMIType, resultTypes, numGroups, rewriter);
  if (failed(maybeSlots)) {
    return failure();
  }
  int64_t slots = *maybeSlots;
  results.reserve(results.size() + resultTypes.size());
  if (slots == 8) {
    return lowerGroupSlotLoadSlots8(op, source, offset, sourceGroupStride,
                                    resultVMIType, resultTypes, numGroups,
                                    rewriter, results);
  }
  if (slots == 1) {
    return lowerGroupSlotLoadSlots1(op, source, offset, sourceGroupStride,
                                    resultVMIType, resultTypes, rewriter,
                                    results);
  }
  return failure();
}

static LogicalResult
validateGroupBroadcastMappingDivisors(Operation *op, int64_t groupSize,
                                      int64_t selectorPeriod,
                                      int64_t sourceSlots,
                                      int64_t lanesPerPart,
                                      bool requiresSelectorPeriod,
                                      OneToNPatternRewriter &rewriter);

static FailureOr<std::optional<int64_t>> mapSlots1GroupBroadcastLane(
    Operation *op, VMIVRegType resultVMIType, int64_t part, int64_t chunk,
    int64_t lane, int64_t firstGroup, int64_t groupSize,
    int64_t selectorPeriod, int64_t sourcePartCount,
    OneToNPatternRewriter &rewriter);

static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
mapSlots1GroupBroadcastSources(
    Operation *op, VMIVRegType resultVMIType,
    ValueRange sourceParts, int64_t part, int64_t chunk, int64_t firstGroup,
    int64_t groupSize, int64_t selectorPeriod, int64_t lanesPerPart,
    OneToNPatternRewriter &rewriter) {
  if (failed(validateGroupBroadcastMappingDivisors(
          op, groupSize, selectorPeriod, /*sourceSlots=*/1, lanesPerPart,
          /*requiresSelectorPeriod=*/true, rewriter))) {
    return failure();
  }
  SmallVector<int64_t> laneSourceChunks(lanesPerPart, -1);
  SmallVector<int64_t> activeSourceChunks;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<std::optional<int64_t>> sourceChunk =
        mapSlots1GroupBroadcastLane(
            op, resultVMIType, part, chunk, lane, firstGroup, groupSize,
            selectorPeriod, static_cast<int64_t>(sourceParts.size()), rewriter);
    if (failed(sourceChunk)) {
      return failure();
    }
    if (!*sourceChunk) {
      continue;
    }
    laneSourceChunks[lane] = **sourceChunk;
    bool isNewSourceChunk =
        llvm::find(activeSourceChunks, **sourceChunk) == activeSourceChunks.end();
    if (isNewSourceChunk) {
      activeSourceChunks.push_back(**sourceChunk);
    }
  }
  if (activeSourceChunks.empty()) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast result chunk has no active lanes");
  }
  return std::make_pair(std::move(laneSourceChunks),
                        std::move(activeSourceChunks));
}

static FailureOr<Value> materializeSlots1GroupBroadcastMerge(
    Operation *op, Type resultType, ValueRange sourceParts, int64_t lanesPerPart,
    MaskType resultMaskType, ArrayRef<int64_t> laneSourceChunks,
    ArrayRef<int64_t> activeSourceChunks, OneToNPatternRewriter &rewriter,
    Value allMask);

static FailureOr<Value> materializeSlots1GroupBroadcastChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, int64_t part, int64_t chunk, int64_t firstGroup,
    int64_t groupSize, int64_t selectorPeriod, int64_t lanesPerPart,
    OneToNPatternRewriter &rewriter, Value allMask) {
  auto resultVRegType = dyn_cast<VRegType>(resultType);
  if (!resultVRegType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical vreg types");
  }
  FailureOr<MaskType> resultMaskType =
      getMaskTypeForVReg(resultVRegType, rewriter.getContext());
  if (failed(resultMaskType)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast cannot derive result mask type");
  }
  FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> mapping =
      mapSlots1GroupBroadcastSources(op, resultVMIType, sourceParts,
                                     part, chunk, firstGroup, groupSize,
                                     selectorPeriod, lanesPerPart, rewriter);
  if (failed(mapping)) {
    return failure();
  }
  return materializeSlots1GroupBroadcastMerge(
      op, resultType, sourceParts, lanesPerPart, *resultMaskType,
      mapping->first, mapping->second, rewriter, allMask);
}

static FailureOr<Value> materializeSlots1GroupBroadcastMerge(
    Operation *op, Type resultType, ValueRange sourceParts, int64_t lanesPerPart,
    MaskType resultMaskType, ArrayRef<int64_t> laneSourceChunks,
    ArrayRef<int64_t> activeSourceChunks, OneToNPatternRewriter &rewriter,
    Value allMask) {
  auto splatSource = [&rewriter, op, resultType, sourceParts, allMask](
                         int64_t chunkIndex) {
    return rewriter
        .create<VdupOp>(op->getLoc(), resultType, sourceParts[chunkIndex],
                        allMask, rewriter.getStringAttr("LOWEST"))
        .getResult();
  };
  Value merged = splatSource(activeSourceChunks.front());
  for (int64_t chunkIndex : llvm::drop_begin(activeSourceChunks)) {
    SmallVector<int8_t> laneMaskBits(lanesPerPart, 0);
    for (auto [lane, laneSourceChunk] : llvm::enumerate(laneSourceChunks)) {
      if (laneSourceChunk == chunkIndex) {
        laneMaskBits[lane] = 1;
      }
    }
    FailureOr<Value> laneMask = materializeConstantMaskChunk(
        op->getLoc(), resultMaskType, laneMaskBits, rewriter);
    if (failed(laneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create group_broadcast source merge mask");
    }
    Value splat = splatSource(chunkIndex);
    merged = rewriter
                 .create<VselOp>(op->getLoc(), resultType, splat, merged,
                                 *laneMask)
                 .getResult();
  }
  return merged;
}

enum class GroupBroadcastSelectorKind { Constant, LogicalRamp, VCGBlockRamp };

struct GroupBroadcastSelectorPlan {
  GroupBroadcastSelectorKind kind;
  int64_t period;
};

static LogicalResult validateGroupBroadcastMappingDivisors(
    Operation *op, int64_t groupSize, int64_t selectorPeriod,
    int64_t sourceSlots, int64_t lanesPerPart, bool requiresSelectorPeriod,
    OneToNPatternRewriter &rewriter) {
  if (groupSize <= 0 || sourceSlots <= 0 || lanesPerPart <= 0 ||
      (requiresSelectorPeriod && selectorPeriod <= 0)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires positive mapping divisors");
  }
  return success();
}

static FailureOr<std::optional<int64_t>> mapSlots1GroupBroadcastLane(
    Operation *op, VMIVRegType resultVMIType, int64_t part, int64_t chunk,
    int64_t lane, int64_t firstGroup, int64_t groupSize,
    int64_t selectorPeriod, int64_t sourcePartCount,
    OneToNPatternRewriter &rewriter) {
  FailureOr<bool> padding = isPaddingLane(resultVMIType, part, chunk, lane);
  if (failed(padding)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast failed to map result padding lanes");
  }
  if (*padding) {
    return std::optional<int64_t>();
  }
  FailureOr<int64_t> logical =
      mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
  if (failed(logical)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast failed to map a result lane");
  }
  int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
  int64_t safeSelectorPeriod = selectorPeriod > 0 ? selectorPeriod : 1;
  int64_t actualGroup = *logical / safeGroupSize;
  int64_t expectedGroup = firstGroup + lane / safeSelectorPeriod;
  if (actualGroup != expectedGroup) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast layout table row does not match its selector "
            "lowering plan");
  }
  if (actualGroup < 0 || actualGroup >= sourcePartCount) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast source chunk is out of range");
  }
  return std::optional<int64_t>(actualGroup);
}

static LogicalResult verifyGroupBroadcastChunkMapping(
    Operation *op, VMIVRegType resultVMIType,
    GroupBroadcastSelectorKind selectorKind, int64_t selectorPeriod,
    int64_t sourceSlots, int64_t groupSize, int64_t part, int64_t chunk,
    int64_t firstGroup, int64_t sourceChunk, int64_t lanesPerPart,
    OneToNPatternRewriter &rewriter) {
  bool requiresSelectorPeriod =
      selectorKind != GroupBroadcastSelectorKind::Constant;
  if (failed(validateGroupBroadcastMappingDivisors(
          op, groupSize, selectorPeriod, sourceSlots, lanesPerPart,
          requiresSelectorPeriod, rewriter))) {
    return failure();
  }
  int64_t safeGroupSize = groupSize;
  int64_t safeSelectorPeriod = selectorPeriod > 0 ? selectorPeriod : 1;
  int64_t safeSourceSlots = sourceSlots;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<bool> padding =
        isPaddingLane(resultVMIType, part, chunk, lane);
    if (failed(padding)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast failed to map result padding lanes");
    }
    if (*padding) {
      continue;
    }
    FailureOr<int64_t> logical =
        mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
    if (failed(logical)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast failed to map a result lane");
    }
    int64_t actualGroup = *logical / safeGroupSize;
    int64_t expectedGroup = firstGroup;
    if (selectorKind != GroupBroadcastSelectorKind::Constant) {
      expectedGroup += lane / safeSelectorPeriod;
    }
    if (actualGroup != expectedGroup ||
        actualGroup / safeSourceSlots != sourceChunk) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast layout table row does not match its selector "
              "lowering plan");
    }
  }
  return success();
}

static FailureOr<GroupBroadcastSelectorPlan> chooseGroupBroadcastSelectorPlan(
    Operation *op, const VMIGroupBroadcastLayoutFact &fact,
    VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) {
  if (fact.blockClass == VMIGroupBlockClass::FullPartMultiple) {
    return GroupBroadcastSelectorPlan{GroupBroadcastSelectorKind::Constant, 0};
  }
  bool isUnitStrideContiguous =
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (isUnitStrideContiguous) {
    return GroupBroadcastSelectorPlan{GroupBroadcastSelectorKind::LogicalRamp,
                                      fact.groupSize};
  }
  bool isStridedContiguous =
      resultLayout.isContiguous() && resultLayout.getLaneStride() > 1;
  if (isStridedContiguous) {
    return GroupBroadcastSelectorPlan{
        GroupBroadcastSelectorKind::VCGBlockRamp,
        fact.groupSize * resultLayout.getLaneStride()};
  }
  bool isDeinterleaved =
      resultLayout.isDeinterleaved() || resultLayout.isBlockDeinterleaved();
  if (isDeinterleaved) {
    return GroupBroadcastSelectorPlan{GroupBroadcastSelectorKind::VCGBlockRamp,
                                      fact.vcgBlockElems};
  }
  (void)rewriter.notifyMatchFailure(
      op, "group_broadcast layout table row has no selector lowering plan");
  return failure();
}

struct GroupBroadcastSelectorContext {
  Operation *op;
  GroupBroadcastSelectorKind kind;
  int64_t sourceLaneStride;
  std::optional<int64_t> selectorShift;
  std::optional<int64_t> sourceLaneStrideShift;
  Type indexScalarType;
  VRegType indexType;
  Value allMask;
  Value sharedRamp;
  llvm::DenseMap<int64_t, Value> selectorByBaseIndex;
};

struct GroupBroadcastLoweringContext {
  GroupBroadcastSelectorContext selector;
  int64_t sourceSlots;
  int64_t selectorPeriod;
};

struct GroupBroadcastSelectorMetadata {
  int64_t sourceSlots;
  int64_t sourceLaneStride;
  GroupBroadcastSelectorPlan selectorPlan;
  std::optional<int64_t> selectorShift;
  std::optional<int64_t> sourceLaneStrideShift;
};

static FailureOr<GroupBroadcastSelectorMetadata>
getGroupBroadcastSelectorMetadata(
    Operation *op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
    const VMIGroupBroadcastLayoutFact &fact,
    OneToNPatternRewriter &rewriter) {
  VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
  VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
  int64_t sourceSlots = sourceLayout.getSlots();
  int64_t sourceLaneStride = sourceLayout.getLaneStride();
  if (sourceSlots <= 0 || sourceLaneStride <= 0) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires explicit positive source group slots");
  }
  FailureOr<GroupBroadcastSelectorPlan> selectorPlan =
      chooseGroupBroadcastSelectorPlan(op, fact, resultLayout, rewriter);
  if (failed(selectorPlan)) {
    return failure();
  }
  std::optional<int64_t> selectorShift;
  std::optional<int64_t> sourceLaneStrideShift;
  if (selectorPlan->kind != GroupBroadcastSelectorKind::Constant) {
    selectorShift = getPowerOfTwoLog2(selectorPlan->period);
    sourceLaneStrideShift = getPowerOfTwoLog2(sourceLaneStride);
    if (!selectorShift || !sourceLaneStrideShift) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast ramp requires power-of-two selector period "
              "and source lane stride");
    }
  }
  return GroupBroadcastSelectorMetadata{
      sourceSlots, sourceLaneStride, *selectorPlan, selectorShift,
      sourceLaneStrideShift};
}

static FailureOr<GroupBroadcastSelectorContext>
createGroupBroadcastSelectorContext(
    Operation *op, VRegType sourceType,
    const GroupBroadcastSelectorMetadata &metadata,
    OneToNPatternRewriter &rewriter) {
  unsigned indexBits = pto::getPTOStorageElemBitWidth(
      sourceType.getElementType());
  auto indexElementType = IntegerType::get(
      rewriter.getContext(), indexBits,
      IntegerType::SignednessSemantics::Unsigned);
  auto indexScalarType = IntegerType::get(rewriter.getContext(), indexBits);
  auto indexType = VRegType::get(rewriter.getContext(),
                                 sourceType.getElementCount(), indexElementType);
  FailureOr<Value> allMask =
      createAllTrueMaskForVReg(op->getLoc(), indexType, rewriter);
  if (failed(allMask)) {
    return rewriter.notifyMatchFailure(
        op, "failed to create group_broadcast all mask");
  }
  return GroupBroadcastSelectorContext{
      op,
      metadata.selectorPlan.kind,
      metadata.sourceLaneStride,
      metadata.selectorShift,
      metadata.sourceLaneStrideShift,
      indexScalarType,
      indexType,
      *allMask,
      Value(),
      llvm::DenseMap<int64_t, Value>()};
}

static FailureOr<Value> materializeConstantGroupBroadcastSelector(
    GroupBroadcastSelectorContext &context, int64_t baseIndex,
    OneToNPatternRewriter &rewriter) {
  FailureOr<Value> baseScalar = createScalarOffsetConstant(
      context.op->getLoc(), context.indexScalarType, baseIndex, rewriter);
  if (failed(baseScalar)) {
    return failure();
  }
  return rewriter
      .create<VdupOp>(context.op->getLoc(), context.indexType, *baseScalar,
                      context.allMask, /*position=*/nullptr)
      .getResult();
}

static FailureOr<Value> materializeGroupBroadcastRamp(
    GroupBroadcastSelectorContext &context, int64_t baseIndex,
    OneToNPatternRewriter &rewriter) {
  if (!context.sharedRamp) {
    FailureOr<Value> zero = createScalarOffsetConstant(
        context.op->getLoc(), context.indexScalarType, 0, rewriter);
    if (failed(zero)) {
      return failure();
    }
    context.sharedRamp =
        rewriter.create<VciOp>(context.op->getLoc(), context.indexType, *zero,
                               StringAttr{})
            .getResult();
    if (*context.selectorShift != 0) {
      Value shift = createI16Constant(context.op->getLoc(),
                                      *context.selectorShift, rewriter);
      context.sharedRamp =
          rewriter
              .create<VshrsOp>(context.op->getLoc(), context.indexType,
                               context.sharedRamp, shift, context.allMask)
              .getResult();
    }
    if (*context.sourceLaneStrideShift != 0) {
      Value shift = createI16Constant(
          context.op->getLoc(), *context.sourceLaneStrideShift, rewriter);
      context.sharedRamp =
          rewriter
              .create<VshlsOp>(context.op->getLoc(), context.indexType,
                               context.sharedRamp, shift, context.allMask)
              .getResult();
    }
  }
  Value selector = context.sharedRamp;
  if (baseIndex != 0) {
    FailureOr<Value> baseScalar = createScalarOffsetConstant(
        context.op->getLoc(), context.indexScalarType, baseIndex, rewriter);
    if (failed(baseScalar)) {
      return failure();
    }
    selector = rewriter
                   .create<VaddsOp>(context.op->getLoc(), context.indexType,
                                    selector, *baseScalar, context.allMask)
                   .getResult();
  }
  return selector;
}

static FailureOr<GroupBroadcastLoweringContext>
createGroupBroadcastLoweringContext(
    Operation *op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
    const VMIGroupBroadcastLayoutFact &fact, VRegType sourceType,
    OneToNPatternRewriter &rewriter) {
  FailureOr<GroupBroadcastSelectorMetadata> metadata =
      getGroupBroadcastSelectorMetadata(op, sourceVMIType, resultVMIType, fact,
                                        rewriter);
  if (failed(metadata)) {
    return failure();
  }
  FailureOr<GroupBroadcastSelectorContext> selectorContext =
      createGroupBroadcastSelectorContext(op, sourceType, *metadata, rewriter);
  if (failed(selectorContext)) {
    return failure();
  }
  return GroupBroadcastLoweringContext{std::move(*selectorContext),
                                       metadata->sourceSlots,
                                       metadata->selectorPlan.period};
}

static FailureOr<Value> getGroupBroadcastSelector(
    GroupBroadcastSelectorContext &context, int64_t baseSlot,
    OneToNPatternRewriter &rewriter) {
  int64_t baseIndex = baseSlot * context.sourceLaneStride;
  auto cached = context.selectorByBaseIndex.find(baseIndex);
  if (cached != context.selectorByBaseIndex.end()) {
    return cached->second;
  }

  if (context.kind == GroupBroadcastSelectorKind::Constant) {
    FailureOr<Value> selector = materializeConstantGroupBroadcastSelector(
        context, baseIndex, rewriter);
    if (failed(selector)) {
      return failure();
    }
    context.selectorByBaseIndex.try_emplace(baseIndex, *selector);
    return *selector;
  }
  FailureOr<Value> selector =
      materializeGroupBroadcastRamp(context, baseIndex, rewriter);
  if (failed(selector)) {
    return failure();
  }
  context.selectorByBaseIndex.try_emplace(baseIndex, *selector);
  return *selector;
}

static FailureOr<Value> materializeGroupBroadcastChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts,
    GroupBroadcastSelectorKind selectorKind, int64_t selectorPeriod,
    int64_t sourceSlots, int64_t groupSize, int64_t part, int64_t chunk,
    int64_t firstGroup, int64_t sourceChunk, int64_t baseSlot,
    int64_t lanesPerPart, Value allMask,
    llvm::function_ref<FailureOr<Value>(int64_t)> getSelector,
    OneToNPatternRewriter &rewriter) {
  auto resultVRegType = dyn_cast<VRegType>(resultType);
  if (!resultVRegType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical vreg types");
  }
  if (failed(verifyGroupBroadcastChunkMapping(
          op, resultVMIType, selectorKind, selectorPeriod, sourceSlots,
          groupSize, part, chunk, firstGroup, sourceChunk, lanesPerPart,
          rewriter))) {
    return failure();
  }

  if (selectorKind == GroupBroadcastSelectorKind::Constant && sourceSlots == 1) {
    return rewriter
        .create<VdupOp>(op->getLoc(), resultType, sourceParts[sourceChunk],
                        allMask, rewriter.getStringAttr("LOWEST"))
        .getResult();
  }
  FailureOr<Value> selector = getSelector(baseSlot);
  if (failed(selector)) {
    return rewriter.notifyMatchFailure(
        op, "failed to create group_broadcast selector ramp");
  }
  return rewriter
      .create<VselrOp>(op->getLoc(), resultType, sourceParts[sourceChunk],
                       *selector)
      .getResult();
}

static FailureOr<Value> lowerGroupBroadcastChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, int64_t part, int64_t chunk,
    OneToNPatternRewriter &rewriter) {
  FailureOr<int64_t> firstLogical =
      mapPhysicalLaneToLogical(resultVMIType, part, chunk, 0);
  if (failed(firstLogical)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast failed to map the first result lane");
  }
  int64_t firstGroup = *firstLogical / fact.groupSize;
  int64_t sourceChunk = firstGroup / context.sourceSlots;
  int64_t baseSlot = firstGroup % context.sourceSlots;
  bool sourceChunkOutOfRange =
      sourceChunk < 0 || sourceChunk >= static_cast<int64_t>(sourceParts.size());
  if (sourceChunkOutOfRange) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast source chunk is out of range");
  }
  Value allMask = context.selector.allMask;
  if (context.sourceSlots == 1 &&
      context.selector.kind != GroupBroadcastSelectorKind::Constant) {
    return materializeSlots1GroupBroadcastChunk(
        op, resultType, resultVMIType, sourceParts, part, chunk, firstGroup,
        fact.groupSize, context.selectorPeriod, fact.lanesPerPart, rewriter,
        allMask);
  }
  if (failed(verifyGroupBroadcastChunkMapping(
          op, resultVMIType, context.selector.kind, context.selectorPeriod,
          context.sourceSlots, fact.groupSize, part, chunk, firstGroup,
          sourceChunk, fact.lanesPerPart, rewriter))) {
    return failure();
  }
  auto getSelector = [&context, &rewriter](int64_t slot) {
    return getGroupBroadcastSelector(context.selector, slot, rewriter);
  };
  return materializeGroupBroadcastChunk(
      op, resultType, resultVMIType, sourceParts, context.selector.kind,
      context.selectorPeriod, context.sourceSlots, fact.groupSize, part, chunk,
      firstGroup, sourceChunk, baseSlot, fact.lanesPerPart, allMask, getSelector,
      rewriter);
}

static FailureOr<VRegType> validateGroupBroadcastSources(
    Operation *op, ValueRange sourceParts,
    const VMIGroupBroadcastLayoutFact &fact,
    OneToNPatternRewriter &rewriter) {
  auto firstSourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!firstSourceType) {
    return rewriter.notifyMatchFailure(op,
                                       "group_broadcast source must be vreg");
  }
  bool hasNonUniformSourceType =
      llvm::any_of(sourceParts, [&firstSourceType](Value sourcePart) {
        return sourcePart.getType() != firstSourceType;
      });
  if (hasNonUniformSourceType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical source vreg types");
  }
  bool sourceLaneCountMismatch =
      firstSourceType.getElementCount() != fact.lanesPerPart;
  if (sourceLaneCountMismatch) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast physical source lanes do not match the supported "
            "layout row");
  }
  unsigned indexBits =
      pto::getPTOStorageElemBitWidth(firstSourceType.getElementType());
  bool unsupportedIndexBits =
      indexBits != 8 && indexBits != 16 && indexBits != 32;
  if (unsupportedIndexBits) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires 8/16/32-bit index elements");
  }
  return firstSourceType;
}

static FailureOr<Value> lowerGroupBroadcastResultChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, VRegType expectedSourceType,
    int64_t part, int64_t chunk, OneToNPatternRewriter &rewriter);

static LogicalResult lowerGroupBroadcastResultChunks(
    Operation *op, ValueRange sourceParts, VMIVRegType resultVMIType,
    TypeRange resultTypes, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, VRegType expectedSourceType,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<int64_t> resultLayoutFactor = getDataLayoutFactor(resultVMIType);
  if (failed(resultLayoutFactor)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires a computable result layout factor");
  }
  results.clear();
  results.resize(resultTypes.size());
  int64_t flatIndex = 0;
  for (int64_t part = 0; part < *resultLayoutFactor; ++part) {
    FailureOr<int64_t> chunks =
        *resultLayoutFactor == 1
            ? FailureOr<int64_t>(resultTypes.size())
            : getDataChunksInPart(resultVMIType, part);
    if (failed(chunks)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast failed to enumerate result chunks");
    }
    for (int64_t chunk = 0; chunk < *chunks; ++chunk, ++flatIndex) {
      if (flatIndex >= static_cast<int64_t>(resultTypes.size())) {
        return rewriter.notifyMatchFailure(
            op, "group_broadcast physical result count is too small");
      }
      FailureOr<Value> chunkResult = lowerGroupBroadcastResultChunk(
          op, resultTypes[flatIndex], resultVMIType, sourceParts, fact, context,
          expectedSourceType, part, chunk, rewriter);
      if (failed(chunkResult)) {
        return failure();
      }
      results[flatIndex] = *chunkResult;
    }
  }
  if (flatIndex != static_cast<int64_t>(resultTypes.size())) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast physical result count is too large");
  }
  return success();
}

static FailureOr<Value> lowerGroupBroadcastResultChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, VRegType expectedSourceType,
    int64_t part, int64_t chunk, OneToNPatternRewriter &rewriter) {
  auto resultVRegType = dyn_cast<VRegType>(resultType);
  bool mismatchedResultType =
      !resultVRegType || resultVRegType != expectedSourceType;
  if (mismatchedResultType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical vreg types");
  }
  return lowerGroupBroadcastChunk(op, resultType, resultVMIType, sourceParts,
                                  fact, context, part, chunk, rewriter);
}

static LogicalResult lowerGroupBroadcastParts(
    Operation *op, ValueRange sourceParts, VMIVRegType sourceVMIType,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  bool emptyArity = sourceParts.empty() || resultTypes.empty();
  if (emptyArity) {
    return rewriter.notifyMatchFailure(op, "group_broadcast arity mismatch");
  }

  std::string layoutReason;
  VMILayoutSupport supports;
  FailureOr<VMIGroupBroadcastLayoutFact> fact =
      supports.getGroupBroadcastLayoutFactForLayouts(
          sourceVMIType, resultVMIType, numGroups, &layoutReason);
  if (failed(fact)) {
    return rewriter.notifyMatchFailure(
        op, Twine("group_broadcast requires a supported layout table row; ") +
                layoutReason);
  }

  FailureOr<VRegType> firstSourceType =
      validateGroupBroadcastSources(op, sourceParts, *fact, rewriter);
  if (failed(firstSourceType)) {
    return failure();
  }
  FailureOr<GroupBroadcastLoweringContext> loweringContext =
      createGroupBroadcastLoweringContext(
          op, sourceVMIType, resultVMIType, *fact, *firstSourceType, rewriter);
  if (failed(loweringContext)) {
    return failure();
  }
  GroupBroadcastLoweringContext &context = *loweringContext;
  return lowerGroupBroadcastResultChunks(
      op, sourceParts, resultVMIType, resultTypes, *fact, context,
      *firstSourceType, rewriter, results);
}


