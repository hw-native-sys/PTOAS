// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals5.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

constexpr int64_t kWidePartElemThreshold = 128;

template <typename OpTy, typename GroupReduceOpTy, typename RowReduceOpTy,
          typename CombineOpTy>
struct OneToNVMIGroupReduceOpPattern : OneToNOpConversionPattern<OpTy> {
  using OneToNOpConversionPattern<OpTy>::OneToNOpConversionPattern;

private:
  int64_t getCompactIntegerIdentity(IntegerType type) const {
    unsigned width = type.getWidth();
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxIOp>) {
      return type.isSigned() ? APInt::getSignedMinValue(width).getSExtValue()
                             : 0;
    }
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinIOp>) {
      return type.isSigned() ? APInt::getSignedMaxValue(width).getSExtValue()
                             : APInt::getMaxValue(width).getZExtValue();
    }
    return 0;
  }

  FailureOr<std::pair<Value, Value>> prepareCompactReduction(
      OpTy op, Value source, Value mask, int64_t partIndex,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = cast<VRegType>(source.getType());
    auto elementType = dyn_cast<IntegerType>(sourceType.getElementType());
    bool needsWidening = elementType && elementType.getWidth() == kElementBits8;
    if (!needsWidening) {
      return std::make_pair(source, mask);
    }
    // Widen each half inside the instruction lowering, without changing the
    // logical VMI value's one-carrier layout.
    auto wideElementType = IntegerType::get(
        rewriter.getContext(), kElementBits16,
        elementType.isSigned() ? IntegerType::SignednessSemantics::Signed
                               : IntegerType::SignednessSemantics::Unsigned);
    auto wideType = VRegType::get(rewriter.getContext(),
                                  sourceType.getElementCount() / kPairWidth,
                                  wideElementType);
    Value part = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), partIndex);
    Value extended = elementType.isSigned()
                         ? rewriter.create<VsunpackOp>(op.getLoc(), wideType,
                                                       source, part).getResult()
                         : rewriter.create<VzunpackOp>(op.getLoc(), wideType,
                                                       source, part).getResult();
    auto wideMaskType = MaskType::get(rewriter.getContext(), "b16");
    Value wideMask = rewriter.create<PunpackOp>(
        op.getLoc(), wideMaskType, mask,
        rewriter.getStringAttr(partIndex == 0 ? "LOWER" : "HIGHER"));
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddIOp>) {
      // Add has the same zero identity before and after extension.
      return std::make_pair(extended, wideMask);
    }
    FailureOr<Value> allMask = createAllTrueMask(op.getLoc(), wideMaskType, rewriter);
    FailureOr<Value> identity = createScalarOffsetConstant(
        op.getLoc(), rewriter.getI16Type(), getCompactIntegerIdentity(elementType), rewriter);
    bool failedMaterialization = failed(allMask) || failed(identity);
    if (failedMaterialization) {
      return failure();
    }
    // Fill inactive lanes with the original element type's identity. Using
    // the widened type's extrema would change an empty signed min/max group
    // when its result is narrowed back to eight bits.
    Value neutral = rewriter.create<VdupOp>(op.getLoc(), wideType, *identity,
                                           *allMask, /*position=*/nullptr);
    Value selected = rewriter.create<VselOp>(op.getLoc(), wideType, extended,
                                            neutral, wideMask);
    return std::make_pair(selected, *allMask);
  }

  LogicalResult lowerSingletonGroups(
      OpTy op, Value source, Value mask, VRegType resultType,
      OneToNPatternRewriter &rewriter) const {
    auto elementType = cast<IntegerType>(resultType.getElementType());
    auto scalarType = rewriter.getIntegerType(elementType.getWidth());
    FailureOr<Value> identity = createScalarOffsetConstant(
        op.getLoc(), scalarType, getCompactIntegerIdentity(elementType), rewriter);
    FailureOr<Value> allMask = createAllTrueMaskForVReg(
        op.getLoc(), resultType, rewriter);
    if (failed(identity) || failed(allMask)) {
      return failure();
    }
    Value neutral = rewriter.create<VdupOp>(op.getLoc(), resultType, *identity,
                                           *allMask, /*position=*/nullptr);
    Value selected = rewriter.create<VselOp>(op.getLoc(), resultType, source,
                                            neutral, mask);
    replaceOpWithFlatConvertedValues(rewriter, op, ValueRange{selected},
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> reduceCompactPart(
      OpTy op, std::pair<Value, Value> input, int64_t begin, int64_t end,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = cast<VRegType>(input.first.getType());
    auto maskType = cast<MaskType>(input.second.getType());
    Value active = input.second;
    if (begin != 0 || end != sourceType.getElementCount()) {
      FailureOr<Value> groupMask = createLaneRangeMask(
          op.getLoc(), maskType, begin, end, rewriter);
      FailureOr<Value> allMask = createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(groupMask) || failed(allMask)) {
        return failure();
      }
      active = rewriter.create<PandOp>(op.getLoc(), maskType, active,
                                      *groupMask, *allMask);
    }
    FailureOr<VRegType> rowType = getRowResultType(sourceType, sourceType);
    if (failed(rowType)) {
      return failure();
    }
    return rewriter.create<RowReduceOpTy>(op.getLoc(), *rowType,
                                          input.first, active).getResult();
  }

  FailureOr<Value> buildCompactGroupResult(
      OpTy op, ArrayRef<std::pair<Value, Value>> inputs, VRegType resultType,
      int64_t group, int64_t groupSize,
      OneToNPatternRewriter &rewriter) const {
    int64_t lanes = cast<VRegType>(inputs.front().first.getType()).getElementCount();
    Value reduced;
    for (auto [index, input] : llvm::enumerate(inputs)) {
      int64_t partBegin = static_cast<int64_t>(index) * lanes;
      int64_t begin = std::max<int64_t>(0, group * groupSize - partBegin);
      int64_t end = std::min<int64_t>(lanes, (group + 1) * groupSize - partBegin);
      if (begin >= end) {
        continue;
      }
      FailureOr<Value> partial = reduceCompactPart(op, input, begin, end, rewriter);
      if (failed(partial)) {
        return failure();
      }
      if (!reduced) {
        reduced = *partial;
        continue;
      }
      auto rowType = cast<VRegType>(reduced.getType());
      FailureOr<MaskType> rowMaskType =
          getMaskTypeForVReg(rowType, rewriter.getContext());
      if (failed(rowMaskType)) {
        return failure();
      }
      FailureOr<Value> firstLane = createPrefixMaskForActiveLanes(
          op.getLoc(), *rowMaskType, 1, rewriter);
      if (failed(firstLane)) {
        return failure();
      }
      reduced = rewriter.create<CombineOpTy>(op.getLoc(), rowType, reduced,
                                            *partial, *firstLane);
    }
    if (!reduced) {
      return failure();
    }
    // A 16-bit integer vcadd also returns a 32-bit sum, typed by
    // getRowResultType(). Combine wide partials before taking the low bits.
    // Only lane zero of this view is a logical group value; buildCompactPacket
    // selects each later group's low lane into its own slot. Unlike VCG's
    // eight simultaneous sums, this needs no per-register vpack.
    return bitcastVReg(op.getLoc(), reduced, resultType, rewriter);
  }

  FailureOr<Value> buildCompactPacket(
      OpTy op, ArrayRef<std::pair<Value, Value>> inputs, VRegType resultType,
      MaskType resultMaskType, int64_t groupSize,
      OneToNPatternRewriter &rewriter) const {
    Value packet;
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    for (int64_t group = 0; group < numGroups; ++group) {
      FailureOr<Value> reduced = buildCompactGroupResult(
          op, inputs, resultType, group, groupSize, rewriter);
      if (failed(reduced)) {
        return failure();
      }
      if (!packet) {
        // Only slot zero is live initially: no broadcast or slot mask needed.
        packet = *reduced;
        continue;
      }
      FailureOr<Value> slotMask = createLaneRangeMask(
          op.getLoc(), resultMaskType, group, group + 1, rewriter);
      if (failed(slotMask)) {
        return failure();
      }
      Value splat = rewriter.create<VdupOp>(
          op.getLoc(), resultType, *reduced, *slotMask,
          rewriter.getStringAttr("LOWEST"));
      packet = rewriter.create<VselOp>(op.getLoc(), resultType, splat,
                                       packet, *slotMask);
    }
    return packet;
  }

  // True when the compact result hands back one value per group rather than a
  // single eight-slot packet.
  bool isRowLocalSlots1Result(OpTy op, int64_t numGroups) const {
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    return layout && layout.isGroupSlots() &&
           layout.getNumGroups() == numGroups && layout.getSlots() == 1;
  }

  // Collects the (source, mask) carriers each group value is reduced from.
  FailureOr<SmallVector<std::pair<Value, Value>, kPairWidth>>
  collectCompactInputs(OpTy op, ValueRange sourceParts, ValueRange maskParts,
                       bool twoWideParts,
                       OneToNPatternRewriter &rewriter) const {
    SmallVector<std::pair<Value, Value>, kPairWidth> inputs;
    if (!twoWideParts) {
      // A grouped reduction over a contiguous multi-carrier source maps each
      // group window across the physical parts (buildCompactGroupResult).
      for (size_t index = 0; index < sourceParts.size(); ++index) {
        inputs.push_back(std::make_pair(sourceParts[index], maskParts[index]));
      }
      return inputs;
    }
    // Eight-bit two-wide sources unpack each carrier into a pair of 16-bit
    // reductions, so they consume exactly one physical source part.
    const bool singleCarrier = sourceParts.size() == 1;
    if (!singleCarrier) {
      return rewriter.notifyMatchFailure(
          op, "compact eight-bit group_reduce requires one source carrier");
    }
    for (int64_t part = 0; part < kPairWidth; ++part) {
      auto input = prepareCompactReduction(op, sourceParts.front(),
                                           maskParts.front(), part, rewriter);
      if (failed(input)) {
        return failure();
      }
      inputs.push_back(*input);
    }
    return inputs;
  }

  // Row-local slots=1 results hand back one physical part per group.
  LogicalResult lowerRowLocalSlots1Result(
      OpTy op, ArrayRef<std::pair<Value, Value>> inputs, VRegType resultType,
      int64_t groupSize, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(numGroups);
    for (int64_t group = 0; group < numGroups; ++group) {
      FailureOr<Value> reduced = buildCompactGroupResult(
          op, inputs, resultType, group, groupSize, rewriter);
      if (failed(reduced)) {
        return failure();
      }
      results.push_back(*reduced);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerCompactRows(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t groupSize,
      OneToNPatternRewriter &rewriter) const {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    const bool rowLocalSlots1 = isRowLocalSlots1Result(op, numGroups);
    size_t expectedResultCount =
        rowLocalSlots1 ? static_cast<size_t>(numGroups) : 1;
    bool invalidArity = sourceParts.empty() ||
                        sourceParts.size() != maskParts.size() ||
                        resultTypes.size() != expectedResultCount;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "compact group_reduce requires matching source/mask parts and "
              "one result part per group");
    }
    auto resultType = cast<VRegType>(resultTypes.front());
    if (groupSize == 1 && isa<IntegerType>(resultType.getElementType())) {
      return lowerSingletonGroups(op, sourceParts.front(), maskParts.front(),
                                   resultType, rewriter);
    }
    auto logicalType = cast<VMIVRegType>(op.getSource().getType());
    auto integerType = dyn_cast<IntegerType>(logicalType.getElementType());
    bool twoWideParts = integerType && integerType.getWidth() == kElementBits8 &&
                        logicalType.getElementCount() > kWidePartElemThreshold;
    FailureOr<SmallVector<std::pair<Value, Value>, kPairWidth>> inputs =
        collectCompactInputs(op, sourceParts, maskParts, twoWideParts, rewriter);
    if (failed(inputs)) {
      return failure();
    }
    if (rowLocalSlots1) {
      return lowerRowLocalSlots1Result(op, *inputs, resultType, groupSize,
                                       numGroups, rewriter);
    }
    FailureOr<MaskType> resultMaskType =
        getMaskTypeForVReg(resultType, rewriter.getContext());
    if (failed(resultMaskType)) {
      return failure();
    }
    FailureOr<Value> packet = buildCompactPacket(
        op, *inputs, resultType, *resultMaskType, groupSize, rewriter);
    if (failed(packet)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, ValueRange{*packet},
                                     *this->getTypeConverter());
    return success();
  }

  Value createNativeGroupResult(OpTy op, VRegType resultType, Value source,
                                Value mask, OneToNPatternRewriter &rewriter) const {
    Value reduced = rewriter.create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                    source, mask);
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddIOp>) {
      auto elementType = cast<IntegerType>(resultType.getElementType());
      if (elementType.getWidth() == kElementBits16) {
        // A5 VCG integer addition returns eight 32-bit sums. Restore the
        // declared 16-bit group slots by truncating each sum, not by reading
        // alternating low/high halves as distinct logical groups.
        // VCG max/min retain 16-bit values and need no such narrowing.
        // Compact vcadd also widens, but builds its packet one scalar at a
        // time (see buildCompactGroupResult). Keep gs(8) here; exposing raw
        // gs(8, 2) results requires a separate consumer-layout audit, described
        // in docs/isa/vmi-isa/05-reduce.md.
        auto wideElementType = IntegerType::get(
            rewriter.getContext(), kElementBits32,
            IntegerType::SignednessSemantics::Unsigned);
        auto wideType = VRegType::get(
            rewriter.getContext(), resultType.getElementCount() / 2,
            wideElementType);
        Value wide = rewriter.create<VbitcastOp>(op.getLoc(), wideType, reduced);
        auto packedType = VRegType::get(
            rewriter.getContext(), resultType.getElementCount(),
            IntegerType::get(rewriter.getContext(), kElementBits16,
                             IntegerType::SignednessSemantics::Unsigned));
        Value packed = rewriter.create<VpackOp>(op.getLoc(), packedType, wide,
                                                rewriter.getStringAttr("LOWER"));
        return rewriter.create<VbitcastOp>(op.getLoc(), resultType, packed);
      }
    }
    return reduced;
  }

  FailureOr<Value> buildOneBlockGroupResult(
      OpTy op, Value sourcePart, Value maskPart, Type resultType,
      VRegType expectedResultType, MaskType expectedMaskType,
      OneToNPatternRewriter *rewriter) const {
    bool mismatchedTypes = sourcePart.getType() != expectedResultType ||
                           maskPart.getType() != expectedMaskType ||
                           resultType != expectedResultType;
    if (mismatchedTypes) {
      return rewriter->notifyMatchFailure(
          op, "vcg group_reduce path requires uniform physical chunk types");
    }
    return createNativeGroupResult(op, expectedResultType, sourcePart,
                                   maskPart, *rewriter);
  }

  LogicalResult lowerOneBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.size() != maskParts.size() ||
                        sourceParts.size() != resultTypes.size() ||
                        sourceParts.empty();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires matching physical arity");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires physical vreg/mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourceIndex, sourcePart] : llvm::enumerate(sourceParts)) {
      FailureOr<Value> result = buildOneBlockGroupResult(
          op, sourcePart, maskParts[sourceIndex], resultTypes[sourceIndex],
          resultType, maskType, &rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildTwoBlockGroupResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t resultIndex, int64_t resultPartCount,
      int64_t numGroups, VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    Value loSource = sourceParts[resultIndex];
    Value hiSource = sourceParts[resultPartCount + resultIndex];
    Value loMask = maskParts[resultIndex];
    Value hiMask = maskParts[resultPartCount + resultIndex];
    bool mismatchedTypes = resultTypes[resultIndex] != resultType ||
                           loSource.getType() != resultType ||
                           hiSource.getType() != resultType ||
                           loMask.getType() != maskType ||
                           hiMask.getType() != maskType;
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "two-block group_reduce requires uniform physical types");
    }
    int64_t activeGroups = std::min<int64_t>(8, numGroups - resultIndex * 8);
    FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(combineMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create two-block group_reduce combine mask");
    }
    Value lo = createNativeGroupResult(op, resultType, loSource, loMask, rewriter);
    Value hi = createNativeGroupResult(op, resultType, hiSource, hiMask, rewriter);
    return rewriter
        .create<CombineOpTy>(op.getLoc(), resultType, lo, hi, *combineMask)
        .getResult();
  }

  LogicalResult lowerTwoBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    int64_t resultPartCount = resultTypes.size();
    bool invalidArity = static_cast<int64_t>(sourceParts.size()) !=
                            resultPartCount * 2 ||
                        maskParts.size() != sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "two-block group_reduce arity mismatch");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "two-block group_reduce requires physical vreg/mask");
    }

    SmallVector<Value> results;
    results.reserve(resultPartCount);
    for (int64_t resultIndex = 0; resultIndex < resultPartCount;
         ++resultIndex) {
      FailureOr<Value> result = buildTwoBlockGroupResult(
          op, sourceParts, maskParts, resultTypes, resultIndex, resultPartCount,
          numGroups, resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildFourBlockGroupResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t resultIndex, int64_t resultPartCount,
      int64_t numGroups, VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    int64_t activeGroups = std::min<int64_t>(8, numGroups - resultIndex * 8);
    FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(combineMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create four-block group_reduce combine mask");
    }
    SmallVector<Value, kQuadWidth> partials;
    partials.reserve(kQuadWidth);
    for (int64_t part = 0; part < kQuadWidth; ++part) {
      int64_t sourceIndex = part * resultPartCount + resultIndex;
      Value source = sourceParts[sourceIndex];
      Value mask = maskParts[sourceIndex];
      bool mismatchedTypes = resultTypes[resultIndex] != resultType ||
                             source.getType() != resultType ||
                             mask.getType() != maskType;
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "four-block group_reduce requires uniform physical types");
      }
      partials.push_back(
          createNativeGroupResult(op, resultType, source, mask, rewriter));
    }
    Value sum01 = rewriter
                      .create<CombineOpTy>(op.getLoc(), resultType, partials[0],
                                           partials[1], *combineMask)
                      .getResult();
    Value sum23 = rewriter
                      .create<CombineOpTy>(op.getLoc(), resultType, partials[2],
                                           partials[3], *combineMask)
                      .getResult();
    return rewriter
        .create<CombineOpTy>(op.getLoc(), resultType, sum01, sum23,
                             *combineMask)
        .getResult();
  }

  LogicalResult lowerFourBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    int64_t resultPartCount = resultTypes.size();
    bool invalidArity = static_cast<int64_t>(sourceParts.size()) !=
                            resultPartCount * 4 ||
                        maskParts.size() != sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "four-block group_reduce arity mismatch");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "four-block group_reduce requires physical vreg/mask");
    }

    SmallVector<Value> results;
    results.reserve(resultPartCount);
    for (int64_t resultIndex = 0; resultIndex < resultPartCount;
         ++resultIndex) {
      FailureOr<Value> result = buildFourBlockGroupResult(
          op, sourceParts, maskParts, resultTypes, resultIndex,
          resultPartCount, numGroups, resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<SmallVector<Value>> buildDeinterleaved2GroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      int64_t groupCount, int64_t chunksPerGroup, int64_t chunksPerPart,
      VRegType sourcePartType, VRegType rowResultType, MaskType maskType,
      Value firstLaneMask, OneToNPatternRewriter *rewriter) const {
    SmallVector<Value> results;
    results.reserve(groupCount);
    for (int64_t group = 0; group < groupCount; ++group) {
      Value accumulator;
      for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
        int64_t loIndex = group * chunksPerGroup + chunk;
        int64_t hiIndex = chunksPerPart + loIndex;
        bool mismatchedTypes =
            sourceParts[loIndex].getType() != sourcePartType ||
            sourceParts[hiIndex].getType() != sourcePartType ||
            maskParts[loIndex].getType() != maskType ||
            maskParts[hiIndex].getType() != maskType;
        if (mismatchedTypes) {
          return rewriter->notifyMatchFailure(
              op, "deinterleaved=2 group_reduce requires uniform physical "
                  "chunk types");
        }
        Value low = rewriter
                        ->create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                               sourceParts[loIndex],
                                               maskParts[loIndex])
                        .getResult();
        Value high = rewriter
                         ->create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                                sourceParts[hiIndex],
                                                maskParts[hiIndex])
                         .getResult();
        Value pair = rewriter
                         ->create<CombineOpTy>(op.getLoc(), rowResultType, low,
                                              high, firstLaneMask)
                         .getResult();
        accumulator =
            accumulator
                ? rewriter
                      ->create<CombineOpTy>(op.getLoc(), rowResultType, pair,
                                           accumulator, firstLaneMask)
                      .getResult()
                : pair;
      }
      results.push_back(accumulator);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> restoreDeinterleaved2GroupResults(
      OpTy op, ArrayRef<Value> reducedResults, TypeRange resultTypes,
      VRegType resultType, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Value reducedResult : reducedResults) {
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), reducedResult, resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore deinterleaved=2 group result type");
      }
      results.push_back(*finalResult);
    }
    return results;
  }

  struct Deinterleaved2GroupReduceTypes {
    VRegType resultType;
    MaskType maskType;
    VRegType sourcePartType;
    VRegType rowResultType;
    MaskType rowMaskType;
  };

  FailureOr<Deinterleaved2GroupReduceTypes>
  getDeinterleaved2GroupReduceTypes(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce result must be vreg");
      }
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires physical vreg/mask");
    }
    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce source must be vreg");
    }
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive deinterleaved=2 row-reduction type");
    }
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive deinterleaved=2 combine mask type");
    }
    return Deinterleaved2GroupReduceTypes{
        resultType, maskType, sourcePartType, *rowResultType, *rowMaskType};
  }

  FailureOr<std::pair<int64_t, int64_t>> validateFullDeinterleaved2Shape(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter *rewriter) const {
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getSlots() == 1;
    if (!rowLocalSlots1Result) {
      return rewriter->notifyMatchFailure(
          op, "deinterleaved=2 full group_reduce requires slots=1 result");
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(sourceVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter->notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires known physical lanes");
    }
    int64_t safeLanesPerPart = *lanesPerPart > 0 ? *lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool invalidGroupSize = groupSize % (2 * safeLanesPerPart) != 0;
    if (invalidGroupSize) {
      return rewriter->notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires group size to be a "
              "multiple of two physical chunks");
    }
    int64_t groupCount = sourceVMIType.getElementCount() / safeGroupSize;
    int64_t chunksPerGroup = groupSize / (2 * safeLanesPerPart);
    int64_t chunksPerPart = groupCount * chunksPerGroup;
    bool invalidArity = sourceParts.size() != maskParts.size() ||
                        static_cast<int64_t>(sourceParts.size()) !=
                            2 * chunksPerPart ||
                        static_cast<int64_t>(resultTypes.size()) != groupCount;
    if (invalidArity) {
      return rewriter->notifyMatchFailure(
          op, "deinterleaved=2 group_reduce arity mismatch");
    }
    return std::make_pair(groupCount, chunksPerGroup);
  }

  LogicalResult lowerFullDeinterleaved2(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    FailureOr<std::pair<int64_t, int64_t>> shape =
        validateFullDeinterleaved2Shape(
            op, sourceVMIType, resultVMIType, sourceParts, maskParts,
            resultTypes, groupSize, &rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t groupCount = shape->first;
    int64_t chunksPerGroupPerPart = shape->second;
    int64_t chunksPerPart = groupCount * chunksPerGroupPerPart;
    FailureOr<Deinterleaved2GroupReduceTypes> types =
        getDeinterleaved2GroupReduceTypes(op, sourceParts, maskParts,
                                          resultTypes, rewriter);
    if (failed(types)) {
      return failure();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), types->rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create deinterleaved=2 group_reduce lane mask");
    }
    FailureOr<SmallVector<Value>> reducedResults =
        buildDeinterleaved2GroupResults(
            op, sourceParts, maskParts, groupCount, chunksPerGroupPerPart,
            chunksPerPart, types->sourcePartType, types->rowResultType,
            types->maskType,
            *firstLaneMask, &rewriter);
    if (failed(reducedResults)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = restoreDeinterleaved2GroupResults(
        op, *reducedResults, resultTypes, types->resultType, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildContiguousGroupReduceResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts, int64_t group,
      int64_t chunksPerGroup, VRegType sourcePartType, VRegType rowResultType,
      MaskType maskType, Value firstLaneMask,
      OneToNPatternRewriter *rewriter) const {
    Value accumulator;
    for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
      int64_t index = group * chunksPerGroup + chunk;
      bool mismatchedTypes = sourceParts[index].getType() != sourcePartType ||
                             maskParts[index].getType() != maskType;
      if (mismatchedTypes) {
        return rewriter->notifyMatchFailure(
            op, "group_reduce requires uniform physical chunk types");
      }
      Value reduced = rewriter
                          ->create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                                 sourceParts[index],
                                                 maskParts[index])
                          .getResult();
      accumulator = accumulator
                        ? rewriter
                              ->create<CombineOpTy>(op.getLoc(), rowResultType,
                                                   reduced, accumulator,
                                                   firstLaneMask)
                              .getResult()
                        : reduced;
    }
    return accumulator;
  }

  FailureOr<SmallVector<Value>> buildContiguousGroupReduceResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      int64_t groupCount, int64_t chunksPerGroup, VRegType sourcePartType,
      VRegType rowResultType, MaskType maskType, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(groupCount);
    for (int64_t group = 0; group < groupCount; ++group) {
      FailureOr<Value> result = buildContiguousGroupReduceResult(
          op, sourceParts, maskParts, group, chunksPerGroup, sourcePartType,
          rowResultType, maskType, firstLaneMask, &rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> restoreContiguousGroupResults(
      OpTy op, ArrayRef<Value> reducedResults, TypeRange resultTypes,
      VRegType resultType, int64_t groupCount, int64_t chunksPerGroup,
      bool rowLocalSlots1Result,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results(resultTypes.size());
    for (int64_t group = 0; group < groupCount; ++group) {
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), reducedResults[group], resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");
      }
      int64_t destChunk = rowLocalSlots1Result ? group : group * chunksPerGroup;
      if (rowLocalSlots1Result) {
        results[destChunk] = *finalResult;
      } else {
        for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
          results[destChunk + chunk] = *finalResult;
        }
      }
    }
    return results;
  }

  struct ContiguousGroupReduceTypes {
    VRegType resultType;
    MaskType maskType;
    VRegType sourcePartType;
    VRegType rowResultType;
    MaskType rowMaskType;
  };

  struct ContiguousGroupReduceShape {
    int64_t lanesPerPart;
    int64_t groupCount;
    int64_t chunksPerGroup;
    bool rowLocalSlots1Result;
    int64_t expectedResultParts;
  };

  FailureOr<ContiguousGroupReduceShape> getContiguousGroupReduceShape(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(op, sourceVMIType, groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter))) {
      return failure();
    }
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getNumGroups() == groupCount &&
                                resultLayout.getSlots() == 1;
    int64_t expectedResultParts =
        rowLocalSlots1Result ? groupCount : groupCount * chunksPerGroup;
    bool invalidArity =
        sourceParts.size() != maskParts.size() ||
        static_cast<int64_t>(sourceParts.size()) != groupCount * chunksPerGroup ||
        static_cast<int64_t>(resultTypes.size()) != expectedResultParts;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires matching source/mask/result arity");
    }
    return ContiguousGroupReduceShape{lanesPerPart, groupCount, chunksPerGroup,
                                      rowLocalSlots1Result, expectedResultParts};
  }

  FailureOr<ContiguousGroupReduceTypes> getContiguousGroupReduceTypes(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(
            op, "group_reduce result must be vreg");
      }
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires physical vreg result and mask");
    }
    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType) {
      return rewriter.notifyMatchFailure(op,
                                         "group_reduce source must be vreg");
    }
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group row-reduction type");
    }
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group combine mask type");
    }
    return ContiguousGroupReduceTypes{
        resultType, maskType, sourcePartType, *rowResultType, *rowMaskType};
  }

  LogicalResult lowerRowLocalContiguousGroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, const ContiguousGroupReduceShape &shape,
      const ContiguousGroupReduceTypes &types, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    // One contiguous group maps to one result chunk: materialize the row
    // reduction and its final type view per group so the emitted ops keep
    // master's per-group ordering.
    SmallVector<Value> results(resultTypes.size());
    for (int64_t group = 0;
         group < shape.groupCount &&
         static_cast<size_t>(group) < results.size();
         ++group) {
      FailureOr<Value> reduced = buildContiguousGroupReduceResult(
          op, sourceParts, maskParts, group, shape.chunksPerGroup,
          types.sourcePartType, types.rowResultType, types.maskType,
          firstLaneMask, &rewriter);
      if (failed(reduced)) {
        return failure();
      }
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), *reduced, types.resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");
      }
      results[group] = *finalResult;
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerContiguousRows(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    FailureOr<ContiguousGroupReduceShape> shape =
        getContiguousGroupReduceShape(op, sourceVMIType, resultVMIType,
                                      sourceParts, maskParts, resultTypes,
                                      groupSize, rewriter);
    if (failed(shape)) {
      return failure();
    }
    FailureOr<ContiguousGroupReduceTypes> types =
        getContiguousGroupReduceTypes(op, sourceParts, maskParts, resultTypes,
                                      rewriter);
    if (failed(types)) {
      return failure();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), types->rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to create group_reduce masks");
    }
    if (shape->rowLocalSlots1Result) {
      return lowerRowLocalContiguousGroupResults(
          op, sourceParts, maskParts, resultTypes, *shape, *types,
          *firstLaneMask, rewriter);
    }
    return lowerBatchedContiguousGroupResults(
        op, sourceParts, maskParts, resultTypes, *shape, *types,
        *firstLaneMask, rewriter);
  }

  LogicalResult lowerBatchedContiguousGroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, const ContiguousGroupReduceShape &shape,
      const ContiguousGroupReduceTypes &types, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Value>> reducedResults =
        buildContiguousGroupReduceResults(
            op, sourceParts, maskParts, shape.groupCount, shape.chunksPerGroup,
            types.sourcePartType, types.rowResultType, types.maskType,
            firstLaneMask, rewriter);
    if (failed(reducedResults)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = restoreContiguousGroupResults(
        op, *reducedResults, resultTypes, types.resultType, shape.groupCount,
        shape.chunksPerGroup, shape.rowLocalSlots1Result, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(getSupport(supports, op, &supportReason))) {
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no layout support: " + supportReason);
    }
    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    FailureOr<GroupReduceLoweringPlan> plan = classifyGroupReduceLoweringPlan(
        sourceVMIType, maskVMIType, resultVMIType,
        op.getNumGroupsAttr().getInt(), &supportReason);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no lowering plan: " + supportReason);
    }

    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        sourceVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group reduce requires num_groups to evenly divide lane count");
    }

    return lowerByPlan(op, *plan, *groupSize, sourceVMIType, resultVMIType,
                       sourceParts, maskParts, resultTypes, rewriter);
  }

private:
  LogicalResult lowerByPlan(
      OpTy op, GroupReduceLoweringPlan plan, int64_t groupSize,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    switch (plan) {
    case GroupReduceLoweringPlan::CompactMaskedRows:
      return lowerCompactRows(op, sourceParts, maskParts, resultTypes,
                               groupSize, rewriter);
    case GroupReduceLoweringPlan::OneBlockVcgadd:
      return lowerOneBlock(op, sourceParts, maskParts, resultTypes, rewriter);
    case GroupReduceLoweringPlan::TwoBlockDeinterleaved2VcgaddVadd:
      return lowerTwoBlock(op, sourceParts, maskParts, resultTypes, numGroups,
                           rewriter);
    case GroupReduceLoweringPlan::FourBlockDeinterleaved4VcgaddTree:
      return lowerFourBlock(op, sourceParts, maskParts, resultTypes, numGroups,
                            rewriter);
    case GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows:
      return lowerFullDeinterleaved2(
          op, sourceVMIType, resultVMIType, sourceParts, maskParts, resultTypes,
          groupSize, rewriter);
    case GroupReduceLoweringPlan::ContiguousVcaddRows:
      return lowerContiguousRows(op, sourceVMIType, resultVMIType, sourceParts,
                                 maskParts, resultTypes, groupSize, rewriter);
    }
    return rewriter.notifyMatchFailure(op, "unknown group_reduce lowering plan");
  }

  FailureOr<VRegType> getRowResultType(VRegType sourceType,
                                       VRegType resultType) const {
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddIOp>) {
      return getVcaddResultType(sourceType);
    }
    return resultType;
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceAddFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceAddFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceAddIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceAddISupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMaxIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMaxISupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMaxFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMaxFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMinFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMinFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMinIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMinISupport(op, reason);
  }

  ;
};


struct OneToNVMIGroupBroadcastOpPattern
    : OneToNOpConversionPattern<VMIGroupBroadcastOp> {
  using OneToNOpConversionPattern<VMIGroupBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerGroupBroadcastParts(
            op, input->sourceParts, input->sourceVMIType,
            input->resultVMIType, input->resultTypes,
            op.getNumGroupsAttr().getInt(), rewriter, results))) {
      return failure();
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// VMI vdhist / vchist → VPTO dhistv2 / chistv2 lowering (shared template)
//===----------------------------------------------------------------------===//

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult lowerHistogramChunk(
    VMIOp op, Value source, Value userMask, int64_t firstLane,
    int64_t lanesPerPart, SmallVectorImpl<Value> &halves,
    ArrayRef<Value> binConsts, VRegType partType,
    OneToNPatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType || !maskType.isB8()) {
    return rewriter.notifyMatchFailure(op, "expected b8 source mask");
  }
  Value chunkMask = userMask;
  int64_t activeLanes = std::min<int64_t>(
      lanesPerPart,
      cast<VMIVRegType>(op.getSource().getType()).getElementCount() - firstLane);
  if (activeLanes < lanesPerPart) {
    FailureOr<Value> validMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeLanes, rewriter);
    FailureOr<Value> allMask =
        createAllTrueMask(op.getLoc(), maskType, rewriter);
    bool failedMask = failed(validMask) || failed(allMask);
    if (failedMask) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize tail-valid b8 mask");
    }
    chunkMask = rewriter
                    .create<PandOp>(op.getLoc(), maskType, chunkMask, *validMask,
                                    *allMask)
                    .getResult();
  }
  for (size_t half = 0; half < halves.size(); ++half) {
    halves[half] = rewriter
                       .create<VPTOHistOp>(op.getLoc(), partType, halves[half],
                                            source, chunkMask, binConsts[half])
                       .getResult();
  }
  return success();
}

template <typename VMIOp>
struct HistogramPhysicalPlan {
  ValueRange sourceParts;
  ValueRange maskParts;
  SmallVector<Value, kPairWidth> halves;
  SmallVector<Value, kPairWidth> binConsts;
  VRegType partType;
  int64_t lanesPerPart;
  size_t halfCount;
};

template <typename VMIOp>
static FailureOr<HistogramPhysicalPlan<VMIOp>> prepareHistogramPhysicalPlan(
    VMIOp op,
    typename OneToNOpConversionPattern<VMIOp>::OpAdaptor adaptor,
    OneToNPatternRewriter &rewriter) {
  ValueRange accParts = adaptor.getAcc();
  ValueRange sourceParts = adaptor.getSource();
  ValueRange maskParts = adaptor.getMask();
  size_t halfCount = accParts.size();
  const bool invalidHalfCount = halfCount != 1 && halfCount != 2;
  if (invalidHalfCount) {
      (void)rewriter.notifyMatchFailure(op,
                                        "expected one or two accumulator parts");
      return failure();
  }
  const bool invalidSourceMaskArity =
      sourceParts.empty() || sourceParts.size() != maskParts.size();
  if (invalidSourceMaskArity) {
      (void)rewriter.notifyMatchFailure(op,
                                        "expected matching source/mask chunks");
      return failure();
  }
  auto partType = dyn_cast<VRegType>(accParts.front().getType());
  if (!partType) {
    (void)rewriter.notifyMatchFailure(op, "expected ui16 acc parts");
    return failure();
  }
  const bool mismatchedSecondHalf =
      halfCount == 2 && accParts[1].getType() != partType;
  if (mismatchedSecondHalf) {
    (void)rewriter.notifyMatchFailure(op, "expected matching ui16 acc parts");
    return failure();
  }
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    (void)rewriter.notifyMatchFailure(op, "failed to compute source lanes");
    return failure();
  }
  Location loc = op.getLoc();
  SmallVector<Value, kPairWidth> binConsts;
  binConsts.push_back(createI32Constant(loc, 0, rewriter));
  if (halfCount == kPairWidth) {
      binConsts.push_back(createI32Constant(loc, 1, rewriter));
  }
  return HistogramPhysicalPlan<VMIOp>{
      sourceParts, maskParts,
      SmallVector<Value, 2>(accParts.begin(), accParts.end()),
      std::move(binConsts), partType, *lanesPerPart, halfCount};
}

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult
lowerVMIHistogramToVPTO(VMIOp op,
                        typename OneToNOpConversionPattern<VMIOp>::OpAdaptor
                            adaptor,
                        TypeConverter *typeConverter,
                        OneToNPatternRewriter &rewriter) {
  FailureOr<HistogramPhysicalPlan<VMIOp>> plan =
      prepareHistogramPhysicalPlan(op, adaptor, rewriter);
  if (failed(plan)) {
    return failure();
  }

  for (size_t index = 0, e = plan->sourceParts.size(); index < e; ++index) {
    if (failed(lowerHistogramChunk<VMIOp, VPTOHistOp>(
            op, plan->sourceParts[index], plan->maskParts[index],
            static_cast<int64_t>(index) * plan->lanesPerPart,
            plan->lanesPerPart, plan->halves, plan->binConsts, plan->partType,
            rewriter))) {
      return failure();
    }
  }

  replaceOpWithFlatConvertedValues(
      rewriter, op,
      SmallVector<Value>(plan->halves.begin(),
                         plan->halves.begin() + plan->halfCount),
      *typeConverter);
  return success();
}

struct OneToNVMIVdhistOpPattern : OneToNOpConversionPattern<VMIVdhistOp> {
  using OneToNOpConversionPattern<VMIVdhistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVdhistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVdhistOp, Dhistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

struct OneToNVMIVchistOpPattern : OneToNOpConversionPattern<VMIVchistOp> {
  using OneToNOpConversionPattern<VMIVchistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVchistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVchistOp, Chistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

template <typename SourceOp, typename ChunkReduceOp, typename CombineOp>
struct OneToNVMIReduceMinMaxOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSequentialReduction(
      SourceOp op, ValueRange sourceParts, ValueRange maskParts,
      VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    Value accumulator =
        rewriter
            .create<ChunkReduceOp>(op.getLoc(), resultType, sourceParts.front(),
                                   maskParts.front())
            .getResult();
    const bool singlePart = sourceParts.size() == 1;
    if (singlePart) {
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{accumulator},
          *this->getTypeConverter());
      return success();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create min/max reduction first-lane mask");
    }
    for (size_t part = 1; part < sourceParts.size(); ++part) {
      Value reduced = rewriter
                          .create<ChunkReduceOp>(op.getLoc(), resultType,
                                                 sourceParts[part],
                                                 maskParts[part])
                          .getResult();
      accumulator = rewriter
                        .create<CombineOp>(op.getLoc(), resultType, reduced,
                                           accumulator, *firstLaneMask)
                        .getResult();
    }
    replaceOpWithFlatConvertedValues(
        rewriter, op, SmallVector<Value>{accumulator},
        *this->getTypeConverter());
    return success();
  }

  FailureOr<std::pair<VRegType, MaskType>> validatePhysicalParts(
      SourceOp op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter *rewriter) const {
    bool invalidArity = sourceParts.empty() || sourceParts.size() != maskParts.size() ||
                        resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter->notifyMatchFailure(
          op, "min/max reduction requires matching source/mask chunks and one result chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter->notifyMatchFailure(
          op, "min/max reduction requires matching physical source/result vregs and one mask");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatch = sourcePart.getType() != resultType;
      if (mismatch) {
        return rewriter->notifyMatchFailure(
            op, "min/max reduction requires every source chunk to match result vreg type");
      }
    }
    for (Value maskPart : maskParts) {
      bool mismatch = maskPart.getType() != maskType;
      if (mismatch) {
        return rewriter->notifyMatchFailure(
            op, "min/max reduction requires every mask chunk to have the same predicate type");
      }
    }
    return std::make_pair(resultType, maskType);
  }

  LogicalResult lowerReduction(SourceOp op, ValueRange sourceParts,
                               ValueRange maskParts, VRegType resultType,
                               MaskType maskType,
                               OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOp>(
        op.getLoc(), sourceParts, maskParts, resultType, rewriter);
    if (succeeded(combined)) {
      Value reduced =
          rewriter
              .create<ChunkReduceOp>(op.getLoc(), resultType, *combined,
                                     maskParts.front())
              .getResult();
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{reduced}, *this->getTypeConverter());
      return success();
    }

    return lowerSequentialReduction(op, sourceParts, maskParts, resultType,
                                    maskType, rewriter);
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<std::pair<VRegType, MaskType>> physical =
        validatePhysicalParts(op, sourceParts, maskParts, resultTypes, &rewriter);
    if (failed(physical)) {
      return failure();
    }
    return lowerReduction(op, sourceParts, maskParts, physical->first,
                          physical->second, rewriter);
  }
};
