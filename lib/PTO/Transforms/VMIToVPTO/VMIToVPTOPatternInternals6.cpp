// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals6.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

template <typename OpT>
struct OneToNVMIExtIOpPattern : OneToNOpConversionPattern<OpT> {
  using OneToNOpConversionPattern<OpT>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildFactorExtensionResult(
      OpT op, Value sourcePart, VRegType resultType, Value mask,
      StringRef part, OneToNPatternRewriter &rewriter) const {
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                        /*rnd=*/nullptr, /*sat=*/nullptr,
                        rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult emitFactorExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes,
      ArrayRef<StringRef> parts, int64_t factor, Value mask,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultVRegTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        FailureOr<Value> result = buildFactorExtensionResult(
            op, sourcePart, resultType, mask, parts[partIndex], rewriter);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildDenseLaneExtensionResult(
      OpT op, Value sourcePart, VRegType resultType, Value mask,
      StringRef part, OneToNPatternRewriter &rewriter) const {
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                        /*rnd=*/nullptr, /*sat=*/nullptr,
                        rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult emitDenseLaneExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      VRegType sourceType, StringRef part,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildDenseLaneExtensionResult(
          op, sourcePart, resultType, *mask, part, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> getExtensionPartPlan(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    if (resultBits == sourceBits * 2 &&
        resultTypes.size() == 2 * sourceParts.size()) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return std::make_pair(ArrayRef<StringRef>(kEvenOddParts), int64_t{2});
    }
    if (resultBits == sourceBits * 4 &&
        resultTypes.size() == 4 * sourceParts.size()) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      return std::make_pair(ArrayRef<StringRef>(kPacked4Parts), int64_t{4});
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported physical integer extension source/result width relation");
  }

  FailureOr<VRegType> validateDenseGroupSlotExtensionResult(
      OpT op, Type resultType, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 0 || resultBits == 0) {
      return rewriter.notifyMatchFailure(
          op, "group-slot integer extension requires positive bit widths");
    }
    FailureOr<int64_t> sourceLanes =
        getDataLanesPerPart(sourceVMIType.getElementType());
    FailureOr<int64_t> resultLanes =
        getDataLanesPerPart(resultVMIType.getElementType());
    bool carrierShapeMismatch =
        failed(sourceLanes) || failed(resultLanes) ||
        *sourceLanes !=
            *resultLanes * static_cast<int64_t>(resultBits / sourceBits);
    auto physicalResultType = dyn_cast<VRegType>(resultType);
    bool invalidResultType =
        !physicalResultType || failed(resultLanes) ||
        physicalResultType.getElementCount() != *resultLanes ||
        pto::getPTOStorageElemBitWidth(physicalResultType.getElementType()) !=
            resultBits;
    if (carrierShapeMismatch || invalidResultType) {
      return rewriter.notifyMatchFailure(
          op, carrierShapeMismatch
                  ? "unsupported dense group-slot integer extension carrier shape"
                  : "unsupported dense group-slot integer extension result type");
    }
    return physicalResultType;
  }

  FailureOr<Value> buildDenseGroupSlotExtensionResult(
      OpT op, Value sourcePart, Type resultType, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, IntegerType resultIntegerType,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> physicalResultType =
        validateDenseGroupSlotExtensionResult(
            op, resultType, sourceVMIType, resultVMIType, sourceBits,
            resultBits, rewriter);
    if (failed(physicalResultType)) {
      return failure();
    }

    Value current = sourcePart;
    unsigned currentBits = sourceBits;
    while (currentBits < resultBits) {
      FailureOr<Value> next = extendDenseGroupSlotCarrier(
          op, current, currentBits, resultIntegerType, rewriter);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      currentBits *= 2;
    }
    FailureOr<Value> result =
        bitcastVReg(op.getLoc(), current, *physicalResultType, rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize dense group-slot unpack result");
    }
    return *result;
  }

  FailureOr<Value> extendDenseGroupSlotCarrier(
      OpT op, Value current, unsigned currentBits,
      IntegerType resultIntegerType,
      OneToNPatternRewriter &rewriter) const {
    unsigned nextBits = currentBits * 2;
    auto nextElementType = IntegerType::get(
        rewriter.getContext(), nextBits, resultIntegerType.getSignedness());
    FailureOr<int64_t> nextLanes = getDataLanesPerPart(nextElementType);
    if (failed(nextLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive dense group-slot unpack result lanes");
    }
    auto nextType =
        VRegType::get(rewriter.getContext(), *nextLanes, nextElementType);
    auto currentType = dyn_cast<VRegType>(current.getType());
    bool unpackLaneMismatch =
        !currentType || currentType.getElementCount() != *nextLanes * 2;
    if (unpackLaneMismatch) {
      return rewriter.notifyMatchFailure(
          op, "dense group-slot unpack source/result lane mismatch");
    }
    Value part = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    if constexpr (std::is_same_v<OpT, VMIExtSIOp>) {
      return rewriter.create<VsunpackOp>(op.getLoc(), nextType, current, part)
          .getResult();
    }
    return rewriter.create<VzunpackOp>(op.getLoc(), nextType, current, part)
        .getResult();
  }

  LogicalResult lowerDenseGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      IntegerType resultIntegerType, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildDenseGroupSlotExtensionResult(
          op, sourcePart, resultType, sourceVMIType, resultVMIType,
          resultIntegerType, sourceBits, resultBits, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildLegacyGroupSlotExtensionResult(
      OpT op, Value sourcePart, Type resultType, VRegType conversionSourceType,
      Value slotMask, StringAttr part, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    bool invalidResultType =
        !resultVRegType ||
        pto::getPTOStorageElemBitWidth(resultVRegType.getElementType()) !=
            resultBits;
    if (invalidResultType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot integer extension result type");
    }
    FailureOr<Value> conversionSource = bitcastVReg(
        op.getLoc(), sourcePart, conversionSourceType, rewriter);
    if (failed(conversionSource)) {
      return rewriter.notifyMatchFailure(
          op, "failed to expose group-slot extension source elements");
    }
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultVRegType, *conversionSource, slotMask,
                        /*rnd=*/nullptr, /*sat=*/nullptr, part)
        .getResult();
  }

  FailureOr<std::tuple<int64_t, VRegType, Value>> prepareLegacyGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        sourceBits == 0 || sourceBits >= resultBits ||
        resultBits % sourceBits != 0 ||
        (resultBits / sourceBits != 2 && resultBits / sourceBits != 4) ||
        (sourceLayout.getSlots() == 8 &&
         sourceLayout.getLaneStride() != resultBits / sourceBits) ||
        resultLayout.getLaneStride() != 1 ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot integer extension shape");
    }
    FailureOr<int64_t> sourceLanes =
        getDataLanesPerPart(sourceVMIType.getElementType());
    if (failed(sourceLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group-slot integer extension source lanes");
    }
    auto conversionSourceType = VRegType::get(
        rewriter.getContext(), *sourceLanes, sourceVMIType.getElementType());
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(conversionSourceType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create group-slot integer extension mask type");
    }
    FailureOr<Value> slotMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *maskType,
        sourceLayout.getSlots() * sourceLayout.getLaneStride(), rewriter);
    if (failed(slotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot integer extension mask");
    }
    return std::make_tuple(static_cast<int64_t>(resultBits / sourceBits), conversionSourceType,
                           *slotMask);
  }

  LogicalResult lowerLegacyGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<std::tuple<int64_t, VRegType, Value>> preparation =
        prepareLegacyGroupSlotExtension(
            op, sourceParts, resultTypes, sourceVMIType, sourceLayout,
            resultLayout, sourceBits, resultBits, rewriter);
    if (failed(preparation)) {
      return failure();
    }
    int64_t widenFactor = std::get<0>(*preparation);
    VRegType conversionSourceType = std::get<1>(*preparation);
    Value slotMask = std::get<2>(*preparation);
    StringAttr part =
        rewriter.getStringAttr(widenFactor == 2 ? "EVEN" : "P0");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildLegacyGroupSlotExtensionResult(
          op, sourcePart, resultType, conversionSourceType, slotMask, part,
          resultBits, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerPhysicalExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes,
      ArrayRef<Type> resultTypes, VRegType sourceType, unsigned sourceBits,
      unsigned resultBits, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool denseLaneExtension =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((resultBits == sourceBits * 2 && sourceLayout.getLaneStride() == 2) ||
         (resultBits == sourceBits * 4 && sourceLayout.getLaneStride() == 4)) &&
        resultTypes.size() == sourceParts.size();
    if (denseLaneExtension) {
      StringRef part = resultBits == sourceBits * 2 ? StringRef("EVEN")
                                                    : StringRef("P0");
      return emitDenseLaneExtension(op, sourceParts, resultVRegTypes,
                                    sourceType, part, rewriter);
    }

    FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> partPlan =
        getExtensionPartPlan(op, sourceParts, resultTypes, sourceBits,
                             resultBits, rewriter);
    if (failed(partPlan)) {
      return failure();
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");
    }

    return emitFactorExtension(op, sourceParts, resultVRegTypes, partPlan->first,
                               partPlan->second,
                               *mask, rewriter);
  }

  FailureOr<SmallVector<VRegType>> collectExtensionResultTypes(
      OpT op, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidType =
          !resultVRegType || !isa<IntegerType>(resultVRegType.getElementType()) ||
          (!resultVRegTypes.empty() &&
           resultVRegType != resultVRegTypes.front());
      if (invalidType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical integer extension result type");
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return resultVRegTypes;
  }

  FailureOr<VRegType> getUniformExtensionSourceType(
      OpT op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(
          op, "integer extension requires at least one physical source chunk");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected physical integer extension source");
    }
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "integer extension source physical parts must have matching "
                "type");
      }
    }
    return sourceType;
  }

  struct ExtensionLoweringInput {
    VMIVRegType sourceVMIType;
    VMIVRegType resultVMIType;
    ValueRange sourceParts;
    SmallVector<Type> resultTypes;
    VRegType sourceType;
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
  };

  FailureOr<ExtensionLoweringInput> getLoweringInput(
      OpT op, typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    FailureOr<VRegType> sourceType =
        getUniformExtensionSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    return ExtensionLoweringInput{
        sourceVMIType, resultVMIType, sourceParts, std::move(*resultTypes),
        *sourceType, sourceVMIType.getLayoutAttr(), resultVMIType.getLayoutAttr()};
  }

  LogicalResult lowerGroupSlotByLayout(
      OpT op, const ExtensionLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input.sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(input.resultVMIType.getElementType());
    auto sourceIntegerType =
        dyn_cast<IntegerType>(input.sourceVMIType.getElementType());
    auto resultIntegerType =
        dyn_cast<IntegerType>(input.resultVMIType.getElementType());
    int64_t slots = input.sourceLayout.getSlots();
    bool denseGroupSlotExtension =
        sourceIntegerType && resultIntegerType &&
        input.sourceLayout.getNumGroups() == input.resultLayout.getNumGroups() &&
        input.sourceLayout.getLaneStride() == 1 &&
        input.resultLayout.getLaneStride() == 1 &&
        input.sourceLayout.getSlots() == input.resultLayout.getSlots() &&
        (slots == 2 || slots == 4 || slots == 8) && sourceBits > 0 &&
        resultBits > sourceBits && resultBits % sourceBits == 0 &&
        (resultBits / sourceBits == 2 || resultBits / sourceBits == 4) &&
        input.sourceParts.size() == input.resultTypes.size();
    if (denseGroupSlotExtension) {
      return lowerDenseGroupSlotExtension(
          op, input.sourceParts, input.resultTypes, input.sourceVMIType,
          input.resultVMIType, resultIntegerType, sourceBits, resultBits,
          rewriter);
    }
    return lowerLegacyGroupSlotExtension(
        op, input.sourceParts, input.resultTypes, input.sourceVMIType,
        input.resultVMIType, input.sourceLayout, input.resultLayout,
        sourceBits, resultBits, rewriter);
  }

  LogicalResult lowerNonGroupSlotByLayout(
      OpT op, const ExtensionLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        collectExtensionResultTypes(op, input.resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input.sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    return lowerPhysicalExtension(
        op, input.sourceParts, *resultVRegTypes, input.resultTypes,
        input.sourceType, sourceBits, resultBits, input.sourceLayout,
        input.resultLayout, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(OpT op,
                  typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<ExtensionLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    bool groupSlotLayouts =
        input->sourceLayout && input->resultLayout &&
        input->sourceLayout.isGroupSlots() && input->resultLayout.isGroupSlots();
    if (groupSlotLayouts) {
      return lowerGroupSlotByLayout(op, *input, rewriter);
    }
    return lowerNonGroupSlotByLayout(op, *input, rewriter);
  }
};

// TruncI lowering support matrix
//
// Keep this comment aligned with both:
//   1. verifySupportedVMIToVPTOOps() diagnostics below, and
//   2. the actual OneToN lowering implemented in this pattern.
//
// Dense logical layouts
//   - deinterleaved factor 2/4 -> contiguous
//     Example: 32 -> 16 or 32 -> 8.
//     Lowering shape: emit vcvt parts EVEN/ODD or P0/P1/P2/P3, then merge
//     physical results when multiple source chunks contribute to one result.
//   - deinterleaved factor 4 -> deinterleaved factor 2
//     Example: 32 -> 16.
//     Lowering shape: emit vcvt EVEN/ODD per source chunk pair.
//   - contiguous lane_stride = 1 -> contiguous lane_stride = 2/4
//     Example: 16 -> 8 lane_stride=2, 32 -> 8 lane_stride=4.
//     Lowering shape: NOSAT keeps/bitcasts the source carrier; SAT emits vcvt
//     into the logical-element vector whose live results occupy the requested
//     strided lanes.
//
// Group-slots logical layouts
//   - slots = 1 preserves the layout for 2x/4x narrowing.
//   - slots = 8 records the 2x/4x narrowing factor as the result lane_stride.
//   - 2x narrowing lowers with part = EVEN; 4x narrowing uses part = P0.
//   - 32-bit integer -> 8-bit integer, slots = 8, result lane_stride = 4
//     Lowering shape: no vcvt; keep/bitcast the 32-bit carrier and let the
//     later store consume it as PK4_B32.
struct OneToNVMITruncIOpPattern : OneToNOpConversionPattern<VMITruncIOp> {
  using OneToNOpConversionPattern<VMITruncIOp>::OneToNOpConversionPattern;

private:
  FailureOr<std::pair<VRegType, VRegType>> getUniformTruncTypes(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    bool emptyPhysicalParts = sourceParts.empty() || resultTypes.empty();
    if (emptyPhysicalParts) {
      return rewriter.notifyMatchFailure(
          op, "trunci requires non-empty physical source and result parts");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    bool invalidTypes =
        !sourceType || !isa<IntegerType>(sourceType.getElementType()) ||
        !resultType || !isa<IntegerType>(resultType.getElementType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    }
    for (Value sourcePart : sourceParts) {
      auto currentType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentType || currentType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "trunci source physical parts must have matching integer type");
      }
    }
    for (Type physicalResultType : resultTypes) {
      auto currentType = dyn_cast<VRegType>(physicalResultType);
      if (!currentType || currentType != resultType) {
        return rewriter.notifyMatchFailure(
            op, "trunci result physical parts must have matching integer type");
      }
    }
    return std::make_pair(sourceType, resultType);
  }

  void finalizeResults(VMITruncIOp op, SmallVectorImpl<Value> &results,
                       bool s32ToS8Alias, ArrayRef<Type> originalResultTypes,
                       OneToNPatternRewriter &rewriter) const {
    if (s32ToS8Alias) {
      for (auto &&[index, result] : llvm::enumerate(results)) {
        result = rewriter
                     .create<VbitcastOp>(op.getLoc(), originalResultTypes[index],
                                         result)
                     .getResult();
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
  }

  FailureOr<Value> lowerGroupSlotTruncPart(
      VMITruncIOp op, Value sourcePart, VRegType sourceType,
      VRegType resultType, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout, unsigned sourceLogicalBits,
      unsigned resultLogicalBits, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    unsigned physicalResultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    bool directCarrier = resultLayout.hasLaneStride() &&
                         resultLayout.getLaneStride() == 4 &&
                         resultLogicalBits == 8 && physicalResultBits == 32;
    if (directCarrier) {
      return lowerGroupSlotDirectCarrier(op, sourcePart, resultType, rewriter);
    }
    bool wideCarrier = resultLayout.hasLaneStride() &&
                       resultLayout.getLaneStride() == 2 &&
                       resultLogicalBits == 16 && physicalResultBits == 32;
    if (wideCarrier) {
      return lowerGroupSlotWideCarrier(op, sourcePart, resultType,
                                       resultVMIType, activeSlotMask, sat,
                                       rewriter);
    }
    bool validNarrowResult = physicalResultBits == 16 || physicalResultBits == 8;
    if (!validNarrowResult) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot trunci physical type");
    }
    StringAttr part = rewriter.getStringAttr(
        sourceLogicalBits == 2 * resultLogicalBits ? "EVEN" : "P0");
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask,
                        nullptr, sat, part)
        .getResult();
  }

  FailureOr<Value> lowerGroupSlotDirectCarrier(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      OneToNPatternRewriter &rewriter) const {
    return sourcePart.getType() == resultType
               ? sourcePart
               : rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
                     .getResult();
  }

  FailureOr<Value> lowerGroupSlotWideCarrier(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      VMIVRegType resultVMIType, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> lanes =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group-slot trunci conversion lanes");
    }
    auto conversionType = VRegType::get(
        rewriter.getContext(), *lanes, resultVMIType.getElementType());
    Value converted = rewriter
                          .create<VcvtOp>(op.getLoc(), conversionType,
                                          sourcePart, activeSlotMask, nullptr,
                                          sat, rewriter.getStringAttr("EVEN"))
                          .getResult();
    FailureOr<Value> carrier =
        bitcastVReg(op.getLoc(), converted, resultType, rewriter);
    if (failed(carrier)) {
      return rewriter.notifyMatchFailure(
          op, "failed to expose group-slot trunci result carrier");
    }
    return *carrier;
  }

  FailureOr<std::pair<bool, bool>> getGroupSlotTruncModes(
      VMITruncIOp op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool supportsDirect =
        (sourceBits == 32 && (resultBits == 16 || resultBits == 8)) ||
        (sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 1);
    bool supportsPacked =
        sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 8 &&
        resultLayout.getSlots() == 8 && resultLayout.hasLaneStride() &&
        resultLayout.getLaneStride() == 2;
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        (!supportsDirect && !supportsPacked) ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      (void)rewriter.notifyMatchFailure(op,
                                        "unsupported group-slot trunci shape");
      return failure();
    }
    return std::make_pair(supportsDirect, supportsPacked);
  }

  FailureOr<SmallVector<Value>> lowerGroupSlotTruncParts(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
      bool supportsPacked, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      bool validPhysicalTypes =
          sourceType &&
          pto::getPTOStorageElemBitWidth(sourceType.getElementType()) ==
              sourceBits &&
          resultType;
      if (!validPhysicalTypes) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported group-slot trunci physical type");
        return failure();
      }
      if (supportsPacked) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                              activeSlotMask, nullptr, sat,
                                              rewriter.getStringAttr("EVEN"))
                              .getResult());
        continue;
      }
      FailureOr<Value> lowered = lowerGroupSlotTruncPart(
          op, sourcePart, sourceType, resultType, resultVMIType, resultLayout,
          sourceBits, resultBits, activeSlotMask, sat, rewriter);
      if (failed(lowered)) {
        return failure();
      }
      results.push_back(*lowered);
    }
    return results;
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncIOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      ArrayRef<Type> resultTypes) const {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<std::pair<bool, bool>> modes = getGroupSlotTruncModes(
        op, sourceVMIType, resultVMIType, sourceLayout, resultLayout,
        sourceParts, resultTypes, rewriter);
    if (failed(modes)) {
      return failure();
    }
    unsigned sourceLogicalBits =
        pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    unsigned resultLogicalBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool supportsPacked = modes->second;

    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    const char *activeSlotPattern =
        sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
    StringRef activeSlotGranularity = sourceLogicalBits == 16 ? "b16" : "b32";
    FailureOr<Value> activeSlotMask = createPrefixMask(
        op.getLoc(), MaskType::get(rewriter.getContext(), activeSlotGranularity),
        activeSlotPattern, rewriter);
    if (failed(activeSlotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot trunci active slot mask");
    }
    FailureOr<SmallVector<Value>> results = lowerGroupSlotTruncParts(
        op, sourceParts, resultTypes, sourceVMIType, resultVMIType,
        resultLayout, sourceLogicalBits, resultLogicalBits, supportsPacked,
        *activeSlotMask, sat, rewriter);
    if (failed(results)) {
      return failure();
    }
    finalizeResults(op, *results, false, resultTypes, rewriter);
    return success();
  }

  LogicalResult lowerDenseLaneStrideTrunc(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      int64_t factor, StringAttr sat, bool s32ToS8Alias,
      ArrayRef<Type> originalResultTypes,
      OneToNPatternRewriter &rewriter) const {
    StringAttr part = rewriter.getStringAttr(factor == 2 ? "EVEN" : "P0");
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported dense trunci source type");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported dense trunci result type");
      }
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultVRegType,
                                            sourcePart, *sourceMask,
                                            /*rnd=*/nullptr, sat, part)
                            .getResult());
    }
    finalizeResults(op, results, s32ToS8Alias, originalResultTypes, rewriter);
    return success();
  }

  LogicalResult lowerNoSatDenseCarrier(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result =
          bitcastVReg(op.getLoc(), sourcePart, resultType, rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to forward NOSAT trunci carrier");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildFactorTruncResult(
      VMITruncIOp op, ValueRange sourceParts, Type resultType,
      int64_t resultIndex, int64_t factor, ArrayRef<StringRef> parts,
      StringAttr sat, Value sourceMask, Value resultMask,
      OneToNPatternRewriter &rewriter) const {
    bool invalidFactor =
        factor <= 0 || static_cast<size_t>(factor) > parts.size();
    if (invalidFactor) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci conversion factor");
    }
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    if (!resultVRegType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci result type");
    }

    SmallVector<Value> partials;
    partials.reserve(static_cast<size_t>(factor));
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      int64_t sourceIndex = resultIndex * factor + partIndex;
      if (sourceIndex < 0 ||
          sourceIndex >= static_cast<int64_t>(sourceParts.size())) {
        return rewriter.notifyMatchFailure(
            op, "trunci source part index exceeds physical arity");
      }
      partials.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultVRegType,
                              sourceParts[sourceIndex], sourceMask, nullptr, sat,
                              rewriter.getStringAttr(parts[partIndex]))
              .getResult());
    }

    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultVRegType, merged, partial,
                                  resultMask)
                   .getResult();
    }
    return merged;
  }

  LogicalResult lowerFactorTrunc(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      ArrayRef<StringRef> parts, int64_t factor, StringAttr sat,
      bool s32ToS8Alias, ArrayRef<Type> originalResultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!sourceType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    bool failedMasks = failed(sourceMask) || failed(resultMask);
    if (failedMasks) {
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t resultIndex = 0;
         resultIndex < static_cast<int64_t>(resultTypes.size()); ++resultIndex) {
      FailureOr<Value> result = buildFactorTruncResult(
          op, sourceParts, resultTypes[resultIndex], resultIndex, factor, parts,
          sat, *sourceMask, *resultMask, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    finalizeResults(op, results, s32ToS8Alias, originalResultTypes, rewriter);
    return success();
  }

  struct TruncIPhysicalPlan {
    VRegType sourceType;
    VRegType resultType;
    int64_t factor;
    bool denseLaneStride;
    StringAttr saturate;
  };

  FailureOr<TruncIPhysicalPlan> buildPhysicalPlan(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<std::pair<VRegType, VRegType>> uniformTypes =
        getUniformTruncTypes(op, sourceParts, resultTypes, rewriter);
    if (failed(uniformTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(uniformTypes->first.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(uniformTypes->second.getElementType());
    bool invalidWidth = sourceBits == 0 || resultBits == 0 ||
                        sourceBits % resultBits != 0;
    if (invalidWidth) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result width relation");
    }
    int64_t factor = sourceBits / resultBits;
    bool denseLaneStride =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == factor &&
        sourceParts.size() == resultTypes.size();
    bool unsupportedDenseFactor = denseLaneStride && factor != 2 && factor != 4;
    if (unsupportedDenseFactor) {
      return rewriter.notifyMatchFailure(
          op, "unsupported dense lane_stride trunci result layout");
    }
    return TruncIPhysicalPlan{uniformTypes->first, uniformTypes->second, factor,
                              denseLaneStride,
                              op->getAttrOfType<StringAttr>("saturate")};
  }

  struct TruncIAliasPlan {
    SmallVector<Value> sourceParts;
    SmallVector<Type> resultTypes;
    SmallVector<Type> originalResultTypes;
    bool requiresResultBitcast = false;
  };

  TruncIAliasPlan materializeS32ToS8Alias(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      const TruncIPhysicalPlan &physicalPlan,
      OneToNPatternRewriter &rewriter) const {
    TruncIAliasPlan aliasPlan;
    bool requiresAlias =
        physicalPlan.sourceType.getElementType().isSignedInteger(32) &&
        physicalPlan.resultType.getElementType().isSignedInteger(8);
    if (!requiresAlias) {
      aliasPlan.sourceParts.assign(sourceParts.begin(), sourceParts.end());
      aliasPlan.resultTypes.assign(resultTypes.begin(), resultTypes.end());
      return aliasPlan;
    }
    auto u32ElemTy = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto u8ElemTy = rewriter.getIntegerType(8, /*isSigned=*/false);
    aliasPlan.originalResultTypes.assign(resultTypes.begin(), resultTypes.end());
    aliasPlan.sourceParts.reserve(sourceParts.size());
    for (Value sourcePart : sourceParts) {
      auto sourcePartType = cast<VRegType>(sourcePart.getType());
      aliasPlan.sourceParts.push_back(
          rewriter.create<VbitcastOp>(
              op.getLoc(), VRegType::get(sourcePartType.getContext(),
                                          sourcePartType.getElementCount(),
                                          u32ElemTy),
              sourcePart).getResult());
    }
    aliasPlan.resultTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = cast<VRegType>(resultType);
      aliasPlan.resultTypes.push_back(VRegType::get(
          resultVRegType.getContext(), resultVRegType.getElementCount(),
          u8ElemTy));
    }
    aliasPlan.requiresResultBitcast = true;
    return aliasPlan;
  }

  FailureOr<ArrayRef<StringRef>> getFactorTruncParts(
      VMITruncIOp op, int64_t factor, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    bool invalidFactorArity =
        (factor != 2 && factor != 4) ||
        sourcePartCount != resultPartCount * static_cast<size_t>(factor);
    if (invalidFactorArity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result arity relation");
    }
    if (factor == 2) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return ArrayRef<StringRef>(kEvenOddParts);
    }
    static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
    return ArrayRef<StringRef>(kPacked4Parts);
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncIOp op, ValueRange sourceParts, SmallVector<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncIPhysicalPlan> physicalPlan = buildPhysicalPlan(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(physicalPlan)) {
      return failure();
    }
    bool useNoSatCarrier =
        physicalPlan->denseLaneStride && physicalPlan->saturate &&
        physicalPlan->saturate.getValue() == "NOSAT";
    if (useNoSatCarrier) {
      return lowerNoSatDenseCarrier(op, sourceParts, resultTypes, rewriter);
    }
    TruncIAliasPlan aliasPlan = materializeS32ToS8Alias(
        op, sourceParts, resultTypes, *physicalPlan, rewriter);
    if (physicalPlan->denseLaneStride) {
      return lowerDenseLaneStrideTrunc(
          op, aliasPlan.sourceParts, aliasPlan.resultTypes,
          physicalPlan->factor, physicalPlan->saturate,
          aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes,
          rewriter);
    }
    FailureOr<ArrayRef<StringRef>> parts = getFactorTruncParts(
        op, physicalPlan->factor, aliasPlan.sourceParts.size(),
        aliasPlan.resultTypes.size(), rewriter);
    if (failed(parts)) {
      return failure();
    }
    return lowerFactorTrunc(
        op, aliasPlan.sourceParts, aliasPlan.resultTypes, *parts,
        physicalPlan->factor, physicalPlan->saturate,
        aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes,
        rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMITruncIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
    bool groupSlotLayouts =
        sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots();
    if (groupSlotLayouts) {
      return lowerGroupSlotTrunc(op, adaptor, rewriter, input->sourceVMIType,
                                 input->resultVMIType, sourceLayout,
                                 resultLayout, input->resultTypes);
    }

    return lowerNonGroupSlotTrunc(op, input->sourceParts,
                                  std::move(input->resultTypes), sourceLayout,
                                  resultLayout, rewriter);
  }
};

static LogicalResult lowerSameWidthFpToInt(
    Operation *op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
    StringAttr rnd, StringAttr sat, StringRef arityDiagnostic,
    StringRef maskDiagnostic, TypeConverter *typeConverter,
    OneToNPatternRewriter &rewriter) {
  bool invalidArity = sourceParts.size() != resultTypes.size();
  if (invalidArity) {
    return rewriter.notifyMatchFailure(op, arityDiagnostic);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [sourcePart, resultType] :
       llvm::zip_equal(sourceParts, resultTypes)) {
    FailureOr<Value> mask = createAllTrueMaskForVReg(
        op->getLoc(), cast<VRegType>(sourcePart.getType()), rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, maskDiagnostic);
    }
    results.push_back(
        rewriter
            .create<VcvtOp>(op->getLoc(), resultType, sourcePart, *mask, rnd,
                            sat, /*part=*/nullptr)
            .getResult());
  }
  replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
  return success();
}

static FailureOr<Value> buildNarrowFpToIntResult(
    Operation *op, ValueRange sourceParts, VRegType resultType,
    int64_t chunkIndex, int64_t sourceFactor, int64_t partStride,
    ArrayRef<StringRef> parts, StringAttr rnd, StringAttr sat,
    Value sourceMask, OneToNPatternRewriter &rewriter) {
  if (sourceFactor <= 0) {
    return rewriter.notifyMatchFailure(
        op, "narrow fp-to-int requires a positive source factor");
  }
  int64_t safeSourceFactor = sourceFactor;
  FailureOr<Value> resultMask =
      createAllTrueMaskForVReg(op->getLoc(), resultType, rewriter);
  if (failed(resultMask)) {
    return failure();
  }

  SmallVector<Value> partials;
  partials.reserve(safeSourceFactor);
  int64_t resultCount = sourceParts.size() / safeSourceFactor;
  for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
    Value sourcePart = sourceParts[partIndex * resultCount + chunkIndex];
    partials.push_back(
        rewriter
            .create<VcvtOp>(op->getLoc(), resultType, sourcePart, sourceMask,
                            rnd, sat, rewriter.getStringAttr(
                                          parts[partIndex * partStride]))
            .getResult());
  }

  Value merged = partials.front();
  for (Value partial : llvm::drop_begin(partials)) {
    merged = rewriter
                 .create<VorOp>(op->getLoc(), resultType, merged, partial,
                                *resultMask)
                 .getResult();
  }
  return merged;
}

static LogicalResult lowerNarrowFpToInt(
    Operation *op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
    int64_t sourceFactor, int64_t partStride, ArrayRef<StringRef> parts,
    StringAttr rnd, StringAttr sat,
    StringRef sourceMaskDiagnostic, StringRef resultMaskDiagnostic,
    TypeConverter *typeConverter, OneToNPatternRewriter &rewriter) {
  if (sourceFactor <= 0 || partStride <= 0 ||
      (sourceFactor - 1) * partStride >= static_cast<int64_t>(parts.size()) ||
      sourceParts.size() !=
                              static_cast<size_t>(sourceFactor) *
                                  resultTypes.size()) {
    return rewriter.notifyMatchFailure(
        op, "narrow fp-to-int source arity does not match conversion factor");
  }

  auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!sourceType) {
    return rewriter.notifyMatchFailure(op, "expected physical fp source type");
  }
  FailureOr<Value> sourceMask =
      createAllTrueMaskForVReg(op->getLoc(), sourceType, rewriter);
  if (failed(sourceMask)) {
    return rewriter.notifyMatchFailure(op, sourceMaskDiagnostic);
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<Value> result = buildNarrowFpToIntResult(
        op, sourceParts, resultType, chunkIndex, sourceFactor, partStride,
        parts, rnd, sat, *sourceMask, rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(op, resultMaskDiagnostic);
    }
    results.push_back(*result);
  }

  replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
  return success();
}

static int64_t getElementDeinterleaveFactor(VMILayoutAttr layout) {
  bool contiguous = layout && layout.isContiguous() && layout.getLaneStride() == 1;
  if (contiguous) {
    return 1;
  }
  bool deinterleaved =
      layout && layout.isDeinterleaved() && layout.getLaneStride() == 1;
  if (deinterleaved) {
    return layout.getFactor();
  }
  return 0;
}

static LogicalResult lowerWidenFpToInt(
    Operation *op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
    ArrayRef<StringRef> parts, StringAttr rnd, StringAttr sat,
    StringRef arityDiagnostic, StringRef maskDiagnostic,
    TypeConverter *typeConverter, OneToNPatternRewriter &rewriter) {
  bool invalidArity =
      parts.empty() || resultTypes.size() != parts.size() * sourceParts.size();
  if (invalidArity) {
    return rewriter.notifyMatchFailure(op, arityDiagnostic);
  }
  auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!sourceType) {
    return rewriter.notifyMatchFailure(op, maskDiagnostic);
  }
  FailureOr<Value> mask =
      createAllTrueMaskForVReg(op->getLoc(), sourceType, rewriter);
  if (failed(mask)) {
    return rewriter.notifyMatchFailure(op, maskDiagnostic);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (size_t partIndex = 0; partIndex < parts.size(); ++partIndex) {
    for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
      VRegType resultType =
          resultTypes[partIndex * sourceParts.size() + chunkIndex];
      results.push_back(
          rewriter
              .create<VcvtOp>(op->getLoc(), resultType, sourcePart, *mask, rnd,
                              sat, rewriter.getStringAttr(parts[partIndex]))
              .getResult());
    }
  }
  replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
  return success();
}

template <typename OpTy>
static FailureOr<VRegType> validateFpToIntSourceParts(
    OpTy op, ValueRange sourceParts, StringRef emptyDiagnostic,
    StringRef expectedTypeDiagnostic, StringRef mismatchDiagnostic,
    OneToNPatternRewriter &rewriter) {
  if (sourceParts.empty()) {
    return rewriter.notifyMatchFailure(op, emptyDiagnostic);
  }
  auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!sourceType) {
    return rewriter.notifyMatchFailure(op, expectedTypeDiagnostic);
  }
  for (Value sourcePart : sourceParts) {
    auto currentType = dyn_cast<VRegType>(sourcePart.getType());
    bool mismatchedType = !currentType || currentType != sourceType;
    if (mismatchedType) {
      return rewriter.notifyMatchFailure(op, mismatchDiagnostic);
    }
  }
  return sourceType;
}

template <typename OpTy>
static FailureOr<SmallVector<VRegType>> validateFpToIntResultParts(
    OpTy op, TypeRange resultTypes, StringRef emptyDiagnostic,
    StringRef typeDiagnostic, OneToNPatternRewriter &rewriter) {
  if (resultTypes.empty()) {
    return rewriter.notifyMatchFailure(op, emptyDiagnostic);
  }
  SmallVector<VRegType> resultVRegTypes;
  resultVRegTypes.reserve(resultTypes.size());
  for (Type physicalResultType : resultTypes) {
    auto resultType = dyn_cast<VRegType>(physicalResultType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, typeDiagnostic);
    }
    resultVRegTypes.push_back(resultType);
  }
  return resultVRegTypes;
}

struct FpToIntPartValidation {
  VRegType sourceType;
  SmallVector<VRegType> resultVRegTypes;
};

template <typename OpTy>
static FailureOr<FpToIntPartValidation> validateFpToIntConversionParts(
    OpTy op, ValueRange sourceParts, TypeRange resultTypes,
    OneToNPatternRewriter &rewriter, StringRef emptySourceDiagnostic,
    StringRef sourceTypeDiagnostic, StringRef sourceMismatchDiagnostic,
    StringRef emptyResultDiagnostic, StringRef resultTypeDiagnostic) {
  FailureOr<VRegType> sourceType = validateFpToIntSourceParts(
      op, sourceParts, emptySourceDiagnostic, sourceTypeDiagnostic,
      sourceMismatchDiagnostic, rewriter);
  if (failed(sourceType)) {
    return failure();
  }
  FailureOr<SmallVector<VRegType>> resultVRegTypes = validateFpToIntResultParts(
      op, resultTypes, emptyResultDiagnostic, resultTypeDiagnostic, rewriter);
  if (failed(resultVRegTypes)) {
    return failure();
  }
  return FpToIntPartValidation{*sourceType, std::move(*resultVRegTypes)};
}

// Shared prechecks for narrow fp-to-int lowering: derives the source factor
// from the element bit widths, validates the physical result lane stride, and
// checks the source chunk arity against the expected result count.
struct NarrowFpToIntPlan {
  int64_t factor;
  int64_t resultLaneStride;
  int64_t sourceFactor;
};

template <typename OpTy>
static FailureOr<NarrowFpToIntPlan> buildNarrowFpToIntPlan(
    OpTy op, ValueRange sourceParts, TypeRange physicalResultTypes,
    VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
    OneToNPatternRewriter &rewriter, StringRef positiveWidthDiagnostic,
    StringRef unsupportedLaneStrideDiagnostic,
    StringRef invalidSourceArityDiagnostic) {
  if (resultBits == 0) {
    return rewriter.notifyMatchFailure(op, positiveWidthDiagnostic);
  }
  int64_t factor = sourceBits / resultBits;
  VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
  int64_t resultLaneStride = resultLayout && resultLayout.isContiguous()
                                 ? resultLayout.getLaneStride()
                                 : 1;
  bool invalidResultLaneStride =
      resultLaneStride <= 0 || factor % resultLaneStride != 0;
  if (invalidResultLaneStride) {
    return rewriter.notifyMatchFailure(op, unsupportedLaneStrideDiagnostic);
  }
  int64_t sourceFactor = factor / resultLaneStride;
  bool invalidSourceArity =
      sourceParts.size() != sourceFactor * physicalResultTypes.size();
  if (invalidSourceArity) {
    return rewriter.notifyMatchFailure(op, invalidSourceArityDiagnostic);
  }
  return NarrowFpToIntPlan{factor, resultLaneStride, sourceFactor};
}

struct OneToNVMIFPToSIOpPattern : OneToNOpConversionPattern<VMIFPToSIOp> {
  using OneToNOpConversionPattern<VMIFPToSIOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerDenseWiden(
      VMIFPToSIOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, VRegType sourceType, unsigned sourceBits,
      StringAttr rnd, StringAttr sat, OneToNPatternRewriter &rewriter) const {
    StringRef part = sourceBits == 16 ? StringRef("EVEN") : StringRef("P0");
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build fptosi widen 1:1 mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask, rnd,
                              sat, rewriter.getStringAttr(part))
              .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerWiden(
      VMIFPToSIOp op, ValueRange sourceParts,
      ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VRegType sourceType, unsigned sourceBits, StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool denseOneToOne =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((sourceBits == 16 && sourceLayout.getLaneStride() == 2) ||
         (sourceBits == 8 && sourceLayout.getLaneStride() == 4)) &&
        physicalResultTypes.size() == sourceParts.size();
    if (denseOneToOne) {
      return lowerDenseWiden(op, sourceParts, resultTypes, sourceType,
                             sourceBits, rnd, sat, rewriter);
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    return lowerWidenFpToInt(
        op, sourceParts, resultTypes, kEvenOddParts, rnd, sat,
        "widen fptosi requires result arity = 2 × source arity",
        "failed to build fptosi widen mask", this->getTypeConverter(),
        rewriter);
  }

  LogicalResult lowerNarrow(
      VMIFPToSIOp op, ValueRange sourceParts,
      ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
      VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
      StringAttr rnd, StringAttr sat, OneToNPatternRewriter &rewriter) const {
    FailureOr<NarrowFpToIntPlan> plan = buildNarrowFpToIntPlan(
        op, sourceParts, physicalResultTypes, resultVMIType, sourceBits,
        resultBits, rewriter, "narrow fptosi requires positive result bit width",
        "narrow fptosi: unsupported result lane stride",
        "narrow fptosi: source arity != sourceFactor × result arity");
    if (failed(plan)) {
      return failure();
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
    ArrayRef<StringRef> parts = plan->factor == 2
                                    ? ArrayRef<StringRef>(kEvenOddParts)
                                    : ArrayRef<StringRef>(kPacked4Parts);
    return lowerNarrowFpToInt(
        op, sourceParts, resultTypes, plan->sourceFactor,
        plan->resultLaneStride, parts, rnd, sat,
        "failed to build fptosi source mask",
        "failed to build narrow fptosi result mask", this->getTypeConverter(),
        rewriter);
  }

  LogicalResult lowerConversion(
      VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    Type sourceElementType = sourceVMIType.getElementType();
    Type resultElementType = resultVMIType.getElementType();
    auto contract =
        lookupVMIFpToSiContract(sourceElementType, resultElementType);
    if (!contract) {
      return rewriter.notifyMatchFailure(
          op, "unsupported fp-to-si conversion element type pair");
    }
    FailureOr<FpToIntPartValidation> validation =
        validateFpToIntConversionParts(
            op, sourceParts, resultTypes, rewriter,
            "fptosi requires at least one physical source chunk",
            "expected physical fptosi source type",
            "fptosi source physical parts must have matching type",
            "fptosi requires at least one physical result chunk",
            "unsupported physical fptosi result type");
    if (failed(validation)) {
      return failure();
    }

    StringAttr rnd = op->getAttrOfType<StringAttr>("rounding");
    if (!rnd) {
      rnd = rewriter.getStringAttr("R");
    }
    StringAttr sat = contract->requiresSat
                         ? op->getAttrOfType<StringAttr>("saturate")
                         : nullptr;
    if (!contract->requiresPart) {
      return lowerSameWidthFpToInt(
          op, sourceParts, validation->resultVRegTypes, rnd, sat,
          "same-width fptosi requires matching physical arity",
          "failed to build fptosi mask", this->getTypeConverter(), rewriter);
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceElementType);
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultElementType);
    if (resultBits > sourceBits) {
      return lowerWiden(op, sourceParts, resultTypes,
                        validation->resultVRegTypes, sourceVMIType,
                        resultVMIType, validation->sourceType, sourceBits, rnd,
                        sat, rewriter);
    }
    return lowerNarrow(op, sourceParts, resultTypes,
                       validation->resultVRegTypes, resultVMIType, sourceBits,
                       resultBits, rnd, sat, rewriter);
  }

  LogicalResult lowerWithResultTypes(
      VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    return lowerConversion(op, sourceParts, resultTypes, sourceVMIType,
                           resultVMIType, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMIFPToSIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          return lowerWithResultTypes(op, sourceParts, resultTypes, rewriter);
        });
  }
};

struct OneToNVMIFPToUIOpPattern
    : OneToNOpConversionPattern<VMIFPToUIOp> {
  using OneToNOpConversionPattern<VMIFPToUIOp>::OneToNOpConversionPattern;

private:
  struct FPToUILoweringInput {
    ValueRange sourceParts;
    ArrayRef<Type> resultTypes;
    VMIVRegType sourceVMIType;
    VMIVRegType resultVMIType;
    VRegType sourceType;
    SmallVector<VRegType> resultVRegTypes;
    VMIFpToUiContract contract;
    StringAttr rounding;
    StringAttr saturate;
  };

  LogicalResult lowerWiden(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes,
      ArrayRef<VRegType> resultTypes, StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    bool invalidWidenArity =
        physicalResultTypes.size() != 2 * sourceParts.size();
    if (invalidWidenArity) {
      return rewriter.notifyMatchFailure(
          op, "widen fptoui requires result arity = 2 × source arity");
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    return lowerWidenFpToInt(
        op, sourceParts, resultTypes, kEvenOddParts, rnd, sat,
        "widen fptoui requires result arity = 2 × source arity",
        "failed to build fptoui widen mask", this->getTypeConverter(),
        rewriter);
  }

  LogicalResult lowerNarrow(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes,
      ArrayRef<VRegType> resultTypes, VMIVRegType resultVMIType,
      unsigned sourceBits, unsigned resultBits, StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<NarrowFpToIntPlan> plan = buildNarrowFpToIntPlan(
        op, sourceParts, physicalResultTypes, resultVMIType, sourceBits,
        resultBits, rewriter, "narrow fptoui requires positive result bit width",
        "narrow fptoui: unsupported result lane stride",
        "narrow fptoui: source arity != sourceFactor × result arity");
    if (failed(plan)) {
      return failure();
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    return lowerNarrowFpToInt(
        op, sourceParts, resultTypes, plan->sourceFactor,
        plan->resultLaneStride, kEvenOddParts, rnd, sat,
        "failed to build fptoui source mask",
        "failed to build narrow fptoui result mask", this->getTypeConverter(),
        rewriter);
  }

  FailureOr<FPToUILoweringInput> getLoweringInput(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    auto contract = lookupVMIFpToUIContract(sourceVMIType.getElementType(),
                                             resultVMIType.getElementType());
    if (!contract) {
      return rewriter.notifyMatchFailure(
          op, "unsupported fp-to-ui conversion element type pair");
    }
    FailureOr<FpToIntPartValidation> validation =
        validateFpToIntConversionParts(
            op, sourceParts, resultTypes, rewriter,
            "fptoui requires at least one physical source chunk",
            "expected physical fptoui source type",
            "fptoui source physical parts must have matching type",
            "fptoui requires at least one physical result chunk",
            "unsupported physical fptoui result type");
    if (failed(validation)) {
      return failure();
    }
    StringAttr rounding = op->getAttrOfType<StringAttr>("rounding");
    if (!rounding) {
      rounding = rewriter.getStringAttr("R");
    }
    StringAttr saturate = contract->requiresSat
                              ? op->getAttrOfType<StringAttr>("saturate")
                              : nullptr;
    return FPToUILoweringInput{sourceParts, resultTypes, sourceVMIType,
                               resultVMIType, validation->sourceType,
                               std::move(validation->resultVRegTypes), *contract,
                               rounding, saturate};
  }

  LogicalResult lowerConversion(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<FPToUILoweringInput> input = getLoweringInput(
        op, sourceParts, resultTypes, sourceVMIType, resultVMIType, rewriter);
    if (failed(input)) {
      return failure();
    }
    if (!input->contract.requiresPart) {
      return lowerSameWidthFpToInt(
          op, input->sourceParts, input->resultVRegTypes, input->rounding,
          input->saturate,
          "same-width fptoui requires matching physical arity",
          "failed to build fptoui mask", this->getTypeConverter(), rewriter);
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input->sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(input->resultVMIType.getElementType());
    if (resultBits > sourceBits) {
      return lowerWiden(op, input->sourceParts, input->resultTypes,
                        input->resultVRegTypes, input->rounding,
                        input->saturate, rewriter);
    }
    if (resultBits < sourceBits) {
      return lowerNarrow(op, input->sourceParts, input->resultTypes,
                         input->resultVRegTypes, input->resultVMIType,
                         sourceBits, resultBits, input->rounding,
                         input->saturate, rewriter);
    }
    return failure();
  }

  LogicalResult lowerWithResultTypes(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    return lowerConversion(op, sourceParts, resultTypes, sourceVMIType,
                           resultVMIType, rewriter);
  }

  LogicalResult lowerConversionWithResolvedTypes(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    return lowerWithResultTypes(op, sourceParts, resultTypes, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMIFPToUIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          return lowerConversionWithResolvedTypes(op, sourceParts, resultTypes,
                                                  rewriter);
        });
  }
};

struct OneToNVMISIToFPOpPattern : OneToNOpConversionPattern<VMISIToFPOp> {
  using OneToNOpConversionPattern<VMISIToFPOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSameWidth(
      VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      Value mask, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "si32->f32 requires matching physical arity");
    }
    StringAttr rnd = rewriter.getStringAttr("R");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask, rnd,
                              nullptr, nullptr)
              .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerWiden(
      VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      Value mask, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = resultTypes.size() != 2 * sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "si8->f16 requires result arity = 2 x source arity");
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < 2; ++partIndex) {
      for (auto [chunkIndex, sourcePart] :
           llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                                nullptr, nullptr,
                                rewriter.getStringAttr(kEvenOddParts[partIndex]))
                .getResult());
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerConversion(
      VMISIToFPOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, Value mask, unsigned sourceBits,
      unsigned resultBits, OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 32 && resultBits == 32) {
      return lowerSameWidth(op, sourceParts, resultTypes, mask, rewriter);
    } else if (sourceBits == 8 && resultBits == 16) {
      return lowerWiden(op, sourceParts, resultTypes, mask, rewriter);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported sitofp source/result width relation");
    }
  }

  FailureOr<SmallVector<VRegType>> collectResultTypes(
      VMISIToFPOp op, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (resultTypes.empty()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical sitofp result type");
    }
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool mismatchedType =
          !resultVRegType ||
          (!resultVRegTypes.empty() &&
           resultVRegType != resultVRegTypes.front());
      if (mismatchedType) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported physical sitofp result type");
        return failure();
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return resultVRegTypes;
  }

  FailureOr<VRegType> collectSourceType(
      VMISIToFPOp op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "sitofp requires integer source chunks");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    bool invalidSourceType =
        !sourceType || !isa<IntegerType>(sourceType.getElementType());
    if (invalidSourceType) {
      return rewriter.notifyMatchFailure(op,
                                         "sitofp requires integer source chunks");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatchedSourceType = sourcePart.getType() != sourceType;
      if (mismatchedSourceType) {
        return rewriter.notifyMatchFailure(
            op, "sitofp requires integer source chunks");
      }
    }
    return sourceType;
  }

  LogicalResult lowerPhysicalConversion(
      VMISIToFPOp op, ValueRange sourceParts, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> sourceType =
        collectSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        collectResultTypes(op, resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType->getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), *sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
    }
    return lowerConversion(op, sourceParts, *resultVRegTypes, *mask,
                           sourceBits, resultBits, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMISIToFPOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    return lowerPhysicalConversion(op, sourceParts, resultTypes, rewriter);
  }
};

struct OneToNVMIBitcastOpPattern : OneToNOpConversionPattern<VMIBitcastOp> {
  using OneToNOpConversionPattern<VMIBitcastOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildBitcastPart(
      VMIBitcastOp op, Value sourcePart, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    bool invalidPartTypes = !isa<VRegType>(sourcePart.getType()) ||
                            !isa<VRegType>(resultType);
    if (invalidPartTypes) {
      return rewriter.notifyMatchFailure(
          op, "physical bitcast part type mismatch");
    }
    return rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
        .getResult();
  }

  LogicalResult lowerParts(VMIBitcastOp op, ValueRange sourceParts,
                           ArrayRef<Type> resultTypes,
                           OneToNPatternRewriter &rewriter) const {
    bool arityMismatch = sourceParts.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op, "physical bitcast arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "physical bitcast arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return buildBitcastPart(op, sourceParts[index], resultType, rewriter);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIBitcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerParts(op, sourceParts, resultTypes, rewriter);
  }
};


