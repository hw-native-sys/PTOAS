// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals6b.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//
static LogicalResult lowerSameWidthFpToInt(
    Operation* op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks, StringAttr rnd,
    StringAttr sat, StringRef arityDiagnostic, StringRef maskDiagnostic, TypeConverter* typeConverter,
    OneToNPatternRewriter& rewriter)
{
    bool invalidArity = sourceParts.size() != resultTypes.size();
    if (invalidArity) {
        return rewriter.notifyMatchFailure(op, arityDiagnostic);
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
        auto [sourcePart, resultType] = sourceAndResult;
        // The unmasked sub-path keeps its all-true seed; a masked cast reads the
        // authored predicate of this source chunk instead (issue #1671).
        Value partMask;
        if (masks.empty()) {
            FailureOr<Value> mask =
                createAllTrueMaskForVReg(op->getLoc(), cast<VRegType>(sourcePart.getType()), rewriter);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, maskDiagnostic);
            }
            partMask = *mask;
        } else {
            FailureOr<Value> mask = getConversionMaskForChunk(masks, index, Value());
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, maskDiagnostic);
            }
            partMask = *mask;
        }
        results.push_back(
            rewriter.create<VcvtOp>(op->getLoc(), resultType, sourcePart, partMask, rnd, sat, /*part=*/nullptr)
                .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
    return success();
}

static FailureOr<Value> buildNarrowFpToIntResult(
    Operation* op, ValueRange sourceParts, VRegType resultType, int64_t chunkIndex, int64_t sourceFactor,
    int64_t partStride, ArrayRef<StringRef> parts, StringAttr rnd, StringAttr sat, Value sourceMask, ValueRange masks,
    OneToNPatternRewriter& rewriter)
{
    if (sourceFactor <= 0) {
        return rewriter.notifyMatchFailure(op, "narrow fp-to-int requires a positive source factor");
    }
    int64_t safeSourceFactor = sourceFactor;
    FailureOr<Value> resultMask = createAllTrueMaskForVReg(op->getLoc(), resultType, rewriter);
    if (failed(resultMask)) {
        return failure();
    }

    SmallVector<Value> partials;
    partials.reserve(safeSourceFactor);
    int64_t resultCount = sourceParts.size() / safeSourceFactor;
    for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
        int64_t sourceIndex = partIndex * resultCount + chunkIndex;
        Value sourcePart = sourceParts[sourceIndex];
        // Each partial converts its own source chunk, so a masked cast carries that
        // chunk's predicate (issue #1671).
        Value partMask = sourceMask;
        if (!masks.empty()) {
            FailureOr<Value> mask = getConversionMaskForChunk(masks, sourceIndex, Value());
            if (failed(mask)) {
                return failure();
            }
            partMask = *mask;
        }
        partials.push_back(rewriter
                               .create<VcvtOp>(
                                   op->getLoc(), resultType, sourcePart, partMask, rnd, sat,
                                   rewriter.getStringAttr(parts[partIndex * partStride]))
                               .getResult());
    }

    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
        merged = rewriter.create<VorOp>(op->getLoc(), resultType, merged, partial, *resultMask).getResult();
    }
    return merged;
}

static LogicalResult lowerNarrowFpToInt(
    Operation* op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks, int64_t sourceFactor,
    int64_t partStride, ArrayRef<StringRef> parts, StringAttr rnd, StringAttr sat, StringRef sourceMaskDiagnostic,
    StringRef resultMaskDiagnostic, TypeConverter* typeConverter, OneToNPatternRewriter& rewriter)
{
    bool nonPositiveFactor = sourceFactor <= 0 || partStride <= 0;
    bool strideOutOfRange = !nonPositiveFactor && (sourceFactor - 1) * partStride >= static_cast<int64_t>(parts.size());
    bool arityMismatch = sourceParts.size() != static_cast<size_t>(sourceFactor) * resultTypes.size();
    if (nonPositiveFactor || strideOutOfRange || arityMismatch) {
        return rewriter.notifyMatchFailure(op, "narrow fp-to-int source arity does not match conversion factor");
    }

    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
        return rewriter.notifyMatchFailure(op, "expected physical fp source type");
    }
    Value sourceMask;
    if (masks.empty()) {
        FailureOr<Value> seed = createAllTrueMaskForVReg(op->getLoc(), sourceType, rewriter);
        if (failed(seed)) {
            return rewriter.notifyMatchFailure(op, sourceMaskDiagnostic);
        }
        sourceMask = *seed;
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
        FailureOr<Value> result = buildNarrowFpToIntResult(
            op, sourceParts, resultType, chunkIndex, sourceFactor, partStride, parts, rnd, sat, sourceMask, masks,
            rewriter);
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
    Operation* op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks, ArrayRef<StringRef> parts,
    StringAttr rnd, StringAttr sat, StringRef arityDiagnostic, StringRef maskDiagnostic, TypeConverter* typeConverter,
    OneToNPatternRewriter& rewriter)
{
    bool invalidArity = parts.empty() || resultTypes.size() != parts.size() * sourceParts.size();
    if (invalidArity) {
        return rewriter.notifyMatchFailure(op, arityDiagnostic);
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
        return rewriter.notifyMatchFailure(op, maskDiagnostic);
    }
    Value unmaskedSeed;
    if (masks.empty()) {
        FailureOr<Value> mask = createAllTrueMaskForVReg(op->getLoc(), sourceType, rewriter);
        if (failed(mask)) {
            return rewriter.notifyMatchFailure(op, maskDiagnostic);
        }
        unmaskedSeed = *mask;
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (size_t partIndex = 0; partIndex < parts.size(); ++partIndex) {
        for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
            VRegType resultType = resultTypes[partIndex * sourceParts.size() + chunkIndex];
            // Both parts of a widened chunk read the same authored source predicate.
            FailureOr<Value> mask = getConversionMaskForChunk(masks, chunkIndex, unmaskedSeed);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, maskDiagnostic);
            }
            results.push_back(
                rewriter
                    .create<VcvtOp>(
                        op->getLoc(), resultType, sourcePart, *mask, rnd, sat, rewriter.getStringAttr(parts[partIndex]))
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
  // The narrowing result lane stride comes from the carrier view of the
  // declared result layout: a single-carrier group-slot packet with slots = 1
  // or with the narrowing factor as its lane stride places its groups exactly
  // where the dense lane-strided result does, so the same converted lane
  // ordering applies and the physical part is forwarded unchanged.
  VMILayoutAttr resultView = getVMICastDenseCarrierView(
      resultLayout, resultVMIType.getElementType());
  int64_t resultLaneStride = resultView && resultView.isContiguous()
                                 ? resultView.getLaneStride()
                                 : 1;
  bool invalidResultLaneStride =
      resultLaneStride <= 0 || factor % resultLaneStride != 0;
  if (invalidResultLaneStride) {
    return rewriter.notifyMatchFailure(op, unsupportedLaneStrideDiagnostic);
  }
  int64_t sourceFactor = factor / resultLaneStride;
  if (isVMISingleGroupPerPartPacket(resultLayout)) {
    // One group per result part is fed by one source part: the part family maps
    // source lane 0 to result lane 0 whatever the narrowing radix, so a packet
    // pair like this narrows 1:1 per part instead of merging source parts.
    sourceFactor = 1;
  }
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
        VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks, VRegType sourceType,
        unsigned sourceBits, StringAttr rnd, StringAttr sat, OneToNPatternRewriter& rewriter) const
    {
        StringRef part = sourceBits == 16 ? StringRef("EVEN") : StringRef("P0");
        Value unmaskedSeed;
        if (masks.empty()) {
            FailureOr<Value> mask = createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build fptosi widen 1:1 mask");
            }
            unmaskedSeed = *mask;
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
            auto [sourcePart, resultType] = sourceAndResult;
            FailureOr<Value> mask = getConversionMaskForChunk(masks, index, unmaskedSeed);
            if (failed(mask)) {
                return rewriter.notifyMatchFailure(op, "failed to build fptosi widen 1:1 mask");
            }
            results.push_back(
                rewriter
                    .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask, rnd, sat, rewriter.getStringAttr(part))
                    .getResult());
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    LogicalResult lowerWiden(
        VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
        ValueRange masks, VMIVRegType sourceVMIType, VMIVRegType resultVMIType, VRegType sourceType,
        unsigned sourceBits, StringAttr rnd, StringAttr sat, OneToNPatternRewriter& rewriter) const
    {
        VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
        VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
        // Same carrier-view plan as pto.vmi.extf: a single-carrier group-slot
        // packet with the widening factor as its lane stride, or a packet with one
        // group per part, is the dense lane-strided carrier in every respect the
        // part selection depends on.
        VMILayoutAttr sourceView = getVMICastDenseCarrierView(sourceLayout, sourceType.getElementType());
        VMILayoutAttr resultView = resultTypes.empty() ?
                                       VMILayoutAttr() :
                                       getVMICastDenseCarrierView(resultLayout, resultTypes.front().getElementType());
        bool packedRadix = sourceBits == kElementBits16 && sourceView.getLaneStride() == kPairWidth;
        bool quadRadix = sourceBits == kElementBits8 && sourceView.getLaneStride() == kQuadWidth;
        bool sourceSelectsPartLanes =
            sourceView && (packedRadix || quadRadix || isVMISingleGroupPerPartPacket(sourceLayout));
        bool denseOneToOne = sourceView && resultView && sourceView.isContiguous() && resultView.isContiguous() &&
                             resultView.getLaneStride() == 1 && sourceSelectsPartLanes &&
                             physicalResultTypes.size() == sourceParts.size();
        if (denseOneToOne) {
            return lowerDenseWiden(op, sourceParts, resultTypes, masks, sourceType, sourceBits, rnd, sat, rewriter);
        }
        static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
        return lowerWidenFpToInt(
            op, sourceParts, resultTypes, masks, kEvenOddParts, rnd, sat,
            "widen fptosi requires result arity = 2 × source arity", "failed to build fptosi widen mask",
            this->getTypeConverter(), rewriter);
    }

    LogicalResult lowerNarrow(
        VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
        ValueRange masks, VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits, StringAttr rnd,
        StringAttr sat, OneToNPatternRewriter& rewriter) const
    {
        FailureOr<NarrowFpToIntPlan> plan = buildNarrowFpToIntPlan(
            op, sourceParts, physicalResultTypes, resultVMIType, sourceBits, resultBits, rewriter,
            "narrow fptosi requires positive result bit width", "narrow fptosi: unsupported result lane stride",
            "narrow fptosi: source arity != sourceFactor × result arity");
        if (failed(plan)) {
            return failure();
        }
        static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
        static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
        ArrayRef<StringRef> parts = plan->factor == mlir::pto::kValue2 ? ArrayRef<StringRef>(kEvenOddParts) :
                                                                         ArrayRef<StringRef>(kPacked4Parts);
        return lowerNarrowFpToInt(
            op, sourceParts, resultTypes, masks, plan->sourceFactor, plan->resultLaneStride, parts, rnd, sat,
            "failed to build fptosi source mask", "failed to build narrow fptosi result mask", this->getTypeConverter(),
            rewriter);
    }

    LogicalResult lowerConversion(
        VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks, VMIVRegType sourceVMIType,
        VMIVRegType resultVMIType, OneToNPatternRewriter& rewriter) const
    {
        Type sourceElementType = sourceVMIType.getElementType();
        Type resultElementType = resultVMIType.getElementType();
        auto contract = lookupVMIFpToSiContract(sourceElementType, resultElementType);
        if (!contract) {
            return rewriter.notifyMatchFailure(op, "unsupported fp-to-si conversion element type pair");
        }
        FailureOr<FpToIntPartValidation> validation = validateFpToIntConversionParts(
            op, sourceParts, resultTypes, rewriter, "fptosi requires at least one physical source chunk",
            "expected physical fptosi source type", "fptosi source physical parts must have matching type",
            "fptosi requires at least one physical result chunk", "unsupported physical fptosi result type");
        if (failed(validation)) {
            return failure();
        }

        StringAttr rnd = op->getAttrOfType<StringAttr>("rounding");
        if (!rnd) {
            rnd = rewriter.getStringAttr("R");
        }
        StringAttr sat = contract->requiresSat ? op->getAttrOfType<StringAttr>("saturate") : nullptr;
        if (!contract->requiresPart) {
            return lowerSameWidthFpToInt(
                op, sourceParts, validation->resultVRegTypes, masks, rnd, sat,
                "same-width fptosi requires matching physical arity", "failed to build fptosi mask",
                this->getTypeConverter(), rewriter);
        }

        unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceElementType);
        unsigned resultBits = pto::getPTOStorageElemBitWidth(resultElementType);
        if (resultBits > sourceBits) {
            return lowerWiden(
                op, sourceParts, resultTypes, validation->resultVRegTypes, masks, sourceVMIType, resultVMIType,
                validation->sourceType, sourceBits, rnd, sat, rewriter);
        }
        return lowerNarrow(
            op, sourceParts, resultTypes, validation->resultVRegTypes, masks, resultVMIType, sourceBits, resultBits,
            rnd, sat, rewriter);
    }

    LogicalResult lowerWithResultTypes(
        VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
        auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
        return lowerConversion(op, sourceParts, resultTypes, masks, sourceVMIType, resultVMIType, rewriter);
    }

public:

  LogicalResult
  matchAndRewrite(VMIFPToSIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerCastWithConversionPredicate(
        op, sourceParts, adaptor.getMask(), *this->getTypeConverter(), rewriter,
        [&](ArrayRef<Type> resultTypes, ValueRange masks) {
            return lowerWithResultTypes(op, sourceParts, resultTypes, masks, rewriter);
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
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
      ValueRange masks, StringAttr rnd, StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      bool invalidWidenArity = physicalResultTypes.size() != 2 * sourceParts.size();
      if (invalidWidenArity) {
          return rewriter.notifyMatchFailure(op, "widen fptoui requires result arity = 2 × source arity");
      }
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return lowerWidenFpToInt(
          op, sourceParts, resultTypes, masks, kEvenOddParts, rnd, sat,
          "widen fptoui requires result arity = 2 × source arity", "failed to build fptoui widen mask",
          this->getTypeConverter(), rewriter);
  }

  LogicalResult lowerNarrow(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
      ValueRange masks, VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits, StringAttr rnd,
      StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<NarrowFpToIntPlan> plan = buildNarrowFpToIntPlan(
          op, sourceParts, physicalResultTypes, resultVMIType, sourceBits, resultBits, rewriter,
          "narrow fptoui requires positive result bit width", "narrow fptoui: unsupported result lane stride",
          "narrow fptoui: source arity != sourceFactor × result arity");
      if (failed(plan)) {
          return failure();
      }
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return lowerNarrowFpToInt(
          op, sourceParts, resultTypes, masks, plan->sourceFactor, plan->resultLaneStride, kEvenOddParts, rnd, sat,
          "failed to build fptoui source mask", "failed to build narrow fptoui result mask", this->getTypeConverter(),
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
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<FPToUILoweringInput> input =
          getLoweringInput(op, sourceParts, resultTypes, sourceVMIType, resultVMIType, rewriter);
      if (failed(input)) {
          return failure();
      }
      if (!input->contract.requiresPart) {
          return lowerSameWidthFpToInt(
              op, input->sourceParts, input->resultVRegTypes, masks, input->rounding, input->saturate,
              "same-width fptoui requires matching physical arity", "failed to build fptoui mask",
              this->getTypeConverter(), rewriter);
      }
      unsigned sourceBits = pto::getPTOStorageElemBitWidth(input->sourceVMIType.getElementType());
      unsigned resultBits = pto::getPTOStorageElemBitWidth(input->resultVMIType.getElementType());
      if (resultBits > sourceBits) {
          return lowerWiden(
              op, input->sourceParts, input->resultTypes, input->resultVRegTypes, masks, input->rounding,
              input->saturate, rewriter);
      }
      if (resultBits < sourceBits) {
          return lowerNarrow(
              op, input->sourceParts, input->resultTypes, input->resultVRegTypes, masks, input->resultVMIType,
              sourceBits, resultBits, input->rounding, input->saturate, rewriter);
      }
      return failure();
  }

  LogicalResult lowerWithResultTypes(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
      auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
      return lowerConversion(op, sourceParts, resultTypes, masks, sourceVMIType, resultVMIType, rewriter);
  }

  LogicalResult lowerConversionWithResolvedTypes(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      return lowerWithResultTypes(op, sourceParts, resultTypes, masks, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMIFPToUIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerCastWithConversionPredicate(
        op, sourceParts, adaptor.getMask(), *this->getTypeConverter(), rewriter,
        [&](ArrayRef<Type> resultTypes, ValueRange masks) {
            return lowerConversionWithResolvedTypes(op, sourceParts, resultTypes, masks, rewriter);
        });
  }
};

struct OneToNVMISIToFPOpPattern : OneToNOpConversionPattern<VMISIToFPOp> {
  using OneToNOpConversionPattern<VMISIToFPOp>::OneToNOpConversionPattern;

private:
    LogicalResult lowerSameWidth(
        VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, Value mask, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        bool invalidArity = sourceParts.size() != resultTypes.size();
        if (invalidArity) {
            return rewriter.notifyMatchFailure(op, "si32->f32 requires matching physical arity");
        }
        StringAttr rnd = rewriter.getStringAttr("R");
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
            auto [sourcePart, resultType] = sourceAndResult;
            FailureOr<Value> partMask = getConversionMaskForChunk(masks, index, mask);
            if (failed(partMask)) {
                return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
            }
            results.push_back(
                rewriter.create<VcvtOp>(op.getLoc(), resultType, sourcePart, *partMask, rnd, nullptr, nullptr)
                    .getResult());
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    LogicalResult lowerWiden(
        VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, Value mask, ValueRange masks,
        OneToNPatternRewriter& rewriter) const
    {
        // The physical arity is not the widening ratio: a 256-byte part holds both
        // 256 si8 lanes and the 128 f16 lanes they widen to, so a small logical
        // shape converts part-for-part.  The radix-2 relation then selects the even
        // source lanes with one EVEN vcvt per part, exactly like f16->f32.
        bool partForPart = resultTypes.size() == sourceParts.size();
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        if (partForPart) {
            // EVEN selects the second byte of each 16-bit carrier, which is the source
            // value only after layout assignment unpacked the bytes to lane_stride 2.
            // Any other carrier form would convert the wrong lanes silently, so fail
            // the match instead of emitting it.
            VMILayoutAttr sourceLayout = cast<VMIVRegType>(op.getSource().getType()).getLayoutAttr();
            bool unpackedLaneStride = sourceLayout && sourceLayout.getLaneStride() == 2;
            if (!unpackedLaneStride) {
                return rewriter.notifyMatchFailure(
                    op, "si8->f16 1:1 widening requires the unpacked lane_stride = 2 "
                        "source");
            }
            for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
                auto [sourcePart, resultType] = sourceAndResult;
                FailureOr<Value> partMask = getConversionMaskForChunk(masks, index, mask);
                if (failed(partMask)) {
                    return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
                }
                results.push_back(rewriter
                                      .create<VcvtOp>(
                                          op.getLoc(), resultType, sourcePart, *partMask, nullptr, nullptr,
                                          rewriter.getStringAttr("EVEN"))
                                      .getResult());
            }
            replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
            return success();
        }
        bool invalidArity = resultTypes.size() != 2 * sourceParts.size();
        if (invalidArity) {
            return rewriter.notifyMatchFailure(op, "si8->f16 requires result arity = 1 x or 2 x source arity");
        }
        static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
        for (size_t partIndex = 0; partIndex < std::size(kEvenOddParts); ++partIndex) {
            for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
                VRegType resultType = resultTypes[partIndex * sourceParts.size() + chunkIndex];
                FailureOr<Value> partMask = getConversionMaskForChunk(masks, chunkIndex, mask);
                if (failed(partMask)) {
                    return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
                }
                results.push_back(rewriter
                                      .create<VcvtOp>(
                                          op.getLoc(), resultType, sourcePart, *partMask, nullptr, nullptr,
                                          rewriter.getStringAttr(kEvenOddParts[partIndex]))
                                      .getResult());
            }
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
    }

    LogicalResult lowerConversion(
        VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, Value mask, ValueRange masks,
        unsigned sourceBits, unsigned resultBits, OneToNPatternRewriter& rewriter) const
    {
        if (sourceBits == mlir::pto::kValue32 && resultBits == mlir::pto::kValue32) {
            return lowerSameWidth(op, sourceParts, resultTypes, mask, masks, rewriter);
        } else if (sourceBits == mlir::pto::kValue8 && resultBits == mlir::pto::kValue16) {
            return lowerWiden(op, sourceParts, resultTypes, mask, masks, rewriter);
        } else {
            return rewriter.notifyMatchFailure(op, "unsupported sitofp source/result width relation");
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
      VMISIToFPOp op, ValueRange sourceParts, TypeRange resultTypes, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      FailureOr<VRegType> sourceType = collectSourceType(op, sourceParts, rewriter);
      if (failed(sourceType)) {
          return failure();
      }
      FailureOr<SmallVector<VRegType>> resultVRegTypes = collectResultTypes(op, resultTypes, rewriter);
      if (failed(resultVRegTypes)) {
          return failure();
      }
      unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceType->getElementType());
      unsigned resultBits = pto::getPTOStorageElemBitWidth(resultVRegTypes->front().getElementType());
      Value unmaskedSeed;
      if (masks.empty()) {
          FailureOr<Value> mask = createAllTrueMaskForVReg(op.getLoc(), *sourceType, rewriter);
          if (failed(mask)) {
              return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
          }
          unmaskedSeed = *mask;
      }
      return lowerConversion(op, sourceParts, *resultVRegTypes, unmaskedSeed, masks, sourceBits, resultBits, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMISIToFPOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerCastWithConversionPredicate(
        op, sourceParts, adaptor.getMask(), *this->getTypeConverter(), rewriter,
        [&](ArrayRef<Type> resultTypes, ValueRange masks) {
            return lowerPhysicalConversion(op, sourceParts, resultTypes, masks, rewriter);
        });
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
        op, resultTypes,
        rewriter,
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

