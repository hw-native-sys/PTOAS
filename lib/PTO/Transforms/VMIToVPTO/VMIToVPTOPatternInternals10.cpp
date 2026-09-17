// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals10.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

struct OneToNVMIExtFOpPattern : OneToNOpConversionPattern<VMIExtFOp> {
  using OneToNOpConversionPattern<VMIExtFOp>::OneToNOpConversionPattern;

private:
  struct ExtFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
  };

  FailureOr<ExtFPhysicalPlan> buildPhysicalPlan(
      VMIExtFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(
          op, "extf requires at least one physical source chunk");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected physical extf source");
    }
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "extf source physical parts must have matching type");
      }
    }
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidFirstResult =
          resultVRegTypes.empty() &&
          (!resultVRegType ||
           !(resultVRegType.getElementType().isF32() ||
             pto::isPTOBF16x2Type(resultVRegType.getElementType())));
      bool mismatchedResult =
          !resultVRegTypes.empty() && resultVRegType != resultVRegTypes.front();
      if (invalidFirstResult || mismatchedResult) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical extf result type");
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return ExtFPhysicalPlan{sourceType, std::move(resultVRegTypes)};
  }

  struct ResultViewPlan {
    bool isPackedBF16x2;
    VRegType vcvtResultType;
  };

  ResultViewPlan buildResultViewPlan(ArrayRef<VRegType> resultTypes,
                                     OneToNPatternRewriter &rewriter) const {
    bool isPackedBF16x2 =
        pto::isPTOBF16x2Type(resultTypes.front().getElementType());
    VRegType vcvtResultType = resultTypes.front();
    if (isPackedBF16x2) {
        vcvtResultType = VRegType::get(
            rewriter.getContext(), resultTypes.front().getElementCount() * mlir::pto::kValue2,
            BFloat16Type::get(rewriter.getContext()));
    }
    return ResultViewPlan{isPackedBF16x2, vcvtResultType};
  }

  static Value createVcvtResult(Location loc, VRegType resultType,
                                Value sourcePart, Value mask, StringAttr rnd,
                                StringAttr sat, StringAttr part,
                                bool resultIsPackedBF16x2,
                                VRegType vcvtResultVRegType,
                                OneToNPatternRewriter &rewriter) {
    VRegType vcvtType = resultIsPackedBF16x2 ? vcvtResultVRegType : resultType;
    Value vcvt = rewriter
                     .create<VcvtOp>(loc, vcvtType, sourcePart, mask, rnd, sat,
                                     part)
                     .getResult();
    if (!resultIsPackedBF16x2) {
      return vcvt;
    }
    return rewriter.create<VbitcastOp>(loc, resultType, vcvt).getResult();
  }

  LogicalResult lowerLaneStride(
      VMIExtFOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, Value mask, StringRef part,
      bool resultIsPackedBF16x2, VRegType vcvtResultVRegType) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(createVcvtResult(
          op.getLoc(), resultType, sourcePart, mask, /*rnd=*/nullptr,
          /*sat=*/nullptr, rewriter.getStringAttr(part),
          resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerPackedE2M1LaneStride2(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      OneToNPatternRewriter &rewriter) const {
    static constexpr StringRef kPacked2Parts[] = {"P0", "P2"};
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
    return lowerFactor(
        op, rewriter, sourceParts, plan.resultTypes, kPacked2Parts, mlir::pto::kValue2, *mask, viewPlan.isPackedBF16x2,
        viewPlan.vcvtResultType);
  }

  LogicalResult lowerFactor(
      VMIExtFOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, ArrayRef<StringRef> parts,
      int64_t factor, Value mask, bool resultIsPackedBF16x2,
      VRegType vcvtResultVRegType) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(createVcvtResult(
            op.getLoc(), resultType, sourcePart, mask, /*rnd=*/nullptr,
            /*sat=*/nullptr, rewriter.getStringAttr(parts[partIndex]),
            resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  /// Dense lane-stride widening plan.  `applies` is set when each declared
  /// layout describes the same carrier as its dense lane-strided form, and
  /// `part` is then the vcvt part that selects the source lanes.  Planning from
  /// the carrier view of a single-carrier group-slot packet -- rather than from
  /// its declared form -- is what lets a packet source or result take the same
  /// one-vcvt plan as its dense counterpart, with the physical part forwarded
  /// unchanged and no pack, zip or shuffle implied.
  struct ExtFDenseLaneStridePlan {
    bool applies = false;
    StringRef part;
  };

  static ExtFDenseLaneStridePlan buildDenseLaneStridePlan(
      const ExtFPhysicalPlan &plan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, size_t sourcePartCount) {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(plan.sourceType.getElementType());
    VMILayoutAttr sourceView = getVMICastDenseCarrierView(
        sourceLayout, plan.sourceType.getElementType());
    VMILayoutAttr resultView = getVMICastDenseCarrierView(
        resultLayout, plan.resultTypes.front().getElementType());
    int64_t widenFactor = 0;
    if (sourceBits == kElementBits16) {
      widenFactor = kPairWidth;
    } else if (sourceBits == kElementBits8) {
      widenFactor = kQuadWidth;
    }
    // A packet that keeps one group per part (slots = 1) only fills lane 0,
    // which every part family maps to result lane 0, so it widens 1:1 per part
    // like the dense lane-strided form.
    bool sourceSelectsLanes =
        sourceView && widenFactor != 0 &&
        (sourceView.getLaneStride() == widenFactor ||
         isVMISingleGroupPerPartPacket(sourceLayout));
    bool applies = sourceView && resultView && sourceView.isContiguous() &&
                   resultView.isContiguous() &&
                   resultView.getLaneStride() == 1 && sourceSelectsLanes &&
                   plan.resultTypes.size() == sourcePartCount;
    StringRef part =
        sourceBits == kElementBits16 ? StringRef("EVEN") : StringRef("P0");
    return ExtFDenseLaneStridePlan{applies, part};
  }

  struct ExtFFactorPlan {
    ArrayRef<StringRef> parts;
    int64_t factor;
  };

  FailureOr<ExtFFactorPlan> buildFactorPlan(
      VMIExtFOp op, unsigned sourceBits, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
      if (sourceBits == mlir::pto::kValue16 && resultPartCount == 2 * sourcePartCount) {
          static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
          return ExtFFactorPlan{ArrayRef<StringRef>(kEvenOddParts), 2};
      }
      if (sourceBits == mlir::pto::kValue8 && resultPartCount == 4 * sourcePartCount) {
          static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
          return ExtFFactorPlan{ArrayRef<StringRef>(kPacked4Parts), 4};
      }
    return rewriter.notifyMatchFailure(
        op, "unsupported physical extf source/result width relation");
  }

  FailureOr<Value> createSeedMask(VMIExtFOp op, VRegType sourceType,
                                   OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, "failed to build extf seed mask");
    }
    return *mask;
  }

  // Spine-scoped composite widening: the narrow source is deinterleaved=4 with
  // each physical part carrying one modulo-4 lane partition, so one
  // part-per-chunk vcvt (EVEN for 16-bit, P0 for 8-bit) is the whole
  // conversion and no pto.vor merge is needed.
  FailureOr<bool> tryLowerSpineCompositeWidening(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      unsigned sourceBits, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, const ResultViewPlan &viewPlan,
      OneToNPatternRewriter &rewriter) const {
    bool spineCompositeWidening =
        sourceLayout && resultLayout && sourceLayout.isDeinterleaved() &&
        sourceLayout.getFactor() == 4 && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 4 && resultLayout.getLaneStride() == 1 &&
        ((sourceBits == 8 && sourceLayout.getLaneStride() == 4) ||
         (sourceBits == 16 && sourceLayout.getLaneStride() == 2)) &&
        plan.resultTypes.size() == sourceParts.size();
    if (!spineCompositeWidening) {
      return false;
    }
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    StringRef part = sourceBits == 8 ? StringRef("P0") : StringRef("EVEN");
    if (failed(lowerLaneStride(op, rewriter, sourceParts, plan.resultTypes,
                               *mask, part, viewPlan.isPackedBF16x2,
                               viewPlan.vcvtResultType))) {
      return failure();
    }
    return true;
  }

  LogicalResult lowerPhysicalExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(plan.sourceType.getElementType());
    // A packed bf16x2 physical result cannot be produced directly by
    // pto.vcvt (classifyVcvtElemType has no BF16x2 branch); the widest native
    // f4 conversion result element is bf16. Build the bf16 view type (2 bf16
    // lanes per bf16x2 lane) and reinterpret each vcvt result with a
    // physical-noop VbitcastOp, mirroring the source-side reinterpret in
    // OneToNVMITruncFOpPattern (viewVcvtSource).
    ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
    ExtFDenseLaneStridePlan laneStridePlan =
        buildDenseLaneStridePlan(plan, sourceLayout, resultLayout,
                                 sourceParts.size());
    if (laneStridePlan.applies) {
      FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
      if (failed(mask)) {
        return failure();
      }
      return lowerLaneStride(op, rewriter, sourceParts, plan.resultTypes, *mask,
                             laneStridePlan.part, viewPlan.isPackedBF16x2,
                             viewPlan.vcvtResultType);
    }

    // Packed f4E2M1x2 sources stored with lane_stride = 2 (UNPK_B8) have
    // valid bytes on the even lanes; P0 (lanes 0 mod 4) plus P2 (lanes 2 mod
    // 4) cover them while P1/P3 are zero-fill gaps, so widen through the
    // {P0, P2} part pair instead of the dense factor-4 selection.
    bool packedE2M1LaneStride2 =
        sourceLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 2 &&
        isa<pto::F4E2M1x2Type>(plan.sourceType.getElementType()) &&
        plan.resultTypes.size() == 2 * sourceParts.size();
    if (packedE2M1LaneStride2) {
      return lowerPackedE2M1LaneStride2(op, sourceParts, plan, rewriter);
    }

    FailureOr<bool> spineCompositeWidening = tryLowerSpineCompositeWidening(
        op, sourceParts, plan, sourceBits, sourceLayout, resultLayout, viewPlan,
        rewriter);
    if (failed(spineCompositeWidening)) {
      return failure();
    }
    if (*spineCompositeWidening) {
      return success();
    }

    FailureOr<ExtFFactorPlan> factorPlan = buildFactorPlan(
        op, sourceBits, sourceParts.size(), plan.resultTypes.size(), rewriter);
    if (failed(factorPlan)) {
      return failure();
    }
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    return lowerFactor(op, rewriter, sourceParts, plan.resultTypes,
                       factorPlan->parts, factorPlan->factor, *mask,
                       viewPlan.isPackedBF16x2,
                       viewPlan.vcvtResultType);
  }

public:

  LogicalResult
  matchAndRewrite(VMIExtFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    FailureOr<ExtFPhysicalPlan> plan =
        buildPhysicalPlan(op, input->sourceParts, input->resultTypes, rewriter);
    if (failed(plan)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
    return lowerPhysicalExtF(op, input->sourceParts, *plan, sourceLayout,
                             resultLayout, rewriter);
  }
};

static bool hasUnsupportedPackedTruncFConversion(Type sourceElementType,
                                                 Type resultElementType) {
  bool usesPackedCarrier =
      isVMIPackedFloatCarrierType(sourceElementType) ||
      isVMIPackedFloatCarrierType(resultElementType);
  return usesPackedCarrier &&
         !lookupVMIFpToFpContract(sourceElementType, resultElementType);
}

static bool hasGroupSlotTruncFLayouts(VMILayoutAttr sourceLayout,
                                      VMILayoutAttr resultLayout) {
  return sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
         resultLayout.isGroupSlots();
}

struct OneToNVMITruncFOpPattern : OneToNOpConversionPattern<VMITruncFOp> {
  using OneToNOpConversionPattern<VMITruncFOp>::OneToNOpConversionPattern;

private:
  struct TruncFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
    VRegType sourceViewType;
    unsigned sourceBits;
    unsigned resultBits;
    bool sourceIsPackedBF16x2;
  };

  struct TruncFNarrowingPlan {
    ArrayRef<StringRef> parts;
    int64_t sourceFactor;
    int64_t resultLaneStride;
  };

  FailureOr<VRegType> getUniformSourceType(
      VMITruncFOp op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "truncf requires source chunks");
    }
    auto firstType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!firstType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source type");
    }
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != firstType) {
        return rewriter.notifyMatchFailure(
            op, "truncf source physical parts must have matching type");
      }
    }
    return firstType;
  }

  FailureOr<SmallVector<VRegType>> getUniformResultTypes(
      VMITruncFOp op, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type physicalResultType : resultTypes) {
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      bool invalidType = !resultType ||
                         (!resultVRegTypes.empty() &&
                          resultType != resultVRegTypes.front());
      if (invalidType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical truncf result type");
      }
      resultVRegTypes.push_back(resultType);
    }
    return resultVRegTypes;
  }

  FailureOr<TruncFPhysicalPlan> buildPhysicalPlan(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "truncf requires result chunks");
    }
    FailureOr<VRegType> sourceType =
        getUniformSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType->getElementType());
    if (sourceBits != mlir::pto::kValue32 && sourceBits != 16) {
        return rewriter.notifyMatchFailure(op, "truncf source bit width must be 32 or 16");
    }
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        getUniformResultTypes(op, resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    if (resultBits == 0) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result type");
    }
    bool sourceIsPackedBF16x2 =
        pto::isPTOBF16x2Type(sourceType->getElementType());
    VRegType sourceViewType = *sourceType;
    if (sourceIsPackedBF16x2) {
        sourceViewType = VRegType::get(
            rewriter.getContext(), sourceType->getElementCount() * mlir::pto::kValue2,
            BFloat16Type::get(rewriter.getContext()));
    }
    return TruncFPhysicalPlan{*sourceType, std::move(*resultVRegTypes),
                              sourceViewType, sourceBits, resultBits,
                              sourceIsPackedBF16x2};
  }

  LogicalResult lowerDenseLaneStride(
      VMITruncFOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, StringRef part,
      bool sourceIsPackedBF16x2, VRegType vcvtSourceVRegType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), vcvtSourceVRegType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    StringAttr partAttr = rewriter.getStringAttr(part);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(rewriter
                            .create<VcvtOp>(
                                op.getLoc(), resultType,
                                makeVcvtSourceView(
                                    op.getLoc(), sourcePart,
                                    sourceIsPackedBF16x2, vcvtSourceVRegType,
                                    rewriter),
                                *sourceMask, rnd, sat, partAttr)
                            .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> lowerGroupSlotTruncPart(
      VMITruncFOp op, Value sourcePart, Type physicalResultType,
      Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
    auto resultType = dyn_cast<VRegType>(physicalResultType);
    const bool invalidTypes =
        !sourceType || !sourceType.getElementType().isF32() || !resultType;
    if (invalidTypes) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported group-slot truncf physical type");
      return failure();
    }
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    const bool unsupportedResultBits = resultBits != 16 && resultBits != 8;
    if (unsupportedResultBits) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported group-slot truncf physical type");
      return failure();
    }
    StringAttr part =
        rewriter.getStringAttr(resultBits == 16 ? "EVEN" : "P0");
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultType.getElementType()));
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask,
                        rnd, sat, part)
        .getResult();
  }

  static Value makeVcvtSourceView(Location loc, Value sourcePart,
                                  bool sourceIsPackedBF16x2,
                                  VRegType vcvtSourceVRegType,
                                  OneToNPatternRewriter &rewriter) {
    if (!sourceIsPackedBF16x2) {
      return sourcePart;
    }
    if (auto vbc = sourcePart.getDefiningOp<VbitcastOp>()) {
      if (auto srcVReg = dyn_cast<VRegType>(vbc.getInput().getType());
          srcVReg && srcVReg.getElementType().isBF16()) {
        return vbc.getInput();
      }
    }
    return rewriter.create<VbitcastOp>(loc, vcvtSourceVRegType, sourcePart)
        .getResult();
  }

  LogicalResult lowerSameWidth(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      VRegType sourceViewType, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceViewType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultTypes.front().getElementType()));
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                            *sourceMask, rnd, sat,
                                            /*part=*/nullptr)
                            .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildNarrowTruncResult(
      VMITruncFOp op, ValueRange sourceParts, VRegType resultType,
      ArrayRef<StringRef> allParts, int64_t chunkIndex, int64_t sourceFactor,
      int64_t resultLaneStride, VRegType sourceViewType,
      bool sourceIsPackedBF16x2, Value sourceMask, StringAttr rnd,
      StringAttr sat, OneToNPatternRewriter &rewriter) const {
    if (sourceFactor <= 0) {
      return rewriter.notifyMatchFailure(
          op, "narrow truncf requires a positive source factor");
    }
    int64_t safeSourceFactor = sourceFactor;
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(resultMask)) {
      return failure();
    }
    SmallVector<Value> partials;
    partials.reserve(safeSourceFactor);
    for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
      Value sourcePart =
          sourceParts[partIndex * (sourceParts.size() / safeSourceFactor) +
                      chunkIndex];
      bool hasIndexedPart =
          partIndex * resultLaneStride < static_cast<int64_t>(allParts.size());
      StringRef part =
          hasIndexedPart ? allParts[partIndex * resultLaneStride]
                         : allParts[partIndex];
      partials.push_back(
          rewriter
              .create<VcvtOp>(
                  op.getLoc(), resultType,
                  makeVcvtSourceView(op.getLoc(), sourcePart,
                                     sourceIsPackedBF16x2, sourceViewType,
                                     rewriter),
                  sourceMask, rnd, sat, rewriter.getStringAttr(part))
              .getResult());
    }
    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                  *resultMask)
                   .getResult();
    }
    return merged;
  }

  LogicalResult lowerNarrow(
      VMITruncFOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, ArrayRef<StringRef> allParts,
      int64_t sourceFactor, int64_t resultLaneStride,
      VRegType sourceViewType, bool sourceIsPackedBF16x2,
      StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    if (sourceFactor <= 0 || resultLaneStride <= 0 ||
        sourceParts.size() !=
            static_cast<size_t>(sourceFactor) * resultTypes.size()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceViewType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> result = buildNarrowTruncResult(
          op, sourceParts, resultType, allParts, chunkIndex, sourceFactor,
          resultLaneStride, sourceViewType, sourceIsPackedBF16x2, *sourceMask,
          rnd, sat, rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to build truncf result mask");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<TruncFNarrowingPlan> buildNarrowingPlan(
      VMITruncFOp op, unsigned sourceBits, unsigned resultBits,
      VMILayoutAttr resultLayout, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (resultBits * mlir::pto::kValue2 == sourceBits) {
        static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
        parts = kEvenOddParts;
        factor = mlir::pto::kValue2;
    } else if (resultBits * mlir::pto::kValue4 == sourceBits) {
        static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
        parts = kPacked4Parts;
        factor = mlir::pto::kValue4;
    } else {
        return rewriter.notifyMatchFailure(op, "unsupported physical truncf source/result width relation");
    }
    int64_t resultLaneStride = resultLayout && resultLayout.isContiguous()
                                   ? resultLayout.getLaneStride()
                                   : 1;
    bool invalidResultLaneStride =
        resultLaneStride <= 0 || factor % resultLaneStride != 0;
    if (invalidResultLaneStride) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result lane stride");
    }
    int64_t sourceFactor = factor / resultLaneStride;
    bool sourceArityMismatch =
        sourcePartCount != static_cast<size_t>(sourceFactor) * resultPartCount;
    if (sourceArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");
    }
    return TruncFNarrowingPlan{parts, sourceFactor, resultLaneStride};
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        !sourceVMIType.getElementType().isF32() ||
        (resultBits != 16 && resultBits != 8) ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      return rewriter.notifyMatchFailure(op, "unsupported group-slot truncf shape");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    const char *activeSlotPattern =
        sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
    FailureOr<Value> activeSlotMask = createPrefixMask(
        op.getLoc(), MaskType::get(rewriter.getContext(), "b32"),
        activeSlotPattern, rewriter);
    if (failed(activeSlotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot truncf active slot mask");
    }
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = lowerGroupSlotTruncPart(
          op, sourcePart, physicalResultType, *activeSlotMask, sat, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<bool> tryLowerSameWidthTrunc(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool sameWidthContiguous =
        physicalPlan.sourceBits == physicalPlan.resultBits && sourceLayout &&
        resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == 1 &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!sameWidthContiguous) {
      return false;
    }
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    if (failed(lowerSameWidth(op, sourceParts, physicalPlan.resultTypes,
                              physicalPlan.sourceViewType, sat, rewriter))) {
      return failure();
    }
    return true;
  }

  FailureOr<bool> tryLowerDenseLaneStrideTrunc(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool denseLaneStrideNarrowing =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() != 1 &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!denseLaneStrideNarrowing) {
      return false;
    }
    bool isEven32To16 = physicalPlan.resultBits == 16 &&
                          resultLayout.getLaneStride() == 2;
    bool isPacked32To8 = physicalPlan.resultBits == 8 &&
                           resultLayout.getLaneStride() == 4;
    bool isEven16To8 = physicalPlan.resultBits == 8 &&
                         resultLayout.getLaneStride() == 2;
    if (!isEven32To16 && !isPacked32To8 && !isEven16To8) {
      return rewriter.notifyMatchFailure(
          op, "unsupported dense lane_stride truncf result layout");
    }
    StringRef part = isPacked32To8 ? "P0" : "EVEN";
    if (failed(lowerDenseLaneStride(
            op, sourceParts, physicalPlan.resultTypes, part,
            physicalPlan.sourceIsPackedBF16x2, physicalPlan.sourceViewType,
            rewriter))) {
      return failure();
    }
    return true;
  }

  // Spine-scoped composite narrowing: the f32 source is deinterleaved=4 and the
  // narrow result keeps the same four parts, so one part-per-chunk
  // lowerDenseLaneStride (EVEN for 16-bit, P0 for 8-bit) is the whole
  // conversion and no pto.vor merge is emitted.
  FailureOr<bool> tryLowerSpineCompositeNarrowing(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool spineCompositeNarrowing =
        physicalPlan.sourceBits == 32 && sourceLayout && resultLayout &&
        sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 4 &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 4 &&
        ((physicalPlan.resultBits == 8 && resultLayout.getLaneStride() == 4) ||
         (physicalPlan.resultBits == 16 &&
          resultLayout.getLaneStride() == 2)) &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!spineCompositeNarrowing) {
      return false;
    }
    StringRef part = physicalPlan.resultBits == 8 ? StringRef("P0")
                                                  : StringRef("EVEN");
    if (failed(lowerDenseLaneStride(
            op, sourceParts, physicalPlan.resultTypes, part,
            physicalPlan.sourceIsPackedBF16x2, physicalPlan.sourceViewType,
            rewriter))) {
      return failure();
    }
    return true;
  }

  // Dense factor-N narrowing for the residual truncf shapes the composite
  // forms above did not claim.
  LogicalResult lowerGenericTruncNarrowing(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncFNarrowingPlan> narrowingPlan = buildNarrowingPlan(
        op, physicalPlan.sourceBits, physicalPlan.resultBits, resultLayout,
        sourceParts.size(), resultTypes.size(), rewriter);
    if (failed(narrowingPlan)) {
      return failure();
    }
    StringAttr rnd = rewriter.getStringAttr(getTruncFRoundMode(
        op, physicalPlan.resultTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    return lowerNarrow(op, sourceParts, physicalPlan.resultTypes,
                       narrowingPlan->parts, narrowingPlan->sourceFactor,
                       narrowingPlan->resultLaneStride,
                       physicalPlan.sourceViewType,
                       physicalPlan.sourceIsPackedBF16x2, rnd, sat, rewriter);
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncFPhysicalPlan> physicalPlan =
        buildPhysicalPlan(op, sourceParts, resultTypes, rewriter);
    if (failed(physicalPlan)) {
      return failure();
    }
    bool unsupportedGroupSlotLayout = sourceLayout && sourceLayout.isGroupSlots();
    if (unsupportedGroupSlotLayout) {
      return rewriter.notifyMatchFailure(
          op, "group-slot layout for non-f32 truncf not supported");
    }
    FailureOr<bool> sameWidth = tryLowerSameWidthTrunc(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(sameWidth)) {
      return failure();
    }
    if (*sameWidth) {
      return success();
    }
    FailureOr<bool> denseLaneStride = tryLowerDenseLaneStrideTrunc(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(denseLaneStride)) {
      return failure();
    }
    if (*denseLaneStride) {
      return success();
    }
    FailureOr<bool> spineCompositeNarrowing = tryLowerSpineCompositeNarrowing(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(spineCompositeNarrowing)) {
      return failure();
    }
    if (*spineCompositeNarrowing) {
      return success();
    }
    return lowerGenericTruncNarrowing(op, sourceParts, resultTypes,
                                      *physicalPlan, resultLayout, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMITruncFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    Type sourceElementType = sourceVMIType.getElementType();
    Type resultElementType = resultVMIType.getElementType();
    if (hasUnsupportedPackedTruncFConversion(sourceElementType,
                                             resultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported packed fp-to-fp truncf conversion");
    }
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (hasGroupSlotTruncFLayouts(sourceLayout, resultLayout)) {
      return lowerGroupSlotTrunc(op, sourceParts, resultTypes, sourceLayout,
                                 resultLayout, sourceVMIType, resultVMIType,
                                 rewriter);
    }

    return lowerNonGroupSlotTrunc(op, sourceParts, resultTypes, sourceLayout,
                                  resultLayout, rewriter);
  }
};
