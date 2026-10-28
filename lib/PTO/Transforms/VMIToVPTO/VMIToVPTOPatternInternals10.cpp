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

// vunzip/vzip split or rejoin a half-width part into one or a pair of wide
// parts, so the per-part chunk counts they accept are the unit count and this
// pair size.
constexpr size_t kSplitWidePartsPerHalf = 2;

static bool isBF16ReduceResultElementCount(int64_t elementCount) {
  return elementCount == 2 || elementCount == 4 || elementCount == 8;
}

//===----------------------------------------------------------------------===//
// Masked conversions (issue #1671)
//
// The logical VMI conversion predicate ZEROES every inactive lane
// (docs/isa/vmi-isa/03-eltwise-compute.md:23) and the physical pto.vcvt
// predicate has the same zero-inactive contract.  The predicate therefore rides
// on the conversion itself: every sub-path below attaches the authored mask to
// the pto.vcvt it emits (or to a predicate companion when the sub-path has to
// re-lay-out the source first).  Nothing clears the source and nothing
// re-selects on the result.
//
// The VMI verifier forces the mask layout to equal the source layout, so the
// converted mask part and the converted source part describe the same lanes.
// A lane-strided source changes the *physical* predicate granularity: a mask
// whose layout carries lane_stride = W is lowered with granularity
// min(element bits * W, 32), i.e. one predicate bit per carrier group.  That
// granularity *is* the element-granularity view pto.vcvt consumes: predicate
// bit positions track the byte offset of a lane in the predicate register, so
// the carrier-group bit of group g already sits on the bit of the strided
// element lane W*g the conversion reads.  Reinterpreting the carrier predicate
// at the source-element granularity is therefore the complete mapping; a
// second predicate interleave would shift every lane by another factor of W
// and leave the alternating lanes of the carrier inactive.
//===----------------------------------------------------------------------===//

/// Predicate granularity pto.vcvt consumes for a source element of
/// elementBits (the conversion family never addresses wider than 32 bits).
static StringRef getConversionMaskGranularity(unsigned elementBits)
{
    if (elementBits <= kElementBits8) {
        return "b8";
    }
    if (elementBits <= kElementBits16) {
        return "b16";
    }
    return "b32";
}

/// Re-express one carrier-group predicate part at the element granularity
/// pto.vcvt requires.  `sourceElementBits` is the element width of the vector
/// the conversion actually consumes.
///
/// A lane-strided mask is materialized with granularity element_bits * W, one
/// predicate bit per carrier group, and that bit already sits on the
/// element-granularity bit of the lane the group carries: predicate bit
/// positions track the byte offset of the lane in the predicate register, so
/// group g covers the same strided lane W*g on both granularities.  The
/// pbitcast is therefore the whole mapping; interleaving the bits with false
/// would shift the mask one extra time and deactivate alternating lanes.
static FailureOr<Value> materializeConversionMaskPart(
    Location loc, Value carrierMask, unsigned sourceElementBits, PatternRewriter& rewriter)
{
    auto carrierType = dyn_cast<MaskType>(carrierMask.getType());
    if (!carrierType) {
        return failure();
    }
    int64_t carrierBits = getMaskGranularityBits(carrierType.getGranularity());
    int64_t targetBits = getMaskGranularityBits(getConversionMaskGranularity(sourceElementBits));
    bool unsupportedGranularity =
        carrierBits == 0 || targetBits == 0 || carrierBits < targetBits || carrierBits % targetBits != 0;
    if (unsupportedGranularity) {
        return failure();
    }
    if (carrierBits == targetBits) {
        return carrierMask;
    }
    auto targetType = MaskType::get(rewriter.getContext(), getConversionMaskGranularity(sourceElementBits));
    return rewriter.create<PbitcastOp>(loc, targetType, carrierMask).getResult();
}

/// Interleave the predicate with false so bit 2*i + 1 of each result carries
/// bit i of `mask`.  This is the predicate companion of the vintlv+bitcast
/// same-layout BF16 widening: the low/high carriers hold the authored source
/// lanes in odd positions, so their predicates must hold the authored mask
/// lanes in odd positions too.
static FailureOr<std::pair<Value, Value>> materializeInterleavedPredicatePair(
    Location loc, Value mask, PatternRewriter& rewriter)
{
    auto maskType = dyn_cast<MaskType>(mask.getType());
    if (!maskType) {
        return failure();
    }
    FailureOr<Value> allFalse = createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);
    if (failed(allFalse)) {
        return failure();
    }
    return createPredicateIntlv(loc, maskType, maskType, *allFalse, mask, rewriter);
}

/// Map the optional conversion predicate onto the physical source chunks at
/// the element granularity of `sourceElementBits`.  Returns an empty list when
/// the cast carries no predicate, so unmasked lowerings stay byte-identical.
template <typename OpTy>
static FailureOr<SmallVector<Value>> applyConversionPredicate(
    OpTy op, ValueRange sourceParts, ValueRange maskParts, unsigned sourceElementBits, OneToNPatternRewriter& rewriter)
{
    if (!op.getMask()) {
        return SmallVector<Value>();
    }
    bool arityMismatch = maskParts.size() != sourceParts.size();
    if (arityMismatch) {
        return rewriter.notifyMatchFailure(
            op, "conversion predicate must have one physical mask part per "
                "physical source part");
    }
    SmallVector<Value> masks;
    masks.reserve(maskParts.size());
    for (Value maskPart : maskParts) {
        FailureOr<Value> mask = materializeConversionMaskPart(op.getLoc(), maskPart, sourceElementBits, rewriter);
        if (failed(mask)) {
            return rewriter.notifyMatchFailure(
                op, "failed to express the conversion predicate at the source "
                    "element granularity");
        }
        masks.push_back(*mask);
    }
    return masks;
}

/// Lower a cast whose conversion predicate is spread onto the physical source
/// chunks.  `lower` receives the converted result types and the predicate parts:
/// empty for an unmasked cast, so its lowering stays byte-identical.  Every cast
/// pattern shares this prologue.
template <typename OpTy, typename LowerFn>
static LogicalResult lowerCastWithConversionPredicate(
    OpTy op, ValueRange sourceParts, ValueRange maskOperands, TypeConverter& typeConverter,
    OneToNPatternRewriter& rewriter, LowerFn&& lower)
{
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    FailureOr<SmallVector<Value>> masks = applyConversionPredicate(op, sourceParts, maskOperands, sourceBits, rewriter);
    if (failed(masks)) {
        return failure();
    }
    return lowerWithConvertedResultTypes(
        op, 0, typeConverter, [&](ArrayRef<Type> resultTypes) { return lower(resultTypes, ValueRange(*masks)); });
}

/// Predicate a conversion consumes for one source chunk: the authored mask when
/// the cast carries one, otherwise the all-true seed the sub-path materialized
/// once for the whole conversion (exactly as the unmasked lowering always did).
static FailureOr<Value> getConversionMaskForChunk(ValueRange masks, size_t index, Value unmaskedSeed)
{
    if (!masks.empty()) {
        if (index >= masks.size()) {
            return failure();
        }
        return masks[index];
    }
    if (!unmaskedSeed) {
        return failure();
    }
    return unmaskedSeed;
}

/// All-true predicate the unmasked sub-paths pass to every pto.vcvt.
static FailureOr<Value> createUnmaskedConversionSeed(Location loc, VRegType seedType, PatternRewriter& rewriter)
{
    return createAllTrueMaskForVReg(loc, seedType, rewriter);
}

struct OneToNVMIExtFOpPattern : OneToNOpConversionPattern<VMIExtFOp> {
  using OneToNOpConversionPattern<VMIExtFOp>::OneToNOpConversionPattern;

private:
  struct ExtFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
  };

  struct BF16PairWideningPlan {
    VRegType sourceType;
    size_t resultsPerSource;
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
            rewriter.getContext(), resultTypes.front().getElementCount() * kPairWidth,
            BFloat16Type::get(rewriter.getContext()));
    }
    return ResultViewPlan{isPackedBF16x2, vcvtResultType};
  }

  FailureOr<BF16PairWideningPlan> buildBF16PairWideningPlan(
      VMIExtFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    bool invalidArity =
        sourceParts.empty() || resultTypes.empty() ||
        resultTypes.size() % sourceParts.size() != 0;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported BF16 pair widening arity");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    bool invalidSourceType =
        !sourceType || !sourceType.getElementType().isBF16();
    if (invalidSourceType) {
      return rewriter.notifyMatchFailure(
          op, "BF16 pair widening requires BF16 physical sources");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatchedSourceType = sourcePart.getType() != sourceType;
      if (mismatchedSourceType) {
        return rewriter.notifyMatchFailure(
            op, "BF16 pair widening sources must have matching types");
      }
    }
    for (VRegType resultType : resultTypes) {
      if (!resultType.getElementType().isF32()) {
        return rewriter.notifyMatchFailure(
            op, "BF16 pair widening requires F32 physical results");
      }
    }
    size_t resultsPerSource = resultTypes.size() / sourceParts.size();
    if (resultsPerSource != 1 && resultsPerSource != kSplitWidePartsPerHalf) {
      return rewriter.notifyMatchFailure(
          op, "BF16 pair widening expects one or two results per source");
    }

    return BF16PairWideningPlan{sourceType, resultsPerSource};
  }

  LogicalResult lowerBF16PairWideningResults(
      VMIExtFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks,
      const BF16PairWideningPlan& plan, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<Value> zero = createZeroVector(op.getLoc(), plan.sourceType, rewriter);
      if (failed(zero)) {
          return rewriter.notifyMatchFailure(op, "failed to build BF16 pair widening zero vector");
      }
      SmallVector<std::pair<Value, Value>> interleaved;
      interleaved.reserve(sourceParts.size());
      for (Value sourcePart : sourceParts) {
          auto pair =
              rewriter.create<VintlvOp>(op.getLoc(), TypeRange{plan.sourceType, plan.sourceType}, *zero, sourcePart);
          interleaved.emplace_back(pair.getLow(), pair.getHigh());
      }

      // A masked widening keeps the interleaved carrier, whose odd lanes carry the
      // authored source lanes, and turns the predicate into the ODD-part
      // conversion operand.  The pto.vcvt zero-inactive contract then leaves the
      // inactive result lanes zero.  Unmasked widening stays the bitcast below.
      SmallVector<std::pair<Value, Value>> interleavedMasks;
      if (!masks.empty()) {
          interleavedMasks.reserve(masks.size());
          for (Value mask : masks) {
              FailureOr<std::pair<Value, Value>> paired =
                  materializeInterleavedPredicatePair(op.getLoc(), mask, rewriter);
              if (failed(paired)) {
                  return rewriter.notifyMatchFailure(op, "failed to interleave the BF16 pair widening predicate");
              }
              interleavedMasks.push_back(*paired);
          }
      }

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      // Result physical parts are source-major: both widened halves of the first
      // source come before the halves of the next source. Iterating half-major
      // here transposes multi-source results and shifts every part after the
      // first source.
      for (size_t sourceIndex = 0; sourceIndex < sourceParts.size(); ++sourceIndex) {
          for (size_t half = 0; half < plan.resultsPerSource; ++half) {
              Value carrier = half == 0 ? interleaved[sourceIndex].first : interleaved[sourceIndex].second;
              VRegType resultType = resultTypes[sourceIndex * plan.resultsPerSource + half];
              if (!interleavedMasks.empty()) {
                  Value mask = half == 0 ? interleavedMasks[sourceIndex].first : interleavedMasks[sourceIndex].second;
                  results.push_back(createVcvtResult(
                      op.getLoc(), resultType, carrier, mask, /*rnd=*/nullptr,
                      /*sat=*/nullptr, rewriter.getStringAttr("ODD"),
                      /*resultIsPackedBF16x2=*/false, resultType, rewriter));
                  continue;
              }
              FailureOr<Value> result = bitcastVReg(op.getLoc(), carrier, resultType, rewriter);
              if (failed(result)) {
                  return rewriter.notifyMatchFailure(op, "failed to bitcast BF16 pair widening result");
              }
              results.push_back(*result);
          }
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
  }

  LogicalResult lowerSameLayoutBF16PairWidening(
      VMIExtFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      FailureOr<BF16PairWideningPlan> plan = buildBF16PairWideningPlan(op, sourceParts, resultTypes, rewriter);
      if (failed(plan)) {
          return failure();
      }
      return lowerBF16PairWideningResults(op, sourceParts, resultTypes, masks, *plan, rewriter);
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
      VMIExtFOp op, OneToNPatternRewriter& rewriter, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      ValueRange masks, StringRef part, bool resultIsPackedBF16x2, VRegType vcvtResultVRegType) const
  {
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      Value unmaskedSeed;
      if (masks.empty()) {
          auto seedType = dyn_cast<VRegType>(sourceParts.front().getType());
          if (!seedType) {
              return rewriter.notifyMatchFailure(op, "unsupported lane-stride extf source type");
          }
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), seedType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build a lane-stride conversion predicate");
          }
          unmaskedSeed = *seed;
      }
      for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
          auto [sourcePart, resultType] = sourceAndResult;
          FailureOr<Value> mask = getConversionMaskForChunk(masks, index, unmaskedSeed);
          if (failed(mask)) {
              return rewriter.notifyMatchFailure(op, "failed to build a lane-stride conversion predicate");
          }
          results.push_back(createVcvtResult(
              op.getLoc(), resultType, sourcePart, *mask, /*rnd=*/nullptr,
              /*sat=*/nullptr, rewriter.getStringAttr(part), resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
  }

  LogicalResult lowerPackedLaneStride2(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      static constexpr StringRef kPacked2Parts[] = {"P0", "P2"};
      ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
      return lowerFactor(
          op, rewriter, sourceParts, plan.resultTypes, kPacked2Parts, kPairWidth, masks, viewPlan.isPackedBF16x2,
          viewPlan.vcvtResultType);
  }

  LogicalResult lowerFactor(
      VMIExtFOp op, OneToNPatternRewriter& rewriter, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      ArrayRef<StringRef> parts, int64_t factor, ValueRange masks, bool resultIsPackedBF16x2,
      VRegType vcvtResultVRegType) const
  {
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      Value unmaskedSeed;
      if (masks.empty()) {
          auto seedType = dyn_cast<VRegType>(sourceParts.front().getType());
          if (!seedType) {
              return rewriter.notifyMatchFailure(op, "unsupported factor-N extf source");
          }
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), seedType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build a factor-N conversion predicate");
          }
          unmaskedSeed = *seed;
      }
      for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
          for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
              VRegType resultType = resultTypes[partIndex * sourceParts.size() + chunkIndex];
              FailureOr<Value> mask = getConversionMaskForChunk(masks, chunkIndex, unmaskedSeed);
              if (failed(mask)) {
                  return rewriter.notifyMatchFailure(op, "failed to build a factor-N conversion predicate");
              }
              results.push_back(createVcvtResult(
                  op.getLoc(), resultType, sourcePart, *mask, /*rnd=*/nullptr,
                  /*sat=*/nullptr, rewriter.getStringAttr(parts[partIndex]), resultIsPackedBF16x2, vcvtResultVRegType,
                  rewriter));
          }
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
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
      if (sourceBits == kElementBits16 && resultPartCount == kPairWidth * sourcePartCount) {
          static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
          return ExtFFactorPlan{ArrayRef<StringRef>(kEvenOddParts), kPairWidth};
      }
      if (sourceBits == kElementBits8 && resultPartCount == kQuadWidth * sourcePartCount) {
          static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
          return ExtFFactorPlan{ArrayRef<StringRef>(kPacked4Parts), kQuadWidth};
      }
    return rewriter.notifyMatchFailure(
        op, "unsupported physical extf source/result width relation");
  }

  // Spine-scoped composite widening: the narrow source is deinterleaved=4 with
  // each physical part carrying one modulo-4 lane partition, so one
  // part-per-chunk vcvt (EVEN for 16-bit, P0 for 8-bit) is the whole
  // conversion and no pto.vor merge is needed.
  FailureOr<bool> tryLowerSpineCompositeWidening(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks, unsigned sourceBits,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, const ResultViewPlan& viewPlan,
      OneToNPatternRewriter& rewriter) const
  {
      bool spineCompositeWidening =
          sourceLayout && resultLayout && sourceLayout.isDeinterleaved() &&
          (sourceLayout.getFactor() == 2 || sourceLayout.getFactor() == 4) && resultLayout.isDeinterleaved() &&
          (resultLayout.getFactor() == 2 || resultLayout.getFactor() == 4) && resultLayout.getLaneStride() == 1 &&
          ((sourceBits == 8 && sourceLayout.getLaneStride() == 4) ||
           (sourceBits == 16 && sourceLayout.getLaneStride() == 2)) &&
          plan.resultTypes.size() == sourceParts.size();
      if (!spineCompositeWidening) {
          return false;
      }
      // The part selector follows the widening relation (packed four-byte or
      // pair), not the layout factor: the packed verifier only accepts P0..P3.
      StringRef part = sourceBits == 8 ? StringRef("P0") : StringRef("EVEN");
      if (failed(lowerLaneStride(
              op, rewriter, sourceParts, plan.resultTypes, masks, part, viewPlan.isPackedBF16x2,
              viewPlan.vcvtResultType))) {
          return failure();
      }
      return true;
  }

  FailureOr<bool> tryLowerDenseLaneStrideExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, const ResultViewPlan& viewPlan, OneToNPatternRewriter& rewriter) const
  {
      ExtFDenseLaneStridePlan laneStridePlan =
          buildDenseLaneStridePlan(plan, sourceLayout, resultLayout, sourceParts.size());
      if (!laneStridePlan.applies) {
          return false;
      }
      if (failed(lowerLaneStride(
              op, rewriter, sourceParts, plan.resultTypes, masks, laneStridePlan.part, viewPlan.isPackedBF16x2,
              viewPlan.vcvtResultType))) {
          return failure();
      }
      return true;
  }

  FailureOr<bool> tryLowerPackedLaneStride2ExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks, VMILayoutAttr sourceLayout,
      OneToNPatternRewriter& rewriter) const
  {
      // 8-bit sources stored with lane_stride = 2 (UNPK_B8) keep valid bytes on
      // even lanes. P0 plus P2 cover them; P1/P3 are zero-fill gaps. This
      // applies both to the packed f4 formats (F4E1M2x2 and F4E2M1x2) and to
      // dense FP8/HiF8 alike. The f4 encoding does not affect the byte-lane
      // selection the {P0, P2} pair performs, so both packed formats take it.
      Type sourceElementType = plan.sourceType.getElementType();
      bool packedLaneStride2 = sourceLayout && sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 2 &&
                               (pto::isPTOFloat4PackedType(sourceElementType) ||
                                pto::isPTOFloat8Type(sourceElementType) || pto::isPTOHiFloat8Type(sourceElementType)) &&
                               plan.resultTypes.size() == 2 * sourceParts.size();
      if (!packedLaneStride2) {
          return false;
      }
      if (failed(lowerPackedLaneStride2(op, sourceParts, plan, masks, rewriter))) {
          return failure();
      }
      return true;
  }

  FailureOr<bool> tryLowerSameLayoutBF16PairExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
      auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
      bool sameLayoutBF16PairWidening =
          isBF16SameLayoutCastPair(sourceVMIType, resultVMIType, sourceLayout, resultLayout) &&
          plan.resultTypes.front().getElementType().isF32();
      if (!sameLayoutBF16PairWidening) {
          return false;
      }
      if (failed(lowerSameLayoutBF16PairWidening(op, sourceParts, plan.resultTypes, masks, rewriter))) {
          return failure();
      }
      return true;
  }

  LogicalResult lowerGenericExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks, unsigned sourceBits,
      const ResultViewPlan& viewPlan, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<ExtFFactorPlan> factorPlan =
          buildFactorPlan(op, sourceBits, sourceParts.size(), plan.resultTypes.size(), rewriter);
      if (failed(factorPlan)) {
          return failure();
      }
      return lowerFactor(
          op, rewriter, sourceParts, plan.resultTypes, factorPlan->parts, factorPlan->factor, masks,
          viewPlan.isPackedBF16x2, viewPlan.vcvtResultType);
  }

  LogicalResult lowerPhysicalExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan& plan, ValueRange masks, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      unsigned sourceBits = pto::getPTOStorageElemBitWidth(plan.sourceType.getElementType());
      // A packed bf16x2 physical result cannot be produced directly by
      // pto.vcvt (classifyVcvtElemType has no BF16x2 branch); the widest native
      // f4 conversion result element is bf16. Build the bf16 view type (2 bf16
      // lanes per bf16x2 lane) and reinterpret each vcvt result with a
      // physical-noop VbitcastOp, mirroring the source-side reinterpret in
      // OneToNVMITruncFOpPattern (viewVcvtSource).
      ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
      FailureOr<bool> denseLaneStride =
          tryLowerDenseLaneStrideExtF(op, sourceParts, plan, masks, sourceLayout, resultLayout, viewPlan, rewriter);
      if (failed(denseLaneStride)) {
          return failure();
      }
      if (*denseLaneStride) {
          return success();
      }

      FailureOr<bool> packedLaneStride2 =
          tryLowerPackedLaneStride2ExtF(op, sourceParts, plan, masks, sourceLayout, rewriter);
      if (failed(packedLaneStride2)) {
          return failure();
      }
      if (*packedLaneStride2) {
          return success();
      }

      FailureOr<bool> sameLayoutBF16Pair =
          tryLowerSameLayoutBF16PairExtF(op, sourceParts, plan, masks, sourceLayout, resultLayout, rewriter);
      if (failed(sameLayoutBF16Pair)) {
          return failure();
      }
      if (*sameLayoutBF16Pair) {
          return success();
      }

      FailureOr<bool> spineCompositeWidening = tryLowerSpineCompositeWidening(
          op, sourceParts, plan, masks, sourceBits, sourceLayout, resultLayout, viewPlan, rewriter);
      if (failed(spineCompositeWidening)) {
          return failure();
      }
      if (*spineCompositeWidening) {
          return success();
      }

      return lowerGenericExtF(op, sourceParts, plan, masks, sourceBits, viewPlan, rewriter);
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
    unsigned sourceBits = pto::getPTOStorageElemBitWidth(plan->sourceType.getElementType());
    FailureOr<SmallVector<Value>> masks =
        applyConversionPredicate(op, input->sourceParts, adaptor.getMask(), sourceBits, rewriter);
    if (failed(masks)) {
        return failure();
    }
    return lowerPhysicalExtF(op, input->sourceParts, *plan, *masks, sourceLayout, resultLayout, rewriter);
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
  OneToNVMITruncFOpPattern(VMIToVPTOTypeConverter &typeConverter,
                           MLIRContext *context, bool enableVmiFastmath)
      : OneToNOpConversionPattern<VMITruncFOp>(typeConverter, context),
        enableVmiFastmath(enableVmiFastmath) {}

private:
  const bool enableVmiFastmath;

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

  struct BF16PairNarrowingPlan {
    VRegType sourceType;
    VRegType carrierType;
    size_t sourcesPerResult;
  };

  struct GroupSlotBF16ReducePlan {
    VRegType sourceType;
    VRegType carrierType;
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
    if (sourceBits != kElementBits32 && sourceBits != kElementBits16) {
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
            rewriter.getContext(), sourceType->getElementCount() * kPairWidth,
            BFloat16Type::get(rewriter.getContext()));
    }
    return TruncFPhysicalPlan{*sourceType, std::move(*resultVRegTypes),
                              sourceViewType, sourceBits, resultBits,
                              sourceIsPackedBF16x2};
  }

  LogicalResult lowerDenseLaneStride(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks, StringRef part,
      bool sourceIsPackedBF16x2, VRegType vcvtSourceVRegType, OneToNPatternRewriter& rewriter) const
  {
      StringAttr rnd = rewriter.getStringAttr(getTruncFRoundMode(op, resultTypes.front().getElementType()));
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      StringAttr partAttr = rewriter.getStringAttr(part);
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      Value unmaskedSeed;
      if (masks.empty()) {
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), vcvtSourceVRegType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
          }
          unmaskedSeed = *seed;
      }
      for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
          auto [sourcePart, resultType] = sourceAndResult;
          FailureOr<Value> mask = getConversionMaskForChunk(masks, index, unmaskedSeed);
          if (failed(mask)) {
              return rewriter.notifyMatchFailure(op, "failed to build a lane-stride truncf predicate");
          }
          results.push_back(
              rewriter
                  .create<VcvtOp>(
                      op.getLoc(), resultType,
                      makeVcvtSourceView(op.getLoc(), sourcePart, sourceIsPackedBF16x2, vcvtSourceVRegType, rewriter),
                      *mask, rnd, sat, partAttr)
                  .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
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
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks, VRegType sourceViewType,
      StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      StringAttr rnd = rewriter.getStringAttr(getTruncFRoundMode(op, resultTypes.front().getElementType()));
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      Value unmaskedSeed;
      if (masks.empty()) {
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), sourceViewType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
          }
          unmaskedSeed = *seed;
      }
      for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
          auto [sourcePart, resultType] = sourceAndResult;
          FailureOr<Value> mask = getConversionMaskForChunk(masks, index, unmaskedSeed);
          if (failed(mask)) {
              return rewriter.notifyMatchFailure(op, "failed to build a same-width truncf predicate");
          }
          results.push_back(rewriter
                                .create<VcvtOp>(
                                    op.getLoc(), resultType, sourcePart, *mask, rnd, sat,
                                    /*part=*/nullptr)
                                .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
  }

  FailureOr<Value> buildNarrowTruncResult(
      VMITruncFOp op, ValueRange sourceParts, VRegType resultType, ArrayRef<StringRef> allParts, int64_t chunkIndex,
      int64_t sourceFactor, int64_t resultLaneStride, VRegType sourceViewType, bool sourceIsPackedBF16x2,
      ValueRange masks, Value unmaskedSeed, StringAttr rnd, StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      if (sourceFactor <= 0) {
          return rewriter.notifyMatchFailure(op, "narrow truncf requires a positive source factor");
      }
      int64_t safeSourceFactor = sourceFactor;
      FailureOr<Value> resultMask = createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
      if (failed(resultMask)) {
          return failure();
      }
      SmallVector<Value> partials;
      partials.reserve(safeSourceFactor);
      for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
          size_t sourceIndex = partIndex * (sourceParts.size() / safeSourceFactor) + chunkIndex;
          Value sourcePart = sourceParts[sourceIndex];
          bool hasIndexedPart = partIndex * resultLaneStride < static_cast<int64_t>(allParts.size());
          StringRef part = hasIndexedPart ? allParts[partIndex * resultLaneStride] : allParts[partIndex];
          // Every partial converts its own source chunk, so it carries that
          // chunk's predicate.  The vor merge below is a pure lane combine: the
          // inactive lanes of each partial are already zero.
          FailureOr<Value> mask = getConversionMaskForChunk(masks, sourceIndex, unmaskedSeed);
          if (failed(mask)) {
              return rewriter.notifyMatchFailure(op, "failed to build a narrow truncf predicate");
          }
          partials.push_back(
              rewriter
                  .create<VcvtOp>(
                      op.getLoc(), resultType,
                      makeVcvtSourceView(op.getLoc(), sourcePart, sourceIsPackedBF16x2, sourceViewType, rewriter),
                      *mask, rnd, sat, rewriter.getStringAttr(part))
                  .getResult());
      }
      Value merged = partials.front();
      for (Value partial : llvm::drop_begin(partials)) {
          merged = rewriter.create<VorOp>(op.getLoc(), resultType, merged, partial, *resultMask).getResult();
      }
      return merged;
  }

  LogicalResult lowerNarrow(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks,
      ArrayRef<StringRef> allParts, int64_t sourceFactor, int64_t resultLaneStride, VRegType sourceViewType,
      bool sourceIsPackedBF16x2, StringAttr rnd, StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      if (sourceFactor <= 0 || resultLaneStride <= 0 ||
          sourceParts.size() != static_cast<size_t>(sourceFactor) * resultTypes.size()) {
          return rewriter.notifyMatchFailure(op, "unsupported physical truncf source/result arity relation");
      }
      Value unmaskedSeed;
      if (masks.empty()) {
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), sourceViewType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
          }
          unmaskedSeed = *seed;
      }
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
          FailureOr<Value> result = buildNarrowTruncResult(
              op, sourceParts, resultType, allParts, chunkIndex, sourceFactor, resultLaneStride, sourceViewType,
              sourceIsPackedBF16x2, masks, unmaskedSeed, rnd, sat, rewriter);
          if (failed(result)) {
              return rewriter.notifyMatchFailure(op, "failed to build truncf result mask");
          }
          results.push_back(*result);
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
  }

  FailureOr<TruncFNarrowingPlan> buildNarrowingPlan(
      VMITruncFOp op, unsigned sourceBits, unsigned resultBits,
      VMILayoutAttr resultLayout, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (resultBits * kPairWidth == sourceBits) {
        static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
        parts = kEvenOddParts;
        factor = kPairWidth;
    } else if (resultBits * kQuadWidth == sourceBits) {
        static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
        parts = kPacked4Parts;
        factor = kQuadWidth;
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

  FailureOr<BF16PairNarrowingPlan> buildBF16PairNarrowingPlan(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    bool invalidSourceType =
        !sourceType || !sourceType.getElementType().isF32();
    if (invalidSourceType) {
      return rewriter.notifyMatchFailure(
          op, "BF16 pair narrowing requires F32 physical sources");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatchedSourceType = sourcePart.getType() != sourceType;
      if (mismatchedSourceType) {
        return rewriter.notifyMatchFailure(
            op, "BF16 pair narrowing sources must have matching types");
      }
    }
    for (VRegType resultType : resultTypes) {
      if (!resultType.getElementType().isBF16()) {
        return rewriter.notifyMatchFailure(
            op, "BF16 pair narrowing requires BF16 physical results");
      }
    }
    bool invalidPairArity =
        sourceParts.size() != resultTypes.size() &&
        sourceParts.size() != kSplitWidePartsPerHalf * resultTypes.size();
    if (invalidPairArity) {
      return rewriter.notifyMatchFailure(
          op, "BF16 pair narrowing expects one or two sources per result");
    }
    Type carrierElementType = IntegerType::get(
        rewriter.getContext(), kElementBits16,
        IntegerType::SignednessSemantics::Signless);
    VRegType carrierType = VRegType::get(
        rewriter.getContext(), sourceType.getElementCount() * kPairWidth,
        carrierElementType);
    size_t sourcesPerResult = sourceParts.size() / resultTypes.size();
    return BF16PairNarrowingPlan{sourceType, carrierType, sourcesPerResult};
  }

  LogicalResult lowerBF16PairNarrowingResults(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      const BF16PairNarrowingPlan &plan,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> carrierSources;
    carrierSources.reserve(sourceParts.size());
    for (Value sourcePart : sourceParts) {
      FailureOr<Value> carrier =
          bitcastVReg(op.getLoc(), sourcePart, plan.carrierType, rewriter);
      if (failed(carrier)) {
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast BF16 pair narrowing source");
      }
      carrierSources.push_back(*carrier);
    }

    Value zero;
    if (plan.sourcesPerResult == 1) {
      FailureOr<Value> zeroValue =
          createZeroVector(op.getLoc(), plan.carrierType, rewriter);
      if (failed(zeroValue)) {
        return rewriter.notifyMatchFailure(
            op, "failed to build BF16 pair narrowing zero vector");
      }
      zero = *zeroValue;
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (size_t resultIndex = 0; resultIndex < resultTypes.size();
         ++resultIndex) {
      size_t sourceIndex = resultIndex * plan.sourcesPerResult;
      Value rhs = plan.sourcesPerResult == kSplitWidePartsPerHalf
                      ? carrierSources[sourceIndex + 1]
                      : zero;
      auto pair = rewriter.create<VdintlvOp>(
          op.getLoc(), TypeRange{plan.carrierType, plan.carrierType},
          carrierSources[sourceIndex], rhs);
      FailureOr<Value> result = bitcastVReg(
          op.getLoc(), pair.getHigh(), resultTypes[resultIndex], rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast BF16 pair narrowing result");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  // A contiguous f32 vector with two physical source parts lowers to one
  // contiguous bf16 result as follows: convert each contiguous source part
  // into an EVEN bf16 carrier, bitcast both carriers to i16, then take the
  // low vdintlv lane stream.  That stream is the concatenation of the two
  // converted EVEN lane streams in logical order, so it is the required
  // f32 contiguous -> bf16 contiguous result.  Using EVEN for one source and
  // ODD for the other is only valid when the source parts are already the
  // logical even/odd partitions (deinterleaved=2), not for contiguous parts.
  FailureOr<Value> lowerPairedSameLayoutBF16NarrowingResult(
      VMITruncFOp op, Value firstSource, Value secondSource, VRegType resultType, Type carrierElementType,
      Value firstMask, Value secondMask, StringAttr rnd, StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      VRegType carrierType = VRegType::get(rewriter.getContext(), resultType.getElementCount(), carrierElementType);
      // Each partial converts one source chunk and carries that chunk's
      // predicate; the vdintlv below only moves the converted lanes.
      Value even0 =
          rewriter
              .create<VcvtOp>(op.getLoc(), resultType, firstSource, firstMask, rnd, sat, rewriter.getStringAttr("EVEN"))
              .getResult();
      Value even1 = rewriter
                        .create<VcvtOp>(
                            op.getLoc(), resultType, secondSource, secondMask, rnd, sat, rewriter.getStringAttr("EVEN"))
                        .getResult();
      FailureOr<Value> carrier0 = bitcastVReg(op.getLoc(), even0, carrierType, rewriter);
      FailureOr<Value> carrier1 = bitcastVReg(op.getLoc(), even1, carrierType, rewriter);
      bool invalidCarriers = failed(carrier0) || failed(carrier1);
      if (invalidCarriers) {
          return rewriter.notifyMatchFailure(op, "failed to bitcast same-layout BF16 narrowing carriers");
      }
      auto pair = rewriter.create<VdintlvOp>(op.getLoc(), TypeRange{carrierType, carrierType}, *carrier0, *carrier1);
      FailureOr<Value> result = bitcastVReg(op.getLoc(), pair.getLow(), resultType, rewriter);
      if (failed(result)) {
          return rewriter.notifyMatchFailure(op, "failed to bitcast same-layout BF16 narrowing result");
      }
      return *result;
  }

  LogicalResult lowerPairedSameLayoutBF16NarrowingWithVcvt(
      VMITruncFOp op, ValueRange orderedSources, ValueRange orderedMasks, ArrayRef<VRegType> resultTypes,
      VRegType sourceType, OneToNPatternRewriter& rewriter) const
  {
      bool invalidArity = orderedSources.size() != kSplitWidePartsPerHalf * resultTypes.size();
      if (invalidArity) {
          return rewriter.notifyMatchFailure(op, "unsupported paired same-layout BF16 narrowing arity");
      }
      bool maskArityMismatch = !orderedMasks.empty() && orderedMasks.size() != orderedSources.size();
      if (maskArityMismatch) {
          return rewriter.notifyMatchFailure(op, "same-layout BF16 narrowing predicate arity mismatch");
      }
      StringAttr rnd = rewriter.getStringAttr(getTruncFRoundMode(op, resultTypes.front().getElementType()));
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      Type carrierElementType =
          IntegerType::get(rewriter.getContext(), kElementBits16, IntegerType::SignednessSemantics::Signless);
      Value unmaskedSeed;
      if (orderedMasks.empty()) {
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), sourceType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build same-layout BF16 narrowing mask");
          }
          unmaskedSeed = *seed;
      }
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
          size_t firstIndex = resultIndex;
          size_t secondIndex = resultIndex + resultTypes.size();
          FailureOr<Value> firstMask = getConversionMaskForChunk(orderedMasks, firstIndex, unmaskedSeed);
          FailureOr<Value> secondMask = getConversionMaskForChunk(orderedMasks, secondIndex, unmaskedSeed);
          bool missingMask = failed(firstMask) || failed(secondMask);
          if (missingMask) {
              return rewriter.notifyMatchFailure(op, "failed to build a paired same-layout BF16 predicate");
          }
          FailureOr<Value> result = lowerPairedSameLayoutBF16NarrowingResult(
              op, orderedSources[firstIndex], orderedSources[secondIndex], resultType, carrierElementType, *firstMask,
              *secondMask, rnd, sat, rewriter);
          if (failed(result)) {
              return failure();
          }
          results.push_back(*result);
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
  }

  LogicalResult lowerSameLayoutBF16PairNarrowingWithVcvt(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
      if (!sourceType || !sourceType.getElementType().isF32()) {
          return rewriter.notifyMatchFailure(op, "same-layout BF16 narrowing requires F32 physical sources");
      }
      bool matchingPhysicalArity = sourceParts.size() == resultTypes.size();
      if (matchingPhysicalArity) {
          // One source part per result part: each part is converted on its own.  For
          // a same-layout *split* pair (d(2)/d(4) or lane-stride 2/4 on both sides)
          // this is the honest lowering, because the result layout itself says the
          // converted lanes sit where the vcvt puts them.  The dense contiguous
          // `{c, c}` pair is the exception - there the same per-part vcvt writes
          // the lanes at lane_stride 2 into a value that claims to be dense - and it
          // is rejected by the caller before reaching this branch.
          return lowerDenseLaneStride(
              op, sourceParts, resultTypes, masks, "EVEN",
              /*sourceIsPackedBF16x2=*/false, sourceType, rewriter);
      }
      bool pairedPhysicalArity = sourceParts.size() == kSplitWidePartsPerHalf * resultTypes.size();
      if (!pairedPhysicalArity) {
          return rewriter.notifyMatchFailure(op, "unsupported same-layout BF16 narrowing physical arity");
      }

      // lowerNarrow stores sources part-major; transpose adjacent pairs into the
      // order used by the paired lowering (all part0s, then all part1s).
      SmallVector<Value> orderedSources;
      orderedSources.reserve(sourceParts.size());
      for (size_t i = 0; i < resultTypes.size(); ++i) {
          orderedSources.push_back(sourceParts[i * kSplitWidePartsPerHalf]);
      }
      for (size_t i = 0; i < resultTypes.size(); ++i) {
          orderedSources.push_back(sourceParts[i * kSplitWidePartsPerHalf + 1]);
      }
      // The predicate follows the same part-major source order.
      SmallVector<Value> orderedMasks;
      orderedMasks.reserve(masks.size());
      for (size_t i = 0; i < resultTypes.size() && !masks.empty(); ++i) {
          orderedMasks.push_back(masks[i * kSplitWidePartsPerHalf]);
      }
      for (size_t i = 0; i < resultTypes.size() && !masks.empty(); ++i) {
          orderedMasks.push_back(masks[i * kSplitWidePartsPerHalf + 1]);
      }
      return lowerPairedSameLayoutBF16NarrowingWithVcvt(
          op, orderedSources, orderedMasks, resultTypes, sourceType, rewriter);
  }

  FailureOr<bool> tryLowerSameLayoutBF16PairNarrowing(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes, ValueRange masks,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
      auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
      StringAttr rounding = op->getAttrOfType<StringAttr>("rounding");
      StringAttr saturate = op->getAttrOfType<StringAttr>("saturate");
      bool sameLayoutBF16PairNarrowing =
          sourceVMIType.getElementType().isF32() && resultVMIType.getElementType().isBF16() &&
          isBF16SameLayoutCastPair(sourceVMIType, resultVMIType, sourceLayout, resultLayout) && !sourceParts.empty() &&
          !resultTypes.empty();
      if (!sameLayoutBF16PairNarrowing) {
          return false;
      }

      // The vdintlv fast path moves data without a conversion, so it cannot
      // carry a predicate; a masked cast stays on the vcvt path below.
      bool useNaNRelaxedFastPath = masks.empty() && enableVmiFastmath && rounding && rounding.getValue() == "Z" &&
                                   saturate && saturate.getValue() == "NOSAT";
      if (!useNaNRelaxedFastPath) {
          // The matching-arity dense contiguous same-layout pair is only ever
          // produced densely by the NaN-relaxed fastmath lowering above.  A legal
          // layout assignment can no longer select it outside that state (candidate
          // enumeration is state-qualified), so this is a defensive backstop for IR
          // that bypasses both the candidate set and the up-front support
          // verification: refuse it loudly instead of writing the converted lanes at
          // lane_stride 2 into a value a consumer reads as dense (issue #1619).
          // A masked dense pair lands here too: the NaN-relaxed fast path is the
          // predicate-free vdintlv, so no state lowers this pair with a mask.
          if (requiresBF16NaNRelaxedLowering(sourceVMIType, resultVMIType, sourceLayout, resultLayout)) {
              op.emitError() << "a dense same-layout f32-to-bf16 narrowing (contiguous -> "
                                "contiguous) is only implemented by the NaN-relaxed fastmath "
                                "lowering, which requires "
                             << describeBF16NarrowingState(op, enableVmiFastmath);
              return failure();
          }
          if (failed(lowerSameLayoutBF16PairNarrowingWithVcvt(op, sourceParts, resultTypes, masks, rewriter))) {
              return failure();
          }
          return true;
      }
      FailureOr<BF16PairNarrowingPlan> plan = buildBF16PairNarrowingPlan(op, sourceParts, resultTypes, rewriter);
      if (failed(plan)) {
          return failure();
      }
      if (failed(lowerBF16PairNarrowingResults(op, sourceParts, resultTypes, *plan, rewriter))) {
          return failure();
      }
      return true;
  }

  FailureOr<GroupSlotBF16ReducePlan> buildGroupSlotBF16ReducePlan(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    bool invalidSourceType =
        !sourceType || !sourceType.getElementType().isF32();
    if (invalidSourceType) {
      return rewriter.notifyMatchFailure(
          op, "BF16 group-slot narrowing requires F32 physical sources");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatchedSourceType = sourcePart.getType() != sourceType;
      if (mismatchedSourceType) {
        return rewriter.notifyMatchFailure(
            op, "BF16 group-slot narrowing sources must have matching types");
      }
    }
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidResultType =
          !resultVRegType || !resultVRegType.getElementType().isBF16() ||
          resultVRegType.getElementCount() !=
              sourceType.getElementCount() * kPairWidth;
      if (invalidResultType) {
        return rewriter.notifyMatchFailure(
            op, "BF16 group-slot narrowing requires matching physical widths");
      }
    }
    Type carrierElementType = IntegerType::get(
        rewriter.getContext(), kElementBits16,
        IntegerType::SignednessSemantics::Signless);
    VRegType carrierType = VRegType::get(
        rewriter.getContext(), sourceType.getElementCount() * kPairWidth,
        carrierElementType);
    return GroupSlotBF16ReducePlan{sourceType, carrierType};
  }

  LogicalResult lowerGroupSlotBF16ReduceResults(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      const GroupSlotBF16ReducePlan &plan,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> zero =
        createZeroVector(op.getLoc(), plan.carrierType, rewriter);
    if (failed(zero)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build BF16 group-slot narrowing zero vector");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> carrier =
          bitcastVReg(op.getLoc(), sourcePart, plan.carrierType, rewriter);
      if (failed(carrier)) {
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast BF16 group-slot narrowing source");
      }
      auto pair = rewriter.create<VdintlvOp>(
          op.getLoc(), TypeRange{plan.carrierType, plan.carrierType}, *carrier,
          *zero);
      FailureOr<Value> result = bitcastVReg(
          op.getLoc(), pair.getHigh(), cast<VRegType>(physicalResultType),
          rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast BF16 group-slot narrowing result");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<bool> tryLowerGroupSlotBF16ReduceNarrowing(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    StringAttr rounding = op->getAttrOfType<StringAttr>("rounding");
    StringAttr saturate = op->getAttrOfType<StringAttr>("saturate");
    bool supportedLogicalShape =
        sourceVMIType.getElementType().isF32() &&
        resultVMIType.getElementType().isBF16() &&
        sourceVMIType.getElementCount() == resultVMIType.getElementCount() &&
        isBF16ReduceResultElementCount(sourceVMIType.getElementCount());
    // Zero-paired vdintlv preserves group-slot placement when both layouts
    // have the same slot width and lane stride, including slots=8 packets.
    bool supportedLayouts =
        sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots() &&
        sourceLayout.getNumGroups() == resultLayout.getNumGroups() &&
        sourceLayout.getSlots() == resultLayout.getSlots() &&
        (sourceLayout.getSlots() == 1 || sourceLayout.getSlots() == 8) &&
        sourceLayout.getLaneStride() == resultLayout.getLaneStride() &&
        sourceLayout.getNumGroups() == sourceVMIType.getElementCount();
    // SAT may clamp overflow, so the bitwise fast path is safe to opt into
    // only for the non-saturating conversion contract.  The zero-paired
    // vdintlv fast path moves data without a conversion, so it cannot carry a
    // predicate; a masked cast uses the vcvt group-slot path instead.
    bool applicable = op.getMask() == nullptr && enableVmiFastmath && supportedLogicalShape && supportedLayouts &&
                      rounding && rounding.getValue() == "Z" && saturate && saturate.getValue() == "NOSAT" &&
                      !sourceParts.empty() && !resultTypes.empty() && sourceParts.size() == resultTypes.size();
    if (!applicable) {
      return false;
    }
    FailureOr<GroupSlotBF16ReducePlan> plan =
        buildGroupSlotBF16ReducePlan(op, sourceParts, resultTypes, rewriter);
    if (failed(plan)) {
      return failure();
    }
    if (failed(lowerGroupSlotBF16ReduceResults(op, sourceParts, resultTypes,
                                                *plan, rewriter))) {
      return failure();
    }
    return true;
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter& rewriter) const
  {
      unsigned resultBits = pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
      bool invalidShape = sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
                          sourceLayout.getSlots() != resultLayout.getSlots() ||
                          (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
                          !sourceVMIType.getElementType().isF32() || (resultBits != 16 && resultBits != 8) ||
                          sourceParts.size() != resultTypes.size();
      if (invalidShape) {
          return rewriter.notifyMatchFailure(op, "unsupported group-slot truncf shape");
      }
      FailureOr<bool> reduceNarrowing =
          tryLowerGroupSlotBF16ReduceNarrowing(op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
      if (failed(reduceNarrowing)) {
          return failure();
      }
      if (*reduceNarrowing) {
          return success();
      }
      const char* activeSlotPattern = sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
      FailureOr<Value> activeSlotMask =
          createPrefixMask(op.getLoc(), MaskType::get(rewriter.getContext(), "b32"), activeSlotPattern, rewriter);
      if (failed(activeSlotMask)) {
          return rewriter.notifyMatchFailure(op, "failed to build group-slot truncf active slot mask");
      }
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      FailureOr<SmallVector<Value>> results =
          lowerGroupSlotTruncParts(op, sourceParts, resultTypes, masks, *activeSlotMask, sat, rewriter);
      if (failed(results)) {
          return failure();
      }
      replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
      return success();
  }

  /// Convert every physical source part of a group-slot truncf and pair it with
  /// its conversion predicate.  A group-slot part carries padding plus the active
  /// slots, and the authored predicate only describes the active slots, so the
  /// two are ANDed: the conversion never converts a padding slot while it zeroes
  /// the inactive active slots.
  FailureOr<SmallVector<Value>> lowerGroupSlotTruncParts(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks, Value activeSlotMask,
      StringAttr sat, OneToNPatternRewriter& rewriter) const
  {
      auto b32MaskType = MaskType::get(rewriter.getContext(), "b32");
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [index, sourceAndResult] : llvm::enumerate(llvm::zip_equal(sourceParts, resultTypes))) {
          auto [sourcePart, physicalResultType] = sourceAndResult;
          Value partMask = activeSlotMask;
          if (!masks.empty()) {
              auto seedType = dyn_cast<VRegType>(sourcePart.getType());
              if (!seedType) {
                  return rewriter.notifyMatchFailure(op, "unsupported group-slot truncf physical source type");
              }
              FailureOr<Value> userMask = getConversionMaskForChunk(masks, index, Value());
              if (failed(userMask)) {
                  return rewriter.notifyMatchFailure(op, "failed to build a group-slot truncf predicate");
              }
              FailureOr<Value> allTrue = createAllTrueMaskForVReg(op.getLoc(), seedType, rewriter);
              if (failed(allTrue)) {
                  return rewriter.notifyMatchFailure(op, "failed to build the group-slot truncf all-true mask");
              }
              partMask =
                  rewriter.create<PandOp>(op.getLoc(), b32MaskType, *userMask, activeSlotMask, *allTrue).getResult();
          }
          FailureOr<Value> result =
              lowerGroupSlotTruncPart(op, sourcePart, physicalResultType, partMask, sat, rewriter);
          if (failed(result)) {
              return failure();
          }
          results.push_back(*result);
      }
      return results;
  }

  FailureOr<bool> tryLowerSameWidthTrunc(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      bool sameWidthContiguous = physicalPlan.sourceBits == physicalPlan.resultBits && sourceLayout && resultLayout &&
                                 sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
                                 resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
                                 sourceParts.size() == physicalPlan.resultTypes.size();
      if (!sameWidthContiguous) {
          return false;
      }
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      if (failed(lowerSameWidth(
              op, sourceParts, physicalPlan.resultTypes, masks, physicalPlan.sourceViewType, sat, rewriter))) {
          return failure();
      }
      return true;
  }

  FailureOr<bool> tryLowerDenseLaneStrideTrunc(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      bool denseLaneStrideNarrowing = sourceLayout && resultLayout && sourceLayout.isContiguous() &&
                                      sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
                                      resultLayout.getLaneStride() != 1 &&
                                      sourceParts.size() == physicalPlan.resultTypes.size();
      if (!denseLaneStrideNarrowing) {
          return false;
      }
      bool isEven32To16 = physicalPlan.resultBits == 16 && resultLayout.getLaneStride() == 2;
      bool isPacked32To8 = physicalPlan.resultBits == 8 && resultLayout.getLaneStride() == 4;
      bool isEven16To8 = physicalPlan.resultBits == 8 && resultLayout.getLaneStride() == 2;
      if (!isEven32To16 && !isPacked32To8 && !isEven16To8) {
          return rewriter.notifyMatchFailure(op, "unsupported dense lane_stride truncf result layout");
      }
      StringRef part = isPacked32To8 ? "P0" : "EVEN";
      if (failed(lowerDenseLaneStride(
              op, sourceParts, physicalPlan.resultTypes, masks, part, physicalPlan.sourceIsPackedBF16x2,
              physicalPlan.sourceViewType, rewriter))) {
          return failure();
      }
      return true;
  }

  // Spine-scoped composite narrowing: the f32 source is deinterleaved=4 and the
  // narrow result keeps the same four parts, so one part-per-chunk
  // lowerDenseLaneStride (EVEN for 16-bit, P0 for 8-bit) is the whole
  // conversion and no pto.vor merge is emitted.
  FailureOr<bool> tryLowerSpineCompositeNarrowing(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      bool spineCompositeNarrowing =
          physicalPlan.sourceBits == 32 && sourceLayout && resultLayout && sourceLayout.isDeinterleaved() &&
          (sourceLayout.getFactor() == 2 || sourceLayout.getFactor() == 4) && sourceLayout.getLaneStride() == 1 &&
          resultLayout.isDeinterleaved() && (resultLayout.getFactor() == 2 || resultLayout.getFactor() == 4) &&
          ((physicalPlan.resultBits == 8 && resultLayout.getLaneStride() == 4) ||
           (physicalPlan.resultBits == 16 && resultLayout.getLaneStride() == 2)) &&
          sourceParts.size() == physicalPlan.resultTypes.size();
      if (!spineCompositeNarrowing) {
          return false;
      }
      StringRef part = physicalPlan.resultBits == 8 ? StringRef("P0") : StringRef("EVEN");
      if (failed(lowerDenseLaneStride(
              op, sourceParts, physicalPlan.resultTypes, masks, part, physicalPlan.sourceIsPackedBF16x2,
              physicalPlan.sourceViewType, rewriter))) {
          return failure();
      }
      return true;
  }

  // F32 d(4) -> BF16 contiguous is a composite narrowing: the source has four
  // modulo-4 lane partitions while the result occupies one full BF16 register.
  // First use the existing d(4) -> d(2) narrowing relation to pack the four
  // source parts into two BF16 parts, then reuse the d(2) -> contiguous
  // materialization to interleave those two parts into logical lane order.
  FailureOr<SmallVector<Value, 2>> buildDeinterleaved2BF16Parts(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      Value unmaskedSeed, StringAttr rnd, StringAttr sat, VRegType intermediateType,
      OneToNPatternRewriter& rewriter) const
  {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      SmallVector<Value, 2> intermediateResults;
      intermediateResults.reserve(2);
      for (size_t part = 0; part < 2; ++part) {
          FailureOr<Value> result = buildNarrowTruncResult(
              op, sourceParts, intermediateType, kEvenOddParts, part,
              /*sourceFactor=*/kPairWidth, /*resultLaneStride=*/1, physicalPlan.sourceViewType,
              physicalPlan.sourceIsPackedBF16x2, masks, unmaskedSeed, rnd, sat, rewriter);
          if (failed(result)) {
              return failure();
          }
          intermediateResults.push_back(*result);
      }
      return intermediateResults;
  }

  LogicalResult lowerDeinterleaved4ToContiguousBF16(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      OneToNPatternRewriter& rewriter) const
  {
      Type resultElementType = physicalPlan.resultTypes.front().getElementType();
      FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(resultElementType);
      if (failed(lanesPerPart)) {
          return rewriter.notifyMatchFailure(op, "failed to determine BF16 lanes per physical part");
      }
      VRegType intermediateType = VRegType::get(rewriter.getContext(), *lanesPerPart, resultElementType);

      StringAttr rnd =
          rewriter.getStringAttr(getTruncFRoundMode(op, physicalPlan.resultTypes.front().getElementType()));
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      Value unmaskedSeed;
      if (masks.empty()) {
          FailureOr<Value> seed = createUnmaskedConversionSeed(op.getLoc(), physicalPlan.sourceViewType, rewriter);
          if (failed(seed)) {
              return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
          }
          unmaskedSeed = *seed;
      }
      FailureOr<SmallVector<Value, 2>> intermediateResults = buildDeinterleaved2BF16Parts(
          op, sourceParts, physicalPlan, masks, unmaskedSeed, rnd, sat, intermediateType, rewriter);
      if (failed(intermediateResults)) {
          return failure();
      }

      SmallVector<Type> finalResultTypes(physicalPlan.resultTypes.begin(), physicalPlan.resultTypes.end());
      FailureOr<SmallVector<Value>> results =
          materializeDeinterleaved2ToContiguous(op, *intermediateResults, finalResultTypes, rewriter);
      if (failed(results)) {
          return failure();
      }
      replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
      return success();
  }

  FailureOr<bool> tryLowerDeinterleaved4ToContiguousBF16(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      bool applies = physicalPlan.sourceBits == kElementBits32 && physicalPlan.resultBits == kElementBits16 &&
                     sourceLayout && sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 4 &&
                     sourceLayout.getLaneStride() == 1 && resultLayout && resultLayout.isContiguous() &&
                     resultLayout.getLaneStride() == 1 && sourceParts.size() == 4 &&
                     physicalPlan.resultTypes.size() <= 2;
      if (!applies) {
          return false;
      }
      if (failed(lowerDeinterleaved4ToContiguousBF16(op, sourceParts, physicalPlan, masks, rewriter))) {
          return failure();
      }
      return true;
  }

  // Dense factor-N narrowing for the residual truncf shapes the composite
  // forms above did not claim.
  LogicalResult lowerGenericTruncNarrowing(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, const TruncFPhysicalPlan& physicalPlan,
      ValueRange masks, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<TruncFNarrowingPlan> narrowingPlan = buildNarrowingPlan(
          op, physicalPlan.sourceBits, physicalPlan.resultBits, resultLayout, sourceParts.size(), resultTypes.size(),
          rewriter);
      if (failed(narrowingPlan)) {
          return failure();
      }
      StringAttr rnd =
          rewriter.getStringAttr(getTruncFRoundMode(op, physicalPlan.resultTypes.front().getElementType()));
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      return lowerNarrow(
          op, sourceParts, physicalPlan.resultTypes, masks, narrowingPlan->parts, narrowingPlan->sourceFactor,
          narrowingPlan->resultLaneStride, physicalPlan.sourceViewType, physicalPlan.sourceIsPackedBF16x2, rnd, sat,
          rewriter);
  }

  // Runs the composite narrowings that claim the deinterleaved=4 source shapes
  // left after the same-layout, same-width and dense lane-stride fast paths.
  FailureOr<bool> tryLowerCompositeTruncNarrowing(
      VMITruncFOp op, ValueRange sourceParts, const TruncFPhysicalPlan& physicalPlan, ValueRange masks,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<bool> deinterleaved4ToContiguous = tryLowerDeinterleaved4ToContiguousBF16(
          op, sourceParts, physicalPlan, masks, sourceLayout, resultLayout, rewriter);
      if (failed(deinterleaved4ToContiguous)) {
          return failure();
      }
      if (*deinterleaved4ToContiguous) {
          return true;
      }
      return tryLowerSpineCompositeNarrowing(
          op, sourceParts, physicalPlan, masks, sourceLayout, resultLayout, rewriter);
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes, ValueRange masks, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter& rewriter) const
  {
      FailureOr<TruncFPhysicalPlan> physicalPlan = buildPhysicalPlan(op, sourceParts, resultTypes, rewriter);
      if (failed(physicalPlan)) {
          return failure();
      }
      if (sourceLayout && sourceLayout.isGroupSlots()) {
          return rewriter.notifyMatchFailure(op, "group-slot layout for non-f32 truncf not supported");
      }
      FailureOr<bool> pairNarrowing = tryLowerSameLayoutBF16PairNarrowing(
          op, sourceParts, physicalPlan->resultTypes, masks, sourceLayout, resultLayout, rewriter);
      if (failed(pairNarrowing)) {
          return failure();
      }
      if (*pairNarrowing) {
          return success();
      }
      FailureOr<bool> sameWidth =
          tryLowerSameWidthTrunc(op, sourceParts, *physicalPlan, masks, sourceLayout, resultLayout, rewriter);
      if (failed(sameWidth)) {
          return failure();
      }
      if (*sameWidth) {
          return success();
      }
      FailureOr<bool> denseLaneStride =
          tryLowerDenseLaneStrideTrunc(op, sourceParts, *physicalPlan, masks, sourceLayout, resultLayout, rewriter);
      if (failed(denseLaneStride)) {
          return failure();
      }
      if (*denseLaneStride) {
          return success();
      }
      FailureOr<bool> compositeNarrowing =
          tryLowerCompositeTruncNarrowing(op, sourceParts, *physicalPlan, masks, sourceLayout, resultLayout, rewriter);
      if (failed(compositeNarrowing)) {
          return failure();
      }
      if (*compositeNarrowing) {
          return success();
      }
      return lowerGenericTruncNarrowing(op, sourceParts, resultTypes, *physicalPlan, masks, resultLayout, rewriter);
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
    // The conversion consumes the bf16 view of a packed bf16x2 source, so its
    // predicate has bf16 granularity; every other source keeps its own storage
    // element width.
    unsigned sourceElementBits = pto::getPTOStorageElemBitWidth(sourceElementType);
    if (pto::isPTOBF16x2Type(sourceElementType)) {
        sourceElementBits = kElementBits16;
    }
    FailureOr<SmallVector<Value>> masks =
        applyConversionPredicate(op, sourceParts, adaptor.getMask(), sourceElementBits, rewriter);
    if (failed(masks)) {
        return failure();
    }
    if (hasGroupSlotTruncFLayouts(sourceLayout, resultLayout)) {
        return lowerGroupSlotTrunc(
            op, sourceParts, resultTypes, *masks, sourceLayout, resultLayout, sourceVMIType, resultVMIType, rewriter);
    }

    return lowerNonGroupSlotTrunc(op, sourceParts, resultTypes, *masks, sourceLayout, resultLayout, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// VMIVUnzipOp
//===----------------------------------------------------------------------===//

struct OneToNVMIVUnzipOpPattern : OneToNOpConversionPattern<VMIVUnzipOp> {
  using OneToNOpConversionPattern<VMIVUnzipOp>::OneToNOpConversionPattern;

private:
  // Emit the pto.vdintlv of one result part and append its low and high halves.
  // The half-full pairing reuses its only source register as both operands,
  // its unused lanes being padding.
  LogicalResult
  emitSplitPart(VMIVUnzipOp op, Value firstSource, Value secondSource,
                size_t sourcePartsPerResult, VRegType lowType, VRegType highType,
                Type carrierElem, SmallVectorImpl<Value> &lows,
                SmallVectorImpl<Value> &highs,
                OneToNPatternRewriter &rewriter) const {
    // The carrier keeps the half-width result lane count, so bitcasting a wide
    // source part to it has the same register footprint.
    auto carrierType =
        VRegType::get(rewriter.getContext(), lowType.getElementCount(),
                      carrierElem);
    FailureOr<Value> lhs =
        bitcastVReg(op.getLoc(), firstSource, carrierType, rewriter);
    if (failed(lhs)) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast the vunzip source");
    }
    FailureOr<Value> rhs =
        sourcePartsPerResult == kSplitWidePartsPerHalf
            ? bitcastVReg(op.getLoc(), secondSource, carrierType, rewriter)
            : lhs;
    if (failed(rhs)) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast the vunzip source");
    }
    auto deinterleaved = rewriter.create<VdintlvOp>(
        op.getLoc(), TypeRange{carrierType, carrierType}, *lhs, *rhs);
    FailureOr<Value> lowValue =
        bitcastVReg(op.getLoc(), deinterleaved.getLow(), lowType, rewriter);
    FailureOr<Value> highValue =
        bitcastVReg(op.getLoc(), deinterleaved.getHigh(), highType, rewriter);
    bool resultBitcastFailed = failed(lowValue) || failed(highValue);
    if (resultBitcastFailed) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast the vunzip result");
    }
    lows.push_back(*lowValue);
    highs.push_back(*highValue);
    return success();
  }

  // Emit one pto.vdintlv per result part.  Bitcasting a wide source register to
  // the half-width carrier turns the two halves of every element into adjacent
  // lanes, so the deinterleave reads both at once; the low parts come first in
  // the result list, then the high parts.
  FailureOr<SmallVector<Value>>
  buildSplit(VMIVUnzipOp op, ValueRange sourceParts, ArrayRef<Type> lowTypes,
             ArrayRef<Type> highTypes, unsigned halfBits,
             size_t sourcePartsPerResult,
             OneToNPatternRewriter &rewriter) const {
    Type carrierElem = IntegerType::get(
        rewriter.getContext(), halfBits,
        IntegerType::SignednessSemantics::Unsigned);
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    for (size_t chunkIndex = 0; chunkIndex < lowTypes.size(); ++chunkIndex) {
      auto lowType = dyn_cast<VRegType>(lowTypes[chunkIndex]);
      auto highType = dyn_cast<VRegType>(highTypes[chunkIndex]);
      if (!lowType || !highType ||
          lowType.getElementCount() != highType.getElementCount()) {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported vunzip result type");
      }
      Value firstSource = sourceParts[chunkIndex * sourcePartsPerResult];
      Value secondSource =
          sourcePartsPerResult == kSplitWidePartsPerHalf
              ? sourceParts[chunkIndex * sourcePartsPerResult + 1]
              : firstSource;
      LogicalResult split =
          emitSplitPart(op, firstSource, secondSource, sourcePartsPerResult,
                        lowType, highType, carrierElem, lows, highs, rewriter);
      if (failed(split)) {
        return failure();
      }
    }
    lows.append(highs.begin(), highs.end());
    return lows;
  }

public:
  LogicalResult
  matchAndRewrite(VMIVUnzipOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> lowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> highTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    bool resultTypesMissing = failed(lowTypes) || failed(highTypes);
    if (resultTypesMissing) {
      return failure();
    }
    ValueRange sourceParts = adaptor.getSource();
    bool invalidShape = lowTypes->empty() ||
                        lowTypes->size() != highTypes->size() ||
                        sourceParts.size() < lowTypes->size() ||
                        sourceParts.size() % lowTypes->size() != 0;
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "vunzip needs at least one source part per result part");
    }
    // One wide register feeds one half register, or two when the lane count
    // leaves a result part half full; wider ratios are not a register pairing.
    size_t sourcePartsPerResult = sourceParts.size() / lowTypes->size();
    if (sourcePartsPerResult != 1 &&
        sourcePartsPerResult != kSplitWidePartsPerHalf) {
      return rewriter.notifyMatchFailure(
          op, "vunzip expects one or two source parts per result part");
    }
    auto firstLowType = dyn_cast<VRegType>(lowTypes->front());
    if (!firstLowType) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported vunzip result type");
    }
    unsigned halfBits =
        pto::getPTOStorageElemBitWidth(firstLowType.getElementType());
    FailureOr<SmallVector<Value>> results =
        buildSplit(op, sourceParts, *lowTypes, *highTypes, halfBits,
                   sourcePartsPerResult, rewriter);
    if (failed(results)) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, *results,
                                  *this->getTypeConverter());
  }
};
//===----------------------------------------------------------------------===//
// VMIVZipOp
//===----------------------------------------------------------------------===//

struct OneToNVMIVZipOpPattern : OneToNOpConversionPattern<VMIVZipOp> {
  using OneToNOpConversionPattern<VMIVZipOp>::OneToNOpConversionPattern;

private:
  // Validate the vzip shape and return how many wide result parts each
  // half-width operand part feeds (1 or 2).
  FailureOr<unsigned>
  getMergeRatio(VMIVZipOp op, ValueRange lowParts, ValueRange highParts,
                ArrayRef<Type> resultTypes,
                OneToNPatternRewriter &rewriter) const {
    bool invalidShape =
        resultTypes.empty() || lowParts.empty() ||
        lowParts.size() != highParts.size() ||
        resultTypes.size() < lowParts.size() ||
        resultTypes.size() % lowParts.size() != 0;
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "vzip expects one or two wide parts per half-width part");
    }
    unsigned widePartsPerHalf =
        static_cast<unsigned>(resultTypes.size() / lowParts.size());
    if (widePartsPerHalf != 1 && widePartsPerHalf != kSplitWidePartsPerHalf) {
      return rewriter.notifyMatchFailure(
          op, "vzip expects one or two wide parts per half-width part");
    }
    return widePartsPerHalf;
  }

  // pto.vintlv is the exact inverse of the pto.vdintlv the split uses: the
  // half-width carrier bitcast interleaved the two halves of every element, so
  // interleaving them back rebuilds the wide register pair.  One low/high part
  // pair therefore yields one or two wide parts in order.
  FailureOr<SmallVector<Value>>
  mergeParts(VMIVZipOp op, ValueRange lowParts, ValueRange highParts,
             ArrayRef<Type> resultTypes, unsigned widePartsPerHalf,
             unsigned halfBits, OneToNPatternRewriter &rewriter) const {
    MLIRContext *ctx = rewriter.getContext();
    Type halfCarrierElem = IntegerType::get(
        ctx, halfBits, IntegerType::SignednessSemantics::Unsigned);
    SmallVector<Value> results;
    for (size_t partIndex = 0; partIndex < lowParts.size(); ++partIndex) {
      auto halfPartType = dyn_cast<VRegType>(lowParts[partIndex].getType());
      if (!halfPartType) {
        return rewriter.notifyMatchFailure(op, "unsupported vzip operand type");
      }
      auto halfCarrierType =
          VRegType::get(ctx, halfPartType.getElementCount(), halfCarrierElem);
      FailureOr<Value> lowCarrier = bitcastVReg(
          op.getLoc(), lowParts[partIndex], halfCarrierType, rewriter);
      FailureOr<Value> highCarrier = bitcastVReg(
          op.getLoc(), highParts[partIndex], halfCarrierType, rewriter);
      bool operandBitcastFailed = failed(lowCarrier) || failed(highCarrier);
      if (operandBitcastFailed) {
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast the vzip operand");
      }
      auto interleaved = rewriter.create<VintlvOp>(
          op.getLoc(), TypeRange{halfCarrierType, halfCarrierType},
          *lowCarrier, *highCarrier);
      for (unsigned chunk = 0; chunk < widePartsPerHalf; ++chunk) {
        auto resultType = dyn_cast<VRegType>(
            resultTypes[partIndex * widePartsPerHalf + chunk]);
        if (!resultType) {
          return rewriter.notifyMatchFailure(op,
                                             "unsupported vzip result type");
        }
        Value half = chunk == 0 ? interleaved.getLow() : interleaved.getHigh();
        FailureOr<Value> result =
            bitcastVReg(op.getLoc(), half, resultType, rewriter);
        if (failed(result)) {
          return rewriter.notifyMatchFailure(
              op, "failed to bitcast the vzip result");
        }
        results.push_back(*result);
      }
    }
    return results;
  }

  // vpack only narrows down to 8 or 16 bits, so the inverse widens 8/16 to
  // 16/32; a wider half has no packed carrier form.
  FailureOr<SmallVector<Value>>
  buildMerge(VMIVZipOp op, ValueRange lowParts, ValueRange highParts,
             ArrayRef<Type> resultTypes,
             OneToNPatternRewriter &rewriter) const {
    FailureOr<unsigned> widePartsPerHalf =
        getMergeRatio(op, lowParts, highParts, resultTypes, rewriter);
    if (failed(widePartsPerHalf)) {
      return failure();
    }
    auto halfType = dyn_cast<VRegType>(lowParts.front().getType());
    if (!halfType) {
      return rewriter.notifyMatchFailure(op, "unsupported vzip operand type");
    }
    unsigned halfBits =
        pto::getPTOStorageElemBitWidth(halfType.getElementType());
    if (halfBits != kElementBits8 && halfBits != kElementBits16) {
      return rewriter.notifyMatchFailure(
          op, "unsupported vzip half element width");
    }
    return mergeParts(op, lowParts, highParts, resultTypes,
                      *widePartsPerHalf, halfBits, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(VMIVZipOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> merged =
        buildMerge(op, adaptor.getLow(), adaptor.getHigh(), *resultTypes,
                   rewriter);
    if (failed(merged)) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, *merged,
                                  *this->getTypeConverter());
  }
};
