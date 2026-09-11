// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals0.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<Value, 4>> materializeDeintToContiguousMaskGroup(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes, int64_t factor,
    int64_t groups, int64_t groupIndex, size_t resultOffset,
    PatternRewriter &rewriter) {
  SmallVector<Value, 4> sources;
  sources.reserve(factor);
  for (int64_t part = 0; part < factor; ++part) {
    sources.push_back(sourceParts[part * groups + groupIndex]);
  }
  SmallVector<Value, 4> results;
  if (factor == 2) {
    FailureOr<std::array<Value, 2>> materialized =
        materializeFactor2DeintToContiguousGroup(
            op, sources, resultTypes, resultOffset, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    results.append(materialized->begin(), materialized->end());
    return results;
  }
    FailureOr<std::array<Value, 4>> materialized =
        materializeFactor4DeintToContiguousGroup(
            op, sources, resultTypes, resultOffset, rewriter);
  if (failed(materialized)) {
    return failure();
  }
  results.append(materialized->begin(), materialized->end());
  return results;
}

FailureOr<SmallVector<Value>> materializeStagingDeintToContiguousMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidGroups = (factor != 2 && factor != 4) || sourceParts.empty() ||
                       sourceParts.size() % factor != 0;
  if (invalidGroups) {
    return fail("staging deinterleaved mask layout requires grouped source "
                "parts");
  }

  int64_t groups = sourceParts.size() / factor;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t i = 0; i < groups && results.size() < resultTypes.size(); ++i) {
    FailureOr<SmallVector<Value, 4>> materialized =
        materializeDeintToContiguousMaskGroup(
            op, sourceParts, resultTypes, factor, groups, i, results.size(),
            rewriter);
    if (failed(materialized)) {
      return failure();
    }
    for (Value value : *materialized) {
      bool resultCapacityReached = results.size() >= resultTypes.size();
      if (resultCapacityReached) {
        break;
      }
      results.push_back(value);
    }
  }
  bool resultArityMismatch = results.size() != resultTypes.size();
  if (resultArityMismatch) {
    return fail("staging deinterleaved mask layout result arity mismatch");
  }
  return results;
}

FailureOr<std::array<Value, 4>> materializeFactor4ContiguousToDeintGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    int64_t groups, int64_t groupIndex, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 4>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidSources = sources.size() != 4;
  bool insufficientResults =
      resultTypes.size() < static_cast<size_t>(4 * groups);
  if (invalidSources || insufficientResults) {
    return fail("factor-4 staging mask conversion requires four grouped results");
  }
  FailureOr<std::pair<Value, Value>> low = createPredicateDintlv(
      op->getLoc(), resultTypes[groupIndex], resultTypes[groups + groupIndex],
      sources[0], sources[1], rewriter);
  FailureOr<std::pair<Value, Value>> high = createPredicateDintlv(
      op->getLoc(), resultTypes[2 * groups + groupIndex],
      resultTypes[3 * groups + groupIndex], sources[2], sources[3], rewriter);
  if (failed(low)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  if (failed(high)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  FailureOr<std::pair<Value, Value>> even = createPredicateDintlv(
      op->getLoc(), resultTypes[groupIndex], resultTypes[2 * groups + groupIndex],
      low->first, high->first, rewriter);
  FailureOr<std::pair<Value, Value>> odd = createPredicateDintlv(
      op->getLoc(), resultTypes[groups + groupIndex],
      resultTypes[3 * groups + groupIndex], low->second, high->second, rewriter);
  if (failed(even)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  if (failed(odd)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  return std::array<Value, 4>{even->first, odd->first, even->second,
                              odd->second};
}

static FailureOr<std::array<Value, 2>> materializeFactor2ContiguousToDeintGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    int64_t groups, int64_t groupIndex, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 2>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidGroup =
      sources.size() != 2 ||
      resultTypes.size() < static_cast<size_t>(2 * groups);
  if (invalidGroup) {
    return fail("factor-2 staging mask conversion requires two grouped results");
  }
  FailureOr<std::pair<Value, Value>> materialized = createPredicateDintlv(
      op->getLoc(), resultTypes[groupIndex], resultTypes[groups + groupIndex],
      sources[0], sources[1], rewriter);
  if (failed(materialized)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  return std::array<Value, 2>{materialized->first, materialized->second};
}

static FailureOr<SmallVector<Value, 4>> materializeContiguousToDeintMaskGroup(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes, int64_t factor,
    int64_t groups, int64_t groupIndex, PatternRewriter &rewriter) {
  SmallVector<Value, 4> sources;
  size_t sourceBase = static_cast<size_t>(groupIndex * factor);
  if (sourceBase >= sourceParts.size()) {
    (void)rewriter.notifyMatchFailure(
        op, "staging contiguous mask layout ran out of source parts");
    return failure();
  }
  sources.reserve(factor);
  for (int64_t lane = 0; lane < factor; ++lane) {
    size_t index = sourceBase + lane;
    if (index < sourceParts.size()) {
      sources.push_back(sourceParts[index]);
      continue;
    }
    FailureOr<Value> zero = createAllFalseMaskLike(
        op->getLoc(), sourceParts[sourceBase], rewriter);
    if (failed(zero)) {
      (void)rewriter.notifyMatchFailure(
          op, "failed to create all-false staging mask");
      return failure();
    }
    sources.push_back(*zero);
  }
  SmallVector<Value, 4> results;
  if (factor == 2) {
    FailureOr<std::array<Value, 2>> materialized =
        materializeFactor2ContiguousToDeintGroup(
            op, sources, resultTypes, groups, groupIndex, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    results.append(materialized->begin(), materialized->end());
    return results;
  }
  FailureOr<std::array<Value, 4>> materialized =
      materializeFactor4ContiguousToDeintGroup(
          op, sources, resultTypes, groups, groupIndex, rewriter);
  if (failed(materialized)) {
    return failure();
  }
  results.append(materialized->begin(), materialized->end());
  return results;
}

struct StagingMaskPartAccumulator {
  SmallVector<SmallVector<Value, 4>, 4> parts;
  int64_t factor;
  int64_t groups;

  StagingMaskPartAccumulator(int64_t factor, int64_t groups)
      : parts(factor), factor(factor), groups(groups) {
    for (SmallVector<Value, 4> &part : parts) {
      part.reserve(static_cast<size_t>(groups));
    }
  }

  LogicalResult append(Operation *op, ArrayRef<Value> values,
                       PatternRewriter &rewriter) {
    bool invalidArity = values.size() != static_cast<size_t>(factor);
    if (invalidArity) {
      (void)rewriter.notifyMatchFailure(
          op, "staging contiguous mask layout result arity mismatch");
      return failure();
    }
    for (int64_t part = 0; part < factor; ++part) {
      parts[part].push_back(values[part]);
    }
    return success();
  }

  FailureOr<SmallVector<Value>> flatten(Operation *op,
                                        TypeRange resultTypes,
                                        PatternRewriter &rewriter) {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      bool invalidPartArity =
          parts[part].size() != static_cast<size_t>(groups);
      if (invalidPartArity) {
        (void)rewriter.notifyMatchFailure(
            op, "staging contiguous mask layout result arity mismatch");
        return failure();
      }
      results.append(parts[part]);
    }
    return results;
  }
};

FailureOr<SmallVector<Value>> materializeStagingContiguousToDeintMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidGroupedParts =
      (factor != 2 && factor != 4) || sourceParts.empty() ||
      resultTypes.size() % factor != 0;
  if (invalidGroupedParts) {
    return fail("staging contiguous mask layout requires grouped result parts");
  }

  int64_t groups = resultTypes.size() / factor;
  bool tooManySourceParts =
      sourceParts.size() > static_cast<size_t>(groups * factor);
  if (tooManySourceParts) {
    return fail("staging contiguous mask layout has too many source parts");
  }

  StagingMaskPartAccumulator accumulator(factor, groups);

  for (int64_t i = 0; i < groups; ++i) {
    FailureOr<SmallVector<Value, 4>> materialized =
        materializeContiguousToDeintMaskGroup(
            op, sourceParts, resultTypes, factor, groups, i, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    if (failed(accumulator.append(op, *materialized, rewriter))) {
      return failure();
    }
  }

  return accumulator.flatten(op, resultTypes, rewriter);
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastLayoutConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

FailureOr<SmallVector<Value>>
materializeMaskGranularityCastLayoutConversionViaContiguous(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

FailureOr<std::optional<SmallVector<Value>>>
materializeMaskGranularityCastStagingLayout(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

static FailureOr<SmallVector<Value>> forwardIdentityMaskParts(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
}

static bool requiresMaskDenseSplitFallback(VMILayoutAttr sourceLayout,
                                           VMILayoutAttr resultLayout) {
  return sourceLayout.isDenseSplit() || resultLayout.isDenseSplit();
}

FailureOr<std::optional<SmallVector<Value>>>
materializeMaskGranularityCastLayoutFallback(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter &rewriter) {
  FailureOr<SmallVector<Value>> layoutParts = materializeMaskLayoutConversion(
      op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
  if (succeeded(layoutParts)) {
    return std::optional<SmallVector<Value>>(std::move(*layoutParts));
  }

  FailureOr<std::optional<SmallVector<Value>>> staging =
      materializeMaskGranularityCastStagingLayout(
          op, sourceType, resultType, sourceParts, resultTypes, rewriter);
  if (failed(staging)) {
    return failure();
  }
  if (staging->has_value()) {
    return std::move(*staging);
  }

  if (!requiresMaskDenseSplitFallback(sourceLayout, resultLayout)) {
    return std::optional<SmallVector<Value>>{};
  }
  FailureOr<SmallVector<Value>> contiguous =
      materializeMaskGranularityCastLayoutConversionViaContiguous(
          op, sourceType, resultType, sourceParts, resultTypes, rewriter);
  if (failed(contiguous)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*contiguous));
}

FailureOr<SmallVector<Value>>
materializeMaskGranularityCastLayoutConversionViaContiguous(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
  VMIMaskType contiguousType =
      VMIMaskType::get(op->getContext(), sourceType.getElementCount(),
                       sourceType.getGranularity(), contiguous);
  FailureOr<SmallVector<Type>> contiguousTypes =
      getConvertedMaskPartTypes(contiguousType);
  if (failed(contiguousTypes)) {
    return failure();
  }
  FailureOr<SmallVector<Value>> contiguousParts =
      materializeMaskGranularityCastLayoutConversion(
          op, sourceType, contiguousType, sourceParts, *contiguousTypes,
          rewriter);
  if (failed(contiguousParts)) {
    return failure();
  }
  return materializeMaskGranularityCastLayoutConversion(
      op, contiguousType, resultType, *contiguousParts, resultTypes, rewriter);
}

static std::optional<bool> getMaskStagingDirection(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t factor) {
  bool sourceContiguous = sourceLayout && sourceLayout.isContiguous() &&
                          sourceLayout.getLaneStride() == 1;
  bool resultContiguous = resultLayout && resultLayout.isContiguous() &&
                          resultLayout.getLaneStride() == 1;
  bool sourceDeinterleaved =
      isElementDeinterleavedLayout(sourceLayout, factor) && resultContiguous;
  bool resultDeinterleaved =
      sourceContiguous && isElementDeinterleavedLayout(resultLayout, factor);
  if (!sourceDeinterleaved && !resultDeinterleaved) {
    return std::nullopt;
  }
  return sourceDeinterleaved;
}

FailureOr<std::optional<SmallVector<Value>>>
materializeMaskGranularityCastStagingLayout(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  for (int64_t factor : {2L, 4L}) {
    std::optional<bool> sourceIsDeinterleaved =
        getMaskStagingDirection(sourceLayout, resultLayout, factor);
    if (!sourceIsDeinterleaved) {
      continue;
    }
    FailureOr<SmallVector<Value>> materialized =
        *sourceIsDeinterleaved
            ? materializeStagingDeintToContiguousMaskLayout(
                  op, sourceParts, resultTypes, factor, rewriter)
            : materializeStagingContiguousToDeintMaskLayout(
                  op, sourceParts, resultTypes, factor, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*materialized));
  }
  return std::optional<SmallVector<Value>>{};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastLayoutConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  bool hasLayouts = sourceLayout && resultLayout;
  if (!hasLayouts) {
    return fail("mask granularity cast layout conversion requires layouts");
  }

  bool identityLayout = sourceLayout == resultLayout;
  if (identityLayout) {
    return forwardIdentityMaskParts(op, sourceParts, resultTypes, rewriter);
  }

  FailureOr<std::optional<SmallVector<Value>>> fallback =
      materializeMaskGranularityCastLayoutFallback(
          op, sourceType, resultType, sourceParts, resultTypes, sourceLayout,
          resultLayout, rewriter);
  if (failed(fallback)) {
    return failure();
  }
  if (fallback->has_value()) {
    return std::move(**fallback);
  }

  return fail("unsupported mask granularity cast layout conversion");
}

struct MaskGranularityCastPlan {
  VMIMaskType physicalSourceType;
  VMIMaskType physicalResultType;
};

static FailureOr<SmallVector<Value>> materializeMaskGranularityCastThroughLayout(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes,
    const MaskGranularityCastPlan &plan, PatternRewriter &rewriter) {
  VMIMaskType granularityType = VMIMaskType::get(
      op->getContext(), sourceType.getElementCount(),
      plan.physicalResultType.getGranularity(),
      plan.physicalSourceType.getLayoutAttr());
  FailureOr<SmallVector<Value>> granularityParts =
      materializeMaskGranularityConversion(
          op, plan.physicalSourceType, granularityType, sourceParts, rewriter);
  if (failed(granularityParts)) {
    return failure();
  }
  return materializeMaskGranularityCastLayoutConversion(
      op, granularityType, plan.physicalResultType, *granularityParts,
      resultTypes, rewriter);
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityCastParts(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes,
    const MaskGranularityCastPlan &plan, PatternRewriter &rewriter) {
  bool samePhysicalLayout =
      plan.physicalSourceType.getLayoutAttr() ==
      plan.physicalResultType.getLayoutAttr();
  if (samePhysicalLayout) {
    return materializeMaskGranularityConversion(
        op, plan.physicalSourceType, plan.physicalResultType, sourceParts,
        rewriter);
  }

  return materializeMaskGranularityCastThroughLayout(
      op, sourceType, resultType, sourceParts, resultTypes, plan, rewriter);
}

static FailureOr<MaskGranularityCastPlan> buildMaskGranularityCastPlan(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<MaskGranularityCastPlan> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool laneCountMismatch =
      sourceType.getElementCount() != resultType.getElementCount();
  if (laneCountMismatch) {
    return fail("requires source and result mask lane counts to match");
  }
  FailureOr<VMIMaskType> physicalSourceType =
      getVMIMaskPhysicalCarrierType(sourceType);
  FailureOr<VMIMaskType> physicalResultType =
      getVMIMaskPhysicalCarrierType(resultType);
  bool missingCarrierType =
      failed(physicalSourceType) || failed(physicalResultType);
  if (missingCarrierType) {
    return fail("requires source/result mask physical carrier types");
  }
  return MaskGranularityCastPlan{*physicalSourceType, *physicalResultType};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  FailureOr<MaskGranularityCastPlan> plan =
      buildMaskGranularityCastPlan(op, sourceType, resultType, rewriter);
  if (failed(plan)) {
    return failure();
  }

  if (plan->physicalSourceType == plan->physicalResultType) {
    FailureOr<SmallVector<Value>> identity =
        forwardIdentityMaskParts(op, sourceParts, resultTypes, rewriter);
    if (failed(identity)) {
      return failure();
    }
    return std::move(*identity);
  }

  return materializeMaskGranularityCastParts(
      op, sourceType, resultType, sourceParts, resultTypes, *plan, rewriter);
}

struct OneToNVMIEnsureLayoutOpPattern
    : OneToNOpConversionPattern<VMIEnsureLayoutOp> {
  using OneToNOpConversionPattern<VMIEnsureLayoutOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureLayoutOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    return lowerMaterializedResults(
        op, *this->getTypeConverter(), rewriter,
        [&]() -> FailureOr<SmallVector<Value>> {
          return materializeEnsureLayoutConversion(
              op, adaptor.getSource(), sourceType, resultType,
              *this->getTypeConverter(), rewriter);
        });
  }
};

struct OneToNVMIEnsureMaskLayoutOpPattern
    : OneToNOpConversionPattern<VMIEnsureMaskLayoutOp> {
  using OneToNOpConversionPattern<
      VMIEnsureMaskLayoutOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureMaskLayoutOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(supports.getEnsureMaskLayoutFact(sourceType, resultType,
                                                &supportReason))) {
      return rewriter.notifyMatchFailure(
          op, Twine("ensure_mask_layout has no registered materialization "
                    "support: ") +
                  supportReason);
    }
    bool granularityMismatch =
        sourceType.getGranularity() != resultType.getGranularity();
    if (granularityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "mask layout helper cannot also change granularity");
    }
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(results)) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, *results,
                                  *this->getTypeConverter());
  }
};

struct OneToNVMIEnsureMaskGranularityOpPattern
    : OneToNOpConversionPattern<VMIEnsureMaskGranularityOp> {
  using OneToNOpConversionPattern<
      VMIEnsureMaskGranularityOp>::OneToNOpConversionPattern;

private:
  LogicalResult replaceCheckedResults(
      VMIEnsureMaskGranularityOp op, OneToNPatternRewriter &rewriter,
      FailureOr<SmallVector<Value>> results, ArrayRef<Type> resultTypes) const {
    if (failed(results)) {
      return failure();
    }
    bool resultArityMismatch = results->size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "mask granularity cast result arity mismatch");
    }
    for (auto [result, type] : llvm::zip_equal(*results, resultTypes)) {
      bool resultTypeMismatch = result.getType() != type;
      if (resultTypeMismatch) {
        return rewriter.notifyMatchFailure(
            op, "mask granularity cast result type mismatch");
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIEnsureMaskGranularityOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    bool identity = sourceType.getGranularity() == resultType.getGranularity() &&
                    sourceType.getLayoutAttr() == resultType.getLayoutAttr();
    if (!identity) {
      std::string reason;
      if (failed(supports.getMaskGranularityCastLayoutFactForLayouts(
              sourceType, resultType, sourceType.getLayoutAttr(),
              resultType.getLayoutAttr(), &reason))) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask granularity cast layout relation: " + reason);
      }
    }

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    FailureOr<SmallVector<Value>> results =
        materializeMaskGranularityCastConversion(
            op, sourceType, resultType, sourceParts, resultTypes, rewriter);
    return replaceCheckedResults(op, rewriter, std::move(results), resultTypes);
  }

};

struct OneToNVMIBroadcastOpPattern : OneToNOpConversionPattern<VMIBroadcastOp> {
  using OneToNOpConversionPattern<VMIBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange inputParts = adaptor.getValue();
    bool invalidInputArity = inputParts.size() != 1;
    if (invalidInputArity) {
      return rewriter.notifyMatchFailure(
          op, "broadcast input must convert to one value");
    }
    bool inputIsVReg = isa<VMIVRegType>(op.getValue().getType());

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "broadcast result must be vreg");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for broadcast mask");
      }
      StringAttr position =
          inputIsVReg ? rewriter.getStringAttr("LOWEST") : StringAttr{};
      results.push_back(rewriter
                            .create<VdupOp>(op.getLoc(), resultType,
                                            inputParts.front(), *mask, position)
                            .getResult());
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }
};

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return rewriter
        .create<arith::ConstantOp>(loc, IntegerAttr::get(intType, value))
        .getResult();
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return rewriter
        .create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(floatType, static_cast<double>(value)))
        .getResult();
  }
  return failure();
}

FailureOr<Value> createIotaChunkBase(Location loc, Value base,
                                     int64_t laneOffset, StringRef order,
                                     PatternRewriter &rewriter) {
  if (laneOffset == 0) {
    return base;
  }

  FailureOr<Value> offset =
      createScalarOffsetConstant(loc, base.getType(), laneOffset, rewriter);
  if (failed(offset)) {
    return failure();
  }

  if (isa<IntegerType>(base.getType())) {
    if (order == "DESC") {
      return rewriter.create<arith::SubIOp>(loc, base, *offset).getResult();
    }
    return rewriter.create<arith::AddIOp>(loc, base, *offset).getResult();
  }
  if (isa<FloatType>(base.getType())) {
    if (order == "DESC") {
      return rewriter.create<arith::SubFOp>(loc, base, *offset).getResult();
    }
    return rewriter.create<arith::AddFOp>(loc, base, *offset).getResult();
  }

  return failure();
}

struct IotaMaterializationContext {
  Location loc;
  Value base;
  StringAttr orderAttr;
  PatternRewriter &rewriter;
};

static StringRef getIotaOrder(const IotaMaterializationContext &context) {
  return context.orderAttr ? context.orderAttr.getValue() : StringRef("ASC");
}

FailureOr<Value> createIotaContiguousChunk(
    const IotaMaterializationContext &context, Type resultType,
    int64_t laneOffset) {
  // Contiguous iota is a direct VCI of the absolute chunk base (ASC
  // `vci(offset_sreg)`). Group-periodic VL128 {group=2} shares one such VL64
  // result across both physical parts (see sharedChunks below).
  StringRef order = getIotaOrder(context);
  FailureOr<Value> chunkBase =
      createIotaChunkBase(context.loc, context.base, laneOffset, order,
                          context.rewriter);
  if (failed(chunkBase)) {
    return failure();
  }
  return context.rewriter
      .create<VciOp>(context.loc, resultType, *chunkBase, context.orderAttr)
      .getResult();
}

FailureOr<std::optional<Value>> createPowerOfTwoSubVLChunk(
    Location loc, Type resultType, Value base, int64_t groupSize,
    StringRef order, Value allMask, PatternRewriter &rewriter) {
  bool unsupportedShape =
      !llvm::isPowerOf2_64(static_cast<uint64_t>(groupSize)) ||
      !isa<IntegerType>(base.getType());
  if (unsupportedShape) {
    return std::optional<Value>{};
  }

  FailureOr<Value> zeroScalar =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  FailureOr<Value> maskScalar = createScalarOffsetConstant(
      loc, base.getType(), groupSize - 1, rewriter);
  bool failedScalars = failed(zeroScalar) || failed(maskScalar);
  if (failedScalars) {
    return failure();
  }

  Value laneIds =
      rewriter.create<VciOp>(loc, resultType, *zeroScalar, StringAttr{})
          .getResult();
  Value maskVec =
      rewriter
          .create<VdupOp>(loc, resultType, *maskScalar, allMask,
                          /*position=*/nullptr)
          .getResult();
  Value rem = rewriter
                  .create<VandOp>(loc, resultType, laneIds, maskVec, allMask)
                  .getResult();
  if (order == "DESC") {
    Value baseVec =
        rewriter
            .create<VdupOp>(loc, resultType, base, allMask,
                            /*position=*/nullptr)
            .getResult();
    return std::optional<Value>(
        rewriter.create<VsubOp>(loc, resultType, baseVec, rem, allMask)
            .getResult());
  }
  return std::optional<Value>(
      rewriter.create<VaddsOp>(loc, resultType, rem, base, allMask).getResult());
}

/// Pack group-periodic ramps inside one physical VL when S < physVL and
/// physVL % S == 0 (e.g. i32 L=64,group=2 → [base..base+31 | base..base+31]).
///
/// Preferred recipes (O(1), independent of G = physVL/S):
///   * S == 1 → vdup(base)
///   * S power-of-2 integer (all legal sub-VL S on this ISA) →
///       ASC:  vadds(vand(vci(0), S-1), base)
///       DESC: vsub(vdup(base), vand(vci(0), S-1))
///
/// Residual fallback (non-integer base): per-group vci(base) ∓ g*S +
/// lane-range vsel. Index iota is integer-only in practice.
///
/// When S == physVL this is just `vci(base)` (single group fills the VL).
static FailureOr<Value> materializeResidualSubVLGroup(
    Location loc, Type resultType, Value base, StringRef order, Value full,
    MaskType maskType, Value zeroScalar, Value allMask, Value previousResult,
    int64_t groupSize, int64_t localGroup, PatternRewriter &rewriter) {
  Value adjusted = full;
  if (localGroup != 0) {
    int64_t delta = localGroup * groupSize;
    FailureOr<Value> offsetScalar =
        createScalarOffsetConstant(loc, base.getType(), delta, rewriter);
    if (failed(offsetScalar)) {
      return failure();
    }
    if (order == "DESC") {
      adjusted = rewriter
                     .create<VaddsOp>(loc, resultType, full, *offsetScalar,
                                      allMask)
                     .getResult();
    } else {
      Value negOffset = isa<FloatType>(base.getType())
                            ? rewriter.create<arith::NegFOp>(
                                  loc, *offsetScalar)
                                  .getResult()
                            : rewriter
                                  .create<arith::SubIOp>(loc, zeroScalar,
                                                         *offsetScalar)
                                  .getResult();
      adjusted = rewriter
                     .create<VaddsOp>(loc, resultType, full, negOffset, allMask)
                     .getResult();
    }
  }
  FailureOr<Value> laneMask = createLaneRangeMask(
      loc, maskType, localGroup * groupSize, (localGroup + 1) * groupSize,
      rewriter);
  if (failed(laneMask)) {
    return failure();
  }
  return rewriter
      .create<VselOp>(loc, resultType, adjusted, previousResult, *laneMask)
      .getResult();
}

FailureOr<Value> createResidualSubVLGroupPeriodicChunk(
    Location loc, Type resultType, Value base, StringRef order,
    Value full, MaskType maskType, Value zeroScalar, Value allMask,
    int64_t groupSize,
    int64_t groupsPerChunk, PatternRewriter &rewriter) {
  Value result =
      rewriter
          .create<VdupOp>(loc, resultType, zeroScalar,
                          allMask,
                          /*position=*/nullptr)
          .getResult();
  for (int64_t localGroup = 0; localGroup < groupsPerChunk; ++localGroup) {
    FailureOr<Value> nextResult = materializeResidualSubVLGroup(
        loc, resultType, base, order, full, maskType, zeroScalar, allMask,
        result, groupSize, localGroup, rewriter);
    if (failed(nextResult)) {
      return failure();
    }
    result = *nextResult;
  }
  return result;
}

static FailureOr<std::optional<Value>> createSubVLPeriodicFastPath(
    const IotaMaterializationContext &context, Type resultType,
    int64_t groupSize, StringRef order, Value allMask) {
  Location loc = context.loc;
  Value base = context.base;
  PatternRewriter &rewriter = context.rewriter;
  if (groupSize == 1) {
    return std::optional<Value>(
        rewriter
            .create<VdupOp>(loc, resultType, base, allMask,
                            /*position=*/nullptr)
            .getResult());
  }

  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    return failure();
  }
  int64_t groupsPerChunk = vregType.getElementCount() / groupSize;
  if (groupsPerChunk == 1) {
    FailureOr<Value> result =
        createIotaContiguousChunk(context, resultType, /*laneOffset=*/0);
    if (failed(result)) {
      return failure();
    }
    return std::optional<Value>(*result);
  }

  FailureOr<std::optional<Value>> powerOfTwo = createPowerOfTwoSubVLChunk(
      loc, resultType, base, groupSize, order, allMask, rewriter);
  if (failed(powerOfTwo)) {
    return failure();
  }
  return *powerOfTwo;
}

FailureOr<Value> createSubVLGroupPeriodicChunk(
    const IotaMaterializationContext &context, Type resultType,
    int64_t groupSize) {
  Location loc = context.loc;
  Value base = context.base;
  PatternRewriter &rewriter = context.rewriter;
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    return failure();
  }

  if (groupSize <= 0) {
    return failure();
  }

  int64_t lanesPerPart = vregType.getElementCount();
  if (lanesPerPart % groupSize != 0) {
    return failure();
  }

  FailureOr<Value> allMask =
      createAllTrueMaskForVReg(loc, vregType, rewriter);
  if (failed(allMask)) {
    return failure();
  }

  StringRef order = getIotaOrder(context);
  FailureOr<std::optional<Value>> fastPath = createSubVLPeriodicFastPath(
      context, resultType, groupSize, order, *allMask);
  if (failed(fastPath)) {
    return failure();
  }
  if (fastPath->has_value()) {
    return **fastPath;
  }

  int64_t groupsPerChunk = lanesPerPart / groupSize;
  FailureOr<Value> full =
      createIotaContiguousChunk(context, resultType, /*laneOffset=*/0);
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  FailureOr<Value> zeroScalar =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  bool failedResidualInputs =
      failed(full) || failed(maskType) || failed(zeroScalar);
  if (failedResidualInputs) {
    return failure();
  }

  return createResidualSubVLGroupPeriodicChunk(
      loc, resultType, base, order, *full, *maskType, *zeroScalar, *allMask,
      groupSize, groupsPerChunk, rewriter);
}

FailureOr<Value> createIotaDeinterleavedChunk(
    const IotaMaterializationContext &context, Type resultType, int64_t factor,
    int64_t part, int64_t chunk, int64_t lanesPerPart) {
  Location loc = context.loc;
  Value base = context.base;
  StringAttr orderAttr = context.orderAttr;
  PatternRewriter &rewriter = context.rewriter;
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    return failure();
  }

  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, vregType, rewriter);
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  FailureOr<Value> factorScalar =
      createScalarOffsetConstant(loc, base.getType(), factor, rewriter);
  bool failedIotaInputs = failed(mask) || failed(zero) || failed(factorScalar);
  if (failedIotaInputs) {
    return failure();
  }

  Value local =
      rewriter.create<VciOp>(loc, resultType, *zero, StringAttr{}).getResult();
  Value scaled =
      rewriter.create<VmulsOp>(loc, resultType, local, *factorScalar, *mask)
          .getResult();

  StringRef order = orderAttr ? orderAttr.getValue() : StringRef("ASC");
  int64_t partOffset = part + factor * chunk * lanesPerPart;
  FailureOr<Value> biasedBase =
      createIotaChunkBase(loc, base, partOffset, order, rewriter);
  if (failed(biasedBase)) {
    return failure();
  }

  if (order == "DESC") {
    Value baseVector = rewriter
                           .create<VdupOp>(loc, resultType, *biasedBase, *mask,
                                           /*position=*/nullptr)
                           .getResult();
    return rewriter.create<VsubOp>(loc, resultType, baseVector, scaled, *mask)
        .getResult();
  }

  return rewriter.create<VaddsOp>(loc, resultType, scaled, *biasedBase, *mask)
      .getResult();
}

template <typename IotaOp>
struct OneToNVMIIotaOpPattern : OneToNOpConversionPattern<IotaOp> {
  using OneToNOpConversionPattern<IotaOp>::OneToNOpConversionPattern;
  using OpAdaptor =
      typename OneToNOpConversionPattern<IotaOp>::OpAdaptor;

private:
  struct IotaLoweringInput {
    VMIVRegType resultVMIType;
    VMILayoutAttr layout;
    Value base;
    SmallVector<Type> resultTypes;
    int64_t lanesPerPart;
  };

  FailureOr<IotaLoweringInput> getIotaLoweringInput(
      IotaOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout) {
      return rewriter.notifyMatchFailure(op, "iota requires assigned layout");
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "iota requires known physical lanes per part");
    }
    FailureOr<Value> base = getSingleValue(
        op, adaptor.getBase(), "iota base must convert to one value", rewriter);
    if (failed(base)) {
      return failure();
    }
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    return IotaLoweringInput{resultVMIType, layout, *base,
                             std::move(*resultTypes), *lanesPerPart};
  }

  FailureOr<std::pair<int64_t, int64_t>> validateGroupedIotaShape(
      IotaOp op, VMIVRegType resultVMIType, VMILayoutAttr layout,
      TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    if (lanesPerPart <= 0) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires positive physical lanes per part");
    }
    int64_t numGroups = op.getGroupAttr().getInt();
    int64_t logicalLanes = resultVMIType.getElementCount();
    if (numGroups <= 0) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires positive group count");
    }
    int64_t safeNumGroups = numGroups > 0 ? numGroups : 1;
    if (logicalLanes % safeNumGroups != 0) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires group to divide logical lane count");
    }
    int64_t groupSize = logicalLanes / safeNumGroups;
    int64_t safeLanesPerPart = lanesPerPart > 0 ? lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool compatibleShape = groupSize % safeLanesPerPart == 0 ||
                           lanesPerPart % safeGroupSize == 0;
    if (!compatibleShape) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires group_size to divide or be a multiple of physical lanes per part");
    }
    if (!layout.isContiguous()) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota currently supports contiguous layout only; ensure_layout to contiguous before lowering");
    }
    int64_t expectedArity =
        (logicalLanes + safeLanesPerPart - 1) / safeLanesPerPart;
    bool resultArityMismatch =
        static_cast<int64_t>(resultTypes.size()) != expectedArity;
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "grouped contiguous iota physical result count mismatch");
    }
    return std::make_pair(groupSize, expectedArity);
  }

  LogicalResult lowerGroupedIota(
      IotaOp op, Value base, VMIVRegType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) const {
    FailureOr<std::pair<int64_t, int64_t>> shape = validateGroupedIotaShape(
        op, resultVMIType, layout, resultTypes, lanesPerPart, rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t groupSize = shape->first;
    int64_t safeLanesPerPart = lanesPerPart > 0 ? lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool groupSizeMultipleOfPhys = groupSize % safeLanesPerPart == 0;
    bool physMultipleOfGroupSize = lanesPerPart % safeGroupSize == 0;

    llvm::DenseMap<std::pair<Type, int64_t>, Value> sharedChunks;
    IotaMaterializationContext context{op.getLoc(), base, op.getOrderAttr(),
                                       rewriter};
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "iota result must be vreg");
      }
      int64_t laneOffset = 0;
      if (groupSizeMultipleOfPhys) {
        laneOffset = (static_cast<int64_t>(index) * lanesPerPart) % groupSize;
      }
      auto key = std::make_pair(resultType, laneOffset);
      auto it = sharedChunks.find(key);
      if (it == sharedChunks.end()) {
        FailureOr<Value> result;
        if (physMultipleOfGroupSize && groupSize < lanesPerPart) {
          result = createSubVLGroupPeriodicChunk(context, resultType,
                                                 groupSize);
        } else {
          result = createIotaContiguousChunk(context, resultType, laneOffset);
        }
        if (failed(result)) {
          return rewriter.notifyMatchFailure(
              op, "failed to materialize grouped iota chunk");
        }
        it = sharedChunks.try_emplace(key, *result).first;
      }
      results.push_back(it->second);
    }
    return success();
  }

  LogicalResult lowerContiguousIota(
      IotaOp op, Value base, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) const {
    IotaMaterializationContext context{op.getLoc(), base, op.getOrderAttr(),
                                       rewriter};
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "iota result must be vreg");
      }
      FailureOr<Value> result = createIotaContiguousChunk(
          context, resultType, static_cast<int64_t>(index) * lanesPerPart);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to materialize contiguous iota chunk");
      }
      results.push_back(*result);
    }
    return success();
  }

  LogicalResult lowerDeinterleavedIota(
      IotaOp op, Value base, VMILayoutAttr layout, TypeRange resultTypes,
      int64_t lanesPerPart, OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    int64_t factor = layout.getFactor();
    int64_t safeFactor = factor > 0 ? factor : 1;
    bool resultFactorMismatch = resultTypes.size() % safeFactor != 0;
    if (resultFactorMismatch) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved iota physical result count does not match "
              "layout factor");
    }
    int64_t chunksPerPart = resultTypes.size() / safeFactor;
    IotaMaterializationContext context{op.getLoc(), base, op.getOrderAttr(),
                                       rewriter};
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        Type resultType = resultTypes[part * chunksPerPart + chunk];
        FailureOr<Value> result = createIotaDeinterleavedChunk(
            context, resultType, factor, part, chunk, lanesPerPart);
        if (failed(result)) {
          return rewriter.notifyMatchFailure(
              op, "failed to materialize deinterleaved iota chunk");
        }
        results.push_back(*result);
      }
    }
    return success();
  }

  LogicalResult lowerAndReplaceIota(
      IotaOp op, Value base, VMIVRegType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    if constexpr (std::is_same_v<IotaOp, VMIGroupIotaOp>) {
      if (failed(lowerGroupedIota(op, base, resultVMIType, layout, resultTypes,
                                  lanesPerPart, rewriter, results))) {
        return failure();
      }
    } else if (layout.isContiguous()) {
      if (failed(lowerContiguousIota(op, base, resultTypes, lanesPerPart,
                                     rewriter, results))) {
        return failure();
      }
    } else if (failed(lowerDeinterleavedIota(
                   op, base, layout, resultTypes, lanesPerPart, rewriter,
                   results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(IotaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<IotaLoweringInput> input =
        getIotaLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    SmallVector<Value> results;
    results.reserve(input->resultTypes.size());
    return lowerAndReplaceIota(op, input->base, input->resultVMIType,
                               input->layout, input->resultTypes,
                               input->lanesPerPart, rewriter, results);
  }
};

struct OneToNVMIConstantOpPattern : OneToNOpConversionPattern<VMIConstantOp> {
  using OneToNOpConversionPattern<VMIConstantOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSplat(VMIConstantOp op, TypedAttr splatAttr,
                           ArrayRef<Type> resultTypes,
                           OneToNPatternRewriter &rewriter) const {
    Value scalar =
        rewriter.create<arith::ConstantOp>(op.getLoc(), splatAttr).getResult();
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "constant result must be vreg");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for constant mask");
      }
      results.push_back(
          rewriter
              .create<VdupOp>(op.getLoc(), resultType, scalar, *mask,
                              /*position=*/nullptr)
              .getResult());
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIConstantOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat()) {
      return rewriter.notifyMatchFailure(
          op, "only splat dense data constants are supported");
    }
    auto splatAttr = dyn_cast<TypedAttr>(denseAttr.getSplatValue<Attribute>());
    if (!splatAttr) {
      return rewriter.notifyMatchFailure(op, "splat constant must be typed");
    }

    // arith.constant only accepts signless integer types, whereas VMI vregs may
    // carry signed/unsigned element types (e.g. ui16). Remap an unsigned/signed
    // integer splat to its signless equivalent; the downstream pto.vdup accepts
    // a signless scalar for a signed/unsigned result element.
    if (auto intAttr = dyn_cast<IntegerAttr>(splatAttr)) {
      if (auto intTy = dyn_cast<IntegerType>(intAttr.getType());
          intTy && !intTy.isSignless()) {
        splatAttr = IntegerAttr::get(rewriter.getIntegerType(intTy.getWidth()),
                                     intAttr.getValue());
      }
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerSplat(op, splatAttr, resultTypes, rewriter);
  }
};

struct OneToNVMIConstantMaskOpPattern
    : OneToNOpConversionPattern<VMIConstantMaskOp> {
  using OneToNOpConversionPattern<VMIConstantMaskOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Value>> materializePhysicalMasks(
      VMIConstantMaskOp op, ArrayRef<Type> resultTypes,
      ArrayRef<ConstantMaskChunkMaterialization> materializations,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         materializations) {
      bool tooManyMasks = results.size() >= resultTypes.size();
      if (tooManyMasks) {
        return rewriter.notifyMatchFailure(
            op, "constant_mask produced too many physical masks");
      }
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType) {
        return rewriter.notifyMatchFailure(op,
                                           "constant_mask result must be mask");
      }
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to materialize constant_mask physical chunk");
      }
      results.push_back(*mask);
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "constant_mask physical result count mismatch");
    }
    return results;
  }

public:

  LogicalResult
  matchAndRewrite(VMIConstantMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeConstantMaskMaterialization(op, &reason);
    if (failed(materializations)) {
      return rewriter.notifyMatchFailure(op, Twine("constant_mask ") + reason);
    }
    FailureOr<SmallVector<Value>> results = materializePhysicalMasks(
        op, resultTypes, *materializations, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICreateMaskOpPattern
    : OneToNOpConversionPattern<VMICreateMaskOp> {
  using OneToNOpConversionPattern<VMICreateMaskOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Type>> getResultTypes(VMICreateMaskOp op) const {
    return getConvertedResultTypes(op, 0, *this->getTypeConverter());
  }

  LogicalResult lowerDynamicCreateMask(
      VMICreateMaskOp op, OpAdaptor adaptor, VMIMaskType resultVMIType,
      VMILayoutAttr layout, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> active = getSingleValue(
        op, adaptor.getActiveLanes(),
        "create_mask active_lanes must convert to one value", rewriter);
    if (failed(active)) {
      return failure();
    }
    FailureOr<SmallVector<Type>> maybeResultTypes = getResultTypes(op);
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerDynamicMask(op, *active, resultVMIType, layout,
                                *maybeResultTypes, lanesPerPart, rewriter,
                                results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerConstantCreateMask(
      VMICreateMaskOp op, int64_t activeLanes, VMIMaskType resultVMIType,
      VMILayoutAttr layout, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> maybeResultTypes = getResultTypes(op);
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerConstantMask(op, activeLanes, resultVMIType, layout,
                                 *maybeResultTypes, lanesPerPart, rewriter,
                                 results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerDynamicMask(
      VMICreateMaskOp op, Value active, VMIMaskType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
    bool resultFactorMismatch = resultTypes.size() % factor != 0;
    if (resultFactorMismatch) {
      return rewriter.notifyMatchFailure(
          op, "dynamic create_mask physical result count does not match "
              "layout factor");
    }
    int64_t chunksPerPart = resultTypes.size() / factor;
    Value activeI32 = clampDynamicActiveLanes(
        op.getLoc(), active, resultVMIType.getElementCount(), rewriter);
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      Value remaining = createPartitionActiveLanes(op.getLoc(), activeI32,
                                                   factor, part, rewriter);
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        Type resultType = resultTypes[part * chunksPerPart + chunk];
        FailureOr<std::pair<Value, Value>> maskAndRemaining =
            buildDynamicMaskChunk(op, resultType, remaining, rewriter);
        if (failed(maskAndRemaining)) {
          return failure();
        }
        results.push_back(maskAndRemaining->first);
        remaining = maskAndRemaining->second;
      }
    }
    return success();
  }

  FailureOr<std::pair<bool, int64_t>> getConstantMaskChunkActivity(
      VMICreateMaskOp op, VMIMaskType resultVMIType, int64_t part,
      int64_t chunk, int64_t activeLanes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    bool anyLane = false;
    int64_t activeInChunk = 0;
    for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
      FailureOr<bool> padding =
          isPaddingLane(resultVMIType, part, chunk, lane);
      if (failed(padding)) {
        return rewriter.notifyMatchFailure(
            op, "failed to map create_mask physical padding lane");
      }
      if (*padding) {
        continue;
      }
      anyLane = true;
      FailureOr<int64_t> logicalLane =
          mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
      if (failed(logicalLane)) {
        return rewriter.notifyMatchFailure(
            op, "failed to map create_mask physical lane");
      }
      if (*logicalLane < activeLanes) {
        ++activeInChunk;
      }
    }
    return std::make_pair(anyLane, activeInChunk);
  }

  FailureOr<std::pair<Value, Value>> buildDynamicMaskChunk(
      VMICreateMaskOp op, Type resultType, Value remaining,
      OneToNPatternRewriter &rewriter) const {
    auto maskType = dyn_cast<MaskType>(resultType);
    if (!maskType) {
      return rewriter.notifyMatchFailure(op, "create_mask result must be mask");
    }
    FailureOr<std::pair<Value, Value>> maskAndRemaining =
        createRuntimePrefixMask(op.getLoc(), maskType, remaining, rewriter);
    if (failed(maskAndRemaining)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported mask type for dynamic create_mask");
    }
    return *maskAndRemaining;
  }

  FailureOr<Value> materializeConstantMaskValue(
      VMICreateMaskOp op, Type resultType, int64_t activeInChunk,
      int64_t lanesPerPart, OneToNPatternRewriter &rewriter) const {
    auto maskType = dyn_cast<MaskType>(resultType);
    if (!maskType) {
      return rewriter.notifyMatchFailure(op, "create_mask result must be mask");
    }
    std::optional<std::string> pattern =
        getPrefixPattern(activeInChunk, lanesPerPart);
    if (pattern) {
      FailureOr<Value> mask =
          createPrefixMask(op.getLoc(), maskType, *pattern, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for create_mask");
      }
      return *mask;
    }
    FailureOr<std::pair<Value, Value>> maskAndRemaining =
        createRuntimePrefixMask(
            op.getLoc(), maskType,
            createI32Constant(op.getLoc(), activeInChunk, rewriter), rewriter);
    if (failed(maskAndRemaining)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported mask type for create_mask plt fallback");
    }
    return maskAndRemaining->first;
  }

  LogicalResult lowerConstantMask(
      VMICreateMaskOp op, int64_t activeLanes, VMIMaskType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0;; ++chunk) {
        FailureOr<std::pair<bool, int64_t>> activity =
            getConstantMaskChunkActivity(op, resultVMIType, part, chunk,
                                         activeLanes, lanesPerPart, rewriter);
        if (failed(activity)) {
          return failure();
        }
        bool anyLane = activity->first;
        int64_t activeInChunk = activity->second;
        if (!anyLane) {
          break;
        }
        bool tooManyResults = results.size() >= resultTypes.size();
        if (tooManyResults) {
          return rewriter.notifyMatchFailure(
              op, "create_mask produced too many physical masks");
        }
        FailureOr<Value> mask = materializeConstantMaskValue(
            op, resultTypes[results.size()], activeInChunk, lanesPerPart,
            rewriter);
        if (failed(mask)) {
          return failure();
        }
        results.push_back(*mask);
      }
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "create_mask physical result count mismatch");
    }
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMICreateMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto activeConstant =
        op.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout ||
        !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity())) {
      return rewriter.notifyMatchFailure(
          op, "create_mask requires concrete layout and granularity");
    }
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(resultVMIType);
    FailureOr<int64_t> lanesPerPart =
        failed(physicalGranularity)
            ? FailureOr<int64_t>(failure())
            : getMaskLanesPerPart(*physicalGranularity);
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "create_mask requires known physical mask lanes per part");
    }

    if (!activeConstant) {
      return lowerDynamicCreateMask(op, adaptor, resultVMIType, layout,
                                    *lanesPerPart, rewriter);
    }

    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr) {
      return rewriter.notifyMatchFailure(
          op, "create_mask active_lanes must be an integer constant");
    }

    int64_t activeLanes = activeAttr.getInt();
    if (activeLanes < 0) {
      activeLanes = 0;
    }
    if (activeLanes > resultVMIType.getElementCount()) {
      activeLanes = resultVMIType.getElementCount();
    }

    return lowerConstantCreateMask(op, activeLanes, resultVMIType, layout,
                                   *lanesPerPart, rewriter);
  }
};

struct OneToNVMICreateGroupMaskOpPattern
    : OneToNOpConversionPattern<VMICreateGroupMaskOp> {
  using OneToNOpConversionPattern<
      VMICreateGroupMaskOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Value>> materializeGroupMaskResults(
      VMICreateGroupMaskOp op,
      ArrayRef<ConstantMaskChunkMaterialization> materializations,
      ArrayRef<Type> resultTypes, StringRef overflowDiagnostic,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         materializations) {
      bool tooManyMasks = results.size() >= resultTypes.size();
      if (tooManyMasks) {
        return rewriter.notifyMatchFailure(op, overflowDiagnostic);
      }
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType) {
        return rewriter.notifyMatchFailure(
            op, "create_group_mask result must be mask");
      }
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to materialize create_group_mask physical chunk");
      }
      results.push_back(*mask);
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "create_group_mask physical result count mismatch");
    }
    return results;
  }

  LogicalResult lowerDynamicMask(
      VMICreateGroupMaskOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIMaskType resultVMIType,
      VMILayoutAttr resultLayout, ArrayRef<Type> resultTypes) const {
    FailureOr<Value> active = getSingleValue(
        op, adaptor.getActiveElemsPerGroup(),
        "create_group_mask active_elems_per_group must convert to one value",
        rewriter);
    if (failed(active)) {
      return failure();
    }

    if (resultLayout && resultLayout.isDeinterleaved()) {
      VMILayoutAttr contiguousLayout =
          VMILayoutAttr::getContiguous(op.getContext());
      auto contiguousType = VMIMaskType::get(
          op.getContext(), resultVMIType.getElementCount(),
          resultVMIType.getGranularity(), contiguousLayout);
      FailureOr<SmallVector<Value>> contiguousParts =
          materializeDynamicGroupMaskForType(op, *active, contiguousType,
                                             resultTypes, rewriter);
      if (failed(contiguousParts)) {
        return failure();
      }
      return replaceMaterializedResults(
          rewriter, op,
          materializeMaskLayoutConversion(op, *contiguousParts, resultTypes,
                                          contiguousLayout, resultLayout,
                                          rewriter),
          *this->getTypeConverter());
    }

    FailureOr<SmallVector<Value>> results = materializeDynamicGroupMaskForType(
        op, *active, resultVMIType, resultTypes, rewriter);
    if (failed(results)) {
      return failure();
    }
    return replaceMaterializedResults(rewriter, op, std::move(results),
                                      *this->getTypeConverter());
  }

  LogicalResult lowerConstantMask(
      VMICreateGroupMaskOp op, OneToNPatternRewriter &rewriter,
      ArrayRef<Type> resultTypes) const {
    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeGroupMaskMaterialization(op, &reason);
    if (failed(materializations)) {
      return rewriter.notifyMatchFailure(
          op, Twine("create_group_mask ") + reason);
    }

    FailureOr<SmallVector<Value>> results = materializeGroupMaskResults(
        op, *materializations, resultTypes,
        "create_group_mask produced too many physical masks", rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<SmallVector<Value>> buildFactor4ContiguousParts(
      VMICreateGroupMaskOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIMaskType contiguousType,
      ArrayRef<Type> resultTypes) const {
    auto activeConstant =
        op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      FailureOr<Value> active = getSingleValue(
          op, adaptor.getActiveElemsPerGroup(),
          "create_group_mask active_elems_per_group must convert to one value",
          rewriter);
      if (failed(active)) {
        return failure();
      }
      return materializeDynamicGroupMaskForType(
          op, *active, contiguousType, resultTypes, rewriter);
    }

    std::string contiguousReason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeGroupMaskMaterializationForType(op, contiguousType,
                                               &contiguousReason);
    if (failed(materializations)) {
      return rewriter.notifyMatchFailure(
          op, Twine("create_group_mask ") + contiguousReason);
    }
    return materializeGroupMaskResults(
        op, *materializations, resultTypes,
        "create_group_mask produced too many contiguous masks", rewriter);
  }

  LogicalResult lowerFactor4Block(
      VMICreateGroupMaskOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIMaskType resultVMIType,
      VMILayoutAttr resultLayout, ArrayRef<Type> resultTypes) const {
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(op.getContext());
    auto contiguousType =
        VMIMaskType::get(op.getContext(), resultVMIType.getElementCount(),
                         resultVMIType.getGranularity(), contiguousLayout);
    FailureOr<SmallVector<Value>> contiguousParts =
        buildFactor4ContiguousParts(op, adaptor, rewriter, contiguousType,
                                    resultTypes);
    if (failed(contiguousParts)) {
      return failure();
    }
    bool resultCountMismatch = contiguousParts->size() != resultTypes.size();
    if (resultCountMismatch) {
      return rewriter.notifyMatchFailure(
          op, "create_group_mask contiguous physical result count mismatch");
    }
    return replaceMaterializedResults(
        rewriter, op,
        materializeMaskLayoutConversion(op, *contiguousParts, resultTypes,
                                        contiguousLayout, resultLayout,
                                        rewriter),
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMICreateGroupMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool needsFactor4ContiguousMaterialization =
        resultLayout && resultLayout.isBlockDeinterleaved() &&
        resultLayout.getFactor() == 4;
    if (needsFactor4ContiguousMaterialization) {
      return lowerFactor4Block(op, adaptor, rewriter, resultVMIType,
                               resultLayout, resultTypes);
    }

    auto activeConstant =
        op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      return lowerDynamicMask(op, adaptor, rewriter, resultVMIType,
                              resultLayout, resultTypes);
    }
    return lowerConstantMask(op, rewriter, resultTypes);
  }
};


