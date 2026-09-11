// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTODataLayoutInternals.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> materializeLaneStrideToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, PatternRewriter &rewriter) {
  FailureOr<unsigned> elementBits = validateDenseLaneStrideShape(
      op, sourceParts, resultTypes, elementType, laneStride, false, rewriter);
  if (failed(elementBits)) {
    return failure();
  }

  unsigned carrierBits =
      static_cast<unsigned>(*elementBits * static_cast<unsigned>(laneStride));
  FailureOr<VRegType> sourceCarrier =
      getUnsignedCarrierVRegType(rewriter.getContext(), carrierBits);
  if (failed(sourceCarrier)) {
    return failure();
  }

  return materializeLaneStrideResultList(
      op, sourceParts, resultTypes, *elementBits, carrierBits, *sourceCarrier,
      laneStride, rewriter);
}

static LogicalResult checkGroupSlotLaneStrideContract(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidArity = sourceParts.size() != resultTypes.size() ||
                     sourceParts.empty();
  if (invalidArity) {
    return fail("group-slot lane_stride materialization requires matching "
                "non-empty source/result physical arity");
  }
  bool unsupportedStride =
      (sourceStride != 1 && sourceStride != 2 && sourceStride != 4) ||
      (resultStride != 1 && resultStride != 2 && resultStride != 4);
  if (unsupportedStride) {
    return fail("unsupported group-slot lane_stride factor");
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  int64_t maxStride = std::max(sourceStride, resultStride);
  bool unsupportedCarrier =
      (elementBits != 8 && elementBits != 16) || elementBits * maxStride > 32;
  if (unsupportedCarrier) {
    return fail("unsupported group-slot lane_stride carrier shape");
  }
  return success();
}

FailureOr<SmallVector<Value>> materializeGroupSlotLaneStride(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  if (failed(checkGroupSlotLaneStrideContract(
          op, sourceParts, resultTypes, elementType, sourceStride, resultStride,
          rewriter))) {
    return failure();
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [source, resultType] :
       llvm::zip_equal(sourceParts, resultTypes)) {
    FailureOr<Value> result = materializeGroupSlotLaneStridePart(
        op, source, resultType, elementType, sourceStride, resultStride,
        rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast group-slot result carrier");
    }
    results.push_back(*result);
  }
  return results;
}

static FailureOr<std::optional<SmallVector<Value>>> forwardIdentityLayoutParts(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(
      SmallVector<Value>(sourceParts.begin(), sourceParts.end()));
}

static std::optional<SmallVector<Value>>
forwardBlockLayoutCastInputs(ValueRange sourceParts, TypeRange resultTypes) {
  bool invalidSourceArity = sourceParts.size() != 1;
  if (invalidSourceArity) {
    return std::nullopt;
  }
  auto cast = sourceParts.front().getDefiningOp<UnrealizedConversionCastOp>();
  bool invalidCast = !cast || cast.getInputs().size() != resultTypes.size();
  if (invalidCast) {
    return std::nullopt;
  }
  for (auto [input, resultType] : llvm::zip_equal(cast.getInputs(), resultTypes)) {
    bool typeMismatch = input.getType() != resultType;
    if (typeMismatch) {
      return std::nullopt;
    }
  }
  return SmallVector<Value>(cast.getInputs().begin(), cast.getInputs().end());
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeGroupSlotLaneStrideLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  bool supported =
      sourceLayout.isGroupSlots() && resultLayout.isGroupSlots() &&
      sourceLayout.getNumGroups() == resultLayout.getNumGroups() &&
      sourceLayout.getSlots() == 8 && resultLayout.getSlots() == 8;
  if (!supported) {
    return std::optional<SmallVector<Value>>{};
  }
  FailureOr<SmallVector<Value>> result = materializeGroupSlotLaneStride(
      op, sourceParts, resultTypes, sourceVMIElementType,
      sourceLayout.getLaneStride(), resultLayout.getLaneStride(), rewriter);
  if (failed(result)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*result));
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeBlockLayoutForwarding(Operation *op, ValueRange sourceParts,
                                 TypeRange resultTypes,
                                 VMILayoutAttr sourceLayout,
                                 VMILayoutAttr resultLayout,
                                 PatternRewriter &rewriter) {
  auto isBlockDeinterleaved = [](VMILayoutAttr layout, int64_t factor) {
    return layout.isBlockDeinterleaved() && layout.getFactor() == factor;
  };
  bool contiguousToBlock =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      (isBlockDeinterleaved(resultLayout, 2) ||
       isBlockDeinterleaved(resultLayout, 4));
  bool blockToContiguous =
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
      (isBlockDeinterleaved(sourceLayout, 2) ||
       isBlockDeinterleaved(sourceLayout, 4));
  if (!contiguousToBlock && !blockToContiguous) {
    return std::optional<SmallVector<Value>>{};
  }
  if (std::optional<SmallVector<Value>> castInputs =
          forwardBlockLayoutCastInputs(sourceParts, resultTypes)) {
    return std::optional<SmallVector<Value>>(std::move(*castInputs));
  }
  return forwardIdentityLayoutParts(op, sourceParts, resultTypes, rewriter);
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeSimpleDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "layout materialization requires assigned source/result layouts");
    return failure();
  }

  if (sourceLayout == resultLayout) {
    return forwardIdentityLayoutParts(op, sourceParts, resultTypes, rewriter);
  }

  bool oneLaneContiguousToGroup =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isGroupSlots() && resultLayout.getNumGroups() == 1 &&
      resultLayout.getSlots() == 1;
  bool oneLaneGroupToContiguous =
      sourceLayout.isGroupSlots() && sourceLayout.getNumGroups() == 1 &&
      sourceLayout.getSlots() == 1 && resultLayout.isContiguous() &&
      resultLayout.getLaneStride() == 1;
  if (oneLaneContiguousToGroup || oneLaneGroupToContiguous) {
    return forwardIdentityLayoutParts(op, sourceParts, resultTypes, rewriter);
  }

  FailureOr<std::optional<SmallVector<Value>>> groupSlot =
      materializeGroupSlotLaneStrideLayout(
          op, sourceParts, resultTypes, sourceLayout, resultLayout,
          sourceVMIElementType, rewriter);
  if (failed(groupSlot)) {
    return failure();
  }
  if (groupSlot->has_value()) {
    return std::optional<SmallVector<Value>>(std::move(**groupSlot));
  }

  FailureOr<std::optional<SmallVector<Value>>> block =
      materializeBlockLayoutForwarding(op, sourceParts, resultTypes,
                                       sourceLayout, resultLayout, rewriter);
  if (failed(block)) {
    return failure();
  }
  if (block->has_value()) {
    return std::optional<SmallVector<Value>>(std::move(**block));
  }

  return std::optional<SmallVector<Value>>{};
}

static FailureOr<SmallVector<Value>> materializeDeinterleaved2ToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  bool invalidSource = sourceParts.empty() || sourceParts.size() % 2 != 0 ||
                       resultTypes.empty();
  if (invalidSource) {
    return rewriter.notifyMatchFailure(
        op, "deinterleaved=2 to contiguous materialization requires 2*N "
            "source parts and at least one result part");
  }
  int64_t groups = sourceParts.size() / 2;
  bool resultExceedsSource =
      resultTypes.size() > static_cast<size_t>(2 * groups);
  if (resultExceedsSource) {
    return rewriter.notifyMatchFailure(
        op, "deinterleaved=2 to contiguous materialization result arity "
            "exceeds source footprint");
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t i = 0; i < groups && results.size() < resultTypes.size(); ++i) {
    Value lhs = sourceParts[i];
    Value rhs = sourceParts[groups + i];
    Type lhsType = lhs.getType();
    if (lhsType != rhs.getType()) {
      return rewriter.notifyMatchFailure(
          op, "vintlv requires matching source part types");
    }
    Type lowType = resultTypes[results.size()];
    bool hasHighResult = results.size() + 1 < resultTypes.size();
    Type highType = hasHighResult ? resultTypes[results.size() + 1] : lowType;
    if (lhsType != lowType || lhsType != highType) {
      return rewriter.notifyMatchFailure(
          op, "vintlv requires operands and results to share one type");
    }
    auto materialize = rewriter.create<VintlvOp>(
        op->getLoc(), lowType, highType, lhs, rhs);
    results.push_back(materialize.getLow());
    if (hasHighResult) {
      results.push_back(materialize.getHigh());
    }
  }
  return results;
}

static LogicalResult validateContiguousToDeinterleaved2Shape(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter, int64_t &groups) {
  bool invalidResult = sourceParts.empty() || resultTypes.empty() ||
                       resultTypes.size() % 2 != 0;
  if (invalidResult) {
    return rewriter.notifyMatchFailure(
        op, "contiguous to deinterleaved=2 materialization requires at least "
            "one source part and 2*N result parts");
  }
  groups = resultTypes.size() / 2;
  bool sourceExceedsResult =
      sourceParts.size() > static_cast<size_t>(2 * groups);
  if (sourceExceedsResult) {
    return rewriter.notifyMatchFailure(
        op, "contiguous to deinterleaved=2 materialization source footprint "
            "exceeds result arity");
  }
  return success();
}

static FailureOr<std::pair<Value, Value>> materializeContiguousToDeinterleaved2Group(
    Operation *op, Value lhs, Value rhs, Type lowType, Type highType,
    PatternRewriter &rewriter) {
  bool mismatchedTypes = lhs.getType() != rhs.getType() ||
                         lhs.getType() != lowType || lhs.getType() != highType;
  if (mismatchedTypes) {
    return rewriter.notifyMatchFailure(
        op, "vdintlv requires operands and results to share one type");
  }
  auto materialize = rewriter.create<VdintlvOp>(op->getLoc(), lowType, highType,
                                                 lhs, rhs);
  return std::make_pair(materialize.getLow(), materialize.getHigh());
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeinterleaved2(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = 0;
  if (failed(validateContiguousToDeinterleaved2Shape(
          op, sourceParts, resultTypes, rewriter, groups))) {
    return failure();
  }
  SmallVector<Value> part0;
  SmallVector<Value> part1;
  part0.reserve(groups);
  part1.reserve(groups);
  for (int64_t i = 0; i < groups; ++i) {
    size_t lhsIndex = 2 * i;
    if (lhsIndex >= sourceParts.size()) {
      return rewriter.notifyMatchFailure(
          op, "contiguous to deinterleaved=2 materialization missing source "
              "part");
    }
    size_t rhsIndex = lhsIndex + 1 < sourceParts.size() ? lhsIndex + 1
                                                          : lhsIndex;
    Value lhs = sourceParts[lhsIndex];
    Value rhs = sourceParts[rhsIndex];
    FailureOr<std::pair<Value, Value>> materialize =
        materializeContiguousToDeinterleaved2Group(
            op, lhs, rhs, resultTypes[i], resultTypes[groups + i], rewriter);
    if (failed(materialize)) {
      return failure();
    }
    part0.push_back(materialize->first);
    part1.push_back(materialize->second);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  results.append(part0);
  results.append(part1);
  return results;
}

FailureOr<std::optional<SmallVector<Value>>> materializeDeinterleaved2Layout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  auto isElementDeinterleaved = [](VMILayoutAttr layout) {
    return layout.isDeinterleaved() && layout.getFactor() == 2 &&
           layout.getLaneStride() == 1;
  };
  bool toContiguous = sourceLayout && sourceLayout.isDeinterleaved() &&
                      isElementDeinterleaved(sourceLayout) && resultLayout &&
                      resultLayout.isContiguous() &&
                      resultLayout.getLaneStride() == 1;
  bool fromContiguous = sourceLayout && sourceLayout.isContiguous() &&
                        sourceLayout.getLaneStride() == 1 && resultLayout &&
                        resultLayout.isDeinterleaved() &&
                        isElementDeinterleaved(resultLayout);
  if (!toContiguous && !fromContiguous) {
    return std::optional<SmallVector<Value>>{};
  }

  if (toContiguous) {
    FailureOr<SmallVector<Value>> results = materializeDeinterleaved2ToContiguous(
        op, sourceParts, resultTypes, rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  } else {
    FailureOr<SmallVector<Value>> results =
        materializeContiguousToDeinterleaved2(op, sourceParts, resultTypes,
                                              rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }
}

FailureOr<std::optional<SmallVector<Value>>> materializeDataLaneStrideConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    return std::optional<SmallVector<Value>>{};
  }
  bool contiguousToLaneStride =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() != 1;
  if (contiguousToLaneStride) {
    FailureOr<SmallVector<Value>> result = materializeContiguousToLaneStride(
        op, sourceParts, resultTypes, sourceVMIElementType,
        resultLayout.getLaneStride(), rewriter);
    if (failed(result)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*result));
  }
  bool laneStrideToContiguous =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() != 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (laneStrideToContiguous) {
    FailureOr<SmallVector<Value>> result = materializeLaneStrideToContiguous(
        op, sourceParts, resultTypes, sourceVMIElementType,
        sourceLayout.getLaneStride(), rewriter);
    if (failed(result)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*result));
  }
  return std::optional<SmallVector<Value>>{};
}

FailureOr<SmallVector<Value>> materializeDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter);

struct DataLayoutIntermediatePlan {
  enum class Kind { Contiguous, Deinterleaved };
  Kind kind;
  size_t intermediateCount;
};

static bool isContiguousLaneStrideLayout(VMILayoutAttr layout,
                                         int64_t laneStride) {
  return layout.isContiguous() && layout.getLaneStride() == laneStride;
}

static bool isDeinterleavedUnitStrideLayout(VMILayoutAttr layout,
                                            int64_t factor) {
  return layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

static bool isSupportedDeinterleavedIntermediateLayout(VMILayoutAttr layout) {
  return layout.isDeinterleaved() && layout.getLaneStride() == 1 &&
         (layout.getFactor() == 2 || layout.getFactor() == 4);
}

static std::optional<DataLayoutIntermediatePlan>
getDataLayoutIntermediatePlan(VMILayoutAttr sourceLayout,
                              VMILayoutAttr resultLayout,
                              size_t sourcePartCount) {
  bool deint2ToLaneStride =
      isDeinterleavedUnitStrideLayout(sourceLayout, 2) &&
      isContiguousLaneStrideLayout(resultLayout, 2);
  bool laneStrideToDeint2 =
      isContiguousLaneStrideLayout(sourceLayout, 2) &&
      isDeinterleavedUnitStrideLayout(resultLayout, 2);
  bool laneStride2ToLaneStride4 =
      isContiguousLaneStrideLayout(sourceLayout, 2) &&
      isContiguousLaneStrideLayout(resultLayout, 4);
  bool laneStride4ToLaneStride2 =
      isContiguousLaneStrideLayout(sourceLayout, 4) &&
      isContiguousLaneStrideLayout(resultLayout, 2);
  bool useContiguousIntermediate =
      deint2ToLaneStride || laneStrideToDeint2 || laneStride2ToLaneStride4 ||
      laneStride4ToLaneStride2;
  if (useContiguousIntermediate) {
    bool needsPacking = !deint2ToLaneStride;
    size_t intermediateCount = needsPacking ? (sourcePartCount + 1) / 2
                                            : sourcePartCount;
    return DataLayoutIntermediatePlan{
        DataLayoutIntermediatePlan::Kind::Contiguous, intermediateCount};
  }

  bool useDeinterleavedIntermediate =
      isSupportedDeinterleavedIntermediateLayout(sourceLayout) &&
      isSupportedDeinterleavedIntermediateLayout(resultLayout);
  if (useDeinterleavedIntermediate) {
    return DataLayoutIntermediatePlan{
        DataLayoutIntermediatePlan::Kind::Deinterleaved, sourcePartCount};
  }
  return std::nullopt;
}

static FailureOr<SmallVector<Value>> materializeDataLayoutThroughContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter,
    const DataLayoutIntermediatePlan &plan) {
  VMILayoutAttr contiguous =
      VMILayoutAttr::getContiguous(rewriter.getContext());
  SmallVector<Type> intermediateTypes(
      plan.intermediateCount, sourceParts.front().getType());
  FailureOr<SmallVector<Value>> dense = materializeDataLayoutConversion(
      op, sourceParts, intermediateTypes, sourceLayout, contiguous,
      sourceVMIElementType, rewriter);
  if (failed(dense)) {
    return failure();
  }
  return materializeDataLayoutConversion(op, *dense, resultTypes, contiguous,
                                         resultLayout, sourceVMIElementType,
                                         rewriter);
}

static FailureOr<SmallVector<Value>> materializeDataLayoutThroughDeinterleaved(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  VMILayoutAttr contiguous =
      VMILayoutAttr::getContiguous(rewriter.getContext());
  FailureOr<SmallVector<Value>> dense = materializeDataLayoutConversion(
      op, sourceParts, resultTypes, sourceLayout, contiguous,
      sourceVMIElementType, rewriter);
  if (failed(dense)) {
    return failure();
  }
  return materializeDataLayoutConversion(op, *dense, resultTypes, contiguous,
                                         resultLayout, sourceVMIElementType,
                                         rewriter);
}

FailureOr<std::optional<SmallVector<Value>>>
materializeDataLayoutViaContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  std::optional<DataLayoutIntermediatePlan> plan =
      getDataLayoutIntermediatePlan(sourceLayout, resultLayout,
                                    sourceParts.size());
  if (!plan) {
    return std::optional<SmallVector<Value>>{};
  }
  if (sourceParts.empty()) {
    return failure();
  }

  if (plan->kind == DataLayoutIntermediatePlan::Kind::Deinterleaved) {
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutThroughDeinterleaved(
            op, sourceParts, resultTypes, sourceLayout, resultLayout,
            sourceVMIElementType, rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }

  FailureOr<SmallVector<Value>> results = materializeDataLayoutThroughContiguous(
      op, sourceParts, resultTypes, sourceLayout, resultLayout,
      sourceVMIElementType, rewriter, *plan);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

struct DataLayoutMaterializationContext {
  Operation *op;
  ValueRange sourceParts;
  TypeRange resultTypes;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  Type sourceVMIElementType;
  PatternRewriter &rewriter;
};

using DataLayoutMaterializationResult =
    FailureOr<std::optional<SmallVector<Value>>>;

static bool didHandleDataLayoutMaterialization(
    const DataLayoutMaterializationResult &result) {
  return failed(result) || result->has_value();
}

static bool isElementDeinterleavedDataLayout(VMILayoutAttr layout,
                                             int64_t factor) {
  return layout && layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

static FailureOr<SmallVector<Value>> materializeDeint4DataToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 4;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t group = 0; group < groups; ++group) {
    Value p0 = sourceParts[group];
    Value p1 = sourceParts[groups + group];
    Value p2 = sourceParts[2 * groups + group];
    Value p3 = sourceParts[3 * groups + group];
    Type chunkType = p0.getType();
    if (p1.getType() != chunkType || p2.getType() != chunkType ||
        p3.getType() != chunkType) {
      (void)rewriter.notifyMatchFailure(
          op, "vintlv deinterleaved=4 requires matching source part types");
      return failure();
    }
    for (size_t offset = 0; offset < 4; ++offset) {
      if (resultTypes[4 * group + offset] != chunkType) {
        (void)rewriter.notifyMatchFailure(
            op, "vintlv requires operands and results to share one type");
        return failure();
      }
    }
    auto even = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                          p0, p2);
    auto odd = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                         p1, p3);
    auto low = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                         even.getLow(), odd.getLow());
    auto high = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                          even.getHigh(), odd.getHigh());
    results.append(
        {low.getLow(), low.getHigh(), high.getLow(), high.getHigh()});
  }
  return results;
}

static FailureOr<std::array<Value, 4>> materializeContiguousToDeint4DataGroup(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t groups, int64_t group, PatternRewriter &rewriter) {
  Value s0 = sourceParts[4 * group];
  Value s1 = sourceParts[4 * group + 1];
  Value s2 = sourceParts[4 * group + 2];
  Value s3 = sourceParts[4 * group + 3];
  Type chunkType = s0.getType();
  if (s0.getType() != s1.getType() || s0.getType() != s2.getType() ||
      s0.getType() != s3.getType()) {
    (void)rewriter.notifyMatchFailure(
        op, "vdintlv deinterleaved=4 requires matching source part types");
    return failure();
  }
  for (size_t offset = 0; offset < 4; ++offset) {
    if (resultTypes[group + offset * groups] != chunkType) {
      (void)rewriter.notifyMatchFailure(
          op, "vdintlv requires operands and results to share one type");
      return failure();
    }
  }
  auto low = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType, s0,
                                        s1);
  auto high = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType, s2,
                                         s3);
  auto even = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType,
                                         low.getLow(), high.getLow());
  auto odd = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType,
                                        low.getHigh(), high.getHigh());
  return std::array<Value, 4>{
      even.getLow(), odd.getLow(), even.getHigh(), odd.getHigh()};
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeint4Data(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 4;
  SmallVector<Value> part0;
  SmallVector<Value> part1;
  SmallVector<Value> part2;
  SmallVector<Value> part3;
  part0.reserve(groups);
  part1.reserve(groups);
  part2.reserve(groups);
  part3.reserve(groups);
  for (int64_t group = 0; group < groups; ++group) {
    FailureOr<std::array<Value, 4>> groupValues =
        materializeContiguousToDeint4DataGroup(op, sourceParts, resultTypes,
                                               groups, group, rewriter);
    if (failed(groupValues)) {
      return failure();
    }
    part0.push_back((*groupValues)[0]);
    part1.push_back((*groupValues)[1]);
    part2.push_back((*groupValues)[2]);
    part3.push_back((*groupValues)[3]);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  results.append(part0);
  results.append(part1);
  results.append(part2);
  results.append(part3);
  return results;
}

static DataLayoutMaterializationResult materializeDeinterleaved4DataLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  bool contiguousToDeint4 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      isElementDeinterleavedDataLayout(resultLayout, 4);
  bool deint4ToContiguous =
      isElementDeinterleavedDataLayout(sourceLayout, 4) &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (!contiguousToDeint4 && !deint4ToContiguous) {
    return std::optional<SmallVector<Value>>{};
  }
  bool invalidArity = sourceParts.empty() ||
                      sourceParts.size() != resultTypes.size() ||
                      resultTypes.size() % 4 != 0;
  if (invalidArity) {
    (void)rewriter.notifyMatchFailure(
        op, "deinterleaved=4 data layout materialization requires 4*N parts");
    return failure();
  }
  FailureOr<SmallVector<Value>> results =
      deint4ToContiguous
          ? materializeDeint4DataToContiguous(op, sourceParts, resultTypes,
                                              rewriter)
          : materializeContiguousToDeint4Data(op, sourceParts, resultTypes,
                                              rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static DataLayoutMaterializationResult
tryDataLayoutMaterializers(const DataLayoutMaterializationContext &context) {
  DataLayoutMaterializationResult simple =
      materializeSimpleDataLayoutConversion(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout,
          context.sourceVMIElementType, context.rewriter);
  if (didHandleDataLayoutMaterialization(simple)) {
    return simple;
  }
  DataLayoutMaterializationResult deinterleaved2 =
      materializeDeinterleaved2Layout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleDataLayoutMaterialization(deinterleaved2)) {
    return deinterleaved2;
  }
  DataLayoutMaterializationResult deinterleaved4 =
      materializeDeinterleaved4DataLayout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleDataLayoutMaterialization(deinterleaved4)) {
    return deinterleaved4;
  }
  DataLayoutMaterializationResult laneStride =
      materializeDataLaneStrideConversion(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout,
          context.sourceVMIElementType, context.rewriter);
  if (didHandleDataLayoutMaterialization(laneStride)) {
    return laneStride;
  }
  return materializeDataLayoutViaContiguous(
      context.op, context.sourceParts, context.resultTypes,
      context.sourceLayout, context.resultLayout, context.sourceVMIElementType,
      context.rewriter);
}

FailureOr<SmallVector<Value>> materializeDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  DataLayoutMaterializationContext context{
      op, sourceParts, resultTypes, sourceLayout, resultLayout,
      sourceVMIElementType, rewriter};
  FailureOr<std::optional<SmallVector<Value>>> results =
      tryDataLayoutMaterializers(context);
  if (failed(results)) {
    return failure();
  }
  if (results->has_value()) {
    return std::move(**results);
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI data layout materialization");
  return failure();
}

FailureOr<SmallVector<Value>> materializeEnsureLayoutConversion(
    Operation *op, ValueRange sourceParts, VMIVRegType sourceType,
    VMIVRegType resultType, const TypeConverter &typeConverter,
    PatternRewriter &rewriter) {
  VMILayoutSupport supports;
  std::string supportReason;
  if (failed(supports.getEnsureLayoutFact(sourceType, resultType,
                                          &supportReason))) {
    (void)rewriter.notifyMatchFailure(
        op, Twine("ensure_layout has no registered materialization support: ") +
                supportReason);
    return failure();
  }

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "ensure_layout requires assigned source/result layouts");
    return failure();
  }

  SmallVector<Type> resultTypes;
  if (failed(typeConverter.convertType(resultType, resultTypes))) {
    return failure();
  }
  return materializeDataLayoutConversion(op, sourceParts, resultTypes,
                                         sourceLayout, resultLayout,
                                         sourceType.getElementType(), rewriter);
}

enum class PredicateInterleaveKind { Interleave, Deinterleave };

static FailureOr<std::pair<Value, Value>> createPredicateInterleave(
    Location loc, Type lowType, Type highType, Value lhs, Value rhs,
    PredicateInterleaveKind kind, PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(lowType);
  if (!maskType || highType != lowType) {
    return failure();
  }
  bool deinterleave = kind == PredicateInterleaveKind::Deinterleave;
  if (maskType.isB8()) {
    if (deinterleave) {
      auto op = rewriter.create<PdintlvB8Op>(loc, lowType, highType, lhs, rhs);
      return std::make_pair(op.getLow(), op.getHigh());
    }
    auto op = rewriter.create<PintlvB8Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB16()) {
    if (deinterleave) {
      auto op = rewriter.create<PdintlvB16Op>(loc, lowType, highType, lhs, rhs);
      return std::make_pair(op.getLow(), op.getHigh());
    }
    auto op = rewriter.create<PintlvB16Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB32()) {
    if (deinterleave) {
      auto op = rewriter.create<PdintlvB32Op>(loc, lowType, highType, lhs, rhs);
      return std::make_pair(op.getLow(), op.getHigh());
    }
    auto op = rewriter.create<PintlvB32Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  return failure();
}

FailureOr<std::pair<Value, Value>>
createPredicateDintlv(Location loc, Type lowType, Type highType, Value lhs,
                      Value rhs, PatternRewriter &rewriter) {
  return createPredicateInterleave(loc, lowType, highType, lhs, rhs,
                                   PredicateInterleaveKind::Deinterleave,
                                   rewriter);
}

FailureOr<std::pair<Value, Value>>
createPredicateIntlv(Location loc, Type lowType, Type highType, Value lhs,
                     Value rhs, PatternRewriter &rewriter) {
  return createPredicateInterleave(loc, lowType, highType, lhs, rhs,
                                   PredicateInterleaveKind::Interleave,
                                   rewriter);
}

static FailureOr<SmallVector<Value>> materializeDeinterleaved2MaskToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 2;
  SmallVector<Value> results;
  results.reserve(sourceParts.size());
  for (int64_t i = 0; i < groups; ++i) {
    FailureOr<std::pair<Value, Value>> materialize = createPredicateIntlv(
        op->getLoc(), resultTypes[2 * i], resultTypes[2 * i + 1],
        sourceParts[i], sourceParts[groups + i], rewriter);
    if (failed(materialize)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate intlv mask type");
    }
    results.append({materialize->first, materialize->second});
  }
  return results;
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeinterleaved2Mask(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 2;
  SmallVector<Value> part0;
  SmallVector<Value> part1;
  part0.reserve(groups);
  part1.reserve(groups);
  for (int64_t i = 0; i < groups; ++i) {
    FailureOr<std::pair<Value, Value>> materialize = createPredicateDintlv(
        op->getLoc(), resultTypes[i], resultTypes[groups + i],
        sourceParts[2 * i], sourceParts[2 * i + 1], rewriter);
    if (failed(materialize)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate dintlv mask type");
    }
    part0.push_back(materialize->first);
    part1.push_back(materialize->second);
  }
  SmallVector<Value> results;
  results.append(part0);
  results.append(part1);
  return results;
}

enum class Deinterleaved2MaskLayoutDirection {
  Unsupported,
  ToContiguous,
  FromContiguous
};

static Deinterleaved2MaskLayoutDirection getDeinterleaved2MaskDirection(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) {
  bool sourceIsDeinterleaved2 =
      sourceLayout && sourceLayout.isDeinterleaved() &&
      sourceLayout.getFactor() == 2 && sourceLayout.getLaneStride() == 1;
  bool resultIsDeinterleaved2 =
      resultLayout && resultLayout.isDeinterleaved() &&
      resultLayout.getFactor() == 2 && resultLayout.getLaneStride() == 1;
  bool toContiguous = sourceIsDeinterleaved2 && resultLayout &&
                      resultLayout.isContiguous() &&
                      resultLayout.getLaneStride() == 1;
  bool fromContiguous = sourceLayout && sourceLayout.isContiguous() &&
                        sourceLayout.getLaneStride() == 1 &&
                        resultIsDeinterleaved2;
  if (toContiguous) {
    return Deinterleaved2MaskLayoutDirection::ToContiguous;
  }
  if (fromContiguous) {
    return Deinterleaved2MaskLayoutDirection::FromContiguous;
  }
  return Deinterleaved2MaskLayoutDirection::Unsupported;
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeDeinterleaved2MaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  Deinterleaved2MaskLayoutDirection direction =
      getDeinterleaved2MaskDirection(sourceLayout, resultLayout);
  if (direction == Deinterleaved2MaskLayoutDirection::Unsupported) {
    return std::optional<SmallVector<Value>>{};
  }
  bool invalidArity = sourceParts.size() != resultTypes.size() ||
                      sourceParts.empty() || sourceParts.size() % 2 != 0;
  if (invalidArity) {
    (void)rewriter.notifyMatchFailure(
        op, "deinterleaved=2 mask layout materialization requires 2*N parts");
    return failure();
  }
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }

  if (direction == Deinterleaved2MaskLayoutDirection::ToContiguous) {
    FailureOr<SmallVector<Value>> results =
        materializeDeinterleaved2MaskToContiguous(op, sourceParts, resultTypes,
                                                  rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }
  FailureOr<SmallVector<Value>> results =
      materializeContiguousToDeinterleaved2Mask(op, sourceParts, resultTypes,
                                                rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static FailureOr<MaskType> getMaskLaneStrideResultType(
    Operation *op, Type resultType, StringRef diagnostic,
    PatternRewriter &rewriter);

static FailureOr<SmallVector<Value>> materializeMaskLaneStrideUnpack(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t laneStride, PatternRewriter &rewriter) {
  bool resultExceedsSource =
      static_cast<int64_t>(resultTypes.size()) >
      static_cast<int64_t>(sourceParts.size()) * laneStride;
  if (resultExceedsSource) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride unpack materialization result arity "
            "does not fit source arity");
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  StringAttr lower = rewriter.getStringAttr("LOWER");
  StringAttr higher = rewriter.getStringAttr("HIGHER");
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<MaskType> maskType = getMaskLaneStrideResultType(
        op, resultType, "dense mask lane_stride unpack requires mask result type",
        rewriter);
    if (failed(maskType)) {
      return failure();
    }
    int64_t safeLaneStride = laneStride > 0 ? laneStride : 1;
    int64_t sourceIndex = resultIndex / safeLaneStride;
    int64_t part = resultIndex % safeLaneStride;
    Value source = sourceParts[sourceIndex];
    StringAttr firstPart = laneStride == 4
                               ? (part >= 2 ? higher : lower)
                               : (part == 1 ? higher : lower);
    Value current = rewriter
                        .create<PunpackOp>(op->getLoc(), *maskType, source,
                                           firstPart)
                        .getResult();
    if (laneStride == 4) {
      current = rewriter.create<PunpackOp>(
          op->getLoc(), *maskType, current, part % 2 == 0 ? lower : higher);
    }
    results.push_back(current);
  }
  return results;
}

static FailureOr<MaskType> getMaskLaneStrideResultType(
    Operation *op, Type resultType, StringRef diagnostic,
    PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(resultType);
  if (!maskType) {
    (void)rewriter.notifyMatchFailure(op, diagnostic);
    return failure();
  }
  return maskType;
}

struct MaskLaneStridePackContext {
  Operation *op;
  PatternRewriter &rewriter;
  StringAttr lower;
  StringAttr higher;
  Value allTrue;

  FailureOr<Value> merge(Value lhs, Value rhs) {
    if (!allTrue) {
      FailureOr<Value> mask = createAllTrueMask(
          op->getLoc(), cast<MaskType>(lhs.getType()), rewriter);
      if (failed(mask)) {
        return failure();
      }
      allTrue = *mask;
    }
    return rewriter
        .create<PorOp>(op->getLoc(), lhs.getType(), lhs, rhs, allTrue)
        .getResult();
  }

  FailureOr<Value> packPair(Value lowSource, std::optional<Value> highSource,
                            MaskType maskType) {
    Value packed = rewriter.create<PpackOp>(op->getLoc(), maskType, lowSource,
                                            lower);
    if (!highSource) {
      return packed;
    }
    Value higherPacked = rewriter.create<PpackOp>(
        op->getLoc(), maskType, *highSource, higher);
    return merge(packed, higherPacked);
  }
};

static FailureOr<Value> materializeMaskLaneStridePackChunk(
    Operation *op, ValueRange sourceParts, size_t base, int64_t laneStride,
    MaskType maskType, MaskLaneStridePackContext &context,
    PatternRewriter &rewriter) {
  std::optional<Value> source1;
  if (base + 1 < sourceParts.size()) {
    source1 = sourceParts[base + 1];
  }
  FailureOr<Value> lowHalf =
      context.packPair(sourceParts[base], source1, maskType);
  if (failed(lowHalf)) {
    return failure();
  }
  Value current = *lowHalf;
  if (laneStride != 4) {
    return current;
  }
  current = rewriter.create<PpackOp>(op->getLoc(), maskType, current,
                                     context.lower);
  if (base + 2 >= sourceParts.size()) {
    return current;
  }
  std::optional<Value> source3;
  if (base + 3 < sourceParts.size()) {
    source3 = sourceParts[base + 3];
  }
  FailureOr<Value> highHalf =
      context.packPair(sourceParts[base + 2], source3, maskType);
  if (failed(highHalf)) {
    return failure();
  }
  Value higherPacked =
      rewriter.create<PpackOp>(op->getLoc(), maskType, *highHalf,
                                context.higher);
  return context.merge(current, higherPacked);
}

static FailureOr<SmallVector<Value>> materializeMaskLaneStridePack(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t laneStride, PatternRewriter &rewriter) {
  if (sourceParts.empty()) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride pack materialization requires source parts");
  }
  bool sourceExceedsResult =
      static_cast<int64_t>(sourceParts.size()) >
      static_cast<int64_t>(resultTypes.size()) * laneStride;
  if (sourceExceedsResult) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride pack materialization source arity does "
            "not fit result arity");
  }
  MaskLaneStridePackContext context{
      op, rewriter, rewriter.getStringAttr("LOWER"),
      rewriter.getStringAttr("HIGHER"), Value()};

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<MaskType> maskType = getMaskLaneStrideResultType(
        op, resultType, "dense mask lane_stride pack requires mask result type",
        rewriter);
    if (failed(maskType)) {
      return failure();
    }
    size_t base = resultIndex * static_cast<size_t>(laneStride);
    if (base >= sourceParts.size()) {
      break;
    }
    FailureOr<Value> current = materializeMaskLaneStridePackChunk(
        op, sourceParts, base, laneStride, *maskType, context, rewriter);
    if (failed(current)) {
      return failure();
    }
    results.push_back(*current);
  }
  bool resultArityMismatch = results.size() != resultTypes.size();
  if (resultArityMismatch) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride pack materialization result arity mismatch");
  }
  return results;
}

struct MaskLaneStrideLayoutPlan {
  bool unpack;
  int64_t laneStride;
};

static std::optional<MaskLaneStrideLayoutPlan> getMaskLaneStrideLayoutPlan(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) {
  bool unpack = sourceLayout && sourceLayout.isContiguous() &&
                sourceLayout.getLaneStride() == 1 && resultLayout &&
                resultLayout.isContiguous() &&
                resultLayout.getLaneStride() != 1;
  bool pack = sourceLayout && sourceLayout.isContiguous() &&
              sourceLayout.getLaneStride() != 1 && resultLayout &&
              resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (!unpack && !pack) {
    return std::nullopt;
  }
  return MaskLaneStrideLayoutPlan{
      unpack, unpack ? resultLayout.getLaneStride() : sourceLayout.getLaneStride()};
}

static LogicalResult checkMaskLaneStrideFactor(
    Operation *op, const MaskLaneStrideLayoutPlan &plan,
    PatternRewriter &rewriter) {
  bool supportedFactor = plan.laneStride == 2 || plan.laneStride == 4;
  if (supportedFactor) {
    return success();
  }
  return rewriter.notifyMatchFailure(
      op, plan.unpack ? "unsupported dense mask lane_stride unpack factor"
                      : "unsupported dense mask lane_stride pack factor");
}

FailureOr<std::optional<SmallVector<Value>>> materializeMaskLaneStrideLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  std::optional<MaskLaneStrideLayoutPlan> plan =
      getMaskLaneStrideLayoutPlan(sourceLayout, resultLayout);
  if (!plan) {
    return std::optional<SmallVector<Value>>{};
  }

  if (failed(checkMaskLaneStrideFactor(op, *plan, rewriter))) {
    return failure();
  }

  if (plan->unpack) {
    FailureOr<SmallVector<Value>> results = materializeMaskLaneStrideUnpack(
        op, sourceParts, resultTypes, plan->laneStride, rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }
  FailureOr<SmallVector<Value>> results = materializeMaskLaneStridePack(
      op, sourceParts, resultTypes, plan->laneStride, rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static FailureOr<std::optional<SmallVector<Value>>> materializeIdentityMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  if (sourceLayout != resultLayout) {
    return std::optional<SmallVector<Value>>{};
  }
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(
      SmallVector<Value>(sourceParts.begin(), sourceParts.end()));
}


struct MaskLayoutMaterializationContext {
  Operation *op;
  ValueRange sourceParts;
  TypeRange resultTypes;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  PatternRewriter &rewriter;
};

using MaskLayoutMaterializationResult =
    FailureOr<std::optional<SmallVector<Value>>>;

static bool didHandleMaskLayoutMaterialization(
    const MaskLayoutMaterializationResult &result) {
  return failed(result) || result->has_value();
}

FailureOr<SmallVector<Value>> materializeStagingDeintToContiguousMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter);
FailureOr<SmallVector<Value>> materializeStagingContiguousToDeintMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter);

static bool isElementDeinterleavedMaskLayout(VMILayoutAttr layout,
                                             int64_t factor) {
  return layout && layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

static MaskLayoutMaterializationResult materializeDeinterleaved4MaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  bool contiguousToDeint4 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      isElementDeinterleavedMaskLayout(resultLayout, 4);
  bool deint4ToContiguous =
      isElementDeinterleavedMaskLayout(sourceLayout, 4) &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (!contiguousToDeint4 && !deint4ToContiguous) {
    return std::optional<SmallVector<Value>>{};
  }
  bool invalidArity = sourceParts.empty() ||
                      sourceParts.size() != resultTypes.size() ||
                      resultTypes.size() % 4 != 0;
  if (invalidArity) {
    (void)rewriter.notifyMatchFailure(
        op, "deinterleaved=4 mask layout materialization requires 4*N parts");
    return failure();
  }
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  FailureOr<SmallVector<Value>> results =
      contiguousToDeint4
          ? materializeStagingContiguousToDeintMaskLayout(
                op, sourceParts, resultTypes, 4, rewriter)
          : materializeStagingDeintToContiguousMaskLayout(
                op, sourceParts, resultTypes, 4, rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static MaskLayoutMaterializationResult
tryMaskLayoutMaterializers(const MaskLayoutMaterializationContext &context) {
  MaskLayoutMaterializationResult identity = materializeIdentityMaskLayout(
      context.op, context.sourceParts, context.resultTypes,
      context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleMaskLayoutMaterialization(identity)) {
    return identity;
  }

  MaskLayoutMaterializationResult deinterleaved2 =
      materializeDeinterleaved2MaskLayout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleMaskLayoutMaterialization(deinterleaved2)) {
    return deinterleaved2;
  }

  MaskLayoutMaterializationResult deinterleaved4 =
      materializeDeinterleaved4MaskLayout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleMaskLayoutMaterialization(deinterleaved4)) {
    return deinterleaved4;
  }

  return materializeMaskLaneStrideLayout(
      context.op, context.sourceParts, context.resultTypes, context.sourceLayout,
      context.resultLayout, context.rewriter);
}

FailureOr<SmallVector<Value>> materializeMaskLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "mask layout materialization requires assigned source/result "
            "layouts");
    return failure();
  }

  MaskLayoutMaterializationContext context{
      op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter};
  FailureOr<std::optional<SmallVector<Value>>> results =
      tryMaskLayoutMaterializers(context);
  if (failed(results)) {
    return failure();
  }
  if (results->has_value()) {
    return std::move(**results);
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI mask layout materialization");
  return failure();
}

int getMaskGranularityRank(StringRef granularity) {
  if (granularity == "b8") {
    return 0;
  }
  if (granularity == "b16") {
    return 1;
  }
  if (granularity == "b32") {
    return 2;
  }
  return -1;
}

StringRef getMaskGranularityForRank(int rank) {
  switch (rank) {
  case 0:
    return "b8";
  case 1:
    return "b16";
  case 2:
    return "b32";
  default:
    return "";
  }
}

LogicalResult checkSupportedMaskGranularityMaterialization(
    VMIMaskType sourceType,
    VMIMaskType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  bool laneCountMismatch =
      sourceType.getElementCount() != resultType.getElementCount();
  if (laneCountMismatch) {
    return fail("requires source and result mask lane counts to match");
  }
  bool layoutMismatch = sourceType.getLayoutAttr() != resultType.getLayoutAttr();
  if (layoutMismatch) {
    return fail("requires source and result mask layouts to match");
  }

  bool nonConcreteGranularity =
      !VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) ||
      !VMIMaskType::isConcreteGranularity(resultType.getGranularity());
  if (nonConcreteGranularity) {
    return fail("requires concrete b8/b16/b32 source and result "
                "granularities");
  }

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool missingArity = failed(sourceArity) || failed(resultArity);
  if (missingArity) {
    return fail("requires computable source/result physical arity");
  }
  if (*sourceArity < 1 || *resultArity < 1) {
    return fail("requires non-empty source/result physical arity");
  }

  return success();
}

FailureOr<SmallVector<Value>> materializeWideningMaskGranularityPart(
    Operation *op, MaskType resultMaskType, ValueRange sourceParts,
    int64_t sourceOffset, int64_t sourceChunks, int64_t resultChunks,
    PatternRewriter &rewriter) {
  SmallVector<Value> results;
  auto partAttr = StringAttr::get(op->getContext(), "LOWER");
  auto higherAttr = StringAttr::get(op->getContext(), "HIGHER");
  int64_t produced = 0;
  for (int64_t chunk = 0; chunk < sourceChunks && produced < resultChunks;
       ++chunk) {
    Value source = sourceParts[sourceOffset + chunk];
    results.push_back(rewriter
                          .create<PunpackOp>(op->getLoc(), resultMaskType,
                                             source, partAttr)
                          .getResult());
    ++produced;
    if (produced < resultChunks) {
      results.push_back(rewriter
                            .create<PunpackOp>(op->getLoc(), resultMaskType,
                                               source, higherAttr)
                            .getResult());
      ++produced;
    }
  }
  if (produced != resultChunks) {
    (void)rewriter.notifyMatchFailure(
        op, "widening mask granularity conversion produced the wrong number "
            "of result chunks");
    return failure();
  }
  return results;
}

static FailureOr<Value> materializeNarrowingMaskChunk(
    Operation *op, MaskType resultMaskType, ValueRange sourceParts,
    int64_t sourceOffset, int64_t sourceChunks, int64_t &consumed,
    Value &allTrue, StringAttr lowerAttr, StringAttr higherAttr,
    PatternRewriter &rewriter) {
  if (consumed >= sourceChunks) {
    (void)rewriter.notifyMatchFailure(
        op, "narrowing mask granularity conversion ran out of source chunks");
    return failure();
  }
  Value lowerSource = sourceParts[sourceOffset + consumed++];
  Value packed = rewriter
                     .create<PpackOp>(op->getLoc(), resultMaskType, lowerSource,
                                      lowerAttr)
                     .getResult();
  if (consumed >= sourceChunks) {
    return packed;
  }
  Value higherSource = sourceParts[sourceOffset + consumed++];
  Value higher = rewriter
                     .create<PpackOp>(op->getLoc(), resultMaskType, higherSource,
                                      higherAttr)
                     .getResult();
  if (!allTrue) {
    FailureOr<Value> mask =
        createAllTrueMask(op->getLoc(), resultMaskType, rewriter);
    if (failed(mask)) {
      (void)rewriter.notifyMatchFailure(
          op, "failed to create all-true mask for ppack merge");
      return failure();
    }
    allTrue = *mask;
  }
  return rewriter
      .create<PorOp>(op->getLoc(), resultMaskType, packed, higher, allTrue)
      .getResult();
}

FailureOr<SmallVector<Value>> materializeNarrowingMaskGranularityPart(
    Operation *op, MaskType resultMaskType, ValueRange sourceParts,
    int64_t sourceOffset, int64_t sourceChunks, int64_t resultChunks,
    PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  auto lowerAttr = StringAttr::get(op->getContext(), "LOWER");
  auto higherAttr = StringAttr::get(op->getContext(), "HIGHER");
  SmallVector<Value> results;
  Value allTrue;
  int64_t consumed = 0;
  for (int64_t chunk = 0; chunk < resultChunks; ++chunk) {
    FailureOr<Value> packed = materializeNarrowingMaskChunk(
        op, resultMaskType, sourceParts, sourceOffset, sourceChunks, consumed,
        allTrue, lowerAttr, higherAttr, rewriter);
    if (failed(packed)) {
      return failure();
    }
    results.push_back(*packed);
  }
  if (consumed != sourceChunks) {
    return fail("narrowing mask granularity conversion left unused source "
                "chunks");
  }
  return results;
}

static FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityPart(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, int sourceRank, int resultRank,
    MaskType resultMaskType, int64_t part, int64_t sourceOffset,
    PatternRewriter &rewriter) {
  FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
  FailureOr<int64_t> resultChunks = getVMITypeChunksInPart(resultType, part);
  bool invalidChunkCounts = failed(sourceChunks) || failed(resultChunks);
  if (invalidChunkCounts) {
    (void)rewriter.notifyMatchFailure(
        op, "requires computable source/result chunks per layout part");
    return failure();
  }
  bool widening = resultRank > sourceRank;
  if (widening) {
    return materializeWideningMaskGranularityPart(
        op, resultMaskType, sourceParts, sourceOffset, *sourceChunks,
        *resultChunks, rewriter);
  }
  return materializeNarrowingMaskGranularityPart(
      op, resultMaskType, sourceParts, sourceOffset, *sourceChunks,
      *resultChunks, rewriter);
}

struct MaskGranularityConversionPlan {
  int sourceRank;
  int resultRank;
  int64_t sourceArity;
  int64_t layoutFactor;
  MaskType resultMaskType;
};

static FailureOr<MaskGranularityConversionPlan>
buildMaskGranularityConversionPlan(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<MaskGranularityConversionPlan> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  bool nonAdjacent = std::abs(sourceRank - resultRank) != 1;
  if (nonAdjacent) {
    return fail("mask granularity conversion must be adjacent");
  }
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(sourceType);
  bool sourceArityMismatch =
      failed(sourceArity) || failed(factor) ||
      static_cast<int64_t>(sourceParts.size()) != *sourceArity;
  if (sourceArityMismatch) {
    return fail("source mask part count does not match source VMI type");
  }
  return MaskGranularityConversionPlan{
      sourceRank, resultRank, *sourceArity, *factor,
      MaskType::get(op->getContext(), resultType.getGranularity())};
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityParts(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, const MaskGranularityConversionPlan &plan,
    PatternRewriter &rewriter) {
  SmallVector<Value> results;
  int64_t sourceOffset = 0;
  for (int64_t part = 0; part < plan.layoutFactor; ++part) {
    FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
    if (failed(sourceChunks)) {
      (void)rewriter.notifyMatchFailure(
          op, "requires computable source chunks per layout part");
      return failure();
    }
    FailureOr<SmallVector<Value>> partResults =
        materializeAdjacentMaskGranularityPart(
            op, sourceType, resultType, sourceParts, plan.sourceRank,
            plan.resultRank, plan.resultMaskType, part, sourceOffset, rewriter);
    if (failed(partResults)) {
      return failure();
    }
    results.append(*partResults);
    sourceOffset += *sourceChunks;
  }
  return results;
}

static LogicalResult checkMaskGranularityResultArity(
    Operation *op, VMIMaskType resultType, size_t resultCount,
    PatternRewriter &rewriter) {
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool resultArityMismatch =
      failed(resultArity) || static_cast<int64_t>(resultCount) != *resultArity;
  if (resultArityMismatch) {
    (void)rewriter.notifyMatchFailure(
        op, "mask granularity conversion result count mismatch");
    return failure();
  }
  return success();
}

FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, PatternRewriter &rewriter) {
  FailureOr<MaskGranularityConversionPlan> plan =
      buildMaskGranularityConversionPlan(op, sourceType, resultType,
                                          sourceParts, rewriter);
  if (failed(plan)) {
    return failure();
  }
  FailureOr<SmallVector<Value>> results = materializeMaskGranularityParts(
      op, sourceType, resultType, sourceParts, *plan, rewriter);
  if (failed(results)) {
    return failure();
  }

  if (failed(checkMaskGranularityResultArity(op, resultType, results->size(),
                                             rewriter))) {
    return failure();
  }
  return *results;
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityStep(
    Operation *op, VMIMaskType currentType, StringRef nextGranularity,
    ValueRange currentParts, PatternRewriter &rewriter) {
  VMIMaskType nextType = VMIMaskType::get(
      op->getContext(), currentType.getElementCount(), nextGranularity,
      currentType.getLayoutAttr());
  return materializeAdjacentMaskGranularityConversion(
      op, currentType, nextType, currentParts, rewriter);
}

static FailureOr<VMIMaskType> buildNextMaskGranularityType(
    VMIMaskType currentType, int rank, PatternRewriter &rewriter,
    Operation *op) {
  StringRef granularity = getMaskGranularityForRank(rank);
  if (granularity.empty()) {
    (void)rewriter.notifyMatchFailure(
        op, "invalid target mask granularity rank");
    return failure();
  }
  return VMIMaskType::get(op->getContext(), currentType.getElementCount(),
                          granularity, currentType.getLayoutAttr());
}

static FailureOr<SmallVector<Value>> materializeMaskGranularitySteps(
    Operation *op, VMIMaskType sourceType, ValueRange sourceParts,
    int sourceRank, int resultRank, PatternRewriter &rewriter) {
  VMIMaskType currentType = sourceType;
  SmallVector<Value> currentParts(sourceParts.begin(), sourceParts.end());
  int currentRank = sourceRank;
  while (currentRank != resultRank) {
    bool ascending = currentRank < resultRank;
    currentRank += ascending ? 1 : -1;
    FailureOr<VMIMaskType> nextType = buildNextMaskGranularityType(
        currentType, currentRank, rewriter, op);
    if (failed(nextType)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> nextParts = materializeMaskGranularityStep(
        op, currentType, nextType->getGranularity(), currentParts, rewriter);
    if (failed(nextParts)) {
      return failure();
    }
    currentType = *nextType;
    currentParts = std::move(*nextParts);
  }
  return currentParts;
}

struct MaskGranularityRoute {
  int sourceRank;
  int resultRank;
  bool adjacent;
};

static FailureOr<MaskGranularityRoute> classifyMaskGranularityRoute(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    PatternRewriter &rewriter) {
  auto fail = [&rewriter, op](const Twine &message)
      -> FailureOr<MaskGranularityRoute> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  if (sourceRank < 0 || resultRank < 0) {
    return fail("requires concrete source and result mask granularity ranks");
  }
  return MaskGranularityRoute{sourceRank, resultRank,
                              std::abs(sourceRank - resultRank) == 1};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts,
    PatternRewriter &rewriter) {
  std::string reason;
  if (failed(checkSupportedMaskGranularityMaterialization(sourceType, resultType, &reason))) {
    (void)rewriter.notifyMatchFailure(op, reason);
    return failure();
  }

  FailureOr<MaskGranularityRoute> route = classifyMaskGranularityRoute(
      op, sourceType, resultType, rewriter);
  if (failed(route)) {
    return failure();
  }
  if (route->adjacent) {
    return materializeAdjacentMaskGranularityConversion(
        op, sourceType, resultType, sourceParts, rewriter);
  }

  return materializeMaskGranularitySteps(op, sourceType, sourceParts,
                                         route->sourceRank, route->resultRank,
                                         rewriter);
}

static SmallVector<Type> repeatMaskPartType(Type partType, int64_t arity) {
  SmallVector<Type> types;
  types.reserve(arity);
  for (int64_t i = 0; i < arity; ++i) {
    types.push_back(partType);
  }
  return types;
}

FailureOr<SmallVector<Type>> getConvertedMaskPartTypes(VMIMaskType type) {
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(type);
  bool invalidMaskTypes =
      failed(arity) || failed(physicalGranularity) || *arity < 0;
  if (invalidMaskTypes) {
    return failure();
  }
  Type partType = MaskType::get(type.getContext(), *physicalGranularity);
  return repeatMaskPartType(partType, *arity);
}

static FailureOr<VMILayoutAttr>
getVMIMaskPhysicalCarrierLayout(VMIMaskType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout) {
    return failure();
  }
  MLIRContext *ctx = type.getContext();
  if (layout.isContiguous()) {
    return VMILayoutAttr::getContiguous(ctx);
  }
  if (layout.isDeinterleaved()) {
    return VMILayoutAttr::getDeinterleaved(ctx, layout.getFactor());
  }
  if (layout.isBlockDeinterleaved()) {
    return VMILayoutAttr::getBlockDeinterleaved(ctx, layout.getFactor());
  }
  if (layout.isGroupSlots()) {
    return VMILayoutAttr::getGroupSlots(ctx, layout.getNumGroups(),
                                        layout.getSlots());
  }
  return failure();
}

static FailureOr<VMIMaskType>
getVMIMaskPhysicalCarrierType(VMIMaskType type) {
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(type);
  FailureOr<VMILayoutAttr> physicalLayout =
      getVMIMaskPhysicalCarrierLayout(type);
  bool missingPhysicalCarrier = failed(physicalGranularity) ||
                                failed(physicalLayout);
  if (missingPhysicalCarrier) {
    return failure();
  }
  return VMIMaskType::get(type.getContext(), type.getElementCount(),
                          *physicalGranularity, *physicalLayout);
}

static bool isElementDeinterleavedLayout(VMILayoutAttr layout,
                                         int64_t factor) {
  return layout && layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

FailureOr<Value> createAllFalseMaskLike(Location loc, Value value,
                                        PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(value.getType());
  if (!maskType) {
    return failure();
  }
  return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);
}

FailureOr<std::array<Value, 4>> materializeFactor4DeintToContiguousGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    size_t resultOffset, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 4>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidSources = sources.size() != 4;
  bool missingResultTypes = resultTypes.empty();
  if (invalidSources || missingResultTypes) {
    return fail("factor-4 staging mask conversion requires four sources");
  }
  auto resultTypeAt = [resultTypes, resultOffset](size_t offset) -> Type {
    size_t index = resultOffset + offset;
    return index < resultTypes.size() ? resultTypes[index]
                                      : resultTypes[resultTypes.size() - 1];
  };
  FailureOr<std::pair<Value, Value>> even = createPredicateIntlv(
      op->getLoc(), resultTypeAt(0), resultTypeAt(1), sources[0], sources[2],
      rewriter);
  FailureOr<std::pair<Value, Value>> odd = createPredicateIntlv(
      op->getLoc(), resultTypeAt(0), resultTypeAt(1), sources[1], sources[3],
      rewriter);
  if (failed(even)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  if (failed(odd)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  FailureOr<std::pair<Value, Value>> low = createPredicateIntlv(
      op->getLoc(), resultTypeAt(0), resultTypeAt(1), even->first, odd->first,
      rewriter);
  FailureOr<std::pair<Value, Value>> high = createPredicateIntlv(
      op->getLoc(), resultTypeAt(2), resultTypeAt(3), even->second,
      odd->second, rewriter);
  if (failed(low)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  if (failed(high)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  return std::array<Value, 4>{low->first, low->second, high->first,
                              high->second};
}

static FailureOr<std::array<Value, 2>> materializeFactor2DeintToContiguousGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    size_t resultOffset, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 2>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidSources = sources.size() != 2 || resultTypes.empty();
  if (invalidSources) {
    return fail("factor-2 staging mask conversion requires two sources");
  }
  size_t first = std::min(resultOffset, resultTypes.size() - 1);
  size_t second = std::min(resultOffset + 1, resultTypes.size() - 1);
  FailureOr<std::pair<Value, Value>> materialized = createPredicateIntlv(
      op->getLoc(), resultTypes[first], resultTypes[second], sources[0],
      sources[1], rewriter);
  if (failed(materialized)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  return std::array<Value, 2>{materialized->first, materialized->second};
}


