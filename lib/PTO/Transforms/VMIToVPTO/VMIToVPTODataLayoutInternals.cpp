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

constexpr int kVMIDataLayoutFactor2 = 2;
constexpr int kVMIDataLayoutFourthPartIndex = 3;
constexpr int kVMIDataLayoutFactor4 = 4;
constexpr int kVMIDataLayoutPairSize = 2;
constexpr int kVMIDataLayoutMaskGranularityRankB32 = 2;

// The grouped staging mask materializers are defined in
// VMIToVPTOPatternInternals0.cpp; declare them here because the deinterleaved
// mask layout paths in VMIToVPTOMaskLayoutInternals.cpp route unequal part counts into them.
FailureOr<SmallVector<Value>> materializeStagingDeintToContiguousMaskLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, int64_t factor, PatternRewriter& rewriter);
FailureOr<SmallVector<Value>> materializeStagingContiguousToDeintMaskLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, int64_t factor, PatternRewriter& rewriter);

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
  // The same carrier chain as the dense lane-stride materialization: both the
  // element carrier and the widest carrier (element bits times the stride) have
  // to be expressible, which now reaches 64 bit.
  bool unsupportedCarrier =
      failed(getUnsignedCarrierVRegType(rewriter.getContext(), elementBits)) ||
      failed(getUnsignedCarrierVRegType(
          rewriter.getContext(),
          elementBits * static_cast<unsigned>(maxStride)));
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

  // A compact group packet and a dense value share the same carrier lanes.
  FailureOr<int64_t> carrierLanes = getDataLanesPerPart(sourceVMIElementType);
  const bool singleCarrierAlias =
      succeeded(carrierLanes) &&
      isVMISingleCarrierGroupSlotAlias(sourceLayout, resultLayout,
                                       *carrierLanes);
  if (singleCarrierAlias) {
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
  int64_t groups = sourceParts.size() / kVMIDataLayoutFactor2;
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

/// Packs the two deinterleaved=2 parts of a compacted gather back into the
/// contiguous carriers: interleaving them once more yields lanes 0..63 and
/// 64..127.  Split out of materializeDeint4ToDeint2 to keep that materializer
/// within the coding-standard function size.
static FailureOr<std::optional<SmallVector<Value>>>
mergeCompactedPartsIntoContiguous(Operation *op,
                                  const SmallVector<Value> &parts,
                                  TypeRange resultTypes,
                                  PatternRewriter &rewriter) {
  Type partType = parts[0].getType();
  const bool hasMismatchedTypes = parts.size() != mlir::pto::kValue2 || partType != parts[1].getType() ||
                                  partType != resultTypes[0] || partType != resultTypes[1];
  if (hasMismatchedTypes) {
      return rewriter.notifyMatchFailure(op, "vintlv requires operands and results to share one type");
  }
  auto merged = rewriter.create<VintlvOp>(op->getLoc(), partType, partType,
                                          parts[0], parts[1]);
  return std::optional<SmallVector<Value>>(
      SmallVector<Value>{merged.getLow(), merged.getHigh()});
}

/// A deinterleaved=4 value whose deinterleaved=2 form spans half as many
/// carriers (four parts -> two, for example 128 f32) is a carrier-compacting
/// gather.  Part p holds the logical lanes congruent to p modulo 4 in its low
/// half, so the two classes that share a parity interleave back into exactly
/// one deinterleaved=2 part, as follows.
///   d0 = vintlv(p0, p2).low   (lanes 0, 2, 4, ...)
///   d1 = vintlv(p1, p3).low   (lanes 1, 3, 5, ...)
/// The high halves of the sources are padding outside the value, so the high
/// halves of the interleave results are unused.
static FailureOr<std::optional<SmallVector<Value>>> materializeDeint4ToDeint2(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  auto isClassLayout = [](VMILayoutAttr layout, int64_t factor) {
    return layout && layout.isDeinterleaved() &&
           layout.getFactor() == factor && layout.getLaneStride() == 1;
  };
  bool resultIsClass2 = isClassLayout(resultLayout, 2);
  bool resultIsContiguous = resultLayout && resultLayout.isContiguous() &&
                            resultLayout.getLaneStride() == 1;
  bool compacting = isClassLayout(sourceLayout, 4) &&
                    (resultIsClass2 || resultIsContiguous);
  if (!compacting) {
    return std::optional<SmallVector<Value>>{};
  }
  // Only the carrier-compacting 4*N -> 2*N relation belongs to this
  // materializer.  An arity-preserving deinterleaved=4 <-> contiguous pair
  // (4*N -> 4*N, e.g. 256 f32) must stay unclaimed so that the generic
  // deinterleaved=4 path still gets its turn; reporting failure here would
  // abort the materializer chain and leave the ensure_layout unconverted.
  // Exactly one four-part group: the parts are consumed in partition order
  // (see materializeDeint4DataToContiguous), so a wider 4N -> 2N pair would
  // need a group-major index this materializer does not implement.
  bool compactingArity = sourceParts.size() == 4 &&
                         resultTypes.size() == sourceParts.size() / 2;
  if (!compactingArity) {
    return std::optional<SmallVector<Value>>{};
  }
  int64_t groups = sourceParts.size() / 4;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t group = 0; group < groups; ++group) {
    Value classes[4] = {sourceParts[4 * group], sourceParts[4 * group + 1],
                        sourceParts[4 * group + 2],
                        sourceParts[4 * group + 3]};
    for (int64_t parity = 0; parity < mlir::pto::kValue2; ++parity) {
        Value lhs = classes[parity];
        Value rhs = classes[parity + 2];
        Type lhsType = lhs.getType();
        if (lhsType != rhs.getType() || lhsType != resultTypes[2 * group + parity]) {
            return rewriter.notifyMatchFailure(op, "vintlv requires operands and results to share one type");
        }
        auto interleave = rewriter.create<VintlvOp>(op->getLoc(), lhsType, lhsType, lhs, rhs);
        results.push_back(interleave.getLow());
    }
  }
  if (resultIsContiguous) {
    return mergeCompactedPartsIntoContiguous(op, results, resultTypes,
                                             rewriter);
  }
  return std::optional<SmallVector<Value>>(std::move(results));
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
  groups = resultTypes.size() / kVMIDataLayoutFactor2;
  bool sourceExceedsResult =
      sourceParts.size() >
      static_cast<size_t>(kVMIDataLayoutFactor2 * groups);
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
  /// Lane stride of the Contiguous intermediate.  A dense/group-packet pair
  /// whose two strides differ normalizes towards the packet's stride, so that
  /// the first step is the carrier identity (or the dense lane-stride change)
  /// and the second step finishes the other axis.
  int64_t laneStride = 1;
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
         (layout.getFactor() == kVMIDataLayoutFactor2 ||
          layout.getFactor() == kVMIDataLayoutFactor4);
}

/// Physical arity of a dense lane-strided value once it is normalized to the
/// unit lane stride.
/// The dense lane-stride materialization keeps one physical part per stride
/// group, so normalizing a value of `sourcePartCount` parts at
/// `sourceLaneStride` leaves the ceiling of the division: the last stride group
/// may be partial, which is also how the materializer counts it.
static std::optional<size_t> getUnitStridePartCount(size_t sourcePartCount,
                                                    int64_t sourceLaneStride) {
  if (sourcePartCount == 0 || sourceLaneStride <= 0) {
    return std::nullopt;
  }
  return (sourcePartCount + static_cast<size_t>(sourceLaneStride) - 1) /
         static_cast<size_t>(sourceLaneStride);
}

static std::optional<DataLayoutIntermediatePlan>
getDataLayoutIntermediatePlan(VMILayoutAttr sourceLayout,
                              VMILayoutAttr resultLayout,
                              size_t sourcePartCount, Type sourceVMIElementType,
                              int64_t lanesPerPart) {
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
    // The intermediate is the unit-stride form of the same value: a
    // deinterleaved source already carries that arity, a lane-strided one
    // shrinks by its lane stride.
    std::optional<size_t> intermediateCount = getUnitStridePartCount(
        sourcePartCount, sourceLayout.getLaneStride());
    if (!intermediateCount) {
      return std::nullopt;
    }
    return DataLayoutIntermediatePlan{
        DataLayoutIntermediatePlan::Kind::Contiguous, *intermediateCount};
  }

  // A dense value and a single-carrier group packet whose lane strides differ
  // live in the same carrier but not in the same lanes: normalize towards the
  // packet's lane stride first, then forward the carrier unchanged (the
  // single-carrier identity).  The mirror direction is the same plan reversed.
  if (needsVMIDenseLaneStrideGroupSlotBridge(sourceLayout, resultLayout,
                                             sourceVMIElementType,
                                             lanesPerPart)) {
    bool denseToPacket = sourceLayout.isContiguous();
    // Both sides of the bridge stay inside one physical part -- the
    // ensure_layout rows bound the element count -- so the intermediate keeps
    // the source arity: the dense side is normalized onto the packet's lane
    // stride and the single-carrier identity forwards that intermediate
    // unchanged.  In the mirror direction the packet is the source, the first
    // step is that identity and the second one finishes the dense side.
    size_t intermediateCount = sourcePartCount;
    int64_t packetStride = denseToPacket ? resultLayout.getLaneStride()
                                         : sourceLayout.getLaneStride();
    return DataLayoutIntermediatePlan{
        DataLayoutIntermediatePlan::Kind::Contiguous, intermediateCount,
        packetStride};
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
      VMILayoutAttr::getContiguous(rewriter.getContext(), plan.laneStride);
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
  FailureOr<int64_t> carrierLanes = getDataLanesPerPart(sourceVMIElementType);
  std::optional<DataLayoutIntermediatePlan> plan =
      getDataLayoutIntermediatePlan(sourceLayout, resultLayout,
                                    sourceParts.size(), sourceVMIElementType,
                                    succeeded(carrierLanes) ? *carrierLanes
                                                            : 0);
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
    for (size_t offset = 0; offset < kVMIDataLayoutFactor4; ++offset) {
      if (resultTypes[kVMIDataLayoutFactor4 * group + offset] != chunkType) {
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

static FailureOr<std::array<Value, kVMIDataLayoutFactor4>>
materializeContiguousToDeint4DataGroup(
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
  for (size_t offset = 0; offset < kVMIDataLayoutFactor4; ++offset) {
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
  return std::array<Value, kVMIDataLayoutFactor4>{
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
    FailureOr<std::array<Value, kVMIDataLayoutFactor4>> groupValues =
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
  // A short dense value needs four phase carriers, but only its original
  // carriers contain data. Pad the dense register list before splitting;
  // the padding must never be supplied by additional memory reads.
  if (contiguousToDeint4 && sourceParts.size() == mlir::pto::kValue2 && resultTypes.size() == kVMIDataLayoutFactor4) {
      auto chunkType = dyn_cast<VRegType>(sourceParts.front().getType());
      if (!chunkType) {
          return failure();
      }
      FailureOr<Value> zero = createZeroVector(op->getLoc(), chunkType, rewriter);
      if (failed(zero)) {
          return failure();
      }
      SmallVector<Value> padded(sourceParts.begin(), sourceParts.end());
      padded.append({*zero, *zero});
      auto results = materializeContiguousToDeint4Data(op, padded, resultTypes, rewriter);
      if (failed(results)) {
          return failure();
      }
      return std::optional<SmallVector<Value>>(std::move(*results));
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

// A deinterleaved=2 value stores two chunks per phase, laid out as
// [phase-0 chunks][phase-1 chunks]; a deinterleaved=4 value stores one chunk
// per phase, laid out as [part0 chunks][part1 chunks][part2 chunks][part3
// chunks].  The through-contiguous route spends two vintlv and four vdintlv per
// group.  Composing the two reconstructions shows that pairing the two phase-0
// chunks and the two phase-1 chunks and splitting each pair with one vdintlv
// yields exactly the four d4 phases, because the nested route first recovers
// the per-phase carriers and only then separates their even/odd positions.
// The cost model already charges d2 -> d4 two instructions, so this direct
// materialization keeps lowering consistent with that charge.
static FailureOr<SmallVector<Value>> materializeDeint2ToDeint4Data(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, PatternRewriter& rewriter)
{
    int64_t groups = resultTypes.size() / kVMIDataLayoutFactor4;
    SmallVector<Value> part0;
    SmallVector<Value> part1;
    SmallVector<Value> part2;
    SmallVector<Value> part3;
    part0.reserve(groups);
    part1.reserve(groups);
    part2.reserve(groups);
    part3.reserve(groups);
    for (int64_t group = 0; group < groups; ++group) {
        Value phase0Low = sourceParts[kVMIDataLayoutFactor2 * group];
        Value phase0High = sourceParts[kVMIDataLayoutFactor2 * group + 1];
        Value phase1Low = sourceParts[2 * groups + kVMIDataLayoutFactor2 * group];
        Value phase1High = sourceParts[2 * groups + kVMIDataLayoutFactor2 * group + 1];
        Type chunkType = phase0Low.getType();
        bool mismatchedOperands =
            phase0High.getType() != chunkType || phase1Low.getType() != chunkType || phase1High.getType() != chunkType;
        bool mismatchedResults = resultTypes[group] != chunkType || resultTypes[groups + group] != chunkType ||
                                 resultTypes[2 * groups + group] != chunkType ||
                                 resultTypes[3 * groups + group] != chunkType;
        if (mismatchedOperands || mismatchedResults) {
            (void)rewriter.notifyMatchFailure(op, "vdintlv requires operands and results to share one type");
            return failure();
        }
        auto even = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType, phase0Low, phase0High);
        auto odd = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType, phase1Low, phase1High);
        part0.push_back(even.getLow());
        part1.push_back(odd.getLow());
        part2.push_back(even.getHigh());
        part3.push_back(odd.getHigh());
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    results.append(part0);
    results.append(part1);
    results.append(part2);
    results.append(part3);
    return results;
}

static DataLayoutMaterializationResult materializeDeinterleaved2To4DataLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    bool deint2ToDeint4 =
        isElementDeinterleavedDataLayout(sourceLayout, 2) && isElementDeinterleavedDataLayout(resultLayout, 4);
    if (!deint2ToDeint4) {
        return std::optional<SmallVector<Value>>{};
    }
    // Only the equal-arity, factor-4-aligned shape is materialized here; any
    // other shape declines so the established through-contiguous route owns it.
    bool unsupportedArity = sourceParts.empty() || sourceParts.size() != resultTypes.size() ||
                            resultTypes.size() % kVMIDataLayoutFactor4 != 0;
    if (unsupportedArity) {
        return std::optional<SmallVector<Value>>{};
    }
    FailureOr<SmallVector<Value>> results = materializeDeint2ToDeint4Data(op, sourceParts, resultTypes, rewriter);
    if (failed(results)) {
        return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
}

// The inverse of the deinterleaved=2 -> deinterleaved=4 reconstruction above.
// A deinterleaved=4 value stores one chunk per phase, laid out as [part0
// chunks][part1 chunks][part2 chunks][part3 chunks], while a deinterleaved=2
// value stores two chunks per phase, laid out as [phase-0 chunks][phase-1
// chunks].  Pairing the phase-0 chunks (parts 0 and 2) and the phase-1 chunks
// (parts 1 and 3) and interleaving each pair yields exactly the two phase
// carriers, one vintlv per pair.  The through-contiguous route instead spends
// four vintlv plus two vdintlv per group, so the direct materialization both
// matches the inverse operation and keeps lowering consistent with the cost
// model, which already charges d4 -> d2 two instructions.
static FailureOr<SmallVector<Value>> materializeDeint4ToDeint2Data(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, PatternRewriter& rewriter)
{
    int64_t groups = sourceParts.size() / kVMIDataLayoutFactor4;
    SmallVector<Value> phase0;
    SmallVector<Value> phase1;
    phase0.reserve(kVMIDataLayoutFactor2 * groups);
    phase1.reserve(kVMIDataLayoutFactor2 * groups);
    for (int64_t group = 0; group < groups; ++group) {
        Value part0 = sourceParts[group];
        Value part1 = sourceParts[groups + group];
        Value part2 = sourceParts[2 * groups + group];
        Value part3 = sourceParts[3 * groups + group];
        Type chunkType = part0.getType();
        bool mismatchedOperands =
            part1.getType() != chunkType || part2.getType() != chunkType || part3.getType() != chunkType;
        bool mismatchedResults = resultTypes[kVMIDataLayoutFactor2 * group] != chunkType ||
                                 resultTypes[kVMIDataLayoutFactor2 * group + 1] != chunkType ||
                                 resultTypes[2 * groups + kVMIDataLayoutFactor2 * group] != chunkType ||
                                 resultTypes[2 * groups + kVMIDataLayoutFactor2 * group + 1] != chunkType;
        if (mismatchedOperands || mismatchedResults) {
            (void)rewriter.notifyMatchFailure(op, "vintlv requires operands and results to share one type");
            return failure();
        }
        auto even = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType, part0, part2);
        auto odd = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType, part1, part3);
        phase0.push_back(even.getLow());
        phase0.push_back(even.getHigh());
        phase1.push_back(odd.getLow());
        phase1.push_back(odd.getHigh());
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    results.append(phase0);
    results.append(phase1);
    return results;
}

static DataLayoutMaterializationResult materializeDeinterleaved4To2DataLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    bool deint4ToDeint2 =
        isElementDeinterleavedDataLayout(sourceLayout, 4) && isElementDeinterleavedDataLayout(resultLayout, 2);
    if (!deint4ToDeint2) {
        return std::optional<SmallVector<Value>>{};
    }
    // Only the equal-arity, factor-4-aligned shape is materialized here; any
    // other shape declines so the established through-contiguous route owns it.
    bool unsupportedArity = sourceParts.empty() || sourceParts.size() != resultTypes.size() ||
                            resultTypes.size() % kVMIDataLayoutFactor4 != 0;
    if (unsupportedArity) {
        return std::optional<SmallVector<Value>>{};
    }
    FailureOr<SmallVector<Value>> results = materializeDeint4ToDeint2Data(op, sourceParts, resultTypes, rewriter);
    if (failed(results)) {
        return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
}

static DataLayoutMaterializationResult tryDataLayoutMaterializers(const DataLayoutMaterializationContext& context)
{
    auto simple = materializeSimpleDataLayoutConversion(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.sourceVMIElementType, context.rewriter);
    if (didHandleDataLayoutMaterialization(simple)) {
        return simple;
    }
    using Materializer = DataLayoutMaterializationResult (*)(
        Operation*, ValueRange, TypeRange, VMILayoutAttr, VMILayoutAttr, PatternRewriter&);
    const Materializer rearrangements[] = {
        materializeDeinterleaved2Layout, materializeDeint4ToDeint2, materializeDeinterleaved4DataLayout,
        materializeDeinterleaved2To4DataLayout, materializeDeinterleaved4To2DataLayout};
    for (Materializer materialize : rearrangements) {
        auto result = materialize(
            context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
            context.rewriter);
        if (didHandleDataLayoutMaterialization(result)) {
            return result;
        }
    }
    auto laneStride = materializeDataLaneStrideConversion(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.sourceVMIElementType, context.rewriter);
    if (didHandleDataLayoutMaterialization(laneStride)) {
        return laneStride;
    }
    return materializeDataLayoutViaContiguous(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.sourceVMIElementType, context.rewriter);
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
  FailureOr<VMIEnsureLayoutFact> ensureFact = supports.getEnsureLayoutFact(sourceType, resultType, &supportReason);
  if (failed(ensureFact)) {
      (void)rewriter.notifyMatchFailure(
          op, Twine("ensure_layout has no registered materialization support: ") + supportReason);
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
  // The support fact already reports when both sides select the same physical
  // parts, which is the case the cost model charges as a forward.  Forward the
  // parts instead of rebuilding them through the lane-rearranging path.
  if (ensureFact->forwardsPhysicalParts) {
      if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes, rewriter))) {
          return failure();
      }
      return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }
  return materializeDataLayoutConversion(op, sourceParts, resultTypes,
                                         sourceLayout, resultLayout,
                                         sourceType.getElementType(), rewriter);
}
