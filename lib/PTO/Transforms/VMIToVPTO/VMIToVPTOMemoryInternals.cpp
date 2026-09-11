// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOMemoryInternals.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

static FailureOr<VMIStatefulReadContract> getStatefulReadContract(
    Value source, Value offset, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<VMIStatefulReadContract> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> staticElements =
      getStaticMemRefElementCount(source.getType());
  if (failed(staticElements)) {
    return fail("requires statically shaped memref source");
  }
  std::string rangeReason;
  FailureOr<VMIStatefulOffsetRange> offsetRange =
      getStatefulOffsetRange(source, offset, &rangeReason);
  if (failed(offsetRange)) {
    return fail(rangeReason);
  }
  std::string elementSizeReason;
  FailureOr<int64_t> elementBytes = getByteAddressableElementSize(
      resultType.getElementType(), &elementSizeReason);
  if (failed(elementBytes)) {
    return fail(elementSizeReason);
  }
  constexpr int64_t blockBytes = 32;
  std::optional<int64_t> remainder = getKnownAddressRemainderBytes(
      source, offset, resultType.getElementType(), blockBytes);
  if (!remainder) {
    return fail("requires a proven fixed 32-byte address remainder");
  }
  std::string footprintReason;
  FailureOr<int64_t> physicalFootprint =
      getPhysicalReadFootprintElements(resultType, &footprintReason);
  if (failed(physicalFootprint)) {
    return fail(footprintReason);
  }
  return VMIStatefulReadContract{*staticElements, *elementBytes,
                                 *physicalFootprint, *remainder, *offsetRange};
}

static VMIMemorySafeReadProof
computeSafeStatefulReadProof(Value source, Value offset,
                             VMIVRegType resultType) {
  VMIMemorySafeReadProof proof;

  auto fail = [&proof](const Twine &message) {
    proof.proven = false;
    proof.reason = message.str();
    return proof;
  };

  std::string contractReason;
  FailureOr<VMIStatefulReadContract> contract =
      getStatefulReadContract(source, offset, resultType, &contractReason);
  if (failed(contract)) {
    return fail(contractReason);
  }
  std::string envelopeReason;
  FailureOr<VMIStatefulReadEnvelopes> envelopes = buildStatefulReadEnvelopes(
      contract->staticElements, contract->offsetRange, contract->elementBytes,
      contract->physicalFootprint, contract->remainder, &envelopeReason);
  if (failed(envelopes)) {
    return fail(envelopeReason);
  }
  proof.readableEnvelope = envelopes->readable;
  proof.candidateReadEnvelope = envelopes->candidate;
  proof.proven = proof.readableEnvelope->contains(*proof.candidateReadEnvelope);
  if (!proof.proven) {
    proof.reason = (Twine("stateful physical read envelope [") +
                    Twine(proof.candidateReadEnvelope->begin) + ", " +
                    Twine(proof.candidateReadEnvelope->end) +
                    ") exceeds proven allocation envelope [" +
                    Twine(proof.readableEnvelope->begin) + ", " +
                    Twine(proof.readableEnvelope->end) + ")")
                       .str();
  }
  return proof;
}

VMIMemoryAccessPlan buildReadAccessPlan(Value source, Value offset,
                                        VMIVRegType resultType,
                                        VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Read;
  plan.valueType = resultType;
  VMIPhysicalMemorySegment segment;
  segment.address =
      VMIPlannedAddress{source, offset, resultType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, resultType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  segment.readSafety = computeSafeFullReadProof(
      source.getType(), getConstantIndexValue(offset), resultType);
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport =
      requireIdentityMemRefLayout(source.getType(), "source", source);
  return plan;
}

VMIMemoryAccessPlan buildReadAccessPlan(Value source, Type sourceType,
                                        VMIVRegType resultType,
                                        std::optional<int64_t> constantOffset,
                                        VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Read;
  plan.valueType = resultType;
  VMIPhysicalMemorySegment segment;
  segment.address = VMIPlannedAddress{source, {}, resultType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, resultType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  segment.readSafety =
      computeSafeFullReadProof(sourceType, constantOffset, resultType);
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport =
      requireIdentityMemRefLayout(sourceType, "source", source);
  return plan;
}

VMIMemoryAccessPlan buildWriteAccessPlan(Value destination, Value offset,
                                         VMIVRegType valueType,
                                         VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Write;
  plan.valueType = valueType;
  VMIPhysicalMemorySegment segment;
  segment.address =
      VMIPlannedAddress{destination, offset, valueType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, valueType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport = requireIdentityMemRefLayout(destination.getType(),
                                                   "destination", destination);
  return plan;
}

VMIMemoryAccessPlan buildWriteAccessPlan(Value destination,
                                         Type destinationType,
                                         VMIVRegType valueType,
                                         VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Write;
  plan.valueType = valueType;
  VMIPhysicalMemorySegment segment;
  segment.address =
      VMIPlannedAddress{destination, {}, valueType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, valueType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport =
      requireIdentityMemRefLayout(destinationType, "destination", destination);
  return plan;
}

static std::string
getUnavailableReadFallbackReason(VMIMemoryCoverageKind coverageKind) {
  std::string maskedLoadReason;
  if (coverageKind == VMIMemoryCoverageKind::Predicate) {
    maskedLoadReason =
        "; target true masked/non-faulting load is unavailable because the "
        "current VPTO pto.vlds surface has no mask operand";
  }
  std::string scratchReason =
      "; scratch memory fallback resource allocation is not implemented";
  std::string guardedReason =
      "; guarded memory fallback control-flow lowering is not implemented";
  return (Twine("partial/tail read needs a scratch, guarded, or true "
                "masked/non-faulting load fallback, but no such fallback "
                "resource plan is implemented") +
          maskedLoadReason + scratchReason + guardedReason)
      .str();
}

FailureOr<int64_t> verifyFullOrSafeReadVRegChunks(Operation *op,
                                                  VMIVRegType type,
                                                  Value source, Value offset,
                                                  PatternRewriter &rewriter) {
  std::string fullChunkReason;
  FailureOr<int64_t> lanesPerPart =
      checkFullDataPhysicalChunks(type, &fullChunkReason);
  bool usesAlignedLoad =
      isKnownAddressAligned(source, offset, type.getElementType(), 32);
  bool canUseAlignedFullChunk = succeeded(lanesPerPart) && usesAlignedLoad;
  if (canUseAlignedFullChunk) {
    return *lanesPerPart;
  }

  VMIMemorySafeReadProof safeReadProof =
      usesAlignedLoad
          ? computeSafeFullReadProof(source.getType(),
                                     getConstantIndexValue(offset), type)
          : computeSafeStatefulReadProof(source, offset, type);
  if (safeReadProof.proven) {
    lanesPerPart = getDataLanesPerPart(type.getElementType());
    if (succeeded(lanesPerPart)) {
      return *lanesPerPart;
    }
  }

  std::string legalizationReason =
      succeeded(lanesPerPart)
          ? "unaligned load requires a proven stateful physical read envelope"
          : fullChunkReason;
  (void)rewriter.notifyMatchFailure(
      op, Twine("memory lowering ") + legalizationReason +
              "; safe physical-read proof failed: " + safeReadProof.reason);
  return failure();
}

LogicalResult
checkSupportedLoadShape(VMIVRegType type, Value source, Type sourceType,
                        std::optional<int64_t> constantOffset,
                        std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      source, sourceType, type, constantOffset, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }

  VMILayoutSupport supports;
  if (failed(supports.getLoadLayoutFact(type, reason))) {
    return failure();
  }

  if (getDenseLaneStrideLoadDistToken(type)) {
    return success();
  }

  if (failed(getDataLanesPerPart(type.getElementType()))) {
    return fail("requires element type with known physical lane width");
  }
  return success();
}

LogicalResult checkSupportedDeinterleaveLoadShape(
    VMIDeinterleaveLoadOp op,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  VMILayoutSupport supports;
  if (failed(supports.getDeinterleaveLoadLayoutFactForLayouts(
          lowType, highType, reason))) {
    return failure();
  }
  if (!getX2MemoryDistToken(lowType.getElementType(), "DINTLV")) {
    return fail("requires 8/16/32-bit element type for vldsx2 DINTLV");
  }

  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      op.getSource(), op.getOffset(), lowType, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(lowType, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  return success();
}

static LogicalResult checkStorePhysicalCoverage(VMIVRegType type,
                                                std::string *reason) {
  if (getDenseLaneStrideStoreDistToken(type)) {
    return success();
  }

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(type, &fullChunkReason))) {
    return success();
  }

  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned layout");
  }
  if (failed(getDataLanesPerPart(type.getElementType()))) {
    return fail("requires known physical lanes per part");
  }
  bool directContiguousStore =
      layout.isContiguous() && layout.getLaneStride() == 1;
  if (directContiguousStore) {
    return success();
  }

  std::string materializationReason;
  if (succeeded(checkCanMaterializeToContiguous(type, &materializationReason))) {
    return success();
  }
  return fail(Twine("partial/tail store requires contiguous layout or "
                    "deinterleaved layout that can materialize to contiguous; "
                    "value ") +
              fullChunkReason + ", materialization " + materializationReason);
}

LogicalResult checkSupportedStoreShape(VMIVRegType type, Value destination,
                                       Type destinationType,
                                       std::string *reason) {
  VMIMemoryAccessPlan accessPlan = buildWriteAccessPlan(
      destination, destinationType, type, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    if (reason) {
      *reason = accessPlan.layoutSupport.reason;
    }
    return failure();
  }

  if (failed(checkSupportedMaskableVReg(type, reason))) {
    return failure();
  }

  VMILayoutSupport supports;
  if (failed(supports.getStoreLayoutFact(type, reason))) {
    return failure();
  }
  return checkStorePhysicalCoverage(type, reason);
}

LogicalResult checkSupportedInterleaveStoreShape(
    VMIInterleaveStoreOp op,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  VMILayoutAttr lowLayout = lowType.getLayoutAttr();
  VMILayoutAttr highLayout = highType.getLayoutAttr();
  bool nonContiguousInputs =
      !lowLayout || !highLayout || !lowLayout.isContiguous() ||
      !highLayout.isContiguous();
  if (nonContiguousInputs) {
    return fail("requires assigned contiguous low/high input layouts");
  }
  bool mismatchedInputs = lowType.getElementCount() != highType.getElementCount() ||
                        lowType.getElementType() != highType.getElementType();
  if (mismatchedInputs) {
    return fail("requires matching low/high input shape and element type");
  }
  if (!getX2MemoryDistToken(lowType.getElementType(), "INTLV")) {
    return fail("requires 8/16/32-bit element type for vstsx2 INTLV");
  }

  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(op.getDestination(), op.getOffset(), lowType,
                           VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (failed(checkSupportedMaskableVReg(lowType, reason))) {
    return failure();
  }

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(lowType, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  return success();
}

FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                             int64_t numGroups,
                                             std::string *reason = nullptr) {
  if (numGroups <= 0) {
    if (reason) {
      *reason = "requires num_groups to be positive";
    }
    return failure();
  }
  bool unevenGroups = type.getElementCount() % numGroups != 0;
  if (unevenGroups) {
    if (reason) {
      *reason = "requires num_groups to evenly divide logical lane count";
    }
    return failure();
  }
  return type.getElementCount() / numGroups;
}

LogicalResult checkSupportedGroupChunkShape(VMIVRegType type, int64_t groupSize,
                                            std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  bool nonContiguousLayout = !layout || !layout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires assigned contiguous layout");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart)) {
    return fail("requires known physical lanes per part");
  }
  if (groupSize <= 0) {
    return fail("requires a positive derived group size");
  }
  if (type.getElementCount() % groupSize != 0) {
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  }
  if (groupSize % *lanesPerPart != 0) {
    return fail("currently requires group size to be a multiple of physical "
                "lanes per part");
  }
  return success();
}

struct Deinterleaved2GroupStoreShape {
  int64_t lanesPerPart;
  int64_t groupCount;
  int64_t chunksPerGroupPerPart;
};

static FailureOr<int64_t>
getDeinterleaved2StoreLanes(VMIVRegType type, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  bool unsupportedLayout =
      !layout || !layout.isDeinterleaved() || layout.getFactor() != 2 ||
      layout.getLaneStride() != 1;
  if (unsupportedLayout) {
    return fail("requires deinterleaved=2 value layout");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes)) {
    return fail("requires known physical lanes per part");
  }
  if (!getX2MemoryDistToken(type.getElementType(), "INTLV")) {
    return fail("requires 8/16/32-bit element type for vstsx2 INTLV");
  }
  return *lanes;
}

static FailureOr<Deinterleaved2GroupStoreShape>
getDeinterleaved2GroupStoreGeometry(VMIVRegType type, int64_t groupSize,
                                    int64_t lanesPerPart,
                                    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<Deinterleaved2GroupStoreShape> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (groupSize <= 0) {
    return fail("requires positive derived group size");
  }
  int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
  bool unevenLogicalGroups = type.getElementCount() % safeGroupSize != 0;
  if (unevenLogicalGroups) {
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  }
  if (lanesPerPart <= 0) {
    return fail("requires positive physical lane count");
  }
  int64_t pairLanes = 2 * lanesPerPart;
  if (groupSize % pairLanes != 0) {
    return fail("requires group size to be a multiple of two physical chunks");
  }

  FailureOr<int64_t> part0Chunks = getDataChunksInPart(type, /*part=*/0);
  FailureOr<int64_t> part1Chunks = getDataChunksInPart(type, /*part=*/1);
  bool mismatchedPartChunks =
      failed(part0Chunks) || failed(part1Chunks) ||
      *part0Chunks != *part1Chunks;
  if (mismatchedPartChunks) {
    return fail("requires matching deinterleaved part chunk counts");
  }

  int64_t groupCount = type.getElementCount() / groupSize;
  int64_t chunksPerGroupPerPart = groupSize / pairLanes;
  if (*part0Chunks != groupCount * chunksPerGroupPerPart) {
    return fail("requires deinterleaved chunks to align with group rows");
  }
  return Deinterleaved2GroupStoreShape{lanesPerPart, groupCount,
                                      chunksPerGroupPerPart};
}

static FailureOr<Deinterleaved2GroupStoreShape>
getDeinterleaved2GroupStoreShape(VMIVRegType type, int64_t groupSize,
                                 std::string *reason) {
  FailureOr<int64_t> lanesPerPart =
      getDeinterleaved2StoreLanes(type, reason);
  if (failed(lanesPerPart)) {
    return failure();
  }
  return getDeinterleaved2GroupStoreGeometry(type, groupSize,
                                              *lanesPerPart, reason);
}

LogicalResult checkDeinterleaved2GroupStoreChunkShape(
    VMIVRegType type, int64_t groupSize, int64_t *lanesPerPart,
    int64_t *groupCount, int64_t *chunksPerGroupPerPart,
    std::string *reason) {
  FailureOr<Deinterleaved2GroupStoreShape> shape =
      getDeinterleaved2GroupStoreShape(type, groupSize, reason);
  if (succeeded(shape)) {
    *lanesPerPart = shape->lanesPerPart;
    *groupCount = shape->groupCount;
    *chunksPerGroupPerPart = shape->chunksPerGroupPerPart;
    return success();
  }
  return failure();
}

LogicalResult checkSupportedBlockDeinterleavedGroupLoadShape(
    VMIGroupLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutSupport supports;
  if (failed(supports.getGroupLoadLayoutFact(op, reason))) {
    return failure();
  }
  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                          VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("block_deinterleaved group_load requires !pto.ptr source");
  }
  bool hasGroupMultiple = op.getNumGroupsAttr().getInt() % 8 == 0;
  if (!hasGroupMultiple) {
    return fail("block_deinterleaved group_load requires num_groups multiple of 8");
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  if (!rowStride || *rowStride <= 0 || *rowStride % 8 != 0) {
    return fail("block_deinterleaved group_load requires constant positive "
                "row_stride divisible by 8 f32 elements");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
    return fail(Twine("block_deinterleaved group_load requires full physical "
                      "result chunks; ") +
                fullChunkReason);
  }
  return success();
}

LogicalResult
checkSupportedContiguousGroupLoadShape(VMIGroupLoadOp op,
                                       VMIVRegType resultType,
                                       int64_t groupSize, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getGroupLoadLayoutFact(op, reason))) {
    return failure();
  }
  if (failed(checkSupportedLoadShape(resultType, op.getSource(),
                                     op.getSource().getType(), std::nullopt,
                                     reason))) {
    return failure();
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  bool unitGroupRowStride = rowStride && *rowStride == groupSize;
  if (unitGroupRowStride) {
    return success();
  }
  return checkSupportedGroupChunkShape(resultType, groupSize, reason);
}

LogicalResult
checkSupportedGroupLoadShape(VMIGroupLoadOp op, std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return emitLogicalFailure(reason, "requires assigned result layout");
  }
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize)) {
    return failure();
  }

  if (resultLayout.isContiguous()) {
    return checkSupportedContiguousGroupLoadShape(op, resultType, *groupSize,
                                                  reason);
  }

  bool supportsBlockDeinterleaved =
      resultLayout.isBlockDeinterleaved() && resultType.getElementType().isF32();
  if (supportsBlockDeinterleaved) {
    return checkSupportedBlockDeinterleavedGroupLoadShape(op, resultType,
                                                          reason);
  }

  return emitLogicalFailure(
      reason, "requires contiguous or block_deinterleaved f32 result layout");
}

LogicalResult checkSupportedSlots1GroupSlotLoadShape(
    VMIGroupSlotLoadOp op, VMIVRegType resultType, std::string *reason);

LogicalResult checkSupportedSlots8GroupSlotLoadShape(
    VMIGroupSlotLoadOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<int64_t> sourceGroupStride =
      getConstantIndexValue(op.getSourceGroupStride());
  bool nonUnitSourceStride = !sourceGroupStride || *sourceGroupStride != 1;
  if (nonUnitSourceStride) {
    return fail("slots=8 group_slot_load requires constant unit "
                "source_group_stride");
  }
  return success();
}

static LogicalResult checkGroupSlotLoadMemoryContract(
    VMIGroupSlotLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      op.getSource(), op.getOffset(), resultType, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("group_slot_load requires !pto.ptr source");
  }
  return success();
}

LogicalResult checkSupportedGroupSlotLoadShape(
    VMIGroupSlotLoadOp op,
    std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutSupport supports;
  FailureOr<VMIGroupSlotLayoutFact> fact = supports.getGroupSlotLoadLayoutFact(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(fact)) {
    return failure();
  }

  if (failed(checkGroupSlotLoadMemoryContract(op, resultType, reason))) {
    return failure();
  }

  if (fact->slots == 8) {
    return checkSupportedSlots8GroupSlotLoadShape(op, reason);
  }

  return checkSupportedSlots1GroupSlotLoadShape(op, resultType, reason);
}

LogicalResult checkSupportedSlots1GroupSlotLoadShape(
    VMIGroupSlotLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  bool unsupportedElementWidth = elementBits == 0 || 256 % elementBits != 0;
  if (unsupportedElementWidth) {
    return fail("slots=1 group_slot_load requires supported element width");
  }
  int64_t alignedStrideElems = 256 / elementBits;
  std::optional<int64_t> sourceGroupStride =
      getConstantIndexValue(op.getSourceGroupStride());
  if (!sourceGroupStride || *sourceGroupStride <= 0 ||
      *sourceGroupStride % alignedStrideElems != 0) {
    return fail(Twine("slots=1 group_slot_load currently lowers as one "
                      "lane-0 vsldb per group and requires constant "
                      "positive source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B load alignment; packed or unaligned "
                "scalar load lowering is not implemented");
  }
  return success();
}

LogicalResult checkSupportedGroupBroadcastLoadMemory(
    VMIGroupBroadcastLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                          VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("group_broadcast_load requires !pto.ptr source");
  }
  return success();
}

LogicalResult checkSupportedGroupBroadcastLoadShape(
    VMIGroupBroadcastLoadOp op, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getGroupBroadcastLoadSupport(op, reason))) {
    return failure();
  }
  return checkSupportedGroupBroadcastLoadMemory(
      op, cast<VMIVRegType>(op.getResult().getType()), reason);
}

static bool isCompactSmallGroupStore(VMILayoutAttr layout,
                                     VMIVRegType valueType, int64_t numGroups,
                                     std::optional<int64_t> rowStride) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(valueType.getElementType());
  int64_t payloadBits =
      valueType.getElementCount() * static_cast<int64_t>(elementBits);
  return layout && layout.isGroupSlots() && layout.getNumGroups() == numGroups &&
         layout.getSlots() == 8 &&
         (layout.getLaneStride() == 1 || layout.getLaneStride() == 2 ||
          layout.getLaneStride() == 4) &&
         (valueType.getElementCount() == 4 ||
          valueType.getElementCount() == 8) &&
         numGroups == valueType.getElementCount() && elementBits > 0 &&
         payloadBits > 0 && payloadBits < 256 && payloadBits % 32 == 0 &&
         rowStride && *rowStride == 1;
}

struct OneBlockGroupStorePlan {
  int64_t groupSize = 0;
  int64_t groupsPerPart = 0;
  int64_t blockStride = 0;
};

FailureOr<OneBlockGroupStorePlan> getOneBlockGroupStorePlan(
    VMIGroupStoreOp op, VMIVRegType valueType,
    const VMIGroupStoreLayoutFact &fact, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<OneBlockGroupStorePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  bool unsupportedBlockLayout =
      fact.blockClass != VMIGroupBlockClass::OneBlock || !layout ||
      !layout.isContiguous();
  if (unsupportedBlockLayout) {
    return fail("one-block group_store requires contiguous layout");
  }
  bool unsupportedBlockShape =
      fact.groupSize <= 0 || fact.groupSize != fact.vcgBlockElems ||
      fact.lanesPerPart <= 0 || fact.lanesPerPart % fact.groupSize != 0;
  if (unsupportedBlockShape) {
    return fail("one-block group_store requires one 32B group per VCG block");
  }
  if (!isa<PtrType>(op.getDestination().getType())) {
    return fail("one-block group_store requires !pto.ptr destination");
  }

  std::optional<int64_t> rowStride =
      getConstantIndexValue(op.getRowStride());
  bool unalignedRowStride =
      !rowStride || *rowStride <= 0 || *rowStride % fact.groupSize != 0;
  if (unalignedRowStride) {
    return fail("one-block group_store requires a constant positive row_stride "
                "aligned to 32B blocks");
  }

  int64_t blockStride = *rowStride / fact.groupSize;
  if (blockStride > 0xffff) {
    return fail("one-block group_store block_stride exceeds the 16-bit vsstb "
                "control field");
  }

  return OneBlockGroupStorePlan{
      fact.groupSize, fact.lanesPerPart / fact.groupSize, blockStride};
}

LogicalResult
checkSupportedCompactSmallGroupStoreShape(VMIGroupStoreOp op,
                                           VMIVRegType valueType,
                                           std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!isa<PtrType>(op.getDestination().getType())) {
    return fail("compact small group_store requires !pto.ptr destination");
  }
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(op.getDestination(), op.getOffset(), valueType,
                           VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  return success();
}

LogicalResult
checkSupportedGroupSlotsStoreShape(VMIGroupStoreOp op, VMIVRegType valueType,
                                    std::optional<int64_t> rowStride,
                                    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutSupport supports;
  FailureOr<VMIGroupSlotLayoutFact> fact = supports.getGroupStoreLayoutFact(
      valueType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(fact)) {
    return failure();
  }

  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(op.getDestination(), op.getOffset(), valueType,
                           VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }

  if (fact->slots == 1) {
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(valueType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0) {
      return fail("slots=1 group_store requires supported element width");
    }
    if (rowStride && *rowStride <= 0) {
      return fail("slots=1 group_store requires positive row_stride when "
                  "row_stride is constant");
    }
    if (!getPointStoreDistToken(valueType.getElementType())) {
      return fail("slots=1 group_store requires 1PT_B8/B16/B32 store "
                  "support");
    }
    return success();
  }

  if (!rowStride || *rowStride != 1) {
    return fail("slots=8 group_store currently requires constant unit "
                "row_stride");
  }
  return success();
}

LogicalResult
checkSupportedGroupStorePhysicalShape(
    VMIGroupStoreOp op, VMIVRegType valueType,
    const VMIGroupStoreLayoutFact &fact, std::string *reason) {
  if (failed(checkSupportedStoreShape(valueType,
                                      op.getDestination(),
                                      op.getDestination().getType(), reason))) {
    return failure();
  }
  bool oneBlock = fact.blockClass == VMIGroupBlockClass::OneBlock;
  if (oneBlock) {
    if (failed(getOneBlockGroupStorePlan(op, valueType, fact, reason))) {
      return failure();
    }
    return success();
  }
  bool contiguousGroupChunks = succeeded(
      checkSupportedGroupChunkShape(valueType, fact.groupSize, reason));
  if (contiguousGroupChunks) {
    return success();
  }

  int64_t lanesPerPart = 0;
  int64_t groupCount = 0;
  int64_t chunksPerGroupPerPart = 0;
  return checkDeinterleaved2GroupStoreChunkShape(
      valueType, fact.groupSize, &lanesPerPart, &groupCount,
      &chunksPerGroupPerPart, reason);
}

LogicalResult
checkSupportedGroupStoreByLayout(VMIGroupStoreOp op, VMIVRegType valueType,
                                 VMILayoutAttr layout,
                                 std::optional<int64_t> rowStride,
                                 std::string *reason) {
  bool compactSmallGroup = isCompactSmallGroupStore(
      layout, valueType, op.getNumGroupsAttr().getInt(), rowStride);
  if (compactSmallGroup) {
    return checkSupportedCompactSmallGroupStoreShape(op, valueType, reason);
  }
  if (layout && layout.isGroupSlots()) {
    return checkSupportedGroupSlotsStoreShape(op, valueType, rowStride, reason);
  }

  VMILayoutSupport supports;
  FailureOr<VMIGroupStoreLayoutFact> fact =
      supports.getGroupStoreLayoutFact(op, valueType, reason);
  if (failed(fact)) {
    return failure();
  }
  return checkSupportedGroupStorePhysicalShape(op, valueType, *fact, reason);
}

LogicalResult
checkSupportedGroupStoreShape(VMIGroupStoreOp op, std::string *reason) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  VMILayoutAttr layout = valueType.getLayoutAttr();
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  return checkSupportedGroupStoreByLayout(op, valueType, layout, rowStride,
                                          reason);
}

LogicalResult
checkSupportedMaskedLoadShape(VMIMaskedLoadOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                          VMIMemoryCoverageKind::Predicate);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!resultLayout || !passthruLayout || !maskLayout) {
    return fail("requires assigned result, passthru, and mask layouts");
  }
  bool nonContiguousLayout = !resultLayout.isContiguous() ||
                             !passthruLayout.isContiguous() ||
                             !maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous result, passthru, and mask layouts");
  }

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
    return success();
  }

  if (accessPlan.front().readSafety.proven) {
    return success();
  }
  std::string fallbackReason =
      getUnavailableReadFallbackReason(VMIMemoryCoverageKind::Predicate);
  return fail(Twine("partial/tail masked_load requires statically safe "
                    "full-read footprint; value ") +
              fullChunkReason + ", safe-read proof " +
              accessPlan.front().readSafety.reason +
              "; fallback unavailable: " + fallbackReason);
}

LogicalResult checkSupportedGatherPhysicalShape(
    VMIVRegType resultType, VMIVRegType indicesType, VMIVRegType passthruType,
    VMIMaskType maskType, bool requiresFullChunks, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasPhysicalArity = succeeded(resultArity) && succeeded(indicesArity) &&
                          succeeded(passthruArity) && succeeded(maskArity);
  if (!hasPhysicalArity) {
    return fail("requires computable physical arity");
  }
  if (*resultArity != *indicesArity || *resultArity != *passthruArity ||
      *resultArity != *maskArity) {
    return fail("requires result, indices, passthru, and mask to have the "
                "same physical arity");
  }
  if (*resultArity > mlir::pto::kValue4) {
    return fail("gather exceeds the 4 physical register limit per VMI "
                "instruction");
  }
  if (!requiresFullChunks) {
    return success();
  }
  std::string resultReason;
  std::string indicesReason;
  std::string passthruReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultReason))) {
    return fail(Twine("result requires full physical chunks; ") + resultReason);
  }
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason))) {
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  }
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason))) {
    return fail(Twine("passthru requires full physical chunks; ") +
                passthruReason);
  }
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason))) {
    return fail(Twine("mask requires full physical chunks; ") + maskReason);
  }
  return success();
}

static bool haveMatchingIntegerSignedness(IntegerType sourceType,
                                          IntegerType resultType) {
  return (sourceType.isUnsigned() && resultType.isUnsigned()) ||
         (!sourceType.isUnsigned() && !resultType.isUnsigned());
}

static bool isB8To16GatherContract(IntegerType sourceType,
                                   IntegerType resultType,
                                   IntegerType indexType,
                                   VMIMaskType maskType) {
  return sourceType.getWidth() == mlir::pto::kValue8 &&
         resultType.getWidth() == mlir::pto::kValue16 &&
         indexType.isUnsigned() && indexType.getWidth() == 16 &&
         maskType.getGranularity() == "b16" &&
         haveMatchingIntegerSignedness(sourceType, resultType);
}

static bool isSameWidth16IntegerGatherContract(IntegerType sourceType,
                                               IntegerType resultType,
                                               IntegerType indexType,
                                               VMIMaskType maskType) {
  return sourceType.getWidth() == mlir::pto::kValue16 &&
         resultType.getWidth() == mlir::pto::kValue16 &&
         indexType.isUnsigned() && indexType.getWidth() == 16 &&
         maskType.getGranularity() == "b16" &&
         haveMatchingIntegerSignedness(sourceType, resultType);
}

static bool isSameWidth16FloatGatherContract(Type sourceElementType,
                                             Type resultElementType,
                                             IntegerType indexType,
                                             VMIMaskType maskType) {
  return sourceElementType == resultElementType &&
         (sourceElementType.isF16() || sourceElementType.isBF16()) &&
         indexType.isUnsigned() && indexType.getWidth() == 16 &&
         maskType.getGranularity() == "b16";
}

LogicalResult
checkGatherElementContract(VMIVRegType resultType, VMIVRegType indicesType,
                            VMIMaskType maskType, Type sourceElemType,
                            std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned()) {
    return fail("requires signless or unsigned integer indices");
  }
  auto sourceInt = dyn_cast<IntegerType>(sourceElemType);
  auto resultInt = dyn_cast<IntegerType>(resultType.getElementType());
  bool isB8To16Gather =
      resultBits == 16 && sourceInt && resultInt &&
      isB8To16GatherContract(sourceInt, resultInt, indexElementType, maskType);
  bool isSameWidth16Gather =
      resultBits == 16 &&
      ((sourceInt && resultInt && isSameWidth16IntegerGatherContract(
                                      sourceInt, resultInt, indexElementType,
                                      maskType)) ||
       isSameWidth16FloatGatherContract(sourceElemType,
                                        resultType.getElementType(),
                                        indexElementType, maskType));
  bool isB16Gather = isSameWidth16Gather || isB8To16Gather;
  bool isB32Gather = resultBits == 32 && indexElementType.getWidth() == 32 &&
                     maskType.getGranularity() == "b32";
  if (!isB16Gather && !isB32Gather) {
    return fail("requires either 32-bit results with 32-bit indices and b32 "
                "mask, or ui16/i16/f16/bf16 results with ui16 indices and "
                "b16 mask (including i8/ui8 -> i16/ui16 promotion)");
  }
  return success();
}

LogicalResult
checkGatherLayoutAndSource(VMIGatherOp op, VMIVRegType resultType,
                            VMIVRegType indicesType, VMIVRegType passthruType,
                            VMIMaskType maskType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!resultLayout || !indicesLayout || !passthruLayout || !maskLayout) {
    return fail("requires assigned result, indices, passthru, and mask layouts");
  }
  bool nonContiguousLayout =
      !resultLayout.isContiguous() || !indicesLayout.isContiguous() ||
      !passthruLayout.isContiguous() || !maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous result, indices, passthru, and mask layouts");
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("requires !pto.ptr source because pto.vgather2_bc is pointer-only");
  }
  return success();
}

LogicalResult
checkGatherPhysicalChunkRequirement(VMIVRegType resultType,
                                    VMIVRegType indicesType,
                                    VMIVRegType passthruType,
                                    VMIMaskType maskType,
                                    bool isB16Gather, bool isB32Gather,
                                    std::string *reason) {
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity)) {
    return failure();
  }
  bool requiresFullChunks = isB32Gather;
  if (isB16Gather) {
    requiresFullChunks = *resultArity != 1;
  }
  return checkSupportedGatherPhysicalShape(
      resultType, indicesType, passthruType, maskType, requiresFullChunks,
      reason);
}

LogicalResult
checkSupportedGatherShape(VMIGatherOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto indicesType = cast<VMIVRegType>(op.getIndices().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  if (failed(checkGatherLayoutAndSource(op, resultType, indicesType,
                                        passthruType, maskType, reason))) {
    return failure();
  }

  Type sourceElemType = getMemoryElementType(op.getSource().getType());
  std::string elementReason;
  if (failed(checkGatherElementContract(resultType, indicesType, maskType,
                                        sourceElemType, &elementReason))) {
    return fail(elementReason);
  }

  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  bool isB16Gather = resultBits == 16 && indexElementType &&
                     indexElementType.getWidth() == 16 &&
                     maskType.getGranularity() == "b16";
  bool isB32Gather = resultBits == 32 && indexElementType &&
                     indexElementType.getWidth() == 32 &&
                     maskType.getGranularity() == "b32";

  return checkGatherPhysicalChunkRequirement(
      resultType, indicesType, passthruType, maskType, isB16Gather,
      isB32Gather, reason);
}

LogicalResult checkSupportedScatterPhysicalShape(
    VMIVRegType valueType, VMIVRegType indicesType, VMIMaskType maskType,
    bool requiresFullChunks, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasPhysicalArity = succeeded(valueArity) && succeeded(indicesArity) &&
                          succeeded(maskArity);
  if (!hasPhysicalArity) {
    return fail("requires computable physical arity");
  }
  if (*valueArity != *indicesArity || *valueArity != *maskArity) {
    return fail("requires value, indices, and mask to have the same physical "
                "arity");
  }
  if (!requiresFullChunks) {
    return success();
  }
  std::string valueReason;
  std::string indicesReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &valueReason))) {
    return fail(Twine("value requires full physical chunks; ") + valueReason);
  }
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason))) {
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  }
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason))) {
    return fail(Twine("mask requires full physical chunks; ") + maskReason);
  }
  return success();
}

LogicalResult
checkScatterLayoutAndDestination(VMIScatterOp op, VMIVRegType valueType,
                                 VMIVRegType indicesType, VMIMaskType maskType,
                                 std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  bool missingLayout = !valueLayout || !indicesLayout || !maskLayout;
  if (missingLayout) {
    return fail("requires assigned value, indices, and mask layouts");
  }
  bool nonContiguousLayout = !valueLayout.isContiguous() ||
                             !indicesLayout.isContiguous() ||
                             !maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous value, indices, and mask layouts");
  }
  if (!isa<PtrType>(op.getDestination().getType())) {
    return fail("requires !pto.ptr destination because pto.vscatter is "
                "pointer-only");
  }
  return success();
}

LogicalResult
checkScatterElementContract(VMIVRegType valueType, VMIVRegType indicesType,
                            VMIMaskType maskType,
                            std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  unsigned valueBits =
      pto::getPTOStorageElemBitWidth(valueType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned()) {
    return fail("requires signless or unsigned integer indices");
  }
  bool isB8Scatter = valueBits == 8 && indexElementType.getWidth() == 16 &&
                     maskType.getGranularity() == "b16";
  bool isB16Scatter = valueBits == 16 && indexElementType.getWidth() == 16 &&
                      maskType.getGranularity() == "b16";
  bool isB32Scatter = valueBits == 32 && indexElementType.getWidth() == 32 &&
                      maskType.getGranularity() == "b32";
  bool unsupportedContract = !isB8Scatter && !isB16Scatter && !isB32Scatter;
  if (unsupportedContract) {
    return fail("requires either 32-bit values with 32-bit indices and b32 "
                "mask, 16-bit values with 16-bit indices and b16 "
                "mask, or 8-bit values with 16-bit indices and b16 "
                "mask");
  }
  return success();
}

struct ScatterShapeTypes {
  VMIVRegType value;
  VMIVRegType indices;
  VMIMaskType mask;
};

static ScatterShapeTypes getScatterShapeTypes(VMIScatterOp op) {
  return ScatterShapeTypes{
      cast<VMIVRegType>(op.getValue().getType()),
      cast<VMIVRegType>(op.getIndices().getType()),
      cast<VMIMaskType>(op.getMask().getType())};
}

LogicalResult
checkSupportedScatterShape(VMIScatterOp op, std::string *reason) {
  ScatterShapeTypes types = getScatterShapeTypes(op);
  if (failed(checkScatterLayoutAndDestination(op, types.value, types.indices,
                                              types.mask, reason))) {
    return failure();
  }
  if (failed(checkScatterElementContract(types.value, types.indices,
                                         types.mask, reason))) {
    return failure();
  }

  return checkSupportedScatterPhysicalShape(types.value, types.indices,
                                            types.mask, true, reason);
}

LogicalResult
checkSinglePhysicalStrideAccess(Type dataType, Type maskType,
                                StringRef errorMessage, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> dataArity = getVMIPhysicalArity(dataType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasArity = succeeded(dataArity) && succeeded(maskArity);
  if (!hasArity) {
    return fail("requires computable physical arity");
  }
  if (*dataArity != 1 || *maskArity != 1) {
    return fail(errorMessage);
  }
  return success();
}

struct StrideMemoryContract {
  Type dataType;
  Type maskType;
  VMILayoutAttr dataLayout;
  VMILayoutAttr maskLayout;
  Type pointerType;
  StringRef dataName;
  StringRef pointerDiagnostic;
  StringRef arityDiagnostic;
};

static LogicalResult checkStrideMemoryContract(
    const StrideMemoryContract &contract, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!contract.dataLayout || !contract.maskLayout) {
    return fail(Twine("requires assigned ") + contract.dataName +
                " and mask layouts");
  }
  bool nonContiguousLayout =
      !contract.dataLayout.isContiguous() ||
      !contract.maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail(Twine("requires contiguous ") + contract.dataName +
                " and mask layouts");
  }
  if (!isa<PtrType>(contract.pointerType)) {
    return fail(contract.pointerDiagnostic);
  }
  return checkSinglePhysicalStrideAccess(
      contract.dataType, contract.maskType, contract.arityDiagnostic, reason);
}

LogicalResult
checkSupportedStrideStoreShape(VMIStrideStoreOp op, std::string *reason) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  if (failed(checkSupportedStoreShape(valueType, op.getDestination(),
                                      op.getDestination().getType(), reason))) {
    return failure();
  }

  StrideMemoryContract contract{
      valueType,
      maskType,
      valueType.getLayoutAttr(),
      maskType.getLayoutAttr(),
      op.getDestination().getType(),
      "value",
      "requires !pto.ptr destination because pto.vsstb is pointer-only",
      "currently supports one physical value/mask chunk"};
  return checkStrideMemoryContract(contract, reason);
}

LogicalResult
checkSupportedStrideLoadShape(VMIStrideLoadOp op, std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  StrideMemoryContract contract{
      resultType,
      maskType,
      resultType.getLayoutAttr(),
      maskType.getLayoutAttr(),
      op.getSource().getType(),
      "result",
      "requires !pto.ptr source because pto.vsldb is pointer-only",
      "currently supports one physical result/mask chunk"};
  return checkStrideMemoryContract(contract, reason);
}

Value stripMaskMaterialization(Value value) {
  while (true) {
    if (auto ensure = value.getDefiningOp<VMIEnsureMaskLayoutOp>()) {
      value = ensure.getSource();
      continue;
    }
    if (auto ensure = value.getDefiningOp<VMIEnsureMaskGranularityOp>()) {
      value = ensure.getSource();
      continue;
    }
    return value;
  }
}

bool isStaticAllActiveMask(Value mask, int64_t expectedLanes,
                           std::string *reason = nullptr) {
  mask = stripMaskMaterialization(mask);
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return false;
  };

  if (auto createMask = mask.getDefiningOp<VMICreateMaskOp>()) {
    auto activeConstant =
        createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      return fail("create_mask active_lanes is dynamic");
    }
    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr) {
      return fail("create_mask active_lanes is not an integer constant");
    }
    return activeAttr.getInt() >= expectedLanes
               ? true
               : fail("create_mask active_lanes is smaller than the logical "
                      "lane count");
  }

  if (auto constantMask = mask.getDefiningOp<VMIConstantMaskOp>()) {
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantMask.getValue());
    if (!denseAttr) {
      return fail("constant_mask is not a dense integer mask");
    }
    bool invalidElementCount = denseAttr.getNumElements() != expectedLanes;
    if (invalidElementCount) {
      return fail("constant_mask element count does not match the logical "
                  "lane count");
    }
    auto values = denseAttr.getValues<bool>();
    for (bool value : values) {
      if (!value) {
        return fail("constant_mask contains an inactive lane");
      }
    }
    return true;
  }

  return fail("mask is not a static all-active create_mask or constant_mask");
}

static LogicalResult checkExpandLoadRuntimeArity(
    VMIVRegType resultType, VMIVRegType passthruType, VMIMaskType maskType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasComputableArity = succeeded(resultArity) &&
                            succeeded(passthruArity) && succeeded(maskArity);
  if (!hasComputableArity) {
    return fail("runtime-mask path requires computable physical arity");
  }
  bool hasSingleChunk = *resultArity == 1 && *passthruArity == 1 &&
                        *maskArity == 1;
  if (!hasSingleChunk) {
    return fail("runtime-mask path currently supports only one physical "
                "chunk because prefix indices must not reset across chunks");
  }
  return success();
}

static LogicalResult checkExpandLoadRuntimeFullChunks(
    VMIVRegType resultType, VMIVRegType passthruType, VMIMaskType maskType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::string fullChunkReason;
  std::string passthruReason;
  std::string maskFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
    return fail(Twine("runtime-mask result requires full physical chunks; ") +
                fullChunkReason);
  }
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason))) {
    return fail(Twine("runtime-mask passthru requires full physical chunks; ") +
                passthruReason);
  }
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason))) {
    return fail(Twine("runtime-mask mask requires full physical chunks; ") +
                maskFullReason);
  }
  return success();
}

LogicalResult
checkSupportedExpandLoadRuntimePath(
    VMIExpandLoadOp op, VMIVRegType resultType, VMIVRegType passthruType,
    VMIMaskType maskType, StringRef allActivePathReason, std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail(Twine("runtime-mask path requires !pto.ptr source because "
                      "pto.vgather2_bc is pointer-only; all-active path ") +
                allActivePathReason);
  }
  bool unsupportedResultWidth =
      pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 32;
  if (unsupportedResultWidth) {
    return fail("runtime-mask path currently requires 32-bit result element "
                "type so prefix indices and gather result lane counts match");
  }
  bool unsupportedMaskGranularity = maskType.getGranularity() != "b32";
  if (unsupportedMaskGranularity) {
    return fail("runtime-mask path requires b32 mask granularity");
  }
  if (failed(checkExpandLoadRuntimeArity(resultType, passthruType, maskType,
                                         reason))) {
    return failure();
  }
  return checkExpandLoadRuntimeFullChunks(resultType, passthruType, maskType,
                                          reason);
}

LogicalResult
checkSupportedExpandLoadCommonShape(VMIExpandLoadOp op,
                                     VMIVRegType resultType,
                                     VMIVRegType passthruType,
                                     VMIMaskType maskType,
                                     VMIMemoryAccessPlan &accessPlan,
                                     std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  accessPlan = buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                                   VMIMemoryCoverageKind::Predicate);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  bool missingLayout = !resultType.getLayoutAttr() ||
                       !passthruType.getLayoutAttr() ||
                       !maskType.getLayoutAttr();
  if (missingLayout) {
    return fail("requires assigned result, passthru, and mask layouts");
  }
  bool nonContiguousLayout = !resultType.getLayoutAttr().isContiguous() ||
                             !passthruType.getLayoutAttr().isContiguous() ||
                             !maskType.getLayoutAttr().isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous result, passthru, and mask layouts");
  }
  return success();
}

static bool hasSafeExpandLoadAllActivePath(
    VMIVRegType resultType, const VMIMemoryAccessPlan &accessPlan,
    bool staticAllActive, std::string *pathReason) {
  if (!staticAllActive) {
    return false;
  }
  std::string fullChunkReason;
  bool fullChunks =
      succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason));
  if (fullChunks || accessPlan.front().readSafety.proven) {
    return true;
  }
  std::string fallbackReason =
      getUnavailableReadFallbackReason(VMIMemoryCoverageKind::Predicate);
  *pathReason =
      (Twine("requires full physical chunks or statically safe full-read "
             "footprint; value ") +
       fullChunkReason + ", safe-read proof " +
       accessPlan.front().readSafety.reason +
       "; fallback unavailable: " + fallbackReason)
          .str();
  return false;
}

LogicalResult
checkSupportedExpandLoadShape(VMIExpandLoadOp op, std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMIMemoryAccessPlan accessPlan;
  if (failed(checkSupportedExpandLoadCommonShape(
          op, resultType, passthruType, maskType, accessPlan, reason))) {
    return failure();
  }

  std::string maskReason;
  bool staticAllActive = isStaticAllActiveMask(
      op.getMask(), resultType.getElementCount(), &maskReason);

  std::string allActivePathReason;
  bool staticFullChunks = hasSafeExpandLoadAllActivePath(
      resultType, accessPlan, staticAllActive, &allActivePathReason);
  if (staticFullChunks) {
    return success();
  }

  if (!staticAllActive) {
    allActivePathReason =
        maskReason.empty() ? "requires static all-active mask" : maskReason;
  }

  return checkSupportedExpandLoadRuntimePath(
      op, resultType, passthruType, maskType, allActivePathReason, reason);
}

static LogicalResult checkMaskedStoreFullChunks(VMIVRegType valueType,
                                                VMIMaskType maskType,
                                                std::string &valueReason,
                                                std::string &maskReason) {
  return succeeded(checkFullDataPhysicalChunks(valueType, &valueReason)) &&
                 succeeded(checkFullVMIPhysicalChunks(maskType, &maskReason))
             ? success()
             : failure();
}

static LogicalResult checkMaskedStoreLayoutAndArity(
    VMIVRegType valueType, VMIMaskType maskType, std::string &valueReason,
    std::string &maskReason, std::string *reason) {
  FailureOr<AssignedValueMaskLayouts> layouts =
      getAssignedValueMaskLayouts(valueType, maskType, reason);
  if (failed(layouts)) {
    return failure();
  }
  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool mismatchedArity = failed(valueArity) || failed(maskArity) ||
                         *valueArity != *maskArity;
  if (mismatchedArity) {
    return emitLogicalFailure(reason, "requires matching value/mask physical arity");
  }
  if (layouts->value.hasDenseLaneStride()) {
    VMILayoutSupport supports;
    if (succeeded(supports.getMaskedStoreLayoutFact(valueType, maskType,
                                                    reason))) {
      return success();
    }
  }
  std::string valueMaterializationReason;
  FailureOr<int64_t> valueParts = getContiguousMaterializationPartCount(
      valueType, &valueMaterializationReason);
  if (failed(valueParts)) {
    return emitLogicalFailure(reason,
                              Twine("value cannot materialize to contiguous; value ") +
                                  valueReason + ", materialization " +
                                  valueMaterializationReason);
  }
  std::string maskMaterializationReason;
  FailureOr<int64_t> maskParts = getContiguousMaterializationPartCount(
      maskType, &maskMaterializationReason);
  if (failed(maskParts)) {
    return emitLogicalFailure(reason,
                              Twine("mask cannot materialize to contiguous; mask ") +
                                  maskReason + ", materialization " +
                                  maskMaterializationReason);
  }
  if (*valueParts != *maskParts) {
    return emitLogicalFailure(
        reason, "requires value/mask contiguous materialization arity to match");
  }
  return success();
}

LogicalResult
checkSupportedMaskedStoreShape(VMIVRegType valueType, VMIMaskType maskType,
                               Value destination, Type destinationType,
                               std::string *reason) {
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(destination, destinationType, valueType,
                           VMIMemoryCoverageKind::Predicate);
  if (!accessPlan.layoutSupport.isSupported()) {
    if (reason) {
      *reason = accessPlan.layoutSupport.reason;
    }
    return failure();
  }

  std::string valueReason;
  std::string maskReason;
  if (succeeded(checkMaskedStoreFullChunks(valueType, maskType, valueReason,
                                            maskReason))) {
    return success();
  }

  return checkMaskedStoreLayoutAndArity(valueType, maskType, valueReason,
                                        maskReason, reason);
}

FailureOr<int64_t> getContiguousActiveDataLanes(VMIVRegType vmiType,
                                                int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart)) {
    return failure();
  }

  int64_t remaining = vmiType.getElementCount() - chunk * *lanesPerPart;
  return std::clamp<int64_t>(remaining, 0, *lanesPerPart);
}

FailureOr<int64_t> getActiveDataLanesInPhysicalChunk(VMIVRegType vmiType,
                                                     int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart)) {
    return failure();
  }

  int64_t active = 0;
  for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
    FailureOr<bool> padding = isPaddingLane(vmiType, /*part=*/0, chunk, lane);
    if (failed(padding)) {
      return failure();
    }
    if (!*padding) {
      ++active;
    }
  }
  return active;
}

static FailureOr<std::pair<int64_t, int64_t>> getContiguousStoreMaskLaneCounts(
    VMIVRegType vmiType, int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart)) {
    return failure();
  }
  FailureOr<int64_t> activeLanes = getContiguousActiveDataLanes(vmiType, chunk);
  if (failed(activeLanes)) {
    return failure();
  }
  return std::make_pair(*lanesPerPart, *activeLanes);
}

FailureOr<Value> createContiguousStoreMask(Location loc, VMIVRegType vmiType,
                                           int64_t chunk, VRegType vregType,
                                           PatternRewriter &rewriter) {
  FailureOr<std::pair<int64_t, int64_t>> laneCounts =
      getContiguousStoreMaskLaneCounts(vmiType, chunk);
  if (failed(laneCounts)) {
    return failure();
  }
  if (laneCounts->second == laneCounts->first) {
    return createAllTrueMaskForVReg(loc, vregType, rewriter);
  }

  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  if (failed(maskType)) {
    return failure();
  }
  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, *maskType, createI32Constant(loc, laneCounts->second, rewriter),
      rewriter);
  if (failed(maskAndRemaining)) {
    return failure();
  }
  return maskAndRemaining->first;
}

FailureOr<Value> createMaskedStorePredicate(Location loc, VMIVRegType vmiType,
                                            int64_t chunk, Value userMask,
                                            VRegType vregType,
                                            PatternRewriter &rewriter) {
  FailureOr<std::pair<int64_t, int64_t>> laneCounts =
      getContiguousStoreMaskLaneCounts(vmiType, chunk);
  if (failed(laneCounts)) {
    return failure();
  }
  if (laneCounts->second == laneCounts->first) {
    return userMask;
  }

  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType) {
    return failure();
  }
  FailureOr<Value> tailMask =
      createContiguousStoreMask(loc, vmiType, chunk, vregType, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  bool failedTailMaskMaterialization = failed(tailMask) || failed(allTrue);
  if (failedTailMaskMaterialization) {
    return failure();
  }
  return rewriter.create<PandOp>(loc, maskType, userMask, *tailMask, *allTrue)
      .getResult();
}


