// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOMaskInternals.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

static FailureOr<Value> compactDenseLaneStrideStorePredicate(
    Location loc, Value userMask, VMILayoutAttr layout, StringRef targetGranularity,
    PatternRewriter &rewriter) {
  auto sourceMaskType = dyn_cast<MaskType>(userMask.getType());
  if (!sourceMaskType || !layout) {
    return failure();
  }
  auto targetMaskType = MaskType::get(rewriter.getContext(), targetGranularity);
  Value compactMask = userMask;
  StringRef sourceGranularity = sourceMaskType.getGranularity();
  StringAttr lower = rewriter.getStringAttr("LOWER");
  if (sourceGranularity == targetGranularity) {
    return compactMask;
  }
  bool unpackLaneStride2 = layout.getLaneStride() == 2;
  if (unpackLaneStride2) {
    Value unpacked = rewriter
                         .create<PunpackOp>(loc, targetMaskType, compactMask,
                                            lower)
                         .getResult();
    return unpacked;
  }
  bool supportsLaneStride4 = layout.getLaneStride() == 4 &&
                             sourceGranularity == "b8" &&
                             targetGranularity == "b32";
  if (!supportsLaneStride4) {
    return failure();
  }
  auto b16MaskType = MaskType::get(rewriter.getContext(), "b16");
  compactMask = rewriter
                    .create<PunpackOp>(loc, b16MaskType, compactMask, lower)
                    .getResult();
  return rewriter
      .create<PunpackOp>(loc, targetMaskType, compactMask, lower)
      .getResult();
}

FailureOr<Value> createDenseLaneStrideStorePredicate(
    Location loc, VMIVRegType vmiType, int64_t chunk, Value userMask,
    StringRef targetGranularity, PatternRewriter &rewriter) {
  VMILayoutAttr layout = vmiType.getLayoutAttr();
  FailureOr<Value> compactMask = compactDenseLaneStrideStorePredicate(
      loc, userMask, layout, targetGranularity, rewriter);
  if (failed(compactMask)) {
    return failure();
  }

  FailureOr<int64_t> activeLanes =
      getActiveDataLanesInPhysicalChunk(vmiType, chunk);
  FailureOr<int64_t> maskLanes = getMaskLanesPerPart(targetGranularity);
  bool failedLaneCountQuery = failed(activeLanes) || failed(maskLanes);
  if (failedLaneCountQuery) {
    return failure();
  }
  if (*activeLanes == *maskLanes) {
    return *compactMask;
  }

  auto targetMaskType = MaskType::get(rewriter.getContext(), targetGranularity);
  FailureOr<Value> tailMask = createPrefixMaskForActiveLanes(
      loc, targetMaskType, *activeLanes, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, targetMaskType, rewriter);
  bool failedTailMaskMaterialization = failed(tailMask) || failed(allTrue);
  if (failedTailMaskMaterialization) {
    return failure();
  }
  return rewriter
      .create<PandOp>(loc, targetMaskType, *compactMask, *tailMask, *allTrue)
      .getResult();
}

static FailureOr<std::optional<VMIPhysicalLane>> getShuffleSourcePhysicalLane(
    VMIVRegType sourceType, VMIVRegType resultType,
    ArrayRef<int64_t> indices, int64_t resultPart, int64_t resultChunk,
    int64_t lane, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<std::optional<VMIPhysicalLane>> {
    if (reason) {
      *reason = message.str();
    }
    return std::optional<VMIPhysicalLane>();
  };
  FailureOr<bool> padding =
      isPaddingLane(resultType, resultPart, resultChunk, lane);
  if (failed(padding)) {
    return fail("failed to classify result padding lanes");
  }
  if (*padding) {
    return std::optional<VMIPhysicalLane>();
  }
  FailureOr<int64_t> logicalLane =
      mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
  bool logicalLaneOutOfRange =
      failed(logicalLane) ||
      *logicalLane >= static_cast<int64_t>(indices.size());
  if (logicalLaneOutOfRange) {
    return fail("failed to map result lane");
  }
  FailureOr<VMIPhysicalLane> sourcePhysical =
      mapLogicalLaneToPhysical(sourceType, indices[*logicalLane]);
  if (failed(sourcePhysical)) {
    return fail("failed to map source lane");
  }
  if (sourcePhysical->lane != lane) {
    return fail("requires same-lane physical chunks");
  }
  return std::optional<VMIPhysicalLane>(*sourcePhysical);
}

static FailureOr<int64_t> computeShuffleForwardingSourceChunk(
    VMIVRegType sourceType, VMIVRegType resultType, ArrayRef<int64_t> indices,
    int64_t resultPart, int64_t resultChunk, int64_t lanesPerPart,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<int64_t> sourcePart;
  std::optional<int64_t> sourceChunk;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<std::optional<VMIPhysicalLane>> sourcePhysical =
        getShuffleSourcePhysicalLane(
        sourceType, resultType, indices, resultPart, resultChunk, lane,
        reason);
    if (failed(sourcePhysical)) {
      return failure();
    }
    if (!*sourcePhysical) {
      continue;
    }
    if (!sourcePart) {
      sourcePart = (*sourcePhysical)->part;
      sourceChunk = (*sourcePhysical)->chunk;
    } else if (*sourcePart != (*sourcePhysical)->part ||
               *sourceChunk != (*sourcePhysical)->chunk) {
      return fail("requires one source chunk per result chunk");
    }
  }
  if (!sourcePart || !sourceChunk) {
    return fail("requires at least one logical lane per result chunk");
  }
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, *sourcePart, *sourceChunk);
  if (failed(sourceFlatIndex)) {
    return fail("source part range is out of bounds");
  }
  return *sourceFlatIndex;
}

struct ShuffleForwardingInputPlan {
  VMIVRegType sourceType;
  VMIVRegType resultType;
  ArrayRef<int64_t> indices;
  int64_t lanesPerPart;
  int64_t resultFactor;
};

static FailureOr<ShuffleForwardingInputPlan>
getShuffleForwardingInputPlan(VMIShuffleOp op, std::string *reason) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (op.getIndices().empty()) {
    if (reason) {
      *reason = "requires non-empty indices";
    }
    return failure();
  }
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    if (reason) {
      *reason = "requires known lanes per physical part";
    }
    return failure();
  }
  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor)) {
    if (reason) {
      *reason = "requires assigned result layout";
    }
    return failure();
  }
  return ShuffleForwardingInputPlan{sourceType, resultType, op.getIndices(),
                                    *lanesPerPart, *resultFactor};
}

static FailureOr<SmallVector<int64_t>> materializeShuffleForwardingSourceParts(
    const ShuffleForwardingInputPlan &input, std::string *reason) {
  SmallVector<int64_t> sourceFlatIndices;
  for (int64_t resultPart = 0; resultPart < input.resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getDataChunksInPart(input.resultType, resultPart);
    if (failed(resultChunks)) {
      if (reason) {
        *reason = "requires known result physical chunks";
      }
      return failure();
    }
    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      FailureOr<int64_t> sourceFlatIndex =
          computeShuffleForwardingSourceChunk(
              input.sourceType, input.resultType, input.indices, resultPart,
              resultChunk, input.lanesPerPart, reason);
      if (failed(sourceFlatIndex)) {
        return failure();
      }
      sourceFlatIndices.push_back(*sourceFlatIndex);
    }
  }

  return sourceFlatIndices;
}

FailureOr<SmallVector<int64_t>>
computeShuffleForwardingSourceParts(VMIShuffleOp op, std::string *reason) {
  FailureOr<ShuffleForwardingInputPlan> input =
      getShuffleForwardingInputPlan(op, reason);
  if (failed(input)) {
    return failure();
  }
  return materializeShuffleForwardingSourceParts(*input, reason);
}

struct ShuffleVselrPlan {
  int64_t sourceFlatIndex = 0;
  int64_t baseLane = 0;
  bool descending = false;
};

static FailureOr<bool> getShuffleLaneDirection(
    int64_t baseLane, int64_t resultLane, int64_t sourceLane,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<bool> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool ascending = sourceLane == baseLane + resultLane;
  bool descending = sourceLane == baseLane - resultLane;
  bool unsupportedDirection = !ascending && !descending;
  if (unsupportedDirection) {
    return fail("requires ASC or DESC affine source lane indices");
  }
  return descending && !ascending;
}

static FailureOr<VMIPhysicalLane> getShuffleSourceLane(
    VMIVRegType sourceType, VMIVRegType resultType,
    ArrayRef<int64_t> indices, int64_t resultPart, int64_t resultChunk,
    int64_t lane, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<VMIPhysicalLane> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<bool> padding =
      isPaddingLane(resultType, resultPart, resultChunk, lane);
  bool invalidPadding = failed(padding) || *padding;
  if (invalidPadding) {
    return fail("requires full physical result chunks");
  }
  FailureOr<int64_t> logicalLane =
      mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
  bool logicalLaneOutOfRange =
      succeeded(logicalLane) &&
      *logicalLane >= static_cast<int64_t>(indices.size());
  bool invalidLogicalLane = failed(logicalLane) || logicalLaneOutOfRange;
  if (invalidLogicalLane) {
    return fail("failed to map result lane");
  }
  FailureOr<VMIPhysicalLane> sourceLane =
      mapLogicalLaneToPhysical(sourceType, indices[*logicalLane]);
  if (failed(sourceLane)) {
    return fail("failed to map source lane");
  }
  return *sourceLane;
}

struct ShuffleChunkLaneState {
  int64_t sourcePart = 0;
  int64_t sourceChunk = 0;
  int64_t baseLane = 0;
  std::optional<bool> descending;
};

static FailureOr<ShuffleChunkLaneState> updateShuffleChunkLaneState(
    VMIVRegType sourceType, VMIVRegType resultType, ArrayRef<int64_t> indices,
    int64_t resultPart, int64_t resultChunk, int64_t lane,
    std::optional<ShuffleChunkLaneState> state,
    std::string *reason) {
  FailureOr<VMIPhysicalLane> sourcePhysical = getShuffleSourceLane(
      sourceType, resultType, indices, resultPart, resultChunk, lane, reason);
  if (failed(sourcePhysical)) {
    return failure();
  }
  if (!state) {
    return ShuffleChunkLaneState{sourcePhysical->part, sourcePhysical->chunk,
                                 sourcePhysical->lane, std::nullopt};
  }
  bool sourceChunkMismatch = state->sourcePart != sourcePhysical->part ||
                             state->sourceChunk != sourcePhysical->chunk;
  if (sourceChunkMismatch) {
    if (reason) {
      *reason = "requires one source chunk per result chunk";
    }
    return failure();
  }
  FailureOr<bool> laneDescending = getShuffleLaneDirection(
      state->baseLane, lane, sourcePhysical->lane, reason);
  if (failed(laneDescending)) {
    return failure();
  }
  if (state->descending && *laneDescending != *state->descending) {
    if (reason) {
      *reason = "requires one index order per result chunk";
    }
    return failure();
  }
  state->descending = *laneDescending;
  return *state;
}

FailureOr<ShuffleVselrPlan> computeShuffleVselrPlanForChunk(
    VMIVRegType sourceType, VMIVRegType resultType, ArrayRef<int64_t> indices,
    int64_t resultPart, int64_t resultChunk, int64_t lanesPerPart,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<ShuffleVselrPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<ShuffleChunkLaneState> state;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<ShuffleChunkLaneState> nextState = updateShuffleChunkLaneState(
        sourceType, resultType, indices, resultPart, resultChunk, lane,
        state, reason);
    if (failed(nextState)) {
      return failure();
    }
    state = *nextState;
  }
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, state->sourcePart, state->sourceChunk);
  if (failed(sourceFlatIndex)) {
    return fail("source part range is out of bounds");
  }
  return ShuffleVselrPlan{*sourceFlatIndex, state->baseLane,
                          state->descending.value_or(false)};
}

static FailureOr<int64_t> getShuffleResultChunkCount(
    VMIVRegType resultType, int64_t resultPart, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> resultChunks =
      getDataChunksInPart(resultType, resultPart);
  if (failed(resultChunks)) {
    return fail("requires known result physical chunks");
  }
  return *resultChunks;
}

FailureOr<int64_t> computeShuffleLane0SplatSourcePart(VMIShuffleOp op,
                                                      std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty()) {
    return fail("requires non-empty indices");
  }
  bool hasNonZeroIndex =
      !llvm::all_of(indices, [](int64_t index) { return index == 0; });
  if (hasNonZeroIndex) {
    return fail("requires every result lane to select source lane 0");
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<VMIPhysicalLane> sourceLane =
      mapLogicalLaneToPhysical(sourceType, 0);
  if (failed(sourceLane)) {
    return fail("failed to map source lane 0");
  }
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, sourceLane->part, sourceLane->chunk);
  if (failed(sourceFlatIndex)) {
    return fail("source lane 0 part range is out of bounds");
  }
  return *sourceFlatIndex;
}

struct ShuffleVselrInputPlan {
  VMIVRegType sourceType;
  VMIVRegType resultType;
  ArrayRef<int64_t> indices;
  int64_t lanesPerPart;
  int64_t resultFactor;
};

static FailureOr<ShuffleVselrInputPlan> buildShuffleVselrInputPlan(
    VMIShuffleOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<ShuffleVselrInputPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    return fail("requires known lanes per physical part");
  }
  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty()) {
    return fail("requires non-empty indices");
  }
  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor)) {
    return fail("requires assigned result layout");
  }
  return ShuffleVselrInputPlan{sourceType, resultType, indices,
                                *lanesPerPart, *resultFactor};
}

static FailureOr<SmallVector<ShuffleVselrPlan>> materializeShuffleVselrPlans(
    const ShuffleVselrInputPlan &input, std::string *reason) {
  SmallVector<ShuffleVselrPlan> plans;
  for (int64_t resultPart = 0; resultPart < input.resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getShuffleResultChunkCount(input.resultType, resultPart, reason);
    if (failed(resultChunks)) {
      return failure();
    }
    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      FailureOr<ShuffleVselrPlan> plan = computeShuffleVselrPlanForChunk(
          input.sourceType, input.resultType, input.indices, resultPart,
          resultChunk, input.lanesPerPart, reason);
      if (failed(plan)) {
        return failure();
      }
      plans.push_back(*plan);
    }
  }
  return plans;
}

FailureOr<SmallVector<ShuffleVselrPlan>>
computeShuffleVselrPlans(VMIShuffleOp op, std::string *reason) {
  FailureOr<ShuffleVselrInputPlan> input =
      buildShuffleVselrInputPlan(op, reason);
  if (failed(input)) {
    return failure();
  }
  return materializeShuffleVselrPlans(*input, reason);
}

struct ConstantMaskChunkMaterialization {
  SmallVector<int8_t> activeLanes;
};

template <typename LanePredicate>
FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
materializeMaskChunks(VMIMaskType resultVMIType, int64_t lanesPerPart,
                      LanePredicate isActive, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<SmallVector<ConstantMaskChunkMaterialization>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  SmallVector<ConstantMaskChunkMaterialization> materializations;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      ConstantMaskChunkMaterialization materialization;
      materialization.activeLanes.reserve(lanesPerPart);
      for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding)) {
          return fail("failed to map physical padding lane");
        }
        if (*padding) {
          materialization.activeLanes.push_back(0);
          continue;
        }
        anyLane = true;
        FailureOr<int64_t> logicalLane =
            mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
        if (failed(logicalLane)) {
          return fail("failed to map physical lane");
        }
        materialization.activeLanes.push_back(isActive(*logicalLane) ? 1 : 0);
      }
      if (!anyLane) {
        break;
      }
      materializations.push_back(std::move(materialization));
    }
  }
  return materializations;
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeConstantMaskMaterialization(VMIConstantMaskOp op, std::string *reason) {
  auto denseAttr = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!denseAttr) {
    return emitFailure<SmallVector<ConstantMaskChunkMaterialization>>(
        reason, "only dense integer mask constants are supported");
  }

  auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout ||
      !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity())) {
    return emitFailure<SmallVector<ConstantMaskChunkMaterialization>>(
        reason, "requires concrete layout and granularity");
  }

  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  if (failed(lanesPerPart)) {
    return emitFailure<SmallVector<ConstantMaskChunkMaterialization>>(
        reason, "requires known physical mask lanes per part");
  }

  auto boolValues = denseAttr.getValues<bool>();
  return materializeMaskChunks(
      resultVMIType, *lanesPerPart,
      [&boolValues](int64_t logicalLane) { return boolValues[logicalLane]; },
      reason);
}

struct GroupMaskMaterializationPlan {
  int64_t lanesPerPart;
  int64_t groupSize;
  int64_t activeElems;
};

static FailureOr<GroupMaskMaterializationPlan>
buildGroupMaskMaterializationPlan(VMICreateGroupMaskOp op,
                                  VMIMaskType resultVMIType,
                                  std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<GroupMaskMaterializationPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto activeConstant =
      op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
  if (!activeConstant) {
    return fail("requires constant active_elems_per_group");
  }
  auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
  if (!activeAttr) {
    return fail("active_elems_per_group must be an integer constant");
  }
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  bool invalidMaskType =
      !layout || !VMIMaskType::isConcreteGranularity(
                     resultVMIType.getGranularity());
  if (invalidMaskType) {
    return fail("requires concrete layout and granularity");
  }
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  if (failed(lanesPerPart)) {
    return fail("requires known physical mask lanes per part");
  }
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  bool invalidShape =
      numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize;
  if (invalidShape) {
    return fail("requires result lane count to match num_groups * group_size");
  }
  int64_t activeElems = std::clamp<int64_t>(activeAttr.getInt(), 0, groupSize);
  return GroupMaskMaterializationPlan{*lanesPerPart, groupSize, activeElems};
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterializationForType(VMICreateGroupMaskOp op,
                                       VMIMaskType resultVMIType,
                                       std::string *reason) {
  FailureOr<GroupMaskMaterializationPlan> plan =
      buildGroupMaskMaterializationPlan(op, resultVMIType, reason);
  if (failed(plan)) {
    return failure();
  }

  return materializeMaskChunks(
      resultVMIType, plan->lanesPerPart,
      [groupSize = plan->groupSize,
       activeElems = plan->activeElems](int64_t logicalLane) {
        return logicalLane % groupSize < activeElems;
      },
      reason);
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterialization(VMICreateGroupMaskOp op, std::string *reason) {
  return computeGroupMaskMaterializationForType(
      op, cast<VMIMaskType>(op.getResult().getType()), reason);
}

FailureOr<Value> materializeConstantMaskChunk(Location loc, MaskType maskType,
                                              ArrayRef<int8_t> activeLanes,
                                              PatternRewriter &rewriter);

FailureOr<Value> createPowerOfTwoRemainder(Location loc, Value value,
                                           int64_t modulus, Value allMask,
                                           PatternRewriter &rewriter) {
  if (modulus <= 0) {
    return failure();
  }

  auto vectorType = dyn_cast<VRegType>(value.getType());
  if (!vectorType) {
    return failure();
  }

  std::optional<int64_t> shift = getPowerOfTwoLog2(modulus);
  if (!shift) {
    return failure();
  }
  if (*shift == 0) {
    Value zero = createI32Constant(loc, 0, rewriter);
    return rewriter.create<VdupOp>(loc, vectorType, zero, allMask,
                                   /*position=*/nullptr)
        .getResult();
  }

  Value shiftScalar = createI16Constant(loc, *shift, rewriter);
  Value quotient =
      rewriter.create<VshrsOp>(loc, vectorType, value, shiftScalar, allMask)
          .getResult();
  Value base =
      rewriter.create<VshlsOp>(loc, vectorType, quotient, shiftScalar, allMask)
          .getResult();
  return rewriter.create<VsubOp>(loc, vectorType, value, base, allMask)
      .getResult();
}

static FailureOr<Value> applyGroupMaskPadding(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType, MaskType maskType,
    Value predicate, int64_t part, int64_t chunk, int64_t lanesPerPart,
    Value allMask, PatternRewriter &rewriter) {
  SmallVector<int8_t> validLanes;
  validLanes.reserve(lanesPerPart);
  bool hasPadding = false;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<bool> padding =
        isPaddingLane(resultVMIType, part, chunk, lane);
    if (failed(padding)) {
      return rewriter.notifyMatchFailure(
          op, "failed to classify dynamic create_group_mask padding");
    }
    validLanes.push_back(*padding ? 0 : 1);
    hasPadding |= *padding;
  }
  if (!hasPadding) {
    return predicate;
  }
  FailureOr<Value> validMask = materializeConstantMaskChunk(
      op.getLoc(), maskType, validLanes, rewriter);
  if (failed(validMask)) {
    return rewriter.notifyMatchFailure(
        op, "failed to materialize dynamic create_group_mask padding mask");
  }
  return rewriter
      .create<PandOp>(op.getLoc(), maskType, predicate, *validMask, allMask)
      .getResult();
}

struct DynamicGroupMaskBlockIndex {
  Value partBlock;
  Value inBlockLane;
  std::optional<int64_t> blockShift;
};

static FailureOr<DynamicGroupMaskBlockIndex> buildDynamicGroupMaskBlockIndex(
    VMICreateGroupMaskOp op, int64_t blockElems, int64_t chunk,
    int64_t lanesPerPart, Value allMask, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  Type i32 = rewriter.getI32Type();
  auto indexVectorType = VRegType::get(ctx, lanesPerPart, i32);
  Value chunkBase = createI32Constant(loc, chunk * lanesPerPart, rewriter);
  Value indexInPart =
      rewriter.create<VciOp>(loc, indexVectorType, chunkBase, StringAttr{})
          .getResult();
  Value partBlock = indexInPart;
  Value inBlockLane = createI32Constant(loc, 0, rewriter);
  std::optional<int64_t> blockShiftValue = getPowerOfTwoLog2(blockElems);
  if (blockElems != 1) {
    if (!blockShiftValue) {
      return rewriter.notifyMatchFailure(
          op, "dynamic create_group_mask block size must be a power of two");
    }
    Value blockShift = createI16Constant(loc, *blockShiftValue, rewriter);
    partBlock = rewriter
                    .create<VshrsOp>(loc, indexVectorType, indexInPart,
                                     blockShift, allMask)
                    .getResult();
    Value blockBase = rewriter
                          .create<VshlsOp>(loc, indexVectorType, partBlock,
                                           blockShift, allMask)
                          .getResult();
    inBlockLane = rewriter
                      .create<VsubOp>(loc, indexVectorType, indexInPart,
                                      blockBase, allMask)
                      .getResult();
  }
  return DynamicGroupMaskBlockIndex{partBlock, inBlockLane, blockShiftValue};
}

static Value buildDynamicGroupMaskLogicalLane(
    VMICreateGroupMaskOp op, int64_t factor, int64_t blockElems, int64_t part,
    const DynamicGroupMaskBlockIndex &blockIndex, Value allMask,
    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto indexVectorType = cast<VRegType>(blockIndex.partBlock.getType());
  Value factorScalar = createI32Constant(loc, factor, rewriter);
  Value logicalBlock = rewriter
                           .create<VmulsOp>(loc, indexVectorType,
                                            blockIndex.partBlock,
                                            factorScalar, allMask)
                           .getResult();
  if (part != 0) {
    Value partScalar = createI32Constant(loc, part, rewriter);
    logicalBlock = rewriter
                       .create<VaddsOp>(loc, indexVectorType, logicalBlock,
                                        partScalar, allMask)
                       .getResult();
  }
  Value logicalLane = logicalBlock;
  if (blockElems != 1) {
    Value blockShift = createI16Constant(loc, *blockIndex.blockShift, rewriter);
    Value logicalBlockBase = rewriter
                                 .create<VshlsOp>(loc, indexVectorType,
                                                  logicalBlock, blockShift,
                                                  allMask)
                                 .getResult();
    logicalLane = rewriter
                      .create<VaddOp>(loc, indexVectorType, logicalBlockBase,
                                      blockIndex.inBlockLane, allMask)
                      .getResult();
  }
  return logicalLane;
}

static FailureOr<Value> buildDynamicGroupMaskLaneIndex(
    VMICreateGroupMaskOp op, int64_t factor, int64_t blockElems, int64_t part,
    int64_t chunk, int64_t lanesPerPart, Value allMask,
    PatternRewriter &rewriter) {
  FailureOr<DynamicGroupMaskBlockIndex> blockIndex =
      buildDynamicGroupMaskBlockIndex(op, blockElems, chunk, lanesPerPart,
                                      allMask, rewriter);
  if (failed(blockIndex)) {
    return failure();
  }
  return buildDynamicGroupMaskLogicalLane(op, factor, blockElems, part,
                                          *blockIndex, allMask, rewriter);
}

FailureOr<Value> materializeDynamicGroupMaskChunk(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType, Type resultType,
    Value activeI32, int64_t factor, int64_t blockElems, int64_t part,
    int64_t chunk, int64_t lanesPerPart, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<Value> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  auto maskType = dyn_cast<MaskType>(resultType);
  if (!maskType || !maskType.isB32()) {
    return fail("dynamic create_group_mask result must be b32 mask");
  }
  Location loc = op.getLoc();
  FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
  if (failed(allMask)) {
    return fail("failed to create dynamic create_group_mask all mask");
  }
  FailureOr<Value> logicalLane = buildDynamicGroupMaskLaneIndex(
      op, factor, blockElems, part, chunk, lanesPerPart, *allMask, rewriter);
  if (failed(logicalLane)) {
    return fail("failed to compute dynamic create_group_mask lane index");
  }
  FailureOr<Value> laneInGroup = createPowerOfTwoRemainder(
      loc, *logicalLane, op.getGroupSizeAttr().getInt(), *allMask, rewriter);
  if (failed(laneInGroup)) {
    return fail("failed to compute dynamic create_group_mask lane index");
  }
  Value predicate =
      rewriter
          .create<VcmpsOp>(loc, maskType, *laneInGroup, activeI32, *allMask,
                           rewriter.getStringAttr("lt"))
          .getResult();
  FailureOr<Value> paddedPredicate = applyGroupMaskPadding(
      op, resultVMIType, maskType, predicate, part, chunk, lanesPerPart,
      *allMask, rewriter);
  if (failed(paddedPredicate)) {
    return failure();
  }
  return *paddedPredicate;
}

struct DynamicGroupMaskPlan {
  VMILayoutAttr layout;
  int64_t factor;
  int64_t blockElems;
  int64_t lanesPerPart;
  int64_t arity;
};

static LogicalResult checkDynamicGroupMaskLayout(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType,
    VMILayoutAttr *layout, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  *layout = resultVMIType.getLayoutAttr();
  if (!*layout) {
    return fail("dynamic create_group_mask requires assigned layout");
  }
  bool unsupportedLaneStride = layout->getLaneStride() != 1;
  if (unsupportedLaneStride) {
    return fail("dynamic create_group_mask requires lane_stride=1 layout");
  }
  bool unsupportedMaskGranularity = resultVMIType.getGranularity() != "b32";
  if (unsupportedMaskGranularity) {
    return fail("dynamic create_group_mask currently requires b32 granularity");
  }
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  bool invalidLogicalShape =
      numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize;
  if (invalidLogicalShape) {
    return fail("dynamic create_group_mask requires result lane count to match "
                "num_groups * group_size");
  }
  if (!getPowerOfTwoLog2(groupSize)) {
    return fail("dynamic create_group_mask currently requires power-of-two group_size");
  }
  return success();
}

static FailureOr<std::pair<int64_t, int64_t>>
getDynamicGroupMaskPhysicalShape(VMIMaskType resultVMIType,
                                 TypeRange resultTypes,
                                 std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<std::pair<int64_t, int64_t>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultVMIType);
  bool missingPhysicalShape = failed(lanesPerPart) || failed(arity) || *arity < 1;
  if (missingPhysicalShape) {
    return fail("dynamic create_group_mask requires computable physical mask chunks");
  }
  bool resultArityMismatch = static_cast<int64_t>(resultTypes.size()) != *arity;
  if (resultArityMismatch) {
    return fail("dynamic create_group_mask physical result count mismatch");
  }
  return std::make_pair(*lanesPerPart, *arity);
}

static FailureOr<DynamicGroupMaskPlan> buildDynamicGroupMaskPlan(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType, TypeRange resultTypes,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<DynamicGroupMaskPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr layout;
  if (failed(checkDynamicGroupMaskLayout(op, resultVMIType, &layout, reason))) {
    return failure();
  }
  FailureOr<std::pair<int64_t, int64_t>> physicalShape =
      getDynamicGroupMaskPhysicalShape(resultVMIType, resultTypes, reason);
  if (failed(physicalShape)) {
    return failure();
  }
  int64_t lanesPerPart = physicalShape->first;
  int64_t arity = physicalShape->second;
  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  FailureOr<int64_t> blockElems = getVMILayoutBlockElems(resultVMIType);
  bool invalidResultCount =
      factor <= 0 || failed(blockElems) || *blockElems <= 0 ||
      static_cast<int64_t>(resultTypes.size()) % factor != 0;
  if (invalidResultCount) {
    return fail("dynamic create_group_mask physical result count does not match layout factor");
  }
  if (!getPowerOfTwoLog2(*blockElems)) {
    return fail("dynamic create_group_mask requires a power-of-two physical block element count");
  }
  return DynamicGroupMaskPlan{layout, factor, *blockElems, lanesPerPart, arity};
}

static FailureOr<SmallVector<Value>> materializeDynamicGroupMaskChunks(
    VMICreateGroupMaskOp op, Value activeI32, VMIMaskType resultVMIType,
    TypeRange resultTypes, int64_t factor, int64_t blockElems,
    int64_t lanesPerPart, PatternRewriter &rewriter) {
  if (factor <= 0) {
    return rewriter.notifyMatchFailure(
        op, "dynamic group mask requires positive layout factor");
  }
  int64_t safeFactor = factor;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  int64_t chunksPerPart = resultTypes.size() / safeFactor;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
      FailureOr<Value> predicate = materializeDynamicGroupMaskChunk(
          op, resultVMIType, resultTypes[part * chunksPerPart + chunk],
          activeI32, factor, blockElems, part, chunk, lanesPerPart, rewriter);
      if (failed(predicate)) {
        return failure();
      }
      results.push_back(*predicate);
    }
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeDynamicGroupMaskForType(
    VMICreateGroupMaskOp op, Value activeElemsPerGroup,
    VMIMaskType resultVMIType, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  FailureOr<DynamicGroupMaskPlan> plan =
      buildDynamicGroupMaskPlan(op, resultVMIType, resultTypes, nullptr);
  if (failed(plan)) {
    return failure();
  }
  Value activeI32 =
      clampDynamicActiveLanes(loc, activeElemsPerGroup,
                              op.getGroupSizeAttr().getInt(), rewriter);

  return materializeDynamicGroupMaskChunks(
      op, activeI32, resultVMIType, resultTypes, plan->factor,
      plan->blockElems, plan->lanesPerPart, rewriter);
}

std::optional<int64_t> getPrefixActiveLaneCount(ArrayRef<int8_t> activeLanes) {
  bool seenInactive = false;
  int64_t activeCount = 0;
  for (int8_t active : activeLanes) {
    if (active) {
      if (seenInactive) {
        return std::nullopt;
      }
      ++activeCount;
      continue;
    }
    seenInactive = true;
  }
  return activeCount;
}

FailureOr<Value> materializePrefixMask(Location loc, MaskType maskType,
                                       int64_t activeLanes,
                                       int64_t lanesPerPart,
                                       PatternRewriter &rewriter) {
  std::optional<std::string> pattern =
      getPrefixPattern(activeLanes, lanesPerPart);
  if (pattern) {
    return createPatternMask(loc, maskType, *pattern, rewriter);
  }

  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
  if (failed(maskAndRemaining)) {
    return failure();
  }
  return maskAndRemaining->first;
}

static FailureOr<int64_t> validateConstantMaskChunk(
    MaskType maskType, ArrayRef<int8_t> activeLanes) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  bool invalidShape = failed(lanesPerPart) ||
                      static_cast<int64_t>(activeLanes.size()) != *lanesPerPart;
  if (invalidShape) {
    return failure();
  }
  return *lanesPerPart;
}

static FailureOr<Value> materializeNonPrefixConstantMask(
    Location loc, MaskType maskType, ArrayRef<int8_t> activeLanes,
    int64_t lanesPerPart, Value allTrue, PatternRewriter &rewriter) {
  Value result;
  for (int64_t lane = 0; lane < lanesPerPart;) {
    while (lane < lanesPerPart && !activeLanes[lane]) {
      ++lane;
    }
    if (lane >= lanesPerPart) {
      break;
    }
    int64_t runBegin = lane;
    while (lane < lanesPerPart && activeLanes[lane]) {
      ++lane;
    }
    int64_t runEnd = lane;
    FailureOr<Value> prefixEnd =
        materializePrefixMask(loc, maskType, runEnd, lanesPerPart, rewriter);
    if (failed(prefixEnd)) {
      return failure();
    }
    Value runMask = *prefixEnd;
    if (runBegin != 0) {
      FailureOr<Value> prefixBegin = materializePrefixMask(
          loc, maskType, runBegin, lanesPerPart, rewriter);
      if (failed(prefixBegin)) {
        return failure();
      }
      Value notPrefixBegin =
          rewriter.create<PnotOp>(loc, maskType, *prefixBegin, allTrue)
              .getResult();
      runMask = rewriter
                    .create<PandOp>(loc, maskType, *prefixEnd, notPrefixBegin,
                                    allTrue)
                    .getResult();
    }
    if (!result) {
      result = runMask;
    } else {
      result = rewriter.create<PorOp>(loc, maskType, result, runMask, allTrue)
                   .getResult();
    }
  }
  return result;
}

FailureOr<Value> materializeConstantMaskChunk(Location loc, MaskType maskType,
                                              ArrayRef<int8_t> activeLanes,
                                              PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      validateConstantMaskChunk(maskType, activeLanes);
  if (failed(lanesPerPart)) {
    return failure();
  }

  if (std::optional<int64_t> prefixCount =
          getPrefixActiveLaneCount(activeLanes)) {
    return materializePrefixMask(loc, maskType, *prefixCount, *lanesPerPart,
                                 rewriter);
  }

  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  if (failed(allTrue)) {
    return failure();
  }

  FailureOr<Value> result = materializeNonPrefixConstantMask(
      loc, maskType, activeLanes, *lanesPerPart, *allTrue, rewriter);
  if (failed(result)) {
    return failure();
  }

  if (*result) {
    return *result;
  }
  return materializePrefixMask(loc, maskType, 0, *lanesPerPart, rewriter);
}

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter);

Value createChunkOffset(Location loc, Value baseOffset, int64_t laneOffset,
                        PatternRewriter &rewriter) {
  if (laneOffset == 0) {
    return baseOffset;
  }
  Value delta = rewriter.create<arith::ConstantIndexOp>(loc, laneOffset);
  return rewriter.create<arith::AddIOp>(loc, baseOffset, delta).getResult();
}

Value createGroupChunkOffset(Location loc, Value baseOffset, Value rowStride,
                             int64_t group, int64_t inGroupLaneOffset,
                             PatternRewriter &rewriter) {
  Value offset = baseOffset;
  if (group != 0) {
    Value groupIndex = rewriter.create<arith::ConstantIndexOp>(loc, group);
    Value rowOffset =
        rewriter.create<arith::MulIOp>(loc, rowStride, groupIndex).getResult();
    offset = rewriter.create<arith::AddIOp>(loc, offset, rowOffset).getResult();
  }
  return createChunkOffset(loc, offset, inGroupLaneOffset, rewriter);
}

LogicalResult checkContiguousFullGroupChunks(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t *lanesPerPart,
    int64_t *groupCount, int64_t *chunksPerGroup, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return fail("group op requires contiguous VMI layout");
  }
  if (failed(checkFullDataPhysicalChunks(type, nullptr))) {
    return fail("group op requires full physical chunks");
  }
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes)) {
    return fail("group op requires known physical lanes per part");
  }
  if (groupSize <= 0) {
    return fail("group op requires positive derived group size");
  }
  int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
  bool unevenLogicalGroups = type.getElementCount() % safeGroupSize != 0;
  if (unevenLogicalGroups) {
    return fail("group op requires derived group size to evenly divide lane "
                "count");
  }
  if (*lanes <= 0) {
    return fail("group op requires positive physical lanes per part");
  }
  if (groupSize % *lanes != 0) {
    return fail("group op currently requires group size to be a multiple of "
                "physical lanes per part");
  }

  *lanesPerPart = *lanes;
  *groupCount = type.getElementCount() / groupSize;
  *chunksPerGroup = groupSize / *lanes;
  return success();
}

FailureOr<Value> createZeroVector(Location loc, VRegType type,
                                  PatternRewriter &rewriter) {
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, type.getElementType(), 0, rewriter);
  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, type, rewriter);
  bool failedMaterialization = failed(zero) || failed(mask);
  if (failedMaterialization) {
    return failure();
  }
  return rewriter
      .create<VdupOp>(loc, type, *zero, *mask,
                      /*position=*/nullptr)
      .getResult();
}

FailureOr<Value> createLaneRangeMask(Location loc, MaskType maskType,
                                     int64_t begin, int64_t end,
                                     PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  bool invalidRange = failed(lanesPerPart) || begin < 0 || begin > end ||
                      end > *lanesPerPart;
  if (invalidRange) {
    return failure();
  }
  SmallVector<int8_t> active(*lanesPerPart, 0);
  for (int64_t lane = begin; lane < end; ++lane) {
    active[lane] = 1;
  }
  return materializeConstantMaskChunk(loc, maskType, active, rewriter);
}

FailureOr<Value> createGroupSlotIndexVector(Location loc, VRegType indexType,
                                            int64_t groupSize,
                                            int64_t baseGroupSlot,
                                            PatternRewriter &rewriter,
                                            int64_t slotLaneStride = 1) {
  if (groupSize <= 0) {
    return failure();
  }
  int64_t lanesPerPart = indexType.getElementCount();
  FailureOr<Value> baseScalar = createScalarOffsetConstant(
      loc, indexType.getElementType(), baseGroupSlot * slotLaneStride,
      rewriter);
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(indexType, rewriter.getContext());
  FailureOr<Value> allMask = createAllTrueMaskForVReg(loc, indexType, rewriter);
  bool failedSeedMaterialization =
      failed(baseScalar) || failed(maskType) || failed(allMask);
  if (failedSeedMaterialization) {
    return failure();
  }
  Value result = rewriter
                     .create<VdupOp>(loc, indexType, *baseScalar, *allMask,
                                     /*position=*/nullptr)
                     .getResult();
  if (groupSize >= lanesPerPart) {
    return result;
  }
  if (lanesPerPart % groupSize != 0) {
    return failure();
  }

  int64_t groupsPerChunk = lanesPerPart / groupSize;
  for (int64_t localGroup = 1; localGroup < groupsPerChunk; ++localGroup) {
    FailureOr<Value> groupScalar = createScalarOffsetConstant(
        loc, indexType.getElementType(),
        (baseGroupSlot + localGroup) * slotLaneStride, rewriter);
    FailureOr<Value> laneMask =
        createLaneRangeMask(loc, *maskType, localGroup * groupSize,
                            (localGroup + 1) * groupSize, rewriter);
    bool failedGroupMaterialization = failed(groupScalar) || failed(laneMask);
    if (failedGroupMaterialization) {
      return failure();
    }
    Value splat = rewriter
                      .create<VdupOp>(loc, indexType, *groupScalar, *allMask,
                                      /*position=*/nullptr)
                      .getResult();
    result = rewriter.create<VselOp>(loc, indexType, splat, result, *laneMask)
                 .getResult();
  }
  return result;
}

std::optional<std::string> getX2MemoryDistToken(Type elementType,
                                                StringRef prefix) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return std::nullopt;
  }
  return (Twine(prefix) + "_B" + Twine(elementBits)).str();
}

std::optional<std::string> getDenseLaneStrideLoadDistToken(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  if (layout.getLaneStride() == 2 &&
      (elementBits == 8 || elementBits == 16 || elementBits == 32)) {
    return (Twine("UNPK_B") + Twine(elementBits)).str();
  }
  bool isUnpack4 = layout.getLaneStride() == 4 && elementBits == 8;
  if (isUnpack4) {
    return std::string("UNPK4");
  }
  return std::nullopt;
}

std::optional<std::string>
getLaneStrideStoreDistToken(VMILayoutAttr layout, Type elementType) {
  if (!layout || !layout.hasLaneStride()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  bool isPackB16 = layout.getLaneStride() == 2 && elementBits == 8;
  if (isPackB16) {
    return std::string("PK_B16");
  }
  bool isPackB32 = layout.getLaneStride() == 2 && elementBits == 16;
  if (isPackB32) {
    return std::string("PK_B32");
  }
  bool isPackB64 = layout.getLaneStride() == 2 && elementBits == 32;
  if (isPackB64) {
    return std::string("PK_B64");
  }
  bool isPack4B32 = layout.getLaneStride() == 4 && elementBits == 8;
  if (isPack4B32) {
    return std::string("PK4_B32");
  }
  return std::nullopt;
}

std::optional<std::string> getDenseLaneStrideStoreDistToken(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  return getLaneStrideStoreDistToken(layout, type.getElementType());
}

std::optional<StringRef>
getLaneStrideStoreMaskGranularity(VMILayoutAttr layout, Type elementType) {
  if (!layout || !layout.hasLaneStride()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  bool isB16Mask = layout.getLaneStride() == 2 && elementBits == 8;
  if (isB16Mask) {
    return StringRef("b16");
  }
  if (layout.getLaneStride() == 2 &&
      (elementBits == 16 || elementBits == 32)) {
    return StringRef("b32");
  }
  bool isB32MaskForPack4 = layout.getLaneStride() == 4 && elementBits == 8;
  if (isB32MaskForPack4) {
    return StringRef("b32");
  }
  return std::nullopt;
}

std::optional<StringRef>
getDenseLaneStrideStoreMaskGranularity(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  return getLaneStrideStoreMaskGranularity(layout, type.getElementType());
}

std::optional<StringRef>
getDenseLaneStrideMaskedStoreMaskGranularity(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  bool isB16MaskedStore = layout.getLaneStride() == 2 && elementBits == 8;
  if (isB16MaskedStore) {
    return StringRef("b16");
  }
  bool isB32MaskedStore = layout.getLaneStride() == 2 && elementBits == 16;
  if (isB32MaskedStore) {
    return StringRef("b32");
  }
  bool isB32MaskedStorePack4 =
      layout.getLaneStride() == 4 && elementBits == 8;
  if (isB32MaskedStorePack4) {
    return StringRef("b32");
  }
  return std::nullopt;
}

std::optional<std::string> getPointStoreDistToken(Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return std::nullopt;
  }
  return (Twine("1PT_B") + Twine(elementBits)).str();
}

std::optional<std::string> getScalarBroadcastLoadDistToken(Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return std::nullopt;
  }
  return (Twine("BRC_B") + Twine(elementBits)).str();
}

struct VPTOCmpMode {
  StringRef mode;
  std::optional<IntegerType::SignednessSemantics> signedness;
};

std::optional<VPTOCmpMode> getVPTOCmpFMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne" || predicate == "lt" ||
      predicate == "le" || predicate == "gt" || predicate == "ge") {
    return VPTOCmpMode{predicate, std::nullopt};
  }
  if (predicate == "oeq") {
    return VPTOCmpMode{StringRef("eq"), std::nullopt};
  }
  if (predicate == "one") {
    return VPTOCmpMode{StringRef("ne"), std::nullopt};
  }
  if (predicate == "olt") {
    return VPTOCmpMode{StringRef("lt"), std::nullopt};
  }
  if (predicate == "ole") {
    return VPTOCmpMode{StringRef("le"), std::nullopt};
  }
  if (predicate == "ogt") {
    return VPTOCmpMode{StringRef("gt"), std::nullopt};
  }
  if (predicate == "oge") {
    return VPTOCmpMode{StringRef("ge"), std::nullopt};
  }
  return std::nullopt;
}

std::optional<VPTOCmpMode> getVPTOCmpIMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne") {
    return VPTOCmpMode{predicate, std::nullopt};
  }
  if (predicate == "ult") {
    return VPTOCmpMode{
        StringRef("lt"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "ule") {
    return VPTOCmpMode{
        StringRef("le"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "ugt") {
    return VPTOCmpMode{
        StringRef("gt"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "uge") {
    return VPTOCmpMode{
        StringRef("ge"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "slt") {
    return VPTOCmpMode{
        StringRef("lt"), IntegerType::SignednessSemantics::Signed};
  }
  if (predicate == "sle") {
    return VPTOCmpMode{
        StringRef("le"), IntegerType::SignednessSemantics::Signed};
  }
  if (predicate == "sgt") {
    return VPTOCmpMode{
        StringRef("gt"), IntegerType::SignednessSemantics::Signed};
  }
  if (predicate == "sge") {
    return VPTOCmpMode{
        StringRef("ge"), IntegerType::SignednessSemantics::Signed};
  }
  return std::nullopt;
}

template <typename SourceOp>
std::optional<VPTOCmpMode> getVPTOCmpMode(StringRef predicate) {
  if constexpr (std::is_same_v<SourceOp, VMICmpIOp>) {
    return getVPTOCmpIMode(predicate);
  } else {
    return getVPTOCmpFMode(predicate);
  }
}

template <typename SourceOp>
StringRef getSupportedComparePredicateMessage() {
  if constexpr (std::is_same_v<SourceOp, VMICmpIOp>) {
    return "eq/ne, unsigned integer forms ult/ule/ugt/uge, and signed "
           "integer forms slt/sle/sgt/sge";
  } else {
    return "eq/ne/lt/le/gt/ge and ordered FP forms oeq/one/olt/ole/ogt/oge";
  }
}

template <typename SourceOp>
LogicalResult checkSupportedComparePredicate(Operation *op,
                                             StringRef predicate) {
  if (getVPTOCmpMode<SourceOp>(predicate)) {
    return success();
  }
  return op->emitError()
         << kVMIDiagUnsupportedPrefix << "compare predicate " << predicate
         << " cannot be lowered to pto.vcmp; supported predicates are "
         << getSupportedComparePredicateMessage<SourceOp>();
}

struct OneToNVMIUnpackOpPattern : OneToNOpConversionPattern<VMIUnpackOp> {
  using OneToNOpConversionPattern<VMIUnpackOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIUnpackOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    bool arityMismatch = sourceParts.size() != op->getNumResults();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "converted source part count must match unpack results");
    }
    replaceOpWithFlatConvertedValues(rewriter, op, sourceParts,
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIPackOpPattern : OneToNOpConversionPattern<VMIPackOp> {
  using OneToNOpConversionPattern<VMIPackOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIPackOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<int64_t> arity = getVMIPhysicalArity(op.getResult().getType());
    SmallVector<Value> flatOperands = flattenOneToNOperands(adaptor.getOperands());
    bool arityMismatch =
        failed(arity) || static_cast<int64_t>(flatOperands.size()) != *arity;
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "pack part count must match converted VMI result arity");
    }
    replaceOpWithFlatConvertedValues(rewriter, op, flatOperands,
                                     *this->getTypeConverter());
    return success();
  }
};

LogicalResult verifyIdentityPartForwarding(Operation *op,
                                           ValueRange sourceParts,
                                           TypeRange resultTypes,
                                           PatternRewriter &rewriter) {
  bool arityMismatch = sourceParts.size() != resultTypes.size();
  if (arityMismatch) {
    return rewriter.notifyMatchFailure(
        op, "source and result physical arity mismatch");
  }
  for (auto [part, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
    bool typeMismatch = part.getType() != resultType;
    if (typeMismatch) {
      return rewriter.notifyMatchFailure(
          op, "helper requires non-identity physical materialization");
    }
  }
  return success();
}

FailureOr<VRegType> getUnsignedCarrierVRegType(MLIRContext *ctx,
                                               unsigned elementBits) {
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return failure();
  }
  auto elementType = IntegerType::get(
      ctx, elementBits, IntegerType::SignednessSemantics::Unsigned);
  return VRegType::get(ctx, 2048 / elementBits, elementType);
}

FailureOr<VRegType>
getSignednessCarrierVRegType(VRegType inputType,
                             IntegerType::SignednessSemantics signedness) {
  auto inputElementType = dyn_cast<IntegerType>(inputType.getElementType());
  if (!inputElementType) {
    return failure();
  }
  if ((signedness == IntegerType::SignednessSemantics::Signed &&
       !inputElementType.isUnsigned()) ||
      (signedness == IntegerType::SignednessSemantics::Unsigned &&
       inputElementType.isUnsigned())) {
    return inputType;
  }
  auto carrierElementType = IntegerType::get(
      inputType.getContext(), inputElementType.getWidth(), signedness);
  return VRegType::get(inputType.getContext(), inputType.getElementCount(),
                       carrierElementType);
}

FailureOr<Value> bitcastVReg(Location loc, Value value, Type resultType,
                             PatternRewriter &rewriter) {
  bool isIdentity = value.getType() == resultType;
  if (isIdentity) {
    return value;
  }
  auto inputType = dyn_cast<VRegType>(value.getType());
  auto outputType = dyn_cast<VRegType>(resultType);
  if (!inputType || !outputType) {
    return failure();
  }
  return rewriter.create<VbitcastOp>(loc, outputType, value).getResult();
}

FailureOr<VRegType> getVcaddResultType(VRegType inputType) {
  auto inputIntegerType = dyn_cast<IntegerType>(inputType.getElementType());
  bool preservesType =
      !inputIntegerType || inputIntegerType.getWidth() == 32;
  if (preservesType) {
    return inputType;
  }
  unsigned inputWidth = inputIntegerType.getWidth();
  if (inputWidth != 8 && inputWidth != 16) {
    return failure();
  }
  auto resultElementType = IntegerType::get(
      inputType.getContext(), inputWidth * 2,
      inputIntegerType.getSignedness());
  return VRegType::get(inputType.getContext(),
                       inputType.getElementCount() / 2, resultElementType);
}

FailureOr<Value> unpackToNextCarrier(Location loc, Value source,
                                     unsigned sourceBits, int64_t partIndex,
                                     PatternRewriter &rewriter) {
  FailureOr<VRegType> resultType =
      getUnsignedCarrierVRegType(rewriter.getContext(), sourceBits * 2);
  if (failed(resultType)) {
    return failure();
  }
  Value part = rewriter.create<arith::ConstantIndexOp>(loc, partIndex);
  return rewriter.create<VzunpackOp>(loc, *resultType, source, part)
      .getResult();
}

FailureOr<Value> packToPreviousCarrier(Location loc, Value source,
                                       unsigned resultBits,
                                       StringRef part,
                                       PatternRewriter &rewriter) {
  FailureOr<VRegType> resultType =
      getUnsignedCarrierVRegType(rewriter.getContext(), resultBits);
  if (failed(resultType)) {
    return failure();
  }
  return rewriter
      .create<VpackOp>(loc, *resultType, source,
                       rewriter.getStringAttr(part))
      .getResult();
}

static FailureOr<unsigned> validateDenseLaneStrideShape(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, bool unpack,
    PatternRewriter &rewriter) {
  if (laneStride <= 0) {
    return rewriter.notifyMatchFailure(
        op, "dense lane_stride materialization requires positive lane stride");
  }
  int64_t safeLaneStride = laneStride;
  bool emptyParts = sourceParts.empty() || resultTypes.empty();
  bool arityMismatch = unpack
                           ? (resultTypes.size() + safeLaneStride - 1) /
                                     safeLaneStride !=
                                 sourceParts.size()
                           : (sourceParts.size() + safeLaneStride - 1) /
                                     safeLaneStride !=
                                 resultTypes.size();
  if (emptyParts || arityMismatch) {
    StringRef direction = unpack ? "unpack" : "pack";
    return rewriter.notifyMatchFailure(
        op, Twine("dense lane_stride ") + direction +
                " materialization requires one partial or complete stride "
                "group per physical part");
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  bool unsupportedShape =
      (laneStride != 2 && laneStride != 4) ||
      (laneStride == 4 && elementBits != 8) ||
      (elementBits != 8 && elementBits != 16);
  if (unsupportedShape) {
    StringRef direction = unpack ? "unpack" : "pack";
    return rewriter.notifyMatchFailure(
        op, Twine("unsupported dense lane_stride ") + direction +
                " carrier shape");
  }
  return elementBits;
}

static FailureOr<Value> materializeContiguousLaneStridePart(
    Operation *op, Value source, Type resultType, unsigned elementBits,
    VRegType inputCarrier, int64_t laneStride, int64_t part,
    PatternRewriter &rewriter) {
  FailureOr<Value> current =
      bitcastVReg(op->getLoc(), source, inputCarrier, rewriter);
  if (failed(current)) {
    return failure();
  }
  FailureOr<Value> unpacked = unpackToNextCarrier(
      op->getLoc(), *current, elementBits,
      laneStride == 4 ? part / 2 : part, rewriter);
  if (failed(unpacked)) {
    return failure();
  }
  current = *unpacked;
  if (laneStride == 4) {
    unpacked = unpackToNextCarrier(op->getLoc(), *current, elementBits * 2,
                                   part % 2, rewriter);
    if (failed(unpacked)) {
      return failure();
    }
    current = *unpacked;
  }
  return bitcastVReg(op->getLoc(), *current, resultType, rewriter);
}

FailureOr<SmallVector<Value>> materializeContiguousToLaneStride(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, PatternRewriter &rewriter) {
  FailureOr<unsigned> elementBits = validateDenseLaneStrideShape(
      op, sourceParts, resultTypes, elementType, laneStride, true, rewriter);
  if (failed(elementBits)) {
    return failure();
  }

  if (laneStride <= 0) {
    return failure();
  }

  MLIRContext *ctx = rewriter.getContext();
  FailureOr<VRegType> inputCarrier =
      getUnsignedCarrierVRegType(ctx, *elementBits);
  if (failed(inputCarrier)) {
    return failure();
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    int64_t sourceIndex = resultIndex / laneStride;
    if (sourceIndex >= static_cast<int64_t>(sourceParts.size())) {
      return failure();
    }
    int64_t part = resultIndex % laneStride;
    FailureOr<Value> result = materializeContiguousLaneStridePart(
        op, sourceParts[sourceIndex], resultType, *elementBits, *inputCarrier,
        laneStride, part, rewriter);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  return results;
}

static FailureOr<Value> mergeLaneStrideCarrierPair(
    Operation *op, Value lowCarrier, Value highCarrier, unsigned carrierBits,
    PatternRewriter &rewriter) {
  FailureOr<Value> low = packToPreviousCarrier(
      op->getLoc(), lowCarrier, carrierBits / 2, "LOWER", rewriter);
  if (failed(low)) {
    return failure();
  }
  FailureOr<Value> high = packToPreviousCarrier(
      op->getLoc(), highCarrier, carrierBits / 2, "HIGHER", rewriter);
  if (failed(high)) {
    return failure();
  }
  FailureOr<Value> mask = createAllTrueMaskForVReg(
      op->getLoc(), cast<VRegType>((*low).getType()), rewriter);
  if (failed(mask)) {
    return failure();
  }
  return rewriter.create<VorOp>(op->getLoc(), (*low).getType(), *low, *high,
                                *mask)
      .getResult();
}

static FailureOr<Value> materializeLaneStrideResultPart(
    Operation *op, ValueRange sourceParts, Type resultType, size_t sourceBegin,
    size_t sourceEnd, unsigned elementBits, unsigned carrierBits,
    VRegType sourceCarrier, PatternRewriter &rewriter) {
  SmallVector<Value> currentLevel;
  currentLevel.reserve(sourceEnd - sourceBegin);
  for (Value source : sourceParts.slice(sourceBegin, sourceEnd - sourceBegin)) {
    FailureOr<Value> carrier =
        bitcastVReg(op->getLoc(), source, sourceCarrier, rewriter);
    if (failed(carrier)) {
      return failure();
    }
    currentLevel.push_back(*carrier);
  }

  unsigned currentBits = carrierBits;
  while (currentBits > elementBits) {
    SmallVector<Value> nextLevel;
    nextLevel.reserve((currentLevel.size() + 1) / 2);
    for (size_t index = 0; index < currentLevel.size(); index += 2) {
      Value merged;
      if (index + 1 < currentLevel.size()) {
        FailureOr<Value> pair = mergeLaneStrideCarrierPair(
            op, currentLevel[index], currentLevel[index + 1], currentBits,
            rewriter);
        if (failed(pair)) {
          return failure();
        }
        merged = *pair;
      } else {
        FailureOr<Value> low = packToPreviousCarrier(
            op->getLoc(), currentLevel[index], currentBits / 2, "LOWER",
            rewriter);
        if (failed(low)) {
          return failure();
        }
        merged = *low;
      }
      nextLevel.push_back(merged);
    }
    currentLevel = std::move(nextLevel);
    currentBits /= 2;
  }
  bool invalidResultArity = currentLevel.size() != 1;
  if (invalidResultArity) {
    return failure();
  }
  return bitcastVReg(op->getLoc(), currentLevel.front(), resultType, rewriter);
}

static FailureOr<Value> materializeGroupSlotLaneStridePart(
    Operation *op, Value source, Type resultType, Type elementType,
    int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  unsigned carrierBits = elementBits * sourceStride;
  FailureOr<VRegType> carrierType =
      getUnsignedCarrierVRegType(rewriter.getContext(), carrierBits);
  if (failed(carrierType)) {
    return failure();
  }
  FailureOr<Value> current =
      bitcastVReg(op->getLoc(), source, *carrierType, rewriter);
  if (failed(current)) {
    return failure();
  }

  int64_t currentStride = sourceStride;
  while (currentStride < resultStride) {
    FailureOr<Value> unpacked = unpackToNextCarrier(
        op->getLoc(), *current, carrierBits, /*partIndex=*/0, rewriter);
    if (failed(unpacked)) {
      return failure();
    }
    current = *unpacked;
    currentStride *= 2;
    carrierBits *= 2;
  }
  while (currentStride > resultStride) {
    FailureOr<Value> packed = packToPreviousCarrier(
        op->getLoc(), *current, carrierBits / 2, "LOWER", rewriter);
    if (failed(packed)) {
      return failure();
    }
    current = *packed;
    currentStride /= 2;
    carrierBits /= 2;
  }
  return bitcastVReg(op->getLoc(), *current, resultType, rewriter);
}

static FailureOr<SmallVector<Value>> materializeLaneStrideResultList(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    unsigned elementBits, unsigned carrierBits, VRegType sourceCarrier,
    int64_t laneStride, PatternRewriter &rewriter) {
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    size_t sourceBegin = resultIndex * laneStride;
    size_t sourceEnd =
        std::min<size_t>(sourceBegin + laneStride, sourceParts.size());
    FailureOr<Value> result = materializeLaneStrideResultPart(
        op, sourceParts, resultType, sourceBegin, sourceEnd, elementBits,
        carrierBits, sourceCarrier, rewriter);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  return results;
}


