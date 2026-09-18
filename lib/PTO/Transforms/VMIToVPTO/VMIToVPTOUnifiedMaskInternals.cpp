// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOUnifiedMaskInternals.inc - unified masked lowering helpers -*- C++ -*-===//
//===----------------------------------------------------------------------===//

FailureOr<Value> adaptMaskToVRegGranularity(Location loc, Value mask,
                                            VRegType dataType,
                                            PatternRewriter &rewriter) {
  auto sourceType = dyn_cast<MaskType>(mask.getType());
  FailureOr<MaskType> targetType =
      getMaskTypeForVReg(dataType, rewriter.getContext());
  if (!sourceType || failed(targetType)) {
    return failure();
  }

  Value currentMask = mask;
  int64_t currentBits = getMaskGranularityBits(sourceType.getGranularity());
  int64_t targetBits = getMaskGranularityBits((*targetType).getGranularity());
  if (currentBits == 0 || targetBits == 0 || currentBits < targetBits ||
      currentBits % targetBits != 0 || currentBits / targetBits > mlir::pto::kValue4) {
    return failure();
  }
  if (sourceType != *targetType) {
    currentMask =
        rewriter.create<PbitcastOp>(loc, *targetType, currentMask).getResult();
  }
  if (currentMask.getType() != *targetType) {
    return failure();
  }
  return currentMask;
}

static void includeStridePaddingInPrefix(SmallVectorImpl<int8_t> &activeLanes) {
  auto lastActive = activeLanes.end();
  for (auto it = activeLanes.begin(); it != activeLanes.end(); ++it) {
    if (*it != 0) {
      lastActive = it;
    }
  }
  if (lastActive != activeLanes.end()) {
    std::fill(activeLanes.begin(), std::next(lastActive), 1);
  }
}

FailureOr<Value> materializeAllLogicalLanesMaskPart(Location loc,
                                                    Type vmiType,
                                                    MaskType physicalType,
                                                    size_t targetIndex,
                                                    bool includeStridePadding,
                                                    PatternRewriter &rewriter) {
  VMILayoutAttr layout;
  if (auto dataType = dyn_cast<VMIVRegType>(vmiType)) {
    layout = dataType.getLayoutAttr();
  } else if (auto maskType = dyn_cast<VMIMaskType>(vmiType)) {
    layout = maskType.getLayoutAttr();
  }
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(physicalType.getGranularity());
  if (!layout || failed(lanesPerPart)) {
    return failure();
  }

  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  size_t physicalIndex = 0;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      SmallVector<int8_t> activeLanes(*lanesPerPart, 0);
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPaddingLane(vmiType, part, chunk, lane);
        if (failed(padding)) {
          return failure();
        }
        if (!*padding) {
          anyLane = true;
          activeLanes[lane] = 1;
        }
      }
      if (anyLane && includeStridePadding) {
        includeStridePaddingInPrefix(activeLanes);
      }
      if (!anyLane) {
        break;
      }
      if (physicalIndex == targetIndex) {
        return materializeConstantMaskChunk(loc, physicalType, activeLanes,
                                            rewriter);
      }
      ++physicalIndex;
    }
  }
  return failure();
}

struct ImplicitMaskPlan {
  Type logicalType;
  SmallVector<Type> physicalTypes;
  bool includeStridePadding = false;
};

static FailureOr<ImplicitMaskPlan> getImplicitMaskPlan(
    VMIVRegType logicalDataType, VRegType physicalDataType,
    StringRef granularity, const TypeConverter &typeConverter,
    PatternRewriter &rewriter) {
  ImplicitMaskPlan plan;
  plan.logicalType = VMIMaskType::get(
      rewriter.getContext(), logicalDataType.getElementCount(), granularity,
      logicalDataType.getLayoutAttr());
  if (succeeded(
          typeConverter.convertType(plan.logicalType, plan.physicalTypes))) {
    return plan;
  }

  // A b32 lane-stride mask would require an unsupported b64 carrier. Use the
  // data predicate granularity and include harmless gaps up to the last lane.
  plan.logicalType = logicalDataType;
  plan.includeStridePadding = true;
  FailureOr<MaskType> fallbackType =
      getMaskTypeForVReg(physicalDataType, rewriter.getContext());
  FailureOr<int64_t> dataArity = getVMIPhysicalArity(logicalDataType);
  bool invalidFallback =
      failed(fallbackType) || failed(dataArity) || *dataArity < 0;
  if (invalidFallback) {
    return failure();
  }
  plan.physicalTypes.assign(*dataArity, *fallbackType);
  return plan;
}

FailureOr<Value> getUnifiedMaskPart(Operation *op, ValueRange maskParts,
                                    size_t index, VRegType dataType,
                                    const TypeConverter &typeConverter,
                                    PatternRewriter &rewriter) {
  Value physicalMask;
  if (!maskParts.empty()) {
    if (index >= maskParts.size() ||
        !isa<MaskType>(maskParts[index].getType())) {
      return failure();
    }
    physicalMask = maskParts[index];
  } else {
    auto logicalDataType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!logicalDataType) {
      return failure();
    }
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(logicalDataType.getElementType());
    StringRef granularity = getMaskGranularityForBits(elementBits);
    if (granularity.empty()) {
      return failure();
    }

    FailureOr<ImplicitMaskPlan> plan = getImplicitMaskPlan(
        logicalDataType, dataType, granularity, typeConverter, rewriter);
    bool invalidPlan = failed(plan) || index >= plan->physicalTypes.size();
    if (invalidPlan) {
      return failure();
    }
    auto physicalMaskType = dyn_cast<MaskType>(plan->physicalTypes[index]);
    if (!physicalMaskType) {
      return failure();
    }
    FailureOr<Value> allLogicalLanes = materializeAllLogicalLanesMaskPart(
        op->getLoc(), plan->logicalType, physicalMaskType, index,
        plan->includeStridePadding, rewriter);
    if (failed(allLogicalLanes)) {
      return failure();
    }
    physicalMask = *allLogicalLanes;
  }
  return adaptMaskToVRegGranularity(op->getLoc(), physicalMask, dataType,
                                    rewriter);
}

FailureOr<Value> createLayoutPartitionActiveLanes(Location loc,
                                                  Value activeLanesI32,
                                                  VMIMaskType type,
                                                  int64_t part,
                                                  PatternRewriter &rewriter) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout) {
    return failure();
  }
  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  if (!layout.isBlockDeinterleaved()) {
    return createPartitionActiveLanes(loc, activeLanesI32, factor, part,
                                      rewriter);
  }

  FailureOr<int64_t> blockElems = getVMILayoutBlockElems(type);
  if (failed(blockElems) || *blockElems <= 0 || part < 0 || part >= factor) {
    return failure();
  }
  int64_t cycleElems = factor * *blockElems;
  Value cycle = createI32Constant(loc, cycleElems, rewriter);
  Value fullCycles =
      rewriter.create<arith::DivUIOp>(loc, activeLanesI32, cycle);
  Value activeFromFullCycles = rewriter.create<arith::MulIOp>(
      loc, fullCycles, createI32Constant(loc, *blockElems, rewriter));
  Value remainder = rewriter.create<arith::RemUIOp>(loc, activeLanesI32, cycle);

  int64_t partBegin = part * *blockElems;
  Value activeInPartialCycle = remainder;
  if (partBegin != 0) {
    Value begin = createI32Constant(loc, partBegin, rewriter);
    Value clampedRemainder =
        rewriter.create<arith::MaxUIOp>(loc, remainder, begin);
    activeInPartialCycle =
        rewriter.create<arith::SubIOp>(loc, clampedRemainder, begin);
  }
  activeInPartialCycle = rewriter.create<arith::MinUIOp>(
      loc, activeInPartialCycle, createI32Constant(loc, *blockElems, rewriter));
  return rewriter
      .create<arith::AddIOp>(loc, activeFromFullCycles, activeInPartialCycle)
      .getResult();
}
