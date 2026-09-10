// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI.cpp - PTO VMI type and attribute support -----------------------===//
//===----------------------------------------------------------------------===//

#include "VMIInternal.h"

using namespace mlir;
using namespace mlir::pto;

// Legacy (old) VMI op verifiers
//===----------------------------------------------------------------------===//

//===--- Legacy (old) VMI op verifiers ---===//

LogicalResult verifySignedI32OrF16F32ElementType(Operation *op,
                                                        Type elementType) {
  bool supportedInteger = false;
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    supportedInteger =
        intType.getWidth() == mlir::pto::kValue32 &&
        matchesVMIIntSemantics(intType, VMIIntSignSemantics::Signed);
  }
  if (!supportedInteger && !isVMIF16OrF32Type(elementType)) {
    return op->emitOpError("requires si32, f16, or f32 VMI element type");
  }
  return success();
}

// Shared helper: validates mask-data alignment and pmode value.
// Applicable to all VMI elementwise ops that carry mask + pmode.
LogicalResult verifyVMIPmodeMask(Operation *op, VMIMaskType maskType,
                                        VMIVRegType dataType,
                                        std::optional<StringRef> pmode) {
  if (failed(verifyMaskMatchesData(op, maskType, dataType))) {
    return failure();
  }
  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("pmode must be \"merge\" or \"zero\", got \"")
             << mode << "\"";
    }
  }
  return success();
}
// Variadic-aware variant: skips mask validation when no mask operand is
// provided (unified v-ops allow an absent mask meaning "all-true").
LogicalResult verifyVMIVariadicPmodeMask(Operation *op,
                                                ValueRange maskParts,
                                                VMIVRegType dataType,
                                                std::optional<StringRef> pmode) {
  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("pmode must be \"merge\" or \"zero\", got \"")
             << mode << "\"";
    }
  }
  if (maskParts.empty()) {
    return success();
  }
  if (maskParts.size() != 1) {
    return op->emitOpError("expects at most one mask operand");
  }
  return verifyMaskMatchesData(op, cast<VMIMaskType>(maskParts.front().getType()),
                               dataType);
}

//===----------------------------------------------------------------------===//
// VMIvStoreOp

Type mlir::pto::getVMIPhysicalDataElementType(VMIVRegType type) {
  // lane_stride describes where logical elements reside inside a physical
  // vector register; it does not change their element type.  Packed group
  // slots therefore use the same sparse logical-element carrier as dense
  // lane-stride values.
  return type.getElementType();
}

FailureOr<int64_t> mlir::pto::getDataLanesPerPart(Type elementType) {
  unsigned elementBitWidth = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBitWidth == 0) {
    return failure();
  }
  constexpr int64_t kPhysicalVRegBits = mlir::pto::kValue256 * mlir::pto::kValue8;
  if (kPhysicalVRegBits % elementBitWidth != 0) {
    return failure();
  }
  return kPhysicalVRegBits / elementBitWidth;
}

FailureOr<int64_t> mlir::pto::getMaskLanesPerPart(StringRef granularity) {
  if (granularity == "b8") {
    return mlir::pto::kValue256;
  }
  if (granularity == "b16") {
    return mlir::pto::kValue128;
  }
  if (granularity == "b32") {
    return mlir::pto::kValue64;
  }
  return failure();
}

FailureOr<int64_t> mlir::pto::getVMILayoutBlockElems(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout)) {
    return failure();
  }
  if (!(*layout).isBlockDeinterleaved()) {
    return 1;
  }

  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  constexpr int64_t kVCGBlocksPerPart = mlir::pto::kValue8;
  if (failed(lanesPerPart) || *lanesPerPart <= 0 ||
      *lanesPerPart % kVCGBlocksPerPart != 0) {
    return failure();
  }
  return *lanesPerPart / kVCGBlocksPerPart;
}

FailureOr<int64_t> mlir::pto::getVMIPhysicalArity(Type type) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(elementCount) || failed(lanesPerPart) || failed(layout)) {
    return failure();
  }

  if ((*layout).isGroupSlots() && (*layout).getSlots() > 0) {
    return divideCeilNonNegative((*layout).getNumGroups(),
                                 (*layout).getSlots());
  }

  int64_t factor = (*layout).isDenseSplit() ? (*layout).getFactor() : 1;
  FailureOr<int64_t> blockElems = getVMILayoutBlockElems(type);
  if (failed(blockElems)) {
    return failure();
  }
  int64_t laneStride =
      isa<VMIMaskType>(type) ? 1
                             : ((*layout).isDense() ? (*layout).getLaneStride()
                                                    : 1);
  int64_t arity = 0;
  for (int64_t part = 0; part < factor; ++part) {
    int64_t lanesInPart =
        getDenseLogicalLanesInPart(*elementCount, factor, *blockElems, part);
    int64_t requiredPhysicalLanes =
        lanesInPart == 0 ? 0 : (lanesInPart - 1) * laneStride + 1;
    arity += divideCeilNonNegative(requiredPhysicalLanes, *lanesPerPart);
  }
  return arity;
}

namespace {

// Shared five-field layout dimension tuple for the inverse lane-mapping pair.
struct VMILayoutDims {
  int64_t elementCount, factor, blockElems, laneStride, lanesPerPart;
};

FailureOr<VMILayoutDims> getVMILayoutDims(Type type) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> laneStride = getDenseLaneStride(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(laneStride) || failed(lanesPerPart)) {
    return failure();
  }
  return VMILayoutDims{*elementCount, *factor, *blockElems, *laneStride,
                       *lanesPerPart};
}

} // namespace

FailureOr<VMIPhysicalLane>
mlir::pto::mapLogicalLaneToPhysical(Type type, int64_t logicalLane) {
  FailureOr<VMILayoutDims> dims = getVMILayoutDims(type);
  if (failed(dims)) {
    return failure();
  }
  const auto [elementCount, factor, blockElems, laneStride, lanesPerPart] =
      *dims;
  if (logicalLane < 0 || logicalLane >= elementCount) {
    return failure();
  }

  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (succeeded(layout) && (*layout).isGroupSlots() &&
      (*layout).getSlots() > 0) {
    int64_t slots = (*layout).getSlots();
    int64_t lane = logicalLane % slots;
    if (lane >= lanesPerPart) {
      return failure();
    }
    return VMIPhysicalLane{/*part=*/0, logicalLane / slots, lane};
  }

  int64_t part = 0;
  std::optional<int64_t> indexInPart = mapDenseLogicalLaneToPartIndex(
      elementCount, factor, blockElems, logicalLane, part);
  if (!indexInPart) {
    return failure();
  }
  int64_t physicalIndex = *indexInPart * laneStride;
  return VMIPhysicalLane{part, physicalIndex / lanesPerPart,
                         physicalIndex % lanesPerPart};
}

FailureOr<int64_t> mlir::pto::mapPhysicalLaneToLogical(Type type, int64_t part,
                                                       int64_t chunk,
                                                       int64_t lane) {
  FailureOr<VMILayoutDims> dims = getVMILayoutDims(type);
  if (failed(dims)) {
    return failure();
  }
  const auto [elementCount, factor, blockElems, laneStride, lanesPerPart] =
      *dims;
  if (part < 0 || part >= factor || chunk < 0 || lane < 0 ||
      lane >= lanesPerPart) {
    return failure();
  }

  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (succeeded(layout) && (*layout).isGroupSlots() &&
      (*layout).getSlots() > 0) {
    int64_t slots = (*layout).getSlots();
    if (part != 0 || lane >= slots) {
      return failure();
    }
    int64_t logicalLane = chunk * slots + lane;
    if (logicalLane >= elementCount) {
      return failure();
    }
    return logicalLane;
  }

  int64_t physicalIndexInPart = chunk * lanesPerPart + lane;
  if (physicalIndexInPart % laneStride != 0) {
    return failure();
  }
  int64_t indexInPart = physicalIndexInPart / laneStride;
  std::optional<int64_t> logicalLane = mapDensePartIndexToLogicalLane(
      elementCount, factor, blockElems, part, indexInPart);
  if (!logicalLane) {
    return failure();
  }
  return *logicalLane;
}

FailureOr<bool> mlir::pto::isPaddingLane(Type type, int64_t part, int64_t chunk,
                                         int64_t lane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> laneStride = getDenseLaneStride(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(laneStride) || failed(lanesPerPart)) {
    return failure();
  }
  if (part < 0 || part >= *factor || chunk < 0 || lane < 0 ||
      lane >= *lanesPerPart) {
    return failure();
  }

  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (succeeded(layout) && (*layout).isGroupSlots() &&
      (*layout).getSlots() > 0) {
    int64_t slots = (*layout).getSlots();
    if (part != 0) {
      return true;
    }
    if (lane >= slots) {
      return true;
    }
    return chunk * slots + lane >= *elementCount;
  }

  int64_t lanesInPart =
      getDenseLogicalLanesInPart(*elementCount, *factor, *blockElems, part);
  int64_t physicalIndexInPart = chunk * *lanesPerPart + lane;
  if (physicalIndexInPart % *laneStride != 0) {
    return true;
  }
  int64_t indexInPart = physicalIndexInPart / *laneStride;
  return indexInPart >= lanesInPart;
}
