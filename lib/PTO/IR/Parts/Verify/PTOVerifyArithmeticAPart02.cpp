// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticA.cpp; kept as a fragment included by PTOVerifyArithmeticA.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static bool isCommGlobalLikeType(Type ty) {
  if (auto memTy = dyn_cast<MemRefType>(ty))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty);
}

static LogicalResult verifyPositiveStaticShape(Operation *op, Type ty,
                                               StringRef name) {
  SmallVec4<int64_t> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
  }
  return success();
}

static LogicalResult verifyCommGlobalLike(Operation *op, Value value,
                                          StringRef name) {
  Type ty = value.getType();
  if (!isCommGlobalLikeType(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";
  return verifyPositiveStaticShape(op, ty, name);
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name)))
    return failure();
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(kPTOI32BitWidth))
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a tile_buf or memref tile";
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  return verifyPositiveStaticShape(op, ty, name);
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty())
    return op->emitOpError() << "expects at least one " << name << " operand";
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str())))
      return failure();
    if (it.value().getType() != groupTy)
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong)
    return success();
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName)))
    return failure();
  if (ping.getType() != pong.getType())
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVec4<int64_t> shape = getShapeVec(ty);
  if (shape.empty())
    return std::nullopt;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0)
    return std::nullopt;

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace()))
      return as.getAddressSpace();
    return std::nullopt;
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(mr.getMemorySpace()))
      return as.getAddressSpace();
    if (!mr.getMemorySpace())
      return pto::AddressSpace::GM;
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == kPTORowColRank && tb.getValidShape().size() == kPTORowColRank;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (allowBf16 && ty.isBF16())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case kPTOI32BitWidth:
    case kPTOI16BitWidth:
      return true;
    case kPTOI8BitWidth:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != kPTOI32BitWidth)
    return false;
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true))
    return true;
  if (!isTargetArchA5(op))
    return false;
  return ty.isFloat8E4M3() || ty.isFloat8E4M3FN() || ty.isFloat8E4M3FNUZ() ||
         ty.isFloat8E4M3B11FNUZ() || ty.isFloat8E5M2() || ty.isFloat8E5M2FNUZ();
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == kPTOI32BitWidth);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == kPTOI32BitWidth);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy)
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";
  if (isa<pto::PartitionTensorViewType>(memTy)) {
    if (auto layout = getLogicalViewLayout(memValue)) {
      if (*layout != pto::Layout::ND)
        return op->emitOpError(
            "expects mem partition view to use ND logical layout when layout "
            "can be inferred");
    }
    return success();
  }

  if (auto mr = dyn_cast<MemRefType>(memTy)) {
    auto as = getPTOMemorySpaceEnum(mr);
    if (!as || (*as != pto::AddressSpace::GM &&
                 *as != pto::AddressSpace::Zero))
      return op->emitOpError(
          "expects mem memref to use GM or zero address space");
    if (mr.getRank() == kPTOPaddedTensorRank5D) {
      auto shape = mr.getShape();
      bool allStatic = true;
      for (int64_t d : shape)
        if (d == ShapedType::kDynamic)
          allStatic = false;
      if (allStatic && (shape[0] != 1 || shape[1] != 1 || shape[2] != 1))
        return op->emitOpError(
            "expects rank-5 GM memref leading dimensions to be [1,1,1,...] "
            "(GlobalTensor table shape)");
    }
    return success();
  }

  return op->emitOpError(
      "expects mem to be !pto.partition_tensor_view or a GM/ZERO memref");
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);
