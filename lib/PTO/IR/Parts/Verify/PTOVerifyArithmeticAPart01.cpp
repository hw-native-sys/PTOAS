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

LogicalResult pto::TAbsOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy))
    elemTy = tb.getElementType();
  else if (auto mr = dyn_cast<MemRefType>(srcTy))
    elemTy = mr.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<MemRefType, RankedTensorType,
                pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType, MemRefType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto mr = mlir::dyn_cast<MemRefType>(ty)) return mr.getElementType();
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) return tt.getElementType();
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) return tv.getElementType();
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) return tb.getElementType();
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) return tv.getElementType();
  return Type();
}

static SmallVec4<int64_t> getShapeVec(Type ty) {
  SmallVec4<int64_t> s;
  if (auto mr = mlir::dyn_cast<MemRefType>(ty))
    return SmallVec4<int64_t>(mr.getShape().begin(), mr.getShape().end());
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty))
    return SmallVec4<int64_t>(tt.getShape().begin(), tt.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty))
    return SmallVec4<int64_t>(tv.getShape().begin(), tv.getShape().end());
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty))
    return SmallVec4<int64_t>(tb.getShape().begin(), tb.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty))
    return SmallVec4<int64_t>(tv.getShape().begin(), tv.getShape().end());
  return {};
}

static SmallVec4<int64_t> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return SmallVec4<int64_t>(tb.getValidShape().begin(), tb.getValidShape().end());
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy))
    return rawDim;
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * kPTOFloat4PackedExpansion : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  return std::nullopt;
}

static SmallVec4<int64_t> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVec4<int64_t> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != kPTORowColRank)
    return dims;

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i)
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  return dims;
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVec4<int64_t> getValidShapeVec(Value value) {
  if (!value)
    return {};
  auto valid = getValidShapeVec(value.getType());
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (valid.size() >= 1 && bind.getValidRow())
      valid[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (valid.size() >= kPTORowColRank && bind.getValidCol())
      valid[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return valid;
}

static SmallVec4<int64_t> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size())
    return shape;

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic)
      shape[i] = valid[i];
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == kPTOI8BitWidth;
}

static LogicalResult verifyAsyncFlatContiguous1DGMMemRef(Operation *op,
                                                         Value value,
                                                         StringRef name) {
  auto memTy = dyn_cast<MemRefType>(value.getType());
  if (!memTy)
    return op->emitOpError() << "expects " << name << " to be a memref";
  if (!memTy.hasRank())
    return op->emitOpError() << "expects " << name << " to be a ranked memref";
  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError() << "expects " << name
                             << " to be in GM address space";

  ArrayRef<int64_t> shape = memTy.getShape();
  if (shape.empty())
    return op->emitOpError() << "expects " << name
                             << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memTy, strides, offset)))
    return op->emitOpError() << "expects " << name
                             << " to be a strided memref with a known layout";

  bool hasDynamicLayout =
      offset == ShapedType::kDynamic ||
      llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      });
  if (hasDynamicLayout)
    return success();

  bool packed = !strides.empty() && strides.back() == 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0 && packed; --i)
    packed = packed && (strides[i] == strides[i + 1] * shape[i + 1]);
  if (!packed)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D = logical1D && (shape[i] == 1);
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";
  return success();
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  Type ty = value.getType();
  if (isa<MemRefType>(ty))
    return verifyAsyncFlatContiguous1DGMMemRef(op, value, name);
  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a memref/tensor_view/partition_view";

  SmallVec4<int64_t> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D = logical1D && (shape[i] == 1);
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";
  return success();
}
