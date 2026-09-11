// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTStoreA2Acc(TStoreOp op,
                                       pto::TileBufType srcTile,
                                       Type dstElem) {
  if (failed(verifyTStoreA2AccTypes(op, srcTile.getElementType(), dstElem)))
    return failure();
  auto shape = srcTile.getShape();
  if (shape[1] != ShapedType::kDynamic &&
      (shape[1] < 1 || shape[1] > 4095))
    return op.emitOpError(
        "expects A2/A3 acc tstore src cols to be in [1, 4095]");
  auto valid = srcTile.getValidShape();
  if (valid[1] != ShapedType::kDynamic &&
      (valid[1] < 0 || valid[1] > 4095))
    return op.emitOpError(
        "expects A2/A3 acc tstore src valid_shape[1] to be in [0, 4095]");
  return success();
}

static LogicalResult verifyTStoreA2A3(TStoreOp op) {
  auto common = verifyTStoreCommon(op, /*allowLowPrecision=*/false);
  if (failed(common))
    return failure();
  auto [srcTile, dstPart] = *common;
  auto space = getPTOMemorySpaceEnum(srcTile);
  if (!space || (*space != pto::AddressSpace::VEC &&
                 *space != pto::AddressSpace::MAT &&
                 *space != pto::AddressSpace::ACC))
    return op.emitOpError(
        "expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
  if (failed(verifyTStoreForms(op, *space)))
    return failure();
  if (*space == pto::AddressSpace::ACC)
    return verifyTStoreA2Acc(op, srcTile, dstPart.getElementType());
  return verifyTStoreA2VecMat(op, srcTile, dstPart.getElementType());
}

static LogicalResult verifyTStoreA5Vec(TStoreOp op,
                                       pto::TileBufType srcTile,
                                       Type dstElem) {
  if (op.getFp() || op.getPreQuantScalar())
    return op.emitOpError(
        "expects fp/preQuantScalar form to use loc=acc src");
  Type srcElem = srcTile.getElementType();
  if (!isA5TLoadStoreTransferElemType(srcElem))
    return op.emitOpError(
        "expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
    return op.emitOpError(
        "expects A5 vec tstore src and dst element types to have the same bitwidth");
  auto shape = srcTile.getShape();
  bool special = shape.size() == 2 && (shape[0] == 1 || shape[1] == 1);
  if (!special && !hasA5LoadStoreLayout(srcTile))
    return op.emitOpError(
        "expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
  return success();
}

static LogicalResult verifyTStoreA5Acc(TStoreOp op, Type srcElem,
                                       Type dstElem) {
  if (!(srcElem.isInteger(32) || srcElem.isF32()))
    return op.emitOpError(
        "expects A5 acc tstore src element type to be i32 or f32");
  if (op.getPreQuantScalar() &&
      !isA5AccStorePreQuantDstType(srcElem, dstElem))
    return op.emitOpError(
        "expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
  if (!op.getPreQuantScalar() && !op.getFp() &&
      !(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16()))
    return op.emitOpError(
        "expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
  return success();
}

static LogicalResult verifyTStoreA5(TStoreOp op) {
  auto common = verifyTStoreCommon(op, /*allowLowPrecision=*/true);
  if (failed(common))
    return failure();
  auto [srcTile, dstPart] = *common;
  auto space = getPTOMemorySpaceEnum(srcTile);
  if (!space || (*space != pto::AddressSpace::VEC &&
                 *space != pto::AddressSpace::ACC))
    return op.emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
  if (failed(verifyTStoreForms(op, *space)))
    return failure();
  if (*space == pto::AddressSpace::VEC)
    return verifyTStoreA5Vec(op, srcTile, dstPart.getElementType());
  return verifyTStoreA5Acc(op, srcTile.getElementType(),
                           dstPart.getElementType());
}

LogicalResult TStoreOp::verify() {
  bool hasFp = static_cast<bool>(getFp());
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  if (hasFp && hasPreQuant)
    return emitOpError(
        "expects fp and preQuantScalar to be mutually exclusive");
  if (hasFp && getStPhase() != pto::STPhase::Unspecified)
    return emitOpError("expects fp form to use the default stPhase");
  auto verifyA2A3 = [&]() { return verifyTStoreA2A3(*this); };
  auto verifyA5 = [&]() { return verifyTStoreA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false,
                                  /*allowInt8=*/false)) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  Type elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<RankedTensorType, pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return tt.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return tv.getElementType();
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return tb.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return tv.getElementType();
  }
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return SmallVector<int64_t, 4>(tt.getShape().begin(), tt.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getShape().begin(), tb.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  }
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy)) {
    return rawDim;
  }
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  }
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2) {
    return dims;
  }

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i) {
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  }
  return dims;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value) {
    return {};
  }
  auto valid = getValidShapeVec(value.getType());
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size()) {
    return shape;
  }

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic) {
      shape[i] = valid[i];
    }
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static FailureOr<SmallVector<int64_t, 4>> getGlobalLikeShape(
    Operation *op, Type ty, StringRef name) {
  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    op->emitOpError()
        << "expects " << name << " to be a tensor_view or partition_view";
    return failure();
  }
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    op->emitOpError() << "expects " << name << " to have rank >= 1";
    return failure();
  }
  return shape;
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  auto shape = getGlobalLikeShape(op, value.getType(), name);
  if (failed(shape))
    return failure();
  for (int64_t dim : *shape) {
    if (dim == ShapedType::kDynamic) {
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
    }
  }
