// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

constexpr int64_t kA2A3AccStoreMaxCols = 4095;
constexpr unsigned kShapeVecInlineCapacity = 4;
constexpr unsigned kI32ElemBitWidth = 32;
constexpr unsigned kByteElemBitWidth = 8;
constexpr int64_t kFp4PackedDimFactor = 2;
constexpr unsigned kLogicalTileRank = 2;

static LogicalResult verifyTStoreA2Acc(TStoreOp op,
                                       pto::TileBufType srcTile,
                                       Type dstElem) {
  if (failed(verifyTStoreA2AccTypes(op, srcTile.getElementType(), dstElem)))
    return failure();
  auto shape = srcTile.getShape();
  if (shape[1] != ShapedType::kDynamic &&
      (shape[1] < 1 || shape[1] > kA2A3AccStoreMaxCols))  // A2/A3 architecture limit
    return op.emitOpError(
        "expects A2/A3 acc tstore src cols to be in [1, 4095]");
  auto valid = srcTile.getValidShape();
  if (valid[1] != ShapedType::kDynamic &&
      (valid[1] < 0 || valid[1] > kA2A3AccStoreMaxCols))  // A2/A3 architecture limit
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
  bool special = shape.size() == kLogicalTileRank && (shape[0] == 1 || shape[1] == 1);
  if (!special && !hasA5LoadStoreLayout(srcTile))
    return op.emitOpError(
        "expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
  return success();
}

static LogicalResult verifyTStoreA5Acc(TStoreOp op, Type srcElem,
                                       Type dstElem) {
  if (!(srcElem.isInteger(kI32ElemBitWidth) || srcElem.isF32())) {
    return op.emitOpError(
        "expects A5 acc tstore src element type to be i32 or f32");
  }
  if (op.getPreQuantScalar() &&
      !isA5AccStorePreQuantDstType(srcElem, dstElem)) {
    return op.emitOpError(
        "expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
  }
  if (!op.getPreQuantScalar() && !op.getFp() &&
      !(dstElem.isInteger(kI32ElemBitWidth) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16())) {
    return op.emitOpError(
        "expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
  }
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
  if (getL2HintAttr()) {
    if (failed(verifyA5L2HintTarget(getOperation()))) {
      return failure();
    }
  }
  bool hasFp = static_cast<bool>(getFp());
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  if (hasFp && hasPreQuant)
    return emitOpError(
        "expects fp and preQuantScalar to be mutually exclusive");
  if (hasFp && getStPhase() != pto::STPhase::Unspecified)
    return emitOpError("expects fp form to use the default stPhase");
  auto verifyA2A3 = [this]() { return verifyTStoreA2A3(*this); };
  auto verifyA5 = [this]() { return verifyTStoreA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTensorDumpElemType(TensorDumpOp op,
                                              pto::TileBufType srcTile) {
  Type elem = srcTile.getElementType();
  const bool dumpable = elem.isF16() || elem.isBF16() || elem.isF32() ||
                        elem.isInteger(8) || elem.isInteger(16) ||
                        elem.isInteger(32) || elem.isInteger(64);
  if (!dumpable) {
    return op.emitOpError(
        "expects dumpable tile element type (f16/bf16/f32/i8/i16/i32/i64)");
  }
  return success();
}

/// The dump copies one destination row per valid tile row, using the tile's row
/// pitch as the source stride, so every destination dimension must match the
/// tile valid shape. Comparing element counts alone is not enough: a same-count
/// reshaped destination (for example 64x1 against a 4x16 valid shape) still
/// walks past the tile.
///
/// The frontend normalises a 2-D view to the pto-isa 5-D form by prepending
/// unit dimensions (8x16 becomes 1x1x1x8x16), so compare the trailing
/// dimensions and require any extra leading ones to be unit.
static LogicalResult verifyTensorDumpDstShape(
    TensorDumpOp op, pto::PartitionTensorViewType dstPart,
    pto::TileBufType srcTile) {
  const auto dstShape = dstPart.getShape();
  const auto validShape = srcTile.getValidShape();
  if (failed(verifyShapeSign(op, dstShape, "dst shape", true))) {
    return failure();
  }
  const bool dstRankTooSmall = dstShape.size() < validShape.size();
  if (dstRankTooSmall) {
    return op.emitOpError() << "expects dst rank (" << dstShape.size()
                            << ") to be at least the src valid_shape rank ("
                            << validShape.size() << ")";
  }
  const size_t leading = dstShape.size() - validShape.size();
  for (size_t i = 0; i < leading; ++i) {
    if (dstShape[i] != 1) {
      return op.emitOpError()
             << "expects dst shape[" << i << "] (" << dstShape[i]
             << ") to be 1: the dump covers exactly the tile valid region";
    }
  }
  for (auto [idx, dims] :
       llvm::enumerate(llvm::zip(dstShape.drop_front(leading), validShape))) {
    auto [dstDim, validDim] = dims;
    const bool dstDynamic = dstDim == ShapedType::kDynamic;
    const bool validDynamic = validDim == ShapedType::kDynamic;
    if (dstDynamic != validDynamic) {
      return op.emitOpError()
             << "expects dst shape[" << idx + leading
             << "] and src valid_shape[" << idx
             << "] to both be static or both be dynamic";
    }
    const bool mismatched = !dstDynamic && dstDim != validDim;
    if (mismatched) {
      return op.emitOpError()
             << "expects dst shape[" << idx + leading << "] (" << dstDim
             << ") to match src valid_shape[" << idx << "] (" << validDim << ")";
    }
  }
  return success();
}

LogicalResult TensorDumpOp::verify() {
  auto srcTile = dyn_cast<pto::TileBufType>(getSrc().getType());
  auto dstPart = dyn_cast<pto::PartitionTensorViewType>(getDst().getType());
  if (!srcTile) {
    return emitOpError("expects src to be !pto.tile_buf");
  }
  if (!dstPart) {
    return emitOpError("expects dst to be !pto.partition_tensor_view");
  }
  if (failed(verifyTensorDumpElemType(*this, srcTile))) {
    return failure();
  }
  const bool sameElementSize = getElemByteSize(srcTile.getElementType()) ==
                               getElemByteSize(dstPart.getElementType());
  if (!sameElementSize) {
    return emitOpError("expects src and dst element types to have the same "
                       "bitwidth: destination strides use the tile element size");
  }
  const auto shape = srcTile.getShape();
  const bool rank2 = shape.size() == kLogicalTileRank &&
                     srcTile.getValidShape().size() == kLogicalTileRank;
  if (!rank2) {
    return emitOpError("expects rank-2 src shape and valid_shape");
  }
  for (int64_t dim : shape) {
    const bool fitsHeader = dim > 0 && static_cast<uint64_t>(dim) <= UINT32_MAX;
    if (!fitsHeader) {
      return emitOpError("expects static positive src shape dimensions that "
                         "fit in the 32-bit dump header");
    }
  }
  auto space = getPTOMemorySpaceEnum(getSrc().getType());
  if (!space || *space != pto::AddressSpace::VEC) {
    return emitOpError("v1 expects a vec-space (UB) tile; "
                       "mat/acc dumps are not yet supported");
  }
  return verifyTensorDumpDstShape(*this, dstPart, srcTile);
}

static LogicalResult verifyTAbsArch(pto::TAbsOp op, PTOArch arch) {
  const bool isA5 = arch == PTOArch::A5;
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileUnaryOp(op, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false,
                                  /*allowInt8=*/isA5)) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  // pto-isa A5 TABS (vabs) supports i8/i16/i32/f16/f32; the A2/A3 TABS clause
  // is float-only. bf16 is in neither clause, so it stays rejected.
  Type elemTy = getElemTy(srcTy);
  // The A5 clause lists the signed 8/16/32-bit forms, not the unsigned ones.
  const bool supported =
      isA5 ? (elemTy.isSignlessInteger(mlir::pto::kValue8) ||
              elemTy.isSignlessInteger(mlir::pto::kValue16) ||
              elemTy.isSignlessInteger(mlir::pto::kValue32) || elemTy.isF16() ||
              elemTy.isF32())
           : (elemTy.isF16() || elemTy.isF32());
  if (!supported) {
    return op.emitOpError()
           << (isA5 ? "expects A5 element type to be i8, i16, i32, f16, or f32"
                    : "expects A2/A3 element type to be f16 or f32");
  }
  return success();
}

LogicalResult pto::TAbsOp::verify() {
  auto verifyA2A3 = [this]() { return verifyTAbsArch(*this, PTOArch::A3); };
  auto verifyA5 = [this]() { return verifyTAbsArch(*this, PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<RankedTensorType, pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType, pto::ConvTileBufType>(ty);
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
  if (auto ct = mlir::dyn_cast<pto::ConvTileBufType>(ty)) {
    return ct.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return tv.getElementType();
  }
  return Type();
}

static SmallVector<int64_t, kShapeVecInlineCapacity> getShapeVec(Type ty) {
  SmallVector<int64_t, kShapeVecInlineCapacity> s;
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return SmallVector<int64_t, kShapeVecInlineCapacity>(tt.getShape().begin(), tt.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return SmallVector<int64_t, kShapeVecInlineCapacity>(tv.getShape().begin(), tv.getShape().end());
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, kShapeVecInlineCapacity>(tb.getShape().begin(), tb.getShape().end());
  }
  if (auto ct = mlir::dyn_cast<pto::ConvTileBufType>(ty)) {
    return SmallVector<int64_t, kShapeVecInlineCapacity>(ct.getShape().begin(), ct.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return SmallVector<int64_t, kShapeVecInlineCapacity>(tv.getShape().begin(), tv.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, kShapeVecInlineCapacity> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, kShapeVecInlineCapacity>(tb.getValidShape().begin(), tb.getValidShape().end());
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
  return dimIdx == packedDim ? rawDim * kFp4PackedDimFactor : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  }
  return std::nullopt;
}

static SmallVector<int64_t, kShapeVecInlineCapacity> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, kShapeVecInlineCapacity> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != kLogicalTileRank) {
    return dims;
  }

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i) {
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  }
  return dims;
}

static SmallVector<int64_t, kShapeVecInlineCapacity> getValidShapeVec(Value value) {
  if (!value) {
    return {};
  }
  auto valid = getValidShapeVec(value.getType());
  return valid;
}

static SmallVector<int64_t, kShapeVecInlineCapacity> getMatmulLogicalShapeVec(Type ty) {
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
  return intTy && intTy.getWidth() == kByteElemBitWidth;
}

static FailureOr<SmallVector<int64_t, kShapeVecInlineCapacity>> getGlobalLikeShape(
    Operation *op, Type ty, StringRef name) {
  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    op->emitOpError()
        << "expects " << name << " to be a tensor_view or partition_view";
    return failure();
  }
  SmallVector<int64_t, kShapeVecInlineCapacity> shape = getShapeVec(ty);
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
