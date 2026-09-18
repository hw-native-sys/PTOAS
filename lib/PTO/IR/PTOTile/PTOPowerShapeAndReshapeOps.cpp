// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

mlir::LogicalResult mlir::pto::TPowSOp::verify() {
  Type dstTy = getDst().getType();
  FailureOr<Type> elem = verifyRowMajorScalarTileCommon(
      getOperation(), getSrc().getType(), dstTy, getScalar().getType());
  if (failed(elem))
    return failure();

  // Same dtype matrix as TPowOp; see comment in TPowOp::verify.
  bool isIntElem = elem->isInteger(mlir::pto::kValue32) || elem->isInteger(mlir::pto::kValue16) ||
                   elem->isInteger(mlir::pto::kValue8);
  if (failed(verifyTPowSElemType(*this, *elem, isIntElem))) {
    return failure();
  }
  return verifyTPowSTmp(*this, dstTy, isIntElem);
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic) {
      return std::nullopt;
    }
    if (d < 0) {
      return std::nullopt;
    }
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy) {
    return std::nullopt;
  }
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16()) {
      return mlir::pto::kValue2;
    }
    if (ft.isF32()) {
        return mlir::pto::kValue4;
    }
    if (ft.isF64()) {
        return mlir::pto::kValue8;
    }
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0) {
      return std::nullopt;
    }
    return std::max<int64_t>(1, bits / mlir::pto::kValue8);
  }
  return std::nullopt;
}

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value)) {
    return false;
  }

  if (isa<AllocTileOp, DeclareTileOp>(value.getDefiningOp())) {
    return true;
  }

  if (auto bitcast = value.getDefiningOp<BitcastOp>()) {
    return isLocallyBoundTileSource(bitcast.getSrc());
  }
  if (auto reshape = value.getDefiningOp<TReshapeOp>()) {
    return isLocallyBoundTileSource(reshape.getSrc());
  }

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return cOp.value();
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    return cInt.value();
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      return ia.getInt();
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndexLike(castOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndexLike(truncOp.getIn());
  }
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != mlir::pto::kValue2) {
      return emitOpError("expects rank-2 tile_buf source");
  }

  ArrayRef<int64_t> validShape = srcTy.getValidShape();
  if (validShape.size() != mlir::pto::kValue2) {
      return emitOpError("expects source validShape to be rank-2");
  }
  if (!srcTy.hasDynamicValid()) {
    return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");
  }

  shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

  if (!isLocallyBoundTileSource(getSource())) {
    return emitOpError(
        "requires a locally bound tile source; function arguments/results "
        "are unsupported");
  }

  auto checkDim = [this, &shape](Value operand, unsigned dimIdx,
                                 StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal) {
      return success();
    }

    if (*constVal < 0) {
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    }
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic) {
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    }
    return success();
  };
  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row"))) {
    return failure();
  }
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col"))) {
    return failure();
  }

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != mlir::pto::kValue2) {
      return emitOpError("expects rank-2 tile_buf source");
  }
  if (srcTy.getValidShape().size() != mlir::pto::kValue2) {
      return emitOpError("expects source validShape to be rank-2");
  }
  return success();
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb) {
    return emitOpError("expects src/result to be !pto.tile_buf types");
  }

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst"))) {
    return failure();
  }

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace()) {
    return emitOpError("expects src and dst to use the same loc");
  }

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value()) {
    return emitOpError("failed to get element byte width for src/dst");
  }

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value()) {
    return emitOpError("expects static shapes for treshape");
  }

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value()) {
    return emitOpError("expects src and dst to have the same total byte size");
  }

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed) {
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");
  }

  return success();
}

static LogicalResult verifyNumericBitcastOperands(Operation *op, Type srcType,
                                                   Type dstType) {
  auto srcVectorType = dyn_cast<VectorType>(srcType);
  auto dstVectorType = dyn_cast<VectorType>(dstType);
  const bool mixesScalarAndVector =
      static_cast<bool>(srcVectorType) != static_cast<bool>(dstVectorType);
  if (mixesScalarAndVector) {
    return op->emitOpError()
           << "requires both numeric types to be scalar or both to be "
              "builtin vectors; got "
           << srcType << " -> " << dstType;
  }
  if (srcVectorType &&
      (srcVectorType.getShape() != dstVectorType.getShape() ||
       srcVectorType.getScalableDims() != dstVectorType.getScalableDims())) {
    return op->emitOpError()
           << "requires numeric vectors to have the same shape; got "
           << srcType << " -> " << dstType;
  }

  Type srcElementType = getElementTypeOrSelf(srcType);
  Type dstElementType = getElementTypeOrSelf(dstType);
  const bool hasUnsupportedElementType =
      !isa<IntegerType, FloatType>(srcElementType) ||
      !isa<IntegerType, FloatType>(dstElementType);
  if (hasUnsupportedElementType) {
    return op->emitOpError()
           << "requires integer or floating-point numeric types; got "
           << srcType << " -> " << dstType;
  }
  const bool hasMismatchedElementWidth =
      srcElementType.getIntOrFloatBitWidth() !=
      dstElementType.getIntOrFloatBitWidth();
  if (hasMismatchedElementWidth) {
    return op->emitOpError() << "requires equal element bit widths; got "
                             << srcType << " -> " << dstType;
  }
  return success();
}

static LogicalResult verifyTileBitcastOperands(Operation *op, TileBufType srcTy,
                                               TileBufType dstTy) {
  if (srcTy.getMemorySpace() != dstTy.getMemorySpace()) {
    return op->emitOpError("expects src/result to have the same memorySpace");
  }
  if (srcTy.getElementType() == dstTy.getElementType()) {
    return op->emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");
  }
  if (srcTy.getShape() != dstTy.getShape()) {
    return op->emitOpError(
        "expects src/result to have the same shape; use pto.treshape for shape changes");
  }
  if (srcTy.getValidShape() != dstTy.getValidShape()) {
    return op->emitOpError("expects src/result to have the same validShape");
  }
  const bool sameConfig = srcTy.getConfigAttr() == dstTy.getConfigAttr();
  if (!sameConfig) {
    return op->emitOpError("expects src/result to have the same tile config");
  }

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value()) {
    return op->emitOpError("expects static shapes for bitcast");
  }
  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value()) {
    return op->emitOpError("unsupported element type for bitcast");
  }
  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes) {
    return op->emitOpError("bitcast result requires more bytes than source storage");
  }
  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  for (StringRef attrName :
       {"fastmath", "roundingmode", "overflowFlags", "signedness"}) {
    if ((*this)->hasAttr(attrName)) {
      return emitOpError() << "does not accept " << attrName;
    }
  }

  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  const bool mixesTileAndNumeric =
      static_cast<bool>(srcTy) != static_cast<bool>(dstTy);
  if (mixesTileAndNumeric) {
    return emitOpError(
        "requires both source and result to be tile buffers or both to be "
        "numeric values");
  }
  if (!srcTy) {
    return verifyNumericBitcastOperands(getOperation(), getSrc().getType(),
                                        getResult().getType());
  }
  return verifyTileBitcastOperands(getOperation(), srcTy, dstTy);
}
