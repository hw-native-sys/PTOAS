// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyMisc.cpp; kept as a fragment included by PTOVerifyMisc.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return std::nullopt;
    if (d < 0)
      return std::nullopt;
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy)
    return std::nullopt;
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return kPTOHalfWordBytes;
    if (ft.isF32())
      return kPTOWordBytes;
    if (ft.isF64())
      return kPTODoubleWordBytes;
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(kPTOByteSize, bits / kPTOByteBitWidth);
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isTileBufOrMemref(Type ty) {
  return mlir::isa<MemRefType, pto::TileBufType>(ty);
}

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value))
    return false;

  if (isa<AllocTileOp, DeclareTileOp, BindTileOp, PointerCastOp,
          MaterializeTileOp>(
          value.getDefiningOp()))
    return true;

  if (auto bitcast = value.getDefiningOp<BitcastOp>())
    return isLocallyBoundTileSource(bitcast.getSrc());
  if (auto reshape = value.getDefiningOp<TReshapeOp>())
    return isLocallyBoundTileSource(reshape.getSrc());

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return cOp.value();
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
    return cInt.value();
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue()))
      return ia.getInt();
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexLike(castOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexLike(truncOp.getIn());
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != kPTORowColRank)
      return emitOpError("expects rank-2 tile_buf source");

    ArrayRef<int64_t> validShape = srcTy.getValidShape();
    if (validShape.size() != kPTORowColRank)
      return emitOpError("expects source validShape to be rank-2");
    if (!srcTy.hasDynamicValid())
      return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");

    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

    if (!isLocallyBoundTileSource(getSource()))
      return emitOpError(
          "requires a locally bound tile source; function arguments/results "
          "are unsupported");
  } else if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (!(*this)->hasAttr(kLoweredSetValidShapeAttrName))
      return emitOpError(
          "expects tile_buf source; memref source is only valid for the internal lowered form");
    if (srcTy.getRank() != kPTORowColRank)
      return emitOpError("expects rank-2 memref source after tile lowering");
    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());
  } else {
    return emitOpError("expects tile_buf source (or lowered memref source)");
  }

  auto checkDim = [this, &shape](Value operand, unsigned dimIdx,
                                 StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal)
      return success();

    if (*constVal < 0)
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic)
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    return success();
  };
  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row")))
    return failure();
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col")))
    return failure();

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != kPTORowColRank)
      return emitOpError("expects rank-2 tile_buf source");
    if (srcTy.getValidShape().size() != kPTORowColRank)
      return emitOpError("expects source validShape to be rank-2");
    return success();
  }
  if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (srcTy.getRank() != kPTORowColRank)
      return emitOpError("expects rank-2 memref source after tile lowering");
    return success();
  }
  return emitOpError("expects tile_buf source (or lowered memref source)");
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb)
    return emitOpError("expects src/result to be !pto.tile_buf types");

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst")))
    return failure();

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace())
    return emitOpError("expects src and dst to use the same loc");

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value())
    return emitOpError("failed to get element byte width for src/dst");

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value())
    return emitOpError("expects static shapes for treshape");

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value())
    return emitOpError("expects src and dst to have the same total byte size");

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed)
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return emitOpError("expects src/result to have the same memorySpace");

  if (srcTy.getElementType() == dstTy.getElementType())
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");

  if (srcTy.getValidShape() != dstTy.getValidShape())
    return emitOpError("expects src/result to have the same validShape");

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg)
    return emitOpError("expects src/result to have the same tile config");

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value())
    return emitOpError("expects static shapes for bitcast");

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value())
    return emitOpError("unsupported element type for bitcast");

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes)
    return emitOpError("bitcast result requires more bytes than source storage");

  return success();
}
