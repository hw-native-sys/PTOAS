// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyTTransA2A3Constraints(TTransOp op,
                                                 const TTransVerifyState &state) {
  auto srcTile = dyn_cast<pto::TileBufType>(state.srcTy);
  if (!srcTile)
    return success();
  if (srcTile.getBLayoutValueI32() !=
      static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError()
           << "expects A2/A3 transpose src to use the row_major blayout";
  }
  return success();
}

static LogicalResult verifyTTransA5MajorAlignment(TTransOp op, Type type,
                                                  unsigned elemBytes,
                                                  StringRef name) {
  auto tile = dyn_cast<pto::TileBufType>(type);
  if (!tile)
    return success();
  auto shape = getShapeVec(type);
  if (shape.size() != kPTORowColRank)
    return success();
  bool rowMajor =
      tile.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[1] : shape[0];
  if (major != ShapedType::kDynamic &&
      (major * static_cast<int64_t>(elemBytes)) % kNumber32 != 0) {
    return op.emitOpError()
           << "expects " << name
           << " major dimension times element size to be 32-byte aligned on A5";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    FailureOr<TTransVerifyState> stateOr =
        verifyTTransCommon(*this, "expects src and dst to have the same element type");
    if (failed(stateOr))
      return failure();
    return verifyTTransA2A3Constraints(*this, *stateOr);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    FailureOr<TTransVerifyState> stateOr = verifyTTransCommon(
        *this, "expects src, tmp, and dst to have the same element type");
    if (failed(stateOr))
      return failure();
    if (failed(verifyTTransA5MajorAlignment(*this, stateOr->srcTy,
                                           stateOr->elemBytes, "src")) ||
        failed(verifyTTransA5MajorAlignment(*this, stateOr->dstTy,
                                           stateOr->elemBytes, "dst")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyBase = [this]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [this, &verifyBase]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    Type elem = *elemOr;
    if (getElemTy(tmpTy) != elem)
      return emitOpError("expects tmp to have the same element type as src0, src1, and dst");
    if (!isRowMajorTileBuf(tmpTy))
      return emitOpError("expects tmp to use row-major layout");
    if (failed(verifyTileBufSameValidShape(*this, tmpTy, getDst().getType(), "tmp", "dst")))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth))
      return emitOpError(
          "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [this, &verifyBase]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth &&
                it.getWidth() != kPTOI32BitWidth))
      return emitOpError(
          "expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyCommon = [this]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };
  return verifyArchIntegerWidthOp(
      getOperation(), verifyCommon,
      "expects A2/A3 txors src and dst element type to be i8/i16",
      "expects A5 txors src and dst element type to be i8/i16/i32");
}
mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcType = getSrc().getType();
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(kPTOI8BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth)))
      return emitOpError() << "expects printable tile element type";
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!space || *space != pto::AddressSpace::VEC)
      return emitOpError() << "expects printable tile_buf to be in vec address space";
    return success();
  }
  if (mlir::dyn_cast<MemRefType>(srcType) ||
      mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType))
    return mlir::success();
  return emitOpError() << "expects tile_buf, memref, or partition_tensor_view for src";
}

static LogicalResult verifyMatmulShapedCommon(Operation *op, ShapedType lhsTy,
                                              Value rhs, Value biasOpt,
                                              Type maybeDstElemTy,
                                              Type maybeResultElemTy) {
  auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
  if (!rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return op->emitOpError("expects lhs and rhs to be ranked tensors or memrefs");

  if (lhsTy.getElementType() != rhsTy.getElementType()) {
    return op->emitOpError()
           << "expects lhs and rhs to have the same element type, but got lhs="
           << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();
  }

  if (biasOpt) {
    auto biasTy = dyn_cast<ShapedType>(biasOpt.getType());
    if (!biasTy || !biasTy.hasRank())
      return op->emitOpError("expects bias to be a ranked tensor or memref");
    if (biasTy.getElementType() != lhsTy.getElementType()) {
      return op->emitOpError()
             << "expects bias to have the same element type as lhs and rhs, but got bias="
             << biasTy.getElementType() << " vs " << lhsTy.getElementType();
    }
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType()) {
    return op->emitOpError()
           << "expects dst to have the same element type as lhs and rhs, but got dst="
           << maybeDstElemTy << " vs " << lhsTy.getElementType();
  }
  if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType()) {
    return op->emitOpError()
           << "expects result to have the same element type as lhs and rhs, but got result="
           << maybeResultElemTy << " vs " << lhsTy.getElementType();
  }
  return success();
}

static LogicalResult verifyMatmulTileCommon(Operation *op, TileType lhsTile,
                                            Value rhs, Value biasOpt,
                                            Type maybeDstElemTy,
                                            Type maybeResultElemTy) {
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!rhsTile) {
    return op->emitOpError(
        "expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");
  }
  if (lhsTile.getElementType() != rhsTile.getElementType()) {
    return op->emitOpError()
           << "expects lhs and rhs tiles to have the same element type, but got lhs="
           << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();
  }
  if (static_cast<int64_t>(lhsTile.getShape().size()) != kPTORowColRank ||
      static_cast<int64_t>(rhsTile.getShape().size()) != kPTORowColRank) {
    return op->emitOpError("expects lhs and rhs tiles to be 2D");
  }
  if (lhsTile.getShape()[1] != rhsTile.getShape()[0]) {
    return op->emitOpError()
           << "expects lhs dim1 to equal rhs dim0, but got "
           << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];
  }

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile)
      return op->emitOpError(
          "expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    if (biasTile.getElementType() != lhsTile.getElementType()) {
      return op->emitOpError(
          "expects bias to have the same element type as lhs and rhs");
    }
  }
  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType())
    return op->emitOpError()
           << "expects dst to have the same element type as lhs and rhs";
  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType())
    return op->emitOpError()
           << "expects result to have the same element type as lhs and rhs";
  return success();
}
