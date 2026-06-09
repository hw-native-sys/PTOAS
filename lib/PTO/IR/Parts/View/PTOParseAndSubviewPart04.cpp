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

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        static_cast<bool>(getTmp()), PTOArch::A3,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        static_cast<bool>(getTmp()), PTOArch::A5,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTRowReductionNoTmpCommon(*this, getSrc().getType(),
                                          getDst().getType(),
                                          "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTRowArgReductionCommon(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTRowArgReductionCommon(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTRowReductionNoTmpCommon(*this, getSrc().getType(),
                                          getDst().getType(),
                                          "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32()))
    return emitOpError("expects element type to be f16 or f32");
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value())
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    if (tmpElemBytes.value() * tmpNumel.value() < kNumber32)
      return emitOpError("expects tmp to be at least 32 bytes when provided");
  }
  return mlir::success();
}

static bool isScatterAllowedDataElem(Type type) {
  if (type.isF16() || type.isF32() || type.isBF16())
    return true;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == kPTOI8BitWidth || intTy.getWidth() == kPTOI16BitWidth ||
           intTy.getWidth() == kPTOI32BitWidth;
  return false;
}

static bool isScatterAllowedIndexElem(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == kPTOI16BitWidth || intTy.getWidth() == kPTOI32BitWidth;
  return false;
}

static unsigned getMaskScatterTimes(MaskPatternAttr pattern) {
  switch (pattern.getValue()) {
  case MaskPattern::P1111:
    return 1;
  case MaskPattern::P0101:
  case MaskPattern::P1010:
    return kNumber2;
  default:
    return kNumber4;
  }
}

static LogicalResult verifyTScatterIndexedForm(TScatterOp op) {
  Type srcTy = op.getSrc().getType();
  Type indexTy = op.getIndexes().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, indexTy, "indexes")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type indexElem = getElemTy(indexTy);
  if (!srcElem || !dstElem || !indexElem)
    return op.emitOpError("failed to get element type for operands");
  if (srcElem != dstElem)
    return op.emitOpError("expects src/dst to have the same element type");
  if (!isScatterAllowedDataElem(srcElem))
    return op.emitOpError(
        "expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  if (!isScatterAllowedIndexElem(indexElem))
    return op.emitOpError("expects indexes element type to be i16/i32");

  auto dataWidth = getPTOStorageElemBitWidth(srcElem);
  auto indexWidth = getPTOStorageElemBitWidth(indexElem);
  if (dataWidth != kPTOI8BitWidth && dataWidth != kPTOI16BitWidth &&
      dataWidth != kPTOI32BitWidth)
    return op.emitOpError("unexpected src/dst element bitwidth");

  unsigned dataBytes = dataWidth / kPTOByteBitWidth;
  unsigned expectedIndexBytes = dataBytes == 1 ? 2 : dataBytes;
  if (indexWidth / kPTOByteBitWidth != expectedIndexBytes) {
    return op.emitOpError(
        "expects indexes element size to match the documented scatter rule");
  }
  return success();
}

static LogicalResult verifyTScatterMaskForm(TScatterOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")))
    return failure();

  auto srcTile = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTile = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTile || !dstTile)
    return op.emitOpError("expects src and dst to be tile_buf types");
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError("expects src and dst to have the same element type");
  if (!isScatterAllowedDataElem(getElemTy(srcTy)))
    return op.emitOpError(
        "expects src/dst element type to be i8/i16/i32/f16/bf16/f32");

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");

  auto pattern = op.getMaskPatternAttr();
  if (!pattern)
    return op.emitOpError(
        "expects mask-pattern tscatter to provide maskPattern");
  const unsigned times = getMaskScatterTimes(pattern);
  if (srcValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && srcValid[0] != dstValid[0])
    return op.emitOpError("expects src and dst to have the same valid rows");
  if (srcValid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != static_cast<int64_t>(dstValid[1] * times)) {
    return op.emitOpError(
        "expects src valid cols to equal dst valid cols times the mask expansion factor");
  }

  if (srcTile.getBLayoutValueI32() !=
          static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTile.getBLayoutValueI32() !=
          static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError(
        "expects mask-pattern tscatter to use row_major blayout");
  }
  return success();
}
