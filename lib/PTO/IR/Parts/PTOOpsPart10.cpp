// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

mlir::LogicalResult mlir::pto::TInterleaveOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tinterleave is only supported on A5 targets");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dst0Ty = getDst0().getType();
    Type dst1Ty = getDst1().getType();

    bool invalidTile =
        failed(verifyVecTileCommon(*this, src0Ty, "src0")) ||
        failed(verifyVecTileCommon(*this, src1Ty, "src1")) ||
        failed(verifyVecTileCommon(*this, dst0Ty, "dst0")) ||
        failed(verifyVecTileCommon(*this, dst1Ty, "dst1"));
    if (invalidTile) {
      return failure();
    }

    bool mismatchedElementTypes =
        failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedElementTypes) {
      return failure();
    }
    if (!isSupportedVecElemType(getElemTy(src0Ty), /*allowBf16=*/true,
                                /*allowInt8=*/true)) {
      return emitOpError("expects vec tile element types to be supported");
    }

    bool mismatchedValidShapes =
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedValidShapes) {
      return failure();
    }

    auto validShape = getValidShapeVec(dst0Ty);
    bool hasInvalidRank = validShape.size() != 2;
    if (hasInvalidRank) {
      return emitOpError("expects src0, src1, dst0, and dst1 to have rank-2 valid_shape");
    }
    bool hasOddValidColumns =
        validShape[1] != ShapedType::kDynamic && (validShape[1] & 1) != 0;
    if (hasOddValidColumns) {
      return emitOpError("expects valid_shape[1] to be even");
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTDeInterleaveCommon(TDeInterleaveOp op) {
  Type srcTy = op.getSrc0().getType();
  Type dst0Ty = op.getDst0().getType();
  Type dst1Ty = op.getDst1().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src0")) ||
      failed(verifyVecTileCommon(op, dst0Ty, "dst0")) ||
      failed(verifyVecTileCommon(op, dst1Ty, "dst1")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dst0Ty, "src0", "dst0")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dst1Ty, "src0", "dst1")))
    return failure();
  if (!isSupportedVecElemType(getElemTy(srcTy), true, true))
    return op.emitOpError("expects vec tile element types to be supported");
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dst0Ty) ||
      !isRowMajorTileBuf(dst1Ty))
    return op.emitOpError(
        "expects src and dst tiles to use row-major layout");
  if (getValidShapeVec(srcTy).size() != mlir::pto::kValue2 || getValidShapeVec(dst0Ty).size() != mlir::pto::kValue2 ||
      getValidShapeVec(dst1Ty).size() != mlir::pto::kValue2)
      return op.emitOpError("expects src and dst tiles to have rank-2 valid_shape");
  return success();
}

static LogicalResult verifyTDeInterleaveTwoSources(TDeInterleaveOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  if (failed(verifyVecTileCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, op.getDst0().getType(),
                                         "src0", "dst0")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, op.getDst1().getType(),
                                         "src0", "dst1")))
    return failure();
  if (!isRowMajorTileBuf(src1Ty))
    return op.emitOpError("expects src1 to use row-major layout");
  int64_t columns = getValidShapeVec(src0Ty)[1];
  if (columns != ShapedType::kDynamic && (columns & 1) != 0)
    return op.emitOpError("expects two-source valid_shape[1] to be even");
  return success();
}

static LogicalResult verifyTDeInterleaveSingleOutput(
    TDeInterleaveOp op, ArrayRef<int64_t> srcValid,
    ArrayRef<int64_t> dstValid, StringRef name) {
  if (!hasCompatibleKnownExtent(srcValid[0], dstValid[0]))
    return op.emitOpError() << "expects src0 and " << name
                            << " to have the same valid_shape[0]";
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      dstValid[1] != srcValid[1] / mlir::pto::kValue2)
      return op.emitOpError() << "expects " << name << " valid_shape[1] to be half of src0 valid_shape[1]";
  return success();
}

static LogicalResult verifyTDeInterleaveA5(TDeInterleaveOp op) {
  if (failed(verifyTDeInterleaveCommon(op)))
    return failure();
  if (op.getSrcs().size() == mlir::pto::kValue2)
      return verifyTDeInterleaveTwoSources(op);
  auto srcValid = getValidShapeVec(op.getSrc0());
  if (srcValid[1] != ShapedType::kDynamic && (srcValid[1] & 1) != 0)
    return op.emitOpError("expects single-source valid_shape[1] to be even");
  if (failed(verifyTDeInterleaveSingleOutput(
          op, srcValid, getValidShapeVec(op.getDst0()), "dst0")))
    return failure();
  return verifyTDeInterleaveSingleOutput(
      op, srcValid, getValidShapeVec(op.getDst1()), "dst1");
}

mlir::LogicalResult mlir::pto::TDeInterleaveOp::verify() {
  auto verifyA2A3 = [&]() {
    return emitOpError("tdeinterleave is only supported on A5 targets");
  };
  auto verifyA5 = [&]() { return verifyTDeInterleaveA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A5 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32())) {
    return emitOpError("expects element type to be f16 or f32");
  }
  if (getPrecisionType() == pto::RsqrtPrecision::HighPrecision && !getTmp()) {
    return emitOpError("expects tmp when precisionType is high_precision");
  }
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp"))) {
      return failure();
    }

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value()) {
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    }
    if (tmpElemBytes.value() * tmpNumel.value() < mlir::pto::kValue32) {
        return emitOpError("expects tmp to be at least 32 bytes when provided");
    }
  }
  return mlir::success();
}


static bool isTScatterAllowedDataElem(mlir::Type t) {
  if (t.isF16() || t.isF32() || t.isBF16()) {
    return true;
  }
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      return (
          it.getWidth() == mlir::pto::kValue8 || it.getWidth() == mlir::pto::kValue16 ||
          it.getWidth() == mlir::pto::kValue32);
  }
  return false;
}

static bool isTScatterAllowedIndexElem(mlir::Type t) {
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      return (it.getWidth() == mlir::pto::kValue16 || it.getWidth() == mlir::pto::kValue32);
  }
  return false;
}

static LogicalResult verifyTScatterIndexedElemTypes(TScatterOp op, Type ts,
                                                    Type ti, Type td) {
  Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
  if (!srcElem || !dstElem || !idxElem) {
    return op.emitOpError("failed to get element type for operands");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src/dst to have the same element type");
  }

  if (!isTScatterAllowedDataElem(srcElem)) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  if (!isTScatterAllowedIndexElem(idxElem)) {
    return op.emitOpError("expects indexes element type to be i16/i32");
  }

  auto bwData = getPTOStorageElemBitWidth(srcElem);
  auto bwIdx  = getPTOStorageElemBitWidth(idxElem);
  if (bwData != mlir::pto::kValue8 && bwData != mlir::pto::kValue16 && bwData != mlir::pto::kValue32) {
      return op.emitOpError("unexpected src/dst element bitwidth");
  }

  unsigned dataBytes = bwData / 8;
  unsigned idxBytes  = bwIdx / 8;
  unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
  if (idxBytes != expectedIdxBytes) {
    return op.emitOpError("expects indexes element size to match the documented scatter rule");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedShapes(TScatterOp op, Type ts, Type ti,
                                                 Type td) {
  auto srcValid = getValidShapeVec(ts);
  auto idxValid = getValidShapeVec(ti);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != mlir::pto::kValue2 || idxValid.size() != mlir::pto::kValue2 ||
      dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects src, indexes and dst to have rank-2 valid_shape");
  }

  for (unsigned d = 0; d < mlir::pto::kValue2; ++d) {
      if (srcValid[d] != ShapedType::kDynamic && idxValid[d] != ShapedType::kDynamic && srcValid[d] != idxValid[d]) {
          return op.emitOpError("expects src and indexes to have the same valid_shape");
      }
      if (srcValid[d] != ShapedType::kDynamic && dstValid[d] != ShapedType::kDynamic && dstValid[d] < srcValid[d]) {
          return op.emitOpError("expects dst valid_shape to be >= src valid_shape");
      }
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type ti = op.getIndexes().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, ti, "indexes")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }
  if (failed(verifyTScatterIndexedElemTypes(op, ts, ti, td))) {
    return failure();
  }
  return verifyTScatterIndexedShapes(op, ts, ti, td);
}

static LogicalResult verifyTScatterMaskAxisShapes(TScatterOp op, StringRef axisVal,
                                                  ArrayRef<int64_t> srcValid,
                                                  ArrayRef<int64_t> dstValid,
                                                  unsigned times) {
  if (axisVal == "row") {
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != srcValid[0]) {
      return op.emitOpError("expects dst valid rows to equal src valid rows for row direction");
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != static_cast<int64_t>(srcValid[1] * times)) {
      return op.emitOpError("expects dst valid cols to equal src valid cols times the mask expansion factor for row direction");
    }
  } else if (axisVal == "col") {
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != srcValid[1]) {
      return op.emitOpError("expects dst valid cols to equal src valid cols for col direction");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != static_cast<int64_t>(srcValid[0] * times)) {
      return op.emitOpError("expects dst valid rows to equal src valid rows times the mask expansion factor for col direction");
    }
  } else {
      return op.emitOpError("Invalid axis value, expected \"row\" or \"col\"");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterMaskForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }

  auto srcTB = dyn_cast<pto::TileBufType>(ts);
  auto dstTB = dyn_cast<pto::TileBufType>(td);
  if (!srcTB || !dstTB) {
    return op.emitOpError("expects src and dst to be tile_buf types");
  }

  if (getElemTy(ts) != getElemTy(td)) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (!isTScatterAllowedDataElem(getElemTy(ts))) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }

  auto srcValid = getValidShapeVec(ts);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }

  auto axisAttr = op.getAxisAttr();
  if (!axisAttr) {
    return op.emitOpError("expects mask-pattern tscatter to provide axis attribute");
  }
  StringRef axisVal = axisAttr.getValue();
  auto mp = op.getMaskPatternAttr();
  if (!mp) {
    return op.emitOpError("expects mask-pattern tscatter to provide maskPattern");
  }
  const unsigned times = getMaskGatherTimes(mp);
  if (failed(verifyTScatterMaskAxisShapes(op, axisVal, srcValid, dstValid, times))) {
    return failure();
  }

  if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError("expects mask-pattern tscatter to use row_major blayout");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }
  if (hasIndexes && getAxisAttr()) {
    return emitOpError("axis attribute must not be provided with indexes operand");
  }
  auto verifyForm = [&]() -> LogicalResult {
    if (hasMaskPattern) {
      return verifyTScatterMaskForm(*this);
    }
    return verifyTScatterIndexedForm(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyForm, verifyForm);
}


static FailureOr<Type> verifyTSelCommon(TSelOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  FailureOr<Type> elem = verifyThreeMatchingTiles(op, t0, t1, td, tmpTy);
  if (failed(elem))
    return failure();

  if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
      !isRowMajorTileBuf(td)) {
    op.emitOpError(
        "expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return *elem;
}

static LogicalResult verifyTSelA2A3(TSelOp op) {
  FailureOr<Type> srcElem = verifyTSelCommon(op);
  if (failed(srcElem)) {
    return failure();
  }
  Type elem = *srcElem;
  bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
  if (auto it = dyn_cast<IntegerType>(elem)) {
      ok = it.getWidth() == mlir::pto::kValue16 || it.getWidth() == mlir::pto::kValue32;
  }
  if (!ok) {
    return op.emitOpError(
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32");
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (getElemByteSize(getElemTy(tmpTy)) != mlir::pto::kValue4) {
        return op.emitOpError("expects A2/A3 tsel tmp element type to be 4 bytes wide");
    }
    unsigned elemBits = getPTOStorageElemBitWidth(elem);
    if (elemBits != 16 && elemBits != 32) {
      return op.emitOpError("expects A2/A3 tsel data element type to be 16 or 32 bits");
    }
    uint64_t minBytes = elemBits == 16 ? 16 : 8;
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, minBytes))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTSelA5(TSelOp op) {
  FailureOr<Type> srcElem = verifyTSelCommon(op);
  if (failed(srcElem)) {
    return failure();
  }
  Type elem = *srcElem;
  bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
  if (auto it = dyn_cast<IntegerType>(elem)) {
      ok = it.getWidth() == mlir::pto::kValue8 || it.getWidth() == mlir::pto::kValue16 ||
           it.getWidth() == mlir::pto::kValue32;
  }
  if (!ok) {
    return op.emitOpError(
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTSelA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTSelA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static FailureOr<Type> verifyTSelSCommon(TSelSOp op) {
  Type tMask = op.getMask().getType();
  Type tSrc = op.getSrc().getType();
  Type tTmp = op.getTmp() ? op.getTmp().getType() : Type{};
  Type tDst = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, tMask, "mask")) ||
      failed(verifyTileBufCommon(op, tSrc, "src")) ||
      failed(verifyTileBufCommon(op, tDst, "dst"))) {
    return failure();
  }
  if (tTmp && failed(verifyTileBufCommon(op, tTmp, "tmp"))) {
    return failure();
  }
  Type eMask = getElemTy(tMask), eSrc = getElemTy(tSrc);
  Type eDst = getElemTy(tDst);
  if (!eMask || !eSrc || !eDst) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (eSrc != eDst) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyTileBufSameValidShape(op, tSrc, tDst, "src", "dst"))) {
    return failure();
  }
  return eDst;
}

static LogicalResult verifyTSelSA2A3(TSelSOp op) {
  FailureOr<Type> elemOr = verifyTSelSCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  Type tSrc = op.getSrc().getType();
  Type tDst = op.getDst().getType();
  if (!isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst)) {
    return op.emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError("expects A2/A3 tsels tmp to have the same element type as src and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects A2/A3 tsels tmp to use row-major layout");
    }
    auto srcShape = getShapeVec(tSrc);
    if (srcShape.size() != mlir::pto::kValue2 || srcShape[1] == ShapedType::kDynamic) {
        return op.emitOpError("expects A2/A3 tsels src shape to be static when tmp is provided");
    }
    auto elemBytes = getElemByteSize(elem);
    if (elemBytes == 0 ||
        failed(verifyTmpCapacityAtLeast(
            op, tmpTy, static_cast<uint64_t>(srcShape[1]) * elemBytes))) {
      return failure();
    }
  }
  bool ok = elem.isF16() || elem.isF32();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
      ok = (it.getWidth() == mlir::pto::kValue16 || it.getWidth() == mlir::pto::kValue32);
  }
  if (!ok) {
    return op.emitOpError(
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32");
  }
  return success();
}
static LogicalResult verifyTSelSA5(TSelSOp op) {
  FailureOr<Type> elemOr = verifyTSelSCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  Type tMask = op.getMask().getType();
  Type tSrc = op.getSrc().getType();
  Type tDst = op.getDst().getType();
  if (!isRowMajorTileBuf(tMask) || !isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst)) {
    return op.emitOpError("expects mask, src, and dst to use row-major layout");
  }
  Type elem = *elemOr;
  bool ok = elem.isF16() || elem.isF32();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
      ok =
          (it.getWidth() == mlir::pto::kValue8 || it.getWidth() == mlir::pto::kValue16 ||
           it.getWidth() == mlir::pto::kValue32);
  }
  if (!ok) {
    return op.emitOpError(
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels:
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTSelSA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTSelSA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyShiftOp(Operation *op, Type src0Ty, Type src1Ty,
                                   Type dstTy, StringRef name) {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyShiftLikeBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
                it.getWidth() != mlir::pto::kValue32)) {
        return op->emitOpError() << "expects " << name << " src0 and src1 element type to be i8/i16/i32";
    }
    return success();
  };
  return dispatchVerifierByArch(op, verify, verify);
}

mlir::LogicalResult mlir::pto::TShlOp::verify() {
  return verifyShiftOp(getOperation(), getSrc0().getType(), getSrc1().getType(),
                       getDst().getType(), "tshl");
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  return verifyShiftOp(getOperation(), getSrc0().getType(), getSrc1().getType(),
                       getDst().getType(), "tshr");
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx"))) {
    return failure();
  }
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5) {
    auto requiredBytes = getStaticByteSize(srcTy);
    if (!requiredBytes) {
      return emitOpError(
          "expects A2/A3 tsort32 src shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(*this, getTmp().getType(),
                                        *requiredBytes))) {
      return failure();
    }
  }

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!(srcElem.isF16() || srcElem.isF32())) {
    return emitOpError() << "expects src and dst element type to be f16 or f32";
  }

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != mlir::pto::kValue32) {
      return emitOpError() << "expects idx element type to be i32/u32";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem))) {
    return emitOpError() << "expects src and dst element type to be float or half";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size()) {
    return emitOpError() << "expects all tensors to have the same rank";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size()) {
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  }
  return mlir::success();
}
static bool ttransUsesTmp(Type srcTy, Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  unsigned elemBytes = getPTOStorageElemByteSize(getElemTy(srcTy));
  if (srcShape.size() != mlir::pto::kValue2 || dstShape.size() != mlir::pto::kValue2 || elemBytes == 0 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) || llvm::is_contained(dstShape, ShapedType::kDynamic)) {
      return true;
  }
  int64_t rowStride = elemBytes == 1 ? 32 : 16;
  int64_t elemPerBlock = 32 / elemBytes;
  int64_t srcStride = srcShape[1];
  int64_t dstStride = dstShape[1];
  return dstStride % rowStride == 0 && srcStride % elemPerBlock == 0 &&
         srcStride / elemPerBlock <= mlir::pto::kValue255;
}

static bool ttransIsAllowedWidthType(Type ty, unsigned elemBytes) {
    if (elemBytes == mlir::pto::kValue4) {
        return ty.isInteger(mlir::pto::kValue32) || ty.isF32();
    }
    if (elemBytes == mlir::pto::kValue2) {
        return ty.isInteger(mlir::pto::kValue16) || ty.isF16() || ty.isBF16();
    }
    return ty.isInteger(mlir::pto::kValue8);
}

static LogicalResult verifyTTransElemWidth(TTransOp op, Type srcElem,
                                           unsigned &elemBytes) {
  elemBytes = getPTOStorageElemByteSize(srcElem);
  if (elemBytes == 0) {
    return op.emitOpError() << "failed to get transpose element size";
  }
  if (elemBytes != 1 && elemBytes != mlir::pto::kValue2 && elemBytes != mlir::pto::kValue4) {
      return op.emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
  }
  if (!ttransIsAllowedWidthType(srcElem, elemBytes)) {
    return op.emitOpError() << "expects transpose element type to match the supported set for its width";
  }
  return success();
}

static LogicalResult verifyTTransAlignedMajor(TTransOp op, Type ty, StringRef name,
                                              unsigned elemBytes) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return success();
  }
  auto shape = getShapeVec(ty);
  if (shape.size() != mlir::pto::kValue2) {
      return success();
  }
  bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[1] : shape[0];
  if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % mlir::pto::kValue32 != 0) {
      return op.emitOpError() << "expects " << name
                              << " major dimension times element size to be 32-byte aligned on A5";
  }
  return success();
}

struct TTransTypes {
  Type src;
  Type tmp;
  Type dst;
  Type elem;
  unsigned elemBytes;
};

static FailureOr<TTransTypes> verifyTTransCommon(TTransOp op,
                                                 StringRef typeError) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (tmpTy && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = tmpTy ? getElemTy(tmpTy) : srcElem;
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem) {
    op.emitOpError() << typeError;
    return failure();
  }
  unsigned elemBytes = 0;
  if (failed(verifyTTransElemWidth(op, srcElem, elemBytes))) {
    return failure();
  }
  return TTransTypes{srcTy, tmpTy, dstTy, srcElem, elemBytes};
}

static LogicalResult verifyTTransA2A3(TTransOp op) {
  auto types = verifyTTransCommon(
      op, "expects src and dst to have the same element type");
  if (failed(types))
    return failure();
  Type srcTy = types->src;
  Type tmpTy = types->tmp;
  Type dstTy = types->dst;
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op.emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
  }
  if (tmpTy) {
    uint64_t requiredBytes = 32;
    if (ttransUsesTmp(srcTy, dstTy)) {
      auto srcBytes = getStaticByteSize(srcTy);
      if (!srcBytes) {
        return op.emitOpError(
            "expects A2/A3 transpose src shape to be static when tmp is used");
      }
      requiredBytes = *srcBytes;
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, requiredBytes))) {
      return failure();
    }
  }
  return mlir::success();
}

static LogicalResult verifyTTransA5(TTransOp op) {
  auto types = verifyTTransCommon(
      op, "expects src, tmp, and dst to have the same element type");
  if (failed(types))
    return failure();
  Type srcTy = types->src;
  Type tmpTy = types->tmp;
  Type dstTy = types->dst;
  unsigned elemBytes = types->elemBytes;
  if (tmpTy && failed(verifyTmpCapacityAtLeast(op, tmpTy, mlir::pto::kValue32))) {
      return failure();
  }
  if (failed(verifyTTransAlignedMajor(op, srcTy, "src", elemBytes)) ||
      failed(verifyTTransAlignedMajor(op, dstTy, "dst", elemBytes))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTTransA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTTransA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TXorOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/true,
                                     /*addSegmentSizes=*/true);
}

void mlir::pto::TXorOp::print(OpAsmPrinter &p) {
  printOptionalTmpBinaryDpsOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TXorSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/true,
                                     /*addSegmentSizes=*/true);
}

void mlir::pto::TXorSOp::print(OpAsmPrinter &p) {
  printOptionalTmpBinaryDpsOp(p, getOperation(), getSrc(), getScalar(),
                              getTmp(), getDst());
}

static LogicalResult verifyTXorA2A3(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError(
          "expects tmp to have the same element type as src0, src1, and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects tmp to use row-major layout");
    }
    if (failed(verifyTileBufSameValidShape(
            op, tmpTy, op.getDst().getType(), "tmp", "dst"))) {
      return failure();
    }
    auto requiredBytes = getStaticByteSize(op.getDst().getType());
    if (!requiredBytes) {
      return op.emitOpError(
          "expects A2/A3 txor dst shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
      return failure();
    }
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
              it.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16/i32");
  }
  return success();
}

static LogicalResult verifyTXorA5(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  auto it = mlir::dyn_cast<IntegerType>(*elemOr);
  if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
              it.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTXorA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTXorA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTXorSTmp(TXorSOp op, Type elem) {
  if (!op.getTmp())
    return success();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (getElemTy(tmpTy) != elem)
    return op.emitOpError(
            "expects tmp to have the same element type as src and dst");
  if (!isRowMajorTileBuf(tmpTy))
    return op.emitOpError("expects tmp to use row-major layout");
  auto requiredBytes = getStaticByteSize(op.getDst().getType());
  if (!requiredBytes)
    return op.emitOpError(
            "expects A2/A3 txors dst shape to be static when tmp is provided");
  return verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes);
}

static LogicalResult verifyTXorSArch(TXorSOp op, bool isA5) {
  auto elemResult = verifyDistinctRowMajorUnaryTileOpCommon(
      op, op.getSrc(), op.getDst(), "src", "dst");
  if (failed(elemResult))
    return failure();
  Type elem = *elemResult;
  if (!isA5 && failed(verifyTXorSTmp(op, elem)))
    return failure();
  auto integer = dyn_cast<IntegerType>(elem);
  bool supported = integer &&
                   (integer.getWidth() == 8 || integer.getWidth() == 16 ||
                    (isA5 && integer.getWidth() == 32));
  if (!supported)
    return op.emitOpError(isA5
        ? "expects A5 txors src and dst element type to be i8/i16/i32"
        :
          "expects A2/A3 txors src and dst element type to be i8/i16");
  return success();
}

mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyA2A3 = [&]() { return verifyTXorSArch(*this, false); };
  auto verifyA5 = [&]() { return verifyTXorSArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TPrintOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  Type srcTy, tmpTy;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

  if (failed(parseOptionalTmpIns(parser, src, tmp, srcTy, tmpTy, hasTmp))) {
    return failure();
  }
  if (failed(parsePTOInherentAttrs<TPrintOp>(
          parser, result, parsedAttrs, {"printFormat"}))) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPrintOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (Value tmp = getTPrintTmpIfPresent(*this)) {
    p << ", " << tmp << " : " << getSrc().getType() << ", "
      << tmp.getType();
  } else {
    p << " : " << getSrc().getType();
  }
  p << ")";
  NamedAttrList attrs = getNonInherentAttrs(getOperation(), {"printFormat"});
  if (auto printFormatAttr =
          dyn_cast_or_null<pto::PrintFormatAttr>(getProperties().printFormat)) {
    attrs.append("printFormat", printFormatAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  auto srcType = getSrc().getType();
  Value tmp = getTPrintTmpIfPresent(*this);
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(mlir::pto::kValue8) || elem.isInteger(mlir::pto::kValue16) ||
          elem.isInteger(mlir::pto::kValue32))) {
        return emitOpError() << "expects printable tile element type";
    }
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!tmp) {
      if (!space || *space != pto::AddressSpace::VEC) {
        return emitOpError() << "expects printable tile_buf without tmp to be in vec address space";
}
      return success();
    }

    if (!space) {
      return emitOpError() << "expects printable tile_buf with tmp to use a supported address space";
}
    if (*space == pto::AddressSpace::MAT && isTargetArchA5(getOperation())) {
      return emitOpError() << "expects mat tile printing with tmp only on A2/A3 targets";
}
    if (*space != pto::AddressSpace::VEC && *space != pto::AddressSpace::MAT &&
        *space != pto::AddressSpace::ACC) {
      return emitOpError() << "expects printable tile_buf with tmp to be in vec/mat/acc address space";
}
    if (failed(verifyMGatherMScatterMemOperand(getOperation(), tmp, elem, "tmp"))) {
      return failure();
}
    return success();
  }
  if (tmp) {
    return emitOpError() << "expects tmp only when src is a tile_buf";
}
  if (mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType)) {
    return mlir::success();
  }
  return emitOpError() << "expects tile_buf or partition_tensor_view for src";
}

static LogicalResult verifyMatmulOrGemv(Operation *op, Type lhs, Type rhs,
                                        Type dst, bool isGemv,
                                        bool allowLowPrecision = false) {
  LogicalResult operands =
      isGemv ? verifyGemvTileOperands(op, lhs, rhs, dst)
             : verifyMatTileOperands(op, lhs, rhs, dst, allowLowPrecision);
  if (failed(operands) ||
      failed(verifyMatmulTypeTriple(op, getElemTy(lhs), getElemTy(rhs),
                                    getElemTy(dst))))
    return failure();
  return verifyMatmulLike(op, lhs, rhs, dst);
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulOrGemv(getOperation(), getLhs().getType(),
                              getRhs().getType(), getDst().getType(), false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyMatmulOrGemv(getOperation(), getLhs().getType(),
                              getRhs().getType(), getDst().getType(), false,
                              /*allowLowPrecision=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulOrGemv(getOperation(), getLhs().getType(),
                              getRhs().getType(), getDst().getType(), true);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType()))) {
      return failure();
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
    if (operands.size() < mlir::pto::kValue2) {
        return mlir::Type();
    }

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile) {
    return mlir::Type();
  }

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= mlir::pto::kValue2 && rhsShape.size() >= mlir::pto::kValue2) {
      int64_t M = lhsShape[0];
      int64_t N = rhsShape[1];
      llvm::SmallVector<int64_t, mlir::pto::kValue2> outShape = {M, N};
      return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
    if (operands.size() < mlir::pto::kValue2) {
        return RankedTensorType();
    }

  auto lhsTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhsTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!lhsTy || !rhsTy) {
    return RankedTensorType();
  }

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType())) {
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= mlir::pto::kValue2 && rhsTy.getRank() >= mlir::pto::kValue2) {
      int64_t M = lhsTy.getDimSize(0);
      int64_t N = rhsTy.getDimSize(1);
      return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty()) {
    return RankedTensorType();
  }
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType())) {
    return accRT;
  }
  return RankedTensorType();
}

namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess()) {
    return failure();
  }

  if (parser.parseDimensionList(shape, allowDynamic)) {
    return failure();
  }

  if (parser.parseType(elementType)) {
    return failure();
  }

  if (parser.parseGreater()) {
    return failure();
  }

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << d;
}
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

static LogicalResult parseViewShapeElemAndLayout(
    AsmParser &parser, SmallVectorImpl<int64_t> &shape, Type &elementType,
    Attribute &layout, bool allowDynamic) {
  bool parseFailed =
      parser.parseLess() || parser.parseDimensionList(shape, allowDynamic) ||
      parser.parseType(elementType);
  if (parseFailed) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    LayoutAttr layoutAttr;
    if (parser.parseAttribute(layoutAttr)) {
      return failure();
    }
    layout = layoutAttr;
  }
  return parser.parseGreater();
}

static void printViewShapeElemAndLayout(AsmPrinter &printer,
                                        ArrayRef<int64_t> shape,
                                        Type elementType, Attribute layout) {
  printer << "<";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << dim;
    }
    printer << "x";
  }
  printer.printType(elementType);
  if (layout) {
    printer << ", ";
    printer.printAttribute(layout);
  }
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elemTy;
    Attribute layout;
    if (failed(parseViewShapeElemAndLayout(parser, shape, elemTy, layout, /*allowDynamic=*/true))) {
        return Type();
    }

  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy,
                                      layout);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printViewShapeElemAndLayout(printer, getShape(), getElementType(),
                              getLayout());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elemTy;
    if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true))) {
        return Type();
    }
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &parser) {
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elemTy;
    if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false))) {
        return Type();
    }
  return LocalArrayType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty()) {
    return emitError() << "'!pto.local_array' requires at least one dimension";
  }
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0) {
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
    }
  }
  if (!elementType.isIntOrFloat()) {
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  }
  return success();
}

// ---- StructType ----
// Asm form: !pto.struct<T0, T1, ..., Tn-1>
// A field type must be "scalar-storable": an exactly-nameable scalar or a
// nested !pto.struct (see the two predicates below). The allowlist deliberately
// excludes the vec/cube types (tile_buf / tensor_view / partition view) and any
// other handle type, keeping the scalar struct world disjoint from the
// fractal/layout world.

// A struct field's scalar type must map onto a C++ scalar that the backend can
// name exactly: integers of width 8/16/32/64 and f16/bf16/f32/f64. Widths the
// backend has no spelling for (i1, i24, ...) and the packed low-precision
// vec/cube formats (f8/f4 variants) would otherwise be emitted as `float`,
// silently changing the field's width and semantics, so reject them here.
static bool isStructScalar(Type t) {
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(t)) {
    return true;
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(t)) {
    unsigned w = intTy.getWidth();
    return w == mlir::pto::kValue8 || w == mlir::pto::kValue16 || w == mlir::pto::kValue32 || w == mlir::pto::kValue64;
  }
  return false;
}

// A field is either such a scalar or a nested !pto.struct. !pto.local_array is
// deliberately NOT allowed: a field is reached with `emitc.member`, whose result
// must be an `!emitc.lvalue`, and `!emitc.lvalue` cannot wrap `!emitc.array`
// (the type an array field lowers to). There is no way to spell the access, so
// the restriction is enforced here rather than failing later in the backend.
static bool isStructStorable(Type t) {
  return isStructScalar(t) || llvm::isa<StructType>(t);
}

Type StructType::parse(AsmParser &parser) {
  SmallVector<Type> fields;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            Type t;
            if (parser.parseType(t)) {
              return failure();
            }
            fields.push_back(t);
            return success();
          })) {
    return Type();
  }
  return StructType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), fields);
}

void StructType::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::ArrayRef<Type> fields = getFieldTypes();
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer.printType(fields[i]);
  }
  printer << ">";
}
