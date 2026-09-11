// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
