// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTQuantMxSrcPadding(TQuantMxOp op,
                                              const TQuantMxA5 &s) {
  if ((s.srcElem.isF16() || s.srcElem.isBF16()) &&
      s.srcCols < s.srcPhysicalCols && 128 % s.srcPhysicalCols == 0) {
    auto validPhysicalElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
    if (!validPhysicalElems) {
      return op.emitOpError("cannot compute B16 source padding extent without overflow");
    }
    if (*validPhysicalElems % 128 != 0) {
      return op.emitOpError("does not support padded B16 source whose VL-aligned padding store has an incomplete final VL");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxA5(TQuantMxOp op) {
  if (failed(verifyTQuantMxTilesAndForm(op)) ||
      failed(verifyTQuantMxElemTypes(op)) || failed(verifyTQuantMxShapes(op))) {
    return failure();
  }
  auto stateOr = buildTQuantMxA5State(op);
  if (failed(stateOr)) {
    return failure();
  }
  const TQuantMxA5 &s = *stateOr;
  if (failed(verifyTQuantMxGrouping(op, s)) ||
      failed(verifyTQuantMxDstShape(op, s))) {
    return failure();
  }
  if (s.isDn) {
    if (failed(verifyTQuantMxAxis0(op, s))) {
      return failure();
    }
  } else {
    if (failed(verifyTQuantMxAxis1(op, s))) {
      return failure();
    }
  }
  return verifyTQuantMxSrcPadding(op, s);
}

mlir::LogicalResult mlir::pto::TQuantMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tquant.mx is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantMxA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16)) {
      return emitOpError()
             << "expects src element type i8 or i16";
    }
    if (!getElemTy(getDst().getType()).isF32()) {
      return emitOpError() << "expects dst element type f32";
    }
    if (!getElemTy(getScale().getType()).isF32()) {
      return emitOpError() << "expects scale element type f32";
    }
    if (!getElemTy(getOffset().getType()).isF32()) {
      return emitOpError() << "expects offset element type f32";
    }
    return success();
  };

  if (failed(verifyStructural())) {
    return failure();
  }

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst"))) {
      return failure();
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (failed(verifyF16F32VecUnary(getOperation(), getSrc().getType(),
                                  getDst().getType())))
    return failure();
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst()) {
    return emitOpError("expects A3 trecip src and dst to use different storage");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << errorMessage;
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRemNoTmp(TRemOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trem element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemTmpA2A3(TRemOp op, Type tmpTy, Type elem) {
  Type dstTy = op.getDst().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 2) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 2");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(dstTy);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trem dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tmpTy, static_cast<uint64_t>(2) * static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmpA5(TRemOp op, Type tmpTy, Type elem) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmp(TRemOp op, Type elem) {
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemTmpA2A3(op, tmpTy, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemTmpA5(op, tmpTy, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src0, src1, and dst to use row-major layout");
  }

  Type elem = getElemTy(src0Ty);
  if (!getTmp()) {
    return verifyTRemNoTmp(*this, elem);
  }
  return verifyTRemTmp(*this, elem);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

static LogicalResult verifyTRemSNoTmp(TRemSOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trems element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}
