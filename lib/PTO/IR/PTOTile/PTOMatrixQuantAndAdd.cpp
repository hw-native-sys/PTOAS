// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyA5MxMatScaleTile(Operation *op, Type scaleTy,
                                            Type lhsTy, Type rhsTy,
                                            StringRef scaleName,
                                            bool isLeftScale) {
  if (failed(verifyA5MxScaleStorage(op, scaleTy, scaleName)))
    return failure();

  if (failed(verifyA5MxMatScaleDims(op, scaleName, getShapeVec(scaleTy),
                                    getShapeVec(lhsTy), getShapeVec(rhsTy),
                                    "shape", isLeftScale))) {
    return failure();
  }
  if (failed(verifyA5MxMatScaleDims(
          op, scaleName, getValidShapeVec(scaleTy), getValidShapeVec(lhsTy),
          getValidShapeVec(rhsTy), "valid_shape", isLeftScale))) {
    return failure();
  }

  auto scaleTb = dyn_cast<pto::TileBufType>(scaleTy);
  if (!scaleTb) {
    return success();
  }
  return verifyA5MxMatScaleLayout(op, scaleTb, scaleName, isLeftScale);
}

static LogicalResult verifyA5MxMatScaleTiles(Operation *op, Type lhsScaleTy,
                                             Type rhsScaleTy, Type lhsTy,
                                             Type rhsTy) {
  if (failed(verifyA5MxMatScaleTile(op, lhsScaleTy, lhsTy, rhsTy, "a_scale",
                                    /*isLeftScale=*/true))) {
    return failure();
  }
  return verifyA5MxMatScaleTile(op, rhsScaleTy, lhsTy, rhsTy, "b_scale",
                                /*isLeftScale=*/false);
}

static LogicalResult verifyA5MxGemvTileOperands(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true)) ||
      failed(verifyMatmulValidSizes(op, lhsTy, rhsTy, 1))) {
    return failure();
  }
  return verifyGemvValidShapes(op, lhsTy, rhsTy, dstTy,
                               /*dstMayBeNonTile=*/false);
}

static LogicalResult verifyA5MxGemvScaleTile(Operation *op, Type scaleTy,
                                             Type lhsTy, Type rhsTy,
                                             StringRef scaleName,
                                             bool isLeftScale) {
  if (failed(verifyA5MxScaleStorage(op, scaleTy, scaleName)))
    return failure();

  auto scaleShape = getShapeVec(scaleTy);
  auto scaleValid = getValidShapeVec(scaleTy);
  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (scaleShape.size() != mlir::pto::kValue2 || scaleValid.size() != mlir::pto::kValue2 ||
      lhsShape.size() != mlir::pto::kValue2 || rhsShape.size() != mlir::pto::kValue2 ||
      lhsValid.size() != mlir::pto::kValue2 || rhsValid.size() != mlir::pto::kValue2) {
      return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 shape/valid_shape";
  }

  int64_t logicalM = lhsValid[0];
  int64_t logicalK = lhsValid[1];
  int64_t logicalN = rhsValid[1];
  int64_t scaleK = ceilDivKnown(logicalK, 32);

  int64_t expectedShapeRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedShapeCols = isLeftScale ? scaleK : rhsShape[1];
  int64_t expectedValidRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedValidCols = isLeftScale ? scaleK : logicalN;

  if (!hasCompatibleKnownExtent(scaleShape[0], expectedShapeRows) ||
      !hasCompatibleKnownExtent(scaleShape[1], expectedShapeCols) ||
      !hasCompatibleKnownExtent(scaleValid[0], expectedValidRows) ||
      !hasCompatibleKnownExtent(scaleValid[1], expectedValidCols)) {
    if (isLeftScale) {
      return op->emitOpError()
             << "expects " << scaleName
             << " shape/valid_shape to be [M, ceil(K/32)]";
    }
    return op->emitOpError()
           << "expects " << scaleName
           << " shape/valid_shape to be [ceil(K/32), aligned_N]/[ceil(K/32), N]";
  }
  return success();
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias"))) {
    return failure();
  }
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS) {
    return op->emitOpError("expects bias to be in the bias address space");
  }
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1) {
    return op->emitOpError("expects bias to have 1 row");
  }
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32()) {
      return op->emitOpError("expects bias to have element type f32");
    }
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias))) {
    return failure();
  }
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError("expects bias to use the row_major blayout on A5");
    }
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
  return failure();
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(mlir::pto::kValue32) && isInt8(lhsElemTy) && isInt8(rhsElemTy)) {
      return success();
  }

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy)) {
    return success();
  }

  auto isA5TMatmulFp8Type = [](Type ty) {
    return isPTOFloat8Type(ty);
  };
  if (isA5 && dstElemTy.isF32()) {
    if (isA5TMatmulFp8Type(lhsElemTy) && isA5TMatmulFp8Type(rhsElemTy)) {
      return success();
    }
    if (isPTOHiFloat8Type(lhsElemTy) && lhsElemTy == rhsElemTy) {
      return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", (f32, fp8, fp8), or (f32, hif8, hif8)" : "");
}

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    Type elemTy = *elemOr;
    if (elemTy.isInteger(16) || elemTy.isF16() || elemTy.isF32()) {
      return success();
    }
    return emitOpError("expects element type to be i16/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return emitOpError("taddrelu is only supported on A2/A3 targets");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddCOp::verify() {
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/src2/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd) {
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  }
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

static LogicalResult verifyTAxpyCommon(TAxpyOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (op.getScalar().getType() != getElemTy(srcTy)) {
    return op.emitOpError("expects scalar type to match src element type");
  }
  if (getShapeVec(srcTy) != getShapeVec(dstTy)) {
    return op.emitOpError("expects src and dst to have the same shape");
  }
  return success();
}

static LogicalResult verifyTAxpyArch(TAxpyOp op, bool allowBf16) {
    if (failed(verifyTAxpyCommon(op))) {
      return failure();
    }
    Type srcElem = getElemTy(op.getSrc().getType());
    Type dstElem = getElemTy(op.getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32)) {
      return op.emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    }
    if (!(dstElem.isF16() || dstElem.isF32() ||
          (allowBf16 && dstElem.isBF16()))) {
      return op.emitOpError() << "expects " << (allowBf16 ? "A5" : "A2/A3")
                              << " taxpy dst element type to be "
                              << (allowBf16 ? "f16/bf16/f32" : "f16/f32");
    }
    if (!(srcElem.isF16() || srcElem.isF32() ||
          (allowBf16 && srcElem.isBF16()))) {
      return op.emitOpError() << "expects " << (allowBf16 ? "A5" : "A2/A3")
                              << " taxpy src element type to be "
                              << (allowBf16 ? "f16/bf16/f32" : "f16/f32");
    }
    return success();
}
