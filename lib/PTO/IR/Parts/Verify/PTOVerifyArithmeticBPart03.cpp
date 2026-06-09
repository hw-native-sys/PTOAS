// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticB.cpp; kept as a fragment included by PTOVerifyArithmeticB.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult pto::TConcatidxOp::verify() {
  auto elemOr = verifyTConcatidxCommon(*this);
  if (failed(elemOr))
    return failure();
  auto verifyA2A3 = [this, &elemOr]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVecType(getOperation(), getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    return verifyConcatidxElementTypes(getOperation(), elemOr->first,
                                       elemOr->second);
  };
  auto verifyA5 = [this, &elemOr]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVecType(getOperation(), getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getSrc0Idx().getType()) ||
        !isRowMajorTileBuf(getSrc1Idx().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError("expects all operands to use row-major layout");
    }
    return verifyConcatidxElementTypes(getOperation(), elemOr->first,
                                       elemOr->second);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  return verifyDistinctRowMajorUnaryIntWidthOp(
      getOperation(), getSrc(), getDst(), "src", "dst",
      "expects A2/A3 tands src, scalar, and dst element type to be i8/i16",
      "expects A5 tands src, scalar, and dst element type to be i8/i16/i32");
}

LogicalResult pto::TCIOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy)
    return emitOpError("expects dst element type to be integer");

  unsigned bw = elemTy.getWidth();
  if (bw != kPTOI16BitWidth && bw != kPTOI32BitWidth)
    return emitOpError("expects dst element type to be i16/i32");

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy)
    return emitOpError("expects S to be integer");

  if (sTy != elemTy)
    return emitOpError("expects S and dst element type to be exactly the same type");
  auto shape = getShapeVec(dstTy);
  if (shape.size() != kPTORowColRank)
    return emitOpError("expects dst to be rank-2");
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1)
    return emitOpError("expects dst cols to be different from 1");

  return success();
}

LogicalResult pto::TTriOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy)
    return emitOpError("expects diagonal to be an integer operand");

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1)
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [this, elemTy]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false))
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        return success();
      },
      [this, elemTy]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true))
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        return success();
      });
}

static LogicalResult verifyTCmpA2A3(TCmpOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, src0Ty, "src0")) ||
      failed(verifyVecTileStorage(op, src1Ty, "src1")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst"))) {
    return failure();
  }
  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !dstElem)
    return op.emitOpError("failed to get element type for src0/src1/dst");
  if (src0Elem != src1Elem)
    return op.emitOpError("expects src0 and src1 to have the same element type");
  if (!(src0Elem.isInteger(kPTOI32BitWidth) || src0Elem.isF16() || src0Elem.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tcmp input element type to be i32/f16/f32");
  }
  if (!dstElem.isInteger(kPTOI8BitWidth))
    return op.emitOpError("expects dst element type to be i8");

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != kPTORowColRank || src1Valid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank) {
    return op.emitOpError(
        "expects src0, src1, and dst to have rank-2 valid_shape");
  }
  if (!hasCompatibleKnownExtent(src0Valid[0], src1Valid[0]))
    return op.emitOpError("expects src0 and src1 to have the same valid row");
  if (!hasCompatibleKnownExtent(src0Valid[1], src1Valid[1]))
    return op.emitOpError(
        "expects src0 and src1 to have the same valid column");
  if (!hasCompatibleKnownExtent(src0Valid[0], dstValid[0]))
    return op.emitOpError("expects src0 valid row to equal dst valid row");
  return success();
}

static LogicalResult verifyTCmpA5(TCmpOp op) {
  auto verifyTileBufOperand = [](Operation *op, Type ty, StringRef name) {
    return verifyTileBufCommon(op, ty, name);
  };
  auto infoOr = verifyBinaryTileTypeInfo(op, op.getSrc0(), op.getSrc1(),
                                         op.getDst(), verifyTileBufOperand);
  if (failed(infoOr))
    return failure();
  const auto &info = *infoOr;
  if (info.src0Elem != info.src1Elem)
    return op.emitOpError("expects src0 and src1 to have the same element type");
  if (!(info.src0Elem.isF16() || info.src0Elem.isF32() ||
        info.src0Elem.isBF16() || info.src0Elem.isInteger(kPTOI8BitWidth) ||
        info.src0Elem.isInteger(kPTOI16BitWidth) || info.src0Elem.isInteger(kPTOI32BitWidth))) {
    return op.emitOpError(
        "expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(info.dstElem);
  if (!dstInt || dstInt.getWidth() != kPTOI8BitWidth)
    return op.emitOpError("expects dst element type to be i8");
  if (getShapeVec(info.src0Ty) != getShapeVec(info.src1Ty) ||
      getShapeVec(info.src0Ty) != getShapeVec(info.dstTy)) {
    return op.emitOpError("expects src0, src1, and dst to have the same shape");
  }
  return success();
}

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTCmpA2A3(*this);
  };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTCmpA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
static LogicalResult verifyTCmpSCommon(TCmpSOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst"))) {
    return failure();
  }
  if (!op.getScalar().getType().isIntOrIndexOrFloat())
    return op.emitOpError("expects scalar to be integer, index, or float");
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (srcValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  return success();
}
