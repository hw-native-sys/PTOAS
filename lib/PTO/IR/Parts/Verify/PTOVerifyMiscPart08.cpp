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

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [this](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(kPTOI32BitWidth) || elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << errorMessage;
    return success();
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRemRowMajorTiles(Operation *op, Type src0Ty,
                                             Type src1Ty, Type tmpTy,
                                             Type dstTy) {
  if (isRowMajorTileBuf(src0Ty) && isRowMajorTileBuf(src1Ty) &&
      isRowMajorTileBuf(tmpTy) && isRowMajorTileBuf(dstTy)) {
    return success();
  }
  return op->emitOpError(
      "expects src0, src1, tmp, and dst to use row-major layout");
}

static LogicalResult verifyTRemTmpCoverage(Operation *op, Type tmpTy,
                                           Type dstTy) {
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != kPTORowColRank || tmpValid.size() != kPTORowColRank)
    return op->emitOpError("expects tmp and dst to be rank-2 tiles");
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1)
    return op->emitOpError("expects tmp to have at least 1 valid row");
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op->emitOpError(
        "expects tmp valid columns to cover dst valid columns");
  }
  return success();
}

static FailureOr<Type> verifyTRemCommon(Operation *op, Type src0Ty, Type src1Ty,
                                        Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return op->emitOpError("expects tmp and dst to have the same element type"),
           failure();
  if (failed(verifyTRemRowMajorTiles(op, src0Ty, src1Ty, tmpTy, dstTy)) ||
      failed(verifyTRemTmpCoverage(op, tmpTy, dstTy)))
    return failure();
  return getElemTy(src0Ty);
}

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  FailureOr<Type> elemOr = verifyTRemCommon(*this, getSrc0().getType(),
                                            getSrc1().getType(),
                                            getTmp().getType(),
                                            getDst().getType());
  if (failed(elemOr))
    return failure();
  Type elem = *elemOr;
  auto verifyA2A3 = [this, elem]() -> LogicalResult {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isF32()))
      return emitOpError("expects A2/A3 trem element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [this, elem]() -> LogicalResult {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

static FailureOr<Type> verifyTRemScalarCommon(Operation *op, Type srcTy,
                                              Type tmpTy, Type dstTy,
                                              Type scalarTy) {
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return op->emitOpError("expects tmp and dst to have the same element type"),
           failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(tmpTy) ||
      !isRowMajorTileBuf(dstTy)) {
    return op->emitOpError("expects src, tmp, and dst to use row-major layout"),
           failure();
  }
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return op->emitOpError("expects scalar type to match the tile element type"),
           failure();
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != kPTORowColRank || tmpValid.size() != kPTORowColRank)
    return op->emitOpError("expects tmp and dst to be rank-2 tiles"), failure();
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1)
    return op->emitOpError("expects tmp to have at least 1 valid row"), failure();
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op->emitOpError("expects tmp valid columns to cover dst valid columns"),
           failure();
  }
  return elem;
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  FailureOr<Type> elemOr =
      verifyTRemScalarCommon(*this, getSrc().getType(), getTmp().getType(),
                             getDst().getType(), getScalar().getType());
  if (failed(elemOr))
    return failure();
  Type elem = *elemOr;
  auto verifyA2A3 = [this, elem]() -> LogicalResult {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isF32()))
      return emitOpError("expects A2/A3 trems element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [this, elem]() -> LogicalResult {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src and dst to use row-major layout");

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");

  auto verifyA2A3 = [this, elem]() -> LogicalResult {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [this, elem]() -> LogicalResult {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
