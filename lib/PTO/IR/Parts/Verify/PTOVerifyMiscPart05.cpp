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

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (isFormat1())
    return verifyTMrgSortFormat1(*this);
  if (isFormat2())
    return verifyTMrgSortFormat2(*this);
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return emitOpError() << "failed to get element type for src/dst";
  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!mlir::isa<IntegerType>(srcElem))
    return emitOpError() << "expects integral element types";
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0)
    return emitOpError("expects tshls scalar to be non-negative");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [this]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [this, &verifyCommon]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI16BitWidth && it.getWidth() != kPTOI32BitWidth))
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
    return success();
  };

  auto verifyA5 = [this, &verifyCommon]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth &&
                it.getWidth() != kPTOI32BitWidth))
      return emitOpError(
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTNegCommon(Operation *op, Type srcTy, Type dstTy) {
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")))
    return failure();
  return getElemTy(srcTy);
}

static LogicalResult verifyTNegA2A3(TNegOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTNegCommon(op, srcTy, dstTy);
  if (failed(elemOr) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elemTy = *elemOr;
  if (!(elemTy.isInteger(kPTOI16BitWidth) || elemTy.isInteger(kPTOI32BitWidth) || elemTy.isF16() ||
        elemTy.isF32())) {
    return op.emitOpError()
           << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
  }
  return success();
}

static LogicalResult verifyTNegA5(TNegOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTNegCommon(op, srcTy, dstTy);
  if (failed(elemOr))
    return failure();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op.emitOpError() << "expects src and dst to have rank-2 valid_shape";
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op.emitOpError()
           << "expects src and dst to have the same valid_shape[1]";
  }
  Type elemTy = *elemOr;
  if (!(elemTy.isInteger(kPTOI8BitWidth) || elemTy.isInteger(kPTOI16BitWidth) || elemTy.isInteger(kPTOI32BitWidth) ||
        elemTy.isF16() || elemTy.isF32() || elemTy.isBF16())) {
    return op.emitOpError()
           << "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult { return verifyTNegA2A3(*this); };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTNegA5(*this); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyCommon = [this]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy)) {
      emitOpError() << "expects src and dst to have the same element type";
      return failure();
    }
    return elemTy;
  };
  auto verifyA2A3 = [this, &verifyCommon]() -> LogicalResult {
    FailureOr<Type> elemTy = verifyCommon();
    if (failed(elemTy))
      return failure();
    if (!(*elemTy).isInteger(kPTOI16BitWidth))
      return emitOpError() << "expects A2/A3 tnot element type to be i16";
    return success();
  };
  auto verifyA5 = [this, &verifyCommon]() -> LogicalResult {
    FailureOr<Type> elemTy = verifyCommon();
    if (failed(elemTy))
      return failure();
    if (!((*elemTy).isInteger(kPTOI8BitWidth) || (*elemTy).isInteger(kPTOI16BitWidth) ||
          (*elemTy).isInteger(kPTOI32BitWidth)))
      return emitOpError() << "expects A5 tnot element type to be i8/i16/i32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  return verifyRowMajorBinaryIntWidthOp(
      getOperation(), getSrc0().getType(), getSrc1().getType(),
      getDst().getType(),
      "expects A2/A3 tor src0, src1, and dst element type to be i8/i16",
      "expects A5 tor src0, src1, and dst element type to be i8/i16/i32");
}
