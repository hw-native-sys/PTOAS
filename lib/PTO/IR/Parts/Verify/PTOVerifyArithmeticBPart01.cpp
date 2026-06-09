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

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/src2/dst to be memref/tile_buf types");

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd)
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}

static FailureOr<std::pair<Type, Type>> verifyTAxpyCommon(TAxpyOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  if (op.getScalar().getType() != srcElem)
    return op.emitOpError("expects scalar type to match src element type"),
           failure();
  if (getShapeVec(srcTy) != getShapeVec(dstTy))
    return op.emitOpError("expects src and dst to have the same shape"),
           failure();
  return std::make_pair(srcElem, getElemTy(dstTy));
}

static LogicalResult verifyTAxpyTypePair(Operation *op, Type srcElem,
                                         Type dstElem) {
  bool sameType = srcElem == dstElem;
  bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
  if (!(sameType || widenF16ToF32)) {
    return op->emitOpError(
        "expects dst/src element types to match, or dst=f32 and src=f16");
  }
  return success();
}

LogicalResult pto::TAxpyOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    auto common = verifyTAxpyCommon(*this);
    if (failed(common))
      return failure();
    auto [srcElem, dstElem] = *common;
    if (failed(verifyTAxpyTypePair(*this, srcElem, dstElem)))
      return failure();
    if (!(dstElem.isF16() || dstElem.isF32()))
      return emitOpError("expects A2/A3 taxpy dst element type to be f16/f32");
    if (!(srcElem.isF16() || srcElem.isF32()))
      return emitOpError("expects A2/A3 taxpy src element type to be f16/f32");
    return success();
  };

  auto verifyA5 = [this]() -> LogicalResult {
    auto common = verifyTAxpyCommon(*this);
    if (failed(common))
      return failure();
    auto [srcElem, dstElem] = *common;
    if (failed(verifyTAxpyTypePair(*this, srcElem, dstElem)))
      return failure();
    if (!(dstElem.isF16() || dstElem.isF32() || dstElem.isBF16()))
      return emitOpError("expects A5 taxpy dst element type to be f16/bf16/f32");
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isBF16()))
      return emitOpError("expects A5 taxpy src element type to be f16/bf16/f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd)
    return emitOpError("expects src0/src1/dst to have the same shape");
  return success();
}

template <typename VerifyCommonFn>
static LogicalResult verifyArchIntegerWidthOp(Operation *op,
                                              VerifyCommonFn verifyCommon,
                                              StringRef a2a3Message,
                                              StringRef a5Message) {
  auto verifyA2A3 =
      [&verifyCommon, op, a2a3Message]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth))
      return op->emitOpError(a2a3Message);
    return success();
  };

  auto verifyA5 = [&verifyCommon, op, a5Message]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth &&
                it.getWidth() != kPTOI32BitWidth))
      return op->emitOpError(a5Message);
    return success();
  };

  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyRowMajorBinaryIntWidthOp(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy,
    StringRef a2a3Message, StringRef a5Message) {
  auto verifyCommon = [op, src0Ty, src1Ty, dstTy]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
  };
  return verifyArchIntegerWidthOp(op, verifyCommon, a2a3Message, a5Message);
}

static LogicalResult verifyDistinctRowMajorUnaryIntWidthOp(
    Operation *op, Value src, Value dst, StringRef srcName, StringRef dstName,
    StringRef a2a3Message, StringRef a5Message) {
  auto verifyCommon = [op, src, dst, srcName, dstName]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(op, src, dst, srcName,
                                                   dstName);
  };
  return verifyArchIntegerWidthOp(op, verifyCommon, a2a3Message, a5Message);
}

LogicalResult pto::TAndOp::verify() {
  return verifyRowMajorBinaryIntWidthOp(
      getOperation(), getSrc0().getType(), getSrc1().getType(),
      getDst().getType(),
      "expects A2/A3 tand src0, src1, and dst element type to be i8/i16",
      "expects A5 tand src0, src1, and dst element type to be i8/i16/i32");
}

static LogicalResult verifyLocVecType(Operation *op, Type ty, StringRef name) {
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to use loc=vec";
  return success();
}

static LogicalResult verifyConcatElemType(Operation *op, Type elem) {
  if (elem.isF16() || elem.isF32() || elem.isBF16())
    return success();
  auto it = dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth &&
              it.getWidth() != kPTOI32BitWidth)) {
    return op->emitOpError(
        "expects element type to be i8, i16, i32, f16, f32, or bf16");
  }
  return success();
}

static LogicalResult verifyTConcatValidRows(TConcatOp op,
                                            ArrayRef<int64_t> src0Valid,
                                            ArrayRef<int64_t> src1Valid,
                                            ArrayRef<int64_t> dstValid) {
  if (src0Valid.size() != kPTORowColRank || src1Valid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank) {
    return op.emitOpError(
        "expects src0, src1, and dst to have rank-2 valid_shape");
  }
  if (src0Valid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && src0Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src0 valid row to match dst valid row");
  }
  if (src1Valid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && src1Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src1 valid row to match dst valid row");
  }
  return success();
}
