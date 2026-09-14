// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTNegArch(TNegOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (isA5) {
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return op.emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1])
      return op.emitOpError(
          "expects src and dst to have the same valid_shape[1]");
  } else if (failed(
                 verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(srcTy);
  bool supported = isA5
                       ? isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                                /*allowInt8=*/true)
                       : (elemTy.isInteger(16) || elemTy.isInteger(32) ||
                          elemTy.isF16() || elemTy.isF32());
  if (!supported)
    return op.emitOpError(isA5
        ? "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16"
        : "expects A2/A3 tneg element type to be i16/i32/f16/f32");
  return success();
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() { return verifyTNegArch(*this, false); };
  auto verifyA5 = [&]() { return verifyTNegArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyCommon = [&]() {
    return verifyMatchingVecUnaryTiles(getOperation(), getSrc().getType(),
                                       getDst().getType());
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto elemTy = verifyCommon();
    return failed(elemTy)
               ? failure()
               : verifyIntegerWidths(getOperation(), *elemTy, {16},
                                     "expects A2/A3 tnot element type to be i16");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto elemTy = verifyCommon();
    return failed(elemTy)
               ? failure()
               : verifyIntegerWidths(
                     getOperation(), *elemTy, {8, 16, 32},
                     "expects A5 tnot element type to be i8/i16/i32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  return verifyBitwiseBinaryOp(getOperation(), getSrc0().getType(),
                               getSrc1().getType(), getDst().getType(), "tor");
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  // ORS has the same operand contract as ANDS; diagnostics intentionally omit
  // the scalar noun for compatibility with the existing verifier messages.
  auto verifyFor = [&](bool isA5) -> LogicalResult {
    auto elem = verifyDistinctRowMajorUnaryTileOpCommon(
        getOperation(), getSrc(), getDst(), "src", "dst");
    if (failed(elem))
      return failure();
    return verifyIntegerWidths(
        getOperation(), *elem,
        isA5 ? ArrayRef<unsigned>{8, 16, 32} : ArrayRef<unsigned>{8, 16},
        isA5 ? "expects A5 tors src and dst element type to be i8/i16/i32"
             : "expects A2/A3 tors src and dst element type to be i8/i16");
  };
  auto verifyA2A3 = [&]() { return verifyFor(false); };
  auto verifyA5 = [&]() { return verifyFor(true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op->emitOpError(
               "expects src0/src1/dst to be tensor/tile_buf/tile_view types"),
           failure();
  }
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for operands"), failure();
  }
  if (e0 != e1 || e0 != ed) {
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  }
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd) {
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  }
  return e0;
}

static LogicalResult verifyTPartBinaryA2A3(Operation *op, Type src0Ty,
                                           Type src1Ty, Type dstTy,
                                           StringRef opName) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op->emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op->emitOpError()
           << "expects src0/src1/dst to have the same element type";
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op->emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op->emitOpError()
           << "expects A2/A3 " << opName
           << " element type to be i32/i16/f16/f32";
  }
  return mlir::success();
}

static LogicalResult verifyTPartBinaryA5(Operation *op, Type src0Ty,
                                         Type src1Ty, Type dstTy,
                                         StringRef opName) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy))
    return op->emitOpError() << "expects PTO shaped-like src0/src1/dst";
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy))
    return op->emitOpError()
           << "expects src0/src1/dst to have the same element type";
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32()))
    return op->emitOpError()
           << "expects A5 " << opName
           << " element type to be i32/i16/i8/f16/bf16/f32";
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
    return op->emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  return verifyPartialValidPatternLoose(op, src0Ty, src1Ty, dstTy);
}

static LogicalResult verifyTPartAddA2A3(TPartAddOp op) {
  return verifyTPartBinaryA2A3(op, op.getSrc0().getType(),
                               op.getSrc1().getType(), op.getDst().getType(),
                               "tpartadd");
}

static LogicalResult verifyTPartAddA5(TPartAddOp op) {
  return verifyTPartBinaryA5(op, op.getSrc0().getType(),
                             op.getSrc1().getType(), op.getDst().getType(),
                             "tpartadd");
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartAddA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartAddA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPartMinMax(Operation *op, Type src0, Type src1,
                                       Type dst, StringRef opName, bool isA5) {
  auto elem = verifyPTOShapedBinarySameElemAndShape(op, src0, src1, dst);
  if (failed(elem))
    return failure();
  if (!isA5 && failed(verifyPartialValidPattern(op, src0, src1, dst)))
    return failure();
  bool supported = isA5
                       ? (elem->isInteger(32) || elem->isInteger(16) ||
                          elem->isInteger(8) || elem->isF16() ||
                          elem->isBF16() || elem->isF32())
                       : (elem->isInteger(32) || elem->isInteger(16) ||
                          elem->isF16() || elem->isF32());
  if (!supported)
    return op->emitOpError()
           << "expects " << (isA5 ? "A5 " : "A2/A3 ") << opName
           << (isA5 ? " element type to be i32/i16/i8/f16/bf16/f32"
                    : " element type to be i32/i16/f16/f32");
  return isA5 ? verifyPartialValidPatternLoose(op, src0, src1, dst)
              : success();
}

#define PTO_DEFINE_PART_MINMAX_VERIFY(OpClass, opName)                             \
  mlir::LogicalResult mlir::pto::OpClass::verify() {                               \
    auto verifyA2A3 = [&]() {                                                      \
      return verifyTPartMinMax(getOperation(), getSrc0().getType(),                \
                               getSrc1().getType(), getDst().getType(), opName,     \
                               false);                                             \
    };                                                                             \
    auto verifyA5 = [&]() {                                                        \
      return verifyTPartMinMax(getOperation(), getSrc0().getType(),                \
                               getSrc1().getType(), getDst().getType(), opName,     \
                               true);                                              \
    };                                                                             \
    return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);           \
  }

PTO_DEFINE_PART_MINMAX_VERIFY(TPartMaxOp, "tpartmax")
PTO_DEFINE_PART_MINMAX_VERIFY(TPartMinOp, "tpartmin")

static LogicalResult verifyTPartArgIndices(Operation *op, Type src0Ty,
                                           Type src1Ty, Type src0IdxTy,
                                           Type src1IdxTy, Type dstTy,
                                           Type dstIdxTy) {
  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy)) {
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  }
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy)) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");
  }

  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) ||
      dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  }
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");
  }

  return success();
}
