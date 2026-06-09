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

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  return verifyDistinctRowMajorUnaryIntWidthOp(
      getOperation(), getSrc(), getDst(), "src", "dst",
      "expects A2/A3 tors src and dst element type to be i8/i16",
      "expects A5 tors src and dst element type to be i8/i16/i32");
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy))
    return op->emitOpError(
               "expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types"),
           failure();
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for operands"), failure();
  if (e0 != e1 || e0 != ed)
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd)
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  return e0;
}

static LogicalResult verifyTPartBinaryLikeOp(Operation *op, Type src0Ty,
                                             Type src1Ty, Type dstTy,
                                             StringRef opName);

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartadd");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

static LogicalResult verifyTPartArgIndexOperands(Operation *op, Type src0Ty,
                                                 Type src1Ty, Type src0IdxTy,
                                                 Type src1IdxTy, Type dstTy,
                                                 Type dstIdxTy) {
  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy))
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy)) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != kPTOI32BitWidth) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");
  }
  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) || dataShape != getShapeVec(src1IdxTy) ||
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

static LogicalResult verifyTPartArgElementType(Operation *op, Type elem,
                                               StringRef opName) {
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI8BitWidth) ||
          elem.isF16() || elem.isBF16() || elem.isF32())) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
    }
    return success();
  }
  if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() ||
        elem.isF32())) {
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i32/i16/f16/f32";
  }
  return success();
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElemOr))
    return failure();
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)))
    return failure();

  if (failed(verifyTPartArgIndexOperands(op, src0Ty, src1Ty, src0IdxTy,
                                         src1IdxTy, dstTy, dstIdxTy)))
    return failure();
  return verifyTPartArgElementType(op, *dataElemOr, opName);
}

static LogicalResult verifyTPartBinaryLikeOp(Operation *op, Type src0Ty,
                                             Type src1Ty, Type dstTy,
                                             StringRef opName) {
  FailureOr<Type> elemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(elemOr))
    return failure();
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != kNumber2 || s1.size() != kNumber2 || d.size() != kNumber2)
    return op->emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  PTOArch arch = getTargetArch(op);
  if (arch != PTOArch::A5 &&
      failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)))
    return failure();
  Type elem = *elemOr;
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI8BitWidth) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return op->emitOpError()
             << "expects A5 " << opName
             << " element type to be i32/i16/i8/f16/bf16/f32";
    return success();
  }
  if (!(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() ||
        elem.isF32()))
    return op->emitOpError()
           << "expects A2/A3 " << opName
           << " element type to be i32/i16/f16/f32";
  return success();
}

mlir::LogicalResult mlir::pto::TPartArgMaxOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartArgMinOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyByArch = [this]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartmul");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

struct TPReluCommonInfo {
  Type src0Ty;
  Type src1Ty;
  Type tmpTy;
  Type dstTy;
};
