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

static LogicalResult verifyTPReluElemAndLayout(TPReluOp op, Type src0Ty,
                                               Type src1Ty, Type tmpTy,
                                               Type dstTy) {
  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type tmpElem = getElemTy(tmpTy);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !tmpElem || !dstElem)
    return op.emitOpError("failed to get element type for operands");
  if (src0Elem != src1Elem || src0Elem != dstElem)
    return op.emitOpError("expects dst/src0/src1 to have the same element type");
  if (!(src0Elem.isF16() || src0Elem.isF32())) {
    return op.emitOpError("expects dst/src0/src1 element type to be f16 or f32");
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError("expects src0, src1, and dst to use row-major layout");
  }
  return success();
}

static LogicalResult verifyTPReluMatchingShapes(TPReluOp op, Type src0Ty,
                                                Type src1Ty, Type tmpTy,
                                                Type dstTy) {
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst")))
    return failure();
  if (getShapeVec(src0Ty) != getShapeVec(src1Ty) ||
      getShapeVec(src0Ty) != getShapeVec(tmpTy) ||
      getShapeVec(src0Ty) != getShapeVec(dstTy)) {
    return op.emitOpError("expects src0/src1/tmp/dst to have the same shape");
  }
  return success();
}

static FailureOr<TPReluCommonInfo> verifyTPReluCommon(TPReluOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type tmpTy = op.getTmp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }

  if (failed(verifyTPReluElemAndLayout(op, src0Ty, src1Ty, tmpTy, dstTy)))
    return failure();
  if (failed(verifyTPReluMatchingShapes(op, src0Ty, src1Ty, tmpTy, dstTy)))
    return failure();
  return TPReluCommonInfo{src0Ty, src1Ty, tmpTy, dstTy};
}

static LogicalResult verifyTPReluA2A3(TPReluOp op,
                                      const TPReluCommonInfo &common) {
  Type tmpElem = getElemTy(common.tmpTy);
  auto tmpIntTy = dyn_cast<IntegerType>(tmpElem);
  if (!tmpIntTy || tmpIntTy.getWidth() != kPTOI8BitWidth)
    return op.emitOpError("expects A2/A3 tmp element type to be u8");
  if (!isRowMajorTileBuf(common.tmpTy))
    return op.emitOpError("expects tmp to use row-major layout");
  if (auto arch = getVerifierArchName(op.getOperation());
      arch && arch->equals_insensitive("a3")) {
    if (op.getSrc0() == op.getSrc1() || op.getSrc0() == op.getTmp() ||
        op.getSrc0() == op.getDst() || op.getSrc1() == op.getTmp() ||
        op.getSrc1() == op.getDst() || op.getTmp() == op.getDst()) {
      return op.emitOpError(
          "expects A3 src0, src1, tmp, and dst to use different storage");
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto common = verifyTPReluCommon(*this);
  if (failed(common))
    return failure();
  auto verifyA2A3 = [this, &common]() -> LogicalResult {
    return verifyTPReluA2A3(*this, *common);
  };
  auto verifyA5 = []() -> LogicalResult { return success(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTQuantStructural(TQuantOp op) {
  Type dstElemTy = getElemTy(op.getDst().getType());
  auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
  if (op.getQuantType() == mlir::pto::QuantType::INT8_SYM) {
    if (!dstIntTy || dstIntTy.getWidth() != kPTOI8BitWidth) {
      return op.emitOpError(
          "expects dst element type i8/ui8 for INT8_SYM quantization");
    }
    if (op.getOffset()) {
      return op.emitOpError(
          "INT8_SYM quantization must not have an offset operand");
    }
    return success();
  }

  if (!dstIntTy || dstIntTy.getWidth() != kPTOI8BitWidth) {
    return op.emitOpError(
        "expects dst element type i8/ui8 for INT8_ASYM quantization");
  }
  if (!op.getOffset())
    return op.emitOpError("INT8_ASYM quantization requires an offset operand");
  return success();
}

static LogicalResult verifyTQuantCommon(TQuantOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (!getElemTy(srcTy).isF32())
    return op.emitOpError("expects src to have element type f32");
  if (!op.getOffset())
    return success();

  Type offsetTy = op.getOffset().getType();
  if (failed(verifyTileBufCommon(op, offsetTy, "offset")))
    return failure();
  if (!getElemTy(offsetTy).isF32())
    return op.emitOpError("expects offset to have element type f32");
  return success();
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  if (failed(verifyTQuantStructural(*this)))
    return failure();
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto verifyA2A3 = [this]() -> LogicalResult {
    if (failed(verifyTQuantCommon(*this)))
      return failure();
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError("expects A2/A3 src and dst to use row-major layout");
    }
    return success();
  };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTQuantCommon(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [this]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == kPTOI8BitWidth || srcIntTy.getWidth() == kPTOI16BitWidth))
      return emitOpError()
             << "expects src element type i8 or i16";
    if (!getElemTy(getDst().getType()).isF32())
      return emitOpError() << "expects dst element type f32";
    if (!getElemTy(getScale().getType()).isF32())
      return emitOpError() << "expects scale element type f32";
    if (!getElemTy(getOffset().getType()).isF32())
      return emitOpError() << "expects offset element type f32";
    return success();
  };
  if (failed(verifyStructural()))
    return failure();

  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto verifyCommon = [this]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst")))
      return failure();
    return success();
  };

  auto verifyA2A3 = [this, &verifyCommon]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType()))
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    return success();
  };

  auto verifyA5 = [&verifyCommon]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst())
    return emitOpError("expects A3 trecip src and dst to use different storage");
  return mlir::success();
}
