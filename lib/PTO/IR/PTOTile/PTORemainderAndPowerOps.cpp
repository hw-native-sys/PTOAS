// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTRemSTmpA2A3(TRemSOp op, Type tt, Type elem) {
  Type td = op.getDst().getType();
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (getElemTy(tt) != getElemTy(td)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(td);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trems dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmpA5(TRemSOp op, Type tt, Type elem) {
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmp(TRemSOp op, Type elem) {
  Type tt = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tt);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemSTmpA2A3(op, tt, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemSTmpA5(op, tt, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyRowMajorScalarTileCommon(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy) {
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    op->emitOpError("expects scalar type to match the tile element type");
    return failure();
  }
  return elem;
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  FailureOr<Type> elem = verifyRowMajorScalarTileCommon(
      getOperation(), getSrc().getType(), getDst().getType(),
      getScalar().getType());
  if (failed(elem))
    return failure();
  if (!getTmp()) {
    return verifyTRemSNoTmp(*this, *elem);
  }
  return verifyTRemSTmp(*this, *elem);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src and dst to use row-major layout");
  }

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpShape(Operation *op, Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufSameElemType(op, tmpTy, dstTy, "tmp", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op->emitOpError("expects tmp to use row-major layout");
  }
  return verifyTileBufSameValidShape(op, tmpTy, dstTy, "tmp", "dst");
}

static LogicalResult verifyTPowElemType(TPowOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpow element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpow element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpow element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpCommon(Operation *op, Value tmp, Type dstTy,
                                         bool isIntElem,
                                         StringRef integerError,
                                         StringRef staticShapeError) {
  if (isIntElem && tmp) {
    return op->emitOpError() << integerError;
  }
  if (tmp) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (failed(verifyTPowTmpShape(op, tmpTy, dstTy))) {
      return failure();
    }
    if (getTargetArch(op) != PTOArch::A5) {
      auto requiredBytes = getStaticByteSize(dstTy);
      if (!requiredBytes) {
        return op->emitOpError() << staticShapeError;
      }
      if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult verifyTPowTmp(TPowOp op, Type dstTy, bool isIntElem) {
  return verifyTPowTmpCommon(
      op.getOperation(), op.getTmp(), dstTy, isIntElem,
      "does not accept tmp when element type is integer (the integer pow "
      "lowering uses the 3-operand form TPOW(dst, base, exp))",
      "expects A2/A3 tpow dst shape to be static when tmp is provided");
}

mlir::LogicalResult mlir::pto::TPowOp::verify() {
  Type baseTy = getBase().getType();
  Type expTy = getExp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, baseTy, "base")) ||
      failed(verifyTileBufCommon(*this, expTy, "exp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameElemType(*this, baseTy, dstTy, "base", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, dstTy, "base", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(baseTy) || !isRowMajorTileBuf(expTy) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects base, exp, and dst to use row-major layout");
  }

  Type elem = getElemTy(baseTy);
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  if (failed(verifyTPowElemType(*this, elem, isIntElem))) {
    return failure();
  }
  return verifyTPowTmp(*this, dstTy, isIntElem);
}

static LogicalResult verifyTPowSElemType(TPowSOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpows element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpows element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpows element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowSTmp(TPowSOp op, Type dstTy, bool isIntElem) {
  return verifyTPowTmpCommon(
      op.getOperation(), op.getTmp(), dstTy, isIntElem,
      "does not accept tmp when element type is integer (the integer pows "
      "lowering uses the 3-operand form TPOWS(dst, src, scalar))",
      "expects A2/A3 tpows dst shape to be static when tmp is provided");
}
