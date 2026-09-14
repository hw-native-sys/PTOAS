// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTRowExpandImplicitTmpContract(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch) {
  if (!hasTmp || targetArch == PTOArch::A5) {
    return success();
  }

  if (classifyTRowExpandBinaryMode(src0Ty, src1Ty, dstTy) !=
      TRowExpandBinaryMode::Mode1ColMajorScalar) {
    return op->emitOpError(
        "expects A2/A3 tmp-form trowexpand to use mode 1 "
        "(ColMajor per-row scalar expanded operand)");
  }

  if (failed(verifyVecTileStorage(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects tmp and dst to have the same element type");
  }

  auto dstValid = getValidShapeVec(dstTy);
  if (dstValid.size() != mlir::pto::kValue2) {
      return op->emitOpError("expects dst to have rank-2 valid_shape");
  }
  int64_t minBytes = getTRowExpandTmpMinBytes(dstValid[0]);
  std::optional<int64_t> tmpBytes = getStaticTileCapacityBytes(tmpTy);
  if (!tmpBytes) {
    return op->emitOpError(
        "expects A2/A3 trowexpand tmp capacity to be statically known");
  }
  if (*tmpBytes < minBytes) {
    return op->emitOpError()
           << "expects A2/A3 trowexpand tmp capacity to be at least "
           << minBytes << " bytes, but got " << *tmpBytes << " bytes";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported =
        elem.isF16() || elem.isF32() ||
        (targetArch == PTOArch::A5 &&
         (elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError("expects element type to be f16 or f32");
    }
    if (getPrecisionType() == pto::DivPrecision::HighPrecision && !getTmp()) {
      return emitOpError("expects tmp when precisionType is high_precision");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRowExpandMulSub(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Value tmp,
    PTOArch targetArch, StringRef opName) {
  Type tmpTy = tmp ? tmp.getType() : Type{};
  auto elem = verifyTRowExpandBinaryCore(op, src0Ty, src1Ty, dstTy, tmpTy,
                                        static_cast<bool>(tmp));
  if (failed(elem))
    return failure();
  bool supported = elem->isF16() || elem->isF32() || elem->isInteger(16) ||
                   elem->isInteger(32) ||
                   (targetArch == PTOArch::A5 && elem->isInteger(8));
  if (!supported)
    return op->emitOpError()
           << "expects " << (targetArch == PTOArch::A5 ? "A5 " : "A2/A3 ")
           << opName
           << (targetArch == PTOArch::A5
                   ? " element type to be i8/i16/i32/f16/f32"
                   : " element type to be i16/i32/f16/f32");
  return verifyTRowExpandImplicitTmpContract(
      op, src0Ty, src1Ty, dstTy, tmpTy, static_cast<bool>(tmp), targetArch);
}

mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch arch) {
    return verifyTRowExpandMulSub(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType(), getTmp(), arch, "trowexpandmul");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch arch) {
    return verifyTRowExpandMulSub(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType(), getTmp(), arch, "trowexpandsub");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTRowExpandAddCore(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
      op, src0Ty, src1Ty, dstTy, op.getTmp() ? op.getTmp().getType() : Type{},
      static_cast<bool>(op.getTmp()));
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty)) {
    return op.emitOpError("expects src0 to use row-major layout");
  }
  Type elem = *elemOr;
  bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                   elem.isInteger(32) ||
                   (targetArch == PTOArch::A5 && elem.isInteger(8));
  if (!supported) {
    if (targetArch == PTOArch::A5) {
      return op.emitOpError(
          "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32");
    }
    return op.emitOpError(
        "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32");
  }
  return elem;
}

static LogicalResult verifyTRowExpandAddSrc1(TRowExpandAddOp op, Type elem,
                                             PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src1Valid.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects src1 and dst to have rank-2 valid_shape");
  }
  if (src1Valid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      src1Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src1 valid_shape[0] to equal dst valid_shape[0]");
  }
  bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
  int64_t expectedCol = elem.isInteger(8)
                            ? 32
                            : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t src1Col = src1Valid[1];
  if (src1IsRowMajor) {
    if (src1Col != ShapedType::kDynamic && src1Col != expectedCol) {
      return op.emitOpError("expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    }
  } else {
    if (src1Col != ShapedType::kDynamic && src1Col != 1) {
      return op.emitOpError("expects non-row-major src1 valid_shape[1] to be 1");
    }
  }
  if (failed(verifyTRowExpandImplicitTmpContract(
          op.getOperation(), src0Ty, src1Ty, dstTy,
          op.getTmp() ? op.getTmp().getType() : Type{},
          static_cast<bool>(op.getTmp()), targetArch))) {
    return failure();
  }
  return mlir::success();
}

static LogicalResult verifyTRowExpandAddByArch(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  FailureOr<Type> elemOr = verifyTRowExpandAddCore(op, targetArch);
  if (failed(elemOr)) {
    return failure();
  }
  return verifyTRowExpandAddSrc1(op, *elemOr, targetArch);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A5);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTRowExpandReduceTypes(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch, StringRef opName, bool allowIntegerTypes) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp"))))
    return failure();
  if (hasTmp && getElemTy(tmpTy) != getElemTy(dstTy)) {
    op->emitOpError("expects tmp and dst to have the same element type");
    return failure();
  }
  Type elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem) {
    op->emitOpError(
        "expects src0, src1, and dst to have the same element type");
    return failure();
  }
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (supported)
    return elem;
  if (!allowIntegerTypes)
    op->emitOpError() << "expects " << opName
                      << " element type to be f16 or f32";
  else if (targetArch == PTOArch::A5)
    op->emitOpError() << "expects A5 " << opName
                      << " element type to be i8/i16/i32/f16/f32";
  else
    op->emitOpError() << "expects A2/A3 " << opName
                      << " element type to be i16/i32/f16/f32";
  return failure();
}

static bool rowExpandValidShapesMatch(ArrayRef<int64_t> lhs,
                                      ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    auto [left, right] = pair;
    return left == ShapedType::kDynamic || right == ShapedType::kDynamic ||
           left == right;
  });
}
