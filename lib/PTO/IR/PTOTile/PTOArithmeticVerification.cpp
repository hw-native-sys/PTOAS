// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static FailureOr<Type> verifyMatchingElementTypes(Operation *op, Type lhs,
                                                  Type rhs) {
  Type lhsElem = getElemTy(lhs);
  Type rhsElem = getElemTy(rhs);
  if (!lhsElem || !rhsElem) {
    op->emitOpError("failed to get element type for src/dst");
    return failure();
  }
  if (lhsElem != rhsElem) {
    op->emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  return lhsElem;
}

static FailureOr<Type> verifyDistinctRowMajorUnaryTileOpCommon(
    Operation *op, Value src, Value dst, StringRef srcName = "src",
    StringRef dstName = "dst") {
  if (src == dst) {
    op->emitOpError("expects src and dst to use different storage");
    return failure();
  }
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName))) {
    return failure();
  }

  FailureOr<Type> elem = verifyMatchingElementTypes(op, srcTy, dstTy);
  if (failed(elem))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }
  return *elem;
}

static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool supported = elemTy.isInteger(32) || elemTy.isInteger(16) ||
                   elemTy.isF16() || elemTy.isF32();
  if (targetArch == PTOArch::A5) {
      supported =
          supported || (allowInt8OnA5 && elemTy.isInteger(mlir::pto::kValue8)) || (allowBf16OnA5 && elemTy.isBF16());
  }
  if (supported) {
    return success();
  }
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyArithmeticBinaryTileOpWithArchDispatch(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr)) {
      return failure();
    }
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyArithmeticScalarTileOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error,
    bool requireValidRowsEqualOnA2A3 = true,
    bool requireValidRowsEqualOnA5 = false) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireValidRowsEqual) -> LogicalResult {
    FailureOr<Type> elemOr = verifyNumericScalarTileOpCommon(
        op, srcTy, dstTy, scalarTy, requireValidRowsEqual);
    if (failed(elemOr)) {
      return failure();
    }
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireValidRowsEqualOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireValidRowsEqualOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyTColReductionElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool ok = elemTy.isF16() || elemTy.isF32() || elemTy.isInteger(16) ||
            elemTy.isInteger(32);
  if (targetArch == PTOArch::A5) {
      ok = ok || (allowInt8OnA5 && elemTy.isInteger(mlir::pto::kValue8)) || (allowBf16OnA5 && elemTy.isBF16());
  }
  if (ok) {
    return success();
  }
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyTColReductionOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, bool requireNonZeroSrcOnA2A3,
    bool requireNonZeroSrcOnA5, bool allowInt8OnA5, bool allowBf16OnA5,
    StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireNonZeroSrc) -> LogicalResult {
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
      return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return op->emitOpError("expects src and dst to have the same element type");
    }
    if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    return verifyTColReductionElemTypeForArch(op, elem, targetArch, allowInt8OnA5,
                                              allowBf16OnA5, a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireNonZeroSrcOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireNonZeroSrcOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

static bool isKnownUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

static bool isKnownZeroOrUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 0 || value == 1;
}

static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == 0 || lhs == rhs;
}

static LogicalResult verifyVecTileStorage(Operation *op, Type ty, StringRef name) {
  return verifyTileBufInVec(op, ty, name);
}
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto tb = dyn_cast<pto::TileBufType>(ty);
  auto as = getPTOMemorySpaceEnum(ty);
  if (as && *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (tb && tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  return success();
}

static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyVecTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyVecTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyVecTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName,
                                          StringRef dstName,
                                          bool allowBf16,
                                          bool allowInt8) {
  if (failed(verifyVecTileCommon(op, srcTy, srcName)) ||
      failed(verifyVecTileCommon(op, dstTy, dstName))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), allowBf16, allowInt8)) {
    return op->emitOpError() << "expects vec tile element types to be supported";
  }
  return success();
}

static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::ACC) {
    return op->emitOpError() << "expects " << name << " to be in the acc address space";
  }
  return success();
}

static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyAccTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyAccTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyAccTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyMatmulValidSizes(Operation *op, Type lhsTy,
                                            Type rhsTy, int64_t minValue) {
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() != mlir::pto::kValue2 || rhsValid.size() != mlir::pto::kValue2) {
      return success();
  }
  for (int64_t value : {lhsValid[0], lhsValid[1], rhsValid[1]}) {
      if (value != ShapedType::kDynamic && (value < minValue || value > mlir::pto::kValue4095)) {
          return op->emitOpError() << "expects m, k, and n valid sizes to be in [" << minValue << ", 4095]";
      }
  }
  return success();
}
