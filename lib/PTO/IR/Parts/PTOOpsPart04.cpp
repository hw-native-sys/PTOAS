// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isF16()) {
    return isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isBF16()) {
    return isPTOFloat4PackedType(dstElem);
  }
  if (isPTOFloat4PackedType(srcElem)) {
    return dstElem.isBF16();
  }
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem)) {
    return dstElem.isF32();
  }
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  }
  return true;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != 2) {
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
    }
    Type elemTy = tb.getElementType();
    if (!allowLowPrecision && isPTOLowPrecisionType(elemTy)) {
      return op->emitOpError() << name << ": dtype " << elemTy
                               << " is not supported by this op yet";
    }
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != mlir::pto::kValue2) {
      return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  }
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < mlir::pto::kValue2; ++i) {
      if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic && validShape[i] > shape[i]) {
          return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i << "] <= shape[" << i
                                   << "]";
      }
  }
  return success();
}

static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf";
  }
  if (getElemTy(lhs) != getElemTy(rhs)) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  }
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return success();
  }
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i]) {
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    }
  }
  if (lhsValid.size() != rhsValid.size()) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  }
  return success();
}

static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return success();
  }

  auto lhsExtent = getLogicalTileExtentVec(lhs, compareValidShape);
  auto rhsExtent = getLogicalTileExtentVec(rhs, compareValidShape);
  auto emitMismatch = [&]() -> LogicalResult {
    if (compareValidShape) {
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    }
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have compatible shapes";
  };
  if (lhsExtent.size() != rhsExtent.size()) {
    return emitMismatch();
  }

  for (size_t i = 0, e = lhsExtent.size(); i < e; ++i) {
    if (lhsExtent[i] != ShapedType::kDynamic &&
        rhsExtent[i] != ShapedType::kDynamic && lhsExtent[i] != rhsExtent[i]) {
      return emitMismatch();
    }
  }
  return success();
}

static LogicalResult verifyPartialValidPatternImpl(Operation *op, Type src0Ty,
                                                   Type src1Ty, Type dstTy,
                                                   bool requireExactInput) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2 ||
      dstValid.size() != mlir::pto::kValue2) {
      return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };
  auto equalsKnown = [](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    for (auto [a, b] : llvm::zip(lhs, rhs)) {
      if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b) {
        return false;
      }
    }
    return true;
  };

  for (unsigned i = 0; i < mlir::pto::kValue2; ++i) {
      if (!lessEqualKnown(src0Valid[i], dstValid[i]) || !lessEqualKnown(src1Valid[i], dstValid[i])) {
          return op->emitOpError("expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
      }
  }
  if (requireExactInput && !equalsKnown(src0Valid, dstValid) &&
      !equalsKnown(src1Valid, dstValid)) {
    return op->emitOpError(
        "expects at least one of src0/src1 valid_shape to match dst valid_shape");
  }
  return success();
}

static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy) {
  return verifyPartialValidPatternImpl(op, src0Ty, src1Ty, dstTy,
                                       /*requireExactInput=*/true);
}

static LogicalResult verifyPartialValidPatternLoose(Operation *op, Type src0Ty,
                                                    Type src1Ty, Type dstTy) {
  return verifyPartialValidPatternImpl(op, src0Ty, src1Ty, dstTy,
                                       /*requireExactInput=*/false);
}

[[maybe_unused]] static bool hasKnownZeroValidRegion(Type ty) {
  auto valid = getValidShapeVec(ty);
  if (valid.size() != mlir::pto::kValue2) {
      return false;
  }
  return valid[0] == 0 || valid[1] == 0;
}

static LogicalResult verifyScalarTileOp(Operation *op, Type srcTy, Type dstTy,
                                        StringRef srcName, StringRef dstName,
                                        bool requireValidRowsEqual,
                                        bool requireValidColsEqual) {
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName))) {
    return failure();
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << srcName
                             << " to be in the vec address space";
  }
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << dstName
                             << " to be in the vec address space";
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2) {
      return op->emitOpError() << "expects " << srcName << " and " << dstName << " to have rank-2 valid_shape";
  }
  if (requireValidRowsEqual &&
      srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[0]";
  }
  if (requireValidColsEqual &&
      srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[1]";
  }
  return success();
}

static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                         Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual) {
  if (failed(verifyScalarTileOp(op, srcTy, dstTy, "src", "dst",
                                requireValidRowsEqual,
                                /*requireValidColsEqual=*/true))) {
    return failure();
  }
  if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
    op->emitOpError("scalar must be a scalar type (integer/float)");
    return failure();
  }
  return getElemTy(srcTy);
}

static FailureOr<Type> verifyThreeMatchingElementTypes(Operation *op, Type t0,
                                                       Type t1, Type td) {
  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    op->emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op->emitOpError(
        "expects src0, src1, and dst to have the same element type");
    return failure();
  }
  return e0;
}

static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                   Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  FailureOr<Type> elem =
      verifyThreeMatchingElementTypes(op, src0Ty, src1Ty, dstTy);
  if (failed(elem))
    return failure();
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst"))) {
    return failure();
  }
  return *elem;
}

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

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", allowLowPrecision)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }
  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] || lhsShape[1] != rhsShape[0])) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }
  return verifyMatmulValidSizes(op, lhsTy, rhsTy, 0);
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                       allowLowPrecision))) {
    return failure();
  }

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb) {
    return success();
  }

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  }
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects dst to use the col_major blayout on A5");
  }

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  }
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  }
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  }
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           bool allowLowPrecision) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                     allowLowPrecision);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                   allowLowPrecision);
  }
  return failure();
}

static LogicalResult verifyGemvValidShapes(Operation *op, Type lhsTy,
                                           Type rhsTy, Type dstTy,
                                           bool dstMayBeNonTile) {
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1)
    return op->emitOpError(
        "expects lhs valid_shape[0] to be 1 for tgemv");
  if ((!dstMayBeNonTile || isa<pto::TileBufType>(dstTy)) &&
      dstValid[0] != ShapedType::kDynamic && dstValid[0] != 1)
    return op->emitOpError(
        "expects dst valid_shape[0] to be 1 for tgemv");
  if (lhsValid[1] != ShapedType::kDynamic &&
      rhsValid[0] != ShapedType::kDynamic && lhsValid[1] != rhsValid[0])
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  if (rhsValid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic && rhsValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  return success();
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace) {
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT) {
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");
  }

  return verifyGemvValidShapes(op, lhsTy, rhsTy, dstTy,
                               /*dstMayBeNonTile=*/true);
}
static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy))) {
    return failure();
  }
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyA5MxMatTileOperands(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true))) {
    return failure();
  }

  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  if (lhsShape.size() == mlir::pto::kValue2 && rhsShape.size() == mlir::pto::kValue2) {
      int64_t lhsK = lhsShape[1];
      int64_t rhsK = rhsShape[0];
      auto checkPhysicalK = [&](int64_t value, StringRef name) -> LogicalResult {
          if (value != ShapedType::kDynamic && (value < 1 || (value % mlir::pto::kValue64) != 0)) {
              return op->emitOpError() << "expects " << name
                                       << " physical K shape to be a positive multiple of 64 on A5";
          }
          return success();
      };
      if (failed(checkPhysicalK(lhsK, "lhs"))) {
          return failure();
      }
      if (failed(checkPhysicalK(rhsK, "rhs"))) {
          return failure();
      }
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > mlir::pto::kValue4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > mlir::pto::kValue4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > mlir::pto::kValue4095))) {
        return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
    }
  }
  return success();
}

static int64_t ceilDivKnown(int64_t value, int64_t divisor) {
  if (divisor == 0 || divisor < 0) {
    return ShapedType::kDynamic;
  }
  if (value == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }
  return (value + divisor - 1) / divisor;
}

static LogicalResult verifyA5MxMatScaleDims(
    Operation *op, StringRef scaleName, ArrayRef<int64_t> scaleDims,
    ArrayRef<int64_t> lhsDims, ArrayRef<int64_t> rhsDims, StringRef dimsName,
    bool isLeftScale) {
    if (scaleDims.size() != mlir::pto::kValue2 || lhsDims.size() != mlir::pto::kValue2 ||
        rhsDims.size() != mlir::pto::kValue2) {
        return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 " << dimsName;
    }
  int64_t scaleK = ceilDivKnown(lhsDims[1], 32);
  int64_t expectedRows = isLeftScale ? lhsDims[0] : scaleK;
  int64_t expectedCols = isLeftScale ? scaleK : rhsDims[1];
  if (!hasCompatibleKnownExtent(scaleDims[0], expectedRows) ||
      !hasCompatibleKnownExtent(scaleDims[1], expectedCols)) {
    return op->emitOpError()
           << "expects " << scaleName << " " << dimsName << " to be "
           << (isLeftScale ? "[M, ceil(K/32)]" : "[ceil(K/32), N]");
  }
  return success();
}

static LogicalResult verifyA5MxMatScaleLayout(Operation *op,
                                              pto::TileBufType scaleTb,
                                              StringRef scaleName,
                                              bool isLeftScale) {
  if (scaleTb.getBLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::BLayout::RowMajor
                                       : pto::BLayout::ColMajor)) {
    return op->emitOpError() << "expects " << scaleName << " to use the "
                             << (isLeftScale ? "row_major" : "col_major")
                             << " blayout on A5";
  }
  if (scaleTb.getSLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::SLayout::RowMajor
                                       : pto::SLayout::ColMajor)) {
    return op->emitOpError() << "expects " << scaleName << " to use the "
                             << (isLeftScale ? "row_major" : "col_major")
                             << " slayout on A5";
  }
  if (scaleTb.getSFractalSizeI32() != mlir::pto::kValue32) {
      return op->emitOpError() << "expects " << scaleName << " to use fractal=32 on A5";
  }
  return success();
}

static LogicalResult verifyA5MxScaleStorage(Operation *op, Type scaleTy,
                                            StringRef scaleName) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError()
           << "expects " << scaleName << " to be in the scaling address space";
  return success();
}

static LogicalResult verifyA5MxMatScaleTile(Operation *op, Type scaleTy,
                                            Type lhsTy, Type rhsTy,
                                            StringRef scaleName,
                                            bool isLeftScale) {
  if (failed(verifyA5MxScaleStorage(op, scaleTy, scaleName)))
    return failure();

  if (failed(verifyA5MxMatScaleDims(op, scaleName, getShapeVec(scaleTy),
                                    getShapeVec(lhsTy), getShapeVec(rhsTy),
                                    "shape", isLeftScale))) {
    return failure();
  }
  if (failed(verifyA5MxMatScaleDims(
          op, scaleName, getValidShapeVec(scaleTy), getValidShapeVec(lhsTy),
          getValidShapeVec(rhsTy), "valid_shape", isLeftScale))) {
    return failure();
  }

  auto scaleTb = dyn_cast<pto::TileBufType>(scaleTy);
  if (!scaleTb) {
    return success();
  }
  return verifyA5MxMatScaleLayout(op, scaleTb, scaleName, isLeftScale);
}

static LogicalResult verifyA5MxMatScaleTiles(Operation *op, Type lhsScaleTy,
                                             Type rhsScaleTy, Type lhsTy,
                                             Type rhsTy) {
  if (failed(verifyA5MxMatScaleTile(op, lhsScaleTy, lhsTy, rhsTy, "a_scale",
                                    /*isLeftScale=*/true))) {
    return failure();
  }
  return verifyA5MxMatScaleTile(op, rhsScaleTy, lhsTy, rhsTy, "b_scale",
                                /*isLeftScale=*/false);
}

static LogicalResult verifyA5MxGemvTileOperands(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true)) ||
      failed(verifyMatmulValidSizes(op, lhsTy, rhsTy, 1))) {
    return failure();
  }
  return verifyGemvValidShapes(op, lhsTy, rhsTy, dstTy,
                               /*dstMayBeNonTile=*/false);
}

static LogicalResult verifyA5MxGemvScaleTile(Operation *op, Type scaleTy,
                                             Type lhsTy, Type rhsTy,
                                             StringRef scaleName,
                                             bool isLeftScale) {
  if (failed(verifyA5MxScaleStorage(op, scaleTy, scaleName)))
    return failure();

  auto scaleShape = getShapeVec(scaleTy);
  auto scaleValid = getValidShapeVec(scaleTy);
  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (scaleShape.size() != mlir::pto::kValue2 || scaleValid.size() != mlir::pto::kValue2 ||
      lhsShape.size() != mlir::pto::kValue2 || rhsShape.size() != mlir::pto::kValue2 ||
      lhsValid.size() != mlir::pto::kValue2 || rhsValid.size() != mlir::pto::kValue2) {
      return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 shape/valid_shape";
  }

  int64_t logicalM = lhsValid[0];
  int64_t logicalK = lhsValid[1];
  int64_t logicalN = rhsValid[1];
  int64_t scaleK = ceilDivKnown(logicalK, 32);

  int64_t expectedShapeRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedShapeCols = isLeftScale ? scaleK : rhsShape[1];
  int64_t expectedValidRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedValidCols = isLeftScale ? scaleK : logicalN;

  if (!hasCompatibleKnownExtent(scaleShape[0], expectedShapeRows) ||
      !hasCompatibleKnownExtent(scaleShape[1], expectedShapeCols) ||
      !hasCompatibleKnownExtent(scaleValid[0], expectedValidRows) ||
      !hasCompatibleKnownExtent(scaleValid[1], expectedValidCols)) {
    if (isLeftScale) {
      return op->emitOpError()
             << "expects " << scaleName
             << " shape/valid_shape to be [M, ceil(K/32)]";
    }
    return op->emitOpError()
           << "expects " << scaleName
           << " shape/valid_shape to be [ceil(K/32), aligned_N]/[ceil(K/32), N]";
  }
  return success();
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias"))) {
    return failure();
  }
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS) {
    return op->emitOpError("expects bias to be in the bias address space");
  }
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1) {
    return op->emitOpError("expects bias to have 1 row");
  }
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32()) {
      return op->emitOpError("expects bias to have element type f32");
    }
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias))) {
    return failure();
  }
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError("expects bias to use the row_major blayout on A5");
    }
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
  return failure();
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(mlir::pto::kValue32) && isInt8(lhsElemTy) && isInt8(rhsElemTy)) {
      return success();
  }

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy)) {
    return success();
  }

  auto isA5TMatmulFp8Type = [](Type ty) {
    return isPTOFloat8Type(ty);
  };
  if (isA5 && dstElemTy.isF32()) {
    if (isA5TMatmulFp8Type(lhsElemTy) && isA5TMatmulFp8Type(rhsElemTy)) {
      return success();
    }
    if (isPTOHiFloat8Type(lhsElemTy) && lhsElemTy == rhsElemTy) {
      return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", (f32, fp8, fp8), or (f32, hif8, hif8)" : "");
}

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    Type elemTy = *elemOr;
    if (elemTy.isInteger(16) || elemTy.isF16() || elemTy.isF32()) {
      return success();
    }
    return emitOpError("expects element type to be i16/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return emitOpError("taddrelu is only supported on A2/A3 targets");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddCOp::verify() {
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/src2/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd) {
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  }
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

static LogicalResult verifyTAxpyCommon(TAxpyOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (op.getScalar().getType() != getElemTy(srcTy)) {
    return op.emitOpError("expects scalar type to match src element type");
  }
  if (getShapeVec(srcTy) != getShapeVec(dstTy)) {
    return op.emitOpError("expects src and dst to have the same shape");
  }
  return success();
}

static LogicalResult verifyTAxpyArch(TAxpyOp op, bool allowBf16) {
    if (failed(verifyTAxpyCommon(op))) {
      return failure();
    }
    Type srcElem = getElemTy(op.getSrc().getType());
    Type dstElem = getElemTy(op.getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32)) {
      return op.emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    }
    if (!(dstElem.isF16() || dstElem.isF32() ||
          (allowBf16 && dstElem.isBF16()))) {
      return op.emitOpError() << "expects " << (allowBf16 ? "A5" : "A2/A3")
                              << " taxpy dst element type to be "
                              << (allowBf16 ? "f16/bf16/f32" : "f16/f32");
    }
    if (!(srcElem.isF16() || srcElem.isF32() ||
          (allowBf16 && srcElem.isBF16()))) {
      return op.emitOpError() << "expects " << (allowBf16 ? "A5" : "A2/A3")
                              << " taxpy src element type to be "
                              << (allowBf16 ? "f16/bf16/f32" : "f16/f32");
    }
    return success();
}

LogicalResult pto::TAxpyOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTAxpyArch(*this, /*allowBf16=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTAxpyArch(*this, /*allowBf16=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd) {
    return emitOpError("expects src0/src1/dst to have the same shape");
  }
  return success();
}

static LogicalResult verifyIntegerWidths(Operation *op, Type type,
                                         ArrayRef<unsigned> widths,
                                         StringRef diagnostic) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || !llvm::is_contained(widths, intType.getWidth()))
    return op->emitOpError(diagnostic);
  return success();
}

static LogicalResult verifyBitwiseBinaryOp(Operation *op, Type src0,
                                           Type src1, Type dst,
                                           StringRef opName) {
  auto verifyCommon = [&]() {
    return verifyMatchingRowMajorBinaryTileOpCommon(op, src0, src1, dst);
  };
  auto verifyFor = [&](StringRef arch) -> LogicalResult {
    auto elem = verifyCommon();
    if (failed(elem))
      return failure();
    std::string diagnostic =
        (Twine("expects ") + arch + " " + opName +
         " src0, src1, and dst element type to be i8/i16/i32")
            .str();
    return verifyIntegerWidths(op, *elem, {8, 16, 32}, diagnostic);
  };
  auto verifyA2A3 = [&]() { return verifyFor("A2/A3"); };
  auto verifyA5 = [&]() { return verifyFor("A5"); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyBitwiseScalarOp(Operation *op, Value src,
                                           Value dst, StringRef opName) {
  auto verifyFor = [&](bool isA5) -> LogicalResult {
    auto elem = verifyDistinctRowMajorUnaryTileOpCommon(op, src, dst, "src",
                                                        "dst");
    if (failed(elem))
      return failure();
    std::string diagnostic =
        (Twine("expects ") + (isA5 ? "A5 " : "A2/A3 ") + opName +
         " src, scalar, and dst element type to be " +
         (isA5 ? "i8/i16/i32" : "i8/i16"))
            .str();
    return verifyIntegerWidths(op, *elem, isA5 ? ArrayRef<unsigned>{8, 16, 32}
                                               : ArrayRef<unsigned>{8, 16},
                               diagnostic);
  };
  auto verifyA2A3 = [&]() { return verifyFor(false); };
  auto verifyA5 = [&]() { return verifyFor(true); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

LogicalResult pto::TAndOp::verify() {
  return verifyBitwiseBinaryOp(getOperation(), getSrc0().getType(),
                               getSrc1().getType(), getDst().getType(), "tand");
}

static LogicalResult verifyTConcatValidShapes(TConcatOp op, ArrayRef<int64_t> v0,
                                              ArrayRef<int64_t> v1,
                                              ArrayRef<int64_t> vd, Type dstTy) {
    if (v0.size() != mlir::pto::kValue2 || v1.size() != mlir::pto::kValue2 || vd.size() != mlir::pto::kValue2) {
        return op.emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    }
  if (v0[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
      v0[0] != vd[0]) {
    return op.emitOpError("expects src0 valid row to match dst valid row");
  }
  if (v1[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
      v1[0] != vd[0]) {
    return op.emitOpError("expects src1 valid row to match dst valid row");
  }
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == mlir::pto::kValue2 && dstShape[1] != ShapedType::kDynamic && v0[1] != ShapedType::kDynamic &&
      v1[1] != ShapedType::kDynamic && v0[1] + v1[1] > dstShape[1]) {
      return op.emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
  }
  return success();
}

static FailureOr<Type> verifyThreeMatchingTiles(Operation *op, Type t0,
                                                Type t1, Type td,
                                                Type optionalTmp = {}) {
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst")) ||
      (optionalTmp && failed(verifyVecTileCommon(op, optionalTmp, "tmp")))) {
    return failure();
  }
  return verifyThreeMatchingElementTypes(op, t0, t1, td);
}

static FailureOr<Type> verifyTConcatCommon(TConcatOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  FailureOr<Type> elem = verifyThreeMatchingTiles(op, t0, t1, td);
  if (failed(elem))
    return failure();

  auto v0 = getValidShapeVec(op.getSrc0());
  auto v1 = getValidShapeVec(op.getSrc1());
  auto vd = getValidShapeVec(op.getDst());
  if (failed(verifyTConcatValidShapes(op, v0, v1, vd, td))) {
    return failure();
  }

  return *elem;
}

static LogicalResult verifyTConcatElemType(TConcatOp op, Type elem) {
  if (elem.isF16() || elem.isF32() || elem.isBF16()) {
    return success();
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
              it.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("expects element type to be i8, i16, i32, f16, f32, or bf16");
  }
  return success();
}

static LogicalResult verifyTConcatLocVec(TConcatOp op, Type ty, StringRef name) {
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op.emitOpError() << "expects " << name << " to use loc=vec";
  }
  return success();
}

static FailureOr<Type> verifyTConcatBase(TConcatOp op) {
  auto elem = verifyTConcatCommon(op);
  if (failed(elem))
    return failure();
  if (failed(verifyTConcatLocVec(op, op.getSrc0().getType(), "src0")) ||
      failed(verifyTConcatLocVec(op, op.getSrc1().getType(), "src1")) ||
      failed(verifyTConcatLocVec(op, op.getDst().getType(), "dst")))
    return failure();
  return *elem;
}

static LogicalResult verifyTConcatA2A3(TConcatOp op) {
  auto elem = verifyTConcatBase(op);
  return failed(elem) ? failure() : verifyTConcatElemType(op, *elem);
}

static LogicalResult verifyTConcatA5(TConcatOp op) {
  auto elem = verifyTConcatBase(op);
  if (failed(elem))
    return failure();
  if (!isRowMajorTileBuf(op.getSrc0().getType()) || !isRowMajorTileBuf(op.getSrc1().getType()) ||
      !isRowMajorTileBuf(op.getDst().getType())) {
    return op.emitOpError("expects src0, src1, and dst to use row-major layout");
  }
  return verifyTConcatElemType(op, *elem);
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTConcatA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTConcatA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTConcatidxValidRows(
    TConcatidxOp op, ArrayRef<int64_t> dstValid) {
  SmallVector<Value, 4> values = {op.getSrc0(), op.getSrc1(), op.getSrc0Idx(),
                                  op.getSrc1Idx()};
  SmallVector<StringRef, mlir::pto::kValue4> names = {"src0", "src1", "src0Idx", "src1Idx"};
  for (auto [value, name] : llvm::zip_equal(values, names)) {
    auto valid = getValidShapeVec(value);
    if (valid.size() != mlir::pto::kValue2) {
        return op.emitOpError("expects all operands to have rank-2 valid_shape");
    }
    if (valid[0] != ShapedType::kDynamic &&
        dstValid[0] != ShapedType::kDynamic && valid[0] != dstValid[0]) {
      return op.emitOpError("expects ") << name
                                         << " valid row to match dst valid row";
    }
  }
  return success();
}

static FailureOr<std::pair<Type, Type>> verifyTConcatidxElementAgreement(
    TConcatidxOp op, ArrayRef<Type> types) {
  Type dataElem = getElemTy(types[0]);
  Type secondDataElem = getElemTy(types[1]);
  Type dstElem = getElemTy(types[4]);
  if (!dataElem || !secondDataElem || !dstElem) {
    op.emitOpError("failed to get element type for data operands");
    return failure();
  }
  if (dataElem != secondDataElem || dataElem != dstElem) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  Type indexElem = getElemTy(types[2]);
  Type secondIndexElem = getElemTy(types[3]);
  if (!indexElem || !secondIndexElem) {
    op.emitOpError("failed to get element type for index operands");
    return failure();
  }
  if (indexElem != secondIndexElem) {
    op.emitOpError("expects src0Idx and src1Idx to have the same element type");
    return failure();
  }
  return std::make_pair(dataElem, indexElem);
}

static LogicalResult verifyTConcatidxIndexColumns(TConcatidxOp op) {
  for (Value index : {op.getSrc0Idx(), op.getSrc1Idx()}) {
    auto valid = getValidShapeVec(index);
    if (valid[1] != ShapedType::kDynamic && valid[1] < 1)
      return op.emitOpError() << "expects "
                              << (index == op.getSrc0Idx() ? "src0Idx"
                                                           : "src1Idx")
                              << " valid_col >= 1";
  }
  return success();
}

static FailureOr<std::pair<Type, Type>> verifyTConcatidxCommon(
    TConcatidxOp op) {
    SmallVector<Type, mlir::pto::kValue5> types = {
        op.getSrc0().getType(), op.getSrc1().getType(), op.getSrc0Idx().getType(), op.getSrc1Idx().getType(),
        op.getDst().getType()};
    SmallVector<StringRef, mlir::pto::kValue5> names = {"src0", "src1", "src0Idx", "src1Idx", "dst"};
    for (auto [type, name] : llvm::zip_equal(types, names)) {
        if (failed(verifyTileBufCommon(op, type, name))) {
            return failure();
        }
    }
  auto elementTypes = verifyTConcatidxElementAgreement(op, types);
  if (failed(elementTypes))
    return failure();
  auto dstValid = getValidShapeVec(op.getDst());
  if (dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects all operands to have rank-2 valid_shape");
  }
  if (failed(verifyTConcatidxValidRows(op, dstValid))) {
    return failure();
  }
  if (failed(verifyTConcatidxIndexColumns(op)))
    return failure();
  return *elementTypes;
}

static LogicalResult verifyTConcatidxElementTypes(TConcatidxOp op,
                                                  Type dataElem,
                                                  Type idxElem) {
    // Data element type: f16, f32, bf16, i8, i16, i32 (signless).
    if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
      auto it = mlir::dyn_cast<IntegerType>(dataElem);
      if (!it || !it.isSignless() ||
          (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
           it.getWidth() != mlir::pto::kValue32)) {
          return op.emitOpError() << "expects data element type to be i8, i16, i32, f16, f32, or bf16";
      }
    }

    // Index element type: i8, i16, i32 (signless).
    auto it = mlir::dyn_cast<IntegerType>(idxElem);
    if (!it || !it.isSignless() ||
        (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
         it.getWidth() != mlir::pto::kValue32)) {
        return op.emitOpError() << "expects index element type to be i8, i16, or i32";
    }
    return success();
}

static LogicalResult verifyTConcatidxLocVec(TConcatidxOp op, Type ty,
                                            StringRef name) {
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC) {
      return op.emitOpError() << "expects " << name << " to use loc=vec";
    }
    return success();
}

static LogicalResult verifyTConcatidxArch(TConcatidxOp op,
                                          bool requireRowMajor) {
    auto elemOr = verifyTConcatidxCommon(op);
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyTConcatidxLocVec(op, op.getSrc0().getType(), "src0")) ||
        failed(verifyTConcatidxLocVec(op, op.getSrc1().getType(), "src1")) ||
        failed(verifyTConcatidxLocVec(op, op.getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyTConcatidxLocVec(op, op.getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyTConcatidxLocVec(op, op.getDst().getType(), "dst"))) {
      return failure();
    }
    if (requireRowMajor &&
        (!isRowMajorTileBuf(op.getSrc0().getType()) ||
         !isRowMajorTileBuf(op.getSrc1().getType()) ||
         !isRowMajorTileBuf(op.getSrc0Idx().getType()) ||
         !isRowMajorTileBuf(op.getSrc1Idx().getType()) ||
         !isRowMajorTileBuf(op.getDst().getType()))) {
      return op.emitOpError(
          "expects all operands to use row-major layout");
    }
    return verifyTConcatidxElementTypes(op, elemOr->first, elemOr->second);
}

LogicalResult pto::TConcatidxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTConcatidxArch(*this, /*requireRowMajor=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTConcatidxArch(*this, /*requireRowMajor=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  return verifyBitwiseScalarOp(getOperation(), getSrc(), getDst(), "tands");
}

struct TCILikeParseState {
  OpAsmParser::UnresolvedOperand source;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type sourceType;
  Type tmpType;
  Type dstType;
  bool hasTmp = false;
};

static ParseResult parseTCILikeSyntax(OpAsmParser &parser,
                                     OperationState &result,
                                     TCILikeParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.source)) {
    return failure();
  }
  state.hasTmp = succeeded(parser.parseOptionalComma());
  if (state.hasTmp && parser.parseOperand(state.tmp)) {
    return failure();
  }
  if (parser.parseColonType(state.sourceType)) {
    return failure();
  }
  if (state.hasTmp) {
    if (parser.parseComma() || parser.parseType(state.tmpType)) {
      return failure();
    }
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(state.dst) || parser.parseColonType(state.dstType) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

static ParseResult resolveOptionalUnaryAndAddSegments(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType, bool hasTmp,
    OpAsmParser::UnresolvedOperand tmp, Type tmpType,
    OpAsmParser::UnresolvedOperand dst, Type dstType) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpType, result.operands)) ||
      parser.resolveOperand(dst, dstType, result.operands))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

static ParseResult parseTCILikeOp(OpAsmParser &parser, OperationState &result) {
  TCILikeParseState state;
  if (failed(parseTCILikeSyntax(parser, result, state)))
    return failure();

  return resolveOptionalUnaryAndAddSegments(
      parser, result, state.source, state.sourceType, state.hasTmp, state.tmp,
      state.tmpType, state.dst, state.dstType);
}

static void printTCILikeOp(OpAsmPrinter &p, Operation *op, Value s, Value tmp,
                           Value dst) {
  p << " ins(" << s;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << s.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TCIOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTCILikeOp(parser, result);
}

void mlir::pto::TCIOp::print(OpAsmPrinter &p) {
  printTCILikeOp(p, getOperation(), getOperand(0), getTmp(), getDst());
}

static LogicalResult verifyTCITmp(TCIOp op, unsigned bitWidth) {
    auto tmpTy = mlir::dyn_cast<TileBufType>(op.getTmp().getType());
    if (!tmpTy) {
      return op.emitOpError("expects tmp to be a tile buffer");
    }
    auto tmpSpace =
        mlir::dyn_cast_or_null<AddressSpaceAttr>(tmpTy.getMemorySpace());
    if (!tmpSpace || tmpSpace.getAddressSpace() != AddressSpace::VEC) {
      return op.emitOpError("expects tmp to be in vec address space");
    }
    Type tmpElemTy = tmpTy.getElementType();
    if (!(tmpElemTy.isF32() || tmpElemTy.isInteger(mlir::pto::kValue32))) {
        return op.emitOpError("expects A2/A3 tmp element type to be a 4-byte type");
    }
    if (tmpTy.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor)) {
      return op.emitOpError("expects tmp blayout to be row_major");
    }
    if (tmpTy.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox)) {
      return op.emitOpError("expects tmp slayout to be none_box");
    }
    if (tmpTy.getSFractalSizeI32() != 512) {
      return op.emitOpError("expects tmp fractal size to be 512");
    }
    auto tmpBytes = getStaticByteSize(tmpTy);
    if (!tmpBytes) {
      return op.emitOpError("expects tmp to have static byte size");
    }
    uint64_t minTmpBytes = bitWidth == 32 ? 768 : 1792;
    if (*tmpBytes < minTmpBytes) {
      return op.emitOpError("expects A2/A3 tmp capacity to be at least ")
             << minTmpBytes << " bytes for " << bitWidth
             << "-bit dst element type";
    }
    return success();
}

LogicalResult pto::TCIOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
      (getTmp() && failed(verifyTileBufCommon(*this, getTmp().getType(), "tmp")))) {
    return failure();
  }
  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy) {
    return emitOpError("expects dst element type to be integer");
  }
  unsigned bw = elemTy.getWidth();
  if (bw != mlir::pto::kValue16 && bw != mlir::pto::kValue32) {
      return emitOpError("expects dst element type to be i16/i32");
  }
  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5 &&
      failed(verifyTCITmp(*this, bw))) {
    return failure();
  }

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy) {
    return emitOpError("expects S to be integer");
  }

  if (sTy != elemTy) {
    return emitOpError("expects S and dst element type to be exactly the same type");
  }
  auto shape = getShapeVec(dstTy);
  if (shape.size() != mlir::pto::kValue2) {
      return emitOpError("expects dst to be rank-2");
  }
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1) {
    return emitOpError("expects dst cols to be different from 1");
  }

  return success();
}

LogicalResult pto::TTriOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy) {
    return emitOpError("expects diagonal to be an integer operand");
  }

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1) {
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");
  }

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false)) {
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        }
        return success();
      },
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true)) {
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        }
        return success();
      });
}
