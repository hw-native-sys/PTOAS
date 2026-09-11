// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
