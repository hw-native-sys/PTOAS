// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTCmpElementTypes(TCmpOp op, Type t0, Type t1,
                                             Type td, bool isA5) {
  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return op.emitOpError("failed to get element type for src0/src1/dst");
  if (e0 != e1)
    return op.emitOpError("expects src0 and src1 to have the same element type");
  bool inputOk = isA5 ? isSupportedVecElemType(e0, /*allowBf16=*/true,
                                               /*allowInt8=*/true)
                      : (e0.isInteger(32) || e0.isF16() || e0.isF32());
  if (!inputOk)
    return op.emitOpError(isA5
        ? "expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32"
        : "expects A2/A3 tcmp input element type to be i32/f16/f32");
  return ed.isInteger(mlir::pto::kValue8) ? success() : op.emitOpError("expects dst element type to be i8");
}

static LogicalResult verifyTCmpA2ValidShapes(TCmpOp op, Type t0, Type t1,
                                              Type td) {
  auto valid0 = getValidShapeVec(t0);
  auto valid1 = getValidShapeVec(t1);
  auto validd = getValidShapeVec(td);
  if (valid0.size() != 2 || valid1.size() != 2 || validd.size() != 2)
    return op.emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  if (!hasCompatibleKnownExtent(valid0[0], valid1[0]))
    return op.emitOpError("expects src0 and src1 to have the same valid row");
  if (!hasCompatibleKnownExtent(valid0[1], valid1[1]))
    return op.emitOpError("expects src0 and src1 to have the same valid column");
  if (!hasCompatibleKnownExtent(valid0[0], validd[0]))
    return op.emitOpError("expects src0 valid row to equal dst valid row");
  return success();
}

static LogicalResult verifyTCmpA5Shapes(TCmpOp op, Type t0, Type t1,
                                        Type td) {
  auto src0Shape = getShapeVec(t0);
  auto src1Shape = getShapeVec(t1);
  if (src0Shape != src1Shape)
    return op.emitOpError("expects src0 and src1 to have the same shape");

  // dst carries a packed predicate mask, so its column extent differs from
  // the source tiles. The valid row extent must still match.
  auto src0Valid = getValidShapeVec(t0);
  auto dstValid = getValidShapeVec(td);
  bool validShapesAreRankTwo = src0Valid.size() == 2 && dstValid.size() == 2;
  if (!validShapesAreRankTwo)
    return op.emitOpError("expects src0 and dst to have rank-2 valid_shape");
  if (!hasCompatibleKnownExtent(src0Valid[0], dstValid[0]))
    return op.emitOpError("expects src0 and dst to have the same valid_shape[0]");
  return success();
}

static LogicalResult verifyTCmpArch(TCmpOp op, bool isA5) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  auto verifyTile = [&](Type type, StringRef name) {
    return isA5 ? verifyTileBufCommon(op, type, name)
                : verifyVecTileStorage(op, type, name);
  };
  if (failed(verifyTile(t0, "src0")) || failed(verifyTile(t1, "src1")) ||
      failed(verifyTile(td, "dst")) ||
      failed(verifyTCmpElementTypes(op, t0, t1, td, isA5)))
    return failure();
  if (!isA5)
    return verifyTCmpA2ValidShapes(op, t0, t1, td);
  return verifyTCmpA5Shapes(op, t0, t1, td);
}

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTCmpArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTCmpArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
static LogicalResult verifyTCmpSArch(TCmpSOp op, bool allowInt8) {
    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
        failed(verifyVecTileStorage(op, dstTy, "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!((allowInt8 && elemTy.isInteger(mlir::pto::kValue8)) || elemTy.isInteger(mlir::pto::kValue16) ||
          elemTy.isInteger(mlir::pto::kValue32) || elemTy.isF16() || elemTy.isF32())) {
        return op.emitOpError(
            allowInt8 ? "expects A5 tcmps input element type to be i8/i16/i32/f16/f32" :
                        "expects A2/A3 tcmps input element type to be i16/i32/f16/f32");
    }
    if (!op.getScalar().getType().isIntOrIndexOrFloat()) {
      return op.emitOpError("expects scalar to be integer, index, or float");
    }
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2) {
        return op.emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0]) {
      return op.emitOpError("expects src and dst to have the same valid_shape[0]");
    }
    return success();
}

LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTCmpSArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTCmpSArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return emitOpError("expects tcolexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2) {
      return emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  }
  return success();
}
static bool isSupportedTColExpandElem(Type elemTy, PTOArch targetArch,
                                     bool allowIntegerTypes) {
  if (elemTy.isF16() || elemTy.isF32()) {
    return true;
  }
  if (!allowIntegerTypes) {
    return false;
  }
  if (elemTy.isInteger(16) || elemTy.isInteger(32)) {
    return true;
  }
  return targetArch == PTOArch::A5 && elemTy.isInteger(mlir::pto::kValue8);
}

static LogicalResult verifyTColExpandRowMajor(Operation *op, Type type,
                                              StringRef name) {
  auto tileTy = dyn_cast<TileBufType>(type);
  if (tileTy && tileTy.getBLayoutValueI32() != 0) {
    return op->emitOpError() << "expects " << name
                             << " to use row-major layout";
  }
  return success();
}

static LogicalResult verifyTColExpandElementTypes(Operation *op, Type e0,
                                                  Type e1, Type ed,
                                                  PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  bool supported = isSupportedTColExpandElem(e0, targetArch, allowIntegerTypes) &&
                   isSupportedTColExpandElem(e1, targetArch, allowIntegerTypes) &&
                   isSupportedTColExpandElem(ed, targetArch, allowIntegerTypes);
  if (supported)
    return success();
  if (!allowIntegerTypes)
    return op->emitOpError() << "expects " << opName
                             << " element type to be f16 or f32";
  if (targetArch == PTOArch::A5)
    return op->emitOpError() << "expects A5 " << opName
                             << " element type to be i8/i16/i32/f16/f32";
  return op->emitOpError() << "expects A2/A3 " << opName
                           << " element type to be i16/i32/f16/f32";
}

static LogicalResult verifyTColExpandValidColumn(Operation *op, Type t1,
                                                 Type td) {
  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == mlir::pto::kValue2 && dstValid.size() == mlir::pto::kValue2 &&
      src1Valid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic && src1Valid[1] != dstValid[1])
      return op->emitOpError("expects src1 valid_shape[1] to equal dst valid_shape[1]");
  return success();
}

static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td)) {
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }
