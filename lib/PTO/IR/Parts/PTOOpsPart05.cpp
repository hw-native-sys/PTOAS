// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

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

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for src0/src1/dst");
  }

  if (failed(verifyTColExpandElementTypes(op, e0, e1, ed, targetArch,
                                         opName, allowIntegerTypes)))
    return failure();

  if (getShapeVec(t0) != getShapeVec(td)) {
    return op->emitOpError("expects src0/dst to have same shape");
  }
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst"))) {
    return failure();
  }

  if (failed(verifyTColExpandRowMajor(op, t0, "src0")) ||
      failed(verifyTColExpandRowMajor(op, t1, "src1")) ||
      failed(verifyTColExpandRowMajor(op, td, "dst"))) {
    return failure();
  }

  return verifyTColExpandValidColumn(op, t1, td);
}
LogicalResult pto::TColExpandMulOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmul",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandAddOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandadd",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    bool allowIntegerTypes = (targetArch == PTOArch::A5);
    return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        targetArch, "tcolexpanddiv",
                                        /*allowIntegerTypes=*/allowIntegerTypes);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandSubOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandsub",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandExpdifOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandexpdif",
                                      /*allowIntegerTypes=*/false);
}
LogicalResult pto::TColExpandMaxOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmax",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandMinOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmin",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColMaxOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmax element type to be f16/f32/i16/i32",
      "expects A5 tcolmax element type to be i8/i16/i32/f16/bf16/f32");
}

static LogicalResult verifyTColArgReductionOp(Operation *op, Type srcTy,
                                              Value tmp, Type dstTy) {
  if (!tmp)
    return verifyTColArgReductionNoTmp(op, srcTy, dstTy);
  auto verifyA2A3 = [&]() {
    return verifyTColArgReductionOpA2A3(op, srcTy, tmp.getType(), dstTy);
  };
  auto verifyA5 = [&]() {
    return verifyTColArgReductionOpA5(op, srcTy, tmp.getType(), dstTy);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

LogicalResult pto::TColArgMaxOp::verify() {
  return verifyTColArgReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                  getDst().getType());
}

LogicalResult pto::TColMinOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmin element type to be f16/f32/i16/i32",
      "expects A5 tcolmin element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMinOp::verify() {
  return verifyTColArgReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                  getDst().getType());
}

static ParseResult resolveRequiredOperand(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand operand, Type type) {
  return parser.resolveOperand(operand, type, result.operands);
}

static ParseResult resolveOptionalOperand(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand operand, Type type, bool present) {
  return present ? resolveRequiredOperand(parser, result, operand, type)
                 : success();
}

static ParseResult resolveOptionalTmpOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand src, Type srcTy,
    OpAsmParser::UnresolvedOperand tmp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dst, Type dstTy, bool hasTmp) {
  if (failed(resolveRequiredOperand(parser, result, src, srcTy)) ||
      failed(resolveOptionalOperand(parser, result, tmp, tmpTy, hasTmp)))
    return failure();
  return resolveRequiredOperand(parser, result, dst, dstTy);
}

static ParseResult parseTColSumInputs(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &src, Type &srcTy,
    OpAsmParser::UnresolvedOperand &tmp, Type &tmpTy, bool &hasTmp) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src))
    return failure();
  hasTmp = succeeded(parser.parseOptionalComma());
  if (!hasTmp)
    return parser.parseColonType(srcTy) || parser.parseRParen() ? failure()
                                                                : success();
  if (parser.parseOperand(tmp) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcTy) || parser.parseComma() ||
      parser.parseType(tmpTy) || parser.parseRParen())
    return failure();
  return success();
}

static ParseResult parseTColSumOutput(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &dst, Type &dstTy) {
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  return success();
}

ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (failed(parseTColSumInputs(parser, result, src, srcTy, tmp, tmpTy,
                                hasTmp)) ||
      failed(parseTColSumOutput(parser, dst, dstTy))) {
    return failure();
  }

  // Parse any remaining attributes (for format 1)
  if (!hasTmp && parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return resolveOptionalTmpOperands(parser, result, src, srcTy, tmp, tmpTy,
                                    dst, dstTy, hasTmp);
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, mlir::pto::kValue2> elidedAttrs = {"operandSegmentSizes"};
    if (!getIsBinaryAttr() || getIsBinaryAttr().getValue() == false) {
      elidedAttrs.push_back("isBinary");
    }
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : " << getSrc().getType() << ", " << getTmp().getType() << ")";
  } else {
    // Format 1: ins(%src : type) outs(%dst : type)
    p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  }

  p << " outs(" << getDst() << " : " << getDst().getType() << ")";

  // Print remaining attributes for format 1 (excluding isBinary)
  if (!getTmp()) {
      SmallVector<StringRef, mlir::pto::kValue2> elidedAttrs = {"isBinary", "operandSegmentSizes"};
      p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

static LogicalResult verifyTColSumTmp(TColSumOp op, Type srcTy, Type dstTy) {
  if (!op.getTmp()) {
    return success();
  }
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyNDStyleVecTile(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy) ||
      getElemTy(srcTy) != getElemTy(tmpTy)) {
    return op.emitOpError("expects src/tmp/dst element types to match");
  }
  if (failed(verifyTColSumTmpStride(op, srcTy, tmpTy, op.getIsBinary()))) {
    return failure();
  }
  if (!op.getIsBinary()) {
    return success();
  }
  auto srcValid = getValidShapeVec(srcTy);
  auto elemBytes = getElemByteSize(getElemTy(srcTy));
  if (srcValid.size() != mlir::pto::kValue2 || srcValid[0] == ShapedType::kDynamic ||
      srcValid[1] == ShapedType::kDynamic || elemBytes == 0) {
      return op.emitOpError("expects static src valid_shape and element size to verify tcolsum tmp");
  }
  uint64_t requiredBytes =
      static_cast<uint64_t>(ceilDivInt64(srcValid[0], 2)) *
      static_cast<uint64_t>(srcValid[1]) * elemBytes;
  return verifyTmpCapacityAtLeast(op, tmpTy, requiredBytes);
}

static LogicalResult verifyTColSumArch(TColSumOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
    return failure();
  }
  if (op.getTmp() && !op.getIsBinaryAttr()) {
    return op.emitOpError("tmp operand requires isBinary attribute");
  }
  if (failed(verifyTColSumTmp(op, srcTy, dstTy))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects src/dst element types to match");
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/isA5))) {
    return failure();
  }
  Type elem = getElemTy(srcTy);
  bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                   elem.isInteger(32) ||
                   (isA5 && (elem.isBF16() || elem.isInteger(8)));
  if (!supported) {
    return op.emitOpError(isA5
        ? "expects A5 tcolsum element type to be i8/i16/i32/f16/bf16/f32"
        : "expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
  }
  return success();
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTColSumArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTColSumArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColProdOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/false,
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolprod element type to be f16/f32/i16/i32",
      "expects A5 tcolprod element type to be i16/ui16/i32/ui32/f16/bf16/f32");
}

static bool tcvtIsResolvedSubview(Value value) {
  auto alloc = value.getDefiningOp<pto::AllocTileOp>();
  auto semantics =
      alloc ? alloc->getAttrOfType<StringAttr>("pto.view_semantics")
            : StringAttr();
  return semantics && semantics.getValue() == "subview";
}

static bool tcvtNeedsTmp(TCvtOp op, Type srcElem, Type dstElem) {
  if (op.getSatMode() != pto::SaturationMode::OFF) {
    return false;
  }
  return (srcElem.isF32() && dstElem.isInteger(mlir::pto::kValue16)) ||
         (srcElem.isF16() && (dstElem.isInteger(mlir::pto::kValue16) || dstElem.isInteger(mlir::pto::kValue8)));
}

static int64_t computeTCvtTmpRequiredBytes(Type srcElem, Type dstElem,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> dstValid) {
  int64_t rows = dstValid[0], cols = dstValid[1];
  int64_t requiredBytes = 0;
  if (rows > 0 && cols > 0 && srcElem.isF32()) {
    int64_t head = 4 * 64 * std::min<int64_t>(cols / 64, 255);
    int64_t remainder = cols % 64;
    int64_t tail = remainder == 0
                       ? 0
                       : 32 * ((std::min<int64_t>(rows, 255) - 1) *
                                   (srcShape[1] / 8) +
                               llvm::divideCeil(remainder, int64_t{8}));
    requiredBytes = std::max(head, tail);
  } else if (cols > 0 && srcElem.isF16()) {
    int64_t width = std::min<int64_t>(cols, 64);
    int64_t halfToI16 = 32 * llvm::divideCeil(width, int64_t{8});
    int64_t halfToI8 = std::max<int64_t>(
        halfToI16,
        128 + 32 * static_cast<int64_t>(
                       llvm::divideCeil(width, int64_t{16})));
    requiredBytes = dstElem.isInteger(mlir::pto::kValue8) ? halfToI8 : halfToI16;
  }
  return requiredBytes;
}

static LogicalResult verifyTCvtTmp(TCvtOp op, Type srcTy, Type dstTy,
                                   Type srcElem, Type dstElem) {
  if (!op.getTmp()) {
    return success();
  }
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!tcvtNeedsTmp(op, srcElem, dstElem)) {
    return success();
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) || llvm::is_contained(dstValid, ShapedType::kDynamic)) {
      return op.emitOpError("expects static src shape and dst valid_shape to verify tcvt tmp");
  }
  int64_t requiredBytes =
      computeTCvtTmpRequiredBytes(srcElem, dstElem, srcShape, dstValid);
  auto tmpBytes = getStaticByteSize(tmpTy);
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(requiredBytes)) {
    return op.emitOpError()
           << "expects tcvt tmp capacity to be at least " << requiredBytes
           << " bytes";
  }
  return success();
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true))) {
    return failure();
  }
  // A resolved subview keeps its parent's physical shape so the generated
  // Tile retains the parent stride. Its valid shape is the logical tcvt
  // extent, so comparing physical shapes would reject a valid sliced tile.
  if (!tcvtIsResolvedSubview(getSrc()) && !tcvtIsResolvedSubview(getDst()) &&
      failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    }
    return verifyTCvtTmp(*this, srcTy, dstTy, srcElem, dstElem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem)) {
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    }
    if (getTmp() && failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy)) {
      return emitOpError("expects dst to use row-major layout");
    }

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32)) {
      return emitOpError("expects dst element type to be i32 or ui32");
    }

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32) {
        return emitOpError() << "expects " << name << " to be i32/ui32";
      }
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3"))) {
      return failure();
    }

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10) {
      return emitOpError("expects rounds to be 7 or 10");
    }

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32())) {
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32))) {
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);
    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile)) {
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");
    }

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true))) {
      return failure();
    }
    if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
      return emitOpError("scalar must be a scalar type (integer/float)");
    }
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    }
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyF16F32VecUnary(Operation *op, Type srcTy,
                                          Type dstTy) {
  if (failed(verifyVecTileUnaryOp(op, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false,
                                  /*allowInt8=*/false)) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elem = getElemTy(srcTy);
  if (!elem.isF16() && !elem.isF32())
    return op->emitOpError("expects element type to be f16 or f32");
  return success();
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() {
    return verifyF16F32VecUnary(getOperation(), getSrc().getType(),
                                getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTExpandsArch(TExpandsOp op, bool isA5) {
    Type dstTy = op.getDst().getType();
    if (failed(verifyTileBufCommon(op, dstTy, "dst"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return op.emitOpError("expects dst to be in the vec or mat address space");
    }
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = op.getScalar().getType();
    if (scalarTy != dstElem) {
      return op.emitOpError("expects scalar type == dst element type");
    }
    if (!isA5 && *dstSpace == pto::AddressSpace::VEC &&
        !isRowMajorTileBuf(dstTy)) {
      return op.emitOpError("expects vec dst to use row-major layout on A2/A3");
    }
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()) {
      return mlir::success();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if ((isA5 && w == mlir::pto::kValue8) || w == mlir::pto::kValue16 || w == mlir::pto::kValue32) {
          return mlir::success();
      }
    }
    return op.emitOpError(isA5
        ? "expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32"
        : "expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTExpandsArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTExpandsArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isA2A3AccCastExtractTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
      return dstElem.isInteger(mlir::pto::kValue8);
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isInteger(mlir::pto::kValue16);
  }
  return false;
}

static bool isA2A3AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  return isA2A3AccQuantTypePair(srcElem, dstElem);
}

static bool isA5AccCastExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue32);
  }
  return false;
}

static bool isA5AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32() ||
             (llvm::isa<FloatType>(dstElem) && llvm::cast<FloatType>(dstElem).getWidth() == mlir::pto::kValue8);
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static bool hasMatExtractSourceLayoutA2A3(pto::TileBufType srcTy) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool hasMatExtractSourceLayoutA5(pto::TileBufType srcTy,
                                        pto::AddressSpace dstSpace) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  if (dstSpace == pto::AddressSpace::LEFT) {
    return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor)) ||
           bl == static_cast<int32_t>(pto::BLayout::RowMajor);
  }
  return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool isA2A3ExtractElemType(Type ty) {
    return ty.isInteger(mlir::pto::kValue8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5ExtractElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
      return it.getWidth() == mlir::pto::kValue8;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
      return ft.getWidth() == mlir::pto::kValue8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isRowMajorNoneBoxND(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
}

struct TileTransferCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TileTransferCommon> verifyTileTransferCommon(
    Operation *op, Value src, Value dst, Value indexRow, Value indexCol,
    bool allowLowPrecision, bool includeIndexAndIntOpsInConstFold,
    bool insert) {
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb)
    return op->emitOpError("expects src and dst to be !pto.tile_buf");
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
      failed(verifyNonNegativeIndexRowCol(
          *op, indexRow, indexCol, includeIndexAndIntOpsInConstFold)))
    return failure();
  LogicalResult bounds =
      insert ? verifyInsertStaticBoundsCommon(
                   *op, indexRow, indexCol, srcTy, dstTy,
                   includeIndexAndIntOpsInConstFold)
             : verifyExtractStaticBoundsCommon(
                   *op, indexRow, indexCol, srcTy, dstTy,
                   includeIndexAndIntOpsInConstFold);
  if (failed(bounds))
    return failure();
  return TileTransferCommon{srcTy,
                            dstTy,
                            srcTb,
                            dstTb,
                            getElemTy(srcTy),
                            getElemTy(dstTy),
                            getPTOMemorySpaceEnum(srcTy),
                            getPTOMemorySpaceEnum(dstTy)};
}

using TExtractCommon = TileTransferCommon;

static FailureOr<TExtractCommon> verifyTExtractCommon(TExtractOp op,
                                                      bool allowLowPrecision) {
  const bool hasFp = static_cast<bool>(op.getFp());
  auto common = verifyTileTransferCommon(
      op, op.getSrc(), op.getDst(), op.getIndexRow(), op.getIndexCol(),
      allowLowPrecision, /*includeIndexAndIntOpsInConstFold=*/hasFp,
      /*insert=*/false);
  if (failed(common))
    return failure();
  if (hasFp) {
    Type fpTy = op.getFp().getType();
    if (failed(verifyTileBufCommon(op, fpTy, "fp", allowLowPrecision))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to use loc=scaling");
    }
  }
  if (!common->srcElem || !common->dstElem) {
    return op.emitOpError("expects src and dst to have element types");
  }
  if ((!common->srcSpace || *common->srcSpace != pto::AddressSpace::ACC) &&
      common->srcElem != common->dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  return *common;
}

static LogicalResult
verifyTExtractFpFormLoc(TExtractOp op,
                        std::optional<pto::AddressSpace> srcSpace) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("expects fp and preQuantScalar to be mutually exclusive");
  }
  if (hasFp && !srcIsAcc) {
    return op.emitOpError("expects fp form to use loc=acc src");
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && !srcIsAcc) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Acc(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use loc=mat");
  }
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != mlir::pto::kValue512) {
      return op.emitOpError("expects A2/A3 acc-source textract dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 acc textract element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Mat(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.srcSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 textract src to use loc=mat, loc=acc, or loc=vec");
  }
  if (*c.dstSpace != pto::AddressSpace::LEFT &&
      *c.dstSpace != pto::AddressSpace::RIGHT) {
    return op.emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, loc=mat, or loc=vec");
  }
  if (!hasMatExtractSourceLayoutA2A3(c.srcTb)) {
    return op.emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    }
  } else {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA2A3(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA2A3ExtractElemType(c.dstElem) && !(hasFp && c.dstElem.isInteger(mlir::pto::kValue16))) {
      return op.emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr()) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec->vec textract to use the base form without preQuantScalar or reluPreMode");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA2A3Acc(op, c);
  }
  return verifyTExtractA2A3Mat(op, c);
}

static bool isTExtractA5SupportedPair(pto::AddressSpace srcSpace,
                                      pto::AddressSpace dstSpace) {
  return (srcSpace == pto::AddressSpace::MAT &&
          (dstSpace == pto::AddressSpace::LEFT ||
           dstSpace == pto::AddressSpace::RIGHT ||
           dstSpace == pto::AddressSpace::SCALING)) ||
         (srcSpace == pto::AddressSpace::VEC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC)) ||
         (srcSpace == pto::AddressSpace::ACC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC));
}

static LogicalResult verifyTExtractA5Mat(TExtractOp op,
                                         const TExtractCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError("expects mat-source textract to use the base form without preQuantScalar or reluPreMode");
  }
  if (!hasMatExtractSourceLayoutA5(c.srcTb, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
    }
  } else if (*c.dstSpace == pto::AddressSpace::RIGHT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA5Acc(TExtractOp op,
                                         const TExtractCommon &c) {
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A5 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 acc-source textract dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    if (!isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError("expects A5 acc->vec textract dst to use ND layout (blayout=row_major, slayout=none_box)");
    }
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA5AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A5 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)");
    }
  } else if (!isA5AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A5 acc textract element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
  }
  return success();
}

static LogicalResult verifyTExtractA5(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA5ExtractElemType(c.dstElem)) {
    return op.emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr() &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (!isTExtractA5SupportedPair(*c.srcSpace, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract to use a supported src/dst loc pair");
  }
  if (*c.srcSpace == pto::AddressSpace::MAT) {
    return verifyTExtractA5Mat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec-source textract to use the base form without preQuantScalar or reluPreMode");
    }
    if (!isRowMajorNoneBoxND(c.srcTb) || !isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError(
          "expects A5 vec->vec textract src/dst to use ND layout "
          "(blayout=row_major, slayout=none_box)");
    }
    return success();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA5Acc(op, c);
  }
  return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTExtractA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTExtractA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
static bool isA2A3AccCastInsertTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantInsertTypePair(Type srcElem, Type dstElem) {
  return isA2A3AccQuantTypePair(srcElem, dstElem);
}

static bool isColMajorRowMajorNZ(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5SupportedVecElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8 || it.getWidth() == 32;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
      return ft.getWidth() == mlir::pto::kValue8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isA2A3VecInsertElemType(Type ty) {
    return ty.isInteger(mlir::pto::kValue8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

using TInsertCommon = TileTransferCommon;

static FailureOr<TInsertCommon> verifyTInsertCommon(TInsertOp op,
                                                    bool allowLowPrecision) {
  return verifyTileTransferCommon(
      op, op.getSrc(), op.getDst(), op.getIndexRow(), op.getIndexCol(),
      allowLowPrecision, /*includeIndexAndIntOpsInConstFold=*/true,
      /*insert=*/true);
}

static LogicalResult
verifyTInsertOptionalFp(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                        bool isA5) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("fp and preQuantScalar are mutually exclusive");
  }
  if (hasFp) {
    if (!srcIsAcc) {
      return op.emitOpError("fp is only valid with src loc=acc");
    }
    auto fpTy = op.getFp().getType();
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    if (!fpTb) {
      return op.emitOpError("expects fp to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(op, fpTy, "fp", /*allowLowPrecision=*/isA5))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to be loc=scaling");
    }
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("preQuantScalar is only valid with src loc=acc");
  }
  if (reluNonDefault && !srcIsAcc) {
    return op.emitOpError("reluPreMode is only valid with src loc=acc");
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalAttrs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                           std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasAccToVecMode) {
    if (!isA5) {
      return op.emitOpError("accToVecMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
        *dstSpace != pto::AddressSpace::VEC)
      return op.emitOpError("accToVecMode is only valid with src=acc, dst=vec");
  }
  if (hasInsertMode) {
    if (!isA5) {
      return op.emitOpError("tinsertMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::MAT) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) is only valid with src=vec, dst=mat");
    }
    auto srcTb = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!srcTb || !isColMajorRowMajorNZ(srcTb)) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) requires src NZ layout "
          "(blayout=col_major, slayout=row_major)");
    }
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalArgs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                          std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  if (failed(verifyTInsertOptionalFp(op, srcSpace, isA5))) {
    return failure();
  }
  return verifyTInsertOptionalAttrs(op, srcSpace, dstSpace, isA5);
}

static LogicalResult verifyTInsertA2A3AccMat(TInsertOp op,
                                             const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != mlir::pto::kValue512) {
      return op.emitOpError("expects A2/A3 tinsert dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantInsertTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc fp/preQuantScalar tinsert element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastInsertTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTInsertA2A3(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  if (failed(verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/false))) {
    return failure();
  }
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError(
          "expects vec->vec tinsert to use the base form without "
          "preQuantScalar or reluPreMode");
    }
    if (c.srcElem != c.dstElem || !isA2A3VecInsertElemType(c.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
          "(i8/f16/bf16/f32)");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
      *c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");
  }
  return verifyTInsertA2A3AccMat(op, c);
}

static LogicalResult verifyTInsertA5Acc(TInsertOp op, const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (!isColMajorRowMajorNZ(c.dstTb)) {
      return op.emitOpError("expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
    bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
    if (!dstIsND && !dstIsNZ) {
      return op.emitOpError(
          "expects A5 acc->vec tinsert dst to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
    }
  }
  const bool hasQuant =
      static_cast<bool>(op.getFp()) || static_cast<bool>(op.getPreQuantScalar());
  bool okTypes;
  if (hasQuant) {
    okTypes = isA5VectorPreQuantTypePair(c.srcElem, c.dstElem);
  } else {
      okTypes = (c.srcElem.isF32() && (c.dstElem.isF16() || c.dstElem.isBF16() || c.dstElem.isF32())) ||
                (c.srcElem.isInteger(mlir::pto::kValue32) && c.dstElem.isInteger(mlir::pto::kValue32));
  }
  if (!okTypes) {
    return op.emitOpError(
        "expects A5 acc-source tinsert element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)" +
        (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
  }
  return success();
}

static LogicalResult verifyTInsertA5VecMat(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->mat tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  if (!srcIsND && !srcIsNZ) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
  }
  if (hasTInsertMode && !srcIsNZ) {
    return op.emitOpError("expects tinsertMode vec->mat tinsert src to use NZ(col_major/row_major) layout");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecVec(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->vec tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
  if (srcIsND && dstIsND) {
    // ND->ND path
  } else if (srcIsNZ && dstIsNZ) {
    // NZ->NZ path
  } else {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst layouts to match: "
        "both ND(row_major/none_box) or both NZ(col_major/row_major)");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5Modes(TInsertOp op,
                                          const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  if (hasAccToVecMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec tinsert forms");
  }
  if (hasTInsertMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::VEC ||
       *c.dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects tinsertMode only on A5 vec->mat tinsert forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects A5 tinsert src/dst to have explicit loc");
  }
  return verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/true);
}

static LogicalResult verifyTInsertA5(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  if (failed(verifyTInsertA5Modes(op, c))) {
    return failure();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC &&
      (*c.dstSpace == pto::AddressSpace::MAT || *c.dstSpace == pto::AddressSpace::VEC)) {
    return verifyTInsertA5Acc(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::MAT) {
    return verifyTInsertA5VecMat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::VEC) {
    return verifyTInsertA5VecVec(op, c);
  }
  return op.emitOpError(
      "expects A5 tinsert to use a supported src/dst loc pair: "
      "acc->mat, acc->vec, vec->mat, or vec->vec");
}

mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTInsertA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTInsertA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty)) {
      return ft.getWidth() == mlir::pto::kValue8;
  }
  return false;
}

static bool isA5MxFp8InputType(Type ty) {
  return ty && isa<Float8E4M3FNType, Float8E5M2Type>(ty);
}

static bool isA5MxInputTypePair(Type lhsTy, Type rhsTy) {
  return (isA5MxFp8InputType(lhsTy) && isA5MxFp8InputType(rhsTy)) ||
         (isPTOFloat4PackedType(lhsTy) && isPTOFloat4PackedType(rhsTy));
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputTypePair(lhsElem, rhsElem)) {
    return op->emitOpError()
           << "expects A5 mx " << lhsName << "/" << rhsName
           << " element types to be a supported fp8/fp8 or fp4/fp4 pair";
  }

  if (!dstElem.isF32()) {
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";
  }

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
      return dstElem.isInteger(mlir::pto::kValue8) || isA5Fp8LikeType(dstElem) || isPTOHiFloat8Type(dstElem) ||
             dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}
