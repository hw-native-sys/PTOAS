// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, t0, "src0")) ||
        failed(verifyVecTileStorage(*this, t1, "src1")) ||
        failed(verifyVecTileStorage(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for src0/src1/dst");
    if (e0 != e1)
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!(e0.isInteger(32) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tcmp input element type to be i32/f16/f32");
    if (!ed.isInteger(8))
      return emitOpError("expects dst element type to be i8");

    auto valid0 = getValidShapeVec(t0);
    auto valid1 = getValidShapeVec(t1);
    auto validd = getValidShapeVec(td);
    if (valid0.size() != 2 || valid1.size() != 2 || validd.size() != 2)
      return emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    if (!hasCompatibleKnownExtent(valid0[0], valid1[0]))
      return emitOpError("expects src0 and src1 to have the same valid row");
    if (!hasCompatibleKnownExtent(valid0[1], valid1[1]))
      return emitOpError("expects src0 and src1 to have the same valid column");
    if (!hasCompatibleKnownExtent(valid0[0], validd[0]))
      return emitOpError("expects src0 valid row to equal dst valid row");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for src0/src1/dst");
    if (e0 != e1)
      return emitOpError("expects src0 and src1 to have the same element type");
    bool inputOk = e0.isF16() || e0.isF32() || e0.isBF16() ||
                   e0.isInteger(8) || e0.isInteger(16) || e0.isInteger(32);
    if (!inputOk)
      return emitOpError("expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32");
    if (auto it = dyn_cast<IntegerType>(ed)) {
      if (it.getWidth() != 8)
        return emitOpError("expects dst element type to be i8");
    } else {
      return emitOpError("expects dst element type to be i8");
    }

    if (getShapeVec(t0) != getShapeVec(t1) || getShapeVec(t0) != getShapeVec(td))
      return emitOpError("expects src0, src1, and dst to have the same shape");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32()))
      return emitOpError("expects A2/A3 tcmps input element type to be i16/i32/f16/f32");

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat()))
      return emitOpError("expects scalar to be integer, index, or float");

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32()))
      return emitOpError("expects A5 tcmps input element type to be i8/i16/i32/f16/f32");

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat()))
      return emitOpError("expects scalar to be integer, index, or float");

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true))
    return emitOpError("expects tcolexpand element type to be supported");
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for src0/src1/dst");

  auto isSupportedElem = [&](Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32())
      return true;
    if (!allowIntegerTypes)
      return false;
    if (elemTy.isInteger(16) || elemTy.isInteger(32))
      return true;
    return targetArch == PTOArch::A5 && elemTy.isInteger(8);
  };
  if (!isSupportedElem(e0) || !isSupportedElem(e1) || !isSupportedElem(ed)) {
    if (!allowIntegerTypes)
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    if (targetArch == PTOArch::A5)
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (getShapeVec(t0) != getShapeVec(td))
    return op->emitOpError("expects src0/dst to have same shape");
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")))
    return failure();

  if (auto src0TileTy = dyn_cast<TileBufType>(t0)) {
    if (src0TileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects src0 to use row-major layout");
  }

  if (auto src1TileTy = dyn_cast<TileBufType>(t1)) {
    if (src1TileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects src1 to use row-major layout");
  }
  if (auto dstTileTy = dyn_cast<TileBufType>(td)) {
    if (dstTileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects dst to use row-major layout");
  }

  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == 2 && dstValid.size() == 2 &&
      src1Valid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      src1Valid[1] != dstValid[1])
    return op->emitOpError("expects src1 valid_shape[1] to equal dst valid_shape[1]");

  return success();
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

LogicalResult pto::TColArgMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
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
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}



ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  // Parse: ins(%src : type) or ins(%src, %tmp {isBinary = ...}: type, type)
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();

  // Check for optional tmp operand (format 2)
  if (succeeded(parser.parseOptionalComma())) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type)
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;

    // Parse attributes (isBinary)
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

    // Parse types: : type, type
    if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  } else {
    // Format 1: ins(%src : type)
    if (parser.parseColonType(srcTy))
      return failure();
  }

  if (parser.parseRParen())
    return failure();

  // Parse: outs(%dst : type)
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  // Parse any remaining attributes (for format 1)
  if (!hasTmp) {
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
  }

  // Resolve operands
  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();

  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }

  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, 1> elidedAttrs;
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
    SmallVector<StringRef, 1> elidedAttrs = {"isBinary"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp != hasIsBinary) {
      if (hasTmp)
        return emitOpError("tmp operand requires isBinary attribute");
      return emitOpError("isBinary attribute requires tmp operand");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp")))
        return failure();
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy))
        return emitOpError("expects src/tmp/dst element types to match");
      if (failed(verifyTColSumTmpStride(*this, srcTy, tmpTy, getIsBinary())))
        return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src/dst element types to match");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false)))
      return failure();
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp != hasIsBinary) {
      if (hasTmp)
        return emitOpError("tmp operand requires isBinary attribute");
      return emitOpError("isBinary attribute requires tmp operand");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp")))
        return failure();
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy))
        return emitOpError("expects src/tmp/dst element types to match");
      if (failed(verifyTColSumTmpStride(*this, srcTy, tmpTy, getIsBinary())))
        return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src/dst element types to match");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true)))
      return failure();
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() || elem.isInteger(8) ||
          elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A5 tcolsum element type to be i8/i16/i32/f16/bf16/f32");
    return success();
  };
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

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true)))
    return failure();
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false)))
    return failure();
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem))
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (shouldBypassDecodedMemrefVerifier(getOperation()))
      return success();

    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32))
      return emitOpError("expects dst element type to be i32 or ui32");

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32)
        return emitOpError() << "expects " << name << " to be i32/ui32";
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3")))
      return failure();

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10)
      return emitOpError("expects rounds to be 7 or 10");

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32()))
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32)))
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, MemRefType, RankedTensorType,
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

    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile))
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    if (!mlir::isa<IntegerType, FloatType>(scalarTy))
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false)))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32())
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects dst to be in the vec or mat address space");
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem)
      return emitOpError("expects scalar type == dst element type");
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy))
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())
      return mlir::success();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 16 || w == 32)
        return mlir::success();
    }
    return emitOpError("expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects dst to be in the vec or mat address space");
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem)
      return emitOpError("expects scalar type == dst element type");
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())
      return mlir::success();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 8 || w == 16 || w == 32)
        return mlir::success();
    }
    return emitOpError("expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto isA2A3AccCastExtractTypePair = [&](Type srcElem, Type dstElem) -> bool {
    return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
  };
  auto isA2A3AccQuantExtractTypePair = [&](Type srcElem, Type dstElem) -> bool {
    if (srcElem.isF32())
      return dstElem.isInteger(8);
    if (srcElem.isInteger(32))
      return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
    return false;
  };
  auto isA5AccCastExtractTypePair = [&](Type srcElem, Type dstElem) -> bool {
    if (srcElem.isF32())
      return dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
    if (srcElem.isInteger(32))
      return dstElem.isInteger(32);
    return false;
  };
  auto isA5AccQuantExtractTypePair = [&](Type srcElem, Type dstElem) -> bool {
    if (srcElem.isF32())
      return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
             dstElem.isF32() ||
             (llvm::isa<FloatType>(dstElem) &&
              llvm::cast<FloatType>(dstElem).getWidth() == 8);
    if (srcElem.isInteger(32))
      return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
    return false;
  };
  auto hasMatExtractSourceLayoutA2A3 = [&](pto::TileBufType srcTy) -> bool {
    int32_t bl = srcTy.getBLayoutValueI32();
    int32_t sl = srcTy.getSLayoutValueI32();
    return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor));
  };
  auto hasMatExtractSourceLayoutA5 = [&](pto::TileBufType srcTy,
                                         pto::AddressSpace dstSpace) -> bool {
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
  };
  auto isA2A3ExtractElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isA5ExtractElemType = [&](Type ty) -> bool {
    if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) ||
        isPTOFloat4PackedType(ty))
      return true;
    if (auto it = dyn_cast<IntegerType>(ty))
      return it.getWidth() == 8;
    if (auto ft = dyn_cast<FloatType>(ty))
      return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
    return false;
  };
  auto isRowMajorNoneBoxND = [&](pto::TileBufType ty) -> bool {
    return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
  };
  Value preQuantScalar = getPreQuantScalar();
  auto reluMode = getReluPreMode();
  auto accToVecModeAttr = getAccToVecModeAttr();
  const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
  const bool hasRelu = reluMode != pto::ReluPreMode::NoRelu;
  const bool hasAccToVecMode = static_cast<bool>(accToVecModeAttr);
  auto verifyCommon = [&](bool allowLowPrecision)
      -> FailureOr<std::tuple<Type, Type, pto::TileBufType,
                                                    pto::TileBufType, Type, Type,
                                                    std::optional<pto::AddressSpace>,
                                                    std::optional<pto::AddressSpace>>> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !dstTb)
      return emitOpError("expects src and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src", allowLowPrecision)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", allowLowPrecision)) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/false)) ||
        failed(verifyExtractStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/false)))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem)
      return emitOpError("expects src and dst to have element types");
    if ((!srcSpace || *srcSpace != pto::AddressSpace::ACC) && srcElem != dstElem)
      return emitOpError("expects src and dst to have the same element type");
    return std::make_tuple(srcTy, dstTy, srcTb, dstTb, srcElem, dstElem,
                           srcSpace, dstSpace);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    if (!isA2A3ExtractElemType(dstElem))
      return emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
    if (hasPreQuantScalar && (!srcSpace || *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (hasRelu && (!srcSpace || *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects reluPreMode form to use loc=acc src");
    if (hasAccToVecMode)
      return emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
    if (srcSpace && dstSpace && *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::VEC) {
      if (hasPreQuantScalar || hasRelu)
        return emitOpError("expects vec->vec textract to use the base form without preQuantScalar or reluPreMode");
      return mlir::success();
    }
    if (!srcSpace || !dstSpace)
      return emitOpError("expects src and dst to have explicit loc");
    if (*srcSpace == pto::AddressSpace::ACC) {
      if (*dstSpace != pto::AddressSpace::MAT)
        return emitOpError("expects A2/A3 acc-source textract dst to use loc=mat");
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError("expects A2/A3 acc-source textract src to use blayout=col_major and slayout=row_major");
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError("expects A2/A3 acc-source textract dst to use blayout=col_major and slayout=row_major");
      if (dstTb.getSFractalSizeI32() != 512)
        return emitOpError("expects A2/A3 acc-source textract dst fractal size to be 512");
      if (hasPreQuantScalar) {
        if (!isA2A3AccQuantExtractTypePair(srcElem, dstElem))
          return emitOpError(
              "expects A2/A3 acc preQuantScalar textract element types to be "
              "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
      } else if (!isA2A3AccCastExtractTypePair(srcElem, dstElem)) {
        return emitOpError(
            "expects A2/A3 acc textract element types to be src=f32, dst=f16/bf16");
      }
      return mlir::success();
    }
    if (*srcSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 textract src to use loc=mat, loc=acc, or loc=vec");
    if (*dstSpace != pto::AddressSpace::LEFT &&
        *dstSpace != pto::AddressSpace::RIGHT)
      return emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, loc=mat, or loc=vec");
    if (!hasMatExtractSourceLayoutA2A3(srcTb))
      return emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
    if (*dstSpace == pto::AddressSpace::LEFT) {
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    } else {
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
        return emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    if (!isA5ExtractElemType(dstElem))
      return emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
    if (hasPreQuantScalar && (!srcSpace || *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (hasRelu && (!srcSpace || *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects reluPreMode form to use loc=acc src");
    if (hasAccToVecMode &&
        (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
         *dstSpace != pto::AddressSpace::VEC))
      return emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
    if (!srcSpace || !dstSpace)
      return emitOpError("expects src and dst to have explicit loc");
    bool okPair =
        (*srcSpace == pto::AddressSpace::MAT &&
         (*dstSpace == pto::AddressSpace::LEFT ||
          *dstSpace == pto::AddressSpace::RIGHT ||
          *dstSpace == pto::AddressSpace::SCALING)) ||
        (*srcSpace == pto::AddressSpace::VEC &&
         (*dstSpace == pto::AddressSpace::MAT ||
          *dstSpace == pto::AddressSpace::VEC)) ||
        (*srcSpace == pto::AddressSpace::ACC &&
         (*dstSpace == pto::AddressSpace::MAT ||
          *dstSpace == pto::AddressSpace::VEC));
    if (!okPair)
      return emitOpError("expects A5 textract to use a supported src/dst loc pair");
    if (*srcSpace == pto::AddressSpace::MAT) {
      if (hasPreQuantScalar || hasRelu)
        return emitOpError("expects mat-source textract to use the base form without preQuantScalar or reluPreMode");
      if (!hasMatExtractSourceLayoutA5(srcTb, *dstSpace))
        return emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
      if (*dstSpace == pto::AddressSpace::LEFT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
            dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
          return emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
      } else if (*dstSpace == pto::AddressSpace::RIGHT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
          return emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
      }
    } else if (*srcSpace == pto::AddressSpace::VEC &&
               *dstSpace == pto::AddressSpace::VEC) {
      if (hasPreQuantScalar || hasRelu)
        return emitOpError("expects vec-source textract to use the base form without preQuantScalar or reluPreMode");
      if (!isRowMajorNoneBoxND(srcTb) || !isRowMajorNoneBoxND(dstTb))
        return emitOpError(
            "expects A5 vec->vec textract src/dst to use ND layout "
            "(blayout=row_major, slayout=none_box)");
    } else if (*srcSpace == pto::AddressSpace::ACC) {
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError("expects A5 acc-source textract src to use blayout=col_major and slayout=row_major");
      if (*dstSpace == pto::AddressSpace::MAT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
            dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
          return emitOpError("expects A5 acc-source textract dst to use blayout=col_major and slayout=row_major");
      } else {
        if (!isRowMajorNoneBoxND(dstTb))
          return emitOpError("expects A5 acc->vec textract dst to use ND layout (blayout=row_major, slayout=none_box)");
      }
      if (hasPreQuantScalar) {
        if (!isA5AccQuantExtractTypePair(srcElem, dstElem))
          return emitOpError(
              "expects A5 acc preQuantScalar textract element types to be "
              "(src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)");
      } else if (!isA5AccCastExtractTypePair(srcElem, dstElem)) {
        return emitOpError(
            "expects A5 acc textract element types to be "
            "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  auto isA2A3AccCastInsertTypePair = [&](Type srcElem, Type dstElem) -> bool {
    return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
  };
  auto isA2A3AccQuantInsertTypePair = [&](Type srcElem, Type dstElem) -> bool {
    if (srcElem.isF32())
      return dstElem.isInteger(8);
    if (srcElem.isInteger(32))
      return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
    return false;
  };
  auto isColMajorRowMajorNZ = [&](pto::TileBufType ty) -> bool {
    return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
  };
  auto isRowMajorNoneBoxND = [&](pto::TileBufType ty) -> bool {
    return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
  };
  auto isA5SupportedVecElemType = [&](Type ty) -> bool {
    if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) ||
        isPTOFloat4PackedType(ty))
      return true;
    if (auto it = dyn_cast<IntegerType>(ty))
      return it.getWidth() == 8 || it.getWidth() == 32;
    if (auto ft = dyn_cast<FloatType>(ty))
      return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
    return false;
  };
  auto isA2A3VecInsertElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto getSpace = [](Type ty) -> std::optional<pto::AddressSpace> {
    return getPTOMemorySpaceEnum(ty);
  };
  const bool hasFp = static_cast<bool>(getFp());
  const bool hasPreQuantScalar = static_cast<bool>(getPreQuantScalar());
  const bool hasRelu = getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasAccToVecMode = static_cast<bool>(getAccToVecModeAttr());
  const bool hasTInsertMode = static_cast<bool>(getTinsertModeAttr());
  auto verifyCommon = [&](bool allowLowPrecision)
      -> FailureOr<std::tuple<Type, Type, pto::TileBufType,
                                                    pto::TileBufType, Type, Type,
                                                    std::optional<pto::AddressSpace>,
                                                    std::optional<pto::AddressSpace>>> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !dstTb)
      return emitOpError("expects src and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src", allowLowPrecision)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", allowLowPrecision)) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/true)) ||
        failed(verifyInsertStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/true)))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    return std::make_tuple(srcTy, dstTy, srcTb, dstTb, srcElem, dstElem,
                           srcSpace, dstSpace);
  };
  // Shared validation for optional operands and attributes.
  auto verifyOptionalArgs = [&](const std::optional<pto::AddressSpace> &srcSpace,
                                const std::optional<pto::AddressSpace> &dstSpace,
                                bool isA5) -> LogicalResult {
    const bool hasFp = static_cast<bool>(getFp());
    const bool hasPreQuantScalar = static_cast<bool>(getPreQuantScalar());
    const bool hasAccToVecMode = static_cast<bool>(getAccToVecModeAttr());
    const bool hasInsertMode = static_cast<bool>(getTinsertModeAttr());
    const bool reluNonDefault = getReluPreMode() != pto::ReluPreMode::NoRelu;

    if (hasFp && hasPreQuantScalar)
      return emitOpError("fp and preQuantScalar are mutually exclusive");

    // fp tile is only valid with Acc source.
    if (hasFp) {
      if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
        return emitOpError("fp is only valid with src loc=acc");
      auto fpTy = getFp().getType();
      auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
      if (!fpTb) return emitOpError("expects fp to be !pto.tile_buf");
      auto fpSpace = getSpace(fpTy);
      if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
        return emitOpError("expects fp to be loc=scaling");
    }

    // preQuantScalar is only valid with Acc source.
    if (hasPreQuantScalar) {
      if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
        return emitOpError("preQuantScalar is only valid with src loc=acc");
    }

    // reluPreMode is only valid with Acc source.
    if (reluNonDefault) {
      if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
        return emitOpError("reluPreMode is only valid with src loc=acc");
    }

    // accToVecMode is only valid with Acc->Vec (A5 only).
    if (hasAccToVecMode) {
      if (!isA5)
        return emitOpError("accToVecMode is only supported on A5");
      if (!srcSpace || !dstSpace ||
          *srcSpace != pto::AddressSpace::ACC ||
          *dstSpace != pto::AddressSpace::VEC)
        return emitOpError("accToVecMode is only valid with src=acc, dst=vec");
    }

    // tinsertMode (SPLIT2/SPLIT4) is only valid with Vec(NZ)->Mat on A5.
    if (hasInsertMode) {
      if (!isA5)
        return emitOpError("tinsertMode is only supported on A5");
      if (!srcSpace || !dstSpace ||
          *srcSpace != pto::AddressSpace::VEC ||
          *dstSpace != pto::AddressSpace::MAT)
        return emitOpError(
            "tinsertMode (SPLIT2/SPLIT4) is only valid with src=vec, dst=mat");
      auto srcTb = dyn_cast<pto::TileBufType>(getSrc().getType());
      if (!srcTb || !isColMajorRowMajorNZ(srcTb))
        return emitOpError(
            "tinsertMode (SPLIT2/SPLIT4) requires src NZ layout "
            "(blayout=col_major, slayout=row_major)");
    }

    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    if (failed(verifyOptionalArgs(srcSpace, dstSpace, /*isA5=*/false)))
      return failure();
    if (srcSpace && dstSpace && *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::VEC) {
      if (hasPreQuantScalar || hasRelu)
        return emitOpError(
            "expects vec->vec tinsert to use the base form without "
            "preQuantScalar or reluPreMode");
      if (srcElem != dstElem || !isA2A3VecInsertElemType(srcElem))
        return emitOpError(
            "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
            "(i8/f16/bf16/f32)");
      return success();
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
        *dstSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");

    if (!isColMajorRowMajorNZ(srcTb))
      return emitOpError("expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
    if (!isColMajorRowMajorNZ(dstTb))
      return emitOpError("expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
    if (dstTb.getSFractalSizeI32() != 512)
      return emitOpError("expects A2/A3 tinsert dst fractal size to be 512");

    if (hasPreQuantScalar) {
      if (!isA2A3AccQuantInsertTypePair(srcElem, dstElem))
        return emitOpError(
            "expects A2/A3 acc preQuantScalar tinsert element types to be "
            "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    } else if (!isA2A3AccCastInsertTypePair(srcElem, dstElem)) {
      return emitOpError(
          "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    if (hasPreQuantScalar && (!srcSpace || *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (hasRelu && (!srcSpace || *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects reluPreMode form to use loc=acc src");
    if (hasAccToVecMode &&
        (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
         *dstSpace != pto::AddressSpace::VEC))
      return emitOpError("expects accToVecMode only on A5 acc->vec tinsert forms");
    if (hasTInsertMode &&
        (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
         *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects tinsertMode only on A5 vec->mat tinsert forms");
    if (!srcSpace || !dstSpace)
      return emitOpError("expects A5 tinsert src/dst to have explicit loc");
    if (failed(verifyOptionalArgs(srcSpace, dstSpace, /*isA5=*/true)))
      return failure();

    // A5 regular acc->mat/acc->vec path.
    if (*srcSpace == pto::AddressSpace::ACC &&
        (*dstSpace == pto::AddressSpace::MAT || *dstSpace == pto::AddressSpace::VEC)) {
      if (!isColMajorRowMajorNZ(srcTb))
        return emitOpError("expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
      if (*dstSpace == pto::AddressSpace::MAT) {
        if (!isColMajorRowMajorNZ(dstTb))
          return emitOpError("expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
      } else {
        bool dstIsND = isRowMajorNoneBoxND(dstTb);
        bool dstIsNZ = isColMajorRowMajorNZ(dstTb);
        if (!dstIsND && !dstIsNZ)
          return emitOpError(
              "expects A5 acc->vec tinsert dst to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
      }
      const bool hasQuant = hasFp || hasPreQuantScalar;
      bool okTypes;
      if (hasQuant) {
        // With fp/scalar quantization, allow wider dst types (i8/fp8/f16/bf16/f32).
        okTypes = isA5VectorPreQuantTypePair(srcElem, dstElem);
      } else {
        okTypes = (srcElem.isF32() &&
                   (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())) ||
                  (srcElem.isInteger(32) && dstElem.isInteger(32));
      }
      if (!okTypes)
        return emitOpError(
            "expects A5 acc-source tinsert element types to be "
            "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)"
            + (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
      return success();
    }

    // A5 vec->mat path (ND/NZ modes in pto-isa).
    if (*srcSpace == pto::AddressSpace::VEC && *dstSpace == pto::AddressSpace::MAT) {
      if (hasPreQuantScalar || hasRelu)
        return emitOpError(
            "expects vec->mat tinsert to use the base form without "
            "preQuantScalar or reluPreMode");
      if (!isColMajorRowMajorNZ(dstTb))
        return emitOpError("expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
      bool srcIsND = isRowMajorNoneBoxND(srcTb);
      bool srcIsNZ = isColMajorRowMajorNZ(srcTb);
      if (!srcIsND && !srcIsNZ)
        return emitOpError(
            "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
      if (hasTInsertMode && !srcIsNZ)
        return emitOpError("expects tinsertMode vec->mat tinsert src to use NZ(col_major/row_major) layout");
      if (srcElem != dstElem || !isA5SupportedVecElemType(srcElem))
        return emitOpError(
            "expects A5 vec->mat tinsert src/dst to have same supported dtype "
            "(fp8/f16/bf16/f32/i8/i32)");
      return success();
    }

    // A5 vec->vec path: supports ND->ND and NZ->NZ.
    if (*srcSpace == pto::AddressSpace::VEC && *dstSpace == pto::AddressSpace::VEC) {
      if (hasPreQuantScalar || hasRelu)
        return emitOpError(
            "expects vec->vec tinsert to use the base form without "
            "preQuantScalar or reluPreMode");
      bool srcIsND = isRowMajorNoneBoxND(srcTb);
      bool dstIsND = isRowMajorNoneBoxND(dstTb);
      bool srcIsNZ = isColMajorRowMajorNZ(srcTb);
      bool dstIsNZ = isColMajorRowMajorNZ(dstTb);
      if (srcIsND && dstIsND) {
        // ND->ND path
      } else if (srcIsNZ && dstIsNZ) {
        // NZ->NZ path
      } else {
        return emitOpError(
            "expects A5 vec->vec tinsert src/dst layouts to match: "
            "both ND(row_major/none_box) or both NZ(col_major/row_major)");
      }
      if (srcElem != dstElem || !isA5SupportedVecElemType(srcElem))
        return emitOpError(
            "expects A5 vec->vec tinsert src/dst to have same supported dtype "
            "(fp8/f16/bf16/f32/i8/i32)");
      return success();
    }

    return emitOpError(
        "expects A5 tinsert to use a supported src/dst loc pair: "
        "acc->mat, acc->vec, vec->mat, or vec->vec");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isRowMajorNoneBoxNDTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
}

static bool isA2A3VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(8);
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  return false;
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == 8;
  return false;
}

static bool isA5MxFp8InputType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.isFloat8E4M3FN() || ft.isFloat8E5M2();
  return false;
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

  if (!isA5MxInputTypePair(lhsElem, rhsElem))
    return op->emitOpError()
           << "expects A5 mx " << lhsName << "/" << rhsName
           << " element types to be a supported fp8/fp8 or fp4/fp4 pair";

  if (!dstElem.isF32())
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(8) || isA5Fp8LikeType(dstElem) ||
           isPTOHiFloat8Type(dstElem) || dstElem.isF16() ||
           dstElem.isBF16() || dstElem.isF32();
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  return false;
}

mlir::LogicalResult mlir::pto::TExtractFPOp::verify() {
  auto verifyCommon = [&](bool allowLowPrecision)
      -> FailureOr<std::tuple<Type, Type, Type, pto::TileBufType,
                                                    pto::TileBufType, pto::TileBufType,
                                                    pto::AddressSpace, pto::AddressSpace,
                                                    pto::AddressSpace>> {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !fpTb || !dstTb)
      return emitOpError("expects src, fp, and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src", allowLowPrecision)) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp", allowLowPrecision)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", allowLowPrecision)) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/true)) ||
        failed(verifyExtractStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/true)))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !fpSpace || !dstSpace)
      return emitOpError("expects src, fp, and dst to have explicit loc");
    if (*srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects src to use loc=acc");
    if (*fpSpace != pto::AddressSpace::SCALING)
      return emitOpError("expects fp to use loc=scaling");
    if (*dstSpace != pto::AddressSpace::MAT && *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects dst to use loc=mat or loc=vec");
    if (!isColMajorRowMajorNZTileBuf(srcTb))
      return emitOpError("expects src to use blayout=col_major and slayout=row_major");
    if (*dstSpace == pto::AddressSpace::MAT) {
      if (!isColMajorRowMajorNZTileBuf(dstTb))
        return emitOpError("expects mat dst to use blayout=col_major and slayout=row_major");
    } else {
      if (!(dstTb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            dstTb.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox)))
        return emitOpError("expects vec dst to use ND layout (blayout=row_major, slayout=none_box)");
    }
    return std::make_tuple(srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, *srcSpace,
                           *fpSpace, *dstSpace);
  };
  auto accToVecModeAttr = getAccToVecModeAttr();
  const bool hasAccToVecMode = static_cast<bool>(accToVecModeAttr);
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    if (hasAccToVecMode)
      return emitOpError("expects accToVecMode only on A5 acc->vec textract_fp forms");
    if (dstSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 textract_fp dst to use loc=mat");
    if (dstTb.getSFractalSizeI32() != 512)
      return emitOpError("expects dst fractal size to be 512");
    if (hasAccToVecMode && dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects accToVecMode only on A5 acc->vec textract_fp forms");
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA2A3VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A2/A3 textract_fp element types to be (src=f32,dst=i8) "
          "or (src=i32,dst=i8/f16/i16)");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcTb;
    (void)fpTb;
    (void)dstTb;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA5VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A5 textract_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
          "or (src=i32,dst=i8/f16/bf16)");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TInsertFPOp::verify() {
  auto verifyCommon = [&](bool allowLowPrecision, bool isA5)
      -> FailureOr<std::tuple<Type, Type, Type, pto::TileBufType,
                                                    pto::TileBufType, pto::TileBufType,
                                                    pto::AddressSpace, pto::AddressSpace,
                                                    pto::AddressSpace>> {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !fpTb || !dstTb)
      return emitOpError("expects src, fp, and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src", allowLowPrecision)) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp", allowLowPrecision)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", allowLowPrecision)) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/true)) ||
        failed(verifyInsertStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/true)))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !fpSpace || !dstSpace)
      return emitOpError("expects src, fp, and dst to have explicit loc");
    if (*srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects src to use loc=acc");
    if (*fpSpace != pto::AddressSpace::SCALING)
      return emitOpError("expects fp to use loc=scaling");
    // A2/A3: only acc->mat; A5: acc->mat or acc->vec.
    if (*dstSpace != pto::AddressSpace::MAT &&
        !(isA5 && *dstSpace == pto::AddressSpace::VEC))
      return emitOpError("expects dst to use loc=mat" +
                         (isA5 ? StringRef(" or loc=vec (A5)") : StringRef("")));
    if (!isColMajorRowMajorNZTileBuf(srcTb))
      return emitOpError("expects src to use blayout=col_major and slayout=row_major");
    if (*dstSpace == pto::AddressSpace::MAT && !isColMajorRowMajorNZTileBuf(dstTb))
      return emitOpError("expects dst (mat) to use blayout=col_major and slayout=row_major");
    if (*dstSpace == pto::AddressSpace::VEC &&
        !isRowMajorNoneBoxNDTileBuf(dstTb) && !isColMajorRowMajorNZTileBuf(dstTb))
      return emitOpError("expects dst (vec) to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
    // accToVecMode is only valid when dst=vec.
    if (static_cast<bool>(getAccToVecModeAttr()) &&
        *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("accToVecMode is only valid with dst=vec");
    return std::make_tuple(srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, *srcSpace,
                           *fpSpace, *dstSpace);
  };
  auto accToVecModeAttr = getAccToVecModeAttr();
  const bool hasAccToVecMode = static_cast<bool>(accToVecModeAttr);
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false, /*isA5=*/false);
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcTb;
    (void)fpTb;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    if (hasAccToVecMode)
      return emitOpError("expects accToVecMode only on A5 acc->vec tinsert_fp forms");
    if (dstSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 tinsert_fp dst to use loc=mat");
    if (dstTb.getSFractalSizeI32() != 512)
      return emitOpError("expects dst fractal size to be 512");
    if (hasAccToVecMode && dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects accToVecMode only on A5 acc->vec tinsert_fp forms");
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA2A3VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A2/A3 tinsert_fp element types to be (src=f32,dst=i8) "
          "or (src=i32,dst=i8/f16/i16)");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true, /*isA5=*/true);
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcTb;
    (void)fpTb;
    (void)dstTb;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA5VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A5 tinsert_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
          "or (src=i32,dst=i8/f16/bf16)");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy, Type dstTy,
                                              bool allowDstExpand,
                                              llvm::StringRef opName) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op->emitError("expects rank-2 shaped types for src/dst");

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    unsigned elemBytes = getPTOStorageElemByteSize(t);
    return elemBytes == 0 ? -1 : static_cast<int64_t>(elemBytes);
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0)
    return op->emitError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");

  // pto.tfillpad lowers to TFILLPAD(dst, src). For loc=mat, pto-isa only
  // exposes the homogeneous overload, so src/dst must use the same Tile<...>
  // specialization (including valid_shape and pad).
  // Note: tfillpad_expand is intentionally not covered here because its
  // cross-layer ABI contract for loc=mat heterogeneous shape expansion is not
  // finalized yet.
  if (opName == "tfillpad") {
    auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
    auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (srcTb && dstTb && srcSpace && dstSpace &&
        *srcSpace == mlir::pto::AddressSpace::MAT &&
        *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb) {
      auto dimToStr = [](int64_t dim) -> std::string {
        return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
      };
      SmallVector<std::string, 4> mismatchFields;
      auto srcValid = getValidShapeVec(srcTy);
      auto dstValid = getValidShapeVec(dstTy);
      if (srcValid.size() == 2 && dstValid.size() == 2) {
        if (srcValid[0] != dstValid[0])
          mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                                   dimToStr(dstValid[0]) + ")");
        if (srcValid[1] != dstValid[1])
          mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                                   dimToStr(dstValid[1]) + ")");
      }
      if (srcTb.getPadValueI32() != dstTb.getPadValueI32())
        mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                                 " vs " + std::to_string(dstTb.getPadValueI32()) +
                                 ")");

      auto diag = op->emitError()
                  << "expects src/dst tile types to be lowerable to TFILLPAD "
                     "for loc=mat";
      if (!mismatchFields.empty())
        diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
      diag << "\n  src: " << srcTy;
      diag << "\n  dst: " << dstTy;
      diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
      return failure();
    }
  }

  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError() << "expects dst PadVal != Null for " << opName;
  }

  if (!allowDstExpand) {
    if (srcShape != dstShape)
      return op->emitError()
             << "expects src and dst to have the same static shape for " << opName;
    return mlir::success();
  }

  if (srcShape[0] > dstShape[0] || srcShape[1] > dstShape[1]) {
    return op->emitError()
           << "expects dst static shape to be >= src static shape for " << opName;
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  if (failed(verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                                /*allowDstExpand=*/false, "tfillpad")))
    return failure();

  if (auto padValueAttr = getPadValueAttr()) {
    auto dstSpace = getPTOMemorySpaceEnum(getDst().getType());
    if (!dstSpace || *dstSpace != pto::AddressSpace::MAT)
      return emitOpError("expects padValue attribute only for loc=mat tfillpad");
    if (auto dstTileTy = dyn_cast<pto::TileBufType>(getDst().getType())) {
      if (dstTileTy.getPadValueI32() != static_cast<int32_t>(padValueAttr.getValue()))
        return emitOpError("expects padValue attribute to match dst tile pad configuration");
    } else if (!isa<MemRefType>(getDst().getType())) {
      return emitOpError("expects dst to be tile_buf or memref when padValue is specified");
    }
  }

  return success();
}

mlir::LogicalResult mlir::pto::TFillPadExpandOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/true, "tfillpad_expand");
}

mlir::LogicalResult mlir::pto::TFillPadInplaceOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad_inplace");
}
