// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
