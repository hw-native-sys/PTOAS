// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticB.cpp; kept as a fragment included by PTOVerifyArithmeticB.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static ParseResult parseTColSumInsClause(OpAsmParser &parser, OperationState &result,
                                         OpAsmParser::UnresolvedOperand &src,
                                         OpAsmParser::UnresolvedOperand &tmp,
                                         Type &srcTy, Type &tmpTy, bool &hasTmp) {
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  if (failed(parser.parseOptionalComma())) {
    if (parseTColSumFormatWithoutTmp(parser, srcTy))
      return failure();
    return success();
  }

  if (parser.parseOperand(tmp))
    return failure();
  hasTmp = true;
  return parseTColSumFormatWithTmp(parser, result, srcTy, tmpTy);
}

static ParseResult parseTColSumOutsClause(OpAsmParser &parser,
                                          OpAsmParser::UnresolvedOperand &dst,
                                          Type &dstTy) {
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  return success();
}

static LogicalResult verifyTColSumCommon(TColSumOp op, bool requireNonZeroSrc,
                                         bool allowInt8, bool allowBf16,
                                         StringRef errorMessage) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst")))
    return failure();
  bool hasTmp = static_cast<bool>(op.getTmp());
  bool hasIsBinary = static_cast<bool>(op.getIsBinaryAttr());
  if (hasTmp != hasIsBinary) {
    if (hasTmp)
      return op.emitOpError("tmp operand requires isBinary attribute");
    return op.emitOpError("isBinary attribute requires tmp operand");
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyNDStyleVecTile(op, tmpTy, "tmp")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy) ||
        getElemTy(srcTy) != getElemTy(tmpTy))
      return op.emitOpError("expects src/tmp/dst element types to match");
  }
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError("expects src/dst element types to match");
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc)))
    return failure();
  Type elem = getElemTy(srcTy);
  if (!(elem.isF16() || elem.isF32() || (allowBf16 && elem.isBF16()) ||
        elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth) ||
        (allowInt8 && elem.isInteger(kPTOI8BitWidth))))
    return op.emitOpError(errorMessage);
  return success();
}

ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;
  if (parseTColSumInsClause(parser, result, src, tmp, srcTy, tmpTy, hasTmp) ||
      parseTColSumOutsClause(parser, dst, dstTy))
    return failure();
  if (!hasTmp && parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVec1<StringRef> elidedAttrs;
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
    SmallVec1<StringRef> elidedAttrs = {"isBinary"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTColSumCommon(*this, /*requireNonZeroSrc=*/false,
                               /*allowInt8=*/false, /*allowBf16=*/false,
                               "expects A2/A3 tcolsum element type to be "
                               "f16/f32/i16/i32");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTColSumCommon(*this, /*requireNonZeroSrc=*/true,
                               /*allowInt8=*/true, /*allowBf16=*/true,
                               "expects A5 tcolsum element type to be "
                               "i8/i16/i32/f16/bf16/f32");
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
  auto verifyA2A3 = [this, srcElem, dstElem]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    return success();
  };
  auto verifyA5 = [this, srcElem, dstElem]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem))
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    if (shouldBypassDecodedMemrefVerifier(getOperation()))
      return success();

    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(kPTOI32BitWidth))
      return emitOpError("expects dst element type to be i32 or ui32");

    auto checkWord = [this](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != kPTOI32BitWidth)
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
