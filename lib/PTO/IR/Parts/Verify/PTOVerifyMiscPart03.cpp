// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyMisc.cpp; kept as a fragment included by PTOVerifyMisc.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult MScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  if (!isTargetArchA5(getOperation()))
    return emitOpError("pto.mscatter is only supported on A5 targets");

  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();
  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1)
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  if (!srcElem || !idxElem)
    return emitOpError("failed to resolve element types for src or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem))
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src")))
    return failure();

  if (getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
      getScatterOob() != pto::ScatterOOB::Undefined) {
    if (!isTargetArchA5(getOperation()))
      return emitOpError(
          "expects non-default scatterAtomicOp/scatterOob only on A5 targets");
  }

  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, getScatterAtomicOp()))
    return emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src")))
    return failure();

  return success();
}

// ---- MGatherOp ----
LogicalResult MGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  if (!isTargetArchA5(getOperation()))
    return emitOpError("pto.mgather is only supported on A5 targets");

  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();
  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1)
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(*this, dstTy, "dst")) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  if (!dstElem || !idxElem)
    return emitOpError("failed to resolve element types for dst or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), dstElem))
    return emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), dstElem,
                                             "dst")))
    return failure();

  if (getGatherOob() != pto::GatherOOB::Undefined &&
      !isTargetArchA5(getOperation()))
    return emitOpError(
        "expects non-default gatherOob only on A5 targets");

  if (failed(verifyMGatherMScatterTileShape(getOperation(), dstTy, idxTy, "dst")))
    return failure();

  return success();
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs());
  p << " : " << getSrc().getType();
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(srcTy))
    return failure();
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode"))
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  return success();
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    p << ", " << getTmp();
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    p << ", " << getTmp().getType();
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

static ParseResult parseTMrgSortFormat1(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  Type srcTy, blockLenTy, dstTy;
  if (parser.parseType(srcTy) || parser.parseComma() ||
      parser.parseType(blockLenTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand dstOp;
  if (parser.parseOperand(dstOp) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, 1, 0, 0}));
  if (parser.resolveOperand(first, srcTy, result.operands) ||
      parser.resolveOperand(second, blockLenTy, result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
  return success();
}

static ParseResult parseTMrgSortFormat2Sources(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcs,
    OpAsmParser::UnresolvedOperand &tmpOp) {
  while (parser.parseOptionalComma().succeeded()) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next))
      return failure();
    srcs.push_back(next);
  }
  if (srcs.size() < kNumber3 || srcs.size() > kNumber5) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "tmrgsort format2 expects 2 to 4 src operands plus one tmp operand");
  }
  tmpOp = srcs.pop_back_val();
  return success();
}

static ParseResult parseTMrgSortFormat2Exhausted(OpAsmParser &parser,
                                                 bool &exhaustedVal) {
  if (failed(parser.parseOptionalLBrace()))
    return success();
  if (parser.parseKeyword("exhausted") || parser.parseEqual())
    return failure();
  StringRef kw;
  if (parser.parseKeyword(&kw) || parser.parseRBrace())
    return failure();
  exhaustedVal = kw == "true";
  return success();
}

