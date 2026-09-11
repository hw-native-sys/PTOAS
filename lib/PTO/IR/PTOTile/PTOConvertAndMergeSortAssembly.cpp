// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static ParseResult resolveTCvtOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand src, Type srcTy,
    OpAsmParser::UnresolvedOperand tmp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dst, Type dstTy, bool hasTmp) {
  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }
  return parser.resolveOperand(dst, dstTy, result.operands);
}

struct TCvtParseState {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy;
  Type tmpTy;
  Type dstTy;
  bool hasTmp = false;
};

static ParseResult parseTCvtSyntax(OpAsmParser &parser, OperationState &result,
                                  TCvtParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src))
    return failure();
  state.hasTmp = succeeded(parser.parseOptionalComma());
  if (state.hasTmp && parser.parseOperand(state.tmp))
    return failure();
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(state.srcTy))
    return failure();
  if (state.hasTmp && (parser.parseComma() || parser.parseType(state.tmpTy)))
    return failure();
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode"))
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(state.dst) ||
      parser.parseColonType(state.dstTy) || parser.parseRParen())
    return failure();
  return success();
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  TCvtParseState state;
  if (failed(parseTCvtSyntax(parser, result, state)) ||
      failed(resolveTCvtOperands(parser, result, state.src, state.srcTy,
                                 state.tmp, state.tmpTy, state.dst, state.dstTy,
                                 state.hasTmp)))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, state.hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TDeInterleaveOp::print(OpAsmPrinter &p) {
  p << " ins(";
  llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
  p << " : ";
  llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type type) { p << type; });
  p << ") outs(" << getDst0() << ", " << getDst1() << " : "
    << getDst0().getType() << ", " << getDst1().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseTDeInterleaveSources(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcs,
    SmallVectorImpl<Type> &srcTypes) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand src;
  if (parser.parseOperand(src))
    return failure();
  srcs.push_back(src);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(src))
      return failure();
    srcs.push_back(src);
  }
  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();
  srcTypes.push_back(srcType);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(srcType))
      return failure();
    srcTypes.push_back(srcType);
  }
  if (srcs.size() < 1 || srcs.size() > 2)
    return parser.emitError(parser.getCurrentLocation(),
                            "tdeinterleave expects one or two source operands");
  return success();
}

static ParseResult parseTDeInterleaveOutputs(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &dst0,
    OpAsmParser::UnresolvedOperand &dst1, Type &dst0Ty, Type &dst1Ty) {
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst0) || parser.parseComma() ||
      parser.parseOperand(dst1) || parser.parseColonType(dst0Ty) ||
      parser.parseComma() || parser.parseType(dst1Ty) || parser.parseRParen())
    return failure();
  return success();
}

ParseResult mlir::pto::TDeInterleaveOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> srcs;
  SmallVector<Type, 2> srcTypes;
  OpAsmParser::UnresolvedOperand dst0, dst1;
  Type dst0Ty, dst1Ty;
  if (failed(parseTDeInterleaveSources(parser, srcs, srcTypes)) ||
      failed(parseTDeInterleaveOutputs(parser, dst0, dst1, dst0Ty, dst1Ty)))
    return failure();
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dst0, dst0Ty, result.operands) ||
      parser.resolveOperand(dst1, dst1Ty, result.operands))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(srcs.size()), 2}));
  return parser.parseOptionalAttrDict(result.attributes);
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2() || isFormat2WithoutTmp()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    if (getTmp()) {
      p << ", " << getTmp();
    } else {
      p << " no_tmp";
}
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    if (getTmp()) {
      p << ", " << getTmp().getType();
    }
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

struct TMrgSortFormat2State {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs;
  SmallVector<Type, 4> srcTypes;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  OpAsmParser::UnresolvedOperand executed;
  Type tmpTy;
  Type dstTy;
  Type executedTy;
  bool noTmp = false;
  bool exhausted = false;
};

static ParseResult parseTMrgSortFormat1(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  Type srcTy, blockLenTy, dstTy;
  OpAsmParser::UnresolvedOperand dst;
  if (parser.parseType(srcTy) || parser.parseComma() ||
      parser.parseType(blockLenTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0, 0}));
  if (parser.resolveOperand(first, srcTy, result.operands) ||
      parser.resolveOperand(second, blockLenTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
  return success();
}

static ParseResult parseTMrgSortFormat2Inputs(
    OpAsmParser &parser, TMrgSortFormat2State &state,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  state.srcs = {first, second};
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next))
      return failure();
    state.srcs.push_back(next);
  }
  state.noTmp = succeeded(parser.parseOptionalKeyword("no_tmp"));
  size_t min = state.noTmp ? 2 : 3;
  size_t max = state.noTmp ? 4 : 5;
  if (state.srcs.size() < min || state.srcs.size() > max)
    return parser.emitError(
        parser.getCurrentLocation(),
        "tmrgsort format2 expects 2 to 4 src operands and optional no_tmp marker");
  if (!state.noTmp)
    state.tmp = state.srcs.pop_back_val();
  if (succeeded(parser.parseOptionalLBrace())) {
    StringRef keyword;
    if (parser.parseKeyword("exhausted") || parser.parseEqual() ||
        parser.parseKeyword(&keyword) || parser.parseRBrace())
      return failure();
    state.exhausted = keyword == "true";
  }
  return success();
}

static ParseResult parseTMrgSortFormat2TypesAndOutputs(
    OpAsmParser &parser, TMrgSortFormat2State &state) {
  if (parser.parseColon())
    return failure();
  Type type;
  if (parser.parseType(type))
    return failure();
  state.srcTypes.push_back(type);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(type))
      return failure();
    state.srcTypes.push_back(type);
  }
  size_t expectedTypes = state.srcs.size() + (state.noTmp ? 0 : 1);
  if (state.srcTypes.size() != expectedTypes || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  if (!state.noTmp)
    state.tmpTy = state.srcTypes.pop_back_val();
  if (parser.parseOperand(state.dst) || parser.parseComma() ||
      parser.parseOperand(state.executed) || parser.parseColon() ||
      parser.parseType(state.dstTy) || parser.parseComma() ||
      parser.parseType(state.executedTy) || parser.parseRParen())
    return failure();
  return success();
}
