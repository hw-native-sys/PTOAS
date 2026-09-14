// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTRowExpandValidShapes(TRowExpandOp op, Type srcTy,
                                                 Type dstTy) {
  auto srcValid = getValidShapeVec(op.getSrc());
  auto dstValid = getValidShapeVec(op.getDst());
  if (srcValid.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
    return op.emitOpError("expects src valid_shape[0] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
    return op.emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0) {
    return op.emitOpError("expects dst valid_shape[0] to be non-zero");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0) {
    return op.emitOpError("expects dst valid_shape[1] to be non-zero");
  }
  return success();
}

static LogicalResult verifyTRowExpandCommon(TRowExpandOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst")))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects src to be in the vec address space");
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  if (srcTb && srcTb.getSLayoutValueI32() !=
                   static_cast<int32_t>(pto::SLayout::NoneBox))
    return op.emitOpError("expects src to use the none_box slayout");
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError(
        "expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true))
    return op.emitOpError("expects trowexpand element type to be supported");
  return verifyTRowExpandValidShapes(op, srcTy, dstTy);
}

mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verify = [&]() -> LogicalResult { return verifyTRowExpandCommon(*this); };
  return dispatchVerifierByArch(getOperation(), verify, verify);
}


static ParseResult parseOptionalSortTmp(OpAsmParser &parser,
                                       OpAsmParser::UnresolvedOperand &tmp,
                                       bool &hasTmp) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }
  if (parser.parseOperand(tmp)) {
    return failure();
  }
  hasTmp = true;
  return success();
}

static ParseResult resolveTSort32Operands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand src, Type srcTy,
    OpAsmParser::UnresolvedOperand idx, Type idxTy,
    OpAsmParser::UnresolvedOperand tmp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dst, Type dstTy, bool hasTmp) {
  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }
  return parser.resolveOperand(dst, dstTy, result.operands);
}

struct TSort32ParseState {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, idxTy, tmpTy, dstTy;
  bool hasTmp = false;
};

static ParseResult parseTSort32Syntax(OpAsmParser &parser,
                                     OperationState &result,
                                     TSort32ParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src) || failed(parser.parseOptionalComma()) ||
      parser.parseOperand(state.idx) ||
      failed(parseOptionalSortTmp(parser, state.tmp, state.hasTmp)) ||
      parser.parseColonType(state.srcTy) || parser.parseComma() ||
      parser.parseType(state.idxTy))
    return failure();
  if (state.hasTmp &&
      (parser.parseComma() || parser.parseType(state.tmpTy)))
    return failure();
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(state.dst) ||
      parser.parseColonType(state.dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  TSort32ParseState state;
  if (failed(parseTSort32Syntax(parser, result, state)) ||
      failed(resolveTSort32Operands(
          parser, result, state.src, state.srcTy, state.idx, state.idxTy,
          state.tmp, state.tmpTy, state.dst, state.dstTy, state.hasTmp)))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, 1, state.hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TSort32Op::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx();
  if (getTmp()) {
    p << ", " << getTmp();
    p << " : " << getSrc().getType() << ", " << getIdx().getType()
      << ", " << getTmp().getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getIdx().getType() << ")";
  }
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult resolveOptionalTmpAfterDst(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand src, Type srcTy,
    OpAsmParser::UnresolvedOperand tmp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dst, Type dstTy, bool hasTmp) {
  if (failed(resolveRequiredOperand(parser, result, src, srcTy)) ||
      failed(resolveRequiredOperand(parser, result, dst, dstTy))) {
    return failure();
  }
  return resolveOptionalOperand(parser, result, tmp, tmpTy, hasTmp);
}

static ParseResult parseOptionalTmpIns(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &src,
    OpAsmParser::UnresolvedOperand &tmp, Type &srcTy, Type &tmpTy,
    bool &hasTmp) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy))
    return failure();
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy)))
    return failure();
  return parser.parseRParen();
}

ParseResult mlir::pto::TRsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (failed(parseOptionalTmpIns(parser, src, tmp, srcTy, tmpTy, hasTmp))) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (failed(resolveOptionalTmpAfterDst(parser, result, src, srcTy, tmp, tmpTy,
                                        dst, dstTy, hasTmp))) {
    return failure();
  }

  return success();
}

void mlir::pto::TRsqrtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

// TPOW assembly format (mirrors TRsqrt's optional-tmp style):
//   pto.tpow ins(%base, %exp[, %tmp] : !tile, !tile[, !tile])
//            outs(%dst : !tile) [attr-dict]
struct OptionalTmpBinaryParseState {
  OpAsmParser::UnresolvedOperand lhs, rhs, tmp, dst;
  Type lhsTy, rhsTy, tmpTy, dstTy;
  bool hasTmp = false;
};

static ParseResult parseOptionalTmpBinaryInputs(
    OpAsmParser &parser, OptionalTmpBinaryParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.lhs) || parser.parseComma() ||
      parser.parseOperand(state.rhs)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(state.tmp)) {
      return failure();
    }
    state.hasTmp = true;
  }
  if (parser.parseColon() || parser.parseType(state.lhsTy) ||
      parser.parseComma() || parser.parseType(state.rhsTy)) {
    return failure();
  }
  if (state.hasTmp &&
      (parser.parseComma() || parser.parseType(state.tmpTy))) {
    return failure();
  }
  return parser.parseRParen();
}

static ParseResult parseOptionalTmpBinaryOutput(
    OpAsmParser &parser, OperationState &result,
    OptionalTmpBinaryParseState &state) {
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(state.dst) || parser.parseColonType(state.dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  return parser.parseOptionalAttrDict(result.attributes);
}
