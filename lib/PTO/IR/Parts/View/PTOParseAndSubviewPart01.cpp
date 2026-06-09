// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static ParseResult parseKeywordedOperand(OpAsmParser &parser, StringRef keyword,
                                         OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseKeyword(keyword) || parser.parseLParen() ||
      parser.parseOperand(operand))
    return failure();
  return success();
}

static ParseResult parseRequiredCommaOperand(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseComma() || parser.parseOperand(operand))
    return failure();
  return success();
}

static ParseResult parseOptionalCommaOperand(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    bool &isPresent) {
  if (failed(parser.parseOptionalComma()))
    return success();
  if (parser.parseOperand(operand))
    return failure();
  isPresent = true;
  return success();
}

static ParseResult parseOptionalCommaType(OpAsmParser &parser, bool isPresent,
                                          Type &type) {
  if (!isPresent)
    return success();
  if (parser.parseComma() || parser.parseType(type))
    return failure();
  return success();
}

static ParseResult parseOutsClauseWithOptionalAttrDict(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &dst, Type &dstTy) {
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

static ParseResult resolveOptionalOperand(
    OpAsmParser &parser, bool isPresent, OpAsmParser::UnresolvedOperand &operand,
    Type type, SmallVectorImpl<Value> &operands) {
  if (!isPresent)
    return success();
  return parser.resolveOperand(operand, type, operands);
}

static void addOperandSegmentSizesAttr(OpAsmParser &parser, OperationState &result,
                                       ArrayRef<int32_t> sizes) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(sizes));
}

ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, dstTy, idxTy, tmpTy;
  bool hasTmp = false;

  if (parseKeywordedOperand(parser, "ins", src))
    return failure();
  if (parseRequiredCommaOperand(parser, idx))
    return failure();
  if (parseOptionalCommaOperand(parser, tmp, hasTmp))
    return failure();
  if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(idxTy))
    return failure();
  if (parseOptionalCommaType(parser, hasTmp, tmpTy))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parseOutsClauseWithOptionalAttrDict(parser, result, dst, dstTy))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands))
    return failure();
  if (resolveOptionalOperand(parser, hasTmp, tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  addOperandSegmentSizesAttr(parser, result, {1, 1, hasTmp ? 1 : 0, 1});
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

ParseResult mlir::pto::TRsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parseKeywordedOperand(parser, "ins", src))
    return failure();
  if (parseOptionalCommaOperand(parser, tmp, hasTmp))
    return failure();
  if (parser.parseColonType(srcTy))
    return failure();
  if (parseOptionalCommaType(parser, hasTmp, tmpTy))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parseOutsClauseWithOptionalAttrDict(parser, result, dst, dstTy))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (resolveOptionalOperand(parser, hasTmp, tmp, tmpTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TRsqrtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp())
    p << ", " << getTmp();
  p << " : " << getSrc().getType();
  if (getTmp())
    p << ", " << getTmp().getType();
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

static ParseResult parseTRowExpandBinaryLikeOp(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;

  if (parseKeywordedOperand(parser, "ins", src0) ||
      parseRequiredCommaOperand(parser, src1))
    return failure();
  if (parseOptionalCommaOperand(parser, tmp, hasTmp))
    return failure();
  if (parser.parseColon())
    return failure();
  if (parser.parseType(src0Ty) || parser.parseComma() || parser.parseType(src1Ty))
    return failure();
  if (parseOptionalCommaType(parser, hasTmp, tmpTy))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parseOutsClauseWithOptionalAttrDict(parser, result, dst, dstTy))
    return failure();

  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands))
    return failure();
  if (resolveOptionalOperand(parser, hasTmp, tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  addOperandSegmentSizesAttr(parser, result, {1, 1, hasTmp ? 1 : 0, 1});
  return success();
}

static void printTRowExpandBinaryLikeOp(OpAsmPrinter &p, Operation *op, Value src0,
                                        Value src1, Value tmp, Value dst) {
  p << " ins(" << src0 << ", " << src1;
  if (tmp) {
    p << ", " << tmp;
    p << " : " << src0.getType() << ", " << src1.getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << src0.getType() << ", " << src1.getType() << ")";
  }
  p << " outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRowExpandDivOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandDivOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMulOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

