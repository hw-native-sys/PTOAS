// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static ParseResult resolveTScatterInputs(OpAsmParser& parser, OperationState& result, TScatterParseState& state)
{
    if (parser.resolveOperand(state.src, state.srcTy, result.operands) ||
        parser.resolveOperand(state.dst, state.dstTy, result.operands) ||
        (state.hasIndexes && parser.resolveOperand(state.indexes, state.idxTy, result.operands))) {
        return failure();
    }

    return success();
}

ParseResult mlir::pto::TScatterOp::parse(OpAsmParser& parser, OperationState& result)
{
    TScatterParseState state;
    if (parseTScatterInputs(parser, result, state) || parseTScatterOutput(parser, state) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }
    if (result.attributes.get("maskPattern")) {
        state.hasMask = true;
    }
    if (validateTScatterInputs(parser, state)) {
        return failure();
    }
    return resolveTScatterInputs(parser, result, state);
}

void mlir::pto::TGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (auto mp = getMaskPatternAttr()) {
    p << "{maskPattern = " << mp << "} : " << getSrc().getType();
    if (auto axisAttr = getAxisAttr()) {
      p << ", " << axisAttr;
    }
  } else if (getCdst()) {
    p << getKValue();
    if (getTmp()) {
      p << ", " << getTmp();
      p << " : " << getSrc().getType() << ", " << getKValue().getType()
        << ", " << getTmp().getType();
    } else {
      p << " : " << getSrc().getType() << ", " << getKValue().getType();
    }
  } else {
    p << getIndices();
    if (getTmp()) {
      p << ", " << getTmp();
      p << " : " << getSrc().getType() << ", " << getIndices().getType()
        << ", " << getTmp().getType();
    } else {
      p << " : " << getSrc().getType() << ", " << getIndices().getType();
    }
  }
  p << ") outs(" << getDst();
  if (getCdst()) {
    p << ", " << getCdst();
  }
  p << " : " << getDst().getType();
  if (getCdst()) {
    p << ", " << getCdst().getType();
  }
  p << ")";

  if (getMaskPatternAttr()) {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"maskPattern", "axis", "operandSegmentSizes"});
  } else {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"axis", "operandSegmentSizes"});
  }
}


void mlir::pto::TScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (getMaskPatternAttr()) {
    p << "{maskPattern = " << getMaskPatternAttr() << "} : " << getSrc().getType();
    if (auto axisAttr = getAxisAttr()) {
      p << ", " << axisAttr;
    }
  } else {
    p << getIndexes() << " : " << getSrc().getType() << ", "
      << getIndexes().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"maskPattern", "axis"});
}

namespace {
struct CommRecvClause {
  OpAsmParser::UnresolvedOperand ping;
  std::optional<OpAsmParser::UnresolvedOperand> pong;
  Type pingTy;
  Type pongTy;
};

static ParseResult parseCommRecvClause(OpAsmParser &parser,
                                       CommRecvClause &recvClause) {
  if (parser.parseKeyword("recv") || parser.parseLParen() ||
      parser.parseOperand(recvClause.ping)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand pong;
    if (parser.parseOperand(pong)) {
      return failure();
    }
    recvClause.pong = pong;
  }
  return parser.parseRParen();
}

static ParseResult parseCommGroupOperands(
    OpAsmParser& parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand>& groupOps)
{
    if (parser.parseComma() || parser.parseKeyword("group") || parser.parseLParen()) {
        return failure();
    }

    OpAsmParser::UnresolvedOperand group;
    if (parser.parseOperand(group)) {
        return failure();
    }
    groupOps.push_back(group);
    while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseOperand(group)) {
            return failure();
        }
        groupOps.push_back(group);
    }

    if (parser.parseRParen()) {
        return failure();
    }

    return success();
}

static ParseResult parseCommOperandTypes(
    OpAsmParser& parser, SmallVectorImpl<Type>& fixedTypes, CommRecvClause& recvClause, size_t groupCount,
    SmallVectorImpl<Type>& groupTypes)
{
    if (parser.parseColon()) {
        return failure();
    }

    for (size_t i = 0; i < fixedTypes.size(); ++i) {
        if (i != 0 && parser.parseComma()) {
            return failure();
        }
        if (parser.parseType(fixedTypes[i])) {
            return failure();
        }
    }
    if (parser.parseComma() || parser.parseType(recvClause.pingTy)) {
        return failure();
    }
    if (recvClause.pong) {
        if (parser.parseComma() || parser.parseType(recvClause.pongTy)) {
            return failure();
        }
    }
    for (size_t i = 0; i < groupCount; ++i) {
        Type groupTy;
        if (parser.parseComma() || parser.parseType(groupTy)) {
            return failure();
        }
        groupTypes.push_back(groupTy);
    }
    if (parser.parseRParen()) {
        return failure();
    }

    return success();
}

static ParseResult parseCommRequiredAttributes(
    OpAsmParser& parser, OperationState& result, ArrayRef<StringRef> requiredAttrs)
{
    NamedAttrList attrs;
    if (parser.parseOptionalAttrDict(attrs)) {
        return failure();
    }
    for (StringRef attrName : requiredAttrs) {
        if (!attrs.get(attrName)) {
            return parser.emitError(parser.getCurrentLocation()) << "expected '" << attrName << "' attribute";
        }
    }
    result.addAttributes(attrs);

    return success();
}

static ParseResult resolveCommOperands(
    OpAsmParser& parser, OperationState& result, ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    ArrayRef<Type> fixedTypes, CommRecvClause& recvClause, ArrayRef<OpAsmParser::UnresolvedOperand> groupOps,
    ArrayRef<Type> groupTypes)
{
    for (auto [operand, type] : llvm::zip_equal(fixedOperands, fixedTypes)) {
        if (parser.resolveOperand(operand, type, result.operands)) {
            return failure();
        }
    }
    if (parser.resolveOperand(recvClause.ping, recvClause.pingTy, result.operands)) {
        return failure();
    }
    if (recvClause.pong && parser.resolveOperand(*recvClause.pong, recvClause.pongTy, result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(groupOps, groupTypes, parser.getCurrentLocation(), result.operands)) {
        return failure();
    }

    return success();
}

static ParseResult parseCommCollectiveTail(
    OpAsmParser& parser, OperationState& result, ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    SmallVectorImpl<Type>& fixedTypes, CommRecvClause& recvClause,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& groupOps, SmallVectorImpl<Type>& groupTypes,
    ArrayRef<int32_t> operandSegmentsPrefix, ArrayRef<StringRef> requiredAttrs)
{
    if (parseCommGroupOperands(parser, groupOps) ||
        parseCommOperandTypes(parser, fixedTypes, recvClause, groupOps.size(), groupTypes) ||
        parseCommRequiredAttributes(parser, result, requiredAttrs) ||
        resolveCommOperands(parser, result, fixedOperands, fixedTypes, recvClause, groupOps, groupTypes)) {
        return failure();
    }
    SmallVector<int32_t, mlir::pto::kValue5> segmentSizes(operandSegmentsPrefix.begin(), operandSegmentsPrefix.end());
    segmentSizes.push_back(static_cast<int32_t>(groupOps.size()));
    result.addAttribute("operandSegmentSizes", parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
    return success();
}

static void printCommRecvClause(OpAsmPrinter& p, Value ping, Value pong)
{
    p << "recv(" << ping;
    if (pong) {
        p << ", " << pong;
    }
    p << ")";
}

static void printCommGroupTypes(OpAsmPrinter& p, ValueRange group)
{
    for (Value groupValue : group) {
        p << ", " << groupValue.getType();
    }
}

static void printCommGroupClause(OpAsmPrinter& p, ValueRange group)
{
    p << "group(";
    p.printOperands(group);
    p << ")";
}

} // namespace

static ParseResult parseCommSingleFixedOperand(OpAsmParser& parser, OperationState& result)
{
    OpAsmParser::UnresolvedOperand src;
    CommRecvClause recvClause;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> groupOps;
    SmallVector<Type, mlir::pto::kValue4> groupTypes;

    if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma()) {
        return failure();
    }
    if (failed(parseCommRecvClause(parser, recvClause))) {
        return failure();
    }
