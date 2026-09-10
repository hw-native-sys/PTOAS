// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTOOpsPart01.cpp after the shared type helper declarations.

static ParseResult parseGatherScatterSource(OpAsmParser& parser, OpAsmParser::UnresolvedOperand& source)
{
    if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(source)) {
        return failure();
    }
    if (!succeeded(parser.parseOptionalComma())) {
        return parser.emitError(parser.getCurrentLocation(), "expected ',' after src operand in ins(...)");
    }
    return success();
}

static ParseResult addGatherScatterMask(
    OpAsmParser& parser, OperationState& result, Attribute rawMaskAttr, bool& hasMask)
{
    auto mask = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mask) {
        return parser.emitError(parser.getCurrentLocation(), "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    }
    result.addAttribute("maskPattern", mask);
    hasMask = true;
    return success();
}

static ParseResult parseGatherScatterAxis(OpAsmParser& parser, OperationState& result)
{
    if (!succeeded(parser.parseOptionalComma())) {
        return success();
    }
    StringAttr axisAttr;
    if (parser.parseAttribute(axisAttr)) {
        return failure();
    }
    if (axisAttr.getValue() != "row" && axisAttr.getValue() != "col") {
        return parser.emitError(parser.getCurrentLocation(), "axis must be \"row\" or \"col\"");
    }
    result.addAttribute("axis", axisAttr);
    return success();
}

struct TGatherParseState {
    OpAsmParser::UnresolvedOperand src, dst, cdst;
    SmallVector<OpAsmParser::UnresolvedOperand, 3> insOps;
    SmallVector<Type, 3> insTypes;
    Type srcTy, dstTy, cdstTy;
    bool hasCdst = false;
    bool hasMask = false;
    bool hasIndices = false;
    bool hasTmp = false;
    bool hasKValue = false;
};

static ParseResult parseGatherScatterMaskInputs(
    OpAsmParser& parser, OperationState& result, Type& srcType, bool& hasMask, bool braceBeforeValidation)
{
    if (parser.parseKeyword("maskPattern") || parser.parseEqual()) {
        return failure();
    }
    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr)) {
        return failure();
    }
    // Preserve each operation's existing diagnostic order for malformed input.
    if (braceBeforeValidation && parser.parseRBrace()) {
        return failure();
    }
    if (addGatherScatterMask(parser, result, rawMaskAttr, hasMask)) {
        return failure();
    }
    if (!braceBeforeValidation && parser.parseRBrace()) {
        return failure();
    }
    if (parser.parseColonType(srcType) || parseGatherScatterAxis(parser, result) || parser.parseRParen()) {
        return failure();
    }
    return success();
}
static ParseResult parseTGatherExtraInputs(OpAsmParser& parser, TGatherParseState& state)
{
    OpAsmParser::UnresolvedOperand extra;
    if (parser.parseOperand(extra)) {
        return failure();
    }
    state.insOps.push_back(extra);
    while (succeeded(parser.parseOptionalComma())) {
        if (state.insOps.size() == 3) {
            return parser.emitError(
                parser.getCurrentLocation(), "expected at most 3 extra operands in tgather ins(...)");
        }
        if (parser.parseOperand(extra)) {
            return failure();
        }
        state.insOps.push_back(extra);
    }

    if (parser.parseColon() || parser.parseType(state.srcTy)) {
        return failure();
    }
    for (size_t i = 0; i < state.insOps.size(); ++i) {
        Type ty;
        if (parser.parseComma() || parser.parseType(ty)) {
            return failure();
        }
        state.insTypes.push_back(ty);
    }
    if (parser.parseRParen()) {
        return failure();
    }
    return success();
}

static ParseResult parseTGatherInputs(OpAsmParser& parser, OperationState& result, TGatherParseState& state)
{
    if (parseGatherScatterSource(parser, state.src)) {
        return failure();
    }
    if (succeeded(parser.parseOptionalLBrace())) {
        return parseGatherScatterMaskInputs(
            parser, result, state.srcTy, state.hasMask,
            /*braceBeforeValidation=*/true);
    }
    return parseTGatherExtraInputs(parser, state);
}

static ParseResult parseTGatherOutputs(OpAsmParser& parser, TGatherParseState& state)
{
    if (parser.parseKeyword("outs") || parser.parseLParen() || parser.parseOperand(state.dst)) {
        return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseOperand(state.cdst)) {
            return failure();
        }
        state.hasCdst = true;
    }
    if (parser.parseColonType(state.dstTy)) {
        return failure();
    }
    if (state.hasCdst && (parser.parseComma() || parser.parseType(state.cdstTy))) {
        return failure();
    }
    if (parser.parseRParen()) {
        return failure();
    }

    return success();
}

static ParseResult parseTGatherOptionalMask(OpAsmParser& parser, OperationState& result, TGatherParseState& state)
{
    if (succeeded(parser.parseOptionalKeyword("maskPattern"))) {
        if (state.hasMask) {
            return parser.emitError(parser.getCurrentLocation(), "maskPattern may only be specified once");
        }
        if (parser.parseEqual()) {
            return failure();
        }
        Attribute rawMaskAttr;
        if (parser.parseAttribute(rawMaskAttr)) {
            return failure();
        }
        if (addGatherScatterMask(parser, result, rawMaskAttr, state.hasMask)) {
            return failure();
        }
    }

    return success();
}

static ParseResult validateTGatherMaskInputs(OpAsmParser& parser, TGatherParseState& state)
{
    if (!state.insOps.empty()) {
        return parser.emitError(parser.getCurrentLocation(), "mask-pattern tgather does not take extra ins operands");
    }
    if (state.hasCdst) {
        return parser.emitError(parser.getCurrentLocation(), "mask-pattern tgather expects a single outs operand");
    }
    return success();
}

static ParseResult validateTGatherCompareInputs(OpAsmParser& parser, TGatherParseState& state)
{
    if (state.insOps.empty() ||
        !(mlir::isa<IntegerType>(state.insTypes.front()) || mlir::isa<FloatType>(state.insTypes.front()))) {
        return parser.emitError(parser.getCurrentLocation(), "compare-form tgather expects a scalar kValue operand");
    }
    state.hasKValue = true;
    if (state.insOps.size() >= 2) {
        if (!isTileLikeType(state.insTypes[1])) {
            return parser.emitError(parser.getCurrentLocation(), "compare-form tgather tmp must be tile-like");
        }
        state.hasTmp = true;
    }
    if (state.insOps.size() == 3) {
        return parser.emitError(
            parser.getCurrentLocation(), "compare-form tgather expects at most src, kValue, tmp in ins(...)");
    }
    return success();
}

static ParseResult validateTGatherIndexInputs(OpAsmParser& parser, TGatherParseState& state)
{
    if (!state.insOps.empty() && !isTileLikeType(state.insTypes.front())) {
        return parser.emitError(
            parser.getCurrentLocation(), "index-form tgather expects tile-like indices; "
                                         "compare-form must use outs(dst, cdst)");
    }
    if (!state.insOps.empty()) {
        state.hasIndices = true;
        if (state.insOps.size() >= 2) {
            if (!isTileLikeType(state.insTypes[1])) {
                return parser.emitError(parser.getCurrentLocation(), "index-form tgather tmp must be tile-like");
            }
            state.hasTmp = true;
        }
    }
    if (state.insOps.size() == 3) {
        return parser.emitError(
            parser.getCurrentLocation(), "index-form tgather expects at most src, indices, tmp in ins(...)");
    }
    return success();
}

static ParseResult validateTGatherInputs(OpAsmParser& parser, TGatherParseState& state)
{
    if (state.hasMask) {
        return validateTGatherMaskInputs(parser, state);
    }
    if (state.hasCdst) {
        return validateTGatherCompareInputs(parser, state);
    }
    return validateTGatherIndexInputs(parser, state);
}

static ParseResult resolveTGatherInputs(OpAsmParser& parser, OperationState& result, TGatherParseState& state)
{
    if (parser.resolveOperand(state.src, state.srcTy, result.operands) ||
        parser.resolveOperand(state.dst, state.dstTy, result.operands)) {
        return failure();
    }
    if (state.hasCdst && parser.resolveOperand(state.cdst, state.cdstTy, result.operands)) {
        return failure();
    }
    if (state.hasIndices && parser.resolveOperand(state.insOps[0], state.insTypes[0], result.operands)) {
        return failure();
    }
    if (state.hasTmp && parser.resolveOperand(state.insOps[1], state.insTypes[1], result.operands)) {
        return failure();
    }
    if (state.hasKValue && parser.resolveOperand(state.insOps[0], state.insTypes[0], result.operands)) {
        return failure();
    }

    return success();
}

ParseResult mlir::pto::TGatherOp::parse(OpAsmParser& parser, OperationState& result)
{
    TGatherParseState state;
    if (parseTGatherInputs(parser, result, state) || parseTGatherOutputs(parser, state) ||
        parseTGatherOptionalMask(parser, result, state) || parser.parseOptionalAttrDict(result.attributes) ||
        validateTGatherInputs(parser, state) || resolveTGatherInputs(parser, result, state)) {
        return failure();
    }
    result.addAttribute(
        "operandSegmentSizes",
        parser.getBuilder().getDenseI32ArrayAttr(
            {1, 1, state.hasCdst ? 1 : 0, state.hasIndices ? 1 : 0, state.hasTmp ? 1 : 0, state.hasKValue ? 1 : 0}));
    return success();
}

struct TScatterParseState {
    OpAsmParser::UnresolvedOperand src, indexes, dst;
    Type srcTy, idxTy, dstTy;
    bool hasMask = false;
    bool hasIndexes = false;
};

static ParseResult parseTScatterIndexInputs(OpAsmParser& parser, TScatterParseState& state)
{
    if (parser.parseOperand(state.indexes)) {
        return failure();
    }
    state.hasIndexes = true;
    if (parser.parseColon() || parser.parseType(state.srcTy) || parser.parseComma() || parser.parseType(state.idxTy) ||
        parser.parseRParen()) {
        return failure();
    }
    return success();
}

static ParseResult parseTScatterInputs(OpAsmParser& parser, OperationState& result, TScatterParseState& state)
{
    if (parseGatherScatterSource(parser, state.src)) {
        return failure();
    }
    if (succeeded(parser.parseOptionalLBrace())) {
        return parseGatherScatterMaskInputs(
            parser, result, state.srcTy, state.hasMask,
            /*braceBeforeValidation=*/false);
    }
    return parseTScatterIndexInputs(parser, state);
}

static ParseResult parseTScatterOutput(OpAsmParser& parser, TScatterParseState& state)
{
    if (parser.parseKeyword("outs") || parser.parseLParen() || parser.parseOperand(state.dst) ||
        parser.parseColonType(state.dstTy) || parser.parseRParen()) {
        return failure();
    }

    return success();
}

static ParseResult validateTScatterInputs(OpAsmParser& parser, TScatterParseState& state)
{
    if (state.hasMask && state.hasIndexes) {
        return parser.emitError(parser.getCurrentLocation(), "mask-pattern tscatter does not take indexes");
    }
    if (!state.hasMask && !state.hasIndexes) {
        return parser.emitError(parser.getCurrentLocation(), "expected indexes operand or maskPattern for tscatter");
    }

    return success();
}

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
