// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static Type parseShapedPTOType(OpAsmParser& parser, StringRef head)
{
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elementType;
    if (failed(parsePTOShapeAndElement(parser, shape, elementType))) {
        return {};
    }
    MLIRContext* context = parser.getContext();
    if (head == "pto.tile_view") {
        return PartitionTensorViewType::get(context, shape, elementType);
    }
    if (head == "pto.tile") {
        return TileType::get(context, shape, elementType);
    }
    return TensorViewType::get(context, shape, elementType);
}

static Type parsePtrPTOType(OpAsmParser& parser)
{
    Type elementType;
    if (parser.parseLess() || parser.parseType(elementType)) {
        return {};
    }
    MLIRContext* context = parser.getContext();
    auto memorySpace = AddressSpaceAttr::get(context, AddressSpace::GM);
    if (succeeded(parser.parseOptionalComma())) {
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
            return {};
        }
        auto parsed = parsePtrAddressSpaceKeyword(keyword);
        if (!parsed) {
            parser.emitError(
                parser.getCurrentLocation(), "!pto.ptr address space must be one of "
                                             "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
            return {};
        }
        memorySpace = AddressSpaceAttr::get(context, *parsed);
    }
    if (parser.parseGreater()) {
        return {};
    }
    return PtrType::get(context, elementType, memorySpace);
}

static Type parseKnownPTOType(OpAsmParser& parser, StringRef head)
{
    if (head == "pto.ptr") {
        return parsePtrPTOType(parser);
    }
    if (head == "pto.tile_view" || head == "pto.tile" || head == "pto.tensor_view") {
        return parseShapedPTOType(parser, head);
    }
    return {};
}

[[maybe_unused]] static Type parsePTOTypeAllowNoBang(OpAsmParser& parser)
{
    Type type;
    OptionalParseResult optionalType = parser.parseOptionalType(type);
    if (optionalType.has_value()) {
        return failed(*optionalType) ? Type() : type;
    }
    StringRef head;
    if (parser.parseKeyword(&head)) {
        return {};
    }
    return parseKnownPTOType(parser, head);
}

mlir::Type TensorViewType::parse(::mlir::AsmParser& parser)
{
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elementType;
    Attribute layout;
    if (failed(parseViewShapeElemAndLayout(
            parser, shape, elementType, layout, /*allowDynamic=*/true))) {
        return Type();
    }
    return TensorViewType::get(parser.getContext(), shape, elementType, layout);
}

void TensorViewType::print(::mlir::AsmPrinter& printer) const
{
    printViewShapeElemAndLayout(printer, getShape(), getElementType(),
                                getLayout());
}

mlir::Type PtrType::parse(::mlir::AsmParser& parser)
{
    Type elementType;
    if (failed(parser.parseLess()) || failed(parser.parseType(elementType))) {
        return {};
    }

    auto memorySpace = pto::AddressSpaceAttr::get(parser.getContext(), pto::AddressSpace::GM);
    if (succeeded(parser.parseOptionalComma())) {
        StringRef memorySpaceKeyword;
        if (failed(parser.parseKeyword(&memorySpaceKeyword))) {
            return {};
        }
        auto parsed = parsePtrAddressSpaceKeyword(memorySpaceKeyword);
        if (!parsed) {
            parser.emitError(
                parser.getCurrentLocation(), "!pto.ptr address space must be one of "
                                             "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
            return {};
        }
        memorySpace = pto::AddressSpaceAttr::get(parser.getContext(), *parsed);
    }

    if (failed(parser.parseGreater())) {
        return {};
    }
    return PtrType::get(parser.getContext(), elementType, memorySpace);
}

void PtrType::print(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getElementType();
    StringRef memorySpaceKeyword = printPtrAddressSpaceKeyword(getMemorySpace().getAddressSpace());
    if (!memorySpaceKeyword.empty()) {
        printer << ", " << memorySpaceKeyword;
    }
    printer << ">";
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm to support both:
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op follows textual input order.
//===----------------------------------------------------------------------===//

static ParseResult parseTDivSOperands(
    OpAsmParser& parser, OpAsmParser::UnresolvedOperand& op0, OpAsmParser::UnresolvedOperand& op1, Type& ty0, Type& ty1)
{
    if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(op0) || parser.parseComma() ||
        parser.parseOperand(op1) || parser.parseColonType(ty0) || parser.parseComma() || parser.parseType(ty1) ||
        parser.parseRParen()) {
        return failure();
    }
    return success();
}

static ParseResult parseTDivSResult(OpAsmParser& parser, OpAsmParser::UnresolvedOperand& dst, Type& dstType)
{
    if (parser.parseKeyword("outs") || parser.parseLParen() || parser.parseOperand(dst) ||
        parser.parseColonType(dstType) || parser.parseRParen()) {
        return failure();
    }
    return success();
}

static ParseResult validateTDivSTypes(OpAsmParser& parser, Type ty0, Type ty1, Type dstType)
{
    auto tile0 = dyn_cast<mlir::pto::TileBufType>(ty0);
    auto tile1 = dyn_cast<mlir::pto::TileBufType>(ty1);
    if ((tile0 && tile1) || (!tile0 && !tile1)) {
        return parser.emitError(
            parser.getCurrentLocation(), "expected exactly one tile_buf operand and one scalar operand");
    }

    if (!dyn_cast<mlir::pto::TileBufType>(dstType)) {
        return parser.emitError(parser.getCurrentLocation(), "expected outs type to be !pto.tile_buf<...>");
    }
    return success();
}

static ParseResult resolveTDivSOperands(
    OpAsmParser& parser, OperationState& result, OpAsmParser::UnresolvedOperand op0, OpAsmParser::UnresolvedOperand op1,
    OpAsmParser::UnresolvedOperand dst, Type ty0, Type ty1, Type dstType)
{
    if (parser.resolveOperand(op0, ty0, result.operands) || parser.resolveOperand(op1, ty1, result.operands) ||
        parser.resolveOperand(dst, dstType, result.operands)) {
        return failure();
    }
    return success();
}

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser& parser, OperationState& result)
{
    OpAsmParser::UnresolvedOperand op0, op1, dst;
    Type ty0, ty1, dstType;
    NamedAttrList attrs;
    if (parseTDivSOperands(parser, op0, op1, ty0, ty1) || parseTDivSResult(parser, dst, dstType) ||
        parser.parseOptionalAttrDict(attrs) || validateTDivSTypes(parser, ty0, ty1, dstType) ||
        resolveTDivSOperands(parser, result, op0, op1, dst, ty0, ty1, dstType)) {
        return failure();
    }

    result.addAttributes(attrs);
    return success();
}

void mlir::pto::TDivSOp::print(OpAsmPrinter& p)
{
    p << " ins(";
    p << getSrc() << ", " << getScalar() << " : " << getSrc().getType() << ", " << getScalar().getType();
    p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

    p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// pto.tgather custom asm supports three PTO-ISA forms:
//   1) index+tmp   : ins(%src, %indices, %tmp : srcTy, indicesTy, tmpTy) outs(%dst : dstTy)
//   2) compare+tmp : ins(%src, %kValue, %tmp : srcTy, scalarTy, tmpTy)
//                    outs(%dst, %cdst : dstTy, cdstTy) {cmpMode = #pto.cmp<gt>, offset = 7}
//   3) mask        : ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : srcTy) outs(%dst : dstTy)
//===----------------------------------------------------------------------===//

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
