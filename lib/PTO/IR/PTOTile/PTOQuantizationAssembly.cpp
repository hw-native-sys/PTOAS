// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static ParseResult resolveTQuantOperands(OpAsmParser &parser,
                                        OperationState &result,
                                        TQuantParseState &state) {
  if (parser.resolveOperand(state.src, state.srcTy, result.operands) ||
      parser.resolveOperand(state.fp, state.fpTy, result.operands)) {
    return failure();
  }
  auto resolveOptional = [&](bool present, OpAsmParser::UnresolvedOperand value,
                             Type type) -> ParseResult {
    if (present && parser.resolveOperand(value, type, result.operands))
      return failure();
    return success();
  };
  if (failed(resolveOptional(state.hasOffset, state.offset, state.offsetTy))) {
    return failure();
  }
  if (failed(resolveOptional(state.hasTmp, state.tmp, state.tmpTy))) {
    return failure();
  }
  return parser.resolveOperand(state.dst, state.dstTy, result.operands);
}

ParseResult mlir::pto::TQuantOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  TQuantParseState state;
  NamedAttrList parsedAttrs;
  if (failed(parseTQuantInputs(parser, state)) ||
      failed(parseTQuantOutputs(parser, state)) ||
      failed(parsePTOInherentAttrs<TQuantOp>(
          parser, result, parsedAttrs, {"quant_type", "operandSegmentSizes"}))) {
    return failure();
  }
  if (failed(resolveTQuantOperands(parser, result, state))) {
    return failure();
  }
  auto &properties = result.getOrAddProperties<TQuantOp::Properties>();
  llvm::copy(ArrayRef<int32_t>(
                 {1, 1, state.hasOffset ? 1 : 0, state.hasTmp ? 1 : 0, 1}),
             properties.operandSegmentSizes.begin());
  return success();
}

void mlir::pto::TQuantOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getFp();
  if (auto offset = getOffset()) {
    p << ", " << offset << " : " << getSrc().getType() << ", "
      << getFp().getType() << ", " << offset.getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getFp().getType() << ")";
  }
  p << " outs(" << getDst();
  if (auto tmp = getTmp()) {
    p << ", " << tmp << " : " << getDst().getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << getDst().getType() << ")";
  }
  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"quant_type", "operandSegmentSizes"});
  attrs.append("quant_type", getQuantTypeAttr());
  p.printOptionalAttrDict(attrs.getAttrs());
}

static ParseResult parseTQuantMxOutputs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &outOperands,
    SmallVectorImpl<Type> &outTypes) {
  if (parser.parseKeyword("outs") || parser.parseLParen()) {
    return failure();
  }
  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    outOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseColon()) {
    return failure();
  }
  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    outTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

static ParseResult validateTQuantMxParse(OpAsmParser &parser,
                                         OperationState &result,
                                         size_t operandCount,
                                         size_t typeCount) {
  if (operandCount != typeCount) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "expects the number of outs operands to match the number of outs types");
  }
  if (operandCount != 4 && operandCount != 5) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects 4 or 5 operands in outs(...)");
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (!llvm::isa_and_nonnull<pto::QuantTypeAttr>(
          result.attributes.get("quant_type"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects quant_type attribute");
  }
  return success();
}

ParseResult mlir::pto::TQuantMxOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  Type srcTy;
  SmallVector<OpAsmParser::UnresolvedOperand, 5> outOperands;
  SmallVector<Type, 5> outTypes;
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseColonType(srcTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (failed(parseTQuantMxOutputs(parser, outOperands, outTypes)) ||
      failed(validateTQuantMxParse(parser, result, outOperands.size(),
                                   outTypes.size()))) {
    return failure();
  }
  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  for (auto [operand, type] : llvm::zip_equal(outOperands, outTypes)) {
    if (parser.resolveOperand(operand, type, result.operands)) {
      return failure();
    }
  }

  return success();
}

void mlir::pto::TQuantMxOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  p << " outs(" << getDst() << ", " << getExp() << ", " << getMax() << ", "
    << getScaling();
  if (auto expZz = getExpZz()) {
    p << ", " << expZz;
  }
  p << " : " << getDst().getType() << ", " << getExp().getType() << ", "
    << getMax().getType() << ", " << getScaling().getType();
  if (auto expZz = getExpZz()) {
    p << ", " << expZz.getType();
  }
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

static LogicalResult verifyTQuantStructural(TQuantOp op) {
  Type dstElemTy = getElemTy(op.getDst().getType());
  auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
  if (op.getQuantType() == mlir::pto::QuantType::INT8_SYM) {
    if (!op.getFp()) {
      return op.emitOpError()
             << "INT8_SYM quantization requires an fp operand";
    }
    if (op.getOffset()) {
      return op.emitOpError()
             << "INT8_SYM quantization must not have an offset operand";
    }
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError()
             << "expects dst element type i8/ui8 for INT8_SYM quantization";
    }
  } else if (op.getQuantType() == mlir::pto::QuantType::INT8_ASYM) {
    if (!op.getFp()) {
      return op.emitOpError()
             << "INT8_ASYM quantization requires an fp operand";
    }
    if (!op.getOffset()) {
      return op.emitOpError()
             << "INT8_ASYM quantization requires an offset operand";
    }
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError()
             << "expects dst element type i8/ui8 for INT8_ASYM quantization";
    }
  } else {
    return op.emitOpError("expects plain tquant quant_type to be INT8_SYM or INT8_ASYM; use tquant.mx for MX quantization");
  }
  return success();
}
static LogicalResult verifyTQuantInt8Common(TQuantOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (op.getTmp() && failed(verifyTileBufCommon(op, op.getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (!getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects src to have element type f32";
  }
  if (op.getOffset()) {
    Type offsetTy = op.getOffset().getType();
    if (failed(verifyTileBufCommon(op, offsetTy, "offset"))) {
      return failure();
    }
    if (!getElemTy(offsetTy).isF32()) {
      return op.emitOpError() << "expects offset to have element type f32";
    }
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (!getElemTy(tmpTy).isF32()) {
      return op.emitOpError() << "expects tmp to have element type f32";
    }
  }
  return success();
}

static LogicalResult verifyTQuantA2A3Tmp(TQuantOp op, Type srcTy, Type tmpTy) {
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op.emitOpError() << "expects A2/A3 tmp to use row-major layout";
  }
  if (getShapeVec(srcTy) != getShapeVec(tmpTy)) {
    return op.emitOpError() << "expects A2/A3 tmp to have the same shape as src";
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }
  auto requiredBytes = getStaticByteSize(srcTy);
  if (!requiredBytes) {
    return op.emitOpError(
        "expects A2/A3 tquant src shape to be static when tmp is provided");
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes);
}

static LogicalResult verifyTQuantA2A3Param(TQuantOp op, Type paramTy, Type dstTy,
                                           StringRef paramName) {
  if (isRowMajorTileBuf(paramTy)) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " to use non-row-major layout";
  }
  auto paramValid = getValidShapeVec(paramTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (paramValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " and dst to have rank-2 valid_shape";
  }
  if (paramValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && paramValid[0] != dstValid[0]) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " valid_shape[0] to equal dst valid_shape[0]";
  }
  if (paramValid[1] != ShapedType::kDynamic && paramValid[1] != 1) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " valid_shape[1] to be 1";
  }
  return success();
}
