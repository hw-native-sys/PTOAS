// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyTPartAddA5(TPartAddOp op) {
  return verifyTPartBinaryA5(op, op.getSrc0().getType(),
                             op.getSrc1().getType(), op.getDst().getType(),
                             "tpartadd");
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartAddA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartAddA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPartMinMax(Operation *op, Type src0, Type src1,
                                       Type dst, StringRef opName, bool isA5) {
  auto elem = verifyPTOShapedBinarySameElemAndShape(op, src0, src1, dst);
  if (failed(elem))
    return failure();
  if (!isA5 && failed(verifyPartialValidPattern(op, src0, src1, dst)))
    return failure();
  bool supported = isA5
                       ? (elem->isInteger(32) || elem->isInteger(16) ||
                          elem->isInteger(8) || elem->isF16() ||
                          elem->isBF16() || elem->isF32())
                       : (elem->isInteger(32) || elem->isInteger(16) ||
                          elem->isF16() || elem->isF32());
  if (!supported)
    return op->emitOpError()
           << "expects " << (isA5 ? "A5 " : "A2/A3 ") << opName
           << (isA5 ? " element type to be i32/i16/i8/f16/bf16/f32"
                    : " element type to be i32/i16/f16/f32");
  return isA5 ? verifyPartialValidPatternLoose(op, src0, src1, dst)
              : success();
}

#define PTO_DEFINE_PART_MINMAX_VERIFY(OpClass, opName)                             \
  mlir::LogicalResult mlir::pto::OpClass::verify() {                               \
    auto verifyA2A3 = [&]() {                                                      \
      return verifyTPartMinMax(getOperation(), getSrc0().getType(),                \
                               getSrc1().getType(), getDst().getType(), opName,     \
                               false);                                             \
    };                                                                             \
    auto verifyA5 = [&]() {                                                        \
      return verifyTPartMinMax(getOperation(), getSrc0().getType(),                \
                               getSrc1().getType(), getDst().getType(), opName,     \
                               true);                                              \
    };                                                                             \
    return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);           \
  }

PTO_DEFINE_PART_MINMAX_VERIFY(TPartMaxOp, "tpartmax")
PTO_DEFINE_PART_MINMAX_VERIFY(TPartMinOp, "tpartmin")

static LogicalResult verifyTPartArgIndices(Operation *op, Type src0Ty,
                                           Type src1Ty, Type src0IdxTy,
                                           Type src1IdxTy, Type dstTy,
                                           Type dstIdxTy) {
  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy)) {
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  }
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy)) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");
  }

  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) ||
      dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  }
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");
  }

  return success();
}

static LogicalResult verifyTPartArgElementType(Operation *op, Type elem,
                                               StringRef opName) {
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32())) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
    }
  } else {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op->emitOpError() << "expects A2/A3 " << opName
                               << " element type to be i32/i16/f16/f32";
    }
  }
  return success();
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElem =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElem) ||
      failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)) ||
      failed(verifyTPartArgIndices(op, src0Ty, src1Ty, src0IdxTy, src1IdxTy,
                                   dstTy, dstIdxTy)))
    return failure();
  return verifyTPartArgElementType(op, *dataElem, opName);
}

mlir::LogicalResult mlir::pto::TPartArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

static LogicalResult verifyTPartMulA2A3(TPartMulOp op) {
  return verifyTPartBinaryA2A3(op, op.getSrc0().getType(),
                               op.getSrc1().getType(), op.getDst().getType(),
                               "tpartmul");
}

static LogicalResult verifyTPartMulA5(TPartMulOp op) {
  return verifyTPartBinaryA5(op, op.getSrc0().getType(),
                             op.getSrc1().getType(), op.getDst().getType(),
                             "tpartmul");
}

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartMulA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartMulA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTPReluElementTypes(TPReluOp op, Type t0, Type t1,
                                                Type td) {
  Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op.emitOpError("expects dst/src0/src1 to have the same element type");
    return failure();
  }
  if (!(e0.isF16() || e0.isF32())) {
    op.emitOpError("expects dst/src0/src1 element type to be f16 or f32");
    return failure();
  }
  return e0;
}

static FailureOr<std::tuple<Type, Type, Type, Type>> verifyTPReluCommon(
    TPReluOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type tt = op.getTmp() ? op.getTmp().getType() : Type{};
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }
  if (tt && failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }

  if (failed(verifyTPReluElementTypes(op, t0, t1, td)))
    return failure();
  if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
      !isRowMajorTileBuf(td)) {
    op.emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, t1, td, "src1", "dst"))) {
    return failure();
  }

  if (getShapeVec(t0) != getShapeVec(t1) ||
      getShapeVec(t0) != getShapeVec(td)) {
    op.emitOpError("expects src0/src1/dst to have the same shape");
    return failure();
  }
  return std::make_tuple(t0, t1, tt, td);
}

static LogicalResult verifyTPReluA2A3Tmp(TPReluOp op, Type tt, Type td) {
  Type tmpElem = getElemTy(tt);
  auto tmpIntTy = mlir::dyn_cast<IntegerType>(tmpElem);
  if (!tmpIntTy || tmpIntTy.getWidth() != 8) {
    return op.emitOpError("expects A2/A3 tmp element type to be u8");
  }
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto tmpShape = getShapeVec(tt);
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (tmpShape.size() != 2 || dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  if (dstValid[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] < dstValid[0] + 1) {
    return op.emitOpError()
           << "expects A2/A3 tmp shape[0] to be at least dst valid_shape[0] + 1 ("
           << (dstValid[0] + 1) << ")";
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic) {
    int64_t packedMaskCols = llvm::divideCeil(dstValid[1], int64_t{8});
    if (tmpValid[1] < packedMaskCols) {
      return op.emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least ceil(dst valid_shape[1] / 8) ("
             << packedMaskCols << ")";
    }
  }
  if (dstValid[0] == ShapedType::kDynamic ||
      dstValid[1] == ShapedType::kDynamic) {
    return op.emitOpError(
        "expects A2/A3 tprelu dst valid_shape to be static when tmp is provided");
  }
  int64_t packedCols = std::max<int64_t>(
      32, llvm::divideCeil(llvm::divideCeil(dstValid[1], int64_t{8}),
                           int64_t{32}) *
              32);
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstValid[0] + 1) * static_cast<uint64_t>(packedCols)))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTPReluA2A3(TPReluOp op) {
  auto tysOr = verifyTPReluCommon(op);
  if (failed(tysOr)) {
    return failure();
  }
  auto [t0, t1, tt, td] = *tysOr;
  (void)t0;
  (void)t1;
  if (!tt) {
    return success();
  }
  if (failed(verifyTPReluA2A3Tmp(op, tt, td))) {
    return failure();
  }
  if (auto arch = getVerifierArchName(op.getOperation());
      arch && arch->equals_insensitive("a3")) {
    if (op.getSrc0() == op.getSrc1() || op.getSrc0() == op.getTmp() ||
        op.getSrc0() == op.getDst() || op.getSrc1() == op.getTmp() ||
        op.getSrc1() == op.getDst() || op.getTmp() == op.getDst()) {
      return op.emitOpError(
          "expects A3 src0, src1, tmp, and dst to use different storage");
    }
  }
  return success();
}

static LogicalResult verifyTPReluA5(TPReluOp op) {
  auto tysOr = verifyTPReluCommon(op);
  if (failed(tysOr)) {
    return failure();
  }
  auto [t0, t1, tt, td] = *tysOr;
  (void)t0;
  (void)t1;
  (void)td;
  if (tt && failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPReluA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPReluA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

struct TQuantParseState {
  OpAsmParser::UnresolvedOperand src, fp, offset, dst, tmp;
  Type srcTy, fpTy, offsetTy, dstTy, tmpTy;
  bool hasOffset = false;
  bool hasTmp = false;
};

static ParseResult parseTQuantInputs(OpAsmParser &parser,
                                    TQuantParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src) || parser.parseComma() ||
      parser.parseOperand(state.fp)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(state.offset)) {
      return failure();
    }
    state.hasOffset = true;
  }
  if (parser.parseColon() || parser.parseType(state.srcTy) ||
      parser.parseComma() || parser.parseType(state.fpTy)) {
    return failure();
  }
  if (state.hasOffset &&
      (parser.parseComma() || parser.parseType(state.offsetTy))) {
    return failure();
  }
  return parser.parseRParen();
}

static ParseResult parseTQuantOutputs(OpAsmParser &parser,
                                     TQuantParseState &state) {
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(state.dst)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(state.tmp)) {
      return failure();
    }
    state.hasTmp = true;
  }
  if (parser.parseColonType(state.dstTy)) {
    return failure();
  }
  if (state.hasTmp && (parser.parseComma() || parser.parseType(state.tmpTy))) {
    return failure();
  }
  return parser.parseRParen();
}

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

static LogicalResult verifyTQuantA2A3(TQuantOp op) {
  if (failed(verifyTQuantInt8Common(op))) {
    return failure();
  }
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError()
           << "expects A2/A3 src and dst to use row-major layout";
  }
  if (op.getTmp() &&
      failed(verifyTQuantA2A3Tmp(op, srcTy, op.getTmp().getType()))) {
    return failure();
  }
  if (failed(verifyTQuantA2A3Param(op, fpTy, dstTy, "fp"))) {
    return failure();
  }
  if (op.getOffset() &&
      failed(verifyTQuantA2A3Param(op, op.getOffset().getType(), dstTy, "offset"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  if (failed(verifyTQuantStructural(*this))) {
    return failure();
  }
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTQuantA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantInt8Common(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static std::optional<int64_t> mxCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> mxCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> mxCeilDiv(int64_t value, int64_t divisor) {
  if (value < 0 || divisor == 0 || divisor < 0) {
    return std::nullopt;
  }
  auto plus = mxCheckedAdd(value, divisor - 1);
  return plus ? std::optional<int64_t>(*plus / divisor) : std::nullopt;
}

static std::optional<int64_t> mxAlignTo(int64_t value, int64_t alignment) {
  if (alignment == 0 || alignment < 0) {
    return std::nullopt;
  }
  auto quotient = mxCeilDiv(value, alignment);
  if (!quotient || *quotient > std::numeric_limits<int64_t>::max() / alignment) {
    return std::nullopt;
  }
  return *quotient * alignment;
}

static std::optional<int64_t> mxCapacityElems(Type type) {
  auto shape = getShapeVec(type);
  return mxCheckedMul(shape[0], shape[1]);
}

static std::optional<int64_t> mxCapacityBytes(Type type) {
  auto elems = mxCapacityElems(type);
  unsigned bytes = getElemByteSize(getElemTy(type));
  if (!elems || bytes == 0 ||
      *elems >
          std::numeric_limits<int64_t>::max() / static_cast<int64_t>(bytes)) {
    return std::nullopt;
  }
  return *elems * static_cast<int64_t>(bytes);
}

static LogicalResult mxRequireCapacity(TQuantMxOp op, StringRef name, Type type,
                                       int64_t required) {
  auto actual = mxCapacityElems(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " elements";
  }
  return success();
}

static LogicalResult mxRequireCapacityBytes(TQuantMxOp op, StringRef name,
                                            Type type, int64_t required) {
  auto actual = mxCapacityBytes(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " bytes";
  }
  return success();
}

static LogicalResult mxRequireCompact(TQuantMxOp op, StringRef name, Type type) {
  auto valid = getValidShapeVec(type);
  auto physical = getShapeVec(type);
  if (valid[0] != 1 && physical[1] != valid[1]) {
    return op.emitOpError() << "expects " << name
                            << " valid elements to form a compact physical prefix";
  }
  return success();
}

static LogicalResult mxRequireStaticShape(TQuantMxOp op, StringRef name,
                                          Type type) {
  for (int64_t dim : getValidShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  for (int64_t dim : getShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxTilesAndForm(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst", /*allowLowPrecision=*/true)) ||
      failed(verifyNDStyleVecTile(op, expTy, "exp")) ||
      failed(verifyNDStyleVecTile(op, maxTy, "max")) ||
      failed(verifyNDStyleVecTile(op, scalingTy, "scaling"))) {
    return failure();
  }
  if (op.getExpZz() &&
      failed(verifyNDStyleVecTile(op, op.getExpZz().getType(), "exp_zz"))) {
    return failure();
  }
  const bool isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  if (op.getInterleave() && !isDn) {
    return op.emitOpError("expects interleave to be used only with grpAxis=axis0");
  }
  if (op.getExpZz() && isDn) {
    return op.emitOpError("expects the deprecated exp_zz form to use grpAxis=axis1; use pto.tmov with a non-scaling tmp for axis0 exponents");
  }
  auto quantType = op.getQuantType();
  if (quantType != mlir::pto::QuantType::MXFP8 &&
      quantType != mlir::pto::QuantType::MXFP4_E2M1) {
    return op.emitOpError("expects quant_type to be MXFP8 or MXFP4_E2M1");
  }
  if (op.getExpZz() && !op.getStoreMode()) {
    return op.emitOpError("expects storeMode when exp_zz is present");
  }
  if (op.getStoreMode() && !op.getExpZz()) {
    return op.emitOpError("expects exp_zz when storeMode is present");
  }
  if (op.getStoreMode() && op.getQuantScaleAlg() != mlir::pto::QuantScaleAlg::OCP) {
    return op.emitOpError("storeMode form must not override quantScaleAlg");
  }
  return success();
}

static LogicalResult verifyTQuantMxElemTypes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(op.getDst().getType());
  Type expElem = getElemTy(op.getExp().getType());
  Type maxElem = getElemTy(op.getMax().getType());
  Type scalingElem = getElemTy(op.getScaling().getType());
  if (!(srcElem.isF32() || srcElem.isF16() || srcElem.isBF16())) {
    return op.emitOpError("expects src element type to be f32/f16/bf16");
  }
  if (!expElem.isInteger(8)) {
    return op.emitOpError("expects exp element type to be i8/ui8");
  }
  if (op.getExpZz() && !getElemTy(op.getExpZz().getType()).isInteger(8)) {
    return op.emitOpError("expects exp_zz element type to be i8/ui8");
  }
  if (maxElem != srcElem) {
    return op.emitOpError("expects max element type to match src element type");
  }
  if (scalingElem != srcElem) {
    return op.emitOpError("expects scaling element type to match src element type");
  }
  if (op.getQuantType() == mlir::pto::QuantType::MXFP8) {
    if (!dstElem.isInteger(8)) {
      return op.emitOpError("expects MXFP8 dst element type to be i8/ui8");
    }
  } else {
    if (!isa<pto::F4E2M1x2Type>(dstElem)) {
      return op.emitOpError("expects MXFP4_E2M1 dst element type to be !pto.f4E2M1x2");
    }
    if (!(srcElem.isF16() || srcElem.isBF16())) {
      return op.emitOpError("expects MXFP4_E2M1 src element type to be f16/bf16");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxStaticShapes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  for (auto [name, type] : {std::pair<StringRef, Type>("src", srcTy),
                            {"dst", dstTy}, {"exp", expTy}, {"max", maxTy},
                            {"scaling", scalingTy}}) {
    if (failed(mxRequireStaticShape(op, name, type))) {
      return failure();
    }
  }
  if (op.getExpZz()) {
    Type expZzTy = op.getExpZz().getType();
    if (llvm::is_contained(getValidShapeVec(expZzTy), ShapedType::kDynamic) ||
        llvm::is_contained(getShapeVec(expZzTy), ShapedType::kDynamic)) {
      return op.emitOpError("expects static valid and physical shapes for exp_zz in the deprecated fused MX quantization form");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxShapes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  for (Type type : {srcTy, dstTy, expTy, maxTy, scalingTy}) {
    if (getValidShapeVec(type).size() != 2 || getShapeVec(type).size() != 2)
      return op.emitOpError(
          "expects rank-2 valid and physical shapes for MX quantization");
  }
  if (op.getExpZz() &&
      (getValidShapeVec(op.getExpZz().getType()).size() != 2 ||
       getShapeVec(op.getExpZz().getType()).size() != 2))
    return op.emitOpError(
        "expects rank-2 valid and physical shapes for exp_zz");
  if (failed(verifyTQuantMxStaticShapes(op)))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, maxTy, "src", "max")) ||
      failed(verifyTileBufSameElemType(op, srcTy, scalingTy, "src", "scaling")) ||
      failed(verifyTileBufSameLogicalExtent(op, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  return success();
}

struct TQuantMxA5 {
  Type srcTy, dstTy, expTy, maxTy, scalingTy;
  Type srcElem;
  SmallVector<int64_t, 4> dstValid, expValid, expPhysical;
  SmallVector<int64_t, 4> dstPhysical, maxPhysical, scalingPhysical;
  bool isDn, isMxFp4;
  int64_t srcRows, srcCols, srcPhysicalRows, srcPhysicalCols;
  int64_t pack, dstValidCols, groups;
};

static LogicalResult mxRequireGroupedShape(TQuantMxOp op, const TQuantMxA5 &s,
                                           StringRef name, Type type,
                                           bool allowLegacy) {
  auto valid = getValidShapeVec(type);
  SmallVector<int64_t, 2> canonical = {s.isDn ? s.srcRows / 32 : s.srcRows,
                                       s.isDn ? s.srcCols : s.srcCols / 32};
  if (llvm::equal(valid, canonical)) {
    return success();
  }
  if (!s.isDn && allowLegacy && valid[0] == 1 && valid[1] == s.groups) {
    return success();
  }
  return op.emitOpError()
         << "expects " << name << " valid_shape to match "
         << (s.isDn ? "canonical [M/32, N] for grpAxis=axis0"
                    : "canonical [M, N/32] or legacy flat [1, M*N/32] for grpAxis=axis1");
}

static FailureOr<TQuantMxA5> buildTQuantMxA5State(TQuantMxOp op) {
  TQuantMxA5 s;
  s.srcTy = op.getSrc().getType();
  s.dstTy = op.getDst().getType();
  s.expTy = op.getExp().getType();
  s.maxTy = op.getMax().getType();
  s.scalingTy = op.getScaling().getType();
  s.srcElem = getElemTy(s.srcTy);
  auto srcValid = getValidShapeVec(s.srcTy);
  s.dstValid = getValidShapeVec(s.dstTy);
  s.expValid = getValidShapeVec(s.expTy);
  auto srcPhysical = getShapeVec(s.srcTy);
  s.dstPhysical = getShapeVec(s.dstTy);
  s.expPhysical = getShapeVec(s.expTy);
  s.maxPhysical = getShapeVec(s.maxTy);
  s.scalingPhysical = getShapeVec(s.scalingTy);
  s.isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  s.srcRows = srcValid[0];
  s.srcCols = srcValid[1];
  s.srcPhysicalRows = srcPhysical[0];
  s.srcPhysicalCols = srcPhysical[1];
  if (s.srcRows <= 0 || s.srcCols <= 0 || s.srcPhysicalRows < s.srcRows ||
      s.srcPhysicalCols < s.srcCols) {
    return op.emitOpError("expects positive source valid shape within physical shape");
  }
  if ((s.isDn ? s.srcRows : s.srcCols) % 32 != 0) {
    return op.emitOpError() << "expects src valid_shape[" << (s.isDn ? 0 : 1)
                            << "] to be a multiple of 32 when grpAxis is "
                            << (s.isDn ? "axis0" : "axis1");
  }
  auto groupsOr = mxCheckedMul(s.srcRows, s.srcCols / 32);
  if (!groupsOr) {
    return op.emitOpError("cannot compute MX quantization group count without overflow");
  }
  s.groups = *groupsOr;
  s.isMxFp4 = op.getQuantType() == mlir::pto::QuantType::MXFP4_E2M1;
  s.pack = s.isMxFp4 ? 2 : 1;
  s.dstValidCols = s.isMxFp4 ? s.srcCols / 2 : s.srcCols;
  return s;
}

static LogicalResult verifyTQuantMxGrouping(TQuantMxOp op, const TQuantMxA5 &s) {
  if (op.getExpZz()) {
    auto expZzElements =
        mxCheckedMul(getValidShapeVec(op.getExpZz().getType())[0],
                     getValidShapeVec(op.getExpZz().getType())[1]);
    if (!expZzElements || *expZzElements != s.groups) {
      return op.emitOpError("expects exp_zz valid element count to equal MX group count");
    }
  }
  if (failed(mxRequireGroupedShape(op, s, "max", s.maxTy, /*allowLegacy=*/true)) ||
      failed(mxRequireGroupedShape(op, s, "scaling", s.scalingTy,
                                   /*allowLegacy=*/true))) {
    return failure();
  }
  if (!op.getInterleave()) {
    return mxRequireGroupedShape(op, s, "exp", s.expTy, /*allowLegacy=*/true);
  }
  if (s.srcRows % 64 != 0) {
    return op.emitOpError("expects src valid rows to be a multiple of 64 when interleave is true");
  }
  if (s.srcPhysicalRows % 64 != 0) {
    return op.emitOpError("expects src physical rows to be a multiple of 64 when interleave is true");
  }
  auto doubledValidCols = mxCheckedMul(s.srcCols, 2);
  if (!doubledValidCols) {
    return op.emitOpError("cannot compute interleaved exp valid shape without overflow");
  }
  if (s.expValid[0] != s.srcRows / 64 || s.expValid[1] != *doubledValidCols) {
    return op.emitOpError("expects exp valid_shape to match [M/64, 2N] for grpAxis=axis0 with interleave=true");
  }
  return success();
}

static LogicalResult verifyTQuantMxDstShape(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.isMxFp4 && s.srcPhysicalCols % 2 != 0) {
    return op.emitOpError("expects MXFP4 src physical cols to be even for packed destination addressing");
  }
  if (s.dstValid[0] != s.srcRows || s.dstValid[1] != s.dstValidCols) {
    return op.emitOpError() << "expects dst valid_shape to be [" << s.srcRows
                            << ", " << s.dstValidCols << "] for MX quantization";
  }
  if (s.dstPhysical[0] < s.srcRows) {
    return op.emitOpError("expects dst physical rows to cover src valid rows");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0Dst(TQuantMxOp op,
                                            const TQuantMxA5 &s) {
  if (s.isMxFp4) {
    if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
      return op.emitOpError("expects MXFP4 axis0 dst physical cols to equal src physical cols / 2");
    }
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
    if ((s.srcPhysicalCols / s.pack) % 32 != 0 && s.srcElem.isF16()) {
      return op.emitOpError("does not support FP16 MXFP4 axis0 when packed source stride is not a multiple of 32 bytes");
    }
  } else {
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.dstPhysical[1]);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0Aux(TQuantMxOp op,
                                            const TQuantMxA5 &s) {
  if (s.maxPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects max physical cols to equal src physical cols for grpAxis=axis0");
  }
  if (s.scalingPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects scaling physical cols to equal src physical cols for grpAxis=axis0");
  }
  auto auxRequired = mxCheckedMul(s.srcRows / 32, s.srcPhysicalCols);
  if (!auxRequired || failed(mxRequireCapacity(op, "max", s.maxTy, *auxRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, *auxRequired))) {
    return failure();
  }
  if (!op.getInterleave()) {
    if (s.expPhysical[1] != s.srcPhysicalCols) {
      return op.emitOpError("expects exp physical cols to equal src physical cols for grpAxis=axis0");
    }
    if (failed(mxRequireCapacity(op, "exp", s.expTy, *auxRequired))) {
      return failure();
    }
  } else {
    auto doubledPhysicalCols = mxCheckedMul(s.srcPhysicalCols, 2);
    auto alignedPhysicalCols =
        doubledPhysicalCols ? mxAlignTo(*doubledPhysicalCols, 32) : std::nullopt;
    if (!alignedPhysicalCols) {
      return op.emitOpError("cannot compute interleaved exp physical cols without overflow");
    }
    if (s.expPhysical[0] != s.srcPhysicalRows / 64) {
      return op.emitOpError("expects interleaved exp physical rows to be src physical rows / 64");
    }
    if (s.expPhysical[1] != *alignedPhysicalCols) {
      return op.emitOpError("expects interleaved exp physical cols to be align32(2 * src physical cols)");
    }
  }
  if (s.srcElem.isF32() && op.getInterleave()) {
    return op.emitOpError("does not support FP32 interleave with the pinned pto-isa revision");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0(TQuantMxOp op,
                                         const TQuantMxA5 &s) {
  if (failed(verifyTQuantMxAxis0Dst(op, s)))
    return failure();
  return verifyTQuantMxAxis0Aux(op, s);
}

static LogicalResult verifyTQuantMxAxis1Flat(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.srcPhysicalCols != s.srcCols) {
    return op.emitOpError("expects axis1 flat exp to use a tight source with physical cols equal to valid cols");
  }
  if (s.expValid[0] != 1 || s.expValid[1] != s.groups) {
    return op.emitOpError("expects axis1 flat exp valid_shape to match legacy flat [1, M*N/32]");
  }
  if (failed(mxRequireCapacity(op, "exp", s.expTy, s.groups))) {
    return failure();
  }
  auto srcElems = mxCheckedMul(s.srcPhysicalRows, s.srcPhysicalCols);
  auto validElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
  bool unroll = srcElems && validElems && *srcElems > 1024 &&
                *srcElems % 256 == 0 && *validElems % 256 == 0;
  if (s.srcElem.isF32()) {
    auto scaleGroups =
        unroll ? mxCheckedMul(s.groups, 2) : std::optional<int64_t>(s.groups);
    auto aligned = scaleGroups ? mxAlignTo(*scaleGroups, 64) : std::nullopt;
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 4) : std::nullopt;
    StringRef scalingName = unroll ? "axis1 flat unrolled f32 scaling"
                                   : "axis1 flat f32 scaling";
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, scalingName, s.scalingTy,
                                      *requiredBytes))) {
      return failure();
    }
  } else if (op.getQuantScaleAlg() == pto::QuantScaleAlg::OCP) {
    auto aligned = mxAlignTo(s.groups, 128);
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 2) : std::nullopt;
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, "axis1 flat B16 OCP scaling",
                                      s.scalingTy, *requiredBytes))) {
      return failure();
    }
  } else {
    return op.emitOpError("does not support axis1 flat B16 NV quantization with the pinned pto-isa revisions");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1Canonical(TQuantMxOp op,
                                                  const TQuantMxA5 &s) {
  if (s.expValid[0] == 1) {
    return op.emitOpError("expects legacy flat exp to use physical rows == 1");
  }
  if (s.srcElem.isF16() || s.srcElem.isBF16()) {
    return op.emitOpError("does not support axis1 canonical 2D B16 quantization with the pinned pto-isa revision");
  }
  SmallVector<int64_t, 2> canonicalShape = {s.srcRows, s.srcCols / 32};
  if (!llvm::equal(s.expValid, canonicalShape)) {
    return op.emitOpError("expects exp valid_shape to match canonical [M, N/32] for grpAxis=axis1");
  }
  if (s.expPhysical[0] < s.srcRows || s.expPhysical[1] < s.srcCols / 32) {
    return op.emitOpError("expects axis1 canonical exp physical shape to cover [M, N/32]");
  }
  auto expPrefix = mxCheckedMul(s.srcRows - 1, s.expPhysical[1]);
  auto expRequired =
      expPrefix ? mxCheckedAdd(*expPrefix, s.srcCols / 32) : std::nullopt;
  if (!expRequired || failed(mxRequireCapacity(op, "exp", s.expTy, *expRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, s.groups))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
    if (s.isMxFp4) {
      return op.emitOpError("expects MXFP4 axis1 dst physical cols to equal src physical cols / 2");
    }
    return op.emitOpError("expects MXFP8 axis1 dst physical cols to equal src physical cols");
  }
  auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
  auto dstRequired =
      dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
  if (!dstRequired || failed(mxRequireCapacity(op, "dst", s.dstTy, *dstRequired))) {
    return failure();
  }
  if (failed(mxRequireCompact(op, "max", s.maxTy)) ||
      failed(mxRequireCompact(op, "scaling", s.scalingTy)) ||
      failed(mxRequireCapacity(op, "max", s.maxTy, s.groups))) {
    return failure();
  }
  const bool flat = s.expPhysical[0] == 1;
  if (flat) {
    return verifyTQuantMxAxis1Flat(op, s);
  }
  return verifyTQuantMxAxis1Canonical(op, s);
}

static LogicalResult verifyTQuantMxSrcPadding(TQuantMxOp op,
                                              const TQuantMxA5 &s) {
  if ((s.srcElem.isF16() || s.srcElem.isBF16()) &&
      s.srcCols < s.srcPhysicalCols && 128 % s.srcPhysicalCols == 0) {
    auto validPhysicalElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
    if (!validPhysicalElems) {
      return op.emitOpError("cannot compute B16 source padding extent without overflow");
    }
    if (*validPhysicalElems % 128 != 0) {
      return op.emitOpError("does not support padded B16 source whose VL-aligned padding store has an incomplete final VL");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxA5(TQuantMxOp op) {
  if (failed(verifyTQuantMxTilesAndForm(op)) ||
      failed(verifyTQuantMxElemTypes(op)) || failed(verifyTQuantMxShapes(op))) {
    return failure();
  }
  auto stateOr = buildTQuantMxA5State(op);
  if (failed(stateOr)) {
    return failure();
  }
  const TQuantMxA5 &s = *stateOr;
  if (failed(verifyTQuantMxGrouping(op, s)) ||
      failed(verifyTQuantMxDstShape(op, s))) {
    return failure();
  }
  if (s.isDn) {
    if (failed(verifyTQuantMxAxis0(op, s))) {
      return failure();
    }
  } else {
    if (failed(verifyTQuantMxAxis1(op, s))) {
      return failure();
    }
  }
  return verifyTQuantMxSrcPadding(op, s);
}

mlir::LogicalResult mlir::pto::TQuantMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tquant.mx is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantMxA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16)) {
      return emitOpError()
             << "expects src element type i8 or i16";
    }
    if (!getElemTy(getDst().getType()).isF32()) {
      return emitOpError() << "expects dst element type f32";
    }
    if (!getElemTy(getScale().getType()).isF32()) {
      return emitOpError() << "expects scale element type f32";
    }
    if (!getElemTy(getOffset().getType()).isF32()) {
      return emitOpError() << "expects offset element type f32";
    }
    return success();
  };

  if (failed(verifyStructural())) {
    return failure();
  }

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst"))) {
      return failure();
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (failed(verifyF16F32VecUnary(getOperation(), getSrc().getType(),
                                  getDst().getType())))
    return failure();
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst()) {
    return emitOpError("expects A3 trecip src and dst to use different storage");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << errorMessage;
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRemNoTmp(TRemOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trem element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemTmpA2A3(TRemOp op, Type tmpTy, Type elem) {
  Type dstTy = op.getDst().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 2) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 2");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(dstTy);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trem dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tmpTy, static_cast<uint64_t>(2) * static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmpA5(TRemOp op, Type tmpTy, Type elem) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmp(TRemOp op, Type elem) {
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemTmpA2A3(op, tmpTy, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemTmpA5(op, tmpTy, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src0, src1, and dst to use row-major layout");
  }

  Type elem = getElemTy(src0Ty);
  if (!getTmp()) {
    return verifyTRemNoTmp(*this, elem);
  }
  return verifyTRemTmp(*this, elem);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

static LogicalResult verifyTRemSNoTmp(TRemSOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trems element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemSTmpA2A3(TRemSOp op, Type tt, Type elem) {
  Type td = op.getDst().getType();
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (getElemTy(tt) != getElemTy(td)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(td);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trems dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmpA5(TRemSOp op, Type tt, Type elem) {
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmp(TRemSOp op, Type elem) {
  Type tt = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tt);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemSTmpA2A3(op, tt, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemSTmpA5(op, tt, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyRowMajorScalarTileCommon(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy) {
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    op->emitOpError("expects scalar type to match the tile element type");
    return failure();
  }
  return elem;
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  FailureOr<Type> elem = verifyRowMajorScalarTileCommon(
      getOperation(), getSrc().getType(), getDst().getType(),
      getScalar().getType());
  if (failed(elem))
    return failure();
  if (!getTmp()) {
    return verifyTRemSNoTmp(*this, *elem);
  }
  return verifyTRemSTmp(*this, *elem);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src and dst to use row-major layout");
  }

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpShape(Operation *op, Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufSameElemType(op, tmpTy, dstTy, "tmp", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op->emitOpError("expects tmp to use row-major layout");
  }
  return verifyTileBufSameValidShape(op, tmpTy, dstTy, "tmp", "dst");
}

static LogicalResult verifyTPowElemType(TPowOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpow element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpow element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpow element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpCommon(Operation *op, Value tmp, Type dstTy,
                                         bool isIntElem,
                                         StringRef integerError,
                                         StringRef staticShapeError) {
  if (isIntElem && tmp) {
    return op->emitOpError() << integerError;
  }
  if (tmp) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (failed(verifyTPowTmpShape(op, tmpTy, dstTy))) {
      return failure();
    }
    if (getTargetArch(op) != PTOArch::A5) {
      auto requiredBytes = getStaticByteSize(dstTy);
      if (!requiredBytes) {
        return op->emitOpError() << staticShapeError;
      }
      if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult verifyTPowTmp(TPowOp op, Type dstTy, bool isIntElem) {
  return verifyTPowTmpCommon(
      op.getOperation(), op.getTmp(), dstTy, isIntElem,
      "does not accept tmp when element type is integer (the integer pow "
      "lowering uses the 3-operand form TPOW(dst, base, exp))",
      "expects A2/A3 tpow dst shape to be static when tmp is provided");
}

mlir::LogicalResult mlir::pto::TPowOp::verify() {
  Type baseTy = getBase().getType();
  Type expTy = getExp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, baseTy, "base")) ||
      failed(verifyTileBufCommon(*this, expTy, "exp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameElemType(*this, baseTy, dstTy, "base", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, dstTy, "base", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(baseTy) || !isRowMajorTileBuf(expTy) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects base, exp, and dst to use row-major layout");
  }

  Type elem = getElemTy(baseTy);
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  if (failed(verifyTPowElemType(*this, elem, isIntElem))) {
    return failure();
  }
  return verifyTPowTmp(*this, dstTy, isIntElem);
}
