// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyTPowSElemType(TPowSOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpows element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpows element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpows element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowSTmp(TPowSOp op, Type dstTy, bool isIntElem) {
  return verifyTPowTmpCommon(
      op.getOperation(), op.getTmp(), dstTy, isIntElem,
      "does not accept tmp when element type is integer (the integer pows "
      "lowering uses the 3-operand form TPOWS(dst, src, scalar))",
      "expects A2/A3 tpows dst shape to be static when tmp is provided");
}

mlir::LogicalResult mlir::pto::TPowSOp::verify() {
  Type dstTy = getDst().getType();
  FailureOr<Type> elem = verifyRowMajorScalarTileCommon(
      getOperation(), getSrc().getType(), dstTy, getScalar().getType());
  if (failed(elem))
    return failure();

  // Same dtype matrix as TPowOp; see comment in TPowOp::verify.
  bool isIntElem = elem->isInteger(32) || elem->isInteger(16) ||
                   elem->isInteger(8);
  if (failed(verifyTPowSElemType(*this, *elem, isIntElem))) {
    return failure();
  }
  return verifyTPowSTmp(*this, dstTy, isIntElem);
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic) {
      return std::nullopt;
    }
    if (d < 0) {
      return std::nullopt;
    }
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy) {
    return std::nullopt;
  }
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16()) {
      return 2;
    }
    if (ft.isF32()) {
        return mlir::pto::kValue4;
    }
    if (ft.isF64()) {
        return mlir::pto::kValue8;
    }
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0) {
      return std::nullopt;
    }
    return std::max<int64_t>(1, bits / mlir::pto::kValue8);
  }
  return std::nullopt;
}

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value)) {
    return false;
  }

  if (isa<AllocTileOp, DeclareTileOp>(value.getDefiningOp())) {
    return true;
  }

  if (auto bitcast = value.getDefiningOp<BitcastOp>()) {
    return isLocallyBoundTileSource(bitcast.getSrc());
  }
  if (auto reshape = value.getDefiningOp<TReshapeOp>()) {
    return isLocallyBoundTileSource(reshape.getSrc());
  }

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return cOp.value();
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    return cInt.value();
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      return ia.getInt();
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndexLike(castOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndexLike(truncOp.getIn());
  }
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != mlir::pto::kValue2) {
      return emitOpError("expects rank-2 tile_buf source");
  }

  ArrayRef<int64_t> validShape = srcTy.getValidShape();
  if (validShape.size() != mlir::pto::kValue2) {
      return emitOpError("expects source validShape to be rank-2");
  }
  if (!srcTy.hasDynamicValid()) {
    return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");
  }

  shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

  if (!isLocallyBoundTileSource(getSource())) {
    return emitOpError(
        "requires a locally bound tile source; function arguments/results "
        "are unsupported");
  }

  auto checkDim = [&](Value operand, unsigned dimIdx,
                      StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal) {
      return success();
    }

    if (*constVal < 0) {
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    }
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic) {
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    }
    return success();
  };
  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row"))) {
    return failure();
  }
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col"))) {
    return failure();
  }

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != mlir::pto::kValue2) {
      return emitOpError("expects rank-2 tile_buf source");
  }
  if (srcTy.getValidShape().size() != mlir::pto::kValue2) {
      return emitOpError("expects source validShape to be rank-2");
  }
  return success();
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb) {
    return emitOpError("expects src/result to be !pto.tile_buf types");
  }

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst"))) {
    return failure();
  }

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace()) {
    return emitOpError("expects src and dst to use the same loc");
  }

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value()) {
    return emitOpError("failed to get element byte width for src/dst");
  }

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value()) {
    return emitOpError("expects static shapes for treshape");
  }

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value()) {
    return emitOpError("expects src and dst to have the same total byte size");
  }

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed) {
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");
  }

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace()) {
    return emitOpError("expects src/result to have the same memorySpace");
  }

  if (srcTy.getElementType() == dstTy.getElementType()) {
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");
  }

  if (srcTy.getShape() != dstTy.getShape()) {
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");
  }

  if (srcTy.getValidShape() != dstTy.getValidShape()) {
    return emitOpError("expects src/result to have the same validShape");
  }

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg) {
    return emitOpError("expects src/result to have the same tile config");
  }

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value()) {
    return emitOpError("expects static shapes for bitcast");
  }

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value()) {
    return emitOpError("unsupported element type for bitcast");
  }

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes) {
    return emitOpError("bitcast result requires more bytes than source storage");
  }

  return success();
}


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

static ParseResult resolveOptionalTmpBinaryOperands(
    OpAsmParser &parser, OperationState &result,
    OptionalTmpBinaryParseState &state, bool tmpBeforeDst) {
  if (parser.resolveOperand(state.lhs, state.lhsTy, result.operands) ||
      parser.resolveOperand(state.rhs, state.rhsTy, result.operands)) {
    return failure();
  }
  if (tmpBeforeDst && state.hasTmp &&
      parser.resolveOperand(state.tmp, state.tmpTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(state.dst, state.dstTy, result.operands)) {
    return failure();
  }
  if (!tmpBeforeDst && state.hasTmp &&
      parser.resolveOperand(state.tmp, state.tmpTy, result.operands)) {
    return failure();
  }
  return success();
}

static ParseResult parseOptionalTmpBinaryDpsOp(OpAsmParser &parser,
                                               OperationState &result,
                                               bool tmpBeforeDst,
                                               bool addSegmentSizes) {
  OptionalTmpBinaryParseState state;
  if (failed(parseOptionalTmpBinaryInputs(parser, state)) ||
      failed(parseOptionalTmpBinaryOutput(parser, result, state)) ||
      failed(resolveOptionalTmpBinaryOperands(parser, result, state,
                                               tmpBeforeDst))) {
    return failure();
  }
  if (addSegmentSizes) {
    result.addAttribute(
        "operandSegmentSizes",
        parser.getBuilder().getDenseI32ArrayAttr(
            {1, 1, state.hasTmp ? 1 : 0, 1}));
  }
  return success();
}

ParseResult mlir::pto::TPowOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/false,
                                     /*addSegmentSizes=*/false);
}
static void printOptionalTmpBinaryDpsOp(OpAsmPrinter &p, Operation *op,
                                        Value lhs, Value rhs, Value tmp,
                                        Value dst) {
  p << " ins(" << lhs << ", " << rhs;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << lhs.getType() << ", " << rhs.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

void mlir::pto::TPowOp::print(OpAsmPrinter &p) {
  printOptionalTmpBinaryDpsOp(p, getOperation(), getBase(), getExp(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TPowSOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/false,
                                     /*addSegmentSizes=*/false);
}

void mlir::pto::TPowSOp::print(OpAsmPrinter &p) {
  printOptionalTmpBinaryDpsOp(p, getOperation(), getSrc(), getScalar(),
                              getTmp(), getDst());
}

static ParseResult parseTRowExpandBinaryLikeOp(OpAsmParser &parser,
                                               OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/true,
                                     /*addSegmentSizes=*/true);
}

static void printTRowExpandBinaryLikeOp(OpAsmPrinter &p, Operation *op, Value src0,
                                        Value src1, Value tmp, Value dst) {
  printOptionalTmpBinaryDpsOp(p, op, src0, src1, tmp, dst);
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

ParseResult mlir::pto::TRowExpandSubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandSubOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandAddOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandAddOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandExpdifOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandExpdifOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMaxOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMaxOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMinOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMinOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

static FailureOr<Type> verifyTRowExpandBinaryCore(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty)) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

enum class TRowExpandBinaryMode {
  Unknown,
  Mode1ColMajorScalar,
  Mode2RowMajorBlock,
};

static bool validShapesCompatibleForTRowExpand(ArrayRef<int64_t> lhs,
                                               ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r) {
      return false;
    }
  }
  return true;
}

static TRowExpandBinaryMode classifyTRowExpandBinaryMode(Type src0Ty,
                                                         Type src1Ty,
                                                         Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2 ||
      dstValid.size() != mlir::pto::kValue2) {
      return TRowExpandBinaryMode::Unknown;
  }

  Type expandedTy;
  ArrayRef<int64_t> expandedValid;
  if (validShapesCompatibleForTRowExpand(src0Valid, dstValid)) {
    expandedTy = src1Ty;
    expandedValid = src1Valid;
  } else if (validShapesCompatibleForTRowExpand(src1Valid, dstValid)) {
    expandedTy = src0Ty;
    expandedValid = src0Valid;
  } else {
    return TRowExpandBinaryMode::Unknown;
  }

  int64_t expandedCols = expandedValid[1];
  if (isColMajorTileBuf(expandedTy) &&
      (expandedCols == ShapedType::kDynamic || expandedCols == 1)) {
    return TRowExpandBinaryMode::Mode1ColMajorScalar;
  }

  std::optional<int64_t> elemBytes = getElemBytes(getElemTy(dstTy));
  if (!elemBytes || *elemBytes == 0) {
    return TRowExpandBinaryMode::Unknown;
  }
  int64_t expectedMode2Cols = 32 / *elemBytes;
  if (isRowMajorTileBuf(expandedTy) &&
      (expandedCols == ShapedType::kDynamic ||
       expandedCols == expectedMode2Cols)) {
    return TRowExpandBinaryMode::Mode2RowMajorBlock;
  }

  return TRowExpandBinaryMode::Unknown;
}

static int64_t getTRowExpandTmpMinBytes(int64_t dstValidRows) {
  if (dstValidRows == ShapedType::kDynamic) {
      return mlir::pto::kValue8192;
  }
  if (dstValidRows < 0) {
      return mlir::pto::kValue8192;
  }
  if (dstValidRows < mlir::pto::kValue256) {
      return ceilDivInt64(dstValidRows, mlir::pto::kValue8) * mlir::pto::kValue256;
  }
  return mlir::pto::kValue30 * mlir::pto::kValue256;
}

static std::optional<int64_t> getStaticTileCapacityBytes(Type ty) {
  auto numElems = getStaticNumElements(getShapeVec(ty));
  auto elemBytes = getElemBytes(getElemTy(ty));
  if (!numElems || !elemBytes) {
    return std::nullopt;
  }
  return *numElems * *elemBytes;
}

static LogicalResult verifyTRowExpandImplicitTmpContract(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch) {
  if (!hasTmp || targetArch == PTOArch::A5) {
    return success();
  }

  if (classifyTRowExpandBinaryMode(src0Ty, src1Ty, dstTy) !=
      TRowExpandBinaryMode::Mode1ColMajorScalar) {
    return op->emitOpError(
        "expects A2/A3 tmp-form trowexpand to use mode 1 "
        "(ColMajor per-row scalar expanded operand)");
  }

  if (failed(verifyVecTileStorage(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects tmp and dst to have the same element type");
  }

  auto dstValid = getValidShapeVec(dstTy);
  if (dstValid.size() != mlir::pto::kValue2) {
      return op->emitOpError("expects dst to have rank-2 valid_shape");
  }
  int64_t minBytes = getTRowExpandTmpMinBytes(dstValid[0]);
  std::optional<int64_t> tmpBytes = getStaticTileCapacityBytes(tmpTy);
  if (!tmpBytes) {
    return op->emitOpError(
        "expects A2/A3 trowexpand tmp capacity to be statically known");
  }
  if (*tmpBytes < minBytes) {
    return op->emitOpError()
           << "expects A2/A3 trowexpand tmp capacity to be at least "
           << minBytes << " bytes, but got " << *tmpBytes << " bytes";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported =
        elem.isF16() || elem.isF32() ||
        (targetArch == PTOArch::A5 &&
         (elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError("expects element type to be f16 or f32");
    }
    if (getPrecisionType() == pto::DivPrecision::HighPrecision && !getTmp()) {
      return emitOpError("expects tmp when precisionType is high_precision");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRowExpandMulSub(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Value tmp,
    PTOArch targetArch, StringRef opName) {
  Type tmpTy = tmp ? tmp.getType() : Type{};
  auto elem = verifyTRowExpandBinaryCore(op, src0Ty, src1Ty, dstTy, tmpTy,
                                        static_cast<bool>(tmp));
  if (failed(elem))
    return failure();
  bool supported = elem->isF16() || elem->isF32() || elem->isInteger(16) ||
                   elem->isInteger(32) ||
                   (targetArch == PTOArch::A5 && elem->isInteger(8));
  if (!supported)
    return op->emitOpError()
           << "expects " << (targetArch == PTOArch::A5 ? "A5 " : "A2/A3 ")
           << opName
           << (targetArch == PTOArch::A5
                   ? " element type to be i8/i16/i32/f16/f32"
                   : " element type to be i16/i32/f16/f32");
  return verifyTRowExpandImplicitTmpContract(
      op, src0Ty, src1Ty, dstTy, tmpTy, static_cast<bool>(tmp), targetArch);
}

mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch arch) {
    return verifyTRowExpandMulSub(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType(), getTmp(), arch, "trowexpandmul");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch arch) {
    return verifyTRowExpandMulSub(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType(), getTmp(), arch, "trowexpandsub");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTRowExpandAddCore(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
      op, src0Ty, src1Ty, dstTy, op.getTmp() ? op.getTmp().getType() : Type{},
      static_cast<bool>(op.getTmp()));
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty)) {
    return op.emitOpError("expects src0 to use row-major layout");
  }
  Type elem = *elemOr;
  bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                   elem.isInteger(32) ||
                   (targetArch == PTOArch::A5 && elem.isInteger(8));
  if (!supported) {
    if (targetArch == PTOArch::A5) {
      return op.emitOpError(
          "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32");
    }
    return op.emitOpError(
        "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32");
  }
  return elem;
}

static LogicalResult verifyTRowExpandAddSrc1(TRowExpandAddOp op, Type elem,
                                             PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src1Valid.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects src1 and dst to have rank-2 valid_shape");
  }
  if (src1Valid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      src1Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src1 valid_shape[0] to equal dst valid_shape[0]");
  }
  bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
  int64_t expectedCol = elem.isInteger(8)
                            ? 32
                            : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t src1Col = src1Valid[1];
  if (src1IsRowMajor) {
    if (src1Col != ShapedType::kDynamic && src1Col != expectedCol) {
      return op.emitOpError("expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    }
  } else {
    if (src1Col != ShapedType::kDynamic && src1Col != 1) {
      return op.emitOpError("expects non-row-major src1 valid_shape[1] to be 1");
    }
  }
  if (failed(verifyTRowExpandImplicitTmpContract(
          op.getOperation(), src0Ty, src1Ty, dstTy,
          op.getTmp() ? op.getTmp().getType() : Type{},
          static_cast<bool>(op.getTmp()), targetArch))) {
    return failure();
  }
  return mlir::success();
}

static LogicalResult verifyTRowExpandAddByArch(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  FailureOr<Type> elemOr = verifyTRowExpandAddCore(op, targetArch);
  if (failed(elemOr)) {
    return failure();
  }
  return verifyTRowExpandAddSrc1(op, *elemOr, targetArch);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A5);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTRowExpandReduceTypes(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch, StringRef opName, bool allowIntegerTypes) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp"))))
    return failure();
  if (hasTmp && getElemTy(tmpTy) != getElemTy(dstTy)) {
    op->emitOpError("expects tmp and dst to have the same element type");
    return failure();
  }
  Type elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem) {
    op->emitOpError(
        "expects src0, src1, and dst to have the same element type");
    return failure();
  }
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (supported)
    return elem;
  if (!allowIntegerTypes)
    op->emitOpError() << "expects " << opName
                      << " element type to be f16 or f32";
  else if (targetArch == PTOArch::A5)
    op->emitOpError() << "expects A5 " << opName
                      << " element type to be i8/i16/i32/f16/f32";
  else
    op->emitOpError() << "expects A2/A3 " << opName
                      << " element type to be i16/i32/f16/f32";
  return failure();
}

static bool rowExpandValidShapesMatch(ArrayRef<int64_t> lhs,
                                      ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    auto [left, right] = pair;
    return left == ShapedType::kDynamic || right == ShapedType::kDynamic ||
           left == right;
  });
}

static LogicalResult verifyTRowExpandBroadcastOperand(
    Operation *op, Type operandTy, ArrayRef<int64_t> operandValid,
    ArrayRef<int64_t> dstValid, Type elem, StringRef name,
    bool requireNonRowMajor) {
  if (!hasCompatibleKnownExtent(operandValid[0], dstValid[0]))
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to equal dst valid_shape[0]";
  bool rowMajor = isRowMajorTileBuf(operandTy);
  if (requireNonRowMajor && rowMajor)
    return op->emitOpError() << "expects " << name
                             << " to use a non-row-major layout when tmp is present";
  int64_t expectedCol = elem.isInteger(8)
                            ? 32
                            : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t column = operandValid[1];
  if (rowMajor && column != ShapedType::kDynamic && column != expectedCol)
    return op->emitOpError() << "expects row-major " << name
                             << " valid_shape[1] to be 32/sizeof(dtype)";
  if (!rowMajor && column != ShapedType::kDynamic && column != 1)
    return op->emitOpError() << "expects non-row-major " << name
                             << " valid_shape[1] to be 1";
  return success();
}

static LogicalResult verifyTRowExpandFullAndBroadcast(
    Operation *op, Type fullTy, ArrayRef<int64_t> fullValid,
    StringRef fullName, Type broadcastTy, ArrayRef<int64_t> broadcastValid,
    StringRef broadcastName, ArrayRef<int64_t> dstValid, Type elem,
    bool requireNonRowMajor) {
  if (!isRowMajorTileBuf(fullTy))
    return op->emitOpError() << "expects " << fullName
                             << " to use row-major layout when it matches dst";
  if (!rowExpandValidShapesMatch(fullValid, dstValid))
    return op->emitOpError() << "expects " << fullName
                             << " valid_shape to equal dst valid_shape";
  return verifyTRowExpandBroadcastOperand(
      op, broadcastTy, broadcastValid, dstValid, elem, broadcastName,
      requireNonRowMajor);
}

static LogicalResult verifyTRowExpandReduceValidBasics(
    Operation *op, ArrayRef<int64_t> src0Valid, ArrayRef<int64_t> src1Valid,
    ArrayRef<int64_t> dstValid) {
    if (src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2 ||
        dstValid.size() != mlir::pto::kValue2)
        return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
        return op->emitOpError("expects dst valid_shape[0] to be non-zero");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
        return op->emitOpError("expects dst valid_shape[1] to be non-zero");
    return success();
}

static LogicalResult verifyTRowExpandReduceLikeOp(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch, bool enforceTmpContract, StringRef opName,
    bool allowIntegerTypes) {
  auto elemResult = verifyTRowExpandReduceTypes(
      op, src0Ty, src1Ty, dstTy, tmpTy, hasTmp, targetArch, opName,
      allowIntegerTypes);
  if (failed(elemResult))
    return failure();
  Type elem = *elemResult;

  if (!isRowMajorTileBuf(dstTy)) {
    return op->emitOpError("expects dst to use row-major layout");
  }

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2 ||
      dstValid.size() != mlir::pto::kValue2)
      return verifyTRowExpandReduceValidBasics(op, src0Valid, src1Valid, dstValid);

  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. Element
  // type/layout were already checked above; the op writes no elements, so accept
  // and skip the non-empty broadcast/width constraints. One-sided empties still
  // fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }

  if (failed(verifyTRowExpandReduceValidBasics(op, src0Valid, src1Valid,
                                               dstValid)))
    return failure();

  const bool src0MatchesDst = rowExpandValidShapesMatch(src0Valid, dstValid);
  const bool src1MatchesDst = rowExpandValidShapesMatch(src1Valid, dstValid);

  // (A5 tmp-form invariant is checked earlier, before the empty-marker accept.)

  auto verifyTmpContract = [&]() {
    return enforceTmpContract
               ? verifyTRowExpandImplicitTmpContract(
                     op, src0Ty, src1Ty, dstTy, tmpTy, hasTmp, targetArch)
               : success();
  };
  bool requireNonRowMajor = hasTmp && targetArch == PTOArch::A3;

  if (src0MatchesDst) {
    if (succeeded(verifyTRowExpandFullAndBroadcast(
            op, src0Ty, src0Valid, "src0", src1Ty, src1Valid, "src1",
            dstValid, elem, requireNonRowMajor)) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }
  if (src1MatchesDst) {
    if (succeeded(verifyTRowExpandFullAndBroadcast(
            op, src1Ty, src1Valid, "src1", src0Ty, src0Valid, "src0",
            dstValid, elem, requireNonRowMajor)) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }

  return op->emitOpError() << "expects one of src0/src1 to match dst valid_shape"
                           << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


struct OptionalTmpUnaryParseState {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;
};

static ParseResult parseOptionalTmpUnarySyntax(
    OpAsmParser &parser, OperationState &result,
    OptionalTmpUnaryParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(state.tmp)) {
      return failure();
    }
    state.hasTmp = true;
  }
  if (parser.parseColonType(state.srcTy)) {
    return failure();
  }
  if (state.hasTmp &&
      (parser.parseComma() || parser.parseType(state.tmpTy))) {
    return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(state.dst) ||
      parser.parseColonType(state.dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

static ParseResult parseOptionalTmpRowReductionOp(OpAsmParser &parser,
                                                  OperationState &result) {
  OptionalTmpUnaryParseState state;
  if (failed(parseOptionalTmpUnarySyntax(parser, result, state))) {
    return failure();
  }
  return resolveOptionalUnaryAndAddSegments(
      parser, result, state.src, state.srcTy, state.hasTmp, state.tmp,
      state.tmpTy, state.dst, state.dstTy);
}

static void printOptionalTmpRowReductionOp(OpAsmPrinter &p, Operation *op,
                                           Value src, Value tmp, Value dst) {
  p << " ins(" << src;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << src.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseFixedDpsInputs(
    OpAsmParser &parser, unsigned minInputs, unsigned maxInputs,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
    SmallVectorImpl<Type> &inputTypes) {
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  do {
    inputs.emplace_back();
    if (parser.parseOperand(inputs.back())) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  if (inputs.size() < minInputs || inputs.size() > maxInputs ||
      parser.parseColon()) {
    return failure();
  }
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if ((i && parser.parseComma()) || parser.parseType(inputTypes.emplace_back())) {
      return failure();
    }
  }
  return parser.parseRParen();
}

static ParseResult parseOptionalTmpFixedDpsOp(
    OpAsmParser &parser, OperationState &result, unsigned minInputs,
    unsigned maxInputs, ArrayRef<int32_t> noTmpSegments,
    ArrayRef<int32_t> withTmpSegments) {
  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SmallVector<Type> inputTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstType;
  if (failed(parseFixedDpsInputs(parser, minInputs, maxInputs, inputs,
                                 inputTypes)) ||
      parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstType) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperands(inputs, inputTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          inputs.size() == minInputs ? noTmpSegments : withTmpSegments));
  return success();
}

static void printOptionalTmpFixedDpsOp(OpAsmPrinter &p, Operation *op,
                                       ArrayRef<Value> inputs, Value dst) {
  p << " ins(";
  llvm::interleaveComma(inputs, p, [&](Value value) { p << value; });
  p << " : ";
  llvm::interleaveComma(inputs, p,
                        [&](Value value) { p << value.getType(); });
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TTransOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
    return parseOptionalTmpFixedDpsOp(parser, result, 1, mlir::pto::kValue2, {1, 0, 1}, {1, 1, 1});
}
void mlir::pto::TTransOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TPReluOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
    return parseOptionalTmpFixedDpsOp(
        parser, result, mlir::pto::kValue2, mlir::pto::kValue3, {1, 1, 0, 1}, {1, 1, 1, 1});
}
void mlir::pto::TPReluOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TRemOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
    return parseOptionalTmpFixedDpsOp(
        parser, result, mlir::pto::kValue2, mlir::pto::kValue3, {1, 1, 0, 1}, {1, 1, 1, 1});
}
void mlir::pto::TRemOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TRemSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
    return parseOptionalTmpFixedDpsOp(
        parser, result, mlir::pto::kValue2, mlir::pto::kValue3, {1, 1, 0, 1}, {1, 1, 1, 1});
}
void mlir::pto::TRemSOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc(), getScalar()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TSelOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
    return parseOptionalTmpFixedDpsOp(
        parser, result, mlir::pto::kValue3, mlir::pto::kValue4, {1, 1, 1, 0, 1}, {1, 1, 1, 1, 1});
}
void mlir::pto::TSelOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getMask(), getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TSelSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
    return parseOptionalTmpFixedDpsOp(
        parser, result, mlir::pto::kValue3, mlir::pto::kValue4, {1, 1, 0, 1, 1}, {1, 1, 1, 1, 1});
}
void mlir::pto::TSelSOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getMask(), getSrc()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  inputs.push_back(getScalar());
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TColArgMaxOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TColArgMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TColArgMinOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TColArgMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowMaxOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowArgMaxOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowArgMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowMinOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowArgMinOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowArgMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowSumOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowSumOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowProdOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowProdOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

static LogicalResult verifyTRowReductionOp(Operation *op, Type srcTy,
                                           Value tmp, Type dstTy) {
  if (!tmp)
    return verifyTRowReductionNoTmpCommon(
        op, srcTy, dstTy, "expects element type to be i16/i32/f16/f32");
  return verifyTRowReductionWithTmpCommon(
      op, srcTy, tmp.getType(), dstTy,
      "expects element type to be i16/i32/f16/f32");
}

static LogicalResult verifyTRowArgReductionOp(Operation *op, Type srcTy,
                                              Value tmp, Type dstTy) {
  if (!tmp)
    return verifyTRowArgReductionNoTmp(op, srcTy, dstTy);
  auto verifyA2A3 = [&]() {
    return verifyTRowArgReductionOpA2A3(op, srcTy, tmp.getType(), dstTy);
  };
  auto verifyA5 = [&]() {
    return verifyTRowArgReductionOpA5(op, srcTy, tmp.getType(), dstTy);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [&]() {
    return verifyTRowReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                 getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  return verifyTRowArgReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                  getDst().getType());
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [&]() {
    return verifyTRowReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                 getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  return verifyTRowArgReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                  getDst().getType());
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [&]() {
    return verifyTRowReductionOp(getOperation(), getSrc().getType(), getTmp(),
                                 getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}
