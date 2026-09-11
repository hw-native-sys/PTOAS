// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
