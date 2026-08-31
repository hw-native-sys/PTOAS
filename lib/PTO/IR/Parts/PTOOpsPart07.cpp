// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return emitOpError("expects src/result to have the same memorySpace");

  if (srcTy.getElementType() == dstTy.getElementType())
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");

  if (srcTy.getValidShape() != dstTy.getValidShape())
    return emitOpError("expects src/result to have the same validShape");

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg)
    return emitOpError("expects src/result to have the same tile config");

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value())
    return emitOpError("expects static shapes for bitcast");

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value())
    return emitOpError("unsupported element type for bitcast");

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes)
    return emitOpError("bitcast result requires more bytes than source storage");

  return success();
}


mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src to be in the vec address space");
    if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
      if (srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return emitOpError("expects src to use the none_box slayout");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                                /*allowInt8=*/true))
      return emitOpError("expects trowexpand element type to be supported");
    auto srcValid = getValidShapeVec(getSrc());
    auto dstValid = getValidShapeVec(getDst());
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
    // writes no elements; accept and skip the non-empty constraints. One-sided
    // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
    if (dstValid[0] == 0 && dstValid[1] == 0)
      return success();
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return emitOpError("expects src valid_shape[1] to be non-zero");
    if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
      return emitOpError("expects dst valid_shape[0] to be non-zero");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
      return emitOpError("expects dst valid_shape[1] to be non-zero");
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, dstTy, idxTy, tmpTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(idx))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(tmp))
        return failure();
      hasTmp = true;
    }
  } else {
    return failure();
  }
  if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(idxTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands))
    return failure();
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
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

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
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

// TPOW assembly format (mirrors TRsqrt's optional-tmp style):
//   pto.tpow ins(%base, %exp[, %tmp] : !tile, !tile[, !tile])
//            outs(%dst : !tile) [attr-dict]
ParseResult mlir::pto::TPowOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand base, exp, tmp, dst;
  Type baseTy, expTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(base) || parser.parseComma() ||
      parser.parseOperand(exp))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColon())
    return failure();
  if (parser.parseType(baseTy) || parser.parseComma() || parser.parseType(expTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(base, baseTy, result.operands) ||
      parser.resolveOperand(exp, expTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TPowOp::print(OpAsmPrinter &p) {
  p << " ins(" << getBase() << ", " << getExp();
  if (getTmp())
    p << ", " << getTmp();
  p << " : " << getBase().getType() << ", " << getExp().getType();
  if (getTmp())
    p << ", " << getTmp().getType();
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

// TPOWS assembly format:
//   pto.tpows ins(%src, %scalar[, %tmp] : !tile, scalar_t[, !tile])
//             outs(%dst : !tile) [attr-dict]
ParseResult mlir::pto::TPowSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, scalar, tmp, dst;
  Type srcTy, scalarTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(scalar))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColon())
    return failure();
  if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(scalarTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(scalar, scalarTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TPowSOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getScalar();
  if (getTmp())
    p << ", " << getTmp();
  p << " : " << getSrc().getType() << ", " << getScalar().getType();
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

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src0) || parser.parseComma() || parser.parseOperand(src1))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColon())
    return failure();
  if (parser.parseType(src0Ty) || parser.parseComma() || parser.parseType(src1Ty))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands))
    return failure();
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
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
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
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

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    Type elem = *elemOr;
    bool supported =
        elem.isF16() || elem.isF32() ||
        (targetArch == PTOArch::A5 &&
         (elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
      return emitOpError("expects element type to be f16 or f32");
    }
    if (getPrecisionType() == pto::DivPrecision::HighPrecision && !getTmp())
      return emitOpError("expects tmp when precisionType is high_precision");
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpandmul element type to be i8/i16/i32/f16/f32");
      return emitOpError(
          "expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpandsub element type to be i8/i16/i32/f16/f32");
      return emitOpError(
          "expects A2/A3 trowexpandsub element type to be i16/i32/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty))
      return emitOpError("expects src0 to use row-major layout");
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32");
      return emitOpError(
          "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32");
    }
    auto src1Valid = getValidShapeVec(src1Ty);
    auto dstValid = getValidShapeVec(dstTy);
    if (src1Valid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src1 and dst to have rank-2 valid_shape");
    if (src1Valid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        src1Valid[0] != dstValid[0])
      return emitOpError("expects src1 valid_shape[0] to equal dst valid_shape[0]");
    bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
    int64_t expectedCol = elem.isInteger(8)
                              ? 32
                              : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
    int64_t src1Col = src1Valid[1];
    if (src1IsRowMajor) {
      if (src1Col != ShapedType::kDynamic && src1Col != expectedCol)
        return emitOpError("expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    } else {
      if (src1Col != ShapedType::kDynamic && src1Col != 1)
        return emitOpError("expects non-row-major src1 valid_shape[1] to be 1");
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTRowExpandReduceLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (hasTmp) {
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp")))
      return failure();
    if (getElemTy(tmpTy) != getElemTy(dstTy))
      return op->emitOpError() << "expects tmp and dst to have the same element type";
  }

  Type elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem)
    return op->emitOpError("expects src0, src1, and dst to have the same element type");
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (!supported) {
    if (!allowIntegerTypes)
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    if (targetArch == PTOArch::A5)
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (!isRowMajorTileBuf(dstTy))
    return op->emitOpError("expects dst to use row-major layout");

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. Element
  // type/layout were already checked above; the op writes no elements, so accept
  // and skip the non-empty broadcast/width constraints. One-sided empties still
  // fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0)
    return success();

  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
    return op->emitOpError("expects dst valid_shape[0] to be non-zero");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
    return op->emitOpError("expects dst valid_shape[1] to be non-zero");

  auto validShapeMatches = [](ArrayRef<int64_t> lhs,
                              ArrayRef<int64_t> rhs) -> bool {
    if (lhs.size() != rhs.size())
      return false;
    for (auto [l, r] : llvm::zip(lhs, rhs)) {
      if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r)
        return false;
    }
    return true;
  };

  const bool src0MatchesDst = validShapeMatches(src0Valid, dstValid);
  const bool src1MatchesDst = validShapeMatches(src1Valid, dstValid);

  auto checkBroadcastOperand = [&](Type operandTy, ArrayRef<int64_t> operandValid,
                                   StringRef operandName,
                                   bool requireNonRowMajor) -> LogicalResult {
    if (operandValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        operandValid[0] != dstValid[0]) {
      return op->emitOpError() << "expects " << operandName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    }
    int64_t expectedCol = elem.isInteger(8) ? 32 : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
    int64_t operandCol = operandValid[1];
    bool operandIsRowMajor = isRowMajorTileBuf(operandTy);
    if (requireNonRowMajor && operandIsRowMajor) {
      return op->emitOpError() << "expects " << operandName
                               << " to use a non-row-major layout when tmp is present";
    }
    if (operandIsRowMajor) {
      if (operandCol != ShapedType::kDynamic && operandCol != expectedCol) {
        return op->emitOpError()
               << "expects row-major " << operandName
               << " valid_shape[1] to be 32/sizeof(dtype)";
      }
      return success();
    }
    if (operandCol != ShapedType::kDynamic && operandCol != 1) {
      return op->emitOpError() << "expects non-row-major " << operandName
                               << " valid_shape[1] to be 1";
    }
    return success();
  };

  auto checkFullAndBroadcast = [&](Type fullTy, ArrayRef<int64_t> fullValid,
                                   StringRef fullName, Type broadcastTy,
                                   ArrayRef<int64_t> broadcastValid,
                                   StringRef broadcastName) -> LogicalResult {
    if (!isRowMajorTileBuf(fullTy))
      return op->emitOpError() << "expects " << fullName
                               << " to use row-major layout when it matches dst";
    if (fullValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        fullValid[0] != dstValid[0])
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    if (fullValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        fullValid[1] != dstValid[1])
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[1] to equal dst valid_shape[1]";
    return checkBroadcastOperand(broadcastTy, broadcastValid, broadcastName,
                                 /*requireNonRowMajor=*/hasTmp &&
                                     targetArch == PTOArch::A3);
  };

  // (A5 tmp-form invariant is checked earlier, before the empty-marker accept.)

  if (src0MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src0Ty, src0Valid, "src0", src1Ty,
                                        src1Valid, "src1")))
      return success();
  }
  if (src1MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src1Ty, src1Valid, "src1", src0Ty,
                                        src0Valid, "src0")))
      return success();
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
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
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
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
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
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32()))
    return emitOpError("expects element type to be f16 or f32");
  if (getPrecisionType() == pto::RsqrtPrecision::HighPrecision && !getTmp())
    return emitOpError("expects tmp when precisionType is high_precision");
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value())
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    if (tmpElemBytes.value() * tmpNumel.value() < 32)
      return emitOpError("expects tmp to be at least 32 bytes when provided");
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }

  auto isAllowedDataElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
    return false;
  };
  auto isAllowedIndexElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return (it.getWidth() == 16 || it.getWidth() == 32);
    return false;
  };
  auto getMaskScatterTimes = [&](mlir::pto::MaskPatternAttr mp) -> unsigned {
    switch (mp.getValue()) {
    case mlir::pto::MaskPattern::P1111:
      return 1;
    case mlir::pto::MaskPattern::P0101:
    case mlir::pto::MaskPattern::P1010:
      return 2;
    default:
      return 4;
    }
  };

  auto verifyIndexedForm = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type ti = getIndexes().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, ts, "src")) ||
        failed(verifyVecTileStorage(*this, ti, "indexes")) ||
        failed(verifyVecTileStorage(*this, td, "dst")))
      return failure();

    Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
    if (!srcElem || !dstElem || !idxElem)
      return emitOpError("failed to get element type for operands");
    if (srcElem != dstElem)
      return emitOpError("expects src/dst to have the same element type");

    if (!isAllowedDataElem(srcElem))
      return emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
    if (!isAllowedIndexElem(idxElem))
      return emitOpError("expects indexes element type to be i16/i32");

    auto bwData = getPTOStorageElemBitWidth(srcElem);
    auto bwIdx  = getPTOStorageElemBitWidth(idxElem);
    if (bwData != 8 && bwData != 16 && bwData != 32)
      return emitOpError("unexpected src/dst element bitwidth");

    unsigned dataBytes = bwData / 8;
    unsigned idxBytes  = bwIdx / 8;
    unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
    if (idxBytes != expectedIdxBytes)
      return emitOpError("expects indexes element size to match the documented scatter rule");
    return mlir::success();
  };

  auto verifyMaskForm = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileCommon(*this, ts, "src")) ||
        failed(verifyVecTileCommon(*this, td, "dst")))
      return failure();

    auto srcTB = dyn_cast<pto::TileBufType>(ts);
    auto dstTB = dyn_cast<pto::TileBufType>(td);
    if (!srcTB || !dstTB)
      return emitOpError("expects src and dst to be tile_buf types");

    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (!isAllowedDataElem(getElemTy(ts)))
      return emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");

    auto srcValid = getValidShapeVec(ts);
    auto dstValid = getValidShapeVec(td);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");

    auto mp = getMaskPatternAttr();
    if (!mp)
      return emitOpError("expects mask-pattern tscatter to provide maskPattern");
    const unsigned times = getMaskScatterTimes(mp);
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid rows");
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != static_cast<int64_t>(dstValid[1] * times))
      return emitOpError("expects src valid cols to equal dst valid cols times the mask expansion factor");

    if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return emitOpError("expects mask-pattern tscatter to use row_major blayout");
    return mlir::success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (hasMaskPattern)
      return verifyMaskForm();
    return verifyIndexedForm();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (hasMaskPattern)
      return verifyMaskForm();
    return verifyIndexedForm();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type srcElem = getElemTy(t0);
    Type src1Elem = getElemTy(t1);
    Type dstElem = getElemTy(td);
    if (!srcElem || !src1Elem || !dstElem) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (srcElem != src1Elem || srcElem != dstElem) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
        !isRowMajorTileBuf(td)) {
      emitOpError(
          "expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> srcElem = verifyCommon();
    if (failed(srcElem))
      return failure();
    Type elem = *srcElem;
    bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
    if (auto it = dyn_cast<IntegerType>(elem))
      ok = it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> srcElem = verifyCommon();
    if (failed(srcElem))
      return failure();
    Type elem = *srcElem;
    bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
    if (auto it = dyn_cast<IntegerType>(elem))
      ok = it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels:
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type tMask = getMask().getType();
    Type tSrc = getSrc().getType();
    Type tTmp = getTmp().getType();
    Type tDst = getDst().getType();
    if (failed(verifyTileBufCommon(*this, tMask, "mask")) ||
        failed(verifyTileBufCommon(*this, tSrc, "src")) ||
        failed(verifyTileBufCommon(*this, tTmp, "tmp")) ||
        failed(verifyTileBufCommon(*this, tDst, "dst")))
      return failure();
    Type eMask = getElemTy(tMask), eSrc = getElemTy(tSrc);
    Type eTmp = getElemTy(tTmp), eDst = getElemTy(tDst);
    if (!eMask || !eSrc || !eTmp || !eDst) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (eSrc != eDst)
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyTileBufSameValidShape(*this, tSrc, tDst, "src", "dst")))
      return failure();
    return eDst;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type tSrc = getSrc().getType();
    Type tDst = getDst().getType();
    if (!isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst))
      return emitOpError("expects src and dst to use row-major layout");
    Type elem = *elemOr;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem))
      ok = (it.getWidth() == 16 || it.getWidth() == 32);
    if (!ok)
      return emitOpError(
          "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type tMask = getMask().getType();
    Type tSrc = getSrc().getType();
    Type tDst = getDst().getType();
    if (!isRowMajorTileBuf(tMask) || !isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst))
      return emitOpError("expects mask, src, and dst to use row-major layout");
    Type elem = *elemOr;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem))
      ok = (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
    if (!ok)
      return emitOpError(
          "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx")))
    return failure();
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!(srcElem.isF16() || srcElem.isF32()))
    return emitOpError() << "expects src and dst element type to be f16 or f32";

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32)
    return emitOpError() << "expects idx element type to be i32/u32";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem)))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}



mlir::LogicalResult mlir::pto::TStoreFPOp::verify() {
  auto shouldBypassDecoded = [&]() -> bool {
    Value src = getSrc();
    Value fp = getFp();
    return isa<MemRefType>(src.getType()) || isa<MemRefType>(fp.getType()) ||
           src.getDefiningOp<pto::BindTileOp>() ||
           fp.getDefiningOp<pto::BindTileOp>();
  };

  auto verifySrcDtypeAlways = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    auto srcElemTy = getElemTy(srcTy);
    if (!srcElemTy)
      return success();
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    return success();
  };

  if (failed(verifySrcDtypeAlways()))
    return failure();

  auto verifyDstType = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (!isa<MemRefType, pto::PartitionTensorViewType>(dstTy))
      return emitOpError()
             << "expects dst to be a memref or !pto.partition_tensor_view";
    if (auto dstPart = dyn_cast<pto::PartitionTensorViewType>(dstTy)) {
      for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
        if (dim != ShapedType::kDynamic && dim <= 0)
          return emitOpError()
                 << "expects dst shape[" << idx << "] to be positive";
      }
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    if (!isa<pto::TileBufType>(srcTy))
      return emitOpError() << "expects src to be a !pto.tile_buf";
    if (!isa<pto::TileBufType>(fpTy))
      return emitOpError() << "expects fp to be a !pto.tile_buf";
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")))
      return failure();
    if (failed(verifyDstType()))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    auto srcShape = getShapeVec(srcTy);
    if (srcShape.size() != 2)
      return emitOpError() << "expects src to have rank 2";
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095))
      return emitOpError() << "expects src.cols to be in the range [1, 4095]";
    auto srcValid = getValidShapeVec(srcTy);
    if (srcValid.size() != 2)
      return emitOpError() << "expects src to have a rank-2 valid_shape";
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 0 || srcValid[1] > 4095))
      return emitOpError()
             << "expects src.valid_shape[1] to be in the range [0, 4095]";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    if (!isa<pto::TileBufType>(srcTy))
      return emitOpError() << "expects src to be a !pto.tile_buf";
    if (!isa<pto::TileBufType>(fpTy))
      return emitOpError() << "expects fp to be a !pto.tile_buf";
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")))
      return failure();
    if (failed(verifyDstType()))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    return mlir::success();
  };
  if (shouldBypassDecoded())
    return success();
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}


mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size())
    return emitOpError() << "expects all tensors to have the same rank";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  return mlir::success();
}
mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type tmpTy = getTmp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type tmpElem = getElemTy(tmpTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem)
      return emitOpError() << "expects src and dst to have the same element type";
    if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
        return emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
    unsigned elemBytes = getPTOStorageElemByteSize(srcElem);
    if (elemBytes == 0)
      return emitOpError() << "failed to get transpose element size";
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
      return emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
    auto isAllowedWidthType = [&](Type ty) {
      if (elemBytes == 4)
        return ty.isInteger(32) || ty.isF32();
      if (elemBytes == 2)
        return ty.isInteger(16) || ty.isF16() || ty.isBF16();
      return ty.isInteger(8);
    };
    if (!isAllowedWidthType(srcElem))
      return emitOpError() << "expects transpose element type to match the supported set for its width";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type tmpTy = getTmp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type tmpElem = getElemTy(tmpTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem)
      return emitOpError() << "expects src, tmp, and dst to have the same element type";
    unsigned elemBytes = getPTOStorageElemByteSize(srcElem);
    if (elemBytes == 0)
      return emitOpError() << "failed to get transpose element size";
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
      return emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
    auto isAllowedWidthType = [&](Type ty) {
      if (elemBytes == 4)
        return ty.isInteger(32) || ty.isF32();
      if (elemBytes == 2)
        return ty.isInteger(16) || ty.isF16() || ty.isBF16();
      return ty.isInteger(8);
    };
    if (!isAllowedWidthType(srcElem))
      return emitOpError() << "expects transpose element type to match the supported set for its width";
    auto checkAlignedMajor = [&](Type ty, StringRef name) -> LogicalResult {
      auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
      if (!tb)
        return success();
      auto shape = getShapeVec(ty);
      if (shape.size() != 2)
        return success();
      bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
      int64_t major = rowMajor ? shape[1] : shape[0];
      if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % 32 != 0)
        return emitOpError() << "expects " << name << " major dimension times element size to be 32-byte aligned on A5";
      return success();
    };
    if (failed(checkAlignedMajor(srcTy, "src")) || failed(checkAlignedMajor(dstTy, "dst")))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyBase = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    Type elem = *elemOr;
    if (getElemTy(tmpTy) != elem)
      return emitOpError("expects tmp to have the same element type as src0, src1, and dst");
    if (!isRowMajorTileBuf(tmpTy))
      return emitOpError("expects tmp to use row-major layout");
    if (failed(verifyTileBufSameValidShape(*this, tmpTy, getDst().getType(), "tmp", "dst")))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    Type elem = *elemOr;
    if (getElemTy(tmpTy) != elem)
      return emitOpError("expects tmp to have the same element type as src and dst");
    if (!isRowMajorTileBuf(tmpTy))
      return emitOpError("expects tmp to use row-major layout");
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 txors src and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 txors src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TPrintOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  Type srcTy, tmpTy;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

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
  if (parser.parseRParen())
    return failure();
  if (failed(parsePTOInherentAttrs<TPrintOp>(
          parser, result, parsedAttrs, {"printFormat"})))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TPrintOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (Value tmp = getTPrintTmpIfPresent(*this)) {
    p << ", " << tmp << " : " << getSrc().getType() << ", "
      << tmp.getType();
  } else {
    p << " : " << getSrc().getType();
  }
  p << ")";
  NamedAttrList attrs = getNonInherentAttrs(getOperation(), {"printFormat"});
  if (auto printFormatAttr =
          dyn_cast_or_null<pto::PrintFormatAttr>(getProperties().printFormat))
    attrs.append("printFormat", printFormatAttr);
  p.printOptionalAttrDict(attrs.getAttrs());
}

mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  auto srcType = getSrc().getType();
  Value tmp = getTPrintTmpIfPresent(*this);
  auto printFormatAttr =
      dyn_cast_or_null<pto::PrintFormatAttr>(getProperties().printFormat);
  if (printFormatAttr && !tmp)
    return emitOpError() << "expects printFormat only when tmp is present";

  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError() << "expects printable tile element type";
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!tmp) {
      if (!space || *space != pto::AddressSpace::VEC)
        return emitOpError() << "expects printable tile_buf without tmp to be in vec address space";
      return success();
    }

    if (!space)
      return emitOpError() << "expects printable tile_buf with tmp to use a supported address space";
    if (*space == pto::AddressSpace::MAT && isTargetArchA5(getOperation()))
      return emitOpError() << "expects mat tile printing with tmp only on A2/A3 targets";
    if (*space != pto::AddressSpace::VEC && *space != pto::AddressSpace::MAT &&
        *space != pto::AddressSpace::ACC)
      return emitOpError() << "expects printable tile_buf with tmp to be in vec/mat/acc address space";
    if (failed(verifyMGatherMScatterMemOperand(getOperation(), tmp, elem, "tmp")))
      return failure();
    return success();
  }
  if (tmp)
    return emitOpError() << "expects tmp only when src is a tile_buf";
  if (mlir::dyn_cast<MemRefType>(srcType) ||
      mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType))
    return mlir::success();
  return emitOpError() << "expects tile_buf, memref, or partition_tensor_view for src";
}



[[maybe_unused]] static LogicalResult verifyMatmulCommon(Operation *op, Value lhs, Value rhs,
                                       Value biasOpt, Type maybeDstElemTy,
                                       Type maybeResultElemTy) {
  // ---- case A: tensor/memref (ShapedType) ----
  if (auto lhsTy = dyn_cast<ShapedType>(lhs.getType())) {
    auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
    if (!rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
      return op->emitOpError("expects lhs and rhs to be ranked tensors or memrefs");

    if (lhsTy.getElementType() != rhsTy.getElementType())
      return op->emitOpError()
             << "expects lhs and rhs to have the same element type, but got lhs="
             << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();

    if (biasOpt) {
      auto biasTy = dyn_cast<ShapedType>(biasOpt.getType());
      if (!biasTy || !biasTy.hasRank())
        return op->emitOpError("expects bias to be a ranked tensor or memref");
      if (biasTy.getElementType() != lhsTy.getElementType())
        return op->emitOpError()
               << "expects bias to have the same element type as lhs and rhs, but got bias="
               << biasTy.getElementType() << " vs " << lhsTy.getElementType();
    }

    if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "expects dst to have the same element type as lhs and rhs, but got dst="
             << maybeDstElemTy << " vs " << lhsTy.getElementType();

    if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "expects result to have the same element type as lhs and rhs, but got result="
             << maybeResultElemTy << " vs " << lhsTy.getElementType();

    return success();
  }

  // ---- case B: tile ----
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!lhsTile || !rhsTile)
    return op->emitOpError("expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");

  if (lhsTile.getElementType() != rhsTile.getElementType())
    return op->emitOpError() << "expects lhs and rhs tiles to have the same element type, but got lhs="
                             << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();

  if ((int64_t)lhsTile.getShape().size() != 2 || (int64_t)rhsTile.getShape().size() != 2)
    return op->emitOpError("expects lhs and rhs tiles to be 2D");

  if (lhsTile.getShape()[1] != rhsTile.getShape()[0])
    return op->emitOpError() << "expects lhs dim1 to equal rhs dim0, but got "
                             << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile)
      return op->emitOpError("expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    if (biasTile.getElementType() != lhsTile.getElementType())
      return op->emitOpError("expects bias to have the same element type as lhs and rhs");
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType())
    return op->emitOpError() << "expects dst to have the same element type as lhs and rhs";

  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType())
    return op->emitOpError() << "expects result to have the same element type as lhs and rhs";

  return success();
}
