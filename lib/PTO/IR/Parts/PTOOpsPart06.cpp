// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second))
    return failure();

  if (parser.parseOptionalColon().succeeded()) {
    Type srcTy, blockLenTy, dstTy;
    if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(blockLenTy) ||
        parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen())
      return failure();
    OpAsmParser::UnresolvedOperand dstOp;
    if (parser.parseOperand(dstOp) || parser.parseColon() || parser.parseType(dstTy) ||
        parser.parseRParen())
      return failure();
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0, 0}));
    if (parser.resolveOperand(first, srcTy, result.operands) ||
        parser.resolveOperand(second, blockLenTy, result.operands) ||
        parser.resolveOperand(dstOp, dstTy, result.operands))
      return failure();
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (!result.attributes.get("exhausted"))
      result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
    return success();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs = {first, second};
  while (parser.parseOptionalComma().succeeded()) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next))
      return failure();
    srcs.push_back(next);
  }
  if (srcs.size() < 3 || srcs.size() > 5)
    return parser.emitError(parser.getCurrentLocation(),
                            "tmrgsort format2 expects 2 to 4 src operands plus one tmp operand");
  OpAsmParser::UnresolvedOperand tmpOp = srcs.pop_back_val();
  bool exhaustedVal = false;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parser.parseKeyword("exhausted") || parser.parseEqual())
      return failure();
    StringRef kw;
    if (parser.parseKeyword(&kw) || parser.parseRBrace())
      return failure();
    exhaustedVal = (kw == "true");
  }
  SmallVector<Type, 4> srcTypes;
  srcTypes.reserve(srcs.size());
  if (parser.parseColon())
    return failure();
  Type firstSrcTy;
  if (parser.parseType(firstSrcTy))
    return failure();
  srcTypes.push_back(firstSrcTy);
  while (parser.parseOptionalComma().succeeded()) {
    Type nextTy;
    if (parser.parseType(nextTy))
      return failure();
    srcTypes.push_back(nextTy);
  }
  if (srcTypes.size() != srcs.size() + 1 || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  Type tmpTy = srcTypes.pop_back_val();
  OpAsmParser::UnresolvedOperand dstOp, excutedOp;
  Type dstTy, excutedTy;
  if (parser.parseOperand(dstOp) || parser.parseComma() || parser.parseOperand(excutedOp) ||
      parser.parseColon() || parser.parseType(dstTy) || parser.parseComma() ||
      parser.parseType(excutedTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(srcs.size()), 0, 1, 1, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      parser.resolveOperand(tmpOp, tmpTy, result.operands) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(exhaustedVal));
  return success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (isFormat1()) {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
      return emitOpError() << "format1 expects PTO shaped-like types for src/dst";
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError() << "expects src/dst to have the same element type";
    if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32())
      return emitOpError() << "expects element type to be f16 or f32";
    auto ss = getShapeVec(srcTy);
    auto ds = getShapeVec(dstTy);
    if (ss.size() != 2 || ds.size() != 2)
      return emitOpError() << "expects src/dst to be rank-2 tile-shaped";
    if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
      return emitOpError() << "expects src rows == 1";
    if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
      return emitOpError() << "expects dst rows == 1";
    if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1])
      return emitOpError() << "expects src/dst cols to match";
    if (getBlockLen()) {
      if (auto cstOp = getBlockLen().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
          int64_t v = intAttr.getValue().getSExtValue();
          if (v <= 0 || (v % 64) != 0)
            return emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
    return mlir::success();
  }
  if (isFormat2()) {
    for (Value v : getSrcs())
      if (!isPTOShapedLike(v.getType()))
        return emitOpError() << "format2 expects PTO shaped-like type for each src";
    if (getSrcs().size() < 2u || getSrcs().size() > 4u)
      return emitOpError() << "format2 expects 2 to 4 srcs";
    if (getDsts().size() != 1u || !getTmp() || !getExcuted())
      return emitOpError() << "format2 expects ins(srcs..., tmp), outs(dst), and excuted=vector";
    Type dstTy = getDst().getType();
    Type tmpTy = getTmp().getType();
    if (!isPTOShapedLike(dstTy) || !isPTOShapedLike(tmpTy))
      return emitOpError() << "format2 dst/tmp must be PTO shaped-like";
    auto excutedTy = mlir::dyn_cast<mlir::VectorType>(getExcuted().getType());
    if (!excutedTy || excutedTy.getRank() != 1 || excutedTy.getNumElements() != 4 ||
        !excutedTy.getElementType().isInteger(16))
      return emitOpError() << "format2 excuted must be vector<4xi16>";
    Type elemTy = getElemTy(dstTy);
    if (elemTy != getElemTy(tmpTy))
      return emitOpError() << "format2 expects dst/tmp element types to match";
    auto dstShape = getShapeVec(dstTy);
    auto tmpShape = getShapeVec(tmpTy);
    if (dstShape.size() != 2 || tmpShape.size() != 2)
      return emitOpError() << "format2 expects dst/tmp to be rank-2 tile-shaped";
    if ((dstShape[0] != mlir::ShapedType::kDynamic && dstShape[0] != 1) ||
        (tmpShape[0] != mlir::ShapedType::kDynamic && tmpShape[0] != 1))
      return emitOpError() << "format2 expects dst/tmp rows == 1";
    if (dstShape[1] != mlir::ShapedType::kDynamic &&
        tmpShape[1] != mlir::ShapedType::kDynamic &&
        tmpShape[1] < dstShape[1])
      return emitOpError() << "format2 expects tmp.cols >= dst.cols";
    for (Value src : getSrcs()) {
      Type srcTy = src.getType();
      auto srcShape = getShapeVec(srcTy);
      if (srcShape.size() != 2)
        return emitOpError() << "format2 expects src to be rank-2 tile-shaped";
      if (srcShape[0] != mlir::ShapedType::kDynamic && srcShape[0] != 1)
        return emitOpError() << "format2 expects src rows == 1";
      if (getElemTy(srcTy) != elemTy)
        return emitOpError() << "format2 expects src/dst/tmp element types to match";
    }
    return mlir::success();
  }
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return emitOpError() << "failed to get element type for src/dst";
  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!mlir::isa<IntegerType>(srcElem))
    return emitOpError() << "expects integral element types";
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0)
    return emitOpError("expects tshls scalar to be non-negative");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32))
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
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
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
          elemTy.isF32()))
      return emitOpError()
             << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError() << "expects src and dst to have rank-2 valid_shape";
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1])
      return emitOpError()
             << "expects src and dst to have the same valid_shape[1]";

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32() || elemTy.isBF16()))
      return emitOpError()
             << "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16";
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy))
      return emitOpError() << "expects src and dst to have the same element type";
    if (!elemTy.isInteger(16))
      return emitOpError() << "expects A2/A3 tnot element type to be i16";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy))
      return emitOpError() << "expects src and dst to have the same element type";
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32)))
      return emitOpError() << "expects A5 tnot element type to be i8/i16/i32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tor src0, src1, and dst element type to be i8/i16");
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
          "expects A5 tor src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tors src and dst element type to be i8/i16");
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
          "expects A5 tors src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy))
    return op->emitOpError(
               "expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types"),
           failure();
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for operands"), failure();
  if (e0 != e1 || e0 != ed)
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd)
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  return e0;
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError() << "expects src0/src1/dst to have the same element type";
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPattern(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tpartadd element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError() << "expects src0/src1/dst to have the same element type";
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tpartadd element type to be i32/i16/i8/f16/bf16/f32");
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPatternLoose(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    if (failed(verifyPartialValidPattern(*this, t0, t1, td)))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tpartmax element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32()))
      return emitOpError("expects A5 tpartmax element type to be i32/i16/i8/f16/bf16/f32");
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    if (failed(verifyPartialValidPattern(*this, t0, t1, td)))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tpartmin element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32()))
      return emitOpError("expects A5 tpartmin element type to be i32/i16/i8/f16/bf16/f32");
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElemOr))
    return failure();
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)))
    return failure();

  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy))
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy))
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32)
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");

  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) ||
      dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy))
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy))
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");

  Type elem = *dataElemOr;
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
  } else {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return op->emitOpError() << "expects A2/A3 " << opName
                               << " element type to be i32/i16/f16/f32";
  }
  return success();
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

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError()
             << "expects src0/src1/dst to have the same element type";
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError()
             << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPattern(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return emitOpError(
          "expects A2/A3 tpartmul element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError()
             << "expects src0/src1/dst to have the same element type";
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError(
          "expects A5 tpartmul element type to be i32/i16/i8/f16/bf16/f32");
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError()
             << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPatternLoose(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto verifyCommon = [&]() -> FailureOr<std::tuple<Type, Type, Type, Type>> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type tt = getTmp().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, tt, "tmp")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0), e1 = getElemTy(t1), et = getElemTy(tt), ed = getElemTy(td);
    if (!e0 || !e1 || !et || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects dst/src0/src1 to have the same element type");
      return failure();
    }
    if (!(e0.isF16() || e0.isF32())) {
      emitOpError("expects dst/src0/src1 element type to be f16 or f32");
      return failure();
    }
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t1, td, "src1", "dst")))
      return failure();

    auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
    if (s0 != s1 || s0 != sd) {
      emitOpError("expects src0/src1/dst to have the same shape");
      return failure();
    }
    return std::make_tuple(t0, t1, tt, td);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto tysOr = verifyCommon();
    if (failed(tysOr))
      return failure();
    auto [t0, t1, tt, td] = *tysOr;
    Type tmpElem = getElemTy(tt);
    auto tmpIntTy = mlir::dyn_cast<IntegerType>(tmpElem);
    if (!tmpIntTy || tmpIntTy.getWidth() != 8)
      return emitOpError("expects A2/A3 tmp element type to be u8");
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();
    auto tmpShape = getShapeVec(tt);
    auto dstValid = getValidShapeVec(td);
    auto tmpValid = getValidShapeVec(tt);
    if (tmpShape.size() != 2 || dstValid.size() != 2 || tmpValid.size() != 2)
      return emitOpError("expects tmp and dst to be rank-2 tiles");
    if (dstValid[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
        tmpShape[0] < dstValid[0] + 1)
        return emitOpError()
             << "expects A2/A3 tmp shape[0] to be at least dst valid_shape[0] + 1 ("
             << (dstValid[0] + 1) << ")";
    if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic) {
      int64_t packedMaskCols = llvm::divideCeil(dstValid[1], int64_t{8});
      if (tmpValid[1] < packedMaskCols)
        return emitOpError()
               << "expects A2/A3 tmp valid_shape[1] to be at least ceil(dst valid_shape[1] / 8) ("
               << packedMaskCols << ")";
    }
    if (auto arch = getVerifierArchName(getOperation());
        arch && arch->equals_insensitive("a3")) {
      if (getSrc0() == getSrc1() || getSrc0() == getTmp() || getSrc0() == getDst() ||
          getSrc1() == getTmp() || getSrc1() == getDst() || getTmp() == getDst())
        return emitOpError(
            "expects A3 src0, src1, tmp, and dst to use different storage");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto tysOr = verifyCommon();
    if (failed(tysOr))
      return failure();
    auto [t0, t1, tt, td] = *tysOr;
    (void)t0;
    (void)t1;
    (void)td;
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TQuantOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src, fp, offset, dst, tmp;
  Type srcTy, fpTy, offsetTy, dstTy, tmpTy;
  bool hasOffset = false;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() || parser.parseOperand(fp))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(offset))
      return failure();
    hasOffset = true;
  }
  if (parser.parseColon())
    return failure();
  if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(fpTy))
    return failure();
  if (hasOffset) {
    if (parser.parseComma() || parser.parseType(offsetTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColonType(dstTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();
  if (failed(parsePTOInherentAttrs<TQuantOp>(
          parser, result, parsedAttrs, {"quant_type", "operandSegmentSizes"})))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(fp, fpTy, result.operands))
    return failure();
  if (hasOffset) {
    if (parser.resolveOperand(offset, offsetTy, result.operands))
      return failure();
  }
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  auto &properties = result.getOrAddProperties<TQuantOp::Properties>();
  llvm::copy(ArrayRef<int32_t>({1, 1, hasOffset ? 1 : 0, hasTmp ? 1 : 0, 1}),
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

ParseResult mlir::pto::TQuantMxOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  Type srcTy;
  SmallVector<OpAsmParser::UnresolvedOperand, 5> outOperands;
  SmallVector<Type, 5> outTypes;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseColonType(srcTy) ||
      parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen())
    return failure();

  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();
    outOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseColon())
    return failure();

  do {
    Type type;
    if (parser.parseType(type))
      return failure();
    outTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  if (outOperands.size() != outTypes.size())
    return parser.emitError(parser.getCurrentLocation(),
                            "expects the number of outs operands to match the number of outs types");
  if (outOperands.size() != 4 && outOperands.size() != 5)
    return parser.emitError(parser.getCurrentLocation(),
                            "expects 4 or 5 operands in outs(...)");

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!llvm::isa_and_nonnull<pto::QuantTypeAttr>(
          result.attributes.get("quant_type")))
    return parser.emitError(parser.getCurrentLocation(),
                            "expects quant_type attribute");

  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();
  for (auto [operand, type] : llvm::zip_equal(outOperands, outTypes)) {
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }

  return success();
}

void mlir::pto::TQuantMxOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  p << " outs(" << getDst() << ", " << getExp() << ", " << getMax() << ", "
    << getScaling();
  if (auto expZz = getExpZz())
    p << ", " << expZz;
  p << " : " << getDst().getType() << ", " << getExp().getType() << ", "
    << getMax().getType() << ", " << getScaling().getType();
  if (auto expZz = getExpZz())
    p << ", " << expZz.getType();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  auto verifyStructural = [&]() -> LogicalResult {
    Type dstElemTy = getElemTy(getDst().getType());
    auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
    if (getQuantType() == mlir::pto::QuantType::INT8_SYM) {
      if (!getFp())
        return emitOpError()
               << "INT8_SYM quantization requires an fp operand";
      if (getOffset())
        return emitOpError()
               << "INT8_SYM quantization must not have an offset operand";
      if (!dstIntTy || dstIntTy.getWidth() != 8)
        return emitOpError()
               << "expects dst element type i8/ui8 for INT8_SYM quantization";
    } else if (getQuantType() == mlir::pto::QuantType::INT8_ASYM) {
      if (!getFp())
        return emitOpError()
               << "INT8_ASYM quantization requires an fp operand";
      if (!getOffset())
        return emitOpError()
               << "INT8_ASYM quantization requires an offset operand";
      if (!dstIntTy || dstIntTy.getWidth() != 8)
        return emitOpError()
               << "expects dst element type i8/ui8 for INT8_ASYM quantization";
    } else {
      return emitOpError("expects plain tquant quant_type to be INT8_SYM or INT8_ASYM; use tquant.mx for MX quantization");
    }
    return success();
  };

  if (failed(verifyStructural()))
    return failure();

  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto verifyInt8Common = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    if (getTmp() && failed(verifyTileBufCommon(*this, getTmp().getType(), "tmp")))
      return failure();
    if (!getElemTy(srcTy).isF32())
      return emitOpError() << "expects src to have element type f32";
    if (getOffset()) {
      Type offsetTy = getOffset().getType();
      if (failed(verifyTileBufCommon(*this, offsetTy, "offset")))
        return failure();
      if (!getElemTy(offsetTy).isF32())
        return emitOpError() << "expects offset to have element type f32";
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (!getElemTy(tmpTy).isF32())
        return emitOpError() << "expects tmp to have element type f32";
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyInt8Common()))
      return failure();
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    auto verifyA2A3Tmp = [&](Type tmpTy) -> LogicalResult {
      if (failed(verifyTileBufSameElemType(*this, srcTy, tmpTy, "src", "tmp")))
        return failure();
      if (!isRowMajorTileBuf(tmpTy))
        return emitOpError() << "expects A2/A3 tmp to use row-major layout";
      if (getShapeVec(srcTy) != getShapeVec(tmpTy))
        return emitOpError() << "expects A2/A3 tmp to have the same shape as src";
      if (failed(verifyTileBufSameValidShape(*this, srcTy, tmpTy, "src", "tmp")))
        return failure();
      return success();
    };
    if (getTmp() && failed(verifyA2A3Tmp(getTmp().getType())))
      return failure();
    auto verifyA2A3Param = [&](Type paramTy, StringRef paramName) -> LogicalResult {
      if (isRowMajorTileBuf(paramTy))
        return emitOpError() << "expects A2/A3 " << paramName
                             << " to use non-row-major layout";
      auto paramValid = getValidShapeVec(paramTy);
      auto dstValid = getValidShapeVec(dstTy);
      if (paramValid.size() != 2 || dstValid.size() != 2)
        return emitOpError() << "expects A2/A3 " << paramName
                             << " and dst to have rank-2 valid_shape";
      if (paramValid[0] != ShapedType::kDynamic &&
          dstValid[0] != ShapedType::kDynamic && paramValid[0] != dstValid[0])
        return emitOpError() << "expects A2/A3 " << paramName
                             << " valid_shape[0] to equal dst valid_shape[0]";
      if (paramValid[1] != ShapedType::kDynamic && paramValid[1] != 1)
        return emitOpError() << "expects A2/A3 " << paramName
                             << " valid_shape[1] to be 1";
      return success();
    };
    if (failed(verifyA2A3Param(fpTy, "fp")))
      return failure();
    if (getOffset() && failed(verifyA2A3Param(getOffset().getType(), "offset")))
      return failure();
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    return verifyInt8Common();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TQuantMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tquant.mx is only supported on A5");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    if (shouldBypassDecodedMemrefVerifier(getOperation()))
      return success();

    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Type expTy = getExp().getType();
    Type maxTy = getMax().getType();
    Type scalingTy = getScaling().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst", /*allowLowPrecision=*/true)) ||
        failed(verifyNDStyleVecTile(*this, expTy, "exp")) ||
        failed(verifyNDStyleVecTile(*this, maxTy, "max")) ||
        failed(verifyNDStyleVecTile(*this, scalingTy, "scaling")))
      return failure();
    if (getExpZz() &&
        failed(verifyNDStyleVecTile(*this, getExpZz().getType(), "exp_zz")))
      return failure();

    auto quantType = getQuantType();
    if (quantType != mlir::pto::QuantType::MXFP8 &&
        quantType != mlir::pto::QuantType::MXFP4_E2M1)
      return emitOpError("expects quant_type to be MXFP8 or MXFP4_E2M1");
    if (getExpZz() && !getStoreMode())
      return emitOpError("expects storeMode when exp_zz is present");
    if (getStoreMode() && !getExpZz())
      return emitOpError("expects exp_zz when storeMode is present");
    if (getStoreMode() &&
        getQuantScaleAlg() != mlir::pto::QuantScaleAlg::OCP)
      return emitOpError("storeMode form must not override quantScaleAlg");

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    Type expElem = getElemTy(expTy);
    Type maxElem = getElemTy(maxTy);
    Type scalingElem = getElemTy(scalingTy);

    if (!(srcElem.isF32() || srcElem.isF16() || srcElem.isBF16()))
      return emitOpError("expects src element type to be f32/f16/bf16");
    if (!expElem.isInteger(8))
      return emitOpError("expects exp element type to be i8/ui8");
    if (getExpZz() && !getElemTy(getExpZz().getType()).isInteger(8))
      return emitOpError("expects exp_zz element type to be i8/ui8");
    if (maxElem != srcElem)
      return emitOpError("expects max element type to match src element type");
    if (scalingElem != srcElem)
      return emitOpError("expects scaling element type to match src element type");

    if (quantType == mlir::pto::QuantType::MXFP8) {
      if (!dstElem.isInteger(8))
        return emitOpError("expects MXFP8 dst element type to be i8/ui8");
    } else {
      if (!isa<pto::F4E2M1x2Type>(dstElem))
        return emitOpError("expects MXFP4_E2M1 dst element type to be !pto.f4E2M1x2");
      if (!(srcElem.isF16() || srcElem.isBF16()))
        return emitOpError("expects MXFP4_E2M1 src element type to be f16/bf16");
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    auto expValid = getValidShapeVec(expTy);
    auto maxValid = getValidShapeVec(maxTy);
    auto scalingValid = getValidShapeVec(scalingTy);
    if (srcValid.size() != 2 || dstValid.size() != 2 || expValid.size() != 2 ||
        maxValid.size() != 2 || scalingValid.size() != 2)
      return emitOpError("expects rank-2 valid_shape for src/dst/exp/max/scaling");
    if (getExpZz() && getValidShapeVec(getExpZz().getType()).size() != 2)
      return emitOpError("expects rank-2 valid_shape for exp_zz");
    // scaling is a per-group tile (like exp/max), NOT per-element: the ISA
    // flattens it to 1D and writes one reciprocal scale per 32-element group.
    // Only enforce element type match with src here; the element-count constraint
    // is checked below alongside exp/max.
    if (failed(verifyTileBufSameElemType(*this, srcTy, maxTy, "src", "max")) ||
        failed(verifyTileBufSameElemType(*this, srcTy, scalingTy, "src", "scaling")))
      return failure();
    // dst must carry the same logical element count as src. For MXFP8 this is a
    // plain valid_shape match; for MXFP4_E2M1 the packed dst (f4E2M1x2, 2 elems
    // per byte) is handled by verifyTileBufSameLogicalExtent.
    if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                              /*compareValidShape=*/true)))
      return failure();

    int64_t srcRows = srcValid[0];
    int64_t srcCols = srcValid[1];
    if (srcCols != ShapedType::kDynamic && srcCols % 32 != 0)
      return emitOpError("expects src valid_shape[1] to be a multiple of 32 for tquant.mx");
    if (srcRows != ShapedType::kDynamic && srcCols != ShapedType::kDynamic) {
      int64_t groups = (srcRows * srcCols) / 32;
      int64_t expElems = expValid[0] == ShapedType::kDynamic || expValid[1] == ShapedType::kDynamic
                             ? ShapedType::kDynamic
                             : expValid[0] * expValid[1];
      int64_t maxElems = maxValid[0] == ShapedType::kDynamic || maxValid[1] == ShapedType::kDynamic
                             ? ShapedType::kDynamic
                             : maxValid[0] * maxValid[1];
      int64_t scalingElems = scalingValid[0] == ShapedType::kDynamic ||
                                     scalingValid[1] == ShapedType::kDynamic
                                 ? ShapedType::kDynamic
                                 : scalingValid[0] * scalingValid[1];
      int64_t expZzElems = ShapedType::kDynamic;
      if (auto expZz = getExpZz()) {
        auto expZzValid = getValidShapeVec(expZz.getType());
        expZzElems = expZzValid[0] == ShapedType::kDynamic ||
                             expZzValid[1] == ShapedType::kDynamic
                         ? ShapedType::kDynamic
                         : expZzValid[0] * expZzValid[1];
      }
      if (expElems != ShapedType::kDynamic && expElems != groups)
        return emitOpError("expects exp valid element count to equal src valid elements / 32");
      if (maxElems != ShapedType::kDynamic && maxElems != groups)
        return emitOpError("expects max valid element count to equal src valid elements / 32");
      if (scalingElems != ShapedType::kDynamic && scalingElems != groups)
        return emitOpError(
            "expects scaling valid element count to equal src valid elements / 32");
      if (expZzElems != ShapedType::kDynamic && expZzElems != groups)
        return emitOpError(
            "expects exp_zz valid element count to equal src valid elements / 32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16))
      return emitOpError()
             << "expects src element type i8 or i16";
    if (!getElemTy(getDst().getType()).isF32())
      return emitOpError() << "expects dst element type f32";
    if (!getElemTy(getScale().getType()).isF32())
      return emitOpError() << "expects scale element type f32";
    if (!getElemTy(getOffset().getType()).isF32())
      return emitOpError() << "expects offset element type f32";
    return success();
  };

  if (failed(verifyStructural()))
    return failure();

  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst")))
      return failure();
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType()))
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst())
    return emitOpError("expects A3 trecip src and dst to use different storage");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << errorMessage;
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


mlir::LogicalResult mlir::pto::TRemOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type tmpTy = getTmp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src0, src1, and dst to use row-major layout");
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2)
    return emitOpError("expects tmp and dst to be rank-2 tiles");

  Type elem = getElemTy(src0Ty);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileCommon(*this, tmpTy, "tmp")))
      return failure();
    if (getElemTy(tmpTy) != getElemTy(dstTy))
      return emitOpError("expects tmp and dst to have the same element type");
    if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 2)
      return emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 2");
    if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
        tmpValid[1] < dstValid[1])
      return emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
    if (!(elem.isInteger(32) || elem.isF32()))
      return emitOpError("expects A2/A3 trem element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileCommon(*this, tmpTy, "tmp")))
      return failure();
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tt = getTmp().getType();
  Type td = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tt, "tmp")) ||
      failed(verifyTileBufCommon(*this, td, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, ts, td, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(ts) || !isRowMajorTileBuf(td))
    return emitOpError("expects src and dst to use row-major layout");
  Type elem = getElemTy(ts);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (dstValid.size() != 2 || tmpValid.size() != 2)
    return emitOpError("expects tmp and dst to be rank-2 tiles");
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();
    if (getElemTy(tt) != getElemTy(td))
      return emitOpError("expects tmp and dst to have the same element type");
    if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1)
      return emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
    if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
        tmpValid[1] < dstValid[1])
      return emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
    if (!(elem.isInteger(32) || elem.isF32()))
      return emitOpError("expects A2/A3 trems element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src and dst to use row-major layout");

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpShape(Operation *op, Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufSameElemType(op, tmpTy, dstTy, "tmp", "dst")))
    return failure();
  if (!isRowMajorTileBuf(tmpTy))
    return op->emitOpError("expects tmp to use row-major layout");
  return verifyTileBufSameValidShape(op, tmpTy, dstTy, "tmp", "dst");
}

mlir::LogicalResult mlir::pto::TPowOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type baseTy = getBase().getType();
  Type expTy = getExp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, baseTy, "base")) ||
      failed(verifyTileBufCommon(*this, expTy, "exp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameElemType(*this, baseTy, dstTy, "base", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, dstTy, "base", "dst")))
    return failure();
  if (!isRowMajorTileBuf(baseTy) || !isRowMajorTileBuf(expTy) ||
      !isRowMajorTileBuf(dstTy))
    return emitOpError("expects base, exp, and dst to use row-major layout");

  Type elem = getElemTy(baseTy);
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  bool isFpElem = elem.isF16() || elem.isF32() || elem.isBF16();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (getPrecisionType() == pto::PowPrecision::HighPrecision)
      return emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    if (!(isIntElem || elem.isF32()))
      return emitOpError(
          "expects A2/A3 tpow element type to be i8/i16/i32 or f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16()))
        return emitOpError("expects A5 tpow element type to be f16/f32/bf16 "
                           "when precisionType=high_precision");
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32()))
        return emitOpError(
            "expects A5 tpow element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
    }
    return success();
  };
  if (failed(dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5)))
    return failure();

  if (isFpElem && !getTmp())
    return emitOpError(
        "expects tmp when element type is floating-point (required by the "
        "floating-point pow lowering)");
  if (isIntElem && getTmp())
    return emitOpError(
        "does not accept tmp when element type is integer (the integer pow "
        "lowering uses the 3-operand form TPOW(dst, base, exp))");
  if (auto tmp = getTmp()) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    if (failed(verifyTPowTmpShape(getOperation(), tmpTy, dstTy)))
      return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPowSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src and dst to use row-major layout");
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");

  // Same dtype matrix as TPowOp; see comment in TPowOp::verify.
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  bool isFpElem = elem.isF16() || elem.isF32() || elem.isBF16();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (getPrecisionType() == pto::PowPrecision::HighPrecision)
      return emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    if (!(isIntElem || elem.isF32()))
      return emitOpError(
          "expects A2/A3 tpows element type to be i8/i16/i32 or f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16()))
        return emitOpError("expects A5 tpows element type to be f16/f32/bf16 "
                           "when precisionType=high_precision");
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32()))
        return emitOpError(
            "expects A5 tpows element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
    }
    return success();
  };
  if (failed(dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5)))
    return failure();

  if (isFpElem && !getTmp())
    return emitOpError(
        "expects tmp when element type is floating-point (required by the "
        "floating-point pow lowering)");
  if (isIntElem && getTmp())
    return emitOpError(
        "does not accept tmp when element type is integer (the integer pows "
        "lowering uses the 3-operand form TPOWS(dst, src, scalar))");
  if (auto tmp = getTmp()) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    if (failed(verifyTPowTmpShape(getOperation(), tmpTy, dstTy)))
      return failure();
  }
  return success();
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return std::nullopt;
    if (d < 0)
      return std::nullopt;
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy)
    return std::nullopt;
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return 2;
    if (ft.isF32())
      return 4;
    if (ft.isF64())
      return 8;
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(1, bits / 8);
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isTileBufOrMemref(Type ty) {
  return mlir::isa<MemRefType, pto::TileBufType>(ty);
}

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value))
    return false;

  if (isa<AllocTileOp, DeclareTileOp, BindTileOp, PointerCastOp,
          MaterializeTileOp>(
          value.getDefiningOp()))
    return true;

  if (auto bitcast = value.getDefiningOp<BitcastOp>())
    return isLocallyBoundTileSource(bitcast.getSrc());
  if (auto reshape = value.getDefiningOp<TReshapeOp>())
    return isLocallyBoundTileSource(reshape.getSrc());

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return cOp.value();
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
    return cInt.value();
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue()))
      return ia.getInt();
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexLike(castOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexLike(truncOp.getIn());
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 tile_buf source");

    ArrayRef<int64_t> validShape = srcTy.getValidShape();
    if (validShape.size() != 2)
      return emitOpError("expects source validShape to be rank-2");
    if (!srcTy.hasDynamicValid())
      return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");

    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

    if (!isLocallyBoundTileSource(getSource()))
      return emitOpError(
          "requires a locally bound tile source; function arguments/results "
          "are unsupported");
  } else if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (!(*this)->hasAttr(kLoweredSetValidShapeAttrName))
      return emitOpError(
          "expects tile_buf source; memref source is only valid for the internal lowered form");
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 memref source after tile lowering");
    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());
  } else {
    return emitOpError("expects tile_buf source (or lowered memref source)");
  }

  auto checkDim = [&](Value operand, unsigned dimIdx,
                      StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal)
      return success();

    if (*constVal < 0)
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic)
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    return success();
  };

  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row")))
    return failure();
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col")))
    return failure();

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 tile_buf source");
    if (srcTy.getValidShape().size() != 2)
      return emitOpError("expects source validShape to be rank-2");
    return success();
  }
  if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 memref source after tile lowering");
    return success();
  }
  return emitOpError("expects tile_buf source (or lowered memref source)");
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb)
    return emitOpError("expects src/result to be !pto.tile_buf types");

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst")))
    return failure();

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace())
    return emitOpError("expects src and dst to use the same loc");

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value())
    return emitOpError("failed to get element byte width for src/dst");

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value())
    return emitOpError("expects static shapes for treshape");

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value())
    return emitOpError("expects src and dst to have the same total byte size");

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed)
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");

  return success();
}
