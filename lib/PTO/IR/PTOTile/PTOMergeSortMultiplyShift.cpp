// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static ParseResult resolveTMrgSortFormat2(OpAsmParser &parser,
                                          OperationState &result,
                                          TMrgSortFormat2State &state) {
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(state.srcs.size()), 0, 1,
           state.noTmp ? 0 : 1, 1}));
  if (parser.resolveOperands(state.srcs, state.srcTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(state.dst, state.dstTy, result.operands) ||
      (!state.noTmp && parser.resolveOperand(state.tmp, state.tmpTy,
                                             result.operands)) ||
      parser.resolveOperand(state.executed, state.executedTy,
                            result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted",
                        parser.getBuilder().getBoolAttr(state.exhausted));
  return success();
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(first) || parser.parseComma() ||
      parser.parseOperand(second))
    return failure();
  if (succeeded(parser.parseOptionalColon()))
    return parseTMrgSortFormat1(parser, result, first, second);
  TMrgSortFormat2State state;
  if (failed(parseTMrgSortFormat2Inputs(parser, state, first, second)) ||
      failed(parseTMrgSortFormat2TypesAndOutputs(parser, state)))
    return failure();
  return resolveTMrgSortFormat2(parser, result, state);
}

static LogicalResult verifyTMrgSortFormat1(TMrgSortOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "format1 expects PTO shaped-like types for src/dst";
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src/dst to have the same element type";
  }
  if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects element type to be f16 or f32";
  }
  auto ss = getShapeVec(srcTy);
  auto ds = getShapeVec(dstTy);
  if (ss.size() != 2 || ds.size() != 2) {
    return op.emitOpError() << "expects src/dst to be rank-2 tile-shaped";
  }
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1) {
    return op.emitOpError() << "expects src rows == 1";
  }
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1) {
    return op.emitOpError() << "expects dst rows == 1";
  }
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1]) {
    return op.emitOpError() << "expects src/dst cols to match";
  }
  if (op.getBlockLen()) {
    if (auto cstOp = op.getBlockLen().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
        int64_t v = intAttr.getValue().getSExtValue();
        if (v <= 0 || (v % 64) != 0) {
          return op.emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
  }
  return mlir::success();
}

static LogicalResult verifyTMrgSortOutputShapes(TMrgSortOp op, Type dstTy,
                                                Type tmpTy) {
  auto dstShape = getShapeVec(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  if (dstShape.size() != 2 || (tmpTy && tmpShape.size() != 2))
    return op.emitOpError(
        "format2 expects dst/tmp to be rank-2 tile-shaped");
  if (dstShape[0] != ShapedType::kDynamic && dstShape[0] != 1)
    return op.emitOpError("format2 expects dst/tmp rows == 1");
  if (tmpTy && tmpShape[0] != ShapedType::kDynamic && tmpShape[0] != 1)
    return op.emitOpError("format2 expects dst/tmp rows == 1");
  if (tmpTy && dstShape[1] != ShapedType::kDynamic &&
      tmpShape[1] != ShapedType::kDynamic && tmpShape[1] < dstShape[1])
    return op.emitOpError("format2 expects tmp.cols >= dst.cols");
  return success();
}

static LogicalResult verifyTMrgSortFormat2Outputs(TMrgSortOp op, Type dstTy,
                                                  Type tmpTy) {
  if (!isPTOShapedLike(dstTy) || (tmpTy && !isPTOShapedLike(tmpTy)))
    return op.emitOpError("format2 dst/tmp must be PTO shaped-like");
  auto executedTy = dyn_cast<mlir::VectorType>(op.getExcuted().getType());
  if (!executedTy || executedTy.getRank() != 1 ||
      executedTy.getNumElements() != 4 ||
      !executedTy.getElementType().isInteger(16))
    return op.emitOpError("format2 excuted must be vector<4xi16>");
  Type elemTy = getElemTy(dstTy);
  if (tmpTy && elemTy != getElemTy(tmpTy))
    return op.emitOpError(
        "format2 expects dst/tmp element types to match");
  return verifyTMrgSortOutputShapes(op, dstTy, tmpTy);
}

static LogicalResult verifyTMrgSortFormat2Basics(TMrgSortOp op) {
  for (Value v : op.getSrcs()) {
    if (!isPTOShapedLike(v.getType())) {
      return op.emitOpError() << "format2 expects PTO shaped-like type for each src";
    }
  }
  if (op.getSrcs().size() < 2u || op.getSrcs().size() > 4u) {
    return op.emitOpError() << "format2 expects 2 to 4 srcs";
  }
  if (op.getDsts().size() != 1u || !op.getExcuted()) {
    return op.emitOpError()
           << "format2 expects 2 to 4 srcs, one dst, and excuted=vector";
  }
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  return verifyTMrgSortFormat2Outputs(op, dstTy, tmpTy);
}

static LogicalResult verifyTMrgSortFormat2Srcs(TMrgSortOp op) {
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type elemTy = getElemTy(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  int64_t requiredTmpCols = 0;
  for (Value src : op.getSrcs()) {
    Type srcTy = src.getType();
    auto srcShape = getShapeVec(srcTy);
    auto srcValidShape = getValidShapeVec(src);
    if (srcShape.size() != 2 || srcValidShape.size() != 2) {
      return op.emitOpError() << "format2 expects src to be rank-2 tile-shaped";
    }
    if (srcShape[0] != mlir::ShapedType::kDynamic && srcShape[0] != 1) {
      return op.emitOpError() << "format2 expects src rows == 1";
    }
    if (getElemTy(srcTy) != elemTy) {
      return op.emitOpError() << "format2 expects src/dst/tmp element types to match";
    }
    if (srcValidShape[1] == mlir::ShapedType::kDynamic) {
      requiredTmpCols = mlir::ShapedType::kDynamic;
    } else if (requiredTmpCols != mlir::ShapedType::kDynamic) {
      requiredTmpCols += srcValidShape[1];
    }
  }
  if (tmpTy && requiredTmpCols != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < requiredTmpCols) {
    return op.emitOpError()
           << "format2 expects tmp.cols >= sum(src.cols) = "
           << requiredTmpCols;
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (isFormat1()) {
    return verifyTMrgSortFormat1(*this);
  }
  if (isFormat2() || isFormat2WithoutTmp()) {
    if (failed(verifyTMrgSortFormat2Basics(*this))) {
      return failure();
    }
    return verifyTMrgSortFormat2Srcs(*this);
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
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return emitOpError() << "failed to get element type for src/dst";
  }
  if (srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!mlir::isa<IntegerType>(srcElem)) {
    return emitOpError() << "expects integral element types";
  }
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0) {
    return emitOpError("expects tshls scalar to be non-negative");
  }
  return mlir::success();
}

static FailureOr<Type> verifyMatchingVecUnaryTiles(Operation *op, Type srcTy,
                                                   Type dstTy) {
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  return verifyMatchingElementTypes(op, srcTy, dstTy);
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingVecUnaryTiles(getOperation(), getSrc().getType(),
                                       getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
