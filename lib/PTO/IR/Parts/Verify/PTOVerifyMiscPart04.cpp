// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyMisc.cpp; kept as a fragment included by PTOVerifyMisc.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static ParseResult parseTMrgSortFormat2Types(
    OpAsmParser &parser, MutableArrayRef<OpAsmParser::UnresolvedOperand> srcs,
    SmallVectorImpl<Type> &srcTypes, Type &tmpTy) {
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
  tmpTy = srcTypes.pop_back_val();
  return success();
}

static ParseResult parseTMrgSortFormat2Outputs(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &dstOp,
    OpAsmParser::UnresolvedOperand &excutedOp, Type &dstTy, Type &excutedTy) {
  if (parser.parseOperand(dstOp) || parser.parseComma() ||
      parser.parseOperand(excutedOp) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseComma() ||
      parser.parseType(excutedTy) || parser.parseRParen())
    return failure();
  return success();
}

static ParseResult resolveTMrgSortFormat2Operands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> srcs, ArrayRef<Type> srcTypes,
    OpAsmParser::UnresolvedOperand tmpOp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dstOp, Type dstTy,
    OpAsmParser::UnresolvedOperand excutedOp, Type excutedTy,
    bool exhaustedVal) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(srcs.size()), 0, 1, 1, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      parser.resolveOperand(tmpOp, tmpTy, result.operands) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted")) {
    result.addAttribute("exhausted",
                        parser.getBuilder().getBoolAttr(exhaustedVal));
  }
  return success();
}

static ParseResult parseTMrgSortFormat2(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  SmallVec4<OpAsmParser::UnresolvedOperand> srcs = {first, second};
  OpAsmParser::UnresolvedOperand tmpOp;
  if (failed(parseTMrgSortFormat2Sources(parser, srcs, tmpOp)))
    return failure();
  bool exhaustedVal = false;
  if (failed(parseTMrgSortFormat2Exhausted(parser, exhaustedVal)))
    return failure();
  SmallVec4<Type> srcTypes;
  srcTypes.reserve(srcs.size());
  Type tmpTy;
  if (failed(parseTMrgSortFormat2Types(parser, srcs, srcTypes, tmpTy)))
    return failure();
  OpAsmParser::UnresolvedOperand dstOp, excutedOp;
  Type dstTy, excutedTy;
  if (failed(
          parseTMrgSortFormat2Outputs(parser, dstOp, excutedOp, dstTy, excutedTy)))
    return failure();
  return resolveTMrgSortFormat2Operands(parser, result, srcs, srcTypes, tmpOp,
                                        tmpTy, dstOp, dstTy, excutedOp,
                                        excutedTy, exhaustedVal);
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second))
    return failure();
  if (parser.parseOptionalColon().succeeded())
    return parseTMrgSortFormat1(parser, result, first, second);
  return parseTMrgSortFormat2(parser, result, first, second);
}

static LogicalResult verifyTMrgSortFormat1(TMrgSortOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op.emitOpError()
           << "format1 expects PTO shaped-like types for src/dst";
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError() << "expects src/dst to have the same element type";
  if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32())
    return op.emitOpError() << "expects element type to be f16 or f32";
  auto ss = getShapeVec(srcTy);
  auto ds = getShapeVec(dstTy);
  if (ss.size() != kNumber2 || ds.size() != kNumber2)
    return op.emitOpError() << "expects src/dst to be rank-2 tile-shaped";
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
    return op.emitOpError() << "expects src rows == 1";
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
    return op.emitOpError() << "expects dst rows == 1";
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic &&
      ss[1] != ds[1])
    return op.emitOpError() << "expects src/dst cols to match";
  if (auto cstOp = op.getBlockLen().getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
      int64_t v = intAttr.getValue().getSExtValue();
      if (v <= 0 || (v % kNumber64) != 0)
        return op.emitOpError()
               << "expects blockLen > 0 and multiple of 64";
    }
  }
  return mlir::success();
}

static LogicalResult verifyTMrgSortSingleRowTile(Operation *op, Type ty,
                                                 StringRef name) {
  auto shape = getShapeVec(ty);
  if (shape.size() != kPTORowColRank)
    return op->emitOpError() << "format2 expects " << name
                             << " to be rank-2 tile-shaped";
  if (shape[0] != mlir::ShapedType::kDynamic && shape[0] != 1)
    return op->emitOpError() << "format2 expects " << name << " rows == 1";
  return success();
}

static LogicalResult verifyTMrgSortFormat2Executed(Operation *op,
                                                   Value excuted) {
  auto excutedTy = mlir::dyn_cast<mlir::VectorType>(excuted.getType());
  if (!excutedTy || excutedTy.getRank() != 1 ||
      excutedTy.getNumElements() != kNumber4 ||
      !excutedTy.getElementType().isInteger(kPTOI16BitWidth))
    return op->emitOpError() << "format2 excuted must be vector<4xi16>";
  return success();
}

static LogicalResult verifyTMrgSortFormat2Src(Operation *op, Value src,
                                              Type elemTy) {
  Type srcTy = src.getType();
  if (failed(verifyTMrgSortSingleRowTile(op, srcTy, "src")))
    return failure();
  if (getElemTy(srcTy) != elemTy)
    return op->emitOpError()
           << "format2 expects src/dst/tmp element types to match";
  return success();
}

static LogicalResult verifyTMrgSortFormat2(TMrgSortOp op) {
  for (Value v : op.getSrcs()) {
    if (!isPTOShapedLike(v.getType()))
      return op.emitOpError()
             << "format2 expects PTO shaped-like type for each src";
  }
  if (op.getSrcs().size() < 2u || op.getSrcs().size() > 4u)
    return op.emitOpError() << "format2 expects 2 to 4 srcs";
  if (op.getDsts().size() != 1u || !op.getTmp() || !op.getExcuted())
    return op.emitOpError()
           << "format2 expects ins(srcs..., tmp), outs(dst), and excuted=vector";
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp().getType();
  if (!isPTOShapedLike(dstTy) || !isPTOShapedLike(tmpTy))
    return op.emitOpError() << "format2 dst/tmp must be PTO shaped-like";
  if (failed(verifyTMrgSortFormat2Executed(op, op.getExcuted())))
    return failure();
  Type elemTy = getElemTy(dstTy);
  if (elemTy != getElemTy(tmpTy))
    return op.emitOpError() << "format2 expects dst/tmp element types to match";
  auto dstShape = getShapeVec(dstTy);
  auto tmpShape = getShapeVec(tmpTy);
  if (failed(verifyTMrgSortSingleRowTile(op, dstTy, "dst")) ||
      failed(verifyTMrgSortSingleRowTile(op, tmpTy, "tmp")))
    return failure();
  if (dstShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < dstShape[1])
    return op.emitOpError() << "format2 expects tmp.cols >= dst.cols";
  for (Value src : op.getSrcs()) {
    if (failed(verifyTMrgSortFormat2Src(op, src, elemTy)))
      return failure();
  }
  return mlir::success();
}

