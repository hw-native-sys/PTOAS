// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
