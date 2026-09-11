// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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

mlir::LogicalResult mlir::pto::TInterleaveOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tinterleave is only supported on A5 targets");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dst0Ty = getDst0().getType();
    Type dst1Ty = getDst1().getType();

    bool invalidTile =
        failed(verifyVecTileCommon(*this, src0Ty, "src0")) ||
        failed(verifyVecTileCommon(*this, src1Ty, "src1")) ||
        failed(verifyVecTileCommon(*this, dst0Ty, "dst0")) ||
        failed(verifyVecTileCommon(*this, dst1Ty, "dst1"));
    if (invalidTile) {
      return failure();
    }
