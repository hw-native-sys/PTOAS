// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

    SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{src};
    SmallVector<Type, 1> fixedTypes(1);
    if (failed(parseCommCollectiveTail(
            parser, result, fixedOperands, fixedTypes, recvClause, groupOps, groupTypes,
            {1, 1, recvClause.pong ? 1 : 0}, {"root"}))) {
        return failure();
    }
    return success();
}

static void printCommSingleFixedOperand(
    OpAsmPrinter& p, Operation* op, Value fixed, Value ping, Value pong, ValueRange group)
{
    p << "(" << fixed << ", ";
    printCommRecvClause(p, ping, pong);
    p << ", ";
    printCommGroupClause(p, group);
    p << " : " << fixed.getType() << ", " << ping.getType();
    if (pong) {
        p << ", " << pong.getType();
    }
    printCommGroupTypes(p, group);
    p << ")";
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TBroadcastOp::parse(OpAsmParser& parser, OperationState& result)
{
    return parseCommSingleFixedOperand(parser, result);
}

void mlir::pto::TBroadcastOp::print(OpAsmPrinter& p)
{
    printCommSingleFixedOperand(p, getOperation(), getSrc(), getPing(), getPong(), getGroup());
}

ParseResult mlir::pto::CommTGatherOp::parse(OpAsmParser& parser, OperationState& result)
{
    return parseCommSingleFixedOperand(parser, result);
}

void mlir::pto::CommTGatherOp::print(OpAsmPrinter& p)
{
    printCommSingleFixedOperand(p, getOperation(), getDst(), getPing(), getPong(), getGroup());
}

ParseResult mlir::pto::CommTScatterOp::parse(OpAsmParser& parser, OperationState& result)
{
    return parseCommSingleFixedOperand(parser, result);
}

void mlir::pto::CommTScatterOp::print(OpAsmPrinter &p) {
  printCommSingleFixedOperand(p, getOperation(), getSrc(),
                             getPing(), getPong(), getGroup());
}

ParseResult mlir::pto::TReduceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand dst, acc;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> groupOps;
  SmallVector<Type, mlir::pto::kValue4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma() ||
      parser.parseOperand(acc) || parser.parseComma()) {
    return failure();
  }
  if (failed(parseCommRecvClause(parser, recvClause))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue2> fixedOperands{dst, acc};
  SmallVector<Type, mlir::pto::kValue2> fixedTypes(mlir::pto::kValue2);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, 1, recvClause.pong ? 1 : 0},
          {"reduceOp", "root"}))) {
    return failure();
  }
  return success();
}

void mlir::pto::TReduceOp::print(OpAsmPrinter &p) {
  p << "(" << getDst() << ", " << getAcc() << ", ";
  printCommRecvClause(p, getRecvPing(), getRecvPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getDst().getType() << ", " << getAcc().getType() << ", "
    << getRecvPing().getType();
  if (getRecvPong()) {
    p << ", " << getRecvPong().getType();
  }
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand ptr;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> shapeOps;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> strideOps;

  Type resultTy;

  // %ptr
  if (parser.parseOperand(ptr)) {
    return failure();
  }

  // , shape = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(shapeOps) ||
      parser.parseRSquare()) {
    return failure();
  }

  // strides = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(strideOps) ||
      parser.parseRSquare()) {
    return failure();
  }

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // : result-type
  if (parser.parseColonType(resultTy)) {
    return failure();
  }
  result.addTypes(resultTy);

  auto tvTy = llvm::dyn_cast<mlir::pto::TensorViewType>(resultTy);
  if (!tvTy) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected result type pto.tensor_view<...>");
  }

  Type elemTy = tvTy.getElementType();

  Type ptrTy = mlir::pto::PtrType::get(parser.getContext(), elemTy);
  // resolve %ptr
  if (parser.resolveOperand(ptr, ptrTy, result.operands)) {
    return failure();
  }

  // resolve shape/strides 为 index
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(shapeOps, indexTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(strideOps, indexTy, result.operands)) {
    return failure();
  }

  auto segAttr = parser.getBuilder().getDenseI32ArrayAttr(
      {1, (int32_t)shapeOps.size(), (int32_t)strideOps.size()});
  result.addAttribute("operandSegmentSizes", segAttr);

  return success();
}

void mlir::pto::MakeTensorViewOp::print(OpAsmPrinter &p) {
  p << " " << getPtr();

  p << ", shape = [";
  p.printOperands(getShape());
  p << "]";

  p << ", strides = [";
  p.printOperands(getStrides());
  p << "]";

  p.printOptionalAttrDict((*this)->getAttrs(),
                        /*elidedAttrs=*/{"operandSegmentSizes"});

  p << " : " << getResult().getType();
}

// Layout inference helpers for make_tensor_view
static std::optional<int64_t> getConstIndexValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }
  return std::nullopt;
}

static FailureOr<mlir::pto::PartitionTensorViewType>
inferPartitionViewResultTypeFromSizes(Type sourceType, ValueRange sizes) {
  int64_t sourceRank = 0;
  Type elementType;
  Attribute layout;
  if (auto tensorView = dyn_cast<mlir::pto::TensorViewType>(sourceType)) {
    sourceRank = tensorView.getRank();
    elementType = tensorView.getElementType();
    layout = tensorView.getLayout();
  } else if (auto partitionView =
                 dyn_cast<mlir::pto::PartitionTensorViewType>(sourceType)) {
    sourceRank = partitionView.getRank();
    elementType = partitionView.getElementType();
    layout = partitionView.getLayout();
  } else {
    return failure();
  }

  if ((int64_t)sizes.size() != sourceRank) {
    return failure();
  }

  SmallVector<int64_t, mlir::pto::kValue4> shape;
  shape.reserve(sizes.size());
  for (Value size : sizes) {
    auto constSize = getConstIndexValue(size);
    if (constSize && *constSize >= 0) {
      shape.push_back(*constSize);
    } else {
      shape.push_back(ShapedType::kDynamic);
}
  }

  return mlir::pto::PartitionTensorViewType::get(sourceType.getContext(), shape,
                                                 elementType, layout);
}

namespace {
struct PartitionViewParseState {
    OpAsmParser::UnresolvedOperand source;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> sizes;
    Type sourceTy;
    Type resultTy;
    bool hasExplicitResultTy = false;
};

static ParseResult parseOptionalResultType(OpAsmParser &parser, Type &resultTy,
                                           bool &hasExplicitResultTy) {
  if (failed(parser.parseOptionalArrow()))
    return success();
  if (parser.parseType(resultTy))
    return failure();
  hasExplicitResultTy = true;
  return success();
}

static ParseResult parsePartitionViewSyntax(OpAsmParser& parser, OperationState& result, PartitionViewParseState& state)
{
    if (parser.parseOperand(state.source) || parser.parseComma() || parser.parseKeyword("offsets") ||
        parser.parseEqual() || parser.parseLSquare() || parser.parseOperandList(state.offsets) ||
        parser.parseRSquare() || parser.parseComma() || parser.parseKeyword("sizes") || parser.parseEqual() ||
        parser.parseLSquare() || parser.parseOperandList(state.sizes) || parser.parseRSquare() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(state.sourceTy)) {
        return failure();
    }
    return parseOptionalResultType(parser, state.resultTy,
                                   state.hasExplicitResultTy);
}
