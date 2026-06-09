// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

[[maybe_unused]] static LogicalResult verifyMatmulCommon(Operation *op, Value lhs,
                                                         Value rhs, Value biasOpt,
                                                         Type maybeDstElemTy,
                                                         Type maybeResultElemTy) {
  if (auto lhsTy = dyn_cast<ShapedType>(lhs.getType())) {
    return verifyMatmulShapedCommon(op, lhsTy, rhs, biasOpt, maybeDstElemTy,
                                    maybeResultElemTy);
  }
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  if (!lhsTile) {
    return op->emitOpError(
        "expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");
  }
  return verifyMatmulTileCommon(op, lhsTile, rhs, biasOpt, maybeDstElemTy,
                                maybeResultElemTy);
}

using VerifyMatTileOperandsFn = LogicalResult (*)(Operation *, Type, Type, Type);

static LogicalResult verifyMatmulLikeTileOp(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy,
                                            VerifyMatTileOperandsFn verifyOperands) {
  if (failed(verifyOperands(op, lhsTy, rhsTy, dstTy)))
    return failure();
  if (failed(verifyMatmulTypeTriple(op, getElemTy(lhsTy), getElemTy(rhsTy),
                                    getElemTy(dstTy))))
    return failure();
  return verifyMatmulLike(op, lhsTy, rhsTy, dstTy);
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyMatmulLikeTileOp(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), verifyMatTileOperands);
  };
  auto verifyA5 = [&verifyA2A3]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyMatmulLikeTileOp(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), verifyGemvTileOperands);
  };
  auto verifyA5 = [&verifyA2A3]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                   getDst().getType())))
    return failure();
  return success();
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < kNumber2)
    return mlir::Type();

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile)
    return mlir::Type();

  Type elemTy = lhsTile.getElementType();
  if (operands.size() >= kNumber3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= kPTORowColRank &&
      rhsShape.size() >= kPTORowColRank) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    SmallVec2<int64_t> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < kNumber2)
    return RankedTensorType();

  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  auto rhsTy = dyn_cast<ShapedType>(operands[1].getType());
  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return RankedTensorType();

  Type elemTy = lhsTy.getElementType();
  if (operands.size() >= kNumber3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType()))
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    if (auto biasMR = dyn_cast<MemRefType>(operands[2].getType())) {
      if (biasMR.hasStaticShape())
        return RankedTensorType::get(biasMR.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= kPTORowColRank &&
      rhsTy.getRank() >= kPTORowColRank) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty())
    return RankedTensorType();
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType()))
    return accRT;
  return RankedTensorType();
}

namespace mlir {
namespace pto {

static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess())
    return failure();
  if (parser.parseDimensionList(shape, allowDynamic))
    return failure();
  if (parser.parseType(elementType))
    return failure();
  if (parser.parseGreater())
    return failure();
  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic)
      printer << "?";
    else
      printer << d;
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &odsParser) {
  SmallVec4<int64_t> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(odsParser, shape, elemTy,
                               /*allowDynamic=*/true)))
    return Type();
  return PartitionTensorViewType::get(odsParser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &odsPrinter) const {
  printShapeAndElem(odsPrinter, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &odsParser) {
  SmallVec4<int64_t> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(odsParser, shape, elemTy,
                               /*allowDynamic=*/true)))
    return Type();
  return TileType::get(odsParser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &odsPrinter) const {
  printShapeAndElem(odsPrinter, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &odsParser) {
  SmallVec4<int64_t> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(odsParser, shape, elemTy,
                               /*allowDynamic=*/false)))
    return Type();
  return LocalArrayType::getChecked(
      [&odsParser]() { return odsParser.emitError(odsParser.getNameLoc()); },
      odsParser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &odsPrinter) const {
  printShapeAndElem(odsPrinter, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty())
    return emitError() << "'!pto.local_array' requires at least one dimension";
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0)
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
  }
  if (!elementType.isIntOrFloat())
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  return success();
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
