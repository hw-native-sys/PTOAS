// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType()))) {
      return failure();
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
    if (operands.size() < mlir::pto::kValue2) {
        return mlir::Type();
    }

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile) {
    return mlir::Type();
  }

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= mlir::pto::kValue2 && rhsShape.size() >= mlir::pto::kValue2) {
      int64_t M = lhsShape[0];
      int64_t N = rhsShape[1];
      llvm::SmallVector<int64_t, mlir::pto::kValue2> outShape = {M, N};
      return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
    if (operands.size() < mlir::pto::kValue2) {
        return RankedTensorType();
    }

  auto lhsTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhsTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!lhsTy || !rhsTy) {
    return RankedTensorType();
  }

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType())) {
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= mlir::pto::kValue2 && rhsTy.getRank() >= mlir::pto::kValue2) {
      int64_t M = lhsTy.getDimSize(0);
      int64_t N = rhsTy.getDimSize(1);
      return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty()) {
    return RankedTensorType();
  }
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType())) {
    return accRT;
  }
  return RankedTensorType();
}

namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess()) {
    return failure();
  }

  if (parser.parseDimensionList(shape, allowDynamic)) {
    return failure();
  }

  if (parser.parseType(elementType)) {
    return failure();
  }

  if (parser.parseGreater()) {
    return failure();
  }

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << d;
}
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

static LogicalResult parseViewShapeElemAndLayout(
    AsmParser &parser, SmallVectorImpl<int64_t> &shape, Type &elementType,
    Attribute &layout, bool allowDynamic) {
  bool parseFailed =
      parser.parseLess() || parser.parseDimensionList(shape, allowDynamic) ||
      parser.parseType(elementType);
  if (parseFailed) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    LayoutAttr layoutAttr;
    if (parser.parseAttribute(layoutAttr)) {
      return failure();
    }
    layout = layoutAttr;
  }
  return parser.parseGreater();
}

static void printViewShapeElemAndLayout(AsmPrinter &printer,
                                        ArrayRef<int64_t> shape,
                                        Type elementType, Attribute layout) {
  printer << "<";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << dim;
    }
    printer << "x";
  }
  printer.printType(elementType);
  if (layout) {
    printer << ", ";
    printer.printAttribute(layout);
  }
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elemTy;
    Attribute layout;
    if (failed(parseViewShapeElemAndLayout(parser, shape, elemTy, layout, /*allowDynamic=*/true))) {
        return Type();
    }

  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy,
                                      layout);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printViewShapeElemAndLayout(printer, getShape(), getElementType(),
                              getLayout());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elemTy;
    if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true))) {
        return Type();
    }
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &parser) {
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elemTy;
    if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false))) {
        return Type();
    }
  return LocalArrayType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty()) {
    return emitError() << "'!pto.local_array' requires at least one dimension";
  }
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0) {
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
    }
  }
  if (!elementType.isIntOrFloat()) {
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  }
  return success();
}
