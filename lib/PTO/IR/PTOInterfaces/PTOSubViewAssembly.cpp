// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

// =============================================================================
// Printer Implementation
// =============================================================================

[[maybe_unused]] static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) {
    return;
  }
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) {
    return;
  }

  // 1. [核心修改] 反解 Strides
  SmallVector<int64_t> strides;
  decomposeStridedLayout(map, strides);

  printer << ", strided<[";
  // 2. 打印真实的 strides
  llvm::interleaveComma(strides, printer);
  printer << "]";

  // Print Offset: [?, ?]
  unsigned numSyms = map.getNumSymbols();
  if (numSyms > 0) {
    printer << ", offset: [";
    for (unsigned i = 0; i < numSyms; ++i) {
      printer << "?";
      if (i < numSyms - 1) {
        printer << ", ";
      }
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subview 相关实现

// =============================================================================
// Op Interface Implementation: SubViewOp
// =============================================================================

struct SubViewParseState {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue2> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;
};

static ParseResult parseSubViewSyntax(OpAsmParser &parser,
                                     OperationState &result,
                                     SubViewParseState &state) {
  if (parser.parseOperand(state.source) || parser.parseLSquare() ||
      parser.parseOperandList(state.offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes")) {
    return failure();
  }

  ArrayAttr sizesAttr;
  if (parser.parseAttribute(sizesAttr, "sizes", result.attributes)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("valid"))) {
    OpAsmParser::UnresolvedOperand vrow, vcol;
    if (parser.parseLSquare() || parser.parseOperand(vrow) || parser.parseComma() ||
        parser.parseOperand(vcol) || parser.parseRSquare()) {
      return failure();
    }
    state.valids.push_back(vrow);
    state.valids.push_back(vcol);
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(state.sourceTy)) {
    return failure();
  }

  return parseOptionalResultType(parser, state.resultTy,
                                 state.hasExplicitResultTy);
}

static ParseResult resolveSubViewOperands(OpAsmParser &parser,
                                          OperationState &result,
                                          SubViewParseState &state) {
  if (parser.resolveOperand(state.source, state.sourceTy, result.operands)) {
    return failure();
  }
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(state.offsets, indexTy, result.operands)) {
    return failure();
  }
  if (!state.valids.empty() &&
      parser.resolveOperands(state.valids, indexTy, result.operands)) {
    return failure();
  }
  int32_t hasValid = state.valids.empty() ? 0 : 1;
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(state.offsets.size()), hasValid, hasValid}));
  return success();
}

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SubViewParseState state;
  if (failed(parseSubViewSyntax(parser, result, state)) ||
      failed(resolveSubViewOperands(parser, result, state)))
    return failure();

  if (state.hasExplicitResultTy) {
    result.addTypes(state.resultTy);
    return success();
  }

  SmallVector<Type> inferredReturnTypes;
  DictionaryAttr attrs = result.attributes.getDictionary(parser.getContext());
  if (failed(SubViewOp::inferReturnTypes(
          parser.getContext(), std::nullopt, result.operands, attrs, nullptr,
          RegionRange(), inferredReturnTypes))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.subview result type");
  }
  result.addTypes(inferredReturnTypes);
  return success();
}

void mlir::pto::SubViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << "[";
  printer.printOperands(getOffsets());
  printer << "] sizes " << getSizes();
  if (getValidRow()) {
    printer << " valid [" << getValidRow() << ", " << getValidCol() << "]";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "sizes"});
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

// The inferred result type derives valid_shape from `sizes` (or the explicit
// valid operands). With the operand omitted the result type is authoritative for
// the valid extent (any static value, including the v=0 no-op-replay marker or a
// partial valid), so accept a static declared valid that differs from the
// size-inferred one here; SubViewOp::verify() enforces the precise per-path rule
// (operand clamping vs the [0, size] range). Only a dynamic declared valid that
// disagrees with the inferred extent is incompatible -- it needs an explicit
// operand to supply the runtime value. Every other difference (shape, element
// type, address space, config) is still rejected as the default check would.
bool SubViewOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [inferred, declared] : llvm::zip(lhs, rhs)) {
    if (inferred == declared) {
      continue;
    }
    auto inferredTb = dyn_cast<TileBufType>(inferred);
    auto declaredTb = dyn_cast<TileBufType>(declared);
    if (!inferredTb || !declaredTb) {
      return false;
    }
    if (inferredTb.getShape() != declaredTb.getShape() ||
        inferredTb.getElementType() != declaredTb.getElementType() ||
        inferredTb.getMemorySpace() != declaredTb.getMemorySpace() ||
        inferredTb.getConfigAttr() != declaredTb.getConfigAttr()) {
      return false;
    }
    auto inferredValid = inferredTb.getValidShape();
    auto declaredValid = declaredTb.getValidShape();
    if (inferredValid.size() != declaredValid.size()) {
      return false;
    }
    for (auto [inferredDim, declaredDim] : llvm::zip(inferredValid, declaredValid)) {
      // Any static declared valid extent is accepted in place of the inferred
      // one; only a dynamic declared valid that disagrees is incompatible.
      if (inferredDim != declaredDim && declaredDim == ShapedType::kDynamic) {
        return false;
      }
    }
  }
  return true;
}

static ArrayAttr getSubViewSizeAttr(DictionaryAttr attributes,
                                    OpaqueProperties properties) {
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop)
      sizeAttr = prop->sizes;
  }
  if (!sizeAttr && attributes)
    sizeAttr = attributes.getAs<ArrayAttr>("sizes");
  return sizeAttr;
}

static std::pair<Value, Value> getSubViewExplicitValids(
    ValueRange operands, DictionaryAttr attributes, int64_t rank) {
  Value row, col;
  if (attributes) {
    if (auto segAttr = attributes.getAs<DenseI32ArrayAttr>(
            "operandSegmentSizes")) {
      ArrayRef<int32_t> segs = segAttr.asArrayRef();
      if (segs.size() == mlir::pto::kValue4) {
          size_t index = static_cast<size_t>(segs[0] + segs[1]);
          if (segs[0] == 1 && segs[1] >= 0 && segs[mlir::pto::kValue2] == 1 && index < operands.size()) {
              row = operands[index++];
          }
          if (segs[0] == 1 && segs[1] >= 0 && segs[mlir::pto::kValue3] == 1 && index < operands.size()) {
              col = operands[index];
          }
      }
    }
  }
  if (!row && !col && rank == mlir::pto::kValue2) {
      size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
      if (operands.size() >= expectedWithoutValid + mlir::pto::kValue2) {
          row = operands[expectedWithoutValid];
          col = operands[expectedWithoutValid + 1];
      }
  }
  return {row, col};
}

static SmallVector<int64_t> getSubViewValidShape(
    ArrayRef<int64_t> subviewShape, Value explicitRow, Value explicitCol) {
  SmallVector<int64_t> validShape;
  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = i == 0 ? explicitRow : (i == 1 ? explicitCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : -1;
    }
    validShape.push_back(vdim);
  }
  return validShape;
}
