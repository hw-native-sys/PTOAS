// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true)))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType())))
      return failure();
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true)))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
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
  if (operands.size() < 2)
    return mlir::Type();

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile)
    return mlir::Type();

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    llvm::SmallVector<int64_t, 2> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2)
    return RankedTensorType();

  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  auto rhsTy = dyn_cast<ShapedType>(operands[1].getType());
  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return RankedTensorType();

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType()))
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    if (auto biasMR = dyn_cast<MemRefType>(operands[2].getType())) {
      if (biasMR.hasStaticShape())
        return RankedTensorType::get(biasMR.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
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

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();

  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
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
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false)))
    return Type();
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
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (add.getKind() == AffineExprKind::Add) {
      flattenAddExpr(add.getLHS(), terms);
      flattenAddExpr(add.getRHS(), terms);
      return;
    }
  }
  terms.push_back(expr);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);

  if (map.getNumResults() != 1) return;

  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    // 情况 A: dN * Const 或 Const * dN
    if (auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term)) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();

        // 尝试匹配 LHS=Dim, RHS=Const
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(lhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(rhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }

        // 尝试匹配 LHS=Const, RHS=Dim (乘法交换律)
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(rhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(lhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
      }
    }
    // 情况 B: 单独的 dN (隐含 Stride = 1)
    else if (auto dim = llvm::dyn_cast<AffineDimExpr>(term)) {
      strides[dim.getPosition()] = 1;
    }
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures that the order of AffineExpr addition is:
//   0 + (d0*str0 + d1*str1...) + (s0*str0 + s1*str1...)
// This guarantees bitwise-identical AffineMaps for verification.
static AffineMap buildStrictBitwiseAffineMap(MLIRContext *ctx,
                                             ArrayRef<int64_t> strides,
                                             bool isMultiDimSymbol) {
  unsigned rank = strides.size();

  // Step 1: Initialize with Constant(0)
  AffineExpr totalExpr = getAffineConstantExpr(0, ctx);

  // Step 2: Add Dimensions (d0*str0 + d1*str1...)
  // Strictly in order: 0, 1, 2...
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = getAffineDimExpr(i, ctx);
    auto str = getAffineConstantExpr(strides[i], ctx);
    totalExpr = totalExpr + (dim * str);
  }

  // Step 3: Add Symbols (s0*str0 + s1*str1...)
  // Strictly in order: 0, 1, 2...
  if (isMultiDimSymbol) {
    for (unsigned i = 0; i < rank; ++i) {
      auto sym = getAffineSymbolExpr(i, ctx);
      auto str = getAffineConstantExpr(strides[i], ctx);
      totalExpr = totalExpr + (sym * str);
    }
  }
  // (Optional: handle single dynamic offset case if needed, omitted for clarity)

  // numSymbols is rank if multi-dim (for offsets), else 0
  unsigned numSymbols = isMultiDimSymbol ? rank : 0;
  return AffineMap::get(rank, numSymbols, totalExpr);
}


// =============================================================================
// Parser Implementation
// =============================================================================

// Helper for parsing [64, 1]
static ParseResult parseStrideList(AsmParser &parser, SmallVectorImpl<int64_t> &strides) {
  if (parser.parseLSquare()) return failure();
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) return failure();
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) return failure();
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
[[maybe_unused]] static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) return failure();

  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) return failure();

  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) return failure();

    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) return failure();
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) return failure();
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }

  if (parser.parseGreater()) return failure();

  // 3. Validation
  if (isMultiDim && numSymbols != strides.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Number of offset symbols must match rank");
  }

  // 4. [CALL SHARED BUILDER]
  // Delegate to the strict builder
  MLIRContext *ctx = parser.getContext();
  AffineMap map = buildStrictBitwiseAffineMap(ctx, strides, isMultiDim);

  layout = AffineMapAttr::get(map);
  return success();
}

// =============================================================================
// Printer Implementation
// =============================================================================

[[maybe_unused]] static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) return;
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) return;

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
      if (i < numSyms - 1) printer << ", ";
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

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperandList(offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes"))
    return failure();

  ArrayAttr sizesAttr;
  if (parser.parseAttribute(sizesAttr, "sizes", result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("valid"))) {
    OpAsmParser::UnresolvedOperand vrow, vcol;
    if (parser.parseLSquare() || parser.parseOperand(vrow) || parser.parseComma() ||
        parser.parseOperand(vcol) || parser.parseRSquare())
      return failure();
    valids.push_back(vrow);
    valids.push_back(vcol);
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy))
    return failure();

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultTy))
      return failure();
    hasExplicitResultTy = true;
  }

  if (parser.resolveOperand(source, sourceTy, result.operands))
    return failure();

  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsets, indexTy, result.operands))
    return failure();
  if (!valids.empty() &&
      parser.resolveOperands(valids, indexTy, result.operands))
    return failure();

  int32_t hasValid = valids.empty() ? 0 : 1;
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(offsets.size()), hasValid, hasValid}));

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
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
  if (lhs.size() != rhs.size())
    return false;
  for (auto [inferred, declared] : llvm::zip(lhs, rhs)) {
    if (inferred == declared)
      continue;
    auto inferredTb = dyn_cast<TileBufType>(inferred);
    auto declaredTb = dyn_cast<TileBufType>(declared);
    if (!inferredTb || !declaredTb)
      return false;
    if (inferredTb.getShape() != declaredTb.getShape() ||
        inferredTb.getElementType() != declaredTb.getElementType() ||
        inferredTb.getMemorySpace() != declaredTb.getMemorySpace() ||
        inferredTb.getConfigAttr() != declaredTb.getConfigAttr())
      return false;
    auto inferredValid = inferredTb.getValidShape();
    auto declaredValid = declaredTb.getValidShape();
    if (inferredValid.size() != declaredValid.size())
      return false;
    for (auto [inferredDim, declaredDim] : llvm::zip(inferredValid, declaredValid)) {
      // Any static declared valid extent is accepted in place of the inferred
      // one; only a dynamic declared valid that disagrees is incompatible.
      if (inferredDim != declaredDim && declaredDim == ShapedType::kDynamic)
        return false;
    }
  }
  return true;
}

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // 1. 获取 Source Type
  if (operands.empty()) return failure();
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType) return failure();

  // 2. 获取 subview 逻辑窗口（sizes）
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop) sizeAttr = prop->sizes;
  }
  if (!sizeAttr && attributes) {
    sizeAttr = attributes.getAs<ArrayAttr>("sizes");
  }
  if (!sizeAttr) return failure();

  SmallVector<int64_t> subviewShape;
  for (auto attr : sizeAttr) {
    int64_t dim = llvm::cast<IntegerAttr>(attr).getInt();
    subviewShape.push_back(dim);
  }

  // Design: subview 的结果 tile 类型显式表达逻辑子窗口 shape（sizes）。
  ArrayRef<int64_t> parentShape = sourceType.getShape();
  if (subviewShape.size() != parentShape.size())
    return failure();

  // Derive valid shape from explicit valid_row/valid_col when provided.
  // Otherwise default to subview shape (no parent valid-shape inheritance).
  SmallVector<int64_t> validShape;
  constexpr int64_t kDynamicValidDim = -1;
  int64_t rank = static_cast<int64_t>(subviewShape.size());
  Value explicitVRow;
  Value explicitVCol;

  // Robustly decode optional valid operands using AttrSizedOperandSegments:
  //   [source, offsets..., valid_row?, valid_col?]
  if (attributes) {
    if (auto segAttr =
            attributes.getAs<DenseI32ArrayAttr>("operandSegmentSizes")) {
      ArrayRef<int32_t> segs = segAttr.asArrayRef();
      if (segs.size() == 4) {
        int32_t srcSeg = segs[0];
        int32_t offSeg = segs[1];
        int32_t vRowSeg = segs[2];
        int32_t vColSeg = segs[3];
        if (srcSeg == 1 && offSeg >= 0 && (vRowSeg == 0 || vRowSeg == 1) &&
            (vColSeg == 0 || vColSeg == 1)) {
          size_t idx = static_cast<size_t>(srcSeg + offSeg);
          if (vRowSeg == 1 && idx < operands.size())
            explicitVRow = operands[idx++];
          if (vColSeg == 1 && idx < operands.size())
            explicitVCol = operands[idx];
        }
      }
    }
  }

  // Fallback for legacy callers that may not provide operandSegmentSizes.
  if (!explicitVRow && !explicitVCol && rank == 2) {
    size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
    if (operands.size() >= expectedWithoutValid + 2) {
      explicitVRow = operands[expectedWithoutValid];
      explicitVCol = operands[expectedWithoutValid + 1];
    }
  }

  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = (i == 0) ? explicitVRow : (i == 1 ? explicitVCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : kDynamicValidDim;
    }
    validShape.push_back(vdim);
  }

  // 3. 继承 Config (若为空使用默认)
  auto cfg = sourceType.getConfigAttr();
  if (!cfg) cfg = TileBufConfigAttr::getDefault(context);

  // 4. 构建 Result Type
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto resultType = TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg);

  inferredReturnTypes.push_back(resultType);
  return success();
}

// =============================================================================
// SubViewOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndex(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndex(truncOp.getIn(), out);
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  auto readBLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<BLayoutAttr>(attr)) {
      out = (int32_t)a.getValue();
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = (int32_t)a.getInt();
      return true;
    }
    return false;
  };
  auto readSLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<SLayoutAttr>(attr)) {
      out = (int32_t)a.getValue();
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = (int32_t)a.getInt();
      return true;
    }
    return false;
  };
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) fr = (int32_t)attr.getInt();

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0) return failure();

  if (fr == 1024) {
    innerRows = 16;
    innerCols = 16;
    return success();
  }
  if (fr == 32) {
    innerRows = 16;
    innerCols = 2;
    return success();
  }
  if (fr == 512) {
    if (sl == 1) {
      innerRows = 16;
      innerCols = 32 / elemBytes;
      return success();
    }
    if (sl == 2) {
      innerRows = 32 / elemBytes;
      innerCols = 16;
      return success();
    }
  }
  return failure();
}

static LogicalResult
computeExpectedTileBufMemrefStrides(TileBufType tileTy,
                                    SmallVectorImpl<int64_t> &expectedStrides) {
  if (tileTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = tileTy.getShape();
  if (shape.size() != 2)
    return failure();
  if (shape[0] == ShapedType::kDynamic || shape[1] == ShapedType::kDynamic)
    return failure();

  auto cfg = tileTy.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(tileTy.getContext());

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, tileTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl)))
    return failure();

  expectedStrides.clear();
  if (!boxed) {
    if (bl == 1) {
      expectedStrides.push_back(1);
      expectedStrides.push_back(shape[0]);
    } else {
      expectedStrides.push_back(shape[1]);
      expectedStrides.push_back(1);
    }
    return success();
  }

  if (bl == 1) {
    if (sl != 1)
      return failure();
    expectedStrides.push_back(innerCols);
    expectedStrides.push_back(shape[0]);
    return success();
  }

  expectedStrides.push_back(shape[1]);
  expectedStrides.push_back(innerRows);
  return success();
}

mlir::LogicalResult mlir::pto::SimdTileToMemrefOp::verify() {
  auto memTy = dyn_cast<MemRefType>(getDst().getType());
  if (!memTy)
    return emitOpError("expects result to be memref");

  Type srcTy = getSrc().getType();
  if (auto tileTy = dyn_cast<TileBufType>(srcTy)) {
    if (memTy.getElementType() != tileTy.getElementType())
      return emitOpError(
          "expects memref element type to match tile_buf element type");

    if (memTy.getMemorySpace() != tileTy.getMemorySpace())
      return emitOpError(
          "expects memref memory space to match tile_buf memory space");

    if (memTy.getRank() != tileTy.getRank())
      return emitOpError("expects memref rank to match tile_buf rank");

    ArrayRef<int64_t> tileShape = tileTy.getShape();
    ArrayRef<int64_t> validShape = tileTy.getValidShape();
    ArrayRef<int64_t> memShape = memTy.getShape();
    if (tileShape.size() != memShape.size())
      return emitOpError(
          "expects memref shape rank to match tile_buf shape rank");

    if (validShape.size() != memShape.size())
      return emitOpError(
          "expects tile_buf valid shape rank to match memref shape rank");

    for (unsigned i = 0; i < validShape.size(); ++i) {
      int64_t expect = validShape[i];
      if (expect < 0) {
        if (memShape[i] >= 0 && memShape[i] != tileShape[i]) {
          return emitOpError()
                 << "expects memref dim " << i
                 << " to be dynamic or match physical tile dim " << tileShape[i]
                 << " because tile_buf valid dim is ?";
        }
        continue;
      }

      if (memShape[i] != expect) {
        return emitOpError() << "expects memref dim " << i
                             << " to match tile_buf valid dim; got "
                             << memShape[i] << ", expected " << expect;
      }
    }

    SmallVector<int64_t, 4> expectedStrides;
    if (failed(computeExpectedTileBufMemrefStrides(tileTy, expectedStrides)))
      return emitOpError("cannot infer expected strides from tile_buf layout");

    SmallVector<int64_t, 4> memStrides;
    int64_t memOffset = ShapedType::kDynamic;
    if (failed(getStridesAndOffset(memTy, memStrides, memOffset)))
      return emitOpError("expects memref to use strided layout");
    if (memOffset != 0)
      return emitOpError("expects memref offset to be 0");
    if (memStrides.size() != expectedStrides.size())
      return emitOpError("expects memref stride rank to match tile_buf rank");
    for (unsigned i = 0; i < expectedStrides.size(); ++i) {
      if (memStrides[i] != expectedStrides[i]) {
        return emitOpError()
               << "expects memref strides to match tile_buf layout; got "
               << memStrides[i] << " at dim " << i << ", expected "
               << expectedStrides[i];
      }
    }
    return success();
  }

  auto srcMemTy = dyn_cast<MemRefType>(srcTy);
  if (!srcMemTy)
    return emitOpError("expects src to be !pto.tile_buf or memref");

  if (srcMemTy.getElementType() != memTy.getElementType())
    return emitOpError("expects src/result memref element types to match");

  if (srcMemTy.getMemorySpace() != memTy.getMemorySpace())
    return emitOpError("expects src/result memref memory spaces to match");

  if (srcMemTy.getRank() != memTy.getRank())
    return emitOpError("expects src/result memref ranks to match");

  ArrayRef<int64_t> srcShape = srcMemTy.getShape();
  ArrayRef<int64_t> dstShape = memTy.getShape();
  for (unsigned i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] >= 0 && dstShape[i] >= 0 && srcShape[i] != dstShape[i]) {
      return emitOpError()
             << "expects compatible src/result memref shapes; dim " << i
             << " mismatches (" << srcShape[i] << " vs " << dstShape[i] << ")";
    }
  }

  return success();
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 tilebuf for src/dst");

  auto sizesAttr = getSizes();
  if (!sizesAttr || sizesAttr.size() != 2)
    return emitOpError("subview expects 2D sizes");
  int64_t sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (sizeR <= 0 || sizeC <= 0)
    return emitOpError("subview sizes must be positive");
  if (getOffsets().size() != 2)
    return emitOpError("subview expects 2D offsets");

  int64_t offR = 0, offC = 0;
  bool offRConst = getConstIndex(getOffsets()[0], offR);
  bool offCConst = getConstIndex(getOffsets()[1], offC);
  if (offRConst && offR < 0)
    return emitOpError("subview offsets must be non-negative");
  if (offCConst && offC < 0)
    return emitOpError("subview offsets must be non-negative");

  bool hasValidRow = static_cast<bool>(getValidRow());
  bool hasValidCol = static_cast<bool>(getValidCol());
  if (hasValidRow != hasValidCol)
    return emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");

  if (hasValidRow) {
    int64_t vRow = 0, vCol = 0;
    if (getConstIndex(getValidRow(), vRow)) {
      if (vRow < 0)
        return emitOpError("valid_row must be non-negative when constant");
      if (vRow > sizeR)
        return emitOpError("valid_row must be <= subview row size");
    }
    if (getConstIndex(getValidCol(), vCol)) {
      if (vCol < 0)
        return emitOpError("valid_col must be non-negative when constant");
      if (vCol > sizeC)
        return emitOpError("valid_col must be <= subview col size");
    }
  }

  auto dstShape = dstTy.getShape();
  if (dstShape.size() != 2)
    return emitOpError("expects result to be rank-2");
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2)
    return emitOpError("expects source to be rank-2");
  if (dstShape[0] != sizeR || dstShape[1] != sizeC)
    return emitOpError("expects result shape to match subview sizes");

  if (dstTy.getElementType() != srcTy.getElementType())
    return emitOpError("expects result element type to match source");
  if (dstTy.getMemorySpace() != srcTy.getMemorySpace())
    return emitOpError("expects result address space to match source");
  auto srcCfg = srcTy.getConfigAttr();
  if (!srcCfg) srcCfg = TileBufConfigAttr::getDefault(getContext());
  auto dstCfg = dstTy.getConfigAttr();
  if (!dstCfg) dstCfg = TileBufConfigAttr::getDefault(getContext());
  if (dstCfg != srcCfg)
    return emitOpError("expects result tile config to match source");

  // Design choice: when valid[...] is omitted, infer result valid_shape from
  // subview sizes directly. We intentionally do not constrain it by source
  // valid_shape to allow user-controlled subview semantics.

  auto expectedValidDim = [&](Value explicitValid, int64_t defaultSize) {
    if (!explicitValid)
      return defaultSize;
    int64_t c = 0;
    if (getConstIndex(explicitValid, c))
      return std::min<int64_t>(c, defaultSize);
    return ShapedType::kDynamic;
  };
  int64_t expectedVRow = expectedValidDim(getValidRow(), sizeR);
  int64_t expectedVCol = expectedValidDim(getValidCol(), sizeC);
  auto dstValid = dstTy.getValidShape();
  if (dstValid.size() != 2)
    return emitOpError("expects result to have rank-2 valid_shape");
  // With the valid operand omitted, the result type is authoritative for the
  // valid extent: accept any static value in [0, size] (this subsumes both the
  // full-size default and the v=0 no-op-replay empty marker). Lowering derives
  // the bind_tile valid operand from this type. A dynamic result valid still
  // requires an explicit operand to supply the runtime extent, so it stays
  // rejected on this path.
  bool rowInferred = !getValidRow() && dstValid[0] != ShapedType::kDynamic &&
                     dstValid[0] >= 0 && dstValid[0] <= sizeR;
  bool colInferred = !getValidCol() && dstValid[1] != ShapedType::kDynamic &&
                     dstValid[1] >= 0 && dstValid[1] <= sizeC;
  if (dstValid[0] != expectedVRow && !rowInferred)
    return emitOpError("expects result valid_shape[0] to match inferred/explicit valid_row");
  if (dstValid[1] != expectedVCol && !colInferred)
    return emitOpError("expects result valid_shape[1] to match inferred/explicit valid_col");

  auto cfg = srcTy.getConfigAttr();
  if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl)))
    return emitOpError("unsupported tile layout for subview");

  if (!boxed)
    return success();

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  if (sizeR % innerRows != 0 || sizeC % innerCols != 0)
    return emitOpError("boxed layout subview sizes must be multiples of inner shape");

  if (offRConst) {
    if (offR % innerRows != 0)
      return emitOpError("boxed layout subview offsets must be multiples of inner shape");
  }
  if (offCConst) {
    if (offC % innerCols != 0)
      return emitOpError("boxed layout subview offsets must be multiples of inner shape");
  }

  (void)bl;
  if (srcShape.size() != 2 ||
      srcShape[0] == ShapedType::kDynamic ||
      srcShape[1] == ShapedType::kDynamic) {
    return emitOpError("boxed layout subview requires static source shape");
  }

  return success();
}

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

// =============================================================================
// Helper Functions
// =============================================================================

[[maybe_unused]] static AddressSpace getAddressSpace(Value val) {
  auto type = llvm::dyn_cast<MemRefType>(val.getType());
  if (!type) return AddressSpace::Zero; // Default

  // 假设你的 AddressSpaceAttr 存储在 MemRef 的 memorySpace 中
  // 需要根据你的 getPTOAddressSpaceAttr 实现来调整
  auto attr = llvm::dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (attr) return attr.getAddressSpace();
  return AddressSpace::Zero;
}

// =============================================================================
// Side Effects Implementation
// =============================================================================

// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value

// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand)
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
}

// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result)
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

void TPrefetchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty())
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty())
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty())
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_QUATERNARY_EFFECTS(OpClass, op0, op1, op2, op3, dstOperand)      \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_READ(op3);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  // GM -> L1 Elem mode stages the gathered elements into the GM scratch buffer
  // before the bulk copy: the op clobbers scratch, so model it as a write.
  auto scratchRange = getScratchMutable();
  if (!scratchRange.empty())
    addEffect(effects, &*scratchRange.begin(), MemoryEffects::Write::get());
}

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

void THistogramOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TGetScaleAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// SET_VALIDSHAPE: update runtime valid row/col metadata on source tile in-place.
void SetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getSourceMutable());
}

// GET_VALIDSHAPE: read runtime valid row/col metadata from source tile.
void GetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSourceMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TAxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TConcatOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_QUATERNARY_EFFECTS(TConcatidxOp, getSrc0Mutable(), getSrc1Mutable(), getSrc0IdxMutable(), getSrc1IdxMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TTRI: Write(dst) (generates triangular mask)
void TTriOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandExpdifOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColProdOp, getSrcMutable(), getDstMutable())

void TColArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TColArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TCvtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty())
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// TEXTRACT_FP: Read(src), Read(fp) -> Write(dst)
void TExtractFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT_FP: Read(src), Read(fp) -> Write(dst)
void TInsertFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadInplaceOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty())
    PTO_ADD_WRITE(cdst[0]);
  if (auto indices = getIndicesMutable(); !indices.empty())
    PTO_ADD_READ(indices[0]);
  if (auto tmp = getTmpMutable(); !tmp.empty())
    PTO_ADD_READ(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMovFPOp, getSrcMutable(), getFpMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TPartArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
void TPartArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty())
    PTO_ADD_READ(offsetRange[0]);
  auto tmpRange = getTmpMutable();
  if (!tmpRange.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmpRange[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getExpMutable());
  PTO_ADD_WRITE(getMaxMutable());
  PTO_ADD_WRITE(getScalingMutable());
  auto expZzRange = getExpZzMutable();
  if (!expZzRange.empty())
    PTO_ADD_WRITE(expZzRange[0]);
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
void TRemOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRemSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TPowOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getBaseMutable());
  PTO_ADD_READ(getExpMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TPowSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

void TRowExpandDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}
