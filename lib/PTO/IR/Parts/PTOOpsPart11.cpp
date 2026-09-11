// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

LogicalResult StructType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<Type> fieldTypes) {
  if (fieldTypes.empty()) {
    return emitError() << "'!pto.struct' requires at least one field";
  }
  for (auto [i, f] : llvm::enumerate(fieldTypes)) {
    if (!isStructStorable(f)) {
      return emitError()
             << "'!pto.struct' field " << i << " type " << f
             << " is not scalar-storable; only i8/i16/i32/i64 (signed, "
                "unsigned or signless), f16/bf16/f32/f64, or a nested "
                "!pto.struct are allowed (!pto.local_array cannot be a field "
                "because emitc.member cannot yield an array lvalue; tile_buf / "
                "tensor_view belong to the vec/cube world)";
    }
  }
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

static std::optional<std::pair<unsigned, int64_t>>
matchDimTimesConstant(AffineExpr dimExpr, AffineExpr constantExpr) {
  auto dim = llvm::dyn_cast<AffineDimExpr>(dimExpr);
  auto constant = llvm::dyn_cast<AffineConstantExpr>(constantExpr);
  if (!dim || !constant)
    return std::nullopt;
  return std::make_pair(dim.getPosition(), constant.getValue());
}

static std::optional<std::pair<unsigned, int64_t>>
getStridedLayoutTerm(AffineExpr term) {
  if (auto dim = llvm::dyn_cast<AffineDimExpr>(term))
    return std::make_pair(dim.getPosition(), 1);
  auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term);
  if (!mul || mul.getKind() != AffineExprKind::Mul)
    return std::nullopt;
  if (auto matched = matchDimTimesConstant(mul.getLHS(), mul.getRHS()))
    return matched;
  return matchDimTimesConstant(mul.getRHS(), mul.getLHS());
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);

  if (map.getNumResults() != 1) {
    return;
  }

  // 2. 摊平表达式
  SmallVector<AffineExpr, mlir::pto::kValue4> terms;
  flattenAddExpr(map.getResult(0), terms);

  for (auto term : terms) {
    auto matched = getStridedLayoutTerm(term);
    if (matched)
      strides[matched->first] = matched->second;
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
  if (parser.parseLSquare()) {
    return failure();
  }
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) {
      return failure();
    }
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) {
    return failure();
  }
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
[[maybe_unused]] static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) {
    return failure();
  }

  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) {
    return failure();
  }

  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) {
      return failure();
    }

    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) {
          return failure();
        }
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) {
        return failure();
      }
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }

  if (parser.parseGreater()) {
    return failure();
  }

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

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  auto sourceType = dyn_cast<TileBufType>(operands[0].getType());
  ArrayAttr sizeAttr = getSubViewSizeAttr(attributes, properties);
  if (!sourceType || !sizeAttr)
    return failure();
  SmallVector<int64_t> subviewShape;
  for (Attribute attr : sizeAttr)
    subviewShape.push_back(cast<IntegerAttr>(attr).getInt());
  if (subviewShape.size() != sourceType.getShape().size())
    return failure();
  auto [row, col] = getSubViewExplicitValids(
      operands, attributes, static_cast<int64_t>(subviewShape.size()));
  SmallVector<int64_t> validShape =
      getSubViewValidShape(subviewShape, row, col);

  auto cfg = sourceType.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(context);
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
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndex(castOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndex(truncOp.getIn(), out);
  }
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutValue(cfg.getBLayout(), bl);
  (void)readSLayoutValue(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) {
    fr = static_cast<int32_t>(attr.getInt());
  }

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0) {
    return failure();
  }

  if (fr == mlir::pto::kValue1024) {
      innerRows = mlir::pto::kValue16;
      innerCols = mlir::pto::kValue16;
      return success();
  }
  if (fr == mlir::pto::kValue32) {
      innerRows = mlir::pto::kValue16;
      innerCols = mlir::pto::kValue2;
      return success();
  }
  if (fr == mlir::pto::kValue512) {
      if (sl == 1) {
          innerRows = mlir::pto::kValue16;
          innerCols = mlir::pto::kValue32 / elemBytes;
          return success();
      }
      if (sl == mlir::pto::kValue2) {
          innerRows = mlir::pto::kValue32 / elemBytes;
          innerCols = mlir::pto::kValue16;
          return success();
      }
  }
  return failure();
}

struct SubViewInfo {
  int64_t sizeR = 0, sizeC = 0;
  int64_t offR = 0, offC = 0;
  bool offRConst = false, offCConst = false;
};

static LogicalResult verifySubViewSizesAndOffsets(SubViewOp op,
                                                  SubViewInfo &info) {
  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != mlir::pto::kValue2) {
      return op.emitOpError("subview expects 2D sizes");
  }
  info.sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  info.sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (info.sizeR <= 0 || info.sizeC <= 0) {
    return op.emitOpError("subview sizes must be positive");
  }
  if (op.getOffsets().size() != mlir::pto::kValue2) {
      return op.emitOpError("subview expects 2D offsets");
  }

  info.offRConst = getConstIndex(op.getOffsets()[0], info.offR);
  info.offCConst = getConstIndex(op.getOffsets()[1], info.offC);
  if (info.offRConst && info.offR < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  if (info.offCConst && info.offC < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  return success();
}

static LogicalResult verifySubViewValidBounds(SubViewOp op, int64_t sizeR,
                                              int64_t sizeC) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");
  }

  if (hasValidRow) {
    int64_t vRow = 0, vCol = 0;
    if (getConstIndex(op.getValidRow(), vRow)) {
      if (vRow < 0) {
        return op.emitOpError("valid_row must be non-negative when constant");
      }
      if (vRow > sizeR) {
        return op.emitOpError("valid_row must be <= subview row size");
      }
    }
    if (getConstIndex(op.getValidCol(), vCol)) {
      if (vCol < 0) {
        return op.emitOpError("valid_col must be non-negative when constant");
      }
      if (vCol > sizeC) {
        return op.emitOpError("valid_col must be <= subview col size");
      }
    }
  }
  return success();
}
static LogicalResult verifySubViewShapeAndConfig(SubViewOp op, TileBufType srcTy,
                                                 TileBufType dstTy, int64_t sizeR,
                                                 int64_t sizeC) {
  auto dstShape = dstTy.getShape();
  if (dstShape.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects result to be rank-2");
  }
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects source to be rank-2");
  }
  if (dstShape[0] != sizeR || dstShape[1] != sizeC) {
    return op.emitOpError("expects result shape to match subview sizes");
  }

  if (dstTy.getElementType() != srcTy.getElementType()) {
    return op.emitOpError("expects result element type to match source");
  }
  if (dstTy.getMemorySpace() != srcTy.getMemorySpace()) {
    return op.emitOpError("expects result address space to match source");
  }
  auto srcCfg = srcTy.getConfigAttr();
  if (!srcCfg) {
    srcCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  auto dstCfg = dstTy.getConfigAttr();
  if (!dstCfg) {
    dstCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  if (dstCfg != srcCfg) {
    return op.emitOpError("expects result tile config to match source");
  }
  return success();
}

static LogicalResult verifySubViewValidShape(SubViewOp op, TileBufType dstTy,
                                             int64_t sizeR, int64_t sizeC) {
  // Design choice: when valid[...] is omitted, infer result valid_shape from
  // subview sizes directly. We intentionally do not constrain it by source
  // valid_shape to allow user-controlled subview semantics.

  auto expectedValidDim = [&](Value explicitValid, int64_t defaultSize) {
    if (!explicitValid) {
      return defaultSize;
    }
    int64_t c = 0;
    if (getConstIndex(explicitValid, c)) {
      return std::min<int64_t>(c, defaultSize);
    }
    return ShapedType::kDynamic;
  };
  int64_t expectedVRow = expectedValidDim(op.getValidRow(), sizeR);
  int64_t expectedVCol = expectedValidDim(op.getValidCol(), sizeC);
  auto dstValid = dstTy.getValidShape();
  if (dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects result to have rank-2 valid_shape");
  }
  // With the valid operand omitted, the result type is authoritative for the
  // valid extent: accept any static value in [0, size] (this subsumes both the
  // full-size default and the v=0 no-op-replay empty marker). A dynamic result valid still
  // requires an explicit operand to supply the runtime extent, so it stays
  // rejected on this path.
  bool rowInferred = !op.getValidRow() && dstValid[0] != ShapedType::kDynamic &&
                     dstValid[0] >= 0 && dstValid[0] <= sizeR;
  bool colInferred = !op.getValidCol() && dstValid[1] != ShapedType::kDynamic &&
                     dstValid[1] >= 0 && dstValid[1] <= sizeC;
  if (dstValid[0] != expectedVRow && !rowInferred) {
    return op.emitOpError("expects result valid_shape[0] to match inferred/explicit valid_row");
  }
  if (dstValid[1] != expectedVCol && !colInferred) {
    return op.emitOpError("expects result valid_shape[1] to match inferred/explicit valid_col");
  }
  return success();
}

static LogicalResult verifySubViewBoxed(SubViewOp op, TileBufType srcTy,
                                        const SubViewInfo &info) {
  auto cfg = srcTy.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(op.getContext());
  }

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl))) {
    return op.emitOpError("unsupported tile layout for subview");
  }

  if (!boxed) {
    return success();
  }

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  if (info.sizeR % innerRows != 0 || info.sizeC % innerCols != 0) {
    return op.emitOpError("boxed layout subview sizes must be multiples of inner shape");
  }

  if (info.offRConst) {
    if (info.offR % innerRows != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }
  if (info.offCConst) {
    if (info.offC % innerCols != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }

  (void)bl;
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != mlir::pto::kValue2 || srcShape[0] == ShapedType::kDynamic ||
      srcShape[1] == ShapedType::kDynamic) {
      return op.emitOpError("boxed layout subview requires static source shape");
  }

  return success();
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }
  if (srcTy.getRank() != mlir::pto::kValue2 || dstTy.getRank() != mlir::pto::kValue2) {
      return emitOpError("expects rank-2 tilebuf for src/dst");
  }

  SubViewInfo info;
  if (failed(verifySubViewSizesAndOffsets(*this, info))) {
    return failure();
  }
  if (failed(verifySubViewValidBounds(*this, info.sizeR, info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewShapeAndConfig(*this, srcTy, dstTy, info.sizeR,
                                         info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewValidShape(*this, dstTy, info.sizeR, info.sizeC))) {
    return failure();
  }
  return verifySubViewBoxed(*this, srcTy, info);
}

} // namespace pto
} // namespace mlir

// =============================================================================
// Helper Functions
// =============================================================================

// =============================================================================
// Side Effects Implementation
// =============================================================================

// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value

// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand) {
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
  }
}

using PTOEffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

static void addOptionalEffects(PTOEffectList &effects,
                               mlir::MutableOperandRange operands,
                               bool read, bool write) {
  if (operands.empty())
    return;
  OpOperand &operand = *operands.begin();
  if (read)
    addEffect(effects, &operand, MemoryEffects::Read::get());
  if (write)
    addEffect(effects, &operand, MemoryEffects::Write::get());
}

static void addA2A3ScratchEffects(PTOEffectList &effects, Operation *op,
                                  mlir::MutableOperandRange scratch) {
  if (getTargetArch(op) != PTOArch::A5)
    addOptionalEffects(effects, scratch, /*read=*/true, /*write=*/true);
}

static void addStoreLikeEffects(PTOEffectList &effects, OpOperand &src,
                                mlir::MutableOperandRange fp,
                                mlir::MutableOperandRange preQuant,
                                OpOperand &dst) {
  addEffect(effects, &src, MemoryEffects::Read::get());
  addOptionalEffects(effects, fp, /*read=*/true, /*write=*/false);
  addOptionalEffects(effects, preQuant, /*read=*/true, /*write=*/false);
  addEffect(effects, &dst, MemoryEffects::Write::get());
}

// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result) {
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
  }
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
  addStoreLikeEffects(effects, getSrcMutable(), getFpMutable(),
                      getPreQuantScalarMutable(), getDstMutable());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (classifyTMovForm(getFp()) == TMovForm::XToZz) {
    const MxGroupAxis axis = getGrpAxisAttr()
                                 ? getGrpAxisAttr().getValue()
                                 : MxGroupAxis::Axis1;
    if (axis == MxGroupAxis::Axis1) {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
      addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
    } else {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
    }
    addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
    auto fp = getFpMutable();
    if (axis == MxGroupAxis::Axis1 && !fp.empty()) {
      addEffect(effects, &fp[0], MemoryEffects::Read::get());
      addEffect(effects, &fp[0], MemoryEffects::Write::get());
    }
    return;
  }
  addStoreLikeEffects(effects, getSrcMutable(), getFpMutable(),
                      getPreQuantScalarMutable(), getDstMutable());
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

#define PTO_DEFINE_UNARY_SCRATCH_EFFECTS(OpClass, srcOperand, tmpOperand,          \
                                         dstOperand)                               \
  void OpClass::getEffects(PTOEffectList &effects) {                               \
    PTO_ADD_READ(srcOperand);                                                      \
    addA2A3ScratchEffects(effects, getOperation(), tmpOperand);                    \
    PTO_ADD_WRITE(dstOperand);                                                     \
  }

#define PTO_DEFINE_BINARY_SCRATCH_EFFECTS(OpClass, lhsOperand, rhsOperand,         \
                                          tmpOperand, dstOperand)                  \
  void OpClass::getEffects(PTOEffectList &effects) {                               \
    PTO_ADD_READ(lhsOperand);                                                      \
    PTO_ADD_READ(rhsOperand);                                                      \
    addA2A3ScratchEffects(effects, getOperation(), tmpOperand);                    \
    PTO_ADD_WRITE(dstOperand);                                                     \
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
  if (!scratchRange.empty()) {
    addEffect(effects, &*scratchRange.begin(), MemoryEffects::Write::get());
  }
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
PTO_DEFINE_BINARY_EFFECTS(TAddReluOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
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
  if (auto tmp = getTmpMutable();
      !tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
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

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TColArgMaxOp, getSrcMutable(),
                                 getTmpMutable(), getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TColArgMinOp, getSrcMutable(),
                                 getTmpMutable(), getDstMutable())

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  addOptionalEffects(effects, getTmpMutable(), /*read=*/true, /*write=*/true);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TCvtOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
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
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addOptionalEffects(effects, getFpMutable(), /*read=*/true, /*write=*/false);
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addOptionalEffects(effects, getFpMutable(), /*read=*/true, /*write=*/false);
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty()) {
    PTO_ADD_WRITE(cdst[0]);
  }
  if (auto indices = getIndicesMutable(); !indices.empty()) {
    PTO_ADD_READ(indices[0]);
  }
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
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
void TInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getDst0Mutable());
  PTO_ADD_WRITE(getDst1Mutable());
}
void TDeInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &operand : getSrcsMutable()) {
    PTO_ADD_READ(operand);
  }
  for (auto &operand : getDstsMutable()) {
    PTO_ADD_WRITE(operand);
  }
}
#define PTO_DEFINE_PART_ARG_EFFECTS(OpClass)                                       \
  void OpClass::getEffects(PTOEffectList &effects) {                               \
    PTO_ADD_READ(getSrc0Mutable());                                                \
    PTO_ADD_READ(getSrc1Mutable());                                                \
    PTO_ADD_READ(getSrc0IdxMutable());                                             \
    PTO_ADD_READ(getSrc1IdxMutable());                                             \
    PTO_ADD_WRITE(getDstMutable());                                                \
    PTO_ADD_WRITE(getDstIdxMutable());                                             \
  }
PTO_DEFINE_PART_ARG_EFFECTS(TPartArgMaxOp)
PTO_DEFINE_PART_ARG_EFFECTS(TPartArgMinOp)
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty()) {
    PTO_ADD_READ(offsetRange[0]);
  }
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  Type srcTy = getSrc().getType();
  auto valid = getValidShapeVec(srcTy);
  auto physical = getShapeVec(srcTy);
  Type elem = getElemTy(srcTy);
  if ((elem.isF16() || elem.isBF16()) && valid.size() == mlir::pto::kValue2 && physical.size() == mlir::pto::kValue2 &&
      valid[1] < physical[1]) {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
      addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
  } else {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  }
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getExpMutable());
  PTO_ADD_WRITE(getMaxMutable());
  PTO_ADD_WRITE(getScalingMutable());
  auto expZzRange = getExpZzMutable();
  if (!expZzRange.empty()) {
    PTO_ADD_WRITE(expZzRange[0]);
  }
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRemOp, getSrc0Mutable(), getSrc1Mutable(),
                                  getTmpMutable(), getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRemSOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TPowOp, getBaseMutable(), getExpMutable(),
                                  getTmpMutable(), getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TPowSOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandDivOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandMulOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandSubOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandAddOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandMaxOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandMinOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())

// Row reductions use tmp scratch tile.
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowMaxOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowMinOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowSumOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowProdOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  addOptionalEffects(effects, getTmpMutable(), /*read=*/true, /*write=*/true);
  PTO_ADD_WRITE(getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty()) {
      PTO_ADD_READ(idx[0]);
    }
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp on A2/A3, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TSEL; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(mask, src) -> Write(tmp on A2/A3, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TSELS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp on A2/A3, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TXORS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp on A2/A3, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TXOR; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && ttransUsesTmp(getSrc().getType(), getDst().getType())) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
