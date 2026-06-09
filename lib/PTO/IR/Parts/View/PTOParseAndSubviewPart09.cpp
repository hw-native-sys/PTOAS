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

static bool extractStrideFromMulExpr(AffineExpr lhs, AffineExpr rhs,
                                     unsigned &position, int64_t &stride) {
  auto dim = llvm::dyn_cast<AffineDimExpr>(lhs);
  auto constant = llvm::dyn_cast<AffineConstantExpr>(rhs);
  if (!dim || !constant)
    return false;
  position = dim.getPosition();
  stride = constant.getValue();
  return true;
}

static bool extractStrideFromAffineTerm(AffineExpr term, unsigned &position,
                                        int64_t &stride) {
  auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term);
  if (!mul || mul.getKind() != AffineExprKind::Mul)
    return false;
  return extractStrideFromMulExpr(mul.getLHS(), mul.getRHS(), position,
                                  stride) ||
         extractStrideFromMulExpr(mul.getRHS(), mul.getLHS(), position,
                                  stride);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);
  if (map.getNumResults() != 1)
    return;

  // 2. 摊平表达式
  SmallVec4<AffineExpr> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    unsigned position = 0;
    int64_t stride = 0;
    if (extractStrideFromAffineTerm(term, position, stride)) {
      strides[position] = stride;
      continue;
    }
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(term))
      strides[dim.getPosition()] = 1;
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures the AffineExpr addition order below
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

static ParseResult parseSubViewSourceOffsetsAndSizes(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets) {
  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperandList(offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes")) {
    return failure();
  }
  ArrayAttr sizesAttr;
  return parser.parseAttribute(sizesAttr, "sizes", result.attributes);
}

static ParseResult parseSubViewValids(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &valids) {
  if (failed(parser.parseOptionalKeyword("valid")))
    return success();
  OpAsmParser::UnresolvedOperand rowValid;
  OpAsmParser::UnresolvedOperand colValid;
  if (parser.parseLSquare() || parser.parseOperand(rowValid) ||
      parser.parseComma() || parser.parseOperand(colValid) ||
      parser.parseRSquare()) {
    return failure();
  }
  valids.push_back(rowValid);
  valids.push_back(colValid);
  return success();
}

static ParseResult resolveSubViewSourceAndIndices(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source, Type sourceTy, Type &resultTy,
    bool &hasExplicitResultTy,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &valids) {
  if (parseOptionalArrowTypeAndResolveSource(parser, result, source, sourceTy,
                                             resultTy, hasExplicitResultTy))
    return failure();
  if (resolveIndexOperandsToResult(parser, offsets, result))
    return failure();
  if (!valids.empty() && resolveIndexOperandsToResult(parser, valids, result))
    return failure();
  return success();
}

static ParseResult finalizeSubViewResultTypes(OpAsmParser &parser,
                                              OperationState &result,
                                              Type resultTy,
                                              bool hasExplicitResultTy) {
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

