// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

// ---- StructType ----
// Asm form: !pto.struct<T0, T1, ..., Tn-1>
// A field type must be "scalar-storable": an exactly-nameable scalar or a
// nested !pto.struct (see the two predicates below). The allowlist deliberately
// excludes the vec/cube types (tile_buf / tensor_view / partition view) and any
// other handle type, keeping the scalar struct world disjoint from the
// fractal/layout world.

// A struct field's scalar type must map onto a C++ scalar that the backend can
// name exactly: integers of width 8/16/32/64 and f16/bf16/f32/f64. Widths the
// backend has no spelling for (i1, i24, ...) and the packed low-precision
// vec/cube formats (f8/f4 variants) would otherwise be emitted as `float`,
// silently changing the field's width and semantics, so reject them here.
static bool isStructScalar(Type t) {
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(t)) {
    return true;
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(t)) {
    unsigned w = intTy.getWidth();
    return w == mlir::pto::kValue8 || w == mlir::pto::kValue16 || w == mlir::pto::kValue32 || w == mlir::pto::kValue64;
  }
  return false;
}

// A field is either such a scalar or a nested !pto.struct. !pto.local_array is
// deliberately NOT allowed: a field is reached with `emitc.member`, whose result
// must be an `!emitc.lvalue`, and `!emitc.lvalue` cannot wrap `!emitc.array`
// (the type an array field lowers to). There is no way to spell the access, so
// the restriction is enforced here rather than failing later in the backend.
static bool isStructStorable(Type t) {
  return isStructScalar(t) || llvm::isa<StructType>(t);
}

Type StructType::parse(AsmParser &parser) {
  SmallVector<Type> fields;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            Type t;
            if (parser.parseType(t)) {
              return failure();
            }
            fields.push_back(t);
            return success();
          })) {
    return Type();
  }
  return StructType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), fields);
}

void StructType::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::ArrayRef<Type> fields = getFieldTypes();
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer.printType(fields[i]);
  }
  printer << ">";
}

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
