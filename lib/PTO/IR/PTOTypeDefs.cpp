// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTypeDefs.cpp --------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/DialectImplementation.h"
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {
thread_local PTOParserTargetArch currentParserTargetArch =
    PTOParserTargetArch::Unspecified;
}

void mlir::pto::setPTOParserTargetArch(PTOParserTargetArch arch) {
  currentParserTargetArch = arch;
}

PTOParserTargetArch mlir::pto::getPTOParserTargetArch() {
  return currentParserTargetArch;
}

mlir::pto::ScopedPTOParserTargetArch::ScopedPTOParserTargetArch(
    PTOParserTargetArch arch)
    : previousArch(getPTOParserTargetArch()) {
  setPTOParserTargetArch(arch);
}

mlir::pto::ScopedPTOParserTargetArch::~ScopedPTOParserTargetArch() {
  setPTOParserTargetArch(previousArch);
}

static SmallVector<int64_t, 4> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  SmallVector<int64_t, 4> canonical;
  canonical.reserve(validShape.size());
  for (int64_t dim : validShape)
    canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  return canonical;
}

static LogicalResult parseTileBufKeyEq(AsmParser &parser,
                                       StringRef expectedKey) {
  if (failed(parser.parseKeyword(expectedKey)))
    return failure();
  return parser.parseEqual();
}

static LogicalResult parseTileBufComma(AsmParser &parser) {
  return parser.parseComma();
}

static LogicalResult parseTileBufKeywordField(AsmParser &parser, StringRef key,
                                              std::string &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();
  if (failed(parser.parseKeywordOrString(&value)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufTypeField(AsmParser &parser, StringRef key,
                                           Type &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();
  if (failed(parser.parseType(value)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufIntegerField(AsmParser &parser, StringRef key,
                                              int64_t &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();
  if (failed(parser.parseInteger(value)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufValidDim(AsmParser &parser, StringRef key,
                                          int64_t &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();

  if (succeeded(parser.parseOptionalQuestion())) {
    value = -1;
    return success();
  }

  if (failed(parser.parseInteger(value)))
    return failure();
  if (value < -1) {
    parser.emitError(parser.getCurrentLocation(),
                     key + " must be '?', -1, or a non-negative integer");
    return failure();
  }
  return success();
}

static LogicalResult parseTileBufValidShapeFields(AsmParser &parser,
                                                  int64_t &vrow,
                                                  int64_t &vcol) {
  if (failed(parseTileBufValidDim(parser, "v_row", vrow)))
    return failure();
  if (failed(parseTileBufComma(parser)))
    return failure();
  if (failed(parseTileBufValidDim(parser, "v_col", vcol)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufPadField(AsmParser &parser, uint32_t &padInt) {
  int64_t parsedPad = 0;
  if (failed(parseTileBufKeyEq(parser, "pad")))
    return failure();
  if (failed(parser.parseInteger(parsedPad)))
    return failure();
  if (parsedPad < 0 || parsedPad > std::numeric_limits<uint32_t>::max()) {
    parser.emitError(parser.getCurrentLocation(),
                     "pad must be a non-negative 32-bit integer");
    return failure();
  }
  padInt = static_cast<uint32_t>(parsedPad);
  return success();
}

static std::optional<AddressSpace> resolveTileBufMemorySpace(StringRef locStr) {
  return ::llvm::StringSwitch<::std::optional<AddressSpace>>(locStr)
      .Case("mat", AddressSpace::MAT)
      .Case("left", AddressSpace::LEFT)
      .Case("right", AddressSpace::RIGHT)
      .Case("acc", AddressSpace::ACC)
      .Case("vec", AddressSpace::VEC)
      .Case("bias", AddressSpace::BIAS)
      .Case("scaling", AddressSpace::SCALING)
      .Default(::std::nullopt);
}

static BLayout resolveTileBufBLayout(AddressSpace memorySpace,
                                     BLayout parsedLayout) {
  if (memorySpace != AddressSpace::LEFT)
    return parsedLayout;

  switch (getPTOParserTargetArch()) {
  case PTOParserTargetArch::A3:
    return BLayout::RowMajor;
  case PTOParserTargetArch::A5:
    return BLayout::ColMajor;
  case PTOParserTargetArch::Unspecified:
    return parsedLayout;
  }
  return parsedLayout;
}

TileBufConfigAttr TileBufType::getConfigAttr() const {
  // 情况 A：getConfig() 已经是 TileBufConfigAttr
  if constexpr (std::is_same_v<decltype(getConfig()), TileBufConfigAttr>) {
    auto cfg = getConfig();
    if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());
    return cfg;
  } else {
    // 情况 B：getConfig() 是 Attribute
    auto cfg = llvm::dyn_cast_or_null<TileBufConfigAttr>(getConfig());
    if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());
    return cfg;
  }
}
bool TileBufType::hasNonDefaultConfig() const {
  return !getConfigAttr().isDefault();
}

mlir::Attribute TileBufType::getBLayoutAttr() const { return getConfigAttr().getBLayout(); }
mlir::Attribute TileBufType::getSLayoutAttr() const { return getConfigAttr().getSLayout(); }
mlir::Attribute TileBufType::getPadValueAttr() const { return getConfigAttr().getPad(); }

// ✅ numeric getters（可选）
int32_t TileBufType::getSFractalSizeI32() const {
  return (int32_t)getConfigAttr().getSFractalSize().getInt();
}

int32_t TileBufType::getBLayoutValueI32() const {
  if (auto a = llvm::dyn_cast<BLayoutAttr>(getBLayoutAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getSLayoutValueI32() const {
  if (auto a = llvm::dyn_cast<SLayoutAttr>(getSLayoutAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getPadValueI32() const {
  if (auto a = llvm::dyn_cast<PadValueAttr>(getPadValueAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

// ---- TileBufType custom asm ----
// !pto.tile_buf<<loc=.., dtype=.., rows=.., cols=.., blayout=.., valid=..x.., slayout=.., fractal=.., pad=..>>
Type TileBufType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  if (failed(parser.parseLess()))
    return Type();

  std::string locStr;
  Type dtype;
  int64_t rows = 0, cols = 0;
  int64_t vrow = -1, vcol = -1;
  std::string blayoutStr, slayoutStr;
  int64_t fractal = 0;
  uint32_t padInt;
  if (failed(parseTileBufKeywordField(parser, "loc", locStr)) ||
      failed(parseTileBufTypeField(parser, "dtype", dtype)) ||
      failed(parseTileBufIntegerField(parser, "rows", rows)) ||
      failed(parseTileBufIntegerField(parser, "cols", cols)) ||
      failed(parseTileBufValidShapeFields(parser, vrow, vcol)) ||
      failed(parseTileBufKeywordField(parser, "blayout", blayoutStr)) ||
      failed(parseTileBufKeywordField(parser, "slayout", slayoutStr)) ||
      failed(parseTileBufIntegerField(parser, "fractal", fractal)) ||
      failed(parseTileBufPadField(parser, padInt))) {
    return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  // -------- 语义校验/构造 --------
  if (rows < 0 || cols < 0) {
    parser.emitError(parser.getNameLoc(), "rows/cols must be non-negative");
    return Type();
  }

  auto memorySpace = resolveTileBufMemorySpace(locStr);
  if (!memorySpace.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown loc: ") << locStr;
    return Type();
  }

  auto bl = symbolizeBLayout(blayoutStr);
  auto sl = symbolizeSLayout(slayoutStr);
  auto pv = symbolizePadValue(padInt);
  if (!bl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown blayout: ") << blayoutStr;
    return Type();
  }
  if (!sl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown slayout: ") << slayoutStr;
    return Type();
  }
  if (!pv.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown pad: ") << padInt;
    return Type();
  }

  BLayout effectiveBLayout =
      resolveTileBufBLayout(memorySpace.value(), bl.value());

  auto blAttr = BLayoutAttr::get(ctx, effectiveBLayout);
  auto slAttr = SLayoutAttr::get(ctx, sl.value());
  auto fractalAttr =
      IntegerAttr::get(IntegerType::get(ctx, 32), fractal);
  auto padAttr = PadValueAttr::get(ctx, pv.value());
  auto memorySpaceAttr = AddressSpaceAttr::get(ctx, memorySpace.value());
  auto cfg = TileBufConfigAttr::get(ctx, blAttr, slAttr, fractalAttr, padAttr);

  SmallVector<int64_t, 2> shape{rows, cols};
  SmallVector<int64_t, 2> validShape{vrow, vcol};
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);

  return TileBufType::get(ctx, shape, dtype, memorySpaceAttr,
                          llvm::ArrayRef<int64_t>(canonicalValidShape), cfg);
}

static llvm::StringRef stringifyLocFromMemorySpace(mlir::Attribute memorySpace) {
  auto asAttr = llvm::dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  switch (asAttr.getAddressSpace()) {
    case AddressSpace::MAT: return "mat";
    case AddressSpace::LEFT: return "left";
    case AddressSpace::RIGHT: return "right";
    case AddressSpace::ACC: return "acc";
    case AddressSpace::VEC: return "vec";
    case AddressSpace::BIAS: return "bias";
    case AddressSpace::SCALING: return "scaling";
    default: return "illegal";
  }
}

static llvm::StringRef stringifyLocFromPad(mlir::Attribute pad) {
  auto padAttr = llvm::dyn_cast_or_null<PadValueAttr>(pad);
  if (!padAttr) return "9999";

  switch (padAttr.getValue()) {
    case PadValue::Null: return "0";
    case PadValue::Zero: return "1";
    case PadValue::Max: return "2";
    case PadValue::Min: return "3";
    default:
      return "9999";
  }
}

void mlir::pto::TileBufType::print(mlir::AsmPrinter &printer) const {
    auto shape = getShape();
    int64_t rows = shape.size() > 0 ? shape[0] : 0;
    int64_t cols = shape.size() > 1 ? shape[1] : 0;

    auto cfg = getConfigAttr();
    if (!cfg) cfg = mlir::pto::TileBufConfigAttr::getDefault(getContext());

    llvm::StringRef locStr = stringifyLocFromMemorySpace(getMemorySpace());

    printer << "<"
            << "loc=" << locStr
            << ", dtype=";
    printer.printType(getElementType());

    auto blayout = llvm::dyn_cast<BLayoutAttr>(cfg.getBLayout());
    auto slayout = llvm::dyn_cast<SLayoutAttr>(cfg.getSLayout());

    auto vs = getValidShape(); // ArrayRef<int64_t>
    int64_t vrow = rows;
    int64_t vcol = cols;

    if (vs.size() >= 2) {
        vrow = vs[0];
        vcol = vs[1];
    }
    printer << ", rows=" << rows
            << ", cols=" << cols;
    printer << ", v_row=";
    if (vrow < 0) printer << "?";
    else printer << vrow;

    printer << ", v_col=";
    if (vcol < 0) printer << "?";
    else printer << vcol;

    printer << ", blayout=" << stringifyBLayout(blayout.getValue())
        << ", slayout=" << stringifySLayout(slayout.getValue())
        << ", fractal=" << cfg.getSFractalSize().getInt()
        << ", pad=" << stringifyLocFromPad(cfg.getPad())
        << ">";
}
