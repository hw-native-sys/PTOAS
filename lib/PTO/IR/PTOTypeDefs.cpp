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
#include <mutex>
#include <unordered_map>

using namespace mlir;
using namespace mlir::pto;

namespace {
std::mutex parserTargetArchMutex;
std::unordered_map<const MLIRContext *, PTOParserTargetArch>
    parserTargetArchByContext;

constexpr unsigned kTileBufRank2D = 2;
constexpr unsigned kTileBufValidShapeInlineCapacity = 4;
constexpr unsigned kI32BitWidth = 32;

using TileBufShape = SmallVector<int64_t, kTileBufRank2D>;
using TileBufValidShapeVector =
    SmallVector<int64_t, kTileBufValidShapeInlineCapacity>;
}

void mlir::pto::setPTOParserTargetArch(MLIRContext *context,
                                       PTOParserTargetArch arch) {
  if (!context)
    return;

  std::lock_guard<std::mutex> lock(parserTargetArchMutex);
  if (arch == PTOParserTargetArch::Unspecified) {
    parserTargetArchByContext.erase(context);
    return;
  }
  parserTargetArchByContext[context] = arch;
}

PTOParserTargetArch mlir::pto::getPTOParserTargetArch(MLIRContext *context) {
  if (!context)
    return PTOParserTargetArch::Unspecified;

  std::lock_guard<std::mutex> lock(parserTargetArchMutex);
  auto it = parserTargetArchByContext.find(context);
  if (it == parserTargetArchByContext.end())
    return PTOParserTargetArch::Unspecified;
  return it->second;
}

mlir::pto::ScopedPTOParserTargetArch::ScopedPTOParserTargetArch(
    MLIRContext *context, PTOParserTargetArch arch)
    : context(context), previousArch(getPTOParserTargetArch(context)) {
  setPTOParserTargetArch(context, arch);
}

mlir::pto::ScopedPTOParserTargetArch::~ScopedPTOParserTargetArch() {
  setPTOParserTargetArch(context, previousArch);
}

static TileBufValidShapeVector
canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  TileBufValidShapeVector canonical;
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

static BLayout resolveTileBufBLayout(MLIRContext *context,
                                     AddressSpace memorySpace,
                                     BLayout parsedLayout) {
  if (memorySpace != AddressSpace::LEFT)
    return parsedLayout;

  switch (getPTOParserTargetArch(context)) {
  case PTOParserTargetArch::A3:
    return BLayout::RowMajor;
  case PTOParserTargetArch::A5:
  case PTOParserTargetArch::Kirin9030:
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
mlir::Attribute TileBufType::getCompactModeAttr() const {
  return getConfigAttr().getCompactMode();
}

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

int32_t TileBufType::getCompactModeI32() const {
  if (auto a = llvm::dyn_cast<CompactModeAttr>(getCompactModeAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

namespace {

struct ParsedTileBufFields {
  std::string locStr;
  Type dtype;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t vrow = -1;
  int64_t vcol = -1;
  std::string blayoutStr;
  std::string slayoutStr;
  int64_t fractal = 0;
  uint32_t padInt = 0;
  uint32_t compactInt = 0;
};

static LogicalResult parseTileBufUInt32Value(AsmParser &parser, StringRef key,
                                             uint32_t &value) {
  int64_t parsedValue = 0;
  if (failed(parser.parseInteger(parsedValue)))
    return failure();
  if (parsedValue < 0 ||
      parsedValue > std::numeric_limits<uint32_t>::max()) {
    parser.emitError(parser.getCurrentLocation())
        << key << " must be a non-negative 32-bit integer";
    return failure();
  }
  value = static_cast<uint32_t>(parsedValue);
  return success();
}

static LogicalResult parseLegacyTileBufFields(AsmParser &parser,
                                              ParsedTileBufFields &fields) {
  if (failed(parser.parseEqual()))
    return failure();
  if (failed(parser.parseKeywordOrString(&fields.locStr)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseTileBufTypeField(parser, "dtype", fields.dtype)) ||
      failed(parseTileBufIntegerField(parser, "rows", fields.rows)) ||
      failed(parseTileBufIntegerField(parser, "cols", fields.cols)) ||
      failed(parseTileBufValidShapeFields(parser, fields.vrow, fields.vcol)) ||
      failed(parseTileBufKeywordField(parser, "blayout", fields.blayoutStr)) ||
      failed(parseTileBufKeywordField(parser, "slayout", fields.slayoutStr)) ||
      failed(parseTileBufIntegerField(parser, "fractal", fields.fractal)) ||
      failed(parseTileBufPadField(parser, fields.padInt))) {
    return failure();
  }

  return success();
}

static LogicalResult parseCompactTileBufFields(AsmParser &parser,
                                               StringRef firstToken,
                                               ParsedTileBufFields &fields) {
  fields.locStr = firstToken.str();

  if (failed(parser.parseComma()))
    return failure();

  TileBufShape shape;
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/false)))
    return failure();
  if (failed(parser.parseType(fields.dtype)))
    return failure();
  if (shape.size() != kTileBufRank2D) {
    parser.emitError(parser.getCurrentLocation(),
                     "tile_buf compact syntax expects exactly two shape dims");
    return failure();
  }

  fields.rows = shape[0];
  fields.cols = shape[1];
  fields.vrow = fields.rows;
  fields.vcol = fields.cols;

  auto defaultConfig = TileBufConfigAttr::getDefault(parser.getContext());
  auto defaultBLayout = llvm::dyn_cast<BLayoutAttr>(defaultConfig.getBLayout());
  auto defaultSLayout = llvm::dyn_cast<SLayoutAttr>(defaultConfig.getSLayout());
  auto defaultPad = llvm::dyn_cast<PadValueAttr>(defaultConfig.getPad());
  auto defaultCompact =
      llvm::dyn_cast<CompactModeAttr>(defaultConfig.getCompactMode());
  if (!defaultBLayout || !defaultSLayout || !defaultPad || !defaultCompact) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to load default tile_buf config");
    return failure();
  }
  fields.blayoutStr = stringifyBLayout(defaultBLayout.getValue()).str();
  fields.slayoutStr = stringifySLayout(defaultSLayout.getValue()).str();
  fields.fractal = defaultConfig.getSFractalSize().getInt();
  fields.padInt = static_cast<uint32_t>(defaultPad.getValue());
  fields.compactInt = static_cast<uint32_t>(defaultCompact.getValue());

  bool seenValid = false;
  bool seenBLayout = false;
  bool seenSLayout = false;
  bool seenFractal = false;
  bool seenPad = false;
  bool seenCompact = false;

  while (succeeded(parser.parseOptionalComma())) {
    StringRef key;
    if (failed(parser.parseKeyword(&key)) || failed(parser.parseEqual()))
      return failure();

    if (key == "valid") {
      if (seenValid) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate valid in tile_buf compact syntax");
        return failure();
      }
      seenValid = true;

      TileBufShape validShape;
      if (failed(parser.parseDimensionList(validShape, /*allowDynamic=*/true,
                                           /*withTrailingX=*/false))) {
        return failure();
      }
      if (validShape.size() != kTileBufRank2D) {
        parser.emitError(parser.getCurrentLocation(),
                         "tile_buf valid must have exactly two dims");
        return failure();
      }
      fields.vrow = validShape[0];
      fields.vcol = validShape[1];
      continue;
    }

    if (key == "blayout") {
      if (seenBLayout) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate blayout in tile_buf compact syntax");
        return failure();
      }
      seenBLayout = true;
      if (failed(parser.parseKeywordOrString(&fields.blayoutStr)))
        return failure();
      continue;
    }

    if (key == "slayout") {
      if (seenSLayout) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate slayout in tile_buf compact syntax");
        return failure();
      }
      seenSLayout = true;
      if (failed(parser.parseKeywordOrString(&fields.slayoutStr)))
        return failure();
      continue;
    }

    if (key == "fractal") {
      if (seenFractal) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate fractal in tile_buf compact syntax");
        return failure();
      }
      seenFractal = true;
      if (failed(parser.parseInteger(fields.fractal)))
        return failure();
      continue;
    }

    if (key == "pad") {
      if (seenPad) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate pad in tile_buf compact syntax");
        return failure();
      }
      seenPad = true;
      if (failed(parseTileBufUInt32Value(parser, key, fields.padInt)))
        return failure();
      continue;
    }

    if (key == "compact") {
      if (seenCompact) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate compact in tile_buf compact syntax");
        return failure();
      }
      seenCompact = true;
      if (failed(parseTileBufUInt32Value(parser, key, fields.compactInt)))
        return failure();
      continue;
    }

    parser.emitError(parser.getCurrentLocation(),
                     "unknown key in tile_buf compact syntax: ")
        << key;
    return failure();
  }

  return success();
}

static Type buildTileBufType(AsmParser &parser,
                             const ParsedTileBufFields &fields) {
  MLIRContext *ctx = parser.getContext();

  if (fields.rows < 0 || fields.cols < 0) {
    parser.emitError(parser.getNameLoc(), "rows/cols must be non-negative");
    return Type();
  }

  auto memorySpace = resolveTileBufMemorySpace(fields.locStr);
  if (!memorySpace.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown loc: ") << fields.locStr;
    return Type();
  }

  auto bl = symbolizeBLayout(fields.blayoutStr);
  auto sl = symbolizeSLayout(fields.slayoutStr);
  auto pv = symbolizePadValue(fields.padInt);
  auto compact = symbolizeCompactMode(fields.compactInt);
  if (!bl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown blayout: ")
        << fields.blayoutStr;
    return Type();
  }
  if (!sl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown slayout: ")
        << fields.slayoutStr;
    return Type();
  }
  if (!pv.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown pad: ") << fields.padInt;
    return Type();
  }
  if (!compact.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown compact: ")
        << fields.compactInt;
    return Type();
  }

  BLayout effectiveBLayout =
      resolveTileBufBLayout(parser.getContext(), memorySpace.value(),
                            bl.value());

  auto blAttr = BLayoutAttr::get(ctx, effectiveBLayout);
  auto slAttr = SLayoutAttr::get(ctx, sl.value());
  auto fractalAttr =
      IntegerAttr::get(IntegerType::get(ctx, kI32BitWidth), fields.fractal);
  auto padAttr = PadValueAttr::get(ctx, pv.value());
  auto compactAttr = CompactModeAttr::get(ctx, compact.value());
  auto memorySpaceAttr = AddressSpaceAttr::get(ctx, memorySpace.value());
  auto cfg = TileBufConfigAttr::get(ctx, blAttr, slAttr, fractalAttr, padAttr,
                                    compactAttr);

  TileBufShape shape{fields.rows, fields.cols};
  TileBufShape validShape{fields.vrow, fields.vcol};
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);

  return TileBufType::get(ctx, shape, fields.dtype, memorySpaceAttr,
                          llvm::ArrayRef<int64_t>(canonicalValidShape), cfg);
}

} // namespace

// ---- TileBufType custom asm ----
// !pto.tile_buf<<loc=.., dtype=.., rows=.., cols=.., blayout=.., valid=..x..,
//                slayout=.., fractal=.., pad=.., compact=..>>
Type TileBufType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();

  std::string firstToken;
  if (failed(parser.parseKeywordOrString(&firstToken)))
    return Type();

  ParsedTileBufFields fields;
  const bool isLegacySyntax = firstToken == "loc";
  if (isLegacySyntax) {
    if (failed(parseLegacyTileBufFields(parser, fields)))
      return Type();
  } else {
    if (failed(parseCompactTileBufFields(parser, firstToken, fields)))
      return Type();
  }

  if (isLegacySyntax && succeeded(parser.parseOptionalComma())) {
    if (failed(parseTileBufKeyEq(parser, "compact")) ||
        failed(parseTileBufUInt32Value(parser, "compact", fields.compactInt))) {
      return Type();
    }
  }

  if (failed(parser.parseGreater()))
    return Type();

  return buildTileBufType(parser, fields);
}

static llvm::StringRef stringifyLocFromMemorySpace(mlir::Attribute memorySpace) {
  auto asAttr = llvm::dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  switch (asAttr.getAddressSpace()) {
    case AddressSpace::Zero:
    case AddressSpace::GM:
      return "illegal";
    case AddressSpace::MAT: return "mat";
    case AddressSpace::LEFT: return "left";
    case AddressSpace::RIGHT: return "right";
    case AddressSpace::ACC: return "acc";
    case AddressSpace::VEC: return "vec";
    case AddressSpace::BIAS: return "bias";
    case AddressSpace::SCALING: return "scaling";
  }
  return "illegal";
}

static llvm::StringRef stringifyLocFromPad(mlir::Attribute pad) {
  auto padAttr = llvm::dyn_cast_or_null<PadValueAttr>(pad);
  if (!padAttr) return "9999";

  switch (padAttr.getValue()) {
    case PadValue::Null: return "0";
    case PadValue::Zero: return "1";
    case PadValue::Max: return "2";
    case PadValue::Min: return "3";
  }
  return "9999";
}

static llvm::StringRef stringifyCompactModeInt(mlir::Attribute compactMode) {
  auto compactAttr = llvm::dyn_cast_or_null<CompactModeAttr>(compactMode);
  if (!compactAttr)
    return "9999";

  switch (compactAttr.getValue()) {
  case CompactMode::Null:
    return "0";
  case CompactMode::Normal:
    return "1";
  case CompactMode::RowPlusOne:
    return "2";
  }
  return "9999";
}

static void printTileBufDim(AsmPrinter &printer, int64_t dim) {
  if (dim == ShapedType::kDynamic)
    printer << "?";
  else
    printer << dim;
}

void mlir::pto::TileBufType::print(mlir::AsmPrinter &printer) const {
  auto shape = getShape();
  int64_t rows = shape.size() > 0 ? shape[0] : ShapedType::kDynamic;
  int64_t cols = shape.size() > 1 ? shape[1] : ShapedType::kDynamic;

  auto cfg = getConfigAttr();
  if (!cfg)
    cfg = mlir::pto::TileBufConfigAttr::getDefault(getContext());
  auto defaultCfg = TileBufConfigAttr::getDefault(getContext());

  llvm::StringRef locStr = stringifyLocFromMemorySpace(getMemorySpace());
  auto blayout = llvm::dyn_cast<BLayoutAttr>(cfg.getBLayout());
  auto slayout = llvm::dyn_cast<SLayoutAttr>(cfg.getSLayout());
  auto pad = llvm::dyn_cast<PadValueAttr>(cfg.getPad());
  auto compact = llvm::dyn_cast<CompactModeAttr>(cfg.getCompactMode());
  auto defaultBLayout = llvm::dyn_cast<BLayoutAttr>(defaultCfg.getBLayout());
  auto defaultSLayout = llvm::dyn_cast<SLayoutAttr>(defaultCfg.getSLayout());
  auto defaultPad = llvm::dyn_cast<PadValueAttr>(defaultCfg.getPad());
  auto defaultCompact =
      llvm::dyn_cast<CompactModeAttr>(defaultCfg.getCompactMode());

  auto vs = getValidShape();
  int64_t vrow = rows;
  int64_t vcol = cols;
  if (vs.size() >= kTileBufRank2D) {
    vrow = vs[0];
    vcol = vs[1];
  }

  const bool printValid = vrow != rows || vcol != cols;
  const bool printBLayout =
      blayout && defaultBLayout && blayout.getValue() != defaultBLayout.getValue();
  const bool printSLayout =
      slayout && defaultSLayout && slayout.getValue() != defaultSLayout.getValue();
  const bool printFractal =
      cfg.getSFractalSize().getInt() != defaultCfg.getSFractalSize().getInt();
  const bool printPad =
      pad && defaultPad && pad.getValue() != defaultPad.getValue();
  const bool printCompact =
      compact && defaultCompact &&
      compact.getValue() != defaultCompact.getValue();

  printer << "<" << locStr << ", ";
  printTileBufDim(printer, rows);
  printer << "x";
  printTileBufDim(printer, cols);
  printer << "x";
  printer.printType(getElementType());

  if (printValid) {
    printer << ", valid=";
    printTileBufDim(printer, vrow);
    printer << "x";
    printTileBufDim(printer, vcol);
  }
  if (printBLayout)
    printer << ", blayout=" << stringifyBLayout(blayout.getValue());
  if (printSLayout)
    printer << ", slayout=" << stringifySLayout(slayout.getValue());
  if (printFractal)
    printer << ", fractal=" << cfg.getSFractalSize().getInt();
  if (printPad)
    printer << ", pad=" << stringifyLocFromPad(cfg.getPad());
  if (printCompact)
    printer << ", compact=" << stringifyCompactModeInt(cfg.getCompactMode());

  printer << ">";
}
