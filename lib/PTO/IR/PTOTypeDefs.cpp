//===- PTOTypeDefs.cpp --------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/DialectImplementation.h"

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
};

static LogicalResult parseLegacyTileBufFields(AsmParser &parser,
                                              ParsedTileBufFields &fields) {
  auto parseKeyEq = [&](StringRef expectedKey) -> LogicalResult {
    if (failed(parser.parseKeyword(expectedKey)))
      return failure();
    if (failed(parser.parseEqual()))
      return failure();
    return success();
  };

  if (failed(parser.parseEqual()))
    return failure();
  if (failed(parser.parseKeywordOrString(&fields.locStr)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("dtype")))
    return failure();
  if (failed(parser.parseType(fields.dtype)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("rows")))
    return failure();
  if (failed(parser.parseInteger(fields.rows)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("cols")))
    return failure();
  if (failed(parser.parseInteger(fields.cols)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("v_row")))
    return failure();
  if (succeeded(parser.parseOptionalQuestion())) {
    fields.vrow = -1;
  } else {
    if (failed(parser.parseInteger(fields.vrow)))
      return failure();
    if (fields.vrow < -1) {
      parser.emitError(parser.getCurrentLocation(),
                       "v_row must be '?', -1, or a non-negative integer");
      return failure();
    }
  }
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("v_col")))
    return failure();
  if (succeeded(parser.parseOptionalQuestion())) {
    fields.vcol = -1;
  } else {
    if (failed(parser.parseInteger(fields.vcol)))
      return failure();
    if (fields.vcol < -1) {
      parser.emitError(parser.getCurrentLocation(),
                       "v_col must be '?', -1, or a non-negative integer");
      return failure();
    }
  }
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("blayout")))
    return failure();
  if (failed(parser.parseKeywordOrString(&fields.blayoutStr)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("slayout")))
    return failure();
  if (failed(parser.parseKeywordOrString(&fields.slayoutStr)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("fractal")))
    return failure();
  if (failed(parser.parseInteger(fields.fractal)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseKeyEq("pad")))
    return failure();
  if (failed(parser.parseInteger(fields.padInt)))
    return failure();

  return success();
}

static LogicalResult parseCompactTileBufFields(AsmParser &parser,
                                               StringRef firstToken,
                                               ParsedTileBufFields &fields) {
  fields.locStr = firstToken.str();

  if (failed(parser.parseComma()))
    return failure();

  SmallVector<int64_t, 2> shape;
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/false)))
    return failure();
  if (failed(parser.parseType(fields.dtype)))
    return failure();
  if (shape.size() != 2) {
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
  if (!defaultBLayout || !defaultSLayout || !defaultPad) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to load default tile_buf config");
    return failure();
  }
  fields.blayoutStr = stringifyBLayout(defaultBLayout.getValue()).str();
  fields.slayoutStr = stringifySLayout(defaultSLayout.getValue()).str();
  fields.fractal = defaultConfig.getSFractalSize().getInt();
  fields.padInt = static_cast<uint32_t>(defaultPad.getValue());

  bool seenValid = false;
  bool seenBLayout = false;
  bool seenSLayout = false;
  bool seenFractal = false;
  bool seenPad = false;

  while (succeeded(parser.parseOptionalComma())) {
    StringRef key;
    if (failed(parser.parseKeyword(&key)))
      return failure();
    if (failed(parser.parseEqual()))
      return failure();

    if (key == "valid") {
      if (seenValid) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate valid in tile_buf compact syntax");
        return failure();
      }
      seenValid = true;

      SmallVector<int64_t, 2> validShape;
      if (failed(parser.parseDimensionList(validShape, /*allowDynamic=*/true,
                                           /*withTrailingX=*/false)))
        return failure();
      if (validShape.size() != 2) {
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
      if (failed(parser.parseInteger(fields.padInt)))
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

  auto memorySpace = ::llvm::StringSwitch<::std::optional<AddressSpace>>(
                         fields.locStr)
                         .Case("mat", AddressSpace::MAT)
                         .Case("left", AddressSpace::LEFT)
                         .Case("right", AddressSpace::RIGHT)
                         .Case("acc", AddressSpace::ACC)
                         .Case("vec", AddressSpace::VEC)
                         .Case("bias", AddressSpace::BIAS)
                         .Case("scaling", AddressSpace::SCALING)
                         .Default(::std::nullopt);
  if (!memorySpace.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown loc: ") << fields.locStr;
    return Type();
  }

  auto bl = symbolizeBLayout(fields.blayoutStr);
  auto sl = symbolizeSLayout(fields.slayoutStr);
  auto pv = symbolizePadValue(fields.padInt);
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

  auto normalizeParserBLayout = [&](AddressSpace memorySpace,
                                    BLayout parsedBLayout) -> BLayout {
    // LEFT tiles are parser-normalized from the scoped target arch rather than
    // from the textual blayout spelling. This preserves the longstanding
    // --pto-arch behavior for both legacy and compact tile_buf syntax.
    if (memorySpace != AddressSpace::LEFT)
      return parsedBLayout;

    switch (getPTOParserTargetArch()) {
    case PTOParserTargetArch::A3:
      return BLayout::RowMajor;
    case PTOParserTargetArch::A5:
      return BLayout::ColMajor;
    case PTOParserTargetArch::Unspecified:
      return parsedBLayout;
    }

    return parsedBLayout;
  };

  BLayout effectiveBLayout =
      normalizeParserBLayout(memorySpace.value(), bl.value());

  auto blAttr = BLayoutAttr::get(ctx, effectiveBLayout);
  auto slAttr = SLayoutAttr::get(ctx, sl.value());
  auto fractalAttr = IntegerAttr::get(IntegerType::get(ctx, 32), fields.fractal);
  auto padAttr = PadValueAttr::get(ctx, pv.value());
  auto memorySpaceAttr = AddressSpaceAttr::get(ctx, memorySpace.value());
  auto cfg = TileBufConfigAttr::get(ctx, blAttr, slAttr, fractalAttr, padAttr);

  SmallVector<int64_t, 2> shape{fields.rows, fields.cols};
  SmallVector<int64_t, 2> validShape{fields.vrow, fields.vcol};
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);

  return TileBufType::get(ctx, shape, fields.dtype, memorySpaceAttr,
                          llvm::ArrayRef<int64_t>(canonicalValidShape), cfg);
}

} // namespace

// ---- TileBufType custom asm ----
// !pto.tile_buf<<loc=.., dtype=.., rows=.., cols=.., blayout=.., valid=..x.., slayout=.., fractal=.., pad=..>>
Type TileBufType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();

  std::string firstToken;
  if (failed(parser.parseKeywordOrString(&firstToken)))
    return Type();

  ParsedTileBufFields fields;
  if (firstToken == "loc") {
    if (failed(parseLegacyTileBufFields(parser, fields)))
      return Type();
  } else {
    if (failed(parseCompactTileBufFields(parser, firstToken, fields)))
      return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  return buildTileBufType(parser, fields);
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
  auto defaultBLayout = llvm::dyn_cast<BLayoutAttr>(defaultCfg.getBLayout());
  auto defaultSLayout = llvm::dyn_cast<SLayoutAttr>(defaultCfg.getSLayout());
  auto defaultPad = llvm::dyn_cast<PadValueAttr>(defaultCfg.getPad());

  auto vs = getValidShape();
  int64_t vrow = rows;
  int64_t vcol = cols;
  if (vs.size() >= 2) {
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

  printer << ">";
}
