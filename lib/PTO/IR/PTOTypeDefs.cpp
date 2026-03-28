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
// 支持旧格式：
//   !pto.tile_buf<loc=.., dtype=.., rows=.., cols=.., v_row=.., v_col=.., blayout=.., slayout=.., fractal=.., pad=..>
// 支持新格式（默认打印）：
//   !pto.tile_buf<vec, 16x16xf16, valid=16x8, blayout=ColMajor>
Type TileBufType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  if (failed(parser.parseLess()))
    return Type();

  auto defCfg = TileBufConfigAttr::getDefault(ctx);

  std::optional<AddressSpace> memorySpace;
  Type dtype;
  bool hasDType = false;
  int64_t rows = 0, cols = 0;
  bool hasRows = false, hasCols = false;
  int64_t vrow = ShapedType::kDynamic, vcol = ShapedType::kDynamic;
  bool hasValid = false;
  bool hasVRow = false;
  bool hasVCol = false;
  BLayout blayout = defCfg.getBLayout().getValue();
  SLayout slayout = defCfg.getSLayout().getValue();
  int64_t fractal = defCfg.getSFractalSize().getInt();
  PadValue pad = defCfg.getPad().getValue();

  auto parseDim = [&](int64_t &dim) -> LogicalResult {
    if (succeeded(parser.parseOptionalQuestion())) {
      dim = ShapedType::kDynamic;
      return success();
    }
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();
    if (value < 0) {
      parser.emitError(parser.getCurrentLocation(),
                       "dimension must be '?' or a non-negative integer");
      return failure();
    }
    dim = value;
    return success();
  };

  auto parseValidShape = [&]() -> LogicalResult {
    SmallVector<int64_t, 2> dims;
    if (failed(parser.parseDimensionList(dims, /*allowDynamic=*/true,
                                         /*withTrailingX=*/false)))
      return failure();
    if (dims.size() != 2) {
      parser.emitError(parser.getCurrentLocation(),
                       "valid shape expects exactly 2 dimensions");
      return failure();
    }
    vrow = dims[0];
    vcol = dims[1];
    hasValid = true;
    hasVRow = true;
    hasVCol = true;
    return success();
  };

  auto setRowsCols = [&](int64_t r, int64_t c) -> LogicalResult {
    if ((hasRows && rows != r) || (hasCols && cols != c)) {
      parser.emitError(parser.getCurrentLocation(),
                       "conflicting rows/cols values");
      return failure();
    }
    rows = r;
    cols = c;
    hasRows = true;
    hasCols = true;
    return success();
  };

  auto setDType = [&](Type t) -> LogicalResult {
    if (hasDType && t != dtype) {
      parser.emitError(parser.getCurrentLocation(),
                       "conflicting dtype values");
      return failure();
    }
    dtype = t;
    hasDType = true;
    return success();
  };

  auto parseLocValue = [&](StringRef locStr) -> LogicalResult {
    auto ms = ::llvm::StringSwitch<::std::optional<AddressSpace>>(locStr)
        .Case("mat", AddressSpace::MAT)
        .Case("left", AddressSpace::LEFT)
        .Case("right", AddressSpace::RIGHT)
        .Case("acc", AddressSpace::ACC)
        .Case("vec", AddressSpace::VEC)
        .Case("bias", AddressSpace::BIAS)
        .Case("scaling", AddressSpace::SCALING)
        .Default(::std::nullopt);
    if (!ms.has_value()) {
      parser.emitError(parser.getNameLoc(), "unknown loc: ") << locStr;
      return failure();
    }
    if (memorySpace.has_value() && memorySpace.value() != ms.value()) {
      parser.emitError(parser.getCurrentLocation(),
                       "conflicting loc values");
      return failure();
    }
    memorySpace = ms.value();
    return success();
  };

  while (true) {
    if (succeeded(parser.parseOptionalGreater()))
      break;

    bool parsedClause = false;
    // Positional shape: 16x16xf16
    {
      int64_t firstDim = 0;
      if (succeeded(parser.parseOptionalQuestion())) {
        firstDim = ShapedType::kDynamic;
        parsedClause = true;
      } else {
        OptionalParseResult intRes = parser.parseOptionalInteger(firstDim);
        if (intRes.has_value()) {
          if (failed(*intRes))
            return Type();
          parsedClause = true;
        }
      }
      if (parsedClause) {
        int64_t secondDim = 0;
        if (failed(parser.parseXInDimensionList()))
          return Type();
        if (failed(parseDim(secondDim)))
          return Type();
        if (failed(parser.parseXInDimensionList()))
          return Type();
        Type elemTy;
        if (failed(parser.parseType(elemTy)))
          return Type();
        if (failed(setRowsCols(firstDim, secondDim)))
          return Type();
        if (failed(setDType(elemTy)))
          return Type();
      }
    }

    if (!parsedClause) {
      std::string keyOrToken;
      if (failed(parser.parseKeywordOrString(&keyOrToken)))
        return Type();
      if (succeeded(parser.parseOptionalEqual())) {
        if (keyOrToken == "loc") {
          std::string locStr;
          if (failed(parser.parseKeywordOrString(&locStr)))
            return Type();
          if (failed(parseLocValue(locStr)))
            return Type();
        } else if (keyOrToken == "dtype") {
          Type ty;
          if (failed(parser.parseType(ty)))
            return Type();
          if (failed(setDType(ty)))
            return Type();
        } else if (keyOrToken == "rows") {
          int64_t r = 0;
          if (failed(parser.parseInteger(r)))
            return Type();
          if (r < 0) {
            parser.emitError(parser.getCurrentLocation(),
                             "rows must be non-negative");
            return Type();
          }
          if (hasRows && rows != r) {
            parser.emitError(parser.getCurrentLocation(),
                             "conflicting rows values");
            return Type();
          }
          rows = r;
          hasRows = true;
        } else if (keyOrToken == "cols") {
          int64_t c = 0;
          if (failed(parser.parseInteger(c)))
            return Type();
          if (c < 0) {
            parser.emitError(parser.getCurrentLocation(),
                             "cols must be non-negative");
            return Type();
          }
          if (hasCols && cols != c) {
            parser.emitError(parser.getCurrentLocation(),
                             "conflicting cols values");
            return Type();
          }
          cols = c;
          hasCols = true;
        } else if (keyOrToken == "v_row") {
          int64_t vr = 0;
          if (succeeded(parser.parseOptionalQuestion())) {
            vr = ShapedType::kDynamic;
          } else {
            if (failed(parser.parseInteger(vr)))
              return Type();
            if (vr < -1) {
              parser.emitError(parser.getCurrentLocation(),
                               "v_row must be '?', -1, or a non-negative integer");
              return Type();
            }
            if (vr == -1)
              vr = ShapedType::kDynamic;
          }
          vrow = vr;
          hasValid = true;
          hasVRow = true;
        } else if (keyOrToken == "v_col") {
          int64_t vc = 0;
          if (succeeded(parser.parseOptionalQuestion())) {
            vc = ShapedType::kDynamic;
          } else {
            if (failed(parser.parseInteger(vc)))
              return Type();
            if (vc < -1) {
              parser.emitError(parser.getCurrentLocation(),
                               "v_col must be '?', -1, or a non-negative integer");
              return Type();
            }
            if (vc == -1)
              vc = ShapedType::kDynamic;
          }
          vcol = vc;
          hasValid = true;
          hasVCol = true;
        } else if (keyOrToken == "valid") {
          if (failed(parseValidShape()))
            return Type();
        } else if (keyOrToken == "blayout") {
          std::string blayoutStr;
          if (failed(parser.parseKeywordOrString(&blayoutStr)))
            return Type();
          auto bl = symbolizeBLayout(blayoutStr);
          if (!bl.has_value()) {
            parser.emitError(parser.getNameLoc(), "unknown blayout: ")
                << blayoutStr;
            return Type();
          }
          blayout = bl.value();
        } else if (keyOrToken == "slayout") {
          std::string slayoutStr;
          if (failed(parser.parseKeywordOrString(&slayoutStr)))
            return Type();
          auto sl = symbolizeSLayout(slayoutStr);
          if (!sl.has_value()) {
            parser.emitError(parser.getNameLoc(), "unknown slayout: ")
                << slayoutStr;
            return Type();
          }
          slayout = sl.value();
        } else if (keyOrToken == "fractal") {
          if (failed(parser.parseInteger(fractal)))
            return Type();
        } else if (keyOrToken == "pad") {
          uint32_t padInt = 0;
          if (failed(parser.parseInteger(padInt)))
            return Type();
          auto pv = symbolizePadValue(padInt);
          if (!pv.has_value()) {
            parser.emitError(parser.getNameLoc(), "unknown pad: ") << padInt;
            return Type();
          }
          pad = pv.value();
        } else {
          parser.emitError(parser.getNameLoc(), "unknown key in tile_buf: ")
              << keyOrToken;
          return Type();
        }
      } else {
        if (failed(parseLocValue(keyOrToken)))
          return Type();
      }
    }

    if (succeeded(parser.parseOptionalGreater()))
      break;
    if (failed(parser.parseComma()))
      return Type();
  }

  if (!memorySpace.has_value()) {
    parser.emitError(parser.getNameLoc(), "missing loc");
    return Type();
  }
  if (!hasDType) {
    parser.emitError(parser.getNameLoc(), "missing dtype");
    return Type();
  }
  if (!(hasRows && hasCols)) {
    parser.emitError(parser.getNameLoc(), "missing rows/cols");
    return Type();
  }
  if (!hasValid) {
    vrow = rows;
    vcol = cols;
  } else if (hasVRow != hasVCol) {
    parser.emitError(parser.getNameLoc(),
                     "v_row and v_col must be provided together");
    return Type();
  }
  if (rows < 0 || cols < 0) {
    parser.emitError(parser.getNameLoc(), "rows/cols must be non-negative");
    return Type();
  }

  // -------- 语义校验/构造 --------
  auto blAttr = BLayoutAttr::get(ctx, blayout);
  auto slAttr = SLayoutAttr::get(ctx, slayout);
  auto fractalAttr =
      IntegerAttr::get(IntegerType::get(ctx, 32), fractal);
  auto padAttr = PadValueAttr::get(ctx, pad);
  auto memorySpaceAttr = AddressSpaceAttr::get(ctx, memorySpace.value());
  auto cfg = TileBufConfigAttr::get(ctx, blAttr, slAttr, fractalAttr, padAttr);

  SmallVector<int64_t, 2> shape{rows, cols};
  SmallVector<int64_t, 2> validShape{vrow, vcol};

  return TileBufType::get(ctx, shape, dtype, memorySpaceAttr, llvm::ArrayRef<int64_t>(validShape), cfg);
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
    int64_t rows = shape.size() > 0 ? shape[0] : ShapedType::kDynamic;
    int64_t cols = shape.size() > 1 ? shape[1] : ShapedType::kDynamic;

    auto cfg = getConfigAttr();
    if (!cfg) cfg = mlir::pto::TileBufConfigAttr::getDefault(getContext());
    auto defCfg = mlir::pto::TileBufConfigAttr::getDefault(getContext());

    llvm::StringRef locStr = stringifyLocFromMemorySpace(getMemorySpace());

    printer << "<" << locStr << ", ";
    auto printDim = [&](int64_t dim) {
      if (dim < 0)
        printer << "?";
      else
        printer << dim;
    };
    printDim(rows);
    printer << "x";
    printDim(cols);
    printer << "x";
    printer.printType(getElementType());

    auto vs = getValidShape(); // ArrayRef<int64_t>
    int64_t vrow = rows;
    int64_t vcol = cols;
    if (vs.size() >= 2) {
      vrow = vs[0];
      vcol = vs[1];
    }
    if (!(vrow == rows && vcol == cols)) {
      printer << ", valid=";
      printDim(vrow);
      printer << "x";
      printDim(vcol);
    }

    auto blayout = llvm::dyn_cast<BLayoutAttr>(cfg.getBLayout());
    auto slayout = llvm::dyn_cast<SLayoutAttr>(cfg.getSLayout());
    if (cfg.getBLayout() != defCfg.getBLayout())
      printer << ", blayout=" << stringifyBLayout(blayout.getValue());
    if (cfg.getSLayout() != defCfg.getSLayout())
      printer << ", slayout=" << stringifySLayout(slayout.getValue());
    if (cfg.getSFractalSize() != defCfg.getSFractalSize())
      printer << ", fractal=" << cfg.getSFractalSize().getInt();
    if (cfg.getPad() != defCfg.getPad())
      printer << ", pad=" << stringifyLocFromPad(cfg.getPad());

    printer << ">";
}
