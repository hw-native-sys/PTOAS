// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrs.cpp ------------------------------------------------*- C++ -*-===//
#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"          // parseAttribute
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kI32BitWidth = 32;
constexpr int32_t kFractalSize512 = 512;
constexpr int32_t kBLayoutRowMajor = static_cast<int32_t>(BLayout::RowMajor);
constexpr int32_t kBLayoutColMajor = static_cast<int32_t>(BLayout::ColMajor);
constexpr int32_t kSLayoutNoneBox = static_cast<int32_t>(SLayout::NoneBox);
constexpr int32_t kSLayoutColMajor = static_cast<int32_t>(SLayout::ColMajor);
constexpr int32_t kPadValueNull = static_cast<int32_t>(PadValue::Null);
constexpr int32_t kPadValueMin = static_cast<int32_t>(PadValue::Min);
constexpr int32_t kCompactModeNull = static_cast<int32_t>(CompactMode::Null);
constexpr int32_t kCompactModeRowPlusOne =
    static_cast<int32_t>(CompactMode::RowPlusOne);

} // namespace

TileBufConfigAttr TileBufConfigAttr::getDefault(MLIRContext *ctx) {
  Builder b(ctx);
  BLayoutAttr bl = BLayoutAttr::get(ctx, BLayout::RowMajor);
  SLayoutAttr sl = SLayoutAttr::get(ctx, SLayout::NoneBox);
  PadValueAttr pv = PadValueAttr::get(ctx, PadValue::Null);
  CompactModeAttr compact = CompactModeAttr::get(ctx, CompactMode::Null);
  IntegerAttr sz = b.getI32IntegerAttr(kFractalSize512);
  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

bool TileBufConfigAttr::isDefault() const {
  auto d = getDefault(getContext());
  return getBLayout() == d.getBLayout() &&
         getSLayout() == d.getSLayout() &&
         getSFractalSize() == d.getSFractalSize() &&
         getPad() == d.getPad() &&
         getCompactMode() == d.getCompactMode();
}

static int32_t getLayoutInt(Attribute a, int32_t def) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) {
    return static_cast<int32_t>(bl.getValue());
  }
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) {
    return static_cast<int32_t>(sl.getValue());
  }
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) {
    return static_cast<int32_t>(pv.getValue());
  }
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) {
    return static_cast<int32_t>(cm.getValue());
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return static_cast<int32_t>(ia.getInt());
  }
  return def;
}

LogicalResult TileBufConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                       Attribute bLayout,
                                       Attribute sLayout,
                                       IntegerAttr sFractalSize,
                                       Attribute pad,
                                       Attribute compactMode) {
  if (!bLayout || (!mlir::isa<BLayoutAttr>(bLayout) && !mlir::isa<IntegerAttr>(bLayout))) {
    return emitError() << "blayout must be BLayoutAttr or i32 integer attr", failure();
  }
  if (!sLayout || (!mlir::isa<SLayoutAttr>(sLayout) && !mlir::isa<IntegerAttr>(sLayout))) {
    return emitError() << "slayout must be SLayoutAttr or i32 integer attr", failure();
  }
  if (!pad || (!mlir::isa<PadValueAttr>(pad) && !mlir::isa<IntegerAttr>(pad))) {
    return emitError() << "pad must be PadValueAttr or i32 integer attr", failure();
  }
  if (!compactMode ||
      (!mlir::isa<CompactModeAttr>(compactMode) &&
       !mlir::isa<IntegerAttr>(compactMode))) {
    return emitError() << "compact_mode must be CompactModeAttr or i32 integer attr", failure();
  }

  if (!sFractalSize || !sFractalSize.getType().isInteger(kI32BitWidth)) {
    return emitError() << "s_fractal_size must be i32", failure();
  }

  int32_t s = static_cast<int32_t>(sFractalSize.getInt());
  if (s != kFractalMxSize && s != kFractalABSize && s != kFractalCSize) {
    return emitError() << "unsupported s_fractal_size: " << s
                       << ", must be one of {"
                       << kFractalMxSize << ", "
                       << kFractalABSize << ", "
                       << kFractalCSize << "}",
           failure();
  }

  int32_t blv = getLayoutInt(bLayout, -1);
  if (blv != kBLayoutRowMajor && blv != kBLayoutColMajor) {
    return emitError() << "unsupported blayout value: " << blv, failure();
  }

  int32_t slv = getLayoutInt(sLayout, -1);
  if (slv < kSLayoutNoneBox || slv > kSLayoutColMajor) {
    return emitError() << "unsupported slayout value: " << slv, failure();
  }

  int32_t pvv = getLayoutInt(pad, -1);
  if (pvv < kPadValueNull || pvv > kPadValueMin) {
    return emitError() << "unsupported pad value: " << pvv, failure();
  }

  int32_t cmv = getLayoutInt(compactMode, -1);
  if (cmv < kCompactModeNull || cmv > kCompactModeRowPlusOne) {
    return emitError() << "unsupported compact_mode value: " << cmv, failure();
  }

  return success();
}

// Helper: parse Attribute and convert to BLayoutAttr/SLayoutAttr/PadValueAttr
static BLayoutAttr toBLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) {
    return bl;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return BLayoutAttr::get(ctx, static_cast<BLayout>(ia.getInt()));
  }
  return {};
}
static SLayoutAttr toSLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) {
    return sl;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return SLayoutAttr::get(ctx, static_cast<SLayout>(ia.getInt()));
  }
  return {};
}
static PadValueAttr toPadValueAttr(MLIRContext *ctx, Attribute a) {
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) {
    return pv;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return PadValueAttr::get(ctx, static_cast<PadValue>(ia.getInt()));
  }
  return {};
}
static CompactModeAttr toCompactModeAttr(MLIRContext *ctx, Attribute a) {
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) {
    return cm;
  }
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) {
    return CompactModeAttr::get(ctx, static_cast<CompactMode>(ia.getInt()));
  }
  return {};
}

// Parse a single `key = value` field of a tile_buf_config into the matching
// accumulator. Returns failure() (caller returns {}) on a parse or validation
// error, including an unknown key.
static LogicalResult parseTileBufConfigField(AsmParser &odsParser,
                                             MLIRContext *ctx, StringRef key,
                                             BLayoutAttr &bl, SLayoutAttr &sl,
                                             IntegerAttr &sz, PadValueAttr &pv,
                                             CompactModeAttr &compact) {
  if (key == "blayout") {
    Attribute a;
    if (odsParser.parseAttribute(a)) {
      return failure();
    }
    bl = toBLayoutAttr(ctx, a);
    return success(static_cast<bool>(bl));
  }
  if (key == "slayout") {
    Attribute a;
    if (odsParser.parseAttribute(a)) {
      return failure();
    }
    sl = toSLayoutAttr(ctx, a);
    return success(static_cast<bool>(sl));
  }
  if (key == "s_fractal_size") {
    int32_t v = 0;
    if (odsParser.parseInteger(v)) {
      return failure();
    }
    sz = IntegerAttr::get(IntegerType::get(ctx, kI32BitWidth), v);
    return success();
  }
  if (key == "pad") {
    Attribute a;
    if (odsParser.parseAttribute(a)) {
      return failure();
    }
    pv = toPadValueAttr(ctx, a);
    return success(static_cast<bool>(pv));
  }
  if (key == "compact") {
    Attribute a;
    if (odsParser.parseAttribute(a)) {
      return failure();
    }
    compact = toCompactModeAttr(ctx, a);
    return success(static_cast<bool>(compact));
  }
  odsParser.emitError(odsParser.getCurrentLocation(),
                      "unknown key in tile_buf_config: ")
      << key;
  return failure();
}

Attribute TileBufConfigAttr::parse(AsmParser &odsParser, [[maybe_unused]] Type odsType) {
  MLIRContext *ctx = odsParser.getContext();
  auto def = TileBufConfigAttr::getDefault(ctx);
  BLayoutAttr bl = def.getBLayout();
  SLayoutAttr sl = def.getSLayout();
  IntegerAttr sz = def.getSFractalSize();
  PadValueAttr pv = def.getPad();
  CompactModeAttr compact = def.getCompactMode();

  if (odsParser.parseLess()) {
    return {};
  }

  if (succeeded(odsParser.parseOptionalGreater())) {
    return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
  }

  bool closed = false;
  while (!closed) {
    StringRef key;
    if (odsParser.parseKeyword(&key) || odsParser.parseEqual()) {
      return {};
    }
    if (failed(parseTileBufConfigField(odsParser, ctx, key, bl, sl, sz, pv,
                                       compact))) {
      return {};
    }
    if (succeeded(odsParser.parseOptionalGreater())) {
      closed = true;
    } else if (odsParser.parseComma()) {
      return {};
    }
  }

  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

void TileBufConfigAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << "blayout=" << getBLayout();
  odsPrinter << ", slayout=" << getSLayout();
  odsPrinter << ", s_fractal_size=" << static_cast<int32_t>(getSFractalSize().getInt());
  odsPrinter << ", pad=" << getPad();
  odsPrinter << ", compact=" << getCompactMode();
  odsPrinter << ">";
}

namespace {

constexpr unsigned kConvTilePadListSize = 4;
constexpr int32_t kConvTileDefaultZero = 0;
constexpr int32_t kConvTileDefaultOne = 1;

static LogicalResult parseConvTileIntField(AsmParser &parser,
                                           IntegerAttr &value) {
  int64_t parsed = 0;
  if (failed(parser.parseInteger(parsed))) {
    return failure();
  }
  value = IntegerAttr::get(IntegerType::get(parser.getContext(), kI32BitWidth),
                           parsed);
  return success();
}

static LogicalResult parseConvTileBoolField(AsmParser &parser, StringRef key,
                                            BoolAttr &value) {
  bool parsed = false;
  if (succeeded(parser.parseOptionalKeyword("true"))) {
    parsed = true;
  } else if (succeeded(parser.parseOptionalKeyword("false"))) {
    parsed = false;
  } else {
    parser.emitError(parser.getCurrentLocation())
        << key << " must be true or false";
    return failure();
  }
  value = BoolAttr::get(parser.getContext(), parsed);
  return success();
}

static LogicalResult parseConvTileAttrField(AsmParser &parser,
                                            Attribute &value) {
  if (failed(parser.parseAttribute(value))) {
    return failure();
  }
  return success();
}

static LogicalResult parseConvTilePadListField(AsmParser &parser,
                                               SmallVectorImpl<int64_t> &padList) {
  SmallVector<int64_t, kConvTilePadListSize> parsed;
  if (failed(parser.parseDimensionList(parsed, /*allowDynamic=*/false,
                                       /*withTrailingX=*/false))) {
    return failure();
  }
  if (parsed.size() != kConvTilePadListSize) {
    parser.emitError(parser.getCurrentLocation())
        << "pad_list must have exactly " << kConvTilePadListSize
        << " dimensions";
    return failure();
  }
  padList.assign(parsed.begin(), parsed.end());
  return success();
}

static void printConvTileIntField(AsmPrinter &p, StringRef key, IntegerAttr value) {
  p << key << "=" << value.getInt();
}

static void printConvTilePadList(AsmPrinter &p, ArrayRef<int64_t> padList) {
  for (auto it = padList.begin(); it != padList.end(); ++it) {
    if (it != padList.begin()) {
      p << "x";
    }
    p << *it;
  }
}

static StringRef printBoolAttr(BoolAttr value) {
  return value.getValue() ? "true" : "false";
}

} // namespace

ConvTileConfigAttr ConvTileConfigAttr::getDefault(MLIRContext *ctx) {
  Builder b(ctx);
  auto zero = b.getI32IntegerAttr(kConvTileDefaultZero);
  auto one = b.getI32IntegerAttr(kConvTileDefaultOne);
  SmallVector<int64_t, kConvTilePadListSize> padList(kConvTilePadListSize,
                                                     kConvTileDefaultZero);
  auto padValue = b.getI32IntegerAttr(kConvTileDefaultZero);
  auto transpose = BoolAttr::get(ctx, false);
  return ConvTileConfigAttr::get(ctx, zero, zero, padList, one, one, one, one,
                                 one, one, padValue, zero, zero, one, zero,
                                 zero, zero, transpose);
}

bool ConvTileConfigAttr::isDefault() const {
  auto d = getDefault(getContext());
  return getFmapH() == d.getFmapH() && getFmapW() == d.getFmapW() &&
         getPadList() == d.getPadList() && getFilterH() == d.getFilterH() &&
         getFilterW() == d.getFilterW() &&
         getDilationH() == d.getDilationH() &&
         getDilationW() == d.getDilationW() &&
         getStrideH() == d.getStrideH() && getStrideW() == d.getStrideW() &&
         getPadValue() == d.getPadValue() &&
         getChannelSize() == d.getChannelSize() &&
         getRepeatStride() == d.getRepeatStride() &&
         getRepeatTime() == d.getRepeatTime() &&
         getRepeatMode() == d.getRepeatMode() &&
         getDstStride() == d.getDstStride() &&
         getDstMposition() == d.getDstMposition() &&
         getTranspose() == d.getTranspose();
}

LogicalResult ConvTileConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                         IntegerAttr fmapH,
                                         IntegerAttr fmapW,
                                         ArrayRef<int64_t> padList,
                                         IntegerAttr filterH,
                                         IntegerAttr filterW,
                                         IntegerAttr dilationH,
                                         IntegerAttr dilationW,
                                         IntegerAttr strideH,
                                         IntegerAttr strideW,
                                         Attribute padValue,
                                         IntegerAttr channelSize,
                                         IntegerAttr repeatStride,
                                         IntegerAttr repeatTime,
                                         IntegerAttr repeatMode,
                                         IntegerAttr dstStride,
                                         IntegerAttr dstMposition,
                                         BoolAttr transpose) {
  auto checkI32 = [&emitError](IntegerAttr attr, StringRef name) -> LogicalResult {
    if (!attr || !attr.getType().isSignlessInteger(kI32BitWidth)) {
      return emitError() << name << " must be an i32 integer attr", failure();
    }
    return success();
  };
  if (failed(checkI32(fmapH, "fmap_h")) || failed(checkI32(fmapW, "fmap_w")) ||
      failed(checkI32(filterH, "filter_h")) ||
      failed(checkI32(filterW, "filter_w")) ||
      failed(checkI32(dilationH, "dilation_h")) ||
      failed(checkI32(dilationW, "dilation_w")) ||
      failed(checkI32(strideH, "stride_h")) ||
      failed(checkI32(strideW, "stride_w")) ||
      failed(checkI32(channelSize, "channel_size")) ||
      failed(checkI32(repeatStride, "repeat_stride")) ||
      failed(checkI32(repeatTime, "repeat_time")) ||
      failed(checkI32(repeatMode, "repeat_mode")) ||
      failed(checkI32(dstStride, "dst_stride")) ||
      failed(checkI32(dstMposition, "dst_mposition"))) {
    return failure();
  }

  if (!transpose) {
    return emitError() << "transpose must be a bool attr", failure();
  }

  if (padList.size() != kConvTilePadListSize) {
    return emitError() << "pad_list must have exactly 4 elements", failure();
  }
  for (int64_t dim : padList) {
      if (dim < 0 || dim > mlir::pto::kValue255) {
          return emitError() << "pad_list entries must be in [0, 255]", failure();
      }
  }

  auto checkNonNegative = [&emitError](IntegerAttr attr, StringRef name) -> LogicalResult {
    if (attr.getInt() < 0) {
      return emitError() << name << " must be non-negative", failure();
    }
    return success();
  };
  if (failed(checkNonNegative(fmapH, "fmap_h")) ||
      failed(checkNonNegative(fmapW, "fmap_w")) ||
      failed(checkNonNegative(filterH, "filter_h")) ||
      failed(checkNonNegative(filterW, "filter_w")) ||
      failed(checkNonNegative(dilationH, "dilation_h")) ||
      failed(checkNonNegative(dilationW, "dilation_w")) ||
      failed(checkNonNegative(strideH, "stride_h")) ||
      failed(checkNonNegative(strideW, "stride_w")) ||
      failed(checkNonNegative(channelSize, "channel_size")) ||
      failed(checkNonNegative(repeatStride, "repeat_stride")) ||
      failed(checkNonNegative(repeatTime, "repeat_time")) ||
      failed(checkNonNegative(repeatMode, "repeat_mode")) ||
      failed(checkNonNegative(dstStride, "dst_stride")) ||
      failed(checkNonNegative(dstMposition, "dst_mposition"))) {
    return failure();
  }

  // Upper bounds mirror the pto-isa conv config storage: the getters for
  // dilation/stride/repeat_time/repeat_mode hand out uint8_t and every other
  // counter uint16_t, so a wider value would silently truncate on the way
  // into the runtime (e.g. stride_h = 256 becomes 0).
  constexpr int64_t kU8Max = 255;
  constexpr int64_t kU16Max = 65535;
  auto checkUpperBound = [&emitError](IntegerAttr attr, int64_t bound,
                                      StringRef name) -> LogicalResult {
    if (attr.getInt() > bound) {
      return emitError() << name << " must be in [0, " << bound << "]", failure();
    }
    return success();
  };
  if (failed(checkUpperBound(dilationH, kU8Max, "dilation_h")) ||
      failed(checkUpperBound(dilationW, kU8Max, "dilation_w")) ||
      failed(checkUpperBound(strideH, kU8Max, "stride_h")) ||
      failed(checkUpperBound(strideW, kU8Max, "stride_w")) ||
      failed(checkUpperBound(repeatTime, kU8Max, "repeat_time")) ||
      failed(checkUpperBound(repeatMode, kU8Max, "repeat_mode")) ||
      failed(checkUpperBound(fmapH, kU16Max, "fmap_h")) ||
      failed(checkUpperBound(fmapW, kU16Max, "fmap_w")) ||
      failed(checkUpperBound(filterH, kU16Max, "filter_h")) ||
      failed(checkUpperBound(filterW, kU16Max, "filter_w")) ||
      failed(checkUpperBound(channelSize, kU16Max, "channel_size")) ||
      failed(checkUpperBound(repeatStride, kU16Max, "repeat_stride")) ||
      failed(checkUpperBound(dstStride, kU16Max, "dst_stride")) ||
      failed(checkUpperBound(dstMposition, kU16Max, "dst_mposition"))) {
    return failure();
  }

  if (!padValue ||
      (!llvm::isa<IntegerAttr>(padValue) &&
       !llvm::isa<FloatAttr>(padValue))) {
    return emitError() << "pad_value must be an integer or float attr", failure();
  }

  return success();
}

namespace {

struct ConvTileConfigFields {
  IntegerAttr fmapH;
  IntegerAttr fmapW;
  IntegerAttr filterH;
  IntegerAttr filterW;
  IntegerAttr dilationH;
  IntegerAttr dilationW;
  IntegerAttr strideH;
  IntegerAttr strideW;
  IntegerAttr channelSize;
  IntegerAttr repeatStride;
  IntegerAttr repeatTime;
  IntegerAttr repeatMode;
  IntegerAttr dstStride;
  IntegerAttr dstMposition;
  SmallVector<int64_t, kConvTilePadListSize> padList;
  Attribute padValue;
  BoolAttr transpose;
};

static ConvTileConfigFields getDefaultConvTileConfigFields(MLIRContext *ctx) {
  ConvTileConfigAttr def = ConvTileConfigAttr::getDefault(ctx);
  ConvTileConfigFields fields;
  fields.fmapH = def.getFmapH();
  fields.fmapW = def.getFmapW();
  fields.filterH = def.getFilterH();
  fields.filterW = def.getFilterW();
  fields.dilationH = def.getDilationH();
  fields.dilationW = def.getDilationW();
  fields.strideH = def.getStrideH();
  fields.strideW = def.getStrideW();
  fields.channelSize = def.getChannelSize();
  fields.repeatStride = def.getRepeatStride();
  fields.repeatTime = def.getRepeatTime();
  fields.repeatMode = def.getRepeatMode();
  fields.dstStride = def.getDstStride();
  fields.dstMposition = def.getDstMposition();
  fields.padList.assign(def.getPadList().begin(), def.getPadList().end());
  fields.padValue = def.getPadValue();
  fields.transpose = def.getTranspose();
  return fields;
}

static Attribute
getConvTileConfigAttr(function_ref<InFlightDiagnostic()> emitError,
                      MLIRContext *ctx, const ConvTileConfigFields &fields) {
  // getChecked forwards its arguments to the storage constructor without the
  // implicit conversions a fixed-signature get() would apply, so padList
  // must be spelled as the ArrayRef the storage takes.
  return ConvTileConfigAttr::getChecked(
      emitError, ctx, fields.fmapH, fields.fmapW,
      llvm::ArrayRef<int64_t>(fields.padList), fields.filterH,
      fields.filterW, fields.dilationH, fields.dilationW, fields.strideH,
      fields.strideW, fields.padValue, fields.channelSize,
      fields.repeatStride, fields.repeatTime, fields.repeatMode,
      fields.dstStride, fields.dstMposition, fields.transpose);
}

static LogicalResult parseConvTileConfigField(AsmParser &parser,
                                              StringRef key,
                                              ConvTileConfigFields &fields) {
  static constexpr struct {
    StringRef name;
    IntegerAttr ConvTileConfigFields::*field;
  } kIntFields[] = {
      {"fmap_h", &ConvTileConfigFields::fmapH},
      {"fmap_w", &ConvTileConfigFields::fmapW},
      {"filter_h", &ConvTileConfigFields::filterH},
      {"filter_w", &ConvTileConfigFields::filterW},
      {"dilation_h", &ConvTileConfigFields::dilationH},
      {"dilation_w", &ConvTileConfigFields::dilationW},
      {"stride_h", &ConvTileConfigFields::strideH},
      {"stride_w", &ConvTileConfigFields::strideW},
      {"channel_size", &ConvTileConfigFields::channelSize},
      {"repeat_stride", &ConvTileConfigFields::repeatStride},
      {"repeat_time", &ConvTileConfigFields::repeatTime},
      {"repeat_mode", &ConvTileConfigFields::repeatMode},
      {"dst_stride", &ConvTileConfigFields::dstStride},
      {"dst_mposition", &ConvTileConfigFields::dstMposition},
  };
  for (const auto &entry : kIntFields) {
    if (key == entry.name) {
      return parseConvTileIntField(parser, fields.*(entry.field));
    }
  }
  if (key == "pad_list") {
    return parseConvTilePadListField(parser, fields.padList);
  }
  if (key == "pad_value") {
    return parseConvTileAttrField(parser, fields.padValue);
  }
  if (key == "transpose") {
    return parseConvTileBoolField(parser, key, fields.transpose);
  }
  parser.emitError(parser.getCurrentLocation(),
                   "unknown key in conv_tile_config: ")
      << key;
  return failure();
}

} // namespace

Attribute ConvTileConfigAttr::parse(AsmParser &odsParser, Type odsType)
{
    (void)odsType;
  MLIRContext *ctx = odsParser.getContext();
  ConvTileConfigFields fields = getDefaultConvTileConfigFields(ctx);
  auto emitError = [&odsParser]() -> InFlightDiagnostic {
    return odsParser.emitError(odsParser.getCurrentLocation());
  };
  bool parsedGreater = false;

  auto consumeFieldTerminator = [&odsParser, &parsedGreater]() -> LogicalResult {
    if (succeeded(odsParser.parseOptionalGreater())) {
      parsedGreater = true;
      return success();
    }
    return odsParser.parseComma();
  };

  if (odsParser.parseLess()) {
    return {};
  }

  if (succeeded(odsParser.parseOptionalGreater())) {
    return getConvTileConfigAttr(emitError, ctx, fields);
  }

  while (!parsedGreater) {
    StringRef key;
    if (odsParser.parseKeyword(&key) || odsParser.parseEqual()) {
      return {};
    }
    if (failed(parseConvTileConfigField(odsParser, key, fields)) ||
        failed(consumeFieldTerminator())) {
      return {};
    }
  }

  return getConvTileConfigAttr(emitError, ctx, fields);
}

void ConvTileConfigAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  printConvTileIntField(odsPrinter, "fmap_h", getFmapH());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "fmap_w", getFmapW());
  odsPrinter << ", pad_list=";
  printConvTilePadList(odsPrinter, getPadList());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "filter_h", getFilterH());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "filter_w", getFilterW());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "dilation_h", getDilationH());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "dilation_w", getDilationW());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "stride_h", getStrideH());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "stride_w", getStrideW());
  odsPrinter << ", pad_value=" << getPadValue();
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "channel_size", getChannelSize());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "repeat_stride", getRepeatStride());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "repeat_time", getRepeatTime());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "repeat_mode", getRepeatMode());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "dst_stride", getDstStride());
  odsPrinter << ", ";
  printConvTileIntField(odsPrinter, "dst_mposition", getDstMposition());
  odsPrinter << ", transpose=" << printBoolAttr(getTranspose());
  odsPrinter << ">";
}
