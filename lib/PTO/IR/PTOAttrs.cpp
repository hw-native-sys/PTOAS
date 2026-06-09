// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrs.cpp ------------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
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
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) return static_cast<int32_t>(bl.getValue());
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) return static_cast<int32_t>(sl.getValue());
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) return static_cast<int32_t>(pv.getValue());
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) return static_cast<int32_t>(cm.getValue());
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return static_cast<int32_t>(ia.getInt());
  return def;
}

static LogicalResult verifyTileBufConfigAttrKind(
    function_ref<InFlightDiagnostic()> emitError, StringRef attrName,
    bool isValid, StringRef expectedType) {
  if (isValid)
    return success();
  return emitError() << attrName << " must be " << expectedType, failure();
}

static bool isSupportedFractalSize(int32_t size) {
  return size == kFractalSize16 || size == kFractalSize32 ||
         size == kFractalSize512 || size == kFractalSize1024;
}

static LogicalResult verifyTileBufConfigRange(
    function_ref<InFlightDiagnostic()> emitError, StringRef attrName,
    int32_t value, bool isValid) {
  if (isValid)
    return success();
  return emitError() << "unsupported " << attrName << " value: " << value,
         failure();
}

static LogicalResult verifyTileBufConfigAttrKinds(
    function_ref<InFlightDiagnostic()> emitError, Attribute bLayout,
    Attribute sLayout, Attribute pad, Attribute compactMode) {
  if (failed(verifyTileBufConfigAttrKind(
          emitError, "blayout",
          bLayout && (mlir::isa<BLayoutAttr>(bLayout) ||
                      mlir::isa<IntegerAttr>(bLayout)),
          "BLayoutAttr or i32 integer attr")))
    return failure();
  if (failed(verifyTileBufConfigAttrKind(
          emitError, "slayout",
          sLayout && (mlir::isa<SLayoutAttr>(sLayout) ||
                      mlir::isa<IntegerAttr>(sLayout)),
          "SLayoutAttr or i32 integer attr")))
    return failure();
  if (failed(verifyTileBufConfigAttrKind(
          emitError, "pad",
          pad &&
              (mlir::isa<PadValueAttr>(pad) || mlir::isa<IntegerAttr>(pad)),
          "PadValueAttr or i32 integer attr")))
    return failure();
  return verifyTileBufConfigAttrKind(
      emitError, "compact_mode",
      compactMode && (mlir::isa<CompactModeAttr>(compactMode) ||
                      mlir::isa<IntegerAttr>(compactMode)),
      "CompactModeAttr or i32 integer attr");
}

static LogicalResult verifyTileBufConfigValues(
    function_ref<InFlightDiagnostic()> emitError, Attribute bLayout,
    Attribute sLayout, IntegerAttr sFractalSize, Attribute pad,
    Attribute compactMode) {
  if (!sFractalSize || !sFractalSize.getType().isInteger(kI32BitWidth))
    return emitError() << "s_fractal_size must be i32", failure();

  int32_t s = static_cast<int32_t>(sFractalSize.getInt());
  if (!isSupportedFractalSize(s))
    return emitError() << "unsupported s_fractal_size: " << s, failure();

  if (failed(verifyTileBufConfigRange(
          emitError, "blayout", getLayoutInt(bLayout, -1),
          getLayoutInt(bLayout, -1) == kBLayoutRowMajor ||
              getLayoutInt(bLayout, -1) == kBLayoutColMajor)))
    return failure();
  if (failed(verifyTileBufConfigRange(
          emitError, "slayout", getLayoutInt(sLayout, -1),
          getLayoutInt(sLayout, -1) >= kSLayoutNoneBox &&
              getLayoutInt(sLayout, -1) <= kSLayoutColMajor)))
    return failure();
  if (failed(verifyTileBufConfigRange(
          emitError, "pad", getLayoutInt(pad, -1),
          getLayoutInt(pad, -1) >= kPadValueNull &&
              getLayoutInt(pad, -1) <= kPadValueMin)))
    return failure();
  int32_t cmv = getLayoutInt(compactMode, -1);
  return verifyTileBufConfigRange(
      emitError, "compact_mode", cmv,
      cmv >= kCompactModeNull && cmv <= kCompactModeRowPlusOne);
}

LogicalResult TileBufConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                       Attribute bLayout,
                                       Attribute sLayout,
                                       IntegerAttr sFractalSize,
                                       Attribute pad,
                                       Attribute compactMode) {
  if (failed(verifyTileBufConfigAttrKinds(emitError, bLayout, sLayout, pad,
                                          compactMode)))
    return failure();
  return verifyTileBufConfigValues(emitError, bLayout, sLayout, sFractalSize,
                                   pad, compactMode);
}

// Helper: parse Attribute and convert to BLayoutAttr/SLayoutAttr/PadValueAttr
static BLayoutAttr toBLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) return bl;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return BLayoutAttr::get(ctx, static_cast<BLayout>(ia.getInt()));
  return {};
}
static SLayoutAttr toSLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) return sl;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return SLayoutAttr::get(ctx, static_cast<SLayout>(ia.getInt()));
  return {};
}
static PadValueAttr toPadValueAttr(MLIRContext *ctx, Attribute a) {
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) return pv;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return PadValueAttr::get(ctx, static_cast<PadValue>(ia.getInt()));
  return {};
}
static CompactModeAttr toCompactModeAttr(MLIRContext *ctx, Attribute a) {
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) return cm;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a))
    return CompactModeAttr::get(ctx, static_cast<CompactMode>(ia.getInt()));
  return {};
}

static ParseResult parseTileBufConfigAttrValue(
    AsmParser &p, MLIRContext *ctx, StringRef key, BLayoutAttr &bl,
    SLayoutAttr &sl, IntegerAttr &sz, PadValueAttr &pv,
    CompactModeAttr &compact) {
  if (key == "blayout") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    bl = toBLayoutAttr(ctx, a);
    return success(static_cast<bool>(bl));
  }
  if (key == "slayout") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    sl = toSLayoutAttr(ctx, a);
    return success(static_cast<bool>(sl));
  }
  if (key == "s_fractal_size") {
    int32_t v = 0;
    if (p.parseInteger(v))
      return failure();
    sz = IntegerAttr::get(IntegerType::get(ctx, kI32BitWidth), v);
    return success();
  }
  if (key == "pad") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    pv = toPadValueAttr(ctx, a);
    return success(static_cast<bool>(pv));
  }
  if (key == "compact") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    compact = toCompactModeAttr(ctx, a);
    return success(static_cast<bool>(compact));
  }
  p.emitError(p.getCurrentLocation(), "unknown key in tile_buf_config: ")
      << key;
  return failure();
}

Attribute TileBufConfigAttr::parse(AsmParser &odsParser, Type odsType) {
  (void)odsType;
  MLIRContext *ctx = odsParser.getContext();
  auto def = TileBufConfigAttr::getDefault(ctx);
  BLayoutAttr bl = def.getBLayout();
  SLayoutAttr sl = def.getSLayout();
  IntegerAttr sz = def.getSFractalSize();
  PadValueAttr pv = def.getPad();
  CompactModeAttr compact = def.getCompactMode();

  if (odsParser.parseLess()) return {};

  if (succeeded(odsParser.parseOptionalGreater()))
    return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);

  bool parsedGreater = false;
  while (!parsedGreater) {
    StringRef key;
    if (odsParser.parseKeyword(&key) || odsParser.parseEqual() ||
        failed(parseTileBufConfigAttrValue(odsParser, ctx, key, bl, sl, sz, pv,
                                           compact)))
      return {};

    parsedGreater = succeeded(odsParser.parseOptionalGreater());
    if (parsedGreater)
      break;
    if (odsParser.parseComma()) return {};
  }

  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

void TileBufConfigAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << "blayout=" << getBLayout();
  odsPrinter << ", slayout=" << getSLayout();
  odsPrinter << ", s_fractal_size="
    << static_cast<int32_t>(getSFractalSize().getInt());
  odsPrinter << ", pad=" << getPad();
  odsPrinter << ", compact=" << getCompactMode();
  odsPrinter << ">";
}
