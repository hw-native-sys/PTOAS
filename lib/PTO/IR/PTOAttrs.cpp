// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrs.cpp ------------------------------------------------*- C++ -*-===//
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

llvm::StringRef AccPhaseAttr::getEnumCaseSymbol() const {
  switch (getValue()) {
  case AccPhase::Unspecified:
    return "Unspecified";
  case AccPhase::Partial:
    return "Partial";
  case AccPhase::Final:
    return "Final";
  }
  llvm_unreachable("unknown PTO AccPhase case");
}

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

Attribute TileBufConfigAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto def = TileBufConfigAttr::getDefault(ctx);
  BLayoutAttr bl = def.getBLayout();
  SLayoutAttr sl = def.getSLayout();
  IntegerAttr sz = def.getSFractalSize();
  PadValueAttr pv = def.getPad();
  CompactModeAttr compact = def.getCompactMode();

  if (p.parseLess()) {
    return {};
  }

  if (succeeded(p.parseOptionalGreater())) {
    return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
  }

  bool parsedGreater = false;
  while (!parsedGreater) {
    StringRef key;
    if (p.parseKeyword(&key)) {
      return {};
    }
    if (p.parseEqual()) {
      return {};
    }

    if (key == "blayout") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      bl = toBLayoutAttr(ctx, a);
      if (!bl) {
        return {};
      }
    } else if (key == "slayout") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      sl = toSLayoutAttr(ctx, a);
      if (!sl) {
        return {};
      }
    } else if (key == "s_fractal_size") {
      int32_t v = 0;
      if (p.parseInteger(v)) {
        return {};
      }
      sz = IntegerAttr::get(IntegerType::get(ctx, kI32BitWidth), v);
    } else if (key == "pad") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      pv = toPadValueAttr(ctx, a);
      if (!pv) {
        return {};
      }
    } else if (key == "compact") {
      Attribute a;
      if (p.parseAttribute(a)) {
        return {};
      }
      compact = toCompactModeAttr(ctx, a);
      if (!compact) {
        return {};
      }
    } else {
      p.emitError(p.getCurrentLocation(), "unknown key in tile_buf_config: ") << key;
      return {};
    }

    parsedGreater = succeeded(p.parseOptionalGreater());
    if (parsedGreater) {
      break;
    }
    if (p.parseComma()) {
      return {};
    }
  }

  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

void TileBufConfigAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "blayout=" << getBLayout();
  p << ", slayout=" << getSLayout();
  p << ", s_fractal_size=" << static_cast<int32_t>(getSFractalSize().getInt());
  p << ", pad=" << getPad();
  p << ", compact=" << getCompactMode();
  p << ">";
}
