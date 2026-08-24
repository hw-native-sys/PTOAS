// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeTokens.cpp - C++ template token building ----------------===//
//===----------------------------------------------------------------------===//
//
// Implementation of the bridge-side PTO-ISA C++ template token builders. See
// include/PTO/Transforms/VPTOBridgeTokens.h. The IR-fact -> C++ spelling
// mapping rules are shared with the EmitC backend through PTOCppTokens;
// this file holds the bridge assembly rules (fully qualified spellings and
// the NoneBox trailing-argument omission).
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/PTOCppTokens.h"
#include "llvm/ADT/Twine.h"
#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace {

/// The bridge wrapper is a standalone translation unit, so every pto-isa
/// spelling is emitted fully qualified.
constexpr llvm::StringLiteral kBridgeQualifier = "pto::";

} // namespace

std::string pto::buildBridgeElementTypeToken(Type elementType) {
  return getPTOCppElementTypeToken(elementType);
}

FailureOr<std::string> pto::buildBridgePipeToken(InitializeL2LPipeOp init) {
  IntegerAttr flagBaseAttr = init.getFlagBaseAttr();
  if (!flagBaseAttr)
    return failure();
  auto dirTok = getPTOCppDirectionToken(init.getDirMask(), kBridgeQualifier);
  if (failed(dirTok))
    return failure();

  // The local-to-local pipe always uses a localSlotNum of 2 (see EmitC's
  // buildTPipeTokenFromInitOp for the InitializeL2LPipeOp case).
  constexpr int32_t localSlotNum = 2;
  bool nosplit = init.getNosplitAttr() && init.getNosplitAttr().getValue();

  return renderTPipeSpelling(
      static_cast<int32_t>(flagBaseAttr.getInt()), *dirTok,
      init.getSlotSize(), init.getSlotNum(), localSlotNum, nosplit,
      kBridgeQualifier);
}

FailureOr<std::string> pto::buildBridgeTileSplitToken(int64_t split) {
  return getPTOCppTileSplitToken(split, kBridgeQualifier);
}

FailureOr<std::string> pto::buildBridgeTileToken(TileBufType tile) {
  auto addressSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
  if (!addressSpaceAttr)
    return failure();
  auto tileTypeTok = getPTOCppTileTypeToken(addressSpaceAttr.getAddressSpace(),
                                            kBridgeQualifier);
  if (failed(tileTypeTok))
    return failure();

  ArrayRef<int64_t> shape = tile.getShape();
  ArrayRef<int64_t> validShape = tile.getValidShape();
  if (shape.size() != 2 || validShape.size() != 2)
    return failure();

  auto bLayoutTok = getPTOCppBLayoutToken(
      static_cast<BLayout>(tile.getBLayoutValueI32()), kBridgeQualifier);
  if (failed(bLayoutTok))
    return failure();

  std::string token =
      "pto::Tile<" + *tileTypeTok + ", " +
      buildBridgeElementTypeToken(tile.getElementType()) + ", " +
      std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " +
      *bLayoutTok + ", " + std::to_string(validShape[0]) + ", " +
      std::to_string(validShape[1]);

  // Boxed storage layouts carry the inner-fractal template arguments; the
  // default NoneBox layout relies on the Tile template defaults, matching the
  // hand-written wrapper specializations.
  int32_t sLayoutValue = tile.getSLayoutValueI32();
  if (sLayoutValue != 0) {
    auto sLayoutTok = getPTOCppSLayoutToken(static_cast<SLayout>(sLayoutValue),
                                            kBridgeQualifier);
    if (failed(sLayoutTok))
      return failure();
    token += ", " + *sLayoutTok + ", " +
             std::to_string(tile.getSFractalSizeI32());
  }
  token += ">";
  return token;
}
