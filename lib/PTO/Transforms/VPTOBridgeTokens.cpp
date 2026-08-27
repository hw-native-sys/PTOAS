// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.huawei.com/
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeTokens.cpp - C++ template token building ----------------===//
//===----------------------------------------------------------------------===//
//
// Implementation of the bridge-side PTO-ISA C++ template token builders. See
// include/PTO/Transforms/VPTOBridgeTokens.h. Both the IR-fact -> C++ spelling
// mapping rules and the bridge assembly rules (fully qualified spellings,
// NoneBox trailing-argument omission) live here.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "llvm/ADT/Twine.h"
#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace {

/// The bridge wrapper is a standalone translation unit, so every pto-isa
/// spelling is emitted fully qualified.
constexpr llvm::StringLiteral kBridgeQualifier = "pto::";

/// Builds the `TileSplitAxis::TILE_*` token for a split value (0..4).
/// Fails for values outside that range.
FailureOr<std::string> buildTileSplitToken(int64_t split) {
  switch (split) {
  case 0:
    return (kBridgeQualifier + "TileSplitAxis::TILE_NO_SPLIT").str();
  case 1:
    return (kBridgeQualifier + "TileSplitAxis::TILE_UP_DOWN").str();
  case 2:
    return (kBridgeQualifier + "TileSplitAxis::TILE_LEFT_RIGHT").str();
  case 3:
    return (kBridgeQualifier + "TileSplitAxis::TILE_UP_DOWN_ODD").str();
  case 4:
    return (kBridgeQualifier + "TileSplitAxis::TILE_LEFT_RIGHT_ODD").str();
  default:
    return failure();
  }
}

/// Builds the `Direction::DIR_*` token for a local pipe dir_mask (1=C2V,
/// 2=V2C, 3=BOTH). Fails for other masks.
FailureOr<std::string> buildDirectionToken(int8_t dirMask) {
  switch (dirMask) {
  case 1:
    return (kBridgeQualifier + "Direction::DIR_C2V").str();
  case 2:
    return (kBridgeQualifier + "Direction::DIR_V2C").str();
  case 3:
    return (kBridgeQualifier + "Direction::DIR_BOTH").str();
  default:
    return failure();
  }
}

/// Builds the `TileType::*` token for a local tile address space. Fails
/// for address spaces with no TileType mapping (e.g. global memory);
/// callers apply their own fallback policy for those.
FailureOr<std::string> buildTileTypeToken(AddressSpace addressSpace) {
  switch (addressSpace) {
  case AddressSpace::MAT:
    return (kBridgeQualifier + "TileType::Mat").str();
  case AddressSpace::LEFT:
    return (kBridgeQualifier + "TileType::Left").str();
  case AddressSpace::RIGHT:
    return (kBridgeQualifier + "TileType::Right").str();
  case AddressSpace::ACC:
    return (kBridgeQualifier + "TileType::Acc").str();
  case AddressSpace::VEC:
    return (kBridgeQualifier + "TileType::Vec").str();
  case AddressSpace::BIAS:
    return (kBridgeQualifier + "TileType::Bias").str();
  case AddressSpace::SCALING:
    return (kBridgeQualifier + "TileType::Scaling").str();
  default:
    return failure();
  }
}

/// Builds the `BLayout::*` token. Fails for values outside the closed set.
FailureOr<std::string> buildBLayoutToken(BLayout bLayout) {
  switch (bLayout) {
  case BLayout::RowMajor:
    return (kBridgeQualifier + "BLayout::RowMajor").str();
  case BLayout::ColMajor:
    return (kBridgeQualifier + "BLayout::ColMajor").str();
  }
  return failure();
}

/// Builds the `SLayout::*` token. Fails for values outside the closed set.
FailureOr<std::string> buildSLayoutToken(SLayout sLayout) {
  switch (sLayout) {
  case SLayout::NoneBox:
    return (kBridgeQualifier + "SLayout::NoneBox").str();
  case SLayout::RowMajor:
    return (kBridgeQualifier + "SLayout::RowMajor").str();
  case SLayout::ColMajor:
    return (kBridgeQualifier + "SLayout::ColMajor").str();
  }
  return failure();
}

/// Renders the `TPipe<flagBase, Direction, slotSize, slotNum, localSlotNum,
/// nosplit>` spelling; `dirTok` is an already rendered direction token.
std::string renderTPipeSpelling(int32_t flagBase, llvm::StringRef dirTok,
                                int32_t slotSize, int32_t slotNum,
                                int32_t localSlotNum, bool nosplit) {
  return (kBridgeQualifier + "TPipe<").str() + std::to_string(flagBase) +
         ", " + dirTok.str() + ", " + std::to_string(slotSize) + ", " +
         std::to_string(slotNum) + ", " + std::to_string(localSlotNum) + ", " +
         (nosplit ? "true" : "false") + ">";
}

} // namespace

std::string pto::buildBridgeElementTypeToken(Type elementType) {
  if (pto::isPTOFloat8E4M3LikeType(elementType))
    return "float8_e4m3_t";
  if (pto::isPTOFloat8E5M2LikeType(elementType))
    return "float8_e5m2_t";
  if (pto::isPTOF8E8M0Type(elementType))
    return "float8_e8m0_t";
  if (isa<pto::HiF8Type>(elementType))
    return "hifloat8_t";
  if (isa<pto::F4E1M2x2Type>(elementType))
    return "float4_e1m2x2_t";
  if (isa<pto::F4E2M1x2Type>(elementType))
    return "float4_e2m1x2_t";
  if (elementType.isF16())
    return "half";
  if (elementType.isBF16())
    return "bfloat16_t";
  if (elementType.isF32())
    return "float";
  if (elementType.isF64())
    return "double";
  if (elementType.isInteger(8))
    return (elementType.isSignlessInteger(8) ||
            elementType.isSignedInteger(8))
               ? "int8_t"
               : "uint8_t";
  if (elementType.isInteger(16))
    return (elementType.isSignlessInteger(16) ||
            elementType.isSignedInteger(16))
               ? "int16_t"
               : "uint16_t";
  if (elementType.isInteger(32))
    return (elementType.isSignlessInteger(32) ||
            elementType.isSignedInteger(32))
               ? "int32_t"
               : "uint32_t";
  if (elementType.isInteger(64))
    return cast<IntegerType>(elementType).isUnsigned() ? "uint64_t"
                                                       : "int64_t";
  return "float";
}

FailureOr<std::string> pto::buildBridgePipeToken(InitializeL2LPipeOp init) {
  IntegerAttr flagBaseAttr = init.getFlagBaseAttr();
  if (!flagBaseAttr)
    return failure();
  auto dirTok = buildDirectionToken(init.getDirMask());
  if (failed(dirTok))
    return failure();

  // The local-to-local pipe always uses a localSlotNum of 2 (see EmitC's
  // buildTPipeTokenFromInitOp for the InitializeL2LPipeOp case).
  constexpr int32_t localSlotNum = 2;
  bool nosplit = init.getNosplitAttr() && init.getNosplitAttr().getValue();

  return renderTPipeSpelling(
      static_cast<int32_t>(flagBaseAttr.getInt()), *dirTok,
      init.getSlotSize(), init.getSlotNum(), localSlotNum, nosplit);
}

FailureOr<std::string> pto::buildBridgeTileSplitToken(int64_t split) {
  return buildTileSplitToken(split);
}

FailureOr<std::string> pto::buildBridgeTileToken(TileBufType tile) {
  auto addressSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
  if (!addressSpaceAttr)
    return failure();
  auto tileTypeTok = buildTileTypeToken(addressSpaceAttr.getAddressSpace());
  if (failed(tileTypeTok))
    return failure();

  ArrayRef<int64_t> shape = tile.getShape();
  ArrayRef<int64_t> validShape = tile.getValidShape();
  if (shape.size() != 2 || validShape.size() != 2)
    return failure();

  auto bLayoutTok =
      buildBLayoutToken(static_cast<BLayout>(tile.getBLayoutValueI32()));
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
    auto sLayoutTok = buildSLayoutToken(static_cast<SLayout>(sLayoutValue));
    if (failed(sLayoutTok))
      return failure();
    token += ", " + *sLayoutTok + ", " +
             std::to_string(tile.getSFractalSizeI32());
  }
  token += ">";
  return token;
}
