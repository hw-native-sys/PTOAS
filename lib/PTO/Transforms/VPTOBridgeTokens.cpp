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
// include/PTO/Transforms/VPTOBridgeTokens.h. The construction rules mirror
// EmitC's token builders (PTOToEmitC.cpp) but are implemented independently:
// the two backends share no code, only the documented mapping rules.
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

/// Maps a local-to-local dir_mask to a fully qualified pto::Direction token.
/// The rules follow EmitC's getTPipeDirectionToken for the non-L2G2L case:
/// the A5 "_GM" variants only apply to L2G2L pipes, so they never appear
/// here.
FailureOr<std::string> bridgeDirectionToken(int8_t dirMask) {
  switch (dirMask) {
  case 1:
    return std::string("pto::Direction::DIR_C2V");
  case 2:
    return std::string("pto::Direction::DIR_V2C");
  case 3:
    return std::string("pto::Direction::DIR_BOTH");
  default:
    return failure();
  }
}

/// Maps a tile buffer address space to a fully qualified pto::TileType token.
/// Global-memory and default address spaces are not local tiles and have no
/// TileType mapping.
FailureOr<std::string> bridgeTileTypeToken(AddressSpace addressSpace) {
  switch (addressSpace) {
  case AddressSpace::MAT:
    return std::string("pto::TileType::Mat");
  case AddressSpace::LEFT:
    return std::string("pto::TileType::Left");
  case AddressSpace::RIGHT:
    return std::string("pto::TileType::Right");
  case AddressSpace::ACC:
    return std::string("pto::TileType::Acc");
  case AddressSpace::VEC:
    return std::string("pto::TileType::Vec");
  case AddressSpace::BIAS:
    return std::string("pto::TileType::Bias");
  case AddressSpace::SCALING:
    return std::string("pto::TileType::Scaling");
  default:
    return failure();
  }
}

/// Maps a BLayout value (TileBufType::getBLayoutValueI32) to a token.
FailureOr<std::string> bridgeBLayoutToken(int32_t bLayout) {
  switch (bLayout) {
  case 0:
    return std::string("pto::BLayout::RowMajor");
  case 1:
    return std::string("pto::BLayout::ColMajor");
  default:
    return failure();
  }
}

/// Maps an SLayout value (TileBufType::getSLayoutValueI32) to a token.
FailureOr<std::string> bridgeSLayoutToken(int32_t sLayout) {
  switch (sLayout) {
  case 0:
    return std::string("pto::SLayout::NoneBox");
  case 1:
    return std::string("pto::SLayout::RowMajor");
  case 2:
    return std::string("pto::SLayout::ColMajor");
  default:
    return failure();
  }
}

} // namespace

std::string pto::buildBridgeElementTypeToken(Type elementType) {
  // Narrow float types mirror EmitC's getEmitCScalarTypeToken so the bridge
  // wrapper and the EmitC backend name the same pto-isa scalar types.
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
  auto dirTok = bridgeDirectionToken(init.getDirMask());
  if (failed(dirTok))
    return failure();

  // The local-to-local pipe always uses a localSlotNum of 2 (see EmitC's
  // buildTPipeTokenFromInitOp for the InitializeL2LPipeOp case).
  constexpr int32_t localSlotNum = 2;
  bool nosplit = init.getNosplitAttr() && init.getNosplitAttr().getValue();

  std::string token = "pto::TPipe<" + std::to_string(flagBaseAttr.getInt()) +
                      ", " + *dirTok + ", " +
                      std::to_string(init.getSlotSize()) + ", " +
                      std::to_string(init.getSlotNum()) + ", " +
                      std::to_string(localSlotNum) + ", " +
                      (nosplit ? "true" : "false") + ">";
  return token;
}

FailureOr<std::string> pto::buildBridgeTileSplitToken(int64_t split) {
  switch (split) {
  case 0:
    return std::string("pto::TileSplitAxis::TILE_NO_SPLIT");
  case 1:
    return std::string("pto::TileSplitAxis::TILE_UP_DOWN");
  case 2:
    return std::string("pto::TileSplitAxis::TILE_LEFT_RIGHT");
  case 3:
    return std::string("pto::TileSplitAxis::TILE_UP_DOWN_ODD");
  case 4:
    return std::string("pto::TileSplitAxis::TILE_LEFT_RIGHT_ODD");
  default:
    return failure();
  }
}

FailureOr<std::string> pto::buildBridgeTileToken(TileBufType tile) {
  auto addressSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
  if (!addressSpaceAttr)
    return failure();
  auto tileTypeTok = bridgeTileTypeToken(addressSpaceAttr.getAddressSpace());
  if (failed(tileTypeTok))
    return failure();

  ArrayRef<int64_t> shape = tile.getShape();
  ArrayRef<int64_t> validShape = tile.getValidShape();
  if (shape.size() != 2 || validShape.size() != 2)
    return failure();

  auto bLayoutTok = bridgeBLayoutToken(tile.getBLayoutValueI32());
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
    auto sLayoutTok = bridgeSLayoutToken(sLayoutValue);
    if (failed(sLayoutTok))
      return failure();
    token += ", " + *sLayoutTok + ", " +
             std::to_string(tile.getSFractalSizeI32());
  }
  token += ">";
  return token;
}
