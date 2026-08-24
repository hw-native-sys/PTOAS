// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.huawei.com/
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOCppTokens.cpp - shared PTO-ISA C++ token mappings --------------===//
//===----------------------------------------------------------------------===//
//
// Implementation of the shared IR-fact -> C++ spelling mapping functions;
// see include/PTO/Transforms/PTOCppTokens.h.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/PTOCppTokens.h"
#include "PTO/IR/PTOTypeUtils.h"

using namespace mlir;
using namespace mlir::pto;

std::string pto::getPTOCppElementTypeToken(Type elementType) {
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

FailureOr<std::string> pto::getPTOCppTileSplitToken(int64_t split,
                                                    StringRef qualifier) {
  switch (split) {
  case 0:
    return (qualifier + "TileSplitAxis::TILE_NO_SPLIT").str();
  case 1:
    return (qualifier + "TileSplitAxis::TILE_UP_DOWN").str();
  case 2:
    return (qualifier + "TileSplitAxis::TILE_LEFT_RIGHT").str();
  case 3:
    return (qualifier + "TileSplitAxis::TILE_UP_DOWN_ODD").str();
  case 4:
    return (qualifier + "TileSplitAxis::TILE_LEFT_RIGHT_ODD").str();
  default:
    return failure();
  }
}

FailureOr<std::string> pto::getPTOCppDirectionToken(int8_t dirMask,
                                                    StringRef qualifier) {
  switch (dirMask) {
  case 1:
    return (qualifier + "Direction::DIR_C2V").str();
  case 2:
    return (qualifier + "Direction::DIR_V2C").str();
  case 3:
    return (qualifier + "Direction::DIR_BOTH").str();
  default:
    return failure();
  }
}

FailureOr<std::string> pto::getPTOCppTileTypeToken(AddressSpace addressSpace,
                                                   StringRef qualifier) {
  switch (addressSpace) {
  case AddressSpace::MAT:
    return (qualifier + "TileType::Mat").str();
  case AddressSpace::LEFT:
    return (qualifier + "TileType::Left").str();
  case AddressSpace::RIGHT:
    return (qualifier + "TileType::Right").str();
  case AddressSpace::ACC:
    return (qualifier + "TileType::Acc").str();
  case AddressSpace::VEC:
    return (qualifier + "TileType::Vec").str();
  case AddressSpace::BIAS:
    return (qualifier + "TileType::Bias").str();
  case AddressSpace::SCALING:
    return (qualifier + "TileType::Scaling").str();
  default:
    return failure();
  }
}

FailureOr<std::string> pto::getPTOCppBLayoutToken(BLayout bLayout,
                                                  StringRef qualifier) {
  switch (bLayout) {
  case BLayout::RowMajor:
    return (qualifier + "BLayout::RowMajor").str();
  case BLayout::ColMajor:
    return (qualifier + "BLayout::ColMajor").str();
  }
  return failure();
}

FailureOr<std::string> pto::getPTOCppSLayoutToken(SLayout sLayout,
                                                  StringRef qualifier) {
  switch (sLayout) {
  case SLayout::NoneBox:
    return (qualifier + "SLayout::NoneBox").str();
  case SLayout::RowMajor:
    return (qualifier + "SLayout::RowMajor").str();
  case SLayout::ColMajor:
    return (qualifier + "SLayout::ColMajor").str();
  }
  return failure();
}

std::string pto::renderTPipeSpelling(int32_t flagBase, StringRef dirTok,
                                     int32_t slotSize, int32_t slotNum,
                                     int32_t localSlotNum, bool nosplit,
                                     StringRef qualifier) {
  std::string token = (qualifier + "TPipe<").str() +
                      std::to_string(flagBase) + ", " + dirTok.str() + ", " +
                      std::to_string(slotSize) + ", " +
                      std::to_string(slotNum) + ", " +
                      std::to_string(localSlotNum) + ", " +
                      (nosplit ? "true" : "false") + ">";
  return token;
}
