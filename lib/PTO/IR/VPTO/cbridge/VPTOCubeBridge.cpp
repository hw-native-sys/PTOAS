// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOCubeBridge.cpp - shared cube bridge asm/verify helpers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<CubeLoadFracMode>
parseCubeLoadFracModeKeyword(StringRef keyword) {
  if (std::optional<CubeLoadFracMode> mode = symbolizeCubeLoadFracMode(keyword)) {
    return *mode;
  }
  return failure();
}

ParseResult parseCubeLoadFracSrcLayoutGroup(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword("src_layout") || parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand innerStride;
  if (parser.parseOperand(innerStride)) {
    return failure();
  }
  operands.push_back(innerStride);
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand outerStride;
    if (parser.parseOperand(outerStride)) {
      return failure();
    }
    operands.push_back(outerStride);
  }
  return parser.parseRParen();
}

ParseResult parseCubeLoadFracSrcLayoutTypes(OpAsmParser &parser,
                                                   SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword("src_layout") || parser.parseLParen()) {
    return failure();
  }
  Type innerStrideType;
  if (parser.parseType(innerStrideType)) {
    return failure();
  }
  types.push_back(innerStrideType);
  if (succeeded(parser.parseOptionalComma())) {
    Type outerStrideType;
    if (parser.parseType(outerStrideType)) {
      return failure();
    }
    types.push_back(outerStrideType);
  }
  return parser.parseRParen();
}

void printCubeLoadFracSrcLayoutGroup(OpAsmPrinter &printer,
                                            Value srcInnerStride,
                                            Value srcOuterStride) {
  printer << ", src_layout(" << srcInnerStride;
  if (srcOuterStride) {
    printer << ", " << srcOuterStride;
  }
  printer << ")";
}

void printCubeLoadFracSrcLayoutTypes(OpAsmPrinter &printer,
                                            Type srcInnerStrideType,
                                            Type srcOuterStrideType) {
  printer << ", src_layout(" << srcInnerStrideType;
  if (srcOuterStrideType) {
    printer << ", " << srcOuterStrideType;
  }
  printer << ")";
}

static std::optional<unsigned> getCubeBridgeLoadOperandIndex(
    StringRef keyword, ArrayRef<StringRef> shapeNames,
    ArrayRef<StringRef> fullNames) {
  for (auto [index, name] : llvm::enumerate(shapeNames)) {
    if (keyword == name) {
      return index;
    }
}

  for (auto [index, name] : llvm::enumerate(fullNames)) {
    if (keyword == name) {
      return shapeNames.size() + index;
}
}
  return std::nullopt;
}

static ParseResult parseCubeBridgeNamedOperand(
    OpAsmParser &parser, StringRef keyword, ArrayRef<StringRef> shapeNames,
    ArrayRef<StringRef> fullNames,
    SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands,
    SmallVectorImpl<unsigned> &namedOperandOrder) {
  std::optional<unsigned> index =
      getCubeBridgeLoadOperandIndex(keyword, shapeNames, fullNames);
  if (!index) {
    return parser.emitError(parser.getCurrentLocation(),
                            "unknown cube bridge load operand '")
           << keyword << "'";
  }
  if (namedOperands[*index].present) {
    return parser.emitError(parser.getCurrentLocation(),
                            "duplicate cube bridge load operand '")
           << keyword << "'";
  }
  if (parser.parseLParen() ||
      parser.parseOperand(namedOperands[*index].operand) ||
      parser.parseRParen()) {
    return failure();
  }
  namedOperands[*index].present = true;
  namedOperandOrder.push_back(*index);
  return success();
}

ParseResult parseCubeBridgeOptionalOperands(
    OpAsmParser &parser, ArrayRef<StringRef> shapeNames,
    ArrayRef<StringRef> fullNames,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &legacyOperands,
    SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands,
    SmallVectorImpl<unsigned> &namedOperandOrder, bool &usesNamedOperands) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword))) {
    usesNamedOperands = true;
    if (failed(parseCubeBridgeNamedOperand(parser, keyword, shapeNames,
                                           fullNames, namedOperands,
                                           namedOperandOrder))) {
      return failure();
    }
    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseKeyword(&keyword) ||
          failed(parseCubeBridgeNamedOperand(parser, keyword, shapeNames,
                                              fullNames, namedOperands,
                                              namedOperandOrder))) {
        return failure();
      }
    }
  } else {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    legacyOperands.push_back(operand);
    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(operand)) {
        return failure();
      }
      legacyOperands.push_back(operand);
    }
  }
  return success();
}

ParseResult parseCubeBridgeOptionalTypes(
    OpAsmParser &parser, bool usesNamedOperands,
    SmallVectorImpl<unsigned> &namedOperandOrder,
    SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &legacyOperands,
    SmallVectorImpl<Type> &legacyTypes) {
  if (usesNamedOperands) {
    for (unsigned index : namedOperandOrder) {
      Type type;
      if (parser.parseComma() || parser.parseType(type)) {
        return failure();
      }
      namedOperands[index].type = type;
    }
  } else {
    for (size_t index = 0; index < legacyOperands.size(); ++index) {
      Type type;
      if (parser.parseComma() || parser.parseType(type)) {
        return failure();
      }
      legacyTypes.push_back(type);
    }
  }
  return success();
}

ParseResult resolveCubeBridgeOperands(
    OpAsmParser &parser, OperationState &result, bool usesNamedOperands,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &legacyOperands,
    SmallVectorImpl<Type> &legacyTypes,
    SmallVectorImpl<int32_t> &segmentSizes) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands)) {
    return failure();
  }
  if (usesNamedOperands) {
    for (unsigned index = 0; index < namedOperands.size(); ++index) {
      if (!namedOperands[index].present) {
        continue;
      }
      segmentSizes[2 + index] = 1;
      if (parser.resolveOperand(namedOperands[index].operand,
                                 namedOperands[index].type,
                                 result.operands)) {
        return failure();
      }
    }
  } else {
    const unsigned base = legacyOperands.size() <= 4 ? 0 : 4;
    for (unsigned index = 0; index < legacyOperands.size(); ++index) {
      segmentSizes[2 + base + index] = 1;
      if (parser.resolveOperand(legacyOperands[index], legacyTypes[index],
                                 result.operands)) {
        return failure();
      }
    }
  }
  return success();
}
