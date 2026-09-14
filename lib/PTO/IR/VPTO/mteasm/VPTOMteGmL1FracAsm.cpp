// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteGmL1FracAsm.cpp - pto.mte_gm_l1_frac asm helpers ------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

ParseResult parseMteGmL1FracBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination, StringRef &modeKeyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapeOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcLayoutOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstGroupOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrlOperands) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseKeyword(&modeKeyword) ||
      failed(parseCubeLoadFracModeKeyword(modeKeyword)) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "shape", mlir::pto::kValue2,
                                    shapeOperands) ||
      parser.parseComma() ||
      parseCubeLoadFracSrcLayoutGroup(parser, srcLayoutOperands) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "dst_group", mlir::pto::kValue4,
                                    dstGroupOperands) ||
      parser.parseComma() ||
      parseFixedKeywordOperandGroup(parser, "ctrl", mlir::pto::kValue2,
                                    ctrlOperands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteGmL1FracBasicTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    StringRef modeKeyword, SmallVectorImpl<Type> &shapeTypes,
    SmallVectorImpl<Type> &srcLayoutTypes,
    SmallVectorImpl<Type> &dstGroupTypes,
    SmallVectorImpl<Type> &ctrlTypes) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseKeyword(modeKeyword) || parser.parseComma() ||
      parseFixedKeywordTypes(parser, "shape", mlir::pto::kValue2,
                            shapeTypes) ||
      parser.parseComma() ||
      parseCubeLoadFracSrcLayoutTypes(parser, srcLayoutTypes) ||
      parser.parseComma() ||
      parseFixedKeywordTypes(parser, "dst_group", mlir::pto::kValue4,
                            dstGroupTypes) ||
      parser.parseComma() ||
      parseFixedKeywordTypes(parser, "ctrl", mlir::pto::kValue2, ctrlTypes)) {
    return failure();
  }
  return success();
}

ParseResult validateMteGmL1FracOperands(
    OpAsmParser &parser, size_t shapeOps, size_t shapeTypes,
    size_t srcLayoutOps, size_t srcLayoutTypes,
    size_t dstGroupOps, size_t dstGroupTypes,
    size_t ctrlOps, size_t ctrlTypes) {
  if (shapeOps != 2 || shapeTypes != 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "shape requires exactly two operands and types");
  }
  if (srcLayoutOps == 0 || srcLayoutOps > mlir::pto::kValue2 ||
      srcLayoutTypes == 0 || srcLayoutTypes > mlir::pto::kValue2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "src_layout requires one or two operands and types");
  }
  if (dstGroupOps != mlir::pto::kValue4 || dstGroupTypes != mlir::pto::kValue4) {
    return parser.emitError(parser.getCurrentLocation(),
                            "dst_group requires exactly four operands and types");
  }
  if (ctrlOps != mlir::pto::kValue2 || ctrlTypes != mlir::pto::kValue2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "ctrl requires exactly two operands and types");
  }
  if (srcLayoutOps != srcLayoutTypes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "src_layout operand and type groups must match");
  }
  return success();
}

ParseResult resolveMteGmL1FracOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapeOperands,
    SmallVectorImpl<Type> &shapeTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcLayoutOperands,
    SmallVectorImpl<Type> &srcLayoutTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstGroupOperands,
    SmallVectorImpl<Type> &dstGroupTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrlOperands,
    SmallVectorImpl<Type> &ctrlTypes) {
  bool hasSrcOuterStride = srcLayoutOperands.size() == 2;
  SmallVector<Type> flatTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> flatOperands;
  flatOperands.append({shapeOperands[0], shapeOperands[1], srcLayoutOperands[0]});
  flatTypes.append({shapeTypes[0], shapeTypes[1], srcLayoutTypes[0]});
  flatOperands.append(dstGroupOperands.begin(), dstGroupOperands.end());
  flatTypes.append(dstGroupTypes.begin(), dstGroupTypes.end());
  flatOperands.append(ctrlOperands.begin(), ctrlOperands.end());
  flatTypes.append(ctrlTypes.begin(), ctrlTypes.end());
  if (hasSrcOuterStride) {
    flatOperands.push_back(srcLayoutOperands[1]);
    flatTypes.push_back(srcLayoutTypes[1]);
  }
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperands(flatOperands, flatTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}
