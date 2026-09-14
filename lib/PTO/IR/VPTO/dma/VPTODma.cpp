// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTODma.cpp - shared VPTO DMA asm/verify helpers -------------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// DMA triple-group/loop/pad parse-print infrastructure shared by the DMA op
// parsers (VPTOMte.cpp / VPTOMteAsm.cpp) and the loop-group verifier.
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

// Shared kValue3 operand loop for the DMA triple-group parsers.
static ParseResult parseDmaTripleOperandLoop(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  for (int i = 0; i < mlir::pto::kValue3; ++i) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
    if (i != mlir::pto::kValue2 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

ParseResult parseDmaTripleGroup(
    OpAsmParser &parser, StringRef keyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return failure();
  }
  if (failed(parseDmaTripleOperandLoop(parser, operands))) {
    return failure();
  }
  return parser.parseRParen();
}

ParseResult parseOptionalDmaTripleGroupAlias(
    OpAsmParser &parser, ArrayRef<StringRef> keywords,
    StringRef &parsedKeyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  parsedKeyword = {};
  for (StringRef keyword : keywords) {
    if (failed(parser.parseOptionalKeyword(keyword))) {
      continue;
    }
    parsedKeyword = keyword;
    if (parser.parseLParen()) {
      return failure();
    }
    if (failed(parseDmaTripleOperandLoop(parser, operands))) {
      return failure();
    }
    return parser.parseRParen();
  }
  return success();
}

static bool isDmaLoopKeyword(StringRef keyword) {
  if (keyword == "loop") {
    return true;
  }
  if (!keyword.consume_front("loop")) {
    return false;
  }
  if (keyword.empty()) {
    return false;
  }
  return llvm::all_of(keyword, llvm::isDigit);
}

ParseResult parseDmaTripleTypes(OpAsmParser &parser,
                                       SmallVectorImpl<Type> &types) {
  for (int i = 0; i < mlir::pto::kValue3; ++i) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    if (i != mlir::pto::kValue2 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

static ParseResult parseDmaPadTypes(OpAsmParser &parser,
                                    SmallVectorImpl<Type> &types) {
  Type valueType;
  if (parser.parseType(valueType)) {
    return failure();
  }
  types.push_back(valueType);
  if (succeeded(parser.parseOptionalComma())) {
    Type leftType;
    Type rightType;
    if (parser.parseType(leftType) || parser.parseComma() ||
        parser.parseType(rightType)) {
      return failure();
    }
    types.push_back(leftType);
    types.push_back(rightType);
  }
  return success();
}

ParseResult parseDmaPadOperandGroup(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &padOperands) {
  if (failed(parser.parseOptionalKeyword("pad"))) {
    return success();
  }
  if (parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand value;
  if (parser.parseOperand(value)) {
    return failure();
  }
  padOperands.push_back(value);
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand left;
    OpAsmParser::UnresolvedOperand right;
    if (parser.parseOperand(left) || parser.parseComma() ||
        parser.parseOperand(right)) {
      return failure();
    }
    padOperands.push_back(left);
    padOperands.push_back(right);
  }
  if (parser.parseRParen()) {
    return failure();
  }
  return success();
}

ParseResult parseDmaLoopOperandGroups(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands) {
  bool hasMore = true;
  while (hasMore) {
    StringRef parsedKeyword;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loopGroupOperands;
    if (parseOptionalDmaTripleGroupAlias(parser, {"loop", "loop1", "loop2"},
                                         parsedKeyword, loopGroupOperands)) {
      return failure();
    }
    if (parsedKeyword.empty()) {
      hasMore = false;
      continue;
    }
    loopCountOperands.push_back(loopGroupOperands[0]);
    loopSrcStrideOperands.push_back(loopGroupOperands[1]);
    loopDstStrideOperands.push_back(loopGroupOperands[mlir::pto::kValue2]);
  }
  return success();
}

ParseResult parseDmaLoopTypeGroups(
    OpAsmParser &parser, SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<Type> &loopDstStrideTypes) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (!isDmaLoopKeyword(keyword)) {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop'");
    }
    SmallVector<Type> loopGroupTypes;
    if (parseDmaTripleTypes(parser, loopGroupTypes)) {
      return failure();
    }
    loopCountTypes.push_back(loopGroupTypes[0]);
    loopSrcStrideTypes.push_back(loopGroupTypes[1]);
    loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
  }
  return success();
}

ParseResult parseDmaLoopAndPadTypeGroups(
    OpAsmParser &parser, SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<Type> &loopDstStrideTypes,
    SmallVectorImpl<Type> &padTypes) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (isDmaLoopKeyword(keyword)) {
      SmallVector<Type> loopGroupTypes;
      if (parseDmaTripleTypes(parser, loopGroupTypes)) {
        return failure();
      }
      loopCountTypes.push_back(loopGroupTypes[0]);
      loopSrcStrideTypes.push_back(loopGroupTypes[1]);
      loopDstStrideTypes.push_back(loopGroupTypes[mlir::pto::kValue2]);
      continue;
    }
    if (keyword == "pad") {
      if (!padTypes.empty() || parseDmaPadTypes(parser, padTypes)) {
        return failure();
      }
      continue;
    }
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'loop' or 'pad'");
  }
  return success();
}

ParseResult verifyDmaLoopGroupConsistency(
    OpAsmParser &parser, size_t countOperands, size_t srcStrideOperands,
    size_t dstStrideOperands, size_t countTypes, size_t srcStrideTypes,
    size_t dstStrideTypes) {
  if (countOperands != srcStrideOperands || countOperands != dstStrideOperands ||
      countTypes != srcStrideTypes || countTypes != dstStrideTypes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires each loop group to provide count, src stride, and dst stride");
  }
  if (countOperands != countTypes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "requires loop operand and type groups to match");
  }
  return success();
}

ParseResult resolveDmaBasicOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands,
    SmallVectorImpl<Type> &nburstTypes) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}

ParseResult resolveDmaLoopOperands(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands,
    SmallVectorImpl<Type> &loopDstStrideTypes) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(loopCountOperands, loopCountTypes, loc,
                             result.operands) ||
      parser.resolveOperands(loopSrcStrideOperands, loopSrcStrideTypes, loc,
                             result.operands) ||
      parser.resolveOperands(loopDstStrideOperands, loopDstStrideTypes, loc,
                             result.operands)) {
    return failure();
  }
  return success();
}

// Unified resolver for the DMA triple-group ops; covers the plain loop ops
// (hasL2CacheCtl=false) and the MteUbGm ops that interleave l2_cache_ctl
// between the basic operands and the loop groups (resolve order is kept).
ParseResult resolveDmaTripleOperands(
    OpAsmParser &parser, OperationState &result, bool hasL2CacheCtl,
    OpAsmParser::UnresolvedOperand l2CacheCtl, Type l2CacheCtlType,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands,
    SmallVectorImpl<Type> &nburstTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands,
    SmallVectorImpl<Type> &loopDstStrideTypes) {
  if (failed(resolveDmaBasicOperands(parser, result, source, sourceType,
                                     destination, destinationType, lenBurst,
                                     lenBurstType, nburstOperands,
                                     nburstTypes))) {
    return failure();
  }
  if (hasL2CacheCtl &&
      parser.resolveOperand(l2CacheCtl, l2CacheCtlType, result.operands)) {
    return failure();
  }
  return resolveDmaLoopOperands(parser, result, loopCountOperands,
                                loopCountTypes, loopSrcStrideOperands,
                                loopSrcStrideTypes, loopDstStrideOperands,
                                loopDstStrideTypes);
}

void printDmaTripleGroup(OpAsmPrinter &printer, StringRef keyword,
                                Value first, Value second, Value third) {
  printer << " " << keyword << "(" << first << ", " << second << ", " << third
          << ")";
}

void printDmaTripleTypes(OpAsmPrinter &printer, StringRef keyword,
                                Type first, Type second, Type third) {
  printer << ", " << keyword << " " << first << ", " << second << ", " << third;
}

void printDmaPadGroup(OpAsmPrinter &printer, Value value, Value left,
                             Value right) {
  printer << " pad(" << value;
  if (left || right) {
    printer << ", " << left << ", " << right;
  }
  printer << ")";
}

void printDmaPadTypes(OpAsmPrinter &printer, Type valueType,
                             Type leftType, Type rightType) {
  printer << ", pad " << valueType;
  if (leftType || rightType) {
    printer << ", " << leftType << ", " << rightType;
  }
}

ParseResult parseFixedKeywordOperandGroup(
    OpAsmParser &parser, StringRef keyword, int operandCount,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return failure();
  }
  for (int i = 0; i < operandCount; ++i) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
    if (i + 1 != operandCount && parser.parseComma()) {
      return failure();
    }
  }
  return parser.parseRParen();
}

ParseResult parseFixedKeywordTypes(OpAsmParser &parser, StringRef keyword,
                                          int typeCount,
                                          SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword(keyword)) {
    return failure();
  }
  for (int i = 0; i < typeCount; ++i) {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
    if (i + 1 != typeCount && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

template <typename DmaOp>
[[maybe_unused]] static LogicalResult verifyOptionalDmaLoopGroup(DmaOp op, Value count,
                                                Value srcStride,
                                                Value dstStride,
                                                StringRef name) {
  bool hasAny = static_cast<bool>(count) || static_cast<bool>(srcStride) ||
                static_cast<bool>(dstStride);
  bool hasAll = static_cast<bool>(count) && static_cast<bool>(srcStride) &&
                static_cast<bool>(dstStride);
  if (hasAny && !hasAll) {
    return op.emitOpError() << "requires " << name
                            << " group to provide count, src stride, and dst stride together";
  }
  return success();
}

LogicalResult verifyDmaLoadStoreLoopGroups(Operation *op,
                                                  ValueRange loopCounts,
                                                  ValueRange loopSrcStrides,
                                                  ValueRange loopDstStrides) {
  if (loopCounts.size() != loopSrcStrides.size() ||
      loopCounts.size() != loopDstStrides.size()) {
    return op->emitOpError()
           << "requires each loop group to provide count, src stride, and dst stride together";
  }
  return success();
}
