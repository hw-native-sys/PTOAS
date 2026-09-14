// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteGmUbAsm.cpp - pto.mte_gm_ub asm helpers ---------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

ParseResult parseMteGmUbBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    OpAsmParser::UnresolvedOperand &l2CacheCtl,
    OpAsmParser::UnresolvedOperand &lenBurst,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, l2CacheCtl) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteGmUbBasicTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &l2CacheCtlType, Type &lenBurstType,
    SmallVectorImpl<Type> &nburstTypes) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(l2CacheCtlType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  return success();
}

void setMteGmUbSegmentSizes(OperationState &result,
                                    int32_t loopGroupCount,
                                    size_t padOperandCount) {
  auto &segments =
      result.getOrAddProperties<MteGmUbOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1, 1,
                               loopGroupCount, loopGroupCount, loopGroupCount,
                               static_cast<int32_t>(padOperandCount != 0 ? 1 : 0),
                               static_cast<int32_t>(padOperandCount == 3 ? 1 : 0),
                               static_cast<int32_t>(padOperandCount == 3 ? 1 : 0)},
              segments.begin());
}

ParseResult resolveMteGmUbOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand l2CacheCtl, Type l2CacheCtlType,
    OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands,
    SmallVectorImpl<Type> &nburstTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands,
    SmallVectorImpl<Type> &loopCountTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands,
    SmallVectorImpl<Type> &loopSrcStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands,
    SmallVectorImpl<Type> &loopDstStrideTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &padOperands,
    SmallVectorImpl<Type> &padTypes) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(l2CacheCtl, l2CacheCtlType, result.operands) ||
      parser.resolveOperand(lenBurst, lenBurstType, result.operands) ||
      parser.resolveOperands(nburstOperands, nburstTypes, loc,
                             result.operands) ||
      failed(resolveDmaLoopOperands(parser, result, loopCountOperands,
                                    loopCountTypes, loopSrcStrideOperands,
                                    loopSrcStrideTypes, loopDstStrideOperands,
                                    loopDstStrideTypes)) ||
      parser.resolveOperands(padOperands, padTypes, loc, result.operands)) {
    return failure();
  }
  return success();
}
