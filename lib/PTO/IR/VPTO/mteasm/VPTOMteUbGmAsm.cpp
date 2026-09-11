// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteUbGmAsm.cpp - pto.mte_ub_gm asm helpers ---------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

ParseResult parseMteUbGmBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    OpAsmParser::UnresolvedOperand &lenBurst,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parser.parseOperand(lenBurst) ||
      parseDmaTripleGroup(parser, "nburst", nburstOperands)) {
    return failure();
  }
  return success();
}

ParseResult parseMteUbGmBasicTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &lenBurstType, SmallVectorImpl<Type> &nburstTypes) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(lenBurstType) || parser.parseComma() ||
      parseDmaTripleTypes(parser, nburstTypes)) {
    return failure();
  }
  return success();
}

ParseResult parseMteUbGmL2CacheCtlOperand(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &l2CacheCtl,
    bool &hasL2CacheCtl) {
  hasL2CacheCtl = succeeded(parser.parseOptionalKeyword("l2_cache_ctl"));
  if (!hasL2CacheCtl) {
    return success();
  }
  if (parser.parseLParen() || parser.parseOperand(l2CacheCtl) ||
      parser.parseRParen()) {
    return failure();
  }
  return success();
}

void setMteUbGmSegmentSizes(OperationState &result, bool hasL2CacheCtl,
                                    size_t loopGroupCount) {
  auto &segments =
      result.getOrAddProperties<MteUbGmOp::Properties>().operandSegmentSizes;
  llvm::copy(ArrayRef<int32_t>{1, 1, 1, 1, 1, 1,
                               hasL2CacheCtl ? 1 : 0,
                               static_cast<int32_t>(loopGroupCount),
                               static_cast<int32_t>(loopGroupCount),
                               static_cast<int32_t>(loopGroupCount)},
              segments.begin());
}
