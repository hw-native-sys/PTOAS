// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL0cUbAsm.cpp - pto.mte_l0c_ub asm helpers -------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

ParseResult parseMteL0cUbBasicOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    OpAsmParser::UnresolvedOperand &m, OpAsmParser::UnresolvedOperand &n,
    OpAsmParser::UnresolvedOperand &srcStride,
    OpAsmParser::UnresolvedOperand &dstStride) {
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride)) {
    return failure();
  }
  return success();
}

ParseResult parseMteL0cUbTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &mType,
    Type &nType, Type &srcStrideType, Type &dstStrideType, bool hasSubBlockId,
    Type &subBlockIdType, StructuredAccStoreAsmState &state) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() || parser.parseType(nType) ||
      parser.parseComma() || parser.parseType(srcStrideType) ||
      parser.parseComma() || parser.parseType(dstStrideType)) {
    return failure();
  }
  if (hasSubBlockId &&
      (parser.parseComma() || parser.parseType(subBlockIdType))) {
    return failure();
  }
  if (parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }
  return success();
}

ParseResult parseMteL0cUbDstMode(OpAsmParser &parser,
                                        AccStoreUbDstMode &dstMode,
                                        OpAsmParser::UnresolvedOperand &subBlockId,
                                        bool &hasSubBlockId) {
  if (parser.parseKeyword("dst_mode") || parser.parseLParen()) {
    return failure();
  }
  OptionalParseResult subBlockIdParse = parser.parseOptionalOperand(subBlockId);
  if (subBlockIdParse.has_value()) {
    if (failed(*subBlockIdParse)) {
      return failure();
    }
    hasSubBlockId = true;
  } else {
    StringRef dstModeKeyword;
    if (parser.parseKeyword(&dstModeKeyword)) {
      return failure();
    }
    if (dstModeKeyword == "split_m") {
      dstMode = AccStoreUbDstMode::SplitM;
    } else if (dstModeKeyword == "split_n") {
      dstMode = AccStoreUbDstMode::SplitN;
    } else {
      return parser.emitError(parser.getCurrentLocation(),
          "expected dst_mode(%sub_blockid), dst_mode(split_m), or dst_mode(split_n)");
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  return success();
}

ParseResult resolveMteL0cUbOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    bool hasSubBlockId, OpAsmParser::UnresolvedOperand subBlockId,
    Type subBlockIdType, const StructuredAccStoreAsmState &state) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(m, mType, result.operands) ||
      parser.resolveOperand(n, nType, result.operands) ||
      parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
      parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
      parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             parser.getCurrentLocation(), result.operands) ||
      (hasSubBlockId &&
       parser.resolveOperand(subBlockId, subBlockIdType, result.operands)) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  return success();
}
