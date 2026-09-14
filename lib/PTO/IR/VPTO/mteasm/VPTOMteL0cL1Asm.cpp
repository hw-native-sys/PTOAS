// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL0cL1Asm.cpp - pto.mte_l0c_l1 asm helpers -------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"
#include "VPTOMteAsmInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalLoop3(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3CountOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3SrcStrideOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loop3DstStrideOperands) {
  StringRef parsedKeyword;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> loop3Operands;
  if (parseOptionalDmaTripleGroupAlias(parser, {"loop3"}, parsedKeyword,
                                       loop3Operands)) {
    return failure();
  }
  if (!parsedKeyword.empty()) {
    loop3CountOperands.push_back(loop3Operands[0]);
    loop3SrcStrideOperands.push_back(loop3Operands[1]);
    loop3DstStrideOperands.push_back(loop3Operands[mlir::pto::kValue2]);
  }
  return success();
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalFpc(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &fpcOperands) {
  if (failed(parser.parseOptionalKeyword("fpc"))) {
    return success();
  }
  if (parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand) || parser.parseRParen()) {
    return failure();
  }
  fpcOperands.push_back(operand);
  return success();
}

[[maybe_unused]] static void printMteL0cL1OptionalFpc(OpAsmPrinter &printer,
                                                      Value fpc) {
  if (fpc) {
    printer << ", fpc(" << fpc << ")";
  }
}

[[maybe_unused]] static void
printMteL0cL1OptionalFpcType(OpAsmPrinter &printer, Type fpcType) {
  if (fpcType) {
    printer << ", fpc(" << fpcType << ")";
  }
}

[[maybe_unused]] static ParseResult parseMteL0cL1OptionalLoop3Types(
    OpAsmParser &parser, SmallVectorImpl<Type> &loop3CountTypes,
    SmallVectorImpl<Type> &loop3SrcStrideTypes,
    SmallVectorImpl<Type> &loop3DstStrideTypes, StringRef opName) {
  if (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    if (keyword != "loop3") {
      return parser.emitError(parser.getCurrentLocation(), "expected 'loop3'");
    }
    SmallVector<Type> loop3GroupTypes;
    if (parseDmaTripleTypes(parser, loop3GroupTypes)) {
      return failure();
    }
    loop3CountTypes.push_back(loop3GroupTypes[0]);
    loop3SrcStrideTypes.push_back(loop3GroupTypes[1]);
    loop3DstStrideTypes.push_back(loop3GroupTypes[mlir::pto::kValue2]);
    if (succeeded(parser.parseOptionalComma())) {
      return parser.emitError(parser.getCurrentLocation(),
                              (Twine(opName) +
                               " accepts at most one loop3 group")
                                  .str());
    }
  }
  return success();
}

[[maybe_unused]] static ParseResult resolveStructuredMteL0cL1OptionalOperands(
    OpAsmParser &parser, StructuredAccStoreAsmState &state,
    SmallVectorImpl<Value> &resolvedOperands, OperationState &result) {
  auto location = parser.getCurrentLocation();
  if (parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.splitOperands, state.splitTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop0SrcStrideOperands,
                             state.loop0SrcStrideTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop3CountOperands, state.loop3CountTypes,
                             location, result.operands) ||
      parser.resolveOperands(state.loop3SrcStrideOperands,
                             state.loop3SrcStrideTypes, location,
                             result.operands) ||
      parser.resolveOperands(state.loop3DstStrideOperands,
                             state.loop3DstStrideTypes, location,
                             result.operands)) {
    return failure();
  }

  auto extractResolved = [&result, &resolvedOperands](SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
                             SmallVectorImpl<Type> &types) -> Value {
    (void)types;
    if (ops.empty()) {
      return {};
    }
    return result.operands[resolvedOperands.size()];
  };
  resolvedOperands.push_back(extractResolved(state.preQuantOperands,
                                             state.preQuantTypes));
  resolvedOperands.push_back(extractResolved(state.preReluOperands,
                                             state.preReluTypes));
  resolvedOperands.push_back(extractResolved(state.clipValueOperands,
                                             state.clipValueTypes));
  resolvedOperands.push_back(extractResolved(state.splitOperands,
                                             state.splitTypes));
  resolvedOperands.push_back(extractResolved(state.loop0SrcStrideOperands,
                                             state.loop0SrcStrideTypes));
  resolvedOperands.push_back(extractResolved(state.loop3CountOperands,
                                             state.loop3CountTypes));
  resolvedOperands.push_back(extractResolved(state.loop3SrcStrideOperands,
                                             state.loop3SrcStrideTypes));
  resolvedOperands.push_back(extractResolved(state.loop3DstStrideOperands,
                                             state.loop3DstStrideTypes));
  return success();
}

void setMteL0cL1SegmentSizes(OperationState &result,
                                        const StructuredAccStoreAsmState &st) {
  setStructuredAccStoreSegmentSizes<MteL0cL1Op>(
      result, {1, 1, 1, 1, 1, 1, !st.preQuantOperands.empty() ? 1 : 0,
               !st.preReluOperands.empty() ? 1 : 0,
               !st.clipValueOperands.empty() ? 1 : 0,
               !st.splitOperands.empty() ? 1 : 0,
               !st.loop0SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3CountOperands.empty() ? 1 : 0,
               !st.loop3SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3DstStrideOperands.empty() ? 1 : 0});
}

ParseResult parseMteL0cL1Types(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType,
    StructuredAccStoreAsmState &state) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() ||
      parser.parseType(nType) || parser.parseComma() ||
      parser.parseType(srcStrideType) || parser.parseComma() ||
      parser.parseType(dstStrideType) ||
      parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }
  return success();
}

ParseResult resolveMteL0cL1Operands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    StructuredAccStoreAsmState &state) {
  if (failed(resolveMteL0cPrefix(parser, result, source, sourceType,
                                 destination, destinationType, m, mType, n,
                                 nType, srcStride, srcStrideType, dstStride,
                                 dstStrideType, state)) ||
      failed(resolveMteL0cTail(parser, result, state))) {
    return failure();
  }
  return success();
}
