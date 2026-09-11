// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL0cGmAsm.cpp - pto.mte_l0c_gm asm helpers -------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"
#include "VPTOMteAsmInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void setMteL0cGmSegmentSizes(OperationState &result,
                                        const StructuredAccStoreAsmState &st) {
  setStructuredAccStoreSegmentSizes<MteL0cGmOp>(
      result, {1, 1, 1, 1, 1, 1, !st.preQuantOperands.empty() ? 1 : 0,
               !st.preReluOperands.empty() ? 1 : 0,
               !st.clipValueOperands.empty() ? 1 : 0, 1, 1,
               !st.splitOperands.empty() ? 1 : 0,
               !st.loop0SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3CountOperands.empty() ? 1 : 0,
               !st.loop3SrcStrideOperands.empty() ? 1 : 0,
               !st.loop3DstStrideOperands.empty() ? 1 : 0});
}

ParseResult parseMteL0cGmTypes(
    OpAsmParser &parser, Type &sourceType, Type &destinationType,
    Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType,
    Type &sidType, Type &l2CacheCtrlType,
    StructuredAccStoreAsmState &state) {
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(mType) || parser.parseComma() ||
      parser.parseType(nType) || parser.parseComma() ||
      parser.parseType(srcStrideType) || parser.parseComma() ||
      parser.parseType(dstStrideType) || parser.parseComma() ||
      parser.parseType(sidType) || parser.parseComma() ||
      parser.parseType(l2CacheCtrlType) ||
      parseStructuredAccStoreTailTypes(parser, state)) {
    return failure();
  }
  return success();
}

ParseResult resolveMteL0cGmOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType,
    OpAsmParser::UnresolvedOperand destination, Type destinationType,
    OpAsmParser::UnresolvedOperand m, Type mType,
    OpAsmParser::UnresolvedOperand n, Type nType,
    OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
    OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
    OpAsmParser::UnresolvedOperand sid, Type sidType,
    OpAsmParser::UnresolvedOperand l2CacheCtrl, Type l2CacheCtrlType,
    StructuredAccStoreAsmState &state) {
  if (failed(resolveMteL0cPrefix(parser, result, source, sourceType,
                                 destination, destinationType, m, mType, n,
                                 nType, srcStride, srcStrideType, dstStride,
                                 dstStrideType, state)) ||
      parser.resolveOperand(sid, sidType, result.operands) ||
      parser.resolveOperand(l2CacheCtrl, l2CacheCtrlType, result.operands) ||
      failed(resolveMteL0cTail(parser, result, state))) {
    return failure();
  }
  return success();
}
