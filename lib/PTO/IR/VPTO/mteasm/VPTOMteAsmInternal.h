// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteAsmInternal.h - shared MTE asm operand resolvers ------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// L0c prefix/tail operand resolvers shared by the L0cL1 and L0cGm asm files.
// Internal to lib/PTO/IR/VPTO/mteasm; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_MTEASM_INTERNAL_H
#define PTO_IR_VPTO_MTEASM_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::mte_detail {

using namespace mlir;
using namespace mlir::pto;

  [[maybe_unused]] static ParseResult resolveMteL0cPrefix(
      OpAsmParser &parser, OperationState &result,
      OpAsmParser::UnresolvedOperand source, Type sourceType,
      OpAsmParser::UnresolvedOperand destination, Type destinationType,
      OpAsmParser::UnresolvedOperand m, Type mType,
      OpAsmParser::UnresolvedOperand n, Type nType,
      OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType,
      OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType,
      StructuredAccStoreAsmState &state) {
    auto loc = parser.getCurrentLocation();
    if (parser.resolveOperand(source, sourceType, result.operands) ||
        parser.resolveOperand(destination, destinationType, result.operands) ||
        parser.resolveOperand(m, mType, result.operands) ||
        parser.resolveOperand(n, nType, result.operands) ||
        parser.resolveOperand(srcStride, srcStrideType, result.operands) ||
        parser.resolveOperand(dstStride, dstStrideType, result.operands) ||
        parser.resolveOperands(state.preQuantOperands, state.preQuantTypes,
                               loc, result.operands) ||
        parser.resolveOperands(state.preReluOperands, state.preReluTypes,
                               loc, result.operands) ||
        parser.resolveOperands(state.clipValueOperands, state.clipValueTypes,
                               loc, result.operands)) {
      return failure();
    }
    return success();
  }

  [[maybe_unused]] static ParseResult resolveMteL0cTail(OpAsmParser &parser,
                                       OperationState &result,
                                       StructuredAccStoreAsmState &state) {
    auto loc = parser.getCurrentLocation();
    if (parser.resolveOperands(state.splitOperands, state.splitTypes,
                               loc, result.operands) ||
        parser.resolveOperands(state.loop0SrcStrideOperands,
                                state.loop0SrcStrideTypes, loc,
                                result.operands) ||
        parser.resolveOperands(state.loop3CountOperands,
                                state.loop3CountTypes, loc,
                                result.operands) ||
        parser.resolveOperands(state.loop3SrcStrideOperands,
                                state.loop3SrcStrideTypes, loc,
                                result.operands) ||
        parser.resolveOperands(state.loop3DstStrideOperands,
                                state.loop3DstStrideTypes, loc,
                                result.operands)) {
      return failure();
    }
    return success();
  }

} // namespace mlir::pto::mte_detail

#endif // PTO_IR_VPTO_MTEASM_INTERNAL_H
