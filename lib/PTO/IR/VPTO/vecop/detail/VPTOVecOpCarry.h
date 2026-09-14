// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOpCarry.h - shared vecop Carry helpers ----------------------===//
//===----------------------------------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
//
// Carry helper group of the vecop per-instruction TUs.
// Internal to lib/PTO/IR/VPTO/vecop/detail; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VECOP_DETAIL_CARRY_H
#define PTO_IR_VPTO_VECOP_DETAIL_CARRY_H

#include "VPTOInternal.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

  template <typename CarryOp>
  [[maybe_unused]] static LogicalResult verifyCarryVecOp(CarryOp op) {
    if (failed(verifyIntegerVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
        failed(verifyIntegerVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
        failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
        failed(verifyIntegerVRegTypeLike(op, op.getResult().getType(),
                                         "result type")) ||
        failed(verifyMaskTypeLike(op, op.getCarry().getType(), "carry type"))) {
      return failure();
    }

    auto lhsType = cast<VRegType>(op.getLhs().getType());
    auto rhsType = cast<VRegType>(op.getRhs().getType());
    auto resultType = cast<VRegType>(op.getResult().getType());
    auto lhsElemType = cast<IntegerType>(lhsType.getElementType());
    if (lhsType != rhsType || lhsType != resultType) {
      return op.emitOpError("requires lhs, rhs, and result to have matching vector types");
    }
    if (lhsElemType.getWidth() != mlir::pto::kValue32) {
      return op.emitOpError("currently requires 32-bit integer vector elements");
    }
    return success();
  }

  template <typename CarryWithInputOp>
  [[maybe_unused]] static LogicalResult verifyCarryVecOpWithInput(CarryWithInputOp op) {
    if (failed(verifyCarryVecOp(op)) ||
        failed(verifyMaskTypeLike(op, op.getCarryIn().getType(),
                                  "carry_in type"))) {
      return failure();
    }
    return success();
  }

} // namespace mlir::pto::vecop_detail

#endif // PTO_IR_VPTO_VECOP_DETAIL_CARRY_H
