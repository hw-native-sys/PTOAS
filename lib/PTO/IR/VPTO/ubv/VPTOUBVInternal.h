// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOUBVInternal.h - shared VPTOUBV helpers -------------------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
// Internal to lib/PTO/IR/VPTO/ubv; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VPTOUBV_INTERNAL_H
#define PTO_IR_VPTO_VPTOUBV_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::ubv_detail {

using namespace mlir;
using namespace mlir::pto;

  [[maybe_unused]] static LogicalResult verifyUBBinaryOperands(Operation *op, Value dst,
                                                  Value src0, Value src1) {
    if (!isBufferLike(dst.getType()) || !isBufferLike(src0.getType()) ||
        !isBufferLike(src1.getType())) {
      return op->emitOpError("requires pointer-like operands");
    }
    if (classifyMemoryRole(dst.getType()) != MemoryRole::UB ||
        classifyMemoryRole(src0.getType()) != MemoryRole::UB ||
        classifyMemoryRole(src1.getType()) != MemoryRole::UB) {
      return op->emitOpError("requires UB-backed operands");
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyUBUnaryOperands(Operation *op, Value dst, Value src) {
    if (!isBufferLike(dst.getType()) || !isBufferLike(src.getType())) {
      return op->emitOpError("requires pointer-like operands");
    }
    if (classifyMemoryRole(dst.getType()) != MemoryRole::UB ||
        classifyMemoryRole(src.getType()) != MemoryRole::UB) {
      return op->emitOpError("requires UB-backed operands");
    }
    return success();
  }


} // namespace mlir::pto::ubv_detail

#endif // PTO_IR_VPTO_VPTOUBV_INTERNAL_H
