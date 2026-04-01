// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//

#include "OptMemPlanForPipeline.h"

using namespace mlir;
using namespace mlir::detail;
using namespace mlir::pto;

void OptMemPlanForDma::build(func::FuncOp func) {
  auto result = func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      UpdateScalarBuffers(loadOp);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      UpdateScalarBuffers(storeOp);
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("OptMemPlanForLoop Traverse IR Failed! ");
  }
}

void OptMemPlanForDma::UpdateDmaBuffers(SmallVector<Value> dpsOperand) {
  for (Value operand : dpsOperand) {
    auto memorySpaceAttr = GetBufferSpaceAttr(operand);
    if (!isLocalBuffer(memorySpaceAttr)) {
      continue;
    }
    DmaBuffers.insert(tracebackMemRef(operand));
  }
}

bool OptMemPlanForDma::IsDmaBuffer(const Value buf) const {
  if (DmaBuffers.empty()) {
    return false;
  }
  for (auto buffer : DmaBuffers) {
    if (buffer == buf) {
      return true;
    }
  }
  return false;
}

bool OptMemPlanForDma::BufferPipeConflict(const Value buf1,
                                          const Value buf2) const {
  if (IsScalarBuffer(buf1) && IsScalarBuffer(buf2)) {
    return false;
  }

  if (IsScalarBuffer(buf1) || IsScalarBuffer(buf2)) {
    return true;
  }

  if (IsDmaBuffer(buf1) || IsDmaBuffer(buf2)) {
    // Process the operation of ForOp as follows:
    // scf.for %arg4 = %c0 to %c1024 step %c128 ->
    //   alloca %allocA
    //   gm2ub(allocA, gm)
    //   ...
    //   alloca %allocB
    //   ub2gm(gm, allocB)
    // There is a conflict in the reuse of allocA and allocB here.
    // MTE3 and MTE3, MTE2 and MTE2 also have similar conflicts.
    return true;
  }
  return false;
}

template <typename OP>
typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                            std::is_same_v<OP, memref::StoreOp>,
                        void>::type
OptMemPlanForDma::UpdateScalarBuffers(OP op) {
  auto memorySpaceAttr = GetBufferSpaceAttr(op.getMemRef());
  if (!isLocalBuffer(memorySpaceAttr)) {
    return;
  }
  ScalarBuffers.insert(tracebackMemRef(op.getMemRef()));
}

bool OptMemPlanForDma::IsScalarBuffer(const Value buf) const {
  if (ScalarBuffers.empty()) {
    return false;
  }
  for (auto buffer : ScalarBuffers) {
    if (buffer == buf) {
      return true;
    }
  }
  return false;
}

void OptMemPlanForDma::UpdateScalarBuffersForLowerToLoops(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto memorySpaceAttr = GetBufferSpaceAttr(operand);
    if (!isLocalBuffer(memorySpaceAttr)) {
      continue;
    }
    ScalarBuffers.insert(tracebackMemRef(operand));
  }
}
