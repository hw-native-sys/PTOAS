// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOASSIGNDEFAULTFRONTENDPIPEID
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

template <typename OpT>
static void assignDefaultIdIfMissing(OpT op, IntegerAttr zeroAttr) {
  if (!op.getIdAttr()) {
    op.setIdAttr(zeroAttr);
  }
}

struct PTOAssignDefaultFrontendPipeIdPass
    : public mlir::pto::impl::PTOAssignDefaultFrontendPipeIdBase<
          PTOAssignDefaultFrontendPipeIdPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Builder builder(funcOp.getContext());
    auto zeroAttr = builder.getI32IntegerAttr(0);

    funcOp.walk([&](Operation *op) {
      if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
        assignDefaultIdIfMissing(init, zeroAttr);
        return WalkResult::advance();
      }
      if (auto init = dyn_cast<AivInitializePipeOp>(op)) {
        assignDefaultIdIfMissing(init, zeroAttr);
        return WalkResult::advance();
      }
      if (auto alloc = dyn_cast<TAllocToAivOp>(op)) {
        assignDefaultIdIfMissing(alloc, zeroAttr);
        return WalkResult::advance();
      }
      if (auto alloc = dyn_cast<TAllocToAicOp>(op)) {
        assignDefaultIdIfMissing(alloc, zeroAttr);
        return WalkResult::advance();
      }
      if (auto push = dyn_cast<TPushToAivOp>(op)) {
        assignDefaultIdIfMissing(push, zeroAttr);
        return WalkResult::advance();
      }
      if (auto push = dyn_cast<TPushToAicOp>(op)) {
        assignDefaultIdIfMissing(push, zeroAttr);
        return WalkResult::advance();
      }
      if (auto pop = dyn_cast<TPopFromAicOp>(op)) {
        assignDefaultIdIfMissing(pop, zeroAttr);
        return WalkResult::advance();
      }
      if (auto pop = dyn_cast<TPopFromAivOp>(op)) {
        assignDefaultIdIfMissing(pop, zeroAttr);
        return WalkResult::advance();
      }
      if (auto free = dyn_cast<TFreeFromAicOp>(op)) {
        assignDefaultIdIfMissing(free, zeroAttr);
        return WalkResult::advance();
      }
      if (auto free = dyn_cast<TFreeFromAivOp>(op)) {
        assignDefaultIdIfMissing(free, zeroAttr);
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOAssignDefaultFrontendPipeIdPass() {
  return std::make_unique<PTOAssignDefaultFrontendPipeIdPass>();
}
