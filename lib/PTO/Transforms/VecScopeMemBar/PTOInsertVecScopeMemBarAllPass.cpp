// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarIR.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINSERTVECSCOPEMEMBARALL
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

// This debug-only pass intentionally bypasses dependence analysis. It inserts
// one independent VV_ALL immediately before every UB-backed vector memory op
// in the nearest vecscope.
static bool isNearestScope(Operation *op, Operation *scope) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(parent))
      return parent == scope;
  }
  return false;
}

static LogicalResult insertVecScopeMemBarAll(Operation *scope) {
  if (!scope || !isa<pto::VecScopeOp, pto::StrictVecScopeOp>(scope))
    return failure();

  SmallVector<Operation *, 16> accesses;
  scope->walk([&](Operation *op) {
    if (isNearestScope(op, scope) && isUBVectorMemoryOp(op))
      accesses.push_back(op);
  });

  for (Operation *access : accesses) {
    OpBuilder builder(access);
    auto attr = pto::MemBarAttr::get(access->getContext(), MemBarKind::VV_ALL);
    builder.create<pto::MemBarOp>(access->getLoc(), attr);
  }
  return success();
}

struct PTOInsertVecScopeMemBarAllPass
    : pto::impl::PTOInsertVecScopeMemBarAllBase<
          PTOInsertVecScopeMemBarAllPass> {
  PTOInsertVecScopeMemBarAllPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Operation *, 4> scopes;
    func.walk([&](Operation *op) {
      if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(op))
        scopes.push_back(op);
    });

    for (Operation *scope : scopes) {
      if (failed(insertVecScopeMemBarAll(scope))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertVecScopeMemBarAllPass() {
  return std::make_unique<PTOInsertVecScopeMemBarAllPass>();
}
