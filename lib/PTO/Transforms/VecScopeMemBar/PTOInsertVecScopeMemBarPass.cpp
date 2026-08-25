// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarAnalysis.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarCodegen.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarPlacement.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINSERTVECSCOPEMEMBAR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

struct PTOInsertVecScopeMemBarPass
    : pto::impl::PTOInsertVecScopeMemBarBase<PTOInsertVecScopeMemBarPass> {
  PTOInsertVecScopeMemBarPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<Operation *, 4> scopes;
    func.walk([&](Operation *op) {
      if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(op))
        scopes.push_back(op);
    });

    auto analyzeAndApply = [&](FailureOr<VecScopeMemBarAnalysisResult> result) {
      if (failed(result))
        return failure();
      auto plan = solveVecScopeMemBarPlacement(*result);
      if (failed(plan))
        return failure();
      return applyVecScopeMemBarPlan(*result, *plan);
    };

    for (Operation *scope : scopes) {
      if (failed(analyzeAndApply(runVecScopeMemBarAnalysis(scope)))) {
        signalPassFailure();
        return;
      }
    }

    // Cross-vecscope RAW/WAW hazards between sibling vecscopes are NOT handled
    // here. pto.mem_bar (SMEM_BAR) is an intra-vecscope primitive (see
    // docs/vpto-spec.md, "Intra-Pipeline Memory Barriers (within __VEC_SCOPE__)"):
    // it only orders vector ops within a single vecscope and cannot make a
    // producer vecscope's writes visible to a later sibling vecscope, whose
    // execution ordering/visibility must instead be managed by the host with
    // pto.set_flag / pto.wait_flag (or pto.get_buf / pto.rls_buf).
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertVecScopeMemBarPass() {
  return std::make_unique<PTOInsertVecScopeMemBarPass>();
}
