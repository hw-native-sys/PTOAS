// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOGraphSyncSolver.cpp - Pass entry --------------------------------===//
//
// Wires the GraphSyncSolver components (IRTranslator -> Solver ->
// CodeGenerator) under a new Pass `pto-graph-sync-solver` that coexists with
// `pto-insert-sync` (the legacy InsertSync flow).
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "PTO/Transforms/Passes.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>

namespace mlir {
namespace pto {
// Mirror the alias-trick PTOInsertSync.cpp uses so the tablegen-generated
// `func::FuncOp` references resolve under `mlir::pto::`.
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOGRAPHSYNCSOLVER
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

namespace {

struct PTOGraphSyncSolverPass
    : public mlir::pto::impl::PTOGraphSyncSolverBase<PTOGraphSyncSolverPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // If the function already contains explicit synchronization ops we leave
    // it alone, matching the behavior of the legacy InsertSync pass to avoid
    // double-inserting on hand-written kernels.
    bool hasExplicitSync = false;
    func.walk([&](Operation *op) {
      if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::RecordEventOp,
              pto::WaitEventOp>(op)) {
        hasExplicitSync = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasExplicitSync) {
      return;
    }

    // Derive the arch mode from the module's --pto-arch attribute (same
    // source as LoweringSyncToPipe / PTOA5NormalizeTMov / PTOPlanMemory).
    // A2/A3 stay memory-based; A5 is register-based and lets
    // handleBarrierConflict() drop the PIPE_V barrier that A5 hardware
    // does not support.
    const bool isA5 = pto::isTargetArchA5(func.getOperation());
    SyncSolverOptions opts(SyncMode::INTRA_CORE_SYNC,
                           /*isMemBasedArch=*/!isA5,
                           /*isRegBasedArch=*/isA5);
    opts.eventIdNumMax = eventIdNumMax;
    auto translator = std::make_unique<IRTranslator>(func, opts);

    // Trivial / empty function bodies have nothing to solve.
    if (translator->processingOrders.empty()) {
      return;
    }

    auto solver = std::make_unique<Solver>(std::move(translator));
    solver->solve();

    CodeGenerator codegen(std::move(solver));
    codegen.generateResultOps();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOGraphSyncSolverPass(
    const PTOGraphSyncSolverOptions &options) {
  return std::make_unique<PTOGraphSyncSolverPass>(options);
}
