// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===---------- SyncSolverCodeGen.h ---- Graph Sync Solver ----------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir::pto::syncsolver {

class CodeGenerator {
public:
  const SyncSolverOptions options;
  func::FuncOp funcOp;
  std::unique_ptr<OperationBase> funcIr;

private:
  SyncMap syncMapBefore, syncMapAfter;

public:
  CodeGenerator() = delete;

  explicit CodeGenerator(std::unique_ptr<Solver> solver)
      : options(solver->options) {
    auto [syncBefore, syncAfter] = solver->getBeforeAfterSyncMaps();
    syncMapBefore = std::move(syncBefore);
    syncMapAfter = std::move(syncAfter);
    funcOp = solver->funcOp;
    funcIr = std::move(solver->funcIr);
  }

  void generateResultOps();

private:
  Operation *resolveSyncAnchor(const OperationBase *opBase) const;
  Location resolveSyncLoc(OperationBase *opBase);
  void setInsertionPoint(IRRewriter &rewriter, OperationBase *opBase,
                         bool insertAfter);
  void emitSyncOp(IRRewriter &rewriter, SyncOp *syncOp);
  void emitSyncMap(IRRewriter &rewriter, SyncMap &syncMap, bool insertAfter);
};

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
