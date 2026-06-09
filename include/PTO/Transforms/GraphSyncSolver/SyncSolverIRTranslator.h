// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===--------- IRTranslator.h ---- Graph Sync Solver ------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <utility>

namespace mlir::pto::syncsolver {

class IRTranslator {
public:
  const SyncSolverOptions options;
  func::FuncOp funcOp;
  std::unique_ptr<OperationBase> funcIr;
  std::vector<std::unique_ptr<Occurrence>> syncIr;
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;
  llvm::DenseMap<const OperationBase *, std::vector<Occurrence *>>
      opAllOccurrences;
  std::vector<ProcessingOrder> processingOrders;
  llvm::DenseMap<Value, llvm::SmallVector<Value>> blockArgAliases;

  IRTranslator(func::FuncOp func, SyncSolverOptions options)
      : options(options), funcOp(func) {
    auto funcOp = std::make_unique<syncsolver::Function>(func.getOperation());
    auto scopeOp = funcIrBuilder(func.getRegion(), funcOp.get());
    funcOp->body.push_back(std::move(scopeOp));
    funcIr = std::move(funcOp);
    syncIrBuilder(funcIr.get());
  }

  IRTranslator(std::unique_ptr<OperationBase> funcIr, SyncSolverOptions options)
      : options(options), funcIr(std::move(funcIr)) {
    syncIrBuilder(this->funcIr.get());
  }

private:
  int64_t globalIndex{0};

  std::unique_ptr<Scope> funcIrBuilder(Region &region, OperationBase *parentOp,
                                       bool skipEmptyScopes = false);

  void generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);
  void generateProcessingOrders(const Loop *loopOp, Occurrence *occ,
                                bool isUseless);
  void generateProcessingOrders(const Scope *scopeOp, Occurrence *occ,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs1,
                                const llvm::SmallVector<Occurrence *> &occs2,
                                bool isUseless);
  void generateProcessingOrders(RWOperation *rwOp1, RWOperation *rwOp2,
                                Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);

  bool skipLaterIterations(Occurrence *occ1, Occurrence *occ2) const;

  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc = nullptr,
                     int depth = 0, bool isUseless = false);

  llvm::SmallVector<Value> tracebackMemVals(Value val);
  llvm::SmallVector<Value> tracebackMemValsStep(Value val);
  llvm::SmallVector<Value> getMemoryOps(const SmallVector<Value> &vals);

  std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
  getReadWriteMemoryOps(Operation *op);

  template <typename OP>
  std::unique_ptr<OperationBase> getLoadStoreOp(OP op, OperationBase *parentOp);

  std::unique_ptr<OperationBase> getPipeInterfaceOp(pto::OpPipeInterface op,
                                                    OperationBase *parentOp);

  std::unique_ptr<OperationBase> getTensorExtractOp(tensor::ExtractOp extractOp,
                                                    OperationBase *parentOp);

  std::unique_ptr<OperationBase> getCallOp(func::CallOp,
                                           const OperationBase *parentOp) const;

  void updateBlockArgAliases(Block *block, OperandRange destOperands);
  bool isUnlikelyCondition(const Condition *condOp) const;
  bool isParallelLoop(const Loop *loopOp) const;
  std::optional<int64_t> getLoopMultibufferUnrollNum(Loop *) const;
  std::optional<int64_t> getScopePreloadNum(Scope *) const;
  std::optional<int64_t> getScopeMaxPreloadNum(Scope *) const;
  Scope *prepareBlockScope(std::unique_ptr<Scope> &scopeOp, Block &block,
                           bool isFunctionRegion) const;
  void appendBlockBoundaries(Scope *parScope, Block &block) const;
  bool handleIfLikeOp(Operation &op, Scope *parScope, bool skipEmptyScopes);
  bool handleLoopLikeOp(Operation &op, Scope *parScope, bool skipEmptyScopes);
  bool handleBranchAliasOp(Operation &op);
  void appendRWOpFromOperation(Operation &op, Scope *parScope);
};

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
