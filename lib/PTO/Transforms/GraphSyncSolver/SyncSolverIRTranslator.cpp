// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===--------- IRTranslator.cpp ------- Graph Sync Solver -------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"

#include "PTO/IR/PTO.h"
#include "../Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

#define DEBUG_TYPE "pto-gss-ir-translator"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

namespace {
constexpr unsigned kMemoryEffectInlineCapacity = 4;
constexpr int64_t kBalancedOccurrenceSplitFactor = 2;
} // namespace

llvm::SmallVector<Value> IRTranslator::tracebackMemValsStep(Value val) {
  llvm::SmallVector<Value> out;
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp =
            dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (auto *init = forOp.getTiedLoopInit(blockArg))
        out.push_back(init->get());
      if (auto *yield = forOp.getTiedLoopYieldedValue(blockArg))
        out.push_back(yield->get());
    }
    if (blockArgAliases.contains(val))
      llvm::append_range(out, blockArgAliases[val]);
    return out;
  }

  auto result = dyn_cast<OpResult>(val);
  if (!result)
    return out;

  Operation *defOp = result.getDefiningOp();
  unsigned resultNo = result.getResultNumber();
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    out.push_back(ifOp.thenYield()->getOperand(resultNo));
    if (ifOp.elseBlock())
      out.push_back(ifOp.elseYield()->getOperand(resultNo));
  } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    if (forOp.getYieldedValues().size() > resultNo)
      out.push_back(forOp.getYieldedValues()[resultNo]);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
    if (whileOp.getConditionOp().getArgs().size() > resultNo)
      out.push_back(whileOp.getConditionOp().getArgs()[resultNo]);
    if (whileOp.getYieldedValues().size() > resultNo)
      out.push_back(whileOp.getYieldedValues()[resultNo]);
  }

  if (auto alias = pto::getOperationAliasInfo(defOp)) {
    if (alias->first == result)
      out.push_back(alias->second);
  } else if (auto dsi = dyn_cast<DestinationStyleOpInterface>(defOp)) {
    for (Value init : dsi.getDpsInits())
      out.push_back(init);
  }
  return out;
}

llvm::SmallVector<Value> IRTranslator::tracebackMemVals(Value val) {
  std::queue<Value> queue;
  llvm::DenseSet<Value> visited;
  llvm::SetVector<Value> leaves;
  queue.push(val);
  visited.insert(val);

  while (!queue.empty()) {
    Value cur = queue.front();
    queue.pop();

    auto nextVals = tracebackMemValsStep(cur);
    if (!nextVals.empty()) {
      for (Value next : nextVals) {
        if (next && !visited.contains(next)) {
          queue.push(next);
          visited.insert(next);
        }
      }
      continue;
    }

    if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
      leaves.insert(blockArg);
      continue;
    }

    auto result = dyn_cast<OpResult>(cur);
    if (!result)
      continue;
    Operation *defOp = result.getDefiningOp();
    if (isa<pto::PointerCastOp, pto::AllocTileOp, tensor::EmptyOp,
            memref::AllocOp>(defOp)) {
      leaves.insert(result);
      continue;
    }
    leaves.insert(result);
  }

  return leaves.takeVector();
}

llvm::SmallVector<Value>
IRTranslator::getMemoryOps(const SmallVector<Value> &vals) {
  llvm::SetVector<Value> out;
  for (Value val : vals)
    for (Value memVal : tracebackMemVals(val))
      out.insert(memVal);
  return out.takeVector();
}

std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
IRTranslator::getReadWriteMemoryOps(Operation *op) {
  llvm::SmallVector<Value> reads;
  llvm::SmallVector<Value> writes;

  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
                kMemoryEffectInlineCapacity>
        effects;
    memEffect.getEffects(effects);
    for (auto &effect : effects) {
      Value value = effect.getValue();
      if (!value)
        continue;
      if (isa<MemoryEffects::Read>(effect.getEffect()))
        llvm::append_range(reads, getMemoryOps({value}));
      else if (isa<MemoryEffects::Write>(effect.getEffect()))
        llvm::append_range(writes, getMemoryOps({value}));
    }
  }

  if (auto dsi = dyn_cast<DestinationStyleOpInterface>(op)) {
    llvm::append_range(reads, getMemoryOps(dsi.getDpsInputs()));
    llvm::append_range(writes, getMemoryOps(dsi.getDpsInits()));
  }
  return {std::move(reads), std::move(writes)};
}

template <typename OP>
std::unique_ptr<OperationBase>
IRTranslator::getLoadStoreOp(OP op, OperationBase *parentOp) {
  auto pipe = pto::PIPE::PIPE_S;
  llvm::SmallVector<Value> reads;
  llvm::SmallVector<Value> writes;
  if constexpr (std::is_same_v<OP, memref::LoadOp> ||
                std::is_same_v<OP, affine::AffineLoadOp>) {
    reads = getMemoryOps({op.getMemRef()});
  } else {
    writes = getMemoryOps({op.getMemRef()});
  }
  return std::make_unique<RWOperation>(
      op.getOperation(), parentOp, TCoreType::CUBE_OR_VECTOR, pipe, pipe, reads,
      writes);
}

std::unique_ptr<OperationBase>
IRTranslator::getPipeInterfaceOp(pto::OpPipeInterface op,
                                 OperationBase *parentOp) {
  auto [reads, writes] = getReadWriteMemoryOps(op.getOperation());
  pto::PIPE pipeRead = op.getPipe();
  pto::PIPE pipeWrite = op.getPipe();
  if (pipeRead == pto::PIPE::PIPE_UNASSIGNED ||
      pipeWrite == pto::PIPE::PIPE_UNASSIGNED)
    return nullptr;
  return std::make_unique<RWOperation>(
      op.getOperation(), parentOp, TCoreType::CUBE_OR_VECTOR, pipeRead,
      pipeWrite, reads, writes);
}

std::unique_ptr<OperationBase>
IRTranslator::getTensorExtractOp(tensor::ExtractOp extractOp,
                                 OperationBase *parentOp) {
  auto reads = getMemoryOps({extractOp.getTensor()});
  return std::make_unique<RWOperation>(
      extractOp.getOperation(), parentOp, TCoreType::CUBE_OR_VECTOR,
      pto::PIPE::PIPE_S, pto::PIPE::PIPE_S, reads,
      llvm::SmallVector<Value>());
}

std::unique_ptr<OperationBase>
IRTranslator::getCallOp(func::CallOp, const OperationBase *parentOp) const {
  (void)parentOp;
  return nullptr;
}

void IRTranslator::updateBlockArgAliases(Block *block,
                                         OperandRange destOperands) {
  if (block->getNumArguments() != destOperands.size())
    return;
  for (auto [arg, operand] : llvm::zip(block->getArguments(), destOperands))
    blockArgAliases[arg].push_back(operand);
}

bool IRTranslator::isUnlikelyCondition(const Condition *condOp) const {
  return condOp && condOp->op &&
         condOp->op->hasAttrOfType<UnitAttr>("pto.unlikely_condition");
}

bool IRTranslator::isParallelLoop(const Loop *loopOp) const {
  return loopOp && loopOp->op &&
         loopOp->op->hasAttrOfType<UnitAttr>("pto.parallel_loop");
}

std::optional<int64_t>
IRTranslator::getLoopMultibufferUnrollNum(Loop *) const {
  return {};
}

std::optional<int64_t> IRTranslator::getScopePreloadNum(Scope *) const {
  return {};
}

std::optional<int64_t>
IRTranslator::getScopeMaxPreloadNum(Scope *) const {
  return {};
}

Scope *IRTranslator::prepareBlockScope(std::unique_ptr<Scope> &scopeOp,
                                       Block &block,
                                       bool isFunctionRegion) const {
  Scope *parScope = scopeOp.get();
  if (isFunctionRegion) {
    auto blockOp = std::make_unique<FunctionBlock>();
    blockOp->parentOp = scopeOp.get();
    parScope = blockOp.get();
    scopeOp->body.push_back(std::move(blockOp));
  }
  auto blockBegin = std::make_unique<PlaceHolder>(nullptr, parScope);
  blockBegin->scopeBegin = parScope;
  blockBegin->block = &block;
  parScope->body.push_back(std::move(blockBegin));
  return parScope;
}

void IRTranslator::appendBlockBoundaries(Scope *parScope, Block &block) const {
  auto blockEnd = std::make_unique<PlaceHolder>(nullptr, parScope);
  blockEnd->scopeEnd = parScope;
  blockEnd->block = &block;
  parScope->body.push_back(std::move(blockEnd));
}

bool IRTranslator::handleIfLikeOp(Operation &op, Scope *parScope,
                                  bool skipEmptyScopes) {
  auto ifOp = dyn_cast<scf::IfOp>(op);
  if (!ifOp)
    return false;
  auto trueScope = funcIrBuilder(ifOp.getThenRegion(), nullptr, skipEmptyScopes);
  std::unique_ptr<Scope> falseScope;
  if (ifOp.elseBlock())
    falseScope = funcIrBuilder(ifOp.getElseRegion(), nullptr, skipEmptyScopes);
  auto cond = std::make_unique<Condition>(&op, parScope, std::move(trueScope),
                                          std::move(falseScope));
  cond->isUnlikely = isUnlikelyCondition(cond.get());
  if (!skipEmptyScopes || !isEmptyScope(cond.get()))
    parScope->body.push_back(std::move(cond));
  return true;
}

bool IRTranslator::handleLoopLikeOp(Operation &op, Scope *parScope,
                                    bool skipEmptyScopes) {
  if (!isa<LoopLikeOpInterface>(op))
    return false;
  auto loop = std::make_unique<Loop>(&op, parScope);
  loop->isParallel = isParallelLoop(loop.get());
  loop->multibufferUnrollNum = getLoopMultibufferUnrollNum(loop.get());
  for (Region &nested : op.getRegions()) {
    auto innerScope = funcIrBuilder(nested, loop.get(), skipEmptyScopes);
    loop->body.push_back(std::move(innerScope));
  }
  auto before = std::make_unique<PlaceHolder>(nullptr, loop->parentOp);
  before->beforeOp = loop.get();
  auto after = std::make_unique<PlaceHolder>(nullptr, loop->parentOp);
  after->afterOp = loop.get();
  if (!skipEmptyScopes || !isEmptyScope(loop.get())) {
    parScope->body.push_back(std::move(before));
    parScope->body.push_back(std::move(loop));
    parScope->body.push_back(std::move(after));
  }
  return true;
}

bool IRTranslator::handleBranchAliasOp(Operation &op) {
  if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
    updateBlockArgAliases(branchOp.getDest(), branchOp.getDestOperands());
    return true;
  }
  if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
    updateBlockArgAliases(condBranchOp.getTrueDest(),
                          condBranchOp.getTrueDestOperands());
    updateBlockArgAliases(condBranchOp.getFalseDest(),
                          condBranchOp.getFalseDestOperands());
    return true;
  }
  return false;
}

void IRTranslator::appendRWOpFromOperation(Operation &op, Scope *parScope) {
  std::unique_ptr<OperationBase> rw;
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op))
    rw = getPipeInterfaceOp(pipeOp, parScope);
  else if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    rw = getLoadStoreOp(storeOp, parScope);
  else if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    rw = getLoadStoreOp(loadOp, parScope);
  else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
    rw = getLoadStoreOp(storeOp, parScope);
  else if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
    rw = getLoadStoreOp(loadOp, parScope);
  else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op))
    rw = getTensorExtractOp(extractOp, parScope);
  else if (auto callOp = dyn_cast<func::CallOp>(op))
    rw = getCallOp(callOp, parScope);
  if (rw)
    parScope->body.push_back(std::move(rw));
}

std::unique_ptr<Scope> IRTranslator::funcIrBuilder(Region &region,
                                                   OperationBase *parentOp,
                                                   bool skipEmptyScopes) {
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = parentOp;
  bool isFunctionRegion = isa_and_present<Function>(parentOp);
  if (!isFunctionRegion && region.getBlocks().size() > 1)
    return scopeOp;

  for (Block &block : region.getBlocks()) {
    Scope *parScope = prepareBlockScope(scopeOp, block, isFunctionRegion);
    for (Operation &op : block.getOperations()) {
      if (handleIfLikeOp(op, parScope, skipEmptyScopes) ||
          handleLoopLikeOp(op, parScope, skipEmptyScopes) ||
          handleBranchAliasOp(op)) {
        continue;
      }
      appendRWOpFromOperation(op, parScope);
    }
    appendBlockBoundaries(parScope, block);
  }
  return scopeOp;
}

bool IRTranslator::skipLaterIterations(Occurrence *occ1,
                                       Occurrence *occ2) const {
  auto skip = [](const Occurrence *occ, const Occurrence *other) {
    if (!occ->parentOcc || !isa<Loop>(occ->parentOcc->op))
      return false;
    int split = occ->parentOcc->loopSplitIndex;
    return occ->syncIrIndex < split && split <= other->syncIrIndex;
  };
  return skip(occ1, occ2) || skip(occ2, occ1);
}

void IRTranslator::generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  if (skipLaterIterations(occ1, occ2))
    return;
  if (isa<Scope>(occ1->op) && isa<Scope>(occ2->op)) {
    generateProcessingOrders(occ1->childOccs, occ2->childOccs, isUseless);
  }
  if (isa<RWOperation>(occ1->op) && isa<Scope>(occ2->op)) {
    generateProcessingOrders({occ1}, occ2->childOccs, isUseless);
  }
  if (isa<Scope>(occ1->op) && isa<RWOperation>(occ2->op)) {
    generateProcessingOrders(occ1->childOccs, {occ2}, isUseless);
  }
  if (auto *rw1 = dyn_cast<RWOperation>(occ1->op)) {
    if (auto *rw2 = dyn_cast<RWOperation>(occ2->op))
      generateProcessingOrders(rw1, rw2, occ1, occ2, isUseless);
  }
}

void IRTranslator::generateProcessingOrders(
    const llvm::SmallVector<Occurrence *> &occs, bool isUseless) {
  int64_t n = static_cast<int64_t>(occs.size());
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i - 1; j >= 0; --j)
      generateProcessingOrders(occs[j], occs[i], isUseless);
}

void IRTranslator::generateProcessingOrders(
    const llvm::SmallVector<Occurrence *> &occs1,
    const llvm::SmallVector<Occurrence *> &occs2, bool isUseless) {
  for (auto *occ2 : occs2)
    for (auto *occ1 : llvm::reverse(occs1))
      generateProcessingOrders(occ1, occ2, isUseless);
}

void IRTranslator::generateProcessingOrders(const Scope *scopeOp, Occurrence *occ,
                                            bool isUseless) {
  (void)scopeOp;
  generateProcessingOrders(occ->childOccs, isUseless);
}

void IRTranslator::generateProcessingOrders(const Loop *loopOp, Occurrence *occ,
                                            bool isUseless) {
  (void)loopOp;
  int64_t childNum = static_cast<int64_t>(occ->childOccs.size());
  if (childNum == 0 || childNum % kBalancedOccurrenceSplitFactor != 0)
    return;
  int64_t halfChildNum = childNum / kBalancedOccurrenceSplitFactor;
  SmallVector<Occurrence *> first(occ->childOccs.begin(),
                                  occ->childOccs.begin() + halfChildNum);
  SmallVector<Occurrence *> second(occ->childOccs.begin() + halfChildNum,
                                   occ->childOccs.end());
  generateProcessingOrders(first, isUseless);
  generateProcessingOrders(second, /*isUseless=*/true);
  for (auto *occ2 : second)
    for (auto *occ1 : llvm::reverse(first))
      generateProcessingOrders(occ1->childOccs, occ2->childOccs, isUseless);
}

void IRTranslator::generateProcessingOrders(RWOperation *rwOp1,
                                            RWOperation *rwOp2,
                                            Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  processingOrders.emplace_back(occ1, occ2, rwOp1, rwOp2, isUseless);
}

void IRTranslator::syncIrBuilder(OperationBase *op, Occurrence *parentOcc,
                                 int depth, bool isUseless) {
  int startIndex = globalIndex++;
  auto occ = std::make_unique<Occurrence>(op, parentOcc, depth, startIndex, -1);
  occ->syncIrIndex = static_cast<int>(syncIr.size());
  if (auto *rwOp = dyn_cast<RWOperation>(op))
    occ->hasUnitFlagFeat = rwOp->hasUnitFlagFeat;
  syncIr.push_back(std::move(occ));
  Occurrence *occPtr = syncIr.back().get();
  opAllOccurrences[op].push_back(occPtr);
  if (parentOcc)
    parentOcc->childOccs.push_back(occPtr);

  if (auto *loopOp = dyn_cast<Loop>(op)) {
    for (auto &child : loopOp->body)
      syncIrBuilder(child.get(), occPtr, depth + 1, isUseless);
    occPtr->loopSplitIndex = static_cast<int>(syncIr.size());
    for (auto &child : loopOp->body)
      syncIrBuilder(child.get(), occPtr, depth + 1, true);
    generateProcessingOrders(loopOp, occPtr, isUseless);
  } else if (auto *scopeOp = dyn_cast<Scope>(op)) {
    for (auto &child : scopeOp->body)
      syncIrBuilder(child.get(), occPtr, depth + 1, isUseless);
    generateProcessingOrders(scopeOp, occPtr, isUseless);
  }

  int endIndex = globalIndex++;
  occPtr->endIndex = endIndex;
  occPtr->syncIrEndIndex = static_cast<int>(syncIr.size());
}
