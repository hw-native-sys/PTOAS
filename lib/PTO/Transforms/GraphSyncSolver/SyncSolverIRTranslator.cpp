// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// This file is derived from the HIVM GraphSyncSolver in AscendNPU-IR/bishengir
// (https://github.com/AscendNPU-IR/bishengir), licensed under the Apache
// License, Version 2.0.  Upstream copyright and license notice:
//   Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//       http://www.apache.org/licenses/LICENSE-2.0
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//===--------- IRTranslator.cpp ------- Graph Sync Solver -------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"

#include "PTO/IR/PTO.h"
#include "../Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

bool isTransparentGraphSyncRegionOp(Operation &op) {
  return isa<pto::SectionCubeOp, pto::SectionVectorOp>(op);
}
} // namespace

llvm::SmallVector<Value> IRTranslator::tracebackMemValsStep(Value val) {
  llvm::SmallVector<Value> out;
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp =
            dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (auto *init = forOp.getTiedLoopInit(blockArg)) {
        out.push_back(init->get());
      }
      if (auto *yield = forOp.getTiedLoopYieldedValue(blockArg)) {
        out.push_back(yield->get());
      }
    }
    if (auto whileOp = dyn_cast_if_present<scf::WhileOp>(
            blockArg.getOwner()->getParentOp())) {
      unsigned index = blockArg.getArgNumber();
      if (blockArg.getOwner() == &whileOp.getBefore().front()) {
        if (index < whileOp.getInits().size()) {
          out.push_back(whileOp.getInits()[index]);
        }
        if (index < whileOp.getYieldedValues().size()) {
          out.push_back(whileOp.getYieldedValues()[index]);
        }
      } else if (blockArg.getOwner() == &whileOp.getAfter().front()) {
        auto conditionArgs = whileOp.getConditionOp().getArgs();
        if (index < conditionArgs.size()) {
          out.push_back(conditionArgs[index]);
        }
        if (index < whileOp.getYieldedValues().size()) {
          out.push_back(whileOp.getYieldedValues()[index]);
        }
      }
    }
    if (blockArgAliases.contains(val)) {
      llvm::append_range(out, blockArgAliases[val]);
    }
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

  // Stop at `pto.multi_tile_get` so getMemInfo preserves the selected slot
  // instead of tracing through the metadata-only view.
  if (isa<pto::MultiTileGetOp>(defOp)) {
    return out;
  }

  if (auto select = dyn_cast<arith::SelectOp>(defOp)) {
    out.push_back(select.getTrueValue());
    out.push_back(select.getFalseValue());
  } else if (auto addPtr = dyn_cast<pto::AddPtrOp>(defOp)) {
    out.push_back(addPtr.getPtr());
  } else if (auto intToPtr = dyn_cast<pto::IntToPtrOp>(defOp)) {
    if (auto ptrToInt = intToPtr.getAddr().getDefiningOp<pto::PtrToIntOp>())
      out.push_back(ptrToInt.getPtr());
  } else if (auto castPtr = dyn_cast<pto::CastPtrOp>(defOp);
             castPtr &&
             isa<pto::PtrType>(castPtr.getInput().getType()) &&
             isa<pto::PtrType>(castPtr.getResult().getType())) {
    out.push_back(castPtr.getInput());
  } else if (auto alias = pto::getOperationAliasInfo(defOp)) {
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
    // `pto.multi_tile_get` stops traversal so getMemInfo can extract the
    // slot-narrowed physical addresses.
    if (isa<pto::AllocTileOp, pto::AllocMultiTileOp, tensor::EmptyOp,
            pto::MultiTileGetOp>(defOp)) {
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
IRTranslator::getScalarMemoryOp(Operation *op, OperationBase *parentOp) {
  auto [reads, writes] = getReadWriteMemoryOps(op);
  if (reads.empty() && writes.empty())
    return nullptr;
  return std::make_unique<RWOperation>(
      op, parentOp, TCoreType::CUBE_OR_VECTOR, pto::PIPE::PIPE_S,
      pto::PIPE::PIPE_S, reads, writes);
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
IRTranslator::getCallOp(func::CallOp callOp, OperationBase *parentOp) {
  return nullptr;
}

void IRTranslator::updateBlockArgAliases(Block *block,
                                         OperandRange destOperands) {
  if (block->getNumArguments() != destOperands.size())
    return;
  for (auto [arg, operand] : llvm::zip(block->getArguments(), destOperands))
    blockArgAliases[arg].push_back(operand);
}

bool IRTranslator::isUnlikelyCondition(Condition *condOp) {
  return condOp && condOp->op &&
         condOp->op->hasAttrOfType<UnitAttr>("pto.unlikely_condition");
}

bool IRTranslator::isParallelLoop(Loop *loopOp) {
  return loopOp && loopOp->op &&
         loopOp->op->hasAttrOfType<UnitAttr>("pto.parallel_loop");
}

std::optional<int64_t>
IRTranslator::getLoopMultibufferUnrollNum(Loop *loopOp) {
  return {};
}

std::optional<int64_t> IRTranslator::getScopePreloadNum(Scope *scopeOp) {
  return {};
}

std::optional<int64_t> IRTranslator::getScopeMaxPreloadNum(Scope *scopeOp) {
  return {};
}

std::unique_ptr<Scope> IRTranslator::funcIrBuilder(Region &region,
                                                   OperationBase *parentOp,
                                                   bool skipEmptyScopes) {
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = parentOp;
  bool isFunctionRegion = isa_and_present<Function>(parentOp);
  if (!isFunctionRegion && region.getBlocks().size() > 1)
    return scopeOp;

  translateRegionIntoScope(region, scopeOp.get(), skipEmptyScopes,
                           isFunctionRegion);
  return scopeOp;
}

void IRTranslator::translateRegionIntoScope(Region &region, Scope *scopeOp,
                                            bool skipEmptyScopes,
                                            bool createFunctionBlocks) {
  for (Block &block : region.getBlocks()) {
    Scope *parScope = scopeOp;
    if (createFunctionBlocks) {
      auto blockOp = std::make_unique<FunctionBlock>();
      blockOp->parentOp = scopeOp;
      parScope = blockOp.get();
      scopeOp->body.push_back(std::move(blockOp));
    }

    auto blockBegin = std::make_unique<PlaceHolder>(nullptr, parScope);
    blockBegin->scopeBegin = parScope;
    blockBegin->block = &block;
    parScope->body.push_back(std::move(blockBegin));

    translateBlockIntoScope(block, parScope, skipEmptyScopes);

    auto blockEnd = std::make_unique<PlaceHolder>(nullptr, parScope);
    blockEnd->scopeEnd = parScope;
    blockEnd->block = &block;
    parScope->body.push_back(std::move(blockEnd));
  }
}

void IRTranslator::translateBlockIntoScope(Block &block, Scope *parScope,
                                           bool skipEmptyScopes) {
  for (Operation &op : block.getOperations()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      auto trueScope =
          funcIrBuilder(ifOp.getThenRegion(), nullptr, skipEmptyScopes);
      std::unique_ptr<Scope> falseScope;
      if (ifOp.elseBlock())
        falseScope =
            funcIrBuilder(ifOp.getElseRegion(), nullptr, skipEmptyScopes);
      auto cond = std::make_unique<Condition>(
          &op, parScope, std::move(trueScope), std::move(falseScope));
      cond->isUnlikely = isUnlikelyCondition(cond.get());
      if (!skipEmptyScopes || !isEmptyScope(cond.get()))
        parScope->body.push_back(std::move(cond));
      continue;
    }

    if (isa<LoopLikeOpInterface>(op)) {
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
      continue;
    }

    if (isTransparentGraphSyncRegionOp(op)) {
      for (Region &nested : op.getRegions())
        translateRegionIntoScope(nested, parScope, skipEmptyScopes);
      continue;
    }

    if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
      updateBlockArgAliases(branchOp.getDest(), branchOp.getDestOperands());
      continue;
    }
    if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
      updateBlockArgAliases(condBranchOp.getTrueDest(),
                            condBranchOp.getTrueDestOperands());
      updateBlockArgAliases(condBranchOp.getFalseDest(),
                            condBranchOp.getFalseDestOperands());
      continue;
    }

    if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
      if (auto rw = getPipeInterfaceOp(pipeOp, parScope))
        parScope->body.push_back(std::move(rw));
    } else if (isa<pto::LoadScalarOp, pto::StoreScalarOp>(op)) {
      if (auto rw = getScalarMemoryOp(&op, parScope))
        parScope->body.push_back(std::move(rw));
    } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      if (auto rw = getTensorExtractOp(extractOp, parScope))
        parScope->body.push_back(std::move(rw));
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      if (auto rw = getCallOp(callOp, parScope))
        parScope->body.push_back(std::move(rw));
    }
  }
}

bool IRTranslator::skipLaterIterations(Occurrence *occ1, Occurrence *occ2) {
  auto skip = [](Occurrence *occ, Occurrence *other) {
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

void IRTranslator::generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                            bool isUseless) {
  generateProcessingOrders(occ->childOccs, isUseless);
}

void IRTranslator::generateProcessingOrders(Loop *loopOp, Occurrence *occ,
                                            bool isUseless) {
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
