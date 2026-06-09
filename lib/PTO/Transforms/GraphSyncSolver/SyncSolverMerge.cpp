// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===--------- SyncSolver.cpp ------- Graph Sync Solver -------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/GraphSolver.h"
#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <tuple>
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>

#define DEBUG_TYPE "PTO-gss-solver"

using namespace mlir;
using namespace pto::syncsolver;

using BackwardSyncEventKey = CorePipeEventKey;

static llvm::SmallVector<pto::TCoreType> getBackwardSyncCoreTypes(
    const SyncSolverOptions &options) {
  if (options.isCrossCoreMode()) {
    return {pto::TCoreType::VECTOR, pto::TCoreType::CUBE};
  }
  return {pto::TCoreType::CUBE_OR_VECTOR};
}

bool Solver::tryCollectMergeableBackwardSyncEvent(
    Scope *scopeOp, CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
    int64_t eventId, CorePipeEventDenseSet &toBeErased) {
  if (checkBackwardSyncEventsContains(scopeOp, corePipeSrc, corePipeDst, eventId))
    return false;
  if (!checkMergeable(scopeOp, corePipeSrc, corePipeDst, eventId))
    return false;
  toBeErased.insert({corePipeSrc, corePipeDst, eventId});
  backwardSyncEvents[scopeOp][{corePipeSrc, corePipeDst}].insert({eventId, 1});
  return true;
}

void Solver::calcAllEventIds() {
  for (auto &[pipes, eventIdSolver] : eventIdSolver) {
    ASSERT(eventIdSolver != nullptr);

    [[maybe_unused]] auto result =
        eventIdSolver->shrinkEventIdMaxToEventIdNum();
    ASSERT(llvm::succeeded(result));
    ASSERT(eventIdSolver->isColorable());
  }
}

void Solver::collectBackwardSyncEventIds() {
  LLVM_DEBUG(llvm::dbgs() << "collectBackwardSyncEventIds\n";);
  for (auto &conflictPair : chosenConflictedPairs) {
    if (!conflictPair->isUseless && conflictPair->isInnerBackward &&
        conflictPair->eventIdNode != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "  " << conflictPair->str() << "\n";);
      for (auto eventId : conflictPair->eventIdNode->getEventIds()) {
        auto &e = backwardSyncEvents[conflictPair->backwardSyncLoopOp]
                                    [{conflictPair->setCorePipeInfo,
                                      conflictPair->waitCorePipeInfo}][eventId];
        e = std::max(e, conflictPair->eventIdInfo.eventIdRepeatNum);
      }
    }
  }
}

void Solver::resetAndBuildSetWaitOpIndex(const SyncMap &syncMapBefore,
                                         const SyncMap &syncMapAfter) {
  globalSetWaitIndex = 0;
  setWaitStartIndex.clear();
  setWaitEndIndex.clear();
  setWaitStartIndexInclusive.clear();
  setWaitEndIndexInclusive.clear();
  setWaitFlagOpsIndex.clear();
  collectSetWaitOpsIndexes(funcIr.get(), syncMapBefore, syncMapAfter);
}

std::set<std::pair<int64_t, SetWaitOp *>> &
Solver::getSetWaitOpsIndexRef(pto::PIPE pipeSrc, pto::PIPE pipeDst,
                              int64_t eventId) {
  auto key = std::make_tuple(pipeSrc, pipeDst, eventId);
  return setWaitFlagOpsIndex[key];
}

// Collect indices for all Set/Wait ops to facilitate merging decisions.
void Solver::collectSetWaitOpsIndexes(OperationBase *op,
                                      const SyncMap &syncMapBefore,
                                      const SyncMap &syncMapAfter) {
  ASSERT(op != nullptr);
  auto collectSyncOpIndexesForMap =
      [this, op](const SyncMap &syncMap) {
    if (syncMap.count(op) == 0)
      return;
    auto *it = syncMap.find(op);
    ASSERT(it != syncMap.end());
    for (auto &syncOp : it->second) {
      if (auto *setWaitOp = llvm::dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          auto &index = getSetWaitOpsIndexRef(setWaitOp->pipeSrc,
                                              setWaitOp->pipeDst, eventId);
          index.insert({globalSetWaitIndex++, setWaitOp});
        }
      }
    }
  };
  setWaitStartIndexInclusive[op] = globalSetWaitIndex++;
  collectSyncOpIndexesForMap(syncMapBefore);
  setWaitStartIndex[op] = globalSetWaitIndex++;
  if (auto *scopeOp = llvm::dyn_cast<Scope>(op)) {
    for (auto &childOp : scopeOp->body) {
      collectSetWaitOpsIndexes(childOp.get(), syncMapBefore, syncMapAfter);
    }
  }
  setWaitEndIndex[op] = globalSetWaitIndex++;
  collectSyncOpIndexesForMap(syncMapAfter);
  setWaitEndIndexInclusive[op] = globalSetWaitIndex++;
}

bool Solver::checkBackwardSyncEventsContains(OperationBase *op,
                                             CorePipeInfo corePipeSrc,
                                             CorePipeInfo corePipeDst,
                                             int64_t eventId) {
  auto *it1 = backwardSyncEvents.find(op);
  if (it1 == backwardSyncEvents.end()) {
    return false;
  }
  auto it2 = it1->second.find({corePipeSrc, corePipeDst});
  if (it2 == it1->second.end()) {
    return false;
  }
  return it2->second.contains(eventId);
}

bool Solver::checkBackwardSyncEventsContainsAfterMerge(
    OperationBase *op, CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst) {
  auto *it1 = backwardSyncEventsAfterMerge.find(op);
  if (it1 == backwardSyncEventsAfterMerge.end()) {
    return false;
  }
  return it1->second.contains({corePipeSrc, corePipeDst});
}

template <typename IndexT>
static bool isSetWaitIndexUsedInRange(const IndexT &index, int64_t startInclusive,
                                      int64_t endInclusive) {
  auto it = index.lower_bound({startInclusive, nullptr});
  return it != index.end() && it->first < endInclusive;
}

template <typename IndexT, typename ContainsFn, typename ContainsAfterFn>
static bool isBackwardSyncChildUsable(
    OperationBase *childOp, const IndexT &index, int64_t childStartInclusive,
    int64_t childEnd, int64_t childEndInclusive, ContainsFn containsFn,
    ContainsAfterFn containsAfterFn, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, int64_t eventId) {
  auto it1 = index.lower_bound({childStartInclusive, nullptr});
  auto it2 = index.lower_bound({childEnd, nullptr});
  bool usedAtleastOnce =
      it1 != index.end() && it1->first < childEndInclusive;
  if (!usedAtleastOnce)
    return true;

  bool before = it1 != index.end() && it1->first < childStartInclusive;
  bool after = it2 != index.end() && it2->first < childEndInclusive;
  if (before || after)
    return false;
  if (!containsFn(childOp, corePipeSrc, corePipeDst, eventId))
    return false;
  if (containsAfterFn(childOp, corePipeSrc, corePipeDst))
    return false;
  return true;
}

// Check whether a backward-sync event id can be merged at scope level.
bool Solver::checkMergeable(Scope *scopeOp, CorePipeInfo corePipeSrc,
                            CorePipeInfo corePipeDst, int64_t eventId,
                            bool shouldBeUsedAtleastOnce) {
  auto &index =
      getSetWaitOpsIndexRef(corePipeSrc.pipe, corePipeDst.pipe, eventId);
  if (shouldBeUsedAtleastOnce) {
    if (!isSetWaitIndexUsedInRange(index, setWaitStartIndexInclusive[scopeOp],
                                   setWaitEndIndexInclusive[scopeOp])) {
      return false;
    }
  }
  auto it1 = index.lower_bound({setWaitStartIndexInclusive[scopeOp], nullptr});
  auto it2 = index.lower_bound({setWaitEndIndex[scopeOp], nullptr});
  bool usedBefore = it1 != index.end() && it1->first < setWaitStartIndex[scopeOp];
  bool usedAfter = it2 != index.end() &&
                   it2->first < setWaitEndIndexInclusive[scopeOp];
  if (usedBefore || usedAfter) {
    return false;
  }
  if (auto *conditionOp = llvm::dyn_cast<Condition>(scopeOp)) {
    return checkMergeableConditionScopes(conditionOp, corePipeSrc, corePipeDst,
                                         eventId);
  }
  if (auto *loopOp = llvm::dyn_cast<Loop>(scopeOp)) {
    return checkMergeableLoopScopes(loopOp, corePipeSrc, corePipeDst, eventId);
  }
  return checkMergeableScopeBody(scopeOp, index, corePipeSrc, corePipeDst,
                                 eventId);
}

bool Solver::checkMergeableConditionScopes(const Condition *conditionOp,
                                           CorePipeInfo corePipeSrc,
                                           CorePipeInfo corePipeDst,
                                           int64_t eventId) {
  if (!conditionOp->hasFalseScope())
    return false;
  return checkMergeable(conditionOp->getTrueScope(), corePipeSrc, corePipeDst,
                        eventId, true) &&
         checkMergeable(conditionOp->getFalseScope(), corePipeSrc, corePipeDst,
                        eventId, true);
}

bool Solver::checkMergeableLoopScopes(const Loop *loopOp,
                                      CorePipeInfo corePipeSrc,
                                      CorePipeInfo corePipeDst,
                                      int64_t eventId) {
  for (auto &childOp : loopOp->body) {
    auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get());
    if (childScopeOp &&
        !checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId,
                        false)) {
      return false;
    }
  }
  for (auto &childOp : loopOp->body) {
    auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get());
    if (childScopeOp &&
        checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId, true)) {
      return true;
    }
  }
  return false;
}

bool Solver::checkMergeableScopeBody(
    const Scope *scopeOp,
    const std::set<std::pair<int64_t, SetWaitOp *>> &index,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, int64_t eventId) {
  for (auto &childOp : scopeOp->body) {
    if (!isBackwardSyncChildUsable(
            childOp.get(), index, setWaitStartIndexInclusive[childOp.get()],
            setWaitEndIndex[childOp.get()],
            setWaitEndIndexInclusive[childOp.get()],
            [this](OperationBase *targetOp, CorePipeInfo src, CorePipeInfo dst,
                   int64_t targetEventId) {
              return checkBackwardSyncEventsContains(targetOp, src, dst,
                                                    targetEventId);
            },
            [this](OperationBase *targetOp, CorePipeInfo src, CorePipeInfo dst) {
              return checkBackwardSyncEventsContainsAfterMerge(targetOp, src,
                                                               dst);
            },
            corePipeSrc, corePipeDst, eventId)) {
      return false;
    }
  }
  return true;
}

// Attempt to merge backward sync events across children and prune duplicates.
void Solver::mergeBackwardSyncEventIds(OperationBase *op) {
  auto *scopeOp = llvm::dyn_cast_if_present<Scope>(op);
  if (scopeOp == nullptr) {
    return;
  }
  for (auto &op : scopeOp->body) {
    mergeBackwardSyncEventIds(op.get());
  }

  if (shouldSkipBackwardSyncMerge(op)) {
    return;
  }
  CorePipeEventDenseSet toBeErased;
  collectMergeableBackwardSyncEvents(scopeOp, toBeErased);
  eraseMergedBackwardSyncEventsFromNestedScopes(scopeOp, toBeErased);
}

void Solver::mergeBackwardSyncPairs(SyncMap &syncMapBefore,
                                    SyncMap &syncMapAfter) {
  if (!options.moveOutAndMergeBackwardSyncPairs) {
    return;
  }
  if (options.isIntraCoreMode()) {
    resetAndBuildSetWaitOpIndex(syncMapBefore, syncMapAfter);
    auto *scopeOp = llvm::dyn_cast<Scope>(funcIr.get());
    ASSERT(scopeOp != nullptr && scopeOp->body.front() != nullptr);
    mergeBackwardSyncEventIds(scopeOp->body.front().get());
  }
}

bool Solver::shouldSkipBackwardSyncMerge(OperationBase *op) const {
  if (llvm::isa_and_present<FunctionBlock>(op))
    return true;
  if (llvm::isa_and_present<Condition, Loop>(op->parentOp))
    return true;
  auto *conditionOp = llvm::dyn_cast<Condition>(op);
  return conditionOp != nullptr && !conditionOp->hasFalseScope();
}

void Solver::collectMergeableBackwardSyncEvents(
    Scope *scopeOp, CorePipeEventDenseSet &toBeErased) {
  llvm::SmallVector<pto::TCoreType> coreTypes =
      getBackwardSyncCoreTypes(options);
  int64_t eventIdMax = getHWAvailableEventIdNum(options.syncMode);
  for (int64_t eventId = 0; eventId < eventIdMax; ++eventId) {
    collectMergeableBackwardSyncEventsForEventId(scopeOp, coreTypes, eventId,
                                                 toBeErased);
  }
}

void Solver::collectMergeableBackwardSyncEventsForEventId(
    Scope *scopeOp, ArrayRef<pto::TCoreType> coreTypes, int64_t eventId,
    CorePipeEventDenseSet &toBeErased) {
  for (auto coreSrc : coreTypes) {
    for (auto coreDst : coreTypes) {
      collectMergeableBackwardSyncEventsForCorePair(scopeOp, coreSrc, coreDst,
                                                    eventId, toBeErased);
    }
  }
}

void Solver::collectMergeableBackwardSyncEventsForCorePair(
    Scope *scopeOp, pto::TCoreType coreSrc, pto::TCoreType coreDst,
    int64_t eventId, CorePipeEventDenseSet &toBeErased) {
  size_t pipeNumMax = static_cast<size_t>(pto::PIPE::PIPE_NUM);
  for (size_t pipeSrcInt = 0; pipeSrcInt < pipeNumMax; ++pipeSrcInt) {
    for (size_t pipeDstInt = 0; pipeDstInt < pipeNumMax; ++pipeDstInt) {
      auto corePipeSrc = CorePipeInfo(coreSrc, static_cast<pto::PIPE>(pipeSrcInt));
      auto corePipeDst = CorePipeInfo(coreDst, static_cast<pto::PIPE>(pipeDstInt));
      (void)tryCollectMergeableBackwardSyncEvent(scopeOp, corePipeSrc, corePipeDst,
                                                 eventId, toBeErased);
    }
  }
}

void Solver::eraseMergedBackwardSyncEventsFromChildScope(
    Scope *childScopeOp, const CorePipeEventDenseSet &toBeErased) {
  for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
    if (!checkBackwardSyncEventsContains(childScopeOp, corePipeSrc, corePipeDst,
                                         eventId)) {
      continue;
    }
    auto key = std::make_tuple(corePipeSrc, corePipeDst);
    backwardSyncEvents[childScopeOp][key].erase(eventId);
    if (backwardSyncEvents[childScopeOp][key].empty())
      backwardSyncEvents[childScopeOp].erase(key);
  }
}

void Solver::eraseMergedBackwardSyncEventsFromNestedScopes(
    Scope *scopeOp, const CorePipeEventDenseSet &toBeErased) {
  if (isa<Condition, Loop>(scopeOp)) {
    for (auto &op : scopeOp->body) {
      auto *block = llvm::dyn_cast<Scope>(op.get());
      if (!block)
        continue;
      for (auto &childOp : block->body) {
        if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get()))
          eraseMergedBackwardSyncEventsFromChildScope(childScopeOp, toBeErased);
      }
    }
    return;
  }
  for (auto &childOp : scopeOp->body) {
    if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get()))
      eraseMergedBackwardSyncEventsFromChildScope(childScopeOp, toBeErased);
  }
}

SyncBeforeAfterMap Solver::getBeforeAfterSyncMaps() {
  calcAllEventIds();
  SyncMap syncMapBefore, syncMapAfter;
  std::vector<ConflictPair *> conflictPairs;
  for (auto &conflictPair : chosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }
  for (auto &conflictPair : persistentChosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }

  for (auto *conflictPair : conflictPairs) {
    appendSyncOpsForConflictPair(conflictPair, syncMapBefore, syncMapAfter);
  }

  collectBackwardSyncEventIds();
  mergeBackwardSyncPairs(syncMapBefore, syncMapAfter);
  appendMergedBackwardSyncScopeOps(syncMapBefore, syncMapAfter);
  return std::make_pair(std::move(syncMapBefore), std::move(syncMapAfter));
}

void Solver::appendSyncOpsForConflictPair(ConflictPair *conflictPair,
                                          SyncMap &syncMapBefore,
                                          SyncMap &syncMapAfter) {
  if (conflictPair->isUseless || conflictPair->replacedWithUnitFlag)
    return;
  ASSERT(conflictPair->setOp != nullptr && conflictPair->waitOp != nullptr);
  if (conflictPair->isBarrier()) {
    auto barrierOp = std::make_unique<BarrierOp>(conflictPair->waitOp->op,
                                                 conflictPair->waitOp->parentOp,
                                                 conflictPair->waitCorePipeInfo.pipe);
    LLVM_DEBUG(barrierOp->debugId = conflictPair->id);
    syncMapBefore[conflictPair->waitOp].push_back(std::move(barrierOp));
    return;
  }

  ASSERT(conflictPair->eventIdNode != nullptr);
  auto setOp = std::make_unique<SetFlagOp>(
      conflictPair->setOp->op, conflictPair->setOp->parentOp,
      conflictPair->eventIdNode->getEventIds(), conflictPair->setCorePipeInfo.pipe,
      conflictPair->waitCorePipeInfo.pipe);
  auto waitOp = std::make_unique<WaitFlagOp>(
      conflictPair->waitOp->op, conflictPair->waitOp->parentOp,
      conflictPair->eventIdNode->getEventIds(), conflictPair->setCorePipeInfo.pipe,
      conflictPair->waitCorePipeInfo.pipe);
  if (options.isCrossCoreMode()) {
    setOp->coreType = conflictPair->setCorePipeInfo.coreType;
    waitOp->coreType = conflictPair->waitCorePipeInfo.coreType;
  }
  setOp->eventIdInfo = conflictPair->eventIdInfo;
  waitOp->eventIdInfo = conflictPair->eventIdInfo;
  setOp->checkLastIter = conflictPair->setOnLastIterOnly;
  waitOp->checkFirstIter = conflictPair->waitOnFirstIterOnly;
  LLVM_DEBUG({
    setOp->debugId = conflictPair->id;
    waitOp->debugId = conflictPair->id;
  });
  syncMapAfter[conflictPair->setOp].push_back(std::move(setOp));
  syncMapBefore[conflictPair->waitOp].push_front(std::move(waitOp));
}

void Solver::appendMergedBackwardSyncScopeOps(SyncMap &syncMapBefore,
                                              SyncMap &syncMapAfter) {
  for (auto &[op, mp] : backwardSyncEvents) {
    if (mp.empty())
      continue;
    auto *scopeOp = llvm::dyn_cast<Scope>(op);
    ASSERT(scopeOp != nullptr);
    for (auto [setWaitCorePipes, eventIdsMp] : mp) {
      if (eventIdsMp.empty())
        continue;
      llvm::SmallVector<int64_t> eventIds;
      for (auto [eventId, repeatNum] : eventIdsMp) {
        llvm::SmallVector<int64_t> curEventIds(repeatNum, eventId);
        llvm::append_range(eventIds, curEventIds);
      }
      llvm::sort(eventIds);
      auto [corePipeSrc, corePipeDst] = setWaitCorePipes;
      auto setOp = std::make_unique<SetFlagOp>(scopeOp->op, scopeOp->parentOp,
                                               eventIds, corePipeSrc.pipe,
                                               corePipeDst.pipe);
      auto waitOp = std::make_unique<WaitFlagOp>(scopeOp->op, scopeOp->parentOp,
                                                 eventIds, corePipeSrc.pipe,
                                                 corePipeDst.pipe);
      setOp->allAtOnce = true;
      waitOp->allAtOnce = true;
      if (options.isCrossCoreMode()) {
        setOp->coreType = corePipeSrc.coreType;
        waitOp->coreType = corePipeDst.coreType;
      }
      syncMapBefore[scopeOp].push_back(std::move(setOp));
      syncMapAfter[scopeOp].push_front(std::move(waitOp));
    }
  }
}

void Solver::processConflict(Occurrence *occ1, Occurrence *occ2,
                             RWOperation *rwOp1, RWOperation *rwOp2,
                             bool isUseless) {
  for (auto [corePipeSrc, corePipeDst] : checkMemoryConflicts(rwOp1, rwOp2)) {
    if (options.alwaysUsePipeSAsWaitingPipe) {
      corePipeDst.pipe = pto::PIPE::PIPE_S;
    }
    auto eventIdInfo =
        getEventIdInfo(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
    handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                   eventIdInfo, isUseless);
  }
}

// Main processing loop that iterates processingOrders and attempts to
// discover and record conflicts.
void Solver::processOrders() {
  for (auto &[occ1, occ2, rwOp1, rwOp2, isUseless] : processingOrders) {
    ASSERT(occ1 != occ2);
    ASSERT(occ1->syncIrIndex < occ2->syncIrIndex);
    if (checkVisited(occ1, occ2)) {
      ASSERT(false && "expected to not check a pair more than once.");
      continue;
    }
    if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
        skipMMad1DecomposedLoopOpt(occ1, occ2) ||
        checkSkipParallelLoop(occ1, occ2) ||
        checkSkipCrossCorePair(occ1, occ2)) {
      continue;
    }
    DEBUG_WITH_TYPE("gss-sync-solver-checking", {
      llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
      llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                   << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
      llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                   << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
    });
    if (checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
      continue;
    }
    processConflict(occ1, occ2, rwOp1, rwOp2, isUseless);
  }
}

void Solver::insertMergedBackwardSyncPairs() {
  for (auto &[scopeOp, st] : backwardSyncEventsAfterMerge) {
    for (auto &corePipeInfoPair : st) {
      auto [corePipeSrc, corePipeDst] = corePipeInfoPair;
      for (auto *scopeOcc : opAllOccurrences[scopeOp]) {
        auto *parentScopeOcc = scopeOcc->parentOcc;
        ASSERT(parentScopeOcc != nullptr);
        Occurrence *setOcc = nullptr;
        Occurrence *waitOcc = nullptr;
        auto startIndex = scopeOcc->startIndex;
        auto endIndex = scopeOcc->endIndex;
        if (isa<Loop>(scopeOp)) {
          setOcc = getBeforePlaceHolderOcc(scopeOcc);
          waitOcc = getAfterPlaceHolderOcc(scopeOcc);
          startIndex = setOcc->endIndex;
          endIndex = waitOcc->startIndex;
        }
        auto conflictPair = std::make_unique<ConflictPair>(
            nullptr, nullptr, nullptr, nullptr, setOcc, waitOcc, corePipeSrc,
            corePipeDst, startIndex, endIndex);
        ASSERT(conflictPair->startIndex <= conflictPair->endIndex);
        conflictPair->isUseless = true;
        conflictPair->dontReuse = true;
        conflictPair->dontCheckForConflict = true;
        conflictPair->couldNotRun = false; // notice this
        LLVM_DEBUG({
          llvm::dbgs() << "consider-merged-backward-pair: "
                       << scopeOp->str(0, false) << ' ' << conflictPair->str()
                       << "\n";
        });
        scopeOccChosenConflicts[parentScopeOcc].insert(conflictPair.get());
        chosenConflictedPairs.push_back(std::move(conflictPair));
      }
    }
  }
}

llvm::LogicalResult Solver::considerOuterBackwardSyncPairs() {
  if (!options.considerOuterBackwardSyncPairs) {
    return llvm::failure();
  }
  bool backwardPairsPositionChanged = pruneStaleMergedBackwardSyncPairs();
  SmallVector<OperationBase *> chosenOps = collectDeepestUnmergedBackwardSyncScopes();
  if (chosenOps.empty()) {
    return llvm::failure();
  }
  bool newPairIsInserted = insertOuterBackwardSyncPairs(chosenOps);
  return llvm::success(backwardPairsPositionChanged || newPairIsInserted);
}

bool Solver::pruneStaleMergedBackwardSyncPairs() {
  bool backwardPairsPositionChanged = false;
  for (auto &[scopeOp, st] : backwardSyncEventsAfterMerge) {
    SmallVector<CorePipePairKey> toBeErased;
    for (auto &corePipeInfoPair : st) {
      if (!backwardSyncEvents.contains(scopeOp) ||
          !backwardSyncEvents[scopeOp].contains(corePipeInfoPair)) {
        toBeErased.push_back(corePipeInfoPair);
      }
    }
    if (toBeErased.empty())
      continue;
    backwardPairsPositionChanged = true;
    for (auto &corePipeInfoPair : toBeErased)
      st.erase(corePipeInfoPair);
  }
  return backwardPairsPositionChanged;
}

SmallVector<OperationBase *> Solver::collectDeepestUnmergedBackwardSyncScopes() {
  int chosenOpsDepth = -1;
  SmallVector<OperationBase *> chosenOps;
  for (auto &[scopeOp, mp] : backwardSyncEvents) {
    if (backwardSyncEventsAfterMerge.contains(scopeOp))
      continue;
    int scopeOpDepth = scopeOp->getDepth();
    if (chosenOpsDepth == scopeOpDepth) {
      chosenOps.push_back(scopeOp);
    } else if (chosenOpsDepth == -1 || chosenOpsDepth < scopeOpDepth) {
      chosenOps.clear();
      chosenOps.push_back(scopeOp);
      chosenOpsDepth = scopeOpDepth;
    }
  }
  return chosenOps;
}

bool Solver::insertOuterBackwardSyncPairs(
    const SmallVectorImpl<OperationBase *> &chosenOps) {
  bool newPairIsInserted = false;
  for (auto *chosenOp : chosenOps) {
    for (auto &[corePipeInfoPair, eventIdsMp] : backwardSyncEvents[chosenOp]) {
      ASSERT(!eventIdsMp.empty());
      if (eventIdsMp.empty())
        continue;
      auto [it, isInserted] =
          backwardSyncEventsAfterMerge[chosenOp].insert(corePipeInfoPair);
      (void)it;
      newPairIsInserted = newPairIsInserted || isInserted;
    }
  }
  return newPairIsInserted;
}

llvm::LogicalResult Solver::reuseSyncPairToSaveEventIds() {
  if (!options.reuseSyncPairToSaveEventIds || barrierAllPairs.empty()) {
    return llvm::failure();
  }
  bool limitReached = true;
  for (auto [corePipeSrc, corePipeDst] : barrierAllPairs) {
    if (reusePairs[{corePipeSrc, corePipeDst}] < maxReuseNum) {
      if (reusePairs[{corePipeSrc, corePipeDst}] <=
          reusedPairs[{corePipeSrc, corePipeDst}]) {
        reusePairs[{corePipeSrc, corePipeDst}] += 1;
        limitReached = false;
      }
    }
  }
  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    llvm::dbgs() << "reusePairs: \n";
    for (auto [pipeCorePairs, cnt] : reusePairs) {
      llvm::dbgs() << get<0>(pipeCorePairs).pipe << ' '
                   << get<1>(pipeCorePairs).pipe << ' ' << cnt << '\n';
    }
  });
  return llvm::success(!limitReached);
}

llvm::LogicalResult Solver::disableMultiEventIdForBarrierAllPairs() {
  if (!options.disableMultiEventIdForBarrierAllPairs ||
      barrierAllPairs.empty()) {
    return llvm::failure();
  }
  bool newPairIsInserted = false;
  for (auto corePipeInfoPair : barrierAllPairs) {
    auto [it, isInserted] = disabledMultiEventIdPairs.insert(corePipeInfoPair);
    newPairIsInserted = newPairIsInserted || isInserted;
  }
  LLVM_DEBUG({
    if (newPairIsInserted) {
      llvm::dbgs() << "disabled-multi-event-id-pairs: \n";
      for (auto &[corePipeSrc, corePipeDst] : disabledMultiEventIdPairs) {
        llvm::dbgs() << corePipeSrc.coreType << ' ' << corePipeSrc.pipe << ' '
                     << corePipeDst.coreType << ' ' << corePipeDst.pipe << '\n';
      }
    }
  });
  return llvm::success(newPairIsInserted);
}

llvm::LogicalResult Solver::tryMovingOutBackwardSyncPairsToOuterLoops() {
  if (!options.moveOutAndMergeBackwardSyncPairs || !options.isCrossCoreMode() ||
      dontMoveBackwardSyncPairsToOutmostLoop) {
    return llvm::failure();
  }
  if (!moveBackwardSyncPairsToOutmostLoop) {
    moveBackwardSyncPairsToOutmostLoop = true;
    return llvm::success();
  }
  if (!barrierAllPairs.empty()) {
    moveBackwardSyncPairsToOutmostLoop = false;
    dontMoveBackwardSyncPairsToOutmostLoop = true;
    return llvm::success();
  }
  return llvm::failure();
}

// High-level solve orchestration with multiple passes and optional merging
// iterations.
llvm::LogicalResult Solver::runSolver(bool enableOpts1, bool enableOpts2) {
  reset(/*resetEventIdRanOutOpts=*/true);

  int64_t runNum = 0;
  while (runNum++ < maxRunNum) {
    LLVM_DEBUG(llvm::dbgs() << "runNum: " << runNum << '\n');

    reset();
    insertMergedBackwardSyncPairs();
    processOrders();

    if (llvm::succeeded(tryMovingOutBackwardSyncPairsToOuterLoops())) {
      continue;
    }

    if (enableOpts1) {
      if (options.considerOuterBackwardSyncPairs) {
        getBeforeAfterSyncMaps();
        if (llvm::succeeded(considerOuterBackwardSyncPairs())) {
          continue;
        }
        if (!barrierAllPairs.empty()) {
          backwardSyncEventsAfterMerge.clear();
        }
      }
    }

    if (enableOpts2) {
      if (!barrierAllPairs.empty()) {
        if (llvm::succeeded(reuseSyncPairToSaveEventIds())) {
          continue;
        }
        if (llvm::succeeded(disableMultiEventIdForBarrierAllPairs())) {
          continue;
        }
      }
    }

    if (!barrierAllPairs.empty()) {
      pickAndInsertABarrierAll();
      reset(/*resetEventIdRanOutOpts=*/true);
      continue;
    }
    break;
  }

  reset();
  insertMergedBackwardSyncPairs();
  processOrders();

  return llvm::success(runNum < maxRunNum);
}

void Solver::solve() {
  if (llvm::succeeded(runSolver())) {
    return;
  }
  if (!options.isTestMode()) {
    if (llvm::succeeded(runSolver(/*enableOpts1=*/false))) {
      return;
    }
    if (llvm::succeeded(
            runSolver(/*enableOpts1=*/false, /*enableOpts2=*/false))) {
      return;
    }
  }
  llvm_unreachable("GSS: runSolver() failed.");
}
