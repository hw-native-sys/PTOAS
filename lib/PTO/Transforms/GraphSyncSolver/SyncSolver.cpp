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
#include "PTO/Transforms/GraphSyncSolver/GraphSolver.h"
#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "PTO-gss-solver"

namespace {
constexpr int kCrossCoreParentDepthOffset = 2;
constexpr size_t kConditionScopeParentPairSize = 2;

[[noreturn]] void reportSyncSolverInvariantFailure(const char *expr,
                                                  const char *func,
                                                  int line) {
  llvm::report_fatal_error(llvm::Twine("SyncSolver invariant failed at ") +
                           func + ":" + llvm::Twine(line) + ": " + expr);
}
} // namespace

#define PTO_SYNC_SOLVER_CHECK(cond)                                            \
  do {                                                                         \
    if (!(cond))                                                               \
      reportSyncSolverInvariantFailure(#cond, __func__, __LINE__);             \
  } while (false)

using namespace mlir;
using namespace pto::syncsolver;

// Reset per-pass bookkeeping to start fresh.
void Solver::reset(bool resetEventIdRanOutOpts) {
  if (resetEventIdRanOutOpts) {
    reusePairs.clear();
    disabledMultiEventIdPairs.clear();
    backwardSyncEventsAfterMerge.clear();
    moveBackwardSyncPairsToOutmostLoop = false;
    dontMoveBackwardSyncPairsToOutmostLoop = false;
  }
  skipOcc.clear();
  syncedPairs.clear();
  processedOccPairs.clear();
  chosenConflictedPairs.clear();
  scopeOccChosenConflicts.clear();
  scopeOccPairChosenConflicts.clear();
  backwardSyncEvents.clear();
  replacedWithReusableSyncedPairs.clear();
  reusedPairs.clear();
  barrierAllPairs.clear();
  insertedBarrierAllBefore.clear();
  eventIdSolver.clear();
  resetUnitFlag();
}

void Solver::resetUnitFlag() {
  for (auto *rwOp : unitFlagFeaturedOps) {
    rwOp->mergedUnitFlagInfo.reset();
    for (auto *occ : opAllOccurrences[rwOp]) {
      occ->unitFlagInfo.reset();
    }
  }
}

// Helpers to find first/last iteration occurrences relative to parent
// occurrences.
Occurrence *Solver::getFirstIterOcc(Occurrence *occ, Occurrence *parOcc) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getFirstIterOcc(
        Occurrence::getNthParent(occ, occ->depth - parOcc->depth - 1),
        parOcc);
    return getFirstIterOcc(occ, newParOcc);
  }
  auto *it =
      std::find_if(parOcc->childOccs.begin(), parOcc->childOccs.end(),
                   [occ](const Occurrence *curOcc) {
                     return occ->op == curOcc->op;
                   });
  PTO_SYNC_SOLVER_CHECK(it != parOcc->childOccs.end());
  return *it;
}

Occurrence *Solver::getLastIterOcc(Occurrence *occ, Occurrence *parOcc) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getLastIterOcc(
        Occurrence::getNthParent(occ, occ->depth - parOcc->depth - 1),
        parOcc);
    return getLastIterOcc(occ, newParOcc);
  }
  auto it =
      std::find_if(parOcc->childOccs.rbegin(), parOcc->childOccs.rend(),
                   [occ](const Occurrence *curOcc) {
                     return occ->op == curOcc->op;
                   });
  PTO_SYNC_SOLVER_CHECK(it != parOcc->childOccs.rend());
  return *it;
}

bool Solver::checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2) {
  if (!options.isCrossCoreMode()) {
    return false;
  }
  auto *rwOp1 = llvm::dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast<RWOperation>(occ2->op);
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(rwOp1->coreType != pto::TCoreType::CUBE_OR_VECTOR);
  PTO_SYNC_SOLVER_CHECK(rwOp2->coreType != pto::TCoreType::CUBE_OR_VECTOR);
  if (rwOp1->coreType == rwOp2->coreType) {
    return true;
  }
  if (rwOp1->coreType == pto::TCoreType::CUBE_AND_VECTOR) {
    return true;
  }
  return false;
}

bool Solver::checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2) const {
  if (!isBackwardSync(occ1, occ2)) {
    return false;
  }
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  PTO_SYNC_SOLVER_CHECK(parOcc1 != nullptr && parOcc2 != nullptr);
  auto *parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
  PTO_SYNC_SOLVER_CHECK(parentLCALoopOcc != nullptr);
  auto *parentLCALoopOp = llvm::cast<Loop>(parentLCALoopOcc->op);
  return parentLCALoopOp->isParallel;
}

// Check whether occurrences belong to impossible (if-else) pairing.
bool Solver::checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2) const {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  if (occ1->op == occ2->op) {
    return false;
  }
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  PTO_SYNC_SOLVER_CHECK(parOcc1 != nullptr && parOcc2 != nullptr);
  bool isIfElseSituation =
      parOcc1->parentOcc != nullptr &&
      parOcc1->parentOcc == parOcc2->parentOcc &&
      llvm::isa_and_present<Condition>(parOcc1->parentOcc->op);
  return isIfElseSituation;
}

// Detect whether occ1 and occ2 have already been covered by an earlier sync.
bool Solver::checkAlreadySynced(Occurrence *occ1, Occurrence *occ2) const {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(occ1->op != nullptr && occ2->op != nullptr);

  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  PTO_SYNC_SOLVER_CHECK(parOcc1 != nullptr && parOcc2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(parOcc1->parentOcc != nullptr && parOcc2->parentOcc != nullptr);

  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  PTO_SYNC_SOLVER_CHECK(parOp1 != nullptr && parOp2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);

  auto *parentLoop = OperationBase::getParentloop(parOcc1->op);
  auto *curLoop = OperationBase::getParentloop(parOp1);
  if (parentLoop == nullptr || parentLoop == curLoop) {
    return false;
  }

  PTO_SYNC_SOLVER_CHECK(curLoop != nullptr);
  PTO_SYNC_SOLVER_CHECK(parentLoop->isProperAncestor(curLoop));
  while (curLoop != parentLoop) {
    if (!llvm::cast<Loop>(curLoop)->isParallel) {
      return true;
    }
    curLoop = OperationBase::getParentloop(curLoop);
    PTO_SYNC_SOLVER_CHECK(curLoop != nullptr);
  }
  return false;
}

// Unit-flag reuse check between two RWOperations.
bool Solver::checkAlreadySyncedWithUnitFlag(Occurrence *occ1,
                                            const Occurrence *occ2) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  if (!options.enableUnitFlagFeature) {
    return false;
  }
  if (!occ1->hasUnitFlagFeat || !occ2->hasUnitFlagFeat) {
    return false;
  }
  llvm::DenseSet<Occurrence *> visited;
  DEBUG_WITH_TYPE("gss-sync-solver-check-unit-flag", {
    llvm::dbgs() << "unit-flag-step: " << occ1->syncIrIndex << ' '
                 << occ1->op->str(0, false) << "\n";
  });
  Occurrence *curOcc = occ1->unitFlagInfo.linkedElementAsSet;
  while (curOcc != nullptr) {
    DEBUG_WITH_TYPE("gss-sync-solver-check-unit-flag", {
      llvm::dbgs() << "unit-flag-step: " << curOcc->syncIrIndex << ' '
                   << curOcc->op->str(0, false) << "\n";
    });
    auto [it, isInserted] = visited.insert(curOcc);
    if (!isInserted) {
      break;
    }
    if (curOcc == occ2) {
      return true;
    }
    curOcc = curOcc->unitFlagInfo.linkedElementAsSet;
  }
  return false;
}

bool Solver::ignoreMemoryConflict(const RWOperation *rwOp1,
                                  const RWOperation *rwOp2,
                                  const MemInfo &memInfo1,
                                  const MemInfo &memInfo2) {
  (void)rwOp1;
  (void)rwOp2;
  if (options.isIntraCoreMode()) {
    if (memInfo1.isWorkSpace && memInfo2.isWorkSpace) {
      if (options.intraCoreIgnoreWorkSpaceFunctionArguments) {
        return true;
      }
    }
  }
  return false;
}

bool Solver::checkMemInfoConflict(RWOperation *rwOp1, RWOperation *rwOp2,
                                  const MemInfo &memInfo1,
                                  const MemInfo &memInfo2,
                                  std::optional<int64_t> lcmLen,
                                  std::optional<int64_t> eventIdNum) {
  if (ignoreMemoryConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
    return false;
  }
  return MemInfo::checkConflict(memInfo1, memInfo2, lcmLen, eventIdNum);
}

bool Solver::checkMemInfoConflict(
    RWOperation *rwOp1, RWOperation *rwOp2,
    const llvm::SmallVector<MemInfo> &memInfoList1,
    const llvm::SmallVector<MemInfo> &memInfoList2,
    std::optional<int64_t> lcmLen, std::optional<int64_t> eventIdNum) {
  for (auto &memInfo1 : memInfoList1) {
    for (auto &memInfo2 : memInfoList2) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2, lcmLen,
                               eventIdNum)) {
        return true;
      }
    }
  }
  return false;
}

// High-level wrapper computing pipe pairs that represent memory conflicts
// between two RW ops.
llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>>
Solver::checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2) {
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  auto [it, isInserted] = checkMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!isInserted) {
    return it->second;
  }
  auto coreSrc = rwOp1->coreType;
  auto coreDst = rwOp2->coreType;
  if (options.isCrossCoreMode()) {
    if (coreDst == pto::TCoreType::CUBE_AND_VECTOR) {
      coreDst = (coreSrc == pto::TCoreType::VECTOR) ? pto::TCoreType::CUBE
                                                     : pto::TCoreType::VECTOR;
    }
    PTO_SYNC_SOLVER_CHECK(coreSrc == pto::TCoreType::VECTOR ||
           coreSrc == pto::TCoreType::CUBE);
    PTO_SYNC_SOLVER_CHECK(coreDst == pto::TCoreType::VECTOR ||
           coreDst == pto::TCoreType::CUBE);
  }
  CorePipePairDenseSet collectedConflictsSet;
  llvm::SmallVector<CorePipePairKey> collectedConflicts;
  auto addCollectedConflict = [&collectedConflictsSet, &collectedConflicts](
                                  CorePipeInfo src, CorePipeInfo dst) {
    CorePipePairKey key{src, dst};
    if (collectedConflictsSet.insert(key).second) {
      collectedConflicts.push_back(key);
    }
  };
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->readMemInfo,
                           rwOp2->writeMemInfo)) {
    addCollectedConflict(CorePipeInfo(coreSrc, rwOp1->pipeRead),
                         CorePipeInfo(coreDst, rwOp2->pipeWrite));
  }
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                           rwOp2->readMemInfo)) {
    addCollectedConflict(CorePipeInfo(coreSrc, rwOp1->pipeWrite),
                         CorePipeInfo(coreDst, rwOp2->pipeRead));
  }
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                           rwOp2->writeMemInfo)) {
    addCollectedConflict(CorePipeInfo(coreSrc, rwOp1->pipeWrite),
                         CorePipeInfo(coreDst, rwOp2->pipeWrite));
  }
  return it->second = collectedConflicts;
}

bool Solver::checkMemoryConflictBetweenOccExclusive(
    Occurrence *occ1, Occurrence *occ2,
    std::function<bool(RWOperation *)> filter) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  for (int i = occ1->syncIrEndIndex; i < occ2->syncIrIndex; i++) {
    if (auto *otherOp = llvm::dyn_cast_if_present<RWOperation>(syncIr[i]->op)) {
      if (!filter(otherOp)) {
        continue;
      }
      if (!checkMemoryConflicts(rwOp1, otherOp).empty()) {
        return true;
      }
      if (!checkMemoryConflicts(rwOp2, otherOp).empty()) {
        return true;
      }
    }
  }
  return false;
}

std::optional<LoopLikeOpInterface>
Solver::getMultiBufferLoop(RWOperation *rwOp1, RWOperation *rwOp2,
                           const llvm::SmallVector<MemInfo> &memInfoList1,
                           const llvm::SmallVector<MemInfo> &memInfoList2) {
  std::optional<LoopLikeOpInterface> multibufferLoop;
  for (auto &memInfo1 : memInfoList1) {
    for (auto &memInfo2 : memInfoList2) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
        if (!memInfo1.pointerLikeInfo.has_value() ||
            !memInfo2.pointerLikeInfo.has_value()) {
          return {};
        }
        auto multibufferLoop1 = memInfo1.pointerLikeInfo->parentLoop;
        auto multibufferLoop2 = memInfo2.pointerLikeInfo->parentLoop;
        if (multibufferLoop1 == nullptr ||
            multibufferLoop1 != multibufferLoop2) {
          return {};
        }
        if (multibufferLoop.has_value() &&
            multibufferLoop.value() != multibufferLoop1) {
          return {};
        }
        multibufferLoop = multibufferLoop1;
      }
    }
  }
  return multibufferLoop;
}

std::optional<LoopLikeOpInterface>
Solver::getMultiBufferLoop(RWOperation *rwOp1, RWOperation *rwOp2) {
  std::optional<LoopLikeOpInterface> multibufferLoop;
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->readMemInfo,
                           rwOp2->writeMemInfo)) {
    auto curMultibufferLoop = getMultiBufferLoop(
        rwOp1, rwOp2, rwOp1->readMemInfo, rwOp2->writeMemInfo);
    if (multibufferLoop.has_value() &&
        multibufferLoop.value() != curMultibufferLoop) {
      return {};
    }
    multibufferLoop = curMultibufferLoop;
  }
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                           rwOp2->readMemInfo)) {
    auto curMultibufferLoop = getMultiBufferLoop(
        rwOp1, rwOp2, rwOp1->writeMemInfo, rwOp2->readMemInfo);
    if (multibufferLoop.has_value() &&
        multibufferLoop.value() != curMultibufferLoop) {
      return {};
    }
    multibufferLoop = curMultibufferLoop;
  }
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                           rwOp2->writeMemInfo)) {
    auto curMultibufferLoop = getMultiBufferLoop(
        rwOp1, rwOp2, rwOp1->writeMemInfo, rwOp2->writeMemInfo);
    if (multibufferLoop.has_value() &&
        multibufferLoop.value() != curMultibufferLoop) {
      return {};
    }
    multibufferLoop = curMultibufferLoop;
  }
  return multibufferLoop;
}

bool Solver::validateMultiBufferScope(Occurrence *occ1, Occurrence *occ2,
                                      RWOperation *rwOp1, RWOperation *rwOp2,
                                      LoopLikeOpInterface &multibufferLoop) {
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (options.isTestMode()) {
    auto *parLoop1 = occ1->getParentOfType<Loop>();
    auto *parLoop2 = occ2->getParentOfType<Loop>();
    if (!parLoop1 || parLoop1 != parLoop2)
      return false;
    return parLoop1->isProperAncestor(setOcc) &&
           parLoop1->isProperAncestor(waitOcc);
  }
  auto multibufferLoopOpt = getMultiBufferLoop(rwOp1, rwOp2);
  if (!multibufferLoopOpt.has_value() || !multibufferLoopOpt.value())
    return false;
  multibufferLoop = multibufferLoopOpt.value();
  PTO_SYNC_SOLVER_CHECK(multibufferLoop != nullptr);
  return Occurrence::getParentWithOp(setOcc, multibufferLoop,
                                     /*assertExists=*/false) &&
         Occurrence::getParentWithOp(waitOcc, multibufferLoop,
                                     /*assertExists=*/false);
}

void Solver::updateMultiBufferConflictStats(
    RWOperation *rwOp1, RWOperation *rwOp2,
    const llvm::SmallVector<MemInfo> &lhs, const llvm::SmallVector<MemInfo> &rhs,
    bool includeMemInfo1Min, bool includeMemInfo2Min, int64_t &lcm,
    int64_t &minWriteSize) {
  for (auto &memInfo1 : lhs) {
    for (auto &memInfo2 : rhs) {
      if (!checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2))
        continue;
      lcm = std::lcm(lcm, std::lcm(memInfo1.getSz(), memInfo2.getSz()));
      if (includeMemInfo1Min)
        minWriteSize = std::min(minWriteSize, memInfo1.getSz());
      if (includeMemInfo2Min)
        minWriteSize = std::min(minWriteSize, memInfo2.getSz());
    }
  }
}

std::optional<int64_t>
Solver::findMultiBufferEventIdNum(RWOperation *rwOp1, RWOperation *rwOp2,
                                  int64_t lcm, int64_t minWriteSize) {
  if (minWriteSize == LONG_MAX)
    return {};
  for (int64_t eventIdNum = minWriteSize; eventIdNum >= 1; --eventIdNum) {
    int64_t curLcm = std::lcm(lcm, eventIdNum);
    bool okRW = !checkMemInfoConflict(rwOp1, rwOp2, rwOp1->readMemInfo,
                                      rwOp2->writeMemInfo, curLcm, eventIdNum);
    bool okWR = !checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                                      rwOp2->readMemInfo, curLcm, eventIdNum);
    bool okWW = !checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                                      rwOp2->writeMemInfo, curLcm, eventIdNum);
    if (okRW && okWR && okWW)
      return eventIdNum;
  }
  return {};
}

std::optional<EventIdInfo>
Solver::getMultiBufferEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                  RWOperation *rwOp1, RWOperation *rwOp2) {
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);

  int64_t lcm = 1;
  int64_t minWriteSize = LONG_MAX;
  LoopLikeOpInterface multibufferLoop{nullptr};
  if (!validateMultiBufferScope(occ1, occ2, rwOp1, rwOp2, multibufferLoop))
    return {};

  updateMultiBufferConflictStats(rwOp1, rwOp2, rwOp1->readMemInfo,
                                 rwOp2->writeMemInfo, false, true, lcm,
                                 minWriteSize);
  updateMultiBufferConflictStats(rwOp1, rwOp2, rwOp1->writeMemInfo,
                                 rwOp2->readMemInfo, true, false, lcm,
                                 minWriteSize);
  updateMultiBufferConflictStats(rwOp1, rwOp2, rwOp1->writeMemInfo,
                                 rwOp2->writeMemInfo, true, true, lcm,
                                 minWriteSize);

  auto eventIdNum = findMultiBufferEventIdNum(rwOp1, rwOp2, lcm, minWriteSize);
  if (!eventIdNum || *eventIdNum <= 1)
    return {};
  EventIdInfo eventIdInfo(*eventIdNum);
  eventIdInfo.multibufferLoop = multibufferLoop;
  return eventIdInfo;
}

std::optional<EventIdInfo>
Solver::checkMultiBufferEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                    RWOperation *rwOp1, RWOperation *rwOp2) {
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isTestMode()) {
    if (!checkAllParentLoopsAreForLoops(rwOp1->op) ||
        !checkAllParentLoopsAreForLoops(rwOp2->op)) {
      return {};
    }
  }
  if (auto eventIdInfo = getMultiBufferEventIdInfo(occ1, occ2, rwOp1, rwOp2)) {
    return eventIdInfo;
  }
  return {};
}

std::optional<EventIdInfo>
Solver::checkCVMultiBufferUnrollEventIdInfo(RWOperation *rwOp1,
                                            RWOperation *rwOp2) {
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isCrossCoreMode()) {
    return {};
  }
  auto *parentLoop1 = rwOp1->getParentOfType<Loop>();
  auto *parentLoop2 = rwOp2->getParentOfType<Loop>();
  while (parentLoop1 != nullptr && !parentLoop1->multibufferUnrollNum) {
    parentLoop1 = parentLoop1->getParentOfType<Loop>();
  }
  while (parentLoop2 != nullptr && !parentLoop2->multibufferUnrollNum) {
    parentLoop2 = parentLoop2->getParentOfType<Loop>();
  }
  if (!parentLoop1 || !parentLoop2) {
    return {};
  }
  if (auto *parCond1 = rwOp1->getParentOfType<Condition>()) {
    if (!parCond1->isProperAncestor(rwOp2)) {
      return {};
    }
  }
  if (auto *parCond2 = rwOp2->getParentOfType<Condition>()) {
    if (!parCond2->isProperAncestor(rwOp1)) {
      return {};
    }
  }
  PTO_SYNC_SOLVER_CHECK(parentLoop1->multibufferUnrollNum.value() ==
         parentLoop2->multibufferUnrollNum.value());
  EventIdInfo eventIdInfo;
  eventIdInfo.eventIdNum = parentLoop1->multibufferUnrollNum.value();
  eventIdInfo.multibufferUnrollLoop1 =
      cast<LoopLikeOpInterface>(parentLoop1->op);
  eventIdInfo.multibufferUnrollLoop2 =
      cast<LoopLikeOpInterface>(parentLoop2->op);
  return eventIdInfo;
}

std::optional<EventIdInfo>
Solver::checkCVMultiBufferPreloadEventIdInfo(RWOperation *rwOp1,
                                             RWOperation *rwOp2) {
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isCrossCoreMode()) {
    return {};
  }
  auto *parentScope1 = rwOp1->getParentOfType<Scope>();
  auto *parentScope2 = rwOp2->getParentOfType<Scope>();
  while (parentScope1 != nullptr && !parentScope1->maxPreloadNum.has_value()) {
    parentScope1 = parentScope1->getParentOfType<Scope>();
  }
  while (parentScope2 != nullptr && !parentScope2->maxPreloadNum.has_value()) {
    parentScope2 = parentScope2->getParentOfType<Scope>();
  }
  if (!parentScope1 || !parentScope2) {
    return {};
  }
  if (auto *parCond1 = rwOp1->getParentOfType<Condition>()) {
    if (!parCond1->isProperAncestor(rwOp2)) {
      return {};
    }
  }
  if (auto *parCond2 = rwOp2->getParentOfType<Condition>()) {
    if (!parCond2->isProperAncestor(rwOp1)) {
      return {};
    }
  }

  auto *parentLoop1 = parentScope1->getParentOfType<Loop>();
  auto *parentLoop2 = parentScope2->getParentOfType<Loop>();
  if (parentLoop1 == nullptr || parentLoop1 != parentLoop2) {
    return {};
  }

  PTO_SYNC_SOLVER_CHECK(parentScope1->preloadNum.has_value());
  PTO_SYNC_SOLVER_CHECK(parentScope2->preloadNum.has_value());
  PTO_SYNC_SOLVER_CHECK(parentScope1->maxPreloadNum.value() ==
         parentScope2->maxPreloadNum.value());

  auto parentForLoop = llvm::dyn_cast_if_present<scf::ForOp>(parentLoop1->op);
  PTO_SYNC_SOLVER_CHECK(parentForLoop != nullptr);

  EventIdInfo eventIdInfo;
  eventIdInfo.eventIdNum = parentScope1->maxPreloadNum.value();
  eventIdInfo.preloadOffset1 = parentScope1->maxPreloadNum.value() -
                               parentScope1->preloadNum.value() - 1;
  eventIdInfo.preloadOffset2 = parentScope2->maxPreloadNum.value() -
                               parentScope2->preloadNum.value() - 1;
  eventIdInfo.multibufferLoop = parentForLoop;
  return eventIdInfo;
}

// Determine required event id count and optional multibuffer loop parent for
// occurrences.
EventIdInfo Solver::getEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   CorePipeInfo, CorePipeInfo) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  EventIdInfo singleEventId(1);
  if (!isBackwardSync(occ1, occ2)) {
    return singleEventId;
  }
  if (auto eventIdInfo = checkCVMultiBufferUnrollEventIdInfo(rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  if (auto eventIdInfo = checkCVMultiBufferPreloadEventIdInfo(rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  if (auto eventIdInfo =
          checkMultiBufferEventIdInfo(occ1, occ2, rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  return singleEventId;
}

// Graph-based check to determine if adding a sync between occ1 and occ2 would
// block progress. Uses GraphSolver (Dijkstra) to estimate minimal reachable
// index.
void Solver::addGraphConflictPair(
    GraphSolver &graphSolver, llvm::DenseSet<ConflictPair *> &visited,
    ConflictPair *conflictPair, EventIdInfo eventIdInfo, int startIndex,
    int endIndex,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) const {
  if (conflictPair->couldNotRun)
    return;
  if (conflictPair->endIndex < startIndex ||
      conflictPair->startIndex > endIndex) {
    return;
  }
  if (conflictPair->isInnerBackward) {
    int64_t lhs = eventIdInfo.eventIdNum * eventIdInfo.eventIdRepeatNum;
    int64_t rhs = conflictPair->eventIdInfo.eventIdNum *
                  conflictPair->eventIdInfo.eventIdRepeatNum;
    if (lhs < rhs)
      return;
  }
  if (llvm::find(ignoreConflictPairs, conflictPair) != ignoreConflictPairs.end())
    return;
  auto [it, isInserted] = visited.insert(conflictPair);
  if (!isInserted)
    return;
  DEBUG_WITH_TYPE("gss-sync-solver-check-graph-conflict", {
    llvm::dbgs() << "add-conflict-pair: " << conflictPair->str() << '\n';
  });
  graphSolver.addConflictPair(conflictPair);
}

void Solver::addGraphConflictsFromScopes(
    const Occurrence *occ, GraphSolver &graphSolver,
    llvm::DenseSet<ConflictPair *> &visited, EventIdInfo eventIdInfo,
    int startIndex, int endIndex,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs,
    bool persistent) {
  auto &conflictsMap = persistent ? persistentScopeOccChosenConflicts
                                  : scopeOccChosenConflicts;
  for (auto *parOcc : occ->getAllParents()) {
    if (!conflictsMap.contains(parOcc))
      continue;
    for (auto *conflictPair : conflictsMap[parOcc]) {
      addGraphConflictPair(graphSolver, visited, conflictPair, eventIdInfo,
                           startIndex, endIndex, ignoreConflictPairs);
    }
  }
}

void Solver::addGraphConflictsFromScopePairs(
    Occurrence *occ1, Occurrence *occ2, GraphSolver &graphSolver,
    llvm::DenseSet<ConflictPair *> &visited, EventIdInfo eventIdInfo,
    int startIndex, int endIndex,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) {
  for (auto &[scopeOccPair, chosenConflicts] : scopeOccPairChosenConflicts) {
    auto [scopeOcc1, scopeOcc2] = scopeOccPair;
    if (!scopeOcc1->isProperAncestor(occ1) ||
        !scopeOcc2->isProperAncestor(occ2)) {
      continue;
    }
    for (auto *conflictPair : chosenConflicts) {
      addGraphConflictPair(graphSolver, visited, conflictPair, eventIdInfo,
                           startIndex, endIndex, ignoreConflictPairs);
    }
  }
}

bool Solver::checkGraphConflict(
    Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    std::optional<int> startIndex, std::optional<int> endIndex,
    const llvm::SmallVector<ConflictPair *> &extraConflictPairs,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  if (!startIndex.has_value())
    startIndex = occ1->endIndex;
  if (!endIndex.has_value())
    endIndex = occ2->startIndex;
  GraphSolver graphSolver(options);
  llvm::DenseSet<ConflictPair *> visited;
  addGraphConflictsFromScopes(occ1, graphSolver, visited, eventIdInfo,
                              *startIndex, *endIndex, ignoreConflictPairs,
                              /*persistent=*/false);
  addGraphConflictsFromScopes(occ2, graphSolver, visited, eventIdInfo,
                              *startIndex, *endIndex, ignoreConflictPairs,
                              /*persistent=*/false);
  addGraphConflictsFromScopePairs(occ1, occ2, graphSolver, visited,
                                  eventIdInfo, *startIndex, *endIndex,
                                  ignoreConflictPairs);
  addGraphConflictsFromScopes(occ1, graphSolver, visited, eventIdInfo,
                              *startIndex, *endIndex, ignoreConflictPairs,
                              /*persistent=*/true);
  addGraphConflictsFromScopes(occ2, graphSolver, visited, eventIdInfo,
                              *startIndex, *endIndex, ignoreConflictPairs,
                              /*persistent=*/true);
  for (auto *conflictPair : extraConflictPairs)
    addGraphConflictPair(graphSolver, visited, conflictPair, eventIdInfo,
                         *startIndex, *endIndex, ignoreConflictPairs);
  std::optional<int> mnDistance;
  if (options.enableUnitFlagFeature) {
    mnDistance = graphSolver.runDijkstraUnitFlagEnabled(
        occ1, occ2, corePipeSrc, corePipeDst, *startIndex, *endIndex);
  } else {
    mnDistance = graphSolver.runDijkstra(corePipeSrc, corePipeDst, *startIndex,
                                         *endIndex);
  }
  return !mnDistance.has_value() || mnDistance.value() > *endIndex;
}

bool Solver::checkSetSyncOpsConflict(ConflictPair *conflictPair1,
                                     ConflictPair *conflictPair2) {
  if (conflictPair1->setCorePipeInfo == conflictPair2->setCorePipeInfo)
    return false;
  auto corePipeSrc = conflictPair1->setCorePipeInfo;
  auto corePipeDst = conflictPair2->setCorePipeInfo;
  auto startIndex = conflictPair1->startIndex + 1;
  auto endIndex = conflictPair2->startIndex;
  conflictPair1->startIndex += 1;
  PTO_SYNC_SOLVER_CHECK(conflictPair1->setOcc != nullptr &&
                        conflictPair2->setOcc != nullptr);
  bool result = checkGraphConflict(
      conflictPair1->setOcc, conflictPair2->setOcc, corePipeSrc, corePipeDst,
      conflictPair1->eventIdInfo, startIndex, endIndex, {conflictPair1},
      {conflictPair2});
  conflictPair1->startIndex -= 1;
  return result;
}

bool Solver::checkWaitSyncOpsConflict(ConflictPair *conflictPair1,
                                      ConflictPair *conflictPair2) {
  if (conflictPair1->waitCorePipeInfo == conflictPair2->waitCorePipeInfo)
    return false;
  auto corePipeSrc = conflictPair1->waitCorePipeInfo;
  auto corePipeDst = conflictPair2->waitCorePipeInfo;
  auto startIndex = conflictPair1->endIndex;
  auto endIndex = conflictPair2->endIndex - 1;
  conflictPair2->endIndex -= 1;
  PTO_SYNC_SOLVER_CHECK(conflictPair1->waitOcc != nullptr &&
                        conflictPair2->waitOcc != nullptr);
  bool result = checkGraphConflict(
      conflictPair1->waitOcc, conflictPair2->waitOcc, corePipeSrc, corePipeDst,
      conflictPair1->eventIdInfo, startIndex, endIndex, {conflictPair1},
      {conflictPair2});
  conflictPair2->endIndex += 1;
  return result;
}

bool Solver::checkSyncOpsConflicts(ConflictPair *conflictPair1,
                                   ConflictPair *conflictPair2) {
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier())
    return false;
  if (conflictPair1->startIndex > conflictPair2->startIndex)
    std::swap(conflictPair1, conflictPair2);
  if (conflictPair1->startIndex >= conflictPair2->startIndex ||
      conflictPair1->endIndex >= conflictPair2->endIndex) {
    return true;
  }
  bool result = checkSetSyncOpsConflict(conflictPair1, conflictPair2) ||
                checkWaitSyncOpsConflict(conflictPair1, conflictPair2);
  DEBUG_WITH_TYPE("gss-check-sync-ops-conflicts", {
    if (result) {
      llvm::dbgs() << "sync-ops-conflict-found: " << "\n";
      llvm::dbgs() << " " << conflictPair1->str() << '\n';
      llvm::dbgs() << " " << conflictPair2->str() << '\n';
    }
  });
  return result;
}

// Check whether two ConflictPair entries conflict in pipe and time ranges.
bool Solver::checkIntersect(ConflictPair *conflictPair1,
                            ConflictPair *conflictPair2) {
  PTO_SYNC_SOLVER_CHECK(conflictPair1 != nullptr && conflictPair2 != nullptr);
  if (conflictPair1 == conflictPair2) {
    return false;
  }
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->dontCheckForConflict ||
      conflictPair2->dontCheckForConflict) {
    return false;
  }
  if (options.isCrossCoreMode()) {
    return checkSyncOpsConflicts(conflictPair1, conflictPair2);
  }
  if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo ||
      conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    return false;
  }
  for (auto [l1, r1] : getRanges(conflictPair1)) {
    for (auto [l2, r2] : getRanges(conflictPair2)) {
      if (checkRangesIntersect(l1, r1 + 1, l2, r2 + 1)) {
        return true;
      }
    }
  }
  return false;
}

// Obtain available event ids while accounting for already chosen conflicts.
std::vector<ConflictPair *>
Solver::getIntersectingConflictPairs(ConflictPair *conflictPair) {
  PTO_SYNC_SOLVER_CHECK(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  if (conflictPair->dontCheckForConflict) {
    return {};
  }
  std::vector<ConflictPair *> intersectingConflictPairs;
  for (auto &curConflictPair : chosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  for (auto &curConflictPair : persistentChosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  return intersectingConflictPairs;
}

// Processed-pair tracking helpers.
bool Solver::checkVisited(Occurrence *occ1, Occurrence *occ2) {
  auto [it, isInserted] = processedOccPairs.insert(std::make_pair(occ1, occ2));
  return !isInserted;
}

bool Solver::checkSkippable(bool reverseOrder, Occurrence *occ) {
  return skipOcc[reverseOrder].contains(occ);
}

// Synced-pair memoization helpers.
EventIdNode *Solver::getOldEventIdNodeIfExists(ConflictPair *conflictPair) {
  PTO_SYNC_SOLVER_CHECK(conflictPair != nullptr);
  auto oldConflictPairs = getMemorizedSyncedPairs(conflictPair);
  if (oldConflictPairs.empty()) {
    return {};
  }
  ConflictPair *oldConflictPair = *oldConflictPairs.begin();
  PTO_SYNC_SOLVER_CHECK(oldConflictPair != nullptr && oldConflictPair->eventIdNode != nullptr);
  return oldConflictPair->eventIdNode;
}

llvm::DenseSet<ConflictPair *>
Solver::getMemorizedSyncedPairs(ConflictPair *conflictPair) {
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  return syncedPairs[key];
}

void Solver::memorizeSyncedPair(ConflictPair *conflictPair) {
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  syncedPairs[key].insert(conflictPair);
#ifndef NDEBUG
  for (auto *oldConflictPair : syncedPairs[key]) {
    PTO_SYNC_SOLVER_CHECK(oldConflictPair->eventIdNode == conflictPair->eventIdNode);
  }
#endif
}

void Solver::forgetSyncedPair(ConflictPair *conflictPair) {
  PTO_SYNC_SOLVER_CHECK(conflictPair != nullptr);
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  syncedPairs[key].erase(conflictPair);
}

void Solver::memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                      ConflictPair *reusedConflictPair) {
  PTO_SYNC_SOLVER_CHECK(conflictPair != nullptr);
  replacedWithReusableSyncedPairs[{
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo}] =
      reusedConflictPair;
}

bool Solver::skipMMad1DecomposedLoopOpt(Occurrence *occ1,
                                        Occurrence *occ2) const {
  auto *parentLoopOp1 = OperationBase::getParentloop(occ1->op);
  auto *parentLoopOp2 = OperationBase::getParentloop(occ2->op);
  if (parentLoopOp1 != nullptr && parentLoopOp2 != nullptr) {
    if (parentLoopOp1 != parentLoopOp2) {
      if (isa<MmadL1LoopOp>(parentLoopOp1) &&
          isa<MmadL1LoopOp>(parentLoopOp2)) {
        return true;
      }
    }
  }
  return false;
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                                   Occurrence *occ2, Occurrence *parOcc1,
                                   Occurrence *parOcc2) {
  if (!options.decomposeMmadl1Op) {
    return {};
  }
  if (occ1->parentOcc != nullptr && occ1->parentOcc->parentOcc != nullptr &&
      occ1->parentOcc->parentOcc->parentOcc == parOcc1 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ1->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(
          occ1->parentOcc->parentOcc->op)) {
    conflictPair->setOnLastIterOnly = true;
    return std::make_pair(occ1, parOcc2);
  }
  if (!conflictPair->isInnerBackward && occ2->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc->parentOcc == parOcc2 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ2->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(
          occ2->parentOcc->parentOcc->op)) {
    conflictPair->waitOnFirstIterOnly = true;
    return std::make_pair(parOcc1, occ2);
  }
  return {};
}

std::optional<UnitFlagInfo> Solver::checkUnitFlagPatterns(Occurrence *,
                                                          Occurrence *) const {
  return {};
}

Occurrence *Solver::getBeforePlaceHolderOcc(Occurrence *occ) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr);
  PTO_SYNC_SOLVER_CHECK(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrIndex - 1;
  PTO_SYNC_SOLVER_CHECK(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp != nullptr);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp->beforeOp == occ->op);
#endif
  return placeHolderOcc;
}

Occurrence *Solver::getAfterPlaceHolderOcc(Occurrence *occ) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr);
  PTO_SYNC_SOLVER_CHECK(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrEndIndex;
  PTO_SYNC_SOLVER_CHECK(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp != nullptr);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp->afterOp == occ->op);
#endif
  return placeHolderOcc;
}

Occurrence *Solver::getScopeBeginPlaceHolderOcc(Occurrence *occ) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr);
  PTO_SYNC_SOLVER_CHECK(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrIndex + 1;
  PTO_SYNC_SOLVER_CHECK(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp != nullptr);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp->scopeBegin == occ->op);
#endif
  return placeHolderOcc;
}

Occurrence *Solver::getScopeEndPlaceHolderOcc(Occurrence *occ) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr);
  PTO_SYNC_SOLVER_CHECK(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrEndIndex - 1;
  PTO_SYNC_SOLVER_CHECK(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp != nullptr);
  PTO_SYNC_SOLVER_CHECK(placeHolderOp->scopeEnd == occ->op);
#endif
  return placeHolderOcc;
}

std::pair<Occurrence *, Occurrence *>
Solver::getSetWaitLCAPairOcc(Occurrence *occ1, Occurrence *occ2) const {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);

  auto [grandParOcc1, grandParOcc2] = Occurrence::getLCAPair(occ1, occ2);
  PTO_SYNC_SOLVER_CHECK(grandParOcc1 != nullptr && grandParOcc2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(grandParOcc1->parentOcc != nullptr &&
         grandParOcc2->parentOcc != nullptr);

  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  PTO_SYNC_SOLVER_CHECK(parOp1 != nullptr && parOp2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);
  PTO_SYNC_SOLVER_CHECK(parOp1->parentOp == parOp2->parentOp);

  auto *parOcc1 = Occurrence::getParentWithOp(occ1, parOp1->parentOp);
  auto *parOcc2 = Occurrence::getParentWithOp(occ2, parOp2->parentOp);
  PTO_SYNC_SOLVER_CHECK(parOcc1 != nullptr && parOcc2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(parOcc1 != occ1 && parOcc2 != occ2);

  auto *setOcc =
      Occurrence::getNthParent(occ1, occ1->depth - parOcc1->depth - 1);
  auto *waitOcc =
      Occurrence::getNthParent(occ2, occ2->depth - parOcc2->depth - 1);
  PTO_SYNC_SOLVER_CHECK(setOcc != nullptr && waitOcc != nullptr);
  PTO_SYNC_SOLVER_CHECK(parOcc1->isProperAncestor(setOcc));
  PTO_SYNC_SOLVER_CHECK(parOcc2->isProperAncestor(waitOcc));

  auto *parLoop = Occurrence::getParentloop(setOcc);
  while (parLoop != nullptr && grandParOcc1->isProperAncestor(parLoop)) {
    setOcc = parLoop;
    waitOcc = Occurrence::getParentloop(waitOcc);
    parLoop = Occurrence::getParentloop(setOcc);
  }
  return std::make_pair(setOcc, waitOcc);
}

void Solver::adjustSetWaitForLoopBoundary(Occurrence *occ1, Occurrence *&setOcc,
                                          Occurrence *&waitOcc) {
  if (setOcc->op == waitOcc->op)
    return;
  auto *parLoopOp = llvm::dyn_cast_if_present<Loop>(setOcc->parentOcc->op);
  if (!parLoopOp || parLoopOp->body.size() <= 1 || isa<PlaceHolder>(waitOcc->op))
    return;
  auto *placeHolderOcc = getScopeEndPlaceHolderOcc(setOcc);
  std::tie(setOcc, waitOcc) = getSetWaitLCAPairOcc(occ1, placeHolderOcc);
}

void Solver::adjustSetWaitForBackwardCondition(Occurrence *occ1,
                                               Occurrence *occ2,
                                               Occurrence *&setOcc,
                                               Occurrence *&waitOcc) const {
  if (!isBackwardSync(occ1, occ2))
    return;
  if (setOcc->parentOcc &&
      llvm::isa_and_present<Condition>(setOcc->parentOcc->op)) {
    setOcc = setOcc->parentOcc;
  }
  if (waitOcc->parentOcc &&
      llvm::isa_and_present<Condition>(waitOcc->parentOcc->op)) {
    waitOcc = waitOcc->parentOcc;
  }
}

void Solver::adjustSetWaitForCrossCoreLoops(Occurrence *occ1, Occurrence *occ2,
                                            Occurrence *&setOcc,
                                            Occurrence *&waitOcc) {
  if (!options.isCrossCoreMode())
    return;
  PTO_SYNC_SOLVER_CHECK(setOcc->op != nullptr && waitOcc->op != nullptr);
  auto *forOp1 = llvm::dyn_cast_if_present<Loop>(setOcc->op);
  auto *forOp2 = llvm::dyn_cast_if_present<Loop>(waitOcc->op);
  if (!forOp1 || !forOp2 || !forOp1->multibufferUnrollNum ||
      !forOp2->multibufferUnrollNum) {
    return;
  }
  PTO_SYNC_SOLVER_CHECK(forOp1->multibufferUnrollNum ==
                        forOp2->multibufferUnrollNum);
  setOcc = Occurrence::getNthParent(
      occ1, occ1->depth - setOcc->depth - kCrossCoreParentDepthOffset);
  waitOcc = Occurrence::getNthParent(
      occ2, occ2->depth - waitOcc->depth - kCrossCoreParentDepthOffset);
}

void Solver::adjustSetWaitForCrossCoreScopes(Occurrence *occ1,
                                             Occurrence *occ2,
                                             Occurrence *&setOcc,
                                             Occurrence *&waitOcc) {
  if (!options.isCrossCoreMode())
    return;
  PTO_SYNC_SOLVER_CHECK(setOcc->op != nullptr && waitOcc->op != nullptr);
  auto *scopeOp1 = llvm::dyn_cast_if_present<Scope>(setOcc->op);
  auto *scopeOp2 = llvm::dyn_cast_if_present<Scope>(waitOcc->op);
  if (!scopeOp1 || !scopeOp2 || !scopeOp1->maxPreloadNum ||
      !scopeOp2->maxPreloadNum) {
    return;
  }
  PTO_SYNC_SOLVER_CHECK(scopeOp1->maxPreloadNum == scopeOp2->maxPreloadNum);
  setOcc = Occurrence::getNthParent(
      occ1, occ1->depth - setOcc->depth - kCrossCoreParentDepthOffset);
  waitOcc = Occurrence::getNthParent(
      occ2, occ2->depth - waitOcc->depth - kCrossCoreParentDepthOffset);
}

void Solver::adjustSetWaitForLoopPlaceholders(Occurrence *&setOcc,
                                              Occurrence *&waitOcc) {
  if (llvm::isa_and_present<Loop>(setOcc->op))
    setOcc = getAfterPlaceHolderOcc(setOcc);
  if (llvm::isa_and_present<Loop>(waitOcc->op))
    waitOcc = getBeforePlaceHolderOcc(waitOcc);
}

std::pair<Occurrence *, Occurrence *>
Solver::getFixedSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  auto [setOcc, waitOcc] = getSetWaitLCAPairOcc(occ1, occ2);
  adjustSetWaitForLoopBoundary(occ1, setOcc, waitOcc);
  adjustSetWaitForBackwardCondition(occ1, occ2, setOcc, waitOcc);
  adjustSetWaitForCrossCoreLoops(occ1, occ2, setOcc, waitOcc);
  adjustSetWaitForCrossCoreScopes(occ1, occ2, setOcc, waitOcc);
  adjustSetWaitForLoopPlaceholders(setOcc, waitOcc);
  return std::make_pair(setOcc, waitOcc);
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getFunctionBlockSetWaitOcc(const Occurrence *occ1, Occurrence *occ2) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  auto *parFunctionBlock1 = occ1->getParentOfType<FunctionBlock>();
  auto *parFunctionBlock2 = occ2->getParentOfType<FunctionBlock>();
  if (parFunctionBlock1 == parFunctionBlock2) {
    return {};
  }
  auto *placeHolderOcc = getScopeBeginPlaceHolderOcc(parFunctionBlock2);
  return std::make_pair(placeHolderOcc, occ2);
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  if (options.isCrossCoreMode() && isBackwardSync(occ1, occ2)) {
    return {};
  }
  if (auto *unlikelyParCondOcc1 =
          Occurrence::getUnlikelyParentCondition(occ1)) {
    if (!unlikelyParCondOcc1->isProperAncestor(occ2)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc1);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ2)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(
            Occurrence::getNthParent(
                occ1, occ1->depth - unlikelyParCondOcc1->depth - 1));
        return std::make_pair(occ1, placeHolderOcc);
      }
    }
  }
  if (auto *unlikelyParCondOcc2 =
          Occurrence::getUnlikelyParentCondition(occ2)) {
    if (!unlikelyParCondOcc2->isProperAncestor(occ1)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc2);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ1)) {
        auto *placeHolderOcc = getScopeBeginPlaceHolderOcc(
            Occurrence::getNthParent(
                occ2, occ2->depth - unlikelyParCondOcc2->depth - 1));
        return std::make_pair(placeHolderOcc, occ2);
      }
    }
  }
  return {};
}

std::pair<Occurrence *, Occurrence *> Solver::getSetWaitOcc(Occurrence *occ1,
                                                            Occurrence *occ2) {
  if (auto functionBlockOpt = getFunctionBlockSetWaitOcc(occ1, occ2)) {
    std::tie(occ1, occ2) = functionBlockOpt.value();
  }
  if (auto unlikelyOpt = getUnlikelyCondSetWaitOcc(occ1, occ2)) {
    std::tie(occ1, occ2) = unlikelyOpt.value();
  }
  return getFixedSetWaitOcc(occ1, occ2);
}

Occurrence *Solver::getBarrierWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (!waitOcc->isProperAncestor(occ2)) {
    return waitOcc;
  }
  auto allParents = occ2->getAllParents();
  while (!allParents.empty() && allParents.back()->isProperAncestor(waitOcc)) {
    allParents.pop_back();
  }
  while (allParents.size() >= kConditionScopeParentPairSize &&
         llvm::isa_and_present<Condition>(allParents.back()->op)) {
    allParents.pop_back();
    PTO_SYNC_SOLVER_CHECK(llvm::isa_and_present<Scope>(allParents.back()->op));
    allParents.pop_back();
  }
  waitOcc = !allParents.empty() ? allParents.back() : occ2;
  return waitOcc;
}

void Solver::insertBarrierAllBeforeOcc(Occurrence *occ, bool isUseless,
                                       bool isPersistent) {
  PTO_SYNC_SOLVER_CHECK(occ != nullptr);
  auto *rwOp = llvm::dyn_cast_if_present<RWOperation>(occ->op);
  PTO_SYNC_SOLVER_CHECK(rwOp != nullptr);
  auto conflictPair = std::make_unique<ConflictPair>(
      nullptr, nullptr, rwOp, rwOp, occ, occ,
      CorePipeInfo(pto::TCoreType::CUBE_OR_VECTOR, pto::PIPE::PIPE_ALL),
      CorePipeInfo(pto::TCoreType::CUBE_OR_VECTOR, pto::PIPE::PIPE_ALL),
      occ->startIndex, occ->startIndex);
  conflictPair->isUseless = isUseless;
  auto *normScopeOcc = occ->parentOcc;
  PTO_SYNC_SOLVER_CHECK(normScopeOcc != nullptr);
  LLVM_DEBUG(llvm::dbgs() << (isPersistent ? "is-persistent " : "")
                          << occ->op->str(0, false) << ' '
                          << conflictPair->str() << '\n';);
  if (isPersistent) {
    persistentScopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
    persistentChosenConflictedPairs.push_back(std::move(conflictPair));
  } else {
    insertedBarrierAllBefore[occ->op].insert({occ, isUseless});
    scopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
    chosenConflictedPairs.push_back(std::move(conflictPair));
  }
}

void Solver::insertBarrierAllBeforeOp(const OperationBase *op, bool isUseless,
                                      bool isPersistent) {
  PTO_SYNC_SOLVER_CHECK(op != nullptr);
  for (auto *occ : opAllOccurrences[op]) {
    insertBarrierAllBeforeOcc(occ, isUseless, isPersistent);
    isUseless = true;
  }
}

// When barrier-all markers need to be chosen, insert them before all
// occurrences for the chosen op.
void Solver::pickAndInsertABarrierAll() {
  PTO_SYNC_SOLVER_CHECK(!insertedBarrierAllBefore.empty());
  OperationBase *chosenOp = nullptr;
  for (auto &[op, vec] : insertedBarrierAllBefore) {
    if (vec.empty()) {
      continue;
    }
    if (chosenOp == nullptr || chosenOp->id > op->id) {
      chosenOp = op;
    }
  }
  PTO_SYNC_SOLVER_CHECK(chosenOp != nullptr);
  insertBarrierAllBeforeOp(chosenOp, /*isUseless=*/false,
                           /*isPersistent=*/true);
}

bool Solver::isBackwardSync(Occurrence *occ1, Occurrence *occ2) const {
  if (occ1->op->id >= occ2->op->id) {
    return true;
  }
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(occ1->op != nullptr && occ2->op != nullptr);
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  return parOcc1->parentOcc->op != parOp1->parentOp;
}

bool Solver::reuseCmp(const ConflictPair *conflictPair1,
                      const ConflictPair *conflictPair2) const {
  PTO_SYNC_SOLVER_CHECK(conflictPair1 != nullptr && conflictPair2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(conflictPair1->op1 != nullptr && conflictPair1->op2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(conflictPair2->op1 != nullptr && conflictPair2->op2 != nullptr);
  if (conflictPair1->startIndex != conflictPair2->startIndex) {
    return conflictPair1->startIndex < conflictPair2->startIndex;
  }
  if (conflictPair1->endIndex != conflictPair2->endIndex) {
    return conflictPair1->endIndex > conflictPair2->endIndex;
  }
  if (conflictPair1->op1 != conflictPair2->op1) {
    return conflictPair1->op1->id > conflictPair2->op1->id;
  }
  if (conflictPair1->op2 != conflictPair2->op2) {
    return conflictPair1->op2->id > conflictPair2->op2->id;
  }
  return false;
}

ConflictPair *Solver::getReusableConflictPair(
    ConflictPair *conflictPair,
    const llvm::DenseSet<ConflictPair *> &conflictPairsSet) {
  PTO_SYNC_SOLVER_CHECK(conflictPair != nullptr);
  ConflictPair *ret = nullptr;
  for (auto *curConflictPair : conflictPairsSet) {
    if (curConflictPair->isBarrier() || curConflictPair->dontReuse) {
      continue;
    }
    if (curConflictPair->op1 != conflictPair->op1 ||
        curConflictPair->op2 != conflictPair->op2 ||
        curConflictPair->setCorePipeInfo != conflictPair->setCorePipeInfo ||
        curConflictPair->waitCorePipeInfo != conflictPair->waitCorePipeInfo) {
      continue;
    }
    if (!checkIntersect(conflictPair, curConflictPair)) {
      continue;
    }
    if (curConflictPair->startIndex >= conflictPair->startIndex) {
      continue;
    }
    if (conflictPair->eventIdNode->eventIdNum <
        curConflictPair->eventIdNode->eventIdNum) {
      continue;
    }
    PTO_SYNC_SOLVER_CHECK(conflictPair->eventIdNode != nullptr);
    PTO_SYNC_SOLVER_CHECK(curConflictPair->eventIdNode != nullptr);
    if (conflictPair->eventIdNode->eventIdNum >
        curConflictPair->eventIdNode->eventIdNum) {
      if ((conflictPair->eventIdNode->eventIdNum %
           curConflictPair->eventIdNode->eventIdNum) != 0) {
        continue;
      }
    }
    PTO_SYNC_SOLVER_CHECK(conflictPair->startIndex <= curConflictPair->endIndex);
    PTO_SYNC_SOLVER_CHECK(curConflictPair->endIndex <= conflictPair->endIndex);
    if (ret == nullptr || reuseCmp(ret, curConflictPair)) {
      ret = curConflictPair;
    }
  }
  return ret;
}

ConflictPair *Solver::findOldReusedConflictPair(ConflictPair *conflictPair) {
  if (!conflictPair->isUseless)
    return nullptr;
  auto it = replacedWithReusableSyncedPairs.find(
      {conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
       conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
  if (it == replacedWithReusableSyncedPairs.end())
    return nullptr;
  return it->second;
}

bool Solver::hasReusableConflictBudget(
    const ConflictPair *conflictPair,
    const ConflictPair *oldReusedConflictPair) {
  auto corePipeSrc = conflictPair->setCorePipeInfo;
  auto corePipeDst = conflictPair->waitCorePipeInfo;
  if (oldReusedConflictPair)
    return true;
  if (!reusePairs.contains({corePipeSrc, corePipeDst}) ||
      reusePairs[{corePipeSrc, corePipeDst}] <=
          reusedPairs[{corePipeSrc, corePipeDst}]) {
    return false;
  }
  PTO_SYNC_SOLVER_CHECK(
      reusePairs.contains(std::make_tuple(corePipeSrc, corePipeDst)));
  PTO_SYNC_SOLVER_CHECK(
      reusePairs[std::make_tuple(corePipeSrc, corePipeDst)] >=
      reusedPairs[std::make_tuple(corePipeSrc, corePipeDst)]);
  return true;
}

ConflictPair *Solver::findBestReusableConflictPair(ConflictPair *conflictPair,
                                                   Occurrence *scopeOcc1,
                                                   Occurrence *scopeOcc2) {
  ConflictPair *reusableConflictPair = nullptr;
  auto updateReusableCandidate = [this, &reusableConflictPair](
                                     ConflictPair *candidate) {
    if (!candidate)
      return;
    if (!reusableConflictPair || reuseCmp(reusableConflictPair, candidate))
      reusableConflictPair = candidate;
  };
  auto it1 = scopeOccChosenConflicts.find(scopeOcc1);
  auto it2 = scopeOccChosenConflicts.find(scopeOcc2);
  auto it3 = scopeOccPairChosenConflicts.find({scopeOcc1, scopeOcc2});
  auto it4 = persistentScopeOccChosenConflicts.find(scopeOcc1);
  auto it5 = persistentScopeOccChosenConflicts.find(scopeOcc2);
  if (it1 != scopeOccChosenConflicts.end())
    updateReusableCandidate(getReusableConflictPair(conflictPair, it1->second));
  if (it2 != scopeOccChosenConflicts.end())
    updateReusableCandidate(getReusableConflictPair(conflictPair, it2->second));
  if (it3 != scopeOccPairChosenConflicts.end())
    updateReusableCandidate(getReusableConflictPair(conflictPair, it3->second));
  if (it4 != persistentScopeOccChosenConflicts.end())
    updateReusableCandidate(getReusableConflictPair(conflictPair, it4->second));
  if (it5 != persistentScopeOccChosenConflicts.end())
    updateReusableCandidate(getReusableConflictPair(conflictPair, it5->second));
  return reusableConflictPair;
}

void Solver::applyReusableConflictPair(ConflictPair *conflictPair,
                                       ConflictPair *reusableConflictPair) const {
  PTO_SYNC_SOLVER_CHECK(reusableConflictPair->startIndex < conflictPair->startIndex);
  PTO_SYNC_SOLVER_CHECK(reusableConflictPair->endIndex <= conflictPair->endIndex);
  reusableConflictPair->setOp = conflictPair->setOp;
  reusableConflictPair->setOcc = conflictPair->setOcc;
  reusableConflictPair->startIndex = conflictPair->startIndex;
}

bool Solver::reuseConflictPair(ConflictPair *conflictPair,
                               Occurrence *scopeOcc1, Occurrence *scopeOcc2) {
  if (conflictPair->isBarrier())
    return false;
  if (scopeOcc1->op != scopeOcc2->op)
    return false;
  if (!barrierAllPairs.empty())
    return false;
  ConflictPair *oldReusedConflictPair = findOldReusedConflictPair(conflictPair);

#ifndef NDEBUG
  if (!conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
         conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
    PTO_SYNC_SOLVER_CHECK(it == replacedWithReusableSyncedPairs.end());
  }
#endif

  if (conflictPair->isUseless && oldReusedConflictPair == nullptr)
    return false;
  if (!hasReusableConflictBudget(conflictPair, oldReusedConflictPair))
    return false;
  ConflictPair *reusableConflictPair =
      findBestReusableConflictPair(conflictPair, scopeOcc1, scopeOcc2);
  if (reusableConflictPair == nullptr)
    return false;

  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    llvm::dbgs() << "reuse: " << conflictPair->str() << '\n';
    llvm::dbgs() << "with: " << reusableConflictPair->str() << '\n';
  });

  applyReusableConflictPair(conflictPair, reusableConflictPair);

  if (!conflictPair->isUseless)
    memorizeReusedSyncedPair(conflictPair, reusableConflictPair);

  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    if (oldReusedConflictPair != nullptr)
      llvm::dbgs() << "old-reuse: " << oldReusedConflictPair->str() << '\n';
  });

  if (oldReusedConflictPair != nullptr) {
    PTO_SYNC_SOLVER_CHECK(oldReusedConflictPair->op1 == reusableConflictPair->op1);
    PTO_SYNC_SOLVER_CHECK(oldReusedConflictPair->op2 == reusableConflictPair->op2);
    PTO_SYNC_SOLVER_CHECK(oldReusedConflictPair->waitOp == reusableConflictPair->waitOp);
  }

  if (!conflictPair->isUseless)
    reusedPairs[{conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo}] += 1;

  return true;
}

std::unique_ptr<EventIdSolver> &
Solver::getEventIdSolverRef(pto::PIPE pipeSrc, pto::PIPE pipeDst) {
  if (options.isCrossCoreMode()) {
    pipeSrc = pto::PIPE::PIPE_UNASSIGNED;
    pipeDst = pto::PIPE::PIPE_UNASSIGNED;
  }
  auto key = std::make_tuple(pipeSrc, pipeDst);
  if (!eventIdSolver.contains(key)) {
    int64_t eventIdNumMax =
        getHWAvailableEventIdNum(options.syncMode, pipeSrc, pipeDst);
    if (options.eventIdNumMax.has_value()) {
      eventIdNumMax = std::min(eventIdNumMax, options.eventIdNumMax.value());
      eventIdNumMax = std::max<int64_t>(eventIdNumMax, 1);
    }
    eventIdSolver[key] = std::make_unique<EventIdSolver>(eventIdNumMax);
  }
  return eventIdSolver[key];
}

bool Solver::checkReuseMultiBufferFlagId(ConflictPair *conflictPair) {
  if (options.useDifferentMultiBufferFlagIds) {
    return false;
  }
  if (!conflictPair->isInnerBackward ||
      conflictPair->eventIdInfo.eventIdNum <= 1 ||
      conflictPair->movedToOuterLoop) {
    return false;
  }
  auto [setOcc, waitOcc] =
      std::tie(conflictPair->setOcc, conflictPair->waitOcc);
  auto *backwardSyncLoopOcc = conflictPair->backwardSyncLoopOcc;
  PTO_SYNC_SOLVER_CHECK(backwardSyncLoopOcc != nullptr);
  if (auto *parCondOcc1 = setOcc->getParentOfType<Condition>()) {
    if (!parCondOcc1->isProperAncestor(backwardSyncLoopOcc)) {
      return false;
    }
  }
  if (auto *parCondOcc2 = waitOcc->getParentOfType<Condition>()) {
    if (!parCondOcc2->isProperAncestor(backwardSyncLoopOcc)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<ConflictPair> Solver::createSetWaitConflictPair(
    Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1, RWOperation *rwOp2,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    bool isUseless, Occurrence *&setOcc, Occurrence *&waitOcc,
    Occurrence *&normScopeOcc1, Occurrence *&normScopeOcc2,
    OperationBase *&normScopeOp, Loop *&parentLCALoopOp,
    Occurrence *&parentLCALoopOcc, Occurrence *&parentLCALoopBeforePHOcc,
    Occurrence *&parentLCALoopAfterPHOcc) {
  std::tie(setOcc, waitOcc) = getSetWaitOcc(occ1, occ2);
  auto [lcaSetOp, lcaWaitOp] = OperationBase::getLCAPair(setOcc->op, waitOcc->op);
  normScopeOcc1 = Occurrence::getParentWithOp(setOcc, lcaSetOp->parentOp);
  normScopeOcc2 = Occurrence::getParentWithOp(waitOcc, lcaWaitOp->parentOp);
  PTO_SYNC_SOLVER_CHECK(normScopeOcc1->op == normScopeOcc2->op);
  normScopeOp = normScopeOcc1->op;
  PTO_SYNC_SOLVER_CHECK(normScopeOp != nullptr && normScopeOp->parentOp != nullptr);

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  PTO_SYNC_SOLVER_CHECK(conflictPair->startIndex <= conflictPair->endIndex);
  conflictPair->isUseless = isUseless;
  conflictPair->isInnerBackward = isBackwardSync(setOcc, waitOcc);
  conflictPair->eventIdInfo = eventIdInfo;
  if (!conflictPair->isInnerBackward)
    return conflictPair;

  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  PTO_SYNC_SOLVER_CHECK(parOcc1 != nullptr && parOcc2 != nullptr);
  parentLCALoopOcc = parOcc1->getParentOfType<Loop>();
  if (moveBackwardSyncPairsToOutmostLoop) {
    while (auto *grandParentLoopOcc = parentLCALoopOcc->getParentOfType<Loop>()) {
      conflictPair->movedToOuterLoop = true;
      parentLCALoopOcc = grandParentLoopOcc;
    }
  }
  PTO_SYNC_SOLVER_CHECK(parentLCALoopOcc != nullptr);
  conflictPair->backwardSyncLoopOcc = parentLCALoopOcc;
  parentLCALoopOp = llvm::dyn_cast<Loop>(parentLCALoopOcc->op);
  PTO_SYNC_SOLVER_CHECK(parentLCALoopOp != nullptr);
  conflictPair->backwardSyncLoopOp = parentLCALoopOp;
  parentLCALoopBeforePHOcc = getBeforePlaceHolderOcc(parentLCALoopOcc);
  parentLCALoopAfterPHOcc = getAfterPlaceHolderOcc(parentLCALoopOcc);
  PTO_SYNC_SOLVER_CHECK(parentLCALoopBeforePHOcc != nullptr &&
                        parentLCALoopAfterPHOcc != nullptr);
  return conflictPair;
}

void Solver::normalizeSetWaitConflictPair(ConflictPair *conflictPair,
                                          Occurrence *occ1, Occurrence *occ2,
                                          Occurrence *&setOcc,
                                          Occurrence *&waitOcc,
                                          CorePipeInfo corePipeSrc,
                                          CorePipeInfo corePipeDst) {
  if (auto setWaitOccs = checkAndApplyMmadl0LoopOpt(conflictPair, occ1, occ2,
                                                    setOcc, waitOcc)) {
    std::tie(setOcc, waitOcc) = setWaitOccs.value();
    conflictPair->updateSetWaitOccs(setOcc, waitOcc);
  }
  if (!conflictPair->isInnerBackward ||
      disabledMultiEventIdPairs.contains({corePipeSrc, corePipeDst})) {
    conflictPair->eventIdInfo = EventIdInfo(1);
  }
  if (!checkReuseMultiBufferFlagId(conflictPair))
    return;
  conflictPair->eventIdInfo.eventIdRepeatNum =
      conflictPair->eventIdInfo.eventIdNum;
  conflictPair->eventIdInfo.eventIdNum = 1;
}

bool Solver::prepareSetWaitConflictPairWithEventSolver(
    EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
    Occurrence *occ1, Occurrence *occ2, Occurrence *normScopeOcc1,
    Occurrence *normScopeOcc2, const OperationBase *normScopeOp,
    OperationBase *barrierOp, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst) {
  curEventIdSolver.pushActionNone();
  initializeConflictEventIdNode(conflictPair, curEventIdSolver, occ1, occ2,
                                normScopeOp);
  if (options.reuseSyncPairToSaveEventIds &&
      reuseConflictPair(conflictPair, normScopeOcc1, normScopeOcc2)) {
    curEventIdSolver.undoActions();
    return false;
  }
  auto intersectingConflictPairs = getIntersectingConflictPairs(conflictPair);
  curEventIdSolver.addConflicts(conflictPair, intersectingConflictPairs);
  return checkColorableOrConvertToBarrierAll(curEventIdSolver, conflictPair,
                                             barrierOp, corePipeSrc,
                                             corePipeDst);
}

bool Solver::checkColorableOrConvertToBarrierAll(EventIdSolver &curEventIdSolver,
                                                 ConflictPair *conflictPair,
                                                 OperationBase *barrierOp,
                                                 CorePipeInfo corePipeSrc,
                                                 CorePipeInfo corePipeDst) {
  if (curEventIdSolver.isColorable())
    return true;
  LLVM_DEBUG(llvm::dbgs() << "will-be-converted-to-barrier-all "
                          << conflictPair->str() << '\n';);
  insertBarrierAllBeforeOp(barrierOp, conflictPair->isUseless,
                           /*isPersistent=*/false);
  barrierAllPairs.insert({corePipeSrc, corePipeDst});
  curEventIdSolver.undoActions();
  return false;
}

void Solver::initializeConflictEventIdNode(ConflictPair *conflictPair,
                                           EventIdSolver &curEventIdSolver,
                                           Occurrence *occ1, Occurrence *occ2,
                                           const OperationBase *normScopeOp) {
  if (auto *oldEventIdNode = getOldEventIdNodeIfExists(conflictPair)) {
    conflictPair->eventIdNode = oldEventIdNode;
    curEventIdSolver.insertConflictPair(oldEventIdNode, conflictPair);
    return;
  }
  bool reversedPriority = false;
  if (conflictPair->isInnerBackward &&
      OperationBase::getParentloop(occ1->op) == normScopeOp->parentOp &&
      OperationBase::getParentloop(occ2->op) == normScopeOp->parentOp) {
    reversedPriority = true;
  }
  conflictPair->eventIdNode = curEventIdSolver.createNode(
      conflictPair, conflictPair->eventIdInfo.eventIdNum, reversedPriority);
}

bool Solver::insertExtraConflictPair(
    EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
    Occurrence *setOcc, Occurrence *waitOcc, Occurrence *parentScope,
    OperationBase *barrierOp, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, ExtraConflictPairs &extraConflictPairs,
    bool couldNotRun) {
  PTO_SYNC_SOLVER_CHECK(setOcc != nullptr && waitOcc != nullptr && parentScope != nullptr);
  auto extraConflictPair = conflictPair->clone(setOcc, waitOcc);
  extraConflictPair->isUseless = true;
  extraConflictPair->dontReuse = true;
  if (couldNotRun || options.moveOutAndMergeBackwardSyncPairs)
    extraConflictPair->couldNotRun = true;
  LLVM_DEBUG({
    llvm::dbgs() << "extra-conflict-pair: " << extraConflictPair->str()
                 << "\n";
  });
  curEventIdSolver.insertConflictPair(conflictPair->eventIdNode,
                                      extraConflictPair.get());
  auto intersectingConflictPairs =
      getIntersectingConflictPairs(extraConflictPair.get());
  curEventIdSolver.addConflicts(extraConflictPair.get(),
                                intersectingConflictPairs);
  if (!checkColorableOrConvertToBarrierAll(curEventIdSolver, conflictPair,
                                           barrierOp, corePipeSrc,
                                           corePipeDst)) {
    return false;
  }
  extraConflictPairs.push_back(
      std::make_pair(std::move(extraConflictPair), parentScope));
  return true;
}

bool Solver::insertOuterBackwardConflictPairIfNeeded(
    EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
    const Occurrence *setOcc, const Occurrence *waitOcc,
    const Loop *parentLCALoopOp,
    const Occurrence *parentLCALoopOcc, Occurrence *parentLCALoopBeforePHOcc,
    Occurrence *parentLCALoopAfterPHOcc, OperationBase *barrierOp,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
    ExtraConflictPairs &extraConflictPairs) {
  if (!conflictPair->isInnerBackward || conflictPair->eventIdNode == nullptr)
    return true;
  bool insertOuterBwdConflictPair = false;
  if ((conflictPair->eventIdInfo.eventIdNum *
       conflictPair->eventIdInfo.eventIdRepeatNum) > 1) {
    insertOuterBwdConflictPair = true;
  } else if (options.isCrossCoreMode()) {
    if (setOcc->parentOcc == nullptr || setOcc->parentOcc->parentOcc == nullptr ||
        setOcc->parentOcc->parentOcc->op != parentLCALoopOp) {
      insertOuterBwdConflictPair = true;
    } else if (waitOcc->parentOcc == nullptr ||
               waitOcc->parentOcc->parentOcc == nullptr ||
               waitOcc->parentOcc->parentOcc->op != parentLCALoopOp) {
      insertOuterBwdConflictPair = true;
    }
  }
  if (!insertOuterBwdConflictPair)
    return true;
  return insertExtraConflictPair(
      curEventIdSolver, conflictPair, parentLCALoopBeforePHOcc,
      parentLCALoopAfterPHOcc, parentLCALoopOcc->parentOcc, barrierOp,
      corePipeSrc, corePipeDst, extraConflictPairs);
}

bool Solver::insertBoundaryBackwardConflictPairsIfNeeded(
    EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
    Occurrence *setOcc, Occurrence *waitOcc, Occurrence *normScopeOcc1,
    Occurrence *normScopeOcc2, Occurrence *parentLCALoopOcc,
    Occurrence *parentLCALoopBeforePHOcc,
    Occurrence *parentLCALoopAfterPHOcc, OperationBase *barrierOp,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
    ExtraConflictPairs &extraConflictPairs) {
  if (!conflictPair->isInnerBackward || conflictPair->eventIdNode == nullptr)
    return true;
  auto *loopOpOcc1 = getFirstIterOcc(waitOcc, normScopeOcc1);
  auto *loopOpOcc2 = getLastIterOcc(setOcc, normScopeOcc2);
  if (!insertExtraConflictPair(curEventIdSolver, conflictPair,
                               parentLCALoopBeforePHOcc, loopOpOcc1,
                               parentLCALoopOcc, barrierOp, corePipeSrc,
                               corePipeDst, extraConflictPairs,
                               /*couldNotRun=*/true)) {
    return false;
  }
  return insertExtraConflictPair(curEventIdSolver, conflictPair, loopOpOcc2,
                                 parentLCALoopAfterPHOcc, parentLCALoopOcc,
                                 barrierOp, corePipeSrc, corePipeDst,
                                 extraConflictPairs, /*couldNotRun=*/true);
}

bool Solver::collectBackwardConflictExtras(
    EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
    Occurrence *setOcc, Occurrence *waitOcc, Occurrence *normScopeOcc1,
    Occurrence *normScopeOcc2, Loop *parentLCALoopOp,
    Occurrence *parentLCALoopOcc, Occurrence *parentLCALoopBeforePHOcc,
    Occurrence *parentLCALoopAfterPHOcc, OperationBase *barrierOp,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
    ExtraConflictPairs &extraConflictPairs) {
  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr)
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
  });
  if (!insertOuterBackwardConflictPairIfNeeded(
          curEventIdSolver, conflictPair, setOcc, waitOcc, parentLCALoopOp,
          parentLCALoopOcc, parentLCALoopBeforePHOcc, parentLCALoopAfterPHOcc,
          barrierOp, corePipeSrc, corePipeDst, extraConflictPairs)) {
    return false;
  }
  return insertBoundaryBackwardConflictPairsIfNeeded(
      curEventIdSolver, conflictPair, setOcc, waitOcc, normScopeOcc1,
      normScopeOcc2, parentLCALoopOcc, parentLCALoopBeforePHOcc,
      parentLCALoopAfterPHOcc, barrierOp, corePipeSrc, corePipeDst,
      extraConflictPairs);
}

void Solver::recordChosenConflictPair(ConflictPair *conflictPair,
                                      Occurrence *normScopeOcc1,
                                      Occurrence *normScopeOcc2,
                                      const Occurrence *parentLCALoopOcc) {
  bool dontInsert = false;
  if (conflictPair->isInnerBackward && normScopeOcc1 != normScopeOcc2) {
    auto *parCond = OperationBase::getParentCondition(conflictPair->setOp);
    if (auto *conditionOp = llvm::dyn_cast_if_present<Condition>(parCond)) {
      if (parentLCALoopOcc->op->isProperAncestor(conditionOp)) {
        scopeOccPairChosenConflicts[{normScopeOcc1, normScopeOcc2}].insert(
            conflictPair);
        dontInsert = true;
      }
    }
  }
  if (dontInsert)
    return;
  PTO_SYNC_SOLVER_CHECK(parentLCALoopOcc != nullptr || normScopeOcc1 == normScopeOcc2);
  scopeOccChosenConflicts[normScopeOcc1].insert(conflictPair);
  scopeOccChosenConflicts[normScopeOcc2].insert(conflictPair);
}

void Solver::appendExtraConflictPairs(ExtraConflictPairs &extraConflictPairs) {
  for (auto &[extraConflictPair, parentScope] : extraConflictPairs) {
    scopeOccChosenConflicts[parentScope].insert(extraConflictPair.get());
    chosenConflictedPairs.push_back(std::move(extraConflictPair));
  }
}

void Solver::handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst,
                                   EventIdInfo eventIdInfo, bool isUseless) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(corePipeSrc != corePipeDst);

  Loop *parentLCALoopOp{nullptr};
  Occurrence *parentLCALoopOcc{nullptr};
  Occurrence *parentLCALoopBeforePHOcc{nullptr};
  Occurrence *parentLCALoopAfterPHOcc{nullptr};
  Occurrence *setOcc = nullptr;
  Occurrence *waitOcc = nullptr;
  Occurrence *normScopeOcc1 = nullptr;
  Occurrence *normScopeOcc2 = nullptr;
  OperationBase *normScopeOp = nullptr;
  auto conflictPair = createSetWaitConflictPair(
      occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst, eventIdInfo,
      isUseless, setOcc, waitOcc, normScopeOcc1, normScopeOcc2, normScopeOp,
      parentLCALoopOp, parentLCALoopOcc, parentLCALoopBeforePHOcc,
      parentLCALoopAfterPHOcc);
  normalizeSetWaitConflictPair(conflictPair.get(), occ1, occ2, setOcc, waitOcc,
                               corePipeSrc, corePipeDst);
  auto &curEventIdSolver = getEventIdSolverRef(
      conflictPair->setCorePipeInfo.pipe, conflictPair->waitCorePipeInfo.pipe);
  if (!prepareSetWaitConflictPairWithEventSolver(
          *curEventIdSolver, conflictPair.get(), occ1, occ2, normScopeOcc1,
          normScopeOcc2, normScopeOp, occ2->op, corePipeSrc, corePipeDst)) {
    return;
  }
  ExtraConflictPairs extraConflictPairs;
  if (!collectBackwardConflictExtras(
          *curEventIdSolver, conflictPair.get(), setOcc, waitOcc,
          normScopeOcc1, normScopeOcc2, parentLCALoopOp, parentLCALoopOcc,
          parentLCALoopBeforePHOcc, parentLCALoopAfterPHOcc, occ2->op,
          corePipeSrc, corePipeDst, extraConflictPairs)) {
    return;
  }
  recordChosenConflictPair(conflictPair.get(), normScopeOcc1, normScopeOcc2,
                           parentLCALoopOcc);
  memorizeSyncedPair(conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
  appendExtraConflictPairs(extraConflictPairs);
  curEventIdSolver->clearActionStack();
}

void Solver::handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst, bool isUseless) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);

  PTO_SYNC_SOLVER_CHECK(corePipeSrc == corePipeDst);
  if (corePipeSrc.pipe == pto::PIPE::PIPE_S) {
    return;
  }
  if (options.isRegBasedArch) {
    if (corePipeSrc.pipe == pto::PIPE::PIPE_V ||
        corePipeSrc.pipe == pto::PIPE::PIPE_M) {
      return;
    }
  }
  auto *waitOcc = getBarrierWaitOcc(occ1, occ2);

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, waitOcc->op, waitOcc->op, waitOcc, waitOcc, corePipeSrc,
      corePipeDst, waitOcc->startIndex, waitOcc->startIndex);
  conflictPair->isUseless = isUseless;
  PTO_SYNC_SOLVER_CHECK(conflictPair->startIndex <= conflictPair->endIndex);

  LLVM_DEBUG({ llvm::dbgs() << conflictPair->str() << '\n'; });

  auto *normScopeOcc = waitOcc->parentOcc;
  scopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

void Solver::handleUnitFlagConflict(Occurrence *occ1, Occurrence *occ2,
                                    CorePipeInfo corePipeSrc,
                                    CorePipeInfo corePipeDst,
                                    UnitFlagInfo unitFlagInfo, bool isUseless) {
  PTO_SYNC_SOLVER_CHECK(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  PTO_SYNC_SOLVER_CHECK(rwOp1 != nullptr && rwOp2 != nullptr);
  PTO_SYNC_SOLVER_CHECK(corePipeSrc != corePipeDst);

  auto *setOcc = occ1;
  auto *waitOcc = occ2;
  auto *normScopeOcc1 = setOcc->parentOcc;
  auto *normScopeOcc2 = waitOcc->parentOcc;

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  PTO_SYNC_SOLVER_CHECK(conflictPair->startIndex <= conflictPair->endIndex);

  conflictPair->isUseless = true;
  conflictPair->dontReuse = true;
  conflictPair->replacedWithUnitFlag = true;
  conflictPair->dontCheckForConflict = true;
  conflictPair->isInnerBackward = isBackwardSync(setOcc, waitOcc);

#ifndef NDEBUG
  Occurrence *parentLCALoopOcc{nullptr};
  if (conflictPair->isInnerBackward) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    PTO_SYNC_SOLVER_CHECK(parOcc1 != nullptr && parOcc2 != nullptr);
    parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    PTO_SYNC_SOLVER_CHECK(parentLCALoopOcc != nullptr);
  }

  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr) {
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    }
  });
#endif

  occ1->unitFlagInfo.merge(unitFlagInfo, occ1, occ2,
                           /*asSet=*/true, /*asWait=*/false);
  occ2->unitFlagInfo.merge(unitFlagInfo, occ1, occ2,
                           /*asSet=*/false, /*asWait=*/true);
  if (!isUseless) {
    rwOp1->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/true,
                                    /*asWait=*/false);
    rwOp2->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/false,
                                    /*asWait=*/true);
  }

  scopeOccPairChosenConflicts[{normScopeOcc1, normScopeOcc2}].insert(
      conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

void Solver::handleConflict(Occurrence *occ1, Occurrence *occ2,
                            const RWOperation *rwOp1,
                            const RWOperation *rwOp2,
                            CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                            EventIdInfo eventIdInfo, bool isUseless) {
  if (!checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo)) {
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "conflict found: " << "eventIdNum("
                 << eventIdInfo.eventIdNum << ")\n";
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << rwOp1->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << rwOp2->str(0, false) << '\n';
  });
  if (corePipeSrc == corePipeDst) {
    handleBarrierConflict(occ1, occ2, corePipeSrc, corePipeDst, isUseless);
  } else if (auto unitFlagInfo = checkUnitFlagPatterns(occ1, occ2)) {
    handleUnitFlagConflict(occ1, occ2, corePipeSrc, corePipeDst,
                           unitFlagInfo.value(), isUseless);
  } else {
    handleSetWaitConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo,
                          isUseless);
  }
}
