// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <climits>
#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "BufidSyncIdAlloc.h"


#define DEBUG_TYPE "pto-bufid-sync"

namespace {
constexpr size_t kMinimumReusableIdGroupSize = 2;

// A get_buf/rls_buf occurrence, ordered by (syncIRIndex, phase, ...) for the
// nesting check. phase 0 = pipeBefore (get), phase 1 = pipeAfter (rls).
struct Event {
  unsigned syncIRIndex;
  unsigned phase;
  mlir::pto::BufSyncType type;
  int logicId;
  int physicalId;
  mlir::pto::PipelineType pipe;
};

// Flatten every sync in op2BufSync into `events`, resolving each logic id to a
// physical id. Returns false (and fills `error`) if any logic id is unmapped.
static bool collectSyncEvents(
    const mlir::DenseMap<mlir::Operation *, mlir::pto::BufSyncPipeBuild>
        &op2BufSync,
    const mlir::DenseMap<int, int> &logicToPhysical,
    mlir::SmallVector<Event> &events, std::string *error) {
  auto appendEvents = [&events, &logicToPhysical, error](const auto &syncs,
                                                         unsigned phase) -> bool {
    for (auto &sync : syncs) {
      auto it = logicToPhysical.find(sync.logicId);
      if (it == logicToPhysical.end()) {
        if (error) {
          *error = "missing physical bufid for logic id " +
                   std::to_string(sync.logicId);
        }
        return false;
      }
      events.push_back({sync.syncIRIndex, phase, sync.type, sync.logicId,
                        it->second, sync.pipe});
    }
    return true;
  };
  for (auto &[op, build] : op2BufSync) {
    (void)op;
    if (!appendEvents(build.pipeBefore, 0) ||
        !appendEvents(build.pipeAfter, 1)) {
      return false;
    }
  }
  return true;
}

// Scan ordered events, enforcing that a physical bufid is never re-acquired
// while still active. Returns false (and fills `error`) on any violation.
static bool checkNoNestedGetBuf(const mlir::SmallVector<Event> &events,
                                std::string *error) {
  mlir::DenseMap<int, Event> activeByPhysicalId;
  for (const Event &event : events) {
    if (event.type == mlir::pto::BufSyncType::GET_BUF) {
      auto activeIt = activeByPhysicalId.find(event.physicalId);
      if (activeIt != activeByPhysicalId.end()) {
        if (error) {
          const Event &active = activeIt->second;
          *error = "nested get_buf for physical bufid " +
                   std::to_string(event.physicalId) + " at SyncIR " +
                   std::to_string(event.syncIRIndex) +
                   " while logic id " + std::to_string(active.logicId) +
                   " from SyncIR " + std::to_string(active.syncIRIndex) +
                   " is still active";
        }
        return false;
      }
      activeByPhysicalId[event.physicalId] = event;
      continue;
    }

    auto activeIt = activeByPhysicalId.find(event.physicalId);
    if (activeIt == activeByPhysicalId.end()) {
      if (error) {
        *error = "rls_buf without active get_buf for physical bufid " +
                 std::to_string(event.physicalId) + " at SyncIR " +
                 std::to_string(event.syncIRIndex);
      }
      return false;
    }
    activeByPhysicalId.erase(activeIt);
  }

  if (!activeByPhysicalId.empty()) {
    if (error) {
      auto activeIt = activeByPhysicalId.begin();
      const Event &active = activeIt->second;
      *error = "unclosed get_buf for physical bufid " +
               std::to_string(active.physicalId) + " from SyncIR " +
               std::to_string(active.syncIRIndex);
    }
    return false;
  }
  return true;
}
}

using namespace mlir;
using namespace mlir::pto;

void BufidSyncIdAlloc::collectPipeSignature(
    int logicId, SmallVector<PipelineType> &pipes) const {
  DenseSet<PipelineType> seen;
  for (auto &[op, build] : op2BufSync_) {
    for (auto &s : build.pipeBefore) {
      if (s.logicId == logicId && seen.insert(s.pipe).second) {
        pipes.push_back(s.pipe);
      }
    }
    for (auto &s : build.pipeAfter) {
      if (s.logicId == logicId && seen.insert(s.pipe).second) {
        pipes.push_back(s.pipe);
      }
    }
  }
}

unsigned BufidSyncIdAlloc::getOutermostLoopBegin(Operation *op) const {
  unsigned begin = UINT_MAX;
  Operation *current = op;
  while (auto loop = current->getParentOfType<LoopLikeOpInterface>()) {
    Operation *loopOp = loop.getOperation();
    for (auto &e : syncIR_) {
      auto *loop = dyn_cast<LoopInstanceElement>(e.get());
      if (loop && loop->getLoopKind() == KindOfLoop::LOOP_BEGIN &&
          loop->elementOp == loopOp) {
        begin = std::min(begin, loop->beginId);
        break;
      }
    }
    current = loopOp;
  }
  return begin;
}

unsigned BufidSyncIdAlloc::getOutermostLoopEnd(Operation *op) const {
  unsigned end = 0;
  Operation *current = op;
  while (auto loop = current->getParentOfType<LoopLikeOpInterface>()) {
    Operation *loopOp = loop.getOperation();
    for (auto &e : syncIR_) {
      auto *loop = dyn_cast<LoopInstanceElement>(e.get());
      if (loop && loop->getLoopKind() == KindOfLoop::LOOP_END &&
          loop->elementOp == loopOp) {
        end = std::max(end, loop->endId);
        break;
      }
    }
    current = loopOp;
  }
  return end;
}

void BufidSyncIdAlloc::computeLifeIntervals() {
  DenseMap<int, unsigned> logicIdStartPos;
  DenseMap<int, unsigned> logicIdEndPos;

  for (auto &[op, build] : op2BufSync_) {
    unsigned loopBegin = getOutermostLoopBegin(op);
    unsigned loopEnd = getOutermostLoopEnd(op);
    bool inLoop = (loopBegin != UINT_MAX && loopEnd > 0);

    for (auto &s : build.pipeBefore) {
      unsigned pos = inLoop ? loopBegin : s.syncIRIndex;
      if (!logicIdStartPos.contains(s.logicId)) {
        logicIdStartPos[s.logicId] = pos;
      } else {
        logicIdStartPos[s.logicId] = std::min(logicIdStartPos[s.logicId], pos);
}
    }
    for (auto &s : build.pipeAfter) {
      unsigned pos = inLoop ? loopEnd : s.syncIRIndex;
      if (!logicIdEndPos.contains(s.logicId)) {
        logicIdEndPos[s.logicId] = pos;
      } else {
        logicIdEndPos[s.logicId] = std::max(logicIdEndPos[s.logicId], pos);
}
    }
  }

  for (auto &vbid : virtualBufIds_) {
    BufIdInterval interval;
    interval.logicId = vbid.logicId;

    auto itStart = logicIdStartPos.find(vbid.logicId);
    auto itEnd = logicIdEndPos.find(vbid.logicId);
    if (itStart != logicIdStartPos.end() && itEnd != logicIdEndPos.end()) {
      interval.startPos = itStart->second;
      interval.endPos = itEnd->second;
    } else {
      interval.startPos = 0;
      interval.endPos = 0;
    }

    collectPipeSignature(vbid.logicId, interval.pipes);
    intervals_.push_back(std::move(interval));
  }

  std::sort(intervals_.begin(), intervals_.end(),
            [](const BufIdInterval &a, const BufIdInterval &b) {
              return a.startPos < b.startPos;
            });

  if (debugEnabled_) {
    printLifeIntervals(llvm::outs(), intervals_);
  }
}

void BufidSyncIdAlloc::linearScanAllocate() {
  SmallVector<unsigned> active;
  std::set<int> freeIds;

  int nextPhysicalId = 0;
  maxPhysicalIdUsed_ = -1;

  for (unsigned intervalIdx = 0; intervalIdx < intervals_.size(); ++intervalIdx) {
    auto &interval = intervals_[intervalIdx];
    SmallVector<unsigned> newActive;

    for (unsigned idx : active) {
      if (intervals_[idx].endPos < interval.startPos) {
        auto it = logicToPhysical_.find(intervals_[idx].logicId);
        if (it != logicToPhysical_.end()) {
          freeIds.insert(it->second);
        }
      } else {
        newActive.push_back(idx);
      }
    }
    active = newActive;

    int physicalId;
    if (!freeIds.empty()) {
      auto freeIt = freeIds.begin();
      physicalId = *freeIt;
      freeIds.erase(freeIt);
    } else {
      physicalId = nextPhysicalId++;
    }

    logicToPhysical_[interval.logicId] = physicalId;
    maxPhysicalIdUsed_ = std::max(maxPhysicalIdUsed_, physicalId);
    active.push_back(intervalIdx);
  }

  if (debugEnabled_) {
    llvm::outs() << "[bufid_sync] LinearScan result: maxPhysicalIdUsed="
                 << maxPhysicalIdUsed_ << "\n";
    printLogicToPhysical(llvm::outs(), logicToPhysical_, "LinearScan logicId->physicalId");
  }
}

void BufidSyncIdAlloc::compactPhysicalIds() {
  DenseMap<int, unsigned> logicIdFirstPos;
  DenseSet<int> activeLogicIds;
  for (auto &[op, build] : op2BufSync_) {
    for (auto &s : build.pipeBefore) {
      activeLogicIds.insert(s.logicId);
      if (!logicIdFirstPos.contains(s.logicId)) {
        logicIdFirstPos[s.logicId] = s.syncIRIndex;
      } else {
        logicIdFirstPos[s.logicId] = std::min(logicIdFirstPos[s.logicId], s.syncIRIndex);
}
    }
    for (auto &s : build.pipeAfter) {
      activeLogicIds.insert(s.logicId);
      if (!logicIdFirstPos.contains(s.logicId)) {
        logicIdFirstPos[s.logicId] = s.syncIRIndex;
      } else {
        logicIdFirstPos[s.logicId] = std::min(logicIdFirstPos[s.logicId], s.syncIRIndex);
}
    }
  }

  SmallVector<int> logicIdsByPos;
  for (auto &[lid, pid] : logicToPhysical_) {
    if (activeLogicIds.contains(lid)) {
      logicIdsByPos.push_back(lid);
    }
  }
  std::sort(logicIdsByPos.begin(), logicIdsByPos.end(), [&](int a, int b) {
    return logicIdFirstPos[a] < logicIdFirstPos[b];
  });

  DenseMap<int, int> oldPidToNew;
  for (unsigned i = 0; i < logicIdsByPos.size(); ++i) {
    int lid = logicIdsByPos[i];
    int oldPid = logicToPhysical_[lid];
    if (!oldPidToNew.contains(oldPid)) {
      oldPidToNew[oldPid] = static_cast<int>(oldPidToNew.size());
    }
  }

  for (auto &[lid, pid] : logicToPhysical_) {
    if (oldPidToNew.contains(pid)) {
      pid = oldPidToNew[pid];
    }
  }

  maxPhysicalIdUsed_ = static_cast<int>(oldPidToNew.size()) - 1;

  if (debugEnabled_) {
    llvm::outs() << "[bufid_sync] After compactPhysicalIds: maxPhysicalIdUsed="
                 << maxPhysicalIdUsed_ << "\n";
    printLogicToPhysical(llvm::outs(), logicToPhysical_, "compactPhysicalIds logicId->physicalId");
  }
}

namespace {

// Encode a set of pipeline types as a sorted, comma-joined key string.
static std::string encodeSig(const SmallVector<PipelineType> &sig) {
  SmallVector<PipelineType> s = sig;
  std::sort(s.begin(), s.end());
  std::string key;
  for (auto p : s) {
    if (!key.empty()) {
      key += ",";
    }
    key += std::to_string(static_cast<int>(p));
  }
  return key;
}

// Inverse of encodeSig.
static SmallVector<PipelineType> decodeSig(const std::string &key) {
  SmallVector<PipelineType> pipes;
  std::string token;
  std::istringstream iss(key);
  while (std::getline(iss, token, ',')) {
    if (!token.empty()) {
      pipes.push_back(static_cast<PipelineType>(std::stoi(token)));
    }
  }
  return pipes;
}

// Lower score == more contended pipe (prefer merging on it).
static int getPipeScore(PipelineType p) {
  switch (p) {
  case PipelineType::PIPE_MTE2:
    return 1;
  case PipelineType::PIPE_MTE3:
  case PipelineType::PIPE_FIX:
    return 2;
  default:
    return 3;
  }
}

static int getMinPipeScore(const SmallVector<PipelineType> &pipes) {
  int minScore = 99;
  for (auto p : pipes) {
    minScore = std::min(minScore, getPipeScore(p));
  }
  return minScore;
}

// The pipe within a signature with the lowest score; used as the consecutivity
// check pipe.
static PipelineType pickCheckPipe(const SmallVector<PipelineType> &sigPipes) {
  PipelineType checkPipe = sigPipes[0];
  int minPipeScore = getPipeScore(sigPipes[0]);
  for (auto p : sigPipes) {
    if (getPipeScore(p) < minPipeScore) {
      minPipeScore = getPipeScore(p);
      checkPipe = p;
    }
  }
  return checkPipe;
}

static void sortLogicIdsByFirstPos(SmallVector<int> &ids,
                                   const DenseMap<int, unsigned> &firstPos) {
  std::sort(ids.begin(), ids.end(), [&firstPos](int a, int b) {
    auto itA = firstPos.find(a);
    auto itB = firstPos.find(b);
    if (itA == firstPos.end() || itB == firstPos.end()) {
      return a < b;
    }
    return itA->second < itB->second;
  });
}

// Pair off the group's logic ids into a target->donor merge map, then collapse
// donor chains so every target maps to a terminal donor.
static DenseMap<int, int> buildMergeMap(const SmallVector<int> &groupIds,
                                        bool consecutive) {
  DenseMap<int, int> mergeMap;
  unsigned halfSize = groupIds.size() / 2;
  if (consecutive) {
    for (unsigned i = 0; i < halfSize; ++i) {
      mergeMap[groupIds[2 * i + 1]] = groupIds[2 * i];
    }
  } else {
    for (unsigned i = 0; i < halfSize; ++i) {
      mergeMap[groupIds[i + halfSize]] = groupIds[i];
    }
  }
  for (auto &[lid, donorLid] : mergeMap) {
    while (mergeMap.contains(donorLid)) {
      donorLid = mergeMap[donorLid];
    }
  }
  return mergeMap;
}

static void debugSigGroups(bool enabled, int iteration,
                           const std::map<std::string, SmallVector<int>> &sigGroups) {
  if (!enabled) {
    return;
  }
  llvm::outs() << "[bufid_sync] reuseIds iteration=" << iteration
               << " sigGroups=" << sigGroups.size() << "\n";
  for (auto &[sigKey, ids] : sigGroups) {
    llvm::outs() << "  sig=" << sigKey << " count=" << ids.size() << "\n";
  }
}

static void debugSelection(bool enabled, const std::string &bestSigKey,
                           size_t groupSize, PipelineType checkPipe,
                           bool consecutive) {
  if (!enabled) {
    return;
  }
  llvm::outs() << "[bufid_sync] reuseIds: bestSig=" << bestSigKey
               << " groupSize=" << groupSize
               << " checkPipe=" << static_cast<int>(checkPipe)
               << " consecutive=" << consecutive << "\n";
}

static void debugIterationResult(bool enabled, int iteration,
                                 int maxPhysicalIdUsed,
                                 const DenseMap<int, int> &mergeMap,
                                 const DenseMap<int, int> &logicToPhysical) {
  if (!enabled) {
    return;
  }
  llvm::outs() << "[bufid_sync] reuseIds: iteration=" << iteration
               << " maxPhysicalIdUsed=" << maxPhysicalIdUsed << "\n";
  for (auto &[lid, donorLid] : mergeMap) {
    llvm::outs() << "  merge logicId=" << lid << " -> donor logicId="
                 << donorLid << " physicalId=" << logicToPhysical.lookup(lid)
                 << "\n";
  }
}

static void debugReuseBreak(bool enabled, const char *msg) {
  if (enabled) {
    llvm::outs() << msg;
  }
}

} // namespace

void BufidSyncIdAlloc::collectLogicIdPipes(
    DenseMap<int, SmallVector<PipelineType>> &logicIdPipes,
    DenseMap<int, unsigned> &logicIdFirstPos) const {
  DenseMap<int, DenseSet<PipelineType>> logicIdSeenPipes;
  auto accumulate = [&logicIdPipes, &logicIdFirstPos,
                     &logicIdSeenPipes](const auto &syncs) {
    for (auto &s : syncs) {
      if (logicIdSeenPipes[s.logicId].insert(s.pipe).second) {
        logicIdPipes[s.logicId].push_back(s.pipe);
      }
      if (!logicIdFirstPos.contains(s.logicId)) {
        logicIdFirstPos[s.logicId] = s.syncIRIndex;
      } else {
        logicIdFirstPos[s.logicId] =
            std::min(logicIdFirstPos[s.logicId], s.syncIRIndex);
      }
    }
  };
  for (auto &[op, build] : op2BufSync_) {
    (void)op;
    accumulate(build.pipeBefore);
    accumulate(build.pipeAfter);
  }
}

std::string BufidSyncIdAlloc::selectBestSigGroup(
    const std::map<std::string, SmallVector<int>> &sigGroups) const {
  int bestScore = -1;
  std::string bestSigKey;
  for (auto &[sigKey, ids] : sigGroups) {
    if (ids.size() < kMinimumReusableIdGroupSize) {
      continue;
    }
    SmallVector<PipelineType> sigPipes = decodeSig(sigKey);
    int pipeScore = getMinPipeScore(sigPipes);
    int idNum = static_cast<int>(ids.size());
    int score = pipeScore * idNum * idNum;
    if (score > bestScore) {
      bestScore = score;
      bestSigKey = sigKey;
    }
  }
  return bestSigKey;
}

bool BufidSyncIdAlloc::isConsecutiveOnPipe(const SmallVector<int> &ids,
                                           PipelineType pipe) const {
  SmallVector<std::pair<unsigned, unsigned>> idRanges;
  for (int lid : ids) {
    unsigned minIdx = UINT_MAX, maxIdx = 0;
    for (auto &[op, build] : op2BufSync_) {
      (void)op;
      for (auto &s : build.pipeBefore) {
        if (s.logicId == lid && s.pipe == pipe) {
          minIdx = std::min(minIdx, s.syncIRIndex);
          maxIdx = std::max(maxIdx, s.syncIRIndex);
        }
      }
      for (auto &s : build.pipeAfter) {
        if (s.logicId == lid && s.pipe == pipe) {
          minIdx = std::min(minIdx, s.syncIRIndex);
          maxIdx = std::max(maxIdx, s.syncIRIndex);
        }
      }
    }
    if (minIdx != UINT_MAX) {
      idRanges.push_back({minIdx, maxIdx});
    }
  }
  std::sort(idRanges.begin(), idRanges.end());
  for (unsigned i = 1; i < idRanges.size(); ++i) {
    if (idRanges[i].first <= idRanges[i - 1].second) {
      return false;
    }
  }
  for (unsigned i = 1; i < idRanges.size(); ++i) {
    unsigned gapStart = idRanges[i - 1].second;
    unsigned gapEnd = idRanges[i].first;
    for (auto &[op, build] : op2BufSync_) {
      (void)op;
      for (auto &s : build.pipeBefore) {
        if (s.syncIRIndex > gapStart && s.syncIRIndex < gapEnd) {
          return false;
        }
      }
      for (auto &s : build.pipeAfter) {
        if (s.syncIRIndex > gapStart && s.syncIRIndex < gapEnd) {
          return false;
        }
      }
    }
  }
  return true;
}

void BufidSyncIdAlloc::applyMerges(const DenseMap<int, int> &mergeMap) {
  DenseMap<Operation *, BufSyncPipeBuild> newOp2BufSync;
  for (auto &[op, build] : op2BufSync_) {
    BufSyncPipeBuild newBuild;
    remapSyncList(build.pipeBefore, mergeMap, newBuild.pipeBefore);
    remapSyncList(build.pipeAfter, mergeMap, newBuild.pipeAfter);
    newOp2BufSync[op] = std::move(newBuild);
  }
  op2BufSync_ = std::move(newOp2BufSync);

  for (auto &[lid, donorLid] : mergeMap) {
    logicToPhysical_[lid] = logicToPhysical_[donorLid];
  }

  virtualBufIds_.erase(
      std::remove_if(virtualBufIds_.begin(), virtualBufIds_.end(),
                     [&mergeMap](const VirtualBufId &vbid) {
                       return mergeMap.contains(vbid.logicId);
                     }),
      virtualBufIds_.end());

  for (auto &[lid, pid] : logicToPhysical_) {
    auto it = mergeMap.find(lid);
    if (it != mergeMap.end()) {
      pid = logicToPhysical_[it->second];
    }
  }

  compactPhysicalIds();
}

bool BufidSyncIdAlloc::reuseIdsStep(int iteration) {
  DenseMap<int, SmallVector<PipelineType>> logicIdPipes;
  DenseMap<int, unsigned> logicIdFirstPos;
  collectLogicIdPipes(logicIdPipes, logicIdFirstPos);

  std::map<std::string, SmallVector<int>> sigGroups;
  for (auto &[lid, pipes] : logicIdPipes) {
    sigGroups[encodeSig(pipes)].push_back(lid);
  }
  debugSigGroups(debugEnabled_, iteration, sigGroups);

  std::string bestSigKey = selectBestSigGroup(sigGroups);
  if (bestSigKey.empty()) {
    debugReuseBreak(
        debugEnabled_,
        "[bufid_sync] reuseIds: no group with >=2 IDs to reuse, breaking\n");
    return false;
  }

  auto &groupIds = sigGroups[bestSigKey];
  sortLogicIdsByFirstPos(groupIds, logicIdFirstPos);

  SmallVector<PipelineType> bestSigPipes = decodeSig(bestSigKey);
  PipelineType checkPipe = pickCheckPipe(bestSigPipes);
  bool consecutive = isConsecutiveOnPipe(groupIds, checkPipe);
  debugSelection(debugEnabled_, bestSigKey, groupIds.size(), checkPipe,
                 consecutive);

  if (groupIds.size() / 2 == 0) {
    debugReuseBreak(debugEnabled_,
                    "[bufid_sync] reuseIds: halfSize=0, breaking\n");
    return false;
  }

  DenseMap<int, int> mergeMap = buildMergeMap(groupIds, consecutive);
  applyMerges(mergeMap);
  debugIterationResult(debugEnabled_, iteration, maxPhysicalIdUsed_, mergeMap,
                       logicToPhysical_);

  if (mergeMap.empty()) {
    debugReuseBreak(debugEnabled_,
                    "[bufid_sync] reuseIds: no merges in this iteration, breaking\n");
    return false;
  }
  return true;
}

void BufidSyncIdAlloc::reuseIds() {
  int iteration = 0;
  while (maxPhysicalIdUsed_ >= static_cast<int>(physicalBufIdCount_)) {
    ++iteration;
    if (!reuseIdsStep(iteration)) {
      break;
    }
  }

  for (auto &vbid : virtualBufIds_) {
    if (!logicToPhysical_.contains(vbid.logicId)) {
      logicToPhysical_[vbid.logicId] = 0;
    }
  }
}

bool BufidSyncIdAlloc::validateNoSamePhysicalIdNesting(
    std::string *error) const {
  SmallVector<Event> events;
  if (!collectSyncEvents(op2BufSync_, logicToPhysical_, events, error)) {
    return false;
  }

  std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
    return std::make_tuple(a.syncIRIndex, a.phase,
                           static_cast<int>(a.type), a.physicalId,
                           a.logicId, static_cast<int>(a.pipe)) <
           std::make_tuple(b.syncIRIndex, b.phase,
                           static_cast<int>(b.type), b.physicalId,
                           b.logicId, static_cast<int>(b.pipe));
  });

  return checkNoNestedGetBuf(events, error);
}
