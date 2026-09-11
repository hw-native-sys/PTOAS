// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <functional>
#include <map>
#include <tuple>

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "BufidSyncAnalysis.h"


#define DEBUG_TYPE "pto-bufid-sync"

using namespace mlir;
using namespace mlir::pto;

namespace {

// insertSyncOperations() helper: append a sync to `list` unless one with the
// same (logicId, pipe) is already present.
void addSyncIfAbsent(SmallVector<BufSyncOperation> &list, BufSyncType type,
                     PipelineType pipe, int logicId, unsigned syncIRIndex,
                     unsigned depSyncIRIndex) {
  for (auto &s : list) {
    if (s.logicId == logicId && s.pipe == pipe) {
      return;
    }
  }
  BufSyncOperation op;
  op.type = type;
  op.pipe = pipe;
  op.logicId = logicId;
  op.syncIRIndex = syncIRIndex;
  op.depSyncIRIndex = depSyncIRIndex;
  list.push_back(op);
}

// mergeGetRls() helpers. Cancel/get state is tracked per region in an RlsMap
// keyed by (logicId, pipe).
using RlsKey = std::pair<int, int>;
using RlsMap = DenseMap<RlsKey, Operation *>;

// Try to cancel a pending get_buf against a matching rls_buf recorded in
// `rlsMap`. On a match, erase the paired rls op and return true (the get should
// be dropped). Otherwise drop any stale same-logicId/different-pipe entries and
// return false.
bool tryCancelGet(const BufSyncOperation &sync, RlsMap &rlsMap,
                  DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync,
                  unsigned &cancelCount) {
  RlsKey key(sync.logicId, static_cast<int>(sync.pipe));
  auto mapIt = rlsMap.find(key);
  if (mapIt != rlsMap.end()) {
    Operation *rlsOp = mapIt->second;
    auto rlsIt = op2BufSync.find(rlsOp);
    if (rlsIt != op2BufSync.end()) {
      auto &rlsPipeAfter = rlsIt->second.pipeAfter;
      rlsPipeAfter.erase(
          std::remove_if(rlsPipeAfter.begin(), rlsPipeAfter.end(),
                         [&sync](const BufSyncOperation &s) {
                           return s.type == BufSyncType::RLS_BUF &&
                                  s.logicId == sync.logicId &&
                                  s.pipe == sync.pipe;
                         }),
          rlsPipeAfter.end());
    }
    rlsMap.erase(mapIt);
    ++cancelCount;
    return true;
  }
  SmallVector<RlsKey> keysToErase;
  for (auto &[existingKey, rlsOp] : rlsMap) {
    if (existingKey.first == sync.logicId &&
        existingKey.second != static_cast<int>(sync.pipe)) {
      keysToErase.push_back(existingKey);
    }
  }
  for (auto &k : keysToErase) {
    rlsMap.erase(k);
  }
  return false;
}

// Record every rls_buf in `build.pipeAfter` into `rlsMap` for later matching.
void recordRls(Operation &op, BufSyncPipeBuild &build, RlsMap &rlsMap,
               bool debugEnabled) {
  for (auto &sync : build.pipeAfter) {
    if (sync.type != BufSyncType::RLS_BUF) {
      continue;
    }
    RlsKey key(sync.logicId, static_cast<int>(sync.pipe));
    rlsMap[key] = &op;
    if (debugEnabled) {
      llvm::outs() << "  ADD to map: RLS_BUF(pipe=" << static_cast<int>(sync.pipe)
                   << ", lid=" << sync.logicId << ")\n";
    }
  }
}

// Cancel matching get/rls pairs for a single op: filter pipeBefore, then record
// this op's pipeAfter releases.
void cancelBuildGetRls(Operation &op, BufSyncPipeBuild &build, RlsMap &rlsMap,
                       DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync,
                       bool debugEnabled, unsigned &cancelCount) {
  SmallVector<BufSyncOperation> newPipeBefore;
  for (auto &sync : build.pipeBefore) {
    if (sync.type == BufSyncType::GET_BUF &&
        tryCancelGet(sync, rlsMap, op2BufSync, cancelCount)) {
      continue;
    }
    newPipeBefore.push_back(sync);
  }
  build.pipeBefore = std::move(newPipeBefore);
  recordRls(op, build, rlsMap, debugEnabled);
}

// Walk a block cancelling get/rls pairs. scf::For/While/If regions are each
// processed with a fresh map and do not carry linear-scan state across the loop
// boundary: the loop may execute zero or many times, so an outer linear state
// would be unsound.
void cancelGetRlsInBlock(Block *block, RlsMap &rlsMap,
                         DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync,
                         bool debugEnabled, unsigned &cancelCount) {
  for (auto &op : *block) {
    auto it = op2BufSync.find(&op);
    if (it != op2BufSync.end()) {
      cancelBuildGetRls(op, it->second, rlsMap, op2BufSync, debugEnabled,
                        cancelCount);
    }

    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op)) {
      for (auto &region : op.getRegions()) {
        RlsMap freshMap;
        for (auto &subBlock : region.getBlocks()) {
          cancelGetRlsInBlock(&subBlock, freshMap, op2BufSync, debugEnabled,
                              cancelCount);
        }
      }
      rlsMap.clear();
    }
  }
}

} // namespace

// Shared with BufidSyncIdAlloc.cpp: remap each sync's logicId through mergeMap
// and drop duplicates by (pipe, remapped logicId).
void mlir::pto::remapSyncList(const SmallVector<BufSyncOperation> &in,
                              const DenseMap<int, int> &mergeMap,
                              SmallVector<BufSyncOperation> &out) {
  DenseSet<std::pair<int, int>> seen;
  for (auto &s : in) {
    int newLogicId = s.logicId;
    auto it = mergeMap.find(newLogicId);
    if (it != mergeMap.end()) {
      newLogicId = it->second;
    }
    auto key = std::make_pair(static_cast<int>(s.pipe), newLogicId);
    if (seen.insert(key).second) {
      BufSyncOperation newS = s;
      newS.logicId = newLogicId;
      out.push_back(newS);
    }
  }
}

bool BufidSyncAnalysis::isGMDependency(
    const SmallVector<const BaseMemInfo *> &tiles) const {
  for (auto *tile : tiles) {
    if (tile->scope == pto::AddressSpace::GM) {
      return true;
    }
  }
  return false;
}

bool BufidSyncAnalysis::isSamePipe(const CompoundInstanceElement *a,
                                   const CompoundInstanceElement *b) const {
  return a->kPipeValue == b->kPipeValue;
}

bool BufidSyncAnalysis::detectDependency(CompoundInstanceElement *src,
                                         CompoundInstanceElement *dst,
                                         DepBaseMemInfoPairVec &out) const {
  if (memAnalyzer_.DepBetween(src->defVec, dst->useVec, out)) {
    return true;
  }
  out.clear();
  if (memAnalyzer_.DepBetween(src->useVec, dst->defVec, out)) {
    return true;
  }
  out.clear();
  return memAnalyzer_.DepBetween(src->defVec, dst->defVec, out);
}

void BufidSyncAnalysis::collectDependencies() {
  SmallVector<CompoundInstanceElement *> compounds;
  for (auto &element : syncIR_) {
    if (auto *comp = dyn_cast<CompoundInstanceElement>(element.get())) {
      compounds.push_back(comp);
    }
  }
  if (debugEnabled_) {
    llvm::outs() << "[bufid_sync] collectDependencies: compounds count="
               << compounds.size() << "\n";
  }

  for (unsigned i = 0; i < compounds.size(); ++i) {
    for (unsigned j = i + 1; j < compounds.size(); ++j) {
      if (auto dp = tryBuildDepPair(compounds[i], compounds[j])) {
        depPairs_.push_back(std::move(*dp));
      }
    }
  }

  collectTilesFromDepPairs();

  if (debugEnabled_) {
    printDepPairs(llvm::outs(), depPairs_);
  }
}

std::optional<DepPair>
BufidSyncAnalysis::tryBuildDepPair(CompoundInstanceElement *src,
                                   CompoundInstanceElement *dst) {
  if (isSamePipe(src, dst)) {
    return std::nullopt;
  }

  DepBaseMemInfoPairVec depBaseMemInfosVec;
  if (!detectDependency(src, dst, depBaseMemInfosVec)) {
    return std::nullopt;
  }

  SmallVector<const BaseMemInfo *> depTiles;
  for (auto &pair : depBaseMemInfosVec) {
    if (pair.first->scope == pto::AddressSpace::GM ||
        pair.second->scope == pto::AddressSpace::GM) {
      continue;
    }
    if (pair.first->scope != pair.second->scope) {
      continue;
    }
    depTiles.push_back(pair.first);
    depTiles.push_back(pair.second);
  }

  if (depTiles.empty()) {
    return std::nullopt;
  }

  DepPair dp;
  dp.srcElement = src;
  dp.dstElement = dst;
  dp.depTiles = depTiles;
  dp.srcPipe = src->kPipeValue;
  dp.dstPipe = dst->kPipeValue;
  return dp;
}

void BufidSyncAnalysis::collectRawTiles() {
  DenseSet<Value> seenBaseBuffers;
  for (auto &dp : depPairs_) {
    for (auto *memInfo : dp.depTiles) {
      Value baseBuf = memInfo->baseBuffer;
      if (!seenBaseBuffers.insert(baseBuf).second) {
        continue;
      }
      TileInfo ti;
      ti.memInfo = memInfo;
      ti.scope = memInfo->scope;
      ti.baseAddr =
          memInfo->baseAddresses.empty() ? 0 : memInfo->baseAddresses[0];
      ti.size = memInfo->allocateSize;
      ti.tileValue = baseBuf;
      Value rootBuf = memInfo->rootBuffer;
      if (rootBuf && rootBuf.getDefiningOp() &&
          rootBuf.getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>()) {
        rootBuf = baseBuf;
      }
      ti.rootBuffer = rootBuf;
      allTiles_.push_back(ti);
    }
  }
}

DenseSet<unsigned> BufidSyncAnalysis::computeRemovedTileIndices() const {
  std::map<std::tuple<int, uint64_t, uint64_t>, SmallVector<unsigned>> keyToIndices;
  for (unsigned i = 0; i < allTiles_.size(); ++i) {
    auto &t = allTiles_[i];
    auto key = std::make_tuple(static_cast<int>(t.scope), t.baseAddr, t.size);
    keyToIndices[key].push_back(i);
  }

  DenseSet<unsigned> removed;
  for (auto &[key, indices] : keyToIndices) {
    if (indices.size() <= 1) {
      continue;
    }
    DenseMap<Value, unsigned> rootToFirst;
    for (unsigned idx : indices) {
      Value root = allTiles_[idx].rootBuffer;
      auto it = rootToFirst.find(root);
      if (it != rootToFirst.end()) {
        removed.insert(idx);
      } else {
        rootToFirst[root] = idx;
      }
    }
  }

  for (auto &[key, indices] : keyToIndices) {
    if (indices.size() <= 1) {
      continue;
    }
    for (unsigned k = 1; k < indices.size(); ++k) {
      if (removed.contains(indices[k])) {
        continue;
      }
      auto &a = allTiles_[indices[0]];
      auto &b = allTiles_[indices[k]];
      if (a.scope == b.scope && a.baseAddr == b.baseAddr && a.size == b.size &&
          memAnalyzer_.MemAlias(a.memInfo, b.memInfo)) {
        removed.insert(indices[k]);
      }
    }
  }
  return removed;
}

void BufidSyncAnalysis::dedupTiles() {
  DenseSet<unsigned> removed = computeRemovedTileIndices();
  if (removed.empty()) {
    return;
  }
  SmallVector<TileInfo> deduped;
  for (unsigned i = 0; i < allTiles_.size(); ++i) {
    if (!removed.contains(i)) {
      deduped.push_back(std::move(allTiles_[i]));
    }
  }
  if (debugEnabled_) {
    llvm::outs() << "[bufid_sync] allTiles dedup: " << allTiles_.size()
               << " -> " << deduped.size() << "\n";
  }
  allTiles_ = std::move(deduped);
}

void BufidSyncAnalysis::collectTilesFromDepPairs() {
  collectRawTiles();
  dedupTiles();
  if (debugEnabled_) {
    printAllTiles(llvm::outs(), allTiles_);
  }
}

void BufidSyncAnalysis::classifyTiles() {
  if (allTiles_.empty()) {
    return;
  }

  DenseMap<pto::AddressSpace, SmallVector<unsigned>> spaceGroups;
  for (unsigned i = 0; i < allTiles_.size(); ++i) {
    spaceGroups[allTiles_[i].scope].push_back(i);
  }

  for (auto &[space, indices] : spaceGroups) {
    unsigned n = indices.size();
    SmallVector<unsigned> parent(n);
    for (unsigned i = 0; i < n; ++i) {
      parent[i] = i;
    }

    auto find = [&parent](unsigned x) -> unsigned {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };

    auto unite = [&find, &parent](unsigned a, unsigned b) {
      a = find(a);
      b = find(b);
      if (a != b) {
        parent[a] = b;
      }
    };

    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = i + 1; j < n; ++j) {
        if (memAnalyzer_.MemAlias(allTiles_[indices[i]].memInfo,
                                  allTiles_[indices[j]].memInfo)) {
          unite(i, j);
        }
      }
    }

    DenseMap<unsigned, SmallVector<unsigned>> components;
    for (unsigned i = 0; i < n; ++i) {
      components[find(i)].push_back(indices[i]);
    }

    for (auto &[root, tileIndices] : components) {
      tileGroups_.push_back(tileIndices);
    }
  }

  if (debugEnabled_) {
    printTileGroups(llvm::outs(), tileGroups_, allTiles_);
  }
}

void BufidSyncAnalysis::bronKerbosch(
    const SmallVector<unsigned> &R, SmallVector<unsigned> P,
    SmallVector<unsigned> X,
    const SmallVector<SmallVector<bool>> &adjMatrix,
    SmallVector<SmallVector<unsigned>> &maximalCliques) {
  if (P.empty() && X.empty()) {
    if (!R.empty()) {
      maximalCliques.push_back(R);
    }
    return;
  }

  if (maximalCliques.size() > mlir::pto::kValue100000) {
    llvm::errs() << "[bufid_sync] bronKerbosch WARNING: too many cliques ("
                 << maximalCliques.size() << "), aborting!\n";
    return;
  }

  unsigned pivot = P.empty() ? X[0] : P[0];
  SmallVector<unsigned> candidates;
  for (unsigned v : P) {
    if (!adjMatrix[pivot][v]) {
      candidates.push_back(v);
    }
  }

  for (unsigned v : candidates) {
    SmallVector<unsigned> newR = R;
    newR.push_back(v);

    SmallVector<unsigned> newP, newX;
    for (unsigned u : P) {
      if (adjMatrix[v][u]) {
        newP.push_back(u);
      }
    }
    for (unsigned u : X) {
      if (adjMatrix[v][u]) {
        newX.push_back(u);
      }
    }

    bronKerbosch(newR, newP, newX, adjMatrix, maximalCliques);

    auto it = std::find(P.begin(), P.end(), v);
    if (it != P.end()) {
      P.erase(it);
    }
    X.push_back(v);
  }
}

void BufidSyncAnalysis::allocateVirtualBufIds() {
  int nextLogicId = 0;

  for (unsigned gi = 0; gi < tileGroups_.size(); ++gi) {
    auto &group = tileGroups_[gi];
    unsigned n = group.size();
    if (n == 0) {
      continue;
    }

    SmallVector<SmallVector<bool>> adjMatrix(
        n, SmallVector<bool>(n, false));
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = i + 1; j < n; ++j) {
        if (memAnalyzer_.MemAlias(allTiles_[group[i]].memInfo,
                                  allTiles_[group[j]].memInfo)) {
          adjMatrix[i][j] = true;
          adjMatrix[j][i] = true;
        }
      }
    }

    SmallVector<unsigned> P;
    for (unsigned i = 0; i < n; ++i) {
      P.push_back(i);
    }

    SmallVector<SmallVector<unsigned>> maximalCliques;
    bronKerbosch({}, P, {}, adjMatrix, maximalCliques);
    if (debugEnabled_) {
      llvm::outs() << "[bufid_sync] allocateVirtualBufIds: group[" << gi
                 << "] bronKerbosch done, maximalCliques="
                 << maximalCliques.size() << "\n";
    }

    for (auto &clique : maximalCliques) {
      VirtualBufId vbid;
      vbid.logicId = nextLogicId++;
      vbid.scope = allTiles_[group[clique[0]]].scope;
      for (unsigned idx : clique) {
        vbid.tiles.push_back(allTiles_[group[idx]]);
      }
      virtualBufIds_.push_back(std::move(vbid));
    }
  }

  if (debugEnabled_) {
    printVirtualBufIds(llvm::outs(), virtualBufIds_);
  }
}

bool BufidSyncAnalysis::virtualBufIdContainsTile(const VirtualBufId &vbid,
                                                 const BaseMemInfo *tile) const {
  if (!tile) {
    return false;
  }
  if (vbid.scope != tile->scope) {
    return false;
  }
  for (auto &t : vbid.tiles) {
    bool ptrMatch = (t.memInfo == tile);
    bool valMatch = (t.tileValue == tile->baseBuffer);
    bool aliasMatch =
        (!ptrMatch && !valMatch && memAnalyzer_.MemAlias(t.memInfo, tile));
    if (ptrMatch || valMatch || aliasMatch) {
      return true;
    }
  }
  return false;
}

int BufidSyncAnalysis::findBestVirtualBufId(const BaseMemInfo *tile) const {
  int bestLogicId = -1;
  unsigned bestTileCount = 0;
  for (auto &vbid : virtualBufIds_) {
    if (virtualBufIdContainsTile(vbid, tile) &&
        (vbid.tiles.size() > bestTileCount ||
         (vbid.tiles.size() == bestTileCount &&
          (bestLogicId < 0 || vbid.logicId < bestLogicId)))) {
      bestLogicId = vbid.logicId;
      bestTileCount = vbid.tiles.size();
    }
  }
  return bestLogicId;
}

int BufidSyncAnalysis::findBestVirtualBufId(const DepPair &depPair) const {
  int bestLogicId = -1;
  unsigned bestMatchedTiles = 0;
  unsigned bestCliqueSize = 0;

  for (auto &vbid : virtualBufIds_) {
    unsigned matchedTiles = 0;
    for (auto *tile : depPair.depTiles) {
      if (virtualBufIdContainsTile(vbid, tile)) {
        ++matchedTiles;
      }
    }

    if (matchedTiles == 0) {
      continue;
    }

    bool isBetter =
        matchedTiles > bestMatchedTiles ||
        (matchedTiles == bestMatchedTiles &&
         vbid.tiles.size() > bestCliqueSize) ||
        (matchedTiles == bestMatchedTiles &&
         vbid.tiles.size() == bestCliqueSize &&
         (bestLogicId < 0 || vbid.logicId < bestLogicId));
    if (isBetter) {
      bestLogicId = vbid.logicId;
      bestMatchedTiles = matchedTiles;
      bestCliqueSize = vbid.tiles.size();
    }
  }

  return bestLogicId;
}

void BufidSyncAnalysis::insertSyncOperations() {
  for (auto &dp : depPairs_) {
    int bestLogicId = findBestVirtualBufId(dp);
    if (bestLogicId < 0) {
      continue;
    }

    Operation *srcOp = dp.srcElement->elementOp;
    Operation *dstOp = dp.dstElement->elementOp;
    unsigned srcIdx = dp.srcElement->GetIndex();
    unsigned dstIdx = dp.dstElement->GetIndex();

    addSyncIfAbsent(op2BufSync_[srcOp].pipeBefore, BufSyncType::GET_BUF,
                    dp.srcPipe, bestLogicId, srcIdx, dstIdx);
    addSyncIfAbsent(op2BufSync_[srcOp].pipeAfter, BufSyncType::RLS_BUF,
                    dp.srcPipe, bestLogicId, srcIdx, dstIdx);
    addSyncIfAbsent(op2BufSync_[dstOp].pipeBefore, BufSyncType::GET_BUF,
                    dp.dstPipe, bestLogicId, dstIdx, srcIdx);
    addSyncIfAbsent(op2BufSync_[dstOp].pipeAfter, BufSyncType::RLS_BUF,
                    dp.dstPipe, bestLogicId, dstIdx, srcIdx);
  }

  if (debugEnabled_) {
    printOp2BufSync(llvm::outs(), op2BufSync_, func_);
  }
}

DenseMap<int, DenseSet<int>> BufidSyncAnalysis::computeLogicIdToPipeInts() const {
  DenseMap<int, DenseSet<int>> logicIdToPipeInts;
  for (auto &[op, build] : op2BufSync_) {
    DenseSet<std::pair<int, int>> seen;
    for (auto &s : build.pipeBefore) {
      auto key = std::make_pair(static_cast<int>(s.pipe), s.logicId);
      if (seen.insert(key).second) {
        logicIdToPipeInts[s.logicId].insert(static_cast<int>(s.pipe));
      }
    }
    for (auto &s : build.pipeAfter) {
      auto key = std::make_pair(static_cast<int>(s.pipe), s.logicId);
      if (seen.insert(key).second) {
        logicIdToPipeInts[s.logicId].insert(static_cast<int>(s.pipe));
      }
    }
  }
  return logicIdToPipeInts;
}

DenseMap<int, SmallVector<int>>
BufidSyncAnalysis::computeCrossPipeSignatureGroups(
    int pipeInt, const SmallVector<int> &logicIds,
    const DenseMap<int, DenseSet<int>> &logicIdToPipeInts) const {
  DenseMap<int, SmallVector<int>> sigPipeToIds;
  for (int lid : logicIds) {
    auto it = logicIdToPipeInts.find(lid);
    if (it == logicIdToPipeInts.end()) {
      continue;
    }
    SmallVector<int> otherPipes;
    for (int p : it->second) {
      if (p != pipeInt) {
        otherPipes.push_back(p);
      }
    }
    if (otherPipes.size() == 1) {
      sigPipeToIds[otherPipes[0]].push_back(lid);
    }
  }
  return sigPipeToIds;
}

void BufidSyncAnalysis::mergeSignatureGroups(
    const DenseMap<int, SmallVector<int>> &sigGroups,
    DenseMap<int, int> &mergeMap) const {
  for (auto &[sigPipe, ids] : sigGroups) {
    if (ids.size() <= 1) {
      continue;
    }
    int survivor = *std::min_element(ids.begin(), ids.end());
    for (int id : ids) {
      if (id != survivor && !mergeMap.contains(id)) {
        mergeMap[id] = survivor;
      }
    }
  }
}

void BufidSyncAnalysis::accumulateMergesForOp(
    const DenseMap<int, SmallVector<int>> &pipeIntToLogicIds,
    const DenseMap<int, DenseSet<int>> &logicIdToPipeInts,
    DenseMap<int, int> &mergeMap) const {
  for (auto &[pipeInt, logicIds] : pipeIntToLogicIds) {
    if (logicIds.size() <= 1) {
      continue;
    }
    DenseMap<int, SmallVector<int>> sigGroups =
        computeCrossPipeSignatureGroups(pipeInt, logicIds, logicIdToPipeInts);
    mergeSignatureGroups(sigGroups, mergeMap);
  }
}

DenseMap<int, int> BufidSyncAnalysis::computeSamePipeMergeMap(
    const DenseMap<int, DenseSet<int>> &logicIdToPipeInts) const {
  DenseMap<int, int> mergeMap;
  for (auto &[op, build] : op2BufSync_) {
    DenseMap<int, SmallVector<int>> pipeIntToLogicIds;
    DenseSet<std::pair<int, int>> seen;
    for (auto &s : build.pipeBefore) {
      auto key = std::make_pair(static_cast<int>(s.pipe), s.logicId);
      if (seen.insert(key).second) {
        pipeIntToLogicIds[static_cast<int>(s.pipe)].push_back(s.logicId);
      }
    }
    for (auto &s : build.pipeAfter) {
      auto key = std::make_pair(static_cast<int>(s.pipe), s.logicId);
      if (seen.insert(key).second) {
        pipeIntToLogicIds[static_cast<int>(s.pipe)].push_back(s.logicId);
      }
    }
    accumulateMergesForOp(pipeIntToLogicIds, logicIdToPipeInts, mergeMap);
  }
  return mergeMap;
}

void BufidSyncAnalysis::rebuildOp2BufSyncWithMerge(
    const DenseMap<int, int> &mergeMap) {
  DenseMap<Operation *, BufSyncPipeBuild> newOp2BufSync;
  for (auto &[op, build] : op2BufSync_) {
    BufSyncPipeBuild newBuild;
    remapSyncList(build.pipeBefore, mergeMap, newBuild.pipeBefore);
    remapSyncList(build.pipeAfter, mergeMap, newBuild.pipeAfter);
    newOp2BufSync[op] = std::move(newBuild);
  }
  op2BufSync_ = std::move(newOp2BufSync);
}

void BufidSyncAnalysis::optimizeSamePipeMerge() {
  DenseMap<int, DenseSet<int>> logicIdToPipeInts = computeLogicIdToPipeInts();
  DenseMap<int, int> mergeMap = computeSamePipeMergeMap(logicIdToPipeInts);

  if (debugEnabled_) {
    llvm::outs() << "[bufid_sync] optimizeSamePipeMerge: mergeMap size=" << mergeMap.size() << "\n";
    for (auto &[id, target] : mergeMap) {
      llvm::outs() << "[bufid_sync]   merge logicId=" << id << " -> " << target << "\n";
    }
  }

  for (auto &[id, target] : mergeMap) {
    while (mergeMap.contains(target)) {
      target = mergeMap[target];
    }
  }

  rebuildOp2BufSyncWithMerge(mergeMap);

  virtualBufIds_.erase(
      std::remove_if(virtualBufIds_.begin(), virtualBufIds_.end(),
                     [&mergeMap](const VirtualBufId &vbid) {
                       return mergeMap.count(vbid.logicId);
                     }),
      virtualBufIds_.end());

  if (debugEnabled_) {
    printOp2BufSync(llvm::outs(), op2BufSync_, func_,
                    "After optimizeSamePipeMerge");
  }
}

void BufidSyncAnalysis::mergeGetRls() {
  unsigned cancelCount = 0;

  RlsMap rootMap;
  for (auto &block : func_.getBody().getBlocks()) {
    cancelGetRlsInBlock(&block, rootMap, op2BufSync_, debugEnabled_,
                        cancelCount);
  }

  SmallVector<Operation *> emptyOps;
  for (auto &[op, build] : op2BufSync_) {
    if (build.pipeBefore.empty() && build.pipeAfter.empty()) {
      emptyOps.push_back(op);
    }
  }
  for (auto *op : emptyOps) {
    op2BufSync_.erase(op);
  }

  if (debugEnabled_) {
    printOp2BufSync(llvm::outs(), op2BufSync_, func_,
                    "After mergeGetRls");
  }
}
