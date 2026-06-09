// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===----------- GraphSolver.cpp ---- Graph Sync Solver -------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/GraphSolver.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/Support/Debug.h"
#include <cstddef>
#include <optional>
#include <queue>
#include <utility>

#define DEBUG_TYPE "PTO-gss-graph-solver"

using namespace mlir;
using namespace pto::syncsolver;

using UnitDistKey = std::tuple<int, int, CorePipeInfo>;
constexpr size_t kUnitDistSourceIndex = 0;
constexpr size_t kUnitDistTargetIndex = 1;
constexpr size_t kUnitDistPipeIndex = 2;

struct UnitDistKeyInfo {
  static inline UnitDistKey getEmptyKey() {
    return {llvm::DenseMapInfo<int>::getEmptyKey(),
            llvm::DenseMapInfo<int>::getEmptyKey(),
            CorePipeInfoKeyInfo::getEmptyKey()};
  }

  static inline UnitDistKey getTombstoneKey() {
    return {llvm::DenseMapInfo<int>::getTombstoneKey(),
            llvm::DenseMapInfo<int>::getTombstoneKey(),
            CorePipeInfoKeyInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const UnitDistKey &val) {
    unsigned hash = llvm::DenseMapInfo<int>::getHashValue(
        std::get<kUnitDistSourceIndex>(val));
    hash = combineDenseHash(
        hash, llvm::DenseMapInfo<int>::getHashValue(
                  std::get<kUnitDistTargetIndex>(val)));
    return combineDenseHash(
        hash,
        CorePipeInfoKeyInfo::getHashValue(std::get<kUnitDistPipeIndex>(val)));
  }

  static bool isEqual(const UnitDistKey &lhs, const UnitDistKey &rhs) {
    return llvm::DenseMapInfo<int>::isEqual(
               std::get<kUnitDistSourceIndex>(lhs),
               std::get<kUnitDistSourceIndex>(rhs)) &&
           llvm::DenseMapInfo<int>::isEqual(
               std::get<kUnitDistTargetIndex>(lhs),
               std::get<kUnitDistTargetIndex>(rhs)) &&
           CorePipeInfoKeyInfo::isEqual(std::get<kUnitDistPipeIndex>(lhs),
                                        std::get<kUnitDistPipeIndex>(rhs));
  }
};

using UnitDistMap = llvm::DenseMap<UnitDistKey, int, UnitDistKeyInfo>;
using UnitQueueItem = std::pair<int, UnitDistKey>;
using UnitPriorityQueue =
    std::priority_queue<UnitQueueItem, std::vector<UnitQueueItem>,
                        std::greater<UnitQueueItem>>;

static bool hasBetterDistance(const UnitDistMap &distance,
                              const UnitDistKey &distKey, int curIndex) {
  return (distance.count(distKey) != 0) &&
         distance.lookup(distKey) < curIndex;
}

static bool exceededSearchBound(const UnitDistMap &distance,
                                const UnitDistKey &distKey, int endIndex) {
  return (distance.count(distKey) != 0) &&
         distance.lookup(distKey) > endIndex;
}

static bool reachedUnitFlagDestination(CorePipeInfo curCorePipe,
                                       pto::TCoreType coreDst, int startIndex,
                                       int curIndex) {
  auto [curCore, curPipe] = curCorePipe;
  return curCore == coreDst &&
         ((curIndex != startIndex && curPipe == pto::PIPE::PIPE_S) ||
          curPipe == pto::PIPE::PIPE_ALL);
}

static void relaxUnitFlagEdge(const GraphSolver::Edge &edge,
                              const Occurrence *occ2, UnitDistMap &distance,
                              UnitPriorityQueue &que) {
  ASSERT(edge.conflictPair != nullptr);
  UnitDistKey nxtKey(edge.isUnitFlag, edge.conflictPair->waitOcc == occ2,
                     edge.corePipeDst);
  auto it = distance.find(nxtKey);
  if (it == distance.end() || it->second > edge.endIndex) {
    distance[nxtKey] = edge.endIndex;
    que.emplace(edge.endIndex, nxtKey);
  }
}

static std::optional<int> collectUnitFlagResult(const UnitDistMap &distance,
                                                CorePipeInfo corePipeDst) {
  std::optional<int> retDist;
  for (const UnitDistKey &key :
       {UnitDistKey(false, false, corePipeDst), UnitDistKey(false, true, corePipeDst),
        UnitDistKey(true, true, corePipeDst)}) {
    auto it = distance.find(key);
    if (it == distance.end())
      continue;
    retDist = retDist.has_value() ? std::min(retDist.value(), it->second)
                                  : it->second;
  }
  return retDist;
}

// Compare edges (used for ordered sets). Edges must share endpoints when
// compared.
bool GraphSolver::Edge::operator<(const Edge &other) const {
  ASSERT(corePipeSrc == other.corePipeSrc);
  ASSERT(corePipeDst == other.corePipeDst);
  if (startIndex != other.startIndex) {
    return startIndex < other.startIndex;
  }
  return endIndex < other.endIndex;
}

// Add an adjacency edge annotated with an active index interval.
void GraphSolver::addPair(ConflictPair *conflictPair, CorePipeInfo corePipeSrc,
                          CorePipeInfo corePipeDst, int startIndex,
                          int endIndex, bool isUnitFlag) {
  Edge edge(conflictPair, corePipeSrc, corePipeDst, startIndex, endIndex,
            isUnitFlag);
  adjacencyList[edge.corePipeSrc][edge.corePipeDst].insert(edge);
}

// Convert a ConflictPair into adjacency edges (handles PIPE_ALL
// special-casing).
void GraphSolver::addConflictPair(ConflictPair *conflictPair) {
  ASSERT(conflictPair != nullptr);
  DEBUG_WITH_TYPE("gss-graph-solver-add-conflict-pair", {
    llvm::dbgs() << "add-conflict-pair:\n";
    llvm::dbgs() << conflictPair->str() << '\n';
  });
  if (conflictPair->isBarrier() &&
      conflictPair->setCorePipeInfo.pipe == pto::PIPE::PIPE_ALL) {
    llvm::SmallVector<std::pair<pto::TCoreType, pto::TCoreType>> srcDstCores;
    if (options.isCrossCoreMode()) {
      srcDstCores.push_back(
          std::make_pair(pto::TCoreType::CUBE, pto::TCoreType::VECTOR));
      srcDstCores.push_back(
          std::make_pair(pto::TCoreType::VECTOR, pto::TCoreType::CUBE));
    } else {
      srcDstCores.push_back(std::make_pair(pto::TCoreType::CUBE_OR_VECTOR,
                                           pto::TCoreType::CUBE_OR_VECTOR));
    }
    for (auto [srcCore, dstCore] : srcDstCores) {
      for (int i = 0; i < static_cast<int>(pto::PIPE::PIPE_NUM); i++) {
        auto setPipe = static_cast<pto::PIPE>(i);
        auto waitPipe = pto::PIPE::PIPE_ALL;
        int startIndex = conflictPair->startIndex;
        int endIndex = conflictPair->endIndex;
        ASSERT(startIndex == endIndex);
        addPair(conflictPair, CorePipeInfo(srcCore, setPipe),
                CorePipeInfo(dstCore, waitPipe), startIndex, endIndex);
      }
    }
  } else if (!conflictPair->isBarrier() &&
             conflictPair->waitCorePipeInfo.pipe == pto::PIPE::PIPE_S) {
    for (int i = 0; i < static_cast<int>(pto::PIPE::PIPE_NUM); i++) {
      auto coreDst = conflictPair->waitCorePipeInfo.coreType;
      auto waitPipe = static_cast<pto::PIPE>(i);
      addPair(conflictPair, conflictPair->setCorePipeInfo,
              CorePipeInfo(coreDst, waitPipe), conflictPair->startIndex,
              conflictPair->endIndex);
    }
  } else {
    auto corePipeSrc = conflictPair->setCorePipeInfo;
    auto corePipeDst = conflictPair->waitCorePipeInfo;
    int startIndex = conflictPair->startIndex;
    int endIndex = conflictPair->endIndex;
    addPair(conflictPair, corePipeSrc, corePipeDst, startIndex, endIndex,
            conflictPair->replacedWithUnitFlag);
  }
}

// Compact adjacency lists by removing dominated edges to accelerate queries.
void GraphSolver::optimizeAdjacencyList() {
  for (auto &[corePipeSrc, dstMap] : adjacencyList) {
    for (auto &[corePipeDst, edges] : dstMap) {
      std::set<Edge> optimizedEdges;
      for (auto &edge : edges) {
        while (!optimizedEdges.empty() &&
               optimizedEdges.rbegin()->endIndex >= edge.endIndex) {
          optimizedEdges.erase(*optimizedEdges.rbegin());
        }
        optimizedEdges.insert(edge);
      }
      edges = std::move(optimizedEdges);
    }
  }
}

// Run a Dijkstra-like search over pipes using index intervals as
// weights/constraints. Returns minimal reachable index for endPipe or empty
// optional if unreachable.
std::optional<int> GraphSolver::runDijkstra(CorePipeInfo corePipeSrc,
                                            CorePipeInfo corePipeDst,
                                            int startIndex, int endIndex) {
  CorePipeDenseMap<int> distance;
  std::priority_queue<std::pair<int, CorePipeInfo>,
                      std::vector<std::pair<int, CorePipeInfo>>,
                      std::greater<std::pair<int, CorePipeInfo>>>
      que;
  que.emplace(startIndex, corePipeSrc);
  auto [coreDst, pipeDst] = corePipeDst;

  LLVM_DEBUG(llvm::dbgs() << "dij-start-end-indices: " << startIndex << ' '
                          << endIndex << '\n');

  while (!que.empty()) {
    auto [curIndex, curCorePipe] = que.top();
    auto [curCore, curPipe] = curCorePipe;
    que.pop();

    LLVM_DEBUG(llvm::dbgs() << "dij-step: " << curCore << ' ' << curPipe << ' '
                            << curIndex << '\n');

    if (curCorePipe == corePipeDst &&
        (distance.count(corePipeDst) != 0)) {
      break;
    }

    if ((distance.count(curCorePipe) != 0) &&
        distance[curCorePipe] < curIndex) {
      continue;
    }

    if ((distance.count(curCorePipe) != 0) &&
        distance[curCorePipe] > endIndex) {
      break;
    }

    if (curCore == coreDst &&
        ((curIndex != startIndex && curPipe == pto::PIPE::PIPE_S) ||
         curPipe == pto::PIPE::PIPE_ALL)) {
      distance[corePipeDst] = curIndex;
      break;
    }

    for (auto &[endCorePipe, edges] : adjacencyList[curCorePipe]) {
      auto it = edges.lower_bound(Edge(curCorePipe, endCorePipe, curIndex, -1));
      for (; it != edges.end(); it++) {
        if (distance.count(endCorePipe) == 0 ||
            (distance[endCorePipe] > (it->endIndex))) {
          distance[endCorePipe] = it->endIndex;
          que.emplace(it->endIndex, endCorePipe);
        }
      }
    }
  }

  return distance.count(corePipeDst) != 0 ? distance[corePipeDst]
                                          : std::optional<int>();
}

std::optional<int> GraphSolver::runDijkstraUnitFlagEnabled(
    const Occurrence *occ1, const Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, int startIndex, int endIndex) {
  UnitDistMap distance;
  UnitPriorityQueue que;
  que.emplace(startIndex, UnitDistKey(false, false, corePipeSrc));
  auto [coreDst, pipeDst] = corePipeDst;
  LLVM_DEBUG(llvm::dbgs() << "dij-start-end-indices: " << startIndex << ' '
                          << endIndex << '\n');

  while (!que.empty()) {
    auto [curIndex, curDistKey] = que.top();
    auto [curIsUnitFlag, curIsOccDst, curCorePipe] = curDistKey;
    que.pop();

    LLVM_DEBUG(llvm::dbgs()
               << "dij-step: " << static_cast<int>(curCorePipe.coreType) << ' '
               << static_cast<int>(curCorePipe.pipe) << ' ' << curIsUnitFlag
               << ' ' << curIsOccDst << ' ' << curIndex << '\n');
    if (hasBetterDistance(distance, curDistKey, curIndex))
      continue;
    if (exceededSearchBound(distance, curDistKey, endIndex))
      break;
    if (reachedUnitFlagDestination(curCorePipe, coreDst, startIndex,
                                   curIndex)) {
      distance[UnitDistKey(false, false, corePipeDst)] = curIndex;
      break;
    }
    for (auto &[endCorePipe, edges] : adjacencyList[curCorePipe]) {
      auto it = edges.lower_bound(Edge(curCorePipe, endCorePipe, curIndex, -1));
      for (; it != edges.end(); ++it) {
        const auto &edge = *it;
        if (edge.isUnitFlag &&
            curIndex == startIndex && edge.startIndex != startIndex)
          continue;
        relaxUnitFlagEdge(edge, occ2, distance, que);
      }
    }
  }
  (void)occ1;
  (void)pipeDst;
  return collectUnitFlagResult(distance, corePipeDst);
}
