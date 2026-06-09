// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- GraphSolver.h ---- Graph Sync Solver -------------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H

#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include <set>

namespace mlir::pto::syncsolver {

class GraphSolver {
public:
  struct Edge {
    ConflictPair *const conflictPair;
    const CorePipeInfo corePipeSrc;
    const CorePipeInfo corePipeDst;
    const int startIndex;
    const int endIndex;
    const bool isUnitFlag;
    Edge() = delete;
    Edge(ConflictPair *conflictPair, CorePipeInfo corePipeSrc,
         CorePipeInfo corePipeDst, int startIndex, int endIndex,
         bool isUnitFlag)
        : conflictPair(conflictPair), corePipeSrc(corePipeSrc),
          corePipeDst(corePipeDst), startIndex(startIndex), endIndex(endIndex),
          isUnitFlag(isUnitFlag) {}
    Edge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, int startIndex,
         int endIndex)
        : Edge(nullptr, corePipeSrc, corePipeDst, startIndex, endIndex, false) {
    }
    bool operator<(const Edge &other) const;
  };

  // Configuration options.
  const SyncSolverOptions options;

  // adjacencyList[pipeSrc][pipeDst] stores a set of Edge objects representing
  // directed transitions from pipeSrc to pipeDst that are valid for a given
  // (startIndex,endIndex) lifetime. Used by runDijkstra to compute minimum
  // distance paths between two pipe ids taking ordering constraints into
  // account.
  CorePipeDenseMap<CorePipeDenseMap<std::set<Edge>>> adjacencyList;

  GraphSolver(const SyncSolverOptions &options) : options(options) {}

  // Add a pipe-pair edge annotated with its active index interval.
  void addPair(ConflictPair *conflictPair, CorePipeInfo corePipeSrc,
               CorePipeInfo corePipeDst, int startIndex, int endIndex,
               bool isUnitFlag = false);

  // Build adjacency list from a ConflictPair by decomposing it into edges.
  void addConflictPair(syncsolver::ConflictPair *conflictPair);

  // Compact or merge overlapping edges to speed up Dijkstra queries.
  void optimizeAdjacencyList();

  // Run shortest-path search (Dijkstra-like) with ordering constraints to find
  // the minimal reachable index for a path from startPipe to endPipe.
  std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                 CorePipeInfo corePipeDst, int startIndex,
                                 int endIndex);

  std::optional<int> runDijkstraUnitFlagEnabled(const Occurrence *occ1,
                                                const Occurrence *occ2,
                                                CorePipeInfo corePipeSrc,
                                                CorePipeInfo corePipeDst,
                                                int startIndex, int endIndex);
};
} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
