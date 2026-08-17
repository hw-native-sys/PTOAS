// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAG.cpp - VPTO scheduling DAG ----------------------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

using namespace mlir;
using namespace mlir::pto;

VPTOSchedDAG::VPTOSchedDAG(const VPTOSchedRegion &region) : region(region) {
  units.reserve(region.operations.size());
  for (auto [index, op] : llvm::enumerate(region.operations)) {
    auto unit = std::make_unique<VPTOSUnit>(index, index, op);
    unitByOperation.try_emplace(op, unit.get());
    units.push_back(std::move(unit));
  }
}

VPTOSUnit *VPTOSchedDAG::lookup(Operation *op) const {
  auto found = unitByOperation.find(op);
  return found == unitByOperation.end() ? nullptr : found->second;
}

VPTOSchedEdge &VPTOSchedDAG::addEdge(
    VPTOSUnit &predecessor, VPTOSUnit &successor, VPTOSchedEdgeKind kind,
    VPTOSchedEdgeStrength strength, unsigned latency, Twine reason) {
  assert(&predecessor != &successor && "self dependencies are invalid");
  auto edge = std::make_unique<VPTOSchedEdge>(
      &predecessor, &successor, kind, strength, latency, reason.str());
  VPTOSchedEdge *edgePtr = edge.get();
  predecessor.successors.push_back(edgePtr);
  successor.predecessors.push_back(edgePtr);
  edges.push_back(std::move(edge));
  return *edgePtr;
}

void VPTOSchedDAG::resetDependencyCounts() {
  for (const std::unique_ptr<VPTOSUnit> &unit : units) {
    unit->remainingPredecessors = llvm::count_if(
        unit->predecessors, [](VPTOSchedEdge *edge) { return edge->isMust(); });
    unit->remainingSuccessors = llvm::count_if(
        unit->successors, [](VPTOSchedEdge *edge) { return edge->isMust(); });
  }
}

void VPTOSchedDAG::addLiveIn(Value value) {
  if (value && !llvm::is_contained(liveIns, value))
    liveIns.push_back(value);
}

void VPTOSchedDAG::addLiveOut(Value value) {
  if (value && !llvm::is_contained(liveOuts, value))
    liveOuts.push_back(value);
}

LogicalResult VPTOSchedDAG::computeCriticalPaths() {
  SmallVector<unsigned> indegree(units.size(), 0);
  SmallVector<VPTOSUnit *> ready;
  SmallVector<VPTOSUnit *> topologicalOrder;

  for (const std::unique_ptr<VPTOSUnit> &unit : units) {
    unit->setDepth(0);
    unit->setHeight(0);
    indegree[unit->getId()] = llvm::count_if(
        unit->getPredecessors(),
        [](VPTOSchedEdge *edge) { return edge->isMust(); });
    if (indegree[unit->getId()] == 0)
      ready.push_back(unit.get());
  }

  for (size_t cursor = 0; cursor < ready.size(); ++cursor) {
    VPTOSUnit *unit = ready[cursor];
    topologicalOrder.push_back(unit);
    for (VPTOSchedEdge *edge : unit->getSuccessors()) {
      if (!edge->isMust())
        continue;
      VPTOSUnit *successor = edge->getSuccessor();
      successor->setDepth(
          std::max(successor->getDepth(),
                   unit->getDepth() + edge->getLatency()));
      unsigned &remaining = indegree[successor->getId()];
      assert(remaining != 0 && "invalid dependency count");
      if (--remaining == 0)
        ready.push_back(successor);
    }
  }

  if (topologicalOrder.size() != units.size())
    return failure();

  for (VPTOSUnit *unit : llvm::reverse(topologicalOrder)) {
    for (VPTOSchedEdge *edge : unit->getSuccessors()) {
      if (!edge->isMust())
        continue;
      unit->setHeight(std::max(
          unit->getHeight(), edge->getSuccessor()->getHeight() +
                                 edge->getLatency()));
    }
  }
  return success();
}

StringRef mlir::pto::stringifyVPTOSchedEdgeKind(VPTOSchedEdgeKind kind) {
  switch (kind) {
  case VPTOSchedEdgeKind::Data:
    return "data";
  case VPTOSchedEdgeKind::Anti:
    return "anti";
  case VPTOSchedEdgeKind::Output:
    return "output";
  case VPTOSchedEdgeKind::Memory:
    return "memory";
  case VPTOSchedEdgeKind::Control:
    return "control";
  case VPTOSchedEdgeKind::Sync:
    return "sync";
  case VPTOSchedEdgeKind::Artificial:
    return "artificial";
  case VPTOSchedEdgeKind::Cluster:
    return "cluster";
  }
  llvm_unreachable("unknown VPTO scheduling edge kind");
}

StringRef mlir::pto::stringifyVPTOSchedEdgeStrength(
    VPTOSchedEdgeStrength strength) {
  switch (strength) {
  case VPTOSchedEdgeStrength::Must:
    return "must";
  case VPTOSchedEdgeStrength::Weak:
    return "weak";
  }
  llvm_unreachable("unknown VPTO scheduling edge strength");
}
