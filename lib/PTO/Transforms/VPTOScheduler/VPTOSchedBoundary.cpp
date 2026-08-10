// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOSchedBoundary.cpp - VPTO scheduling boundary ------------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedBoundary.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

VPTOSchedBoundary::VPTOSchedBoundary(VPTOSchedDAG &dag,
                                     VPTOSchedDirection direction)
    : direction(direction) {
  dag.resetDependencyCounts();
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    unsigned dependencies = direction == VPTOSchedDirection::Top
                                ? unit->getRemainingPredecessors()
                                : unit->getRemainingSuccessors();
    if (dependencies == 0)
      insertAvailable(unit.get());
  }
}

void VPTOSchedBoundary::insertAvailable(VPTOSUnit *unit) {
  if (scheduled.contains(unit) || llvm::is_contained(available, unit))
    return;
  auto position =
      llvm::lower_bound(available, unit, [](VPTOSUnit *lhs, VPTOSUnit *rhs) {
        return lhs->getOriginalIndex() < rhs->getOriginalIndex();
      });
  available.insert(position, unit);
}

LogicalResult VPTOSchedBoundary::defer(VPTOSUnit &unit, unsigned readyCycle) {
  auto found = llvm::find(available, &unit);
  if (found == available.end() || readyCycle <= currentCycle)
    return failure();
  available.erase(found);
  pending.push_back({&unit, readyCycle});
  llvm::sort(
      pending, [](const VPTOPendingUnit &lhs, const VPTOPendingUnit &rhs) {
        if (lhs.readyCycle != rhs.readyCycle)
          return lhs.readyCycle < rhs.readyCycle;
        return lhs.unit->getOriginalIndex() < rhs.unit->getOriginalIndex();
      });
  return success();
}

void VPTOSchedBoundary::releasePending() {
  SmallVector<VPTOPendingUnit> stillPending;
  for (const VPTOPendingUnit &entry : pending) {
    if (entry.readyCycle <= currentCycle)
      insertAvailable(entry.unit);
    else
      stillPending.push_back(entry);
  }
  pending = std::move(stillPending);
}

bool VPTOSchedBoundary::advanceToNextPendingCycle() {
  if (pending.empty())
    return false;
  currentCycle = std::max(currentCycle, pending.front().readyCycle);
  releasePending();
  return true;
}

LogicalResult VPTOSchedBoundary::commit(VPTOSUnit &unit) {
  auto availablePosition = llvm::find(available, &unit);
  if (availablePosition == available.end() || scheduled.contains(&unit))
    return failure();
  available.erase(availablePosition);
  scheduled.insert(&unit);

  ArrayRef<VPTOSchedEdge *> edges = direction == VPTOSchedDirection::Top
                                        ? unit.getSuccessors()
                                        : unit.getPredecessors();
  for (VPTOSchedEdge *edge : edges) {
    if (!edge->isMust())
      continue;
    VPTOSUnit *neighbor = direction == VPTOSchedDirection::Top
                              ? edge->getSuccessor()
                              : edge->getPredecessor();
    unsigned remaining = direction == VPTOSchedDirection::Top
                             ? neighbor->getRemainingPredecessors()
                             : neighbor->getRemainingSuccessors();
    if (remaining == 0)
      return failure();
    --remaining;
    if (direction == VPTOSchedDirection::Top)
      neighbor->setRemainingPredecessors(remaining);
    else
      neighbor->setRemainingSuccessors(remaining);
    if (remaining == 0)
      insertAvailable(neighbor);
  }
  return success();
}

StringRef mlir::pto::stringifyVPTOSchedDirection(VPTOSchedDirection direction) {
  switch (direction) {
  case VPTOSchedDirection::Top:
    return "top";
  case VPTOSchedDirection::Bottom:
    return "bottom";
  }
  llvm_unreachable("unknown VPTO scheduling direction");
}
