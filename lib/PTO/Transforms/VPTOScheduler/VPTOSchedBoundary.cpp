// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedBoundary.cpp - VPTO scheduling boundary ------------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedBoundary.h"
#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedResourceTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

struct mlir::pto::VPTOPressureEvaluationCache {
  SmallVector<VPTORegPressureEvaluation> evaluations;
  SmallVector<uint8_t> valid;
  DenseMap<Value, SmallVector<unsigned, 2>> usersByRepresentative;
};

namespace {

static uint64_t getTreeOperationWork(size_t bucketCount) {
  static_assert(sizeof(size_t) <= sizeof(uint64_t));
  if (bucketCount <= 1) {
    return 1;
  }
  uint64_t depth = llvm::Log2_64_Ceil(static_cast<uint64_t>(bucketCount));
  return 2 * depth + 1;
}

struct DependencyUpdate {
  VPTOSUnit *unit = nullptr;
  unsigned count = 0;
  unsigned readyCycle = 0;
};

static LogicalResult
collectDependencyUpdates(VPTOSUnit &unit, VPTOSchedDirection direction,
                         unsigned issueCycle, VPTOSchedulingBudget &budget,
                         SmallVectorImpl<DependencyUpdate> &updates,
                         std::string &detail) {
  ArrayRef<VPTOSchedEdge *> edges = direction == VPTOSchedDirection::Top
                                        ? unit.getSuccessors()
                                        : unit.getPredecessors();
  llvm::SmallDenseMap<VPTOSUnit *, size_t, 8> updateByUnit;
  for (VPTOSchedEdge *edge : edges) {
    if (!budget.consume()) {
      detail = "work budget exhausted while inspecting dependencies";
      return mlir::failure();
    }
    if (!edge->isMust()) {
      continue;
    }
    VPTOSUnit *neighbor = direction == VPTOSchedDirection::Top
                              ? edge->getSuccessor()
                              : edge->getPredecessor();
    unsigned latency = edge->getLatency();
    unsigned maxIssueCycle = std::numeric_limits<unsigned>::max() - latency;
    if (issueCycle > maxIssueCycle) {
      detail = "dependency-ready cycle overflow";
      return mlir::failure();
    }
    unsigned edgeReadyCycle = issueCycle + latency;
    auto [found, inserted] = updateByUnit.try_emplace(neighbor, updates.size());
    if (inserted) {
      updates.push_back({neighbor, 1, edgeReadyCycle});
      continue;
    }
    DependencyUpdate &update = updates[found->second];
    if (update.count == std::numeric_limits<unsigned>::max()) {
      detail = "dependency count overflow";
      return mlir::failure();
    }
    ++update.count;
    update.readyCycle = std::max(update.readyCycle, edgeReadyCycle);
  }
  return success();
}

static LogicalResult
validateDependencyUpdates(ArrayRef<DependencyUpdate> updates,
                          ArrayRef<unsigned> remainingDependencies,
                          VPTOSchedulingBudget &budget, std::string &detail) {
  if (!budget.consume(updates.size())) {
    detail = "work budget exhausted while validating dependencies";
    return mlir::failure();
  }
  for (const DependencyUpdate &update : updates) {
    unsigned id = update.unit->getId();
    if (id >= remainingDependencies.size()) {
      detail = "invalid remaining dependency count";
      return mlir::failure();
    }
    if (remainingDependencies[id] < update.count) {
      detail = "invalid remaining dependency count";
      return mlir::failure();
    }
  }
  return success();
}

static LogicalResult prepayCommitWork(ArrayRef<DependencyUpdate> updates,
                                      ArrayRef<unsigned> remainingDependencies,
                                      ArrayRef<unsigned> readyCycles,
                                      unsigned currentCycle,
                                      size_t pendingBucketCount,
                                      VPTOSchedulingBudget &budget,
                                      std::string &detail) {
  if (!budget.consume()) {
    detail = "work budget exhausted before removing the available node";
    return mlir::failure();
  }
  if (!budget.consume(updates.size())) {
    detail = "work budget exhausted before applying dependency updates";
    return mlir::failure();
  }

  bool bucketCountWouldOverflow =
      updates.size() >
      std::numeric_limits<size_t>::max() - pendingBucketCount;
  if (bucketCountWouldOverflow) {
    detail = "pending cycle bucket count overflow";
    return mlir::failure();
  }
  size_t maximumBucketCount = pendingBucketCount + updates.size();
  for (const DependencyUpdate &update : updates) {
    unsigned id = update.unit->getId();
    if (remainingDependencies[id] != update.count) {
      continue;
    }
    unsigned readyCycle = std::max(readyCycles[id], update.readyCycle);
    if (readyCycle <= currentCycle) {
      if (!budget.consume()) {
        detail = "work budget exhausted before adding an available node";
        return mlir::failure();
      }
      continue;
    }
    if (!budget.consume(getTreeOperationWork(maximumBucketCount))) {
      detail = "work budget exhausted before adding a pending node";
      return mlir::failure();
    }
  }
  return success();
}

static void
invalidateOperandPressureEvaluations(Operation *operation,
                                     const VPTORegPressureTracker &tracker,
                                     VPTOPressureEvaluationCache &cache) {
  DenseSet<Value> consumedRepresentatives;
  for (Value operand : operation->getOperands()) {
    Value representative = tracker.getPressureRepresentative(operand);
    if (!consumedRepresentatives.insert(representative).second) {
      continue;
    }
    auto found = cache.usersByRepresentative.find(representative);
    if (found == cache.usersByRepresentative.end()) {
      continue;
    }
    for (unsigned affected : found->second) {
      cache.valid[affected] = 0;
    }
  }
}
} // namespace

VPTOSchedBoundary::VPTOSchedBoundary(const VPTOSchedDAG &dag,
                                     const VPTOSchedModel &model,
                                     VPTOSchedDirection direction)
    : VPTOSchedBoundary(dag, model, direction,
                        std::make_unique<VPTONullHazardRecognizer>()) {}

VPTOSchedBoundary::VPTOSchedBoundary(
    const VPTOSchedDAG &dag, const VPTOSchedModel &model,
    VPTOSchedDirection direction,
    std::unique_ptr<VPTOHazardRecognizer> hazardRecognizer)
    : direction(direction),
      resourceTracker(std::make_unique<VPTOResourceTracker>(model)),
      pressureTracker(
          std::make_unique<VPTORegPressureTracker>(model, dag, direction)),
      pressureEvaluationCache(
          std::make_unique<VPTOPressureEvaluationCache>()),
      hazardRecognizer(std::move(hazardRecognizer)) {
  if (!this->hazardRecognizer) {
    this->hazardRecognizer = std::make_unique<VPTONullHazardRecognizer>();
  }
  size_t nodeCount = dag.getUnits().size();
  available.reserve(nodeCount);
  availablePositions.assign(nodeCount, 0);
  remainingDependencies.assign(nodeCount, 0);
  readyCycles.assign(nodeCount, 0);
  states.assign(nodeCount, UnitState::Unavailable);
  pressureEvaluationCache->evaluations.resize(nodeCount);
  pressureEvaluationCache->valid.assign(nodeCount, 0);
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    DenseSet<Value> seenRepresentatives;
    for (Value operand : unit->getOperation()->getOperands()) {
      Value representative =
          pressureTracker->getPressureRepresentative(operand);
      if (seenRepresentatives.insert(representative).second) {
        pressureEvaluationCache->usersByRepresentative[representative]
            .push_back(unit->getId());
      }
    }
    unsigned dependencies = direction == VPTOSchedDirection::Top
                                ? unit->getRemainingPredecessors()
                                : unit->getRemainingSuccessors();
    remainingDependencies[unit->getId()] = dependencies;
    if (dependencies == 0) {
      insertAvailable(unit.get());
    }
  }
}

VPTOSchedBoundary::~VPTOSchedBoundary() = default;

const VPTOPendingUnit *VPTOSchedBoundary::getNextPending() const {
  if (pendingByCycle.empty()) {
    return nullptr;
  }
  const auto &[readyCycle, units] = *pendingByCycle.begin();
  nextPending = {units.front(), readyCycle};
  return &nextPending;
}

VPTOResourceTracker &VPTOSchedBoundary::getResourceTracker() {
  return *resourceTracker;
}

const VPTOResourceTracker &VPTOSchedBoundary::getResourceTracker() const {
  return *resourceTracker;
}

VPTORegPressureTracker &VPTOSchedBoundary::getPressureTracker() {
  return *pressureTracker;
}

const VPTORegPressureTracker &VPTOSchedBoundary::getPressureTracker() const {
  return *pressureTracker;
}

VPTORegPressureEvaluation
VPTOSchedBoundary::evaluatePressure(const VPTOSUnit &unit) const {
  unsigned id = unit.getId();
  if (direction != VPTOSchedDirection::Top ||
      id >= pressureEvaluationCache->valid.size())
    return pressureTracker->evaluate(unit);
  if (!pressureEvaluationCache->valid[id]) {
    pressureEvaluationCache->evaluations[id] =
        pressureTracker->evaluate(unit);
    pressureEvaluationCache->valid[id] = 1;
  }
  VPTORegPressureEvaluation evaluation =
      pressureEvaluationCache->evaluations[id];
  pressureTracker->refreshSummary(evaluation);
  return evaluation;
}

VPTOHazardRecognizer &VPTOSchedBoundary::getHazardRecognizer() {
  return *hazardRecognizer;
}

const VPTOHazardRecognizer &VPTOSchedBoundary::getHazardRecognizer() const {
  return *hazardRecognizer;
}

bool VPTOSchedBoundary::isScheduled(const VPTOSUnit *unit) const {
  return unit && unit->getId() < states.size() &&
         states[unit->getId()] == UnitState::Scheduled;
}

bool VPTOSchedBoundary::isAvailable(const VPTOSUnit *unit) const {
  bool invalidUnit = !unit || unit->getId() >= states.size();
  if (invalidUnit) {
    return false;
  }
  unsigned id = unit->getId();
  size_t position = availablePositions[id];
  return states[id] == UnitState::Available && position < available.size() &&
         available[position] == unit;
}

unsigned VPTOSchedBoundary::getRemainingDependencyCount(
    const VPTOSUnit &unit) const {
  unsigned id = unit.getId();
  return id < remainingDependencies.size() ? remainingDependencies[id] : 0;
}

void VPTOSchedBoundary::insertAvailable(VPTOSUnit *unit) {
  unsigned id = unit->getId();
  if (id >= states.size()) {
    return;
  }
  bool cannotInsert =
      states[id] == UnitState::Scheduled || states[id] == UnitState::Available;
  if (cannotInsert) {
    return;
  }
  availablePositions[id] = available.size();
  available.push_back(unit);
  states[id] = UnitState::Available;
}

void VPTOSchedBoundary::eraseAvailable(VPTOSUnit *unit) {
  unsigned id = unit->getId();
  size_t position = availablePositions[id];
  VPTOSUnit *last = available.back();
  available[position] = last;
  availablePositions[last->getId()] = position;
  available.pop_back();
}

void VPTOSchedBoundary::insertPending(VPTOSUnit *unit, unsigned readyCycle) {
  unsigned id = unit->getId();
  states[id] = UnitState::Pending;
  readyCycles[id] = readyCycle;
  SmallVector<VPTOSUnit *> &bucket = pendingByCycle[readyCycle];
  auto position = llvm::lower_bound(
      bucket, unit, [](VPTOSUnit *lhs, VPTOSUnit *rhs) {
        return lhs->getOriginalIndex() < rhs->getOriginalIndex();
      });
  bucket.insert(position, unit);
  ++pendingCount;
}

LogicalResult VPTOSchedBoundary::defer(VPTOSUnit &unit, unsigned readyCycle,
                                       VPTOSchedulingBudget &budget) {
  unsigned id = unit.getId();
  if (id >= states.size()) {
    return failure();
  }
  bool cannotDefer = !isAvailable(&unit) || readyCycle <= currentCycle;
  if (cannotDefer) {
    return failure();
  }
  bool pendingSizeOverflow =
      pendingCount == std::numeric_limits<size_t>::max();
  if (pendingSizeOverflow) {
    return failure();
  }
  uint64_t work = getTreeOperationWork(pendingByCycle.size() + 1);
  if (!budget.consume()) {
    return failure();
  }
  if (!budget.consume(work)) {
    return failure();
  }
  eraseAvailable(&unit);
  insertPending(&unit, readyCycle);
  return success();
}

void VPTOSchedBoundary::releasePending() {
  bool hasReadyBucket = !pendingByCycle.empty() &&
                        pendingByCycle.begin()->first <= currentCycle;
  while (hasReadyBucket) {
    auto bucket = pendingByCycle.extract(pendingByCycle.begin());
    for (VPTOSUnit *unit : bucket.mapped()) {
      insertAvailable(unit);
      --pendingCount;
    }
    hasReadyBucket = !pendingByCycle.empty() &&
                     pendingByCycle.begin()->first <= currentCycle;
  }
}

FailureOr<bool>
VPTOSchedBoundary::advanceToNextPendingCycle(VPTOSchedulingBudget &budget) {
  if (pendingByCycle.empty()) {
    return false;
  }
  unsigned nextCycle = std::max(currentCycle, pendingByCycle.begin()->first);
  size_t releaseCount = pendingByCycle.begin()->second.size();
  if (!budget.consume(1 + releaseCount)) {
    return mlir::failure();
  }
  currentCycle = nextCycle;
  releasePending();
  return true;
}

LogicalResult VPTOSchedBoundary::commit(VPTOSUnit &unit, unsigned issueCycle,
                                        VPTOSchedulingBudget &budget,
                                        std::string &detail) {
  unsigned id = unit.getId();
  bool selectedAtCurrentCycle =
      isAvailable(&unit) && issueCycle == currentCycle;
  if (!selectedAtCurrentCycle) {
    detail = "selected node is not available at the current cycle";
    return mlir::failure();
  }

  SmallVector<DependencyUpdate, 8> updates;
  if (failed(collectDependencyUpdates(unit, direction, issueCycle, budget,
                                      updates, detail))) {
    return mlir::failure();
  }
  if (failed(validateDependencyUpdates(updates, remainingDependencies, budget,
                                       detail))) {
    return mlir::failure();
  }
  if (failed(prepayCommitWork(updates, remainingDependencies, readyCycles,
                              currentCycle, pendingByCycle.size(), budget,
                              detail))) {
    return mlir::failure();
  }
  if (failed(pressureTracker->commit(unit))) {
    detail = "register-pressure tracker rejected selected node";
    return mlir::failure();
  }
  if (direction == VPTOSchedDirection::Top) {
    invalidateOperandPressureEvaluations(unit.getOperation(), *pressureTracker,
                                         *pressureEvaluationCache);
  }

  eraseAvailable(&unit);
  states[id] = UnitState::Scheduled;
  ++scheduledCount;
  for (const DependencyUpdate &update : updates) {
    unsigned neighborId = update.unit->getId();
    readyCycles[neighborId] =
        std::max(readyCycles[neighborId], update.readyCycle);
    remainingDependencies[neighborId] -= update.count;
    if (remainingDependencies[neighborId] != 0) {
      continue;
    }
    if (readyCycles[neighborId] <= currentCycle) {
      insertAvailable(update.unit);
    } else {
      insertPending(update.unit, readyCycles[neighborId]);
    }
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
