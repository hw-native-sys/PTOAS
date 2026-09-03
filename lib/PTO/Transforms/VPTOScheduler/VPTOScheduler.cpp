// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOScheduler.cpp - VPTO list scheduler ---------------------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedBoundary.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kMaxClosureGroupNodes = 96;

struct PressureClosureGroup {
  VPTOSUnit *target = nullptr;
  std::optional<unsigned> pressureSet;
  SmallVector<unsigned, 2> bundleOriginalIndices;
  SmallVector<Value, 2> targetValues;
  DenseSet<VPTOSUnit *> units;
  SmallVector<VPTOSUnit *, 8> witness;
  SmallVector<int64_t, 2> peak;
  SmallVector<int64_t, 2> end;
  unsigned steps = 0;
  int64_t supportPressure = 0;
  int64_t effectiveEnd = 0;
  int64_t netRelief = 0;

  void clear() {
    target = nullptr;
    pressureSet.reset();
    bundleOriginalIndices.clear();
    targetValues.clear();
    units.clear();
    witness.clear();
    peak.clear();
    end.clear();
    steps = 0;
    supportPressure = 0;
    effectiveEnd = 0;
    netRelief = 0;
  }
};

static void saturatingMultiplyAdd(int64_t lhs, int64_t rhs, int64_t &total) {
  int64_t product = 0;
  int64_t updated = 0;
  bool scoreWouldOverflow = llvm::MulOverflow(lhs, rhs, product) ||
                            llvm::AddOverflow(total, product, updated);
  if (scoreWouldOverflow) {
    total = std::numeric_limits<int64_t>::max();
    return;
  }
  total = updated;
}

static bool isCriticalPressure(const VPTOSchedBoundary &boundary,
                               const VPTOSchedModel &model) {
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    if (pressureSet.limit &&
        boundary.getPressureTracker().getCurrent()[index] * 2 >=
            static_cast<int64_t>(*pressureSet.limit)) {
      return true;
    }
  }
  return false;
}

static int64_t getEvaluationExcess(
    const VPTORegPressureEvaluation &evaluation,
    ArrayRef<VPTORegPressureSet> pressureSets) {
  int64_t cost = 0;
  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    saturatingMultiplyAdd(evaluation.projectedExcess[index],
                          pressureSet.spillCost, cost);
  }
  return cost;
}

static int64_t getEvaluationPressure(
    const VPTORegPressureEvaluation &evaluation,
    ArrayRef<VPTORegPressureSet> pressureSets) {
  int64_t cost = 0;
  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    saturatingMultiplyAdd(evaluation.projected[index], pressureSet.weight,
                          cost);
  }
  return cost;
}

static VPTOSchedCandidate
initializeCandidate(VPTOSUnit &unit, const VPTOSchedBoundary &boundary,
                    const PressureClosureGroup *closureGroup) {
  VPTOSchedDirection direction = boundary.getDirection();
  unsigned criticalPath =
      direction == VPTOSchedDirection::Top ? unit.getHeight() : unit.getDepth();
  VPTORegPressureEvaluation pressure = boundary.evaluatePressure(unit);
  VPTOSchedCandidate candidate{&unit,
                               direction,
                               boundary.getCurrentCycle(),
                               criticalPath,
                               unit.getOriginalIndex(),
                               pressure,
                               pressure.projected,
                               pressure.projected,
                               0,
                               false};
  candidate.opensPressureFrontier =
      llvm::any_of(pressure.introduced,
                   [](int64_t value) { return value > 0; }) &&
      llvm::all_of(pressure.released,
                   [](int64_t value) { return value == 0; });
  candidate.advancesPressureClosure =
      closureGroup && closureGroup->units.contains(&unit);
  return candidate;
}

static bool isBetterLookaheadUnit(VPTOSUnit *lhs, VPTOSUnit *rhs,
                                  const VPTORegPressureTracker &tracker,
                                  ArrayRef<VPTORegPressureSet> pressureSets) {
  VPTORegPressureEvaluation lhsPressure = tracker.evaluate(*lhs);
  VPTORegPressureEvaluation rhsPressure = tracker.evaluate(*rhs);
  int64_t lhsExcess = getEvaluationExcess(lhsPressure, pressureSets);
  int64_t rhsExcess = getEvaluationExcess(rhsPressure, pressureSets);
  if (lhsExcess != rhsExcess) {
    return lhsExcess < rhsExcess;
  }
  int64_t lhsCost = getEvaluationPressure(lhsPressure, pressureSets);
  int64_t rhsCost = getEvaluationPressure(rhsPressure, pressureSets);
  if (lhsCost != rhsCost) {
    return lhsCost < rhsCost;
  }
  return lhs->getOriginalIndex() < rhs->getOriginalIndex();
}

static void updateLookaheadPressure(VPTOSchedCandidate &candidate,
                                    const VPTORegPressureTracker &tracker) {
  candidate.lookaheadEnd.assign(tracker.getCurrent().begin(),
                                tracker.getCurrent().end());
  for (auto [index, value] : llvm::enumerate(candidate.lookaheadEnd)) {
    candidate.lookaheadPeak[index] =
        std::max(candidate.lookaheadPeak[index], value);
  }
}

static LogicalResult discoverLookaheadSuccessors(
    VPTOSUnit &selected, VPTOSchedDirection direction,
    const VPTOSchedBoundary &boundary, VPTOSchedulingBudget &budget,
    DenseMap<VPTOSUnit *, unsigned> &remaining, DenseSet<VPTOSUnit *> &queued,
    SmallVectorImpl<VPTOSUnit *> &ready) {
  ArrayRef<VPTOSchedEdge *> edges = direction == VPTOSchedDirection::Top
                                        ? selected.getSuccessors()
                                        : selected.getPredecessors();
  for (VPTOSchedEdge *edge : edges) {
    if (!budget.consume()) {
      return failure();
    }
    if (!edge->isMust()) {
      continue;
    }
    VPTOSUnit *neighbor = direction == VPTOSchedDirection::Top
                              ? edge->getSuccessor()
                              : edge->getPredecessor();
    bool alreadyHandled =
        boundary.isScheduled(neighbor) || queued.contains(neighbor);
    if (alreadyHandled) {
      continue;
    }
    auto [position, inserted] = remaining.try_emplace(
        neighbor, boundary.getRemainingDependencyCount(*neighbor));
    (void)inserted;
    if (position->second == 0) {
      continue;
    }
    --position->second;
    if (position->second != 0) {
      continue;
    }
    ready.push_back(neighbor);
    queued.insert(neighbor);
  }
  return success();
}

static LogicalResult runCandidateLookahead(VPTOSchedCandidate &candidate,
                                           const VPTOSchedBoundary &boundary,
                                           const VPTOSchedModel &model,
                                           VPTOSchedulingBudget &budget) {
  constexpr unsigned kLookaheadDepth = 8;
  VPTORegPressureTracker tracker = boundary.getPressureTracker();
  SmallVector<VPTOSUnit *, kLookaheadDepth> ready{candidate.unit};
  DenseMap<VPTOSUnit *, unsigned> remaining;
  DenseSet<VPTOSUnit *> queued{candidate.unit};
  for (unsigned step = 0; step < kLookaheadDepth && !ready.empty(); ++step) {
    if (!budget.consume(ready.size())) {
      return failure();
    }
    auto best = llvm::min_element(ready, [&](VPTOSUnit *lhs, VPTOSUnit *rhs) {
      return isBetterLookaheadUnit(lhs, rhs, tracker, model.getPressureSets());
    });
    VPTOSUnit *selected = *best;
    ready.erase(best);
    if (failed(tracker.commit(*selected))) {
      return failure();
    }
    ++candidate.lookaheadSteps;
    updateLookaheadPressure(candidate, tracker);
    if (failed(discoverLookaheadSuccessors(*selected, boundary.getDirection(),
                                           boundary, budget, remaining, queued,
                                           ready))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<VPTOSchedCandidate>
buildCandidate(VPTOSUnit &unit, const VPTOSchedBoundary &boundary,
               const VPTOSchedModel &model, VPTOSchedulingBudget &budget,
               const PressureClosureGroup *closureGroup = nullptr) {
  VPTOSchedCandidate candidate =
      initializeCandidate(unit, boundary, closureGroup);
  if (closureGroup && closureGroup->target) {
    return candidate;
  }
  if (!isCriticalPressure(boundary, model)) {
    return candidate;
  }
  // Follow only dependency nodes made ready by this candidate. This bounded
  // simulation estimates whether the newly opened chain soon reaches a
  // live-range-closing consumer without copying the full ready queue.
  if (failed(runCandidateLookahead(candidate, boundary, model, budget))) {
    return mlir::failure();
  }
  return candidate;
}

static bool regionIsStillContiguous(const VPTOSchedRegion &region) {
  if (!region.block || region.operations.empty()) {
    return false;
  }
  Operation *current = region.precedingBoundary
                           ? region.precedingBoundary->getNextNode()
                           : &region.block->front();
  for (Operation *expected : region.operations) {
    if (!current || current != expected ||
        current->getBlock() != region.block) {
      return false;
    }
    current = current->getNextNode();
  }
  return current == region.followingBoundary;
}

static void setFailure(VPTOScheduleFailure &failure,
                       VPTOScheduleFailureKind kind, StringRef detail) {
  failure.kind = kind;
  failure.detail = detail.str();
}

static void setWorkBudgetFailure(VPTOScheduleFailure &failure,
                                 const VPTOSchedulingBudget &budget) {
  uint64_t actual = budget.getUsed();
  if (actual != std::numeric_limits<uint64_t>::max()) {
    ++actual;
  }
  failure = {VPTOScheduleFailureKind::Budget,
             "work-units",
             actual,
             budget.getLimit(),
             {}};
}

static bool isNearPressureSet(const VPTOSchedBoundary &boundary,
                              ArrayRef<VPTORegPressureSet> pressureSets,
                              unsigned index) {
  bool invalidIndex = index >= pressureSets.size();
  bool hasNoLimit = !invalidIndex && !pressureSets[index].limit;
  if (invalidIndex || hasNoLimit) {
    return false;
  }
  int64_t current = boundary.getPressureTracker().getCurrent()[index];
  int64_t limit = static_cast<int64_t>(*pressureSets[index].limit);
  return current * 2 >= limit;
}

static std::optional<unsigned>
selectHighestPressureSet(const VPTOSchedBoundary &boundary,
                         ArrayRef<VPTORegPressureSet> pressureSets) {
  std::optional<unsigned> selected;
  ArrayRef<int64_t> current = boundary.getPressureTracker().getCurrent();
  for (unsigned index = 0; index < pressureSets.size(); ++index) {
    if (!isNearPressureSet(boundary, pressureSets, index)) {
      continue;
    }
    if (!selected) {
      selected = index;
      continue;
    }
    int64_t selectedLimit =
        static_cast<int64_t>(*pressureSets[*selected].limit);
    int64_t candidateLimit =
        static_cast<int64_t>(*pressureSets[index].limit);
    int64_t selectedExcess = std::max<int64_t>(
        0, current[*selected] - selectedLimit);
    int64_t candidateExcess =
        std::max<int64_t>(0, current[index] - candidateLimit);
    bool hasHigherExcess = candidateExcess > selectedExcess;
    bool hasHigherRatio =
        candidateExcess == selectedExcess &&
        current[index] * selectedLimit >
            current[*selected] * candidateLimit;
    if (hasHigherExcess || hasHigherRatio) {
      selected = index;
    }
  }
  return selected;
}

static bool valueContributesToPressureSet(const VPTOSchedModel &model,
                                          Value value,
                                          VPTOPressureSetID pressureSet) {
  return llvm::any_of(
      model.getPressure(value), [&](const VPTORegPressureContribution &item) {
        return item.pressureSet == pressureSet && item.units > 0;
      });
}

static unsigned getValueOriginalIndex(Value value, const VPTOSchedDAG &dag) {
  Operation *definingOp = value.getDefiningOp();
  VPTOSUnit *unit = definingOp ? dag.lookup(definingOp) : nullptr;
  return unit ? unit->getOriginalIndex() : std::numeric_limits<unsigned>::max();
}

static bool isValueBefore(Value lhs, Value rhs, const VPTOSchedDAG &dag) {
  unsigned lhsIndex = getValueOriginalIndex(lhs, dag);
  unsigned rhsIndex = getValueOriginalIndex(rhs, dag);
  if (lhsIndex != rhsIndex) {
    return lhsIndex < rhsIndex;
  }
  auto lhsResult = dyn_cast<OpResult>(lhs);
  auto rhsResult = dyn_cast<OpResult>(rhs);
  if (lhsResult && rhsResult) {
    return lhsResult.getResultNumber() < rhsResult.getResultNumber();
  }
  return llvm::find(dag.getLiveIns(), lhs) < llvm::find(dag.getLiveIns(), rhs);
}

static SmallVector<Value, 2>
buildPressureBundle(Value value, VPTOPressureSetID pressureSet,
                    const VPTORegPressureTracker &tracker,
                    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
                    DenseSet<Operation *> &checkedDefinitions) {
  Operation *definingOp = value.getDefiningOp();
  VPTOSUnit *definingUnit = definingOp ? dag.lookup(definingOp) : nullptr;
  if (!definingUnit) {
    return {value};
  }
  if (!checkedDefinitions.insert(definingOp).second) {
    return {};
  }
  SmallVector<Value, 2> bundle;
  for (Value result : definingOp->getResults()) {
    bool contributes =
        tracker.isLive(result) &&
        valueContributesToPressureSet(model, result, pressureSet);
    if (contributes) {
      bundle.push_back(result);
    }
  }
  return bundle;
}

struct ClosureBundleUsers {
  bool hasUnscheduled = false;
  unsigned count = 0;
};

static FailureOr<ClosureBundleUsers> collectClosureBundleUsers(
    ArrayRef<Value> bundle, const VPTOSchedBoundary &boundary,
    const VPTOSchedDAG &dag, VPTOSchedulingBudget &budget) {
  DenseSet<VPTOSUnit *> users;
  ClosureBundleUsers result;
  for (Value value : bundle) {
    for (Operation *user : value.getUsers()) {
      if (!budget.consume()) {
        return failure();
      }
      VPTOSUnit *unit = dag.lookup(user);
      if (!unit) {
        continue;
      }
      users.insert(unit);
      result.hasUnscheduled |= !boundary.isScheduled(unit);
    }
  }
  result.count = users.size();
  return result;
}

static LogicalResult collectPressureClosureBundles(
    const VPTOSchedBoundary &boundary, const VPTOSchedModel &model,
    const VPTOSchedDAG &dag, unsigned pressureSet,
    VPTOSchedulingBudget &budget,
    SmallVectorImpl<SmallVector<Value, 2>> &bundles) {
  constexpr unsigned kMaxClosureTargetBundles = 1;
  constexpr unsigned kMaxClosureDirectUsers = 8;
  VPTOPressureSetID pressureSetID = model.getPressureSets()[pressureSet].id;
  const VPTORegPressureTracker &tracker = boundary.getPressureTracker();
  SmallVector<Value> liveValues(tracker.getLiveValues().begin(),
                                tracker.getLiveValues().end());
  llvm::sort(liveValues, [&](Value lhs, Value rhs) {
    return isValueBefore(lhs, rhs, dag);
  });
  DenseSet<Operation *> checkedDefinitions;
  for (Value value : liveValues) {
    if (!budget.consume()) {
      return failure();
    }
    if (!valueContributesToPressureSet(model, value, pressureSetID)) {
      continue;
    }
    SmallVector<Value, 2> bundle = buildPressureBundle(
        value, pressureSetID, tracker, model, dag, checkedDefinitions);
    if (bundle.empty()) {
      continue;
    }
    FailureOr<ClosureBundleUsers> users =
        collectClosureBundleUsers(bundle, boundary, dag, budget);
    if (failed(users)) {
      return failure();
    }
    if (!users->hasUnscheduled || users->count > kMaxClosureDirectUsers) {
      continue;
    }
    bundles.push_back(std::move(bundle));
    bool reachedBundleLimit = bundles.size() == kMaxClosureTargetBundles;
    if (reachedBundleLimit) {
      break;
    }
  }
  return success();
}

static bool isBetterClosureGroup(const PressureClosureGroup &candidate,
                                 const PressureClosureGroup &selected,
                                 unsigned pressureSet) {
  if (candidate.peak[pressureSet] != selected.peak[pressureSet]) {
    return candidate.peak[pressureSet] < selected.peak[pressureSet];
  }
  if (candidate.end[pressureSet] != selected.end[pressureSet]) {
    return candidate.end[pressureSet] < selected.end[pressureSet];
  }
  if (candidate.steps != selected.steps) {
    return candidate.steps < selected.steps;
  }
  return candidate.target->getOriginalIndex() <
         selected.target->getOriginalIndex();
}

struct PressureClosureSimulation {
  SmallVector<VPTOSUnit *, 8> ready;
  DenseMap<VPTOSUnit *, unsigned> remaining;
  DenseSet<VPTOSUnit *> discovered;
  DenseSet<VPTOSUnit *> expanded;
  DenseSet<VPTOSUnit *> enqueued;
  DenseSet<VPTOSUnit *> core;
  DenseSet<VPTOSUnit *> scheduled;
};

static FailureOr<unsigned> getRemainingClosureDependencies(
    VPTOSUnit &unit, const VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, const PressureClosureSimulation &simulation) {
  unsigned dependencies = boundary.getRemainingDependencyCount(unit);
  for (VPTOSchedEdge *edge : unit.getPredecessors()) {
    if (!budget.consume()) {
      return failure();
    }
    bool selectedDependency =
        edge->isMust() && simulation.scheduled.contains(edge->getPredecessor());
    if (!selectedDependency) {
      continue;
    }
    if (dependencies == 0) {
      return failure();
    }
    --dependencies;
  }
  return dependencies;
}

static LogicalResult enqueueClosurePredecessors(
    VPTOSUnit &unit, const VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, const PressureClosureSimulation &simulation,
    SmallVectorImpl<std::pair<VPTOSUnit *, bool>> &worklist) {
  for (VPTOSchedEdge *edge : unit.getPredecessors()) {
    if (!budget.consume()) {
      return failure();
    }
    if (!edge->isMust()) {
      continue;
    }
    VPTOSUnit *predecessor = edge->getPredecessor();
    bool needsPredecessor = !boundary.isScheduled(predecessor) &&
                            !simulation.discovered.contains(predecessor);
    if (needsPredecessor) {
      worklist.push_back({predecessor, false});
    }
  }
  return success();
}

static FailureOr<bool> discoverClosureUnit(
    VPTOSUnit &root, bool isCore, const VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, PressureClosureSimulation &simulation) {
  SmallVector<std::pair<VPTOSUnit *, bool>, 8> worklist{{&root, isCore}};
  while (!worklist.empty()) {
    if (!budget.consume()) {
      return failure();
    }
    auto [unit, coreUnit] = worklist.pop_back_val();
    if (boundary.isScheduled(unit)) {
      continue;
    }
    if (coreUnit) {
      simulation.core.insert(unit);
    }
    bool newlyDiscovered = simulation.discovered.insert(unit).second;
    if (newlyDiscovered &&
        simulation.discovered.size() > kMaxClosureGroupNodes) {
      return false;
    }
    auto found = simulation.remaining.find(unit);
    if (found == simulation.remaining.end()) {
      FailureOr<unsigned> dependencies =
          getRemainingClosureDependencies(*unit, boundary, budget, simulation);
      if (failed(dependencies)) {
        return failure();
      }
      found = simulation.remaining.try_emplace(unit, *dependencies).first;
    }
    if (found->second == 0) {
      if (simulation.enqueued.insert(unit).second) {
        simulation.ready.push_back(unit);
      }
      continue;
    }
    if (!simulation.expanded.insert(unit).second) {
      continue;
    }
    if (failed(enqueueClosurePredecessors(*unit, boundary, budget, simulation,
                                          worklist))) {
      return failure();
    }
  }
  return true;
}

static FailureOr<int64_t> getLiveSupportPressure(
    const PressureClosureSimulation &simulation,
    const VPTORegPressureTracker &tracker, const VPTOSchedModel &model,
    unsigned pressureSet, VPTOSchedulingBudget &budget) {
  VPTOPressureSetID pressureSetID = model.getPressureSets()[pressureSet].id;
  int64_t pressure = 0;
  DenseSet<Value> countedValues;
  for (VPTOSUnit *unit : simulation.scheduled) {
    if (simulation.core.contains(unit)) {
      continue;
    }
    for (Value result : unit->getOperation()->getResults()) {
      if (!budget.consume()) {
        return failure();
      }
      Value representative = tracker.getPressureRepresentative(result);
      bool isLive = tracker.isLive(representative);
      bool isFirstContribution = countedValues.insert(representative).second;
      if (!isLive || !isFirstContribution) {
        continue;
      }
      for (const VPTORegPressureContribution &contribution :
           model.getPressure(representative)) {
        if (contribution.pressureSet != pressureSetID) {
          continue;
        }
        int64_t updated = 0;
        if (llvm::AddOverflow(
                pressure, static_cast<int64_t>(contribution.units), updated)) {
          return failure();
        }
        pressure = updated;
      }
    }
  }
  return pressure;
}

static FailureOr<bool>
discoverClosureTargets(ArrayRef<Value> targetValues,
                       const VPTOSchedBoundary &boundary,
                       const VPTOSchedDAG &dag, VPTOSchedulingBudget &budget,
                       PressureClosureSimulation &simulation) {
  for (Value targetValue : targetValues) {
    for (Operation *user : targetValue.getUsers()) {
      if (!budget.consume()) {
        return failure();
      }
      VPTOSUnit *unit = dag.lookup(user);
      if (!unit || boundary.isScheduled(unit)) {
        continue;
      }
      FailureOr<bool> discovered = discoverClosureUnit(
          *unit, true, boundary, budget, simulation);
      bool incomplete = failed(discovered) || !*discovered;
      if (incomplete) {
        return discovered;
      }
    }
  }
  return true;
}

static void initializeClosureGroup(ArrayRef<Value> targetValues,
                                   ArrayRef<int64_t> initialPressure,
                                   const VPTOSchedDAG &dag,
                                   unsigned pressureSet,
                                   PressureClosureGroup &group) {
  group.clear();
  group.pressureSet = pressureSet;
  group.targetValues.assign(targetValues.begin(), targetValues.end());
  for (Value value : targetValues) {
    VPTOSUnit *unit = getValueOriginalIndex(value, dag) ==
                              std::numeric_limits<unsigned>::max()
                          ? nullptr
                          : dag.lookup(value.getDefiningOp());
    group.bundleOriginalIndices.push_back(
        unit ? unit->getOriginalIndex() : std::numeric_limits<unsigned>::max());
  }
  group.peak.assign(initialPressure.begin(), initialPressure.end());
  group.end.assign(initialPressure.begin(), initialPressure.end());
}

static VPTOSUnit *
selectClosureReadyUnit(ArrayRef<VPTOSUnit *> ready,
                       const VPTORegPressureTracker &tracker,
                       ArrayRef<VPTORegPressureSet> pressureSets) {
  return *llvm::min_element(ready, [&](VPTOSUnit *lhs, VPTOSUnit *rhs) {
    return isBetterLookaheadUnit(lhs, rhs, tracker, pressureSets);
  });
}

static LogicalResult commitClosureUnit(VPTOSUnit &unit,
                                       VPTORegPressureTracker &tracker,
                                       PressureClosureSimulation &simulation,
                                       PressureClosureGroup &group) {
  auto position = llvm::find(simulation.ready, &unit);
  simulation.ready.erase(position);
  if (failed(tracker.commit(unit))) {
    return mlir::failure();
  }
  simulation.scheduled.insert(&unit);
  group.units.insert(&unit);
  group.witness.push_back(&unit);
  group.target = &unit;
  ++group.steps;
  group.end.assign(tracker.getCurrent().begin(), tracker.getCurrent().end());
  for (auto [index, pressure] : llvm::enumerate(group.end)) {
    group.peak[index] = std::max(group.peak[index], pressure);
  }
  return success();
}

static FailureOr<bool> closureMeetsTarget(
    ArrayRef<Value> targetValues, int64_t start, int64_t limit,
    unsigned pressureSet, const PressureClosureSimulation &simulation,
    const VPTORegPressureTracker &tracker, const VPTOSchedModel &model,
    VPTOSchedulingBudget &budget, PressureClosureGroup &group) {
  bool targetsAreDead = llvm::all_of(
      targetValues, [&](Value value) { return !tracker.isLive(value); });
  bool avoidsNewExcess = group.peak[pressureSet] <= std::max(start, limit);
  if (!targetsAreDead || !avoidsNewExcess) {
    return false;
  }
  FailureOr<int64_t> supportPressure =
      getLiveSupportPressure(simulation, tracker, model, pressureSet, budget);
  bool invalidSupport =
      failed(supportPressure) || *supportPressure > group.end[pressureSet];
  if (invalidSupport) {
    return mlir::failure();
  }
  group.supportPressure = *supportPressure;
  group.effectiveEnd = group.end[pressureSet] - *supportPressure;
  if (group.effectiveEnd > start) {
    return false;
  }
  group.netRelief = start - group.effectiveEnd;
  return true;
}

static FailureOr<bool>
advanceClosureSuccessors(VPTOSUnit &selected, const VPTOSchedBoundary &boundary,
                         VPTOSchedulingBudget &budget,
                         PressureClosureSimulation &simulation) {
  for (VPTOSchedEdge *edge : selected.getSuccessors()) {
    if (!budget.consume()) {
      return mlir::failure();
    }
    bool skipEdge =
        !edge->isMust() || boundary.isScheduled(edge->getSuccessor());
    if (skipEdge) {
      continue;
    }
    VPTOSUnit *successor = edge->getSuccessor();
    bool followsClosure = simulation.core.contains(&selected) &&
                          edge->getKind() == VPTOSchedEdgeKind::Data;
    if (!followsClosure && !simulation.discovered.contains(successor)) {
      continue;
    }
    if (followsClosure) {
      simulation.core.insert(successor);
    }
    auto position = simulation.remaining.find(successor);
    if (position != simulation.remaining.end()) {
      if (position->second == 0) {
        continue;
      }
      --position->second;
    }
    FailureOr<bool> discovered = discoverClosureUnit(
        *successor, followsClosure, boundary, budget, simulation);
    bool incomplete = failed(discovered) || !*discovered;
    if (incomplete) {
      return discovered;
    }
  }
  return true;
}

static FailureOr<bool> populatePressureClosureGroup(
    ArrayRef<Value> targetValues, const VPTOSchedBoundary &boundary,
    const VPTOSchedModel &model, const VPTOSchedDAG &dag, unsigned pressureSet,
    VPTOSchedulingBudget &budget, PressureClosureGroup &group) {
  VPTORegPressureTracker simulatedTracker = boundary.getPressureTracker();
  ArrayRef<int64_t> initialPressure = simulatedTracker.getCurrent();
  ArrayRef<VPTORegPressureSet> pressureSets = model.getPressureSets();
  int64_t start = initialPressure[pressureSet];
  int64_t limit = static_cast<int64_t>(*pressureSets[pressureSet].limit);

  PressureClosureSimulation simulation;
  FailureOr<bool> discovered =
      discoverClosureTargets(targetValues, boundary, dag, budget, simulation);
  bool incomplete = failed(discovered) || !*discovered;
  if (incomplete) {
    return discovered;
  }
  initializeClosureGroup(targetValues, initialPressure, dag, pressureSet,
                         group);
  for (unsigned step = 0;
       step < kMaxClosureGroupNodes && !simulation.ready.empty(); ++step) {
    if (!budget.consume(simulation.ready.size())) {
      return failure();
    }
    VPTOSUnit *selected = selectClosureReadyUnit(
        simulation.ready, simulatedTracker, pressureSets);
    if (failed(commitClosureUnit(*selected, simulatedTracker, simulation,
                                 group))) {
      return failure();
    }
    FailureOr<bool> complete =
        closureMeetsTarget(targetValues, start, limit, pressureSet, simulation,
                           simulatedTracker, model, budget, group);
    bool finished = failed(complete) || *complete;
    if (finished) {
      return complete;
    }
    FailureOr<bool> advanced =
        advanceClosureSuccessors(*selected, boundary, budget, simulation);
    bool stopped = failed(advanced) || !*advanced;
    if (stopped) {
      return advanced;
    }
  }
  return false;
}

static LogicalResult
refreshPressureClosureGroup(PressureClosureGroup &closureGroup,
                            const VPTOSchedBoundary &boundary,
                            const VPTOSchedModel &model,
                            const VPTOSchedDAG &dag,
                            VPTOSchedulingBudget &budget) {
  ArrayRef<VPTORegPressureSet> pressureSets = model.getPressureSets();
  std::optional<unsigned> pressureSet =
      selectHighestPressureSet(boundary, pressureSets);
  bool keepsCurrentClosure =
      closureGroup.target && closureGroup.pressureSet &&
      !boundary.isScheduled(closureGroup.target) &&
      pressureSet == closureGroup.pressureSet;
  if (keepsCurrentClosure) {
    return success();
  }
  closureGroup.clear();

  if (!pressureSet) {
    return success();
  }

  SmallVector<SmallVector<Value, 2>, 2> bundles;
  LogicalResult collected = collectPressureClosureBundles(
      boundary, model, dag, *pressureSet, budget, bundles);
  if (failed(collected)) {
    return mlir::failure();
  }
  PressureClosureGroup selected;
  for (ArrayRef<Value> bundle : bundles) {
    PressureClosureGroup candidate;
    FailureOr<bool> built = populatePressureClosureGroup(
        bundle, boundary, model, dag, *pressureSet, budget, candidate);
    if (failed(built)) {
      return failure();
    }
    if (*built &&
        (!selected.target ||
         isBetterClosureGroup(candidate, selected, *pressureSet))) {
      selected = std::move(candidate);
    }
  }
  closureGroup = std::move(selected);
  return success();
}

static FailureOr<SmallVector<VPTOSchedCandidate>>
buildCandidates(const VPTOSchedBoundary &boundary, const VPTOSchedModel &model,
                VPTOSchedulingBudget &budget,
                VPTOScheduleFailure &failure,
                const PressureClosureGroup *closureGroup = nullptr) {
  SmallVector<VPTOSchedCandidate> candidates;
  candidates.reserve(boundary.getAvailable().size());
  for (VPTOSUnit *unit : boundary.getAvailable()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    FailureOr<VPTOSchedCandidate> candidate =
        buildCandidate(*unit, boundary, model, budget, closureGroup);
    if (failed(candidate)) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    candidates.push_back(std::move(*candidate));
  }
  return candidates;
}

static VPTOSUnit *getNextClosureWitness(
    const PressureClosureGroup &closureGroup,
    const VPTOSchedBoundary &boundary) {
  for (VPTOSUnit *unit : closureGroup.witness) {
    if (!boundary.isScheduled(unit)) {
      return unit;
    }
  }
  return nullptr;
}

static bool exceedsPressureLimit(const VPTORegPressureTracker &tracker,
                                 const VPTOSchedModel &model) {
  ArrayRef<int64_t> current = tracker.getCurrent();
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    if (pressureSet.limit &&
        current[index] > static_cast<int64_t>(*pressureSet.limit)) {
      return true;
    }
  }
  return false;
}

static FailureOr<bool> extendRecoveryTargets(
    const VPTOSchedCandidate &candidate,
    const PressureClosureGroup &closureGroup, const VPTOSchedBoundary &boundary,
    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
    VPTOSchedulingBudget &budget, PressureClosureGroup &extendedGroup) {
  SmallVector<Value, 4> targets(closureGroup.targetValues.begin(),
                                closureGroup.targetValues.end());
  VPTOPressureSetID pressureSetID =
      model.getPressureSets()[*closureGroup.pressureSet].id;
  bool addedTarget = false;
  for (Value result : candidate.unit->getOperation()->getResults()) {
    if (!budget.consume()) {
      return failure();
    }
    if (valueContributesToPressureSet(model, result, pressureSetID)) {
      targets.push_back(result);
      addedTarget = true;
    }
  }
  if (!addedTarget) {
    extendedGroup = closureGroup;
    return true;
  }
  FailureOr<bool> built = populatePressureClosureGroup(
      targets, boundary, model, dag, *closureGroup.pressureSet, budget,
      extendedGroup);
  bool incomplete = failed(built) || !*built;
  if (incomplete) {
    return built;
  }
  return extendedGroup.units.contains(candidate.unit);
}

static FailureOr<bool>
replayExtendedRecovery(const VPTOSchedCandidate &candidate,
                       const PressureClosureGroup &extendedGroup,
                       const VPTOSchedBoundary &boundary,
                       const VPTOSchedModel &model,
                       VPTOSchedulingBudget &budget) {
  VPTORegPressureTracker tracker = boundary.getPressureTracker();
  bool commitFailed =
      !budget.consume() || failed(tracker.commit(*candidate.unit));
  if (commitFailed) {
    return mlir::failure();
  }
  if (exceedsPressureLimit(tracker, model)) {
    return false;
  }
  for (VPTOSUnit *unit : extendedGroup.witness) {
    if (unit == candidate.unit || boundary.isScheduled(unit)) {
      continue;
    }
    bool commitFailed = !budget.consume() || failed(tracker.commit(*unit));
    if (commitFailed) {
      return failure();
    }
    if (exceedsPressureLimit(tracker, model)) {
      return false;
    }
  }
  return true;
}

static FailureOr<bool> extendZeroReliefRecovery(
    const VPTOSchedCandidate &candidate,
    const PressureClosureGroup &closureGroup, const VPTOSchedBoundary &boundary,
    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
    VPTOSchedulingBudget &budget, PressureClosureGroup &extendedGroup) {
  if (!closureGroup.pressureSet) {
    return mlir::failure();
  }
  FailureOr<bool> extended = extendRecoveryTargets(
      candidate, closureGroup, boundary, model, dag, budget, extendedGroup);
  bool incomplete = failed(extended) || !*extended;
  if (incomplete) {
    return extended;
  }
  return replayExtendedRecovery(candidate, extendedGroup, boundary, model,
                                budget);
}

static bool increasesPressureRisk(const VPTOSchedModel &model,
                                  ArrayRef<int64_t> currentPressure,
                                  const VPTOSchedCandidate &candidate) {
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    if (!pressureSet.limit) {
      continue;
    }
    int64_t currentExcess = std::max<int64_t>(
        0, currentPressure[index] - static_cast<int64_t>(*pressureSet.limit));
    if (candidate.pressure.projectedExcess[index] > currentExcess) {
      return true;
    }
    int64_t limit = static_cast<int64_t>(*pressureSet.limit);
    bool critical = currentPressure[index] * 2 >= limit;
    if (critical &&
        candidate.pressure.projected[index] > currentPressure[index])
      return true;
  }
  return false;
}

static bool allCandidatesIncreasePressureRisk(
    const VPTOSchedModel &model, ArrayRef<int64_t> currentPressure,
    ArrayRef<VPTOSchedCandidate> candidates) {
  return !candidates.empty() &&
         llvm::all_of(candidates, [&](const VPTOSchedCandidate &candidate) {
           return increasesPressureRisk(model, currentPressure, candidate);
         });
}

static FailureOr<bool>
advanceForPressure(VPTOSchedBoundary &boundary,
                   ArrayRef<VPTOSchedCandidate> candidates,
                   const VPTOSchedModel &model,
                   VPTOSchedulingBudget &budget,
                   VPTOScheduleFailure &failure) {
  if (!allCandidatesIncreasePressureRisk(
          model, boundary.getPressureTracker().getCurrent(), candidates)) {
    return false;
  }
  if (!boundary.getNextPending()) {
    return false;
  }
  FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
  if (failed(advanced)) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  return *advanced;
}

static LogicalResult advanceReplayBoundary(VPTOSchedBoundary &boundary,
                                           VPTOSchedulingBudget &budget,
                                           VPTOScheduleFailure &failure,
                                           StringRef missingEventDetail) {
  FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
  if (failed(advanced)) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  if (!*advanced) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               missingEventDetail);
    return mlir::failure();
  }
  return success();
}

static LogicalResult replayPressureDrivenIdle(
    const VPTOScheduleEntry &entry, VPTOSchedBoundary &boundary,
    const VPTOSchedModel &model,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (!entry.pressureDrivenIdle) {
    return success();
  }

  bool advancedAtLeastOnce = false;
  unsigned currentCycle = boundary.getCurrentCycle();
  while (currentCycle < entry.issueCycle) {
    FailureOr<SmallVector<VPTOSchedCandidate>> candidates =
        buildCandidates(boundary, model, budget, failure);
    if (failed(candidates)) {
      return mlir::failure();
    }
    if (!allCandidatesIncreasePressureRisk(
            model, boundary.getPressureTracker().getCurrent(), *candidates)) {
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                 "pressure idle occurred while a non-growing-risk candidate was "
                 "available");
      return mlir::failure();
    }
    if (failed(advanceReplayBoundary(
            boundary, budget, failure,
            "pressure idle has no pending dependency event"))) {
      return mlir::failure();
    }
    advancedAtLeastOnce = true;
    currentCycle = boundary.getCurrentCycle();
  }
  if (!advancedAtLeastOnce) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               "pressure idle does not advance the logical cycle");
    return mlir::failure();
  }
  return success();
}

static LogicalResult replayRecoveryDrivenIdle(
    const VPTOScheduleEntry &entry, VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (!entry.recoveryDrivenIdle) {
    return success();
  }
  bool advancedAtLeastOnce = false;
  unsigned currentCycle = boundary.getCurrentCycle();
  while (currentCycle < entry.issueCycle) {
    if (failed(advanceReplayBoundary(
            boundary, budget, failure,
            "recovery idle has no pending dependency event"))) {
      return mlir::failure();
    }
    advancedAtLeastOnce = true;
    currentCycle = boundary.getCurrentCycle();
  }
  if (!advancedAtLeastOnce) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               "recovery idle does not advance the logical cycle");
    return mlir::failure();
  }
  return success();
}

static LogicalResult replayMandatoryDependencyIdle(
    const VPTOScheduleEntry &entry, VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  bool noCandidate = boundary.getAvailable().empty();
  while (noCandidate) {
    unsigned currentCycle = boundary.getCurrentCycle();
    if (currentCycle >= entry.issueCycle) {
      break;
    }
    if (failed(
            advanceReplayBoundary(boundary, budget, failure,
                                  "replay has no pending dependency event"))) {
      return mlir::failure();
    }
    noCandidate = boundary.getAvailable().empty();
  }
  return success();
}

static LogicalResult validateScheduleInputs(const VPTOSchedModel &model,
                                            const VPTOSchedDAG &dag,
                                            const VPTOSchedulerLimits &limits,
                                            VPTOScheduleFailure &failure) {
  uint64_t nodeCount = dag.getUnits().size();
  uint64_t edgeCount = dag.getEdges().size();
  if (nodeCount > limits.maxNodes) {
    failure = {VPTOScheduleFailureKind::Budget,
               "nodes",
               nodeCount,
               limits.maxNodes,
               {}};
    return mlir::failure();
  }
  if (edgeCount > limits.maxEdges) {
    failure = {VPTOScheduleFailureKind::Budget,
               "edges",
               edgeCount,
               limits.maxEdges,
               {}};
    return mlir::failure();
  }
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (!model.getSchedClass(unit->getOperation()).known) {
      setFailure(failure, VPTOScheduleFailureKind::InvalidModel,
                 "region contains an unknown scheduling class");
      return mlir::failure();
    }
  }
  return success();
}

static LogicalResult ensureScheduleCandidate(VPTOSchedBoundary &boundary,
                                             VPTOSchedulingBudget &budget,
                                             VPTOScheduleFailure &failure) {
  if (!boundary.getAvailable().empty()) {
    return success();
  }
  FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
  if (failed(advanced)) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  if (!*advanced) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "no candidate or pending dependency event remains");
    return mlir::failure();
  }
  return success();
}

static FailureOr<SmallVector<VPTOSchedCandidate>>
buildCurrentCandidates(VPTOSchedBoundary &boundary,
                       PressureClosureGroup &closureGroup,
                       const VPTOSchedModel &model, const VPTOSchedDAG &dag,
                       VPTOSchedulingBudget &budget,
                       VPTOScheduleFailure &failure, bool &zeroRelief) {
  if (failed(refreshPressureClosureGroup(closureGroup, boundary, model, dag,
                                         budget))) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  zeroRelief = closureGroup.target && closureGroup.netRelief == 0;
  const PressureClosureGroup *rankingClosure =
      zeroRelief ? nullptr : &closureGroup;
  FailureOr<SmallVector<VPTOSchedCandidate>> candidates =
      buildCandidates(boundary, model, budget, failure, rankingClosure);
  if (failed(candidates)) {
    return mlir::failure();
  }
  if (zeroRelief && !getNextClosureWitness(closureGroup, boundary)) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "zero-relief closure has no remaining recovery witness");
    return mlir::failure();
  }
  return candidates;
}

static FailureOr<SmallVector<VPTOSchedCandidate>>
prepareCandidates(VPTOSchedBoundary &boundary,
                  PressureClosureGroup &closureGroup,
                  const VPTOSchedModel &model, const VPTOSchedDAG &dag,
                  VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure,
                  bool &pressureDrivenIdle) {
  bool advancedForPressure = false;
  do {
    if (failed(ensureScheduleCandidate(boundary, budget, failure))) {
      return mlir::failure();
    }
    bool zeroRelief = false;
    FailureOr<SmallVector<VPTOSchedCandidate>> candidates =
        buildCurrentCandidates(boundary, closureGroup, model, dag, budget,
                               failure, zeroRelief);
    if (failed(candidates)) {
      return mlir::failure();
    }
    if (zeroRelief) {
      return candidates;
    }
    FailureOr<bool> advanced =
        advanceForPressure(boundary, *candidates, model, budget, failure);
    if (failed(advanced)) {
      return mlir::failure();
    }
    advancedForPressure = *advanced;
    pressureDrivenIdle |= advancedForPressure;
    if (!advancedForPressure) {
      return candidates;
    }
  } while (advancedForPressure);
  llvm_unreachable("pressure advancement loop must return");
}

static const VPTOSchedCandidate *
findCandidate(ArrayRef<VPTOSchedCandidate> candidates, const VPTOSUnit *unit) {
  auto position = llvm::find_if(candidates, [&](const auto &candidate) {
    return candidate.unit == unit;
  });
  return position == candidates.end() ? nullptr : &*position;
}

static FailureOr<bool> preserveOrDelayZeroRelief(
    VPTOSchedDecision &decision, const VPTOScheduleContext &context,
    const VPTOSchedCandidate &selected,
    const VPTOSchedCandidate *witnessCandidate, VPTOSUnit &witness,
    PressureClosureGroup &closureGroup, VPTOSchedBoundary &boundary,
    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (!witnessCandidate ||
      selected.criticalPath > witnessCandidate->criticalPath) {
    PressureClosureGroup extendedGroup;
    FailureOr<bool> preservesRecovery = extendZeroReliefRecovery(
        selected, closureGroup, boundary, model, dag, budget, extendedGroup);
    if (failed(preservesRecovery)) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    if (*preservesRecovery) {
      closureGroup = std::move(extendedGroup);
      decision.reason = "zero-closure-safe-interleave";
      return false;
    }
  }
  if (witnessCandidate) {
    decision = {&witness, context.direction, context.issueCycle,
                "zero-closure-recovery"};
    return false;
  }
  FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
  if (failed(advanced)) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  if (!*advanced) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "zero-relief closure recovery witness is unavailable");
    return mlir::failure();
  }
  return true;
}

static FailureOr<bool> adjustZeroReliefDecision(
    VPTOSchedDecision &decision, const VPTOScheduleContext &context,
    ArrayRef<VPTOSchedCandidate> candidates, PressureClosureGroup &closureGroup,
    VPTOSchedBoundary &boundary, const VPTOSchedModel &model,
    const VPTOSchedDAG &dag, VPTOSchedulingBudget &budget,
    VPTOScheduleFailure &failure) {
  if (!closureGroup.pressureSet) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "zero-relief closure has no pressure set");
    return mlir::failure();
  }
  VPTOSUnit *witness = getNextClosureWitness(closureGroup, boundary);
  if (!witness) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "zero-relief closure lost its recovery witness");
    return mlir::failure();
  }
  if (decision.unit == witness) {
    decision.reason = "zero-closure-recovery";
    return false;
  }
  const VPTOSchedCandidate *selected = findCandidate(candidates, decision.unit);
  if (!selected) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "zero-relief closure selected an unknown candidate");
    return mlir::failure();
  }
  const VPTOSchedCandidate *witnessCandidate =
      findCandidate(candidates, witness);
  return preserveOrDelayZeroRelief(decision, context, *selected,
                                   witnessCandidate, *witness, closureGroup,
                                   boundary, model, dag, budget, failure);
}

static const VPTOSchedCandidate *
findSafeAlternative(ArrayRef<VPTOSchedCandidate> candidates,
                    const VPTOSUnit *selectedUnit) {
  const VPTOSchedCandidate *selected = nullptr;
  for (const VPTOSchedCandidate &candidate : candidates) {
    bool isSafe = llvm::all_of(candidate.pressure.projectedExcess,
                               [](int64_t excess) { return excess == 0; });
    if (candidate.unit == selectedUnit || candidate.advancesPressureClosure ||
        !isSafe) {
      continue;
    }
    bool isBetter = !selected ||
                    candidate.criticalPath > selected->criticalPath ||
                    (candidate.criticalPath == selected->criticalPath &&
                     candidate.originalIndex < selected->originalIndex);
    if (isBetter) {
      selected = &candidate;
    }
  }
  return selected;
}

static void populateClosureDiagnostic(const PressureClosureGroup &closureGroup,
                                      VPTOScheduleDiagnostic &diagnostic) {
  diagnostic.closurePressureSet = closureGroup.pressureSet;
  diagnostic.closureBundleOriginalIndices.assign(
      closureGroup.bundleOriginalIndices.begin(),
      closureGroup.bundleOriginalIndices.end());
  if (closureGroup.target) {
    diagnostic.closureTargetOriginalIndex =
        closureGroup.target->getOriginalIndex();
  }
  diagnostic.closureGroupSize = closureGroup.units.size();
  diagnostic.closureSteps = closureGroup.steps;
  diagnostic.closureSupportPressure = closureGroup.supportPressure;
  diagnostic.closureEffectiveEnd = closureGroup.effectiveEnd;
  diagnostic.closureNetRelief = closureGroup.netRelief;
  diagnostic.closurePeak.assign(closureGroup.peak.begin(),
                                closureGroup.peak.end());
  diagnostic.closureEnd.assign(closureGroup.end.begin(),
                               closureGroup.end.end());
}

static void
populateCandidateDiagnostic(const VPTOSchedCandidate *selected,
                            const VPTOSchedCandidate *safeAlternative,
                            VPTOScheduleDiagnostic &diagnostic) {
  if (selected) {
    diagnostic.selectedCriticalPath = selected->criticalPath;
    diagnostic.selectedAdvancesClosure = selected->advancesPressureClosure;
    diagnostic.selectedProjectedPressure.assign(
        selected->pressure.projected.begin(),
        selected->pressure.projected.end());
    diagnostic.selectedReleasedPressure.assign(
        selected->pressure.released.begin(), selected->pressure.released.end());
  }
  if (!safeAlternative) {
    return;
  }
  diagnostic.safeAlternativeOriginalIndex = safeAlternative->originalIndex;
  diagnostic.safeAlternativeCriticalPath = safeAlternative->criticalPath;
  diagnostic.safeAlternativeOpensPressureFrontier =
      safeAlternative->opensPressureFrontier;
  diagnostic.safeAlternativeProjectedPressure.assign(
      safeAlternative->pressure.projected.begin(),
      safeAlternative->pressure.projected.end());
  diagnostic.safeAlternativeReleasedPressure.assign(
      safeAlternative->pressure.released.begin(),
      safeAlternative->pressure.released.end());
}

static VPTOScheduleDiagnostic buildScheduleDiagnostic(
    ArrayRef<int64_t> currentPressure, ArrayRef<VPTOSchedCandidate> candidates,
    const VPTOSUnit *selectedUnit, const PressureClosureGroup &closureGroup) {
  VPTOScheduleDiagnostic diagnostic;
  diagnostic.currentPressure.assign(currentPressure.begin(),
                                    currentPressure.end());
  diagnostic.candidateCount = candidates.size();
  populateClosureDiagnostic(closureGroup, diagnostic);
  populateCandidateDiagnostic(findCandidate(candidates, selectedUnit),
                              findSafeAlternative(candidates, selectedUnit),
                              diagnostic);
  return diagnostic;
}

class ScheduleRunner {
public:
  ScheduleRunner(const VPTOSchedModel &model, VPTOSchedDAG &dag,
                 VPTOSchedulingBudget &budget, bool collectDiagnostics,
                 const VPTOSchedStrategy &strategy,
                 VPTOScheduleFailure &failure)
      : model(model), dag(dag), budget(budget),
        collectDiagnostics(collectDiagnostics), strategy(strategy),
        failure(failure), boundary(dag, model, VPTOSchedDirection::Top) {
    result.entries.reserve(dag.getUnits().size());
  }

  FailureOr<VPTOScheduleResult> run() {
    while (!boundary.isComplete()) {
      if (failed(scheduleNext())) {
        return mlir::failure();
      }
    }
    ArrayRef<int64_t> peak = boundary.getPressureTracker().getPeak();
    result.peakPressure.assign(peak.begin(), peak.end());
    return std::move(result);
  }

private:
  LogicalResult scheduleNext();
  FailureOr<bool> chooseDecision(ArrayRef<VPTOSchedCandidate> candidates,
                                 bool pressureDrivenIdle,
                                 bool recoveryDrivenIdle);
  LogicalResult commitDecision(const VPTOSchedDecision &decision,
                               const VPTOScheduleContext &context,
                               ArrayRef<VPTOSchedCandidate> candidates,
                               bool pressureDrivenIdle,
                               bool recoveryDrivenIdle);

  const VPTOSchedModel &model;
  VPTOSchedDAG &dag;
  VPTOSchedulingBudget &budget;
  bool collectDiagnostics;
  const VPTOSchedStrategy &strategy;
  VPTOScheduleFailure &failure;
  VPTOSchedBoundary boundary;
  PressureClosureGroup closureGroup;
  VPTOScheduleResult result;
  bool carriedPressureDrivenIdle = false;
  bool carriedRecoveryDrivenIdle = false;
};

LogicalResult ScheduleRunner::scheduleNext() {
  bool pressureDrivenIdle = std::exchange(carriedPressureDrivenIdle, false);
  bool recoveryDrivenIdle = std::exchange(carriedRecoveryDrivenIdle, false);
  FailureOr<SmallVector<VPTOSchedCandidate>> candidates = prepareCandidates(
      boundary, closureGroup, model, dag, budget, failure, pressureDrivenIdle);
  if (failed(candidates)) {
    return mlir::failure();
  }
  FailureOr<bool> retry =
      chooseDecision(*candidates, pressureDrivenIdle, recoveryDrivenIdle);
  if (failed(retry)) {
    return mlir::failure();
  }
  if (*retry) {
    carriedRecoveryDrivenIdle = true;
  }
  return success();
}

FailureOr<bool>
ScheduleRunner::chooseDecision(ArrayRef<VPTOSchedCandidate> candidates,
                               bool pressureDrivenIdle,
                               bool recoveryDrivenIdle) {
  bool zeroRelief = closureGroup.target && closureGroup.netRelief == 0;
  std::optional<unsigned> closurePressureSet =
      zeroRelief ? std::nullopt : closureGroup.pressureSet;
  VPTOScheduleContext context{model,
                              dag,
                              boundary.getDirection(),
                              boundary.getCurrentCycle(),
                              boundary.getPressureTracker().getCurrent(),
                              closurePressureSet};
  std::string detail;
  FailureOr<VPTOSchedDecision> decision =
      strategy.pickCandidate(context, candidates, detail);
  if (failed(decision)) {
    setFailure(failure, VPTOScheduleFailureKind::InvalidModel, detail);
    return mlir::failure();
  }
  if (zeroRelief) {
    FailureOr<bool> retry =
        adjustZeroReliefDecision(*decision, context, candidates, closureGroup,
                                 boundary, model, dag, budget, failure);
    bool stopDecision = failed(retry) || *retry;
    if (stopDecision) {
      return retry;
    }
  }
  if (failed(commitDecision(*decision, context, candidates, pressureDrivenIdle,
                            recoveryDrivenIdle))) {
    return mlir::failure();
  }
  return false;
}

LogicalResult ScheduleRunner::commitDecision(
    const VPTOSchedDecision &decision, const VPTOScheduleContext &context,
    ArrayRef<VPTOSchedCandidate> candidates, bool pressureDrivenIdle,
    bool recoveryDrivenIdle) {
  bool selectsCandidate = findCandidate(candidates, decision.unit);
  bool matchesContext = decision.direction == context.direction &&
                        decision.issueCycle == context.issueCycle;
  if (!selectsCandidate || !matchesContext || decision.reason.empty()) {
    setFailure(failure, VPTOScheduleFailureKind::Scheduling,
               "strategy returned an invalid scheduling decision");
    return mlir::failure();
  }
  SmallVector<int64_t, 2> pressureBeforeCommit;
  if (collectDiagnostics) {
    pressureBeforeCommit.assign(context.currentPressure.begin(),
                                context.currentPressure.end());
  }
  std::string detail;
  if (failed(boundary.commit(*decision.unit, decision.issueCycle, budget,
                             detail))) {
    if (budget.hasExceeded()) {
      setWorkBudgetFailure(failure, budget);
    } else {
      setFailure(failure, VPTOScheduleFailureKind::Scheduling, detail);
    }
    return mlir::failure();
  }
  result.entries.push_back({decision.unit, decision.direction,
                            decision.issueCycle, decision.reason,
                            pressureDrivenIdle, recoveryDrivenIdle});
  if (collectDiagnostics) {
    result.diagnostics.push_back(buildScheduleDiagnostic(
        pressureBeforeCommit, candidates, decision.unit, closureGroup));
  }
  return success();
}

} // namespace

bool VPTOSchedulingBudget::consume(uint64_t amount) {
  if (amount > limit - std::min(used, limit)) {
    used = limit;
    exceeded = true;
    return false;
  }
  used += amount;
  return true;
}

FailureOr<VPTOScheduleResult>
VPTOScheduler::schedule(VPTOScheduleFailure &failure) const {
  if (failed(validateScheduleInputs(model, dag, limits, failure))) {
    return mlir::failure();
  }
  ScheduleRunner runner(model, dag, budget, collectDiagnostics, strategy,
                        failure);
  return runner.run();
}

static LogicalResult verifyScheduleRegion(const VPTOSchedDAG &dag,
                                          const VPTOScheduleResult &result,
                                          VPTOSchedulingBudget &budget,
                                          VPTOScheduleFailure &failure) {
  const VPTOSchedRegion &region = dag.getRegion();
  if (!budget.consume(region.operations.size())) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  if (!regionIsStillContiguous(region)) {
    setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
               "region ownership or boundaries changed before apply");
    return mlir::failure();
  }
  bool incompleteSchedule = result.entries.size() != dag.getUnits().size();
  if (incompleteSchedule) {
    setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
               "schedule is not a complete region permutation");
    return mlir::failure();
  }
  return success();
}

static LogicalResult buildSchedulePositions(
    const VPTOSchedDAG &dag, const VPTOScheduleResult &result,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure,
    DenseMap<const VPTOSUnit *, unsigned> &positions) {
  const VPTOSchedRegion &region = dag.getRegion();
  for (auto [index, entry] : llvm::enumerate(result.entries)) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    bool belongsToRegion =
        entry.unit && entry.unit->getOperation()->getBlock() == region.block;
    bool inserted =
        entry.unit && positions.try_emplace(entry.unit, index).second;
    if (!belongsToRegion || !inserted) {
      setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                 "schedule contains a null, foreign, or duplicate node");
      return mlir::failure();
    }
  }
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    if (!positions.count(unit.get())) {
      setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                 "schedule omits a region node");
      return mlir::failure();
    }
  }
  return success();
}

static LogicalResult
verifyScheduleEdges(const VPTOSchedDAG &dag,
                    const DenseMap<const VPTOSUnit *, unsigned> &positions,
                    VPTOSchedulingBudget &budget,
                    VPTOScheduleFailure &failure) {
  for (const std::unique_ptr<VPTOSchedEdge> &edge : dag.getEdges()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    unsigned predecessorPosition = positions.lookup(edge->getPredecessor());
    unsigned successorPosition = positions.lookup(edge->getSuccessor());
    bool violatesDependency =
        edge->isMust() && predecessorPosition >= successorPosition;
    if (violatesDependency) {
      setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                 "schedule violates a Must dependency");
      return mlir::failure();
    }
  }
  return success();
}

static LogicalResult
verifyScheduleSSA(const VPTOSchedDAG &dag, const VPTOScheduleResult &result,
                  const DenseMap<const VPTOSUnit *, unsigned> &positions,
                  VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  for (const VPTOScheduleEntry &entry : result.entries) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    for (Value operand : entry.unit->getOperation()->getOperands()) {
      if (!budget.consume()) {
        setWorkBudgetFailure(failure, budget);
        return mlir::failure();
      }
      Operation *definingOp = operand.getDefiningOp();
      VPTOSUnit *definingUnit = definingOp ? dag.lookup(definingOp) : nullptr;
      if (definingUnit &&
          positions.lookup(definingUnit) >= positions.lookup(entry.unit)) {
        setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                   "schedule violates SSA dominance");
        return mlir::failure();
      }
    }
  }
  return success();
}

LogicalResult mlir::pto::verifyVPTOScheduleResult(
    const VPTOSchedDAG &dag, const VPTOScheduleResult &result,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (failed(verifyScheduleRegion(dag, result, budget, failure))) {
    return mlir::failure();
  }
  DenseMap<const VPTOSUnit *, unsigned> positions;
  bool invalidSchedule =
      failed(buildSchedulePositions(dag, result, budget, failure, positions)) ||
      failed(verifyScheduleEdges(dag, positions, budget, failure)) ||
      failed(verifyScheduleSSA(dag, result, positions, budget, failure));
  if (invalidSchedule) {
    return mlir::failure();
  }
  return success();
}

static LogicalResult ensureReplayCandidate(VPTOSchedBoundary &boundary,
                                           VPTOSchedulingBudget &budget,
                                           VPTOScheduleFailure &failure) {
  if (!boundary.getAvailable().empty()) {
    return success();
  }
  return advanceReplayBoundary(boundary, budget, failure,
                               "replay has no candidate or pending event");
}

static LogicalResult validateReplayEntry(const VPTOScheduleEntry &entry,
                                         const VPTOSchedBoundary &boundary,
                                         VPTOScheduleFailure &failure) {
  bool isCandidate = boundary.isAvailable(entry.unit);
  bool hasDirection = entry.direction == boundary.getDirection();
  bool hasCycle = entry.issueCycle == boundary.getCurrentCycle();
  if (!isCandidate || !hasDirection || !hasCycle) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               "recorded node is not dependency-ready in its direction and "
               "issue cycle");
    return mlir::failure();
  }
  return success();
}

static LogicalResult commitReplayEntry(const VPTOScheduleEntry &entry,
                                       VPTOSchedBoundary &boundary,
                                       VPTOSchedulingBudget &budget,
                                       VPTOScheduleFailure &failure) {
  std::string detail;
  if (succeeded(
          boundary.commit(*entry.unit, entry.issueCycle, budget, detail))) {
    return success();
  }
  if (budget.hasExceeded()) {
    setWorkBudgetFailure(failure, budget);
  } else {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay, detail);
  }
  return mlir::failure();
}

static LogicalResult replayScheduleEntry(const VPTOScheduleEntry &entry,
                                         VPTOSchedBoundary &boundary,
                                         const VPTOSchedModel &model,
                                         VPTOSchedulingBudget &budget,
                                         VPTOScheduleFailure &failure) {
  if (!budget.consume()) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  bool replayFailed =
      failed(replayMandatoryDependencyIdle(entry, boundary, budget, failure)) ||
      failed(
          replayPressureDrivenIdle(entry, boundary, model, budget, failure)) ||
      failed(replayRecoveryDrivenIdle(entry, boundary, budget, failure)) ||
      failed(ensureReplayCandidate(boundary, budget, failure)) ||
      failed(validateReplayEntry(entry, boundary, failure)) ||
      failed(commitReplayEntry(entry, boundary, budget, failure));
  if (replayFailed) {
    return mlir::failure();
  }
  return success();
}

LogicalResult mlir::pto::replayVPTOScheduleResult(
    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
    const VPTOScheduleResult &result, VPTOSchedulingBudget &budget,
    VPTOScheduleFailure &failure) {
  VPTOSchedBoundary boundary(dag, model, VPTOSchedDirection::Top);
  for (const VPTOScheduleEntry &entry : result.entries) {
    if (failed(replayScheduleEntry(entry, boundary, model, budget, failure))) {
      return mlir::failure();
    }
  }
  bool scheduleComplete = boundary.isComplete();
  bool peakMatches =
      llvm::equal(boundary.getPressureTracker().getPeak(), result.peakPressure);
  if (!scheduleComplete || !peakMatches) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               "fresh replay does not reproduce the schedule summary");
    return mlir::failure();
  }
  return success();
}

LogicalResult mlir::pto::applyVPTOScheduleResult(
    const VPTOSchedDAG &dag, const VPTOScheduleResult &result,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (!budget.consume(result.entries.size())) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  const VPTOSchedRegion &region = dag.getRegion();
  for (const VPTOScheduleEntry &entry : result.entries) {
    if (region.followingBoundary) {
      entry.unit->getOperation()->moveBefore(region.followingBoundary);
    } else {
      entry.unit->getOperation()->moveBefore(region.block, region.block->end());
    }
  }
  return success();
}

StringRef
mlir::pto::stringifyVPTOScheduleFailureKind(VPTOScheduleFailureKind kind) {
  switch (kind) {
  case VPTOScheduleFailureKind::None:
    return "none";
  case VPTOScheduleFailureKind::Budget:
    return "budget";
  case VPTOScheduleFailureKind::InvalidModel:
    return "invalid-model";
  case VPTOScheduleFailureKind::Scheduling:
    return "scheduling";
  case VPTOScheduleFailureKind::SemanticVerification:
    return "semantic-verification";
  case VPTOScheduleFailureKind::ModelReplay:
    return "model-replay";
  case VPTOScheduleFailureKind::Apply:
    return "apply";
  }
  llvm_unreachable("unknown VPTO schedule failure kind");
}
