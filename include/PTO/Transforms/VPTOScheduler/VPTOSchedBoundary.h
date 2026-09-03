// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedBoundary.h - VPTO scheduling boundary ----------*- C++ -*-===//
//
// A boundary owns all direction-local scheduling state: dependency counts,
// ready queues, logical cycles, register pressure, resource reservations, and
// hazard recognition.  Boundaries read a shared DAG but never mutate it or
// share tracker state, so scheduling and replay can each use a fresh boundary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDBOUNDARY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDBOUNDARY_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace mlir::pto {

enum class VPTOSchedDirection { Top, Bottom };

// These trackers include this header back (to reach VPTOSchedBoundary), so they
// must stay forward-declared here to break the include cycle.
class VPTOResourceTracker;
class VPTORegPressureTracker;
struct VPTORegPressureEvaluation;
struct VPTOPressureEvaluationCache;
class VPTOHazardRecognizer;
class VPTOSchedulingBudget;

struct VPTOPendingUnit {
  VPTOSUnit *unit = nullptr;
  unsigned readyCycle = 0;
};

class VPTOSchedBoundary {
public:
  VPTOSchedBoundary(const VPTOSchedDAG &dag, const VPTOSchedModel &model,
                    VPTOSchedDirection direction);
  VPTOSchedBoundary(const VPTOSchedDAG &dag, const VPTOSchedModel &model,
                    VPTOSchedDirection direction,
                    std::unique_ptr<VPTOHazardRecognizer> hazardRecognizer);
  ~VPTOSchedBoundary();

  VPTOSchedDirection getDirection() const { return direction; }
  unsigned getCurrentCycle() const { return currentCycle; }
  ArrayRef<VPTOSUnit *> getAvailable() const { return available; }
  size_t getPendingCount() const { return pendingCount; }
  /// Return the earliest pending event. The cycle buckets remain private.
  const VPTOPendingUnit *getNextPending() const;
  bool empty() const { return available.empty() && pendingCount == 0; }
  bool isComplete() const { return scheduledCount == states.size(); }
  bool isAvailable(const VPTOSUnit *unit) const;
  bool isScheduled(const VPTOSUnit *unit) const;
  unsigned getRemainingDependencyCount(const VPTOSUnit &unit) const;

  VPTOResourceTracker &getResourceTracker();
  const VPTOResourceTracker &getResourceTracker() const;
  VPTORegPressureTracker &getPressureTracker();
  const VPTORegPressureTracker &getPressureTracker() const;
  VPTORegPressureEvaluation evaluatePressure(const VPTOSUnit &unit) const;
  VPTOHazardRecognizer &getHazardRecognizer();
  const VPTOHazardRecognizer &getHazardRecognizer() const;

  /// Move a dependency-ready unit to a future cycle.  Resource and hazard
  /// trackers use this without mutating DAG readiness.
  LogicalResult defer(VPTOSUnit &unit, unsigned readyCycle,
                      VPTOSchedulingBudget &budget);

  /// Advance to the earliest pending cycle and release all units ready there.
  /// Returns false when no pending unit exists and failure when the shared
  /// work-unit budget cannot pay for inspecting the pending queue.
  FailureOr<bool> advanceToNextPendingCycle(VPTOSchedulingBudget &budget);

  /// Commit an available unit at the current cycle, update pressure, and
  /// release or defer newly dependency-ready neighbors using edge latency.
  /// Dependency traversal and queue work are paid before tracker or queue
  /// state changes, so budget failure cannot leave a partial commit.
  LogicalResult commit(VPTOSUnit &unit, unsigned issueCycle,
                       VPTOSchedulingBudget &budget, std::string &detail);

private:
  enum class UnitState { Unavailable, Pending, Available, Scheduled };

  void insertAvailable(VPTOSUnit *unit);
  void eraseAvailable(VPTOSUnit *unit);
  void insertPending(VPTOSUnit *unit, unsigned readyCycle);
  void releasePending();

  VPTOSchedDirection direction;
  unsigned currentCycle = 0;
  size_t scheduledCount = 0;
  SmallVector<VPTOSUnit *> available;
  SmallVector<size_t> availablePositions;
  std::map<unsigned, SmallVector<VPTOSUnit *>> pendingByCycle;
  size_t pendingCount = 0;
  mutable VPTOPendingUnit nextPending;
  SmallVector<unsigned> remainingDependencies;
  SmallVector<unsigned> readyCycles;
  SmallVector<UnitState> states;
  std::unique_ptr<VPTOResourceTracker> resourceTracker;
  std::unique_ptr<VPTORegPressureTracker> pressureTracker;
  std::unique_ptr<VPTOPressureEvaluationCache> pressureEvaluationCache;
  std::unique_ptr<VPTOHazardRecognizer> hazardRecognizer;
};

StringRef stringifyVPTOSchedDirection(VPTOSchedDirection direction);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDBOUNDARY_H
