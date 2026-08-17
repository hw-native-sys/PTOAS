// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedBoundary.h - VPTO scheduling boundary ----------*- C++ -*-===//
//
// A boundary owns all direction-local scheduling state: ready queues, cycle,
// resource reservations, register pressure, and hazard recognition. Top and
// bottom boundaries share a DAG but never share mutable tracker state.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDBOUNDARY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDBOUNDARY_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir::pto {

enum class VPTOSchedDirection { Top, Bottom };

class VPTOSchedModel;
class VPTOResourceTracker;
class VPTORegPressureTracker;
class VPTOHazardRecognizer;

struct VPTOPendingUnit {
  VPTOSUnit *unit = nullptr;
  unsigned readyCycle = 0;
};

class VPTOSchedBoundary {
public:
  VPTOSchedBoundary(VPTOSchedDAG &dag, const VPTOSchedModel &model,
                    VPTOSchedDirection direction);
  VPTOSchedBoundary(VPTOSchedDAG &dag, const VPTOSchedModel &model,
                    VPTOSchedDirection direction,
                    std::unique_ptr<VPTOHazardRecognizer> hazardRecognizer);
  ~VPTOSchedBoundary();

  VPTOSchedDirection getDirection() const { return direction; }
  unsigned getCurrentCycle() const { return currentCycle; }
  ArrayRef<VPTOSUnit *> getAvailable() const { return available; }
  ArrayRef<VPTOPendingUnit> getPending() const { return pending; }
  bool empty() const { return available.empty() && pending.empty(); }
  bool isScheduled(VPTOSUnit *unit) const { return scheduled.contains(unit); }

  VPTOResourceTracker &getResourceTracker();
  const VPTOResourceTracker &getResourceTracker() const;
  VPTORegPressureTracker &getPressureTracker();
  const VPTORegPressureTracker &getPressureTracker() const;
  VPTOHazardRecognizer &getHazardRecognizer();
  const VPTOHazardRecognizer &getHazardRecognizer() const;

  /// Move a dependency-ready unit to a future cycle.  Resource and hazard
  /// trackers use this without mutating DAG readiness.
  LogicalResult defer(VPTOSUnit &unit, unsigned readyCycle);

  /// Advance to the earliest pending cycle and release all units ready there.
  bool advanceToNextPendingCycle();

  /// Commit an available unit and release newly dependency-ready neighbors.
  LogicalResult commit(VPTOSUnit &unit);

private:
  void insertAvailable(VPTOSUnit *unit);
  void releasePending();

  VPTOSchedDirection direction;
  unsigned currentCycle = 0;
  SmallVector<VPTOSUnit *> available;
  SmallVector<VPTOPendingUnit> pending;
  DenseSet<VPTOSUnit *> scheduled;
  std::unique_ptr<VPTOResourceTracker> resourceTracker;
  std::unique_ptr<VPTORegPressureTracker> pressureTracker;
  std::unique_ptr<VPTOHazardRecognizer> hazardRecognizer;
};

StringRef stringifyVPTOSchedDirection(VPTOSchedDirection direction);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDBOUNDARY_H
