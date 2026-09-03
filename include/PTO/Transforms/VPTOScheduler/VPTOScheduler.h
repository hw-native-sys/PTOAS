// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOScheduler.h - VPTO list scheduler -------------------*- C++ -*-===//
//
// The scheduler returns an immutable permutation with scheduling direction,
// logical issue cycles, and decision reasons. It never mutates IR. Callers
// must run semantic verification and a fresh model replay before applying a
// result to its block-local scheduling region.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULER_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedStrategy.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto {

struct VPTOSchedulerLimits {
  uint64_t maxNodes = 4096;
  uint64_t maxEdges = 262144;
  uint64_t maxWorkUnits = 1ULL << 20;
};

class VPTOSchedulingBudget {
public:
  explicit VPTOSchedulingBudget(uint64_t limit) : limit(limit) {}

  bool consume(uint64_t amount = 1);
  uint64_t getUsed() const { return used; }
  uint64_t getLimit() const { return limit; }
  bool hasExceeded() const { return exceeded; }

private:
  uint64_t limit;
  uint64_t used = 0;
  bool exceeded = false;
};

struct VPTOScheduleDiagnostic {
  SmallVector<int64_t, 2> currentPressure;
  unsigned candidateCount = 0;
  std::optional<unsigned> closurePressureSet;
  SmallVector<unsigned, 2> closureBundleOriginalIndices;
  std::optional<unsigned> closureTargetOriginalIndex;
  unsigned closureGroupSize = 0;
  unsigned closureSteps = 0;
  int64_t closureSupportPressure = 0;
  int64_t closureEffectiveEnd = 0;
  int64_t closureNetRelief = 0;
  SmallVector<int64_t, 2> closurePeak;
  SmallVector<int64_t, 2> closureEnd;
  unsigned selectedCriticalPath = 0;
  bool selectedAdvancesClosure = false;
  SmallVector<int64_t, 2> selectedProjectedPressure;
  SmallVector<int64_t, 2> selectedReleasedPressure;
  std::optional<unsigned> safeAlternativeOriginalIndex;
  unsigned safeAlternativeCriticalPath = 0;
  bool safeAlternativeOpensPressureFrontier = false;
  SmallVector<int64_t, 2> safeAlternativeProjectedPressure;
  SmallVector<int64_t, 2> safeAlternativeReleasedPressure;
};

struct VPTOScheduleEntry {
  VPTOSUnit *unit = nullptr;
  VPTOSchedDirection direction = VPTOSchedDirection::Top;
  unsigned issueCycle = 0;
  std::string reason;
  bool pressureDrivenIdle = false;
  bool recoveryDrivenIdle = false;
};

struct VPTOScheduleResult {
  SmallVector<VPTOScheduleEntry> entries;
  SmallVector<int64_t> peakPressure;
  SmallVector<std::optional<VPTOScheduleDiagnostic>, 0> diagnostics;
};

enum class VPTOScheduleFailureKind {
  None,
  Budget,
  InvalidModel,
  Scheduling,
  SemanticVerification,
  ModelReplay,
  Apply,
};

struct VPTOScheduleFailure {
  VPTOScheduleFailureKind kind = VPTOScheduleFailureKind::None;
  std::string name;
  uint64_t actual = 0;
  uint64_t limit = 0;
  std::string detail;
};

class VPTOScheduler {
public:
  VPTOScheduler(
      const VPTOSchedModel &model, VPTOSchedDAG &dag,
      const VPTOSchedulerLimits &limits, VPTOSchedulingBudget &budget,
      bool collectDiagnostics = false,
      const VPTOSchedStrategy &strategy = getDefaultVPTOSchedStrategy())
      : model(model), dag(dag), limits(limits), budget(budget),
        collectDiagnostics(collectDiagnostics), strategy(strategy) {}

  FailureOr<VPTOScheduleResult> schedule(VPTOScheduleFailure &failure) const;

private:
  const VPTOSchedModel &model;
  VPTOSchedDAG &dag;
  const VPTOSchedulerLimits &limits;
  VPTOSchedulingBudget &budget;
  bool collectDiagnostics;
  const VPTOSchedStrategy &strategy;
};

LogicalResult verifyVPTOScheduleResult(const VPTOSchedDAG &dag,
                                       const VPTOScheduleResult &result,
                                       VPTOSchedulingBudget &budget,
                                       VPTOScheduleFailure &failure);

LogicalResult replayVPTOScheduleResult(const VPTOSchedModel &model,
                                       const VPTOSchedDAG &dag,
                                       const VPTOScheduleResult &result,
                                       VPTOSchedulingBudget &budget,
                                       VPTOScheduleFailure &failure);

LogicalResult applyVPTOScheduleResult(const VPTOSchedDAG &dag,
                                      const VPTOScheduleResult &result,
                                      VPTOSchedulingBudget &budget,
                                      VPTOScheduleFailure &failure);

StringRef stringifyVPTOScheduleFailureKind(VPTOScheduleFailureKind kind);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULER_H
