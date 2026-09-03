// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedStrategy.h - VPTO scheduling strategy ----------*- C++ -*-===//
//
// Candidate construction, policy selection, and boundary mutation are separate
// contracts.  Strategies inspect immutable candidate snapshots and return a
// decision; only the scheduler commits that decision to its boundary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDSTRATEGY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDSTRATEGY_H

#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto {

struct VPTOScheduleContext {
  const VPTOSchedModel &model;
  const VPTOSchedDAG &dag;
  VPTOSchedDirection direction;
  unsigned issueCycle;
  ArrayRef<int64_t> currentPressure;
  /// Pressure set whose currently live value the active closure cone is
  /// trying to release. Empty outside high-pressure closure mode.
  std::optional<unsigned> closurePressureSet;
};

struct VPTOSchedCandidate {
  VPTOSUnit *unit = nullptr;
  VPTOSchedDirection direction = VPTOSchedDirection::Top;
  unsigned issueCycle = 0;
  unsigned criticalPath = 0;
  unsigned originalIndex = 0;
  VPTORegPressureEvaluation pressure;
  SmallVector<int64_t, 2> lookaheadPeak;
  SmallVector<int64_t, 2> lookaheadEnd;
  unsigned lookaheadSteps = 0;
  bool opensPressureFrontier = false;
  bool advancesPressureClosure = false;
};

struct VPTOSchedDecision {
  VPTOSUnit *unit = nullptr;
  VPTOSchedDirection direction = VPTOSchedDirection::Top;
  unsigned issueCycle = 0;
  std::string reason;
};

class VPTOSchedStrategy {
public:
  virtual ~VPTOSchedStrategy() = default;

  virtual FailureOr<VPTOSchedDecision>
  pickCandidate(const VPTOScheduleContext &context,
                ArrayRef<VPTOSchedCandidate> candidates,
                std::string &detail) const = 0;
};

class VPTODefaultSchedStrategy final : public VPTOSchedStrategy {
public:
  FailureOr<VPTOSchedDecision>
  pickCandidate(const VPTOScheduleContext &context,
                ArrayRef<VPTOSchedCandidate> candidates,
                std::string &detail) const override;
};

const VPTOSchedStrategy &getDefaultVPTOSchedStrategy();

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDSTRATEGY_H
