// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedResourceTracker.h - VPTO resource tracking -----*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDRESOURCETRACKER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDRESOURCETRACKER_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedBoundary.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::pto {

struct VPTOResourceEvaluation {
  bool legal = true;
  unsigned earliestCycle = 0;
  unsigned issueSlot = 0;
  unsigned stallCycles = 0;
  std::string reason;
};

class VPTOResourceTracker {
public:
  explicit VPTOResourceTracker(const VPTOSchedModel &model) : model(model) {}

  VPTOResourceEvaluation evaluate(const VPTOSUnit &unit,
                                  unsigned requestedCycle) const;
  LogicalResult commit(const VPTOSUnit &unit, unsigned cycle);

  unsigned getIssueOccupancy(unsigned cycle) const;
  unsigned getResourceOccupancy(VPTOSchedResourceID resource,
                                unsigned cycle) const;

private:
  const VPTOSchedResource *findResource(VPTOSchedResourceID id) const;
  bool canReserve(const VPTOSchedClass &schedClass, unsigned cycle,
                  std::string &reason) const;
  void reserve(const VPTOSchedClass &schedClass, unsigned cycle);

  const VPTOSchedModel &model;
  SmallVector<unsigned> issueOccupancy;
  DenseMap<VPTOSchedResourceID, SmallVector<unsigned>> resourceOccupancy;
};

struct VPTOHazardResult {
  bool legal = true;
  unsigned earliestCycle = 0;
  std::string reason;
};

/// Target hook for pair/spacing/issue restrictions which cannot be expressed
/// as ordinary resource reservations. Correctness dependencies still belong in
/// the DAG rather than in this recognizer.
class VPTOHazardRecognizer {
public:
  virtual ~VPTOHazardRecognizer() = default;
  virtual VPTOHazardResult check(const VPTOSUnit &unit,
                                 VPTOSchedDirection direction,
                                 unsigned cycle) const = 0;
  virtual void commit(const VPTOSUnit &unit, VPTOSchedDirection direction,
                      unsigned cycle) = 0;
};

class VPTONullHazardRecognizer final : public VPTOHazardRecognizer {
public:
  VPTOHazardResult check(const VPTOSUnit &unit, VPTOSchedDirection direction,
                         unsigned cycle) const override;
  void commit(const VPTOSUnit &unit, VPTOSchedDirection direction,
              unsigned cycle) override;
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDRESOURCETRACKER_H
