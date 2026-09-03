// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTORegPressureTracker.h - VPTO pressure tracking --------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOREGPRESSURETRACKER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOREGPRESSURETRACKER_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedBoundary.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::pto {

struct VPTORegPressureEvaluation {
  SmallVector<int64_t, 2> delta;
  /// Direction-local pressure removed by operand last uses (top-down) or
  /// result definitions (bottom-up).
  SmallVector<int64_t, 2> released;
  /// Direction-local pressure introduced by result definitions (top-down) or
  /// operand liveness (bottom-up).
  SmallVector<int64_t, 2> introduced;
  SmallVector<int64_t, 2> projected;
  SmallVector<int64_t, 2> projectedExcess;
};

class VPTORegPressureTracker {
public:
  VPTORegPressureTracker(const VPTOSchedModel &model, const VPTOSchedDAG &dag,
                         VPTOSchedDirection direction);

  VPTORegPressureEvaluation evaluate(const VPTOSUnit &unit) const;
  void refreshSummary(VPTORegPressureEvaluation &evaluation) const;
  LogicalResult commit(const VPTOSUnit &unit);

  VPTOSchedDirection getDirection() const { return direction; }
  ArrayRef<int64_t> getCurrent() const { return current; }
  ArrayRef<int64_t> getPeak() const { return peak; }
  const DenseSet<Value> &getLiveValues() const { return liveValues; }
  Value getPressureRepresentative(Value value) const;
  bool isLive(Value value) const;

private:
  bool isLiveOut(Value value) const;
  bool resultNeedsLiveness(Value value) const;
  void addValuePressure(Value value, int sign,
                        MutableArrayRef<int64_t> values) const;
  void initializeTop();
  void initializeBottom();
  VPTORegPressureEvaluation evaluateTop(const VPTOSUnit &unit) const;
  VPTORegPressureEvaluation evaluateBottom(const VPTOSUnit &unit) const;

  const VPTOSchedModel &model;
  const VPTOSchedDAG &dag;
  VPTOSchedDirection direction;
  DenseMap<VPTOPressureSetID, unsigned> pressureSetIndex;
  DenseMap<Value, unsigned> remainingUses;
  DenseSet<Value> liveOutValues;
  DenseSet<Value> liveValues;
  SmallVector<int64_t> current;
  SmallVector<int64_t> peak;
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOREGPRESSURETRACKER_H
