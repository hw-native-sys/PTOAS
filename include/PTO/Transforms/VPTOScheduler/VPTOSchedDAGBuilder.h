// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAGBuilder.h - VPTO scheduling DAG builder -----*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAGBUILDER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAGBUILDER_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"
#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir::pto {

class VPTOSchedDAGBuilder {
public:
  explicit VPTOSchedDAGBuilder(const VPTOSchedModel *model = nullptr)
      : model(model) {}
  VPTOSchedDAGBuilder(const VPTOSchedModel *model,
                      const VPTOSchedulerLimits &limits,
                      VPTOSchedulingBudget &budget)
      : model(model), limits(&limits), budget(&budget) {}

  FailureOr<std::unique_ptr<VPTOSchedDAG>>
  build(const VPTOSchedRegion &region) const;
  FailureOr<std::unique_ptr<VPTOSchedDAG>>
  build(const VPTOSchedRegion &region,
        VPTOScheduleFailure &failure) const;

private:
  LogicalResult buildSSAEdges(VPTOSchedDAG &dag,
                              VPTOScheduleFailure &failure) const;
  LogicalResult buildMemoryEdges(VPTOSchedDAG &dag,
                                 VPTOScheduleFailure &failure) const;
  LogicalResult buildImplicitAndSyncEdges(
      VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const;
  LogicalResult buildModelFallbackEdges(
      VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const;
  LogicalResult addEdge(VPTOSchedDAG &dag, VPTOSUnit &predecessor,
                        VPTOSUnit &successor, VPTOSchedEdgeKind kind,
                        VPTOSchedEdgeStrength strength, unsigned latency,
                        Twine reason, VPTOScheduleFailure &failure) const;
  LogicalResult consumeWork(VPTOScheduleFailure &failure,
                            uint64_t amount = 1) const;

  const VPTOSchedModel *model;
  const VPTOSchedulerLimits *limits = nullptr;
  VPTOSchedulingBudget *budget = nullptr;
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAGBUILDER_H
