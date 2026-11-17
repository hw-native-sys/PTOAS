// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Incremental physical layout state shared by candidate cost evaluation.
// Cost counters and the evaluation entry points are in VMILayoutCostModel.h.

#ifndef PTO_TRANSFORMS_VMILAYOUTPHYSICALSTATE_H
#define PTO_TRANSFORMS_VMILAYOUTPHYSICALSTATE_H

#include "PTO/Transforms/VMILayoutPlanner.h"
#include <memory>
#include <string>

namespace mlir::pto {

struct VMILayoutScopeCost;

class VMILayoutPhysicalState {
public:
    VMILayoutPhysicalState();
    VMILayoutPhysicalState(const VMILayoutPhysicalState&);
    VMILayoutPhysicalState(VMILayoutPhysicalState&&) noexcept;
    VMILayoutPhysicalState& operator=(const VMILayoutPhysicalState&);
    VMILayoutPhysicalState& operator=(VMILayoutPhysicalState&&) noexcept;
    ~VMILayoutPhysicalState();

private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend FailureOr<VMILayoutPhysicalState> appendVMILayoutPhysicalRelation(
        const VMILayoutPhysicalState& state, const VMILayoutOpRelation& relation, const VMILayoutPlan& partialPlan);
    friend FailureOr<std::string> getVMILayoutContinuationKey(
        const VMILayoutPhysicalState& state, ArrayRef<Operation*> remainingOps);
    friend VMILayoutScopeCost getVMILayoutPhysicalCost(const VMILayoutPhysicalState& state);
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTPHYSICALSTATE_H
