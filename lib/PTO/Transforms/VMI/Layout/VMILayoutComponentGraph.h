// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Component discovery for the immutable layout relation graph. Both SSA values
// and structural control-flow endpoints participate in each component.

#ifndef PTO_TRANSFORMS_VMI_VMILAYOUTCOMPONENTGRAPH_H
#define PTO_TRANSFORMS_VMI_VMILAYOUTCOMPONENTGRAPH_H

#include "PTO/Support/CodeConstants.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::pto {

struct VMILayoutSolverOp;
SmallVector<unsigned, kValue8> getVMILayoutSSAComponents(ArrayRef<VMILayoutSolverOp> ops);

class VMILayoutComponentGraph {
public:
    explicit VMILayoutComponentGraph(Operation* scope);
    void collect(Value seed, SmallVectorImpl<Operation*>& ops, DenseSet<Value>& globallySeenValues) const;

private:
    DenseMap<Operation*, SmallVector<Value, mlir::pto::kValue4>> operationValues;
    DenseMap<Value, SmallVector<Operation*, mlir::pto::kValue4>> valueOperations;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMI_VMILAYOUTCOMPONENTGRAPH_H
