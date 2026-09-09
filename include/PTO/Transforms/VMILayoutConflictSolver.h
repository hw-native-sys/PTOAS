// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutConflictSolver.h - VMI layout cost solver -----*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTCONFLICTSOLVER_H
#define PTO_TRANSFORMS_VMILAYOUTCONFLICTSOLVER_H

#include "PTO/Transforms/VMILayoutCostModel.h"

#include "llvm/ADT/FunctionExtras.h"

namespace mlir::pto {

struct VMILayoutSolverOp {
  Operation *op = nullptr;
  SmallVector<VMILayoutOpRelation, mlir::pto::kValue4> relations;
};

// Optional solver-level sharing of relation decisions.  A group relation is
// mapped to one relation per member operation; it does not equate SSA value
// layouts or physical values.
struct VMILayoutDecisionGroup {
  SmallVector<unsigned, mlir::pto::kValue4> opIndices;
  SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue4>
      memberRelationIndices;
};

struct VMILayoutConflictSolverOptions {
  unsigned maxFrontierEntries = 128;
  unsigned maxTransitions = 4096;
  ArrayRef<VMILayoutEqualityConstraint> equalityConstraints;
  ArrayRef<VMILayoutFixedAssignment> fixedAssignments;
  ArrayRef<VMILayoutDecisionGroup> decisionGroups;
};

FailureOr<VMILayoutPlan>
solveVMILayoutConflictComponent(ArrayRef<VMILayoutSolverOp> ops,
                                const VMILayoutConflictSolverOptions &options);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTCONFLICTSOLVER_H
