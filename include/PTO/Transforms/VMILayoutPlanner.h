// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutPlanner.h - VMI layout planning model --------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTPLANNER_H
#define PTO_TRANSFORMS_VMILAYOUTPLANNER_H

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::pto {

class VMILayoutPropagator;

enum class VMILayoutPortKind {
  Operand,
  Result,
};

struct VMILayoutPortAssignment {
  VMILayoutPortKind kind = VMILayoutPortKind::Operand;
  unsigned index = 0;
  VMILayoutAttr layout;
};

struct VMILayoutRelationEndpoint {
  Value value;
  OpOperand *use = nullptr;
  VMILayoutAttr layout;
};

inline Value getVMILayoutEndpointValue(
    const VMILayoutRelationEndpoint &endpoint) {
  return endpoint.use ? endpoint.use->get() : endpoint.value;
}

// A hard equality between two value layouts.  Relations and analyses may
// contribute these independently of the search algorithm.
struct VMILayoutEqualityConstraint {
  Value source;
  Value destination;
  bool abiBoundary = false;
};

struct VMILayoutOpRelation {
  Operation *op = nullptr;
  SmallVector<VMILayoutPortAssignment, mlir::pto::kValue4> ports;
  bool directProducer = false;
  int64_t intrinsicRearrangementCost = 0;
  // Used only after physical rearrangement cost and materialization count tie.
  // A zero value means that the support model has no preference or that this
  // relation is its preferred choice.
  uint64_t preferencePenalty = 0;
  SmallVector<VMILayoutRelationEndpoint, mlir::pto::kValue4> endpoints;
  SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue4> equalities;
};

struct VMILayoutPlan {
  DenseMap<Value, VMILayoutAttr> valueLayouts;
  DenseMap<OpOperand *, VMILayoutAttr> useLayouts;
  DenseMap<Operation *, unsigned> selectedRelations;
};

// Algorithm-independent incremental layout constraints.  A solver keeps a
// copy of this state per candidate; the state itself has no knowledge of the
// search algorithm or of structural operations.
class VMILayoutRelationConstraintState {
public:
  VMILayoutRelationConstraintState() = default;
  explicit VMILayoutRelationConstraintState(
      ArrayRef<VMILayoutEqualityConstraint> equalities);
  LogicalResult accept(const VMILayoutOpRelation &relation,
                       const VMILayoutPlan &plan);
  LogicalResult materialize(VMILayoutPlan &plan) const;
  FailureOr<std::string> fingerprint() const;

private:
  Value find(Value value);
  LogicalResult unite(Value lhs, Value rhs);
  LogicalResult assign(Value value, VMILayoutAttr layout);

  DenseMap<Value, Value> parent;
  DenseMap<Value, VMILayoutAttr> assignedLayouts;
};

struct VMILayoutPlannerOptions {
  unsigned maxFrontierEntriesPerComponent = 128;
  unsigned maxTransitionsPerComponent = 4096;
};

struct VMILayoutPlannerResult {
  SmallVector<VMILayoutPlan, mlir::pto::kValue4> plans;
};

bool isVMISameLayoutOp(Operation *op);
bool isVMILayoutCastOp(Operation *op);

class VMILayoutRelationProvider {
public:
  FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>
  enumerateRelations(Operation *op,
                     ArrayRef<VMILayoutAttr> polymorphicLayouts = {}) const;
};

class VMILayoutStructuralEdgeProvider {
public:
  // Extracts hard layout equalities from MLIR control-flow interfaces.  This
  // analysis layer handles structural transport; the solver only consumes the
  // resulting equalities and never models structural operations physically.
  SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue8>
  enumerateEdges(Operation *op) const;

};

LogicalResult commitVMILayoutPlan(const VMILayoutPlan &plan,
                                  VMILayoutPropagator &propagator);

FailureOr<VMILayoutPlannerResult>
selectCostedVMILayoutPlans(Operation *scope,
                           const VMILayoutPlannerOptions &options = {});

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTPLANNER_H
