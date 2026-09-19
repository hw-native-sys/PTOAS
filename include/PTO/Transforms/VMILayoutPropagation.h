// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutPropagation.h - VMI layout request propagation -*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTPROPAGATION_H
#define PTO_TRANSFORMS_VMILAYOUTPROPAGATION_H

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir::pto {

struct VMILayoutConflict {
  OpOperand *operand = nullptr;
  VMILayoutAttr layout;
};

struct VMIValueLayoutAssignment {
  VMILayoutAttr layout;
  SmallVector<VMILayoutConflict, mlir::pto::kValue2> conflicts;
};

// True when \p op is a VMI op whose operands and results must carry one and
// the same layout (elementwise and lane-local ops).  It is the single source of
// truth for the "same layout" transfer used by the propagator, and is also the
// relation layout consumers must use when they need to walk a
// layout-transparent chain (for example the direction-spine recognition in
// VMILayoutAssignment).
bool isVMISameLayoutOp(Operation *op);

// True when \p op is an edge of a VMI layout equivalence class: a walk that
// asks "does the value still carry its layout here" may cross it.  This is the
// elementwise/lane-local family (isVMISameLayoutOp) plus an *equal-width*
// bitcast, whose only relation is the identical layout for every layout
// (getBitcastLayoutFactsForLayout).  A width-changing bitcast is deliberately
// NOT a class edge: its only legal relation is the contiguous row of
// kWidthChangingBitcastLayoutPatterns and its element width changes, so a walk
// must stop there and let the reconciler materialize an ensure_layout on the
// boundary.  This is the shared definition of "class edge" for every consumer
// that walks a chain; the solver splits the two kinds the same way in
// VMILayoutAssignment::addBasicConstraint.
bool isVMIClassTransparentOp(Operation *op);

class VMILayoutPropagator {
public:
  explicit VMILayoutPropagator(Operation *scope);

  LogicalResult request(Value value, VMILayoutAttr layout);
  LogicalResult request(OpOperand &operand, VMILayoutAttr layout);
  void addEquivalentValues(Value lhs, Value rhs);

  LogicalResult run();
  LogicalResult apply(RewriterBase &rewriter);

  // Cast ops for which the cast transfer must consult the *spine-scoped* cast
  // layout table instead of the generic one (see
  // VMILayoutSupport::getSpineScopedCastLayoutFact).  The set is null by
  // default, so every propagation that does not opt in - including all
  // propagation for a chain the direction-spine peephole did not match - can
  // only ever see the generic tables.
  void
  setSpineScopedCastOps(const llvm::SmallPtrSetImpl<Operation *> *scopedOps) {
    spineScopedCastOps = scopedOps;
  }
  bool isSpineScopedCast(Operation *op) const {
    return spineScopedCastOps != nullptr && spineScopedCastOps->contains(op);
  }

  bool canUseOperandLayout(OpOperand &operand, VMILayoutAttr layout) const;
  VMILayoutAttr getRequestedLayout(Value value) const;
  VMILayoutAttr getRequestedOrCurrentLayout(Value value) const;
  const VMIValueLayoutAssignment *lookup(Value value) const;

private:
  using LayoutFact = std::pair<Value, VMILayoutAttr>;
  using OperandLayoutFact = std::pair<OpOperand *, VMILayoutAttr>;

  bool isLayoutValue(Value value) const;
  VMILayoutAttr getCurrentLayout(Value value) const;
  Type getTypeWithLayout(Value value, VMILayoutAttr layout) const;
  bool isTypeRewriteable(Value value) const;
  VMILayoutAttr getOperandLayout(OpOperand &operand) const;
  bool canProduceValueLayout(Value value, VMILayoutAttr layout) const;
  bool canMaterializeLayout(Value value, VMILayoutAttr sourceLayout,
                            VMILayoutAttr resultLayout) const;

  void enqueue(Value value, VMILayoutAttr layout);
  LogicalResult addUseConflict(OpOperand &operand,
                               VMIValueLayoutAssignment &assignment,
                               VMILayoutAttr layout) const;
  LogicalResult propagateFact(Value value, VMILayoutAttr layout);
  LogicalResult propagateOperandFact(OpOperand &operand, VMILayoutAttr layout);
  LogicalResult propagateThrough(Operation *op, Value changedValue,
                                 VMILayoutAttr changedLayout,
                                 OpOperand *changedOperand = nullptr);
  LogicalResult verifyMaterializationPlan() const;

  LogicalResult materializePrimary(Value value,
                                   const VMIValueLayoutAssignment &assignment,
                                   RewriterBase &rewriter,
                                   DenseMap<Value, Value> &assignedValues) const;
  FailureOr<Value> materializeAt(Value source, VMILayoutAttr layout,
                                 RewriterBase &rewriter, Location loc) const;
  LogicalResult materializeUseConflict(Value assignedValue,
                                       VMILayoutConflict conflict,
                                       RewriterBase &rewriter) const;

  // Borrowed; owned by the caller (the layout-assignment solver).
  const llvm::SmallPtrSetImpl<Operation *> *spineScopedCastOps = nullptr;
  Operation *scope = nullptr;
  MLIRContext *ctx = nullptr;
  DenseMap<Value, VMIValueLayoutAssignment> assignments;
  SmallVector<Value, mlir::pto::kValue16> orderedValues;
  SmallVector<LayoutFact, mlir::pto::kValue16> worklist;
  SmallVector<LayoutFact, mlir::pto::kValue16> seenFacts;
  SmallVector<OperandLayoutFact, mlir::pto::kValue16> seenOperandFacts;
  DenseMap<Value, SmallVector<Value, mlir::pto::kValue2>> equivalentValues;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTPROPAGATION_H
