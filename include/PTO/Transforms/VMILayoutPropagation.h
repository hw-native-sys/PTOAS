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

class VMILayoutPropagator {
public:
  explicit VMILayoutPropagator(Operation *scope);

  LogicalResult request(Value value, VMILayoutAttr layout);
  LogicalResult request(OpOperand &operand, VMILayoutAttr layout);
  void addEquivalentValues(Value lhs, Value rhs);

  LogicalResult run();
  LogicalResult apply(RewriterBase &rewriter);

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
                               VMILayoutAttr layout);
  LogicalResult propagateFact(Value value, VMILayoutAttr layout);
  LogicalResult propagateOperandFact(OpOperand &operand, VMILayoutAttr layout);
  LogicalResult propagateThrough(Operation *op, Value changedValue,
                                 VMILayoutAttr changedLayout,
                                 OpOperand *changedOperand = nullptr);
  LogicalResult verifyMaterializationPlan() const;

  LogicalResult materializePrimary(Value value,
                                   const VMIValueLayoutAssignment &assignment,
                                   RewriterBase &rewriter,
                                   DenseMap<Value, Value> &assignedValues);
  FailureOr<Value> materializeAt(Value source, VMILayoutAttr layout,
                                 RewriterBase &rewriter, Location loc);
  LogicalResult materializeUseConflict(Value assignedValue,
                                       VMILayoutConflict conflict,
                                       RewriterBase &rewriter);

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
