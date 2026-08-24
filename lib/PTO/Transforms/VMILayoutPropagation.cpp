// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutPropagation.cpp - VMI layout request propagation ----------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutPropagation.h"

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutSupport.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

struct VMILayoutFact {
  Value value;
  OpOperand *operand = nullptr;
  VMILayoutAttr layout;
};

struct VMILayoutRelation {
  SmallVector<VMILayoutFact, mlir::pto::kValue4> facts;
};

static VMILayoutFact valueFact(Value value, VMILayoutAttr layout) {
  return VMILayoutFact{value, /*operand=*/nullptr, layout};
}

static VMILayoutFact operandFact(OpOperand &operand, VMILayoutAttr layout) {
  return VMILayoutFact{/*value=*/{}, &operand, layout};
}

static VMILayoutRelation makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4> facts) {
  VMILayoutRelation relation;
  relation.facts = std::move(facts);
  return relation;
}

static SmallVector<VMILayoutRelation, mlir::pto::kValue4>
makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4> facts) {
  SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
  relations.push_back(makeRelation(std::move(facts)));
  return relations;
}

static bool hasAmbiguousTransferTargets(ArrayRef<VMILayoutFact> facts) {
  auto sameTarget = [](const VMILayoutFact &lhs, const VMILayoutFact &rhs) {
    if (lhs.operand || rhs.operand) {
      return lhs.operand && lhs.operand == rhs.operand;
    }
    return lhs.value && lhs.value == rhs.value;
  };

  for (auto [index, fact] : llvm::enumerate(facts)) {
    for (const VMILayoutFact &other : facts.drop_front(index + 1)) {
      if (sameTarget(fact, other) && fact.layout != other.layout) {
        return true;
      }
    }
  }
  return false;
}

static bool relationContainsOperandLayout(const VMILayoutRelation &relation,
                                          OpOperand &operand,
                                          VMILayoutAttr layout) {
  for (const VMILayoutFact &fact : relation.facts) {
    if (fact.operand == &operand) {
      return fact.layout == layout;
    }
  }
  return false;
}

static bool relationContainsValueLayout(const VMILayoutRelation &relation,
                                        Value value, VMILayoutAttr layout) {
  for (const VMILayoutFact &fact : relation.facts) {
    if (!fact.operand && fact.value == value) {
      return fact.layout == layout;
    }
  }
  return false;
}

static Type getValueTypeWithLayout(Value value, VMILayoutAttr layout) {
  if (auto type = dyn_cast<VMIVRegType>(value.getType())) {
    return VMIVRegType::get(type.getContext(), type.getElementCount(),
                            type.getElementType(), layout);
  }
  if (auto type = dyn_cast<VMIMaskType>(value.getType())) {
    return VMIMaskType::get(type.getContext(), type.getElementCount(),
                            type.getGranularity(), layout);
  }
  return {};
}

class VMILayoutTransfer {
public:
  virtual ~VMILayoutTransfer() = default;

  virtual FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const = 0;

  LogicalResult propagate(Operation *op, Value changedValue,
                          VMILayoutAttr changedLayout,
                          VMILayoutPropagator &propagator,
                          OpOperand *changedOperand) const {
    FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>> relations =
        query(op, changedValue, changedLayout, propagator, changedOperand);
    if (failed(relations) || relations->empty()) {
      return success();
    }
    if (relations->size() != 1) {
      return success();
    }
    const VMILayoutRelation &relation = relations->front();
    if (hasAmbiguousTransferTargets(relation.facts)) {
      return success();
    }
    for (const VMILayoutFact &fact : relation.facts) {
      if (fact.operand) {
        if (failed(propagator.request(*fact.operand, fact.layout))) {
          return failure();
        }
        continue;
      }
      if (failed(propagator.request(fact.value, fact.layout))) {
        return failure();
      }
    }
    return success();
  }
};

class VMILayoutMaterializationTransfer final {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Value source, VMILayoutAttr sourceLayout,
        VMILayoutAttr resultLayout) const {
    if (!source || !sourceLayout || !resultLayout) {
      return failure();
    }
    if (sourceLayout == resultLayout) {
      return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          valueFact(source, sourceLayout)});
    }

    Type sourceAssignedType = getValueTypeWithLayout(source, sourceLayout);
    Type resultAssignedType = getValueTypeWithLayout(source, resultLayout);
    if (!sourceAssignedType || !resultAssignedType) {
      return failure();
    }

    VMILayoutSupport supports;
    if (auto sourceVRegType = dyn_cast<VMIVRegType>(sourceAssignedType)) {
      auto resultVRegType = dyn_cast<VMIVRegType>(resultAssignedType);
      if (!resultVRegType ||
          failed(supports.getEnsureLayoutFact(sourceVRegType,
                                              resultVRegType))) {
        return failure();
      }
      return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          valueFact(source, sourceLayout)});
    }

    if (auto sourceMaskType = dyn_cast<VMIMaskType>(sourceAssignedType)) {
      auto resultMaskType = dyn_cast<VMIMaskType>(resultAssignedType);
      if (!resultMaskType ||
          failed(supports.getEnsureMaskLayoutFact(sourceMaskType,
                                                  resultMaskType))) {
        return failure();
      }
      return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          valueFact(source, sourceLayout)});
    }
    return failure();
  }
};

static bool isSameLayoutOp(Operation *op) {
  return isa<VMIAddFOp, VMIAddIOp, VMISubFOp, VMISubIOp, VMIMulFOp, VMIMulIOp,
             VMIVaddcOp, VMIVaddcsOp, VMIAddSOp, VMIMulSOp, VMIMaxSOp,
             VMIMinSOp, VMIShlSOp, VMIShrSOp, VMIVmullOp, VMIFmaOp, VMIDivFOp,
             VMIMinFOp, VMIMinIOp, VMIMaxFOp, VMIMaxIOp, VMINegFOp, VMIAbsFOp,
             VMIAbsIOp, VMISqrtOp, VMIExpOp, VMILnOp, VMIReluOp, VMIFPToSIOp,
             VMISIToFPOp, VMIAndIOp, VMIOrIOp, VMIXOrIOp, VMIShLIOp, VMIShRUIOp,
             VMIShRSIOp, VMINotOp, VMICmpFOp, VMICmpIOp, VMISelectOp,
             VMIMaskAndOp, VMIMaskOrOp, VMIMaskXOrOp, VMIMaskNotOp,
             VMIActivePrefixIndexOp, VMICompressOp, VMIExpandLoadOp>(op);
}

static bool isCastOp(Operation *op) {
  return isa<VMIExtFOp, VMIExtSIOp, VMIExtUIOp, VMITruncFOp, VMITruncIOp,
               VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op);
}

class VMISameLayoutTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    SmallVector<VMILayoutFact, mlir::pto::kValue4> facts;
    for (OpOperand &operand : op->getOpOperands()) {
      facts.push_back(operandFact(operand, changedLayout));
    }
    for (Value result : op->getResults()) {
      facts.push_back(valueFact(result, changedLayout));
    }
    return makeSingleRelation(std::move(facts));
  }
};

class VMICastTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }

    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    VMILayoutSupport supports;

    if (changedValue == op->getOperand(0)) {
      FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> facts =
          supports.getCastLayoutFactsForLayout(
              sourceType, resultType, VMICastLayoutPort::Source,
              changedLayout);
      if (failed(facts) || facts->empty()) {
        return failure();
      }
      SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
      for (const VMICastLayoutFact &fact : *facts) {
        relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
            operandFact(op->getOpOperand(0), fact.sourceLayout),
            valueFact(op->getResult(0), fact.resultLayout)}));
      }
      return relations;
    }

    if (changedValue == op->getResult(0)) {
      FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> facts =
          supports.getCastLayoutFactsForLayout(
              sourceType, resultType, VMICastLayoutPort::Result,
              changedLayout);
      if (failed(facts) || facts->empty()) {
        return failure();
      }
      SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
      for (const VMICastLayoutFact &fact : *facts) {
        relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
            operandFact(op->getOpOperand(0), fact.sourceLayout),
            valueFact(op->getResult(0), fact.resultLayout)}));
      }
      return relations;
    }
    return failure();
  }
};

class VMIVexpdifTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, 4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto vexpdif = dyn_cast<VMIVexpdifOp>(op);
    if (!vexpdif) {
      return failure();
    }

    auto sourceType = dyn_cast<VMIVRegType>(vexpdif.getX().getType());
    auto resultType = dyn_cast<VMIVRegType>(vexpdif.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    auto makeFacts = [&](VMILayoutAttr sourceLayout,
                         VMILayoutAttr resultLayout) {
      return makeRelation(SmallVector<VMILayoutFact, 4>{
          operandFact(vexpdif.getXMutable(), sourceLayout),
          operandFact(vexpdif.getMaxMutable(), sourceLayout),
          operandFact(vexpdif.getMaskMutable(), sourceLayout),
          valueFact(vexpdif.getResult(), resultLayout)});
    };

    if (sourceType.getElementType().isF32()) {
      return makeSingleRelation(SmallVector<VMILayoutFact, 4>{
          operandFact(vexpdif.getXMutable(), changedLayout),
          operandFact(vexpdif.getMaxMutable(), changedLayout),
          operandFact(vexpdif.getMaskMutable(), changedLayout),
          valueFact(vexpdif.getResult(), changedLayout)});
    }

    VMICastLayoutPort port;
    if (changedValue == vexpdif.getResult()) {
      port = VMICastLayoutPort::Result;
    } else if (changedValue == vexpdif.getX() ||
               changedValue == vexpdif.getMax() ||
               changedValue == vexpdif.getMask() ||
               changedOperand == &vexpdif.getXMutable() ||
               changedOperand == &vexpdif.getMaxMutable() ||
               changedOperand == &vexpdif.getMaskMutable()) {
      port = VMICastLayoutPort::Source;
    } else {
      return failure();
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMICastLayoutFact, 4>> facts =
        supports.getCastLayoutFactsForLayout(sourceType, resultType, port,
                                             changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }

    SmallVector<VMILayoutRelation, 4> relations;
    for (const VMICastLayoutFact &fact : *facts) {
      relations.push_back(makeFacts(fact.sourceLayout, fact.resultLayout));
    }
    return relations;
  }
};

class VMIBitcastTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto bitcast = dyn_cast<VMIBitcastOp>(op);
    if (!bitcast) {
      return failure();
    }

    auto sourceType = dyn_cast<VMIVRegType>(bitcast.getSource().getType());
    auto resultType = dyn_cast<VMIVRegType>(bitcast.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    VMICastLayoutPort port;
    if (changedOperand == &bitcast.getSourceMutable() ||
        changedValue == bitcast.getSource()) {
      port = VMICastLayoutPort::Source;
    } else if (changedValue == bitcast.getResult()) {
      port = VMICastLayoutPort::Result;
    } else {
      return failure();
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>> facts =
        supports.getBitcastLayoutFactsForLayout(sourceType, resultType, port,
                                                changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }

    SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
    for (const VMIBitcastLayoutFact &fact : *facts) {
      relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          operandFact(bitcast.getSourceMutable(), fact.sourceLayout),
          valueFact(bitcast.getResult(), fact.resultLayout)}));
    }
    return relations;
  }
};

class VMIMaskGranularityCastTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op);
    if (!ensure) {
      return failure();
    }

    auto sourceType = dyn_cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = dyn_cast<VMIMaskType>(ensure.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    VMILayoutSupport supports;
    if (changedValue == ensure.getSource()) {
      FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>> facts =
          supports.getMaskGranularityCastLayoutFactsForLayout(
              sourceType, resultType, VMICastLayoutPort::Source,
              changedLayout);
      if (failed(facts) || facts->empty()) {
        return failure();
      }
      SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
      for (const VMIMaskGranularityCastLayoutFact &fact : *facts) {
        relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
            operandFact(ensure.getSourceMutable(), fact.sourceLayout),
            valueFact(ensure.getResult(), fact.resultLayout)}));
      }
      return relations;
    }

    if (changedValue == ensure.getResult()) {
      FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>> facts =
          supports.getMaskGranularityCastLayoutFactsForLayout(
              sourceType, resultType, VMICastLayoutPort::Result,
              changedLayout);
      if (failed(facts) || facts->empty()) {
        return failure();
      }
      SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
      for (const VMIMaskGranularityCastLayoutFact &fact : *facts) {
        relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
            operandFact(ensure.getSourceMutable(), fact.sourceLayout),
            valueFact(ensure.getResult(), fact.resultLayout)}));
      }
      return relations;
    }
    return failure();
  }
};

class VMIFreeResultLayoutTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    if (isa<OpResult>(changedValue) && changedValue.getDefiningOp() == op) {
      return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          valueFact(changedValue, changedLayout)});
    }
    return failure();
  }
};

class VMIContiguousResultLayoutTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    if (!isa<OpResult>(changedValue) || changedValue.getDefiningOp() != op ||
        !changedLayout.isContiguous()) {
      return failure();
    }
    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        valueFact(changedValue, changedLayout)});
  }
};

class VMILoadTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto load = dyn_cast<VMILoadOp>(op);
    if (!load || changedValue != load.getResult()) {
      return failure();
    }
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return failure();
    }
    auto assignedType = VMIVRegType::get(
        resultType.getContext(), resultType.getElementCount(),
        resultType.getElementType(), changedLayout);
    VMILayoutSupport supports;
    if (failed(supports.getLoadLayoutFact(assignedType))) {
      return failure();
    }
    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        valueFact(load.getResult(), changedLayout)});
  }
};

class VMIDeinterleaveLoadTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto load = dyn_cast<VMIDeinterleaveLoadOp>(op);
    if (!load) {
      return failure();
    }

    VMIDeinterleaveLoadLayoutPort port;
    if (changedValue == load.getLow()) {
      port = VMIDeinterleaveLoadLayoutPort::Low;
    } else if (changedValue == load.getHigh()) {
      port = VMIDeinterleaveLoadLayoutPort::High;
    } else {
      return failure();
    }

    auto valueType = dyn_cast<VMIVRegType>(changedValue.getType());
    if (!valueType) {
      return failure();
    }
    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>> facts =
        supports.getDeinterleaveLoadLayoutFactsForLayout(
            valueType, port, changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }

    SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
    for (const VMIDeinterleaveLoadLayoutFact &fact : *facts) {
      relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          valueFact(load.getLow(), fact.lowLayout),
          valueFact(load.getHigh(), fact.highLayout)}));
    }
    return relations;
  }
};

class VMIGroupLoadTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto load = dyn_cast<VMIGroupLoadOp>(op);
    if (!load || changedValue != load.getResult()) {
      return failure();
    }
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return failure();
    }
    auto assignedType = VMIVRegType::get(
        resultType.getContext(), resultType.getElementCount(),
        resultType.getElementType(), changedLayout);
    VMILayoutSupport supports;
    if (failed(supports.getGroupLoadLayoutFact(
            assignedType, load.getRowStride(),
            load.getNumGroupsAttr().getInt()))) {
      return failure();
    }
    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        valueFact(load.getResult(), changedLayout)});
  }
};

class VMIGroupReduceTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
      return queryReduce(reduce, changedValue, changedLayout, changedOperand);
    }
    if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
      return queryReduce(reduce, changedValue, changedLayout, changedOperand);
    }
    if (auto reduce = dyn_cast<VMIGroupReduceMinFOp>(op)) {
      return queryReduce(reduce, changedValue, changedLayout, changedOperand);
    }
    if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
      return queryReduce(reduce, changedValue, changedLayout, changedOperand);
    }
    if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
      return queryReduce(reduce, changedValue, changedLayout, changedOperand);
    }
    if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
      return queryReduce(reduce, changedValue, changedLayout, changedOperand);
    }
    return failure();
  }

private:
  template <typename OpTy>
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  queryReduce(OpTy reduce, Value changedValue, VMILayoutAttr changedLayout,
              OpOperand *changedOperand) const {
    auto sourceType = dyn_cast<VMIVRegType>(reduce.getSource().getType());
    auto resultType = dyn_cast<VMIVRegType>(reduce.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    VMIGroupReduceLayoutPort port;
    if (changedValue == reduce.getSource()) {
      port = VMIGroupReduceLayoutPort::Source;
    } else if (changedValue == reduce.getMask()) {
      port = VMIGroupReduceLayoutPort::Mask;
    } else if (changedValue == reduce.getResult()) {
      port = VMIGroupReduceLayoutPort::Result;
    } else {
      return failure();
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>> facts =
        supports.getGroupReduceLayoutFactsForLayout(
            sourceType, reduce.getNumGroupsAttr().getInt(), port,
            changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }
    SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
    for (const VMIGroupReduceLayoutFact &fact : *facts) {
      relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          operandFact(reduce.getSourceMutable(), fact.sourceLayout),
          operandFact(reduce.getMaskMutable(), fact.maskLayout),
          valueFact(reduce.getResult(), fact.resultLayout)}));
    }
    return relations;
  }
};

class VMIGroupSlotLoadTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto load = dyn_cast<VMIGroupSlotLoadOp>(op);
    if (!load || changedValue != load.getResult()) {
      return failure();
    }
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return failure();
    }
    auto assignedType = VMIVRegType::get(
        resultType.getContext(), resultType.getElementCount(),
        resultType.getElementType(), changedLayout);
    VMILayoutSupport supports;
    if (failed(supports.getGroupSlotLoadLayoutFact(
            assignedType, load.getNumGroupsAttr().getInt()))) {
      return failure();
    }
    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        valueFact(load.getResult(), changedLayout)});
  }
};

class VMIGroupBroadcastLoadTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op);
    if (!load || changedValue != load.getResult()) {
      return failure();
    }
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return failure();
    }
    auto assignedType = VMIVRegType::get(
        resultType.getContext(), resultType.getElementCount(),
        resultType.getElementType(), changedLayout);
    VMILayoutSupport supports;
    if (failed(supports.getGroupBroadcastLoadLayoutFact(
            assignedType, load.getSourceGroupStride(),
            load.getNumGroupsAttr().getInt()))) {
      return failure();
    }
    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        valueFact(load.getResult(), changedLayout)});
  }
};

class VMIGroupBroadcastTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op);
    if (!broadcast) {
      return failure();
    }
    auto sourceType = dyn_cast<VMIVRegType>(broadcast.getSource().getType());
    auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    int64_t numGroups = broadcast.getNumGroupsAttr().getInt();
    VMIGroupBroadcastLayoutPort port;
    if (changedValue == broadcast.getSource()) {
      port = VMIGroupBroadcastLayoutPort::Source;
    } else if (changedValue == broadcast.getResult()) {
      port = VMIGroupBroadcastLayoutPort::Result;
    } else {
      return failure();
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4>> facts =
        supports.getGroupBroadcastLayoutFactsForLayout(
            sourceType, resultType, numGroups, port, changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }
    SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
    for (const VMIGroupBroadcastLayoutFact &fact : *facts) {
      relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          operandFact(broadcast.getSourceMutable(), fact.sourceLayout),
          valueFact(broadcast.getResult(), fact.resultLayout)}));
    }
    return relations;
  }
};

class VMIInterleaveTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    if (auto vintlv = dyn_cast<VMIVintlvOp>(op)) {
      return queryInterleave(vintlv, /*vintlv=*/true, changedValue,
                             changedLayout, changedOperand);
    }
    if (auto vdintlv = dyn_cast<VMIVdintlvOp>(op)) {
      return queryInterleave(vdintlv, /*vintlv=*/false, changedValue,
                             changedLayout, changedOperand);
    }
    return failure();
  }

private:
  template <typename OpTy>
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  queryInterleave(OpTy op, bool vintlv, Value changedValue,
                  VMILayoutAttr changedLayout,
                  OpOperand *changedOperand) const {
    auto lowType = dyn_cast<VMIVRegType>(op.getLow().getType());
    if (!lowType) {
      return failure();
    }

    VMIInterleaveLayoutPort port;
    if (changedOperand == &op.getLhsMutable() || changedValue == op.getLhs()) {
      port = VMIInterleaveLayoutPort::Lhs;
    } else if (changedOperand == &op.getRhsMutable() ||
               changedValue == op.getRhs()) {
      port = VMIInterleaveLayoutPort::Rhs;
    } else if (changedOperand == &op.getMaskMutable() ||
               changedValue == op.getMask()) {
      port = VMIInterleaveLayoutPort::Mask;
    } else if (changedValue == op.getLow()) {
      port = VMIInterleaveLayoutPort::Low;
    } else if (changedValue == op.getHigh()) {
      port = VMIInterleaveLayoutPort::High;
    } else {
      return failure();
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> facts =
        vintlv ? supports.getVintlvLayoutFactsForLayout(lowType, port,
                                                        changedLayout)
               : supports.getVdintlvLayoutFactsForLayout(lowType, port,
                                                         changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }

    SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
    for (const VMIInterleaveLayoutFact &fact : *facts) {
      relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          operandFact(op.getLhsMutable(), fact.lhsLayout),
          operandFact(op.getRhsMutable(), fact.rhsLayout),
          operandFact(op.getMaskMutable(), fact.maskLayout),
          valueFact(op.getLow(), fact.lowLayout),
          valueFact(op.getHigh(), fact.highLayout)}));
    }
    return relations;
  }
};

class VMIGatherTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto gather = dyn_cast<VMIGatherOp>(op);
    if (!gather) {
      return failure();
    }
    if (changedValue != gather.getIndices() && changedValue != gather.getMask() &&
        changedValue != gather.getPassthru() && changedValue != gather.getResult()) {
      return failure();
    }
    if (!changedLayout.isContiguous() || changedLayout.getLaneStride() != 1) {
      return failure();
    }
    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        operandFact(gather.getIndicesMutable(), changedLayout),
        operandFact(gather.getMaskMutable(), changedLayout),
        operandFact(gather.getPassthruMutable(), changedLayout),
        valueFact(gather.getResult(), changedLayout)});
  }
};

class VMIStoreTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto store = dyn_cast<VMIStoreOp>(op);
    if (!store || changedValue != store.getValue()) {
      return failure();
    }

    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    if (!valueType) {
      return failure();
    }

    auto assignedValueType =
        VMIVRegType::get(valueType.getContext(), valueType.getElementCount(),
                         valueType.getElementType(), changedLayout);
    VMILayoutSupport supports;
    VMILayoutAttr useLayout = changedLayout;
    if (failed(supports.getStoreLayoutFact(assignedValueType))) {
      useLayout = VMILayoutAttr::getContiguous(valueType.getContext());
    }

    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        operandFact(store.getValueMutable(), useLayout)});
  }
};

class VMIGroupStoreTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto store = dyn_cast<VMIGroupStoreOp>(op);
    if (!store || changedValue != store.getValue()) {
      return failure();
    }

    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    if (!valueType) {
      return failure();
    }

    VMILayoutSupport supports;
    FailureOr<SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>> facts =
        supports.getGroupStoreLayoutFactsForLayout(store, valueType,
                                                   changedLayout);
    if (failed(facts) || facts->empty()) {
      return failure();
    }
    SmallVector<VMILayoutRelation, mlir::pto::kValue4> relations;
    for (const VMIGroupStoreLayoutFact &fact : *facts) {
      relations.push_back(makeRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
          operandFact(store.getValueMutable(), fact.valueLayout)}));
    }
    return relations;
  }
};

class VMIMaskedLoadTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto load = dyn_cast<VMIMaskedLoadOp>(op);
    if (!load) {
      return failure();
    }
    if (changedValue != load.getResult() && changedValue != load.getMask() &&
        changedValue != load.getPassthru()) {
      return failure();
    }

    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    auto maskType = dyn_cast<VMIMaskType>(load.getMask().getType());
    auto passthruType = dyn_cast<VMIVRegType>(load.getPassthru().getType());
    if (!resultType || !maskType || !passthruType) {
      return failure();
    }

    auto assignedResultType =
        VMIVRegType::get(resultType.getContext(), resultType.getElementCount(),
                         resultType.getElementType(), changedLayout);
    auto assignedMaskType =
        VMIMaskType::get(maskType.getContext(), maskType.getElementCount(),
                         maskType.getGranularity(), changedLayout);
    auto assignedPassthruType = VMIVRegType::get(
        passthruType.getContext(), passthruType.getElementCount(),
        passthruType.getElementType(), changedLayout);
    VMILayoutSupport supports;
    if (failed(supports.getMaskedLoadLayoutFact(
            assignedResultType, assignedMaskType, assignedPassthruType))) {
      return failure();
    }

    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        valueFact(load.getResult(), changedLayout),
        operandFact(load.getMaskMutable(), changedLayout),
        operandFact(load.getPassthruMutable(), changedLayout)});
  }
};

class VMIMaskedStoreTransfer final : public VMILayoutTransfer {
public:
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>>
  query(Operation *op, Value changedValue, VMILayoutAttr changedLayout,
        const VMILayoutPropagator &propagator,
        OpOperand *changedOperand) const override {
    auto store = dyn_cast<VMIMaskedStoreOp>(op);
    if (!store) {
      return failure();
    }
    if (changedValue != store.getValue() && changedValue != store.getMask()) {
      return failure();
    }

    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    auto maskType = dyn_cast<VMIMaskType>(store.getMask().getType());
    if (!valueType || !maskType) {
      return failure();
    }

    VMILayoutSupport supports;
    VMILayoutAttr useLayout = changedLayout;
    auto assignedValueType =
        VMIVRegType::get(valueType.getContext(), valueType.getElementCount(),
                         valueType.getElementType(), useLayout);
    auto assignedMaskType =
        VMIMaskType::get(maskType.getContext(), maskType.getElementCount(),
                         maskType.getGranularity(), useLayout);
    if (failed(supports.getMaskedStoreLayoutFact(assignedValueType,
                                                 assignedMaskType))) {
      useLayout = VMILayoutAttr::getContiguous(valueType.getContext());
    }

    return makeSingleRelation(SmallVector<VMILayoutFact, mlir::pto::kValue4>{
        operandFact(store.getValueMutable(), useLayout),
        operandFact(store.getMaskMutable(), useLayout)});
  }
};

const VMILayoutTransfer *getTransfer(Operation *op) {
  static VMISameLayoutTransfer sameLayoutTransfer;
  static VMICastTransfer castTransfer;
  static VMIVexpdifTransfer vexpdifTransfer;
  static VMIBitcastTransfer bitcastTransfer;
  static VMIMaskGranularityCastTransfer maskGranularityCastTransfer;
  static VMIFreeResultLayoutTransfer freeResultLayoutTransfer;
  static VMIContiguousResultLayoutTransfer contiguousResultLayoutTransfer;
  static VMILoadTransfer loadTransfer;
  static VMIDeinterleaveLoadTransfer deinterleaveLoadTransfer;
  static VMIGroupLoadTransfer groupLoadTransfer;
  static VMIGroupReduceTransfer groupReduceTransfer;
  static VMIGroupSlotLoadTransfer groupSlotLoadTransfer;
  static VMIGroupBroadcastLoadTransfer groupBroadcastLoadTransfer;
  static VMIGroupBroadcastTransfer groupBroadcastTransfer;
  static VMIInterleaveTransfer interleaveTransfer;
  static VMIGatherTransfer gatherTransfer;
  static VMIStoreTransfer storeTransfer;
  static VMIGroupStoreTransfer groupStoreTransfer;
  static VMIMaskedLoadTransfer maskedLoadTransfer;
  static VMIMaskedStoreTransfer maskedStoreTransfer;

  if (isa<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMICreateMaskOp,
          VMICreateGroupMaskOp, VMIConstantMaskOp>(op)) {
    return &freeResultLayoutTransfer;
  }
  if (isa<VMIGroupIotaOp>(op)) {
    return &contiguousResultLayoutTransfer;
  }
  if (isa<VMILoadOp>(op)) {
    return &loadTransfer;
  }
  if (isa<VMIDeinterleaveLoadOp>(op)) {
    return &deinterleaveLoadTransfer;
  }
  if (isa<VMIGroupLoadOp>(op)) {
    return &groupLoadTransfer;
  }
  if (isa<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp,
          VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
          VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(op)) {
    return &groupReduceTransfer;
  }
  if (isa<VMIGroupSlotLoadOp>(op)) {
    return &groupSlotLoadTransfer;
  }
  if (isa<VMIGroupBroadcastLoadOp>(op)) {
    return &groupBroadcastLoadTransfer;
  }
  if (isa<VMIGroupBroadcastOp>(op)) {
    return &groupBroadcastTransfer;
  }
  if (isa<VMIVintlvOp, VMIVdintlvOp>(op)) {
    return &interleaveTransfer;
  }
  if (isa<VMIGatherOp>(op)) {
    return &gatherTransfer;
  }
  if (isa<VMIBitcastOp>(op)) {
    return &bitcastTransfer;
  }
  if (isa<VMIVexpdifOp>(op)) {
    return &vexpdifTransfer;
  }
  if (isa<VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op)) {
    // Same-width fp<->int: same-layout.  Widen/narrow: cast.
    auto srcType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto dstType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (srcType && dstType) {
      auto srcElem = srcType.getElementType();
      auto dstElem = dstType.getElementType();
      unsigned srcBits = 0;
      unsigned dstBits = 0;
      if (auto ft = dyn_cast<FloatType>(srcElem)) {
        srcBits = ft.getWidth();
      } else if (auto it = dyn_cast<IntegerType>(srcElem)) {
        srcBits = it.getWidth();
      }
      if (auto ft = dyn_cast<FloatType>(dstElem)) {
        dstBits = ft.getWidth();
      } else if (auto it = dyn_cast<IntegerType>(dstElem)) {
        dstBits = it.getWidth();
      }
      if (srcBits > 0 && srcBits == dstBits) {
        return &sameLayoutTransfer;
      }
    }
    return &castTransfer;
  }
  if (isSameLayoutOp(op)) {
    return &sameLayoutTransfer;
  }
  if (isa<VMIEnsureMaskGranularityOp>(op)) {
    return &maskGranularityCastTransfer;
  }
  if (isCastOp(op)) {
    return &castTransfer;
  }
  if (isa<VMIStoreOp>(op)) {
    return &storeTransfer;
  }
  if (isa<VMIGroupStoreOp>(op)) {
    return &groupStoreTransfer;
  }
  if (isa<VMIMaskedLoadOp>(op)) {
    return &maskedLoadTransfer;
  }
  if (isa<VMIMaskedStoreOp>(op)) {
    return &maskedStoreTransfer;
  }
  return nullptr;
}

} // namespace

VMILayoutPropagator::VMILayoutPropagator(Operation *scope)
    : scope(scope), ctx(scope ? scope->getContext() : nullptr) {}

bool VMILayoutPropagator::isLayoutValue(Value value) const {
  return isa<VMIVRegType, VMIMaskType>(value.getType());
}

VMILayoutAttr VMILayoutPropagator::getCurrentLayout(Value value) const {
  if (auto type = dyn_cast<VMIVRegType>(value.getType())) {
    return type.getLayoutAttr();
  }
  if (auto type = dyn_cast<VMIMaskType>(value.getType())) {
    return type.getLayoutAttr();
  }
  return {};
}

Type VMILayoutPropagator::getTypeWithLayout(Value value,
                                            VMILayoutAttr layout) const {
  if (auto type = dyn_cast<VMIVRegType>(value.getType())) {
    return VMIVRegType::get(ctx, type.getElementCount(),
                            type.getElementType(), layout);
  }
  if (auto type = dyn_cast<VMIMaskType>(value.getType())) {
    return VMIMaskType::get(ctx, type.getElementCount(), type.getGranularity(),
                            layout);
  }
  return {};
}

bool VMILayoutPropagator::canUseOperandLayout(OpOperand &operand,
                                              VMILayoutAttr layout) const {
  if (!layout) {
    return false;
  }
  if (!isLayoutValue(operand.get())) {
    return true;
  }
  const VMILayoutTransfer *transfer = getTransfer(operand.getOwner());
  if (!transfer) {
    return false;
  }
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>> relations = transfer->query(
      operand.getOwner(), operand.get(), layout, *this, &operand);
  if (failed(relations)) {
    return false;
  }
  for (const VMILayoutRelation &relation : *relations) {
    if (relationContainsOperandLayout(relation, operand, layout)) {
      return true;
    }
  }
  return false;
}

VMILayoutAttr VMILayoutPropagator::getRequestedLayout(Value value) const {
  auto it = assignments.find(value);
  if (it == assignments.end()) {
    return {};
  }
  return it->second.layout;
}

VMILayoutAttr
VMILayoutPropagator::getRequestedOrCurrentLayout(Value value) const {
  if (VMILayoutAttr layout = getRequestedLayout(value)) {
    return layout;
  }
  return getCurrentLayout(value);
}

VMILayoutAttr VMILayoutPropagator::getOperandLayout(OpOperand &operand) const {
  const VMIValueLayoutAssignment *assignment = lookup(operand.get());
  if (!assignment) {
    return getCurrentLayout(operand.get());
  }

  for (const VMILayoutConflict &conflict : assignment->conflicts) {
    if (conflict.operand == &operand) {
      return conflict.layout;
    }
  }
  return assignment->layout;
}

const VMIValueLayoutAssignment *
VMILayoutPropagator::lookup(Value value) const {
  auto it = assignments.find(value);
  if (it == assignments.end()) {
    return nullptr;
  }
  return &it->second;
}

void VMILayoutPropagator::addEquivalentValues(Value lhs, Value rhs) {
  if (!isLayoutValue(lhs) || !isLayoutValue(rhs) || lhs == rhs) {
    return;
  }
  if (isa<VMIVRegType>(lhs.getType()) != isa<VMIVRegType>(rhs.getType())) {
    return;
  }

  auto addEdge = [&](Value from, Value to) {
    SmallVector<Value, 2> &values = equivalentValues[from];
    if (!llvm::is_contained(values, to)) {
      values.push_back(to);
    }
  };
  addEdge(lhs, rhs);
  addEdge(rhs, lhs);
}

void VMILayoutPropagator::enqueue(Value value, VMILayoutAttr layout) {
  if (!value || !layout) {
    return;
  }
  LayoutFact fact(value, layout);
  if (llvm::is_contained(seenFacts, fact)) {
    return;
  }
  seenFacts.push_back(fact);
  worklist.push_back(fact);
}

LogicalResult
VMILayoutPropagator::addUseConflict(OpOperand &operand,
                                    VMIValueLayoutAssignment &assignment,
                                    VMILayoutAttr layout) {
  for (VMILayoutConflict &conflict : assignment.conflicts) {
    if (conflict.operand != &operand) {
      continue;
    }
    if (conflict.layout == layout) {
      return success();
    }
    return success();
  }
  assignment.conflicts.push_back(VMILayoutConflict{&operand, layout});
  return success();
}

bool VMILayoutPropagator::canProduceValueLayout(Value value,
                                                VMILayoutAttr layout) const {
  if (!layout || !isLayoutValue(value)) {
    return false;
  }
  if (getCurrentLayout(value) == layout) {
    return true;
  }
  if (isa<BlockArgument>(value)) {
    return isTypeRewriteable(value);
  }
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *definingOp = result.getDefiningOp();
    const VMILayoutTransfer *transfer = getTransfer(definingOp);
    if (!transfer) {
      return isTypeRewriteable(value);
    }
    FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>> relations = transfer->query(
        definingOp, value, layout, *this, /*changedOperand=*/nullptr);
    if (failed(relations)) {
      return false;
    }
    for (const VMILayoutRelation &relation : *relations) {
      if (relationContainsValueLayout(relation, value, layout)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

bool VMILayoutPropagator::canMaterializeLayout(
    Value value, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) const {
  static VMILayoutMaterializationTransfer materializationTransfer;
  FailureOr<SmallVector<VMILayoutRelation, mlir::pto::kValue4>> relations =
      materializationTransfer.query(value, sourceLayout, resultLayout);
  return succeeded(relations) && !relations->empty();
}

LogicalResult VMILayoutPropagator::request(Value value, VMILayoutAttr layout) {
  if (!layout) {
    return failure();
  }
  if (!isLayoutValue(value)) {
    return success();
  }

  auto it = assignments.find(value);
  if (it == assignments.end()) {
    auto inserted = assignments.try_emplace(value);
    it = inserted.first;
    orderedValues.push_back(value);
  }
  VMIValueLayoutAssignment &assignment = it->second;
  if (!assignment.layout) {
    if (!canProduceValueLayout(value, layout)) {
      return success();
    }
    assignment.layout = layout;
    enqueue(value, layout);
    return success();
  }
  if (assignment.layout == layout) {
    return success();
  }
  return success();
}

LogicalResult VMILayoutPropagator::request(OpOperand &operand,
                                           VMILayoutAttr layout) {
  if (!layout) {
    return failure();
  }
  Value value = operand.get();
  if (!isLayoutValue(value)) {
    return success();
  }

  auto it = assignments.find(value);
  if (it == assignments.end()) {
    auto inserted = assignments.try_emplace(value);
    it = inserted.first;
    orderedValues.push_back(value);
  }

  VMIValueLayoutAssignment &assignment = it->second;
  if (assignment.layout == layout) {
    return propagateOperandFact(operand, layout);
  }
  if (!assignment.layout && isa<OpResult>(value)) {
    if (failed(request(value, layout))) {
      return failure();
    }
    if (assignment.layout == layout) {
      return propagateOperandFact(operand, layout);
    }
  }
  if (failed(addUseConflict(operand, assignment, layout))) {
    return failure();
  }
  if (getOperandLayout(operand) != layout) {
    return success();
  }
  return propagateOperandFact(operand, layout);
}

LogicalResult VMILayoutPropagator::propagateFact(Value value,
                                                 VMILayoutAttr layout) {
  if (!isLayoutValue(value)) {
    return success();
  }

  auto equivalentIt = equivalentValues.find(value);
  if (equivalentIt != equivalentValues.end()) {
    for (Value equivalent : equivalentIt->second) {
      if (failed(request(equivalent, layout))) {
        return failure();
      }
    }
  }

  if (auto result = dyn_cast<OpResult>(value)) {
    if (failed(propagateThrough(result.getDefiningOp(), value, layout))) {
      return failure();
    }

  }

  SmallVector<OpOperand *, mlir::pto::kValue8> uses;
  for (OpOperand &use : value.getUses()) {
    uses.push_back(&use);
  }
  for (OpOperand *use : uses) {
    if (getOperandLayout(*use) == layout &&
        failed(propagateThrough(use->getOwner(), value, layout))) {
      return failure();
    }
  }

  return success();
}

LogicalResult VMILayoutPropagator::propagateOperandFact(OpOperand &operand,
                                                        VMILayoutAttr layout) {
  OperandLayoutFact fact(&operand, layout);
  if (llvm::is_contained(seenOperandFacts, fact)) {
    return success();
  }
  seenOperandFacts.push_back(fact);
  return propagateThrough(operand.getOwner(), operand.get(), layout, &operand);
}

LogicalResult VMILayoutPropagator::propagateThrough(
    Operation *op, Value changedValue, VMILayoutAttr changedLayout,
    OpOperand *changedOperand) {
  const VMILayoutTransfer *transfer = op ? getTransfer(op) : nullptr;
  if (!transfer) {
    return success();
  }
  return transfer->propagate(op, changedValue, changedLayout, *this,
                             changedOperand);
}

LogicalResult VMILayoutPropagator::run() {
  while (!worklist.empty()) {
    LayoutFact fact = worklist.pop_back_val();
    if (failed(propagateFact(fact.first, fact.second))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMILayoutPropagator::verifyMaterializationPlan() const {
  for (Value value : orderedValues) {
    auto it = assignments.find(value);
    if (it == assignments.end()) {
      continue;
    }

    const VMIValueLayoutAssignment &assignment = it->second;
    if (!assignment.layout) {
      return emitError(value.getLoc())
             << kVMIDiagLayoutContractPrefix
             << "layout assignment has conflicts but no primary layout";
    }

    VMILayoutAttr currentLayout = getCurrentLayout(value);
    if (currentLayout != assignment.layout && !isTypeRewriteable(value)) {
      if (!currentLayout) {
        return emitError(value.getLoc())
               << kVMIDiagLayoutContractPrefix
               << "cannot materialize primary VMI layout from an unassigned "
                  "boundary value";
      }
      if (!canMaterializeLayout(value, currentLayout, assignment.layout)) {
        return emitError(value.getLoc())
               << kVMIDiagLayoutContractPrefix
               << "cannot materialize primary VMI layout "
               << assignment.layout << " from " << currentLayout;
      }
    }

    for (const VMILayoutConflict &conflict : assignment.conflicts) {
      if (!canMaterializeLayout(value, assignment.layout, conflict.layout)) {
        return emitError(conflict.operand ? conflict.operand->getOwner()->getLoc()
                                          : value.getLoc())
               << kVMIDiagLayoutContractPrefix
               << "cannot materialize requested VMI layout "
               << conflict.layout << " from assigned layout "
               << assignment.layout;
      }
    }
  }
  return success();
}

bool VMILayoutPropagator::isTypeRewriteable(Value value) const {
  auto result = dyn_cast<OpResult>(value);
  if (result) {
    Operation *definingOp = result.getDefiningOp();
    if (!definingOp || !scope) {
      return false;
    }
    return definingOp == scope || scope->isAncestor(definingOp);
  }

  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || !scope) {
    return false;
  }
  Operation *parentOp = arg.getOwner()->getParentOp();
  return parentOp && (parentOp == scope || scope->isAncestor(parentOp));
}

FailureOr<Value> VMILayoutPropagator::materializeAt(Value source,
                                                    VMILayoutAttr layout,
                                                    RewriterBase &rewriter,
                                                    Location loc) {
  VMILayoutAttr sourceLayout = getCurrentLayout(source);
  if (!sourceLayout) {
    return failure();
  }
  if (sourceLayout == layout) {
    return source;
  }

  Type resultType = getTypeWithLayout(source, layout);
  if (!resultType) {
    return failure();
  }
  if (isa<VMIVRegType>(source.getType())) {
    return rewriter.create<VMIEnsureLayoutOp>(loc, resultType, source)
        .getResult();
  }
  if (isa<VMIMaskType>(source.getType())) {
    return rewriter.create<VMIEnsureMaskLayoutOp>(loc, resultType, source)
        .getResult();
  }
  return failure();
}

LogicalResult VMILayoutPropagator::materializePrimary(
    Value value, const VMIValueLayoutAssignment &assignment,
    RewriterBase &rewriter, DenseMap<Value, Value> &assignedValues) {
  auto sourceType = dyn_cast<VMIVRegType>(value.getType());
  auto sourceMaskType = dyn_cast<VMIMaskType>(value.getType());
  if (!sourceType && !sourceMaskType) {
    return success();
  }

  Type assignedType = getTypeWithLayout(value, assignment.layout);
  if (!assignedType) {
    return failure();
  }
  if (getCurrentLayout(value) == assignment.layout) {
    assignedValues[value] = value;
    return success();
  }

  if (isTypeRewriteable(value)) {
    value.setType(assignedType);
    assignedValues[value] = value;
    return success();
  }

  if (!getCurrentLayout(value)) {
    Operation *owner = scope;
    if (auto result = dyn_cast<OpResult>(value)) {
      owner = result.getDefiningOp();
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      owner = arg.getOwner()->getParentOp();
    }
    if (!owner) {
      return failure();
    }
    return owner->emitError()
           << kVMIDiagLayoutContractPrefix
           << "cannot materialize a primary VMI layout from an unassigned "
              "boundary value";
  }

  OpBuilder::InsertionGuard guard(rewriter);
  if (auto result = dyn_cast<OpResult>(value)) {
    rewriter.setInsertionPointAfter(result.getDefiningOp());
  } else if (auto arg = dyn_cast<BlockArgument>(value)) {
    rewriter.setInsertionPointToStart(arg.getOwner());
  } else {
    return failure();
  }

  FailureOr<Value> materialized =
      materializeAt(value, assignment.layout, rewriter, value.getLoc());
  if (failed(materialized)) {
    return failure();
  }

  if (*materialized != value) {
    value.replaceAllUsesExcept(*materialized,
                               (*materialized).getDefiningOp());
  }
  assignedValues[value] = *materialized;
  return success();
}

LogicalResult VMILayoutPropagator::materializeUseConflict(
    Value assignedValue, VMILayoutConflict conflict, RewriterBase &rewriter) {
  if (!conflict.operand) {
    return success();
  }
  if (getCurrentLayout(assignedValue) == conflict.layout) {
    conflict.operand->set(assignedValue);
    return success();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  Operation *owner = conflict.operand->getOwner();
  rewriter.setInsertionPoint(owner);
  FailureOr<Value> materialized =
      materializeAt(assignedValue, conflict.layout, rewriter, owner->getLoc());
  if (failed(materialized)) {
    return owner->emitError()
           << kVMIDiagLayoutContractPrefix
           << "cannot materialize requested VMI operand layout "
           << conflict.layout;
  }
  conflict.operand->set(*materialized);
  return success();
}

LogicalResult VMILayoutPropagator::apply(RewriterBase &rewriter) {
  DenseMap<Value, Value> assignedValues;
  for (Value value : orderedValues) {
    auto it = assignments.find(value);
    if (it == assignments.end()) {
      continue;
    }
    if (failed(materializePrimary(value, it->second, rewriter,
                                  assignedValues))) {
      return failure();
    }
  }

  for (Value value : orderedValues) {
    auto it = assignments.find(value);
    if (it == assignments.end()) {
      continue;
    }
    Value assignedValue = assignedValues.lookup(value);
    if (!assignedValue) {
      assignedValue = value;
    }
    for (VMILayoutConflict conflict : it->second.conflicts) {
      if (failed(materializeUseConflict(assignedValue, conflict, rewriter))) {
        return failure();
      }
    }
  }
  return success();
}
