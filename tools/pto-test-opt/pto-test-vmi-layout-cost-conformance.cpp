// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-test-vmi-layout-cost-conformance.cpp --------------------------===//
//
// Checks that every VMI layout relation exposed for a marked operation is in
// the cost model's domain.  It clones the containing function once per
// relation so every relation, including alternate operand layouts, is also
// lowered independently and compared with its predicted rearrangement count.
// This pass is registered only by pto-test-opt.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/VMILayoutCostModel.h"
#include "PTO/Transforms/VMILayoutPlanner.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr StringLiteral kConformanceMarker = "test.vmi_layout_cost_conformance";
constexpr StringLiteral kExpectedRearrangements =
    "test.vmi_layout_expected_rearrangements";
constexpr StringLiteral kExpectedLocation = "location";
constexpr StringLiteral kExpectedCost = "cost";
constexpr StringLiteral kExpectedSourceOp = "source_op";
constexpr StringLiteral kRelationTarget = "test.vmi_layout_relation_target";

struct ExpectedRearrangement {
  LocationAttr location;
  StringAttr sourceOp;
  int64_t cost = 0;
};

struct CostedRelation {
  VMILayoutOpRelation relation;
  int64_t cost = 0;
};

struct CostedOperation {
  Operation *op = nullptr;
  SmallVector<CostedRelation, mlir::pto::kValue4> relations;
};

static bool sameRelationTuple(const VMILayoutOpRelation &lhs,
                              const VMILayoutOpRelation &rhs) {
  if (lhs.op != rhs.op || lhs.directProducer != rhs.directProducer ||
      lhs.intrinsicRearrangementCost != rhs.intrinsicRearrangementCost ||
      lhs.ports.size() != rhs.ports.size()) {
    return false;
  }
  return llvm::all_of(llvm::zip_equal(lhs.ports, rhs.ports), [](auto pair) {
    const auto &[lhsPort, rhsPort] = pair;
    return lhsPort.kind == rhsPort.kind && lhsPort.index == rhsPort.index &&
           lhsPort.layout == rhsPort.layout;
  });
}

static VMILayoutAttr getLayout(Type type) {
  if (auto vreg = dyn_cast<VMIVRegType>(type)) {
    return vreg.getLayoutAttr();
  }
  if (auto mask = dyn_cast<VMIMaskType>(type)) {
    return mask.getLayoutAttr();
  }
  return {};
}

static void rememberLayout(VMILayoutAttr layout,
                           SmallVectorImpl<VMILayoutAttr> &layouts) {
  if (layout && !llvm::is_contained(layouts, layout)) {
    layouts.push_back(layout);
  }
}

static SmallVector<VMILayoutAttr, mlir::pto::kValue16>
collectCandidateLayouts(ModuleOp module) {
  MLIRContext *context = module.getContext();
  SmallVector<VMILayoutAttr, mlir::pto::kValue16> layouts;
  rememberLayout(VMILayoutAttr::getContiguous(context), layouts);
  for (int64_t factor : {int64_t(2), int64_t(4)}) {
    rememberLayout(VMILayoutAttr::getDeinterleaved(context, factor), layouts);
    rememberLayout(VMILayoutAttr::getBlockDeinterleaved(context, factor),
                   layouts);
    rememberLayout(VMILayoutAttr::getContiguous(context, factor), layouts);
  }
  module.walk([&](Operation *op) {
    for (Type type : op->getOperandTypes()) {
      rememberLayout(getLayout(type), layouts);
    }
    for (Type type : op->getResultTypes()) {
      rememberLayout(getLayout(type), layouts);
    }
    // Include polymorphic group-slot candidates even when the operation's
    // result type is intentionally left layout-polymorphic.  Otherwise the
    // conformance pass would fail to exercise Support relations that are
    // legal only for a particular group count/slot packing.
    for (StringRef attrName : {StringRef("group"), StringRef("num_groups")}) {
      auto groupsAttr = op->getAttrOfType<IntegerAttr>(attrName);
      if (!groupsAttr || groupsAttr.getInt() <= 0) {
        // This operation does not define a positive group count.
        continue;
      }
      int64_t groups = groupsAttr.getInt();
      for (int64_t slots : {int64_t(1), int64_t(2), int64_t(4), int64_t(8)}) {
        rememberLayout(VMILayoutAttr::getGroupSlots(context, groups, slots),
                       layouts);
        for (int64_t laneStride : {int64_t(2), int64_t(4)}) {
          rememberLayout(VMILayoutAttr::getGroupSlots(
                              context, groups, slots, laneStride),
                         layouts);
        }
      }
    }
  });
  return layouts;
}

static LogicalResult addRelationToPlan(const VMILayoutOpRelation &relation,
                                       VMILayoutPlan &plan) {
  Operation *op = relation.op;
  if (!op) {
    return failure();
  }
  for (const VMILayoutPortAssignment &port : relation.ports) {
    if (!port.layout) {
      return failure();
    }
    if (port.kind == VMILayoutPortKind::Operand) {
      if (port.index >= op->getNumOperands()) {
        return failure();
      }
      OpOperand &operand = op->getOpOperand(port.index);
      VMILayoutAttr primaryLayout = getLayout(operand.get().getType());
      plan.valueLayouts[operand.get()] =
          primaryLayout ? primaryLayout : port.layout;
      plan.useLayouts[&operand] = port.layout;
      continue;
    }
    if (port.index >= op->getNumResults()) {
      return failure();
    }
    plan.valueLayouts[op->getResult(port.index)] = port.layout;
  }
  plan.selectedRelations[op] = 0;
  return success();
}

static void printRelation(StringRef functionName, unsigned relationIndex,
                          const VMILayoutOpRelation &relation,
                          VMILayoutScopeCost cost) {
  llvm::outs() << "vmi-layout-cost-conformance " << functionName << ' '
               << relation.op->getName() << " relation=" << relationIndex
               << " cost=" << cost.total;
  for (const VMILayoutPortAssignment &port : relation.ports) {
    llvm::outs() << ' '
                 << (port.kind == VMILayoutPortKind::Operand ? "operand"
                                                             : "result")
                 << port.index << '=' << port.layout;
  }
  llvm::outs() << '\n';
}

static LogicalResult checkOperation(Operation *op, StringRef functionName,
                                    ArrayRef<VMILayoutAttr> candidateLayouts,
                                    SmallVectorImpl<CostedRelation> &costed) {
  auto relations =
      VMILayoutRelationProvider().enumerateRelations(op, candidateLayouts);
  if (failed(relations) || relations->empty()) {
    InFlightDiagnostic diagnostic = op->emitError()
        << "test layout-cost conformance marker has no exposed relation";
    if (isa<VMICvtOp, VMIvLoadOp, VMIvStoreOp>(op)) {
      diagnostic << "; run the unified-to-legacy canonicalization first, or "
                    "add a canonical relation fixture for this operation";
    }
    return failure();
  }
  for (auto [index, relation] : llvm::enumerate(*relations)) {
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> singletonDomain;
    for (const VMILayoutPortAssignment &port : relation.ports) {
      rememberLayout(port.layout, singletonDomain);
    }
    auto singletonRelations =
        VMILayoutRelationProvider().enumerateRelations(op, singletonDomain);
    const VMILayoutOpRelation *singletonMatch = nullptr;
    if (succeeded(singletonRelations)) {
      for (const VMILayoutOpRelation &candidate : *singletonRelations) {
        if (sameRelationTuple(candidate, relation)) {
          singletonMatch = &candidate;
          break;
        }
      }
    }
    if (!singletonMatch) {
      op->emitError() << "relation is not stable under its singleton layout "
                         "domain #" << index;
      return failure();
    }
    VMILayoutPlan plan;
    if (failed(addRelationToPlan(relation, plan))) {
      op->emitError() << "malformed exposed layout relation #" << index;
      return failure();
    }
    auto cost = evaluateVMILayoutPlanCost({relation}, plan);
    if (failed(cost)) {
      InFlightDiagnostic diagnostic =
          op->emitError() << "cost model rejected exposed layout relation #"
                          << index;
      for (const VMILayoutPortAssignment &port : relation.ports) {
        diagnostic << ' '
                   << (port.kind == VMILayoutPortKind::Operand ? "operand"
                                                               : "result")
                   << port.index << '=' << port.layout;
      }
      return failure();
    }
    VMILayoutPlan singletonPlan;
    if (failed(addRelationToPlan(*singletonMatch, singletonPlan))) {
      op->emitError() << "singleton relation cannot be materialized #" << index;
      return failure();
    }
    auto singletonCost =
        evaluateVMILayoutPlanCost({*singletonMatch}, singletonPlan);
    if (failed(singletonCost) || singletonCost->total != cost->total) {
      op->emitError() << "relation cost is not stable under its singleton "
                         "layout domain #" << index;
      return failure();
    }
    printRelation(functionName, index, relation, *cost);
    costed.push_back(CostedRelation{relation, cost->total});
  }
  llvm::outs() << "vmi-layout-cost-conformance-summary " << functionName << ' '
               << op->getName() << " relations=" << relations->size() << '\n';
  return success();
}

static Type assignLayout(Type type, VMILayoutAttr layout) {
  if (auto vreg = dyn_cast<VMIVRegType>(type)) {
    return VMIVRegType::get(type.getContext(), vreg.getElementCount(),
                            vreg.getElementType(), layout);
  }
  if (auto mask = dyn_cast<VMIMaskType>(type)) {
    return VMIMaskType::get(type.getContext(), mask.getElementCount(),
                            mask.getGranularity(), layout);
  }
  return {};
}

static LogicalResult assignRelation(Operation *op,
                                    const VMILayoutOpRelation &relation,
                                    Location relationLoc, OpBuilder &builder) {
  op->setLoc(relationLoc);
  for (const VMILayoutPortAssignment &port : relation.ports) {
    if (port.kind == VMILayoutPortKind::Result) {
      Type targetType =
          assignLayout(op->getResult(port.index).getType(), port.layout);
      if (!targetType || targetType != op->getResult(port.index).getType()) {
        return failure();
      }
      continue;
    }
    Value operand = op->getOperand(port.index);
    Type targetType = assignLayout(operand.getType(), port.layout);
    if (!targetType) {
      return failure();
    }
    if (operand.getType() == targetType) {
      // The relation already matches the operand's concrete type.
      continue;
    }
    builder.setInsertionPoint(op);
    Value converted;
    if (isa<VMIVRegType>(targetType)) {
      converted =
          builder.create<VMIEnsureLayoutOp>(relationLoc, targetType, operand)
              .getResult();
    } else {
      converted =
          builder
              .create<VMIEnsureMaskLayoutOp>(relationLoc, targetType, operand)
              .getResult();
    }
    op->setOperand(port.index, converted);
  }
  return success();
}

static LogicalResult
materializeRelationFunction(const CostedOperation &costedOperation,
                            const CostedRelation &costed, unsigned cloneIndex,
                            OpBuilder &builder) {
  auto function = costedOperation.op->getParentOfType<FunctionOpInterface>();
  if (!function) {
    return failure();
  }

  costedOperation.op->setAttr(kRelationTarget, builder.getUnitAttr());
  builder.setInsertionPointAfter(function.getOperation());
  Operation *clone = builder.clone(*function.getOperation());
  costedOperation.op->removeAttr(kRelationTarget);

  std::string cloneName =
      (Twine(function.getName()) + "__vmi_layout_relation_" + Twine(cloneIndex))
          .str();
  clone->setAttr(SymbolTable::getSymbolAttrName(),
                 builder.getStringAttr(cloneName));

  Operation *target = nullptr;
  clone->walk([&](Operation *nested) {
    nested->removeAttr(kConformanceMarker);
    if (nested->hasAttr(kRelationTarget)) {
      target = nested;
      nested->removeAttr(kRelationTarget);
    }
  });
  if (!target) {
    clone->erase();
    return failure();
  }

  Location relationLoc =
      NameLoc::get(builder.getStringAttr((Twine(cloneName) + "_target").str()),
                   target->getLoc());
  if (failed(assignRelation(target, costed.relation, relationLoc, builder))) {
    clone->erase();
    return failure();
  }

  auto expectation = builder.getDictionaryAttr(
      {{builder.getStringAttr(kExpectedLocation), relationLoc},
       {builder.getStringAttr(kExpectedSourceOp),
        builder.getStringAttr(costedOperation.op->getName().getStringRef())},
       {builder.getStringAttr(kExpectedCost),
        builder.getI64IntegerAttr(costed.cost)}});
  clone->setAttr(kExpectedRearrangements, builder.getArrayAttr({expectation}));
  return success();
}

class TestVMILayoutCostConformancePass final
    : public PassWrapper<TestVMILayoutCostConformancePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVMILayoutCostConformancePass)

  StringRef getArgument() const final {
    return "test-vmi-layout-cost-conformance";
  }

  StringRef getDescription() const final {
    return "check that exposed VMI layout relations are costable";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<VMILayoutAttr, mlir::pto::kValue16> candidateLayouts =
        collectCandidateLayouts(module);
    SmallVector<CostedOperation, mlir::pto::kValue16> costedOperations;
    WalkResult result = module.walk([&](Operation *op) {
      if (!op->hasAttr(kConformanceMarker)) {
        return WalkResult::advance();
      }
      auto function = op->getParentOfType<FunctionOpInterface>();
      StringRef functionName = function ? function.getName() : StringRef("-");
      CostedOperation costedOperation;
      costedOperation.op = op;
      if (failed(checkOperation(op, functionName, candidateLayouts,
                                costedOperation.relations))) {
        return WalkResult::interrupt();
      }
      if (!function) {
        op->emitError() << "layout-cost lowering conformance requires a "
                           "function parent";
        return WalkResult::interrupt();
      }
      costedOperations.push_back(std::move(costedOperation));
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    OpBuilder builder(module.getContext());
    unsigned cloneIndex = 0;
    for (const CostedOperation &costedOperation : costedOperations) {
      for (const CostedRelation &costed : costedOperation.relations) {
        if (failed(materializeRelationFunction(costedOperation, costed,
                                               cloneIndex++, builder))) {
          costedOperation.op->emitError()
              << "failed to materialize relation-specific lowering fixture";
          signalPassFailure();
          return;
        }
      }
      costedOperation.op->removeAttr(kConformanceMarker);
    }
  }
};

static bool isLayoutRearrangement(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  if (opName == "pto.por" || opName == "pto.vor") {
    StringRef packName = opName == "pto.por" ? "pto.ppack" : "pto.vpack";
    for (Value operand : op->getOperands().take_front(2)) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || producer->getLoc() != op->getLoc()) {
        // A conversion producer from another location is not part of this
        // relation's rearrangement count.
        continue;
      }
      StringRef producerName = producer->getName().getStringRef();
      if (producerName == packName ||
          (opName == "pto.vor" && producerName == "pto.vcvt" &&
           producer->hasAttr("part"))) {
        return true;
      }
    }
    return false;
  }
  return llvm::StringSwitch<bool>(opName)
      .Cases("pto.vintlv", "pto.vdintlv", "pto.vzunpack", "pto.vsunpack", true)
      .Case("pto.vpack", true)
      .Cases("pto.pintlv_b8", "pto.pintlv_b16", "pto.pintlv_b32", true)
      .Cases("pto.pdintlv_b8", "pto.pdintlv_b16", "pto.pdintlv_b32", true)
      .Cases("pto.ppack", "pto.punpack", true)
      .Default(false);
}

static bool isDirectSemanticLowering(StringRef sourceOp, Operation *loweredOp) {
  StringRef loweredName = loweredOp->getName().getStringRef();
  return (sourceOp == "pto.vmi.vintlv" && loweredName == "pto.vintlv") ||
         (sourceOp == "pto.vmi.vdintlv" && loweredName == "pto.vdintlv");
}

class TestVMILayoutLoweringConformancePass final
    : public PassWrapper<TestVMILayoutLoweringConformancePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVMILayoutLoweringConformancePass)

  StringRef getArgument() const final {
    return "test-vmi-layout-lowering-conformance";
  }

  StringRef getDescription() const final {
    return "compare VMI relation costs with emitted VPTO rearrangements";
  }

  void runOnOperation() final {
    WalkResult result = getOperation().walk([&](FunctionOpInterface function) {
      auto expectations =
          function->getAttrOfType<ArrayAttr>(kExpectedRearrangements);
      if (!expectations) {
        return WalkResult::advance();
      }
      for (Attribute expectationAttr : expectations) {
        auto expectation = dyn_cast<DictionaryAttr>(expectationAttr);
        auto expectedLocation =
            expectation ? expectation.getAs<LocationAttr>(kExpectedLocation)
                        : LocationAttr{};
        auto expectedCost = expectation
                                ? expectation.getAs<IntegerAttr>(kExpectedCost)
                                : IntegerAttr{};
        auto expectedSourceOp =
            expectation ? expectation.getAs<StringAttr>(kExpectedSourceOp)
                        : StringAttr{};
        if (!expectedLocation || !expectedCost || !expectedSourceOp) {
          function.emitError() << "malformed layout conformance expectation";
          return WalkResult::interrupt();
        }
        int64_t actual = 0;
        function.walk([&](Operation *op) {
          if (isLayoutRearrangement(op) && op->getLoc() == expectedLocation &&
              !isDirectSemanticLowering(expectedSourceOp.getValue(), op)) {
            ++actual;
          }
        });
        int64_t expectedValue = expectedCost.getInt();
        if (actual != expectedValue) {
          function.emitError()
              << "layout-cost/lowering mismatch at " << expectedLocation
              << ": relation cost=" << expectedValue
              << ", emitted rearrangements=" << actual;
          return WalkResult::interrupt();
        }
        llvm::outs() << "vmi-layout-lowering-conformance " << function.getName()
                     << " cost=" << expectedValue
                     << " rearrangements=" << actual << '\n';
      }
      function->removeAttr(kExpectedRearrangements);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::pto {

void registerTestVMILayoutCostConformancePass() {
  PassRegistration<TestVMILayoutCostConformancePass>();
  PassRegistration<TestVMILayoutLoweringConformancePass>();
}

} // namespace mlir::pto
