// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutPlanner.cpp - VMI layout planning model ------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutPlanner.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutConflictSolver.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "vmi-layout-planner"

namespace {

static VMILayoutPortAssignment operandPort(unsigned index,
                                           VMILayoutAttr layout) {
  return VMILayoutPortAssignment{VMILayoutPortKind::Operand, index, layout};
}

static VMILayoutPortAssignment resultPort(unsigned index,
                                          VMILayoutAttr layout) {
  return VMILayoutPortAssignment{VMILayoutPortKind::Result, index, layout};
}

static std::string getDecisionLayoutKey(VMILayoutAttr layout) {
  if (!layout) {
    return "<none>";
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << layout;
  return stream.str();
}

static std::string getDecisionRelationKey(const VMILayoutOpRelation &relation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (const VMILayoutPortAssignment &port : relation.ports) {
    stream << (port.kind == VMILayoutPortKind::Operand ? 'o' : 'r')
           << port.index << ':';
    if (port.kind == VMILayoutPortKind::Operand &&
        port.index < relation.op->getNumOperands()) {
      stream << relation.op->getOperand(port.index).getType();
    } else if (port.kind == VMILayoutPortKind::Result &&
               port.index < relation.op->getNumResults()) {
      stream << relation.op->getResult(port.index).getType();
    } else {
      stream << "<invalid>";
    }
    stream << '=' << getDecisionLayoutKey(port.layout) << ';';
  }
  stream << "direct=" << relation.directProducer
         << ";rearrange=" << relation.intrinsicRearrangementCost
         << ";preference=" << relation.preferencePenalty;
  return stream.str();
}

static SmallVector<VMILayoutDecisionGroup, mlir::pto::kValue4>
buildDecisionGroups(ArrayRef<VMILayoutSolverOp> plannerOps) {
  SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue8>
      adjacency(plannerOps.size());
  DenseMap<Operation *, unsigned> indices;
  for (auto [index, solverOp] : llvm::enumerate(plannerOps)) {
    indices[solverOp.op] = index;
  }
  for (auto [consumerIndex, solverOp] : llvm::enumerate(plannerOps)) {
    for (Value operand : solverOp.op->getOperands()) {
      auto producerIt = indices.find(operand.getDefiningOp());
      if (producerIt == indices.end()) {
        continue;
      }
      adjacency[consumerIndex].push_back(producerIt->second);
      adjacency[producerIt->second].push_back(consumerIndex);
    }
  }
  SmallVector<unsigned, mlir::pto::kValue8> components(plannerOps.size(),
                                                        0);
  unsigned nextComponent = 0;
  llvm::SmallBitVector visited(plannerOps.size());
  for (unsigned root = 0; root < plannerOps.size(); ++root) {
    if (visited.test(root)) {
      continue;
    }
    SmallVector<unsigned, mlir::pto::kValue4> worklist{root};
    visited.set(root);
    while (!worklist.empty()) {
      unsigned current = worklist.pop_back_val();
      components[current] = nextComponent;
      for (unsigned neighbor : adjacency[current]) {
        if (!visited.test(neighbor)) {
          visited.set(neighbor);
          worklist.push_back(neighbor);
        }
      }
    }
    ++nextComponent;
  }

  llvm::StringMap<unsigned> classIndices;
  SmallVector<VMILayoutDecisionGroup, mlir::pto::kValue4> groups;
  for (auto [opIndex, solverOp] : llvm::enumerate(plannerOps)) {
    if (solverOp.relations.size() < 2) {
      continue;
    }
    SmallVector<std::string, mlir::pto::kValue4> relations;
    for (const VMILayoutOpRelation &relation : solverOp.relations) {
      relations.push_back(getDecisionRelationKey(relation));
    }
    llvm::sort(relations);
    std::string key;
    llvm::raw_string_ostream stream(key);
    stream << components[opIndex] << '|';
    for (const std::string &relation : relations) {
      stream << '[' << relation << ']';
    }
    auto classIt = classIndices.find(key);
    unsigned groupIndex = 0;
    if (classIt == classIndices.end()) {
      groupIndex = groups.size();
      classIndices.try_emplace(key, groupIndex);
      groups.emplace_back();
    } else {
      groupIndex = classIt->second;
    }
    VMILayoutDecisionGroup &group = groups[groupIndex];
    group.opIndices.push_back(opIndex);
    SmallVector<unsigned, mlir::pto::kValue4> mapping;
    for (const std::string &representative : relations) {
      auto relationIt = llvm::find_if(solverOp.relations,
                                      [&](const VMILayoutOpRelation &relation) {
                                        return getDecisionRelationKey(relation) ==
                                               representative;
                                      });
      if (relationIt == solverOp.relations.end()) {
        mapping.clear();
        break;
      }
      mapping.push_back(relationIt - solverOp.relations.begin());
    }
    if (mapping.size() != relations.size()) {
      group.opIndices.pop_back();
      continue;
    }
    group.memberRelationIndices.push_back(std::move(mapping));
  }
  llvm::erase_if(groups, [](const VMILayoutDecisionGroup &group) {
    return group.opIndices.size() < 2;
  });
  return groups;
}

static bool isLayoutType(Type type) {
  return isa<VMIVRegType, VMIMaskType>(type);
}

static bool isVMILayoutStructuralOp(Operation *op) {
  return isa<cf::BranchOp, cf::CondBranchOp, cf::SwitchOp, func::CallOp,
             func::ReturnOp, scf::ConditionOp, scf::ExecuteRegionOp, scf::ForOp,
             scf::IfOp, scf::IndexSwitchOp, scf::YieldOp, scf::WhileOp>(op);
}

static VMILayoutAttr getExplicitLayout(Type type) {
  if (auto vreg = dyn_cast<VMIVRegType>(type)) {
    return vreg.getLayoutAttr();
  }
  if (auto mask = dyn_cast<VMIMaskType>(type)) {
    return mask.getLayoutAttr();
  }
  return {};
}

static bool haveSamePortTuple(const VMILayoutOpRelation &lhs,
                              const VMILayoutOpRelation &rhs) {
  if (lhs.op != rhs.op || lhs.ports.size() != rhs.ports.size() ||
      lhs.directProducer != rhs.directProducer ||
      lhs.intrinsicRearrangementCost != rhs.intrinsicRearrangementCost ||
      lhs.preferencePenalty != rhs.preferencePenalty) {
    return false;
  }
  return llvm::all_of(llvm::zip_equal(lhs.ports, rhs.ports), [](auto ports) {
    const auto &[lhsPort, rhsPort] = ports;
    return lhsPort.kind == rhsPort.kind && lhsPort.index == rhsPort.index &&
           lhsPort.layout == rhsPort.layout;
  });
}

static bool haveSameInterleaveLayouts(const VMIInterleaveLayoutFact &lhs,
                                      const VMIInterleaveLayoutFact &rhs) {
  return lhs.lhsLayout == rhs.lhsLayout && rhs.rhsLayout == lhs.rhsLayout &&
         lhs.maskLayout == rhs.maskLayout && lhs.lowLayout == rhs.lowLayout &&
         lhs.highLayout == rhs.highLayout;
}

static void
appendUniqueRelation(SmallVectorImpl<VMILayoutOpRelation> &relations,
                     VMILayoutOpRelation relation) {
  if (llvm::none_of(relations, [&](const VMILayoutOpRelation &existing) {
        return haveSamePortTuple(existing, relation);
      })) {
    relations.push_back(std::move(relation));
  }
}

static bool isReachableFromExplicitOperands(const VMILayoutOpRelation &relation,
                                            const VMILayoutSupport &supports) {
  for (const VMILayoutPortAssignment &port : relation.ports) {
    if (port.kind != VMILayoutPortKind::Operand ||
        port.index >= relation.op->getNumOperands()) {
      continue;
    }
    Type sourceType = relation.op->getOperand(port.index).getType();
    VMILayoutAttr sourceLayout = getExplicitLayout(sourceType);
    if (!sourceLayout || sourceLayout == port.layout) {
      continue;
    }
    if (auto vreg = dyn_cast<VMIVRegType>(sourceType)) {
      auto target = VMIVRegType::get(vreg.getContext(), vreg.getElementCount(),
                                     vreg.getElementType(), port.layout);
      if (failed(supports.getEnsureLayoutFact(vreg, target))) {
        return false;
      }
      continue;
    }
    if (auto mask = dyn_cast<VMIMaskType>(sourceType)) {
      auto target = VMIMaskType::get(mask.getContext(), mask.getElementCount(),
                                     mask.getGranularity(), port.layout);
      if (failed(supports.getEnsureMaskLayoutFact(mask, target))) {
        return false;
      }
    }
  }
  return true;
}


static bool
relationRespectsExplicitResults(const VMILayoutOpRelation &relation) {
  if (!relation.op) {
    return false;
  }
  for (const VMILayoutPortAssignment &port : relation.ports) {
    if (port.kind != VMILayoutPortKind::Result ||
        port.index >= relation.op->getNumResults()) {
      continue;
    }
    VMILayoutAttr explicitLayout =
        getExplicitLayout(relation.op->getResult(port.index).getType());
    if (explicitLayout && explicitLayout != port.layout) {
      return false;
    }
  }
  return true;
}

static void
appendReachableUniqueRelation(SmallVectorImpl<VMILayoutOpRelation> &relations,
                              VMILayoutOpRelation relation,
                              const VMILayoutSupport &supports) {
  if (isReachableFromExplicitOperands(relation, supports) &&
      relationRespectsExplicitResults(relation)) {
    appendUniqueRelation(relations, std::move(relation));
  }
}

static FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>
makeReachableRelation(VMILayoutOpRelation relation,
                      const VMILayoutSupport &supports) {
  SmallVector<VMILayoutOpRelation, mlir::pto::kValue4> relations;
  appendReachableUniqueRelation(relations, std::move(relation), supports);
  if (relations.empty()) {
    return failure();
  }
  return relations;
}

static bool isStructuralTransportValue(Value value) {
  if (!value) {
    return false;
  }
  Operation *owner = value.getDefiningOp();
  if (isa<BlockArgument>(value)) {
    owner = value.getParentBlock()->getParentOp();
  }
  for (Operation *parent = owner; parent; parent = parent->getParentOp()) {
    if (isVMILayoutStructuralOp(parent)) {
      return true;
    }
  }
  return false;
}

static bool isVMILayoutABIBoundaryOp(Operation *op) {
  return isa<func::CallOp, func::ReturnOp>(op);
}

static VMILayoutAttr getABIBoundaryLayout(Type type) {
  if (!isLayoutType(type)) {
    return {};
  }
  if (VMILayoutAttr explicitLayout = getExplicitLayout(type)) {
    return explicitLayout;
  }
  return VMILayoutAttr::getContiguous(type.getContext());
}

static VMILayoutAttr getFunctionArgumentBoundaryLayout(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument) {
    return {};
  }
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front() ||
      argument.getArgNumber() >= function.getNumArguments()) {
    return {};
  }
  return getABIBoundaryLayout(
      function.getArgumentTypes()[argument.getArgNumber()]);
}

static SmallVector<VMILayoutFixedAssignment, mlir::pto::kValue4>
collectFixedAssignments(ArrayRef<Operation *> component) {
  SmallVector<VMILayoutFixedAssignment, mlir::pto::kValue4> assignments;
  DenseSet<Value> seen;
  for (Operation *op : component) {
    for (Value operand : op->getOperands()) {
      VMILayoutAttr layout = getFunctionArgumentBoundaryLayout(operand);
      if (layout && seen.insert(operand).second) {
        assignments.push_back({operand, layout});
      }
    }
  }
  return assignments;
}

static SmallVector<VMILayoutAttr, mlir::pto::kValue4>
collectPlanLayouts(const VMILayoutPlan &plan);

} // namespace

namespace mlir::pto {

SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue8>
VMILayoutStructuralEdgeProvider::enumerateEdges(Operation *op) const {
  SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue8> edges;
  if (!op) {
    return edges;
  }
  auto add = [&](Value source, Value destination, bool abi = false) {
    if (source && destination && isLayoutType(source.getType()) &&
        isLayoutType(destination.getType())) {
      edges.push_back({source, destination, abi});
    }
  };
  if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    for (auto [successorIndex, successor] :
         llvm::enumerate(op->getSuccessors())) {
      SuccessorOperands operands = branch.getSuccessorOperands(successorIndex);
      Block *destination = successor;
      unsigned produced = operands.getProducedOperandCount();
      for (auto [index, value] :
           llvm::enumerate(operands.getForwardedOperands())) {
        unsigned argumentIndex = produced + index;
        if (argumentIndex < destination->getNumArguments()) {
          add(value, destination->getArgument(argumentIndex));
        }
      }
    }
  }
  if (auto regionBranch = dyn_cast<RegionBranchOpInterface>(op)) {
    SmallVector<RegionSuccessor, mlir::pto::kValue4> successors;
    regionBranch.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (const RegionSuccessor &successor : successors) {
      if (successor.isParent()) {
        continue;
      }
      ValueRange destinations = successor.getSuccessorInputs();
      OperandRange sources = regionBranch.getEntrySuccessorOperands(
          RegionBranchPoint(successor));
      for (auto [index, source] : llvm::enumerate(sources)) {
        if (index < destinations.size()) {
          add(source, destinations[index]);
        }
      }
    }
    for (Region &region : op->getRegions()) {
      SmallVector<RegionSuccessor, mlir::pto::kValue4> regionSuccessors;
      regionBranch.getSuccessorRegions(region, regionSuccessors);
      for (const RegionSuccessor &successor : regionSuccessors) {
        ValueRange destinations = successor.getSuccessorInputs();
        if (destinations.empty()) {
          continue;
        }
        for (Block &block : region) {
          auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
              block.getTerminator());
          if (!terminator) {
            continue;
          }
          OperandRange sources = terminator.getSuccessorOperands(
              RegionBranchPoint(successor));
          for (auto [index, source] : llvm::enumerate(sources)) {
            if (index < destinations.size()) {
              add(source, destinations[index]);
            }
          }
        }
      }
    }
  }
  return edges;
}

} // namespace mlir::pto

namespace {

static SmallVector<VMILayoutAttr, mlir::pto::kValue4>
collectPlanLayouts(const VMILayoutPlan &plan) {
  SmallVector<VMILayoutAttr, mlir::pto::kValue4> layouts;
  auto remember = [&](VMILayoutAttr layout) {
    if (layout && !llvm::is_contained(layouts, layout)) {
      layouts.push_back(layout);
    }
  };
  for (const auto &[value, layout] : plan.valueLayouts) {
    remember(layout);
  }
  for (const auto &[operand, layout] : plan.useLayouts) {
    remember(layout);
  }
  return layouts;
}

static bool relationMatchesPlan(const VMILayoutOpRelation &relation,
                                const VMILayoutPlan &plan) {
  Operation *op = relation.op;
  if (!op) {
    return false;
  }
  if (relation.ports.empty() && !relation.endpoints.empty()) {
    return llvm::all_of(relation.endpoints, [&](const VMILayoutRelationEndpoint &endpoint) {
      if (!endpoint.value || !endpoint.layout)
        return false;
      if (endpoint.use) {
        auto it = plan.useLayouts.find(endpoint.use);
        return it != plan.useLayouts.end() && it->second == endpoint.layout;
      }
      auto it = plan.valueLayouts.find(endpoint.value);
      return it != plan.valueLayouts.end() && it->second == endpoint.layout;
    });
  }
  SmallVector<bool, mlir::pto::kValue4> operandCovered(op->getNumOperands(),
                                                       false);
  SmallVector<bool, mlir::pto::kValue4> resultCovered(op->getNumResults(),
                                                      false);
  for (const VMILayoutPortAssignment &port : relation.ports) {
    if (port.kind == VMILayoutPortKind::Operand) {
      if (port.index >= op->getNumOperands()) {
        return false;
      }
      if (!isLayoutType(op->getOperand(port.index).getType()) ||
          operandCovered[port.index]) {
        return false;
      }
      operandCovered[port.index] = true;
      auto it = plan.useLayouts.find(&op->getOpOperand(port.index));
      if (it == plan.useLayouts.end() || it->second != port.layout) {
        return false;
      }
      continue;
    }
    if (port.index >= op->getNumResults()) {
      return false;
    }
    if (!isLayoutType(op->getResult(port.index).getType()) ||
        resultCovered[port.index]) {
      return false;
    }
    resultCovered[port.index] = true;
    auto it = plan.valueLayouts.find(op->getResult(port.index));
    if (it == plan.valueLayouts.end() || it->second != port.layout) {
      return false;
    }
  }
  for (unsigned index = 0; index < op->getNumOperands(); ++index) {
    if (isLayoutType(op->getOperand(index).getType()) &&
        !operandCovered[index]) {
      return false;
    }
  }
  for (unsigned index = 0; index < op->getNumResults(); ++index) {
    if (isLayoutType(op->getResult(index).getType()) && !resultCovered[index]) {
      return false;
    }
  }
  return true;
}

static LogicalResult verifySelectedRelations(const VMILayoutPlan &plan,
                                             ArrayRef<VMILayoutAttr> layouts) {
  VMILayoutRelationProvider provider;
  for (const auto &selectedRelation : plan.selectedRelations) {
    Operation *op = selectedRelation.first;
    if (!op) {
      return failure();
    }
    FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>> relations =
        provider.enumerateRelations(op, layouts);
    if (failed(relations) ||
        llvm::none_of(*relations, [&](const VMILayoutOpRelation &relation) {
          return relationMatchesPlan(relation, plan);
        })) {
      return failure();
    }
  }
  return success();
}

static LogicalResult requestPlan(const VMILayoutPlan &plan,
                                 VMILayoutPropagator &propagator) {
  SmallVector<Operation *, mlir::pto::kValue16> orderedOps;
  for (const auto &[op, relation] : plan.selectedRelations) {
    (void)relation;
    if (op) {
      orderedOps.push_back(op);
    }
  }
  llvm::sort(orderedOps, [](Operation *lhs, Operation *rhs) {
    if (lhs == rhs) {
      return false;
    }
    if (lhs->getBlock() == rhs->getBlock()) {
      return lhs->isBeforeInBlock(rhs);
    }
    return lhs->getBlock() < rhs->getBlock();
  });

  DenseSet<Value> installedValues;
  DenseSet<OpOperand *> installedUses;
  for (Operation *op : orderedOps) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto it = plan.useLayouts.find(&operand);
      if (it == plan.useLayouts.end()) {
        continue;
      }
      Value source = operand.get();
      auto valueIt = plan.valueLayouts.find(source);
      VMILayoutAttr sourceLayout = valueIt != plan.valueLayouts.end()
                                       ? valueIt->second
                                       : getExplicitLayout(source.getType());
      if (!sourceLayout && isStructuralTransportValue(source)) {
        // Structural transport values (block arguments, call results, and
        // SCF region results) have no producer-owned physical layout before
        // assignment.  Seed their primary layout from the validated use
        // relation; conversions at other uses remain explicit conflicts.
        sourceLayout = isa<BlockArgument>(source)
                           ? VMILayoutAttr::getContiguous(source.getContext())
                           : it->second;
      }
      if (sourceLayout && installedValues.insert(source).second &&
          failed(propagator.installPlanned(source, sourceLayout))) {
        op->emitError() << kVMIDiagLayoutContractPrefix
                        << "failed to install source layout for operand #"
                        << operand.getOperandNumber();
        return failure();
      }
      if (failed(propagator.installPlanned(operand, it->second))) {
        op->emitError() << kVMIDiagLayoutContractPrefix
                        << "failed to install operand layout for operand #"
                        << operand.getOperandNumber();
        return failure();
      }
      installedUses.insert(&operand);
    }
    for (OpResult result : op->getResults()) {
      auto it = plan.valueLayouts.find(result);
      if (it == plan.valueLayouts.end()) {
        continue;
      }
      if (installedValues.insert(result).second &&
          failed(propagator.installPlanned(result, it->second))) {
        op->emitError() << kVMIDiagLayoutContractPrefix
                        << "failed to install result layout #"
                        << result.getResultNumber();
        return failure();
      }
    }
  }
  for (const auto &[value, layout] : plan.valueLayouts) {
    if (installedValues.insert(value).second &&
        failed(propagator.installPlanned(value, layout))) {
      return failure();
    }
  }
  for (const auto &[operand, layout] : plan.useLayouts) {
    if (operand && installedUses.insert(operand).second &&
        failed(propagator.installPlanned(*operand, layout))) {
      return failure();
    }
  }
  // The plan has already been validated against the immutable relation graph.
  // Running the legacy propagator here would perform a second layout choice
  // from transfer-specific preferences and could overwrite the selected
  // relation.  Only record the plan's value/use assignments; propagation is
  // intentionally owned by the cost solver.
  return success();
}

static bool planWasCommitted(const VMILayoutPlan &plan,
                             const VMILayoutPropagator &propagator) {
  for (const auto &[value, layout] : plan.valueLayouts) {
    if (isStructuralTransportValue(value)) {
      continue;
    }
    if (propagator.getRequestedOrCurrentLayout(value) != layout) {
      return false;
    }
  }
  for (const auto &[operand, layout] : plan.useLayouts) {
    if (operand && isStructuralTransportValue(operand->get())) {
      continue;
    }
    if (!operand || propagator.getRequestedLayout(*operand) != layout) {
      return false;
    }
  }
  return true;
}

static bool isLayoutValue(Value value) {
  return value && isLayoutType(value.getType());
}

static void appendLayoutValues(Operation *op, SmallVectorImpl<Value> &values) {
  for (Value operand : op->getOperands()) {
    if (isLayoutValue(operand)) {
      values.push_back(operand);
    }
  }
  for (Value result : op->getResults()) {
    if (isLayoutValue(result)) {
      values.push_back(result);
    }
  }
}

static bool collectComponent(Value seed, SmallVectorImpl<Operation *> &ops,
                             llvm::SmallPtrSetImpl<Operation *> &seenOps,
                             DenseSet<Value> &globallySeenValues) {
  SmallVector<Value, mlir::pto::kValue16> worklist{seed};
  VMILayoutStructuralEdgeProvider structuralEdges;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!isLayoutValue(value) || !globallySeenValues.insert(value).second) {
      continue;
    }
    auto visitOp = [&](Operation *op) {
      if (!op || !seenOps.insert(op).second) {
        return;
      }
      SmallVector<Value, mlir::pto::kValue4> connected;
      appendLayoutValues(op, connected);
      if (connected.empty()) {
        return;
      }
      ops.push_back(op);
      worklist.append(connected);
      for (const VMILayoutEqualityConstraint &edge :
           structuralEdges.enumerateEdges(op)) {
        worklist.push_back(edge.source);
        worklist.push_back(edge.destination);
      }
      if (isVMILayoutStructuralOp(op)) {
        op->walk([&](Operation *nested) {
          if (nested == op || !seenOps.insert(nested).second) {
            return;
          }
          SmallVector<Value, mlir::pto::kValue4> nestedValues;
          appendLayoutValues(nested, nestedValues);
          if (nestedValues.empty()) {
            return;
          }
          ops.push_back(nested);
          worklist.append(nestedValues);
        });
      }
    };
    visitOp(value.getDefiningOp());
    for (Operation *user : value.getUsers()) {
      visitOp(user);
    }
  }
  return !ops.empty();
}

static void rememberRelationLayouts(ArrayRef<VMILayoutOpRelation> relations,
                                    SmallVectorImpl<VMILayoutAttr> &layouts) {
  for (const VMILayoutOpRelation &relation : relations) {
    for (const VMILayoutPortAssignment &port : relation.ports) {
      if (port.layout && !llvm::is_contained(layouts, port.layout)) {
        layouts.push_back(port.layout);
      }
    }
    for (const VMILayoutRelationEndpoint &endpoint : relation.endpoints) {
      if (endpoint.layout && !llvm::is_contained(layouts, endpoint.layout))
        layouts.push_back(endpoint.layout);
    }
  }
}

static void rememberLayout(VMILayoutAttr layout,
                           SmallVectorImpl<VMILayoutAttr> &layouts) {
  if (layout && !llvm::is_contained(layouts, layout)) {
    layouts.push_back(layout);
  }
}

/// Layouts to query the vexpdif facts table with.  The table itself decides
/// which of them form a legal relation, so the query set only has to cover the
/// layouts the table can mention, plus the layouts the operation already
/// spells out (an operation whose ports carry explicit layouts keeps feeding
/// relation enumeration even when no other candidate is available).
static SmallVector<VMILayoutAttr, mlir::pto::kValue8>
getVexpdifQueryLayouts(Operation *op, ArrayRef<VMILayoutAttr> seedLayouts) {
  SmallVector<VMILayoutAttr, mlir::pto::kValue8> layouts;
  for (VMILayoutAttr layout : seedLayouts) {
    rememberLayout(layout, layouts);
  }
  rememberLayout(VMILayoutAttr::getContiguous(op->getContext()), layouts);
  for (int64_t factor : {int64_t(2), int64_t(4)}) {
    rememberLayout(VMILayoutAttr::getDeinterleaved(op->getContext(), factor),
                   layouts);
    rememberLayout(
        VMILayoutAttr::getBlockDeinterleaved(op->getContext(), factor),
        layouts);
  }
  for (Type type : op->getOperandTypes()) {
    rememberLayout(getExplicitLayout(type), layouts);
  }
  for (Type type : op->getResultTypes()) {
    rememberLayout(getExplicitLayout(type), layouts);
  }
  return layouts;
}

static SmallVector<VMILayoutAttr, mlir::pto::kValue8>
getGroupReduceQueryLayouts(MLIRContext *context,
                           ArrayRef<VMILayoutAttr> seedLayouts) {
  SmallVector<VMILayoutAttr, mlir::pto::kValue8> layouts;
  for (VMILayoutAttr layout : seedLayouts) {
    rememberLayout(layout, layouts);
  }
  rememberLayout(VMILayoutAttr::getContiguous(context), layouts);
  for (int64_t factor : {int64_t(2), int64_t(4)}) {
    rememberLayout(VMILayoutAttr::getDeinterleaved(context, factor), layouts);
    rememberLayout(VMILayoutAttr::getBlockDeinterleaved(context, factor),
                   layouts);
  }
  return layouts;
}

static std::optional<int64_t> getGroupReduceNumGroups(Operation *op) {
  if (isa<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp,
          VMIGroupReduceAddIOp, VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(
          op)) {
    if (auto groups = op->getAttrOfType<IntegerAttr>("num_groups")) {
      return groups.getInt();
    }
    return std::nullopt;
  }
  if (isa<VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>(op)) {
    if (auto groups = op->getAttrOfType<IntegerAttr>("group")) {
      return groups.getInt();
    }
  }
  return std::nullopt;
}

static void
rememberGroupReduceRelationLayouts(VMIVRegType sourceType, int64_t numGroups,
                                   SmallVectorImpl<VMILayoutAttr> &layouts) {
  VMILayoutSupport supports;
  for (VMILayoutAttr candidate :
       getGroupReduceQueryLayouts(sourceType.getContext(), layouts)) {
    auto facts = supports.getGroupReduceLayoutFactsForLayout(
        sourceType, numGroups, VMIGroupReduceLayoutPort::Source, candidate);
    if (failed(facts)) {
      continue;
    }
    for (const VMIGroupReduceLayoutFact &fact : *facts) {
      rememberLayout(fact.sourceLayout, layouts);
      rememberLayout(fact.maskLayout, layouts);
      rememberLayout(fact.resultLayout, layouts);
    }
  }
}

/// Reports an op for which no legal layout relation could be enumerated.  The
/// support model usually knows exactly why (a shape table row, an alignment
/// rule, or a lowering capability), so surface that reason when there is one
/// instead of a generic sentence that hides it.
static void reportMissingRelation(Operation *op, const std::string &reason) {
  if (!reason.empty()) {
    op->emitError() << kVMIDiagUnsupportedPrefix << reason;
    return;
  }
  op->emitError() << kVMIDiagUnsupportedPrefix
                  << "no legal VMI layout relation is registered for "
                  << op->getName();
}

static FailureOr<SmallVector<VMILayoutSolverOp, mlir::pto::kValue8>>
buildPlannerOps(ArrayRef<Operation *> ops) {
  VMILayoutRelationProvider provider;
  SmallVector<VMILayoutAttr, mlir::pto::kValue8> layouts;
  SmallVector<VMILayoutSolverOp, mlir::pto::kValue8> plannerOps;
  plannerOps.reserve(ops.size());

  // Collect operation-induced layout candidates before enumerating any
  // relation.  A helper such as ensure_mask_granularity may appear before its
  // consumer in IR order, while the consumer is what introduces the useful
  // physical layout (for example deinterleaved=4 for partial group-reduce).
  // A single forward pass would permanently under-constrain that helper.
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      rememberLayout(getFunctionArgumentBoundaryLayout(operand), layouts);
    }
    if (isVMILayoutABIBoundaryOp(op)) {
      auto relations = provider.enumerateRelations(op);
      if (succeeded(relations)) {
        rememberRelationLayouts(*relations, layouts);
      }
    }
    if (isVMILayoutStructuralOp(op)) {
      continue;
    }
    std::optional<int64_t> groups = getGroupReduceNumGroups(op);
    auto sourceType = op->getNumOperands() == 0
                          ? VMIVRegType{}
                          : dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    if (groups && sourceType) {
      rememberGroupReduceRelationLayouts(sourceType, *groups, layouts);
    }
  }
  for (Operation *op : ops) {
    if (isVMILayoutStructuralOp(op))
      continue;
    for (Type type : op->getOperandTypes()) {
      VMILayoutAttr layout = getExplicitLayout(type);
      if (layout && !llvm::is_contained(layouts, layout)) {
        layouts.push_back(layout);
      }
    }
    for (Type type : op->getResultTypes()) {
      VMILayoutAttr layout = getExplicitLayout(type);
      if (layout && !llvm::is_contained(layouts, layout)) {
        layouts.push_back(layout);
      }
    }
    if (auto interleave = dyn_cast<VMIVintlvOp>(op)) {
      VMILayoutSupport supports;
      if (auto preferred = supports.getPreferredVintlvLayoutFact(
              cast<VMIVRegType>(interleave.getLow().getType()));
          succeeded(preferred) &&
          !llvm::is_contained(layouts, preferred->lhsLayout)) {
        layouts.push_back(preferred->lhsLayout);
      }
    } else if (auto interleave = dyn_cast<VMIVdintlvOp>(op)) {
      VMILayoutSupport supports;
      if (auto preferred = supports.getPreferredVdintlvLayoutFact(
              cast<VMIVRegType>(interleave.getLow().getType()));
          succeeded(preferred) &&
          !llvm::is_contained(layouts, preferred->lhsLayout)) {
        layouts.push_back(preferred->lhsLayout);
      }
    }
    std::string relationReason;
    auto relations = provider.enumerateRelations(op, layouts, &relationReason);
    if (failed(relations) || relations->empty()) {
      VMILayoutSupport supports;
      if (auto groupSlot = dyn_cast<VMIGroupSlotLoadOp>(op)) {
        auto resultType =
            dyn_cast<VMIVRegType>(groupSlot.getResult().getType());
        if (resultType) {
          VMILayoutAttr layout = VMILayoutAttr::getGroupSlots(
              op->getContext(), groupSlot.getNumGroupsAttr().getInt(),
              groupSlot.getNumGroupsAttr().getInt());
          auto typed = VMIVRegType::get(resultType.getContext(),
                                        resultType.getElementCount(),
                                        resultType.getElementType(), layout);
          if (succeeded(supports.getGroupSlotLoadLayoutFact(
                  typed, groupSlot.getSourceGroupStride(),
                  groupSlot.getNumGroupsAttr().getInt()))) {
            layouts.push_back(layout);
            continue;
          }
        }
      }
      if (auto groupLoad = dyn_cast<VMIGroupLoadOp>(op)) {
        auto resultType =
            dyn_cast<VMIVRegType>(groupLoad.getResult().getType());
        if (resultType) {
          VMILayoutAttr contiguous =
              VMILayoutAttr::getContiguous(op->getContext());
          auto typed = VMIVRegType::get(
              resultType.getContext(), resultType.getElementCount(),
              resultType.getElementType(), contiguous);
          auto fact = supports.getGroupLoadLayoutFact(
              typed, groupLoad.getRowStride(),
              groupLoad.getNumGroupsAttr().getInt());
          if (succeeded(fact)) {
            layouts.push_back(fact->resultLayout);
            continue;
          }
          int64_t groups = groupLoad.getNumGroupsAttr().getInt();
          if (groups > 0 && resultType.getElementCount() % groups == 0) {
            int64_t groupSize = resultType.getElementCount() / groups;
            bool foundBlockLayout = false;
            for (int64_t factor : {int64_t(2), int64_t(4)}) {
              if ((groupSize == 16 && factor != 2) ||
                  (groupSize == 32 && factor != 4)) {
                continue;
              }
              VMILayoutAttr block = VMILayoutAttr::getBlockDeinterleaved(
                  op->getContext(), factor);
              auto typedBlock = VMIVRegType::get(
                  resultType.getContext(), resultType.getElementCount(),
                  resultType.getElementType(), block);
              if (succeeded(supports.getGroupLoadLayoutFact(
                      typedBlock, groupLoad.getRowStride(), groups))) {
                layouts.push_back(block);
                foundBlockLayout = true;
                break;
              }
            }
            if (foundBlockLayout) {
              continue;
            }
          }
        }
      }
      if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
        auto sourceType = cast<VMIVRegType>(broadcast.getSource().getType());
        auto resultType = cast<VMIVRegType>(broadcast.getResult().getType());
        int64_t groups = broadcast.getNumGroupsAttr().getInt();
        auto fact = supports.getGroupBroadcastLayoutFactForLayouts(
            sourceType, resultType, groups);
        if (succeeded(fact)) {
          layouts.push_back(fact->sourceLayout);
          layouts.push_back(fact->resultLayout);
          continue;
        }
        VMILayoutAttr slots = VMILayoutAttr::getGroupSlots(
            op->getContext(), groups, mlir::pto::kValue8);
        auto facts = supports.getGroupBroadcastLayoutFactsForLayout(
            sourceType, resultType, groups, VMIGroupBroadcastLayoutPort::Source,
            slots);
        if (succeeded(facts) && !facts->empty()) {
          for (const VMIGroupBroadcastLayoutFact &fact : *facts) {
            layouts.push_back(fact.sourceLayout);
            layouts.push_back(fact.resultLayout);
          }
          continue;
        }
      }
      reportMissingRelation(op, relationReason);
      return failure();
    }
    rememberRelationLayouts(*relations, layouts);
  }
  if (layouts.empty()) {
    if (!ops.empty()) {
      ops.front()->emitError() << kVMIDiagUnsupportedPrefix
                               << "no candidate VMI layout is available for "
                                  "this component";
    }
    return failure();
  }
  for (Operation *op : ops) {
    if (isVMILayoutStructuralOp(op) && !isVMILayoutABIBoundaryOp(op)) {
      continue;
    }
    std::string relationReason;
    auto relations = provider.enumerateRelations(op, layouts, &relationReason);
    if (failed(relations)) {
      auto preferredRelations = provider.enumerateRelations(op, {},
                                                            &relationReason);
      if (failed(preferredRelations) || preferredRelations->empty()) {
        reportMissingRelation(op, relationReason);
        return failure();
      }
      relations = std::move(preferredRelations);
    }
    if (relations->empty()) {
      // Some table-backed producers (notably group_slot_load) have a single
      // preferred relation that is independent of the component candidate
      // pool.  Preserve that relation when candidate filtering has no rows.
      auto preferredRelations = provider.enumerateRelations(op, {},
                                                            &relationReason);
      if (succeeded(preferredRelations) && !preferredRelations->empty()) {
        relations = std::move(preferredRelations);
      }
    }
    if (failed(relations) || relations->empty()) {
      reportMissingRelation(op, relationReason);
      return failure();
    }
    llvm::erase_if(*relations, [](const VMILayoutOpRelation &relation) {
      return !relationRespectsExplicitResults(relation);
    });
    if (relations->empty()) {
      op->emitError() << kVMIDiagUnsupportedPrefix
                      << "all registered VMI layout relations conflict with "
                         "explicit result layouts";
      return failure();
    }
    plannerOps.push_back(VMILayoutSolverOp{op, std::move(*relations)});
  }
  return plannerOps;
}

static FailureOr<VMILayoutPlan>
solveComponent(ArrayRef<Operation *> component,
               const VMILayoutPlannerOptions &options) {
  auto ops = buildPlannerOps(component);
  if (failed(ops)) {
    LLVM_DEBUG(llvm::dbgs() << "layout planner: unsupported component\n");
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "layout planner: component with " << ops->size()
                 << " ops\n";
    for (const VMILayoutSolverOp &op : *ops) {
      llvm::dbgs() << "  " << op.op->getName() << ": " << op.relations.size()
                   << " relations\n";
    }
  });
  VMILayoutConflictSolverOptions solverOptions;
  solverOptions.maxFrontierEntries = options.maxFrontierEntriesPerComponent;
  solverOptions.maxTransitions = options.maxTransitionsPerComponent;
  SmallVector<VMILayoutFixedAssignment, mlir::pto::kValue4> fixedAssignments =
      collectFixedAssignments(component);
  solverOptions.fixedAssignments = fixedAssignments;
  SmallVector<VMILayoutDecisionGroup, mlir::pto::kValue4> decisionGroups =
      buildDecisionGroups(*ops);
  solverOptions.decisionGroups = decisionGroups;
  VMILayoutStructuralEdgeProvider edgeProvider;
  SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue16> equalities;
  for (Operation *op : component) {
    auto opEqualities = edgeProvider.enumerateEdges(op);
    equalities.append(opEqualities.begin(), opEqualities.end());
  }
  solverOptions.equalityConstraints = equalities;
  auto plan = solveVMILayoutConflictComponent(*ops, solverOptions);
  if (failed(plan)) {
    LLVM_DEBUG(llvm::dbgs() << "layout planner: solver failed component ops="
                            << ops->size() << "\n");
    // The relation graph was built, so the search itself found no complete
    // assignment.  Report that here: buildPlannerOps failures already emitted
    // their own, more specific diagnostic.
    if (!component.empty()) {
      component.front()->emitError()
          << kVMIDiagLayoutContractPrefix
          << "no complete legal VMI layout plan exists for this component";
    }
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "layout planner: solved component ops="
                          << ops->size() << "\n");
  return plan;
}

// Candidate layouts for an operation whose layout-bearing ports have to agree.
// The planner normally supplies the domain; when it does not, fall back to the
// layouts spelled on the operation itself so relation enumeration does not
// depend on which caller supplied the domain.
static void
appendSameLayoutCandidates(Operation *op,
                           ArrayRef<VMILayoutAttr> polymorphicLayouts,
                           SmallVectorImpl<VMILayoutAttr> &candidates) {
  candidates.append(polymorphicLayouts.begin(), polymorphicLayouts.end());
  if (!candidates.empty()) {
    return;
  }
  for (Type type : op->getOperandTypes()) {
    if (VMILayoutAttr layout = getExplicitLayout(type);
        layout && !llvm::is_contained(candidates, layout)) {
      candidates.push_back(layout);
    }
  }
  for (Type type : op->getResultTypes()) {
    if (VMILayoutAttr layout = getExplicitLayout(type);
        layout && !llvm::is_contained(candidates, layout)) {
      candidates.push_back(layout);
    }
  }
}

static void appendSameLayoutRelations(
    Operation *op, ArrayRef<VMILayoutAttr> candidates,
    const VMILayoutSupport &supports,
    SmallVectorImpl<VMILayoutOpRelation> &relations,
    std::string *reason = nullptr) {
  for (VMILayoutAttr layout : candidates) {
    if (!layout) {
      continue;
    }
    if (failed(supports.getSameLayoutRelationSupport(op, layout, reason))) {
      continue;
    }
    VMILayoutOpRelation relation;
    relation.op = op;
    for (OpOperand &operand : op->getOpOperands()) {
      if (isLayoutType(operand.get().getType())) {
        relation.ports.push_back(
            operandPort(operand.getOperandNumber(), layout));
      }
    }
    for (OpResult result : op->getResults()) {
      if (isLayoutType(result.getType())) {
        relation.ports.push_back(resultPort(result.getResultNumber(), layout));
      }
    }
    if (!relation.ports.empty()) {
      appendReachableUniqueRelation(relations, std::move(relation), supports);
    }
  }
}

} // namespace

bool mlir::pto::isVMISameLayoutOp(Operation *op) {
  return isa<VMIAddFOp, VMIAddIOp, VMISubFOp, VMISubIOp, VMIMulFOp, VMIMulIOp,
             VMIVaddcOp, VMIVaddcsOp, VMIAddSOp, VMIMulSOp, VMIMaxSOp,
             VMIMinSOp, VMIShlSOp, VMIShrSOp, VMIVmullOp, VMIFmaOp, VMIDivFOp,
             VMIMinFOp, VMIMinIOp, VMIMaxFOp, VMIMaxIOp, VMINegFOp, VMINegIOp,
             VMIAbsFOp,
             VMIAbsIOp, VMISqrtOp, VMIExpOp, VMILnOp, VMIReluOp, VMIAndIOp,
             VMIOrIOp, VMIXOrIOp, VMIShLIOp, VMIShRUIOp, VMIShRSIOp, VMINotOp,
             VMICmpFOp, VMICmpIOp, VMISelectOp, VMIMaskAndOp, VMIMaskOrOp,
             VMIMaskXOrOp, VMIMaskNotOp, VMIVaddOp, VMIVsubOp, VMIVmulOp,
             VMIVdivOp, VMIVminOp, VMIVmaxOp, VMIVnegOp, VMIVabsOp, VMIVsqrtOp,
             VMIVexpOp, VMIVlnOp, VMIVreluOp, VMIVandOp, VMIVorOp, VMIVxorOp,
             VMIVshlOp, VMIVshrOp, VMIVnotOp, VMIVcmpOp, VMIVcmpsOp, VMIvSelOp,
             VMIVaxpyOp, VMIVlreluOp, VMIVpreluOp, VMIVmulaOp>(op);
}

bool mlir::pto::isVMILayoutCastOp(Operation *op) {
  return isa<VMIExtFOp, VMIExtSIOp, VMIExtUIOp, VMITruncFOp, VMITruncIOp,
             VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op);
}

FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>
VMILayoutRelationProvider::enumerateRelations(
    Operation *op, ArrayRef<VMILayoutAttr> polymorphicLayouts,
    std::string *reason) const {
  if (!op) {
    return failure();
  }

  VMILayoutSupport supports;
  SmallVector<VMILayoutOpRelation, mlir::pto::kValue4> relations;
  if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
    auto function = returnOp->getParentOfType<func::FuncOp>();
    if (!function || returnOp.getNumOperands() != function.getNumResults()) {
      return failure();
    }
    VMILayoutOpRelation relation;
    relation.op = op;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      VMILayoutAttr layout =
          getABIBoundaryLayout(function.getResultTypes()[index]);
      if (layout) {
        relation.endpoints.push_back(
            {operand, &returnOp->getOpOperand(index), layout});
      }
    }
    if (relation.endpoints.empty()) {
      return failure();
    }
    relations.push_back(std::move(relation));
    return relations;
  }
  if (auto call = dyn_cast<func::CallOp>(op)) {
    VMILayoutOpRelation relation;
    relation.op = op;
    for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
      if (VMILayoutAttr layout = getABIBoundaryLayout(operand.getType())) {
        relation.endpoints.push_back(
            {operand, &call->getOpOperand(index), layout});
      }
    }
    for (Value result : call.getResults()) {
      if (VMILayoutAttr layout = getABIBoundaryLayout(result.getType())) {
        relation.endpoints.push_back({result, nullptr, layout});
      }
    }
    if (relation.endpoints.empty()) {
      return failure();
    }
    relations.push_back(std::move(relation));
    return relations;
  }
  if (isVMILayoutCastOp(op)) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !resultType) {
      return failure();
    }
    bool sameWidthNumericCast =
        isa<VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op) &&
        pto::getPTOStorageElemBitWidth(sourceType.getElementType()) != 0 &&
        pto::getPTOStorageElemBitWidth(sourceType.getElementType()) ==
            pto::getPTOStorageElemBitWidth(resultType.getElementType());
    auto preferred =
        supports.getPreferredCastLayoutFact(sourceType, resultType);
    auto appendFacts = [&](VMIVRegType concreteSourceType) {
      if (sameWidthNumericCast) {
        VMILayoutAttr layout = concreteSourceType.getLayoutAttr();
        if (!layout) {
          return;
        }
        VMIVRegType concreteResultType = VMIVRegType::get(
            resultType.getContext(), resultType.getElementCount(),
            resultType.getElementType(), layout);
        auto fact = supports.getSameWidthCastLayoutFact(concreteSourceType,
                                                        concreteResultType);
        if (succeeded(fact)) {
          appendReachableUniqueRelation(
              relations,
              VMILayoutOpRelation{op,
                                  {operandPort(0, fact->sourceLayout),
                                   resultPort(0, fact->resultLayout)},
                                  /*directProducer=*/false},
              supports);
        }
        return;
      }
      auto facts = supports.getCastLayoutFacts(concreteSourceType, resultType);
      if (failed(facts)) {
        return;
      }
      for (const VMICastLayoutFact &fact : *facts) {
        if (failed(supports.validateCastOperationRelation(op, fact.sourceLayout,
                                                          fact.resultLayout))) {
          continue;
        }
        uint64_t preferencePenalty =
            succeeded(preferred) &&
                    (fact.sourceLayout != preferred->sourceLayout ||
                     fact.resultLayout != preferred->resultLayout)
                ? 1
                : 0;
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, fact.sourceLayout),
                                 resultPort(0, fact.resultLayout)},
                                /*directProducer=*/false,
                                fact.intrinsicRearrangementCost,
                                preferencePenalty},
            supports);
      }
    };
    appendFacts(sourceType);
    // Casts whose producer has no explicit layout still need a contiguous
    // seed so the cost solver can materialize their result into a downstream
    // relation when no table row uses the component's first candidate.
    if (relations.empty() && !sourceType.getLayoutAttr()) {
      appendFacts(VMIVRegType::get(
          sourceType.getContext(), sourceType.getElementCount(),
          sourceType.getElementType(),
          VMILayoutAttr::getContiguous(op->getContext())));
    }
    SmallVector<int64_t, mlir::pto::kValue2> instantiatedGroups;
    for (VMILayoutAttr layout : polymorphicLayouts) {
      if (!layout || !layout.isGroupSlots() ||
          llvm::is_contained(instantiatedGroups, layout.getNumGroups())) {
        continue;
      }
      instantiatedGroups.push_back(layout.getNumGroups());
      appendFacts(VMIVRegType::get(sourceType.getContext(),
                                   sourceType.getElementCount(),
                                   sourceType.getElementType(), layout));
    }
    if (relations.empty()) {
      return failure();
    }
    return relations;
  }

  if (isa<VMIEnsureLayoutOp, VMIEnsureMaskLayoutOp>(op)) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }
    VMILayoutAttr sourceLayout = getExplicitLayout(op->getOperand(0).getType());
    VMILayoutAttr resultLayout = getExplicitLayout(op->getResult(0).getType());
    if (!sourceLayout || !resultLayout) {
      return failure();
    }
    LogicalResult supported = failure();
    if (auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType())) {
      auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
      if (resultType) {
        supported = success(
            succeeded(supports.getEnsureLayoutFact(sourceType, resultType)));
      }
    } else if (auto sourceType =
                   dyn_cast<VMIMaskType>(op->getOperand(0).getType())) {
      auto resultType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
      if (resultType) {
        supported = success(succeeded(
            supports.getEnsureMaskLayoutFact(sourceType, resultType)));
      }
    }
    if (failed(supported)) {
      return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(0, sourceLayout), resultPort(0, resultLayout)},
         false},
        supports);
  }

  if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
    auto source = dyn_cast<VMIMaskType>(ensure.getSource().getType());
    auto result = dyn_cast<VMIMaskType>(ensure.getResult().getType());
    if (!source || !result) {
      return failure();
    }
    VMILayoutAttr explicitSource = source.getLayoutAttr();
    VMILayoutAttr explicitResult = result.getLayoutAttr();
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> sourceCandidates;
    if (explicitSource) {
      sourceCandidates.push_back(explicitSource);
    } else {
      sourceCandidates.push_back(
          VMILayoutAttr::getContiguous(op->getContext()));
      for (VMILayoutAttr layout : polymorphicLayouts) {
        if (layout && !llvm::is_contained(sourceCandidates, layout)) {
          sourceCandidates.push_back(layout);
        }
      }
    }

    SmallVector<VMILayoutOpRelation, mlir::pto::kValue4> relations;
    for (VMILayoutAttr sourceLayout : sourceCandidates) {
      auto facts = supports.getMaskGranularityCastLayoutFactsForLayout(
          source, result, VMICastLayoutPort::Source, sourceLayout);
      if (failed(facts)) {
        continue;
      }
      for (const auto &fact : *facts) {
        if (explicitResult && fact.resultLayout != explicitResult) {
          continue;
        }
        VMILayoutOpRelation relation{op,
                                     {operandPort(0, fact.sourceLayout),
                                      resultPort(0, fact.resultLayout)},
                                     false,
                                     fact.intrinsicRearrangementCost};
        appendReachableUniqueRelation(relations, std::move(relation), supports);
      }
    }
    return relations.empty()
               ? FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     failure())
               : FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     std::move(relations));
  }

  // vinterpret_cast is normalized to bitcast by unified-to-legacy, so both
  // spellings must expose the same Support relation before that pass runs.
  if (isa<VMIBitcastOp, VMIVinterpretCastOp>(op)) {
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !resultType) {
      return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidates;
    if (sourceType.getLayoutAttr()) {
      candidates.push_back(sourceType.getLayoutAttr());
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
      if (layout && !llvm::is_contained(candidates, layout)) {
        candidates.push_back(layout);
      }
    }
    if (candidates.empty()) {
      candidates.push_back(VMILayoutAttr::getContiguous(op->getContext()));
    }
    for (VMILayoutAttr layout : candidates) {
      auto facts = supports.getBitcastLayoutFactsForLayout(
          sourceType, resultType, VMICastLayoutPort::Source, layout);
      if (failed(facts)) {
        continue;
      }
      for (const VMIBitcastLayoutFact &fact : *facts) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, fact.sourceLayout),
                                 resultPort(0, fact.resultLayout)},
                                false},
            supports);
      }
    }
    return relations;
  }

  if (auto load = dyn_cast<VMILoadOp>(op)) {
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    FailureOr<SmallVector<VMILoadLayoutFact, mlir::pto::kValue4>> facts =
        supports.getLoadLayoutFacts(resultType);
    if (failed(facts)) {
      return failure();
    }
    for (const VMILoadLayoutFact &fact : *facts) {
      if (VMILayoutAttr explicitLayout = resultType.getLayoutAttr();
          explicitLayout && explicitLayout != fact.resultLayout) {
        continue;
      }
      appendReachableUniqueRelation(
          relations,
          VMILayoutOpRelation{op,
                              {resultPort(0, fact.resultLayout)},
                              /*directProducer=*/true},
          supports);
    }
    if (relations.empty()) {
      return failure();
    }
    return relations;
  }

  auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
  if (isa<VMIStrideLoadOp>(op)) {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1) {
      return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(4, contiguous), resultPort(0, contiguous)}, true},
        supports);
  }
  if (isa<VMIGatherOp>(op)) {
    if (op->getNumOperands() != 4 || op->getNumResults() != 1) {
      return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(1, contiguous), operandPort(2, contiguous),
          operandPort(3, contiguous), resultPort(0, contiguous)},
         true},
        supports);
  }
  if (isa<VMIStrideStoreOp>(op)) {
    if (op->getNumOperands() < 1 || op->getNumResults() != 0) {
      return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, contiguous), operandPort(5, contiguous)}, false},
        supports);
  }
  if (isa<VMIScatterOp>(op)) {
    if (op->getNumOperands() != 4 || op->getNumResults() != 0) {
      return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(0, contiguous), operandPort(2, contiguous),
          operandPort(3, contiguous)},
         false},
        supports);
  }
  if (isa<VMICompressStoreOp>(op)) {
    if (op->getNumOperands() != 4 || op->getNumResults() != 0) {
      return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, contiguous), operandPort(3, contiguous)}, false},
        supports);
  }
  if (isa<VMIActivePrefixIndexOp>(op)) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, contiguous), resultPort(0, contiguous)}, false},
        supports);
  }
  if (isa<VMICompressOp>(op)) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(0, contiguous), operandPort(1, contiguous),
          resultPort(0, contiguous)},
         false},
        supports);
  }
  if (isa<VMIExpandLoadOp>(op)) {
    if (op->getNumOperands() != 4 || op->getNumResults() != 1) {
      return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(2, contiguous), operandPort(3, contiguous),
          resultPort(0, contiguous)},
         true},
        supports);
  }
  if (isa<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp, VMIReduceMinFOp,
          VMIReduceMaxIOp, VMIReduceMinIOp>(op)) {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1) {
      return failure();
    }
    auto fact = supports.getReduceLayoutFactForLayouts(
        VMIVRegType::get(
            op->getContext(),
            cast<VMIVRegType>(op->getOperand(0).getType()).getElementCount(),
            cast<VMIVRegType>(op->getOperand(0).getType()).getElementType(),
            contiguous),
        VMIMaskType::get(
            op->getContext(),
            cast<VMIMaskType>(op->getOperand(1).getType()).getElementCount(),
            cast<VMIMaskType>(op->getOperand(1).getType()).getGranularity(),
            contiguous),
        VMIVRegType::get(
            op->getContext(),
            cast<VMIVRegType>(op->getResult(0).getType()).getElementCount(),
            cast<VMIVRegType>(op->getResult(0).getType()).getElementType(),
            contiguous));
    if (failed(fact)) {
      return failure();
    }
    VMILayoutOpRelation relation{op,
                                 {operandPort(0, fact->sourceLayout),
                                  operandPort(1, fact->maskLayout),
                                  resultPort(0, fact->resultLayout)},
                                 false};
    return makeReachableRelation(std::move(relation), supports);
  }

  if (auto interleave = dyn_cast<VMIVintlvOp>(op)) {
    auto lhs = dyn_cast<VMIVRegType>(interleave.getLhs().getType());
    auto rhs = dyn_cast<VMIVRegType>(interleave.getRhs().getType());
    auto mask = dyn_cast<VMIMaskType>(interleave.getMask().getType());
    auto low = dyn_cast<VMIVRegType>(interleave.getLow().getType());
    auto high = dyn_cast<VMIVRegType>(interleave.getHigh().getType());
    if (!lhs || !rhs || !mask || !low || !high) {
      return failure();
    }
    if (lhs.getLayoutAttr() && rhs.getLayoutAttr() && mask.getLayoutAttr() &&
        low.getLayoutAttr() && high.getLayoutAttr()) {
      auto fact = supports.getVintlvLayoutFactForLayouts(lhs, rhs, mask, low,
                                                         high);
      if (failed(fact)) {
        return failure();
      }
      return makeReachableRelation(
          {op,
           {operandPort(0, fact->lhsLayout), operandPort(1, fact->rhsLayout),
            operandPort(2, fact->maskLayout), resultPort(0, fact->lowLayout),
            resultPort(1, fact->highLayout)},
           true},
          supports);
    }
    auto facts = supports.getVintlvLayoutFacts(lhs);
    if (failed(facts)) {
      return failure();
    }
    auto preferred = supports.getPreferredVintlvLayoutFact(lhs);
    for (const VMIInterleaveLayoutFact &fact : *facts) {
      auto relation = VMILayoutOpRelation{
          op,
          {operandPort(0, fact.lhsLayout), operandPort(1, fact.rhsLayout),
           operandPort(2, fact.maskLayout), resultPort(0, fact.lowLayout),
           resultPort(1, fact.highLayout)},
          true,
          /*intrinsicRearrangementCost=*/0,
          /*preferencePenalty=*/
          succeeded(preferred) && haveSameInterleaveLayouts(fact, *preferred)
              ? 0U
              : 1U};
      auto reachable = makeReachableRelation(std::move(relation), supports);
      if (succeeded(reachable)) {
        for (VMILayoutOpRelation &reachableRelation : *reachable) {
          appendUniqueRelation(relations, std::move(reachableRelation));
        }
      }
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, 4>>(
                                   failure())
                             : FailureOr<SmallVector<VMILayoutOpRelation, 4>>(
                                   std::move(relations));
  }

  if (auto interleave = dyn_cast<VMIVdintlvOp>(op)) {
    auto lhs = dyn_cast<VMIVRegType>(interleave.getLhs().getType());
    auto rhs = dyn_cast<VMIVRegType>(interleave.getRhs().getType());
    auto mask = dyn_cast<VMIMaskType>(interleave.getMask().getType());
    auto low = dyn_cast<VMIVRegType>(interleave.getLow().getType());
    auto high = dyn_cast<VMIVRegType>(interleave.getHigh().getType());
    if (!lhs || !rhs || !mask || !low || !high) {
      return failure();
    }
    if (lhs.getLayoutAttr() && rhs.getLayoutAttr() && mask.getLayoutAttr() &&
        low.getLayoutAttr() && high.getLayoutAttr()) {
      auto fact = supports.getVdintlvLayoutFactForLayouts(lhs, rhs, mask, low,
                                                          high);
      if (failed(fact)) {
        return failure();
      }
      return makeReachableRelation(
          {op,
           {operandPort(0, fact->lhsLayout), operandPort(1, fact->rhsLayout),
            operandPort(2, fact->maskLayout), resultPort(0, fact->lowLayout),
            resultPort(1, fact->highLayout)},
           true},
          supports);
    }
    auto facts = supports.getVdintlvLayoutFacts(lhs);
    if (failed(facts)) {
      return failure();
    }
    auto preferred = supports.getPreferredVdintlvLayoutFact(lhs);
    for (const VMIInterleaveLayoutFact &fact : *facts) {
      auto relation = VMILayoutOpRelation{
          op,
          {operandPort(0, fact.lhsLayout), operandPort(1, fact.rhsLayout),
           operandPort(2, fact.maskLayout), resultPort(0, fact.lowLayout),
           resultPort(1, fact.highLayout)},
          true,
          /*intrinsicRearrangementCost=*/0,
          /*preferencePenalty=*/
          succeeded(preferred) && haveSameInterleaveLayouts(fact, *preferred)
              ? 0U
              : 1U};
      auto reachable = makeReachableRelation(std::move(relation), supports);
      if (succeeded(reachable)) {
        for (VMILayoutOpRelation &reachableRelation : *reachable) {
          appendUniqueRelation(relations, std::move(reachableRelation));
        }
      }
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, 4>>(
                                   failure())
                             : FailureOr<SmallVector<VMILayoutOpRelation, 4>>(
                                   std::move(relations));
  }

  if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
    auto source = dyn_cast<VMIVRegType>(split.getSource().getType());
    if (!source || (split.getNumResults() != 2 && split.getNumResults() != 4)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = VMILayoutAttr::getDeinterleaved(
        op->getContext(), split.getNumResults());
    for (OpResult result : split.getResults()) {
      if (auto explicitLayout =
              cast<VMIVRegType>(result.getType()).getLayoutAttr();
          explicitLayout &&
          explicitLayout != VMILayoutAttr::getContiguous(op->getContext())) {
        return failure();
      }
    }
    SmallVector<VMILayoutPortAssignment, mlir::pto::kValue4> ports;
    ports.push_back(operandPort(0, sourceLayout));
    VMILayoutAttr resultLayout = VMILayoutAttr::getContiguous(op->getContext());
    for (unsigned i = 0; i < split.getNumResults(); ++i) {
      ports.push_back(resultPort(i, resultLayout));
    }
    VMILayoutOpRelation relation{op, std::move(ports), true};
    return makeReachableRelation(std::move(relation), supports);
  }

  if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
    auto result = dyn_cast<VMIVRegType>(merge.getResult().getType());
    unsigned channels = merge.getInputs().size();
    if (!result || (channels != 2 && channels != 4)) {
      return failure();
    }
    VMILayoutAttr inputLayout = VMILayoutAttr::getContiguous(op->getContext());
    VMILayoutAttr resultLayout =
        VMILayoutAttr::getDeinterleaved(op->getContext(), channels);
    SmallVector<VMILayoutPortAssignment, mlir::pto::kValue4> ports;
    for (unsigned i = 0; i < channels; ++i) {
      ports.push_back(operandPort(i, inputLayout));
    }
    ports.push_back(resultPort(0, resultLayout));
    VMILayoutOpRelation relation{op, std::move(ports), true};
    return makeReachableRelation(std::move(relation), supports);
  }

  if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
    auto source = dyn_cast<VMIVRegType>(shuffle.getSource().getType());
    auto result = dyn_cast<VMIVRegType>(shuffle.getResult().getType());
    if (!source || !result) {
      return failure();
    }
    VMILayoutAttr layout = source.getLayoutAttr();
    if (!layout) {
      layout = VMILayoutAttr::getContiguous(op->getContext());
    }
    if (auto explicitResult = result.getLayoutAttr();
        explicitResult && explicitResult != layout) {
      return failure();
    }
    VMILayoutOpRelation relation{
        op, {operandPort(0, layout), resultPort(0, layout)}, false};
    return makeReachableRelation(std::move(relation), supports);
  }

  if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    auto maskType = dyn_cast<VMIMaskType>(load.getMask().getType());
    auto passthruType = dyn_cast<VMIVRegType>(load.getPassthru().getType());
    if (!resultType || !maskType || !passthruType) {
      return failure();
    }
    SmallVector<VMIMaskedLoadLayoutFact, mlir::pto::kValue4> facts;
    auto addFact = [&](VMILayoutAttr result, VMILayoutAttr mask,
                       VMILayoutAttr passthru) {
      VMIVRegType r = VMIVRegType::get(resultType.getContext(),
                                       resultType.getElementCount(),
                                       resultType.getElementType(), result);
      VMIVRegType p = VMIVRegType::get(passthruType.getContext(),
                                       passthruType.getElementCount(),
                                       passthruType.getElementType(), passthru);
      VMIMaskType m =
          VMIMaskType::get(maskType.getContext(), maskType.getElementCount(),
                           maskType.getGranularity(), mask);
      if (succeeded(supports.getMaskedLoadLayoutFact(r, m, p))) {
        facts.push_back({result, mask, passthru});
      }
    };
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    VMILayoutAttr maskLayout = maskType.getLayoutAttr();
    VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
    if (resultLayout && maskLayout && passthruLayout) {
      addFact(resultLayout, maskLayout, passthruLayout);
    }
    if (facts.empty()) {
      VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
      addFact(resultLayout ? resultLayout : contiguous,
              maskLayout ? maskLayout : contiguous,
              passthruLayout ? passthruLayout : contiguous);
    }
    for (const auto &fact : facts) {
      appendReachableUniqueRelation(
          relations,
          VMILayoutOpRelation{op,
                              {operandPort(2, fact.maskLayout),
                               operandPort(3, fact.passthruLayout),
                               resultPort(0, fact.resultLayout)},
                              true},
          supports);
    }
    return relations.empty()
               ? FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     failure())
               : FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     std::move(relations));
  }

  if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidates;
    if (VMILayoutAttr layout = resultType.getLayoutAttr()) {
      candidates.push_back(layout);
    }
    int64_t groups = load.getNumGroupsAttr().getInt();
    for (int64_t slots : {int64_t(1), int64_t(8), groups}) {
      VMILayoutAttr candidate =
          VMILayoutAttr::getGroupSlots(op->getContext(), groups, slots);
      if (!llvm::is_contained(candidates, candidate)) {
        candidates.push_back(candidate);
      }
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
      if (layout && !llvm::is_contained(candidates, layout)) {
        candidates.push_back(layout);
      }
    }
    for (VMILayoutAttr layout : candidates) {
      if (resultType.getLayoutAttr() && resultType.getLayoutAttr() != layout) {
        continue;
      }
      auto typed = VMIVRegType::get(resultType.getContext(),
                                    resultType.getElementCount(),
                                    resultType.getElementType(), layout);
      if (succeeded(supports.getGroupSlotLoadLayoutFact(
              typed, load.getSourceGroupStride(),
              load.getNumGroupsAttr().getInt()))) {
        // slots=8 reads a whole group per block load, while slots=1 needs one
        // scalar broadcast load per slot.  Both are legal for a unit stride,
        // so prefer the fewer-part form and keep slots=1 available for the
        // strides that slots=8 rejects.
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {resultPort(0, layout)},
                                /*directProducer=*/true,
                                /*intrinsicRearrangementCost=*/0,
                                /*preferencePenalty=*/layout.getSlots() == 1
                                    ? 1U
                                    : 0U},
            supports);
      }
    }
    return relations;
  }

  if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
    auto lowType = dyn_cast<VMIVRegType>(load.getLow().getType());
    auto highType = dyn_cast<VMIVRegType>(load.getHigh().getType());
    if (!lowType || !highType) {
      return failure();
    }
    VMILayoutAttr lowLayout = lowType.getLayoutAttr();
    VMILayoutAttr highLayout = highType.getLayoutAttr();
    if (lowLayout && highLayout) {
      auto typedLow =
          VMIVRegType::get(lowType.getContext(), lowType.getElementCount(),
                           lowType.getElementType(), lowLayout);
      auto typedHigh =
          VMIVRegType::get(highType.getContext(), highType.getElementCount(),
                           highType.getElementType(), highLayout);
      auto fact =
          supports.getDeinterleaveLoadLayoutFactForLayouts(typedLow, typedHigh);
      if (succeeded(fact)) {
        return makeReachableRelation(
            {op,
             {resultPort(0, fact->lowLayout), resultPort(1, fact->highLayout)},
             true},
            supports);
      }
      return failure();
    }
    if (highLayout) {
      auto facts = supports.getDeinterleaveLoadLayoutFactsForLayout(
          lowType, VMIDeinterleaveLoadLayoutPort::High, highLayout);
      if (succeeded(facts)) {
        for (const auto &fact : *facts) {
          if (!fact.highLayout || fact.highLayout != highLayout) {
            continue;
          }
          appendReachableUniqueRelation(
              relations,
              VMILayoutOpRelation{op,
                                  {resultPort(0, fact.lowLayout),
                                   resultPort(1, fact.highLayout)},
                                  true},
              supports);
        }
      }
      if (!relations.empty()) {
        return relations;
      }
      return failure();
    }
    auto preferred = supports.getPreferredDeinterleaveLoadLayoutFact(lowType);
    if (failed(preferred)) {
      return failure();
    }
    return makeReachableRelation({op,
                                  {resultPort(0, preferred->lowLayout),
                                   resultPort(1, preferred->highLayout)},
                                  true},
                                 supports);
  }

  if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidates;
    if (VMILayoutAttr explicitLayout = resultType.getLayoutAttr()) {
      candidates.push_back(explicitLayout);
    }
    for (VMILayoutAttr candidate : polymorphicLayouts) {
      if (candidate && !llvm::is_contained(candidates, candidate)) {
        candidates.push_back(candidate);
      }
    }
    if (candidates.empty()) {
      auto preferred = supports.getGroupLoadLayoutFact(load);
      if (succeeded(preferred)) {
        candidates.push_back(preferred->resultLayout);
      }
    }
    for (VMILayoutAttr layout : candidates) {
      if (resultType.getLayoutAttr() && resultType.getLayoutAttr() != layout) {
        continue;
      }
      auto typed = VMIVRegType::get(resultType.getContext(),
                                    resultType.getElementCount(),
                                    resultType.getElementType(), layout);
      if (succeeded(supports.getGroupLoadLayoutFact(
              typed, load.getRowStride(), load.getNumGroupsAttr().getInt()))) {
        appendReachableUniqueRelation(
            relations, VMILayoutOpRelation{op, {resultPort(0, layout)}, true},
            supports);
      }
    }
    return relations;
  }

  if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    if (!valueType) {
      return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidateLayouts;
    if (VMILayoutAttr explicitLayout = valueType.getLayoutAttr()) {
      candidateLayouts.push_back(explicitLayout);
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
      if (layout && !llvm::is_contained(candidateLayouts, layout)) {
        candidateLayouts.push_back(layout);
      }
    }
    if (candidateLayouts.empty()) {
      auto preferred =
          supports.getPreferredGroupStoreLayoutFact(store, valueType);
      if (succeeded(preferred)) {
        candidateLayouts.push_back(preferred->valueLayout);
      }
    }
    if (candidateLayouts.empty()) {
      candidateLayouts.push_back(
          VMILayoutAttr::getContiguous(op->getContext()));
    }
    for (VMILayoutAttr layout : candidateLayouts) {
      auto facts =
          supports.getGroupStoreLayoutFactsForLayout(store, valueType, layout);
      if (failed(facts)) {
        continue;
      }
      for (const VMIGroupStoreLayoutFact &fact : *facts) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{
                op,
                {operandPort(store.getValueMutable().getOperandNumber(),
                             fact.valueLayout)},
                false},
            supports);
      }
    }
    return relations.empty()
               ? FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     failure())
               : FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     std::move(relations));
  }

  if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
    auto lowType = dyn_cast<VMIVRegType>(store.getLow().getType());
    auto highType = dyn_cast<VMIVRegType>(store.getHigh().getType());
    if (!lowType || !highType) {
      return failure();
    }
    VMILayoutAttr contiguous =
        VMILayoutAttr::getContiguous(op->getContext());
    VMIVRegType queryLow = lowType.getLayoutAttr()
                               ? lowType
                               : VMIVRegType::get(lowType.getContext(),
                                                  lowType.getElementCount(),
                                                  lowType.getElementType(),
                                                  contiguous);
    VMIVRegType queryHigh = highType.getLayoutAttr()
                                ? highType
                                : VMIVRegType::get(highType.getContext(),
                                                   highType.getElementCount(),
                                                   highType.getElementType(),
                                                   contiguous);
    auto support = VMILayoutSupport().getInterleaveStoreSupport(queryLow,
                                                                queryHigh);
    if (failed(support)) {
      return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(0, support->lowLayout),
          operandPort(1, support->highLayout)},
         false},
        supports);
  }

  if (auto store = dyn_cast<VMIStoreOp>(op)) {
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    FailureOr<SmallVector<VMIStoreLayoutFact, mlir::pto::kValue4>> facts =
        supports.getStoreLayoutFacts(valueType);
    if (failed(facts)) {
      return failure();
    }
    auto preferred = supports.getPreferredStoreLayoutFact(valueType);
    unsigned valueIndex = store.getValueMutable().getOperandNumber();
    for (const VMIStoreLayoutFact &fact : *facts) {
      appendReachableUniqueRelation(
          relations,
          VMILayoutOpRelation{op,
                              {operandPort(valueIndex, fact.valueLayout)},
                              /*directProducer=*/false,
                              /*intrinsicRearrangementCost=*/0,
                              /*preferencePenalty=*/
                              succeeded(preferred) &&
                                      fact.valueLayout != preferred->valueLayout
                                  ? 1U
                                  : 0U},
          supports);
    }
    if (relations.empty()) {
      return failure();
    }
    return relations;
  }

  if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    auto maskType = dyn_cast<VMIMaskType>(store.getMask().getType());
    if (!valueType || !maskType) {
      return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidates;
    if (valueType.getLayoutAttr()) {
      candidates.push_back(valueType.getLayoutAttr());
    }
    if (maskType.getLayoutAttr() &&
        !llvm::is_contained(candidates, maskType.getLayoutAttr())) {
      candidates.push_back(maskType.getLayoutAttr());
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
      if (layout && !llvm::is_contained(candidates, layout)) {
        candidates.push_back(layout);
      }
    }
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (!llvm::is_contained(candidates, contiguous)) {
      candidates.push_back(contiguous);
    }
    for (VMILayoutAttr layout : candidates) {
      auto concreteValue =
          VMIVRegType::get(valueType.getContext(), valueType.getElementCount(),
                           valueType.getElementType(), layout);
      auto concreteMask =
          VMIMaskType::get(maskType.getContext(), maskType.getElementCount(),
                           maskType.getGranularity(), layout);
      auto fact =
          supports.getMaskedStoreLayoutFact(concreteValue, concreteMask);
      if (succeeded(fact)) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, fact->valueLayout),
                                 operandPort(3, fact->maskLayout)},
                                false},
            supports);
      }
    }
    return relations;
  }

  if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    FailureOr<SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4>>
        facts = supports.getGroupBroadcastLoadLayoutFacts(load);
    if (failed(facts)) {
      return failure();
    }
    VMILayoutAttr explicitLayout =
        getExplicitLayout(load.getResult().getType());
    FailureOr<VMIGroupBroadcastLoadDirectFact> directFact =
        supports.getGroupBroadcastLoadDirectFact(load);
    for (const VMIGroupBroadcastLoadLayoutFact &fact : *facts) {
      if (explicitLayout && explicitLayout != fact.resultLayout) {
        continue;
      }
      // An E2B-capable load has one physical producer layout: E2B produces
      // the split packet form (d2/d4), not a dense contiguous VMI value.
      // Consumers that require contiguous data must pay for an explicit
      // d4->contiguous materialization at their use edge.
      if (succeeded(directFact) &&
          directFact->kind == VMIGroupBroadcastLoadDirectKind::E2B &&
          fact.resultLayout != directFact->layout.resultLayout) {
        continue;
      }
      auto candidateType = VMIVRegType::get(
          resultType.getContext(), resultType.getElementCount(),
          resultType.getElementType(), fact.resultLayout);
      bool directProducer = succeeded(supports.getGroupBroadcastLoadDirectFact(
          candidateType, load.getSource().getType(),
          load.getSourceGroupStride(), load.getNumGroupsAttr().getInt()));
      appendReachableUniqueRelation(
          relations,
          VMILayoutOpRelation{
              op, {resultPort(0, fact.resultLayout)}, directProducer},
          supports);
    }
    return relations;
  }

  if (isa<VMIVdhistOp, VMIVchistOp>(op)) {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
      return failure();
    }
    auto acc = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto source = dyn_cast<VMIVRegType>(op->getOperand(1).getType());
    auto mask = dyn_cast<VMIMaskType>(op->getOperand(2).getType());
    auto result = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!acc || !source || !mask || !result) {
      return failure();
    }
    bool assigned = acc.getLayoutAttr() && source.getLayoutAttr() &&
                    mask.getLayoutAttr() && result.getLayoutAttr();
    FailureOr<VMIHistogramLayoutFact> fact =
        isa<VMIVdhistOp>(op)
            ? (assigned ? supports.getVdhistLayoutFact(cast<VMIVdhistOp>(op))
                        : supports.getPreferredVdhistLayoutFact(
                              cast<VMIVdhistOp>(op)))
            : (assigned ? supports.getVchistLayoutFact(cast<VMIVchistOp>(op))
                        : supports.getPreferredVchistLayoutFact(
                              cast<VMIVchistOp>(op)));
    if (failed(fact)) {
      return failure();
    }
    appendReachableUniqueRelation(
        relations,
        VMILayoutOpRelation{op,
                            {operandPort(0, fact->accLayout),
                             operandPort(1, fact->sourceLayout),
                             operandPort(2, fact->maskLayout),
                             resultPort(0, fact->resultLayout)},
                            false},
        supports);
    return relations;
  }

  if (auto vselr = dyn_cast<VMIVselrOp>(op)) {
    auto sourceType = cast<VMIVRegType>(vselr.getSource().getType());
    auto indexType = cast<VMIVRegType>(vselr.getIndex().getType());
    auto resultType = cast<VMIVRegType>(vselr.getResult().getType());
    bool assigned = sourceType.getLayoutAttr() && indexType.getLayoutAttr() &&
                    resultType.getLayoutAttr();
    auto fact = assigned ? supports.getVselrLayoutFact(vselr, reason)
                         : supports.getPreferredVselrLayoutFact(vselr, reason);
    if (failed(fact)) {
      return failure();
    }
    VMILayoutOpRelation relation{op,
                                 {operandPort(0, fact->sourceLayout),
                                  operandPort(1, fact->indexLayout),
                                  resultPort(0, fact->resultLayout)},
                                 false};
    return makeReachableRelation(std::move(relation), supports);
  }

  // Mask producers do not impose a physical rearrangement themselves, but
  // their result layout is still a solver variable when downstream masked
  // operations participate in the component.
  if (isa<VMICreateMaskOp, VMICreateGroupMaskOp, VMIConstantMaskOp, VMIPsetOp,
          VMIPgeOp, VMIPltOp>(op)) {
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> maskLayouts;
    auto maskType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
    VMILayoutAttr explicitLayout =
        maskType ? maskType.getLayoutAttr() : VMILayoutAttr{};
    if (explicitLayout) {
      maskLayouts.push_back(explicitLayout);
    } else {
      maskLayouts.push_back(VMILayoutAttr::getContiguous(op->getContext()));
      for (VMILayoutAttr layout : polymorphicLayouts) {
        if (!llvm::is_contained(maskLayouts, layout)) {
          maskLayouts.push_back(layout);
        }
      }
    }
    for (VMILayoutAttr layout : maskLayouts) {
      if (!layout) {
        continue;
      }
      if (isa<VMICreateGroupMaskOp>(op) &&
          failed(supports.getGeneratedMaskLayoutFact(op, layout))) {
        continue;
      }
      appendReachableUniqueRelation(
          relations, VMILayoutOpRelation{op, {resultPort(0, layout)}, false},
          supports);
    }
    return relations;
  }

  if (isa<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMIGroupIotaOp, VMIVciOp,
          VMIVbrcOp>(op)) {
    if (auto groupIota = dyn_cast<VMIGroupIotaOp>(op)) {
      auto resultType = dyn_cast<VMIVRegType>(groupIota.getResult().getType());
      if (!resultType) {
        return failure();
      }
      auto facts = supports.getGroupIotaLayoutFacts(resultType);
      if (failed(facts)) {
        return failure();
      }
      for (const VMIGroupIotaLayoutFact &fact : *facts) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op, {resultPort(0, fact.resultLayout)}, true},
            supports);
      }
      return relations;
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> producerLayouts;
    if (auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
        resultType && resultType.getLayoutAttr()) {
      producerLayouts.push_back(resultType.getLayoutAttr());
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
      if (layout && !llvm::is_contained(producerLayouts, layout)) {
        producerLayouts.push_back(layout);
      }
    }
    if (auto broadcast = dyn_cast<VMIBroadcastOp>(op)) {
      auto sourceType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
      if (sourceType) {
        for (Operation *user : broadcast.getResult().getUsers()) {
          auto groupBroadcast = dyn_cast<VMIGroupBroadcastOp>(user);
          if (!groupBroadcast ||
              groupBroadcast.getSource() != broadcast.getResult()) {
            continue;
          }
          auto resultType =
              dyn_cast<VMIVRegType>(groupBroadcast.getResult().getType());
          if (!resultType) {
            continue;
          }
          int64_t groups = groupBroadcast.getNumGroupsAttr().getInt();
          VMILayoutAttr slots = VMILayoutAttr::getGroupSlots(
              op->getContext(), groups, mlir::pto::kValue8);
          auto facts = VMILayoutSupport().getGroupBroadcastLayoutFactsForLayout(
              sourceType, resultType, groups,
              VMIGroupBroadcastLayoutPort::Source, slots);
          if (succeeded(facts) && !facts->empty() &&
              !llvm::is_contained(producerLayouts,
                                  facts->front().sourceLayout)) {
            producerLayouts.push_back(facts->front().sourceLayout);
          }
        }
      }
    }
    if (producerLayouts.empty()) {
      producerLayouts.push_back(VMILayoutAttr::getContiguous(op->getContext()));
    }
    for (VMILayoutAttr layout : producerLayouts) {
      // Grouped vci lowering is a contiguous group_iota producer.  A
      // non-contiguous result requires a separate ensure_layout materializer;
      // it is therefore not a direct producer relation and must be represented
      // by an explicit structural op before entering this planner.
      if (auto vci = dyn_cast<VMIVciOp>(op)) {
        auto group = vci.getGroupAttr();
        if (group && group.getInt() > 1 && !layout.isContiguous()) {
          continue;
        }
      }
      appendReachableUniqueRelation(
          relations, VMILayoutOpRelation{op, {resultPort(0, layout)}, true},
          supports);
    }
    return relations;
  }

  if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
    auto sourceType = dyn_cast<VMIVRegType>(broadcast.getSource().getType());
    auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }
    int64_t groups = broadcast.getNumGroupsAttr().getInt();
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidateLayouts;
    candidateLayouts.append(polymorphicLayouts.begin(),
                            polymorphicLayouts.end());
    if (candidateLayouts.empty()) {
      candidateLayouts.push_back(
          VMILayoutAttr::getContiguous(op->getContext()));
    }
    if (auto preferred = supports.getGroupBroadcastLayoutFactForLayouts(
            sourceType, resultType, groups);
        succeeded(preferred) &&
        !llvm::is_contained(candidateLayouts, preferred->sourceLayout)) {
      candidateLayouts.push_back(preferred->sourceLayout);
    }
    VMILayoutAttr slots = VMILayoutAttr::getGroupSlots(op->getContext(), groups,
                                                       mlir::pto::kValue8);
    if (!llvm::is_contained(candidateLayouts, slots)) {
      candidateLayouts.push_back(slots);
    }
    for (VMILayoutAttr layout : candidateLayouts) {
      if (!layout) {
        continue;
      }
      if (layout.isContiguous()) {
        auto lanes = getDataLanesPerPart(resultType.getElementType());
        if (succeeded(lanes) && resultType.getElementCount() < *lanes) {
          continue;
        }
      }
      auto facts = supports.getGroupBroadcastLayoutFactsForLayout(
          sourceType, resultType, groups, VMIGroupBroadcastLayoutPort::Source,
          layout);
      if (failed(facts)) {
        continue;
      }
      for (const VMIGroupBroadcastLayoutFact &fact : *facts) {
        if (resultType.getLayoutAttr() &&
            resultType.getLayoutAttr() != fact.resultLayout) {
          continue;
        }
        if (auto lanes = getDataLanesPerPart(resultType.getElementType());
            succeeded(lanes) && resultType.getElementCount() < *lanes &&
            fact.resultLayout.isContiguous() &&
            fact.resultLayout.getLaneStride() == 1) {
          continue;
        }
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, fact.sourceLayout),
                                 resultPort(0, fact.resultLayout)},
                                false},
            supports);
      }
    }
    if (relations.empty()) {
      auto preferred = supports.getGroupBroadcastLayoutFactForLayouts(
          sourceType, resultType, groups);
      bool partialResult = false;
      if (auto lanes = getDataLanesPerPart(resultType.getElementType());
          succeeded(lanes)) {
        partialResult = resultType.getElementCount() < *lanes;
      }
      if (succeeded(preferred) &&
          (!partialResult || !preferred->resultLayout.isContiguous() ||
           preferred->resultLayout.getLaneStride() != 1) &&
          (!resultType.getLayoutAttr() ||
           resultType.getLayoutAttr() == preferred->resultLayout)) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, preferred->sourceLayout),
                                 resultPort(0, preferred->resultLayout)},
                                false},
            supports);
      }
    }
    return relations.empty()
               ? FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     failure())
               : FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     std::move(relations));
  }

  if (isa<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp,
          VMIGroupReduceAddIOp, VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(
          op)) {
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto maskType = dyn_cast<VMIMaskType>(op->getOperand(1).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !maskType || !resultType || !op->hasAttr("num_groups")) {
      return failure();
    }
    int64_t groups = op->getAttrOfType<IntegerAttr>("num_groups").getInt();
    for (VMILayoutAttr layout :
         getGroupReduceQueryLayouts(op->getContext(), polymorphicLayouts)) {
      if (!layout) {
        continue;
      }
      auto facts = supports.getGroupReduceLayoutFactsForLayout(
          sourceType, groups, VMIGroupReduceLayoutPort::Source, layout);
      if (failed(facts)) {
        continue;
      }
      for (const VMIGroupReduceLayoutFact &fact : *facts) {
        if ((sourceType.getLayoutAttr() &&
             sourceType.getLayoutAttr() != fact.sourceLayout) ||
            (maskType.getLayoutAttr() &&
             maskType.getLayoutAttr() != fact.maskLayout) ||
            (resultType.getLayoutAttr() &&
             resultType.getLayoutAttr() != fact.resultLayout)) {
          continue;
        }
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, fact.sourceLayout),
                                 operandPort(1, fact.maskLayout),
                                 resultPort(0, fact.resultLayout)},
                                false},
            supports);
      }
    }
    if (relations.empty()) {
      auto preferred =
          supports.getPreferredGroupReduceLayoutFact(sourceType, groups);
      if (succeeded(preferred)) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, preferred->sourceLayout),
                                 operandPort(1, preferred->maskLayout),
                                 resultPort(0, preferred->resultLayout)},
                                false},
            supports);
      }
    }
    return relations.empty()
               ? FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     failure())
               : FailureOr<
                     SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(
                     std::move(relations));
  }

  if (isa<VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>(op)) {
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto maskType = dyn_cast<VMIMaskType>(op->getOperand(1).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !maskType || !resultType) {
      return failure();
    }
    auto group = op->getAttrOfType<IntegerAttr>("group");
    if (!group) {
      VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
      auto fact = supports.getReduceLayoutFactForLayouts(
          VMIVRegType::get(op->getContext(), sourceType.getElementCount(),
                           sourceType.getElementType(), contiguous),
          VMIMaskType::get(op->getContext(), maskType.getElementCount(),
                           maskType.getGranularity(), contiguous),
          VMIVRegType::get(op->getContext(), resultType.getElementCount(),
                           resultType.getElementType(), contiguous));
      if (failed(fact)) {
        return failure();
      }
      return makeReachableRelation({op,
                                    {operandPort(0, fact->sourceLayout),
                                     operandPort(1, fact->maskLayout),
                                     resultPort(0, fact->resultLayout)},
                                    false},
                                   supports);
    }
    for (VMILayoutAttr layout :
         getGroupReduceQueryLayouts(op->getContext(), polymorphicLayouts)) {
      if (!layout) {
        continue;
      }
      auto facts = supports.getGroupReduceLayoutFactsForLayout(
          sourceType, group.getInt(), VMIGroupReduceLayoutPort::Source, layout);
      if (failed(facts)) {
        continue;
      }
      for (const VMIGroupReduceLayoutFact &fact : *facts) {
        if ((sourceType.getLayoutAttr() &&
             sourceType.getLayoutAttr() != fact.sourceLayout) ||
            (maskType.getLayoutAttr() &&
             maskType.getLayoutAttr() != fact.maskLayout) ||
            (resultType.getLayoutAttr() &&
             resultType.getLayoutAttr() != fact.resultLayout)) {
          continue;
        }
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, fact.sourceLayout),
                                 operandPort(1, fact.maskLayout),
                                 resultPort(0, fact.resultLayout)},
                                false},
            supports);
      }
    }
    if (polymorphicLayouts.empty() && relations.empty()) {
      auto preferred = supports.getPreferredGroupReduceLayoutFact(
          sourceType, group.getInt());
      if (succeeded(preferred)) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{op,
                                {operandPort(0, preferred->sourceLayout),
                                 operandPort(1, preferred->maskLayout),
                                 resultPort(0, preferred->resultLayout)},
                                false},
            supports);
      }
    }
    return relations;
  }

  if (auto vexpdif = dyn_cast<VMIVexpdifOp>(op)) {
    auto sourceType = dyn_cast<VMIVRegType>(vexpdif.getX().getType());
    auto resultType = dyn_cast<VMIVRegType>(vexpdif.getResult().getType());
    if (!sourceType || !resultType) {
      return failure();
    }

    // pto.vmi.vexpdif reads one physical source register at a time, so the
    // only legal plans are the rows of the vexpdif layout table.  Enumerating
    // any other combination - notably a lane-strided source, whose shared mask
    // layout has to change predicate granularity as well - selects a plan that
    // the conversion to VPTO cannot realize.  The widening form is therefore
    // expressed by the table, not by the generic cast relation table.
    auto preferred = supports.getPreferredVexpdifLayoutFact(vexpdif);
    for (VMILayoutAttr layout :
         getVexpdifQueryLayouts(op, polymorphicLayouts)) {
      for (VMIVexpdifLayoutPort port :
           {VMIVexpdifLayoutPort::Source, VMIVexpdifLayoutPort::Result}) {
        auto facts =
            supports.getVexpdifLayoutFactsForLayout(vexpdif, port, layout);
        if (failed(facts)) {
          continue;
        }
        for (const VMIVexpdifLayoutFact &fact : *facts) {
          uint64_t preferencePenalty =
              succeeded(preferred) &&
                      (fact.sourceLayout != preferred->sourceLayout ||
                       fact.resultLayout != preferred->resultLayout)
                  ? 1
                  : 0;
          // x, max, and the mask always share one layout: the VPTO lowering
          // reads the predicate with the data's lane order.
          appendReachableUniqueRelation(
              relations,
              VMILayoutOpRelation{op,
                                  {operandPort(0, fact.sourceLayout),
                                   operandPort(1, fact.sourceLayout),
                                   operandPort(2, fact.sourceLayout),
                                   resultPort(0, fact.resultLayout)},
                                  /*directProducer=*/false,
                                  /*intrinsicRearrangementCost=*/0,
                                  preferencePenalty},
              supports);
        }
      }
    }
    bool explainMissingRelation = relations.empty() && reason != nullptr;
    if (explainMissingRelation) {
      // Surface the table requirement instead of a generic "no legal relation"
      // when the shape itself has no supported combination.
      (void)supports.getVexpdifLayoutFactsForLayout(
          vexpdif, VMIVexpdifLayoutPort::Source,
          VMILayoutAttr::getContiguous(op->getContext()), reason);
    }
    return relations.empty()
               ? FailureOr<SmallVector<VMILayoutOpRelation,
                                       mlir::pto::kValue4>>(failure())
               : FailureOr<SmallVector<VMILayoutOpRelation,
                                       mlir::pto::kValue4>>(
                     std::move(relations));
  }

  if (isVMISameLayoutOp(op)) {
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> sameLayoutCandidates;
    appendSameLayoutCandidates(op, polymorphicLayouts, sameLayoutCandidates);
    appendSameLayoutRelations(op, sameLayoutCandidates, supports, relations,
                              reason);
    if (!relations.empty()) {
      return relations;
    }
  }

  return failure();
}

LogicalResult mlir::pto::commitVMILayoutPlan(const VMILayoutPlan &plan,
                                             VMILayoutPropagator &propagator) {
  Operation *anchor = plan.selectedRelations.empty()
                          ? nullptr
                          : plan.selectedRelations.begin()->first;
  auto failCommit = [&](StringRef message) {
    if (anchor) {
      anchor->emitError() << kVMIDiagLayoutContractPrefix << message;
    }
    return failure();
  };
  // Commit primary layouts before use layouts so an intentional use-edge
  // conversion is represented as a conflict, rather than as a producer seed.
  SmallVector<VMILayoutAttr, mlir::pto::kValue4> layouts =
      collectPlanLayouts(plan);
  if (failed(verifySelectedRelations(plan, layouts))) {
    return failCommit("selected cost plan no longer matches registered layout "
                      "relations");
  }
  if (failed(requestPlan(plan, propagator))) {
    return failCommit("failed to install validated cost-plan recipe");
  }
  if (!planWasCommitted(plan, propagator)) {
    return failCommit("installed cost-plan recipe does not match the selected "
                      "value/use layouts");
  }
  propagator.endExactRequests();
  return success();
}

FailureOr<VMILayoutPlannerResult>
mlir::pto::selectCostedVMILayoutPlans(Operation *scope,
                                      const VMILayoutPlannerOptions &options) {
  if (!scope || options.maxFrontierEntriesPerComponent == 0 ||
      options.maxTransitionsPerComponent == 0) {
    return failure();
  }
  VMILayoutPlannerResult result;
  DenseSet<Value> seenValues;
  SmallVector<Value, mlir::pto::kValue16> seeds;
  scope->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (isLayoutValue(operand) && !llvm::is_contained(seeds, operand)) {
        seeds.push_back(operand);
      }
    }
    for (Value resultValue : op->getResults()) {
      if (isLayoutValue(resultValue)) {
        seeds.push_back(resultValue);
      }
    }
  });
  for (Value seed : seeds) {
    if (seenValues.contains(seed)) {
      continue;
    }
    SmallVector<Operation *, mlir::pto::kValue16> component;
    SmallPtrSet<Operation *, mlir::pto::kValue16> seenOps;
    if (!collectComponent(seed, component, seenOps, seenValues)) {
      continue;
    }
    if (llvm::none_of(component, [](Operation *op) {
          return !isVMILayoutStructuralOp(op);
        })) {
      continue;
    }
    auto plan = solveComponent(component, options);
    if (failed(plan)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "layout planner: component has no complete legal plan\n");
      return failure();
    }
    result.plans.push_back(std::move(*plan));
  }
  return result;
}
