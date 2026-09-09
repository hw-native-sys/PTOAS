// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutConflictSolver.cpp - VMI layout cost solver --------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutConflictSolver.h"

#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "vmi-layout-conflict-solver"

namespace {

static Value getRelationPortValue(const VMILayoutOpRelation &relation,
                                  const VMILayoutPortAssignment &port) {
  if (!relation.op) {
    return {};
  }
  if (port.kind == VMILayoutPortKind::Operand &&
      port.index < relation.op->getNumOperands())
    return relation.op->getOperand(port.index);
  if (port.kind == VMILayoutPortKind::Result &&
      port.index < relation.op->getNumResults())
    return relation.op->getResult(port.index);
  return {};
}

} // namespace

namespace mlir::pto {

VMILayoutRelationConstraintState::VMILayoutRelationConstraintState(
    ArrayRef<VMILayoutEqualityConstraint> equalities) {
  for (const VMILayoutEqualityConstraint &equality : equalities) {
    if (failed(unite(equality.source, equality.destination))) {
      parent.clear();
      assignedLayouts.clear();
      return;
    }
  }
}

Value VMILayoutRelationConstraintState::find(Value value) {
  auto it = parent.find(value);
  if (it == parent.end() || it->second == value) {
    return value;
  }
  Value root = find(it->second);
  it->second = root;
  return root;
}

LogicalResult VMILayoutRelationConstraintState::unite(Value lhs, Value rhs) {
  if (!lhs || !rhs) {
    return failure();
  }
  parent.try_emplace(lhs, lhs);
  parent.try_emplace(rhs, rhs);
  Value left = find(lhs);
  Value right = find(rhs);
  if (left != right) {
    auto leftLayout = assignedLayouts.find(left);
    auto rightLayout = assignedLayouts.find(right);
    if (leftLayout != assignedLayouts.end() &&
        rightLayout != assignedLayouts.end() &&
        leftLayout->second != rightLayout->second) {
      return failure();
    }
    parent[right] = left;
    if (leftLayout == assignedLayouts.end() &&
        rightLayout != assignedLayouts.end()) {
      assignedLayouts[left] = rightLayout->second;
    }
    assignedLayouts.erase(right);
  }
  return success();
}

LogicalResult VMILayoutRelationConstraintState::assign(Value value,
                                                        VMILayoutAttr layout) {
  if (!value || !layout || !parent.count(value)) {
    return success();
  }
  Value root = find(value);
  auto [it, inserted] = assignedLayouts.try_emplace(root, layout);
  return inserted || it->second == layout ? success() : failure();
}

LogicalResult VMILayoutRelationConstraintState::accept(
    const VMILayoutOpRelation &relation, const VMILayoutPlan &plan) {
  (void)plan;
  for (const VMILayoutEqualityConstraint &equality : relation.equalities) {
    if (failed(unite(equality.source, equality.destination))) {
      return failure();
    }
  }
  for (const VMILayoutPortAssignment &port : relation.ports) {
    // Operand layout belongs to a use and may differ from its source value.
    if (port.kind == VMILayoutPortKind::Result &&
        failed(assign(getRelationPortValue(relation, port), port.layout))) {
      return failure();
    }
  }
  for (const VMILayoutRelationEndpoint &endpoint : relation.endpoints) {
    if (!endpoint.use && failed(assign(endpoint.value, endpoint.layout))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
VMILayoutRelationConstraintState::materialize(VMILayoutPlan &plan) const {
  auto findRoot = [&](Value value) {
    Value current = value;
    while (true) {
      auto it = parent.find(current);
      if (it == parent.end() || it->second == current)
        break;
      current = it->second;
    }
    return current;
  };
  for (const auto &[value, ignored] : parent) {
    (void)ignored;
    Value root = findRoot(value);
    auto layout = assignedLayouts.find(root);
    if (layout == assignedLayouts.end()) {
      continue;
    }
    auto [it, inserted] = plan.valueLayouts.try_emplace(value, layout->second);
    if (!inserted && it->second != layout->second) {
      return failure();
    }
  }
  return success();
}

FailureOr<std::string> VMILayoutRelationConstraintState::fingerprint() const {
  SmallVector<size_t, mlir::pto::kValue8> entries;
  entries.reserve(assignedLayouts.size());
  for (const auto &[value, layout] : assignedLayouts) {
    if (!value || !layout) {
      return failure();
    }
    entries.push_back(static_cast<size_t>(llvm::hash_combine(value, layout)));
  }
  llvm::sort(entries);
  std::string result;
  llvm::raw_string_ostream stream(result);
  for (size_t entry : entries) {
    stream << entry << ';';
  }
  return stream.str();
}

} // namespace mlir::pto

namespace {

struct SolverState {
  SmallVector<llvm::SmallBitVector, mlir::pto::kValue8> domains;
};

struct FrontierEntry {
  VMILayoutPlan plan;
  VMILayoutPhysicalState physicalState;
  VMILayoutRelationConstraintState constraints;
  uint64_t preferencePenalty = 0;
};

static VMILayoutAttr getPortLayout(const VMILayoutOpRelation &relation,
                                   VMILayoutPortKind kind, unsigned index) {
  for (const VMILayoutPortAssignment &port : relation.ports) {
    if (port.kind == kind && port.index == index) {
      return port.layout;
    }
  }
  return {};
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

static bool isLayoutBearingType(Type type) {
  return isa<VMIVRegType, VMIMaskType>(type);
}

static bool canMaterialize(Value value, VMILayoutAttr source,
                           VMILayoutAttr target) {
  if (!value || !source || !target) {
    return false;
  }
  if (source == target) {
    return true;
  }
  VMILayoutSupport support;
  if (auto type = dyn_cast<VMIVRegType>(value.getType())) {
    auto sourceType =
        VMIVRegType::get(type.getContext(), type.getElementCount(),
                         type.getElementType(), source);
    auto targetType =
        VMIVRegType::get(type.getContext(), type.getElementCount(),
                         type.getElementType(), target);
    return succeeded(support.getEnsureLayoutFact(sourceType, targetType));
  }
  if (auto type = dyn_cast<VMIMaskType>(value.getType())) {
    auto sourceType =
        VMIMaskType::get(type.getContext(), type.getElementCount(),
                         type.getGranularity(), source);
    auto targetType =
        VMIMaskType::get(type.getContext(), type.getElementCount(),
                         type.getGranularity(), target);
    return succeeded(support.getEnsureMaskLayoutFact(sourceType, targetType));
  }
  return false;
}

static VMILayoutAttr getBoundaryLayout(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !isLayoutBearingType(value.getType())) {
    return {};
  }
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front()) {
    return {};
  }
  if (VMILayoutAttr explicitLayout = getExplicitLayout(value.getType())) {
    return explicitLayout;
  }
  return VMILayoutAttr::getContiguous(value.getContext());
}

static bool scopeCostDominates(const VMILayoutScopeCost &lhs,
                               const VMILayoutScopeCost &rhs) {
  return lhs.total < rhs.total;
}

static bool scopeCostsEqual(const VMILayoutScopeCost &lhs,
                            const VMILayoutScopeCost &rhs) {
  return lhs.total == rhs.total;
}

static unsigned countMaterializations(const VMILayoutPlan &plan) {
  unsigned count = 0;
  for (const auto &[operand, useLayout] : plan.useLayouts) {
    if (!operand) {
      continue;
    }
    auto valueLayout = plan.valueLayouts.find(operand->get());
    if (valueLayout != plan.valueLayouts.end() &&
        valueLayout->second != useLayout) {
      ++count;
    }
  }
  return count;
}

static bool tieBreaksStrictlyBefore(const FrontierEntry &lhs,
                                      const FrontierEntry &rhs) {
  unsigned lhsMaterializations = countMaterializations(lhs.plan);
  unsigned rhsMaterializations = countMaterializations(rhs.plan);
  if (lhsMaterializations != rhsMaterializations) {
    return lhsMaterializations < rhsMaterializations;
  }
  return lhs.preferencePenalty < rhs.preferencePenalty;
}

static bool tieBreaksLessOrEqual(const FrontierEntry &lhs,
                                 const FrontierEntry &rhs) {
  unsigned lhsMaterializations = countMaterializations(lhs.plan);
  unsigned rhsMaterializations = countMaterializations(rhs.plan);
  if (lhsMaterializations != rhsMaterializations) {
    return lhsMaterializations < rhsMaterializations;
  }
  return lhs.preferencePenalty <= rhs.preferencePenalty;
}

class FrontierConflictSolver {
public:
  FrontierConflictSolver(ArrayRef<VMILayoutSolverOp> ops,
                         const VMILayoutConflictSolverOptions &options)
      : ops(ops), options(options) {
    for (auto [index, op] : llvm::enumerate(ops)) {
      opIndices[op.op] = index;
    }
    initialConstraints =
        VMILayoutRelationConstraintState(options.equalityConstraints);
  }

  FailureOr<VMILayoutPlan> solve() {
    if (!hasValidInput()) {
      return failure();
    }
    SolverState state;
    for (const VMILayoutSolverOp &op : ops) {
      state.domains.emplace_back(op.relations.size(), true);
    }
    if (!propagate(state)) {
      return failure();
    }
    auto order = getTopologicalOrder();
    if (failed(order)) {
      return failure();
    }
    SmallVector<FrontierEntry, mlir::pto::kValue8> frontier;
    frontier.emplace_back();
    frontier.back().constraints = initialConstraints;
    for (auto [position, opIndex] : llvm::enumerate(*order)) {
      SmallVector<Operation *, mlir::pto::kValue16> remainingOps;
      for (unsigned remaining : llvm::drop_begin(*order, position + 1)) {
        remainingOps.push_back(ops[remaining].op);
      }
      auto next =
          extend(frontier, opIndex, state.domains[opIndex], remainingOps);
      if (failed(next)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "layout frontier failed at " << ops[opIndex].op->getName()
                   << " transitions=" << transitions << "\n");
        return failure();
      }
      frontier = std::move(*next);
      LLVM_DEBUG(llvm::dbgs()
                 << "layout frontier " << ops[opIndex].op->getName()
                 << " entries=" << frontier.size()
                 << " transitions=" << transitions << "\n");
    }
    if (frontier.empty()) {
      return failure();
    }
    const FrontierEntry *best = &frontier.front();
    for (const FrontierEntry &candidate : llvm::drop_begin(frontier)) {
      VMILayoutScopeCost candidateCost =
          getVMILayoutPhysicalCost(candidate.physicalState);
      VMILayoutScopeCost bestCost =
          getVMILayoutPhysicalCost(best->physicalState);
      if (scopeCostDominates(candidateCost, bestCost)) {
        best = &candidate;
        continue;
      }
      if (scopeCostsEqual(candidateCost, bestCost) &&
          tieBreaksStrictlyBefore(candidate, *best)) {
        best = &candidate;
        continue;
      }
      if (!scopeCostsEqual(candidateCost, bestCost) &&
          !scopeCostDominates(bestCost, candidateCost)) {
        return failure();
      }
    }
    auto fullCost = evaluateFullPlanCost(best->plan);
    if (failed(fullCost) ||
        !scopeCostsEqual(*fullCost,
                         getVMILayoutPhysicalCost(best->physicalState))) {
      return failure();
    }
    VMILayoutPlan result = best->plan;
    if (failed(best->constraints.materialize(result)))
      return failure();
    return result;
  }

private:
  bool hasValidInput() const {
    if (opIndices.size() != ops.size()) {
      return false;
    }
    for (const VMILayoutSolverOp &solverOp : ops) {
      if (!solverOp.op || solverOp.relations.empty()) {
        return false;
      }
      for (const VMILayoutOpRelation &relation : solverOp.relations) {
        if (relation.op != solverOp.op ||
            (relation.ports.empty() && relation.endpoints.empty())) {
          return false;
        }
        if (relation.ports.empty() && !relation.endpoints.empty())
          continue;
        llvm::SmallBitVector operandPorts(solverOp.op->getNumOperands());
        llvm::SmallBitVector resultPorts(solverOp.op->getNumResults());
        for (const VMILayoutPortAssignment &port : relation.ports) {
          llvm::SmallBitVector *ports = nullptr;
          switch (port.kind) {
          case VMILayoutPortKind::Operand:
            ports = &operandPorts;
            break;
          case VMILayoutPortKind::Result:
            ports = &resultPorts;
            break;
          default:
            return false;
          }
          if (!port.layout || port.index >= ports->size() ||
              (*ports)[port.index]) {
            return false;
          }
          ports->set(port.index);
        }
        // Every layout-bearing SSA port must participate in the relation.
        // Allowing an omitted port would turn an operation relation into an
        // under-constrained candidate and could make the solver select a plan
        // that the applier cannot validate.
        for (unsigned index = 0; index < solverOp.op->getNumOperands();
             ++index) {
          if (isLayoutBearingType(solverOp.op->getOperand(index).getType()) &&
              !operandPorts[index]) {
            return false;
          }
        }
        for (unsigned index = 0; index < solverOp.op->getNumResults();
             ++index) {
          if (isLayoutBearingType(solverOp.op->getResult(index).getType()) &&
              !resultPorts[index]) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool relationHasSupport(unsigned opIndex, unsigned relationIndex,
                          const SolverState &state) const {
    const VMILayoutOpRelation &relation = ops[opIndex].relations[relationIndex];
    Operation *op = relation.op;
    for (const VMILayoutPortAssignment &port : relation.ports) {
      if (port.kind == VMILayoutPortKind::Result) {
        Value result = op->getResult(port.index);
        // An explicitly typed SSA result is a fixed layout constraint.  The
        // use edge may still request another layout (and pay an ensure
        // conversion), but the defining operation itself may not select a
        // different result layout.
        if (VMILayoutAttr explicitLayout = getExplicitLayout(result.getType());
            explicitLayout && explicitLayout != port.layout) {
          return false;
        }
        for (OpOperand &use : result.getUses()) {
          auto consumerIt = opIndices.find(use.getOwner());
          if (consumerIt == opIndices.end()) {
            continue;
          }
          unsigned consumerIndex = consumerIt->second;
          bool supported = false;
          for (int candidate = state.domains[consumerIndex].find_first();
               candidate >= 0;
               candidate = state.domains[consumerIndex].find_next(candidate)) {
            VMILayoutAttr targetLayout = getPortLayout(
                ops[consumerIndex].relations[candidate],
                VMILayoutPortKind::Operand, use.getOperandNumber());
            supported |= canMaterialize(result, port.layout, targetLayout);
          }
          if (!supported) {
            return false;
          }
        }
        continue;
      }
      Value source = op->getOperand(port.index);
      Operation *producer = source.getDefiningOp();
      auto producerIt = opIndices.find(producer);
      if (producerIt == opIndices.end()) {
        VMILayoutAttr explicitLayout = getBoundaryLayout(source);
        if (auto type = dyn_cast<VMIVRegType>(source.getType())) {
          if (type.getLayoutAttr()) {
            explicitLayout = type.getLayoutAttr();
          }
        } else if (auto type = dyn_cast<VMIMaskType>(source.getType())) {
          if (type.getLayoutAttr()) {
            explicitLayout = type.getLayoutAttr();
          }
        }
        if (explicitLayout &&
            !canMaterialize(source, explicitLayout, port.layout)) {
          return false;
        }
        continue;
      }
      unsigned producerIndex = producerIt->second;
      bool supported = false;
      for (int candidate = state.domains[producerIndex].find_first();
           candidate >= 0;
           candidate = state.domains[producerIndex].find_next(candidate)) {
        VMILayoutAttr sourceLayout = getPortLayout(
            ops[producerIndex].relations[candidate], VMILayoutPortKind::Result,
            cast<OpResult>(source).getResultNumber());
        supported |= canMaterialize(source, sourceLayout, port.layout);
      }
      if (!supported) {
        return false;
      }
    }
    return true;
  }

  bool propagate(SolverState &state) const {
    bool changed;
    do {
      changed = false;
      for (auto [opIndex, domain] : llvm::enumerate(state.domains)) {
        for (int relation = domain.find_first(); relation >= 0;
             relation = domain.find_next(relation)) {
          if (relationHasSupport(opIndex, relation, state)) {
            continue;
          }
          domain.reset(relation);
          changed = true;
        }
        if (domain.none()) {
          return false;
        }
      }
    } while (changed);
    return true;
  }

  FailureOr<SmallVector<unsigned, mlir::pto::kValue16>>
  getTopologicalOrder() const {
    SmallVector<unsigned, mlir::pto::kValue16> indegrees(ops.size(), 0);
    SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue16>
        successors(ops.size());
    for (auto [consumerIndex, solverOp] : llvm::enumerate(ops)) {
      SmallPtrSet<Operation *, mlir::pto::kValue4> producers;
      for (Value operand : solverOp.op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        auto producerIt = opIndices.find(producer);
        if (producerIt == opIndices.end() ||
            !producers.insert(producer).second) {
          continue;
        }
        ++indegrees[consumerIndex];
        successors[producerIt->second].push_back(consumerIndex);
      }
    }
    SmallVector<unsigned, mlir::pto::kValue16> ready;
    for (auto [index, indegree] : llvm::enumerate(indegrees)) {
      if (indegree == 0) {
        ready.push_back(index);
      }
    }
    SmallVector<unsigned, mlir::pto::kValue16> order;
    while (!ready.empty()) {
      unsigned current = ready.front();
      ready.erase(ready.begin());
      order.push_back(current);
      for (unsigned successor : successors[current]) {
        if (--indegrees[successor] == 0) {
          ready.push_back(successor);
        }
      }
    }
    if (order.size() != ops.size()) {
      return failure();
    }
    return order;
  }

  static LogicalResult addRelationToPlan(const VMILayoutOpRelation &relation,
                                         unsigned relationIndex,
                                         VMILayoutPlan &plan) {
    if (!relation.op) {
      return failure();
    }
    for (const VMILayoutPortAssignment &port : relation.ports) {
      if (!port.layout) {
        return failure();
      }
      if (port.kind == VMILayoutPortKind::Operand) {
        if (port.index >= relation.op->getNumOperands()) {
          return failure();
        }
        plan.useLayouts[&relation.op->getOpOperand(port.index)] = port.layout;
        Value source = relation.op->getOperand(port.index);
        if (VMILayoutAttr boundary = getBoundaryLayout(source)) {
          plan.valueLayouts.try_emplace(source, boundary);
        }
      } else {
        if (port.index >= relation.op->getNumResults()) {
          return failure();
        }
        plan.valueLayouts[relation.op->getResult(port.index)] = port.layout;
      }
    }
    for (const VMILayoutRelationEndpoint &endpoint : relation.endpoints) {
      if (!endpoint.value || !endpoint.layout) {
        return failure();
      }
      if (endpoint.use) {
        plan.useLayouts[endpoint.use] = endpoint.layout;
      } else {
        plan.valueLayouts[endpoint.value] = endpoint.layout;
      }
    }
    plan.selectedRelations[relation.op] = relationIndex;
    return success();
  }

  static void insertPareto(
      FrontierEntry candidate, StringRef key,
      llvm::StringMap<SmallVector<FrontierEntry, mlir::pto::kValue2>> &groups) {
    auto &entries = groups[key];
    VMILayoutScopeCost candidateCost =
        getVMILayoutPhysicalCost(candidate.physicalState);
    for (const FrontierEntry &entry : entries) {
      VMILayoutScopeCost cost = getVMILayoutPhysicalCost(entry.physicalState);
      if (scopeCostDominates(cost, candidateCost) ||
          (scopeCostsEqual(cost, candidateCost) &&
           tieBreaksLessOrEqual(entry, candidate))) {
        return;
      }
    }
    llvm::erase_if(entries, [&](const FrontierEntry &entry) {
      VMILayoutScopeCost cost = getVMILayoutPhysicalCost(entry.physicalState);
      return scopeCostDominates(candidateCost, cost) ||
             (scopeCostsEqual(candidateCost, cost) &&
              tieBreaksLessOrEqual(candidate, entry));
    });
    entries.push_back(std::move(candidate));
  }

  FailureOr<SmallVector<FrontierEntry, mlir::pto::kValue8>>
  extend(ArrayRef<FrontierEntry> frontier, unsigned opIndex,
         const llvm::SmallBitVector &domain,
         ArrayRef<Operation *> remainingOps) {
    llvm::StringMap<SmallVector<FrontierEntry, mlir::pto::kValue2>> groups;
    for (const FrontierEntry &entry : frontier) {
      for (int relationIndex = domain.find_first(); relationIndex >= 0;
           relationIndex = domain.find_next(relationIndex)) {
        if (transitions >= options.maxTransitions) {
          return failure();
        }
        ++transitions;
        const VMILayoutOpRelation &relation =
            ops[opIndex].relations[relationIndex];
        FrontierEntry candidate = entry;
        if (relation.preferencePenalty >
            std::numeric_limits<uint64_t>::max() -
                candidate.preferencePenalty) {
          return failure();
        }
        candidate.preferencePenalty += relation.preferencePenalty;
        if (failed(
                addRelationToPlan(relation, relationIndex, candidate.plan))) {
          return failure();
        }
        if (failed(candidate.constraints.accept(relation, candidate.plan)))
          continue;
        auto physicalState = appendVMILayoutPhysicalRelation(
            entry.physicalState, relation, candidate.plan);
        if (failed(physicalState)) {
          continue;
        }
        candidate.physicalState = std::move(*physicalState);
        auto key =
            getVMILayoutContinuationKey(candidate.physicalState, remainingOps);
        if (failed(key)) {
          continue;
        }
        auto constraintKey = candidate.constraints.fingerprint();
        if (failed(constraintKey))
          return failure();
        std::string combinedKey = *key + "|constraints=" + *constraintKey;
        insertPareto(std::move(candidate), combinedKey, groups);
      }
    }
    SmallVector<FrontierEntry, mlir::pto::kValue8> result;
    for (auto &group : groups) {
      result.append(std::make_move_iterator(group.second.begin()),
                    std::make_move_iterator(group.second.end()));
      if (result.size() > options.maxFrontierEntries) {
        return failure();
      }
    }
    return result;
  }

  FailureOr<VMILayoutScopeCost>
  evaluateFullPlanCost(const VMILayoutPlan &plan) const {
    SmallVector<VMILayoutOpRelation, mlir::pto::kValue16> relations;
    relations.reserve(ops.size());
    for (const VMILayoutSolverOp &solverOp : ops) {
      auto selected = plan.selectedRelations.find(solverOp.op);
      if (selected == plan.selectedRelations.end() ||
          selected->second >= solverOp.relations.size()) {
        return failure();
      }
      relations.push_back(solverOp.relations[selected->second]);
    }
    return evaluateVMILayoutPlanCost(relations, plan);
  }

  ArrayRef<VMILayoutSolverOp> ops;
  const VMILayoutConflictSolverOptions &options;
  DenseMap<Operation *, unsigned> opIndices;
  VMILayoutRelationConstraintState initialConstraints;
  unsigned transitions = 0;
};

} // namespace

FailureOr<VMILayoutPlan> mlir::pto::solveVMILayoutConflictComponent(
    ArrayRef<VMILayoutSolverOp> ops,
    const VMILayoutConflictSolverOptions &options) {
  if (ops.empty() || options.maxFrontierEntries == 0 ||
      options.maxTransitions == 0) {
    return failure();
  }
  return FrontierConflictSolver(ops, options).solve();
}
