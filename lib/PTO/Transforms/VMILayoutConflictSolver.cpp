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

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <optional>
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
    ArrayRef<VMILayoutEqualityConstraint> equalities,
    ArrayRef<VMILayoutFixedAssignment> fixedAssignments) {
  for (const VMILayoutEqualityConstraint &equality : equalities) {
    if (failed(unite(equality.source, equality.destination))) {
      valid = false;
      return;
    }
  }
  for (const VMILayoutFixedAssignment &fixed : fixedAssignments) {
    if (!fixed.value || !fixed.layout) {
      valid = false;
      return;
    }
    parent.try_emplace(fixed.value, fixed.value);
    if (failed(assign(fixed.value, fixed.layout))) {
      valid = false;
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

LogicalResult
VMILayoutRelationConstraintState::accept(const VMILayoutOpRelation &relation,
                                         const VMILayoutPlan &plan) {
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
  uint64_t materializationPositionScore = 0;
  DenseMap<unsigned, unsigned> selectedDecisionGroups;
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

static VMILayoutAttr getUseLayout(const VMILayoutOpRelation &relation,
                                  OpOperand &use) {
  if (VMILayoutAttr layout = getPortLayout(relation, VMILayoutPortKind::Operand,
                                           use.getOperandNumber())) {
    return layout;
  }
  for (const VMILayoutRelationEndpoint &endpoint : relation.endpoints) {
    if (endpoint.use == &use) {
      return endpoint.layout;
    }
  }
  return {};
}

static VMILayoutAttr getResultLayout(const VMILayoutOpRelation &relation,
                                     OpResult result) {
  if (VMILayoutAttr layout = getPortLayout(relation, VMILayoutPortKind::Result,
                                           result.getResultNumber())) {
    return layout;
  }
  for (const VMILayoutRelationEndpoint &endpoint : relation.endpoints) {
    if (!endpoint.use && endpoint.value == result) {
      return endpoint.layout;
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

// A memory access and an element-type cast realize a layout change through
// their own lowering: the access carries a lane-strided layout in its
// distribution mode, and the cast rewrites every element anyway.  A lane-stride
// or a conversion that only such an operation consumes therefore costs nothing
// extra.  A layout cast is *not* in this set: an explicit ensure_layout has to
// move the lanes it converts.
static bool absorbsLayoutInOwnLowering(Operation *op) {
  if (!op) {
    return false;
  }
  // isVMILayoutCastOp covers the element-type conversions (extend, truncate,
  // float <-> integer).
  if (isVMILayoutCastOp(op)) {
    return true;
  }
  return isa<VMIBitcastOp, VMIVinterpretCastOp, VMICvtOp, VMILoadOp,
             VMIMaskedLoadOp, VMIDeinterleaveLoadOp, VMIGroupLoadOp,
             VMIGroupSlotLoadOp, VMIGroupBroadcastLoadOp, VMIStrideLoadOp,
             VMIExpandLoadOp, VMIvLoadOp, VMIStoreOp, VMIMaskedStoreOp,
             VMIInterleaveStoreOp, VMIGroupStoreOp, VMIStrideStoreOp,
             VMICompressStoreOp, VMIvStoreOp>(op);
}

[[maybe_unused]] static unsigned
countMaterializations(const VMILayoutPlan &plan) {
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

// Fewer lane-strided values is better, but only where the compute chain has to
// carry them: a lane-strided layout packs a wider carrier into fewer registers
// and is free on a load, a store, or an element-type cast, yet a plain compute
// op that receives it has to keep splitting it back out.  Counting the layouts
// that a memory access or a type cast already absorbs would contradict the
// per-operation layout preferences instead of refining them.
static unsigned
countLaneStrideLayouts(const VMILayoutPlan &plan) {
  unsigned count = 0;
  for (const auto &[value, layout] : plan.valueLayouts) {
    if (!layout || !layout.hasLaneStride()) {
      continue;
    }
    if (absorbsLayoutInOwnLowering(value.getDefiningOp())) {
      continue;
    }
    bool consumedByCompute = false;
    for (OpOperand &use : value.getUses()) {
      if (!absorbsLayoutInOwnLowering(use.getOwner())) {
        consumedByCompute = true;
        break;
      }
    }
    if (consumedByCompute) {
      ++count;
    }
  }
  return count;
}

// Data merges are the plan's Merge and Interleave actions: both rejoin data
// that a layout spread over separate registers.  Fewer of them is better.
// Layout decision hierarchy.  Two candidates are only compared here when
// their layout-conversion cost is equal (scopeCostsEqual), so no tie-break key
// can override a cheaper layout conversion.  Within one cost the keys apply in
// this order:
//
//   1. total (layer 1) - PhysicalGraph::getCost(): one unit per Interleave,
//      Deinterleave, Unpack, Pack or Merge action the lowering has to
//      materialize.  A rearrangement that a memory access already realizes
//      through its distribution mode, or that a type cast realizes through its
//      part selection, is marked absorbed and is not charged at all.
//   2. data couplings - Merge + Interleave actions.  A "vor" merge and a
//      "vintlv" interleave both join data that a layout spread over separate
//      registers, so either counts once: two plans that only pick a different
//      joining operation tie here and fall through.
//   3. compute lane-strides - lane-strided values a compute operation has to
//      carry.  Loads, stores and type casts absorb their own lane-strides, so
//      only the ones a compute op must split back out are counted.
//   4. lane-stride weight - the sum of the lane strides of every lane-strided
//      value, absorbed ones included.  A wider stride packs more elements into
//      one carrier and splits the stream more finely, so lane_stride = 4 costs
//      more than = 2; fewer wins.
//   5. preference penalty - the per-operation layout preferences.  They rank
//      below both lane-stride keys because the tables only describe what a
//      single operation prefers, not what the whole plan pays.
//   6. materialization position - sum of the topological positions of the
//      plan's conversions; materializing later is better.
//
// Layer 2: data couplings.  A Merge ("vor") and an Interleave ("vintlv") are
// both points where data that a layout spread over separate registers is
// joined again, so either counts once.  Two plans that only pick a different
// joining op therefore tie here and fall through to the next layer.
static int64_t countDataCouplings(const VMILayoutPhysicalState &state) {
  VMILayoutScopeCost cost = getVMILayoutPhysicalCost(state);
  return cost.merges + cost.interleaves;
}

// Layer 4: the overall lane-stride count, i.e. every lane-strided value in the
// plan including the ones a memory access or a type cast absorbs.  The
// per-operation preference tables rank above it; it only separates plans those
// tables cannot distinguish, and then fewer lane-strided values win.
static unsigned countLaneStrideWeight(const VMILayoutPlan &plan) {
  unsigned weight = 0;
  for (const auto &[value, layout] : plan.valueLayouts) {
    (void)value;
    if (layout && layout.hasLaneStride()) {
      weight += layout.getLaneStride();
    }
  }
  return weight;
}

static bool tieBreaksStrictlyBefore(const FrontierEntry &lhs,
                                    const FrontierEntry &rhs) {
  int64_t lhsCouplings = countDataCouplings(lhs.physicalState);
  int64_t rhsCouplings = countDataCouplings(rhs.physicalState);
  if (lhsCouplings != rhsCouplings) {
    return lhsCouplings < rhsCouplings;
  }
  unsigned lhsLaneStrides = countLaneStrideLayouts(lhs.plan);
  unsigned rhsLaneStrides = countLaneStrideLayouts(rhs.plan);
  if (lhsLaneStrides != rhsLaneStrides) {
    return lhsLaneStrides < rhsLaneStrides;
  }
  unsigned lhsAllLaneStrides = countLaneStrideWeight(lhs.plan);
  unsigned rhsAllLaneStrides = countLaneStrideWeight(rhs.plan);
  if (lhsAllLaneStrides != rhsAllLaneStrides) {
    return lhsAllLaneStrides < rhsAllLaneStrides;
  }
  if (lhs.preferencePenalty != rhs.preferencePenalty) {
    return lhs.preferencePenalty < rhs.preferencePenalty;
  }
  return lhs.materializationPositionScore > rhs.materializationPositionScore;
}

static bool tieBreaksLessOrEqual(const FrontierEntry &lhs,
                                 const FrontierEntry &rhs) {
  int64_t lhsCouplings = countDataCouplings(lhs.physicalState);
  int64_t rhsCouplings = countDataCouplings(rhs.physicalState);
  if (lhsCouplings != rhsCouplings) {
    return lhsCouplings < rhsCouplings;
  }
  unsigned lhsLaneStrides = countLaneStrideLayouts(lhs.plan);
  unsigned rhsLaneStrides = countLaneStrideLayouts(rhs.plan);
  if (lhsLaneStrides != rhsLaneStrides) {
    return lhsLaneStrides < rhsLaneStrides;
  }
  unsigned lhsAllLaneStrides = countLaneStrideWeight(lhs.plan);
  unsigned rhsAllLaneStrides = countLaneStrideWeight(rhs.plan);
  if (lhsAllLaneStrides != rhsAllLaneStrides) {
    return lhsAllLaneStrides < rhsAllLaneStrides;
  }
  if (lhs.preferencePenalty != rhs.preferencePenalty) {
    return lhs.preferencePenalty < rhs.preferencePenalty;
  }
  return lhs.materializationPositionScore >= rhs.materializationPositionScore;
}

class FrontierConflictSolver {
public:
  FrontierConflictSolver(ArrayRef<VMILayoutSolverOp> ops,
                         const VMILayoutConflictSolverOptions &options)
      : ops(ops), options(options) {
    for (auto [index, op] : llvm::enumerate(ops)) {
      opIndices[op.op] = index;
    }
    opDecisionGroups.assign(ops.size(), -1);
    SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue8>
        adjacency(ops.size());
    for (auto [consumerIndex, solverOp] : llvm::enumerate(ops)) {
      for (Value operand : solverOp.op->getOperands()) {
        auto producerIt = opIndices.find(operand.getDefiningOp());
        if (producerIt == opIndices.end() ||
            producerIt->second == consumerIndex) {
          continue;
        }
        adjacency[consumerIndex].push_back(producerIt->second);
        adjacency[producerIt->second].push_back(consumerIndex);
      }
    }
    for (auto &neighbors : adjacency) {
      llvm::sort(neighbors);
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                      neighbors.end());
    }
    SmallVector<unsigned, mlir::pto::kValue8> components(ops.size(), 0);
    llvm::SmallBitVector visited(ops.size());
    unsigned nextComponent = 0;
    for (unsigned root = 0; root < ops.size(); ++root) {
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
    for (auto [groupIndex, group] : llvm::enumerate(options.decisionGroups)) {
      if (group.opIndices.size() != group.memberRelationIndices.size() ||
          group.opIndices.empty()) {
        continue;
      }
      unsigned firstOp = group.opIndices.front();
      if (firstOp >= ops.size()) {
        continue;
      }
      bool valid = true;
      for (unsigned opIndex : group.opIndices) {
        if (opIndex >= ops.size() ||
            components[opIndex] != components[firstOp]) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        continue;
      }
      for (auto [member, opIndex] : llvm::enumerate(group.opIndices)) {
        if (opIndex >= ops.size() || opDecisionGroups[opIndex] != -1 ||
            group.memberRelationIndices[member].empty() ||
            group.memberRelationIndices[member].size() !=
                group.memberRelationIndices.front().size() ||
            llvm::any_of(group.memberRelationIndices[member],
                         [&](unsigned relationIndex) {
                           return relationIndex >=
                                  ops[opIndex].relations.size();
                         })) {
          valid = false;
          break;
        }
        opDecisionGroups[opIndex] = groupIndex;
      }
      if (!valid) {
        for (unsigned opIndex : group.opIndices) {
          if (opIndex < ops.size() &&
              opDecisionGroups[opIndex] == static_cast<int>(groupIndex)) {
            opDecisionGroups[opIndex] = -1;
          }
        }
      }
    }
    initialConstraints = VMILayoutRelationConstraintState(
        options.equalityConstraints, options.fixedAssignments);
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
    topologicalPositions.assign(ops.size(), 0);
    for (auto [position, opIndex] : llvm::enumerate(*order)) {
      topologicalPositions[opIndex] = position;
    }
    // The pilot merges pending group choices only to find a legal upper bound.
    // The exact pass retains them and prunes only partial plans already above
    // it.
    transitions = 0;
    costUpperBound.reset();
    auto pilot = solveOrdered(state, *order, false);
    if (succeeded(pilot)) {
      auto pilotCost = evaluateFullPlanCost(*pilot);
      if (succeeded(pilotCost)) {
        costUpperBound = pilotCost->total;
      }
    }
    transitions = 0;
    return solveOrdered(state, *order, true);
  }

private:
  FailureOr<VMILayoutPlan> solveOrdered(const SolverState &state,
                                        ArrayRef<unsigned> order,
                                        bool preservePendingDecisionGroups) {
    SmallVector<FrontierEntry, mlir::pto::kValue8> frontier;
    frontier.emplace_back();
    frontier.back().constraints = initialConstraints;
    if (failed(frontier.back().constraints.materialize(frontier.back().plan))) {
      return failure();
    }
    for (auto [position, opIndex] : llvm::enumerate(order)) {
      SmallVector<Operation *, mlir::pto::kValue16> remainingOps;
      for (unsigned remaining : llvm::drop_begin(order, position + 1)) {
        remainingOps.push_back(ops[remaining].op);
      }
      auto next = extend(frontier, opIndex, state.domains[opIndex],
                         remainingOps, preservePendingDecisionGroups);
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
    if (failed(best->constraints.materialize(result))) {
      return failure();
    }
    return result;
  }

  bool hasValidInput() const {
    if (!initialConstraints.isValid() || opIndices.size() != ops.size()) {
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
            VMILayoutAttr targetLayout =
                getUseLayout(ops[consumerIndex].relations[candidate], use);
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
        VMILayoutAttr sourceLayout = getResultLayout(
            ops[producerIndex].relations[candidate], cast<OpResult>(source));
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
    SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue16>
        predecessors(ops.size());
    SmallVector<SmallVector<unsigned, mlir::pto::kValue4>, mlir::pto::kValue16>
        successors(ops.size());
    auto addDependency = [&](unsigned producer, unsigned consumer) {
      if (producer == consumer ||
          llvm::is_contained(successors[producer], consumer)) {
        return;
      }
      successors[producer].push_back(consumer);
      predecessors[consumer].push_back(producer);
    };
    for (auto [consumerIndex, solverOp] : llvm::enumerate(ops)) {
      SmallPtrSet<Operation *, mlir::pto::kValue4> producers;
      for (Value operand : solverOp.op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        auto producerIt = opIndices.find(producer);
        if (producerIt == opIndices.end() ||
            !producers.insert(producer).second) {
          continue;
        }
        addDependency(producerIt->second, consumerIndex);
      }
    }
    for (const VMILayoutEqualityConstraint &equality :
         options.equalityConstraints) {
      Operation *producer = equality.source.getDefiningOp();
      auto producerIt = opIndices.find(producer);
      if (producerIt == opIndices.end()) {
        continue;
      }
      for (Operation *consumer : equality.destination.getUsers()) {
        auto consumerIt = opIndices.find(consumer);
        if (consumerIt != opIndices.end()) {
          addDependency(producerIt->second, consumerIt->second);
        }
      }
    }
    enum class VisitState { Unvisited, Active, Finished };
    SmallVector<unsigned, mlir::pto::kValue16> order;
    SmallVector<VisitState, mlir::pto::kValue16> states(ops.size(),
                                                        VisitState::Unvisited);
    auto visit = [&](unsigned root) {
      if (states[root] == VisitState::Finished) {
        return success();
      }
      SmallVector<std::pair<unsigned, unsigned>, mlir::pto::kValue16> stack;
      states[root] = VisitState::Active;
      stack.emplace_back(root, 0);
      while (!stack.empty()) {
        auto &[current, nextPredecessor] = stack.back();
        if (nextPredecessor < predecessors[current].size()) {
          unsigned predecessor = predecessors[current][nextPredecessor++];
          if (states[predecessor] == VisitState::Active) {
            return failure();
          }
          if (states[predecessor] == VisitState::Unvisited) {
            states[predecessor] = VisitState::Active;
            stack.emplace_back(predecessor, 0);
          }
          continue;
        }
        states[current] = VisitState::Finished;
        order.push_back(current);
        stack.pop_back();
      }
      return success();
    };
    // Traverse backwards from consumers so each producer cone is closed by its
    // relation factor before unrelated roots multiply the live frontier.
    for (unsigned index = 0; index < ops.size(); ++index) {
      if (successors[index].empty() && failed(visit(index))) {
        return failure();
      }
    }
    for (unsigned index = 0; index < ops.size(); ++index) {
      if (failed(visit(index))) {
        return failure();
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

  uint64_t getMaterializationPositionScore(const VMILayoutPlan &plan) const {
    uint64_t score = 0;
    for (const auto &[operand, useLayout] : plan.useLayouts) {
      if (!operand) {
        continue;
      }
      auto valueLayout = plan.valueLayouts.find(operand->get());
      if (valueLayout == plan.valueLayouts.end() ||
          valueLayout->second == useLayout) {
        continue;
      }
      auto opIndex = opIndices.find(operand->getOwner());
      if (opIndex != opIndices.end()) {
        uint64_t position = topologicalPositions[opIndex->second];
        if (position > std::numeric_limits<uint64_t>::max() - score) {
          return std::numeric_limits<uint64_t>::max();
        }
        score += position;
      }
    }
    return score;
  }

  std::string
  getDecisionContinuationKey(const FrontierEntry &entry,
                             const llvm::SmallBitVector &liveGroups) const {
    std::string key;
    llvm::raw_string_ostream stream(key);
    for (unsigned groupIndex = 0; groupIndex < options.decisionGroups.size();
         ++groupIndex) {
      auto selected = entry.selectedDecisionGroups.find(groupIndex);
      if (selected == entry.selectedDecisionGroups.end()) {
        continue;
      }
      if (liveGroups.test(groupIndex)) {
        stream << groupIndex << '=' << selected->second << ';';
      }
    }
    return key;
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
         const llvm::SmallBitVector &domain, ArrayRef<Operation *> remainingOps,
         bool preservePendingDecisionGroups) {
    llvm::StringMap<SmallVector<FrontierEntry, mlir::pto::kValue2>> groups;
    SmallVector<std::string, mlir::pto::kValue8> groupOrder;
    llvm::SmallBitVector liveDecisionGroups(options.decisionGroups.size());
    if (preservePendingDecisionGroups) {
      for (Operation *remainingOp : remainingOps) {
        auto opIndexIt = opIndices.find(remainingOp);
        if (opIndexIt == opIndices.end()) {
          continue;
        }
        int groupIndex = opDecisionGroups[opIndexIt->second];
        if (groupIndex >= 0) {
          liveDecisionGroups.set(static_cast<unsigned>(groupIndex));
        }
      }
    }
    unsigned groupIndex =
        opDecisionGroups.empty()
            ? std::numeric_limits<unsigned>::max()
            : static_cast<unsigned>(opDecisionGroups[opIndex]);
    int memberIndex = -1;
    if (groupIndex != std::numeric_limits<unsigned>::max()) {
      memberIndex =
          llvm::find(options.decisionGroups[groupIndex].opIndices, opIndex) -
          options.decisionGroups[groupIndex].opIndices.begin();
    }
    for (const FrontierEntry &entry : frontier) {
      SmallVector<unsigned, mlir::pto::kValue4> candidates;
      if (memberIndex >= 0 && entry.selectedDecisionGroups.count(groupIndex)) {
        unsigned selected = entry.selectedDecisionGroups.lookup(groupIndex);
        if (selected < options.decisionGroups[groupIndex]
                           .memberRelationIndices[memberIndex]
                           .size()) {
          unsigned mapped = options.decisionGroups[groupIndex]
                                .memberRelationIndices[memberIndex][selected];
          if (domain.test(mapped)) {
            candidates.push_back(mapped);
          }
        }
      } else if (memberIndex >= 0) {
        for (unsigned selected = 0;
             selected < options.decisionGroups[groupIndex]
                            .memberRelationIndices[memberIndex]
                            .size();
             ++selected) {
          unsigned mapped = options.decisionGroups[groupIndex]
                                .memberRelationIndices[memberIndex][selected];
          if (domain.test(mapped)) {
            candidates.push_back(mapped);
          }
        }
      } else {
        for (int relationIndex = domain.find_first(); relationIndex >= 0;
             relationIndex = domain.find_next(relationIndex)) {
          candidates.push_back(static_cast<unsigned>(relationIndex));
        }
      }
      for (unsigned relationIndex : candidates) {
        if (transitions >= options.maxTransitions) {
          return failure();
        }
        ++transitions;
        const VMILayoutOpRelation &relation =
            ops[opIndex].relations[relationIndex];
        FrontierEntry candidate = entry;
        if (memberIndex >= 0 &&
            !candidate.selectedDecisionGroups.count(groupIndex)) {
          unsigned selected = 0;
          while (selected < options.decisionGroups[groupIndex]
                                .memberRelationIndices[memberIndex]
                                .size() &&
                 options.decisionGroups[groupIndex]
                         .memberRelationIndices[memberIndex][selected] !=
                     relationIndex) {
            ++selected;
          }
          if (selected == options.decisionGroups[groupIndex]
                              .memberRelationIndices[memberIndex]
                              .size()) {
            continue;
          }
          candidate.selectedDecisionGroups[groupIndex] = selected;
        }
        if (relation.preferencePenalty > std::numeric_limits<uint64_t>::max() -
                                             candidate.preferencePenalty) {
          return failure();
        }
        candidate.preferencePenalty += relation.preferencePenalty;
        if (failed(
                addRelationToPlan(relation, relationIndex, candidate.plan))) {
          return failure();
        }
        if (failed(candidate.constraints.accept(relation, candidate.plan))) {
          continue;
        }
        // Keep the partial plan consistent with structural equality classes
        // before evaluating its physical cost.  Otherwise block arguments and
        // region-carried values appear as unconstrained fresh inputs until the
        // final plan is materialized, hiding conversions at structural edges.
        if (failed(candidate.constraints.materialize(candidate.plan))) {
          continue;
        }
        candidate.materializationPositionScore =
            getMaterializationPositionScore(candidate.plan);
        auto physicalState = appendVMILayoutPhysicalRelation(
            entry.physicalState, relation, candidate.plan);
        if (failed(physicalState)) {
          continue;
        }
        candidate.physicalState = std::move(*physicalState);
        if (costUpperBound &&
            getVMILayoutPhysicalCost(candidate.physicalState).total >
                *costUpperBound) {
          continue;
        }
        auto key =
            getVMILayoutContinuationKey(candidate.physicalState, remainingOps);
        if (failed(key)) {
          continue;
        }
        auto constraintKey = candidate.constraints.fingerprint();
        if (failed(constraintKey)) {
          return failure();
        }
        std::string combinedKey = *key;
        if (preservePendingDecisionGroups) {
          combinedKey += "|decisions=" + getDecisionContinuationKey(
                                             candidate, liveDecisionGroups);
        }
        combinedKey += "|constraints=" + *constraintKey;
        if (!groups.contains(combinedKey)) {
          groupOrder.push_back(combinedKey);
        }
        insertPareto(std::move(candidate), combinedKey, groups);
      }
    }
    SmallVector<FrontierEntry, mlir::pto::kValue8> result;
    for (const std::string &key : groupOrder) {
      auto group = groups.find(key);
      if (group == groups.end()) {
        return failure();
      }
      result.append(std::make_move_iterator(group->second.begin()),
                    std::make_move_iterator(group->second.end()));
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
  SmallVector<int, mlir::pto::kValue8> opDecisionGroups;
  SmallVector<unsigned, mlir::pto::kValue8> topologicalPositions;
  VMILayoutRelationConstraintState initialConstraints;
  std::optional<int64_t> costUpperBound;
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
