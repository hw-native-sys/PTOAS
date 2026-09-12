// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerRematCandidates.cpp - Remat candidate analysis -----===//

#include "VPTOSchedulerRematerializationInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::remat;

namespace {

struct LoopCost {
    uint64_t multiplier = 1;
    bool crossedLoop = false;
};

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    if (lhs > maxValue - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    if (lhs != 0 && rhs > maxValue / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static std::optional<unsigned> getVectorPressureIndex(const VPTOSchedModel& model)
{
    for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
        if (pressureSet.name == "vector") {
            return index;
        }
    }
    return std::nullopt;
}

static std::optional<uint64_t> getStaticTripCount(scf::ForOp loop)
{
    std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(loop.getStep());
    if (!lower || !upper || !step || *step <= 0) {
        return std::nullopt;
    }
    if (*upper <= *lower) {
        return 0;
    }
    int64_t span = 0;
    bool spanOverflow = llvm::SubOverflow(*upper, *lower, span);
    if (spanOverflow || span <= 0) {
        return std::nullopt;
    }
    uint64_t unsignedSpan = static_cast<uint64_t>(span);
    uint64_t unsignedStep = static_cast<uint64_t>(*step);
    uint64_t roundedSpan = 0;
    if (!checkedAdd(unsignedSpan, unsignedStep - 1, roundedSpan)) {
        return std::nullopt;
    }
    return roundedSpan / unsignedStep;
}

static std::optional<LoopCost> getLoopCost(Operation* producer, Operation* insertionPoint)
{
    LoopCost cost;
    Operation* producerParent = producer->getParentOp();
    for (Operation* ancestor = insertionPoint->getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
        if (ancestor == producerParent) {
            break;
        }
        auto loop = dyn_cast<LoopLikeOpInterface>(ancestor);
        if (!loop) {
            continue;
        }
        auto forOp = dyn_cast<scf::ForOp>(ancestor);
        if (!forOp) {
            return std::nullopt;
        }
        std::optional<uint64_t> tripCount = getStaticTripCount(forOp);
        uint64_t updatedMultiplier = 0;
        bool validMultiplier = tripCount && checkedMultiply(cost.multiplier, *tripCount, updatedMultiplier);
        if (!validMultiplier) {
            return std::nullopt;
        }
        cost.multiplier = updatedMultiplier;
        cost.crossedLoop = true;
    }
    return cost;
}

static bool isVectorPressureValue(Value value, const VPTOSchedModel& model)
{
    std::optional<unsigned> vectorIndex = getVectorPressureIndex(model);
    if (!vectorIndex) {
        return false;
    }
    VPTOPressureSetID vectorSet = model.getPressureSets()[*vectorIndex].id;
    return llvm::any_of(model.getPressure(value), [&](const auto& contribution) {
        return contribution.pressureSet == vectorSet && contribution.units != 0;
    });
}

static bool collectRecipeOperations(
    Value value, const VPTOSchedModel& model, unsigned predecessorDepth, DenseSet<Operation*>& visited,
    SmallVectorImpl<Operation*>& operations, std::string& rejection)
{
    Operation* op = value.getDefiningOp();
    if (!op) {
        return true;
    }
    if (predecessorDepth > kMaxPredecessorDepth) {
        rejection = "producer-depth-limit";
        return false;
    }
    bool validShape = op->getNumRegions() == 0 && op->getNumResults() == 1 && !op->hasTrait<OpTrait::IsTerminator>();
    if (!validShape) {
        rejection = "unsupported-producer-shape";
        return false;
    }
    if (!model.isCheapToRematerialize(op)) {
        rejection = "target-cost-rejected";
        return false;
    }
    bool isPureSchedulable = isPure(op) && model.getSchedClass(op).known;
    if (!isPureSchedulable) {
        rejection = "producer-not-pure-schedulable";
        return false;
    }
    if (!visited.insert(op).second) {
        return true;
    }
    for (Value operand : op->getOperands()) {
        bool hasVectorProducer = isVectorPressureValue(operand, model) && operand.getDefiningOp();
        if (!hasVectorProducer) {
            continue;
        }
        if (!collectRecipeOperations(operand, model, predecessorDepth + 1, visited, operations, rejection)) {
            return false;
        }
    }
    operations.push_back(op);
    return true;
}

static DenseMap<Operation*, TargetPosition> buildTargetPositions(ArrayRef<PressureRegion> regions)
{
    DenseMap<Operation*, TargetPosition> positions;
    for (auto [regionIndex, region] : llvm::enumerate(regions)) {
        for (const auto& entry : region.operationIndices) {
            positions.try_emplace(entry.first, TargetPosition{static_cast<unsigned>(regionIndex), entry.second});
        }
    }
    return positions;
}

static bool usePrecedes(OpOperand* lhs, OpOperand* rhs, const DenseMap<Operation*, TargetPosition>& positions)
{
    TargetPosition lhsPosition = positions.lookup(lhs->getOwner());
    TargetPosition rhsPosition = positions.lookup(rhs->getOwner());
    if (lhsPosition.regionIndex != rhsPosition.regionIndex) {
        return lhsPosition.regionIndex < rhsPosition.regionIndex;
    }
    if (lhsPosition.operationIndex != rhsPosition.operationIndex) {
        return lhsPosition.operationIndex < rhsPosition.operationIndex;
    }
    return lhs->getOperandNumber() < rhs->getOperandNumber();
}

static void appendUseToGroups(
    OpOperand* use, const DenseMap<Operation*, TargetPosition>& positions, SmallVectorImpl<UseGroup>& groups)
{
    TargetPosition position = positions.lookup(use->getOwner());
    bool canJoin = false;
    if (!groups.empty()) {
        UseGroup& last = groups.back();
        canJoin = last.regionIndex == position.regionIndex && last.uses.size() < kMaxUsesPerGroup &&
                  position.operationIndex - last.lastIndex <= kMaxUseGap &&
                  last.insertionPoint->getName() == use->getOwner()->getName();
    }
    if (!canJoin) {
        UseGroup& group = groups.emplace_back();
        group.regionIndex = position.regionIndex;
        group.firstIndex = position.operationIndex;
        group.lastIndex = position.operationIndex;
        group.insertionPoint = use->getOwner();
    }
    groups.back().lastIndex = position.operationIndex;
    groups.back().uses.push_back(use);
}

static bool collectSortedUses(
    Value value, const DenseMap<Operation*, TargetPosition>& positions, SmallVectorImpl<OpOperand*>& uses,
    std::string& rejection)
{
    for (OpOperand& use : value.getUses()) {
        if (!positions.contains(use.getOwner())) {
            rejection = "uncovered-use";
            return false;
        }
        uses.push_back(&use);
    }
    if (uses.empty()) {
        rejection = "no-use";
        return false;
    }
    llvm::sort(uses, [&](OpOperand* lhs, OpOperand* rhs) { return usePrecedes(lhs, rhs, positions); });
    return true;
}

static DenseSet<unsigned> collectAffectedRegions(
    Value value, ArrayRef<Operation*> recipeOperations, ArrayRef<PressureRegion> regions,
    const DenseMap<Operation*, TargetPosition>& positions)
{
    DenseSet<unsigned> affected;
    for (auto [regionIndex, region] : llvm::enumerate(regions)) {
        if (llvm::is_contained(region.liveIns, value)) {
            affected.insert(regionIndex);
        }
    }
    for (Operation* operation : recipeOperations) {
        auto position = positions.find(operation);
        if (position != positions.end()) {
            affected.insert(position->second.regionIndex);
        }
    }
    return affected;
}

static bool externalOperandsDominate(
    ArrayRef<Operation*> recipeOperations, const DenseSet<Operation*>& recipeSet, Operation* insertionPoint,
    DominanceInfo& dominance)
{
    for (Operation* operation : recipeOperations) {
        for (Value operand : operation->getOperands()) {
            bool isRecipeOperand = recipeSet.contains(operand.getDefiningOp());
            if (isRecipeOperand) {
                continue;
            }
            bool operandDominates = dominance.dominates(operand, insertionPoint);
            if (!operandDominates) {
                return false;
            }
        }
    }
    return true;
}

static bool evaluateUseGroups(
    RematCandidate& candidate, const DenseSet<Operation*>& recipeSet, DominanceInfo& dominance,
    DenseSet<unsigned>& affected, std::string& rejection)
{
    bool hasLoopCarriedGroup = false;
    for (UseGroup& group : candidate.groups) {
        if (!externalOperandsDominate(candidate.recipeOperations, recipeSet, group.insertionPoint, dominance)) {
            rejection = "external-operand-does-not-dominate";
            return false;
        }
        std::optional<LoopCost> loopCost = getLoopCost(candidate.value.getDefiningOp(), group.insertionPoint);
        if (!loopCost) {
            rejection = "unknown-loop-cost";
            return false;
        }
        hasLoopCarriedGroup |= loopCost->crossedLoop;
        group.dynamicMultiplier = loopCost->multiplier;
        uint64_t groupCost = 0;
        uint64_t updatedCost = 0;
        bool validGroupCost = checkedMultiply(group.dynamicMultiplier, candidate.recipeOperations.size(), groupCost);
        bool validTotalCost = validGroupCost && checkedAdd(candidate.dynamicMicroOps, groupCost, updatedCost);
        if (!validTotalCost) {
            rejection = "dynamic-cost-overflow";
            return false;
        }
        candidate.dynamicMicroOps = updatedCost;
        affected.insert(group.regionIndex);
    }
    if (!hasLoopCarriedGroup) {
        rejection = "no-loop-carried-use";
    }
    return hasLoopCarriedGroup;
}

static uint64_t computeCoverageBenefit(const RematCandidate& candidate, ArrayRef<PressureRegion> regions)
{
    uint64_t benefit = 0;
    for (unsigned regionIndex : candidate.affectedRegions) {
        if (!regions[regionIndex].needsRelief) {
            continue;
        }
        unsigned lastUse = 0;
        uint64_t localSpan = 0;
        for (const UseGroup& group : candidate.groups) {
            if (group.regionIndex == regionIndex) {
                lastUse = std::max(lastUse, group.lastIndex);
                localSpan += group.lastIndex - group.firstIndex + 1;
            }
        }
        uint64_t originalSpan = static_cast<uint64_t>(lastUse) + 1;
        if (originalSpan > localSpan) {
            benefit += originalSpan - localSpan;
        }
    }
    return benefit;
}

static std::optional<RematCandidate> buildCandidate(
    Value value, ArrayRef<PressureRegion> regions, const DenseMap<Operation*, TargetPosition>& positions,
    const VPTOSchedModel& model, DominanceInfo& dominance, std::string& rejection)
{
    bool hasVectorProducer = value.getDefiningOp() && isVectorPressureValue(value, model);
    if (!hasVectorProducer) {
        rejection = "not-vector-pressure-producer";
        return std::nullopt;
    }

    DenseSet<Operation*> recipeSet;
    SmallVector<Operation*> recipeOperations;
    if (!collectRecipeOperations(value, model, 0, recipeSet, recipeOperations, rejection)) {
        return std::nullopt;
    }

    SmallVector<OpOperand*> uses;
    if (!collectSortedUses(value, positions, uses, rejection)) {
        return std::nullopt;
    }

    RematCandidate candidate;
    candidate.value = value;
    candidate.recipeOperations = std::move(recipeOperations);
    for (OpOperand* use : uses) {
        appendUseToGroups(use, positions, candidate.groups);
    }
    size_t groupCount = candidate.groups.size();
    if (groupCount > kMaxGroupsPerCandidate) {
        rejection = "group-budget";
        return std::nullopt;
    }

    DenseSet<unsigned> affected = collectAffectedRegions(value, candidate.recipeOperations, regions, positions);
    if (!evaluateUseGroups(candidate, recipeSet, dominance, affected, rejection)) {
        return std::nullopt;
    }
    candidate.cloneOperations = static_cast<unsigned>(candidate.groups.size() * candidate.recipeOperations.size());
    candidate.affectedRegions.append(affected.begin(), affected.end());
    llvm::sort(candidate.affectedRegions);
    candidate.coverageBenefit = computeCoverageBenefit(candidate, regions);
    return candidate;
}

} // namespace

SmallVector<RematCandidate, 0> mlir::pto::remat::collectCandidates(
    ArrayRef<PressureRegion> regions, const VPTOSchedModel& model, func::FuncOp func, llvm::raw_ostream& os, bool trace)
{
    DenseMap<Operation*, TargetPosition> positions = buildTargetPositions(regions);
    DenseSet<Value> visited;
    SmallVector<RematCandidate, 0> candidates;
    DominanceInfo dominance(func);
    unsigned candidateIndex = 0;
    for (const PressureRegion& region : regions) {
        if (!region.needsRelief) {
            continue;
        }
        for (Value liveIn : region.liveIns) {
            bool isNewProducer =
                visited.insert(liveIn).second && liveIn.getDefiningOp() && isVectorPressureValue(liveIn, model);
            if (!isNewProducer) {
                continue;
            }
            std::string rejection;
            unsigned diagnosticId = candidateIndex++;
            std::optional<RematCandidate> candidate =
                buildCandidate(liveIn, regions, positions, model, dominance, rejection);
            if (!candidate) {
                if (trace) {
                    os << "vpto-scheduler: remat-candidate id=" << diagnosticId
                       << " selected=false reason=" << rejection << '\n';
                }
                continue;
            }
            candidate->diagnosticId = diagnosticId;
            if (trace) {
                os << "vpto-scheduler: remat-candidate id=" << diagnosticId
                   << " selected=pending groups=" << candidate->groups.size()
                   << " recipe-ops=" << candidate->recipeOperations.size() << " clones=" << candidate->cloneOperations
                   << " dynamic-micro-ops=" << candidate->dynamicMicroOps
                   << " coverage-benefit=" << candidate->coverageBenefit << '\n';
            }
            candidates.push_back(std::move(*candidate));
        }
    }
    return candidates;
}
