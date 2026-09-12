// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerRematPlanner.cpp - Bounded remat selection ----------===//

#include "VPTOSchedulerRematerializationInternal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::pto::remat;

namespace {

static unsigned countUnmetRegions(const RematCandidate& candidate, ArrayRef<int64_t> remaining)
{
    return llvm::count_if(candidate.affectedRegions, [&](unsigned index) { return remaining[index] > 0; });
}

static bool candidateFitsBudget(
    const RematCandidate& candidate, unsigned selectedCount, unsigned cloneOperations, uint64_t dynamicMicroOps)
{
    if (selectedCount >= kMaxCandidates || candidate.cloneOperations > kMaxCloneOperations - cloneOperations) {
        return false;
    }
    return candidate.dynamicMicroOps <= kMaxDynamicMicroOps - dynamicMicroOps;
}

static unsigned countDeadRecipeOperations(
    ArrayRef<RematCandidate> candidates, ArrayRef<unsigned> selected, std::optional<unsigned> additional)
{
    DenseSet<Operation*> recipeOperations;
    DenseSet<Operation*> deadOperations;
    auto addCandidate = [&](const RematCandidate& candidate) {
        recipeOperations.insert(candidate.recipeOperations.begin(), candidate.recipeOperations.end());
        deadOperations.insert(candidate.value.getDefiningOp());
    };
    for (unsigned index : selected) {
        addCandidate(candidates[index]);
    }
    if (additional) {
        addCandidate(candidates[*additional]);
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (Operation* operation : recipeOperations) {
            if (deadOperations.contains(operation)) {
                continue;
            }
            bool allUsersDead =
                llvm::all_of(operation->getUsers(), [&](Operation* user) { return deadOperations.contains(user); });
            if (allUsersDead) {
                changed |= deadOperations.insert(operation).second;
            }
        }
    }
    return deadOperations.size();
}

static bool isBetterCandidate(
    const RematCandidate& candidate, const RematCandidate& best, unsigned help, unsigned bestHelp,
    unsigned marginalDeadOperations, unsigned bestMarginalDeadOperations)
{
    if (help != bestHelp) {
        return help > bestHelp;
    }
    if (candidate.coverageBenefit != best.coverageBenefit) {
        return candidate.coverageBenefit > best.coverageBenefit;
    }
    if (marginalDeadOperations != bestMarginalDeadOperations) {
        return marginalDeadOperations > bestMarginalDeadOperations;
    }
    return candidate.dynamicMicroOps < best.dynamicMicroOps;
}

static void printUnselectedCandidates(
    ArrayRef<RematCandidate> candidates, const DenseSet<unsigned>& selected, bool hasUnmetRegion,
    unsigned cloneOperations, uint64_t dynamicMicroOps, llvm::raw_ostream& os)
{
    for (auto [index, candidate] : llvm::enumerate(candidates)) {
        if (selected.contains(index)) {
            continue;
        }
        bool fitsBudget = candidateFitsBudget(candidate, selected.size(), cloneOperations, dynamicMicroOps);
        StringRef reason =
            fitsBudget ? (hasUnmetRegion ? "no-additional-relief" : "pressure-target-satisfied") : "budget-limit";
        os << "vpto-scheduler: remat-candidate id=" << candidate.diagnosticId << " selected=false reason=" << reason
           << '\n';
    }
}

static std::optional<unsigned> findBestCandidate(
    ArrayRef<RematCandidate> candidates, ArrayRef<unsigned> selected, const DenseSet<unsigned>& selectedSet,
    ArrayRef<int64_t> remaining, unsigned cloneOperations, uint64_t dynamicMicroOps)
{
    std::optional<unsigned> best;
    unsigned bestHelp = 0;
    unsigned bestMarginalDeadOperations = 0;
    unsigned currentDeadOperations = countDeadRecipeOperations(candidates, selected, std::nullopt);
    for (auto [index, candidate] : llvm::enumerate(candidates)) {
        bool alreadySelected = selectedSet.contains(index);
        bool fitsBudget = candidateFitsBudget(candidate, selected.size(), cloneOperations, dynamicMicroOps);
        if (alreadySelected || !fitsBudget) {
            continue;
        }
        unsigned help = countUnmetRegions(candidate, remaining);
        if (help == 0) {
            continue;
        }
        unsigned deadOperations = countDeadRecipeOperations(candidates, selected, index);
        unsigned marginalDeadOperations = deadOperations - currentDeadOperations;
        if (!best ||
            isBetterCandidate(
                candidate, candidates[*best], help, bestHelp, marginalDeadOperations, bestMarginalDeadOperations)) {
            best = index;
            bestHelp = help;
            bestMarginalDeadOperations = marginalDeadOperations;
        }
    }
    return best;
}

static void applyCandidateRelief(const RematCandidate& candidate, SmallVectorImpl<int64_t>& remaining)
{
    for (unsigned regionIndex : candidate.affectedRegions) {
        remaining[regionIndex] = std::max<int64_t>(0, remaining[regionIndex] - 1);
    }
}

} // namespace

SmallVector<unsigned> mlir::pto::remat::selectCandidates(
    ArrayRef<RematCandidate> candidates, ArrayRef<PressureRegion> regions, llvm::raw_ostream& os, bool trace,
    unsigned& cloneOperations, uint64_t& dynamicMicroOps)
{
    SmallVector<int64_t> remaining;
    for (const PressureRegion& region : regions) {
        remaining.push_back(region.peak - region.target);
    }
    SmallVector<unsigned> selected;
    DenseSet<unsigned> selectedSet;
    while (llvm::any_of(remaining, [](int64_t value) { return value > 0; })) {
        std::optional<unsigned> best =
            findBestCandidate(candidates, selected, selectedSet, remaining, cloneOperations, dynamicMicroOps);
        if (!best) {
            break;
        }
        const RematCandidate& candidate = candidates[*best];
        selected.push_back(*best);
        selectedSet.insert(*best);
        cloneOperations += candidate.cloneOperations;
        dynamicMicroOps += candidate.dynamicMicroOps;
        applyCandidateRelief(candidate, remaining);
    }

    bool hasUnmetRegion = llvm::any_of(remaining, [](int64_t value) { return value > 0; });
    if (trace) {
        printUnselectedCandidates(candidates, selectedSet, hasUnmetRegion, cloneOperations, dynamicMicroOps, os);
    }
    if (trace && hasUnmetRegion && !selected.empty()) {
        os << "vpto-scheduler: remat-partial-plan reason=insufficient-estimated-relief"
           << " selected=" << selected.size() << " clones=" << cloneOperations
           << " dynamic-micro-ops=" << dynamicMicroOps << '\n';
    }
    return selected;
}
