// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Runtime versioning supplies a checked domain, never a trusted upper bound.
// Speculate with increasing finite envelopes and retain a version only when
// re-running the production analysis proves a lower cost. The original loop
// remains the fallback, including all of its original memory operations.
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarVersioning.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarAnalysis.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarPlacement.h"
#include "PTO/Transforms/VPTOStatefulStreamFusion.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;
namespace {
constexpr int64_t kLargestProbe = 256;
constexpr unsigned kOperationBudget = 512;

static std::optional<uint64_t> synchronizationCost(Operation* scope)
{
    auto analysis = runVecScopeMemBarAnalysis(scope);
    auto plan = succeeded(analysis) ? solveVecScopeMemBarPlacement(*analysis) :
                                      FailureOr<VecScopeMemBarPlan>(failure());
    if (failed(plan)) {
        return std::nullopt;
    }
    uint64_t cost = plan->barriers.size();
    scope->walk([&](VstasOp flush) {
        cost += flush->getParentOfType<scf::ForOp>() ? kValue2 : 1;
    });
    return cost;
}

static bool eligibleLoop(scf::ForOp loop)
{
    if (
        !isa<VecScopeOp>(loop->getParentOp()) || loop.getNumResults() != 0 ||
        integer_range::getConstantInt(loop.getUpperBound()).has_value() ||
        integer_range::getConstantInt(loop.getLowerBound()) != 0 ||
        integer_range::getConstantInt(loop.getStep()) != 1) {
        return false;
    }
    unsigned operations = 0;
    loop.walk([&](Operation*) { ++operations; });
    if (operations > kOperationBudget) {
        return false;
    }
    unsigned stores = 0;
    unsigned barriers = 0;
    loop.walk([&](VstusOp) { ++stores; });
    loop.walk([&](MemBarOp) { ++barriers; });
    // One short-store pipeline bounds code growth and avoids duplicating
    // regions already carrying explicit synchronization.
    return stores == 1 && barriers == 0;
}

struct Candidate {
    scf::IfOp branch;
    Operation* fastScope;
    SmallVector<std::pair<Value, Value>> externalValues;
    SmallVector<Operation*> guardOperations;
};

static Candidate buildCandidate(scf::ForOp loop, int64_t bound, OpBuilder& builder)
{
    IRMapping mapping;
    Operation* scope = loop->getParentOp()->clone(mapping);
    SmallVector<std::pair<Value, Value>> externalValues;
    for (Operation& op : loop->getParentOp()->getRegion(0).front()) {
        if (&op != loop) {
            for (Value value : op.getResults()) {
                externalValues.emplace_back(mapping.lookup(value), value);
            }
        }
    }
    builder.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    Type type = loop.getUpperBound().getType();
    Value limit = builder.create<arith::ConstantOp>(loc, type, builder.getIntegerAttr(type, bound));
    Value zero = builder.create<arith::ConstantOp>(loc, type, builder.getIntegerAttr(type, 0));
    Value lower = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, loop.getUpperBound(), zero);
    Value upper = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, loop.getUpperBound(), limit);
    Value condition = builder.create<arith::AndIOp>(loc, lower, upper);
    auto branch = builder.create<scf::IfOp>(loc, condition, true);
    builder.setInsertionPoint(branch.thenYield());
    // The clone precedes creation of the branch, so it never contains its
    // own probe. No instruction executes during compile-time analysis.
    builder.insert(scope);
    return {branch, scope, std::move(externalValues),
            {limit.getDefiningOp(), zero.getDefiningOp(), lower.getDefiningOp(),
             upper.getDefiningOp(), condition.getDefiningOp()}};
}

static void eraseCandidate(Candidate candidate)
{
    candidate.branch.erase();
    for (Operation* op : llvm::reverse(candidate.guardOperations)) {
        op->erase();
    }
}

static int64_t largestProbe(scf::ForOp loop)
{
    auto integer = dyn_cast<IntegerType>(loop.getUpperBound().getType());
    unsigned width = integer ? integer.getWidth() : 64;
    int64_t largest = kLargestProbe;
    if (width > 0 && width < kValue16) {
        largest = std::min(largest, (int64_t{1} << (width - 1)) - 1);
    }
    auto envelope = integer_range::getLoopEnvelope(loop);
    // A type-derived upper bound may be too wide to prove disjoint accesses.
    // Only probe guards that narrow the domain already known to the analysis.
    if (envelope.valid && envelope.ub <= largest) {
        largest = envelope.ub > 0 ? envelope.ub - 1 : 0;
    }
    return largest;
}

static bool tryVersion(scf::ForOp loop)
{
    Operation* originalScope = loop->getParentOp();
    // Restrict to a scope whose only region-bearing operation is this loop.
    // Moving the branch into that scope then preserves all surrounding SSA.
    for (Operation& op : originalScope->getRegion(0).front()) {
        if (
            &op != loop && !op.hasTrait<OpTrait::IsTerminator>() &&
            (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))) {
            return false;
        }
    }
    auto baseline = synchronizationCost(originalScope);
    if (!baseline || *baseline == 0) {
        return false;
    }
    OpBuilder builder(loop.getContext());
    int64_t selected = 0;
    uint64_t bestCost = *baseline;
    int64_t largest = largestProbe(loop);
    for (int64_t bound = kValue2; bound <= largest; bound *= kValue2) {
        auto candidate = buildCandidate(loop, bound, builder);
        runVPTOStatefulStreamFusion(candidate.fastScope);
        auto cost = synchronizationCost(candidate.fastScope);
        bool better = cost && *cost < *baseline && *cost <= bestCost;
        if (better) {
            bestCost = *cost;
        }
        eraseCandidate(candidate);
        if (!better) {
            break;
        }
        selected = bound;
    }
    if (selected == 0) {
        return false;
    }
    auto candidate = buildCandidate(loop, selected, builder);
    // Use only the cloned loop; the surrounding scope operations stay once.
    auto clonedScope = cast<VecScopeOp>(candidate.fastScope);
    auto fastLoop = llvm::find_if(clonedScope->getRegion(0).front(), [](Operation& op) {
        return isa<scf::ForOp>(op);
    });
    for (auto [clone, original] : candidate.externalValues) {
        clone.replaceAllUsesWith(original);
    }
    fastLoop->moveBefore(candidate.branch.thenYield());
    clonedScope.erase();
    loop->moveBefore(candidate.branch.elseYield());
    runVPTOStatefulStreamFusion(candidate.branch.getThenRegion().getParentOp());
    return true;
}
} // namespace

void vecscopemembar::versionDynamicStoreStreams(func::FuncOp function)
{
    SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) {
        if (eligibleLoop(loop)) {
            loops.push_back(loop);
        }
    });
    for (scf::ForOp loop : loops) {
        tryVersion(loop);
    }
}
