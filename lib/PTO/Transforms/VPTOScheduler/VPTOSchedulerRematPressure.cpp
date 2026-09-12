// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerRematPressure.cpp - Remat pressure analysis ---------===//

#include "VPTOSchedulerRematerializationInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"
#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "mlir/Analysis/Liveness.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::remat;

namespace {

static std::optional<unsigned> getVectorPressureIndex(const VPTOSchedModel& model)
{
    for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
        if (pressureSet.name == "vector") {
            return index;
        }
    }
    return std::nullopt;
}

static std::optional<SmallVector<int64_t>> evaluateRegionPressure(
    const VPTOSchedRegion& region, const VPTOSchedModel& model, SmallVectorImpl<Value>* liveIns = nullptr)
{
    VPTOSchedulerLimits limits;
    VPTOSchedulingBudget budget(limits.maxWorkUnits);
    VPTOScheduleFailure failure;
    VPTOSchedDAGBuilder builder(&model, limits, budget);
    FailureOr<std::unique_ptr<VPTOSchedDAG>> dag = builder.build(region, failure);
    if (failed(dag)) {
        return std::nullopt;
    }
    VPTORegPressureTracker tracker(model, **dag, VPTOSchedDirection::Top);
    for (const std::unique_ptr<VPTOSUnit>& unit : (*dag)->getUnits()) {
        if (failed(tracker.commit(*unit))) {
            return std::nullopt;
        }
    }
    if (liveIns) {
        liveIns->append((*dag)->getLiveIns().begin(), (*dag)->getLiveIns().end());
    }
    SmallVector<int64_t> peak;
    peak.append(tracker.getPeak().begin(), tracker.getPeak().end());
    return peak;
}

template <typename Callback>
static void walkSchedulingBlocks(Region& parentRegion, unsigned& blockIndex, Callback&& callback)
{
    for (Block& block : parentRegion) {
        callback(block, blockIndex++);
        for (Operation& op : block) {
            if (isa<VecScopeOp, StrictVecScopeOp>(op)) {
                continue;
            }
            for (Region& nestedRegion : op.getRegions()) {
                walkSchedulingBlocks(nestedRegion, blockIndex, callback);
            }
        }
    }
}

template <typename Callback>
static void walkFunctionSchedulingBlocks(func::FuncOp func, Callback&& callback)
{
    SmallVector<Operation*> vecScopes;
    func.walk([&](Operation* op) {
        if (isa<VecScopeOp, StrictVecScopeOp>(op)) {
            vecScopes.push_back(op);
        }
    });
    unsigned blockIndex = 0;
    for (Operation* vecScope : vecScopes) {
        walkSchedulingBlocks(vecScope->getRegion(0), blockIndex, callback);
    }
}

} // namespace

SmallVector<PressureRegion, 0> mlir::pto::remat::collectPressureRegions(
    func::FuncOp func, const VPTOSchedModel& model, llvm::raw_ostream& os, bool trace,
    SmallVectorImpl<int64_t>& maxPressure)
{
    SmallVector<PressureRegion, 0> pressureRegions;
    std::optional<unsigned> vectorIndex = getVectorPressureIndex(model);
    if (!vectorIndex || !model.getPressureSets()[*vectorIndex].limit) {
        return pressureRegions;
    }
    maxPressure.assign(model.getPressureSets().size(), 0);
    int64_t limit = static_cast<int64_t>(*model.getPressureSets()[*vectorIndex].limit);
    Liveness liveness(func);
    VPTOSchedulingCoverage coverage;
    walkFunctionSchedulingBlocks(func, [&](Block& block, unsigned blockIndex) {
        VPTOSchedRegionBuilder regionBuilder(&coverage, &liveness);
        for (VPTOSchedRegion& region : regionBuilder.build(block)) {
            SmallVector<Value> liveIns;
            std::optional<SmallVector<int64_t>> peak = evaluateRegionPressure(region, model, &liveIns);
            if (!peak) {
                if (trace) {
                    os << "vpto-scheduler: remat-region block=" << blockIndex << " region=" << region.index
                       << " fallback=analysis-failed\n";
                }
                continue;
            }
            for (auto [index, value] : llvm::enumerate(*peak)) {
                maxPressure[index] = std::max(maxPressure[index], value);
            }
            int64_t vectorPeak = (*peak)[*vectorIndex];
            bool highPressure = vectorPeak > limit;
            if (trace) {
                os << "vpto-scheduler: remat-region block=" << blockIndex << " region=" << region.index
                   << " peak=" << vectorPeak << " limit=" << limit << " trigger=" << (highPressure ? "true" : "false")
                   << '\n';
            }
            PressureRegion& target = pressureRegions.emplace_back();
            target.region = std::move(region);
            target.blockIndex = blockIndex;
            target.needsRelief = highPressure;
            target.peak = vectorPeak;
            target.limit = limit;
            target.target = highPressure ? std::max<int64_t>(0, limit - kPressureHeadroom) : vectorPeak;
            target.liveIns = std::move(liveIns);
            for (auto [index, op] : llvm::enumerate(target.region.operations)) {
                target.operationIndices.try_emplace(op, index);
            }
        }
    });
    return pressureRegions;
}

std::optional<SmallVector<int64_t>> mlir::pto::remat::evaluateMaxPressure(
    func::FuncOp func, const VPTOSchedModel& model)
{
    SmallVector<int64_t> maxPressure(model.getPressureSets().size(), 0);
    bool failedAnalysis = false;
    Liveness liveness(func);
    VPTOSchedulingCoverage coverage;
    walkFunctionSchedulingBlocks(func, [&](Block& block, unsigned) {
        VPTOSchedRegionBuilder regionBuilder(&coverage, &liveness);
        for (VPTOSchedRegion& region : regionBuilder.build(block)) {
            std::optional<SmallVector<int64_t>> peak = evaluateRegionPressure(region, model);
            if (!peak) {
                failedAnalysis = true;
                continue;
            }
            for (auto [index, value] : llvm::enumerate(*peak)) {
                maxPressure[index] = std::max(maxPressure[index], value);
            }
        }
    });
    if (failedAnalysis) {
        return std::nullopt;
    }
    return maxPressure;
}
