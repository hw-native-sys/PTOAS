// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerRematerialization.cpp - Transactional VPTO remat ---===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedulerRematerialization.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"
#include "VPTOSchedulerRematerializationInternal.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::remat;

namespace {

static void appendUniqueOperation(SmallVectorImpl<Operation*>& operations, Operation* operation)
{
    if (!llvm::is_contained(operations, operation)) {
        operations.push_back(operation);
    }
}

static void setInitialVectorPressure(
    const VPTOSchedModel& model, ArrayRef<int64_t> initialMaxPressure, VPTORematerializationStats& stats)
{
    for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
        if (pressureSet.name == "vector" && index < initialMaxPressure.size()) {
            stats.initialMaxVectorPressure = initialMaxPressure[index];
            return;
        }
    }
}

} // namespace

struct VPTORematerializationTransaction::Builder {
    static void cloneCandidate(const RematCandidate& candidate, VPTORematerializationTransaction& transaction)
    {
        transaction.stats.cloneGroups += candidate.groups.size();
        for (const UseGroup& group : candidate.groups) {
            OpBuilder builder(group.insertionPoint);
            IRMapping mapping;
            Operation* firstClone = nullptr;
            for (Operation* operation : candidate.recipeOperations) {
                Operation* clone = builder.clone(*operation, mapping);
                firstClone = firstClone ? firstClone : clone;
                transaction.clones.push_back(clone);
            }
            Value replacement = mapping.lookup(candidate.value);
            for (OpOperand* use : group.uses) {
                transaction.replacedUses.push_back({use, candidate.value});
                use->set(replacement);
            }
            transaction.anchors.insert(firstClone);
        }
        for (Operation* operation : candidate.recipeOperations) {
            appendUniqueOperation(transaction.originalOperations, operation);
        }
    }

    static void applyCandidates(
        ArrayRef<RematCandidate> candidates, ArrayRef<unsigned> selected, VPTORematerializationTransaction& transaction,
        llvm::raw_ostream& os, bool trace)
    {
        for (unsigned candidateIndex : selected) {
            const RematCandidate& candidate = candidates[candidateIndex];
            if (trace) {
                os << "vpto-scheduler: remat-select id=" << candidate.diagnosticId
                   << " groups=" << candidate.groups.size() << " recipe-ops=" << candidate.recipeOperations.size()
                   << " clones=" << candidate.cloneOperations << " dynamic-micro-ops=" << candidate.dynamicMicroOps
                   << " coverage-benefit=" << candidate.coverageBenefit << '\n';
            }
            cloneCandidate(candidate, transaction);
        }
    }
};

VPTORematerializationTransaction mlir::pto::prepareVPTORematerialization(
    func::FuncOp func, const VPTOSchedModel& model, llvm::raw_ostream& os, bool trace)
{
    VPTORematerializationTransaction transaction;
    SmallVector<PressureRegion, 0> regions =
        collectPressureRegions(func, model, os, trace, transaction.stats.initialMaxPressure);
    setInitialVectorPressure(model, transaction.stats.initialMaxPressure, transaction.stats);
    transaction.stats.highPressureRegions =
        llvm::count_if(regions, [](const PressureRegion& region) { return region.needsRelief; });
    if (transaction.stats.highPressureRegions == 0) {
        if (trace) {
            os << "vpto-scheduler: remat-summary function=" << func.getSymName()
               << " changed=false reason=no-high-pressure-region\n";
        }
        return transaction;
    }

    SmallVector<RematCandidate, 0> candidates = collectCandidates(regions, model, func, os, trace);
    unsigned cloneOperations = 0;
    uint64_t dynamicMicroOps = 0;
    SmallVector<unsigned> selected = selectCandidates(candidates, regions, os, trace, cloneOperations, dynamicMicroOps);
    if (selected.empty()) {
        if (trace) {
            os << "vpto-scheduler: remat-summary function=" << func.getSymName()
               << " changed=false reason=no-budgeted-plan\n";
        }
        return transaction;
    }

    VPTORematerializationTransaction::Builder::applyCandidates(candidates, selected, transaction, os, trace);
    transaction.stats.changed = true;
    transaction.stats.selectedCandidates = selected.size();
    transaction.stats.cloneOperations = cloneOperations;
    transaction.stats.estimatedDynamicMicroOps = dynamicMicroOps;
    if (trace) {
        os << "vpto-scheduler: remat-summary function=" << func.getSymName()
           << " changed=true high-regions=" << transaction.stats.highPressureRegions << " selected=" << selected.size()
           << " groups=" << transaction.stats.cloneGroups << " clones=" << cloneOperations
           << " dynamic-micro-ops=" << dynamicMicroOps << '\n';
    }
    return transaction;
}

void VPTORematerializationTransaction::commit()
{
    for (Operation* operation : llvm::reverse(originalOperations)) {
        if (operation->use_empty()) {
            operation->erase();
        }
    }
    // A clone can feed an original recipe operation that another selected
    // candidate makes dead. Erase consumers before producers so those
    // transaction-local clone chains do not survive a successful commit.
    for (Operation* clone : llvm::reverse(clones)) {
        if (clone->use_empty()) {
            clone->erase();
        }
    }
    replacedUses.clear();
    clones.clear();
    anchors.clear();
    originalOperations.clear();
}

void VPTORematerializationTransaction::rollback()
{
    for (ReplacedUse& replacement : replacedUses) {
        replacement.use->set(replacement.original);
    }
    for (Operation* clone : llvm::reverse(clones)) {
        clone->erase();
    }
    replacedUses.clear();
    clones.clear();
    anchors.clear();
    originalOperations.clear();
    stats.changed = false;
}

std::optional<SmallVector<int64_t>> mlir::pto::evaluateVPTOMaxPressure(func::FuncOp func, const VPTOSchedModel& model)
{
    return remat::evaluateMaxPressure(func, model);
}
