// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Public cost/state API and continuation fingerprints. Operation pricing and
// layout materialization live in separate physical graph implementation units.
#include "VMILayoutCostGraph.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::layout_cost;
namespace {

static void rememberCanonicalId(DenseMap<unsigned, unsigned>& canonicalIds, unsigned physicalId)
{
    if (!canonicalIds.contains(physicalId)) {
        canonicalIds[physicalId] = canonicalIds.size();
    }
}

static FailureOr<std::string> buildContinuationKey(
    const PhysicalStateStorage& storage, ArrayRef<Operation*> remainingOps)
{
    constexpr unsigned maxObservableValues = 256;
    DenseSet<Value> seenValues;
    SmallVector<Value, mlir::pto::kValue16> liveValues;
    for (Operation* op : remainingOps) {
        for (Value operand : op->getOperands()) {
            if (storage.values.contains(operand) && seenValues.insert(operand).second) {
                liveValues.push_back(operand);
            }
        }
    }

    DenseMap<unsigned, unsigned> canonicalIds;
    for (Value value : liveValues) {
        for (PhysicalValue part : storage.values.lookup(value)) {
            rememberCanonicalId(canonicalIds, part.id);
        }
    }
    if (canonicalIds.size() > maxObservableValues) {
        return failure();
    }

    std::string key;
    llvm::raw_string_ostream os(key);
    for (Value value : liveValues) {
        VMILayoutAttr layout = storage.layouts.lookup(value);
        if (!layout) {
            return failure();
        }
        os << "V{" << layout << ':';
        for (PhysicalValue part : storage.values.lookup(value)) {
            os << canonicalIds.lookup(part.id) << ',';
        }
        os << '}';
    }
    return os.str();
}

} // namespace

struct VMILayoutPhysicalState::Impl {
    PhysicalStateStorage storage;
};

VMILayoutPhysicalState::VMILayoutPhysicalState() : impl(std::make_shared<Impl>()) {}

VMILayoutPhysicalState::VMILayoutPhysicalState(const VMILayoutPhysicalState&) = default;
VMILayoutPhysicalState::VMILayoutPhysicalState(VMILayoutPhysicalState&&) noexcept = default;
VMILayoutPhysicalState& VMILayoutPhysicalState::operator=(const VMILayoutPhysicalState&) = default;
VMILayoutPhysicalState& VMILayoutPhysicalState::operator=(VMILayoutPhysicalState&&) noexcept = default;
VMILayoutPhysicalState::~VMILayoutPhysicalState() = default;

FailureOr<VMILayoutPhysicalState> mlir::pto::appendVMILayoutPhysicalRelation(
    const VMILayoutPhysicalState& state, const VMILayoutOpRelation& relation, const VMILayoutPlan& partialPlan)
{
    if (!state.impl || !relation.op) {
        return failure();
    }
    PlanGraphBuilder builder(state.impl->storage, relation, partialPlan);
    auto storage = builder.append(relation);
    if (failed(storage)) {
        return failure();
    }
    VMILayoutPhysicalState result;
    result.impl->storage = std::move(*storage);
    return result;
}

FailureOr<std::string> mlir::pto::getVMILayoutContinuationKey(
    const VMILayoutPhysicalState& state, ArrayRef<Operation*> remainingOps)
{
    if (!state.impl) {
        return failure();
    }
    return buildContinuationKey(state.impl->storage, remainingOps);
}

VMILayoutScopeCost mlir::pto::getVMILayoutPhysicalCost(const VMILayoutPhysicalState& state)
{
    if (state.impl) {
        return state.impl->storage.graph.getCost();
    }
    VMILayoutScopeCost empty;
    return empty;
}

FailureOr<VMILayoutScopeCost> mlir::pto::evaluateVMILayoutPlanCost(
    ArrayRef<VMILayoutOpRelation> selectedRelations, const VMILayoutPlan& plan)
{
    return PlanGraphBuilder(selectedRelations, plan).build();
}
