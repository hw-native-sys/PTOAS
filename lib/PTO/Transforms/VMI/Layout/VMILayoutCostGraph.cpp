// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Physical cost graph: preserves the action graph selected by the planner.
#include "VMILayoutCostGraph.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_cost {

PlanGraphBuilder::PlanGraphBuilder(ArrayRef<VMILayoutOpRelation> relations, const VMILayoutPlan& plan) : plan(plan)
{
    for (const VMILayoutOpRelation& relation : relations) {
        selectedRelations[relation.op] = relation;
    }
}

PlanGraphBuilder::PlanGraphBuilder(
    const PhysicalStateStorage& initial, const VMILayoutOpRelation& relation, const VMILayoutPlan& plan)
    : plan(plan),
      values(initial.values),
      layouts(initial.layouts),
      builtOps(initial.builtOps),
      sharedMaterializations(initial.sharedMaterializations),
      graph(initial.graph)
{
    selectedRelations = initial.selectedRelations;
    selectedRelations[relation.op] = relation;
}

FailureOr<VMILayoutScopeCost> PlanGraphBuilder::build()
{
    for (const auto& entry : selectedRelations) {
        if (failed(buildOp(entry.first))) {
            return failure();
        }
    }
    return graph.getCost();
}

FailureOr<PhysicalStateStorage> PlanGraphBuilder::append(const VMILayoutOpRelation& relation)
{
    if (failed(buildOp(relation.op))) {
        return failure();
    }
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind == VMILayoutPortKind::Result) {
            layouts[relation.op->getResult(port.index)] = port.layout;
        }
    }
    return PhysicalStateStorage{std::move(graph),    std::move(values),
                                std::move(layouts),  std::move(selectedRelations),
                                std::move(builtOps), std::move(sharedMaterializations)};
}

FailureOr<PhysicalValues> PlanGraphBuilder::buildBlockArgumentValue(Value value)
{
    auto type = dyn_cast<VMIVRegType>(value.getType());
    auto planned = plan.valueLayouts.find(value);
    VMILayoutAttr layout =
        planned == plan.valueLayouts.end() ? (type ? type.getLayoutAttr() : VMILayoutAttr{}) : planned->second;
    if (auto mask = dyn_cast<VMIMaskType>(value.getType())) {
        if (planned == plan.valueLayouts.end()) {
            layout = mask.getLayoutAttr();
        }
        if (!layout) {
            return failure();
        }
        values[value] = graph.createInputs(1);
        layouts[value] = layout;
        return values.lookup(value);
    }
    auto arity = getArity(type, layout);
    if (!type || !layout || failed(arity) || *arity <= 0 ||
        static_cast<uint64_t>(*arity) > std::numeric_limits<unsigned>::max()) {
        return failure();
    }
    values[value] = graph.createInputs(static_cast<unsigned>(*arity));
    layouts[value] = layout;
    return values.lookup(value);
}
FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::buildValue(Value value)
{
    auto cached = values.find(value);
    if (cached != values.end()) {
        return cached->second;
    }
    Operation* definingOp = value.getDefiningOp();
    if (!definingOp) {
        return buildBlockArgumentValue(value);
    }
    if (!selectedRelations.count(definingOp)) {
        auto type = dyn_cast<VMIVRegType>(value.getType());
        VMILayoutAttr layout = plan.valueLayouts.lookup(value);
        if (!layout) {
            layout = layouts.lookup(value);
        }
        if (type && layout) {
            auto arity = getArity(type, layout);
            if (succeeded(arity) && *arity > 0) {
                values[value] = graph.createInputs(static_cast<unsigned>(*arity));
                layouts[value] = layout;
                return values.lookup(value);
            }
        }
        if (isa<VMIMaskType>(value.getType()) && layout) {
            values[value] = graph.createInputs(1);
            layouts[value] = layout;
            return values.lookup(value);
        }
    }
    if (failed(buildOp(definingOp))) {
        return failure();
    }
    cached = values.find(value);
    return cached == values.end() ? FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>(failure()) :
                                    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>>(cached->second);
}

void PlanGraphBuilder::seedStructuralOperand(OpOperand& operand, VMILayoutAttr targetLayout)
{
    Operation* definingOp = operand.get().getDefiningOp();
    bool isStructuralValue = isa<BlockArgument>(operand.get()) || (definingOp && !selectedRelations.count(definingOp));
    if (isStructuralValue && !plan.valueLayouts.contains(operand.get()) && !layouts.lookup(operand.get())) {
        auto type = dyn_cast<VMIVRegType>(operand.get().getType());
        if (type) {
            auto arity = getArity(type, targetLayout);
            if (succeeded(arity) && *arity > 0) {
                values[operand.get()] = graph.createInputs(static_cast<unsigned>(*arity));
                layouts[operand.get()] = targetLayout;
            }
        } else if (isa<VMIMaskType>(operand.get().getType())) {
            values[operand.get()] = graph.createInputs(1);
            layouts[operand.get()] = targetLayout;
        }
    }
}
FailureOr<PhysicalValues> PlanGraphBuilder::buildMaskOperand(
    OpOperand& operand, ArrayRef<PhysicalValue> source, VMILayoutAttr layout, VMILayoutAttr targetLayout)
{
    if (layout != targetLayout) {
        VMILayoutSupport support;
        auto mask = cast<VMIMaskType>(operand.get().getType());
        auto sourceType = VMIMaskType::get(mask.getContext(), mask.getElementCount(), mask.getGranularity(), layout);
        auto targetType =
            VMIMaskType::get(mask.getContext(), mask.getElementCount(), mask.getGranularity(), targetLayout);
        auto ensureFact = support.getEnsureMaskLayoutFact(sourceType, targetType);
        if (failed(ensureFact)) {
            return failure();
        }
        if (ensureFact->forwardsPhysicalParts) {
            return PhysicalValues(source);
        }
        return materializeMaskLayout(source, sourceType, layout, targetLayout, operand.getOwner());
    }
    return PhysicalValues(source);
}
FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::buildOperand(
    OpOperand& operand, VMILayoutAttr targetLayout)
{
    seedStructuralOperand(operand, targetLayout);
    auto source = buildValue(operand.get());
    auto sourceLayout = plan.valueLayouts.find(operand.get());
    VMILayoutAttr layout =
        sourceLayout == plan.valueLayouts.end() ? layouts.lookup(operand.get()) : sourceLayout->second;
    if (!layout && isa<VMIVRegType, VMIMaskType>(operand.get().getType())) {
        // Structural transport (block arguments and function boundaries) is
        // constrained by the assignment propagator rather than a VMI relation.
        // For an unassigned internal transport value, the consuming relation's
        // required layout is the only legal physical realization available to
        // the cost graph.
        layout = targetLayout;
    }
    if (failed(source) || !layout) {
        return failure();
    }
    if (isa<VMIMaskType>(operand.get().getType())) {
        return buildMaskOperand(operand, *source, layout, targetLayout);
    }
    if (layout != targetLayout) {
        Block* block = operand.getOwner()->getBlock();
        for (const PhysicalStateStorage::SharedMaterialization& shared : sharedMaterializations) {
            bool sameMaterialization = shared.source == operand.get() && shared.sourceLayout == layout &&
                                       shared.resultLayout == targetLayout && shared.block == block;
            if (sameMaterialization) {
                return shared.result;
            }
        }
        auto result = materialize(operand.get(), *source, layout, targetLayout, operand.getOwner());
        if (failed(result)) {
            return failure();
        }
        sharedMaterializations.push_back({operand.get(), layout, targetLayout, block, *result});
        return result;
    }
    auto result = materialize(operand.get(), *source, layout, targetLayout, operand.getOwner());
    return result;
}

LogicalResult PlanGraphBuilder::buildRelationEndpoints(Operation* op, const VMILayoutOpRelation& relation)
{
    for (const VMILayoutRelationEndpoint& endpoint : relation.endpoints) {
        if (!endpoint.value || !endpoint.layout) {
            return failure();
        }
        if (endpoint.use) {
            if (endpoint.use->getOwner() != op || failed(buildOperand(*endpoint.use, endpoint.layout))) {
                return failure();
            }
            continue;
        }
        layouts[endpoint.value] = endpoint.layout;
        if (endpoint.value.getDefiningOp() != op || values.contains(endpoint.value)) {
            continue;
        }
        if (auto type = dyn_cast<VMIVRegType>(endpoint.value.getType())) {
            auto arity = getArity(type, endpoint.layout);
            if (failed(arity) || *arity <= 0) {
                return failure();
            }
            values[endpoint.value] = graph.createInputs(static_cast<unsigned>(*arity));
            continue;
        }
        if (isa<VMIMaskType>(endpoint.value.getType())) {
            values[endpoint.value] = graph.createInputs(1);
        }
    }
    return success();
}
LogicalResult PlanGraphBuilder::buildOp(Operation* op)
{
    if (builtOps.contains(op)) {
        return success();
    }
    if (!buildingOps.insert(op).second) {
        return failure();
    }
    auto relationIt = selectedRelations.find(op);
    if (relationIt == selectedRelations.end()) {
        return failure();
    }
    const VMILayoutOpRelation& relation = relationIt->second;
    if (relation.ports.empty() && !relation.endpoints.empty()) {
        if (failed(buildRelationEndpoints(op, relation))) {
            return failure();
        }
        buildingOps.erase(op);
        builtOps.insert(op);
        return success();
    }
    LogicalResult result = buildRelation(relation);
    buildingOps.erase(op);
    if (succeeded(result)) {
        if (failed(graph.addIntrinsicRearrangementCost(relation.intrinsicRearrangementCost, getDynamicScope(op)))) {
            return failure();
        }
        builtOps.insert(op);
    }
    return result;
}

} // namespace mlir::pto::layout_cost
