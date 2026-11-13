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

#include "VMILayoutComponentGraph.h"
#include "VMILayoutRelations.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"
#include "PTO/Transforms/VMILayoutConflictSolver.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::layout_relations;

#define DEBUG_TYPE "vmi-layout-planner"

namespace {

static std::string getDecisionLayoutKey(VMILayoutAttr layout)
{
    if (!layout) {
        return "<none>";
    }
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << layout;
    return stream.str();
}

static std::string getDecisionRelationKey(const VMILayoutOpRelation& relation)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const VMILayoutPortAssignment& port : relation.ports) {
        stream << (port.kind == VMILayoutPortKind::Operand ? 'o' : 'r') << port.index << ':';
        if (port.kind == VMILayoutPortKind::Operand && port.index < relation.op->getNumOperands()) {
            stream << relation.op->getOperand(port.index).getType();
        } else if (port.kind == VMILayoutPortKind::Result && port.index < relation.op->getNumResults()) {
            stream << relation.op->getResult(port.index).getType();
        } else {
            stream << "<invalid>";
        }
        stream << '=' << getDecisionLayoutKey(port.layout) << ';';
    }
    stream << "direct=" << relation.directProducer << ";rearrange=" << relation.intrinsicRearrangementCost
           << ";preference=" << relation.preferencePenalty;
    return stream.str();
}

static SmallVector<unsigned, kValue4> getDecisionRelationMapping(
    const VMILayoutSolverOp& solverOp, ArrayRef<std::string> relations)
{
    SmallVector<unsigned, mlir::pto::kValue4> mapping;
    for (const std::string& representative : relations) {
        auto relationIt = llvm::find_if(solverOp.relations, [&](const VMILayoutOpRelation& relation) {
            return getDecisionRelationKey(relation) == representative;
        });
        if (relationIt == solverOp.relations.end()) {
            mapping.clear();
            break;
        }
        mapping.push_back(relationIt - solverOp.relations.begin());
    }
    return mapping;
}
static SmallVector<VMILayoutDecisionGroup, mlir::pto::kValue4> buildDecisionGroups(
    ArrayRef<VMILayoutSolverOp> plannerOps)
{
    auto components = getVMILayoutSSAComponents(plannerOps);

    llvm::StringMap<unsigned> classIndices;
    SmallVector<VMILayoutDecisionGroup, mlir::pto::kValue4> groups;
    for (auto [opIndex, solverOp] : llvm::enumerate(plannerOps)) {
        if (solverOp.relations.size() < 2) {
            continue;
        }
        SmallVector<std::string, mlir::pto::kValue4> relations;
        for (const VMILayoutOpRelation& relation : solverOp.relations) {
            relations.push_back(getDecisionRelationKey(relation));
        }
        llvm::sort(relations);
        std::string key;
        llvm::raw_string_ostream stream(key);
        stream << components[opIndex] << '|';
        for (const std::string& relation : relations) {
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
        VMILayoutDecisionGroup& group = groups[groupIndex];
        group.opIndices.push_back(opIndex);
        auto mapping = getDecisionRelationMapping(solverOp, relations);
        if (mapping.size() != relations.size()) {
            group.opIndices.pop_back();
            continue;
        }
        group.memberRelationIndices.push_back(std::move(mapping));
    }
    llvm::erase_if(groups, [](const VMILayoutDecisionGroup& group) { return group.opIndices.size() < 2; });
    return groups;
}

static bool isVMILayoutStructuralOp(Operation* op)
{
    return isa<
        cf::BranchOp, cf::CondBranchOp, cf::SwitchOp, func::CallOp, func::ReturnOp, scf::ConditionOp,
        scf::ExecuteRegionOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp, scf::YieldOp, scf::WhileOp>(op);
}

/// Unified-dialect spellings that -vmi-lower-unified-to-legacy replaces with a
/// legacy op later in the pipeline.  Upstream runs layout assignment before
/// that lowering, and its own constraint walk (collect() in
/// VMILayoutAssignment.cpp) states no relation for these spellings: it knows
/// pto.vmi.load, pto.vmi.store and pto.vmi.extf, not their unified forms.  A
/// component must therefore not fail merely because the relation provider has
/// no case for the unified spelling - the lowering decides the layout from the
/// value it is handed, exactly as it does in upstream's own flow.
static bool isPreLoweringUnifiedOp(Operation* op) { return isa<VMICvtOp, VMIvLoadOp, VMIvStoreOp, VMIVsstbOp>(op); }

static bool isStructuralTransportValue(Value value)
{
    if (!value) {
        return false;
    }
    Operation* owner = value.getDefiningOp();
    if (isa<BlockArgument>(value)) {
        owner = value.getParentBlock()->getParentOp();
    }
    for (Operation* parent = owner; parent; parent = parent->getParentOp()) {
        if (isVMILayoutStructuralOp(parent)) {
            return true;
        }
    }
    return false;
}

static bool isVMILayoutABIBoundaryOp(Operation* op) { return isa<func::CallOp, func::ReturnOp>(op); }

static VMILayoutAttr getFunctionArgumentBoundaryLayout(Value value)
{
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument) {
        return {};
    }
    auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!function || argument.getOwner() != &function.getBody().front() ||
        argument.getArgNumber() >= function.getNumArguments()) {
        return {};
    }
    return getABIBoundaryLayout(function.getArgumentTypes()[argument.getArgNumber()]);
}

static SmallVector<VMILayoutFixedAssignment, mlir::pto::kValue4> collectFixedAssignments(ArrayRef<Operation*> component)
{
    SmallVector<VMILayoutFixedAssignment, mlir::pto::kValue4> assignments;
    DenseSet<Value> seen;
    for (Operation* op : component) {
        for (Value operand : op->getOperands()) {
            VMILayoutAttr layout = getFunctionArgumentBoundaryLayout(operand);
            if (layout && seen.insert(operand).second) {
                assignments.push_back({operand, layout});
            }
        }
    }
    return assignments;
}

static SmallVector<VMILayoutAttr, mlir::pto::kValue4> collectPlanLayouts(const VMILayoutPlan& plan);

// isVMISameLayoutOp comes from VMILayoutPropagation (the single source of truth for
// the same-layout transfer) and isVMILayoutCastOp from VMILayoutOpClasses; the planner
// deliberately defines neither, so a full link has exactly one definition of each.

} // namespace

namespace mlir::pto {

SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue8> VMILayoutStructuralEdgeProvider::enumerateEdges(
    Operation* op) const
{
    SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue8> edges;
    if (!op) {
        return edges;
    }
    // Constraint collection always succeeds: the domain-specific callback only
    // filters values and records the shared control-flow relationships.
    (void)VMIControlFlowSupport::addConstraints(op, [&](Value source, Value destination, Operation*) {
        bool hasLayoutEndpoints =
            source && destination && isLayoutType(source.getType()) && isLayoutType(destination.getType());
        if (hasLayoutEndpoints) {
            edges.push_back({source, destination, false});
        }
        return success();
    });
    return edges;
}

} // namespace mlir::pto

namespace {

static SmallVector<VMILayoutAttr, mlir::pto::kValue4> collectPlanLayouts(const VMILayoutPlan& plan)
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> layouts;
    auto remember = [&](VMILayoutAttr layout) {
        if (layout && !llvm::is_contained(layouts, layout)) {
            layouts.push_back(layout);
        }
    };
    for (const auto& [value, layout] : plan.valueLayouts) {
        remember(layout);
    }
    for (const auto& [operand, layout] : plan.useLayouts) {
        remember(layout);
    }
    return layouts;
}

static bool matchRelationPort(
    const VMILayoutPortAssignment& port, Operation* op, const VMILayoutPlan& plan,
    SmallVectorImpl<bool>& operandCovered, SmallVectorImpl<bool>& resultCovered)
{
    if (port.kind == VMILayoutPortKind::Operand) {
        if (port.index >= op->getNumOperands()) {
            return false;
        }
        if (!isLayoutType(op->getOperand(port.index).getType()) || operandCovered[port.index]) {
            return false;
        }
        operandCovered[port.index] = true;
        auto it = plan.useLayouts.find(&op->getOpOperand(port.index));
        if (it == plan.useLayouts.end() || it->second != port.layout) {
            return false;
        }
        return true;
    }
    if (port.index >= op->getNumResults()) {
        return false;
    }
    if (!isLayoutType(op->getResult(port.index).getType()) || resultCovered[port.index]) {
        return false;
    }
    resultCovered[port.index] = true;
    auto it = plan.valueLayouts.find(op->getResult(port.index));
    if (it == plan.valueLayouts.end() || it->second != port.layout) {
        return false;
    }
    return true;
}

static bool relationMatchesPlan(const VMILayoutOpRelation& relation, const VMILayoutPlan& plan)
{
    Operation* op = relation.op;
    if (!op) {
        return false;
    }
    if (relation.ports.empty() && !relation.endpoints.empty()) {
        return llvm::all_of(relation.endpoints, [&](const VMILayoutRelationEndpoint& endpoint) {
            if (!endpoint.value || !endpoint.layout) {
                return false;
            }
            if (endpoint.use) {
                auto it = plan.useLayouts.find(endpoint.use);
                return it != plan.useLayouts.end() && it->second == endpoint.layout;
            }
            auto it = plan.valueLayouts.find(endpoint.value);
            return it != plan.valueLayouts.end() && it->second == endpoint.layout;
        });
    }
    SmallVector<bool, mlir::pto::kValue4> operandCovered(op->getNumOperands(), false);
    SmallVector<bool, mlir::pto::kValue4> resultCovered(op->getNumResults(), false);
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (!matchRelationPort(port, op, plan, operandCovered, resultCovered)) {
            return false;
        }
    }
    for (unsigned index = 0; index < op->getNumOperands(); ++index) {
        if (isLayoutType(op->getOperand(index).getType()) && !operandCovered[index]) {
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

static LogicalResult verifySelectedRelations(const VMILayoutPlan& plan, ArrayRef<VMILayoutAttr> layouts)
{
    VMILayoutRelationProvider provider;
    for (const auto& selectedRelation : plan.selectedRelations) {
        Operation* op = selectedRelation.first;
        if (!op) {
            return failure();
        }
        FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>> relations =
            provider.enumerateRelations(op, layouts);
        if (failed(relations) || llvm::none_of(*relations, [&](const VMILayoutOpRelation& relation) {
                return relationMatchesPlan(relation, plan);
            })) {
            return failure();
        }
    }
    return success();
}

static SmallVector<Operation*, mlir::pto::kValue16> getPlanInstallOrder(const VMILayoutPlan& plan)
{
    SmallVector<Operation*, mlir::pto::kValue16> orderedOps;
    for (const auto& [op, relation] : plan.selectedRelations) {
        (void)relation;
        if (op) {
            orderedOps.push_back(op);
        }
    }
    llvm::sort(orderedOps, [](Operation* lhs, Operation* rhs) {
        if (lhs == rhs) {
            return false;
        }
        if (lhs->getBlock() == rhs->getBlock()) {
            return lhs->isBeforeInBlock(rhs);
        }
        return lhs->getBlock() < rhs->getBlock();
    });

    return orderedOps;
}

// Installs the validated plan in producer/use order without invoking legacy propagation.
class VMILayoutPlanner {
public:
    VMILayoutPlanner(const VMILayoutPlan& plan, VMILayoutPropagator& propagator) : plan(plan), propagator(propagator) {}
    LogicalResult install()
    {
        for (Operation* op : getPlanInstallOrder(plan)) {
            for (OpOperand& operand : op->getOpOperands()) {
                if (failed(installOperand(operand))) {
                    return failure();
                }
            }
            for (OpResult result : op->getResults()) {
                if (failed(installResult(result))) {
                    return failure();
                }
            }
        }
        return installRemaining();
    }

private:
    LogicalResult installOperand(OpOperand& operand)
    {
        Operation* op = operand.getOwner();

        auto it = plan.useLayouts.find(&operand);
        if (it == plan.useLayouts.end()) {
            return success();
        }
        Value source = operand.get();
        auto valueIt = plan.valueLayouts.find(source);
        VMILayoutAttr sourceLayout =
            valueIt != plan.valueLayouts.end() ? valueIt->second : getVMIExplicitLayout(source.getType());
        if (!sourceLayout && isStructuralTransportValue(source)) {
            // Structural transport values (block arguments, call results, and
            // SCF region results) have no producer-owned physical layout before
            // assignment.  Seed their primary layout from the validated use
            // relation; conversions at other uses remain explicit conflicts.
            sourceLayout = isa<BlockArgument>(source) ? VMILayoutAttr::getContiguous(source.getContext()) : it->second;
        }
        if (sourceLayout && installedValues.insert(source).second &&
            failed(propagator.installPlanned(source, sourceLayout))) {
            op->emitError() << kVMIDiagLayoutContractPrefix << "failed to install source layout for operand #"
                            << operand.getOperandNumber();
            return failure();
        }
        if (failed(propagator.installPlanned(operand, it->second))) {
            op->emitError() << kVMIDiagLayoutContractPrefix << "failed to install operand layout for operand #"
                            << operand.getOperandNumber();
            return failure();
        }
        installedUses.insert(&operand);

        return success();
    }
    LogicalResult installResult(OpResult result)
    {
        Operation* op = result.getOwner();

        auto it = plan.valueLayouts.find(result);
        if (it == plan.valueLayouts.end()) {
            return success();
        }
        if (installedValues.insert(result).second && failed(propagator.installPlanned(result, it->second))) {
            op->emitError() << kVMIDiagLayoutContractPrefix << "failed to install result layout #"
                            << result.getResultNumber();
            return failure();
        }

        return success();
    }
    LogicalResult installRemaining()
    {
        for (const auto& [value, layout] : plan.valueLayouts) {
            if (installedValues.insert(value).second && failed(propagator.installPlanned(value, layout))) {
                return failure();
            }
        }
        for (const auto& [operand, layout] : plan.useLayouts) {
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
    const VMILayoutPlan& plan;
    VMILayoutPropagator& propagator;
    DenseSet<Value> installedValues;
    DenseSet<OpOperand*> installedUses;
};
static LogicalResult requestPlan(const VMILayoutPlan& plan, VMILayoutPropagator& propagator)
{
    return VMILayoutPlanner(plan, propagator).install();
}

static bool planWasCommitted(const VMILayoutPlan& plan, const VMILayoutPropagator& propagator)
{
    for (const auto& [value, layout] : plan.valueLayouts) {
        if (propagator.getRequestedOrCurrentLayout(value) != layout) {
            return false;
        }
    }
    for (const auto& [operand, layout] : plan.useLayouts) {
        if (!operand || propagator.getRequestedLayout(*operand) != layout) {
            return false;
        }
    }
    return true;
}

static bool isLayoutValue(Value value) { return value && isLayoutType(value.getType()); }

static void rememberRelationLayouts(ArrayRef<VMILayoutOpRelation> relations, SmallVectorImpl<VMILayoutAttr>& layouts)
{
    for (const VMILayoutOpRelation& relation : relations) {
        for (const VMILayoutPortAssignment& port : relation.ports) {
            if (port.layout && !llvm::is_contained(layouts, port.layout)) {
                layouts.push_back(port.layout);
            }
        }
        for (const VMILayoutRelationEndpoint& endpoint : relation.endpoints) {
            if (endpoint.layout && !llvm::is_contained(layouts, endpoint.layout)) {
                layouts.push_back(endpoint.layout);
            }
        }
    }
}

static std::optional<int64_t> getGroupReduceNumGroups(Operation* op)
{
    if (isa<VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
            VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(op)) {
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

static void rememberGroupReduceRelationLayouts(
    VMIGroupReduceKind kind, VMIVRegType sourceType, int64_t numGroups, SmallVectorImpl<VMILayoutAttr>& layouts)
{
    VMILayoutSupport supports;
    for (VMILayoutAttr candidate : getGroupReduceQueryLayouts(sourceType.getContext(), layouts)) {
        auto facts = supports.getGroupReduceLayoutFactsForLayout(
            kind, sourceType, numGroups, VMIGroupReduceLayoutPort::Source, candidate);
        if (failed(facts)) {
            continue;
        }
        for (const VMIGroupReduceLayoutFact& fact : *facts) {
            rememberLayout(fact.sourceLayout, layouts);
            rememberLayout(fact.maskLayout, layouts);
            rememberLayout(fact.resultLayout, layouts);
        }
    }
}

/// A legacy group reduction is refused by the support model rather than by this
/// planner's enumeration: the group size has no row in its layout table.  The
/// shape checks word that refusal with the table's own reason, so the family is
/// reported the same way instead of with the generic sentence below.
static bool isLegacyGroupReduceOp(Operation* op)
{
    return isa<
        VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp, VMIGroupReduceAddIOp, VMIGroupReduceMaxIOp,
        VMIGroupReduceMinIOp>(op);
}

/// Reports an op for which no legal layout relation could be enumerated.  The
/// support model usually knows exactly why (a shape table row, an alignment
/// rule, or a lowering capability), so surface that reason when there is one
/// instead of a generic sentence that hides it.
static void reportMissingRelation(Operation* op, const std::string& reason)
{
    if (!reason.empty()) {
        if (isLegacyGroupReduceOp(op)) {
            op->emitError() << kVMIDiagLayoutContractPrefix << op->getName()
                            << " has no registered group_slots layout support: " << reason;
            return;
        }
        op->emitError() << kVMIDiagUnsupportedPrefix << reason;
        return;
    }
    op->emitError() << kVMIDiagUnsupportedPrefix << "no legal VMI layout relation is registered for " << op->getName();
}

static FailureOr<SmallVector<VMILayoutSolverOp, mlir::pto::kValue8>> buildPlannerOps(ArrayRef<Operation*> ops)
{
    VMILayoutRelationProvider provider;
    SmallVector<VMILayoutAttr, mlir::pto::kValue8> layouts;
    SmallVector<VMILayoutSolverOp, mlir::pto::kValue8> plannerOps;
    plannerOps.reserve(ops.size());

    // A component whose every op is either a structural transport or a unified
    // spelling that -vmi-lower-unified-to-legacy replaces after layout assignment
    // carries no layout decision at all: upstream's own flow never constrained
    // those spellings, so the lowering owns them and no plan is emitted.
    bool needsRelations = false;
    for (Operation* op : ops) {
        if (isPreLoweringUnifiedOp(op)) {
            continue;
        }
        if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
            // A function boundary contributes a relation only for the results whose
            // type spells a layout out; every other result is decided by the value
            // the function returns.
            auto function = returnOp->getParentOfType<func::FuncOp>();
            needsRelations |= !function || returnOp.getNumOperands() != function.getNumResults();
            if (!needsRelations) {
                needsRelations = llvm::any_of(
                    llvm::zip_equal(function.getResultTypes(), returnOp.getOperands()),
                    [](auto pair) { return static_cast<bool>(getVMIExplicitLayout(std::get<0>(pair))); });
            }
            continue;
        }
        if (isVMILayoutABIBoundaryOp(op) || !isVMILayoutStructuralOp(op)) {
            needsRelations = true;
            break;
        }
    }
    if (!needsRelations) {
        return plannerOps;
    }

    // Collect operation-induced layout candidates before enumerating any
    // relation.  A helper such as ensure_mask_granularity may appear before its
    // consumer in IR order, while the consumer is what introduces the useful
    // physical layout (for example deinterleaved=4 for partial group-reduce).
    // A single forward pass would permanently under-constrain that helper.
    for (Operation* op : ops) {
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
        auto sourceType =
            op->getNumOperands() == 0 ? VMIVRegType{} : dyn_cast<VMIVRegType>(op->getOperand(0).getType());
        if (groups && sourceType) {
            rememberGroupReduceRelationLayouts(getVMIGroupReduceKind(op), sourceType, *groups, layouts);
        }
    }
    for (Operation* op : ops) {
        if (isVMILayoutStructuralOp(op)) {
            continue;
        }
        for (Type type : op->getOperandTypes()) {
            VMILayoutAttr layout = getVMIExplicitLayout(type);
            if (layout && !llvm::is_contained(layouts, layout)) {
                layouts.push_back(layout);
            }
        }
        for (Type type : op->getResultTypes()) {
            VMILayoutAttr layout = getVMIExplicitLayout(type);
            if (layout && !llvm::is_contained(layouts, layout)) {
                layouts.push_back(layout);
            }
        }
        if (auto interleave = dyn_cast<VMIVintlvOp>(op)) {
            VMILayoutSupport supports;
            if (auto preferred =
                    supports.getPreferredVintlvLayoutFact(cast<VMIVRegType>(interleave.getLow().getType()));
                succeeded(preferred) && !llvm::is_contained(layouts, preferred->lhsLayout)) {
                layouts.push_back(preferred->lhsLayout);
            }
        } else if (auto interleave = dyn_cast<VMIVdintlvOp>(op)) {
            VMILayoutSupport supports;
            if (auto preferred =
                    supports.getPreferredVdintlvLayoutFact(cast<VMIVRegType>(interleave.getLow().getType()));
                succeeded(preferred) && !llvm::is_contained(layouts, preferred->lhsLayout)) {
                layouts.push_back(preferred->lhsLayout);
            }
        }
        std::string relationReason;
        auto relations = provider.enumerateRelations(op, layouts, &relationReason);
        if (failed(relations) && isPreLoweringUnifiedOp(op)) {
            continue;
        }
        if (failed(relations) || relations->empty()) {
            VMILayoutSupport supports;
            if (auto groupSlot = dyn_cast<VMIGroupSlotLoadOp>(op)) {
                auto resultType = dyn_cast<VMIVRegType>(groupSlot.getResult().getType());
                if (resultType) {
                    VMILayoutAttr layout = VMILayoutAttr::getGroupSlots(
                        op->getContext(), groupSlot.getNumGroupsAttr().getInt(), groupSlot.getNumGroupsAttr().getInt());
                    auto typed = VMIVRegType::get(
                        resultType.getContext(), resultType.getElementCount(), resultType.getElementType(), layout);
                    auto fact = supports.getGroupSlotLoadLayoutFact(
                        typed, groupSlot.getSourceGroupStride(), groupSlot.getNumGroupsAttr().getInt());
                    if (succeeded(fact)) {
                        layouts.push_back(layout);
                        continue;
                    }
                }
            }
            if (auto groupLoad = dyn_cast<VMIGroupLoadOp>(op)) {
                auto resultType = dyn_cast<VMIVRegType>(groupLoad.getResult().getType());
                if (resultType) {
                    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
                    auto typed = VMIVRegType::get(
                        resultType.getContext(), resultType.getElementCount(), resultType.getElementType(), contiguous);
                    auto fact = supports.getGroupLoadLayoutFact(
                        typed, groupLoad.getRowStride(), groupLoad.getNumGroupsAttr().getInt());
                    if (succeeded(fact)) {
                        layouts.push_back(fact->resultLayout);
                        continue;
                    }
                    int64_t groups = groupLoad.getNumGroupsAttr().getInt();
                    if (groups > 0 && resultType.getElementCount() % groups == 0) {
                        int64_t groupSize = resultType.getElementCount() / groups;
                        bool foundBlockLayout = false;
                        for (int64_t factor : {int64_t(2), int64_t(4)}) {
                            if ((groupSize == 16 && factor != 2) || (groupSize == 32 && factor != 4)) {
                                continue;
                            }
                            VMILayoutAttr block = VMILayoutAttr::getBlockDeinterleaved(op->getContext(), factor);
                            auto typedBlock = VMIVRegType::get(
                                resultType.getContext(), resultType.getElementCount(), resultType.getElementType(),
                                block);
                            auto blockFact =
                                supports.getGroupLoadLayoutFact(typedBlock, groupLoad.getRowStride(), groups);
                            if (succeeded(blockFact)) {
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
                auto fact = supports.getGroupBroadcastLayoutFactForLayouts(sourceType, resultType, groups);
                if (succeeded(fact)) {
                    layouts.push_back(fact->sourceLayout);
                    layouts.push_back(fact->resultLayout);
                    continue;
                }
                VMILayoutAttr slots = VMILayoutAttr::getGroupSlots(op->getContext(), groups, mlir::pto::kValue8);
                auto facts = supports.getGroupBroadcastLayoutFactsForLayout(
                    sourceType, resultType, groups, VMIGroupBroadcastLayoutPort::Source, slots);
                if (succeeded(facts) && !facts->empty()) {
                    for (const VMIGroupBroadcastLayoutFact& fact : *facts) {
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
    for (Operation* op : ops) {
        if (isVMILayoutStructuralOp(op) && !isVMILayoutABIBoundaryOp(op)) {
            continue;
        }
        std::string relationReason;
        auto relations = provider.enumerateRelations(op, layouts, &relationReason);
        if (succeeded(relations) && relations->empty() && isVMILayoutABIBoundaryOp(op)) {
            // A function boundary whose result types spell out no layout pins no
            // layout either, so it contributes no relation to the component: the
            // value it transports is decided by its producer.
            continue;
        }
        if (failed(relations) && isPreLoweringUnifiedOp(op)) {
            continue;
        }
        if (failed(relations)) {
            auto preferredRelations = provider.enumerateRelations(op, {}, &relationReason);
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
            auto preferredRelations = provider.enumerateRelations(op, {}, &relationReason);
            if (succeeded(preferredRelations) && !preferredRelations->empty()) {
                relations = std::move(preferredRelations);
            }
        }
        if (failed(relations) || relations->empty()) {
            reportMissingRelation(op, relationReason);
            return failure();
        }
        llvm::erase_if(
            *relations, [](const VMILayoutOpRelation& relation) { return !relationRespectsExplicitResults(relation); });
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

static FailureOr<VMILayoutPlan> solveComponent(ArrayRef<Operation*> component, const VMILayoutPlannerOptions& options)
{
    auto ops = buildPlannerOps(component);
    if (failed(ops)) {
        LLVM_DEBUG(llvm::dbgs() << "layout planner: unsupported component\n");
        return failure();
    }
    if (ops->empty()) {
        // No op in the component states a relation, so the component has no layout
        // decision to make and no plan to commit.
        VMILayoutPlan empty;
        return empty;
    }
    LLVM_DEBUG({
        llvm::dbgs() << "layout planner: component with " << ops->size() << " ops\n";
        for (const VMILayoutSolverOp& op : *ops) {
            llvm::dbgs() << "  " << op.op->getName() << ": " << op.relations.size() << " relations\n";
        }
    });
    VMILayoutConflictSolverOptions solverOptions;
    solverOptions.maxFrontierEntries = options.maxFrontierEntriesPerComponent;
    solverOptions.maxTransitions = options.maxTransitionsPerComponent;
    SmallVector<VMILayoutFixedAssignment, mlir::pto::kValue4> fixedAssignments = collectFixedAssignments(component);
    solverOptions.fixedAssignments = fixedAssignments;
    SmallVector<VMILayoutDecisionGroup, mlir::pto::kValue4> decisionGroups = buildDecisionGroups(*ops);
    solverOptions.decisionGroups = decisionGroups;
    VMILayoutStructuralEdgeProvider edgeProvider;
    SmallVector<VMILayoutEqualityConstraint, mlir::pto::kValue16> equalities;
    for (Operation* op : component) {
        auto opEqualities = edgeProvider.enumerateEdges(op);
        equalities.append(opEqualities.begin(), opEqualities.end());
    }
    solverOptions.equalityConstraints = equalities;
    std::string solverReason;
    auto plan = solveVMILayoutConflictComponent(*ops, solverOptions, &solverReason);
    if (failed(plan)) {
        LLVM_DEBUG(llvm::dbgs() << "layout planner: solver failed component ops=" << ops->size() << "\n");
        // A budget-limited search has not proved the component unsatisfiable.
        if (!component.empty()) {
            component.front()->emitError()
                << kVMIDiagLayoutContractPrefix
                << (solverReason.empty() ? "no complete legal VMI layout plan exists for this component" :
                                           solverReason);
        }
        return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "layout planner: solved component ops=" << ops->size() << "\n");
    plan->structuralEqualities.append(equalities);
    return plan;
}

// Candidate layouts for an operation whose layout-bearing ports have to agree.
// The planner normally supplies the domain; when it does not, fall back to the
// layouts spelled on the operation itself so relation enumeration does not
// depend on which caller supplied the domain.

} // namespace

LogicalResult mlir::pto::commitVMILayoutPlan(const VMILayoutPlan& plan, VMILayoutPropagator& propagator)
{
    Operation* anchor = plan.selectedRelations.empty() ? nullptr : plan.selectedRelations.begin()->first;
    auto failCommit = [&](StringRef message) {
        if (anchor) {
            anchor->emitError() << kVMIDiagLayoutContractPrefix << message;
        }
        return failure();
    };
    // Commit primary layouts before use layouts so an intentional use-edge
    // conversion is represented as a conflict, rather than as a producer seed.
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> layouts = collectPlanLayouts(plan);
    if (failed(verifySelectedRelations(plan, layouts))) {
        return failCommit("selected cost plan no longer matches registered layout "
                          "relations");
    }
    for (const VMILayoutEqualityConstraint& edge : plan.structuralEqualities) {
        VMILayoutAttr source = plan.valueLayouts.lookup(edge.source);
        VMILayoutAttr destination = plan.valueLayouts.lookup(edge.destination);
        if (!source || source != destination) {
            return failCommit("selected cost plan violates a control-flow layout "
                              "equality");
        }
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

FailureOr<VMILayoutPlannerResult> mlir::pto::selectCostedVMILayoutPlans(
    Operation* scope, const VMILayoutPlannerOptions& options)
{
    if (!scope || options.maxFrontierEntriesPerComponent == 0 || options.maxTransitionsPerComponent == 0) {
        return failure();
    }
    VMILayoutPlannerResult result;
    VMILayoutComponentGraph graph(scope);
    DenseSet<Value> seenValues;
    SmallVector<Value, mlir::pto::kValue16> seeds;
    scope->walk([&](Operation* op) {
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
        SmallVector<Operation*, mlir::pto::kValue16> component;
        graph.collect(seed, component, seenValues);
        if (component.empty()) {
            continue;
        }
        if (llvm::none_of(component, [](Operation* op) { return !isVMILayoutStructuralOp(op); })) {
            continue;
        }
        auto plan = solveComponent(component, options);
        if (failed(plan)) {
            LLVM_DEBUG(llvm::dbgs() << "layout planner: component has no complete legal plan\n");
            return failure();
        }
        result.plans.push_back(std::move(*plan));
    }
    return result;
}
