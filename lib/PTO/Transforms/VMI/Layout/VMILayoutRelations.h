// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Private relation enumeration helpers shared with planner domain construction.
#ifndef PTO_TRANSFORMS_VMI_VMILAYOUTRELATIONS_H
#define PTO_TRANSFORMS_VMI_VMILAYOUTRELATIONS_H
#include "PTO/Transforms/VMILayoutPlanner.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>
namespace mlir::pto::layout_relations {
using Relations = SmallVector<VMILayoutOpRelation, kValue4>;
using RelationResult = FailureOr<Relations>;
inline VMILayoutPortAssignment operandPort(unsigned index, VMILayoutAttr layout)
{
    return VMILayoutPortAssignment{VMILayoutPortKind::Operand, index, layout};
}

inline VMILayoutPortAssignment resultPort(unsigned index, VMILayoutAttr layout)
{
    return VMILayoutPortAssignment{VMILayoutPortKind::Result, index, layout};
}

// Group-slot packets contain one logical result lane per group.  Rejecting an
// incompatible component-wide candidate here keeps verifier-invalid VMI types
// out of support queries and cost construction.
inline bool isLegalLayoutForType(Type type, VMILayoutAttr layout)
{
    auto vreg = dyn_cast<VMIVRegType>(type);
    return !vreg || !layout || !layout.isGroupSlots() || vreg.getElementCount() == layout.getNumGroups();
}

inline bool isLegalLayoutForValue(Value value, VMILayoutAttr layout)
{
    return value && isLegalLayoutForType(value.getType(), layout);
}

inline bool isLayoutType(Type type) { return isa<VMIVRegType, VMIMaskType>(type); }

inline bool haveSamePortTuple(const VMILayoutOpRelation& lhs, const VMILayoutOpRelation& rhs)
{
    if (lhs.op != rhs.op || lhs.ports.size() != rhs.ports.size() || lhs.directProducer != rhs.directProducer ||
        lhs.intrinsicRearrangementCost != rhs.intrinsicRearrangementCost ||
        lhs.preferencePenalty != rhs.preferencePenalty) {
        return false;
    }
    return llvm::all_of(llvm::zip_equal(lhs.ports, rhs.ports), [](auto ports) {
        const auto& [lhsPort, rhsPort] = ports;
        return lhsPort.kind == rhsPort.kind && lhsPort.index == rhsPort.index && lhsPort.layout == rhsPort.layout;
    });
}

inline bool haveSameInterleaveLayouts(const VMIInterleaveLayoutFact& lhs, const VMIInterleaveLayoutFact& rhs)
{
    return lhs.lhsLayout == rhs.lhsLayout && rhs.rhsLayout == lhs.rhsLayout && lhs.maskLayout == rhs.maskLayout &&
           lhs.lowLayout == rhs.lowLayout && lhs.highLayout == rhs.highLayout;
}

inline void appendUniqueRelation(SmallVectorImpl<VMILayoutOpRelation>& relations, VMILayoutOpRelation relation)
{
    bool unique = llvm::none_of(
        relations, [&](const VMILayoutOpRelation& existing) { return haveSamePortTuple(existing, relation); });
    if (unique) {
        relations.push_back(std::move(relation));
    }
}

inline bool isReachableFromExplicitOperands(const VMILayoutOpRelation& relation, const VMILayoutSupport& supports)
{
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind != VMILayoutPortKind::Operand || port.index >= relation.op->getNumOperands()) {
            continue;
        }
        Type sourceType = relation.op->getOperand(port.index).getType();
        VMILayoutAttr sourceLayout = getVMIExplicitLayout(sourceType);
        if (!sourceLayout || sourceLayout == port.layout) {
            continue;
        }
        if (auto vreg = dyn_cast<VMIVRegType>(sourceType)) {
            auto target =
                VMIVRegType::get(vreg.getContext(), vreg.getElementCount(), vreg.getElementType(), port.layout);
            if (failed(supports.getEnsureLayoutFact(vreg, target))) {
                return false;
            }
            continue;
        }
        if (auto mask = dyn_cast<VMIMaskType>(sourceType)) {
            auto target =
                VMIMaskType::get(mask.getContext(), mask.getElementCount(), mask.getGranularity(), port.layout);
            if (failed(supports.getEnsureMaskLayoutFact(mask, target))) {
                return false;
            }
        }
    }
    return true;
}

inline bool relationRespectsExplicitResults(const VMILayoutOpRelation& relation)
{
    if (!relation.op) {
        return false;
    }
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind != VMILayoutPortKind::Result || port.index >= relation.op->getNumResults()) {
            continue;
        }
        VMILayoutAttr explicitLayout = getVMIExplicitLayout(relation.op->getResult(port.index).getType());
        if (explicitLayout && explicitLayout != port.layout) {
            return false;
        }
    }
    return true;
}

inline void appendReachableUniqueRelation(
    SmallVectorImpl<VMILayoutOpRelation>& relations, VMILayoutOpRelation relation, const VMILayoutSupport& supports)
{
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (!relation.op) {
            return;
        }
        Value value = port.kind == VMILayoutPortKind::Operand ?
                          (port.index < relation.op->getNumOperands() ? relation.op->getOperand(port.index) : Value()) :
                          (port.index < relation.op->getNumResults() ? relation.op->getResult(port.index) : Value());
        if (!isLegalLayoutForValue(value, port.layout)) {
            return;
        }
    }
    if (isReachableFromExplicitOperands(relation, supports) && relationRespectsExplicitResults(relation)) {
        appendUniqueRelation(relations, std::move(relation));
    }
}

inline FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>> makeReachableRelation(
    VMILayoutOpRelation relation, const VMILayoutSupport& supports)
{
    SmallVector<VMILayoutOpRelation, mlir::pto::kValue4> relations;
    appendReachableUniqueRelation(relations, std::move(relation), supports);
    if (relations.empty()) {
        return failure();
    }
    return relations;
}

inline VMILayoutAttr getABIBoundaryLayout(Type type)
{
    if (!isLayoutType(type)) {
        return {};
    }
    if (VMILayoutAttr explicitLayout = getVMIExplicitLayout(type)) {
        return explicitLayout;
    }
    return VMILayoutAttr::getContiguous(type.getContext());
}

inline void rememberLayout(VMILayoutAttr layout, SmallVectorImpl<VMILayoutAttr>& layouts)
{
    if (layout && !llvm::is_contained(layouts, layout)) {
        layouts.push_back(layout);
    }
}

/// Layouts to query the vexpdif facts table with.  The table itself decides
/// which of them form a legal relation, so the query set only has to cover the
/// layouts the table can mention, plus the layouts the operation already
/// spells out (an operation whose ports carry explicit layouts keeps feeding
/// relation enumeration even when no other candidate is available).
inline SmallVector<VMILayoutAttr, mlir::pto::kValue8> getVexpdifQueryLayouts(
    Operation* op, ArrayRef<VMILayoutAttr> seedLayouts)
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue8> layouts;
    for (VMILayoutAttr layout : seedLayouts) {
        rememberLayout(layout, layouts);
    }
    rememberLayout(VMILayoutAttr::getContiguous(op->getContext()), layouts);
    for (int64_t factor : {int64_t(2), int64_t(4)}) {
        rememberLayout(VMILayoutAttr::getDeinterleaved(op->getContext(), factor), layouts);
        rememberLayout(VMILayoutAttr::getBlockDeinterleaved(op->getContext(), factor), layouts);
    }
    for (Type type : op->getOperandTypes()) {
        rememberLayout(getVMIExplicitLayout(type), layouts);
    }
    for (Type type : op->getResultTypes()) {
        rememberLayout(getVMIExplicitLayout(type), layouts);
    }
    return layouts;
}

inline SmallVector<VMILayoutAttr, mlir::pto::kValue8> getGroupReduceQueryLayouts(
    MLIRContext* context, ArrayRef<VMILayoutAttr> seedLayouts)
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue8> layouts;
    for (VMILayoutAttr layout : seedLayouts) {
        rememberLayout(layout, layouts);
    }
    rememberLayout(VMILayoutAttr::getContiguous(context), layouts);
    for (int64_t factor : {int64_t(2), int64_t(4)}) {
        rememberLayout(VMILayoutAttr::getDeinterleaved(context, factor), layouts);
        rememberLayout(VMILayoutAttr::getBlockDeinterleaved(context, factor), layouts);
    }
    return layouts;
}

inline void appendSameLayoutCandidates(
    Operation* op, ArrayRef<VMILayoutAttr> polymorphicLayouts, SmallVectorImpl<VMILayoutAttr>& candidates)
{
    candidates.append(polymorphicLayouts.begin(), polymorphicLayouts.end());
    if (!candidates.empty()) {
        return;
    }
    for (Type type : op->getOperandTypes()) {
        if (VMILayoutAttr layout = getVMIExplicitLayout(type); layout && !llvm::is_contained(candidates, layout)) {
            candidates.push_back(layout);
        }
    }
    for (Type type : op->getResultTypes()) {
        if (VMILayoutAttr layout = getVMIExplicitLayout(type); layout && !llvm::is_contained(candidates, layout)) {
            candidates.push_back(layout);
        }
    }
}

inline void appendSameLayoutRelations(
    Operation* op, ArrayRef<VMILayoutAttr> candidates, const VMILayoutSupport& supports,
    SmallVectorImpl<VMILayoutOpRelation>& relations, std::string* reason = nullptr)
{
    for (VMILayoutAttr layout : candidates) {
        if (!layout) {
            continue;
        }
        if (llvm::any_of(op->getOperandTypes(), [&](Type type) { return !isLegalLayoutForType(type, layout); }) ||
            llvm::any_of(op->getResultTypes(), [&](Type type) { return !isLegalLayoutForType(type, layout); })) {
            continue;
        }
        if (failed(supports.getSameLayoutRelationSupport(op, layout, reason))) {
            continue;
        }
        VMILayoutOpRelation relation;
        relation.op = op;
        for (OpOperand& operand : op->getOpOperands()) {
            if (isLayoutType(operand.get().getType())) {
                relation.ports.push_back(operandPort(operand.getOperandNumber(), layout));
            }
        }
        for (OpResult result : op->getResults()) {
            if (isLayoutType(result.getType())) {
                relation.ports.push_back(resultPort(result.getResultNumber(), layout));
            }
        }
        if (!relation.ports.empty()) {
            appendReachableUniqueRelation(relations, std::move(relation), supports);
        }
    }
}
template <typename FactT>
inline uint64_t getPairPreferencePenalty(const FactT& fact, const FailureOr<FactT>& preferred)
{
    return succeeded(preferred) &&
                   (fact.sourceLayout != preferred->sourceLayout || fact.resultLayout != preferred->resultLayout) ?
               1 :
               0;
}
inline SmallVector<VMILayoutAttr, kValue4> getDenseCandidateLayouts(
    VMILayoutAttr explicitLayout, ArrayRef<VMILayoutAttr> seeds, MLIRContext* context)
{
    SmallVector<VMILayoutAttr, kValue4> layouts;
    rememberLayout(explicitLayout, layouts);
    for (VMILayoutAttr layout : seeds) {
        rememberLayout(layout, layouts);
    }
    rememberLayout(VMILayoutAttr::getContiguous(context), layouts);
    return layouts;
}
inline FailureOr<VMIVRegType> assignCandidateType(VMIVRegType type, VMILayoutAttr layout)
{
    if ((type.getLayoutAttr() && type.getLayoutAttr() != layout) || !isLegalLayoutForType(type, layout)) {
        return failure();
    }
    return VMIVRegType::get(type.getContext(), type.getElementCount(), type.getElementType(), layout);
}
struct CastRelationState {
    VMIVRegType sourceType;
    VMIVRegType resultType;
    bool sameWidthNumericCast;
    FailureOr<VMICastLayoutFact> preferred;
};
class VMILayoutRelations {
public:
    VMILayoutRelations(Operation* op, ArrayRef<VMILayoutAttr> layouts, std::string* reason)
        : op(op), polymorphicLayouts(layouts), reason(reason)
    {}
    RelationResult build();

private:
    void appendSourceCastFacts(const CastRelationState& state, VMIVRegType concreteSourceType);
    void appendAnnotatedCastFacts(const CastRelationState& state);
    void appendResultCastFacts(const CastRelationState& state);
    void appendSameWidthCastFacts(const CastRelationState& state);
    void appendGroupSlotSourceCastFacts(const CastRelationState& state, VMIVRegType sourceType);
    SmallVector<VMILayoutAttr, kValue4> getProducerLayouts();
    SmallVector<VMILayoutAttr, kValue4> getGroupBroadcastCandidates(
        VMIVRegType sourceType, VMIVRegType resultType, int64_t groups);
    std::optional<RelationResult> buildMemoryRelation();

    RelationResult buildReturn(func::ReturnOp returnOp);
    RelationResult buildCall(func::CallOp call);
    /// Whether an enumerated cast fact is a legal *candidate* in the current
    /// lowering state.  Only the dense same-layout BF16 narrowing relation is
    /// state-qualified; every other fact is table-defined.
    bool isAdmissibleCastFact(const VMICastLayoutFact& fact) const;
    void appendCastFact(const VMICastLayoutFact& fact, const FailureOr<VMICastLayoutFact>& preferred);
    /// The state every cast phase threads through: the equal-width-numeric-cast
    /// flag and the (state-vetoed) stateless preferred row.
    CastRelationState makeCastRelationState(VMIVRegType sourceType, VMIVRegType resultType);
    /// The source-side candidate phases, in append order: the source's own
    /// layout rows, the contiguous fallback, and one row set per instantiated
    /// group-slot candidate.
    void appendSourceCandidateFacts(const CastRelationState& state, VMIVRegType sourceType);
    // Ports of one cast relation.  A masked cast gives its predicate operand the
    // same layout as the source data value: the verifier pairs one predicate
    // lane with each source lane in one layout (verifyMaskMatchesData), so the
    // predicate travels with the source through the same layout conversion and
    // is reconciled through vmi.ensure_mask_layout when its producer disagrees.
    SmallVector<VMILayoutPortAssignment, mlir::pto::kValue4> makeCastPorts(
        VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) const;
    RelationResult buildCast();
    RelationResult buildEnsure();
    RelationResult buildMaskGranularity(VMIEnsureMaskGranularityOp ensure);
    RelationResult buildBitcast();
    RelationResult buildLoad(VMILoadOp load);
    RelationResult buildStrideLoad();
    RelationResult buildGather();
    RelationResult buildStrideStore();
    RelationResult buildScatter();
    RelationResult buildCompressStore();
    RelationResult buildActivePrefixIndex();
    RelationResult buildCompress();
    RelationResult buildExpandLoad();
    RelationResult buildReduce();
    RelationResult buildInterleave(VMIVintlvOp interleave);
    RelationResult buildDeinterleave(VMIVdintlvOp interleave);
    RelationResult buildChannelSplit(VMIChannelSplitOp split);
    RelationResult buildChannelMerge(VMIChannelMergeOp merge);
    RelationResult buildShuffle(VMIShuffleOp shuffle);
    RelationResult buildMaskedLoad(VMIMaskedLoadOp load);
    RelationResult buildGroupSlotLoad(VMIGroupSlotLoadOp load);
    RelationResult buildDeinterleaveLoad(VMIDeinterleaveLoadOp load);
    RelationResult buildGroupLoad(VMIGroupLoadOp load);
    RelationResult buildGroupStore(VMIGroupStoreOp store);
    RelationResult buildInterleaveStore(VMIInterleaveStoreOp store);
    RelationResult buildStore(VMIStoreOp store);
    RelationResult buildMaskedStore(VMIMaskedStoreOp store);
    RelationResult buildGroupBroadcastLoad(VMIGroupBroadcastLoadOp load);
    RelationResult buildHistogram();
    RelationResult buildSelect(VMIVselrOp vselr);
    RelationResult buildMaskProducer();
    RelationResult buildGroupIota(VMIGroupIotaOp groupIota);
    RelationResult buildProducer();
    RelationResult buildGroupBroadcast(VMIGroupBroadcastOp broadcast);
    RelationResult buildGroupReduce();
    RelationResult buildLegacyReduce();
    RelationResult buildVexpdif(VMIVexpdifOp vexpdif);
    RelationResult buildSameLayout();
    Operation* op;
    ArrayRef<VMILayoutAttr> polymorphicLayouts;
    std::string* reason;
    VMILayoutSupport supports;
    Relations relations;
    // Operand index of the optional conversion predicate of the cast currently
    // being enumerated; unset for every other operation and for an unmasked
    // cast.  Only buildCast() and the cast helpers read it.
    std::optional<unsigned> castMaskOperand;
};
} // namespace mlir::pto::layout_relations
#endif // PTO_TRANSFORMS_VMI_VMILAYOUTRELATIONS_H
