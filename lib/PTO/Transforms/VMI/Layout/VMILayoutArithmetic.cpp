// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Layout relation enumeration: arithmetic.
#include "VMILayoutRelations.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_relations {

RelationResult VMILayoutRelations::buildReduce()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() < 2 || op->getNumResults() != 1) {
        return failure();
    }
    auto fact = supports.getReduceLayoutFactForLayouts(
        VMIVRegType::get(
            op->getContext(), cast<VMIVRegType>(op->getOperand(0).getType()).getElementCount(),
            cast<VMIVRegType>(op->getOperand(0).getType()).getElementType(), contiguous),
        VMIMaskType::get(
            op->getContext(), cast<VMIMaskType>(op->getOperand(1).getType()).getElementCount(),
            cast<VMIMaskType>(op->getOperand(1).getType()).getGranularity(), contiguous),
        VMIVRegType::get(
            op->getContext(), cast<VMIVRegType>(op->getResult(0).getType()).getElementCount(),
            cast<VMIVRegType>(op->getResult(0).getType()).getElementType(), contiguous));
    if (failed(fact)) {
        return failure();
    }
    VMILayoutOpRelation relation{
        op,
        {operandPort(0, fact->sourceLayout), operandPort(1, fact->maskLayout), resultPort(0, fact->resultLayout)},
        false};
    return makeReachableRelation(std::move(relation), supports);
}

static void appendInterleaveFacts(
    Operation* op, ArrayRef<VMIInterleaveLayoutFact> facts, const FailureOr<VMIInterleaveLayoutFact>& preferred,
    const VMILayoutSupport& supports, Relations& relations)
{
    for (const VMIInterleaveLayoutFact& fact : facts) {
        auto relation = VMILayoutOpRelation{
            op,
            {operandPort(0, fact.lhsLayout), operandPort(1, fact.rhsLayout), operandPort(2, fact.maskLayout),
             resultPort(0, fact.lowLayout), resultPort(1, fact.highLayout)},
            true,
            0,

            succeeded(preferred) && haveSameInterleaveLayouts(fact, *preferred) ? 0U : 1U};
        auto reachable = makeReachableRelation(std::move(relation), supports);
        if (succeeded(reachable)) {
            for (VMILayoutOpRelation& reachableRelation : *reachable) {
                appendUniqueRelation(relations, std::move(reachableRelation));
            }
        }
    }
}

template <typename OpT>
static RelationResult enumerateInterleave(OpT interleave, bool deinterleave, const VMILayoutSupport& supports)
{
    Operation* op = interleave.getOperation();
    Relations relations;

    auto lhs = dyn_cast<VMIVRegType>(interleave.getLhs().getType());
    auto rhs = dyn_cast<VMIVRegType>(interleave.getRhs().getType());
    auto mask = dyn_cast<VMIMaskType>(interleave.getMask().getType());
    auto low = dyn_cast<VMIVRegType>(interleave.getLow().getType());
    auto high = dyn_cast<VMIVRegType>(interleave.getHigh().getType());
    if (!lhs || !rhs || !mask || !low || !high) {
        return failure();
    }
    if (lhs.getLayoutAttr() && rhs.getLayoutAttr() && mask.getLayoutAttr() && low.getLayoutAttr() &&
        high.getLayoutAttr()) {
        auto fact = deinterleave ? supports.getVdintlvLayoutFactForLayouts(lhs, rhs, mask, low, high) :
                                   supports.getVintlvLayoutFactForLayouts(lhs, rhs, mask, low, high);
        if (failed(fact)) {
            return failure();
        }
        return makeReachableRelation(
            {op,
             {operandPort(0, fact->lhsLayout), operandPort(1, fact->rhsLayout), operandPort(2, fact->maskLayout),
              resultPort(0, fact->lowLayout), resultPort(1, fact->highLayout)},
             true},
            supports);
    }
    auto facts = deinterleave ? supports.getVdintlvLayoutFacts(lhs) : supports.getVintlvLayoutFacts(lhs);
    if (failed(facts)) {
        return failure();
    }
    auto preferred =
        deinterleave ? supports.getPreferredVdintlvLayoutFact(lhs) : supports.getPreferredVintlvLayoutFact(lhs);
    appendInterleaveFacts(op, *facts, preferred, supports, relations);
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, 4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, 4>>(std::move(relations));
}

RelationResult VMILayoutRelations::buildInterleave(VMIVintlvOp interleave)
{
    return enumerateInterleave(interleave, false, supports);
}

RelationResult VMILayoutRelations::buildDeinterleave(VMIVdintlvOp interleave)
{
    return enumerateInterleave(interleave, true, supports);
}

RelationResult VMILayoutRelations::buildChannelSplit(VMIChannelSplitOp split)
{
    auto source = dyn_cast<VMIVRegType>(split.getSource().getType());
    if (!source || (split.getNumResults() != 2 && split.getNumResults() != 4)) {
        return failure();
    }
    VMILayoutAttr sourceLayout = VMILayoutAttr::getDeinterleaved(op->getContext(), split.getNumResults());
    for (OpResult result : split.getResults()) {
        if (auto explicitLayout = cast<VMIVRegType>(result.getType()).getLayoutAttr();
            explicitLayout && explicitLayout != VMILayoutAttr::getContiguous(op->getContext())) {
            return failure();
        }
    }
    SmallVector<VMILayoutPortAssignment, mlir::pto::kValue4> ports;
    ports.push_back(operandPort(0, sourceLayout));
    VMILayoutAttr resultLayout = VMILayoutAttr::getContiguous(op->getContext());
    for (unsigned i = 0; i < split.getNumResults(); ++i) {
        ports.push_back(resultPort(i, resultLayout));
    }
    VMILayoutOpRelation relation{op, std::move(ports), true};
    return makeReachableRelation(std::move(relation), supports);
}

RelationResult VMILayoutRelations::buildChannelMerge(VMIChannelMergeOp merge)
{
    auto result = dyn_cast<VMIVRegType>(merge.getResult().getType());
    unsigned channels = merge.getInputs().size();
    if (!result || (channels != 2 && channels != 4)) {
        return failure();
    }
    VMILayoutAttr inputLayout = VMILayoutAttr::getContiguous(op->getContext());
    VMILayoutAttr resultLayout = VMILayoutAttr::getDeinterleaved(op->getContext(), channels);
    SmallVector<VMILayoutPortAssignment, mlir::pto::kValue4> ports;
    for (unsigned i = 0; i < channels; ++i) {
        ports.push_back(operandPort(i, inputLayout));
    }
    ports.push_back(resultPort(0, resultLayout));
    VMILayoutOpRelation relation{op, std::move(ports), true};
    return makeReachableRelation(std::move(relation), supports);
}

RelationResult VMILayoutRelations::buildShuffle(VMIShuffleOp shuffle)
{
    auto source = dyn_cast<VMIVRegType>(shuffle.getSource().getType());
    auto result = dyn_cast<VMIVRegType>(shuffle.getResult().getType());
    if (!source || !result) {
        return failure();
    }
    VMILayoutAttr layout = source.getLayoutAttr();
    if (!layout) {
        layout = VMILayoutAttr::getContiguous(op->getContext());
    }
    if (auto explicitResult = result.getLayoutAttr(); explicitResult && explicitResult != layout) {
        return failure();
    }
    VMILayoutOpRelation relation{op, {operandPort(0, layout), resultPort(0, layout)}, false};
    return makeReachableRelation(std::move(relation), supports);
}

RelationResult VMILayoutRelations::buildHistogram()
{
    if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
        return failure();
    }
    auto acc = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto source = dyn_cast<VMIVRegType>(op->getOperand(1).getType());
    auto mask = dyn_cast<VMIMaskType>(op->getOperand(2).getType());
    auto result = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!acc || !source || !mask || !result) {
        return failure();
    }
    bool assigned = acc.getLayoutAttr() && source.getLayoutAttr() && mask.getLayoutAttr() && result.getLayoutAttr();
    FailureOr<VMIHistogramLayoutFact> fact =
        isa<VMIVdhistOp>(op) ? (assigned ? supports.getVdhistLayoutFact(cast<VMIVdhistOp>(op)) :
                                           supports.getPreferredVdhistLayoutFact(cast<VMIVdhistOp>(op))) :
                               (assigned ? supports.getVchistLayoutFact(cast<VMIVchistOp>(op)) :
                                           supports.getPreferredVchistLayoutFact(cast<VMIVchistOp>(op)));
    if (failed(fact)) {
        return failure();
    }
    appendReachableUniqueRelation(
        relations,
        VMILayoutOpRelation{
            op,
            {operandPort(0, fact->accLayout), operandPort(1, fact->sourceLayout), operandPort(2, fact->maskLayout),
             resultPort(0, fact->resultLayout)},
            false},
        supports);
    return relations;
}

RelationResult VMILayoutRelations::buildSelect(VMIVselrOp vselr)
{
    auto sourceType = cast<VMIVRegType>(vselr.getSource().getType());
    auto indexType = cast<VMIVRegType>(vselr.getIndex().getType());
    auto resultType = cast<VMIVRegType>(vselr.getResult().getType());
    bool assigned = sourceType.getLayoutAttr() && indexType.getLayoutAttr() && resultType.getLayoutAttr();
    auto fact =
        assigned ? supports.getVselrLayoutFact(vselr, reason) : supports.getPreferredVselrLayoutFact(vselr, reason);
    if (failed(fact)) {
        return failure();
    }
    VMILayoutOpRelation relation{
        op,
        {operandPort(0, fact->sourceLayout), operandPort(1, fact->indexLayout), resultPort(0, fact->resultLayout)},
        false};
    return makeReachableRelation(std::move(relation), supports);
}

SmallVector<VMILayoutAttr, kValue4> VMILayoutRelations::getProducerLayouts()
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> producerLayouts;
    if (auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType()); resultType && resultType.getLayoutAttr()) {
        producerLayouts.push_back(resultType.getLayoutAttr());
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
        if (layout && !llvm::is_contained(producerLayouts, layout)) {
            producerLayouts.push_back(layout);
        }
    }
    if (auto broadcast = dyn_cast<VMIBroadcastOp>(op)) {
        auto sourceType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
        if (sourceType) {
            for (Operation* user : broadcast.getResult().getUsers()) {
                auto groupBroadcast = dyn_cast<VMIGroupBroadcastOp>(user);
                if (!groupBroadcast || groupBroadcast.getSource() != broadcast.getResult()) {
                    continue;
                }
                auto resultType = dyn_cast<VMIVRegType>(groupBroadcast.getResult().getType());
                if (!resultType) {
                    continue;
                }
                int64_t groups = groupBroadcast.getNumGroupsAttr().getInt();
                VMILayoutAttr slots = VMILayoutAttr::getGroupSlots(op->getContext(), groups, mlir::pto::kValue8);
                auto facts = VMILayoutSupport().getGroupBroadcastLayoutFactsForLayout(
                    sourceType, resultType, groups, VMIGroupBroadcastLayoutPort::Source, slots);
                if (succeeded(facts) && !facts->empty() &&
                    !llvm::is_contained(producerLayouts, facts->front().sourceLayout)) {
                    producerLayouts.push_back(facts->front().sourceLayout);
                }
            }
        }
    }
    if (producerLayouts.empty()) {
        producerLayouts.push_back(VMILayoutAttr::getContiguous(op->getContext()));
    }
    return producerLayouts;
}
RelationResult VMILayoutRelations::buildGroupIota(VMIGroupIotaOp groupIota)
{
    auto resultType = dyn_cast<VMIVRegType>(groupIota.getResult().getType());
    if (!resultType) {
        return failure();
    }
    auto facts = supports.getGroupIotaLayoutFacts(resultType);
    if (failed(facts)) {
        return failure();
    }
    for (const VMIGroupIotaLayoutFact& fact : *facts) {
        appendReachableUniqueRelation(
            relations, VMILayoutOpRelation{op, {resultPort(0, fact.resultLayout)}, true}, supports);
    }
    return relations;
}
RelationResult VMILayoutRelations::buildProducer()
{
    if (auto groupIota = dyn_cast<VMIGroupIotaOp>(op)) {
        return buildGroupIota(groupIota);
    }
    auto producerLayouts = getProducerLayouts();
    // pto.vmi.broadcast reads one scalar or one 1-lane vector, and
    // -vmi-lower-unified-to-legacy produces the second form for every vbrc
    // without a group attribute.  Only that second form carries a layout, and a
    // relation has to state every layout-bearing port the band consumes:
    // leaving the source out makes the solver reject the whole component,
    // because a port no relation mentions cannot be validated.
    bool sourceCarriesLayout =
        isa<VMIBroadcastOp>(op) && op->getNumOperands() == 1 && isLayoutType(op->getOperand(0).getType());
    // Such a broadcast reads the source's physical parts and duplicates them
    // (vdup -LOWEST), so consuming the source in the layout it already carries
    // is the conversion-free choice.  A relation cannot state "the source's own
    // layout", so the source port is enumerated over the layouts the source may
    // carry; the cost model charges a conversion whenever a pair picks another
    // source layout, which keeps the conversion-free pair preferred.
    SmallVector<VMILayoutAttr, mlir::pto::kValue8> sourceCandidates;
    if (sourceCarriesLayout) {
        rememberLayout(getVMIExplicitLayout(op->getOperand(0).getType()), sourceCandidates);
        for (VMILayoutAttr candidate : polymorphicLayouts) {
            rememberLayout(candidate, sourceCandidates);
        }
    }
    auto appendProducerRelation = [&](VMILayoutAttr resultLayout, VMILayoutAttr sourceLayout) {
        SmallVector<VMILayoutPortAssignment, mlir::pto::kValue2> ports;
        if (sourceCarriesLayout) {
            ports.push_back(operandPort(0, sourceLayout));
        }
        ports.push_back(resultPort(0, resultLayout));
        appendReachableUniqueRelation(relations, VMILayoutOpRelation{op, std::move(ports), true}, supports);
    };
    for (VMILayoutAttr layout : producerLayouts) {
        // Grouped vci lowering is a contiguous group_iota producer.  A
        // non-contiguous result requires a separate ensure_layout materializer;
        // it is therefore not a direct producer relation and must be represented
        // by an explicit structural op before entering this planner.
        if (auto vci = dyn_cast<VMIVciOp>(op)) {
            auto group = vci.getGroupAttr();
            if (group && group.getInt() > 1 && !layout.isContiguous()) {
                continue;
            }
        }
        if (!sourceCarriesLayout) {
            appendProducerRelation(layout, VMILayoutAttr{});
            continue;
        }
        for (VMILayoutAttr sourceLayout : sourceCandidates) {
            appendProducerRelation(layout, sourceLayout);
        }
    }
    return relations;
}

SmallVector<VMILayoutAttr, kValue4> VMILayoutRelations::getGroupBroadcastCandidates(
    VMIVRegType sourceType, VMIVRegType resultType, int64_t groups)
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidateLayouts;
    candidateLayouts.append(polymorphicLayouts.begin(), polymorphicLayouts.end());
    if (candidateLayouts.empty()) {
        candidateLayouts.push_back(VMILayoutAttr::getContiguous(op->getContext()));
    }
    if (auto preferred = supports.getGroupBroadcastLayoutFactForLayouts(sourceType, resultType, groups);
        succeeded(preferred) && !llvm::is_contained(candidateLayouts, preferred->sourceLayout)) {
        candidateLayouts.push_back(preferred->sourceLayout);
    }
    VMILayoutAttr slots = VMILayoutAttr::getGroupSlots(op->getContext(), groups, mlir::pto::kValue8);
    if (!llvm::is_contained(candidateLayouts, slots)) {
        candidateLayouts.push_back(slots);
    }
    return candidateLayouts;
}
static bool isSupportedBroadcastResult(VMIVRegType resultType, const VMIGroupBroadcastLayoutFact& fact)
{
    if (resultType.getLayoutAttr() && resultType.getLayoutAttr() != fact.resultLayout) {
        return false;
    }
    auto lanes = getDataLanesPerPart(resultType.getElementType());
    return fact.blockClass == VMIGroupBlockClass::Compact || failed(lanes) || resultType.getElementCount() >= *lanes ||
           !fact.resultLayout.isContiguous() || fact.resultLayout.getLaneStride() != 1;
}
RelationResult VMILayoutRelations::buildGroupBroadcast(VMIGroupBroadcastOp broadcast)
{
    auto sourceType = dyn_cast<VMIVRegType>(broadcast.getSource().getType());
    auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
    if (!sourceType || !resultType) {
        return failure();
    }
    int64_t groups = broadcast.getNumGroupsAttr().getInt();
    auto candidateLayouts = getGroupBroadcastCandidates(sourceType, resultType, groups);
    for (VMILayoutAttr layout : candidateLayouts) {
        if (!layout) {
            continue;
        }
        auto facts = supports.getGroupBroadcastLayoutFactsForLayout(
            sourceType, resultType, groups, VMIGroupBroadcastLayoutPort::Source, layout);
        if (failed(facts)) {
            continue;
        }
        for (const VMIGroupBroadcastLayoutFact& fact : *facts) {
            if (!isSupportedBroadcastResult(resultType, fact)) {
                continue;
            }
            appendReachableUniqueRelation(
                relations,
                VMILayoutOpRelation{op, {operandPort(0, fact.sourceLayout), resultPort(0, fact.resultLayout)}, false},
                supports);
        }
    }
    if (relations.empty()) {
        auto preferred = supports.getGroupBroadcastLayoutFactForLayouts(sourceType, resultType, groups);
        if (succeeded(preferred) && isSupportedBroadcastResult(resultType, *preferred)) {
            appendReachableUniqueRelation(
                relations,
                VMILayoutOpRelation{
                    op, {operandPort(0, preferred->sourceLayout), resultPort(0, preferred->resultLayout)}, false},
                supports);
        }
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(std::move(relations));
}

static void appendGroupReduceFact(
    Operation* op, const VMIGroupReduceLayoutFact& fact, const VMILayoutSupport& supports,
    SmallVectorImpl<VMILayoutOpRelation>& relations)
{
    appendReachableUniqueRelation(
        relations,
        VMILayoutOpRelation{
            op,
            {operandPort(0, fact.sourceLayout), operandPort(1, fact.maskLayout), resultPort(0, fact.resultLayout)},
            false},
        supports);
}
static void appendGroupReduceFacts(
    Operation* op, int64_t groups, ArrayRef<VMILayoutAttr> layouts, const VMILayoutSupport& supports,
    SmallVectorImpl<VMILayoutOpRelation>& relations, std::string* reason = nullptr)
{
    auto sourceType = cast<VMIVRegType>(op->getOperand(0).getType());
    auto maskType = cast<VMIMaskType>(op->getOperand(1).getType());
    auto resultType = cast<VMIVRegType>(op->getResult(0).getType());
    for (VMILayoutAttr layout : getGroupReduceQueryLayouts(op->getContext(), layouts)) {
        if (!layout) {
            continue;
        }
        auto facts = supports.getGroupReduceLayoutFactsForLayout(
            getVMIGroupReduceKind(op), sourceType, groups, VMIGroupReduceLayoutPort::Source, layout, reason);
        if (failed(facts)) {
            continue;
        }
        for (const VMIGroupReduceLayoutFact& fact : *facts) {
            if ((sourceType.getLayoutAttr() && sourceType.getLayoutAttr() != fact.sourceLayout) ||
                (maskType.getLayoutAttr() && maskType.getLayoutAttr() != fact.maskLayout) ||
                (resultType.getLayoutAttr() && resultType.getLayoutAttr() != fact.resultLayout)) {
                continue;
            }
            appendGroupReduceFact(op, fact, supports, relations);
        }
    }
}
static void appendPreferredGroupReduce(
    Operation* op, VMIVRegType sourceType, int64_t groups, const VMILayoutSupport& supports,
    SmallVectorImpl<VMILayoutOpRelation>& relations)
{
    auto preferred = supports.getPreferredGroupReduceLayoutFact(getVMIGroupReduceKind(op), sourceType, groups);
    if (succeeded(preferred)) {
        appendGroupReduceFact(op, *preferred, supports, relations);
    }
}
RelationResult VMILayoutRelations::buildGroupReduce()
{
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto maskType = dyn_cast<VMIMaskType>(op->getOperand(1).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !maskType || !resultType || !op->hasAttr("num_groups")) {
        return failure();
    }
    int64_t groups = op->getAttrOfType<IntegerAttr>("num_groups").getInt();
    // The support model words the refusal when the group size has no row, and
    // that wording is what the shape checks report; keep it so the failure can
    // name the cause through reportMissingRelation.
    std::string familyReason;
    appendGroupReduceFacts(op, groups, polymorphicLayouts, supports, relations, &familyReason);
    if (relations.empty()) {
        appendPreferredGroupReduce(op, sourceType, groups, supports, relations);
    }
    bool refusedBySupportModel = relations.empty() && reason && !familyReason.empty();
    if (refusedBySupportModel) {
        *reason = familyReason;
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(std::move(relations));
}

RelationResult VMILayoutRelations::buildLegacyReduce()
{
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto maskType = dyn_cast<VMIMaskType>(op->getOperand(1).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !maskType || !resultType) {
        return failure();
    }
    auto group = op->getAttrOfType<IntegerAttr>("group");
    if (!group) {
        VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
        auto fact = supports.getReduceLayoutFactForLayouts(
            VMIVRegType::get(op->getContext(), sourceType.getElementCount(), sourceType.getElementType(), contiguous),
            VMIMaskType::get(op->getContext(), maskType.getElementCount(), maskType.getGranularity(), contiguous),
            VMIVRegType::get(op->getContext(), resultType.getElementCount(), resultType.getElementType(), contiguous));
        if (failed(fact)) {
            return failure();
        }
        return makeReachableRelation(
            {op,
             {operandPort(0, fact->sourceLayout), operandPort(1, fact->maskLayout), resultPort(0, fact->resultLayout)},
             false},
            supports);
    }
    appendGroupReduceFacts(op, group.getInt(), polymorphicLayouts, supports, relations);
    if (polymorphicLayouts.empty() && relations.empty()) {
        appendPreferredGroupReduce(op, sourceType, group.getInt(), supports, relations);
    }
    return relations;
}

RelationResult VMILayoutRelations::buildVexpdif(VMIVexpdifOp vexpdif)
{
    auto sourceType = dyn_cast<VMIVRegType>(vexpdif.getX().getType());
    auto resultType = dyn_cast<VMIVRegType>(vexpdif.getResult().getType());
    if (!sourceType || !resultType) {
        return failure();
    }

    // pto.vmi.vexpdif reads one physical source register at a time, so the
    // only legal plans are the rows of the vexpdif layout table.  Enumerating
    // any other combination - notably a lane-strided source, whose shared mask
    // layout has to change predicate granularity as well - selects a plan that
    // the conversion to VPTO cannot realize.  The widening form is therefore
    // expressed by the table, not by the generic cast relation table.
    auto preferred = supports.getPreferredVexpdifLayoutFact(vexpdif);
    for (VMILayoutAttr layout : getVexpdifQueryLayouts(op, polymorphicLayouts)) {
        for (VMIVexpdifLayoutPort port : {VMIVexpdifLayoutPort::Source, VMIVexpdifLayoutPort::Result}) {
            auto facts = supports.getVexpdifLayoutFactsForLayout(vexpdif, port, layout);
            if (failed(facts)) {
                continue;
            }
            for (const VMIVexpdifLayoutFact& fact : *facts) {
                uint64_t preferencePenalty = succeeded(preferred) && (fact.sourceLayout != preferred->sourceLayout ||
                                                                      fact.resultLayout != preferred->resultLayout) ?
                                                 1 :
                                                 0;
                // x, max, and the mask always share one layout: the VPTO lowering
                // reads the predicate with the data's lane order.
                appendReachableUniqueRelation(
                    relations,
                    VMILayoutOpRelation{
                        op,
                        {operandPort(0, fact.sourceLayout), operandPort(1, fact.sourceLayout),
                         operandPort(2, fact.sourceLayout), resultPort(0, fact.resultLayout)},
                        false,
                        0,
                        preferencePenalty},
                    supports);
            }
        }
    }
    bool explainMissingRelation = relations.empty() && reason != nullptr;
    if (explainMissingRelation) {
        // Surface the table requirement instead of a generic "no legal relation"
        // when the shape itself has no supported combination.
        (void)supports.getVexpdifLayoutFactsForLayout(
            vexpdif, VMIVexpdifLayoutPort::Source, VMILayoutAttr::getContiguous(op->getContext()), reason);
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(std::move(relations));
}

} // namespace mlir::pto::layout_relations
