// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Layout relation enumeration: memory.
#include "VMILayoutRelations.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_relations {

RelationResult VMILayoutRelations::buildLoad(VMILoadOp load)
{
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    FailureOr<SmallVector<VMILoadLayoutFact, mlir::pto::kValue4>> facts = supports.getLoadLayoutFacts(resultType);
    if (failed(facts)) {
        return failure();
    }
    for (const VMILoadLayoutFact& fact : *facts) {
        if (VMILayoutAttr explicitLayout = resultType.getLayoutAttr();
            explicitLayout && explicitLayout != fact.resultLayout) {
            continue;
        }
        appendReachableUniqueRelation(
            relations, VMILayoutOpRelation{op, {resultPort(0, fact.resultLayout)}, true}, supports);
    }
    if (relations.empty()) {
        return failure();
    }
    return relations;
}

RelationResult VMILayoutRelations::buildStrideLoad()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() < 1 || op->getNumResults() != 1) {
        return failure();
    }
    // The mask is the last operand: source, offset, block_stride, mask.  The
    // relation has to constrain the mask, not the block stride, so the index is
    // taken from this dialect's operand list (an out-of-range port makes the
    // solver reject the only relation the op has).
    unsigned maskIndex = op->getNumOperands() - 1;
    if (!isa<VMIMaskType>(op->getOperand(maskIndex).getType())) {
        return failure();
    }
    return makeReachableRelation({op, {operandPort(maskIndex, contiguous), resultPort(0, contiguous)}, true}, supports);
}

RelationResult VMILayoutRelations::buildGather()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() != 4 || op->getNumResults() != 1) {
        return failure();
    }
    return makeReachableRelation(
        {op,
         {operandPort(1, contiguous), operandPort(2, contiguous), operandPort(3, contiguous),
          resultPort(0, contiguous)},
         true},
        supports);
}

RelationResult VMILayoutRelations::buildStrideStore()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() < 1 || op->getNumResults() != 0) {
        return failure();
    }
    // The mask is the last operand: value, destination, offset, block_stride,
    // mask.  The relation has to constrain the mask, not the block stride, so
    // the index is taken from this dialect's operand list (an out-of-range port
    // makes the solver reject the only relation the op has).
    unsigned maskIndex = op->getNumOperands() - 1;
    if (!isa<VMIMaskType>(op->getOperand(maskIndex).getType())) {
        return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, contiguous), operandPort(maskIndex, contiguous)}, false}, supports);
}

RelationResult VMILayoutRelations::buildScatter()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() != 4 || op->getNumResults() != 0) {
        return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, contiguous), operandPort(2, contiguous), operandPort(3, contiguous)}, false}, supports);
}

RelationResult VMILayoutRelations::buildCompressStore()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() != 4 || op->getNumResults() != 0) {
        return failure();
    }
    return makeReachableRelation({op, {operandPort(0, contiguous), operandPort(3, contiguous)}, false}, supports);
}

RelationResult VMILayoutRelations::buildExpandLoad()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() != 4 || op->getNumResults() != 1) {
        return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(2, contiguous), operandPort(3, contiguous), resultPort(0, contiguous)}, true}, supports);
}

RelationResult VMILayoutRelations::buildMaskedLoad(VMIMaskedLoadOp load)
{
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    auto maskType = dyn_cast<VMIMaskType>(load.getMask().getType());
    auto passthruType = dyn_cast<VMIVRegType>(load.getPassthru().getType());
    if (!resultType || !maskType || !passthruType) {
        return failure();
    }
    SmallVector<VMIMaskedLoadLayoutFact, mlir::pto::kValue4> facts;
    auto addFact = [&](VMILayoutAttr result, VMILayoutAttr mask, VMILayoutAttr passthru) {
        if (!isLegalLayoutForType(resultType, result) || !isLegalLayoutForType(passthruType, passthru)) {
            return;
        }
        VMIVRegType r = VMIVRegType::get(
            resultType.getContext(), resultType.getElementCount(), resultType.getElementType(), result);
        VMIVRegType p = VMIVRegType::get(
            passthruType.getContext(), passthruType.getElementCount(), passthruType.getElementType(), passthru);
        VMIMaskType m =
            VMIMaskType::get(maskType.getContext(), maskType.getElementCount(), maskType.getGranularity(), mask);
        if (succeeded(supports.getMaskedLoadLayoutFact(r, m, p))) {
            facts.push_back({result, mask, passthru});
        }
    };
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    VMILayoutAttr maskLayout = maskType.getLayoutAttr();
    VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
    if (resultLayout && maskLayout && passthruLayout) {
        addFact(resultLayout, maskLayout, passthruLayout);
    }
    if (facts.empty()) {
        VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
        addFact(
            resultLayout ? resultLayout : contiguous, maskLayout ? maskLayout : contiguous,
            passthruLayout ? passthruLayout : contiguous);
    }
    for (const auto& fact : facts) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{
                op,
                {operandPort(2, fact.maskLayout), operandPort(3, fact.passthruLayout),
                 resultPort(0, fact.resultLayout)},
                true},
            supports);
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(std::move(relations));
}

RelationResult VMILayoutRelations::buildGroupSlotLoad(VMIGroupSlotLoadOp load)
{
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
        return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidates;
    if (VMILayoutAttr layout = resultType.getLayoutAttr()) {
        candidates.push_back(layout);
    }
    int64_t groups = load.getNumGroupsAttr().getInt();
    for (int64_t slots : {int64_t(1), int64_t(8), groups}) {
        VMILayoutAttr candidate = VMILayoutAttr::getGroupSlots(op->getContext(), groups, slots);
        if (!llvm::is_contained(candidates, candidate)) {
            candidates.push_back(candidate);
        }
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
        if (layout && !llvm::is_contained(candidates, layout)) {
            candidates.push_back(layout);
        }
    }
    for (VMILayoutAttr layout : candidates) {
        auto typed = assignCandidateType(resultType, layout);
        if (failed(typed)) {
            continue;
        }
        auto fact =
            supports.getGroupSlotLoadLayoutFact(*typed, load.getSourceGroupStride(), load.getNumGroupsAttr().getInt());
        if (succeeded(fact)) {
            // slots=8 reads a whole group per block load, while slots=1 needs one
            // scalar broadcast load per slot.  Both are legal for a unit stride,
            // so prefer the fewer-part form and keep slots=1 available for the
            // strides that slots=8 rejects.
            appendReachableUniqueRelation(
                relations, VMILayoutOpRelation{op, {resultPort(0, layout)}, true, 0, layout.getSlots() == 1 ? 1U : 0U},
                supports);
        }
    }
    return relations;
}

RelationResult VMILayoutRelations::buildDeinterleaveLoad(VMIDeinterleaveLoadOp load)
{
    auto lowType = dyn_cast<VMIVRegType>(load.getLow().getType());
    auto highType = dyn_cast<VMIVRegType>(load.getHigh().getType());
    if (!lowType || !highType) {
        return failure();
    }
    VMILayoutAttr lowLayout = lowType.getLayoutAttr();
    VMILayoutAttr highLayout = highType.getLayoutAttr();
    if (lowLayout && highLayout) {
        auto typedLow =
            VMIVRegType::get(lowType.getContext(), lowType.getElementCount(), lowType.getElementType(), lowLayout);
        auto typedHigh =
            VMIVRegType::get(highType.getContext(), highType.getElementCount(), highType.getElementType(), highLayout);
        auto fact = supports.getDeinterleaveLoadLayoutFactForLayouts(typedLow, typedHigh);
        if (succeeded(fact)) {
            return makeReachableRelation(
                {op, {resultPort(0, fact->lowLayout), resultPort(1, fact->highLayout)}, true}, supports);
        }
        return failure();
    }
    if (highLayout) {
        auto facts =
            supports.getDeinterleaveLoadLayoutFactsForLayout(lowType, VMIDeinterleaveLoadLayoutPort::High, highLayout);
        if (succeeded(facts)) {
            for (const auto& fact : *facts) {
                if (!fact.highLayout || fact.highLayout != highLayout) {
                    continue;
                }
                appendReachableUniqueRelation(
                    relations,
                    VMILayoutOpRelation{op, {resultPort(0, fact.lowLayout), resultPort(1, fact.highLayout)}, true},
                    supports);
            }
        }
        if (!relations.empty()) {
            return relations;
        }
        return failure();
    }
    auto preferred = supports.getPreferredDeinterleaveLoadLayoutFact(lowType);
    if (failed(preferred)) {
        return failure();
    }
    return makeReachableRelation(
        {op, {resultPort(0, preferred->lowLayout), resultPort(1, preferred->highLayout)}, true}, supports);
}

RelationResult VMILayoutRelations::buildGroupLoad(VMIGroupLoadOp load)
{
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
        return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidates;
    if (VMILayoutAttr explicitLayout = resultType.getLayoutAttr()) {
        candidates.push_back(explicitLayout);
    }
    for (VMILayoutAttr candidate : polymorphicLayouts) {
        if (candidate && !llvm::is_contained(candidates, candidate)) {
            candidates.push_back(candidate);
        }
    }
    if (candidates.empty()) {
        auto preferred = supports.getGroupLoadLayoutFact(load);
        if (succeeded(preferred)) {
            candidates.push_back(preferred->resultLayout);
        }
    }
    for (VMILayoutAttr layout : candidates) {
        auto typed = assignCandidateType(resultType, layout);
        if (failed(typed)) {
            continue;
        }
        auto fact = supports.getGroupLoadLayoutFact(*typed, load.getRowStride(), load.getNumGroupsAttr().getInt());
        if (succeeded(fact)) {
            appendReachableUniqueRelation(relations, VMILayoutOpRelation{op, {resultPort(0, layout)}, true}, supports);
        }
    }
    return relations;
}

RelationResult VMILayoutRelations::buildGroupStore(VMIGroupStoreOp store)
{
    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    if (!valueType) {
        return failure();
    }
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> candidateLayouts;
    if (VMILayoutAttr explicitLayout = valueType.getLayoutAttr()) {
        candidateLayouts.push_back(explicitLayout);
    }
    for (VMILayoutAttr layout : polymorphicLayouts) {
        if (layout && !llvm::is_contained(candidateLayouts, layout)) {
            candidateLayouts.push_back(layout);
        }
    }
    if (candidateLayouts.empty()) {
        auto preferred = supports.getPreferredGroupStoreLayoutFact(store, valueType);
        if (succeeded(preferred)) {
            candidateLayouts.push_back(preferred->valueLayout);
        }
    }
    if (candidateLayouts.empty()) {
        candidateLayouts.push_back(VMILayoutAttr::getContiguous(op->getContext()));
    }
    auto appendStoreFactsFor = [&](VMILayoutAttr layout) {
        auto facts = supports.getGroupStoreLayoutFactsForLayout(store, valueType, layout);
        if (failed(facts)) {
            return;
        }
        for (const VMIGroupStoreLayoutFact& fact : *facts) {
            appendReachableUniqueRelation(
                relations,
                VMILayoutOpRelation{
                    op, {operandPort(store.getValueMutable().getOperandNumber(), fact.valueLayout)}, false},
                supports);
        }
    };
    for (VMILayoutAttr layout : candidateLayouts) {
        appendStoreFactsFor(layout);
    }
    if (relations.empty()) {
        // The pool holds the layouts the rest of the component can produce, and
        // for a small group count the only row left may be the preferred
        // group-slots one, which no other operation introduces (for example
        // group_store of a 2x16 value with num_groups = 2 has exactly the
        // preferred num_groups = 2, slots = 8 row).  A legal row must still be
        // offered, so ask for the preferred row instead of reporting that this
        // operation has no relation at all.
        auto preferred = supports.getPreferredGroupStoreLayoutFact(store, valueType);
        if (succeeded(preferred) && preferred->valueLayout &&
            !llvm::is_contained(candidateLayouts, preferred->valueLayout)) {
            appendStoreFactsFor(preferred->valueLayout);
        }
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(std::move(relations));
}

RelationResult VMILayoutRelations::buildInterleaveStore(VMIInterleaveStoreOp store)
{
    auto lowType = dyn_cast<VMIVRegType>(store.getLow().getType());
    auto highType = dyn_cast<VMIVRegType>(store.getHigh().getType());
    if (!lowType || !highType) {
        return failure();
    }
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
    VMIVRegType queryLow =
        lowType.getLayoutAttr() ?
            lowType :
            VMIVRegType::get(lowType.getContext(), lowType.getElementCount(), lowType.getElementType(), contiguous);
    VMIVRegType queryHigh =
        highType.getLayoutAttr() ?
            highType :
            VMIVRegType::get(highType.getContext(), highType.getElementCount(), highType.getElementType(), contiguous);
    auto support = VMILayoutSupport().getInterleaveStoreSupport(queryLow, queryHigh);
    if (failed(support)) {
        return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, support->lowLayout), operandPort(1, support->highLayout)}, false}, supports);
}

RelationResult VMILayoutRelations::buildStore(VMIStoreOp store)
{
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    FailureOr<SmallVector<VMIStoreLayoutFact, mlir::pto::kValue4>> facts = supports.getStoreLayoutFacts(valueType);
    if (failed(facts)) {
        return failure();
    }
    auto preferred = supports.getPreferredStoreLayoutFact(valueType);
    unsigned valueIndex = store.getValueMutable().getOperandNumber();
    for (const VMIStoreLayoutFact& fact : *facts) {
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{
                op,
                {operandPort(valueIndex, fact.valueLayout)},
                false,
                0,

                succeeded(preferred) && fact.valueLayout != preferred->valueLayout ? 1U : 0U},
            supports);
    }
    if (relations.empty()) {
        return failure();
    }
    return relations;
}

RelationResult VMILayoutRelations::buildMaskedStore(VMIMaskedStoreOp store)
{
    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    auto maskType = dyn_cast<VMIMaskType>(store.getMask().getType());
    if (!valueType || !maskType) {
        return failure();
    }
    SmallVector<VMILayoutAttr, kValue4> seeds;
    rememberLayout(maskType.getLayoutAttr(), seeds);
    seeds.append(polymorphicLayouts.begin(), polymorphicLayouts.end());
    auto candidates = getDenseCandidateLayouts(valueType.getLayoutAttr(), seeds, op->getContext());
    for (VMILayoutAttr layout : candidates) {
        if (!isLegalLayoutForType(valueType, layout)) {
            continue;
        }
        auto concreteValue =
            VMIVRegType::get(valueType.getContext(), valueType.getElementCount(), valueType.getElementType(), layout);
        auto concreteMask =
            VMIMaskType::get(maskType.getContext(), maskType.getElementCount(), maskType.getGranularity(), layout);
        auto fact = supports.getMaskedStoreLayoutFact(concreteValue, concreteMask);
        if (succeeded(fact)) {
            appendReachableUniqueRelation(
                relations,
                VMILayoutOpRelation{op, {operandPort(0, fact->valueLayout), operandPort(3, fact->maskLayout)}, false},
                supports);
        }
    }
    return relations;
}

RelationResult VMILayoutRelations::buildGroupBroadcastLoad(VMIGroupBroadcastLoadOp load)
{
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    FailureOr<SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4>> facts =
        supports.getGroupBroadcastLoadLayoutFacts(load);
    if (failed(facts)) {
        return failure();
    }
    VMILayoutAttr explicitLayout = getVMIExplicitLayout(load.getResult().getType());
    FailureOr<VMIGroupBroadcastLoadDirectFact> directFact = supports.getGroupBroadcastLoadDirectFact(load);
    VMILayoutAttr unalignedFallback = succeeded(directFact) ?
                                          getUnalignedE2BFallbackLayout(load, resultType, *directFact, supports) :
                                          VMILayoutAttr();
    for (const VMIGroupBroadcastLoadLayoutFact& fact : *facts) {
        if (explicitLayout && explicitLayout != fact.resultLayout) {
            continue;
        }
        if (load.getNumGroupsAttr().getInt() > 8 && !fact.resultLayout.isContiguous()) {
            continue;
        }
        // An aligned E2B lowering and the contiguous BRC lowering are both
        // legal physical producers.  Keep both relations in the solver domain;
        // the physical cost model must decide whether the packet load plus a
        // layout materialization is preferable to per-group BRC and its merge
        // tree.  Only an unaligned E2B address is restricted to the BRC
        // fallback, because selecting E2B in that case would make lowering
        // reject an otherwise valid layout plan.
        if (unalignedFallback && fact.resultLayout != unalignedFallback) {
            continue;
        }
        if (!isLegalLayoutForType(resultType, fact.resultLayout)) {
            continue;
        }
        auto candidateType = VMIVRegType::get(
            resultType.getContext(), resultType.getElementCount(), resultType.getElementType(), fact.resultLayout);
        FailureOr<VMIGroupBroadcastLoadDirectFact> candidateDirectFact = supports.getGroupBroadcastLoadDirectFact(
            candidateType, load.getSource().getType(), load.getSourceGroupStride(), load.getNumGroupsAttr().getInt());
        bool directProducer =
            succeeded(candidateDirectFact) && isGroupBroadcastLoadDirectAddressLegal(load, *candidateDirectFact);
        appendReachableUniqueRelation(
            relations, VMILayoutOpRelation{op, {resultPort(0, fact.resultLayout)}, directProducer}, supports);
    }
    return relations;
}

} // namespace mlir::pto::layout_relations
