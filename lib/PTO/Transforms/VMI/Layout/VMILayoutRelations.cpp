// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Layout relation enumeration: relations.
#include "VMILayoutRelations.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_relations {

std::optional<RelationResult> VMILayoutRelations::buildMemoryRelation()
{
    return llvm::TypeSwitch<Operation*, std::optional<RelationResult>>(op)
        .Case<VMILoadOp>([&](VMILoadOp load) { return buildLoad(load); })
        .Case<VMIStrideLoadOp>([&](VMIStrideLoadOp) { return buildStrideLoad(); })
        .Case<VMIGatherOp>([&](VMIGatherOp) { return buildGather(); })
        .Case<VMIStrideStoreOp>([&](VMIStrideStoreOp) { return buildStrideStore(); })
        .Case<VMIScatterOp>([&](VMIScatterOp) { return buildScatter(); })
        .Case<VMICompressStoreOp>([&](VMICompressStoreOp) { return buildCompressStore(); })

        .Case<VMIExpandLoadOp>([&](VMIExpandLoadOp) { return buildExpandLoad(); })

        .Case<VMIMaskedLoadOp>([&](VMIMaskedLoadOp load) { return buildMaskedLoad(load); })
        .Case<VMIGroupSlotLoadOp>([&](VMIGroupSlotLoadOp load) { return buildGroupSlotLoad(load); })
        .Case<VMIDeinterleaveLoadOp>([&](VMIDeinterleaveLoadOp load) { return buildDeinterleaveLoad(load); })
        .Case<VMIGroupLoadOp>([&](VMIGroupLoadOp load) { return buildGroupLoad(load); })
        .Case<VMIGroupStoreOp>([&](VMIGroupStoreOp store) { return buildGroupStore(store); })
        .Case<VMIInterleaveStoreOp>([&](VMIInterleaveStoreOp store) { return buildInterleaveStore(store); })
        .Case<VMIStoreOp>([&](VMIStoreOp store) { return buildStore(store); })
        .Case<VMIMaskedStoreOp>([&](VMIMaskedStoreOp store) { return buildMaskedStore(store); })
        .Case<VMIGroupBroadcastLoadOp>([&](VMIGroupBroadcastLoadOp load) { return buildGroupBroadcastLoad(load); })
        .Default([](Operation*) { return std::nullopt; });
}
RelationResult VMILayoutRelations::build()
{
    if (!op) {
        return failure();
    }
    if (isVMILayoutCastOp(op)) {
        return buildCast();
    }
    if (auto memory = buildMemoryRelation()) {
        return *memory;
    }
    return llvm::TypeSwitch<Operation*, RelationResult>(op)
        .Case<func::ReturnOp>([&](func::ReturnOp returnOp) { return buildReturn(returnOp); })
        .Case<func::CallOp>([&](func::CallOp call) { return buildCall(call); })
        .Case<VMIEnsureLayoutOp, VMIEnsureMaskLayoutOp>([&](auto) { return buildEnsure(); })
        .Case<VMIEnsureMaskGranularityOp>(
            [&](VMIEnsureMaskGranularityOp ensure) { return buildMaskGranularity(ensure); })
        .Case<VMIBitcastOp, VMIVinterpretCastOp>([&](auto) { return buildBitcast(); })
        .Case<VMIActivePrefixIndexOp>([&](VMIActivePrefixIndexOp) { return buildActivePrefixIndex(); })
        .Case<VMICompressOp>([&](VMICompressOp) { return buildCompress(); })
        .Case<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp, VMIReduceMinFOp, VMIReduceMaxIOp, VMIReduceMinIOp>(
            [&](auto) { return buildReduce(); })
        .Case<VMIVintlvOp>([&](VMIVintlvOp interleave) { return buildInterleave(interleave); })
        .Case<VMIVdintlvOp>([&](VMIVdintlvOp interleave) { return buildDeinterleave(interleave); })
        .Case<VMIChannelSplitOp>([&](VMIChannelSplitOp split) { return buildChannelSplit(split); })
        .Case<VMIChannelMergeOp>([&](VMIChannelMergeOp merge) { return buildChannelMerge(merge); })
        .Case<VMIShuffleOp>([&](VMIShuffleOp shuffle) { return buildShuffle(shuffle); })
        .Case<VMIVdhistOp, VMIVchistOp>([&](auto) { return buildHistogram(); })
        .Case<VMIVselrOp>([&](VMIVselrOp vselr) { return buildSelect(vselr); })
        .Case<VMICreateMaskOp, VMICreateGroupMaskOp, VMIConstantMaskOp, VMIPsetOp, VMIPgeOp, VMIPltOp>(
            [&](auto) { return buildMaskProducer(); })
        .Case<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMIGroupIotaOp, VMIVciOp, VMIVbrcOp>(
            [&](auto) { return buildProducer(); })
        .Case<VMIGroupBroadcastOp>([&](VMIGroupBroadcastOp broadcast) { return buildGroupBroadcast(broadcast); })
        .Case<
            VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
            VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>([&](auto) { return buildGroupReduce(); })
        .Case<VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>([&](auto) { return buildLegacyReduce(); })
        .Case<VMIVexpdifOp>([&](VMIVexpdifOp vexpdif) { return buildVexpdif(vexpdif); })
        .Default([&](Operation*) { return isVMISameLayoutOp(op) ? buildSameLayout() : RelationResult(failure()); });
}

RelationResult VMILayoutRelations::buildReturn(func::ReturnOp returnOp)
{
    auto function = returnOp->getParentOfType<func::FuncOp>();
    if (!function || returnOp.getNumOperands() != function.getNumResults()) {
        return failure();
    }
    VMILayoutOpRelation relation;
    relation.op = op;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
        // Only a result whose type spells out a layout is an ABI contract the
        // plan has to materialize.  An unannotated result takes the layout of the
        // value it returns: rewriteFunctionType adopts that operand type, and a
        // call site pins the result through getConsistentCallResultTypes.  Pinning
        // it to the contiguous boundary here instead demands a conversion the
        // layout tables do not offer - a compact group result has no ensure_layout
        // row from its group-slots layout to the dense form - which turns a
        // satisfiable component into "no complete legal VMI layout plan".
        VMILayoutAttr layout = getVMIExplicitLayout(function.getResultTypes()[index]);
        if (layout) {
            relation.endpoints.push_back({operand, &returnOp->getOpOperand(index), layout});
        }
    }
    if (relation.endpoints.empty()) {
        // Nothing is pinned, so the returned value keeps whatever layout its
        // producer selected.
        return SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>{};
    }
    relations.push_back(std::move(relation));
    return relations;
}

RelationResult VMILayoutRelations::buildCall(func::CallOp call)
{
    VMILayoutOpRelation relation;
    relation.op = op;
    for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
        if (VMILayoutAttr layout = getABIBoundaryLayout(operand.getType())) {
            relation.endpoints.push_back({operand, &call->getOpOperand(index), layout});
        }
    }
    for (Value result : call.getResults()) {
        if (VMILayoutAttr layout = getABIBoundaryLayout(result.getType())) {
            relation.endpoints.push_back({result, nullptr, layout});
        }
    }
    if (relation.endpoints.empty()) {
        return failure();
    }
    relations.push_back(std::move(relation));
    return relations;
}

RelationResult VMILayoutRelations::buildActivePrefixIndex()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return failure();
    }
    return makeReachableRelation({op, {operandPort(0, contiguous), resultPort(0, contiguous)}, false}, supports);
}

RelationResult VMILayoutRelations::buildCompress()
{
    auto contiguous = VMILayoutAttr::getContiguous(op->getContext());
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
        return failure();
    }
    return makeReachableRelation(
        {op, {operandPort(0, contiguous), operandPort(1, contiguous), resultPort(0, contiguous)}, false}, supports);
}

RelationResult VMILayoutRelations::buildMaskProducer()
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> maskLayouts;
    auto maskType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
    VMILayoutAttr explicitLayout = maskType ? maskType.getLayoutAttr() : VMILayoutAttr{};
    if (explicitLayout) {
        maskLayouts.push_back(explicitLayout);
    } else {
        maskLayouts.push_back(VMILayoutAttr::getContiguous(op->getContext()));
        for (VMILayoutAttr layout : polymorphicLayouts) {
            if (!llvm::is_contained(maskLayouts, layout)) {
                maskLayouts.push_back(layout);
            }
        }
    }
    for (VMILayoutAttr layout : maskLayouts) {
        if (!layout) {
            continue;
        }
        if (isa<VMICreateGroupMaskOp>(op) && failed(supports.getGeneratedMaskLayoutFact(op, layout))) {
            continue;
        }
        appendReachableUniqueRelation(relations, VMILayoutOpRelation{op, {resultPort(0, layout)}, false}, supports);
    }
    return relations;
}

RelationResult VMILayoutRelations::buildSameLayout()
{
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> sameLayoutCandidates;
    appendSameLayoutCandidates(op, polymorphicLayouts, sameLayoutCandidates);
    appendSameLayoutRelations(op, sameLayoutCandidates, supports, relations, reason);
    if (!relations.empty()) {
        return relations;
    }

    return failure();
}

} // namespace mlir::pto::layout_relations
mlir::pto::layout_relations::RelationResult mlir::pto::VMILayoutRelationProvider::enumerateRelations(
    Operation* op, ArrayRef<VMILayoutAttr> layouts, std::string* reason) const
{
    return layout_relations::VMILayoutRelations(op, layouts, reason).build();
}
namespace mlir::pto::layout_relations {} // namespace mlir::pto::layout_relations
