// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Physical cost operations: preserves the action graph selected by the planner.
#include "VMILayoutCostGraph.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_cost {

FailureOr<int64_t> PlanGraphBuilder::getCompactReductionComputeCount(
    const VMIGroupReduceLayoutFact& fact, VMIVRegType source, VMIVRegType result, int64_t numGroups)
{
    // Singleton integer groups lower to a neutral vector and a select.
    if (fact.groupSize == 1 && isa<IntegerType>(source.getElementType())) {
        return 2;
    }
    // Each carrier/group intersection executes a row reduction, and each
    // extra partial in a group executes one combining operation.
    int64_t count = 0;
    int64_t begin = 0;
    for (int64_t group = 0; group < numGroups; ++group) {
        int64_t end = begin + fact.groupSize;
        int64_t partials = (end - 1) / fact.lanesPerPart - begin / fact.lanesPerPart + 1;
        if (partials > (std::numeric_limits<int64_t>::max() - count) / 2) {
            return failure();
        }
        count += partials * 2 - 1;
        begin = end;
    }
    VMILayoutAttr resultLayout = result.getLayoutAttr();
    bool packetResult = resultLayout.getSlots() != 1;
    if (packetResult) {
        int64_t extraGroups = numGroups - 1;
        if (extraGroups > (std::numeric_limits<int64_t>::max() - count) / 2) {
            return failure();
        }
        count += extraGroups * 2; // One Vdup and Vsel per additional slot.
    }
    return count;
}

FailureOr<int64_t> PlanGraphBuilder::getGroupReductionComputeCount(Operation* op, const VMILayoutOpRelation& relation)
{
    auto source = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto mask = dyn_cast<VMIMaskType>(op->getOperand(1).getType());
    auto result = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    auto groups = op->getAttrOfType<IntegerAttr>("num_groups");
    if (!groups) {
        groups = op->getAttrOfType<IntegerAttr>("group");
    }
    if (!source || !mask || !result || !groups || groups.getInt() <= 0) {
        return failure();
    }
    source = VMIVRegType::get(
        source.getContext(), source.getElementCount(), source.getElementType(), relation.ports[0].layout);
    mask = VMIMaskType::get(mask.getContext(), mask.getElementCount(), mask.getGranularity(), relation.ports[1].layout);
    result = VMIVRegType::get(
        result.getContext(), result.getElementCount(), result.getElementType(), relation.ports[2].layout);
    auto fact = VMILayoutSupport().getGroupReduceLayoutFactForLayouts(
        getVMIGroupReduceKind(op), source, mask, result, groups.getInt());
    auto sourceArity = getVMIPhysicalArity(source);
    auto resultArity = getVMIPhysicalArity(result);
    bool validFacts = succeeded(fact) && succeeded(sourceArity) && succeeded(resultArity);
    if (!validFacts) {
        return failure();
    }
    if (fact->blockClass == VMIGroupBlockClass::Compact) {
        return getCompactReductionComputeCount(*fact, source, result, groups.getInt());
    }
    // Native VCG plans reduce every source carrier, then combine the partials
    // in each result packet. Full-part row plans similarly join each row.
    int64_t outputs = fact->blockClass == VMIGroupBlockClass::FullPartMultiple ? groups.getInt() : *resultArity;
    if (*sourceArity < outputs || *sourceArity > std::numeric_limits<int64_t>::max() / 2) {
        return failure();
    }
    return *sourceArity * 2 - outputs;
}

LogicalResult PlanGraphBuilder::buildRelation(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<VMICreateGroupMaskOp>([&](auto) { return buildGeneratedMask(relation); })
        .Case<VMICreateMaskOp, VMIConstantMaskOp, VMIPsetOp, VMIPgeOp, VMIPltOp>(
            [&](auto) { return buildStaticMask(relation); })
        .Case<VMIEnsureLayoutOp, VMIEnsureMaskLayoutOp>([&](auto) { return buildEnsure(relation); })
        .Case<VMIEnsureMaskGranularityOp>([&](auto) { return buildMaskGranularity(relation); })
        .Case<VMIBitcastOp, VMIVinterpretCastOp>([&](auto) { return buildBitcast(relation); })
        .Case<VMIConstantOp, VMIBroadcastOp, VMIIotaOp, VMIGroupIotaOp, VMIVciOp, VMIVbrcOp>(
            [&](auto) { return buildProducer(relation); })
        .Case<VMIGroupBroadcastLoadOp>(
            [&](auto) { return buildGroupBroadcastLoad(cast<VMIGroupBroadcastLoadOp>(op), relation); })
        .Case<VMIShuffleOp>([&](auto) { return buildShuffle(relation); })
        .Case<VMIVdhistOp, VMIVchistOp, VMIVselrOp>([&](auto) { return buildHistogram(relation); })
        .Case<VMIActivePrefixIndexOp>([&](auto) { return buildActivePrefixIndex(relation); })
        .Case<VMICompressOp>([&](auto) { return buildCompress(relation); })
        .Case<VMIExpandLoadOp>([&](auto) { return buildExpandLoad(relation); })
        .Case<VMIGroupBroadcastOp>([&](auto) { return buildGroupBroadcast(relation); })
        .Case<
            VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp, VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
            VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp, VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>(
            [&](auto) { return buildGroupReduce(relation); })
        .Case<VMILoadOp>([&](auto) { return buildLoad(cast<VMILoadOp>(op), relation); })
        .Case<VMIStrideLoadOp, VMIGatherOp>([&](auto) { return buildIndexedLoad(relation); })
        .Case<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp, VMIReduceMinFOp, VMIReduceMaxIOp, VMIReduceMinIOp>(
            [&](auto) { return buildReduce(relation); })
        .Case<VMIVintlvOp, VMIVdintlvOp>([&](auto) { return buildInterleave(relation); })
        .Case<VMIChannelSplitOp>([&](auto) { return buildChannelSplit(relation); })
        .Case<VMIChannelMergeOp>([&](auto) { return buildChannelMerge(relation); })
        .Case<VMIMaskedLoadOp>([&](auto) { return buildMaskedLoad(relation); })
        .Case<VMIGroupLoadOp, VMIGroupSlotLoadOp>([&](auto) { return buildGroupLoad(relation); })
        .Case<VMIDeinterleaveLoadOp>([&](auto) { return buildDeinterleaveLoad(relation); })
        .Case<VMIGroupStoreOp>([&](auto) { return buildGroupStore(relation); })
        .Case<VMIInterleaveStoreOp>([&](auto) { return buildInterleaveStore(relation); })
        .Case<VMIStoreOp>([&](auto) { return buildStore(cast<VMIStoreOp>(op), relation); })
        .Case<VMIStrideStoreOp, VMIScatterOp, VMICompressStoreOp>([&](auto) { return buildIndexedStore(relation); })
        .Case<VMIMaskedStoreOp>([&](auto) { return buildMaskedStore(relation); })
        .Case<VMIVUnzipOp, VMIVZipOp>([&](auto) { return buildPairedShuffle(op, relation); })
        .Case<VMIExtFOp, VMIExtSIOp, VMIExtUIOp, VMITruncFOp, VMITruncIOp, VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(
            [&](auto) { return buildCast(op, relation); })
        .Default([&](Operation*) {
            return isVMISameLayoutOp(op) || isa<VMIVexpdifOp>(op) ? buildSameLayoutOp(op, relation) : failure();
        });
}

LogicalResult PlanGraphBuilder::buildGeneratedMask(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto generated = cast<VMICreateGroupMaskOp>(op);
    if (relation.ports.size() != 1 || relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
    }
    VMILayoutAttr resultLayout = relation.ports.front().layout;
    VMILayoutSupport support;
    auto fact = support.getGeneratedMaskLayoutFact(op, resultLayout);
    auto type = cast<VMIMaskType>(generated.getResult().getType());
    if (failed(fact)) {
        return failure();
    }
    if (fact->generationLayout == fact->resultLayout) {
        values[generated.getResult()] = graph.createInputs(1);
        layouts[generated.getResult()] = resultLayout;
        return success();
    }
    auto arity = getMaskArity(type, fact->generationLayout);
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    auto source = graph.createInputs(1);
    auto converted = materializeMaskLayout(source, type, fact->generationLayout, fact->resultLayout, op);
    if (failed(converted)) {
        return failure();
    }
    values[generated.getResult()] = std::move(*converted);
    layouts[generated.getResult()] = resultLayout;
    return success();
}

LogicalResult PlanGraphBuilder::buildStaticMask(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 1 || relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
    }
    values[op->getResult(relation.ports.front().index)] = graph.createInputs(1);
    layouts[op->getResult(relation.ports.front().index)] = relation.ports.front().layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildEnsure(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
    if (failed(source)) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    if (auto mask = dyn_cast<VMIMaskType>(op->getOperand(0).getType())) {
        auto converted = materializeMaskLayout(*source, mask, relation.ports[0].layout, relation.ports[1].layout, op);
        if (failed(converted)) {
            return failure();
        }
        result = std::move(*converted);
    } else {
        auto converted =
            materialize(op->getOperand(0), *source, relation.ports[0].layout, relation.ports[1].layout, op);
        if (failed(converted)) {
            return failure();
        }
        result = std::move(*converted);
    }
    values[op->getResult(0)] = std::move(result);
    layouts[op->getResult(0)] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildMaskGranularity(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto ensure = cast<VMIEnsureMaskGranularityOp>(op);
    if (relation.ports.size() != 2) {
        return failure();
    }
    auto source = buildOperand(ensure.getSourceMutable(), relation.ports[0].layout);
    if (failed(source) || source->empty()) {
        return failure();
    }
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    VMILayoutAttr sourceCarrier = getMaskCarrierLayout(resultType, relation.ports[0].layout);
    VMILayoutAttr resultCarrier = getMaskCarrierLayout(resultType, relation.ports[1].layout);
    auto converted = materializeMaskLayout(*source, resultType, sourceCarrier, resultCarrier, op);
    if (failed(converted)) {
        return failure();
    }
    values[ensure.getResult()] = std::move(*converted);
    layouts[ensure.getResult()] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildBitcast(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    auto arity = resultType ? getArity(resultType, relation.ports[1].layout) : FailureOr<int64_t>(failure());
    if (failed(source) || failed(arity) || source->empty() || *arity <= 0) {
        return failure();
    }
    if (!sourceType || !resultType) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (int64_t index = 0; index < *arity; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op), {(*source)[static_cast<size_t>(index) % source->size()]},
            "bitcast")[0]);
    }
    values[op->getResult(0)] = std::move(results);
    layouts[op->getResult(0)] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildProducer(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    // pto.vmi.broadcast reads one scalar or one 1-lane vector, and only the
    // second form carries a layout: -vmi-lower-unified-to-legacy produces it
    // for every vbrc without a group attribute, and the planner states the
    // source use layout next to the result layout.  Its lowering
    // (vdup -LOWEST) reads the source part and duplicates it, so the result
    // parts derive from that source part and the relation pays whatever
    // materializing the source in its use layout costs.  Treating this
    // relation as unscoreable would drop a legal row and empty the solver's
    // frontier.
    if (isa<VMIBroadcastOp>(op) && relation.ports.size() == 2 && relation.ports[0].kind == VMILayoutPortKind::Operand &&
        relation.ports[1].kind == VMILayoutPortKind::Result) {
        auto source = buildOperand(op->getOpOperand(relation.ports[0].index), relation.ports[0].layout);
        Value result = op->getResult(relation.ports[1].index);
        auto type = dyn_cast<VMIVRegType>(result.getType());
        if (failed(source) || source->empty() || !type) {
            return failure();
        }
        FailureOr<int64_t> arity = getArity(type, relation.ports[1].layout);
        if (failed(arity) || *arity <= 0) {
            return failure();
        }
        SmallVector<PhysicalValue, mlir::pto::kValue4> results;
        for (int64_t index = 0; index < *arity; ++index) {
            results.push_back(
                graph.addAction(PhysicalActionKind::Semantic, getDynamicScope(op), {(*source)[0]}, "broadcast")[0]);
        }
        values[result] = std::move(results);
        layouts[result] = relation.ports[1].layout;
        return success();
    }
    if (relation.ports.size() != 1 || relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
    }
    Value result = op->getResult(relation.ports.front().index);
    auto type = dyn_cast<VMIVRegType>(result.getType());
    auto arity = type ? getArity(type, relation.ports.front().layout) : FailureOr<int64_t>(failure());
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    values[result] = graph.createInputs(static_cast<unsigned>(*arity));
    layouts[result] = relation.ports.front().layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildShuffle(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto shuffle = cast<VMIShuffleOp>(op);
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto source = buildOperand(shuffle.getSourceMutable(), relation.ports[0].layout);
    auto resultType = dyn_cast<VMIVRegType>(shuffle.getResult().getType());
    if (failed(source) || source->empty() || !resultType) {
        return failure();
    }
    FailureOr<int64_t> resultArity = getArity(resultType, relation.ports[1].layout);
    if (failed(resultArity) || *resultArity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (int64_t index = 0; index < *resultArity; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op), {(*source)[static_cast<size_t>(index) % source->size()]},
            "shuffle")[0]);
    }
    values[shuffle.getResult()] = std::move(results);
    layouts[shuffle.getResult()] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildHistogram(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.empty()) {
        return failure();
    }
    SmallVector<SmallVector<PhysicalValue, mlir::pto::kValue4>, mlir::pto::kValue4> inputs;
    OpResult result;
    VMILayoutAttr resultLayout;
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind == VMILayoutPortKind::Operand) {
            auto input = buildOperand(op->getOpOperand(port.index), port.layout);
            if (failed(input) || input->empty()) {
                return failure();
            }
            inputs.push_back(std::move(*input));
        } else {
            if (result || port.index >= op->getNumResults()) {
                return failure();
            }
            result = op->getResult(port.index);
            resultLayout = port.layout;
        }
    }
    if (!result || !resultLayout || inputs.empty()) {
        return failure();
    }
    auto resultType = dyn_cast<VMIVRegType>(result.getType());
    auto arity = resultType ? getArity(resultType, resultLayout) : FailureOr<int64_t>(failure());
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (int64_t index = 0; index < *arity; ++index) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> args;
        for (const auto& input : inputs) {
            args.push_back(input.size() == 1 ? input.front() : input[static_cast<size_t>(index) % input.size()]);
        }
        results.push_back(
            graph.addAction(PhysicalActionKind::Semantic, getDynamicScope(op), args, op->getName().getStringRef())[0]);
    }
    values[result] = std::move(results);
    layouts[result] = resultLayout;
    return success();
}

LogicalResult PlanGraphBuilder::buildActivePrefixIndex(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto active = cast<VMIActivePrefixIndexOp>(op);
    if (relation.ports.size() != 2) {
        return failure();
    }
    auto mask = buildOperand(active.getMaskMutable(), relation.ports[0].layout);
    auto type = dyn_cast<VMIVRegType>(active.getResult().getType());
    FailureOr<int64_t> arity = failure();
    if (type) {
        arity = getArity(type, relation.ports[1].layout);
    }
    if (failed(mask) || failed(arity) || mask->empty() || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    for (int64_t index = 0; index < *arity; ++index) {
        result.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op), {mask->front()}, "active_prefix_index")[0]);
    }
    values[active.getResult()] = std::move(result);
    layouts[active.getResult()] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildCompress(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto compress = cast<VMICompressOp>(op);
    if (relation.ports.size() != 3) {
        return failure();
    }
    auto source = buildOperand(compress.getSourceMutable(), relation.ports[0].layout);
    auto mask = buildOperand(compress.getMaskMutable(), relation.ports[1].layout);
    auto type = dyn_cast<VMIVRegType>(compress.getResult().getType());
    if (failed(source) || failed(mask) || !type || source->empty() || mask->empty()) {
        return failure();
    }
    auto arity = getArity(type, relation.ports[2].layout);
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    for (int64_t index = 0; index < *arity; ++index) {
        result.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()], mask->front()}, "compress")[0]);
    }
    values[compress.getResult()] = std::move(result);
    layouts[compress.getResult()] = relation.ports[2].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildExpandLoad(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto expand = cast<VMIExpandLoadOp>(op);
    if (relation.ports.size() != 3) {
        return failure();
    }
    auto mask = buildOperand(expand.getMaskMutable(), relation.ports[0].layout);
    auto passthru = buildOperand(expand.getPassthruMutable(), relation.ports[1].layout);
    auto type = dyn_cast<VMIVRegType>(expand.getResult().getType());
    if (failed(mask) || failed(passthru) || !type || mask->empty() || passthru->empty()) {
        return failure();
    }
    auto arity = getArity(type, relation.ports[2].layout);
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    for (int64_t index = 0; index < *arity; ++index) {
        result.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op), {mask->front(), passthru->front()}, "expand_load",
            1)[0]);
    }
    values[expand.getResult()] = std::move(result);
    layouts[expand.getResult()] = relation.ports[2].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildGroupBroadcast(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto broadcast = cast<VMIGroupBroadcastOp>(op);
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto source = buildOperand(broadcast.getSourceMutable(), relation.ports[0].layout);
    auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
    FailureOr<int64_t> resultArity = failure();
    if (resultType) {
        resultArity = getArity(resultType, relation.ports[1].layout);
    }
    if (failed(source) || source->empty() || failed(resultArity)) {
        return failure();
    }
    int64_t resultArityValue = *resultArity;
    if (resultArityValue <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (int64_t index = 0; index < resultArityValue; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op), {(*source)[static_cast<size_t>(index) % source->size()]},
            "group_broadcast")[0]);
    }
    values[broadcast.getResult()] = std::move(results);
    layouts[broadcast.getResult()] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildGroupReduce(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 3 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Operand || relation.ports[2].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
    auto mask = buildOperand(op->getOpOperand(1), relation.ports[1].layout);
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    FailureOr<int64_t> resultArity = failure();
    if (resultType) {
        resultArity = getArity(resultType, relation.ports[2].layout);
    }
    if (failed(source) || failed(mask) || failed(resultArity) || source->empty() || mask->empty()) {
        return failure();
    }
    int64_t resultArityValue = *resultArity;
    if (resultArityValue <= 0) {
        return failure();
    }
    auto computeCount = getGroupReductionComputeCount(op, relation);
    if (failed(computeCount)) {
        return failure();
    }
    for (int64_t index = 0; index < *computeCount; ++index) {
        graph.addAction(PhysicalActionKind::Compute, getDynamicScope(op), *source, "group_reduce_compute", 0);
    }
    // Independent result carriers are not a packing tree. Arithmetic is
    // charged above; actual layout conversions and stores at their uses.
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (int64_t index = 0; index < resultArityValue; ++index) {
        results.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*source)[static_cast<size_t>(index) % source->size()], mask->front()}, "group_reduce")[0]);
    }
    values[op->getResult(0)] = std::move(results);
    layouts[op->getResult(0)] = relation.ports[2].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildIndexedLoad(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.empty()) {
        return failure();
    }
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind != VMILayoutPortKind::Operand) {
            continue;
        }
        if (failed(buildOperand(op->getOpOperand(port.index), port.layout))) {
            return failure();
        }
    }
    auto result = op->getResult(0);
    auto type = dyn_cast<VMIVRegType>(result.getType());
    auto arity = type ? getArity(type, relation.ports.back().layout) : FailureOr<int64_t>(failure());
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> out;
    for (int64_t i = 0; i < *arity; ++i) {
        out.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op), {}, isa<VMIGatherOp>(op) ? "gather" : "stride_load",
            1)[0]);
    }
    values[result] = std::move(out);
    layouts[result] = relation.ports.back().layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildReduce(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 3) {
        return failure();
    }
    auto source = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
    auto mask = buildOperand(op->getOpOperand(1), relation.ports[1].layout);
    auto type = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    FailureOr<int64_t> arity = failure();
    if (type) {
        arity = getArity(type, relation.ports[2].layout);
    }
    if (failed(source) || failed(mask) || failed(arity) || source->empty() || mask->empty()) {
        return failure();
    }
    int64_t arityValue = *arity;
    if (arityValue <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    for (int64_t i = 0; i < arityValue; ++i) {
        result.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op), {(*source)[i % source->size()], mask->front()},
            op->getName().getStringRef())[0]);
    }
    values[op->getResult(0)] = std::move(result);
    layouts[op->getResult(0)] = relation.ports[2].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildInterleave(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 5) {
        return failure();
    }
    auto lhs = buildOperand(op->getOpOperand(0), relation.ports[0].layout);
    auto rhs = buildOperand(op->getOpOperand(1), relation.ports[1].layout);
    auto mask = buildOperand(op->getOpOperand(2), relation.ports[2].layout);
    auto lowType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    auto highType = dyn_cast<VMIVRegType>(op->getResult(1).getType());
    FailureOr<int64_t> lowArity = failure();
    FailureOr<int64_t> highArity = failure();
    if (lowType) {
        lowArity = getArity(lowType, relation.ports[3].layout);
    }
    if (highType) {
        highArity = getArity(highType, relation.ports[4].layout);
    }
    if (failed(lhs) || failed(rhs) || failed(mask) || failed(lowArity) || failed(highArity) || lhs->empty() ||
        rhs->empty() || mask->empty()) {
        return failure();
    }
    int64_t lowArityValue = *lowArity;
    int64_t highArityValue = *highArity;
    if (lowArityValue <= 0 || highArityValue <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> low;
    SmallVector<PhysicalValue, mlir::pto::kValue4> high;
    for (int64_t i = 0; i < lowArityValue; ++i) {
        low.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*lhs)[i % lhs->size()], (*rhs)[i % rhs->size()], mask->front()}, op->getName().getStringRef())[0]);
    }
    for (int64_t i = 0; i < highArityValue; ++i) {
        high.push_back(graph.addAction(
            PhysicalActionKind::Semantic, getDynamicScope(op),
            {(*lhs)[i % lhs->size()], (*rhs)[i % rhs->size()], mask->front()}, op->getName().getStringRef())[0]);
    }
    values[op->getResult(0)] = std::move(low);
    values[op->getResult(1)] = std::move(high);
    layouts[op->getResult(0)] = relation.ports[3].layout;
    layouts[op->getResult(1)] = relation.ports[4].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildChannelSplit(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto split = cast<VMIChannelSplitOp>(op);
    if (relation.ports.size() != split.getNumResults() + 1) {
        return failure();
    }
    auto source = buildOperand(split.getSourceMutable(), relation.ports[0].layout);
    if (failed(source) || source->empty()) {
        return failure();
    }
    int64_t sourceOffset = 0;
    for (unsigned i = 0; i < split.getNumResults(); ++i) {
        auto type = dyn_cast<VMIVRegType>(split.getResult(i).getType());
        auto arity = type ? getArity(type, relation.ports[i + 1].layout) : FailureOr<int64_t>(failure());
        if (failed(arity) || *arity <= 0 || sourceOffset > static_cast<int64_t>(source->size()) - *arity) {
            return failure();
        }
        SmallVector<PhysicalValue, mlir::pto::kValue4> result;
        for (int64_t p = 0; p < *arity; ++p) {
            result.push_back((*source)[sourceOffset + p]);
        }
        sourceOffset += *arity;
        values[split.getResult(i)] = std::move(result);
        layouts[split.getResult(i)] = relation.ports[i + 1].layout;
    }
    if (sourceOffset != static_cast<int64_t>(source->size())) {
        return failure();
    }
    return success();
}

LogicalResult PlanGraphBuilder::buildChannelMerge(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto merge = cast<VMIChannelMergeOp>(op);
    if (relation.ports.size() != merge.getInputs().size() + 1) {
        return failure();
    }
    SmallVector<SmallVector<PhysicalValue, mlir::pto::kValue4>, mlir::pto::kValue4> inputs;
    for (unsigned i = 0; i < merge.getInputs().size(); ++i) {
        auto input = buildOperand(merge->getOpOperand(i), relation.ports[i].layout);
        if (failed(input) || input->empty()) {
            return failure();
        }
        inputs.push_back(std::move(*input));
    }
    auto type = dyn_cast<VMIVRegType>(merge.getResult().getType());
    auto arity = type ? getArity(type, relation.ports.back().layout) : FailureOr<int64_t>(failure());
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    for (const auto& input : inputs) {
        result.append(input.begin(), input.end());
    }
    if (static_cast<int64_t>(result.size()) != *arity) {
        return failure();
    }
    values[merge.getResult()] = std::move(result);
    layouts[merge.getResult()] = relation.ports.back().layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildMaskedLoad(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto load = cast<VMIMaskedLoadOp>(op);
    if (relation.ports.size() != 3 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Operand || relation.ports[2].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto mask = buildOperand(load.getMaskMutable(), relation.ports[0].layout);
    auto passthru = buildOperand(load.getPassthruMutable(), relation.ports[1].layout);
    auto type = dyn_cast<VMIVRegType>(load.getResult().getType());
    FailureOr<int64_t> arity = failure();
    if (type) {
        arity = getArity(type, relation.ports[2].layout);
    }
    if (failed(mask) || failed(passthru) || failed(arity) || mask->empty() || passthru->empty()) {
        return failure();
    }
    int64_t arityValue = *arity;
    if (arityValue <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> result;
    for (int64_t i = 0; i < arityValue; ++i) {
        result.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op), {mask->front(), passthru->front()}, "masked_load",
            1)[0]);
    }
    values[load.getResult()] = std::move(result);
    layouts[load.getResult()] = relation.ports[2].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildGroupLoad(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.size() != 1 || relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto result = op->getResult(0);
    auto type = dyn_cast<VMIVRegType>(result.getType());
    auto arity = type ? getArity(type, relation.ports.front().layout) : FailureOr<int64_t>(failure());
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> valuesForResult;
    for (int64_t index = 0; index < *arity; ++index) {
        valuesForResult.push_back(graph.addAction(
            PhysicalActionKind::MemoryRead, getDynamicScope(op), {},
            isa<VMIGroupSlotLoadOp>(op) ? "group_slot_load" : "group_load", 1)[0]);
    }
    VMILayoutAttr layout = relation.ports.front().layout;
    bool stridedPacketLoad = isa<VMIGroupSlotLoadOp>(op) && layout.getSlots() == mlir::pto::kValue8 &&
                             layout.getNumGroups() > 1 && layout.getLaneStride() > 1;
    if (stridedPacketLoad) {
        auto packedLayout = VMILayoutAttr::getGroupSlots(op->getContext(), layout.getNumGroups(), layout.getSlots());
        auto strided = groupSlotsToGroupSlots(valuesForResult, packedLayout, layout, getDynamicScope(op));
        if (failed(strided)) {
            return failure();
        }
        valuesForResult = std::move(*strided);
    }
    values[result] = std::move(valuesForResult);
    layouts[result] = relation.ports.front().layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildDeinterleaveLoad(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto load = cast<VMIDeinterleaveLoadOp>(op);
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Result ||
        relation.ports[1].kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto lowType = dyn_cast<VMIVRegType>(load.getLow().getType());
    auto highType = dyn_cast<VMIVRegType>(load.getHigh().getType());
    if (!lowType || !highType) {
        return failure();
    }
    auto lowArity = getArity(lowType, relation.ports[0].layout);
    auto highArity = getArity(highType, relation.ports[1].layout);
    if (failed(lowArity) || failed(highArity) || *lowArity <= 0 || *highArity <= 0 || *lowArity != *highArity) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> low;
    SmallVector<PhysicalValue, mlir::pto::kValue4> high;
    for (int64_t index = 0; index < *lowArity; ++index) {
        auto loaded = graph.addAction(PhysicalActionKind::MemoryRead, getDynamicScope(op), {}, "vlds", 2);
        low.push_back(loaded[0]);
        high.push_back(loaded[1]);
    }
    values[load.getLow()] = std::move(low);
    values[load.getHigh()] = std::move(high);
    layouts[load.getLow()] = relation.ports[0].layout;
    layouts[load.getHigh()] = relation.ports[1].layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildGroupStore(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto store = cast<VMIGroupStoreOp>(op);
    if (relation.ports.size() != 1 || relation.ports.front().kind != VMILayoutPortKind::Operand) {
        return failure();
    }
    auto input = buildOperand(store.getValueMutable(), relation.ports.front().layout);
    if (failed(input) || input->empty()) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> storeValues = *input;
    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    if (!valueType) {
        return failure();
    }
    auto concreteType = VMIVRegType::get(
        valueType.getContext(), valueType.getElementCount(), valueType.getElementType(), relation.ports.front().layout);
    auto fact = VMILayoutSupport().getGroupStoreLayoutFact(store, concreteType);
    if (succeeded(fact) && fact->stagingLayout) {
        auto staged = materialize(store.getValue(), *input, relation.ports.front().layout, fact->stagingLayout, op);
        if (failed(staged)) {
            return failure();
        }
        storeValues = std::move(*staged);
    }
    for (PhysicalValue value : storeValues) {
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op), {value}, "group_store", 0);
    }
    return success();
}

LogicalResult PlanGraphBuilder::buildInterleaveStore(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto store = cast<VMIInterleaveStoreOp>(op);
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Operand) {
        return failure();
    }
    auto lowType = cast<VMIVRegType>(store.getLow().getType());
    auto highType = cast<VMIVRegType>(store.getHigh().getType());
    auto support = VMILayoutSupport().getInterleaveStoreSupport(
        VMIVRegType::get(
            lowType.getContext(), lowType.getElementCount(), lowType.getElementType(), relation.ports[0].layout),
        VMIVRegType::get(
            highType.getContext(), highType.getElementCount(), highType.getElementType(), relation.ports[1].layout));
    if (failed(support)) {
        return failure();
    }
    auto low = buildOperand(store.getLowMutable(), relation.ports[0].layout);
    auto high = buildOperand(store.getHighMutable(), relation.ports[1].layout);
    if (failed(low) || failed(high) || low->empty() || high->empty() || low->size() != high->size()) {
        return failure();
    }
    for (size_t index = 0; index < low->size(); ++index) {
        graph.addAction(
            PhysicalActionKind::MemoryWrite, getDynamicScope(op), {(*low)[index], (*high)[index]}, "interleave_store",
            0);
    }
    return success();
}

LogicalResult PlanGraphBuilder::buildIndexedStore(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    if (relation.ports.empty()) {
        return failure();
    }
    auto value = buildOperand(op->getOpOperand(0), relation.ports.front().layout);
    if (failed(value) || value->empty()) {
        return failure();
    }
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind != VMILayoutPortKind::Operand || port.index == 0) {
            continue;
        }
        if (failed(buildOperand(op->getOpOperand(port.index), port.layout))) {
            return failure();
        }
    }
    StringRef tag = isa<VMIScatterOp>(op) ? "scatter" : isa<VMICompressStoreOp>(op) ? "compress_store" : "stride_store";
    for (PhysicalValue v : *value) {
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op), {v}, tag, 0);
    }
    return success();
}

LogicalResult PlanGraphBuilder::buildMaskedStore(const VMILayoutOpRelation& relation)
{
    Operation* op = relation.op;
    auto store = cast<VMIMaskedStoreOp>(op);
    if (relation.ports.size() != 2 || relation.ports[0].kind != VMILayoutPortKind::Operand ||
        relation.ports[1].kind != VMILayoutPortKind::Operand) {
        return failure();
    }
    auto value = buildOperand(store.getValueMutable(), relation.ports[0].layout);
    auto mask = buildOperand(store.getMaskMutable(), relation.ports[1].layout);
    if (failed(value) || failed(mask) || value->empty() || mask->empty()) {
        return failure();
    }
    auto valueType = dyn_cast<VMIVRegType>(store.getValue().getType());
    auto maskType = dyn_cast<VMIMaskType>(store.getMask().getType());
    if (!valueType || !maskType) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> storeValues = *value;
    SmallVector<PhysicalValue, mlir::pto::kValue4> storeMasks = *mask;
    if (relation.ports[0].layout.isDeinterleaved()) {
        // Dense deinterleaved masked stores canonicalize both data and mask to
        // contiguous physical chunks inside the op before issuing vsts.
        VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
        auto convertedValue = materialize(store.getValue(), *value, relation.ports[0].layout, contiguous, op);
        auto convertedMask = materializeMaskLayout(*mask, maskType, relation.ports[1].layout, contiguous, op);
        if (failed(convertedValue) || failed(convertedMask)) {
            return failure();
        }
        storeValues = std::move(*convertedValue);
        storeMasks = std::move(*convertedMask);
    }
    if (storeMasks.empty()) {
        return failure();
    }
    for (auto [index, v] : llvm::enumerate(storeValues)) {
        PhysicalValue m = storeMasks[index % storeMasks.size()];
        graph.addAction(PhysicalActionKind::MemoryWrite, getDynamicScope(op), {v, m}, "masked_store", 0);
    }
    return success();
}

static FailureOr<BroadcastLoadState> getBroadcastLoadState(
    VMIGroupBroadcastLoadOp load, const VMILayoutOpRelation& relation)
{
    if (relation.ports.size() != 1 || relation.ports.front().kind != VMILayoutPortKind::Result) {
        return failure();
    }
    auto type = cast<VMIVRegType>(load.getResult().getType());
    VMILayoutAttr layout = relation.ports.front().layout;
    auto arity = getArity(type, layout);
    if (failed(arity) || *arity <= 0) {
        return failure();
    }
    VMILayoutSupport supports;
    auto assignedType = VMIVRegType::get(type.getContext(), type.getElementCount(), type.getElementType(), layout);
    auto direct = supports.getGroupBroadcastLoadDirectFact(
        assignedType, load.getSource().getType(), load.getSourceGroupStride(), load.getNumGroupsAttr().getInt());
    auto fact = supports.getGroupBroadcastLoadLayoutFact(
        assignedType, load.getSourceGroupStride(), load.getNumGroupsAttr().getInt());
    // The relation layout is the layout requested by the component plan.  A
    // direct E2B producer can have a different native carrier layout (d2/d4),
    // however: the packet load is formed in that layout and the consumer sees
    // the assigned result layout only after an explicit materialization.  Keep
    // this edge in the physical graph so the solver prices the same
    // interleave/deinterleave actions that lowering will emit.
    VMILayoutAttr producerLayout = layout;
    if (relation.directProducer && succeeded(direct) && direct->kind == VMIGroupBroadcastLoadDirectKind::E2B &&
        direct->layout.resultLayout) {
        producerLayout = direct->layout.resultLayout;
    }
    auto producerArity = getArity(type, producerLayout);
    if (failed(producerArity) || *producerArity <= 0) {
        return failure();
    }
    return BroadcastLoadState{layout, producerLayout, *arity, *producerArity, direct, fact};
}
static bool usesPerGroupBroadcastLoads(VMIGroupBroadcastLoadOp load, const BroadcastLoadState& state)
{
    return succeeded(state.fact) && state.layout.isContiguous() && state.layout.getLaneStride() == 1 &&
           state.fact->groupSize > 0 && state.fact->groupSize < state.fact->lanesPerPart &&
           state.fact->lanesPerPart % state.fact->groupSize == 0 &&
           load.getNumGroupsAttr().getInt() == state.arity * (state.fact->lanesPerPart / state.fact->groupSize) &&
           (!succeeded(state.direct) || state.direct->kind == VMIGroupBroadcastLoadDirectKind::BRC);
}
PhysicalValues PlanGraphBuilder::buildBroadcastLoadActions(
    VMIGroupBroadcastLoadOp load, const VMILayoutOpRelation& relation, const BroadcastLoadState& state)
{
    Region* scope = getDynamicScope(load);
    bool perGroupBRC = usesPerGroupBroadcastLoads(load, state);
    int64_t reads = 1;
    if (perGroupBRC) {
        reads = load.getNumGroupsAttr().getInt();
    } else if (relation.directProducer && succeeded(state.direct)) {
        reads = state.direct->kind == VMIGroupBroadcastLoadDirectKind::BRC ?
                    state.arity :
                    std::max<int64_t>(
                        1, state.producerArity /
                               (state.producerLayout.isDeinterleaved() ? state.producerLayout.getFactor() : 1));
    }
    for (int64_t index = 1; index < reads; ++index) {
        graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, "brc");
    }
    if (perGroupBRC) {
        for (int64_t index = state.arity; index < reads; ++index) {
            graph.addAction(PhysicalActionKind::Merge, scope, {}, "vsel");
        }
    }
    StringRef tag =
        succeeded(state.direct) && state.direct->kind == VMIGroupBroadcastLoadDirectKind::E2B ? "e2b" : "brc";
    auto loaded = graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, tag, 1);
    // The E2B packet itself is one physical carrier.  The support table may
    // expose its logical result as contiguous, but a multi-carrier contiguous
    // VMI result still needs one packet expansion per output carrier; VPTO
    // lowers that expansion through VINTLV.  Count those actions here so an
    // E2B choice cannot look free merely because its logical layout is already
    // contiguous.  Deinterleaved E2B results are packet aliases by design and
    // must remain free of this charge.
    if (relation.directProducer && succeeded(state.direct) &&
        state.direct->kind == VMIGroupBroadcastLoadDirectKind::E2B && state.producerLayout == state.layout &&
        state.layout.isContiguous() && state.arity > 1) {
        for (int64_t index = 0; index < state.arity; ++index) {
            graph.addAction(PhysicalActionKind::Interleave, scope, {loaded[0]}, "e2b-contiguous", 2);
        }
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> producerValues(state.producerArity, loaded[0]);
    return producerValues;
}
LogicalResult PlanGraphBuilder::buildGroupBroadcastLoad(
    VMIGroupBroadcastLoadOp load, const VMILayoutOpRelation& relation)
{
    auto state = getBroadcastLoadState(load, relation);
    if (failed(state)) {
        return failure();
    }
    auto producerValues = buildBroadcastLoadActions(load, relation, *state);
    if (state->producerLayout != state->layout) {
        auto converted = materialize(load.getResult(), producerValues, state->producerLayout, state->layout, load);
        if (failed(converted)) {
            return failure();
        }
        values[load.getResult()] = std::move(*converted);
    } else {
        values[load.getResult()] = std::move(producerValues);
    }
    layouts[load.getResult()] = state->layout;
    return success();
}

PhysicalValues PlanGraphBuilder::buildDenseDeinterleavedLoad(int64_t arity, int64_t factor, Region* scope)
{
    PhysicalValues results;
    if (factor == 2) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> lows;
        SmallVector<PhysicalValue, mlir::pto::kValue4> highs;
        for (int64_t group = 0; group < arity / 2; ++group) {
            auto loaded = graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, "vldsx2", 2);
            lows.push_back(loaded[0]);
            highs.push_back(loaded[1]);
        }
        results.append(lows);
        results.append(highs);
    } else {
        for (int64_t group = 0; group < arity / 4; ++group) {
            auto first = graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, "vldsx2", 2);
            auto second = graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, "vldsx2", 2);
            auto even = graph.addAction(PhysicalActionKind::Deinterleave, scope, {first[0], second[0]}, "d2", 2);
            auto odd = graph.addAction(PhysicalActionKind::Deinterleave, scope, {first[1], second[1]}, "d2", 2);
            results.append({even[0], odd[0], even[1], odd[1]});
        }
    }
    return results;
}
LogicalResult PlanGraphBuilder::buildLoad(VMILoadOp load, const VMILayoutOpRelation& relation)
{
    if (relation.ports.size() != 1) {
        return failure();
    }
    auto type = cast<VMIVRegType>(load.getResult().getType());
    VMILayoutAttr layout = relation.ports.front().layout;
    auto arity = getArity(type, layout);
    auto contiguousArity = getArity(type, VMILayoutAttr::getContiguous(type.getContext()));
    if (failed(arity) || failed(contiguousArity) || *arity <= 0) {
        return failure();
    }
    Region* scope = getDynamicScope(load);
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    bool directDenseSplit = isFullPhysicalShape(type) && *arity <= *contiguousArity;
    if (layout.isDeinterleaved() && layout.getFactor() == 2 && directDenseSplit && *arity % 2 == 0) {
        results = buildDenseDeinterleavedLoad(*arity, 2, scope);
    } else if (layout.isDeinterleaved() && layout.getFactor() == 4 && directDenseSplit && *arity % 4 == 0) {
        results = buildDenseDeinterleavedLoad(*arity, 4, scope);
    } else if (layout.isDeinterleaved()) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> contiguous;
        for (int64_t part = 0; part < *contiguousArity; ++part) {
            contiguous.push_back(graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, "vlds", 1)[0]);
        }
        auto converted =
            materialize(load.getResult(), contiguous, VMILayoutAttr::getContiguous(type.getContext()), layout, load);
        if (failed(converted)) {
            return failure();
        }
        results = std::move(*converted);
    } else {
        for (int64_t part = 0; part < *arity; ++part) {
            results.push_back(graph.addAction(PhysicalActionKind::MemoryRead, scope, {}, "vlds", 1)[0]);
        }
    }
    values[load.getResult()] = std::move(results);
    layouts[load.getResult()] = layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildStore(VMIStoreOp store, const VMILayoutOpRelation& relation)
{
    if (relation.ports.size() != 1) {
        return failure();
    }
    unsigned operandIndex = relation.ports.front().index;
    VMILayoutAttr layout = relation.ports.front().layout;
    auto input = buildOperand(store->getOpOperand(operandIndex), layout);
    auto type = cast<VMIVRegType>(store.getValue().getType());
    auto contiguousArity = getArity(type, VMILayoutAttr::getContiguous(type.getContext()));
    if (failed(input) || failed(contiguousArity)) {
        return failure();
    }
    Region* scope = getDynamicScope(store);
    if (layout.isDeinterleaved() && layout.getFactor() == 2 && isFullPhysicalShape(type) && input->size() % 2 == 0 &&
        input->size() <= static_cast<size_t>(*contiguousArity)) {
        for (size_t part = 0; part < input->size() / 2; ++part) {
            graph.addAction(
                PhysicalActionKind::MemoryWrite, scope, {(*input)[part], (*input)[input->size() / 2 + part]}, "vstsx2",
                0);
        }
        return success();
    }
    if (layout.isContiguous() && layout.getLaneStride() > 1) {
        for (PhysicalValue value : *input) {
            graph.addAction(PhysicalActionKind::MemoryWrite, scope, {value}, "vsts", 0);
        }
        return success();
    }
    auto contiguous =
        materialize(store.getValue(), *input, layout, VMILayoutAttr::getContiguous(type.getContext()), store);
    if (failed(contiguous)) {
        return failure();
    }
    for (PhysicalValue value : *contiguous) {
        graph.addAction(PhysicalActionKind::MemoryWrite, scope, {value}, "vsts", 0);
    }
    return success();
}

FailureOr<size_t> PlanGraphBuilder::getPairedShuffleRatio(size_t wide, size_t half)
{
    if (half == 0 || wide < half || wide % half != 0) {
        return failure();
    }
    size_t ratio = wide / half;
    if (ratio != 1 && ratio != 2) {
        return failure();
    }
    return ratio;
}

LogicalResult PlanGraphBuilder::buildPairedShuffle(Operation* op, const VMILayoutOpRelation& relation)
{
    if (relation.ports.size() != 3) {
        return failure();
    }
    VMILayoutAttr layout = relation.ports.front().layout;
    if (!layout || llvm::any_of(relation.ports, [&](const auto& port) { return port.layout != layout; }) ||
        failed(VMILayoutSupport().getSameLayoutRelationSupport(op, layout))) {
        return failure();
    }
    if (isa<VMIVUnzipOp>(op)) {
        return buildUnzip(op, layout);
    }
    return buildZip(op, layout);
}

LogicalResult PlanGraphBuilder::buildUnzip(Operation* op, VMILayoutAttr layout)
{
    auto source = buildOperand(op->getOpOperand(0), layout);
    auto lowArity = getArity(cast<VMIVRegType>(op->getResult(0).getType()), layout);
    auto highArity = getArity(cast<VMIVRegType>(op->getResult(1).getType()), layout);
    if (failed(source) || failed(lowArity) || failed(highArity) || *lowArity <= 0 || *lowArity != *highArity) {
        return failure();
    }
    size_t halfCount = static_cast<size_t>(*lowArity);
    auto ratio = getPairedShuffleRatio(source->size(), halfCount);
    if (failed(ratio)) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> lows;
    SmallVector<PhysicalValue, mlir::pto::kValue4> highs;
    for (size_t part = 0; part < halfCount; ++part) {
        size_t first = part * *ratio;
        size_t second = first + *ratio - 1;
        auto pair = graph.addAction(
            PhysicalActionKind::Deinterleave, getDynamicScope(op), {(*source)[first], (*source)[second]}, "vunzip", 2);
        lows.push_back(pair[0]);
        highs.push_back(pair[1]);
    }
    values[op->getResult(0)] = std::move(lows);
    values[op->getResult(1)] = std::move(highs);
    layouts[op->getResult(0)] = layout;
    layouts[op->getResult(1)] = layout;
    return success();
}

LogicalResult PlanGraphBuilder::buildZip(Operation* op, VMILayoutAttr layout)
{
    auto low = buildOperand(op->getOpOperand(0), layout);
    auto high = buildOperand(op->getOpOperand(1), layout);
    auto arity = getArity(cast<VMIVRegType>(op->getResult(0).getType()), layout);
    if (failed(low) || failed(high) || failed(arity) || *arity <= 0 || low->size() != high->size()) {
        return failure();
    }
    auto ratio = getPairedShuffleRatio(static_cast<size_t>(*arity), low->size());
    if (failed(ratio)) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (size_t part = 0; part < low->size(); ++part) {
        auto pair = graph.addAction(
            PhysicalActionKind::Interleave, getDynamicScope(op), {(*low)[part], (*high)[part]}, "vzip", 2);
        results.append(pair.begin(), pair.begin() + *ratio);
    }
    values[op->getResult(0)] = std::move(results);
    layouts[op->getResult(0)] = layout;
    return success();
}

static FailureOr<std::pair<int64_t, int64_t>> getSameLayoutArities(
    Type resultType, VMILayoutAttr resultLayout, const PhysicalInputs& inputs)
{
    int64_t arity = 0;
    int64_t inputArity = 0;
    if (isa<VMIMaskType>(resultType)) {
        arity = 1;
        for (const auto& input : inputs) {
            inputArity = std::max<int64_t>(inputArity, input.size());
        }
    } else if (auto type = dyn_cast<VMIVRegType>(resultType)) {
        auto physicalArity = getArity(type, resultLayout);
        if (failed(physicalArity)) {
            return failure();
        }
        arity = *physicalArity;
        inputArity = arity;
    } else {
        return failure();
    }
    if (arity <= 0 || llvm::any_of(inputs, [&](ArrayRef<PhysicalValue> input) {
            return input.size() != 1 && input.size() != static_cast<size_t>(inputArity);
        })) {
        return failure();
    }
    return std::make_pair(arity, inputArity);
}
static FailureOr<int64_t> getDenseComputeArity(Operation* op, VMIVRegType type, int64_t resultArity)
{
    int64_t denseComputeArity = resultArity;
    if (isCarrierWiseArithmetic(op) && type) {
        auto denseArity = getArity(type, VMILayoutAttr::getContiguous(op->getContext()));
        if (failed(denseArity) || *denseArity <= 0) {
            return failure();
        }
        denseComputeArity = *denseArity;
    }
    return denseComputeArity;
}
LogicalResult PlanGraphBuilder::buildSameLayoutResult(
    Operation* op, OpResult result, VMILayoutAttr layout, int64_t arity, const PhysicalInputs& inputs)
{
    if (!layout) {
        return failure();
    }
    auto type = dyn_cast<VMIVRegType>(result.getType());
    int64_t resultArity = arity;
    if (isa<VMIMaskType>(result.getType())) {
        resultArity = 1;
    } else if (type) {
        auto physicalArity = getArity(type, layout);
        if (failed(physicalArity) || *physicalArity <= 0) {
            return failure();
        }
        resultArity = *physicalArity;
    }
    if (resultArity != 1 && resultArity != arity) {
        return failure();
    }
    // Normalize away the same dense baseline for every layout of this op.
    // Its lowering emits one arithmetic instruction per result carrier, so
    // score differences must equal differences in replicated instructions.
    auto denseComputeArity = getDenseComputeArity(op, type, resultArity);
    if (failed(denseComputeArity)) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> resultValues;
    for (int64_t part = 0; part < resultArity; ++part) {
        PhysicalActionKind actionKind =
            part >= *denseComputeArity ? PhysicalActionKind::Compute : PhysicalActionKind::Semantic;
        SmallVector<PhysicalValue, mlir::pto::kValue4> partInputs;
        for (const auto& input : inputs) {
            partInputs.push_back(input.size() == 1 ? input.front() : input[static_cast<size_t>(part) % input.size()]);
        }
        resultValues.push_back(
            graph.addAction(actionKind, getDynamicScope(op), partInputs, op->getName().getStringRef(), 1)[0]);
    }
    values[result] = std::move(resultValues);
    layouts[result] = layout;
    return success();
}
LogicalResult PlanGraphBuilder::buildSameLayoutOp(Operation* op, const VMILayoutOpRelation& relation)
{
    VMILayoutAttr candidateLayout;
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind == VMILayoutPortKind::Result) {
            candidateLayout = port.layout;
            break;
        }
    }
    if (failed(VMILayoutSupport().getSameLayoutRelationSupport(op, candidateLayout))) {
        return failure();
    }
    SmallVector<SmallVector<PhysicalValue, mlir::pto::kValue4>, mlir::pto::kValue4> inputs;
    SmallVector<std::pair<OpResult, VMILayoutAttr>, mlir::pto::kValue4> resultsToBuild;
    for (const VMILayoutPortAssignment& port : relation.ports) {
        if (port.kind == VMILayoutPortKind::Operand) {
            auto input = buildOperand(op->getOpOperand(port.index), port.layout);
            if (failed(input)) {
                return failure();
            }
            inputs.push_back(std::move(*input));
        } else {
            resultsToBuild.emplace_back(cast<OpResult>(op->getResult(port.index)), port.layout);
        }
    }
    if (resultsToBuild.empty()) {
        return failure();
    }
    auto resultType = resultsToBuild.front().first.getType();
    VMILayoutAttr resultLayout = resultsToBuild.front().second;
    if (!resultLayout) {
        return failure();
    }
    auto arities = getSameLayoutArities(resultType, resultLayout, inputs);
    if (failed(arities)) {
        return failure();
    }
    for (const auto& [result, layout] : resultsToBuild) {
        if (failed(buildSameLayoutResult(op, result, layout, arities->first, inputs))) {
            return failure();
        }
    }
    return success();
}

FailureOr<PhysicalValues> PlanGraphBuilder::buildWideningCast(
    ArrayRef<PhysicalValue> source, int64_t resultArity, Region* scope, PhysicalActionKind conversionKind,
    ArrayRef<PhysicalValue> predicate)
{
    PhysicalValues results;

    if (source.empty() || resultArity <= 0 || resultArity % source.size() != 0) {
        return failure();
    }
    size_t outputsPerSource = resultArity / source.size();
    for (size_t output = 0; output < outputsPerSource; ++output) {
        for (PhysicalValue input : source) {
            std::string tag = (Twine("ext:") + Twine(output)).str();
            SmallVector<PhysicalValue, mlir::pto::kValue4> inputs{input};
            inputs.append(predicate.begin(), predicate.end());
            auto converted = graph.addAction(conversionKind, scope, inputs, tag, 1);
            results.push_back(converted[0]);
        }
    }
    return results;
}
FailureOr<PhysicalValues> PlanGraphBuilder::buildNarrowingCast(
    ArrayRef<PhysicalValue> source, int64_t resultArity, Region* scope, ArrayRef<PhysicalValue> predicate)
{
    PhysicalValues results;

    if (resultArity == 0) {
        return failure();
    }
    if (resultArity < 0 || source.size() < static_cast<size_t>(resultArity)) {
        return failure();
    }
    for (int64_t result = 0; result < resultArity; ++result) {
        size_t begin = result * source.size() / resultArity;
        size_t end = (result + 1) * source.size() / resultArity;
        if (begin == end) {
            return failure();
        }
        SmallVector<PhysicalValue, mlir::pto::kValue4> partials;
        for (size_t sourceIndex = begin; sourceIndex < end; ++sourceIndex) {
            std::string tag = (Twine("trunc:") + Twine(sourceIndex - begin)).str();
            SmallVector<PhysicalValue, mlir::pto::kValue4> inputs{(source)[sourceIndex]};
            inputs.append(predicate.begin(), predicate.end());
            partials.push_back(graph.addAction(PhysicalActionKind::Semantic, scope, inputs, tag)[0]);
        }
        if (partials.empty()) {
            return failure();
        }
        // The begin == end guard above already keeps this range non-empty, but
        // state it explicitly: SmallVector inline storage is unconstructed, so
        // an unguarded front() leaves GCC -O3 unable to prove the read is live.
        if (partials.empty()) {
            return failure();
        }
        PhysicalValue merged = partials.front();
        for (PhysicalValue partial : llvm::drop_begin(partials)) {
            merged = graph.addAction(PhysicalActionKind::Merge, scope, {merged, partial}, "vor")[0];
        }
        results.push_back(merged);
    }
    return results;
}
std::optional<PlanGraphBuilder::CastPortLayouts> PlanGraphBuilder::resolveCastPortLayouts(
    Operation* op, const VMILayoutOpRelation& relation)
{
    // An unmasked cast relates source and result only.  A masked cast adds the
    // conversion predicate between them, so its relation has three ports and the
    // middle one is the predicate.
    bool masked = op->getNumOperands() == 2 && isa<VMIMaskType>(op->getOperand(1).getType());
    size_t expectedPorts = masked ? 3 : 2;
    bool portShape = relation.ports.size() == expectedPorts && relation.ports[0].kind == VMILayoutPortKind::Operand &&
                     relation.ports[expectedPorts - 1].kind == VMILayoutPortKind::Result;
    if (masked) {
        portShape = portShape && relation.ports[1].kind == VMILayoutPortKind::Operand;
    }
    if (!portShape) {
        return std::nullopt;
    }
    CastPortLayouts layouts{relation.ports[0].layout, std::nullopt, relation.ports[expectedPorts - 1].layout};
    if (masked) {
        layouts.predicate = relation.ports[1].layout;
    }
    return layouts;
}

LogicalResult PlanGraphBuilder::buildCast(Operation* op, const VMILayoutOpRelation& relation)
{
    std::optional<CastPortLayouts> ports = resolveCastPortLayouts(op, relation);
    if (!ports) {
        return failure();
    }
    auto source = buildOperand(op->getOpOperand(0), ports->source);
    auto sourceType = cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = cast<VMIVRegType>(op->getResult(0).getType());
    auto resultArity = getArity(resultType, ports->result);
    if (failed(source) || failed(resultArity) || source->empty() || *resultArity <= 0) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> predicate;
    if (ports->predicate) {
        auto mask = buildOperand(op->getOpOperand(1), *ports->predicate);
        bool maskUsable = succeeded(mask) && !mask->empty();
        if (!maskUsable) {
            return failure();
        }
        predicate.push_back(mask->front());
    }
    FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> results =
        buildCastConversion(op, *source, *resultArity, sourceType, resultType, relation, predicate);
    if (failed(results)) {
        return failure();
    }
    values[op->getResult(0)] = std::move(*results);
    return success();
}

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::buildCastConversion(
    Operation* op, ArrayRef<PhysicalValue> source, int64_t resultArity, VMIVRegType sourceType, VMIVRegType resultType,
    const VMILayoutOpRelation& relation, ArrayRef<PhysicalValue> predicate)
{
    unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
    Region* scope = getDynamicScope(op);
    // Native widening emits one VCVT per result carrier, for both integer and
    // floating types. A replacement recipe (integer unpack or BF16 interleave)
    // is charged intrinsically instead; its result views emit no extra VCVTs.
    bool nativeExtension = isa<VMIExtFOp, VMIExtUIOp, VMIExtSIOp>(op) && relation.intrinsicRearrangementCost == 0;
    PhysicalActionKind conversionKind = nativeExtension ? PhysicalActionKind::Compute : PhysicalActionKind::Semantic;
    if (sourceBits != resultBits) {
        return sourceBits < resultBits ? buildWideningCast(source, resultArity, scope, conversionKind, predicate) :
                                         buildNarrowingCast(source, resultArity, scope, predicate);
    }
    bool arityMismatch = source.size() != static_cast<size_t>(resultArity);
    if (arityMismatch) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (PhysicalValue input : source) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> inputs{input};
        inputs.append(predicate.begin(), predicate.end());
        results.push_back(graph.addAction(PhysicalActionKind::Semantic, scope, inputs, "cast")[0]);
    }
    return results;
}

} // namespace mlir::pto::layout_cost
