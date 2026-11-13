// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Physical cost materialization: preserves the action graph selected by the planner.
#include "VMILayoutCostGraph.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_cost {

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::addMaskActions(
    PhysicalValue source, PhysicalActionKind kind, int64_t count, Region* scope, StringRef tag)
{
    if (count < 0) {
        return failure();
    }
    PhysicalValue current = source;
    for (int64_t index = 0; index < count; ++index) {
        current = graph.addAction(kind, scope, {current}, tag, 1)[0];
    }
    return SmallVector<PhysicalValue, mlir::pto::kValue4>{current};
}

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::materializeMaskPackTree(
    PhysicalValue source, int64_t sourceArity, int64_t levels, Region* scope)
{
    if (sourceArity <= 0 || levels <= 0) {
        return failure();
    }
    PhysicalValue current = source;
    int64_t carriers = sourceArity;
    for (int64_t level = 0; level < levels; ++level) {
        auto packed = addMaskActions(current, PhysicalActionKind::Pack, carriers, scope, "mask-pack");
        if (failed(packed)) {
            return failure();
        }
        current = packed->front();
        auto merged = addMaskActions(current, PhysicalActionKind::Merge, carriers / 2, scope, "mask-merge");
        if (failed(merged)) {
            return failure();
        }
        current = merged->front();
        carriers = (carriers + 1) / 2;
    }
    return SmallVector<PhysicalValue, mlir::pto::kValue4>{current};
}

FailureOr<PhysicalValues> PlanGraphBuilder::materializeMaskLaneStrideEndpoint(
    PhysicalValue input, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t sourceArity,
    int64_t resultArity, Region* scope)
{
    bool sourceContiguous = sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1;
    bool resultContiguous = resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
    if (sourceContiguous && resultLayout.isContiguous()) {
        int64_t stride = resultLayout.getLaneStride();
        if (stride != 2 && stride != 4) {
            return failure();
        }
        int64_t levels = stride == 2 ? 1 : 2;
        return addMaskActions(input, PhysicalActionKind::Unpack, resultArity * levels, scope, "mask-unpack");
    }
    // A lane-strided contiguous mask is still a contiguous carrier family;
    // only its physical packing differs.  Use the same pack tree as the
    // dense data materializer when normalizing it to unit stride.
    if (sourceLayout.isContiguous() && resultContiguous) {
        int64_t stride = sourceLayout.getLaneStride();
        if (stride != 2 && stride != 4) {
            return failure();
        }
        return materializeMaskPackTree(input, sourceArity, stride == 2 ? 1 : 2, scope);
    }
    return failure();
}
FailureOr<PhysicalValues> PlanGraphBuilder::materializeMaskDenseEndpoint(
    PhysicalValue input, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t sourceArity,
    int64_t resultArity, Region* scope)
{
    bool sourceContiguous = sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1;
    bool resultContiguous = resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
    if (sourceContiguous && resultLayout.isDeinterleaved()) {
        int64_t factor = resultLayout.getFactor();
        if ((factor != 2 && factor != 4) || resultArity % factor != 0) {
            return failure();
        }
        int64_t actions = factor == 2 ? resultArity / 2 : resultArity;
        return addMaskActions(input, PhysicalActionKind::Deinterleave, actions, scope, "mask-deinterleave");
    }
    if (sourceLayout.isDeinterleaved() && resultContiguous) {
        int64_t factor = sourceLayout.getFactor();
        if ((factor != 2 && factor != 4) || sourceArity % factor != 0) {
            return failure();
        }
        int64_t actions = factor == 2 ? (resultArity + 1) / 2 : 2 * ((resultArity + 3) / 4) + (resultArity + 1) / 2;
        return addMaskActions(input, PhysicalActionKind::Interleave, actions, scope, "mask-interleave");
    }
    return materializeMaskLaneStrideEndpoint(input, sourceLayout, resultLayout, sourceArity, resultArity, scope);
}
static FailureOr<std::pair<int64_t, int64_t>> getMaskMaterializationArities(
    VMIMaskType type, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout)
{
    auto sourceArity = getMaskArity(type, sourceLayout);
    auto resultArity = getMaskArity(type, resultLayout);
    if (failed(sourceArity) || failed(resultArity) || *sourceArity <= 0 || *resultArity <= 0) {
        return failure();
    }
    return std::make_pair(*sourceArity, *resultArity);
}
static FailureOr<int64_t> getBlockMaskPermutationActions(
    bool toBlock, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t sourceArity, int64_t resultArity)
{
    int64_t factor = toBlock ? resultLayout.getFactor() : sourceLayout.getFactor();
    bool invalidBlockShape = (factor != 2 && factor != 4) || sourceArity != factor || resultArity != factor;
    if (invalidBlockShape) {
        return failure();
    }
    // Two splats and one encoding select per input carrier. Each
    // output builds logical/source indices, gathers F carriers,
    // merges F-1 pairs and compares the selected word back to zero.
    int64_t actions = 2 + factor + factor * (10 + 3 * factor) + (toBlock ? factor - 1 : 0);
    return actions;
}
FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::materializeMaskLayout(
    ArrayRef<PhysicalValue> source, VMIMaskType type, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Operation* placement)
{
    if (source.empty() || !type || !sourceLayout || !resultLayout) {
        return failure();
    }
    if (sourceLayout == resultLayout) {
        return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    auto arities = getMaskMaterializationArities(type, sourceLayout, resultLayout);
    if (failed(arities)) {
        return failure();
    }
    auto [sourceArity, resultArity] = *arities;

    Region* scope = getDynamicScope(placement);
    PhysicalValue input = source.front();
    bool sourceContiguous = sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1;
    bool resultContiguous = resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
    bool toBlock = sourceContiguous && resultLayout.isBlockDeinterleaved();
    bool fromBlock = sourceLayout.isBlockDeinterleaved() && resultContiguous;
    if (toBlock || fromBlock) {
        auto actions = getBlockMaskPermutationActions(toBlock, sourceLayout, resultLayout, sourceArity, resultArity);
        if (failed(actions)) {
            return failure();
        }
        return addMaskActions(input, PhysicalActionKind::Compute, *actions, scope, "block-mask-permutation");
    }
    if (sourceLayout.isContiguous() || resultLayout.isContiguous()) {
        return materializeMaskDenseEndpoint(input, sourceLayout, resultLayout, sourceArity, resultArity, scope);
    }
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(type.getContext());
    auto first = materializeMaskLayout(source, type, sourceLayout, contiguous, placement);
    if (failed(first)) {
        return failure();
    }
    return materializeMaskLayout(*first, type, contiguous, resultLayout, placement);
}

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::groupSlotsToGroupSlots(
    ArrayRef<PhysicalValue> source, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, Region* scope)
{
    if (!sourceLayout.isGroupSlots() || !resultLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() || sourceLayout.getSlots() != mlir::pto::kValue8 ||
        resultLayout.getSlots() != mlir::pto::kValue8) {
        return failure();
    }
    int64_t sourceStride = sourceLayout.getLaneStride();
    int64_t resultStride = resultLayout.getLaneStride();
    if ((sourceStride != 1 && sourceStride != 2 && sourceStride != 4) ||
        (resultStride != 1 && resultStride != 2 && resultStride != 4)) {
        return failure();
    }
    PhysicalActionKind kind = sourceStride < resultStride ? PhysicalActionKind::Unpack : PhysicalActionKind::Pack;
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (PhysicalValue input : source) {
        PhysicalValue current = input;
        int64_t currentStride = sourceStride;
        while (currentStride != resultStride) {
            current = graph.addAction(kind, scope, {current}, "group-slot-stride", 1)[0];
            currentStride = sourceStride < resultStride ? currentStride * 2 : currentStride / 2;
        }
        results.push_back(current);
    }
    return results;
}

FailureOr<PhysicalValues> PlanGraphBuilder::materializeDeint4ToDeint2(const DataMaterialization& context)
{
    auto source = context.source;
    Region* scope = context.scope;

    if (source.size() % 4 != 0 || context.resultArity != context.sourceArity) {
        return failure();
    }
    size_t chunks = source.size() / 4;
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    results.reserve(source.size());
    SmallVector<PhysicalValue> lows;
    SmallVector<PhysicalValue> highs;
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        auto even = graph.addAction(
            PhysicalActionKind::Interleave, scope, {source[chunk], source[chunks * 2 + chunk]}, "d4-d2-even", 2);
        auto odd = graph.addAction(
            PhysicalActionKind::Interleave, scope, {source[chunks + chunk], source[chunks * 3 + chunk]}, "d4-d2-odd",
            2);
        lows.push_back(even[0]);
        lows.push_back(odd[0]);
        highs.push_back(even[1]);
        highs.push_back(odd[1]);
    }
    results.append(lows);
    results.append(highs);
    return results;
}
FailureOr<PhysicalValues> PlanGraphBuilder::materializeDeint2ToDeint4(const DataMaterialization& context)
{
    auto source = context.source;
    Region* scope = context.scope;

    if (source.size() % 4 != 0 || context.resultArity != context.sourceArity) {
        return failure();
    }
    size_t chunks = source.size() / 4;
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    results.reserve(source.size());
    SmallVector<PhysicalValue> part0;
    SmallVector<PhysicalValue> part1;
    SmallVector<PhysicalValue> part2;
    SmallVector<PhysicalValue> part3;
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        size_t lowIndex = 2 * chunk;
        size_t highIndex = 2 * chunks + lowIndex;
        auto even = graph.addAction(
            PhysicalActionKind::Deinterleave, scope, {source[lowIndex], source[highIndex]}, "d2-d4-even", 2);
        auto odd = graph.addAction(
            PhysicalActionKind::Deinterleave, scope, {source[lowIndex + 1], source[highIndex + 1]}, "d2-d4-odd", 2);
        part0.push_back(even[0]);
        part1.push_back(odd[0]);
        part2.push_back(even[1]);
        part3.push_back(odd[1]);
    }
    results.append(part0);
    results.append(part1);
    results.append(part2);
    results.append(part3);
    return results;
}
static bool needsDenseStaging(VMILayoutAttr source, VMILayoutAttr result)
{
    bool deint2ToStride2 = source.isDeinterleaved() && source.getFactor() == 2 && source.getLaneStride() == 1 &&
                           result.isContiguous() && result.getLaneStride() == 2;
    bool stride2ToDeint2 = source.isContiguous() && source.getLaneStride() == 2 && result.isDeinterleaved() &&
                           result.getFactor() == 2 && result.getLaneStride() == 1;
    return deint2ToStride2 || stride2ToDeint2;
}
static bool needsDenseStrideStaging(VMILayoutAttr source, VMILayoutAttr result)
{
    bool stride2ToStride4 =
        source.isContiguous() && source.getLaneStride() == 2 && result.isContiguous() && result.getLaneStride() == 4;
    bool stride4ToStride2 =
        source.isContiguous() && source.getLaneStride() == 4 && result.isContiguous() && result.getLaneStride() == 2;
    return stride2ToStride4 || stride4ToStride2;
}
FailureOr<PhysicalValues> PlanGraphBuilder::materializeDensePacket(const DataMaterialization& context)
{
    auto source = context.source;
    auto sourceLayout = context.sourceLayout;
    auto resultLayout = context.resultLayout;
    Region* scope = context.scope;
    bool denseToPacket = sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
                         resultLayout.isGroupSlots() && resultLayout.getNumGroups() > 0 &&
                         resultLayout.getNumGroups() <= resultLayout.getSlots();
    bool packetToDense = sourceLayout.isGroupSlots() && sourceLayout.getNumGroups() > 0 &&
                         sourceLayout.getNumGroups() <= sourceLayout.getSlots() && resultLayout.isContiguous() &&
                         resultLayout.getLaneStride() == 1;
    if ((denseToPacket || packetToDense) && context.sourceArity == context.resultArity && !source.empty()) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> results;
        results.reserve(source.size());
        for (PhysicalValue part : source) {
            results.push_back(graph.addAction(
                denseToPacket ? PhysicalActionKind::Pack : PhysicalActionKind::Unpack, scope, {part},
                denseToPacket ? "dense-to-packet" : "packet-to-dense", 1)[0]);
        }
        return results;
    }
    return failure();
}
FailureOr<PhysicalValues> PlanGraphBuilder::materializeThroughContiguous(const DataMaterialization& context)
{
    auto contiguous = VMILayoutAttr::getContiguous(context.type.getContext());
    auto first = materialize(context.sourceValue, context.source, context.sourceLayout, contiguous, context.placement);
    if (failed(first)) {
        return failure();
    }
    return materialize(context.sourceValue, *first, contiguous, context.resultLayout, context.placement);
}
FailureOr<PhysicalValues> PlanGraphBuilder::materializeDenseEndpoint(const DataMaterialization& context)
{
    auto source = context.source;
    auto sourceLayout = context.sourceLayout;
    auto resultLayout = context.resultLayout;
    Region* scope = context.scope;
    if (sourceLayout.isDeinterleaved() && resultLayout.isContiguous() && resultLayout.getLaneStride() == 1) {
        return deinterleavedToContiguous(
            source, sourceLayout.getFactor(), static_cast<size_t>(context.resultArity), scope);
    }
    if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 && resultLayout.isDeinterleaved()) {
        return contiguousToDeinterleaved(
            source, resultLayout.getFactor(), static_cast<size_t>(context.resultArity), scope);
    }
    if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() > 1) {
        return contiguousToLaneStride(
            source, resultLayout.getLaneStride(), static_cast<size_t>(context.resultArity), scope);
    }
    if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() > 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == 1) {
        return laneStrideToContiguous(source, sourceLayout.getLaneStride(), scope);
    }
    if (needsDenseStaging(sourceLayout, resultLayout) || needsDenseStrideStaging(sourceLayout, resultLayout)) {
        return materializeThroughContiguous(context);
    }
    return materializeDensePacket(context);
}
FailureOr<PhysicalValues> PlanGraphBuilder::materializeDataRecipe(const DataMaterialization& context)
{
    auto sourceLayout = context.sourceLayout;
    auto resultLayout = context.resultLayout;
    auto source = context.source;
    int64_t sourceArity = context.sourceArity;
    int64_t resultArity = context.resultArity;
    if (sourceLayout.isGroupSlots() && resultLayout.isGroupSlots()) {
        if (sourceArity != resultArity) {
            return failure();
        }
        return groupSlotsToGroupSlots(source, sourceLayout, resultLayout, context.scope);
    }
    if (sourceLayout.isDeinterleaved() && resultLayout.isDeinterleaved() && sourceLayout.getLaneStride() == 1 &&
        resultLayout.getLaneStride() == 1) {
        if (sourceLayout.getFactor() == 4 && resultLayout.getFactor() == 2) {
            return materializeDeint4ToDeint2(context);
        }
        if (sourceLayout.getFactor() == 2 && resultLayout.getFactor() == 4) {
            return materializeDeint2ToDeint4(context);
        }
    }
    if (sourceLayout.isContiguous() || resultLayout.isContiguous()) {
        return materializeDenseEndpoint(context);
    }
    return materializeThroughContiguous(context);
}
FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::materialize(
    Value sourceValue, ArrayRef<PhysicalValue> source, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Operation* placement)
{
    auto type = dyn_cast<VMIVRegType>(sourceValue.getType());
    if (!type || !sourceLayout || !resultLayout) {
        return failure();
    }
    auto sourceArity = getArity(type, sourceLayout);
    auto resultArity = getArity(type, resultLayout);
    if (failed(sourceArity) || failed(resultArity) || *sourceArity <= 0 || *resultArity <= 0 ||
        source.size() != static_cast<size_t>(*sourceArity)) {
        return failure();
    }
    if (sourceLayout == resultLayout) {
        return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    VMILayoutSupport supports;
    auto sourceType = VMIVRegType::get(type.getContext(), type.getElementCount(), type.getElementType(), sourceLayout);
    auto resultType = VMIVRegType::get(type.getContext(), type.getElementCount(), type.getElementType(), resultLayout);
    auto ensureFact = supports.getEnsureLayoutFact(sourceType, resultType);
    if (failed(ensureFact)) {
        return failure();
    }
    if (ensureFact->forwardsPhysicalParts) {
        if (*sourceArity != *resultArity) {
            return failure();
        }
        return SmallVector<PhysicalValue, mlir::pto::kValue4>(source);
    }
    DataMaterialization context{sourceValue,  source,       type,
                                sourceLayout, resultLayout, *sourceArity,
                                *resultArity, placement,    getDynamicScope(placement)};
    return materializeDataRecipe(context);
}

FailureOr<PhysicalValues> PlanGraphBuilder::deint4ToContiguous(
    ArrayRef<PhysicalValue> source, size_t resultCount, Region* scope)
{
    if (source.empty() || resultCount > source.size()) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    size_t groups = (resultCount + 3) / 4;
    size_t base = source.size() / 4;
    size_t remainder = source.size() % 4;
    auto getSource = [&](size_t part, size_t group) {
        size_t count = base + (part < remainder ? 1 : 0);
        size_t offset = part * base + std::min(part, remainder);
        return group < count ? source[offset + group] : source.back();
    };
    for (size_t group = 0; group < groups && results.size() < resultCount; ++group) {
        PhysicalValue p0 = getSource(0, group);
        PhysicalValue p1 = getSource(1, group);
        PhysicalValue p2 = getSource(2, group);
        PhysicalValue p3 = getSource(3, group);
        auto even = graph.addAction(PhysicalActionKind::Interleave, scope, {p0, p2}, "d4-even", 2);
        auto odd = graph.addAction(PhysicalActionKind::Interleave, scope, {p1, p3}, "d4-odd", 2);
        auto low = graph.addAction(PhysicalActionKind::Interleave, scope, {even[0], odd[0]}, "d4-low", 2);
        for (PhysicalValue value : low) {
            if (results.size() == resultCount) {
                break;
            }
            results.push_back(value);
        }
        if (results.size() < resultCount) {
            auto high = graph.addAction(PhysicalActionKind::Interleave, scope, {even[1], odd[1]}, "d4-high", 2);
            for (PhysicalValue value : high) {
                if (results.size() == resultCount) {
                    break;
                }
                results.push_back(value);
            }
        }
    }
    return results;
}
FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::deinterleavedToContiguous(
    ArrayRef<PhysicalValue> source, int64_t factor, size_t resultCount, Region* scope)
{
    if (factor == 2 && source.size() % 2 == 0 && resultCount <= source.size()) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> results;
        size_t groups = source.size() / 2;
        for (size_t group = 0; group < groups && results.size() < resultCount; ++group) {
            auto interleaved = graph.addAction(
                PhysicalActionKind::Interleave, scope, {source[group], source[groups + group]}, "d2", 2);
            for (PhysicalValue value : interleaved) {
                if (results.size() == resultCount) {
                    break;
                }
                results.push_back(value);
            }
        }
        return results;
    }
    if (factor == 4) {
        return deint4ToContiguous(source, resultCount, scope);
    }
    return failure();
}

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::contiguousToDeinterleaved(
    ArrayRef<PhysicalValue> source, int64_t factor, size_t resultCount, Region* scope)
{
    if (factor == 2 && resultCount % 2 == 0 && !source.empty() && source.size() <= resultCount) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> lows;
        SmallVector<PhysicalValue, mlir::pto::kValue4> highs;
        for (size_t part = 0; part < resultCount; part += 2) {
            if (part >= source.size()) {
                return failure();
            }
            PhysicalValue higher = part + 1 < source.size() ? source[part + 1] : source[part];
            auto split = graph.addAction(PhysicalActionKind::Deinterleave, scope, {source[part], higher}, "d2", 2);
            lows.push_back(split[0]);
            highs.push_back(split[1]);
        }
        lows.append(highs);
        return lows;
    }
    if (factor != 4 || resultCount == 0 || resultCount % 4 != 0 || source.empty() || source.size() > resultCount) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> part0;
    SmallVector<PhysicalValue, mlir::pto::kValue4> part1;
    SmallVector<PhysicalValue, mlir::pto::kValue4> part2;
    SmallVector<PhysicalValue, mlir::pto::kValue4> part3;
    size_t groups = resultCount / 4;
    for (size_t group = 0; group < groups; ++group) {
        auto getSource = [&](size_t offset) { return source[std::min(4 * group + offset, source.size() - 1)]; };
        PhysicalValue s0 = getSource(0);
        PhysicalValue s1 = getSource(1);
        PhysicalValue s2 = getSource(2);
        PhysicalValue s3 = getSource(3);
        auto low = graph.addAction(PhysicalActionKind::Deinterleave, scope, {s0, s1}, "d4-low", 2);
        auto high = graph.addAction(PhysicalActionKind::Deinterleave, scope, {s2, s3}, "d4-high", 2);
        auto even = graph.addAction(PhysicalActionKind::Deinterleave, scope, {low[0], high[0]}, "d4-even", 2);
        auto odd = graph.addAction(PhysicalActionKind::Deinterleave, scope, {low[1], high[1]}, "d4-odd", 2);
        part0.push_back(even[0]);
        part1.push_back(odd[0]);
        part2.push_back(even[1]);
        part3.push_back(odd[1]);
    }
    part0.append(part1);
    part0.append(part2);
    part0.append(part3);
    return part0;
}

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::contiguousToLaneStride(
    ArrayRef<PhysicalValue> source, int64_t stride, size_t resultCount, Region* scope)
{
    if ((stride != 2 && stride != 4) || source.empty() || resultCount == 0 ||
        (resultCount + stride - 1) / stride != source.size()) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    results.reserve(resultCount);
    for (size_t resultIndex = 0; resultIndex < resultCount; ++resultIndex) {
        int64_t part = resultIndex % stride;
        PhysicalValue current = source[resultIndex / stride];
        int64_t levels = stride == 4 ? 2 : 1;
        for (int64_t level = 0; level < levels; ++level) {
            int64_t selector = stride == 4 ? (level == 0 ? part / 2 : part % 2) : part;
            current = graph.addAction(
                PhysicalActionKind::Unpack, scope, {current},
                (Twine("level:") + Twine(level) + Twine(":selector:") + Twine(selector)).str(), 1)[0];
        }
        results.push_back(current);
    }
    return results;
}

FailureOr<SmallVector<PhysicalValue, mlir::pto::kValue4>> PlanGraphBuilder::laneStrideToContiguous(
    ArrayRef<PhysicalValue> source, int64_t stride, Region* scope)
{
    if ((stride != 2 && stride != 4) || source.empty()) {
        return failure();
    }
    SmallVector<PhysicalValue, mlir::pto::kValue4> results;
    for (size_t begin = 0; begin < source.size(); begin += stride) {
        SmallVector<PhysicalValue, mlir::pto::kValue4> level(
            source.slice(begin, std::min<size_t>(stride, source.size() - begin)));
        int64_t levels = stride == 4 ? 2 : 1;
        for (int64_t depth = 0; depth < levels; ++depth) {
            SmallVector<PhysicalValue, mlir::pto::kValue4> next;
            for (size_t index = 0; index < level.size(); index += 2) {
                PhysicalValue low = graph.addAction(PhysicalActionKind::Pack, scope, {level[index]}, "lower")[0];
                if (index + 1 == level.size()) {
                    next.push_back(low);
                    continue;
                }
                PhysicalValue high = graph.addAction(PhysicalActionKind::Pack, scope, {level[index + 1]}, "higher")[0];
                next.push_back(graph.addAction(PhysicalActionKind::Merge, scope, {low, high}, "vor")[0]);
            }
            level = std::move(next);
        }
        if (level.size() != 1) {
            return failure();
        }
        results.push_back(level.front());
    }
    return results;
}

} // namespace mlir::pto::layout_cost
