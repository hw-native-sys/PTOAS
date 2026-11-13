// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTSOLVERCOSTS_INC
#define VMILAYOUTSUPPORTSOLVERCOSTS_INC

//===- VMILayoutSupportSolverCosts.inc - cast intrinsic costs -*- C++ -*-===//
// Textual include unit of VMILayoutSupport.cpp, included next to the rule
// tables so the cost helpers are visible to the table funnels.

// A cast lowers as a single operation, so the rearrangement instructions its
// own lowering performs (the per-part pack/unpack steps of a group-slot
// extension, or the predicate-carrier steps of a granularity conversion) never
// appear in the physical graph the cost model builds.  The fact therefore has
// to carry them; a field that is declared but never set would silently charge
// zero for every cast.
//===----------------------------------------------------------------------===//

/// The physical predicate carrier of a mask keeps the logical layout kind while
/// dropping the lane stride - the stride is exactly what the physical
/// granularity counts.  Mirrors getVMIMaskPhysicalCarrierLayout in the VPTO
/// lowering.
inline FailureOr<VMILayoutAttr> getMaskPhysicalCarrierLayout(VMIMaskType type)
{
    auto layout = getVMIMaskCarrierLayout(type.getLayoutAttr());
    return layout ? FailureOr<VMILayoutAttr>(layout) : FailureOr<VMILayoutAttr>(failure());
}

/// Physical predicate carrier width of a mask: its granularity stepped by the
/// layout lane stride, which has to stay a supported carrier.
inline FailureOr<StringRef> getMaskPhysicalGranularity(VMIMaskType type)
{
    int64_t bits = getMaskGranularityBits(type.getGranularity());
    if (bits == 0) {
        return failure();
    }
    VMILayoutAttr layout = type.getLayoutAttr();
    int64_t laneStride = layout && layout.hasLaneStride() ? layout.getLaneStride() : 1;
    StringRef physicalGranularity = getMaskGranularityNameForBits(bits * laneStride);
    if (physicalGranularity.empty()) {
        return failure();
    }
    return physicalGranularity;
}

/// Intrinsic rearrangement instructions performed inside casts. BF16 widening
/// creates one interleave per source carrier for a same-layout pair; integer
/// group-slot widening performs one unpack per carrier level. Every other cast
/// shape is charged through its materialized layout relation.
inline FailureOr<int64_t> getCastIntrinsicRearrangementCost(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout)
{
    if (!sourceLayout || !resultLayout) {
        return failure();
    }
    // Same-layout BF16 widening builds each F32 pair with vintlv(zero, src).
    // This shuffle is internal to extf, so no ensure_layout edge accounts for
    // it. A lane-strided load followed by a native vcvt avoids this shuffle.
    if (sourceType.getElementType().isBF16() && resultType.getElementType().isF32() && sourceLayout == resultLayout) {
        auto assignedSource = VMIVRegType::get(
            sourceType.getContext(), sourceType.getElementCount(), sourceType.getElementType(), sourceLayout);
        auto assignedResult = VMIVRegType::get(
            resultType.getContext(), resultType.getElementCount(), resultType.getElementType(), resultLayout);
        if (isBF16SameLayoutCastPair(assignedSource, assignedResult, sourceLayout, resultLayout)) {
            return getVMIPhysicalArity(assignedSource);
        }
    }
    if (!isa<IntegerType>(sourceType.getElementType()) || !isa<IntegerType>(resultType.getElementType()) ||
        !sourceLayout.isGroupSlots() || !resultLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() || sourceLayout.getLaneStride() != 1 ||
        resultLayout.getLaneStride() != 1 ||
        (sourceLayout.getSlots() != 2 && sourceLayout.getSlots() != 4 && sourceLayout.getSlots() != 8)) {
        return int64_t(0);
    }

    int64_t sourceBits = pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    int64_t resultBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
    if (sourceBits <= 0 || resultBits <= sourceBits || resultBits % sourceBits != 0) {
        return int64_t(0);
    }
    int64_t factor = resultBits / sourceBits;
    int64_t levels = 0;
    while (factor > 1 && factor % 2 == 0) {
        factor /= 2;
        ++levels;
    }
    if (factor != 1) {
        return failure();
    }

    auto assignedSource = VMIVRegType::get(
        sourceType.getContext(), sourceType.getElementCount(), sourceType.getElementType(), sourceLayout);
    auto sourceArity = getVMIPhysicalArity(assignedSource);
    if (failed(sourceArity) || *sourceArity <= 0 || *sourceArity > std::numeric_limits<int64_t>::max() / levels) {
        return failure();
    }
    return *sourceArity * levels;
}

/// Rearrangement instructions a mask granularity conversion performs itself:
/// one step per predicate carrier the conversion walks from the source's
/// physical granularity to the result's.  Widening re-materializes one carrier
/// per result part; narrowing has to read the wide source carriers and merge
/// the surplus ones, so it pays for both sides of the step.
inline FailureOr<int64_t> getMaskGranularityIntrinsicCost(
    VMIMaskType sourceType, VMIMaskType resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout)
{
    auto assignedSource = VMIMaskType::get(
        sourceType.getContext(), sourceType.getElementCount(), sourceType.getGranularity(), sourceLayout);
    auto assignedResult = VMIMaskType::get(
        resultType.getContext(), resultType.getElementCount(), resultType.getGranularity(), resultLayout);
    auto sourceGranularity = getMaskPhysicalGranularity(assignedSource);
    auto resultGranularity = getMaskPhysicalGranularity(assignedResult);
    auto physicalSourceLayout = getMaskPhysicalCarrierLayout(assignedSource);
    if (failed(sourceGranularity) || failed(resultGranularity) || failed(physicalSourceLayout)) {
        return failure();
    }
    int64_t currentBits = getMaskGranularityBits(*sourceGranularity);
    int64_t targetBits = getMaskGranularityBits(*resultGranularity);
    if (currentBits == 0 || targetBits == 0) {
        return failure();
    }

    int64_t cost = 0;
    VMIMaskType currentType = VMIMaskType::get(
        sourceType.getContext(), sourceType.getElementCount(), *sourceGranularity, *physicalSourceLayout);
    while (currentBits != targetBits) {
        int64_t nextBits = currentBits < targetBits ? currentBits * 2 : currentBits / 2;
        StringRef nextGranularity = getMaskGranularityNameForBits(nextBits);
        if (nextGranularity.empty()) {
            return failure();
        }
        VMIMaskType nextType = VMIMaskType::get(
            sourceType.getContext(), sourceType.getElementCount(), nextGranularity, *physicalSourceLayout);
        auto currentArity = getVMIPhysicalArity(currentType);
        auto nextArity = getVMIPhysicalArity(nextType);
        if (failed(currentArity) || failed(nextArity) || *currentArity <= 0 || *nextArity <= 0) {
            return failure();
        }
        int64_t stepCost =
            currentBits < nextBits ? *nextArity : *currentArity + std::max<int64_t>(0, *currentArity - *nextArity);
        if (cost > std::numeric_limits<int64_t>::max() - stepCost) {
            return failure();
        }
        cost += stepCost;
        currentType = nextType;
        currentBits = nextBits;
    }
    return cost;
}

#endif // VMILAYOUTSUPPORTSOLVERCOSTS_INC
