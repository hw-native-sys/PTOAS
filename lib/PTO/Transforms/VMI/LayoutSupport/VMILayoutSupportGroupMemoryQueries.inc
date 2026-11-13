// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Group memory queries over the private layout tables in VMILayoutSupport.cpp.
#ifndef PTO_VMILAYOUTSUPPORTGROUPMEMORYQUERIES_INC
#define PTO_VMILAYOUTSUPPORTGROUPMEMORYQUERIES_INC
FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupSlotLoadLayoutFact(
    VMIVRegType resultType, int64_t numGroups, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<VMIGroupSlotLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout) {
        return fail("requires assigned result layout");
    }

    if (!isSupportedGroupSlotMemoryLayout(layout, numGroups)) {
        return fail("result layout does not match a supported group_slot_load "
                    "table row");
    }

    return VMIGroupSlotLayoutFact{layout, numGroups, layout.getSlots()};
}

FailureOr<VMIGroupLoadLayoutFact> VMILayoutSupport::getGroupLoadLayoutFact(VMIGroupLoadOp op, std::string* reason) const
{
    return getGroupLoadLayoutFact(
        cast<VMIVRegType>(op.getResult().getType()), op.getRowStride(), op.getNumGroupsAttr().getInt(), reason);
}

FailureOr<VMIGroupLoadLayoutFact> VMILayoutSupport::getGroupLoadLayoutFact(
    VMIVRegType resultType, Value rowStride, int64_t numGroups, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<VMIGroupLoadLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout) {
        return fail("requires assigned result layout");
    }

    unsigned elementBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
    if (elementBits == 0) {
        return fail("group_load requires known element bit width");
    }
    std::optional<int64_t> stride = getConstantIndexValue(rowStride);

    FailureOr<GroupLayoutKey> key =
        buildGroupLayoutKey(resultType, numGroups, "group_load layout table has no row for this group size", reason);
    if (failed(key)) {
        return failure();
    }

    for (const GroupLoadLayoutPattern& pattern : kGroupLoadLayoutPatterns) {
        if (pattern.requiresF32 && !resultType.getElementType().isF32()) {
            continue;
        }
        if (!matchesElementBitsPattern(pattern.elementBits, elementBits)) {
            continue;
        }
        if (!matchesGroupBlockPattern(pattern.block, *key)) {
            continue;
        }
        if (!matchesGroupLoadMemoryPattern(pattern.memory, stride, key->groupSize, elementBits)) {
            continue;
        }
        if (!matchesLayoutPattern(resultType.getContext(), pattern.resultLayout, layout)) {
            continue;
        }
        return VMIGroupLoadLayoutFact{getGroupBlockClassFromPattern(pattern.block), layout, key->groupSize};
    }

    return fail("result layout, group size, and row_stride do not match a "
                "supported group_load table row");
}

FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupStoreLayoutFact(
    VMIVRegType valueType, int64_t numGroups, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<VMIGroupSlotLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    VMILayoutAttr layout = valueType.getLayoutAttr();
    if (!layout) {
        return fail("requires assigned value layout");
    }
    if (!isSupportedGroupSlotMemoryLayout(layout, numGroups)) {
        return fail("value layout does not match a supported group_store table "
                    "row");
    }
    return VMIGroupSlotLayoutFact{layout, numGroups, layout.getSlots()};
}

static VMILayoutAttr getGroupStoreStagingLayout(
    VMIGroupStoreOp op, VMIVRegType valueType, VMILayoutAttr layout, int64_t numGroups)
{
    VMILayoutAttr stagingLayout;
    unsigned elementBits = pto::getPTOStorageElemBitWidth(valueType.getElementType());
    int64_t payloadBits = valueType.getElementCount() * static_cast<int64_t>(elementBits);
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    bool compactSmallStore = layout.getSlots() == 8 && layout.getLaneStride() != 1 &&
                             (layout.getLaneStride() == 2 || layout.getLaneStride() == 4) &&
                             (valueType.getElementCount() == 4 || valueType.getElementCount() == 8) &&
                             numGroups == valueType.getElementCount() && elementBits > 0 && payloadBits > 0 &&
                             payloadBits < 256 && payloadBits % 32 == 0 && rowStride && *rowStride == 1;
    if (compactSmallStore) {
        stagingLayout = VMILayoutAttr::getGroupSlots(valueType.getContext(), layout.getNumGroups(), layout.getSlots());
    }
    return stagingLayout;
}
FailureOr<VMIGroupStoreLayoutFact> VMILayoutSupport::getGroupStoreLayoutFact(
    VMIGroupStoreOp op, VMIVRegType valueType, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<VMIGroupStoreLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    VMILayoutAttr layout = valueType.getLayoutAttr();
    if (!layout) {
        return fail("requires assigned value layout");
    }

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (layout.isGroupSlots()) {
        if (failed(getGroupStoreLayoutFact(valueType, numGroups, reason))) {
            return failure();
        }
        // A short packet of four or eight group values whose payload does not fill
        // one physical chunk cannot be written in its lane-strided form directly:
        // the store first packs it into unit-stride group slots, and that staging
        // layout is what the cost model has to charge.
        VMILayoutAttr stagingLayout = getGroupStoreStagingLayout(op, valueType, layout, numGroups);
        return VMIGroupStoreLayoutFact{layout, stagingLayout};
    }

    if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) == 0) {
        return fail("group_store requires known element bit width");
    }

    FailureOr<GroupLayoutKey> key =
        buildGroupLayoutKey(valueType, numGroups, "group_store layout table has no row for this group size", reason);
    if (failed(key)) {
        return failure();
    }

    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    for (const GroupStoreLayoutPattern& pattern : kGroupStoreLayoutPatterns) {
        if (!matchesGroupStoreLayoutPattern(pattern, valueType, *key, rowStride)) {
            continue;
        }
        if (!matchesLayoutPattern(valueType.getContext(), pattern.valueLayout, layout)) {
            continue;
        }
        VMIGroupStoreLayoutFact fact = materializeGroupStoreLayoutFact(valueType.getContext(), pattern, *key);
        fact.valueLayout = layout;
        return fact;
    }

    return fail("value layout, group size, and row_stride do not match a "
                "supported group_store table row");
}

FailureOr<SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getGroupStoreLayoutFactsForLayout(
    VMIGroupStoreOp op, VMIVRegType valueType, VMILayoutAttr layout, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<SmallVector<VMIGroupStoreLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    if (!layout) {
        return fail("requires assigned group_store value layout");
    }

    // A group-slot layout on a vreg describes exactly one logical lane per group,
    // which the type verifier states as num_groups == element count (a two lane
    // value may carry num_groups = 2, slots = 8; a 128 lane value may not).  The
    // assigned type built below is constructed directly rather than parsed, so the
    // verifier never runs on it: without this check the query returns a fact for a
    // type that cannot exist, the relation is exposed, and no lowering can realize
    // it.  Measured on vmi_layout_cost_conformance_group_memory.pto, where the
    // group_store of a 128xf32 value with num_groups = 2 produced exactly that
    // candidate and the cost model - correctly - refused to price it, which aborts
    // the whole conformance run at that case.
    bool groupSlotsLaneMismatch = layout.isGroupSlots() && layout.getNumGroups() != valueType.getElementCount();
    if (groupSlotsLaneMismatch) {
        return fail("group_store value layout does not describe one lane per group");
    }

    MLIRContext* ctx = valueType.getContext();
    auto sourceType = VMIVRegType::get(ctx, valueType.getElementCount(), valueType.getElementType(), layout);
    FailureOr<VMIGroupStoreLayoutFact> directFact = getGroupStoreLayoutFact(op, sourceType, nullptr);
    if (succeeded(directFact)) {
        return SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>{*directFact};
    }

    FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
        valueType, op.getNumGroupsAttr().getInt(), "group_store layout table has no row for this group size", reason);
    if (failed(key)) {
        return failure();
    }

    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4> facts;
    appendMaterializableGroupStoreFacts(*this, {valueType, sourceType, *key, rowStride}, facts);

    if (facts.empty()) {
        return fail("value layout cannot be used directly or materialized to a "
                    "supported group_store layout table row");
    }
    return facts;
}

FailureOr<VMIGroupStoreLayoutFact> VMILayoutSupport::getPreferredGroupStoreLayoutFact(
    VMIGroupStoreOp op, VMIVRegType valueType, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<VMIGroupStoreLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    std::optional<VMIGroupStoreLayoutFact> slotFact = getPreferredGroupSlotStoreFact(*this, op, valueType, numGroups);
    if (slotFact) {
        return *slotFact;
    }

    if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) == 0) {
        return fail("group_store requires known element bit width");
    }
    FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
        valueType, numGroups, "group_store preferred layout table has no row for this group size", reason);
    if (failed(key)) {
        return failure();
    }

    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    const GroupStoreLayoutPattern* selected = selectPreferredGroupStorePattern(valueType, *key, rowStride);
    if (selected) {
        return materializeGroupStoreLayoutFact(valueType.getContext(), *selected, *key);
    }

    return fail("value type, group size, and row_stride do not match a "
                "preferred group_store table row");
}

FailureOr<VMIGroupStoreLayoutFact> VMILayoutSupport::getHighPriorityGroupStoreLayoutFact(
    VMIGroupStoreOp op, VMIVRegType valueType, std::string* reason) const
{
    auto fail = [reason](const Twine& message) -> FailureOr<VMIGroupStoreLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) == 0) {
        return fail("group_store requires known element bit width");
    }
    FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
        valueType, op.getNumGroupsAttr().getInt(),
        "high-priority group_store layout table has no row for this group size", reason);
    if (failed(key)) {
        return failure();
    }

    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    for (const GroupStoreLayoutPattern& pattern : kGroupStoreLayoutPatterns) {
        if (pattern.priority != GroupStoreLayoutPriority::High) {
            continue;
        }
        if (!matchesGroupStoreLayoutPattern(pattern, valueType, *key, rowStride)) {
            continue;
        }
        VMIGroupStoreLayoutFact fact = materializeGroupStoreLayoutFact(valueType.getContext(), pattern, *key);
        if (!fact.valueLayout) {
            continue;
        }
        return fact;
    }

    return fail("value type, group size, and row_stride do not match a "
                "high-priority group_store table row");
}

template <typename OpTy>

#endif // PTO_VMILAYOUTSUPPORTGROUPMEMORYQUERIES_INC
