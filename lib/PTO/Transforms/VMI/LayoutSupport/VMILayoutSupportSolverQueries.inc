// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTSOLVERQUERIES_INC
#define VMILAYOUTSUPPORTSOLVERQUERIES_INC

//===- VMILayoutSupportSolverQueries.inc - solver-driven queries -*- C++ -*-===//
// Textual include unit of VMILayoutSupport.cpp, included at the end of the
// translation unit so the definitions below can use the file-local table
// helpers.

// It holds the candidate-enumerating queries the costed layout planner drives.
// The planner picks a relation for a whole component, so it enumerates the facts
// of a table for a shape instead of validating one assigned layout; upstream's
// per-layout queries answer a different question and are left untouched.  No
// upstream pass calls anything in this unit.
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<VMILoadLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getLoadLayoutFacts(
    VMIVRegType resultType, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMILoadLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    SmallVector<VMILoadLayoutFact, mlir::pto::kValue4> facts;
    for (const DenseMemoryLayoutPattern& pattern : kDenseLoadLayoutPatterns) {
        if (!matchesElementBitsPattern(pattern.elementBits, resultType.getElementType())) {
            continue;
        }
        if (!matchesElementCountPattern(pattern.elementCounts, resultType.getElementCount())) {
            continue;
        }
        VMILayoutAttr layout = materializeLayoutPattern(resultType.getContext(), pattern.layout);
        if (!layout ||
            llvm::any_of(facts, [&](const VMILoadLayoutFact& fact) { return fact.resultLayout == layout; })) {
            continue;
        }
        facts.push_back(VMILoadLayoutFact{layout});
    }
    if (facts.empty()) {
        return fail("result type does not match a supported dense load table row");
    }
    return facts;
}

FailureOr<SmallVector<VMIStoreLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getStoreLayoutFacts(
    VMIVRegType valueType, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMIStoreLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    SmallVector<VMIStoreLayoutFact, mlir::pto::kValue4> facts;
    for (const DenseMemoryLayoutPattern& pattern : kDenseStoreLayoutPatterns) {
        if (!matchesElementBitsPattern(pattern.elementBits, valueType.getElementType())) {
            continue;
        }
        if (!matchesElementCountPattern(pattern.elementCounts, valueType.getElementCount())) {
            continue;
        }
        VMILayoutAttr layout = materializeLayoutPattern(valueType.getContext(), pattern.layout);
        if (!layout ||
            llvm::any_of(facts, [&](const VMIStoreLayoutFact& fact) { return fact.valueLayout == layout; })) {
            continue;
        }
        facts.push_back(VMIStoreLayoutFact{layout});
    }
    if (facts.empty()) {
        return fail("value type does not match a supported dense store table row");
    }
    return facts;
}

FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getCastLayoutFacts(
    VMIVRegType sourceType, VMIVRegType resultType, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMICastLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
    MLIRContext* ctx = sourceType.getContext();

    // A group-slot cast carries its group count on the type; the enumeration has
    // no single assigned layout to read it from, so it comes from whichever side
    // is a group-slot layout.  Upstream's per-layout entry point derives the same
    // count from its port layout, which is a materialized source layout here.
    int64_t numGroups = 0;
    if (VMILayoutAttr sourceLayout = sourceType.getLayoutAttr(); sourceLayout && sourceLayout.isGroupSlots()) {
        numGroups = sourceLayout.getNumGroups();
    }
    if (VMILayoutAttr resultLayout = resultType.getLayoutAttr(); resultLayout && resultLayout.isGroupSlots()) {
        if (numGroups != 0 && numGroups != resultLayout.getNumGroups()) {
            return fail("source/result group-slot layouts require matching "
                        "num_groups");
        }
        numGroups = resultLayout.getNumGroups();
    }

    // The shared funnel filters by port layout, so drive it once per source
    // layout this width pair can materialize.  Two rows that agree on the source
    // layout are answered together by one call, so no relation is returned twice.
    SmallVector<VMILayoutAttr, mlir::pto::kValue4> sourceLayouts;
    for (const LegalCastLayoutPattern& pattern : kLegalCastLayoutPatterns) {
        if (!matchesElementBitsPattern(pattern.sourceBits, sourceBits) ||
            !matchesElementBitsPattern(pattern.resultBits, resultBits) ||
            !matchesCastTypeClass(pattern.typeClass, sourceType.getElementType(), resultType.getElementType())) {
            continue;
        }
        VMILayoutAttr sourceLayout = materializeLayoutPattern(ctx, pattern.sourceLayout, numGroups);
        if (!sourceLayout || llvm::any_of(sourceLayouts, [&](VMILayoutAttr seen) { return seen == sourceLayout; })) {
            continue;
        }
        sourceLayouts.push_back(sourceLayout);
    }

    SmallVector<VMICastLayoutFact, mlir::pto::kValue4> facts;
    for (VMILayoutAttr sourceLayout : sourceLayouts) {
        FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> rowFacts =
            getCastLayoutFactsForLayout(sourceType, resultType, VMICastLayoutPort::Source, sourceLayout, nullptr);
        if (failed(rowFacts)) {
            continue;
        }
        for (const VMICastLayoutFact& fact : *rowFacts) {
            // A row whose intrinsic rearrangement cost cannot be derived is not a
            // candidate for the costed planner: every candidate is priced.  This is
            // the fork's own rule (fork VMILayoutSupport.cpp:2315-2319) and it lives
            // here, in the solver-driven enumeration, never in the shared funnel,
            // where dropping a row would narrow upstream's own decisions (plan 11.5).
            FailureOr<int64_t> intrinsicCost =
                getCastIntrinsicRearrangementCost(sourceType, resultType, fact.sourceLayout, fact.resultLayout);
            if (failed(intrinsicCost)) {
                continue;
            }
            VMICastLayoutFact priced = fact;
            priced.intrinsicRearrangementCost = *intrinsicCost;
            facts.push_back(priced);
        }
    }

    if (facts.empty()) {
        return fail("requires a legal cast relation for the source/result types");
    }
    return facts;
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getSameWidthCastLayoutFact(
    VMIVRegType sourceType, VMIVRegType resultType, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<VMICastLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    if (!sourceLayout || !resultLayout) {
        return fail("same-width cast requires assigned source/result layouts");
    }
    auto [sourceBits, resultBits] = getCastElementBits(sourceType, resultType);
    if (sourceBits == 0 || sourceBits != resultBits) {
        return fail("same-width cast requires equal known storage widths");
    }
    if (sourceLayout != resultLayout) {
        return fail("same-width cast requires matching source/result layouts");
    }
    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
    FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
    if (failed(sourceArity) || failed(resultArity) || *sourceArity != *resultArity) {
        return fail("same-width cast requires matching computable physical arity");
    }
    return makeCastLayoutFact(sourceBits, resultBits, sourceLayout, resultLayout);
}

LogicalResult VMILayoutSupport::validateCastOperationRelation(
    Operation* op, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, std::string* reason) const
{
    if (!op || !sourceLayout || !resultLayout) {
        if (reason) {
            *reason = "cast operation relation requires operation and layouts";
        }
        return failure();
    }
    return success();
}

/// Returns failure when a contiguous source value does not fill its physical
/// chunks, so its padding lanes cannot be told apart from real data.  The reason
/// is a neutral requirement fragment because the caller prefixes its own subject
/// rather than borrowing another op family's wording.
static LogicalResult checkSourceFillsPhysicalChunks(VMIVRegType sourceType, std::string* reason)
{
    auto fail = [&](const Twine& message) -> LogicalResult {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };
    FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(sourceType.getElementType());
    if (failed(lanesPerPart) || *lanesPerPart <= 0) {
        return fail("known physical source chunk width");
    }
    if (sourceType.getElementCount() % *lanesPerPart != 0) {
        return fail("source element count to fill complete physical chunks");
    }
    return success();
}

FailureOr<VMIReduceLayoutFact> VMILayoutSupport::getReduceLayoutFactForLayouts(
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<VMIReduceLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr maskLayout = maskType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    if (!sourceLayout || !maskLayout || !resultLayout) {
        return fail("reduce requires assigned source, mask, and result layouts");
    }
    if (!sourceLayout.isContiguous() || !maskLayout.isContiguous() || !resultLayout.isContiguous()) {
        return fail("reduce requires contiguous source, mask, and result layouts");
    }

    // The legacy vc* lowering consumes complete physical source chunks, so a
    // contiguous value whose element count leaves padding lanes cannot be
    // reduced without folding those padding lanes into the result.  The rule
    // lives in the shared helper so the planner never advertises a relation that
    // lowering must reject; only the wording below is reduce-specific.
    std::string chunkReason;
    if (failed(checkSourceFillsPhysicalChunks(sourceType, &chunkReason))) {
        return fail(Twine("reduce requires ") + chunkReason);
    }
    return VMIReduceLayoutFact{sourceLayout, maskLayout, resultLayout};
}

static FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> getAllInterleaveLayoutFactsImpl(
    ArrayRef<InterleaveLayoutPattern> patterns, VMIVRegType valueType, std::string* reason)
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMIInterleaveLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    FailureOr<InterleaveLayoutKey> key = buildInterleaveLayoutKey(valueType, reason);
    if (failed(key)) {
        return failure();
    }

    SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4> facts;
    for (const InterleaveLayoutPattern& pattern : patterns) {
        if (matchesInterleaveLayoutPattern(pattern, valueType, *key)) {
            facts.push_back(materializeInterleaveLayoutFact(valueType.getContext(), pattern, *key));
        }
    }
    if (facts.empty()) {
        return fail("requires a matching interleave layout table row");
    }
    return facts;
}

FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getVintlvLayoutFacts(
    VMIVRegType valueType, std::string* reason) const
{
    return getAllInterleaveLayoutFactsImpl(kVintlvLayoutPatterns, valueType, reason);
}

FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getVdintlvLayoutFacts(
    VMIVRegType valueType, std::string* reason) const
{
    return getAllInterleaveLayoutFactsImpl(kVdintlvLayoutPatterns, valueType, reason);
}

static SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4> enumerateGroupBroadcastLoadFacts(
    VMIVRegType resultType, int64_t numGroups, unsigned elementBits, std::optional<int64_t> stride,
    const GroupLayoutKey& key)
{
    SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4> facts;
    for (const GroupBroadcastLoadLayoutPattern& pattern : kGroupBroadcastLoadLayoutPatterns) {
        if (!matchesGroupBlockPattern(pattern.block, key) ||
            !matchesElementCountPattern(pattern.numGroups, numGroups) ||
            !matchesElementBitsPattern(pattern.elementBits, elementBits) ||
            !matchesGroupBroadcastLoadMemoryPattern(pattern.memory, stride, elementBits)) {
            continue;
        }
        VMILayoutAttr resultLayout = materializeLayoutPattern(resultType.getContext(), pattern.resultLayout, numGroups);
        if (!resultLayout) {
            continue;
        }
        // Rows that differ only in their block spelling can materialize the same
        // result layout; the fork keeps one fact per layout.
        bool duplicate = llvm::any_of(
            facts, [&](const VMIGroupBroadcastLoadLayoutFact& fact) { return fact.resultLayout == resultLayout; });
        if (duplicate) {
            continue;
        }
        facts.push_back(VMIGroupBroadcastLoadLayoutFact{
            getGroupBlockClassFromPattern(pattern.block), resultLayout, key.groupSize, key.lanesPerPart,
            key.vcgBlockElems, static_cast<int64_t>(elementBits)});
    }
    return facts;
}

FailureOr<SmallVector<VMIGroupBroadcastLoadLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getGroupBroadcastLoadLayoutFacts(VMIGroupBroadcastLoadOp op, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMIGroupBroadcastLoadLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    unsigned elementBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
    if (elementBits == 0) {
        return fail("group_broadcast_load requires known element bit width");
    }
    std::optional<int64_t> stride = getConstantIndexValue(op.getSourceGroupStride());
    FailureOr<GroupLayoutKey> key = buildGroupLayoutKey(
        resultType, numGroups, "group_broadcast_load layout table has no row for this group size", reason);
    if (failed(key)) {
        return failure();
    }

    // Upstream's table is the row set.  The fork's one unique row
    // {gb(1), 16/32-bit, contiguous, d(4)} is not appended here on purpose:
    // upstream's getGroupBroadcastLoadSupport -- the lowering's own gate -- is
    // table-driven off these very rows, so a row added only to this enumeration
    // would be selectable by the solver and rejected by the lowering.
    auto facts = enumerateGroupBroadcastLoadFacts(resultType, numGroups, elementBits, stride, *key);
    if (facts.empty()) {
        int64_t alignedStrideElems = kLayoutBlockBitWidth / elementBits;
        return fail(
            Twine("group_broadcast_load requires a table row for result "
                  "layout, group size, and either constant unit "
                  "source_group_stride or constant positive "
                  "source_group_stride divisible by ") +
            Twine(alignedStrideElems) + " elements");
    }
    return facts;
}

FailureOr<SmallVector<VMIGroupIotaLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getGroupIotaLayoutFacts(
    VMIVRegType resultType, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMIGroupIotaLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    // group_iota is the internal lowering of grouped vci.  Its physical
    // producer is the contiguous vci instruction; non-contiguous consumers are
    // represented by an explicit ensure_layout edge.
    VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(resultType.getContext());
    if (VMILayoutAttr explicitLayout = resultType.getLayoutAttr(); explicitLayout && explicitLayout != contiguous) {
        return fail("group_iota directly produces only contiguous layout");
    }
    return SmallVector<VMIGroupIotaLayoutFact, 4>{VMIGroupIotaLayoutFact{contiguous}};
}

template <typename OpTy>
static FailureOr<VMIHistogramLayoutFact> getPreferredHistogramLayoutFactImpl(
    OpTy op, ArrayRef<HistogramLayoutPattern> patterns, StringRef opName, std::string* reason)
{
    if (patterns.empty()) {
        if (reason) {
            *reason = (opName + " histogram layout table has no row").str();
        }
        return failure();
    }
    const HistogramLayoutPattern& pattern = patterns.front();
    MLIRContext* ctx = op.getContext();
    return VMIHistogramLayoutFact{
        materializeLayoutPattern(ctx, pattern.accLayout), materializeLayoutPattern(ctx, pattern.sourceLayout),
        materializeLayoutPattern(ctx, pattern.maskLayout), materializeLayoutPattern(ctx, pattern.resultLayout)};
}

FailureOr<VMIHistogramLayoutFact> VMILayoutSupport::getPreferredVdhistLayoutFact(
    VMIVdhistOp op, std::string* reason) const
{
    return getPreferredHistogramLayoutFactImpl(op, kVdhistLayoutPatterns, "vdhist", reason);
}

FailureOr<VMIHistogramLayoutFact> VMILayoutSupport::getPreferredVchistLayoutFact(
    VMIVchistOp op, std::string* reason) const
{
    return getPreferredHistogramLayoutFactImpl(op, kVdhistLayoutPatterns, "vchist", reason);
}

/// True when the scalar broadcast load path can read this element width.  A
/// slots = 1 group slot holds one group value that a consumer broadcasts, so it
/// materializes as one scalar broadcast load per slot.  The lowering gates on the
/// same element-width rule; keep it in one place.
static bool isScalarBroadcastLoadElementType(Type elementType)
{
    unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
    return elementBits == 8 || elementBits == 16 || elementBits == 32;
}

FailureOr<VMIGroupSlotLayoutFact> VMILayoutSupport::getGroupSlotLoadLayoutFact(
    VMIVRegType resultType, Value sourceGroupStride, int64_t numGroups, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<VMIGroupSlotLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    auto baseFact = getGroupSlotLoadLayoutFact(resultType, numGroups, reason);
    if (failed(baseFact)) {
        return failure();
    }
    VMILayoutAttr layout = baseFact->layout;
    std::optional<int64_t> stride = getConstantIndexValue(sourceGroupStride);
    if (layout.getSlots() == 8) {
        if (sourceGroupStride && (!stride || *stride != 1)) {
            return fail("slots=8 group_slot_load requires constant unit "
                        "source_group_stride");
        }
    } else if (layout.getSlots() == 1 && sourceGroupStride) {
        if (!isScalarBroadcastLoadElementType(resultType.getElementType())) {
            return fail("slots=1 group_slot_load requires an 8/16/32-bit element "
                        "type so each slot can be read by a scalar broadcast load");
        }
        // Each slot is read from its own element address, so the group stride has
        // to be a known positive constant.  A scalar broadcast load has no 32B
        // block-alignment requirement, so no divisibility rule applies here.
        if (!stride || *stride <= 0) {
            return fail("slots=1 group_slot_load requires constant positive "
                        "source_group_stride");
        }
    }

    return VMIGroupSlotLayoutFact{layout, numGroups, layout.getSlots()};
}

#endif // VMILAYOUTSUPPORTSOLVERQUERIES_INC
