// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTVEXPDIFQUERIES_INC
#define VMILAYOUTSUPPORTVEXPDIFQUERIES_INC

//===- VMILayoutSupportVexpdifQueries.inc - vexpdif layout queries -*- C++ -*-===//
// Textual include unit of VMILayoutSupport.cpp, included at the end of the
// translation unit so the definitions below can use the file-local table
// helpers.  It holds the two pto.vmi.vexpdif layout queries: the per-layout
// candidate enumeration and the preference, both read through the
// kVexpdifLayoutPatterns table and the matchesVexpdifPhysicalShape predicate.
//===----------------------------------------------------------------------===//

static constexpr StringLiteral kVexpdifLayoutRequirement =
    "vexpdif requires an f32 source whose x/max/mask/result share one dense "
    "lane_stride=1 layout, or an f16 source whose x/max/mask are contiguous "
    "and whose result is deinterleaved = 2";

/// Shared op-level precondition: the lowering always emits pto.vexpdif with an
/// implicit zero passthru, so a merge predicate has no lowering at all.
static bool hasUnsupportedVexpdifPmode(VMIVexpdifOp op)
{
    std::optional<StringRef> pmode = op.getPmode();
    return pmode && *pmode == "merge";
}

FailureOr<SmallVector<VMIVexpdifLayoutFact, mlir::pto::kValue4>> VMILayoutSupport::getVexpdifLayoutFactsForLayout(
    VMIVexpdifOp op, VMIVexpdifLayoutPort port, VMILayoutAttr layout, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<SmallVector<VMIVexpdifLayoutFact, 4>> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    if (!layout) {
        return fail("requires an assigned vexpdif query layout");
    }
    if (hasUnsupportedVexpdifPmode(op)) {
        return fail("merge predicate mode requires explicit passthru lowering");
    }

    auto sourceType = dyn_cast<VMIVRegType>(op.getX().getType());
    auto maskType = dyn_cast<VMIMaskType>(op.getMask().getType());
    auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
    if (!sourceType || !maskType || !resultType) {
        return fail("vexpdif requires vreg x/max and mask operands");
    }

    MLIRContext* ctx = op.getContext();
    SmallVector<VMIVexpdifLayoutFact, mlir::pto::kValue4> facts;
    for (const VexpdifLayoutPattern& pattern : kVexpdifLayoutPatterns) {
        if (!matchesElementBitsPattern(pattern.sourceBits, sourceType.getElementType()) ||
            !matchesElementBitsPattern(pattern.resultBits, resultType.getElementType()) ||
            !matchesElementCountPattern(pattern.elementCounts, sourceType.getElementCount())) {
            continue;
        }

        VMIVexpdifLayoutFact fact = materializeVexpdifLayoutFact(ctx, pattern);
        if ((port == VMIVexpdifLayoutPort::Source ? fact.sourceLayout : fact.resultLayout) != layout) {
            continue;
        }
        if (!matchesVexpdifPhysicalShape(sourceType, maskType, resultType, fact)) {
            continue;
        }

        bool duplicate = llvm::any_of(facts, [&](const VMIVexpdifLayoutFact& existing) {
            return existing.sourceLayout == fact.sourceLayout && existing.resultLayout == fact.resultLayout;
        });
        if (!duplicate) {
            facts.push_back(fact);
        }
    }

    if (facts.empty()) {
        return fail(kVexpdifLayoutRequirement);
    }
    return facts;
}

FailureOr<VMIVexpdifLayoutFact> VMILayoutSupport::getPreferredVexpdifLayoutFact(
    VMIVexpdifOp op, std::string* reason) const
{
    auto fail = [&](const Twine& message) -> FailureOr<VMIVexpdifLayoutFact> {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    auto sourceType = dyn_cast<VMIVRegType>(op.getX().getType());
    auto maskType = dyn_cast<VMIMaskType>(op.getMask().getType());
    auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
    if (!sourceType || !maskType || !resultType) {
        return fail("vexpdif requires vreg x/max and mask operands");
    }

    MLIRContext* ctx = op.getContext();
    for (const VexpdifLayoutPattern& pattern : kVexpdifLayoutPatterns) {
        if (!pattern.preferred || !matchesElementBitsPattern(pattern.sourceBits, sourceType.getElementType()) ||
            !matchesElementBitsPattern(pattern.resultBits, resultType.getElementType()) ||
            !matchesElementCountPattern(pattern.elementCounts, sourceType.getElementCount())) {
            continue;
        }

        VMIVexpdifLayoutFact fact = materializeVexpdifLayoutFact(ctx, pattern);
        if (matchesVexpdifPhysicalShape(sourceType, maskType, resultType, fact)) {
            return fact;
        }
    }
    return fail(kVexpdifLayoutRequirement);
}

#endif // VMILAYOUTSUPPORTVEXPDIFQUERIES_INC
