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
// helpers.  It holds the pto.vmi.vexpdif layout surface: the row
// materializer, the physical-shape predicate, and the two layout queries
// (the per-layout candidate enumeration and the preference), all read
// through the kVexpdifLayoutPatterns table.
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

inline VMIVexpdifLayoutFact materializeVexpdifLayoutFact(MLIRContext* ctx, const VexpdifLayoutPattern& pattern)
{
    VMIVexpdifLayoutFact fact;
    fact.sourceLayout = materializeLayoutPattern(ctx, pattern.sourceLayout);
    fact.resultLayout = materializeLayoutPattern(ctx, pattern.resultLayout);
    fact.preferred = pattern.preferred;
    return fact;
}

/// Fills in the physical part counts of one vexpdif table row and reports
/// whether the pto.vexpdif lowering can realize it.  The lowering emits one
/// part per 256-bit source register, so a row is only legal while the mask
/// produces as many parts as the source, the mask keeps the data element
/// width (a physical predicate granularity of b16 for f16 / b32 for f32), and
/// the result produces the element-width ratio times as many parts.
inline bool matchesVexpdifPhysicalShape(
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType, VMIVexpdifLayoutFact& fact)
{
    unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
    if (sourceBits == 0 || resultBits == 0 || resultBits % sourceBits != 0) {
        return false;
    }

    MLIRContext* ctx = sourceType.getContext();
    auto assignedSource =
        VMIVRegType::get(ctx, sourceType.getElementCount(), sourceType.getElementType(), fact.sourceLayout);
    auto assignedMask = VMIMaskType::get(ctx, maskType.getElementCount(), maskType.getGranularity(), fact.sourceLayout);
    auto assignedResult =
        VMIVRegType::get(ctx, resultType.getElementCount(), resultType.getElementType(), fact.resultLayout);
    FailureOr<int64_t> sourceParts = getVMIPhysicalArity(assignedSource);
    FailureOr<int64_t> maskParts = getVMIPhysicalArity(assignedMask);
    FailureOr<int64_t> resultParts = getVMIPhysicalArity(assignedResult);
    FailureOr<StringRef> maskGranularity = getMaskPhysicalGranularity(assignedMask);
    bool knownParts =
        succeeded(sourceParts) && succeeded(maskParts) && succeeded(resultParts) && succeeded(maskGranularity);
    if (!knownParts || *sourceParts < 1) {
        return false;
    }
    bool maskMatchesDataWidth = getMaskGranularityBits(*maskGranularity) == static_cast<int64_t>(sourceBits);
    if (!maskMatchesDataWidth) {
        return false;
    }

    int64_t partsPerSourcePart = static_cast<int64_t>(resultBits / sourceBits);
    // Below one full source physical part, a widening row whose source and
    // result are both contiguous is the merged form: the VMIToVPTO lowering
    // computes the two halves on the packed source part and merges them with
    // one vintlv, so it yields one result part per source part instead of
    // the element-width ratio.
    bool mergedContiguousForm =
        partsPerSourcePart != 1 && fact.sourceLayout.isContiguous() && fact.resultLayout.isContiguous();
    if (mergedContiguousForm) {
        FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(sourceType.getElementType());
        bool belowOnePhysicalPart = succeeded(lanesPerPart) && sourceType.getElementCount() < *lanesPerPart;
        if (belowOnePhysicalPart) {
            partsPerSourcePart = 1;
        }
    }
    if (*maskParts != *sourceParts || *resultParts != partsPerSourcePart * *sourceParts) {
        return false;
    }

    fact.sourceParts = *sourceParts;
    fact.resultParts = *resultParts;
    fact.resultPartsPerSourcePart = partsPerSourcePart;
    return true;
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
