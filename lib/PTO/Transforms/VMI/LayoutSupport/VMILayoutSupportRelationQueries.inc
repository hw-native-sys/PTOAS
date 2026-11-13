// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTRELATIONQUERIES_INC
#define VMILAYOUTSUPPORTRELATIONQUERIES_INC

//===- VMILayoutSupportRelationQueries.inc - relation support queries -*- C++ -*-===//
// Textual include unit of VMILayoutSupport.cpp, included at the end of the
// translation unit so the definitions below can use the file-local table
// helpers.  It holds the layout-support queries the layout cost model reads
// directly and that the planner reads through it: the staging layout of a
// generated group mask, the low/high contract of the two-source interleave
// store, and the op-qualified same-layout relation.
//===----------------------------------------------------------------------===//

FailureOr<VMIGeneratedMaskLayoutFact> VMILayoutSupport::getGeneratedMaskLayoutFact(
    Operation* op, VMILayoutAttr resultLayout, std::string* reason) const
{
    using FactT = FailureOr<VMIGeneratedMaskLayoutFact>;
    if (!op || op->getNumResults() != 1) {
        return failWithReason<FactT>(reason, "generated mask requires a single-result operation");
    }
    if (!resultLayout) {
        return failWithReason<FactT>(reason, "generated mask requires an assigned result layout");
    }
    auto resultType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
    if (!resultType || !VMIMaskType::isConcreteGranularity(resultType.getGranularity())) {
        return failWithReason<FactT>(reason, "generated mask requires concrete mask granularity");
    }
    auto assignedType = VMIMaskType::get(
        resultType.getContext(), resultType.getElementCount(), resultType.getGranularity(), resultLayout);
    FailureOr<int64_t> arity = getVMIPhysicalArity(assignedType);
    if (failed(arity) || *arity <= 0) {
        return failWithReason<FactT>(reason, "generated mask result layout has no physical realization");
    }

    // A constant active-elements-per-group operand is materialized at compile
    // time and is staged differently from the runtime form.
    GeneratedMaskKind kind;
    if (isa<VMICreateGroupMaskOp>(op)) {
        bool dynamic =
            cast<VMICreateGroupMaskOp>(op).getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>() == nullptr;
        kind = dynamic ? GeneratedMaskKind::GroupDynamic : GeneratedMaskKind::GroupConstant;
    } else {
        return failWithReason<FactT>(reason, "operation is not a supported generated group mask");
    }

    for (const GeneratedMaskStagingPattern& pattern : kGeneratedMaskStagingPatterns) {
        if (pattern.kind != kind || !matchesLayoutPattern(op->getContext(), pattern.resultLayout, resultLayout)) {
            continue;
        }
        return VMIGeneratedMaskLayoutFact{
            materializeLayoutPattern(op->getContext(), pattern.generationLayout), resultLayout};
    }

    // No staging row: the mask is generated directly in its assigned layout.
    return VMIGeneratedMaskLayoutFact{resultLayout, resultLayout};
}

FailureOr<VMIInterleaveStoreSupport> VMILayoutSupport::getInterleaveStoreSupport(
    VMIVRegType lowType, VMIVRegType highType, std::string* reason) const
{
    using FactT = FailureOr<VMIInterleaveStoreSupport>;
    VMILayoutAttr lowLayout = lowType.getLayoutAttr();
    VMILayoutAttr highLayout = highType.getLayoutAttr();
    if (!lowLayout || !highLayout || !lowLayout.isContiguous() || !highLayout.isContiguous()) {
        return failWithReason<FactT>(reason, "requires assigned contiguous low/high input layouts");
    }
    if (lowType.getElementCount() != highType.getElementCount() ||
        lowType.getElementType() != highType.getElementType()) {
        return failWithReason<FactT>(reason, "requires matching low/high input shape and element type");
    }
    unsigned elementBits = pto::getPTOStorageElemBitWidth(lowType.getElementType());
    if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
        return failWithReason<FactT>(reason, "requires 8/16/32-bit element type for vstsx2 INTLV");
    }
    FailureOr<int64_t> lanes = getDataLanesPerPart(lowType.getElementType());
    if (failed(lanes) || *lanes <= 0 || lowType.getElementCount() % *lanes != 0) {
        return failWithReason<FactT>(reason, "requires full physical chunks");
    }
    return VMIInterleaveStoreSupport{lowLayout, highLayout};
}

static bool hasMatchingDataMaskArity(Operation* op, VMIVRegType sourceType, VMIMaskType maskType, VMILayoutAttr layout)
{
    auto dataType =
        VMIVRegType::get(op->getContext(), sourceType.getElementCount(), sourceType.getElementType(), layout);
    auto concreteMaskType =
        VMIMaskType::get(op->getContext(), maskType.getElementCount(), maskType.getGranularity(), layout);
    auto dataArity = getVMIPhysicalArity(dataType);
    auto maskArity = getVMIPhysicalArity(concreteMaskType);
    return maskType.getGranularity() == "b32" && succeeded(dataArity) && succeeded(maskArity) && *dataArity >= 1 &&
           *dataArity == *maskArity;
}
static LogicalResult checkAddcRelation(VMIVaddcOp addc, VMILayoutAttr layout, std::string* reason)
{
    Operation* op = addc.getOperation();

    auto sourceType = cast<VMIVRegType>(addc.getLhs().getType());
    auto maskType = cast<VMIMaskType>(addc.getMask().getType());
    if (!hasMatchingDataMaskArity(op, sourceType, maskType, layout)) {
        return failWithReason<LogicalResult>(
            reason, "vaddc relation requires matching 32-bit data and b32 mask "
                    "physical arity");
    }
    return success();
}
static LogicalResult checkAddcsRelation(VMIVaddcsOp addcs, VMILayoutAttr layout, std::string* reason)
{
    Operation* op = addcs.getOperation();

    auto sourceType = cast<VMIVRegType>(addcs.getLhs().getType());
    auto maskType = cast<VMIMaskType>(addcs.getMask().getType());
    if (!hasMatchingDataMaskArity(op, sourceType, maskType, layout)) {
        return failWithReason<LogicalResult>(
            reason, "vaddcs relation requires matching 32-bit data and b32 mask "
                    "physical arity");
    }
    return success();
}
static LogicalResult checkVmullRelation(VMIVmullOp vmull, VMILayoutAttr layout, std::string* reason)
{
    Operation* op = vmull.getOperation();

    auto sourceType = cast<VMIVRegType>(vmull.getA().getType());
    auto maskType = cast<VMIMaskType>(vmull.getMask().getType());
    if ((!layout.isContiguous() &&
         !(layout.isDeinterleaved() && (layout.getFactor() == 2 || layout.getFactor() == 4))) ||
        layout.getLaneStride() != 1) {
        return failWithReason<LogicalResult>(
            reason, "vmull relation requires contiguous or deinterleaved "
                    "factor-2/factor-4 layout with lane_stride=1");
    }
    if (!hasMatchingDataMaskArity(op, sourceType, maskType, layout)) {
        return failWithReason<LogicalResult>(
            reason, "vmull relation requires matching data and b32 mask "
                    "physical arity");
    }
    return success();
}
LogicalResult VMILayoutSupport::getSameLayoutRelationSupport(
    Operation* op, VMILayoutAttr layout, std::string* reason) const
{
    if (!op) {
        return failWithReason<LogicalResult>(reason, "same-layout relation requires an operation");
    }
    if (!layout) {
        return failWithReason<LogicalResult>(reason, "same-layout relation requires an assigned layout");
    }

    // These op families lower to predicated instructions whose merge form has no
    // passthru operand, so a merge predicate mode has to take an explicit
    // materialization instead of the shared-layout relation.
    bool hasLoweringPmodeRestriction =
        isa<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp, VMIShrSOp, VMIVcmpOp, VMIVcmpsOp, VMIVexpdifOp,
            VMIVaxpyOp, VMIVlreluOp, VMIVpreluOp, VMIVmulaOp>(op);
    if (hasLoweringPmodeRestriction) {
        if (auto pmode = op->getAttrOfType<StringAttr>("pmode"); pmode && pmode.getValue() == "merge") {
            return failWithReason<LogicalResult>(reason, "merge predicate mode requires explicit passthru lowering");
        }
    }

    // The carry operations pair a data part with a b32 predicate part, so the
    // relation only holds while both sides have the same physical arity.
    if (auto addc = dyn_cast<VMIVaddcOp>(op)) {
        return checkAddcRelation(addc, layout, reason);
    }
    if (auto addcs = dyn_cast<VMIVaddcsOp>(op)) {
        return checkAddcsRelation(addcs, layout, reason);
    }
    if (auto vmull = dyn_cast<VMIVmullOp>(op)) {
        return checkVmullRelation(vmull, layout, reason);
    }
    return success();
}

#endif // VMILAYOUTSUPPORTRELATIONQUERIES_INC
