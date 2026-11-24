// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIPreAssignmentCombine.cpp - Pre-assignment VMI combines ---------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "VMIInternal.h"
#include "VMIIndexUtils.h"
#include "VMIPackedOutputCombine.h"
#include "PTO/Analysis/PTOScalarOps.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIFastmathState.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIPREASSIGNMENTCOMBINE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static void fuseMultiplyAdds(ModuleOp module)
{
    if (!getVMIFastmathState(module)) {
        return;
    }

    SmallVector<VMIVaddOp> adds;
    module.walk([&](VMIVaddOp add) { adds.push_back(add); });
    for (VMIVaddOp add : adds) {
        auto resultType = cast<VMIVRegType>(add.getResult().getType());
        if (!resultType.getElementType().isF32()) {
            continue;
        }

        for (unsigned productIndex = 0; productIndex < 2; ++productIndex) {
            auto multiply = add->getOperand(productIndex).getDefiningOp<VMIVmulOp>();
            if (!multiply || !multiply.getResult().hasOneUse() ||
                multiply.getResult().getType() != resultType ||
                multiply.getLhs().getType() != resultType || multiply.getRhs().getType() != resultType ||
                add->getOperand(1 - productIndex).getType() != resultType ||
                multiply.getMask() != add.getMask()) {
                continue;
            }
            // Both operations must zero inactive lanes. A matching predicate
            // keeps every lane read by the fused instruction defined.
            if (multiply.getPmode().value_or("zero") != "zero" || add.getPmode().value_or("zero") != "zero") {
                continue;
            }

            OpBuilder builder(add);
            auto fused = builder.create<VMIVmulaOp>(
                add.getLoc(), resultType, add->getOperand(1 - productIndex), multiply.getLhs(), multiply.getRhs(),
                add.getMask(), add.getPmodeAttr());
            add.getResult().replaceAllUsesWith(fused.getResult());
            add.erase();
            multiply.erase();
            break;
        }
    }
}

static LogicalResult canonicalizeContiguousGroupLoads(ModuleOp module)
{
    SmallVector<VMIGroupLoadOp> loads;
    module.walk([&](VMIGroupLoadOp load) {
        auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
        if (!resultType) {
            return;
        }

        int64_t numGroups = load.getNumGroupsAttr().getInt();
        int64_t laneCount = resultType.getElementCount();
        if (numGroups <= 0 || laneCount % numGroups != 0) {
            return;
        }

        std::optional<int64_t> rowStride = getConstantIndexValue(load.getRowStride());
        if (rowStride && *rowStride == laneCount / numGroups) {
            loads.push_back(load);
        }
    });

    OpBuilder builder(module.getContext());
    for (VMIGroupLoadOp load : loads) {
        builder.setInsertionPoint(load);
        auto replacement =
            builder.create<VMILoadOp>(load.getLoc(), load.getResult().getType(), load.getSource(), load.getOffset());
        load.getResult().replaceAllUsesWith(replacement.getResult());
        load.erase();
    }
    return success();
}

static LogicalResult fuseGroupSlotBroadcastLoads(ModuleOp module)
{
    SmallVector<VMIGroupBroadcastOp> broadcasts;
    module.walk([&](VMIGroupBroadcastOp broadcast) {
        auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
        if (!load || !load.getResult().hasOneUse()) {
            return;
        }
        if (load.getNumGroupsAttr().getInt() != broadcast.getNumGroupsAttr().getInt()) {
            return;
        }

        if (!isa<VMIVRegType>(broadcast.getResult().getType())) {
            return;
        }
        broadcasts.push_back(broadcast);
    });

    OpBuilder builder(module.getContext());
    for (VMIGroupBroadcastOp broadcast : broadcasts) {
        auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
        if (!load) {
            continue;
        }

        builder.setInsertionPoint(broadcast);
        auto fused = builder.create<VMIGroupBroadcastLoadOp>(
            broadcast.getLoc(), broadcast.getResult().getType(), load.getSource(), load.getOffset(),
            load.getSourceGroupStride(), broadcast.getNumGroupsAttr());
        broadcast.getResult().replaceAllUsesWith(fused.getResult());
        broadcast.erase();
        if (load->use_empty()) {
            load.erase();
        }
    }
    return success();
}

static bool canMoveLoadPastInterveningOps(
    Operation* load, Operation* broadcast, Value loadSource, AliasAnalysis& aliasAnalysis)
{
    bool isInSameBlock = load->getBlock() == broadcast->getBlock();
    if (!isInSameBlock) {
        return false;
    }

    for (Operation* op = load->getNextNode(); op && op != broadcast; op = op->getNextNode()) {
        if (isPure(op)) {
            continue;
        }

        auto effects = getEffectsRecursively(op);
        if (!effects) {
            return false;
        }
        for (const auto& effect : *effects) {
            bool usesUnsupportedResource = effect.getResource() != SideEffects::DefaultResource::get();
            bool affectsFullRegion = effect.getEffectOnFullRegion();
            if (usesUnsupportedResource || affectsFullRegion) {
                return false;
            }

            if (isa<MemoryEffects::Read>(effect.getEffect())) {
                continue;
            }
            if (!isa<MemoryEffects::Write>(effect.getEffect())) {
                return false;
            }

            Value writtenSource = effect.getValue();
            if (!writtenSource || !aliasAnalysis.alias(loadSource, writtenSource).isNo()) {
                return false;
            }
        }
    }
    return true;
}

static std::optional<int64_t> getLoadBroadcastGroups(Operation* broadcast)
{
    int64_t numGroups = 1;
    if (auto grouped = dyn_cast<VMIGroupBroadcastOp>(broadcast)) {
        numGroups = grouped.getNumGroupsAttr().getInt();
    } else if (!isa<VMIBroadcastOp>(broadcast)) {
        return std::nullopt;
    }
    auto load = broadcast->getOperand(0).getDefiningOp<VMILoadOp>();
    if (!load || !load.getResult().hasOneUse()) {
        return std::nullopt;
    }
    auto loadType = cast<VMIVRegType>(load.getResult().getType());
    int64_t laneCount = loadType.getElementCount();
    if (laneCount != numGroups) {
        return std::nullopt;
    }
    return numGroups;
}

static void fuseVMILoadBroadcast(Operation* broadcast, int64_t numGroups)
{
    auto load = broadcast->getOperand(0).getDefiningOp<VMILoadOp>();
    OpBuilder builder(broadcast);
    // Dense load lane g is the scalar for group g. Scalar broadcasts keep
    // their existing zero-stride representation.
    auto stride = builder.create<arith::ConstantOp>(
        broadcast->getLoc(), builder.getIndexType(), builder.getIndexAttr(numGroups == 1 ? 0 : 1));
    auto fused = builder.create<VMIGroupBroadcastLoadOp>(
        broadcast->getLoc(), broadcast->getResult(0).getType(), load.getSource(), load.getOffset(), stride,
        builder.getI64IntegerAttr(numGroups));
    VMILayoutSupport supports;
    auto resultType = cast<VMIVRegType>(fused.getResult().getType());
    // Require a dense fallback when the result has no assigned layout. A table
    // row for a strided carrier alone does not guarantee legal consumer edges.
    if (!resultType.getLayoutAttr()) {
        resultType = VMIVRegType::get(
            builder.getContext(), resultType.getElementCount(), resultType.getElementType(),
            VMILayoutAttr::getContiguous(builder.getContext()));
    }
    bool hasSupportedLayout = succeeded(supports.getGroupBroadcastLoadLayoutFact(resultType, stride, numGroups));
    if (!hasSupportedLayout) {
        fused.erase();
        stride.erase();
        return;
    }
    broadcast->getResult(0).replaceAllUsesWith(fused.getResult());
    broadcast->erase();
    load.erase();
}

static LogicalResult fuseVMILoadBroadcasts(ModuleOp module, AliasAnalysis& aliasAnalysis)
{
    SmallVector<Operation*> broadcasts;
    module.walk([&](Operation* broadcast) {
        if (!getLoadBroadcastGroups(broadcast)) {
            return;
        }
        auto load = broadcast->getOperand(0).getDefiningOp<VMILoadOp>();
        bool canReorderLoad = canMoveLoadPastInterveningOps(load, broadcast, load.getSource(), aliasAnalysis);
        if (canReorderLoad) {
            broadcasts.push_back(broadcast);
        }
    });

    for (Operation* broadcast : broadcasts) {
        if (auto numGroups = getLoadBroadcastGroups(broadcast)) {
            fuseVMILoadBroadcast(broadcast, *numGroups);
        }
    }
    return success();
}

static PTOLoadOp getFusablePTOScalarUBLoad(VMIBroadcastOp broadcast, AliasAnalysis& aliasAnalysis)
{
    auto load = broadcast.getValue().getDefiningOp<PTOLoadOp>();
    if (!load) {
        return {};
    }

    bool hasSingleUse = load.getValue().hasOneUse();
    bool canReorderLoad = canMoveLoadPastInterveningOps(load, broadcast, load.getPtr(), aliasAnalysis);
    if (!hasSingleUse || !canReorderLoad) {
        return {};
    }

    auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
    if (!resultType) {
        return {};
    }

    Type sourceType = load.getPtr().getType();
    Type elementType = resultType.getElementType();
    bool sourceIsUB = isUBBackedMemoryType(sourceType);
    bool sourceElementMatches = getMemoryElementType(sourceType) == load.getValue().getType();
    bool resultElementMatches = load.getValue().getType() == elementType;
    if (!sourceIsUB || !sourceElementMatches || !resultElementMatches) {
        return {};
    }

    unsigned bitWidth = getVMIElementBitWidth(elementType);
    if (bitWidth != kValue8 && bitWidth != kValue16 && bitWidth != kValue32) {
        return {};
    }
    return load;
}

static LogicalResult fusePTOScalarUBLoadBroadcasts(ModuleOp module, AliasAnalysis& aliasAnalysis)
{
    SmallVector<VMIBroadcastOp> broadcasts;
    module.walk([&](VMIBroadcastOp broadcast) {
        if (getFusablePTOScalarUBLoad(broadcast, aliasAnalysis)) {
            broadcasts.push_back(broadcast);
        }
    });

    OpBuilder builder(module.getContext());
    for (VMIBroadcastOp broadcast : broadcasts) {
        auto load = getFusablePTOScalarUBLoad(broadcast, aliasAnalysis);
        if (!load) {
            continue;
        }

        builder.setInsertionPoint(broadcast);
        Value zeroStride =
            builder.create<arith::ConstantOp>(broadcast.getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
        auto fused = builder.create<VMIGroupBroadcastLoadOp>(
            broadcast.getLoc(), broadcast.getResult().getType(), load.getPtr(), load.getOffset(), zeroStride,
            builder.getI64IntegerAttr(1));
        broadcast.getResult().replaceAllUsesWith(fused.getResult());
        broadcast.erase();
        if (load->use_empty()) {
            load.erase();
        }
    }
    return success();
}

struct DeinterleaveExtfMatch {
    VMIBitcastOp view;
    VMIExtFOp even;
    VMIExtFOp odd;
};

static bool hasUnassignedDeinterleaveExtfTypes(VMIVUnzipOp unzip, DeinterleaveExtfMatch match)
{
    SmallVector<Value> values{match.view.getSource(), match.view.getResult(), unzip.getLow(),
                              unzip.getHigh(),        match.even.getResult(), match.odd.getResult()};
    return llvm::all_of(values, [](Value value) { return !cast<VMIVRegType>(value.getType()).getLayoutAttr(); });
}

static std::optional<DeinterleaveExtfMatch> matchDeinterleaveExtf(VMIVUnzipOp unzip)
{
    auto view = unzip.getSource().getDefiningOp<VMIBitcastOp>();
    bool missingUniqueHalves = !view || !unzip.getLow().hasOneUse() || !unzip.getHigh().hasOneUse();
    if (missingUniqueHalves) {
        return std::nullopt;
    }
    auto even = dyn_cast<VMIExtFOp>(*unzip.getLow().getUsers().begin());
    auto odd = dyn_cast<VMIExtFOp>(*unzip.getHigh().getUsers().begin());
    bool missingLocalExtensions = !even || !odd || even->getBlock() != unzip->getBlock() ||
                                  odd->getBlock() != unzip->getBlock() || even.getMask() || odd.getMask();
    if (missingLocalExtensions) {
        return std::nullopt;
    }
    DeinterleaveExtfMatch match{view, even, odd};
    auto source = cast<VMIVRegType>(view.getSource().getType());
    auto half = cast<VMIVRegType>(unzip.getLow().getType());
    auto result = cast<VMIVRegType>(even.getResult().getType());
    bool supportedTypes = source.getElementType().isBF16() && half.getElementType().isBF16() &&
                          result.getElementType().isF32() && result == odd.getResult().getType() &&
                          source.getElementCount() / 2 == result.getElementCount();
    if (!supportedTypes || !hasUnassignedDeinterleaveExtfTypes(unzip, match)) {
        return std::nullopt;
    }
    return match;
}

// Create the fused conversion, keep it only when the layout tables have a row
// for its source, and move both extension users onto it. Returns nullopt (with
// nothing changed) when no legal layout relation exists.
static std::optional<VMIDeinterleaveExtfOp> tryEmitDeinterleaveExtf(
    Operation* anchor, Value source, Type evenType, Type oddType)
{
    OpBuilder builder(anchor);
    auto fused = builder.create<VMIDeinterleaveExtfOp>(anchor->getLoc(), TypeRange{evenType, oddType}, source);
    if (VMILayoutSupport().getDeinterleaveExtfLayoutFacts(fused).empty()) {
        fused.erase();
        return std::nullopt;
    }
    return fused;
}

// Both extensions consume the matching halves in the same block. Insert at
// the unzip (which dominates both users); never move its input memory access.
static void fuseDeinterleaveExtf(ModuleOp module)
{
    SmallVector<VMIVUnzipOp> unzips;
    module.walk([&](VMIVUnzipOp op) { unzips.push_back(op); });
    for (VMIVUnzipOp unzip : unzips) {
        auto match = matchDeinterleaveExtf(unzip);
        if (!match) {
            continue;
        }
        auto fused = tryEmitDeinterleaveExtf(
            unzip, match->view.getSource(), match->even.getResult().getType(), match->odd.getResult().getType());
        if (!fused) {
            continue;
        }
        match->even.getResult().replaceAllUsesWith(fused->getEven());
        match->odd.getResult().replaceAllUsesWith(fused->getOdd());
        match->even.erase();
        match->odd.erase();
        unzip.erase();
        if (match->view->use_empty()) {
            match->view.erase();
        }
    }
}

// The staged form `vdintlv(extf(load a), extf(load b))` is the same logical
// operation as `deinterleave_extf` over the single 2N-lane span the two loads
// cover: the extension is pointwise and the deinterleave is a lane bijection, so
// the two commute. Rewriting the pair into one wide contiguous load plus the
// fused conversion removes the permutation without introducing a distributed
// memory access. The two loads must be adjacent, memory-effect free in between
// and single-use, and the governing mask must be provably all-active because the
// fused op carries no mask.
static bool isAllActiveMask(Value mask)
{
    Operation* def = mask.getDefiningOp();
    if (!def) {
        return false;
    }
    if (isa<VMIPsetOp>(def)) {
        return true;
    }
    auto createMask = dyn_cast<VMICreateMaskOp>(def);
    if (!createMask) {
        return false;
    }
    auto maskType = dyn_cast<VMIMaskType>(createMask.getResult().getType());
    auto activeLanes = getConstantIndexValue(createMask.getActiveLanes());
    return maskType && activeLanes && *activeLanes >= maskType.getElementCount();
}

// Frontends may route `i32` arithmetic through an `index_cast` (that is the shape
// this strip exists for; a direct integer offset needs no peeling), so peel it
// before comparing anything. `PTOScalarOps.h` owns the PTO/arith spelling split.
static Value stripOffsetCast(Value value)
{
    Operation* definition = value.getDefiningOp();
    if (isScalarIndexCastOp(definition)) {
        return definition->getOperand(0);
    }
    return value;
}

// The signedness of an offset cast decides how the inner value is extended, so
// only equally casted offsets describe comparable index values.
struct OffsetCast {
    bool present = false;
    bool isSigned = false;
};

static OffsetCast getOffsetCast(Value value)
{
    Operation* definition = value.getDefiningOp();
    if (isa_and_nonnull<arith::IndexCastUIOp>(definition)) {
        return OffsetCast{true, false};
    }
    if (isa_and_nonnull<arith::IndexCastOp>(definition)) {
        return OffsetCast{true, true};
    }
    auto cast = dyn_cast_or_null<IndexCastOp>(definition);
    if (!cast) {
        return OffsetCast{};
    }
    return OffsetCast{true, cast.getSignedness().getValue() == Signedness::Signed};
}

struct OffsetExpr {
    Value base;
    int64_t constant = 0;
};

// Decompose an offset into `base + constant`, following one level of integer add
// in either operand order and the PTO spelling of the arithmetic ops.
static std::optional<OffsetExpr> decomposeOffset(Value value)
{
    value = stripOffsetCast(value);
    if (auto constant = getConstantIndexValue(value)) {
        return OffsetExpr{Value(), *constant};
    }
    auto decomposeAdd = [](Value lhs, Value rhs) -> std::optional<OffsetExpr> {
        auto rhsConstant = getConstantIndexValue(rhs);
        if (!rhsConstant) {
            return std::nullopt;
        }
        Value base = stripOffsetCast(lhs);
        if (auto lhsConstant = getConstantIndexValue(base)) {
            return OffsetExpr{Value(), *lhsConstant + *rhsConstant};
        }
        return OffsetExpr{base, *rhsConstant};
    };
    Operation* definition = value.getDefiningOp();
    if (isScalarAddOp(definition)) {
        if (auto expr = decomposeAdd(getScalarLhs(definition), getScalarRhs(definition))) {
            return expr;
        }
        return decomposeAdd(getScalarRhs(definition), getScalarLhs(definition));
    }
    // Any other value is an opaque base; two offsets are adjacent only when they
    // share it, which is exactly what comparing these bases establishes.
    return OffsetExpr{value, 0};
}

static bool areOffsetsAdjacent(Value firstOffset, Value secondOffset, int64_t laneCount)
{
    if (laneCount <= 0) {
        return false;
    }
    OffsetCast firstCast = getOffsetCast(firstOffset);
    OffsetCast secondCast = getOffsetCast(secondOffset);
    if (firstCast.present != secondCast.present) {
        return false;
    }
    if (firstCast.present && firstCast.isSigned != secondCast.isSigned) {
        return false;
    }
    auto first = decomposeOffset(firstOffset);
    auto second = decomposeOffset(secondOffset);
    if (!first || !second || first->base != second->base) {
        return false;
    }
    return second->constant >= first->constant && second->constant - first->constant == laneCount;
}

static bool areMemoryEffectsFreeBetween(Operation* first, Operation* second)
{
    bool sameBlock = first->getBlock() == second->getBlock();
    bool ordered = sameBlock && first->isBeforeInBlock(second);
    if (!ordered) {
        return false;
    }
    for (Operation* op = first->getNextNode(); op && op != second; op = op->getNextNode()) {
        if (!isMemoryEffectFree(op)) {
            return false;
        }
    }
    return true;
}

struct DeinterleaveExtfFromIntlvMatch {
    VMILoadOp first;
    VMILoadOp second;
    VMIExtFOp even;
    VMIExtFOp odd;
};

static std::optional<DeinterleaveExtfFromIntlvMatch> matchDeinterleaveExtfFromIntlv(VMIVdintlvOp dintlv)
{
    if (!isAllActiveMask(dintlv.getMask())) {
        return std::nullopt;
    }
    auto even = dintlv.getLhs().getDefiningOp<VMIExtFOp>();
    auto odd = dintlv.getRhs().getDefiningOp<VMIExtFOp>();
    bool missingUniqueExtensions = !even || !odd || even == odd || even.getMask() || odd.getMask() ||
                                   !even.getResult().hasOneUse() || !odd.getResult().hasOneUse();
    if (missingUniqueExtensions) {
        return std::nullopt;
    }
    auto first = even.getSource().getDefiningOp<VMILoadOp>();
    auto second = odd.getSource().getDefiningOp<VMILoadOp>();
    bool unsupportedLoads = !first || !second || first == second || first.getSource() != second.getSource() ||
                            !first.getResult().hasOneUse() || !second.getResult().hasOneUse();
    if (unsupportedLoads) {
        return std::nullopt;
    }
    auto loadType = cast<VMIVRegType>(first.getResult().getType());
    auto resultType = cast<VMIVRegType>(even.getResult().getType());
    int64_t laneCount = loadType.getElementCount();
    bool supportedTypes = loadType.getElementType().isBF16() && resultType.getElementType().isF32() &&
                          resultType == odd.getResult().getType() && resultType.getElementCount() == laneCount;
    if (!supportedTypes) {
        return std::nullopt;
    }
    DeinterleaveExtfFromIntlvMatch match{first, second, even, odd};
    SmallVector<Value> values{first.getResult(), second.getResult(), even.getResult(),
                              odd.getResult(),   dintlv.getLow(),    dintlv.getHigh()};
    bool unassignedLayouts =
        llvm::all_of(values, [](Value value) { return !cast<VMIVRegType>(value.getType()).getLayoutAttr(); });
    bool adjacent = areOffsetsAdjacent(first.getOffset(), second.getOffset(), laneCount);
    bool independent = areMemoryEffectsFreeBetween(first, second);
    if (!unassignedLayouts || !adjacent || !independent) {
        return std::nullopt;
    }
    return match;
}

static void fuseDeinterleaveExtfFromIntlv(ModuleOp module)
{
    SmallVector<VMIVdintlvOp> deinterleaves;
    module.walk([&](VMIVdintlvOp op) { deinterleaves.push_back(op); });
    for (VMIVdintlvOp dintlv : deinterleaves) {
        auto match = matchDeinterleaveExtfFromIntlv(dintlv);
        if (!match) {
            continue;
        }
        auto sourceType = cast<VMIVRegType>(match->first.getResult().getType());
        auto wideType =
            VMIVRegType::get(module.getContext(), sourceType.getElementCount() * 2, sourceType.getElementType(), {});
        // Read the 2N-lane span where the first load reads it; the fused
        // conversion may sit later, but the memory access must not move across
        // anything that writes the span.
        OpBuilder loadBuilder(match->first);
        auto wide = loadBuilder.create<VMILoadOp>(
            match->first.getLoc(), wideType, match->first.getSource(), match->first.getOffset());
        auto fused =
            tryEmitDeinterleaveExtf(dintlv, wide, match->even.getResult().getType(), match->odd.getResult().getType());
        if (!fused) {
            wide.erase();
            continue;
        }
        dintlv.getLow().replaceAllUsesWith(fused->getEven());
        dintlv.getHigh().replaceAllUsesWith(fused->getOdd());
        dintlv.erase();
        match->even.erase();
        match->odd.erase();
        if (match->first->use_empty()) {
            match->first.erase();
        }
        if (match->second->use_empty()) {
            match->second.erase();
        }
    }
}

// The united form `two adjacent loads + deinterleave` still has a legacy home:
// `deinterleave_load` becomes `vldsx2 "DINTLV_*"`. The two input combines match
// disjoint shapes (`vdintlv` over extensions vs over loads), so this runs after
// the packed widening conversion only to keep the responsibilities separated:
// the downgrade pass lowers, this pass combines.
static bool combineDeinterleaveLoads(VMIVdintlvOp deinterleave)
{
    if (!isAllActiveMask(deinterleave.getMask())) {
        return false;
    }
    auto firstLoad = deinterleave.getLhs().getDefiningOp<VMILoadOp>();
    auto secondLoad = deinterleave.getRhs().getDefiningOp<VMILoadOp>();
    bool unsupportedLoads = !firstLoad || !secondLoad || firstLoad == secondLoad ||
                            firstLoad.getSource() != secondLoad.getSource() || !firstLoad.getResult().hasOneUse() ||
                            !secondLoad.getResult().hasOneUse() ||
                            *firstLoad.getResult().user_begin() != deinterleave.getOperation() ||
                            *secondLoad.getResult().user_begin() != deinterleave.getOperation();
    if (unsupportedLoads) {
        return false;
    }

    auto firstType = cast<VMIVRegType>(firstLoad.getResult().getType());
    int64_t laneCount = firstType.getElementCount();
    // `deinterleave_load` only lowers through the dual load for full physical
    // chunks; a partial chunk has to keep the plain deinterleave.
    auto lanesPerPart = getDataLanesPerPart(firstType.getElementType());
    bool fullChunks = succeeded(lanesPerPart) && *lanesPerPart > 0 && laneCount % *lanesPerPart == 0;
    bool adjacent = laneCount > 0 && areOffsetsAdjacent(firstLoad.getOffset(), secondLoad.getOffset(), laneCount);
    if (!fullChunks || !adjacent || !areMemoryEffectsFreeBetween(firstLoad, secondLoad)) {
        return false;
    }

    OpBuilder builder(firstLoad);
    auto fusedLoad = builder.create<VMIDeinterleaveLoadOp>(
        deinterleave.getLoc(), deinterleave.getLow().getType(), deinterleave.getHigh().getType(), firstLoad.getSource(),
        firstLoad.getOffset());
    deinterleave.getLow().replaceAllUsesWith(fusedLoad.getLow());
    deinterleave.getHigh().replaceAllUsesWith(fusedLoad.getHigh());
    deinterleave.erase();
    firstLoad.erase();
    secondLoad.erase();
    return true;
}

// The united form `interleave + two adjacent stores` becomes `interleave_store`,
// which knows the two destinations and emits `vstsx2 "INTLV_*"`.
static bool combineInterleaveStores(VMIVintlvOp interleave)
{
    bool hasAllActiveMask = isAllActiveMask(interleave.getMask());
    bool lowHasOneUse = interleave.getLow().hasOneUse();
    bool highHasOneUse = interleave.getHigh().hasOneUse();
    if (!hasAllActiveMask || !lowHasOneUse || !highHasOneUse) {
        return false;
    }
    auto firstStore = dyn_cast<VMIStoreOp>(*interleave.getLow().user_begin());
    auto secondStore = dyn_cast<VMIStoreOp>(*interleave.getHigh().user_begin());
    bool unsupportedStores =
        !firstStore || !secondStore || firstStore == secondStore || firstStore.getValue() != interleave.getLow() ||
        secondStore.getValue() != interleave.getHigh() || firstStore.getDestination() != secondStore.getDestination();
    if (unsupportedStores) {
        return false;
    }

    auto valueType = cast<VMIVRegType>(interleave.getLhs().getType());
    int64_t laneCount = valueType.getElementCount();
    bool adjacent = laneCount > 0 && areOffsetsAdjacent(firstStore.getOffset(), secondStore.getOffset(), laneCount);
    if (!adjacent || !areMemoryEffectsFreeBetween(firstStore, secondStore)) {
        return false;
    }

    OpBuilder builder(firstStore);
    builder.create<VMIInterleaveStoreOp>(
        interleave.getLoc(), interleave.getLhs(), interleave.getRhs(), firstStore.getDestination(),
        firstStore.getOffset());
    firstStore.erase();
    secondStore.erase();
    interleave.erase();
    return true;
}

static void combineInterleavedMemoryOps(ModuleOp module)
{
    SmallVector<VMIVdintlvOp, mlir::pto::kValue8> deinterleaves;
    SmallVector<VMIVintlvOp, mlir::pto::kValue8> interleaves;
    module.walk([&](Operation* op) {
        if (auto deinterleave = dyn_cast<VMIVdintlvOp>(op)) {
            deinterleaves.push_back(deinterleave);
        } else if (auto interleave = dyn_cast<VMIVintlvOp>(op)) {
            interleaves.push_back(interleave);
        }
    });

    for (VMIVdintlvOp op : deinterleaves) {
        if (op->getBlock()) {
            (void)combineDeinterleaveLoads(op);
        }
    }
    for (VMIVintlvOp op : interleaves) {
        if (op->getBlock()) {
            (void)combineInterleaveStores(op);
        }
    }
}

struct VMIPreAssignmentCombinePass : pto::impl::VMIPreAssignmentCombineBase<VMIPreAssignmentCombinePass> {
    void runOnOperation() override
    {
        fuseMultiplyAdds(getOperation());
        fuseDeinterleaveExtf(getOperation());
        fuseDeinterleaveExtfFromIntlv(getOperation());
        combineInterleavedMemoryOps(getOperation());
        fusePackedOutputStores(getOperation());
        AliasAnalysis& aliasAnalysis = getAnalysis<AliasAnalysis>();
        if (failed(canonicalizeContiguousGroupLoads(getOperation())) ||
            failed(fuseGroupSlotBroadcastLoads(getOperation())) ||
            failed(fuseVMILoadBroadcasts(getOperation(), aliasAnalysis)) ||
            failed(fusePTOScalarUBLoadBroadcasts(getOperation(), aliasAnalysis))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIPreAssignmentCombinePass()
{
    return std::make_unique<VMIPreAssignmentCombinePass>();
}
