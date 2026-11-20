// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Recognize contiguous output halves as one logical narrowing and store.
// Existing channel-merge and cast relations expose physical packing alternatives
// to the solver; no layout choice or instruction cost is prescribed here.
#include "VMIPackedOutputCombine.h"
#include "VMIIndexUtils.h"
#include "VMIInternal.h"
#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
struct PackedOutputMatch {
    VMITruncFOp lowCast;
    VMITruncFOp highCast;
    VMIStoreOp lowStore;
    VMIStoreOp highStore;
};

static bool isFullMask(Value value)
{
    if (value.getDefiningOp<VMIPsetOp>()) {
        return true;
    }
    auto mask = value.getDefiningOp<VMICreateMaskOp>();
    if (!mask) {
        return false;
    }
    auto active = getConstantIndexValue(mask.getActiveLanes());
    return active && *active >= cast<VMIMaskType>(value.getType()).getElementCount();
}

static VMITruncFOp getUniqueNarrowing(Value value)
{
    if (!value.hasOneUse()) {
        return {};
    }
    auto trunc = dyn_cast<VMITruncFOp>(*value.getUsers().begin());
    if (!trunc || !trunc.getResult().hasOneUse()) {
        return {};
    }
    return trunc;
}

static bool hasUnassignedTypes(VMIVintlvOp interleave, PackedOutputMatch match)
{
    SmallVector<Value> values{interleave.getLhs(),  interleave.getRhs(),       interleave.getLow(),
                              interleave.getHigh(), match.lowCast.getResult(), match.highCast.getResult()};
    return llvm::all_of(values, [](Value value) { return !cast<VMIVRegType>(value.getType()).getLayoutAttr(); });
}

// Use op range interfaces locally: never reassociate finite-width address
// arithmetic without proving that the separating addition cannot wrap.
static ConstantIntRanges inferLocalRange(Value value, DenseMap<Value, ConstantIntRanges>& ranges, unsigned depth = 0)
{
    auto found = ranges.find(value);
    if (found != ranges.end()) {
        return found->second;
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    auto full = ConstantIntRanges::maxRange(width);
    if (depth >= mlir::pto::kValue32) {
        return full;
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
        bool isInductionVariable = loop && loop.getInductionVar() == value;
        if (!isInductionVariable) {
            return full;
        }
        auto lower = getConstantIndexValue(loop.getLowerBound());
        auto upper = getConstantIndexValue(loop.getUpperBound());
        auto step = getConstantIndexValue(loop.getStep());
        if (!lower || !upper || !step || *step <= 0 || *upper <= *lower) {
            return full;
        }
        return ConstantIntRanges::fromSigned(APInt(width, *lower, true), APInt(width, *upper - 1, true));
    }
    auto interface = dyn_cast_or_null<InferIntRangeInterface>(value.getDefiningOp());
    bool hasRangeInterface = interface && value.getDefiningOp()->getNumRegions() == 0;
    if (!hasRangeInterface) {
        return full;
    }
    SmallVector<ConstantIntRanges> operands;
    for (Value operand : value.getDefiningOp()->getOperands()) {
        if (!operand.getType().isIntOrIndex()) {
            return full;
        }
        operands.push_back(inferLocalRange(operand, ranges, depth + 1));
    }
    interface.inferResultRanges(operands, [&](Value result, const ConstantIntRanges& range) {
        if (result == value) {
            full = range;
        }
    });
    ranges.try_emplace(value, full);
    return full;
}

static std::optional<int64_t> getAdjacentPointerDifference(Value low, Value high)
{
    auto first = low.getDefiningOp<AddPtrOp>();
    auto second = high.getDefiningOp<AddPtrOp>();
    bool sameBase = first && second && first.getPtr() == second.getPtr();
    if (!sameBase) {
        return std::nullopt;
    }
    auto lowCast = first.getOffset().getDefiningOp<arith::IndexCastOp>();
    auto highCast = second.getOffset().getDefiningOp<arith::IndexCastOp>();
    bool compatibleCasts =
        lowCast && highCast && lowCast.getIn().getType().isInteger(32) && highCast.getIn().getType().isInteger(32);
    if (!compatibleCasts) {
        return std::nullopt;
    }
    auto add = highCast.getIn().getDefiningOp<arith::AddIOp>();
    bool isOffsetIncrement = add && add.getLhs() == lowCast.getIn();
    if (!isOffsetIncrement) {
        return std::nullopt;
    }
    auto constant = add.getRhs().getDefiningOp<arith::ConstantOp>();
    auto delta = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    if (!delta || !delta.getValue().isSignedIntN(mlir::pto::kValue62)) {
        return std::nullopt;
    }
    DenseMap<Value, ConstantIntRanges> ranges;
    auto range = inferLocalRange(lowCast.getIn(), ranges);
    unsigned width = range.smin().getBitWidth();
    APInt minimum = range.smin().sext(width + 1) + delta.getValue().sext(width + 1);
    APInt maximum = range.smax().sext(width + 1) + delta.getValue().sext(width + 1);
    bool incrementDoesNotWrap = minimum.isSignedIntN(width) && maximum.isSignedIntN(width);
    if (!incrementDoesNotWrap) {
        return std::nullopt;
    }
    return delta.getInt() * mlir::pto::kValue2;
}

static bool areContiguousStores(PackedOutputMatch match, int64_t lanes)
{
    auto pointerType = dyn_cast<PtrType>(match.lowStore.getDestination().getType());
    bool compatiblePointers =
        pointerType && pointerType == match.highStore.getDestination().getType() && isUBBackedMemoryType(pointerType);
    if (!compatiblePointers) {
        return false;
    }
    auto delta = getKnownAddressDifferenceBytes(match.lowStore.getDestination(), match.highStore.getDestination());
    if (!delta) {
        delta = getAdjacentPointerDifference(match.lowStore.getDestination(), match.highStore.getDestination());
    }
    if (!delta) {
        return false;
    }
    // APInt keeps the proof well-defined even for extreme source offsets.
    llvm::APInt difference(mlir::pto::kValue128, static_cast<uint64_t>(*delta), true);
    bool identicalOffsets = match.lowStore.getOffset() == match.highStore.getOffset();
    if (!identicalOffsets) {
        auto low = getConstantIndexValue(match.lowStore.getOffset());
        auto high = getConstantIndexValue(match.highStore.getOffset());
        if (!low || !high) {
            return false;
        }
        difference += (llvm::APInt(mlir::pto::kValue128, static_cast<uint64_t>(*high), true) -
                       llvm::APInt(mlir::pto::kValue128, static_cast<uint64_t>(*low), true)) *
                      mlir::pto::kValue2;
    }
    return difference ==
           llvm::APInt(mlir::pto::kValue128, static_cast<uint64_t>(lanes * mlir::pto::kValue2));
}

static bool canDelayFirstStore(PackedOutputMatch match)
{
    Operation* first = match.lowStore;
    Operation* last = match.highStore;
    bool orderedStores = first->getBlock() == last->getBlock() && first->isBeforeInBlock(last);
    if (!orderedStores) {
        return false;
    }
    for (Operation* op = first->getNextNode(); op != last; op = op->getNextNode()) {
        if (!isPure(op)) {
            return false;
        }
    }
    return true;
}

static std::optional<PackedOutputMatch> matchPackedOutput(VMIVintlvOp interleave)
{
    auto low = getUniqueNarrowing(interleave.getLow());
    auto high = getUniqueNarrowing(interleave.getHigh());
    // The fused store narrows the merged 2N-lane value with a single mask, so a
    // predicated narrowing cannot be combined: the predicate would have to be
    // widened to 2N lanes and the two halves may disagree.
    bool matchingCasts =
        low && high && !low.getMask() && !high.getMask() && low->getAttrDictionary() == high->getAttrDictionary();
    if (!matchingCasts) {
        return std::nullopt;
    }
    auto lowStore = dyn_cast<VMIStoreOp>(*low.getResult().getUsers().begin());
    auto highStore = dyn_cast<VMIStoreOp>(*high.getResult().getUsers().begin());
    bool matchingStores = lowStore && highStore && lowStore->getAttrs() == highStore->getAttrs();
    if (!matchingStores) {
        return std::nullopt;
    }
    PackedOutputMatch match{low, high, lowStore, highStore};
    auto source = cast<VMIVRegType>(interleave.getLhs().getType());
    auto result = cast<VMIVRegType>(low.getResult().getType());
    bool supportedShape = source.getElementType().isF32() && result.getElementType().isBF16() &&
                          (source.getElementCount() == 64 || source.getElementCount() == 128) &&
                          result.getElementCount() == source.getElementCount() && result == high.getResult().getType();
    bool compatibleInterleave =
        supportedShape && isFullMask(interleave.getMask()) && hasUnassignedTypes(interleave, match);
    if (!compatibleInterleave) {
        return std::nullopt;
    }
    bool canCombine = canDelayFirstStore(match) && areContiguousStores(match, source.getElementCount());
    if (!canCombine) {
        return std::nullopt;
    }
    return match;
}

static void replacePackedOutput(VMIVintlvOp interleave, PackedOutputMatch match)
{
    OpBuilder builder(match.highStore);
    auto source = cast<VMIVRegType>(interleave.getLhs().getType());
    auto result = cast<VMIVRegType>(match.lowCast.getResult().getType());
    auto wideSource = VMIVRegType::get(builder.getContext(), source.getElementCount() * 2, source.getElementType(), {});
    auto wideResult = VMIVRegType::get(builder.getContext(), result.getElementCount() * 2, result.getElementType(), {});
    auto merged = builder.create<VMIChannelMergeOp>(
        interleave.getLoc(), wideSource, ValueRange{interleave.getLhs(), interleave.getRhs()});
    auto narrowed = builder.create<VMITruncFOp>(
        match.lowCast.getLoc(), wideResult, merged.getResult(), Value(), match.lowCast.getRoundingAttr(),
        match.lowCast.getSaturateAttr());
    narrowed->setAttrs(match.lowCast->getAttrDictionary());
    auto stored = builder.create<VMIStoreOp>(
        match.highStore.getLoc(), narrowed.getResult(), match.lowStore.getDestination(), match.lowStore.getOffset());
    stored->setAttrs(match.lowStore->getAttrDictionary());
    match.lowStore.erase();
    match.highStore.erase();
    match.lowCast.erase();
    match.highCast.erase();
    interleave.erase();
}
} // namespace

void mlir::pto::fusePackedOutputStores(ModuleOp module)
{
    SmallVector<VMIVintlvOp> interleaves;
    module.walk([&](VMIVintlvOp op) { interleaves.push_back(op); });
    for (VMIVintlvOp interleave : interleaves) {
        if (auto match = matchPackedOutput(interleave)) {
            replacePackedOutput(interleave, *match);
        }
    }
}
