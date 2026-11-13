// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTOMemoryIndexRange.h"
#include "PTO/IR/PTO.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto::integer_range;

namespace {

struct IndexLaneRange {
    int64_t lo;
    int64_t hi;
};

static IntegerType indexElementType(Value value)
{
    auto vector = dyn_cast<pto::VRegType>(value.getType());
    return vector ? dyn_cast<IntegerType>(vector.getElementType()) : IntegerType();
}

static bool indexMaskMatches(Value mask, IntegerType elem)
{
    auto type = dyn_cast<pto::MaskType>(mask.getType());
    bool supported = elem && (elem.getWidth() == 16 || elem.getWidth() == 32);
    return type && supported && type.getGranularity() == (elem.getWidth() == 16 ? "b16" : "b32");
}

static Value stripIndexBitcast(Value value)
{
    if (auto cast = value.getDefiningOp<pto::VbitcastOp>()) {
        auto input = dyn_cast<pto::VRegType>(cast.getInput().getType());
        auto output = dyn_cast<pto::VRegType>(value.getType());
        bool sameGeometry = input && output && indexElementType(cast.getInput()) && indexElementType(value) &&
                            input.getElementCount() == output.getElementCount() &&
                            indexElementType(cast.getInput()).getWidth() == indexElementType(value).getWidth();
        if (sameGeometry) {
            return cast.getInput();
        }
    }
    return value;
}

// Only conjunctions with fully defined predicate producers imply a comparison.
// Do not infer facts through OR, NOT, lane permutations or partial comparisons.
static void constrainIndexLane(Value value, Value mask, unsigned lane, IndexLaneRange& range, unsigned& budget)
{
    if (!budget || !indexMaskMatches(mask, indexElementType(value))) {
        return;
    }
    --budget;
    if (auto conjunction = mask.getDefiningOp<pto::PandOp>()) {
        auto type = cast<pto::VRegType>(value.getType());
        bool defined =
            allIntegerLanes(conjunction.getMask(), type.getElementCount(), indexElementType(value).getWidth());
        if (defined) {
            constrainIndexLane(value, conjunction.getSrc0(), lane, range, budget);
            constrainIndexLane(value, conjunction.getSrc1(), lane, range, budget);
        }
        return;
    }
    auto comparison = mask.getDefiningOp<pto::VcmpOp>();
    auto type = cast<pto::VRegType>(value.getType());
    bool usable = comparison && comparison.getSrc0() == value &&
                  allIntegerLanes(comparison.getMask(), type.getElementCount(), indexElementType(value).getWidth());
    if (!usable) {
        return;
    }
    unsigned constantBudget = 32;
    auto constants = constantIndexLanes(comparison.getSrc1(), constantBudget);
    if (!constants || lane >= constants->size()) {
        return;
    }
    auto elem = indexElementType(value);
    APInt bits(elem.getWidth(), (*constants)[lane]);
    int64_t bound = elem.isUnsigned() ? bits.getZExtValue() : bits.getSExtValue();
    StringRef mode = comparison.getCmpMode();
    if (mode == "ge" || mode == "gt" || mode == "eq") {
        range.lo = std::max(range.lo, bound + (mode == "gt"));
    }
    if (mode == "le" || mode == "lt" || mode == "eq") {
        range.hi = std::min(range.hi, bound - (mode == "lt"));
    }
}

static std::optional<IndexLaneRange> indexLaneRange(Value value, Value mask, unsigned lane, unsigned& budget);

static std::optional<IndexLaneRange> indexArithmeticRange(
    Operation* op, Value mask, unsigned lane, unsigned& budget, IntegerType elem)
{
    bool supported = isa<pto::VaddOp, pto::VmulOp>(op);
    bool defined = supported && allIntegerLanes(
                                    op->getOperand(2),
                                    cast<pto::VRegType>(op->getResult(0).getType()).getElementCount(), elem.getWidth());
    if (!defined) {
        return std::nullopt;
    }
    auto lhs = indexLaneRange(op->getOperand(0), mask, lane, budget);
    auto rhs = indexLaneRange(op->getOperand(1), mask, lane, budget);
    if (!lhs || !rhs || lhs->lo < 0 || rhs->lo < 0) {
        return std::nullopt;
    }
    bool multiply = isa<pto::VmulOp>(op);
    __int128 lo = multiply ? static_cast<__int128>(lhs->lo) * rhs->lo : static_cast<__int128>(lhs->lo) + rhs->lo;
    __int128 hi = multiply ? static_cast<__int128>(lhs->hi) * rhs->hi : static_cast<__int128>(lhs->hi) + rhs->hi;
    uint64_t limit = elem.isUnsigned() ? APInt::getMaxValue(elem.getWidth()).getZExtValue() :
                                         APInt::getSignedMaxValue(elem.getWidth()).getZExtValue();
    if (hi > limit) {
        return std::nullopt;
    }
    return IndexLaneRange{static_cast<int64_t>(lo), static_cast<int64_t>(hi)};
}

static std::optional<IndexLaneRange> indexLaneRange(Value value, Value mask, unsigned lane, unsigned& budget)
{
    auto elem = indexElementType(value);
    bool supported = budget && elem && (elem.getWidth() == 16 || elem.getWidth() == 32) && indexMaskMatches(mask, elem);
    if (!supported) {
        return std::nullopt;
    }
    --budget;
    unsigned constantBudget = 64;
    if (auto constants = constantIndexLanes(value, constantBudget, true)) {
        if (lane >= constants->size()) {
            return std::nullopt;
        }
        APInt bits(elem.getWidth(), (*constants)[lane]);
        int64_t number = elem.isUnsigned() ? bits.getZExtValue() : bits.getSExtValue();
        return IndexLaneRange{number, number};
    }
    IndexLaneRange range{
        elem.isUnsigned() ? 0 : APInt::getSignedMinValue(elem.getWidth()).getSExtValue(),
        elem.isUnsigned() ? int64_t(APInt::getMaxValue(elem.getWidth()).getZExtValue()) :
                            APInt::getSignedMaxValue(elem.getWidth()).getSExtValue()};
    Value input = stripIndexBitcast(value);
    if (input != value) {
        auto source = indexLaneRange(input, mask, lane, budget);
        if (source && source->lo >= 0 && source->hi <= range.hi) {
            range = *source;
        }
    } else if (auto op = value.getDefiningOp()) {
        if (auto arithmetic = indexArithmeticRange(op, mask, lane, budget, elem)) {
            range = *arithmetic;
        }
    }
    unsigned predicateBudget = 64;
    constrainIndexLane(value, mask, lane, range, predicateBudget);
    return range.lo <= range.hi ? std::optional<IndexLaneRange>(range) : std::nullopt;
}

} // namespace

std::optional<std::pair<int64_t, uint64_t>> mlir::pto::integer_range::boundedIndexByteRange(
    Value offsets, Value mask, uint64_t bytes)
{
    auto elem = indexElementType(offsets);
    auto type = dyn_cast<pto::VRegType>(offsets.getType());
    bool supported = elem && elem.isUnsigned() && (elem.getWidth() == 16 || elem.getWidth() == 32) &&
                     (bytes == 2 || bytes == 4) && type.getElementCount() && type.getElementCount() <= 256;
    if (!supported) {
        return std::nullopt;
    }
    int64_t lo = INT64_MAX;
    int64_t hi = 0;
    for (unsigned lane = 0; lane < type.getElementCount(); ++lane) {
        unsigned budget = 32;
        auto range = indexLaneRange(offsets, mask, lane, budget);
        if (!range || range->lo < 0) {
            return std::nullopt;
        }
        lo = std::min(lo, range->lo);
        hi = std::max(hi, range->hi);
    }
    // This is a closed conservative envelope, not a claim of contiguous access.
    bool bounded = hi < int64_t(APInt::getMaxValue(elem.getWidth()).getZExtValue()) &&
                   static_cast<__int128>(hi + 1) * bytes <= INT64_MAX;
    if (!bounded) {
        return std::nullopt;
    }
    return std::make_pair(static_cast<int64_t>(static_cast<__int128>(lo) * bytes),
                          static_cast<uint64_t>(static_cast<__int128>(hi - lo + 1) * bytes));
}
