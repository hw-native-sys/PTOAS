// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "IntervalMemoryAllocator.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <tuple>

using namespace mlir::pto;
namespace {
using Interval = std::pair<uint64_t, uint64_t>;
struct Candidate {
    uint64_t offset = 0;
    uint64_t waste = 0;
    uint64_t cost = 0;
};
bool overlaps(uint64_t lhs, uint64_t lhsSize, uint64_t rhs, uint64_t rhsSize)
{
    return lhsSize != 0 && rhsSize != 0 && lhs < rhs + rhsSize && rhs < lhs + lhsSize;
}
llvm::SmallVector<Interval> getBlockers(
    unsigned index, llvm::ArrayRef<AllocRequest> requests, llvm::ArrayRef<unsigned> placed, const AllocResult& result,
    ConflictFn conflictFn)
{
    llvm::SmallVector<Interval> intervals;
    for (unsigned other : placed) {
        if (requests[other].sizeBytes != 0 &&
            (requests[index].pinned || requests[other].pinned || conflictFn(index, other))) {
            intervals.emplace_back(result.offsets[other], result.offsets[other] + requests[other].sizeBytes);
        }
    }
    llvm::sort(intervals);
    llvm::SmallVector<Interval> merged;
    for (const Interval& interval : intervals) {
        if (merged.empty() || interval.first > merged.back().second) {
            merged.push_back(interval);
        } else {
            merged.back().second = std::max(merged.back().second, interval.second);
        }
    }
    return merged;
}
uint64_t placementCost(
    unsigned index, uint64_t offset, llvm::ArrayRef<AllocRequest> requests, llvm::ArrayRef<unsigned> placed,
    const AllocResult& result, CostFn costFn)
{
    uint64_t cost = 0;
    for (unsigned other : placed) {
        if (overlaps(offset, requests[index].sizeBytes, result.offsets[other], requests[other].sizeBytes)) {
            uint64_t penalty = costFn(index, offset, other, result.offsets[other]);
            cost += std::min(penalty, std::numeric_limits<uint64_t>::max() - cost);
        }
    }
    return cost;
}
std::optional<Candidate> chooseGap(
    unsigned index, llvm::ArrayRef<AllocRequest> requests, llvm::ArrayRef<unsigned> placed, const AllocResult& result,
    llvm::ArrayRef<Interval> blockers, uint64_t capacityBytes, CostFn costFn, AllocFailure& failureOut)
{
    const AllocRequest& request = requests[index];
    std::optional<Candidate> best;
    uint64_t cursor = 0;
    auto consider = [&](uint64_t end) {
        uint64_t remainder = cursor % request.alignmentBytes;
        uint64_t padding = remainder == 0 ? 0 : request.alignmentBytes - remainder;
        if (cursor > end || padding > end - cursor) {
            return;
        }
        uint64_t start = cursor + padding;
        uint64_t extent = end - start;
        failureOut.largestFreeGapBytes = std::max(failureOut.largestFreeGapBytes, extent);
        if (request.sizeBytes > extent) {
            return;
        }
        Candidate candidate{
            start, extent - request.sizeBytes, placementCost(index, start, requests, placed, result, costFn)};
        if (!best || std::tie(candidate.waste, candidate.cost, candidate.offset) <
                         std::tie(best->waste, best->cost, best->offset)) {
            best = candidate;
        }
    };
    for (const Interval& interval : blockers) {
        consider(interval.first);
        cursor = std::max(cursor, interval.second);
    }
    consider(capacityBytes);
    return best;
}
} // namespace

std::optional<AllocResult> mlir::pto::allocateIntervals(
    llvm::ArrayRef<AllocRequest> requests, uint64_t capacityBytes, OrderPolicy policy, ConflictFn conflictFn,
    CostFn costFn, AllocFailure& failureOut)
{
    if (requests.size() >= std::numeric_limits<unsigned>::max()) {
        failureOut = {};
        return std::nullopt;
    }
    AllocResult result;
    result.offsets.resize(requests.size());
    llvm::SmallVector<unsigned> order;
    for (auto [index, request] : llvm::enumerate(requests)) {
        failureOut = {request.id, request.sizeBytes, 0, 0};
        if (request.alignmentBytes == 0 || index > std::numeric_limits<unsigned>::max()) {
            return std::nullopt;
        }
        order.push_back(static_cast<unsigned>(index));
    }
    llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
        const auto& a = requests[lhs];
        const auto& b = requests[rhs];
        if (a.pinned != b.pinned) {
            return a.pinned;
        }
        if ((a.pinned || policy == OrderPolicy::SizeDescending) && a.sizeBytes != b.sizeBytes) {
            return a.sizeBytes > b.sizeBytes;
        }
        return std::tie(a.orderKey, a.id) < std::tie(b.orderKey, b.id);
    });
    llvm::SmallVector<unsigned> placed;
    for (unsigned index : order) {
        const auto& request = requests[index];
        failureOut = {request.id, request.sizeBytes, 0, result.plannedBytes};
        auto blockers = getBlockers(index, requests, placed, result, conflictFn);
        auto candidate = chooseGap(index, requests, placed, result, blockers, capacityBytes, costFn, failureOut);
        if (!candidate) {
            return std::nullopt;
        }
        result.offsets[index] = candidate->offset;
        result.plannedBytes = std::max(result.plannedBytes, candidate->offset + request.sizeBytes);
        placed.push_back(index);
    }
    return result;
}

bool mlir::pto::verifyAllocationInvariants(
    llvm::ArrayRef<AllocRequest> requests, const AllocResult& result, uint64_t capacityBytes, ConflictFn conflictFn)
{
    if (result.offsets.size() != requests.size() || result.plannedBytes > capacityBytes) {
        return false;
    }
    for (auto [index, request] : llvm::enumerate(requests)) {
        uint64_t offset = result.offsets[index];
        if (request.alignmentBytes == 0 || offset % request.alignmentBytes != 0 || offset > capacityBytes ||
            request.sizeBytes > capacityBytes - offset) {
            return false;
        }
    }
    for (unsigned i = 0; i < requests.size(); ++i) {
        for (unsigned j = i + 1; j < requests.size(); ++j) {
            if ((requests[i].pinned || requests[j].pinned || conflictFn(i, j)) &&
                overlaps(result.offsets[i], requests[i].sizeBytes, result.offsets[j], requests[j].sizeBytes)) {
                return false;
            }
        }
    }
    return true;
}
