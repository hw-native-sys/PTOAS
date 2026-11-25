// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Integer-only interval placement. Callback arguments are request indices;
// stable IDs are used for ordering and diagnostics, never for indexing.
#ifndef PTO_TRANSFORMS_SUPPORT_INTERVALMEMORYALLOCATOR_H
#define PTO_TRANSFORMS_SUPPORT_INTERVALMEMORYALLOCATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstdint>
#include <optional>

namespace mlir::pto {
struct AllocRequest {
    unsigned id = 0;
    uint64_t sizeBytes = 0;
    uint64_t alignmentBytes = 1;
    uint64_t orderKey = 0;
    bool pinned = false;
};
struct AllocResult {
    llvm::SmallVector<uint64_t> offsets;
    uint64_t plannedBytes = 0;
};
struct AllocFailure {
    unsigned failedId = 0;
    uint64_t requestedBytes = 0;
    uint64_t largestFreeGapBytes = 0;
    uint64_t plannedBytes = 0;
};
using ConflictFn = llvm::function_ref<bool(unsigned, unsigned)>;
using CostFn = llvm::function_ref<uint64_t(unsigned, uint64_t, unsigned, uint64_t)>;
enum class OrderPolicy { SizeDescending, ProgramOrder };

// Returns nullopt on capacity exhaustion or invalid request alignment/size.
std::optional<AllocResult> allocateIntervals(
    llvm::ArrayRef<AllocRequest> requests, uint64_t capacityBytes, OrderPolicy policy, ConflictFn conflictFn,
    CostFn costFn, AllocFailure& failureOut);
bool verifyAllocationInvariants(
    llvm::ArrayRef<AllocRequest> requests, const AllocResult& result, uint64_t capacityBytes, ConflictFn conflictFn);
} // namespace mlir::pto
#endif
