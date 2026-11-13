// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_ANALYSIS_PTOINTEGER_RANGE_H
#define PTO_ANALYSIS_PTOINTEGER_RANGE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include <cstdint>
#include <optional>

namespace mlir::pto::integer_range {

struct PTOLoopDomain {
    int64_t lb = 0;
    int64_t ub = 0;
    int64_t step = 1;
    bool valid = false;
    bool exact = false;
    // Upper bound on iterations, including for a runtime upper bound.
    std::optional<int64_t> maxTripCount() const
    {
        if (!valid || step <= 0) {
            return std::nullopt;
        }
        __int128 diff = static_cast<__int128>(ub) - lb;
        if (diff <= 0) {
            return 0;
        }
        __int128 stride = static_cast<__int128>(step);
        __int128 count = (diff + stride - 1) / stride;
        if (count > INT64_MAX) {
            return std::nullopt;
        }
        return static_cast<int64_t>(count);
    }
    std::optional<int64_t> exactTripCount() const { return exact ? maxTripCount() : std::nullopt; }
};
struct AddressRange {
    int64_t lo;
    int64_t hi;
};

// Match K*(r + (r<0 ? c : 0)), r=remsi(x,c), for c>0 and K>=0.
// Require the exact sign mask/shift and value-preserving casts/products.
// Return the closed upper bound K*(c-1), or nullopt if uncertified.
std::optional<int64_t> matchSignCorrectedRemSum(Value sum);

// The same proof for an affine sum whose two terms no longer share an addi.
std::optional<int64_t> matchSignCorrectedRemPair(Value lhs, Value rhs);

std::optional<AddressRange> getSignedRangeAt(Value value, Operation* context);

std::optional<int64_t> getConstantInt(Value value);
// Domain-free inference: unknown leaves remain unknown. A nonnegative loop
// IV has only its type maximum, without using the loop's trip count.
std::optional<ConstantIntRanges> inferPointIntegerRange(Value value, unsigned depth = 0);
ConstantIntRanges inferIntegerRange(Value value, unsigned depth = 0);
PTOLoopDomain getConstBounds(scf::ForOp loop);
PTOLoopDomain getLoopEnvelope(scf::ForOp loop);
std::optional<AddressRange> inferredLeafRange(Value value);
std::optional<AddressRange> getAddressRange(
    Value value, ArrayRef<Value> ivs, ArrayRef<PTOLoopDomain> bounds, unsigned depth = 0);

} // namespace mlir::pto::integer_range
#endif
