// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_ANALYSIS_PTOMEMORYALIASANALYSIS_H
#define PTO_ANALYSIS_PTOMEMORYALIASANALYSIS_H
#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "PTO/IR/VPTOScheduling.h"

namespace mlir::pto {
std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation* op);
struct IntegerRange {
    int64_t lowerInclusive = 0;
    int64_t upperInclusive = 0;
};

struct ResolvedMemoryAccess {
    VPTOMemoryAccess semantics;
    Value aliasRoot;
    std::optional<int64_t> absoluteByteOffset;
    std::optional<IntegerRange> absoluteByteRange;
    std::optional<PTOAddressExpr> addressExpression;
};

// Resolve instruction-declared access size and typed address facts without
// applying scheduling, memory-frontier or synchronization policy.
SmallVector<ResolvedMemoryAccess> resolveMemoryAccesses(
    Operation* operation, const VPTOSchedulingSemantics& semantics, PTOAddressAnalysis* addresses);
std::optional<int64_t> getExactByteDifference(const ResolvedMemoryAccess& from, const ResolvedMemoryAccess& to);
bool mayAlias(const ResolvedMemoryAccess& lhs, const ResolvedMemoryAccess& rhs);
} // namespace mlir::pto
#endif
