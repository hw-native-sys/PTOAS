// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_ANALYSIS_PTOMEMORY_INDEX_RANGE_H
#define PTO_ANALYSIS_PTOMEMORY_INDEX_RANGE_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::pto::integer_range {

bool allIntegerLanes(Value mask, uint64_t lanes, unsigned width = 0);
std::optional<llvm::SmallVector<uint64_t, 128>> constantIndexLanes(
    Value value, unsigned& budget, bool allowTable = false);
std::optional<std::pair<int64_t, uint64_t>> boundedIndexByteRange(Value offsets, Value mask, uint64_t bytes);

} // namespace mlir::pto::integer_range
#endif
