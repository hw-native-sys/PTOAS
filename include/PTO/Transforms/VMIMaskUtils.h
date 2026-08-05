// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskUtils.h - Shared VMI predicate / seed helpers -----*- C++ -*-===//
//
// Helpers shared by VMILowerUnifiedToLegacy and VMIPredicateFold for proving
// that a mask SSA value is statically all-active or all-inactive.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMIMASKUTILS_H
#define PTO_TRANSFORMS_VMIMASKUTILS_H

#include "mlir/IR/Value.h"

namespace mlir {
namespace pto {

/// Returns true if `seed` is provably an all-active mask (every lane active),
/// so `mask_and(x, seed)` is the identity. Covers a `pset` and a
/// `create_mask` whose active_lanes is a constant >= the mask lane count.
bool isAllActiveSeed(Value seed);

/// Returns true if `seed` is provably an all-inactive mask (every lane
/// inactive). Covers `create_mask(0)`.
bool isAllInactiveSeed(Value seed);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_VMIMASKUTILS_H
