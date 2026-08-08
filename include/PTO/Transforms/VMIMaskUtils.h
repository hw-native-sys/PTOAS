// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskUtils.h - Shared VMI predicate / seed helpers -----*- C++ -*-===//
//
// Helpers shared by VMILowerUnifiedToLegacy, VMIPredicateFold, and related
// passes for proving mask shape and classifying compile-time predicates.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMIMASKUTILS_H
#define PTO_TRANSFORMS_VMIMASKUTILS_H

#include "mlir/IR/Value.h"
#include <optional>

namespace mlir {
namespace pto {

/// Inclusive integer range used for affine / lane-index proofs.
struct IntRange {
  int64_t lo = 0;
  int64_t hi = 0;

  static IntRange splat(int64_t c) { return {c, c}; }
};

/// Compile-time mask lattice.
enum class MaskLattice { Unknown, AllTrue, AllFalse };

/// Returns true if `seed` is provably an all-active mask (every lane active),
/// so `mask_and(x, seed)` is the identity. Covers a `pset` and a
/// `create_mask` whose active_lanes is a constant >= the mask lane count.
bool isAllActiveSeed(Value seed);

/// Returns true if `seed` is provably an all-inactive mask (every lane
/// inactive). Covers `create_mask(0)`.
bool isAllInactiveSeed(Value seed);

/// Bound an integer SSA value over known constant / affine forms.
std::optional<IntRange> matchAffineIntRange(Value v);

/// Bound every lane of a VMI vector to an inclusive integer range when the
/// producer is a statically analyzable index form (vci / vadds / vbrc / …).
std::optional<IntRange> matchVectorLaneRange(Value v);

/// Classify a VMI mask SSA value as AllTrue / AllFalse / Unknown.
MaskLattice classifyMaskValue(Value mask);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_VMIMASKUTILS_H
