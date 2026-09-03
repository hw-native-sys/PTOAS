// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SlotAffineAnalysis.h - Multi-buffer slot affine compare --*- C++ -*-===//
//
// Small affine helper used by the multi-buffer sync path. InsertSync consumes
// it to decide, for two `pto.multi_tile_get`
// slot-index SSA expressions, whether they are provably equal modulo N,
// provably disjoint modulo N, or indeterminate. The result lets sync
// shrink event-id count or skip same-iter forward syncs entirely when
// producer and consumer touch different slots in every iteration.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_SLOTAFFINEANALYSIS_H
#define PTO_TRANSFORMS_SLOTAFFINEANALYSIS_H

#include "mlir/IR/Value.h"
#include <cstdint>

namespace mlir {
namespace pto {

/// Three-valued relation between two multi-buffer slot SSA expressions
/// taken modulo `N`. Anything we cannot prove statically degrades to
/// `kUnknown`, which the callers treat conservatively (i.e. fall back to
/// the existing all-slots-may-overlap path).
enum class SlotRelation {
  kEqual,    // a(iv) == b(iv)  (mod N) for every iv
  kDisjoint, // a(iv) != b(iv)  (mod N) for every iv
  kUnknown,  // can neither prove equal nor disjoint
};

/// Return the slot SSA value carried by `pto.multi_tile_get`. Returns null
/// if the chain does not pass through a multi_tile_get.
mlir::Value findMultiTileSlotExpr(mlir::Value v);

/// Compare two slot SSA expressions modulo `N`. The analysis is
/// intentionally narrow: it accepts the forms commonly produced by
/// frontends and lowerings (`iv % N`, `(iv + c) % N`, `c`, and same-SSA
/// equality) and bails to `kUnknown` for anything else.
/// Examples (all with N == 2):
///   compareSlotSSA(%iv % 2, %iv % 2)         -> kEqual
///   compareSlotSSA((%iv + 1) % 2, %iv % 2)   -> kDisjoint
///   compareSlotSSA((%iv + 3) % 2, %iv % 2)   -> kDisjoint  // 3 % 2 == 1
///   compareSlotSSA((%iv + 2) % 2, %iv % 2)   -> kEqual     // 2 % 2 == 0
///   compareSlotSSA(%iv % 2, %j % 2)          -> kUnknown   // diff symbols
///   compareSlotSSA(arith.constant 0, arith.constant 1) -> kDisjoint
SlotRelation compareSlotSSA(mlir::Value a, mlir::Value b, uint32_t N);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_SLOTAFFINEANALYSIS_H
