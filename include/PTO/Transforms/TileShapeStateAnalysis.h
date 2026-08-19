// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_TILESHAPESTATEANALYSIS_H
#define PTO_TRANSFORMS_TILESHAPESTATEANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {

/// The shape facts used by candidate selection, expansion and fusion.  A
/// missing or non-dominating valid_shape update is Unknown; it is never
/// silently treated as a full tile.
struct TileShapeState {
  enum class Kind { Full, Partial, Unknown } kind = Kind::Unknown;
  llvm::SmallVector<int64_t, 2> shape;
  llvm::SmallVector<int64_t, 2> validShape;

  bool isFull() const { return kind == Kind::Full; }
  bool isPartial() const { return kind == Kind::Partial; }
  bool isUnknown() const { return kind == Kind::Unknown; }
};

/// Resolve the latest statically-known valid_shape update that dominates
/// `useOp`.  If the control-flow path does not prove one unique value, return
/// false.  With no use operation, only an unambiguous declaration/update is
/// accepted.
bool resolveStaticTileValidShape(Value value,
                                 llvm::SmallVectorImpl<int64_t> &validShape,
                                 Operation *useOp = nullptr);

TileShapeState analyzeTileShape(Value value, Operation *useOp = nullptr);

/// Returns true only when every tile operand has a statically-proven valid
/// shape equal to its physical shape.  Operations without tile operands are
/// not rejected by this helper.
bool hasStaticFullTileValidShape(Operation *op);

} // namespace pto
} // namespace mlir

#endif
