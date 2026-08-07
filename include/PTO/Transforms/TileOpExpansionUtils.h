// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_TILEOPEXPANSIONUTILS_H
#define PTO_TRANSFORMS_TILEOPEXPANSIONUTILS_H

#include "PTO/IR/PTO.h"

#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace mlir::pto {

/// Return whether an operation is a TileLib template expansion candidate.
/// Frontend pipe/sync pseudo-ops use TileOpInterface for surface
/// classification but must be handled by their dedicated lowering instead.
inline bool isTileLibExpandableOp(Operation *op) {
  if (!op || !isa<TileOpInterface>(op))
    return false;
  return !isa<TReshapeOp, TSyncOp, TAllocToAivOp, TAllocToAicOp,
              TPushToAivOp, TPushToAicOp, TPopFromAicOp, TPopFromAivOp,
              TFreeFromAicOp, TFreeFromAivOp>(op);
}

/// Append MScatterOp context attributes to the spec-key attribute list.
/// Only non-default values are included to avoid spurious key divergence.
inline void appendMScatterContextAttrs(
    Operation *op,
    SmallVectorImpl<std::pair<std::string, std::string>> &attrs) {
  auto mscatter = dyn_cast<pto::MScatterOp>(op);
  if (!mscatter)
    return;
  if (auto oobAttr = dyn_cast_or_null<pto::ScatterOOBAttr>(
          mscatter.getProperties().scatterOob)) {
    if (oobAttr.getValue() != pto::ScatterOOB::Undefined) {
      attrs.emplace_back("scatter_oob",
                         stringifyScatterOOB(oobAttr.getValue()).str());
    }
  }
  if (auto atomicOpAttr = dyn_cast_or_null<pto::ScatterAtomicOpAttr>(
          mscatter.getProperties().scatterAtomicOp)) {
    auto atomicOpValue = atomicOpAttr.getValue();
    if (atomicOpValue != pto::ScatterAtomicOp::None) {
      attrs.emplace_back("scatter_atomic_op",
                         stringifyScatterAtomicOp(atomicOpValue).str());
    }
  }
  if (auto conflictAttr = dyn_cast_or_null<pto::ScatterConflictAttr>(
          mscatter.getProperties().scatterConflict)) {
    if (conflictAttr.getValue() != pto::ScatterConflict::Default) {
      attrs.emplace_back("scatter_conflict",
                         stringifyScatterConflict(conflictAttr.getValue()).str());
    }
  }
}

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_TILEOPEXPANSIONUTILS_H
