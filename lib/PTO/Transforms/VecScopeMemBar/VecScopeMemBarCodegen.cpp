// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VecScopeMemBarCodegen.cpp --------------------------------------===//
//
// Applies a barrier plan to the IR. Validates anchors, de-dupes
// by anchor/kind (done in placement), normalizes multiple directed kinds at
// one anchor to a single VV_ALL, inserts `pto.mem_bar` with an IRRewriter in
// deterministic order, and performs no new alias/hazard reasoning.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarCodegen.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

// True if `anchor` still lies inside the scope (was not invalidated by an
// earlier mutation in this run). Inserting before `anchor` keeps prior
// anchors valid because we walk anchors in lexical order.
static bool anchorIsValid(const VecScopeMemBarAnalysisResult &result,
                          Operation *anchor) {
  if (!anchor)
    return false;
  Operation *cur = anchor->getParentOp();
  while (cur) {
    if (cur == result.scope)
      return true;
    cur = cur->getParentOp();
  }
  return false;
}

} // namespace

LogicalResult vecscopemembar::applyVecScopeMemBarPlan(
    const VecScopeMemBarAnalysisResult &result,
    const VecScopeMemBarPlan &plan) {
  // Insert in deterministic order. The plan is already sorted by anchor
  // lexical order Inserting before an anchor does not
  // invalidate later anchors that are lexically after it.
  for (const auto &bp : plan.barriers) {
    if (!anchorIsValid(result, bp.anchor))
      return failure();
    if (bp.kind == MemBarKind::VV_ALL) {
      if (auto existing =
              dyn_cast_or_null<pto::MemBarOp>(bp.anchor->getPrevNode())) {
        MemBarKind existingKind = existing.getKind().getKind();
        if (existingKind != MemBarKind::VV_ALL)
          existing.erase();
      }
    }
    OpBuilder builder(bp.anchor);
    auto attr = pto::MemBarAttr::get(bp.anchor->getContext(), bp.kind);
    builder.create<pto::MemBarOp>(bp.anchor->getLoc(), attr);
  }
  return success();
}
