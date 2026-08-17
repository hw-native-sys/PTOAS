// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFLATTENFUSIONREGION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static LogicalResult flattenFusionRegion(pto::FusionRegionOp fusionRegion) {
  Block &body = fusionRegion.getBody().front();
  auto yieldOp = dyn_cast<pto::YieldOp>(body.getTerminator());
  if (!yieldOp) {
    return fusionRegion.emitOpError("expects body to terminate with pto.yield");
  }

  SmallVector<Value, mlir::pto::kValue8> yieldedValues(yieldOp.getValues().begin(),
                                      yieldOp.getValues().end());

  Operation *anchor = fusionRegion.getOperation();
  SmallVector<Operation *, mlir::pto::kValue16> opsToMove;
  opsToMove.reserve(body.getOperations().size());
  for (Operation &op : body.without_terminator()) {
    opsToMove.push_back(&op);
  }

  for (Operation *op : opsToMove) {
    op->moveBefore(anchor);
  }

  for (auto [result, replacement] :
       llvm::zip(anchor->getResults(), yieldedValues)) {
    result.replaceAllUsesWith(replacement);
  }

  yieldOp.erase();
  anchor->erase();
  return success();
}

struct PTOFlattenFusionRegionPass
    : public pto::impl::PTOFlattenFusionRegionBase<
          PTOFlattenFusionRegionPass> {
  using pto::impl::PTOFlattenFusionRegionBase<
      PTOFlattenFusionRegionPass>::PTOFlattenFusionRegionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    SmallVector<pto::FusionRegionOp, mlir::pto::kValue8> fusionRegions;
    func.walk<WalkOrder::PostOrder>([&](pto::FusionRegionOp fusionRegion) {
      fusionRegions.push_back(fusionRegion);
    });

    if (fusionRegions.empty()) {
      markAllAnalysesPreserved();
      return;
    }

    for (pto::FusionRegionOp fusionRegion : fusionRegions) {
      if (failed(flattenFusionRegion(fusionRegion))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFlattenFusionRegionPass() {
  return std::make_unique<PTOFlattenFusionRegionPass>();
}
