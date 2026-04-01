// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOVERIFYTFREE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static TFreeOp findMatchingTFree(TPopOp tpopOp) {
  Value pipeHandle = tpopOp.getPipeHandle();
  Block *block = tpopOp->getBlock();
  for (auto it = std::next(tpopOp->getIterator()), end = block->end();
       it != end; ++it) {
    if (auto tfreeOp = dyn_cast<TFreeOp>(&*it)) {
      if (tfreeOp.getPipeHandle() == pipeHandle)
        return tfreeOp;
    }
  }
  return {};
}

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  Operation *current = op;
  while (current && current->getBlock() != block) {
    Region *parentRegion = current->getParentRegion();
    if (!parentRegion)
      return nullptr;
    current = parentRegion->getParentOp();
  }
  return current;
}

static bool hasSamePipeTPopInRegion(Operation *op, Value pipeHandle,
                                    TPopOp current) {
  bool found = false;
  op->walk([&](TPopOp nestedTpop) {
    if (nestedTpop == current)
      return WalkResult::advance();
    if (nestedTpop.getPipeHandle() == pipeHandle) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult verifySingleOutstandingUntil(TPopOp tpopOp,
                                                  Operation *freeBoundary) {
  if (!freeBoundary || freeBoundary == tpopOp.getOperation())
    return success();

  Value pipeHandle = tpopOp.getPipeHandle();
  Block *block = tpopOp->getBlock();
  for (auto it = std::next(tpopOp->getIterator()), end = block->end();
       it != end; ++it) {
    Operation *op = &*it;
    if (hasSamePipeTPopInRegion(op, pipeHandle, tpopOp)) {
      return tpopOp.emitOpError(
          "multiple outstanding pops on the same pipe are not supported");
    }
    if (op == freeBoundary)
      break;
  }

  return success();
}

static LogicalResult verifyNoTileUsesAfterTFree(TPopOp tpopOp,
                                                TFreeOp tfreeOp) {
  Value tile = tpopOp.getTile();
  Block *block = tpopOp->getBlock();

  for (OpOperand &use : tile.getUses()) {
    Operation *topLevelOwner = getTopLevelAncestorInBlock(use.getOwner(), block);
    if (!topLevelOwner) {
      return tpopOp.emitOpError(
          "borrowed tile uses must stay in the same parent block as the producing tpop");
    }
    if (tfreeOp->isBeforeInBlock(topLevelOwner)) {
      return tpopOp.emitOpError(
          "tfree must appear after the last use of the borrowed tile");
    }
  }

  return success();
}

static bool isInsideSectionOrAttributedKernel(TPopOp tpopOp, func::FuncOp funcOp) {
  if (tpopOp->getParentOfType<SectionCubeOp>() ||
      tpopOp->getParentOfType<SectionVectorOp>())
    return true;
  return funcOp &&
         funcOp->hasAttr(FunctionKernelKindAttr::name);
}

struct PTOVerifyTFreePass
    : public mlir::pto::impl::PTOVerifyTFreeBase<PTOVerifyTFreePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<TPopOp> tpops;
    funcOp.walk([&](TPopOp op) { tpops.push_back(op); });

    for (TPopOp tpopOp : tpops) {
      if (!isInsideSectionOrAttributedKernel(tpopOp, funcOp))
        continue;

      TFreeOp existingTFree = findMatchingTFree(tpopOp);
      if (!existingTFree) {
        tpopOp.emitOpError("requires an explicit matching tfree");
        signalPassFailure();
        return;
      }

      if (failed(
              verifySingleOutstandingUntil(tpopOp, existingTFree.getOperation()))) {
        signalPassFailure();
        return;
      }

      if (failed(verifyNoTileUsesAfterTFree(tpopOp, existingTFree))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVerifyTFreePass() {
  return std::make_unique<PTOVerifyTFreePass>();
}
