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
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERFRONTENDPIPEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct FrontendPipeHandles {
  Value c2vPipe;
  Value v2cPipe;
  Operation *anchorOp = nullptr;
};

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerFrontendInitOp(InitOpT initOp,
                                                          IRRewriter &rewriter) {
  FrontendPipeHandles handles;
  Location loc = initOp.getLoc();
  MLIRContext *ctx = initOp.getContext();
  auto pipeTy = PipeType::get(ctx);
  PTOArch arch = getTargetArch(initOp.getOperation());

  auto createPipe = [&](int8_t dirMask, int32_t slotNum,
                        Value localAddr) -> FailureOr<Value> {
    auto dirAttr = rewriter.getI8IntegerAttr(dirMask);
    auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
    auto slotNumAttr = rewriter.getI32IntegerAttr(slotNum);

    if (arch == PTOArch::A5) {
      auto pipe = rewriter.create<InitializeL2LPipeOp>(
          loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
          localAddr, /*peer_local_addr=*/Value{});
      return pipe.getPipe();
    }

    if (!initOp.getGmSlotBuffer()) {
      initOp.emitOpError("requires 'gm_slot_buffer' when lowering to a2/a3");
      return failure();
    }

    auto localSlotNumAttr = rewriter.getI32IntegerAttr(slotNum);
    auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
        loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
        IntegerAttr{}, initOp.getGmSlotBuffer(), localAddr,
        /*peer_local_addr=*/Value{});
    return pipe.getPipe();
  };

  switch (initOp.getDirMask()) {
  case 1: {
    auto pipeOr =
        createPipe(/*dirMask=*/1, /*slotNum=*/8, initOp.getC2vConsumerBuf());
    if (failed(pipeOr))
      return failure();
    handles.c2vPipe = *pipeOr;
    handles.anchorOp = handles.c2vPipe.getDefiningOp();
    break;
  }
  case 2: {
    auto pipeOr =
        createPipe(/*dirMask=*/2, /*slotNum=*/8, initOp.getV2cConsumerBuf());
    if (failed(pipeOr))
      return failure();
    handles.v2cPipe = *pipeOr;
    handles.anchorOp = handles.v2cPipe.getDefiningOp();
    break;
  }
  case 3: {
    auto dirAttr = rewriter.getI8IntegerAttr(3);
    auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
    auto slotNumAttr = rewriter.getI32IntegerAttr(4);
    Value c2vAddr = initOp.getC2vConsumerBuf();
    Value v2cAddr = initOp.getV2cConsumerBuf();

    if (arch == PTOArch::A5) {
      auto pipe = rewriter.create<InitializeL2LPipeOp>(
          loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
          c2vAddr, v2cAddr);
      handles.c2vPipe = pipe.getPipe();
      handles.v2cPipe = pipe.getPipe();
      handles.anchorOp = pipe.getOperation();
    } else {
      if (!initOp.getGmSlotBuffer()) {
        initOp.emitOpError("requires 'gm_slot_buffer' when lowering to a2/a3");
        return failure();
      }
      auto localSlotNumAttr = rewriter.getI32IntegerAttr(4);
      auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
          loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
          IntegerAttr{}, initOp.getGmSlotBuffer(), c2vAddr, v2cAddr);
      handles.c2vPipe = pipe.getPipe();
      handles.v2cPipe = pipe.getPipe();
      handles.anchorOp = pipe.getOperation();
    }
    break;
  }
  default:
    break;
  }

  return handles;
}

static FailureOr<FrontendPipeHandles> lowerInitIfPresent(func::FuncOp funcOp,
                                                         IRRewriter &rewriter) {
  FrontendPipeHandles handles;
  AicInitializePipeOp aicInit;
  AivInitializePipeOp aivInit;
  unsigned aicInitCount = 0;
  unsigned aivInitCount = 0;

  funcOp.walk([&](Operation *op) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      ++aicInitCount;
      if (!aicInit)
        aicInit = init;
      return WalkResult::advance();
    }
    if (auto init = dyn_cast<AivInitializePipeOp>(op)) {
      ++aivInitCount;
      if (!aivInit)
        aivInit = init;
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (aicInitCount > 1) {
    funcOp.emitOpError("requires at most one pto.aic_initialize_pipe");
    return failure();
  }

  if (aivInitCount > 1) {
    funcOp.emitOpError("requires at most one pto.aiv_initialize_pipe");
    return failure();
  }

  if (aicInit && aivInit) {
    funcOp.emitOpError(
        "cannot mix pto.aic_initialize_pipe and pto.aiv_initialize_pipe in one function");
    return failure();
  }

  if (!aicInit && !aivInit)
    return handles;

  if (aicInit) {
    rewriter.setInsertionPoint(aicInit);
    auto loweredOr = lowerFrontendInitOp(aicInit, rewriter);
    if (failed(loweredOr))
      return failure();
    handles = *loweredOr;
    rewriter.eraseOp(aicInit);
  } else {
    rewriter.setInsertionPoint(aivInit);
    auto loweredOr = lowerFrontendInitOp(aivInit, rewriter);
    if (failed(loweredOr))
      return failure();
    handles = *loweredOr;
    rewriter.eraseOp(aivInit);
  }

  return handles;
}

static bool hasFrontendPipeOps(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<AicInitializePipeOp, AivInitializePipeOp, TPushToAivOp, TPushToAicOp,
            TPopFromAicOp, TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult lowerFrontendDataOps(func::FuncOp funcOp,
                                          const FrontendPipeHandles &handles,
                                          IRRewriter &rewriter) {
  DominanceInfo dom(funcOp);
  SmallVector<Operation *> frontendOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TPushToAivOp, TPushToAicOp, TPopFromAicOp, TPopFromAivOp,
            TFreeFromAicOp, TFreeFromAivOp>(op))
      frontendOps.push_back(op);
  });

  for (Operation *op : frontendOps) {
    if (!handles.anchorOp) {
      op->emitOpError("requires a frontend initialize_pipe op in the same function");
      return failure();
    }
    if (!dom.dominates(handles.anchorOp, op)) {
      op->emitOpError(
          "requires a dominating frontend initialize_pipe op");
      return failure();
    }

    rewriter.setInsertionPoint(op);

    if (auto push = dyn_cast<TPushToAivOp>(op)) {
      if (!handles.c2vPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable C2V");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.c2vPipe,
                                           push.getSplitAttr());
      continue;
    }

    if (auto push = dyn_cast<TPushToAicOp>(op)) {
      if (!handles.v2cPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable V2C");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.v2cPipe,
                                           push.getSplitAttr());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAicOp>(op)) {
      if (!handles.c2vPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable C2V");
        return failure();
      }
      auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                 pop.getTile().getType());
      rewriter.create<TPopOp>(pop.getLoc(), decl.getTile(), handles.c2vPipe,
                              pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAivOp>(op)) {
      if (!handles.v2cPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable V2C");
        return failure();
      }
      auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                 pop.getTile().getType());
      rewriter.create<TPopOp>(pop.getLoc(), decl.getTile(), handles.v2cPipe,
                              pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    if (auto free = dyn_cast<TFreeFromAicOp>(op)) {
      if (!handles.c2vPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable C2V");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TFreeOp>(free, handles.c2vPipe,
                                           free.getSplitAttr());
      continue;
    }

    auto free = cast<TFreeFromAivOp>(op);
    if (!handles.v2cPipe) {
      op->emitOpError(
          "requires the dominating initialize_pipe op to enable V2C");
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFreeOp>(free, handles.v2cPipe,
                                         free.getSplitAttr());
  }

  return success();
}

struct PTOLowerFrontendPipeOpsPass
    : public mlir::pto::impl::PTOLowerFrontendPipeOpsBase<
          PTOLowerFrontendPipeOpsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasFrontendPipeOps(funcOp))
      return;

    IRRewriter rewriter(funcOp.getContext());
    auto loweredOr = lowerInitIfPresent(funcOp, rewriter);
    if (failed(loweredOr)) {
      signalPassFailure();
      return;
    }

    if (failed(lowerFrontendDataOps(funcOp, *loweredOr, rewriter)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerFrontendPipeOpsPass() {
  return std::make_unique<PTOLowerFrontendPipeOpsPass>();
}
