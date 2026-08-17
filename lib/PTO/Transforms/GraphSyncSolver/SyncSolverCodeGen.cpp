// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===---------- SyncSolverCodeGen.cpp ---- Graph Sync Solver --------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

static PipeAttr makePipe(MLIRContext *ctx, PIPE pipe) {
  return PipeAttr::get(ctx, pipe);
}

static EventAttr makeEvent(MLIRContext *ctx, int64_t eventId) {
  return EventAttr::get(ctx, static_cast<EVENT>(eventId));
}

Operation *CodeGenerator::resolveSyncAnchor(OperationBase *opBase) {
  if (!opBase) {
    return nullptr;
  }
  if (auto *ph = dyn_cast<PlaceHolder>(opBase)) {
    if (ph->beforeOp) {
      return ph->beforeOp->op;
    }
    if (ph->afterOp) {
      return ph->afterOp->op;
    }
    if (ph->block) {
      return ph->block->getParentOp();
    }
    return nullptr;
  }
  return opBase->op;
}

Location CodeGenerator::resolveSyncLoc(OperationBase *opBase) {
  if (Operation *anchor = resolveSyncAnchor(opBase)) {
    return anchor->getLoc();
  }
  return funcOp.getLoc();
}

void CodeGenerator::setInsertionPoint(IRRewriter &rewriter,
                                      OperationBase *opBase,
                                      bool insertAfter) {
  if (auto *ph = dyn_cast_or_null<PlaceHolder>(opBase)) {
    // Block-start placeholder: insert at the very top of the block.
    if (ph->scopeBegin && ph->block) {
      rewriter.setInsertionPointToStart(ph->block);
      return;
    }
    // Block-end placeholder: insert before the terminator if any, otherwise
    // at the end of the block.
    if (ph->scopeEnd && ph->block) {
      if (!ph->block->empty() &&
          ph->block->back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(&ph->block->back());
      } else {
        rewriter.setInsertionPointToEnd(ph->block);
}
      return;
    }
    // Loop-boundary slot. The placeholder names the linked loop op via
    // beforeOp/afterOp; the side is picked by the `insertAfter` flag, which
    // by the solver's convention agrees with the field that is set.
    OperationBase *linked = ph->beforeOp ? ph->beforeOp : ph->afterOp;
    if (linked && linked->op) {
      if (insertAfter) {
        rewriter.setInsertionPointAfter(linked->op);
      } else {
        rewriter.setInsertionPoint(linked->op);
}
      return;
    }
    // Malformed placeholder: fall back to the function entry to keep the
    // pass from crashing.
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    return;
  }
  Operation *anchor = opBase ? opBase->op : nullptr;
  if (!anchor) {
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    return;
  }
  if (insertAfter) {
    rewriter.setInsertionPointAfter(anchor);
  } else {
    rewriter.setInsertionPoint(anchor);
}
}

void CodeGenerator::emitSyncOp(IRRewriter &rewriter, SyncOp *syncOp) {
  if (auto *barrier = dyn_cast<BarrierOp>(syncOp)) {
    rewriter.create<pto::BarrierOp>(resolveSyncLoc(barrier),
                                    makePipe(rewriter.getContext(),
                                             barrier->pipe));
    return;
  }

  auto *setWait = dyn_cast<SetWaitOp>(syncOp);
  if (!setWait || setWait->eventIds.empty()) {
    return;
  }

  // The first/last-iter wrapping path (scf.if(isFirstIter/isLastIter) {
  // set/wait }) lives behind the MmadL1 decomposition optimization in the
  // solver, which is currently force-disabled by SyncSolverOptions ctor.
  // If anyone re-enables it, codegen needs a matching update before this
  // assert can be relaxed.
  assert(!setWait->checkFirstIter &&
         "checkFirstIter wrapping not implemented in codegen");
  assert(!setWait->checkLastIter &&
         "checkLastIter wrapping not implemented in codegen");

  auto srcAttr = makePipe(rewriter.getContext(), setWait->pipeSrc);
  auto dstAttr = makePipe(rewriter.getContext(), setWait->pipeDst);
  Location loc = resolveSyncLoc(setWait);
  bool isSet = isa<SetFlagOp>(setWait);
  bool isWait = isa<WaitFlagOp>(setWait);

  // Multi-buffer dyn-event path: when the sync was produced by a
  // multi-buffer back-edge (eventIds.size() > 1 AND we have a slot SSA),
  // collapse the per-slot fanout into a single `pto.set_flag_dyn` /
  // `pto.wait_flag_dyn`. The hardware event id is selected at runtime by
  // an N-way `arith.select` chain over the allocated event ids keyed off
  // `slotSSAExpr % N`. The `allAtOnce` scopes (pre-loop prime / post-loop
  // drain) keep their static fanout so each per-slot event gets primed /
  // drained exactly once.
  if (!setWait->allAtOnce && setWait->eventIds.size() > 1 &&
      setWait->slotSSAExpr) {
    int64_t n = static_cast<int64_t>(setWait->eventIds.size());
    Value slot = setWait->slotSSAExpr;
    if (slot.getType() != rewriter.getIndexType()) {
      slot = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), slot);
    }
    Value nConst = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value slotMod = rewriter.create<arith::RemUIOp>(loc, slot, nConst);

    Value selected = rewriter.create<arith::ConstantIndexOp>(
        loc, setWait->eventIds[0]);
    for (int64_t i = 1; i < n; ++i) {
      Value iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value isThis = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, slotMod, iIdx);
      Value idI = rewriter.create<arith::ConstantIndexOp>(
          loc, setWait->eventIds[static_cast<size_t>(i)]);
      selected = rewriter.create<arith::SelectOp>(loc, isThis, idI, selected);
    }

    if (isSet) {
      rewriter.create<pto::SetFlagDynOp>(loc, srcAttr, dstAttr, selected);
    } else if (isWait) {
      rewriter.create<pto::WaitFlagDynOp>(loc, srcAttr, dstAttr, selected);
    }
    return;
  }

  // Fallback / scope-anchored path: one static set/wait per assigned event
  // id. Always used for `allAtOnce` prime/drain pairs and for syncs that
  // could not have their slot SSA recovered (in which case we degrade to
  // the conservative N-static fanout rather than dropping the dep).
  for (int64_t eventId : setWait->eventIds) {
    auto eventAttr = makeEvent(rewriter.getContext(), eventId);
    if (isSet) {
      rewriter.create<pto::SetFlagOp>(loc, srcAttr, dstAttr, eventAttr);
    } else if (isWait) {
      rewriter.create<pto::WaitFlagOp>(loc, srcAttr, dstAttr, eventAttr);
    }
  }
}

void CodeGenerator::emitSyncMap(IRRewriter &rewriter, SyncMap &syncMap,
                                bool insertAfter) {
  for (auto &[opBase, syncOps] : syncMap) {
    setInsertionPoint(rewriter, opBase, insertAfter);
    for (auto &syncOp : syncOps) {
      emitSyncOp(rewriter, syncOp.get());
    }
  }
}

void CodeGenerator::generateResultOps() {
  IRRewriter rewriter(funcOp.getContext());
  emitSyncMap(rewriter, syncMapBefore, /*insertAfter=*/false);
  emitSyncMap(rewriter, syncMapAfter, /*insertAfter=*/true);
}
