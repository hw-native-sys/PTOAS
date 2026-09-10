// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOOptimizeCtrlState.cpp - CTRL state guard optimization ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers pto.ctrl_state_guard ops, produced by MAD
// semantic-to-raw expansion, to the minimal set of hardware
// pto.get_ctrl/pto.sbitset*/pto.set_ctrl accesses.
//
// The pass runs in two stages. The analysis stage computes a bottom-up
// summary for every scf.for subtree: whether the subtree is free of
// CTRL-observing/changing operations other than guards, the unique guard
// requirement if one exists, and whether the loop is statically proven to
// execute at least once. The materialization stage then hoists one shared
// configuration (get_ctrl + bit updates + set_ctrl) to the entry of the
// outermost loop whose whole chain to every guard is statically positive
// and CTRL-clean, unwraps all covered guards in that subtree, and restores
// the logical state once after the loop. Loops with a dynamic or possibly
// zero trip count, with mixed requirements, or with explicit CTRL accesses
// inside are not hoisted; their guards fall through to the per-block stage.
//
// The per-block stage forward-scans each remaining block: it shares one
// logical CTRL entry read across compatible guards, reuses the installed
// active state when the next guard has the same requirement, and defers the
// restore write to the first real observation point: an explicit
// get_ctrl/set_ctrl, a CTRL consumer outside a guard, an unknown operation,
// or the block end. Guards inside scf.if regions are materialized
// branch-locally by the scan of their own block.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_VPTOOPTIMIZECTRLSTATE
#include "PTO/Transforms/Passes.h.inc"

namespace {

static constexpr unsigned kCtrlWidth = 64;

// Statically computed temporary CTRL requirement of one guard.
struct CtrlRequirement {
  uint64_t controlled = 0;
  uint64_t required = 0;

  bool operator==(const CtrlRequirement &other) const {
    return controlled == other.controlled && required == other.required;
  }
};

static CtrlRequirement requirementOf(pto::CtrlStateGuardOp guard) {
  auto iface = cast<StateGuardOpInterface>(guard.getOperation());
  return {iface.getControlledStateBits(), iface.getRequiredStateBits()};
}

// Build the active CTRL value (E & ~C) | R from the logical entry value by
// clearing/setting exactly the controlled bits.
static Value buildActiveValue(Location loc, Value entry, uint64_t controlled,
                              uint64_t required, OpBuilder &builder) {
  Value ctrl = entry;
  for (unsigned bit = 0; bit < kCtrlWidth; ++bit) {
    uint64_t mask = uint64_t(1) << bit;
    if ((controlled & mask) == 0) {
      continue;
    }
    bool value = (required & mask) != 0;
    Value bitValue =
        builder.create<arith::ConstantIntOp>(loc, bit, mlir::pto::kValue64);
    if (value) {
      ctrl = builder.create<pto::Sbitset1Op>(loc, ctrl, bitValue);
    } else {
      ctrl = builder.create<pto::Sbitset0Op>(loc, ctrl, bitValue);
    }
  }
  return ctrl;
}

// Central opt-in transparency rule for the per-block scan. An operation is
// CTRL-transparent when it provably neither reads, writes, nor implicitly
// consumes CTRL. Only arith dialect operations that are memory-effect-free
// and dead-after-deduction qualify; unreviewed PTO ops, unknown dialect
// ops, and region-bearing ops are boundaries, not transparent operations.
static bool isCtrlTransparent(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    return true;
  }
  if (op->getNumRegions() != 0) {
    return false;
  }
  if (isa<StateAccessOpInterface>(op)) {
    return false;
  }
  if (!isa<arith::ArithDialect>(op->getDialect())) {
    return false;
  }
  return isMemoryEffectFree(op) && wouldOpBeTriviallyDead(op);
}

// Whether an operation can observe or change the CTRL state, which makes
// installing a configuration across it illegal. Terminators never observe
// CTRL (the per-block scan restores before block ends separately). Guards
// and structured SCF control flow are handled by their own analysis; raw
// MAD consumers (StateAccess Consume) run under the installed state. Within
// the PTO dialect, CTRL access is exactly what StateAccessOpInterface
// declares; foreign dialect operations are treated conservatively.
static bool canObserveOrChangeCtrl(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  if (isa<pto::CtrlStateGuardOp>(op) || isa<scf::ForOp, scf::IfOp>(op)) {
    return false;
  }
  if (auto access = dyn_cast<StateAccessOpInterface>(op)) {
    return access.getStateAccessKind() != StateAccessKind::Consume;
  }
  if (isa<func::CallOp>(op)) {
    return true;
  }
  if (op->getNumRegions() != 0) {
    return true;
  }
  if (isa<arith::ArithDialect, PTODialect>(op->getDialect())) {
    return false;
  }
  return true;
}

// Bottom-up CTRL summary of one scf.for subtree.
struct LoopCtrlSummary {
  // No CTRL-observing/changing operation in the subtree other than guards.
  bool ctrlClean = true;
  bool hasGuards = false;
  // Valid only while all collected guards share one requirement.
  bool uniqueRequirement = true;
  CtrlRequirement requirement;
  SmallVector<pto::CtrlStateGuardOp, 4> guards;
  // Statically proven that the loop body executes at least once.
  bool tripAtLeastOnce = false;
};

static void mergeGuardInto(pto::CtrlStateGuardOp guard, LoopCtrlSummary &s) {
  CtrlRequirement req = requirementOf(guard);
  if (!s.hasGuards) {
    s.requirement = req;
  } else if (!(req == s.requirement)) {
    s.uniqueRequirement = false;
  }
  s.hasGuards = true;
  s.guards.push_back(guard);
}

static LoopCtrlSummary
summarizeLoop(scf::ForOp forOp,
              DenseMap<Operation *, LoopCtrlSummary> &summaries);

static void scanBlockTree(Block &block, LoopCtrlSummary &s,
                          DenseMap<Operation *, LoopCtrlSummary> &summaries) {
  for (Operation &op : block) {
    if (auto guard = dyn_cast<pto::CtrlStateGuardOp>(op)) {
      mergeGuardInto(guard, s);
      continue;
    }
    if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
      // Inner loops are summarized on first encounter; the recursive walk
      // fills the map inner-to-outer.
      auto [it, inserted] = summaries.try_emplace(innerFor.getOperation());
      if (inserted) {
        it->second = summarizeLoop(innerFor, summaries);
      }
      s.ctrlClean = s.ctrlClean && it->second.ctrlClean;
      for (pto::CtrlStateGuardOp g : it->second.guards) {
        mergeGuardInto(g, s);
      }
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      scanBlockTree(ifOp.getThenRegion().front(), s, summaries);
      if (!ifOp.getElseRegion().empty()) {
        scanBlockTree(ifOp.getElseRegion().front(), s, summaries);
      }
      continue;
    }
    if (canObserveOrChangeCtrl(&op)) {
      s.ctrlClean = false;
    }
  }
}

// Read a constant integer bound; covers both i64 and index-typed
// arith.constant producers, since scf.for bounds may use either.
static bool constantIntValue(Value value, int64_t &out) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }
  auto attr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!attr) {
    return false;
  }
  out = attr.getInt();
  return true;
}

static LoopCtrlSummary
summarizeLoop(scf::ForOp forOp,
              DenseMap<Operation *, LoopCtrlSummary> &summaries) {
  LoopCtrlSummary s;
  int64_t lb = 0;
  int64_t ub = 0;
  int64_t step = 0;
  if (constantIntValue(forOp.getLowerBound(), lb) &&
      constantIntValue(forOp.getUpperBound(), ub) &&
      constantIntValue(forOp.getStep(), step)) {
    s.tripAtLeastOnce = (step > 0 && lb < ub) || (step < 0 && lb > ub);
  }
  scanBlockTree(*forOp.getBody(), s, summaries);
  return s;
}

// Whether every loop on the parent chain from `guard` up to (excluding) the
// hoist root `root` is statically proven to execute at least once. Guards
// may only run under the hoisted configuration when no possibly-zero-trip
// loop sits between the configuration point and the guard.
static bool guardChainProvenOnce(Operation *guard, Operation *root,
                                 DenseMap<Operation *, LoopCtrlSummary> &s) {
  for (Operation *parent = guard->getParentOp(); parent && parent != root;
       parent = parent->getParentOp()) {
    auto parentFor = dyn_cast<scf::ForOp>(parent);
    if (!parentFor) {
      continue;
    }
    auto it = s.find(parentFor.getOperation());
    if (it == s.end() || !it->second.tripAtLeastOnce) {
      return false;
    }
  }
  return true;
}

// Per-block forward-scan state: the logical CTRL value an explicit get_ctrl
// must observe, the value installed in hardware, and the requirement of the
// last guard in the open interval (to reuse an installed configuration).
struct BlockScanState {
  Value logical;
  Value physical;
  bool physicalMatchesLogical = true;
  bool haveLastGuard = false;
  uint64_t lastControlled = 0;
  uint64_t lastRequired = 0;
  bool diverged = false;

  void resetUnknown() {
    logical = physical = Value();
    physicalMatchesLogical = true;
  }

  void observe(Value value) {
    logical = physical = value;
    physicalMatchesLogical = true;
  }
};

class BlockCtrlOptimizer {
public:
  void process(Block *block, OpBuilder &builder) {
    BlockScanState state;
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (isa<pto::CtrlStateGuardOp>(op)) {
        handleGuard(state, cast<pto::CtrlStateGuardOp>(op), builder);
        continue;
      }
      if (auto access = dyn_cast<StateAccessOpInterface>(op)) {
        handleStateAccess(state, *block, access, &op, builder);
        continue;
      }
      if (isCtrlTransparent(&op)) {
        continue;
      }
      // Unknown or side-effecting operation: restore the logical state and
      // restart with unknown (but mutually equal) logical/physical values.
      restoreBefore(state, *block, &op, builder);
      state.resetUnknown();
    }
    // No temporary state may leak past the block end.
    restoreBefore(state, *block, nullptr, builder);
  }

private:
  // Restore the logical state before `anchor` (or at the block end when
  // null) if a temporary active state is installed.
  void restoreBefore(BlockScanState &state, Block &block, Operation *anchor,
                     OpBuilder &builder) {
    if (state.diverged && state.logical) {
      if (anchor) {
        builder.setInsertionPoint(anchor);
      } else if (!block.empty() &&
                 block.back().hasTrait<OpTrait::IsTerminator>()) {
        builder.setInsertionPoint(&block.back());
      } else if (!block.empty()) {
        builder.setInsertionPointAfter(&block.back());
      } else {
        builder.setInsertionPointToStart(&block);
      }
      builder.create<pto::SetCtrlOp>(state.logical.getLoc(), state.logical);
      state.physical = state.logical;
      state.physicalMatchesLogical = true;
      state.diverged = false;
    }
    state.haveLastGuard = false;
  }

  void handleGuard(BlockScanState &state, pto::CtrlStateGuardOp guard,
                   OpBuilder &builder) {
    auto guardIface = cast<StateGuardOpInterface>(guard.getOperation());
    uint64_t controlled = guardIface.getControlledStateBits();
    uint64_t required = guardIface.getRequiredStateBits();

    // Establish the logical entry value once per candidate interval.
    if (!state.logical) {
      builder.setInsertionPoint(guard);
      Value entry =
          builder.create<pto::GetCtrlOp>(guard.getLoc()).getResult();
      state.logical = entry;
      if (!state.physical) {
        state.physical = entry;
      }
    }

    if (!(state.haveLastGuard && state.lastControlled == controlled &&
          state.lastRequired == required)) {
      // Different requirement or first guard in the interval: build the
      // active value from the shared logical entry and switch once.
      builder.setInsertionPoint(guard);
      Value active = buildActiveValue(guard.getLoc(), state.logical,
                                      controlled, required, builder);
      if (active != state.physical) {
        builder.create<pto::SetCtrlOp>(guard.getLoc(), active);
        state.physical = active;
        state.physicalMatchesLogical = false;
        state.diverged = true;
      }
    }
    state.haveLastGuard = true;
    state.lastControlled = controlled;
    state.lastRequired = required;

    // Unwrap the guard: its body executes under the installed state.
    Block &body = guard.getBody().front();
    for (Operation &bodyOp : llvm::make_early_inc_range(body)) {
      bodyOp.moveBefore(guard);
    }
    guard.erase();
  }

  void handleStateAccess(BlockScanState &state, Block &block,
                         StateAccessOpInterface access, Operation *op,
                         OpBuilder &builder) {
    if (access.getStateResource() != StateResource::Ctrl) {
      // Other state resources are boundaries for now.
      restoreBefore(state, block, op, builder);
      state.resetUnknown();
      return;
    }
    // Every explicit CTRL access is a real observation point: a pending
    // temporary active state must be restored first.
    restoreBefore(state, block, op, builder);
    switch (access.getStateAccessKind()) {
    case StateAccessKind::Query:
    case StateAccessKind::Write:
      // A query observes the logical state and a write redefines it; both
      // become the new logical/physical base.
      state.observe(access.getAccessedStateValue());
      break;
    case StateAccessKind::Consume:
    case StateAccessKind::Clobber:
      // A consumer outside a guard cannot reuse a pending active
      // configuration; restart with unknown values.
      state.resetUnknown();
      break;
    }
  }
};

static void hoistOneRoot(scf::ForOp root, const LoopCtrlSummary &s,
                         DenseSet<Operation *> &covered, OpBuilder &builder) {
  builder.setInsertionPoint(root);
  Value entry = builder.create<pto::GetCtrlOp>(root.getLoc()).getResult();
  Value active = buildActiveValue(root.getLoc(), entry,
                                  s.requirement.controlled,
                                  s.requirement.required, builder);
  builder.create<pto::SetCtrlOp>(root.getLoc(), active);
  for (pto::CtrlStateGuardOp guard : s.guards) {
    covered.insert(guard.getOperation());
    Block &body = guard.getBody().front();
    for (Operation &bodyOp : llvm::make_early_inc_range(body)) {
      bodyOp.moveBefore(guard);
    }
    guard.erase();
  }
  builder.setInsertionPointAfter(root);
  builder.create<pto::SetCtrlOp>(root.getLoc(), entry);
}

// Per-block materialization for guards not covered by hoisting.
static void materializeRemainingGuards(func::FuncOp func,
                                       OpBuilder &builder) {
  SmallVector<Block *, 16> workBlocks;
  func.walk([&](pto::CtrlStateGuardOp guard) {
    Block *block = guard->getBlock();
    if (!llvm::is_contained(workBlocks, block)) {
      workBlocks.push_back(block);
    }
  });

  BlockCtrlOptimizer optimizer;
  for (Block *block : workBlocks) {
    optimizer.process(block, builder);
  }
}

// Hoist one shared CTRL configuration to the entry of the outermost
// qualifying loop per guard subtree and unwrap all covered guards.
// Whether the subtree of `root` may hoist: it must be CTRL-clean, hold a
// single unique requirement, be statically proven to execute at least once,
// and every loop between `root` and each guard must be proven at least-once.
static bool
loopHoistEligible(scf::ForOp root, const LoopCtrlSummary &s,
                  DenseMap<Operation *, LoopCtrlSummary> &summaries) {
  if (!(s.ctrlClean && s.hasGuards && s.uniqueRequirement &&
        s.tripAtLeastOnce)) {
    return false;
  }
  return llvm::all_of(s.guards, [&](pto::CtrlStateGuardOp guard) {
    return guardChainProvenOnce(guard.getOperation(), root.getOperation(),
                                 summaries);
  });
}

static void hoistLoopConfigurations(
    func::FuncOp func,
    DenseMap<Operation *, LoopCtrlSummary> &summaries, OpBuilder &builder) {
  SmallVector<scf::ForOp, 8> preOrder;
  func.walk<WalkOrder::PreOrder>(
      [&](scf::ForOp forOp) { preOrder.push_back(forOp); });

  // Pointers of guards already covered by a hoisted configuration; used for
  // membership tests only, never dereferenced after the guard is erased.
  DenseSet<Operation *> covered;
  for (scf::ForOp root : preOrder) {
    auto it = summaries.find(root.getOperation());
    if (it == summaries.end() || !loopHoistEligible(root, it->second, summaries)) {
      continue;
    }
    // An earlier (outer) hoist may already cover this loop's guards.
    if (llvm::any_of(it->second.guards, [&](pto::CtrlStateGuardOp guard) {
          return covered.contains(guard.getOperation());
        })) {
      continue;
    }
    hoistOneRoot(root, it->second, covered, builder);
  }
}

struct VPTOOptimizeCtrlStatePass
    : public pto::impl::VPTOOptimizeCtrlStateBase<VPTOOptimizeCtrlStatePass> {
  void runOnOperation() override;
};

} // namespace

void VPTOOptimizeCtrlStatePass::runOnOperation() {
  func::FuncOp func = getOperation();

  bool hasGuards = func.walk([&](pto::CtrlStateGuardOp) {
                       return WalkResult::interrupt();
                     }).wasInterrupted();
  if (!hasGuards) {
    return;
  }

  OpBuilder builder(&getContext());

  // ---- Analysis stage: summarize loops inner-to-outer. ------------------
  DenseMap<Operation *, LoopCtrlSummary> summaries;
  func.walk([&](scf::ForOp forOp) {
    if (!summaries.count(forOp.getOperation())) {
      summaries[forOp.getOperation()] = summarizeLoop(forOp, summaries);
    }
    return WalkResult::advance();
  });

  // ---- Hoisting stage: outermost qualifying loop wins. ------------------
  hoistLoopConfigurations(func, summaries, builder);

  // ---- Per-block stage for guards not covered by hoisting. --------------
  materializeRemainingGuards(func, builder);

  // All guards must be gone; leftovers indicate an analysis gap.
  WalkResult leftover = func.walk([&](pto::CtrlStateGuardOp) {
    return WalkResult::interrupt();
  });
  if (leftover.wasInterrupted()) {
    func.emitError("vpto-optimize-ctrl-state failed to eliminate all "
                   "ctrl_state_guard ops");
    signalPassFailure();
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createVPTOOptimizeCtrlStatePass() {
  return std::make_unique<VPTOOptimizeCtrlStatePass>();
}

} // namespace pto
}
