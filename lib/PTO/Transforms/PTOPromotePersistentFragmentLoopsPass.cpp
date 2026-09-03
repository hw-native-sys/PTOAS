// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPromotePersistentFragmentLoopsPass.cpp --------------------------===//
//
// Promote loops that access persistent SIMT fragment buffers to forced full
// unrolling.
//
// Persistent fragment materialization (pto-materialize-simt-persistent-
// -fragment) requires the loops touching a persistent buffer to be fully
// unrolled, so that every access resolves to a stable resident slot.  This
// pass is the discovery half of that pipeline: it finds the loops and marks
// them for pto-unroll-loops; it does NOT do slot assignment, keep/resume
// generation, SIMT outlining, or LLVM metadata lowering.
//
// Discovery: every llvm.alloca carrying {pto.persistent} is an explicit
// entry point (no structural re-inference of persistence).  The pass walks
// the alloca's pointer use graph - getelementptr chains are followed,
// load/store (and any other direct consumer) are terminal accesses - and
// collects the enclosing scf.for loops of every related op.  A load's data
// result is deliberately NOT followed: materialization only requires every
// access to resolve to a stable resident slot, so a loop that merely
// consumes a loaded value is unrelated to the buffer:
//
//   - loops directly wrapping an access inside a SIMT section;
//   - every layer of multiply nested loops;
//   - EXCEPT loops whose body contains a pto.section.simt region AND whose
//     induction variable is not used by any persistent GEP inside the loop:
//     those are left alone (see sectionLoopSkipsPromotion below).
//     Fully unrolling such a loop clones the section once per iteration,
//     SIMT outlining emits one outlined entry function per clone, and those
//     single-callsite linkonce_odr clones get folded/inlined by BiSheng so
//     their `_simt_entry` ELF symbols go missing - VPTO fatobj emission
//     then fails in the VF_SIMT size patch.  Materialization only needs
//     the accesses inside the section to be static; a section-wrapping loop
//     whose induction variable never reaches a persistent GEP does not
//     affect that.  When the induction variable DOES select the fragment
//     slot, unrolling is the only way to statically resolve the access, so
//     the loop is promoted (each clone then reads its own resident slot).
//
// If any loop on an access's enclosing chain is statically zero-trip
// (ub <= lb), the access never executes and the whole chain is left
// unpromoted - there is no materialization precondition to enforce.
//
// Promotion rules for each collected loop:
//
//   | original state          | result                                |
//   |-------------------------|---------------------------------------|
//   | no unroll attribute     | pto.unroll = "full"                   |
//   | pto.unroll = "enable"   | replaced with pto.unroll = "full"     |
//   | pto.unroll = "full"     | unchanged                             |
//   | pto.unroll_factor = N   | hard error                            |
//   | malformed hint attr     | hard error (never silently repaired)  |
//
// Every promoted loop additionally gets the internal marker
// {pto.persistent_unroll}: pto-unroll-loops turns its usual
// drop-the-hint-with-a-remark fallback into a hard error for marked loops,
// because silently keeping a persistent loop as an ordinary loop would
// break the materialization precondition downstream.  The marker
// disappears together with the loop when it is unrolled.
//
// Fail-fast cases (hard errors, never a silent fallback):
//   - a fixed unroll factor on a persistent loop;
//   - a persistent access nested under scf.while (unsupported control flow);
//   - a statically known trip count above max-persistent-unroll-trip-count.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/LoopUnrollUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPROMOTEPERSISTENTFRAGMENTLOOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-promote-persistent-loops"

namespace {

/// Return true when the loop's bounds and step are all compile-time
/// constants and the loop body never executes (upper bound <= lower bound).
static bool isStaticallyZeroTrip(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  return lb && ub && step && *step > 0 && *ub <= *lb;
}

/// Return true when `loopVar` is reachable from `index` through pure
/// integer arithmetic definitions (the iv*stride+offset shapes a persistent
/// GEP index is built from).  Anything else (loads, calls, block arguments
/// of other regions) stops the walk, so a false negative can only leave an
/// unpromoted loop that materialization then rejects with its own
/// non-constant-index diagnostic - never a wrong promotion.
static bool indexDependsOnLoopVar(Value index, Value loopVar) {
  if (index == loopVar) {
    return true;
  }
  DenseSet<Value> visited;
  SmallVector<Value, 8> worklist{index};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }
    if (value == loopVar) {
      return true;
    }
    Operation *def = value.getDefiningOp();
    if (!def) {
      continue;
    }
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::RemSIOp, arith::IndexCastOp, arith::ExtSIOp,
             arith::TruncIOp>(def)) {
      continue;
    }
    for (Value operand : def->getOperands()) {
      worklist.push_back(operand);
    }
  }
  return false;
}

/// Return true when any persistent-fragment GEP inside `forOp`'s body has an
/// index that depends on `forOp`'s induction variable.  Unrolling such a
/// loop is the only way to statically resolve those accesses for
/// materialization, so the loop must be promoted even though unrolling it
/// clones the section it wraps.
static bool persistentGepDependsOnLoop(
    scf::ForOp forOp, const llvm::SmallSetVector<Operation *, 32> &relatedOps) {
  Value iv = forOp.getInductionVar();
  for (Operation *op : relatedOps) {
    auto gep = dyn_cast<LLVM::GEPOp>(op);
    if (!gep || !forOp->isAncestor(gep)) {
      continue;
    }
    for (auto index : gep.getIndices()) {
      // Constant indices never depend on the loop; only the Value ones can.
      if (!index.template is<Value>()) {
        continue;
      }
      if (indexDependsOnLoopVar(index.template get<Value>(), iv)) {
        return true;
      }
    }
  }
  return false;
}

/// Collect every op related to a persistent fragment buffer by walking each
/// persistent alloca's pointer use graph.  The alloca itself is included so
/// kernel-level loops wrapping both the allocation and the sections are
/// discovered too.  A SmallSetVector keeps the insertion (discovery) order
/// unconditionally, so the diagnostics emitted by the caller have a stable
/// order regardless of how many ops are related.
///
/// The worklist holds only pointer-producing ops (the alloca and GEPs
/// derived from it), so every `cur` has exactly one result: the pointer.
/// GEP users extend the pointer flow; every other user is a terminal
/// access - it is related (its enclosing loops are promoted) but its
/// results are not part of the pointer flow.  In particular a load's data
/// result is not followed, so loops that merely consume a loaded value are
/// never pulled in.
///
/// Returns false when the function has no persistent alloca at all.
static bool collectPersistentRelatedOps(
    func::FuncOp func, llvm::SmallSetVector<Operation *, 32> &relatedOps,
    SmallVector<LLVM::AllocaOp, 4> &persistentAllocas) {
  bool foundPersistent = false;
  func.walk([&](LLVM::AllocaOp allocaOp) {
    if (!allocaOp->hasAttr(pto::kPersistentAttrName)) {
      return;
    }
    foundPersistent = true;
    persistentAllocas.push_back(allocaOp);
    relatedOps.insert(allocaOp.getOperation());
    SmallVector<Operation *, 8> worklist{allocaOp.getOperation()};
    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      for (Operation *user : cur->getUsers()) {
        if (!relatedOps.insert(user)) {
          continue;
        }
        if (isa<LLVM::GEPOp>(user)) {
          worklist.push_back(user);
        }
      }
    }
  });
  return foundPersistent;
}

/// A loop whose body contains a pto.section.simt region is only promoted
/// when a persistent GEP inside it actually depends on the loop's induction
/// variable:
///
///  - no dependency (e.g. a tile loop that only moves data while the
///    fragment is indexed by lane constants, the RMSNorm shape): unrolling
///    would clone the whole section once per iteration, the SIMT outlining
///    emits one single-callsite linkonce_odr entry function per clone,
///    BiSheng folds a subset of them, the `_simt_entry` ELF symbols go
///    missing, and VPTO fatobj emission fails in the VF_SIMT size patch -
///    for zero benefit, because materialization only needs the in-section
///    accesses static, which this loop does not affect.
///
///  - dependency (fragment slot selected per iteration): unrolling is the
///    only way to statically resolve those accesses, so the loop must be
///    promoted; the dep shape is verified to materialize correctly (each
///    clone reads its own resident slot).
static bool sectionLoopSkipsPromotion(
    Operation *loopOp, const llvm::SmallSetVector<Operation *, 32> &relatedOps) {
  bool wrapsSection =
      loopOp->walk([](pto::SectionSimtOp) { return WalkResult::interrupt(); })
          .wasInterrupted();
  return wrapsSection &&
         !persistentGepDependsOnLoop(cast<scf::ForOp>(loopOp), relatedOps);
}

/// Collect every enclosing scf.for of every related op (all nesting layers,
/// including kernel-level loops wrapping pto.section.simt).  A persistent
/// access nested under scf.while is unsupported control flow: the loop
/// structure cannot carry a full-unroll hint, and silently skipping it
/// would break materialization downstream.  Diagnostics are collected
/// across the whole function before the pass fails once: the function pass
/// adaptor may stop scheduling functions after the first failure, so
/// per-function completeness keeps the emitted set deterministic under
/// parallel scheduling.
///
/// relatedOps iterates in discovery order (see
/// collectPersistentRelatedOps), and repeated diagnostics for the same
/// scf.while are deduplicated, so the emitted set is deterministic.
///
/// Returns true when a fail-fast diagnostic was emitted.
static bool collectLoopsToPromote(
    const llvm::SmallSetVector<Operation *, 32> &relatedOps,
    SmallVector<Operation *, 8> &loopOps) {
  bool hadError = false;
  llvm::SmallPtrSet<Operation *, 4> seenWhileOps;
  llvm::SmallPtrSet<Operation *, 8> seenLoopOps;
  for (Operation *op : relatedOps) {
    if (auto whileOp = op->getParentOfType<scf::WhileOp>()) {
      if (seenWhileOps.insert(whileOp.getOperation()).second) {
        whileOp.emitError()
            << "persistent fragment buffer is accessed inside scf.while, "
               "which cannot be fully unrolled; persistent fragment loops "
               "must use scf.for";
        hadError = true;
      }
    }
    // Collect the whole enclosing-loop chain first: if any loop on the
    // chain is statically zero-trip, the access never executes, so no loop
    // on that chain needs promotion (marking e.g. a dynamic outer loop
    // would only produce a spurious "no constant trip count" hard error
    // downstream).
    Operation *cur = op;
    SmallVector<Operation *, 4> chain;
    bool chainIsDead = false;
    while (auto forOp = cur->getParentOfType<scf::ForOp>()) {
      chain.push_back(forOp.getOperation());
      chainIsDead = chainIsDead || isStaticallyZeroTrip(forOp);
      cur = forOp.getOperation();
    }
    if (chainIsDead) {
      continue;
    }
    for (Operation *loopOp : chain) {
      if (sectionLoopSkipsPromotion(loopOp, relatedOps)) {
        continue;
      }
      if (seenLoopOps.insert(loopOp).second) {
        loopOps.push_back(loopOp);
      }
    }
  }
  return hadError;
}

struct PTOPromotePersistentFragmentLoops
    : public pto::impl::PTOPromotePersistentFragmentLoopsBase<
          PTOPromotePersistentFragmentLoops> {
  using pto::impl::PTOPromotePersistentFragmentLoopsBase<
      PTOPromotePersistentFragmentLoops>::PTOPromotePersistentFragmentLoopsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::SmallSetVector<Operation *, 32> relatedOps;
    SmallVector<LLVM::AllocaOp, 4> persistentAllocas;
    if (!collectPersistentRelatedOps(func, relatedOps, persistentAllocas)) {
      return;
    }
    SmallVector<Operation *, 8> loopOps;
    bool hadError = collectLoopsToPromote(relatedOps, loopOps);
    for (Operation *loopOp : loopOps) {
      // No short-circuit: every collected loop must be diagnosed/promoted
      // even when an earlier loop already failed.
      if (!promoteOneLoop(cast<scf::ForOp>(loopOp), func, persistentAllocas)) {
        hadError = true;
      }
    }
    if (hadError) {
      signalPassFailure();
    }
  }

private:
  /// Promote one collected loop to forced full unrolling.  Returns false
  /// when a fail-fast diagnostic was emitted for this loop.
  bool promoteOneLoop(scf::ForOp forOp, func::FuncOp func,
                      ArrayRef<LLVM::AllocaOp> persistentAllocas) {
    // A malformed hint must stay a hard error: promotion would otherwise
    // silently overwrite it with "full" before pto-unroll-loops ever gets
    // to validate it.
    bool carriesHint = forOp->hasAttr(pto::kUnrollAttrName) ||
                       forOp->hasAttr(pto::kUnrollFactorAttrName);
    if (carriesHint && failed(pto::validateLoopUnrollHint(forOp))) {
      return false;
    }
    if (forOp->hasAttr(pto::kUnrollFactorAttrName)) {
      forOp.emitError()
          << "persistent fragment loop requires full unroll; a fixed '"
          << pto::kUnrollFactorAttrName << "' is not supported";
      return false;
    }

    // Guardrail: bound the code expansion of the forced unroll.  A dynamic
    // trip count cannot be checked here; pto-unroll-loops reports a
    // persistent-specific error for it via the marker below.
    int64_t tripCountCap = maxPersistentUnrollTripCount.getValue();
    if (tripCountCap >= 0) {
      if (std::optional<int64_t> tripCount = pto::getStaticTripCount(forOp);
          tripCount && *tripCount > tripCountCap) {
        auto diag = forOp.emitError()
                    << "persistent fragment loop in '" << func.getSymName()
                    << "' has trip count " << *tripCount
                    << ", which exceeds max-persistent-unroll-trip-count="
                    << tripCountCap;
        for (LLVM::AllocaOp allocaOp : persistentAllocas) {
          diag.attachNote(allocaOp.getLoc())
              << "persistent fragment allocation";
        }
        return false;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "PTOPromotePersistentLoops: promoting scf.for "
                            << "at " << forOp.getLoc()
                            << " to full unroll\n");

    // "full" is idempotent; "enable" and no-hint are both overridden.
    forOp->setAttr(pto::kUnrollAttrName,
                   StringAttr::get(forOp->getContext(), pto::kUnrollFullValue));
    // The marker makes pto-unroll-loops fail (instead of dropping the hint)
    // when this loop cannot be unrolled natively.  It is added on
    // already-"full" loops too so their failure path is likewise a hard
    // error, per the fail-fast contract.
    forOp->setAttr(pto::kPersistentUnrollMarkerAttrName,
                   UnitAttr::get(forOp->getContext()));
    return true;
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOPromotePersistentFragmentLoopsPass() {
  return std::make_unique<PTOPromotePersistentFragmentLoops>();
}
