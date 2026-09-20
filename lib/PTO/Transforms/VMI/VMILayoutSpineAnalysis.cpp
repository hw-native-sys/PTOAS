// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSpineAnalysis.cpp - VMI direction-spine analysis ----------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "PTO/Transforms/VMILayoutSpineAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kDirectionSpineLegCount = 4;
constexpr unsigned kDirectionSpineMaxHops = 8;
constexpr unsigned kDirectionSpineSearchBudget = 512;

/// Inline capacity and chunk threshold of the narrow-side class scan: a class a
/// consumer sees is a handful of values, and the "narrow" side of a
/// width-changing cast is the one below a full 32-bit carrier word.
constexpr unsigned kClassScanInlineCapacity = 32;
constexpr unsigned kNarrowSideMaxBits = 32;

struct DirectionSpineLeg {
  Operation *op = nullptr;
  Value source;
  Value result;
  Type sourceElementType;
  Type resultElementType;
  bool widening = false;
};

static bool getDirectionSpineLeg(Operation *op, DirectionSpineLeg &leg) {
  Value source;
  Value result;
  if (auto extf = dyn_cast<VMIExtFOp>(op)) {
    source = extf.getSource();
    result = extf.getResult();
  } else if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
    source = extsi.getSource();
    result = extsi.getResult();
  } else if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
    source = extui.getSource();
    result = extui.getResult();
  } else if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
    source = truncf.getSource();
    result = truncf.getResult();
  } else if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
    source = trunci.getSource();
    result = trunci.getResult();
  } else {
    return false;
  }

  auto sourceType = dyn_cast<VMIVRegType>(source.getType());
  auto resultType = dyn_cast<VMIVRegType>(result.getType());
  if (!sourceType || !resultType) {
    return false;
  }
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0 || sourceBits == resultBits) {
    return false;
  }

  leg.op = op;
  leg.source = source;
  leg.result = result;
  leg.sourceElementType = sourceType.getElementType();
  leg.resultElementType = resultType.getElementType();
  leg.widening = resultBits > sourceBits;
  return true;
}

// Whether the narrow intermediate produced by \p downLeg is handed straight to
// the closing widening leg \p upLeg of the same window and to nothing else.
// This is the sufficiency condition for the deinterleaved family (see the
// recognition comment above): the caller only needs the use-def list of one
// value, and it is decided before any seed exists.
static bool isPureNarrowHandoff(const DirectionSpineLeg &downLeg,
                                const DirectionSpineLeg &upLeg) {
  if (downLeg.result.use_empty()) {
    return false;
  }
  for (OpOperand &use : downLeg.result.getUses()) {
    if (use.getOwner() != upLeg.op) {
      return false;
    }
  }
  return true;
}

// Extend a partially built leg window forward from \p current.  Legs alternate
// up/down starting with up, and between two legs only layout-transparent ops
// are crossed (bounded by kDirectionSpineMaxHops per leg and by \p budget per
// search).  A complete window is recorded only when it closes - the fourth leg
// ends on the element type the first leg started from - and when its narrow
// handoff is pure.
static void extendDirectionSpineWindow(
    Value current, SmallVectorImpl<DirectionSpineLeg> &window,
    llvm::SmallPtrSetImpl<Operation *> &spineLegs, unsigned hops,
    unsigned &budget);

// A full window closes the round trip when its ends carry the same element type
// and the narrow handoff in the middle is pure; a window that fails either test
// is not recognised, so the legs keep the pre-existing behaviour.
static void closeDirectionSpineWindow(
    SmallVectorImpl<DirectionSpineLeg> &window,
    llvm::SmallPtrSetImpl<Operation *> &spineLegs) {
  if (window.front().sourceElementType != window.back().resultElementType ||
      !isPureNarrowHandoff(window[1], window[2])) {
    return;
  }
  for (const DirectionSpineLeg &leg : window) {
    spineLegs.insert(leg.op);
  }
}

// Cross a layout class edge only: the elementwise family and an *equal-width*
// bitcast.  A width-changing bitcast is a boundary (see isVMIClassTransparentOp)
// and is never crossed, so the class walk and the solver agree on what "the same
// layout on this value" means.  While crossing, the value has to stay in that
// one class, and a layout is a property of the physical carrier: two VMI values
// are in the same class exactly when their *storage element width* matches.
// Element-type identity - the stricter form this guard used to test - states
// the same thing for the elementwise family (which never changes the element
// type) but it also stopped
// at an equal-width bitcast, whose entire purpose is to keep the physical
// carrier while reinterpreting the element type.  That made the guard contradict
// isVMIClassTransparentOp, so the class edge was declared but could never be
// used.  Only this guard changes here: the spine's own bookkeeping is read from
// the legs, not from this hop - leg direction comes from getDirectionSpineLeg,
// narrow-handoff purity from isPureNarrowHandoff, and the window closure from
// the exact element type of the first and last leg - so none of them are
// relaxed.
static void extendAcrossClassEdge(
    Operation *user, unsigned currentBits, unsigned hops,
    SmallVectorImpl<DirectionSpineLeg> &window,
    llvm::SmallPtrSetImpl<Operation *> &spineLegs, unsigned &budget) {
  for (Value result : user->getResults()) {
    auto resultType = dyn_cast<VMIVRegType>(result.getType());
    if (!resultType) {
      continue;
    }
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    if (currentBits == 0 || resultBits != currentBits) {
      continue;
    }
    extendDirectionSpineWindow(result, window, spineLegs, hops + 1, budget);
  }
}

static void extendDirectionSpineWindow(
    Value current, SmallVectorImpl<DirectionSpineLeg> &window,
    llvm::SmallPtrSetImpl<Operation *> &spineLegs, unsigned hops,
    unsigned &budget) {
  if (budget == 0) {
    return;
  }
  if (window.size() == kDirectionSpineLegCount) {
    closeDirectionSpineWindow(window, spineLegs);
    return;
  }

  auto currentType = dyn_cast<VMIVRegType>(current.getType());
  if (!currentType) {
    return;
  }
  const bool expectWidening = window.size() % 2 == 0;
  for (OpOperand &use : current.getUses()) {
    if (budget == 0) {
      return;
    }
    --budget;
    Operation *user = use.getOwner();

    DirectionSpineLeg next;
    if (getDirectionSpineLeg(user, next) && next.source == current &&
        next.widening == expectWidening) {
      window.push_back(next);
      extendDirectionSpineWindow(next.result, window, spineLegs, /*hops=*/0,
                                 budget);
      window.pop_back();
      continue;
    }

    if (hops < kDirectionSpineMaxHops && isVMIClassTransparentOp(user)) {
      unsigned currentBits =
          pto::getPTOStorageElemBitWidth(currentType.getElementType());
      extendAcrossClassEdge(user, currentBits, hops, window, spineLegs, budget);
    }
  }
}

// Whether \p op is a widening cast, i.e. the kind of op that closes a
// narrow->wide handoff.
static bool isSpineWideningCast(Operation *op) {
  return isa<VMIExtFOp, VMIExtSIOp, VMIExtUIOp>(op);
}

//===----------------------------------------------------------------------===//
// Narrow-side compute classification.
//===----------------------------------------------------------------------===//

// Ops that unite their operands and results into one layout class live in
// VMILayoutPropagation (mlir::pto::isVMIClassTransparentOp): the same-layout set
// covers the elementwise family and the equal-width bitcast is the one class
// edge.  A width-changing bitcast is a boundary, not an edge.

// Elementwise work that the narrow side would repeat once per physical part.
// The two same-layout conversions are excluded: a conversion is a boundary,
// not repeated elementwise work.
static bool isVMINarrowSideComputeOp(Operation *op) {
  return isVMISameLayoutOp(op) && !isa<VMIFPToSIOp, VMISIToFPOp>(op);
}

// Bounded breadth-first scan of one narrow value's layout equivalence class,
// looking for elementwise compute.  Closing the class costs one hop per visited
// value, so the worklist is bounded by kDirectionSpineSearchBudget.
class NarrowSideComputeScan {
public:
  bool run(Value narrow) {
    enqueue(narrow);
    while (budget > 0 && !worklist.empty()) {
      Value current = worklist.pop_back_val();
      --budget;
      bool found = scanProducer(current) || scanUsers(current);
      if (found) {
        return true;
      }
    }
    return false;
  }

private:
  void enqueue(Value value) {
    if (!value || !isa<VMIVRegType>(value.getType())) {
      return;
    }
    if (visited.insert(value).second) {
      worklist.push_back(value);
    }
  }

  // Close loop-carried values on both ends: a region result reads the matching
  // yield operand and init value, a region iter_arg the same pair.  Without
  // this the amax accumulator - a loop-carried narrow value whose compute sits
  // inside the body - is missed entirely.
  void closeRegionCarried(scf::ForOp forOp, unsigned index) {
    if (index >= forOp.getInitArgs().size()) {
      return;
    }
    enqueue(forOp.getInitArgs()[index]);
    if (auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
      if (index < yield.getNumOperands()) {
        enqueue(yield.getOperand(index));
      }
    }
  }

  // Producer side: the defining op is either the compute itself, a region that
  // carries the value around the loop, or a transparent hop of the class.
  bool scanProducer(Value current) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return scanCarriedBlockArgument(current);
    }
    if (isVMINarrowSideComputeOp(defOp)) {
      return true;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      if (auto result = dyn_cast<OpResult>(current)) {
        closeRegionCarried(forOp, result.getResultNumber());
      }
      return false;
    }
    if (isVMIClassTransparentOp(defOp)) {
      for (Value operand : defOp->getOperands()) {
        enqueue(operand);
      }
    }
    return false;
  }

  bool scanCarriedBlockArgument(Value current) {
    auto blockArg = dyn_cast<BlockArgument>(current);
    if (!blockArg) {
      return false;
    }
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp) {
      return false;
    }
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber >= forOp.getNumInductionVars()) {
      closeRegionCarried(forOp, argNumber - forOp.getNumInductionVars());
    }
    return false;
  }

  // User side: every use is another member of the class, so it either is the
  // compute, closes the loop over the value, or hands the value on.
  bool scanUsers(Value current) {
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (isVMINarrowSideComputeOp(user)) {
        return true;
      }
      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp())) {
          closeRegionCarried(forOp, use.getOperandNumber());
        }
        continue;
      }
      if (isVMIClassTransparentOp(user)) {
        for (Value result : user->getResults()) {
          enqueue(result);
        }
      }
    }
    return false;
  }

  unsigned budget = kDirectionSpineSearchBudget;
  llvm::SmallPtrSet<Value, kClassScanInlineCapacity> visited;
  SmallVector<Value, kClassScanInlineCapacity> worklist;
};

// The narrow side of a width-changing cast, when it is a sub-word vector that
// would pay the lane-stride carrier inflation once per physical part.
static bool getNarrowSideOfCast(Operation *op, Value &narrow) {
  DirectionSpineLeg leg;
  if (!getDirectionSpineLeg(op, leg)) {
    return false;
  }
  // The narrow side is the lower-bit-width end.
  narrow = leg.widening ? leg.source : leg.result;
  auto narrowType = dyn_cast<VMIVRegType>(narrow.getType());
  if (!narrowType) {
    return false;
  }
  unsigned narrowBits =
      pto::getPTOStorageElemBitWidth(narrowType.getElementType());
  return narrowBits != 0 && narrowBits < kNarrowSideMaxBits;
}

static bool narrowSideCarriesCompute(Value narrow) {
  NarrowSideComputeScan scan;
  return scan.run(narrow);
}
} // namespace

// Record every cast op that is a leg of a closed nested round trip.
void mlir::pto::collectDirectionSpineLegs(
    ModuleOp module, llvm::SmallPtrSetImpl<Operation *> &spineLegs) {
  SmallVector<DirectionSpineLeg, kDirectionSpineLegCount> window;
  module.walk([&](Operation *op) {
    DirectionSpineLeg first;
    if (!getDirectionSpineLeg(op, first) || !first.widening) {
      return;
    }
    unsigned budget = kDirectionSpineSearchBudget;
    window.clear();
    window.push_back(first);
    extendDirectionSpineWindow(first.result, window, spineLegs, /*hops=*/0,
                               budget);
  });
}

// Narrow->wide handoff inside a matched direction spine.
// The composite 32<->16 / 32<->8 spine rows keep the narrow value split over
// the four physical parts of its wide side so that the narrowing and the closing
// widening are per-chunk one-to-one.  That is only safe while the narrow value
// has no other consumer: an elementwise op, a store, a broadcast or any layout
// request for a packed/contiguous narrow value would disagree with the
// composite form and force an ensure_layout (or a residual op), which is
// strictly worse than the assembly it was meant to remove.  Requiring every use
// of the narrowing leg's result to be a widening cast that is itself a leg of
// the same spine is therefore a cheap, local sufficiency condition.
// The set is computed from the IR alone, before any seed exists, and is empty
// whenever the direction-spine recognition is off or did not match.
void mlir::pto::collectSpineScopedCasts(
    const llvm::SmallPtrSetImpl<Operation *> &spineLegs,
    llvm::SmallPtrSetImpl<Operation *> &scopedCasts) {
  for (Operation *op : spineLegs) {
    if (!isa<VMITruncFOp, VMITruncIOp>(op) || op->getNumResults() != 1) {
      continue;
    }
    Value narrow = op->getResult(0);
    if (narrow.use_empty()) {
      continue;
    }
    bool handoff = true;
    for (OpOperand &use : narrow.getUses()) {
      Operation *user = use.getOwner();
      if (!isSpineWideningCast(user) || !spineLegs.contains(user)) {
        handoff = false;
        break;
      }
    }
    if (!handoff) {
      continue;
    }
    scopedCasts.insert(op);
    for (OpOperand &use : narrow.getUses()) {
      scopedCasts.insert(use.getOwner());
    }
  }
}

// Decide, for every width-changing cast, whether its narrow side carries
// elementwise compute.  Pure IR analysis, run before any seed exists.
void mlir::pto::collectNarrowSideCompute(
    ModuleOp module, llvm::SmallPtrSetImpl<Operation *> &narrowSideCompute) {
  module.walk([&](Operation *op) {
    Value narrow;
    bool classified = getNarrowSideOfCast(op, narrow);
    if (classified && narrowSideCarriesCompute(narrow)) {
      narrowSideCompute.insert(op);
    }
  });
}
