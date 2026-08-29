// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOGuardedLICM.cpp -----------------------------------------------===//
//
// Hoist loop-invariant scalar and address expressions out of guarded scf.if
// regions nested inside scf.for loops.
//
// The generic MLIR loop-invariant code motion pass only enqueues the
// top-level operations of the scf.for body.  For a guarded access pattern
//
//   scf.for %iv = ...
//     %in_bounds = ...              // depends on the IV
//     scf.if %in_bounds {
//       %base = ...                 // loop-invariant base address chain
//       %dynamic = ... %iv ...      // IV-dependent offset
//       pto.store ...
//     }
//
// the invariant chain inside the guard is therefore never seen by LICM, and
// the whole scf.if cannot be hoisted either: its condition depends on the IV
// and its region contains side-effecting memory operations.  This pass
// extracts only the safe, speculatable scalar/address subexpressions of the
// guard and moves them in dependency (topological) order to just before the
// loop, leaving IV-dependent arithmetic, side-effecting operations and
// vector/container computations in place.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOGUARDEDLICM
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

// Address-like scalar result types we are willing to hoist: plain integers,
// index values and pto.ptr addresses.  Vector registers, masks, tile
// objects and other container types are deliberately excluded so that the
// hoisted code cannot grow vector register pressure outside the guarded
// region.
static bool isHoistableScalarType(Type type) {
  return isa<IntegerType, IndexType, pto::PtrType>(type);
}

// The whole integer division/remainder family is never hoisted, including the
// "total" variants (arith.floordivsi/ceildivsi/ceildivui) that do not even
// carry ConditionallySpeculatable: dividing by zero or INT_MIN / -1 is
// undefined behavior, so evaluating them before the guard could introduce UB
// that the original program never executed.  divf/remf are not hoisted
// either: a float div/rem can raise IEEE exceptions or trap, and the generic
// pure check does not protect the short-circuiting case.
static bool isDivOrRem(Operation *op) {
  return isa<arith::DivSIOp, arith::DivUIOp, arith::DivFOp, arith::RemSIOp,
             arith::RemUIOp, arith::RemFOp, arith::CeilDivSIOp,
             arith::CeilDivUIOp, arith::FloorDivSIOp>(op);
}

// Explicit allow-list of deterministic, referentially transparent scalar
// operations.  The generic isPure() is intentionally not the gate here:
// several PTO operations are marked pure but are not referentially
// transparent, e.g. pto.get_clock32/64 (time sampling), pto.get_vms4_sr
// (VMS4 status register), and pto.vote_*/pto.shuffle_*/pto.redux_*
// (active-lane collectives).  Hoisting any of them changes the value the
// guard observes, so only the operations listed below may move out of a
// guard; everything else stays in place.  Both the signed and the unsigned
// index<->integer casts (arith.index_cast / arith.index_castui) are allowed:
// VPTO soft post-update emits the unsigned form for offset/block computations,
// and excluding it would leave the whole guarded base chain behind.
static bool isHoistableScalarOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
             arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
             arith::ShRUIOp, arith::CmpIOp, arith::MaxSIOp, arith::MaxUIOp,
             arith::MinSIOp, arith::MinUIOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::TruncIOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::SelectOp, arith::ConstantOp, pto::AddPtrOp,
             pto::CastPtrOp>(op);
}

// An operation is a hoist candidate only when it is on the deterministic
// allow-list, still side-effect free (defensive double check), region-free,
// produces only address-like scalars, and is not a terminator.
static bool isHoistCandidate(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  if (!op->getRegions().empty()) {
    return false;
  }
  if (!isHoistableScalarOp(op)) {
    return false;
  }
  if (!isPure(op)) {
    return false;
  }
  if (isDivOrRem(op)) {
    return false;
  }
  for (Type resultType : op->getResultTypes()) {
    if (!isHoistableScalarType(resultType)) {
      return false;
    }
  }
  return true;
}

// Whether a value is defined outside the loop, i.e. available before the loop
// starts.  Loop induction variables, iter_args of the loop body and every
// value produced inside the loop fail this check.
static bool isDefinedOutsideLoop(Value value, Operation *loopOp) {
  Operation *holder = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    holder = blockArg.getOwner()->getParentOp();
  } else {
    holder = value.getDefiningOp();
  }
  if (!holder) {
    return true;
  }
  Operation *ancestor = holder;
  while (ancestor) {
    if (ancestor == loopOp) {
      return false;
    }
    ancestor = ancestor->getParentOp();
  }
  return true;
}

// Number of enclosing scf.for loops; used to sort loops innermost-first.
static int loopNestingDepth(scf::ForOp forOp) {
  int depth = 0;
  Operation *op = forOp.getOperation();
  while (op != nullptr) {
    if (isa<scf::ForOp>(op)) {
      ++depth;
    }
    op = op->getParentOp();
  }
  return depth;
}

// Extract every hoistable invariant subexpression that lives inside an scf.if
// region nested in the loop (and not inside another scf.for), moving the
// expressions in dependency order to just before the loop.
static void hoistInvariantsFromGuards(scf::ForOp forOp) {
  // Collect candidates.  walk() covers every nested scf.if region; the
  // innermost scf.for ancestor test keeps nested loops in charge of their own
  // decisions (they are processed before this loop because loops are visited
  // innermost-first).
  SmallVector<Operation *> candidates;
  forOp->walk([&](Operation *op) {
    if (op == forOp.getOperation()) {
      return;
    }
    scf::IfOp enclosingIf = op->getParentOfType<scf::IfOp>();
    if (enclosingIf == nullptr) {
      return; // not inside a guarded region
    }
    scf::ForOp enclosingFor = op->getParentOfType<scf::ForOp>();
    if (enclosingFor != forOp) {
      return; // inside a nested loop; that loop owns the decision
    }
    if (isHoistCandidate(op)) {
      candidates.push_back(op);
    }
  });

  if (candidates.empty()) {
    return;
  }

  // Iterate to a fixed point: an op joins the hoist set only when all of its
  // operands are either defined outside the loop or produced by already-hoisted
  // ops.  The join order is therefore a valid topological order.
  DenseSet<Value> available;
  DenseSet<Operation *> hoistedSet;
  SmallVector<Operation *> hoisted;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : candidates) {
      if (hoistedSet.count(op)) {
        continue;
      }
      bool allOperandsAvailable = true;
      for (Value operand : op->getOperands()) {
        bool operandAvailable = available.count(operand) != 0;
        bool operandFromOutside = isDefinedOutsideLoop(operand, forOp);
        if (!operandAvailable && !operandFromOutside) {
          allOperandsAvailable = false;
          break;
        }
      }
      if (!allOperandsAvailable) {
        continue;
      }
      hoisted.push_back(op);
      hoistedSet.insert(op);
      for (Value result : op->getResults()) {
        available.insert(result);
      }
      changed = true;
    }
  }

  for (Operation *op : hoisted) {
    op->moveBefore(forOp);
  }
}

// The rewrite is restricted to the A5 VPTO module; keep other targets (A3,
// EmitC) bit-for-bit identical until they have their own performance
// validation.
static bool isA5VPTOModule(func::FuncOp func) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  while (module != nullptr) {
    auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
    if (arch) {
      return arch.getValue() == "a5";
    }
    module = module->getParentOfType<ModuleOp>();
  }
  return false;
}

struct VPTOGuardedLICM
    : public pto::impl::VPTOGuardedLICMBase<VPTOGuardedLICM> {
  using pto::impl::VPTOGuardedLICMBase<VPTOGuardedLICM>::VPTOGuardedLICMBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (!isA5VPTOModule(func)) {
      return;
    }

    // Process loops innermost-first so invariants can climb out through nested
    // guards and nested loops: after an inner loop hoists a chain to its own
    // preheader (still inside the guard of an outer loop), the outer loop pass
    // re-collects the same chain and hoists it once more.
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    llvm::sort(loops, [](scf::ForOp lhs, scf::ForOp rhs) {
      return loopNestingDepth(lhs) > loopNestingDepth(rhs);
    });

    for (scf::ForOp forOp : loops) {
      hoistInvariantsFromGuards(forOp);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOGuardedLICMPass() {
  return std::make_unique<VPTOGuardedLICM>();
}
