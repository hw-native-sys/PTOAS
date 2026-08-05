// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIPredicateFold.cpp - Fold statically proven VMI predicates -------===//
//
// Constant-proves lane ranges for index vectors built from vci / vadds /
// vbrc (and simple affine scf.for bases), folds all-true / all-false vcmp
// results into identity / constant vsel, then DCEs dead defs. Primary
// consumer: expert-pad masking when num_experts is a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIMaskUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIPREDICATEFOLD
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

//===----------------------------------------------------------------------===//
// Scalar / vector range lattices
//===----------------------------------------------------------------------===//

struct IntRange {
  int64_t lo = 0;
  int64_t hi = 0; // inclusive

  static IntRange splat(int64_t c) { return {c, c}; }
};

enum class MaskLattice { Unknown, AllTrue, AllFalse };

static std::optional<int64_t> matchConstantInt(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

/// Bound an integer SSA value over known constant / affine forms.
/// Recognizes: Imm, addi/subi with Imm, muli(iv|Imm, Imm), scf.for IV with
/// constant lb/ub/step (inclusive iteration set).
static std::optional<IntRange> matchAffineIntRange(Value v) {
  if (auto c = matchConstantInt(v))
    return IntRange::splat(*c);

  // Peel casts that preserve integer magnitude (index ↔ i32/i64).
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::IndexCastUIOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::ExtSIOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::ExtUIOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::TruncIOp>())
    return matchAffineIntRange(cast.getIn());

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto lhs = matchAffineIntRange(add.getLhs());
    auto rhs = matchAffineIntRange(add.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t lo, hi;
    if (llvm::AddOverflow(lhs->lo, rhs->lo, lo) ||
        llvm::AddOverflow(lhs->hi, rhs->hi, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  }

  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    auto lhs = matchAffineIntRange(sub.getLhs());
    auto rhs = matchAffineIntRange(sub.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t lo, hi;
    // [a,b] - [c,d] = [a-d, b-c]
    if (llvm::SubOverflow(lhs->lo, rhs->hi, lo) ||
        llvm::SubOverflow(lhs->hi, rhs->lo, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  }

  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto lhsC = matchConstantInt(mul.getLhs());
    auto rhsC = matchConstantInt(mul.getRhs());
    if (lhsC && rhsC) {
      int64_t prod;
      if (llvm::MulOverflow(*lhsC, *rhsC, prod))
        return std::nullopt;
      return IntRange::splat(prod);
    }

    Value dyn = lhsC ? mul.getRhs() : mul.getLhs();
    auto factorOpt = lhsC ? lhsC : rhsC;
    if (!factorOpt)
      return std::nullopt;
    int64_t factor = *factorOpt;
    auto dynR = matchAffineIntRange(dyn);
    if (!dynR)
      return std::nullopt;
    int64_t a, b;
    if (llvm::MulOverflow(dynR->lo, factor, a) ||
        llvm::MulOverflow(dynR->hi, factor, b))
      return std::nullopt;
    return IntRange{std::min(a, b), std::max(a, b)};
  }

  // scf.for induction variable with constant bounds.
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg != forOp.getInductionVar())
        return std::nullopt;
      auto lb = matchConstantInt(forOp.getLowerBound());
      auto ub = matchConstantInt(forOp.getUpperBound());
      auto step = matchConstantInt(forOp.getStep());
      if (!lb || !ub || !step || *step <= 0 || *lb >= *ub)
        return std::nullopt;
      // Last iterate: lb + n*step < ub.
      int64_t last = *lb + ((*ub - 1 - *lb) / *step) * *step;
      return IntRange{*lb, last};
    }
  }

  return std::nullopt;
}

/// Index vector lattice: every lane is in [lo, hi] (inclusive).
static std::optional<IntRange> matchVectorLaneRange(Value v) {
  auto vty = dyn_cast<VMIVRegType>(v.getType());
  if (!vty)
    return std::nullopt;
  int64_t vl = vty.getElementCount();

  // vbrc(C) / broadcast(C) → splat
  if (auto brc = v.getDefiningOp<VMIVbrcOp>()) {
    if (auto c = matchConstantInt(brc.getValue()))
      return IntRange::splat(*c);
    return std::nullopt;
  }
  if (auto brc = v.getDefiningOp<VMIBroadcastOp>()) {
    if (auto c = matchConstantInt(brc.getValue()))
      return IntRange::splat(*c);
    return std::nullopt;
  }

  // vci(base) / iota(base): continuous → [base, base+VL-1]
  // with {group=G}: group size = VL/G → [base, base+GSize-1]
  auto matchIotaLike = [&](Value base, std::optional<int64_t> group,
                           StringRef order) -> std::optional<IntRange> {
    if (!order.empty() && order != "ASC")
      return std::nullopt; // DESC not handled
    auto baseR = matchAffineIntRange(base);
    if (!baseR)
      return std::nullopt;
    // For a concrete or affine-bounded base, index covers
    // [baseLo, baseHi + span - 1].
    int64_t span = vl;
    if (group && *group > 0) {
      if (vl % *group != 0)
        return std::nullopt;
      span = vl / *group;
    }
    int64_t lo = baseR->lo;
    int64_t hi;
    if (llvm::AddOverflow(baseR->hi, span - 1, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  };

  if (auto vci = v.getDefiningOp<VMIVciOp>()) {
    std::optional<int64_t> group;
    if (auto g = vci->getAttrOfType<IntegerAttr>("group"))
      group = g.getInt();
    StringRef order = vci.getOrder() ? *vci.getOrder() : StringRef("ASC");
    return matchIotaLike(vci.getBase(), group, order);
  }
  if (auto iota = v.getDefiningOp<VMIIotaOp>()) {
    std::optional<int64_t> group;
    if (auto g = iota->getAttrOfType<IntegerAttr>("group"))
      group = g.getInt();
    StringRef order = iota.getOrder() ? *iota.getOrder() : StringRef("ASC");
    return matchIotaLike(iota.getBase(), group, order);
  }

  // vadds(src, scalar, mask): if seed all-active (or unused merge), shift range
  if (auto vadds = v.getDefiningOp<VMIAddSOp>()) {
    if (!isAllActiveSeed(vadds.getMask()))
      return std::nullopt;
    auto srcR = matchVectorLaneRange(vadds.getSrc());
    auto sc = matchConstantInt(vadds.getScalar());
    if (!srcR || !sc)
      return std::nullopt;
    int64_t lo, hi;
    if (llvm::AddOverflow(srcR->lo, *sc, lo) ||
        llvm::AddOverflow(srcR->hi, *sc, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  }

  // vadd(v, vbrc(C)) / similar not required for topk; skip.
  return std::nullopt;
}

static MaskLattice classifyCompare(StringRef cmp, const IntRange &lhs,
                                   const IntRange &rhs) {
  if (cmp == "lt" || cmp == "olt") {
    if (lhs.hi < rhs.lo)
      return MaskLattice::AllTrue;
    if (lhs.lo >= rhs.hi)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "le" || cmp == "ole") {
    if (lhs.hi <= rhs.lo)
      return MaskLattice::AllTrue;
    if (lhs.lo > rhs.hi)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "gt" || cmp == "ogt") {
    if (lhs.lo > rhs.hi)
      return MaskLattice::AllTrue;
    if (lhs.hi <= rhs.lo)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "ge" || cmp == "oge") {
    if (lhs.lo >= rhs.hi)
      return MaskLattice::AllTrue;
    if (lhs.hi < rhs.lo)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "eq" || cmp == "oeq") {
    // Only when both sides are the same splat constant.
    if (lhs.lo == lhs.hi && rhs.lo == rhs.hi && lhs.lo == rhs.lo)
      return MaskLattice::AllTrue;
    if (lhs.hi < rhs.lo || lhs.lo > rhs.hi)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "ne" || cmp == "one") {
    if (lhs.hi < rhs.lo || lhs.lo > rhs.hi)
      return MaskLattice::AllTrue;
    if (lhs.lo == lhs.hi && rhs.lo == rhs.hi && lhs.lo == rhs.lo)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  return MaskLattice::Unknown;
}

static MaskLattice classifyMaskValue(Value mask) {
  if (isAllActiveSeed(mask))
    return MaskLattice::AllTrue;
  if (isAllInactiveSeed(mask))
    return MaskLattice::AllFalse;

  if (auto vcmp = mask.getDefiningOp<VMIVcmpOp>()) {
    auto lhs = matchVectorLaneRange(vcmp.getLhs());
    auto rhs = matchVectorLaneRange(vcmp.getRhs());
    if (!lhs || !rhs)
      return MaskLattice::Unknown;
    MaskLattice raw = classifyCompare(vcmp.getCmp(), *lhs, *rhs);
    // Seed ANDs with the raw compare (pmode zeroing).
    MaskLattice seedLat = classifyMaskValue(vcmp.getSeed());
    if (raw == MaskLattice::AllFalse || seedLat == MaskLattice::AllFalse)
      return MaskLattice::AllFalse;
    if (raw == MaskLattice::AllTrue && seedLat == MaskLattice::AllTrue)
      return MaskLattice::AllTrue;
    return MaskLattice::Unknown;
  }

  if (auto vcmps = mask.getDefiningOp<VMIVcmpsOp>()) {
    auto lhs = matchVectorLaneRange(vcmps.getSrc());
    auto sc = matchConstantInt(vcmps.getScalar());
    if (!lhs || !sc)
      return MaskLattice::Unknown;
    MaskLattice raw =
        classifyCompare(vcmps.getCmp(), *lhs, IntRange::splat(*sc));
    MaskLattice seedLat = classifyMaskValue(vcmps.getSeed());
    if (raw == MaskLattice::AllFalse || seedLat == MaskLattice::AllFalse)
      return MaskLattice::AllFalse;
    if (raw == MaskLattice::AllTrue && seedLat == MaskLattice::AllTrue)
      return MaskLattice::AllTrue;
    return MaskLattice::Unknown;
  }

  return MaskLattice::Unknown;
}

static bool isTriviallyDeadPureOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<func::FuncOp, ModuleOp, scf::ForOp, scf::IfOp, scf::WhileOp,
          scf::YieldOp, scf::ConditionOp>(op))
    return false;
  // Only drop side-effect-free ops (VMI compute / mask creators are Pure).
  if (!isMemoryEffectFree(op))
    return false;
  return llvm::all_of(op->getResults(),
                      [](Value r) { return r.use_empty(); });
}

/// Erase pure unused ops to a fixed point. Safer than recursive Value
/// erase chains when one def is reachable via multiple dead users.
static void dcePureUnusedOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (isTriviallyDeadPureOp(op))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct VMIPredicateFoldPass
    : public mlir::pto::impl::VMIPredicateFoldBase<VMIPredicateFoldPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIPredicateFoldPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<VMIvSelOp> sels;
    module.walk([&](VMIvSelOp op) { sels.push_back(op); });

    for (VMIvSelOp sel : llvm::reverse(sels)) {
      if (!sel->getBlock())
        continue;

      // vsel(m, x, x) → x
      if (sel.getTrueValue() == sel.getFalseValue()) {
        sel.getResult().replaceAllUsesWith(sel.getTrueValue());
        sel.erase();
        continue;
      }

      MaskLattice lat = classifyMaskValue(sel.getMask());
      if (lat == MaskLattice::Unknown)
        continue;

      Value replacement =
          lat == MaskLattice::AllTrue ? sel.getTrueValue() : sel.getFalseValue();
      sel.getResult().replaceAllUsesWith(replacement);
      sel.erase();
    }

    dcePureUnusedOps(module);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIPredicateFoldPass() {
  return std::make_unique<VMIPredicateFoldPass>();
}
