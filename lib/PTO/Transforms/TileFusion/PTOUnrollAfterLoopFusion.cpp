// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"

#include "PTO/IR/PTO.h" // FusionRegionOp
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h" // loopUnrollByFactor
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h" // walk, WalkOrder
#include "mlir/Interfaces/LoopLikeInterface.h" // getConstantIntValue
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLAFTERLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-after-loop-fusion"

static constexpr llvm::StringLiteral kRowUnrollFactorAttr =
    "pto.fusion.row_unroll_factor";
static constexpr llvm::StringLiteral kColUnrollFactorAttr =
    "pto.fusion.col_unroll_factor";

namespace {

/// Read an i64 unroll factor from `region`; return the value only if it is
/// present and strictly greater than 1, else 0.
static int64_t getEffectiveFactor(pto::FusionRegionOp region,
                                  llvm::StringRef attrName) {
  if (auto attr = region->getAttrOfType<IntegerAttr>(attrName)) {
    int64_t v = attr.getInt();
    if (v > 1) {
      return v;
    }
  }
  return 0;
}

/// Whether `forOp` has a child scf.for in its body (i.e. it is NOT the
/// innermost loop of the nest).
static bool hasChildForOp(scf::ForOp forOp) {
  bool hasChild = false;
  forOp.getBody()->walk([&](scf::ForOp) {
    hasChild = true;
    return WalkResult::interrupt();
  });
  return hasChild;
}

/// Constant trip count of a half-open scf.for (`lb <= i < ub`, `step > 0`),
/// or std::nullopt if any bound/step is not a compile-time constant.
static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) {
    return std::nullopt;
  }
  return (*ub - *lb + *step - 1) / *step; // ceilDiv
}

struct UnrollFactor {
  int64_t value;
  llvm::StringRef source;
};

static FailureOr<UnrollFactor> selectUnrollFactor(int64_t rowFactor,
                                                  int64_t colFactor) {
  if (colFactor > 1) {
    return UnrollFactor{colFactor, "col"};
  }
  if (rowFactor > 1) {
    return UnrollFactor{rowFactor, "row"};
  }
  return failure();
}

static FailureOr<int64_t> getDivisibleTripCount(scf::ForOp forOp,
                                                UnrollFactor factor) {
  std::optional<int64_t> trip = getConstantTripCount(forOp);
  if (!trip) {
    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip non-constant "
               << "trip scf.for at " << forOp.getLoc() << " factor(from "
               << factor.source << ")=" << factor.value << "\n");
    return failure();
  }
  if (*trip % factor.value != 0) {
    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip indivisible "
               << "trip scf.for at " << forOp.getLoc() << " trip=" << *trip
               << " factor=" << factor.value << "(from " << factor.source
               << ")\n");
    return failure();
  }
  return *trip;
}

/// Attempt to partial-unroll a single leaf `scf.for` inside a fusion_region.
/// Returns success on actual unroll, failure on every skip (non-fusion scope,
/// non-leaf, no factor > 1, non-constant / indivisible trip)
static LogicalResult tryUnrollLeafForOp(scf::ForOp forOp) {
  // Scope gate: only loops inside a fusion_region.
  auto region = forOp->getParentOfType<pto::FusionRegionOp>();
  if (!region) {
    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip non-fusion "
               << "scf.for at " << forOp.getLoc() << "\n");
    return failure();
  }

  int64_t rowF = getEffectiveFactor(region, kRowUnrollFactorAttr);
  int64_t colF = getEffectiveFactor(region, kColUnrollFactorAttr);

  // Only unroll the innermost (leaf) loop.
  if (hasChildForOp(forOp)) {
    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip non-leaf "
               << "(outer) scf.for at " << forOp.getLoc()
               << " -- unrolling outer breaks LoadStoreElision;"
               << " row_f=" << rowF << " col_f=" << colF
               << " (cost-model intent may target this layer but pass "
               << "defers to the leaf)\n");
    return failure();
  }

  // Leaf takes whichever factor is > 1 (col preferred; in a two-layer nest
  // the leaf is the col loop). The cost model may place the > 1 value on
  // either attribute depending on which layer is the effective innermost
  // (e.g. col trip == 1 -> col gets folded away -> row becomes leaf -> the
  // > 1 value lives on row_f). Both > 1 is a legal input; col wins.
  FailureOr<UnrollFactor> factor = selectUnrollFactor(rowF, colF);
  if (failed(factor)) {
    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip no factor>1 "
               << "scf.for at " << forOp.getLoc()
               << " row_f=" << rowF << " col_f=" << colF << "\n");
    return failure();
  }

  FailureOr<int64_t> trip = getDivisibleTripCount(forOp, *factor);
  if (failed(trip)) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: unroll scf.for at "
             << forOp.getLoc() << " factor=" << factor->value << "(from "
             << factor->source << ") trip=" << *trip
             << " row_f=" << rowF << " col_f=" << colF << "\n");

  // loopUnrollByFactor rewrites the loop in place (step scaled, body copied)
  // via its own internal IRRewriter, bypassing any outer rewriter/listener --
  // which is exactly why we are a walk and not an OpRewritePattern.
  auto unrolled =
      loopUnrollByFactor(forOp, static_cast<uint64_t>(factor->value));
  if (failed(unrolled)) {
    LLVM_DEBUG(llvm::dbgs() << "  loopUnrollByFactor failed "
               << "(iter_args live-out?) at " << forOp.getLoc() << "\n");
    return failure();
  }
  (void)unrolled; // mainLoopOp/epilogueLoopOp unused: divisibility => no tail.

  llvm::StringRef consumedAttr =
      factor->source == "col" ? kColUnrollFactorAttr : kRowUnrollFactorAttr;
  region->setAttr(consumedAttr,
                  Builder(region->getContext()).getI64IntegerAttr(/*value=*/1));
  return success();
}

struct PTOUnrollAfterLoopFusion
    : public pto::impl::PTOUnrollAfterLoopFusionBase<
          PTOUnrollAfterLoopFusion> {
  using pto::impl::PTOUnrollAfterLoopFusionBase<
      PTOUnrollAfterLoopFusion>::PTOUnrollAfterLoopFusionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Gather all scf.for in post-order (innermost first). The walk callback
    // only collects; every mutation happens in the loop below, after the walk
    // completes, so we never walk IR being rewritten under us.
    SmallVector<scf::ForOp, mlir::pto::kValue4> candidates;
    func.walk([&](scf::ForOp forOp) { candidates.push_back(forOp); });

    for (scf::ForOp forOp : candidates) {
      (void)tryUnrollLeafForOp(forOp);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOUnrollAfterLoopFusionPass() {
  return std::make_unique<PTOUnrollAfterLoopFusion>();
}
