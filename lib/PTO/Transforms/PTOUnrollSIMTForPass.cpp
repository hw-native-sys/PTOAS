// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnrollSIMTForPass.cpp -------------------------------------------===//
//
// Unroll explicitly annotated scf.for loops inside SIMT contexts to eliminate
// divergent control flow before LLVM lowering.
//
// Only loops carrying the `{pto.unroll = "full"}` attribute are unrolled.
// A SIMT context is either a pto.simt_entry function or an inline
// pto.section.simt region that has not been outlined yet.  General-purpose
// loops outside these contexts are not affected.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLSIMTFOR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-simt-for"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Name of the unroll annotation placed on scf::ForOp by users.
static constexpr llvm::StringLiteral kUnrollAttrName = "pto.unroll";
static constexpr llvm::StringLiteral kUnrollFullValue = "full";

/// Check whether the loop has the explicit "full unroll" annotation.
static bool hasUnrollFullAttr(scf::ForOp forOp) {
  if (auto attr = forOp->getAttrOfType<StringAttr>(kUnrollAttrName))
    return attr.getValue() == kUnrollFullValue;
  return false;
}

/// Check whether this function is a SIMT entry.
static bool isSIMTEntry(func::FuncOp func) {
  return func->hasAttr(pto::kPTOSimtEntryAttrName);
}

/// Check whether a function contains an inline SIMT section.
static bool containsInlineSIMTSection(func::FuncOp func) {
  WalkResult result =
      func.walk([](pto::SectionSimtOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

/// Check whether a loop is inside either supported SIMT representation.
static bool isInSIMTContext(scf::ForOp forOp) {
  func::FuncOp func = forOp->getParentOfType<func::FuncOp>();
  if (!func)
    return false;
  return isSIMTEntry(func) ||
         static_cast<bool>(forOp->getParentOfType<pto::SectionSimtOp>());
}

// ---------------------------------------------------------------------------
// Rewrite pattern
// ---------------------------------------------------------------------------

namespace {

struct UnrollSIMTForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only apply inside an outlined or inline SIMT context.
    if (!isInSIMTContext(forOp))
      return failure();

    // Only unroll loops with explicit {pto.unroll = "full"} annotation.
    if (!hasUnrollFullAttr(forOp))
      return failure();

    std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
      return failure();

    int64_t tripCount = (*ub - *lb + *step - 1) / *step;
    if (tripCount <= 0)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "PTOUnrollSIMTFor: unrolling annotated scf.for tripCount="
               << tripCount << " at " << forOp.getLoc() << "\n");

    // loopUnrollByFactor returns failure if the loop carries iteration
    // arguments that have uses outside the loop (live-out values).  In that
    // case we cannot unroll.
    if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(tripCount))))
      return failure();

    return success();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

namespace {

struct PTOUnrollSIMTFor : public pto::impl::PTOUnrollSIMTForBase<PTOUnrollSIMTFor> {
  using pto::impl::PTOUnrollSIMTForBase<
      PTOUnrollSIMTFor>::PTOUnrollSIMTForBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!isSIMTEntry(func) && !containsInlineSIMTSection(func))
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<UnrollSIMTForPattern>(ctx);

    GreedyRewriteConfig config;
    config.maxIterations = 10; // loops may nest
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOUnrollSIMTForPass() {
  return std::make_unique<PTOUnrollSIMTFor>();
}
