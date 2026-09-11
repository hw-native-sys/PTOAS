// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTONarrowVPTOLoopCounters.cpp -------------------------------------===//
//
// Narrow constant-bounded scf.for counters under VPTO vecscope regions to
// i16. The loop body continues to observe the original induction-variable
// type through a cast inserted at the beginning of the rewritten body.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTONARROWVPTOLOOPCOUNTERS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static bool isNestedInVecScope(Operation *op) {
  return op->getParentOfType<pto::VecScopeOp>() ||
         op->getParentOfType<pto::StrictVecScopeOp>();
}

static bool isNarrowableCounterType(Type type) {
  if (isa<IndexType>(type)) {
    return true;
  }
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.getWidth() > mlir::pto::kValue16;
}

static bool fitsSignedI16(int64_t value) {
  return value >= std::numeric_limits<int16_t>::min() &&
         value <= std::numeric_limits<int16_t>::max();
}

static std::optional<int64_t> getSignedI16Constant(Value value) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  if (!constant || !fitsSignedI16(*constant))
    return std::nullopt;
  return constant;
}

static bool exitValueFitsSignedI16(int64_t lower, int64_t upper, int64_t step) {
  if (lower >= upper) {
    return true;
  }

  int64_t distance = upper - lower;
  int64_t iterationCount = (distance + step - 1) / step;
  return fitsSignedI16(lower + iterationCount * step);
}

static Value createI16Constant(PatternRewriter &rewriter, Location loc,
                               int64_t value) {
  Type i16Type = rewriter.getI16Type();
  return rewriter.create<arith::ConstantOp>(
      loc, i16Type, rewriter.getIntegerAttr(i16Type, value));
}

static Value restoreInductionVariableType(PatternRewriter &rewriter,
                                          Location loc, Value inductionVar,
                                          Type originalType) {
  if (isa<IndexType>(originalType))
    return rewriter.create<arith::IndexCastOp>(loc, originalType, inductionVar);
  return rewriter.create<arith::ExtSIOp>(loc, originalType, inductionVar);
}

struct NarrowVecScopeLoopCounterPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!isNestedInVecScope(forOp)) {
      return failure();
    }

    Type originalCounterType = forOp.getInductionVar().getType();
    if (!isNarrowableCounterType(originalCounterType)) {
      return failure();
    }

    std::optional<int64_t> lower = getSignedI16Constant(forOp.getLowerBound());
    std::optional<int64_t> upper = getSignedI16Constant(forOp.getUpperBound());
    std::optional<int64_t> step = getSignedI16Constant(forOp.getStep());
    if (!lower || !upper || !step || *step <= 0 ||
        !exitValueFitsSignedI16(*lower, *upper, *step)) {
      return failure();
    }

    Location loc = forOp.getLoc();
    Value newLower = createI16Constant(rewriter, loc, *lower);
    Value newUpper = createI16Constant(rewriter, loc, *upper);
    Value newStep = createI16Constant(rewriter, loc, *step);

    auto newFor = rewriter.create<scf::ForOp>(loc, newLower, newUpper, newStep,
                                              forOp.getInitArgs());
    newFor->setAttrs(forOp->getAttrs());

    Block *oldBody = forOp.getBody();
    Block *newBody = newFor.getBody();
    if (!newBody->empty()) {
      rewriter.eraseOp(newBody->getTerminator());
    }

    rewriter.setInsertionPointToStart(newBody);
    Value restoredInductionVar = restoreInductionVariableType(
        rewriter, loc, newFor.getInductionVar(), originalCounterType);

    SmallVector<Value> bodyArgumentReplacements;
    bodyArgumentReplacements.push_back(restoredInductionVar);
    bodyArgumentReplacements.append(newFor.getRegionIterArgs().begin(),
                                    newFor.getRegionIterArgs().end());
    rewriter.mergeBlocks(oldBody, newBody, bodyArgumentReplacements);

    rewriter.replaceOp(forOp, newFor.getResults());
    return success();
  }
};

struct PTONarrowVPTOLoopCounters
    : public pto::impl::PTONarrowVPTOLoopCountersBase<
          PTONarrowVPTOLoopCounters> {
  using pto::impl::PTONarrowVPTOLoopCountersBase<
      PTONarrowVPTOLoopCounters>::PTONarrowVPTOLoopCountersBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<NarrowVecScopeLoopCounterPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTONarrowVPTOLoopCountersPass() {
  return std::make_unique<PTONarrowVPTOLoopCounters>();
}
