// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOCombineReductions.cpp - Combine physical reduction trees -------===//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOCOMBINEREDUCTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kMinReductionLeaves = 2;

bool areEquivalentMasks(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  bool hasDifferentType = lhs.getType() != rhs.getType();
  if (hasDifferentType) {
    return false;
  }

  Operation *lhsOp = lhs.getDefiningOp();
  Operation *rhsOp = rhs.getDefiningOp();
  bool hasDifferentProducer =
      !lhsOp || !rhsOp || lhsOp->getName() != rhsOp->getName();
  if (hasDifferentProducer) {
    return false;
  }

  bool isPatternMask =
      isa<PsetB8Op, PsetB16Op, PsetB32Op, PgeB8Op, PgeB16Op, PgeB32Op>(lhsOp);
  return isPatternMask &&
         lhsOp->getAttr("pattern") == rhsOp->getAttr("pattern");
}

bool isReduction(Value value) {
  Operation *op = value.getDefiningOp();
  return op && isa<VcaddOp, VcmaxOp, VcminOp, VcgaddOp, VcgmaxOp, VcgminOp>(op);
}

template <typename CombineOpTy, typename ReduceOpTy>
struct CombineEquivalentReductionTreePattern : OpRewritePattern<CombineOpTy> {
  using OpRewritePattern<CombineOpTy>::OpRewritePattern;

  struct ReductionLeaf {
    Value source;
    Value mask;
  };

  LogicalResult matchAndRewrite(CombineOpTy op,
                                PatternRewriter &rewriter) const override {
    SmallVector<ReductionLeaf> reductions;
    SmallVector<Value, 1> baseValues;
    bool invalidReductionTree =
        failed(collectReductionTree(op, reductions, baseValues)) ||
        !haveEquivalentMasks(reductions);
    if (invalidReductionTree) {
      return failure();
    }
    // Accumulator lowering builds the tree from the last physical chunk back
    // toward init. Restore source order to match the direct one-to-N recipe.
    if (!baseValues.empty()) {
      std::reverse(reductions.begin(), reductions.end());
    }
    FailureOr<Value> combinedSource =
        combineReductionSources(op, reductions, rewriter);
    if (failed(combinedSource)) {
      return failure();
    }
    Value reduced = rewriter
                        .create<ReduceOpTy>(
                            op.getLoc(), op.getResult().getType(),
                            *combinedSource, reductions.front().mask)
                        .getResult();
    return replaceReduction(op, reduced, baseValues, rewriter);
  }

private:
  LogicalResult
  collectReductionTree(CombineOpTy op,
                       SmallVectorImpl<ReductionLeaf> &reductions,
                       SmallVectorImpl<Value> &baseValues) const {
    bool invalidTree =
        failed(collect(op.getLhs(), op.getMask(), reductions, baseValues)) ||
        failed(collect(op.getRhs(), op.getMask(), reductions, baseValues)) ||
        reductions.size() < kMinReductionLeaves || baseValues.size() > 1;
    return failure(invalidTree);
  }

  static bool
  haveEquivalentMasks(ArrayRef<ReductionLeaf> reductions) {
    Value reductionMask = reductions.front().mask;
    return llvm::all_of(llvm::drop_begin(reductions),
                        [reductionMask](ReductionLeaf leaf) {
                          return areEquivalentMasks(reductionMask, leaf.mask);
                        });
  }

  static FailureOr<Value>
  combineReductionSources(CombineOpTy op, ArrayRef<ReductionLeaf> reductions,
                          PatternRewriter &rewriter) {
    Value combinedSource = reductions.front().source;
    auto sourceType = dyn_cast<VRegType>(combinedSource.getType());
    bool hasDifferentType =
        !sourceType || llvm::any_of(llvm::drop_begin(reductions),
                                   [sourceType](ReductionLeaf leaf) {
                                     return leaf.source.getType() != sourceType;
                                   });
    if (hasDifferentType) {
      return failure();
    }
    for (ReductionLeaf leaf : llvm::drop_begin(reductions)) {
      combinedSource =
          rewriter
              .create<CombineOpTy>(op.getLoc(), sourceType, combinedSource,
                                   leaf.source, reductions.front().mask)
              .getResult();
    }
    return combinedSource;
  }

  static LogicalResult replaceReduction(
      CombineOpTy op, Value reduced, ArrayRef<Value> baseValues,
      PatternRewriter &rewriter) {
    if (baseValues.empty()) {
      rewriter.replaceOp(op, reduced);
      return success();
    }

    Value base = baseValues.front();
    bool hasDifferentType = base.getType() != op.getResult().getType();
    if (hasDifferentType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<CombineOpTy>(op, op.getResult().getType(),
                                             reduced, base, op.getMask());
    return success();
  }

  LogicalResult collect(Value value, Value combineMask,
                        SmallVectorImpl<ReductionLeaf> &reductions,
                        SmallVectorImpl<Value> &baseValues) const {
    if (auto reduction = value.getDefiningOp<ReduceOpTy>()) {
      reductions.push_back({reduction.getInput(), reduction.getMask()});
      return success();
    }

    if (auto combine = value.getDefiningOp<CombineOpTy>()) {
      if (!areEquivalentMasks(combineMask, combine.getMask())) {
        return failure();
      }
      if (failed(
              collect(combine.getLhs(), combineMask, reductions, baseValues))) {
        return failure();
      }
      return collect(combine.getRhs(), combineMask, reductions, baseValues);
    }

    // A reduction from another family is not an init value. Reject mixed
    // trees rather than changing their operation semantics.
    bool cannotUseAsBase = isReduction(value) || baseValues.size() == 1;
    if (cannotUseAsBase) {
      return failure();
    }
    baseValues.push_back(value);
    return success();
  }
};

struct VPTOCombineReductionsPass
    : pto::impl::VPTOCombineReductionsBase<VPTOCombineReductionsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CombineEquivalentReductionTreePattern<VaddOp, VcaddOp>,
                 CombineEquivalentReductionTreePattern<VaddOp, VcgaddOp>,
                 CombineEquivalentReductionTreePattern<VmaxOp, VcmaxOp>,
                 CombineEquivalentReductionTreePattern<VmaxOp, VcgmaxOp>,
                 CombineEquivalentReductionTreePattern<VminOp, VcminOp>,
                 CombineEquivalentReductionTreePattern<VminOp, VcgminOp>>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOCombineReductionsPass() {
  return std::make_unique<VPTOCombineReductionsPass>();
}
