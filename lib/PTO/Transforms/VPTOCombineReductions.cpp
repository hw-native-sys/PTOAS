// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOCombineReductions.cpp - Combine physical reduction trees -------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOCOMBINEREDUCTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

bool areEquivalentMasks(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  Operation *lhsOp = lhs.getDefiningOp();
  Operation *rhsOp = rhs.getDefiningOp();
  if (!lhsOp || !rhsOp || lhsOp->getName() != rhsOp->getName())
    return false;

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
    if (failed(collect(op.getLhs(), op.getMask(), reductions, baseValues)) ||
        failed(collect(op.getRhs(), op.getMask(), reductions, baseValues)) ||
        reductions.size() < 2 || baseValues.size() > 1)
      return failure();

    Value reductionMask = reductions.front().mask;
    if (!llvm::all_of(llvm::drop_begin(reductions), [&](ReductionLeaf leaf) {
          return areEquivalentMasks(reductionMask, leaf.mask);
        }))
      return failure();

    // Accumulator lowering builds the tree from the last physical chunk back
    // toward init. Restore source order to match the direct one-to-N recipe.
    if (!baseValues.empty())
      std::reverse(reductions.begin(), reductions.end());

    Value combinedSource = reductions.front().source;
    auto sourceType = dyn_cast<VRegType>(combinedSource.getType());
    if (!sourceType)
      return failure();
    for (ReductionLeaf leaf : llvm::drop_begin(reductions)) {
      if (leaf.source.getType() != sourceType)
        return failure();
      combinedSource =
          rewriter
              .create<CombineOpTy>(op.getLoc(), sourceType, combinedSource,
                                   leaf.source, reductionMask)
              .getResult();
    }

    Value reduced =
        rewriter
            .create<ReduceOpTy>(op.getLoc(), op.getResult().getType(),
                                combinedSource, reductionMask)
            .getResult();
    if (baseValues.empty()) {
      rewriter.replaceOp(op, reduced);
      return success();
    }

    Value base = baseValues.front();
    if (base.getType() != op.getResult().getType())
      return failure();
    rewriter.replaceOpWithNewOp<CombineOpTy>(op, op.getResult().getType(),
                                             reduced, base, op.getMask());
    return success();
  }

private:
  LogicalResult collect(Value value, Value combineMask,
                        SmallVectorImpl<ReductionLeaf> &reductions,
                        SmallVectorImpl<Value> &baseValues) const {
    if (auto reduction = value.getDefiningOp<ReduceOpTy>()) {
      reductions.push_back({reduction.getInput(), reduction.getMask()});
      return success();
    }

    if (auto combine = value.getDefiningOp<CombineOpTy>()) {
      if (!areEquivalentMasks(combineMask, combine.getMask()))
        return failure();
      if (failed(
              collect(combine.getLhs(), combineMask, reductions, baseValues)))
        return failure();
      return collect(combine.getRhs(), combineMask, reductions, baseValues);
    }

    // A reduction from another family is not an init value. Reject mixed
    // trees rather than changing their operation semantics.
    if (isReduction(value) || baseValues.size() == 1)
      return failure();
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
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOCombineReductionsPass() {
  return std::make_unique<VPTOCombineReductionsPass>();
}
