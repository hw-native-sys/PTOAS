// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOMASKSIMPLIFY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

template <typename OpTy> static bool isAllTrueMaskFrom(Value mask) {
  auto op = mask.getDefiningOp<OpTy>();
  return op && op.getPattern() == "PAT_ALL";
}

static bool isAllTrueMask(Value mask) {
  return isAllTrueMaskFrom<PsetB8Op>(mask) ||
         isAllTrueMaskFrom<PsetB16Op>(mask) ||
         isAllTrueMaskFrom<PsetB32Op>(mask) ||
         isAllTrueMaskFrom<PgeB8Op>(mask) ||
         isAllTrueMaskFrom<PgeB16Op>(mask) || isAllTrueMaskFrom<PgeB32Op>(mask);
}

template <typename OpTy>
struct SimplifyAllTruePredicateReorder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isAllTrueMask(op.getLhs()) || !isAllTrueMask(op.getRhs())) {
      return failure();
    }

    rewriter.replaceOp(op, {op.getLhs(), op.getRhs()});
    return success();
  }
};

struct VPTOMaskSimplifyPass
    : public pto::impl::VPTOMaskSimplifyBase<VPTOMaskSimplifyPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyAllTruePredicateReorder<PintlvB8Op>,
                 SimplifyAllTruePredicateReorder<PintlvB16Op>,
                 SimplifyAllTruePredicateReorder<PintlvB32Op>,
                 SimplifyAllTruePredicateReorder<PdintlvB8Op>,
                 SimplifyAllTruePredicateReorder<PdintlvB16Op>,
                 SimplifyAllTruePredicateReorder<PdintlvB32Op>>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOMaskSimplifyPass() {
  return std::make_unique<VPTOMaskSimplifyPass>();
}
