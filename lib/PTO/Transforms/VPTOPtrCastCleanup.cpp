// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOPTRCASTCLEANUP
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct CollapsePtrMemRefPtrBridgePattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }

    auto resultPtrType = dyn_cast<pto::PtrType>(op.getResult(0).getType());
    if (!resultPtrType) {
      return failure();
    }

    auto castOp = op.getOperand(0).getDefiningOp<memref::CastOp>();
    if (!castOp || castOp->getNumOperands() != 1) {
      return failure();
    }

    auto innerCast =
        castOp.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!innerCast || innerCast->getNumOperands() != 1 ||
        innerCast->getNumResults() != 1) {
      return failure();
    }

    Value basePtr = innerCast.getOperand(0);
    if (basePtr.getType() != resultPtrType) {
      return failure();
    }

    rewriter.replaceOp(op, basePtr);
    if (castOp->use_empty()) {
      rewriter.eraseOp(castOp);
    }
    if (innerCast->use_empty()) {
      rewriter.eraseOp(innerCast);
    }
    return success();
  }
};

struct VPTOPtrCastCleanupPass
    : public pto::impl::VPTOPtrCastCleanupBase<VPTOPtrCastCleanupPass> {
  using pto::impl::VPTOPtrCastCleanupBase<
      VPTOPtrCastCleanupPass>::VPTOPtrCastCleanupBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapsePtrMemRefPtrBridgePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOPtrCastCleanupPass() {
  return std::make_unique<VPTOPtrCastCleanupPass>();
}
