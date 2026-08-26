// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAbsorbAddPtr.cpp -----------------------------------------------===//
//
// Backend-shape canonical fold for VPTO memory operations:
//
//   op(addptr(base, A), O)  ->  op(base, A + O)
//
// Bisheng only emits a post-update load/store when the affine offset sits on
// the operation itself; an offset hidden inside an `addptr` is re-materialized
// as `VLDI + SADD` per iteration. This fold is the "addptr absorption" rule of
// docs/designs/vpto-integer-address-canonicalization-design-zh.md and must run
// before the post-update consumer (VPTOSoftPostUpdate).
//
// Legality comes only from VPTOAddressSemanticsOpInterface: a current access
// with an Element-unit offset, a base that is an `addptr` result, no
// updated-base post-update form, and a no-loss index addition.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/VPTOAddressSemantics.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOABSORBADDPTR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct AbsorbAddPtrIntoOpOffset final : public RewritePattern {
  AbsorbAddPtrIntoOpOffset(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto semantics = dyn_cast<pto::VPTOAddressSemanticsOpInterface>(op);
    if (!semantics) {
      return rewriter.notifyMatchFailure(op, "no address semantics");
    }
    VPTOAddressSemantics contract = semantics.getVPTOAddressSemantics();
    if (contract.currentAccesses.empty()) {
      return rewriter.notifyMatchFailure(op, "no current access");
    }
    const VPTOAddressAccess &access = contract.currentAccesses.front();
    if (!access.offset || access.offset->unit != VPTOAddressUnit::Element) {
      return rewriter.notifyMatchFailure(op, "offset is not element-unit");
    }
    // Post-update form: the offset operand denotes the after-access advance,
    // not a current access; never fold those.
    if (contract.postUpdate && contract.postUpdate->updatedBase) {
      return rewriter.notifyMatchFailure(op, "already in post-update form");
    }

    Value base = access.baseOperand->get();
    auto addptr = base.getDefiningOp<pto::AddPtrOp>();
    if (!addptr) {
      return rewriter.notifyMatchFailure(op, "base is not an addptr");
    }
    if (!addptr.getOffset().getType().isIndex() ||
        !access.offset->operand->get().getType().isIndex()) {
      return rewriter.notifyMatchFailure(op, "offsets are not index-typed");
    }
    // addptr and the op share the same pointer element type because the op's
    // base operand *is* the addptr result (AllTypesMatch on AddPtrOp).

    Value combined = rewriter.create<arith::AddIOp>(
        op->getLoc(), addptr.getOffset(), access.offset->operand->get());

    rewriter.modifyOpInPlace(op, [&]() {
      access.baseOperand->set(addptr.getPtr());
      access.offset->operand->set(combined);
    });
    if (addptr->use_empty()) {
      rewriter.eraseOp(addptr);
    }
    return success();
  }
};

struct PTOAbsorbAddPtrPass final
    : public pto::impl::PTOAbsorbAddPtrBase<PTOAbsorbAddPtrPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<AbsorbAddPtrIntoOpOffset>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOAbsorbAddPtrPass() {
  return std::make_unique<PTOAbsorbAddPtrPass>();
}
