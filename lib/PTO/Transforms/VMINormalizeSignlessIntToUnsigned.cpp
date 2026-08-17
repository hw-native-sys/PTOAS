// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===-- VMINormalizeSignlessIntToUnsigned.cpp - Signless→unsigned boundary --===//
//
// Normalizes signless integer element types on whitelisted VMI ops to their
// unsigned equivalent at the op boundary, using pto.vmi.bitcast to preserve
// the surrounding IR's type contract. This runs before any verifier or layout
// pass so downstream passes only see unsigned types on the whitelisted ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMINORMALIZESIGNLESSINTTOUNSIGNED
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// ---------------------------------------------------------------------------
// Op whitelist — add new ops here to enable signless→unsigned normalization
// ---------------------------------------------------------------------------

static bool isWhitelisted(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<pto::VMIVminOp, pto::VMIVmaxOp, pto::VMIVshrOp,
            pto::VMIvcmaxOp, pto::VMIvcminOp, pto::VMIMaxSOp,
            pto::VMIMinSOp, pto::VMIShrSOp, pto::VMICvtOp,
            pto::VMIMinIOp, pto::VMIMaxIOp, pto::VMIShRUIOp,
            pto::VMIReduceMaxIOp, pto::VMIReduceMinIOp,
            pto::VMIGroupReduceMaxIOp, pto::VMIGroupReduceMinIOp,
            pto::VMIFPToUIOp, pto::VMIExtUIOp,
            pto::VMIVdhistOp, pto::VMIVchistOp>([](auto) { return true; })
      .Default([](auto) { return false; });
}

// ---------------------------------------------------------------------------
// Normalization helpers
// ---------------------------------------------------------------------------

/// Returns true if t is a signless VMIVRegType.
static bool hasSignlessIntElement(pto::VMIVRegType vreg) {
  auto intTy = dyn_cast<IntegerType>(vreg.getElementType());
  return intTy && intTy.isSignless();
}

/// Creates a new VMIVRegType with the element type replaced by an unsigned
/// integer of the same width. Layout is preserved.
static pto::VMIVRegType makeUnsignedVReg(MLIRContext *ctx,
                                         pto::VMIVRegType vreg) {
  auto intTy = cast<IntegerType>(vreg.getElementType());
  auto unsignedElem = IntegerType::get(ctx, intTy.getWidth(),
                                       IntegerType::Unsigned);
  return pto::VMIVRegType::get(ctx, vreg.getElementCount(), unsignedElem,
                               vreg.getLayoutAttr());
}

/// Inserts pto.vmi.bitcast before op to convert value v from its current
/// (signless) type to an unsigned VMIVRegType. Returns the bitcast result.
static Value insertNormalizeCastBefore(Operation *op, Value v,
                                       PatternRewriter &rewriter) {
  auto vreg = cast<pto::VMIVRegType>(v.getType());
  auto newType = makeUnsignedVReg(rewriter.getContext(), vreg);
  rewriter.setInsertionPoint(op);
  return rewriter.create<pto::VMIBitcastOp>(op->getLoc(), newType, v)
      .getResult();
}

/// Inserts pto.vmi.bitcast after op to convert the result back from
/// unsigned to its original (signless) type. Updates all uses of the result
/// (except the bitcast itself) to use the bitcast instead.
static void insertNormalizeCastAfter(Operation *op, OpResult result,
                                     PatternRewriter &rewriter) {
  auto vreg = cast<pto::VMIVRegType>(result.getType());
  auto newType = makeUnsignedVReg(rewriter.getContext(), vreg);
  Type oldType = result.getType();

  result.setType(newType);
  rewriter.setInsertionPointAfter(op);
  Value bitcast =
      rewriter.create<pto::VMIBitcastOp>(op->getLoc(), oldType, result)
          .getResult();
  result.replaceAllUsesExcept(bitcast, bitcast.getDefiningOp());
}

// ---------------------------------------------------------------------------
// Normalization pattern — applied to each whitelisted op
// ---------------------------------------------------------------------------

struct NormalizeSignlessPattern : public RewritePattern {
  explicit NormalizeSignlessPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isWhitelisted(op))
      return failure();

    bool changed = false;

    // Normalize operands
    for (OpOperand &use : op->getOpOperands()) {
      Value v = use.get();
      auto vreg = dyn_cast<pto::VMIVRegType>(v.getType());
      if (!vreg || !hasSignlessIntElement(vreg))
        continue;
      Value cast = insertNormalizeCastBefore(op, v, rewriter);
      use.set(cast);
      changed = true;
    }

    // Normalize results
    for (OpResult r : op->getResults()) {
      auto vreg = dyn_cast<pto::VMIVRegType>(r.getType());
      if (!vreg || !hasSignlessIntElement(vreg))
        continue;
      insertNormalizeCastAfter(op, r, rewriter);
      changed = true;
    }

    return success(changed);
  }
};

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

struct VMINormalizeSignlessIntToUnsignedPass
    : public mlir::pto::impl::VMINormalizeSignlessIntToUnsignedBase<
          VMINormalizeSignlessIntToUnsignedPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VMINormalizeSignlessIntToUnsignedPass)

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<NormalizeSignlessPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> pto::createVMINormalizeSignlessIntToUnsignedPass() {
  return std::make_unique<VMINormalizeSignlessIntToUnsignedPass>();
}
