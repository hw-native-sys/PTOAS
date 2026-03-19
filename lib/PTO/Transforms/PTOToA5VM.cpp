//===- PTOToA5VM.cpp - PTO to A5VM pass wiring ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"
#include "PTO/Transforms/Passes.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOTOA5VM
#include "PTO/Transforms/Passes.h.inc"

namespace {

LogicalResult lowerTLOADOp(TLoadOp op, PatternRewriter &rewriter) {
  return lowerTLOAD(op, rewriter);
}

LogicalResult lowerTABSOp(TAbsOp op, PatternRewriter &rewriter) {
  return lowerTABS(op, rewriter);
}

LogicalResult lowerTADDOp(TAddOp op, PatternRewriter &rewriter) {
  return lowerTADD(op, rewriter);
}

LogicalResult lowerTSUBOp(TSubOp op, PatternRewriter &rewriter) {
  return lowerTSUB(op, rewriter);
}

LogicalResult lowerTMULOp(TMulOp op, PatternRewriter &rewriter) {
  return lowerTMUL(op, rewriter);
}

LogicalResult lowerTDIVOp(TDivOp op, PatternRewriter &rewriter) {
  return lowerTDIV(op, rewriter);
}

LogicalResult lowerTEXPOp(TExpOp op, PatternRewriter &rewriter) {
  return lowerTEXP(op, rewriter);
}

LogicalResult lowerTLOGOp(TLogOp op, PatternRewriter &rewriter) {
  return lowerTLOG(op, rewriter);
}

LogicalResult lowerTSQRTOp(TSqrtOp op, PatternRewriter &rewriter) {
  return lowerTSQRT(op, rewriter);
}

LogicalResult lowerTRECIPOp(TRecipOp op, PatternRewriter &rewriter) {
  return lowerTRECIP(op, rewriter);
}

LogicalResult lowerTRELUOp(TReluOp op, PatternRewriter &rewriter) {
  return lowerTRELU(op, rewriter);
}

LogicalResult lowerTNOTOp(TNotOp op, PatternRewriter &rewriter) {
  return lowerTNOT(op, rewriter);
}

LogicalResult lowerTSTOREOp(TStoreOp op, PatternRewriter &rewriter) {
  return lowerTSTORE(op, rewriter);
}

LogicalResult lowerSetFlagOp(SetFlagOp op, PatternRewriter &rewriter) {
  return lowerSetFlag(op, rewriter);
}

LogicalResult lowerWaitFlagOp(WaitFlagOp op, PatternRewriter &rewriter) {
  return lowerWaitFlag(op, rewriter);
}

LogicalResult lowerBarrierOp(BarrierOp op, PatternRewriter &rewriter) {
  return lowerBarrier(op, rewriter);
}

LogicalResult lowerTensorPipelineOp(Operation *op, PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(op);

  LogicalResult lowered = success();
  if (auto tload = dyn_cast<TLoadOp>(op))
    lowered = lowerTLOADOp(tload, rewriter);
  else if (auto tabs = dyn_cast<TAbsOp>(op))
    lowered = lowerTABSOp(tabs, rewriter);
  else if (auto tadd = dyn_cast<TAddOp>(op))
    lowered = lowerTADDOp(tadd, rewriter);
  else if (auto tsub = dyn_cast<TSubOp>(op))
    lowered = lowerTSUBOp(tsub, rewriter);
  else if (auto tmul = dyn_cast<TMulOp>(op))
    lowered = lowerTMULOp(tmul, rewriter);
  else if (auto tdiv = dyn_cast<TDivOp>(op))
    lowered = lowerTDIVOp(tdiv, rewriter);
  else if (auto texp = dyn_cast<TExpOp>(op))
    lowered = lowerTEXPOp(texp, rewriter);
  else if (auto tlog = dyn_cast<TLogOp>(op))
    lowered = lowerTLOGOp(tlog, rewriter);
  else if (auto tsqrt = dyn_cast<TSqrtOp>(op))
    lowered = lowerTSQRTOp(tsqrt, rewriter);
  else if (auto trecip = dyn_cast<TRecipOp>(op))
    lowered = lowerTRECIPOp(trecip, rewriter);
  else if (auto trelu = dyn_cast<TReluOp>(op))
    lowered = lowerTRELUOp(trelu, rewriter);
  else if (auto tnot = dyn_cast<TNotOp>(op))
    lowered = lowerTNOTOp(tnot, rewriter);
  else if (auto tstore = dyn_cast<TStoreOp>(op))
    lowered = lowerTSTOREOp(tstore, rewriter);
  else
    return success();

  if (failed(lowered))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

LogicalResult lowerResidualPTOOp(Operation *op, PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(op);

  LogicalResult lowered = success();
  if (auto setFlag = dyn_cast<SetFlagOp>(op))
    lowered = lowerSetFlagOp(setFlag, rewriter);
  else if (auto waitFlag = dyn_cast<WaitFlagOp>(op))
    lowered = lowerWaitFlagOp(waitFlag, rewriter);
  else if (auto barrier = dyn_cast<BarrierOp>(op))
    lowered = lowerBarrierOp(barrier, rewriter);
  else if (isa<PointerCastOp, BindTileOp>(op) && op->use_empty())
    lowered = success();
  else
    return success();

  if (failed(lowered))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

struct PTOToA5VMPass : public impl::PTOToA5VMBase<PTOToA5VMPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOToA5VMPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> tensorPipelineOps;
    SmallVector<Operation *> residualPTOOps;
    module.walk([&](Operation *op) {
      if (isa<TLoadOp, TAbsOp, TAddOp, TSubOp, TMulOp, TDivOp, TExpOp, TLogOp,
              TSqrtOp, TRecipOp, TReluOp, TNotOp, TStoreOp>(op))
        tensorPipelineOps.push_back(op);
      else if (isa<PointerCastOp, BindTileOp, SetFlagOp, WaitFlagOp, BarrierOp>(op))
        residualPTOOps.push_back(op);
    });

    PatternRewriter rewriter(&getContext());
    bool sawFailure = false;
    for (Operation *op : tensorPipelineOps) {
      if (!op->getBlock())
        continue;
      if (failed(lowerTensorPipelineOp(op, rewriter)))
        sawFailure = true;
    }
    for (Operation *op : residualPTOOps) {
      if (!op->getBlock())
        continue;
      if (failed(lowerResidualPTOOp(op, rewriter)))
        sawFailure = true;
    }

    if (sawFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerPTOToA5VMPass() {
  return std::make_unique<PTOToA5VMPass>();
}

} // namespace pto
} // namespace mlir
