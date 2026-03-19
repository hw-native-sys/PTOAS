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

LogicalResult lowerTMAXOp(TMaxOp op, PatternRewriter &rewriter) {
  return lowerTMAX(op, rewriter);
}

LogicalResult lowerTMINOp(TMinOp op, PatternRewriter &rewriter) {
  return lowerTMIN(op, rewriter);
}

LogicalResult lowerTANDOp(TAndOp op, PatternRewriter &rewriter) {
  return lowerTAND(op, rewriter);
}

LogicalResult lowerTANDSOp(TAndSOp op, PatternRewriter &rewriter) {
  return lowerTANDS(op, rewriter);
}

LogicalResult lowerTOROp(TOrOp op, PatternRewriter &rewriter) {
  return lowerTOR(op, rewriter);
}

LogicalResult lowerTORSOp(TOrSOp op, PatternRewriter &rewriter) {
  return lowerTORS(op, rewriter);
}

LogicalResult lowerTXOROp(TXorOp op, PatternRewriter &rewriter) {
  return lowerTXOR(op, rewriter);
}

LogicalResult lowerTXORSOp(TXorSOp op, PatternRewriter &rewriter) {
  return lowerTXORS(op, rewriter);
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

LogicalResult lowerTRSQRTOp(TRsqrtOp op, PatternRewriter &rewriter) {
  return lowerTRSQRT(op, rewriter);
}

LogicalResult lowerTRECIPOp(TRecipOp op, PatternRewriter &rewriter) {
  return lowerTRECIP(op, rewriter);
}

LogicalResult lowerTNEGOp(TNegOp op, PatternRewriter &rewriter) {
  return lowerTNEG(op, rewriter);
}

LogicalResult lowerTLRELUOp(TLReluOp op, PatternRewriter &rewriter) {
  return lowerTLRELU(op, rewriter);
}

LogicalResult lowerTCIOp(TCIOp op, PatternRewriter &rewriter) {
  return lowerTCI(op, rewriter);
}

LogicalResult lowerTCVTOp(TCvtOp op, PatternRewriter &rewriter) {
  return lowerTCVT(op, rewriter);
}

LogicalResult lowerTCmpOp(TCmpOp op, PatternRewriter &rewriter) {
  return lowerTCmp(op, rewriter);
}

LogicalResult lowerTCmpSOp(TCmpSOp op, PatternRewriter &rewriter) {
  return lowerTCmpS(op, rewriter);
}

LogicalResult lowerTSelOp(TSelOp op, PatternRewriter &rewriter) {
  return lowerTSel(op, rewriter);
}

LogicalResult lowerTAddCOp(TAddCOp op, PatternRewriter &rewriter) {
  return lowerTAddC(op, rewriter);
}

LogicalResult lowerTAddSOp(TAddSOp op, PatternRewriter &rewriter) {
  return lowerTAddS(op, rewriter);
}

LogicalResult lowerTAddSCOp(TAddSCOp op, PatternRewriter &rewriter) {
  return lowerTAddSC(op, rewriter);
}

LogicalResult lowerTMinSOp(TMinSOp op, PatternRewriter &rewriter) {
  return lowerTMinS(op, rewriter);
}

LogicalResult lowerTSubCOp(TSubCOp op, PatternRewriter &rewriter) {
  return lowerTSubC(op, rewriter);
}

LogicalResult lowerTSubSOp(TSubSOp op, PatternRewriter &rewriter) {
  return lowerTSubS(op, rewriter);
}

LogicalResult lowerTSubSCOp(TSubSCOp op, PatternRewriter &rewriter) {
  return lowerTSubSC(op, rewriter);
}

LogicalResult lowerTMaxSOp(TMaxSOp op, PatternRewriter &rewriter) {
  return lowerTMaxS(op, rewriter);
}

LogicalResult lowerTDivSOp(TDivSOp op, PatternRewriter &rewriter) {
  return lowerTDivS(op, rewriter);
}

LogicalResult lowerTMulSOp(TMulSOp op, PatternRewriter &rewriter) {
  return lowerTMulS(op, rewriter);
}

LogicalResult lowerTSelSOp(TSelSOp op, PatternRewriter &rewriter) {
  return lowerTSelS(op, rewriter);
}

LogicalResult lowerTRELUOp(TReluOp op, PatternRewriter &rewriter) {
  return lowerTRELU(op, rewriter);
}

LogicalResult lowerTNOTOp(TNotOp op, PatternRewriter &rewriter) {
  return lowerTNOT(op, rewriter);
}

LogicalResult lowerTTRANSOp(TTransOp op, PatternRewriter &rewriter) {
  return lowerTTRANS(op, rewriter);
}

LogicalResult lowerTFILLPADOp(TFillPadOp op, PatternRewriter &rewriter) {
  return lowerTFILLPAD(op, rewriter);
}

LogicalResult lowerTFILLPADExpandOp(TFillPadExpandOp op, PatternRewriter &rewriter) {
  return lowerTFILLPADExpand(op, rewriter);
}

LogicalResult lowerTRowMaxOp(TRowMaxOp op, PatternRewriter &rewriter) {
  return lowerTRowMax(op, rewriter);
}

LogicalResult lowerTRowMinOp(TRowMinOp op, PatternRewriter &rewriter) {
  return lowerTRowMin(op, rewriter);
}

LogicalResult lowerTRowSumOp(TRowSumOp op, PatternRewriter &rewriter) {
  return lowerTRowSum(op, rewriter);
}

LogicalResult lowerTColMaxOp(TColMaxOp op, PatternRewriter &rewriter) {
  return lowerTColMax(op, rewriter);
}

LogicalResult lowerTColMinOp(TColMinOp op, PatternRewriter &rewriter) {
  return lowerTColMin(op, rewriter);
}

LogicalResult lowerTColSumOp(TColSumOp op, PatternRewriter &rewriter) {
  return lowerTColSum(op, rewriter);
}

LogicalResult lowerTRowExpandOp(TRowExpandOp op, PatternRewriter &rewriter) {
  return lowerTRowExpand(op, rewriter);
}

LogicalResult lowerTColExpandOp(TColExpandOp op, PatternRewriter &rewriter) {
  return lowerTColExpand(op, rewriter);
}

LogicalResult lowerTRowExpandMulOp(TRowExpandMulOp op, PatternRewriter &rewriter) {
  return lowerTRowExpandMul(op, rewriter);
}

LogicalResult lowerTRowExpandDivOp(TRowExpandDivOp op, PatternRewriter &rewriter) {
  return lowerTRowExpandDiv(op, rewriter);
}

LogicalResult lowerTRowExpandSubOp(TRowExpandSubOp op, PatternRewriter &rewriter) {
  return lowerTRowExpandSub(op, rewriter);
}

LogicalResult lowerTPartAddOp(TPartAddOp op, PatternRewriter &rewriter) {
  return lowerTPartAdd(op, rewriter);
}

LogicalResult lowerTPartMaxOp(TPartMaxOp op, PatternRewriter &rewriter) {
  return lowerTPartMax(op, rewriter);
}

LogicalResult lowerTPartMinOp(TPartMinOp op, PatternRewriter &rewriter) {
  return lowerTPartMin(op, rewriter);
}

LogicalResult lowerTExpandSOp(TExpandsOp op, PatternRewriter &rewriter) {
  return lowerTExpandS(op, rewriter);
}

LogicalResult lowerTGatherOp(TGatherOp op, PatternRewriter &rewriter) {
  return lowerTGather(op, rewriter);
}

LogicalResult lowerTGatherBOp(TGatherBOp op, PatternRewriter &rewriter) {
  return lowerTGatherB(op, rewriter);
}

LogicalResult lowerTScatterOp(TScatterOp op, PatternRewriter &rewriter) {
  return lowerTScatter(op, rewriter);
}

LogicalResult lowerTMrgSortOp(TMrgSortOp op, PatternRewriter &rewriter) {
  return lowerTMrgSort(op, rewriter);
}

LogicalResult lowerTSort32Op(TSort32Op op, PatternRewriter &rewriter) {
  return lowerTSort32(op, rewriter);
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

LogicalResult lowerGetBufOp(GetBufOp op, PatternRewriter &rewriter) {
  return lowerGetBuf(op, rewriter);
}

LogicalResult lowerRlsBufOp(RlsBufOp op, PatternRewriter &rewriter) {
  return lowerRlsBuf(op, rewriter);
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
  else if (auto tmax = dyn_cast<TMaxOp>(op))
    lowered = lowerTMAXOp(tmax, rewriter);
  else if (auto tmin = dyn_cast<TMinOp>(op))
    lowered = lowerTMINOp(tmin, rewriter);
  else if (auto tand = dyn_cast<TAndOp>(op))
    lowered = lowerTANDOp(tand, rewriter);
  else if (auto tands = dyn_cast<TAndSOp>(op))
    lowered = lowerTANDSOp(tands, rewriter);
  else if (auto tor = dyn_cast<TOrOp>(op))
    lowered = lowerTOROp(tor, rewriter);
  else if (auto tors = dyn_cast<TOrSOp>(op))
    lowered = lowerTORSOp(tors, rewriter);
  else if (auto txor = dyn_cast<TXorOp>(op))
    lowered = lowerTXOROp(txor, rewriter);
  else if (auto txors = dyn_cast<TXorSOp>(op))
    lowered = lowerTXORSOp(txors, rewriter);
  else if (auto texp = dyn_cast<TExpOp>(op))
    lowered = lowerTEXPOp(texp, rewriter);
  else if (auto tlog = dyn_cast<TLogOp>(op))
    lowered = lowerTLOGOp(tlog, rewriter);
  else if (auto tsqrt = dyn_cast<TSqrtOp>(op))
    lowered = lowerTSQRTOp(tsqrt, rewriter);
  else if (auto trsqr = dyn_cast<TRsqrtOp>(op))
    lowered = lowerTRSQRTOp(trsqr, rewriter);
  else if (auto trecip = dyn_cast<TRecipOp>(op))
    lowered = lowerTRECIPOp(trecip, rewriter);
  else if (auto tneg = dyn_cast<TNegOp>(op))
    lowered = lowerTNEGOp(tneg, rewriter);
  else if (auto tlrelu = dyn_cast<TLReluOp>(op))
    lowered = lowerTLRELUOp(tlrelu, rewriter);
  else if (auto tci = dyn_cast<TCIOp>(op))
    lowered = lowerTCIOp(tci, rewriter);
  else if (auto tcvt = dyn_cast<TCvtOp>(op))
    lowered = lowerTCVTOp(tcvt, rewriter);
  else if (auto tcmp = dyn_cast<TCmpOp>(op))
    lowered = lowerTCmpOp(tcmp, rewriter);
  else if (auto tcmps = dyn_cast<TCmpSOp>(op))
    lowered = lowerTCmpSOp(tcmps, rewriter);
  else if (auto tsel = dyn_cast<TSelOp>(op))
    lowered = lowerTSelOp(tsel, rewriter);
  else if (auto taddc = dyn_cast<TAddCOp>(op))
    lowered = lowerTAddCOp(taddc, rewriter);
  else if (auto tadds = dyn_cast<TAddSOp>(op))
    lowered = lowerTAddSOp(tadds, rewriter);
  else if (auto taddsc = dyn_cast<TAddSCOp>(op))
    lowered = lowerTAddSCOp(taddsc, rewriter);
  else if (auto tmins = dyn_cast<TMinSOp>(op))
    lowered = lowerTMinSOp(tmins, rewriter);
  else if (auto tsubc = dyn_cast<TSubCOp>(op))
    lowered = lowerTSubCOp(tsubc, rewriter);
  else if (auto tsubs = dyn_cast<TSubSOp>(op))
    lowered = lowerTSubSOp(tsubs, rewriter);
  else if (auto tsubsc = dyn_cast<TSubSCOp>(op))
    lowered = lowerTSubSCOp(tsubsc, rewriter);
  else if (auto tmaxs = dyn_cast<TMaxSOp>(op))
    lowered = lowerTMaxSOp(tmaxs, rewriter);
  else if (auto tdivs = dyn_cast<TDivSOp>(op))
    lowered = lowerTDivSOp(tdivs, rewriter);
  else if (auto tmuls = dyn_cast<TMulSOp>(op))
    lowered = lowerTMulSOp(tmuls, rewriter);
  else if (auto tsels = dyn_cast<TSelSOp>(op))
    lowered = lowerTSelSOp(tsels, rewriter);
  else if (auto trelu = dyn_cast<TReluOp>(op))
    lowered = lowerTRELUOp(trelu, rewriter);
  else if (auto tnot = dyn_cast<TNotOp>(op))
    lowered = lowerTNOTOp(tnot, rewriter);
  else if (auto ttrans = dyn_cast<TTransOp>(op))
    lowered = lowerTTRANSOp(ttrans, rewriter);
  else if (auto tfillpad = dyn_cast<TFillPadOp>(op))
    lowered = lowerTFILLPADOp(tfillpad, rewriter);
  else if (auto tfillpadExpand = dyn_cast<TFillPadExpandOp>(op))
    lowered = lowerTFILLPADExpandOp(tfillpadExpand, rewriter);
  else if (auto trowmax = dyn_cast<TRowMaxOp>(op))
    lowered = lowerTRowMaxOp(trowmax, rewriter);
  else if (auto trowmin = dyn_cast<TRowMinOp>(op))
    lowered = lowerTRowMinOp(trowmin, rewriter);
  else if (auto trowsum = dyn_cast<TRowSumOp>(op))
    lowered = lowerTRowSumOp(trowsum, rewriter);
  else if (auto tcolmax = dyn_cast<TColMaxOp>(op))
    lowered = lowerTColMaxOp(tcolmax, rewriter);
  else if (auto tcolmin = dyn_cast<TColMinOp>(op))
    lowered = lowerTColMinOp(tcolmin, rewriter);
  else if (auto tcolsum = dyn_cast<TColSumOp>(op))
    lowered = lowerTColSumOp(tcolsum, rewriter);
  else if (auto trowexpand = dyn_cast<TRowExpandOp>(op))
    lowered = lowerTRowExpandOp(trowexpand, rewriter);
  else if (auto tcolexpand = dyn_cast<TColExpandOp>(op))
    lowered = lowerTColExpandOp(tcolexpand, rewriter);
  else if (auto trowexpandmul = dyn_cast<TRowExpandMulOp>(op))
    lowered = lowerTRowExpandMulOp(trowexpandmul, rewriter);
  else if (auto trowexpanddiv = dyn_cast<TRowExpandDivOp>(op))
    lowered = lowerTRowExpandDivOp(trowexpanddiv, rewriter);
  else if (auto trowexpandsub = dyn_cast<TRowExpandSubOp>(op))
    lowered = lowerTRowExpandSubOp(trowexpandsub, rewriter);
  else if (auto tpartadd = dyn_cast<TPartAddOp>(op))
    lowered = lowerTPartAddOp(tpartadd, rewriter);
  else if (auto tpartmax = dyn_cast<TPartMaxOp>(op))
    lowered = lowerTPartMaxOp(tpartmax, rewriter);
  else if (auto tpartmin = dyn_cast<TPartMinOp>(op))
    lowered = lowerTPartMinOp(tpartmin, rewriter);
  else if (auto texpands = dyn_cast<TExpandsOp>(op))
    lowered = lowerTExpandSOp(texpands, rewriter);
  else if (auto tgather = dyn_cast<TGatherOp>(op))
    lowered = lowerTGatherOp(tgather, rewriter);
  else if (auto tgatherb = dyn_cast<TGatherBOp>(op))
    lowered = lowerTGatherBOp(tgatherb, rewriter);
  else if (auto tscatter = dyn_cast<TScatterOp>(op))
    lowered = lowerTScatterOp(tscatter, rewriter);
  else if (auto tmrgsort = dyn_cast<TMrgSortOp>(op))
    lowered = lowerTMrgSortOp(tmrgsort, rewriter);
  else if (auto tsort32 = dyn_cast<TSort32Op>(op))
    lowered = lowerTSort32Op(tsort32, rewriter);
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
  else if (auto getBuf = dyn_cast<GetBufOp>(op))
    lowered = lowerGetBufOp(getBuf, rewriter);
  else if (auto rlsBuf = dyn_cast<RlsBufOp>(op))
    lowered = lowerRlsBufOp(rlsBuf, rewriter);
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
      if (isa<TLoadOp, TAbsOp, TAddOp, TSubOp, TMulOp, TDivOp, TMaxOp, TMinOp,
              TAndOp, TAndSOp, TOrOp, TOrSOp, TXorOp, TXorSOp, TExpOp, TLogOp,
              TSqrtOp, TRsqrtOp, TRecipOp, TNegOp, TLReluOp, TCIOp, TCvtOp, TCmpOp, TCmpSOp, TSelOp,
              TAddCOp, TAddSOp, TAddSCOp, TMinSOp, TSubCOp, TSubSOp, TSubSCOp, TMaxSOp,
              TDivSOp, TMulSOp, TSelSOp, TReluOp, TNotOp, TTransOp, TFillPadOp, TFillPadExpandOp,
              TRowMaxOp, TRowMinOp, TRowSumOp, TColMaxOp, TColMinOp, TColSumOp,
              TRowExpandOp, TColExpandOp, TRowExpandMulOp, TRowExpandDivOp,
              TRowExpandSubOp, TPartAddOp,
              TPartMaxOp, TPartMinOp, TExpandsOp, TGatherOp, TGatherBOp,
              TScatterOp, TSort32Op, TMrgSortOp, TStoreOp>(op))
        tensorPipelineOps.push_back(op);
      else if (isa<PointerCastOp, BindTileOp, SetFlagOp, WaitFlagOp, BarrierOp,
                   GetBufOp, RlsBufOp>(op))
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

    bool erasedDeadScaffold = true;
    while (erasedDeadScaffold) {
      erasedDeadScaffold = false;
      SmallVector<Operation *> deadScaffoldOps;
      module.walk([&](Operation *op) {
        if ((isa<PointerCastOp, BindTileOp>(op)) && op->use_empty())
          deadScaffoldOps.push_back(op);
      });
      for (Operation *op : deadScaffoldOps) {
        if (!op->getBlock())
          continue;
        rewriter.setInsertionPoint(op);
        rewriter.eraseOp(op);
        erasedDeadScaffold = true;
      }
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
