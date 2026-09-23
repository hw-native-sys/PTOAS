// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOConvertSCFToCFWithLoopHintsPass.cpp -----------------------------===//
//
// PTOAS-specific SCF-to-CF conversion that preserves loop unroll hints.
//
// This is the pipeline's SCF-to-CF conversion (it replaces
// createConvertSCFToCFPass), extended with the one thing the stock pass
// cannot do on LLVM 19: carry {pto.unroll = "enable"} over to the loop latch
// as LLVM loop metadata.
//
// "enable" is the only hint with a metadata channel: it asks the downstream
// compiler's cost model to unroll (LLVM's ForceEnable semantics - the way of
// unrolling is chosen by the cost model, the budget veto is lifted).  The
// "full" / factor hints are consumed natively by pto-unroll-loops; anything
// this pass sees carrying {pto.unroll = "enable"} is forwarded as
//
//   #llvm.loop_annotation<unroll = <disable = false>>
//
// which the MLIR-to-LLVM-IR translation turns into !llvm.loop.unroll.enable
// metadata.
//
// LLVM 19's convert-scf-to-cf does not propagate llvm.loop_annotation from
// scf.for to the loop latch (that upstream support only exists in newer
// MLIR), so this pass owns the SCF-to-CF conversion for the whole function:
// it runs the upstream conversion patterns together with a higher-benefit
// pattern that lowers annotated scf.for loops and attaches the annotation to
// the latch cf.br.  Downstream CF->LLVM lowering preserves branch attributes
// on llvm.br, and the MLIR-to-LLVM-IR translation attaches the metadata.
//
// Converting the whole function (rather than just the annotated loops) is
// required for correctness: lowering an annotated loop in isolation leaves
// the freshly created condition/body/latch/exit blocks inside whatever
// enclosing single-block region held the loop (an outer unannotated scf.for,
// an scf.if, ...), which immediately fails that op's SingleBlock verifier.
//
// Because this pass performs the full conversion, the pipelines that run it
// must NOT also run createConvertSCFToCFPass afterwards.  It replaces that
// pass and must run at the same position - after every structured-loop
// transformation, so no later pass can clone a loop and lose its hint.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCONVERTSCFTOCFWITHLOOPHINTS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-convert-scf-to-cf-with-loop-hints"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Name of the LLVM annotation attribute as it appears on scf.for (and as
/// LoopAnnotationAttr's ODS name on cf.br; the MLIR-to-LLVM-IR translation
/// looks it up under the bare name via BrOp::getLoopAnnotationAttr()).
static constexpr llvm::StringLiteral kLoopAnnotationAttrName =
    "llvm.loop_annotation";
static constexpr llvm::StringLiteral kBranchLoopAnnotationAttrName =
    "loop_annotation";

/// Merge the enable unroll entry into the loop's existing
/// llvm.loop_annotation (if any) and set the merged attribute on *forOp*.
static void setMergedLoopAnnotation(scf::ForOp forOp) {
  MLIRContext *ctx = forOp.getContext();
  // disableNonforced = false prints as `unroll = <disable = false>`, which
  // the MLIR-to-LLVM-IR translation maps to !llvm.loop.unroll.enable.
  LLVM::LoopUnrollAttr unroll = LLVM::LoopUnrollAttr::get(
      ctx, BoolAttr::get(ctx, false), {}, {}, {}, {}, {}, {});
  auto existing =
      forOp->getAttrOfType<LLVM::LoopAnnotationAttr>(kLoopAnnotationAttrName);

  LLVM::LoopAnnotationAttr merged;
  if (!existing) {
    merged = LLVM::LoopAnnotationAttr::get(ctx, {}, {}, {}, unroll, {}, {}, {},
                                           {}, {}, {}, {}, {}, {}, {}, {});
  } else {
    if (existing.getUnroll()) {
      forOp.emitWarning() << "overwriting an existing unroll entry in '"
                          << kLoopAnnotationAttrName << "'";
    }
    merged = LLVM::LoopAnnotationAttr::get(
        ctx, existing.getDisableNonforced(), existing.getVectorize(),
        existing.getInterleave(), unroll, existing.getUnrollAndJam(),
        existing.getLicm(), existing.getDistribute(), existing.getPipeline(),
        existing.getPeeled(), existing.getUnswitch(),
        existing.getMustProgress(), existing.getIsVectorized(),
        existing.getStartLoc(), existing.getEndLoc(),
        existing.getParallelAccesses());
  }
  forOp->setAttr(kLoopAnnotationAttrName, merged);
}

/// Translate the enable hint on one loop into an llvm.loop_annotation
/// attribute.  Only {pto.unroll = "enable"} is consumed here; every other
/// attribute belongs to pto-unroll-loops and is left untouched.
static LogicalResult translateLoopHint(scf::ForOp forOp) {
  auto unrollAttr = forOp->getAttrOfType<StringAttr>(pto::kUnrollAttrName);
  StringRef hintValue = unrollAttr ? unrollAttr.getValue() : "";
  if (hintValue != pto::kUnrollEnableValue) {
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "PTOConvertSCFToCFWithLoopHints: forwarding enable hint at "
                          << forOp.getLoc() << "\n");
  setMergedLoopAnnotation(forOp);
  forOp->removeAttr(pto::kUnrollAttrName);
  return success();
}

/// Collect the operands a branch into the loop header carries: the induction
/// variable first (in its initial or stepped form), then the values the loop
/// threads through its iterations.
template <typename RangeT>
static SmallVector<Value, mlir::pto::kValue8>
headerOperands(Value inductionVar, const RangeT &carried) {
  SmallVector<Value, mlir::pto::kValue8> operands;
  operands.reserve(1 + carried.size());
  operands.push_back(inductionVar);
  llvm::append_range(operands, carried);
  return operands;
}

/// Lower one annotated scf.for to control-flow ops.
///
/// The CFG built here is the one the SCF dialect requires: a header block
/// holding the iteration test, the blocks of the original body, and a latch
/// branching back to the header.  Owning the lowering instead of leaving it to
/// convert-scf-to-cf only pays off for the last step, where the loop's LLVM
/// dialect attributes are copied onto the latch branch: the MLIR-to-LLVM-IR
/// translation reads loop metadata off the backedge, and LLVM 19's pattern
/// does not carry the annotation there.  Registered with a higher benefit than
/// the upstream pattern so that annotated loops land here and unannotated ones
/// fall through to the upstream lowering.
struct LowerAnnotatedForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!forOp->hasAttr(kLoopAnnotationAttrName)) {
      return failure();
    }

    Location loc = forOp.getLoc();

    // Cut the enclosing block at the insertion point: the loop's blocks end up
    // in the gap, whatever followed the loop stays in the continuation.
    Block *head = rewriter.getInsertionBlock();
    Block *continuation =
        rewriter.splitBlock(head, rewriter.getInsertionPoint());

    // The region's first block holds the induction variable and the
    // loop-carried values as arguments, so it becomes the header.  Its
    // operations move into a fresh entry block, and the region's remaining
    // blocks are appended to the enclosing block list ahead of the
    // continuation.
    Region &loopBody = forOp.getRegion();
    Block *header = &loopBody.front();
    Block *bodyEntry = rewriter.splitBlock(header, header->begin());
    Block *latch = &loopBody.back();
    rewriter.inlineRegionBefore(loopBody, continuation);

    Value inductionVar = header->getArgument(0);

    // The entry branch hands the lower bound and the initial values of the
    // loop-carried variables to the header.
    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::BranchOp>(
        loc, header,
        headerOperands(forOp.getLowerBound(), forOp.getInitArgs()));

    // The header decides between one more iteration and leaving the loop.
    rewriter.setInsertionPointToEnd(header);
    Value inRange = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, inductionVar, forOp.getUpperBound());
    rewriter.create<cf::CondBranchOp>(loc, inRange, bodyEntry, ValueRange{},
                                      continuation, ValueRange{});

    // The latch steps the induction variable, forwards the values the body
    // yielded and branches back to the header.  The backedge op is kept: it is
    // what the loop metadata gets attached to.
    Operation *yield = latch->getTerminator();
    rewriter.setInsertionPointToEnd(latch);
    Value next =
        rewriter.create<arith::AddIOp>(loc, inductionVar, forOp.getStep());
    auto backedge = rewriter.create<cf::BranchOp>(
        loc, header, headerOperands(next, yield->getOperands()));

    // Carry the loop's LLVM dialect attributes over to the backedge so that
    // the translation can emit the !llvm.loop metadata.  The annotation is
    // stored on the branch under its bare ODS name, which is where
    // BrOp::getLoopAnnotationAttr() looks for it.
    for (const NamedAttribute &attr : forOp->getAttrs()) {
      Attribute value = attr.getValue();
      if (!isa<LLVM::LLVMDialect>(value.getDialect())) {
        continue;
      }
      StringRef attrName = attr.getName().getValue();
      StringRef branchName = attrName == kLoopAnnotationAttrName
                                 ? StringRef(kBranchLoopAnnotationAttrName)
                                 : attrName;
      backedge->setAttr(branchName, value);
    }

    rewriter.eraseOp(yield);

    // The results of the loop are the header's arguments past the induction
    // variable, i.e. their values on the last iteration.
    rewriter.replaceOp(forOp, header->getArguments().drop_front());
    return success();
  }
};

struct PTOConvertSCFToCFWithLoopHints
    : public pto::impl::PTOConvertSCFToCFWithLoopHintsBase<PTOConvertSCFToCFWithLoopHints> {
  using pto::impl::PTOConvertSCFToCFWithLoopHintsBase<
      PTOConvertSCFToCFWithLoopHints>::PTOConvertSCFToCFWithLoopHintsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Step 1: translate {pto.unroll = "enable"} attributes into
    // llvm.loop_annotation attributes on scf.for.
    func.walk([&](scf::ForOp forOp) { (void)translateLoopHint(forOp); });

    // Step 2: run the complete SCF-to-CF conversion for the function, with
    // the annotated-loop lowering taking precedence over the upstream
    // ForLowering.  Converting everything in one pass is what keeps the IR
    // verifiable: a partially lowered loop would leave multiple blocks inside
    // an enclosing single-block region (outer scf.for, scf.if, ...).
    RewritePatternSet patterns(&getContext());
    populateSCFToControlFlowConversionPatterns(patterns);
    patterns.add<LowerAnnotatedForPattern>(patterns.getContext(),
                                           /*benefit=*/mlir::pto::kValue2);

    ConversionTarget target(getContext());
    target.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                        scf::IndexSwitchOp, scf::ParallelOp, scf::WhileOp,
                        scf::ExecuteRegionOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (mlir::failed(
            applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOConvertSCFToCFWithLoopHintsPass() {
  return std::make_unique<PTOConvertSCFToCFWithLoopHints>();
}
