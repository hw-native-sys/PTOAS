// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCControlFlow.cpp ------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===

static constexpr llvm::StringLiteral kAutoSyncTailPendingModeAttr =
    "__pto.auto_sync_tail_mode";

struct ReturnToEmitC : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
      if (auto modeAttr =
              emitcFunc->getAttrOfType<StringAttr>(kAutoSyncTailPendingModeAttr)) {
        auto *ctx = rewriter.getContext();
        rewriter.setInsertionPoint(op);
        auto args = rewriter.getArrayAttr(
            {emitc::OpaqueAttr::get(ctx, modeAttr.getValue())});
        rewriter.create<emitc::CallOpaqueOp>(
            op.getLoc(), TypeRange{}, "ptoas_auto_sync_tail",
            args, ArrayAttr{}, ValueRange{});
      }
    }

    auto vals = adaptor.getOperands();
    if (vals.empty()) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, Value{});
      return success();
    }
    if (vals.size() == 1) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, vals[0]);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "EmitC cannot return multiple values");
  }
};

struct CallToEmitC : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot lower calls with multiple results");

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert call result types");

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getCalleeAttr(),
                                               resultTypes,
                                               adaptor.getOperands());
    return success();
  }
};

template <typename SectionOpTy>
struct SectionToEmitC : public OpConversionPattern<SectionOpTy> {
  using OpConversionPattern<SectionOpTy>::OpConversionPattern;

  std::string getMacroName() const {
    if (std::is_same<SectionOpTy, pto::SectionCubeOp>::value)
      return "__DAV_CUBE__";
    if (std::is_same<SectionOpTy, pto::SectionVectorOp>::value)
      return "__DAV_VEC__";
    return "UNKNOWN_MACRO";
  }

  LogicalResult
  matchAndRewrite(SectionOpTy op, typename SectionOpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    std::string startMacro = "\n#if defined(" + getMacroName() + ")";
    rewriter.create<emitc::VerbatimOp>(loc, startMacro);

    if constexpr (std::is_same_v<SectionOpTy, pto::SectionVectorOp>) {
      // Vector mask is a global HW state and may be modified by previous kernels
      // (or earlier sections). Reset it to a well-defined state for deterministic
      // execution of VEC ops.
      rewriter.create<emitc::VerbatimOp>(loc, "set_mask_norm();");
      rewriter.create<emitc::VerbatimOp>(loc, "set_vector_mask(-1, -1);");
    }

    if (needsNoSplitGuard) {
      rewriter.create<emitc::VerbatimOp>(
          loc, "if (get_subblockid() == 0) {");
    }

    Block &innerBlock = op.getBody().front();
    if (!innerBlock.empty()) {
      rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});
    }

    if (needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(loc, "}");

    std::string endMacro = "#endif // " + getMacroName() + "\n";
    rewriter.create<emitc::VerbatimOp>(loc, endMacro);

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCF Control-Flow Pre-Lowering
//
// EmitC translation supports `emitc.for`/`emitc.if` plus CFG-style
// `cf.br`/`cf.cond_br`. Upstream SCFToEmitC patterns only cover `scf.for` and
// `scf.if`, so we pre-lower some SCF ops into those supported forms.
//===----------------------------------------------------------------------===//

namespace {

static bool isTriviallyInlineableExecuteRegion(scf::ExecuteRegionOp op) {
  Region &r = op.getRegion();
  if (!r.hasOneBlock())
    return false;
  Block &b = r.front();
  return isa_and_nonnull<scf::YieldOp>(b.getTerminator());
}

static bool needsWholeFunctionSCFToCF(func::FuncOp func) {
  bool needs = false;
  func.walk([&](Operation *op) {
    if (!isa<scf::WhileOp, scf::IndexSwitchOp, scf::ExecuteRegionOp>(op))
      return WalkResult::advance();
    Operation *parentOp = op->getParentOp();

    // `scf.execute_region` can legally appear in single-block parents. Only
    // require whole-function SCFToCF if we need to lower it into CFG blocks
    // (multi-block region / non-trivial terminators).
    if (auto exec = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>() &&
          !isTriviallyInlineableExecuteRegion(exec)) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      needs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needs;
}

// scf.execute_region is semantically just an inlined region producing results
// via scf.yield. Inline it to the parent block to avoid extra lowering needs.
struct SCFExecuteRegionInline
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Block &innerBlock = op.getRegion().front();
    auto yield = dyn_cast<scf::YieldOp>(innerBlock.getTerminator());
    if (!yield)
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Move the body operations before the execute_region op.
    rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});

    // Replace execute_region results with yielded values, then erase the yield.
    rewriter.replaceOp(op, yield.getOperands());
    rewriter.eraseOp(yield);
    return success();
  }
};

// Lower scf.execute_region into CFG blocks with cf.br/cf.cond_br by inlining the
// region blocks into the parent region and rewriting scf.yield to branch into a
// continuation block carrying results.
//
// Note: This requires the parent region to allow multiple blocks (e.g. the
// function body CFG region). For execute_region nested in single-block regions
// (scf.for/scf.if), run SCFToCF first to eliminate the single-block constraint.
struct SCFExecuteRegionToCF : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (isTriviallyInlineableExecuteRegion(op))
      return rewriter.notifyMatchFailure(op, "trivially inlineable");

    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.execute_region inside a single-block parent region");
    }

    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Location loc = op.getLoc();
    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    // Split the parent block so we can branch to a continuation block with phi
    // arguments for the execute_region results.
    auto execIt = Block::iterator(op.getOperation());
    Block *continueBlock = rewriter.splitBlock(curBlock, std::next(execIt));

    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type t : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(t, loc));

    for (auto it : llvm::enumerate(op.getResults()))
      it.value().replaceAllUsesWith(contArgs[it.index()]);

    // Capture blocks before moving the region.
    SmallVector<Block *> movedBlocks;
    movedBlocks.reserve(op.getRegion().getBlocks().size());
    for (Block &b : op.getRegion())
      movedBlocks.push_back(&b);
    Block *entryBlock = &op.getRegion().front();

    // Inline the execute_region blocks into the parent region right before the
    // continuation block.
    rewriter.inlineRegionBefore(op.getRegion(), *parentRegion,
                                continueBlock->getIterator());

    // Replace all scf.yield terminators with a branch to the continuation.
    for (Block *b : movedBlocks) {
      auto yield = dyn_cast<scf::YieldOp>(b->getTerminator());
      if (!yield)
        continue;
      rewriter.setInsertionPoint(yield);
      rewriter.create<cf::BranchOp>(loc, continueBlock, yield.getOperands());
      rewriter.eraseOp(yield);
    }

    // Replace execute_region itself with a branch to the inlined entry block.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, entryBlock, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.index_switch into CFG blocks with cf.cond_br/cf.br so that we can
// avoid `scf.if` result materialization quirks (and avoid relying on cf.switch,
// which is not supported by EmitC C++ translation).
struct SCFIndexSwitchToCF : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult cloneYieldingBlockAndBranchTo(
      PatternRewriter &rewriter, Location loc, Block &srcBlock, Block *destBlock,
      Block *continueBlock) {
    rewriter.setInsertionPointToEnd(destBlock);

    IRMapping mapping;
    for (Operation &inner : srcBlock.without_terminator())
      rewriter.clone(inner, mapping);

    auto yield = dyn_cast<scf::YieldOp>(srcBlock.getTerminator());
    if (!yield)
      return failure();

    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(yield.getNumOperands());
    for (Value v : yield.getOperands())
      yieldOperands.push_back(mapping.lookupOrDefault(v));

    rewriter.create<cf::BranchOp>(loc, continueBlock, yieldOperands);
    return success();
  }

  static Block *splitBlockForContinuation(PatternRewriter &rewriter,
                                          scf::IndexSwitchOp op) {
    auto switchIt = Block::iterator(op.getOperation());
    return rewriter.splitBlock(op->getBlock(), std::next(switchIt));
  }

  static void addContinuationArguments(PatternRewriter &,
                                       scf::IndexSwitchOp op, Location loc,
                                       Block *continueBlock) {
    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(type, loc));
    for (auto result : llvm::enumerate(op.getResults()))
      result.value().replaceAllUsesWith(contArgs[result.index()]);
  }

  static void createIndexSwitchBlocks(PatternRewriter &rewriter,
                                      Region *parentRegion,
                                      Region::iterator insertPt,
                                      unsigned numCases,
                                      SmallVectorImpl<Block *> &checkBlocks,
                                      Block *&defaultBlock,
                                      SmallVectorImpl<Block *> &caseBlocks) {
    checkBlocks.reserve(numCases);
    caseBlocks.reserve(numCases);
    for (unsigned i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
    defaultBlock = rewriter.createBlock(parentRegion, insertPt);
    for (unsigned i = 0; i < numCases; ++i)
      caseBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
  }

  static void populateIndexSwitchCheckBlocks(
      PatternRewriter &rewriter, Location loc, Value selector,
      ArrayRef<int64_t> cases, ArrayRef<Block *> checkBlocks,
      ArrayRef<Block *> caseBlocks, Block *defaultBlock) {
    for (unsigned i = 0; i < checkBlocks.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      Value caseVal = rewriter.create<arith::ConstantIndexOp>(loc, cases[i]);
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, selector, caseVal);
      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultBlock;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseBlocks[i], ValueRange{},
                                        falseDest, ValueRange{});
    }
  }

  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.index_switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();
    Block *continueBlock = splitBlockForContinuation(rewriter, op);
    addContinuationArguments(rewriter, op, loc, continueBlock);

    unsigned numCases = op.getCases().size();
    auto insertPt = continueBlock->getIterator();

    SmallVector<Block *> checkBlocks;
    SmallVector<Block *> caseBlocks;
    Block *defaultBlock = nullptr;
    createIndexSwitchBlocks(rewriter, parentRegion, insertPt, numCases,
                            checkBlocks, defaultBlock, caseBlocks);

    Value selector = op.getArg();
    auto cases = op.getCases();
    populateIndexSwitchCheckBlocks(rewriter, loc, selector, cases, checkBlocks,
                                   caseBlocks, defaultBlock);

    // Fill case blocks and default block with cloned bodies + branch to cont.
    for (unsigned i = 0; i < numCases; ++i) {
      if (failed(cloneYieldingBlockAndBranchTo(
              rewriter, loc, op.getCaseBlock(i), caseBlocks[i], continueBlock)))
        return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");
    }
    if (failed(cloneYieldingBlockAndBranchTo(rewriter, loc, op.getDefaultBlock(),
                                             defaultBlock, continueBlock)))
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Replace the original switch op with a branch into the check chain.
    Block *entryDest = (numCases != 0) ? checkBlocks[0] : defaultBlock;
    rewriter.setInsertionPointAfter(op);
    rewriter.create<cf::BranchOp>(loc, entryDest, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.while into CFG blocks with cf.br/cf.cond_br.
//
// Note: This requires the parent region to allow multiple blocks. In
// particular, scf.if/scf.for regions are single-block and cannot contain this
// lowering.
struct SCFWhileToCF : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult validateWhileResultUses(scf::WhileOp op) {
    Block *parentBlock = op->getBlock();
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getBlock() != parentBlock)
          return failure();
      }
    }
    return success();
  }

  static Block *splitAfterWhileBlock(PatternRewriter &rewriter,
                                     scf::WhileOp op) {
    auto whileIt = Block::iterator(op.getOperation());
    return rewriter.splitBlock(op->getBlock(), std::next(whileIt));
  }

  static void addWhileExitArguments(PatternRewriter &, scf::WhileOp op,
                                    Location loc, Block *afterWhileBlock) {
    SmallVector<Value> exitArgs;
    exitArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes())
      exitArgs.push_back(afterWhileBlock->addArgument(type, loc));
    for (auto result : llvm::enumerate(op.getResults()))
      result.value().replaceAllUsesWith(exitArgs[result.index()]);
  }

  static Block *createWhileHeaderBlock(PatternRewriter &rewriter,
                                       scf::WhileOp op, Location loc,
                                       Block *afterWhileBlock) {
    SmallVector<Type> headerArgTypes;
    for (Value init : op.getInits())
      headerArgTypes.push_back(init.getType());
    SmallVector<Location> headerArgLocs(headerArgTypes.size(), loc);
    return rewriter.createBlock(afterWhileBlock->getParent(),
                                afterWhileBlock->getIterator(), headerArgTypes,
                                headerArgLocs);
  }

  static Block *createWhileBodyBlock(PatternRewriter &rewriter, scf::WhileOp op,
                                     Location loc, Block *afterWhileBlock) {
    Block &afterRegionBlock = op.getAfter().front();
    SmallVector<Type> bodyArgTypes(afterRegionBlock.getArgumentTypes().begin(),
                                   afterRegionBlock.getArgumentTypes().end());
    SmallVector<Location> bodyArgLocs(bodyArgTypes.size(), loc);
    return rewriter.createBlock(afterWhileBlock->getParent(),
                                afterWhileBlock->getIterator(), bodyArgTypes,
                                bodyArgLocs);
  }

  static void rewriteWhileTerminators(PatternRewriter &rewriter, Location loc,
                                      Block *headerBlock, Block *bodyBlock,
                                      Block *afterWhileBlock) {
    auto condOp = cast<scf::ConditionOp>(headerBlock->getTerminator());
    rewriter.setInsertionPoint(condOp);
    rewriter.create<cf::CondBranchOp>(loc, condOp.getCondition(),
                                      /*trueDest=*/bodyBlock,
                                      /*trueOperands=*/condOp.getArgs(),
                                      /*falseDest=*/afterWhileBlock,
                                      /*falseOperands=*/condOp.getArgs());
    rewriter.eraseOp(condOp);

    auto yieldOp = cast<scf::YieldOp>(bodyBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<cf::BranchOp>(loc, headerBlock, yieldOp.getOperands());
    rewriter.eraseOp(yieldOp);
  }

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.while inside a single-block parent region");
    }

    if (failed(validateWhileResultUses(op)))
      return rewriter.notifyMatchFailure(
          op, "unsupported: while results used outside the parent block");

    auto loc = op.getLoc();
    Block *afterWhileBlock = splitAfterWhileBlock(rewriter, op);
    addWhileExitArguments(rewriter, op, loc, afterWhileBlock);
    Block *headerBlock = createWhileHeaderBlock(rewriter, op, loc,
                                                afterWhileBlock);
    Block *bodyBlock = createWhileBodyBlock(rewriter, op, loc, afterWhileBlock);

    // Move the before/after region bodies into the new CFG blocks.
    Block &afterRegionBlock = op.getAfter().front();
    rewriter.mergeBlocks(&op.getBefore().front(), headerBlock,
                         headerBlock->getArguments());
    rewriter.mergeBlocks(&afterRegionBlock, bodyBlock, bodyBlock->getArguments());
    rewriteWhileTerminators(rewriter, loc, headerBlock, bodyBlock,
                            afterWhileBlock);

    // Replace scf.while itself with a branch to the header.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, headerBlock, op.getInits());
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower cf.switch into chained comparisons and cf.cond_br/cf.br.
//
// EmitC C++ translation currently supports cf.br/cf.cond_br, but not cf.switch.
struct CFSwitchToCondBr : public OpRewritePattern<cf::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static SmallVector<SmallVector<Value>>
  collectSwitchCaseOperands(cf::SwitchOp op) {
    SmallVector<SmallVector<Value>> caseOperands;
    caseOperands.reserve(op.getCaseDestinations().size());
    for (auto range : op.getCaseOperands())
      caseOperands.emplace_back(range.begin(), range.end());
    return caseOperands;
  }

  static SmallVector<APInt> getSwitchCaseValues(cf::SwitchOp op) {
    SmallVector<APInt> caseValues;
    if (auto caseValuesAttr = op.getCaseValues()) {
      for (APInt value : caseValuesAttr->getValues<APInt>())
        caseValues.push_back(value);
    }
    return caseValues;
  }

  static SmallVector<Block *> createSwitchCheckBlocks(PatternRewriter &rewriter,
                                                      Region *parentRegion,
                                                      Block *curBlock,
                                                      size_t numCases) {
    auto insertPt = std::next(curBlock->getIterator());
    SmallVector<Block *> checkBlocks;
    checkBlocks.reserve(numCases);
    for (size_t i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
    return checkBlocks;
  }

  static LogicalResult populateSwitchCheckBlocks(
      PatternRewriter &rewriter, Location loc, Value flag, IntegerType flagTy,
      ArrayRef<APInt> caseValues, ArrayRef<Block *> caseDests,
      ArrayRef<SmallVector<Value>> caseOperands, Block *defaultDest,
      ValueRange defaultOperands, ArrayRef<Block *> checkBlocks,
      cf::SwitchOp op) {
    for (size_t i = 0; i < caseDests.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      APInt caseVal = caseValues[i];
      if (caseVal.getBitWidth() != flagTy.getWidth()) {
        return rewriter.notifyMatchFailure(
            op, "case value bitwidth doesn't match flag type");
      }

      Value caseConst = rewriter.create<arith::ConstantOp>(
          loc, flagTy, rewriter.getIntegerAttr(flagTy, caseVal));
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, flag, caseConst);
      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultDest;
      ValueRange falseOperands =
          (i + 1 < checkBlocks.size()) ? ValueRange{} : defaultOperands;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseDests[i],
                                        caseOperands[i], falseDest,
                                        falseOperands);
    }
    return success();
  }

  LogicalResult matchAndRewrite(cf::SwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower cf.switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    Value flag = op.getFlag();
    auto flagTy = dyn_cast<IntegerType>(flag.getType());
    if (!flagTy)
      return rewriter.notifyMatchFailure(op, "expected integer switch flag");

    SmallVector<Value> defaultOperands(op.getDefaultOperands().begin(),
                                       op.getDefaultOperands().end());
    Block *defaultDest = op.getDefaultDestination();

    SmallVector<Block *> caseDests(op.getCaseDestinations().begin(),
                                   op.getCaseDestinations().end());
    SmallVector<SmallVector<Value>> caseOperands = collectSwitchCaseOperands(op);

    if (caseDests.empty()) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(op, defaultDest, defaultOperands);
      return success();
    }

    if (!op.getCaseValues())
      return rewriter.notifyMatchFailure(op, "missing case_values");
    SmallVector<APInt> caseValues = getSwitchCaseValues(op);

    if (caseValues.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_values/destinations mismatch");
    if (caseOperands.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_operands/destinations mismatch");

    SmallVector<Block *> checkBlocks =
        createSwitchCheckBlocks(rewriter, parentRegion, curBlock,
                                caseDests.size());
    if (failed(populateSwitchCheckBlocks(rewriter, loc, flag, flagTy,
                                         caseValues, caseDests, caseOperands,
                                         defaultDest, defaultOperands,
                                         checkBlocks, op))) {
      return failure();
    }

    // Replace the switch terminator with a branch into the first check block.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, checkBlocks.front(),
                                              ValueRange{});
    return success();
  }
};

} // namespace


} // namespace

LogicalResult runPTOToEmitCSCFPreLowering(ModuleOp mop, MLIRContext *ctx) {
  bool needsAnySCFToCF = false;
  for (auto func : mop.getOps<func::FuncOp>()) {
    if (needsWholeFunctionSCFToCF(func)) {
      needsAnySCFToCF = true;
      break;
    }
  }
  if (needsAnySCFToCF) {
    RewritePatternSet scfToCfPatterns(ctx);
    populateSCFToControlFlowConversionPatterns(scfToCfPatterns);
    FrozenRewritePatternSet frozenSCFToCF(std::move(scfToCfPatterns));

    ConversionTarget scfToCfTarget(*ctx);
    scfToCfTarget.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                               scf::ParallelOp>();
    scfToCfTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });

    for (auto func : mop.getOps<func::FuncOp>()) {
      if (!needsWholeFunctionSCFToCF(func))
        continue;
      if (failed(applyPartialConversion(func, scfToCfTarget,
                                        frozenSCFToCF))) {
        func.emitError()
            << "failed to lower nested SCF to ControlFlow (SCFToCF)";
        return failure();
      }
    }
  }

  RewritePatternSet scfLoweringPatterns(ctx);
  scfLoweringPatterns.add<SCFExecuteRegionInline, SCFExecuteRegionToCF,
                          SCFIndexSwitchToCF, SCFWhileToCF, CFSwitchToCondBr>(ctx);
  (void)applyPatternsAndFoldGreedily(mop, std::move(scfLoweringPatterns));

  bool hasUnsupportedSCF = false;
  mop.walk([&](Operation *op) {
    if (isa<scf::ExecuteRegionOp, scf::IndexSwitchOp, scf::WhileOp>(op)) {
      hasUnsupportedSCF = true;
      op->emitError() << "Unsupported SCF op remained after pre-lowering";
      return WalkResult::interrupt();
    }
    if (isa<cf::SwitchOp>(op)) {
      hasUnsupportedSCF = true;
      op->emitError()
          << "Unsupported CF op remained after pre-lowering: cf.switch";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(hasUnsupportedSCF);
}

void populatePTOToEmitCControlFlowPatterns(RewritePatternSet &patterns,
                                           TypeConverter &typeConverter,
                                           MLIRContext *ctx) {
  patterns.add<SectionToEmitC<pto::SectionCubeOp>>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionVectorOp>>(typeConverter, ctx);
  patterns.add<CallToEmitC, ReturnToEmitC>(typeConverter, ctx);
  populateSCFToEmitCConversionPatterns(patterns);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
}

} // namespace mlir::pto
