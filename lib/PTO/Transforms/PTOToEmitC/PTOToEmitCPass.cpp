// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCPass.cpp - pass entry + pattern registration ----------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCPatterns.h"

#include "mlir/Target/Cpp/CppEmitter.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_EMITPTOMANUAL
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace {



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

  static void addContinuationArguments(PatternRewriter &rewriter,
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


  // Clone each case body (and the default body) into its target block and
  // branch to the continuation block.
  static LogicalResult cloneIndexSwitchBodies(PatternRewriter &rewriter,
                                              Location loc,
                                              scf::IndexSwitchOp op,
                                              ArrayRef<Block *> caseBlocks,
                                              Block *defaultBlock,
                                              Block *continueBlock) {
    for (unsigned i = 0; i < caseBlocks.size(); ++i) {
      if (failed(cloneYieldingBlockAndBranchTo(
              rewriter, loc, op.getCaseBlock(i), caseBlocks[i], continueBlock)))
        return failure();
    }
    return cloneYieldingBlockAndBranchTo(rewriter, loc, op.getDefaultBlock(),
                                         defaultBlock, continueBlock);
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

    if (failed(cloneIndexSwitchBodies(rewriter, loc, op, caseBlocks,
                                      defaultBlock, continueBlock)))
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Replace the original switch op with a branch into the check chain.
    Block *entryDest = numCases ? checkBlocks[0] : defaultBlock;
    rewriter.setInsertionPointAfter(op);
    rewriter.create<cf::BranchOp>(loc, entryDest, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.while into CFG blocks with cf.br/cf.cond_br.
//
// The SCF-to-ControlFlow pre-pass may already have split nested regions into
// multiple blocks. This pattern therefore moves complete regions, rather than
// assuming that each region still consists of one block.
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

  static void addWhileExitArguments(PatternRewriter &rewriter, scf::WhileOp op,
                                    Location loc, Block *afterWhileBlock) {
    SmallVector<Value> exitArgs;
    exitArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes()) {
      exitArgs.push_back(afterWhileBlock->addArgument(type, loc));
    }
    for (auto result : llvm::enumerate(op.getResults())) {
      result.value().replaceAllUsesWith(exitArgs[result.index()]);
    }
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

    // SCFToControlFlow may already have lowered nested scf.if/scf.for ops in
    // either region, leaving the region with several blocks. Move all of the
    // blocks into the parent CFG instead of merging only the entry block.
    // This also keeps existing cf.br/cf.cond_br edges intact.
    SmallVector<Block *> beforeBlocks;
    SmallVector<Block *> afterBlocks;
    for (Block &block : op.getBefore())
      beforeBlocks.push_back(&block);
    for (Block &block : op.getAfter())
      afterBlocks.push_back(&block);
    if (beforeBlocks.empty() || afterBlocks.empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty while regions");

    Block *beforeEntry = beforeBlocks.front();
    Block *afterEntry = afterBlocks.front();
    Region *parentRegion = afterWhileBlock->getParent();
    rewriter.inlineRegionBefore(op.getAfter(), *parentRegion,
                                afterWhileBlock->getIterator());
    rewriter.inlineRegionBefore(op.getBefore(), *parentRegion,
                                afterWhileBlock->getIterator());

    if (failed(wireWhileControlFlow(rewriter, op, loc, beforeBlocks,
                                   afterBlocks, beforeEntry, afterEntry,
                                   afterWhileBlock)))
      return failure();
    return success();
  }

  // Rebuild the loop CFG: the scf.condition becomes a conditional branch
  // into the after region / exit block, and scf.yield terminators become
  // back edges to the header.
  static LogicalResult wireWhileControlFlow(
      PatternRewriter &rewriter, scf::WhileOp op, Location loc,
      ArrayRef<Block *> beforeBlocks, ArrayRef<Block *> afterBlocks,
      Block *beforeEntry, Block *afterEntry, Block *afterWhileBlock) {
    // The before region has one scf.condition terminator. Its true edge enters
    // the after region and its false edge exits the loop with the carried
    // values. The after region's scf.yield terminator(s) form back edges.
    scf::ConditionOp condition;
    for (Block *block : beforeBlocks) {
      if (auto candidate = dyn_cast<scf::ConditionOp>(block->getTerminator())) {
        if (condition)
          return rewriter.notifyMatchFailure(
              op, "expected exactly one scf.condition in the before region");
        condition = candidate;
      }
    }
    if (!condition)
      return rewriter.notifyMatchFailure(op,
                                         "expected scf.condition terminator");

    rewriter.setInsertionPoint(condition);
    rewriter.create<cf::CondBranchOp>(
        loc, condition.getCondition(), afterEntry, condition.getArgs(),
        afterWhileBlock, condition.getArgs());
    rewriter.eraseOp(condition);

    for (Block *block : afterBlocks) {
      auto yield = dyn_cast<scf::YieldOp>(block->getTerminator());
      if (!yield) {
        if (isa<cf::BranchOp, cf::CondBranchOp>(block->getTerminator()))
          continue;
        return rewriter.notifyMatchFailure(
            op, "expected scf.yield or control-flow terminator");
      }
      rewriter.setInsertionPoint(yield);
      rewriter.create<cf::BranchOp>(loc, beforeEntry, yield.getOperands());
      rewriter.eraseOp(yield);
    }

    // Replace scf.while itself with a branch to the header.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, beforeEntry, op.getInits());
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

namespace mlir {
namespace pto {

bool isTriviallyInlineableExecuteRegion(scf::ExecuteRegionOp op) {
  Region &r = op.getRegion();
  if (!r.hasOneBlock())
    return false;
  Block &b = r.front();
  return isa_and_nonnull<scf::YieldOp>(b.getTerminator());
}

bool needsWholeFunctionSCFToCF(func::FuncOp func) {
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

    // SCFToControlFlow must see the whole function for while-like control
    // flow.  A while may be nested below an scf.for/scf.if region even when
    // the immediate parent operation does not advertise SingleBlock.  Running
    // the conversion only on the top-level function also lets it lower the
    // enclosing SCF regions in a consistent order and avoids leaving a while
    // behind for the local single-block-sensitive fallback below.
    if (isa<scf::WhileOp, scf::IndexSwitchOp>(op)) {
      needs = true;
      return WalkResult::interrupt();
    }

    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      needs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needs;
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {

void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx, PTOArch targetArch) {
  populateArithPatterns(patterns, typeConverter, ctx, targetArch);
  populateMemrefPatterns(patterns, typeConverter, ctx, targetArch);
  populateLoadStorePatterns(patterns, typeConverter, ctx, targetArch);
  populateSyncCommPatterns(patterns, typeConverter, ctx, targetArch);
  populateScalarMiscPatterns(patterns, typeConverter, ctx, targetArch);
  populateTilePatterns(patterns, typeConverter, ctx, targetArch);
  populateTensorPatterns(patterns, typeConverter, ctx, targetArch);
  populateTensorReducePatterns(patterns, typeConverter, ctx, targetArch);

  populateSCFToEmitCConversionPatterns(patterns);
  // Keep CFG-style branches type-consistent when block argument types are
  // converted (e.g. after lowering scf.while to cf.br/cf.cond_br).
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
}

} // namespace pto
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct EmitPTOManualPass
    : public PassWrapper<EmitPTOManualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPTOManualPass)

  PTOArch targetArch;

  EmitPTOManualPass() : targetArch(PTOArch::A3) {}

  explicit EmitPTOManualPass(PTOArch arch) : targetArch(arch) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    mlir::cf::ControlFlowDialect, mlir::pto::PTODialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Start PTOToEmitC Pass\n");
    MLIRContext *ctx = &getContext();
    ModuleOp mop = getOperation();

    if (failed(validateAndAnnotate(mop)))
      return signalPassFailure();

    emitPreambleHelpers(mop, ctx);

    if (failed(preLowerSCF(mop, ctx)))
      return signalPassFailure();

    PTOToEmitCTypeConverter typeConverter(ctx, targetArch);
    if (failed(convertStructuralTypes(mop, ctx, typeConverter)))
      return signalPassFailure();
    if (failed(runMainConversion(mop, ctx, typeConverter)))
      return signalPassFailure();
    if (failed(cleanupConversionCasts(mop, ctx, typeConverter)))
      return signalPassFailure();
    eraseDeadPureEmitCValueOps(mop);
    fixLoopInductionVariables(mop);
    eraseDeadTileVariables(mop);
    eraseDeadConstants(mop);
  }

  LogicalResult validateAndAnnotate(ModuleOp mop) {
    if (failed(pto::validatePTOEntryFunctions(mop)))
  return failure();
    if (failed(pto::validateStructProvenance(mop)))
  return failure();
    pto::annotatePTOEntryFunctions(mop);

    // A3 requires explicit FFTS base setup for inter-core sync ops.
    if (targetArch == PTOArch::A3) {
  bool hasMissingSetFFTs = false;
  for (auto func : mop.getOps<func::FuncOp>()) {
    if (!hasInterCoreSyncOp(func))
      continue;
    if (hasSetFFTsOp(func))
      continue;
    hasMissingSetFFTs = true;
    func.emitError()
        << "A3 inter-core sync requires explicit `pto.set_ffts` in the "
           "same function when using `pto.sync.set`/`pto.sync.wait`";
  }
  if (hasMissingSetFFTs)
    return failure();
    }
    return success();
  }

  // Walk the module to decide which runtime helper snippets are needed and
  // emit the pto-inst include plus helper definitions at module start.
  void emitPreambleHelpers(ModuleOp mop, MLIRContext *ctx) {
    HelperFlags flags = collectHelperFlags(mop);

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(mop.getBody());
    emitPreambleIncludesAndStructs(mop, builder, flags);
    emitPreambleRuntimeHelpers(mop, builder, flags);
    emitPreambleTailHelpers(mop, builder, flags);
  }

  // Which runtime helper snippets the module needs, decided by walking ops.
  struct HelperFlags {
    bool eventIdArray = false;
    bool tRandom = false;
    bool globalTensorData = false;
    bool bitcast = false;
  };

  static HelperFlags collectHelperFlags(ModuleOp mop) {
    HelperFlags flags;
    mop.walk([&](Operation *op) {
      if (isa<mlir::pto::DeclareEventIdArrayOp>(op))
        flags.eventIdArray = true;
      if (isa<mlir::pto::TRandomOp>(op))
        flags.tRandom = true;
      if (auto cmo = dyn_cast<mlir::pto::CmoCacheInvalidOp>(op)) {
        if (cmo.getAddr())
          flags.globalTensorData = true;
      }
      if (auto init = dyn_cast<mlir::pto::InitializeL2G2LPipeOp>(op)) {
        if (isa<mlir::pto::TensorViewType>(init.getGmAddr().getType()))
          flags.globalTensorData = true;
      }
      if (isa<mlir::pto::PartitionViewOp>(op))
        flags.globalTensorData = true;
      if (isa<arith::BitcastOp, arith::MaximumFOp, arith::MinimumFOp>(op))
        flags.bitcast = true;
    });
    return flags;
  }

  // Emit the pto-inst include, the pto namespace using-declaration, and the
  // file-scope struct definitions in dependency order.
  static void emitPreambleIncludesAndStructs(ModuleOp mop, OpBuilder &builder,
                                             const HelperFlags &flags) {
    Location loc = mop->getLoc();
    builder.create<emitc::IncludeOp>(
        loc, "pto/pto-inst.hpp", /*is_standard_include=*/false);
    builder.create<emitc::VerbatimOp>(
        loc, builder.getStringAttr("using namespace pto;"));

    llvm::SetVector<pto::StructType> structDefs;
    mop.walk([&](Operation *op) {
      for (Type t : op->getResultTypes())
        collectStructTypes(t, structDefs);
      for (Value v : op->getOperands())
        collectStructTypes(v.getType(), structDefs);
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        for (Type t : func.getArgumentTypes())
          collectStructTypes(t, structDefs);
        for (Type t : func.getResultTypes())
          collectStructTypes(t, structDefs);
      }
    });
    for (pto::StructType st : structDefs)
      builder.create<emitc::VerbatimOp>(
          loc, builder.getStringAttr(renderStructDef(st)));
    (void)flags;
  }

  // Emit the optional runtime helper snippets (global-tensor data accessor,
  // event-id array, TRandom, auto-sync tail, bitcast) needed by this module.
  static void emitPreambleRuntimeHelpers(ModuleOp mop, OpBuilder &builder,
                                         const HelperFlags &flags) {
    Location loc = mop->getLoc();
    if (flags.globalTensorData) {
      builder.create<emitc::VerbatimOp>(
          loc, builder.getStringAttr(R"cpp(
template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}
)cpp"));
    }
    if (flags.eventIdArray) {
      builder.create<emitc::VerbatimOp>(
          loc, builder.getStringAttr(R"cpp(
template <int N>
struct PTOAS_EventIdArray {
  static_assert(N > 0, "PTOAS_EventIdArray requires a positive static size");
  int32_t data[N] = {};

  AICORE inline int32_t &operator[](int32_t idx) { return data[idx]; }
  AICORE inline const int32_t &operator[](int32_t idx) const { return data[idx]; }
};
)cpp"));
    }
    if (flags.tRandom) {
      builder.create<emitc::VerbatimOp>(
          loc, builder.getStringAttr(R"cpp(
template <uint16_t Rounds, typename DstTile>
static AICORE inline void PTOAS__TRANDOM(
    DstTile &dst, uint32_t key0, uint32_t key1, uint32_t counter0,
    uint32_t counter1, uint32_t counter2, uint32_t counter3) {
  TRandomKey key = {key0, key1};
  TRandomCounter counter = {counter0, counter1, counter2, counter3};
  TRANDOM<Rounds>(dst, key, counter);
}
)cpp"));
    }
  }

  // Emit the always-present auto-sync tail helper plus the optional
  // bitcast helper.
  static void emitPreambleTailHelpers(ModuleOp mop, OpBuilder &builder,
                                      const HelperFlags &flags) {
    Location loc = mop->getLoc();
    builder.create<emitc::VerbatimOp>(
        loc, builder.getStringAttr(R"cpp(
enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}
)cpp"));
    if (flags.bitcast) {
      builder.create<emitc::VerbatimOp>(
          loc, builder.getStringAttr(R"cpp(
template <typename To, typename From>
static inline To ptoas_bitcast(From from) {
  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
  To to;
  __builtin_memcpy(&to, &from, sizeof(To));
  return to;
}
)cpp"));
    }
  }


  // Pre-lower SCF constructs not handled by SCFToEmitC into supported forms.
  // Lower whole functions whose SCF nesting cannot stay single-block via
  // SCFToControlFlow. Returns failure when the partial conversion fails.
  LogicalResult lowerWholeFunctionSCF(ModuleOp mop, MLIRContext *ctx,
                                      SmallVectorImpl<func::FuncOp> &functions) {
    bool needsAnySCFToCF = false;
    for (func::FuncOp func : functions) {
      if (needsWholeFunctionSCFToCF(func)) {
        needsAnySCFToCF = true;
        break;
      }
    }
    if (!needsAnySCFToCF)
      return success();

    RewritePatternSet scfToCfPatterns(ctx);
    populateSCFToControlFlowConversionPatterns(scfToCfPatterns);
    FrozenRewritePatternSet frozenSCFToCF(std::move(scfToCfPatterns));

    ConversionTarget scfToCfTarget(*ctx);
    // Only eliminate the single-block SCF constructs; we'll pre-lower
    // scf.while/index_switch/execute_region ourselves afterwards.
    scfToCfTarget.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                               scf::ParallelOp, scf::WhileOp>();
    scfToCfTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });

    for (func::FuncOp func : functions) {
      if (!needsWholeFunctionSCFToCF(func))
        continue;
      if (failed(applyPartialConversion(func, scfToCfTarget, frozenSCFToCF))) {
        func.emitError()
            << "failed to lower nested SCF to ControlFlow (SCFToCF)";
        return failure();
      }
    }
    return success();
  }

  // Verify no SCF/CF op that EmitC cannot print survived pre-lowering.
  static LogicalResult verifyNoUnsupportedSCF(ModuleOp mop) {
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

  LogicalResult preLowerSCF(ModuleOp mop, MLIRContext *ctx) {
    SmallVector<func::FuncOp> functions;
    mop.walk([&](func::FuncOp func) { functions.push_back(func); });
    if (failed(lowerWholeFunctionSCF(mop, ctx, functions)))
      return failure();

    RewritePatternSet scfLoweringPatterns(ctx);
    scfLoweringPatterns.add<SCFExecuteRegionInline, SCFExecuteRegionToCF,
                            SCFIndexSwitchToCF,
                            SCFWhileToCF, CFSwitchToCondBr>(ctx);
    (void)applyPatternsAndFoldGreedily(mop, std::move(scfLoweringPatterns));

    return verifyNoUnsupportedSCF(mop);
  }

  LogicalResult convertStructuralTypes(ModuleOp mop, MLIRContext *ctx,
                                       TypeConverter &typeConverter) {
    // 2. Pre-convert SCF structural op types (e.g. scf.if/scf.for results)
    // using the same type converter. This avoids creating emitc.variable with
    // unsupported types such as memref.
    {
  RewritePatternSet scfTypePatterns(ctx);
  ConversionTarget scfTypeTarget(*ctx);
  scf::populateSCFStructuralTypeConversionsAndLegality(
      typeConverter, scfTypePatterns, scfTypeTarget);
  scfTypeTarget.markUnknownOpDynamicallyLegal(
      [](Operation *) { return true; });

  if (failed(applyPartialConversion(mop, scfTypeTarget,
                                    std::move(scfTypePatterns)))) {
    mop.emitError("failed to reconcile SCF structural types");
    return failure();
  }
    }

    if (failed(rematerializeFixpipeQuantBindings(mop))) {
  mop.emitError("failed to rematerialize fixpipe quant bindings");
  return failure();
    }
    if (failed(insertFixpipeConfigAliases(mop))) {
  mop.emitError("failed to insert fixpipe config aliases");
  return failure();
    }

    return success();
  }

  LogicalResult runMainConversion(ModuleOp mop, MLIRContext *ctx,
                                  PTOToEmitCTypeConverter &typeConverter) {
    // 3. 配置转换目标
    ConversionTarget target(*ctx);

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>();

    // If we introduced CFG branches (e.g. from scf.while), make sure they are
    // updated to use legalized operand types.
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return isLegalForBranchOpInterfaceTypeConversionPattern(
              op, typeConverter);
        });

    // [关键] 允许 Cast 存在，最后统一清理
    target.addLegalOp<UnrealizedConversionCastOp>();

    target.addIllegalOp<func::ReturnOp>();
    target.addIllegalOp<func::FuncOp>();
    target.addIllegalOp<func::CallOp>();

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    populatePTOToEmitCPatterns(patterns, typeConverter, ctx, targetArch);

    // 4. 执行转换
    if (failed(applyPartialConversion(mop, target, std::move(patterns)))) {
  llvm::errs() << "Conversion FAILED! Rolling back executed.\n";
  return failure();
    }

    {
  SmallVector<pto::MakeTensorViewOp> deadStaticMakeViews;
  mop.walk([&](pto::MakeTensorViewOp op) {
    if (op->use_empty())
      deadStaticMakeViews.push_back(op);
  });
  for (pto::MakeTensorViewOp op : deadStaticMakeViews)
    op.erase();
    }

    // =========================================================================
    // 5. [终极清理] 
    // 顺序至关重要：
    // Step A: 先移除所有 Cast，让 Loop 的 Operand 类型变成底层类型 (如 int32)
    // Step B: 再根据新的 Operand 类型，修复 Loop IV 的类型
    // =========================================================================
    return success();
  }

  // Step A/A2/A3: lower or drop leftover UnrealizedConversionCast ops and
  // re-materialize variable reads at their use sites.
  LogicalResult cleanupConversionCasts(ModuleOp mop, MLIRContext *ctx,
                                       TypeConverter &typeConverter) {
    (void)ctx;
    if (failed(lowerUnrealizedCasts(mop, typeConverter)))
      return failure();
    sinkVariableReadCasts(mop);
    sinkTileDataReads(mop);
    return success();
  }

  // Step A: drop or lower leftover UnrealizedConversionCast ops so the C++
  // emitter can print them.
  // Classify and lower one unrealized_conversion_cast into either a value
  // replacement or an emitc.cast; returns failure when the cast cannot be
  // lowered.
  LogicalResult lowerSingleCast(UnrealizedConversionCastOp cast,
                                TypeConverter &typeConverter) const {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
      cast.emitError() << "unsupported unrealized_conversion_cast shape";
      return failure();
    }

    Value input = cast.getOperand(0);
    Value output = cast.getResult(0);
    Type inTy = input.getType();
    Type outTy = output.getType();

    // Dead or identity/bridge casts whose input already carries the lowered
    // value: drop or fold the cast away by forwarding the input.
    if (isFoldableBridgeCast(output, inTy, outTy, typeConverter)) {
      output.replaceAllUsesWith(input);
      return success();
    }

    // Tile-backed pointer extraction must lower via PTOAS__TILE_DATA.
    if (isEmitCTileLikeType(inTy) && isEmitCPointerLikeType(outTy)) {
      OpBuilder builder(cast);
      auto extracted = builder.create<emitc::CallOpaqueOp>(
          cast.getLoc(), outTy, "PTOAS__TILE_DATA", ArrayAttr{},
          ArrayAttr{}, ValueRange{input});
      output.replaceAllUsesWith(extracted.getResult(0));
      return success();
    }

    if (emitc::isSupportedEmitCType(inTy) && emitc::isSupportedEmitCType(outTy)) {
      OpBuilder builder(cast);
      auto c = builder.create<emitc::CastOp>(cast.getLoc(), outTy, input);
      output.replaceAllUsesWith(c.getResult());
      return success();
    }

    cast.emitError() << "cannot lower unrealized_conversion_cast(" << inTy
                     << " -> " << outTy << ") to emitc.cast";
    return failure();
  }

  // Whether the cast result is dead or the cast is a type-conversion bridge
  // whose input already carries the lowered value: it can simply be replaced
  // by its input.
  bool isFoldableBridgeCast(Value output, Type inTy, Type outTy,
                            TypeConverter &typeConverter) const {
    if (output.use_empty() || inTy == outTy) {
      return true;
    }
    // IndexType is lowered to int64_t for EmitC; fold index<->int64 bridges.
    if (isa<IndexType>(inTy) && isLoweredIndexType(outTy)) {
      return true;
    }
    if (isLoweredIndexType(inTy) && isa<IndexType>(outTy)) {
      return true;
    }
    // SCF/CFG type conversion can transiently materialize pointer->memref and
    // tile->tile_buf bridges; the EmitC form is the value we keep.
    if (isEmitCPointerLikeType(inTy) && isa<BaseMemRefType>(outTy)) {
      return true;
    }
    if (isEmitCTileLikeType(inTy) && isa<pto::TileBufType>(outTy)) {
      return true;
    }
    // The converted output type equals the input type: the cast is a no-op.
    Type convertedOutTy = typeConverter.convertType(outTy);
    return convertedOutTy && convertedOutTy == inTy;
  }

  static bool isLoweredIndexType(Type ty) {
    auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
    return opaqueTy && opaqueTy.getValue() == "int64_t";
  }

  LogicalResult lowerUnrealizedCasts(ModuleOp mop,
                                     TypeConverter &typeConverter) {
    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    bool castCleanupFailed = false;
    mop.walk([&](UnrealizedConversionCastOp cast) {
      if (castCleanupFailed)
        return;

      if (failed(lowerSingleCast(cast, typeConverter))) {
        castCleanupFailed = true;
        return;
      }
      castsToErase.push_back(cast);
    });

    for (auto cast : castsToErase)
      cast.erase();

    return failure(castCleanupFailed);
  }

  // Step A2: re-materialize casts of emitc.variable reads at each use site so
  // they observe the latest assignment instead of snapshotting the initial
  // value.
  static void sinkVariableReadCasts(ModuleOp mop) {
    // --- Step A2: Sink casts of emitc.variable "reads" to their use sites ---
    //
    // SCFToEmitC lowers scf.if/scf.for results via mutable `emitc.variable` and
    // `emitc.assign`. During type conversion, casts from the variable handle to
    // the converted type may be materialized right after the variable
    // declaration, effectively snapshotting the value *before* assignments. That
    // produces wrong C++ (use-before-init / stale reads).
    //
    // Fix by re-materializing the cast at each use site so it reads the variable
    // at the point of use.
    {
  SmallVector<emitc::CastOp> castOpsToSink;
  mop.walk([&](emitc::CastOp castOp) {
    if (castOp.getSource().getDefiningOp<emitc::VariableOp>())
      castOpsToSink.push_back(castOp);
  });

  for (emitc::CastOp castOp : castOpsToSink) {
    Value src = castOp.getSource();
    Type dstTy = castOp.getResult().getType();
    Value oldRes = castOp.getResult();

    // Replace each use with a freshly inserted cast right before the user.
    for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
      Operation *user = use.getOwner();
      OpBuilder b(user);
      b.setInsertionPoint(user);
      auto newCast = b.create<emitc::CastOp>(castOp.getLoc(), dstTy, src);
      use.set(newCast.getResult());
    }

    castOp.erase();
  }
    }
  }

  // Step A3: re-materialize PTOAS__TILE_DATA reads of emitc.variable at each
  // use site so they observe the post-TASSIGN backing address.
  static void sinkTileDataReads(ModuleOp mop) {
    // --- Step A3: Sink PTOAS__TILE_DATA reads of emitc.variable to use sites ---
    //
    // Tile-like emitc.variable values are mutable handles whose backing address
    // is typically established by a later `TASSIGN`. If we materialize
    // `PTOAS__TILE_DATA(tileVar)` right after declaration, we snapshot an
    // uninitialized/stale address. Re-materialize each read at the use site so
    // it observes the post-TASSIGN state of the tile variable.
    {
  SmallVector<emitc::CallOpaqueOp> tileDataReadsToSink;
  mop.walk([&](emitc::CallOpaqueOp callOp) {
    if (callOp.getCallee() != "PTOAS__TILE_DATA")
      return;
    if (callOp.getNumOperands() != 1 || callOp.getNumResults() != 1)
      return;
    if (getSourceEmitCVariable(callOp.getOperand(0)))
      tileDataReadsToSink.push_back(callOp);
  });

  for (emitc::CallOpaqueOp callOp : tileDataReadsToSink) {
    Value src = callOp.getOperand(0);
    Type dstTy = callOp.getResult(0).getType();
    Value oldRes = callOp.getResult(0);

    for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
      Operation *user = use.getOwner();
      OpBuilder b(user);
      b.setInsertionPoint(user);
      auto newRead = b.create<emitc::CallOpaqueOp>(
          callOp.getLoc(), dstTy, "PTOAS__TILE_DATA", ArrayAttr{},
          ArrayAttr{}, ValueRange{src});
      use.set(newRead.getResult(0));
    }

    callOp.erase();
    }
  }
  }


  // Step B: keep emitc.for induction-variable types in sync with bounds.
  void fixLoopInductionVariables(ModuleOp mop) {
    // --- Step B: 修复 Loop 归纳变量 (IV) ---
    // 此时 emitc.for 的 operand 已经是 int32 了，我们检查 IV 是否匹配，不匹配则修正
    mop.walk([&](emitc::ForOp forOp) {
   Type boundTy = forOp.getLowerBound().getType(); 
   BlockArgument iv = forOp.getBody()->getArgument(0); 
   
   if (iv.getType() != boundTy) {
     iv.setType(boundTy); // 强制将 IV 类型 (index) 修改为与边界一致 (int32)
   }
    });
  }

  // Step C: remove tile variables that are never read (and their TASSIGNs).
  void eraseDeadTileVariables(ModuleOp mop) {
    // --- Step C: 消除冗余 Tile 变量 (Dead Code Elimination) [新增] ---
    // 逻辑：如果一个 emitc.variable 没有被读取（use_empty），
    // 那么它自己，以及给它赋值的 TASSIGN 都可以删除。
    // 注意：TASSIGN(v15, v9) 会把 v15 作为 Operand 0 使用，所以 v15 不是严格的 use_empty。
    // 我们需要检查：v15 是否除了 TASSIGN 之外没有其他 User。

    llvm::SmallVector<emitc::VariableOp> deadVars;
    mop.walk([&](emitc::VariableOp varOp) {
    // 检查该变量的所有 User
    bool isRead = false;
    for (Operation* user : varOp.getResult().getUsers()) {
        // 如果 User 是 TASSIGN 且变量是第0个参数(dst)，不算"读取"
        if (auto call = dyn_cast<emitc::CallOpaqueOp>(user)) {
            if (call.getCallee() == "TASSIGN" && call.getOperand(0) == varOp.getResult()) {
                continue; // 这是一个赋值操作，不算有效使用
            }
            if (call.getCallee() == "PTOAS__TILE_DATA" &&
                call.getNumResults() == 1 &&
                call.getResult(0).use_empty())
                continue;
        }
        // 如果还有其他用途（如 TLOAD, TMOV, TMATMUL），则该变量有用
        isRead = true;
        break;
    }

    if (!isRead) {
        deadVars.push_back(varOp);
    }
    });

    for (auto varOp : deadVars) {
    // 1. 先删除所有使用该变量的 TASSIGN
    llvm::SmallVector<Operation*> usersToErase;
    for (Operation* user : varOp.getResult().getUsers()) {
         // 上面已经确认过，剩下的 user 只能是 TASSIGN 或无使用的
         // PTOAS__TILE_DATA。
         usersToErase.push_back(user);
    }
    for (auto u : usersToErase) u->erase();

    // 2. 删除变量定义本身
    varOp.erase();
    }
  }

  void eraseDeadConstants(ModuleOp mop) {
    llvm::SmallVector<emitc::ConstantOp> deadConsts;
    mop.walk([&](emitc::ConstantOp constOp) {
  if (constOp.getResult().use_empty())
    deadConsts.push_back(constOp);
    });
    for (auto constOp : deadConsts)
  constOp.erase();

    // =========================================================================
  }
  };
} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createEmitPTOManualPass() {
  return std::make_unique<EmitPTOManualPass>();
}

std::unique_ptr<Pass> createEmitPTOManualPass(PTOArch arch) {
  return std::make_unique<EmitPTOManualPass>(arch);
}

} // namespace pto
} // namespace mlir


