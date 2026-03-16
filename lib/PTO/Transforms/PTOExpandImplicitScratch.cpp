#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOEXPANDIMPLICITSCRATCH
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

static std::pair<Value, Value> inferDynamicValidDims(Value value) {
  while (value) {
    if (auto alloc = value.getDefiningOp<pto::AllocTileOp>())
      return {alloc.getValidRow(), alloc.getValidCol()};
    if (auto bitcast = value.getDefiningOp<pto::BitcastOp>()) {
      value = bitcast.getSrc();
      continue;
    }
    if (auto treshape = value.getDefiningOp<pto::TReshapeOp>()) {
      value = treshape.getSrc();
      continue;
    }
    break;
  }
  return {};
}

struct ExpandTXorPattern : public OpRewritePattern<pto::TXorOp> {
  using OpRewritePattern<pto::TXorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::TXorOp op,
                                PatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(
          op, "expected tile_buf dst before memref lowering");

    auto [validRow, validCol] = inferDynamicValidDims(op.getDst());
    Value tmp = rewriter.create<pto::AllocTileOp>(op.getLoc(), dstTy,
                                                  /*addr=*/Value(), validRow,
                                                  validCol);
    rewriter.replaceOpWithNewOp<pto::TXorWithTmpOp>(
        op, TypeRange{}, op.getSrc0(), op.getSrc1(), tmp, op.getDst());
    return success();
  }
};

struct ExpandTXorSPattern : public OpRewritePattern<pto::TXorSOp> {
  using OpRewritePattern<pto::TXorSOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::TXorSOp op,
                                PatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(
          op, "expected tile_buf dst before memref lowering");

    auto [validRow, validCol] = inferDynamicValidDims(op.getDst());
    Value tmp = rewriter.create<pto::AllocTileOp>(op.getLoc(), dstTy,
                                                  /*addr=*/Value(), validRow,
                                                  validCol);
    rewriter.replaceOpWithNewOp<pto::TXorSWithTmpOp>(
        op, TypeRange{}, op.getSrc(), op.getScalar(), tmp, op.getDst());
    return success();
  }
};

struct PTOExpandImplicitScratchPass
    : public mlir::pto::impl::PTOExpandImplicitScratchBase<
          PTOExpandImplicitScratchPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandTXorPattern, ExpandTXorSPattern>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOExpandImplicitScratchPass() {
  return std::make_unique<PTOExpandImplicitScratchPass>();
}
