#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERINGSYNCTOPIPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {
static FailureOr<SyncOpType> getSyncOpTypeFromAttr(Attribute attr, Operation *op,
                                                   StringRef name) {
  auto opType = parseSyncOpTypeLikeAttr(attr);
  if (succeeded(opType))
    return *opType;
  auto diag =
      op->emitError("expected PipeEventTypeAttr or SyncOpTypeAttr for ");
  diag << name;
  return failure();
}

struct RecordEventLowering : public OpRewritePattern<RecordEventOp> {
  using OpRewritePattern<RecordEventOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RecordEventOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTypeOr = getSyncOpTypeFromAttr(op.getSrcOpAttr(), op, "src_op");
    if (failed(srcTypeOr))
      return failure();
    auto dstTypeOr = getSyncOpTypeFromAttr(op.getDstOpAttr(), op, "dst_op");
    if (failed(dstTypeOr))
      return failure();
    SyncOpType srcType = *srcTypeOr;
    SyncOpType dstType = *dstTypeOr;

    PIPE srcPipe = mapSyncOpTypeToPipe(srcType);
    PIPE dstPipe = mapSyncOpTypeToPipe(dstType);
    if (!isConcreteSyncPipe(srcPipe) || !isConcreteSyncPipe(dstPipe))
      return op.emitError("Failed to map SyncOpType to hardware pipe during lowering.");

    rewriter.replaceOpWithNewOp<SetFlagOp>(
        op, PipeAttr::get(op.getContext(), srcPipe),
        PipeAttr::get(op.getContext(), dstPipe), op.getEventIdAttr());
    return success();
  }
};

struct WaitEventLowering : public OpRewritePattern<WaitEventOp> {
  using OpRewritePattern<WaitEventOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitEventOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTypeOr = getSyncOpTypeFromAttr(op.getSrcOpAttr(), op, "src_op");
    if (failed(srcTypeOr))
      return failure();
    auto dstTypeOr = getSyncOpTypeFromAttr(op.getDstOpAttr(), op, "dst_op");
    if (failed(dstTypeOr))
      return failure();
    SyncOpType srcType = *srcTypeOr;
    SyncOpType dstType = *dstTypeOr;

    PIPE srcPipe = mapSyncOpTypeToPipe(srcType);
    PIPE dstPipe = mapSyncOpTypeToPipe(dstType);
    if (!isConcreteSyncPipe(srcPipe) || !isConcreteSyncPipe(dstPipe))
      return op.emitError("Failed to map SyncOpType to hardware pipe during lowering.");

    rewriter.replaceOpWithNewOp<WaitFlagOp>(
        op, PipeAttr::get(op.getContext(), srcPipe),
        PipeAttr::get(op.getContext(), dstPipe), op.getEventIdAttr());
    return success();
  }
};

// High-level barrier -> barrier with mapped pipe
struct BarrierSyncLowering : public OpRewritePattern<BarrierSyncOp> {
  using OpRewritePattern<BarrierSyncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierSyncOp op,
                                PatternRewriter &rewriter) const override {
    SyncOpType ty = op.getOpType().getOpType();
    PIPE pipe = mapSyncOpTypeToPipe(ty);
    if (!isConcreteSyncPipe(pipe)) {
      auto diag = op.emitError(
          "barrier_sync failed to map SyncOpType to hardware pipe during lowering: ");
      diag << op.getOpType();
      return failure();
    }

    // A5: TVEC single-pipe barrier is unnecessary/unsupported.
    if (pipe == PIPE::PIPE_V && isTargetArchA5(op.getOperation())) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOpWithNewOp<BarrierOp>(
        op, PipeAttr::get(op.getContext(), pipe));
    return success();
  }
};

// Legalize explicit low-level barriers for arch constraints.
struct BarrierLegalizeForArch : public OpRewritePattern<BarrierOp> {
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (isTargetArchA5(op.getOperation()) &&
        op.getPipe().getPipe() == PIPE::PIPE_V) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct LoweringSyncToPipe
    : public mlir::pto::impl::PTOLoweringSyncToPipeBase<LoweringSyncToPipe> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RecordEventLowering, WaitEventLowering, BarrierSyncLowering,
                 BarrierLegalizeForArch>(
        context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createLoweringSyncToPipePass() {
  return std::make_unique<LoweringSyncToPipe>();
}
