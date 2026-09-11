// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- Sync.cpp - SyncComm Sync op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "../PTOToEmitCPatterns.h"
#include "SyncCommInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

template <typename FenceOp>
struct PTOFenceToEmitC : public OpConversionPattern<FenceOp> {
  using OpConversionPattern<FenceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FenceOp op, typename FenceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (op.getScope().getScope() != pto::FenceScope::GM &&
        op.getScope().getScope() != pto::FenceScope::All) {
      return rewriter.notifyMatchFailure(op, "unsupported fence scope");
    }

    if (isInVectorKernel(op)) {
      emitPipeBarrier(rewriter, op.getLoc(), "PIPE_ALL");
    } else {
      emitConservativeGmFencePipeDrains(rewriter, op.getLoc());
    }
    emitDsbDdr(rewriter, op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOBarrierToEmitC : public OpConversionPattern<pto::BarrierOp> {
  using OpConversionPattern<pto::BarrierOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->hasAttr(kAutoSyncTailBarrierAttr)) {
      auto modeAttr = rewriter.getStringAttr(getAutoSyncTailModeToken(op));
      if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
        emitcFunc->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
        funcOp->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // [FIX] op.getPipe() returns PipeAttr. 
    // We must call .getPipe() on the attribute to get the actual Enum value.
    pto::PIPE pipeEnum = op.getPipe().getPipe();

    // Convert Enum to String (e.g., PIPE_ALL -> "PIPE_ALL")
    std::string pipeStr = pto::stringifyPIPE(pipeEnum).str();
    auto *ctx = rewriter.getContext();

    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeStr)
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        TypeRange{},        // void return
        "pipe_barrier",     // function name
        args,               // arguments
        ArrayAttr{},        // template args
        ValueRange{}        // operands
    );

    return success();
  }
};

template <typename FlagOp>
struct PTOFlagToEmitC : public OpConversionPattern<FlagOp> {
  using OpConversionPattern<FlagOp>::OpConversionPattern;

  explicit PTOFlagToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                          StringRef callee)
      : OpConversionPattern<FlagOp>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(FlagOp op, typename FlagOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();
    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, argsAttr, ArrayAttr{}, ValueRange{});
    return success();
  }

  std::string callee;
};

using PTOSetFlagToEmitC = PTOFlagToEmitC<mlir::pto::SetFlagOp>;
using PTOWaitFlagToEmitC = PTOFlagToEmitC<mlir::pto::WaitFlagOp>;

struct PTOSyncToEmitC : public OpConversionPattern<mlir::pto::TSyncOp> {
  using OpConversionPattern<mlir::pto::TSyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::TSyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands;
    operands.reserve(adaptor.getEvents().size());
    for (Value event : adaptor.getEvents())
      operands.push_back(peelUnrealized(event));

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSYNC",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(operands));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncAllToEmitC : public OpConversionPattern<mlir::pto::SyncAllOp> {
  using OpConversionPattern<mlir::pto::SyncAllOp>::OpConversionPattern;

  static StringRef coreTypeTok(pto::SyncCoreType coreType) {
    switch (coreType) {
    case pto::SyncCoreType::AIVOnly:
      return "SyncCoreType::AIVOnly";
    case pto::SyncCoreType::AICOnly:
      return "SyncCoreType::AICOnly";
    case pto::SyncCoreType::Mix:
      return "SyncCoreType::Mix";
    }
    llvm_unreachable("unhandled SyncCoreType");
  }

  LogicalResult matchAndRewrite(mlir::pto::SyncAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mode = op.getMode().getValue();
    auto coreType = op.getCoreType().getValue();

    auto buildGmWorkspace = [&]() -> FailureOr<Value> {
      Value gm = adaptor.getGmWorkspace();
      if (isEmitCGlobalTensorLikeType(gm.getType()))
        return gm;

      auto ptrTy = dyn_cast<pto::PtrType>(op.getGmWorkspace().getType());
      if (!ptrTy)
        return failure();
      return buildSyncAllGlobalTensorFromPointer(
          rewriter, op.getLoc(), gm, ptrTy.getElementType());
    };

    if (mode == pto::SyncAllMode::Hard) {
      std::string callee = "SYNCALL<" + coreTypeTok(coreType).str() + ">";
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<Value> gmWorkspace = buildGmWorkspace();
    if (failed(gmWorkspace))
      return rewriter.notifyMatchFailure(op,
                                         "failed to build gm_workspace GlobalTensor");

    auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    Value usedCores =
        adaptor.getUsedCores()
            ? adaptor.getUsedCores()
            : rewriter
                  .create<emitc::LiteralOp>(op.getLoc(), i32Ty, "int32_t{0}")
                  .getResult();
    if (usedCores.getType() != i32Ty)
      usedCores = rewriter.create<emitc::CastOp>(op.getLoc(), i32Ty, usedCores)
                      .getResult();

    std::string callee =
        "SYNCALL<SyncAllMode::Soft, " + coreTypeTok(coreType).str() + ">";

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*gmWorkspace, usedCores});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncFlagDynToEmitC : public ConversionPattern {
  PTOSyncFlagDynToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                        StringRef opName, StringRef callee)
      : ConversionPattern(typeConverter, opName, /*benefit=*/1, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected exactly one dynamic event-id operand");

    auto srcAttr = op->getAttrOfType<mlir::pto::PipeAttr>("src_pipe");
    auto dstAttr = op->getAttrOfType<mlir::pto::PipeAttr>("dst_pipe");
    if (!srcAttr || !dstAttr)
      return rewriter.notifyMatchFailure(op, "missing PipeAttr src_pipe/dst_pipe attrs");

    auto *ctx = rewriter.getContext();
    std::string srcTok = pipeTokFromPipeAttr(srcAttr);
    std::string dstTok = pipeTokFromPipeAttr(dstAttr);

    Value eventVal = operands.front();
    eventVal =
        emitCCast(rewriter, op->getLoc(), emitc::OpaqueType::get(ctx, "event_t"), eventVal);

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee,
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{eventVal});
    return success();
  }

private:
  std::string callee;
};

struct PTOSetFFTsToEmitC : public OpConversionPattern<mlir::pto::SetFFTsOp> {
  using OpConversionPattern<mlir::pto::SetFFTsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFFTsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value fftsAddr = adaptor.getFfts();
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");

    if (isSetFFTsPointerLikeType(fftsAddr.getType())) {
      auto castTyAttr =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      fftsAddr =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                           /*args=*/ArrayAttr{},
                                           /*templateArgs=*/castTyAttr,
                                           /*operands=*/ValueRange{fftsAddr})
              .getResult(0);
    } else if (fftsAddr.getType() != u64Ty) {
      fftsAddr =
          rewriter.create<emitc::CastOp>(loc, u64Ty, fftsAddr).getResult();
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_ffts_base_addr",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{fftsAddr});
    return success();
  }
};

struct PTOSyncSetToEmitC : public OpConversionPattern<mlir::pto::SyncSetOp> {
  PTOSyncSetToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                    PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncSetOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();
    int64_t fftsMode = 2;
    if (IntegerAttr fftsModeAttr = op.getFftsModeAttr())
      fftsMode = getIntegerAttrSignedValue(fftsModeAttr);

    const bool hasStaticEventId = eventIdAttr != nullptr;
    const bool hasDynamicEventId = static_cast<bool>(eventIdDyn);
    if (hasStaticEventId == hasDynamicEventId) {
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");
    }

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncSetCall(rewriter, loc, targetArch, op.getPipe(),
                                       eventIdAttr, fftsMode);
    } else {
      desc = buildInterCoreSyncSetCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                          eventIdDyn, fftsMode);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         /*args=*/desc.args,
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

struct PTOSyncWaitToEmitC : public OpConversionPattern<mlir::pto::SyncWaitOp> {
  PTOSyncWaitToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                     PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncWaitOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();

    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncWaitCall(rewriter, targetArch, op.getPipe(),
                                        eventIdAttr);
    } else {
      desc = buildInterCoreSyncWaitCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                           eventIdDyn);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         desc.args, ArrayAttr{}, desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

template <typename SyncOp>
struct PTONamedIntraSyncToEmitC : public OpConversionPattern<SyncOp> {
  PTONamedIntraSyncToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                           PTOArch targetArch)
      : OpConversionPattern<SyncOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();
    const bool hasStaticEventId = eventIdAttr != nullptr;
    const bool hasDynamicEventId = static_cast<bool>(eventIdDyn);
    if (hasStaticEventId == hasDynamicEventId) {
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");
    }

    if (targetArch != PTOArch::A5) {
      InterCoreSyncCallDesc desc;
      if constexpr (std::is_same_v<SyncOp, mlir::pto::SetIntraBlockOp>) {
        desc = eventIdAttr
                   ? buildInterCoreSyncSetCall(rewriter, loc, targetArch,
                                               op.getPipe(), eventIdAttr, 2)
                   : buildInterCoreSyncSetCallDyn(rewriter, loc, targetArch,
                                                  op.getPipe(), eventIdDyn, 2);
      } else {
        desc = eventIdAttr
                   ? buildInterCoreSyncWaitCall(rewriter, targetArch,
                                                op.getPipe(), eventIdAttr)
                   : buildInterCoreSyncWaitCallDyn(rewriter, loc, targetArch,
                                                   op.getPipe(), eventIdDyn);
      }
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          op, TypeRange{}, desc.callee, desc.args, ArrayAttr{}, desc.operands);
      return success();
    }

    auto *ctx = rewriter.getContext();
    std::string pipeTok = pipeTokFromPipeAttr(op.getPipe());
    Value eventValue;
    if (eventIdDyn) {
      eventValue = castInterCoreEventIdToI32(rewriter, loc, eventIdDyn);
    }

    StringRef callee;
    if constexpr (std::is_same_v<SyncOp, mlir::pto::SetIntraBlockOp>) {
      callee = "__builtin_cce_set_intra_block";
    } else {
      callee = "__builtin_cce_wait_intra_block";
    }

    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        eventIdAttr ? eventIdAttr : IntegerAttr::get(IndexType::get(ctx), 0),
    });
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, args, ArrayAttr{},
        eventValue ? ValueRange{eventValue} : ValueRange{});
    return success();
  }

  PTOArch targetArch;
};

template <typename CrossOp, typename SyncOp>
struct PTOCrossSyncToSync : public OpConversionPattern<CrossOp> {
  PTOCrossSyncToSync(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<CrossOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(CrossOp op, typename CrossOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mode0 = IntegerAttr::get(rewriter.getI32Type(), 0);
    rewriter.replaceOpWithNewOp<SyncOp>(op, op.getPipe(), op.getEventIdAttr(),
                                        mode0, adaptor.getEventIdDyn());
    return success();
  }
};

void populateSyncCommSyncPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch) {
  patterns.add<PTOBarrierToEmitC>(typeConverter, ctx);
  patterns.add<PTOFenceToEmitC<pto::FenceBarrierAllOp>>(typeConverter, ctx);
  patterns.add<PTOSetFlagToEmitC>(typeConverter, ctx, "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_dyn",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_dyn",
                                      "wait_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_d",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_d",
                                      "wait_flag");
  patterns.add<PTOWaitFlagToEmitC>(typeConverter, ctx, "wait_flag");
  patterns.add<PTOSyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncAllToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFFTsToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOCrossSyncToSync<pto::SetCrossBlockOp, pto::SyncSetOp>,
               PTOCrossSyncToSync<pto::WaitCrossBlockOp, pto::SyncWaitOp>>(typeConverter, ctx);
  patterns.add<PTONamedIntraSyncToEmitC<pto::SetIntraBlockOp>,
               PTONamedIntraSyncToEmitC<pto::WaitIntraBlockOp>>(typeConverter,
                                                               ctx, targetArch);
}

} // namespace pto
} // namespace mlir
