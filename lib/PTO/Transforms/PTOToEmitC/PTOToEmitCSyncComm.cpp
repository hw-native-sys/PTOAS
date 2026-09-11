// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCSyncComm.cpp - sync/barrier/comm/async/declare lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCPatterns.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===


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

    SmallVector<Value> operands;
    operands.reserve(adaptor.getOperands().size());
    auto calleeType = op.getCalleeType();
    unsigned originalArgCount = calleeType.getNumInputs();
    if (originalArgCount != adaptor.getOperands().size())
      return rewriter.notifyMatchFailure(
          op, "call operand count mismatch after type conversion");

    for (auto [index, loweredOperand] : llvm::enumerate(adaptor.getOperands())) {
      FailureOr<Value> adapted = adaptCallOperandForEmitC(
          getTypeConverter(), rewriter, op.getLoc(), calleeType.getInput(index),
          op.getOperand(index),
          loweredOperand);
      if (failed(adapted))
        return rewriter.notifyMatchFailure(op,
                                           "failed to adapt call operand for EmitC ABI");
      operands.push_back(*adapted);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getCalleeAttr(),
                                               resultTypes, operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering
//===----------------------------------------------------------------------===


[[maybe_unused]] static std::string getPipeName(pto::PIPE pipe) {
  switch (pipe) {
    case pto::PIPE::PIPE_S: return "PIPE_S";
    case pto::PIPE::PIPE_V: return "PIPE_V";
    case pto::PIPE::PIPE_M: return "PIPE_M";
    case pto::PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case pto::PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case pto::PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case pto::PIPE::PIPE_ALL: return "PIPE_ALL";
    case pto::PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case pto::PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case pto::PIPE::PIPE_V2: return "PIPE_V2";
    case pto::PIPE::PIPE_FIX: return "PIPE_FIX";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    // 默认回退
    default: return "PIPE_ALL"; 
  }
}

//===----------------------------------------------------------------------===//
// pto.barrier lowering -> pipe_barrier(...)
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

//===----------------------------------------------------------------------===//




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

// Sync lowering (robust for bracket form pto.set_flag[...] / pto.wait_flag[...])
// Replace your PTOSyncToRuntimeCall with the code below.
//===----------------------------------------------------------------------===//

static bool tryConvertPipeAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto pipe = dyn_cast<mlir::pto::PipeAttr>(attr)) {
    token = mlir::pto::stringifyPIPE(pipe.getPipe()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryConvertEventAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto event = dyn_cast<mlir::pto::EventAttr>(attr)) {
    token = mlir::pto::stringifyEVENT(event.getEvent()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryAssignSyncTokens(Attribute srcAttr, Attribute dstAttr,
                                Attribute evtAttr, std::string &srcTok,
                                std::string &dstTok, std::string &evtTok) {
  std::string localSrc;
  std::string localDst;
  std::string localEvt;
  if (!tryConvertPipeAttrToToken(srcAttr, localSrc) ||
      !tryConvertPipeAttrToToken(dstAttr, localDst) ||
      !tryConvertEventAttrToToken(evtAttr, localEvt)) {
    return false;
  }
  srcTok = std::move(localSrc);
  dstTok = std::move(localDst);
  evtTok = std::move(localEvt);
  return true;
}

static bool tryExtractSyncTokensFromNamedAttrs(Operation *op,
                                               StringRef srcName,
                                               StringRef dstName,
                                               StringRef evtName,
                                               std::string &srcTok,
                                               std::string &dstTok,
                                               std::string &evtTok) {
  return tryAssignSyncTokens(op->getAttr(srcName), op->getAttr(dstName),
                             op->getAttr(evtName), srcTok, dstTok, evtTok);
}

static bool tryExtractSyncTokensFromArrayAttr(Operation *op, StringRef attrName,
                                              std::string &srcTok,
                                              std::string &dstTok,
                                              std::string &evtTok) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.size() < 3)
    return false;
  return tryAssignSyncTokens(arrayAttr[0], arrayAttr[1], arrayAttr[2], srcTok,
                             dstTok, evtTok);
}

static bool tryExtractFallbackSyncTokens(Operation *op, std::string &srcTok,
                                         std::string &dstTok,
                                         std::string &evtTok) {
  SmallVector<std::string, 2> pipes;
  std::string event;
  for (NamedAttribute namedAttr : op->getAttrs()) {
    std::string token;
    if (tryConvertPipeAttrToToken(namedAttr.getValue(), token)) {
      pipes.push_back(std::move(token));
      continue;
    }
    if (event.empty() &&
        tryConvertEventAttrToToken(namedAttr.getValue(), token)) {
      event = std::move(token);
    }
  }
  if (pipes.size() < 2 || event.empty())
    return false;
  srcTok = pipes[0];
  dstTok = pipes[1];
  evtTok = event;
  return true;
}

LogicalResult extractSyncTripletTokens(Operation *op,
                                             std::string &srcTok,
                                             std::string &dstTok,
                                             std::string &evtTok,
                                             ConversionPatternRewriter &rewriter) {
  if (tryExtractSyncTokensFromNamedAttrs(op, "src_pipe", "dst_pipe", "event_id",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "srcPipe", "dstPipe", "eventId",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "src", "dst", "event", srcTok,
                                         dstTok, evtTok)) {
    return success();
  }

  for (StringRef attrName : {"args", "pipes", "sync", "triplet", "attrs"}) {
    if (tryExtractSyncTokensFromArrayAttr(op, attrName, srcTok, dstTok,
                                          evtTok)) {
      return success();
    }
  }

  if (tryExtractFallbackSyncTokens(op, srcTok, dstTok, evtTok))
    return success();
  return rewriter.notifyMatchFailure(
      op, "cannot extract PIPE/PIPE/EVENT tokens from pto.{set,wait}_flag");
}
// set_flag/wait_flag lowering: resolve the PIPE/PIPE/EVENT token triplet and
// emit the matching runtime call.
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

struct PTOGetBufToEmitC : public OpConversionPattern<mlir::pto::GetBufOp> {
  using OpConversionPattern<mlir::pto::GetBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::GetBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "get_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "get_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "get_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOGetBufDynToEmitC : public OpConversionPattern<mlir::pto::GetBufDynOp> {
  using OpConversionPattern<mlir::pto::GetBufDynOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::GetBufDynOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "get_buf_dyn expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "get_buf_dyn op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "get_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{adaptor.getBufId()});
    return success();
  }
};

struct PTORlsBufToEmitC : public OpConversionPattern<mlir::pto::RlsBufOp> {
  using OpConversionPattern<mlir::pto::RlsBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::RlsBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "rls_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "rls_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "rls_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTORlsBufDynToEmitC : public OpConversionPattern<mlir::pto::RlsBufDynOp> {
  using OpConversionPattern<mlir::pto::RlsBufDynOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::RlsBufDynOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "rls_buf_dyn expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "rls_buf_dyn op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "rls_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{adaptor.getBufId()});
    return success();
  }
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

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())
struct PTOGetBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_idx", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetBlockNumOp Lowering (pto.get_block_num -> get_block_num())
struct PTOGetBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_num", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockIdxOp Lowering (pto.get_block_idx -> get_subblockid())
struct PTOGetSubBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockid", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockNumOp Lowering.
struct PTOGetSubBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockdim", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};


void populateSyncCommPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
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
  patterns.add<PTOGetBufToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBufDynToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufDynToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFFTsToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOCrossSyncToSync<pto::SetCrossBlockOp, pto::SyncSetOp>,
               PTOCrossSyncToSync<pto::WaitCrossBlockOp, pto::SyncWaitOp>>(
      typeConverter, ctx);
  patterns.add<PTONamedIntraSyncToEmitC<pto::SetIntraBlockOp>,
               PTONamedIntraSyncToEmitC<pto::WaitIntraBlockOp>>(typeConverter,
                                                               ctx, targetArch);
  patterns.add<PTOGetBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<CallToEmitC, ReturnToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
