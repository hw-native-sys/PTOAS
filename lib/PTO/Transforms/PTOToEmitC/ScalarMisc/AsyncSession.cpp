// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- AsyncSession.cpp - ScalarMisc AsyncSession op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOInitializeL2G2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp> {
  PTOInitializeL2G2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2G2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    Value gmAddr = adaptor.getGmAddr();
    gmAddr = materializeGlobalTensorDataPointer(
        rewriter, op.getLoc(), gmAddr, op.getGmAddr().getType());
    Value localAddr =
        op.getLocalAddr() ? adaptor.getLocalAddr() : Value();
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1) {
      c2vBuf = localAddr ? localAddr : zero;
    } else if (op.getDirMask() == 2) {
      v2cBuf = localAddr ? localAddr : zero;
    } else if (op.getDirMask() == 3) {
      if (localAddr) {
        if (!op.getPeerLocalAddr()) {
          return rewriter.notifyMatchFailure(
              op, "bidirectional l2g2l pipe requires peer local buffer");
        }
        c2vBuf = localAddr;
        v2cBuf = adaptor.getPeerLocalAddr();
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{gmAddr, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOInitializeL2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2LPipeOp> {
  PTOInitializeL2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                              PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    auto gmPtrTy =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ void"));
    Value nullGm =
        makeEmitCOpaqueConstant(rewriter, op.getLoc(), gmPtrTy, "nullptr");
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
    Value localAddr = adaptor.getLocalAddr();

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1) {
      c2vBuf = localAddr;
    } else if (op.getDirMask() == 2) {
      v2cBuf = localAddr;
    } else if (op.getDirMask() == 3) {
      c2vBuf = localAddr;
      v2cBuf = adaptor.getPeerLocalAddr();
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{nullGm, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOBuildAsyncSessionToEmitC
    : public OpConversionPattern<mlir::pto::BuildAsyncSessionOp> {
  PTOBuildAsyncSessionToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<mlir::pto::BuildAsyncSessionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(mlir::pto::BuildAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();

    auto sessionTy = dyn_cast<emitc::OpaqueType>(
        getTypeConverter()->convertType(op.getSession().getType()));
    if (!sessionTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert async session type");

    FailureOr<Value> scratchTile =
        buildAsyncScratchTileValue(rewriter, loc, op.getScratch(),
                                   adaptor.getScratch());
    if (failed(scratchTile))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize async scratch tile");

    Value workspace =
        castToGMBytePointer(rewriter, loc, adaptor.getWorkspace());

    Value session = rewriter
                        .create<emitc::VariableOp>(
                            loc, getEmitCVariableResultType(sessionTy),
                            emitc::OpaqueAttr::get(ctx, ""))
                        .getResult();
    session = loadEmitCVariableIfNeeded(rewriter, loc, session);

    Value syncIdVal = makeAsyncU32Constant(rewriter, loc, ctx, op.getSyncIdAttr(), 0);
    Value channelGroupIdxVal =
        buildChannelGroupIdxValue(rewriter, loc, ctx, op.getChannelGroupIdxAttr());
    Value baseConfig =
        buildSdmaBaseConfig(rewriter, loc, ctx, op.getBlockBytesAttr(),
                            op.getCommBlockOffsetAttr(), op.getQueueNumAttr());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{*scratchTile, workspace, session, syncIdVal, baseConfig,
                   channelGroupIdxVal});

    rewriter.replaceOp(op, session);
    return success();
  }

  // u32 constant from an optional integer attribute with a default.
  static Value makeAsyncU32Constant(ConversionPatternRewriter &rewriter,
                                    Location loc, MLIRContext *ctx,
                                    IntegerAttr attr, uint64_t defaultValue) {
    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");
    uint64_t value =
        attr ? static_cast<uint64_t>(getIntegerAttrSignedValue(attr))
             : defaultValue;
    return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                    std::to_string(value) + "u");
  }

  // channel_group_idx defaults to UINT32_MAX when the attribute is absent.
  static Value buildChannelGroupIdxValue(ConversionPatternRewriter &rewriter,
                                         Location loc, MLIRContext *ctx,
                                         IntegerAttr attr) {
    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");
    if (!attr)
      return makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX");
    uint64_t value = static_cast<uint64_t>(getIntegerAttrSignedValue(attr));
    if (value == UINT32_MAX)
      return makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX");
    return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                    std::to_string(value) + "u");
  }

  // SdmaBaseConfig{blockBytes, commBlockOffset, queueNum} aggregate.
  static Value buildSdmaBaseConfig(ConversionPatternRewriter &rewriter,
                                   Location loc, MLIRContext *ctx,
                                   IntegerAttr blockBytesAttr,
                                   IntegerAttr commBlockOffsetAttr,
                                   IntegerAttr queueNumAttr) {
    uint64_t blockBytes =
        blockBytesAttr
            ? static_cast<uint64_t>(
                  getIntegerAttrSignedValue(blockBytesAttr))
            : 32 * 1024;
    uint64_t commBlockOffset =
        commBlockOffsetAttr
            ? static_cast<uint64_t>(
                  getIntegerAttrSignedValue(commBlockOffsetAttr))
            : 0;
    uint64_t queueNum =
        queueNumAttr
            ? static_cast<uint64_t>(getIntegerAttrSignedValue(queueNumAttr))
            : 1;

    auto baseConfigTy =
        emitc::OpaqueType::get(ctx, "pto::comm::sdma::SdmaBaseConfig");
    Value baseConfig =
        rewriter
            .create<emitc::VariableOp>(
                loc, getEmitCVariableResultType(baseConfigTy),
                emitc::OpaqueAttr::get(
                    ctx, "{" + std::to_string(blockBytes) + "ULL, " +
                             std::to_string(commBlockOffset) + "ULL, " +
                             std::to_string(queueNum) + "u}"))
            .getResult();
    return loadEmitCVariableIfNeeded(rewriter, loc, baseConfig);
  }
};

template <typename AsyncOp>
struct PTOAsyncTransferToEmitC : public OpConversionPattern<AsyncOp> {
  using OpConversionPattern<AsyncOp>::OpConversionPattern;

  explicit PTOAsyncTransferToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                   StringRef callee)
      : OpConversionPattern<AsyncOp>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncOp op, typename AsyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelGlobalTensorConversionBridge(adaptor.getDst());
    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Type convertedDstTy =
        this->getTypeConverter()->convertType(op.getDst().getType());
    Type convertedSrcTy =
        this->getTypeConverter()->convertType(op.getSrc().getType());
    if (!convertedDstTy || !convertedSrcTy ||
        !isEmitCGlobalTensorLikeType(convertedDstTy) ||
        !isEmitCGlobalTensorLikeType(convertedSrcTy))
      return rewriter.notifyMatchFailure(
          op, "expected GlobalTensor-like src and dst");

    Type eventTy = this->getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{eventTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, adaptor.getSession()});
    return success();
  }

  std::string callee;
};

template <typename AsyncEventOp>
struct PTOAsyncEventToEmitC : public OpConversionPattern<AsyncEventOp> {
  explicit PTOAsyncEventToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<AsyncEventOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncEventOp op,
                                typename AsyncEventOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        this->getTypeConverter()->convertType(op.getCompleted().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event result type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{adaptor.getEvent(),
                   adaptor.getSession()});
    return success();
  }

  std::string callee;
};

void populateScalarMiscAsyncSessionPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch) {
  patterns.add<PTOBuildAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOAsyncTransferToEmitC<pto::TPutAsyncOp>>(typeConverter, ctx,
      "pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncTransferToEmitC<pto::TGetAsyncOp>>(typeConverter, ctx,
      "pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncEventToEmitC<pto::WaitAsyncEventOp>>(typeConverter, ctx, "PTOAS__ASYNC_EVENT_WAIT");
  patterns.add<PTOAsyncEventToEmitC<pto::TestAsyncEventOp>>(typeConverter, ctx, "PTOAS__ASYNC_EVENT_TEST");
  patterns.add<PTOInitializeL2G2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOInitializeL2LPipeToEmitC>(typeConverter, ctx, targetArch);
}

} // namespace pto
} // namespace mlir
