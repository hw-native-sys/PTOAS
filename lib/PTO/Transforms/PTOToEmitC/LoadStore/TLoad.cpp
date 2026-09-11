// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TLoad.cpp - LoadStore TLoad op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "LoadStoreInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOAddPtrToEmitC : public OpConversionPattern<pto::AddPtrOp> {
  using OpConversionPattern<pto::AddPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert pointer type");
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    rewriter.replaceOpWithNewOp<emitc::AddOp>(op, resultType, ptr, offset);
    return success();
  }
};

struct PTOTLoadToTLOAD : public OpConversionPattern<pto::TLoadOp> {
  using OpConversionPattern<pto::TLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tload");

    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Value dst = adaptor.getDst();

    ArrayAttr templateArgs = ArrayAttr{};
    if (auto policy = op.getCachePolicyAttr();
        policy && policy.getValue() == pto::LoadCachePolicy::L2Bypass) {
      templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
          rewriter.getContext(), "pto::TLoadL2Hint::NotAllocKeep")});
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TLOAD",
                                         ArrayAttr{}, templateArgs,
                                         ValueRange{dst, src});

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct PTOTPrefetchToTPREFETCH : public OpConversionPattern<pto::TPrefetchOp> {
  using OpConversionPattern<pto::TPrefetchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tprefetch");

    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TPREFETCH",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{dst, src});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTPrefetchAsyncToEmitC
    : public OpConversionPattern<pto::TPrefetchAsyncOp> {
  using OpConversionPattern<pto::TPrefetchAsyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchAsyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Type convertedSrcTy = getTypeConverter()->convertType(op.getSrc().getType());
    if (!convertedSrcTy || !isEmitCGlobalTensorLikeType(convertedSrcTy))
      return rewriter.notifyMatchFailure(op, "expected GlobalTensor-like src");

    Value prefetchCtx = adaptor.getCtx();

    Type eventTy = getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert tprefetch_async result type");

    Value event =
        rewriter
            .create<emitc::CallOpaqueOp>(
                op.getLoc(), TypeRange{eventTy}, "TPREFETCH_ASYNC", ArrayAttr{},
                ArrayAttr{}, ValueRange{src, prefetchCtx})
            .getResult(0);

    rewriter.replaceOp(op, ValueRange{event});
    return success();
  }
};

struct PTOMakePrefetchAsyncContextToEmitC
    : public OpConversionPattern<pto::MakePrefetchAsyncContextOp> {
  using OpConversionPattern<pto::MakePrefetchAsyncContextOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MakePrefetchAsyncContextOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type ctxTy = getTypeConverter()->convertType(op.getCtx().getType());
    if (!ctxTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert make_prefetch_async_context result type");

    Value workspace = adaptor.getWorkspace();
    workspace = castToGMBytePointer(rewriter, op.getLoc(), workspace);

    Value ctx = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), TypeRange{ctxTy}, "pto::PrefetchAsyncContext",
                        ArrayAttr{}, ArrayAttr{}, ValueRange{workspace})
                    .getResult(0);

    rewriter.replaceOp(op, ValueRange{ctx});
    return success();
  }
};

struct PTOGetPrefetchAsyncSessionToEmitC
    : public OpConversionPattern<pto::GetPrefetchAsyncSessionOp> {
  using OpConversionPattern<pto::GetPrefetchAsyncSessionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GetPrefetchAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type sessionTy = getTypeConverter()->convertType(op.getSession().getType());
    if (!sessionTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert get_prefetch_async_session result type");

    Value ctx = adaptor.getCtx();
    Value session = rewriter
                        .create<emitc::CallOpaqueOp>(
                            op.getLoc(), TypeRange{sessionTy},
                            "PTOAS__PREFETCH_CTX_SESSION", ArrayAttr{},
                            ArrayAttr{}, ValueRange{ctx})
                        .getResult(0);

    rewriter.replaceOp(op, ValueRange{session});
    return success();
  }
};

void populateLoadStoreTLoadPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTLoadToTLOAD>(typeConverter, ctx);
  patterns.add<PTOTPrefetchToTPREFETCH>(typeConverter, ctx);
  patterns.add<PTOMakePrefetchAsyncContextToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetPrefetchAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOTPrefetchAsyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddPtrToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
