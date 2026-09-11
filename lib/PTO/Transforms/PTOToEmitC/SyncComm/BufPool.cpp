// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- BufPool.cpp - SyncComm BufPool op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "SyncCommInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

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

void populateSyncCommBufPoolPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOGetBufToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBufDynToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufDynToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
