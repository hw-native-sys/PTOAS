// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TPart.cpp - Reduce TPart op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOPartAddToEmitC : public OpConversionPattern<pto::TPartAddOp> {
  using OpConversionPattern<pto::TPartAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartMaxToEmitC : public OpConversionPattern<pto::TPartMaxOp> {
  using OpConversionPattern<pto::TPartMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartMinToEmitC : public OpConversionPattern<pto::TPartMinOp> {
  using OpConversionPattern<pto::TPartMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMaxToEmitC
    : public OpConversionPattern<pto::TPartArgMaxOp> {
  using OpConversionPattern<pto::TPartArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value src0Idx = adaptor.getSrc0Idx();
    Value src1Idx = adaptor.getSrc1Idx();
    Value dst = adaptor.getDst();
    Value dstIdx = adaptor.getDstIdx();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMinToEmitC
    : public OpConversionPattern<pto::TPartArgMinOp> {
  using OpConversionPattern<pto::TPartArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value src0Idx = adaptor.getSrc0Idx();
    Value src1Idx = adaptor.getSrc1Idx();
    Value dst = adaptor.getDst();
    Value dstIdx = adaptor.getDstIdx();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartMulToEmitC : public OpConversionPattern<pto::TPartMulOp> {
  using OpConversionPattern<pto::TPartMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateReduceTPartPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOPartMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartArgMaxToEmitC, PTOPartArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartAddToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
