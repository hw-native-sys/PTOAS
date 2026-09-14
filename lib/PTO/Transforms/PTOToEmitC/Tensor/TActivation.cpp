// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TActivation.cpp - Tensor TActivation op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOLogToEmitC : public OpConversionPattern<pto::TLogOp> {
  using OpConversionPattern<pto::TLogOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLogOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::LogPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::LogPrecision::Default:
        precisionTok = "pto::LogAlgorithm::DEFAULT";
        break;
      case pto::LogPrecision::HighPrecision:
        precisionTok = "pto::LogAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLOG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOLReluToEmitC : public OpConversionPattern<pto::TLReluOp> {
  using OpConversionPattern<pto::TLReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value slope = adaptor.getSlope();
    Value dst = adaptor.getDst();

          SmallVector<Value, 3> operands{dst, src, slope};

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOPreluToEmitC : public OpConversionPattern<pto::TPReluOp> {
  using OpConversionPattern<pto::TPReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value tmp  = adaptor.getTmp();
    Value dst  = adaptor.getDst();

    // C++ interface: TPRELU(dst, src0, src1, tmp) — last parameter is tmp.
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOReluToEmitC : public OpConversionPattern<pto::TReluOp> {
  using OpConversionPattern<pto::TReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTActivationPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOLogToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOPreluToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
