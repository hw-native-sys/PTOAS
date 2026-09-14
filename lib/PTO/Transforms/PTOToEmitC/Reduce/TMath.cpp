// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TMath.cpp - Reduce TMath op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTORecipToEmitC : public OpConversionPattern<pto::TRecipOp> {
  using OpConversionPattern<pto::TRecipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRecipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::RecipPrecision::Default,
        "RecipAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRECIP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORemToEmitC : public OpConversionPattern<pto::TRemOp> {
  using OpConversionPattern<pto::TRemOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value tmp  = adaptor.getTmp();
    Value dst  = adaptor.getDst();
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::RemPrecision::Default,
        "RemAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModToEmitC : public OpConversionPattern<pto::TFModOp> {
  using OpConversionPattern<pto::TFModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::FmodPrecision::Default,
        "FmodAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMOD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORemSToEmitC : public OpConversionPattern<pto::TRemSOp> {
  using OpConversionPattern<pto::TRemSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();
    Value scalar = adaptor.getScalar();
    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREMS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModSToEmitC : public OpConversionPattern<pto::TFModSOp> {
  using OpConversionPattern<pto::TFModSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMODS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPowToEmitC : public OpConversionPattern<pto::TPowOp> {
  using OpConversionPattern<pto::TPowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value base = adaptor.getBase();
    Value exp  = adaptor.getExp();
    Value dst  = adaptor.getDst();

    // Forms:
    //   integer:  TPOW(dst, base, exp)
    //   float:    TPOW(dst, base, exp, tmp)
    SmallVector<Value, 4> operands{dst, base, exp};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::PowPrecision::Default,
        "PowAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPOW",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPowSToEmitC : public OpConversionPattern<pto::TPowSOp> {
  using OpConversionPattern<pto::TPowSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPowSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src    = adaptor.getSrc();
    Value dst    = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    // Forms:
    //   integer:  TPOWS(dst, src, scalar)
    //   float:    TPOWS(dst, src, scalar, tmp)
    SmallVector<Value, 4> operands{dst, src, scalar};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));

    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::PowPrecision::Default,
        "PowAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPOWS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORsqrtToEmitC : public OpConversionPattern<pto::TRsqrtOp> {
  using OpConversionPattern<pto::TRsqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRsqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    SmallVector<Value, 3> operands{dst, src};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::RsqrtPrecision::Default,
        "RsqrtAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSqrtSToEmitC : public OpConversionPattern<pto::TSqrtOp> {
  using OpConversionPattern<pto::TSqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::SqrtPrecision::Default,
        "SqrtAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateReduceTMathPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOSqrtSToEmitC>(typeConverter, ctx);
  patterns.add<PTORsqrtToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModToEmitC>(typeConverter, ctx);
  patterns.add<PTORemToEmitC>(typeConverter, ctx);
  patterns.add<PTORecipToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModSToEmitC>(typeConverter, ctx);
  patterns.add<PTORemSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPowToEmitC>(typeConverter, ctx);
  patterns.add<PTOPowSToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
