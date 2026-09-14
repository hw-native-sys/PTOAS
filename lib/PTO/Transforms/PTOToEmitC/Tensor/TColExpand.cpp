// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TColExpand.cpp - Tensor TColExpand op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOColExpandToEmitC : public OpConversionPattern<pto::TColExpandOp> {
  using OpConversionPattern<pto::TColExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value dst = adaptor.getDst();
    Value src = adaptor.getSrc();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPAND",
        /*args=*/ArrayAttr(),           
        /*templateArgs=*/ArrayAttr(),
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandDivToEmitC : public OpConversionPattern<pto::TColExpandDivOp> {
  using OpConversionPattern<pto::TColExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::DivPrecision::Default,
        "DivAlgorithm");

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDDIV",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy>
struct PTOColExpandBinaryToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTOColExpandBinaryToEmitC(TypeConverter &typeConverter,
                                     MLIRContext *ctx, StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, callee,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

using PTOColExpandMulToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandMulOp>;
using PTOColExpandAddToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandAddOp>;
using PTOColExpandSubToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandSubOp>;
using PTOColExpandMaxToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandMaxOp>;
using PTOColExpandMinToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandMinOp>;

struct PTOColExpandExpdifToEmitC
    : public OpConversionPattern<pto::TColExpandExpdifOp> {
  using OpConversionPattern<pto::TColExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDEXPDIF",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTColExpandPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOColExpandAddToEmitC>(typeConverter, ctx, "TCOLEXPANDADD");
  patterns.add<PTOColExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMulToEmitC>(typeConverter, ctx, "TCOLEXPANDMUL");
  patterns.add<PTOColExpandMaxToEmitC>(typeConverter, ctx, "TCOLEXPANDMAX");
  patterns.add<PTOColExpandMinToEmitC>(typeConverter, ctx, "TCOLEXPANDMIN");
  patterns.add<PTOColExpandSubToEmitC>(typeConverter, ctx, "TCOLEXPANDSUB");
  patterns.add<PTOColExpandToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
