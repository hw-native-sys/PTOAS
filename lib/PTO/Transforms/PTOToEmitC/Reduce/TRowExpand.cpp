// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TRowExpand.cpp - Reduce TRowExpand op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTORowExpandToEmitC : public OpConversionPattern<pto::TRowExpandOp> {
  using OpConversionPattern<pto::TRowExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy, bool LastUseAware = false>
struct PTORowExpandBinaryToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTORowExpandBinaryToEmitC(TypeConverter &typeConverter,
                                     MLIRContext *ctx, StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if constexpr (LastUseAware) {
      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, callee,
          collectRowExpandOperands(op, adaptor));
    } else {
      rewriter.create<emitc::CallOpaqueOp>(
          op.getLoc(), TypeRange{}, callee,
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/collectRowExpandOperands(op, adaptor));
    }
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};
struct PTORowExpandDivToEmitC
    : public PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true> {
  using PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true>::
      PTORowExpandBinaryToEmitC;

  PTORowExpandDivToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true>(
            typeConverter, ctx, "TROWEXPANDDIV") {}

  LogicalResult matchAndRewrite(pto::TRowExpandDivOp op,
                                OpConversionPattern<pto::TRowExpandDivOp>::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getPrecisionType() == pto::DivPrecision::Default)
      return PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true>::
          matchAndRewrite(op, adaptor, rewriter);

    createLastUseAwareOpaqueCall(
        rewriter, op.getOperation(), TypeRange{}, "TROWEXPANDDIV",
        collectRowExpandOperands(op, adaptor), ArrayAttr{},
        buildPrecisionTemplateArgs(rewriter, op.getPrecisionType(),
                                   pto::DivPrecision::Default,
                                   "DivAlgorithm"));
    rewriter.eraseOp(op);
    return success();
  }
};

using PTORowExpandAddToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandAddOp>;



using PTORowExpandExpdifToEmitC =
    PTORowExpandBinaryToEmitC<pto::TRowExpandExpdifOp>;
using PTORowExpandMulToEmitC =
    PTORowExpandBinaryToEmitC<pto::TRowExpandMulOp, true>;
using PTORowExpandSubToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandSubOp>;
using PTORowExpandMaxToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandMaxOp>;
using PTORowExpandMinToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandMinOp>;
void populateReduceTRowExpandPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTORowExpandMulToEmitC>(typeConverter, ctx, "TROWEXPANDMUL");
  patterns.add<PTORowExpandAddToEmitC>(typeConverter, ctx, "TROWEXPANDADD");
  patterns.add<PTORowExpandExpdifToEmitC>(typeConverter, ctx,
                                          "TROWEXPANDEXPDIF");
  patterns.add<PTORowExpandMaxToEmitC>(typeConverter, ctx, "TROWEXPANDMAX");
  patterns.add<PTORowExpandMinToEmitC>(typeConverter, ctx, "TROWEXPANDMIN");
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx, "TROWEXPANDSUB");
  patterns.add<PTORowExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandDivToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
