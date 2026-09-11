// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TRowReduce.cpp - Reduce TRowReduce op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

template <typename OpTy>
struct PTORowReduceToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTORowReduceToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                               StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    createLastUseAwareOpaqueCall(
        rewriter, op.getOperation(), TypeRange{}, callee,
        ValueRange{adaptor.getDst(), adaptor.getSrc(), adaptor.getTmp()});
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

using PTORowMaxToEmitC = PTORowReduceToEmitC<pto::TRowMaxOp>;

struct PTORowArgMaxToEmitC
    : public OpConversionPattern<pto::TRowArgMaxOp> {
  using OpConversionPattern<pto::TRowArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTORowArgMinToEmitC
    : public OpConversionPattern<pto::TRowArgMinOp> {
  using OpConversionPattern<pto::TRowArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTORowProdToEmitC : public OpConversionPattern<pto::TRowProdOp> {
  using OpConversionPattern<pto::TRowProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWPROD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

using PTORowMinToEmitC = PTORowReduceToEmitC<pto::TRowMinOp>;
using PTORowSumToEmitC = PTORowReduceToEmitC<pto::TRowSumOp>;
void populateReduceTRowReducePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTORowMaxToEmitC>(typeConverter, ctx, "TROWMAX");
  patterns.add<PTORowArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowProdToEmitC>(typeConverter, ctx);
  patterns.add<PTORowSumToEmitC>(typeConverter, ctx, "TROWSUM");
  patterns.add<PTORowMinToEmitC>(typeConverter, ctx, "TROWMIN");
  patterns.add<PTORowArgMinToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
