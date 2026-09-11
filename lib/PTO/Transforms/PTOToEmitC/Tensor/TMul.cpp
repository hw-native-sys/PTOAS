// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TMul.cpp - Tensor TMul op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOMulToEmitC : public OpConversionPattern<pto::TMulOp> {
using OpConversionPattern<pto::TMulOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMulOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst  = adaptor.getDst();

  SmallVector<Value, 3> operands{dst, src0, src1};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMUL", operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOMulsToEmitC : public OpConversionPattern<pto::TMulSOp> {
using OpConversionPattern<pto::TMulSOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMulSOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Value src = adaptor.getSrc0();
  Value dst = adaptor.getDst();
  Value scalar = adaptor.getScalar();

  SmallVector<Value, 3> operands{dst, src, scalar};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMULS", operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOTAxpyToEmitC : public OpConversionPattern<pto::TAxpyOp> {
  using OpConversionPattern<pto::TAxpyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAxpyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TAXPY",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTMulPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOMulsToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAxpyToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
