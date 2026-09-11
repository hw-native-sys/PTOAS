// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TMinMax.cpp - Tensor TMinMax op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOMaxToEmitC : public OpConversionPattern<pto::TMaxOp> {
using OpConversionPattern<pto::TMaxOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMaxOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst  = adaptor.getDst();

  SmallVector<Value, 3> operands{dst, src0, src1};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMAX", operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOMaxSToEmitC : public OpConversionPattern<pto::TMaxSOp> {
  using OpConversionPattern<pto::TMaxSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMaxSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, scalar};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMAXS", operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOMinToEmitC : public OpConversionPattern<pto::TMinOp> {
using OpConversionPattern<pto::TMinOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMinOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst  = adaptor.getDst();

  SmallVector<Value, 3> operands{dst, src0, src1};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMIN", operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOMinsToEmitC : public OpConversionPattern<pto::TMinSOp> {
using OpConversionPattern<pto::TMinSOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMinSOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Value src = adaptor.getSrc();
  Value dst = adaptor.getDst();
  Value scalar = adaptor.getScalar();

  SmallVector<Value, 3> operands{dst, src, scalar};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMINS", operands);

  rewriter.eraseOp(op);
  return success();
}
};

void populateTensorTMinMaxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinsToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
