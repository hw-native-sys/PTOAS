// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TSub.cpp - Tensor TSub op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOSubSToEmitC : public OpConversionPattern<pto::TSubOp> {
  using OpConversionPattern<pto::TSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src0, src1};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TSUB", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSubCSToEmitC : public OpConversionPattern<pto::TSubCOp> {
  using OpConversionPattern<pto::TSubCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TSUBC yet.
    // Decompose: dst = src0 - src1 + src2
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TSUB",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getSrc1(), adaptor.getSrc2());
    return success();
  }
};

struct PTOSubSCToEmitC : public OpConversionPattern<pto::TSubSCOp> {
  using OpConversionPattern<pto::TSubSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TSUBSC yet.
    // Decompose: dst = src0 - scalar + src1
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TSUBS",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getScalar(), adaptor.getSrc1());
    return success();
  }
};

void populateTensorTSubPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOSubSCToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubCSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
