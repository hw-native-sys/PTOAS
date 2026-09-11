// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TConcat.cpp - Tensor TConcat op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOConcatToEmitC : public OpConversionPattern<pto::TConcatOp> {
  using OpConversionPattern<pto::TConcatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOConcatidxToEmitC : public OpConversionPattern<pto::TConcatidxOp> {
  using OpConversionPattern<pto::TConcatidxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatidxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value src0Idx = adaptor.getSrc0Idx();
    Value src1Idx = adaptor.getSrc1Idx();
    Value dst  = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTConcatPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOConcatToEmitC, PTOConcatidxToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
