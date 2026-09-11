// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TAnd.cpp - Tensor TAnd op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOTAndToEmitC : public OpConversionPattern<pto::TAndOp> {
  using OpConversionPattern<pto::TAndOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a   = adaptor.getSrc0();
    Value b   = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TAND",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, a, b});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOAndSToEmitC : public OpConversionPattern<pto::TAndSOp> {
  using OpConversionPattern<pto::TAndSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst    = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTAndPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
