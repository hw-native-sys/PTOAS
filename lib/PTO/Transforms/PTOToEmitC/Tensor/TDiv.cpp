// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TDiv.cpp - Tensor TDiv op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTODivToTDIV : public OpConversionPattern<pto::TDivOp> {
  using OpConversionPattern<pto::TDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::DivPrecision::Default,
        "DivAlgorithm");

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIV", ValueRange{dst, src0, src1}, ArrayAttr{}, templateArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTDivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst    = adaptor.getDst();
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIVS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTDivPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTDivSToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
