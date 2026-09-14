// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TExp.cpp - Tensor TExp op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOExpToEmitC : public OpConversionPattern<pto::TExpOp> {
  using OpConversionPattern<pto::TExpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::ExpPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::ExpPrecision::Default:
        precisionTok = "pto::ExpAlgorithm::DEFAULT";
        break;
      case pto::ExpPrecision::HighPrecision:
        precisionTok = "pto::ExpAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TEXP", ValueRange{dst, src}, ArrayAttr{}, templateArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOExpandsToEmitC : public OpConversionPattern<pto::TExpandsOp> {
  using OpConversionPattern<pto::TExpandsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpandsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value scalar = adaptor.getScalar();
    Value dst    = adaptor.getDst();

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TEXPANDS", ValueRange{dst, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTExpPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOExpToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpandsToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
