// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TInterleave.cpp - Reduce TInterleave op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOTInterleaveToEmitC
    : public OpConversionPattern<pto::TInterleaveOp> {
  using OpConversionPattern<pto::TInterleaveOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pto::TInterleaveOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    createLastUseAwareOpaqueCall(
        rewriter, op.getOperation(), TypeRange{}, "TINTERLEAVE",
        ValueRange{adaptor.getDst1(), adaptor.getDst0(), adaptor.getSrc1(),
                   adaptor.getSrc0()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTDeInterleaveToEmitC
    : public OpConversionPattern<pto::TDeInterleaveOp> {
  using OpConversionPattern<pto::TDeInterleaveOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pto::TDeInterleaveOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value dst1 = adaptor.getDsts()[1];
    Value dst0 = adaptor.getDsts()[0];
    Value src0 = adaptor.getSrcs()[0];
    bool hasSecondSource = adaptor.getSrcs().size() == 2;
    if (hasSecondSource) {
      Value src1 = adaptor.getSrcs()[1];
      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TDEINTERLEAVE",
          ValueRange{dst1, dst0, src1, src0});
    } else {
      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TDEINTERLEAVE",
          ValueRange{dst1, dst0, src0});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void populateReduceTInterleavePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTInterleaveToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDeInterleaveToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
