// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithCmp.cpp - Arith ArithCmp op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ArithCmpFToEmitC : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getLhs().getType()))
      return rewriter.notifyMatchFailure(op, "cmpf only supported on scalar floats");

    auto loc = op.getLoc();
    auto i1Ty = rewriter.getI1Type();
    if (auto special = buildSpecialCmpFResult(op.getPredicate(), rewriter, loc,
                                              i1Ty, adaptor.getLhs(),
                                              adaptor.getRhs())) {
      rewriter.replaceOp(op, *special);
      return success();
    }

    auto config = getCmpFConfig(op.getPredicate());
    if (!config)
      return rewriter.notifyMatchFailure(op, "unsupported cmpf predicate");
    rewriter.replaceOp(op, buildCmpFResult(*config, rewriter, loc, i1Ty,
                                           adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

void populateArithArithCmpPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithCmpFToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
