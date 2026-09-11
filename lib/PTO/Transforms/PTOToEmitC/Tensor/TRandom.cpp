// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TRandom.cpp - Tensor TRandom op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTORandomToEmitC : public OpConversionPattern<pto::TRandomOp> {
  using OpConversionPattern<pto::TRandomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRandomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = adaptor.getDst();
    SmallVector<Value, 7> operands{
        dst,
        adaptor.getKey0(),
        adaptor.getKey1(),
        adaptor.getCounter0(),
        adaptor.getCounter1(),
        adaptor.getCounter2(),
        adaptor.getCounter3(),
    };
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, std::to_string(op.getRounds()))});

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "PTOAS__TRANDOM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTRandomPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTORandomToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
