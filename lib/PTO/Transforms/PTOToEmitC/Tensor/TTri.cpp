// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TTri.cpp - Tensor TTri op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOTTriToEmitC : public OpConversionPattern<pto::TTriOp> {
  using OpConversionPattern<pto::TTriOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTriOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = adaptor.getDst();
    Value diagonal = adaptor.getDiagonal();

    ArrayAttr templateArgs;
    if (auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, std::to_string(op.getUpperOrLower())),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 2> operands{dst, diagonal};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRI",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTTriPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTTriToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
