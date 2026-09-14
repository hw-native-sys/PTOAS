// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TCvt.cpp - Tensor TCvt op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOCvtToEmitC : public OpConversionPattern<pto::TCvtOp> {
  using OpConversionPattern<pto::TCvtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCvtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    pto::RoundModeAttr rmAttr = op.getRmodeAttr();
    std::string rmTok = rmAttr ? roundModeTok(rmAttr)
                               : std::string("RoundMode::CAST_RINT");
    auto rmodeTy = emitc::OpaqueType::get(ctx, "RoundMode");
    Value rmodeVal = rewriter.create<emitc::ConstantOp>(
        loc, rmodeTy, emitc::OpaqueAttr::get(ctx, rmTok));

    auto satModeTy = emitc::OpaqueType::get(ctx, "SaturationMode");
    auto satAttr = op.getSatModeAttr();
    std::string satTok = satAttr ? saturationModeTok(satAttr)
                                 : std::string("SaturationMode::OFF");
    Value satModeVal = rewriter.create<emitc::ConstantOp>(
        loc, satModeTy, emitc::OpaqueAttr::get(ctx, satTok));

    SmallVector<Value, 5> operands{dst, src};
    if (adaptor.getTmp())
      operands.push_back(peelUnrealized(adaptor.getTmp()));
    operands.push_back(rmodeVal);
    operands.push_back(satModeVal);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCVT",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTCvtPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
