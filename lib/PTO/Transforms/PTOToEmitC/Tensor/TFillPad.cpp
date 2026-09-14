// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TFillPad.cpp - Tensor TFillPad op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOFillPadToEmitC : public OpConversionPattern<pto::TFillPadOp> {
  using OpConversionPattern<pto::TFillPadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    auto loweringKind = pto::inferTFillPadLoweringKindAfterMemoryPlanning(op);
    if (failed(loweringKind)) {
      op.emitOpError(
          "cannot infer a supported lowering; expand and in-place forms "
          "require loc=vec, statically comparable physical shapes, and "
          "resolved planned addresses");
      return failure();
    }

    auto padValueTok = [&](pto::PadValue mode) -> StringRef {
      switch (mode) {
      case pto::PadValue::Null:
        return "pto::PadValue::Null";
      case pto::PadValue::Zero:
        return "pto::PadValue::Zero";
      case pto::PadValue::Max:
        return "pto::PadValue::Max";
      case pto::PadValue::Min:
        return "pto::PadValue::Min";
      }
      llvm_unreachable("unknown PadValue");
    };

    ArrayAttr templateArgs{};
    if (auto padValueAttr = op.getPadValueAttr()) {
      // The verifier only accepts explicit padValue for loc=mat tile-form
      // tfillpad, so lowering can trust the preserved semantic contract.
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, padValueTok(padValueAttr.getValue()))});
    } else if (*loweringKind != pto::TFillPadLoweringKind::Normal) {
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, getTFillPadModeToken(*loweringKind))});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTFillPadPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOFillPadToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
