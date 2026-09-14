// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TScatter.cpp - Reduce TScatter op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOScatterToEmitC : public OpConversionPattern<pto::TScatterOp> {
  using OpConversionPattern<pto::TScatterOp>::OpConversionPattern;

  // maskPattern flavor: TSCATTER<MaskPattern[, Axis]>(dst, src).
  void emitMaskPatternScatter(pto::TScatterOp op,
                              ConversionPatternRewriter &rewriter,
                              Value dst, Value src) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    SmallVector<Attribute, 2> targsList;
    targsList.push_back(
        emitc::OpaqueAttr::get(ctx, maskPatternTok(op.getMaskPatternAttr())));
    if (auto axisAttr = op.getAxisAttr()) {
      StringRef axisVal = axisAttr.getValue();
      std::string scatterAxis = (axisVal == "col")
                                    ? "pto::ScatterAxis::SCATTER_COL"
                                    : "pto::ScatterAxis::SCATTER_ROW";
      targsList.push_back(emitc::OpaqueAttr::get(ctx, scatterAxis));
    }
    auto targs = rewriter.getArrayAttr(targsList);
    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSCATTER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
        /*operands=*/operands);
  }

  LogicalResult matchAndRewrite(pto::TScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const bool hasMaskPattern = static_cast<bool>(op.getMaskPatternAttr());
    const bool hasIndexes = static_cast<bool>(op.getIndexes());
    if (hasMaskPattern == hasIndexes) {
      return rewriter.notifyMatchFailure(
          op, "expected exactly one of indexes operand or maskPattern attribute");
    }
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    if (hasMaskPattern) {
      emitMaskPatternScatter(op, rewriter, dst, src);
    } else {
      Value idx = adaptor.getIndexes();
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TSCATTER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src, idx});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void populateReduceTScatterPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOScatterToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
