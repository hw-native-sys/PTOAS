// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithIntMinMax.cpp - Arith ArithIntMinMax op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ArithMinMaxIToEmitCBase {
  static Value makeSelect(ConversionPatternRewriter &rewriter, Location loc,
                          Type dstTy, Value cond, Value trueV, Value falseV) {
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, cond, trueV, falseV)
        .getResult();
  }
};

template <typename ArithOp, emitc::CmpPredicate Pred, bool TakeRhsOnTrue,
          bool IsUnsigned>
struct ArithMinMaxIToEmitC : public OpConversionPattern<ArithOp>,
                             ArithMinMaxIToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if constexpr (IsUnsigned) {
      unsigned bitWidth = getScalarIntOrIndexBitWidth(op.getType());
      lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
      rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
    }
    Value cond =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(), Pred, lhs,
                                      rhs)
            .getResult();
    Value onTrue = TakeRhsOnTrue ? adaptor.getRhs() : adaptor.getLhs();
    Value onFalse = TakeRhsOnTrue ? adaptor.getLhs() : adaptor.getRhs();
    rewriter.replaceOp(
        op, makeSelect(rewriter, loc, dstTy, cond, onTrue, onFalse));
    return success();
  }
};
using ArithMaxSIToEmitC =
    ArithMinMaxIToEmitC<arith::MaxSIOp, emitc::CmpPredicate::lt, true, false>;
using ArithMinSIToEmitC =
    ArithMinMaxIToEmitC<arith::MinSIOp, emitc::CmpPredicate::lt, false, false>;
using ArithMaxUIToEmitC =
    ArithMinMaxIToEmitC<arith::MaxUIOp, emitc::CmpPredicate::lt, true, true>;
using ArithMinUIToEmitC =
    ArithMinMaxIToEmitC<arith::MinUIOp, emitc::CmpPredicate::lt, false, true>;
void populateArithArithIntMinMaxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithMaxSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinUIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
