// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithFloatMinMax.cpp - Arith ArithFloatMinMax op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ArithFloatMinMaxToEmitCBase {
  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value makeFZero(ConversionPatternRewriter &rewriter, Location loc,
                         Type ty) {
    return makeEmitCOpaqueConstant(rewriter, loc, ty, "0.0f");
  }
};

template <typename ArithOp, bool IsMax>
struct ArithNumFToEmitC : public OpConversionPattern<ArithOp>,
                          ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    // max picks rhs when lhs < rhs; min picks lhs.
    Value noNaN = rewriter
                      .create<emitc::ConditionalOp>(
                          loc, dstTy, cmpLt,
                          IsMax ? adaptor.getRhs() : adaptor.getLhs(),
                          IsMax ? adaptor.getLhs() : adaptor.getRhs())
                      .getResult();

    // Propagate whichever operand is NaN, preferring rhs on double-NaN.
    Value rhsOrPick = rewriter
                          .create<emitc::ConditionalOp>(
                              loc, dstTy, rhsNaN, adaptor.getLhs(), noNaN)
                          .getResult();
    Value res = rewriter
                    .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN,
                                                  adaptor.getRhs(), rhsOrPick)
                    .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

using ArithMaxNumFToEmitC = ArithNumFToEmitC<arith::MaxNumFOp, true>;
using ArithMinNumFToEmitC = ArithNumFToEmitC<arith::MinNumFOp, false>;

template <typename ArithOp, bool isMaximum>
struct ArithMinMaxFPropagateNaNToEmitC : public OpConversionPattern<ArithOp>,
                                        ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  static Value buildPrimaryCandidate(ConversionPatternRewriter &rewriter,
                                     Location loc, Type dstTy, Value lhs,
                                     Value rhs) {
    Value cmpLt =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::lt, lhs, rhs)
            .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(
            loc, dstTy, cmpLt, isMaximum ? rhs : lhs, isMaximum ? lhs : rhs)
        .getResult();
  }

  static Value buildSignBitValue(ConversionPatternRewriter &rewriter,
                                 Location loc, Value lhs, FloatType floatTy) {
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    auto templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
        rewriter.getContext(), cast<emitc::OpaqueType>(bitsTy).getValue())});
    Value lhsBits =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, TypeRange{bitsTy}, "ptoas_bitcast",
                                         ValueRange{lhs}, ArrayAttr{},
                                         templateArgs)
            .getResult(0);
    Value oneBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 1);
    Value shiftAmount =
        makeEmitCIntConstant(rewriter, loc, bitsTy, floatTy.getWidth() - 1);
    Value signMask = rewriter
                         .create<emitc::BitwiseLeftShiftOp>(loc, bitsTy, oneBits,
                                                            shiftAmount)
                         .getResult();
    return rewriter
        .create<emitc::BitwiseAndOp>(loc, bitsTy, lhsBits, signMask)
        .getResult();
  }

  static Value buildSignedZeroCandidate(ConversionPatternRewriter &rewriter,
                                        Location loc, Type dstTy, Value lhs,
                                        Value rhs, FloatType floatTy) {
    Value zero = makeFZero(rewriter, loc, dstTy);
    Value equal = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::eq, lhs, rhs)
                      .getResult();
    Value lhsZero = rewriter
                        .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                              emitc::CmpPredicate::eq, lhs,
                                              zero)
                        .getResult();
    Value bothZero = rewriter
                         .create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                      equal, lhsZero)
                         .getResult();
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    Value zeroBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 0);
    Value lhsIsNegZero =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne,
                                  buildSignBitValue(rewriter, loc, lhs, floatTy),
                                  zeroBits)
            .getResult();
    Value tie = rewriter
                    .create<emitc::ConditionalOp>(
                        loc, dstTy, lhsIsNegZero, isMaximum ? rhs : lhs,
                        isMaximum ? lhs : rhs)
                    .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, bothZero, tie,
                                      buildPrimaryCandidate(rewriter, loc, dstTy,
                                                            lhs, rhs))
        .getResult();
  }

  static Value buildNaNPropagatingResult(ConversionPatternRewriter &rewriter,
                                         Location loc, Type dstTy, Value lhs,
                                         Value rhs, FloatType floatTy) {
    Value lhsNaN = isNaN(rewriter, loc, lhs);
    Value rhsNaN = isNaN(rewriter, loc, rhs);
    Value noNaN =
        buildSignedZeroCandidate(rewriter, loc, dstTy, lhs, rhs, floatTy);
    Value rhsOrNoNaN = rewriter
                           .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, rhs,
                                                         noNaN)
                           .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, lhs, rhsOrNoNaN)
        .getResult();
  }

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getType()))
      return rewriter.notifyMatchFailure(op, "expected scalar float type");

    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto floatTy = cast<FloatType>(op.getType());
    rewriter.replaceOp(op, buildNaNPropagatingResult(
                               rewriter, loc, dstTy, adaptor.getLhs(),
                               adaptor.getRhs(), floatTy));
    return success();
  }
};
using ArithMaximumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MaximumFOp, /*isMaximum=*/true>;
using ArithMinimumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MinimumFOp, /*isMaximum=*/false>;
void populateArithArithFloatMinMaxPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithMaximumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinimumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinNumFToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
