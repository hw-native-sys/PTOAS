// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithIntDiv.cpp - Arith ArithIntDiv op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ArithCeilDivUIToEmitC : public OpConversionPattern<arith::CeilDivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto operands = getUnsignedBinaryOperands(op.getOperation(), adaptor.getLhs(),
                                              adaptor.getRhs(), rewriter);
    if (failed(operands))
      return failure();
    auto &uTy = operands->uTy;
    Value &lhsU = operands->lhs;
    Value &rhsU = operands->rhs;
    Value one = makeEmitCIntConstant(rewriter, loc, uTy, 1);
    Value rhsMinusOne = rewriter.create<emitc::SubOp>(loc, uTy, rhsU, one);
    Value num = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsMinusOne);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, num, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp, bool IsCeil>
struct ArithSignedRoundedDivToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    SignedDivParts parts = buildSignedDivParts(rewriter, loc, dstTy,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    Value signHolds = IsCeil ? parts.signsSame : parts.signsDiffer;
    Value adjust = rewriter.create<emitc::LogicalAndOp>(
        loc, rewriter.getI1Type(), parts.remainderNonZero, signHolds);
    Value compensated =
        IsCeil ? rewriter.create<emitc::AddOp>(loc, dstTy, parts.quotient,
                                               one).getResult()
               : rewriter.create<emitc::SubOp>(loc, dstTy, parts.quotient,
                                               one).getResult();
    Value result =
        rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust, compensated,
                                              parts.quotient);
    rewriter.replaceOp(op, result);
    return success();
  }
};
struct ArithDivSIToEmitC : public OpConversionPattern<arith::DivSIOp> {
  using OpConversionPattern<arith::DivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithRemSIToEmitC : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::RemOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

using ArithCeilDivSIToEmitC =
    ArithSignedRoundedDivToEmitC<arith::CeilDivSIOp, true>;
using ArithFloorDivSIToEmitC =
    ArithSignedRoundedDivToEmitC<arith::FloorDivSIOp, false>;
void populateArithArithIntDivPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithCeilDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithFloorDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemSIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
