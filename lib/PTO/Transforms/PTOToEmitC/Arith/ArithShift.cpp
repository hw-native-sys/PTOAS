// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithShift.cpp - Arith ArithShift op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

template <typename ArithOp, typename EmitCShiftOp>
struct ArithShiftToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;

    if (getScalarIntOrIndexBitWidth(op.getType()) == 1) {
      // Widen to u8, shift, and truncate back to i1.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<EmitCShiftOp>(loc, u8Ty, lhsU8, rhsU8);
      Value masked = rewriter.create<emitc::BitwiseAndOp>(
          loc, u8Ty, sh,
          makeEmitCIntConstant(rewriter, loc, u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    return emitUnsignedBinaryAndReplace<EmitCShiftOp>(op, adaptor, rewriter,
                                                      loc, dstTy);
  }
};
struct ArithShiftRightSIToEmitC : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;
    const unsigned bitWidth = getScalarIntOrIndexBitWidth(op.getType());

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    // Signed arithmetic shift; cast RHS to unsigned to interpret shift amount.
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value sh =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, dstTy, adaptor.getLhs(),
                                                    rhsU);
    rewriter.replaceOp(op, sh);
    return success();
  }
};

using ArithShiftLeftToEmitC =
    ArithShiftToEmitC<arith::ShLIOp, emitc::BitwiseLeftShiftOp>;
using ArithShiftRightUIToEmitC =
    ArithShiftToEmitC<arith::ShRUIOp, emitc::BitwiseRightShiftOp>;
void populateArithArithShiftPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithShiftLeftToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightSIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
