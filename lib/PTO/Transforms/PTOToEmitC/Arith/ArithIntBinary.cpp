// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithIntBinary.cpp - Arith ArithIntBinary op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

template <typename ArithOp, typename EmitCOp, typename EmitCI1Op>
struct ArithIntBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using PatternTy = ArithIntBinaryToEmitC;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return emitScalarIntBinary<PatternTy, ArithOp, EmitCOp, EmitCI1Op>(
        this, op, adaptor, rewriter);
  }
};
using ArithMulIToEmitC =
    ArithIntBinaryToEmitC<arith::MulIOp, emitc::MulOp, emitc::BitwiseAndOp>;
using ArithAddIToEmitC =
    ArithIntBinaryToEmitC<arith::AddIOp, emitc::AddOp, emitc::BitwiseXorOp>;
using ArithSubIToEmitC =
    ArithIntBinaryToEmitC<arith::SubIOp, emitc::SubOp, emitc::BitwiseXorOp>;
void populateArithArithIntBinaryPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithMulIToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddIToEmitC>(typeConverter, ctx);
  patterns.add<ArithSubIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
