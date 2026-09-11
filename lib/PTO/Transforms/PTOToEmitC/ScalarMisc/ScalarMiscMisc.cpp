// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ScalarMiscMisc.cpp - ScalarMisc ScalarMiscMisc op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOPtrToIntToEmitC : public OpConversionPattern<pto::PtrToIntOp> {
  using OpConversionPattern<pto::PtrToIntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PtrToIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return failure();

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), dstTy, "reinterpret_cast", ArrayAttr{}, templateArgs,
        ValueRange{ptr});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct PTOIntToPtrToEmitC : public OpConversionPattern<pto::IntToPtrOp> {
  using OpConversionPattern<pto::IntToPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::IntToPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value addr = adaptor.getAddr();
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    Type dstElemTy = getPointerLikeElementType(op.getResult().getType());
    if (!dstElemTy)
      return failure();

    std::string castType =
        std::string("__gm__ ") + getEmitCScalarTypeToken(dstElemTy) + "*";
    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      castType)});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), dstTy, "reinterpret_cast", ArrayAttr{}, templateArgs,
        ValueRange{addr});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

void populateScalarMiscScalarMiscMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOPtrToIntToEmitC>(typeConverter, ctx);
  patterns.add<PTOIntToPtrToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
