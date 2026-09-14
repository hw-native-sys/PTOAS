// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- LocalArray.cpp - ScalarMisc LocalArray op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTODeclareLocalArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareLocalArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareLocalArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareLocalArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map !pto.local_array type");

    auto var = rewriter
                   .create<emitc::VariableOp>(
                       op.getLoc(), getEmitCVariableResultType(arrayTy),
                       emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                   .getResult();
    var = loadEmitCVariableIfNeeded(rewriter, op.getLoc(), var);
    rewriter.replaceOp(op, var);
    return success();
  }
};

struct PTOLocalArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(
          op, "failed to map local_array element type");

    Value array = adaptor.getArray();
    SmallVector<Value> indices;
    indices.reserve(adaptor.getIndices().size());
    for (Value index : adaptor.getIndices())
      indices.push_back(peelUnrealized(index));

    auto sub = rewriter.create<emitc::SubscriptOp>(op.getLoc(), resultTy,
                                                   array, indices);
    auto snapshot =
        rewriter
            .create<emitc::VariableOp>(
                op.getLoc(), resultTy,
                emitc::OpaqueAttr::get(rewriter.getContext(), ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), snapshot, sub.getResult());
    rewriter.replaceOp(op, snapshot);
    return success();
  }
};

struct PTOLocalArraySetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArraySetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Type elemTy = value.getType();

    Value slot = rewriter
                     .create<emitc::SubscriptOp>(
                         op.getLoc(), elemTy, adaptor.getArray(),
                         adaptor.getIndices())
                     .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), slot, value);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateScalarMiscLocalArrayPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTODeclareLocalArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArraySetToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
