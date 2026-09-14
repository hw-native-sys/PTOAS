// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- GlobalEventArray.cpp - ScalarMisc GlobalEventArray op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTODeclareGlobalToEmitC
    : public OpConversionPattern<mlir::pto::DeclareGlobalOp> {
  using OpConversionPattern<
      mlir::pto::DeclareGlobalOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareGlobalOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getEntry().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_global result type");
    if (auto tvTy = dyn_cast<TensorViewType>(op.getEntry().getType())) {
      if (auto stridesAttr =
              op->getAttrOfType<DenseI64ArrayAttr>(kGlobalTensorStridesAttrName)) {
        auto strides = stridesAttr.asArrayRef();
        if (strides.size() == static_cast<size_t>(tvTy.getRank())) {
          convertedType = emitc::OpaqueType::get(
              rewriter.getContext(),
              getGlobalTensorTypeStringFromShapeAndStrides(
                  tvTy.getElementType(), tvTy.getShape(), strides));
        }
      }
    }
    auto var = rewriter.create<emitc::VariableOp>(
        op.getLoc(), getEmitCVariableResultType(convertedType),
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(
        op, loadEmitCVariableIfNeeded(rewriter, op.getLoc(), var.getResult()));
    return success();
  }
};

struct PTODeclareEventIdArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareEventIdArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareEventIdArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareEventIdArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map declared eventid_array type");

    auto array = rewriter
                     .create<emitc::VariableOp>(
                         op.getLoc(), getEmitCVariableResultType(arrayTy),
                         emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                     .getResult();
    array = loadEmitCVariableIfNeeded(rewriter, op.getLoc(), array);
    rewriter.replaceOp(op, array);
    return success();
  }
};

struct PTOEventIdArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = adaptor.getArray();
    Value index = adaptor.getIndex();

    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map eventid_array get result type");

    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), resultTy, array, ValueRange{index});
    rewriter.replaceOp(op, subscript.getResult());
    return success();
  }
};

struct PTOEventIdArraySetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArraySetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = adaptor.getArray();
    Value index = adaptor.getIndex();
    Value value = adaptor.getValue();

    Value slot = rewriter
                     .create<emitc::SubscriptOp>(
                         op.getLoc(), value.getType(), array,
                         ValueRange{index})
                     .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), slot, value);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateScalarMiscGlobalEventArrayPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTODeclareGlobalToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareEventIdArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArraySetToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
