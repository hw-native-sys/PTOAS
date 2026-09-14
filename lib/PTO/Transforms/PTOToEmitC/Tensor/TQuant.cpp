// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TQuant.cpp - Tensor TQuant op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOQuantToEmitC : public OpConversionPattern<pto::TQuantOp> {
using OpConversionPattern<pto::TQuantOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TQuantOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value dst = adaptor.getDst();
  Value src = adaptor.getSrc();
  Value fp = adaptor.getFp();


  Value tmp;
  if (op.getTmp())
    tmp = adaptor.getTmp();
  Value offsetPtr;
  if (op.getOffset())
    offsetPtr = materializeOffsetAddress(rewriter, loc, ctx, adaptor.getOffset());

  FailureOr<ArrayAttr> templateArgsOr = buildTQuantTemplateArgs(
      op, rewriter, ctx, dst, src, fp, tmp);
  if (failed(templateArgsOr))
    return failure();
  ArrayAttr templateArgs = *templateArgsOr;

  SmallVector<Value> operands{dst, src, fp};
  if (tmp)
    operands.push_back(tmp);
  if (offsetPtr)
    operands.push_back(offsetPtr);

  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TQUANT", ArrayAttr{}, templateArgs, operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOQuantMxToEmitC : public OpConversionPattern<pto::TQuantMxOp> {
using OpConversionPattern<pto::TQuantMxOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TQuantMxOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value dst = adaptor.getDst();
  Value src = adaptor.getSrc();
  Value exp = adaptor.getExp();
  Value max = adaptor.getMax();
  Value scaling = adaptor.getScaling();
  Value expZz = adaptor.getExpZz()
                    ? adaptor.getExpZz()
                    : Value{};

  auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  auto expOT = mlir::dyn_cast<emitc::OpaqueType>(exp.getType());
  auto maxOT = mlir::dyn_cast<emitc::OpaqueType>(max.getType());
  auto scalingOT = mlir::dyn_cast<emitc::OpaqueType>(scaling.getType());
  auto expZzOT = expZz ? mlir::dyn_cast<emitc::OpaqueType>(expZz.getType())
                       : emitc::OpaqueType{};
  if (!dstOT || !srcOT || !expOT || !maxOT || !scalingOT)
    return rewriter.notifyMatchFailure(
        op, "expected all operands to be emitc::OpaqueType");
  if (expZz && !expZzOT)
    return rewriter.notifyMatchFailure(
        op, "expected exp_zz operand to be emitc::OpaqueType");


  Value expPtr = addressOfEmitCValue(rewriter, loc, ctx, exp, expOT);
  Value maxPtr = addressOfEmitCValue(rewriter, loc, ctx, max, maxOT);
  Value scalingPtr = addressOfEmitCValue(rewriter, loc, ctx, scaling, scalingOT);
  Value expZzPtr = expZz ? addressOfEmitCValue(rewriter, loc, ctx, expZz, expZzOT) : Value{};

  std::string quantTypeStr =
      op.getQuantType() == pto::QuantType::MXFP8
          ? "pto::QuantType::MXFP8"
          : "pto::QuantType::MXFP4_E2M1";

  SmallVector<Attribute> templateArgsStorage;
  if (expZz) {
    appendLegacyExpZzMxTemplateArgs(op, ctx, quantTypeStr, dstOT, srcOT,
                                    expOT, maxOT, scalingOT,
                                    templateArgsStorage);
  } else {
    appendModernMxTemplateArgs(op, ctx, templateArgsStorage);
  }
  ArrayAttr templateArgs = rewriter.getArrayAttr(templateArgsStorage);

  SmallVector<Value> operands{dst, src, expPtr, maxPtr, scalingPtr};
  if (expZzPtr)
    operands.push_back(expZzPtr);
  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TQUANT",
      /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
      /*operands=*/operands);

  rewriter.eraseOp(op);
  return success();
}
};

struct PTODequantToEmitC : public OpConversionPattern<pto::TDequantOp> {
using OpConversionPattern<pto::TDequantOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TDequantOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value dst    = adaptor.getDst();
  Value src    = adaptor.getSrc();
  Value scale  = adaptor.getScale();
  Value offset = adaptor.getOffset();

  // TDEQUANT<DstTile, SrcTile, ParaTile>(dst, src, scale, offset)
  ArrayAttr templateArgs;
  auto dstOT   = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT   = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  auto scaleOT = mlir::dyn_cast<emitc::OpaqueType>(scale.getType());
  if (dstOT && srcOT && scaleOT) {
    templateArgs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
        emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
        emitc::OpaqueAttr::get(ctx, scaleOT.getValue().str()),
    });
  } else {
    templateArgs = ArrayAttr{};
  }

  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TDEQUANT",
      /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
      /*operands=*/SmallVector<Value>{dst, src, scale, offset});

  rewriter.eraseOp(op);
  return success();
}
};

struct PTOSetQuantScalarToEmitC
    : public OpConversionPattern<mlir::pto::SetQuantScalarOp> {
  PTOSetQuantScalarToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                           PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SetQuantScalarOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::SetQuantScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outTypeAttr =
        op->getAttrOfType<StringAttr>(kEmitCScalarOutTypeAttrName);
    if (!outTypeAttr)
      return rewriter.notifyMatchFailure(
          op, "expected rematerialized fixpipe set_quant_scalar to carry emitc out type");

    std::string outTok = outTypeAttr.getValue().str();
    Value scale = adaptor.getScale();
    auto floatTy = emitc::OpaqueType::get(rewriter.getContext(), "float");
    if (scale.getType() != floatTy)
      scale = rewriter.create<emitc::CastOp>(op.getLoc(), floatTy, scale).getResult();

    ArrayAttr targs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(rewriter.getContext(), outTok)});
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "SET_QUANT_SCALAR", ArrayAttr{}, targs,
        ValueRange{scale});
    return success();
  }

  PTOArch targetArch;
};

struct PTOSetQuantVectorToEmitC
    : public OpConversionPattern<mlir::pto::SetQuantVectorOp> {
  PTOSetQuantVectorToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                           PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SetQuantVectorOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::SetQuantVectorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "SET_QUANT_VECTOR", ArrayAttr{}, ArrayAttr{},
        ValueRange{adaptor.getScalingTile()});
    return success();
  }

  PTOArch targetArch;
};

void populateTensorTQuantPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch) {
  patterns.add<PTOQuantToEmitC,
               PTOQuantMxToEmitC>(typeConverter, ctx);
  patterns.add<PTODequantToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetQuantScalarToEmitC, PTOSetQuantVectorToEmitC>(typeConverter, ctx, targetArch);
}

} // namespace pto
} // namespace mlir
