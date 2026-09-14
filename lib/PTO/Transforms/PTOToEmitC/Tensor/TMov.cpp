// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TMov.cpp - Tensor TMov op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOMovToEmitC : public OpConversionPattern<pto::TMovOp> {
using OpConversionPattern<pto::TMovOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMovOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value src = adaptor.getSrc();
  Value dst = adaptor.getDst();
  Value fp;
  if (op.getFp())
    fp = adaptor.getFp();
  Value preQuantScalar;
  if (op.getPreQuantScalar())
    preQuantScalar = adaptor.getPreQuantScalar();

  auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  if (!dstOT || !srcOT)
    return rewriter.notifyMatchFailure(
        op, "tmov lowering expects opaque dst/src types");

  auto modeAttr = op.getAccToVecModeAttr();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasMode = static_cast<bool>(modeAttr);
  const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;

  SmallVector<Value, 4> operands{dst, src};
  SmallVector<Attribute, 5> templateArgVec{
      emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
      emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
  };
  StringRef callee = "TMOV";

  if (failed(buildTMovOperandsAndTemplates(op, rewriter, ctx, fp,
                                           preQuantScalar, modeAttr, operands,
                                           templateArgVec, callee)))
    return failure();

  const bool isXToZz =
      hasFp && pto::classifyTMovForm(op.getFp()) == pto::TMovForm::XToZz;
  ArrayAttr templateArgs =
      (isXToZz && templateArgVec.empty()) ||
              (templateArgVec.size() == 2 && !hasFp && !preQuantScalar &&
              !hasMode && !reluNonDefault)
          ? ArrayAttr{}
          : rewriter.getArrayAttr(templateArgVec);

  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, callee,
      /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
      /*operands=*/operands);

  rewriter.eraseOp(op);
  return success();
}

  // Collect the TMOV operands and template tokens for the operand flavor
  // (fp / preQuant / mode / relu).
  LogicalResult buildTMovOperandsAndTemplates(
      pto::TMovOp op, ConversionPatternRewriter &rewriter,
      MLIRContext *ctx, Value fp, Value preQuantScalar,
      pto::AccToVecModeAttr modeAttr, SmallVectorImpl<Value> &operands,
      SmallVectorImpl<Attribute> &templateArgVec,
      StringRef &callee) const {
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    const bool isXToZz =
        hasFp && pto::classifyTMovForm(op.getFp()) == pto::TMovForm::XToZz;

    if (hasFp) {
      auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
      if (!fpOT)
        return rewriter.notifyMatchFailure(
            op, "tmov fp lowering expects opaque fp type");
      operands.push_back(fp);
      if (isXToZz) {
        templateArgVec.clear();
        if (op.getGrpAxisAttr() &&
            op.getGrpAxisAttr().getValue() == pto::MxGroupAxis::Axis0)
          templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, "0"));
      } else {
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
        pushModeAndReluTemplateArgs(templateArgVec, ctx, modeAttr,
                                    reluNonDefault, op.getReluPreMode());
        callee = hasMode ? "TMOV" : "TMOV_FP";
      }
    } else if (hasPreQuantScalar) {
      operands.push_back(preQuantScalar);
      pushModeAndReluTemplateArgs(templateArgVec, ctx, modeAttr,
                                  reluNonDefault, op.getReluPreMode());
    } else if (hasMode) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, getAccToVecModeToken(modeAttr.getValue())));
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, getReluPreModeToken(op.getReluPreMode())));
    } else if (reluNonDefault) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, getReluPreModeToken(op.getReluPreMode())));
    }
    return success();
  }
};

void populateTensorTMovPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOMovToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
