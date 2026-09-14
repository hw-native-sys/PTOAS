// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TExtractInsert.cpp - Tensor TExtractInsert op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOExtractToEmitC : public OpConversionPattern<pto::TExtractOp> {
  using OpConversionPattern<pto::TExtractOp>::OpConversionPattern;

  static SmallVector<Value, 6> collectOperands(OpAdaptor adaptor) {
    SmallVector<Value, 6> operands{adaptor.getDst(), adaptor.getSrc()};
    if (Value fp = adaptor.getFp()) {
      operands.push_back(fp);
    }
    if (Value preQuantScalar = adaptor.getPreQuantScalar()) {
      operands.push_back(preQuantScalar);
    }
    operands.push_back(adaptor.getIndexRow());
    operands.push_back(adaptor.getIndexCol());
    return operands;
  }

  static FailureOr<ArrayAttr>
  buildTemplateArgs(pto::TExtractOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) {
    auto modeAttr = op.getAccToVecModeAttr();
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (!hasMode && !hasRelu) {
      return ArrayAttr{};
    }

    auto dstType = dyn_cast<emitc::OpaqueType>(adaptor.getDst().getType());
    auto srcType = dyn_cast<emitc::OpaqueType>(adaptor.getSrc().getType());
    if (!dstType || !srcType) {
      return failure();
    }
    SmallVector<Attribute, 4> args{
        emitc::OpaqueAttr::get(rewriter.getContext(), dstType.getValue().str()),
        emitc::OpaqueAttr::get(rewriter.getContext(), srcType.getValue().str()),
    };
    if (Value fp = adaptor.getFp()) {
      auto fpType = dyn_cast<emitc::OpaqueType>(fp.getType());
      if (!fpType) {
        return failure();
      }
      args.push_back(
          emitc::OpaqueAttr::get(rewriter.getContext(), fpType.getValue().str()));
    }
    if (hasMode) {
      args.push_back(emitc::OpaqueAttr::get(
          rewriter.getContext(), getAccToVecModeToken(modeAttr.getValue())));
    }
    args.push_back(emitc::OpaqueAttr::get(
        rewriter.getContext(), getReluPreModeToken(op.getReluPreMode())));
    return rewriter.getArrayAttr(args);
  }

  LogicalResult matchAndRewrite(pto::TExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto templateArgs = buildTemplateArgs(op, adaptor, rewriter);
    if (failed(templateArgs)) {
      return rewriter.notifyMatchFailure(
          op, "textract template lowering expects opaque dst/src/fp types");
    }

    const bool hasFp = static_cast<bool>(adaptor.getFp());
    const bool hasMode = static_cast<bool>(op.getAccToVecModeAttr());
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, hasFp && !hasMode ? "TEXTRACT_FP" : "TEXTRACT",
        ArrayAttr{}, *templateArgs, collectOperands(adaptor));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOInsertToEmitC : public OpConversionPattern<pto::TInsertOp> {
  using OpConversionPattern<pto::TInsertOp>::OpConversionPattern;

  // Resolve the TINSERT overload's template arguments: the tinsert mode when
  // present, otherwise the spelled-out dst/src (and fp) types plus mode/relu
  // tokens when any optional operand is in play.
  LogicalResult buildTemplateArgs(pto::TInsertOp op,
                                  ConversionPatternRewriter &rewriter,
                                  Value dst, Value src, Value fp, Value preQuantScalar,
                                  ArrayAttr &templateArgs) const {
    auto *ctx = rewriter.getContext();
    auto modeAttr = op.getAccToVecModeAttr();
    auto tinsertModeAttr = op.getTinsertModeAttr();
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault =
        op.getReluPreMode() != pto::ReluPreMode::NoRelu;

    templateArgs = ArrayAttr{};
    if (tinsertModeAttr) {
      templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
          ctx, getTInsertModeToken(tinsertModeAttr.getValue()))});
    } else if (hasFp || hasPreQuantScalar || hasMode || reluNonDefault) {
      auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
      auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
      if (!dstOT || !srcOT)
        return rewriter.notifyMatchFailure(
            op, "tinsert template lowering expects opaque dst/src types");
      SmallVector<Attribute, 5> args{
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
      };
      if (hasFp) {
        auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
        if (!fpOT)
          return rewriter.notifyMatchFailure(
              op, "tinsert template lowering expects opaque fp type");
        args.push_back(emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
      }
      pushModeAndReluTemplateArgs(args, ctx, modeAttr, reluNonDefault,
                                  op.getReluPreMode());
      // TINSERT always spells out the relu-pre-mode token.
      if (!modeAttr && !reluNonDefault)
        args.push_back(emitc::OpaqueAttr::get(
            ctx, getReluPreModeToken(op.getReluPreMode())));
      templateArgs = rewriter.getArrayAttr(args);
    }
    if (hasFp && !hasMode && !reluNonDefault)
      templateArgs = ArrayAttr{};
    return success();
  }

  LogicalResult matchAndRewrite(pto::TInsertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value r0 = adaptor.getIndexRow();
    Value c0 = adaptor.getIndexCol();
    Value fp;
    if (op.getFp())
      fp = adaptor.getFp();
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = adaptor.getPreQuantScalar();

    SmallVector<Value, 6> operands{dst, src};
    if (fp)
      operands.push_back(fp);
    if (preQuantScalar)
      operands.push_back(preQuantScalar);
    operands.push_back(r0);
    operands.push_back(c0);

    ArrayAttr templateArgs;
    if (failed(buildTemplateArgs(op, rewriter, dst, src, fp, preQuantScalar,
                                 templateArgs)))
      return failure();

    const bool hasMode = static_cast<bool>(op.getAccToVecModeAttr());
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, fp && !hasMode ? "TINSERT_FP" : "TINSERT",
        ArrayAttr{}, templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }

};

void populateTensorTExtractInsertPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOExtractToEmitC, PTOInsertToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
