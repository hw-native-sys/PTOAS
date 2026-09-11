// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TMatmul.cpp - LoadStore TMatmul op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "LoadStoreInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOTGemvBiasToTGEMV_BIAS
    : public OpConversionPattern<pto::TGemvBiasOp> {
  using OpConversionPattern<pto::TGemvBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = adaptor.getA();
    Value b    = adaptor.getB();
    Value bias = adaptor.getBias();
    Value dst  = adaptor.getDst();

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_BIAS",
                                             {dst, a, b, bias}, templateArgs, rewriter);
    return success();
  }
};

struct PTOTGemvMXToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxOp> {
  using OpConversionPattern<pto::TGemvMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = adaptor.getA();
    Value aScale  = adaptor.getAScale();
    Value b       = adaptor.getB();
    Value bScale  = adaptor.getBScale();
    Value dst     = adaptor.getDst();

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTGemvMXAccToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxAccOp> {
  using OpConversionPattern<pto::TGemvMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = adaptor.getCIn();
    Value a       = adaptor.getA();
    Value aScale  = adaptor.getAScale();
    Value b       = adaptor.getB();
    Value bScale  = adaptor.getBScale();
    Value dst     = adaptor.getDst();

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, cIn, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTGemvMXBiasToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxBiasOp> {
  using OpConversionPattern<pto::TGemvMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = adaptor.getA();
    Value aScale  = adaptor.getAScale();
    Value b       = adaptor.getB();
    Value bScale  = adaptor.getBScale();
    Value bias    = adaptor.getBias();
    Value dst     = adaptor.getDst();

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale, bias}, ArrayAttr{},
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulBiasToTMATMUL_BIAS
    : public OpConversionPattern<pto::TMatmulBiasOp> {
  using OpConversionPattern<pto::TMatmulBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = adaptor.getA();
    Value b    = adaptor.getB();
    Value bias = adaptor.getBias();
    Value dst  = adaptor.getDst();

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_BIAS",
                                             {dst, a, b, bias}, templateArgs, rewriter);
    return success();
  }
};

struct PTOTMatmulMXToTMATMUL_MX
    : public OpConversionPattern<pto::TMatmulMxOp> {
  using OpConversionPattern<pto::TMatmulMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = adaptor.getA();
    Value aScale  = adaptor.getAScale();
    Value b       = adaptor.getB();
    Value bScale  = adaptor.getBScale();
    Value dst     = adaptor.getDst();

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulMXAccToTMATMUL_MX_ACC
    : public OpConversionPattern<pto::TMatmulMxAccOp> {
  using OpConversionPattern<pto::TMatmulMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = adaptor.getCIn();
    Value a       = adaptor.getA();
    Value aScale  = adaptor.getAScale();
    Value b       = adaptor.getB();
    Value bScale  = adaptor.getBScale();
    Value dst     = adaptor.getDst();

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, cIn, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulMXBiasToTMATMUL_MX_BIAS
    : public OpConversionPattern<pto::TMatmulMxBiasOp> {
  using OpConversionPattern<pto::TMatmulMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = adaptor.getA();
    Value aScale  = adaptor.getAScale();
    Value b       = adaptor.getB();
    Value bScale  = adaptor.getBScale();
    Value bias    = adaptor.getBias();
    Value dst     = adaptor.getDst();

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, a, aScale, b, bScale, bias}, ArrayAttr{},
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulToTMATMUL : public OpConversionPattern<pto::TMatmulOp> {
  using OpConversionPattern<pto::TMatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取转换后的目标侧操作数
    Value lhs = adaptor.getLhs(); // A (Left)
    Value rhs = adaptor.getRhs(); // B (Right)
    Value dst = adaptor.getDst(); // C (Acc)

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TMATMUL",
                           templateArgs, ValueRange{dst, lhs, rhs}, dst);
    return success();
  }
};

struct PTOTGemvToTGEMV : public OpConversionPattern<pto::TGemvOp> {
  using OpConversionPattern<pto::TGemvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取转换后的目标侧操作数
    Value lhs = adaptor.getLhs(); // A (Matrix)
    Value rhs = adaptor.getRhs(); // B (Vector)
    Value dst = adaptor.getDst(); // C (Result)

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TGEMV",
                           templateArgs, ValueRange{dst, lhs, rhs}, dst);
    return success();
  }
};

struct PTOTGemvAccToTGEMVACC : public OpConversionPattern<pto::TGemvAccOp> {
  using OpConversionPattern<pto::TGemvAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tgemv.acc");

    // 1. 获取操作数
    Value accIn = adaptor.getAccIn(); // AccOld
    Value lhs   = adaptor.getLhs();   // A (Matrix)
    Value rhs   = adaptor.getRhs();   // B (Vector)
    Value dst   = adaptor.getDst();   // AccNew

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TGEMV_ACC",
                           templateArgs, ValueRange{dst, accIn, lhs, rhs}, dst);
    return success();
  }
};

struct PTOTMatmulAccToTMATMULACC : public OpConversionPattern<pto::TMatmulAccOp> {
  using OpConversionPattern<pto::TMatmulAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tmatmul.acc");

    // 1. 获取操作数
    Value accIn = adaptor.getAccIn(); // AccOld
    Value lhs   = adaptor.getLhs();   // A (Left)
    Value rhs   = adaptor.getRhs();   // B (Right)
    Value dst   = adaptor.getDst();   // AccNew

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL_ACC<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TMATMUL_ACC",
                           templateArgs, ValueRange{dst, accIn, lhs, rhs}, dst);
    return success();
  }
};

void populateLoadStoreTMatmulPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMXToTMATMUL_MX,
    PTOTMatmulMXAccToTMATMUL_MX_ACC,
    PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMXToTMATMUL_MX,
    PTOTMatmulMXAccToTMATMUL_MX_ACC,
    PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
    PTOTGemvBiasToTGEMV_BIAS,
    PTOTGemvMXToTGEMV_MX,
    PTOTGemvMXAccToTGEMV_MX,
    PTOTGemvMXBiasToTGEMV_MX
  >(typeConverter, ctx);
  patterns.add<PTOTMatmulToTMATMUL>(typeConverter, ctx);
  patterns.add<PTOTMatmulAccToTMATMULACC>(typeConverter, ctx);
  patterns.add<PTOTGemvToTGEMV>(typeConverter, ctx);
  patterns.add<PTOTGemvAccToTGEMVACC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
