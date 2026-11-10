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
#include "PTO/Support/CodeConstants.h"

#include <string>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// Per-operation HF32 control scope. `tf32_mode` on a tile matmul selects HF32
// input rounding for that one operation. The hardware keeps both the enable bit
// and the rounding-mode bit in CTRL (CTRL[46] = HF32 enable, CTRL[47] = rounding
// mode), so the generated code saves the whole CTRL value, applies the requested
// bits, issues the matmul and restores the entry state. Restoring the full word
// (instead of clearing bit 46 only) keeps an HF32 op from leaking its rounding
// mode into later ordinary FP32 matmuls.

/// Unique suffix so that sibling scopes never produce identical verbatim text
/// (keeps the statements distinguishable in generated code and in tests).
static unsigned getNextTf32CtrlScopeId()
{
    static unsigned nextId = 0;
    return nextId++;
}

/// Emits `{ uint64_t <saved> = get_ctrl(); set_ctrl(...);` and returns the
/// variable name that the matching exit call must restore.
static std::string emitTf32CtrlScopeEnter(
    Location loc, pto::Tf32Mode mode, ConversionPatternRewriter &rewriter)
{
    std::string saved =
        "__pto_saved_ctrl_" + std::to_string(getNextTf32CtrlScopeId());
    std::string enableExpr =
        "sbitset1(" + saved + ", " + std::to_string(kValue46) + ")";
    std::string ctrlExpr =
        mode == pto::Tf32Mode::RoundAway
            ? "sbitset1(" + enableExpr + ", " + std::to_string(kValue47) + ")"
            : "sbitset0(" + enableExpr + ", " + std::to_string(kValue47) + ")";

    rewriter.create<emitc::VerbatimOp>(loc, "{");
    rewriter.create<emitc::VerbatimOp>(
        loc, "uint64_t " + saved + " = get_ctrl();");
    rewriter.create<emitc::VerbatimOp>(loc, "set_ctrl(" + ctrlExpr + ");");
    return saved;
}

/// Closes the scope opened by emitTf32CtrlScopeEnter, restoring CTRL.
static void emitTf32CtrlScopeExit(
    Location loc, StringRef saved, ConversionPatternRewriter &rewriter)
{
    rewriter.create<emitc::VerbatimOp>(loc, "set_ctrl(" + saved.str() + ");");
    rewriter.create<emitc::VerbatimOp>(loc, "}");
}

/// Shared body of the tile matmul EmitC lowerings: prepares the acc_phase
/// template arguments and wraps the emitted call in the per-operation HF32 CTRL
/// scope when the op asks for one. Keeping the bookkeeping here avoids repeating
/// it in the tmatmul / tmatmul.acc / tmatmul.bias patterns.
template <typename OpT, typename EmitCallFn>
static LogicalResult lowerTileMatmulEmitC(
    OpT op, ConversionPatternRewriter &rewriter, EmitCallFn emitCall)
{
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    Location loc = op.getLoc();
    std::string savedCtrl;
    if (std::optional<pto::Tf32Mode> mode = op.getTf32Mode()) {
        savedCtrl = emitTf32CtrlScopeEnter(loc, *mode, rewriter);
    }
    emitCall(templateArgs);
    if (!savedCtrl.empty()) {
        emitTf32CtrlScopeExit(loc, savedCtrl, rewriter);
    }
    return success();
}

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

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale, bias}, templateArgs,
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

    return lowerTileMatmulEmitC(op, rewriter, [&](ArrayAttr templateArgs) {
        replaceOrEraseWithOpaqueCallAndReturnDst(
            op.getOperation(), dst, "TMATMUL_BIAS", {dst, a, b, bias},
            templateArgs, rewriter);
    });
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

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, a, aScale, b, bScale, bias}, templateArgs,
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

    // 2. acc_phase 与 tf32_mode 的公共处理（模板参数 + CTRL 作用域）
    return lowerTileMatmulEmitC(op, rewriter, [&](ArrayAttr templateArgs) {
        emitTileCallAndReplace(op.getOperation(), rewriter, "TMATMUL",
            templateArgs, ValueRange{dst, lhs, rhs}, dst);
    });
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

    // 2. acc_phase 与 tf32_mode 的公共处理（模板参数 + CTRL 作用域）
    return lowerTileMatmulEmitC(op, rewriter, [&](ArrayAttr templateArgs) {
        emitTileCallAndReplace(op.getOperation(), rewriter, "TMATMUL_ACC",
            templateArgs, ValueRange{dst, accIn, lhs, rhs}, dst);
    });
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
