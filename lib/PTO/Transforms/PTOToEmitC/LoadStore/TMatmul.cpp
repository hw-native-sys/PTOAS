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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <string>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// TF32 regions are planned before dialect conversion, independently of rewrite
// order. Only known CTRL-neutral operations and structured single-entry/exit
// control flow may share a region. Unknown operations are hard boundaries.
// CTRL-to-MMAD ordering follows the target intrinsic contract, as in pto-isa;
// region boundaries do not introduce M-pipe memory synchronization.

/// Emits `uint64_t <saved> = get_ctrl(); set_ctrl(...);` and returns the
/// variable name that the matching exit call must restore.
static std::string emitTf32CtrlScopeEnter(
    Location loc, pto::Tf32Mode mode, unsigned scopeId, OpBuilder &rewriter)
{
    std::string saved =
        "__pto_saved_ctrl_" + std::to_string(scopeId);
    std::string enableExpr =
        "sbitset1(" + saved + ", " + std::to_string(kValue46) + ")";
    std::string ctrlExpr =
        mode == pto::Tf32Mode::RoundAway
            ? "sbitset1(" + enableExpr + ", " + std::to_string(kValue47) + ")"
            : "sbitset0(" + enableExpr + ", " + std::to_string(kValue47) + ")";

    rewriter.create<emitc::VerbatimOp>(
        loc, "uint64_t " + saved + " = get_ctrl();");
    rewriter.create<emitc::VerbatimOp>(loc, "set_ctrl(" + ctrlExpr + ");");
    return saved;
}

/// Closes the scope opened by emitTf32CtrlScopeEnter, restoring CTRL.
static void emitTf32CtrlScopeExit(
    Location loc, StringRef saved, OpBuilder &rewriter)
{
    rewriter.create<emitc::VerbatimOp>(loc, "set_ctrl(" + saved.str() + ");");
}

static bool isTileMatmul(Operation *op)
{
    return isa<pto::TMatmulOp, pto::TMatmulAccOp, pto::TMatmulBiasOp>(op);
}

static bool collectTf32Region(Operation *op, Attribute &mode,
                              SmallVectorImpl<Operation *> &matmuls);

static bool collectNestedTf32Regions(Operation *op, Attribute &mode,
                                     SmallVectorImpl<Operation *> &matmuls)
{
    for (Region &region : op->getRegions()) {
        for (Block &block : region) {
            for (Operation &nested : block) {
                if (!collectTf32Region(&nested, mode, matmuls)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// A missing mode is neutral only for operations known not to inspect CTRL.
// Ordinary FP32 matmul and unknown calls must terminate the current region.
static bool collectTf32Region(Operation *op, Attribute &mode,
                              SmallVectorImpl<Operation *> &matmuls)
{
    if (isTileMatmul(op)) {
        Attribute requested = op->getAttr("tf32_mode");
        if (!requested || (mode && requested != mode)) {
            return false;
        }
        mode = requested;
        matmuls.push_back(op);
        return true;
    }
    if (isa<scf::ForOp, scf::IfOp>(op)) {
        return collectNestedTf32Regions(op, mode, matmuls);
    }
    StringRef name = op->getName().getStringRef();
    if (isa<func::CallOp, pto::GetCtrlOp, pto::SetCtrlOp>(op)) {
        return false;
    }
    return op->getNumRegions() == 0 &&
           (isMemoryEffectFree(op) || isa<scf::YieldOp>(op) ||
            name == "pto.tload" || name == "pto.tstore" ||
            name == "pto.tmov" || name == "pto.set_flag" ||
            name == "pto.wait_flag" || name == "pto.barrier");
}

static void emitTf32Region(Operation *first, Operation *last, Attribute mode, unsigned scopeId)
{
    OpBuilder builder(first);
    // Keep descriptors visible after CTRL restoration; no C++ brace scope.
    std::string saved = emitTf32CtrlScopeEnter(
        first->getLoc(), cast<pto::Tf32ModeAttr>(mode).getValue(), scopeId, builder);
    builder.setInsertionPointAfter(last);
    emitTf32CtrlScopeExit(last->getLoc(), saved, builder);
}

static void prepareTf32Block(Block &block, unsigned &scopeId);

static void prepareNestedTf32Blocks(Operation &op, unsigned &scopeId)
{
    for (Region &region : op.getRegions()) {
        for (Block &nested : region) {
            prepareTf32Block(nested, scopeId);
        }
    }
}

static void prepareTf32Block(Block &block, unsigned &scopeId)
{
    Operation *first = nullptr;
    Operation *last = nullptr;
    Attribute mode;
    auto flush = [&first, &last, &mode, &scopeId]() {
        if (first) {
            emitTf32Region(first, last, mode, scopeId++);
        }
        first = last = nullptr;
        mode = {};
    };
    for (Operation &op : llvm::make_early_inc_range(block)) {
        Attribute nextMode = mode;
        SmallVector<Operation *> nextMatmuls;
        if (!collectTf32Region(&op, nextMode, nextMatmuls)) {
            flush();
            nextMode = {};
            nextMatmuls.clear();
            if (!collectTf32Region(&op, nextMode, nextMatmuls)) {
                prepareNestedTf32Blocks(op, scopeId);
                continue;
            }
        }
        if (!nextMatmuls.empty()) {
            if (!first) {
                first = &op;
            }
            last = &op;
            mode = nextMode;
        }
    }
    flush();
}

void prepareTileMatmulTf32Regions(ModuleOp module)
{
    unsigned scopeId = 0;
    module.walk([&](func::FuncOp func) {
        for (Block &block : func.getBody()) {
            prepareTf32Block(block, scopeId);
        }
    });
}

// CTRL regions have already been materialized before conversion. Keep the
// per-op pattern concerned only with the matmul call and accumulation phase.
template <typename OpT, typename EmitCallFn>
static LogicalResult lowerTileMatmulEmitC(
    OpT op, ConversionPatternRewriter &rewriter, EmitCallFn emitCall)
{
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    emitCall(templateArgs);
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

    // 2. Emit accumulation-phase template arguments; CTRL regions are preplanned.
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

    // 2. Emit accumulation-phase template arguments; CTRL regions are preplanned.
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
