// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTensorReduce.cpp - tensor elementwise op lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

void populateTensorReducePatternsPart2(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx, PTOArch targetArch);

struct PTONotToEmitC : public OpConversionPattern<pto::TNotOp> {
  using OpConversionPattern<pto::TNotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNOT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrToEmitC : public OpConversionPattern<pto::TOrOp> {
  using OpConversionPattern<pto::TOrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrsToEmitC : public OpConversionPattern<pto::TOrSOp> {
  using OpConversionPattern<pto::TOrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc();
    Value dst  = adaptor.getDst();
    // NOTE: The conversion type system may materialize integers as emitc.opaque
    // (e.g. "int32_t"). For EmitC call emission we can pass the scalar through
    // directly without arith casts here.
    Value s = adaptor.getScalar();

    SmallVector<Value, 3> operands{dst, src0, s};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartAddToEmitC : public OpConversionPattern<pto::TPartAddOp> {
  using OpConversionPattern<pto::TPartAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMaxToEmitC : public OpConversionPattern<pto::TPartMaxOp> {
  using OpConversionPattern<pto::TPartMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMinToEmitC : public OpConversionPattern<pto::TPartMinOp> {
  using OpConversionPattern<pto::TPartMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMaxToEmitC
    : public OpConversionPattern<pto::TPartArgMaxOp> {
  using OpConversionPattern<pto::TPartArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value src0Idx = adaptor.getSrc0Idx();
    Value src1Idx = adaptor.getSrc1Idx();
    Value dst = adaptor.getDst();
    Value dstIdx = adaptor.getDstIdx();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMinToEmitC
    : public OpConversionPattern<pto::TPartArgMinOp> {
  using OpConversionPattern<pto::TPartArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value src0Idx = adaptor.getSrc0Idx();
    Value src1Idx = adaptor.getSrc1Idx();
    Value dst = adaptor.getDst();
    Value dstIdx = adaptor.getDstIdx();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMulToEmitC : public OpConversionPattern<pto::TPartMulOp> {
  using OpConversionPattern<pto::TPartMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPreluToEmitC : public OpConversionPattern<pto::TPReluOp> {
  using OpConversionPattern<pto::TPReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value tmp  = adaptor.getTmp();
    Value dst  = adaptor.getDst();

    // C++ interface: TPRELU(dst, src0, src1, tmp) — last parameter is tmp.
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORecipToEmitC : public OpConversionPattern<pto::TRecipOp> {
  using OpConversionPattern<pto::TRecipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRecipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::RecipPrecision::Default,
        "RecipAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRECIP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOReluToEmitC : public OpConversionPattern<pto::TReluOp> {
  using OpConversionPattern<pto::TReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemToEmitC : public OpConversionPattern<pto::TRemOp> {
  using OpConversionPattern<pto::TRemOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value tmp  = adaptor.getTmp();
    Value dst  = adaptor.getDst();
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::RemPrecision::Default,
        "RemAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModToEmitC : public OpConversionPattern<pto::TFModOp> {
  using OpConversionPattern<pto::TFModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, src1};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::FmodPrecision::Default,
        "FmodAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMOD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemSToEmitC : public OpConversionPattern<pto::TRemSOp> {
  using OpConversionPattern<pto::TRemSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();
    Value scalar = adaptor.getScalar();
    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREMS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModSToEmitC : public OpConversionPattern<pto::TFModSOp> {
  using OpConversionPattern<pto::TFModSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMODS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPOW DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPowToEmitC : public OpConversionPattern<pto::TPowOp> {
  using OpConversionPattern<pto::TPowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value base = adaptor.getBase();
    Value exp  = adaptor.getExp();
    Value dst  = adaptor.getDst();

    // Forms:
    //   integer:  TPOW(dst, base, exp)
    //   float:    TPOW(dst, base, exp, tmp)
    SmallVector<Value, 4> operands{dst, base, exp};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::PowPrecision::Default,
        "PowAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPOW",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPOWS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPowSToEmitC : public OpConversionPattern<pto::TPowSOp> {
  using OpConversionPattern<pto::TPowSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPowSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src    = adaptor.getSrc();
    Value dst    = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    // Forms:
    //   integer:  TPOWS(dst, src, scalar)
    //   float:    TPOWS(dst, src, scalar, tmp)
    SmallVector<Value, 4> operands{dst, src, scalar};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));

    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::PowPrecision::Default,
        "PowAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPOWS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandToEmitC : public OpConversionPattern<pto::TRowExpandOp> {
  using OpConversionPattern<pto::TRowExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};



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


// Binary row-expand ops (add/sub/mul/div/max/min): emit
// TROWEXPAND<OP>(dst, src0, src1[, tmp]) with the optional tmp operand.
// LastUseAware selects the last-use-aware call so consumed tiles carry
// their pto.last_use marker into the generated C++ (mul/div keep the
// original behavior; the other flavors never emitted it).
template <typename OpTy, bool LastUseAware = false>
struct PTORowExpandBinaryToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTORowExpandBinaryToEmitC(TypeConverter &typeConverter,
                                     MLIRContext *ctx, StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if constexpr (LastUseAware) {
      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, callee,
          collectRowExpandOperands(op, adaptor));
    } else {
      rewriter.create<emitc::CallOpaqueOp>(
          op.getLoc(), TypeRange{}, callee,
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/collectRowExpandOperands(op, adaptor));
    }
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

// Gather the (dst, src0, src1[, tmp]) operand list shared by all binary
// row-expand lowerings.
template <typename OpTy>
static SmallVector<Value, 4>
collectRowExpandOperands(OpTy op, typename OpTy::Adaptor adaptor) {
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst = adaptor.getDst();
  Value tmp = op.getTmp() ? adaptor.getTmp() : Value();

  SmallVector<Value, 4> operands;
  if (tmp)
    operands.assign({dst, src0, src1, tmp});
  else
    operands.assign({dst, src0, src1});
  return operands;
}

using PTORowExpandExpdifToEmitC =
    PTORowExpandBinaryToEmitC<pto::TRowExpandExpdifOp>;

struct PTORowExpandDivToEmitC
    : public PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true> {
  using PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true>::
      PTORowExpandBinaryToEmitC;

  PTORowExpandDivToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true>(
            typeConverter, ctx, "TROWEXPANDDIV") {}

  LogicalResult matchAndRewrite(pto::TRowExpandDivOp op,
                                OpConversionPattern<pto::TRowExpandDivOp>::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getPrecisionType() == pto::DivPrecision::Default)
      return PTORowExpandBinaryToEmitC<pto::TRowExpandDivOp, true>::
          matchAndRewrite(op, adaptor, rewriter);

    createLastUseAwareOpaqueCall(
        rewriter, op.getOperation(), TypeRange{}, "TROWEXPANDDIV",
        collectRowExpandOperands(op, adaptor), ArrayAttr{},
        buildPrecisionTemplateArgs(rewriter, op.getPrecisionType(),
                                   pto::DivPrecision::Default,
                                   "DivAlgorithm"));
    rewriter.eraseOp(op);
    return success();
  }
};

using PTORowExpandAddToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandAddOp>;
using PTORowExpandMulToEmitC =
    PTORowExpandBinaryToEmitC<pto::TRowExpandMulOp, true>;
using PTORowExpandSubToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandSubOp>;
using PTORowExpandMaxToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandMaxOp>;
using PTORowExpandMinToEmitC = PTORowExpandBinaryToEmitC<pto::TRowExpandMinOp>;

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

// Row-wise reduce ops (max/min/sum): emit TROW<OP>(dst, src, tmp) with a
// last-use-aware call so consumed tiles can be released eagerly.
template <typename OpTy>
struct PTORowReduceToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTORowReduceToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                               StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    createLastUseAwareOpaqueCall(
        rewriter, op.getOperation(), TypeRange{}, callee,
        ValueRange{adaptor.getDst(), adaptor.getSrc(), adaptor.getTmp()});
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

using PTORowMaxToEmitC = PTORowReduceToEmitC<pto::TRowMaxOp>;

struct PTORowArgMaxToEmitC
    : public OpConversionPattern<pto::TRowArgMaxOp> {
  using OpConversionPattern<pto::TRowArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//

using PTORowMinToEmitC = PTORowReduceToEmitC<pto::TRowMinOp>;

struct PTORowArgMinToEmitC
    : public OpConversionPattern<pto::TRowArgMinOp> {
  using OpConversionPattern<pto::TRowArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//

using PTORowSumToEmitC = PTORowReduceToEmitC<pto::TRowSumOp>;

struct PTOTInterleaveToEmitC
    : public OpConversionPattern<pto::TInterleaveOp> {
  using OpConversionPattern<pto::TInterleaveOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pto::TInterleaveOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    createLastUseAwareOpaqueCall(
        rewriter, op.getOperation(), TypeRange{}, "TINTERLEAVE",
        ValueRange{adaptor.getDst1(), adaptor.getDst0(), adaptor.getSrc1(),
                   adaptor.getSrc0()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTDeInterleaveToEmitC
    : public OpConversionPattern<pto::TDeInterleaveOp> {
  using OpConversionPattern<pto::TDeInterleaveOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pto::TDeInterleaveOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value dst1 = adaptor.getDsts()[1];
    Value dst0 = adaptor.getDsts()[0];
    Value src0 = adaptor.getSrcs()[0];
    bool hasSecondSource = adaptor.getSrcs().size() == 2;
    if (hasSecondSource) {
      Value src1 = adaptor.getSrcs()[1];
      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TDEINTERLEAVE",
          ValueRange{dst1, dst0, src1, src0});
    } else {
      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TDEINTERLEAVE",
          ValueRange{dst1, dst0, src0});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowProdToEmitC : public OpConversionPattern<pto::TRowProdOp> {
  using OpConversionPattern<pto::TRowProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWPROD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRSQRT DPS/memref op)
// - no-tmp form : TRSQRT(dst, src)
// - tmp form    : TRSQRT(dst, src, tmp)
//===----------------------------------------------------------------------===//

struct PTORsqrtToEmitC : public OpConversionPattern<pto::TRsqrtOp> {
  using OpConversionPattern<pto::TRsqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRsqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    SmallVector<Value, 3> operands{dst, src};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::RsqrtPrecision::Default,
        "RsqrtAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOScatterToEmitC : public OpConversionPattern<pto::TScatterOp> {
  using OpConversionPattern<pto::TScatterOp>::OpConversionPattern;

  // maskPattern flavor: TSCATTER<MaskPattern[, Axis]>(dst, src).
  void emitMaskPatternScatter(pto::TScatterOp op,
                              ConversionPatternRewriter &rewriter,
                              Value dst, Value src) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    SmallVector<Attribute, 2> targsList;
    targsList.push_back(
        emitc::OpaqueAttr::get(ctx, maskPatternTok(op.getMaskPatternAttr())));
    if (auto axisAttr = op.getAxisAttr()) {
      StringRef axisVal = axisAttr.getValue();
      std::string scatterAxis = (axisVal == "col")
                                    ? "pto::ScatterAxis::SCATTER_COL"
                                    : "pto::ScatterAxis::SCATTER_ROW";
      targsList.push_back(emitc::OpaqueAttr::get(ctx, scatterAxis));
    }
    auto targs = rewriter.getArrayAttr(targsList);
    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSCATTER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
        /*operands=*/operands);
  }

  LogicalResult matchAndRewrite(pto::TScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const bool hasMaskPattern = static_cast<bool>(op.getMaskPatternAttr());
    const bool hasIndexes = static_cast<bool>(op.getIndexes());
    if (hasMaskPattern == hasIndexes) {
      return rewriter.notifyMatchFailure(
          op, "expected exactly one of indexes operand or maskPattern attribute");
    }
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    if (hasMaskPattern) {
      emitMaskPatternScatter(op, rewriter, dst, src);
    } else {
      Value idx = adaptor.getIndexes();
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TSCATTER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src, idx});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

// tsel/tsels lowering: TSEL(dst, mask, src0, src1[, tmp]) /
// TSELS(dst, mask, src, tmp, scalar).
struct PTOSelToEmitC : public OpConversionPattern<pto::TSelOp> {
  using OpConversionPattern<pto::TSelOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(pto::TSelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value, 5> operands{adaptor.getDst(), adaptor.getMask(),
                                   adaptor.getSrc0(), adaptor.getSrc1(),
                                   adaptor.getTmp()};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSEL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSelSToEmitC : public OpConversionPattern<pto::TSelSOp> {
  using OpConversionPattern<pto::TSelSOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(pto::TSelSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value, 5> operands{adaptor.getDst(), adaptor.getMask(),
                                   adaptor.getSrc(), adaptor.getTmp(),
                                   adaptor.getScalar()};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSELS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShlSToEmitC : public OpConversionPattern<pto::TShlOp> {
  using OpConversionPattern<pto::TShlOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShrSToEmitC : public OpConversionPattern<pto::TShrOp> {
  using OpConversionPattern<pto::TShrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TSHLS/TSHRS DPS: shift by scalar)
//===----------------------------------------------------------------------===//

struct PTOShlSConstToEmitC : public OpConversionPattern<pto::TShlSOp> {
  using OpConversionPattern<pto::TShlSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = adaptor.getDst();
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHLS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOShrSConstToEmitC : public OpConversionPattern<pto::TShrSOp> {
  using OpConversionPattern<pto::TShrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = adaptor.getDst();
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHRS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (TSORT32 DPS/memref op: ins(src, idx[, tmp]) outs(dst))
//===----------------------------------------------------------------------===//

struct PTOSORT32SToEmitC : public OpConversionPattern<pto::TSort32Op> {
  using OpConversionPattern<pto::TSort32Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSort32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value idx = adaptor.getIdx();
    Value tmp = op.getTmp() ? adaptor.getTmp() : Value();

    SmallVector<Value, 4> operands;
    if (tmp) {
      operands.assign({dst, src, idx, tmp});
    } else {
      operands.assign({dst, src, idx});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSORT32",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSqrtSToEmitC : public OpConversionPattern<pto::TSqrtOp> {
  using OpConversionPattern<pto::TSqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src};
    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::SqrtPrecision::Default,
        "SqrtAlgorithm");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSTORE_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSToEmitC : public OpConversionPattern<pto::TSubOp> {
  using OpConversionPattern<pto::TSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src0, src1};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TSUB", operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubCSToEmitC : public OpConversionPattern<pto::TSubCOp> {
  using OpConversionPattern<pto::TSubCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TSUBC yet.
    // Decompose: dst = src0 - src1 + src2
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TSUB",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getSrc1(), adaptor.getSrc2());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSSToEmitC : public OpConversionPattern<pto::TSubSOp> {
  using OpConversionPattern<pto::TSubSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src, scalar};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TSUBS", operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBSC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSCToEmitC : public OpConversionPattern<pto::TSubSCOp> {
  using OpConversionPattern<pto::TSubSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TSUBSC yet.
    // Decompose: dst = src0 - scalar + src1
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TSUBS",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getScalar(), adaptor.getSrc1());
    return success();
  }
};


//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORToEmitC : public OpConversionPattern<pto::TXorOp> {
  using OpConversionPattern<pto::TXorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();
    Value tmp = adaptor.getTmp();
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTTransToEmitC : public OpConversionPattern<pto::TTransOp> {
  using OpConversionPattern<pto::TTransOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTransOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRANS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORSToEmitC : public OpConversionPattern<pto::TXorSOp> {
  using OpConversionPattern<pto::TXorSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value tmp  = adaptor.getTmp();
    Value dst = adaptor.getDst();

    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOPrintToTPRINT : public OpConversionPattern<pto::TPrintOp> {
  using OpConversionPattern<pto::TPrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto printFormatTok = [&](pto::PrintFormat format) -> StringRef {
      switch (format) {
      case pto::PrintFormat::Width8_Precision4:
        return "pto::PrintFormat::Width8_Precision4";
      case pto::PrintFormat::Width8_Precision2:
        return "pto::PrintFormat::Width8_Precision2";
      case pto::PrintFormat::Width10_Precision6:
        return "pto::PrintFormat::Width10_Precision6";
      }
      llvm_unreachable("unknown PrintFormat");
    };

    Value src = adaptor.getSrc();
    if (isa<MemRefType>(op.getSrc().getType()) ||
        isa<mlir::pto::PartitionTensorViewType>(op.getSrc().getType())) {
      src = maybeWrapGlobalMemrefAsGlobalTensor(
          rewriter, loc, src, op.getSrc().getType(), op.getOperation());
    }

    SmallVector<Value, 4> operands{src};
    if (Value tmp = op->getNumOperands() > 1 ? op->getOperand(1) : Value()) {
      Value tmpValue = adaptor.getOperands().size() > 1 ? adaptor.getOperands()[1]
                                                        : Value();
      tmpValue = peelUnrealized(tmpValue);
      if (isa<MemRefType>(tmp.getType()) ||
          isa<mlir::pto::PartitionTensorViewType>(tmp.getType())) {
        tmpValue = maybeWrapGlobalMemrefAsGlobalTensor(
            rewriter, loc, tmpValue, tmp.getType(), op.getOperation());
      }
      operands.push_back(tmpValue);
    }

    SmallVector<Attribute, 1> templateArgVec;
    if (auto formatAttr =
            dyn_cast_or_null<pto::PrintFormatAttr>(
                op.getProperties().printFormat)) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, printFormatTok(formatAttr.getValue())));
    }
    ArrayAttr templateArgs =
        templateArgVec.empty() ? ArrayAttr{} : rewriter.getArrayAttr(templateArgVec);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRINT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.print "format", %scalar -> PRINTF("format", scalar)
struct PTOPrintOpToEmitC : public OpConversionPattern<pto::PrintOp> {
  using OpConversionPattern<pto::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    std::string fmt = op.getFormat().str();
    if (fmt.empty())
      fmt = "%f";
    std::string quoted = "\"";
    for (char c : fmt) {
      if (c == '"' || c == '\\') {
        quoted += '\\';
      } else if (c == '\n') {
        quoted += "\\n";
      } else if (c == '\t') {
        quoted += "\\t";
      } else {
        quoted += c;
      }
    }
    quoted += "\"";

    Value scalar = adaptor.getScalar();
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, quoted),
         IntegerAttr::get(IndexType::get(ctx), 0)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "cce::printf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.trap -> TRAP()
struct PTOTrapOpToEmitC : public OpConversionPattern<pto::TrapOp> {
  using OpConversionPattern<pto::TrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TrapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "trap",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// Arith CmpI -> EmitC Cmp
// =============================================================================
class ArithCmpIToEmitC : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Map arith.cmpi's predicate onto the EmitC spelling; unsigned variants
  // reuse the signed EmitC predicates after operand re-interpretation.
  static LogicalResult resolveCmpIPredicate(arith::CmpIOp op,
                                            emitc::CmpPredicate &pred,
                                            bool &isUnsignedPred) {
    isUnsignedPred = op.getPredicate() == arith::CmpIPredicate::ult ||
                     op.getPredicate() == arith::CmpIPredicate::ule ||
                     op.getPredicate() == arith::CmpIPredicate::ugt ||
                     op.getPredicate() == arith::CmpIPredicate::uge;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  pred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  pred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: pred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: pred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: pred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: pred = emitc::CmpPredicate::ge; break;
      case arith::CmpIPredicate::ult: pred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::ule: pred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: pred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: pred = emitc::CmpPredicate::ge; break;
    }
    return success();
  }

  // Reinterpret unsigned-comparison operands in the unsigned C++ type of the
  // same width (i1 keeps its operands as-is).
  static FailureOr<std::pair<Value, Value>>
  adaptCmpIOperands(arith::CmpIOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    bool isUnsignedPred) {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!isUnsignedPred)
      return std::make_pair(lhs, rhs);

    Type opTy = op.getLhs().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(
          op, "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
    if (bitWidth != 1) {
      lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
      rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
    }
    return std::make_pair(lhs, rhs);
  }

  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    emitc::CmpPredicate emitcPred = emitc::CmpPredicate::eq;
    bool isUnsignedPred = false;
    if (failed(resolveCmpIPredicate(op, emitcPred, isUnsignedPred)))
      return failure();

    Type resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();

    auto operands = adaptCmpIOperands(op, adaptor, rewriter, loc,
                                      isUnsignedPred);
    if (failed(operands))
      return failure();
    Value lhs = operands->first;
    Value rhs = operands->second;

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, 
        /*resultType=*/resTy, // i1 -> bool/i1
        emitcPred,
        lhs,
        rhs
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Section Op Lowering
//===----------------------------------------------------------------------===//


void replaceOrEraseWithOpaqueCallAndReturnDst(Operation *op, Value dst,
                                                     StringRef callee,
                                                     ArrayRef<Value> args,
                                                     ArrayAttr templateArgs,
                                                     ConversionPatternRewriter &rewriter) {
  createLastUseAwareOpaqueCall(rewriter, op, TypeRange{}, callee, args, ArrayAttr{}, templateArgs);
  if (op->getNumResults() == 1) {
    rewriter.replaceOp(op, dst);
  } else {
    rewriter.eraseOp(op);
  }
}


void populateTensorReducePatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  patterns.add<ArithCmpIToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSCToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubCSToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORToEmitC>(typeConverter, ctx);
  patterns.add<PTOReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOScatterToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSqrtSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTransToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelSToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandAddToEmitC>(typeConverter, ctx, "TROWEXPANDADD");
  patterns.add<PTORowExpandExpdifToEmitC>(typeConverter, ctx,
                                          "TROWEXPANDEXPDIF");
  patterns.add<PTORowExpandMaxToEmitC>(typeConverter, ctx, "TROWEXPANDMAX");
  patterns.add<PTORowExpandMinToEmitC>(typeConverter, ctx, "TROWEXPANDMIN");
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx, "TROWEXPANDSUB");
  patterns.add<PTOShrSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOSORT32SToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTORsqrtToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMaxToEmitC>(typeConverter, ctx, "TROWMAX");
  patterns.add<PTORowArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMulToEmitC>(typeConverter, ctx, "TROWEXPANDMUL");
  patterns.add<PTORowExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTORowProdToEmitC>(typeConverter, ctx);
  patterns.add<PTORowSumToEmitC>(typeConverter, ctx, "TROWSUM");
  patterns.add<PTORowMinToEmitC>(typeConverter, ctx, "TROWMIN");
  patterns.add<PTORowArgMinToEmitC>(typeConverter, ctx);
    populateTensorReducePatternsPart2(patterns, typeConverter, ctx, targetArch);
}

void populateTensorReducePatternsPart2(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx, PTOArch targetArch) {
  (void)typeConverter;
  (void)ctx;
  (void)targetArch;
  patterns.add<PTOFModToEmitC>(typeConverter, ctx);
  patterns.add<PTORemToEmitC>(typeConverter, ctx);
  patterns.add<PTORecipToEmitC>(typeConverter, ctx);
  patterns.add<PTOPreluToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModSToEmitC>(typeConverter, ctx);
  patterns.add<PTORemSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPowToEmitC>(typeConverter, ctx);
  patterns.add<PTOPowSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTONotToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartArgMaxToEmitC, PTOPartArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrsToEmitC>(typeConverter, ctx);
  patterns.add<PTOTInterleaveToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDeInterleaveToEmitC>(typeConverter, ctx);
  patterns.add<PTOPrintToTPRINT>(typeConverter, ctx);
  patterns.add<PTOPrintOpToEmitC>(typeConverter, ctx);
  patterns.add<PTOTrapOpToEmitC>(typeConverter, ctx);
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
}

} // namespace pto
} // namespace mlir
