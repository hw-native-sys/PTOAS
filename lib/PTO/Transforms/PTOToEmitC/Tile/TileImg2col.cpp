// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. Do not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TileImg2col.cpp - ConvTile / IMG2COL op lowering ------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOSetFmatrixToEmitC : public OpConversionPattern<pto::SetFmatrixOp> {
  using OpConversionPattern<pto::SetFmatrixOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetFmatrixOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "SETFMATRIX",
                                 ValueRange{peelUnrealized(adaptor.getSrc())},
                                 ArrayAttr{}, templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetImg2colRptToEmitC
    : public OpConversionPattern<pto::SetImg2colRptOp> {
  using OpConversionPattern<pto::SetImg2colRptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetImg2colRptOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "SET_IMG2COL_RPT",
                                 ValueRange{peelUnrealized(adaptor.getSrc())},
                                 ArrayAttr{}, templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetImg2colPaddingToEmitC
    : public OpConversionPattern<pto::SetImg2colPaddingOp> {
  using OpConversionPattern<pto::SetImg2colPaddingOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetImg2colPaddingOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "SET_IMG2COL_PADDING",
                                 ValueRange{peelUnrealized(adaptor.getSrc())},
                                 ArrayAttr{}, templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTImg2colToEmitC : public OpConversionPattern<pto::TImg2colOp> {
  using OpConversionPattern<pto::TImg2colOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TImg2colOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Type u16Ty = emitc::OpaqueType::get(ctx, "uint16_t");
    // posM/posK are runtime offsets. A constant operand folds to the same
    // uint16_t constant the old attribute form emitted; absent means 0.
    // Either way a non-constant operand passes through as a cast so a loop
    // induction variable can drive the in-loop Split-K pattern.
    auto materializePos = [&rewriter, &loc, &u16Ty, &ctx](Value operand) -> Value {
      if (!operand) {
        return makeEmitCIntConstant(rewriter, loc, u16Ty, 0);
      }
      operand = peelUnrealized(operand);
      return rewriter
          .create<emitc::CallOpaqueOp>(
              loc, u16Ty, "static_cast", ArrayAttr{},
              rewriter.getArrayAttr(
                  {emitc::OpaqueAttr::get(ctx, "uint16_t")}),
              ValueRange{operand})
          .getResult(0);
    };
    Value posM = materializePos(adaptor.getPosM());
    Value posK = materializePos(adaptor.getPosK());
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCTileTypeString(cast<pto::TileBufType>(op.getDst().getType())),
         *getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TIMG2COL",
                                 ValueRange{dst, src, posM, posK}, ArrayAttr{},
                                 templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateTileTileImg2colPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOSetFmatrixToEmitC, PTOSetImg2colRptToEmitC,
               PTOSetImg2colPaddingToEmitC, PTOTImg2colToEmitC>(
      typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
