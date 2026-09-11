// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TColReduce.cpp - Tensor TColReduce op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOColMaxToEmitC : public OpConversionPattern<pto::TColMaxOp> {
  using OpConversionPattern<pto::TColMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // intrinsic: TCOLMAX(dst, src)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TCOLMAX", ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMaxToEmitC : public OpConversionPattern<pto::TColArgMaxOp> {
  using OpConversionPattern<pto::TColArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColMinToEmitC : public OpConversionPattern<pto::TColMinOp> {
  using OpConversionPattern<pto::TColMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // intrinsic: TCOLMIN(dst, src)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TCOLMIN", ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMinToEmitC : public OpConversionPattern<pto::TColArgMinOp> {
  using OpConversionPattern<pto::TColArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColSumToEmitC : public OpConversionPattern<pto::TColSumOp> {
  using OpConversionPattern<pto::TColSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // Check if tmp exists before accessing it
    if (op.getTmp()) {
      // Format 2: with tmp and isBinary
      Value tmp = adaptor.getTmp();
      bool isBinary = false;
      if (auto a = op.getIsBinaryAttr())
        isBinary = a.getValue();

      auto boolTy = emitc::OpaqueType::get(ctx, "bool");
      auto tok = isBinary ? "true" : "false";
      Value isBinaryVal = rewriter.create<emitc::ConstantOp>(
          loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

      SmallVector<unsigned, 3> tileSlotOrder;
      tileSlotOrder.push_back(op.getDstMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getSrcMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getTmpMutable().begin()->getOperandNumber());

      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TCOLSUM",
          ValueRange{dst, src, tmp, isBinaryVal}, ArrayAttr{}, ArrayAttr{},
          tileSlotOrder);
    } else {
      // Format 1: without tmp and isBinary
      SmallVector<unsigned, 2> tileSlotOrder;
      tileSlotOrder.push_back(op.getDstMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getSrcMutable().getOperandNumber());
      createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                   "TCOLSUM", ValueRange{dst, src},
                                   ArrayAttr{}, ArrayAttr{}, tileSlotOrder);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColProdToEmitC : public OpConversionPattern<pto::TColProdOp> {
  using OpConversionPattern<pto::TColProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLPROD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTColReducePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColProdToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMinToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
