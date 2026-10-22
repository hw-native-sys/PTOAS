// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithCast.cpp - Arith ArithCast op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ArithExtUIToEmitC : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");
    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();
    return emitWidenedInt(op.getOperation(), adaptor.getIn(), srcIntTy,
                          dstIntTy, dstTy, /*isSigned=*/false, rewriter);
  }
};

struct ArithExtSIToEmitC : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern<arith::ExtSIOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");
    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();
    return emitWidenedInt(op.getOperation(), adaptor.getIn(), srcIntTy,
                          dstIntTy, dstTy, /*isSigned=*/true, rewriter);
  }
};

template <typename CastOp>
struct ArithCastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  // NOLINTNEXTLINE: ConversionPattern requires a mutable rewriter to replace
  // the matched operation.
  LogicalResult matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithIndexCastUIToEmitC : public OpConversionPattern<arith::IndexCastUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // MemRef casts are handled elsewhere; for safety, fall back to emitc.cast.
    if (isa<MemRefType>(op.getIn().getType()) || isa<MemRefType>(op.getType())) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto getBW = [](Type t) -> std::optional<unsigned> {
      if (auto i = dyn_cast<IntegerType>(t))
        return i.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };

    auto srcBW = getBW(op.getIn().getType());
    auto dstBW = getBW(op.getType());
    if (!srcBW || !dstBW)
      return rewriter.notifyMatchFailure(op, "unsupported index_castui types");

    if (*dstBW <= *srcBW) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    // bool already carries the required 0/1 value. Avoid unnecessary nested
    // unsigned casts, which can lose comparison parentheses in LLVM 19.
    if (*srcBW == 1) {
      rewriter.replaceOp(op, emitI1ToInteger(rewriter, loc, adaptor.getIn(),
                                           dstTy, /*isSigned=*/false));
      return success();
    }

    auto uSrcTy = getUnsignedIntOpaqueType(rewriter.getContext(), *srcBW);
    auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), *dstBW);
    Value srcU = emitCCast(rewriter, loc, uSrcTy, adaptor.getIn());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename CastOp, bool isSigned>
struct ArithIntToFPToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!srcIntTy) {
      return rewriter.notifyMatchFailure(op, "expected scalar integer input");
    }

    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy) {
      return failure();
    }

    Value input = adaptor.getIn();
    if (srcIntTy.getWidth() == 1) {
      // A3 CCEC rejects bool/unsigned-to-float casts. Both i1 interpretations
      // fit in int32_t, preserving 0/-1 (signed) or 0/1 (unsigned).
      Type carrierTy = getSignedIntOpaqueType(rewriter.getContext(), kInt32BitWidth);
      input = emitI1ToInteger(rewriter, loc, input, carrierTy, isSigned);
    } else if (!isSigned) {
      input = castSignlessIntToUnsignedSameWidth(rewriter, loc, input,
                                               srcIntTy.getWidth());
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, input);
    return success();
  }
};

struct ArithFPToUIToEmitC : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    if (!dstIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer result");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value asU = rewriter.create<emitc::CastOp>(loc, uDstTy, adaptor.getIn()).getResult();
    Value result = emitCCast(rewriter, loc, dstTy, asU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithBitcastToEmitC : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // For pointer-like types, a regular cast is fine.
    if (isa<emitc::PointerType>(dstTy)) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    // Only support scalar int/float/index bitcasts here.
    auto srcTy = op.getIn().getType();
    auto dstOrigTy = op.getType();

    auto getBitWidth = [](Type t) -> std::optional<unsigned> {
      if (auto it = dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto ft = dyn_cast<FloatType>(t))
        return ft.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };
    auto srcBW = getBitWidth(srcTy);
    auto dstBW = getBitWidth(dstOrigTy);
    if (!srcBW || !dstBW || *srcBW != *dstBW)
      return rewriter.notifyMatchFailure(op, "bitcast requires equal bitwidth");

    // Determine the template argument from the destination type string.
    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return rewriter.notifyMatchFailure(op, "expected emitc opaque dest type");

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{dstTy}, "ptoas_bitcast", /*operands=*/ValueRange{adaptor.getIn()},
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct ArithCastOPToEmitC : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    if (op.getIn().getType().isInteger(1)) {
      rewriter.replaceOp(op, emitI1ToInteger(rewriter, op.getLoc(),
                                           adaptor.getIn(), newTy,
                                           /*isSigned=*/true));
      return success();
    }
    if (adaptor.getIn().getType() == newTy) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, newTy, adaptor.getIn());
    return success();
  }
};

struct ArithTruncIToEmitC : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // to-i1 conversions: Arith wants truncation to the low bit, while C/C++
    // casts to bool are equivalent to `v != 0`. Implement as `(bool)(v & 1)`.
    if (dstIntTy.getWidth() == 1) {
      if (srcIntTy.getWidth() == 1) {
        rewriter.replaceOp(op, adaptor.getIn());
        return success();
      }

      auto uSrcTy =
          getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
      Value inU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                                     srcIntTy.getWidth());
      Value one = makeEmitCIntConstant(rewriter, loc, uSrcTy, 1);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, uSrcTy, inU, one);
      Value asBool = emitCCast(rewriter, loc, dstTy, masked);
      rewriter.replaceOp(op, asBool);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

void populateArithArithCastPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithExtUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<ArithIntToFPToEmitC<arith::UIToFPOp, false>>(typeConverter, ctx);
  patterns.add<ArithIntToFPToEmitC<arith::SIToFPOp, true>>(typeConverter, ctx);
  patterns.add<ArithFPToUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<ArithIndexCastUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithBitcastToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<ArithTruncIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
