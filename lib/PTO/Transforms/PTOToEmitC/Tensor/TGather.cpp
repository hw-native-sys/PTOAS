// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TGather.cpp - Tensor TGather op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOGatherToEmitC : public OpConversionPattern<pto::TGatherOp> {
  using OpConversionPattern<pto::TGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Case 1: index-based TGATHER(dst, src0, indices[, tmp])
    if (Value idx = adaptor.getIndices())
      return emitIndexedGather(op, adaptor, rewriter, idx);

    // Case 2: compare-based TGATHER<DstT, SrcT, TmpT, CDstT, CmpMode::GT>(
    //            dst, src0, kValue, tmp, cdst, offset)
    if (Value cdst = adaptor.getCdst())
      return emitCompareGather(op, adaptor, rewriter, cdst);

    // Case 3: mask-pattern TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src0)
    if (!op.getMaskPatternAttr())
      return rewriter.notifyMatchFailure(
          op, "expected maskPattern, indices, or cdst on tgather");
    return emitMaskPatternGather(op, adaptor, rewriter);
  }

  // Mask-pattern gather: TGATHER<DstT, SrcT, MaskPattern::XXXX>(dst, src0).
  LogicalResult emitMaskPatternGather(pto::TGatherOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc();

    auto getOpaqueTok = [&](Value v,
                            StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(
          op, (name + " must be emitc::OpaqueType (tile)").str());
    };
    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    if (failed(dstTokOr) || failed(srcTokOr))
      return failure();

    // mp is an EnumAttr; stringify name is "P0101" etc.
    // We emit MaskPattern::P0101 (because generated C++ has `using namespace pto;`)
    auto mp = op.getMaskPatternAttr();
    std::string mpTok = std::string("MaskPattern::") +
                        mlir::pto::stringifyMaskPattern(mp.getValue()).str();

    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, mpTok),
    });

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0});

    rewriter.eraseOp(op);
    return success();
  }

  // TGATHER(dst, src0, indices[, tmp]) without template arguments.
  LogicalResult emitIndexedGather(pto::TGatherOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  Value idx) const {
    auto loc = op.getLoc();
    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc();
    idx = peelUnrealized(idx);
    SmallVector<Value, 4> operands{dst, src0, idx};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }

  // Compare-based gather: spell out the operand types plus the compare mode.
  LogicalResult emitCompareGather(pto::TGatherOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  Value cdst) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc();
    cdst = peelUnrealized(cdst);
    Value tmp = adaptor.getTmp();
    Value kValue = adaptor.getKValue();

    auto getOpaqueTok = [&](Value v,
                            StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(
          op, (name + " must be emitc::OpaqueType (tile)").str());
    };

    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    auto cdstTokOr = getOpaqueTok(cdst, "cdst");
    auto tmpTokOr = getOpaqueTok(tmp, "tmp");
    if (failed(dstTokOr) || failed(srcTokOr) || failed(cdstTokOr) ||
        failed(tmpTokOr))
      return failure();

    auto cmpAttr = op.getCmpModeAttr();
    std::string cmpTok = cmpAttr ? cmpModeTok(cmpAttr) : "CmpMode::EQ";
    int64_t offset = 0;
    if (auto offsetAttr = op.getOffsetAttr())
      offset = getIntegerAttrSignedValue(offsetAttr);
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value offsetVal = makeEmitCIntConstant(rewriter, loc, i32Ty, offset);

    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, *tmpTokOr),
        emitc::OpaqueAttr::get(ctx, *cdstTokOr),
        emitc::OpaqueAttr::get(ctx, cmpTok),
    });

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0, kValue, tmp, cdst, offsetVal});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGatherbToEmitC : public OpConversionPattern<pto::TGatherBOp> {
  using OpConversionPattern<pto::TGatherBOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherBOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src     = adaptor.getSrc();
    Value offsets = adaptor.getOffsets();
    Value dst     = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHERB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, offsets});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTGatherPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOGatherToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherbToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
