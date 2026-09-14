// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TSort.cpp - Tensor TSort op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOMrgSortToEmitC : public OpConversionPattern<pto::TMrgSortOp> {
using OpConversionPattern<pto::TMrgSortOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMrgSortOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  if (op.isFormat1())
    return emitFormat1(op, adaptor, rewriter);
  if (op.isFormat2())
    return emitFormat2(op, adaptor, rewriter);
  return op.emitOpError("unsupported mrgsort_dps format");
}

  // TMRGSORT(dst, src, blockLen).
  LogicalResult emitFormat1(pto::TMrgSortOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = adaptor.getSrcs().front();
    Value dst = adaptor.getDsts().front();
    Value blockLen = adaptor.getBlockLen();

    SmallVector<Value, 3> operands{dst, src, blockLen};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMRGSORT",
        ArrayAttr{}, ArrayAttr{}, operands);
    rewriter.eraseOp(op);
    return success();
  }

  // TMRGSORT<DstTile, TmpTile, Src0..SrcN, exhausted>(
  //     dst, executedNumList, tmp, src0..srcN).
  LogicalResult emitFormat2(pto::TMrgSortOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = adaptor.getDsts()[0];
    Value tmp = adaptor.getTmp();
    Value excuted = adaptor.getExcuted();

    SmallVector<Value, 4> srcs;
    srcs.reserve(adaptor.getSrcs().size());
    for (Value v : adaptor.getSrcs())
      srcs.push_back(v);

    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto tmpOT = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
    if (!dstOT || !tmpOT || srcs.size() < 2 || srcs.size() > 4)
      return op.emitOpError("format2 expects dst/tmp tilebufs and 2 to 4 srcs");

    SmallVector<Attribute, 8> targs;
    targs.reserve(2 + srcs.size() + 1);
    targs.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
    targs.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
    for (Value v : srcs) {
      auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return op.emitOpError("format2 expects tilebuf srcs");
      targs.push_back(emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
    }
    targs.push_back(emitc::OpaqueAttr::get(ctx, op.getExhausted() ? "true" : "false"));
    ArrayAttr templateArgs = rewriter.getArrayAttr(targs);

    SmallVector<Value, 7> operands{dst, excuted, tmp};
    operands.append(srcs.begin(), srcs.end());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMRGSORT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

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

void populateTensorTSortPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<PTOSORT32SToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
