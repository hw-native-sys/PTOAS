// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TAdd.cpp - Tensor TAdd op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TensorInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOTAddCToTADDC : public OpConversionPattern<pto::TAddCOp> {
  using OpConversionPattern<pto::TAddCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TADDC yet.
    // Decompose: dst = src0 + src1 + src2
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TADD",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getSrc1(), adaptor.getSrc2());
    return success();
  }
};

struct PTOAddSToTADDS : public OpConversionPattern<pto::TAddSOp> {
  using OpConversionPattern<pto::TAddSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = adaptor.getSrc();
    Value dst    = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TADDS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOAddSCToTADDSC : public OpConversionPattern<pto::TAddSCOp> {
  using OpConversionPattern<pto::TAddSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TADDSC yet.
    // Decompose: dst = src0 + scalar + src1
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TADDS",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getScalar(), adaptor.getSrc1());
    return success();
  }
};

struct PTOTAddToTADD : public OpConversionPattern<pto::TAddOp> {
  using OpConversionPattern<pto::TAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TADD", ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateTensorTAddPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOAddSToTADDS>(typeConverter, ctx);
  patterns.add<PTOTAddCToTADDC>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
  patterns.add<PTOTAddToTADD>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
