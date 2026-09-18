// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ScalarPtr.cpp - ScalarMisc ScalarPtr op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOLoadToEmitC : public OpConversionPattern<pto::PTOLoadOp> {
  using OpConversionPattern<pto::PTOLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PTOLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    Type dstTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!dstTy)
      return failure();

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{dstTy}, "PTOAS__PTR_LOAD",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOStoreToEmitC : public OpConversionPattern<pto::PTOStoreOp> {
  using OpConversionPattern<pto::PTOStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PTOStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    Value val = adaptor.getValue();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__PTR_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset, val});
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__SCALAR_GM_STORE_FLUSH",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateScalarMiscScalarPtrPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOLoadToEmitC>(typeConverter, ctx);
  patterns.add<PTOStoreToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
