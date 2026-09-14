// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ReduceMisc.cpp - Reduce ReduceMisc op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ReduceInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

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

void populateReduceReduceMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOSubSSToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
