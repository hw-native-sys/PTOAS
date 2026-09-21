// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- RuntimeQuery.cpp - ScalarMisc runtime-query lowering ----------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOGetL2CacheOffsetToEmitC
    : public OpConversionPattern<pto::GetL2CacheOffsetOp> {
  using OpConversionPattern<pto::GetL2CacheOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::GetL2CacheOffsetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_l2_cache_offset", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

void populateScalarMiscRuntimeQueryPatterns(RewritePatternSet &patterns,
                                            TypeConverter &typeConverter,
                                            MLIRContext *ctx) {
  patterns.add<PTOGetL2CacheOffsetToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
