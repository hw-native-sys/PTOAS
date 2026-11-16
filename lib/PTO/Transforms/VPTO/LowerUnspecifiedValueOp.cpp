// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

class LowerUnspecifiedValueOp final
    : public OpConversionPattern<pto::PTOUnspecifiedValueOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        pto::PTOUnspecifiedValueOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        (void)adaptor;
        Type resultType = getTypeConverter()->convertType(op.getResult().getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                op, "failed to convert unspecified value type");
        }
        // Keep the value arbitrary at the LLVM boundary. Bisheng 15 fails to
        // select the zero-vector BUILD_VECTOR that freeze(poison) may produce for
        // v64f32. LLVM undef is valid here because the PTO op promises no initial
        // contents, and the caller must define lanes before observing them.
        Value unspecified = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultType);
        rewriter.replaceOp(op, unspecified);
        return success();
    }
};

void populateLowerUnspecifiedValueOpPattern(
    const TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<LowerUnspecifiedValueOp>(typeConverter, patterns.getContext());
}

} // namespace mlir::pto::detail
