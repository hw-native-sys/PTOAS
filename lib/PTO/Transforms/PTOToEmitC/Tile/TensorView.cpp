// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TensorView.cpp - Tile TensorView op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOMakeTensorViewToEmitC
    : public OpConversionPattern<mlir::pto::MakeTensorViewOp> {
  using OpConversionPattern<mlir::pto::MakeTensorViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::MakeTensorViewOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto resultType = dyn_cast<pto::TensorViewType>(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected tensor_view result");
    std::string layout = "pto::Layout::ND";
    if (auto attr = op.getLayoutAttr()) {
      layout = layoutToEmitCString(attr.getLayout());
    } else if (auto attr = resultType.getLayoutAttr()) {
      layout = layoutToEmitCString(attr.getLayout());
    }
    auto result = buildRuntimeGlobalTensor(
        rewriter, op.getLoc(), adaptor.getPtr(),
        resultType.getElementType(), resultType.getShape(), adaptor.getShape(),
        adaptor.getStrides(), layout);
    if (failed(result))
      return rewriter.notifyMatchFailure(
          op, "failed to build runtime GlobalTensor descriptor");
    rewriter.replaceOp(op, *result);
    return success();
  }
};

template <typename OpTy, bool IsStride>
struct PTOGetTensorViewMetadataToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type sourceType = op.getTensorView().getType();
    int64_t rank = 0;
    if (auto type = dyn_cast<pto::TensorViewType>(sourceType)) {
      rank = type.getRank();
    } else if (auto type = dyn_cast<pto::PartitionTensorViewType>(sourceType)) {
      rank = type.getRank();
    } else {
      return rewriter.notifyMatchFailure(op, "expected PTO tensor view");
    }

    Value result = getRuntimeGlobalTensorMetadata(
        rewriter, op.getLoc(),
        peelGlobalTensorConversionBridge(adaptor.getTensorView()),
        adaptor.getDimIndex(), rank, IsStride);
    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateTileTensorViewPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOMakeTensorViewToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetTensorViewMetadataToEmitC<pto::GetTensorViewDimOp, false>>(typeConverter, ctx);
  patterns.add<PTOGetTensorViewMetadataToEmitC<pto::GetTensorViewStrideOp,
                                                true>>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
