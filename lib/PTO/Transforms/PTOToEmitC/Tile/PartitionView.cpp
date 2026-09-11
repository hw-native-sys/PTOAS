// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- PartitionView.cpp - Tile PartitionView op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOPartitionViewToEmitC
    : public OpConversionPattern<mlir::pto::PartitionViewOp> {
  using OpConversionPattern<mlir::pto::PartitionViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::PartitionViewOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
    Type sourceElementType;
    int64_t sourceRank = 0;
    if (auto sourceType =
            dyn_cast<pto::TensorViewType>(op.getSource().getType())) {
      sourceElementType = sourceType.getElementType();
      sourceRank = sourceType.getRank();
    } else if (auto sourceType = dyn_cast<pto::PartitionTensorViewType>(
                   op.getSource().getType())) {
      sourceElementType = sourceType.getElementType();
      sourceRank = sourceType.getRank();
    }
    if (!sourceElementType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "expected tensor_view or partition_tensor_view source and "
              "partition_tensor_view result");
    }

    Value source = peelGlobalTensorConversionBridge(adaptor.getSource());

    auto sourceStrides =
        gatherRuntimeSourceStrides(op, rewriter, source, sourceRank);
    if (failed(sourceStrides))
      return failure();

    return emitRuntimePartitionGlobalTensor(
        op, adaptor, rewriter, resultType, source, sourceElementType,
        *sourceStrides, sourceRank);
  }

  // Collect the per-dimension source strides: static strides come from the
  // make_tensor_view template token, dynamic ones from runtime metadata.
  FailureOr<SmallVector<Value, 5>>
  gatherRuntimeSourceStrides(mlir::pto::PartitionViewOp op,
                             ConversionPatternRewriter &rewriter, Value source,
                             int64_t sourceRank) const {
    SmallVector<Value, 5> sourceStrides;
    sourceStrides.reserve(sourceRank);
sourceStrides.reserve(sourceRank);
if (auto makeView = op.getSource().getDefiningOp<pto::MakeTensorViewOp>()) {
  if (makeView.getStrides().size() !=
      static_cast<size_t>(sourceRank)) {
    return rewriter.notifyMatchFailure(op, "source stride rank mismatch");
  }
  for (Value stride : makeView.getStrides()) {
    Value mapped = rewriter.getRemappedValue(stride);
    if (!mapped) {
      return rewriter.notifyMatchFailure(op, "source stride is not remapped");
    }
    sourceStrides.push_back(castViewIndexToEmitC(rewriter, op.getLoc(),
                                                 mapped));
  }
} else {
  for (int64_t dim = 0; dim < sourceRank; ++dim) {
    Value logicalDim = makeViewIndexConstant(rewriter, op.getLoc(), dim);
    sourceStrides.push_back(getRuntimeGlobalTensorMetadata(
        rewriter, op.getLoc(), source, logicalDim, sourceRank,
        /*isStride=*/true));
  }
}
    return sourceStrides;
  }

  // Fold offsets*strides into a linear byte offset and wrap the data pointer
  // into a runtime-shaped GlobalTensor descriptor.
  LogicalResult emitRuntimePartitionGlobalTensor(
      mlir::pto::PartitionViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      pto::PartitionTensorViewType resultType, Value source,
      Type sourceElementType, const SmallVector<Value, 5> &sourceStrides,
      int64_t sourceRank) const {

Value linearOffset = makeViewIndexConstant(rewriter, op.getLoc(), 0);
for (auto [offset, stride] :
     llvm::zip(adaptor.getOffsets(), sourceStrides)) {
  Value term = rewriter
                   .create<emitc::MulOp>(
                       op.getLoc(), linearOffset.getType(),
                       castViewIndexToEmitC(rewriter, op.getLoc(), offset),
                       stride)
                   .getResult();
  linearOffset = rewriter
                     .create<emitc::AddOp>(op.getLoc(),
                                           linearOffset.getType(),
                                           linearOffset, term)
                     .getResult();
}

std::string elemTypeStr = getElemTypeStringForGT(sourceElementType);
auto ptrType = emitc::PointerType::get(emitc::OpaqueType::get(
    rewriter.getContext(), "__gm__ " + elemTypeStr));
Value data = rewriter
                 .create<emitc::CallOpaqueOp>(
                     op.getLoc(), ptrType, "PTOAS__GLOBAL_TENSOR_DATA",
                     ArrayAttr{}, ArrayAttr{}, ValueRange{source})
                 .getResult(0);
Value ptr = rewriter

                .create<emitc::AddOp>(op.getLoc(), ptrType, data,
                                      linearOffset)
                .getResult();

auto layout = resolveLayoutForGlobalTensor(op.getOperation(), op.getSource());
std::string layoutString =
    layout ? layoutToEmitCString(*layout) : "pto::Layout::ND";
auto result = buildRuntimeGlobalTensor(
    rewriter, op.getLoc(), ptr, resultType.getElementType(),
    resultType.getShape(), adaptor.getSizes(), sourceStrides, layoutString);
if (failed(result))
  return rewriter.notifyMatchFailure(
      op, "failed to build partition GlobalTensor descriptor");
rewriter.replaceOp(op, *result);
return success();
  }
};

struct PTOPartitionViewStaticToEmitC
    : public OpConversionPattern<mlir::pto::PartitionViewOp> {
  using OpConversionPattern<
      mlir::pto::PartitionViewOp>::OpConversionPattern;

  // Verified source shape/strides and split static/dynamic offset terms.
  struct StaticPartitionInputs {
    SmallVector<int64_t> srcStrides;
    int64_t staticLinearOffset = 0;
    SmallVector<std::pair<Value, int64_t>> dynamicOffsetTerms;
  };

  LogicalResult matchAndRewrite(mlir::pto::PartitionViewOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resTy = dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
    Type srcElemTy;
    int64_t srcRank = 0;
    if (auto srcTy = dyn_cast<pto::TensorViewType>(op.getSource().getType())) {
      srcElemTy = srcTy.getElementType();
      srcRank = srcTy.getRank();
    } else if (auto srcTy =
                   dyn_cast<pto::PartitionTensorViewType>(
                       op.getSource().getType())) {
      srcElemTy = srcTy.getElementType();
      srcRank = srcTy.getRank();
    }
    if (!srcElemTy || !resTy) {
      return rewriter.notifyMatchFailure(
          op, "expected tensor_view or partition_tensor_view source and "
              "partition_tensor_view result");
    }

    if (op.getOffsets().size() != static_cast<size_t>(srcRank) ||
        op.getSizes().size() != static_cast<size_t>(srcRank)) {
      return rewriter.notifyMatchFailure(op, "rank mismatch");
    }

    if (!partitionViewHasStaticResultShape(op)) {
      return rewriter.notifyMatchFailure(
          op, "globaltensor partition_view requires static result shape");
    }

    auto inputs = resolveStaticPartitionInputs(op, adaptor, rewriter, srcRank);
    if (failed(inputs))
      return failure();

    return emitStaticPartitionGlobalTensor(op, adaptor, rewriter, resTy,
                                           srcElemTy, *inputs);
  }

  // Verify the source strides are fully static and split the offsets into a
  // static linear contribution plus dynamic (value, stride) terms.
  FailureOr<StaticPartitionInputs>
  resolveStaticPartitionInputs(mlir::pto::PartitionViewOp op,
                               OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter,
                               int64_t srcRank) const {
    StaticPartitionInputs inputs;
    if (failed(getStaticTensorViewStrides(op.getSource(), adaptor.getSource(),
                                          srcRank, inputs.srcStrides))) {
      return rewriter.notifyMatchFailure(
          op, "cannot resolve exact partition source strides; refusing to "
              "assume a compact layout");
    }
    for (auto [idx, values] :
         llvm::enumerate(llvm::zip(op.getOffsets(), adaptor.getOffsets()))) {
      Value originalOffset = std::get<0>(values);
      Value convertedOffset = std::get<1>(values);
      int64_t stride = inputs.srcStrides[idx];
      if (stride == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(
            op, "dynamic source stride is not supported");
      }

      if (auto cst = getStaticIndexLikeValue(originalOffset)) {
        if (*cst != 0) {
          inputs.staticLinearOffset += (*cst) * stride;
        }
        continue;
      }
      inputs.dynamicOffsetTerms.push_back({convertedOffset, stride});
    }
    return inputs;
  }

  // Materialize the offset-applied GM pointer and wrap it into a static-shape
  // GlobalTensor view.
  // Apply the static + dynamic offset terms to the base data pointer,
  // falling back to a static byte offset when no dynamic terms exist.
  Value applyPartitionOffsetTerms(
      ConversionPatternRewriter &rewriter, Location loc, Value data,
      const StaticPartitionInputs &inputs, Type indexTy,
      const std::function<Value(int64_t)> &mkIndex) const {
    auto asIndex = [&](Value value) -> Value {
      if (value.getType() == indexTy)
        return value;
      return rewriter.create<emitc::CastOp>(loc, indexTy, value).getResult();
    };

    if (inputs.dynamicOffsetTerms.empty())
      return applyStaticMemrefOffset(rewriter, loc, data,
                                     inputs.staticLinearOffset);

    Value totalOffset = mkIndex(inputs.staticLinearOffset);
    for (auto [offsetValue, stride] : inputs.dynamicOffsetTerms) {
      Value term = asIndex(offsetValue);
      if (stride != 1) {
        Value strideValue = mkIndex(stride);
        term = rewriter.create<emitc::MulOp>(loc, indexTy, term, strideValue)
                      .getResult();
      }
      totalOffset = rewriter
                        .create<emitc::AddOp>(loc, indexTy, totalOffset, term)
                        .getResult();
    }
    return rewriter
        .create<emitc::AddOp>(loc, data.getType(), data, totalOffset)
        .getResult();
  }

  LogicalResult emitStaticPartitionGlobalTensor(
      mlir::pto::PartitionViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, pto::PartitionTensorViewType resTy,
      Type srcElemTy, const StaticPartitionInputs &inputs) const {
    auto *ctx = rewriter.getContext();
    std::string elemTypeStr = getElemTypeStringForGT(srcElemTy);
    auto ptrTy = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
    Value src = peelUnrealized(adaptor.getSource());
    Value data = materializeGlobalTensorDataPointer(
        rewriter, op.getLoc(), src, op.getSource().getType());
    if (data.getType() != ptrTy) {
      data = rewriter.create<emitc::CastOp>(op.getLoc(), ptrTy, data)
                 .getResult();
    }
    Type indexTy = emitc::OpaqueType::get(ctx, "int64_t");
    auto mkIndex = [&](int64_t value) {
      return makeEmitCIntConstant(rewriter, op.getLoc(), indexTy, value);
    };
    Value ptr = applyPartitionOffsetTerms(rewriter, op.getLoc(), data,
                                           inputs, indexTy, mkIndex);
    auto resultOr = buildGlobalTensorViewFromPointer(
        rewriter, op.getLoc(), ptr, resTy.getElementType(), resTy.getShape(),
        inputs.srcStrides,
        getSpecialGlobalTensorTypeSpecForLayout(
            resolveLayoutForGlobalTensor(op.getOperation(), op.getSource()),
            resTy.getShape(), resTy.getElementType()),
        resolveLayoutForGlobalTensor(op.getOperation(), op.getSource())
                ? layoutToEmitCString(
                      *resolveLayoutForGlobalTensor(op.getOperation(),
                                                    op.getSource()))
                : "pto::Layout::ND");
    if (failed(resultOr))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize partition GlobalTensor");

    rewriter.replaceOp(op, *resultOr);
    return success();
  }
};

void populateTilePartitionViewPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOPartitionViewToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartitionViewStaticToEmitC>(typeConverter, ctx,
                                              PatternBenefit(2));
}

} // namespace pto
} // namespace mlir
