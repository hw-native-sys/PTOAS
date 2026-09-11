// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTile.cpp - tile/view/pipe lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

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


struct PTOTAllocToEmitC : public OpConversionPattern<mlir::pto::TAllocOp> {
  PTOTAllocToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                   PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TAllocOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    Value entry = peelGlobalTensorConversionBridge(adaptor.getEntry());
    auto entryTok = getPipeDataTypeToken(entry);
    if (failed(entryTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve entry token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TALLOC<" + *pipeTok + ", " + *entryTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{adaptor.getPipeHandle(), entry});
    return success();
  }

  PTOArch targetArch;
};

struct PTOSetQuantScalarToEmitC
    : public OpConversionPattern<mlir::pto::SetQuantScalarOp> {
  PTOSetQuantScalarToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                           PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SetQuantScalarOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::SetQuantScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outTypeAttr =
        op->getAttrOfType<StringAttr>(kEmitCScalarOutTypeAttrName);
    if (!outTypeAttr)
      return rewriter.notifyMatchFailure(
          op, "expected rematerialized fixpipe set_quant_scalar to carry emitc out type");

    std::string outTok = outTypeAttr.getValue().str();
    Value scale = adaptor.getScale();
    auto floatTy = emitc::OpaqueType::get(rewriter.getContext(), "float");
    if (scale.getType() != floatTy)
      scale = rewriter.create<emitc::CastOp>(op.getLoc(), floatTy, scale).getResult();

    ArrayAttr targs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(rewriter.getContext(), outTok)});
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "SET_QUANT_SCALAR", ArrayAttr{}, targs,
        ValueRange{scale});
    return success();
  }

  PTOArch targetArch;
};

struct PTOSetQuantVectorToEmitC
    : public OpConversionPattern<mlir::pto::SetQuantVectorOp> {
  PTOSetQuantVectorToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                           PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SetQuantVectorOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::SetQuantVectorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "SET_QUANT_VECTOR", ArrayAttr{}, ArrayAttr{},
        ValueRange{adaptor.getScalingTile()});
    return success();
  }

  PTOArch targetArch;
};

// Resolve the third TPUSH/TPOP template token: the fixpipe config alias
// when the pipe carries an acc-push epilogue, otherwise the split mode.
template <typename OpTy>
static FailureOr<std::string>
resolvePipeTileConfigToken(OpTy op, PTOArch targetArch) {
  (void)targetArch;
  if constexpr (std::is_same_v<OpTy, mlir::pto::TPushOp>) {
    if (auto accPushEpilogue =
            getPipeInitAccPushEpilogue(getPipeInitDef(op.getPipeHandle()))) {
      auto pipeId = getFrontendPipeIdFromHandle(op.getPipeHandle());
      if (pipeId)
        return buildFixpipeConfigAliasName(*pipeId);
      auto configTokOr = buildFixpipeConfigTypeToken(accPushEpilogue);
      if (failed(configTokOr))
        return failure();
      return *configTokOr;
    }
  }
  return getTileSplitToken(op.getSplit());
}

// tpush/tpop lowering: resolve the TPipe/tile/split template tokens and emit
// T(PUSH|POP)<pipe, tile, split>(pipeHandle, tile[, aivSubblockId]).
template <typename OpTy>
struct PTOPipeTileOpToEmitC : public OpConversionPattern<OpTy> {
  PTOPipeTileOpToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                       PTOArch targetArch, StringRef calleePrefix)
      : OpConversionPattern<OpTy>(typeConverter, ctx),
        targetArch(targetArch), calleePrefix(calleePrefix.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    Value convertedTile = peelGlobalTensorConversionBridge(adaptor.getTile());
    auto tileTok = getPipeDataTypeToken(convertedTile);
    if (failed(tileTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve tile token");
    auto configTok = resolvePipeTileConfigToken(op, targetArch);
    if (failed(configTok))
      return rewriter.notifyMatchFailure(op,
                                         "failed to resolve config/split token");
    std::string callee =
        calleePrefix + "<" + *pipeTok + ", " + *tileTok + ", " + *configTok +
        ">";
    SmallVector<Value> callOperands{adaptor.getPipeHandle(), convertedTile};
    if (Value aivSubblockId = adaptor.getAivSubblockid()) {
      Value aivSubblockIdI32 = rewriter.create<emitc::CastOp>(
          op.getLoc(), rewriter.getI32Type(), peelUnrealized(aivSubblockId));
      callOperands.push_back(aivSubblockIdI32);
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{}, callOperands);
    return success();
  }

  PTOArch targetArch;
  std::string calleePrefix;
};

using PTOTPushToEmitC = PTOPipeTileOpToEmitC<mlir::pto::TPushOp>;
using PTOTPopToEmitC = PTOPipeTileOpToEmitC<mlir::pto::TPopOp>;

struct PTOTFreeToEmitC : public OpConversionPattern<mlir::pto::TFreeOp> {
  PTOTFreeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TFreeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    SmallVector<Value> operands{adaptor.getPipeHandle()};
    std::string callee;
    if (op.getEntry()) {
      Value entry = peelGlobalTensorConversionBridge(adaptor.getEntry());
      auto entryTok = getPipeDataTypeToken(entry);
      if (failed(entryTok))
        return rewriter.notifyMatchFailure(op, "failed to resolve entry token");
      callee = "TFREE<" + *pipeTok + ", " + *entryTok + ", " + *splitTok + ">";
      operands.push_back(entry);
    } else {
      callee = "TFREE<" + *pipeTok + ", " + *splitTok + ">";
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{}, operands);
    return success();
  }

  PTOArch targetArch;
};

//===----------------------------------------------------------------------===//
// populate patterns

// Tile role token for a non-GM reinterpret_cast target address space.
// Scaling tiles infer their role from the source value when possible.
static const char *reinterpretCastTileRole(pto::AddressSpace as,
                                           Value source) {
  if (as == pto::AddressSpace::SCALING)
    if (const char *inferredRole = inferScalingRoleFromValue(source))
      return inferredRole;
  return tileRoleToken(pto::AddressSpaceAttr::get(source.getContext(), as));
}

// Conservative Tile<...> type string for a reinterpret_cast target: the
// result shape (fallback 32x32) with a default config.
static std::string buildReinterpretCastTileTypeString(MemRefType resMrTy,
                                                      Type elemTy,
                                                      const char *roleTok) {
  int64_t rows = 32, cols = 32;
  if (resMrTy.getRank() >= 2 && resMrTy.hasStaticShape()) {
    rows = resMrTy.getDimSize(0);
    cols = resMrTy.getDimSize(1);
  }
  int64_t templateRows =
      renderTileTemplateDim(rows, elemTy, pto::BLayout::RowMajor, 0);
  int64_t templateCols =
      renderTileTemplateDim(cols, elemTy, pto::BLayout::RowMajor, 1);
  return std::string("Tile<") + roleTok + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " + std::to_string(templateRows) +
         ", " + std::to_string(templateCols) + ", BLayout::RowMajor, " +
         std::to_string(templateRows) + ", " + std::to_string(templateCols) +
         ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>";
}

// Resolve the underlying u64 address of a non-GM reinterpret_cast source:
// tiles contribute their `.data()` pointer, plain pointers pass through.
static Value reinterpretCastBaseAddress(ConversionPatternRewriter &rewriter,
                                       Location loc, Value source,
                                       pto::AddressSpace as, StringRef elemTok,
                                       Type u64Ty) {
  Value rawPtr = source;
  if (auto ot = dyn_cast<emitc::OpaqueType>(source.getType())) {
    // Only Tiles have a `.data()` member. For plain address-space pointers
    // (e.g. `__ubuf__ float*`), use the pointer value directly.
    if (ot.getValue().starts_with("Tile<"))
      rawPtr = materializeTileDataValue(rewriter, loc, source, as, elemTok);
  }

  return coerceToU64Address(rewriter, loc, rawPtr, u64Ty);
}

struct ReinterpretCastToEmitC : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy)
      return failure();

    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace());
    const bool isGm = (!asAttr || asAttr.getAddressSpace() == pto::AddressSpace::GM);

    // GM: keep pointer arithmetic.
    if (isGm)
      return emitGmReinterpretCast(op, adaptor, rewriter);

    // UB/L1/L0 tiles: materialize a new Tile view by assigning an adjusted
    // underlying pointer (in elements).
    return emitTileReinterpretCast(op, adaptor, rewriter, resMrTy, asAttr);
  }

  // GM lowering: fold into pointer arithmetic (emitc.add) plus an optional
  // PTOAS__ADDPTR_TRACE call.
  LogicalResult emitGmReinterpretCast(memref::ReinterpretCastOp op,
                                      OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    bool emitAddPtrTrace = op->hasAttr("pto.addptr_trace");
    Value source = adaptor.getSource();
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];
    auto mixedOffsets = op.getMixedOffsets();
    std::optional<int64_t> constantOffset =
        mixedOffsets.empty() ? std::nullopt
                             : getConstantIntValue(mixedOffsets.front());
    const bool isZeroOffset = constantOffset && *constantOffset == 0;

    if (!offsetVal || (isZeroOffset && !emitAddPtrTrace)) {
      rewriter.replaceOp(op, source);
      return success();
    }

    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto addOp = rewriter.create<emitc::AddOp>(loc, resultType, source, offsetVal);
    if (emitAddPtrTrace) {
      rewriter.setInsertionPointAfter(addOp);
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "PTOAS__ADDPTR_TRACE",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{addOp.getResult(), source, offsetVal});
    }
    rewriter.replaceOp(op, addOp.getResult());
    return success();
  }

  // UB/L1/L0 tile lowering: build the tile Variable, compute the adjusted
  // base address, and bind it with TASSIGN.
  LogicalResult emitTileReinterpretCast(
      memref::ReinterpretCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, MemRefType resMrTy,
      pto::AddressSpaceAttr asAttr) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value source = adaptor.getSource();
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];
    auto mixedOffsets = op.getMixedOffsets();
    std::optional<int64_t> constantOffset =
        mixedOffsets.empty() ? std::nullopt
                             : getConstantIntValue(mixedOffsets.front());
    const bool isZeroOffset = constantOffset && *constantOffset == 0;

    pto::AddressSpace as = asAttr.getAddressSpace();

    // Element type token.
    Type elemTy = resMrTy.getElementType();
    std::string elemTok = getEmitCScalarTypeToken(elemTy);
    int64_t elemBytes = getEmitCScalarByteWidth(elemTy);

    const char *roleTok = reinterpretCastTileRole(as, source);
    std::string tileTypeStr =
        buildReinterpretCastTileTypeString(resMrTy, elemTy, roleTok);

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value tile = rewriter
                     .create<emitc::VariableOp>(loc,
                                                getEmitCVariableResultType(tileType),
                                                emitc::OpaqueAttr::get(ctx, ""))
                     .getResult();
    tile = loadEmitCVariableIfNeeded(rewriter, loc, tile);

    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    Value baseAddr = reinterpretCastBaseAddress(rewriter, loc, source, as,
                                               elemTok, u64Ty);

    Value addr = baseAddr;
    if (offsetVal && !isZeroOffset) {
      Value offU64 = offsetVal;
      if (offU64.getType() != u64Ty)
        offU64 = rewriter.create<emitc::CastOp>(loc, u64Ty, offU64).getResult();

      auto bytesAttr = emitc::OpaqueAttr::get(ctx, std::to_string(elemBytes));
      Value bytesVal = rewriter.create<emitc::ConstantOp>(loc, u64Ty, bytesAttr);
      Value byteOff = rewriter.create<emitc::MulOp>(loc, u64Ty, offU64, bytesVal);
      addr = rewriter.create<emitc::AddOp>(loc, u64Ty, baseAddr, byteOff);
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{tile, addr});

    rewriter.replaceOp(op, tile);
    return success();
  }
};

struct MemRefCastToEmitC : public OpConversionPattern<memref::CastOp> {
  using OpConversionPattern<memref::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.taddc lowering -> TADDC(dst, src0, src1, src2)
//===----------------------------------------------------------------------===//

// Reinterpret an alloc_tile address operand as u64 and TASSIGN it to the
// freshly created tile variable.
static void assignTileAddress(ConversionPatternRewriter &rewriter,
                              Location loc, MLIRContext *ctx, Value tile,
                              Value addr) {
  addr = peelUnrealized(addr);
  auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
  bool isPointerLike =
      isa<emitc::PointerType>(addr.getType()) ||
      (isa<emitc::OpaqueType>(addr.getType()) &&
       cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"));
  if (isPointerLike) {
    auto rcU64 =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
    addr = rewriter
               .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                            ArrayAttr{}, rcU64,
                                            ValueRange{addr})
               .getResult(0);
  } else if (addr.getType() != u64Ty) {
    addr = rewriter.create<emitc::CastOp>(loc, u64Ty, addr).getResult();
  }

  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, addr});
}

struct PTOAllocTileToEmitC
    : public OpConversionPattern<pto::AllocTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AllocTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto tileTy = cast<pto::TileBufType>(op.getResult().getType());
    auto tileTypeString = getEmitCTileTypeString(tileTy);
    if (!tileTypeString)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 alloc_tile handles can be converted to EmitC");

    Type convertedTy = getTypeConverter()->convertType(tileTy);
    if (!convertedTy)
      convertedTy = emitc::OpaqueType::get(ctx, *tileTypeString);

    auto validShape = tileTy.getValidShape();
    bool hasDynamicValidDim =
        llvm::any_of(validShape, [](int64_t dim) { return dim < 0; });
    SmallVector<Value> constructorArgs;
    if (hasDynamicValidDim) {
      auto args = buildDynamicValidShapeArgs(op, adaptor, rewriter, tileTy);
      if (failed(args))
        return failure();
      constructorArgs = std::move(*args);
    }

    Value tile;
    if (hasDynamicValidDim) {
      tile = rewriter
                 .create<emitc::CallOpaqueOp>(
                     loc, convertedTy, *tileTypeString, ArrayAttr{},
                     ArrayAttr{}, ValueRange(constructorArgs))
                 .getResult(0);
    } else {
      tile =
          rewriter
              .create<emitc::VariableOp>(
                  loc, getEmitCVariableResultType(convertedTy),
                  emitc::OpaqueAttr::get(ctx, ""))
              .getResult();
      tile = loadEmitCVariableIfNeeded(rewriter, loc, tile);
    }

    if (Value addr = adaptor.getAddr())
      assignTileAddress(rewriter, loc, ctx, tile, addr);

    rewriter.replaceOp(op, tile);
    return success();
  }

  // Build the runtime constructor arguments for a dynamic-valid-shape
  // alloc_tile, doubling the packed FP4 dimension when required.
  FailureOr<SmallVector<Value>>
  buildDynamicValidShapeArgs(pto::AllocTileOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             pto::TileBufType tileTy) const {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto validShape = tileTy.getValidShape();
    Type elemTy = tileTy.getElementType();
    pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
    auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
      if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
        return emitted;
      int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
      if (dimIdx != packedDim)
        return emitted;
      auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
      Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
      return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two)
          .getResult();
    };

    SmallVector<Value> constructorArgs;
    if (validShape.size() > 0 && validShape[0] < 0) {
      Value validRow = adaptor.getValidRow();
      if (!validRow)
        return rewriter.notifyMatchFailure(
            op, "dynamic alloc_tile valid row must have an operand");
      validRow = peelUnrealized(validRow);
      constructorArgs.push_back(maybeScaleDynamicValid(validRow, 0));
    }
    if (validShape.size() > 1 && validShape[1] < 0) {
      Value validCol = adaptor.getValidCol();
      if (!validCol)
        return rewriter.notifyMatchFailure(
            op, "dynamic alloc_tile valid col must have an operand");
      validCol = peelUnrealized(validCol);
      constructorArgs.push_back(maybeScaleDynamicValid(validCol, 1));
    }
    return constructorArgs;
  }
};
static FailureOr<Value>
createEmitCTileVariable(ConversionPatternRewriter &rewriter, Location loc,
                        const TypeConverter *typeConverter,
                        pto::TileBufType tileTy,
                        bool initializeDynamicValidToShape = false) {
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();

  Type convertedTy = typeConverter->convertType(tileTy);
  if (!convertedTy)
    convertedTy = emitc::OpaqueType::get(rewriter.getContext(), *tileTypeString);

  if (initializeDynamicValidToShape && tileTy.hasDynamicValid()) {
    auto shape = tileTy.getShape();
    if (shape.size() != 2 || llvm::is_contained(shape, ShapedType::kDynamic))
      return failure();
    Type i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
    SmallVector<Value, 2> constructorArgs;
    constructorArgs.push_back(makeEmitCIntConstant(
        rewriter, loc, i32Ty,
        renderTileTemplateDim(shape[0], tileTy.getElementType(), blayout, 0)));
    constructorArgs.push_back(makeEmitCIntConstant(
        rewriter, loc, i32Ty,
        renderTileTemplateDim(shape[1], tileTy.getElementType(), blayout, 1)));
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, convertedTy, *tileTypeString,
                                     ArrayAttr{}, ArrayAttr{}, constructorArgs)
        .getResult(0);
  }

  Value tile = rewriter
                   .create<emitc::VariableOp>(
                       loc, getEmitCVariableResultType(convertedTy),
                       emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                   .getResult();
  return loadEmitCVariableIfNeeded(rewriter, loc, tile);
}

struct PTODeclareTileToEmitC
    : public OpConversionPattern<pto::DeclareTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::DeclareTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto tileType = dyn_cast<pto::TileBufType>(op.getTile().getType());
    if (!tileType)
      return rewriter.notifyMatchFailure(op, "expected a tile_buf result");
    FailureOr<Value> tile = createEmitCTileVariable(
        rewriter, op.getLoc(), getTypeConverter(), tileType,
        /*initializeDynamicValidToShape=*/true);
    if (failed(tile))
      return rewriter.notifyMatchFailure(
          op, "only rank-2 declare_tile handles can be converted to EmitC");
    rewriter.replaceOp(op, *tile);
    return success();
  }
};

struct PTOTReshapeToEmitC : public OpConversionPattern<pto::TReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tileTy = dyn_cast<pto::TileBufType>(op.getResult().getType());
    if (!tileTy)
      return failure();

    FailureOr<Value> dst =
        createEmitCTileVariable(rewriter, op.getLoc(), getTypeConverter(), tileTy);
    if (failed(dst))
      return failure();

    Value src = adaptor.getSrc();
    if (auto castOp = src.getDefiningOp<emitc::CastOp>())
      src = castOp.getOperand();

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TRESHAPE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*dst, src});
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOBitcastToEmitC : public OpConversionPattern<pto::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<pto::TileBufType>(op.getResult().getType());
    auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!dstTy || !srcTy)
      return failure();

    FailureOr<Value> dst =
        createEmitCTileVariable(rewriter, op.getLoc(), getTypeConverter(), dstTy);
    if (failed(dst))
      return failure();

    Value src = adaptor.getSrc();
    if (auto castOp = src.getDefiningOp<emitc::CastOp>())
      src = castOp.getOperand();

    pto::AddressSpace as = pto::AddressSpace::GM;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcTy.getMemorySpace()))
      as = asAttr.getAddressSpace();
    std::string elemTok = getEmitCScalarTypeToken(srcTy.getElementType());

    Value rawPtr = materializeTileDataValue(rewriter, op.getLoc(), src, as, elemTok);
    auto u64Ty = emitc::OpaqueType::get(rewriter.getContext(), "uint64_t");
    Value addr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                        "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(op.getLoc(), u64Ty,
                                              "reinterpret_cast", ArrayAttr{},
                                              rcU64, ValueRange{rawPtr})
                 .getResult(0);
    } else if (addr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(op.getLoc(), u64Ty, addr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*dst, addr});
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOTileBufAddrToEmitC : public OpConversionPattern<pto::TileBufAddrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TileBufAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    if (isEmitCTileLikeType(src.getType())) {
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          op, TypeRange{dstTy},
          "PTOAS__TILE_DATA", ArrayAttr{}, ArrayAttr{}, ValueRange{src});
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, src);
    return success();
  }
};

template <typename SectionOpTy>
struct SectionToEmitC : public OpConversionPattern<SectionOpTy> {
  using OpConversionPattern<SectionOpTy>::OpConversionPattern;

  std::string getMacroName() const {
    if (std::is_same<SectionOpTy, pto::SectionCubeOp>::value)
      return "__DAV_CUBE__";
    if (std::is_same<SectionOpTy, pto::SectionVectorOp>::value)
      return "__DAV_VEC__";
    return "UNKNOWN_MACRO";
  }

  LogicalResult
  matchAndRewrite(SectionOpTy op, typename SectionOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    std::string startMacro = "\n#if defined(" + getMacroName() + ")";
    rewriter.create<emitc::VerbatimOp>(loc, startMacro);

    if constexpr (std::is_same_v<SectionOpTy, pto::SectionVectorOp>) {
      // Vector mask is a global HW state and may be modified by previous kernels
      // (or earlier sections). Reset it to a well-defined state for deterministic
      // execution of VEC ops.
      rewriter.create<emitc::VerbatimOp>(loc, "set_mask_norm();");
      rewriter.create<emitc::VerbatimOp>(loc, "set_vector_mask(-1, -1);");
    }

    if (needsNoSplitGuard) {
      rewriter.create<emitc::VerbatimOp>(
          loc, "if (get_subblockid() == 0) {");
    }

    Block &innerBlock = op.getBody().front();
    if (!innerBlock.empty()) {
      rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});
    }

    if (needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(loc, "}");

    std::string endMacro = "#endif // " + getMacroName() + "\n";
    rewriter.create<emitc::VerbatimOp>(loc, endMacro);

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCF Control-Flow Pre-Lowering
//
// EmitC translation supports `emitc.for`/`emitc.if` plus CFG-style
// `cf.br`/`cf.cond_br`. Upstream SCFToEmitC patterns only cover `scf.for` and
// `scf.if`, so we pre-lower some SCF ops into those supported forms.
//===----------------------------------------------------------------------===//


FailureOr<Value> buildGlobalTensorViewFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    std::optional<SpecialGlobalTensorTypeSpec> specialSpec,
    StringRef layoutEnum) {
  if (llvm::any_of(shape, [](int64_t dim) {
        return dim == ShapedType::kDynamic;
      }))
    return failure();

  auto *ctx = rewriter.getContext();
  SmallVector<int64_t> rowMajorStrides;
  ArrayRef<int64_t> effectiveStrides = strides;
  if (effectiveStrides.empty()) {
    rowMajorStrides = buildRowMajorStrides(shape);
    effectiveStrides = rowMajorStrides;
  }
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, effectiveStrides, shape5D, stride5D);

  std::string shapeType;
  std::string strideType;
  if (specialSpec) {
    shapeType = specialSpec->shapeTypeExpr;
    strideType = specialSpec->strideTypeExpr;
    layoutEnum = specialSpec->layoutEnum;
  } else {
    shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
    strideType = "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  }
  auto shapeVal = rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, emitc::OpaqueType::get(ctx, shapeType),
                          shapeType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                      .getResult(0);
  auto strideVal = rewriter
                       .create<emitc::CallOpaqueOp>(
                           loc, emitc::OpaqueType::get(ctx, strideType),
                           strideType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                       .getResult(0);

  // Keep the GlobalTensor template descriptors identical to the constructor
  // arguments, including the specialized MX shape and stride types.
  std::string gtTypeStr =
      "GlobalTensor<" + getElemTypeStringForGT(elemTy) + ", " + shapeType +
      ", " + strideType + ", " + layoutEnum.str() + ">";
  auto gtType = emitc::OpaqueType::get(ctx, gtTypeStr);
  auto gt = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, gtTypeStr, ArrayAttr{}, ArrayAttr{},
      ValueRange{ptr, shapeVal, strideVal});
  return gt.getResult(0);
}

// Right-align runtime shape/stride values to 5 dims: leading shape dims are
// 1, leading strides derive by tight packing (or 1 in the fully-dynamic case).
static std::pair<SmallVector<Value, 5>, SmallVector<Value, 5>>
buildRuntime5DValues(ConversionPatternRewriter &rewriter, Location loc,
                     ValueRange runtimeShape, ValueRange runtimeStrides,
                     int64_t shift) {
  SmallVector<Value, 5> shapeValues;
  SmallVector<Value, 5> strideValues;
  for (int64_t dim = 0; dim < shift; ++dim)
    shapeValues.push_back(makeViewIndexConstant(rewriter, loc, 1));
  for (Value value : runtimeShape)
    shapeValues.push_back(castViewIndexToEmitC(rewriter, loc, value));

  strideValues.resize(5);
  for (auto [index, value] : llvm::enumerate(runtimeStrides))
    strideValues[shift + static_cast<int64_t>(index)] =
        castViewIndexToEmitC(rewriter, loc, value);
  if (shift == 5) {
    for (int64_t dim = 0; dim < 5; ++dim)
      strideValues[dim] = makeViewIndexConstant(rewriter, loc, 1);
  } else {
    for (int64_t dim = shift - 1; dim >= 0; --dim) {
      strideValues[dim] =
          rewriter
              .create<emitc::MulOp>(loc, strideValues[dim + 1].getType(),
                                    shapeValues[dim + 1],
                                    strideValues[dim + 1])
              .getResult();
    }
  }
  return {shapeValues, strideValues};
}

FailureOr<Value> buildRuntimeGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> staticShape, ValueRange runtimeShape,
    ValueRange runtimeStrides, StringRef layoutEnum) {
  if (staticShape.size() > 5 || runtimeShape.size() != staticShape.size() ||
      runtimeStrides.size() != staticShape.size())
    return failure();

  SmallVector<int64_t, 5> shape5D(5, 1);
  SmallVector<int64_t, 5> stride5D(5, -1);
  int64_t shift = 5 - static_cast<int64_t>(staticShape.size());
  for (auto [index, dim] : llvm::enumerate(staticShape))
    shape5D[shift + static_cast<int64_t>(index)] =
        ShapedType::isDynamic(dim) ? -1 : dim;

  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  auto [shapeValues, strideValues] =
      buildRuntime5DValues(rewriter, loc, runtimeShape, runtimeStrides, shift);

  Value shape = rewriter
                    .create<emitc::CallOpaqueOp>(
                        loc, emitc::OpaqueType::get(rewriter.getContext(),
                                                   shapeType),
                        shapeType, ArrayAttr{}, ArrayAttr{}, shapeValues)
                    .getResult(0);
  Value stride = rewriter
                     .create<emitc::CallOpaqueOp>(
                         loc, emitc::OpaqueType::get(rewriter.getContext(),
                                                    strideType),
                         strideType, ArrayAttr{}, ArrayAttr{}, strideValues)
                     .getResult(0);
  auto resultType = getRuntimeGlobalTensorOpaqueType(
      rewriter.getContext(), elemTy, staticShape, layoutEnum);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, resultType, resultType.getValue(),
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{ptr, shape, stride})
      .getResult(0);
}

FailureOr<Value> buildSyncAllGlobalTensorFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy) {
  constexpr int64_t kWorkspaceElements = 16;
  SmallVector<int64_t, 1> shape{kWorkspaceElements};
  SmallVector<int64_t, 1> strides{1};
  return buildGlobalTensorViewFromPointer(rewriter, loc, ptr, elemTy, shape,
                                          strides);
}

Value castViewIndexToEmitC(ConversionPatternRewriter &rewriter,
                                  Location loc, Value value) {
  Type indexTy = emitc::OpaqueType::get(rewriter.getContext(), "int64_t");
  value = peelUnrealized(value);
  if (value.getType() == indexTy)
    return value;
  return rewriter.create<emitc::CastOp>(loc, indexTy, value).getResult();
}

Value getRuntimeGlobalTensorMetadata(
    ConversionPatternRewriter &rewriter, Location loc, Value tensor,
    Value logicalDim, int64_t rank, bool isStride) {
  Value dim = castViewIndexToEmitC(rewriter, loc, logicalDim);
  int64_t shift = 5 - rank;
  if (shift != 0) {
    dim = rewriter
              .create<emitc::AddOp>(loc, dim.getType(), dim,
                                    makeViewIndexConstant(rewriter, loc, shift))
              .getResult();
  }
  StringRef marker = isStride ? StringRef("PTOAS__GLOBAL_TENSOR_GET_STRIDE")
                              : StringRef("PTOAS__GLOBAL_TENSOR_GET_SHAPE");
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, dim.getType(), marker, ArrayAttr{}, ArrayAttr{},
          ValueRange{tensor, dim})
      .getResult(0);
}

std::optional<int64_t> getStaticIndexLikeValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return getIntegerAttrSignedValue(intAttr);
  }
  return std::nullopt;
}

Value makeViewIndexConstant(ConversionPatternRewriter &rewriter,
                                   Location loc, int64_t value) {
  return makeEmitCIntConstant(
      rewriter, loc, emitc::OpaqueType::get(rewriter.getContext(), "int64_t"),
      value);
}

bool parseIntegerTemplateList(StringRef token, StringRef marker,
                                     SmallVectorImpl<int64_t> &values) {
  size_t pos = token.find(marker);
  if (pos == StringRef::npos)
    return false;
  pos += marker.size();
  size_t end = token.find('>', pos);
  if (end == StringRef::npos)
    return false;

  SmallVector<StringRef, 8> parts;
  token.slice(pos, end).split(parts, ',');
  values.clear();
  for (StringRef part : parts) {
    int64_t value = 0;
    if (part.trim().getAsInteger(10, value))
      return false;
    values.push_back(value);
  }
  return true;
}

bool partitionViewHasStaticResultShape(pto::PartitionViewOp op) {
  auto resTy = dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
  if (!resTy) {
    return false;
  }

  int64_t sourceRank = 0;
  if (auto srcTy = dyn_cast<pto::TensorViewType>(op.getSource().getType())) {
    sourceRank = srcTy.getRank();
  } else if (auto srcTy =
                 dyn_cast<pto::PartitionTensorViewType>(op.getSource().getType())) {
    sourceRank = srcTy.getRank();
  } else {
    return false;
  }

  if (op.getOffsets().size() != static_cast<size_t>(sourceRank) ||
      op.getSizes().size() != static_cast<size_t>(sourceRank)) {
    return false;
  }

  for (auto [idx, value] : llvm::enumerate(op.getSizes())) {
    auto cst = getStaticIndexLikeValue(value);
    if (!cst) {
      return false;
    }
    int64_t resultDim = resTy.getShape()[idx];
    if (resultDim != ShapedType::kDynamic && resultDim != *cst) {
      return false;
    }
  }
  return true;
}


FailureOr<std::string> getPipeDataTypeToken(Value value) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType());
  if (!opaqueTy)
    return failure();
  StringRef token = opaqueTy.getValue();
  if (!token.contains("Tile<") && !token.contains("GlobalTensor<"))
    return failure();
  return token.str();
}


LogicalResult getStaticTensorViewStrides(
    Value source, Value convertedSource, int64_t rank,
    SmallVectorImpl<int64_t> &strides) {
  strides.clear();

  if (auto makeView = source.getDefiningOp<pto::MakeTensorViewOp>()) {
    if (static_cast<int64_t>(makeView.getStrides().size()) != rank)
      return failure();
    for (Value strideValue : makeView.getStrides()) {
      auto cst = getStaticIndexLikeValue(strideValue);
      if (!cst)
        return failure();
      strides.push_back(*cst);
    }
    return success();
  }

  Value src = peelUnrealized(convertedSource);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(src.getType())) {
    SmallVector<int64_t, 5> stride5D;
    StringRef token = opaqueTy.getValue();
    if ((parseIntegerTemplateList(token, "pto::Stride<", stride5D) ||
         parseIntegerTemplateList(token, "Stride<", stride5D)) &&
        static_cast<int64_t>(stride5D.size()) >= rank) {
      strides.append(stride5D.end() - rank, stride5D.end());
      return success();
    }
  }

  return failure();
}


void populateTilePatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  patterns.add<PTOAllocTileToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOTileBufAddrToEmitC>(typeConverter, ctx);
  patterns.add<MemRefCastToEmitC>(typeConverter, ctx);
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
  patterns.add<PTOMakeTensorViewToEmitC, PTOPartitionViewToEmitC,
               PTOGetTensorViewMetadataToEmitC<pto::GetTensorViewDimOp, false>,
               PTOGetTensorViewMetadataToEmitC<pto::GetTensorViewStrideOp,
                                                true>>(typeConverter, ctx);
  patterns.add<PTOPartitionViewStaticToEmitC>(typeConverter, ctx,
                                              PatternBenefit(2));
  patterns.add<PTOTReshapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOBitcastToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetQuantScalarToEmitC, PTOSetQuantVectorToEmitC>(
      typeConverter, ctx, targetArch);
  patterns.add<PTOTAllocToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTPushToEmitC>(typeConverter, ctx, targetArch, "TPUSH");
  patterns.add<PTOTPopToEmitC>(typeConverter, ctx, targetArch, "TPOP");
  patterns.add<PTOTFreeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<SectionToEmitC<pto::SectionCubeOp>>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionVectorOp>>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
