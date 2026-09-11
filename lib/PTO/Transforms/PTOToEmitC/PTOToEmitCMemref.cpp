// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCMemref.cpp - memref/global-tensor/pointer lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {



// =============================================================================
// 4. MemRef SubView -> Explicit Shape/Stride Construction (Full Implementation)
// =============================================================================
// Lower an OpFoldResult to an int64_t EmitC value using the given index type.
static Value ofrToEmitCIndexValue(ConversionPatternRewriter &rewriter,
                                  Location loc, Type indexTy,
                                  OpFoldResult ofr) {
  auto mkIndex = [&](int64_t v) -> Value {
    return rewriter.create<emitc::ConstantOp>(
        loc, indexTy, emitc::OpaqueAttr::get(rewriter.getContext(),
                                             std::to_string(v)));
  };
  auto asIndex = [&](Value value) -> Value {
    if (value.getType() == indexTy)
      return value;
    return rewriter.create<emitc::CastOp>(loc, indexTy, value).getResult();
  };
  if (isa<Value>(ofr)) {
    Value v = cast<Value>(ofr);
    Value rv = rewriter.getRemappedValue(v);
    return asIndex(rv);
  }
  if (isa<Attribute>(ofr)) {
    Attribute attr = cast<Attribute>(ofr);
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return mkIndex(getIntegerAttrSignedValue(ia));
  }
  return mkIndex(0);
}

// C++ scalar token for a MemRef element type, defaulting to float.
static std::string memrefElemTypeToString(Type elemTy) {
  if (elemTy.isF16())
    return "half";
  if (elemTy.isBF16())
    return "bfloat16_t";
  if (elemTy.isF32())
    return "float";
  if (elemTy.isF64())
    return "double";
  if (elemTy.isInteger(8)) {
    if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
      return "int8_t";
    return "uint8_t";
  }
  if (elemTy.isInteger(16)) {
    if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
      return "int16_t";
    return "uint16_t";
  }
  if (elemTy.isInteger(32)) {
    if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
      return "int32_t";
    return "uint32_t";
  }
  if (elemTy.isInteger(64)) {
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  }
  return "float";
}

struct SubviewToEmitCPattern : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  // 辅助函数：尝试从 OpFoldResult 中提取静态整数值
  std::optional<int64_t> extractStaticInt(OpFoldResult ofr) const {
    if (isa<Attribute>(ofr)) {
      Attribute attr = cast<Attribute>(ofr);
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        return getIntegerAttrSignedValue(intAttr);
      }
    } else {
      Value v = cast<Value>(ofr);
      if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto iAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
          return getIntegerAttrSignedValue(iAttr);
      } else if (auto idxOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return idxOp.value();
      }
    }
    return std::nullopt;
  }

  LogicalResult appendComposedStride(
      OpFoldResult parentStride, OpFoldResult step,
      PatternRewriter &rewriter,
      SmallVectorImpl<OpFoldResult> &strides) const {
    auto parentStatic = extractStaticInt(parentStride);
    auto stepStatic = extractStaticInt(step);
    if (parentStatic && stepStatic) {
      int64_t product = 0;
      if (llvm::MulOverflow(*parentStatic, *stepStatic, product)) {
        return failure();
      }
      strides.push_back(rewriter.getIndexAttr(product));
      return success();
    }
    if (stepStatic && *stepStatic == 1) {
      strides.push_back(parentStride);
      return success();
    }
    if (parentStatic && *parentStatic == 1) {
      strides.push_back(step);
      return success();
    }
    return failure();
  }

  LogicalResult resolveSubviewStrides(
      memref::SubViewOp subview, int64_t rank, PatternRewriter &rewriter,
      SmallVectorImpl<OpFoldResult> &strides) const {
    SmallVector<OpFoldResult> parentStrides;
    if (failed(resolveSourceStrides(subview.getSource(), rewriter,
                                    parentStrides))) {
      return failure();
    }
    auto steps = subview.getMixedStrides();
    if (parentStrides.size() != static_cast<size_t>(rank) ||
        steps.size() != static_cast<size_t>(rank)) {
      return failure();
    }

    strides.reserve(rank);
    for (auto [parentStride, step] :
         llvm::zip_equal(parentStrides, steps)) {
      if (failed(
              appendComposedStride(parentStride, step, rewriter, strides))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult resolveStaticTypeStrides(
      MemRefType sourceType, PatternRewriter &rewriter,
      SmallVectorImpl<OpFoldResult> &strides) const {
    SmallVector<int64_t> typeStrides;
    int64_t offset = ShapedType::kDynamic;
    if (failed(mlir::pto::getPTOMemRefStridesAndOffset(
            sourceType, typeStrides, offset)) ||
        typeStrides.size() != static_cast<size_t>(sourceType.getRank()) ||
        llvm::any_of(typeStrides, [](int64_t stride) {
          return stride == ShapedType::kDynamic;
        })) {
      return failure();
    }
    for (int64_t stride : typeStrides) {
      strides.push_back(rewriter.getIndexAttr(stride));
    }
    return success();
  }

  LogicalResult
  resolveSourceStrides(Value source, PatternRewriter &rewriter,
                       SmallVectorImpl<OpFoldResult> &strides) const {
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType) {
      return failure();
    }
    int64_t rank = sourceType.getRank();
    if (auto reinterpretCast =
            source.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto mixedStrides = reinterpretCast.getMixedStrides();
      if (mixedStrides.size() != static_cast<size_t>(rank)) {
        return failure();
      }
      strides.assign(mixedStrides.begin(), mixedStrides.end());
      return success();
    }
    if (auto subview = source.getDefiningOp<memref::SubViewOp>()) {
      return resolveSubviewStrides(subview, rank, rewriter, strides);
    }
    if (auto cast = source.getDefiningOp<memref::CastOp>()) {
      return resolveSourceStrides(cast.getSource(), rewriter, strides);
    }
    return resolveStaticTypeStrides(sourceType, rewriter, strides);
  }

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());

    // 1. 获取 Source 的 Strides (支持动态 Stride 收集)
    SmallVector<OpFoldResult> sourceStrides;
    if (failed(resolveSourceStrides(op.getSource(), rewriter, sourceStrides)))
      return rewriter.notifyMatchFailure(
          op, "cannot resolve exact source strides; refusing to assume a "
              "compact layout");

    // 2. 计算运行时 Offset
    FailureOr<Value> totalOffset = computeTotalOffset(op, adaptor, rewriter,
                                                      sourceStrides);
    if (failed(totalOffset))
      return rewriter.notifyMatchFailure(op, "failed to compute subview offset");

    // 3. 生成新指针
    Value newPtr =
        computeOffsetPointer(op, adaptor, rewriter, srcType, *totalOffset);

    // For non-GM memrefs, keep the raw pointer (no GlobalTensor).
    if (!isGlobalMemref(srcType)) {
      Type dstTy = getTypeConverter()->convertType(op.getType());
      if (!dstTy)
        return failure();
      if (newPtr.getType() != dstTy)
        newPtr = rewriter.create<emitc::CastOp>(loc, dstTy, newPtr);
      rewriter.replaceOp(op, newPtr);
      return success();
    }

    return emitGlobalTensor(op, adaptor, rewriter, srcType, sourceStrides,
                            newPtr);
  }

  // Fold every dimension's offset*stride contribution into one index value.
  FailureOr<Value> computeTotalOffset(
      memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      const SmallVectorImpl<OpFoldResult> &sourceStrides) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());
    int64_t rank = srcType.getRank();
    Type indexTy = emitc::OpaqueType::get(ctx, "int64_t");

    auto mkIndex = [&](int64_t v) -> Value {
      return rewriter.create<emitc::ConstantOp>(
          loc, indexTy, emitc::OpaqueAttr::get(ctx, std::to_string(v)));
    };
    auto asIndex = [&](Value value) -> Value {
      if (value.getType() == indexTy)
        return value;
      return rewriter.create<emitc::CastOp>(loc, indexTy, value).getResult();
    };

    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = adaptor.getOffsets();
    int dynOffIdx = 0;
    Value totalOffset = mkIndex(0);
    for (int i = 0; i < rank; ++i) {
      Value offVal;
      if (staticOffsets[i] == ShapedType::kDynamic) {
        Value rawDyn = dynamicOffsets[dynOffIdx++];
        offVal = asIndex(rawDyn);
      } else {
        offVal = mkIndex(staticOffsets[i]);
      }

      Value strideVal = mkIndex(1);
      if (i < static_cast<int>(sourceStrides.size()))
        strideVal = ofrToEmitCIndexValue(rewriter, loc, indexTy, sourceStrides[i]);

      Value term =
          rewriter.create<emitc::MulOp>(loc, indexTy, offVal, strideVal);
      totalOffset =
          rewriter.create<emitc::AddOp>(loc, indexTy, totalOffset, term);
    }
    return totalOffset;
  }

  // Resolve the source base pointer and apply the subview offset to it.
  Value computeOffsetPointer(memref::SubViewOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             MemRefType srcType, Value totalOffset) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();


    Value convertedSource = adaptor.getSource();
    if (auto cast =
            convertedSource.getDefiningOp<UnrealizedConversionCastOp>())
      convertedSource = cast.getOperand(0);
    Value sourcePtr = materializeGlobalTensorDataPointer(
        rewriter, loc, convertedSource, op.getSource().getType());
    Value tileCandidate = sourcePtr;
    if (auto castOp = sourcePtr.getDefiningOp<emitc::CastOp>()) {
      tileCandidate = castOp.getOperand();
    } else if (auto uc =
                   sourcePtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      tileCandidate = uc.getOperand(0);
    }
    if (auto ot = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
      auto tyStr = ot.getValue();
      if (tyStr.find("Tile<") != std::string::npos ||
          tyStr.find("ConvTile<") != std::string::npos) {
        std::string elemTok = memrefElemTypeToString(srcType.getElementType());
        pto::AddressSpace as = pto::AddressSpace::GM;
        if (auto asAttr =
                dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace()))
          as = asAttr.getAddressSpace();
        sourcePtr =
            materializeTileDataValue(rewriter, loc, tileCandidate, as, elemTok);
        if (tileDataReturnsIntegralAddress(as))
          sourcePtr =
              materializeAddressAsPointer(rewriter, loc, sourcePtr, as, elemTok);
      }
    }

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    Type elemTy = resTy.getElementType();
    std::string castElemTypeStr = getEmitCScalarTypeToken(elemTy);

    std::string qualifier = "__gm__";
    if (Attribute ms = srcType.getMemorySpace()) {
      if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(ms))
        qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
    }

    auto typedPtrTy = getEmitCPointerType(ctx, qualifier, castElemTypeStr);
    Value typedSourcePtr = sourcePtr;
    if (typedSourcePtr.getType() != typedPtrTy)
      typedSourcePtr =
          rewriter.create<emitc::CastOp>(loc, typedPtrTy, typedSourcePtr);
    return rewriter.create<emitc::AddOp>(loc, typedPtrTy, typedSourcePtr,
                                         totalOffset);
  }

  static bool isGlobalMemref(MemRefType srcType) {
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      return as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero;
    }
    return true;
  }

  // Build the Shape/Stride/GlobalTensor construction for GM subviews.
  // Compose per-dimension subview strides: source stride x step, keeping
  // static products in the template list and dynamic ones as runtime values.
  LogicalResult composeSubviewStrides(
      memref::SubViewOp op,
      const SmallVectorImpl<OpFoldResult> &sourceStrides,
      ConversionPatternRewriter &rewriter, Type indexTy,
      const std::function<Value(int64_t)> &mkIndex,
      SmallVectorImpl<int64_t> &strideTemplateVec,
      SmallVectorImpl<Value> &strideValues) const {
    int64_t rank = sourceStrides.size();
    auto subViewSteps = op.getMixedStrides();
    auto ofrToValue = [&](OpFoldResult ofr) -> Value {
      if (isa<Value>(ofr)) {
        Value v = cast<Value>(ofr);
        Value rv = rewriter.getRemappedValue(v);
        if (rv.getType() == indexTy)
          return rv;
        return rewriter
            .create<emitc::CastOp>(op.getLoc(), indexTy, rv)
            .getResult();
      }
      Attribute attr = cast<Attribute>(ofr);
      if (auto ia = dyn_cast<IntegerAttr>(attr))
        return mkIndex(getIntegerAttrSignedValue(ia));
      return mkIndex(0);
    };

    for (int i = 0; i < rank; ++i) {
      OpFoldResult srcStrideOfr =
          (i < static_cast<int>(sourceStrides.size())) ? sourceStrides[i]
                                                       : rewriter.getIndexAttr(1);
      OpFoldResult stepOfr = (i < static_cast<int>(subViewSteps.size()))
                                 ? subViewSteps[i]
                                 : rewriter.getIndexAttr(1);

      auto srcStatic = extractStaticInt(srcStrideOfr);
      auto stepStatic = extractStaticInt(stepOfr);
      if (srcStatic && stepStatic) {
        int64_t finalStride = 0;
        if (llvm::MulOverflow(*srcStatic, *stepStatic, finalStride)) {
          return rewriter.notifyMatchFailure(
              op, "source stride and subview step product overflows");
        }
        strideTemplateVec.push_back(finalStride);
        strideValues.push_back(mkIndex(finalStride));
        continue;
      }

      strideTemplateVec.push_back(-1);
      Value srcV = ofrToValue(srcStrideOfr);
      Value stepV = ofrToValue(stepOfr);
      if (stepStatic && *stepStatic == 1) {
        strideValues.push_back(srcV);
      } else if (srcStatic && *srcStatic == 1) {
        strideValues.push_back(stepV);
      } else {
        strideValues.push_back(rewriter.create<emitc::MulOp>(
            op.getLoc(), indexTy, srcV, stepV));
      }
    }
    return success();
  }

  // Right-aligned 5D shape/stride bundle: template dims plus runtime values.
  struct Aligned5DDims {
    SmallVector<int64_t, 5> shape;
    SmallVector<int64_t, 5> stride;
    SmallVector<Value, 5> shapeValues;
    SmallVector<Value, 5> strideValues;
  };

  // Right-align the subview's shape/strides to 5 dims: existing dims inherit
  // their values, prepended dims derive strides by tight packing.
  Aligned5DDims buildRightAligned5DDims(
      ConversionPatternRewriter &rewriter, Location loc,
      const std::function<Value(int64_t)> &mkIndex, int64_t rank,
      const SmallVectorImpl<int64_t> &shapeParamsVec,
      const SmallVectorImpl<int64_t> &strideTemplateVec,
      const SmallVectorImpl<Value> &sizeValues,
      const SmallVectorImpl<Value> &strideValues) const {
    Aligned5DDims dims;
    buildGlobalTensorShapeAndStride(shapeParamsVec, strideTemplateVec,
                                    dims.shape, dims.stride);
    Value oneIndex = mkIndex(1);
    dims.shapeValues.assign(5, oneIndex);
    dims.strideValues.assign(5, oneIndex);
    int shift = 5 - rank;

    for (int i = 0; i < rank && i < 5; ++i) {
      dims.shapeValues[shift + i] = sizeValues[i];
      dims.strideValues[shift + i] = strideValues[i];
    }

    for (int i = 3; i >= 0; --i) {
      if (i >= shift)
        continue;
      if (dims.stride[i] != -1) {
        dims.strideValues[i] = mkIndex(dims.stride[i]);
        continue;
      }
      if (dims.shape[i + 1] == 1) {
        dims.strideValues[i] = dims.strideValues[i + 1];
      } else {
        dims.strideValues[i] = rewriter.create<emitc::MulOp>(
            loc, dims.shapeValues[i + 1].getType(),
            dims.shapeValues[i + 1], dims.strideValues[i + 1]);
      }
    }
    return dims;
  }

  // Resolve the layout enum for the subview result: the special scale
  // spec's layout wins, then the explicitly resolved layout, then local
  // inference from shape/strides.
  std::string resolveSubviewLayoutEnum(
      memref::SubViewOp op, const SmallVectorImpl<int64_t> &shapeParamsVec,
      const SmallVectorImpl<int64_t> &strideTemplateVec,
      MemRefType resTy) const {
    auto resolvedLayout = resolveLayoutForGlobalTensor(op, op.getSource());
    auto specialScaleSpec = getSpecialGlobalTensorTypeSpecForLayout(
        resolvedLayout, resTy.getShape(), resTy.getElementType());
    if (specialScaleSpec)
      return specialScaleSpec->layoutEnum;
    if (resolvedLayout)
      return layoutToEmitCString(*resolvedLayout);
    if (auto inferred = inferLayout5D(
            shapeParamsVec, strideTemplateVec,
            getPTOStorageElemByteSize(resTy.getElementType())))
      return layoutToEmitCString(*inferred);
    return "pto::Layout::ND";
  }

  LogicalResult emitGlobalTensor(memref::SubViewOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 MemRefType srcType,
                                 const SmallVectorImpl<OpFoldResult> &sourceStrides,
                                 Value newPtr) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Type indexTy = emitc::OpaqueType::get(ctx, "int64_t");

    auto mkIndex = [&](int64_t v) -> Value {
      return rewriter.create<emitc::ConstantOp>(
          loc, indexTy, emitc::OpaqueAttr::get(ctx, std::to_string(v)));
    };

    // When emitting C++ with `declareVariablesAtTop`, value declarations are
    // hoisted before body statements. Avoid introducing local `using` aliases
    // for templated types (Shape/Stride/GlobalTensor) because those aliases
    // would appear after the hoisted declarations and break compilation
    // (`unknown type name`).
    //
    // Instead, use the fully spelled template types as EmitC opaque types.

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    std::string elemTypeStr = getElemTypeStringForGT(resTy.getElementType());

    GlobalTensorTypes types;
    if (failed(resolveGlobalTensorTypes(op, rewriter, srcType, sourceStrides,
                                        indexTy, mkIndex, types)))
      return failure();

    return emitGlobalTensorObjects(op, rewriter, elemTypeStr,
                                    types.shapeCppType, types.strideCppType,
                                    types.layoutEnum, adaptor, newPtr,
                                    types.dims);
  }

  // Shape/Stride template parameters, aligned 5D dims and layout token for
  // the GlobalTensor lowering of a subview.
  struct GlobalTensorTypes {
    Aligned5DDims dims;
    std::string shapeCppType;
    std::string strideCppType;
    std::string layoutEnum;
  };

  // Resolve the Shape/Stride C++ types (with special scaling-layout
  // spellings) and right-align the subview to 5 dims.
  LogicalResult
  resolveGlobalTensorTypes(memref::SubViewOp op,
                           ConversionPatternRewriter &rewriter,
                           MemRefType srcType,
                           const SmallVectorImpl<OpFoldResult> &sourceStrides,
                           Type indexTy,
                           const std::function<Value(int64_t)> &mkIndex,
                           GlobalTensorTypes &out) const {
    auto loc = op.getLoc();
    int64_t rank = srcType.getRank();
    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());

    // 生成 Shape 模板参数
    SmallVector<int64_t> shapeParamsVec;
    SmallVector<Value> sizeValues;
    auto resShape = resTy.getShape();
    auto mixedSizes = op.getMixedSizes();
    sizeValues.reserve(rank);
    for (int i = 0; i < resTy.getRank(); ++i) {
      if (resShape[i] == ShapedType::kDynamic) {
        shapeParamsVec.push_back(-1);
      } else {
        shapeParamsVec.push_back(resShape[i]);
      }
      if (i < static_cast<int>(mixedSizes.size())) {
        sizeValues.push_back(ofrToEmitCIndexValue(rewriter, loc, indexTy, mixedSizes[i]));
      } else {
        sizeValues.push_back(
            mkIndex(resShape[i] == ShapedType::kDynamic ? 1 : resShape[i]));
      }
    }
    SmallVector<int64_t> strideTemplateVec;
    SmallVector<Value> strideValues;
    if (failed(composeSubviewStrides(op, sourceStrides, rewriter, indexTy,
                                     mkIndex, strideTemplateVec,
                                     strideValues)))
      return failure();

    // 右对齐到 5 维
    out.dims = buildRightAligned5DDims(
        rewriter, loc, mkIndex, rank, shapeParamsVec, strideTemplateVec,
        sizeValues, strideValues);
    const auto &finalShape = out.dims.shape;
    const auto &finalStride = out.dims.stride;

    std::string shapeParams = joinIntTemplateParams(finalShape);
    std::string strideParams = joinIntTemplateParams(finalStride);

    auto resolvedLayout = resolveLayoutForGlobalTensor(op, op.getSource());
    auto specialScaleSpec = getSpecialGlobalTensorTypeSpecForLayout(
        resolvedLayout, resTy.getShape(), resTy.getElementType());
    out.shapeCppType =
        specialScaleSpec ? specialScaleSpec->shapeTypeExpr
                         : "pto::Shape<" + shapeParams + ">";
    out.strideCppType =
        specialScaleSpec ? specialScaleSpec->strideTypeExpr
                         : "pto::Stride<" + strideParams + ">";
    out.layoutEnum = resolveSubviewLayoutEnum(op, shapeParamsVec,
                                              strideTemplateVec, resTy);
    return success();
  }

  // Instantiate the Shape/Stride/GlobalTensor objects and replace the op.
  LogicalResult emitGlobalTensorObjects(
      memref::SubViewOp op, ConversionPatternRewriter &rewriter,
      const std::string &elemTypeStr, const std::string &shapeCppType,
      const std::string &strideCppType, const std::string &layoutEnum,
      OpAdaptor adaptor, Value newPtr, const Aligned5DDims &dims) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // A. Instantiate Shape object.
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeCppType);
    SmallVector<Value> shapeArgs;
    for (Value dynSize : adaptor.getSizes())
      shapeArgs.push_back(dynSize);

    auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, shapeTypeOpaque, shapeCppType,
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(shapeArgs));

    // B. Instantiate Stride object: only the dynamic stride dims match the
    // pto::Stride N-parameter ctor (and its static_assert).
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideCppType);
    SmallVector<Value> strideCtorArgs;
    strideCtorArgs.reserve(5);
    for (int i = 0; i < 5; ++i) {
      if (dims.stride[i] == -1)
        strideCtorArgs.push_back(dims.strideValues[i]);
    }
    auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, strideTypeOpaque, strideCppType,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(strideCtorArgs));

    // C. Instantiate GlobalTensor object (ptr + shape + stride).
    std::string gtCppType = "GlobalTensor<" + elemTypeStr + ", " + shapeCppType +
                            ", " + strideCppType + ", " + layoutEnum + ">";
    auto gtType = emitc::OpaqueType::get(ctx, gtCppType);

    SmallVector<Value> gtConstructorArgs;
    gtConstructorArgs.push_back(newPtr);
    gtConstructorArgs.push_back(shapeInstOp.getResult(0));
    gtConstructorArgs.push_back(strideInstOp.getResult(0));

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, gtType, gtCppType,
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(gtConstructorArgs));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

























//===----------------------------------------------------------------------===//
// PTO pointer lowering
//===----------------------------------------------------------------------===

struct CastPtrConversion : public OpConversionPattern<pto::CastPtrOp> {
  using OpConversionPattern<pto::CastPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getInput();
    Value peeledInput = peelUnrealized(input);
    if (peeledInput.getType() == convertedResultType) {
      rewriter.replaceOp(op, peeledInput);
      return success();
    }

    return rewriteCastPtr(op, rewriter, input, peeledInput,
                          convertedResultType);
  }

  // Dispatch on the result type family: !pto.ptr, integer, or plain EmitC.
  LogicalResult rewriteCastPtr(pto::CastPtrOp op,
                               ConversionPatternRewriter &rewriter,
                               Value input, Value peeledInput,
                               Type convertedResultType) const {
    if (isa<pto::PtrType>(op.getResult().getType()))
      return rewritePtrResult(op, rewriter, peeledInput, convertedResultType);

    if (isa<IntegerType>(op.getResult().getType()) &&
        emitc::isSupportedEmitCType(convertedResultType))
      return rewriteIntegerResult(op, rewriter, input, peeledInput,
                                  convertedResultType);

    if (emitc::isSupportedEmitCType(input.getType()) &&
        emitc::isSupportedEmitCType(convertedResultType)) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, convertedResultType, input);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported castptr conversion");
  }

  // cast_ptr to another !pto.ptr: unwrap tiles via PTOAS__TILE_DATA or
  // re-materialize the address with the target address-space qualifier.
  LogicalResult rewritePtrResult(pto::CastPtrOp op,
                                 ConversionPatternRewriter &rewriter,
                                 Value peeledInput,
                                 Type convertedResultType) const {
    auto resultPtrTy = cast<pto::PtrType>(op.getResult().getType());
    std::string elemTok = getEmitCScalarTypeToken(resultPtrTy.getElementType());
    std::optional<pto::AddressSpace> as =
        getAddressSpaceOrGM(resultPtrTy.getMemorySpace());
    if (!as)
      return rewriter.notifyMatchFailure(op, "unsupported ptr address space");

    if (isEmitCTileLikeType(peeledInput.getType())) {
      Value ptr = rewriter
                      .create<emitc::CallOpaqueOp>(
                          op.getLoc(), convertedResultType, "PTOAS__TILE_DATA",
                          ArrayAttr{}, ArrayAttr{}, ValueRange{peeledInput})
                      .getResult(0);
      rewriter.replaceOp(op, ptr);
      return success();
    }

    Value ptr = materializeAddressAsPointer(rewriter, op.getLoc(), peeledInput,
                                            *as, elemTok);
    if (ptr.getType() != convertedResultType)
      ptr = rewriter.create<emitc::CastOp>(op.getLoc(), convertedResultType, ptr)
                .getResult();
    rewriter.replaceOp(op, ptr);
    return success();
  }

  // cast_ptr to an integer: reify the pointer value then reinterpret_cast.
  LogicalResult rewriteIntegerResult(pto::CastPtrOp op,
                                     ConversionPatternRewriter &rewriter,
                                     Value input, Value peeledInput,
                                     Type convertedResultType) const {
    Value source = input;
    if (!emitc::isSupportedEmitCType(source.getType())) {
      if (auto inputPtrTy = dyn_cast<pto::PtrType>(op.getInput().getType())) {
        std::string elemTok =
            getEmitCScalarTypeToken(inputPtrTy.getElementType());
        std::optional<pto::AddressSpace> as =
            getAddressSpaceOrGM(inputPtrTy.getMemorySpace());
        if (!as)
          return rewriter.notifyMatchFailure(op, "unsupported ptr address space");
        if (isEmitCTileLikeType(peeledInput.getType())) {
          Type convertedInputType =
              getTypeConverter()->convertType(op.getInput().getType());
          if (!convertedInputType)
            return rewriter.notifyMatchFailure(op,
                                               "failed to convert input ptr type");
          source = rewriter
                       .create<emitc::CallOpaqueOp>(
                           op.getLoc(), convertedInputType, "PTOAS__TILE_DATA",
                           ArrayAttr{}, ArrayAttr{}, ValueRange{peeledInput})
                       .getResult(0);
        } else {
          source = materializeAddressAsPointer(rewriter, op.getLoc(),
                                               peeledInput, *as, elemTok);
        }
      }
    }
    if (!emitc::isSupportedEmitCType(source.getType()))
      return rewriter.notifyMatchFailure(op, "unsupported castptr integer source");
    auto templateArgs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(rewriter.getContext(),
                                cast<emitc::OpaqueType>(convertedResultType)
                                    .getValue())});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), convertedResultType, "reinterpret_cast", ArrayAttr{},
        templateArgs, ValueRange{source});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};


void populateMemrefPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
  patterns.add<CastPtrConversion>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
