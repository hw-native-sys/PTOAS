// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCSubview.cpp ---------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

enum class Role { A, B, C, Unknown };
constexpr unsigned kInlineCapacity5 = 5;
constexpr unsigned kNumber2 = 2;
constexpr unsigned kNumber4 = 4;
constexpr unsigned kNumber16 = 16;

template <typename T>
using SmallVec5 = SmallVector<T, kInlineCapacity5>;

template <typename MatmulLikeOp>
static std::optional<Role> inferMatmulLikeSubviewRole(MatmulLikeOp op,
                                                      Value buffer) {
  if (op.getLhs() == buffer)
    return Role::A;
  if (op.getRhs() == buffer)
    return Role::B;
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromLoadUser(mlir::pto::TLoadOp load) {
  Value buffer = load.getDst();
  if (!buffer)
    return std::nullopt;
  for (Operation *user : buffer.getUsers()) {
    if (auto matmul = dyn_cast<mlir::pto::TMatmulOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmul, buffer))
        return role;
      continue;
    }
    if (auto matmulAcc = dyn_cast<mlir::pto::TMatmulAccOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmulAcc, buffer))
        return role;
    }
  }
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromUser(Operation *user, Value result) {
  if (auto load = dyn_cast<mlir::pto::TLoadOp>(user))
    return inferSubviewRoleFromLoadUser(load);
  if (auto store = dyn_cast<mlir::pto::TStoreOp>(user)) {
    if (store.getDst() == result)
      return Role::C;
  }
  return std::nullopt;
}

[[maybe_unused]] static Role inferSubviewRole(memref::SubViewOp sv) {
  Value result = sv.getResult();
  for (Operation *user : result.getUsers()) {
    if (auto role = inferSubviewRoleFromUser(user, result))
      return *role;
  }
  return Role::Unknown;
}

struct SubviewToEmitCPattern : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  struct SourceStrideInfo {
    SmallVector<OpFoldResult> sourceStrides;
  };

  struct OffsetComputation {
    Type u32Ty;
    Value totalOffset;
  };

  struct SubviewStrideShapeInfo {
    SmallVec5<int64_t> finalShape;
    SmallVec5<int64_t> finalStride;
    SmallVec5<Value> finalShapeValues;
    SmallVec5<Value> finalStrideValues;
  };

  struct SubviewTemplateStrideInfo {
    SmallVector<int64_t> strideTemplateVec;
    SmallVector<Value> strideValues;
  };

  std::optional<int64_t> extractStaticInt(OpFoldResult ofr) const {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getInt();
    } else {
      Value v = ofr.get<Value>();
      if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto iAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
          return iAttr.getInt();
      } else if (auto idxOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return idxOp.value();
      }
    }
    return std::nullopt;
  }

  static std::string elemTypeToString(Type elemTy) {
    return getEmitCScalarTypeToken(elemTy);
  }

  static Value makeU32Constant(ConversionPatternRewriter &rewriter, Location loc,
                               Type u32Ty, int64_t value) {
    auto *ctx = rewriter.getContext();
    return rewriter.create<emitc::ConstantOp>(
        loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(value)));
  }

  static Value convertOfrToU32Value(ConversionPatternRewriter &rewriter,
                                    Location loc, Type u32Ty,
                                    OpFoldResult ofr) {
    if (auto v = ofr.dyn_cast<Value>()) {
      Value rv = rewriter.getRemappedValue(v);
      if (rv.getType() != u32Ty)
        return rewriter.create<emitc::CastOp>(loc, u32Ty, rv).getResult();
      return rv;
    }
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto ia = dyn_cast<IntegerAttr>(attr))
        return makeU32Constant(rewriter, loc, u32Ty,
                               ia.getValue().getSExtValue());
    }
    return makeU32Constant(rewriter, loc, u32Ty, 0);
  }

  SourceStrideInfo getSourceStrideInfo(memref::SubViewOp op, MemRefType srcType,
                                       ConversionPatternRewriter &rewriter) const {
    SourceStrideInfo result;
    int64_t rank = srcType.getRank();
    if (auto rc = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
      result.sourceStrides = llvm::to_vector(rc.getMixedStrides());
      return result;
    }

    SmallVector<int64_t> strideInts;
    int64_t offset = ShapedType::kDynamic;
    bool useTypeStrides = succeeded(getStridesAndOffset(srcType, strideInts, offset));
    (void)offset;
    if (useTypeStrides) {
      for (int64_t s : strideInts) {
        if (s == ShapedType::kDynamic)
          useTypeStrides = false;
      }
    }
    if (useTypeStrides) {
      for (int64_t s : strideInts)
        result.sourceStrides.push_back(rewriter.getIndexAttr(s));
      return result;
    }

    auto shape = srcType.getShape();
    int64_t current = 1;
    result.sourceStrides.resize(rank);
    for (int i = rank - 1; i >= 0; --i) {
      result.sourceStrides[i] = rewriter.getIndexAttr(current);
      if (shape[i] != ShapedType::kDynamic)
        current *= shape[i];
    }
    return result;
  }

  OffsetComputation buildSubviewOffset(memref::SubViewOp op, OpAdaptor adaptor,
                                       ArrayRef<OpFoldResult> sourceStrides,
                                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = adaptor.getOffsets();
    int dynOffIdx = 0;
    Value totalOffset = makeU32Constant(rewriter, loc, u32Ty, 0);
    for (int i = 0, rank = op.getSourceType().getRank(); i < rank; ++i) {
      Value offVal = staticOffsets[i] == ShapedType::kDynamic
                         ? rewriter.create<emitc::CastOp>(
                               loc, u32Ty, dynamicOffsets[dynOffIdx++])
                               .getResult()
                         : makeU32Constant(rewriter, loc, u32Ty, staticOffsets[i]);
      Value strideVal = makeU32Constant(rewriter, loc, u32Ty, 1);
      if (i < static_cast<int>(sourceStrides.size()))
        strideVal = convertOfrToU32Value(rewriter, loc, u32Ty, sourceStrides[i]);
      Value term = rewriter.create<emitc::MulOp>(loc, u32Ty, offVal, strideVal);
      totalOffset = rewriter.create<emitc::AddOp>(loc, u32Ty, totalOffset, term);
    }
    return {u32Ty, totalOffset};
  }

  static Value peelCastLikeValue(Value sourcePtr) {
    if (auto castOp = sourcePtr.getDefiningOp<emitc::CastOp>())
      return castOp.getOperand();
    if (auto uc = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>())
      return uc.getOperand(0);
    return sourcePtr;
  }

  static pto::AddressSpace getMemRefAddressSpace(MemRefType srcType) {
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      return asAttr.getAddressSpace();
    }
    return pto::AddressSpace::GM;
  }

  static Value materializeSubviewSourcePointer(
      Value sourcePtr, Value tileCandidate, MemRefType srcType, Location loc,
      ConversionPatternRewriter &rewriter) {
    if (auto ot = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
      auto tyStr = ot.getValue();
      if (tyStr.find("Tile<") != std::string::npos ||
          tyStr.find("ConvTile<") != std::string::npos) {
        std::string elemTok = elemTypeToString(srcType.getElementType());
        pto::AddressSpace as = getMemRefAddressSpace(srcType);
        sourcePtr =
            materializeTileDataValue(rewriter, loc, tileCandidate, as, elemTok);
        if (tileDataReturnsIntegralAddress(as))
          sourcePtr =
              materializeAddressAsPointer(rewriter, loc, sourcePtr, as, elemTok);
      }
    }
    return sourcePtr;
  }

  static Value buildSubviewResultPointer(memref::SubViewOp op, MemRefType srcType,
                                         Value sourcePtr, Value totalOffset,
                                         ConversionPatternRewriter &rewriter) {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    Type elemTy = resTy.getElementType();
    if (!elemTy.isInteger(kPTOI16BitWidth))
      return rewriter.create<emitc::AddOp>(loc, sourcePtr.getType(), sourcePtr,
                                           totalOffset);

    std::string castElemTypeStr =
        cast<IntegerType>(elemTy).isUnsigned() ? "uint16_t" : "int16_t";
    std::string qualifier = "__gm__";
    if (Attribute ms = srcType.getMemorySpace()) {
      if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(ms))
        qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
    }
    auto typedPtrTy = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, qualifier + " " + castElemTypeStr));
    Value typedSourcePtr =
        rewriter.create<emitc::CastOp>(loc, typedPtrTy, sourcePtr);
    return rewriter.create<emitc::AddOp>(loc, typedPtrTy, typedSourcePtr,
                                         totalOffset);
  }

  Type convertSubviewResultType(memref::SubViewOp op) const {
    return getTypeConverter()->convertType(op.getType());
  }

  LogicalResult replaceNonGlobalSubview(memref::SubViewOp op, MemRefType srcType,
                                        Value newPtr,
                                        ConversionPatternRewriter &rewriter) const {
    bool isGlobal = true;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
    }
    if (isGlobal)
      return failure();

    Type dstTy = convertSubviewResultType(op);
    if (!dstTy)
      return failure();
    if (newPtr.getType() != dstTy)
      newPtr = rewriter.create<emitc::CastOp>(op.getLoc(), dstTy, newPtr);
    rewriter.replaceOp(op, newPtr);
    return success();
  }

  SmallVector<Value> buildSubviewSizeValues(memref::SubViewOp op, Type u32Ty,
                                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    auto resShape = resTy.getShape();
    auto mixedSizes = op.getMixedSizes();
    SmallVector<Value> sizeValues;
    sizeValues.reserve(resTy.getRank());
    for (int i = 0; i < resTy.getRank(); ++i) {
      if (i < static_cast<int>(mixedSizes.size())) {
        sizeValues.push_back(convertOfrToU32Value(rewriter, loc, u32Ty, mixedSizes[i]));
      } else {
        sizeValues.push_back(makeU32Constant(
            rewriter, loc, u32Ty, resShape[i] == ShapedType::kDynamic ? 1 : resShape[i]));
      }
    }
    return sizeValues;
  }

  SubviewTemplateStrideInfo buildSubviewTemplateStrideInfo(
      memref::SubViewOp op, Type u32Ty, ArrayRef<OpFoldResult> sourceStrides,
      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    int64_t rank = cast<MemRefType>(op.getResult().getType()).getRank();
    SubviewTemplateStrideInfo result;
    auto subViewSteps = op.getMixedStrides();
    result.strideTemplateVec.reserve(rank);
    result.strideValues.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      OpFoldResult srcStrideOfr =
          i < static_cast<int>(sourceStrides.size()) ? sourceStrides[i]
                                                     : rewriter.getIndexAttr(1);
      OpFoldResult stepOfr =
          i < static_cast<int>(subViewSteps.size()) ? subViewSteps[i]
                                                    : rewriter.getIndexAttr(1);
      auto srcStatic = extractStaticInt(srcStrideOfr);
      auto stepStatic = extractStaticInt(stepOfr);
      if (srcStatic && stepStatic) {
        int64_t finalStride = (*srcStatic) * (*stepStatic);
        result.strideTemplateVec.push_back(finalStride);
        result.strideValues.push_back(
            makeU32Constant(rewriter, loc, u32Ty, finalStride));
        continue;
      }
      result.strideTemplateVec.push_back(-1);
      Value srcV = convertOfrToU32Value(rewriter, loc, u32Ty, srcStrideOfr);
      Value stepV = convertOfrToU32Value(rewriter, loc, u32Ty, stepOfr);
      if (stepStatic && *stepStatic == 1)
        result.strideValues.push_back(srcV);
      else if (srcStatic && *srcStatic == 1)
        result.strideValues.push_back(stepV);
      else
        result.strideValues.push_back(
            rewriter.create<emitc::MulOp>(loc, u32Ty, srcV, stepV));
    }
    return result;
  }

  static void finalizeSubviewStrideShapeInfo(
      SubviewStrideShapeInfo &result, Type u32Ty, ArrayRef<Value> sizeValues,
      ArrayRef<Value> strideValues, int64_t rank,
      ConversionPatternRewriter &rewriter, Location loc) {
    Value oneU32 = makeU32Constant(rewriter, loc, u32Ty, 1);
    result.finalShapeValues.assign(kPTOPaddedTensorRank5D, oneU32);
    result.finalStrideValues.assign(kPTOPaddedTensorRank5D, oneU32);
    int shift = static_cast<int>(kPTOPaddedTensorRank5D) - rank;
    for (int i = 0; i < rank &&
                    i < static_cast<int>(kPTOPaddedTensorRank5D);
         ++i) {
      result.finalShapeValues[shift + i] = sizeValues[i];
      result.finalStrideValues[shift + i] = strideValues[i];
    }
    for (int i = 3; i >= 0; --i) {
      if (i >= shift)
        continue;
      if (result.finalStride[i] != -1) {
        result.finalStrideValues[i] =
            makeU32Constant(rewriter, loc, u32Ty, result.finalStride[i]);
        continue;
      }
      if (result.finalShape[i + 1] == 1) {
        result.finalStrideValues[i] = result.finalStrideValues[i + 1];
      } else {
        result.finalStrideValues[i] = rewriter.create<emitc::MulOp>(
            loc, u32Ty, result.finalShapeValues[i + 1],
            result.finalStrideValues[i + 1]);
      }
    }
  }

  SubviewStrideShapeInfo buildSubviewStrideShapeInfo(
      memref::SubViewOp op, OpAdaptor, Type u32Ty,
      ArrayRef<OpFoldResult> sourceStrides,
      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    int64_t rank = resTy.getRank();
    SubviewStrideShapeInfo result;
    SmallVector<int64_t> shapeParamsVec;
    auto resShape = resTy.getShape();
    for (int i = 0; i < rank; ++i) {
      shapeParamsVec.push_back(resShape[i] == ShapedType::kDynamic ? -1 : resShape[i]);
    }
    SmallVector<Value> sizeValues = buildSubviewSizeValues(op, u32Ty, rewriter);
    SubviewTemplateStrideInfo strideInfo =
        buildSubviewTemplateStrideInfo(op, u32Ty, sourceStrides, rewriter);
    buildGlobalTensorShapeAndStride(shapeParamsVec, strideInfo.strideTemplateVec,
                                    result.finalShape, result.finalStride);
    finalizeSubviewStrideShapeInfo(result, u32Ty, sizeValues,
                                   strideInfo.strideValues, rank, rewriter, loc);
    return result;
  }

  static std::string resolveSubviewLayout(memref::SubViewOp op, StringRef elemTypeStr,
                                          ArrayRef<int64_t> finalShape,
                                          ArrayRef<int64_t> finalStride) {
    if (auto layout = resolveLayoutForGlobalTensor(op, op.getSource()))
      return layoutToEmitCString(*layout);

    bool allStatic =
        llvm::all_of(finalShape, [](int64_t value) { return value != -1; }) &&
        llvm::all_of(finalStride, [](int64_t value) { return value != -1; });

    int layoutTag = 0;
    auto elemBytes = kPTOWordBytes;
    if (elemTypeStr.find("half") != std::string::npos ||
        elemTypeStr.find("f16") != std::string::npos ||
        elemTypeStr.find("bf16") != std::string::npos) {
      elemBytes = kPTOHalfWordBytes;
    } else if (elemTypeStr.find("double") != std::string::npos ||
               elemTypeStr.find("f64") != std::string::npos) {
      elemBytes = kPTODoubleWordBytes;
    }

    if (allStatic) {
      if (finalShape[kNumber2] == kNumber16 &&
          finalShape[2] * finalShape[3] * elemBytes == kFractalSize512 &&
          finalStride[4] == 1 && finalStride[3] == finalShape[4]) {
        layoutTag = kNumber2;
      } else {
        bool isCol = finalStride[0] == 1;
        for (int i = 0; i < static_cast<int>(kNumber4); ++i)
          isCol = isCol && (finalStride[i + 1] ==
                            multiplyOrDynamic(finalStride[i], finalShape[i]));
        if (isCol)
          layoutTag = 1;
      }
    }

    if (layoutTag == 1)
      return "pto::Layout::DN";
    if (layoutTag == static_cast<int32_t>(kNumber2))
      return "pto::Layout::NZ";
    return "pto::Layout::ND";
  }

  static Value buildSubviewShapeValue(ConversionPatternRewriter &rewriter,
                                      Location loc, Type shapeTypeOpaque,
                                      StringRef shapeCppType,
                                      ValueRange shapeArgs) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, shapeTypeOpaque, shapeCppType,
                                     ArrayAttr{}, ArrayAttr{}, shapeArgs)
        .getResult(0);
  }

  static Value buildSubviewStrideValue(ConversionPatternRewriter &rewriter,
                                       Location loc, Type strideTypeOpaque,
                                       StringRef strideCppType,
                                       ValueRange strideCtorArgs) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, strideTypeOpaque, strideCppType,
                                     ArrayAttr{}, ArrayAttr{}, strideCtorArgs)
        .getResult(0);
  }

  LogicalResult replaceGlobalSubview(
      memref::SubViewOp op, OpAdaptor adaptor, Value newPtr,
      const SubviewStrideShapeInfo &strideShapeInfo, StringRef elemTypeStr,
      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    std::string shapeParams = joinIntTemplateParams(strideShapeInfo.finalShape);
    std::string strideParams = joinIntTemplateParams(strideShapeInfo.finalStride);
    std::string shapeCppType = "pto::Shape<" + shapeParams + ">";
    std::string strideCppType = "pto::Stride<" + strideParams + ">";
    std::string layoutEnum = resolveSubviewLayout(op, elemTypeStr,
                                                  strideShapeInfo.finalShape,
                                                  strideShapeInfo.finalStride);
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeCppType);
    SmallVector<Value> shapeArgs(adaptor.getSizes().begin(), adaptor.getSizes().end());
    Value shapeValue = buildSubviewShapeValue(rewriter, loc, shapeTypeOpaque,
                                              shapeCppType, shapeArgs);
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideCppType);
    SmallVector<Value> strideCtorArgs;
    for (int i = 0; i < static_cast<int>(kPTOPaddedTensorRank5D); ++i) {
      if (strideShapeInfo.finalStride[i] == -1)
        strideCtorArgs.push_back(strideShapeInfo.finalStrideValues[i]);
    }
    Value strideValue = buildSubviewStrideValue(rewriter, loc, strideTypeOpaque,
                                                strideCppType, strideCtorArgs);
    std::string gtCppType = "GlobalTensor<" + elemTypeStr.str() + ", " +
                            shapeCppType + ", " + strideCppType + ", " +
                            layoutEnum + ">";
    auto gtType = emitc::OpaqueType::get(ctx, gtCppType);
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, gtType, gtCppType, ArrayAttr{}, ArrayAttr{},
        ValueRange{newPtr, shapeValue, strideValue});
    return success();
  }

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());
    SourceStrideInfo sourceStrideInfo = getSourceStrideInfo(op, srcType, rewriter);
    OffsetComputation offsetInfo =
        buildSubviewOffset(op, adaptor, sourceStrideInfo.sourceStrides, rewriter);

    Value sourcePtr = adaptor.getSource();
    Value tileCandidate = peelCastLikeValue(sourcePtr);
    sourcePtr = materializeSubviewSourcePointer(sourcePtr, tileCandidate, srcType,
                                                loc, rewriter);
    Value newPtr = buildSubviewResultPointer(op, srcType, sourcePtr,
                                             offsetInfo.totalOffset, rewriter);
    if (succeeded(replaceNonGlobalSubview(op, srcType, newPtr, rewriter)))
      return success();

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    std::string elemTypeStr = getElemTypeStringForGT(resTy.getElementType());
    SubviewStrideShapeInfo strideShapeInfo = buildSubviewStrideShapeInfo(
        op, adaptor, offsetInfo.u32Ty, sourceStrideInfo.sourceStrides, rewriter);
    return replaceGlobalSubview(op, adaptor, newPtr, strideShapeInfo, elemTypeStr,
                                rewriter);
  }
};

} // namespace

void populatePTOToEmitCSubviewPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
}

} // namespace mlir::pto
