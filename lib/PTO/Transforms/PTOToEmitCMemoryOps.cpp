// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCMemoryOps.cpp --------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";
constexpr unsigned kInlineCapacity5 = 5;
constexpr unsigned kInlineCapacity8 = 8;
constexpr unsigned kNumber10 = 10;

template <typename T>
using SmallVec5 = SmallVector<T, kInlineCapacity5>;
template <typename T>
using SmallVec8 = SmallVector<T, kInlineCapacity8>;

struct PointerCastConversion : public OpConversionPattern<pto::PointerCastOp> {
  using OpConversionPattern<pto::PointerCastOp>::OpConversionPattern;

  enum class TileRole { Vec, Mat, Left, Right, Acc, Bias, Scaling };

  struct PointerCastConfigStrings {
    pto::BLayout blayout = pto::BLayout::RowMajor;
    std::string layoutParams = "BLayout::RowMajor";
    std::string extraParams =
        ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null";
  };

  struct ValidShapeInfo {
    std::string vrowTok;
    std::string vcolTok;
    bool useConstructor = false;
    SmallVector<Value> constructorArgs;
  };

  static void collectUserOpsThroughCasts(Value v, SmallVectorImpl<Operation *> &out) {
    for (Operation *u : v.getUsers()) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(u)) {
        for (Value r : castOp.getResults())
          collectUserOpsThroughCasts(r, out);
        continue;
      }
      out.push_back(u);
    }
  }

  static Value peelUnrealized(Value v) {
    while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      v = castOp.getOperand(0);
    }
    return v;
  }

  static TileRole inferRoleFromAddressSpace(MemRefType memRefTy) {
    Attribute memorySpace = memRefTy.getMemorySpace();
    auto ptoAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace);
    if (!ptoAttr)
      return TileRole::Vec;
    switch (ptoAttr.getAddressSpace()) {
    case pto::AddressSpace::LEFT:
      return TileRole::Left;
    case pto::AddressSpace::RIGHT:
      return TileRole::Right;
    case pto::AddressSpace::ACC:
      return TileRole::Acc;
    case pto::AddressSpace::BIAS:
      return TileRole::Bias;
    case pto::AddressSpace::MAT:
      return TileRole::Mat;
    case pto::AddressSpace::SCALING:
      return TileRole::Scaling;
    default:
      return TileRole::Vec;
    }
  }

  static TileRole inferRoleFromUsers(pto::PointerCastOp op) {
    SmallVec8<Operation *> users;
    collectUserOpsThroughCasts(op.getResult(), users);
    for (Operation *user : users) {
      if (auto mm = dyn_cast<pto::TMatmulOp>(user)) {
        if (mm.getDst() && peelUnrealized(mm.getDst()) == op.getResult())
          return TileRole::Acc;
        if (peelUnrealized(mm.getLhs()) == op.getResult())
          return TileRole::Left;
        if (peelUnrealized(mm.getRhs()) == op.getResult())
          return TileRole::Right;
      }
      if (auto mmacc = dyn_cast<pto::TMatmulAccOp>(user)) {
        if (mmacc.getDst() && peelUnrealized(mmacc.getDst()) == op.getResult())
          return TileRole::Acc;
        if (peelUnrealized(mmacc.getAccIn()) == op.getResult())
          return TileRole::Acc;
        if (peelUnrealized(mmacc.getLhs()) == op.getResult())
          return TileRole::Left;
        if (peelUnrealized(mmacc.getRhs()) == op.getResult())
          return TileRole::Right;
      }
    }
    return TileRole::Vec;
  }

  static TileRole inferRole(pto::PointerCastOp op) {
    if (auto memRefTy = dyn_cast<MemRefType>(op.getType())) {
      TileRole role = inferRoleFromAddressSpace(memRefTy);
      if (role != TileRole::Vec)
        return role;
    }
    return inferRoleFromUsers(op);
  }

  // [新增] 辅助函数：判断 Value 是否源自 arith.constant
  static bool isConstant(Value v, int64_t &outVal) {
    if (!v) return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
       if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
           outVal = attr.getInt();
           return true;
       }
    }
    return false;
  }

  static llvm::StringRef getRoleToken(TileRole role) {
    switch (role) {
    case TileRole::Left:
      return "TileType::Left";
    case TileRole::Right:
      return "TileType::Right";
    case TileRole::Acc:
      return "TileType::Acc";
    case TileRole::Bias:
      return "TileType::Bias";
    case TileRole::Mat:
      return "TileType::Mat";
    case TileRole::Scaling:
      return "TileType::Scaling";
    case TileRole::Vec:
      return "TileType::Vec";
    }
    return "TileType::Vec";
  }

  static std::string buildDimToken(int64_t dim, llvm::StringRef symbol,
                                   Type elemType, pto::BLayout blayout,
                                   int dimIdx) {
    if (dim == ShapedType::kDynamic)
      return symbol.str();
    return std::to_string(
        renderTileTemplateDim(dim, elemType, blayout, dimIdx));
  }

  static std::string buildDimString(TileRole role, ArrayRef<int64_t> shape,
                                    Type elemType, pto::BLayout blayout) {
    if (role == TileRole::Left) {
      return buildDimToken(shape[0], "M", elemType, blayout, 0) + ", " +
             buildDimToken(shape[1], "K", elemType, blayout, 1);
    }
    if (role == TileRole::Right) {
      return buildDimToken(shape[0], "K", elemType, blayout, 0) + ", " +
             buildDimToken(shape[1], "N", elemType, blayout, 1);
    }
    if (role == TileRole::Bias)
      return "1, " + buildDimToken(shape[1], "N", elemType, blayout, 1);
    return buildDimToken(shape[0], "M", elemType, blayout, 0) + ", " +
           buildDimToken(shape[1], "N", elemType, blayout, 1);
  }

  static std::string getPadToken(pto::TileBufConfigAttr config) {
    return getTileBufPadToken(config);
  }

  static std::string getCompactToken(pto::TileBufConfigAttr config) {
    return getTileBufCompactToken(config);
  }

  static PointerCastConfigStrings buildConfigStrings(pto::PointerCastOp op) {
    PointerCastConfigStrings result;
    auto configOpt = op.getConfig();
    if (!configOpt)
      return result;

    auto config = *configOpt;
    int32_t blVal = 0;
    if (auto attr = dyn_cast<BLayoutAttr>(config.getBLayout()))
      blVal = static_cast<int32_t>(attr.getValue());
    if (blVal == 1) {
      result.layoutParams = "BLayout::ColMajor";
      result.blayout = pto::BLayout::ColMajor;
    }

    int32_t slVal = 0;
    if (auto attr = dyn_cast<SLayoutAttr>(config.getSLayout()))
      slVal = static_cast<int32_t>(attr.getValue());
    std::string slStr =
        slVal == 1 ? "SLayout::RowMajor"
                   : (slVal == 2 ? "SLayout::ColMajor" : "SLayout::NoneBox");

    int32_t fractal = 0;
    if (auto attr = dyn_cast<IntegerAttr>(config.getSFractalSize()))
      fractal = attr.getInt();
    result.extraParams = ", " + slStr + ", " + std::to_string(fractal) + ", " +
                         getPadToken(config) + ", " + getCompactToken(config);
    return result;
  }

  static Value maybeScaleDynamicValid(ConversionPatternRewriter &rewriter,
                                      Location loc, Type elemType,
                                      pto::BLayout blayout, Value emitted,
                                      int dimIdx) {
    return scalePackedTileDynamicDim(rewriter, loc, elemType, blayout, emitted,
                                     dimIdx);
  }

  static ValidShapeInfo buildForcedDynamicValidShapeInfo(
      ConversionPatternRewriter &rewriter, Location loc, Type elemType,
      pto::BLayout blayout, ArrayRef<int64_t> shape, Value vRowEmitC,
      Value vColEmitC, bool rowIsConst, bool colIsConst, int64_t cRow,
      int64_t cCol) {
    ValidShapeInfo result;
    result.vrowTok = "-1";
    result.vcolTok = "-1";
    result.useConstructor = true;
    result.constructorArgs.push_back(buildTileCtorDimValue(
        rewriter, loc,
        maybeScaleDynamicValid(rewriter, loc, elemType, blayout, vRowEmitC, 0),
        renderTileTemplateDim(rowIsConst ? cRow : shape[0], elemType, blayout, 0)));
    result.constructorArgs.push_back(buildTileCtorDimValue(
        rewriter, loc,
        maybeScaleDynamicValid(rewriter, loc, elemType, blayout, vColEmitC, 1),
        renderTileTemplateDim(colIsConst ? cCol : shape[1], elemType, blayout, 1)));
    return result;
  }

  static void assignRegularValidDim(ValidShapeInfo &result, std::string &targetTok,
                                    bool &isDynamic, Value originalValue,
                                    bool isConst, int64_t constValue,
                                    int64_t staticShapeValue, Type elemType,
                                    pto::BLayout blayout, int dimIdx) {
    if (isConst) {
      targetTok = std::to_string(
          renderTileTemplateDim(constValue, elemType, blayout, dimIdx));
      return;
    }
    if (originalValue) {
      targetTok = "-1";
      isDynamic = true;
      result.useConstructor = true;
      return;
    }
    targetTok = std::to_string(
        renderTileTemplateDim(staticShapeValue, elemType, blayout, dimIdx));
  }

  static ValidShapeInfo buildValidShapeInfo(
      pto::PointerCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, Location loc, Type elemType,
      pto::BLayout blayout, ArrayRef<int64_t> shape) {
    ValidShapeInfo result;
    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();
    Value vRowEmitC = adaptor.getValidRow();
    Value vColEmitC = adaptor.getValidCol();
    bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);

    int64_t cRow = 0;
    int64_t cCol = 0;
    bool rowIsConst = vRow && isConstant(vRow, cRow);
    bool colIsConst = vCol && isConstant(vCol, cCol);
    bool rowIsDynamic = false;
    bool colIsDynamic = false;
    if (forceDynamicValid)
      return buildForcedDynamicValidShapeInfo(
          rewriter, loc, elemType, blayout, shape, vRowEmitC, vColEmitC,
          rowIsConst, colIsConst, cRow, cCol);

    assignRegularValidDim(result, result.vrowTok, rowIsDynamic, vRow, rowIsConst,
                          cRow, shape[0], elemType, blayout, 0);
    assignRegularValidDim(result, result.vcolTok, colIsDynamic, vCol, colIsConst,
                          cCol, shape[1], elemType, blayout, 1);
    if (result.useConstructor) {
      if (rowIsDynamic && vRowEmitC)
        result.constructorArgs.push_back(maybeScaleDynamicValid(
            rewriter, loc, elemType, blayout, vRowEmitC, 0));
      if (colIsDynamic && vColEmitC)
        result.constructorArgs.push_back(maybeScaleDynamicValid(
            rewriter, loc, elemType, blayout, vColEmitC, 1));
    }
    return result;
  }

  static Value buildTileValue(ConversionPatternRewriter &rewriter, Location loc,
                              MLIRContext *ctx, Type tileType,
                              llvm::StringRef tileTypeStr,
                              const ValidShapeInfo &validInfo) {
    if (validInfo.useConstructor) {
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, tileType, tileTypeStr, ArrayAttr{},
                                       ArrayAttr{},
                                       ValueRange(validInfo.constructorArgs))
          .getResult(0);
    }
    return rewriter
        .create<emitc::VariableOp>(loc, tileType, emitc::OpaqueAttr::get(ctx, ""))
        .getResult();
  }

  static Value castAddressToIntegral(ConversionPatternRewriter &rewriter,
                                     Location loc, Value addr) {
    if (!isSetFFTsPointerLikeType(addr.getType()))
      return addr;
    return castAddressToU64(rewriter, loc, addr);
  }

  LogicalResult matchAndRewrite(pto::PointerCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto selfType = mlir::cast<MemRefType>(op.getType());
    ArrayRef<int64_t> shape = selfType.getShape();
    Type elemType = selfType.getElementType();
    TileRole role = inferRole(op);
    std::string elemTypeStr = getEmitCScalarTypeToken(elemType);
    PointerCastConfigStrings configStrings = buildConfigStrings(op);
    std::string dimStr =
        buildDimString(role, shape, elemType, configStrings.blayout);
    ValidShapeInfo validInfo = buildValidShapeInfo(
        op, adaptor, rewriter, loc, elemType, configStrings.blayout, shape);
    std::string tileTypeStr =
        (llvm::Twine("Tile<") + getRoleToken(role) + ", " + elemTypeStr +
         ", " + dimStr + ", " + configStrings.layoutParams + ", " +
         validInfo.vrowTok + ", " + validInfo.vcolTok +
         configStrings.extraParams + ">")
            .str();

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value resultValue =
        buildTileValue(rewriter, loc, ctx, tileType, tileTypeStr, validInfo);
    Value addr = castAddressToIntegral(rewriter, loc, adaptor.getAddrs()[0]);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{resultValue, addr});

    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_dps / pto.store_dps lowering (FIX: keep optional result)
//===----------------------------------------------------------------------===

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())


static std::optional<int64_t> getStaticIndexLikeValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static FailureOr<Value> buildGlobalTensorViewFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides = {},
    StringRef layoutEnum = "pto::Layout::ND") {
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
  SmallVec5<int64_t> shape5D;
  SmallVec5<int64_t> stride5D;
  buildGlobalTensorShapeAndStride(shape, effectiveStrides, shape5D, stride5D);

  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
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

  std::string gtTypeStr =
      getGlobalTensorTypeStringFromShapeAndStrides(elemTy, shape,
                                                   effectiveStrides,
                                                   layoutEnum);
  auto gtType = emitc::OpaqueType::get(ctx, gtTypeStr);
  auto gt = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, gtTypeStr, ArrayAttr{}, ArrayAttr{},
      ValueRange{ptr, shapeVal, strideVal});
  return gt.getResult(0);
}

static bool parseIntegerTemplateList(StringRef token, StringRef marker,
                                     SmallVectorImpl<int64_t> &values) {
  size_t pos = token.find(marker);
  if (pos == StringRef::npos)
    return false;
  pos += marker.size();
  size_t end = token.find('>', pos);
  if (end == StringRef::npos)
    return false;

  SmallVec8<StringRef> parts;
  token.slice(pos, end).split(parts, ',');
  values.clear();
  for (StringRef part : parts) {
    int64_t value = 0;
    if (part.trim().getAsInteger(kNumber10, value))
      return false;
    values.push_back(value);
  }
  return true;
}

static LogicalResult getStaticTensorViewStrides(
    Value source, Value convertedSource, pto::TensorViewType sourceType,
    SmallVectorImpl<int64_t> &strides) {
  int64_t rank = sourceType.getRank();
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
    SmallVec5<int64_t> stride5D;
    StringRef token = opaqueTy.getValue();
    if ((parseIntegerTemplateList(token, "pto::Stride<", stride5D) ||
         parseIntegerTemplateList(token, "Stride<", stride5D)) &&
        static_cast<int64_t>(stride5D.size()) >= rank) {
      strides.append(stride5D.end() - rank, stride5D.end());
      return success();
    }
  }

  auto fallback = buildRowMajorStrides(sourceType.getShape());
  strides.append(fallback.begin(), fallback.end());
  return success();
}

struct PTOPartitionViewToEmitC
    : public OpConversionPattern<mlir::pto::PartitionViewOp> {
  using OpConversionPattern<
      mlir::pto::PartitionViewOp>::OpConversionPattern;

  static LogicalResult verifyStaticPartitionView(
      mlir::pto::PartitionViewOp op, pto::TensorViewType srcTy,
      pto::PartitionTensorViewType resTy,
      ConversionPatternRewriter &rewriter) {
    if (op.getOffsets().size() != static_cast<size_t>(srcTy.getRank()) ||
        op.getSizes().size() != static_cast<size_t>(srcTy.getRank())) {
      return rewriter.notifyMatchFailure(op, "rank mismatch");
    }
    for (auto [idx, value] : llvm::enumerate(op.getSizes())) {
      auto cst = getStaticIndexLikeValue(value);
      if (!cst) {
        return rewriter.notifyMatchFailure(
            op, "globaltensor partition_view requires static sizes");
      }
      int64_t resultDim = resTy.getShape()[idx];
      if (resultDim != ShapedType::kDynamic && resultDim != *cst) {
        return rewriter.notifyMatchFailure(
            op, "partition_view static size does not match result type");
      }
    }
    return success();
  }

  static FailureOr<Value> buildDynamicPartitionOffset(
      mlir::pto::PartitionViewOp op, ConversionPatternRewriter &rewriter,
      Value data, int64_t staticLinearOffset,
      ArrayRef<std::pair<Value, int64_t>> dynamicOffsetTerms) {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    auto makeU32 = [&rewriter, loc, u32Ty](int64_t value) {
      return makeEmitCIntConstant(rewriter, loc, u32Ty, value);
    };
    auto asU32 = [&rewriter, loc, u32Ty](Value value) -> Value {
      if (value.getType() == u32Ty)
        return value;
      return rewriter.create<emitc::CastOp>(loc, u32Ty, value).getResult();
    };

    Value totalOffset = makeU32(staticLinearOffset);
    for (auto [offsetValue, stride] : dynamicOffsetTerms) {
      Value term = asU32(offsetValue);
      if (stride != 1) {
        Value strideValue = makeU32(stride);
        term = rewriter
                   .create<emitc::MulOp>(op.getLoc(), u32Ty, term, strideValue)
                   .getResult();
      }
      totalOffset = rewriter
                        .create<emitc::AddOp>(op.getLoc(), u32Ty, totalOffset,
                                              term)
                        .getResult();
    }
    return rewriter.create<emitc::AddOp>(op.getLoc(), data.getType(), data,
                                         totalOffset)
        .getResult();
  }

  static FailureOr<Value> buildPartitionViewPointer(
      mlir::pto::PartitionViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      ArrayRef<int64_t> srcStrides, pto::TensorViewType srcTy) {
    int64_t staticLinearOffset = 0;
    SmallVector<std::pair<Value, int64_t>> dynamicOffsetTerms;
    for (auto [idx, values] :
         llvm::enumerate(llvm::zip(op.getOffsets(), adaptor.getOffsets()))) {
      Value originalOffset = std::get<0>(values);
      Value convertedOffset = std::get<1>(values);
      int64_t stride = srcStrides[idx];
      if (stride == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(op,
                                           "dynamic source stride is not supported");
      }
      if (auto cst = getStaticIndexLikeValue(originalOffset)) {
        if (*cst != 0)
          staticLinearOffset += (*cst) * stride;
        continue;
      }
      dynamicOffsetTerms.push_back({convertedOffset, stride});
    }

    auto *ctx = rewriter.getContext();
    std::string elemTypeStr = getElemTypeStringForGT(srcTy.getElementType());
    auto ptrTy = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
    Value src = peelUnrealized(adaptor.getSource());
    Value data = rewriter
                     .create<emitc::CallOpaqueOp>(op.getLoc(), ptrTy,
                                                  "PTOAS__GLOBAL_TENSOR_DATA",
                                                  ArrayAttr{}, ArrayAttr{},
                                                  ValueRange{src})
                     .getResult(0);
    if (dynamicOffsetTerms.empty())
      return applyStaticMemrefOffset(rewriter, op.getLoc(), data,
                                     staticLinearOffset);
    return buildDynamicPartitionOffset(op, rewriter, data, staticLinearOffset,
                                       dynamicOffsetTerms);
  }

  LogicalResult matchAndRewrite(mlir::pto::PartitionViewOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<pto::TensorViewType>(op.getSource().getType());
    auto resTy = dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
    if (!srcTy || !resTy)
      return rewriter.notifyMatchFailure(
          op, "expected tensor_view source and partition_tensor_view result");
    if (failed(verifyStaticPartitionView(op, srcTy, resTy, rewriter)))
      return failure();

    SmallVector<int64_t> srcStrides;
    if (failed(getStaticTensorViewStrides(op.getSource(), adaptor.getSource(),
                                          srcTy, srcStrides)))
      return rewriter.notifyMatchFailure(
          op, "partition_view requires static source strides");
    FailureOr<Value> ptr =
        buildPartitionViewPointer(op, adaptor, rewriter, srcStrides, srcTy);
    if (failed(ptr))
      return failure();

    auto resultOr = buildGlobalTensorViewFromPointer(
        rewriter, op.getLoc(), *ptr, resTy.getElementType(), resTy.getShape(),
        srcStrides);
    if (failed(resultOr))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize partition GlobalTensor");

    rewriter.replaceOp(op, *resultOr);
    return success();
  }
};


} // namespace

void populatePTOToEmitCMemoryOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx) {
  patterns.add<PointerCastConversion>(typeConverter, ctx);
  patterns.add<PTOPartitionViewToEmitC>(typeConverter, ctx);
}

} // namespace mlir::pto
