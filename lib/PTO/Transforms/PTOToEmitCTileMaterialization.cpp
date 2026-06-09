// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTileMaterialization.cpp ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTO.h"

#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";
constexpr size_t kTileRank2D = 2;

static pto::AddressSpace getMemRefAddressSpace(MemRefType memRefTy) {
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(memRefTy.getMemorySpace())) {
    return asAttr.getAddressSpace();
  }
  return pto::AddressSpace::GM;
}

static int32_t getTileFractalSize(pto::TileBufConfigAttr configAttr) {
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    return frAttr.getInt();
  return kFractalSize512;
}

static Value createEmitCTileValue(Location loc, Type convertedTy,
                                  llvm::StringRef tileTypeString,
                                  ArrayRef<Value> constructorArgs,
                                  ConversionPatternRewriter &rewriter) {
  if (!constructorArgs.empty()) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, convertedTy, tileTypeString,
                                     ArrayAttr{}, ArrayAttr{},
                                     ValueRange(constructorArgs))
        .getResult(0);
  }
  return rewriter
      .create<emitc::VariableOp>(loc, convertedTy,
                                 emitc::OpaqueAttr::get(rewriter.getContext(), ""))
      .getResult();
}

static void emitTileAssign(Location loc, Value tile, Value addr,
                           ConversionPatternRewriter &rewriter) {
  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TASSIGN", ArrayAttr{}, ArrayAttr{},
      ValueRange{tile, castAddressToU64(rewriter, loc, peelUnrealized(addr))});
}

// =============================================================================
// 2. BindTileOp Lowering (FIX: Trace back to physical address)
// =============================================================================
struct PTOBindTileToEmitC : public OpConversionPattern<pto::BindTileOp> {
  using OpConversionPattern::OpConversionPattern;

  struct ValidShapeSpec {
    std::string vrowTok;
    std::string vcolTok;
    bool useConstructor = false;
    SmallVector<Value> constructorArgs;
  };

  struct TileBuildSpec {
    std::string tileTypeStr;
    bool useConstructor = false;
    SmallVector<Value> constructorArgs;
  };

  static bool getIndexConst(Value v, int64_t &out) {
    if (!v)
      return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  static void appendForcedDynamicValidArg(ValidShapeSpec &result,
                                          ConversionPatternRewriter &rewriter,
                                          Location loc, Type elemTy,
                                          pto::BLayout blayout, Value emittedDim,
                                          int64_t fallbackDim, int dimIdx) {
    result.constructorArgs.push_back(buildTileCtorDimValue(
        rewriter, loc,
        scalePackedTileDynamicDim(rewriter, loc, elemTy, blayout, emittedDim,
                                  dimIdx),
        renderTileTemplateDim(fallbackDim, elemTy, blayout, dimIdx)));
  }

  static void configureRegularValidDim(std::string &token, bool &isDynamic,
                                       bool &useConstructor, Value rawDim,
                                       bool isConst, int64_t constDim,
                                       int64_t fallbackDim, Type elemTy,
                                       pto::BLayout blayout, int dimIdx) {
    if (isConst) {
      token = std::to_string(
          renderTileTemplateDim(constDim, elemTy, blayout, dimIdx));
      return;
    }
    if (rawDim) {
      token = "-1";
      isDynamic = true;
      useConstructor = true;
      return;
    }
    token = std::to_string(
        renderTileTemplateDim(fallbackDim, elemTy, blayout, dimIdx));
  }

  static void appendRegularDynamicValidArg(ValidShapeSpec &result,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc, Type elemTy,
                                           pto::BLayout blayout, Value emittedDim,
                                           int dimIdx, bool isDynamic) {
    if (!isDynamic || !emittedDim)
      return;
    result.constructorArgs.push_back(scalePackedTileDynamicDim(
        rewriter, loc, elemTy, blayout, emittedDim, dimIdx));
  }

  ValidShapeSpec buildValidShapeSpec(pto::BindTileOp op, OpAdaptor adaptor,
                                     Type elemTy, pto::BLayout blayout,
                                     int64_t rows, int64_t cols,
                                     ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    ValidShapeSpec result;
    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();
    Value vRowEmitC = adaptor.getValidRow();
    Value vColEmitC = adaptor.getValidCol();
    bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
    int64_t cRow = 0, cCol = 0;
    bool rowIsConst = vRow && getIndexConst(vRow, cRow);
    bool colIsConst = vCol && getIndexConst(vCol, cCol);
    bool rowIsDynamic = false;
    bool colIsDynamic = false;
    if (forceDynamicValid) {
      result.vrowTok = "-1";
      result.vcolTok = "-1";
      result.useConstructor = true;
      appendForcedDynamicValidArg(result, rewriter, loc, elemTy, blayout,
                                  vRowEmitC, rowIsConst ? cRow : rows, 0);
      appendForcedDynamicValidArg(result, rewriter, loc, elemTy, blayout,
                                  vColEmitC, colIsConst ? cCol : cols, 1);
      return result;
    }

    configureRegularValidDim(result.vrowTok, rowIsDynamic, result.useConstructor,
                             vRow, rowIsConst, cRow, rows, elemTy, blayout, 0);
    configureRegularValidDim(result.vcolTok, colIsDynamic, result.useConstructor,
                             vCol, colIsConst, cCol, cols, elemTy, blayout, 1);
    if (result.useConstructor) {
      appendRegularDynamicValidArg(result, rewriter, loc, elemTy, blayout,
                                   vRowEmitC, 0, rowIsDynamic);
      appendRegularDynamicValidArg(result, rewriter, loc, elemTy, blayout,
                                   vColEmitC, 1, colIsDynamic);
    }
    return result;
  }

  FailureOr<TileBuildSpec> buildTileSpec(pto::BindTileOp op, OpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter) const {
    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy || resMrTy.getRank() < static_cast<int64_t>(kTileRank2D))
      return failure();
    int64_t rows = resMrTy.getDimSize(0);
    int64_t cols = resMrTy.getDimSize(1);
    if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
      return failure();

    auto configAttr = op.getConfigAttr();
    Type elemTy = resMrTy.getElementType();
    Type emitElemTy = getTypeConverter()->convertType(elemTy);
    auto emitElemOpaque = dyn_cast_or_null<emitc::OpaqueType>(emitElemTy);
    if (!emitElemOpaque)
      return failure();

    pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
    ValidShapeSpec validSpec =
        buildValidShapeSpec(op, adaptor, elemTy, blayout, rows, cols, rewriter);
    std::string tileTypeStr =
        "Tile<" + getTileRoleToken(resMrTy.getMemorySpace()) + ", " +
        emitElemOpaque.getValue().str() + ", " +
        std::to_string(renderTileTemplateDim(rows, elemTy, blayout, 0)) + ", " +
        std::to_string(renderTileTemplateDim(cols, elemTy, blayout, 1)) + ", " +
        getTileBufBLayoutToken(configAttr) + ", " + validSpec.vrowTok + ", " +
        validSpec.vcolTok + ", " + getTileBufSLayoutToken(configAttr) + ", " +
        std::to_string(getTileFractalSize(configAttr)) + ", " +
        getTileBufPadToken(configAttr) + ", " +
        getTileBufCompactToken(configAttr) + ">";
    return TileBuildSpec{tileTypeStr, validSpec.useConstructor,
                         validSpec.constructorArgs};
  }

  static Value buildTileValue(const TileBuildSpec &spec, Location loc,
                              MLIRContext *ctx,
                              ConversionPatternRewriter &rewriter,
                              bool forceDeclaration = false) {
    auto tileType = emitc::OpaqueType::get(ctx, spec.tileTypeStr);
    if (spec.useConstructor && !forceDeclaration) {
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, tileType, spec.tileTypeStr,
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange(spec.constructorArgs))
          .getResult(0);
    }
    return rewriter
        .create<emitc::VariableOp>(loc, tileType, emitc::OpaqueAttr::get(ctx, ""))
        .getResult();
  }

  FailureOr<Value> buildIntegralAddress(pto::BindTileOp op, Value sourceValue,
                                        ConversionPatternRewriter &rewriter) const {
    auto srcMrTy = dyn_cast<MemRefType>(op.getSource().getType());
    if (!srcMrTy)
      return failure();
    Value rawPtr = sourceValue;
    if (isEmitCTileLikeValue(sourceValue)) {
      std::string elemTok = getEmitCScalarTypeToken(srcMrTy.getElementType());
      pto::AddressSpace as = getMemRefAddressSpace(srcMrTy);
      rawPtr =
          materializeTileDataValue(rewriter, op.getLoc(), sourceValue, as, elemTok);
    }
    return castAddressToU64(rewriter, op.getLoc(), rawPtr);
  }

  LogicalResult rewriteTileAssignment(pto::BindTileOp op, Value tileCandidate,
                                      const TileBuildSpec &tileSpec,
                                      ConversionPatternRewriter &rewriter,
                                      bool forceDeclaration = false) const {
    Value dstTile = buildTileValue(tileSpec, op.getLoc(), rewriter.getContext(),
                                   rewriter, forceDeclaration);
    FailureOr<Value> addr = buildIntegralAddress(op, tileCandidate, rewriter);
    if (failed(addr))
      return failure();
    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{dstTile, *addr});
    rewriter.replaceOp(op, dstTile);
    return success();
  }

  LogicalResult rewriteDeclaredTile(pto::BindTileOp op,
                                    const TileBuildSpec &tileSpec,
                                    ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOp(
        op, buildTileValue(tileSpec, op.getLoc(), rewriter.getContext(), rewriter));
    return success();
  }

  LogicalResult rewriteReshapeTile(pto::BindTileOp op, Value tileCandidate,
                                   const TileBuildSpec &tileSpec,
                                   ConversionPatternRewriter &rewriter) const {
    Value dstTile = buildTileValue(tileSpec, op.getLoc(), rewriter.getContext(),
                                   rewriter, /*forceDeclaration=*/true);
    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TRESHAPE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{dstTile, tileCandidate});
    rewriter.replaceOp(op, dstTile);
    return success();
  }

  static bool canReuseGenericTile(Value tileCandidate,
                                  const TileBuildSpec &tileSpec) {
    if (tileSpec.useConstructor)
      return false;
    auto srcTy = dyn_cast<emitc::OpaqueType>(tileCandidate.getType());
    return srcTy && srcTy.getValue() == tileSpec.tileTypeStr;
  }

  LogicalResult rewriteFallbackPointerCast(pto::BindTileOp op, OpAdaptor adaptor,
                                           StringAttr viewSemantics,
                                           ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> physAddrs;
    Value source = op.getSource();
    while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>())
      source = castOp.getOperand(0);
    if (auto upstreamCast = source.getDefiningOp<pto::PointerCastOp>()) {
      auto upstreamOperands = upstreamCast.getAddrs();
      physAddrs.append(upstreamOperands.begin(), upstreamOperands.end());
    } else {
      physAddrs.push_back(adaptor.getSource());
    }

    auto newCast = rewriter.create<pto::PointerCastOp>(
        op.getLoc(), op.getType(), physAddrs,
        op.getValidRow() ? op.getValidRow() : Value(),
        op.getValidCol() ? op.getValidCol() : Value(), op.getConfigAttr());
    if (viewSemantics)
      newCast->setAttr("pto.view_semantics", viewSemantics);
    if (op->hasAttr(kForceDynamicValidShapeAttrName))
      newCast->setAttr(kForceDynamicValidShapeAttrName,
                       op->getAttr(kForceDynamicValidShapeAttrName));
    rewriter.replaceOp(op, newCast.getResult());
    return success();
  }

  LogicalResult matchAndRewrite(pto::BindTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool isSubView = viewSemantics && viewSemantics.getValue() == "subview";
    FailureOr<TileBuildSpec> tileSpec = buildTileSpec(op, adaptor, rewriter);
    if (failed(tileSpec))
      return failure();
    if (op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>())
      return rewriteDeclaredTile(op, *tileSpec, rewriter);

    Value tileCandidate = peelEmitCCasts(adaptor.getSource());
    if (viewSemantics && viewSemantics.getValue() == "bitcast" &&
        isEmitCTileLikeValue(tileCandidate)) {
      return rewriteTileAssignment(op, tileCandidate, *tileSpec, rewriter);
    }

    if (viewSemantics && viewSemantics.getValue() == "treshape" &&
        isEmitCTileLikeValue(tileCandidate))
      return rewriteReshapeTile(op, tileCandidate, *tileSpec, rewriter);

    // Subview origins are kept distinct from generic tile rebinding
    // even when source/destination C++ tile types match, subview may carry
    // shifted base address semantics and should materialize a fresh handle.
    if (isSubView)
      return rewriteTileAssignment(op, tileCandidate, *tileSpec, rewriter);

    // Generic tile-to-tile rebind path: preserve the same backing storage and
    // rebuild a sibling tile with updated metadata/valid dims.
    if (isEmitCTileLikeValue(tileCandidate)) {
      if (canReuseGenericTile(tileCandidate, *tileSpec)) {
        rewriter.replaceOp(op, tileCandidate);
        return success();
      }
      return rewriteTileAssignment(op, tileCandidate, *tileSpec, rewriter);
    }
    return rewriteFallbackPointerCast(op, adaptor, viewSemantics, rewriter);
  }
};

struct PTOAllocTileToEmitC
    : public OpConversionPattern<pto::AllocTileOp> {
  using OpConversionPattern::OpConversionPattern;

  FailureOr<SmallVector<Value>>
  buildConstructorArgs(pto::AllocTileOp op, OpAdaptor adaptor,
                       pto::TileBufType tileTy,
                       ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> constructorArgs;
    auto validShape = tileTy.getValidShape();
    if (!llvm::any_of(validShape, [](int64_t dim) { return dim < 0; }))
      return constructorArgs;

    Type elemTy = tileTy.getElementType();
    pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
    if (validShape.size() > 0 && validShape[0] < 0) {
      Value validRow = adaptor.getValidRow();
      if (!validRow)
        return failure();
      constructorArgs.push_back(scalePackedTileDynamicDim(
          rewriter, op.getLoc(), elemTy, blayout, peelUnrealized(validRow), 0));
    }
    if (validShape.size() > 1 && validShape[1] < 0) {
      Value validCol = adaptor.getValidCol();
      if (!validCol)
        return failure();
      constructorArgs.push_back(scalePackedTileDynamicDim(
          rewriter, op.getLoc(), elemTy, blayout, peelUnrealized(validCol), 1));
    }
    return constructorArgs;
  }

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

    FailureOr<SmallVector<Value>> constructorArgs =
        buildConstructorArgs(op, adaptor, tileTy, rewriter);
    if (failed(constructorArgs))
      return rewriter.notifyMatchFailure(
          op, "dynamic alloc_tile valid shape operand is missing");

    Value tile = createEmitCTileValue(loc, convertedTy, *tileTypeString,
                                      *constructorArgs, rewriter);

    Value addr = adaptor.getAddr();
    if (addr)
      emitTileAssign(loc, tile, addr, rewriter);

    rewriter.replaceOp(op, tile);
    return success();
  }
};

static FailureOr<Value>
createEmitCTileVariable(ConversionPatternRewriter &rewriter, Location loc,
                        const TypeConverter *typeConverter,
                        pto::TileBufType tileTy) {
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();

  Type convertedTy = typeConverter->convertType(tileTy);
  if (!convertedTy)
    convertedTy = emitc::OpaqueType::get(rewriter.getContext(), *tileTypeString);
  return rewriter
      .create<emitc::VariableOp>(
          loc, convertedTy, emitc::OpaqueAttr::get(rewriter.getContext(), ""))
      .getResult();
}

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

    Value src = peelEmitCCasts(adaptor.getSrc());

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

    Value src = peelEmitCCasts(adaptor.getSrc());

    pto::AddressSpace as = pto::AddressSpace::GM;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcTy.getMemorySpace()))
      as = asAttr.getAddressSpace();
    std::string elemTok = getEmitCScalarTypeToken(srcTy.getElementType());

    Value rawPtr = materializeTileDataValue(rewriter, op.getLoc(), src, as, elemTok);
    emitTileAssign(op.getLoc(), *dst, rawPtr, rewriter);
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOMaterializeTileToEmitC
    : public OpConversionPattern<pto::MaterializeTileOp> {
  using OpConversionPattern::OpConversionPattern;

  static SmallVector<Value>
  buildConstructorArgs(pto::TileBufType tileTy, OpAdaptor adaptor, Location loc,
                       bool forceDynamicValid,
                       ConversionPatternRewriter &rewriter) {
    SmallVector<Value> constructorArgs;
    pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
    Type elemTy = tileTy.getElementType();
    auto shape = tileTy.getShape();
    auto validShape = tileTy.getValidShape();
    auto fallbackDim = [shape, elemTy, blayout](int dimIdx) {
      return renderTileTemplateDim(shape[dimIdx], elemTy, blayout, dimIdx);
    };
    auto appendCtorDim = [&constructorArgs, &rewriter, loc, elemTy, blayout,
                          &fallbackDim](Value emitted, int dimIdx) {
      constructorArgs.push_back(buildTileCtorDimValue(
          rewriter, loc,
          scalePackedTileDynamicDim(rewriter, loc, elemTy, blayout, emitted,
                                    dimIdx),
          fallbackDim(dimIdx)));
    };
    if (forceDynamicValid) {
      appendCtorDim(adaptor.getValidRow(), 0);
      appendCtorDim(adaptor.getValidCol(), 1);
      return constructorArgs;
    }
    if (validShape[0] == ShapedType::kDynamic)
      appendCtorDim(adaptor.getValidRow(), 0);
    if (validShape[1] == ShapedType::kDynamic)
      appendCtorDim(adaptor.getValidCol(), 1);
    return constructorArgs;
  }

  static bool canReuseSourceTile(Value source, llvm::StringRef tileTypeString,
                                 bool isSubview, bool forceDynamicValid) {
    if (isSubview || forceDynamicValid || !isEmitCTileLikeValue(source))
      return false;
    auto srcTy = dyn_cast<emitc::OpaqueType>(source.getType());
    return srcTy && srcTy.getValue() == tileTypeString;
  }

  static Value materializeSourceAddress(pto::TileBufType tileTy, Value source,
                                        Location loc,
                                        ConversionPatternRewriter &rewriter) {
    Value rawPtr = source;
    if (isEmitCTileLikeValue(rawPtr)) {
      pto::AddressSpace as = pto::AddressSpace::GM;
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(tileTy.getMemorySpace())) {
        as = asAttr.getAddressSpace();
      }
      std::string elemTok = getEmitCScalarTypeToken(tileTy.getElementType());
      rawPtr = materializeTileDataValue(rewriter, loc, rawPtr, as, elemTok);
    }
    return castAddressToU64(rewriter, loc, rawPtr);
  }

  static LogicalResult rewriteReshape(Location loc, Value tile, Value source,
                                      Operation *op,
                                      ConversionPatternRewriter &rewriter) {
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TRESHAPE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{tile, source});
    rewriter.replaceOp(op, tile);
    return success();
  }

  LogicalResult matchAndRewrite(pto::MaterializeTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto tileTy = cast<pto::TileBufType>(op.getResult().getType());
    auto tileTypeString = getEmitCTileTypeString(tileTy);
    if (!tileTypeString)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 tile_buf handles can be materialized to EmitC");

    Type convertedTy = getTypeConverter()->convertType(tileTy);
    if (!convertedTy)
      convertedTy = emitc::OpaqueType::get(ctx, *tileTypeString);

    Value source = peelEmitCCasts(adaptor.getSource());

    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
    bool isReshape = viewSemantics && viewSemantics.getValue() == "treshape";
    bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
    bool sourceIsDeclaredTile =
        op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>();
    if (canReuseSourceTile(source, *tileTypeString, isSubview, forceDynamicValid)) {
      rewriter.replaceOp(op, source);
      return success();
    }

    SmallVector<Value> constructorArgs =
        buildConstructorArgs(tileTy, adaptor, loc, forceDynamicValid, rewriter);
    Value tile = createEmitCTileValue(loc, convertedTy, *tileTypeString,
                                      constructorArgs, rewriter);
    if (sourceIsDeclaredTile) {
      rewriter.replaceOp(op, tile);
      return success();
    }

    if (isReshape && isEmitCTileLikeValue(source))
      return rewriteReshape(loc, tile, source, op, rewriter);

    emitTileAssign(loc, tile, materializeSourceAddress(tileTy, source, loc, rewriter),
                   rewriter);
    rewriter.replaceOp(op, tile);
    return success();
  }
};


} // namespace

void populatePTOToEmitCTileMaterializationPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOAllocTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaterializeTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOBindTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOTReshapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOBitcastToEmitC>(typeConverter, ctx);
}

} // namespace mlir::pto
