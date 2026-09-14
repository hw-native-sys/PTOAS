// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TileMisc.cpp - Tile TileMisc op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

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

void populateTileTileMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch) {
  patterns.add<PTOAllocTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOTPushToEmitC>(typeConverter, ctx, targetArch, "TPUSH");
  patterns.add<PTOTPopToEmitC>(typeConverter, ctx, targetArch, "TPOP");
}

} // namespace pto
} // namespace mlir
