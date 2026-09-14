// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TileAlloc.cpp - Tile TileAlloc op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

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

void populateTileTileAllocPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx,
                        PTOArch targetArch) {
  patterns.add<PTODeclareTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOTileBufAddrToEmitC>(typeConverter, ctx);
  patterns.add<PTOTReshapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOBitcastToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAllocToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTFreeToEmitC>(typeConverter, ctx, targetArch);
}

} // namespace pto
} // namespace mlir
