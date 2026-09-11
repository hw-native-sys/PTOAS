// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ScalarMiscOps.cpp - ScalarMisc ScalarMiscOps op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOMScatterToMSCATTER : public OpConversionPattern<pto::MScatterOp> {
  using OpConversionPattern<pto::MScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    // MSCATTER is a template intrinsic that accepts the concrete descriptor
    // directly, so peel any type-converter materialization bridge and feed the
    // producing value (static-stride GlobalTensor). See MGATHER above / #1165.
    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value mem = peelUnrealized(adaptor.getMem());
    auto coalesceAttr =
        dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
    auto scatterAtomicAttr =
        dyn_cast_or_null<pto::ScatterAtomicOpAttr>(
            op.getProperties().scatterAtomicOp);
    auto scatterOobAttr = dyn_cast_or_null<pto::ScatterOOBAttr>(
        op.getProperties().scatterOob);
    auto scatterConflictAttr =
        dyn_cast_or_null<pto::ScatterConflictAttr>(
            op.getProperties().scatterConflict);
    pto::ScatterAtomicOp scatterAtomicOp =
        scatterAtomicAttr ? scatterAtomicAttr.getValue()
                          : pto::ScatterAtomicOp::None;
    pto::ScatterOOB scatterOob =
        scatterOobAttr ? scatterOobAttr.getValue()
                       : pto::ScatterOOB::Undefined;

    Value memArg = mem;

    SmallVector<Attribute, 4> templateArgVec;
    if (coalesceAttr) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, coalesceTok(coalesceAttr.getValue())));
      if (scatterConflictAttr) {
        templateArgVec.push_back(emitc::OpaqueAttr::get(
            ctx, scatterAtomicTok(scatterAtomicOp)));
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, scatterOobTok(scatterOob)));
        templateArgVec.push_back(emitc::OpaqueAttr::get(
            ctx, scatterConflictTok(scatterConflictAttr.getValue())));
      } else if (scatterAtomicOp != pto::ScatterAtomicOp::None ||
                 scatterOob != pto::ScatterOOB::Undefined) {
        templateArgVec.push_back(emitc::OpaqueAttr::get(
            ctx, scatterAtomicTok(scatterAtomicOp)));
        if (scatterOob != pto::ScatterOOB::Undefined)
          templateArgVec.push_back(
              emitc::OpaqueAttr::get(ctx, scatterOobTok(scatterOob)));
      }
    }
    ArrayAttr templateArgs =
        templateArgVec.empty() ? ArrayAttr{} : rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MSCATTER",
        ArrayAttr{}, templateArgs,
        ValueRange{memArg, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetValToSETVAL : public OpConversionPattern<pto::TSetValOp> {
  using OpConversionPattern<pto::TSetValOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSetValOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = adaptor.getDst();
    Value val = adaptor.getVal();

    // ---- offset: SSA index operand ----
    Value offset = adaptor.getOffset();

    // Emit a marker call and let the ptoas post-processing step lower it to
    // the corresponding tile setter.
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALUE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, offset, val});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGetValToGETVAL : public OpConversionPattern<pto::TGetValOp> {
  using OpConversionPattern<pto::TGetValOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGetValOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();

    // ---- offset: SSA index operand ----
    Value offset = adaptor.getOffset();

    // Emit a marker call and let the ptoas post-processing step lower it to
    // the corresponding tile getter.
    Type dstTy = getTypeConverter()->convertType(op.getDst().getType());
    if (!dstTy)
      return failure();
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(),
        TypeRange{dstTy},
        "PTOAS__TILE_GET_VALUE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{src, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOHistogramToEmitC : public OpConversionPattern<pto::THistogramOp> {
  using OpConversionPattern<pto::THistogramOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::THistogramOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value idx = adaptor.getIdx();
    Value dst = adaptor.getDst();

    StringRef histByte = "HistByte::BYTE_1";
    int64_t byte = 1;
    auto byteAttr = op.getByteAttr();
    if (byteAttr)
      byte = getIntegerAttrSignedValue(byteAttr);
    if (auto legacyIsMSB = op->getAttrOfType<BoolAttr>("isMSB")) {
      int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
      if (byteAttr && byte != legacyByte)
        return rewriter.notifyMatchFailure(
            op, "conflicting 'byte' and legacy 'isMSB' attributes");
      byte = legacyByte;
    }
    switch (byte) {
    case 0:
      histByte = "HistByte::BYTE_0";
      break;
    case 1:
      histByte = "HistByte::BYTE_1";
      break;
    case 2:
      histByte = "HistByte::BYTE_2";
      break;
    case 3:
      histByte = "HistByte::BYTE_3";
      break;
    default:
      return rewriter.notifyMatchFailure(op, "expected byte to be in range [0, 3]");
    }

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, histByte)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "THISTOGRAM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGetScaleAddrToEmitC
    : public OpConversionPattern<pto::TGetScaleAddrOp> {
  using OpConversionPattern<pto::TGetScaleAddrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGetScaleAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    std::optional<pto::AddressSpace> srcSpace;
    Type srcElemTy;
    if (auto srcTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(srcTy.getMemorySpace()))
        srcSpace = asAttr.getAddressSpace();
      srcElemTy = srcTy.getElementType();
    } else if (auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType())) {
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(srcTy.getMemorySpace()))
        srcSpace = asAttr.getAddressSpace();
      srcElemTy = srcTy.getElementType();
    }
    if (!srcSpace || !srcElemTy)
      return rewriter.notifyMatchFailure(
          op, "failed to resolve src address space or element type");

    std::string srcElemTok = getEmitCScalarTypeToken(srcElemTy);
    auto isEmitCTileLike = [](Type ty) {
      auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
      return opaqueTy &&
             (opaqueTy.getValue().contains("Tile<") ||
              opaqueTy.getValue().contains("ConvTile<"));
    };
    Type convertedSrcTy = getTypeConverter()->convertType(op.getSrc().getType());
    if (!convertedSrcTy || !isEmitCTileLike(convertedSrcTy))
      return rewriter.notifyMatchFailure(op,
                                         "expected src to lower to a tile-like value");
    Value rawPtr = src;
    rawPtr = materializeTileDataValue(rewriter, loc, src, *srcSpace, srcElemTok);
    if (tileDataReturnsIntegralAddress(*srcSpace))
      rawPtr = materializeAddressAsPointer(rewriter, loc, rawPtr, *srcSpace,
                                           srcElemTok);

    auto u64Ty = emitc::OpaqueType::get(rewriter.getContext(), "uint64_t");
    auto scaleAddr = rewriter
                         .create<emitc::CallOpaqueOp>(
                             loc, TypeRange{u64Ty}, "GetScaleAddr",
                             /*args=*/ArrayAttr{},
                             /*templateArgs=*/ArrayAttr{},
                             /*operands=*/ValueRange{rawPtr})
                         .getResult(0);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, scaleAddr});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetValidShapeToEmitC : public OpConversionPattern<pto::SetValidShapeOp> {
  using OpConversionPattern<pto::SetValidShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetValidShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    

    Value src = peelAllConversionCasts(adaptor.getSource());
    Value row = adaptor.getValidRow();
    Value col = adaptor.getValidCol();

    if (!isTileLikeValue(src))
      return rewriter.notifyMatchFailure(
          op, "set_validshape source must lower to a tile-like value");

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALIDSHAPE", ArrayAttr{},
        ArrayAttr{}, ValueRange{src, row, col});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGetValidShapeToEmitC
    : public OpConversionPattern<pto::GetValidShapeOp> {
  using OpConversionPattern<pto::GetValidShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GetValidShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    

    Value src = peelAllConversionCasts(adaptor.getSource());
    if (!isTileLikeValue(src))
      return rewriter.notifyMatchFailure(
          op, "get_validshape source must lower to a tile-like value");

    auto resultTy = getTypeConverter()->convertType(rewriter.getIndexType());
    if (!resultTy)
      return failure();
    Location rowLoc = getIndexedNameHintLoc(op.getLoc(), 0);
    Location colLoc = getIndexedNameHintLoc(op.getLoc(), 1);

    Value row = rewriter
                    .create<emitc::CallOpaqueOp>(
                        rowLoc, resultTy,
                        "PTOAS__TILE_GET_VALID_ROW", ArrayAttr{},
                        ArrayAttr{}, ValueRange{src})
                    .getResult(0);
    Value col = rewriter
                    .create<emitc::CallOpaqueOp>(
                        colLoc, resultTy,
                        "PTOAS__TILE_GET_VALID_COL", ArrayAttr{},
                        ArrayAttr{}, ValueRange{src})
                    .getResult(0);
    rewriter.replaceOp(op, ValueRange{row, col});
    return success();
  }
};

struct PTOTAssignToEmitC : public OpConversionPattern<pto::TAssignOp> {
  using OpConversionPattern<pto::TAssignOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAssignOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value tile = peelAllConversionCasts(adaptor.getTile());
    if (!isTileLikeValue(tile))
      return rewriter.notifyMatchFailure(
          op, "tassign tile must lower to a tile-like value");

    Value addr = coerceToU64Address(rewriter, loc, adaptor.getAddr(),
                                     emitc::OpaqueType::get(ctx, "uint64_t"));

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{tile, addr});
    rewriter.replaceOp(op, tile);
    return success();
  }
};

struct PTOCmoCacheInvalidToEmitC
    : public OpConversionPattern<pto::CmoCacheInvalidOp> {
  using OpConversionPattern<pto::CmoCacheInvalidOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::CmoCacheInvalidOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->hasAttr(kCmoCacheInvalidSkipLoweringAttrName)) {
      rewriter.eraseOp(op);
      return success();
    }
    if (!isGmCmoSpace(op.getSpace().getAddressSpace()))
      return rewriter.notifyMatchFailure(op, "unsupported CMO invalidate space");
    if (op.getAddr()) {
      Value addr = peelGlobalTensorConversionBridge(adaptor.getAddr());
      addr = materializeGlobalTensorDataPointer(
          rewriter, op.getLoc(), addr, op.getAddr().getType());
      emitInvalidateGmCacheSingleLine(rewriter, op.getLoc(), addr);
    } else {
      emitInvalidateGmCacheAll(rewriter, op.getLoc());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void populateScalarMiscScalarMiscOpsPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOCmoCacheInvalidToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetValToSETVAL>(typeConverter, ctx);
  patterns.add<PTOGetValToGETVAL>(typeConverter, ctx);
  patterns.add<PTOSetValidShapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetValidShapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAssignToEmitC>(typeConverter, ctx);
  patterns.add<PTOHistogramToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetScaleAddrToEmitC>(typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
