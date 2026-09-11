// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCScalarMisc.cpp - sync/barrier/comm/async/declare lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

void populateScalarMiscPatternsPart2(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     MLIRContext *ctx, PTOArch targetArch);

static StringRef scatterAtomicTok(pto::ScatterAtomicOp atomic) {
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return "pto::ScatterAtomicOp::None";
  case pto::ScatterAtomicOp::Add:
    return "pto::ScatterAtomicOp::Add";
  case pto::ScatterAtomicOp::Max:
    return "pto::ScatterAtomicOp::Max";
  case pto::ScatterAtomicOp::Min:
    return "pto::ScatterAtomicOp::Min";
  }
  llvm_unreachable("unknown ScatterAtomicOp");
}

static StringRef scatterOobTok(pto::ScatterOOB mode) {
  switch (mode) {
  case pto::ScatterOOB::Undefined:
    return "pto::ScatterOOB::Undefined";
  case pto::ScatterOOB::Skip:
    return "pto::ScatterOOB::Skip";
  case pto::ScatterOOB::Clamp:
    return "pto::ScatterOOB::Clamp";
  case pto::ScatterOOB::Wrap:
    return "pto::ScatterOOB::Wrap";
  }
  llvm_unreachable("unknown ScatterOOB");
}

static StringRef scatterConflictTok(pto::ScatterConflict mode) {
  switch (mode) {
  case pto::ScatterConflict::Last:
    return "pto::ScatterConflict::Last";
  case pto::ScatterConflict::Default:
    return "pto::ScatterConflict::Default";
  }
  llvm_unreachable("unknown ScatterConflict");
}

static StringRef coalesceTok(pto::Coalesce mode) {
  switch (mode) {
  case pto::Coalesce::Row:
    return "pto::Coalesce::Row";
  case pto::Coalesce::Elem:
    return "pto::Coalesce::Elem";
  }
  llvm_unreachable("unknown Coalesce");
}

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

struct PTOTAxpyToEmitC : public OpConversionPattern<pto::TAxpyOp> {
  using OpConversionPattern<pto::TAxpyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAxpyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TAXPY",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
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

// Strip conversion casts (unrealized + emitc) down to the producing value.
static Value peelAllConversionCasts(Value v) {
  while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    v = castOp.getOperand(0);
  if (auto castOp = v.getDefiningOp<emitc::CastOp>())
    v = castOp.getOperand();
  return v;
}

static bool isTileLikeValue(Value v) {
  auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
  if (!ot)
    return false;
  StringRef s = ot.getValue();
  return s.contains("Tile<") || s.contains("ConvTile<");
}

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

//===----------------------------------------------------------------------===//
// pto.load_scalar / pto.store_scalar lowering -> ptr[offset]
//===----------------------------------------------------------------------===//




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


struct PTOPtrToIntToEmitC : public OpConversionPattern<pto::PtrToIntOp> {
  using OpConversionPattern<pto::PtrToIntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PtrToIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return failure();

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), dstTy, "reinterpret_cast", ArrayAttr{}, templateArgs,
        ValueRange{ptr});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct PTOIntToPtrToEmitC : public OpConversionPattern<pto::IntToPtrOp> {
  using OpConversionPattern<pto::IntToPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::IntToPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value addr = adaptor.getAddr();
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    Type dstElemTy = getPointerLikeElementType(op.getResult().getType());
    if (!dstElemTy)
      return failure();

    std::string castType =
        std::string("__gm__ ") + getEmitCScalarTypeToken(dstElemTy) + "*";
    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      castType)});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), dstTy, "reinterpret_cast", ArrayAttr{}, templateArgs,
        ValueRange{addr});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct PTOLoadScalarToEmitC : public OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern<pto::LoadScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    Type dstTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!dstTy)
      return failure();

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{dstTy}, "PTOAS__PTR_LOAD",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOStoreScalarToEmitC : public OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern<pto::StoreScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    Value val = adaptor.getValue();

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__PTR_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset, val});
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__SCALAR_GM_STORE_FLUSH",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tabs lowering -> TABS(dst, src)
//===----------------------------------------------------------------------===//

struct PTOTAbsToTABS : public OpConversionPattern<pto::TAbsOp> {
  using OpConversionPattern<pto::TAbsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAbsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // intrinsic: TABS(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TABS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadd lowering -> TADD(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTOTAddToTADD : public OpConversionPattern<pto::TAddOp> {
  using OpConversionPattern<pto::TAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TADD", ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOInitializeL2G2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp> {
  PTOInitializeL2G2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2G2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    Value gmAddr = adaptor.getGmAddr();
    gmAddr = materializeGlobalTensorDataPointer(
        rewriter, op.getLoc(), gmAddr, op.getGmAddr().getType());
    Value localAddr =
        op.getLocalAddr() ? adaptor.getLocalAddr() : Value();
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1) {
      c2vBuf = localAddr ? localAddr : zero;
    } else if (op.getDirMask() == 2) {
      v2cBuf = localAddr ? localAddr : zero;
    } else if (op.getDirMask() == 3) {
      if (localAddr) {
        if (!op.getPeerLocalAddr()) {
          return rewriter.notifyMatchFailure(
              op, "bidirectional l2g2l pipe requires peer local buffer");
        }
        c2vBuf = localAddr;
        v2cBuf = adaptor.getPeerLocalAddr();
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{gmAddr, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOInitializeL2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2LPipeOp> {
  PTOInitializeL2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                              PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    auto gmPtrTy =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ void"));
    Value nullGm =
        makeEmitCOpaqueConstant(rewriter, op.getLoc(), gmPtrTy, "nullptr");
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
    Value localAddr = adaptor.getLocalAddr();

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1) {
      c2vBuf = localAddr;
    } else if (op.getDirMask() == 2) {
      v2cBuf = localAddr;
    } else if (op.getDirMask() == 3) {
      c2vBuf = localAddr;
      v2cBuf = adaptor.getPeerLocalAddr();
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{nullGm, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOBuildAsyncSessionToEmitC
    : public OpConversionPattern<mlir::pto::BuildAsyncSessionOp> {
  PTOBuildAsyncSessionToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<mlir::pto::BuildAsyncSessionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(mlir::pto::BuildAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();

    auto sessionTy = dyn_cast<emitc::OpaqueType>(
        getTypeConverter()->convertType(op.getSession().getType()));
    if (!sessionTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert async session type");

    FailureOr<Value> scratchTile =
        buildAsyncScratchTileValue(rewriter, loc, op.getScratch(),
                                   adaptor.getScratch());
    if (failed(scratchTile))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize async scratch tile");

    Value workspace =
        castToGMBytePointer(rewriter, loc, adaptor.getWorkspace());

    Value session = rewriter
                        .create<emitc::VariableOp>(
                            loc, getEmitCVariableResultType(sessionTy),
                            emitc::OpaqueAttr::get(ctx, ""))
                        .getResult();
    session = loadEmitCVariableIfNeeded(rewriter, loc, session);

    Value syncIdVal = makeAsyncU32Constant(rewriter, loc, ctx, op.getSyncIdAttr(), 0);
    Value channelGroupIdxVal =
        buildChannelGroupIdxValue(rewriter, loc, ctx, op.getChannelGroupIdxAttr());
    Value baseConfig =
        buildSdmaBaseConfig(rewriter, loc, ctx, op.getBlockBytesAttr(),
                            op.getCommBlockOffsetAttr(), op.getQueueNumAttr());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{*scratchTile, workspace, session, syncIdVal, baseConfig,
                   channelGroupIdxVal});

    rewriter.replaceOp(op, session);
    return success();
  }

  // u32 constant from an optional integer attribute with a default.
  static Value makeAsyncU32Constant(ConversionPatternRewriter &rewriter,
                                    Location loc, MLIRContext *ctx,
                                    IntegerAttr attr, uint64_t defaultValue) {
    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");
    uint64_t value =
        attr ? static_cast<uint64_t>(getIntegerAttrSignedValue(attr))
             : defaultValue;
    return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                    std::to_string(value) + "u");
  }

  // channel_group_idx defaults to UINT32_MAX when the attribute is absent.
  static Value buildChannelGroupIdxValue(ConversionPatternRewriter &rewriter,
                                         Location loc, MLIRContext *ctx,
                                         IntegerAttr attr) {
    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");
    if (!attr)
      return makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX");
    uint64_t value = static_cast<uint64_t>(getIntegerAttrSignedValue(attr));
    if (value == UINT32_MAX)
      return makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX");
    return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                    std::to_string(value) + "u");
  }

  // SdmaBaseConfig{blockBytes, commBlockOffset, queueNum} aggregate.
  static Value buildSdmaBaseConfig(ConversionPatternRewriter &rewriter,
                                   Location loc, MLIRContext *ctx,
                                   IntegerAttr blockBytesAttr,
                                   IntegerAttr commBlockOffsetAttr,
                                   IntegerAttr queueNumAttr) {
    uint64_t blockBytes =
        blockBytesAttr
            ? static_cast<uint64_t>(
                  getIntegerAttrSignedValue(blockBytesAttr))
            : 32 * 1024;
    uint64_t commBlockOffset =
        commBlockOffsetAttr
            ? static_cast<uint64_t>(
                  getIntegerAttrSignedValue(commBlockOffsetAttr))
            : 0;
    uint64_t queueNum =
        queueNumAttr
            ? static_cast<uint64_t>(getIntegerAttrSignedValue(queueNumAttr))
            : 1;

    auto baseConfigTy =
        emitc::OpaqueType::get(ctx, "pto::comm::sdma::SdmaBaseConfig");
    Value baseConfig =
        rewriter
            .create<emitc::VariableOp>(
                loc, getEmitCVariableResultType(baseConfigTy),
                emitc::OpaqueAttr::get(
                    ctx, "{" + std::to_string(blockBytes) + "ULL, " +
                             std::to_string(commBlockOffset) + "ULL, " +
                             std::to_string(queueNum) + "u}"))
            .getResult();
    return loadEmitCVariableIfNeeded(rewriter, loc, baseConfig);
  }
};

template <typename AsyncOp>
struct PTOAsyncTransferToEmitC : public OpConversionPattern<AsyncOp> {
  using OpConversionPattern<AsyncOp>::OpConversionPattern;

  explicit PTOAsyncTransferToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                   StringRef callee)
      : OpConversionPattern<AsyncOp>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncOp op, typename AsyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelGlobalTensorConversionBridge(adaptor.getDst());
    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Type convertedDstTy =
        this->getTypeConverter()->convertType(op.getDst().getType());
    Type convertedSrcTy =
        this->getTypeConverter()->convertType(op.getSrc().getType());
    if (!convertedDstTy || !convertedSrcTy ||
        !isEmitCGlobalTensorLikeType(convertedDstTy) ||
        !isEmitCGlobalTensorLikeType(convertedSrcTy))
      return rewriter.notifyMatchFailure(
          op, "expected GlobalTensor-like src and dst");

    Type eventTy = this->getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{eventTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, adaptor.getSession()});
    return success();
  }

  std::string callee;
};

template <typename AsyncEventOp>
struct PTOAsyncEventToEmitC : public OpConversionPattern<AsyncEventOp> {
  explicit PTOAsyncEventToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<AsyncEventOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncEventOp op,
                                typename AsyncEventOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        this->getTypeConverter()->convertType(op.getCompleted().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event result type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{adaptor.getEvent(),
                   adaptor.getSession()});
    return success();
  }

  std::string callee;
};





// Historical hook for pre-annotated TNotify release drains. The automatic
// MemoryConsistency analysis pass that used to produce these attrs has been
// removed from the default pipeline; keeping the lowering hook is harmless for
// hand-authored or legacy IR that already carries the internal attrs.



template <typename CollectiveOp>
struct PTOCommCollectiveToEmitC : public OpConversionPattern<CollectiveOp> {
  using OpConversionPattern<CollectiveOp>::OpConversionPattern;

  explicit PTOCommCollectiveToEmitC(TypeConverter &typeConverter,
                                    MLIRContext *ctx, StringRef apiName)
      : OpConversionPattern<CollectiveOp>(typeConverter, ctx),
        apiName(apiName.str()) {}

  // Operand bundle for a collective: the main global tensor, the ping tile,
  // the optional pong tile, and the parallel group.
  struct CollectiveOperands {
    FailureOr<Value> mainGT;
    FailureOr<Value> pingTile;
    FailureOr<Value> pongTile;
    FailureOr<Value> parallelGroup;
  };

  // Emit the collective call, appending the pong tile only when present.
  void emitCollectiveCall(CollectiveOp op, ConversionPatternRewriter &rewriter,
                          StringRef callee, const CollectiveOperands &ops) const {
    Location loc = op.getLoc();
    SmallVector<Value> args{*ops.parallelGroup, *ops.mainGT, *ops.pingTile};
    if (succeeded(ops.pongTile))
      args.push_back(*ops.pongTile);
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, callee, ArrayAttr{},
                                         ArrayAttr{}, ValueRange(args));
  }

  // Shared operand resolution for collectives with a (src|dst)-GT + ping/pong
  // tiles + parallel group shape.
  LogicalResult resolvePingPongOperands(
      CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter, Value mainValue, Value mainAdaptor,
      Value pongValue, Value pongAdaptor, StringRef what,
      CollectiveOperands &out) const {
    Location loc = op.getLoc();
    out.mainGT = buildCommGlobalTensorValue(rewriter, loc, mainValue, mainAdaptor,
                                            op.getOperation());
    out.pingTile =
        buildCommTileValue(rewriter, loc, op.getPing(), adaptor.getPing());
    auto groupGTs = buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(),
                                                adaptor.getGroup());
    if (failed(out.mainGT) || failed(out.pingTile) || failed(groupGTs))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize " + what + " operands");
    out.parallelGroup =
        buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
    if (failed(out.parallelGroup))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize " + what + " group");
    if (pongValue)
      out.pongTile =
          buildCommTileValue(rewriter, loc, pongValue, pongAdaptor);
    if (pongValue && failed(out.pongTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
    return success();
  }

  LogicalResult matchAndRewrite(CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    CollectiveOperands ops;
    if constexpr (std::is_same_v<CollectiveOp, pto::TBroadcastOp>) {
      if (failed(resolvePingPongOperands(op, adaptor, rewriter, op.getSrc(),
                                          adaptor.getSrc(), op.getPong(),
                                          adaptor.getPong(), "broadcast", ops)))
        return failure();
      emitCollectiveCall(op, rewriter, "pto::comm::TBROADCAST", ops);
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTGatherOp>) {
      if (failed(resolvePingPongOperands(op, adaptor, rewriter, op.getDst(),
                                          adaptor.getDst(), op.getPong(),
                                          adaptor.getPong(), "gather", ops)))
        return failure();
      emitCollectiveCall(op, rewriter, "pto::comm::TGATHER", ops);
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTScatterOp>) {
      if (failed(resolvePingPongOperands(op, adaptor, rewriter, op.getSrc(),
                                          adaptor.getSrc(), op.getPong(),
                                          adaptor.getPong(), "scatter", ops)))
        return failure();
      emitCollectiveCall(op, rewriter, "pto::comm::TSCATTER", ops);
    } else {
      return matchAndRewriteReduce(op, adaptor, rewriter);
    }
    rewriter.eraseOp(op);
    return success();
  }

  // TREDUCE carries an extra ReduceOp constant plus acc/recvPing/recvPong tiles.
  LogicalResult
  matchAndRewriteReduce(CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    FailureOr<Value> dstGT = buildCommGlobalTensorValue(
        rewriter, loc, op.getDst(), adaptor.getDst(), op.getOperation());
    FailureOr<Value> accTile =
        buildCommTileValue(rewriter, loc, op.getAcc(), adaptor.getAcc());
    FailureOr<Value> recvPing =
        buildCommTileValue(rewriter, loc, op.getRecvPing(), adaptor.getRecvPing());
    auto groupGTs = buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(),
                                                adaptor.getGroup());
    if (failed(dstGT) || failed(accTile) || failed(recvPing) || failed(groupGTs))
      return rewriter.notifyMatchFailure(op, "failed to materialize reduce operands");
    FailureOr<Value> pg =
        buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
    if (failed(pg))
      return rewriter.notifyMatchFailure(op, "failed to materialize reduce group");

    auto reduceTy =
        emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::ReduceOp");
    Value reduceOp = makeEmitCOpaqueConstant(rewriter, loc, reduceTy,
                                             reduceOpTok(op.getReduceOp()));
    SmallVector<Value> args{*pg, *dstGT, *accTile, *recvPing};
    if (op.getRecvPong()) {
      FailureOr<Value> recvPong =
          buildCommTileValue(rewriter, loc, op.getRecvPong(), adaptor.getRecvPong());
      if (failed(recvPong))
        return rewriter.notifyMatchFailure(op, "failed to materialize recv_pong");
      args.push_back(*recvPong);
    }
    args.push_back(reduceOp);
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "pto::comm::TREDUCE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(args));
    rewriter.eraseOp(op);
    return success();
  }

  std::string apiName;
};

template <typename OpTy>
struct PTOP2PCommToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTOP2PCommToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                             StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> dstGT =
        buildCommGlobalTensorValue(rewriter, op.getLoc(), op.getDst(), adaptor.getDst(),
                                   op.getOperation());
    FailureOr<Value> srcGT =
        buildCommGlobalTensorValue(rewriter, op.getLoc(), op.getSrc(), adaptor.getSrc(),
                                   op.getOperation());
    FailureOr<Value> pingTile =
        buildCommTileValue(rewriter, op.getLoc(), op.getPing(), adaptor.getPing());
    if (failed(dstGT) || failed(srcGT) || failed(pingTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize p2p operands");

    SmallVector<Value> operands{*dstGT, *srcGT, *pingTile};
    std::string actualCallee = callee;
    if constexpr (std::is_same_v<OpTy, pto::TPutOp>) {
      if (op.getAtomicType() == pto::AtomicType::AtomicAdd)
        actualCallee = "pto::comm::TPUT<pto::AtomicType::AtomicAdd>";
    }
    if (op.getPong()) {
      FailureOr<Value> pongTile =
          buildCommTileValue(rewriter, op.getLoc(), op.getPong(), adaptor.getPong());
      if (failed(pongTile))
        return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
      operands.push_back(*pongTile);
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, actualCallee,
                                         ArrayAttr{}, ArrayAttr{}, operands);
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

template <typename SignalOp>
struct PTOSignalCommToEmitC : public OpConversionPattern<SignalOp> {
  using OpConversionPattern<SignalOp>::OpConversionPattern;

  explicit PTOSignalCommToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<SignalOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(SignalOp op, typename SignalOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> signalGT = buildCommGlobalTensorValue(
        rewriter, op.getLoc(), op.getSignal(), adaptor.getSignal(), op.getOperation());
    if (failed(signalGT))
      return rewriter.notifyMatchFailure(op, "failed to materialize signal operand");

    if constexpr (std::is_same_v<SignalOp, pto::TNotifyOp>) {
      auto notifyTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::NotifyOp");
      Value notifyOp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), notifyTy, notifyOpTok(op.getNotifyOp()));
      SmallVector<Value> operands{*signalGT, adaptor.getValue(),
                                  notifyOp};
      // See emitTNotifyReleaseActions comment: drain in-flight MTE work before the
      // scalar-pipe signal store so the notify/wait handshake is honored.
      bool drainMte2 = op->hasAttr(kTNotifyDrainMte2AttrName);
      bool drainMte3 = op->hasAttr(kTNotifyDrainMte3AttrName);
      emitTNotifyReleaseActions(rewriter, op.getLoc(), drainMte2, drainMte3);
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{}, operands);
      rewriter.eraseOp(op);
    } else {
      auto waitCmpTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::WaitCmp");
      Value waitCmp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), waitCmpTy, waitCmpTok(op.getCmp()));
      SmallVector<Value> operands{*signalGT, adaptor.getCmpValue(),
                                  waitCmp};
      if constexpr (std::is_same_v<SignalOp, pto::TTestOp>) {
        Type resultTy = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!resultTy)
          return rewriter.notifyMatchFailure(op, "failed to convert ttest result type");
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{}, operands);
      } else {
        rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                             ArrayAttr{}, ArrayAttr{}, operands);
        rewriter.eraseOp(op);
      }
    }
    return success();
  }

  std::string callee;
};

struct PTODeclareGlobalToEmitC
    : public OpConversionPattern<mlir::pto::DeclareGlobalOp> {
  using OpConversionPattern<
      mlir::pto::DeclareGlobalOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareGlobalOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getEntry().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_global result type");
    if (auto tvTy = dyn_cast<TensorViewType>(op.getEntry().getType())) {
      if (auto stridesAttr =
              op->getAttrOfType<DenseI64ArrayAttr>(kGlobalTensorStridesAttrName)) {
        auto strides = stridesAttr.asArrayRef();
        if (strides.size() == static_cast<size_t>(tvTy.getRank())) {
          convertedType = emitc::OpaqueType::get(
              rewriter.getContext(),
              getGlobalTensorTypeStringFromShapeAndStrides(
                  tvTy.getElementType(), tvTy.getShape(), strides));
        }
      }
    }
    auto var = rewriter.create<emitc::VariableOp>(
        op.getLoc(), getEmitCVariableResultType(convertedType),
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(
        op, loadEmitCVariableIfNeeded(rewriter, op.getLoc(), var.getResult()));
    return success();
  }
};

struct PTODeclareEventIdArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareEventIdArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareEventIdArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareEventIdArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map declared eventid_array type");

    auto array = rewriter
                     .create<emitc::VariableOp>(
                         op.getLoc(), getEmitCVariableResultType(arrayTy),
                         emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                     .getResult();
    array = loadEmitCVariableIfNeeded(rewriter, op.getLoc(), array);
    rewriter.replaceOp(op, array);
    return success();
  }
};

struct PTOEventIdArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = adaptor.getArray();
    Value index = adaptor.getIndex();

    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map eventid_array get result type");

    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), resultTy, array, ValueRange{index});
    rewriter.replaceOp(op, subscript.getResult());
    return success();
  }
};

struct PTOEventIdArraySetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArraySetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = adaptor.getArray();
    Value index = adaptor.getIndex();
    Value value = adaptor.getValue();

    Value slot = rewriter
                     .create<emitc::SubscriptOp>(
                         op.getLoc(), value.getType(), array,
                         ValueRange{index})
                     .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), slot, value);
    rewriter.eraseOp(op);
    return success();
  }
};

// pto.declare_local_array -> emitc.variable of !emitc.array<...>.
// Renders as `T a[D1][D2]...;` in the emitted C++.
struct PTODeclareLocalArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareLocalArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareLocalArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareLocalArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map !pto.local_array type");

    auto var = rewriter
                   .create<emitc::VariableOp>(
                       op.getLoc(), getEmitCVariableResultType(arrayTy),
                       emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                   .getResult();
    var = loadEmitCVariableIfNeeded(rewriter, op.getLoc(), var);
    rewriter.replaceOp(op, var);
    return success();
  }
};

// pto.local_array_get %a[%i0, %i1, ...] -> scalar snapshot.
// Materialize the subscript read immediately so the MLIR SSA result keeps its
// value even if a later pto.local_array_set mutates the same backing array slot.
struct PTOLocalArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(
          op, "failed to map local_array element type");

    Value array = adaptor.getArray();
    SmallVector<Value> indices;
    indices.reserve(adaptor.getIndices().size());
    for (Value index : adaptor.getIndices())
      indices.push_back(peelUnrealized(index));

    auto sub = rewriter.create<emitc::SubscriptOp>(op.getLoc(), resultTy,
                                                   array, indices);
    auto snapshot =
        rewriter
            .create<emitc::VariableOp>(
                op.getLoc(), resultTy,
                emitc::OpaqueAttr::get(rewriter.getContext(), ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), snapshot, sub.getResult());
    rewriter.replaceOp(op, snapshot);
    return success();
  }
};

// pto.local_array_set %a[%i0, %i1, ...], %v -> emitc.assign to subscript slot.
// The C++ emitter prints this as `a[i0][i1]... = v;`. As above, adaptor values
// are already target-typed; pass them through directly.
struct PTOLocalArraySetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArraySetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Type elemTy = value.getType();

    Value slot = rewriter
                     .create<emitc::SubscriptOp>(
                         op.getLoc(), elemTy, adaptor.getArray(),
                         adaptor.getIndices())
                     .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), slot, value);
    rewriter.eraseOp(op);
    return success();
  }
};

// pto.declare_struct -> emitc.variable of !emitc.opaque<"PtoStruct_...">.
// Renders as `PtoStruct_X s;` in the emitted C++.
struct PTODeclareStructToEmitC
    : public OpConversionPattern<mlir::pto::DeclareStructOp> {
  using OpConversionPattern<mlir::pto::DeclareStructOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareStructOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    const bool hasUnexpectedResultCount = op->getNumResults() != 1;
    if (hasUnexpectedResultCount) {
      return rewriter.notifyMatchFailure(op, "expected one !pto.struct result");
    }
    Type structTy =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!structTy) {
      return rewriter.notifyMatchFailure(op, "failed to map !pto.struct type");
    }

    // The struct converts to a pointer, so declare the storage as a local
    // variable and hand out its address. buildStructMemberChain recognises the
    // address-of and walks that variable directly, so a struct that never
    // leaves the function still prints as `s.f0` rather than `p->f0`.
    auto ptrTy = dyn_cast<emitc::PointerType>(structTy);
    if (!ptrTy)
      return rewriter.notifyMatchFailure(op,
                                         "!pto.struct did not map to a pointer");

    Value storage = rewriter
                        .create<emitc::VariableOp>(
                            op.getLoc(), ptrTy.getPointee(),
                            emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                        .getResult();
    rewriter.replaceOpWithNewOp<emitc::ApplyOp>(op, ptrTy, "&", storage);
    return success();
  }
};

// The EmitC *value* type of a struct field, i.e. the type an lvalue to that
// field wraps. A nested struct is spelled directly here rather than going
// through the converter, which would hand back the pointer form used for
// passing whole structs around — a field lives inside its parent's storage and
// is reached with `.`, not through another pointer.
// The EmitC *value* type of a struct field, i.e. the type an lvalue to that
// field wraps. A nested struct is spelled directly here rather than going
// through the converter, which would hand back the pointer form used for
// passing whole structs around — a field lives inside its parent's storage and
// is reached with `.`, not through another pointer.
static Type getStructFieldValueType(const TypeConverter *tc, Type fieldPtoTy) {
  if (auto st = dyn_cast<pto::StructType>(fieldPtoTy)) {
    return emitc::OpaqueType::get(st.getContext(), getStructTypeName(st));
  }
  return tc->convertType(fieldPtoTy);
}


static FailureOr<Type> getStructMemberFieldType(mlir::pto::StructType structTy,
                                                int64_t index,
                                                const TypeConverter *tc) {
  if (index < 0 ||
      static_cast<unsigned>(index) >= structTy.getFieldTypes().size()) {
    return failure();
  }
  return getStructFieldValueType(
      tc, structTy.getFieldType(static_cast<unsigned>(index)));
}

static FailureOr<Value> getStructAdaptorValue(ValueRange operands) {
  if (operands.empty()) {
    return failure();
  }
  return operands.front();
}

// Build the `s.fA.fB...` member-access chain for a constant struct path and
// return the final lvalue. `rootPtoTy` is the PTO struct type, walked in
// parallel to look up field types per step.
//
// Every step is an `emitc.member`, which requires an lvalue operand and yields
// an lvalue result, so the chain stays in lvalue form throughout — that is what
// makes a write land in the struct rather than in a copy of it.
//
// `root` is the converted struct, i.e. a pointer. Two shapes reach here:
//   - a local declared by pto.declare_struct, whose pointer is an address-of;
//     that is unwrapped back to the variable so the access prints as `s.f0`.
//   - any other pointer, notably a function argument. `emitc.member_of_ptr`
//     needs an lvalue *holding* the pointer rather than the raw pointer, so it
//     is parked in a variable first and the access prints as `p->f0`.
static FailureOr<Value> buildStructMemberChain(
    ConversionPatternRewriter &rewriter, Location loc, const TypeConverter *tc,
    Value root, mlir::pto::StructType rootPtoTy, llvm::ArrayRef<int64_t> path) {
  Value ptr = peelUnrealized(root);

  // lvalue of the struct itself when we can name it; otherwise an lvalue
  // holding the pointer, consumed by the first member_of_ptr step.
  Value structLValue;
  Value ptrSlot;
  auto applyOp = ptr.getDefiningOp<emitc::ApplyOp>();
  if (applyOp && applyOp.getApplicableOperator() == "&") {
    structLValue = applyOp.getOperand();
  } else {
    if (!isa<emitc::PointerType>(ptr.getType())) {
      return failure();
    }
    ptrSlot = rewriter
                  .create<emitc::VariableOp>(
                      loc, ptr.getType(),
                      emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                  .getResult();
    rewriter.create<emitc::AssignOp>(loc, ptrSlot, ptr);
  }

  Type curPtoTy = rootPtoTy;
  for (int64_t idx : path) {
    auto st = dyn_cast<mlir::pto::StructType>(curPtoTy);
    if (!st) {
      return failure();
    }
    FailureOr<Type> fieldTy = getStructMemberFieldType(st, idx, tc);
    if (failed(fieldTy)) {
      return failure();
    }
    Type resultTy = *fieldTy;
    auto name = rewriter.getStringAttr("f" + std::to_string(idx));
    // Only the first step off a bare pointer uses `->`; from there on the
    // chain is walking storage we can name, so it is all `.`.
    structLValue =
        structLValue
            ? rewriter.create<emitc::MemberOp>(loc, resultTy, name, structLValue)
                  .getResult()
            : rewriter
                  .create<emitc::MemberOfPtrOp>(loc, resultTy, name, ptrSlot)
                  .getResult();
    curPtoTy = st.getFieldType(static_cast<unsigned>(idx));
  }
  return structLValue;
}

/// Resolve the struct operand and member-access chain shared by struct_get
// and struct_set; returns the member lvalue or match failure.
static FailureOr<Value>
resolveStructMember(Operation *op, ValueRange adaptorOperands, Type structPtoTy,
                    ArrayRef<int64_t> path, ConversionPatternRewriter &rewriter,
                    const TypeConverter *typeConverter) {
  FailureOr<Value> structValue = getStructAdaptorValue(adaptorOperands);
  const bool hasInvalidStructOperand =
      failed(structValue) || op->getNumOperands() == 0;
  if (hasInvalidStructOperand)
    return rewriter.notifyMatchFailure(op, "expected struct operand");
  auto structTy = dyn_cast<mlir::pto::StructType>(structPtoTy);
  if (!structTy)
    return rewriter.notifyMatchFailure(op, "expected !pto.struct operand");
  return buildStructMemberChain(rewriter, op->getLoc(), typeConverter,
                                *structValue, structTy, path);
}

// pto.struct_get %s[i, j, ...] -> `s.fi.fj...`. The verifier guarantees the path
// ends on a scalar, so the member lvalue is read with emitc.load. That load is
// materialized into its own C++ variable, which is what gives the SSA result
// value semantics: it keeps its value even if a later pto.struct_set writes the
// same field (mirrors pto.local_array_get).
struct PTOStructGetToEmitC
    : public OpConversionPattern<mlir::pto::StructGetOp> {
  using OpConversionPattern<mlir::pto::StructGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::StructGetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "failed to map struct field type");
    }

    FailureOr<Value> member = resolveStructMember(
        op.getOperation(), adaptor.getOperands(),
        op->getOperand(0).getType(), op.getPath(), rewriter,
        getTypeConverter());
    if (failed(member)) {
      return rewriter.notifyMatchFailure(op, "failed to map struct field type");
    }

    // Materialize the read into its own C++ variable so the SSA result keeps
    // its value even if a later pto.struct_set writes the same field.
    auto snapshot =
        rewriter
            .create<emitc::VariableOp>(
                op.getLoc(), resultTy,
                emitc::OpaqueAttr::get(rewriter.getContext(), ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), snapshot, *member);
    rewriter.replaceOp(op, snapshot);
    return success();
  }
};

// pto.struct_set %s[i, j, ...], %v -> `s.fi.fj... = v;`.
struct PTOStructSetToEmitC
    : public OpConversionPattern<mlir::pto::StructSetOp> {
  using OpConversionPattern<mlir::pto::StructSetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::StructSetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> member = resolveStructMember(
        op.getOperation(), adaptor.getOperands(),
        op->getOperand(0).getType(), op.getPath(), rewriter,
        getTypeConverter());
    if (failed(member)) {
      return rewriter.notifyMatchFailure(op, "failed to map struct field type");
    }

    rewriter.create<emitc::AssignOp>(op.getLoc(), *member, adaptor.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};



FailureOr<Value> buildCollectiveParallelGroup(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Value> groupGTs, int64_t root) {
  if (groupGTs.empty())
    return failure();

  auto firstTy = dyn_cast<emitc::OpaqueType>(groupGTs.front().getType());
  if (!firstTy)
    return failure();

  auto *ctx = rewriter.getContext();
  auto arrayTy = emitc::ArrayType::get({static_cast<int64_t>(groupGTs.size())},
                                       firstTy);
  auto groupArray = cast<TypedValue<emitc::ArrayType>>(
      rewriter
          .create<emitc::VariableOp>(loc, getEmitCVariableResultType(arrayTy),
                                     emitc::OpaqueAttr::get(ctx, "{}"))
          .getResult());

  auto indexTy = emitc::OpaqueType::get(ctx, "int");
  for (auto [idx, groupVal] : llvm::enumerate(groupGTs)) {
    Value idxVal =
        makeEmitCIntConstant(rewriter, loc, indexTy, static_cast<int64_t>(idx));
    Value slot =
        rewriter.create<emitc::SubscriptOp>(loc, groupArray, ValueRange{idxVal})
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, slot, groupVal);
  }

  std::string pgTypeStr =
      (Twine("pto::comm::ParallelGroup<") + firstTy.getValue() + ">").str();
  auto pgTy = emitc::OpaqueType::get(ctx, pgTypeStr);
  Value sizeVal = makeEmitCIntConstant(rewriter, loc, indexTy,
                                       static_cast<int64_t>(groupGTs.size()));
  Value rootVal = makeEmitCIntConstant(rewriter, loc, indexTy, root);
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, TypeRange{pgTy}, (Twine(pgTypeStr) + "::Create").str(),
          ArrayAttr{}, ArrayAttr{}, ValueRange{groupArray, sizeVal, rootVal})
      .getResult(0);
}

FailureOr<Value> buildCommGlobalTensorValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalValue,
    Value emittedValue, Operation *anchor) {
  Value value = peelUnrealized(emittedValue);
  if (isEmitCGlobalTensorLikeType(value.getType()))
    return value;
  return failure();
}

template <typename OpTy>
FailureOr<SmallVector<Value>> buildCommGroupGlobalTensors(
    ConversionPatternRewriter &rewriter, Location loc, OpTy op,
    ValueRange originalGroup, ValueRange emittedGroup) {
  SmallVector<Value> groupGTs;
  groupGTs.reserve(originalGroup.size());
  for (auto [orig, emitted] : llvm::zip(originalGroup, emittedGroup)) {
    FailureOr<Value> gt =
        buildCommGlobalTensorValue(rewriter, loc, orig, emitted, op.getOperation());
    if (failed(gt))
      return failure();
    groupGTs.push_back(*gt);
  }
  return groupGTs;
}

FailureOr<Value> buildCommTileValue(ConversionPatternRewriter &rewriter,
                                           Location loc, Value originalValue,
                                           Value emittedValue) {
  Value value = peelUnrealized(emittedValue);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return value;
  }
  return buildAsyncScratchTileValue(rewriter, loc, originalValue, emittedValue);
}

void emitTNotifyReleaseActions(ConversionPatternRewriter &rewriter,
                                      Location loc, bool drainMte2,
                                      bool drainMte3) {
  if (drainMte2)
    emitPipeBarrier(rewriter, loc, "PIPE_MTE2");
  if (drainMte3)
    emitPipeBarrier(rewriter, loc, "PIPE_MTE3");
}

std::string notifyOpTok(pto::NotifyOp op) {
  switch (op) {
  case pto::NotifyOp::AtomicAdd:
    return "pto::comm::NotifyOp::AtomicAdd";
  case pto::NotifyOp::Set:
    return "pto::comm::NotifyOp::Set";
  }
  return "pto::comm::NotifyOp::Set";
}

std::string reduceOpTok(pto::ReduceOp op) {
  switch (op) {
  case pto::ReduceOp::Sum:
    return "pto::comm::ReduceOp::Sum";
  case pto::ReduceOp::Max:
    return "pto::comm::ReduceOp::Max";
  case pto::ReduceOp::Min:
    return "pto::comm::ReduceOp::Min";
  }
  return "pto::comm::ReduceOp::Sum";
}

std::string waitCmpTok(pto::WaitCmp cmp) {
  switch (cmp) {
  case pto::WaitCmp::EQ:
    return "pto::comm::WaitCmp::EQ";
  case pto::WaitCmp::NE:
    return "pto::comm::WaitCmp::NE";
  case pto::WaitCmp::GT:
    return "pto::comm::WaitCmp::GT";
  case pto::WaitCmp::GE:
    return "pto::comm::WaitCmp::GE";
  case pto::WaitCmp::LT:
    return "pto::comm::WaitCmp::LT";
  case pto::WaitCmp::LE:
    return "pto::comm::WaitCmp::LE";
  }
  return "pto::comm::WaitCmp::EQ";
}


void emitConservativeGmFencePipeDrains(
    ConversionPatternRewriter &rewriter, Location loc) {
  emitPipeBarrier(rewriter, loc, "PIPE_MTE2");
  emitPipeBarrier(rewriter, loc, "PIPE_MTE3");
  emitPipeBarrier(rewriter, loc, "PIPE_FIX");
}

void emitDsbDdr(ConversionPatternRewriter &rewriter, Location loc) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "DSB_DDR")});
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "dsb", args,
                                       ArrayAttr{}, ValueRange{});
}

void emitPipeBarrier(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef pipeTok) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, pipeTok)});
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "pipe_barrier", args,
                                       ArrayAttr{}, ValueRange{});
}

std::string getAutoSyncTailModeToken(Operation *op) {
  if (op) {
    if (auto hintAttr = op->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr)) {
      if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
        return kAutoSyncTailModeBarrierAllToken.str();
      if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
        return kAutoSyncTailModeMte3ToSEvent0Token.str();
    }
  }

  auto func = op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!func)
    return kAutoSyncTailModeBarrierAllToken.str();

  auto hintAttr = func->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr);
  if (!hintAttr)
    return kAutoSyncTailModeBarrierAllToken.str();

  if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
    return kAutoSyncTailModeBarrierAllToken.str();
  if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
    return kAutoSyncTailModeMte3ToSEvent0Token.str();

  // Fallback to the conservative behavior when seeing unknown policies.
  return kAutoSyncTailModeBarrierAllToken.str();
}

bool isInVectorKernel(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<pto::SectionVectorOp>(parent))
      return true;

    auto kernelKindAttr = parent->getAttrOfType<FunctionKernelKindAttr>(
        FunctionKernelKindAttr::name);
    if (kernelKindAttr)
      return kernelKindAttr.getKernelKind() == FunctionKernelKind::Vector;
  }
  return false;
}


void emitInvalidateGmCacheAll(ConversionPatternRewriter &rewriter,
                                     Location loc) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, "(__gm__ void*)0"),
      emitc::OpaqueAttr::get(ctx, "cache_line_t::ENTIRE_DATA_CACHE"),
  });
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "dcci", args,
                                       ArrayAttr{}, ValueRange{});
}

void emitInvalidateGmCacheSingleLine(ConversionPatternRewriter &rewriter,
                                            Location loc, Value addr) {
  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "PTOAS__DCCI_SINGLE_CACHE_LINE",
      ArrayAttr{}, ArrayAttr{}, ValueRange{addr});
}

Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

bool isGmCmoSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}


void populateScalarMiscPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  patterns.add<PTOCmoCacheInvalidToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetValToSETVAL, PTOGetValToGETVAL, PTOSetValidShapeToEmitC,
               PTOGetValidShapeToEmitC, PTOTAssignToEmitC,
               PTOPtrToIntToEmitC, PTOIntToPtrToEmitC, PTOLoadScalarToEmitC,
               PTOStoreScalarToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAxpyToEmitC, PTOHistogramToEmitC, PTOGetScaleAddrToEmitC>(
      typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
  patterns.add<PTOTAbsToTABS>(typeConverter, ctx);
  patterns.add<PTOTAddToTADD>(typeConverter, ctx);
  patterns.add<PTOBuildAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOAsyncTransferToEmitC<pto::TPutAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncTransferToEmitC<pto::TGetAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOP2PCommToEmitC<pto::TPutOp>>(typeConverter, ctx,
                                               "pto::comm::TPUT");
  patterns.add<PTOP2PCommToEmitC<pto::TGetOp>>(typeConverter, ctx,
                                               "pto::comm::TGET");
  patterns.add<PTOSignalCommToEmitC<pto::TNotifyOp>>(typeConverter, ctx,
                                                     "pto::comm::TNOTIFY");
    populateScalarMiscPatternsPart2(patterns, typeConverter, ctx, targetArch);
}

void populateScalarMiscPatternsPart2(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx, PTOArch targetArch) {
  (void)typeConverter;
  (void)ctx;
  (void)targetArch;
  patterns.add<PTOSignalCommToEmitC<pto::TWaitOp>>(typeConverter, ctx,
                                                   "pto::comm::TWAIT");
  patterns.add<PTOSignalCommToEmitC<pto::TTestOp>>(typeConverter, ctx,
                                                   "pto::comm::TTEST");
  patterns.add<PTOCommCollectiveToEmitC<pto::TBroadcastOp>>(typeConverter, ctx,
                                                            "TBROADCAST");
  patterns.add<PTOCommCollectiveToEmitC<pto::CommTGatherOp>>(typeConverter, ctx,
                                                             "TGATHER");
  patterns.add<PTOCommCollectiveToEmitC<pto::CommTScatterOp>>(typeConverter, ctx,
                                                              "TSCATTER");
  patterns.add<PTOCommCollectiveToEmitC<pto::TReduceOp>>(typeConverter, ctx,
                                                         "TREDUCE");
  patterns.add<PTOAsyncEventToEmitC<pto::WaitAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_WAIT");
  patterns.add<PTOAsyncEventToEmitC<pto::TestAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_TEST");
  patterns.add<PTOInitializeL2G2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOInitializeL2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTODeclareGlobalToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareEventIdArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArraySetToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareLocalArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArraySetToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareStructToEmitC>(typeConverter, ctx);
  patterns.add<PTOStructGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOStructSetToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
