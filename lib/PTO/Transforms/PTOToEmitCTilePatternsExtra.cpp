// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTilePatternsExtra.cpp -----------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"
#include "PTOToEmitCTilePatternCommon.h"

#include <string>

#include "PTO/IR/PTO.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

constexpr size_t kNumber1 = 1;
constexpr size_t kNumber2 = 2;
constexpr size_t kNumber4 = 4;

[[maybe_unused]] static std::string maskPatternTok(mlir::pto::MaskPatternAttr a) {
  auto value = a.getValue();
  return (std::string("pto::MaskPattern::") +
          mlir::pto::stringifyMaskPattern(value).str());
}

template <typename RowExpandOp, typename AdaptorT>
static LogicalResult lowerRowExpandBinaryLikeOp(RowExpandOp op, AdaptorT adaptor,
                                                StringRef callee,
                                                ConversionPatternRewriter *rewriter) {
  Value src0 = peelUnrealized(adaptor.getSrc0());
  Value src1 = peelUnrealized(adaptor.getSrc1());
  Value dst = peelUnrealized(adaptor.getDst());
  Value tmp = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

  SmallVec4<Value> operands;
  if (tmp)
    operands.assign({dst, src0, src1, tmp});
  else
    operands.assign({dst, src0, src1});

  rewriter->create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                        /*args=*/ArrayAttr{},
                                        /*templateArgs=*/ArrayAttr{},
                                        /*operands=*/operands);
  rewriter->eraseOp(op);
  return success();
}

template <typename RowExpandOp, typename AdaptorT>
static LogicalResult lowerRowExpandBinaryNoTmpOp(
    RowExpandOp op, AdaptorT adaptor, StringRef callee,
    ConversionPatternRewriter *rewriter) {
  Value src0 = peelUnrealized(adaptor.getSrc0());
  Value src1 = peelUnrealized(adaptor.getSrc1());
  Value dst = peelUnrealized(adaptor.getDst());
  SmallVec3<Value> operands{dst, src0, src1};
  rewriter->create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                        /*args=*/ArrayAttr{},
                                        /*templateArgs=*/ArrayAttr{},
                                        /*operands=*/operands);
  rewriter->eraseOp(op);
  return success();
}

struct PTOMovFPToEmitC : public OpConversionPattern<pto::TMovFPOp> {
  using OpConversionPattern<pto::TMovFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMovFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // TMOV_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile)
    ArrayAttr templateArgs;
    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    auto fpOT  = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVec3<Value> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMOV_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOQuantToEmitC : public OpConversionPattern<pto::TQuantOp> {
  using OpConversionPattern<pto::TQuantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TQuantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // Optional offset (INT8_ASYM only): passed as pointer (&offset)
    Value offsetPtr;
    if (op.getOffset()) {
      Value offset = peelUnrealized(adaptor.getOffset());
      auto offsetOT = mlir::dyn_cast<emitc::OpaqueType>(offset.getType());
      if (offsetOT) {
        offsetPtr = rewriter
                        .create<emitc::ApplyOp>(
                            loc, emitc::PointerType::get(offsetOT), "&", offset)
                        .getResult();
      }
    }

    // TQUANT<QuantType, DstTile, SrcTile, FpTile>(dst, src, fp[, &offset])
    std::string quantTypeStr =
        op.getQuantType() == pto::QuantType::INT8_SYM
            ? "pto::QuantType::INT8_SYM"
            : "pto::QuantType::INT8_ASYM";
    ArrayAttr templateArgs;
    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    auto fpOT  = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, quantTypeStr),
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value> operands{dst, src, fp};
    if (offsetPtr)
      operands.push_back(offsetPtr);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TQUANT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTODequantToEmitC : public OpConversionPattern<pto::TDequantOp> {
  using OpConversionPattern<pto::TDequantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDequantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scale  = peelUnrealized(adaptor.getScale());
    Value offset = peelUnrealized(adaptor.getOffset());

    // TDEQUANT<DstTile, SrcTile, ParaTile>(dst, src, scale, offset)
    ArrayAttr templateArgs;
    auto dstOT   = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT   = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    auto scaleOT = mlir::dyn_cast<emitc::OpaqueType>(scale.getType());
    if (dstOT && srcOT && scaleOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, scaleOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TDEQUANT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/SmallVector<Value>{dst, src, scale, offset});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMRGSORT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMrgSortToEmitC : public OpConversionPattern<pto::TMrgSortOp> {
  using OpConversionPattern<pto::TMrgSortOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMrgSortOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.isFormat1()) {
      Value src = peelUnrealized(adaptor.getSrcs().front());
      Value dst = peelUnrealized(adaptor.getDsts().front());
      Value blockLen = peelUnrealized(adaptor.getBlockLen());

      SmallVec3<Value> operands{dst, src, blockLen};
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          ArrayAttr{}, ArrayAttr{}, operands);
    } else if (op.isFormat2()) {
      // pto-isa API
      //   TMRGSORT<DstTile, TmpTile, Src0, Src1[, Src2[, Src3]], exhausted>(
      //       dst, executedNumList, tmp, src0, src1[, src2[, src3]]);
      auto *ctx = rewriter.getContext();

      Value dst = peelUnrealized(adaptor.getDsts()[0]);
      Value tmp = peelUnrealized(adaptor.getTmp());
      Value excuted = peelUnrealized(adaptor.getExcuted());

      SmallVec4<Value> srcs;
      srcs.reserve(adaptor.getSrcs().size());
      for (Value v : adaptor.getSrcs())
        srcs.push_back(peelUnrealized(v));

      auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
      auto tmpOT = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
      if (!dstOT || !tmpOT || srcs.size() < kNumber2 || srcs.size() > kNumber4)
        return op.emitOpError("format2 expects dst/tmp tilebufs and 2 to 4 srcs");

      SmallVec8<Attribute> targs;
      targs.reserve(kNumber2 + srcs.size() + kNumber1);
      targs.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
      targs.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
      for (Value v : srcs) {
        auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType());
        if (!ot)
          return op.emitOpError("format2 expects tilebuf srcs");
        targs.push_back(emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
      }
      targs.push_back(emitc::OpaqueAttr::get(ctx, op.getExhausted() ? "true" : "false"));
      ArrayAttr templateArgs = rewriter.getArrayAttr(targs);

      SmallVec7<Value> operands{dst, excuted, tmp};
      operands.append(srcs.begin(), srcs.end());

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    } else {
      return op.emitOpError("unsupported mrgsort_dps format");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulToEmitC : public OpConversionPattern<pto::TMulOp> {
  using OpConversionPattern<pto::TMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMULS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulsToEmitC : public OpConversionPattern<pto::TMulSOp> {
  using OpConversionPattern<pto::TMulSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMulSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc0());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVec3<Value> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMULS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNEG DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONegToEmitC : public OpConversionPattern<pto::TNegOp> {
  using OpConversionPattern<pto::TNegOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNegOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec2<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNEG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNOT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONotToEmitC : public OpConversionPattern<pto::TNotOp> {
  using OpConversionPattern<pto::TNotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec2<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNOT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrToEmitC : public OpConversionPattern<pto::TOrOp> {
  using OpConversionPattern<pto::TOrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrsToEmitC : public OpConversionPattern<pto::TOrSOp> {
  using OpConversionPattern<pto::TOrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc());
    Value dst  = peelUnrealized(adaptor.getDst());
    // NOTE: The conversion type system may materialize integers as emitc.opaque
    // (e.g. "int32_t"). For EmitC call emission we can pass the scalar through
    // directly without arith casts here.
    Value s = adaptor.getScalar();

    SmallVec3<Value> operands{dst, src0, s};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartAddToEmitC : public OpConversionPattern<pto::TPartAddOp> {
  using OpConversionPattern<pto::TPartAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMaxToEmitC : public OpConversionPattern<pto::TPartMaxOp> {
  using OpConversionPattern<pto::TPartMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMinToEmitC : public OpConversionPattern<pto::TPartMinOp> {
  using OpConversionPattern<pto::TPartMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMaxToEmitC
    : public OpConversionPattern<pto::TPartArgMaxOp> {
  using OpConversionPattern<pto::TPartArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst = peelUnrealized(adaptor.getDst());
    Value dstIdx = peelUnrealized(adaptor.getDstIdx());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMinToEmitC
    : public OpConversionPattern<pto::TPartArgMinOp> {
  using OpConversionPattern<pto::TPartArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst = peelUnrealized(adaptor.getDst());
    Value dstIdx = peelUnrealized(adaptor.getDstIdx());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMulToEmitC : public OpConversionPattern<pto::TPartMulOp> {
  using OpConversionPattern<pto::TPartMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPreluToEmitC : public OpConversionPattern<pto::TPReluOp> {
  using OpConversionPattern<pto::TPReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    // C++ interface: TPRELU(dst, src0, src1, tmp) — last parameter is tmp.
    SmallVec4<Value> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORecipToEmitC : public OpConversionPattern<pto::TRecipOp> {
  using OpConversionPattern<pto::TRecipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRecipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec2<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRECIP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOReluToEmitC : public OpConversionPattern<pto::TReluOp> {
  using OpConversionPattern<pto::TReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec2<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemToEmitC : public OpConversionPattern<pto::TRemOp> {
  using OpConversionPattern<pto::TRemOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());
    SmallVec4<Value> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModToEmitC : public OpConversionPattern<pto::TFModOp> {
  using OpConversionPattern<pto::TFModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMOD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemSToEmitC : public OpConversionPattern<pto::TRemSOp> {
  using OpConversionPattern<pto::TRemSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVec4<Value> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREMS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModSToEmitC : public OpConversionPattern<pto::TFModSOp> {
  using OpConversionPattern<pto::TFModSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVec3<Value> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMODS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandToEmitC : public OpConversionPattern<pto::TRowExpandOp> {
  using OpConversionPattern<pto::TRowExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec2<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandAddToEmitC : public OpConversionPattern<pto::TRowExpandAddOp> {
  using OpConversionPattern<pto::TRowExpandAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryNoTmpOp(op, adaptor, "TROWEXPANDADD",
                                       &rewriter);
  }
};

struct PTORowExpandExpdifToEmitC
    : public OpConversionPattern<pto::TRowExpandExpdifOp> {
  using OpConversionPattern<pto::TRowExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryLikeOp(op, adaptor, "TROWEXPANDEXPDIF",
                                      &rewriter);
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDDIV DPS/memref op)
//===----------------------------------------------------------------------===//
// Helper: replace or erase based on whether op has results.
static void replaceOrEraseWithOpaqueCall(Operation *op,
                                        StringRef callee,
                                        ArrayRef<Value> args,
                                        ConversionPatternRewriter &rewriter) {
  TypeRange resultTypes = op->getResultTypes();
  auto call = rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(), resultTypes, callee, ArrayAttr{}, ArrayAttr{}, ValueRange(args));
  if (resultTypes.empty())
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, call.getResults());
}

static void replaceOrEraseWithOpaqueCallAndReturnDst(Operation *op, Value dst,
                                                     StringRef callee,
                                                     ArrayRef<Value> args,
                                                     ConversionPatternRewriter &rewriter) {
  rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(), TypeRange{}, callee, ArrayAttr{}, ArrayAttr{}, ValueRange(args));
  if (op->getNumResults() == 1)
    rewriter.replaceOp(op, dst);
  else
    rewriter.eraseOp(op);
}

static void emitBinaryThenAccumulate(ConversionPatternRewriter &rewriter,
                                     Location loc, StringRef firstCallee,
                                     Value dst, Value lhs, Value rhs,
                                     Value acc) {
  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, firstCallee,
      /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
      /*operands=*/ValueRange{dst, lhs, rhs});
  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TADD",
      /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
      /*operands=*/ValueRange{dst, dst, acc});
}

// ---------- TOp ----------
struct PTOTGemvBiasToTGEMV_BIAS
    : public OpConversionPattern<pto::TGemvBiasOp> {
  using OpConversionPattern<pto::TGemvBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TGEMV_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOTGemvMXToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxOp> {
  using OpConversionPattern<pto::TGemvMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTGemvMXAccToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxAccOp> {
  using OpConversionPattern<pto::TGemvMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, cIn, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTGemvMXBiasToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxBiasOp> {
  using OpConversionPattern<pto::TGemvMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale, bias}, rewriter);
    return success();
  }
};

struct PTOTMatmulBiasToTMATMUL_BIAS
    : public OpConversionPattern<pto::TMatmulBiasOp> {
  using OpConversionPattern<pto::TMatmulBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOTMatmulMXToTMATMUL_MX
    : public OpConversionPattern<pto::TMatmulMxOp> {
  using OpConversionPattern<pto::TMatmulMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTMatmulMXAccToTMATMUL_MX_ACC
    : public OpConversionPattern<pto::TMatmulMxAccOp> {
  using OpConversionPattern<pto::TMatmulMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, cIn, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTMatmulMXBiasToTMATMUL_MX_BIAS
    : public OpConversionPattern<pto::TMatmulMxBiasOp> {
  using OpConversionPattern<pto::TMatmulMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, a, aScale, b, bScale, bias}, rewriter);
    return success();
  }
};

struct PTORowExpandDivToEmitC : public OpConversionPattern<pto::TRowExpandDivOp> {
  using OpConversionPattern<pto::TRowExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryLikeOp(op, adaptor, "TROWEXPANDDIV", &rewriter);
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandMulToEmitC : public OpConversionPattern<pto::TRowExpandMulOp> {
  using OpConversionPattern<pto::TRowExpandMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryLikeOp(op, adaptor, "TROWEXPANDMUL", &rewriter);
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandSubToEmitC : public OpConversionPattern<pto::TRowExpandSubOp> {
  using OpConversionPattern<pto::TRowExpandSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryLikeOp(op, adaptor, "TROWEXPANDSUB", &rewriter);
  }
};

struct PTORowExpandMaxToEmitC : public OpConversionPattern<pto::TRowExpandMaxOp> {
  using OpConversionPattern<pto::TRowExpandMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryLikeOp(op, adaptor, "TROWEXPANDMAX", &rewriter);
  }
};

struct PTORowExpandMinToEmitC : public OpConversionPattern<pto::TRowExpandMinOp> {
  using OpConversionPattern<pto::TRowExpandMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerRowExpandBinaryLikeOp(op, adaptor, "TROWEXPANDMIN", &rewriter);
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMaxToEmitC : public OpConversionPattern<pto::TRowMaxOp> {
  using OpConversionPattern<pto::TRowMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowArgMaxToEmitC
    : public OpConversionPattern<pto::TRowArgMaxOp> {
  using OpConversionPattern<pto::TRowArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMinToEmitC : public OpConversionPattern<pto::TRowMinOp> {
  using OpConversionPattern<pto::TRowMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowArgMinToEmitC
    : public OpConversionPattern<pto::TRowArgMinOp> {
  using OpConversionPattern<pto::TRowArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowSumToEmitC : public OpConversionPattern<pto::TRowSumOp> {
  using OpConversionPattern<pto::TRowSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWSUM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowProdToEmitC : public OpConversionPattern<pto::TRowProdOp> {
  using OpConversionPattern<pto::TRowProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWPROD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRSQRT DPS/memref op)
// - no-tmp form : TRSQRT(dst, src)
// - tmp form    : TRSQRT(dst, src, tmp)
//===----------------------------------------------------------------------===//

struct PTORsqrtToEmitC : public OpConversionPattern<pto::TRsqrtOp> {
  using OpConversionPattern<pto::TRsqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRsqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    SmallVec3<Value> operands{dst, src};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOScatterToEmitC : public OpConversionPattern<pto::TScatterOp> {
  using OpConversionPattern<pto::TScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const bool hasMaskPattern = static_cast<bool>(op.getMaskPatternAttr());
    const bool hasIndexes = static_cast<bool>(op.getIndexes());
    if (hasMaskPattern == hasIndexes) {
      return rewriter.notifyMatchFailure(
          op, "expected exactly one of indexes operand or maskPattern attribute");
    }

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    if (auto mp = op.getMaskPatternAttr()) {
      auto *ctx = rewriter.getContext();
      auto targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, maskPatternTok(mp)),
      });
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TSCATTER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
          /*operands=*/ValueRange{dst, src});
    } else {
      Value idx = peelUnrealized(adaptor.getIndexes());
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TSCATTER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src, idx});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelToEmitC : public OpConversionPattern<pto::TSelOp> {
  using OpConversionPattern<pto::TSelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec5<Value> operands{dst, mask, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSEL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSELS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelSToEmitC : public OpConversionPattern<pto::TSelSOp> {
  using OpConversionPattern<pto::TSelSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSelSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src  = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec5<Value> operands{dst, mask, src, tmp, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSELS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShlSToEmitC : public OpConversionPattern<pto::TShlOp> {
  using OpConversionPattern<pto::TShlOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShrSToEmitC : public OpConversionPattern<pto::TShrOp> {
  using OpConversionPattern<pto::TShrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TSHLS/TSHRS DPS: shift by scalar)
//===----------------------------------------------------------------------===//

struct PTOShlSConstToEmitC : public OpConversionPattern<pto::TShlSOp> {
  using OpConversionPattern<pto::TShlSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVec3<Value> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHLS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOShrSConstToEmitC : public OpConversionPattern<pto::TShrSOp> {
  using OpConversionPattern<pto::TShrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVec3<Value> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHRS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (TSORT32 DPS/memref op: ins(src, idx[, tmp]) outs(dst))
//===----------------------------------------------------------------------===//

struct PTOSORT32SToEmitC : public OpConversionPattern<pto::TSort32Op> {
  using OpConversionPattern<pto::TSort32Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSort32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value tmp = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVec4<Value> operands;
    if (tmp)
      operands.assign({dst, src, idx, tmp});
    else
      operands.assign({dst, src, idx});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSORT32",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSqrtSToEmitC : public OpConversionPattern<pto::TSqrtOp> {
  using OpConversionPattern<pto::TSqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSTORE_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOStoreFPSToEmitC : public OpConversionPattern<pto::TStoreFPOp> {
  using OpConversionPattern<pto::TStoreFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TStoreFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value dstArg = dst;
    if (auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType())) {
      bool isGlobal = true;
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(dstMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, loc, dst, dstMrTy,
                                                  op.getOperation()))
          dstArg = gt;
      }
    }

    SmallVec3<Value> operands{dstArg, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSToEmitC : public OpConversionPattern<pto::TSubOp> {
  using OpConversionPattern<pto::TSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubCSToEmitC : public OpConversionPattern<pto::TSubCOp> {
  using OpConversionPattern<pto::TSubCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBC yet.
    // Decompose: dst = src0 - src1 + src2
    emitBinaryThenAccumulate(rewriter, loc, "TSUB", dst, src0, src1, src2);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSSToEmitC : public OpConversionPattern<pto::TSubSOp> {
  using OpConversionPattern<pto::TSubSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBSC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSCToEmitC : public OpConversionPattern<pto::TSubSCOp> {
  using OpConversionPattern<pto::TSubSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBSC yet.
    // Decompose: dst = src0 - scalar + src1
    emitBinaryThenAccumulate(rewriter, loc, "TSUBS", dst, src0, scalar, src1);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORToEmitC : public OpConversionPattern<pto::TXorOp> {
  using OpConversionPattern<pto::TXorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());
    Value tmp = peelUnrealized(adaptor.getTmp());
    SmallVec4<Value> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTTransToEmitC : public OpConversionPattern<pto::TTransOp> {
  using OpConversionPattern<pto::TTransOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTransOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRANS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORSToEmitC : public OpConversionPattern<pto::TXorSOp> {
  using OpConversionPattern<pto::TXorSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec4<Value> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOPrintToTPRINT : public OpConversionPattern<pto::TPrintOp> {
  using OpConversionPattern<pto::TPrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());

    SmallVec4<Value> operands{src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRINT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.print "format", %scalar -> PRINTF("format", scalar)

} // namespace

static void populatePTOToEmitCTileExtraCorePatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.add<PTOMovFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOQuantToEmitC>(typeConverter, ctx);
  patterns.add<PTODequantToEmitC>(typeConverter, ctx);
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulsToEmitC>(typeConverter, ctx);
  patterns.add<PTONegToEmitC>(typeConverter, ctx);
  patterns.add<PTONotToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrsToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartArgMaxToEmitC, PTOPartArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMulToEmitC>(typeConverter, ctx);
}

static void populatePTOToEmitCTileExtraRowPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.add<PTOPreluToEmitC>(typeConverter, ctx);
  patterns.add<PTORecipToEmitC>(typeConverter, ctx);
  patterns.add<PTOReluToEmitC>(typeConverter, ctx);
  patterns.add<PTORemToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModToEmitC>(typeConverter, ctx);
  patterns.add<PTORemSToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModSToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandAddToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowSumToEmitC>(typeConverter, ctx);
  patterns.add<PTORowProdToEmitC>(typeConverter, ctx);
}

static void populatePTOToEmitCTileExtraTailPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.add<PTORsqrtToEmitC>(typeConverter, ctx);
  patterns.add<PTOScatterToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOSORT32SToEmitC>(typeConverter, ctx);
  patterns.add<PTOSqrtSToEmitC>(typeConverter, ctx);
  patterns.add<PTOStoreFPSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubCSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSCToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTransToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPrintToTPRINT>(typeConverter, ctx);
  patterns.add<
      PTOTMatmulBiasToTMATMUL_BIAS,
      PTOTMatmulMXToTMATMUL_MX,
      PTOTMatmulMXAccToTMATMUL_MX_ACC,
      PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
      PTOTGemvBiasToTGEMV_BIAS,
      PTOTGemvMXToTGEMV_MX,
      PTOTGemvMXAccToTGEMV_MX,
      PTOTGemvMXBiasToTGEMV_MX>(typeConverter, ctx);
}

void populatePTOToEmitCTileExtraPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx) {
  populatePTOToEmitCTileExtraCorePatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCTileExtraRowPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCTileExtraTailPatterns(patterns, typeConverter, ctx);
}

} // namespace mlir::pto
