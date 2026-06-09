// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTilePatterns.cpp ----------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"
#include "PTOToEmitCTilePatternCommon.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

constexpr size_t kNumber2 = 2;

static LogicalResult lowerTDivSLikeOp(pto::TDivSOp op,
                                      pto::TDivSOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  Value src = peelUnrealized(adaptor.getSrc());
  Value scalar = peelUnrealized(adaptor.getScalar());
  Value dst = peelUnrealized(adaptor.getDst());
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TDIVS", ArrayAttr{},
                                       ArrayAttr{}, ValueRange{dst, src, scalar});
  rewriter.eraseOp(op);
  return success();
}

struct PTOTAndToEmitC : public OpConversionPattern<pto::TAndOp> {
  using OpConversionPattern<pto::TAndOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a   = peelUnrealized(adaptor.getSrc0());
    Value b   = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TAND",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, a, b});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOConcatToEmitC : public OpConversionPattern<pto::TConcatOp> {
  using OpConversionPattern<pto::TConcatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOConcatidxToEmitC : public OpConversionPattern<pto::TConcatidxOp> {
  using OpConversionPattern<pto::TConcatidxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatidxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOAndSToEmitC : public OpConversionPattern<pto::TAndSOp> {
  using OpConversionPattern<pto::TAndSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOTCIToEmitC : public OpConversionPattern<pto::TCIOp> {
  using OpConversionPattern<pto::TCIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value S = peelUnrealized(adaptor.getOperands()[0]);

    // The TCI scalar template parameter should follow the original PTO IR
    // scalar type, not the converted EmitC value type.
    std::string scalarTok = "int32_t";
    if (auto it = dyn_cast<IntegerType>(op->getOperand(0).getType())) {
      bool isUnsigned = it.isUnsigned();
      if (it.getWidth() == kPTOI16BitWidth)
        scalarTok = isUnsigned ? "uint16_t" : "int16_t";
      else
        scalarTok = isUnsigned ? "uint32_t" : "int32_t";
    }

    // descending -> "0"/"1"
    std::string descTok = op.getDescending() ? "1" : "0";

    ArrayAttr targs;
    if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      std::string tileTok = ot.getValue().str(); // "Tile<...>"
      targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, tileTok),
          emitc::OpaqueAttr::get(ctx, scalarTok),
          emitc::OpaqueAttr::get(ctx, descTok),
      });
    } else {
      targs = rewriter.getArrayAttr({});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCI",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, S});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string cmpModeTok(pto::CmpModeAttr a) {
  // 生成 "CmpMode::GT" 这种 token
  auto m = a.getValue(); // 取 enum
  switch (m) {
    case pto::CmpMode::EQ: return "CmpMode::EQ";
    case pto::CmpMode::NE: return "CmpMode::NE";
    case pto::CmpMode::LT: return "CmpMode::LT";
    case pto::CmpMode::LE: return "CmpMode::LE";
    case pto::CmpMode::GT: return "CmpMode::GT";
    case pto::CmpMode::GE: return "CmpMode::GE";
  }
  return "CmpMode::EQ";
}
struct PTOColExpandToEmitC : public OpConversionPattern<pto::TColExpandOp> {
  using OpConversionPattern<pto::TColExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPAND",
        /*args=*/ArrayAttr(),
        /*templateArgs=*/ArrayAttr(),
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMulToEmitC : public OpConversionPattern<pto::TColExpandMulOp> {
  using OpConversionPattern<pto::TColExpandMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMUL",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandAddToEmitC : public OpConversionPattern<pto::TColExpandAddOp> {
  using OpConversionPattern<pto::TColExpandAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDADD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandDivToEmitC : public OpConversionPattern<pto::TColExpandDivOp> {
  using OpConversionPattern<pto::TColExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDDIV",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandExpdifToEmitC
    : public OpConversionPattern<pto::TColExpandExpdifOp> {
  using OpConversionPattern<pto::TColExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDEXPDIF",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandSubToEmitC : public OpConversionPattern<pto::TColExpandSubOp> {
  using OpConversionPattern<pto::TColExpandSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDSUB",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMaxToEmitC : public OpConversionPattern<pto::TColExpandMaxOp> {
  using OpConversionPattern<pto::TColExpandMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMinToEmitC : public OpConversionPattern<pto::TColExpandMinOp> {
  using OpConversionPattern<pto::TColExpandMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTTriToEmitC : public OpConversionPattern<pto::TTriOp> {
  using OpConversionPattern<pto::TTriOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTriOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value diagonal = peelUnrealized(adaptor.getDiagonal());

    ArrayAttr templateArgs;
    if (auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, std::to_string(op.getUpperOrLower())),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVec2<Value> operands{dst, diagonal};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRI",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpToEmitC : public OpConversionPattern<pto::TCmpOp> {
  using OpConversionPattern<pto::TCmpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());

    std::string tok = "CmpMode::EQ";
    if (auto a = op.getCmpModeAttr())
      tok = cmpModeTok(a);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMP",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpSToEmitC : public OpConversionPattern<pto::TCmpSOp> {
  using OpConversionPattern<pto::TCmpSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCmpSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());

    // cmpMode -> token
    auto cmpAttr = op.getCmpModeAttr();          // PTO_CmpModeAttr
    std::string tok = cmpModeTok(cmpAttr);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMPS",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOColMaxToEmitC : public OpConversionPattern<pto::TColMaxOp> {
  using OpConversionPattern<pto::TColMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMAX(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLMAX",
        /*args=*/ArrayAttr{},          // default: print all operands
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMaxToEmitC : public OpConversionPattern<pto::TColArgMaxOp> {
  using OpConversionPattern<pto::TColArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColMinToEmitC : public OpConversionPattern<pto::TColMinOp> {
  using OpConversionPattern<pto::TColMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMIN(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLMIN",
        /*args=*/ArrayAttr{},          // default: print all operands
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMinToEmitC : public OpConversionPattern<pto::TColArgMinOp> {
  using OpConversionPattern<pto::TColArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColSumToEmitC : public OpConversionPattern<pto::TColSumOp> {
  using OpConversionPattern<pto::TColSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // Check if tmp exists before accessing it
    if (op.getTmp()) {
      // Format 2: with tmp and isBinary
      Value tmp = peelUnrealized(adaptor.getTmp());
      bool isBinary = false;
      if (auto a = op.getIsBinaryAttr())
        isBinary = a.getValue();

      auto boolTy = emitc::OpaqueType::get(ctx, "bool");
      auto tok = isBinary ? "true" : "false";
      Value isBinaryVal = rewriter.create<emitc::ConstantOp>(
          loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TCOLSUM",
          /*args=*/ArrayAttr(),
          /*templateArgs=*/ArrayAttr(),
          /*operands=*/ValueRange{dst, src, tmp, isBinaryVal});
    } else {
      // Format 1: without tmp and isBinary
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TCOLSUM",
          /*args=*/ArrayAttr(),
          /*templateArgs=*/ArrayAttr(),
          /*operands=*/ValueRange{dst, src});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColProdToEmitC : public OpConversionPattern<pto::TColProdOp> {
  using OpConversionPattern<pto::TColProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLPROD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string roundModeTok(mlir::pto::RoundModeAttr attr) {
  using RM = mlir::pto::RoundMode;
  switch (attr.getValue()) {
  case RM::NONE:      return "RoundMode::CAST_NONE";
  case RM::RINT:      return "RoundMode::CAST_RINT";
  case RM::ROUND:     return "RoundMode::CAST_ROUND";
  case RM::FLOOR:     return "RoundMode::CAST_FLOOR";
  case RM::CEIL:      return "RoundMode::CAST_CEIL";
  case RM::TRUNC:     return "RoundMode::CAST_TRUNC";
  case RM::ODD:       return "RoundMode::CAST_ODD";
  case RM::CAST_RINT: return "RoundMode::CAST_RINT";
  }
  return "RoundMode::CAST_RINT";
}
static std::string saturationModeTok(mlir::pto::SaturationModeAttr attr) {
  using SM = mlir::pto::SaturationMode;
  switch (attr.getValue()) {
  case SM::ON:  return "SaturationMode::ON";
  case SM::OFF: return "SaturationMode::OFF";
  }
  return "SaturationMode::OFF";
}
struct PTOCvtToEmitC : public OpConversionPattern<pto::TCvtOp> {
  using OpConversionPattern<pto::TCvtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCvtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    pto::RoundModeAttr rmAttr = op.getRmodeAttr();
    std::string rmTok = rmAttr ? roundModeTok(rmAttr)
                               : std::string("RoundMode::CAST_RINT");
    auto rmodeTy = emitc::OpaqueType::get(ctx, "RoundMode");
    Value rmodeVal = rewriter.create<emitc::ConstantOp>(
        loc, rmodeTy, emitc::OpaqueAttr::get(ctx, rmTok));

    auto satModeTy = emitc::OpaqueType::get(ctx, "SaturationMode");
    auto satAttr = op.getSatModeAttr();
    std::string satTok = satAttr ? saturationModeTok(satAttr)
                                 : std::string("SaturationMode::OFF");
    Value satModeVal = rewriter.create<emitc::ConstantOp>(
        loc, satModeTy, emitc::OpaqueAttr::get(ctx, satTok));

    SmallVec4<Value> operands{dst, src, rmodeVal, satModeVal};

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCVT",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTORandomToEmitC : public OpConversionPattern<pto::TRandomOp> {
  using OpConversionPattern<pto::TRandomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRandomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    SmallVec7<Value> operands{
        dst,
        peelUnrealized(adaptor.getKey0()),
        peelUnrealized(adaptor.getKey1()),
        peelUnrealized(adaptor.getCounter0()),
        peelUnrealized(adaptor.getCounter1()),
        peelUnrealized(adaptor.getCounter2()),
        peelUnrealized(adaptor.getCounter3()),
    };
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, std::to_string(op.getRounds()))});

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "PTOAS__TRANDOM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdiv lowering -> TDIV(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTODivToTDIV : public OpConversionPattern<pto::TDivOp> {
  using OpConversionPattern<pto::TDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TDIV",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdivs lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTODivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return lowerTDivSLikeOp(op, adaptor, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// pto.tdivs (TDivSOp) lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTOTDivSToEmitC : public PTODivSToEmitC {
  using PTODivSToEmitC::PTODivSToEmitC;
};
//===----------------------------------------------------------------------===//
// pto.texp lowering -> TEXP(dst, src)
//===----------------------------------------------------------------------===//

struct PTOExpToEmitC : public OpConversionPattern<pto::TExpOp> {
  using OpConversionPattern<pto::TExpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXP",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texpands lowering -> TEXPANDS(dst, scalar)
//===----------------------------------------------------------------------===//

struct PTOExpandsToEmitC : public OpConversionPattern<pto::TExpandsOp> {
  using OpConversionPattern<pto::TExpandsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpandsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXPANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract lowering -> TEXTRACT(dst, src, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractToEmitC : public OpConversionPattern<pto::TExtractOp> {
  using OpConversionPattern<pto::TExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract_fp lowering -> TEXTRACT_FP(dst, src, fp, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractFPToEmitC : public OpConversionPattern<pto::TExtractFPOp> {
  using OpConversionPattern<pto::TExtractFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExtractFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0 = peelUnrealized(adaptor.getIndexRow());
    Value c0 = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, fp, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tinsert lowering -> TINSERT(dst, src, indexRow, indexCol)
// Keep lowering arch-agnostic and let PTO-ISA infer proper A5 path.
//===----------------------------------------------------------------------===//

struct PTOInsertToEmitC : public OpConversionPattern<pto::TInsertOp> {
  using OpConversionPattern<pto::TInsertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TInsertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TINSERT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tinsert_fp lowering -> TINSERT_FP(dst, src, fp, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOInsertFPToEmitC : public OpConversionPattern<pto::TInsertFPOp> {
  using OpConversionPattern<pto::TInsertFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TInsertFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0 = peelUnrealized(adaptor.getIndexRow());
    Value c0 = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TINSERT_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, fp, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad lowering -> TFILLPAD(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadToEmitC : public OpConversionPattern<pto::TFillPadOp> {
  using OpConversionPattern<pto::TFillPadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad_inplace lowering -> TFILLPAD_INPLACE(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadInplaceToEmitC
    : public OpConversionPattern<pto::TFillPadInplaceOp> {
  using OpConversionPattern<pto::TFillPadInplaceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadInplaceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_INPLACE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad_expand lowering -> TFILLPAD_EXPAND(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadExpandToEmitC
    : public OpConversionPattern<pto::TFillPadExpandOp> {
  using OpConversionPattern<pto::TFillPadExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_EXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tgather lowering
// - Index form  : TGATHER(dst, src0, indices, tmp)
// - Compare form: TGATHER<DstT, SrcT, CDstT, TmpT, CmpMode::GT, 7>(dst, src0, kValue, cdst, tmp)
// - Mask form : TGATHER<dstTileTok, srcTileTok, pto::MaskPattern::Pxxxx>(dst, src0)
//===----------------------------------------------------------------------===//

[[maybe_unused]] static std::string maskPatternTok(mlir::pto::MaskPatternAttr a) {
  auto v = a.getValue(); // enum
  return (std::string("pto::MaskPattern::") + mlir::pto::stringifyMaskPattern(v).str());
}

struct PTOGatherToEmitC : public OpConversionPattern<pto::TGatherOp> {
  using OpConversionPattern<pto::TGatherOp>::OpConversionPattern;

  static FailureOr<std::string> getOpaqueTok(ConversionPatternRewriter &rewriter,
                                             pto::TGatherOp op, Value value,
                                             StringRef name) {
    if (auto opaqueTy = mlir::dyn_cast<emitc::OpaqueType>(value.getType()))
      return opaqueTy.getValue().str();
    return rewriter.notifyMatchFailure(
        op, (name + " must be emitc::OpaqueType (tile)").str());
  }

  static void rewriteIndexForm(ConversionPatternRewriter &rewriter,
                               Location loc, pto::TGatherOp op, Value dst,
                               Value src0, Value idx, Value tmp) {
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TGATHER",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{dst, src0, idx,
                                                                 tmp});
    rewriter.eraseOp(op);
  }

  static LogicalResult rewriteCompareForm(ConversionPatternRewriter &rewriter,
                                          Location loc, MLIRContext *ctx,
                                          pto::TGatherOp op, Value dst,
                                          Value src0, Value cdst, Value tmp,
                                          Value kValue) {
    auto dstTokOr = getOpaqueTok(rewriter, op, dst, "dst");
    auto srcTokOr = getOpaqueTok(rewriter, op, src0, "src0");
    auto cdstTokOr = getOpaqueTok(rewriter, op, cdst, "cdst");
    auto tmpTokOr = getOpaqueTok(rewriter, op, tmp, "tmp");
    if (failed(dstTokOr) || failed(srcTokOr) || failed(cdstTokOr) ||
        failed(tmpTokOr))
      return failure();

    auto cmpAttr = op.getCmpModeAttr();
    std::string cmpTok = cmpAttr ? cmpModeTok(cmpAttr) : "CmpMode::EQ";
    int64_t offset = 0;
    if (auto offsetAttr = op.getOffsetAttr())
      offset = offsetAttr.getInt();
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value offsetVal = makeEmitCIntConstant(rewriter, loc, i32Ty, offset);
    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, *tmpTokOr),
        emitc::OpaqueAttr::get(ctx, *cdstTokOr),
        emitc::OpaqueAttr::get(ctx, cmpTok),
    });
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER", /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0, kValue, tmp, cdst, offsetVal});
    rewriter.eraseOp(op);
    return success();
  }

  static LogicalResult rewriteMaskForm(ConversionPatternRewriter &rewriter,
                                       Location loc, MLIRContext *ctx,
                                       pto::TGatherOp op, Value dst,
                                       Value src0) {
    auto mp = op.getMaskPatternAttr();
    if (!mp)
      return rewriter.notifyMatchFailure(
          op, "expected maskPattern, indices, or cdst on tgather");

    auto dstTokOr = getOpaqueTok(rewriter, op, dst, "dst");
    auto srcTokOr = getOpaqueTok(rewriter, op, src0, "src0");
    if (failed(dstTokOr) || failed(srcTokOr))
      return failure();

    std::string mpTok = std::string("MaskPattern::") +
                        mlir::pto::stringifyMaskPattern(mp.getValue()).str();
    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, mpTok),
    });
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TGATHER",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/targs,
                                         /*operands=*/ValueRange{dst, src0});
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult matchAndRewrite(pto::TGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc());

    // Case 1: index-based TGATHER(dst, src0, indices, tmp)
    if (Value idx = adaptor.getIndices()) {
      idx = peelUnrealized(idx);
      Value tmp = peelUnrealized(adaptor.getTmp());
      rewriteIndexForm(rewriter, loc, op, dst, src0, idx, tmp);
      return success();
    }

    // Case 2: compare-based TGATHER<DstT, SrcT, TmpT, CDstT, CmpMode::GT>(
    //            dst, src0, kValue, tmp, cdst, offset)
    if (Value cdst = adaptor.getCdst()) {
      cdst = peelUnrealized(cdst);
      Value tmp = peelUnrealized(adaptor.getTmp());
      Value kValue = peelUnrealized(adaptor.getKValue());
      return rewriteCompareForm(rewriter, loc, ctx, op, dst, src0, cdst, tmp,
                                kValue);
    }

    // Case 3: mask-pattern TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src0)
    return rewriteMaskForm(rewriter, loc, ctx, op, dst, src0);
  }
};


struct PTOGatherbToEmitC : public OpConversionPattern<pto::TGatherBOp> {
  using OpConversionPattern<pto::TGatherBOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherBOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src     = peelUnrealized(adaptor.getSrc());
    Value offsets = peelUnrealized(adaptor.getOffsets());
    Value dst     = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHERB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, offsets});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TLOG lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOLogToEmitC : public OpConversionPattern<pto::TLogOp> {
  using OpConversionPattern<pto::TLogOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLogOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVec2<Value> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLOG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TLRELU lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOLReluToEmitC : public OpConversionPattern<pto::TLReluOp> {
	  using OpConversionPattern<pto::TLReluOp>::OpConversionPattern;

	  LogicalResult matchAndRewrite(pto::TLReluOp op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();

	    Value src = peelUnrealized(adaptor.getSrc());
	    Value slope = peelUnrealized(adaptor.getSlope());
	    Value dst = peelUnrealized(adaptor.getDst());

            SmallVec3<Value> operands{dst, src, slope};

	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TLRELU",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
	        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAX lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMaxToEmitC : public OpConversionPattern<pto::TMaxOp> {
  using OpConversionPattern<pto::TMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAXS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOMaxSToEmitC : public OpConversionPattern<pto::TMaxSOp> {
	  using OpConversionPattern<pto::TMaxSOp>::OpConversionPattern;

	  LogicalResult matchAndRewrite(pto::TMaxSOp op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();

	    Value src0 = peelUnrealized(adaptor.getSrc());
	    Value scalar = peelUnrealized(adaptor.getScalar());
	    Value dst  = peelUnrealized(adaptor.getDst());

	    SmallVec3<Value> operands{dst, src0, scalar};
	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TMAXS",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// TMIN lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinToEmitC : public OpConversionPattern<pto::TMinOp> {
  using OpConversionPattern<pto::TMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVec3<Value> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (fix APFloat -> FloatAttr)  (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinsToEmitC : public OpConversionPattern<pto::TMinSOp> {
  using OpConversionPattern<pto::TMinSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMinSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVec3<Value> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMINS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TMOV op -> EmitC)
//===----------------------------------------------------------------------===//

struct PTOMovToEmitC : public OpConversionPattern<pto::TMovOp> {
  using OpConversionPattern<pto::TMovOp>::OpConversionPattern;

  static StringRef modeTok(pto::AccToVecMode mode) {
    switch (mode) {
    case pto::AccToVecMode::SingleModeVec0:
      return "pto::AccToVecMode::SingleModeVec0";
    case pto::AccToVecMode::SingleModeVec1:
      return "pto::AccToVecMode::SingleModeVec1";
    case pto::AccToVecMode::DualModeSplitM:
      return "pto::AccToVecMode::DualModeSplitM";
    case pto::AccToVecMode::DualModeSplitN:
      return "pto::AccToVecMode::DualModeSplitN";
    }
    llvm_unreachable("unknown AccToVecMode");
  }

  static StringRef reluTok(pto::ReluPreMode mode) {
    switch (mode) {
    case pto::ReluPreMode::NoRelu:
      return "ReluPreMode::NoRelu";
    case pto::ReluPreMode::NormalRelu:
      return "ReluPreMode::NormalRelu";
    }
    llvm_unreachable("unknown ReluPreMode");
  }

  static LogicalResult appendTMovFpOrScalarArgs(
      ConversionPatternRewriter &rewriter, pto::TMovOp op, Value fp,
      Value preQuantScalar, StringRef &callee,
      SmallVectorImpl<Value> &operands,
      SmallVectorImpl<Attribute> &templateArgVec) {
    auto *ctx = rewriter.getContext();
    auto modeAttr = op.getAccToVecModeAttr();
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (fp) {
      auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
      if (!fpOT)
        return rewriter.notifyMatchFailure(
            op, "tmov fp lowering expects opaque fp type");
      operands.push_back(fp);
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
      if (hasMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      if (hasMode || reluNonDefault)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
      callee = hasMode ? "TMOV" : "TMOV_FP";
      return success();
    }

    if (!preQuantScalar)
      return success();
    operands.push_back(preQuantScalar);
    if (hasMode)
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
    if (hasMode || reluNonDefault)
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    return success();
  }

  static FailureOr<ArrayAttr> buildTMovTemplateArgs(
      ConversionPatternRewriter &rewriter, pto::TMovOp op, Value src, Value dst,
      Value fp, Value preQuantScalar, StringRef &callee,
      SmallVectorImpl<Value> &operands) {
    auto *ctx = rewriter.getContext();
    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    if (!dstOT || !srcOT)
      return rewriter.notifyMatchFailure(
          op, "tmov lowering expects opaque dst/src types");

    auto modeAttr = op.getAccToVecModeAttr();
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuant = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;

    operands.assign({dst, src});
    SmallVec5<Attribute> templateArgVec{
        emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
        emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
    };
    callee = "TMOV";
    if (failed(appendTMovFpOrScalarArgs(rewriter, op, fp, preQuantScalar, callee,
                                        operands, templateArgVec)))
      return failure();
    if (!hasFp && !hasPreQuant && hasMode) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    } else if (reluNonDefault) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    }

    if (templateArgVec.size() == kNumber2 && !hasFp && !hasPreQuant && !hasMode &&
        !reluNonDefault)
      return ArrayAttr{};
    return rewriter.getArrayAttr(templateArgVec);
  }

  LogicalResult matchAndRewrite(pto::TMovOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value fp;
    if (op.getFp())
      fp = peelUnrealized(adaptor.getFp());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());
    StringRef callee = "TMOV";
    SmallVec4<Value> operands;
    FailureOr<ArrayAttr> templateArgs = buildTMovTemplateArgs(
        rewriter, op, src, dst, fp, preQuantScalar, callee, operands);
    if (failed(templateArgs))
      return failure();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, callee,
        /*args=*/ArrayAttr{}, /*templateArgs=*/*templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

void populatePTOToEmitCTilePatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx) {
  patterns.add<PTOColExpandAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColProdToEmitC>(typeConverter, ctx);
  patterns.add<PTODivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOConcatToEmitC, PTOConcatidxToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpandsToEmitC>(typeConverter, ctx);
  patterns.add<PTOExtractToEmitC, PTOExtractFPToEmitC, PTOInsertToEmitC,
               PTOInsertFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOFillPadToEmitC, PTOFillPadInplaceToEmitC, PTOFillPadExpandToEmitC>(
      typeConverter, ctx);
  patterns.add<PTOGatherToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherbToEmitC>(typeConverter, ctx);
  patterns.add<PTOLogToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovToEmitC>(typeConverter, ctx);
  patterns.add<PTOTCIToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTriToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTORandomToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinsToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinToEmitC>(typeConverter, ctx);
  populatePTOToEmitCTileExtraPatterns(patterns, typeConverter, ctx);
}

} // namespace mlir::pto
