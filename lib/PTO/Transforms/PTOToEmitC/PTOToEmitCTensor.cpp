// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTensor.cpp - tensor elementwise op lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

void populateTensorPatternsPart2(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx, PTOArch targetArch);

struct PTOTAddCToTADDC : public OpConversionPattern<pto::TAddCOp> {
  using OpConversionPattern<pto::TAddCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TADDC yet.
    // Decompose: dst = src0 + src1 + src2
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TADD",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getSrc1(), adaptor.getSrc2());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadds lowering -> TADDS(dst, src, scalar)
//===----------------------------------------------------------------------===//

struct PTOAddSToTADDS : public OpConversionPattern<pto::TAddSOp> {
  using OpConversionPattern<pto::TAddSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = adaptor.getSrc();
    Value dst    = adaptor.getDst();
    Value scalar = adaptor.getScalar();

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TADDS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddsc lowering -> TADDSC(dst, src0, scalar, src1)
//===----------------------------------------------------------------------===//

struct PTOAddSCToTADDSC : public OpConversionPattern<pto::TAddSCOp> {
  using OpConversionPattern<pto::TAddSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // pto-isa does not provide NPU implementation for TADDSC yet.
    // Decompose: dst = src0 + scalar + src1
    emitDecomposedPairAndErase(op.getOperation(), rewriter, "TADDS",
                               adaptor.getDst(), adaptor.getSrc0(),
                               adaptor.getScalar(), adaptor.getSrc1());
    return success();
  }
};
struct PTOTAndToEmitC : public OpConversionPattern<pto::TAndOp> {
  using OpConversionPattern<pto::TAndOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a   = adaptor.getSrc0();
    Value b   = adaptor.getSrc1();
    Value dst = adaptor.getDst();

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
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

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
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value src0Idx = adaptor.getSrc0Idx();
    Value src1Idx = adaptor.getSrc1Idx();
    Value dst  = adaptor.getDst();

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
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst    = adaptor.getDst();

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

    Value dst = adaptor.getDst();
    Value S = adaptor.getOperands()[0];
    Value tmp = op.getTmp() ? adaptor.getTmp() : Value();

    // The TCI scalar template parameter should follow the original PTO IR
    // scalar type, not the converted EmitC value type.
    std::string scalarTok = "int32_t";
    if (auto it = dyn_cast<IntegerType>(op->getOperand(0).getType())) {
      bool isUnsigned = it.isUnsigned();
      if (it.getWidth() == 16) {
        scalarTok = isUnsigned ? "uint16_t" : "int16_t";
      } else {
        scalarTok = isUnsigned ? "uint32_t" : "int32_t";
      }
    }

    // descending -> "0"/"1"
    std::string descTok = op.getDescending() ? "1" : "0";

    ArrayAttr targs;
    if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      SmallVector<Attribute, 4> templateArgVec;
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
      if (tmp) {
        auto tmpOt = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
        if (!tmpOt)
          return rewriter.notifyMatchFailure(
              op, "expected tmp tile to lower to emitc::OpaqueType");
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, tmpOt.getValue().str()));
      }
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, scalarTok));
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, descTok));
      targs = rewriter.getArrayAttr(templateArgVec);
    } else {
      targs = rewriter.getArrayAttr({});
    }

    SmallVector<Value, 3> operands{dst, S};
    if (tmp)
      operands.push_back(tmp);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCI",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/operands);

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

    Value dst = adaptor.getDst();
    Value src = adaptor.getSrc();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPAND",
        /*args=*/ArrayAttr(),           
        /*templateArgs=*/ArrayAttr(),
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandDivToEmitC : public OpConversionPattern<pto::TColExpandDivOp> {
  using OpConversionPattern<pto::TColExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::DivPrecision::Default,
        "DivAlgorithm");

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDDIV",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

// Binary col-expand ops (add/sub/mul/div/max/min): emit
// TCOLEXPAND<OP>(dst, src0, src1).
template <typename OpTy>
struct PTOColExpandBinaryToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTOColExpandBinaryToEmitC(TypeConverter &typeConverter,
                                     MLIRContext *ctx, StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, callee,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

using PTOColExpandMulToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandMulOp>;
using PTOColExpandAddToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandAddOp>;
using PTOColExpandSubToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandSubOp>;
using PTOColExpandMaxToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandMaxOp>;
using PTOColExpandMinToEmitC = PTOColExpandBinaryToEmitC<pto::TColExpandMinOp>;



struct PTOColExpandExpdifToEmitC
    : public OpConversionPattern<pto::TColExpandExpdifOp> {
  using OpConversionPattern<pto::TColExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDEXPDIF",
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

    Value dst = adaptor.getDst();
    Value diagonal = adaptor.getDiagonal();

    ArrayAttr templateArgs;
    if (auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, std::to_string(op.getUpperOrLower())),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 2> operands{dst, diagonal};
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
  
    Value dst  = adaptor.getDst();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();

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

    Value dst    = adaptor.getDst();
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();

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
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // intrinsic: TCOLMAX(dst, src)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TCOLMAX", ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMaxToEmitC : public OpConversionPattern<pto::TColArgMaxOp> {
  using OpConversionPattern<pto::TColArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

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
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // intrinsic: TCOLMIN(dst, src)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TCOLMIN", ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMinToEmitC : public OpConversionPattern<pto::TColArgMinOp> {
  using OpConversionPattern<pto::TColArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value tmp = adaptor.getTmp();
    Value dst = adaptor.getDst();

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

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    // Check if tmp exists before accessing it
    if (op.getTmp()) {
      // Format 2: with tmp and isBinary
      Value tmp = adaptor.getTmp();
      bool isBinary = false;
      if (auto a = op.getIsBinaryAttr())
        isBinary = a.getValue();

      auto boolTy = emitc::OpaqueType::get(ctx, "bool");
      auto tok = isBinary ? "true" : "false";
      Value isBinaryVal = rewriter.create<emitc::ConstantOp>(
          loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

      SmallVector<unsigned, 3> tileSlotOrder;
      tileSlotOrder.push_back(op.getDstMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getSrcMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getTmpMutable().begin()->getOperandNumber());

      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TCOLSUM",
          ValueRange{dst, src, tmp, isBinaryVal}, ArrayAttr{}, ArrayAttr{},
          tileSlotOrder);
    } else {
      // Format 1: without tmp and isBinary
      SmallVector<unsigned, 2> tileSlotOrder;
      tileSlotOrder.push_back(op.getDstMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getSrcMutable().getOperandNumber());
      createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                   "TCOLSUM", ValueRange{dst, src},
                                   ArrayAttr{}, ArrayAttr{}, tileSlotOrder);
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

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

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

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

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

    SmallVector<Value, 5> operands{dst, src};
    if (adaptor.getTmp())
      operands.push_back(peelUnrealized(adaptor.getTmp()));
    operands.push_back(rmodeVal);
    operands.push_back(satModeVal);

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

    Value dst = adaptor.getDst();
    SmallVector<Value, 7> operands{
        dst,
        adaptor.getKey0(),
        adaptor.getKey1(),
        adaptor.getCounter0(),
        adaptor.getCounter1(),
        adaptor.getCounter2(),
        adaptor.getCounter3(),
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
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value dst  = adaptor.getDst();

    ArrayAttr templateArgs = buildPrecisionTemplateArgs(
        rewriter, op.getPrecisionType(), pto::DivPrecision::Default,
        "DivAlgorithm");

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIV", ValueRange{dst, src0, src1}, ArrayAttr{}, templateArgs);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdivs lowering -> TDIVS(dst, src, scalar)
// Preserve source order from textual parse:
// ins(tile, scalar)   -> TDIVS(dst, tile, scalar)
// ins(scalar, tile)   -> TDIVS(dst, scalar, tile)
//===----------------------------------------------------------------------===//

struct PTOTDivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst    = adaptor.getDst();
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIVS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texp lowering -> TEXP(dst, src)
//===----------------------------------------------------------------------===//

struct PTOExpToEmitC : public OpConversionPattern<pto::TExpOp> {
  using OpConversionPattern<pto::TExpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::ExpPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::ExpPrecision::Default:
        precisionTok = "pto::ExpAlgorithm::DEFAULT";
        break;
      case pto::ExpPrecision::HighPrecision:
        precisionTok = "pto::ExpAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TEXP", ValueRange{dst, src}, ArrayAttr{}, templateArgs);

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
    Value scalar = adaptor.getScalar();
    Value dst    = adaptor.getDst();

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TEXPANDS", ValueRange{dst, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract lowering -> TEXTRACT(dst, src, indexRow, indexCol)
//===----------------------------------------------------------------------===//

static StringRef getReluPreModeToken(pto::ReluPreMode mode) {
  switch (mode) {
  case pto::ReluPreMode::NoRelu:
    return "ReluPreMode::NoRelu";
  case pto::ReluPreMode::NormalRelu:
    return "ReluPreMode::NormalRelu";
  case pto::ReluPreMode::ScalarRelu:
    return "ReluPreMode::ScalarRelu";
  case pto::ReluPreMode::VectorRelu:
    return "ReluPreMode::VectorRelu";
  case pto::ReluPreMode::Pwl:
    return "ReluPreMode::Pwl";
  }
  llvm_unreachable("unknown ReluPreMode");
}

static StringRef getAccToVecModeToken(pto::AccToVecMode mode) {
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

static StringRef getTInsertModeToken(pto::TInsertMode mode) {
  switch (mode) {
  case pto::TInsertMode::SPLIT2:
    return "pto::TInsertMode::SPLIT2";
  case pto::TInsertMode::SPLIT4:
    return "pto::TInsertMode::SPLIT4";
  }
  llvm_unreachable("unknown TInsertMode");
}

struct PTOExtractToEmitC : public OpConversionPattern<pto::TExtractOp> {
  using OpConversionPattern<pto::TExtractOp>::OpConversionPattern;

  static SmallVector<Value, 6> collectOperands(OpAdaptor adaptor) {
    SmallVector<Value, 6> operands{adaptor.getDst(), adaptor.getSrc()};
    if (Value fp = adaptor.getFp()) {
      operands.push_back(fp);
    }
    if (Value preQuantScalar = adaptor.getPreQuantScalar()) {
      operands.push_back(preQuantScalar);
    }
    operands.push_back(adaptor.getIndexRow());
    operands.push_back(adaptor.getIndexCol());
    return operands;
  }

  static FailureOr<ArrayAttr>
  buildTemplateArgs(pto::TExtractOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) {
    auto modeAttr = op.getAccToVecModeAttr();
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (!hasMode && !hasRelu) {
      return ArrayAttr{};
    }

    auto dstType = dyn_cast<emitc::OpaqueType>(adaptor.getDst().getType());
    auto srcType = dyn_cast<emitc::OpaqueType>(adaptor.getSrc().getType());
    if (!dstType || !srcType) {
      return failure();
    }
    SmallVector<Attribute, 4> args{
        emitc::OpaqueAttr::get(rewriter.getContext(), dstType.getValue().str()),
        emitc::OpaqueAttr::get(rewriter.getContext(), srcType.getValue().str()),
    };
    if (Value fp = adaptor.getFp()) {
      auto fpType = dyn_cast<emitc::OpaqueType>(fp.getType());
      if (!fpType) {
        return failure();
      }
      args.push_back(
          emitc::OpaqueAttr::get(rewriter.getContext(), fpType.getValue().str()));
    }
    if (hasMode) {
      args.push_back(emitc::OpaqueAttr::get(
          rewriter.getContext(), getAccToVecModeToken(modeAttr.getValue())));
    }
    args.push_back(emitc::OpaqueAttr::get(
        rewriter.getContext(), getReluPreModeToken(op.getReluPreMode())));
    return rewriter.getArrayAttr(args);
  }

  LogicalResult matchAndRewrite(pto::TExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto templateArgs = buildTemplateArgs(op, adaptor, rewriter);
    if (failed(templateArgs)) {
      return rewriter.notifyMatchFailure(
          op, "textract template lowering expects opaque dst/src/fp types");
    }

    const bool hasFp = static_cast<bool>(adaptor.getFp());
    const bool hasMode = static_cast<bool>(op.getAccToVecModeAttr());
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, hasFp && !hasMode ? "TEXTRACT_FP" : "TEXTRACT",
        ArrayAttr{}, *templateArgs, collectOperands(adaptor));
    rewriter.eraseOp(op);
    return success();
  }
};

// Append the acc-to-vec mode (when present) and relu-pre-mode template
// tokens shared by the TMOV overload family.
static void pushModeAndReluTemplateArgs(SmallVectorImpl<Attribute> &args,
                                        MLIRContext *ctx,
                                        pto::AccToVecModeAttr modeAttr,
                                        bool reluNonDefault,
                                        pto::ReluPreMode reluPreMode) {
  if (modeAttr)
    args.push_back(emitc::OpaqueAttr::get(
        ctx, getAccToVecModeToken(modeAttr.getValue())));
  if (modeAttr || reluNonDefault)
    args.push_back(
        emitc::OpaqueAttr::get(ctx, getReluPreModeToken(reluPreMode)));
}

struct PTOInsertToEmitC : public OpConversionPattern<pto::TInsertOp> {
  using OpConversionPattern<pto::TInsertOp>::OpConversionPattern;

  // Resolve the TINSERT overload's template arguments: the tinsert mode when
  // present, otherwise the spelled-out dst/src (and fp) types plus mode/relu
  // tokens when any optional operand is in play.
  LogicalResult buildTemplateArgs(pto::TInsertOp op,
                                  ConversionPatternRewriter &rewriter,
                                  Value dst, Value src, Value fp, Value preQuantScalar,
                                  ArrayAttr &templateArgs) const {
    auto *ctx = rewriter.getContext();
    auto modeAttr = op.getAccToVecModeAttr();
    auto tinsertModeAttr = op.getTinsertModeAttr();
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault =
        op.getReluPreMode() != pto::ReluPreMode::NoRelu;

    templateArgs = ArrayAttr{};
    if (tinsertModeAttr) {
      templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
          ctx, getTInsertModeToken(tinsertModeAttr.getValue()))});
    } else if (hasFp || hasPreQuantScalar || hasMode || reluNonDefault) {
      auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
      auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
      if (!dstOT || !srcOT)
        return rewriter.notifyMatchFailure(
            op, "tinsert template lowering expects opaque dst/src types");
      SmallVector<Attribute, 5> args{
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
      };
      if (hasFp) {
        auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
        if (!fpOT)
          return rewriter.notifyMatchFailure(
              op, "tinsert template lowering expects opaque fp type");
        args.push_back(emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
      }
      pushModeAndReluTemplateArgs(args, ctx, modeAttr, reluNonDefault,
                                  op.getReluPreMode());
      // TINSERT always spells out the relu-pre-mode token.
      if (!modeAttr && !reluNonDefault)
        args.push_back(emitc::OpaqueAttr::get(
            ctx, getReluPreModeToken(op.getReluPreMode())));
      templateArgs = rewriter.getArrayAttr(args);
    }
    if (hasFp && !hasMode && !reluNonDefault)
      templateArgs = ArrayAttr{};
    return success();
  }

  LogicalResult matchAndRewrite(pto::TInsertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value r0 = adaptor.getIndexRow();
    Value c0 = adaptor.getIndexCol();
    Value fp;
    if (op.getFp())
      fp = adaptor.getFp();
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = adaptor.getPreQuantScalar();

    SmallVector<Value, 6> operands{dst, src};
    if (fp)
      operands.push_back(fp);
    if (preQuantScalar)
      operands.push_back(preQuantScalar);
    operands.push_back(r0);
    operands.push_back(c0);

    ArrayAttr templateArgs;
    if (failed(buildTemplateArgs(op, rewriter, dst, src, fp, preQuantScalar,
                                 templateArgs)))
      return failure();

    const bool hasMode = static_cast<bool>(op.getAccToVecModeAttr());
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, fp && !hasMode ? "TINSERT_FP" : "TINSERT",
        ArrayAttr{}, templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }

};

static StringRef getTFillPadModeToken(pto::TFillPadLoweringKind loweringKind) {
  switch (loweringKind) {
  case pto::TFillPadLoweringKind::Normal:
    return "pto::TFillPadMode::Normal";
  case pto::TFillPadLoweringKind::InPlace:
    return "pto::TFillPadMode::InPlace";
  case pto::TFillPadLoweringKind::Expand:
    return "pto::TFillPadMode::Expand";
  }
  llvm_unreachable("unknown TFillPadLoweringKind");
}

struct PTOFillPadToEmitC : public OpConversionPattern<pto::TFillPadOp> {
  using OpConversionPattern<pto::TFillPadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    auto loweringKind = pto::inferTFillPadLoweringKindAfterMemoryPlanning(op);
    if (failed(loweringKind)) {
      op.emitOpError(
          "cannot infer a supported lowering; expand and in-place forms "
          "require loc=vec, statically comparable physical shapes, and "
          "resolved planned addresses");
      return failure();
    }

    auto padValueTok = [&](pto::PadValue mode) -> StringRef {
      switch (mode) {
      case pto::PadValue::Null:
        return "pto::PadValue::Null";
      case pto::PadValue::Zero:
        return "pto::PadValue::Zero";
      case pto::PadValue::Max:
        return "pto::PadValue::Max";
      case pto::PadValue::Min:
        return "pto::PadValue::Min";
      }
      llvm_unreachable("unknown PadValue");
    };

    ArrayAttr templateArgs{};
    if (auto padValueAttr = op.getPadValueAttr()) {
      // The verifier only accepts explicit padValue for loc=mat tile-form
      // tfillpad, so lowering can trust the preserved semantic contract.
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, padValueTok(padValueAttr.getValue()))});
    } else if (*loweringKind != pto::TFillPadLoweringKind::Normal) {
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, getTFillPadModeToken(*loweringKind))});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
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


struct PTOGatherToEmitC : public OpConversionPattern<pto::TGatherOp> {
  using OpConversionPattern<pto::TGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Case 1: index-based TGATHER(dst, src0, indices[, tmp])
    if (Value idx = adaptor.getIndices())
      return emitIndexedGather(op, adaptor, rewriter, idx);

    // Case 2: compare-based TGATHER<DstT, SrcT, TmpT, CDstT, CmpMode::GT>(
    //            dst, src0, kValue, tmp, cdst, offset)
    if (Value cdst = adaptor.getCdst())
      return emitCompareGather(op, adaptor, rewriter, cdst);

    // Case 3: mask-pattern TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src0)
    if (!op.getMaskPatternAttr())
      return rewriter.notifyMatchFailure(
          op, "expected maskPattern, indices, or cdst on tgather");
    return emitMaskPatternGather(op, adaptor, rewriter);
  }

  // Mask-pattern gather: TGATHER<DstT, SrcT, MaskPattern::XXXX>(dst, src0).
  LogicalResult emitMaskPatternGather(pto::TGatherOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc();

    auto getOpaqueTok = [&](Value v,
                            StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(
          op, (name + " must be emitc::OpaqueType (tile)").str());
    };
    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    if (failed(dstTokOr) || failed(srcTokOr))
      return failure();

    // mp is an EnumAttr; stringify name is "P0101" etc.
    // We emit MaskPattern::P0101 (because generated C++ has `using namespace pto;`)
    auto mp = op.getMaskPatternAttr();
    std::string mpTok = std::string("MaskPattern::") +
                        mlir::pto::stringifyMaskPattern(mp.getValue()).str();

    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, mpTok),
    });

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0});

    rewriter.eraseOp(op);
    return success();
  }

  // TGATHER(dst, src0, indices[, tmp]) without template arguments.
  LogicalResult emitIndexedGather(pto::TGatherOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  Value idx) const {
    auto loc = op.getLoc();
    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc();
    idx = peelUnrealized(idx);
    SmallVector<Value, 4> operands{dst, src0, idx};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }

  // Compare-based gather: spell out the operand types plus the compare mode.
  LogicalResult emitCompareGather(pto::TGatherOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  Value cdst) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc();
    cdst = peelUnrealized(cdst);
    Value tmp = adaptor.getTmp();
    Value kValue = adaptor.getKValue();

    auto getOpaqueTok = [&](Value v,
                            StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(
          op, (name + " must be emitc::OpaqueType (tile)").str());
    };

    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    auto cdstTokOr = getOpaqueTok(cdst, "cdst");
    auto tmpTokOr = getOpaqueTok(tmp, "tmp");
    if (failed(dstTokOr) || failed(srcTokOr) || failed(cdstTokOr) ||
        failed(tmpTokOr))
      return failure();

    auto cmpAttr = op.getCmpModeAttr();
    std::string cmpTok = cmpAttr ? cmpModeTok(cmpAttr) : "CmpMode::EQ";
    int64_t offset = 0;
    if (auto offsetAttr = op.getOffsetAttr())
      offset = getIntegerAttrSignedValue(offsetAttr);
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
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0, kValue, tmp, cdst, offsetVal});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOGatherbToEmitC : public OpConversionPattern<pto::TGatherBOp> {
  using OpConversionPattern<pto::TGatherBOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherBOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src     = adaptor.getSrc();
    Value offsets = adaptor.getOffsets();
    Value dst     = adaptor.getDst();

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
    auto *ctx = rewriter.getContext();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    SmallVector<Value, 2> operands{dst, src};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::LogPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::LogPrecision::Default:
        precisionTok = "pto::LogAlgorithm::DEFAULT";
        break;
      case pto::LogPrecision::HighPrecision:
        precisionTok = "pto::LogAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLOG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
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

    Value src = adaptor.getSrc();
    Value slope = adaptor.getSlope();
    Value dst = adaptor.getDst();

          SmallVector<Value, 3> operands{dst, src, slope};

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
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst  = adaptor.getDst();

  SmallVector<Value, 3> operands{dst, src0, src1};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMAX", operands);

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
    Value src0 = adaptor.getSrc();
    Value scalar = adaptor.getScalar();
    Value dst  = adaptor.getDst();

    SmallVector<Value, 3> operands{dst, src0, scalar};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMAXS", operands);

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
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst  = adaptor.getDst();

  SmallVector<Value, 3> operands{dst, src0, src1};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMIN", operands);

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
  Value src = adaptor.getSrc();
  Value dst = adaptor.getDst();
  Value scalar = adaptor.getScalar();

  SmallVector<Value, 3> operands{dst, src, scalar};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMINS", operands);

  rewriter.eraseOp(op);
  return success();
}
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TMOV op -> EmitC)
//===----------------------------------------------------------------------===//

struct PTOMovToEmitC : public OpConversionPattern<pto::TMovOp> {
using OpConversionPattern<pto::TMovOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TMovOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value src = adaptor.getSrc();
  Value dst = adaptor.getDst();
  Value fp;
  if (op.getFp())
    fp = adaptor.getFp();
  Value preQuantScalar;
  if (op.getPreQuantScalar())
    preQuantScalar = adaptor.getPreQuantScalar();

  auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  if (!dstOT || !srcOT)
    return rewriter.notifyMatchFailure(
        op, "tmov lowering expects opaque dst/src types");

  auto modeAttr = op.getAccToVecModeAttr();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasMode = static_cast<bool>(modeAttr);
  const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;

  SmallVector<Value, 4> operands{dst, src};
  SmallVector<Attribute, 5> templateArgVec{
      emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
      emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
  };
  StringRef callee = "TMOV";

  if (failed(buildTMovOperandsAndTemplates(op, rewriter, ctx, fp,
                                           preQuantScalar, modeAttr, operands,
                                           templateArgVec, callee)))
    return failure();

  const bool isXToZz =
      hasFp && pto::classifyTMovForm(op.getFp()) == pto::TMovForm::XToZz;
  ArrayAttr templateArgs =
      (isXToZz && templateArgVec.empty()) ||
              (templateArgVec.size() == 2 && !hasFp && !preQuantScalar &&
              !hasMode && !reluNonDefault)
          ? ArrayAttr{}
          : rewriter.getArrayAttr(templateArgVec);

  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, callee,
      /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
      /*operands=*/operands);

  rewriter.eraseOp(op);
  return success();
}

  // Collect the TMOV operands and template tokens for the operand flavor
  // (fp / preQuant / mode / relu).
  LogicalResult buildTMovOperandsAndTemplates(
      pto::TMovOp op, ConversionPatternRewriter &rewriter,
      MLIRContext *ctx, Value fp, Value preQuantScalar,
      pto::AccToVecModeAttr modeAttr, SmallVectorImpl<Value> &operands,
      SmallVectorImpl<Attribute> &templateArgVec,
      StringRef &callee) const {
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    const bool isXToZz =
        hasFp && pto::classifyTMovForm(op.getFp()) == pto::TMovForm::XToZz;

    if (hasFp) {
      auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
      if (!fpOT)
        return rewriter.notifyMatchFailure(
            op, "tmov fp lowering expects opaque fp type");
      operands.push_back(fp);
      if (isXToZz) {
        templateArgVec.clear();
        if (op.getGrpAxisAttr() &&
            op.getGrpAxisAttr().getValue() == pto::MxGroupAxis::Axis0)
          templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, "0"));
      } else {
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
        pushModeAndReluTemplateArgs(templateArgVec, ctx, modeAttr,
                                    reluNonDefault, op.getReluPreMode());
        callee = hasMode ? "TMOV" : "TMOV_FP";
      }
    } else if (hasPreQuantScalar) {
      operands.push_back(preQuantScalar);
      pushModeAndReluTemplateArgs(templateArgVec, ctx, modeAttr,
                                  reluNonDefault, op.getReluPreMode());
    } else if (hasMode) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, getAccToVecModeToken(modeAttr.getValue())));
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, getReluPreModeToken(op.getReluPreMode())));
    } else if (reluNonDefault) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, getReluPreModeToken(op.getReluPreMode())));
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

// Materialize an lvalue for `offset` (reusing an existing emitc.variable when
// possible) and return its address; TQuant/TQuantMx pass &offset to the
// intrinsic.
static Value materializeOffsetAddress(ConversionPatternRewriter &rewriter,
                                      Location loc, MLIRContext *ctx,
                                      Value offset) {
  Type offsetValueTy = offset.getType();
  Value offsetLValue = getSourceEmitCVariable(offset);
  if (!offsetLValue) {
    offsetLValue =
        rewriter
            .create<emitc::VariableOp>(
                loc, getEmitCVariableResultType(offsetValueTy),
                emitc::OpaqueAttr::get(ctx, ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, offsetLValue, offset);
  }
  return rewriter
      .create<emitc::ApplyOp>(
          loc, emitc::PointerType::get(offsetValueTy), "&", offsetLValue)
      .getResult();
}

// TQUANT template arguments: QuantType, dst/src (and fp, tmp) opaque type
// spellings; empty when any operand is not opaque.
static FailureOr<ArrayAttr>
buildTQuantTemplateArgs(pto::TQuantOp op,
                        ConversionPatternRewriter &rewriter,
                        MLIRContext *ctx, Value dst, Value src, Value fp,
                        Value tmp) {
  auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
  if (!(dstOT && srcOT && fpOT))
    return ArrayAttr{};

  auto quantTypeTok = [&]() -> StringRef {
    switch (op.getQuantType()) {
    case pto::QuantType::INT8_SYM:
      return "pto::QuantType::INT8_SYM";
    case pto::QuantType::INT8_ASYM:
      return "pto::QuantType::INT8_ASYM";
    case pto::QuantType::MXFP8:
    case pto::QuantType::MXFP4_E2M1:
      break;
    }
    llvm_unreachable("unknown QuantType");
  };

  SmallVector<Attribute, 5> args{
      emitc::OpaqueAttr::get(ctx, quantTypeTok()),
      emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
      emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
      emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
  };
  if (tmp) {
    auto tmpOT = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
    if (!tmpOT)
      return rewriter.notifyMatchFailure(
          op, "tquant tmp lowering expects opaque tmp type");
    args.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
  }
  return rewriter.getArrayAttr(args);
}

struct PTOQuantToEmitC : public OpConversionPattern<pto::TQuantOp> {
using OpConversionPattern<pto::TQuantOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TQuantOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value dst = adaptor.getDst();
  Value src = adaptor.getSrc();
  Value fp = adaptor.getFp();


  Value tmp;
  if (op.getTmp())
    tmp = adaptor.getTmp();
  Value offsetPtr;
  if (op.getOffset())
    offsetPtr = materializeOffsetAddress(rewriter, loc, ctx, adaptor.getOffset());

  FailureOr<ArrayAttr> templateArgsOr = buildTQuantTemplateArgs(
      op, rewriter, ctx, dst, src, fp, tmp);
  if (failed(templateArgsOr))
    return failure();
  ArrayAttr templateArgs = *templateArgsOr;

  SmallVector<Value> operands{dst, src, fp};
  if (tmp)
    operands.push_back(tmp);
  if (offsetPtr)
    operands.push_back(offsetPtr);

  rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TQUANT", ArrayAttr{}, templateArgs, operands);

  rewriter.eraseOp(op);
  return success();
}
};

// Take the address of an EmitC value: reuse an existing emitc.variable
// lvalue when possible, otherwise materialize a temporary.
static Value addressOfEmitCValue(ConversionPatternRewriter &rewriter,
                                 Location loc, MLIRContext *ctx, Value v,
                                 emitc::OpaqueType ot) {
  if (Value variable = getSourceEmitCVariable(v))
    return rewriter
        .create<emitc::ApplyOp>(loc, emitc::PointerType::get(v.getType()), "&",
                                variable)
        .getResult();

  Value tmp = rewriter
                  .create<emitc::VariableOp>(
                      loc, getEmitCVariableResultType(ot),
                      emitc::OpaqueAttr::get(ctx, ""))
                  .getResult();
  rewriter.create<emitc::AssignOp>(loc, tmp, v);
  return rewriter
      .create<emitc::ApplyOp>(loc, emitc::PointerType::get(ot), "&", tmp)
      .getResult();
}

// Modern (non exp_zz) TQUANT-MX template tokens: group axis, the
// OCP/NV algorithm variant, and the optional interleave flag.
static void appendModernMxTemplateArgs(pto::TQuantMxOp op,
                                       MLIRContext *ctx,
                                       SmallVectorImpl<Attribute> &out) {
  const StringRef axisTok =
      op.getGrpAxis() == pto::MxGroupAxis::Axis0 ? "0" : "1";
  StringRef algTok;
  if (op.getQuantType() == pto::QuantType::MXFP8)
    algTok = op.getQuantScaleAlg() == pto::QuantScaleAlg::NV
                 ? "pto::MxQuantAlg::NvMxFp8E4M3"
                 : "pto::MxQuantAlg::OcpMxFp8E4M3";
  else
    algTok = op.getQuantScaleAlg() == pto::QuantScaleAlg::NV
                 ? "pto::MxQuantAlg::NvMxFp4E2M1"
                 : "pto::MxQuantAlg::OcpMxFp4E2M1";
  out.push_back(emitc::OpaqueAttr::get(ctx, axisTok));
  out.push_back(emitc::OpaqueAttr::get(ctx, algTok));
  if (op.getInterleave())
    out.push_back(emitc::OpaqueAttr::get(ctx, "true"));
}

static StringRef quantScaleAlgTok(pto::QuantScaleAlg alg) {
  switch (alg) {
  case pto::QuantScaleAlg::OCP:
    return "pto::QuantScaleAlg::OCP";
  case pto::QuantScaleAlg::NV:
    return "pto::QuantScaleAlg::NV";
  }
  llvm_unreachable("unknown QuantScaleAlg");
}

static StringRef vecStoreModeTok(pto::VecStoreMode mode) {
  switch (mode) {
  case pto::VecStoreMode::ND:
    return "pto::VecStoreMode::ND";
  case pto::VecStoreMode::NZ:
    return "pto::VecStoreMode::NZ";
  }
  llvm_unreachable("unknown VecStoreMode");
}

// Deprecated fused TQUANT-MX form: retain the existing PTO-ISA overload and
// complete tile-type template list for wire/API compatibility.
static void appendLegacyExpZzMxTemplateArgs(
    pto::TQuantMxOp op, MLIRContext *ctx, StringRef quantTypeStr,
    emitc::OpaqueType dstOT, emitc::OpaqueType srcOT,
    emitc::OpaqueType expOT, emitc::OpaqueType maxOT,
    emitc::OpaqueType scalingOT, SmallVectorImpl<Attribute> &out) {
  out.push_back(emitc::OpaqueAttr::get(ctx, quantTypeStr));
  if (auto storeMode = op.getStoreMode())
    out.push_back(emitc::OpaqueAttr::get(ctx, vecStoreModeTok(*storeMode)));
  out.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, expOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, maxOT.getValue().str()));
  out.push_back(emitc::OpaqueAttr::get(ctx, scalingOT.getValue().str()));
  if (!op.getStoreMode() && op.getQuantScaleAlg() != pto::QuantScaleAlg::OCP)
    out.push_back(emitc::OpaqueAttr::get(ctx, quantScaleAlgTok(op.getQuantScaleAlg())));
}

struct PTOQuantMxToEmitC : public OpConversionPattern<pto::TQuantMxOp> {
using OpConversionPattern<pto::TQuantMxOp>::OpConversionPattern;

LogicalResult matchAndRewrite(pto::TQuantMxOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto *ctx = rewriter.getContext();

  Value dst = adaptor.getDst();
  Value src = adaptor.getSrc();
  Value exp = adaptor.getExp();
  Value max = adaptor.getMax();
  Value scaling = adaptor.getScaling();
  Value expZz = adaptor.getExpZz()
                    ? adaptor.getExpZz()
                    : Value{};

  auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
  auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
  auto expOT = mlir::dyn_cast<emitc::OpaqueType>(exp.getType());
  auto maxOT = mlir::dyn_cast<emitc::OpaqueType>(max.getType());
  auto scalingOT = mlir::dyn_cast<emitc::OpaqueType>(scaling.getType());
  auto expZzOT = expZz ? mlir::dyn_cast<emitc::OpaqueType>(expZz.getType())
                       : emitc::OpaqueType{};
  if (!dstOT || !srcOT || !expOT || !maxOT || !scalingOT)
    return rewriter.notifyMatchFailure(
        op, "expected all operands to be emitc::OpaqueType");
  if (expZz && !expZzOT)
    return rewriter.notifyMatchFailure(
        op, "expected exp_zz operand to be emitc::OpaqueType");


  Value expPtr = addressOfEmitCValue(rewriter, loc, ctx, exp, expOT);
  Value maxPtr = addressOfEmitCValue(rewriter, loc, ctx, max, maxOT);
  Value scalingPtr = addressOfEmitCValue(rewriter, loc, ctx, scaling, scalingOT);
  Value expZzPtr = expZz ? addressOfEmitCValue(rewriter, loc, ctx, expZz, expZzOT) : Value{};

  std::string quantTypeStr =
      op.getQuantType() == pto::QuantType::MXFP8
          ? "pto::QuantType::MXFP8"
          : "pto::QuantType::MXFP4_E2M1";

  SmallVector<Attribute> templateArgsStorage;
  if (expZz) {
    appendLegacyExpZzMxTemplateArgs(op, ctx, quantTypeStr, dstOT, srcOT,
                                    expOT, maxOT, scalingOT,
                                    templateArgsStorage);
  } else {
    appendModernMxTemplateArgs(op, ctx, templateArgsStorage);
  }
  ArrayAttr templateArgs = rewriter.getArrayAttr(templateArgsStorage);

  SmallVector<Value> operands{dst, src, expPtr, maxPtr, scalingPtr};
  if (expZzPtr)
    operands.push_back(expZzPtr);
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

  Value dst    = adaptor.getDst();
  Value src    = adaptor.getSrc();
  Value scale  = adaptor.getScale();
  Value offset = adaptor.getOffset();

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
  if (op.isFormat1())
    return emitFormat1(op, adaptor, rewriter);
  if (op.isFormat2())
    return emitFormat2(op, adaptor, rewriter);
  return op.emitOpError("unsupported mrgsort_dps format");
}

  // TMRGSORT(dst, src, blockLen).
  LogicalResult emitFormat1(pto::TMrgSortOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = adaptor.getSrcs().front();
    Value dst = adaptor.getDsts().front();
    Value blockLen = adaptor.getBlockLen();

    SmallVector<Value, 3> operands{dst, src, blockLen};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMRGSORT",
        ArrayAttr{}, ArrayAttr{}, operands);
    rewriter.eraseOp(op);
    return success();
  }

  // TMRGSORT<DstTile, TmpTile, Src0..SrcN, exhausted>(
  //     dst, executedNumList, tmp, src0..srcN).
  LogicalResult emitFormat2(pto::TMrgSortOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = adaptor.getDsts()[0];
    Value tmp = adaptor.getTmp();
    Value excuted = adaptor.getExcuted();

    SmallVector<Value, 4> srcs;
    srcs.reserve(adaptor.getSrcs().size());
    for (Value v : adaptor.getSrcs())
      srcs.push_back(v);

    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto tmpOT = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
    if (!dstOT || !tmpOT || srcs.size() < 2 || srcs.size() > 4)
      return op.emitOpError("format2 expects dst/tmp tilebufs and 2 to 4 srcs");

    SmallVector<Attribute, 8> targs;
    targs.reserve(2 + srcs.size() + 1);
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

    SmallVector<Value, 7> operands{dst, excuted, tmp};
    operands.append(srcs.begin(), srcs.end());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMRGSORT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
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
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  Value dst  = adaptor.getDst();

  SmallVector<Value, 3> operands{dst, src0, src1};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMUL", operands);

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
  Value src = adaptor.getSrc0();
  Value dst = adaptor.getDst();
  Value scalar = adaptor.getScalar();

  SmallVector<Value, 3> operands{dst, src, scalar};
  createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                               "TMULS", operands);

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

  Value src = adaptor.getSrc();
  Value dst = adaptor.getDst();

  SmallVector<Value, 2> operands{dst, src};
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


void populateTensorPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  patterns.add<PTOColExpandAddToEmitC>(typeConverter, ctx, "TCOLEXPANDADD");
  patterns.add<PTOColExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMulToEmitC>(typeConverter, ctx, "TCOLEXPANDMUL");
  patterns.add<PTOColExpandMaxToEmitC>(typeConverter, ctx, "TCOLEXPANDMAX");
  patterns.add<PTOColExpandMinToEmitC>(typeConverter, ctx, "TCOLEXPANDMIN");
  patterns.add<PTOColExpandSubToEmitC>(typeConverter, ctx, "TCOLEXPANDSUB");
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColProdToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOConcatToEmitC, PTOConcatidxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulsToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpandsToEmitC>(typeConverter, ctx);
  patterns.add<PTOExtractToEmitC, PTOInsertToEmitC>(typeConverter, ctx);
  patterns.add<PTOFillPadToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherbToEmitC>(typeConverter, ctx);
  patterns.add<PTOQuantToEmitC,
               PTOQuantMxToEmitC>(typeConverter, ctx);
  patterns.add<PTODequantToEmitC>(typeConverter, ctx);
  patterns.add<PTOLogToEmitC>(typeConverter, ctx);
    populateTensorPatternsPart2(patterns, typeConverter, ctx, targetArch);
}

void populateTensorPatternsPart2(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx, PTOArch targetArch) {
  (void)typeConverter;
  (void)ctx;
  (void)targetArch;
  patterns.add<PTOMovToEmitC>(typeConverter, ctx);
  patterns.add<PTONegToEmitC>(typeConverter, ctx);
  patterns.add<PTOTCIToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTriToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<PTORandomToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddSToTADDS>(typeConverter, ctx);
  patterns.add<PTOColExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAddCToTADDC>(typeConverter, ctx);
  patterns.add<PTOMinsToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
