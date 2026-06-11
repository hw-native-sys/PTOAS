// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCArith.cpp ------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

using namespace mlir;

namespace mlir::pto {
namespace {

static constexpr unsigned kPTOIndexBitWidth = 32;
static constexpr size_t kNumber2 = 2;
static constexpr unsigned kNumber32 = 32;

struct SignedDivAdjustOperands {
  Value q0;
  Value one;
  Value adjust;
};

static SignedDivAdjustOperands buildSignedDivAdjustOperands(
    ConversionPatternRewriter &rewriter, Location loc, Type dstTy, Value lhs,
    Value rhs, bool expectSameSigns) {
  Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
  Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);
  Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, lhs, rhs);
  Value r = rewriter.create<emitc::RemOp>(loc, dstTy, lhs, rhs);
  Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                emitc::CmpPredicate::ne, r, zero);
  Value lhsLt0 = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                               emitc::CmpPredicate::lt, lhs, zero);
  Value rhsLt0 = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                               emitc::CmpPredicate::lt, rhs, zero);
  auto pred = expectSameSigns ? emitc::CmpPredicate::eq : emitc::CmpPredicate::ne;
  Value signCond =
      rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(), pred, lhsLt0, rhsLt0);
  Value adjust = rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                      rNeZero, signCond);
  return {q0, one, adjust};
}

static LogicalResult rewriteI1RightShiftApprox(Operation *op, Location loc,
                                               Type dstTy, Value lhs, Value rhs,
                                               ConversionPatternRewriter &rewriter) {
  auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
  Value lhsU8 = emitCCast(rewriter, loc, u8Ty, lhs);
  Value rhsU8 = emitCCast(rewriter, loc, u8Ty, rhs);
  Value sh =
      rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8, rhsU8);
  Value masked = rewriter.create<emitc::BitwiseAndOp>(
      loc, u8Ty, sh, makeEmitCIntConstant(rewriter, loc, u8Ty, 1));
  rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
  return success();
}

static LogicalResult getScalarIntegerOpInfo(Operation *, Type opTy,
                                            const TypeConverter *typeConverter,
                                            unsigned &bitWidth, Type &dstTy) {
  auto intTy = dyn_cast<IntegerType>(opTy);
  const bool isIndex = isa<IndexType>(opTy);
  if (!intTy && !isIndex)
    return failure();

  bitWidth = intTy ? intTy.getWidth()
                   : static_cast<unsigned>(kPTOIndexBitWidth);
  dstTy = typeConverter->convertType(opTy);
  return dstTy ? success() : failure();
}

template <typename EmitCOp>
static LogicalResult rewriteUnsignedBinaryIntLikeOp(
    Operation *op, Location loc, Type opTy, Value lhs, Value rhs,
    ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter,
    std::optional<int64_t> i1MaskValue = std::nullopt) {
  unsigned bitWidth = 0;
  Type dstTy;
  if (failed(getScalarIntegerOpInfo(op, opTy, typeConverter, bitWidth, dstTy)))
    return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

  if (bitWidth == 1) {
    if (!i1MaskValue) {
      rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, lhs, rhs);
      return success();
    }
    auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
    Value lhsU8 = emitCCast(rewriter, loc, u8Ty, lhs);
    Value rhsU8 = emitCCast(rewriter, loc, u8Ty, rhs);
    Value raw = rewriter.create<EmitCOp>(loc, u8Ty, lhsU8, rhsU8);
    Value masked = rewriter.create<emitc::BitwiseAndOp>(
        loc, u8Ty, raw,
        makeEmitCIntConstant(rewriter, loc, u8Ty, *i1MaskValue));
    rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
    return success();
  }

  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  Value lhsU =
      castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
  Value rhsU =
      castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
  Value resU = rewriter.create<EmitCOp>(loc, uTy, lhsU, rhsU);
  rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, resU));
  return success();
}

static LogicalResult rewriteDirectCastOp(Operation *op, Value in, Type dstTy,
                                         ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, in);
  return success();
}

static LogicalResult getScalarIntegerCastTypes(Operation *, Type dstSrcTy,
                                               Type inSrcTy,
                                               const TypeConverter *typeConverter,
                                               Type &dstTy,
                                               IntegerType &dstIntTy,
                                               IntegerType &srcIntTy) {
  dstIntTy = dyn_cast<IntegerType>(dstSrcTy);
  srcIntTy = dyn_cast<IntegerType>(inSrcTy);
  if (!dstIntTy || !srcIntTy)
    return failure();
  dstTy = typeConverter->convertType(dstIntTy);
  return dstTy ? success() : failure();
}

struct ScalarIntegerCastInfo {
  Type dstTy;
  IntegerType dstIntTy;
  IntegerType srcIntTy;
};

static FailureOr<ScalarIntegerCastInfo> getScalarIntegerCastInfo(
    Operation *op, Type dstSrcTy, Type inSrcTy,
    const TypeConverter *typeConverter) {
  Type dstTy;
  IntegerType dstIntTy;
  IntegerType srcIntTy;
  if (failed(getScalarIntegerCastTypes(op, dstSrcTy, inSrcTy, typeConverter,
                                       dstTy, dstIntTy, srcIntTy))) {
    return failure();
  }
  return ScalarIntegerCastInfo{dstTy, dstIntTy, srcIntTy};
}

//===----------------------------------------------------------------------===//
// Arith -> EmitC (full dialect coverage for scalar ops)
//===----------------------------------------------------------------------===//

template <typename ArithOp, typename EmitCOp>
struct ArithSimpleBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteSimpleBinary(op, adaptor, &rewriter);
  }

private:
  LogicalResult rewriteSimpleBinary(ArithOp op,
                                    typename ArithOp::Adaptor adaptor,
                                    ConversionPatternRewriter *rewriter) const {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter->replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getOperands());
    return success();
  }
};

// Integer bitwise ops (andi/ori/xori) on signless integers: perform in unsigned
// to avoid signedness pitfalls, then cast back.
template <typename ArithOp, typename EmitCOp>
struct ArithUnsignedBitwiseBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<EmitCOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, this->getTypeConverter());
  }
};

struct ArithDivUIToEmitC : public OpConversionPattern<arith::DivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<emitc::DivOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter());
  }
};

struct ArithRemUIToEmitC : public OpConversionPattern<arith::RemUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<emitc::RemOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter());
  }
};

struct ArithCeilDivUIToEmitC : public OpConversionPattern<arith::CeilDivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    unsigned bitWidth = 0;
    Type dstTy;
    if (failed(getScalarIntegerOpInfo(op, op.getType(), getTypeConverter(),
                                      bitWidth, dstTy)))
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value one = makeEmitCIntConstant(rewriter, loc, uTy, 1);
    Value rhsMinusOne = rewriter.create<emitc::SubOp>(loc, uTy, rhsU, one);
    Value num = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsMinusOne);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, num, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivSIToEmitC : public OpConversionPattern<arith::CeilDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    unsigned bitWidth = 0;
    Type dstTy;
    if (failed(getScalarIntegerOpInfo(op, op.getType(), getTypeConverter(),
                                      bitWidth, dstTy)))
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    auto data = buildSignedDivAdjustOperands(rewriter, op.getLoc(), dstTy,
                                             adaptor.getLhs(), adaptor.getRhs(),
                                             /*expectSameSigns=*/true);
    Value qPlusOne =
        rewriter.create<emitc::AddOp>(op.getLoc(), dstTy, data.q0, data.one);
    Value result = rewriter.create<emitc::ConditionalOp>(
        op.getLoc(), dstTy, data.adjust, qPlusOne, data.q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithFloorDivSIToEmitC : public OpConversionPattern<arith::FloorDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    unsigned bitWidth = 0;
    Type dstTy;
    if (failed(getScalarIntegerOpInfo(op, op.getType(), getTypeConverter(),
                                      bitWidth, dstTy)))
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    auto data = buildSignedDivAdjustOperands(rewriter, op.getLoc(), dstTy,
                                             adaptor.getLhs(), adaptor.getRhs(),
                                             /*expectSameSigns=*/false);
    Value qMinusOne =
        rewriter.create<emitc::SubOp>(op.getLoc(), dstTy, data.q0, data.one);
    Value result = rewriter.create<emitc::ConditionalOp>(
        op.getLoc(), dstTy, data.adjust, qMinusOne, data.q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftLeftToEmitC : public OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<emitc::BitwiseLeftShiftOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter(), /*i1MaskValue=*/1);
  }
};

struct ArithShiftRightUIToEmitC : public OpConversionPattern<arith::ShRUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    unsigned bitWidth = 0;
    Type dstTy;
    if (failed(getScalarIntegerOpInfo(op, op.getType(), getTypeConverter(),
                                      bitWidth, dstTy)))
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");
    if (bitWidth == 1)
      return rewriteI1RightShiftApprox(op, op.getLoc(), dstTy, adaptor.getLhs(),
                                       adaptor.getRhs(), rewriter);
    return rewriteUnsignedBinaryIntLikeOp<emitc::BitwiseRightShiftOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter());
  }
};

struct ArithShiftRightSIToEmitC : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    unsigned bitWidth = 0;
    Type dstTy;
    if (failed(getScalarIntegerOpInfo(op, op.getType(), getTypeConverter(),
                                      bitWidth, dstTy)))
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    if (bitWidth == 1)
      return rewriteI1RightShiftApprox(op, loc, dstTy, adaptor.getLhs(),
                                       adaptor.getRhs(), rewriter);

    // Signed arithmetic shift; cast RHS to unsigned to interpret shift amount.
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value sh =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, dstTy, adaptor.getLhs(),
                                                    rhsU);
    rewriter.replaceOp(op, sh);
    return success();
  }
};

struct ArithNegFToEmitC : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, dstTy, adaptor.getOperand());
    return success();
  }
};

struct ArithRemFToEmitC : public OpConversionPattern<arith::RemFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Use builtin `fmod` when possible. For f16, compute in float and cast back.
    Type callTy = dstTy;
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF16()) {
        auto f32Ty = emitc::OpaqueType::get(rewriter.getContext(), "float");
        lhs = emitCCast(rewriter, loc, f32Ty, lhs);
        rhs = emitCCast(rewriter, loc, f32Ty, rhs);
        callTy = f32Ty;
      }
    }

    // Prefer `__builtin_fmod*` to avoid relying on extra headers.
    llvm::StringRef callee = "__builtin_fmod";
    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF32() || opFloatTy.isF16())
        callee = "__builtin_fmodf";
      else if (opFloatTy.isF64())
        callee = "__builtin_fmod";
    }

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{callTy}, callee, ValueRange{lhs, rhs},
        /*args=*/ArrayAttr{}, /*template_args=*/ArrayAttr{});
    Value result = call.getResult(0);
    if (callTy != dstTy)
      result = emitCCast(rewriter, loc, dstTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithSelectToEmitC : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "only scalar i1 conditions supported for arith.select");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto cond =
        rewriter.create<emitc::ConditionalOp>(op.getLoc(), dstTy,
                                              adaptor.getCondition(),
                                              adaptor.getTrueValue(),
                                              adaptor.getFalseValue());
    rewriter.replaceOp(op, cond.getResult());
    return success();
  }
};

struct ArithExtUIToEmitC : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto castInfo = getScalarIntegerCastInfo(op, op.getType(), op.getIn().getType(),
                                             getTypeConverter());
    if (failed(castInfo))
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    // i1 -> iN: bool to integer already behaves as 0/1.
    if (castInfo->srcIntTy.getWidth() == 1) {
      return rewriteDirectCastOp(op, adaptor.getIn(), castInfo->dstTy, rewriter);
    }

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), castInfo->dstIntTy.getWidth());
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           castInfo->srcIntTy.getWidth());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, castInfo->dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithExtSIToEmitC : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto castInfo = getScalarIntegerCastInfo(op, op.getType(), op.getIn().getType(),
                                             getTypeConverter());
    if (failed(castInfo))
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    // i1 sign-extension: 0 -> 0, 1 -> -1.
    if (castInfo->srcIntTy.getWidth() == 1) {
      Value zero = makeEmitCIntConstant(rewriter, loc, castInfo->dstTy, 0);
      Value asInt = emitCCast(rewriter, loc, castInfo->dstTy, adaptor.getIn());
      Value neg =
          rewriter.create<emitc::SubOp>(loc, castInfo->dstTy, zero, asInt).getResult();
      rewriter.replaceOp(op, neg);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, castInfo->dstTy, adaptor.getIn());
    return success();
  }
};

template <typename CastOp>
struct ArithCastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult match(CastOp op) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    return success();
  }

  void rewrite(CastOp op, typename CastOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
  }
};

struct ArithIndexCastUIToEmitC : public OpConversionPattern<arith::IndexCastUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // MemRef casts are handled elsewhere; for safety, fall back to emitc.cast.
    if (isa<MemRefType>(op.getIn().getType()) || isa<MemRefType>(op.getType())) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto getBW = [](Type t) -> std::optional<unsigned> {
      if (auto i = dyn_cast<IntegerType>(t))
        return i.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };

    auto srcBW = getBW(op.getIn().getType());
    auto dstBW = getBW(op.getType());
    if (!srcBW || !dstBW)
      return rewriter.notifyMatchFailure(op, "unsupported index_castui types");

    if (*dstBW <= *srcBW) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uSrcTy = getUnsignedIntOpaqueType(rewriter.getContext(), *srcBW);
    auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), *dstBW);
    Value srcU = emitCCast(rewriter, loc, uSrcTy, adaptor.getIn());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithUIToFPToEmitC : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer input");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Convert via an unsigned integer type of the same width.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value fp = rewriter.create<emitc::CastOp>(loc, dstTy, srcU).getResult();
    rewriter.replaceOp(op, fp);
    return success();
  }
};

struct ArithFPToUIToEmitC : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    if (!dstIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer result");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value asU = rewriter.create<emitc::CastOp>(loc, uDstTy, adaptor.getIn()).getResult();
    Value result = emitCCast(rewriter, loc, dstTy, asU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithBitcastToEmitC : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // For pointer-like types, a regular cast is fine.
    if (isa<emitc::PointerType>(dstTy)) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    // Only support scalar int/float/index bitcasts here.
    auto srcTy = op.getIn().getType();
    auto dstOrigTy = op.getType();

    auto getBitWidth = [](Type t) -> std::optional<unsigned> {
      if (auto it = dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto ft = dyn_cast<FloatType>(t))
        return ft.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };
    auto srcBW = getBitWidth(srcTy);
    auto dstBW = getBitWidth(dstOrigTy);
    if (!srcBW || !dstBW || *srcBW != *dstBW)
      return rewriter.notifyMatchFailure(op, "bitcast requires equal bitwidth");

    // Determine the template argument from the destination type string.
    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return rewriter.notifyMatchFailure(op, "expected emitc opaque dest type");

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{dstTy}, "ptoas_bitcast", /*operands=*/ValueRange{adaptor.getIn()},
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

// arith.cmpf lowering with ordered/unordered semantics.
struct ArithCmpFToEmitC : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  struct CmpFConfig {
    bool unordered = false;
    emitc::CmpPredicate predicate = emitc::CmpPredicate::eq;
  };

  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                        Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::eq,
                              v, v)
        .getResult();
  }

  static std::optional<Value> buildSpecialCmpFResult(
      arith::CmpFPredicate predicate, ConversionPatternRewriter &rewriter,
      Location loc, Type i1Ty, Value lhs, Value rhs) {
    switch (predicate) {
    case arith::CmpFPredicate::AlwaysFalse:
      return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "false");
    case arith::CmpFPredicate::AlwaysTrue:
      return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "true");
    case arith::CmpFPredicate::ORD:
      return rewriter.create<emitc::LogicalAndOp>(
                 loc, i1Ty, isNotNaN(rewriter, loc, lhs),
                 isNotNaN(rewriter, loc, rhs))
          .getResult();
    case arith::CmpFPredicate::UNO:
      return rewriter.create<emitc::LogicalOrOp>(
                 loc, i1Ty, isNaN(rewriter, loc, lhs),
                 isNaN(rewriter, loc, rhs))
          .getResult();
    default:
      return std::nullopt;
    }
  }

  static std::optional<CmpFConfig>
  getCmpFConfig(arith::CmpFPredicate predicate) {
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return CmpFConfig{false, emitc::CmpPredicate::eq};
    case arith::CmpFPredicate::OGT:
      return CmpFConfig{false, emitc::CmpPredicate::gt};
    case arith::CmpFPredicate::OGE:
      return CmpFConfig{false, emitc::CmpPredicate::ge};
    case arith::CmpFPredicate::OLT:
      return CmpFConfig{false, emitc::CmpPredicate::lt};
    case arith::CmpFPredicate::OLE:
      return CmpFConfig{false, emitc::CmpPredicate::le};
    case arith::CmpFPredicate::ONE:
      return CmpFConfig{false, emitc::CmpPredicate::ne};
    case arith::CmpFPredicate::UEQ:
      return CmpFConfig{true, emitc::CmpPredicate::eq};
    case arith::CmpFPredicate::UGT:
      return CmpFConfig{true, emitc::CmpPredicate::gt};
    case arith::CmpFPredicate::UGE:
      return CmpFConfig{true, emitc::CmpPredicate::ge};
    case arith::CmpFPredicate::ULT:
      return CmpFConfig{true, emitc::CmpPredicate::lt};
    case arith::CmpFPredicate::ULE:
      return CmpFConfig{true, emitc::CmpPredicate::le};
    case arith::CmpFPredicate::UNE:
      return CmpFConfig{true, emitc::CmpPredicate::ne};
    default:
      return std::nullopt;
    }
  }

  static Value buildCmpFResult(const CmpFConfig &config,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Type i1Ty, Value lhs, Value rhs) {
    Value cmp = rewriter
                    .create<emitc::CmpOp>(loc, i1Ty, config.predicate, lhs, rhs)
                    .getResult();
    Value unord = rewriter.create<emitc::LogicalOrOp>(
        loc, i1Ty, isNaN(rewriter, loc, lhs), isNaN(rewriter, loc, rhs));
    if (config.unordered)
      return rewriter
          .create<emitc::LogicalOrOp>(loc, i1Ty, unord, cmp)
          .getResult();
    Value ord = rewriter.create<emitc::LogicalAndOp>(
        loc, i1Ty, isNotNaN(rewriter, loc, lhs), isNotNaN(rewriter, loc, rhs));
    return rewriter
        .create<emitc::LogicalAndOp>(loc, i1Ty, ord, cmp)
        .getResult();
  }

  LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getLhs().getType()))
      return rewriter.notifyMatchFailure(op, "cmpf only supported on scalar floats");

    auto loc = op.getLoc();
    auto i1Ty = rewriter.getI1Type();
    if (auto special = buildSpecialCmpFResult(op.getPredicate(), rewriter, loc,
                                              i1Ty, adaptor.getLhs(),
                                              adaptor.getRhs())) {
      rewriter.replaceOp(op, *special);
      return success();
    }

    auto config = getCmpFConfig(op.getPredicate());
    if (!config)
      return rewriter.notifyMatchFailure(op, "unsupported cmpf predicate");
    rewriter.replaceOp(op, buildCmpFResult(*config, rewriter, loc, i1Ty,
                                           adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ArithAddUIExtendedToEmitC
    : public OpConversionPattern<arith::AddUIExtendedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getSum().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    if (newResultTypes.size() != kNumber2)
      return failure();

    Type sumDstTy = newResultTypes[0];
    Type overflowDstTy = newResultTypes[1];

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    auto wideTy = getWiderUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);

    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
    Value rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    Value sumWide =
        rewriter.create<emitc::AddOp>(loc, wideTy, lhsWide, rhsWide).getResult();

    Value sumN = emitCCast(rewriter, loc, uTy, sumWide);
    Value sum = emitCCast(rewriter, loc, sumDstTy, sumN);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value high = rewriter
                     .create<emitc::BitwiseRightShiftOp>(loc, wideTy, sumWide,
                                                         shiftAmt)
                     .getResult();
    Value zeroWide = makeEmitCIntConstant(rewriter, loc, wideTy, 0);
    Value overflow =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne, high, zeroWide)
            .getResult();
    overflow = emitCCast(rewriter, loc, overflowDstTy, overflow);

    rewriter.replaceOp(op, {sum, overflow});
    return success();
  }
};

template <typename ArithOp, bool isUnsigned>
struct ArithMulExtendedToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getResult(0).getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResultTypes)))
      return failure();
    if (newResultTypes.size() != kNumber2)
      return failure();

    Type lowDstTy = newResultTypes[0];
    Type highDstTy = newResultTypes[1];

    Type wideTy =
        isUnsigned
            ? static_cast<Type>(
                  getWiderUnsignedIntOpaqueType(rewriter.getContext(), bitWidth))
            : static_cast<Type>(
                  getWiderSignedIntOpaqueType(rewriter.getContext(), bitWidth));

    Value lhsWide;
    Value rhsWide;
    if constexpr (isUnsigned) {
      Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                      bitWidth);
      Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                      bitWidth);
      lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
      rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    } else {
      lhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getLhs());
      rhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getRhs());
    }

    Value prodWide =
        rewriter.create<emitc::MulOp>(loc, wideTy, lhsWide, rhsWide).getResult();
    Value low = emitCCast(rewriter, loc, lowDstTy, prodWide);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value highWide = rewriter
                         .create<emitc::BitwiseRightShiftOp>(loc, wideTy, prodWide,
                                                             shiftAmt)
                         .getResult();
    Value high = emitCCast(rewriter, loc, highDstTy, highWide);

    rewriter.replaceOp(op, {low, high});
    return success();
  }
};

using ArithMulSIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulSIExtendedOp, /*isUnsigned=*/false>;
using ArithMulUIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulUIExtendedOp, /*isUnsigned=*/true>;

struct ArithMinMaxIToEmitCBase {
  static Value makeSelect(ConversionPatternRewriter &rewriter, Location loc,
                          Type dstTy, Value cond, Value trueV, Value falseV) {
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, cond, trueV, falseV)
        .getResult();
  }

  static Value makeLessThanCond(ConversionPatternRewriter &rewriter, Location loc,
                                Value lhs, Value rhs) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::lt,
                              lhs, rhs)
        .getResult();
  }

  static LogicalResult rewriteSignedMinMax(Operation *op, Value lhs, Value rhs,
                                           bool chooseMax,
                                           ConversionPatternRewriter &rewriter,
                                           const TypeConverter *typeConverter) {
    Type dstTy = typeConverter->convertType(op->getResult(0).getType());
    if (!dstTy)
      return failure();
    Value cond = makeLessThanCond(rewriter, op->getLoc(), lhs, rhs);
    Value res = makeSelect(rewriter, op->getLoc(), dstTy, cond,
                           chooseMax ? rhs : lhs, chooseMax ? lhs : rhs);
    rewriter.replaceOp(op, res);
    return success();
  }

  static LogicalResult rewriteUnsignedMinMax(Operation *op, Value lhs, Value rhs,
                                             bool chooseMax,
                                             ConversionPatternRewriter &rewriter,
                                             const TypeConverter *typeConverter) {
    Type opTy = op->getResult(0).getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
    Type dstTy = typeConverter->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, op->getLoc(), lhs,
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, op->getLoc(), rhs,
                                                    bitWidth);
    Value cond = makeLessThanCond(rewriter, op->getLoc(), lhsU, rhsU);
    Value res = makeSelect(rewriter, op->getLoc(), dstTy, cond,
                           chooseMax ? rhs : lhs, chooseMax ? lhs : rhs);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMaxSIToEmitC : public OpConversionPattern<arith::MaxSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteSignedMinMax(op, adaptor.getLhs(), adaptor.getRhs(),
                               /*chooseMax=*/true, rewriter,
                               getTypeConverter());
  }
};

struct ArithMinSIToEmitC : public OpConversionPattern<arith::MinSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteSignedMinMax(op, adaptor.getLhs(), adaptor.getRhs(),
                               /*chooseMax=*/false, rewriter,
                               getTypeConverter());
  }
};

struct ArithMaxUIToEmitC : public OpConversionPattern<arith::MaxUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedMinMax(op, adaptor.getLhs(), adaptor.getRhs(),
                                 /*chooseMax=*/true, rewriter,
                                 getTypeConverter());
  }
};

struct ArithMinUIToEmitC : public OpConversionPattern<arith::MinUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedMinMax(op, adaptor.getLhs(), adaptor.getRhs(),
                                 /*chooseMax=*/false, rewriter,
                                 getTypeConverter());
  }
};

// Floating-point max/min variants.
struct ArithFloatMinMaxToEmitCBase {
  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value makeFZero(ConversionPatternRewriter &rewriter, Location loc,
                         Type ty) {
    return makeEmitCOpaqueConstant(rewriter, loc, ty, "0.0f");
  }

  template <bool isMaximum>
  static LogicalResult rewriteMinMaxNumFOp(Operation *op, Location loc, Type dstTy,
                                           Value lhs, Value rhs,
                                           ConversionPatternRewriter &rewriter) {
    Value lhsNaN = isNaN(rewriter, loc, lhs);
    Value rhsNaN = isNaN(rewriter, loc, rhs);
    Value cmpLt =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::lt, lhs, rhs)
            .getResult();
    Value noNaNResult =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt,
                                          isMaximum ? rhs : lhs,
                                          isMaximum ? lhs : rhs)
            .getResult();
    Value rhsOrResult =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, lhs, noNaNResult)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, rhs, rhsOrResult)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMaxNumFToEmitC : public OpConversionPattern<arith::MaxNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    return rewriteMinMaxNumFOp<true>(op, loc, dstTy, adaptor.getLhs(),
                                     adaptor.getRhs(), rewriter);
  }
};

struct ArithMinNumFToEmitC : public OpConversionPattern<arith::MinNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    return rewriteMinMaxNumFOp<false>(op, loc, dstTy, adaptor.getLhs(),
                                      adaptor.getRhs(), rewriter);
  }
};

template <typename ArithOp, bool isMaximum>
struct ArithMinMaxFPropagateNaNToEmitC : public OpConversionPattern<ArithOp>,
                                        ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  static Value buildPrimaryCandidate(ConversionPatternRewriter &rewriter,
                                     Location loc, Type dstTy, Value lhs,
                                     Value rhs) {
    Value cmpLt =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::lt, lhs, rhs)
            .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(
            loc, dstTy, cmpLt, isMaximum ? rhs : lhs, isMaximum ? lhs : rhs)
        .getResult();
  }

  static Value buildSignBitValue(ConversionPatternRewriter &rewriter,
                                 Location loc, Value lhs, FloatType floatTy) {
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    auto templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
        rewriter.getContext(), cast<emitc::OpaqueType>(bitsTy).getValue())});
    Value lhsBits =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, TypeRange{bitsTy}, "ptoas_bitcast",
                                         ValueRange{lhs}, ArrayAttr{},
                                         templateArgs)
            .getResult(0);
    Value oneBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 1);
    Value shiftAmount =
        makeEmitCIntConstant(rewriter, loc, bitsTy, floatTy.getWidth() - 1);
    Value signMask = rewriter
                         .create<emitc::BitwiseLeftShiftOp>(loc, bitsTy, oneBits,
                                                            shiftAmount)
                         .getResult();
    return rewriter
        .create<emitc::BitwiseAndOp>(loc, bitsTy, lhsBits, signMask)
        .getResult();
  }

  static Value buildSignedZeroCandidate(ConversionPatternRewriter &rewriter,
                                        Location loc, Type dstTy, Value lhs,
                                        Value rhs, FloatType floatTy) {
    Value zero = makeFZero(rewriter, loc, dstTy);
    Value equal = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::eq, lhs, rhs)
                      .getResult();
    Value lhsZero = rewriter
                        .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                              emitc::CmpPredicate::eq, lhs,
                                              zero)
                        .getResult();
    Value bothZero = rewriter
                         .create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                      equal, lhsZero)
                         .getResult();
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    Value zeroBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 0);
    Value lhsIsNegZero =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne,
                                  buildSignBitValue(rewriter, loc, lhs, floatTy),
                                  zeroBits)
            .getResult();
    Value tie = rewriter
                    .create<emitc::ConditionalOp>(
                        loc, dstTy, lhsIsNegZero, isMaximum ? rhs : lhs,
                        isMaximum ? lhs : rhs)
                    .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, bothZero, tie,
                                      buildPrimaryCandidate(rewriter, loc, dstTy,
                                                            lhs, rhs))
        .getResult();
  }

  static Value buildNaNPropagatingResult(ConversionPatternRewriter &rewriter,
                                         Location loc, Type dstTy, Value lhs,
                                         Value rhs, FloatType floatTy) {
    Value lhsNaN = isNaN(rewriter, loc, lhs);
    Value rhsNaN = isNaN(rewriter, loc, rhs);
    Value noNaN =
        buildSignedZeroCandidate(rewriter, loc, dstTy, lhs, rhs, floatTy);
    Value rhsOrNoNaN = rewriter
                           .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, rhs,
                                                         noNaN)
                           .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, lhs, rhsOrNoNaN)
        .getResult();
  }

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getType()))
      return rewriter.notifyMatchFailure(op, "expected scalar float type");

    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto floatTy = cast<FloatType>(op.getType());
    rewriter.replaceOp(op, buildNaNPropagatingResult(
                               rewriter, loc, dstTy, adaptor.getLhs(),
                               adaptor.getRhs(), floatTy));
    return success();
  }
};

using ArithMaximumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MaximumFOp, /*isMaximum=*/true>;
using ArithMinimumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MinimumFOp, /*isMaximum=*/false>;

struct ArithMulIToEmitC : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<emitc::MulOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter());
  }
};

struct ArithAddIToEmitC : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<emitc::AddOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter());
  }
};

struct ArithCastOPToEmitC : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    if (adaptor.getIn().getType() == newTy) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, newTy, adaptor.getIn());
    return success();
  }
};

struct ArithSubIToEmitC : public OpConversionPattern<arith::SubIOp> {
  using OpConversionPattern<arith::SubIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SubIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    return rewriteUnsignedBinaryIntLikeOp<emitc::SubOp>(
        op, op.getLoc(), op.getType(), adaptor.getLhs(), adaptor.getRhs(),
        rewriter, getTypeConverter());
  }
};

struct ArithDivSIToEmitC : public OpConversionPattern<arith::DivSIOp> {
  using OpConversionPattern<arith::DivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, dstTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithRemSIToEmitC : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::RemOp>(op, dstTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithTruncIToEmitC : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // to-i1 conversions: Arith wants truncation to the low bit, while C/C++
    // casts to bool are equivalent to `v != 0`. Implement as `(bool)(v & 1)`.
    if (dstIntTy.getWidth() == 1) {
      if (srcIntTy.getWidth() == 1) {
        rewriter.replaceOp(op, adaptor.getIn());
        return success();
      }

      auto uSrcTy =
          getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
      Value inU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                                     srcIntTy.getWidth());
      Value one = makeEmitCIntConstant(rewriter, loc, uSrcTy, 1);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, uSrcTy, inU, one);
      Value asBool = emitCCast(rewriter, loc, dstTy, masked);
      rewriter.replaceOp(op, asBool);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithConstantToEmitC : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    if (!newType)
      return failure();

    // `adaptor.getValue()` may be null if attribute conversion isn't defined.
    // Use the original attribute as fallback and always cast null-safely.
    Attribute valueAttr = adaptor.getValue();
    if (!valueAttr)
      valueAttr = op.getValue();

    if (auto opaqueLiteral = buildEmitCOpaqueConstantLiteral(newType, valueAttr);
        succeeded(opaqueLiteral)) {
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), *opaqueLiteral);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto floatAttr = dyn_cast_or_null<FloatAttr>(valueAttr)) {
      SmallString<kNumber32> valStr;
      floatAttr.getValue().toString(valStr);
      llvm::StringRef s(valStr);
      // Ensure the literal parses as a floating-point constant in C/C++.
      // `APFloat::toString` may emit "1" for integral values; make it "1.0".
      const bool hasFloatMarker =
          s.contains('.') || s.contains('e') || s.contains('E') ||
          s.contains('p') || s.contains('P') || s.starts_with("0x") ||
          s.starts_with("0X") || s.starts_with("nan") ||
          s.starts_with("-nan") || s.starts_with("inf") ||
          s.starts_with("-inf");
      if (!hasFloatMarker)
        valStr.append(".0");
      // Suffix: keep `f` for f16/f32; omit for f64.
      if (!floatAttr.getType().isF64())
        valStr.append("f");
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(valueAttr)) {
      std::string valStr = std::to_string(intAttr.getValue().getSExtValue());
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    return failure();
  }
};

} // namespace

static void populatePTOToEmitCIntegerArithPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &typeConverter,
                                                   MLIRContext *ctx) {
  patterns.add<ArithConstantToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulSIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulIToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddIToEmitC>(typeConverter, ctx);
  patterns.add<ArithSubIToEmitC>(typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::AndIOp, emitc::BitwiseAndOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::OrIOp, emitc::BitwiseOrOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::XOrIOp, emitc::BitwiseXorOp>>(
      typeConverter, ctx);
  patterns.add<ArithShiftLeftToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithFloorDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinUIToEmitC>(typeConverter, ctx);
}

static void populatePTOToEmitCFloatArithPatterns(RewritePatternSet &patterns,
                                                 TypeConverter &typeConverter,
                                                 MLIRContext *ctx) {
  patterns.add<ArithNegFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::SubFOp, emitc::SubOp>>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::MulFOp, emitc::MulOp>>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::DivFOp, emitc::DivOp>>(typeConverter, ctx);
  patterns.add<ArithRemFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaximumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinimumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinNumFToEmitC>(typeConverter, ctx);
}

static void populatePTOToEmitCCastPatterns(RewritePatternSet &patterns,
                                           TypeConverter &typeConverter,
                                           MLIRContext *ctx) {
  patterns.add<ArithSelectToEmitC>(typeConverter, ctx);
  patterns.add<ArithCmpFToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<ArithUIToFPToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<ArithFPToUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<ArithIndexCastUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithBitcastToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<ArithTruncIToEmitC>(typeConverter, ctx);
}

void populatePTOToEmitCArithPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     MLIRContext *ctx) {
  populatePTOToEmitCIntegerArithPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCFloatArithPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCCastPatterns(patterns, typeConverter, ctx);
}

} // namespace mlir::pto
