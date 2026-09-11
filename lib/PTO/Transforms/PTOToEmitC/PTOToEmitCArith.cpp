// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCArith.cpp - arith/inter-core helpers to EmitC patterns ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

void populateArithPatternsPart2(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx, PTOArch targetArch);











//===----------------------------------------------------------------------===//
// Arith -> EmitC (full dialect coverage for scalar ops)
//===----------------------------------------------------------------------===//

// Shared prologue for unsigned-interpretation arith lowering: resolves the
// unsigned-cast operands and their type for a binary integer op. Returns
// failure when the operand type is not a scalar integer/index.
struct UnsignedBinaryOperands {
  emitc::OpaqueType uTy;
  Value lhs;
  Value rhs;
};

static FailureOr<UnsignedBinaryOperands>
getUnsignedBinaryOperands(Operation *op, Value lhs, Value rhs,
                          ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Type opTy = op->getResult(0).getType();
  auto intTy = dyn_cast<IntegerType>(opTy);
  if (!intTy && !isa<IndexType>(opTy)) {
    op->emitError("expected scalar integer or index type");
    return failure();
  }
  const unsigned bitWidth =
      intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return UnsignedBinaryOperands{
      uTy, castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth),
      castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth)};
}

// Scalar integer/index operands are lowered at the 64-bit index width when the
// operand is an index; otherwise the integer's own width is kept.
static unsigned getScalarIntOrIndexBitWidth(Type opTy) {
  if (auto intTy = dyn_cast<IntegerType>(opTy))
    return intTy.getWidth();
  return kPTOIndexBitWidth;
}

static bool isScalarIntOrIndex(Type opTy) {
  return isa<IntegerType, IndexType>(opTy);
}

template <typename ArithOp, typename EmitCOp>
struct ArithSimpleBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getOperands());
    return success();
  }
};

// Integer bitwise/div/rem ops (andi/ori/xori/divui/remui) on signless
// integers: perform in unsigned to avoid signedness pitfalls, then cast back.
// Shared prologue for scalar integer/index arith lowering: validates the
// operand type and resolves the converted result type.
struct ScalarIntOpPrologue {
  Location loc;
  Type dstTy;
};

template <typename PatternTy, typename ArithOp>
static FailureOr<ScalarIntOpPrologue> resolveScalarIntPrologue(
    const PatternTy *pattern, ArithOp op, typename ArithOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  (void)adaptor;
  if (!isScalarIntOrIndex(op.getType()))
    return rewriter.notifyMatchFailure(
        op, "expected scalar integer or index type");
  Type dstTy = pattern->getTypeConverter()->convertType(op.getType());
  if (!dstTy)
    return failure();
  return ScalarIntOpPrologue{op.getLoc(), dstTy};
}

template <typename ArithOp, typename EmitCOp>
struct ArithUnsignedBitwiseBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;
    const unsigned bitWidth = getScalarIntOrIndexBitWidth(op.getType());

    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getLhs(),
                                           adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value resU = rewriter.create<EmitCOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, resU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

using ArithDivUIToEmitC =
    ArithUnsignedBitwiseBinaryToEmitC<arith::DivUIOp, emitc::DivOp>;
using ArithRemUIToEmitC =
    ArithUnsignedBitwiseBinaryToEmitC<arith::RemUIOp, emitc::RemOp>;



struct ArithCeilDivUIToEmitC : public OpConversionPattern<arith::CeilDivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto operands = getUnsignedBinaryOperands(op.getOperation(), adaptor.getLhs(),
                                              adaptor.getRhs(), rewriter);
    if (failed(operands))
      return failure();
    auto &uTy = operands->uTy;
    Value &lhsU = operands->lhs;
    Value &rhsU = operands->rhs;
    Value one = makeEmitCIntConstant(rewriter, loc, uTy, 1);
    Value rhsMinusOne = rewriter.create<emitc::SubOp>(loc, uTy, rhsU, one);
    Value num = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsMinusOne);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, num, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Shared pieces for signed ceil/floor division lowering: the truncating
// quotient/remainder plus the (signs differ?) predicate used by both
// adjustment directions.
struct SignedDivParts {
  Value quotient;
  Value remainder;
  Value remainderNonZero;
  Value signsDiffer;
  Value signsSame;
};

static SignedDivParts buildSignedDivParts(ConversionPatternRewriter &rewriter,
                                          Location loc, Type dstTy, Value lhs,
                                          Value rhs) {
  SignedDivParts parts;
  Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
  parts.quotient = rewriter.create<emitc::DivOp>(loc, dstTy, lhs, rhs);
  parts.remainder = rewriter.create<emitc::RemOp>(loc, dstTy, lhs, rhs);
  parts.remainderNonZero = rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, parts.remainder,
      zero);
  Value lhsLt0 = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                               emitc::CmpPredicate::lt, lhs,
                                               zero);
  Value rhsLt0 = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                               emitc::CmpPredicate::lt, rhs,
                                               zero);
  parts.signsDiffer = rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, lhsLt0, rhsLt0);
  parts.signsSame = rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::eq, lhsLt0, rhsLt0);
  return parts;
}

// Shared tail for unsigned-interpretation binary lowering: resolve the
// unsigned-cast operands, apply the binary op, and cast back to dstTy.
template <typename EmitCOp, typename ArithOp>
static LogicalResult emitUnsignedBinaryAndReplace(
    ArithOp op, typename ArithOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, Location loc, Type dstTy) {
  auto operandsOr = getUnsignedBinaryOperands(op.getOperation(),
                                              adaptor.getLhs(),
                                              adaptor.getRhs(), rewriter);
  if (failed(operandsOr))
    return failure();
  Value resU = rewriter.create<EmitCOp>(loc, operandsOr->uTy,
                                        operandsOr->lhs, operandsOr->rhs);
  rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, resU));
  return success();
}

// Full scalar-int binary lowering flow: shared prologue, i1 special case
// (EmitCI1Op), then the unsigned-interpretation tail. EmitCI1Op is used only
// when the operands are i1; pass EmitCOp itself when no distinct i1 op exists
// (callers with their own i1 handling should not use this driver).
template <typename PatternTy, typename ArithOp, typename EmitCOp,
          typename EmitCI1Op>
static LogicalResult
emitScalarIntBinary(const PatternTy *pattern, ArithOp op,
                    typename ArithOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) {
  FailureOr<ScalarIntOpPrologue> prologue =
      resolveScalarIntPrologue(pattern, op, adaptor, rewriter);
  if (failed(prologue))
    return failure();
  auto [loc, dstTy] = *prologue;

  if (getScalarIntOrIndexBitWidth(op.getType()) == 1) {
    rewriter.replaceOpWithNewOp<EmitCI1Op>(op, op.getType(), adaptor.getLhs(),
                                           adaptor.getRhs());
    return success();
  }
  return emitUnsignedBinaryAndReplace<EmitCOp>(op, adaptor, rewriter, loc,
                                               dstTy);
}

// Shared signed ceil/floor division emission: compensate the truncating
// quotient by +/- 1 when the remainder is non-zero and the sign condition
// holds. CeilDiv adjusts when signs are the same; FloorDiv when they differ.
template <typename ArithOp, bool IsCeil>
struct ArithSignedRoundedDivToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    SignedDivParts parts = buildSignedDivParts(rewriter, loc, dstTy,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    Value signHolds = IsCeil ? parts.signsSame : parts.signsDiffer;
    Value adjust = rewriter.create<emitc::LogicalAndOp>(
        loc, rewriter.getI1Type(), parts.remainderNonZero, signHolds);
    Value compensated =
        IsCeil ? rewriter.create<emitc::AddOp>(loc, dstTy, parts.quotient,
                                               one).getResult()
               : rewriter.create<emitc::SubOp>(loc, dstTy, parts.quotient,
                                               one).getResult();
    Value result =
        rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust, compensated,
                                              parts.quotient);
    rewriter.replaceOp(op, result);
    return success();
  }
};

using ArithCeilDivSIToEmitC =
    ArithSignedRoundedDivToEmitC<arith::CeilDivSIOp, true>;
using ArithFloorDivSIToEmitC =
    ArithSignedRoundedDivToEmitC<arith::FloorDivSIOp, false>;

// Integer shifts on signless operands: compute in the unsigned C++ type of
// the same width, then cast back. i1 shifts are widened to u8 and truncated.
template <typename ArithOp, typename EmitCShiftOp>
struct ArithShiftToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;

    if (getScalarIntOrIndexBitWidth(op.getType()) == 1) {
      // Widen to u8, shift, and truncate back to i1.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<EmitCShiftOp>(loc, u8Ty, lhsU8, rhsU8);
      Value masked = rewriter.create<emitc::BitwiseAndOp>(
          loc, u8Ty, sh,
          makeEmitCIntConstant(rewriter, loc, u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    return emitUnsignedBinaryAndReplace<EmitCShiftOp>(op, adaptor, rewriter,
                                                      loc, dstTy);
  }
};

using ArithShiftLeftToEmitC =
    ArithShiftToEmitC<arith::ShLIOp, emitc::BitwiseLeftShiftOp>;
using ArithShiftRightUIToEmitC =
    ArithShiftToEmitC<arith::ShRUIOp, emitc::BitwiseRightShiftOp>;

struct ArithShiftRightSIToEmitC : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ScalarIntOpPrologue> prologue =
        resolveScalarIntPrologue(this, op, adaptor, rewriter);
    if (failed(prologue))
      return failure();
    auto [loc, dstTy] = *prologue;
    const unsigned bitWidth = getScalarIntOrIndexBitWidth(op.getType());

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

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

// Emit the sign-extension or zero-extension of a signless integer operand to
// `dstTy`. i1 sources are materialized as 0/-1 (signed) or passed through
// (unsigned) before widening.
static LogicalResult emitWidenedInt(Operation *op, Value in, IntegerType srcIntTy,
                                    IntegerType dstIntTy, Type dstTy,
                                    bool isSigned,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  if (srcIntTy.getWidth() == 1) {
    if (isSigned) {
      Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
      Value asInt = emitCCast(rewriter, loc, dstTy, in);
      Value neg = rewriter.create<emitc::SubOp>(loc, dstTy, zero, asInt).getResult();
      rewriter.replaceOp(op, neg);
    } else {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, in);
    }
    return success();
  }
  if (isSigned) {
    // Signed widening relies on the C++ assignment conversion.
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, in);
    return success();
  }
  auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
  Value srcU = castSignlessIntToUnsignedSameWidth(rewriter, loc, in,
                                                  srcIntTy.getWidth());
  Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
  rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, extU));
  return success();
}

struct ArithExtUIToEmitC : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");
    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();
    return emitWidenedInt(op.getOperation(), adaptor.getIn(), srcIntTy,
                          dstIntTy, dstTy, /*isSigned=*/false, rewriter);
  }
};

struct ArithExtSIToEmitC : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern<arith::ExtSIOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");
    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();
    return emitWidenedInt(op.getOperation(), adaptor.getIn(), srcIntTy,
                          dstIntTy, dstTy, /*isSigned=*/true, rewriter);
  }
};

template <typename CastOp>
struct ArithCastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
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

// cmpf helpers: NaN tests, special always-true/false/ORD/UNO forms, and the
// ordered/unordered comparison composition.
struct ArithCmpFConfig {
  bool unordered = false;
  emitc::CmpPredicate predicate = emitc::CmpPredicate::eq;
};

static Value cmpFIsNaN(ConversionPatternRewriter &rewriter, Location loc,
                       Value v) {
  return rewriter
      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                            emitc::CmpPredicate::ne, v, v)
      .getResult();
}

static Value cmpFIsNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                          Value v) {
  return rewriter
      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                            emitc::CmpPredicate::eq, v, v)
      .getResult();
}

static std::optional<Value>
buildSpecialCmpFResult(arith::CmpFPredicate predicate,
                       ConversionPatternRewriter &rewriter, Location loc,
                       Type i1Ty, Value lhs, Value rhs) {
  switch (predicate) {
  case arith::CmpFPredicate::AlwaysFalse:
    return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "false");
  case arith::CmpFPredicate::AlwaysTrue:
    return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "true");
  case arith::CmpFPredicate::ORD:
    return rewriter
        .create<emitc::LogicalAndOp>(loc, i1Ty,
                                     cmpFIsNotNaN(rewriter, loc, lhs),
                                     cmpFIsNotNaN(rewriter, loc, rhs))
        .getResult();
  case arith::CmpFPredicate::UNO:
    return rewriter
        .create<emitc::LogicalOrOp>(loc, i1Ty, cmpFIsNaN(rewriter, loc, lhs),
                                    cmpFIsNaN(rewriter, loc, rhs))
        .getResult();
  default:
    return std::nullopt;
  }
}

static std::optional<ArithCmpFConfig>
getCmpFConfig(arith::CmpFPredicate predicate) {
  switch (predicate) {
  case arith::CmpFPredicate::OEQ:
    return ArithCmpFConfig{false, emitc::CmpPredicate::eq};
  case arith::CmpFPredicate::OGT:
    return ArithCmpFConfig{false, emitc::CmpPredicate::gt};
  case arith::CmpFPredicate::OGE:
    return ArithCmpFConfig{false, emitc::CmpPredicate::ge};
  case arith::CmpFPredicate::OLT:
    return ArithCmpFConfig{false, emitc::CmpPredicate::lt};
  case arith::CmpFPredicate::OLE:
    return ArithCmpFConfig{false, emitc::CmpPredicate::le};
  case arith::CmpFPredicate::ONE:
    return ArithCmpFConfig{false, emitc::CmpPredicate::ne};
  case arith::CmpFPredicate::UEQ:
    return ArithCmpFConfig{true, emitc::CmpPredicate::eq};
  case arith::CmpFPredicate::UGT:
    return ArithCmpFConfig{true, emitc::CmpPredicate::gt};
  case arith::CmpFPredicate::UGE:
    return ArithCmpFConfig{true, emitc::CmpPredicate::ge};
  case arith::CmpFPredicate::ULT:
    return ArithCmpFConfig{true, emitc::CmpPredicate::lt};
  case arith::CmpFPredicate::ULE:
    return ArithCmpFConfig{true, emitc::CmpPredicate::le};
  case arith::CmpFPredicate::UNE:
    return ArithCmpFConfig{true, emitc::CmpPredicate::ne};
  default:
    return std::nullopt;
  }
}

static Value buildCmpFResult(const ArithCmpFConfig &config,
                             ConversionPatternRewriter &rewriter, Location loc,
                             Type i1Ty, Value lhs, Value rhs) {
  Value cmp = rewriter
                  .create<emitc::CmpOp>(loc, i1Ty, config.predicate, lhs, rhs)
                  .getResult();
  Value unord = rewriter.create<emitc::LogicalOrOp>(
      loc, i1Ty, cmpFIsNaN(rewriter, loc, lhs), cmpFIsNaN(rewriter, loc, rhs));
  if (config.unordered)
    return rewriter.create<emitc::LogicalOrOp>(loc, i1Ty, unord, cmp)
        .getResult();
  Value ord = rewriter.create<emitc::LogicalAndOp>(
      loc, i1Ty, cmpFIsNotNaN(rewriter, loc, lhs),
      cmpFIsNotNaN(rewriter, loc, rhs));
  return rewriter.create<emitc::LogicalAndOp>(loc, i1Ty, ord, cmp).getResult();
}

struct ArithCmpFToEmitC : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

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
    if (newResultTypes.size() != 2)
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
    if (newResultTypes.size() != 2)
      return failure();

    Type lowDstTy = newResultTypes[0];
    Type highDstTy = newResultTypes[1];

    Type wideTy = isUnsigned ? static_cast<Type>(getWiderUnsignedIntOpaqueType(rewriter.getContext(),
                                                                               bitWidth))
                             : static_cast<Type>(getWiderSignedIntOpaqueType(rewriter.getContext(),
                                                                             bitWidth));

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
};

// min/max integer lowering: `select(lhs < rhs, A, B)` where (A, B) picks the
// smaller operand for min and the larger for max. Unsigned variants compare
// through the unsigned C++ type of the same width so values with the sign bit
// set order correctly.
template <typename ArithOp, emitc::CmpPredicate Pred, bool TakeRhsOnTrue,
          bool IsUnsigned>
struct ArithMinMaxIToEmitC : public OpConversionPattern<ArithOp>,
                             ArithMinMaxIToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if constexpr (IsUnsigned) {
      unsigned bitWidth = getScalarIntOrIndexBitWidth(op.getType());
      lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
      rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
    }
    Value cond =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(), Pred, lhs,
                                      rhs)
            .getResult();
    Value onTrue = TakeRhsOnTrue ? adaptor.getRhs() : adaptor.getLhs();
    Value onFalse = TakeRhsOnTrue ? adaptor.getLhs() : adaptor.getRhs();
    rewriter.replaceOp(
        op, makeSelect(rewriter, loc, dstTy, cond, onTrue, onFalse));
    return success();
  }
};

using ArithMaxSIToEmitC =
    ArithMinMaxIToEmitC<arith::MaxSIOp, emitc::CmpPredicate::lt, true, false>;
using ArithMinSIToEmitC =
    ArithMinMaxIToEmitC<arith::MinSIOp, emitc::CmpPredicate::lt, false, false>;
using ArithMaxUIToEmitC =
    ArithMinMaxIToEmitC<arith::MaxUIOp, emitc::CmpPredicate::lt, true, true>;
using ArithMinUIToEmitC =
    ArithMinMaxIToEmitC<arith::MinUIOp, emitc::CmpPredicate::lt, false, true>;

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
};

// maxnum/minnum lowering: a plain lt-based min/max plus NaN-propagation
// selects on both operands.
template <typename ArithOp, bool IsMax>
struct ArithNumFToEmitC : public OpConversionPattern<ArithOp>,
                          ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    // max picks rhs when lhs < rhs; min picks lhs.
    Value noNaN = rewriter
                      .create<emitc::ConditionalOp>(
                          loc, dstTy, cmpLt,
                          IsMax ? adaptor.getRhs() : adaptor.getLhs(),
                          IsMax ? adaptor.getLhs() : adaptor.getRhs())
                      .getResult();

    // Propagate whichever operand is NaN, preferring rhs on double-NaN.
    Value rhsOrPick = rewriter
                          .create<emitc::ConditionalOp>(
                              loc, dstTy, rhsNaN, adaptor.getLhs(), noNaN)
                          .getResult();
    Value res = rewriter
                    .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN,
                                                  adaptor.getRhs(), rhsOrPick)
                    .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

using ArithMaxNumFToEmitC = ArithNumFToEmitC<arith::MaxNumFOp, true>;
using ArithMinNumFToEmitC = ArithNumFToEmitC<arith::MinNumFOp, false>;

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

//===----------------------------------------------------------------------===//
// Arith -> EmitC helpers
//===----------------------------------------------------------------------===//









// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.

// muli/addi/subi on signless integers: compute in the unsigned C++ type of
// the same width, then cast back. i1 arithmetic wraps to a single bit, so
// mul lowers to AND and add/sub to XOR.
template <typename ArithOp, typename EmitCOp, typename EmitCI1Op>
struct ArithIntBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using PatternTy = ArithIntBinaryToEmitC;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return emitScalarIntBinary<PatternTy, ArithOp, EmitCOp, EmitCI1Op>(
        this, op, adaptor, rewriter);
  }
};

using ArithMulIToEmitC =
    ArithIntBinaryToEmitC<arith::MulIOp, emitc::MulOp, emitc::BitwiseAndOp>;
using ArithAddIToEmitC =
    ArithIntBinaryToEmitC<arith::AddIOp, emitc::AddOp, emitc::BitwiseXorOp>;
using ArithSubIToEmitC =
    ArithIntBinaryToEmitC<arith::SubIOp, emitc::SubOp, emitc::BitwiseXorOp>;

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

struct ArithDivSIToEmitC : public OpConversionPattern<arith::DivSIOp> {
  using OpConversionPattern<arith::DivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithRemSIToEmitC : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::RemOp>(op, newTy, adaptor.getLhs(),
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
      SmallString<32> valStr;
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
      std::string valStr = std::to_string(getIntegerAttrSignedValue(intAttr));
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    return failure();
  }
};
//===----------------------------------------------------------------------===//
// pto.mgather lowering -> MGATHER(dst, src, indexes)  (pto-isa)
//===----------------------------------------------------------------------===//

struct PTOMGatherToMGATHER : public OpConversionPattern<pto::MGatherOp> {
  using OpConversionPattern<pto::MGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // MGATHER is a template intrinsic that accepts the concrete descriptor
    // directly, so peel any type-converter materialization bridge and feed the
    // producing value. This keeps the compile-time static-stride GlobalTensor
    // from the partition_view pattern instead of the dynamic-stride bridge,
    // whose GlobalTensor<...> C-style cast would not compile (issue #1165).
    Value mem = peelUnrealized(adaptor.getMem());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    Value memArg = mem;
    auto coalescePropAttr =
        dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
    auto gatherOobAttr =
        dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
    pto::GatherOOB gatherOob =
        gatherOobAttr ? gatherOobAttr.getValue() : pto::GatherOOB::Undefined;

    // GM -> L1 uses a partition view; GM -> UB uses a tile.
    Value idxArg = idx;

    if (!coalescePropAttr)
      return op.emitError(
          "expects mgather to specify an explicit coalesce attribute (row or "
          "elem)");

    ArrayAttr templateArgs = buildMGatherTemplateArgs(op, rewriter, gatherOob);

    // GM -> L1 Coalesce::Elem stages elements through a GM scratch buffer, passed
    // as the 4th MGATHER argument; Row and the GM -> UB path have no scratch.
    SmallVector<Value, 4> callArgs{dst, memArg, idxArg};
    if (Value scratch = adaptor.getScratch()) {
      callArgs.push_back(peelUnrealized(scratch));
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MGATHER",
        ArrayAttr{}, templateArgs,
        ValueRange(callArgs));

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, dst);
    }
    return success();
  }

  // MGATHER's template list: the mandatory Coalesce mode plus the optional
  // out-of-bounds handling mode.
  ArrayAttr buildMGatherTemplateArgs(pto::MGatherOp op,
                                     ConversionPatternRewriter &rewriter,
                                     pto::GatherOOB gatherOob) const {
    auto *ctx = rewriter.getContext();
    auto coalescePropAttr =
        dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
    auto coalesceTok = [](pto::Coalesce mode) -> StringRef {
      switch (mode) {
      case pto::Coalesce::Row:
        return "pto::Coalesce::Row";
      case pto::Coalesce::Elem:
        return "pto::Coalesce::Elem";
      }
      llvm_unreachable("unknown Coalesce");
    };
    auto gatherOobTok = [](pto::GatherOOB mode) -> StringRef {
      switch (mode) {
      case pto::GatherOOB::Undefined:
        return "pto::GatherOOB::Undefined";
      case pto::GatherOOB::Clamp:
        return "pto::GatherOOB::Clamp";
      case pto::GatherOOB::Wrap:
        return "pto::GatherOOB::Wrap";
      case pto::GatherOOB::Zero:
        return "pto::GatherOOB::Zero";
      }
      llvm_unreachable("unknown GatherOOB");
    };

    SmallVector<Attribute, 2> templateArgVec;
    templateArgVec.push_back(
        emitc::OpaqueAttr::get(ctx, coalesceTok(coalescePropAttr.getValue())));
    if (op.getGatherOob() != pto::GatherOOB::Undefined) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, gatherOobTok(gatherOob)));
    }
    return templateArgVec.empty() ? ArrayAttr{}
                                  : rewriter.getArrayAttr(templateArgVec);
  }
};

struct AffineApplyMulConstToEmitC
    : public OpConversionPattern<affine::AffineApplyOp> {
  using OpConversionPattern<affine::AffineApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();

    if (map.getNumDims() != 0 || map.getNumSymbols() != 1)
      return failure();

    auto expr = map.getResult(0);
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin || bin.getKind() != AffineExprKind::Mul)
      return failure();

    auto lhs = bin.getLHS();
    auto rhs = bin.getRHS();

    auto symExpr = dyn_cast<AffineSymbolExpr>(lhs);
    auto constExpr = dyn_cast<AffineConstantExpr>(rhs);
    if (!symExpr || !constExpr)
      return failure();

    Value inputVal = adaptor.getMapOperands()[0];

    std::string valStr = std::to_string(constExpr.getValue());
    auto cstAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
    auto cstOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), inputVal.getType(), cstAttr);

    rewriter.replaceOpWithNewOp<emitc::MulOp>(
        op, inputVal.getType(), inputVal, cstOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Kernel inference helpers
//===----------------------------------------------------------------------===//

enum class KernelKind { VecAdd, Matmul, Unknown };

[[maybe_unused]] static KernelKind inferKernelKind(func::FuncOp f) {
  bool hasAdd = false;
  bool hasMM  = false;
  f.walk([&](Operation *op) {
    if (isa<mlir::pto::TAddOp>(op)) {
      hasAdd = true;
    }
    if (isa<mlir::pto::TMatmulOp>(op)) {
      hasMM = true;
    }
    if (isa<mlir::pto::TMatmulAccOp>(op)) {
      hasMM = true;
    }
  });
  if (hasMM) {
    return KernelKind::Matmul;
  }
  if (hasAdd) {
    return KernelKind::VecAdd;
  }
  return KernelKind::Unknown;
}

[[maybe_unused]] static void inferTileMNK(func::FuncOp f, int &M, int &N, int &K) {
  M = 32; N = 32; K = 32;
  SmallVector<memref::SubViewOp, 4> subs;
  f.walk([&](memref::SubViewOp sv) { subs.push_back(sv); });

  auto readShape2D = [&](memref::SubViewOp sv, int &d0, int &d1) {
    auto resTy = mlir::cast<MemRefType>(sv.getResult().getType());
    if (resTy.getRank() == 2 && resTy.hasStaticShape()) {
      d0 = static_cast<int>(resTy.getDimSize(0));
      d1 = static_cast<int>(resTy.getDimSize(1));
    }
  };

  if (subs.empty()) {
    return;
  }

  int a0=32, a1=32;
  readShape2D(subs[0], a0, a1);
  M = a0; N = a1;

  if (subs.size() >= 2) {
    int b0=32, b1=32;
    readShape2D(subs[0], a0, a1);
    readShape2D(subs[1], b0, b1);
    M = a0; K = a1; N = b1;
  }
}




// Pick the C++ specifiers (extern "C"/static/__global__) for the emitted
// AICORE function based on its linkage and PTO entry attributes.
static void applyFuncSpecifiers(func::FuncOp op,
                                ConversionPatternRewriter &rewriter,
                                emitc::FuncOp &emitcFunc) {
  if (pto::isPTOEntryFunction(op)) {
    emitcFunc.setSpecifiersAttr(
        rewriter.getStrArrayAttr({"extern \"C\"", "__global__ AICORE"}));
  } else if (op.isPrivate()) {
    emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"static", "AICORE"}));
  } else if (pto::hasExternalArtifactVisibility(op)) {
    emitcFunc.setSpecifiersAttr(
        rewriter.getStrArrayAttr({"extern \"C\"", "AICORE"}));
  } else {
    emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"AICORE"}));
  }
}

struct FuncToEmitC : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature with the type converter.
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    // Create the EmitC function with the converted signature.
    auto emitcFunc =
        rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), funcType);

    for (const auto &namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName().strref();
      if (name == op.getFunctionTypeAttrName() ||
          name == SymbolTable::getSymbolAttrName() ||
          name == pto::kPTOEntryAttrName ||
          name == pto::kLegacyHACCEntryAttrName)
        continue;
      emitcFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    if (op.isDeclaration()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"extern \"C\"", "AICORE"}));
      rewriter.eraseOp(op);
      return success();
    }

    applyFuncSpecifiers(op, rewriter, emitcFunc);

    // The no-split guard decision walks the function body, so it must be
    // resolved before the body is inlined into the EmitC function below.
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    if (failed(convertFuncBody(op, emitcFunc, funcType, rewriter)))
      return failure();

    emitFuncPrologueAndEpilogueMacros(op, emitcFunc, rewriter, needsNoSplitGuard);

    rewriter.eraseOp(op);
    return success();
  }

  // Inline the original body, then convert region/block argument types to
  // match the converted signature (also covers CFG blocks introduced by
  // pre-lowering, e.g. scf.while -> cf.br/cf.cond_br).
  LogicalResult convertFuncBody(func::FuncOp op, emitc::FuncOp emitcFunc,
                                FunctionType funcType,
                                ConversionPatternRewriter &rewriter) const {
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(),
                                emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));

    return rewriter.convertRegionTypes(&emitcFunc.getBody(),
                                       *getTypeConverter(), &entryConv);
  }

  // Preserve the existing function prologue shape. `kernel_kind` functions are
  // emitted with the same macro guard/reset sequence that used to come from
  // early pto.section wrapping, but only after SCF pre-lowering has finished.
  void emitFuncPrologueAndEpilogueMacros(func::FuncOp op,
                                         emitc::FuncOp emitcFunc,
                                         ConversionPatternRewriter &rewriter,
                                         bool needsNoSplitGuard) const {
    std::optional<StringRef> kernelKindMacro = getKernelKindMacro(op);

    {
      Block &entryBlock = emitcFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
      if (kernelKindMacro) {
        std::string startMacro = "\n#if defined(" + kernelKindMacro->str() + ")";
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), startMacro);
        if (*kernelKindMacro == "__DAV_VEC__") {
          rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_mask_norm();");
          rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                             "set_vector_mask(-1, -1);");
          if (needsNoSplitGuard)
            rewriter.create<emitc::VerbatimOp>(
                op.getLoc(), "if (get_subblockid() == 0) {");
        }
      }
    }

    if (kernelKindMacro) {
      Block &lastBlock = emitcFunc.getBody().back();
      rewriter.setInsertionPoint(lastBlock.getTerminator());
      if (*kernelKindMacro == "__DAV_VEC__" && needsNoSplitGuard)
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), "}");
      std::string endMacro = "#endif // " + kernelKindMacro->str() + "\n";
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), endMacro);
    }
  }
};

//===----------------------------------------------------------------------===//
// SubView lowering to GlobalTensor (keep your existing code)
//===----------------------------------------------------------------------===


static InterCoreSyncCallDesc buildInterCoreSyncSetCallImpl(
    ConversionPatternRewriter &rewriter, Value msgVal, PTOArch targetArch,
    pto::PipeAttr pipeAttr) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  (void)targetArch;
  InterCoreSyncCallDesc desc;
  desc.callee = "__builtin_cce_ffts_cross_core_sync";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(msgVal);
  return desc;
}

InterCoreSyncCallDesc buildInterCoreSyncSetCall(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) {
  auto indexTy = emitc::OpaqueType::get(rewriter.getContext(), "int64_t");
  Value eventVal =
      makeEmitCIntConstant(rewriter, loc, indexTy,
                           getIntegerAttrSignedValue(eventIdAttr));
  Value msgVal = createFFTSMsg(rewriter, loc, eventVal, fftsMode);
  return buildInterCoreSyncSetCallImpl(rewriter, msgVal, targetArch, pipeAttr);
}

InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) {
  Value msgVal = createFFTSMsg(rewriter, loc, eventIdVal, fftsMode);
  return buildInterCoreSyncSetCallImpl(rewriter, msgVal, targetArch, pipeAttr);
}

InterCoreSyncCallDesc buildInterCoreSyncWaitCall(
    ConversionPatternRewriter &rewriter, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) {
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  InterCoreSyncCallDesc desc;
  (void)targetArch;
  (void)pipeTok;
  desc.callee = "__builtin_cce_wait_flag_dev";
  desc.args = rewriter.getArrayAttr({eventIdAttr});
  return desc;
}

InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  InterCoreSyncCallDesc desc;
  (void)targetArch;
  (void)pipeTok;
  desc.callee = "__builtin_cce_wait_flag_dev";
  desc.args = rewriter.getArrayAttr({IntegerAttr::get(IndexType::get(ctx), 0)});
  desc.operands.push_back(castInterCoreEventIdToI32(rewriter, loc, eventIdVal));
  return desc;
}

Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter,
                                       Location loc, Value eventId) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  if (eventId.getType() == i32Ty)
    return eventId;
  return emitCCast(rewriter, loc, i32Ty, eventId);
}

Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventId, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  auto msgTy = emitc::OpaqueType::get(ctx, "uint16_t");
  auto msgArgs = rewriter.getArrayAttr({
      getFFTSModeCodegenArg(rewriter, fftsMode),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, msgTy, "getFFTSMsg",
                                   /*args=*/msgArgs,
                                   /*templateArgs=*/ArrayAttr{},
                                   /*operands=*/ValueRange{eventId})
      .getResult(0);
}

Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter,
                                       int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  if (fftsMode == 2)
    return emitc::OpaqueAttr::get(ctx, "FFTS_MODE_VAL");
  return emitc::OpaqueAttr::get(ctx, std::to_string(fftsMode));
}

bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SyncSetOp, pto::SyncWaitOp, pto::SetCrossBlockOp,
            pto::WaitCrossBlockOp, pto::SetIntraBlockOp,
            pto::WaitIntraBlockOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

bool hasSetFFTsOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}


std::optional<StringRef> getKernelKindMacro(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return StringRef("__DAV_CUBE__");
  case FunctionKernelKind::Vector:
    return StringRef("__DAV_VEC__");
  }

  llvm_unreachable("unexpected kernel kind");
}


void populateArithPatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  patterns.add<ArithConstantToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulSIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<AffineApplyMulConstToEmitC>(typeConverter, ctx);
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
    populateArithPatternsPart2(patterns, typeConverter, ctx, targetArch);
}

void populateArithPatternsPart2(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx, PTOArch targetArch) {
  (void)typeConverter;
  (void)ctx;
  (void)targetArch;
  patterns.add<ArithMinSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithNegFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::SubFOp, emitc::SubOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::MulFOp, emitc::MulOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::DivFOp, emitc::DivOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithRemFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaximumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinimumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinNumFToEmitC>(typeConverter, ctx);
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
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<ArithTruncIToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
