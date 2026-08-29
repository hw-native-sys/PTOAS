// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOLowerGenericOps.cpp --------------------------------------------===//
//
// Lower execution-domain-independent frontend pto.* scalar and builtin-vector
// operations to core arith/math/LLVM operations. This pass runs only after
// frontend PTO IR has been formed, so the public DSL and its seam IR expose one
// PTO dialect instead of leaking implementation dialect choices.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWERGENERICOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static Value castIntegerSignedness(PatternRewriter &rewriter, Location loc,
                                   Value value, Type targetType) {
  bool alreadyHasTargetType = value.getType() == targetType;
  if (alreadyHasTargetType) {
    return value;
  }
  return rewriter
      .create<UnrealizedConversionCastOp>(loc, TypeRange{targetType}, value)
      .getResult(0);
}

static Type getValueElementType(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.getElementType();
  }
  return type;
}

static void copyAttrIfPresent(Operation *source, Operation *target,
                              StringRef name) {
  Attribute attr = source->getAttr(name);
  if (!attr) {
    return;
  }
  MLIRContext *context = source->getContext();
  if (name == "overflowFlags") {
    auto sourceAttr = cast<pto::IntegerOverflowFlagsAttr>(attr);
    target->setAttr(
        name, arith::IntegerOverflowFlagsAttr::get(
                  context, static_cast<arith::IntegerOverflowFlags>(
                               static_cast<uint32_t>(sourceAttr.getValue()))));
    return;
  }
  if (name == "fastmath") {
    auto sourceAttr = cast<pto::FastMathFlagsAttr>(attr);
    target->setAttr(
        name, arith::FastMathFlagsAttr::get(
                  context, static_cast<arith::FastMathFlags>(
                               static_cast<uint32_t>(sourceAttr.getValue()))));
    return;
  }
  if (name == "roundingmode") {
    auto sourceAttr = cast<pto::FloatRoundingModeAttr>(attr);
    target->setAttr(
        name, arith::RoundingModeAttr::get(
                  context, static_cast<arith::RoundingMode>(
                               static_cast<uint32_t>(sourceAttr.getValue()))));
    return;
  }
  target->setAttr(name, attr);
}

static void copyFastMathToLLVM(Operation *source, Operation *target) {
  auto sourceAttr = source->getAttrOfType<pto::FastMathFlagsAttr>("fastmath");
  if (!sourceAttr) {
    return;
  }
  pto::FastMathFlags sourceFlags = sourceAttr.getValue();
  LLVM::FastmathFlags targetFlags{};
  const std::pair<pto::FastMathFlags, LLVM::FastmathFlags> flags[] = {
      {pto::FastMathFlags::nnan, LLVM::FastmathFlags::nnan},
      {pto::FastMathFlags::ninf, LLVM::FastmathFlags::ninf},
      {pto::FastMathFlags::nsz, LLVM::FastmathFlags::nsz},
      {pto::FastMathFlags::arcp, LLVM::FastmathFlags::arcp},
      {pto::FastMathFlags::contract, LLVM::FastmathFlags::contract},
      {pto::FastMathFlags::afn, LLVM::FastmathFlags::afn},
      {pto::FastMathFlags::reassoc, LLVM::FastmathFlags::reassoc},
  };
  for (auto [sourceFlag, targetFlag] : flags) {
    if (bitEnumContainsAny(sourceFlags, sourceFlag)) {
      targetFlags = targetFlags | targetFlag;
    }
  }
  target->setAttr("fastmathFlags", LLVM::FastmathFlagsAttr::get(
                                       source->getContext(), targetFlags));
}

static bool isPackedTwoLaneFloat(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  bool isNotTwoLaneVector = !vectorType || vectorType.getNumElements() != 2;
  if (isNotTwoLaneVector) {
    return false;
  }
  Type elementType = vectorType.getElementType();
  return elementType.isF16() || elementType.isBF16();
}

static bool isInsideSimtExecutionScope(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return (funcOp && funcOp->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static bool hasLowPrecisionConversionPayload(Type type) {
  return pto::isPTOLowPrecisionType(getValueElementType(type));
}

static bool isPublicGenericPTOOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.constant" || name == "pto.addi" ||
         name == "pto.addf" || name == "pto.subi" || name == "pto.subf" ||
         name == "pto.muli" || name == "pto.mulf" ||
         name == "pto.addui_extended" || name == "pto.mul_extended" ||
         name == "pto.negi" || name == "pto.negf" || name == "pto.divi" ||
         name == "pto.divf" || name == "pto.floordiv" ||
         name == "pto.ceildiv" || name == "pto.remi" || name == "pto.remf" ||
         name == "pto.and" || name == "pto.or" || name == "pto.xor" ||
         name == "pto.shl" || name == "pto.shr" || name == "pto.cmpi" ||
         name == "pto.cmpf" || name == "pto.maxi" || name == "pto.maxf" ||
         name == "pto.mini" || name == "pto.minf" ||
         name == "pto.maximum" || name == "pto.minimum" ||
         name == "pto.absi" || name == "pto.absf" || name == "pto.exti" ||
         name == "pto.trunci" || name == "pto.ftof" || name == "pto.ftoi" ||
         name == "pto.itof" || name == "pto.index_cast" ||
         name == "pto.select" || name == "pto.exp" ||
         name == "pto.log" || name == "pto.sqrt" || name == "pto.pow" ||
         name == "pto.fma";
}

static bool requiresSimtHardwareConversion(Operation *op, Type srcType,
                                           Type dstType,
                                           pto::FloatRoundingModeAttr rounding,
                                           pto::Saturation saturation) {
  return isInsideSimtExecutionScope(op) &&
         (hasLowPrecisionConversionPayload(srcType) ||
          hasLowPrecisionConversionPayload(dstType) ||
          saturation == pto::Saturation::Enable || rounding);
}

static bool isPackedTwoLaneF16(Type type);

static bool isExpectedGenericResidual(Operation *op) {
  if (auto absf = dyn_cast<pto::AbsFOp>(op)) {
    return isPackedTwoLaneFloat(absf.getResult().getType());
  }
  if (auto exp = dyn_cast<pto::ExpOp>(op)) {
    return isPackedTwoLaneF16(exp.getResult().getType());
  }
  if (auto log = dyn_cast<pto::LogOp>(op)) {
    return isPackedTwoLaneF16(log.getResult().getType());
  }
  if (auto sqrt = dyn_cast<pto::SqrtOp>(op)) {
    return isPackedTwoLaneF16(sqrt.getResult().getType());
  }
  if (auto pow = dyn_cast<pto::PowOp>(op)) {
    return isPackedTwoLaneFloat(pow.getResult().getType());
  }
  if (auto fma = dyn_cast<pto::FmaOp>(op)) {
    return isPackedTwoLaneFloat(fma.getResult().getType());
  }
  if (auto ftof = dyn_cast<pto::FToFOp>(op)) {
    return requiresSimtHardwareConversion(
        op, ftof.getSrc().getType(), ftof.getDst().getType(),
        ftof.getRoundingmodeAttr(), ftof.getSaturation());
  }
  if (auto ftoi = dyn_cast<pto::FToIOp>(op)) {
    return requiresSimtHardwareConversion(
        op, ftoi.getSrc().getType(), ftoi.getDst().getType(),
        ftoi.getRoundingmodeAttr(), ftoi.getSaturation());
  }
  if (auto itof = dyn_cast<pto::IToFOp>(op)) {
    return requiresSimtHardwareConversion(
        op, itof.getSrc().getType(), itof.getDst().getType(),
        itof.getRoundingmodeAttr(), itof.getSaturation());
  }
  return false;
}

static bool isPackedTwoLaneF16(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getNumElements() == 2 &&
         vectorType.getElementType().isF16();
}

static Type getSignlessIntegerType(PatternRewriter &rewriter, Type type) {
  auto intType = cast<IntegerType>(getValueElementType(type));
  Type signlessElementType = rewriter.getIntegerType(intType.getWidth());
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return VectorType::Builder(vectorType).setElementType(signlessElementType);
  }
  return signlessElementType;
}

static Value stripIntegerSignedness(PatternRewriter &rewriter, Location loc,
                                    Value value) {
  auto intType = dyn_cast<IntegerType>(getValueElementType(value.getType()));
  if (!intType || intType.isSignless()) {
    return value;
  }
  return castIntegerSignedness(
      rewriter, loc, value, getSignlessIntegerType(rewriter, value.getType()));
}

struct LowerConstantPattern final : OpRewritePattern<pto::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getResult().getType();
    if (auto intType = dyn_cast<IntegerType>(getValueElementType(type));
        intType && !intType.isSignless()) {
      Type signlessType = getSignlessIntegerType(rewriter, type);
      TypedAttr signlessValue;
      if (auto value = dyn_cast<IntegerAttr>(op.getValue())) {
        signlessValue = rewriter.getIntegerAttr(signlessType, value.getValue());
      } else if (auto value = dyn_cast<DenseElementsAttr>(op.getValue())) {
        signlessValue = value.mapValues(
            cast<IntegerType>(getValueElementType(signlessType)),
            [](const APInt &element) { return element; });
      } else {
        return rewriter.notifyMatchFailure(
            op, "expected integer or dense integer attribute");
      }
      Value constant = rewriter.create<arith::ConstantOp>(
          op.getLoc(), signlessType, signlessValue);
      rewriter.replaceOp(
          op, castIntegerSignedness(rewriter, op.getLoc(), constant, type));
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, type, op.getValue());
    return success();
  }
};

template <typename PTOOp, typename IntegerOp>
struct LowerBinaryIntegerPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getResult().getType();
    if (isa<IndexType>(type)) {
      auto lowered = rewriter.template create<IntegerOp>(
          op.getLoc(), op.getLhs(), op.getRhs());
      copyAttrIfPresent(op, lowered, "overflowFlags");
      rewriter.replaceOp(op, lowered);
      return success();
    }
    Value lhs = stripIntegerSignedness(rewriter, loc, op.getLhs());
    Value rhs = stripIntegerSignedness(rewriter, loc, op.getRhs());
    auto lowered = rewriter.template create<IntegerOp>(loc, lhs, rhs);
    copyAttrIfPresent(op, lowered, "overflowFlags");
    Value result = lowered;
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, result, type));
    return success();
  }
};

template <typename PTOOp, typename FloatOp>
struct LowerBinaryFloatPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    auto lowered = rewriter.template create<FloatOp>(op.getLoc(), op.getLhs(),
                                                     op.getRhs());
    copyAttrIfPresent(op, lowered, "fastmath");
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

using LowerAddIPattern = LowerBinaryIntegerPattern<pto::AddIOp, arith::AddIOp>;
using LowerAddFPattern = LowerBinaryFloatPattern<pto::AddFOp, arith::AddFOp>;
using LowerSubIPattern = LowerBinaryIntegerPattern<pto::SubIOp, arith::SubIOp>;
using LowerSubFPattern = LowerBinaryFloatPattern<pto::SubFOp, arith::SubFOp>;
using LowerMulIPattern = LowerBinaryIntegerPattern<pto::MulIOp, arith::MulIOp>;
using LowerMulFPattern = LowerBinaryFloatPattern<pto::MulFOp, arith::MulFOp>;

struct LowerNegIPattern final : OpRewritePattern<pto::NegIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::NegIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getResult().getType();
    if (isa<IndexType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto lowered = rewriter.create<arith::SubIOp>(loc, zero, op.getValue());
      copyAttrIfPresent(op, lowered, "overflowFlags");
      rewriter.replaceOp(op, lowered);
      return success();
    }
    Value value = stripIntegerSignedness(rewriter, loc, op.getValue());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, value.getType(), rewriter.getZeroAttr(value.getType()));
    auto lowered = rewriter.create<arith::SubIOp>(loc, zero, value);
    copyAttrIfPresent(op, lowered, "overflowFlags");
    Value result = lowered;
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, result, type));
    return success();
  }
};

struct LowerNegFPattern final : OpRewritePattern<pto::NegFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::NegFOp op,
                                PatternRewriter &rewriter) const override {
    auto lowered = rewriter.create<arith::NegFOp>(op.getLoc(), op.getValue());
    copyAttrIfPresent(op, lowered, "fastmath");
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct LowerDivFPattern final : OpRewritePattern<pto::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DivFOp op,
                                PatternRewriter &rewriter) const override {
    auto lowered =
        rewriter.create<arith::DivFOp>(op.getLoc(), op.getLhs(), op.getRhs());
    copyAttrIfPresent(op, lowered, "fastmath");
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

template <typename PTOOp, typename SignedArithOp, typename UnsignedArithOp>
struct LowerSignednessIntegerPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getResult().getType();
    auto signedness = op.getSignednessAttr().getValue();
    if (isa<IndexType>(type)) {
      auto lowered = signedness == pto::Signedness::Unsigned
                         ? rewriter.template create<UnsignedArithOp>(
                               loc, op.getLhs(), op.getRhs())
                         : rewriter.template create<SignedArithOp>(
                               loc, op.getLhs(), op.getRhs());
      rewriter.replaceOp(op, lowered);
      return success();
    }
    if (!isa<IntegerType>(getValueElementType(type))) {
      return rewriter.notifyMatchFailure(
          op, "expected integer, index, or builtin vector of integers");
    }
    auto lowered = signedness == pto::Signedness::Unsigned
                       ? rewriter.template create<UnsignedArithOp>(
                             loc, op.getLhs(), op.getRhs())
                       : rewriter.template create<SignedArithOp>(
                             loc, op.getLhs(), op.getRhs());
    copyAttrIfPresent(op, lowered, "overflowFlags");
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

using LowerDivPattern =
    LowerSignednessIntegerPattern<pto::DivIOp, arith::DivSIOp, arith::DivUIOp>;
using LowerFloorDivPattern =
    LowerSignednessIntegerPattern<pto::FloorDivOp, arith::FloorDivSIOp,
                                  arith::DivUIOp>;
using LowerCeilDivPattern =
    LowerSignednessIntegerPattern<pto::CeilDivOp, arith::CeilDivSIOp,
                                  arith::CeilDivUIOp>;
using LowerRemPattern =
    LowerSignednessIntegerPattern<pto::RemIOp, arith::RemSIOp, arith::RemUIOp>;
using LowerRemFPattern = LowerBinaryFloatPattern<pto::RemFOp, arith::RemFOp>;

struct LowerMulExtendedPattern final : OpRewritePattern<pto::MulExtendedOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::MulExtendedOp op,
    PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool isUnsigned =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned;
    if (isUnsigned) {
      auto lowered = rewriter.create<arith::MulUIExtendedOp>(loc, op.getLhs(),
                                                             op.getRhs());
      rewriter.replaceOp(op, ValueRange{lowered.getLow(), lowered.getHigh()});
    } else {
      auto lowered = rewriter.create<arith::MulSIExtendedOp>(loc, op.getLhs(),
                                                             op.getRhs());
      rewriter.replaceOp(op, ValueRange{lowered.getLow(), lowered.getHigh()});
    }
    return success();
  }
};

struct LowerAddUIExtendedPattern final
    : OpRewritePattern<pto::AddUIExtendedOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::AddUIExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getSum().getType();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (isa<IntegerType>(getValueElementType(type))) {
      lhs = stripIntegerSignedness(rewriter, loc, lhs);
      rhs = stripIntegerSignedness(rewriter, loc, rhs);
    }
    auto lowered = rewriter.create<arith::AddUIExtendedOp>(loc, lhs, rhs);
    Value sum = castIntegerSignedness(rewriter, loc, lowered.getSum(), type);
    rewriter.replaceOp(op, ValueRange{sum, lowered.getOverflow()});
    return success();
  }
};

template <typename PTOOp, typename ArithOp>
struct LowerBitwisePattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getResult().getType();
    if (isa<IndexType>(type)) {
      rewriter.template replaceOpWithNewOp<ArithOp>(op, op.getLhs(),
                                                    op.getRhs());
      return success();
    }
    if (!isa<IntegerType>(getValueElementType(type))) {
      return rewriter.notifyMatchFailure(
          op, "expected integer, index, or builtin vector of integers");
    }
    Value lhs = stripIntegerSignedness(rewriter, loc, op.getLhs());
    Value rhs = stripIntegerSignedness(rewriter, loc, op.getRhs());
    auto lowered = rewriter.template create<ArithOp>(loc, lhs, rhs);
    copyAttrIfPresent(op, lowered, "overflowFlags");
    Value result = lowered;
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, result, type));
    return success();
  }
};

using LowerAndPattern = LowerBitwisePattern<pto::AndOp, arith::AndIOp>;
using LowerOrPattern = LowerBitwisePattern<pto::OrOp, arith::OrIOp>;
using LowerXorPattern = LowerBitwisePattern<pto::XorOp, arith::XOrIOp>;
using LowerShlPattern = LowerBitwisePattern<pto::ShlOp, arith::ShLIOp>;
using LowerShrPattern =
    LowerSignednessIntegerPattern<pto::ShrOp, arith::ShRSIOp, arith::ShRUIOp>;

static arith::CmpFPredicate
getFloatPredicate(pto::ScalarCmpPredicate predicate) {
  switch (predicate) {
  case pto::ScalarCmpPredicate::Eq:
    return arith::CmpFPredicate::OEQ;
  case pto::ScalarCmpPredicate::Ne:
    return arith::CmpFPredicate::ONE;
  case pto::ScalarCmpPredicate::Lt:
    return arith::CmpFPredicate::OLT;
  case pto::ScalarCmpPredicate::Le:
    return arith::CmpFPredicate::OLE;
  case pto::ScalarCmpPredicate::Gt:
    return arith::CmpFPredicate::OGT;
  case pto::ScalarCmpPredicate::Ge:
    return arith::CmpFPredicate::OGE;
  case pto::ScalarCmpPredicate::AlwaysFalse:
    return arith::CmpFPredicate::AlwaysFalse;
  case pto::ScalarCmpPredicate::OEq:
    return arith::CmpFPredicate::OEQ;
  case pto::ScalarCmpPredicate::OGt:
    return arith::CmpFPredicate::OGT;
  case pto::ScalarCmpPredicate::OGe:
    return arith::CmpFPredicate::OGE;
  case pto::ScalarCmpPredicate::OLt:
    return arith::CmpFPredicate::OLT;
  case pto::ScalarCmpPredicate::OLe:
    return arith::CmpFPredicate::OLE;
  case pto::ScalarCmpPredicate::ONe:
    return arith::CmpFPredicate::ONE;
  case pto::ScalarCmpPredicate::Ord:
    return arith::CmpFPredicate::ORD;
  case pto::ScalarCmpPredicate::UEq:
    return arith::CmpFPredicate::UEQ;
  case pto::ScalarCmpPredicate::UGt:
    return arith::CmpFPredicate::UGT;
  case pto::ScalarCmpPredicate::UGe:
    return arith::CmpFPredicate::UGE;
  case pto::ScalarCmpPredicate::ULt:
    return arith::CmpFPredicate::ULT;
  case pto::ScalarCmpPredicate::ULe:
    return arith::CmpFPredicate::ULE;
  case pto::ScalarCmpPredicate::UNe:
    return arith::CmpFPredicate::UNE;
  case pto::ScalarCmpPredicate::Uno:
    return arith::CmpFPredicate::UNO;
  case pto::ScalarCmpPredicate::AlwaysTrue:
    return arith::CmpFPredicate::AlwaysTrue;
  }
  llvm_unreachable("unknown PTO scalar comparison predicate");
}

static arith::CmpIPredicate
getIntegerPredicate(pto::ScalarCmpPredicate predicate,
                    pto::Signedness signedness) {
  switch (predicate) {
  case pto::ScalarCmpPredicate::Eq:
    return arith::CmpIPredicate::eq;
  case pto::ScalarCmpPredicate::Ne:
    return arith::CmpIPredicate::ne;
  case pto::ScalarCmpPredicate::Lt:
    return signedness == pto::Signedness::Unsigned ? arith::CmpIPredicate::ult
                                                   : arith::CmpIPredicate::slt;
  case pto::ScalarCmpPredicate::Le:
    return signedness == pto::Signedness::Unsigned ? arith::CmpIPredicate::ule
                                                   : arith::CmpIPredicate::sle;
  case pto::ScalarCmpPredicate::Gt:
    return signedness == pto::Signedness::Unsigned ? arith::CmpIPredicate::ugt
                                                   : arith::CmpIPredicate::sgt;
  case pto::ScalarCmpPredicate::Ge:
    return signedness == pto::Signedness::Unsigned ? arith::CmpIPredicate::uge
                                                   : arith::CmpIPredicate::sge;
  default:
    llvm_unreachable("unsupported predicate used for integer comparison");
  }
}

struct LowerCmpIPattern final : OpRewritePattern<pto::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getLhs().getType();
    pto::ScalarCmpPredicate predicate = op.getPredicateAttr().getValue();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (isa<IntegerType>(getValueElementType(type))) {
      lhs = stripIntegerSignedness(rewriter, loc, lhs);
      rhs = stripIntegerSignedness(rewriter, loc, rhs);
    }
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, getIntegerPredicate(predicate, op.getSignednessAttr().getValue()),
        lhs, rhs);
    return success();
  }
};

struct LowerCmpFPattern final : OpRewritePattern<pto::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::CmpFOp op,
                                PatternRewriter &rewriter) const override {
    auto lowered = rewriter.create<arith::CmpFOp>(
        op.getLoc(), getFloatPredicate(op.getPredicateAttr().getValue()),
        op.getLhs(), op.getRhs());
    copyAttrIfPresent(op, lowered, "fastmath");
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

template <typename PTOOp, typename SignedIntOp, typename UnsignedIntOp>
struct LowerIntegerExtremumPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    bool isUnsigned =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned;
    if (isUnsigned) {
      rewriter.replaceOpWithNewOp<UnsignedIntOp>(op, op.getLhs(), op.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<SignedIntOp>(op, op.getLhs(), op.getRhs());
    }
    return success();
  }
};

using LowerMaxIPattern =
    LowerIntegerExtremumPattern<pto::MaxIOp, arith::MaxSIOp, arith::MaxUIOp>;
using LowerMinIPattern =
    LowerIntegerExtremumPattern<pto::MinIOp, arith::MinSIOp, arith::MinUIOp>;

template <typename PTOOp, typename LLVMOp>
struct LowerFloatExtremumPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    auto lowered =
        rewriter.template create<LLVMOp>(op.getLoc(), op.getLhs(), op.getRhs());
    copyFastMathToLLVM(op, lowered);
    rewriter.replaceOp(op, lowered);
    return success();
  }
};
using LowerMaximumPattern =
    LowerFloatExtremumPattern<pto::MaximumOp, LLVM::MaximumOp>;
using LowerMinimumPattern =
    LowerFloatExtremumPattern<pto::MinimumOp, LLVM::MinimumOp>;
using LowerMaxFPattern = LowerFloatExtremumPattern<pto::MaxFOp, LLVM::MaxNumOp>;
using LowerMinFPattern = LowerFloatExtremumPattern<pto::MinFOp, LLVM::MinNumOp>;

struct LowerAbsIPattern final : OpRewritePattern<pto::AbsIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::AbsIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getResult().getType();
    bool isUnsigned =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned;
    if (isUnsigned) {
      rewriter.replaceOp(op, op.getValue());
      return success();
    }
    Value value = stripIntegerSignedness(rewriter, loc, op.getValue());
    Value result = rewriter.create<math::AbsIOp>(loc, value);
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, result, type));
    return success();
  }
};

struct LowerAbsFPattern final : OpRewritePattern<pto::AbsFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::AbsFOp op,
                                PatternRewriter &rewriter) const override {
    if (isPackedTwoLaneFloat(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "keep packed PTO operation");
    }
    rewriter.replaceOpWithNewOp<math::AbsFOp>(op, op.getValue());
    return success();
  }
};

struct LowerExtIPattern final : OpRewritePattern<pto::ExtIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::ExtIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = stripIntegerSignedness(rewriter, loc, op.getSrc());
    Type dstType = getSignlessIntegerType(rewriter, op.getDst().getType());
    Value result =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned
            ? rewriter.create<arith::ExtUIOp>(loc, dstType, src).getResult()
            : rewriter.create<arith::ExtSIOp>(loc, dstType, src).getResult();
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, result,
                                                 op.getDst().getType()));
    return success();
  }
};

struct LowerTruncIPattern final : OpRewritePattern<pto::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::TruncIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = stripIntegerSignedness(rewriter, loc, op.getSrc());
    Type dstType = getSignlessIntegerType(rewriter, op.getDst().getType());
    auto lowered = rewriter.create<arith::TruncIOp>(loc, dstType, src);
    copyAttrIfPresent(op, lowered, "overflowFlags");
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, lowered,
                                                 op.getDst().getType()));
    return success();
  }
};

struct LowerIToFPattern final : OpRewritePattern<pto::IToFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::IToFOp op,
                                PatternRewriter &rewriter) const override {
    if (requiresSimtHardwareConversion(
            op, op.getSrc().getType(), op.getDst().getType(),
            op.getRoundingmodeAttr(), op.getSaturation())) {
      return rewriter.notifyMatchFailure(
          op, "keep SIMT conversion for backend lowering");
    }
    Value src = stripIntegerSignedness(rewriter, op.getLoc(), op.getSrc());
    bool isUnsigned =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned;
    if (isUnsigned) {
      rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, op.getDst().getType(),
                                                   src);
    } else {
      rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, op.getDst().getType(),
                                                   src);
    }
    return success();
  }
};

struct LowerFToIPattern final : OpRewritePattern<pto::FToIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::FToIOp op,
                                PatternRewriter &rewriter) const override {
    if (requiresSimtHardwareConversion(
            op, op.getSrc().getType(), op.getDst().getType(),
            op.getRoundingmodeAttr(), op.getSaturation())) {
      return rewriter.notifyMatchFailure(
          op, "keep SIMT conversion for backend lowering");
    }
    Location loc = op.getLoc();
    Type dstType = getSignlessIntegerType(rewriter, op.getDst().getType());
    Value result =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned
            ? rewriter.create<arith::FPToUIOp>(loc, dstType, op.getSrc())
                  .getResult()
            : rewriter.create<arith::FPToSIOp>(loc, dstType, op.getSrc())
                  .getResult();
    rewriter.replaceOp(op, castIntegerSignedness(rewriter, loc, result,
                                                 op.getDst().getType()));
    return success();
  }
};

struct LowerFToFPattern final : OpRewritePattern<pto::FToFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::FToFOp op,
                                PatternRewriter &rewriter) const override {
    if (requiresSimtHardwareConversion(
            op, op.getSrc().getType(), op.getDst().getType(),
            op.getRoundingmodeAttr(), op.getSaturation())) {
      return rewriter.notifyMatchFailure(
          op, "keep SIMT conversion for backend lowering");
    }
    bool hasNonFloatElement =
        !isa<FloatType>(getValueElementType(op.getSrc().getType())) ||
        !isa<FloatType>(getValueElementType(op.getDst().getType()));
    if (hasNonFloatElement) {
      return rewriter.notifyMatchFailure(
          op, "keep packed conversion for backend lowering");
    }
    Location loc = op.getLoc();
    Type srcType = getValueElementType(op.getSrc().getType());
    Type dstType = getValueElementType(op.getDst().getType());
    unsigned srcWidth = cast<FloatType>(srcType).getWidth();
    unsigned dstWidth = cast<FloatType>(dstType).getWidth();
    if (srcWidth < dstWidth) {
      auto lowered = rewriter.create<arith::ExtFOp>(loc, op.getDst().getType(),
                                                    op.getSrc());
      copyAttrIfPresent(op, lowered, "fastmath");
      rewriter.replaceOp(op, lowered);
      return success();
    }
    if (srcWidth > dstWidth) {
      auto lowered = rewriter.create<arith::TruncFOp>(
          loc, op.getDst().getType(), op.getSrc());
      copyAttrIfPresent(op, lowered, "roundingmode");
      copyAttrIfPresent(op, lowered, "fastmath");
      rewriter.replaceOp(op, lowered);
      return success();
    }

    Type wideType = rewriter.getF32Type();
    if (auto vectorType = dyn_cast<VectorType>(op.getSrc().getType())) {
      wideType = VectorType::Builder(vectorType).setElementType(wideType);
    }
    auto extended = rewriter.create<arith::ExtFOp>(loc, wideType, op.getSrc());
    copyAttrIfPresent(op, extended, "fastmath");
    auto truncated =
        rewriter.create<arith::TruncFOp>(loc, op.getDst().getType(), extended);
    copyAttrIfPresent(op, truncated, "fastmath");
    rewriter.replaceOp(op, truncated);
    return success();
  }
};

struct LowerIndexCastPattern final : OpRewritePattern<pto::IndexCastOp> {
  using OpRewritePattern<pto::IndexCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type dstType = op.getDst().getType();
    Value src = op.getSrc();
    Type srcElement = getValueElementType(src.getType());
    Type dstElement = getValueElementType(dstType);
    Type loweredDst = dstType;
    if (isa<IntegerType>(dstElement)) {
      loweredDst = getSignlessIntegerType(rewriter, dstType);
    }
    if (isa<IntegerType>(srcElement)) {
      src = stripIntegerSignedness(rewriter, loc, src);
    }
    Value result;
    bool isUnsigned =
        op.getSignednessAttr().getValue() == pto::Signedness::Unsigned;
    if (isUnsigned) {
      result = rewriter.create<arith::IndexCastUIOp>(loc, loweredDst, src);
    } else {
      result = rewriter.create<arith::IndexCastOp>(loc, loweredDst, src);
    }
    rewriter.replaceOp(op,
                       castIntegerSignedness(rewriter, loc, result, dstType));
    return success();
  }
};

struct LowerValueBitcastPattern final : OpRewritePattern<pto::BitcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    Type srcType = op.getSrc().getType();
    Type dstType = op.getResult().getType();
    if (isa<pto::TileBufType>(srcType)) {
      return rewriter.notifyMatchFailure(op, "expected numeric bitcast");
    }

    Location loc = op.getLoc();
    Value src = stripIntegerSignedness(rewriter, loc, op.getSrc());
    Type loweredDstType = dstType;
    if (isa<IntegerType>(getValueElementType(dstType))) {
      loweredDstType = getSignlessIntegerType(rewriter, dstType);
    }
    Value result = src;
    bool requiresBitcast = src.getType() != loweredDstType;
    if (requiresBitcast) {
      result = rewriter.create<arith::BitcastOp>(loc, loweredDstType, src);
    }
    rewriter.replaceOp(op,
                       castIntegerSignedness(rewriter, loc, result, dstType));
    return success();
  }
};

struct LowerSelectPattern final : OpRewritePattern<pto::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::SelectOp op,
    PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    bool broadcastsScalarCondition =
        op.getCondition().getType().isInteger(1) &&
        isa<VectorType>(resultType);
    if (broadcastsScalarCondition) {
      Location loc = op.getLoc();
      Value trueValue =
          stripIntegerSignedness(rewriter, loc, op.getTrueValue());
      Value falseValue =
          stripIntegerSignedness(rewriter, loc, op.getFalseValue());
      Value result = rewriter.create<arith::SelectOp>(
          loc, trueValue.getType(), op.getCondition(), trueValue, falseValue);
      rewriter.replaceOp(
          op, castIntegerSignedness(rewriter, loc, result, resultType));
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, op.getCondition(), op.getTrueValue(), op.getFalseValue());
    return success();
  }
};

template <typename PTOOp, typename MathOp>
struct LowerScalarMathPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    if (isPackedTwoLaneF16(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "keep packed PTO operation");
    }
    rewriter.template replaceOpWithNewOp<MathOp>(op, op.getValue());
    return success();
  }
};

template <typename PTOOp, typename MathOp>
struct LowerBinaryScalarMathPattern final : OpRewritePattern<PTOOp> {
  using OpRewritePattern<PTOOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PTOOp op,
                                PatternRewriter &rewriter) const override {
    if (isPackedTwoLaneF16(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "keep packed PTO operation");
    }
    rewriter.template replaceOpWithNewOp<MathOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

struct LowerFmaToMathPattern final : OpRewritePattern<pto::FmaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(pto::FmaOp op,
                                PatternRewriter &rewriter) const override {
    if (isPackedTwoLaneFloat(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "keep packed PTO operation");
    }
    rewriter.replaceOpWithNewOp<math::FmaOp>(op, op.getLhs(), op.getRhs(),
                                             op.getAcc());
    return success();
  }
};

struct PTOLowerGenericOpsPass final
    : pto::impl::PTOLowerGenericOpsBase<PTOLowerGenericOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerGenericOpsPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<
        LowerConstantPattern, LowerAddIPattern, LowerAddFPattern,
        LowerSubIPattern, LowerSubFPattern, LowerMulIPattern, LowerMulFPattern,
        LowerAddUIExtendedPattern, LowerMulExtendedPattern, LowerNegIPattern,
        LowerNegFPattern, LowerDivFPattern, LowerDivPattern,
        LowerFloorDivPattern, LowerCeilDivPattern, LowerRemPattern,
        LowerRemFPattern, LowerAndPattern, LowerOrPattern, LowerXorPattern,
        LowerShlPattern, LowerShrPattern, LowerCmpIPattern, LowerCmpFPattern,
        LowerMaxIPattern, LowerMaxFPattern, LowerMinIPattern, LowerMinFPattern,
        LowerMaximumPattern, LowerMinimumPattern, LowerAbsIPattern,
        LowerAbsFPattern, LowerExtIPattern, LowerTruncIPattern,
        LowerIToFPattern, LowerFToIPattern, LowerFToFPattern,
        LowerIndexCastPattern, LowerSelectPattern, LowerValueBitcastPattern,
        LowerScalarMathPattern<pto::ExpOp, math::ExpOp>,
        LowerScalarMathPattern<pto::LogOp, math::LogOp>,
        LowerScalarMathPattern<pto::SqrtOp, math::SqrtOp>,
        LowerBinaryScalarMathPattern<pto::PowOp, math::PowFOp>,
        LowerFmaToMathPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasResidualGenericOp = false;
    getOperation()->walk([&](Operation *op) {
      const bool isUnexpectedResidual =
          isPublicGenericPTOOp(op) && !isExpectedGenericResidual(op);
      if (!isUnexpectedResidual) {
        return;
      }
      op->emitOpError("generic PTO operation remained after lowering");
      hasResidualGenericOp = true;
    });
    if (hasResidualGenericOp) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerGenericOpsPass() {
  return std::make_unique<PTOLowerGenericOpsPass>();
}
