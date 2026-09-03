// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

static bool isMxElementType(Type ty) {
  if (auto floatType = dyn_cast<FloatType>(ty))
  {
    return floatType.getWidth() == 8;
  }
  if (isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(ty))
  {
    return true;
  }
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  ty.print(os);
  os.flush();
  return StringRef(typeText).starts_with("f8");
}

static std::string getMadMxElementFragment(Type type) {
  if (type.isF16())
  {
    return "f16";
  }
  if (type.isBF16())
  {
    return "bf16";
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();

  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3"))
  {
    return "e4m3";
  }
  if (StringRef(lower).contains("e5m2"))
  {
    return "e5m2";
  }
  if (StringRef(lower).contains("hif4"))
  {
    return "hif4";
  }
  if (StringRef(lower).contains("e2m1x2"))
  {
    return "e2m1x2";
  }
  if (StringRef(lower).contains("e1m2x2"))
  {
    return "e1m2x2";
  }
  return {};
}

static FailureOr<StringRef> buildMadMxCalleeName(MLIRContext *context,
                                                 Type lhsElem, Type rhsElem) {
  std::string lhs = getMadMxElementFragment(lhsElem);
  std::string rhs = getMadMxElementFragment(rhsElem);
  if (lhs.empty() || rhs.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MMAD.MX." + lhs + rhs).getValue();
}

static bool isSignedOrSignlessInteger(IntegerType intType, unsigned width) {
  return intType && intType.getWidth() == width &&
         (intType.isSigned() || intType.isSignless());
}

static std::string getMadRhsFragment(Type type) {
  if (type.isF16())
  {
    return "f16";
  }
  if (type.isBF16())
  {
    return "bf16";
  }
  if (type.isF32())
  {
    return "f32";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (isSignedOrSignlessInteger(intType, 4))
    {
      return "s4";
    }
    if (isSignedOrSignlessInteger(intType, 8))
    {
      return "s8";
    }
    if (intType.isUnsigned() && intType.getWidth() == 2)
    {
      return "u2";
    }
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e8m0"))
  {
    return "e8m0";
  }
  return {};
}

static bool isMadE4M3ElementType(Type type) {
  return pto::isPTOFloat8E4M3LikeType(type);
}

static bool isMadE5M2ElementType(Type type) {
  return pto::isPTOFloat8E5M2LikeType(type);
}

static std::string getMadDstFragment(Type type) {
  if (type.isF16())
  {
    return "f16";
  }
  if (type.isF32())
  {
    return "f32";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (isSignedOrSignlessInteger(intType, 32))
    {
      return "s32";
    }
  }
  return {};
}

static FailureOr<StringRef> buildMadTypedCalleeName(MLIRContext *context,
                                                     Type lhsElem, Type rhsElem,
                                                     Type dstElem) {
  std::string rhs = getMadRhsFragment(rhsElem);
  std::string dst = getMadDstFragment(dstElem);
  if (lhsElem.isF16() && rhs == "f16" && dst == "f32")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f162f32.c310").getValue();
  }
  if (lhsElem.isF16() && rhs == "f16" && dst == "f16")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f162f16").getValue();
  }
  if (lhsElem.isF16() && rhs == "f16" && dst == "s32")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f162s32.1952").getValue();
  }
  if (lhsElem.isBF16() && rhs == "bf16" && dst == "f32")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.bf162f32.c310").getValue();
  }
  if (lhsElem.isF32() && rhs == "f32" && dst == "f32")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f322f32.c310").getValue();
  }
  if (isSignedOrSignlessInteger(dyn_cast<IntegerType>(lhsElem), 8) &&
      rhs == "s8" && dst == "s32") {
    return StringAttr::get(context, "llvm.hivm.MAD.s8.c310").getValue();
  }
  if (isMadE4M3ElementType(lhsElem) && isMadE4M3ElementType(rhsElem) &&
      dst == "f32") {
    return StringAttr::get(context, "llvm.hivm.MAD.e4m3e4m3.c310").getValue();
  }
  if (isMadE4M3ElementType(lhsElem) && isMadE5M2ElementType(rhsElem) &&
      dst == "f32") {
    return StringAttr::get(context, "llvm.hivm.MAD.e4m3e5m2.c310").getValue();
  }
  if (isMadE5M2ElementType(lhsElem) && isMadE4M3ElementType(rhsElem) &&
      dst == "f32") {
    return StringAttr::get(context, "llvm.hivm.MAD.e5m2e4m3.c310").getValue();
  }
  if (isMadE5M2ElementType(lhsElem) && isMadE5M2ElementType(rhsElem) &&
      dst == "f32") {
    return StringAttr::get(context, "llvm.hivm.MAD.e5m2e5m2.c310").getValue();
  }
  if (pto::isPTOHiFloat8Type(lhsElem) && pto::isPTOHiFloat8Type(rhsElem) &&
      dst == "f32") {
    return StringAttr::get(context, "llvm.hivm.MAD.e4m3e4m3.c310").getValue();
  }
  if (lhsElem.isF16() && rhs == "s4")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f16s4.c310").getValue();
  }
  if (lhsElem.isF16() && rhs == "s8")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f16s8.c310").getValue();
  }
  if (lhsElem.isF16() && rhs == "u2")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f16u2").getValue();
  }
  if (lhsElem.isF16() && rhs == "e8m0")
  {
    return StringAttr::get(context, "llvm.hivm.MAD.f16e8m0.c310").getValue();
  }
  return failure();
}
static Value buildMadBiasDestination(Operation *anchor,
                                     ConversionPatternRewriter &rewriter,
                                     Value dst, Value bias) {
  Type i64Ty = rewriter.getI64Type();
  Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, dst);
  Value biasAddr =
      rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, bias);
  Value lowMask = getI64Constant(rewriter, anchor->getLoc(), 0xffffffffULL);
  Value dstLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), dstAddr, lowMask);
  Value biasLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), biasAddr, lowMask);
  Value biasHigh = rewriter.create<arith::ShLIOp>(
      anchor->getLoc(), biasLow, getI64Constant(rewriter, anchor->getLoc(), 32));
  Value packed = rewriter.create<arith::OrIOp>(anchor->getLoc(), dstLow, biasHigh);
  return rewriter.create<LLVM::IntToPtrOp>(anchor->getLoc(), dst.getType(), packed);
}

static FailureOr<StringRef> buildOrdinaryMadCallee(MLIRContext *context,
                                                   pto::MadRawOpInterface op) {
  auto lhsType = dyn_cast<pto::PtrType>(op.getLhs().getType());
  auto rhsType = dyn_cast<pto::PtrType>(op.getRhs().getType());
  auto dstType = dyn_cast<pto::PtrType>(op.getDst().getType());
  if (!lhsType || !rhsType || !dstType)
  {
    return failure();
  }

  return buildMadTypedCalleeName(context, lhsType.getElementType(),
                                  rhsType.getElementType(),
                                  dstType.getElementType());
}

static FailureOr<StringRef> buildMxMadCallee(MLIRContext *context,
                                             pto::MadRawOpInterface op) {
  auto lhsType = dyn_cast<pto::PtrType>(op.getLhs().getType());
  auto rhsType = dyn_cast<pto::PtrType>(op.getRhs().getType());
  if (!lhsType || !rhsType)
  {
    return failure();
  }
  if (isMxElementType(lhsType.getElementType()) &&
      isMxElementType(rhsType.getElementType())) {
    return buildMadMxCalleeName(context, lhsType.getElementType(),
                                rhsType.getElementType());
  }
  return failure();
}

static LogicalResult lowerMadRawOp(pto::MadRawOpInterface op,
                                   ValueRange convertedOperands,
                                   ConversionPatternRewriter &rewriter,
                                   LoweringState &state) {
  Value lhsRaw = convertedOperands[0];
  Value rhsRaw = convertedOperands[1];
  Value dstRaw = convertedOperands[2];
  Value biasRaw = op.hasBiasOperand() ? convertedOperands[3] : Value();
  Value xt = convertedOperands[op.hasBiasOperand() ? 4 : 3];
  if (!lhsRaw || !rhsRaw || !dstRaw || !xt ||
      (op.hasBiasOperand() && !biasRaw)) {
    return rewriter.notifyMatchFailure(op, "expected converted mad raw operands");
  }

  if (!isa<LLVM::LLVMPointerType>(lhsRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(rhsRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(dstRaw.getType()) ||
      (biasRaw && !isa<LLVM::LLVMPointerType>(biasRaw.getType()))) {
    return rewriter.notifyMatchFailure(
        op, "expected LLVM pointer lhs/rhs/dst/bias operands");
  }

  Type i64Ty = rewriter.getI64Type();
  constexpr unsigned caAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::LEFT);
  constexpr unsigned cbAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::RIGHT);
  constexpr unsigned ccAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::ACC);
  constexpr unsigned btAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::BIAS);
  FailureOr<Value> lhs =
      reinterpretPointerToAddrSpace(op, lhsRaw, caAddressSpace);
  FailureOr<Value> rhs =
      reinterpretPointerToAddrSpace(op, rhsRaw, cbAddressSpace);
  FailureOr<Value> dst =
      reinterpretPointerToAddrSpace(op, dstRaw, ccAddressSpace);
  FailureOr<Value> bias;
  if (biasRaw)
  {
    bias = reinterpretPointerToAddrSpace(op, biasRaw, btAddressSpace);
  }
  if (failed(lhs) || failed(rhs) || failed(dst) ||
      (biasRaw && failed(bias))) {
    return rewriter.notifyMatchFailure(op, "failed to map cube pointer spaces");
  }

  FailureOr<StringRef> calleeName =
      op.isMadMxFamily() ? buildMxMadCallee(op.getContext(), op)
                         : buildOrdinaryMadCallee(op.getContext(), op);
  if (failed(calleeName)) {
    return rewriter.notifyMatchFailure(
        op, "unsupported mad element types for raw dispatch");
  }

  Value callDst = *dst;
  if (biasRaw)
  {
    callDst = buildMadBiasDestination(op, rewriter, *dst, *bias);
  }
  auto funcType = rewriter.getFunctionType(
      TypeRange{dst->getType(), lhs->getType(), rhs->getType(), i64Ty},
      TypeRange{});
  auto call = rewriter.create<func::CallOp>(
      op->getLoc(), *calleeName, TypeRange{},
      ValueRange{callDst, *lhs, *rhs, xt});
  state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
  rewriter.replaceOp(op, call.getResults());
  return success();
}

template <typename RawOp>
class LowerMadRawPattern final : public OpConversionPattern<RawOp> {
public:
  explicit LowerMadRawPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<RawOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(RawOp op, typename RawOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto raw = dyn_cast<pto::MadRawOpInterface>(op.getOperation());
    if (!raw)
    {
      return failure();
    }
    return lowerMadRawOp(raw, adaptor.getOperands(), rewriter, state);
  }

private:
  LoweringState &state;
};

} // namespace

void populateVPTOCubeMadPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 LoweringState &state) {
  patterns.add<LowerMadRawPattern<pto::MadRawOp>,
               LowerMadRawPattern<pto::MadBiasRawOp>,
               LowerMadRawPattern<pto::MadMxRawOp>,
               LowerMadRawPattern<pto::MadMxBiasRawOp>>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
