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

static FailureOr<StringRef> buildMadTypedCalleeName(MLIRContext *context,
                                                     Type lhsElem, Type rhsElem,
                                                     Type dstElem) {
  struct MadContract {
    StringRef lhs;
    StringRef rhs;
    StringRef dst;
    StringRef callee;
  };

  static const MadContract contracts[] = {
      {"f16", "f16", "f32", "llvm.hivm.MAD.f162f32.c310"},
      {"f16", "f16", "f16", "llvm.hivm.MAD.f162f16"},
      {"f16", "f16", "s32", "llvm.hivm.MAD.f162s32.1952"},
      {"bf16", "bf16", "f32", "llvm.hivm.MAD.bf162f32.c310"},
      {"f32", "f32", "f32", "llvm.hivm.MAD.f322f32.c310"},
      {"s8", "s8", "s32", "llvm.hivm.MAD.s8.c310"},
      {"e4m3", "e4m3", "f32", "llvm.hivm.MAD.e4m3e4m3.c310"},
      {"e4m3", "e5m2", "f32", "llvm.hivm.MAD.e4m3e5m2.c310"},
      {"e5m2", "e4m3", "f32", "llvm.hivm.MAD.e5m2e4m3.c310"},
      {"e5m2", "e5m2", "f32", "llvm.hivm.MAD.e5m2e5m2.c310"},
      {"hif8", "hif8", "f32", "llvm.hivm.MAD.e4m3e4m3.c310"},
      {"f16", "s4", "", "llvm.hivm.MAD.f16s4.c310"},
      {"f16", "s8", "", "llvm.hivm.MAD.f16s8.c310"},
      {"f16", "u2", "", "llvm.hivm.MAD.f16u2"},
      {"f16", "e8m0", "", "llvm.hivm.MAD.f16e8m0.c310"},
  };

  std::string lhs = getMadLhsFragment(lhsElem);
  std::string rhs = getMadRhsFragment(rhsElem);
  std::string dst = getMadDstFragment(dstElem);
  for (const MadContract &contract : contracts) {
    if (contract.lhs == lhs && contract.rhs == rhs && contract.dst == dst) {
      return StringAttr::get(context, contract.callee).getValue();
    }
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

struct MadConvertedOperands {
  Value lhs;
  Value rhs;
  Value dst;
  Value bias;
  Value xt;
};

static FailureOr<MadConvertedOperands>
extractMadOperands(pto::MadRawOpInterface op, ValueRange convertedOperands) {
  const bool hasBias = op.hasBiasOperand();
  if (convertedOperands.size() < (hasBias ? 5U : 4U))
  {
    return failure();
  }

  MadConvertedOperands operands{convertedOperands[0], convertedOperands[1],
                                convertedOperands[2],
                                hasBias ? convertedOperands[3] : Value(),
                                convertedOperands[hasBias ? 4 : 3]};
  if (!operands.lhs || !operands.rhs || !operands.dst || !operands.xt ||
      (hasBias && !operands.bias))
  {
    return failure();
  }

  if (!isa<LLVM::LLVMPointerType>(operands.lhs.getType()) ||
      !isa<LLVM::LLVMPointerType>(operands.rhs.getType()) ||
      !isa<LLVM::LLVMPointerType>(operands.dst.getType()) ||
      (operands.bias && !isa<LLVM::LLVMPointerType>(operands.bias.getType())))
  {
    return failure();
  }

  return operands;
}

static FailureOr<MadConvertedOperands>
reinterpretMadOperands(pto::MadRawOpInterface op,
                       const MadConvertedOperands &operands) {
  constexpr unsigned caAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::LEFT);
  constexpr unsigned cbAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::RIGHT);
  constexpr unsigned ccAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::ACC);
  constexpr unsigned btAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::BIAS);
  FailureOr<Value> lhs = reinterpretPointerToAddrSpace(
      op, operands.lhs, caAddressSpace);
  FailureOr<Value> rhs = reinterpretPointerToAddrSpace(
      op, operands.rhs, cbAddressSpace);
  FailureOr<Value> dst = reinterpretPointerToAddrSpace(
      op, operands.dst, ccAddressSpace);
  if (failed(lhs) || failed(rhs) || failed(dst))
  {
    return failure();
  }
  Value bias;
  if (operands.bias)
  {
    FailureOr<Value> convertedBias = reinterpretPointerToAddrSpace(
        op, operands.bias, btAddressSpace);
    if (failed(convertedBias))
    {
      return failure();
    }
    bias = *convertedBias;
  }
  return MadConvertedOperands{*lhs, *rhs, *dst, bias, operands.xt};
}

static FailureOr<MadConvertedOperands>
prepareMadConvertedOperands(pto::MadRawOpInterface op,
                            ValueRange convertedOperands) {
  FailureOr<MadConvertedOperands> operands =
      extractMadOperands(op, convertedOperands);
  if (failed(operands))
  {
    return failure();
  }
  return reinterpretMadOperands(op, *operands);
}

static LogicalResult lowerMadRawOp(pto::MadRawOpInterface op,
                                   ValueRange convertedOperands,
                                   ConversionPatternRewriter &rewriter,
                                   LoweringState &state) {
  FailureOr<MadConvertedOperands> operands =
      prepareMadConvertedOperands(op, convertedOperands);
  if (failed(operands)) {
    return rewriter.notifyMatchFailure(op,
                                       "invalid converted mad raw operands");
  }

  Type i64Ty = rewriter.getI64Type();

  FailureOr<StringRef> calleeName =
      op.isMadMxFamily() ? buildMxMadCallee(op.getContext(), op)
                         : buildOrdinaryMadCallee(op.getContext(), op);
  if (failed(calleeName)) {
    return rewriter.notifyMatchFailure(
        op, "unsupported mad element types for raw dispatch");
  }

  Value callDst = operands->dst;
  if (operands->bias)
  {
    callDst = buildMadBiasDestination(op, rewriter, operands->dst,
                                      operands->bias);
  }
  auto funcType = rewriter.getFunctionType(
      TypeRange{operands->dst.getType(), operands->lhs.getType(),
                operands->rhs.getType(), i64Ty},
      TypeRange{});
  auto call = rewriter.create<func::CallOp>(
      op->getLoc(), *calleeName, TypeRange{},
      ValueRange{callDst, operands->lhs, operands->rhs, operands->xt});
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
