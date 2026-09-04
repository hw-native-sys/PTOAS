// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// The CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {

namespace {

enum class VcvtElemKind {
  Invalid,
  F16,
  BF16,
  F32,
  F8E4M3,
  F8E5M2,
  HiF8,
  F4E1M2x2,
  F4E2M1x2,
  S8,
  U8,
  S16,
  U16,
  S32,
  U32,
  S64,
};

struct VcvtContract {
  const char *intrinsic;
  bool requiresRnd;
  bool requiresSat;
  bool requiresPart;
  unsigned maskBitWidth;
  bool satBeforeRnd = false;
};


[[maybe_unused]] static Value getI1Constant(OpBuilder &builder, Location loc,
                                            bool value) {
  return builder
      .create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(builder.getI1Type(), value ? 1 : 0))
      .getResult();
}

static VcvtElemKind classifyVcvtElemType(Type type) {
  if (type.isF16())
  {
    return VcvtElemKind::F16;
  }
  if (type.isBF16())
  {
    return VcvtElemKind::BF16;
  }
  if (type.isF32())
  {
    return VcvtElemKind::F32;
  }
  if (pto::isPTOFloat8E4M3LikeType(type))
  {
    return VcvtElemKind::F8E4M3;
  }
  if (pto::isPTOFloat8E5M2LikeType(type))
  {
    return VcvtElemKind::F8E5M2;
  }
  if (pto::isPTOHiFloat8Type(type))
  {
    return VcvtElemKind::HiF8;
  }
  if (isa<pto::F4E1M2x2Type>(type))
  {
    return VcvtElemKind::F4E1M2x2;
  }
  if (isa<pto::F4E2M1x2Type>(type))
  {
    return VcvtElemKind::F4E2M1x2;
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 8:
      return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
    case 16:
      return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
    case 32:
      return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
    case 64:
      return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
    default:
      return VcvtElemKind::Invalid;
    }
  }
  return VcvtElemKind::Invalid;
}

static std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src,
                                                        VcvtElemKind dst) {
  static const llvm::DenseMap<VcvtElemKind, llvm::DenseMap<VcvtElemKind,
                                                               VcvtContract>>
      contracts = {
          {VcvtElemKind::F32,
           {{VcvtElemKind::F8E4M3, {"llvm.hivm.vcvtff.f322f8e4m3.x", true, true, true, 32}},
            {VcvtElemKind::F8E5M2, {"llvm.hivm.vcvtff.f322f8e5m2.x", true, true, true, 32}},
            {VcvtElemKind::HiF8, {"llvm.hivm.vcvtff.f322hif8.x", true, true, true, 32}},
            {VcvtElemKind::F16, {"llvm.hivm.vcvtff.f322f16.x", true, true, true, 32}},
            {VcvtElemKind::BF16, {"llvm.hivm.vcvtff.f322bf16.x", true, true, true, 32}},
            {VcvtElemKind::S16, {"llvm.hivm.vcvtfi.f322s16.x", true, true, true, 32}},
            {VcvtElemKind::S32, {"llvm.hivm.vcvtfi.f322s32.x", true, true, false, 32}},
            {VcvtElemKind::S64, {"llvm.hivm.vcvtfi.f322s64.x", true, true, true, 32}}}},
          {VcvtElemKind::F16,
           {{VcvtElemKind::F8E4M3, {"llvm.hivm.vcvtff.f162f8e4m3.x", true, true, true, 16}},
            {VcvtElemKind::F8E5M2, {"llvm.hivm.vcvtff.f162f8e5m2.x", true, true, true, 16}},
            {VcvtElemKind::HiF8, {"llvm.hivm.vcvtff.f162hif8.x", true, true, true, 16}},
            {VcvtElemKind::F32, {"llvm.hivm.vcvtff.f162f32.x", false, false, true, 16}},
            {VcvtElemKind::S32, {"llvm.hivm.vcvtfi.f162s32.x", true, false, true, 16}},
            {VcvtElemKind::S16, {"llvm.hivm.vcvtfi.f162s16.x", true, true, false, 16}},
            {VcvtElemKind::S8, {"llvm.hivm.vcvtfi.f162s8.x", true, true, true, 16}},
            {VcvtElemKind::U8, {"llvm.hivm.vcvtfi.f162u8.x", true, true, true, 16}},
            {VcvtElemKind::BF16, {"llvm.hivm.vcvtff.f162bf16.x", true, false, false, 16}}}},
          {VcvtElemKind::BF16,
           {{VcvtElemKind::F8E4M3, {"llvm.hivm.vcvtff.bf162f8e4m3.x", true, true, true, 16}},
            {VcvtElemKind::F8E5M2, {"llvm.hivm.vcvtff.bf162f8e5m2.x", true, true, true, 16}},
            {VcvtElemKind::F4E1M2x2, {"llvm.hivm.vcvtff2.bf162f4e1m2x2.x", true, false, true, 16}},
            {VcvtElemKind::F4E2M1x2, {"llvm.hivm.vcvtff2.bf162f4e2m1x2.x", true, false, true, 16}},
            {VcvtElemKind::F16, {"llvm.hivm.vcvtff.bf162f16.x", true, true, false, 16, true}},
            {VcvtElemKind::F32, {"llvm.hivm.vcvtff.bf162f32.x", false, false, true, 16}},
            {VcvtElemKind::S32, {"llvm.hivm.vcvtfi.bf162s32.x", true, true, true, 16}}}},
          {VcvtElemKind::U8,
           {{VcvtElemKind::F16, {"llvm.hivm.vcvtif.u82f16.x", false, false, true, 8}},
            {VcvtElemKind::U16, {"llvm.hivm.vcvtii.u82u16.x", false, false, true, 8}},
            {VcvtElemKind::U32, {"llvm.hivm.vcvtii.u82u32.x", false, false, true, 8}}}},
          {VcvtElemKind::S8,
           {{VcvtElemKind::F16, {"llvm.hivm.vcvtif.s82f16.x", false, false, true, 8}},
            {VcvtElemKind::S16, {"llvm.hivm.vcvtii.s82s16.x", false, false, true, 8}},
            {VcvtElemKind::S32, {"llvm.hivm.vcvtii.s82s32.x", false, false, true, 8}}}},
          {VcvtElemKind::U16,
           {{VcvtElemKind::U8, {"llvm.hivm.vcvtii.u162u8.x", false, true, true, 16}},
            {VcvtElemKind::U32, {"llvm.hivm.vcvtii.u162u32.x", false, false, true, 16}}}},
          {VcvtElemKind::S16,
           {{VcvtElemKind::F16, {"llvm.hivm.vcvtif.s162f16.x", true, false, false, 16}},
            {VcvtElemKind::F32, {"llvm.hivm.vcvtif.s162f32.x", false, false, true, 16}},
            {VcvtElemKind::U8, {"llvm.hivm.vcvtii.s162u8.x", false, true, true, 16}},
            {VcvtElemKind::U32, {"llvm.hivm.vcvtii.s162u32.x", false, false, true, 16}},
            {VcvtElemKind::S32, {"llvm.hivm.vcvtii.s162s32.x", false, false, true, 16}}}},
          {VcvtElemKind::U32,
           {{VcvtElemKind::U8, {"llvm.hivm.vcvtii.u322u8.x", false, true, true, 32}},
            {VcvtElemKind::U16, {"llvm.hivm.vcvtii.u322u16.x", false, true, true, 32}},
            {VcvtElemKind::S16, {"llvm.hivm.vcvtii.u322s16.x", false, true, true, 32}}}},
          {VcvtElemKind::S32,
           {{VcvtElemKind::F32, {"llvm.hivm.vcvtif.s322f32.x", true, false, false, 32}},
            {VcvtElemKind::U8, {"llvm.hivm.vcvtii.s322u8.x", false, true, true, 32}},
            {VcvtElemKind::U16, {"llvm.hivm.vcvtii.s322u16.x", false, true, true, 32}},
            {VcvtElemKind::S16, {"llvm.hivm.vcvtii.s322s16.x", false, true, true, 32}},
            {VcvtElemKind::S64, {"llvm.hivm.vcvtii.s322s64.x", false, false, true, 32}}}},
          {VcvtElemKind::S64,
           {{VcvtElemKind::F32, {"llvm.hivm.vcvtif.s642f32.x", true, false, true, 32}},
            {VcvtElemKind::S32, {"llvm.hivm.vcvtii.s642s32.x", false, true, true, 32}}}},
          {VcvtElemKind::F8E4M3,
           {{VcvtElemKind::F32, {"llvm.hivm.vcvtff.f8e4m32f32.x", false, false, true, 8}}}},
          {VcvtElemKind::F8E5M2,
           {{VcvtElemKind::F32, {"llvm.hivm.vcvtff.f8e5m22f32.x", false, false, true, 8}}}},
          {VcvtElemKind::HiF8,
           {{VcvtElemKind::F32, {"llvm.hivm.vcvtff.hif82f32.x", false, false, true, 8}}}},
          {VcvtElemKind::F4E1M2x2,
           {{VcvtElemKind::BF16, {"llvm.hivm.vcvtff2.f4e1m2x22bf16.x", false, false, true, 8}}}},
          {VcvtElemKind::F4E2M1x2,
           {{VcvtElemKind::BF16, {"llvm.hivm.vcvtff2.f4e2m1x22bf16.x", false, false, true, 8}}}}};

  auto source = contracts.find(src);
  if (source == contracts.end()) {
    return std::nullopt;
  }
  auto destination = source->second.find(dst);
  if (destination == source->second.end()) {
    return std::nullopt;
  }
  return destination->second;
}

static FailureOr<VcvtContract> buildVcvtContract(pto::VcvtOp op) {
  Type inputElemType = getElementTypeFromVectorLike(op.getInput().getType());
  Type resultElemType = getElementTypeFromVectorLike(op.getResult().getType());
  if (!inputElemType || !resultElemType)
  {
    return failure();
  }
  auto contract = lookupVcvtContract(classifyVcvtElemType(inputElemType),
                                     classifyVcvtElemType(resultElemType));
  if (!contract)
  {
    return failure();
  }
  return *contract;
}

static LogicalResult appendVcvtImmediate(pto::VcvtOp op,
                                         ConversionPatternRewriter &rewriter,
                                         StringRef attrName, bool required,
                                         std::optional<uint64_t> value,
                                         SmallVector<Value> &args,
                                         SmallVector<Type> &types) {
  if (!required) {
    return success();
  }
  if (!value) {
    return rewriter.notifyMatchFailure(op,
                                       "vcvt requires valid " + attrName + " attr");
  }
  Value immediate = getI32Constant(rewriter, op.getLoc(), *value);
  args.push_back(immediate);
  types.push_back(immediate.getType());
  return success();
}

class LowerVcvtOpPattern final : public OpConversionPattern<pto::VcvtOp> {
public:
  explicit LowerVcvtOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VcvtOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VcvtOp op, pto::VcvtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<VcvtContract> contract = buildVcvtContract(op);
    if (failed(contract))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vcvt type pair");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vcvt result type");
    }

    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;
    callArgs.push_back(adaptor.getInput());
    argTypes.push_back(adaptor.getInput().getType());
    callArgs.push_back(adaptor.getMask());
    argTypes.push_back(adaptor.getMask().getType());

    std::optional<uint64_t> rnd = std::nullopt;
    if (op.getRndAttr()) {
      rnd = parseRoundModeImmediate(*op.getRnd());
    }
    std::optional<uint64_t> sat = std::nullopt;
    if (op.getSatAttr()) {
      sat = parseSaturationImmediate(*op.getSat());
    }

    if ((*contract).satBeforeRnd) {
      if (failed(appendVcvtImmediate(op, rewriter, "sat", (*contract).requiresSat,
                                     sat, callArgs, argTypes)))
      {
        return failure();
      }
      if (failed(appendVcvtImmediate(op, rewriter, "rnd", (*contract).requiresRnd,
                                     rnd, callArgs, argTypes)))
      {
        return failure();
      }
    } else {
      if (failed(appendVcvtImmediate(op, rewriter, "rnd", (*contract).requiresRnd,
                                     rnd, callArgs, argTypes)))
      {
        return failure();
      }
      if (failed(appendVcvtImmediate(op, rewriter, "sat", (*contract).requiresSat,
                                     sat, callArgs, argTypes)))
      {
        return failure();
      }
    }

    if ((*contract).requiresPart) {
      auto part =
          op.getPartAttr() ? parseVcvtPartImmediate(*op.getPart()) : std::nullopt;
      if (!part)
      {
        return rewriter.notifyMatchFailure(op, "vcvt requires valid part attr");
      }
      Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
      callArgs.push_back(partValue);
      argTypes.push_back(partValue.getType());
    }

    auto funcType = rewriter.getFunctionType(argTypes, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), StringRef((*contract).intrinsic), TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{std::string((*contract).intrinsic), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};


} // namespace

bool needsV300CtrlModeForVPTOFunc(func::FuncOp funcOp) {
  if (!pto::isPTOEntryFunction(funcOp) || funcOp.getBlocks().empty()) {
    return false;
  }

  bool needsCtrlSetup = false;
  funcOp.walk([&](pto::VcvtOp vcvtOp) {
    FailureOr<VcvtContract> contract = buildVcvtContract(vcvtOp);
    if (succeeded(contract) && (*contract).requiresSat) {
      needsCtrlSetup = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needsCtrlSetup;
}

void populateVPTOVcvtPatterns(TypeConverter &typeConverter,
                              RewritePatternSet &patterns,
                              LoweringState &state) {
  patterns.add<LowerVcvtOpPattern>(typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
