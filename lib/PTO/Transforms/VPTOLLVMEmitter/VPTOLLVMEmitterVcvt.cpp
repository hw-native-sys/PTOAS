// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// The CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

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


static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

[[maybe_unused]] static Value getI1Constant(OpBuilder &builder, Location loc,
                                            bool value) {
  return builder
      .create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(builder.getI1Type(), value ? 1 : 0))
      .getResult();
}

static std::optional<uint64_t> parseRoundModeImmediate(StringRef roundMode) {
  if (roundMode == "R" || roundMode == "ROUND_R")
  {
    return 0;
  }
  if (roundMode == "A" || roundMode == "ROUND_A")
  {
    return 1;
  }
  if (roundMode == "F" || roundMode == "ROUND_F")
  {
    return 2;
  }
  if (roundMode == "C" || roundMode == "ROUND_C")
  {
    return 3;
  }
  if (roundMode == "Z" || roundMode == "ROUND_Z")
  {
    return 4;
  }
  if (roundMode == "O" || roundMode == "ROUND_O")
  {
    return 5;
  }
  if (roundMode == "H" || roundMode == "ROUND_H")
  {
    return 6;
  }
  return std::nullopt;
}

static std::optional<uint64_t> parseSaturationImmediate(StringRef sat) {
  if (sat == "SAT")
  {
    return 1;
  }
  if (sat == "NOSAT")
  {
    return 0;
  }
  return std::nullopt;
}

static std::optional<uint64_t> parseVcvtPartImmediate(StringRef part) {
  if (part == "EVEN" || part == "PART_EVEN" || part == "P0" ||
      part == "PART_P0") {
    return 0;
  }
  if (part == "ODD" || part == "PART_ODD" || part == "P1" ||
      part == "PART_P1") {
    return 1;
  }
  if (part == "P2" || part == "PART_P2")
  {
    return 2;
  }
  if (part == "P3" || part == "PART_P3")
  {
    return 3;
  }
  return std::nullopt;
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
  switch (src) {
  case VcvtElemKind::F32:
    switch (dst) {
    case VcvtElemKind::F8E4M3:
      return VcvtContract{"llvm.hivm.vcvtff.f322f8e4m3.x", true, true, true, 32};
    case VcvtElemKind::F8E5M2:
      return VcvtContract{"llvm.hivm.vcvtff.f322f8e5m2.x", true, true, true, 32};
    case VcvtElemKind::HiF8:
      return VcvtContract{"llvm.hivm.vcvtff.f322hif8.x", true, true, true, 32};
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtff.f322f16.x", true, true, true, 32};
    case VcvtElemKind::BF16:
      return VcvtContract{"llvm.hivm.vcvtff.f322bf16.x", true, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s16.x", true, true, true, 32};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s32.x", true, true, false, 32};
    case VcvtElemKind::S64:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s64.x", true, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F16:
    switch (dst) {
    case VcvtElemKind::F8E4M3:
      return VcvtContract{"llvm.hivm.vcvtff.f162f8e4m3.x", true, true, true, 16};
    case VcvtElemKind::F8E5M2:
      return VcvtContract{"llvm.hivm.vcvtff.f162f8e5m2.x", true, true, true, 16};
    case VcvtElemKind::HiF8:
      return VcvtContract{"llvm.hivm.vcvtff.f162hif8.x", true, true, true, 16};
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.f162f32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s32.x", true, false, true, 16};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s16.x", true, true, false, 16};
    case VcvtElemKind::S8:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s8.x", true, true, true, 16};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtfi.f162u8.x", true, true, true, 16};
    case VcvtElemKind::BF16:
      return VcvtContract{"llvm.hivm.vcvtff.f162bf16.x", true, false, false, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::BF16:
    switch (dst) {
    case VcvtElemKind::F8E4M3:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f8e4m3.x", true, true, true, 16};
    case VcvtElemKind::F8E5M2:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f8e5m2.x", true, true, true, 16};
    case VcvtElemKind::F4E1M2x2:
      return VcvtContract{"llvm.hivm.vcvtff2.bf162f4e1m2x2.x", true, false, true, 16};
    case VcvtElemKind::F4E2M1x2:
      return VcvtContract{"llvm.hivm.vcvtff2.bf162f4e2m1x2.x", true, false, true, 16};
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f16.x", true, true, false, 16,
                          true};
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.bf162s32.x", true, true, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U8:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.u82f16.x", false, false, true, 8};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.u82u16.x", false, false, true, 8};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.u82u32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S8:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.s82f16.x", false, false, true, 8};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.s82s16.x", false, false, true, 8};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s82s32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U16:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.u162u8.x", false, true, true, 16};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.u162u32.x", false, false, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S16:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.s162f16.x", true, false, false, 16};
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s162f32.x", false, false, true, 16};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.s162u8.x", false, true, true, 16};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.s162u32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s162s32.x", false, false, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U32:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.u322u8.x", false, true, true, 32};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.u322u16.x", false, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.u322s16.x", false, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S32:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s322f32.x", true, false, false, 32};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.s322u8.x", false, true, true, 32};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.s322u16.x", false, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.s322s16.x", false, true, true, 32};
    case VcvtElemKind::S64:
      return VcvtContract{"llvm.hivm.vcvtii.s322s64.x", false, false, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S64:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s642f32.x", true, false, true, 32};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s642s32.x", false, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F8E4M3:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.f8e4m32f32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F8E5M2:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.f8e5m22f32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::HiF8:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.hif82f32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F4E1M2x2:
    switch (dst) {
    case VcvtElemKind::BF16:
      return VcvtContract{"llvm.hivm.vcvtff2.f4e1m2x22bf16.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F4E2M1x2:
    switch (dst) {
    case VcvtElemKind::BF16:
      return VcvtContract{"llvm.hivm.vcvtff2.f4e2m1x22bf16.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
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

    auto appendRndArg = [&]() -> LogicalResult {
      auto roundMode =
          op.getRndAttr() ? parseRoundModeImmediate(*op.getRnd()) : std::nullopt;
      if (!roundMode)
      {
        return rewriter.notifyMatchFailure(op, "vcvt requires valid rnd attr");
      }
      Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
      callArgs.push_back(roundValue);
      argTypes.push_back(roundValue.getType());
      return success();
    };

    auto appendSatArg = [&]() -> LogicalResult {
      auto saturation =
          op.getSatAttr() ? parseSaturationImmediate(*op.getSat()) : std::nullopt;
      if (!saturation)
      {
        return rewriter.notifyMatchFailure(op, "vcvt requires valid sat attr");
      }
      Value satValue = getI32Constant(rewriter, op.getLoc(), *saturation);
      callArgs.push_back(satValue);
      argTypes.push_back(satValue.getType());
      return success();
    };

    if ((*contract).satBeforeRnd) {
      if ((*contract).requiresSat && failed(appendSatArg()))
      {
        return failure();
      }
      if ((*contract).requiresRnd && failed(appendRndArg()))
      {
        return failure();
      }
    } else {
      if ((*contract).requiresRnd && failed(appendRndArg()))
      {
        return failure();
      }
      if ((*contract).requiresSat && failed(appendSatArg()))
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
