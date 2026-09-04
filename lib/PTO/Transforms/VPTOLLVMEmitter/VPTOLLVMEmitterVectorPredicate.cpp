// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::pto {
namespace {

static FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc,
                                                   Value input, Type resultType) {
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != 8) {
    return input;
  }
  Type resultElemType = getElementTypeFromVectorLike(resultType);
  std::string resultElemFragment = getElementTypeFragment(resultElemType);
  if ((resultElemFragment != "s8" && resultElemFragment != "u8") ||
      intType.isSignless()) {
    return input;
  }
  Type signlessType = builder.getIntegerType(intType.getWidth());
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{signlessType}, input)
      .getResult(0);
}

static Value normalizeByteScalarOperandForHivmCall(OpBuilder &builder, Location loc,
                                                   Value input,
                                                   Type semanticElementType) {
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != 8) {
    return input;
  }
  Type i16Type = builder.getIntegerType(16);
  auto semanticIntType = dyn_cast<IntegerType>(semanticElementType);
  if (semanticIntType && semanticIntType.isUnsigned()) {
    return builder.create<arith::ExtUIOp>(loc, i16Type, input).getResult();
  }
  return builder.create<arith::ExtSIOp>(loc, i16Type, input).getResult();
}

static bool isCompatibleScalarForSemanticType(Type semanticType, Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }
  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt) {
    return false;
  }
  if (semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }
  bool compatible;
  if (semanticInt.isSigned()) {
    compatible = scalarInt.isSigned() || scalarInt.isSignless();
  } else if (semanticInt.isUnsigned()) {
    compatible = scalarInt.isUnsigned() || scalarInt.isSignless();
  } else {
    compatible = scalarInt.isSignless();
  }
  return compatible;
}

static Type getLowpPayloadCarrierType(Type vectorLikeType, MLIRContext *context) {
  Type elementType = getElementTypeFromVectorLike(vectorLikeType);
  if (!elementType || (!pto::isPTOFloat8Type(elementType) &&
                       !pto::isPTOHiFloat8Type(elementType) &&
                       !pto::isPTOFloat4PackedType(elementType))) {
    return {};
  }
  auto lanes = getElementCountFromVectorLike(vectorLikeType);
  if (!lanes) {
    return {};
  }
  return VectorType::get({*lanes}, IntegerType::get(context, 8));
}

static std::string getVbrScalarFragment(Type type) {
  if (type.isF16()) return "f16";
  if (type.isBF16()) return "bf16";
  if (type.isF32()) return "f32";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return (intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
  }
  return {};
}

static std::optional<uint64_t> parseHiLoPartImmediate(StringRef part) {
  if (part == "LOWER") return 0;
  if (part == "HIGHER") return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicatePatternImmediate(StringRef pattern) {
  static constexpr llvm::StringLiteral names[] = {
      "PAT_ALL", "PAT_VL1", "PAT_VL2", "PAT_VL3", "PAT_VL4", "PAT_VL8",
      "PAT_VL16", "PAT_VL32", "PAT_VL64", "PAT_VL128", "PAT_M3", "PAT_M4",
      "PAT_H", "PAT_Q"};
  for (uint64_t index = 0; index < std::size(names); ++index) {
    if (pattern == names[index]) return index;
  }
  if (pattern == "PAT_ALLF") return 15;
  return std::nullopt;
}

static bool isMaskOnlyUsedByOnePointStores(Value mask) {
  return !mask.use_empty() && llvm::all_of(mask.getUsers(), [](Operation *user) {
    auto store = dyn_cast<pto::VstsOp>(user);
    if (!store || !store.getDist()) return false;
    const auto *contract =
        lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, *store.getDist());
    return contract && contract->isOnePointStore();
  });
}

static FailureOr<StringRef> buildVselCallee(MLIRContext *context, Type resultType) {
  std::string vec = getElementTypeFragment(
      cast<pto::VRegType>(resultType).getElementType());
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) return failure();
  return StringAttr::get(context, "llvm.hivm.vsel.v" + std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVselrCallee(MLIRContext *context, Type resultType) {
  Type elemType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!elemType || !lanes) return failure();
  std::string vec = getElementTypeFragment(elemType);
  if (auto floatType = dyn_cast<FloatType>(elemType); floatType && floatType.isF32()) {
    vec = "u32";
  }
  if (pto::isPTOFloat8Type(elemType) || pto::isPTOHiFloat8Type(elemType) ||
      pto::isPTOFloat4PackedType(elemType)) {
    vec = "u8";
  }
  if (vec.empty()) return failure();
  return StringAttr::get(context, "llvm.hivm.vselr.v" + std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVdupCallee(MLIRContext *context, pto::VdupOp op) {
  Type inputType = op.getInput().getType();
  Type resultType = op.getResult().getType();
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) return failure();
  if (isa<VectorType, pto::VRegType>(inputType)) {
    StringRef position = op.getPosition().value_or("LOWEST");
    StringRef family = position == "HIGHEST" ? "vdupm" : "vdup";
    return StringAttr::get(context, "llvm.hivm." + family.str() + ".v" +
                                        std::to_string(*lanes) + vec + ".z")
        .getValue();
  }
  return StringAttr::get(context, "llvm.hivm.vdups.v" + std::to_string(*lanes) +
                                      vec + ".z")
      .getValue();
}

static FailureOr<StringRef> buildVbrCallee(MLIRContext *context, Type elementType) {
  std::string scalar = getVbrScalarFragment(elementType);
  if (scalar.empty()) return failure();
  return StringAttr::get(context, "llvm.hivm.vbr." + scalar + ".v300").getValue();
}

static FailureOr<StringRef> buildVcmpCallee(MLIRContext *context, Type inputType,
                                            StringRef cmpMode, bool isScalarCompare) {
  std::string elem = getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  if (elem.empty()) return failure();
  StringRef stem = isScalarCompare ? "vcmps" : "vcmp";
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." +
                                      cmpMode.str() + "." + elem + ".z")
      .getValue();
}

static StringRef buildPnotCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pnot.z").getValue();
}
static StringRef buildPselCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psel").getValue();
}
static StringRef buildPandCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pand.z").getValue();
}
static StringRef buildPorCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.por.z").getValue();
}
static StringRef buildPxorCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pxor.z").getValue();
}
static StringRef buildPpackCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.ppack.z").getValue();
}
static StringRef buildPunpackCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.punpack").getValue();
}

template <typename Op>
static StringRef buildPredicatePairReorderCallee(MLIRContext *context);
template <> StringRef buildPredicatePairReorderCallee<pto::PdintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b8").getValue();
}
template <> StringRef buildPredicatePairReorderCallee<pto::PdintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b16").getValue();
}
template <> StringRef buildPredicatePairReorderCallee<pto::PdintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b32").getValue();
}
template <> StringRef buildPredicatePairReorderCallee<pto::PintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b8").getValue();
}
template <> StringRef buildPredicatePairReorderCallee<pto::PintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b16").getValue();
}
template <> StringRef buildPredicatePairReorderCallee<pto::PintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b32").getValue();
}

static FailureOr<StringRef> buildInterleaveCallee(MLIRContext *context, Type resultType,
                                                  StringRef stem) {
  auto lanes = getElementCountFromVectorLike(resultType);
  if (pto::isPTOBF16x2Type(getElementTypeFromVectorLike(resultType)) && lanes) {
    return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                        std::to_string(*lanes) + "i32")
        .getValue();
  }
  std::string elem = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (elem.empty() || !lanes) return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + elem)
      .getValue();
}

static FailureOr<StringRef> buildUnpackCallee(MLIRContext *context, Type inputType,
                                              Type resultType, StringRef stem) {
  std::string input = getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  std::string result = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (input.empty() || result.empty()) return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + input + "2" + result)
      .getValue();
}

static FailureOr<StringRef> buildVpackCallee(MLIRContext *context, Type inputType,
                                             Type resultType) {
  std::string input = getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  std::string result = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (input.empty() || result.empty()) return failure();
  return StringAttr::get(context, "llvm.hivm.vpack." + input + "2" + result + ".x")
      .getValue();
}

template <typename PredicateMaskOp>
static StringRef getPredicateMaskCallee(MLIRContext *context);
template <> StringRef getPredicateMaskCallee<pto::PnotOp>(MLIRContext *context) {
  return buildPnotCallee(context);
}
template <> StringRef getPredicateMaskCallee<pto::PselOp>(MLIRContext *context) {
  return buildPselCallee(context);
}
template <> StringRef getPredicateMaskCallee<pto::PandOp>(MLIRContext *context) {
  return buildPandCallee(context);
}
template <> StringRef getPredicateMaskCallee<pto::PorOp>(MLIRContext *context) {
  return buildPorCallee(context);
}
template <> StringRef getPredicateMaskCallee<pto::PxorOp>(MLIRContext *context) {
  return buildPxorCallee(context);
}

template <typename PackOp>
static StringRef getPredicatePackCallee(MLIRContext *context);
template <> StringRef getPredicatePackCallee<pto::PpackOp>(MLIRContext *context) {
  return buildPpackCallee(context);
}
template <> StringRef getPredicatePackCallee<pto::PunpackOp>(MLIRContext *context) {
  return buildPunpackCallee(context);
}

template <typename PltOp>
static StringRef buildPltCallee(MLIRContext *context);
template <> StringRef buildPltCallee<pto::PltB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b8.v300").getValue();
}
template <> StringRef buildPltCallee<pto::PltB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b16.v300").getValue();
}
template <> StringRef buildPltCallee<pto::PltB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b32.v300").getValue();
}

template <typename PltmOp>
static StringRef buildPltmCallee(MLIRContext *context);
template <> StringRef buildPltmCallee<pto::PltmB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b8.v300").getValue();
}
template <> StringRef buildPltmCallee<pto::PltmB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b16.v300").getValue();
}
template <> StringRef buildPltmCallee<pto::PltmB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pltm.b32.v300").getValue();
}

template <typename PsetOp>
static StringRef buildPsetCallee(MLIRContext *context);
template <> StringRef buildPsetCallee<pto::PsetB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b8").getValue();
}
template <> StringRef buildPsetCallee<pto::PsetB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b16").getValue();
}
template <> StringRef buildPsetCallee<pto::PsetB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b32").getValue();
}

template <typename PgeOp>
static StringRef buildPgeCallee(MLIRContext *context);
template <> StringRef buildPgeCallee<pto::PgeB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b8").getValue();
}
template <> StringRef buildPgeCallee<pto::PgeB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b16").getValue();
}
template <> StringRef buildPgeCallee<pto::PgeB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b32").getValue();
}

static func::CallOp createPlannedCall(Location loc, StringRef callee,
                                      Type resultType, ValueRange args,
                                      ConversionPatternRewriter &rewriter,
                                      LoweringState &state) {
  auto call = rewriter.create<func::CallOp>(loc, callee, TypeRange{resultType}, args);
  state.plannedDecls.push_back(PlannedDecl{callee.str(), call.getCalleeType()});
  return call;
}

static FailureOr<func::CallOp>
buildVdupPlannedCall(pto::VdupOp op, pto::VdupOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter, LoweringState &state,
                     StringRef callee, Type resultType, Type maskType,
                     SmallString<64> &reason) {
  Value mask = adaptor.getMask();
  if (!mask || mask.getType() != maskType) {
    reason = "unexpected converted vdup mask type";
    return failure();
  }

  SmallVector<Value> callArgs;
  bool vectorInput = isa<VectorType, pto::VRegType>(op.getInput().getType());
  if (vectorInput) {
    Value input = adaptor.getInput();
    if (!input || input.getType() != resultType) {
      reason = "vector-input vdup requires matching result type";
      return failure();
    }
    callArgs.push_back(input);
  } else {
    Type scalarType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!scalarType ||
        (op.getInput().getType() != scalarType &&
         !isCompatibleScalarForSemanticType(scalarType,
                                            op.getInput().getType()))) {
      reason = "unexpected scalar-input vdup type";
      return failure();
    }
    FailureOr<Value> normalizedScalar =
        normalizeVdupScalarOperand(rewriter, op.getLoc(), adaptor.getInput(),
                                   op.getResult().getType());
    if (failed(normalizedScalar)) {
      reason = "failed to normalize scalar vdup input";
      return failure();
    }
    callArgs.push_back(normalizeByteScalarOperandForHivmCall(
        rewriter, op.getLoc(), *normalizedScalar, scalarType));
  }

  callArgs.push_back(mask);
  callArgs.push_back(getI32Constant(rewriter, op.getLoc(), 1));
  return createPlannedCall(op.getLoc(), callee, resultType, callArgs, rewriter,
                           state);
}

static FailureOr<Type>
getVselrIntrinsicResultType(pto::VselrOp op, ConversionPatternRewriter &rewriter,
                            Type resultType, Type resultElementType,
                            std::optional<int64_t> lanes) {
  Type intrinsicResultType = resultType;
  if (auto floatType = dyn_cast<FloatType>(resultElementType);
      floatType && floatType.isF32()) {
    intrinsicResultType = VectorType::get({*lanes}, rewriter.getI32Type());
  }
  if (Type carrierType =
          getLowpPayloadCarrierType(op.getResult().getType(), rewriter.getContext())) {
    intrinsicResultType = carrierType;
  }
  return intrinsicResultType;
}

static FailureOr<func::CallOp>
materializeVselrCall(pto::VselrOp op, pto::VselrOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter, LoweringState &state,
                     StringRef callee, Type resultType,
                     Type intrinsicResultType, Type indexType,
                     SmallString<64> &reason) {
  Value src0 = adaptor.getSrc0();
  Value src1 = adaptor.getSrc1();
  if (!src0 || !src1 || src1.getType() != indexType) {
    reason = "unexpected converted vselr operand types";
    return failure();
  }
  if (src0.getType() != intrinsicResultType) {
    if (src0.getType() != resultType) {
      reason = "unexpected converted vselr source type";
      return failure();
    }
    src0 = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intrinsicResultType, src0);
  }
  return createPlannedCall(op.getLoc(), callee, intrinsicResultType,
                           ValueRange{src0, src1}, rewriter, state);
}

template <typename CmpOp>
static FailureOr<SmallVector<Value>> collectCompareCallArgs(
    CmpOp op, typename CmpOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, Type maskType,
    SmallString<64> &reason) {
  SmallVector<Value> callArgs;
  callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
  if constexpr (std::is_same_v<CmpOp, pto::VcmpsOp>) {
    if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
        callArgs[2].getType() != maskType) {
      reason = "unexpected converted scalar-compare operand types";
      return failure();
    }
    callArgs[1] = normalizeByteScalarOperandForHivmCall(
        rewriter, op.getLoc(), callArgs[1],
        cast<pto::VRegType>(op.getSrc().getType()).getElementType());
  } else {
    if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
        callArgs[0].getType() != callArgs[1].getType() ||
        callArgs[2].getType() != maskType) {
      reason = "unexpected converted compare operand types";
      return failure();
    }
  }
  return callArgs;
}

class LowerVselOpPattern final : public OpConversionPattern<pto::VselOp> {
public:
  explicit LowerVselOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VselOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VselOp op, pto::VselOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVselCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vsel VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vsel result type");
    }

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType ||
        src1.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vsel operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVdupOpPattern final : public OpConversionPattern<pto::VdupOp> {
public:
  explicit LowerVdupOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VdupOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VdupOp op, pto::VdupOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVdupCallee(op.getContext(), op);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vdup VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vdup result type");
    }

    SmallString<64> failureReason;
    FailureOr<func::CallOp> call = buildVdupPlannedCall(
        op, adaptor, rewriter, state, *calleeName, resultType, maskType,
        failureReason);
    if (failed(call))
    {
      return rewriter.notifyMatchFailure(op, failureReason);
    }
    rewriter.replaceOp(op, call->getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbrOpPattern final : public OpConversionPattern<pto::VbrOp> {
public:
  explicit LowerVbrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::VbrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VbrOp op, pto::VbrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVbrCallee(op.getContext(),
                       cast<pto::VRegType>(op.getResult().getType()).getElementType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vbr VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vbr result type");
    }

    Value scalar = adaptor.getValue();
    Type expectedScalarType =
        this->getTypeConverter()->convertType(op.getValue().getType());
    if (!scalar || !expectedScalarType || scalar.getType() != expectedScalarType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vbr operand type");
    }

    scalar = normalizeByteScalarOperandForHivmCall(
        rewriter, op.getLoc(), scalar,
        cast<pto::VRegType>(op.getResult().getType()).getElementType());

    auto funcType = rewriter.getFunctionType(TypeRange{scalar.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{scalar});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVselrOpPattern final : public OpConversionPattern<pto::VselrOp> {
public:
  explicit LowerVselrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VselrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VselrOp op, pto::VselrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVselrCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vselr VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vselr result type");
    }
    auto lanes = getElementCountFromVectorLike(resultType);
    Type resultElementType = getElementTypeFromVectorLike(resultType);
    if (!lanes || !resultElementType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vselr result type");
    }

    FailureOr<Type> intrinsicResultType = getVselrIntrinsicResultType(
        op, rewriter, resultType, resultElementType, lanes);
    if (failed(intrinsicResultType)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vselr result type");
    }
    Type indexType = this->getTypeConverter()->convertType(op.getSrc1().getType());
    if (!indexType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vselr index type");
    }

    SmallString<64> failureReason;
    FailureOr<func::CallOp> call = materializeVselrCall(
        op, adaptor, rewriter, state, *calleeName, resultType,
        *intrinsicResultType, indexType, failureReason);
    if (failed(call))
    {
      return rewriter.notifyMatchFailure(op, failureReason);
    }

    Value result = call->getResult(0);
    if (*intrinsicResultType != resultType)
    {
      result = rewriter.create<LLVM::BitcastOp>(op.getLoc(), resultType, result);
    }
    rewriter.replaceOp(op, ValueRange{result});
    return success();
  }

private:
  LoweringState &state;
};

class LowerPnotOpPattern final : public OpConversionPattern<pto::PnotOp> {
public:
  explicit LowerPnotOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::PnotOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::PnotOp op, pto::PnotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert pnot result type");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted pnot operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<pto::PnotOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename InterleaveOp>
class LowerInterleaveOpPattern final
    : public OpConversionPattern<InterleaveOp> {
public:
  explicit LowerInterleaveOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<InterleaveOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(InterleaveOp op, typename InterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<InterleaveOp, pto::VintlvOp> ? "vintlv" : "vdintlv";
    FailureOr<StringRef> calleeName =
        buildInterleaveCallee(op.getContext(), op.getLow().getType(), stem);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported interleave VPTO signature");
    }

    Type lowType = this->getTypeConverter()->convertType(op.getLow().getType());
    Type highType = this->getTypeConverter()->convertType(op.getHigh().getType());
    if (!lowType || !highType || lowType != highType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert interleave result types");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != lowType || rhs.getType() != lowType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted interleave operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lowType, lowType},
                                             TypeRange{lowType, highType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{lowType, highType}, ValueRange{lhs, rhs});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PackOp>
class LowerPredicatePackOpPattern final : public OpConversionPattern<PackOp> {
public:
  explicit LowerPredicatePackOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<PackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PackOp op, typename PackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-pack result type");
    }

    auto part = parseHiLoPartImmediate(op.getPart());
    if (!part) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-pack part immediate");
    }

    Value input = adaptor.getInput();
    if (!input || input.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-pack operand type");
    }

    Value partValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*part));
    StringRef calleeName = getPredicatePackCallee<PackOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, rewriter.getI32Type()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{input, partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnpackOp>
class LowerUnpackOpPattern final : public OpConversionPattern<UnpackOp> {
public:
  explicit LowerUnpackOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnpackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnpackOp op, typename UnpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<UnpackOp, pto::VsunpackOp> ? "vsunpack"
                                                               : "vzunpack";
    FailureOr<StringRef> calleeName = buildUnpackCallee(
        op.getContext(), op.getSrc().getType(), op.getResult().getType(), stem);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported unpack VPTO signature");
    }

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert unpack types");
    }

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted unpack source type");
    }

    Value part = castIntegerLikeTo(op, adaptor.getPart(), rewriter.getI32Type());
    if (!part)
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize unpack part");
    }

    auto call = createPlannedCall(op.getLoc(), *calleeName, resultType,
                                  ValueRange{src, part}, rewriter, state);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVpackOpPattern final : public OpConversionPattern<pto::VpackOp> {
public:
  explicit LowerVpackOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VpackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VpackOp op, pto::VpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVpackCallee(op.getContext(), op.getSrc().getType(),
                         op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vpack VPTO signature");
    }

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vpack types");
    }

    auto partImm = parseHiLoPartImmediate(op.getPart());
    if (!partImm)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vpack part immediate");
    }

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vpack source type");
    }

    Value part = getI32Constant(rewriter, op.getLoc(), *partImm);
    auto call = createPlannedCall(op.getLoc(), *calleeName, resultType,
                                  ValueRange{src, part}, rewriter, state);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PredicateMaskOp>
class LowerPredicateMaskBinaryOpPattern final
    : public OpConversionPattern<PredicateMaskOp> {
public:
  explicit LowerPredicateMaskBinaryOpPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             LoweringState &state)
      : OpConversionPattern<PredicateMaskOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(PredicateMaskOp op, typename PredicateMaskOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-mask result type");
    }

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType ||
        src1.getType() != resultType || mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-mask operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<PredicateMaskOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReorderOp>
class LowerPredicatePairReorderOpPattern final
    : public OpConversionPattern<ReorderOp> {
public:
  explicit LowerPredicatePairReorderOpPattern(TypeConverter &typeConverter,
                                              MLIRContext *context,
                                              LoweringState &state)
      : OpConversionPattern<ReorderOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReorderOp op, typename ReorderOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-pair-reorder result types");
    }
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1]) {
      return rewriter.notifyMatchFailure(
          op, "unexpected predicate-pair-reorder converted result types");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != resultTypes[0] ||
        rhs.getType() != resultTypes[0]) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-pair-reorder operand types");
    }

    StringRef calleeName =
        buildPredicatePairReorderCallee<ReorderOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes,
                                              ValueRange{lhs, rhs});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CmpOp>
class LowerCmpOpPattern final : public OpConversionPattern<CmpOp> {
public:
  explicit LowerCmpOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<CmpOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CmpOp op, typename CmpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isScalarCompare = std::is_same_v<CmpOp, pto::VcmpsOp>;
    Type inputType = Type();
    if constexpr (isScalarCompare)
    {
      inputType = op.getSrc().getType();
    }
    else
    {
      inputType = op.getSrc0().getType();
    }
    FailureOr<StringRef> calleeName =
        buildVcmpCallee(op.getContext(), inputType, op.getCmpMode(),
                        isScalarCompare);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported compare VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType =
        this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert compare result type");
    }
    if (resultType != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected compare mask conversion");
    }

    SmallString<64> failureReason;
    FailureOr<SmallVector<Value>> callArgs =
        collectCompareCallArgs(op, adaptor, rewriter, maskType, failureReason);
    if (failed(callArgs)) {
      return rewriter.notifyMatchFailure(op, failureReason);
    }

    auto call = createPlannedCall(op.getLoc(), *calleeName, resultType,
                                  *callArgs, rewriter, state);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PltOp>
class LowerPltOpPattern final : public OpConversionPattern<PltOp> {
public:
  explicit LowerPltOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<PltOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PltOp op, typename PltOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value laneCount = castIntegerLikeTo(op, adaptor.getScalar(), rewriter.getI32Type());
    if (!laneCount)
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize plt lane count");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
    {
      return rewriter.notifyMatchFailure(op, "failed to convert plt result types");
    }

    StringRef calleeName = buildPltCallee<PltOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{laneCount});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PltmOp>
class LowerPltmOpPattern final : public OpConversionPattern<PltmOp> {
public:
  explicit LowerPltmOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<PltmOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PltmOp op, typename PltmOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert pltm result type");
    }

    Value loop = adaptor.getLoop();
    Value bound = adaptor.getBound();
    if (!loop || !bound || !loop.getType().isInteger(16) ||
        !bound.getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted pltm operand types");
    }

    StringRef calleeName = buildPltmCallee<PltmOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI16Type(), rewriter.getI32Type()}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{loop, bound});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PsetOp>
class LowerPsetOpPattern final : public OpConversionPattern<PsetOp> {
public:
  explicit LowerPsetOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<PsetOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PsetOp op, typename PsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern)
    {
      return rewriter.notifyMatchFailure(op, "unsupported pset pattern");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
    {
      return rewriter.notifyMatchFailure(op, "failed to convert pset result types");
    }

    if (isMaskOnlyUsedByOnePointStores(op.getResult())) {
      auto undef = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultTypes.front());
      rewriter.replaceOp(op, undef.getResult());
      return success();
    }

    StringRef calleeName = buildPsetCallee<PsetOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{patternValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PgeOp>
class LowerPgeOpPattern final : public OpConversionPattern<PgeOp> {
public:
  explicit LowerPgeOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<PgeOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PgeOp op, typename PgeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern)
    {
      return rewriter.notifyMatchFailure(op, "unsupported pge pattern");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
    {
      return rewriter.notifyMatchFailure(op, "failed to convert pge result types");
    }

    if (isMaskOnlyUsedByOnePointStores(op.getResult())) {
      auto undef = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultTypes.front());
      rewriter.replaceOp(op, undef.getResult());
      return success();
    }

    StringRef calleeName = buildPgeCallee<PgeOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(0));
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI32Type(), rewriter.getI32Type()}, resultTypes);
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes,
                                      ValueRange{patternValue, zero});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

} // namespace

void populateVPTOVectorPredicatePatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         LoweringState &state) {
  patterns.add<LowerVdupOpPattern, LowerVbrOpPattern,
               LowerPredicatePackOpPattern<pto::PpackOp>,
               LowerPredicatePackOpPattern<pto::PunpackOp>,
               LowerVselOpPattern, LowerVselrOpPattern, LowerPnotOpPattern,
               LowerPredicateMaskBinaryOpPattern<pto::PselOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PandOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PorOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PxorOp>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB8Op>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB16Op>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB32Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB8Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB16Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB32Op>,
               LowerUnpackOpPattern<pto::VsunpackOp>,
               LowerUnpackOpPattern<pto::VzunpackOp>,
               LowerVpackOpPattern,
               LowerInterleaveOpPattern<pto::VintlvOp>,
               LowerInterleaveOpPattern<pto::VdintlvOp>,
               LowerCmpOpPattern<pto::VcmpOp>,
               LowerCmpOpPattern<pto::VcmpsOp>,
               LowerPltOpPattern<pto::PltB8Op>,
               LowerPltOpPattern<pto::PltB16Op>,
               LowerPltOpPattern<pto::PltB32Op>,
               LowerPltmOpPattern<pto::PltmB8Op>,
               LowerPltmOpPattern<pto::PltmB16Op>,
               LowerPltmOpPattern<pto::PltmB32Op>,
               LowerPsetOpPattern<pto::PsetB8Op>,
               LowerPsetOpPattern<pto::PsetB16Op>,
               LowerPsetOpPattern<pto::PsetB32Op>,
               LowerPgeOpPattern<pto::PgeB8Op>,
               LowerPgeOpPattern<pto::PgeB16Op>,
               LowerPgeOpPattern<pto::PgeB32Op>>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
