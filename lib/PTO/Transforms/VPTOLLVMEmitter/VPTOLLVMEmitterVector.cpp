// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
using namespace detail;
namespace {

template <typename UnaryOp>
StringRef getUnaryMaskedStem() {
  if constexpr (std::is_same_v<UnaryOp, pto::VabsOp>) return "vabs";
  if constexpr (std::is_same_v<UnaryOp, pto::VexpOp>) return "vexp";
  if constexpr (std::is_same_v<UnaryOp, pto::VlnOp>) return "vln";
  if constexpr (std::is_same_v<UnaryOp, pto::VnegOp>) return "vneg";
  if constexpr (std::is_same_v<UnaryOp, pto::VsqrtOp>) return "vsqrt";
  if constexpr (std::is_same_v<UnaryOp, pto::VreluOp>) return "vrelu";
  if constexpr (std::is_same_v<UnaryOp, pto::VnotOp>) return "vnot";
  return {};
}

FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context, Type resultType,
                                          StringRef stem, StringRef suffix) {
  std::string vec = detail::getElementTypeFragment(
      detail::getElementTypeFromVectorLike(resultType));
  auto lanes = detail::getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec + suffix.str())
      .getValue();
}

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value));
}

static uint64_t determineVsqzStoreHint(pto::VsqzOp vsqz) {
  Value result = vsqz.getResult();
  for (Operation *user : result.getUsers()) {
    auto vstur = dyn_cast<pto::VsturOp>(user);
    if (vstur && vstur.getValue() == result) return 1;
  }
  return 0;
}

static FailureOr<StringRef> buildVsqzCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vsqz", ".x.v300");
}

static FailureOr<StringRef> buildVusqzCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vusqz", ".m");
}

static FailureOr<StringRef> buildVmulaCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmula", ".m");
}

static FailureOr<StringRef> buildVmullCallee(MLIRContext *context, Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmull", "");
}

template <typename BinaryOp>
static StringRef getBinaryMaskedStem() {
  if constexpr (std::is_same_v<BinaryOp, pto::VaddOp>) return "vadd";
  if constexpr (std::is_same_v<BinaryOp, pto::VsubOp>) return "vsub";
  if constexpr (std::is_same_v<BinaryOp, pto::VmulOp>) return "vmul";
  if constexpr (std::is_same_v<BinaryOp, pto::VdivOp>) return "vdiv";
  if constexpr (std::is_same_v<BinaryOp, pto::VmaxOp>) return "vmax";
  if constexpr (std::is_same_v<BinaryOp, pto::VminOp>) return "vmin";
  if constexpr (std::is_same_v<BinaryOp, pto::VandOp>) return "vand";
  if constexpr (std::is_same_v<BinaryOp, pto::VorOp>) return "vor";
  if constexpr (std::is_same_v<BinaryOp, pto::VxorOp>) return "vxor";
  if constexpr (std::is_same_v<BinaryOp, pto::VshlOp>) return "vshl";
  if constexpr (std::is_same_v<BinaryOp, pto::VshrOp>) return "vshr";
  if constexpr (std::is_same_v<BinaryOp, pto::VpreluOp>) return "vprelu";
  return {};
}

template <typename TernaryOp>
static StringRef getTernaryMaskedStem() {
  if constexpr (std::is_same_v<TernaryOp, pto::VmaddOp>) return "vmadd";
  return {};
}

template <typename CarryOp>
static StringRef getCarryBinaryStem() {
  if constexpr (std::is_same_v<CarryOp, pto::VaddcOp>) return "vaddc";
  if constexpr (std::is_same_v<CarryOp, pto::VsubcOp>) return "vsubc";
  if constexpr (std::is_same_v<CarryOp, pto::VaddcsOp>) return "vaddcs";
  if constexpr (std::is_same_v<CarryOp, pto::VsubcsOp>) return "vsubcs";
  return {};
}

template <typename CarryOp>
static constexpr bool hasCarryInput() {
  return std::is_same_v<CarryOp, pto::VaddcsOp> ||
         std::is_same_v<CarryOp, pto::VsubcsOp>;
}

static FailureOr<StringRef> buildCarryBinaryCallee(MLIRContext *context,
                                                   Type resultType,
                                                   StringRef stem) {
  std::string vec = detail::getElementTypeFragment(
      cast<pto::VRegType>(resultType).getElementType());
  auto lanes = detail::getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

template <typename TernaryOp>
class LowerTernaryMaskedOpPattern final : public OpConversionPattern<TernaryOp> {
public:
  explicit LowerTernaryMaskedOpPattern(TypeConverter &converter, MLIRContext *context,
                                       detail::LoweringState &state)
      : OpConversionPattern<TernaryOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(TernaryOp op, typename TernaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> callee = buildLaneTypedCallee(
        op.getContext(), op.getResult().getType(), getTernaryMaskedStem<TernaryOp>(), ".m");
    Type result = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (failed(callee) || !result || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert ternary VPTO types");
    Value acc = adaptor.getAcc(), lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!acc || !lhs || !rhs || !mask || acc.getType() != result ||
        lhs.getType() != result || rhs.getType() != result || mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op, "unexpected converted ternary operand types");
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee, TypeRange{result},
                                               ValueRange{acc, lhs, rhs, mask});
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
private:
  detail::LoweringState &state;
};

template <typename CarryOp>
class LowerCarryBinaryOpPattern final : public OpConversionPattern<CarryOp> {
public:
  explicit LowerCarryBinaryOpPattern(TypeConverter &converter, MLIRContext *context,
                                     detail::LoweringState &state)
      : OpConversionPattern<CarryOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(CarryOp op, typename CarryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> callee = buildCarryBinaryCallee(
        op.getContext(), op.getResult().getType(), getCarryBinaryStem<CarryOp>());
    Type result = this->getTypeConverter()->convertType(op.getResult().getType());
    Type carry = this->getTypeConverter()->convertType(op->getResult(1).getType());
    if (failed(callee) || !result || !carry)
      return rewriter.notifyMatchFailure(op, "failed to convert carry result types");
    SmallVector<Value> args(adaptor.getOperands().begin(), adaptor.getOperands().end());
    size_t expected = hasCarryInput<CarryOp>() ? 4 : 3;
    if (args.size() != expected || args[0].getType() != result ||
        args[1].getType() != result || args.back().getType() != carry ||
        (hasCarryInput<CarryOp>() && args[2].getType() != carry))
      return rewriter.notifyMatchFailure(op, "unexpected converted carry operands");
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee,
                                               TypeRange{result, carry}, args);
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
private:
  detail::LoweringState &state;
};

static Type getLowpPayloadCarrierType(Type vectorLikeType, MLIRContext *context) {
  Type elementType = detail::getElementTypeFromVectorLike(vectorLikeType);
  if (!elementType || !pto::isPTOLowPrecisionType(elementType)) return {};
  auto lanes = detail::getElementCountFromVectorLike(vectorLikeType);
  if (!lanes) return {};
  return VectorType::get({*lanes}, IntegerType::get(context, 8));
}

static Value castToPayloadABI(Location loc, Value value, Type semanticType,
                              ConversionPatternRewriter &rewriter) {
  Type carrier = getLowpPayloadCarrierType(semanticType, rewriter.getContext());
  if (!carrier || carrier == value.getType()) return value;
  return rewriter.create<LLVM::BitcastOp>(loc, carrier, value);
}

static Value castFromPayloadABI(Location loc, Value value, Type semanticType,
                                Type convertedType,
                                ConversionPatternRewriter &rewriter) {
  Type carrier = getLowpPayloadCarrierType(semanticType, rewriter.getContext());
  if (!carrier || carrier == convertedType) return value;
  return rewriter.create<LLVM::BitcastOp>(loc, convertedType, value);
}

static FailureOr<StringRef> buildDirectLowpVLogicCallee(MLIRContext *context,
                                                        Type vectorType,
                                                        StringRef stem,
                                                        StringRef mode) {
  Type element = detail::getElementTypeFromVectorLike(vectorType);
  auto lanes = detail::getElementCountFromVectorLike(vectorType);
  std::string elem;
  if (pto::isPTOFloat8E4M3LikeType(element)) elem = "fp8e4m3";
  if (pto::isPTOFloat8E5M2LikeType(element)) elem = "fp8e5m2";
  if (elem.empty() || !lanes) return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + mode.str() +
                                      ".v" + std::to_string(*lanes) + elem)
      .getValue();
}

static FailureOr<StringRef> buildLowpPayloadVLogicCallee(MLIRContext *context,
                                                         Type vectorType,
                                                         StringRef stem,
                                                         StringRef mode) {
  auto lanes = detail::getElementCountFromVectorLike(vectorType);
  if (!lanes || !detail::getElementTypeFromVectorLike(vectorType)) return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + "u8." + mode.str())
      .getValue();
}

template <typename BinaryOp>
class LowerBinaryMaskedOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryMaskedOpPattern(TypeConverter &converter, MLIRContext *context,
                                      detail::LoweringState &state)
      : OpConversionPattern<BinaryOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type result = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!result || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert binary result type");
    Value lhs = adaptor.getOperands()[0], rhs = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    if (!lhs || !rhs || !mask || lhs.getType() != result || rhs.getType() != result ||
        mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op, "unexpected converted binary operand types");
    StringRef stem = getBinaryMaskedStem<BinaryOp>();
    Type callResult = result;
    Value callLhs = lhs, callRhs = rhs;
    FailureOr<StringRef> callee = buildLaneTypedCallee(
        op.getContext(), op.getResult().getType(), stem, ".x");
    if constexpr (std::is_same_v<BinaryOp, pto::VandOp> ||
                  std::is_same_v<BinaryOp, pto::VorOp> ||
                  std::is_same_v<BinaryOp, pto::VxorOp>) {
      Type element = detail::getElementTypeFromVectorLike(op.getResult().getType());
      if (element && pto::isPTOLowPrecisionType(element)) {
        callee = buildDirectLowpVLogicCallee(op.getContext(), op.getResult().getType(),
                                             stem, "x");
        if (failed(callee)) {
          Type carrier = getLowpPayloadCarrierType(op.getResult().getType(), rewriter.getContext());
          if (!carrier) return rewriter.notifyMatchFailure(op, "unsupported low-precision payload ABI");
          callResult = carrier;
          callLhs = castToPayloadABI(op.getLoc(), lhs, op.getResult().getType(), rewriter);
          callRhs = castToPayloadABI(op.getLoc(), rhs, op.getResult().getType(), rewriter);
          callee = buildLowpPayloadVLogicCallee(op.getContext(), op.getResult().getType(), stem, "x");
        }
      }
    }
    if (failed(callee)) return rewriter.notifyMatchFailure(op, "unsupported binary VPTO signature");
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee, TypeRange{callResult},
                                               ValueRange{callLhs, callRhs, mask});
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), call.getCalleeType()});
    rewriter.replaceOp(op, castFromPayloadABI(op.getLoc(), call.getResult(0),
                                              op.getResult().getType(), result, rewriter));
    return success();
  }
private:
  detail::LoweringState &state;
};

class LowerVmullOpPattern final : public OpConversionPattern<pto::VmullOp> {
public:
  explicit LowerVmullOpPattern(TypeConverter &converter, MLIRContext *context,
                               detail::LoweringState &state)
      : OpConversionPattern<pto::VmullOp>(converter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VmullOp op, pto::VmullOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> callee =
        buildVmullCallee(op.getContext(), op.getLow().getType());
    Type inputType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    SmallVector<Type> resultTypes;
    if (failed(callee) || !inputType || !maskType ||
        failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert vmull types");
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1])
      return rewriter.notifyMatchFailure(op, "unexpected converted vmull results");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!lhs || !rhs || !mask || lhs.getType() != inputType ||
        rhs.getType() != inputType || mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op, "unexpected converted vmull operand types");

    auto functionType = rewriter.getFunctionType(
        TypeRange{inputType, inputType, maskType}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee, resultTypes,
                                               ValueRange{lhs, rhs, mask});
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), functionType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  detail::LoweringState &state;
};

class LowerVmulaOpPattern final : public OpConversionPattern<pto::VmulaOp> {
public:
  explicit LowerVmulaOpPattern(TypeConverter &converter, MLIRContext *context,
                               detail::LoweringState &state)
      : OpConversionPattern<pto::VmulaOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(pto::VmulaOp op, pto::VmulaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> callee = buildVmulaCallee(op.getContext(), op.getResult().getType());
    Type result = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (failed(callee) || !result || !maskType)
      return rewriter.notifyMatchFailure(op, "unsupported vmula VPTO signature");
    Value acc = adaptor.getAcc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!acc || !lhs || !rhs || !mask || acc.getType() != result ||
        lhs.getType() != result || rhs.getType() != result ||
        mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op, "unexpected converted vmula operand types");
    auto functionType = rewriter.getFunctionType(
        TypeRange{result, result, result, maskType}, TypeRange{result});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee,
                                               TypeRange{result},
                                               ValueRange{acc, lhs, rhs, mask});
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), functionType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
private:
  detail::LoweringState &state;
};

class LowerVsqzOpPattern final : public OpConversionPattern<pto::VsqzOp> {
public:
  explicit LowerVsqzOpPattern(TypeConverter &converter, MLIRContext *context,
                              detail::LoweringState &state)
      : OpConversionPattern<pto::VsqzOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(pto::VsqzOp op, pto::VsqzOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> callee = buildVsqzCallee(op.getContext(), op.getResult().getType());
    Type result = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (failed(callee) || !result || !maskType)
      return rewriter.notifyMatchFailure(op, "unsupported vsqz VPTO signature");
    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != result || mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op, "unexpected converted vsqz operand types");
    Value hint = getI32Constant(rewriter, op.getLoc(), determineVsqzStoreHint(op));
    auto funcType = rewriter.getFunctionType(TypeRange{result, maskType, hint.getType()},
                                              TypeRange{result});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee, TypeRange{result},
                                               ValueRange{input, mask, hint});
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
private:
  detail::LoweringState &state;
};

class LowerVusqzOpPattern final : public OpConversionPattern<pto::VusqzOp> {
public:
  explicit LowerVusqzOpPattern(TypeConverter &converter, MLIRContext *context,
                               detail::LoweringState &state)
      : OpConversionPattern<pto::VusqzOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(pto::VusqzOp op, pto::VusqzOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> callee = buildVusqzCallee(op.getContext(), op.getResult().getType());
    Type result = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (failed(callee) || !result || !maskType)
      return rewriter.notifyMatchFailure(op, "unsupported vusqz VPTO signature");
    Value src = adaptor.getSrc();
    Value mask = adaptor.getMask();
    if (!src || !mask || src.getType() != result || mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op, "unexpected converted vusqz operand types");
    auto funcType = rewriter.getFunctionType(TypeRange{result, maskType}, TypeRange{result});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callee, TypeRange{result},
                                               ValueRange{src, mask});
    state.plannedDecls.push_back(detail::PlannedDecl{callee->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
private:
  detail::LoweringState &state;
};

template <typename UnaryOp>
class LowerUnaryMaskedOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryMaskedOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     detail::LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildLaneTypedCallee(
        op.getContext(), op.getResult().getType(), getUnaryMaskedStem<UnaryOp>(),
        ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported unary VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert unary result type");

    Value input = adaptor.getOperands()[0];
    Value mask = adaptor.getOperands()[1];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(1).getType());
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != expectedMaskType)
      return rewriter.notifyMatchFailure(
          op, "unexpected converted unary VPTO operand types");

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{input, mask});
    state.plannedDecls.push_back(
        detail::PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  detail::LoweringState &state;
};

} // namespace

void populateVPTOVectorUnaryPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     detail::LoweringState &state) {
  patterns.add<LowerUnaryMaskedOpPattern<pto::VabsOp>,
               LowerUnaryMaskedOpPattern<pto::VexpOp>,
               LowerUnaryMaskedOpPattern<pto::VlnOp>,
               LowerUnaryMaskedOpPattern<pto::VnegOp>,
               LowerUnaryMaskedOpPattern<pto::VsqrtOp>,
               LowerUnaryMaskedOpPattern<pto::VreluOp>,
               LowerUnaryMaskedOpPattern<pto::VnotOp>>(
      typeConverter, patterns.getContext(), state);
}

void populateVPTOVectorCompactionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          detail::LoweringState &state) {
  patterns.add<LowerVsqzOpPattern, LowerVusqzOpPattern>(
      typeConverter, patterns.getContext(), state);
}

void populateVPTOVectorMulaPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    detail::LoweringState &state) {
  patterns.add<LowerVmulaOpPattern>(typeConverter, patterns.getContext(), state);
}

void populateVPTOVectorBinaryPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      detail::LoweringState &state) {
  patterns.add<LowerBinaryMaskedOpPattern<pto::VaddOp>,
               LowerBinaryMaskedOpPattern<pto::VsubOp>,
               LowerBinaryMaskedOpPattern<pto::VmulOp>,
               LowerBinaryMaskedOpPattern<pto::VdivOp>,
               LowerBinaryMaskedOpPattern<pto::VmaxOp>,
               LowerBinaryMaskedOpPattern<pto::VminOp>,
               LowerBinaryMaskedOpPattern<pto::VandOp>,
               LowerBinaryMaskedOpPattern<pto::VorOp>,
               LowerBinaryMaskedOpPattern<pto::VxorOp>,
               LowerBinaryMaskedOpPattern<pto::VpreluOp>,
               LowerBinaryMaskedOpPattern<pto::VshlOp>,
               LowerBinaryMaskedOpPattern<pto::VshrOp>>(
      typeConverter, patterns.getContext(), state);
}

void populateVPTOVectorVmullPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     detail::LoweringState &state) {
  patterns.add<LowerVmullOpPattern>(typeConverter, patterns.getContext(), state);
}

void populateVPTOVectorCarryPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     detail::LoweringState &state) {
  patterns.add<LowerTernaryMaskedOpPattern<pto::VmaddOp>,
               LowerCarryBinaryOpPattern<pto::VaddcOp>,
               LowerCarryBinaryOpPattern<pto::VsubcOp>,
               LowerCarryBinaryOpPattern<pto::VaddcsOp>,
               LowerCarryBinaryOpPattern<pto::VsubcsOp>>(
      typeConverter, patterns.getContext(), state);
}


} // namespace mlir::pto
