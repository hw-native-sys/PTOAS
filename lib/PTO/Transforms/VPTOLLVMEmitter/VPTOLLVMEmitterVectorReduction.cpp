// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

static FailureOr<StringRef> buildLaneTypedCalleeFromInput(MLIRContext *context,
                                                          Type inputType,
                                                          StringRef stem,
                                                          StringRef suffix) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  auto lanes = getElementCountFromVectorLike(inputType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec +
                                      suffix.str())
      .getValue();
}

static FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context,
                                                 Type resultType,
                                                 StringRef stem,
                                                 StringRef suffix) {
  return buildLaneTypedCalleeFromInput(context, resultType, stem, suffix);
}

template <typename VecScalarOp>
static StringRef getVecScalarMaskedStem() {
  if constexpr (std::is_same_v<VecScalarOp, pto::VmulsOp>)
  {
    return "vmuls";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VaddsOp>)
  {
    return "vadds";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VmaxsOp>)
  {
    return "vmaxs";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VminsOp>)
  {
    return "vmins";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VlreluOp>)
  {
    return "vlrelu";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VshlsOp>)
  {
    return "vshls";
  }
  if constexpr (std::is_same_v<VecScalarOp, pto::VshrsOp>)
  {
    return "vshrs";
  }
  return {};
}

template <typename ReductionOp>
static StringRef getReductionUnaryStem() {
  if constexpr (std::is_same_v<ReductionOp, pto::VcaddOp>)
  {
    return "vcadd";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcmaxOp>)
  {
    return "vcmax";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcminOp>)
  {
    return "vcmin";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgaddOp>)
  {
    return "vcgadd";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgmaxOp>)
  {
    return "vcgmax";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcgminOp>)
  {
    return "vcgmin";
  }
  if constexpr (std::is_same_v<ReductionOp, pto::VcpaddOp>)
  {
    return "vcpadd";
  }
  return {};
}

template <typename HistOp>
static StringRef getHistogramCallee(MLIRContext *context) {
  if constexpr (std::is_same_v<HistOp, pto::Chistv2Op>)
  {
    return StringAttr::get(context, "llvm.hivm.chistv2.m").getValue();
  }
  if constexpr (std::is_same_v<HistOp, pto::Dhistv2Op>)
  {
    return StringAttr::get(context, "llvm.hivm.dhistv2.m").getValue();
  }
  return {};
}

template <typename ExtremaOp>
static StringRef getExtremaPredicateStem() {
  if constexpr (std::is_same_v<ExtremaOp, pto::VcbmaxOp>)
  {
    return "vcbmax";
  }
  if constexpr (std::is_same_v<ExtremaOp, pto::VcbminOp>)
  {
    return "vcbmin";
  }
  return {};
}

template <typename ExtremaOp>
static FailureOr<StringRef> buildExtremaPredicateCallee(MLIRContext *context,
                                                        Type resultType) {
  return buildLaneTypedCallee(context, resultType,
                              getExtremaPredicateStem<ExtremaOp>(), ".x");
}

template <typename VecScalarOp>
class LowerVecScalarMaskedOpPattern final
    : public OpConversionPattern<VecScalarOp> {
public:
  explicit LowerVecScalarMaskedOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<VecScalarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(VecScalarOp op, typename VecScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getVecScalarMaskedStem<VecScalarOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported vec-scalar VPTO signature");
    }

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert vec-scalar result type");
    }

    Value input = adaptor.getOperands()[0];
    Value scalar = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!input || !scalar || !mask || input.getType() != resultType ||
        mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vec-scalar VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{input, scalar, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp>
class LowerReductionUnaryOpPattern final
    : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerReductionUnaryOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getReductionUnaryStem<ReductionOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported reduction VPTO signature");
    }

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert reduction result type");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename HistOp>
class LowerHistogramOpPattern final : public OpConversionPattern<HistOp> {
public:
  explicit LowerHistogramOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<HistOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(HistOp op, typename HistOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = getHistogramCallee<HistOp>(op.getContext());
    if (calleeName.empty())
    {
      return rewriter.notifyMatchFailure(op, "unsupported histogram op");
    }

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type sourceType =
        this->getTypeConverter()->convertType(op.getSource().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !sourceType || !maskType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert histogram types");
    }

    Value acc = adaptor.getAcc();
    Value source = adaptor.getSource();
    Value mask = adaptor.getMask();
    Value bin = adaptor.getBin();
    if (!acc || !source || !mask || !bin || acc.getType() != resultType ||
        source.getType() != sourceType || mask.getType() != maskType ||
        !bin.getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted histogram operand types");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, sourceType, maskType, rewriter.getI32Type()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{resultType},
        ValueRange{acc, source, mask, bin});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ExtremaOp>
class LowerExtremaPredicateOpPattern final
    : public OpConversionPattern<ExtremaOp> {
public:
  explicit LowerExtremaPredicateOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<ExtremaOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ExtremaOp op, typename ExtremaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildExtremaPredicateCallee<ExtremaOp>(op.getContext(),
                                               op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported extrema-predicate VPTO signature");
    }

    Type valueType =
        this->getTypeConverter()->convertType(op.getValue().getType());
    Type predicateType =
        this->getTypeConverter()->convertType(op.getPredicate().getType());
    if (!valueType || !predicateType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert extrema-predicate result types");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != valueType ||
        mask.getType() != predicateType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted extrema-predicate operand types");
    }

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{valueType, predicateType},
        ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp>
class LowerWideningReductionUnaryOpPattern final
    : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerWideningReductionUnaryOpPattern(TypeConverter &typeConverter,
                                                MLIRContext *context,
                                                LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildLaneTypedCalleeFromInput(
        op.getContext(), op.getInput().getType(),
        getReductionUnaryStem<ReductionOp>(), ".x");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported widening reduction VPTO signature");
    }

    Type inputType =
        this->getTypeConverter()->convertType(op.getInput().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!inputType || !resultType || !maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert widening reduction types");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != inputType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted widening reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};


} // namespace

void populateVPTOVectorReductionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         LoweringState &state) {
  patterns.add<LowerVecScalarMaskedOpPattern<pto::VmulsOp>,
               LowerVecScalarMaskedOpPattern<pto::VaddsOp>,
               LowerVecScalarMaskedOpPattern<pto::VmaxsOp>,
               LowerVecScalarMaskedOpPattern<pto::VminsOp>,
               LowerVecScalarMaskedOpPattern<pto::VlreluOp>,
               LowerVecScalarMaskedOpPattern<pto::VshlsOp>,
               LowerVecScalarMaskedOpPattern<pto::VshrsOp>,
               LowerWideningReductionUnaryOpPattern<pto::VcaddOp>,
               LowerReductionUnaryOpPattern<pto::VcmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcminOp>,
               LowerReductionUnaryOpPattern<pto::VcgaddOp>,
               LowerReductionUnaryOpPattern<pto::VcgmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcgminOp>,
               LowerReductionUnaryOpPattern<pto::VcpaddOp>,
               LowerHistogramOpPattern<pto::Chistv2Op>,
               LowerHistogramOpPattern<pto::Dhistv2Op>,
               LowerExtremaPredicateOpPattern<pto::VcbmaxOp>,
               LowerExtremaPredicateOpPattern<pto::VcbminOp>>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
