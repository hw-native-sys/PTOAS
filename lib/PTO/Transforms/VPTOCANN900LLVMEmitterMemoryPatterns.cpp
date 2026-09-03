// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterTemplates.h"

namespace mlir::pto::detail {

template <typename UnpackOp> class LowerUnpackOpPattern final : public OpConversionPattern<UnpackOp> {
public:
  explicit LowerUnpackOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnpackOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(UnpackOp op, typename UnpackOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<UnpackOp, pto::VsunpackOp> ? "vsunpack" : "vzunpack";
    FailureOr<StringRef> calleeName =
        buildUnpackCallee(op.getContext(), op.getSrc().getType(), op.getResult().getType(), stem);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported unpack VPTO signature");
    }

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert unpack types");
    }

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted unpack source type");
    }

    Value part = castIntegerLikeTo(op, adaptor.getPart(), rewriter.getI32Type());
    if (!part) {
      return rewriter.notifyMatchFailure(op, "failed to materialize unpack part");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{srcType, part.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, part});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVpackOpPattern final : public OpConversionPattern<pto::VpackOp> {
public:
  explicit LowerVpackOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VpackOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VpackOp op, pto::VpackOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVpackCallee(op.getContext(), op.getSrc().getType(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vpack VPTO signature");
    }

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vpack types");
    }

    auto partImm = parseHiLoPartImmediate(op.getPart());
    if (!partImm) {
      return rewriter.notifyMatchFailure(op, "unsupported vpack part immediate");
    }

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vpack source type");
    }

    Value part = getI32Constant(rewriter, op.getLoc(), *partImm);
    auto funcType = rewriter.getFunctionType(TypeRange{srcType, part.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, part});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PredicateMaskOp>
class LowerPredicateMaskBinaryOpPattern final : public OpConversionPattern<PredicateMaskOp> {
public:
  explicit LowerPredicateMaskBinaryOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<PredicateMaskOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(PredicateMaskOp op, typename PredicateMaskOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert predicate-mask result type");
    }

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType || src1.getType() != resultType ||
        mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted predicate-mask operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<PredicateMaskOp>(op.getContext());
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReorderOp> class LowerPredicatePairReorderOpPattern final : public OpConversionPattern<ReorderOp> {
public:
  explicit LowerPredicatePairReorderOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ReorderOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ReorderOp op, typename ReorderOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert predicate-pair-reorder result types");
    }
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op, "unexpected predicate-pair-reorder converted result types");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != resultTypes[0] || rhs.getType() != resultTypes[0]) {
      return rewriter.notifyMatchFailure(op, "unexpected converted predicate-pair-reorder operand types");
    }

    StringRef calleeName = buildPredicatePairReorderCallee<ReorderOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, ValueRange{lhs, rhs});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CmpOp> class LowerCmpOpPattern final : public OpConversionPattern<CmpOp> {
public:
  explicit LowerCmpOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CmpOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(CmpOp op, typename CmpOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    constexpr bool isScalarCompare = std::is_same_v<CmpOp, pto::VcmpsOp>;
    Type inputType = Type();
    if constexpr (isScalarCompare) {
      inputType = op.getSrc().getType();
    } else {
      inputType = op.getSrc0().getType();
    }
    FailureOr<StringRef> calleeName = buildVcmpCallee(op.getContext(), inputType, op.getCmpMode(), isScalarCompare);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported compare VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert compare result type");
    }
    if (resultType != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected compare mask conversion");
    }

    SmallVector<Value> callArgs;
    callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    if constexpr (isScalarCompare) {
      if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] || callArgs[2].getType() != maskType) {
        return rewriter.notifyMatchFailure(op, "unexpected converted scalar-compare operand types");
      }
      callArgs[1] = normalizeByteScalarOperandForCANN900VectorCall(
          rewriter, op.getLoc(), callArgs[1], cast<pto::VRegType>(op.getSrc().getType()).getElementType());
    } else {
      if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
          callArgs[0].getType() != callArgs[1].getType() || callArgs[2].getType() != maskType) {
        return rewriter.notifyMatchFailure(op, "unexpected converted compare operand types");
      }
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PltOp> class LowerPltOpPattern final : public OpConversionPattern<PltOp> {
public:
  explicit LowerPltOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<PltOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(PltOp op, typename PltOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value laneCount = castIntegerLikeTo(op, adaptor.getScalar(), rewriter.getI32Type());
    if (!laneCount) {
      return rewriter.notifyMatchFailure(op, "failed to materialize plt lane count");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert plt result types");
    }

    StringRef calleeName = buildPltCallee<PltOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, ValueRange{laneCount});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PltmOp> class LowerPltmOpPattern final : public OpConversionPattern<PltmOp> {
public:
  explicit LowerPltmOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<PltmOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(PltmOp op, typename PltmOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert pltm result type");
    }

    Value loop = adaptor.getLoop();
    Value bound = adaptor.getBound();
    if (!loop || !bound || !loop.getType().isInteger(16) || !bound.getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(op, "unexpected converted pltm operand types");
    }

    StringRef calleeName = buildPltmCallee<PltmOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI16Type(), rewriter.getI32Type()}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, ValueRange{loop, bound});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PsetOp> class LowerPsetOpPattern final : public OpConversionPattern<PsetOp> {
public:
  explicit LowerPsetOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<PsetOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(PsetOp op, typename PsetOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern) {
      return rewriter.notifyMatchFailure(op, "unsupported pset pattern");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert pset result types");
    }

    if (isMaskOnlyUsedByOnePointStores(op.getResult())) {
      auto undef = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultTypes.front());
      rewriter.replaceOp(op, undef.getResult());
      return success();
    }

    StringRef calleeName = buildPsetCallee<PsetOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, ValueRange{patternValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PgeOp> class LowerPgeOpPattern final : public OpConversionPattern<PgeOp> {
public:
  explicit LowerPgeOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<PgeOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(PgeOp op, typename PgeOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern) {
      return rewriter.notifyMatchFailure(op, "unsupported pge pattern");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert pge result types");
    }

    if (isMaskOnlyUsedByOnePointStores(op.getResult())) {
      auto undef = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultTypes.front());
      rewriter.replaceOp(op, undef.getResult());
      return success();
    }

    StringRef calleeName = buildPgeCallee<PgeOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(0));
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type(), rewriter.getI32Type()}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, ValueRange{patternValue, zero});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

static SmallVector<Type> getVldsCallResultTypes(Type ptoResultType, ArrayRef<Type> resultTypes, bool usePostIntrinsic,
                                                MLIRContext *context) {
  SmallVector<Type> callResultTypes{getPayloadABIType(ptoResultType, resultTypes[0], context)};
  if (usePostIntrinsic) {
    callResultTypes.push_back(resultTypes[1]);
  }
  return callResultTypes;
}

static SmallVector<Value> getVldsReplacements(pto::VldsOp op, const VPTOLoweredAddressOffset &offset, func::CallOp call,
                                              ArrayRef<Type> resultTypes, ConversionPatternRewriter &rewriter) {
  Value loaded = castFromPayloadABI(op.getLoc(), call.getResult(0), op.getResult().getType(), resultTypes[0], rewriter);
  SmallVector<Value> replacements{loaded};
  if (op.getUpdatedBase()) {
    replacements.push_back(offset.updatedBase ? offset.updatedBase : call.getResult(1));
  }
  return replacements;
}

static SmallVector<Value> getVldsx2Replacements(pto::Vldsx2Op op, const VPTOLoweredAddressOffset &offset,
                                                func::CallOp call, ArrayRef<Type> resultTypes,
                                                ConversionPatternRewriter &rewriter) {
  Value low = castFromPayloadABI(op.getLoc(), call.getResult(0), op.getLow().getType(), resultTypes[0], rewriter);
  Value high = castFromPayloadABI(op.getLoc(), call.getResult(1), op.getHigh().getType(), resultTypes[1], rewriter);
  SmallVector<Value> replacements{low, high};
  if (op.getUpdatedBase()) {
    replacements.push_back(offset.updatedBase ? offset.updatedBase : call.getResult(2));
  }
  return replacements;
}

class LowerVldsOpPattern final : public OpConversionPattern<pto::VldsOp> {
public:
  explicit LowerVldsOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldsOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VldsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type ptoResultType = op.getResult().getType();
    Type elementType = getElementTypeFromVectorLike(ptoResultType);
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "unsupported vlds element type");
    }
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(op, adaptor.getSource(), adaptor.getOffset(), elementType,
                                                            usePostIntrinsic, rewriter);
    auto dist = parseLoadDistImmediate(op.getDist().value_or("NORM"), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vlds operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vlds result types");
    }

    if (usePostIntrinsic) {
      if (resultTypes.size() != 2 || resultTypes[1] != adaptor.getSource().getType()) {
        return rewriter.notifyMatchFailure(op, "unsupported vlds post-update results");
      }
    } else if (resultTypes.size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported vlds result count");
    }
    SmallVector<Type> callResultTypes =
        getVldsCallResultTypes(ptoResultType, resultTypes, usePostIntrinsic, rewriter.getContext());

    FailureOr<StringRef> calleeName = usePostIntrinsic ? buildVldsPostCallee(op.getContext(), ptoResultType)
                                                       : buildVldsCallee(op.getContext(), ptoResultType);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vlds signature");
    }

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{loweredOffset->base, loweredOffset->intrinsicOffset, distValue, postValue};
    auto funcType =
        rewriter.getFunctionType(TypeRange{loweredOffset->base.getType(), loweredOffset->intrinsicOffset.getType(),
                                           distValue.getType(), postValue.getType()},
                                 callResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, callResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, getVldsReplacements(op, *loweredOffset, call, resultTypes, rewriter));
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsx2OpPattern final : public OpConversionPattern<pto::Vldsx2Op> {
public:
  explicit LowerVldsx2OpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vldsx2Op>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::Vldsx2Op op, pto::Vldsx2Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 element type");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(op, adaptor.getSource(), adaptor.getOffset(), elementType,
                                                            usePostIntrinsic, rewriter);
    auto dist = parseLoadX2DistImmediate(op.getDist(), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vldsx2 operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 3U : 2U)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vldsx2 result types");
    }
    Type lowCallType = getPayloadABIType(op.getLow().getType(), resultTypes[0], rewriter.getContext());
    Type highCallType = getPayloadABIType(op.getHigh().getType(), resultTypes[1], rewriter.getContext());
    SmallVector<Type> callResultTypes{lowCallType, highCallType};
    if (usePostIntrinsic) {
      callResultTypes.push_back(resultTypes[2]);
    }

    FailureOr<StringRef> calleeName = buildVldsx2Callee(op.getContext(), op.getLow().getType(), usePostIntrinsic);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 signature");
    }

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{loweredOffset->base, loweredOffset->intrinsicOffset, distValue, postValue};
    auto funcType =
        rewriter.getFunctionType(TypeRange{loweredOffset->base.getType(), loweredOffset->intrinsicOffset.getType(),
                                           distValue.getType(), postValue.getType()},
                                 callResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, callResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, getVldsx2Replacements(op, *loweredOffset, call, resultTypes, rewriter));
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsldbOpPattern final : public OpConversionPattern<pto::VsldbOp> {
public:
  explicit LowerVsldbOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsldbOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VsldbOp op, pto::VsldbOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Value packedStride = packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vsldb operands");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 2U : 1U)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsldb result type");
    }

    Type callResultType = getPayloadABIType(op.getResult().getType(), resultTypes[0], rewriter.getContext());
    SmallVector<Type> callResultTypes{callResultType};
    if (usePostIntrinsic) {
      callResultTypes.push_back(resultTypes[1]);
    }

    FailureOr<StringRef> calleeName = buildVsldbCallee(op.getContext(), op.getResult().getType(), usePostIntrinsic);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vsldb signature");
    }
    Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{adaptor.getSource(), packedStride, postValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getSource().getType(), packedStride.getType(),
                                                       postValue.getType(), adaptor.getMask().getType()},
                                             callResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, callResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value result =
        castFromPayloadABI(op.getLoc(), call.getResult(0), op.getResult().getType(), resultTypes[0], rewriter);
    if (usePostIntrinsic) {
      rewriter.replaceOp(op, ValueRange{result, call.getResult(1)});
    } else {
      rewriter.replaceOp(op, ValueRange{result});
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerInitAlignOpPattern final : public OpConversionPattern<pto::InitAlignOp> {
public:
  explicit LowerInitAlignOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::InitAlignOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::InitAlignOp op, pto::InitAlignOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert init_align result type");
    }

    StringRef calleeName = buildInitAlignCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldasOpPattern final : public OpConversionPattern<pto::VldasOp> {
public:
  explicit LowerVldasOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldasOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VldasOp op, pto::VldasOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!sourceType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected converted vldas operand/result types");
    }

    StringRef calleeName = buildVldasCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getSource().getType()}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{adaptor.getSource()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

struct VldusCallOperands {
  SmallVector<Value> args;
  Value explicitUpdatedBase;
};

static FailureOr<VldusCallOperands> buildVldusCallOperands(pto::VldusOp op, pto::VldusOp::Adaptor adaptor,
                                                           ConversionPatternRewriter &rewriter) {
  SmallVector<Value> args{adaptor.getSource(), adaptor.getAlign()};
  if (!op.getUpdatedBase()) {
    return VldusCallOperands{std::move(args), Value()};
  }
  Type elementType = getElementTypeFromVectorLike(op.getResult().getType());
  auto loweredIncrement =
      lowerVPTOElementOffsetForIntrinsic(op, adaptor.getSource(), adaptor.getIncrement(), elementType,
                                         /*isPostUpdate=*/true, rewriter);
  if (failed(loweredIncrement)) {
    return failure();
  }
  args.front() = loweredIncrement->base;
  args.push_back(loweredIncrement->intrinsicOffset);
  return VldusCallOperands{std::move(args), loweredIncrement->updatedBase};
}

class LowerVldusOpPattern final : public OpConversionPattern<pto::VldusOp> {
public:
  explicit LowerVldusOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldusOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VldusOp op, pto::VldusOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    SmallVector<Type> resultTypes;
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (!sourceType || failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 3U : 2U) || adaptor.getAlign().getType() != resultTypes[1] ||
        (usePostIntrinsic && resultTypes[2] != adaptor.getSource().getType())) {
      return rewriter.notifyMatchFailure(op, "expected converted vldus operand/result types");
    }

    FailureOr<StringRef> calleeName = usePostIntrinsic ? buildVldusPostCallee(op.getContext(), op.getResult().getType())
                                                       : buildVldusCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vldus signature");
    }

    Type callValueType = getPayloadABIType(op.getResult().getType(), resultTypes[0], rewriter.getContext());
    SmallVector<Type> intrinsicResultTypes{callValueType, resultTypes[1]};
    // The installed no-post A5 vldus intrinsic returns an extra hidden base ptr.
    intrinsicResultTypes.push_back(adaptor.getSource().getType());

    FailureOr<VldusCallOperands> callOperands = buildVldusCallOperands(op, adaptor, rewriter);
    if (failed(callOperands)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vldus increment");
    }
    SmallVector<Type> argTypes;
    for (Value arg : callOperands->args) {
      argTypes.push_back(arg.getType());
    }
    auto funcType = rewriter.getFunctionType(argTypes, intrinsicResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, intrinsicResultTypes, callOperands->args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value loaded =
        castFromPayloadABI(op.getLoc(), call.getResult(0), op.getResult().getType(), resultTypes[0], rewriter);
    SmallVector<Value> replacements{loaded, call.getResult(1)};
    if (usePostIntrinsic) {
      replacements.push_back(callOperands->explicitUpdatedBase ? callOperands->explicitUpdatedBase : call.getResult(2));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }

private:
  LoweringState &state;
};

class LowerSprclrOpPattern final : public OpConversionPattern<pto::SprclrOp> {
public:
  explicit LowerSprclrOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::SprclrOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::SprclrOp op, pto::SprclrOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto spr = parseSprImmediate(op.getSpr());
    if (!spr) {
      return rewriter.notifyMatchFailure(op, "unsupported sprclr target");
    }

    StringRef calleeName = buildSprclrCallee(op.getContext());
    Value sprValue = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI16IntegerAttr(*spr));
    auto funcType = rewriter.getFunctionType(TypeRange{sprValue.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{sprValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SprStoreOp> class LowerSprStoreOpPattern final : public OpConversionPattern<SprStoreOp> {
public:
  explicit LowerSprStoreOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<SprStoreOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(SprStoreOp op, typename SprStoreOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto spr = parseSprImmediate(op.getSpr());
    if (!spr) {
      return rewriter.notifyMatchFailure(op, "unsupported spr store target");
    }
    auto destType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    if (!destType || !adaptor.getOffset().getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(op, "expected converted spr store operands");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 1U : 0U)) {
      return rewriter.notifyMatchFailure(op, "failed to convert spr store result types");
    }

    StringRef calleeName = buildSprStoreCallee<SprStoreOp>(op.getContext(), usePostIntrinsic);
    Value sprValue = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI16IntegerAttr(*spr));
    Value postValue =
        rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(usePostIntrinsic ? 1 : 0));
    SmallVector<Value> args{sprValue, adaptor.getDestination(), adaptor.getOffset(), postValue};
    auto funcType = rewriter.getFunctionType(TypeRange{sprValue.getType(), adaptor.getDestination().getType(),
                                                       adaptor.getOffset().getType(), postValue.getType()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (usePostIntrinsic) {
      rewriter.replaceOp(op, call.getResults());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  LoweringState &state;
};

static Type getVPTOAddressElementType(Type addressType, Type fallbackType) {
  if (auto ptrType = dyn_cast<pto::PtrType>(addressType)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(addressType)) {
    return memrefType.getElementType();
  }
  return fallbackType;
}

static SmallVector<Value> getVstsCallArgs(pto::VstsOp op, pto::VstsOp::Adaptor adaptor,
                                          const VPTOLoweredAddressOffset &offset, uint64_t dist, bool usePostIntrinsic,
                                          ConversionPatternRewriter &rewriter) {
  Value value = castToPayloadABI(op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
  Value mask = adaptor.getMask();
  if (isOnePointStoreDist(op.getDist().value_or(""))) {
    mask = rewriter.create<LLVM::UndefOp>(op.getLoc(), mask.getType());
  }
  Value distValue = getI32Constant(rewriter, op.getLoc(), dist);
  Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
  return {value, offset.base, offset.intrinsicOffset, distValue, postValue, mask};
}

static LogicalResult replaceVstsOp(pto::VstsOp op, bool usePostIntrinsic, const VPTOLoweredAddressOffset &offset,
                                   func::CallOp call, ConversionPatternRewriter &rewriter) {
  if (!usePostIntrinsic) {
    rewriter.eraseOp(op);
    return success();
  }
  rewriter.replaceOp(op, offset.updatedBase ? offset.updatedBase : call.getResult(0));
  return success();
}

class LowerVstsOpPattern final : public OpConversionPattern<pto::VstsOp> {
public:
  explicit LowerVstsOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VstsOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VstsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "unsupported vsts element type");
    }
    Type offsetElementType = getVPTOAddressElementType(op.getDestination().getType(), elementType);
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(op, adaptor.getDestination(), adaptor.getOffset(),
                                                            offsetElementType, usePostIntrinsic, rewriter);
    auto dist = parseStoreDistImmediate(op.getDist().value_or(""), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vsts operands");
    }

    FailureOr<StringRef> calleeName = op.getUpdatedBase()
                                          ? buildVstsPostCallee(op.getContext(), op.getValue().getType())
                                          : buildVstsCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vsts signature");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsts result types");
    }
    if (usePostIntrinsic) {
      if (resultTypes.size() != 1 || resultTypes[0] != adaptor.getDestination().getType()) {
        return rewriter.notifyMatchFailure(op, "unsupported vsts post-update result");
      }
    } else if (!resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported vsts result count");
    }

    SmallVector<Value> args = getVstsCallArgs(op, adaptor, *loweredOffset, *dist, usePostIntrinsic, rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{args[0].getType(), loweredOffset->base.getType(), rewriter.getI32Type(),
                                           rewriter.getI32Type(), rewriter.getI32Type(), args[5].getType()},
                                 resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    return replaceVstsOp(op, usePostIntrinsic, *loweredOffset, call, rewriter);
  }

private:
  LoweringState &state;
};

class LowerVsstbOpPattern final : public OpConversionPattern<pto::VsstbOp> {
public:
  explicit LowerVsstbOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsstbOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VsstbOp op, pto::VsstbOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Value packedStride = packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vsstb operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsstb result types");
    }
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (usePostIntrinsic) {
      if (resultTypes.size() != 1 || resultTypes[0] != adaptor.getDestination().getType()) {
        return rewriter.notifyMatchFailure(op, "unsupported vsstb post-update result");
      }
    } else if (!resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported vsstb result count");
    }

    FailureOr<StringRef> calleeName = buildVsstbCallee(op.getContext(), op.getValue().getType(), usePostIntrinsic);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vsstb signature");
    }
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    Value value = castToPayloadABI(op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    SmallVector<Value> args{value, adaptor.getDestination(), packedStride, zeroValue, adaptor.getMask()};
    auto funcType =
        rewriter.getFunctionType(TypeRange{value.getType(), adaptor.getDestination().getType(), packedStride.getType(),
                                           zeroValue.getType(), adaptor.getMask().getType()},
                                 resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    if (usePostIntrinsic) {
      rewriter.replaceOp(op, call.getResults());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsx2OpPattern final : public OpConversionPattern<pto::Vstsx2Op> {
public:
  explicit LowerVstsx2OpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vstsx2Op>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::Vstsx2Op op, pto::Vstsx2Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 element type");
    }

    auto loweredOffset =
        lowerVPTOElementOffsetForIntrinsic(op, adaptor.getDestination(), adaptor.getOffset(), elementType,
                                           /*isPostUpdate=*/false, rewriter);
    auto dist = parseStoreX2DistImmediate(op.getDist(), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vstsx2 operands");
    }

    FailureOr<StringRef> calleeName = buildVstsx2Callee(op.getContext(), op.getLow().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 signature");
    }

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    Value low = castToPayloadABI(op.getLoc(), adaptor.getLow(), op.getLow().getType(), rewriter);
    Value high = castToPayloadABI(op.getLoc(), adaptor.getHigh(), op.getHigh().getType(), rewriter);
    SmallVector<Value> args{low,       high,      loweredOffset->base, loweredOffset->intrinsicOffset,
                            distValue, zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(TypeRange{low.getType(), high.getType(), loweredOffset->base.getType(),
                                                       loweredOffset->intrinsicOffset.getType(), distValue.getType(),
                                                       zeroValue.getType(), adaptor.getMask().getType()},
                                             TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerPstuOpPattern final : public OpConversionPattern<pto::PstuOp> {
public:
  explicit LowerPstuOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PstuOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PstuOp op, pto::PstuOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildPstuCallee(op.getContext(), op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported pstu signature");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert pstu result types");
    }
    if (resultTypes.size() != 2) {
      return rewriter.notifyMatchFailure(op, "unexpected converted pstu result arity");
    }

    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!baseType || adaptor.getAlignIn().getType() != resultTypes[0] ||
        adaptor.getBase().getType() != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op, "unexpected converted pstu operand/result types");
    }

    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(), adaptor.getAlignIn().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstusOpPattern final : public OpConversionPattern<pto::VstusOp> {
public:
  explicit LowerVstusOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VstusOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VstusOp op, pto::VstusOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "unsupported vstus element type");
    }

    bool usePostIntrinsic = static_cast<bool>(op.getBaseOut());
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(op, adaptor.getBase(), adaptor.getOffset(), elementType,
                                                            usePostIntrinsic, rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vstus offset");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vstus result types");
    }
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!baseType || resultTypes.size() != (usePostIntrinsic ? 2U : 1U) ||
        adaptor.getAlignIn().getType() != resultTypes[0] ||
        (usePostIntrinsic && resultTypes[1] != adaptor.getBase().getType())) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vstus operand/result types");
    }

    FailureOr<StringRef> calleeName = buildVstusCallee(op.getContext(), op.getValue().getType());
    if (usePostIntrinsic) {
      calleeName = buildVstusPostCallee(op.getContext(), op.getValue().getType());
    }
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vstus signature");
    }
    Value value = castToPayloadABI(op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    SmallVector<Value> args{value, loweredOffset->base, loweredOffset->intrinsicOffset, adaptor.getAlignIn()};
    auto funcType =
        rewriter.getFunctionType(TypeRange{value.getType(), loweredOffset->base.getType(),
                                           loweredOffset->intrinsicOffset.getType(), adaptor.getAlignIn().getType()},
                                 resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    if (usePostIntrinsic && loweredOffset->updatedBase) {
      rewriter.replaceOp(op, ValueRange{call.getResult(0), loweredOffset->updatedBase});
    } else {
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsturOpPattern final : public OpConversionPattern<pto::VsturOp> {
public:
  explicit LowerVsturOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsturOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VsturOp op, pto::VsturOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto postMode = parsePostModeImmediate(op.getMode());
    if (!postMode) {
      return rewriter.notifyMatchFailure(op, "unsupported vstur mode immediate");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getAlignOut().getType());
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!resultType || !baseType || adaptor.getAlignIn().getType() != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vstur operand/result types");
    }

    StringRef calleeName = buildVsturCallee(op.getContext());
    Value modeValue = getI32Constant(rewriter, op.getLoc(), *postMode);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    Value value = castToPayloadABI(op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    SmallVector<Value> args{value, adaptor.getBase(), adaptor.getAlignIn(), modeValue, zeroValue};
    auto funcType =
        rewriter.getFunctionType(TypeRange{value.getType(), adaptor.getBase().getType(), adaptor.getAlignIn().getType(),
                                           modeValue.getType(), zeroValue.getType()},
                                 TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstarOpPattern final : public OpConversionPattern<pto::VstarOp> {
public:
  explicit LowerVstarOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VstarOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VstarOp op, pto::VstarOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vstar operand types");
    }

    StringRef calleeName = buildVstarCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(), zeroValue.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstasOpPattern final : public OpConversionPattern<pto::VstasOp> {
public:
  explicit LowerVstasOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VstasOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VstasOp op, pto::VstasOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    auto dstType = dyn_cast<pto::PtrType>(op.getDestination().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType || !dstType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vstas operand types");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(op, adaptor.getDestination(), adaptor.getOffset(),
                                                            dstType.getElementType(), usePostIntrinsic, rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vstas offset");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 1U : 0U)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vstas result types");
    }

    StringRef calleeName = buildVstasCallee(op.getContext(), usePostIntrinsic);
    Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{adaptor.getValue(), loweredOffset->base, loweredOffset->intrinsicOffset, postValue};
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getValue().getType(), loweredOffset->base.getType(),
                                                       loweredOffset->intrinsicOffset.getType(), postValue.getType()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (usePostIntrinsic) {
      Value updatedBase = loweredOffset->updatedBase ? loweredOffset->updatedBase : call.getResult(0);
      rewriter.replaceOp(op, updatedBase);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2OpPattern final : public OpConversionPattern<pto::Vgather2Op> {
public:
  explicit LowerVgather2OpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2Op>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::Vgather2Op op, pto::Vgather2Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    if (!elemType || !basePtr) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vgather2 operand types");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vgather2 result type");
    }

    FailureOr<StringRef> calleeName =
        buildVgather2Callee(op.getContext(), op.getSource().getType(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vgather2 signature");
    }

    Value offsets = adaptor.getOffsets();
    FailureOr<Type> offsetsCarrierType =
        getVgather2OffsetsCarrierType(rewriter, op.getSource().getType(), op.getResult().getType(), offsets.getType());
    if (failed(offsetsCarrierType)) {
      return rewriter.notifyMatchFailure(op, "unsupported vgather2 offsets carrier");
    }
    if (offsets.getType() != *offsetsCarrierType) {
      offsets = rewriter.create<LLVM::BitcastOp>(op.getLoc(), *offsetsCarrierType, offsets);
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), *offsetsCarrierType, adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getSource(), offsets, adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2BcOpPattern final : public OpConversionPattern<pto::Vgather2BcOp> {
public:
  explicit LowerVgather2BcOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2BcOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::Vgather2BcOp op, pto::Vgather2BcOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vgather2_bc operand/result types");
    }

    FailureOr<StringRef> calleeName = buildVgather2BcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vgather2_bc signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgatherbOpPattern final : public OpConversionPattern<pto::VgatherbOp> {
public:
  explicit LowerVgatherbOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VgatherbOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VgatherbOp op, pto::VgatherbOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vgatherb operand/result types");
    }

    FailureOr<StringRef> calleeName = buildVgatherbCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vgatherb signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVscatterOpPattern final : public OpConversionPattern<pto::VscatterOp> {
public:
  explicit LowerVscatterOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VscatterOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VscatterOp op, pto::VscatterOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getValue().getType());
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    if (!elemType || !basePtr) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vscatter operand types");
    }

    FailureOr<StringRef> calleeName = buildVscatterCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vscatter signature");
    }

    FailureOr<Type> offsetsCarrierType = getVscatterOffsetsCarrierType(adaptor.getOffsets().getType());
    if (failed(offsetsCarrierType)) {
      return rewriter.notifyMatchFailure(op, "unsupported vscatter offsets carrier");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                                                       *offsetsCarrierType, adaptor.getMask().getType()},
                                             TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getValue(), adaptor.getDestination(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVaxpyOpPattern final : public OpConversionPattern<pto::VaxpyOp> {
public:
  explicit LowerVaxpyOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VaxpyOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VaxpyOp op, pto::VaxpyOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!elemType) {
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vaxpy result type");
    }

    FailureOr<StringRef> calleeName = buildVaxpyCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy callee");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getSrc1().getType(), adaptor.getSrc0().getType(),
                                                       adaptor.getAlpha().getType(), adaptor.getMask().getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSrc1(), adaptor.getSrc0(), adaptor.getAlpha(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmulscvtOpPattern final : public OpConversionPattern<pto::VmulscvtOp> {
public:
  explicit LowerVmulscvtOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VmulscvtOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VmulscvtOp op, pto::VmulscvtOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto roundMode = parseRoundModeImmediate(op.getRnd());
    if (!roundMode) {
      return rewriter.notifyMatchFailure(op, "vmulscvt requires valid rnd attr");
    }
    if (*roundMode != 1) {
      return rewriter.notifyMatchFailure(op, "current vmulscvt lowering only supports rnd A");
    }

    auto part = parsePartImmediate(op.getPart());
    if (!part) {
      return rewriter.notifyMatchFailure(op, "unsupported vmulscvt part");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vmulscvt result type");
    }

    FailureOr<StringRef> calleeName =
        buildVmulscvtCallee(op.getContext(), op.getInput().getType(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vmulscvt signature");
    }

    Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getInput().getType(), adaptor.getScalar().getType(),
                                                       adaptor.getMask().getType(), partValue.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getScalar(), adaptor.getMask(), partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVciOpPattern final : public OpConversionPattern<pto::VciOp> {
public:
  explicit LowerVciOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VciOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VciOp op, pto::VciOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto order = parseOrderImmediate(op.getOrder().value_or("ASC"));
    if (!order) {
      return rewriter.notifyMatchFailure(op, "unsupported vci order");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vci result type");
    }

    FailureOr<StringRef> calleeName = buildVciCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vci callee");
    }

    Value indexValue = adaptor.getIndex();

    Value orderValue = getI32Constant(rewriter, op.getLoc(), *order);
    auto funcType =
        rewriter.getFunctionType(TypeRange{indexValue.getType(), orderValue.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{indexValue, orderValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVexpdifOpPattern final : public OpConversionPattern<pto::VexpdifOp> {
public:
  explicit LowerVexpdifOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VexpdifOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VexpdifOp op, pto::VexpdifOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto part = parsePartImmediate(op.getPart());
    if (!part) {
      return rewriter.notifyMatchFailure(op, "unsupported vexpdif signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vexpdif result type");
    }

    FailureOr<StringRef> calleeName =
        buildVexpdifCallee(op.getContext(), op.getInput().getType(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vexpdif callee");
    }

    Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getInput().getType(), adaptor.getMax().getType(),
                                                       adaptor.getMask().getType(), partValue.getType()},
                                             TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                      ValueRange{adaptor.getInput(), adaptor.getMax(), adaptor.getMask(), partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbitsortOpPattern final : public OpConversionPattern<pto::VbitsortOp> {
public:
  explicit LowerVbitsortOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbitsortOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VbitsortOp op, pto::VbitsortOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto srcType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto idxType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getIndices().getType());
    if (!dstType || !srcType || !idxType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vbitsort operand types");
    }

    FailureOr<Value> config = packVbitsortConfig(op, adaptor.getRepeatTimes());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to pack vbitsort config");
    }

    FailureOr<StringRef> calleeName = buildVbitsortCallee(op.getContext(), op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vbitsort signature");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getDestination().getType(), adaptor.getSource().getType(),
                                           adaptor.getIndices().getType(), (*config).getType()},
                                 TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getDestination(), adaptor.getSource(), adaptor.getIndices(), *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmrgsort4OpPattern final : public OpConversionPattern<pto::Vmrgsort4Op> {
public:
  explicit LowerVmrgsort4OpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vmrgsort4Op>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::Vmrgsort4Op op, pto::Vmrgsort4Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto src0Type = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource0().getType());
    auto src1Type = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource1().getType());
    auto src2Type = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource2().getType());
    auto src3Type = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource3().getType());
    if (!dstType || !src0Type || !src1Type || !src2Type || !src3Type) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vmrgsort4 operand types");
    }

    Type elemType = cast<pto::PtrType>(op.getDestination().getType()).getElementType();
    FailureOr<Value> packedSrc = packVmrgsort4SourceAddr(op, adaptor.getSource0(), adaptor.getSource1(),
                                                         adaptor.getSource2(), adaptor.getSource3(), elemType);
    if (failed(packedSrc)) {
      return rewriter.notifyMatchFailure(op, "failed to pack vmrgsort4 source addresses");
    }

    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, adaptor.getDestination(), 6);
    if (failed(dst)) {
      return rewriter.notifyMatchFailure(op, "failed to normalize vmrgsort4 destination");
    }

    FailureOr<StringRef> calleeName = buildVmrgsort4Callee(op.getContext(), op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vmrgsort4 signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{(*dst).getType(), (*packedSrc).getType(),
                                                       adaptor.getCount().getType(), adaptor.getConfig().getType()},
                                             TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*dst, *packedSrc, adaptor.getCount(), adaptor.getConfig()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

void populateVPTOVectorMemoryPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                      LoweringState &state) {
  patterns
      .add<LowerPredicateMaskBinaryOpPattern<pto::PselOp>, LowerPredicateMaskBinaryOpPattern<pto::PandOp>,
           LowerPredicateMaskBinaryOpPattern<pto::PorOp>, LowerPredicateMaskBinaryOpPattern<pto::PxorOp>,
           LowerPredicatePairReorderOpPattern<pto::PdintlvB8Op>, LowerPredicatePairReorderOpPattern<pto::PdintlvB16Op>,
           LowerPredicatePairReorderOpPattern<pto::PdintlvB32Op>, LowerPredicatePairReorderOpPattern<pto::PintlvB8Op>,
           LowerPredicatePairReorderOpPattern<pto::PintlvB16Op>, LowerPredicatePairReorderOpPattern<pto::PintlvB32Op>,
           LowerUnpackOpPattern<pto::VsunpackOp>, LowerUnpackOpPattern<pto::VzunpackOp>, LowerVpackOpPattern,
           LowerCmpOpPattern<pto::VcmpOp>, LowerCmpOpPattern<pto::VcmpsOp>, LowerPltOpPattern<pto::PltB8Op>,
           LowerPltOpPattern<pto::PltB16Op>, LowerPltOpPattern<pto::PltB32Op>, LowerPltmOpPattern<pto::PltmB8Op>,
           LowerPltmOpPattern<pto::PltmB16Op>, LowerPltmOpPattern<pto::PltmB32Op>, LowerPsetOpPattern<pto::PsetB8Op>,
           LowerPsetOpPattern<pto::PsetB16Op>, LowerPsetOpPattern<pto::PsetB32Op>, LowerPgeOpPattern<pto::PgeB8Op>,
           LowerPgeOpPattern<pto::PgeB16Op>, LowerPgeOpPattern<pto::PgeB32Op>, LowerVldsOpPattern, LowerVldsx2OpPattern,
           LowerVsldbOpPattern, LowerVldasOpPattern, LowerInitAlignOpPattern, LowerVldusOpPattern, LowerSprclrOpPattern,
           LowerSprStoreOpPattern<pto::SprstiOp>, LowerSprStoreOpPattern<pto::SprstsOp>, LowerVstsOpPattern,
           LowerVsstbOpPattern, LowerVstsx2OpPattern, LowerVstarOpPattern, LowerVstasOpPattern, LowerVgather2OpPattern,
           LowerVgather2BcOpPattern, LowerVgatherbOpPattern, LowerVscatterOpPattern, LowerVaxpyOpPattern,
           LowerVmulscvtOpPattern, LowerVciOpPattern, LowerVexpdifOpPattern, LowerVbitsortOpPattern,
           LowerVmrgsort4OpPattern, LowerPstuOpPattern, LowerVstusOpPattern, LowerVsturOpPattern>(
          typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto::detail
