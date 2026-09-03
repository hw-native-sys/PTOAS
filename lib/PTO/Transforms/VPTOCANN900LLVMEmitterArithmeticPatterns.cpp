// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterTemplates.h"

namespace mlir::pto::detail {

template <typename UnaryOp> class LowerUnaryMaskedOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryMaskedOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildUnaryMaskedCallee<UnaryOp>(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported unary VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert unary result type");
    }

    Value input = adaptor.getOperands()[0];
    Value mask = adaptor.getOperands()[1];
    Type expectedMaskType = this->getTypeConverter()->convertType(op->getOperand(1).getType());
    if (!input || !mask || input.getType() != resultType || mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted unary VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{input, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsqzOpPattern final : public OpConversionPattern<pto::VsqzOp> {
public:
  explicit LowerVsqzOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsqzOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VsqzOp op, pto::VsqzOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVsqzCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vsqz VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsqz types");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vsqz operand types");
    }

    Value storeHint = getI32Constant(rewriter, op.getLoc(), determineVsqzStoreHint(op));
    auto funcType =
        rewriter.getFunctionType(TypeRange{resultType, maskType, storeHint.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{input, mask, storeHint});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVusqzOpPattern final : public OpConversionPattern<pto::VusqzOp> {
public:
  explicit LowerVusqzOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VusqzOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VusqzOp op, pto::VusqzOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVusqzCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vusqz VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vusqz types");
    }

    Value src = adaptor.getSrc();
    Value mask = adaptor.getMask();
    if (!src || !mask || src.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vusqz operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{resultType, maskType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmulaOpPattern final : public OpConversionPattern<pto::VmulaOp> {
public:
  explicit LowerVmulaOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VmulaOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VmulaOp op, pto::VmulaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVmulaCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vmula VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vmula types");
    }

    Value acc = adaptor.getAcc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!acc || !lhs || !rhs || !mask || acc.getType() != resultType || lhs.getType() != resultType ||
        rhs.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vmula operand types");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{resultType, resultType, resultType, maskType}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{acc, lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmullOpPattern final : public OpConversionPattern<pto::VmullOp> {
public:
  explicit LowerVmullOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VmullOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VmullOp op, pto::VmullOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVmullCallee(op.getContext(), op.getLow().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vmull VPTO signature");
    }

    Type inputType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    SmallVector<Type> resultTypes;
    if (!inputType || !maskType || failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vmull types");
    }
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vmull results");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!lhs || !rhs || !mask || lhs.getType() != inputType || rhs.getType() != inputType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vmull operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{inputType, inputType, maskType}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes, ValueRange{lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp> struct BinaryMaskedCall {
  Type resultType;
  Value lhs;
  Value rhs;
  FailureOr<StringRef> calleeName;
};

template <typename BinaryOp>
static BinaryMaskedCall<BinaryOp> prepareBinaryMaskedCall(BinaryOp op, Value lhs, Value rhs, Type resultType,
                                                          ConversionPatternRewriter &rewriter) {
  StringRef stem = getBinaryMaskedStem<BinaryOp>();
  FailureOr<StringRef> calleeName =
      usesSignedBinaryCANN900Callee<BinaryOp>()
          ? buildCANN900SignedModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "x")
          : buildCANN900ModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "x");
  Type elementType = getElementTypeFromVectorLike(op.getResult().getType());
  if constexpr (std::is_same_v<BinaryOp, pto::VandOp> || std::is_same_v<BinaryOp, pto::VorOp> ||
                std::is_same_v<BinaryOp, pto::VxorOp>) {
    if (elementType && pto::isPTOLowPrecisionType(elementType)) {
      calleeName = buildDirectLowpVLogicCallee(op.getContext(), op.getResult().getType(), stem, "x");
      if (failed(calleeName)) {
        resultType = getLowpPayloadCarrierType(op.getResult().getType(), rewriter.getContext());
        if (resultType) {
          lhs = castToPayloadABI(op.getLoc(), lhs, op.getResult().getType(), rewriter);
          rhs = castToPayloadABI(op.getLoc(), rhs, op.getResult().getType(), rewriter);
          calleeName = buildLowpPayloadVLogicCallee(op.getContext(), op.getResult().getType(), stem, "x");
        }
      }
    }
  }
  return BinaryMaskedCall<BinaryOp>{resultType, lhs, rhs, calleeName};
}

template <typename BinaryOp> class LowerBinaryMaskedOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryMaskedOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert binary result type");
    }

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType = this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!lhs || !rhs || !mask || lhs.getType() != resultType || rhs.getType() != resultType ||
        mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted binary VPTO operand types");
    }

    BinaryMaskedCall<BinaryOp> callInfo = prepareBinaryMaskedCall(op, lhs, rhs, resultType, rewriter);
    if (!callInfo.resultType || failed(callInfo.calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported binary VPTO signature");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *callInfo.calleeName, TypeRange{callInfo.resultType},
                                              ValueRange{callInfo.lhs, callInfo.rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{callInfo.calleeName->str(), call.getCalleeType()});
    Value result = castFromPayloadABI(op.getLoc(), call.getResult(0), op.getResult().getType(), resultType, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

template <typename TernaryOp> class LowerTernaryMaskedOpPattern final : public OpConversionPattern<TernaryOp> {
public:
  explicit LowerTernaryMaskedOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<TernaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(TernaryOp op, typename TernaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getTernaryMaskedStem<TernaryOp>();
    FailureOr<StringRef> calleeName =
        usesSignedTernaryCANN900Callee<TernaryOp>()
            ? buildCANN900SignedModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "m")
            : buildCANN900ModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "m");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported ternary VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type expectedMaskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !expectedMaskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert ternary VPTO types");
    }

    Value acc = adaptor.getAcc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!acc || !lhs || !rhs || !mask || acc.getType() != resultType || lhs.getType() != resultType ||
        rhs.getType() != resultType || mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted ternary VPTO operand types");
    }

    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{acc, lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CarryOp> class LowerCarryBinaryOpPattern final : public OpConversionPattern<CarryOp> {
public:
  explicit LowerCarryBinaryOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CarryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(CarryOp op, typename CarryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getCarryBinaryStem<CarryOp>();
    FailureOr<StringRef> calleeName = buildCarryBinaryCallee(op.getContext(), op.getResult().getType(), stem);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported carry VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type carryType = this->getTypeConverter()->convertType(op->getResult(1).getType());
    if (!resultType || !carryType) {
      return rewriter.notifyMatchFailure(op, "failed to convert carry result types");
    }

    SmallVector<Value> callArgs;
    callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    const size_t expectedArgCount = hasCarryInput<CarryOp>() ? 4 : 3;
    if (callArgs.size() != expectedArgCount || callArgs[0].getType() != resultType ||
        callArgs[1].getType() != resultType || callArgs.back().getType() != carryType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted carry operand types");
    }
    if constexpr (hasCarryInput<CarryOp>()) {
      if (callArgs[2].getType() != carryType) {
        return rewriter.notifyMatchFailure(op, "unexpected converted carry input operand type");
      }
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType, carryType}, callArgs);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp> class LowerCopyOpPattern final : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = failure();
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>) {
      calleeName = buildCopyGmToUbCallee(op.getContext(), op.getSource().getType());
    } else {
      calleeName = buildCopyUbToGmCallee(op.getContext());
    }
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported copy VPTO signature");
    }

    auto llvmSourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");
    }

    FailureOr<Value> config0 = failure();
    FailureOr<Value> config1 = failure();
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>) {
      config0 = packCopyGmToUbConfig0(op, adaptor.getOperands());
      config1 = packCopyGmToUbConfig1(op, adaptor.getOperands());
    } else {
      config0 = packCopyUbToGmConfig0(op, adaptor.getOperands());
      config1 = packCopyUbToGmConfig1(op, adaptor.getOperands());
    }
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0], *config0, *config1};
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmDestType, llvmSourceType, rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyUbufToUbufOpPattern final : public OpConversionPattern<pto::CopyUbufToUbufOp> {
public:
  explicit LowerCopyUbufToUbufOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyUbufToUbufOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyUbufToUbufOp op, pto::CopyUbufToUbufOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");
    }

    FailureOr<Value> config = packCopyUbToUbConfig(op, adaptor.getOperands());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    StringRef calleeName = buildCopyUbToUbCallee(op.getContext());
    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0], *config};
    auto funcType =
        rewriter.getFunctionType(TypeRange{llvmDestType, llvmSourceType, rewriter.getI64Type()}, TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToUbufOpPattern final : public OpConversionPattern<pto::CopyCbufToUbufOp> {
public:
  explicit LowerCopyCbufToUbufOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToUbufOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToUbufOp op, pto::CopyCbufToUbufOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned ubufAddressSpace = static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, ubufAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ubuf pointer spaces");
    }

    FailureOr<Value> config = packCopyCbufToUbConfig(op, adaptor.getOperands());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    StringRef calleeName = buildCopyCbufToUbCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyUbufToCbufOpPattern final : public OpConversionPattern<pto::CopyUbufToCbufOp> {
public:
  explicit LowerCopyUbufToCbufOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyUbufToCbufOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyUbufToCbufOp op, pto::CopyUbufToCbufOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned ubufAddressSpace = static_cast<unsigned>(pto::AddressSpace::VEC);
    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, ubufAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map ubuf/cbuf pointer spaces");
    }

    FailureOr<Value> config = packCopyUbToCbufConfig(op, adaptor.getOperands());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    StringRef calleeName = buildCopyUbToCbufCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

struct CreateCbufMatrixFill {
  StringRef calleeName;
  Value pattern;
};

static FailureOr<CreateCbufMatrixFill> buildCreateCbufMatrixFill(pto::CreateCbufMatrixOp op, Value rawValue,
                                                                 ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Type i64Ty = rewriter.getI64Type();
  const uint64_t fillWordWidth = static_cast<uint64_t>(op.getFillWordBits());
  if (fillWordWidth == 16) {
    Value wordMask = getI32Constant(rewriter, loc, 0xFFFFU);
    Value lowWord = rewriter.create<arith::AndIOp>(loc, rawValue, wordMask);
    Value wordBits = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), lowWord);
    Value pattern = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), wordBits);
    return CreateCbufMatrixFill{"llvm.hivm.CREATE.CBUF.MATRIX.v3.u16.h", pattern};
  }
  if (fillWordWidth == 32) {
    Value pattern = rewriter.create<arith::ExtUIOp>(loc, i64Ty, rawValue);
    return CreateCbufMatrixFill{"llvm.hivm.CREATE.CBUF.MATRIX.v3.u32", pattern};
  }
  return failure();
}

static Value packCreateCbufMatrixConfig(Operation *anchor, Value repeatTimes, Value blockNum32b, Value dstGap32b,
                                        ConversionPatternRewriter &rewriter) {
  Location loc = anchor->getLoc();
  Value fieldMask = getI64Constant(rewriter, loc, 0x7FFFU);
  auto maskField = [&](Value value) -> Value { return rewriter.create<arith::AndIOp>(loc, value, fieldMask); };
  auto shiftField = [&](Value value, uint64_t amount) -> Value {
    return rewriter.create<arith::ShLIOp>(loc, value, getI64Constant(rewriter, loc, amount));
  };
  Value config = maskField(repeatTimes);
  config = rewriter.create<arith::OrIOp>(loc, config, shiftField(maskField(blockNum32b), 16));
  return rewriter.create<arith::OrIOp>(loc, config, shiftField(maskField(dstGap32b), 32));
}

class LowerCreateCbufMatrixOpPattern final : public OpConversionPattern<pto::CreateCbufMatrixOp> {
public:
  explicit LowerCreateCbufMatrixOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CreateCbufMatrixOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CreateCbufMatrixOp op, pto::CreateCbufMatrixOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value destinationRaw = adaptor.getDst();
    Value rawValue = adaptor.getRawValue();
    Value repeatTimes = adaptor.getRepeatTimes();
    Value blockNum32b = adaptor.getBlockNum_32b();
    Value dstGap32b = adaptor.getDstGap_32b();
    if (!destinationRaw || !rawValue || !repeatTimes || !blockNum32b || !dstGap32b) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer destination");
    }

    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    const bool validControlTypes = rawValue.getType() == i32Ty && repeatTimes.getType() == i64Ty &&
                                   blockNum32b.getType() == i64Ty && dstGap32b.getType() == i64Ty;
    if (!validControlTypes) {
      return rewriter.notifyMatchFailure(op, "expected i32 value and i64 controls");
    }

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map destination to mat/l1");
    }

    FailureOr<CreateCbufMatrixFill> fill = buildCreateCbufMatrixFill(op, rawValue, rewriter);
    if (failed(fill)) {
      return rewriter.notifyMatchFailure(op, "expected a 16-bit or 32-bit fill word");
    }

    Location loc = op.getLoc();
    Value config = packCreateCbufMatrixConfig(op, repeatTimes, blockNum32b, dstGap32b, rewriter);

    auto funcType =
        rewriter.getFunctionType(TypeRange{destination->getType(), i64Ty, fill->pattern.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(loc, fill->calleeName, TypeRange{}, ValueRange{*destination, config, fill->pattern});
    state.plannedDecls.push_back(PlannedDecl{fill->calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

static LogicalResult validateMadRawOperands(pto::MadRawOpInterface op, ValueRange operands, Value &bias, Value &xt,
                                            ConversionPatternRewriter &rewriter) {
  unsigned required = op.hasBiasOperand() ? 5 : 4;
  if (operands.size() < required) {
    return rewriter.notifyMatchFailure(op, "expected converted mad raw operands");
  }
  bias = op.hasBiasOperand() ? operands[3] : Value();
  xt = operands[op.hasBiasOperand() ? 4 : 3];
  for (unsigned index : {0U, 1U, 2U}) {
    if (!isa<LLVM::LLVMPointerType>(operands[index].getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer lhs/rhs/dst operands");
    }
  }
  if (bias && !isa<LLVM::LLVMPointerType>(bias.getType())) {
    return rewriter.notifyMatchFailure(op, "expected LLVM pointer bias operand");
  }
  return success();
}

struct MadRawLoweringValues {
  Value lhs;
  Value rhs;
  Value dst;
  Value bias;
  Value xt;
};

static FailureOr<MadRawLoweringValues> mapMadRawOperands(pto::MadRawOpInterface op, ValueRange convertedOperands,
                                                         ConversionPatternRewriter &rewriter) {
  Value biasRaw;
  Value xt;
  if (failed(validateMadRawOperands(op, convertedOperands, biasRaw, xt, rewriter))) {
    return failure();
  }

  constexpr unsigned caAddressSpace = static_cast<unsigned>(pto::AddressSpace::LEFT);
  constexpr unsigned cbAddressSpace = static_cast<unsigned>(pto::AddressSpace::RIGHT);
  constexpr unsigned ccAddressSpace = static_cast<unsigned>(pto::AddressSpace::ACC);
  constexpr unsigned btAddressSpace = static_cast<unsigned>(pto::AddressSpace::BIAS);
  FailureOr<Value> lhs = reinterpretPointerToAddrSpace(op, convertedOperands[0], caAddressSpace);
  FailureOr<Value> rhs = reinterpretPointerToAddrSpace(op, convertedOperands[1], cbAddressSpace);
  FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, convertedOperands[2], ccAddressSpace);
  FailureOr<Value> bias;
  if (biasRaw) {
    bias = reinterpretPointerToAddrSpace(op, biasRaw, btAddressSpace);
  }
  if (failed(lhs) || failed(rhs) || failed(dst) || (biasRaw && failed(bias))) {
    return failure();
  }
  return MadRawLoweringValues{*lhs, *rhs, *dst, biasRaw ? *bias : Value(), xt};
}

static LogicalResult lowerMadRawOp(pto::MadRawOpInterface op, ValueRange convertedOperands,
                                   ConversionPatternRewriter &rewriter, LoweringState &state) {
  FailureOr<MadRawLoweringValues> values = mapMadRawOperands(op, convertedOperands, rewriter);
  if (failed(values)) {
    return failure();
  }
  Type i64Ty = rewriter.getI64Type();
  FailureOr<StringRef> calleeName =
      op.isMadMxFamily() ? buildMxMadCallee(op.getContext(), op) : buildOrdinaryMadCallee(op.getContext(), op);
  if (failed(calleeName)) {
    return rewriter.notifyMatchFailure(op, "unsupported mad element types for raw dispatch");
  }

  Value callDst = values->dst;
  if (values->bias) {
    callDst = buildMadBiasDestination(op, rewriter, values->dst, values->bias);
  }
  auto funcType = rewriter.getFunctionType(
      TypeRange{values->dst.getType(), values->lhs.getType(), values->rhs.getType(), i64Ty}, TypeRange{});
  auto call = rewriter.create<func::CallOp>(op->getLoc(), *calleeName, TypeRange{},
                                            ValueRange{callDst, values->lhs, values->rhs, values->xt});
  state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
  rewriter.replaceOp(op, call.getResults());
  return success();
}

template <typename RawOp> class LowerMadRawPattern final : public OpConversionPattern<RawOp> {
public:
  explicit LowerMadRawPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<RawOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(RawOp op, typename RawOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto raw = dyn_cast<pto::MadRawOpInterface>(op.getOperation());
    if (!raw) {
      return failure();
    }
    return lowerMadRawOp(raw, adaptor.getOperands(), rewriter, state);
  }

private:
  LoweringState &state;
};

class LowerCopyGmToCbufOpPattern final : public OpConversionPattern<pto::CopyGmToCbufOp> {
public:
  explicit LowerCopyGmToCbufOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyGmToCbufOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyGmToCbufOp op, pto::CopyGmToCbufOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value nBurst = adaptor.getNBurst();
    Value lenBurst = adaptor.getLenBurst();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !nBurst || !lenBurst || !srcStride || !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();
    if (nBurst.getType() != i64Ty || lenBurst.getType() != i64Ty || srcStride.getType() != i64Ty ||
        dstStride.getType() != i64Ty) {
      return rewriter.notifyMatchFailure(op, "expected i64 config operands");
    }

    constexpr unsigned gmAddressSpace = static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, gmAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/gm pointer spaces");
    }

    FailureOr<StringRef> calleeName = buildCopyGmToCbufCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported copy_gm_to_cbuf element type");
    }
    FailureOr<Value> config0 = packCopyGmToCbufConfig0(op, nBurst, lenBurst);
    FailureOr<Value> config1 = packCopyGmToCbufConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_gm_to_cbuf config");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp> class LowerCopyGmToCbufMultiOpPattern final : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyGmToCbufMultiOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned gmAddressSpace = static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, gmAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/gm pointer spaces");
    }

    FailureOr<Value> config0 = packCopyGmToCbufMultiConfig0(op, adaptor.getSid(), adaptor.getLoop1SrcStride(),
                                                            adaptor.getL2CacheCtrl(), adaptor.getNValue());
    FailureOr<Value> config1 =
        packCopyGmToCbufMultiConfig1(op, adaptor.getDValue(), adaptor.getLoop4SrcStride(), adaptor.getSmallc0En());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack multi copy config");
    }

    FailureOr<StringRef> calleeName = [&](MLIRContext *ctx, Type sourceType) -> FailureOr<StringRef> {
      if constexpr (std::is_same_v<CopyOp, pto::CopyGmToCbufMultiNd2NzOp>) {
        return buildCopyGmToCbufMultiNd2NzCallee(ctx, op.getSource().getType());
      }
      return buildCopyGmToCbufMultiDn2NzCallee(ctx, sourceType);
    }(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported copy_gm_to_cbuf_multi element type");
    }

    Type i64Ty = rewriter.getI64Type();
    auto funcType =
        rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToBtOpPattern final : public OpConversionPattern<pto::CopyCbufToBtOp> {
public:
  explicit LowerCopyCbufToBtOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToBtOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToBtOp op, pto::CopyCbufToBtOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned btAddressSpace = static_cast<unsigned>(pto::AddressSpace::BIAS);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destinationPtr = reinterpretPointerToAddrSpace(op, destinationRaw, btAddressSpace);
    if (failed(source) || failed(destinationPtr)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/bt pointer spaces");
    }

    FailureOr<Value> config =
        packCopyCbufToBtConfig(op, adaptor.getConvControl(), adaptor.getNBurst(), adaptor.getLenBurst(),
                               adaptor.getSourceGap(), adaptor.getDstGap());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_bt config");
    }

    Type i64Ty = rewriter.getI64Type();
    Value destination = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *destinationPtr);
    FailureOr<StringRef> calleeName = buildCopyCbufToBtCallee(op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported copy_cbuf_to_bt source element type");
    }
    auto funcType = rewriter.getFunctionType(TypeRange{i64Ty, source->getType(), i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, ValueRange{destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToFbufOpPattern final : public OpConversionPattern<pto::CopyCbufToFbufOp> {
public:
  explicit LowerCopyCbufToFbufOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToFbufOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToFbufOp op, pto::CopyCbufToFbufOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned fbufAddressSpace = 7;
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, fbufAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/fbuf pointer spaces");
    }

    FailureOr<Value> config = packCopyCbufToFbufConfig(op, adaptor.getNBurst(), adaptor.getLenBurst(),
                                                       adaptor.getSourceGap(), adaptor.getDstGap());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_fbuf config");
    }

    Type i64Ty = rewriter.getI64Type();
    StringRef calleeName = buildCopyCbufToFbufCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCaOpPattern final : public OpConversionPattern<pto::LoadCbufToCaOp> {
public:
  explicit LowerLoadCbufToCaOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCaOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCaOp op, pto::LoadCbufToCaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value mStart = adaptor.getMStart();
    Value kStart = adaptor.getKStart();
    Value mStep = adaptor.getMStep();
    Value kStep = adaptor.getKStep();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !mStart || !kStart || !mStep || !kStep || !srcStride || !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned caAddressSpace = static_cast<unsigned>(pto::AddressSpace::LEFT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, caAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ca pointer spaces");
    }

    FailureOr<Value> config0 = packLoadCbufToCaConfig0(op, mStart, kStart, mStep, kStep);
    FailureOr<Value> config1 = packLoadCbufToCaConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_ca config");
    }
    Value transpose = getI64Constant(rewriter, op.getLoc(), op.getTranspose() ? 1 : 0);

    FailureOr<StringRef> calleeName = buildLoadCbufToCaCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported load_cbuf_to_ca element type");
    }
    auto funcType = rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty, i64Ty},
                                             TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0, *config1, transpose});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
static FailureOr<std::pair<Value, Value>> mapLoadCbufToS4Pointers(LoadOp op, typename LoadOp::Adaptor adaptor) {
  Value sourceRaw = adaptor.getSource();
  Value destinationRaw = adaptor.getDestination();
  if (!sourceRaw || !destinationRaw || !isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
    return failure();
  }
  constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
  constexpr unsigned targetAddressSpace = std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>
                                              ? static_cast<unsigned>(pto::AddressSpace::LEFT)
                                              : static_cast<unsigned>(pto::AddressSpace::RIGHT);
  FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
  FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, targetAddressSpace);
  if (failed(source) || failed(destination)) {
    return failure();
  }
  return std::pair<Value, Value>{*source, *destination};
}

template <typename LoadOp> static FailureOr<StringRef> buildLoadCbufToS4Callee(LoadOp op) {
  if constexpr (std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>) {
    return buildLoadCbufToCaS4Callee(op.getContext(), op.getSource().getType());
  }
  return buildLoadCbufToCbS4Callee(op.getContext(), op.getSource().getType());
}

template <typename LoadOp> class LowerLoadCbufToS4OpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerLoadCbufToS4OpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<std::pair<Value, Value>> pointers = mapLoadCbufToS4Pointers(op, adaptor);
    if (failed(pointers)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cube pointer spaces");
    }

    FailureOr<Value> config0 =
        packLoadCbufToS4Config0(op, adaptor.getMStart(), adaptor.getKStart(), adaptor.getMStep(), adaptor.getKStep());
    FailureOr<Value> config1 = packLoadCbufToS4Config1(op, adaptor.getSrcStride(), adaptor.getDstStride());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_*_s4 config");
    }

    Value transpose = castIntegerLikeTo(op, adaptor.getTranspose(), rewriter.getI64Type());
    if (!transpose) {
      return rewriter.notifyMatchFailure(op, "failed to cast transpose to i64");
    }

    FailureOr<StringRef> calleeName = buildLoadCbufToS4Callee(op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported load_cbuf_to_*_s4 element type");
    }
    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(
        TypeRange{pointers->second.getType(), pointers->first.getType(), i64Ty, i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{pointers->second, pointers->first, *config0, *config1, transpose});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCbOpPattern final : public OpConversionPattern<pto::LoadCbufToCbOp> {
public:
  explicit LowerLoadCbufToCbOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCbOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCbOp op, pto::LoadCbufToCbOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value mStart = adaptor.getMStart();
    Value kStart = adaptor.getKStart();
    Value mStep = adaptor.getMStep();
    Value kStep = adaptor.getKStep();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !mStart || !kStart || !mStep || !kStep || !srcStride || !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned cbAddressSpace = static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, cbAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cb pointer spaces");
    }

    bool transpose = op.getTranspose();
    FailureOr<Value> config0 = packLoadCbufToCbConfig0(op, mStart, kStart, mStep, kStep);
    FailureOr<Value> config1 = packLoadCbufToCbConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_cb config");
    }
    Value transposeValue = getI64Constant(rewriter, op.getLoc(), transpose ? 1 : 0);

    FailureOr<StringRef> calleeName = buildLoadCbufToCbCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported load_cbuf_to_cb element type");
    }
    auto funcType = rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty, i64Ty},
                                             TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0, *config1, transposeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCaMxOpPattern final : public OpConversionPattern<pto::LoadCbufToCaMxOp> {
public:
  explicit LowerLoadCbufToCaMxOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCaMxOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCaMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value srcRaw = adaptor.getSource();
    Value dstRaw = adaptor.getDestination();
    if (!srcRaw || !dstRaw || !adaptor.getXStartPosition() || !adaptor.getYStartPosition() || !adaptor.getXStep() ||
        !adaptor.getYStep() || !adaptor.getSrcStride() || !adaptor.getDstStride()) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) || !isa<LLVM::LLVMPointerType>(dstRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned caAddressSpace = static_cast<unsigned>(pto::AddressSpace::LEFT);
    FailureOr<Value> src = reinterpretPointerToAddrSpace(op, srcRaw, cbufAddressSpace);
    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, dstRaw, caAddressSpace);
    if (failed(src) || failed(dst)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ca pointer spaces");
    }

    Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
    unsigned elemBitWidth = pto::getPTOStorageElemBitWidth(sourceElemType);
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(op, "unsupported load_cbuf_to_ca_mx element type");
    }
    FailureOr<Value> config0 = packLoadCbufToCaConfig0(op, adaptor.getXStartPosition(), adaptor.getYStartPosition(),
                                                       adaptor.getXStep(), adaptor.getYStep());
    FailureOr<Value> config1 = packLoadCbufToCaConfig1(op, adaptor.getSrcStride(), adaptor.getDstStride());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_ca_mx config");
    }
    auto i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *dst);

    StringRef calleeName = buildLoadCbufToCaMxCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{i64Ty, src->getType(), i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{dstAddr, *src, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCbMxOpPattern final : public OpConversionPattern<pto::LoadCbufToCbMxOp> {
public:
  explicit LowerLoadCbufToCbMxOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCbMxOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCbMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value srcRaw = adaptor.getSource();
    Value dstRaw = adaptor.getDestination();
    if (!srcRaw || !dstRaw || !adaptor.getXStartPosition() || !adaptor.getYStartPosition() || !adaptor.getXStep() ||
        !adaptor.getYStep() || !adaptor.getSrcStride() || !adaptor.getDstStride()) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) || !isa<LLVM::LLVMPointerType>(dstRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace = static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned cbAddressSpace = static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> src = reinterpretPointerToAddrSpace(op, srcRaw, cbufAddressSpace);
    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, dstRaw, cbAddressSpace);
    if (failed(src) || failed(dst)) {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cb pointer spaces");
    }

    Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
    unsigned elemBitWidth = pto::getPTOStorageElemBitWidth(sourceElemType);
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(op, "unsupported load_cbuf_to_cb_mx element type");
    }
    FailureOr<Value> config0 = packLoadCbufToCbConfig0(op, adaptor.getXStartPosition(), adaptor.getYStartPosition(),
                                                       adaptor.getXStep(), adaptor.getYStep());
    FailureOr<Value> config1 = packLoadCbufToCbConfig1(op, adaptor.getSrcStride(), adaptor.getDstStride());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_cb_mx config");
    }
    auto i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *dst);

    StringRef calleeName = buildLoadCbufToCbMxCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{i64Ty, src->getType(), i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{dstAddr, *src, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyMatrixCcToGmOpPattern final : public OpConversionPattern<pto::CopyMatrixCcToGmOp> {
public:
  explicit LowerCopyMatrixCcToGmOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyMatrixCcToGmOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::CopyMatrixCcToGmOp op, pto::CopyMatrixCcToGmOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value xm = adaptor.getXm();
    Value xt = adaptor.getXt();
    if (!sourceRaw || !destinationRaw || !xm || !xt) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();
    if (xm.getType() != i64Ty || xt.getType() != i64Ty) {
      return rewriter.notifyMatchFailure(op, "expected i64 xm/xt operands");
    }

    constexpr unsigned gmAddressSpace = static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned ccAddressSpace = static_cast<unsigned>(pto::AddressSpace::ACC);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, ccAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, gmAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cc/gm pointer spaces");
    }

    StringRef calleeName = buildCopyMatrixCcToGmCallee(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{*destination, *source, xm, xt});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp> class LowerCopyMatrixCcToBufOpPattern final : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyMatrixCcToBufOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) || !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned ccAddressSpace = static_cast<unsigned>(pto::AddressSpace::ACC);
    constexpr unsigned targetAddressSpace = std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
                                                ? static_cast<unsigned>(pto::AddressSpace::MAT)
                                                : static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, ccAddressSpace);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(op, destinationRaw, targetAddressSpace);
    if (failed(source) || failed(destination)) {
      return rewriter.notifyMatchFailure(op, "failed to map cc->buf pointer spaces");
    }

    Type i64Ty = rewriter.getI64Type();
    Value config0 = castIntegerLikeTo(op, adaptor.getConfig0(), i64Ty);
    Value config1 = castIntegerLikeTo(op, adaptor.getConfig1(), i64Ty);
    if (!config0 || !config1) {
      return rewriter.notifyMatchFailure(op, "failed to cast config operands to i64");
    }

    FailureOr<StringRef> calleeName = std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
                                          ? FailureOr<StringRef>(buildCopyMatrixCcToCbufCallee(op.getContext()))
                                          : buildCopyMatrixCcToUbCallee(op.getContext(), op.getDestination().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported copy_matrix_cc_to_{cbuf,ub} element type");
    }
    auto funcType =
        rewriter.getFunctionType(TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, config0, config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename VecScalarOp> class LowerVecScalarMaskedOpPattern final : public OpConversionPattern<VecScalarOp> {
public:
  explicit LowerVecScalarMaskedOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<VecScalarOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(VecScalarOp op, typename VecScalarOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getVecScalarMaskedStem<VecScalarOp>();
    FailureOr<StringRef> calleeName =
        usesSignedVecScalarCANN900Callee<VecScalarOp>()
            ? buildCANN900SignedModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "x")
            : buildCANN900ModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "x");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vec-scalar VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vec-scalar result type");
    }

    Value input = adaptor.getOperands()[0];
    Value scalar = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType = this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!input || !scalar || !mask || input.getType() != resultType || mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vec-scalar VPTO operand types");
    }

    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{input, scalar, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp> class LowerReductionUnaryOpPattern final : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerReductionUnaryOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getReductionUnaryStem<ReductionOp>();
    FailureOr<StringRef> calleeName =
        usesSignedReductionCANN900Callee<ReductionOp>()
            ? buildCANN900SignedModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "x")
            : buildCANN900ModeTypedCallee(op.getContext(), op.getResult().getType(), stem, "x");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported reduction VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert reduction result type");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{input, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename HistOp> class LowerHistogramOpPattern final : public OpConversionPattern<HistOp> {
public:
  explicit LowerHistogramOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<HistOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(HistOp op, typename HistOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = getHistogramCallee<HistOp>(op.getContext());
    if (calleeName.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported histogram op");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type sourceType = this->getTypeConverter()->convertType(op.getSource().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !sourceType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert histogram types");
    }

    Value acc = adaptor.getAcc();
    Value source = adaptor.getSource();
    Value mask = adaptor.getMask();
    Value bin = adaptor.getBin();
    if (!acc || !source || !mask || !bin || acc.getType() != resultType || source.getType() != sourceType ||
        mask.getType() != maskType || !bin.getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(op, "unexpected converted histogram operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{resultType, sourceType, maskType, rewriter.getI32Type()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType},
                                              ValueRange{acc, source, mask, bin});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ExtremaOp> class LowerExtremaPredicateOpPattern final : public OpConversionPattern<ExtremaOp> {
public:
  explicit LowerExtremaPredicateOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ExtremaOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ExtremaOp op, typename ExtremaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildExtremaPredicateCallee<ExtremaOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported extrema-predicate VPTO signature");
    }

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    Type predicateType = this->getTypeConverter()->convertType(op.getPredicate().getType());
    if (!valueType || !predicateType) {
      return rewriter.notifyMatchFailure(op, "failed to convert extrema-predicate result types");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != valueType || mask.getType() != predicateType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted extrema-predicate operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{valueType, predicateType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp>
class LowerWideningReductionUnaryOpPattern final : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerWideningReductionUnaryOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                                                LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getReductionUnaryStem<ReductionOp>();
    FailureOr<StringRef> calleeName = buildCANN900WideningReductionCallee(op.getContext(), op.getInput().getType(),
                                                                          op.getResult().getType(), stem, "x");
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported widening reduction VPTO signature");
    }

    Type inputType = this->getTypeConverter()->convertType(op.getInput().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!inputType || !resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert widening reduction types");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != inputType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted widening reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{input, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVselOpPattern final : public OpConversionPattern<pto::VselOp> {
public:
  explicit LowerVselOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VselOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VselOp op, pto::VselOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVselCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vsel VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsel result type");
    }

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType || src1.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vsel operand types");
    }

    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVdupOpPattern final : public OpConversionPattern<pto::VdupOp> {
public:
  explicit LowerVdupOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VdupOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VdupOp op, pto::VdupOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVdupCallee(op.getContext(), op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vdup VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vdup result type");
    }

    Value mask = adaptor.getMask();
    if (!mask || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vdup mask type");
    }

    SmallVector<Value> callArgs;
    bool vectorInput = isa<VectorType, pto::VRegType>(op.getInput().getType());
    if (vectorInput) {
      Value input = adaptor.getInput();
      if (!input || input.getType() != resultType) {
        return rewriter.notifyMatchFailure(op, "vector-input vdup requires matching result type");
      }
      callArgs.push_back(input);
    } else {
      Type scalarType = getElementTypeFromVectorLike(op.getResult().getType());
      if (!scalarType || (op.getInput().getType() != scalarType &&
                          !isCompatibleScalarForSemanticType(scalarType, op.getInput().getType()))) {
        return rewriter.notifyMatchFailure(op, "unexpected scalar-input vdup type");
      }
      FailureOr<Value> normalizedScalar =
          normalizeVdupScalarOperand(rewriter, op.getLoc(), adaptor.getInput(), op.getResult().getType());
      if (failed(normalizedScalar)) {
        return rewriter.notifyMatchFailure(op, "failed to normalize scalar vdup input");
      }
      Value scalarForCall =
          normalizeByteScalarOperandForCANN900VectorCall(rewriter, op.getLoc(), *normalizedScalar, scalarType);
      callArgs.push_back(scalarForCall);
    }

    callArgs.push_back(mask);
    callArgs.push_back(getI32Constant(rewriter, op.getLoc(), 1));

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbrOpPattern final : public OpConversionPattern<pto::VbrOp> {
public:
  explicit LowerVbrOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbrOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VbrOp op, pto::VbrOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVbrCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vbr VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vbr result type");
    }

    Value scalar = adaptor.getValue();
    Type expectedScalarType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!scalar || !expectedScalarType || scalar.getType() != expectedScalarType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vbr operand type");
    }

    scalar = normalizeByteScalarOperandForCANN900VectorCall(
        rewriter, op.getLoc(), scalar, cast<pto::VRegType>(op.getResult().getType()).getElementType());

    auto funcType = rewriter.getFunctionType(TypeRange{scalar.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{scalar});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

static FailureOr<Type> getVselrIntrinsicResultType(pto::VselrOp op, Type resultType, PatternRewriter &rewriter) {
  auto resultVectorType = dyn_cast<VectorType>(resultType);
  if (!resultVectorType) {
    return failure();
  }
  Type intrinsicResultType = resultType;
  if (auto floatType = dyn_cast<FloatType>(resultVectorType.getElementType()); floatType && floatType.isF32()) {
    intrinsicResultType =
        VectorType::get(resultVectorType.getShape(), rewriter.getI32Type(), resultVectorType.getScalableDims());
  }
  if (Type carrierType = getLowpPayloadCarrierType(op.getResult().getType(), rewriter.getContext())) {
    intrinsicResultType = carrierType;
  }
  return intrinsicResultType;
}

class LowerVselrOpPattern final : public OpConversionPattern<pto::VselrOp> {
public:
  explicit LowerVselrOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VselrOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VselrOp op, pto::VselrOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVselrCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vselr VPTO signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vselr result type");
    }
    FailureOr<Type> intrinsicResultType = getVselrIntrinsicResultType(op, resultType, rewriter);
    if (failed(intrinsicResultType)) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vselr result type");
    }

    Type indexType = this->getTypeConverter()->convertType(op.getSrc1().getType());
    if (!indexType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vselr index type");
    }

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    if (!src0 || !src1 || src1.getType() != indexType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted vselr operand types");
    }

    if (src0.getType() != *intrinsicResultType) {
      if (src0.getType() != resultType) {
        return rewriter.notifyMatchFailure(op, "unexpected converted vselr source type");
      }
      src0 = rewriter.create<LLVM::BitcastOp>(op.getLoc(), *intrinsicResultType, src0);
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{*intrinsicResultType, indexType}, TypeRange{*intrinsicResultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{*intrinsicResultType},
                                              ValueRange{src0, src1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});

    Value result = call.getResult(0);
    if (*intrinsicResultType != resultType) {
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
  explicit LowerPnotOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PnotOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PnotOp op, pto::PnotOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert pnot result type");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType || mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted pnot operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<pto::PnotOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{input, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename InterleaveOp> class LowerInterleaveOpPattern final : public OpConversionPattern<InterleaveOp> {
public:
  explicit LowerInterleaveOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<InterleaveOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(InterleaveOp op, typename InterleaveOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<InterleaveOp, pto::VintlvOp> ? "vintlv" : "vdintlv";
    FailureOr<StringRef> calleeName = buildInterleaveCallee(op.getContext(), op.getLow().getType(), stem);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported interleave VPTO signature");
    }

    Type lowType = this->getTypeConverter()->convertType(op.getLow().getType());
    Type highType = this->getTypeConverter()->convertType(op.getHigh().getType());
    if (!lowType || !highType || lowType != highType) {
      return rewriter.notifyMatchFailure(op, "failed to convert interleave result types");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != lowType || rhs.getType() != lowType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted interleave operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lowType, lowType}, TypeRange{lowType, highType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{lowType, highType}, ValueRange{lhs, rhs});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PackOp> class LowerPredicatePackOpPattern final : public OpConversionPattern<PackOp> {
public:
  explicit LowerPredicatePackOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<PackOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(PackOp op, typename PackOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert predicate-pack result type");
    }

    auto part = parseHiLoPartImmediate(op.getPart());
    if (!part) {
      return rewriter.notifyMatchFailure(op, "unsupported predicate-pack part immediate");
    }

    Value input = adaptor.getInput();
    if (!input || input.getType() != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted predicate-pack operand type");
    }

    Value partValue = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(*part));
    StringRef calleeName = getPredicatePackCallee<PackOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{resultType, rewriter.getI32Type()}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{input, partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

void populateVPTOArithmeticPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                    LoweringState &state) {
  patterns.add<LowerUnaryMaskedOpPattern<pto::VabsOp>, LowerUnaryMaskedOpPattern<pto::VexpOp>,
               LowerUnaryMaskedOpPattern<pto::VlnOp>, LowerUnaryMaskedOpPattern<pto::VnegOp>,
               LowerUnaryMaskedOpPattern<pto::VsqrtOp>, LowerUnaryMaskedOpPattern<pto::VreluOp>,
               LowerUnaryMaskedOpPattern<pto::VnotOp>, LowerVsqzOpPattern, LowerVusqzOpPattern, LowerVmulaOpPattern,
               LowerVmullOpPattern, LowerBinaryMaskedOpPattern<pto::VaddOp>, LowerBinaryMaskedOpPattern<pto::VsubOp>,
               LowerBinaryMaskedOpPattern<pto::VmulOp>, LowerBinaryMaskedOpPattern<pto::VdivOp>,
               LowerBinaryMaskedOpPattern<pto::VmaxOp>, LowerBinaryMaskedOpPattern<pto::VminOp>,
               LowerBinaryMaskedOpPattern<pto::VandOp>, LowerBinaryMaskedOpPattern<pto::VorOp>,
               LowerBinaryMaskedOpPattern<pto::VxorOp>, LowerTernaryMaskedOpPattern<pto::VmaddOp>,
               LowerBinaryMaskedOpPattern<pto::VpreluOp>, LowerCarryBinaryOpPattern<pto::VaddcOp>,
               LowerCarryBinaryOpPattern<pto::VsubcOp>, LowerCarryBinaryOpPattern<pto::VaddcsOp>,
               LowerCarryBinaryOpPattern<pto::VsubcsOp>, LowerBinaryMaskedOpPattern<pto::VshlOp>,
               LowerBinaryMaskedOpPattern<pto::VshrOp>, LowerVecScalarMaskedOpPattern<pto::VmulsOp>,
               LowerVecScalarMaskedOpPattern<pto::VaddsOp>, LowerVecScalarMaskedOpPattern<pto::VmaxsOp>,
               LowerVecScalarMaskedOpPattern<pto::VminsOp>, LowerVecScalarMaskedOpPattern<pto::VlreluOp>,
               LowerVecScalarMaskedOpPattern<pto::VshlsOp>, LowerVecScalarMaskedOpPattern<pto::VshrsOp>,
               LowerWideningReductionUnaryOpPattern<pto::VcaddOp>, LowerReductionUnaryOpPattern<pto::VcmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcminOp>, LowerReductionUnaryOpPattern<pto::VcgaddOp>,
               LowerReductionUnaryOpPattern<pto::VcgmaxOp>, LowerReductionUnaryOpPattern<pto::VcgminOp>,
               LowerReductionUnaryOpPattern<pto::VcpaddOp>, LowerHistogramOpPattern<pto::Chistv2Op>,
               LowerHistogramOpPattern<pto::Dhistv2Op>, LowerExtremaPredicateOpPattern<pto::VcbmaxOp>,
               LowerExtremaPredicateOpPattern<pto::VcbminOp>, LowerVdupOpPattern, LowerVbrOpPattern,
               LowerPredicatePackOpPattern<pto::PpackOp>, LowerPredicatePackOpPattern<pto::PunpackOp>,
               LowerVselOpPattern, LowerVselrOpPattern, LowerPnotOpPattern, LowerInterleaveOpPattern<pto::VintlvOp>,
               LowerInterleaveOpPattern<pto::VdintlvOp>, LowerCopyGmToCbufOpPattern, LowerLoadCbufToCaOpPattern,
               LowerLoadCbufToCbOpPattern, LowerLoadCbufToS4OpPattern<pto::LoadCbufToCaS4Op>,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCbS4Op>, LowerLoadCbufToCaMxOpPattern,
               LowerLoadCbufToCbMxOpPattern, LowerCopyMatrixCcToGmOpPattern,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToCbufOp>,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToUbOp>, LowerCopyCbufToBtOpPattern,
               LowerCopyCbufToFbufOpPattern, LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiNd2NzOp>,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiDn2NzOp>, LowerMadRawPattern<pto::MadRawOp>,
               LowerMadRawPattern<pto::MadBiasRawOp>, LowerMadRawPattern<pto::MadMxRawOp>,
               LowerMadRawPattern<pto::MadMxBiasRawOp>, LowerCopyOpPattern<pto::CopyGmToUbufOp>,
               LowerCopyOpPattern<pto::CopyUbufToGmOp>, LowerCopyUbufToUbufOpPattern, LowerCopyCbufToUbufOpPattern,
               LowerCopyUbufToCbufOpPattern, LowerCreateCbufMatrixOpPattern>(typeConverter, patterns.getContext(),
                                                                             state);
}

} // namespace mlir::pto::detail
