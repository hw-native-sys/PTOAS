// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

class ConvertVPTOUnrealizedCastOp final : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }
    if (!hasVPTOConvertibleType(op->getOperandTypes()) && !hasVPTOConvertibleType(op->getResultTypes())) {
      return failure();
    }

    Type convertedResultType = getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType) {
      return failure();
    }

    Value input = adaptor.getOperands().front();
    if (input.getType() != convertedResultType) {
      return failure();
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

class ConvertPtoDeclareStructOp final : public OpConversionPattern<pto::DeclareStructOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::DeclareStructOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type declaredType = op.getResult().getType();
    auto resultType = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(declaredType));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer result type");
    }
    auto structType = cast<pto::StructType>(declaredType);
    Type storageType = getVPTOStructStorageType(structType, rewriter);
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc) {
      return rewriter.notifyMatchFailure(op, "expected struct declaration inside a function");
    }

    // A non-entry alloca is a dynamic stack allocation. Keep one stack slot per
    // declaration per function invocation even when the declaration is nested
    // in a loop or a region.
    Value storage;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block &entryBlock = parentFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      Value one = rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));
      storage = rewriter.create<LLVM::AllocaOp>(op.getLoc(), resultType, storageType, one, /*alignment=*/0);
    }
    rewriter.replaceOp(op, storage);
    return success();
  }
};

class ConvertPtoStructGetOp final : public OpConversionPattern<pto::StructGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StructGetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }
    Value structValue = adaptor.getOperands().front();
    auto structType = cast<pto::StructType>(op->getOperand(0).getType());
    FailureOr<Value> address = getVPTOStructFieldAddress(rewriter, op.getLoc(), structValue, structType, op.getPath());
    if (failed(address)) {
      return rewriter.notifyMatchFailure(op, "invalid struct field path");
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType, *address, getNaturalByteAlignment(resultType));
    return success();
  }
};

class ConvertPtoStructSetOp final : public OpConversionPattern<pto::StructSetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StructSetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value structValue = adaptor.getOperands().front();
    auto structType = cast<pto::StructType>(op->getOperand(0).getType());
    FailureOr<Value> address = getVPTOStructFieldAddress(rewriter, op.getLoc(), structValue, structType, op.getPath());
    if (failed(address)) {
      return rewriter.notifyMatchFailure(op, "invalid struct field path");
    }
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), *address,
                                               getNaturalByteAlignment(adaptor.getValue().getType()));
    return success();
  }
};

class ConvertArithSelectOp final : public OpConversionPattern<arith::SelectOp> {
public:
  ConvertArithSelectOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(typeConverter, context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "only scalar i1 conditions supported for VPTO arith.select");
    }

    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    Value trueValue = adaptor.getTrueValue();
    Value falseValue = adaptor.getFalseValue();
    if (trueValue.getType() != convertedResultType || falseValue.getType() != convertedResultType) {
      return rewriter.notifyMatchFailure(op, "converted true/false values must match result type");
    }

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, convertedResultType, adaptor.getCondition(), trueValue,
                                                 falseValue);
    return success();
  }
};

class ConvertPtoAddPtrOp final : public OpConversionPattern<pto::AddPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType);
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    auto gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), llvmPtrType,
        normalizeGEPElementTypeForLLVMLowering(cast<pto::PtrType>(op.getPtr().getType()).getElementType(), rewriter),
        adaptor.getPtr(), ValueRange{offset});
    rewriter.replaceOp(op, gep.getResult());
    return success();
  }
};

class ConvertPtoCastPtrOp final : public OpConversionPattern<pto::CastPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType) {
      return rewriter.notifyMatchFailure(op, "could not convert castptr result type");
    }

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    if (inputType == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType)) {
      if (isa<IntegerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, llvmPtrType, input);
        return success();
      }
      auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(inputType);
      if (!sourcePtrType) {
        return rewriter.notifyMatchFailure(op, "expected integer or LLVM pointer input");
      }
      if (sourcePtrType.getAddressSpace() == llvmPtrType.getAddressSpace()) {
        rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmPtrType, input);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "cross-address-space ptr casts are unsupported");
    }

    if (auto resultIntType = dyn_cast<IntegerType>(convertedResultType)) {
      if (isa<LLVM::LLVMPointerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultIntType, input);
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "unsupported castptr conversion");
  }
};

class ConvertPtoLoadScalarOp final : public OpConversionPattern<pto::LoadScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Type convertedValueType = getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType) {
      return rewriter.notifyMatchFailure(op, "could not convert load_scalar result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             normalizeGEPElementTypeForLLVMLowering(convertedValueType, rewriter),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, convertedValueType, elemPtr,
                                              getNaturalByteAlignment(convertedValueType));
    return success();
  }
};

class ConvertPtoStoreScalarOp final : public OpConversionPattern<pto::StoreScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), llvmPtrType, normalizeGEPElementTypeForLLVMLowering(adaptor.getValue().getType(), rewriter),
          adaptor.getPtr(), ValueRange{offset});
    }

    rewriter.create<LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), elemPtr,
                                   getNaturalByteAlignment(adaptor.getValue().getType()));
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertPtoLoadOp final : public OpConversionPattern<pto::PTOLoadOp> {
public:
  ConvertPtoLoadOp(TypeConverter &typeConverter, MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::PTOLoadOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(pto::PTOLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Type convertedValueType = getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType) {
      return rewriter.notifyMatchFailure(op, "could not convert load result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType, convertedValueType, adaptor.getPtr(),
                                             ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, convertedValueType, elemPtr,
                                              getNaturalByteAlignment(convertedValueType));
    return success();
  }
};

static Type getLdgCallResultType(Type valueType, Type convertedValueType, ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8 || width == 16) {
      return rewriter.getI32Type();
    }
    return convertedValueType;
  }
  if (valueType.isF16() || valueType.isBF16() || valueType.isF32()) {
    return rewriter.getI32Type();
  }
  if (valueType.isF64()) {
    return rewriter.getI64Type();
  }
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    return rewriter.getI32Type();
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16) {
      return rewriter.getI32Type();
    }
    if (totalBits == 32) {
      return rewriter.getI32Type();
    }
    if (totalBits == 64) {
      return rewriter.getI64Type();
    }
  }
  return convertedValueType;
}

static Value convertLdgCallResult(Location loc, Type valueType, Type convertedValueType, Value callResult,
                                  ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8 || width == 16) {
      return rewriter.create<arith::TruncIOp>(loc, rewriter.getIntegerType(width), callResult);
    }
    return callResult;
  }

  if (valueType.isF16() || valueType.isBF16()) {
    Value payload = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), callResult);
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, payload);
  }
  if (valueType.isF32() || valueType.isF64()) {
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, callResult);
  }
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    Value payload = rewriter.create<arith::TruncIOp>(loc, rewriter.getI8Type(), callResult);
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, payload);
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16) {
      Value trunc = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), callResult);
      return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, trunc);
    }
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, callResult);
  }
  return callResult;
}

static FailureOr<Value> preparePtoLdgAddress(pto::PTOLdgOp op, pto::PTOLdgOp::Adaptor adaptor, Type convertedValueType,
                                             ConversionPatternRewriter &rewriter) {
  auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
  if (!llvmPtrType) {
    return failure();
  }

  Value offset = adaptor.getOffset();
  if (offset.getType().isIndex()) {
    offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
  }

  Value elemPtr = adaptor.getPtr();
  if (!matchPattern(offset, m_Zero())) {
    Type elementType = normalizeGEPElementTypeForLLVMLowering(convertedValueType, rewriter);
    elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType, elementType, adaptor.getPtr(), ValueRange{offset});
  }

  auto ptrType = cast<pto::PtrType>(op.getPtr().getType());
  return reinterpretPointerToAddrSpace(op, elemPtr, static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
}

class ConvertPtoLdgOp final : public OpConversionPattern<pto::PTOLdgOp> {
public:
  ConvertPtoLdgOp(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PTOLdgOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PTOLdgOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<LLVM::LLVMPointerType>(adaptor.getPtr().getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Type convertedValueType = getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType) {
      return rewriter.notifyMatchFailure(op, "could not convert ldg result type");
    }

    FailureOr<Value> ptr = preparePtoLdgAddress(op, adaptor, convertedValueType, rewriter);
    if (failed(ptr)) {
      return rewriter.notifyMatchFailure(op, "failed to map ldg pointer");
    }

    pto::L1Cache l1cache = op.getL1cacheAttr() ? op.getL1cacheAttr().getValue() : pto::L1Cache::Cache;
    FailureOr<StringRef> calleeName = buildL1CacheLoadCallee(op.getContext(), op.getValue().getType(), l1cache);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported ldg signature");
    }

    pto::LdL2Cache mode = op.getL2cacheAttr() ? op.getL2cacheAttr().getValue() : pto::LdL2Cache::NMFV;
    Value modeValue = getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(mode));
    Type callResultType = getLdgCallResultType(op.getValue().getType(), convertedValueType, rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{ptr->getType(), rewriter.getI32Type()}, TypeRange{callResultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{callResultType}, ValueRange{*ptr, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value result =
        convertLdgCallResult(op.getLoc(), op.getValue().getType(), convertedValueType, call.getResult(0), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

class ConvertPtoStoreOp final : public OpConversionPattern<pto::PTOStoreOp> {
public:
  ConvertPtoStoreOp(TypeConverter &typeConverter, MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::PTOStoreOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(pto::PTOStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType, adaptor.getValue().getType(), adaptor.getPtr(),
                                             ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), elemPtr,
                                               getNaturalByteAlignment(adaptor.getValue().getType()));
    return success();
  }
};

static Value convertStgValue(Location loc, Type valueType, Value value, ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8) {
      return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), value);
    }
    if (width == 16) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
    }
    return value;
  }

  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    Value payload = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI8Type(), value);
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), payload);
  }
  if (valueType.isBF16()) {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
  }
  if (valueType.isF32()) {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
  }
  if (valueType.isF64()) {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
    }
    if (totalBits == 32) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }
    if (totalBits == 64) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    }
  }
  return value;
}

class ConvertPtoStgOp final : public OpConversionPattern<pto::PTOStgOp> {
public:
  ConvertPtoStgOp(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PTOStgOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PTOStgOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), llvmPtrType, normalizeGEPElementTypeForLLVMLowering(adaptor.getValue().getType(), rewriter),
          adaptor.getPtr(), ValueRange{offset});
    }

    auto ptrTy = cast<pto::PtrType>(op.getPtr().getType());
    FailureOr<Value> ptr =
        reinterpretPointerToAddrSpace(op, elemPtr, static_cast<unsigned>(ptrTy.getMemorySpace().getAddressSpace()));
    if (failed(ptr)) {
      return rewriter.notifyMatchFailure(op, "failed to map stg pointer");
    }

    pto::L1Cache l1cache = op.getL1cacheAttr() ? op.getL1cacheAttr().getValue() : pto::L1Cache::Cache;
    FailureOr<StringRef> calleeName = buildL1CacheStoreCallee(op.getContext(), op.getValue().getType(), l1cache);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported stg signature");
    }

    pto::StL2Cache mode = op.getL2cacheAttr() ? op.getL2cacheAttr().getValue() : pto::StL2Cache::NMFV;
    Value modeValue = getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(mode));
    Value storedValue = convertStgValue(op.getLoc(), op.getValue().getType(), adaptor.getValue(), rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{ptr->getType(), storedValue.getType(), rewriter.getI32Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, ValueRange{*ptr, storedValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

static std::string buildLdDevCalleeName(unsigned width) { return "llvm.hivm.LD.DEV.u" + std::to_string(width) + ".GM"; }

static std::string buildStDevCalleeName(unsigned width) { return "llvm.hivm.ST.DEV.u" + std::to_string(width); }

class ConvertPtoLdDevOp final : public OpConversionPattern<pto::PTOLdDevOp> {
public:
  ConvertPtoLdDevOp(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PTOLdDevOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PTOLdDevOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    auto valueType = dyn_cast<IntegerType>(op.getValue().getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(op, "expected integer result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Type convertedValueType = getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType) {
      return rewriter.notifyMatchFailure(op, "could not convert ld_dev result type");
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             normalizeGEPElementTypeForLLVMLowering(convertedValueType, rewriter),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    FailureOr<Value> gmPtr = reinterpretPointerToAddrSpace(op, elemPtr, static_cast<unsigned>(pto::AddressSpace::GM));
    if (failed(gmPtr)) {
      return rewriter.notifyMatchFailure(op, "failed to map ld_dev GM pointer");
    }

    std::string calleeName = buildLdDevCalleeName(valueType.getWidth());
    Value intrinsicOffset = getI64Constant(rewriter, op.getLoc(), 0);
    auto funcType =
        rewriter.getFunctionType(TypeRange{gmPtr->getType(), rewriter.getI64Type()}, TypeRange{rewriter.getI64Type()});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{rewriter.getI64Type()},
                                              ValueRange{*gmPtr, intrinsicOffset});
    state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});

    Value result = call.getResult(0);
    if (valueType.getWidth() < 64) {
      result = rewriter.create<arith::TruncIOp>(op.getLoc(), convertedValueType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

class ConvertPtoStDevOp final : public OpConversionPattern<pto::PTOStDevOp> {
public:
  ConvertPtoStDevOp(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PTOStDevOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PTOStDevOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    auto valueType = dyn_cast<IntegerType>(op.getValue().getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(op, "expected integer value type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(), rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), llvmPtrType, normalizeGEPElementTypeForLLVMLowering(adaptor.getValue().getType(), rewriter),
          adaptor.getPtr(), ValueRange{offset});
    }

    FailureOr<Value> gmPtr = reinterpretPointerToAddrSpace(op, elemPtr, static_cast<unsigned>(pto::AddressSpace::GM));
    if (failed(gmPtr)) {
      return rewriter.notifyMatchFailure(op, "failed to map st_dev GM pointer");
    }

    Value payload = adaptor.getValue();
    if (valueType.getWidth() < 64) {
      payload = rewriter.create<arith::ExtUIOp>(op.getLoc(), rewriter.getI64Type(), payload);
    }

    std::string calleeName = buildStDevCalleeName(valueType.getWidth());
    Value intrinsicOffset = getI64Constant(rewriter, op.getLoc(), 0);
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI64Type(), gmPtr->getType(), rewriter.getI64Type()},
                                             TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{payload, *gmPtr, intrinsicOffset});
    state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class ConvertVPTOTypedCarrierOp final : public ConversionPattern {
public:
  ConvertVPTOTypedCarrierOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<pto::CastPtrOp>(op)) {
      return failure();
    }
    Type propertyType;
    if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(op)) {
      propertyType = allocaOp.getElemType();
    } else if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
      propertyType = gepOp.getElemType();
    }
    if (!hasVPTOConvertibleType(op->getOperandTypes()) && !hasVPTOConvertibleType(op->getResultTypes()) &&
        !hasVPTOConvertibleType(propertyType)) {
      return failure();
    }
    if (op->getNumRegions() != 0) {
      return rewriter.notifyMatchFailure(op, "region ops with VPTO types are handled structurally");
    }

    SmallVector<Type> convertedResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), convertedResultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    }
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    state.propertiesAttr = op->getPropertiesAsAttribute();
    Operation *converted = rewriter.create(state);
    if (propertyType) {
      Type convertedPropertyType = typeConverter->convertType(propertyType);
      if (!convertedPropertyType) {
        return rewriter.notifyMatchFailure(op, "failed to convert LLVM element type");
      }
      if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(converted)) {
        allocaOp.setElemType(convertedPropertyType);
      } else {
        cast<LLVM::GEPOp>(converted).setElemType(convertedPropertyType);
      }
    }
    rewriter.replaceOp(op, converted->getResults());
    return success();
  }
};
void populateVPTOTypePatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns, ConversionTarget &target,
                              LoweringState &state) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertPtoAddPtrOp, ConvertPtoCastPtrOp, ConvertPtoLoadScalarOp, ConvertPtoStoreScalarOp,
               ConvertPtoDeclareStructOp, ConvertPtoStructGetOp, ConvertPtoStructSetOp>(typeConverter, context);
  patterns
      .add<ConvertPtoLoadOp, ConvertPtoStoreOp, ConvertPtoLdgOp, ConvertPtoStgOp, ConvertPtoLdDevOp, ConvertPtoStDevOp>(
          typeConverter, context, state);
  patterns.add<ConvertArithSelectOp, ConvertVPTOUnrealizedCastOp, ConvertVPTOTypedCarrierOp>(typeConverter, context);
}

} // namespace mlir::pto::detail
