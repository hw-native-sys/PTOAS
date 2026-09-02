// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Transforms/VPTOLLVMEmitterHelper.h"
#include "VPTOLLVMEmitter/VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace mlir::pto {

void materializeVecScopeCarrierLoops(ModuleOp module);
LogicalResult applyQueriedTargetAttrs(ModuleOp module,
                                      const VPTOEmissionOptions &options,
                                      llvm::raw_ostream &diagOS);
LogicalResult attachAIVectorScopeMetadata(llvm::Module &llvmModule,
                                          llvm::raw_ostream &diagOS);
void attachHIVMKernelAnnotations(llvm::Module &llvmModule,
                                 ModuleOp sourceModule);

namespace {

constexpr llvm::StringLiteral kVectorSuffix = "_mix_aiv";
constexpr llvm::StringLiteral kCubeSuffix = "_mix_aic";

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

[[maybe_unused]] static FailureOr<Value>
materializeDynamicPltMask(ConversionPatternRewriter &rewriter,
                          LoweringState &state, Location loc, Value laneCount,
                          Type vectorElemType) {
  Type i32Type = rewriter.getI32Type();
  Value laneCountI32 = laneCount;
  if (laneCountI32.getType() != i32Type) {
    laneCountI32 = castIntegerLikeTo(rewriter.getInsertionBlock()->getParentOp(),
                                     laneCountI32, i32Type);
    if (!laneCountI32)
    {
      return failure();
    }
  }

  StringRef calleeName;
  if (vectorElemType.isF32()) {
    calleeName = StringRef("llvm.hivm.plt.b32.v300");
  } else if (vectorElemType.isF16() || vectorElemType.isBF16()) {
    calleeName = StringRef("llvm.hivm.plt.b16.v300");
  } else if (auto intType = dyn_cast<IntegerType>(vectorElemType)) {
    if (intType.getWidth() == 32)
    {
      calleeName = StringRef("llvm.hivm.plt.b32.v300");
    } else if (intType.getWidth() == 16) {
      calleeName = StringRef("llvm.hivm.plt.b16.v300");
    } else if (intType.getWidth() == 8) {
      calleeName = StringRef("llvm.hivm.plt.b8.v300");
    }
  }
  if (calleeName.empty())
  {
    return failure();
  }

  Type maskType = VectorType::get({256}, rewriter.getI1Type());
  auto funcType =
      rewriter.getFunctionType(TypeRange{i32Type}, TypeRange{maskType, i32Type});
  auto call = rewriter.create<func::CallOp>(loc, calleeName, funcType.getResults(),
                                            ValueRange{laneCountI32});
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
  return call.getResult(0);
}


static FailureOr<StringRef> buildL1CacheLoadCallee(MLIRContext *context,
                                                   Type resultType,
                                                   pto::L1Cache l1cache) {
  std::string elem;
  if (auto intType = dyn_cast<IntegerType>(resultType)) {
    if (intType.getWidth() == 8)
    {
      elem = "s8";
    } else if (intType.getWidth() == 16) {
      elem = "s16";
    } else if (intType.getWidth() == 32) {
      elem = "s32";
    } else if (intType.getWidth() == 64) {
      elem = "s64";
    }
  } else if (resultType.isF16() || resultType.isBF16()) {
    elem = "s16";
  } else if (resultType.isF32()) {
    elem = "s32";
  } else if (resultType.isF64()) {
    elem = "s64";
  } else if (pto::isPTOFloat8Type(resultType) ||
             pto::isPTOHiFloat8Type(resultType)) {
    elem = "s8";
  } else if (pto::isPTOPackedLdgStgVectorType(resultType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(resultType);
    if (totalBits == 16)
    {
      elem = "s16";
    } else if (totalBits == 32) {
      elem = "s32";
    } else if (totalBits == 64) {
      elem = "s64";
    }
  }
  if (elem.empty())
  {
    return failure();
  }
  StringRef l1cacheName =
      l1cache == pto::L1Cache::Cache ? "cache" : "uncache";
  return StringAttr::get(context,
                         "llvm.hivm.ldg." + l1cacheName.str() + "." + elem)
      .getValue();
}

static FailureOr<StringRef> buildL1CacheStoreCallee(MLIRContext *context,
                                                    Type valueType,
                                                    pto::L1Cache l1cache) {
  std::string elem;
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    if (intType.getWidth() == 8)
    {
      elem = "b8";
    } else if (intType.getWidth() == 16) {
      elem = "b16";
    } else if (intType.getWidth() == 32) {
      elem = "b32";
    } else if (intType.getWidth() == 64) {
      elem = "b64";
    }
  } else if (valueType.isF16() || valueType.isBF16()) {
    elem = "b16";
  } else if (valueType.isF32()) {
    elem = "b32";
  } else if (valueType.isF64()) {
    elem = "b64";
  } else if (pto::isPTOFloat8Type(valueType) ||
             pto::isPTOHiFloat8Type(valueType)) {
    elem = "b8";
  } else if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16)
    {
      elem = "b16";
    } else if (totalBits == 32) {
      elem = "b32";
    } else if (totalBits == 64) {
      elem = "b64";
    }
  }
  if (elem.empty())
  {
    return failure();
  }
  StringRef l1cacheName =
      l1cache == pto::L1Cache::Cache ? "cache" : "uncache";
  return StringAttr::get(context,
                         "llvm.hivm.stg." + l1cacheName.str() + "." + elem)
      .getValue();
}



static LogicalResult
materializeDecls(ModuleOp module, ArrayRef<PlannedDecl> plannedDecls,
                 llvm::raw_ostream &diagOS) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  for (const PlannedDecl &decl : plannedDecls) {
    if (func::FuncOp existing = module.lookupSymbol<func::FuncOp>(decl.name)) {
      if (existing.getFunctionType() != decl.type) {
        diagOS << "VPTO LLVM emission failed: conflicting declaration for "
               << decl.name << "\n";
        return failure();
      }
      continue;
    }
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), decl.name, decl.type);
    func.setPrivate();
  }
  return success();
}


class LowerVbitcastOpPattern final
    : public OpConversionPattern<pto::VbitcastOp> {
public:
  explicit LowerVbitcastOpPattern(const TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::VbitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::VbitcastOp op, pto::VbitcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // A vbitcast whose result has no users is a dead noop (Pure). Erase it
    // instead of emitting an LLVM bitcast the device compiler may not lower
    // (e.g. bf16x2 <-> bf16 physical views).
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vbitcast result type");
}
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType,
                                                 adaptor.getInput());
    return success();
  }
};

class LowerPbitcastOpPattern final
    : public OpConversionPattern<pto::PbitcastOp> {
public:
  explicit LowerPbitcastOpPattern(const TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::PbitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::PbitcastOp op, pto::PbitcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert pbitcast result type");
    }
    if (adaptor.getInput().getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "pbitcast expects identical lowered input/result types");
    }
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};


class ConvertVPTOUnrealizedCastOp final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    {
      return rewriter.notifyMatchFailure(op, "expected single-operand single-result cast");
    }
    if (!hasVPTOConvertibleType(op->getOperandTypes()) &&
        !hasVPTOConvertibleType(op->getResultTypes())) {
      return rewriter.notifyMatchFailure(op, "no VPTO convertible types");
    }

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
    {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }

    Value input = adaptor.getOperands().front();
    if (input.getType() != convertedResultType)
    {
      return rewriter.notifyMatchFailure(op, "input type does not match converted result type");
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

class ConvertPtoTileBufAddrOp final
    : public OpConversionPattern<pto::TileBufAddrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::TileBufAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType);
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer result");
    }

    Value input = adaptor.getSrc();
    if (isa<MemRefType>(input.getType())) {
      Value alignedIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
              op.getLoc(), rewriter.getIndexType(), input);
      Value i64 = rewriter.create<arith::IndexCastUIOp>(
          op.getLoc(), rewriter.getI64Type(), alignedIdx);
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, llvmPtrType, i64);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported tilebuf address source");
  }
};

class ConvertPtoDeclareStructOp final
    : public OpConversionPattern<pto::DeclareStructOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::DeclareStructOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto resultType = dyn_cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(op.getS().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected LLVM pointer result type");
    }
    auto structType = cast<pto::StructType>(op.getS().getType());
    Type storageType = getVPTOStructStorageType(structType, rewriter);
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc) {
      return rewriter.notifyMatchFailure(
          op, "expected struct declaration inside a function");
    }

    // A non-entry alloca is a dynamic stack allocation. Keep one stack slot per
    // declaration per function invocation even when the declaration is nested
    // in a loop or a region.
    Value storage;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block &entryBlock = parentFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      Value one = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));
      storage = rewriter.create<LLVM::AllocaOp>(
          op.getLoc(), resultType, storageType, one, /*alignment=*/0);
    }
    rewriter.replaceOp(op, storage);
    return success();
  }
};

class ConvertPtoStructGetOp final
    : public OpConversionPattern<pto::StructGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StructGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }
    FailureOr<Value> address = getVPTOStructFieldAddress(
        rewriter, op.getLoc(), adaptor.getS(),
        cast<pto::StructType>(op.getS().getType()), op.getPath());
    if (failed(address))
    {
      return rewriter.notifyMatchFailure(op, "invalid struct field path");
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, resultType, *address, getNaturalByteAlignment(resultType));
    return success();
  }
};

class ConvertPtoStructSetOp final
    : public OpConversionPattern<pto::StructSetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StructSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> address = getVPTOStructFieldAddress(
        rewriter, op.getLoc(), adaptor.getS(),
        cast<pto::StructType>(op.getS().getType()), op.getPath());
    if (failed(address))
    {
      return rewriter.notifyMatchFailure(op, "invalid struct field path");
    }
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getValue(), *address,
        getNaturalByteAlignment(adaptor.getValue().getType()));
    return success();
  }
};

class ConvertArithSelectOp final : public OpConversionPattern<arith::SelectOp> {
public:
  ConvertArithSelectOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(typeConverter, context,
                                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1)) {
      return rewriter.notifyMatchFailure(
          op, "only scalar i1 conditions supported for VPTO arith.select");
    }

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    Value trueValue = adaptor.getTrueValue();
    Value falseValue = adaptor.getFalseValue();
    if (trueValue.getType() != convertedResultType ||
        falseValue.getType() != convertedResultType) {
      return rewriter.notifyMatchFailure(
          op, "converted true/false values must match result type");
    }

    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, convertedResultType, adaptor.getCondition(), trueValue,
        falseValue);
    return success();
  }
};

class ConvertPtoAddPtrOp final : public OpConversionPattern<pto::AddPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType);
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    auto gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), llvmPtrType,
        normalizeGEPElementTypeForLLVMLowering(
            cast<pto::PtrType>(op.getPtr().getType()).getElementType(),
            rewriter),
        adaptor.getPtr(), ValueRange{offset});
    rewriter.replaceOp(op, gep.getResult());
    return success();
  }
};

class ConvertPtoCastPtrOp final : public OpConversionPattern<pto::CastPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType) {
      return rewriter.notifyMatchFailure(op,
                                         "could not convert castptr result type");
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
      if (isa<MemRefType>(inputType)) {
        Value alignedIdx =
            rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
                op.getLoc(), rewriter.getIndexType(), input);
        Value i64 = rewriter.create<arith::IndexCastUIOp>(
            op.getLoc(), rewriter.getI64Type(), alignedIdx);
        rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, llvmPtrType, i64);
        return success();
      }
      auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(inputType);
      if (!sourcePtrType) {
        return rewriter.notifyMatchFailure(op,
                                           "expected integer, memref, or LLVM pointer input");
      }
      if (sourcePtrType.getAddressSpace() == llvmPtrType.getAddressSpace()) {
        rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmPtrType, input);
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "cross-address-space ptr casts are unsupported");
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

class ConvertPtoLoadScalarOp final
    : public OpConversionPattern<pto::LoadScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType) {
      return rewriter.notifyMatchFailure(op,
                                         "could not convert load_scalar result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             normalizeGEPElementTypeForLLVMLowering(
                                                 convertedValueType, rewriter),
                                             adaptor.getPtr(),
                                             ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, convertedValueType, elemPtr,
        getNaturalByteAlignment(convertedValueType));
    return success();
  }
};

class ConvertPtoStoreScalarOp final
    : public OpConversionPattern<pto::StoreScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             normalizeGEPElementTypeForLLVMLowering(
                                                 adaptor.getValue().getType(),
                                                 rewriter),
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
  ConvertPtoLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                   LoweringState &)
      : OpConversionPattern<pto::PTOLoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::PTOLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType)
    {
      return rewriter.notifyMatchFailure(op, "could not convert load result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             convertedValueType,
                                             adaptor.getPtr(),
                                             ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, convertedValueType, elemPtr,
        getNaturalByteAlignment(convertedValueType));
    return success();
  }

};

static Type getLdgCallResultType(Type valueType, Type convertedValueType,
                                 ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8 || width == 16)
    {
      return rewriter.getI32Type();
    }
    return convertedValueType;
  }
  if (valueType.isF16() || valueType.isBF16() || valueType.isF32())
  {
    return rewriter.getI32Type();
  }
  if (valueType.isF64())
  {
    return rewriter.getI64Type();
  }
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType))
  {
    return rewriter.getI32Type();
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16)
    {
      return rewriter.getI32Type();
    }
    if (totalBits == 32)
    {
      return rewriter.getI32Type();
    }
    if (totalBits == 64)
    {
      return rewriter.getI64Type();
    }
  }
  return convertedValueType;
}

static Value convertLdgCallResult(Location loc, Type valueType,
                                  Type convertedValueType, Value callResult,
                                  ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8 || width == 16) {
      return rewriter.create<arith::TruncIOp>(
          loc, rewriter.getIntegerType(width), callResult);
    }
    return callResult;
  }

  if (valueType.isF16() || valueType.isBF16()) {
    Value payload =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), callResult);
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, payload);
  }
  if (valueType.isF32() || valueType.isF64()) {
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType,
                                            callResult);
  }
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    Value payload =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI8Type(), callResult);
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, payload);
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16) {
      Value trunc = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI16Type(), callResult);
      return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, trunc);
    }
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType,
                                            callResult);
  }
  return callResult;
}

class ConvertPtoLdgOp final : public OpConversionPattern<pto::PTOLdgOp> {
public:
  ConvertPtoLdgOp(TypeConverter &typeConverter, MLIRContext *context,
                  LoweringState &state)
      : OpConversionPattern<pto::PTOLdgOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::PTOLdgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType)
    {
      return rewriter.notifyMatchFailure(op, "could not convert ldg result type");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             normalizeGEPElementTypeForLLVMLowering(
                                                 convertedValueType, rewriter),
                                             adaptor.getPtr(),
                                             ValueRange{offset});
    }

    auto ptrTy = cast<pto::PtrType>(op.getPtr().getType());
    FailureOr<Value> ptr = reinterpretPointerToAddrSpace(
        op, elemPtr,
        static_cast<unsigned>(ptrTy.getMemorySpace().getAddressSpace()));
    if (failed(ptr))
    {
      return rewriter.notifyMatchFailure(op, "failed to map ldg pointer");
    }

    pto::L1Cache l1cache = op.getL1cacheAttr()
                               ? op.getL1cacheAttr().getValue()
                               : pto::L1Cache::Cache;
    FailureOr<StringRef> calleeName = buildL1CacheLoadCallee(
        op.getContext(), op.getValue().getType(), l1cache);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported ldg signature");
    }

    pto::LdL2Cache mode = op.getL2cacheAttr()
                               ? op.getL2cacheAttr().getValue()
                               : pto::LdL2Cache::NMFV;
    Value modeValue =
        getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(mode));
    Type callResultType = getLdgCallResultType(op.getValue().getType(),
                                               convertedValueType, rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{ptr->getType(), rewriter.getI32Type()},
                                 TypeRange{callResultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{callResultType},
        ValueRange{*ptr, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value result = convertLdgCallResult(op.getLoc(), op.getValue().getType(),
                                        convertedValueType, call.getResult(0),
                                        rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

class ConvertPtoStoreOp final : public OpConversionPattern<pto::PTOStoreOp> {
public:
  ConvertPtoStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                    LoweringState &)
      : OpConversionPattern<pto::PTOStoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::PTOStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             adaptor.getValue().getType(),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getValue(), elemPtr,
        getNaturalByteAlignment(adaptor.getValue().getType()));
    return success();
  }

};

static Value convertStgValue(Location loc, Type valueType, Value value,
                             ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8)
    {
      return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), value);
    }
    if (width == 16)
    {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
    }
    return value;
  }

  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    Value payload =
        rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI8Type(), value);
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), payload);
  }
  if (valueType.isBF16())
  {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
  }
  if (valueType.isF32())
  {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
  }
  if (valueType.isF64())
  {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
  }
  if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(),
                                              value);
    }
    if (totalBits == 32) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(),
                                              value);
    }
    if (totalBits == 64) {
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(),
                                              value);
    }
  }
  return value;
}

class ConvertPtoStgOp final : public OpConversionPattern<pto::PTOStgOp> {
public:
  ConvertPtoStgOp(TypeConverter &typeConverter, MLIRContext *context,
                  LoweringState &state)
      : OpConversionPattern<pto::PTOStgOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::PTOStgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);
    }

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             normalizeGEPElementTypeForLLVMLowering(
                                                 adaptor.getValue().getType(),
                                                 rewriter),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    auto ptrTy = cast<pto::PtrType>(op.getPtr().getType());
    FailureOr<Value> ptr = reinterpretPointerToAddrSpace(
        op, elemPtr,
        static_cast<unsigned>(ptrTy.getMemorySpace().getAddressSpace()));
    if (failed(ptr))
    {
      return rewriter.notifyMatchFailure(op, "failed to map stg pointer");
    }

    pto::L1Cache l1cache = op.getL1cacheAttr()
                               ? op.getL1cacheAttr().getValue()
                               : pto::L1Cache::Cache;
    FailureOr<StringRef> calleeName = buildL1CacheStoreCallee(
        op.getContext(), op.getValue().getType(), l1cache);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported stg signature");
    }

    pto::StL2Cache mode = op.getL2cacheAttr()
                               ? op.getL2cacheAttr().getValue()
                               : pto::StL2Cache::NMFV;
    Value modeValue =
        getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(mode));
    Value storedValue = convertStgValue(op.getLoc(), op.getValue().getType(),
                                        adaptor.getValue(), rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{ptr->getType(), storedValue.getType(),
                                           rewriter.getI32Type()},
                                 TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*ptr, storedValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
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

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<pto::CastPtrOp>(op))
    {
      return failure();
    }
    Type propertyType;
    if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(op))
    {
      propertyType = allocaOp.getElemType();
    } else if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
      propertyType = gepOp.getElemType();
    }
    if (!hasVPTOConvertibleType(op->getOperandTypes()) &&
        !hasVPTOConvertibleType(op->getResultTypes()) &&
        !hasVPTOConvertibleType(propertyType)) {
      return failure();
    }
    if (op->getNumRegions() != 0) {
      return rewriter.notifyMatchFailure(
          op, "region ops with VPTO types are handled structurally");
    }

    SmallVector<Type> convertedResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           convertedResultTypes))) {
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
        return rewriter.notifyMatchFailure(
            op, "failed to convert LLVM element type");
      }
      if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(converted))
      {
        allocaOp.setElemType(convertedPropertyType);
      }
      else
      {
        cast<LLVM::GEPOp>(converted).setElemType(convertedPropertyType);
      }
    }
    rewriter.replaceOp(op, converted->getResults());
    return success();
  }
};

static void populateVPTOOpLoweringPatterns(VPTOTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           LoweringState &state,
                                           const std::string &march) {
  populateVPTOCubeMadPatterns(typeConverter, patterns, state);
  populateVPTOCubeMemoryPatterns(typeConverter, patterns, state);
  populateVPTOVectorMemoryPatterns(typeConverter, patterns, state);
  populateVPTOVectorGatherPatterns(typeConverter, patterns, state);

  populateVPTOBasicPatterns(typeConverter, patterns, state);
  populateVPTOVectorUnaryPatterns(typeConverter, patterns, state);
  populateVPTOVectorCompactionPatterns(typeConverter, patterns, state);
  populateVPTOVectorMulaPatterns(typeConverter, patterns, state);
  populateVPTOVectorBinaryPatterns(typeConverter, patterns, state);
  populateVPTOVectorVmullPatterns(typeConverter, patterns, state);
  populateVPTOVectorCarryPatterns(typeConverter, patterns, state);
  populateVPTOVectorReductionPatterns(typeConverter, patterns, state);
  populateVPTOVectorPredicatePatterns(typeConverter, patterns, state);
  populateVPTOUbufPatterns(typeConverter, patterns, state, march);
  populateVPTOScalarAndRuntimePatterns(typeConverter, patterns, state);
  populateVPTOSyncAndConfigPatterns(typeConverter, patterns, state);
  populateVPTOVcvtPatterns(typeConverter, patterns, state);
  patterns.add<LowerVbitcastOpPattern,
               LowerPbitcastOpPattern>(
      typeConverter, patterns.getContext(), state);
}

static void configureVPTOOpLoweringTarget(ConversionTarget &target,
                                          VPTOTypeConverter &typeConverter,
                                          const std::string &march) {
  (void)typeConverter;
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<func::FuncOp>();
  target.addLegalOp<pto::TileBufAddrOp>();
  target.addLegalOp<pto::AddPtrOp>();
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                         LLVM::LLVMDialect,
                          func::FuncDialect, scf::SCFDialect>();
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [](UnrealizedConversionCastOp op) {
        return !hasVPTOConvertibleType(op->getOperandTypes()) &&
               !hasVPTOConvertibleType(op->getResultTypes());
      });
  target.addIllegalOp<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp, pto::SyncSetOp,
                      pto::SyncWaitOp, pto::SetIntraBlockOp, pto::WaitIntraBlockOp,
                      pto::BarrierOp, pto::MemBarOp,
                      pto::CmoCacheInvalidOp, pto::FenceBarrierAllOp,
                      pto::DsbOp, pto::DcciOp,
                      pto::GetBufOp, pto::RlsBufOp,
                      pto::GetBufDynOp, pto::RlsBufDynOp>();
  target.addIllegalOp<pto::GetBlockIdxOp, pto::GetSubBlockIdxOp,
                      pto::GetBlockNumOp, pto::GetSubBlockNumOp,
                      pto::GetCtrlOp, pto::GetVms4SrOp, pto::GetTidXOp,
                      pto::GetTidYOp, pto::GetTidZOp,
                      pto::GetBlockDimXOp, pto::GetBlockDimYOp,
                      pto::GetBlockDimZOp, pto::GetGridDimXOp,
                      pto::GetGridDimYOp, pto::GetGridDimZOp,
                      pto::GetBlockIdxXOp, pto::GetBlockIdxYOp,
                      pto::GetBlockIdxZOp, pto::GetVecCoreIdOp,
                      pto::GetLaneIdOp, pto::GetClock32Op, pto::GetClock64Op,
                      pto::GetLaneMaskEqOp, pto::GetLaneMaskLeOp,
                      pto::GetLaneMaskLtOp, pto::GetLaneMaskGeOp,
                      pto::GetLaneMaskGtOp, pto::VoteAllOp, pto::VoteAnyOp,
                      pto::VoteUniOp, pto::VoteBallotOp, pto::ShuffleIdxOp,
                      pto::ShuffleUpOp, pto::ShuffleDownOp,
                      pto::ShuffleBflyOp, pto::ReduxAddOp, pto::ReduxMaxOp,
                      pto::ReduxMinOp, pto::AtomicCasOp, pto::AtomicExchOp,
                      pto::AtomicAddOp, pto::AtomicSubOp,
                      pto::AtomicMinOp, pto::AtomicMaxOp,
                      pto::AtomicAndOp, pto::AtomicOrOp,
                      pto::AtomicXorOp, pto::TrapOp, pto::PrmtOp,
                      pto::MulhiOp, pto::MulI32ToI64Op, pto::SqrtOp,
                      pto::AbsFOp, pto::ExpOp, pto::LogOp, pto::CeilOp,
                      pto::FloorOp, pto::RintOp, pto::RoundOp, pto::FMinOp,
                      pto::FMaxOp, pto::PowOp, pto::FmaOp, pto::ConvertOp,
                      pto::SyncthreadsOp, pto::ThreadfenceOp,
                      pto::ThreadfenceBlockOp, pto::KeepOp, pto::ResumeOp>();
  target.addIllegalOp<pto::SetLoop2StrideOutToUbOp, pto::SetLoop1StrideOutToUbOp,
                      pto::SetLoopSizeOutToUbOp, pto::SetLoop2StrideUbToOutOp,
                      pto::SetLoop1StrideUbToOutOp, pto::SetLoopSizeUbToOutOp,
                      pto::SetLoop3ParaOp, pto::SetChannelParaOp,
                      pto::SetLoop2StrideOutToL1Op, pto::SetLoop1StrideOutToL1Op,
                      pto::SetLoopSizeOutToL1Op, pto::SetMte2NzParaOp,
                      pto::SetPadValOutToL1Op, pto::SetReluAlphaOp,
                      pto::SetFixClipReluOp, pto::SetFpcOp,
                      pto::SetStoreAtomicCfgOp,
                      pto::SetAtomicS32Op, pto::SetAtomicS8Op, pto::SetCtrlOp,
                      pto::StoreVfSimtInfoOp,
                      pto::SetMovPadValOp, pto::SetQuantPreOp>();
  target.addIllegalOp<pto::Sbitset0Op, pto::Sbitset1Op>();
  target.addIllegalOp<pto::VldsOp, pto::Vldsx2Op, pto::VsldbOp,
                      pto::VldasOp, pto::InitAlignOp, pto::VldusOp,
                      pto::SprclrOp, pto::SprstiOp, pto::SprstsOp,
                      pto::VstsOp, pto::VsstbOp, pto::Vstsx2Op,
                      pto::VstarOp, pto::VstasOp, pto::Vgather2Op,
                      pto::Vgather2BcOp, pto::VgatherbOp, pto::VscatterOp,
                      pto::PldiOp, pto::PldsOp, pto::PstiOp, pto::PstsOp,
                      pto::PstuOp, pto::VstusOp, pto::VsturOp>();
  target.addIllegalOp<pto::PltB8Op, pto::PltB16Op, pto::PltB32Op,
                      pto::PltmB8Op, pto::PltmB16Op, pto::PltmB32Op,
                      pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op,
                      pto::PgeB8Op, pto::PgeB16Op, pto::PgeB32Op>();
  target.addIllegalOp<pto::VabsOp, pto::VexpOp, pto::VlnOp, pto::VnegOp,
                      pto::VsqrtOp, pto::VreluOp, pto::VnotOp,
                      pto::VsqzOp,
                      pto::VusqzOp, pto::VmulaOp, pto::VmullOp, pto::VaddOp,
                      pto::VsubOp, pto::VmulOp,
                      pto::VdivOp, pto::VmaxOp, pto::VminOp, pto::VandOp,
                      pto::VorOp, pto::VxorOp, pto::VmaddOp,
                      pto::VaddcOp, pto::VsubcOp,
                      pto::VaddcsOp, pto::VsubcsOp, pto::VshlOp, pto::VshrOp,
                      pto::VmulsOp, pto::VaddsOp, pto::VmaxsOp,
                      pto::VminsOp, pto::VlreluOp, pto::VshlsOp, pto::VshrsOp,
                      pto::VcaddOp, pto::VcmaxOp, pto::VcminOp,
                      pto::VcgaddOp, pto::VcgmaxOp, pto::VcgminOp, pto::VcpaddOp,
                      pto::Chistv2Op, pto::Dhistv2Op,
                      pto::VcbmaxOp, pto::VcbminOp,
                      pto::VdupOp, pto::VbrOp,
                      pto::PpackOp, pto::PunpackOp, pto::PbitcastOp,
                      pto::VselOp, pto::VselrOp,
                      pto::PnotOp, pto::PselOp, pto::PandOp, pto::PorOp, pto::PxorOp,
                      pto::PdintlvB8Op, pto::PdintlvB16Op, pto::PdintlvB32Op,
                      pto::PintlvB8Op, pto::PintlvB16Op, pto::PintlvB32Op,
                      pto::VsunpackOp, pto::VzunpackOp, pto::VpackOp,
                      pto::VintlvOp, pto::VdintlvOp, pto::VpreluOp,
                      pto::VaxpyOp, pto::VmulscvtOp, pto::VciOp, pto::VexpdifOp,
                      pto::VbitsortOp, pto::Vmrgsort4Op, pto::VtrcOp,
                      pto::VcvtOp,
                      pto::VbitcastOp,
                      pto::VcmpOp, pto::VcmpsOp,
                      pto::CopyGmToUbufOp, pto::CopyUbufToGmOp,
                      pto::CopyUbufToUbufOp, pto::CopyCbufToUbufOp,
                      pto::CopyUbufToCbufOp,
                      pto::CopyGmToCbufOp, pto::CreateCbufMatrixOp,
                      pto::LoadCbufToCaOp,
                      pto::LoadCbufToCbOp, pto::LoadCbufToCaS4Op,
                      pto::LoadCbufToCbS4Op, pto::LoadCbufToCaMxOp,
                      pto::LoadCbufToCbMxOp, pto::CopyMatrixCcToGmOp,
                      pto::CopyMatrixCcToCbufOp, pto::CopyMatrixCcToUbOp,
                      pto::CopyCbufToBtOp, pto::CopyCbufToFbufOp,
                      pto::CopyGmToCbufMultiNd2NzOp,
                      pto::CopyGmToCbufMultiDn2NzOp,
                      pto::MadOp, pto::MadAccOp, pto::MadBiasOp, pto::MadMxOp,
                      pto::MadMxAccOp, pto::MadMxBiasOp,
                      pto::MadRawOp, pto::MadBiasRawOp, pto::MadMxRawOp,
                      pto::MadMxBiasRawOp>();

  if (march == "dav-c220-vec") {
    target.addIllegalOp<pto::UBVaddOp>();
    target.addIllegalOp<pto::UBVsubOp>();
    target.addIllegalOp<pto::UBVmulOp>();
    target.addIllegalOp<pto::UBVdivOp>();
    target.addIllegalOp<pto::UBVmaxOp>();
    target.addIllegalOp<pto::UBVminOp>();
    target.addIllegalOp<pto::UBVandOp>();
    target.addIllegalOp<pto::UBVorOp>();
    target.addIllegalOp<pto::UBVaddReluOp>();
    target.addIllegalOp<pto::UBVnotOp>();
    target.addIllegalOp<pto::UBVabsOp>();
    target.addIllegalOp<pto::UBVreluOp>();
    target.addIllegalOp<pto::UBVexpOp>();
    target.addIllegalOp<pto::UBVlnOp>();
    target.addIllegalOp<pto::UBVsqrtOp>();
    target.addIllegalOp<pto::UBVrsqrtOp>();
    target.addIllegalOp<pto::UBVshlOp>();
    target.addIllegalOp<pto::UBVshrOp>();
    target.addIllegalOp<pto::UBVmulSOp>();
    target.addIllegalOp<pto::UBVaddSOp>();
    target.addIllegalOp<pto::UBVmaxSOp>();
    target.addIllegalOp<pto::UBVminSOp>();
    target.addIllegalOp<pto::UBVdupOp>();
    target.addIllegalOp<pto::UBVgatherbOp>();
    target.addIllegalOp<pto::UBVgatherOp>();
    target.addIllegalOp<pto::UBSetMaskOp>();
    target.addIllegalOp<pto::UBSetMaskCountOp>();
    target.addIllegalOp<pto::UBSetMaskNormOp>();
  }

  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return !isa<pto::TrapOp>(op);
  });
}

static void populateVPTOStructuralTypePatterns(
    VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

static void foldVPTOTypeCasts(ModuleOp module, TypeConverter &typeConverter) {
  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
    {
      return;
    }
    if (!hasVPTOConvertibleType(castOp->getOperandTypes()) &&
        !hasVPTOConvertibleType(castOp->getResultTypes())) {
      return;
    }
    Type convertedResultType =
        typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType &&
        convertedResultType == castOp.getOperand(0).getType()) {
      castsToFold.push_back(castOp);
    }
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }
}

static LogicalResult lowerVPTOOps(ModuleOp module,
                                  const std::string &march,
                                  llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  configureVPTOOpLoweringTarget(target, typeConverter, march);
  populateVPTOOpLoweringPatterns(typeConverter, patterns, state, march);
  patterns.add<ConvertVPTOUnrealizedCastOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO op lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS)))
  {
    return failure();
  }
  return success();
}

static LogicalResult lowerVPTOTypes(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                typeConverter);
      });
  target.addDynamicallyLegalOp<arith::SelectOp>([&](arith::SelectOp op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
  target.addIllegalOp<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp,
                      pto::StoreScalarOp, pto::PTOLoadOp, pto::PTOStoreOp,
                      pto::PTOLdgOp, pto::PTOStgOp, pto::DeclareStructOp,
                      pto::StructGetOp, pto::StructSetOp>();
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [&](UnrealizedConversionCastOp op) {
        return !hasVPTOConvertibleType(op->getOperandTypes()) &&
               !hasVPTOConvertibleType(op->getResultTypes());
      });
  target.addDynamicallyLegalOp<LLVM::AllocaOp>([&](LLVM::AllocaOp op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes()) &&
           typeConverter.isLegal(op.getElemType());
  });
  target.addDynamicallyLegalOp<LLVM::GEPOp>([&](LLVM::GEPOp op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes()) &&
           typeConverter.isLegal(op.getElemType());
  });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  populateVPTOStructuralTypePatterns(typeConverter, patterns, target);
  patterns.add<ConvertPtoTileBufAddrOp, ConvertPtoAddPtrOp, ConvertPtoCastPtrOp,
               ConvertPtoLoadScalarOp, ConvertPtoDeclareStructOp,
               ConvertPtoStructGetOp, ConvertPtoStructSetOp,
               ConvertPtoStoreScalarOp>(typeConverter, context);
  patterns.add<ConvertPtoLoadOp, ConvertPtoStoreOp, ConvertPtoLdgOp,
               ConvertPtoStgOp>(
      typeConverter, context, state);
  patterns.add<ConvertArithSelectOp>(typeConverter, context);
  patterns.add<ConvertVPTOUnrealizedCastOp>(typeConverter, context);
  patterns.add<ConvertVPTOTypedCarrierOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO type lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS)))
  {
    return failure();
  }
  foldVPTOTypeCasts(module, typeConverter);
  return success();
}

static Type normalizeTypeForOfficialLLVMLowering(Type type, Builder &builder) {
  type = convertVPTOType(type, builder);
  return type;
}

static void normalizeFuncSignaturesForOfficialLLVMLowering(ModuleOp module) {
  Builder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;
    bool changed = false;

    for (Type input : oldType.getInputs()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(input, builder);
      changed |= (normalized != input);
      newInputs.push_back(normalized);
    }
    for (Type result : oldType.getResults()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(result, builder);
      changed |= (normalized != result);
      newResults.push_back(normalized);
    }

    if (!changed)
    {
      continue;
    }

    auto newType = builder.getFunctionType(newInputs, newResults);
    funcOp.setFunctionTypeAttr(TypeAttr::get(newType));

    if (funcOp.isExternal())
    {
      continue;
    }
    Block &entry = funcOp.getBody().front();
    for (auto [arg, newType] : llvm::zip(entry.getArguments(), newInputs)) {
      if (arg.getType() != newType) {
        arg.setType(newType);
      }
    }
  }
}

static void forceV300CtrlModeForVPTOFuncs(ModuleOp module) {
  OpBuilder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!needsV300CtrlModeForVPTOFunc(funcOp))
    {
      continue;
    }

    Block &entry = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entry);
    auto i64Type = builder.getI64Type();
    auto bit60 = builder.create<arith::ConstantOp>(
        funcOp.getLoc(), i64Type, builder.getI64IntegerAttr(60));
    Value ctrl =
        builder.create<pto::GetCtrlOp>(funcOp.getLoc(), i64Type).getResult();
    Value ctrlV300 = builder
                         .create<pto::Sbitset0Op>(funcOp.getLoc(), i64Type,
                                                  ctrl, bit60.getResult())
                         .getResult();
    builder.create<pto::SetCtrlOp>(funcOp.getLoc(), ctrlV300);
  }
}

static std::optional<FunctionKernelKind> getKernelKind(ModuleOp module) {
  auto kernelKind = module->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kernelKind)
  {
    return std::nullopt;
  }
  return kernelKind.getKernelKind();
}

static VPTOEmissionOptions
makeDeviceEmissionOptions(const VPTOEmissionOptions &baseOptions,
                          FunctionKernelKind kind) {
  VPTOEmissionOptions options = baseOptions;
  constexpr llvm::StringLiteral kC220VecTargetFeatures =
      "+ASAN,+ATOMIC,+AtomicForB64,+AtomicForB8 ,+FFTSBlk,"
      "+MOVX8,+MSTX,+MathOp,+SPR7bits,+dav-c220-vec";
  constexpr llvm::StringLiteral kC220CubeTargetFeatures =
      "+ASAN,+ATOMIC,+AtomicForB64,+AtomicForB8 ,+FFTSBlk,"
      "+MOVX8,+MSTX,+MathOp,+SPR7bits,+dav-c220-cube";
  constexpr llvm::StringLiteral kVecTargetFeatures =
      "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
      "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
      "+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec";
  constexpr llvm::StringLiteral kCubeTargetFeatures =
      "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
      "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
      "+MOVX8,+SPR7bits,+SyncV,+dav-c310-cube";
  if (options.march.empty()) {
    if (kind == FunctionKernelKind::Vector) {
      options.march = "dav-c310-vec";
      options.aicoreArch = "dav-c310-vec";
      options.defaultTargetCPU = "dav-c310-vec";
      options.defaultTargetFeatures = kVecTargetFeatures.str();
    } else if (kind == FunctionKernelKind::Cube) {
      options.march = "dav-c310-cube";
      options.aicoreArch = "dav-c310-cube";
      options.defaultTargetCPU = "dav-c310-cube";
      options.defaultTargetFeatures = kCubeTargetFeatures.str();
    }
  } else {
    options.aicoreArch = options.march;
    options.defaultTargetCPU = options.march;
    if (options.march == "dav-c220-vec")
    {
      options.defaultTargetFeatures = kC220VecTargetFeatures.str();
    } else if (options.march == "dav-c220-cube") {
      options.defaultTargetFeatures = kC220CubeTargetFeatures.str();
    } else if (kind == FunctionKernelKind::Cube) {
      options.defaultTargetFeatures = kCubeTargetFeatures.str();
    } else {
      options.defaultTargetFeatures = kVecTargetFeatures.str();
    }
  }
  return options;
}

static FailureOr<ModuleOp>
getUniqueDeviceModuleByKernelKind(ModuleOp module, FunctionKernelKind kind,
                                  llvm::raw_ostream &diagOS) {
  ModuleOp matched;
  for (ModuleOp child : module.getOps<ModuleOp>()) {
    auto kernelKind = getKernelKind(child);
    if (!kernelKind)
    {
      continue;
    }
    if (*kernelKind != kind)
    {
      continue;
    }
    if (matched) {
      diagOS << "VPTO LLVM emission failed: duplicate device module with "
             << FunctionKernelKindAttr::name << "\n";
      return failure();
    }
    matched = child;
  }
  return matched;
}

static void mergeDeviceModulesByKernelKind(ModuleOp module) {
  ModuleOp vectorModule;
  ModuleOp cubeModule;
  SmallVector<ModuleOp> modulesToErase;

  for (ModuleOp child : module.getOps<ModuleOp>()) {
    auto kernelKind = getKernelKind(child);
    if (!kernelKind)
    {
      continue;
    }

    ModuleOp *target = nullptr;
    if (*kernelKind == FunctionKernelKind::Vector)
    {
      target = &vectorModule;
    } else if (*kernelKind == FunctionKernelKind::Cube) {
      target = &cubeModule;
    } else {
      continue;
    }

    if (!*target) {
      *target = child;
      continue;
    }

    Block *srcBody = child.getBody();
    Block *dstBody = (*target).getBody();
    while (!srcBody->empty()) {
      Operation &op = srcBody->front();
      op.moveBefore(dstBody, dstBody->end());
    }
    modulesToErase.push_back(child);
  }

  for (ModuleOp child : modulesToErase)
  {
    child.erase();
  }
}

static LogicalResult renameKernelFunctionsForKernelKind(ModuleOp module,
                                                        llvm::raw_ostream &diagOS) {
  auto kernelKind = getKernelKind(module);
  if (!kernelKind) {
    diagOS << "VPTO LLVM emission failed: device module missing "
           << FunctionKernelKindAttr::name << "\n";
    return failure();
  }

  StringRef suffix;
  if (*kernelKind == FunctionKernelKind::Vector)
  {
    suffix = kVectorSuffix;
  } else if (*kernelKind == FunctionKernelKind::Cube) {
    suffix = kCubeSuffix;
  } else {
    diagOS << "VPTO LLVM emission failed: unsupported "
           << FunctionKernelKindAttr::name << "\n";
    return failure();
  }

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!pto::hasExplicitPTOEntryAttr(funcOp))
    {
      continue;
    }
    if (funcOp.getSymName().ends_with(suffix))
    {
      continue;
    }
    funcOp.setSymName((funcOp.getSymName() + suffix).str());
  }
  return success();
}

struct LowerVPTOOpsPass final
    : public PassWrapper<LowerVPTOOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVPTOOpsPass)

  LowerVPTOOpsPass() = default;
  explicit LowerVPTOOpsPass(std::string m) : march(std::move(m)) {}

  void runOnOperation() override {
    materializeVecScopeCarrierLoops(getOperation());
    // Remove dead pto.alloc_tile ops before lowering. These can appear when
    // the original kernel's tile_buf intrinsics have already been folded away
    // by FoldTileBufIntrinsics, but a subsequent pass (e.g. AIC-scope cloning)
    // re-introduces alloc_tile copies whose results have no users. The lowering
    // patterns do not cover AllocTileOp, so leaving them in the IR causes
    // translateModuleToLLVMIR to fail.
    {
      SmallVector<pto::AllocTileOp> deadAllocs;
      getOperation().walk([&](pto::AllocTileOp alloc) {
        if (alloc.use_empty())
        {
          deadAllocs.push_back(alloc);
        }
      });
      for (pto::AllocTileOp alloc : llvm::reverse(deadAllocs))
      {
        alloc.erase();
      }
    }
    if (failed(lowerVPTOOps(getOperation(), march, llvm::errs())))
    {
      signalPassFailure();
    }
  }

private:
  std::string march;
};

struct LowerVPTOTypesPass final
    : public PassWrapper<LowerVPTOTypesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVPTOTypesPass)

  void runOnOperation() override {
    if (failed(lowerVPTOTypes(getOperation(), llvm::errs())))
    {
      signalPassFailure();
    }
  }
};

struct NormalizeFuncSignaturesForLLVMLoweringPass final
    : public PassWrapper<NormalizeFuncSignaturesForLLVMLoweringPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NormalizeFuncSignaturesForLLVMLoweringPass)

  void runOnOperation() override {
    normalizeFuncSignaturesForOfficialLLVMLowering(getOperation());
  }
};

struct PrepareVPTOLLVMLoweringPass final
    : public PassWrapper<PrepareVPTOLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareVPTOLLVMLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    pto::annotatePTOEntryFunctions(module);
    forceV300CtrlModeForVPTOFuncs(module);
    if (failed(renameKernelFunctionsForKernelKind(module, llvm::errs())))
    {
      signalPassFailure();
    }
  }
};

static llvm::StringSet<llvm::BumpPtrAllocator>
collectSimtEntryFunctionNames(ModuleOp module) {
  llvm::StringSet<llvm::BumpPtrAllocator> simtEntries;
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp->hasAttr(pto::kPTOSimtEntryAttrName))
    {
      simtEntries.insert(funcOp.getSymName());
    }
  });
  return simtEntries;
}

static void applyArtifactVisibilityLinkage(ModuleOp sourceModule,
                                           llvm::Module &llvmModule) {
  llvm::StringMap<bool> externalByName;
  sourceModule.walk([&](func::FuncOp funcOp) {
    if (funcOp.isDeclaration())
    {
      return;
    }
    externalByName[funcOp.getSymName()] =
        pto::hasExternalArtifactVisibility(funcOp);
  });

  for (llvm::Function &function : llvmModule) {
    auto it = externalByName.find(function.getName());
    if (it == externalByName.end())
    {
      continue;
    }
    if (it->second) {
      function.setLinkage(llvm::GlobalValue::ExternalLinkage);
      continue;
    }
    function.setLinkage(llvm::GlobalValue::InternalLinkage);
  }
}

static void applySimtEntryCallingConvention(
    llvm::Module &llvmModule,
    const llvm::StringSet<llvm::BumpPtrAllocator> &simtEntryNames) {
  for (llvm::Function &function : llvmModule) {
    if (simtEntryNames.contains(function.getName())) {
      function.setCallingConv(llvm::CallingConv::SimtEntry);
      function.addFnAttr(llvm::Attribute::NoInline);
      // Match Bisheng's C++ frontend shape for SIMT outlined bodies. The
      // exported wrapper owns the real kernel metadata, while the SIMT body is
      // an ODR helper called with the SIMT calling convention. In CANN beta.1,
      // leaving the SIMT body as a strong GLOBAL FUNC makes the runtime count it
      // as an extra kernel without matching .ascend.meta, which can corrupt the
      // selected kernel metadata. linkonce_odr lowers to a weak helper symbol
      // and avoids that beta.1 metadata mismatch.
      function.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    }
  }

  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call)
        {
          continue;
        }
        auto *callee = call->getCalledFunction();
        if (!callee || !simtEntryNames.contains(callee->getName()))
        {
          continue;
        }
        call->setCallingConv(llvm::CallingConv::SimtEntry);
      }
    }
  }
}

static FailureOr<EmittedLLVMModule>
emitDeviceLLVMModule(ModuleOp deviceModule, StringRef kernelKind,
                     const VPTOEmissionOptions &options,
                     const llvm::StringSet<llvm::BumpPtrAllocator> &simtEntryNames,
                     llvm::raw_ostream &diagOS) {
  if (!deviceModule) {
    return EmittedLLVMModule{};
  }
  if (failed(applyQueriedTargetAttrs(deviceModule, options, diagOS)))
  {
    return failure();
  }

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  registerBuiltinDialectTranslation(*deviceModule.getContext());
  registerLLVMDialectTranslation(*deviceModule.getContext());
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(deviceModule.getOperation(), *llvmContext);
  if (!llvmModule) {
    diagOS << "VPTO LLVM emission failed: LLVM IR export failed for "
           << kernelKind << " module\n";
    return failure();
  }

  applyArtifactVisibilityLinkage(deviceModule, *llvmModule);
  for (llvm::Function &func : *llvmModule) {
    if (!func.getName().starts_with("llvm.hivm.vscatter."))
    {
      continue;
    }
    // Work around a bug in older Bisheng releases: vscatter was not modeled
    // as writing through its destination pointer, so EarlyCSE could eliminate
    // a load after vscatter as redundant.
    func.setOnlyAccessesArgMemory();
    func.addFnAttr(llvm::Attribute::NoUnwind);
    func.addFnAttr(llvm::Attribute::WriteOnly);
  }
  applySimtEntryCallingConvention(*llvmModule, simtEntryNames);
  if (failed(attachAIVectorScopeMetadata(*llvmModule, diagOS)))
  {
    return failure();
  }
  attachHIVMKernelAnnotations(*llvmModule, deviceModule);
  llvmModule->setModuleIdentifier(("ptoas.hivm.official." + kernelKind).str());
  llvmModule->setSourceFileName(("ptoas.hivm.official." + kernelKind).str());
  return EmittedLLVMModule{std::move(llvmContext), std::move(llvmModule)};
}

template <typename EmitFn>
static LogicalResult runPipeline(ModuleOp module, const std::string &march,
                                 llvm::raw_ostream &diagOS,
                                 EmitFn &&emit) {
  OwningOpRef<Operation *> clonedOp(module->clone());
  ModuleOp clonedModule = cast<ModuleOp>(*clonedOp);

  mergeDeviceModulesByKernelKind(clonedModule);

  if (failed(validateVPTOAuthoringIR(clonedModule, &diagOS))) {
    diagOS << "VPTO LLVM emission failed: authoring-stage VPTO legality "
              "validation failed\n";
    return failure();
  }

  PassManager pm(clonedModule.getContext());
  pm.enableVerifier();
  auto &kernelModulePM = pm.nest<ModuleOp>();
  kernelModulePM.addPass(std::make_unique<PrepareVPTOLLVMLoweringPass>());
  kernelModulePM.addPass(std::make_unique<LowerVPTOOpsPass>(march));
  kernelModulePM.addPass(std::make_unique<LowerVPTOTypesPass>());
  kernelModulePM.addPass(
      std::make_unique<NormalizeFuncSignaturesForLLVMLoweringPass>());
  kernelModulePM.addPass(arith::createArithExpandOpsPass());
  // pto-convert-scf-to-cf-with-loop-hints performs the SCF-to-CF conversion for this pipeline:
  // it runs the upstream conversion patterns plus a higher-benefit lowering
  // for {pto.unroll = "enable"} loops that attaches llvm.loop_annotation to
  // the latch, so the !llvm.loop.unroll.enable metadata survives into the
  // emitted LLVM IR.  It replaces createConvertSCFToCFPass here; running both
  // would be redundant.
  kernelModulePM.addNestedPass<func::FuncOp>(pto::createPTOConvertSCFToCFWithLoopHintsPass());
  kernelModulePM.addPass(createArithToLLVMConversionPass());
  kernelModulePM.addPass(createConvertIndexToLLVMPass());
  kernelModulePM.addPass(createFinalizeMemRefToLLVMConversionPass());
  kernelModulePM.addPass(createConvertFuncToLLVMPass());
  kernelModulePM.addPass(createConvertControlFlowToLLVMPass());
  kernelModulePM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(mlir::applyPassManagerCLOptions(pm))) {
    diagOS << "VPTO LLVM emission failed: unable to apply MLIR pass manager "
              "command-line options\n";
    return failure();
  }
  if (failed(pm.run(clonedModule))) {
    diagOS << "VPTO LLVM emission failed: official lowering pipeline failed\n";
    return failure();
  }
  return emit(clonedModule);
}

} // namespace

LogicalResult lowerVPTOModuleToLLVMModulesBeta1(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS) {
  llvm::StringSet<llvm::BumpPtrAllocator> simtEntryNames =
      collectSimtEntryFunctionNames(module);
  cubeModule.context.reset();
  cubeModule.module.reset();
  vectorModule.context.reset();
  vectorModule.module.reset();
  return runPipeline(module, options.march, diagOS,
                     [&](ModuleOp loweredModule) {
    auto vectorDeviceModule =
        getUniqueDeviceModuleByKernelKind(
            loweredModule, FunctionKernelKind::Vector, diagOS);
    if (failed(vectorDeviceModule))
    {
      return failure();
    }
    auto cubeDeviceModule =
        getUniqueDeviceModuleByKernelKind(
            loweredModule, FunctionKernelKind::Cube, diagOS);
    if (failed(cubeDeviceModule))
    {
      return failure();
    }

    if (*vectorDeviceModule) {
      auto vectorOptions =
          makeDeviceEmissionOptions(options, FunctionKernelKind::Vector);
      auto emitted =
          emitDeviceLLVMModule(*vectorDeviceModule, "vector", vectorOptions,
                               simtEntryNames, diagOS);
      if (failed(emitted))
      {
        return failure();
      }
      vectorModule.context = std::move(emitted->context);
      vectorModule.module = std::move(emitted->module);
    }
    if (*cubeDeviceModule) {
      auto cubeOptions =
          makeDeviceEmissionOptions(options, FunctionKernelKind::Cube);
      auto emitted =
          emitDeviceLLVMModule(*cubeDeviceModule, "cube", cubeOptions,
                               simtEntryNames, diagOS);
      if (failed(emitted))
      {
        return failure();
      }
      cubeModule.context = std::move(emitted->context);
      cubeModule.module = std::move(emitted->module);
    }
    return success();
                     });
}


} // namespace mlir::pto
