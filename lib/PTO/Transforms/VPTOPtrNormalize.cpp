// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOPTRNORMALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static pto::AddressSpaceAttr getPointerMemorySpace(Attribute memorySpace,
                                                   MLIRContext *ctx) {
  if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    return addrSpace;
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    return pto::AddressSpaceAttr::get(
        ctx, static_cast<pto::AddressSpace>(intAttr.getInt()));
  }
  return {};
}

static bool needsSubviewPtrConversion(memref::SubViewOp op) {
  auto resultType = dyn_cast<MemRefType>(op.getType());
  if (!resultType) {
    return false;
  }
  return static_cast<bool>(
      getPointerMemorySpace(resultType.getMemorySpace(), op.getContext()));
}

static Type convertSubviewResultType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType) {
    return type;
  }

  auto memorySpace =
      getPointerMemorySpace(memrefType.getMemorySpace(), type.getContext());
  if (!memorySpace) {
    return type;
  }

  return pto::PtrType::get(type.getContext(), memrefType.getElementType(),
                           memorySpace);
}

static bool hasPtrNormalizeConvertibleType(Type type) {
  if (isa<pto::PtrType>(type)) {
    return true;
  }
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && static_cast<bool>(getPointerMemorySpace(
                           memrefType.getMemorySpace(), type.getContext()));
}

static bool hasPtrNormalizeConvertibleType(TypeRange types) {
  return llvm::any_of(
      types, [](Type type) { return hasPtrNormalizeConvertibleType(type); });
}

static bool isPtrNormalizeMemRefType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && static_cast<bool>(getPointerMemorySpace(
                           memrefType.getMemorySpace(), type.getContext()));
}

static bool hasPtrNormalizeMemRefType(TypeRange types) {
  return llvm::any_of(
      types, [](Type type) { return isPtrNormalizeMemRefType(type); });
}

static bool isTransientPtrMemRefBridge(Value value) {
  if (!isa<pto::PtrType>(value.getType())) {
    return false;
  }
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  return cast && cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
         isa<BaseMemRefType>(cast.getOperand(0).getType());
}

static FailureOr<SmallVector<Type>>
convertTypes(const TypeConverter &typeConverter, TypeRange types) {
  SmallVector<Type> convertedTypes;
  if (failed(typeConverter.convertTypes(types, convertedTypes))) {
    return failure();
  }
  return convertedTypes;
}

static bool isMemRefType(Type type) { return isa<BaseMemRefType>(type); }

static Value materializeUnrealizedCast(OpBuilder &builder, Type resultType,
                                       ValueRange inputs, Location loc) {
  if (inputs.size() != 1) {
    return {};
  }
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
      .getResult(0);
}

static LogicalResult accumulateSubviewElementOffsets(
    ArrayRef<int64_t> staticOffsets, ValueRange dynamicOffsets,
    ArrayRef<int64_t> strides, PatternRewriter &rewriter, Location loc,
    Value &total) {
  unsigned dynamicIndex = 0;
  for (auto [staticOffset, stride] : llvm::zip(staticOffsets, strides)) {
    if (stride == 0) {
      continue;
    }
    if (stride == ShapedType::kDynamic) {
      return failure();
    }

    Value index;
    if (ShapedType::isDynamic(staticOffset)) {
      if (dynamicIndex >= dynamicOffsets.size()) {
        return failure();
      }
      index = dynamicOffsets[dynamicIndex++];
    } else {
      index = rewriter.create<arith::ConstantIndexOp>(loc, staticOffset);
    }
    if (!index.getType().isIndex()) {
      return failure();
    }
    if (stride != 1) {
      Value strideValue = rewriter.create<arith::ConstantIndexOp>(loc, stride);
      index = rewriter.create<arith::MulIOp>(loc, index, strideValue);
    }
    total = rewriter.create<arith::AddIOp>(loc, total, index);
  }
  return dynamicIndex == dynamicOffsets.size() ? success() : failure();
}

static LogicalResult computeSubviewElementOffset(memref::SubViewOp op,
                                                 PatternRewriter &rewriter,
                                                 Value &offset) {
  auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
  if (!sourceType) {
    return failure();
  }

  SmallVector<int64_t> strides;
  int64_t baseOffset = 0;
  if (failed(mlir::pto::getPTOMemRefStridesAndOffset(sourceType, strides,
                                                     baseOffset))) {
    return failure();
  }
  // The SSA source already names the base address after ptr-boundary
  // normalization. A dynamic memref layout offset here is metadata we can
  // ignore for ptr normalization and model as zero.
  if (baseOffset == ShapedType::kDynamic) {
    baseOffset = 0;
  }

  Location loc = op.getLoc();
  Value total = rewriter.create<arith::ConstantIndexOp>(loc, baseOffset);
  ArrayRef<int64_t> staticOffsets = op.getStaticOffsets();
  ValueRange dynamicOffsets = op.getOffsets();
  if (staticOffsets.size() != strides.size()) {
    return failure();
  }

  if (failed(accumulateSubviewElementOffsets(staticOffsets, dynamicOffsets,
                                             strides, rewriter, loc, total))) {
    return failure();
  }

  offset = total;
  return success();
}

static Value materializeSubviewInputPtr(Value source, PatternRewriter &rewriter,
                                        Location loc) {
  if (!source) {
    return {};
  }
  if (isa<pto::PtrType>(source.getType())) {
    return source;
  }

  auto memrefType = dyn_cast<MemRefType>(source.getType());
  if (!memrefType) {
    return {};
  }

  auto memorySpace =
      getPointerMemorySpace(memrefType.getMemorySpace(), rewriter.getContext());
  if (!memorySpace) {
    return {};
  }

  auto ptrType = pto::PtrType::get(rewriter.getContext(),
                                   memrefType.getElementType(), memorySpace);
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, source);
}

static Value materializeScalarAccessPtr(Value source, PatternRewriter &rewriter,
                                        Location loc);

static Value castPtrToResultType(Value pointer, Type resultType,
                                 PatternRewriter &rewriter, Location loc) {
  if (pointer.getType() == resultType) {
    return pointer;
  }
  return rewriter.create<pto::CastPtrOp>(loc, resultType, pointer);
}

static Value materializeReinterpretCastPtr(memref::ReinterpretCastOp reinterpret,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  auto ptrType = dyn_cast<pto::PtrType>(
      convertSubviewResultType(reinterpret.getType()));
  if (!ptrType) {
    return {};
  }
  Value basePtr = materializeScalarAccessPtr(reinterpret.getSource(), rewriter, loc);
  if (!basePtr) {
    return {};
  }
  basePtr = castPtrToResultType(basePtr, ptrType, rewriter, loc);

  ArrayRef<int64_t> staticOffsets = reinterpret.getStaticOffsets();
  if (staticOffsets.size() != 1) {
    return {};
  }
  int64_t staticOffset = staticOffsets.front();
  if (!ShapedType::isDynamic(staticOffset)) {
    if (staticOffset == 0) {
      return basePtr;
    }
    Value offset = rewriter.create<arith::ConstantIndexOp>(loc, staticOffset);
    return rewriter.create<pto::AddPtrOp>(loc, ptrType, basePtr, offset);
  }

  ValueRange dynamicOffsets = reinterpret.getOffsets();
  if (dynamicOffsets.size() != 1 || !dynamicOffsets.front().getType().isIndex()) {
    return {};
  }
  return rewriter.create<pto::AddPtrOp>(loc, ptrType, basePtr,
                                        dynamicOffsets.front());
}

static Value materializeSubviewPtr(memref::SubViewOp subview,
                                   PatternRewriter &rewriter, Location loc) {
  if (!needsSubviewPtrConversion(subview)) {
    return {};
  }
  Value basePtr = materializeScalarAccessPtr(subview.getSource(), rewriter, loc);
  if (!basePtr) {
    return {};
  }
  Value offset;
  if (failed(computeSubviewElementOffset(subview, rewriter, offset))) {
    return {};
  }
  auto ptrType = dyn_cast<pto::PtrType>(convertSubviewResultType(subview.getType()));
  if (!ptrType) {
    return {};
  }
  basePtr = castPtrToResultType(basePtr, ptrType, rewriter, loc);
  return rewriter.create<pto::AddPtrOp>(loc, ptrType, basePtr, offset);
}

static Value materializeScalarAccessPtr(Value source, PatternRewriter &rewriter,
                                        Location loc) {
  if (!source) {
    return {};
  }
  if (auto cast = source.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
      return {};
    }
    Value ptr = materializeScalarAccessPtr(cast.getOperands().front(), rewriter, loc);
    if (!ptr) {
      return {};
    }
    auto resultType = dyn_cast<pto::PtrType>(source.getType());
    return resultType ? castPtrToResultType(ptr, resultType, rewriter, loc) : ptr;
  }
  if (isa<pto::PtrType>(source.getType())) {
    return source;
  }
  if (auto cast = source.getDefiningOp<memref::CastOp>()) {
    return materializeScalarAccessPtr(cast.getSource(), rewriter, loc);
  }

  if (auto reinterpret = source.getDefiningOp<memref::ReinterpretCastOp>()) {
    return materializeReinterpretCastPtr(reinterpret, rewriter, loc);
  }

  if (auto subview = source.getDefiningOp<memref::SubViewOp>()) {
    return materializeSubviewPtr(subview, rewriter, loc);
  }

  // Restrict normalization to memref views that already sit on top of a ptr-like
  // boundary bridge. Materializing fresh memref -> ptr casts here would leave
  // illegal pto.castptr(memref) behind in this pass.
  return {};
}

static Value materializeBoundaryOperandPtr(Value source,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  if (!source) {
    return {};
  }
  if (isa<pto::PtrType>(source.getType())) {
    return source;
  }
  return materializeScalarAccessPtr(source, rewriter, loc);
}

template <typename OpTy>
static LogicalResult rewriteBufferLikeBoundaryOp(
    OpTy op, typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    StringRef sourceRole, StringRef destinationRole) {
  Value source =
      materializeBoundaryOperandPtr(adaptor.getOperands()[0], rewriter, op.getLoc());
  if (!source) {
    return rewriter.notifyMatchFailure(
        op, (Twine("failed to materialize ") + sourceRole + " ptr").str());
  }
  if (!isa<pto::PtrType>(source.getType())) {
    return rewriter.notifyMatchFailure(
        op, (Twine("expected ptr-form ") + sourceRole).str());
  }

  Value destination = materializeBoundaryOperandPtr(adaptor.getOperands()[1],
                                                    rewriter, op.getLoc());
  if (!destination) {
    return rewriter.notifyMatchFailure(
        op, (Twine("failed to materialize ") + destinationRole + " ptr").str());
  }
  if (!isa<pto::PtrType>(destination.getType())) {
    return rewriter.notifyMatchFailure(
        op, (Twine("expected ptr-form ") + destinationRole).str());
  }

  SmallVector<Value> operands(adaptor.getOperands().begin(),
                              adaptor.getOperands().end());
  operands[0] = source;
  operands[1] = destination;

  OperationState state(op.getLoc(), op->getName().getStringRef());
  state.addOperands(operands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  state.propertiesAttr = op->getPropertiesAsAttribute();
  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

struct ConvertTileBufAddrToPtrPattern
    : public OpConversionPattern<pto::TileBufAddrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::TileBufAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getDst().getType());
    if (!isa<pto::PtrType>(convertedType)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<pto::TileBufAddrOp>(op, convertedType,
                                                    adaptor.getSrc());
    return success();
  }
};

struct ConvertIntToPtrToCastPtrPattern
    : public OpConversionPattern<pto::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<pto::PtrType>(convertedType)) {
      return rewriter.notifyMatchFailure(op, "expected pointer result type");
    }

    rewriter.replaceOpWithNewOp<pto::CastPtrOp>(op, convertedType,
                                                adaptor.getAddr());
    return success();
  }
};

struct ConvertPtrToIntToCastPtrPattern
    : public OpConversionPattern<pto::PtrToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<IntegerType>(convertedType)) {
      return rewriter.notifyMatchFailure(op, "expected integer result type");
    }
    if (!isa<pto::PtrType>(adaptor.getPtr().getType())) {
      return rewriter.notifyMatchFailure(op, "expected pointer input type");
    }

    rewriter.replaceOpWithNewOp<pto::CastPtrOp>(op, convertedType,
                                                adaptor.getPtr());
    return success();
  }
};

struct ConvertCastPtrPattern : public OpConversionPattern<pto::CastPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType) {
      return failure();
    }

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    if (isMemRefType(inputType) || isMemRefType(convertedResultType)) {
      return rewriter.notifyMatchFailure(op,
                                         "memref castptr must be eliminated");
    }

    if (!isa<pto::PtrType, IntegerType>(inputType) ||
        !isa<pto::PtrType, IntegerType>(convertedResultType)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected ptr/int castptr operands");
    }

    if (inputType == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    rewriter.replaceOpWithNewOp<pto::CastPtrOp>(op, convertedResultType, input);
    return success();
  }
};

struct ConvertSubviewToAddPtrPattern
    : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsSubviewPtrConversion(op)) {
      return failure();
    }

    auto ptrType =
        dyn_cast<pto::PtrType>(getTypeConverter()->convertType(op.getType()));
    if (!ptrType) {
      return rewriter.notifyMatchFailure(op, "expected ptr result type");
    }

    Value basePtr =
        materializeSubviewInputPtr(adaptor.getSource(), rewriter, op.getLoc());
    if (!basePtr) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize subview input ptr");
    }

    Value offset;
    if (failed(computeSubviewElementOffset(op, rewriter, offset))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute subview element offset");
    }

    rewriter.replaceOpWithNewOp<pto::AddPtrOp>(op, ptrType, basePtr, offset);
    return success();
  }
};

struct ConvertVldsSubviewOperandPattern : public OpConversionPattern<pto::VldsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::VldsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<pto::PtrType>(adaptor.getSource().getType())) {
      return failure();
    }

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands({adaptor.getSource(), adaptor.getOffset()});
    FailureOr<SmallVector<Type>> resultTypes =
        convertTypes(*getTypeConverter(), op->getResultTypes());
    if (failed(resultTypes)) {
      return failure();
    }
    state.addTypes(*resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertVstsSubviewOperandPattern : public OpConversionPattern<pto::VstsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::VstsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<pto::PtrType>(adaptor.getDestination().getType())) {
      return failure();
    }

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands(
        {adaptor.getValue(), adaptor.getDestination(), adaptor.getOffset(),
         adaptor.getMask()});
    FailureOr<SmallVector<Type>> resultTypes =
        convertTypes(*getTypeConverter(), op->getResultTypes());
    if (failed(resultTypes)) {
      return failure();
    }
    state.addTypes(*resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertVsstbSubviewOperandPattern
    : public OpConversionPattern<pto::VsstbOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::VsstbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<pto::PtrType>(adaptor.getDestination().getType())) {
      return failure();
    }

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands({adaptor.getValue(), adaptor.getDestination(),
                       adaptor.getBlockStride(), adaptor.getRepeatStride(),
                       adaptor.getMask()});
    FailureOr<SmallVector<Type>> resultTypes =
        convertTypes(*getTypeConverter(), op->getResultTypes());
    if (failed(resultTypes)) {
      return failure();
    }
    state.addTypes(*resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertLoadScalarOperandToPtrPattern
    : public OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = materializeScalarAccessPtr(adaptor.getPtr(), rewriter, op.getLoc());
    if (!ptr) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize load_scalar ptr");
    }
    if (!isa<pto::PtrType>(ptr.getType())) {
      return rewriter.notifyMatchFailure(op, "expected ptr-form load_scalar input");
    }

    rewriter.replaceOpWithNewOp<pto::LoadScalarOp>(op, op.getValue().getType(),
                                                   ptr, adaptor.getOffset());
    return success();
  }
};

struct ConvertStoreScalarOperandToPtrPattern
    : public OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = materializeScalarAccessPtr(adaptor.getPtr(), rewriter, op.getLoc());
    if (!ptr) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize store_scalar ptr");
    }
    if (!isa<pto::PtrType>(ptr.getType())) {
      return rewriter.notifyMatchFailure(op, "expected ptr-form store_scalar input");
    }

    rewriter.replaceOpWithNewOp<pto::StoreScalarOp>(op, ptr,
                                                    adaptor.getOffset(),
                                                    adaptor.getValue());
    return success();
  }
};

struct ConvertMteUbUbOperandPattern
    : public OpConversionPattern<pto::MteUbUbOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteUbUbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_ub_ub source",
                                       "mte_ub_ub destination");
  }
};

struct ConvertMteUbL1OperandPattern
    : public OpConversionPattern<pto::MteUbL1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteUbL1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_ub_l1 source",
                                       "mte_ub_l1 destination");
  }
};

struct ConvertCubeLoadOperandPattern
    : public OpConversionPattern<pto::MteGmL1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteGmL1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_gm_l1 source",
                                       "mte_gm_l1 destination");
  }
};

struct ConvertRawFillL1OperandPattern
    : public OpConversionPattern<pto::RawFillL1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::RawFillL1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dst =
        materializeBoundaryOperandPtr(adaptor.getDst(), rewriter, op.getLoc());
    if (!dst) {
      return rewriter.notifyMatchFailure(op, "failed to materialize dst ptr");
    }
    if (!isa<pto::PtrType>(dst.getType())) {
      return rewriter.notifyMatchFailure(op, "expected ptr-form dst");
    }

    SmallVector<Value> operands(adaptor.getOperands().begin(),
                                adaptor.getOperands().end());
    operands[0] = dst;

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    state.propertiesAttr = op->getPropertiesAsAttribute();
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertCubeStoreOperandPattern
    : public OpConversionPattern<pto::MteL1UbOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL1UbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "mte_l1_ub source",
                                       "mte_l1_ub destination");
  }
};

struct ConvertBiasLoadOperandPattern
    : public OpConversionPattern<pto::MteL1BtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL1BtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_l1_bt source",
                                       "mte_l1_bt destination");
  }
};

struct ConvertCubeLoadFracOperandPattern
    : public OpConversionPattern<pto::MteGmL1FracOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteGmL1FracOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_gm_l1_frac source",
                                       "mte_gm_l1_frac destination");
  }
};

struct ConvertLeftLoadOperandPattern
    : public OpConversionPattern<pto::MteL1L0aOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL1L0aOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_l1_l0a source",
                                       "mte_l1_l0a destination");
  }
};

struct ConvertRightLoadOperandPattern
    : public OpConversionPattern<pto::MteL1L0bOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL1L0bOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "mte_l1_l0b source",
                                       "mte_l1_l0b destination");
  }
};

struct ConvertLeftLoadMxOperandPattern
    : public OpConversionPattern<pto::MteL1L0aMxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL1L0aMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "mte_l1_l0a_mx source",
                                       "mte_l1_l0a_mx destination");
  }
};

struct ConvertRightLoadMxOperandPattern
    : public OpConversionPattern<pto::MteL1L0bMxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL1L0bMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "mte_l1_l0b_mx source",
                                       "mte_l1_l0b_mx destination");
  }
};

struct ConvertAccStoreOperandPattern
    : public OpConversionPattern<pto::MteL0cL1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL0cL1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_l0c_l1 source",
                                       "mte_l0c_l1 destination");
  }
};

struct ConvertAccStoreGmOperandPattern
    : public OpConversionPattern<pto::MteL0cGmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL0cGmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "mte_l0c_gm source",
                                       "mte_l0c_gm destination");
  }
};

struct ConvertAccStoreUbOperandPattern
    : public OpConversionPattern<pto::MteL0cUbOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteL0cUbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "mte_l0c_ub source",
                                       "mte_l0c_ub destination");
  }
};

struct ConvertMteUbGmOperandPattern
    : public OpConversionPattern<pto::MteUbGmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::MteUbGmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "mte_ub_gm source",
                                       "mte_ub_gm destination");
  }
};

struct ConvertLoadOperandToPtrPattern : public OpConversionPattern<pto::PTOLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::PTOLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = materializeScalarAccessPtr(adaptor.getPtr(), rewriter, op.getLoc());
    if (!ptr) {
      return rewriter.notifyMatchFailure(op, "failed to materialize load ptr");
    }
    if (!isa<pto::PtrType>(ptr.getType())) {
      return rewriter.notifyMatchFailure(op, "expected ptr-form load input");
    }

    rewriter.replaceOpWithNewOp<pto::PTOLoadOp>(
        op, op.getValue().getType(), ptr, adaptor.getOffset());
    return success();
  }
};

struct ConvertStoreOperandToPtrPattern
    : public OpConversionPattern<pto::PTOStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::PTOStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = materializeScalarAccessPtr(adaptor.getPtr(), rewriter, op.getLoc());
    if (!ptr) {
      return rewriter.notifyMatchFailure(op, "failed to materialize store ptr");
    }
    if (!isa<pto::PtrType>(ptr.getType())) {
      return rewriter.notifyMatchFailure(op, "expected ptr-form store input");
    }

    rewriter.replaceOpWithNewOp<pto::PTOStoreOp>(
        op, ptr, adaptor.getOffset(), adaptor.getValue());
    return success();
  }
};

struct ConvertSimtLaunchOp final : public OpConversionPattern<pto::SimtLaunchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::SimtLaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getDimX(), adaptor.getDimY(),
                                adaptor.getDimZ()};
    for (auto [originalArg, convertedArg] :
         llvm::zip(op.getArgs(), adaptor.getArgs())) {
      Value arg = convertedArg;
      if (auto ptrType = dyn_cast<pto::PtrType>(convertedArg.getType())) {
        // Function signature conversion may materialize a ptr -> memref -> ptr
        // bridge for a Tile-derived launch argument. Normalize from the original
        // view so the temporary memref.reinterpret_cast does not leak to VPTO
        // emission validation.
        if (Value normalized =
                materializeScalarAccessPtr(originalArg, rewriter, op.getLoc())) {
          if (normalized.getType() != ptrType) {
            normalized =
                rewriter.create<pto::CastPtrOp>(op.getLoc(), ptrType, normalized);
          }
          arg = normalized;
        }
      }
      operands.push_back(arg);
    }

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    state.propertiesAttr = op->getPropertiesAsAttribute();
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertPtrNormalizeUnrealizedCastOp final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }
    if (!hasPtrNormalizeConvertibleType(op->getOperandTypes()) &&
        !hasPtrNormalizeConvertibleType(op->getResultTypes())) {
      return failure();
    }

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType) {
      return failure();
    }

    Value input = adaptor.getOperands().front();
    if (input.getType() != convertedResultType) {
      auto ptrType = dyn_cast<pto::PtrType>(convertedResultType);
      if (!ptrType) {
        return failure();
      }
      input = materializeScalarAccessPtr(input, rewriter, op.getLoc());
      if (!input) {
        return failure();
      }
      if (input.getType() != ptrType) {
        input = rewriter.create<pto::CastPtrOp>(op.getLoc(), ptrType, input);
      }
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

struct ConvertPtrNormalizeMemRefCastOp final
    : public OpConversionPattern<memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasPtrNormalizeConvertibleType(op.getSource().getType()) &&
        !hasPtrNormalizeConvertibleType(op.getType())) {
      return failure();
    }

    Type convertedResultType = getTypeConverter()->convertType(op.getType());
    if (!convertedResultType) {
      return failure();
    }

    Value source = adaptor.getSource();
    if (source.getType() == convertedResultType) {
      rewriter.replaceOp(op, source);
      return success();
    }

    if (!isa<pto::PtrType, IntegerType>(source.getType()) ||
        !isa<pto::PtrType, IntegerType>(convertedResultType)) {
      return rewriter.notifyMatchFailure(
          op, "expected ptr/int memref.cast after ptr normalization");
    }

    rewriter.replaceOpWithNewOp<pto::CastPtrOp>(op, convertedResultType,
                                                source);
    return success();
  }
};

struct VPTOPtrNormalizePass
    : public pto::impl::VPTOPtrNormalizeBase<VPTOPtrNormalizePass> {
  using pto::impl::VPTOPtrNormalizeBase<
      VPTOPtrNormalizePass>::VPTOPtrNormalizeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [](Type type) { return convertSubviewResultType(type); });
    typeConverter.addTargetMaterialization(materializeUnrealizedCast);
    typeConverter.addSourceMaterialization(materializeUnrealizedCast);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect>();
    target.addDynamicallyLegalDialect<pto::PTODialect>([](Operation *op) {
      return !isa<pto::TileBufAddrOp, pto::CastPtrOp,
                  pto::IntToPtrOp, pto::PtrToIntOp,
                  pto::VldsOp, pto::VstsOp, pto::VsstbOp>(op);
    });
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          return !hasPtrNormalizeConvertibleType(op->getOperandTypes()) &&
                 !hasPtrNormalizeConvertibleType(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<memref::CastOp>([&](memref::CastOp op) {
      return !hasPtrNormalizeConvertibleType(op.getSource().getType()) &&
             !hasPtrNormalizeConvertibleType(op.getType());
    });
    target.addDynamicallyLegalOp<pto::TileBufAddrOp>([&](pto::TileBufAddrOp op) {
      return op.getDst().getType() ==
             typeConverter.convertType(op.getDst().getType());
    });
    target.addDynamicallyLegalOp<pto::CastPtrOp>([&](pto::CastPtrOp op) {
      return !isMemRefType(op.getInput().getType()) &&
             !isMemRefType(op.getResult().getType());
    });
    target.addDynamicallyLegalOp<pto::VldsOp>(
        [&](pto::VldsOp op) {
          return isa<pto::PtrType>(op.getSource().getType()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<pto::VstsOp>(
        [&](pto::VstsOp op) {
          return isa<pto::PtrType>(op.getDestination().getType()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<pto::VsstbOp>(
        [&](pto::VsstbOp op) {
          return isa<pto::PtrType>(op.getDestination().getType()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<pto::LoadScalarOp>(
        [](pto::LoadScalarOp op) { return isa<pto::PtrType>(op.getPtr().getType()); });
    target.addDynamicallyLegalOp<pto::StoreScalarOp>(
        [](pto::StoreScalarOp op) { return isa<pto::PtrType>(op.getPtr().getType()); });
    target.addDynamicallyLegalOp<pto::MteUbUbOp>([](pto::MteUbUbOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::RawFillL1Op>([](pto::RawFillL1Op op) {
      return isa<pto::PtrType>(op.getDst().getType());
    });
    target.addDynamicallyLegalOp<pto::MteUbL1Op>([](pto::MteUbL1Op op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteGmL1Op>([](pto::MteGmL1Op op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL1UbOp>([](pto::MteL1UbOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL1BtOp>([](pto::MteL1BtOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteGmL1FracOp>([](pto::MteGmL1FracOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL1L0aOp>([](pto::MteL1L0aOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL1L0bOp>([](pto::MteL1L0bOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL1L0aMxOp>([](pto::MteL1L0aMxOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL1L0bMxOp>([](pto::MteL1L0bMxOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL0cL1Op>([](pto::MteL0cL1Op op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL0cGmOp>([](pto::MteL0cGmOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteL0cUbOp>([](pto::MteL0cUbOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::MteUbGmOp>([](pto::MteUbGmOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::PTOLoadOp>(
        [](pto::PTOLoadOp op) { return isa<pto::PtrType>(op.getPtr().getType()); });
    target.addDynamicallyLegalOp<pto::PTOStoreOp>(
        [](pto::PTOStoreOp op) { return isa<pto::PtrType>(op.getPtr().getType()); });
    target.addDynamicallyLegalOp<pto::SimtLaunchOp>(
        [](pto::SimtLaunchOp op) {
          return !hasPtrNormalizeMemRefType(op->getOperandTypes()) &&
                 llvm::none_of(op.getArgs(), isTransientPtrMemRefBridge);
        });
    target.addDynamicallyLegalOp<memref::SubViewOp>(
        [](memref::SubViewOp op) { return !needsSubviewPtrConversion(op); });

    RewritePatternSet patterns(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                         target);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<ConvertTileBufAddrToPtrPattern,
                 ConvertIntToPtrToCastPtrPattern,
                 ConvertPtrToIntToCastPtrPattern, ConvertCastPtrPattern,
                 ConvertSubviewToAddPtrPattern, ConvertVldsSubviewOperandPattern,
                 ConvertVstsSubviewOperandPattern,
                 ConvertVsstbSubviewOperandPattern,
                 ConvertLoadScalarOperandToPtrPattern,
                 ConvertStoreScalarOperandToPtrPattern,
                 ConvertMteUbUbOperandPattern,
                 ConvertMteUbL1OperandPattern,
                 ConvertCubeLoadOperandPattern,
                 ConvertRawFillL1OperandPattern,
                 ConvertCubeStoreOperandPattern,
                 ConvertBiasLoadOperandPattern,
                 ConvertCubeLoadFracOperandPattern,
                 ConvertLeftLoadOperandPattern,
                 ConvertRightLoadOperandPattern,
                 ConvertLeftLoadMxOperandPattern,
                 ConvertRightLoadMxOperandPattern,
                 ConvertAccStoreOperandPattern,
                 ConvertAccStoreGmOperandPattern,
                 ConvertAccStoreUbOperandPattern,
                 ConvertMteUbGmOperandPattern,
                 ConvertLoadOperandToPtrPattern,
                 ConvertStoreOperandToPtrPattern,
                 ConvertSimtLaunchOp,
                 ConvertPtrNormalizeUnrealizedCastOp,
                 ConvertPtrNormalizeMemRefCastOp>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::pto {

std::unique_ptr<Pass> createVPTOPtrNormalizePass() {
  return std::make_unique<VPTOPtrNormalizePass>();
}

} // namespace mlir::pto
