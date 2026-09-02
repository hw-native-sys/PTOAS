// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

static FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context,
                                                 Type resultType,
                                                 StringRef stem,
                                                 StringRef suffix) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec +
                                      suffix.str())
      .getValue();
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

static std::optional<uint64_t> parsePartImmediate(StringRef part) {
  if (part == "EVEN" || part == "PART_EVEN")
  {
    return 0;
  }
  if (part == "ODD" || part == "PART_ODD")
  {
    return 1;
  }
  return std::nullopt;
}

static std::optional<uint64_t> parseOrderImmediate(StringRef order) {
  if (order.empty() || order == "ASC")
  {
    return 0;
  }
  if (order == "DESC")
  {
    return 1;
  }
  return std::nullopt;
}

static FailureOr<Value> packVbitsortConfig(Operation *anchor, Value repeatTimes) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value repeatI64 = castIntegerLikeTo(anchor, repeatTimes, builder.getI64Type());
  if (!repeatI64)
  {
    return failure();
  }
  return builder
      .create<arith::ShLIOp>(loc, repeatI64, getI64Constant(builder, loc, 56))
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

static Type getVgather2SourceElementType(Type sourceType) {
  if (auto ptrType = dyn_cast<pto::PtrType>(sourceType))
  {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(sourceType))
  {
    return memrefType.getElementType();
  }
  return {};
}

static FailureOr<StringRef> buildVgather2Callee(MLIRContext *context,
                                                Type sourceType,
                                                Type resultType) {
  Type sourceElemType = getVgather2SourceElementType(sourceType);
  Type resultElemType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!sourceElemType || !resultElemType || !lanes)
  {
    return failure();
  }

  std::string vec;
  int64_t intrinsicLanes = *lanes;
  if (pto::getPTOStorageElemBitWidth(sourceElemType) == 8) {
    vec = getElementTypeFragment(sourceElemType);
    intrinsicLanes *= 2;
  } else {
    vec = getElementTypeFragment(resultElemType);
  }
  if (vec.empty())
  {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.vgather2.v300.v" +
                                      std::to_string(intrinsicLanes) + vec)
      .getValue();
}

static std::optional<uint64_t> getFixedVectorBitWidth(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable())
  {
    return std::nullopt;
  }
  int64_t lanes = vectorType.getDimSize(0);
  if (lanes <= 0)
  {
    return std::nullopt;
  }
  auto elementType = dyn_cast<IntegerType>(vectorType.getElementType());
  if (!elementType)
  {
    return std::nullopt;
  }
  return static_cast<uint64_t>(lanes) * elementType.getWidth();
}

static FailureOr<Type> getVgather2OffsetsCarrierType(PatternRewriter &rewriter,
                                                     Type sourceType,
                                                     Type resultType,
                                                     Type offsetsType) {
  Type sourceElemType = getVgather2SourceElementType(sourceType);
  Type elementType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!sourceElemType || !elementType || !lanes || *lanes <= 0)
  {
    return failure();
  }

  Type carrierType = offsetsType;
  if (pto::getPTOStorageElemBitWidth(elementType) == 16) {
    if (*lanes % 2 != 0)
    {
      return failure();
    }
    carrierType = VectorType::get({*lanes / 2}, rewriter.getI32Type());
  }

  std::optional<uint64_t> offsetsBits = getFixedVectorBitWidth(offsetsType);
  std::optional<uint64_t> carrierBits = getFixedVectorBitWidth(carrierType);
  if (!offsetsBits || !carrierBits || *offsetsBits != *carrierBits)
  {
    return failure();
  }
  return carrierType;
}

static FailureOr<StringRef> buildVgather2BcCallee(MLIRContext *context,
                                                  Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgather2.bc", "");
}

static FailureOr<StringRef> buildVgatherbCallee(MLIRContext *context,
                                                Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgatherb.v310", "");
}

static FailureOr<StringRef> buildVscatterCallee(MLIRContext *context,
                                                Type valueType) {
  return buildLaneTypedCallee(context, valueType, "vscatter", ".v300");
}

static FailureOr<Type> getVscatterOffsetsCarrierType(Type offsetsType) {
  return offsetsType;
}

static FailureOr<StringRef> buildVaxpyCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vaxpy", ".m");
}

static FailureOr<StringRef> buildVmulscvtCallee(MLIRContext *context,
                                                Type inputType,
                                                Type resultType) {
  auto inputElemType = getElementTypeFromVectorLike(inputType);
  auto resultElemType = getElementTypeFromVectorLike(resultType);
  auto inputLanes = getElementCountFromVectorLike(inputType);
  auto resultLanes = getElementCountFromVectorLike(resultType);
  if (!inputElemType || !resultElemType || !inputLanes || !resultLanes)
  {
    return failure();
  }
  if (!inputElemType.isF32() || !resultElemType.isF16() || *inputLanes != 64 ||
      *resultLanes != 128) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vmulscvt.v128f16").getValue();
}

static FailureOr<StringRef> buildVciCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  if (vec == "f16" || vec == "f32") {
    return StringAttr::get(context, "llvm.hivm.vci.v" + std::to_string(*lanes) +
                                        vec + "." + vec)
        .getValue();
  }
  return StringAttr::get(context,
                         "llvm.hivm.vci.v" + std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVtrcCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vtrc." + vec + ".x").getValue();
}

static FailureOr<StringRef> buildVexpdifCallee(MLIRContext *context,
                                               Type inputType,
                                               Type resultType) {
  std::string srcVec =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  auto srcLanes = getElementCountFromVectorLike(inputType);
  std::string dstElem =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (srcVec.empty() || dstElem.empty() || !srcLanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vexpdif.v" +
                                      std::to_string(*srcLanes) + srcVec +
                                      dstElem)
      .getValue();
}

static FailureOr<StringRef> buildVbitsortCallee(MLIRContext *context,
                                                pto::VbitsortOp op) {
  Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
  if (sourceElemType.isF16())
  {
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f16").getValue();
  }
  if (sourceElemType.isF32())
  {
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f32").getValue();
  }
  return failure();
}

static FailureOr<StringRef> buildVmrgsort4Callee(MLIRContext *context,
                                                 pto::Vmrgsort4Op op) {
  Type elemType =
      cast<pto::PtrType>(op.getDestination().getType()).getElementType();
  if (elemType.isF16())
  {
    return StringAttr::get(context, "llvm.hivm.VMRGSORT.f16.V300").getValue();
  }
  if (elemType.isF32())
  {
    return StringAttr::get(context, "llvm.hivm.VMRGSORT.f32.V300").getValue();
  }
  return failure();
}

static FailureOr<Value> packVmrgsort4SourceAddr(Operation *anchor, Value source0,
                                                Value source1, Value source2,
                                                Value source3, Type elemType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  unsigned addrShift = 0;
  if (elemType.isF16())
  {
    addrShift = 3;
  } else if (elemType.isF32()) {
    addrShift = 3;
  } else {
    return failure();
  }

  auto packOne = [&](Value source, uint64_t laneShift) -> FailureOr<Value> {
    FailureOr<Value> ubPtr = reinterpretPointerToAddrSpace(anchor, source, 6);
    if (failed(ubPtr))
    {
      return failure();
    }
    Value asInt =
        builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), *ubPtr);
    Value shifted = builder.create<arith::ShRUIOp>(
        loc, asInt, getI64Constant(builder, loc, addrShift));
    Value masked = builder.create<arith::AndIOp>(
        loc, shifted, getI64Constant(builder, loc, 0xFFFFULL));
    if (laneShift == 0)
    {
      return masked;
    }
    return builder
        .create<arith::ShLIOp>(loc, masked,
                               getI64Constant(builder, loc, laneShift))
        .getResult();
  };

  FailureOr<Value> low0 = packOne(source0, 0);
  FailureOr<Value> low1 = packOne(source1, 16);
  FailureOr<Value> low2 = packOne(source2, 32);
  FailureOr<Value> low3 = packOne(source3, 48);
  if (failed(low0) || failed(low1) || failed(low2) || failed(low3))
  {
    return failure();
  }

  Value packed01 = builder.create<arith::OrIOp>(loc, *low0, *low1);
  Value packed23 = builder.create<arith::OrIOp>(loc, *low2, *low3);
  Value packed = builder.create<arith::OrIOp>(loc, packed01, packed23);
  Type ubPtrTy = LLVM::LLVMPointerType::get(anchor->getContext(), 6);
  return builder.create<LLVM::IntToPtrOp>(loc, ubPtrTy, packed).getResult();
}

class LowerVgather2OpPattern final
    : public OpConversionPattern<pto::Vgather2Op> {
public:
  explicit LowerVgather2OpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vgather2Op op, pto::Vgather2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    if (!elemType || !basePtr) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgather2 operand types");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vgather2 result type");
    }

    FailureOr<StringRef> calleeName =
        buildVgather2Callee(op.getContext(), op.getSource().getType(),
                            op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vgather2 signature");
    }

    Value offsets = adaptor.getOffsets();
    FailureOr<Type> offsetsCarrierType = getVgather2OffsetsCarrierType(
        rewriter, op.getSource().getType(), op.getResult().getType(),
        offsets.getType());
    if (failed(offsetsCarrierType))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vgather2 offsets carrier");
    }
    if (offsets.getType() != *offsetsCarrierType) {
      offsets = rewriter.create<LLVM::BitcastOp>(op.getLoc(), *offsetsCarrierType,
                                                 offsets);
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), *offsetsCarrierType,
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), offsets, adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2BcOpPattern final
    : public OpConversionPattern<pto::Vgather2BcOp> {
public:
  explicit LowerVgather2BcOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2BcOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vgather2BcOp op, pto::Vgather2BcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgather2_bc operand/result types");
    }

    FailureOr<StringRef> calleeName =
        buildVgather2BcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vgather2_bc signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgatherbOpPattern final
    : public OpConversionPattern<pto::VgatherbOp> {
public:
  explicit LowerVgatherbOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VgatherbOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VgatherbOp op, pto::VgatherbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgatherb operand/result types");
    }

    FailureOr<StringRef> calleeName =
        buildVgatherbCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vgatherb signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVscatterOpPattern final
    : public OpConversionPattern<pto::VscatterOp> {
public:
  explicit LowerVscatterOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VscatterOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VscatterOp op, pto::VscatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getValue().getType());
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    if (!elemType || !basePtr) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vscatter operand types");
    }

    FailureOr<StringRef> calleeName =
        buildVscatterCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vscatter signature");
    }

    FailureOr<Type> offsetsCarrierType = getVscatterOffsetsCarrierType(
        adaptor.getOffsets().getType());
    if (failed(offsetsCarrierType))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vscatter offsets carrier");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  *offsetsCarrierType, adaptor.getMask().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getValue(), adaptor.getDestination(),
                   adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVaxpyOpPattern final : public OpConversionPattern<pto::VaxpyOp> {
public:
  explicit LowerVaxpyOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VaxpyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VaxpyOp op, pto::VaxpyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!elemType)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vaxpy result type");
    }

    FailureOr<StringRef> calleeName =
        buildVaxpyCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy callee");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSrc1().getType(), adaptor.getSrc0().getType(),
                  adaptor.getAlpha().getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSrc1(), adaptor.getSrc0(), adaptor.getAlpha(),
                   adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmulscvtOpPattern final
    : public OpConversionPattern<pto::VmulscvtOp> {
public:
  explicit LowerVmulscvtOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VmulscvtOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VmulscvtOp op, pto::VmulscvtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundMode = parseRoundModeImmediate(op.getRnd());
    if (!roundMode)
    {
      return rewriter.notifyMatchFailure(op, "vmulscvt requires valid rnd attr");
    }
    if (*roundMode != 1) {
      return rewriter.notifyMatchFailure(
          op, "current vmulscvt lowering only supports rnd A");
    }

    auto part = parsePartImmediate(op.getPart());
    if (!part)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vmulscvt part");
    }

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vmulscvt result type");
    }

    FailureOr<StringRef> calleeName =
        buildVmulscvtCallee(op.getContext(), op.getInput().getType(),
                            op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vmulscvt signature");
    }

    Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), adaptor.getScalar().getType(),
                  adaptor.getMask().getType(), partValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getScalar(), adaptor.getMask(),
                   partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVciOpPattern final : public OpConversionPattern<pto::VciOp> {
public:
  explicit LowerVciOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::VciOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VciOp op, pto::VciOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto order = parseOrderImmediate(op.getOrder().value_or("ASC"));
    if (!order)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vci order");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vci result type");
    }

    FailureOr<StringRef> calleeName =
        buildVciCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vci callee");
    }

    Value indexValue = adaptor.getIndex();
    Type resultElemType =
        cast<pto::VRegType>(op.getResult().getType()).getElementType();
    if (auto intType = dyn_cast<IntegerType>(resultElemType)) {
      if (intType.getWidth() == 8) {
        Type loweredIndexType = rewriter.getI16Type();
        if (intType.isUnsigned()) {
          indexValue = rewriter.create<arith::ExtUIOp>(op.getLoc(),
                                                       loweredIndexType,
                                                       indexValue);
        } else {
          indexValue = rewriter.create<arith::ExtSIOp>(op.getLoc(),
                                                       loweredIndexType,
                                                       indexValue);
}
      }
    }

    Value orderValue = getI32Constant(rewriter, op.getLoc(), *order);
    auto funcType = rewriter.getFunctionType(
        TypeRange{indexValue.getType(), orderValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{indexValue, orderValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVexpdifOpPattern final
    : public OpConversionPattern<pto::VexpdifOp> {
public:
  explicit LowerVexpdifOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VexpdifOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VexpdifOp op, pto::VexpdifOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto part = parsePartImmediate(op.getPart());
    if (!part)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vexpdif signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vexpdif result type");
    }

    FailureOr<StringRef> calleeName =
        buildVexpdifCallee(op.getContext(), op.getInput().getType(),
                           op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vexpdif callee");
    }

    Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), adaptor.getMax().getType(),
                  adaptor.getMask().getType(), partValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getMax(), adaptor.getMask(),
                   partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbitsortOpPattern final
    : public OpConversionPattern<pto::VbitsortOp> {
public:
  explicit LowerVbitsortOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbitsortOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VbitsortOp op, pto::VbitsortOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto srcType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto idxType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getIndices().getType());
    if (!dstType || !srcType || !idxType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vbitsort operand types");
    }

    FailureOr<Value> config = packVbitsortConfig(op, adaptor.getRepeatTimes());
    if (failed(config))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack vbitsort config");
    }

    FailureOr<StringRef> calleeName = buildVbitsortCallee(op.getContext(), op);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vbitsort signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getDestination().getType(), adaptor.getSource().getType(),
                  adaptor.getIndices().getType(), (*config).getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getDestination(), adaptor.getSource(),
                   adaptor.getIndices(), *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmrgsort4OpPattern final
    : public OpConversionPattern<pto::Vmrgsort4Op> {
public:
  explicit LowerVmrgsort4OpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vmrgsort4Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vmrgsort4Op op, pto::Vmrgsort4Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto src0Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource0().getType());
    auto src1Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource1().getType());
    auto src2Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource2().getType());
    auto src3Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource3().getType());
    if (!dstType || !src0Type || !src1Type || !src2Type || !src3Type) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vmrgsort4 operand types");
    }

    Type elemType =
        cast<pto::PtrType>(op.getDestination().getType()).getElementType();
    FailureOr<Value> packedSrc = packVmrgsort4SourceAddr(
        op, adaptor.getSource0(), adaptor.getSource1(), adaptor.getSource2(),
        adaptor.getSource3(), elemType);
    if (failed(packedSrc)) {
      return rewriter.notifyMatchFailure(
          op, "failed to pack vmrgsort4 source addresses");
    }

    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, adaptor.getDestination(), 6);
    if (failed(dst))
    {
      return rewriter.notifyMatchFailure(op, "failed to normalize vmrgsort4 destination");
    }

    FailureOr<StringRef> calleeName = buildVmrgsort4Callee(op.getContext(), op);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vmrgsort4 signature");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{(*dst).getType(), (*packedSrc).getType(),
                  adaptor.getCount().getType(), adaptor.getConfig().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*dst, *packedSrc, adaptor.getCount(), adaptor.getConfig()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVtrcOpPattern final : public OpConversionPattern<pto::VtrcOp> {
public:
  explicit LowerVtrcOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VtrcOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VtrcOp op, pto::VtrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundMode = parseRoundModeImmediate(op.getRoundMode());
    if (!roundMode)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vtrc signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vtrc result type");
    }

    FailureOr<StringRef> calleeName =
        buildVtrcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vtrc callee");
    }

    Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), roundValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), roundValue, adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

} // namespace

void populateVPTOVectorGatherPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      LoweringState &state) {
  patterns.add<LowerVgather2OpPattern, LowerVgather2BcOpPattern,
               LowerVgatherbOpPattern, LowerVscatterOpPattern,
               LowerVaxpyOpPattern, LowerVmulscvtOpPattern,
               LowerVciOpPattern, LowerVexpdifOpPattern,
               LowerVbitsortOpPattern, LowerVmrgsort4OpPattern,
               LowerVtrcOpPattern>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
