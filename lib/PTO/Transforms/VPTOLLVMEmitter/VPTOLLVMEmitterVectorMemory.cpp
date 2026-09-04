// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"
#include "PTO/Transforms/VPTOLLVMEmitterHelper.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

template <typename Op>
static LogicalResult finishPostUpdateStore(
    Op op, ConversionPatternRewriter &rewriter, Value updatedBase,
    Operation *call, bool usePostIntrinsic) {
  if (usePostIntrinsic) {
    Value result = updatedBase ? updatedBase : call->getResult(0);
    rewriter.replaceOp(op, result);
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}

static bool isLowpPayloadElementType(Type type) {
  return pto::isPTOFloat8Type(type) || pto::isPTOHiFloat8Type(type) ||
         pto::isPTOFloat4PackedType(type);
}

struct LowpPayloadABI {
  Type llvmElementType;
  StringRef intrinsicElementFragment;
};

static std::optional<LowpPayloadABI>
getLowpPayloadABI(Type elementType, MLIRContext *context) {
  if (!isLowpPayloadElementType(elementType))
  {
    return std::nullopt;
  }
  return LowpPayloadABI{IntegerType::get(context, 8), "u8"};
}

static Type getLowpPayloadCarrierType(Type vectorLikeType,
                                      MLIRContext *context) {
  Type elementType = getElementTypeFromVectorLike(vectorLikeType);
  std::optional<LowpPayloadABI> abi =
      getLowpPayloadABI(elementType, context);
  if (!abi) {
    return {};
  }
  auto lanes = getElementCountFromVectorLike(vectorLikeType);
  if (!lanes) {
    return {};
  }
  return VectorType::get({*lanes}, abi->llvmElementType);
}

static Type getPayloadABIType(Type semanticType, Type convertedType,
                              MLIRContext *context) {
  if (Type carrierType = getLowpPayloadCarrierType(semanticType, context))
  {
    return carrierType;
  }
  return convertedType;
}

static Value castToPayloadABI(Location loc, Value value,
                              Type semanticType,
                              ConversionPatternRewriter &rewriter) {
  Type carrierType =
      getLowpPayloadCarrierType(semanticType, rewriter.getContext());
  if (!carrierType || carrierType == value.getType())
  {
    return value;
  }
  return rewriter.create<LLVM::BitcastOp>(loc, carrierType, value);
}

static Value castFromPayloadABI(
    Location loc, Value value, Type semanticType, Type convertedType,
    ConversionPatternRewriter &rewriter) {
  Type carrierType =
      getLowpPayloadCarrierType(semanticType, rewriter.getContext());
  if (!carrierType || carrierType == convertedType)
  {
    return value;
  }
  return rewriter.create<LLVM::BitcastOp>(loc, convertedType, value);
}

static std::optional<int32_t> parsePostModeImmediate(StringRef mode) {
  if (mode == "NO_POST_UPDATE")
  {
    return 0;
  }
  if (mode == "POST_UPDATE")
  {
    return 1;
  }
  return std::nullopt;
}

static Value buildPostModeValue(ConversionPatternRewriter &rewriter,
                                Location loc, bool enabled) {
  return getI32Constant(rewriter, loc, enabled ? 1 : 0);
}

static std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
  {
    return intType.getWidth();
  }
  if (isLowpPayloadElementType(type))
  {
    return 8;
  }
  if (type.isF16() || type.isBF16())
  {
    return 16;
  }
  if (type.isF32())
  {
    return 32;
  }
  if (type.isF64())
  {
    return 64;
  }
  // bf16x2 is a 32-bit packed pair; its dist width is 32 (i32/align4 ABI).
  if (pto::isPTOBF16x2Type(type)) {
    return 32;
  }
  return std::nullopt;
}

static std::optional<uint64_t> parseLoadDistImmediate(StringRef dist,
                                                      Type elementType) {
  const auto *contract = lookupVPTOMemoryDist(VPTOMemoryOpFamily::Load, dist,
                                              getDistElementWidth(elementType));
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

static std::optional<uint64_t> parseLoadX2DistImmediate(StringRef dist,
                                                        Type elementType) {
  const auto *contract = lookupVPTOMemoryDist(VPTOMemoryOpFamily::LoadX2, dist,
                                              getDistElementWidth(elementType));
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

static std::optional<uint64_t> parseStoreDistImmediate(StringRef dist,
                                                       Type elementType) {
  const auto *contract = lookupVPTOMemoryDist(
      VPTOMemoryOpFamily::Store, dist,
      dist.empty() ? getDistElementWidth(elementType) : std::nullopt);
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

static std::optional<uint64_t> parseSprImmediate(StringRef spr) {
  if (spr == "AR") {
    return 74;
  }
  if (spr == "VR") {
    return 1;
  }
  return std::nullopt;
}

static Value packBlockRepeatStride(Operation *anchor, Value blockStride,
                                   Value repeatStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Value blockI32 = castIntegerLikeTo(anchor, blockStride, builder.getI32Type());
  Value repeatI32 = castIntegerLikeTo(anchor, repeatStride, builder.getI32Type());
  if (!blockI32 || !repeatI32) {
    return {};
  }
  Value shift = builder.create<arith::ConstantIntOp>(anchor->getLoc(), 16, 32);
  Value blockShifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), blockI32, shift);
  return builder.create<arith::OrIOp>(anchor->getLoc(), blockShifted,
                                      repeatI32)
      .getResult();
}

static bool isOnePointStoreDist(StringRef dist) {
  const auto *contract = lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
  return contract && contract->isOnePointStore();
}

static std::optional<uint64_t> parseStoreX2DistImmediate(StringRef dist,
                                                         Type) {
  const auto *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::StoreX2, dist);
  return contract ? std::optional<uint64_t>(contract->a5Immediate)
                  : std::nullopt;
}

static FailureOr<StringRef> buildPstuCallee(MLIRContext *context, pto::PstuOp op) {
  if (auto maskType = dyn_cast<pto::MaskType>(op.getValue().getType())) {
    if (maskType.isB16())
    {
      return StringAttr::get(context, "llvm.hivm.pstu.b16").getValue();
    }
    if (maskType.isB32())
    {
      return StringAttr::get(context, "llvm.hivm.pstu.b32").getValue();
    }
  }
  return failure();
}

static FailureOr<StringRef> buildVstusCallee(MLIRContext *context,
                                              Type valueType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstus.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVstusPostCallee(MLIRContext *context,
                                                 Type valueType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstus.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static StringRef buildVsturCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstur").getValue();
}

static StringRef buildInitAlignCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.init.vector.align.data").getValue();
}

static StringRef buildSprclrCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.sprclr").getValue();
}

static StringRef buildSprstiCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.sprsti.post"
                              : "llvm.hivm.sprsti")
      .getValue();
}

static StringRef buildSprstsCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.sprsts.post"
                              : "llvm.hivm.sprsts")
      .getValue();
}

template <typename SprStoreOp>
static StringRef buildSprStoreCallee(MLIRContext *context, bool post);

template <>
StringRef buildSprStoreCallee<pto::SprstiOp>(MLIRContext *context, bool post) {
  return buildSprstiCallee(context, post);
}

template <>
StringRef buildSprStoreCallee<pto::SprstsOp>(MLIRContext *context, bool post) {
  return buildSprstsCallee(context, post);
}

template <typename StoreOp>
static StringRef getPredicateStoreCallee(MLIRContext *context, bool post);

template <>
StringRef getPredicateStoreCallee<pto::PstiOp>(MLIRContext *context,
                                                bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.psti.post.b8" : "llvm.hivm.psti.b8")
      .getValue();
}

template <>
StringRef getPredicateStoreCallee<pto::PstsOp>(MLIRContext *context,
                                                bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.psts.post.b8" : "llvm.hivm.psts.b8")
      .getValue();
}

template <typename LoadOp>
static StringRef getPredicateLoadCallee(MLIRContext *context, bool post);

template <>
StringRef getPredicateLoadCallee<pto::PldiOp>(MLIRContext *context, bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.pldi.post.b8" : "llvm.hivm.pldi.b8")
      .getValue();
}

template <>
StringRef getPredicateLoadCallee<pto::PldsOp>(MLIRContext *context, bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.plds.post.b8" : "llvm.hivm.plds.b8")
      .getValue();
}

static StringRef buildVstarCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstar").getValue();
}

static StringRef buildVstasCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context,
                         post ? "llvm.hivm.vstas.post"
                              : "llvm.hivm.vstas")
      .getValue();
}

static FailureOr<StringRef> buildVldsPostCallee(MLIRContext *context,
                                                Type resultType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldsx1.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVstsPostCallee(MLIRContext *context,
                                                Type valueType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstsx1.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static StringRef buildVldasCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vldas").getValue();
}

static FailureOr<StringRef> buildVldusCallee(MLIRContext *context,
                                             Type resultType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldus.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVldusPostCallee(MLIRContext *context,
                                                 Type resultType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldus.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVldsCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldsx1.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVldsx2Callee(MLIRContext *context,
                                              Type resultType, bool post) {
  Type elementType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!elementType || !lanes)
  {
    return failure();
  }
  std::string element = getMemoryElementTypeFragment(elementType);
  if (element.empty())
  {
    return failure();
  }
  return StringAttr::get(
             context, "llvm.hivm.vldsx2" +
                          std::string(post ? ".post" : "") + ".v" +
                          std::to_string(*lanes) + element)
      .getValue();
}

static FailureOr<StringRef>
buildBlockStridedMemoryCallee(MLIRContext *context, Type vectorType,
                              StringRef stem, bool post) {
  Type elementType = getElementTypeFromVectorLike(vectorType);
  auto lanes = getElementCountFromVectorLike(vectorType);
  if (!elementType || !lanes)
  {
    return failure();
  }

  std::string element;
  if (auto intType = dyn_cast<IntegerType>(elementType))
  {
    element = "i" + std::to_string(intType.getWidth());
  } else if (isLowpPayloadElementType(elementType)) {
    element = "i8";
  } else {
    element = getMemoryElementTypeFragment(elementType);
  }
  if (element.empty())
  {
    return failure();
  }

  return StringAttr::get(context,
                         "llvm.hivm." + stem.str() +
                             std::string(post ? ".post" : "") + ".v" +
                             std::to_string(*lanes) + element)
      .getValue();
}

static FailureOr<StringRef> buildVsldbCallee(MLIRContext *context,
                                              Type resultType, bool post) {
  return buildBlockStridedMemoryCallee(context, resultType, "vsldb",
                                       post);
}

static FailureOr<StringRef> buildVstsCallee(MLIRContext *context, Type valueType) {
  std::string vec =
      getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstsx1.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVstsx2Callee(MLIRContext *context, Type valueType) {
  Type elementType = getElementTypeFromVectorLike(valueType);
  auto lanes = getElementCountFromVectorLike(valueType);
  if (!elementType || !lanes)
  {
    return failure();
  }

  std::string element = getMemoryElementTypeFragment(elementType);
  if (element.empty())
  {
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.vstsx2.v" +
                                      std::to_string(*lanes) + element)
      .getValue();
}

static FailureOr<StringRef> buildVsstbCallee(MLIRContext *context,
                                             Type valueType, bool post) {
  return buildBlockStridedMemoryCallee(context, valueType, "vsstb", post);
}

class LowerVldsOpPattern final : public OpConversionPattern<pto::VldsOp> {
public:
  explicit LowerVldsOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VldsOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptoResultType = op.getResult().getType();
    Type elementType = getElementTypeFromVectorLike(ptoResultType);
    if (!elementType)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vlds element type");
    }
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(
        op, adaptor.getSource(), adaptor.getOffset(), elementType,
        usePostIntrinsic, rewriter);
    auto dist =
        parseLoadDistImmediate(op.getDist().value_or("NORM"), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vlds operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
    {
      return rewriter.notifyMatchFailure(op, "failed to convert vlds result types");
    }

    if (usePostIntrinsic) {
      if (resultTypes.size() != 2 || resultTypes[1] != adaptor.getSource().getType()) {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported vlds post-update results");
      }
    } else if (resultTypes.size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported vlds result count");
    }

    FailureOr<StringRef> calleeName =
        usePostIntrinsic
            ? buildVldsPostCallee(op.getContext(), ptoResultType)
            : buildVldsCallee(op.getContext(), ptoResultType);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vlds signature");
    }

    Type callValueType = getPayloadABIType(
        ptoResultType, resultTypes[0], rewriter.getContext());
    SmallVector<Type> callResultTypes{callValueType};
    if (usePostIntrinsic)
    {
      callResultTypes.push_back(resultTypes[1]);
    }

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue = buildPostModeValue(rewriter, op.getLoc(), usePostIntrinsic);
    SmallVector<Value> args{loweredOffset->base,
                            loweredOffset->intrinsicOffset, distValue,
                            postValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{loweredOffset->base.getType(),
                  loweredOffset->intrinsicOffset.getType(),
                  distValue.getType(), postValue.getType()},
        callResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              callResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value loaded = castFromPayloadABI(
        op.getLoc(), call.getResult(0), ptoResultType, resultTypes[0],
        rewriter);
    if (usePostIntrinsic) {
      Value updatedBase = loweredOffset->updatedBase
                              ? loweredOffset->updatedBase
                              : call.getResult(1);
      rewriter.replaceOp(op, ValueRange{loaded, updatedBase});
    } else {
      rewriter.replaceOp(op, ValueRange{loaded});
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsx2OpPattern final : public OpConversionPattern<pto::Vldsx2Op> {
public:
  explicit LowerVldsx2OpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vldsx2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vldsx2Op op, pto::Vldsx2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 element type");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(
        op, adaptor.getSource(), adaptor.getOffset(), elementType,
        usePostIntrinsic, rewriter);
    auto dist = parseLoadX2DistImmediate(op.getDist(), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vldsx2 operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 3u : 2u)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vldsx2 result types");
    }

    FailureOr<StringRef> calleeName =
        buildVldsx2Callee(op.getContext(), op.getLow().getType(),
                          usePostIntrinsic);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 signature");
    }

    Type lowCallType = getPayloadABIType(
        op.getLow().getType(), resultTypes[0], rewriter.getContext());
    Type highCallType = getPayloadABIType(
        op.getHigh().getType(), resultTypes[1], rewriter.getContext());
    SmallVector<Type> callResultTypes{lowCallType, highCallType};
    if (usePostIntrinsic)
    {
      callResultTypes.push_back(resultTypes[2]);
    }

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue =
        getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{loweredOffset->base,
                            loweredOffset->intrinsicOffset, distValue,
                            postValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{loweredOffset->base.getType(),
                  loweredOffset->intrinsicOffset.getType(),
                  distValue.getType(), postValue.getType()},
        callResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              callResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value low = castFromPayloadABI(
        op.getLoc(), call.getResult(0), op.getLow().getType(), resultTypes[0],
        rewriter);
    Value high = castFromPayloadABI(
        op.getLoc(), call.getResult(1), op.getHigh().getType(), resultTypes[1],
        rewriter);
    if (usePostIntrinsic) {
      Value updatedBase = loweredOffset->updatedBase
                              ? loweredOffset->updatedBase
                              : call.getResult(2);
      rewriter.replaceOp(op, ValueRange{low, high, updatedBase});
    } else {
      rewriter.replaceOp(op, ValueRange{low, high});
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsldbOpPattern final : public OpConversionPattern<pto::VsldbOp> {
public:
  explicit LowerVsldbOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsldbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsldbOp op, pto::VsldbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Value packedStride =
        packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride)
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize vsldb operands");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 2u : 1u)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsldb result type");
    }

    Type callResultType = getPayloadABIType(
        op.getResult().getType(), resultTypes[0], rewriter.getContext());
    SmallVector<Type> callResultTypes{callResultType};
    if (usePostIntrinsic)
    {
      callResultTypes.push_back(resultTypes[1]);
    }

    FailureOr<StringRef> calleeName =
        buildVsldbCallee(op.getContext(), op.getResult().getType(),
                         usePostIntrinsic);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vsldb signature");
    }
    Value postValue =
        getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{adaptor.getSource(), packedStride, postValue,
                            adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), packedStride.getType(),
                  postValue.getType(), adaptor.getMask().getType()},
        callResultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              callResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value result = castFromPayloadABI(
        op.getLoc(), call.getResult(0), op.getResult().getType(), resultTypes[0],
        rewriter);
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

class LowerInitAlignOpPattern final
    : public OpConversionPattern<pto::InitAlignOp> {
public:
  explicit LowerInitAlignOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::InitAlignOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::InitAlignOp op, pto::InitAlignOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
    {
      return rewriter.notifyMatchFailure(op, "failed to convert init_align result type");
    }

    StringRef calleeName = buildInitAlignCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldasOpPattern final : public OpConversionPattern<pto::VldasOp> {
public:
  explicit LowerVldasOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldasOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldasOp op, pto::VldasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!sourceType || !resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected converted vldas operand/result types");
    }

    StringRef calleeName = buildVldasCallee(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getSource().getType()},
                                 TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getSource()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldusOpPattern final : public OpConversionPattern<pto::VldusOp> {
public:
  explicit LowerVldusOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldusOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldusOp op, pto::VldusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    SmallVector<Type> resultTypes;
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (!sourceType ||
        failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 3u : 2u) ||
        adaptor.getAlign().getType() != resultTypes[1] ||
        (usePostIntrinsic && resultTypes[2] != adaptor.getSource().getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "expected converted vldus operand/result types");
    }

    FailureOr<StringRef> calleeName =
        usePostIntrinsic
            ? buildVldusPostCallee(op.getContext(), op.getResult().getType())
            : buildVldusCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vldus signature");
    }

    Type callValueType = getPayloadABIType(
        op.getResult().getType(), resultTypes[0], rewriter.getContext());
    SmallVector<Type> intrinsicResultTypes{callValueType, resultTypes[1]};
    // The installed no-post A5 vldus intrinsic returns an extra hidden base ptr.
    intrinsicResultTypes.push_back(adaptor.getSource().getType());

    SmallVector<Value> args{adaptor.getSource(), adaptor.getAlign()};
    Value explicitUpdatedBase;
    if (usePostIntrinsic) {
      Type elementType = getElementTypeFromVectorLike(op.getResult().getType());
      auto loweredIncrement = lowerVPTOElementOffsetForIntrinsic(
          op, adaptor.getSource(), adaptor.getIncrement(), elementType,
          /*isPostUpdate=*/true, rewriter);
      if (failed(loweredIncrement)) {
        return rewriter.notifyMatchFailure(op,
                                           "failed to convert vldus increment");
      }
      args.front() = loweredIncrement->base;
      args.push_back(loweredIncrement->intrinsicOffset);
      explicitUpdatedBase = loweredIncrement->updatedBase;
    }
    SmallVector<Type> argTypes;
    for (Value arg : args)
    {
      argTypes.push_back(arg.getType());
    }
    auto funcType = rewriter.getFunctionType(argTypes, intrinsicResultTypes);
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, intrinsicResultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value loaded = castFromPayloadABI(
        op.getLoc(), call.getResult(0), op.getResult().getType(),
        resultTypes[0], rewriter);
    SmallVector<Value> replacements{loaded, call.getResult(1)};
    if (usePostIntrinsic)
    {
      replacements.push_back(explicitUpdatedBase ? explicitUpdatedBase
                                                 : call.getResult(2));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }

private:
  LoweringState &state;
};

class LowerSprclrOpPattern final : public OpConversionPattern<pto::SprclrOp> {
public:
  explicit LowerSprclrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                                LoweringState &state)
      : OpConversionPattern<pto::SprclrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::SprclrOp op, pto::SprclrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto spr = parseSprImmediate(op.getSpr());
    if (!spr)
    {
      return rewriter.notifyMatchFailure(op, "unsupported sprclr target");
    }

    StringRef calleeName = buildSprclrCallee(op.getContext());
    Value sprValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI16IntegerAttr(*spr));
    auto funcType = rewriter.getFunctionType(TypeRange{sprValue.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{sprValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SprStoreOp>
class LowerSprStoreOpPattern final : public OpConversionPattern<SprStoreOp> {
public:
  explicit LowerSprStoreOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<SprStoreOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SprStoreOp op, typename SprStoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto spr = parseSprImmediate(op.getSpr());
    if (!spr)
    {
      return rewriter.notifyMatchFailure(op, "unsupported spr store target");
    }
    auto destType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    if (!destType || !adaptor.getOffset().getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected converted spr store operands");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 1u : 0u)) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert spr store result types");
    }

    StringRef calleeName =
        buildSprStoreCallee<SprStoreOp>(op.getContext(), usePostIntrinsic);
    Value sprValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI16IntegerAttr(*spr));
    Value postValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(usePostIntrinsic ? 1 : 0));
    SmallVector<Value> args{sprValue, adaptor.getDestination(),
                            adaptor.getOffset(), postValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{sprValue.getType(), adaptor.getDestination().getType(),
                  adaptor.getOffset().getType(), postValue.getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (usePostIntrinsic)
    {
      rewriter.replaceOp(op, call.getResults());
    }
    else
    {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsOpPattern final : public OpConversionPattern<pto::VstsOp> {
public:
  explicit LowerVstsOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VstsOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vsts element type");
    }
    Type offsetElementType = elementType;
    if (auto ptrType = dyn_cast<pto::PtrType>(op.getDestination().getType()))
    {
      offsetElementType = ptrType.getElementType();
    } else if (auto memrefType = dyn_cast<BaseMemRefType>(op.getDestination().getType())) {
      offsetElementType = memrefType.getElementType();
    }
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(
        op, adaptor.getDestination(), adaptor.getOffset(), offsetElementType,
        usePostIntrinsic, rewriter);
    auto dist =
        parseStoreDistImmediate(op.getDist().value_or(""), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op, "failed to materialize vsts operands");
    }

    FailureOr<StringRef> calleeName =
        op.getUpdatedBase()
            ? buildVstsPostCallee(op.getContext(), op.getValue().getType())
            : buildVstsCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vsts signature");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vsts result types");
    }
    if (usePostIntrinsic) {
      if (resultTypes.size() != 1 ||
          resultTypes[0] != adaptor.getDestination().getType()) {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported vsts post-update result");
      }
    } else if (!resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported vsts result count");
    }

    Value distValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(
                                                        usePostIntrinsic ? 1 : 0));
    Value value = castToPayloadABI(
        op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    Value mask = adaptor.getMask();
    // The 1PT store forms keep a mask operand in the LLVM ABI, but the
    // hardware ignores it.  Do not materialize a pset/pge mask solely for
    // this dead operand; an LLVM undef is sufficient at this boundary.
    StringRef distToken = op.getDist().value_or("");
    if (isOnePointStoreDist(distToken))
    {
      mask = rewriter.create<LLVM::UndefOp>(op.getLoc(), mask.getType());
    }
    SmallVector<Value> args{value, loweredOffset->base,
                            loweredOffset->intrinsicOffset, distValue, zero,
                            mask};
    auto funcType = rewriter.getFunctionType(
        TypeRange{value.getType(), loweredOffset->base.getType(),
                  rewriter.getI32Type(), rewriter.getI32Type(),
                  rewriter.getI32Type(), mask.getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    return finishPostUpdateStore(op, rewriter, loweredOffset->updatedBase,
                                 call.getOperation(), usePostIntrinsic);
  }

private:
  LoweringState &state;
};

class LowerVsstbOpPattern final : public OpConversionPattern<pto::VsstbOp> {
public:
  explicit LowerVsstbOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsstbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsstbOp op, pto::VsstbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Value packedStride =
        packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride)
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize vsstb operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vsstb result types");
    }
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (usePostIntrinsic) {
      if (resultTypes.size() != 1 ||
          resultTypes[0] != adaptor.getDestination().getType()) {
        return rewriter.notifyMatchFailure(
            op, "unsupported vsstb post-update result");
      }
    } else if (!resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported vsstb result count");
    }

    FailureOr<StringRef> calleeName = buildVsstbCallee(
        op.getContext(), op.getValue().getType(), usePostIntrinsic);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vsstb signature");
    }
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    Value value = castToPayloadABI(
        op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    SmallVector<Value> args{value, adaptor.getDestination(),
                            packedStride, zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{value.getType(), adaptor.getDestination().getType(),
                  packedStride.getType(), zeroValue.getType(),
                  adaptor.getMask().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    return finishPostUpdateStore(op, rewriter, Value(), call.getOperation(),
                                 usePostIntrinsic);
  }

private:
  LoweringState &state;
};

class LowerVstsx2OpPattern final : public OpConversionPattern<pto::Vstsx2Op> {
public:
  explicit LowerVstsx2OpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vstsx2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vstsx2Op op, pto::Vstsx2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 element type");
    }

    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(
        op, adaptor.getDestination(), adaptor.getOffset(), elementType,
        /*isPostUpdate=*/false, rewriter);
    auto dist = parseStoreX2DistImmediate(op.getDist(), elementType);
    bool invalidAddress = failed(loweredOffset) || !dist;
    if (invalidAddress) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vstsx2 operands");
    }

    FailureOr<StringRef> calleeName =
        buildVstsx2Callee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 signature");
    }

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    Value low = castToPayloadABI(
        op.getLoc(), adaptor.getLow(), op.getLow().getType(), rewriter);
    Value high = castToPayloadABI(
        op.getLoc(), adaptor.getHigh(), op.getHigh().getType(), rewriter);
    SmallVector<Value> args{low, high, loweredOffset->base,
                            loweredOffset->intrinsicOffset, distValue,
                            zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{low.getType(), high.getType(),
                  loweredOffset->base.getType(),
                  loweredOffset->intrinsicOffset.getType(),
                  distValue.getType(), zeroValue.getType(),
                  adaptor.getMask().getType()},
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
  explicit LowerPstuOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::PstuOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::PstuOp op, pto::PstuOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildPstuCallee(op.getContext(), op);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported pstu signature");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
    {
      return rewriter.notifyMatchFailure(op, "failed to convert pstu result types");
    }
    if (resultTypes.size() != 2)
    {
      return rewriter.notifyMatchFailure(op, "unexpected converted pstu result arity");
    }

    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!baseType || adaptor.getAlignIn().getType() != resultTypes[0] ||
        adaptor.getBase().getType() != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted pstu operand/result types");
    }

    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  adaptor.getAlignIn().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes,
                                              args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstusOpPattern final : public OpConversionPattern<pto::VstusOp> {
public:
  explicit LowerVstusOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstusOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstusOp op, pto::VstusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vstus element type");
    }

    bool usePostIntrinsic = static_cast<bool>(op.getBaseOut());
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(
        op, adaptor.getBase(), adaptor.getOffset(), elementType,
        usePostIntrinsic, rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vstus offset");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vstus result types");
    }
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!baseType || resultTypes.size() != (usePostIntrinsic ? 2u : 1u) ||
        adaptor.getAlignIn().getType() != resultTypes[0] ||
        (usePostIntrinsic && resultTypes[1] != adaptor.getBase().getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstus operand/result types");
    }

    FailureOr<StringRef> calleeName =
        buildVstusCallee(op.getContext(), op.getValue().getType());
    if (usePostIntrinsic) {
      calleeName =
          buildVstusPostCallee(op.getContext(), op.getValue().getType());
    }
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported vstus signature");
    }
    Value value = castToPayloadABI(
        op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    SmallVector<Value> args{value, loweredOffset->base,
                            loweredOffset->intrinsicOffset,
                            adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{value.getType(), loweredOffset->base.getType(),
                  loweredOffset->intrinsicOffset.getType(),
                  adaptor.getAlignIn().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    if (usePostIntrinsic && loweredOffset->updatedBase) {
      rewriter.replaceOp(
          op, ValueRange{call.getResult(0), loweredOffset->updatedBase});
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
  explicit LowerVsturOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VsturOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsturOp op, pto::VsturOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto postMode = parsePostModeImmediate(op.getMode());
    if (!postMode)
    {
      return rewriter.notifyMatchFailure(op, "unsupported vstur mode immediate");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getAlignOut().getType());
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!resultType || !baseType || adaptor.getAlignIn().getType() != resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstur operand/result types");
    }

    StringRef calleeName = buildVsturCallee(op.getContext());
    Value modeValue = getI32Constant(rewriter, op.getLoc(), *postMode);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    Value value = castToPayloadABI(
        op.getLoc(), adaptor.getValue(), op.getValue().getType(), rewriter);
    SmallVector<Value> args{value, adaptor.getBase(), adaptor.getAlignIn(),
                            modeValue, zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{value.getType(), adaptor.getBase().getType(),
                  adaptor.getAlignIn().getType(), modeValue.getType(),
                  zeroValue.getType()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstarOpPattern final : public OpConversionPattern<pto::VstarOp> {
public:
  explicit LowerVstarOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstarOp op, pto::VstarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstar operand types");
    }

    StringRef calleeName = buildVstarCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  zeroValue.getType()},
        TypeRange{});
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
  explicit LowerVstasOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstasOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstasOp op, pto::VstasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    auto dstType = dyn_cast<pto::PtrType>(op.getDestination().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType || !dstType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstas operand types");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    auto loweredOffset = lowerVPTOElementOffsetForIntrinsic(
        op, adaptor.getDestination(), adaptor.getOffset(),
        dstType.getElementType(), usePostIntrinsic, rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(op, "failed to convert vstas offset");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 1u : 0u)) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert vstas result types");
    }

    StringRef calleeName =
        buildVstasCallee(op.getContext(), usePostIntrinsic);
    Value postValue =
        getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{adaptor.getValue(), loweredOffset->base,
                            loweredOffset->intrinsicOffset, postValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), loweredOffset->base.getType(),
                  loweredOffset->intrinsicOffset.getType(),
                  postValue.getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    return finishPostUpdateStore(op, rewriter, loweredOffset->updatedBase,
                                 call.getOperation(), usePostIntrinsic);
  }

private:
  LoweringState &state;
};

template <typename StoreOp>
class LowerPredicateStoreOpPattern final : public OpConversionPattern<StoreOp> {
public:
  explicit LowerPredicateStoreOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<StoreOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(StoreOp op, typename StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!llvmDestType || !valueType) {
      return rewriter.notifyMatchFailure(
          op, "expected converted predicate-store operand types");
    }

    auto dist = parsePredicateStoreDistImmediate(op.getDist());
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-store dist immediate");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    auto loweredOffset = lowerVPTOPredicateOffsetForIntrinsic(
        op, adaptor.getDestination(), adaptor.getOffset(), usePostIntrinsic,
        rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(
          op, "failed to preserve predicate-store index offset");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 1u : 0u)) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-store result types");
    }

    StringRef calleeName =
        getPredicateStoreCallee<StoreOp>(op.getContext(), usePostIntrinsic);
    SmallVector<Value> args;
    args.push_back(adaptor.getValue());
    args.push_back(loweredOffset->base);
    args.push_back(loweredOffset->intrinsicOffset);
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(usePostIntrinsic ? 1 : 0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{valueType, llvmDestType, rewriter.getI32Type(),
                  rewriter.getI32Type(), rewriter.getI32Type()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (usePostIntrinsic)
    {
      if (loweredOffset->updatedBase) {
        rewriter.replaceOp(op, loweredOffset->updatedBase);
      } else {
        rewriter.replaceOp(op, call.getResults());
      }
    }
    else
    {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerPredicateLoadOpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerPredicateLoadOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 2u : 1u)) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-load result types");
    }
    if (!llvmSourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected converted predicate-load operand/result types");
    }

    auto dist = parsePredicateLoadDistImmediate(op.getDist());
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-load dist immediate");
    }

    auto loweredOffset = lowerVPTOPredicateOffsetForIntrinsic(
        op, adaptor.getSource(), adaptor.getOffset(), usePostIntrinsic,
        rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(
          op, "failed to preserve predicate-load index offset");
    }

    StringRef calleeName =
        getPredicateLoadCallee<LoadOp>(op.getContext(), usePostIntrinsic);
    SmallVector<Value> args;
    args.push_back(loweredOffset->base);
    args.push_back(loweredOffset->intrinsicOffset);
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(usePostIntrinsic ? 1 : 0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmSourceType, rewriter.getI32Type(), rewriter.getI32Type(),
                  rewriter.getI32Type()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (loweredOffset->updatedBase) {
      rewriter.replaceOp(
          op, ValueRange{call.getResult(0), loweredOffset->updatedBase});
    } else {
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }

private:
  LoweringState &state;
};

} // namespace

void populateVPTOVectorMemoryPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      LoweringState &state) {
  patterns.add<LowerVldsOpPattern, LowerVldsx2OpPattern, LowerVsldbOpPattern,
               LowerVldasOpPattern, LowerInitAlignOpPattern,
               LowerVldusOpPattern, LowerSprclrOpPattern,
               LowerSprStoreOpPattern<pto::SprstiOp>,
               LowerSprStoreOpPattern<pto::SprstsOp>,
               LowerVstsOpPattern, LowerVsstbOpPattern,
               LowerVstsx2OpPattern, LowerVstarOpPattern,
               LowerVstasOpPattern,
               LowerPredicateLoadOpPattern<pto::PldiOp>,
               LowerPredicateLoadOpPattern<pto::PldsOp>,
               LowerPredicateStoreOpPattern<pto::PstiOp>,
               LowerPredicateStoreOpPattern<pto::PstsOp>,
               LowerPstuOpPattern, LowerVstusOpPattern,
               LowerVsturOpPattern>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
