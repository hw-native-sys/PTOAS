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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

static FailureOr<SmallVector<Value, 7>> castCopyGmToUbConfig0Operands(
    Operation *anchor, ValueRange operands, Type i64Type) {
  if (operands.size() != 11)
  {
    return failure();
  }
  return castIntegerLikeOperands(anchor, operands,
                                 {2u, 3u, 4u, 5u, 6u, 7u, 8u}, i64Type);
}

static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, ValueRange operands) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  FailureOr<SmallVector<Value, 7>> values =
      castCopyGmToUbConfig0Operands(anchor, operands, builder.getI64Type());
  if (failed(values))
  {
    return failure();
  }
  return packShiftedI64Fields(
      builder, loc, (*values)[0],
      {{(*values)[1], 4}, {(*values)[2], 25}, {(*values)[3], 46},
       {(*values)[4], 52}, {(*values)[5], 58}, {(*values)[6], 60}});
}

static FailureOr<Value>
packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 11)
  {
    return failure();
  }
  return packLoopPair(anchor, operands[9], operands[10]);
}

static FailureOr<Value> packCopyV220Config(Operation *anchor,
                                           ValueRange operands,
                                           unsigned expectedSize) {
  if (operands.size() != expectedSize)
  {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto values = castIntegerLikeOperands(anchor, operands, {2u, 4u},
                                        builder.getI64Type());
  if (failed(values))
  {
    return failure();
  }

  Value oneI64 = getI64Constant(builder, loc, 1);
  Value bytesPer32B = getI64Constant(builder, loc, 5);
  auto lenIn32B =
      builder.create<arith::ShRUIOp>(loc, (*values)[1], bytesPer32B).getResult();
  return packShiftedI64Fields(builder, loc, (*values)[0],
                              {{oneI64, 4}, {lenIn32B, 16}});
}

static FailureOr<Value>
packCopyGmToUbCfgV220(Operation *anchor, ValueRange operands) {
  return packCopyV220Config(anchor, operands, 11);
}

static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 8)
  {
    return failure();
  }

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto values = castIntegerLikeOperands(anchor, operands, {2u, 3u, 4u, 5u},
                                        builder.getI64Type());
  if (failed(values))
  {
    return failure();
  }
  return packShiftedI64Fields(builder, loc, (*values)[0],
                              {{(*values)[1], 4}, {(*values)[2], 25},
                               {(*values)[3], 60}});
}

static FailureOr<Value>
packCopyUbToGmConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 8)
  {
    return failure();
  }
  return packLoopPair(anchor, operands[6], operands[7]);
}

static FailureOr<Value>
packCopyUbToGmCfgV220(Operation *anchor, ValueRange operands) {
  return packCopyV220Config(anchor, operands, 8);
}

static FailureOr<Value> buildUbufUnaryConfig(Operation *anchor,
                                             ConversionPatternRewriter &rewriter,
                                             Value repeat, Value dstBlockStride,
                                             Value srcBlockStride,
                                             Value dstRepeatStride,
                                             Value srcRepeatStride) {
  Type i64Type = rewriter.getI64Type();
  Value repeatI64 = castIntegerLikeTo(anchor, repeat, i64Type);
  Value dstBlockStrideI64 =
      castIntegerLikeTo(anchor, dstBlockStride, i64Type);
  Value srcBlockStrideI64 =
      castIntegerLikeTo(anchor, srcBlockStride, i64Type);
  Value dstRepeatStrideI64 =
      castIntegerLikeTo(anchor, dstRepeatStride, i64Type);
  Value srcRepeatStrideI64 =
      castIntegerLikeTo(anchor, srcRepeatStride, i64Type);
  if (!repeatI64 || !dstBlockStrideI64 || !srcBlockStrideI64 ||
      !dstRepeatStrideI64 || !srcRepeatStrideI64) {
    return failure();
  }

  return packMaskedI64Fields(
      rewriter, anchor->getLoc(), getI64Constant(rewriter, anchor->getLoc(), 0),
      {{repeatI64, 56},
       {dstBlockStrideI64, 0},
       {srcBlockStrideI64, 16},
       {dstRepeatStrideI64, 32},
       {srcRepeatStrideI64, 40}},
      0xff);
}

template <typename UBOp>
static std::string getUBufBinaryCallee(StringRef elemFrag) {
  StringRef stem;
  if constexpr (std::is_same_v<UBOp, pto::UBVaddOp>) stem = "VADD";
  else if constexpr (std::is_same_v<UBOp, pto::UBVsubOp>) stem = "VSUB";
  else if constexpr (std::is_same_v<UBOp, pto::UBVmulOp>) stem = "VMUL";
  else if constexpr (std::is_same_v<UBOp, pto::UBVdivOp>) stem = "VDIV";
  else if constexpr (std::is_same_v<UBOp, pto::UBVmaxOp>) stem = "VMAX";
  else if constexpr (std::is_same_v<UBOp, pto::UBVminOp>) stem = "VMIN";
  else if constexpr (std::is_same_v<UBOp, pto::UBVandOp>) stem = "VAND";
  else if constexpr (std::is_same_v<UBOp, pto::UBVorOp>) stem = "VOR";
  else if constexpr (std::is_same_v<UBOp, pto::UBVaddReluOp>) stem = "VADDRELU";
  return stem.empty() ? std::string() : ("llvm.hivm." + stem.str() + "." + elemFrag.str());
}


static FailureOr<StringRef> buildCopyGmToUbCallee(MLIRContext *context,
                                                  Type sourceType,
                                                  const std::string &march,
                                                  bool hasPadding) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  Type elementType = ptrType.getElementType();

  auto getElementSuffix = [&]() -> std::string {
    if ((isa<IntegerType>(elementType) &&
         cast<IntegerType>(elementType).getWidth() == 64) ||
        elementType.isF64()) {
      return "s32";
    }
    return getCopyElementFragment(elementType);
  };

  if (march == "dav-c220-vec") {
    if (hasPadding) {
      std::string elem = getElementSuffix();
      if (elem.empty())
      {
        return failure();
      }
      return StringAttr::get(context,
                             "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2." + elem)
          .getValue();
    }
    return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.v220").getValue();
  }

  std::string elem = getElementSuffix();
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2." + elem +
                                      ".DV")
      .getValue();
}

static StringRef buildCopyUbToGmCallee(MLIRContext *context,
                                       const std::string &march) {
  if (march == "dav-c220-vec") {
    return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.OUT.v220.1")
        .getValue();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV")
      .getValue();
}

// Creates a call to a planned function and records the declaration that must
// be emitted alongside the lowered module.
static void planVPTOLLVMCall(Location loc, StringRef calleeName,
                             TypeRange argTypes, ValueRange args,
                             ConversionPatternRewriter &rewriter,
                             LoweringState &state) {
  auto funcType = rewriter.getFunctionType(argTypes, TypeRange{});
  rewriter.create<func::CallOp>(loc, calleeName, TypeRange{}, args);
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
}

// Callee selection for the GM<->UBUF copy ops.
template <typename CopyOp>
static FailureOr<StringRef> getCopyOpCallee(CopyOp op,
                                            const std::string &march,
                                            bool hasPadding) {
  if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>) {
    return buildCopyGmToUbCallee(op.getContext(), op.getSource().getType(),
                                 march, hasPadding);
  }
  return buildCopyUbToGmCallee(op.getContext(), march);
}

struct CopyGmUbConfigs {
  Value config0;
  Value config1;
  bool singleConfig;
};

// Materializes the packed config arguments for a GM<->UBUF copy op. The c220
// signatures take a single config while the legacy ones need two configs.
template <typename CopyOp>
static FailureOr<CopyGmUbConfigs>
materializeCopyGmUbConfigs(CopyOp op, typename CopyOp::Adaptor adaptor,
                           const std::string &march, bool hasPadding) {
  constexpr bool isGmUb = std::is_same_v<CopyOp, pto::CopyGmToUbufOp>;
  bool isC220 = march == "dav-c220-vec" || march == "dav-c220-cube";
  bool useA3NonPadded = isC220 && isGmUb && !hasPadding;
  bool useA3UbGm = isC220 && !isGmUb;
  bool useSingleConfig = useA3NonPadded || useA3UbGm;
  FailureOr<Value> config0 = failure();
  FailureOr<Value> config1 = failure();
  if (useA3NonPadded)
  {
    config0 = packCopyGmToUbCfgV220(op, adaptor.getOperands());
  } else if (useA3UbGm) {
    config0 = packCopyUbToGmCfgV220(op, adaptor.getOperands());
  } else if constexpr (isGmUb) {
    config0 = packCopyGmToUbConfig0(op, adaptor.getOperands());
    config1 = packCopyGmToUbConfig1(op, adaptor.getOperands());
  } else {
    config0 = packCopyUbToGmConfig0(op, adaptor.getOperands());
    config1 = packCopyUbToGmConfig1(op, adaptor.getOperands());
  }
  if (failed(config0) || (!useSingleConfig && failed(config1)))
    return failure();
  return CopyGmUbConfigs{*config0, useSingleConfig ? Value() : *config1,
                         useSingleConfig};
}

// Callee name builders for the ubuf shift/scalar/unary arithmetic families.
template <typename ShiftOp>
static FailureOr<std::string> buildShiftOpCallee(StringRef elemFrag) {
  StringRef suffix = elemFrag;
  if (suffix == "s16")
    suffix = "u16";
  else if (suffix == "s32")
    suffix = "u32";
  StringRef stem;
  if constexpr (std::is_same_v<ShiftOp, pto::UBVshlOp>) stem = "VSHL";
  if constexpr (std::is_same_v<ShiftOp, pto::UBVshrOp>) stem = "VSHR";
  if (stem.empty())
    return failure();
  return std::string("llvm.hivm.") + stem.str() + "." + suffix.str();
}

template <typename ScalarOp>
static FailureOr<std::string> buildScalarBinaryOpCallee(StringRef elemFrag) {
  StringRef stem;
  if constexpr (std::is_same_v<ScalarOp, pto::UBVmulSOp>) stem = "VMULS";
  else if constexpr (std::is_same_v<ScalarOp, pto::UBVaddSOp>) stem = "VADDS";
  else if constexpr (std::is_same_v<ScalarOp, pto::UBVmaxSOp>) stem = "VMAXS";
  else if constexpr (std::is_same_v<ScalarOp, pto::UBVminSOp>) stem = "VMINS";
  if (stem.empty())
    return failure();
  return std::string("llvm.hivm.") + stem.str() + "." + elemFrag.str();
}

template <typename UnaryOp>
static FailureOr<std::string> buildUnaryOpCallee(StringRef elemFrag) {
  StringRef suffix = elemFrag;
  if (suffix == "s16")
    suffix = "u16";
  StringRef stem;
  if constexpr (std::is_same_v<UnaryOp, pto::UBVnotOp>) stem = "VNOT";
  else if constexpr (std::is_same_v<UnaryOp, pto::UBVabsOp>) stem = "VABS";
  else if constexpr (std::is_same_v<UnaryOp, pto::UBVreluOp>) {
    if (suffix == "u16" || suffix == "u32")
      return failure();
    stem = "VRELU";
  } else if constexpr (std::is_same_v<UnaryOp, pto::UBVexpOp>) {
    stem = "VEXP";
  } else if constexpr (std::is_same_v<UnaryOp, pto::UBVlnOp>) {
    stem = "VLN";
  } else if constexpr (std::is_same_v<UnaryOp, pto::UBVsqrtOp>) {
    stem = "VSQRT";
  } else if constexpr (std::is_same_v<UnaryOp, pto::UBVrsqrtOp>) {
    stem = "VRSQRT";
  }
  if (stem.empty())
    return failure();
  return std::string("llvm.hivm.") + stem.str() + "." + suffix.str();
}

// Shared materialization of the unary-style config (repeat/block/repeat
// strides) used by the shift, scalar-binary and unary op families.
template <typename VecOp, typename Adaptor>
static FailureOr<Value> materializeUbufUnaryConfig(
    VecOp op, Adaptor adaptor, ConversionPatternRewriter &rewriter) {
  return buildUbufUnaryConfig(op, rewriter, adaptor.getRepeat(),
                              adaptor.getDstBlockStride(),
                              adaptor.getSrcBlockStride(),
                              adaptor.getDstRepeatStride(),
                              adaptor.getSrcRepeatStride());
}

// pto.ub.vgatherb -> llvm.hivm.VGATHERB.b16/.b32 helpers.
static FailureOr<std::string> buildVgatherbCallee(pto::UBVgatherbOp op) {
  auto ptrType = mlir::cast<pto::PtrType>(op.getDst().getType());
  Type elemType = ptrType.getElementType();
  unsigned width = pto::getPTOStorageElemBitWidth(elemType);
  if (width != 16 && width != 32)
    return failure();
  return std::string("llvm.hivm.VGATHERB.") + (width == 16 ? "b16" : "b32");
}

static Value buildVgatherbConfig(pto::UBVgatherbOp op,
                                 pto::UBVgatherbOp::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Type i64Ty = rewriter.getI64Type();
  auto constI64 = [&](uint64_t v) -> Value {
    return rewriter.create<arith::ConstantOp>(loc,
                                              rewriter.getI64IntegerAttr(v));
  };
  auto getI64 = [&](Value v) -> Value {
    return castIntegerLikeTo(op, v, i64Ty);
  };
  // config[31:0] = source data address (low 32 bits of the src pointer).
  // Trace back through castptr to get the planned UB offset.
  Value srcAddr;
  if (auto *defOp = op.getSrc().getDefiningOp()) {
    if (auto castOp = dyn_cast<pto::CastPtrOp>(defOp)) {
      srcAddr = castOp.getOperand();
    }
  }
  if (!srcAddr)
    srcAddr = rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, adaptor.getSrc());
  Value config =
      rewriter.create<arith::AndIOp>(loc, srcAddr, constI64(0xffffffff));
  return packMaskedI64Fields(
      rewriter, loc, config,
      {{getI64(adaptor.getDstRepeatStride()), 32},
       {getI64(adaptor.getDstBlockStride()), 40},
       {getI64(adaptor.getRepeat()), 56}},
      0xff);
}

template <typename CopyOp>
class LowerCopyOpPattern final : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state, const std::string &march)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state),
        march(march) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isGmUb = std::is_same_v<CopyOp, pto::CopyGmToUbufOp>;

    bool hasPadding = false;
    if constexpr (isGmUb)
    {
      hasPadding = op->hasAttr("has_pad");
    }

    FailureOr<StringRef> calleeName = getCopyOpCallee(op, march, hasPadding);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported copy VPTO signature");
    }

    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");
    }

    FailureOr<CopyGmUbConfigs> configs =
        materializeCopyGmUbConfigs(op, adaptor, march, hasPadding);
    if (failed(configs))
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0],
                            configs->config0};
    SmallVector<Type> argTypes{llvmDestType, llvmSourceType,
                               rewriter.getI64Type()};
    if (!configs->singleConfig) {
      args.push_back(configs->config1);
      argTypes.push_back(rewriter.getI64Type());
    }

    planVPTOLLVMCall(op.getLoc(), *calleeName, argTypes, args, rewriter, state);
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
  const std::string &march;
};

template <typename UBOp>
static FailureOr<Value> materializeUBufBinaryConfig(
    UBOp op, typename UBOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Type i64Ty = rewriter.getI64Type();
  Value config = getI64Constant(rewriter, loc, 1ULL << 56);
  SmallVector<std::pair<Value, uint64_t>> fields = {
      {adaptor.getRepeat(), 0}, {adaptor.getDstBlockStride(), 8},
      {adaptor.getSrc0BlockStride(), 16}, {adaptor.getSrc1BlockStride(), 24},
      {adaptor.getDstRepeatStride(), 32}, {adaptor.getSrc0RepeatStride(), 40},
      {adaptor.getSrc1RepeatStride(), 48}};
  SmallVector<std::pair<Value, uint64_t>> converted;
  for (auto [value, amount] : fields) {
    Value casted = castIntegerLikeTo(op, value, i64Ty);
    if (!casted)
      return failure();
    converted.push_back({casted, amount});
  }
  return packMaskedI64Fields(rewriter, loc, config, converted, 0xff);
}

template <typename UBOp>
class LowerUBufBinaryOpPattern final : public OpConversionPattern<UBOp> {
public:
  explicit LowerUBufBinaryOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UBOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UBOp op, typename UBOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = mlir::cast<pto::PtrType>(op.getSrc0().getType());
    Type elemType = ptrType.getElementType();
    std::string elemFrag = getElementTypeFragment(elemType);
    if (elemFrag.empty()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for ubuf binary op");
    }

    std::string calleeName = getUBufBinaryCallee<UBOp>(elemFrag);
    if (calleeName.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported ubuf binary op");
    }

    Value dst = adaptor.getDst();
    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    if (!dst || !src0 || !src1 ||
        !isa<LLVM::LLVMPointerType>(dst.getType()) ||
        !isa<LLVM::LLVMPointerType>(src0.getType()) ||
        !isa<LLVM::LLVMPointerType>(src1.getType())) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted ubuf binary operand types");
    }

    FailureOr<Value> config =
        materializeUBufBinaryConfig(op, adaptor, rewriter);
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "invalid ubuf binary config operand");

    auto funcType = rewriter.getFunctionType(
        TypeRange{dst.getType(), src0.getType(), src1.getType(),
                  rewriter.getI64Type()},
        TypeRange{});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{},
        ValueRange{dst, src0, src1, *config});
    (void)call;
    state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

// pto.ub.vgatherb -> llvm.hivm.VGATHERB.b16/.b32(dst_ptr, offset_ptr, i64 config)
// Config (decoded from bisheng IR, see docs/designs/a2a3-vpto-tgather.md):
//   srcAddr[31:0] | dstRepeatStride[39:32] | dstBlockStride[47:40]
//   | reserved[55:48]=0 | repeat[63:56]
// The 2nd pointer operand is the offset buffer; the source data base address
// (low 32 bits of the src pointer) is packed into config[31:0].
class LowerUBVgatherbOpPattern final
    : public OpConversionPattern<pto::UBVgatherbOp> {
public:
  explicit LowerUBVgatherbOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::UBVgatherbOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::UBVgatherbOp op, pto::UBVgatherbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dst = adaptor.getDst();
    Value offset = adaptor.getOffset();
    Value src = adaptor.getSrc();
    if (!dst || !offset || !src ||
        !isa<LLVM::LLVMPointerType>(dst.getType()) ||
        !isa<LLVM::LLVMPointerType>(offset.getType()) ||
        !isa<LLVM::LLVMPointerType>(src.getType())) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted ub.vgatherb operand types");
    }

    FailureOr<std::string> calleeName = buildVgatherbCallee(op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element width for ub.vgatherb");
    }

    Location loc = op.getLoc();
    Value config = buildVgatherbConfig(op, adaptor, rewriter);
    planVPTOLLVMCall(loc, *calleeName,
                     TypeRange{dst.getType(), offset.getType(),
                               rewriter.getI64Type()},
                     ValueRange{dst, offset, config}, rewriter, state);
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

// pto.ub.vgather -> llvm.hivm.VGATHER.b16/.b32(dst_ptr, src_ptr, i64 config)
// Config: offsetAddr[31:0] | dstRepeatStride[39:32] | repeat[63:56].
class LowerUBVgatherOpPattern final
    : public OpConversionPattern<pto::UBVgatherOp> {
public:
  explicit LowerUBVgatherOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::UBVgatherOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::UBVgatherOp op, pto::UBVgatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dst = adaptor.getDst();
    Value src = adaptor.getSrc();
    if (!dst || !src || !isa<LLVM::LLVMPointerType>(dst.getType()) ||
        !isa<LLVM::LLVMPointerType>(src.getType())) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted ub.vgather operand types");
    }

    auto ptrType = mlir::cast<pto::PtrType>(op.getDst().getType());
    Type elemType = ptrType.getElementType();
    unsigned width = pto::getPTOStorageElemBitWidth(elemType);
    if (width != 16 && width != 32) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element width for ub.vgather");
    }
    std::string calleeName =
        std::string("llvm.hivm.VGATHER.") + ((width == 16) ? "b16" : "b32");

    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();
    auto constI64 = [&](uint64_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI64IntegerAttr(v));
    };
    auto getI64 = [&](Value v) -> Value {
      return castIntegerLikeTo(op, v, i64Ty);
    };
    auto maskByte = [&](Value v) -> Value {
      return rewriter.create<arith::AndIOp>(loc, v, constI64(0xff));
    };
    auto shl = [&](Value v, uint64_t amount) -> Value {
      return rewriter.create<arith::ShLIOp>(loc, v, constI64(amount));
    };

    Value config = rewriter.create<arith::AndIOp>(
        loc, getI64(adaptor.getOffsetAddr()), constI64(0xffffffff));
    config = rewriter.create<arith::OrIOp>(
        loc, config, shl(maskByte(getI64(adaptor.getDstRepeatStride())), 32));
    config = rewriter.create<arith::OrIOp>(
        loc, config, shl(maskByte(getI64(adaptor.getRepeat())), 56));

    auto funcType = rewriter.getFunctionType(
        TypeRange{dst.getType(), src.getType(), rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{dst, src, config});
    state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ShiftOp>
class LowerUBufShiftOpPattern final : public OpConversionPattern<ShiftOp> {
public:
  explicit LowerUBufShiftOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ShiftOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ShiftOp op, typename ShiftOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = mlir::cast<pto::PtrType>(op.getSrc().getType());
    Type elemType = ptrType.getElementType();
    std::string elemFrag = getElementTypeFragment(elemType);
    if (elemFrag.empty()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for ubuf shift op");
    }

    FailureOr<std::string> calleeName = buildShiftOpCallee<ShiftOp>(elemFrag);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported ubuf shift op");
    }

    Value dst = adaptor.getDst();
    Value src = adaptor.getSrc();
    if (!dst || !src ||
        !isa<LLVM::LLVMPointerType>(dst.getType()) ||
        !isa<LLVM::LLVMPointerType>(src.getType())) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted ubuf shift operand types");
    }

    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();
    FailureOr<Value> config = materializeUbufUnaryConfig(op, adaptor, rewriter);
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "invalid ubuf shift config operands");
    }

    Value shiftDist = castIntegerLikeTo(op, adaptor.getShiftDist(), i64Ty);

    if constexpr (std::is_same_v<ShiftOp, pto::UBVshlOp>) {
      planVPTOLLVMCall(loc, *calleeName,
                       TypeRange{dst.getType(), src.getType(), i64Ty, i64Ty},
                       ValueRange{dst, src, shiftDist, *config}, rewriter,
                       state);
    } else {
      Value roundZero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(0));
      planVPTOLLVMCall(loc, *calleeName,
                       TypeRange{dst.getType(), src.getType(), i64Ty, i64Ty,
                                 i64Ty},
                       ValueRange{dst, src, shiftDist, *config, roundZero},
                       rewriter, state);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

// LowerUBufScalarBinaryPattern — scalar-tile binary ops (VADDS/VMULS/VMAXS/VMINS).
// Unlike VSHL/VSHR, these have signed intrinsics (s16/s32, not u16/u32) and
// pass the scalar as a float for f32/f16 element types.
template <typename ScalarOp>
class LowerUBufScalarBinaryPattern final : public OpConversionPattern<ScalarOp> {
public:
  explicit LowerUBufScalarBinaryPattern(TypeConverter &typeConverter,
                                     MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ScalarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ScalarOp op, typename ScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = mlir::cast<pto::PtrType>(op.getSrc().getType());
    Type elemType = ptrType.getElementType();
    std::string elemFrag = getElementTypeFragment(elemType);
    if (elemFrag.empty()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for ubuf scalar mul op");
    }

    FailureOr<std::string> calleeName = buildScalarBinaryOpCallee<ScalarOp>(elemFrag);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported ubuf scalar binary op");
    }

    Value dst = adaptor.getDst();
    Value src = adaptor.getSrc();
    if (!dst || !src ||
        !isa<LLVM::LLVMPointerType>(dst.getType()) ||
        !isa<LLVM::LLVMPointerType>(src.getType())) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted ubuf scalar binary operand types");
    }

    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();
    FailureOr<Value> config = materializeUbufUnaryConfig(op, adaptor, rewriter);
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "invalid ubuf unary config operands");
    }

    Value scalarI64 = castIntegerLikeTo(op, adaptor.getShiftDist(), i64Ty);

    if (elemType.isF32() || elemType.isF16()) {
      unsigned width = elemType.isF32() ? 32 : 16;
      Type intTy = rewriter.getIntegerType(width);
      Type floatTy = elemType.isF32()
                          ? rewriter.getF32Type()
                          : rewriter.getF16Type();
      Value trunced = rewriter.create<arith::TruncIOp>(loc, intTy, scalarI64);
      Value scalarFloat = rewriter.create<LLVM::BitcastOp>(loc, floatTy, trunced);
      planVPTOLLVMCall(loc, *calleeName,
                       TypeRange{dst.getType(), src.getType(), floatTy, i64Ty},
                       ValueRange{dst, src, scalarFloat, *config}, rewriter,
                       state);
    } else {
      planVPTOLLVMCall(loc, *calleeName,
                       TypeRange{dst.getType(), src.getType(), i64Ty, i64Ty},
                       ValueRange{dst, src, scalarI64, *config}, rewriter,
                       state);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnaryOp>
class LowerUBufUnaryOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUBufUnaryOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = mlir::cast<pto::PtrType>(op.getSrc().getType());
    Type elemType = ptrType.getElementType();
    std::string elemFrag = getElementTypeFragment(elemType);
    if (elemFrag.empty()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for ubuf unary op");
    }

    FailureOr<std::string> calleeName = buildUnaryOpCallee<UnaryOp>(elemFrag);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported ubuf unary op");
    }

    Value dst = adaptor.getDst();
    Value src = adaptor.getSrc();
    if (!dst || !src ||
        !isa<LLVM::LLVMPointerType>(dst.getType()) ||
        !isa<LLVM::LLVMPointerType>(src.getType())) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted ubuf unary operand types");
    }

    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();
    FailureOr<Value> config = materializeUbufUnaryConfig(op, adaptor, rewriter);
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op,
                                         "invalid ubuf unary config operands");
    }

    planVPTOLLVMCall(loc, *calleeName,
                     TypeRange{dst.getType(), src.getType(), i64Ty},
                     ValueRange{dst, src, *config}, rewriter, state);
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

} // namespace

static void populateVPTOUbufArithmeticPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                LoweringState &state) {
  patterns.add<LowerUBufBinaryOpPattern<pto::UBVaddOp>,
               LowerUBufBinaryOpPattern<pto::UBVsubOp>,
               LowerUBufBinaryOpPattern<pto::UBVmulOp>,
               LowerUBufBinaryOpPattern<pto::UBVdivOp>,
               LowerUBufBinaryOpPattern<pto::UBVmaxOp>,
               LowerUBufBinaryOpPattern<pto::UBVminOp>,
               LowerUBufBinaryOpPattern<pto::UBVandOp>,
               LowerUBufBinaryOpPattern<pto::UBVorOp>,
               LowerUBufBinaryOpPattern<pto::UBVaddReluOp>,
               LowerUBufUnaryOpPattern<pto::UBVnotOp>,
               LowerUBufUnaryOpPattern<pto::UBVabsOp>,
               LowerUBufUnaryOpPattern<pto::UBVreluOp>,
               LowerUBufUnaryOpPattern<pto::UBVexpOp>,
               LowerUBufUnaryOpPattern<pto::UBVlnOp>,
               LowerUBufUnaryOpPattern<pto::UBVsqrtOp>,
               LowerUBufUnaryOpPattern<pto::UBVrsqrtOp>,
               LowerUBufShiftOpPattern<pto::UBVshlOp>,
               LowerUBufShiftOpPattern<pto::UBVshrOp>,
               LowerUBufScalarBinaryPattern<pto::UBVmulSOp>,
               LowerUBufScalarBinaryPattern<pto::UBVaddSOp>,
               LowerUBufScalarBinaryPattern<pto::UBVmaxSOp>,
               LowerUBufScalarBinaryPattern<pto::UBVminSOp>>(
      typeConverter, patterns.getContext(), state);
}

void populateVPTOUbufPatterns(TypeConverter &typeConverter,
                              RewritePatternSet &patterns,
                              LoweringState &state,
                              const std::string &march) {
  patterns.add<LowerCopyOpPattern<pto::CopyGmToUbufOp>>(
      typeConverter, patterns.getContext(), state, march);
  patterns.add<LowerCopyOpPattern<pto::CopyUbufToGmOp>>(
      typeConverter, patterns.getContext(), state, march);

  if (march == "dav-c220-vec") {
    populateVPTOUbufArithmeticPatterns(typeConverter, patterns, state);
    populateVPTOMemoryUbufPatterns(typeConverter, patterns, state);
    patterns.add<LowerUBVgatherbOpPattern>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBVgatherOpPattern>(
        typeConverter, patterns.getContext(), state);
    populateVPTOMemoryMaskPatterns(typeConverter, patterns, state);
  }
}

} // namespace mlir::pto
