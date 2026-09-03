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
  if (operands.size() != 11) {
    return failure();
  }

  SmallVector<Value, 7> values;
  for (unsigned index : {2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
    Value value = castIntegerLikeTo(anchor, operands[index], i64Type);
    if (!value) {
      return failure();
    }
    values.push_back(value);
  }
  return values;
}

static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, ValueRange operands) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  FailureOr<SmallVector<Value, 7>> values =
      castCopyGmToUbConfig0Operands(anchor, operands, builder.getI64Type());
  if (failed(values))
    return failure();
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

static FailureOr<Value>
packCopyGmToUbCfgV220(Operation *anchor, ValueRange operands) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value lenBurst = getI64Operand(4);
  if (!sid || !lenBurst)
  {
    return failure();
  }

  Value oneI64 = getI64Constant(builder, loc, 1);
  Value bytesPer32B = getI64Constant(builder, loc, 5);
  auto lenIn32B =
      builder.create<arith::ShRUIOp>(loc, lenBurst, bytesPer32B).getResult();
  return packShiftedI64Fields(builder, loc, sid,
                              {{oneI64, 4}, {lenIn32B, 16}});
}

[[maybe_unused]] static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, Value sid, Value nBurst,
                      Value lenBurst, Value leftPadding, Value rightPadding,
                      Value dataSelect, Value cacheCtl) {
  SmallVector<Value, 11> operands(11);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = leftPadding;
  operands[6] = rightPadding;
  operands[7] = dataSelect;
  operands[8] = cacheCtl;
  return packCopyGmToUbConfig0(anchor, operands);
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

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value l2CacheCtl = getI64Operand(5);
  if (!sid || !nBurst || !lenBurst || !l2CacheCtl)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, sid,
                              {{nBurst, 4}, {lenBurst, 25},
                               {l2CacheCtl, 60}});
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
  if (operands.size() != 8)
  {
    return failure();
  }

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value lenBurst = getI64Operand(4);
  if (!sid || !lenBurst)
  {
    return failure();
  }

  Value oneI64 = getI64Constant(builder, loc, 1);
  Value bytesPer32B = getI64Constant(builder, loc, 5);
  auto lenIn32B =
      builder.create<arith::ShRUIOp>(loc, lenBurst, bytesPer32B).getResult();
  return packShiftedI64Fields(builder, loc, sid,
                              {{oneI64, 4}, {lenIn32B, 16}});
}

[[maybe_unused]] static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, Value sid, Value nBurst,
                      Value lenBurst, Value l2CacheCtl) {
  SmallVector<Value, 8> operands(8);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = l2CacheCtl;
  return packCopyUbToGmConfig0(anchor, operands);
}

static FailureOr<Value>
packCopyUbToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7)
  {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!nBurst || !lenBurst || !srcStride || !dstStride)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, nBurst,
                              {{lenBurst, 16}, {srcStride, 32},
                               {dstStride, 48}});
}

static FailureOr<Value>
packCopyCbufToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7)
  {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!sid || !nBurst || !lenBurst || !srcStride || !dstStride)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, sid,
                              {{nBurst, 4}, {lenBurst, 16},
                               {srcStride, 32}, {dstStride, 48}});
}

static FailureOr<Value>
packCopyUbToCbufConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7)
  {
    return failure();
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!sid || !nBurst || !lenBurst || !srcStride || !dstStride)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, sid,
                              {{nBurst, 4}, {lenBurst, 16},
                               {srcStride, 32}, {dstStride, 48}});
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

static StringRef buildCopyUbToUbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.UB.v310").getValue();
}

static StringRef buildCopyCbufToUbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.L1.TO.UB.v310").getValue();
}

static StringRef buildCopyUbToCbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.L1.v310").getValue();
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

    FailureOr<StringRef> calleeName = failure();
    if constexpr (isGmUb) {
      calleeName = buildCopyGmToUbCallee(op.getContext(), op.getSource().getType(),
                                         march, hasPadding);
    } else {
      calleeName = buildCopyUbToGmCallee(op.getContext(), march);
    }
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
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0],
                            *config0};
    SmallVector<Type> argTypes{llvmDestType, llvmSourceType,
                               rewriter.getI64Type()};
    if (!useSingleConfig) {
      args.push_back(*config1);
      argTypes.push_back(rewriter.getI64Type());
    }

    auto funcType = rewriter.getFunctionType(argTypes, TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
  const std::string &march;
};

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

    std::string calleeName;
    if constexpr (std::is_same_v<UBOp, pto::UBVaddOp>)
    {
      calleeName = "llvm.hivm.VADD." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVsubOp>) {
      calleeName = "llvm.hivm.VSUB." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVmulOp>) {
      calleeName = "llvm.hivm.VMUL." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVdivOp>) {
      calleeName = "llvm.hivm.VDIV." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVmaxOp>) {
      calleeName = "llvm.hivm.VMAX." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVminOp>) {
      calleeName = "llvm.hivm.VMIN." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVandOp>) {
      calleeName = "llvm.hivm.VAND." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVorOp>) {
      calleeName = "llvm.hivm.VOR." + elemFrag;
    } else if constexpr (std::is_same_v<UBOp, pto::UBVaddReluOp>) {
      calleeName = "llvm.hivm.VADDRELU." + elemFrag;
    } else {
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

    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();
    Value config = getI64Constant(rewriter, loc, 1ULL << 56);
    SmallVector<std::pair<Value, uint64_t>> fields = {
        {adaptor.getRepeat(), 0},
        {adaptor.getDstBlockStride(), 8},
        {adaptor.getSrc0BlockStride(), 16},
        {adaptor.getSrc1BlockStride(), 24},
        {adaptor.getDstRepeatStride(), 32},
        {adaptor.getSrc0RepeatStride(), 40},
        {adaptor.getSrc1RepeatStride(), 48}};
    SmallVector<std::pair<Value, uint64_t>> convertedFields;
    for (auto [value, amount] : fields) {
      Value converted = castIntegerLikeTo(op, value, i64Ty);
      if (!converted) {
        return rewriter.notifyMatchFailure(op, "invalid ubuf binary config operand");
      }
      convertedFields.push_back({converted, amount});
    }
    config = packMaskedI64Fields(rewriter, loc, config, convertedFields, 0xff);

    auto funcType = rewriter.getFunctionType(
        TypeRange{dst.getType(), src0.getType(), src1.getType(),
                  rewriter.getI64Type()},
        TypeRange{});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{},
        ValueRange{dst, src0, src1, config});
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

    auto ptrType = mlir::cast<pto::PtrType>(op.getDst().getType());
    Type elemType = ptrType.getElementType();
    unsigned width = pto::getPTOStorageElemBitWidth(elemType);
    if (width != 16 && width != 32) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element width for ub.vgatherb");
    }
    std::string calleeName =
        std::string("llvm.hivm.VGATHERB.") + ((width == 16) ? "b16" : "b32");

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
    // Trace back through castptr to get the planned UB offset, matching the
    // address loaded from Tile host_ptr metadata by the PTO-ISA reference.
    Value srcAddr;
    if (auto *defOp = op.getSrc().getDefiningOp()) {
      if (auto castOp = dyn_cast<pto::CastPtrOp>(defOp)) {
        srcAddr = castOp.getOperand();
      }
    }
    if (!srcAddr) {
      srcAddr = rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, src);
    }
    Value config =
        rewriter.create<arith::AndIOp>(loc, srcAddr, constI64(0xffffffff));
    config = packMaskedI64Fields(
        rewriter, loc, config,
        {{getI64(adaptor.getDstRepeatStride()), 32},
         {getI64(adaptor.getDstBlockStride()), 40},
         {getI64(adaptor.getRepeat()), 56}},
        0xff);

    auto funcType = rewriter.getFunctionType(
        TypeRange{dst.getType(), offset.getType(), rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{dst, offset, config});
    state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
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

    if (elemFrag == "s16")
    {
      elemFrag = "u16";
    } else if (elemFrag == "s32") {
      elemFrag = "u32";
    }

    std::string calleeName;
    if constexpr (std::is_same_v<ShiftOp, pto::UBVshlOp>)
    {
      calleeName = "llvm.hivm.VSHL." + elemFrag;
    } else if constexpr (std::is_same_v<ShiftOp, pto::UBVshrOp>) {
      calleeName = "llvm.hivm.VSHR." + elemFrag;
    } else {
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
    // Unary config layout (same as VABS):
    //   repeat[63:56], dstBlkStride[15:0], srcBlkStride[31:16],
    //   dstRepStride[39:32], srcRepStride[51:40]
    Type i64Ty = rewriter.getI64Type();
    auto getI64 = [&](Value v) -> Value {
      return castIntegerLikeTo(op, v, i64Ty);
    };
    FailureOr<Value> config = buildUbufUnaryConfig(
        op, rewriter, adaptor.getRepeat(), adaptor.getDstBlockStride(),
        adaptor.getSrcBlockStride(), adaptor.getDstRepeatStride(),
        adaptor.getSrcRepeatStride());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "invalid ubuf shift config operands");
    }

    Value shiftDist = getI64(adaptor.getShiftDist());

    if constexpr (std::is_same_v<ShiftOp, pto::UBVshlOp>) {
      auto funcType = rewriter.getFunctionType(
          TypeRange{dst.getType(), src.getType(), i64Ty, i64Ty},
          TypeRange{});
      rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                    ValueRange{dst, src, shiftDist, *config});
      state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
    } else {
      Value roundZero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(0));
      auto funcType = rewriter.getFunctionType(
          TypeRange{dst.getType(), src.getType(), i64Ty, i64Ty, i64Ty},
          TypeRange{});
      rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                               ValueRange{dst, src, shiftDist, *config,
                                               roundZero});
      state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
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

    // Scalar-tile ops keep signed intrinsic names (s16/s32).
    std::string calleeName;
    if constexpr (std::is_same_v<ScalarOp, pto::UBVmulSOp>)
    {
      calleeName = "llvm.hivm.VMULS." + elemFrag;
    } else if constexpr (std::is_same_v<ScalarOp, pto::UBVaddSOp>) {
      calleeName = "llvm.hivm.VADDS." + elemFrag;
    } else if constexpr (std::is_same_v<ScalarOp, pto::UBVmaxSOp>) {
      calleeName = "llvm.hivm.VMAXS." + elemFrag;
    } else if constexpr (std::is_same_v<ScalarOp, pto::UBVminSOp>) {
      calleeName = "llvm.hivm.VMINS." + elemFrag;
    } else {
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
    auto getI64 = [&](Value v) -> Value {
      return castIntegerLikeTo(op, v, i64Ty);
    };
    // Unary config layout (same as VABS/VSHR): repeat[63:56]
    FailureOr<Value> config = buildUbufUnaryConfig(
        op, rewriter, adaptor.getRepeat(), adaptor.getDstBlockStride(),
        adaptor.getSrcBlockStride(), adaptor.getDstRepeatStride(),
        adaptor.getSrcRepeatStride());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "invalid ubuf unary config operands");
    }

    Value scalarI64 = getI64(adaptor.getShiftDist());

    // For float element types, the scalar was bitcast to i64 for the UB IR.
    // Recover the float value via trunc + bitcast.
    if (elemType.isF32() || elemType.isF16()) {
      unsigned width = elemType.isF32() ? 32 : 16;
      Type intTy = rewriter.getIntegerType(width);
      Type floatTy = elemType.isF32()
                          ? rewriter.getF32Type()
                          : rewriter.getF16Type();
      Value trunced = rewriter.create<arith::TruncIOp>(loc, intTy, scalarI64);
      Value scalarFloat = rewriter.create<LLVM::BitcastOp>(loc, floatTy, trunced);
      auto funcType = rewriter.getFunctionType(
          TypeRange{dst.getType(), src.getType(), floatTy, i64Ty},
          TypeRange{});
      rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                    ValueRange{dst, src, scalarFloat, *config});
      state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
    } else {
      // Integer: VMULS/VADDS/etc .s16/s32 takes i64 scalar directly.
      auto funcType = rewriter.getFunctionType(
          TypeRange{dst.getType(), src.getType(), i64Ty, i64Ty},
          TypeRange{});
      rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                    ValueRange{dst, src, scalarI64, *config});
      state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});
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

    if (elemFrag == "s16")
    {
      elemFrag = "u16";
    }

    std::string calleeName;
    if constexpr (std::is_same_v<UnaryOp, pto::UBVnotOp>)
    {
      calleeName = "llvm.hivm.VNOT." + elemFrag;
    } else if constexpr (std::is_same_v<UnaryOp, pto::UBVabsOp>) {
      calleeName = "llvm.hivm.VABS." + elemFrag;
    } else if constexpr (std::is_same_v<UnaryOp, pto::UBVreluOp>) {
      if (elemFrag == "u16" || elemFrag == "u32") {
        return rewriter.notifyMatchFailure(
            op, "VRELU not available for unsigned integer types");
      }
      calleeName = "llvm.hivm.VRELU." + elemFrag;
    } else if constexpr (std::is_same_v<UnaryOp, pto::UBVexpOp>) {
      calleeName = "llvm.hivm.VEXP." + elemFrag;
    } else if constexpr (std::is_same_v<UnaryOp, pto::UBVlnOp>) {
      calleeName = "llvm.hivm.VLN." + elemFrag;
    } else if constexpr (std::is_same_v<UnaryOp, pto::UBVsqrtOp>) {
      calleeName = "llvm.hivm.VSQRT." + elemFrag;
    } else if constexpr (std::is_same_v<UnaryOp, pto::UBVrsqrtOp>) {
      calleeName = "llvm.hivm.VRSQRT." + elemFrag;
    } else {
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
    // Unary config layout (same as VABS/VSHR): repeat[63:56]
    FailureOr<Value> config = buildUbufUnaryConfig(
        op, rewriter, adaptor.getRepeat(), adaptor.getDstBlockStride(),
        adaptor.getSrcBlockStride(), adaptor.getDstRepeatStride(),
        adaptor.getSrcRepeatStride());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op,
                                         "invalid ubuf unary config operands");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{dst.getType(), src.getType(), i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{dst, src, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName, funcType});

    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerUBSetMaskOpPattern final
    : public OpConversionPattern<pto::UBSetMaskOp> {
public:
  explicit LowerUBSetMaskOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::UBSetMaskOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::UBSetMaskOp op, typename pto::UBSetMaskOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = "llvm.hivm.MOVEMASK";
    Location loc = op.getLoc();

    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});

    Value c0Idx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(0));
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{c0Idx, adaptor.getMask0()});

    Value c1Idx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(1));
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{c1Idx, adaptor.getMask1()});

    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerUBSetMaskCountOpPattern final
    : public OpConversionPattern<pto::UBSetMaskCountOp> {
public:
  explicit LowerUBSetMaskCountOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context)
      : OpConversionPattern<pto::UBSetMaskCountOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::UBSetMaskCountOp op,
                  typename pto::UBSetMaskCountOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    Value ctrl = rewriter.create<pto::GetCtrlOp>(loc, i64Ty).getResult();
    Value bit56 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(56));
    Value set = rewriter
                    .create<pto::Sbitset1Op>(loc, i64Ty, ctrl, bit56)
                    .getResult();
    rewriter.create<pto::SetCtrlOp>(loc, set);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerUBSetMaskNormOpPattern final
    : public OpConversionPattern<pto::UBSetMaskNormOp> {
public:
  explicit LowerUBSetMaskNormOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context)
      : OpConversionPattern<pto::UBSetMaskNormOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::UBSetMaskNormOp op,
                  typename pto::UBSetMaskNormOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    Value ctrl = rewriter.create<pto::GetCtrlOp>(loc, i64Ty).getResult();
    Value bit56 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(56));
    Value reset = rewriter
                      .create<pto::Sbitset0Op>(loc, i64Ty, ctrl, bit56)
                      .getResult();
    rewriter.create<pto::SetCtrlOp>(loc, reset);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerCopyUbufToUbufOpPattern final
    : public OpConversionPattern<pto::CopyUbufToUbufOp> {
public:
  explicit LowerCopyUbufToUbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyUbufToUbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CopyUbufToUbufOp op,
                  pto::CopyUbufToUbufOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");
    }

    FailureOr<Value> config = packCopyUbToUbConfig(op, adaptor.getOperands());
    if (failed(config))
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    StringRef calleeName = buildCopyUbToUbCallee(op.getContext());
    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0],
                            *config};
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmDestType, llvmSourceType, rewriter.getI64Type()},
        TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToUbufOpPattern final
    : public OpConversionPattern<pto::CopyCbufToUbufOp> {
public:
  explicit LowerCopyCbufToUbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToUbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CopyCbufToUbufOp op,
                  pto::CopyCbufToUbufOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
    {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned ubufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, ubufAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ubuf pointer spaces");
    }

    FailureOr<Value> config = packCopyCbufToUbConfig(op, adaptor.getOperands());
    if (failed(config))
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    StringRef calleeName = buildCopyCbufToUbCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyUbufToCbufOpPattern final
    : public OpConversionPattern<pto::CopyUbufToCbufOp> {
public:
  explicit LowerCopyUbufToCbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyUbufToCbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CopyUbufToCbufOp op,
                  pto::CopyUbufToCbufOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
    {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned ubufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::VEC);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, ubufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map ubuf/cbuf pointer spaces");
    }

    FailureOr<Value> config = packCopyUbToCbufConfig(op, adaptor.getOperands());
    if (failed(config))
    {
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");
    }

    StringRef calleeName = buildCopyUbToCbufCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};



} // namespace

void populateVPTOUbufPatterns(TypeConverter &typeConverter,
                               RewritePatternSet &patterns,
                               LoweringState &state,
                               const std::string &march) {
  patterns.add<LowerCopyOpPattern<pto::CopyGmToUbufOp>>(
      typeConverter, patterns.getContext(), state, march);
  patterns.add<LowerCopyOpPattern<pto::CopyUbufToGmOp>>(
      typeConverter, patterns.getContext(), state, march);

  if (march == "dav-c220-vec") {
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVaddOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVsubOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVmulOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVdivOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVmaxOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVminOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVandOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVorOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufBinaryOpPattern<pto::UBVaddReluOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVnotOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVabsOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVreluOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVexpOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVlnOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVsqrtOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufUnaryOpPattern<pto::UBVrsqrtOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufShiftOpPattern<pto::UBVshlOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufShiftOpPattern<pto::UBVshrOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufScalarBinaryPattern<pto::UBVmulSOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufScalarBinaryPattern<pto::UBVaddSOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufScalarBinaryPattern<pto::UBVmaxSOp>>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBufScalarBinaryPattern<pto::UBVminSOp>>(
        typeConverter, patterns.getContext(), state);
    populateVPTOMemoryUbufPatterns(typeConverter, patterns, state);
    patterns.add<LowerUBVgatherbOpPattern>(
        typeConverter, patterns.getContext(), state);
    patterns.add<LowerUBVgatherOpPattern>(
        typeConverter, patterns.getContext(), state);
    populateVPTOMemoryMaskPatterns(typeConverter, patterns, state);
  }
}

} // namespace mlir::pto
