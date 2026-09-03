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

static std::string getL0LoadElementFragment(Type type) {
  std::string elem = getElementTypeFragment(type);
  if (!elem.empty()) {
    return elem;
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") ||
      StringRef(lower).contains("e5m2") ||
      StringRef(lower).contains("e8m0") ||
      StringRef(lower).contains("hif8") ||
      StringRef(lower).contains("e1m2x2") ||
      StringRef(lower).contains("e2m1x2")) {
    return "s8";
  }
  return {};
}

static std::string getNd2NzCopyElementFragment(Type elementType) {
  if (!elementType) {
    return {};
  }
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") ||
      StringRef(lower).contains("e8m0") || StringRef(lower).contains("hif8")) {
    return "U8";
  }
  if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2"))
  {
    return "U8";
  }

  if (elementType.isF16() || elementType.isBF16())
  {
    return "U16";
  }
  if (elementType.isF32())
  {
    return "U32";
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return "U8";
    case 16:
      return "U16";
    case 32:
      return "U32";
    default:
      return {};
    }
  }
  return {};
}


static FailureOr<Value>
packCopyGmToCbufConfig0(Operation *anchor, Value nBurst, Value lenBurst) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  if (!nBurstI64 || !lenBurstI64)
  {
    return failure();
  }

  Value config0 = getI64Constant(builder, loc, 0); // sid
  // burst_num[24:4], burst_len[45:25].
  return packShiftedI64Fields(builder, loc, config0,
                              {{nBurstI64, 4}, {lenBurstI64, 25}});
}

static FailureOr<Value>
packCopyGmToCbufConfig1(Operation *anchor, Value srcStride,
                               Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64)
  {
    return failure();
  }

  // config1 packs burst_src_stride[39:0] and burst_dst_stride[60:40].
  return packShiftedI64Fields(builder, loc, srcStrideI64,
                              {{dstStrideI64, 40}});
}

static FailureOr<Value>
packCopyGmToCbufMultiConfig0(Operation *anchor, Value sid,
                             Value loop1SrcStride, Value l2CacheCtl,
                             Value nValue) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value sidI64 = castIntegerLikeTo(anchor, sid, builder.getI64Type());
  Value loop1SrcStrideI64 =
      castIntegerLikeTo(anchor, loop1SrcStride, builder.getI64Type());
  Value l2CacheCtlI64 = castIntegerLikeTo(anchor, l2CacheCtl, builder.getI64Type());
  Value nValueI64 = castIntegerLikeTo(anchor, nValue, builder.getI64Type());
  if (!sidI64 || !loop1SrcStrideI64 || !l2CacheCtlI64 || !nValueI64)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, sidI64,
                              {{loop1SrcStrideI64, 4},
                               {l2CacheCtlI64, 44}, {nValueI64, 48}});
}

static FailureOr<Value>
packCopyGmToCbufMultiConfig1(Operation *anchor, Value dValue,
                             Value loop4SrcStride, Value smallC0En) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value dValueI64 = castIntegerLikeTo(anchor, dValue, builder.getI64Type());
  Value loop4SrcStrideI64 =
      castIntegerLikeTo(anchor, loop4SrcStride, builder.getI64Type());
  Value smallC0EnI64 = castIntegerLikeTo(anchor, smallC0En, builder.getI64Type());
  if (!dValueI64 || !loop4SrcStrideI64 || !smallC0EnI64)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, dValueI64,
                              {{loop4SrcStrideI64, 21},
                               {smallC0EnI64, 61}});
}

static FailureOr<Value> packCopyCbufToBtConfig(Operation *anchor,
                                               Value convControl,
                                               Value nBurst, Value lenBurst,
                                               Value sourceGap,
                                               Value dstGap) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value convControlI64 =
      castIntegerLikeTo(anchor, convControl, builder.getI64Type());
  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  Value sourceGapI64 = castIntegerLikeTo(anchor, sourceGap, builder.getI64Type());
  Value dstGapI64 = castIntegerLikeTo(anchor, dstGap, builder.getI64Type());
  if (!convControlI64 || !nBurstI64 || !lenBurstI64 || !sourceGapI64 ||
      !dstGapI64) {
    return failure();
  }

  Value config = builder.create<arith::ShLIOp>(
      loc, convControlI64, getI64Constant(builder, loc, 3));
  return packShiftedI64Fields(
      builder, loc, config,
      {{nBurstI64, 4}, {lenBurstI64, 16}, {sourceGapI64, 32},
       {dstGapI64, 48}});
}

static FailureOr<Value> packCopyCbufToFbufConfig(Operation *anchor, Value nBurst,
                                                 Value lenBurst,
                                                 Value sourceGap,
                                                 Value dstGap) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  Value sourceGapI64 = castIntegerLikeTo(anchor, sourceGap, builder.getI64Type());
  Value dstGapI64 = castIntegerLikeTo(anchor, dstGap, builder.getI64Type());
  if (!nBurstI64 || !lenBurstI64 || !sourceGapI64 || !dstGapI64)
  {
    return failure();
  }

  Value config = builder.create<arith::ShLIOp>(
      loc, nBurstI64, getI64Constant(builder, loc, 4));
  return packShiftedI64Fields(
      builder, loc, config,
      {{lenBurstI64, 16}, {sourceGapI64, 32}, {dstGapI64, 48}});
}

static FailureOr<Value>
packLoadCbufToL0Config0(Operation *anchor, Value mStart, Value kStart,
                        Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, mStartI64,
                              {{kStartI64, 16}, {mStepI64, 32},
                               {kStepI64, 40}});
}

static FailureOr<Value>
packLoadCbufToL0Config1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64)
  {
    return failure();
  }

  return packShiftedI64Fields(builder, loc, srcStrideI64,
                              {{dstStrideI64, 16}});
}

static FailureOr<StringRef> buildCopyGmToCbufCallee(MLIRContext *context,
                                                    Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  std::string elem = getCopyElementFragment(ptrType.getElementType());
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2." + elem +
                                      ".DV")
      .getValue();
}

static FailureOr<StringRef>
buildCopyGmToCbufMultiNd2NzCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  std::string elem = getNd2NzCopyElementFragment(ptrType.getElementType());
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ." +
                                      elem + ".V310")
      .getValue();
}

static std::string getDn2NzCopyElementFragment(Type type) {
  auto ptrType = dyn_cast<pto::PtrType>(type);
  if (!ptrType) {
    return {};
  }

  Type elementType = ptrType.getElementType();
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") ||
      StringRef(lower).contains("e8m0") || StringRef(lower).contains("hif8")) {
    return "u8";
  }

  if (elementType.isF16() || elementType.isBF16())
  {
    return "u16";
  }
  if (elementType.isF32())
  {
    return "u32";
  }

  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return "u8";
    case 16:
      return "u16";
    case 32:
      return "u32";
    default:
      return {};
    }
  }
  return {};
}

static FailureOr<StringRef>
buildCopyGmToCbufMultiDn2NzCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  std::string elem = getDn2NzCopyElementFragment(sourceType);
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context,
                         "llvm.hivm.MOV.OUT.TO.L1.MULTI.DN2NZ." + elem)
      .getValue();
}

static FailureOr<StringRef> buildLoadCbufToCaCallee(MLIRContext *context,
                                                     Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  std::string elem = getL0LoadElementFragment(ptrType.getElementType());
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.2Dv2." + elem)
      .getValue();
}

static FailureOr<StringRef> buildLoadCbufToCbCallee(MLIRContext *context,
                                                     Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  std::string elem = getL0LoadElementFragment(ptrType.getElementType());
  if (elem.empty())
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.2Dv2." + elem)
      .getValue();
}

static FailureOr<StringRef> buildLoadCbufToCaS4Callee(MLIRContext *context,
                                                       Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  Type elementType = ptrType.getElementType();
  if (!isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(elementType))
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.s4")
      .getValue();
}

static FailureOr<StringRef> buildLoadCbufToCbS4Callee(MLIRContext *context,
                                                       Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  Type elementType = ptrType.getElementType();
  if (!isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(elementType))
  {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.s4")
      .getValue();
}

static StringRef buildLoadCbufToCaMxCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.MX.2Dv2.v")
      .getValue();
}

[[maybe_unused]] static StringRef buildLoadCbufToCbMxCallee(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.MX.2Dv2.v")
      .getValue();
}

static StringRef buildCopyMatrixCcToGmCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.OUT.f32.EXT")
      .getValue();
}

static StringRef buildCopyMatrixCcToCbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.L1.f32.EXT")
      .getValue();
}

static FailureOr<StringRef> buildCopyMatrixCcToUbCallee(MLIRContext *context,
                                                         Type destinationType) {
  auto ptrType = dyn_cast<pto::PtrType>(destinationType);
  if (!ptrType)
  {
    return failure();
  }
  Type dstElem = ptrType.getElementType();
  if (dstElem.isF16()) {
    return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.UB.f322f16.EXT")
        .getValue();
  }
  if (dstElem.isF32()) {
    return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.UB.f32.EXT")
        .getValue();
  }
  return failure();
}

static FailureOr<StringRef> buildCopyCbufToBtCallee(pto::CopyCbufToBtOp op) {
  auto ptrType = dyn_cast<pto::PtrType>(op.getSource().getType());
  if (!ptrType)
  {
    return failure();
  }
  Type srcElem = ptrType.getElementType();
  if (srcElem.isF16()) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.f16")
        .getValue();
  }
  if (srcElem.isBF16()) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.bf16")
        .getValue();
  }
  if (srcElem.isF32()) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.f32")
        .getValue();
  }
  if (auto intType = dyn_cast<IntegerType>(srcElem);
      intType && intType.getWidth() == 32) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.s32")
        .getValue();
  }
  return failure();
}

static StringRef buildCopyCbufToFbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.L1.TO.FB.v220").getValue();
}

class LowerCreateCbufMatrixOpPattern final
    : public OpConversionPattern<pto::CreateCbufMatrixOp> {
public:
  explicit LowerCreateCbufMatrixOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<pto::CreateCbufMatrixOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CreateCbufMatrixOp op,
                  pto::CreateCbufMatrixOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destinationRaw = adaptor.getDst();
    Value rawValue = adaptor.getRawValue();
    Value repeatTimes = adaptor.getRepeatTimes();
    Value blockNum32b = adaptor.getBlockNum_32b();
    Value dstGap32b = adaptor.getDstGap_32b();
    if (!destinationRaw || !rawValue || !repeatTimes || !blockNum32b ||
        !dstGap32b) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer destination");
    }

    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    const bool validControlTypes =
        rawValue.getType() == i32Ty && repeatTimes.getType() == i64Ty &&
        blockNum32b.getType() == i64Ty && dstGap32b.getType() == i64Ty;
    if (!validControlTypes) {
      return rewriter.notifyMatchFailure(op, "expected i32 value and i64 controls");
    }

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> destination = reinterpretPointerToAddrSpace(
        op, destinationRaw, cbufAddressSpace);
    if (failed(destination)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to map destination to mat/l1");
    }

    Location loc = op.getLoc();
    const uint64_t fillWordWidth = static_cast<uint64_t>(op.getFillWordBits());
    StringRef calleeName;
    Value fillPattern;
    if (fillWordWidth == 16) {
      Value wordMask = getI32Constant(rewriter, loc, 0xFFFFU);
      Value lowWord = rewriter.create<arith::AndIOp>(loc, rawValue, wordMask);
      Value wordBits = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI16Type(), lowWord);
      fillPattern =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), wordBits);
      calleeName = "llvm.hivm.CREATE.CBUF.MATRIX.v3.u16.h";
    } else if (fillWordWidth == 32) {
      fillPattern = rewriter.create<arith::ExtUIOp>(loc, i64Ty, rawValue);
      calleeName = "llvm.hivm.CREATE.CBUF.MATRIX.v3.u32";
    } else {
      return rewriter.notifyMatchFailure(op, "expected a 16-bit or 32-bit fill word");
    }

    Value fieldMask = getI64Constant(rewriter, loc, 0x7FFFU);
    auto maskField = [&](Value value) -> Value {
      return rewriter.create<arith::AndIOp>(loc, value, fieldMask);
    };
    auto shiftField = [&](Value value, uint64_t amount) -> Value {
      return rewriter.create<arith::ShLIOp>(
          loc, value, getI64Constant(rewriter, loc, amount));
    };

    Value config = maskField(repeatTimes);
    config = rewriter.create<arith::OrIOp>(
        loc, config, shiftField(maskField(blockNum32b), 16));
    config = rewriter.create<arith::OrIOp>(
        loc, config, shiftField(maskField(dstGap32b), 32));

    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), i64Ty, fillPattern.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{*destination, config, fillPattern});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyGmToCbufOpPattern final
    : public OpConversionPattern<pto::CopyGmToCbufOp> {
public:
  explicit LowerCopyGmToCbufOpPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             LoweringState &state)
      : OpConversionPattern<pto::CopyGmToCbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(
      pto::CopyGmToCbufOp op,
      pto::CopyGmToCbufOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value nBurst = adaptor.getNBurst();
    Value lenBurst = adaptor.getLenBurst();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !nBurst || !lenBurst || !srcStride ||
        !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();
    if (nBurst.getType() != i64Ty || lenBurst.getType() != i64Ty ||
        srcStride.getType() != i64Ty || dstStride.getType() != i64Ty) {
      return rewriter.notifyMatchFailure(op, "expected i64 config operands");
    }

    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, gmAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/gm pointer spaces");
    }

    FailureOr<StringRef> calleeName =
        buildCopyGmToCbufCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported copy_gm_to_cbuf element type");
    }
    FailureOr<Value> config0 =
        packCopyGmToCbufConfig0(op, nBurst, lenBurst);
    FailureOr<Value> config1 =
        packCopyGmToCbufConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack copy_gm_to_cbuf config");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*destination, *source, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp>
class LowerCopyGmToCbufMultiOpPattern final
    : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyGmToCbufMultiOpPattern(TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
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

    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, gmAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/gm pointer spaces");
    }

    FailureOr<Value> config0 = packCopyGmToCbufMultiConfig0(
        op, adaptor.getSid(), adaptor.getLoop1SrcStride(),
        adaptor.getL2CacheCtrl(), adaptor.getNValue());
    FailureOr<Value> config1 =
        packCopyGmToCbufMultiConfig1(op, adaptor.getDValue(),
                                     adaptor.getLoop4SrcStride(),
                                     adaptor.getSmallc0En());
    if (failed(config0) || failed(config1))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack multi copy config");
    }

    FailureOr<StringRef> calleeName = [&] (MLIRContext *ctx, Type sourceType)
        -> FailureOr<StringRef> {
      if constexpr (std::is_same_v<CopyOp, pto::CopyGmToCbufMultiNd2NzOp>)
      {
        return buildCopyGmToCbufMultiNd2NzCallee(ctx, op.getSource().getType());
      }
      return buildCopyGmToCbufMultiDn2NzCallee(ctx, sourceType);
    }(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_gm_to_cbuf_multi element type");
    }

    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*destination, *source, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToBtOpPattern final
    : public OpConversionPattern<pto::CopyCbufToBtOp> {
public:
  explicit LowerCopyCbufToBtOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToBtOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToBtOp op,
                                pto::CopyCbufToBtOp::Adaptor adaptor,
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
    constexpr unsigned btAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::BIAS);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destinationPtr =
        reinterpretPointerToAddrSpace(op, destinationRaw, btAddressSpace);
    if (failed(source) || failed(destinationPtr))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/bt pointer spaces");
    }

    FailureOr<Value> config = packCopyCbufToBtConfig(
        op, adaptor.getConvControl(), adaptor.getNBurst(), adaptor.getLenBurst(),
        adaptor.getSourceGap(), adaptor.getDstGap());
    if (failed(config))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_bt config");
    }

    Type i64Ty = rewriter.getI64Type();
    Value destination =
        rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *destinationPtr);
    FailureOr<StringRef> calleeName = buildCopyCbufToBtCallee(op);
    if (failed(calleeName))
    {
      return rewriter.notifyMatchFailure(op, "unsupported copy_cbuf_to_bt source element type");
    }
    auto funcType = rewriter.getFunctionType(
        TypeRange{i64Ty, source->getType(), i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToFbufOpPattern final
    : public OpConversionPattern<pto::CopyCbufToFbufOp> {
public:
  explicit LowerCopyCbufToFbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToFbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToFbufOp op,
                                pto::CopyCbufToFbufOp::Adaptor adaptor,
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
    constexpr unsigned fbufAddressSpace = 7;
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, fbufAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/fbuf pointer spaces");
    }

    FailureOr<Value> config = packCopyCbufToFbufConfig(
        op, adaptor.getNBurst(), adaptor.getLenBurst(), adaptor.getSourceGap(),
        adaptor.getDstGap());
    if (failed(config))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_fbuf config");
    }

    Type i64Ty = rewriter.getI64Type();
    StringRef calleeName = buildCopyCbufToFbufCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCaOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCaOp> {
public:
  explicit LowerLoadCbufToCaOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCaOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCaOp op,
                                pto::LoadCbufToCaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value mStart = adaptor.getMStart();
    Value kStart = adaptor.getKStart();
    Value mStep = adaptor.getMStep();
    Value kStep = adaptor.getKStep();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !mStart || !kStart || !mStep ||
        !kStep || !srcStride || !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned caAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::LEFT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, caAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ca pointer spaces");
    }

    FailureOr<Value> config0 =
        packLoadCbufToL0Config0(op, mStart, kStart, mStep, kStep);
    FailureOr<Value> config1 =
        packLoadCbufToL0Config1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_ca config");
    }
    Value transpose =
        getI64Constant(rewriter, op.getLoc(), op.getTranspose() ? 1 : 0);

    FailureOr<StringRef> calleeName =
        buildLoadCbufToCaCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_ca element type");
    }
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty,
                  i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0,
                                             *config1, transpose});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerLoadCbufToS4OpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerLoadCbufToS4OpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
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
    constexpr unsigned targetAddressSpace =
        std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>
            ? static_cast<unsigned>(pto::AddressSpace::LEFT)
            : static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, targetAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cube pointer spaces");
    }

    FailureOr<Value> config0 = packLoadCbufToL0Config0(
        op, adaptor.getMStart(), adaptor.getKStart(), adaptor.getMStep(),
        adaptor.getKStep());
    FailureOr<Value> config1 =
        packLoadCbufToL0Config1(op, adaptor.getSrcStride(),
                                adaptor.getDstStride());
    if (failed(config0) || failed(config1))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_*_s4 config");
    }

    Value transpose =
        castIntegerLikeTo(op, adaptor.getTranspose(), rewriter.getI64Type());
    if (!transpose)
    {
      return rewriter.notifyMatchFailure(op, "failed to cast transpose to i64");
    }

    FailureOr<StringRef> calleeName =
        std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>
            ? buildLoadCbufToCaS4Callee(op.getContext(),
                                        op.getSource().getType())
            : buildLoadCbufToCbS4Callee(op.getContext(),
                                        op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported load_cbuf_to_*_s4 element type");
    }
    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty,
                  i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*destination, *source, *config0, *config1, transpose});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCbOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCbOp> {
public:
  explicit LowerLoadCbufToCbOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCbOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCbOp op,
                                pto::LoadCbufToCbOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value mStart = adaptor.getMStart();
    Value kStart = adaptor.getKStart();
    Value mStep = adaptor.getMStep();
    Value kStep = adaptor.getKStep();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !mStart || !kStart || !mStep ||
        !kStep || !srcStride || !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned cbAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cb pointer spaces");
    }

    bool transpose = op.getTranspose();
    FailureOr<Value> config0 =
        packLoadCbufToL0Config0(op, mStart, kStart, mStep, kStep);
    FailureOr<Value> config1 =
        packLoadCbufToL0Config1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1))
    {
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_cb config");
    }
    Value transposeValue =
        getI64Constant(rewriter, op.getLoc(), transpose ? 1 : 0);

    FailureOr<StringRef> calleeName =
        buildLoadCbufToCbCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_cb element type");
    }
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty,
                  i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0,
                                             *config1, transposeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCaMxOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCaMxOp> {
public:
  explicit LowerLoadCbufToCaMxOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCaMxOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::LoadCbufToCaMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcRaw = adaptor.getSource();
    Value dstRaw = adaptor.getDestination();
    if (!srcRaw || !dstRaw || !adaptor.getXStartPosition() ||
        !adaptor.getYStartPosition() || !adaptor.getXStep() ||
        !adaptor.getYStep() || !adaptor.getSrcStride() ||
        !adaptor.getDstStride()) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(dstRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned caAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::LEFT);
    FailureOr<Value> src = reinterpretPointerToAddrSpace(op, srcRaw, cbufAddressSpace);
    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, dstRaw, caAddressSpace);
    if (failed(src) || failed(dst))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ca pointer spaces");
    }

    Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
    unsigned elemBitWidth = pto::getPTOStorageElemBitWidth(sourceElemType);
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_ca_mx element type");
    }
    FailureOr<Value> config0 =
        packLoadCbufToL0Config0(op, adaptor.getXStartPosition(),
                                adaptor.getYStartPosition(), adaptor.getXStep(),
                                adaptor.getYStep());
    FailureOr<Value> config1 =
        packLoadCbufToL0Config1(op, adaptor.getSrcStride(),
                                adaptor.getDstStride());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack load_cbuf_to_ca_mx config");
    }
    auto i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *dst);

    StringRef calleeName = buildLoadCbufToCaMxCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{i64Ty, src->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{dstAddr, *src, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCbMxOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCbMxOp> {
public:
  explicit LowerLoadCbufToCbMxOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCbMxOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::LoadCbufToCbMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcRaw = adaptor.getSource();
    Value dstRaw = adaptor.getDestination();
    if (!srcRaw || !dstRaw || !adaptor.getXStartPosition() ||
        !adaptor.getYStartPosition() || !adaptor.getXStep() ||
        !adaptor.getYStep() || !adaptor.getSrcStride() ||
        !adaptor.getDstStride()) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(dstRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned cbAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> src = reinterpretPointerToAddrSpace(op, srcRaw, cbufAddressSpace);
    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, dstRaw, cbAddressSpace);
    if (failed(src) || failed(dst))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cb pointer spaces");
    }

    Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
    unsigned elemBitWidth = pto::getPTOStorageElemBitWidth(sourceElemType);
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_cb_mx element type");
    }
    FailureOr<Value> config0 =
        packLoadCbufToL0Config0(op, adaptor.getXStartPosition(),
                                adaptor.getYStartPosition(), adaptor.getXStep(),
                                adaptor.getYStep());
    FailureOr<Value> config1 =
        packLoadCbufToL0Config1(op, adaptor.getSrcStride(),
                                adaptor.getDstStride());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack load_cbuf_to_cb_mx config");
    }
    auto i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *dst);

    StringRef calleeName = buildLoadCbufToCbMxCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{i64Ty, src->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{dstAddr, *src, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyMatrixCcToGmOpPattern final
    : public OpConversionPattern<pto::CopyMatrixCcToGmOp> {
public:
  explicit LowerCopyMatrixCcToGmOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<pto::CopyMatrixCcToGmOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(
      pto::CopyMatrixCcToGmOp op, pto::CopyMatrixCcToGmOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value xm = adaptor.getXm();
    Value xt = adaptor.getXt();
    if (!sourceRaw || !destinationRaw || !xm || !xt)
    {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();
    if (xm.getType() != i64Ty || xt.getType() != i64Ty)
    {
      return rewriter.notifyMatchFailure(op, "expected i64 xm/xt operands");
    }

    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned ccAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::ACC);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, ccAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, gmAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cc/gm pointer spaces");
    }

    StringRef calleeName = buildCopyMatrixCcToGmCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, xm, xt});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp>
class LowerCopyMatrixCcToBufOpPattern final
    : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyMatrixCcToBufOpPattern(TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
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

    constexpr unsigned ccAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::ACC);
    constexpr unsigned targetAddressSpace =
        std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
            ? static_cast<unsigned>(pto::AddressSpace::MAT)
            : static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, ccAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, targetAddressSpace);
    if (failed(source) || failed(destination))
    {
      return rewriter.notifyMatchFailure(op, "failed to map cc->buf pointer spaces");
    }

    Type i64Ty = rewriter.getI64Type();
    Value config0 = castIntegerLikeTo(op, adaptor.getConfig0(), i64Ty);
    Value config1 = castIntegerLikeTo(op, adaptor.getConfig1(), i64Ty);
    if (!config0 || !config1)
    {
      return rewriter.notifyMatchFailure(op, "failed to cast config operands to i64");
    }

    FailureOr<StringRef> calleeName =
        std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
            ? FailureOr<StringRef>(buildCopyMatrixCcToCbufCallee(op.getContext()))
            : buildCopyMatrixCcToUbCallee(op.getContext(),
                                          op.getDestination().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_matrix_cc_to_{cbuf,ub} element type");
    }
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, config0,
                                             config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};


} // namespace

void populateVPTOCubeMemoryPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state) {
  patterns.add<LowerCopyGmToCbufOpPattern, LowerLoadCbufToCaOpPattern,
               LowerLoadCbufToCbOpPattern,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCaS4Op>,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCbS4Op>,
               LowerLoadCbufToCaMxOpPattern, LowerLoadCbufToCbMxOpPattern,
               LowerCopyMatrixCcToGmOpPattern,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToCbufOp>,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToUbOp>,
               LowerCopyCbufToBtOpPattern, LowerCopyCbufToFbufOpPattern,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiNd2NzOp>,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiDn2NzOp>>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
