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

static bool isPackedFloatTypeName(StringRef lower) {
  return lower.contains("e4m3") || lower.contains("e5m2") ||
         lower.contains("e8m0") || lower.contains("hif8") ||
         lower.contains("e1m2x2") || lower.contains("e2m1x2");
}

static std::string getLowerTypeText(Type type) {
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  return StringRef(typeText).lower();
}

static std::string getUnsignedIntWidthFragment(unsigned width) {
  switch (width) {
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

static std::string getL0LoadElementFragment(Type type) {
  std::string elem = getElementTypeFragment(type);
  if (!elem.empty()) {
    return elem;
  }

  std::string lower = getLowerTypeText(type);
  if (isPackedFloatTypeName(lower)) {
    return "s8";
  }
  return {};
}

static std::string getNd2NzCopyElementFragment(Type elementType) {
  if (!elementType) {
    return {};
  }
  std::string lower = getLowerTypeText(elementType);
  if (isPackedFloatTypeName(lower)) {
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
    return getUnsignedIntWidthFragment(intType.getWidth());
  }
  return {};
}


struct PackedI64Config {
  OpBuilder builder;
  Location loc;
  SmallVector<Value> fields;
};

static FailureOr<PackedI64Config>
beginPackI64Config(Operation *anchor, ValueRange values) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Type i64Type = builder.getI64Type();
  SmallVector<Value> converted;
  converted.reserve(values.size());
  for (Value value : values) {
    Value casted = castIntegerLikeTo(anchor, value, i64Type);
    if (!casted)
      return failure();
    converted.push_back(casted);
  }
  return PackedI64Config{builder, anchor->getLoc(), converted};
}

static FailureOr<Value>
packCopyGmToCbufConfig0(Operation *anchor, Value nBurst, Value lenBurst) {
  FailureOr<PackedI64Config> packed = beginPackI64Config(anchor, {nBurst, lenBurst});
  if (failed(packed))
    return failure();
  Value config0 = getI64Constant(packed->builder, packed->loc, 0); // sid
  // burst_num[24:4], burst_len[45:25].
  return packShiftedI64Fields(packed->builder, packed->loc, config0,
                              {{packed->fields[0], 4}, {packed->fields[1], 25}});
}

static FailureOr<Value>
packCopyGmToCbufConfig1(Operation *anchor, Value srcStride,
                               Value dstStride) {
  FailureOr<PackedI64Config> packed = beginPackI64Config(anchor, {srcStride, dstStride});
  if (failed(packed))
    return failure();
  // config1 packs burst_src_stride[39:0] and burst_dst_stride[60:40].
  return packShiftedI64Fields(packed->builder, packed->loc, packed->fields[0],
                              {{packed->fields[1], 40}});
}

static FailureOr<Value>
packCopyGmToCbufMultiConfig0(Operation *anchor, Value sid,
                             Value loop1SrcStride, Value l2CacheCtl,
                             Value nValue) {
  FailureOr<PackedI64Config> packed = beginPackI64Config(
      anchor, {sid, loop1SrcStride, l2CacheCtl, nValue});
  if (failed(packed))
    return failure();
  return packShiftedI64Fields(packed->builder, packed->loc, packed->fields[0],
                              {{packed->fields[1], 4}, {packed->fields[2], 44},
                               {packed->fields[3], 48}});
}

static FailureOr<Value>
packCopyGmToCbufMultiConfig1(Operation *anchor, Value dValue,
                             Value loop4SrcStride, Value smallC0En) {
  FailureOr<PackedI64Config> packed = beginPackI64Config(
      anchor, {dValue, loop4SrcStride, smallC0En});
  if (failed(packed))
    return failure();
  return packShiftedI64Fields(packed->builder, packed->loc, packed->fields[0],
                              {{packed->fields[1], 21}, {packed->fields[2], 61}});
}

static FailureOr<Value> packCopyCbufToBtConfig(Operation *anchor,
                                               Value convControl,
                                               Value nBurst, Value lenBurst,
                                               Value sourceGap,
                                               Value dstGap) {
  FailureOr<PackedI64Config> packed =
      beginPackI64Config(anchor, {convControl, nBurst, lenBurst, sourceGap,
                                dstGap});
  if (failed(packed))
    return failure();
  Value config = packed->builder.create<arith::ShLIOp>(
      packed->loc, packed->fields[0], getI64Constant(packed->builder, packed->loc, 3));
  return packShiftedI64Fields(
      packed->builder, packed->loc, config,
      {{packed->fields[1], 4}, {packed->fields[2], 16}, {packed->fields[3], 32},
       {packed->fields[4], 48}});
}

static FailureOr<Value> packCopyCbufToFbufConfig(Operation *anchor, Value nBurst,
                                                 Value lenBurst,
                                                 Value sourceGap,
                                                 Value dstGap) {
  FailureOr<PackedI64Config> packed =
      beginPackI64Config(anchor, {nBurst, lenBurst, sourceGap, dstGap});
  if (failed(packed))
    return failure();
  Value config = packed->builder.create<arith::ShLIOp>(
      packed->loc, packed->fields[0], getI64Constant(packed->builder, packed->loc, 4));
  return packShiftedI64Fields(
      packed->builder, packed->loc, config,
      {{packed->fields[1], 16}, {packed->fields[2], 32}, {packed->fields[3], 48}});
}

static FailureOr<Value>
packLoadCbufToL0Config0(Operation *anchor, Value mStart, Value kStart,
                        Value mStep, Value kStep) {
  FailureOr<PackedI64Config> packed =
      beginPackI64Config(anchor, {mStart, kStart, mStep, kStep});
  if (failed(packed))
    return failure();
  return packShiftedI64Fields(packed->builder, packed->loc, packed->fields[0],
                              {{packed->fields[1], 16}, {packed->fields[2], 32},
                               {packed->fields[3], 40}});
}

static FailureOr<Value>
packLoadCbufToL0Config1(Operation *anchor, Value srcStride, Value dstStride) {
  FailureOr<PackedI64Config> packed = beginPackI64Config(anchor, {srcStride, dstStride});
  if (failed(packed))
    return failure();
  return packShiftedI64Fields(packed->builder, packed->loc, packed->fields[0],
                              {{packed->fields[1], 16}});
}

// Pack the two common L0 load configuration words used by Ca/Cb variants.
// Keeping this sequence in one place avoids subtle divergence between
// lowering patterns while keeping each rewrite focused on operand mapping and
// call emission.
static LogicalResult packLoadCbufToL0Configs(Operation *anchor, Value mStart,
                                            Value kStart, Value mStep,
                                            Value kStep, Value srcStride,
                                            Value dstStride, Value &config0,
                                            Value &config1) {
  FailureOr<Value> packed0 =
      packLoadCbufToL0Config0(anchor, mStart, kStart, mStep, kStep);
  FailureOr<Value> packed1 = packLoadCbufToL0Config1(anchor, srcStride, dstStride);
  if (failed(packed0) || failed(packed1))
    return failure();
  config0 = *packed0;
  config1 = *packed1;
  return success();
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

static FailureOr<StringRef> buildFragmentCallee(
    MLIRContext *context, Type sourceType,
    const std::function<std::string(Type)> &fragmentFn, StringRef prefix,
    StringRef suffix = {}) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
  {
    return failure();
  }
  std::string fragment = fragmentFn(ptrType);
  if (fragment.empty())
  {
    return failure();
  }
  std::string name = (prefix + fragment + suffix).str();
  return StringAttr::get(context, name).getValue();
}

static FailureOr<StringRef>
buildCopyGmToCbufMultiNd2NzCallee(MLIRContext *context, Type sourceType) {
  return buildFragmentCallee(
      context, sourceType,
      [](Type t) {
        auto ptr = cast<pto::PtrType>(t);
        return getNd2NzCopyElementFragment(ptr.getElementType());
      },
      "llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ.", ".V310");
}

static FailureOr<StringRef>
buildCopyGmToCbufMultiDn2NzCallee(MLIRContext *context, Type sourceType) {
  return buildFragmentCallee(context, sourceType, getDn2NzCopyElementFragment,
                             "llvm.hivm.MOV.OUT.TO.L1.MULTI.DN2NZ.");
}

static FailureOr<StringRef> buildLoadL0Callee(MLIRContext *context,
                                              Type sourceType,
                                              StringRef prefix) {
  return buildFragmentCallee(
      context, sourceType,
      [](Type t) {
        auto ptr = cast<pto::PtrType>(t);
        return getL0LoadElementFragment(ptr.getElementType());
      },
      prefix);
}

static FailureOr<StringRef> buildLoadL0S4Callee(MLIRContext *context,
                                                Type sourceType,
                                                StringRef prefix) {
  return buildFragmentCallee(
      context, sourceType,
      [](Type t) {
        auto ptr = cast<pto::PtrType>(t);
        if (isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(ptr.getElementType()))
        {
          return std::string("s4");
        }
        return std::string();
      },
      prefix);
}

static FailureOr<StringRef> buildLoadCbufToCaCallee(MLIRContext *context,
                                                     Type sourceType) {
  return buildLoadL0Callee(context, sourceType,
                           "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.");
}

static FailureOr<StringRef> buildLoadCbufToCbCallee(MLIRContext *context,
                                                     Type sourceType) {
  return buildLoadL0Callee(context, sourceType,
                           "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.");
}

static FailureOr<StringRef> buildLoadCbufToCaS4Callee(MLIRContext *context,
                                                       Type sourceType) {
  return buildLoadL0S4Callee(context, sourceType,
                             "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.");
}

static FailureOr<StringRef> buildLoadCbufToCbS4Callee(MLIRContext *context,
                                                       Type sourceType) {
  return buildLoadL0S4Callee(context, sourceType,
                             "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.");
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

template <typename OpAdaptor>
static FailureOr<SmallVector<Value, 2>>
mapCubeSourceDestination(Operation *op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         ArrayRef<unsigned> addressSpaces,
                         StringRef mapMessage) {
  Value sourceRaw = adaptor.getSource();
  Value destinationRaw = adaptor.getDestination();
  if (!sourceRaw || !destinationRaw) {
    (void)rewriter.notifyMatchFailure(op, "expected converted operands");
    return failure();
  }
  if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
    (void)rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    return failure();
  }
  FailureOr<SmallVector<Value, 2>> pointers = reinterpretPointerOperands(
      op, {sourceRaw, destinationRaw}, addressSpaces);
  if (failed(pointers)) {
    (void)rewriter.notifyMatchFailure(op, mapMessage);
    return failure();
  }
  return pointers;
}

static void planCubeMemoryCall(Operation *op, StringRef calleeName,
                               ConversionPatternRewriter &rewriter,
                               LoweringState &state, TypeRange argTypes,
                               ValueRange args) {
  auto funcType = rewriter.getFunctionType(argTypes, TypeRange{});
  rewriter.create<func::CallOp>(op->getLoc(), calleeName, TypeRange{}, args);
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
  rewriter.eraseOp(op);
}

struct CubeL0Transfer {
  Value destination;
  Value source;
  Value config0;
  Value config1;
};

template <typename LoadOp>
static FailureOr<CubeL0Transfer>
prepareCubeL0Transfer(LoadOp op, typename LoadOp::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter) {
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
    (void)rewriter.notifyMatchFailure(op, "expected converted operands");
    return failure();
  }
  if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
    (void)rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    return failure();
  }
  constexpr unsigned cbufAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::MAT);
  constexpr unsigned cubeAddressSpace =
      std::is_same_v<LoadOp, pto::LoadCbufToCaOp>
          ? static_cast<unsigned>(pto::AddressSpace::LEFT)
          : static_cast<unsigned>(pto::AddressSpace::RIGHT);
  FailureOr<SmallVector<Value, 2>> pointers = reinterpretPointerOperands(
      op, {sourceRaw, destinationRaw}, {cbufAddressSpace, cubeAddressSpace});
  if (failed(pointers)) {
    (void)rewriter.notifyMatchFailure(
        op, std::is_same_v<LoadOp, pto::LoadCbufToCaOp>
                ? "failed to map cbuf/ca pointer spaces"
                : "failed to map cbuf/cb pointer spaces");
    return failure();
  }
  Value config0, config1;
  if (failed(packLoadCbufToL0Configs(op, mStart, kStart, mStep, kStep,
                                     srcStride, dstStride, config0, config1))) {
    (void)rewriter.notifyMatchFailure(
        op, std::is_same_v<LoadOp, pto::LoadCbufToCaOp>
                ? "failed to pack load_cbuf_to_ca config"
                : "failed to pack load_cbuf_to_cb config");
    return failure();
  }
  return CubeL0Transfer{(*pointers)[1], (*pointers)[0], config0, config1};
}

struct CubeL0Configs {
  Value config0;
  Value config1;
};

template <typename LoadOp>
static LogicalResult
validateCubeL0MxElemBitWidth(LoadOp op, ConversionPatternRewriter &rewriter) {
  Type sourceElemType =
      cast<pto::PtrType>(op.getSource().getType()).getElementType();
  unsigned elemBitWidth = pto::getPTOStorageElemBitWidth(sourceElemType);
  if (elemBitWidth != 0 && (elemBitWidth % 8) == 0) {
    return success();
  }
  return rewriter.notifyMatchFailure(
      op, std::is_same_v<LoadOp, pto::LoadCbufToCaMxOp>
              ? "unsupported load_cbuf_to_ca_mx element type"
              : "unsupported load_cbuf_to_cb_mx element type");
}

template <typename LoadOp>
static FailureOr<CubeL0Configs>
packCubeL0MxConfigs(LoadOp op, typename LoadOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) {
  FailureOr<Value> config0 =
      packLoadCbufToL0Config0(op, adaptor.getXStartPosition(),
                              adaptor.getYStartPosition(), adaptor.getXStep(),
                              adaptor.getYStep());
  FailureOr<Value> config1 =
      packLoadCbufToL0Config1(op, adaptor.getSrcStride(),
                              adaptor.getDstStride());
  if (failed(config0) || failed(config1)) {
    (void)rewriter.notifyMatchFailure(
        op, std::is_same_v<LoadOp, pto::LoadCbufToCaMxOp>
                ? "failed to pack load_cbuf_to_ca_mx config"
                : "failed to pack load_cbuf_to_cb_mx config");
    return failure();
  }
  return CubeL0Configs{*config0, *config1};
}

template <typename LoadOp>
static FailureOr<CubeL0Transfer>
prepareCubeL0MxTransfer(LoadOp op, typename LoadOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) {
  Value srcRaw = adaptor.getSource();
  Value dstRaw = adaptor.getDestination();
  if (!srcRaw || !dstRaw || !adaptor.getXStartPosition() ||
      !adaptor.getYStartPosition() || !adaptor.getXStep() ||
      !adaptor.getYStep() || !adaptor.getSrcStride() ||
      !adaptor.getDstStride()) {
    (void)rewriter.notifyMatchFailure(op, "expected converted operands");
    return failure();
  }
  if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(dstRaw.getType())) {
    (void)rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    return failure();
  }
  constexpr unsigned cbufAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::MAT);
  constexpr unsigned cubeAddressSpace =
      std::is_same_v<LoadOp, pto::LoadCbufToCaMxOp>
          ? static_cast<unsigned>(pto::AddressSpace::LEFT)
          : static_cast<unsigned>(pto::AddressSpace::RIGHT);
  FailureOr<SmallVector<Value, 2>> pointers = reinterpretPointerOperands(
      op, {srcRaw, dstRaw}, {cbufAddressSpace, cubeAddressSpace});
  if (failed(pointers)) {
    (void)rewriter.notifyMatchFailure(
        op, std::is_same_v<LoadOp, pto::LoadCbufToCaMxOp>
                ? "failed to map cbuf/ca pointer spaces"
                : "failed to map cbuf/cb pointer spaces");
    return failure();
  }
  if (failed(validateCubeL0MxElemBitWidth(op, rewriter))) {
    return failure();
  }
  FailureOr<CubeL0Configs> configs =
      packCubeL0MxConfigs(op, adaptor, rewriter);
  if (failed(configs)) {
    return failure();
  }
  return CubeL0Transfer{(*pointers)[1], (*pointers)[0], configs->config0,
                        configs->config1};
}

struct CbufMatrixFill {
  StringRef calleeName;
  Value fillPattern;
};

static FailureOr<CbufMatrixFill>
buildCbufMatrixFill(ConversionPatternRewriter &rewriter, Location loc,
                    uint64_t fillWordWidth, Value rawValue, Type i64Ty) {
  if (fillWordWidth == 16) {
    Value wordMask = getI32Constant(rewriter, loc, 0xFFFFU);
    Value lowWord = rewriter.create<arith::AndIOp>(loc, rawValue, wordMask);
    Value wordBits =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), lowWord);
    Value fillPattern =
        rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), wordBits);
    return CbufMatrixFill{"llvm.hivm.CREATE.CBUF.MATRIX.v3.u16.h",
                          fillPattern};
  }
  if (fillWordWidth == 32) {
    Value fillPattern = rewriter.create<arith::ExtUIOp>(loc, i64Ty, rawValue);
    return CbufMatrixFill{"llvm.hivm.CREATE.CBUF.MATRIX.v3.u32", fillPattern};
  }
  return failure();
}

static Value buildCbufMatrixConfig(ConversionPatternRewriter &rewriter,
                                   Location loc, Value repeatTimes,
                                   Value blockNum32b, Value dstGap32b) {
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
  return rewriter.create<arith::OrIOp>(
      loc, config, shiftField(maskField(dstGap32b), 32));
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
    if (rawValue.getType() != i32Ty || repeatTimes.getType() != i64Ty ||
        blockNum32b.getType() != i64Ty || dstGap32b.getType() != i64Ty) {
      return rewriter.notifyMatchFailure(op, "expected i32 value and i64 controls");
    }
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(destination)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to map destination to mat/l1");
    }
    Location loc = op.getLoc();
    FailureOr<CbufMatrixFill> fill = buildCbufMatrixFill(
        rewriter, loc, static_cast<uint64_t>(op.getFillWordBits()), rawValue,
        i64Ty);
    if (failed(fill)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a 16-bit or 32-bit fill word");
    }
    Value config = buildCbufMatrixConfig(rewriter, loc, repeatTimes,
                                         blockNum32b, dstGap32b);
    planCubeMemoryCall(op, fill->calleeName, rewriter, state,
                       TypeRange{destination->getType(), i64Ty,
                                 fill->fillPattern.getType()},
                       ValueRange{*destination, config, fill->fillPattern});
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
    Value nBurst = adaptor.getNBurst();
    Value lenBurst = adaptor.getLenBurst();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!nBurst || !lenBurst || !srcStride || !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
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
    FailureOr<SmallVector<Value, 2>> pointers =
        mapCubeSourceDestination(op, adaptor, rewriter,
                                 {gmAddressSpace, cbufAddressSpace},
                                 "failed to map cbuf/gm pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    FailureOr<Value> config0 = packCopyGmToCbufConfig0(op, nBurst, lenBurst);
    FailureOr<Value> config1 =
        packCopyGmToCbufConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack copy_gm_to_cbuf config");
    }
    FailureOr<StringRef> calleeName =
        buildCopyGmToCbufCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_gm_to_cbuf element type");
    }
    Value destinationPtr = (*pointers)[1];
    Value sourcePtr = (*pointers)[0];
    planCubeMemoryCall(op, *calleeName, rewriter, state,
                       TypeRange{destinationPtr.getType(), sourcePtr.getType(),
                                 i64Ty, i64Ty},
                       ValueRange{destinationPtr, sourcePtr, *config0,
                                  *config1});
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
    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<SmallVector<Value, 2>> pointers = mapCubeSourceDestination(
        op, adaptor, rewriter, {gmAddressSpace, cbufAddressSpace},
        "failed to map cbuf/gm pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    FailureOr<Value> config0 = packCopyGmToCbufMultiConfig0(
        op, adaptor.getSid(), adaptor.getLoop1SrcStride(),
        adaptor.getL2CacheCtrl(), adaptor.getNValue());
    FailureOr<Value> config1 = packCopyGmToCbufMultiConfig1(
        op, adaptor.getDValue(), adaptor.getLoop4SrcStride(),
        adaptor.getSmallc0En());
    if (failed(config0) || failed(config1)) {
      return rewriter.notifyMatchFailure(op, "failed to pack multi copy config");
    }
    FailureOr<StringRef> calleeName;
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToCbufMultiNd2NzOp>) {
      calleeName = buildCopyGmToCbufMultiNd2NzCallee(
          op.getContext(), op.getSource().getType());
    } else {
      calleeName = buildCopyGmToCbufMultiDn2NzCallee(
          op.getContext(), op.getSource().getType());
    }
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_gm_to_cbuf_multi element type");
    }
    Type i64Ty = rewriter.getI64Type();
    planCubeMemoryCall(
        op, *calleeName, rewriter, state,
        TypeRange{(*pointers)[1].getType(), (*pointers)[0].getType(), i64Ty,
                  i64Ty},
        ValueRange{(*pointers)[1], (*pointers)[0], *config0, *config1});
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
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned btAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::BIAS);
    FailureOr<SmallVector<Value, 2>> pointers = mapCubeSourceDestination(
        op, adaptor, rewriter, {cbufAddressSpace, btAddressSpace},
        "failed to map cbuf/bt pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    FailureOr<Value> config = packCopyCbufToBtConfig(
        op, adaptor.getConvControl(), adaptor.getNBurst(), adaptor.getLenBurst(),
        adaptor.getSourceGap(), adaptor.getDstGap());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_bt config");
    }
    Type i64Ty = rewriter.getI64Type();
    Value destination =
        rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, (*pointers)[1]);
    FailureOr<StringRef> calleeName = buildCopyCbufToBtCallee(op);
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_cbuf_to_bt source element type");
    }
    planCubeMemoryCall(op, *calleeName, rewriter, state,
                       TypeRange{i64Ty, (*pointers)[0].getType(), i64Ty},
                       ValueRange{destination, (*pointers)[0], *config});
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
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned fbufAddressSpace = 7;
    FailureOr<SmallVector<Value, 2>> pointers = mapCubeSourceDestination(
        op, adaptor, rewriter, {cbufAddressSpace, fbufAddressSpace},
        "failed to map cbuf/fbuf pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    FailureOr<Value> config = packCopyCbufToFbufConfig(
        op, adaptor.getNBurst(), adaptor.getLenBurst(), adaptor.getSourceGap(),
        adaptor.getDstGap());
    if (failed(config)) {
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_fbuf config");
    }
    Type i64Ty = rewriter.getI64Type();
    StringRef calleeName = buildCopyCbufToFbufCallee(op.getContext());
    planCubeMemoryCall(
        op, calleeName, rewriter, state,
        TypeRange{(*pointers)[1].getType(), (*pointers)[0].getType(), i64Ty},
        ValueRange{(*pointers)[1], (*pointers)[0], *config});
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerLoadCbufToL0OpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerLoadCbufToL0OpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<CubeL0Transfer> transfer =
        prepareCubeL0Transfer(op, adaptor, rewriter);
    if (failed(transfer)) {
      return failure();
    }
    Type i64Ty = rewriter.getI64Type();
    Value transposeValue = getI64Constant(rewriter, op.getLoc(),
                                          op.getTranspose() ? 1 : 0);
    FailureOr<StringRef> calleeName;
    if constexpr (std::is_same_v<LoadOp, pto::LoadCbufToCaOp>) {
      calleeName =
          buildLoadCbufToCaCallee(op.getContext(), op.getSource().getType());
    } else {
      calleeName =
          buildLoadCbufToCbCallee(op.getContext(), op.getSource().getType());
    }
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(
          op, std::is_same_v<LoadOp, pto::LoadCbufToCaOp>
                  ? "unsupported load_cbuf_to_ca element type"
                  : "unsupported load_cbuf_to_cb element type");
    }
    planCubeMemoryCall(
        op, *calleeName, rewriter, state,
        TypeRange{transfer->destination.getType(), transfer->source.getType(),
                  i64Ty, i64Ty, i64Ty},
        ValueRange{transfer->destination, transfer->source, transfer->config0,
                   transfer->config1, transposeValue});
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
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned targetAddressSpace =
        std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>
            ? static_cast<unsigned>(pto::AddressSpace::LEFT)
            : static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<SmallVector<Value, 2>> pointers = mapCubeSourceDestination(
        op, adaptor, rewriter, {cbufAddressSpace, targetAddressSpace},
        "failed to map cbuf/cube pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    Value config0, config1;
    if (failed(packLoadCbufToL0Configs(
            op, adaptor.getMStart(), adaptor.getKStart(), adaptor.getMStep(),
            adaptor.getKStep(), adaptor.getSrcStride(), adaptor.getDstStride(),
            config0, config1))) {
      return rewriter.notifyMatchFailure(
          op, "failed to pack load_cbuf_to_*_s4 config");
    }
    Type i64Ty = rewriter.getI64Type();
    Value transpose = castIntegerLikeTo(op, adaptor.getTranspose(), i64Ty);
    if (!transpose) {
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
    planCubeMemoryCall(
        op, *calleeName, rewriter, state,
        TypeRange{(*pointers)[1].getType(), (*pointers)[0].getType(), i64Ty,
                  i64Ty, i64Ty},
        ValueRange{(*pointers)[1], (*pointers)[0], config0, config1, transpose});
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerLoadCbufToMxOpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerLoadCbufToMxOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<CubeL0Transfer> transfer =
        prepareCubeL0MxTransfer(op, adaptor, rewriter);
    if (failed(transfer)) {
      return failure();
    }
    Type i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(
        op.getLoc(), i64Ty, transfer->destination);
    StringRef calleeName;
    if constexpr (std::is_same_v<LoadOp, pto::LoadCbufToCaMxOp>) {
      calleeName = buildLoadCbufToCaMxCallee(op.getContext());
    } else {
      calleeName = buildLoadCbufToCbMxCallee(op.getContext());
    }
    planCubeMemoryCall(
        op, calleeName, rewriter, state,
        TypeRange{i64Ty, transfer->source.getType(), i64Ty, i64Ty},
        ValueRange{dstAddr, transfer->source, transfer->config0,
                   transfer->config1});
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
    Value xm = adaptor.getXm();
    Value xt = adaptor.getXt();
    Type i64Ty = rewriter.getI64Type();
    if (!xm || !xt) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    if (xm.getType() != i64Ty || xt.getType() != i64Ty) {
      return rewriter.notifyMatchFailure(op, "expected i64 xm/xt operands");
    }
    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned ccAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::ACC);
    FailureOr<SmallVector<Value, 2>> pointers = mapCubeSourceDestination(
        op, adaptor, rewriter, {ccAddressSpace, gmAddressSpace},
        "failed to map cc/gm pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    StringRef calleeName = buildCopyMatrixCcToGmCallee(op.getContext());
    planCubeMemoryCall(
        op, calleeName, rewriter, state,
        TypeRange{(*pointers)[1].getType(), (*pointers)[0].getType(), i64Ty,
                  i64Ty},
        ValueRange{(*pointers)[1], (*pointers)[0], xm, xt});
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
    constexpr unsigned ccAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::ACC);
    constexpr unsigned targetAddressSpace =
        std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
            ? static_cast<unsigned>(pto::AddressSpace::MAT)
            : static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<SmallVector<Value, 2>> pointers = mapCubeSourceDestination(
        op, adaptor, rewriter, {ccAddressSpace, targetAddressSpace},
        "failed to map cc->buf pointer spaces");
    if (failed(pointers)) {
      return failure();
    }
    Type i64Ty = rewriter.getI64Type();
    Value config0 = castIntegerLikeTo(op, adaptor.getConfig0(), i64Ty);
    Value config1 = castIntegerLikeTo(op, adaptor.getConfig1(), i64Ty);
    if (!config0 || !config1) {
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
    planCubeMemoryCall(
        op, *calleeName, rewriter, state,
        TypeRange{(*pointers)[1].getType(), (*pointers)[0].getType(), i64Ty,
                  i64Ty},
        ValueRange{(*pointers)[1], (*pointers)[0], config0, config1});
    return success();
  }

private:
  LoweringState &state;
};


} // namespace

void populateVPTOCubeMemoryPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state) {
  patterns.add<LowerCopyGmToCbufOpPattern,
               LowerLoadCbufToL0OpPattern<pto::LoadCbufToCaOp>,
               LowerLoadCbufToL0OpPattern<pto::LoadCbufToCbOp>,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCaS4Op>,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCbS4Op>,
               LowerLoadCbufToMxOpPattern<pto::LoadCbufToCaMxOp>,
               LowerLoadCbufToMxOpPattern<pto::LoadCbufToCbMxOp>,
               LowerCopyMatrixCcToGmOpPattern,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToCbufOp>,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToUbOp>,
               LowerCopyCbufToBtOpPattern, LowerCopyCbufToFbufOpPattern,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiNd2NzOp>,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiDn2NzOp>,
               LowerCreateCbufMatrixOpPattern>(
      typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
