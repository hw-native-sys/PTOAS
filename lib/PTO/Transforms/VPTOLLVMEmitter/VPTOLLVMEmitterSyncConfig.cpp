// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// The CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "VPTOLLVMEmitterInternal.h"
#include "PTO/Transforms/VPTOLLVMEmitterHelper.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::pto {

namespace {

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

static std::optional<uint64_t> parsePipeImmediate(StringRef pipe) {
  if (pipe == "PIPE_S")
  {
    return 0;
  }
  if (pipe == "PIPE_V")
  {
    return 1;
  }
  if (pipe == "PIPE_M")
  {
    return 2;
  }
  if (pipe == "PIPE_MTE1")
  {
    return 3;
  }
  if (pipe == "PIPE_MTE2")
  {
    return 4;
  }
  if (pipe == "PIPE_MTE3")
  {
    return 5;
  }
  if (pipe == "PIPE_ALL")
  {
    return 6;
  }
  if (pipe == "PIPE_MTE4")
  {
    return 7;
  }
  if (pipe == "PIPE_MTE5")
  {
    return 8;
  }
  if (pipe == "PIPE_V2")
  {
    return 9;
  }
  if (pipe == "PIPE_FIX")
  {
    return 10;
  }
  if (pipe == "VIRTUAL_PIPE_MTE2_L1A")
  {
    return 11;
  }
  if (pipe == "VIRTUAL_PIPE_MTE2_L1B")
  {
    return 12;
  }
  return std::nullopt;
}

static std::optional<uint64_t> parseEventImmediate(StringRef event) {
  if (!event.consume_front("EVENT_ID"))
  {
    return std::nullopt;
  }
  uint64_t value = 0;
  if (event.getAsInteger(10, value))
  {
    return std::nullopt;
  }
  return value;
}

template <typename ConfigOp>
static StringRef buildUnaryConfigCallee(MLIRContext *context);

template <>
StringRef buildUnaryConfigCallee<pto::SetCtrlOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CTRL").getValue();
}

static StringRef buildStoreVfSimtInfoCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.store.vfsimt.info").getValue();
}

static StringRef buildSyncthreadsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.sync.workitems").getValue();
}

static StringRef buildThreadfenceCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.fence.workitems").getValue();
}

static StringRef buildThreadfenceBlockCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.fenceblock.workitems").getValue();
}


template <typename LoopOp>
static StringRef buildSetLoopCallee(MLIRContext *context);

template <typename ConfigOp>
static StringRef buildUnaryConfigCallee(MLIRContext *context);

template <typename ConfigOp>
static StringRef buildNullaryConfigCallee(MLIRContext *context);

template <>
StringRef buildSetLoopCallee<pto::SetLoop2StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop1StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoopSizeOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop2StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop1StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoopSizeUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop3ParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP3.PARA").getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetChannelParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CHANNEL.PARA").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetMovPadValOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MOV.PAD.VAL").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetQuantPreOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.QUANT.PRE.v300").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetReluAlphaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.RELU.ALPHA").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetFixClipReluOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FIX.CLIP.RELU").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetLoop2StrideOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetLoop1StrideOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetLoopSizeOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetMte2NzParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MTE2.NZ.PARA").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetPadValOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.PAD.VAL.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetFpcOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FPC").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetStoreAtomicCfgOp>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ST.ATOMIC.CFG").getValue();
}

template <>
StringRef buildNullaryConfigCallee<pto::SetAtomicS32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S32").getValue();
}

template <>
StringRef buildNullaryConfigCallee<pto::SetAtomicS8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S8").getValue();
}

static FailureOr<Value> encodeMovPadValue(Location loc, Value value,
                                          ConversionPatternRewriter &rewriter) {
  Type type = value.getType();
  Value payload = value;
  unsigned bitWidth = 0;

  if (auto intType = dyn_cast<IntegerType>(type)) {
    bitWidth = intType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    bitWidth = floatType.getWidth();
    auto intType = rewriter.getIntegerType(bitWidth);
    payload = rewriter.create<arith::BitcastOp>(loc, intType, value);
  } else {
    return failure();
  }

  if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32)
  {
    return failure();
  }

  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), payload)
      .getResult();
}

template <typename SyncOp>
static StringRef buildSyncCallee(MLIRContext *context);

template <>
StringRef buildSyncCallee<pto::SetFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.IMM").getValue();
}

template <>
StringRef buildSyncCallee<pto::WaitFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.IMM").getValue();
}

template <>
StringRef buildSyncCallee<pto::SetFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.REG").getValue();
}

template <>
StringRef buildSyncCallee<pto::WaitFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.REG").getValue();
}

template <>
StringRef buildSyncCallee<pto::BarrierOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.BARRIER").getValue();
}

template <>
StringRef buildSyncCallee<pto::SyncSetOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CROSS.CORE").getValue();
}

template <>
StringRef buildSyncCallee<pto::SyncWaitOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.DEV.REG").getValue();
}

template <>
StringRef buildSyncCallee<pto::SetIntraBlockOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.INTRA.BLOCK.mode").getValue();
}

template <>
StringRef buildSyncCallee<pto::WaitIntraBlockOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.INTRA.BLOCK.mode").getValue();
}

static StringRef buildMemBarCallee(MemBarKind kind, MLIRContext *context) {
  switch (kind) {
  case MemBarKind::VV_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vv.all").getValue();
  case MemBarKind::VST_VLD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.vld").getValue();
  case MemBarKind::VLD_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vld.vst").getValue();
  case MemBarKind::VST_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.vst").getValue();
  case MemBarKind::VS_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vs.all").getValue();
  case MemBarKind::VST_LD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.ld").getValue();
  case MemBarKind::VLD_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vld.st").getValue();
  case MemBarKind::VST_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.st").getValue();
  case MemBarKind::SV_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.sv.all").getValue();
  case MemBarKind::ST_VLD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.vld").getValue();
  case MemBarKind::LD_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ld.vst").getValue();
  case MemBarKind::ST_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.vst").getValue();
  case MemBarKind::SS_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ss.all").getValue();
  case MemBarKind::ST_LD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.ld").getValue();
  case MemBarKind::LD_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ld.st").getValue();
  case MemBarKind::ST_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.st").getValue();
  }
  llvm_unreachable("unexpected membar kind");
}

static uint64_t getDsbMemImmediate(DsbMem kind) {
  return static_cast<uint64_t>(kind);
}

static uint64_t getDcciCacheLineImmediate(DcciCacheLine kind) {
  return static_cast<uint64_t>(kind);
}

static uint64_t getDcciDstImmediate(DcciDst kind) {
  return static_cast<uint64_t>(kind);
}

static StringRef buildDcciCallee(unsigned addressSpace, bool hasDst,
                                 MLIRContext *context) {
  if (addressSpace == static_cast<unsigned>(pto::AddressSpace::GM)) {
    return StringAttr::get(context, hasDst ? "llvm.hivm.DCCI.DST"
                                           : "llvm.hivm.DCCI")
        .getValue();
  }
  if (addressSpace == static_cast<unsigned>(pto::AddressSpace::VEC)) {
    return StringAttr::get(context, hasDst ? "llvm.hivm.DCCI.DST.UB"
                                           : "llvm.hivm.DCCI.UB")
        .getValue();
  }
  llvm_unreachable("unexpected dcci address space");
}

template <>
StringRef buildSyncCallee<pto::GetBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BUFI.mode").getValue();
}

template <>
StringRef buildSyncCallee<pto::RlsBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.RLS.BUFI.mode").getValue();
}

static StringRef buildBufDynSyncCallee(MLIRContext *context, bool isGetBuf) {
  return StringAttr::get(context,
                         isGetBuf ? "llvm.hivm.GET.BUF.mode"
                                  : "llvm.hivm.RLS.BUF.mode")
      .getValue();
}

template <typename LoopOp>
class LowerSetLoopConfigOpPattern final : public OpConversionPattern<LoopOp> {
public:
  explicit LowerSetLoopConfigOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<LoopOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoopOp op, typename LoopOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> packed = failure();
    if constexpr (std::is_same_v<LoopOp, pto::SetLoopSizeOutToUbOp> ||
                  std::is_same_v<LoopOp, pto::SetLoopSizeUbToOutOp>) {
      packed = packLoopSize(op, adaptor.getFirst(), adaptor.getSecond());
    } else {
      packed = packLoopPair(op, adaptor.getFirst(), adaptor.getSecond());
    }
    if (failed(packed)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack loop configuration");
    }

    StringRef calleeName = buildSetLoopCallee<LoopOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*packed});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp>
class LowerUnaryConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerUnaryConfigOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> encoded =
        encodeMovPadValue(op.getLoc(), adaptor.getValue(), rewriter);
    if (failed(encoded)) {
      return rewriter.notifyMatchFailure(
          op, "expected 8/16/32-bit integer or float mov-pad payload");
    }

    StringRef calleeName = buildUnaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*encoded});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp>
class LowerUnaryI64ConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerUnaryI64ConfigOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = buildUnaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getValue().getType()},
                                 TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerStoreVfSimtInfoOpPattern final
    : public OpConversionPattern<pto::StoreVfSimtInfoOp> {
public:
  explicit LowerStoreVfSimtInfoOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<pto::StoreVfSimtInfoOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::StoreVfSimtInfoOp op,
                  pto::StoreVfSimtInfoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value dimZ = adaptor.getDimZ();
    Value dimY = adaptor.getDimY();
    Value dimX = adaptor.getDimX();
    if (!dimZ || !dimY || !dimX)
    {
      return rewriter.notifyMatchFailure(op, "missing converted SIMT dims");
    }

    auto i64Type = rewriter.getI64Type();
    auto castToI64 = [&](Value value) -> Value {
      if (value.getType().isInteger(64))
      {
        return value;
      }
      return rewriter.create<arith::ExtUIOp>(loc, i64Type, value).getResult();
    };

    Value dimZI64 = castToI64(dimZ);
    Value dimYI64 = castToI64(dimY);
    Value dimXI64 = castToI64(dimX);
    Value dimYShift = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(16));
    Value dimZShift = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(32));
    Value packedDimY =
        rewriter.create<arith::ShLIOp>(loc, dimYI64, dimYShift).getResult();
    Value packedDimZ =
        rewriter.create<arith::ShLIOp>(loc, dimZI64, dimZShift).getResult();
    Value payload =
        rewriter.create<arith::OrIOp>(loc, dimXI64, packedDimY).getResult();
    payload =
        rewriter.create<arith::OrIOp>(loc, payload, packedDimZ).getResult();

    StringRef calleeName = buildStoreVfSimtInfoCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{i64Type}, TypeRange{});
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{payload});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename FenceOp>
static StringRef buildSimtFenceCallee(MLIRContext *context);

template <>
StringRef buildSimtFenceCallee<pto::SyncthreadsOp>(MLIRContext *context) {
  return buildSyncthreadsCallee(context);
}

template <>
StringRef buildSimtFenceCallee<pto::ThreadfenceOp>(MLIRContext *context) {
  return buildThreadfenceCallee(context);
}

template <>
StringRef buildSimtFenceCallee<pto::ThreadfenceBlockOp>(MLIRContext *context) {
  return buildThreadfenceBlockCallee(context);
}

template <typename FenceOp>
class LowerSimtFenceOpPattern final : public OpConversionPattern<FenceOp> {
public:
  explicit LowerSimtFenceOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   LoweringState &state)
      : OpConversionPattern<FenceOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(FenceOp op, typename FenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    FunctionType funcType = rewriter.getFunctionType({}, {});
    StringRef calleeName = buildSimtFenceCallee<FenceOp>(op.getContext());
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

struct SimtKeepResumePhysicalRegister {
  int64_t baseRegister;
  unsigned registerCount;
};

// TPERn names one 32-bit register, while TPERLn names the 64-bit pair whose
// base register is R(2n). Keep uses tied inputs so the compiler models the
// value captured by each fixed output without inline assembly instructions.
static std::string buildSimtKeepResumeConstraints(
    ArrayRef<SimtKeepResumePhysicalRegister> physicalRegs, bool tieInputs) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto [index, physicalReg] : llvm::enumerate(physicalRegs)) {
    if (index != 0)
    {
      os << ",";
    }
    if (physicalReg.registerCount == 2) {
      os << "={TPERL" << physicalReg.baseRegister / 2 << "}";
    } else {
      os << "={TPER" << physicalReg.baseRegister << "}";
}
  }
  if (tieInputs) {
    for (size_t index = 0; index < physicalRegs.size(); ++index)
    {
      os << "," << index;
    }
  }
  return os.str();
}

template <typename OpT>
static SmallVector<OpT, 4> collectConsecutiveOps(OpT first) {
  SmallVector<OpT, 4> ops;
  for (Operation *cur = first.getOperation(); cur; cur = cur->getNextNode()) {
    auto typed = dyn_cast<OpT>(cur);
    if (!typed)
    {
      break;
    }
    ops.push_back(typed);
  }
  return ops;
}

static bool hasPreviousSameOp(Operation *op) {
  Operation *prev = op->getPrevNode();
  return prev && prev->getName() == op->getName();
}

static std::optional<unsigned> getSimtKeepResumeBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= 64)
    {
      return intType.getWidth();
    }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16())
  {
    return 16;
  }
  if (type.isF32())
  {
    return 32;
  }
  return std::nullopt;
}

static Value packSimtKeepResumePayload(Location loc, Value value,
                                       ConversionPatternRewriter &rewriter) {
  Type type = value.getType();
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(type);
  if (!width) {
    return {};
  }

  Type intType = rewriter.getIntegerType(*width);
  Value bits = value;
  if (!isa<IntegerType>(type))
  {
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, value);
  } else if (bits.getType() != intType) {
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, bits);
  }
  if (*width < 32)
  {
    return rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI32Type(), bits);
  }
  if (*width == 32 && bits.getType() != rewriter.getI32Type())
  {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), bits);
  }
  return bits;
}

static Value unpackSimtKeepResumePayload(Location loc, Value value,
                                         Type resultType,
                                         ConversionPatternRewriter &rewriter) {
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(resultType);
  if (!width) {
    return {};
  }

  Type intType = rewriter.getIntegerType(*width);
  Value bits = value;
  if (*width < 32)
  {
    bits = rewriter.create<LLVM::TruncOp>(loc, intType, bits);
  } else if (bits.getType() != intType) {
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, bits);
  }

  if (isa<IntegerType>(resultType)) {
    if (bits.getType() == resultType)
    {
      return bits;
    }
    return rewriter.create<LLVM::BitcastOp>(loc, resultType, bits);
  }
  return rewriter.create<LLVM::BitcastOp>(loc, resultType, bits);
}

static unsigned getSimtKeepResumeRegisterCount(Type type) {
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(type);
  return width && *width > 32 ? 2 : 1;
}

static FailureOr<SmallVector<SimtKeepResumePhysicalRegister, 4>>
computeSimtKeepResumePhysicalRegs(
    ArrayRef<std::pair<int64_t, unsigned>> logicalSlots) {
  SmallVector<SimtKeepResumePhysicalRegister, 4> physicalRegs;
  physicalRegs.reserve(logicalSlots.size());
  for (auto [slot, registerCount] : logicalSlots) {
    if (slot < 0 || slot >= 123)
    {
      return failure();
    }
    if (registerCount == 2 && ((slot % 2) != 0 || slot + 1 >= 123))
    {
      return failure();
    }
    // Slots are user-assigned storage words, not dense ordinals in the current
    // keep/resume group. This keeps a consumer that resumes only a subset of
    // slots from changing where the remaining slots are read from.
    int64_t baseRegister = 4 + slot;
    if (baseRegister + static_cast<int64_t>(registerCount) - 1 > 126)
    {
      return failure();
    }
    physicalRegs.push_back({baseRegister, registerCount});
  }
  return physicalRegs;
}

static bool isValidSimtKeepResumeSlot(int64_t slot, unsigned registerCount) {
  if (slot < 0 || slot >= 123)
  {
    return false;
  }
  if (registerCount == 2 && ((slot % 2) != 0 || slot + 1 >= 123))
  {
    return false;
  }
  return true;
}

class LowerKeepOpPattern final : public OpConversionPattern<pto::KeepOp> {
public:
  explicit LowerKeepOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::KeepOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::KeepOp op, pto::KeepOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (hasPreviousSameOp(op.getOperation())) {
      return rewriter.notifyMatchFailure(
          op, "only the first keep in a contiguous group is lowered");
    }

    SmallVector<pto::KeepOp, 4> keepOps = collectConsecutiveOps(op);
    SmallVector<Value, 4> payloads;
    SmallVector<Type, 4> asmResultTypes;
    SmallVector<std::pair<int64_t, unsigned>, 4> logicalSlots;
    for (pto::KeepOp keep : keepOps) {
      Value payload = rewriter.getRemappedValue(keep.getPayload());
      if (!payload)
      {
        return rewriter.notifyMatchFailure(keep, "payload is not remapped");
      }
      payload = packSimtKeepResumePayload(keep.getLoc(), payload, rewriter);
      if (!payload) {
        return rewriter.notifyMatchFailure(
            keep, "expected integer scalar up to 64 bits or f16/bf16/f32");
      }
      int64_t slot = keep.getSlot();
      unsigned registerCount =
          getSimtKeepResumeRegisterCount(payload.getType());
      if (!isValidSimtKeepResumeSlot(slot, registerCount)) {
        return rewriter.notifyMatchFailure(
            keep,
            "slot must be in range [0, 122] and 64-bit slots must be even");
      }
      logicalSlots.push_back({slot, registerCount});
      payloads.push_back(payload);
      asmResultTypes.push_back(payload.getType());
    }
    FailureOr<SmallVector<SimtKeepResumePhysicalRegister, 4>> physicalRegs =
        computeSimtKeepResumePhysicalRegs(logicalSlots);
    if (failed(physicalRegs)) {
      return rewriter.notifyMatchFailure(
          op, "keep slots must map to valid non-overlapping SIMT registers");
    }

    Type asmResultType = asmResultTypes.front();
    if (asmResultTypes.size() > 1) {
      asmResultType =
          LLVM::LLVMStructType::getLiteral(op.getContext(), asmResultTypes);
    }
    rewriter.setInsertionPoint(op);
    rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{asmResultType}, payloads, "",
        buildSimtKeepResumeConstraints(*physicalRegs, true), true, false,
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr{});
    for (pto::KeepOp keep : llvm::reverse(keepOps))
    {
      rewriter.eraseOp(keep);
    }
    return success();
  }
};

class LowerResumeOpPattern final : public OpConversionPattern<pto::ResumeOp> {
public:
  explicit LowerResumeOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::ResumeOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::ResumeOp op, pto::ResumeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (hasPreviousSameOp(op.getOperation())) {
      return rewriter.notifyMatchFailure(
          op, "only the first resume in a contiguous group is lowered");
    }

    SmallVector<pto::ResumeOp, 4> resumeOps = collectConsecutiveOps(op);
    SmallVector<std::pair<int64_t, unsigned>, 4> logicalSlots;
    SmallVector<Type, 4> asmResultTypes;
    for (pto::ResumeOp resume : resumeOps) {
      Type resultType = getTypeConverter()->convertType(resume.getType());
      if (!resultType || !getSimtKeepResumeBitWidth(resultType)) {
        return rewriter.notifyMatchFailure(
            resume, "expected integer scalar up to 64 bits or f16/bf16/f32");
      }
      int64_t slot = resume.getSlot();
      unsigned registerCount = getSimtKeepResumeRegisterCount(resultType);
      if (!isValidSimtKeepResumeSlot(slot, registerCount)) {
        return rewriter.notifyMatchFailure(
            resume,
            "slot must be in range [0, 122] and 64-bit slots must be even");
      }
      logicalSlots.push_back({slot, registerCount});
      asmResultTypes.push_back(rewriter.getIntegerType(
          *getSimtKeepResumeBitWidth(resultType) > 32 ? 64 : 32));
    }
    FailureOr<SmallVector<SimtKeepResumePhysicalRegister, 4>> physicalRegs =
        computeSimtKeepResumePhysicalRegs(logicalSlots);
    if (failed(physicalRegs)) {
      return rewriter.notifyMatchFailure(
          op, "resume slots must map to valid non-overlapping SIMT registers");
    }

    Type asmResultType = asmResultTypes.front();
    if (asmResultTypes.size() > 1) {
      asmResultType =
          LLVM::LLVMStructType::getLiteral(op.getContext(), asmResultTypes);
    }
    rewriter.setInsertionPoint(op);
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{asmResultType}, ValueRange{}, "",
        buildSimtKeepResumeConstraints(*physicalRegs, false), true, false,
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr{});

    if (resumeOps.size() == 1) {
      Type resultType = getTypeConverter()->convertType(op.getType());
      Value result = unpackSimtKeepResumePayload(op.getLoc(), asmOp.getRes(),
                                                 resultType, rewriter);
      if (!result)
      {
        return rewriter.notifyMatchFailure(op, "failed to unpack result");
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    rewriter.setInsertionPointAfter(asmOp);
    SmallVector<Value, 4> results;
    for (auto [index, resume] : llvm::enumerate(resumeOps)) {
      auto extract = rewriter.create<LLVM::ExtractValueOp>(
          resume.getLoc(), asmOp.getRes(),
          ArrayRef<int64_t>{static_cast<int64_t>(index)});
      Type resultType = getTypeConverter()->convertType(resume.getType());
      Value result = unpackSimtKeepResumePayload(
          resume.getLoc(), extract.getRes(), resultType, rewriter);
      if (!result)
      {
        return rewriter.notifyMatchFailure(resume, "failed to unpack result");
      }
      results.push_back(result);
    }
    for (auto [resume, result] : llvm::zip(resumeOps, results))
    {
      rewriter.replaceOp(resume, result);
    }
    return success();
  }
};

template <typename ConfigOp>
class LowerNullaryConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerNullaryConfigOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName = buildNullaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerPipeEventSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventSyncOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    auto event = parseEventImmediate(stringifyEVENT(op.getEventId().getEvent()));
    if (!src || !dst || !event)
    {
      return rewriter.notifyMatchFailure(op, "unsupported sync immediate");
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);
    Value eventValue = getI64Constant(rewriter, op.getLoc(), *event);
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerNamedSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerNamedSyncOpPattern(TypeConverter &tc, MLIRContext *ctx,
                                   LoweringState &state)
      : OpConversionPattern<SyncOp>(tc, ctx), state(state) {}
  LogicalResult matchAndRewrite(
      SyncOp op, typename SyncOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe) {
      return rewriter.notifyMatchFailure(op, "unsupported sync pipe");
    }
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    Value eventValue;
    if (IntegerAttr attr = op.getEventIdAttr()) {
      eventValue = getI64Constant(rewriter, op.getLoc(), attr.getInt());
    } else {
      eventValue = castIntegerLikeTo(op, adaptor.getEventIdDyn(),
                                     rewriter.getI64Type());
      if (!eventValue) {
        return rewriter.notifyMatchFailure(op, "missing event-id operand");
      }
    }
    StringRef callee = buildSyncCallee<SyncOp>(op.getContext());
    auto fnTy = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), callee, TypeRange{},
                                  ValueRange{pipeValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{callee.str(), fnTy});
    rewriter.eraseOp(op);
    return success();
  }
private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerPipeEventDynSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventDynSyncOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    if (!src || !dst)
    {
      return rewriter.notifyMatchFailure(op, "unsupported sync pipe");
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);

    Value eventIdValue = adaptor.getEventId();
    if (!eventIdValue)
    {
      return rewriter.notifyMatchFailure(op, "missing event_id operand");
    }

    Value eventValue = eventIdValue;

    while (eventValue.getDefiningOp()) {
      auto unrealizedCast = dyn_cast<UnrealizedConversionCastOp>(eventValue.getDefiningOp());
      if (!unrealizedCast || unrealizedCast.getInputs().size() != 1)
      {
        break;
      }
      eventValue = unrealizedCast.getInputs()[0];
    }

    if (eventValue.getType().isIndex()) {
      eventValue = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                        rewriter.getI64Type(),
                                                        eventValue);
    } else if (auto intType = dyn_cast<IntegerType>(eventValue.getType())) {
      if (intType.getWidth() < 64) {
        eventValue = rewriter.create<LLVM::ZExtOp>(op.getLoc(),
                                                    rewriter.getI64Type(),
                                                    eventValue);
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unexpected event_id type");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerInterCoreSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerInterCoreSyncOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe)
    {
      return rewriter.notifyMatchFailure(op, "unsupported inter-core sync pipe");
    }

    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    Value eventValue;
    if (IntegerAttr eventIdAttr = op.getEventIdAttr()) {
      eventValue = getI64Constant(rewriter, op.getLoc(), eventIdAttr.getInt());
    } else {
      Value eventIdDyn = adaptor.getEventIdDyn();
      if (!eventIdDyn) {
        return rewriter.notifyMatchFailure(
            op, "expected static or dynamic event-id operand");
      }

      eventValue = castIntegerLikeTo(op, eventIdDyn, rewriter.getI64Type());
      if (!eventValue) {
        return rewriter.notifyMatchFailure(
            op, "failed to cast dynamic event-id to i64");
      }
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    SmallVector<Value> args{pipeValue, eventValue};
    if constexpr (std::is_same_v<SyncOp, pto::SyncSetOp>) {
      int64_t mode = 2;
      if (IntegerAttr attr = op.getFftsModeAttr()) {
        mode = attr.getInt();
      }
      Value modeValue = getI64Constant(rewriter, op.getLoc(), mode);
      Value one = getI64Constant(rewriter, op.getLoc(), 1);
      Value modeMask = getI64Constant(rewriter, op.getLoc(), 0x3);
      Value eventMask = getI64Constant(rewriter, op.getLoc(), 0xf);
      modeValue = rewriter.create<arith::AndIOp>(op.getLoc(), modeValue,
                                                  modeMask);
      eventValue = rewriter.create<arith::AndIOp>(op.getLoc(), eventValue,
                                                   eventMask);
      Value modeShift = rewriter.create<arith::ShLIOp>(op.getLoc(), modeValue,
          getI64Constant(rewriter, op.getLoc(), 4));
      Value eventShift = rewriter.create<arith::ShLIOp>(op.getLoc(), eventValue,
          getI64Constant(rewriter, op.getLoc(), 8));
      Value msg = rewriter.create<arith::OrIOp>(op.getLoc(), one, modeShift);
      msg = rewriter.create<arith::OrIOp>(op.getLoc(), msg, eventShift);
      args = {pipeValue, msg};
    } else if constexpr (std::is_same_v<SyncOp, pto::SyncWaitOp>) {
      calleeName = op.getEventIdAttr()
                       ? StringAttr::get(op.getContext(),
                                         "llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM")
                             .getValue()
                       : StringAttr::get(op.getContext(),
                                         "llvm.hivm.WAIT.FLAG.DEV.PIPE.REG")
                             .getValue();
    }
    auto funcType = rewriter.getFunctionType(
        TypeRange{args[0].getType(), args[1].getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerBarrierOpPattern final : public OpConversionPattern<pto::BarrierOp> {
public:
  explicit LowerBarrierOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::BarrierOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (isTargetArchA5(op.getOperation()) &&
        op.getPipe().getPipe() == PIPE::PIPE_V) {
      op.emitError("internal error: A5 PIPE_V barrier should be erased before "
                   "VPTO LLVM lowering");
      return failure();
    }

    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe)
    {
      return rewriter.notifyMatchFailure(op, "unsupported barrier pipe");
    }

    StringRef calleeName = buildSyncCallee<pto::BarrierOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerMemBarOpPattern final : public OpConversionPattern<pto::MemBarOp> {
public:
  explicit LowerMemBarOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MemBarOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::MemBarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName = buildMemBarCallee(op.getKind().getKind(), op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename MemoryConsistencyOp>
class LowerUnsupportedMemoryConsistencyOpPattern final
    : public OpConversionPattern<MemoryConsistencyOp> {
public:
  explicit LowerUnsupportedMemoryConsistencyOpPattern(
      TypeConverter &typeConverter, MLIRContext *context,
      LoweringState &state)
      : OpConversionPattern<MemoryConsistencyOp>(typeConverter, context) {
    (void)state;
  }

  LogicalResult
  matchAndRewrite(MemoryConsistencyOp op,
                  typename MemoryConsistencyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    (void)rewriter;
    op.emitOpError()
        << "is not supported by the VPTO backend yet; PTOAS validates the "
           "memory-consistency contract, but high-level CMO/fence ops must be "
           "lowered to `pto.dcci` or `pto.dsb` before VPTO LLVM lowering";
    return failure();
  }
};

class LowerDsbOpPattern final : public OpConversionPattern<pto::DsbOp> {
public:
  explicit LowerDsbOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::DsbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::DsbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName =
        StringAttr::get(op.getContext(), "llvm.hivm.DSB").getValue();
    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(TypeRange{i64Ty}, TypeRange{});
    Value mem =
        getI64Constant(rewriter, op.getLoc(),
                       getDsbMemImmediate(op.getMem().getKind()));
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{mem});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerDcciOpPattern final : public OpConversionPattern<pto::DcciOp> {
public:
  explicit LowerDcciOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::DcciOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::DcciOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!ptrType)
    {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    bool hasDst = static_cast<bool>(op.getDstAttr());
    StringRef calleeName =
        buildDcciCallee(ptrType.getAddressSpace(), hasDst, op.getContext());

    Type i64Ty = rewriter.getI64Type();
    SmallVector<Type> argTypes{ptrType, i64Ty};
    SmallVector<Value> args{
        adaptor.getPtr(),
        getI64Constant(rewriter, op.getLoc(),
                       getDcciCacheLineImmediate(op.getCache().getKind()))};
    if (auto dst = op.getDstAttr()) {
      argTypes.push_back(i64Ty);
      args.push_back(getI64Constant(rewriter, op.getLoc(),
                                    getDcciDstImmediate(dst.getKind())));
    }

    auto funcType = rewriter.getFunctionType(argTypes, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BufSyncOp>
class LowerBufSyncOpPattern final : public OpConversionPattern<BufSyncOp> {
public:
  explicit LowerBufSyncOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BufSyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BufSyncOp op, typename BufSyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    if (auto pipeAttr = dyn_cast<PipeAttr>(op.getOpTypeAttr())) {
      pipe = pipeAttr.getPipe();
    } else {
      auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
      if (failed(opTypeOr)) {
        return rewriter.notifyMatchFailure(
            op, "buffer sync expects pipe/sync_op_type/pipe_event_type attr");
      }
      pipe = mapSyncOpTypeToPipe(*opTypeOr);
    }
    if (!isConcreteSyncPipe(pipe)) {
      return rewriter.notifyMatchFailure(op,
                                         "buffer sync op_type cannot map to concrete pipe");
    }

    auto pipeImm = parsePipeImmediate(stringifyPIPE(pipe));
    if (!pipeImm)
    {
      return rewriter.notifyMatchFailure(op, "unsupported buffer sync pipe");
    }

    StringRef calleeName = buildSyncCallee<BufSyncOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipeImm);
    Value bufIdValue =
        getI64Constant(rewriter, op.getLoc(), op.getBufIdAttr().getInt());
    Value modeValue =
        getI64Constant(rewriter, op.getLoc(), op.getModeAttr().getInt());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue, bufIdValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BufDynSyncOp>
class LowerBufDynSyncOpPattern final
    : public OpConversionPattern<BufDynSyncOp> {
public:
  explicit LowerBufDynSyncOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BufDynSyncOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(BufDynSyncOp op, typename BufDynSyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    if (auto pipeAttr = dyn_cast<PipeAttr>(op.getOpTypeAttr())) {
      pipe = pipeAttr.getPipe();
    } else {
      auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
      if (failed(opTypeOr)) {
        return rewriter.notifyMatchFailure(
            op, "buffer sync expects pipe/sync_op_type/pipe_event_type attr");
      }
      pipe = mapSyncOpTypeToPipe(*opTypeOr);
    }
    if (!isConcreteSyncPipe(pipe)) {
      return rewriter.notifyMatchFailure(
          op, "buffer sync op_type cannot map to concrete pipe");
    }

    auto pipeImm = parsePipeImmediate(stringifyPIPE(pipe));
    if (!pipeImm)
    {
      return rewriter.notifyMatchFailure(op, "unsupported buffer sync pipe");
    }

    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipeImm);
    Value bufIdDyn = adaptor.getBufId();
    if (!bufIdDyn) {
      return rewriter.notifyMatchFailure(
          op, "expected dynamic buf-id operand");
    }
    Value bufIdValue = castIntegerLikeTo(op, bufIdDyn, rewriter.getI64Type());
    if (!bufIdValue) {
      return rewriter.notifyMatchFailure(
          op, "failed to cast dynamic buf-id to i64");
    }

    bool isGetBuf =
        std::is_same_v<BufDynSyncOp, pto::GetBufDynOp>;
    StringRef calleeName =
        buildBufDynSyncCallee(op.getContext(), isGetBuf);
    Value modeValue =
        getI64Constant(rewriter, op.getLoc(), op.getModeAttr().getInt());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue, bufIdValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

} // namespace

void populateVPTOSyncAndConfigPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        LoweringState &state) {
  patterns.add<LowerSimtFenceOpPattern<pto::SyncthreadsOp>,
               LowerSimtFenceOpPattern<pto::ThreadfenceOp>,
               LowerSimtFenceOpPattern<pto::ThreadfenceBlockOp>,
               LowerKeepOpPattern, LowerResumeOpPattern,
               LowerSetLoopConfigOpPattern<pto::SetLoop2StrideOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop1StrideOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoopSizeOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop2StrideUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop1StrideUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoopSizeUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop3ParaOp>,
               LowerSetLoopConfigOpPattern<pto::SetChannelParaOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetCtrlOp>,
               LowerStoreVfSimtInfoOpPattern,
               LowerUnaryConfigOpPattern<pto::SetMovPadValOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetQuantPreOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetReluAlphaOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetFixClipReluOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetLoop2StrideOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetLoop1StrideOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetLoopSizeOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetMte2NzParaOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetPadValOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetFpcOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetStoreAtomicCfgOp>,
               LowerNullaryConfigOpPattern<pto::SetAtomicS32Op>,
               LowerNullaryConfigOpPattern<pto::SetAtomicS8Op>,
               LowerPipeEventSyncOpPattern<pto::SetFlagOp>,
               LowerPipeEventSyncOpPattern<pto::WaitFlagOp>,
               LowerPipeEventDynSyncOpPattern<pto::SetFlagDynOp>,
               LowerPipeEventDynSyncOpPattern<pto::WaitFlagDynOp>,
               LowerBarrierOpPattern, LowerMemBarOpPattern,
               LowerUnsupportedMemoryConsistencyOpPattern<pto::CmoCacheInvalidOp>,
               LowerUnsupportedMemoryConsistencyOpPattern<pto::FenceBarrierAllOp>,
               LowerDsbOpPattern, LowerDcciOpPattern,
               LowerBufSyncOpPattern<pto::GetBufOp>,
               LowerBufSyncOpPattern<pto::RlsBufOp>,
               LowerBufDynSyncOpPattern<pto::GetBufDynOp>,
               LowerBufDynSyncOpPattern<pto::RlsBufDynOp>,
               LowerInterCoreSyncOpPattern<pto::SyncSetOp>,
               LowerInterCoreSyncOpPattern<pto::SyncWaitOp>,
               LowerNamedSyncOpPattern<pto::SetIntraBlockOp>,
               LowerNamedSyncOpPattern<pto::WaitIntraBlockOp>>(
      typeConverter, patterns.getContext(), state);
}
} // namespace mlir::pto
