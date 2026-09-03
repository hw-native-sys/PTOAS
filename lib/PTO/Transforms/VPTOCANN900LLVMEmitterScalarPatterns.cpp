// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterTemplates.h"

namespace mlir::pto::detail {

class LowerTrapOpPattern final : public OpConversionPattern<pto::TrapOp> {
public:
  explicit LowerTrapOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::TrapOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::TrapOp op, pto::TrapOp::Adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    constexpr StringLiteral calleeName = "llvm.hivm.TRAP";
    auto funcType = rewriter.getFunctionType({}, {});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

static LogicalResult appendVcvtImmediate(pto::VcvtOp op, StringRef name, std::optional<uint64_t> immediate,
                                         SmallVectorImpl<Value> &callArgs, SmallVectorImpl<Type> &argTypes,
                                         ConversionPatternRewriter &rewriter) {
  if (!immediate) {
    StringRef message = name == "rnd"   ? "vcvt requires valid rnd attr"
                        : name == "sat" ? "vcvt requires valid sat attr"
                                        : "vcvt requires valid part attr";
    return rewriter.notifyMatchFailure(op, message);
  }
  Value value = getI32Constant(rewriter, op.getLoc(), *immediate);
  callArgs.push_back(value);
  argTypes.push_back(value.getType());
  return success();
}

static LogicalResult appendVcvtOptionalArguments(pto::VcvtOp op, const VcvtContract &contract,
                                                 SmallVectorImpl<Value> &callArgs, SmallVectorImpl<Type> &argTypes,
                                                 ConversionPatternRewriter &rewriter) {
  auto appendRound = [&]() {
    return appendVcvtImmediate(op, "rnd", op.getRndAttr() ? parseRoundModeImmediate(*op.getRnd()) : std::nullopt,
                               callArgs, argTypes, rewriter);
  };
  auto appendSaturation = [&]() {
    return appendVcvtImmediate(op, "sat", op.getSatAttr() ? parseSaturationImmediate(*op.getSat()) : std::nullopt,
                               callArgs, argTypes, rewriter);
  };
  if (contract.satBeforeRnd) {
    if (contract.requiresSat && failed(appendSaturation())) {
      return failure();
    }
    if (contract.requiresRnd && failed(appendRound())) {
      return failure();
    }
  } else {
    if (contract.requiresRnd && failed(appendRound())) {
      return failure();
    }
    if (contract.requiresSat && failed(appendSaturation())) {
      return failure();
    }
  }
  if (!contract.requiresPart) {
    return success();
  }
  return appendVcvtImmediate(op, "part", op.getPartAttr() ? parseVcvtPartImmediate(*op.getPart()) : std::nullopt,
                             callArgs, argTypes, rewriter);
}

class LowerVcvtOpPattern final : public OpConversionPattern<pto::VcvtOp> {
public:
  explicit LowerVcvtOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VcvtOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VcvtOp op, pto::VcvtOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<VcvtContract> contract = buildVcvtContract(op);
    if (failed(contract)) {
      return rewriter.notifyMatchFailure(op, "unsupported vcvt type pair");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vcvt result type");
    }

    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;
    callArgs.push_back(adaptor.getInput());
    argTypes.push_back(adaptor.getInput().getType());
    callArgs.push_back(adaptor.getMask());
    argTypes.push_back(adaptor.getMask().getType());
    if (failed(appendVcvtOptionalArguments(op, *contract, callArgs, argTypes, rewriter))) {
      return failure();
    }

    auto funcType = rewriter.getFunctionType(argTypes, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), StringRef((*contract).intrinsic), TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(PlannedDecl{std::string((*contract).intrinsic), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbitcastOpPattern final : public OpConversionPattern<pto::VbitcastOp> {
public:
  explicit LowerVbitcastOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbitcastOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(pto::VbitcastOp op, pto::VbitcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // A vbitcast whose result has no users is a dead noop (Pure). Erase it
    // instead of emitting an LLVM bitcast the device compiler may not lower
    // (e.g. bf16x2 <-> bf16 physical views).
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vbitcast result type");
    }
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType, adaptor.getInput());
    return success();
  }
};

class LowerPbitcastOpPattern final : public OpConversionPattern<pto::PbitcastOp> {
public:
  explicit LowerPbitcastOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PbitcastOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(pto::PbitcastOp op, pto::PbitcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert pbitcast result type");
    }
    if (adaptor.getInput().getType() != resultType) {
      return rewriter.notifyMatchFailure(op, "pbitcast expects identical lowered input/result types");
    }
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

class LowerVtrcOpPattern final : public OpConversionPattern<pto::VtrcOp> {
public:
  explicit LowerVtrcOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VtrcOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::VtrcOp op, pto::VtrcOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto roundMode = parseRoundModeImmediate(op.getRoundMode());
    if (!roundMode) {
      return rewriter.notifyMatchFailure(op, "unsupported vtrc signature");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vtrc result type");
    }

    FailureOr<StringRef> calleeName = buildVtrcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported vtrc callee");
    }

    Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), roundValue.getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getInput(), roundValue, adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename StoreOp>
static SmallVector<Value> buildPredicateStoreCallArgs(StoreOp op, typename StoreOp::Adaptor adaptor,
                                                      const VPTOLoweredAddressOffset &offset, uint64_t dist,
                                                      bool usePostIntrinsic, ConversionPatternRewriter &rewriter) {
  Value distValue = getI32Constant(rewriter, op.getLoc(), dist);
  Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
  return {adaptor.getValue(), offset.base, offset.intrinsicOffset, distValue, postValue};
}

template <typename StoreOp>
static void replacePredicateStoreOp(StoreOp op, bool usePostIntrinsic, const VPTOLoweredAddressOffset &offset,
                                    func::CallOp call, ConversionPatternRewriter &rewriter) {
  if (!usePostIntrinsic) {
    rewriter.eraseOp(op);
    return;
  }
  if (offset.updatedBase) {
    rewriter.replaceOp(op, offset.updatedBase);
    return;
  }
  rewriter.replaceOp(op, call.getResults());
}

template <typename StoreOp> class LowerPredicateStoreOpPattern final : public OpConversionPattern<StoreOp> {
public:
  explicit LowerPredicateStoreOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<StoreOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(StoreOp op, typename StoreOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmDestType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!llvmDestType || !valueType) {
      return rewriter.notifyMatchFailure(op, "expected converted predicate-store operand types");
    }

    auto dist = parsePredicateStoreDistImmediate(op.getDist());
    if (!dist) {
      return rewriter.notifyMatchFailure(op, "unsupported predicate-store dist immediate");
    }

    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    auto loweredOffset = lowerVPTOPredicateOffsetForIntrinsic(op, adaptor.getDestination(), adaptor.getOffset(),
                                                              usePostIntrinsic, rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(op, "failed to preserve predicate-store index offset");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 1U : 0U)) {
      return rewriter.notifyMatchFailure(op, "failed to convert predicate-store result types");
    }

    StringRef calleeName = getPredicateStoreCallee<StoreOp>(op.getContext(), usePostIntrinsic);
    SmallVector<Value> args =
        buildPredicateStoreCallArgs(op, adaptor, *loweredOffset, *dist, usePostIntrinsic, rewriter);
    auto funcType = rewriter.getFunctionType(
        TypeRange{valueType, llvmDestType, rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    replacePredicateStoreOp(op, usePostIntrinsic, *loweredOffset, call, rewriter);
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp> class LowerPredicateLoadOpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerPredicateLoadOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    bool usePostIntrinsic = op.getUpdatedBase() != nullptr;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != (usePostIntrinsic ? 2U : 1U)) {
      return rewriter.notifyMatchFailure(op, "failed to convert predicate-load result types");
    }
    if (!llvmSourceType) {
      return rewriter.notifyMatchFailure(op, "expected converted predicate-load operand/result types");
    }

    auto dist = parsePredicateLoadDistImmediate(op.getDist());
    if (!dist) {
      return rewriter.notifyMatchFailure(op, "unsupported predicate-load dist immediate");
    }

    auto loweredOffset =
        lowerVPTOPredicateOffsetForIntrinsic(op, adaptor.getSource(), adaptor.getOffset(), usePostIntrinsic, rewriter);
    if (failed(loweredOffset)) {
      return rewriter.notifyMatchFailure(op, "failed to preserve predicate-load index offset");
    }

    StringRef calleeName = getPredicateLoadCallee<LoadOp>(op.getContext(), usePostIntrinsic);
    SmallVector<Value> args;
    args.push_back(loweredOffset->base);
    args.push_back(loweredOffset->intrinsicOffset);
    args.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(
        rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(usePostIntrinsic ? 1 : 0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmSourceType, rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type()}, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (loweredOffset->updatedBase) {
      rewriter.replaceOp(op, ValueRange{call.getResult(0), loweredOffset->updatedBase});
    } else {
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoopOp> class LowerSetLoopConfigOpPattern final : public OpConversionPattern<LoopOp> {
public:
  explicit LowerSetLoopConfigOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<LoopOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(LoopOp op, typename LoopOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> packed = failure();
    if constexpr (std::is_same_v<LoopOp, pto::SetLoopSizeOutToUbOp> ||
                  std::is_same_v<LoopOp, pto::SetLoopSizeUbToOutOp>) {
      packed = packLoopSize(op, adaptor.getFirst(), adaptor.getSecond());
    } else {
      packed = packLoopPair(op, adaptor.getFirst(), adaptor.getSecond());
    }
    if (failed(packed)) {
      return rewriter.notifyMatchFailure(op, "failed to pack loop configuration");
    }

    StringRef calleeName = buildSetLoopCallee<LoopOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{*packed});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp> class LowerUnaryConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerUnaryConfigOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> encoded = encodeMovPadValue(op.getLoc(), adaptor.getValue(), rewriter);
    if (failed(encoded)) {
      return rewriter.notifyMatchFailure(op, "expected 8/16/32-bit integer or float mov-pad payload");
    }

    StringRef calleeName = buildUnaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{*encoded});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp> class LowerUnaryI64ConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerUnaryI64ConfigOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = buildUnaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getValue().getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerStoreVfSimtInfoOpPattern final : public OpConversionPattern<pto::StoreVfSimtInfoOp> {
public:
  explicit LowerStoreVfSimtInfoOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::StoreVfSimtInfoOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::StoreVfSimtInfoOp op, pto::StoreVfSimtInfoOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value dimZ = adaptor.getDimZ();
    Value dimY = adaptor.getDimY();
    Value dimX = adaptor.getDimX();
    if (!dimZ || !dimY || !dimX) {
      return rewriter.notifyMatchFailure(op, "missing converted SIMT dims");
    }

    auto i64Type = rewriter.getI64Type();
    auto castToI64 = [&](Value value) -> Value {
      if (value.getType().isInteger(64)) {
        return value;
      }
      return rewriter.create<arith::ExtUIOp>(loc, i64Type, value).getResult();
    };

    Value dimZI64 = castToI64(dimZ);
    Value dimYI64 = castToI64(dimY);
    Value dimXI64 = castToI64(dimX);
    Value dimYShift = rewriter.create<arith::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(16));
    Value dimZShift = rewriter.create<arith::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(32));
    Value packedDimY = rewriter.create<arith::ShLIOp>(loc, dimYI64, dimYShift).getResult();
    Value packedDimZ = rewriter.create<arith::ShLIOp>(loc, dimZI64, dimZShift).getResult();
    Value payload = rewriter.create<arith::OrIOp>(loc, dimXI64, packedDimY).getResult();
    payload = rewriter.create<arith::OrIOp>(loc, payload, packedDimZ).getResult();

    StringRef calleeName = buildStoreVfSimtInfoCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{i64Type}, TypeRange{});
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{}, ValueRange{payload});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename FenceOp> static StringRef buildSimtFenceCallee(MLIRContext *context);

template <> StringRef buildSimtFenceCallee<pto::SyncthreadsOp>(MLIRContext *context) {
  return buildSyncthreadsCallee(context);
}

template <> StringRef buildSimtFenceCallee<pto::ThreadfenceOp>(MLIRContext *context) {
  return buildThreadfenceCallee(context);
}

template <> StringRef buildSimtFenceCallee<pto::ThreadfenceBlockOp>(MLIRContext *context) {
  return buildThreadfenceBlockCallee(context);
}

template <typename FenceOp> class LowerSimtFenceOpPattern final : public OpConversionPattern<FenceOp> {
public:
  explicit LowerSimtFenceOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<FenceOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(FenceOp op, typename FenceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    FunctionType funcType = rewriter.getFunctionType({}, {});
    StringRef calleeName = buildSimtFenceCallee<FenceOp>(op.getContext());
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{});
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
static std::string buildSimtKeepResumeConstraints(ArrayRef<SimtKeepResumePhysicalRegister> physicalRegs,
                                                  bool tieInputs) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto [index, physicalReg] : llvm::enumerate(physicalRegs)) {
    if (index != 0) {
      os << ",";
    }
    if (physicalReg.registerCount == 2) {
      os << "={TPERL" << physicalReg.baseRegister / 2 << "}";
    } else {
      os << "={TPER" << physicalReg.baseRegister << "}";
    }
  }
  if (tieInputs) {
    for (size_t index = 0; index < physicalRegs.size(); ++index) {
      os << "," << index;
    }
  }
  return os.str();
}

template <typename OpT> static SmallVector<OpT, 4> collectConsecutiveOps(OpT first) {
  SmallVector<OpT, 4> ops;
  for (Operation *cur = first.getOperation(); cur; cur = cur->getNextNode()) {
    auto typed = dyn_cast<OpT>(cur);
    if (!typed) {
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
    if (intType.getWidth() <= 64) {
      return intType.getWidth();
    }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16()) {
    return 16;
  }
  if (type.isF32()) {
    return 32;
  }
  return std::nullopt;
}

static Value packSimtKeepResumePayload(Location loc, Value value, ConversionPatternRewriter &rewriter) {
  Type type = value.getType();
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(type);
  if (!width) {
    return {};
  }

  Type intType = rewriter.getIntegerType(*width);
  Value bits = value;
  if (!isa<IntegerType>(type)) {
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, value);
  } else if (bits.getType() != intType) {
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, bits);
  }
  if (*width < 32) {
    return rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI32Type(), bits);
  }
  if (*width == 32 && bits.getType() != rewriter.getI32Type()) {
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), bits);
  }
  return bits;
}

static Value unpackSimtKeepResumePayload(Location loc, Value value, Type resultType,
                                         ConversionPatternRewriter &rewriter) {
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(resultType);
  if (!width) {
    return {};
  }

  Type intType = rewriter.getIntegerType(*width);
  Value bits = value;
  if (*width < 32) {
    bits = rewriter.create<LLVM::TruncOp>(loc, intType, bits);
  } else if (bits.getType() != intType) {
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, bits);
  }

  if (isa<IntegerType>(resultType)) {
    if (bits.getType() == resultType) {
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
computeSimtKeepResumePhysicalRegs(ArrayRef<std::pair<int64_t, unsigned>> logicalSlots) {
  SmallVector<SimtKeepResumePhysicalRegister, 4> physicalRegs;
  physicalRegs.reserve(logicalSlots.size());
  for (auto [slot, registerCount] : logicalSlots) {
    if (slot < 0 || slot >= 123) {
      return failure();
    }
    if (registerCount == 2 && ((slot % 2) != 0 || slot + 1 >= 123)) {
      return failure();
    }
    // Slots are user-assigned storage words, not dense ordinals in the current
    // keep/resume group. This keeps a consumer that resumes only a subset of
    // slots from changing where the remaining slots are read from.
    int64_t baseRegister = 4 + slot;
    if (baseRegister + static_cast<int64_t>(registerCount) - 1 > 126) {
      return failure();
    }
    physicalRegs.push_back({baseRegister, registerCount});
  }
  return physicalRegs;
}

static bool isValidSimtKeepResumeSlot(int64_t slot, unsigned registerCount) {
  if (slot < 0 || slot >= 123) {
    return false;
  }
  if (registerCount == 2 && ((slot % 2) != 0 || slot + 1 >= 123)) {
    return false;
  }
  return true;
}

struct ResumeGroupTypes {
  SmallVector<std::pair<int64_t, unsigned>, 4> logicalSlots;
  SmallVector<Type, 4> asmResultTypes;
};

static FailureOr<ResumeGroupTypes> collectResumeGroupTypes(ArrayRef<pto::ResumeOp> resumeOps,
                                                           const TypeConverter &typeConverter,
                                                           ConversionPatternRewriter &rewriter) {
  ResumeGroupTypes types;
  for (unsigned index = 0; index < resumeOps.size(); ++index) {
    pto::ResumeOp resume = resumeOps[index];
    Type resultType = typeConverter.convertType(resume.getType());
    std::optional<unsigned> bitWidth = getSimtKeepResumeBitWidth(resultType);
    if (!resultType || !bitWidth) {
      return failure();
    }
    unsigned registerCount = getSimtKeepResumeRegisterCount(resultType);
    if (!isValidSimtKeepResumeSlot(resume.getSlot(), registerCount)) {
      return failure();
    }
    types.logicalSlots.push_back({resume.getSlot(), registerCount});
    types.asmResultTypes.push_back(rewriter.getIntegerType(*bitWidth > 32 ? 64 : 32));
  }
  return types;
}

static LogicalResult replaceResumeGroup(ArrayRef<pto::ResumeOp> resumeOps, LLVM::InlineAsmOp asmOp,
                                        const TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  SmallVector<Value, 4> results;
  for (unsigned index = 0; index < resumeOps.size(); ++index) {
    pto::ResumeOp resume = resumeOps[index];
    auto extract = rewriter.create<LLVM::ExtractValueOp>(resume.getLoc(), asmOp.getRes(),
                                                         ArrayRef<int64_t>{static_cast<int64_t>(index)});
    Type resultType = typeConverter.convertType(resume.getType());
    Value result = unpackSimtKeepResumePayload(resume.getLoc(), extract.getRes(), resultType, rewriter);
    if (!result) {
      return failure();
    }
    results.push_back(result);
  }
  for (auto [resume, result] : llvm::zip(resumeOps, results)) {
    rewriter.replaceOp(resume, result);
  }
  return success();
}

class LowerKeepOpPattern final : public OpConversionPattern<pto::KeepOp> {
public:
  explicit LowerKeepOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::KeepOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(pto::KeepOp op, pto::KeepOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (hasPreviousSameOp(op.getOperation())) {
      return rewriter.notifyMatchFailure(op, "only the first keep in a contiguous group is lowered");
    }

    SmallVector<pto::KeepOp, 4> keepOps = collectConsecutiveOps(op);
    SmallVector<Value, 4> payloads;
    SmallVector<Type, 4> asmResultTypes;
    SmallVector<std::pair<int64_t, unsigned>, 4> logicalSlots;
    for (pto::KeepOp keep : keepOps) {
      Value payload = rewriter.getRemappedValue(keep.getPayload());
      if (!payload) {
        return rewriter.notifyMatchFailure(keep, "payload is not remapped");
      }
      payload = packSimtKeepResumePayload(keep.getLoc(), payload, rewriter);
      if (!payload) {
        return rewriter.notifyMatchFailure(keep, "expected integer scalar up to 64 bits or f16/bf16/f32");
      }
      int64_t slot = keep.getSlot();
      unsigned registerCount = getSimtKeepResumeRegisterCount(payload.getType());
      if (!isValidSimtKeepResumeSlot(slot, registerCount)) {
        return rewriter.notifyMatchFailure(keep, "slot must be in range [0, 122] and 64-bit slots must be even");
      }
      logicalSlots.push_back({slot, registerCount});
      payloads.push_back(payload);
      asmResultTypes.push_back(payload.getType());
    }
    FailureOr<SmallVector<SimtKeepResumePhysicalRegister, 4>> physicalRegs =
        computeSimtKeepResumePhysicalRegs(logicalSlots);
    if (failed(physicalRegs)) {
      return rewriter.notifyMatchFailure(op, "keep slots must map to valid non-overlapping SIMT registers");
    }

    Type asmResultType = asmResultTypes.front();
    if (asmResultTypes.size() > 1) {
      asmResultType = LLVM::LLVMStructType::getLiteral(op.getContext(), asmResultTypes);
    }
    rewriter.setInsertionPoint(op);
    rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{asmResultType}, payloads, "", buildSimtKeepResumeConstraints(*physicalRegs, true), true,
        false, LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT), ArrayAttr{});
    for (pto::KeepOp keep : llvm::reverse(keepOps)) {
      rewriter.eraseOp(keep);
    }
    return success();
  }
};

class LowerResumeOpPattern final : public OpConversionPattern<pto::ResumeOp> {
public:
  explicit LowerResumeOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &)
      : OpConversionPattern<pto::ResumeOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(pto::ResumeOp op, pto::ResumeOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (hasPreviousSameOp(op.getOperation())) {
      return rewriter.notifyMatchFailure(op, "only the first resume in a contiguous group is lowered");
    }

    SmallVector<pto::ResumeOp, 4> resumeOps = collectConsecutiveOps(op);
    FailureOr<ResumeGroupTypes> groupTypes = collectResumeGroupTypes(resumeOps, *getTypeConverter(), rewriter);
    if (failed(groupTypes)) {
      return rewriter.notifyMatchFailure(op, "resume slots or result types are unsupported");
    }
    FailureOr<SmallVector<SimtKeepResumePhysicalRegister, 4>> physicalRegs =
        computeSimtKeepResumePhysicalRegs(groupTypes->logicalSlots);
    if (failed(physicalRegs)) {
      return rewriter.notifyMatchFailure(op, "resume slots must map to valid non-overlapping SIMT registers");
    }

    Type asmResultType = groupTypes->asmResultTypes.front();
    if (groupTypes->asmResultTypes.size() > 1) {
      asmResultType = LLVM::LLVMStructType::getLiteral(op.getContext(), groupTypes->asmResultTypes);
    }
    rewriter.setInsertionPoint(op);
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{asmResultType}, ValueRange{}, "", buildSimtKeepResumeConstraints(*physicalRegs, false),
        true, false, LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT), ArrayAttr{});

    if (resumeOps.size() == 1) {
      Type resultType = getTypeConverter()->convertType(op.getType());
      Value result = unpackSimtKeepResumePayload(op.getLoc(), asmOp.getRes(), resultType, rewriter);
      if (!result) {
        return rewriter.notifyMatchFailure(op, "failed to unpack result");
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    rewriter.setInsertionPointAfter(asmOp);
    if (failed(replaceResumeGroup(resumeOps, asmOp, *getTypeConverter(), rewriter))) {
      return rewriter.notifyMatchFailure(op, "failed to unpack resume results");
    }
    return success();
  }
};

template <typename ConfigOp> class LowerNullaryConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerNullaryConfigOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName = buildNullaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp> class LowerPipeEventSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventSyncOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    auto event = parseEventImmediate(stringifyEVENT(op.getEventId().getEvent()));
    if (!src || !dst || !event) {
      return rewriter.notifyMatchFailure(op, "unsupported sync immediate");
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);
    Value eventValue = getI64Constant(rewriter, op.getLoc(), *event);
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp> class LowerPipeEventDynSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventDynSyncOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    if (!src || !dst) {
      return rewriter.notifyMatchFailure(op, "unsupported sync pipe");
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);

    Value eventIdValue = adaptor.getEventId();
    if (!eventIdValue) {
      return rewriter.notifyMatchFailure(op, "missing event_id operand");
    }

    Value eventValue = eventIdValue;

    while (eventValue.getDefiningOp()) {
      auto unrealizedCast = dyn_cast<UnrealizedConversionCastOp>(eventValue.getDefiningOp());
      if (!unrealizedCast || unrealizedCast.getInputs().size() != 1) {
        break;
      }
      eventValue = unrealizedCast.getInputs()[0];
    }

    if (eventValue.getType().isIndex()) {
      eventValue = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI64Type(), eventValue);
    } else if (auto intType = dyn_cast<IntegerType>(eventValue.getType())) {
      if (intType.getWidth() < 64) {
        eventValue = rewriter.create<LLVM::ZExtOp>(op.getLoc(), rewriter.getI64Type(), eventValue);
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unexpected event_id type");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
static FailureOr<Value> getInterCoreEventValue(SyncOp op, typename SyncOp::Adaptor adaptor,
                                               ConversionPatternRewriter &rewriter) {
  if (IntegerAttr eventIdAttr = op.getEventIdAttr()) {
    return getI64Constant(rewriter, op.getLoc(), eventIdAttr.getInt());
  }
  Value eventId = adaptor.getEventIdDyn();
  if (!eventId) {
    return failure();
  }
  Value eventValue = castIntegerLikeTo(op, eventId, rewriter.getI64Type());
  if (!eventValue) {
    return failure();
  }
  return eventValue;
}

template <typename SyncOp>
static SmallVector<Value> buildInterCoreSyncArgs(SyncOp op, Value pipeValue, Value eventValue, StringRef &calleeName,
                                                 ConversionPatternRewriter &rewriter) {
  if constexpr (std::is_same_v<SyncOp, pto::SyncSetOp>) {
    int64_t mode = op.getFftsModeAttr() ? op.getFftsModeAttr().getInt() : 2;
    Value modeValue = getI64Constant(rewriter, op.getLoc(), mode);
    modeValue = rewriter.create<arith::AndIOp>(op.getLoc(), modeValue, getI64Constant(rewriter, op.getLoc(), 0x3));
    eventValue = rewriter.create<arith::AndIOp>(op.getLoc(), eventValue, getI64Constant(rewriter, op.getLoc(), 0xf));
    Value modeShift = rewriter.create<arith::ShLIOp>(op.getLoc(), modeValue, getI64Constant(rewriter, op.getLoc(), 4));
    Value eventShift =
        rewriter.create<arith::ShLIOp>(op.getLoc(), eventValue, getI64Constant(rewriter, op.getLoc(), 8));
    Value message = rewriter.create<arith::OrIOp>(op.getLoc(), getI64Constant(rewriter, op.getLoc(), 1), modeShift);
    message = rewriter.create<arith::OrIOp>(op.getLoc(), message, eventShift);
    return {pipeValue, message};
  }
  if constexpr (std::is_same_v<SyncOp, pto::SyncWaitOp>) {
    calleeName = op.getEventIdAttr() ? StringAttr::get(op.getContext(), "llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM").getValue()
                                     : StringAttr::get(op.getContext(), "llvm.hivm.WAIT.FLAG.DEV.PIPE.REG").getValue();
  }
  return {pipeValue, eventValue};
}

template <typename SyncOp> class LowerInterCoreSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerInterCoreSyncOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe) {
      return rewriter.notifyMatchFailure(op, "unsupported inter-core sync pipe");
    }

    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    FailureOr<Value> eventValue = getInterCoreEventValue(op, adaptor, rewriter);
    if (failed(eventValue)) {
      return rewriter.notifyMatchFailure(op, "expected a valid static or dynamic event-id operand");
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    SmallVector<Value> args = buildInterCoreSyncArgs(op, pipeValue, *eventValue, calleeName, rewriter);
    auto funcType = rewriter.getFunctionType(TypeRange{args[0].getType(), args[1].getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp> class LowerNamedSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerNamedSyncOpPattern(TypeConverter &tc, MLIRContext *ctx, LoweringState &state)
      : OpConversionPattern<SyncOp>(tc, ctx), state(state) {}
  LogicalResult matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
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
      eventValue = castIntegerLikeTo(op, adaptor.getEventIdDyn(), rewriter.getI64Type());
      if (!eventValue) {
        return rewriter.notifyMatchFailure(op, "missing event-id operand");
      }
    }
    StringRef callee = buildSyncCallee<SyncOp>(op.getContext());
    auto fnTy = rewriter.getFunctionType(TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), callee, TypeRange{}, ValueRange{pipeValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{callee.str(), fnTy});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerBarrierOpPattern final : public OpConversionPattern<pto::BarrierOp> {
public:
  explicit LowerBarrierOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::BarrierOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (isTargetArchA5(op.getOperation()) && op.getPipe().getPipe() == PIPE::PIPE_V) {
      op.emitError("internal error: A5 PIPE_V barrier should be erased before "
                   "VPTO LLVM lowering");
      return failure();
    }

    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe) {
      return rewriter.notifyMatchFailure(op, "unsupported barrier pipe");
    }

    StringRef calleeName = buildSyncCallee<pto::BarrierOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{pipeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerMemBarOpPattern final : public OpConversionPattern<pto::MemBarOp> {
public:
  explicit LowerMemBarOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MemBarOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::MemBarOp op, OpAdaptor adaptor,
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
class LowerUnsupportedMemoryConsistencyOpPattern final : public OpConversionPattern<MemoryConsistencyOp> {
public:
  explicit LowerUnsupportedMemoryConsistencyOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                                                      LoweringState &state)
      : OpConversionPattern<MemoryConsistencyOp>(typeConverter, context) {
    (void)state;
  }

  LogicalResult matchAndRewrite(MemoryConsistencyOp op, typename MemoryConsistencyOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    (void)rewriter;
    op.emitOpError() << "is not supported by the VPTO backend yet; PTOAS validates the "
                        "memory-consistency contract, but high-level CMO/fence ops must be "
                        "lowered to `pto.dcci` or `pto.dsb` before VPTO LLVM lowering";
    return failure();
  }
};

class LowerDsbOpPattern final : public OpConversionPattern<pto::DsbOp> {
public:
  explicit LowerDsbOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::DsbOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::DsbOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName = StringAttr::get(op.getContext(), "llvm.hivm.DSB").getValue();
    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(TypeRange{i64Ty}, TypeRange{});
    Value mem = getI64Constant(rewriter, op.getLoc(), getDsbMemImmediate(op.getMem().getKind()));
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{mem});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerDcciOpPattern final : public OpConversionPattern<pto::DcciOp> {
public:
  explicit LowerDcciOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::DcciOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::DcciOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!ptrType) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");
    }

    bool hasDst = static_cast<bool>(op.getDstAttr());
    StringRef calleeName = buildDcciCallee(ptrType.getAddressSpace(), hasDst, op.getContext());

    Type i64Ty = rewriter.getI64Type();
    SmallVector<Type> argTypes{ptrType, i64Ty};
    SmallVector<Value> args{adaptor.getPtr(),
                            getI64Constant(rewriter, op.getLoc(), getDcciCacheLineImmediate(op.getCache().getKind()))};
    if (auto dst = op.getDstAttr()) {
      argTypes.push_back(i64Ty);
      args.push_back(getI64Constant(rewriter, op.getLoc(), getDcciDstImmediate(dst.getKind())));
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

template <typename BufSyncOp> class LowerBufSyncOpPattern final : public OpConversionPattern<BufSyncOp> {
public:
  explicit LowerBufSyncOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BufSyncOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(BufSyncOp op, typename BufSyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    if (auto pipeAttr = dyn_cast<PipeAttr>(op.getOpTypeAttr())) {
      pipe = pipeAttr.getPipe();
    } else {
      auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
      if (failed(opTypeOr)) {
        return rewriter.notifyMatchFailure(op, "buffer sync expects pipe/sync_op_type/pipe_event_type attr");
      }
      pipe = mapSyncOpTypeToPipe(*opTypeOr);
    }
    if (!isConcreteSyncPipe(pipe)) {
      return rewriter.notifyMatchFailure(op, "buffer sync op_type cannot map to concrete pipe");
    }

    auto pipeImm = parsePipeImmediate(stringifyPIPE(pipe));
    if (!pipeImm) {
      return rewriter.notifyMatchFailure(op, "unsupported buffer sync pipe");
    }

    StringRef calleeName = buildSyncCallee<BufSyncOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipeImm);
    Value bufIdValue = getI64Constant(rewriter, op.getLoc(), op.getBufIdAttr().getInt());
    Value modeValue = getI64Constant(rewriter, op.getLoc(), op.getModeAttr().getInt());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{pipeValue, bufIdValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BufDynSyncOp> class LowerBufDynSyncOpPattern final : public OpConversionPattern<BufDynSyncOp> {
public:
  explicit LowerBufDynSyncOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BufDynSyncOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(BufDynSyncOp op, typename BufDynSyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    if (auto pipeAttr = dyn_cast<PipeAttr>(op.getOpTypeAttr())) {
      pipe = pipeAttr.getPipe();
    } else {
      auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
      if (failed(opTypeOr)) {
        return rewriter.notifyMatchFailure(op, "buffer sync expects pipe/sync_op_type/pipe_event_type attr");
      }
      pipe = mapSyncOpTypeToPipe(*opTypeOr);
    }
    if (!isConcreteSyncPipe(pipe)) {
      return rewriter.notifyMatchFailure(op, "buffer sync op_type cannot map to concrete pipe");
    }

    auto pipeImm = parsePipeImmediate(stringifyPIPE(pipe));
    if (!pipeImm) {
      return rewriter.notifyMatchFailure(op, "unsupported buffer sync pipe");
    }

    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipeImm);
    Value bufIdDyn = adaptor.getBufId();
    if (!bufIdDyn) {
      return rewriter.notifyMatchFailure(op, "expected dynamic buf-id operand");
    }
    Value bufIdValue = castIntegerLikeTo(op, bufIdDyn, rewriter.getI64Type());
    if (!bufIdValue) {
      return rewriter.notifyMatchFailure(op, "failed to cast dynamic buf-id to i64");
    }

    bool isGetBuf = std::is_same_v<BufDynSyncOp, pto::GetBufDynOp>;
    StringRef calleeName = buildBufDynSyncCallee(op.getContext(), isGetBuf);
    Value modeValue = getI64Constant(rewriter, op.getLoc(), op.getModeAttr().getInt());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{pipeValue, bufIdValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename QueryOp> class LowerRuntimeQueryOpPattern final : public OpConversionPattern<QueryOp> {
public:
  explicit LowerRuntimeQueryOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<QueryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert runtime-query result type");
    }

    StringRef calleeName = buildRuntimeQueryCallee<QueryOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename QueryOp> class LowerBlockRuntimeQueryOpPattern final : public OpConversionPattern<QueryOp> {
public:
  explicit LowerBlockRuntimeQueryOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<QueryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert block runtime-query result type");
    }

    auto funcOp = op->template getParentOfType<func::FuncOp>();
    bool isSimtEntry = funcOp && funcOp->hasAttr(pto::kPTOSimtEntryAttrName);
    if (isSimtEntry && !resultType.isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "SIMT block runtime-query expects an i64 PTO result");
    }

    StringRef calleeName = isSimtEntry ? buildSimtBlockQueryCallee<QueryOp>(op.getContext())
                                       : buildRuntimeQueryCallee<QueryOp>(op.getContext());
    Type callResultType = isSimtEntry ? rewriter.getI32Type() : resultType;
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{callResultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{callResultType}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});

    Value result = call.getResult(0);
    if (isSimtEntry) {
      result = rewriter.create<arith::ExtUIOp>(op.getLoc(), resultType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

template <typename VoteOp> class LowerVoteOpPattern final : public OpConversionPattern<VoteOp> {
public:
  explicit LowerVoteOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<VoteOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(VoteOp op, typename VoteOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert vote result type");
    }

    Type predType = this->getTypeConverter()->convertType(op.getPred().getType());
    if (!predType || predType != rewriter.getI1Type()) {
      return rewriter.notifyMatchFailure(op, "failed to convert vote predicate type");
    }

    StringRef calleeName = buildVoteCallee<VoteOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{predType}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{adaptor.getPred()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ShuffleOp> class LowerShuffleOpPattern final : public OpConversionPattern<ShuffleOp> {
public:
  explicit LowerShuffleOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ShuffleOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ShuffleOp op, typename ShuffleOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert shuffle result type");
    }

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!valueType || valueType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted shuffle operand type");
    }

    FailureOr<StringRef> calleeName = buildShuffleCallee<ShuffleOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported shuffle VPTO signature");
    }

    IntegerAttr widthAttr = op.getWidthAttr();
    Value controlValue;
    unsigned controlMask = 0;
    if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleIdxOp>) {
      controlValue = adaptor.getIndex();
      controlMask = 0x1f;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleUpOp>) {
      controlValue = adaptor.getOffset();
      controlMask = 0;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleDownOp>) {
      controlValue = adaptor.getOffset();
      controlMask = 0x1f;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleBflyOp>) {
      controlValue = adaptor.getMask();
      controlMask = 0x1f;
    }
    if (!controlValue) {
      return rewriter.notifyMatchFailure(op, "missing shuffle control operand");
    }

    Value control = buildShuffleControlValue(rewriter, op.getLoc(), controlValue, widthAttr.getInt(), controlMask);

    Type i32Type = rewriter.getI32Type();
    auto funcType = rewriter.getFunctionType(TypeRange{resultType, i32Type}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getValue(), control});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReduxOp> class LowerReduxOpPattern final : public OpConversionPattern<ReduxOp> {
public:
  explicit LowerReduxOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ReduxOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ReduxOp op, typename ReduxOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert redux result type");
    }

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!valueType || valueType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected converted redux operand type");
    }

    FailureOr<StringRef> calleeName =
        buildReduxCallee<ReduxOp>(op.getContext(), op.getValue().getType(), op.getSignednessAttr());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported redux VPTO signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{resultType}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename AtomicOp> class LowerAtomicBinaryOpPattern final : public OpConversionPattern<AtomicOp> {
public:
  explicit LowerAtomicBinaryOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<AtomicOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(AtomicOp op, typename AtomicOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOld().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || resultType != valueType) {
      return rewriter.notifyMatchFailure(op, "unexpected atomic operand/result type");
    }

    Type ptrType = this->getTypeConverter()->convertType(op.getPtr().getType());
    if (!ptrType) {
      return rewriter.notifyMatchFailure(op, "failed to convert atomic pointer type");
    }

    FailureOr<StringRef> calleeName = buildAtomicCallee<AtomicOp>(op.getContext(), op.getPtr().getType(),
                                                                  op.getValue().getType(), op.getSignednessAttr());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported atomic VPTO signature");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{ptrType, valueType, rewriter.getI32Type()}, TypeRange{resultType});
    Value modeValue = getI32Constant(
        rewriter, op.getLoc(),
        static_cast<uint64_t>(op.getL2cacheAttr() ? op.getL2cacheAttr().getValue() : pto::StL2Cache::NMFV));
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getPtr(), adaptor.getValue(), modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerAtomicCasOpPattern final : public OpConversionPattern<pto::AtomicCasOp> {
public:
  explicit LowerAtomicCasOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::AtomicCasOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::AtomicCasOp op, pto::AtomicCasOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOld().getType());
    Type compareType = this->getTypeConverter()->convertType(op.getCompare().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !compareType || !valueType || resultType != compareType || resultType != valueType) {
      return rewriter.notifyMatchFailure(op, "unexpected atomic CAS type");
    }

    Type ptrType = this->getTypeConverter()->convertType(op.getPtr().getType());
    if (!ptrType) {
      return rewriter.notifyMatchFailure(op, "failed to convert atomic pointer type");
    }

    FailureOr<StringRef> calleeName = buildAtomicCallee<pto::AtomicCasOp>(
        op.getContext(), op.getPtr().getType(), op.getValue().getType(), op.getSignednessAttr());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported atomic CAS signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{ptrType, compareType, valueType, rewriter.getI32Type()},
                                             TypeRange{resultType});
    Value modeValue = getI32Constant(
        rewriter, op.getLoc(),
        static_cast<uint64_t>(op.getL2cacheAttr() ? op.getL2cacheAttr().getValue() : pto::StL2Cache::NMFV));
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getPtr(), adaptor.getCompare(), adaptor.getValue(), modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ScalarOp> class LowerScalarIntrinsicOpPattern final : public OpConversionPattern<ScalarOp> {
public:
  explicit LowerScalarIntrinsicOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ScalarOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(ScalarOp op, typename ScalarOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert scalar result types");
    }

    SmallVector<Type> operandTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getOperandTypes(), operandTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert scalar operand types");
    }

    StringRef calleeName = buildScalarIntrinsicCallee<ScalarOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(operandTypes, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes, adaptor.getOperands());
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerMulhiOpPattern final : public OpConversionPattern<pto::MulhiOp> {
public:
  explicit LowerMulhiOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MulhiOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::MulhiOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType || lhsType != resultType || rhsType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected mulhi type");
    }

    pto::Signedness signedness = op.getSignednessAttr().getValue();
    FailureOr<StringRef> calleeName = buildMulhiCallee(op.getContext(), op.getResult().getType(), signedness);
    if (succeeded(calleeName)) {
      auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
      auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                                ValueRange{adaptor.getLhs(), adaptor.getRhs()});
      state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (!op.getResult().getType().isInteger(64) || signedness != pto::Signedness::Signed) {
      return rewriter.notifyMatchFailure(op, "unsupported mulhi signature");
    }

    FailureOr<StringRef> unsignedCalleeName =
        buildMulhiCallee(op.getContext(), op.getResult().getType(), pto::Signedness::Unsigned);
    if (failed(unsignedCalleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported mul64hi signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto unsignedCall = rewriter.create<func::CallOp>(op.getLoc(), *unsignedCalleeName, TypeRange{resultType},
                                                      ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{unsignedCalleeName->str(), funcType});

    Value zero = getI64Constant(rewriter, op.getLoc(), 0);
    Value lhsNeg = rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::slt, adaptor.getLhs(), zero);
    Value rhsNeg = rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::slt, adaptor.getRhs(), zero);
    Value subRhs = rewriter.create<LLVM::SubOp>(op.getLoc(), unsignedCall.getResult(0), adaptor.getRhs());
    Value correctedLhs =
        rewriter.create<LLVM::SelectOp>(op.getLoc(), resultType, lhsNeg, subRhs, unsignedCall.getResult(0));
    Value subLhs = rewriter.create<LLVM::SubOp>(op.getLoc(), correctedLhs, adaptor.getLhs());
    Value corrected = rewriter.create<LLVM::SelectOp>(op.getLoc(), resultType, rhsNeg, subLhs, correctedLhs);
    rewriter.replaceOp(op, corrected);
    return success();
  }

private:
  LoweringState &state;
};

class LowerMulI32ToI64OpPattern final : public OpConversionPattern<pto::MulI32ToI64Op> {
public:
  explicit LowerMulI32ToI64OpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MulI32ToI64Op>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::MulI32ToI64Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(op, "unexpected mul_i32toi64 type");
    }

    FailureOr<StringRef> calleeName = buildMulI32ToI64Callee(op.getContext(), op.getSignednessAttr().getValue());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported mul_i32toi64 signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerSqrtOpPattern final : public OpConversionPattern<pto::SqrtOp> {
public:
  explicit LowerSqrtOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::SqrtOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::SqrtOp op, pto::SqrtOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || valueType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected sqrt operand/result type");
    }

    FailureOr<StringRef> calleeName = buildSqrtCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported sqrt VPTO signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{valueType}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnaryOp> class LowerUnaryScalarMathOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryScalarMathOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || valueType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected unary scalar math type");
    }

    FailureOr<StringRef> calleeName = buildUnaryScalarMathCallee<UnaryOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported unary scalar math signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{valueType}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp> class LowerBinaryScalarMathOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryScalarMathOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = this->getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType || lhsType != rhsType || lhsType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected binary scalar math type");
    }

    FailureOr<StringRef> calleeName = buildBinaryScalarMathCallee<BinaryOp>(op.getContext(), op.getLhs().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported binary scalar math signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerFmaOpPattern final : public OpConversionPattern<pto::FmaOp> {
public:
  explicit LowerFmaOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::FmaOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::FmaOp op, pto::FmaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = this->getTypeConverter()->convertType(op.getRhs().getType());
    Type accType = this->getTypeConverter()->convertType(op.getAcc().getType());
    if (!resultType || !lhsType || !rhsType || !accType || lhsType != rhsType || lhsType != accType ||
        lhsType != resultType) {
      return rewriter.notifyMatchFailure(op, "unexpected fma scalar math type");
    }

    FailureOr<StringRef> calleeName = buildFmaCallee(op.getContext(), op.getLhs().getType());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported fma scalar signature");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType, accType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerConvertOpPattern final : public OpConversionPattern<pto::ConvertOp> {
public:
  explicit LowerConvertOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::ConvertOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::ConvertOp op, pto::ConvertOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getDst().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    FailureOr<StringRef> calleeName =
        buildConvertCallee(op.getContext(), op.getSrc().getType(), op.getDst().getType(), op.getSignednessAttr());
    if (failed(calleeName)) {
      return rewriter.notifyMatchFailure(op, "unsupported convert signature");
    }

    Value rounding = getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(op.getRounding()));
    Value saturation = getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(op.getSaturation()));

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSrc().getType(), rewriter.getI32Type(), rewriter.getI32Type()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getSrc(), rounding, saturation});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerGetVms4SrOpPattern final : public OpConversionPattern<pto::GetVms4SrOp> {
public:
  explicit LowerGetVms4SrOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::GetVms4SrOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::GetVms4SrOp op, pto::GetVms4SrOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) || resultTypes.size() != 4) {
      return rewriter.notifyMatchFailure(op, "failed to convert get_vms4_sr result types");
    }

    StringRef calleeName = buildRuntimeQueryCallee<pto::GetVms4SrOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{rewriter.getI64Type()});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{rewriter.getI64Type()}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});

    SmallVector<Value> counts;
    counts.reserve(4);
    Value raw = call.getResult(0);
    for (unsigned i = 0; i < 4; ++i) {
      Value shifted = raw;
      if (i != 0) {
        shifted = rewriter.create<arith::ShRUIOp>(op.getLoc(), raw, getI64Constant(rewriter, op.getLoc(), i * 16));
      }
      counts.push_back(rewriter.create<arith::TruncIOp>(op.getLoc(), resultTypes[i], shifted));
    }
    rewriter.replaceOp(op, counts);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp> class LowerBinaryI64PureOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryI64PureOpPattern(TypeConverter &typeConverter, MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    StringRef calleeName = buildBinaryI64PureCallee<BinaryOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{adaptor.getFirst().getType(), adaptor.getSecond().getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType},
                                              ValueRange{adaptor.getFirst(), adaptor.getSecond()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

static void populateVPTOSIMTAndScalarPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                              LoweringState &state) {
  patterns.add<
      LowerRuntimeQueryOpPattern<pto::GetCtrlOp>, LowerGetVms4SrOpPattern, LowerRuntimeQueryOpPattern<pto::GetTidXOp>,
      LowerRuntimeQueryOpPattern<pto::GetTidYOp>, LowerRuntimeQueryOpPattern<pto::GetTidZOp>,
      LowerRuntimeQueryOpPattern<pto::GetBlockDimXOp>, LowerRuntimeQueryOpPattern<pto::GetBlockDimYOp>,
      LowerRuntimeQueryOpPattern<pto::GetBlockDimZOp>, LowerRuntimeQueryOpPattern<pto::GetGridDimXOp>,
      LowerRuntimeQueryOpPattern<pto::GetGridDimYOp>, LowerRuntimeQueryOpPattern<pto::GetGridDimZOp>,
      LowerRuntimeQueryOpPattern<pto::GetBlockIdxXOp>, LowerRuntimeQueryOpPattern<pto::GetBlockIdxYOp>,
      LowerRuntimeQueryOpPattern<pto::GetBlockIdxZOp>, LowerRuntimeQueryOpPattern<pto::GetVecCoreIdOp>,
      LowerRuntimeQueryOpPattern<pto::GetLaneIdOp>, LowerRuntimeQueryOpPattern<pto::GetClock32Op>,
      LowerRuntimeQueryOpPattern<pto::GetClock64Op>, LowerRuntimeQueryOpPattern<pto::GetLaneMaskEqOp>,
      LowerRuntimeQueryOpPattern<pto::GetLaneMaskLeOp>, LowerRuntimeQueryOpPattern<pto::GetLaneMaskLtOp>,
      LowerRuntimeQueryOpPattern<pto::GetLaneMaskGeOp>, LowerRuntimeQueryOpPattern<pto::GetLaneMaskGtOp>,
      LowerVoteOpPattern<pto::VoteAllOp>, LowerVoteOpPattern<pto::VoteAnyOp>, LowerVoteOpPattern<pto::VoteUniOp>,
      LowerVoteOpPattern<pto::VoteBallotOp>, LowerShuffleOpPattern<pto::ShuffleIdxOp>,
      LowerShuffleOpPattern<pto::ShuffleUpOp>, LowerShuffleOpPattern<pto::ShuffleDownOp>,
      LowerShuffleOpPattern<pto::ShuffleBflyOp>, LowerReduxOpPattern<pto::ReduxAddOp>,
      LowerReduxOpPattern<pto::ReduxMaxOp>, LowerReduxOpPattern<pto::ReduxMinOp>, LowerAtomicCasOpPattern,
      LowerAtomicBinaryOpPattern<pto::AtomicExchOp>, LowerAtomicBinaryOpPattern<pto::AtomicAddOp>,
      LowerAtomicBinaryOpPattern<pto::AtomicSubOp>, LowerAtomicBinaryOpPattern<pto::AtomicMinOp>,
      LowerAtomicBinaryOpPattern<pto::AtomicMaxOp>, LowerAtomicBinaryOpPattern<pto::AtomicAndOp>,
      LowerAtomicBinaryOpPattern<pto::AtomicOrOp>, LowerAtomicBinaryOpPattern<pto::AtomicXorOp>, LowerTrapOpPattern,
      LowerScalarIntrinsicOpPattern<pto::PrmtOp>, LowerMulhiOpPattern, LowerMulI32ToI64OpPattern, LowerSqrtOpPattern,
      LowerUnaryScalarMathOpPattern<pto::AbsFOp>, LowerUnaryScalarMathOpPattern<pto::ExpOp>,
      LowerUnaryScalarMathOpPattern<pto::LogOp>, LowerUnaryScalarMathOpPattern<pto::CeilOp>,
      LowerUnaryScalarMathOpPattern<pto::FloorOp>, LowerUnaryScalarMathOpPattern<pto::RintOp>,
      LowerUnaryScalarMathOpPattern<pto::RoundOp>, LowerBinaryScalarMathOpPattern<pto::FMinOp>,
      LowerBinaryScalarMathOpPattern<pto::FMaxOp>, LowerBinaryScalarMathOpPattern<pto::PowOp>, LowerFmaOpPattern,
      LowerConvertOpPattern, LowerSimtFenceOpPattern<pto::SyncthreadsOp>, LowerSimtFenceOpPattern<pto::ThreadfenceOp>,
      LowerSimtFenceOpPattern<pto::ThreadfenceBlockOp>, LowerKeepOpPattern, LowerResumeOpPattern,
      LowerBinaryI64PureOpPattern<pto::Sbitset0Op>, LowerBinaryI64PureOpPattern<pto::Sbitset1Op>,
      LowerSetLoopConfigOpPattern<pto::SetLoop2StrideOutToUbOp>,
      LowerSetLoopConfigOpPattern<pto::SetLoop1StrideOutToUbOp>, LowerSetLoopConfigOpPattern<pto::SetLoopSizeOutToUbOp>,
      LowerSetLoopConfigOpPattern<pto::SetLoop2StrideUbToOutOp>,
      LowerSetLoopConfigOpPattern<pto::SetLoop1StrideUbToOutOp>, LowerSetLoopConfigOpPattern<pto::SetLoopSizeUbToOutOp>,
      LowerSetLoopConfigOpPattern<pto::SetLoop3ParaOp>, LowerSetLoopConfigOpPattern<pto::SetChannelParaOp>,
      LowerUnaryI64ConfigOpPattern<pto::SetCtrlOp>, LowerStoreVfSimtInfoOpPattern>(typeConverter, patterns.getContext(),
                                                                                   state);
}

static void populateVPTOConfigAndSyncPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                              LoweringState &state) {
  patterns
      .add<LowerUnaryConfigOpPattern<pto::SetMovPadValOp>, LowerUnaryI64ConfigOpPattern<pto::SetQuantPreOp>,
           LowerUnaryI64ConfigOpPattern<pto::SetReluAlphaOp>, LowerUnaryI64ConfigOpPattern<pto::SetFixClipReluOp>,
           LowerUnaryI64ConfigOpPattern<pto::SetLoop2StrideOutToL1Op>,
           LowerUnaryI64ConfigOpPattern<pto::SetLoop1StrideOutToL1Op>,
           LowerUnaryI64ConfigOpPattern<pto::SetLoopSizeOutToL1Op>, LowerUnaryI64ConfigOpPattern<pto::SetMte2NzParaOp>,
           LowerUnaryI64ConfigOpPattern<pto::SetPadValOutToL1Op>, LowerUnaryI64ConfigOpPattern<pto::SetFpcOp>,
           LowerUnaryI64ConfigOpPattern<pto::SetStoreAtomicCfgOp>, LowerNullaryConfigOpPattern<pto::SetAtomicS32Op>,
           LowerNullaryConfigOpPattern<pto::SetAtomicS8Op>, LowerPipeEventSyncOpPattern<pto::SetFlagOp>,
           LowerPipeEventSyncOpPattern<pto::WaitFlagOp>, LowerPipeEventDynSyncOpPattern<pto::SetFlagDynOp>,
           LowerPipeEventDynSyncOpPattern<pto::WaitFlagDynOp>, LowerBarrierOpPattern, LowerMemBarOpPattern,
           LowerUnsupportedMemoryConsistencyOpPattern<pto::CmoCacheInvalidOp>,
           LowerUnsupportedMemoryConsistencyOpPattern<pto::FenceBarrierAllOp>, LowerDsbOpPattern, LowerDcciOpPattern,
           LowerBufSyncOpPattern<pto::GetBufOp>, LowerBufSyncOpPattern<pto::RlsBufOp>,
           LowerBufDynSyncOpPattern<pto::GetBufDynOp>, LowerBufDynSyncOpPattern<pto::RlsBufDynOp>,
           LowerBlockRuntimeQueryOpPattern<pto::GetBlockIdxOp>, LowerRuntimeQueryOpPattern<pto::GetSubBlockIdxOp>,
           LowerBlockRuntimeQueryOpPattern<pto::GetBlockNumOp>, LowerRuntimeQueryOpPattern<pto::GetSubBlockNumOp>,
           LowerVtrcOpPattern, LowerVcvtOpPattern, LowerVbitcastOpPattern, LowerPbitcastOpPattern,
           LowerPredicateLoadOpPattern<pto::PldiOp>, LowerPredicateLoadOpPattern<pto::PldsOp>,
           LowerPredicateStoreOpPattern<pto::PstiOp>, LowerPredicateStoreOpPattern<pto::PstsOp>,
           LowerInterCoreSyncOpPattern<pto::SyncSetOp>, LowerInterCoreSyncOpPattern<pto::SyncWaitOp>,
           LowerNamedSyncOpPattern<pto::SetIntraBlockOp>, LowerNamedSyncOpPattern<pto::WaitIntraBlockOp>>(
          typeConverter, patterns.getContext(), state);
}

void populateVPTOScalarPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns, LoweringState &state) {
  populateVPTOSIMTAndScalarPatterns(typeConverter, patterns, state);
  populateVPTOConfigAndSyncPatterns(typeConverter, patterns, state);
}

} // namespace mlir::pto::detail
