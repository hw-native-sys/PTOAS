// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
namespace {

Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  if (value.getType() == targetType) return value;
  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt)
    return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth())
        return builder.create<arith::ExtUIOp>(anchor->getLoc(), targetType, value);
      if (sourceInt.getWidth() > targetInt.getWidth())
        return builder.create<arith::TruncIOp>(anchor->getLoc(), targetType, value);
      return value;
    }
    if (targetType.isIndex())
      return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  }
  return {};
}

class LowerUBSetMaskOpPattern final
    : public OpConversionPattern<pto::UBSetMaskOp> {
public:
  explicit LowerUBSetMaskOpPattern(TypeConverter &converter, MLIRContext *context,
                                   detail::LoweringState &state)
      : OpConversionPattern<pto::UBSetMaskOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(
      pto::UBSetMaskOp op, pto::UBSetMaskOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = "llvm.hivm.MOVEMASK";
    Location loc = op.getLoc();
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    Value c0 = rewriter.create<arith::ConstantOp>(loc,
                                                   rewriter.getI64IntegerAttr(0));
    Value c1 = rewriter.create<arith::ConstantOp>(loc,
                                                   rewriter.getI64IntegerAttr(1));
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{c0, adaptor.getMask0()});
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{c1, adaptor.getMask1()});
    state.plannedDecls.push_back(detail::PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }
private:
  detail::LoweringState &state;
};

class LowerUBSetMaskCountOpPattern final
    : public OpConversionPattern<pto::UBSetMaskCountOp> {
public:
  explicit LowerUBSetMaskCountOpPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<pto::UBSetMaskCountOp>(converter, context) {}
  LogicalResult matchAndRewrite(
      pto::UBSetMaskCountOp op, pto::UBSetMaskCountOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    Value ctrl = rewriter.create<pto::GetCtrlOp>(loc, i64);
    Value bit = rewriter.create<arith::ConstantOp>(loc,
                                                   rewriter.getI64IntegerAttr(56));
    Value set = rewriter.create<pto::Sbitset1Op>(loc, i64, ctrl, bit);
    rewriter.create<pto::SetCtrlOp>(loc, set);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerUBSetMaskNormOpPattern final
    : public OpConversionPattern<pto::UBSetMaskNormOp> {
public:
  explicit LowerUBSetMaskNormOpPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<pto::UBSetMaskNormOp>(converter, context) {}
  LogicalResult matchAndRewrite(
      pto::UBSetMaskNormOp op, pto::UBSetMaskNormOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    Value ctrl = rewriter.create<pto::GetCtrlOp>(loc, i64);
    Value bit = rewriter.create<arith::ConstantOp>(loc,
                                                   rewriter.getI64IntegerAttr(56));
    Value reset = rewriter.create<pto::Sbitset0Op>(loc, i64, ctrl, bit);
    rewriter.create<pto::SetCtrlOp>(loc, reset);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerUBufVdupPattern final : public OpConversionPattern<pto::UBVdupOp> {
public:
  explicit LowerUBufVdupPattern(TypeConverter &converter, MLIRContext *context,
                                detail::LoweringState &state)
      : OpConversionPattern<pto::UBVdupOp>(converter, context), state(state) {}
  LogicalResult matchAndRewrite(pto::UBVdupOp op, pto::UBVdupOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type elem = cast<pto::PtrType>(op.getDst().getType()).getElementType();
    StringRef suffix;
    if (elem.isF32() || elem.isInteger(32)) suffix = "u32";
    else if (elem.isF16() || elem.isInteger(16)) suffix = "u16";
    else return rewriter.notifyMatchFailure(op, "unsupported element type for ubuf vdup");
    Value dst = adaptor.getDst();
    if (!dst || !isa<LLVM::LLVMPointerType>(dst.getType()))
      return rewriter.notifyMatchFailure(op, "unexpected converted ubuf vdup dst type");
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    auto getI64 = [&](Value value) { return castIntegerLikeTo(op, value, i64); };
    auto byte = [&](Value value) {
      return rewriter.create<arith::AndIOp>(
          loc, value, rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0xff)));
    };
    auto shift = [&](Value value, uint64_t amount) {
      return rewriter.create<arith::ShLIOp>(
          loc, value, rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(amount)));
    };
    Value config = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    config = rewriter.create<arith::OrIOp>(loc, config, shift(byte(getI64(adaptor.getRepeat())), 56));
    config = rewriter.create<arith::OrIOp>(loc, config, byte(getI64(adaptor.getDstBlockStride())));
    config = rewriter.create<arith::OrIOp>(loc, config, shift(byte(getI64(adaptor.getSrcBlockStride())), 16));
    config = rewriter.create<arith::OrIOp>(loc, config, shift(byte(getI64(adaptor.getDstRepeatStride())), 32));
    config = rewriter.create<arith::OrIOp>(loc, config, shift(byte(getI64(adaptor.getSrcRepeatStride())), 40));
    Value scalar = getI64(adaptor.getScalar());
    std::string callee = "llvm.hivm.MOVEV." + suffix.str();
    auto functionType = rewriter.getFunctionType(TypeRange{dst.getType(), i64, i64}, TypeRange{});
    rewriter.create<func::CallOp>(loc, callee, TypeRange{}, ValueRange{dst, scalar, config});
    state.plannedDecls.push_back(detail::PlannedDecl{callee, functionType});
    rewriter.eraseOp(op);
    return success();
  }
private:
  detail::LoweringState &state;
};

} // namespace

void populateVPTOMemoryMaskPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    detail::LoweringState &state) {
  patterns.add<LowerUBSetMaskOpPattern>(typeConverter, patterns.getContext(), state);
  patterns.add<LowerUBSetMaskCountOpPattern, LowerUBSetMaskNormOpPattern>(
      typeConverter, patterns.getContext());
}

void populateVPTOMemoryUbufPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    detail::LoweringState &state) {
  patterns.add<LowerUBufVdupPattern>(typeConverter, patterns.getContext(), state);
}

} // namespace mlir::pto
