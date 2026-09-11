// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- ArithMisc.cpp - Arith ArithMisc op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ArithInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct ArithNegFToEmitC : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, dstTy, adaptor.getOperand());
    return success();
  }
};

struct ArithRemFToEmitC : public OpConversionPattern<arith::RemFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Use builtin `fmod` when possible. For f16, compute in float and cast back.
    Type callTy = dstTy;
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF16()) {
        auto f32Ty = emitc::OpaqueType::get(rewriter.getContext(), "float");
        lhs = emitCCast(rewriter, loc, f32Ty, lhs);
        rhs = emitCCast(rewriter, loc, f32Ty, rhs);
        callTy = f32Ty;
      }
    }

    // Prefer `__builtin_fmod*` to avoid relying on extra headers.
    llvm::StringRef callee = "__builtin_fmod";
    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF32() || opFloatTy.isF16())
        callee = "__builtin_fmodf";
      else if (opFloatTy.isF64())
        callee = "__builtin_fmod";
    }

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{callTy}, callee, ValueRange{lhs, rhs},
        /*args=*/ArrayAttr{}, /*template_args=*/ArrayAttr{});
    Value result = call.getResult(0);
    if (callTy != dstTy)
      result = emitCCast(rewriter, loc, dstTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithSelectToEmitC : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "only scalar i1 conditions supported for arith.select");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto cond =
        rewriter.create<emitc::ConditionalOp>(op.getLoc(), dstTy,
                                              adaptor.getCondition(),
                                              adaptor.getTrueValue(),
                                              adaptor.getFalseValue());
    rewriter.replaceOp(op, cond.getResult());
    return success();
  }
};

struct ArithAddUIExtendedToEmitC
    : public OpConversionPattern<arith::AddUIExtendedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getSum().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type sumDstTy = newResultTypes[0];
    Type overflowDstTy = newResultTypes[1];

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    auto wideTy = getWiderUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);

    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
    Value rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    Value sumWide =
        rewriter.create<emitc::AddOp>(loc, wideTy, lhsWide, rhsWide).getResult();

    Value sumN = emitCCast(rewriter, loc, uTy, sumWide);
    Value sum = emitCCast(rewriter, loc, sumDstTy, sumN);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value high = rewriter
                     .create<emitc::BitwiseRightShiftOp>(loc, wideTy, sumWide,
                                                         shiftAmt)
                     .getResult();
    Value zeroWide = makeEmitCIntConstant(rewriter, loc, wideTy, 0);
    Value overflow =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne, high, zeroWide)
            .getResult();
    overflow = emitCCast(rewriter, loc, overflowDstTy, overflow);

    rewriter.replaceOp(op, {sum, overflow});
    return success();
  }
};

template <typename ArithOp, bool isUnsigned>
struct ArithMulExtendedToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getResult(0).getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type lowDstTy = newResultTypes[0];
    Type highDstTy = newResultTypes[1];

    Type wideTy = isUnsigned ? static_cast<Type>(getWiderUnsignedIntOpaqueType(rewriter.getContext(),
                                                                               bitWidth))
                             : static_cast<Type>(getWiderSignedIntOpaqueType(rewriter.getContext(),
                                                                             bitWidth));

    Value lhsWide;
    Value rhsWide;
    if constexpr (isUnsigned) {
      Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                      bitWidth);
      Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                      bitWidth);
      lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
      rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    } else {
      lhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getLhs());
      rhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getRhs());
    }

    Value prodWide =
        rewriter.create<emitc::MulOp>(loc, wideTy, lhsWide, rhsWide).getResult();
    Value low = emitCCast(rewriter, loc, lowDstTy, prodWide);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value highWide = rewriter
                         .create<emitc::BitwiseRightShiftOp>(loc, wideTy, prodWide,
                                                             shiftAmt)
                         .getResult();
    Value high = emitCCast(rewriter, loc, highDstTy, highWide);

    rewriter.replaceOp(op, {low, high});
    return success();
  }
};
struct ArithConstantToEmitC : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    if (!newType)
      return failure();

    // `adaptor.getValue()` may be null if attribute conversion isn't defined.
    // Use the original attribute as fallback and always cast null-safely.
    Attribute valueAttr = adaptor.getValue();
    if (!valueAttr)
      valueAttr = op.getValue();

    if (auto opaqueLiteral = buildEmitCOpaqueConstantLiteral(newType, valueAttr);
        succeeded(opaqueLiteral)) {
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), *opaqueLiteral);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto floatAttr = dyn_cast_or_null<FloatAttr>(valueAttr)) {
      SmallString<32> valStr;
      floatAttr.getValue().toString(valStr);
      llvm::StringRef s(valStr);
      // Ensure the literal parses as a floating-point constant in C/C++.
      // `APFloat::toString` may emit "1" for integral values; make it "1.0".
      const bool hasFloatMarker =
          s.contains('.') || s.contains('e') || s.contains('E') ||
          s.contains('p') || s.contains('P') || s.starts_with("0x") ||
          s.starts_with("0X") || s.starts_with("nan") ||
          s.starts_with("-nan") || s.starts_with("inf") ||
          s.starts_with("-inf");
      if (!hasFloatMarker)
        valStr.append(".0");
      // Suffix: keep `f` for f16/f32; omit for f64.
      if (!floatAttr.getType().isF64())
        valStr.append("f");
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(valueAttr)) {
      std::string valStr = std::to_string(getIntegerAttrSignedValue(intAttr));
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    return failure();
  }
};

struct PTOMGatherToMGATHER : public OpConversionPattern<pto::MGatherOp> {
  using OpConversionPattern<pto::MGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // MGATHER is a template intrinsic that accepts the concrete descriptor
    // directly, so peel any type-converter materialization bridge and feed the
    // producing value. This keeps the compile-time static-stride GlobalTensor
    // from the partition_view pattern instead of the dynamic-stride bridge,
    // whose GlobalTensor<...> C-style cast would not compile (issue #1165).
    Value mem = peelUnrealized(adaptor.getMem());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    Value memArg = mem;
    auto coalescePropAttr =
        dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
    auto gatherOobAttr =
        dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
    pto::GatherOOB gatherOob =
        gatherOobAttr ? gatherOobAttr.getValue() : pto::GatherOOB::Undefined;

    // GM -> L1 uses a partition view; GM -> UB uses a tile.
    Value idxArg = idx;

    if (!coalescePropAttr)
      return op.emitError(
          "expects mgather to specify an explicit coalesce attribute (row or "
          "elem)");

    ArrayAttr templateArgs = buildMGatherTemplateArgs(op, rewriter, gatherOob);

    // GM -> L1 Coalesce::Elem stages elements through a GM scratch buffer, passed
    // as the 4th MGATHER argument; Row and the GM -> UB path have no scratch.
    SmallVector<Value, 4> callArgs{dst, memArg, idxArg};
    if (Value scratch = adaptor.getScratch()) {
      callArgs.push_back(peelUnrealized(scratch));
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MGATHER",
        ArrayAttr{}, templateArgs,
        ValueRange(callArgs));

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, dst);
    }
    return success();
  }

  // MGATHER's template list: the mandatory Coalesce mode plus the optional
  // out-of-bounds handling mode.
  ArrayAttr buildMGatherTemplateArgs(pto::MGatherOp op,
                                     ConversionPatternRewriter &rewriter,
                                     pto::GatherOOB gatherOob) const {
    auto *ctx = rewriter.getContext();
    auto coalescePropAttr =
        dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
    auto coalesceTok = [](pto::Coalesce mode) -> StringRef {
      switch (mode) {
      case pto::Coalesce::Row:
        return "pto::Coalesce::Row";
      case pto::Coalesce::Elem:
        return "pto::Coalesce::Elem";
      }
      llvm_unreachable("unknown Coalesce");
    };
    auto gatherOobTok = [](pto::GatherOOB mode) -> StringRef {
      switch (mode) {
      case pto::GatherOOB::Undefined:
        return "pto::GatherOOB::Undefined";
      case pto::GatherOOB::Clamp:
        return "pto::GatherOOB::Clamp";
      case pto::GatherOOB::Wrap:
        return "pto::GatherOOB::Wrap";
      case pto::GatherOOB::Zero:
        return "pto::GatherOOB::Zero";
      }
      llvm_unreachable("unknown GatherOOB");
    };

    SmallVector<Attribute, 2> templateArgVec;
    templateArgVec.push_back(
        emitc::OpaqueAttr::get(ctx, coalesceTok(coalescePropAttr.getValue())));
    if (op.getGatherOob() != pto::GatherOOB::Undefined) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, gatherOobTok(gatherOob)));
    }
    return templateArgVec.empty() ? ArrayAttr{}
                                  : rewriter.getArrayAttr(templateArgVec);
  }
};

struct AffineApplyMulConstToEmitC
    : public OpConversionPattern<affine::AffineApplyOp> {
  using OpConversionPattern<affine::AffineApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();

    if (map.getNumDims() != 0 || map.getNumSymbols() != 1)
      return failure();

    auto expr = map.getResult(0);
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin || bin.getKind() != AffineExprKind::Mul)
      return failure();

    auto lhs = bin.getLHS();
    auto rhs = bin.getRHS();

    auto symExpr = dyn_cast<AffineSymbolExpr>(lhs);
    auto constExpr = dyn_cast<AffineConstantExpr>(rhs);
    if (!symExpr || !constExpr)
      return failure();

    Value inputVal = adaptor.getMapOperands()[0];

    std::string valStr = std::to_string(constExpr.getValue());
    auto cstAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
    auto cstOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), inputVal.getType(), cstAttr);

    rewriter.replaceOpWithNewOp<emitc::MulOp>(
        op, inputVal.getType(), inputVal, cstOp);

    return success();
  }
};

struct FuncToEmitC : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature with the type converter.
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    // Create the EmitC function with the converted signature.
    auto emitcFunc =
        rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), funcType);

    for (const auto &namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName().strref();
      if (name == op.getFunctionTypeAttrName() ||
          name == SymbolTable::getSymbolAttrName() ||
          name == pto::kPTOEntryAttrName ||
          name == pto::kLegacyHACCEntryAttrName)
        continue;
      emitcFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    if (op.isDeclaration()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"extern \"C\"", "AICORE"}));
      rewriter.eraseOp(op);
      return success();
    }

    applyFuncSpecifiers(op, rewriter, emitcFunc);

    // The no-split guard decision walks the function body, so it must be
    // resolved before the body is inlined into the EmitC function below.
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    if (failed(convertFuncBody(op, emitcFunc, funcType, rewriter)))
      return failure();

    emitFuncPrologueAndEpilogueMacros(op, emitcFunc, rewriter, needsNoSplitGuard);

    rewriter.eraseOp(op);
    return success();
  }

  // Inline the original body, then convert region/block argument types to
  // match the converted signature (also covers CFG blocks introduced by
  // pre-lowering, e.g. scf.while -> cf.br/cf.cond_br).
  LogicalResult convertFuncBody(func::FuncOp op, emitc::FuncOp emitcFunc,
                                FunctionType funcType,
                                ConversionPatternRewriter &rewriter) const {
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(),
                                emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));

    return rewriter.convertRegionTypes(&emitcFunc.getBody(),
                                       *getTypeConverter(), &entryConv);
  }

  // Preserve the existing function prologue shape. `kernel_kind` functions are
  // emitted with the same macro guard/reset sequence that used to come from
  // early pto.section wrapping, but only after SCF pre-lowering has finished.
  void emitFuncPrologueAndEpilogueMacros(func::FuncOp op,
                                         emitc::FuncOp emitcFunc,
                                         ConversionPatternRewriter &rewriter,
                                         bool needsNoSplitGuard) const {
    std::optional<StringRef> kernelKindMacro = getKernelKindMacro(op);

    {
      Block &entryBlock = emitcFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
      if (kernelKindMacro) {
        std::string startMacro = "\n#if defined(" + kernelKindMacro->str() + ")";
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), startMacro);
        if (*kernelKindMacro == "__DAV_VEC__") {
          rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_mask_norm();");
          rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                             "set_vector_mask(-1, -1);");
          if (needsNoSplitGuard)
            rewriter.create<emitc::VerbatimOp>(
                op.getLoc(), "if (get_subblockid() == 0) {");
        }
      }
    }

    if (kernelKindMacro) {
      Block &lastBlock = emitcFunc.getBody().back();
      rewriter.setInsertionPoint(lastBlock.getTerminator());
      if (*kernelKindMacro == "__DAV_VEC__" && needsNoSplitGuard)
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), "}");
      std::string endMacro = "#endif // " + kernelKindMacro->str() + "\n";
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), endMacro);
    }
  }
};

using ArithMulSIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulSIExtendedOp, /*isUnsigned=*/false>;
using ArithMulUIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulUIExtendedOp, /*isUnsigned=*/true>;
void populateArithArithMiscPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<ArithMulSIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  patterns.add<ArithConstantToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<AffineApplyMulConstToEmitC>(typeConverter, ctx);
  patterns.add<ArithNegFToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSelectToEmitC>(typeConverter, ctx);
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
