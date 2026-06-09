// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCKernelOps.cpp --------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

constexpr unsigned kTemplateTokenInlineCapacity = 4;
constexpr unsigned kStoreTemplateTokenInlineCapacity = 5;
constexpr unsigned kTStoreOperandInlineCapacity = 3;
using TemplateTokenVector = SmallVector<std::string, kTemplateTokenInlineCapacity>;
using StoreTemplateTokenVector =
    SmallVector<std::string, kStoreTemplateTokenInlineCapacity>;
using TStoreOperandVector = SmallVector<Value, kTStoreOperandInlineCapacity>;

static Value materializeKernelSourceArg(ConversionPatternRewriter &rewriter,
                                        Location loc, Operation *anchor,
                                        Value originalSource, Value emittedSource) {
  Value srcArg = peelUnrealized(emittedSource);
  auto srcMrTy = dyn_cast<MemRefType>(originalSource.getType());
  if (!srcMrTy)
    return srcArg;
  bool isGlobal = true;
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  if (!isGlobal)
    return srcArg;
  if (Value gt = buildGlobalTensorFromMemref(rewriter, loc, srcArg, srcMrTy, anchor))
    return gt;
  return srcArg;
}

struct PTOTLoadToTLOAD : public OpConversionPattern<pto::TLoadOp> {
  using OpConversionPattern<pto::TLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tload");

    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = materializeKernelSourceArg(rewriter, op.getLoc(),
                                              op.getOperation(), op.getSrc(),
                                              adaptor.getSrc());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TLOAD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, srcArg});
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct PTOTPrefetchToTPREFETCH : public OpConversionPattern<pto::TPrefetchOp> {
  using OpConversionPattern<pto::TPrefetchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tprefetch");

    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = materializeKernelSourceArg(rewriter, op.getLoc(),
                                              op.getOperation(), op.getSrc(),
                                              adaptor.getSrc());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPREFETCH",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, srcArg});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTPrefetchAsyncToEmitC
    : public OpConversionPattern<pto::TPrefetchAsyncOp> {
  using OpConversionPattern<pto::TPrefetchAsyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchAsyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value srcArg = src;
    if (!isEmitCGlobalTensorLikeType(srcArg.getType())) {
      auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType());
      if (!srcMrTy)
        return rewriter.notifyMatchFailure(
            op, "expected src to lower to GlobalTensor or memref");
      srcArg = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                           op.getSrc().getDefiningOp()
                                               ? op.getSrc().getDefiningOp()
                                               : op.getOperation());
    }
    if (!srcArg)
      return rewriter.notifyMatchFailure(op,
                                         "failed to build GlobalTensor src");

    Value prefetchCtx = peelUnrealized(adaptor.getCtx());

    Type eventTy = getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert tprefetch_async result type");

    Value event = rewriter
                      .create<emitc::CallOpaqueOp>(
                          op.getLoc(), TypeRange{eventTy}, "TPREFETCH_ASYNC",
                          ArrayAttr{}, ArrayAttr{},
                          ValueRange{srcArg, prefetchCtx})
                      .getResult(0);

    rewriter.replaceOp(op, ValueRange{event});
    return success();
  }
};

struct PTOMakePrefetchAsyncContextToEmitC
    : public OpConversionPattern<pto::MakePrefetchAsyncContextOp> {
  using OpConversionPattern<pto::MakePrefetchAsyncContextOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MakePrefetchAsyncContextOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type ctxTy = getTypeConverter()->convertType(op.getCtx().getType());
    if (!ctxTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert make_prefetch_async_context result type");

    Value workspace = peelUnrealized(adaptor.getWorkspace());
    workspace = castToGMBytePointer(rewriter, op.getLoc(), workspace);

    Value ctx = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), TypeRange{ctxTy}, "pto::PrefetchAsyncContext",
                        ArrayAttr{}, ArrayAttr{}, ValueRange{workspace})
                    .getResult(0);

    rewriter.replaceOp(op, ValueRange{ctx});
    return success();
  }
};

struct PTOGetPrefetchAsyncSessionToEmitC
    : public OpConversionPattern<pto::GetPrefetchAsyncSessionOp> {
  using OpConversionPattern<pto::GetPrefetchAsyncSessionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GetPrefetchAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type sessionTy = getTypeConverter()->convertType(op.getSession().getType());
    if (!sessionTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert get_prefetch_async_session result type");

    Value ctx = peelUnrealized(adaptor.getCtx());
    Value session = rewriter
                        .create<emitc::CallOpaqueOp>(
                            op.getLoc(), TypeRange{sessionTy},
                            "PTOAS__PREFETCH_CTX_SESSION", ArrayAttr{},
                            ArrayAttr{}, ValueRange{ctx})
                        .getResult(0);

    rewriter.replaceOp(op, ValueRange{session});
    return success();
  }
};

struct PTOTStoreToTSTORE : public OpConversionPattern<pto::TStoreOp> {
  using OpConversionPattern<pto::TStoreOp>::OpConversionPattern;

  static std::string stPhaseTok(pto::STPhase phase) {
    switch (phase) {
      case pto::STPhase::Unspecified: return "STPhase::Unspecified";
      case pto::STPhase::Partial: return "STPhase::Partial";
      case pto::STPhase::Final: return "STPhase::Final";
    }
    return "STPhase::Unspecified";
  }

  static std::string atomicTypeTok(pto::AtomicType atomicType) {
    switch (atomicType) {
      case pto::AtomicType::AtomicNone: return "AtomicType::AtomicNone";
      case pto::AtomicType::AtomicAdd: return "AtomicType::AtomicAdd";
    }
    return "AtomicType::AtomicNone";
  }

  static std::string reluPreModeTok(pto::ReluPreMode reluPreMode) {
    switch (reluPreMode) {
      case pto::ReluPreMode::NoRelu: return "ReluPreMode::NoRelu";
      case pto::ReluPreMode::NormalRelu: return "ReluPreMode::NormalRelu";
    }
    return "ReluPreMode::NoRelu";
  }

  static Value buildTStoreDstArg(ConversionPatternRewriter &rewriter,
                                 pto::TStoreOp op, Value dst) {
    auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType());
    if (!dstMrTy)
      return dst;

    bool isGlobal = true;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(dstMrTy.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
    }
    if (!isGlobal)
      return dst;
    if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                              op.getOperation()))
      return gt;
    return dst;
  }

  static FailureOr<std::string> getTStoreOpaqueTok(
      ConversionPatternRewriter &rewriter, pto::TStoreOp op, Value value,
      StringRef name) {
    if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType()))
      return opaqueTy.getValue().str();
    return rewriter.notifyMatchFailure(
        op, (name + " must be emitc::OpaqueType").str());
  }

  struct TStoreOpaqueTokPair {
    std::string srcTok;
    std::string dstTok;
  };

  static FailureOr<TStoreOpaqueTokPair> getTStoreSrcDstOpaqueToks(
      ConversionPatternRewriter &rewriter, pto::TStoreOp op, Value src,
      Value dstArg) {
    auto srcTokOr = getTStoreOpaqueTok(rewriter, op, src, "src");
    auto dstTokOr = getTStoreOpaqueTok(rewriter, op, dstArg, "dst");
    if (failed(srcTokOr) || failed(dstTokOr))
      return failure();
    return TStoreOpaqueTokPair{*srcTokOr, *dstTokOr};
  }

  static ArrayAttr buildOpaqueTemplateArgs(
      ConversionPatternRewriter &rewriter, ArrayRef<std::string> tokens) {
    SmallVector<Attribute> attrs;
    attrs.reserve(tokens.size());
    auto *ctx = rewriter.getContext();
    for (const std::string &token : tokens)
      attrs.push_back(emitc::OpaqueAttr::get(ctx, token));
    return rewriter.getArrayAttr(attrs);
  }

  static FailureOr<ArrayAttr> buildTStoreAtomicTemplateArgs(
      ConversionPatternRewriter &rewriter, pto::TStoreOp op, Value src,
      Value dstArg) {
    const auto phase = op.getStPhase();
    const auto atomicType = op.getAtomicType();
    const bool phaseNonDefault = phase != pto::STPhase::Unspecified;
    auto toks = getTStoreSrcDstOpaqueToks(rewriter, op, src, dstArg);
    if (failed(toks))
      return failure();

    TemplateTokenVector templateToks;
    if (phaseNonDefault)
      templateToks.push_back(stPhaseTok(phase));
    templateToks.push_back(toks->srcTok);
    templateToks.push_back(toks->dstTok);
    templateToks.push_back(atomicTypeTok(atomicType));
    return buildOpaqueTemplateArgs(rewriter, templateToks);
  }

  static FailureOr<ArrayAttr> buildTStoreTemplateArgs(
      ConversionPatternRewriter &rewriter, pto::TStoreOp op, Value src,
      Value dstArg, bool hasPreQuantScalar) {
    const auto phase = op.getStPhase();
    const auto atomicType = op.getAtomicType();
    const auto reluPreMode = op.getReluPreMode();
    const bool phaseNonDefault = phase != pto::STPhase::Unspecified;
    const bool atomicNonDefault = atomicType != pto::AtomicType::AtomicNone;
    const bool reluNonDefault = reluPreMode != pto::ReluPreMode::NoRelu;
    if (!hasPreQuantScalar && !reluNonDefault && !atomicNonDefault) {
      if (phaseNonDefault) {
        return buildOpaqueTemplateArgs(rewriter, {stPhaseTok(phase)});
      }
      return ArrayAttr{};
    }

    if (!hasPreQuantScalar && !reluNonDefault)
      return buildTStoreAtomicTemplateArgs(rewriter, op, src, dstArg);

    auto toks = getTStoreSrcDstOpaqueToks(rewriter, op, src, dstArg);
    if (failed(toks))
      return failure();

    StoreTemplateTokenVector templateToks;
    if (phaseNonDefault)
      templateToks.push_back(stPhaseTok(phase));
    templateToks.push_back(toks->srcTok);
    templateToks.push_back(toks->dstTok);
    templateToks.push_back(atomicTypeTok(atomicType));
    templateToks.push_back(reluPreModeTok(reluPreMode));
    return buildOpaqueTemplateArgs(rewriter, templateToks);
  }

  LogicalResult matchAndRewrite(pto::TStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tstore");

    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());
    Value dstArg = buildTStoreDstArg(rewriter, op, dst);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    FailureOr<ArrayAttr> targs =
        buildTStoreTemplateArgs(rewriter, op, src, dstArg, hasPreQuantScalar);
    if (failed(targs))
      return failure();

    TStoreOperandVector operands{dstArg, src};
    if (hasPreQuantScalar)
      operands.push_back(preQuantScalar);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/*targs,
        /*operands=*/operands);
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute value.
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR
// that does not yet emit an explicit phase attribute.
static ArrayAttr buildAccPhaseTemplateArgs(ConversionPatternRewriter &rewriter,
                                           pto::AccPhase phase) {
  StringRef tmpl;
  switch (phase) {
  case pto::AccPhase::Unspecified:
    return ArrayAttr{};
  case pto::AccPhase::Partial:
    tmpl = "AccPhase::Partial";
    break;
  case pto::AccPhase::Final:
    tmpl = "AccPhase::Final";
    break;
  }
  if (tmpl.empty())
    return ArrayAttr{};
  return rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(rewriter.getContext(), tmpl)});
}

struct PTOTMatmulToTMATMUL : public OpConversionPattern<pto::TMatmulOp> {
  using OpConversionPattern<pto::TMatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Left)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Right)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Acc)

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL",
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs,
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tgemv lowering
//===----------------------------------------------------------------------===//
struct PTOTGemvToTGEMV : public OpConversionPattern<pto::TGemvOp> {
  using OpConversionPattern<pto::TGemvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Matrix)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Vector)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Result)

    // 2. 直接生成函数调用 TGEMV(dst, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tgemv.acc lowering
//===----------------------------------------------------------------------===//
struct PTOTGemvAccToTGEMVACC : public OpConversionPattern<pto::TGemvAccOp> {
  using OpConversionPattern<pto::TGemvAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tgemv.acc");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Matrix)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Vector)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 直接生成函数调用 TGEMV_ACC(dst, accIn, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV_ACC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_acc_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
struct PTOTMatmulAccToTMATMULACC : public OpConversionPattern<pto::TMatmulAccOp> {
  using OpConversionPattern<pto::TMatmulAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tmatmul.acc");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Left)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Right)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL_ACC<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL_ACC",
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs,
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};


} // namespace

void populatePTOToEmitCKernelOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx) {
  patterns.add<PTOTLoadToTLOAD>(typeConverter, ctx);
  patterns.add<PTOTPrefetchToTPREFETCH>(typeConverter, ctx);
  patterns.add<PTOMakePrefetchAsyncContextToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetPrefetchAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOTPrefetchAsyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOTStoreToTSTORE>(typeConverter, ctx);
  patterns.add<PTOTMatmulToTMATMUL>(typeConverter, ctx);
  patterns.add<PTOTMatmulAccToTMATMULACC>(typeConverter, ctx);
  patterns.add<PTOTGemvToTGEMV>(typeConverter, ctx);
  patterns.add<PTOTGemvAccToTGEMVACC>(typeConverter, ctx);
}

} // namespace mlir::pto
