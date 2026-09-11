// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCLoadStore.cpp - tload/tstore/matmul lowering ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOAddPtrToEmitC : public OpConversionPattern<pto::AddPtrOp> {
  using OpConversionPattern<pto::AddPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert pointer type");
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    rewriter.replaceOpWithNewOp<emitc::AddOp>(op, resultType, ptr, offset);
    return success();
  }
};

struct PTOTLoadToTLOAD : public OpConversionPattern<pto::TLoadOp> {
  using OpConversionPattern<pto::TLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tload");

    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Value dst = adaptor.getDst();

    ArrayAttr templateArgs = ArrayAttr{};
    if (auto policy = op.getCachePolicyAttr();
        policy && policy.getValue() == pto::LoadCachePolicy::L2Bypass) {
      templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
          rewriter.getContext(), "pto::TLoadL2Hint::NotAllocKeep")});
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TLOAD",
                                         ArrayAttr{}, templateArgs,
                                         ValueRange{dst, src});

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

    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Value dst = adaptor.getDst();

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TPREFETCH",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{dst, src});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTPrefetchAsyncToEmitC
    : public OpConversionPattern<pto::TPrefetchAsyncOp> {
  using OpConversionPattern<pto::TPrefetchAsyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchAsyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelGlobalTensorConversionBridge(adaptor.getSrc());
    Type convertedSrcTy = getTypeConverter()->convertType(op.getSrc().getType());
    if (!convertedSrcTy || !isEmitCGlobalTensorLikeType(convertedSrcTy))
      return rewriter.notifyMatchFailure(op, "expected GlobalTensor-like src");

    Value prefetchCtx = adaptor.getCtx();

    Type eventTy = getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert tprefetch_async result type");

    Value event =
        rewriter
            .create<emitc::CallOpaqueOp>(
                op.getLoc(), TypeRange{eventTy}, "TPREFETCH_ASYNC", ArrayAttr{},
                ArrayAttr{}, ValueRange{src, prefetchCtx})
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

    Value workspace = adaptor.getWorkspace();
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

    Value ctx = adaptor.getCtx();
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

static void emitTileCallAndReplace(Operation *op, ConversionPatternRewriter &rewriter,
                                   StringRef callee, ArrayAttr templateArgs,
                                   ValueRange operands, Value dst);

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
      case pto::ReluPreMode::ScalarRelu: return "ReluPreMode::ScalarRelu";
      case pto::ReluPreMode::VectorRelu: return "ReluPreMode::VectorRelu";
      case pto::ReluPreMode::Pwl: return "ReluPreMode::Pwl";
    }
    return "ReluPreMode::NoRelu";
  }

  LogicalResult matchAndRewrite(pto::TStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tstore");

    Value src = adaptor.getSrc();
    Value dst = peelGlobalTensorConversionBridge(adaptor.getDst());
    Value fp;
    if (op.getFp())
      fp = adaptor.getFp();
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = adaptor.getPreQuantScalar();

    return emitTStore(op, rewriter, src, dst, dst, fp, preQuantScalar);
  }

  // Resolve the TSTORE overload family from the op attributes and emit the
  // call. See the overload table in the original body for the mapping.
  // Resolve the non-FP TSTORE overload's template arguments from the op
  // attributes (phase / atomic / relu / preQuant).
  FailureOr<ArrayAttr> resolveTStoreTemplateArgs(pto::TStoreOp op,
                                      ConversionPatternRewriter &rewriter,
                                      Value src, Value dstArg, Value fp,
                                      Value preQuantScalar) const {
    auto *ctx = rewriter.getContext();
    const auto phase = op.getStPhase();
    const auto atomicType = op.getAtomicType();
    const auto reluPreMode = op.getReluPreMode();
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool phaseNonDefault = phase != pto::STPhase::Unspecified;
    const bool atomicNonDefault = atomicType != pto::AtomicType::AtomicNone;
    const bool reluNonDefault = reluPreMode != pto::ReluPreMode::NoRelu;

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return FailureOr<std::string>();
    };

    ArrayAttr targs;
// Map op attributes/operands to the exact TSTORE overload family:
//  1) TSTORE(dst, src)
//  2) TSTORE<Phase>(dst, src)
//  3) TSTORE<TileData, GlobalData, AtomicType>(dst, src)
//  4) TSTORE<Phase, TileData, GlobalData, AtomicType>(dst, src)
//  5) TSTORE<TileData, GlobalData, AtomicType, ReluPreMode>(dst, src)
//  6) TSTORE<Phase, TileData, GlobalData, AtomicType, ReluPreMode>(dst, src)
//  7) TSTORE<TileData, GlobalData, AtomicType, ReluPreMode>(dst, src, preQuant)
//  8) TSTORE<Phase, TileData, GlobalData, AtomicType, ReluPreMode>(dst, src, preQuant)
if (!hasPreQuantScalar && !reluNonDefault && !atomicNonDefault) {
  if (phaseNonDefault) {
targs = rewriter.getArrayAttr({
    emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
});
  } else {
targs = ArrayAttr{};
  }
} else {
  auto srcTokOr = getOpaqueTok(src, "src");
  auto dstTokOr = getOpaqueTok(dstArg, "dst");
  if (failed(srcTokOr) || failed(dstTokOr))
    return failure();

  // Token list: [Phase], TileData, GlobalData, AtomicType[, ReluPreMode].
  SmallVector<Attribute, 5> targsList;
  if (phaseNonDefault)
    targsList.push_back(
        emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)));
  targsList.push_back(emitc::OpaqueAttr::get(ctx, *srcTokOr));
  targsList.push_back(emitc::OpaqueAttr::get(ctx, *dstTokOr));
  targsList.push_back(
      emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)));
  // Atomic-only overloads (#3/#4) omit ReluPreMode; the relu/preQuant
  // families (#5-#8) keep it.
  if (hasPreQuantScalar || reluNonDefault)
    targsList.push_back(
        emitc::OpaqueAttr::get(ctx, reluPreModeTok(reluPreMode)));
  return rewriter.getArrayAttr(targsList);
}

    return targs;
  }

  LogicalResult emitTStore(pto::TStoreOp op,
                           ConversionPatternRewriter &rewriter, Value src,
                           Value dstArg, Value dst, Value fp,
                           Value preQuantScalar) const {
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);

    if (hasFp)
      return emitTStoreFp(op, rewriter, src, dstArg, dst, fp);

    auto targsOr = resolveTStoreTemplateArgs(op, rewriter, src, dstArg, fp,
                                             preQuantScalar);
    if (failed(targsOr))
      return failure();
    ArrayAttr targs = *targsOr;

    SmallVector<Value, 3> operands{dstArg, src};
    if (hasPreQuantScalar)
      operands.push_back(preQuantScalar);

    emitTileCallAndReplace(op.getOperation(), rewriter, "TSTORE", targs,
                           operands, dst);
    return success();
  }

  // Fixed-point `TSTORE_FP(dst, src, fp)` path with optional atomic/relu
  // template arguments.
  LogicalResult emitTStoreFp(pto::TStoreOp op,
                             ConversionPatternRewriter &rewriter, Value src,
                             Value dstArg, Value dst, Value fp) const {
    auto *ctx = rewriter.getContext();
    const auto atomicType = op.getAtomicType();
    const auto reluPreMode = op.getReluPreMode();

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(op, (name + " must be emitc::OpaqueType").str());
    };

    ArrayAttr targs;
    SmallVector<Value, 3> operands{dstArg, src, fp};
    if (atomicType != pto::AtomicType::AtomicNone ||
        reluPreMode != pto::ReluPreMode::NoRelu) {
      auto srcTokOr = getOpaqueTok(src, "src");
      auto dstTokOr = getOpaqueTok(dstArg, "dst");
      auto fpTokOr = getOpaqueTok(fp, "fp");
      if (failed(srcTokOr) || failed(dstTokOr) || failed(fpTokOr))
        return failure();
      targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, *srcTokOr),
          emitc::OpaqueAttr::get(ctx, *dstTokOr),
          emitc::OpaqueAttr::get(ctx, *fpTokOr),
          emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
          emitc::OpaqueAttr::get(ctx, reluPreModeTok(reluPreMode)),
      });
    } else {
      targs = ArrayAttr{};
    }

    emitTileCallAndReplace(op.getOperation(), rewriter, "TSTORE_FP", targs,
                           operands, dst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute:
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<pto::AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<pto::AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR

//===----------------------------------------------------------------------===//
//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute:
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<pto::AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<pto::AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR
// that does not yet emit an explicit phase attribute.

// Emit an opaque call for a DPS tile op and forward (or erase) the op: when
// the op has a result, it is replaced by its dst operand.
static void emitTileCallAndReplace(Operation *op, ConversionPatternRewriter &rewriter,
                                   StringRef callee, ArrayAttr templateArgs,
                                   ValueRange operands, Value dst) {
  rewriter.create<emitc::CallOpaqueOp>(op->getLoc(), TypeRange{}, callee,
                                       /*args=*/ArrayAttr{},
                                       /*templateArgs=*/templateArgs, operands);
  if (op->getNumResults() == 1) {
    rewriter.replaceOp(op, dst);
  } else {
    rewriter.eraseOp(op);
  }
}

struct PTOTMatmulToTMATMUL : public OpConversionPattern<pto::TMatmulOp> {
  using OpConversionPattern<pto::TMatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取转换后的目标侧操作数
    Value lhs = adaptor.getLhs(); // A (Left)
    Value rhs = adaptor.getRhs(); // B (Right)
    Value dst = adaptor.getDst(); // C (Acc)

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TMATMUL",
                           templateArgs, ValueRange{dst, lhs, rhs}, dst);
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
    // 1. 获取转换后的目标侧操作数
    Value lhs = adaptor.getLhs(); // A (Matrix)
    Value rhs = adaptor.getRhs(); // B (Vector)
    Value dst = adaptor.getDst(); // C (Result)

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TGEMV",
                           templateArgs, ValueRange{dst, lhs, rhs}, dst);
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
    Value accIn = adaptor.getAccIn(); // AccOld
    Value lhs   = adaptor.getLhs();   // A (Matrix)
    Value rhs   = adaptor.getRhs();   // B (Vector)
    Value dst   = adaptor.getDst();   // AccNew

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TGEMV_ACC",
                           templateArgs, ValueRange{dst, accIn, lhs, rhs}, dst);
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
    Value accIn = adaptor.getAccIn(); // AccOld
    Value lhs   = adaptor.getLhs();   // A (Left)
    Value rhs   = adaptor.getRhs();   // B (Right)
    Value dst   = adaptor.getDst();   // AccNew

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL_ACC<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    emitTileCallAndReplace(op.getOperation(), rewriter, "TMATMUL_ACC",
                           templateArgs, ValueRange{dst, accIn, lhs, rhs}, dst);
    return success();
  }
};


ArrayAttr buildAccPhaseTemplateArgs(ConversionPatternRewriter &rewriter,
                                           pto::AccPhase phase) {
  StringRef tmpl;
  switch (phase) {
  case pto::AccPhase::Unspecified:
    return ArrayAttr{};
  case pto::AccPhase::Partial:
    tmpl = "pto::AccPhase::Partial";
    break;
  case pto::AccPhase::Final:
    tmpl = "pto::AccPhase::Final";
    break;
  }
  if (tmpl.empty())
    return ArrayAttr{};
  return rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(rewriter.getContext(), tmpl)});
}


void populateLoadStorePatterns(RewritePatternSet &patterns,
                              TypeConverter &typeConverter,
                              MLIRContext *ctx, PTOArch targetArch) {
  (void)targetArch;
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
  patterns.add<PTOAddPtrToEmitC>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir


