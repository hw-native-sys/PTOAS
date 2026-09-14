// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- TStore.cpp - LoadStore TStore op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "LoadStoreInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

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

void populateLoadStoreTStorePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOTStoreToTSTORE>(typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
