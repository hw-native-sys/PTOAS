// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCSync.cpp --------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>
#include <type_traits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr llvm::StringLiteral kAutoSyncTailPendingModeAttr =
    "__pto.auto_sync_tail_mode";
static constexpr int64_t kPTOFftsModeSymbolic = 2;
constexpr size_t kTileRank2D = 2;
constexpr size_t kNumber2 = 2;
constexpr size_t kNumber3 = 3;
constexpr unsigned kInlineCapacity2 = 2;
constexpr unsigned kInlineCapacity4 = 4;

template <typename T>
using SmallVec2 = SmallVector<T, kInlineCapacity2>;
template <typename T>
using SmallVec4 = SmallVector<T, kInlineCapacity4>;

static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a);

struct InterCoreSyncCallDesc {
  llvm::StringRef callee;
  ArrayAttr args;
  SmallVec2<Value> operands;
};

static Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter,
                                       Location loc, Value eventId) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  if (eventId.getType() == i32Ty)
    return eventId;
  return emitCCast(rewriter, loc, i32Ty, eventId);
}

static Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter,
                                       int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  if (fftsMode == kPTOFftsModeSymbolic)
    return emitc::OpaqueAttr::get(ctx, "FFTS_MODE_VAL");
  return emitc::OpaqueAttr::get(ctx, std::to_string(fftsMode));
}

static Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventI32, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  auto msgTy = emitc::OpaqueType::get(ctx, "uint16_t");
  auto msgArgs = rewriter.getArrayAttr({
      getFFTSModeCodegenArg(rewriter, fftsMode),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, msgTy, "getFFTSMsg",
                                   /*args=*/msgArgs,
                                   /*templateArgs=*/ArrayAttr{},
                                   /*operands=*/ValueRange{eventI32})
      .getResult(0);
}

static ArrayAttr buildPipeZeroArgs(ConversionPatternRewriter &rewriter,
                                   StringRef pipeTok) {
  auto *ctx = rewriter.getContext();
  return rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
}

static InterCoreSyncCallDesc buildA3InterCoreSyncSetCallDesc(
    ConversionPatternRewriter &rewriter, StringRef pipeTok, Value msgVal) {
  InterCoreSyncCallDesc desc;
  desc.callee = "ffts_cross_core_sync";
  desc.args = buildPipeZeroArgs(rewriter, pipeTok);
  desc.operands.push_back(msgVal);
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCall(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  if (targetArch == PTOArch::A3) {
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value eventVal =
        makeEmitCIntConstant(rewriter, loc, i32Ty, eventIdAttr.getInt());
    Value msgVal = createFFTSMsg(rewriter, loc, eventVal, fftsMode);
    return buildA3InterCoreSyncSetCallDesc(rewriter, pipeTok, msgVal);
  }

  InterCoreSyncCallDesc desc;
  desc.callee = "set_intra_block";
  desc.args = rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(ctx, pipeTok), eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) {
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdVal);
  if (targetArch == PTOArch::A3) {
    Value msgVal = createFFTSMsg(rewriter, loc, eventI32, fftsMode);
    return buildA3InterCoreSyncSetCallDesc(rewriter, pipeTok, msgVal);
  }

  InterCoreSyncCallDesc desc;
  desc.callee = "set_intra_block";
  desc.args = buildPipeZeroArgs(rewriter, pipeTok);
  desc.operands.push_back(eventI32);
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCall(
    ConversionPatternRewriter &rewriter, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  InterCoreSyncCallDesc desc;
  if (targetArch == PTOArch::A3) {
    desc.callee = "wait_flag_dev";
    desc.args = rewriter.getArrayAttr({eventIdAttr});
    return desc;
  }

  desc.callee = "wait_intra_block";
  desc.args = rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(ctx, pipeTok), eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdVal);

  InterCoreSyncCallDesc desc;
  if (targetArch == PTOArch::A3) {
    desc.callee = "wait_flag_dev";
    desc.args = rewriter.getArrayAttr({IntegerAttr::get(IndexType::get(ctx), 0)});
    desc.operands.push_back(eventI32);
    return desc;
  }

  desc.callee = "wait_intra_block";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(eventI32);
  return desc;
}

static LogicalResult emitA5SyncSetCall(ConversionPatternRewriter &rewriter,
                                       Location loc, pto::SyncSetOp op,
                                       Value eventIdDyn,
                                       IntegerAttr eventIdAttr) {
  auto *ctx = rewriter.getContext();
  pto::PIPE pipe = op.getPipe().getPipe();
  bool needsMirrorPlus16 = (pipe == pto::PIPE::PIPE_FIX);
  std::string pipeTok = pipeTokFromPipeAttr(op.getPipe());

  auto emitSet = [&rewriter, ctx, loc, pipeTok](Value eventOperand,
                                                IntegerAttr eventLiteral,
                                                bool isDynamic) {
    if (isDynamic) {
      auto args = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, pipeTok),
          IntegerAttr::get(IndexType::get(ctx), 0),
      });
      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                           /*args=*/args,
                                           /*templateArgs=*/ArrayAttr{},
                                           /*operands=*/ValueRange{eventOperand});
      return;
    }
    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        eventLiteral,
    });
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                         /*args=*/args,
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{});
  };
  if (eventIdAttr) {
    emitSet(Value{}, eventIdAttr, /*isDynamic=*/false);
    if (needsMirrorPlus16) {
      auto plus16 =
          IntegerAttr::get(eventIdAttr.getType(), eventIdAttr.getInt() + 16);
      emitSet(Value{}, plus16, /*isDynamic=*/false);
    }
    return success();
  }

  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdDyn);
  emitSet(eventI32, IntegerAttr{}, /*isDynamic=*/true);
  if (needsMirrorPlus16) {
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value c16 = makeEmitCIntConstant(rewriter, loc, i32Ty, 16);
    Value eventI32Plus16 =
        rewriter.create<emitc::AddOp>(loc, i32Ty, eventI32, c16).getResult();
    emitSet(eventI32Plus16, IntegerAttr{}, /*isDynamic=*/true);
  }
  return success();
}

static FailureOr<emitc::OpaqueType>
buildSyncAllWorkspaceEmitType(ConversionPatternRewriter &rewriter,
                              Value originalWorkspace) {
  auto memTy = dyn_cast<MemRefType>(originalWorkspace.getType());
  if (!memTy || !memTy.hasStaticShape())
    return failure();

  ArrayRef<int64_t> rawShape = memTy.getShape();
  if (rawShape.empty() || rawShape.size() > kTileRank2D)
    return failure();

  int64_t rows = rawShape.size() == 1 ? 1 : rawShape[0];
  int64_t cols = rawShape.size() == 1 ? rawShape[0] : rawShape[1];
  SmallVec2<int64_t> shape{rows, cols};
  SmallVec2<int64_t> validShape{rows, cols};

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  if (auto bind = originalWorkspace.getDefiningOp<pto::BindTileOp>()) {
    configAttr = bind.getConfig();
  } else if (auto cast = originalWorkspace.getDefiningOp<pto::PointerCastOp>()) {
    if (auto config = cast.getConfig())
      configAttr = *config;
  }

  Attribute memorySpace = memTy.getMemorySpace();
  if (!memorySpace)
    return failure();

  auto tileTy = pto::TileBufType::get(ctx, shape, memTy.getElementType(),
                                      memorySpace, validShape, configAttr);
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();
  return emitc::OpaqueType::get(ctx, *tileTypeString);
}

static Value castSyncAllWorkspacePtrToU64(
    ConversionPatternRewriter &rewriter, Location loc, Value workspace) {
  auto *ctx = rewriter.getContext();
  auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
  if (isSetFFTsPointerLikeType(workspace.getType())) {
    auto rcU64 =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                     ArrayAttr{}, rcU64, ValueRange{workspace})
        .getResult(0);
  }
  if (workspace.getType() != u64Ty)
    return rewriter.create<emitc::CastOp>(loc, u64Ty, workspace).getResult();
  return workspace;
}

static FailureOr<Value> buildSyncAllWorkspaceTileValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalWorkspace,
    Value emittedWorkspace) {
  Value workspace = peelUnrealized(emittedWorkspace);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(workspace.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return workspace;
  }

  FailureOr<emitc::OpaqueType> tileEmitTy =
      buildSyncAllWorkspaceEmitType(rewriter, originalWorkspace);
  if (failed(tileEmitTy))
    return failure();
  auto *ctx = rewriter.getContext();
  Value tile = rewriter
                   .create<emitc::VariableOp>(loc, *tileEmitTy,
                                              emitc::OpaqueAttr::get(ctx, ""))
                   .getResult();

  Value rawPtr = castSyncAllWorkspacePtrToU64(rewriter, loc, workspace);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, rawPtr});
  return tile;
}

static FailureOr<Value>
buildSyncAllGmWorkspace(ConversionPatternRewriter &rewriter, pto::SyncAllOp op,
                        pto::SyncAllOp::Adaptor adaptor) {
  Value gm = peelUnrealized(adaptor.getGmWorkspace());
  if (isEmitCGlobalTensorLikeType(gm.getType()))
    return gm;

  auto memTy = dyn_cast<MemRefType>(op.getGmWorkspace().getType());
  if (!memTy)
    return failure();

  Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), gm, memTy,
                                        op.getGmWorkspace().getDefiningOp()
                                            ? op.getGmWorkspace().getDefiningOp()
                                            : op.getOperation());
  if (!gt)
    return failure();
  return gt;
}

static Value buildSyncAllUsedCores(ConversionPatternRewriter &rewriter,
                                   pto::SyncAllOp op,
                                   pto::SyncAllOp::Adaptor adaptor) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  Value usedCores = adaptor.getUsedCores()
                        ? peelUnrealized(adaptor.getUsedCores())
                        : makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
  if (usedCores.getType() != i32Ty) {
    usedCores = rewriter.create<emitc::CastOp>(op.getLoc(), i32Ty, usedCores)
                    .getResult();
  }
  return usedCores;
}

static LogicalResult appendSyncAllWorkspaceOperands(
    ConversionPatternRewriter &rewriter, pto::SyncAllOp op,
    pto::SyncAllOp::Adaptor adaptor, pto::SyncCoreType coreType,
    SmallVectorImpl<Value> &operands) {
  switch (coreType) {
  case pto::SyncCoreType::AIVOnly: {
    FailureOr<Value> ubWorkspace = buildSyncAllWorkspaceTileValue(
        rewriter, op.getLoc(), op.getUbWorkspace(), adaptor.getUbWorkspace());
    if (failed(ubWorkspace))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize ub_workspace tile");
    operands.push_back(*ubWorkspace);
    return success();
  }
  case pto::SyncCoreType::AICOnly: {
    FailureOr<Value> l1Workspace = buildSyncAllWorkspaceTileValue(
        rewriter, op.getLoc(), op.getL1Workspace(), adaptor.getL1Workspace());
    if (failed(l1Workspace))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize l1_workspace tile");
    operands.push_back(*l1Workspace);
    return success();
  }
  case pto::SyncCoreType::Mix: {
    FailureOr<Value> ubWorkspace = buildSyncAllWorkspaceTileValue(
        rewriter, op.getLoc(), op.getUbWorkspace(), adaptor.getUbWorkspace());
    FailureOr<Value> l1Workspace = buildSyncAllWorkspaceTileValue(
        rewriter, op.getLoc(), op.getL1Workspace(), adaptor.getL1Workspace());
    if (failed(ubWorkspace) || failed(l1Workspace)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize mixed syncall workspace tiles");
    }
    operands.push_back(*ubWorkspace);
    operands.push_back(*l1Workspace);
    return success();
  }
  }
  llvm_unreachable("unhandled SyncCoreType");
}

//===----------------------------------------------------------------------===//
// Sync lowering
//===----------------------------------------------------------------------===

static constexpr llvm::StringLiteral kAutoSyncTailBarrierAttr =
    "pto.auto_sync_tail_barrier";
static constexpr llvm::StringLiteral kAutoSyncTailHintAttr =
    "pto.auto_sync_tail_hint";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";
static constexpr llvm::StringLiteral kAutoSyncTailModeBarrierAllToken =
    "PTOAutoSyncTailMode::kBarrierAll";
static constexpr llvm::StringLiteral kAutoSyncTailModeMte3ToSEvent0Token =
    "PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0";

static std::string getAutoSyncTailModeToken(Operation *op) {
  if (op) {
    if (auto hintAttr = op->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr)) {
      if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
        return kAutoSyncTailModeBarrierAllToken.str();
      if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
        return kAutoSyncTailModeMte3ToSEvent0Token.str();
    }
  }

  auto func = op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!func)
    return kAutoSyncTailModeBarrierAllToken.str();

  auto hintAttr = func->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr);
  if (!hintAttr)
    return kAutoSyncTailModeBarrierAllToken.str();
  if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
    return kAutoSyncTailModeBarrierAllToken.str();
  if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
    return kAutoSyncTailModeMte3ToSEvent0Token.str();

  // Fallback to the conservative behavior when seeing unknown policies.
  return kAutoSyncTailModeBarrierAllToken.str();
}

[[maybe_unused]] static std::string getPipeName(pto::PIPE pipe) {
  switch (pipe) {
    case pto::PIPE::PIPE_S: return "PIPE_S";
    case pto::PIPE::PIPE_V: return "PIPE_V";
    case pto::PIPE::PIPE_M: return "PIPE_M";
    case pto::PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case pto::PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case pto::PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case pto::PIPE::PIPE_ALL: return "PIPE_ALL";
    case pto::PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case pto::PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case pto::PIPE::PIPE_V2: return "PIPE_V2";
    case pto::PIPE::PIPE_FIX: return "PIPE_FIX";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    // 默认回退
    default: return "PIPE_ALL";
  }
}

//===----------------------------------------------------------------------===//
// pto.barrier lowering -> pipe_barrier(...)
//===----------------------------------------------------------------------===//
struct PTOBarrierToEmitC : public OpConversionPattern<pto::BarrierOp> {
  using OpConversionPattern<pto::BarrierOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->hasAttr(kAutoSyncTailBarrierAttr)) {
      auto modeAttr = rewriter.getStringAttr(getAutoSyncTailModeToken(op));
      if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
        emitcFunc->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
        funcOp->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // Materialize the enum value carried by the PipeAttr.
    pto::PIPE pipeEnum = op.getPipe().getPipe();

    // Convert Enum to String (e.g., PIPE_ALL -> "PIPE_ALL")
    std::string pipeStr = pto::stringifyPIPE(pipeEnum).str();
    auto *ctx = rewriter.getContext();

    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeStr)
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op,
        TypeRange{},        // void return
        "pipe_barrier",     // function name
        args,               // arguments
        ArrayAttr{},        // template args
        ValueRange{}        // operands
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering (robust for bracket form pto.set_flag[...] / pto.wait_flag[...])
// Replace your PTOSyncToRuntimeCall with the code below.
//===----------------------------------------------------------------------===//

static bool tryConvertPipeAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto pipe = dyn_cast<mlir::pto::PipeAttr>(attr)) {
    token = mlir::pto::stringifyPIPE(pipe.getPipe()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryConvertEventAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto event = dyn_cast<mlir::pto::EventAttr>(attr)) {
    token = mlir::pto::stringifyEVENT(event.getEvent()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryAssignSyncTokens(Attribute srcAttr, Attribute dstAttr,
                                Attribute evtAttr, std::string &srcTok,
                                std::string &dstTok, std::string &evtTok) {
  std::string localSrc;
  std::string localDst;
  std::string localEvt;
  if (!tryConvertPipeAttrToToken(srcAttr, localSrc) ||
      !tryConvertPipeAttrToToken(dstAttr, localDst) ||
      !tryConvertEventAttrToToken(evtAttr, localEvt)) {
    return false;
  }
  srcTok = std::move(localSrc);
  dstTok = std::move(localDst);
  evtTok = std::move(localEvt);
  return true;
}

static bool tryExtractSyncTokensFromNamedAttrs(Operation *op,
                                               StringRef srcName,
                                               StringRef dstName,
                                               StringRef evtName,
                                               std::string &srcTok,
                                               std::string &dstTok,
                                               std::string &evtTok) {
  return tryAssignSyncTokens(op->getAttr(srcName), op->getAttr(dstName),
                             op->getAttr(evtName), srcTok, dstTok, evtTok);
}

static bool tryExtractSyncTokensFromArrayAttr(Operation *op, StringRef attrName,
                                              std::string &srcTok,
                                              std::string &dstTok,
                                              std::string &evtTok) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.size() < kNumber3)
    return false;
  return tryAssignSyncTokens(arrayAttr[0], arrayAttr[1], arrayAttr[2], srcTok,
                             dstTok, evtTok);
}

static bool tryExtractFallbackSyncTokens(Operation *op, std::string &srcTok,
                                         std::string &dstTok,
                                         std::string &evtTok) {
  SmallVec2<std::string> pipes;
  std::string event;
  for (NamedAttribute namedAttr : op->getAttrs()) {
    std::string token;
    if (tryConvertPipeAttrToToken(namedAttr.getValue(), token)) {
      pipes.push_back(std::move(token));
      continue;
    }
    if (event.empty() &&
        tryConvertEventAttrToToken(namedAttr.getValue(), token)) {
      event = std::move(token);
    }
  }
  if (pipes.size() < kNumber2 || event.empty())
    return false;
  srcTok = pipes[0];
  dstTok = pipes[1];
  evtTok = event;
  return true;
}

static LogicalResult extractSyncTripletTokens(Operation *op,
                                             std::string &srcTok,
                                             std::string &dstTok,
                                             std::string &evtTok,
                                             ConversionPatternRewriter &rewriter) {
  if (tryExtractSyncTokensFromNamedAttrs(op, "src_pipe", "dst_pipe", "event_id",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "srcPipe", "dstPipe", "eventId",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "src", "dst", "event", srcTok,
                                         dstTok, evtTok)) {
    return success();
  }

  for (StringRef attrName : {"args", "pipes", "sync", "triplet", "attrs"}) {
    if (tryExtractSyncTokensFromArrayAttr(op, attrName, srcTok, dstTok,
                                          evtTok)) {
      return success();
    }
  }

  if (tryExtractFallbackSyncTokens(op, srcTok, dstTok, evtTok))
    return success();
  return rewriter.notifyMatchFailure(
      op, "cannot extract PIPE/PIPE/EVENT tokens from pto.{set,wait}_flag");
}
static inline std::string pipeTokFromPipeEnum(mlir::pto::PIPE p) {
  return mlir::pto::stringifyPIPE(p).str();
}
[[maybe_unused]] static inline std::string evtTokFromEventEnum(mlir::pto::EVENT e) {
  return mlir::pto::stringifyEVENT(e).str();
}
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a) {
  return mlir::pto::stringifyPIPE(a.getPipe()).str();
}
static inline std::string evtTokFromEventAttr(mlir::pto::EventAttr a) {
  return mlir::pto::stringifyEVENT(a.getEvent()).str();
}

template <typename T, typename = void>
struct HasGetSrcPipe : std::false_type {};
template <typename T>
struct HasGetSrcPipe<T, std::void_t<decltype(std::declval<T>().getSrcPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipe : std::false_type {};
template <typename T>
struct HasGetDstPipe<T, std::void_t<decltype(std::declval<T>().getDstPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventId : std::false_type {};
template <typename T>
struct HasGetEventId<T, std::void_t<decltype(std::declval<T>().getEventId())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetSrcPipeAttr : std::false_type {};
template <typename T>
struct HasGetSrcPipeAttr<T, std::void_t<decltype(std::declval<T>().getSrcPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipeAttr : std::false_type {};
template <typename T>
struct HasGetDstPipeAttr<T, std::void_t<decltype(std::declval<T>().getDstPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventIdAttr : std::false_type {};
template <typename T>
struct HasGetEventIdAttr<T, std::void_t<decltype(std::declval<T>().getEventIdAttr())>> : std::true_type {};

template <typename SyncOpT>
static LogicalResult extractSyncTokens(SyncOpT op,
                                      std::string &srcTok,
                                      std::string &dstTok,
                                      std::string &evtTok,
                                      ConversionPatternRewriter &rewriter) {
  if constexpr (HasGetSrcPipe<SyncOpT>::value &&
                HasGetDstPipe<SyncOpT>::value &&
                HasGetEventId<SyncOpT>::value) {
    auto s = op.getSrcPipe();
    auto d = op.getDstPipe();
    auto e = op.getEventId();
    if constexpr (std::is_same<decltype(s), mlir::pto::PIPE>::value) srcTok = pipeTokFromPipeEnum(s);
    else srcTok = pipeTokFromPipeAttr(s);
    if constexpr (std::is_same<decltype(d), mlir::pto::PIPE>::value) dstTok = pipeTokFromPipeEnum(d);
    else dstTok = pipeTokFromPipeAttr(d);
    if constexpr (std::is_same<decltype(e), mlir::pto::EVENT>::value) evtTok = evtTokFromEventEnum(e);
    else evtTok = evtTokFromEventAttr(e);
    return success();
  }

  if constexpr (HasGetSrcPipeAttr<SyncOpT>::value &&
                HasGetDstPipeAttr<SyncOpT>::value &&
                HasGetEventIdAttr<SyncOpT>::value) {
    auto s = op.getSrcPipeAttr();
    auto d = op.getDstPipeAttr();
    auto e = op.getEventIdAttr();
    srcTok = pipeTokFromPipeAttr(s);
    dstTok = pipeTokFromPipeAttr(d);
    evtTok = evtTokFromEventAttr(e);
    return success();
  }

  return extractSyncTripletTokens(op.getOperation(), srcTok, dstTok, evtTok, rewriter);
}

static ArrayAttr buildSyncTokenArgsAttr(ConversionPatternRewriter &rewriter,
                                        StringRef srcTok, StringRef dstTok,
                                        StringRef evtTok) {
  auto *ctx = rewriter.getContext();
  return rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, srcTok),
      emitc::OpaqueAttr::get(ctx, dstTok),
      emitc::OpaqueAttr::get(ctx, evtTok),
  });
}

template <typename SyncFlagOp>
static LogicalResult lowerSyncFlagLikeOp(SyncFlagOp op,
                                         ConversionPatternRewriter &rewriter,
                                         StringRef callee) {
  std::string srcTok, dstTok, evtTok;
  if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
    return failure();

  auto argsAttr = buildSyncTokenArgsAttr(rewriter, srcTok, dstTok, evtTok);
  rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
      op, TypeRange{}, callee,
      /*args=*/argsAttr,
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ValueRange{});
  return success();
}

struct PTOSetFlagToEmitC : public OpConversionPattern<mlir::pto::SetFlagOp> {
  using OpConversionPattern<mlir::pto::SetFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return lowerSyncFlagLikeOp(op, rewriter, "set_flag");
  }
};

struct PTOWaitFlagToEmitC : public OpConversionPattern<mlir::pto::WaitFlagOp> {
  using OpConversionPattern<mlir::pto::WaitFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::WaitFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    return lowerSyncFlagLikeOp(op, rewriter, "wait_flag");
  }
};

struct PTOSyncToEmitC : public OpConversionPattern<mlir::pto::TSyncOp> {
  using OpConversionPattern<mlir::pto::TSyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::TSyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVec4<Value> operands;
    operands.reserve(adaptor.getEvents().size());
    for (Value event : adaptor.getEvents())
      operands.push_back(peelUnrealized(event));

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSYNC",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(operands));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncAllToEmitC : public OpConversionPattern<mlir::pto::SyncAllOp> {
  using OpConversionPattern<mlir::pto::SyncAllOp>::OpConversionPattern;

  static StringRef coreTypeTok(pto::SyncCoreType coreType) {
    switch (coreType) {
    case pto::SyncCoreType::AIVOnly:
      return "SyncCoreType::AIVOnly";
    case pto::SyncCoreType::AICOnly:
      return "SyncCoreType::AICOnly";
    case pto::SyncCoreType::Mix:
      return "SyncCoreType::Mix";
    }
    llvm_unreachable("unhandled SyncCoreType");
  }

  LogicalResult matchAndRewrite(mlir::pto::SyncAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mode = op.getMode().getValue();
    auto coreType = op.getCoreType().getValue();
    if (mode == pto::SyncAllMode::Hard) {
      std::string callee = "SYNCALL<" + coreTypeTok(coreType).str() + ">";
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<Value> gmWorkspace = buildSyncAllGmWorkspace(rewriter, op, adaptor);
    if (failed(gmWorkspace))
      return rewriter.notifyMatchFailure(op,
                                         "failed to build gm_workspace GlobalTensor");
    Value usedCores = buildSyncAllUsedCores(rewriter, op, adaptor);

    std::string callee =
        "SYNCALL<SyncAllMode::Soft, " + coreTypeTok(coreType).str() + ">";

    SmallVec4<Value> operands{*gmWorkspace};
    if (failed(appendSyncAllWorkspaceOperands(rewriter, op, adaptor, coreType,
                                              operands)))
      return failure();

    operands.push_back(usedCores);
    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(operands));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncFlagDynToEmitC : public ConversionPattern {
  PTOSyncFlagDynToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                        StringRef opName, StringRef callee)
      : ConversionPattern(typeConverter, opName, /*benefit=*/1, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected exactly one dynamic event-id operand");

    auto srcAttr = op->getAttrOfType<mlir::pto::PipeAttr>("src_pipe");
    auto dstAttr = op->getAttrOfType<mlir::pto::PipeAttr>("dst_pipe");
    if (!srcAttr || !dstAttr)
      return rewriter.notifyMatchFailure(op, "missing PipeAttr src_pipe/dst_pipe attrs");

    auto *ctx = rewriter.getContext();
    std::string srcTok = pipeTokFromPipeAttr(srcAttr);
    std::string dstTok = pipeTokFromPipeAttr(dstAttr);

    Value eventVal = operands.front();
    eventVal =
        emitCCast(rewriter, op->getLoc(), emitc::OpaqueType::get(ctx, "event_t"), eventVal);

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee,
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{eventVal});
    return success();
  }

private:
  std::string callee;
};

struct PTOGetBufToEmitC : public OpConversionPattern<mlir::pto::GetBufOp> {
  using OpConversionPattern<mlir::pto::GetBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::GetBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "get_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "get_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "get_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTORlsBufToEmitC : public OpConversionPattern<mlir::pto::RlsBufOp> {
  using OpConversionPattern<mlir::pto::RlsBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::RlsBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "rls_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "rls_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "rls_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOSetFFTsToEmitC : public OpConversionPattern<mlir::pto::SetFFTsOp> {
  using OpConversionPattern<mlir::pto::SetFFTsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFFTsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value fftsAddr = peelUnrealized(adaptor.getFfts());
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    if (isSetFFTsPointerLikeType(fftsAddr.getType())) {
      auto castTyAttr =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      fftsAddr =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                           /*args=*/ArrayAttr{},
                                           /*templateArgs=*/castTyAttr,
                                           /*operands=*/ValueRange{fftsAddr})
              .getResult(0);
    } else if (fftsAddr.getType() != u64Ty) {
      fftsAddr =
          rewriter.create<emitc::CastOp>(loc, u64Ty, fftsAddr).getResult();
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_ffts_base_addr",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{fftsAddr});
    return success();
  }
};

struct PTOSyncSetToEmitC : public OpConversionPattern<mlir::pto::SyncSetOp> {
  PTOSyncSetToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                    PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncSetOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();
    int64_t fftsMode = 2;
    if (IntegerAttr fftsModeAttr = op.getFftsModeAttr())
      fftsMode = fftsModeAttr.getInt();
    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    // A5 inter-core sync mirrors +16 only for cube-side producer (PIPE_FIX).
    // Vec-side producer (PIPE_MTE3) emits a single set; hardware handles the
    // subblock mapping in PTO-ISA custom flow.
    if (targetArch == PTOArch::A5) {
      if (failed(emitA5SyncSetCall(rewriter, loc, op, eventIdDyn, eventIdAttr)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncSetCall(rewriter, loc, targetArch, op.getPipe(),
                                       eventIdAttr, fftsMode);
    } else {
      desc = buildInterCoreSyncSetCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                          eventIdDyn, fftsMode);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         /*args=*/desc.args,
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

struct PTOSyncWaitToEmitC : public OpConversionPattern<mlir::pto::SyncWaitOp> {
  PTOSyncWaitToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                     PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncWaitOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();
    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncWaitCall(rewriter, targetArch, op.getPipe(),
                                        eventIdAttr);
    } else {
      desc = buildInterCoreSyncWaitCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                           eventIdDyn);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         desc.args, ArrayAttr{}, desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};


} // namespace

void populatePTOToEmitCSyncPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx, PTOArch targetArch) {
  patterns.add<PTOSetFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_dyn",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_dyn",
                                      "wait_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_d",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_d",
                                      "wait_flag");
  patterns.add<PTOWaitFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncAllToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBufToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFFTsToEmitC>(typeConverter, ctx);
  patterns.add<PTOBarrierToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx, targetArch);
}

} // namespace mlir::pto
