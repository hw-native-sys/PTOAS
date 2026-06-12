// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCComm.cpp --------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <string>
#include <type_traits>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";
static constexpr int8_t kPTOFrontendDirMaskC2V = 1;
static constexpr int8_t kPTOFrontendDirMaskV2C = 2;
static constexpr int8_t kPTOFrontendDirMaskBidirectional = 3;

struct PTOInitializeL2G2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp> {
  PTOInitializeL2G2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2G2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    Value gmAddr = peelUnrealized(adaptor.getGmAddr());
    gmAddr = materializeTensorViewDataPointer(
        rewriter, op.getLoc(), gmAddr, op.getGmAddr().getType());
    Value localAddr =
        op.getLocalAddr() ? peelUnrealized(adaptor.getLocalAddr()) : Value();
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == kPTOFrontendDirMaskC2V)
      c2vBuf = localAddr ? localAddr : zero;
    else if (op.getDirMask() == kPTOFrontendDirMaskV2C)
      v2cBuf = localAddr ? localAddr : zero;
    else if (op.getDirMask() == kPTOFrontendDirMaskBidirectional) {
      if (localAddr) {
        if (!op.getPeerLocalAddr())
          return rewriter.notifyMatchFailure(
              op, "bidirectional l2g2l pipe requires peer local buffer");
        c2vBuf = localAddr;
        v2cBuf = peelUnrealized(adaptor.getPeerLocalAddr());
      }
    } else
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{gmAddr, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOInitializeL2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2LPipeOp> {
  PTOInitializeL2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                              PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    auto gmPtrTy =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ void"));
    Value nullGm =
        makeEmitCOpaqueConstant(rewriter, op.getLoc(), gmPtrTy, "nullptr");
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
    Value localAddr = peelUnrealized(adaptor.getLocalAddr());

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == kPTOFrontendDirMaskC2V)
      c2vBuf = localAddr;
    else if (op.getDirMask() == kPTOFrontendDirMaskV2C)
      v2cBuf = localAddr;
    else if (op.getDirMask() == kPTOFrontendDirMaskBidirectional) {
      c2vBuf = localAddr;
      v2cBuf = peelUnrealized(adaptor.getPeerLocalAddr());
    } else
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{nullGm, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOBuildAsyncSessionToEmitC
    : public OpConversionPattern<mlir::pto::BuildAsyncSessionOp> {
  PTOBuildAsyncSessionToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<mlir::pto::BuildAsyncSessionOp>(typeConverter, ctx) {}

  static Value makeAsyncSessionU32Const(ConversionPatternRewriter &rewriter,
                                        Location loc, Type u32Ty,
                                        uint64_t value) {
    return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                   std::to_string(value) + "u");
  }

  static Value buildAsyncSessionBaseConfig(ConversionPatternRewriter &rewriter,
                                           Location loc, MLIRContext *ctx,
                                           uint64_t blockBytes,
                                           uint64_t commBlockOffset,
                                           uint64_t queueNum) {
    auto baseConfigTy =
        emitc::OpaqueType::get(ctx, "pto::comm::sdma::SdmaBaseConfig");
    return rewriter
        .create<emitc::VariableOp>(
            loc, baseConfigTy,
            emitc::OpaqueAttr::get(ctx, "{" + std::to_string(blockBytes) +
                                            "ULL, " +
                                            std::to_string(commBlockOffset) +
                                            "ULL, " +
                                            std::to_string(queueNum) + "u}"))
        .getResult();
  }

  LogicalResult matchAndRewrite(mlir::pto::BuildAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();

    auto sessionTy =
        dyn_cast<emitc::OpaqueType>(getTypeConverter()->convertType(op.getSession().getType()));
    if (!sessionTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async session type");

    FailureOr<Value> scratchTile =
        buildAsyncScratchTileValue(rewriter, loc, op.getScratch(),
                                   adaptor.getScratch());
    if (failed(scratchTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize async scratch tile");

    Value workspace =
        castToGMBytePointer(rewriter, loc, peelUnrealized(adaptor.getWorkspace()));

    Value session = rewriter
                        .create<emitc::VariableOp>(
                            loc, sessionTy, emitc::OpaqueAttr::get(ctx, ""))
                        .getResult();

    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");
    uint64_t syncId = op.getSyncIdAttr() ? op.getSyncIdAttr().getInt() : 0;
    uint64_t blockBytes =
        op.getBlockBytesAttr() ? op.getBlockBytesAttr().getInt() : 32 * 1024;
    uint64_t commBlockOffset =
        op.getCommBlockOffsetAttr() ? op.getCommBlockOffsetAttr().getInt() : 0;
    uint64_t queueNum = op.getQueueNumAttr() ? op.getQueueNumAttr().getInt() : 1;
    uint64_t channelGroupIdx = op.getChannelGroupIdxAttr()
                                   ? op.getChannelGroupIdxAttr().getInt()
                                   : UINT32_MAX;

    Value syncIdVal = makeAsyncSessionU32Const(rewriter, loc, u32Ty, syncId);
    Value channelGroupIdxVal =
        channelGroupIdx == UINT32_MAX
            ? makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX")
            : makeAsyncSessionU32Const(rewriter, loc, u32Ty, channelGroupIdx);
    Value baseConfig = buildAsyncSessionBaseConfig(rewriter, loc, ctx, blockBytes,
                                                   commBlockOffset, queueNum);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{*scratchTile, workspace, session, syncIdVal, baseConfig,
                   channelGroupIdxVal});

    rewriter.replaceOp(op, session);
    return success();
  }
};

template <typename AsyncOp>
struct PTOAsyncTransferToEmitC : public OpConversionPattern<AsyncOp> {
  using OpConversionPattern<AsyncOp>::OpConversionPattern;

  explicit PTOAsyncTransferToEmitC(const TypeConverter &typeConverter,
                                   MLIRContext *ctx,
                                   StringRef callee)
      : OpConversionPattern<AsyncOp>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncOp op, typename AsyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override { // NOLINT(readability-non-const-parameter)
    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value dstGT = dst;
    Value srcGT = src;
    if (!isEmitCGlobalTensorLikeType(dstGT.getType())) {
      auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType());
      if (!dstMrTy)
        return rewriter.notifyMatchFailure(op, "expected dst to lower to GlobalTensor or memref");
      dstGT = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                          op.getDst().getDefiningOp()
                                              ? op.getDst().getDefiningOp()
                                              : op.getOperation());
    }
    if (!isEmitCGlobalTensorLikeType(srcGT.getType())) {
      auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType());
      if (!srcMrTy)
        return rewriter.notifyMatchFailure(op, "expected src to lower to GlobalTensor or memref");
      srcGT = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                          op.getSrc().getDefiningOp()
                                              ? op.getSrc().getDefiningOp()
                                              : op.getOperation());
    }
    if (!dstGT || !srcGT)
      return rewriter.notifyMatchFailure(op, "failed to build GlobalTensor operands");

    Type eventTy = this->getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{eventTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{dstGT, srcGT, peelUnrealized(adaptor.getSession())});
    return success();
  }

  std::string callee;
};

template <typename AsyncEventOp>
struct PTOAsyncEventToEmitC : public ConversionPattern {
  explicit PTOAsyncEventToEmitC(const TypeConverter &typeConverter,
                                MLIRContext *ctx,
                                StringRef callee)
      : ConversionPattern(typeConverter, AsyncEventOp::getOperationName(),
                          /*benefit=*/1, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &builder) const override {
    return rewriteAsyncEvent(op, operands, &builder);
  }

private:
  LogicalResult rewriteAsyncEvent(Operation *rootOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter *builder) const {
    auto op = cast<AsyncEventOp>(rootOp);
    typename AsyncEventOp::Adaptor adaptor(operands, op);
    Type resultTy =
        this->getTypeConverter()->convertType(op.getCompleted().getType());
    if (!resultTy)
      return failure();

    builder->replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getEvent()),
                   peelUnrealized(adaptor.getSession())});
    return success();
  }

  std::string callee;
};

static FailureOr<Value> buildCommGlobalTensorValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalValue,
    Value emittedValue, Operation *anchor) {
  Value value = peelUnrealized(emittedValue);
  if (isEmitCGlobalTensorLikeType(value.getType()))
    return value;

  auto memTy = dyn_cast<MemRefType>(originalValue.getType());
  if (!memTy)
    return failure();

  Value gt = buildGlobalTensorFromMemref(rewriter, loc, value, memTy, anchor);
  if (!gt)
    return failure();
  return gt;
}

static FailureOr<Value> buildCommTileValue(ConversionPatternRewriter &rewriter,
                                           Location loc, Value originalValue,
                                           Value emittedValue) {
  Value value = peelUnrealized(emittedValue);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return value;
  }
  return buildAsyncScratchTileValue(rewriter, loc, originalValue, emittedValue);
}

static FailureOr<Value> buildCollectiveParallelGroup(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Value> groupGTs, int64_t root) {
  if (groupGTs.empty())
    return failure();

  auto firstTy = dyn_cast<emitc::OpaqueType>(groupGTs.front().getType());
  if (!firstTy)
    return failure();

  auto *ctx = rewriter.getContext();
  auto arrayTy = emitc::ArrayType::get({static_cast<int64_t>(groupGTs.size())},
                                       firstTy);
  auto groupArray = cast<TypedValue<emitc::ArrayType>>(
      rewriter
          .create<emitc::VariableOp>(loc, arrayTy,
                                     emitc::OpaqueAttr::get(ctx, "{}"))
          .getResult());

  auto indexTy = emitc::OpaqueType::get(ctx, "int");
  for (auto [idx, groupVal] : llvm::enumerate(groupGTs)) {
    Value idxVal =
        makeEmitCIntConstant(rewriter, loc, indexTy, static_cast<int64_t>(idx));
    Value slot =
        rewriter.create<emitc::SubscriptOp>(loc, groupArray, ValueRange{idxVal})
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, slot, groupVal);
  }

  std::string pgTypeStr =
      (Twine("pto::comm::ParallelGroup<") + firstTy.getValue() + ">").str();
  auto pgTy = emitc::OpaqueType::get(ctx, pgTypeStr);
  Value sizeVal = makeEmitCIntConstant(rewriter, loc, indexTy,
                                       static_cast<int64_t>(groupGTs.size()));
  Value rootVal = makeEmitCIntConstant(rewriter, loc, indexTy, root);
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, TypeRange{pgTy}, (Twine(pgTypeStr) + "::Create").str(),
          ArrayAttr{}, ArrayAttr{}, ValueRange{groupArray, sizeVal, rootVal})
      .getResult(0);
}

static std::string notifyOpTok(pto::NotifyOp op) {
  switch (op) {
  case pto::NotifyOp::AtomicAdd:
    return "pto::comm::NotifyOp::AtomicAdd";
  case pto::NotifyOp::Set:
    return "pto::comm::NotifyOp::Set";
  }
  return "pto::comm::NotifyOp::Set";
}

static std::string waitCmpTok(pto::WaitCmp cmp) {
  switch (cmp) {
  case pto::WaitCmp::EQ:
    return "pto::comm::WaitCmp::EQ";
  case pto::WaitCmp::NE:
    return "pto::comm::WaitCmp::NE";
  case pto::WaitCmp::GT:
    return "pto::comm::WaitCmp::GT";
  case pto::WaitCmp::GE:
    return "pto::comm::WaitCmp::GE";
  case pto::WaitCmp::LT:
    return "pto::comm::WaitCmp::LT";
  case pto::WaitCmp::LE:
    return "pto::comm::WaitCmp::LE";
  }
  return "pto::comm::WaitCmp::EQ";
}

static std::string reduceOpTok(pto::ReduceOp op) {
  switch (op) {
  case pto::ReduceOp::Sum:
    return "pto::comm::ReduceOp::Sum";
  case pto::ReduceOp::Max:
    return "pto::comm::ReduceOp::Max";
  case pto::ReduceOp::Min:
    return "pto::comm::ReduceOp::Min";
  }
  return "pto::comm::ReduceOp::Sum";
}

template <typename OpTy>
static FailureOr<SmallVector<Value>> buildCommGroupGlobalTensors(
    ConversionPatternRewriter &rewriter, Location loc, OpTy op,
    ValueRange originalGroup, ValueRange emittedGroup) {
  SmallVector<Value> groupGTs;
  groupGTs.reserve(originalGroup.size());
  for (auto [orig, emitted] : llvm::zip(originalGroup, emittedGroup)) {
    FailureOr<Value> gt =
        buildCommGlobalTensorValue(rewriter, loc, orig, emitted, op.getOperation());
    if (failed(gt))
      return failure();
    groupGTs.push_back(*gt);
  }
  return groupGTs;
}

template <typename CollectiveOp>
struct PTOCommCollectiveToEmitC : public OpConversionPattern<CollectiveOp> {
  using OpConversionPattern<CollectiveOp>::OpConversionPattern;

  struct CollectiveBaseOperands {
    Value gt;
    Value pingTile;
    Value parallelGroup;
  };

  explicit PTOCommCollectiveToEmitC(const TypeConverter &typeConverter,
                                    MLIRContext *ctx,
                                    StringRef apiName)
      : OpConversionPattern<CollectiveOp>(typeConverter, ctx),
        apiName(apiName.str()) {}

  static FailureOr<Value> buildCollectivePongTile(
      ConversionPatternRewriter &rewriter, Location loc, Value original,
      Value emitted) {
    if (!original)
      return failure();
    return buildCommTileValue(rewriter, loc, original, emitted);
  }

  static FailureOr<CollectiveBaseOperands> buildCollectiveBaseOperands(
      CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter, Value originalGT, Value emittedGT) {
    Location loc = op.getLoc();
    FailureOr<Value> gt = buildCommGlobalTensorValue(rewriter, loc, originalGT,
                                                     emittedGT, op.getOperation());
    FailureOr<Value> pingTile =
        buildCommTileValue(rewriter, loc, op.getPing(), adaptor.getPing());
    auto groupGTs = buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(),
                                                adaptor.getGroup());
    if (failed(gt) || failed(pingTile) || failed(groupGTs))
      return failure();
    FailureOr<Value> pg =
        buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
    if (failed(pg))
      return failure();
    return CollectiveBaseOperands{*gt, *pingTile, *pg};
  }

  static LogicalResult emitCollectiveCallWithOptionalPong(
      StringRef callee, CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter, const CollectiveBaseOperands &base) {
    Location loc = op.getLoc();
    if (op.getPong()) {
      FailureOr<Value> pongTile =
          buildCollectivePongTile(rewriter, loc, op.getPong(), adaptor.getPong());
      if (failed(pongTile))
        return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
          ValueRange{base.parallelGroup, base.gt, base.pingTile, *pongTile});
      return success();
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{base.parallelGroup, base.gt, base.pingTile});
    return success();
  }

  static LogicalResult emitBroadcast(CollectiveOp op,
                                     typename CollectiveOp::Adaptor adaptor,
                                     ConversionPatternRewriter &rewriter) {
    auto base = buildCollectiveBaseOperands(op, adaptor, rewriter, op.getSrc(),
                                            adaptor.getSrc());
    if (failed(base))
      return rewriter.notifyMatchFailure(op, "failed to materialize broadcast operands");
    return emitCollectiveCallWithOptionalPong("pto::comm::TBROADCAST", op,
                                              adaptor, rewriter, *base);
  }

  static LogicalResult emitGather(CollectiveOp op,
                                  typename CollectiveOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) {
    auto base = buildCollectiveBaseOperands(op, adaptor, rewriter, op.getDst(),
                                            adaptor.getDst());
    if (failed(base))
      return rewriter.notifyMatchFailure(op, "failed to materialize gather operands");
    return emitCollectiveCallWithOptionalPong("pto::comm::TGATHER", op, adaptor,
                                              rewriter, *base);
  }

  static LogicalResult emitScatter(CollectiveOp op,
                                   typename CollectiveOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) {
    auto base = buildCollectiveBaseOperands(op, adaptor, rewriter, op.getSrc(),
                                            adaptor.getSrc());
    if (failed(base))
      return rewriter.notifyMatchFailure(op, "failed to materialize scatter operands");
    return emitCollectiveCallWithOptionalPong("pto::comm::TSCATTER", op, adaptor,
                                              rewriter, *base);
  }

  static LogicalResult emitReduce(CollectiveOp op,
                                  typename CollectiveOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    FailureOr<Value> dstGT = buildCommGlobalTensorValue(
        rewriter, loc, op.getDst(), adaptor.getDst(), op.getOperation());
    FailureOr<Value> accTile =
        buildCommTileValue(rewriter, loc, op.getAcc(), adaptor.getAcc());
    FailureOr<Value> recvPing =
        buildCommTileValue(rewriter, loc, op.getRecvPing(), adaptor.getRecvPing());
    auto groupGTs =
        buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(), adaptor.getGroup());
    if (failed(dstGT) || failed(accTile) || failed(recvPing) || failed(groupGTs))
      return rewriter.notifyMatchFailure(op, "failed to materialize reduce operands");
    FailureOr<Value> pg =
        buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
    if (failed(pg))
      return rewriter.notifyMatchFailure(op, "failed to materialize reduce group");

    auto reduceTy =
        emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::ReduceOp");
    Value reduceOp = makeEmitCOpaqueConstant(rewriter, loc, reduceTy,
                                            reduceOpTok(op.getReduceOp()));
    if (op.getRecvPong()) {
      FailureOr<Value> recvPong = buildCollectivePongTile(
          rewriter, loc, op.getRecvPong(), adaptor.getRecvPong());
      if (failed(recvPong))
        return rewriter.notifyMatchFailure(op, "failed to materialize recv_pong");
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "pto::comm::TREDUCE", ArrayAttr{}, ArrayAttr{},
          ValueRange{*pg, *dstGT, *accTile, *recvPing, *recvPong, reduceOp});
      return success();
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "pto::comm::TREDUCE", ArrayAttr{}, ArrayAttr{},
        ValueRange{*pg, *dstGT, *accTile, *recvPing, reduceOp});
    return success();
  }

  LogicalResult matchAndRewrite(CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override { // NOLINT(readability-non-const-parameter)
    if constexpr (std::is_same_v<CollectiveOp, pto::TBroadcastOp>) {
      if (failed(emitBroadcast(op, adaptor, rewriter)))
        return failure();
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTGatherOp>) {
      if (failed(emitGather(op, adaptor, rewriter)))
        return failure();
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTScatterOp>) {
      if (failed(emitScatter(op, adaptor, rewriter)))
        return failure();
    } else {
      if (failed(emitReduce(op, adaptor, rewriter)))
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  std::string apiName;
};

template <typename OpTy>
struct PTOP2PCommToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTOP2PCommToEmitC(const TypeConverter &typeConverter,
                             MLIRContext *ctx,
                             StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override { // NOLINT(readability-non-const-parameter)
    FailureOr<Value> dstGT =
        buildCommGlobalTensorValue(rewriter, op.getLoc(), op.getDst(), adaptor.getDst(),
                                   op.getOperation());
    FailureOr<Value> srcGT =
        buildCommGlobalTensorValue(rewriter, op.getLoc(), op.getSrc(), adaptor.getSrc(),
                                   op.getOperation());
    FailureOr<Value> pingTile =
        buildCommTileValue(rewriter, op.getLoc(), op.getPing(), adaptor.getPing());
    if (failed(dstGT) || failed(srcGT) || failed(pingTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize p2p operands");

    SmallVector<Value> operands{*dstGT, *srcGT, *pingTile};
    std::string actualCallee = callee;
    if constexpr (std::is_same_v<OpTy, pto::TPutOp>) {
      if (op.getAtomicType() == pto::AtomicType::AtomicAdd)
        actualCallee = "pto::comm::TPUT<pto::AtomicType::AtomicAdd>";
    }
    if (op.getPong()) {
      FailureOr<Value> pongTile =
          buildCommTileValue(rewriter, op.getLoc(), op.getPong(), adaptor.getPong());
      if (failed(pongTile))
        return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
      operands.push_back(*pongTile);
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, actualCallee,
                                         ArrayAttr{}, ArrayAttr{}, operands);
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

template <typename SignalOp>
struct PTOSignalCommToEmitC : public OpConversionPattern<SignalOp> {
  using OpConversionPattern<SignalOp>::OpConversionPattern;

  explicit PTOSignalCommToEmitC(const TypeConverter &typeConverter,
                                MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<SignalOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(SignalOp op, typename SignalOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override { // NOLINT(readability-non-const-parameter)
    FailureOr<Value> signalGT = buildCommGlobalTensorValue(
        rewriter, op.getLoc(), op.getSignal(), adaptor.getSignal(), op.getOperation());
    if (failed(signalGT))
      return rewriter.notifyMatchFailure(op, "failed to materialize signal operand");

    if constexpr (std::is_same_v<SignalOp, pto::TNotifyOp>) {
      auto notifyTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::NotifyOp");
      Value notifyOp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), notifyTy, notifyOpTok(op.getNotifyOp()));
      SmallVector<Value> operands{*signalGT, peelUnrealized(adaptor.getValue()),
                                  notifyOp};
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{}, operands);
      rewriter.eraseOp(op);
    } else {
      auto waitCmpTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::WaitCmp");
      Value waitCmp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), waitCmpTy, waitCmpTok(op.getCmp()));
      SmallVector<Value> operands{*signalGT, peelUnrealized(adaptor.getCmpValue()),
                                  waitCmp};
      if constexpr (std::is_same_v<SignalOp, pto::TTestOp>) {
        Type resultTy = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!resultTy)
          return rewriter.notifyMatchFailure(op, "failed to convert ttest result type");
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{}, operands);
      } else {
        rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                             ArrayAttr{}, ArrayAttr{}, operands);
        rewriter.eraseOp(op);
      }
    }
    return success();
  }

  std::string callee;
};

struct PTODeclareTileMemRefToEmitC
    : public OpConversionPattern<mlir::pto::DeclareTileMemRefOp> {
  using OpConversionPattern<
      mlir::pto::DeclareTileMemRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareTileMemRefOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_tile_memref result type");
    rewriter.replaceOp(op, makeEmitCOpaqueConstant(rewriter, op.getLoc(),
                                                   convertedType, "nullptr"));
    return success();
  }
};

struct PTODeclareGlobalToEmitC
    : public OpConversionPattern<mlir::pto::DeclareGlobalOp> {
  using OpConversionPattern<
      mlir::pto::DeclareGlobalOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareGlobalOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getEntry().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_global result type");
    if (auto tvTy = dyn_cast<TensorViewType>(op.getEntry().getType())) {
      if (auto stridesAttr =
              op->getAttrOfType<DenseI64ArrayAttr>(kGlobalTensorStridesAttrName)) {
        auto strides = stridesAttr.asArrayRef();
        if (strides.size() == static_cast<size_t>(tvTy.getRank())) {
          convertedType = emitc::OpaqueType::get(
              rewriter.getContext(),
              getGlobalTensorTypeStringFromShapeAndStrides(
                  tvTy.getElementType(), tvTy.getShape(), strides));
        }
      }
    }
    auto var = rewriter.create<emitc::VariableOp>(
        op.getLoc(), convertedType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(op, var.getResult());
    return success();
  }
};

struct PTODeclareEventIdArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareEventIdArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareEventIdArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareEventIdArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map declared eventid_array type");

    auto array = rewriter
                     .create<emitc::VariableOp>(
                         op.getLoc(), arrayTy,
                         emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                     .getResult();
    rewriter.replaceOp(op, array);
    return success();
  }
};

struct PTOEventIdArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = peelUnrealized(adaptor.getArray());
    Value index = peelUnrealized(adaptor.getIndex());

    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map eventid_array get result type");

    auto load =
        rewriter.create<emitc::SubscriptOp>(op.getLoc(), resultTy, array, index);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct PTOEventIdArraySetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArraySetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = peelUnrealized(adaptor.getArray());
    Value index = peelUnrealized(adaptor.getIndex());
    Value value = peelUnrealized(adaptor.getValue());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__EVENTID_ARRAY_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{array, index, value});
    rewriter.eraseOp(op);
    return success();
  }
};

// pto.declare_local_array -> emitc.variable of !emitc.array<...>.
// Renders as `T a[D1][D2]...;` in the emitted C++.
struct PTODeclareLocalArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareLocalArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareLocalArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareLocalArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map !pto.local_array type");

    auto var = rewriter
                   .create<emitc::VariableOp>(
                       op.getLoc(), arrayTy,
                       emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                   .getResult();
    rewriter.replaceOp(op, var);
    return success();
  }
};

// pto.local_array_get %a[%i0, %i1, ...] -> rvalue.
// Lowers to a single emitc.subscript with the full index pack; the C++ emitter
// prints it as `a[i0][i1]...`. The adaptor already exposes target-typed values
// (the type converter has remapped !pto.local_array -> !emitc.array and
// index/integer indices), so they're forwarded directly to the builder.
struct PTOLocalArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(
          op, "failed to map local_array element type");

    auto sub = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), resultTy, adaptor.getArray(), adaptor.getIndices());
    rewriter.replaceOp(op, sub.getResult());
    return success();
  }
};

// pto.local_array_set %a[%i0, %i1, ...], %v -> emitc.assign to subscript slot.
// The C++ emitter prints this as `a[i0][i1]... = v;`. As above, adaptor values
// are already target-typed; pass them through directly.
struct PTOLocalArraySetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArraySetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Type elemTy = value.getType();

    Value slot = rewriter
                     .create<emitc::SubscriptOp>(op.getLoc(), elemTy,
                                                 adaptor.getArray(),
                                                 adaptor.getIndices())
                     .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), slot, value);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePTOToEmitCCommPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx, PTOArch targetArch) {
  patterns.add<PTOInitializeL2G2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOInitializeL2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOBuildAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOAsyncTransferToEmitC<pto::TPutAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncTransferToEmitC<pto::TGetAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOP2PCommToEmitC<pto::TPutOp>>(typeConverter, ctx,
                                               "pto::comm::TPUT");
  patterns.add<PTOP2PCommToEmitC<pto::TGetOp>>(typeConverter, ctx,
                                               "pto::comm::TGET");
  patterns.add<PTOSignalCommToEmitC<pto::TNotifyOp>>(typeConverter, ctx,
                                                     "pto::comm::TNOTIFY");
  patterns.add<PTOSignalCommToEmitC<pto::TWaitOp>>(typeConverter, ctx,
                                                   "pto::comm::TWAIT");
  patterns.add<PTOSignalCommToEmitC<pto::TTestOp>>(typeConverter, ctx,
                                                   "pto::comm::TTEST");
  patterns.add<PTOCommCollectiveToEmitC<pto::TBroadcastOp>>(typeConverter, ctx,
                                                            "TBROADCAST");
  patterns.add<PTOCommCollectiveToEmitC<pto::CommTGatherOp>>(typeConverter, ctx,
                                                             "TGATHER");
  patterns.add<PTOCommCollectiveToEmitC<pto::CommTScatterOp>>(typeConverter, ctx,
                                                              "TSCATTER");
  patterns.add<PTOCommCollectiveToEmitC<pto::TReduceOp>>(typeConverter, ctx,
                                                         "TREDUCE");
  patterns.add<PTOAsyncEventToEmitC<pto::WaitAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_WAIT");
  patterns.add<PTOAsyncEventToEmitC<pto::TestAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_TEST");
  patterns.add<PTODeclareTileMemRefToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareGlobalToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareEventIdArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArraySetToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareLocalArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArraySetToEmitC>(typeConverter, ctx);
}

} // namespace mlir::pto
