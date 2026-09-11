// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- Comm.cpp - ScalarMisc Comm op lowering --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "ScalarMiscInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

template <typename CollectiveOp>
struct PTOCommCollectiveToEmitC : public OpConversionPattern<CollectiveOp> {
  using OpConversionPattern<CollectiveOp>::OpConversionPattern;

  explicit PTOCommCollectiveToEmitC(TypeConverter &typeConverter,
                                    MLIRContext *ctx, StringRef apiName)
      : OpConversionPattern<CollectiveOp>(typeConverter, ctx),
        apiName(apiName.str()) {}

  // Operand bundle for a collective: the main global tensor, the ping tile,
  // the optional pong tile, and the parallel group.
  struct CollectiveOperands {
    FailureOr<Value> mainGT;
    FailureOr<Value> pingTile;
    FailureOr<Value> pongTile;
    FailureOr<Value> parallelGroup;
  };

  // Emit the collective call, appending the pong tile only when present.
  void emitCollectiveCall(CollectiveOp op, ConversionPatternRewriter &rewriter,
                          StringRef callee, const CollectiveOperands &ops) const {
    Location loc = op.getLoc();
    SmallVector<Value> args{*ops.parallelGroup, *ops.mainGT, *ops.pingTile};
    if (succeeded(ops.pongTile))
      args.push_back(*ops.pongTile);
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, callee, ArrayAttr{},
                                         ArrayAttr{}, ValueRange(args));
  }

  // Shared operand resolution for collectives with a (src|dst)-GT + ping/pong
  // tiles + parallel group shape.
  LogicalResult resolvePingPongOperands(
      CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter, Value mainValue, Value mainAdaptor,
      Value pongValue, Value pongAdaptor, StringRef what,
      CollectiveOperands &out) const {
    Location loc = op.getLoc();
    out.mainGT = buildCommGlobalTensorValue(rewriter, loc, mainValue, mainAdaptor,
                                            op.getOperation());
    out.pingTile =
        buildCommTileValue(rewriter, loc, op.getPing(), adaptor.getPing());
    auto groupGTs = buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(),
                                                adaptor.getGroup());
    if (failed(out.mainGT) || failed(out.pingTile) || failed(groupGTs))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize " + what + " operands");
    out.parallelGroup =
        buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
    if (failed(out.parallelGroup))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize " + what + " group");
    if (pongValue)
      out.pongTile =
          buildCommTileValue(rewriter, loc, pongValue, pongAdaptor);
    if (pongValue && failed(out.pongTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
    return success();
  }

  LogicalResult matchAndRewrite(CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    CollectiveOperands ops;
    if constexpr (std::is_same_v<CollectiveOp, pto::TBroadcastOp>) {
      if (failed(resolvePingPongOperands(op, adaptor, rewriter, op.getSrc(),
                                          adaptor.getSrc(), op.getPong(),
                                          adaptor.getPong(), "broadcast", ops)))
        return failure();
      emitCollectiveCall(op, rewriter, "pto::comm::TBROADCAST", ops);
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTGatherOp>) {
      if (failed(resolvePingPongOperands(op, adaptor, rewriter, op.getDst(),
                                          adaptor.getDst(), op.getPong(),
                                          adaptor.getPong(), "gather", ops)))
        return failure();
      emitCollectiveCall(op, rewriter, "pto::comm::TGATHER", ops);
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTScatterOp>) {
      if (failed(resolvePingPongOperands(op, adaptor, rewriter, op.getSrc(),
                                          adaptor.getSrc(), op.getPong(),
                                          adaptor.getPong(), "scatter", ops)))
        return failure();
      emitCollectiveCall(op, rewriter, "pto::comm::TSCATTER", ops);
    } else {
      return matchAndRewriteReduce(op, adaptor, rewriter);
    }
    rewriter.eraseOp(op);
    return success();
  }

  // TREDUCE carries an extra ReduceOp constant plus acc/recvPing/recvPong tiles.
  LogicalResult
  matchAndRewriteReduce(CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    FailureOr<Value> dstGT = buildCommGlobalTensorValue(
        rewriter, loc, op.getDst(), adaptor.getDst(), op.getOperation());
    FailureOr<Value> accTile =
        buildCommTileValue(rewriter, loc, op.getAcc(), adaptor.getAcc());
    FailureOr<Value> recvPing =
        buildCommTileValue(rewriter, loc, op.getRecvPing(), adaptor.getRecvPing());
    auto groupGTs = buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(),
                                                adaptor.getGroup());
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
    SmallVector<Value> args{*pg, *dstGT, *accTile, *recvPing};
    if (op.getRecvPong()) {
      FailureOr<Value> recvPong =
          buildCommTileValue(rewriter, loc, op.getRecvPong(), adaptor.getRecvPong());
      if (failed(recvPong))
        return rewriter.notifyMatchFailure(op, "failed to materialize recv_pong");
      args.push_back(*recvPong);
    }
    args.push_back(reduceOp);
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "pto::comm::TREDUCE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(args));
    rewriter.eraseOp(op);
    return success();
  }

  std::string apiName;
};

template <typename OpTy>
struct PTOP2PCommToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTOP2PCommToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                             StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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

  explicit PTOSignalCommToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<SignalOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(SignalOp op, typename SignalOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> signalGT = buildCommGlobalTensorValue(
        rewriter, op.getLoc(), op.getSignal(), adaptor.getSignal(), op.getOperation());
    if (failed(signalGT))
      return rewriter.notifyMatchFailure(op, "failed to materialize signal operand");

    if constexpr (std::is_same_v<SignalOp, pto::TNotifyOp>) {
      auto notifyTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::NotifyOp");
      Value notifyOp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), notifyTy, notifyOpTok(op.getNotifyOp()));
      SmallVector<Value> operands{*signalGT, adaptor.getValue(),
                                  notifyOp};
      // See emitTNotifyReleaseActions comment: drain in-flight MTE work before the
      // scalar-pipe signal store so the notify/wait handshake is honored.
      bool drainMte2 = op->hasAttr(kTNotifyDrainMte2AttrName);
      bool drainMte3 = op->hasAttr(kTNotifyDrainMte3AttrName);
      emitTNotifyReleaseActions(rewriter, op.getLoc(), drainMte2, drainMte3);
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{}, operands);
      rewriter.eraseOp(op);
    } else {
      auto waitCmpTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::WaitCmp");
      Value waitCmp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), waitCmpTy, waitCmpTok(op.getCmp()));
      SmallVector<Value> operands{*signalGT, adaptor.getCmpValue(),
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

void populateScalarMiscCommPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
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
}

} // namespace pto
} // namespace mlir
