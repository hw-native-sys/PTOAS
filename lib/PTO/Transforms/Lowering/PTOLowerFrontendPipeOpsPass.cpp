// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <optional>
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"


namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERFRONTENDPIPEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr int8_t kC2VDirMask = 1;
constexpr int8_t kV2CDirMask = 2;
constexpr int8_t kBidirectionalDirMask = 3;
constexpr int32_t kSingleDirectionSlotNum = 8;
constexpr int32_t kBidirectionalSlotNum = 4;
constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";
constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
constexpr llvm::StringLiteral kPipePeerDirMaskAttrName = "__pto.peer_dir_mask";
constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";

struct FrontendPipeHandles {
  Value c2vPipe;
  Value v2cPipe;
  SmallVector<int64_t> c2vSlotStrides;
  SmallVector<int64_t> v2cSlotStrides;
  Operation *anchorOp = nullptr;
};

using FrontendPipeHandleMap = llvm::DenseMap<int32_t, FrontendPipeHandles>;

// Shared parameters describing the PTO pipe to create for a frontend
// initialize op.
struct FrontendPipeSpec {
  PTOArch arch;
  Type pipeTy;
  int8_t dirMask;
  int32_t slotNum;
};

template <typename InitOpT>
static LogicalResult requireFrontendGmSlotBuffer(InitOpT initOp) {
  if (initOp.getGmSlotBuffer()) {
    return success();
  }
  return initOp.emitOpError("requires 'gm_slot_buffer' when lowering to a2/a3");
}

template <typename InitOpT>
static void propagateFrontendIdAttr(InitOpT initOp, Operation *pipeOp,
                                    IRRewriter &rewriter) {
  if (!pipeOp) {
    return;
  }
  pipeOp->setAttr(kFrontendPipeIdAttrName,
                  rewriter.getI32IntegerAttr(initOp.getId()));
}

template <typename InitOpT>
static void propagateFixpipePeerKeyAttrs(InitOpT initOp, Operation *pipeOp,
                                         IRRewriter &rewriter) {
  if (!pipeOp || !initOp.getAccPushEpilogueAttr() ||
      initOp.getDirMask() != kC2VDirMask || !initOp.getC2vConsumerBuf()) {
    return;
  }

  auto currentFunc = initOp->template getParentOfType<func::FuncOp>();
  if (!currentFunc) {
    return;
  }

  auto setPeerKeyAttrs = [&rewriter, pipeOp](FlatSymbolRefAttr ownerFuncAttr,
                                              StringRef reserveName) {
    pipeOp->setAttr(kPipePeerOwnerFuncAttrName, ownerFuncAttr);
    pipeOp->setAttr(kPipePeerReserveNameAttrName,
                    rewriter.getStringAttr(reserveName));
    pipeOp->setAttr(kPipePeerDirMaskAttrName,
                    rewriter.getI8IntegerAttr(kC2VDirMask));
  };

  if (auto reserveOp =
          initOp.getC2vConsumerBuf().template getDefiningOp<ReserveBufferOp>()) {
    setPeerKeyAttrs(FlatSymbolRefAttr::get(currentFunc), reserveOp.getName());
    return;
  }

  if (auto importOp = initOp.getC2vConsumerBuf()
                          .template getDefiningOp<ImportReservedBufferOp>()) {
    auto peerFunc = lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                                  importOp.getPeerFuncAttr());
    if (!peerFunc) {
      return;
    }
    setPeerKeyAttrs(FlatSymbolRefAttr::get(peerFunc), importOp.getName());
  }
}

template <typename InitOpT>
static int32_t getFrontendSlotNum(InitOpT initOp) {
  if (auto slotNumAttr = initOp.getSlotNumAttr()) {
    return slotNumAttr.getInt();
  }
  return initOp.getDirMask() == kBidirectionalDirMask
             ? kBidirectionalSlotNum
             : kSingleDirectionSlotNum;
}

static std::optional<int64_t> getStaticIndexLikeValue(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return cst.value();
  }
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

static SmallVector<int64_t> getStaticTensorViewStrides(Value tensor) {
  SmallVector<int64_t> strides;
  if (!tensor) {
    return strides;
  }

  auto makeView = tensor.getDefiningOp<MakeTensorViewOp>();
  if (!makeView) {
    return strides;
  }

  auto tvTy = dyn_cast<TensorViewType>(makeView.getResult().getType());
  if (!tvTy ||
      makeView.getStrides().size() != static_cast<size_t>(tvTy.getRank())) {
    return {};
  }

  strides.reserve(makeView.getStrides().size());
  for (Value stride : makeView.getStrides()) {
    auto staticStride = getStaticIndexLikeValue(stride);
    if (!staticStride) {
      return {};
    }
    strides.push_back(*staticStride);
  }
  return strides;
}

static void propagateGlobalTensorStrides(DeclareGlobalOp decl,
                                         ArrayRef<int64_t> strides,
                                         IRRewriter &rewriter) {
  if (strides.empty()) {
    return;
  }
  decl->setAttr(kGlobalTensorStridesAttrName,
                rewriter.getDenseI64ArrayAttr(strides));
}

template <typename InitOpT>
static FailureOr<Value> createFrontendGlobalTensorPipe(
    InitOpT initOp, IRRewriter &rewriter, const FrontendPipeSpec &spec,
    Value localAddr = Value{}, Value peerLocalAddr = Value{}) {
  Location loc = initOp.getLoc();
  auto dirAttr = rewriter.getI8IntegerAttr(spec.dirMask);
  auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
  auto slotNumAttr = rewriter.getI32IntegerAttr(spec.slotNum);
  auto noSplitAttr = initOp.getNosplitAttr();
  auto accPushEpilogueAttr = initOp.getAccPushEpilogueAttr();
  IntegerAttr localSlotNumAttr;
  if (localAddr) {
    localSlotNumAttr = initOp.getLocalSlotNumAttr();
    if (!localSlotNumAttr) {
      localSlotNumAttr = rewriter.getI32IntegerAttr(spec.slotNum);
    }
  }
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, spec.pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
      IntegerAttr{}, noSplitAttr, accPushEpilogueAttr, initOp.getGmSlotTensor(),
      localAddr, peerLocalAddr);
  propagateFrontendIdAttr(initOp, pipe.getOperation(), rewriter);
  propagateFixpipePeerKeyAttrs(initOp, pipe.getOperation(), rewriter);
  return pipe.getPipe();
}

template <typename InitOpT>
static FailureOr<Value> createFrontendLocalPipe(InitOpT initOp,
                                                IRRewriter &rewriter,
                                                const FrontendPipeSpec &spec,
                                                Value localAddr,
                                                Value peerLocalAddr = Value{}) {
  Location loc = initOp.getLoc();
  auto dirAttr = rewriter.getI8IntegerAttr(spec.dirMask);
  auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
  auto slotNumAttr = rewriter.getI32IntegerAttr(spec.slotNum);
  auto noSplitAttr = initOp.getNosplitAttr();
  auto accPushEpilogueAttr = initOp.getAccPushEpilogueAttr();

  if (spec.arch == PTOArch::A5) {
    if (!localAddr) {
      return initOp.emitOpError(
          "requires local consumer buffer operands when lowering to a5");
    }
    auto pipe = rewriter.create<InitializeL2LPipeOp>(
        loc, spec.pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
        noSplitAttr, accPushEpilogueAttr, localAddr, peerLocalAddr);
    propagateFrontendIdAttr(initOp, pipe.getOperation(), rewriter);
    propagateFixpipePeerKeyAttrs(initOp, pipe.getOperation(), rewriter);
    return pipe.getPipe();
  }

  if (failed(requireFrontendGmSlotBuffer(initOp))) {
    return failure();
  }
  if (!localAddr) {
    return initOp.emitOpError(
        "requires local consumer buffer operands for local FIFO pipe lowering");
  }

  IntegerAttr localSlotNumAttr = initOp.getLocalSlotNumAttr();
  if (!localSlotNumAttr) {
    localSlotNumAttr = rewriter.getI32IntegerAttr(spec.slotNum);
  }
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, spec.pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
      IntegerAttr{}, noSplitAttr, accPushEpilogueAttr,
      initOp.getGmSlotBuffer(), localAddr, peerLocalAddr);
  propagateFrontendIdAttr(initOp, pipe.getOperation(), rewriter);
  propagateFixpipePeerKeyAttrs(initOp, pipe.getOperation(), rewriter);
  return pipe.getPipe();
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerSingleDirectionFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                                 const FrontendPipeSpec &spec,
                                 Value localAddr) {
  auto pipeOr = initOp.getGmSlotTensor()
                    ? createFrontendGlobalTensorPipe(initOp, rewriter, spec,
                                                     localAddr)
                    : createFrontendLocalPipe(initOp, rewriter, spec,
                                              localAddr);
  if (failed(pipeOr)) {
    return failure();
  }

  FrontendPipeHandles handles;
  SmallVector<int64_t> slotStrides =
      getStaticTensorViewStrides(initOp.getGmSlotTensor());
  if (spec.dirMask == kC2VDirMask) {
    handles.c2vPipe = *pipeOr;
    handles.c2vSlotStrides = std::move(slotStrides);
  } else {
    handles.v2cPipe = *pipeOr;
    handles.v2cSlotStrides = std::move(slotStrides);
  }
  handles.anchorOp = pipeOr->getDefiningOp();
  return handles;
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerBidirectionalFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                               const FrontendPipeSpec &spec) {
  auto pipeOr = initOp.getGmSlotTensor()
                    ? createFrontendGlobalTensorPipe(
                          initOp, rewriter, spec, initOp.getC2vConsumerBuf(),
                          initOp.getV2cConsumerBuf())
                    : createFrontendLocalPipe(initOp, rewriter, spec,
                                              initOp.getC2vConsumerBuf(),
                                              initOp.getV2cConsumerBuf());
  if (failed(pipeOr)) {
    return failure();
  }

  FrontendPipeHandles handles;
  handles.c2vPipe = *pipeOr;
  handles.v2cPipe = *pipeOr;
  SmallVector<int64_t> slotStrides =
      getStaticTensorViewStrides(initOp.getGmSlotTensor());
  handles.c2vSlotStrides = slotStrides;
  handles.v2cSlotStrides = std::move(slotStrides);
  handles.anchorOp = pipeOr->getDefiningOp();
  return handles;
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerFrontendInitOp(InitOpT initOp,
                                                          IRRewriter &rewriter) {
  MLIRContext *ctx = initOp.getContext();
  FrontendPipeSpec spec{getTargetArch(initOp.getOperation()),
                        PipeType::get(ctx),
                        static_cast<int8_t>(initOp.getDirMask()),
                        getFrontendSlotNum(initOp)};

  switch (initOp.getDirMask()) {
  case kC2VDirMask:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, spec,
                                            initOp.getC2vConsumerBuf());
  case kV2CDirMask:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, spec,
                                            initOp.getV2cConsumerBuf());
  case kBidirectionalDirMask:
    return lowerBidirectionalFrontendInit(initOp, rewriter, spec);
  default:
    return FrontendPipeHandles{};
  }
}

template <typename InitOpT>
static void propagateFrontendNoSplitAttr(InitOpT initOp,
                                         const FrontendPipeHandles &handles) {
  auto noSplitAttr = initOp.getNosplitAttr();
  if (!noSplitAttr) {
    return;
  }

  if (handles.anchorOp) {
    handles.anchorOp->setAttr("nosplit", noSplitAttr);
  }

  Operation *c2vOp =
      handles.c2vPipe ? handles.c2vPipe.getDefiningOp() : nullptr;
  Operation *v2cOp =
      handles.v2cPipe ? handles.v2cPipe.getDefiningOp() : nullptr;

  if (c2vOp && c2vOp != handles.anchorOp) {
    c2vOp->setAttr("nosplit", noSplitAttr);
  }
  if (v2cOp && v2cOp != handles.anchorOp && v2cOp != c2vOp) {
    v2cOp->setAttr("nosplit", noSplitAttr);
  }
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerAndEraseFrontendInit(InitOpT initOp,
                                                                IRRewriter &rewriter) {
  rewriter.setInsertionPoint(initOp);
  auto loweredOr = lowerFrontendInitOp(initOp, rewriter);
  if (failed(loweredOr)) {
    return failure();
  }
  propagateFrontendNoSplitAttr(initOp, *loweredOr);
  rewriter.eraseOp(initOp);
  return *loweredOr;
}

// Collects the frontend initialize ops in `funcOp`, diagnosing duplicate ids.
// Returns false when a duplicate id was found.
static bool collectFrontendInitOps(func::FuncOp funcOp,
                                   SmallVectorImpl<Operation *> &initOps,
                                   llvm::DenseMap<int32_t, Operation *> &seen) {
  // Aic and Aiv initialize ops share the id-uniqueness contract, so both
  // variants run through one shared record-and-check helper.
  auto recordInitOp = [&](Operation *op, int32_t id) {
    initOps.push_back(op);
    auto [it, inserted] = seen.try_emplace(id, op);
    (void)it;
    if (!inserted) {
      op->emitOpError()
          << "requires unique initialize_pipe id in function (duplicate id = "
          << id << ")";
    }
    return inserted;
  };
  bool allUnique = true;
  funcOp.walk([&](Operation *op) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      allUnique &= recordInitOp(op, init.getId());
    } else if (auto init = dyn_cast<AivInitializePipeOp>(op)) {
      allUnique &= recordInitOp(op, init.getId());
    }
  });
  return allUnique;
}

// Lowers one collected frontend initialize op (aic or aiv variant) and
// registers its handles under the op's frontend id.
static LogicalResult
lowerCollectedInitOp(Operation *op, IRRewriter &rewriter,
                     FrontendPipeHandleMap &handlesById) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](AicInitializePipeOp init) {
        auto loweredOr = lowerAndEraseFrontendInit(init, rewriter);
        if (failed(loweredOr)) {
          return failure();
        }
        handlesById.try_emplace(init.getId(), *loweredOr);
        return success();
      })
      .Case([&](AivInitializePipeOp init) {
        auto loweredOr = lowerAndEraseFrontendInit(init, rewriter);
        if (failed(loweredOr)) {
          return failure();
        }
        handlesById.try_emplace(init.getId(), *loweredOr);
        return success();
      })
      .Default([](Operation *) { return success(); });
}

static FailureOr<FrontendPipeHandleMap> lowerInitIfPresent(func::FuncOp funcOp,
                                                           IRRewriter &rewriter) {
  FrontendPipeHandleMap handlesById;
  SmallVector<Operation *> frontendInitOps;
  llvm::DenseMap<int32_t, Operation *> initOpById;
  bool hasAicInit = false;
  bool hasAivInit = false;

  funcOp.walk([&](Operation *op) {
    if (isa<AicInitializePipeOp>(op)) {
      hasAicInit = true;
    } else if (isa<AivInitializePipeOp>(op)) {
      hasAivInit = true;
    }
  });

  if (!collectFrontendInitOps(funcOp, frontendInitOps, initOpById)) {
    return failure();
  }

  if (hasAicInit && hasAivInit) {
    funcOp.emitOpError("cannot mix pto.aic_initialize_pipe and "
                       "pto.aiv_initialize_pipe in one function");
    return failure();
  }

  for (Operation *op : frontendInitOps) {
    if (failed(lowerCollectedInitOp(op, rewriter, handlesById))) {
      return failure();
    }
  }

  return handlesById;
}

static bool hasFrontendPipeOps(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<AicInitializePipeOp, AivInitializePipeOp, TAllocToAivOp,
            TAllocToAicOp, TPushToAivOp, TPushToAicOp, TPopFromAicOp,
            TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Shared context for the per-family frontend data-op lowering helpers.
struct FrontendDataLowering {
  DominanceInfo &dom;
  const FrontendPipeHandleMap &handlesById;
};

// Looks up the pipe handles registered for `id` and verifies that the
// initializing op dominates `op`.
static FailureOr<const FrontendPipeHandles *>
lookupFrontendPipeHandles(Operation *op, int32_t id,
                          const FrontendDataLowering &ctx) {
  auto it = ctx.handlesById.find(id);
  if (it == ctx.handlesById.end()) {
    op->emitOpError()
        << "requires matching frontend initialize_pipe(id = " << id
        << ") in the same function";
    return failure();
  }
  const FrontendPipeHandles &handles = it->second;
  if (!handles.anchorOp || !ctx.dom.dominates(handles.anchorOp, op)) {
    op->emitOpError()
        << "requires dominating frontend initialize_pipe(id = " << id << ")";
    return failure();
  }
  return &handles;
}

// Resolves the pipe handles for `op`'s frontend id, additionally requiring
// the direction `pipe` (c2vPipe/v2cPipe) to be enabled.
template <typename OpT>
static FailureOr<const FrontendPipeHandles *>
resolveFrontendPipe(OpT op, Value FrontendPipeHandles::*pipe,
                    StringRef dirName, const FrontendDataLowering &ctx) {
  auto handlesOr = lookupFrontendPipeHandles(op, op.getId(), ctx);
  if (failed(handlesOr)) {
    return failure();
  }
  const FrontendPipeHandles &handles = **handlesOr;
  if (!(handles.*pipe)) {
    op->emitOpError() << "requires initialize_pipe(id = " << op.getId()
                      << ") to enable " << dirName;
    return failure();
  }
  return &handles;
}

// Allocates the global buffer that receives a popped tensor-view value and
// propagates the frontend slot strides.
static Value createGlobalPopDestination(Value tile, Location loc,
                                        ArrayRef<int64_t> slotStrides,
                                        IRRewriter &rewriter) {
  auto decl = rewriter.create<DeclareGlobalOp>(loc, tile.getType());
  propagateGlobalTensorStrides(decl, slotStrides, rewriter);
  return decl.getEntry();
}

// Allocates the local tile buffer that receives a popped tile value and
// records its valid shape when present.
template <typename PopOpT>
static Value createTilePopDestination(PopOpT pop, IRRewriter &rewriter) {
  auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                             pop.getTile().getType());
  Value entry = decl.getTile();
  if (pop.getValidRow() && pop.getValidCol()) {
    rewriter.create<SetValidShapeOp>(pop.getLoc(), entry, pop.getValidRow(),
                                     pop.getValidCol());
  }
  return entry;
}

// Allocates the destination buffer for a popped value (global for tensor
// views, local tile otherwise).
static Value createPopDestination(TPopFromAicOp pop,
                                  const FrontendPipeHandles &handles,
                                  IRRewriter &rewriter) {
  if (isa<TensorViewType>(pop.getTile().getType())) {
    return createGlobalPopDestination(pop.getTile(), pop.getLoc(),
                                      handles.c2vSlotStrides, rewriter);
  }
  return createTilePopDestination(pop, rewriter);
}

// Same as createPopDestination above, for the aiv (V2C) pop variant.
static Value createPopDestination(TPopFromAivOp pop,
                                  const FrontendPipeHandles &handles,
                                  IRRewriter &rewriter) {
  if (isa<TensorViewType>(pop.getTile().getType())) {
    return createGlobalPopDestination(pop.getTile(), pop.getLoc(),
                                      handles.v2cSlotStrides, rewriter);
  }
  return createTilePopDestination(pop, rewriter);
}

// Shared lowering for the alloc-to-pipe variants: a c2v (cube-to-vec) pipe
// uses the c2v slot strides and pipe, a v2c pipe the v2c ones. `pipeSel`
// picks the handle members; `dirName` names the pipe in diagnostics.
template <typename OpTy>
static LogicalResult lowerFrontendAllocOp(
    OpTy op, Value FrontendPipeHandles::*pipeSel,
    const SmallVector<int64_t> FrontendPipeHandles::*strideSel,
    const char *dirName, const FrontendDataLowering &ctx, IRRewriter &rewriter) {
  auto handlesOr = resolveFrontendPipe(op, pipeSel, dirName, ctx);
  if (failed(handlesOr)) {
    return failure();
  }
  const FrontendPipeHandles &handles = **handlesOr;
  auto decl =
      rewriter.create<DeclareGlobalOp>(op.getLoc(), op.getEntry().getType());
  propagateGlobalTensorStrides(decl, handles.*strideSel, rewriter);
  rewriter.create<TAllocOp>(op.getLoc(), decl.getEntry(), handles.*pipeSel,
                            op.getSplitAttr());
  rewriter.replaceOp(op, decl.getEntry());
  return success();
}

// Shared lowering for the push/pop/free data ops: resolves the pipe handle
// (c2v or v2c via `pipeSel`/`dirName`), then applies `lower` with it.
template <typename OpTy, typename LowerFn>
static LogicalResult lowerFrontendDataWithPipe(
    OpTy op, Value FrontendPipeHandles::*pipeSel, const char *dirName,
    const FrontendDataLowering &ctx, IRRewriter &rewriter, LowerFn &&lower) {
  auto handlesOr = resolveFrontendPipe(op, pipeSel, dirName, ctx);
  if (failed(handlesOr)) {
    return failure();
  }
  return lower(**handlesOr);
}

// A push lowers to a plain pipe push, with an optional subblock id.
template <typename OpTy>
static LogicalResult
lowerFrontendPushOp(OpTy push, Value FrontendPipeHandles::*pipeSel,
                    const char *dirName, Value subblockid,
                    const FrontendDataLowering &ctx, IRRewriter &rewriter) {
  return lowerFrontendDataWithPipe(
      push, pipeSel, dirName, ctx, rewriter,
      [&](const FrontendPipeHandles &handles) {
        rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(),
                                             handles.*pipeSel, subblockid,
                                             push.getSplitAttr());
        return success();
      });
}

// A pop allocates its destination tile, then moves it over the pipe.
template <typename OpTy>
static LogicalResult
lowerFrontendPopOp(OpTy pop, Value FrontendPipeHandles::*pipeSel,
                   const char *dirName, Value subblockid,
                   const FrontendDataLowering &ctx, IRRewriter &rewriter) {
  return lowerFrontendDataWithPipe(
      pop, pipeSel, dirName, ctx, rewriter,
      [&](const FrontendPipeHandles &handles) {
        Value entry = createPopDestination(pop, handles, rewriter);
        rewriter.create<TPopOp>(pop.getLoc(), entry, handles.*pipeSel,
                                subblockid, pop.getSplitAttr());
        rewriter.replaceOp(pop, entry);
        return success();
      });
}

// A free releases the pipe slot backing the entry.
template <typename OpTy>
static LogicalResult
lowerFrontendFreeOp(OpTy free, Value FrontendPipeHandles::*pipeSel,
                    const char *dirName, const FrontendDataLowering &ctx,
                    IRRewriter &rewriter) {
  return lowerFrontendDataWithPipe(
      free, pipeSel, dirName, ctx, rewriter,
      [&](const FrontendPipeHandles &handles) {
        rewriter.replaceOpWithNewOp<TFreeOp>(free, free.getEntry(),
                                             handles.*pipeSel,
                                             free.getSplitAttr());
        return success();
      });
}

// Replaces one frontend alloc/push/pop/free op with its PTO pipe op. The
// insertion point must already be set on `rewriter`.
static LogicalResult lowerOneFrontendDataOp(Operation *op,
                                            const FrontendDataLowering &ctx,
                                            IRRewriter &rewriter) {
  using PipeSel = Value FrontendPipeHandles::*;
  using StrideSel = const SmallVector<int64_t> FrontendPipeHandles::*;
  constexpr PipeSel kC2vPipe = &FrontendPipeHandles::c2vPipe;
  constexpr PipeSel kV2cPipe = &FrontendPipeHandles::v2cPipe;
  constexpr StrideSel kC2vStrides = &FrontendPipeHandles::c2vSlotStrides;
  constexpr StrideSel kV2cStrides = &FrontendPipeHandles::v2cSlotStrides;

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](TAllocToAivOp alloc) {
        return lowerFrontendAllocOp(alloc, kC2vPipe, kC2vStrides, "C2V", ctx,
                                    rewriter);
      })
      .Case([&](TAllocToAicOp alloc) {
        return lowerFrontendAllocOp(alloc, kV2cPipe, kV2cStrides, "V2C", ctx,
                                    rewriter);
      })
      .Case([&](TPushToAivOp push) {
        return lowerFrontendPushOp(push, kC2vPipe, "C2V", Value{}, ctx,
                                   rewriter);
      })
      .Case([&](TPushToAicOp push) {
        return lowerFrontendPushOp(push, kV2cPipe, "V2C",
                                   push.getAivSubblockid(), ctx, rewriter);
      })
      .Case([&](TPopFromAicOp pop) {
        return lowerFrontendPopOp(pop, kC2vPipe, "C2V",
                                  pop.getAivSubblockid(), ctx, rewriter);
      })
      .Case([&](TPopFromAivOp pop) {
        return lowerFrontendPopOp(pop, kV2cPipe, "V2C", Value{}, ctx,
                                  rewriter);
      })
      .Case([&](TFreeFromAicOp free) {
        return lowerFrontendFreeOp(free, kC2vPipe, "C2V", ctx, rewriter);
      })
      .Case([&](TFreeFromAivOp free) {
        return lowerFrontendFreeOp(free, kV2cPipe, "V2C", ctx, rewriter);
      })
      .Default([](Operation *) { return success(); });
}

static LogicalResult lowerFrontendDataOps(func::FuncOp funcOp,
                                          const FrontendPipeHandleMap &handlesById,
                                          IRRewriter &rewriter) {
  DominanceInfo dom(funcOp);
  FrontendDataLowering ctx{dom, handlesById};

  SmallVector<Operation *> frontendOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TAllocToAivOp, TAllocToAicOp, TPushToAivOp, TPushToAicOp,
            TPopFromAicOp, TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp>(op)) {
      frontendOps.push_back(op);
    }
  });

  for (Operation *op : frontendOps) {
    rewriter.setInsertionPoint(op);
    if (failed(lowerOneFrontendDataOp(op, ctx, rewriter))) {
      return failure();
    }
  }
  return success();
}

struct PTOLowerFrontendPipeOpsPass
    : public mlir::pto::impl::PTOLowerFrontendPipeOpsBase<
          PTOLowerFrontendPipeOpsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasFrontendPipeOps(funcOp)) {
      return;
    }

    IRRewriter rewriter(funcOp.getContext());
    auto loweredOr = lowerInitIfPresent(funcOp, rewriter);
    if (failed(loweredOr)) {
      signalPassFailure();
      return;
    }

    if (failed(lowerFrontendDataOps(funcOp, *loweredOr, rewriter))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerFrontendPipeOpsPass() {
  return std::make_unique<PTOLowerFrontendPipeOpsPass>();
}
