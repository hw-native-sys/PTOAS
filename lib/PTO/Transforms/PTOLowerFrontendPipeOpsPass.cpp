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
static FailureOr<Value> createFrontendGlobalTensorPipe(InitOpT initOp,
                                                       IRRewriter &rewriter,
                                                       Type pipeTy,
                                                       int8_t dirMask,
                                                       int32_t slotNum,
                                                       Value localAddr = Value{},
                                                       Value peerLocalAddr =
                                                           Value{}) {
  Location loc = initOp.getLoc();
  auto dirAttr = rewriter.getI8IntegerAttr(dirMask);
  auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
  auto slotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  auto noSplitAttr = initOp.getNosplitAttr();
  auto accPushEpilogueAttr = initOp.getAccPushEpilogueAttr();
  IntegerAttr localSlotNumAttr;
  if (localAddr) {
    localSlotNumAttr = initOp.getLocalSlotNumAttr();
    if (!localSlotNumAttr) {
      localSlotNumAttr = rewriter.getI32IntegerAttr(slotNum);
    }
  }
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
      IntegerAttr{}, noSplitAttr, accPushEpilogueAttr, initOp.getGmSlotTensor(),
      localAddr, peerLocalAddr);
  propagateFrontendIdAttr(initOp, pipe.getOperation(), rewriter);
  propagateFixpipePeerKeyAttrs(initOp, pipe.getOperation(), rewriter);
  return pipe.getPipe();
}

template <typename InitOpT>
static FailureOr<Value> createFrontendLocalPipe(InitOpT initOp,
                                                IRRewriter &rewriter,
                                                PTOArch arch, Type pipeTy,
                                                int8_t dirMask,
                                                int32_t slotNum,
                                                Value localAddr,
                                                Value peerLocalAddr = Value{}) {
  Location loc = initOp.getLoc();
  auto dirAttr = rewriter.getI8IntegerAttr(dirMask);
  auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
  auto slotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  auto noSplitAttr = initOp.getNosplitAttr();
  auto accPushEpilogueAttr = initOp.getAccPushEpilogueAttr();

  if (arch == PTOArch::A5) {
    if (!localAddr) {
      return initOp.emitOpError(
          "requires local consumer buffer operands when lowering to a5");
    }
    auto pipe = rewriter.create<InitializeL2LPipeOp>(
        loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
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
    localSlotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  }
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
      IntegerAttr{}, noSplitAttr, accPushEpilogueAttr, initOp.getGmSlotBuffer(),
      localAddr, peerLocalAddr);
  propagateFrontendIdAttr(initOp, pipe.getOperation(), rewriter);
  propagateFixpipePeerKeyAttrs(initOp, pipe.getOperation(), rewriter);
  return pipe.getPipe();
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerSingleDirectionFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                                 PTOArch arch, Type pipeTy, int8_t dirMask,
                                 Value localAddr) {
  int32_t slotNum = getFrontendSlotNum(initOp);
  auto pipeOr = initOp.getGmSlotTensor()
                    ? createFrontendGlobalTensorPipe(initOp, rewriter, pipeTy,
                                                     dirMask, slotNum,
                                                     localAddr)
                    : createFrontendLocalPipe(initOp, rewriter, arch, pipeTy,
                                              dirMask, slotNum, localAddr);
  if (failed(pipeOr)) {
    return failure();
  }

  FrontendPipeHandles handles;
  SmallVector<int64_t> slotStrides =
      getStaticTensorViewStrides(initOp.getGmSlotTensor());
  if (dirMask == kC2VDirMask) {
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
                               PTOArch arch, Type pipeTy) {
  int32_t slotNum = getFrontendSlotNum(initOp);
  auto pipeOr = initOp.getGmSlotTensor()
                    ? createFrontendGlobalTensorPipe(
                          initOp, rewriter, pipeTy, kBidirectionalDirMask,
                          slotNum, initOp.getC2vConsumerBuf(),
                          initOp.getV2cConsumerBuf())
                    : createFrontendLocalPipe(initOp, rewriter, arch, pipeTy,
                                              kBidirectionalDirMask, slotNum,
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
  auto pipeTy = PipeType::get(ctx);
  PTOArch arch = getTargetArch(initOp.getOperation());

  switch (initOp.getDirMask()) {
  case kC2VDirMask:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, arch, pipeTy,
                                            kC2VDirMask,
                                            initOp.getC2vConsumerBuf());
  case kV2CDirMask:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, arch, pipeTy,
                                            kV2CDirMask,
                                            initOp.getV2cConsumerBuf());
  case kBidirectionalDirMask:
    return lowerBidirectionalFrontendInit(initOp, rewriter, arch, pipeTy);
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

static FailureOr<FrontendPipeHandleMap> lowerInitIfPresent(func::FuncOp funcOp,
                                                           IRRewriter &rewriter) {
  FrontendPipeHandleMap handlesById;
  SmallVector<Operation *> frontendInitOps;
  llvm::DenseMap<int32_t, Operation *> initOpById;
  bool hasDuplicateId = false;
  bool hasAicInit = false;
  bool hasAivInit = false;

  funcOp.walk([&](Operation *op) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      hasAicInit = true;
      frontendInitOps.push_back(op);
      auto [it, inserted] = initOpById.try_emplace(init.getId(), op);
      if (!inserted) {
        op->emitOpError()
            << "requires unique initialize_pipe id in function (duplicate id = "
            << init.getId() << ")";
        hasDuplicateId = true;
      }
      return WalkResult::advance();
    }
    if (auto init = dyn_cast<AivInitializePipeOp>(op)) {
      hasAivInit = true;
      frontendInitOps.push_back(op);
      auto [it, inserted] = initOpById.try_emplace(init.getId(), op);
      if (!inserted) {
        op->emitOpError()
            << "requires unique initialize_pipe id in function (duplicate id = "
            << init.getId() << ")";
        hasDuplicateId = true;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (hasDuplicateId) {
    return failure();
  }

  if (hasAicInit && hasAivInit) {
    funcOp.emitOpError("cannot mix pto.aic_initialize_pipe and "
                       "pto.aiv_initialize_pipe in one function");
    return failure();
  }

  for (Operation *op : frontendInitOps) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      int32_t id = init.getId();
      auto loweredOr = lowerAndEraseFrontendInit(init, rewriter);
      if (failed(loweredOr)) {
        return failure();
      }
      handlesById.try_emplace(id, *loweredOr);
      continue;
    }

    auto init = cast<AivInitializePipeOp>(op);
    int32_t id = init.getId();
    auto loweredOr = lowerAndEraseFrontendInit(init, rewriter);
    if (failed(loweredOr)) {
      return failure();
    }
    handlesById.try_emplace(id, *loweredOr);
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

static LogicalResult lowerFrontendDataOps(func::FuncOp funcOp,
                                          const FrontendPipeHandleMap &handlesById,
                                          IRRewriter &rewriter) {
  DominanceInfo dom(funcOp);
  SmallVector<Operation *> frontendOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TAllocToAivOp, TAllocToAicOp, TPushToAivOp, TPushToAicOp,
            TPopFromAicOp, TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp>(op)) {
      frontendOps.push_back(op);
    }
  });

  auto lookupHandles = [&](Operation *op, int32_t id)
      -> FailureOr<const FrontendPipeHandles *> {
    auto it = handlesById.find(id);
    if (it == handlesById.end()) {
      op->emitOpError()
          << "requires matching frontend initialize_pipe(id = " << id
          << ") in the same function";
      return failure();
    }
    const FrontendPipeHandles &handles = it->second;
    if (!handles.anchorOp || !dom.dominates(handles.anchorOp, op)) {
      op->emitOpError()
          << "requires dominating frontend initialize_pipe(id = " << id << ")";
      return failure();
    }
    return &handles;
  };

  for (Operation *op : frontendOps) {
    rewriter.setInsertionPoint(op);

    if (auto alloc = dyn_cast<TAllocToAivOp>(op)) {
      auto handlesOr = lookupHandles(op, alloc.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << alloc.getId()
                          << ") to enable C2V";
        return failure();
      }
      auto decl = rewriter.create<DeclareGlobalOp>(alloc.getLoc(),
                                                   alloc.getEntry().getType());
      propagateGlobalTensorStrides(decl, handles.c2vSlotStrides, rewriter);
      rewriter.create<TAllocOp>(alloc.getLoc(), decl.getEntry(),
                                handles.c2vPipe, alloc.getSplitAttr());
      rewriter.replaceOp(alloc, decl.getEntry());
      continue;
    }

    if (auto alloc = dyn_cast<TAllocToAicOp>(op)) {
      auto handlesOr = lookupHandles(op, alloc.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.v2cPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << alloc.getId()
                          << ") to enable V2C";
        return failure();
      }
      auto decl = rewriter.create<DeclareGlobalOp>(alloc.getLoc(),
                                                   alloc.getEntry().getType());
      propagateGlobalTensorStrides(decl, handles.v2cSlotStrides, rewriter);
      rewriter.create<TAllocOp>(alloc.getLoc(), decl.getEntry(),
                                handles.v2cPipe, alloc.getSplitAttr());
      rewriter.replaceOp(alloc, decl.getEntry());
      continue;
    }

    if (auto push = dyn_cast<TPushToAivOp>(op)) {
      auto handlesOr = lookupHandles(op, push.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << push.getId()
                          << ") to enable C2V";
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.c2vPipe,
                                           Value{}, push.getSplitAttr());
      continue;
    }

    if (auto push = dyn_cast<TPushToAicOp>(op)) {
      auto handlesOr = lookupHandles(op, push.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.v2cPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << push.getId()
                          << ") to enable V2C";
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.v2cPipe,
                                           push.getAivSubblockid(),
                                           push.getSplitAttr());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAicOp>(op)) {
      auto handlesOr = lookupHandles(op, pop.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << pop.getId()
                          << ") to enable C2V";
        return failure();
      }
      Value entry;
      if (isa<TensorViewType>(pop.getTile().getType())) {
        auto decl = rewriter.create<DeclareGlobalOp>(pop.getLoc(),
                                                     pop.getTile().getType());
        propagateGlobalTensorStrides(decl, handles.c2vSlotStrides, rewriter);
        entry = decl.getEntry();
      } else {
        auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                   pop.getTile().getType());
        entry = decl.getTile();
        if (pop.getValidRow() && pop.getValidCol()) {
          rewriter.create<SetValidShapeOp>(pop.getLoc(), entry,
                                           pop.getValidRow(), pop.getValidCol());
        }
      }
      rewriter.create<TPopOp>(pop.getLoc(), entry, handles.c2vPipe,
                              pop.getAivSubblockid(), pop.getSplitAttr());
      rewriter.replaceOp(pop, entry);
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAivOp>(op)) {
      auto handlesOr = lookupHandles(op, pop.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.v2cPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << pop.getId()
                          << ") to enable V2C";
        return failure();
      }
      Value entry;
      if (isa<TensorViewType>(pop.getTile().getType())) {
        auto decl = rewriter.create<DeclareGlobalOp>(pop.getLoc(),
                                                     pop.getTile().getType());
        propagateGlobalTensorStrides(decl, handles.v2cSlotStrides, rewriter);
        entry = decl.getEntry();
      } else {
        auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                   pop.getTile().getType());
        entry = decl.getTile();
        if (pop.getValidRow() && pop.getValidCol()) {
          rewriter.create<SetValidShapeOp>(pop.getLoc(), entry,
                                           pop.getValidRow(), pop.getValidCol());
        }
      }
      rewriter.create<TPopOp>(pop.getLoc(), entry, handles.v2cPipe,
                              Value{}, pop.getSplitAttr());
      rewriter.replaceOp(pop, entry);
      continue;
    }

    if (auto free = dyn_cast<TFreeFromAicOp>(op)) {
      auto handlesOr = lookupHandles(op, free.getId());
      if (failed(handlesOr)) {
        return failure();
      }
      const FrontendPipeHandles &handles = **handlesOr;
      if (!handles.c2vPipe) {
        op->emitOpError() << "requires initialize_pipe(id = " << free.getId()
                          << ") to enable C2V";
        return failure();
      }
      rewriter.replaceOpWithNewOp<TFreeOp>(free, free.getEntry(),
                                           handles.c2vPipe,
                                           free.getSplitAttr());
      continue;
    }

    auto free = cast<TFreeFromAivOp>(op);
    auto handlesOr = lookupHandles(op, free.getId());
    if (failed(handlesOr)) {
      return failure();
    }
    const FrontendPipeHandles &handles = **handlesOr;
    if (!handles.v2cPipe) {
      op->emitOpError() << "requires initialize_pipe(id = " << free.getId()
                        << ") to enable V2C";
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFreeOp>(free, free.getEntry(),
                                         handles.v2cPipe,
                                         free.getSplitAttr());
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
