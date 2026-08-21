// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOLowerPipeFamilyOps.cpp - TPipe family bridge lowering ----------===//
//===----------------------------------------------------------------------===//
//
// TPipe family pass of the VPTO C++ interface bridge. It understands the
// semantics of the internal pipe ops (initialize_l2l_pipe / tpush / tpop /
// tfree) plus the tile handles they consume (alloc_tile / declare_tile /
// tile_buf_addr) and rewrites them into generic pto.bridge_call /
// pto.bridge_inttoptr ops that carry only wrapper callee names and ABI
// values. All family semantics (config validation, storage handle flow, and
// the runtime rebinding of a declared tile to the FIFO slot returned by
// TPOP) are resolved here; the generic bridge lowering pass only sees the
// resulting bridge ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
namespace {

/// Wrapper callee names shared with the fixed pipe bridge wrapper
/// (test/vpto/cases/kernels/fifo-tile-data-consume/vpto_bridge.cpp).
static constexpr llvm::StringLiteral kPipeInitEntry = "pto_vpto_pipe_init";
static constexpr llvm::StringLiteral kPipeSizeEntry = "pto_vpto_pipe_size";
static constexpr llvm::StringLiteral kPipePushEntry = "pto_vpto_pipe_push";
static constexpr llvm::StringLiteral kPipePopEntry = "pto_vpto_pipe_pop";
static constexpr llvm::StringLiteral kPipeFreeEntry = "pto_vpto_pipe_free";

/// Emits a bridge call with no results and no synthesized storage.
static BridgeCallOp emitVoidBridgeCall(OpBuilder &builder, Location loc,
                                       llvm::StringRef callee,
                                       ValueRange args) {
  return builder.create<BridgeCallOp>(
      loc, /*results=*/TypeRange{}, /*callee=*/callee,
      /*storage_size_callee=*/nullptr, /*args=*/args);
}

/// Returns the address value a tile_buf_addr operand resolves to, or nullptr
/// when the source cannot be resolved (the caller emits the diagnostic).
/// alloc_tile carries the planned address as an i64 operand; a declare_tile
/// rebound by TPOP resolves to the FIFO slot address returned by the pop.
static Value resolveTileAddress(Value tile, OpBuilder &builder,
                                llvm::DenseMap<Value, Value> &popAddresses) {
  if (auto alloc = tile.getDefiningOp<AllocTileOp>()) {
    return alloc.getAddr();
  }
  if (isa_and_nonnull<DeclareTileOp>(tile.getDefiningOp())) {
    auto it = popAddresses.find(tile);
    if (it == popAddresses.end()) {
      return nullptr;
    }
    return it->second;
  }
  return nullptr;
}

struct PTOLowerPipeFamilyOpsPass final
    : public PassWrapper<PTOLowerPipeFamilyOpsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerPipeFamilyOpsPass)

  llvm::StringRef getArgument() const final {
    return "pto-lower-pipe-family-ops";
  }

  llvm::StringRef getDescription() const final {
    return "Lower internal TPipe ops into generic VPTO bridge ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func);
    bool failed = false;

    // Collect first; rewriting during the walk would invalidate the walker.
    SmallVector<InitializeL2LPipeOp> inits;
    SmallVector<TPushOp> pushes;
    SmallVector<TPopOp> pops;
    SmallVector<TFreeOp> frees;
    SmallVector<TileBufAddrOp> addrs;
    SmallVector<AllocTileOp> allocs;
    SmallVector<DeclareTileOp> decls;
    func.walk([&](Operation *op) {
      if (auto init = dyn_cast<InitializeL2LPipeOp>(op)) {
        inits.push_back(init);
      } else if (auto push = dyn_cast<TPushOp>(op)) {
        pushes.push_back(push);
      } else if (auto pop = dyn_cast<TPopOp>(op)) {
        pops.push_back(pop);
      } else if (auto free = dyn_cast<TFreeOp>(op)) {
        frees.push_back(free);
      } else if (auto addr = dyn_cast<TileBufAddrOp>(op)) {
        addrs.push_back(addr);
      } else if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        allocs.push_back(alloc);
      } else if (auto decl = dyn_cast<DeclareTileOp>(op)) {
        decls.push_back(decl);
      }
    });

    // Phase 1: initialize_l2l_pipe -> storage-producing bridge init call.
    // The SSA pipe value becomes the bridge call result (the storage handle);
    // push/pop/free below consume that same value.
    for (InitializeL2LPipeOp init : inits) {
      if (!isSupportedPipeConfig(init) ||
          !isa<IntegerType>(init.getLocalAddr().getType()) ||
          cast<IntegerType>(init.getLocalAddr().getType()).getWidth() != 32) {
        init.emitError(
            "VPTO pipe bridge currently supports only A5 C2V local pipe, "
            "flag_base=0, slot_size=1024, slot_num=8, nosplit=false, with an "
            "i32 local buffer address");
        failed = true;
        continue;
      }
      builder.setInsertionPoint(init);
      BridgeCallOp call = builder.create<BridgeCallOp>(
          init.getLoc(), /*results=*/TypeRange{init.getPipe().getType()},
          /*callee=*/kPipeInitEntry,
          /*storage_size_callee=*/builder.getStringAttr(kPipeSizeEntry),
          /*args=*/ValueRange{init.getLocalAddr()});
      // The bridge call result becomes the storage handle: push/pop/free
      // consume the same SSA value instead of the erased pipe op.
      init.getPipe().replaceAllUsesWith(call.getResults().front());
      init.erase();
    }

    // Phase 2: tpop -> bridge pop call; record the returned FIFO slot
    // address for the declared tile it rebinds.
    llvm::DenseMap<Value, Value> popAddresses;
    for (TPopOp pop : pops) {
      if (pop.getSplit() != 1) {
        pop.emitError("VPTO pipe bridge TPOP requires split=1");
        failed = true;
        continue;
      }
      builder.setInsertionPoint(pop);
      BridgeCallOp call = builder.create<BridgeCallOp>(
          pop.getLoc(), /*results=*/TypeRange{builder.getI64Type()},
          /*callee=*/kPipePopEntry, /*storage_size_callee=*/nullptr,
          /*args=*/ValueRange{pop.getPipeHandle()});
      popAddresses[pop.getTile()] = call.getResults().front();
      pop.erase();
    }

    // Phase 3: tile_buf_addr -> bridge_inttoptr on the resolved address.
    for (TileBufAddrOp addr : addrs) {
      Value address = resolveTileAddress(addr.getSrc(), builder, popAddresses);
      if (!address) {
        addr.emitError(
            "VPTO pipe bridge requires tile_buf_addr sources to be a planned "
            "alloc_tile or a declare_tile rebound by tpop");
        failed = true;
        continue;
      }
      builder.setInsertionPoint(addr);
      BridgeIntToPtrOp pointer = builder.create<BridgeIntToPtrOp>(
          addr.getLoc(), addr.getDst().getType(), address);
      addr.getDst().replaceAllUsesWith(pointer.getResult());
      addr.erase();
    }

    // Phase 4: tpush -> bridge push call on the planned alloc_tile address.
    for (TPushOp push : pushes) {
      auto alloc = push.getTile().getDefiningOp<AllocTileOp>();
      if (push.getSplit() != 1 || !alloc || !alloc.getAddr()) {
        push.emitError("VPTO pipe bridge TPUSH requires split=1 and a tile "
                       "from an alloc_tile with a planned address");
        failed = true;
        continue;
      }
      builder.setInsertionPoint(push);
      emitVoidBridgeCall(builder, push.getLoc(), kPipePushEntry,
                         ValueRange{push.getPipeHandle(), alloc.getAddr()});
      push.erase();
    }

    // Phase 5: tfree -> bridge free call.
    for (TFreeOp free : frees) {
      if (free.getEntry() || free.getSplit() != 1) {
        free.emitError(
            "VPTO pipe bridge TFREE supports the tile-entry form with split=1");
        failed = true;
        continue;
      }
      builder.setInsertionPoint(free);
      emitVoidBridgeCall(builder, free.getLoc(), kPipeFreeEntry,
                         ValueRange{free.getPipeHandle()});
      free.erase();
    }

    // Phase 6: erase tile handles whose consumers are all bridged now.
    for (AllocTileOp alloc : allocs) {
      if (!alloc.use_empty()) {
        alloc.emitError("VPTO pipe bridge: alloc_tile still has users after "
                        "pipe family lowering");
        failed = true;
        continue;
      }
      alloc.erase();
    }
    for (DeclareTileOp decl : decls) {
      if (!decl.use_empty()) {
        decl.emitError("VPTO pipe bridge: declare_tile still has users after "
                       "pipe family lowering");
        failed = true;
        continue;
      }
      decl.erase();
    }

    if (failed) {
      signalPassFailure();
    }
  }

private:
  /// Phase 0 supports the same fixed specialization the bridge wrapper
  /// instantiates; anything else is a diagnostic, never an approximation.
  static bool isSupportedPipeConfig(InitializeL2LPipeOp init) {
    return init.getDirMask() == 1 && init.getSlotSize() == 1024 &&
           init.getSlotNum() == 8 && init.getFlagBaseAttr() &&
           init.getFlagBaseAttr().getInt() == 0 &&
           (!init.getNosplitAttr() || !init.getNosplitAttr().getValue()) &&
           !init.getAccPushEpilogueAttr();
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerPipeFamilyOpsPass() {
  return std::make_unique<PTOLowerPipeFamilyOpsPass>();
}

} // namespace pto
} // namespace mlir
