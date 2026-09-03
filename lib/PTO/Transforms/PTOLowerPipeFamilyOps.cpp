// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

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
// Routing is policy driven, while wrapper entries are resolved from the
// compiler-owned registry. Functions without pipe family ops are
// left untouched entirely (their tile handles keep flowing through the
// regular FoldTileBufIntrinsics path).
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_PTOLOWERPIPEFAMILYOPS
#define GEN_PASS_DEF_PTOLOWERPIPEFAMILYOPS
#include "PTO/Transforms/Passes.h.inc"

namespace {

/// Emits a bridge call with no results and no synthesized storage.
static BridgeCallOp emitVoidBridgeCall(OpBuilder &builder, Location loc,
                                       BridgeEntryId entry, ValueRange args) {
  return builder.create<BridgeCallOp>(
      loc, /*results=*/TypeRange{}, entry,
      /*callee=*/nullptr, /*spec=*/nullptr, /*instanceKey=*/nullptr, args);
}

static DictionaryAttr buildPipeConfigSpec(OpBuilder &builder,
                                          InitializeL2LPipeOp init) {
  SmallVector<NamedAttribute> fields = {
      builder.getNamedAttr("flag_base", builder.getI32IntegerAttr(
                                            init.getFlagBaseAttr().getInt())),
      builder.getNamedAttr("dir_mask",
                           builder.getI32IntegerAttr(init.getDirMask())),
      builder.getNamedAttr("slot_size",
                           builder.getI32IntegerAttr(init.getSlotSize())),
      builder.getNamedAttr("slot_num",
                           builder.getI32IntegerAttr(init.getSlotNum())),
      builder.getNamedAttr("local_slot_num", builder.getI32IntegerAttr(2)),
      builder.getNamedAttr(
          "nosplit", builder.getBoolAttr(init.getNosplitAttr() &&
                                         init.getNosplitAttr().getValue()))};
  return DictionaryAttr::get(builder.getContext(), fields);
}

static DictionaryAttr buildTileSpec(OpBuilder &builder, TileBufType tile) {
  SmallVector<NamedAttribute> fields = {
      builder.getNamedAttr("element_type",
                           TypeAttr::get(tile.getElementType())),
      builder.getNamedAttr("shape",
                           builder.getDenseI64ArrayAttr(tile.getShape())),
      builder.getNamedAttr("valid_shape",
                           builder.getDenseI64ArrayAttr(tile.getValidShape())),
      builder.getNamedAttr(
          "b_layout", builder.getI32IntegerAttr(tile.getBLayoutValueI32())),
      builder.getNamedAttr(
          "s_layout", builder.getI32IntegerAttr(tile.getSLayoutValueI32())),
      builder.getNamedAttr(
          "s_fractal", builder.getI32IntegerAttr(tile.getSFractalSizeI32()))};
  if (Attribute memory = tile.getMemorySpace()) {
    fields.push_back(builder.getNamedAttr("memory_space", memory));
  }
  return DictionaryAttr::get(builder.getContext(), fields);
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
    : public impl::PTOLowerPipeFamilyOpsBase<PTOLowerPipeFamilyOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerPipeFamilyOpsPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func);
    bool hadError = false;
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

    // Only tile handles participating in the pipe protocol belong to this
    // family lowering. Unrelated tiles in a mixed kernel continue through
    // the regular VPTO lowering path.
    llvm::DenseSet<Value> bridgedTiles;
    for (TPushOp push : pushes) {
      bridgedTiles.insert(push.getTile());
    }
    for (TPopOp pop : pops) {
      bridgedTiles.insert(pop.getTile());
    }

    // Whitelist-driven routing: the pass only acts on functions that carry
    // pipe family ops. Tile handles of pipe-less functions keep flowing
    // through the regular lowering (FoldTileBufIntrinsics), matching the
    // pre-bridge behavior.
    if (inits.empty() && pushes.empty() && pops.empty() && frees.empty()) {
      return;
    }

    // The whitelist always resolves through the formal chain (pass option,
    // PTOAS_VPTO_BRIDGE_WHITELIST, built-in default), so routing is
    // guaranteed; `whitelistName` is only for diagnostics.
    FailureOr<BridgeRoutePolicy> policyOr =
        loadBridgeRoutePolicy(whitelistPath, llvm::errs());
    if (failed(policyOr)) {
      signalPassFailure();
      return;
    }
    if (!policyOr->routesFamily("pipe")) {
      return;
    }

    auto routeOp = [&](Operation *op) -> const BridgeFunctionDesc * {
      const BridgeFunctionDesc *entry =
          findBridgeFunctionByOp(op->getName().getStringRef());
      if (!entry || entry->family != BridgeFamily::Pipe) {
        op->emitError("VPTO pipe bridge op has no registered handler");
        hadError = true;
      }
      return entry;
    };

    // Phase 1: initialize_l2l_pipe -> storage-producing bridge init call.
    // The SSA pipe value becomes the bridge call result (the storage handle);
    // push/pop/free below consume that same value.
    for (InitializeL2LPipeOp init : inits) {
      const BridgeFunctionDesc *entry = routeOp(init);
      if (!entry) {
        continue;
      }
      const BridgeFunctionDesc *sizeEntry =
          findBridgeFunction(BridgeEntryId::PipeSize);
      if (!sizeEntry) {
        init.emitError("VPTO pipe bridge registry has no object size entry");
        hadError = true;
        continue;
      }
      if (!isSupportedPipeCapability(init)) {
        init.emitError(
            "VPTO pipe bridge supports only a local pipe with dir_mask 1 "
            "(C2V) or 2 (V2C), no acc_push_epilogue, and an i32 local buffer "
            "address");
        hadError = true;
        continue;
      }
      builder.setInsertionPoint(init);
      BridgeObjectCreateOp call = builder.create<BridgeObjectCreateOp>(
          init.getLoc(), init.getPipe().getType(),
          entry->id, /*callee=*/nullptr,
          /*sizeCallee=*/nullptr, /*spec=*/nullptr, /*instanceKey=*/nullptr,
          ValueRange{init.getLocalAddr()});
      call->setAttr("pipe_config", buildPipeConfigSpec(builder, init));
      // The bridge call result becomes the storage handle: push/pop/free
      // consume the same SSA value instead of the erased pipe op.
      init.getPipe().replaceAllUsesWith(call.getResult());
      init.erase();
    }

    // Phase 2: tpop -> bridge pop call; record the returned FIFO slot
    // address for the declared tile it rebinds.
    llvm::DenseMap<Value, Value> popAddresses;
    // A Pipe specialization renders one shared TileSplitAxis template
    // argument, so every lifecycle op using that object must agree.
    llvm::DenseMap<Value, int64_t> bridgedSplits;
    auto checkSplitConsistency = [&](Operation *op, Value pipeHandle,
                                     int64_t split, llvm::StringRef opName) {
      if (split < 0 || split > 4) {
        op->emitError() << "VPTO pipe bridge " << opName
                        << " carries an unsupported split value " << split;
        hadError = true;
        return false;
      }
      auto [it, inserted] = bridgedSplits.try_emplace(pipeHandle, split);
      if (!inserted && it->second != split) {
        op->emitError() << "VPTO pipe bridge " << opName << " split " << split
                        << " does not match split " << it->second
                        << " used by the same pipe object";
        hadError = true;
        return false;
      }
      return true;
    };
    for (TPopOp pop : pops) {
      const BridgeFunctionDesc *entry = routeOp(pop);
      if (!entry) {
        continue;
      }
      if (popAddresses.count(pop.getTile())) {
        pop.emitError(
            "VPTO pipe bridge supports at most one TPOP per declared tile; "
            "sequential rebind consumption is not supported yet");
        hadError = true;
        continue;
      }
      auto consumerTileTy = dyn_cast<TileBufType>(pop.getTile().getType());
      if (!consumerTileTy) {
        pop.emitError("VPTO pipe bridge TPOP tile must be a tile_buf");
        hadError = true;
        continue;
      }
      if (!checkSplitConsistency(pop, pop.getPipeHandle(), pop.getSplit(),
                                 "TPOP")) {
        continue;
      }
      builder.setInsertionPoint(pop);
      BridgeCallOp call = builder.create<BridgeCallOp>(
          pop.getLoc(), /*results=*/TypeRange{builder.getI64Type()},
          entry->id, /*callee=*/nullptr,
          /*spec=*/nullptr, /*instanceKey=*/nullptr,
          ValueRange{pop.getPipeHandle()});
      call->setAttr("split", builder.getI32IntegerAttr(pop.getSplit()));
      call->setAttr("consumer_tile_spec",
                    buildTileSpec(builder, consumerTileTy));
      popAddresses[pop.getTile()] = call.getResults().front();
      pop.erase();
    }

    // Phase 3: tile_buf_addr -> bridge_inttoptr on the resolved address.
    DominanceInfo dominance(func);
    for (TileBufAddrOp addr : addrs) {
      if (!bridgedTiles.contains(addr.getSrc())) {
        continue;
      }
      Value address = resolveTileAddress(addr.getSrc(), builder, popAddresses);
      if (!address) {
        addr.emitError(
            "VPTO pipe bridge requires tile_buf_addr sources to be a planned "
            "alloc_tile or a declare_tile rebound by tpop");
        hadError = true;
        continue;
      }
      if (auto decl = addr.getSrc().getDefiningOp<DeclareTileOp>()) {
        Operation *producer = address.getDefiningOp();
        if (!producer || !dominance.dominates(producer, addr.getOperation())) {
          addr.emitError("VPTO pipe bridge requires the matching tpop to "
                         "dominate tile_buf_addr");
          hadError = true;
          continue;
        }
      }
      builder.setInsertionPoint(addr);
      BridgeIntToPtrOp pointer = builder.create<BridgeIntToPtrOp>(
          addr.getLoc(), addr.getDst().getType(), address);
      addr.getDst().replaceAllUsesWith(pointer.getResult());
      addr.erase();
    }

    // Phase 4: tpush -> bridge push call on the planned alloc_tile address.
    for (TPushOp push : pushes) {
      const BridgeFunctionDesc *entry = routeOp(push);
      if (!entry) {
        continue;
      }
      auto alloc = push.getTile().getDefiningOp<AllocTileOp>();
      if (!alloc || !alloc.getAddr()) {
        push.emitError("VPTO pipe bridge TPUSH requires a tile from an "
                       "alloc_tile with a planned address");
        hadError = true;
        continue;
      }
      auto producerTileTy = dyn_cast<TileBufType>(push.getTile().getType());
      if (!producerTileTy) {
        push.emitError("VPTO pipe bridge TPUSH tile must be a tile_buf");
        hadError = true;
        continue;
      }
      if (!checkSplitConsistency(push, push.getPipeHandle(), push.getSplit(),
                                 "TPUSH")) {
        continue;
      }
      builder.setInsertionPoint(push);
      BridgeCallOp call =
          emitVoidBridgeCall(builder, push.getLoc(), entry->id,
                             ValueRange{push.getPipeHandle(), alloc.getAddr()});
      call->setAttr("split", builder.getI32IntegerAttr(push.getSplit()));
      call->setAttr("producer_tile_spec",
                    buildTileSpec(builder, producerTileTy));
      push.erase();
    }

    // Phase 5: tfree -> bridge free call.
    for (TFreeOp free : frees) {
      const BridgeFunctionDesc *entry = routeOp(free);
      if (!entry) {
        continue;
      }
      if (free.getEntry()) {
        free.emitError("VPTO pipe bridge TFREE supports the pipe-entry form "
                       "without a tile operand");
        hadError = true;
        continue;
      }
      if (!checkSplitConsistency(free, free.getPipeHandle(), free.getSplit(),
                                 "TFREE")) {
        continue;
      }
      builder.setInsertionPoint(free);
      BridgeCallOp call = emitVoidBridgeCall(builder, free.getLoc(), entry->id,
                                             ValueRange{free.getPipeHandle()});
      call->setAttr("split", builder.getI32IntegerAttr(free.getSplit()));
      free.erase();
    }

    // Phase 6: erase tile handles whose consumers are all bridged now.
    for (AllocTileOp alloc : allocs) {
      if (!bridgedTiles.contains(alloc.getResult())) {
        continue;
      }
      if (!alloc.use_empty()) {
        alloc.emitError("VPTO pipe bridge: alloc_tile still has users after "
                        "pipe family lowering");
        hadError = true;
        continue;
      }
      alloc.erase();
    }
    for (DeclareTileOp decl : decls) {
      if (!bridgedTiles.contains(decl.getResult())) {
        continue;
      }
      if (!decl.use_empty()) {
        decl.emitError("VPTO pipe bridge: declare_tile still has users after "
                       "pipe family lowering");
        hadError = true;
        continue;
      }
      decl.erase();
    }

    if (!hadError) {
      func.walk([&](BridgeObjectCreateOp create) {
        auto config = create->getAttrOfType<DictionaryAttr>("pipe_config");
        if (!config) {
          create.emitError("Pipe bridge object is missing structured config");
          hadError = true;
          return;
        }
        SmallVector<NamedAttribute> fields = {
            builder.getNamedAttr(kBridgeSpecPipeKey, config)};
        IntegerAttr split;
        DictionaryAttr producer;
        DictionaryAttr consumer;
        SmallVector<BridgeCallOp> calls;
        for (Operation *user : create.getResult().getUsers()) {
          auto bridgeCall = dyn_cast<BridgeCallOp>(user);
          if (!bridgeCall) {
            continue;
          }
          calls.push_back(bridgeCall);
          if (auto value = bridgeCall->getAttrOfType<IntegerAttr>("split")) {
            split = value;
          }
          if (auto value = bridgeCall->getAttrOfType<DictionaryAttr>(
                  "producer_tile_spec")) {
            producer = value;
          }
          if (auto value = bridgeCall->getAttrOfType<DictionaryAttr>(
                  "consumer_tile_spec")) {
            consumer = value;
          }
        }
        if (split) {
          fields.push_back(builder.getNamedAttr(kBridgeSpecSplitKey, split));
        }
        if (producer) {
          fields.push_back(
              builder.getNamedAttr(kBridgeSpecProducerTileKey, producer));
        }
        if (consumer) {
          fields.push_back(
              builder.getNamedAttr(kBridgeSpecConsumerTileKey, consumer));
        }
        DictionaryAttr value =
            DictionaryAttr::get(builder.getContext(), fields);
        BridgePipeSpecAttr pipeSpec =
            BridgePipeSpecAttr::get(builder.getContext(), value);
        std::string key;
        llvm::raw_string_ostream os(key);
        os << "pipe|";
        value.print(os);
        os.flush();
        StringAttr keyAttr = builder.getStringAttr(key);
        create->setAttr("spec", pipeSpec);
        create->setAttr(kBridgeInstanceKeyAttrName, keyAttr);
        create->removeAttr("pipe_config");
        for (BridgeCallOp bridgeCall : calls) {
          bridgeCall->setAttr("spec", pipeSpec);
          bridgeCall->setAttr(kBridgeInstanceKeyAttrName, keyAttr);
          bridgeCall->removeAttr("split");
          bridgeCall->removeAttr("producer_tile_spec");
          bridgeCall->removeAttr("consumer_tile_spec");
        }
      });
    }
    if (hadError) {
      signalPassFailure();
    }
  }

private:
  /// Capability check for the pipe bridge. The concrete configuration
  /// (slot_size/slot_num/flag_base/nosplit) is read from the op attributes
  /// and flows into the generated wrapper; only genuinely unsupported forms
  /// are rejected here.
  static bool isSupportedPipeCapability(InitializeL2LPipeOp init) {
    int8_t dirMask = init.getDirMask();
    if (dirMask != 1 && dirMask != 2) {
      return false;
    }
    if (init.getAccPushEpilogueAttr()) {
      return false;
    }
    auto localAddrTy = dyn_cast<IntegerType>(init.getLocalAddr().getType());
    bool hasI32LocalAddress = localAddrTy && localAddrTy.getWidth() == 32;
    if (!hasI32LocalAddress) {
      return false;
    }
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerPipeFamilyOpsPass() {
  return std::make_unique<PTOLowerPipeFamilyOpsPass>();
}

} // namespace pto
} // namespace mlir
