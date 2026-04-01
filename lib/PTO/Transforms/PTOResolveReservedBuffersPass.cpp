// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTORESOLVERESERVEDBUFFERS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct PipePeerKey {
  std::string ownerFunc;
  std::string reserveName;
  int8_t dirMask = 0;

  // Provide a stable lexicographic order so PipePeerKey can be used as the
  // key type of std::map.
  bool operator<(const PipePeerKey &other) const {
    return std::tie(ownerFunc, reserveName, dirMask) <
           std::tie(other.ownerFunc, other.reserveName, other.dirMask);
  }
};

struct PipeInitInfo {
  Operation *op = nullptr;
  func::FuncOp funcOp;
  int8_t dirMask = 0;
};

template <typename InitOpT> static Value getLocalAddrOperand(InitOpT op) {
  // Hide the concrete init-op type and expose the local address operand
  // through one helper used by the shared peer-grouping logic.
  return op.getLocalAddr();
}

template <typename InitOpT> static IntegerAttr getFlagBaseAttr(InitOpT op) {
  return op.getFlagBaseAttr();
}

template <typename InitOpT>
static void setFlagBaseAttr(InitOpT op, IntegerAttr attr) {
  op->setAttr("flag_base", attr);
}

static ReserveBufferOp findReserveBufferByName(func::FuncOp funcOp,
                                               StringRef name) {
  // Reserve-buffer lookup is name-based because import_reserved_buffer only
  // stores the peer function symbol and the logical reserve name.
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name)
      return WalkResult::advance();
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

static std::string getFuncSymbol(func::FuncOp funcOp) {
  return funcOp.getSymName().str();
}


static std::optional<PipePeerKey> getPipePeerKey(Value localAddr,
                                                 func::FuncOp currentFunc) {
  // reserve_buffer identifies the local owner directly, while
  // import_reserved_buffer points back to the owner through peer_func.
  // Normalize both cases so peer pipe inits can be matched on the same logical
  // pipe key.
  if (auto reserveOp = localAddr.getDefiningOp<ReserveBufferOp>()) {
    return PipePeerKey{getFuncSymbol(currentFunc), reserveOp.getName().str(),
                       0};
  }

  if (auto importOp = localAddr.getDefiningOp<ImportReservedBufferOp>()) {
    return PipePeerKey{importOp.getPeerFuncAttr().getValue().str(),
                       importOp.getName().str(), 0};
  }

  return std::nullopt;
}

static bool hasCompletePeerInitPair(const SmallVector<PipeInitInfo> &inits,
                                    const std::set<std::string> &participants) {
  // A peer-aware logical pipe is only well-defined when exactly two init ops
  // participate: one in the reserve owner and one in the peer importer.
  if (participants.size() != 2 || inits.size() != 2)
    return false;

  std::set<std::string> initFuncs;
  for (const PipeInitInfo &info : inits)
    initFuncs.insert(getFuncSymbol(info.funcOp));
  return initFuncs.size() == 2;
}

struct PTOResolveReservedBuffersPass
    : public mlir::pto::impl::PTOResolveReservedBuffersBase<
          PTOResolveReservedBuffersPass> {
  LogicalResult assignPeerAwareFlagBases(ModuleOp moduleOp) {
    // Group internal pipe init ops by their logical pipe identity, then fill
    // missing flag_base attrs so both sides of the same logical pipe agree.
    std::map<PipePeerKey, SmallVector<PipeInitInfo>> keyedInits;
    std::map<PipePeerKey, std::set<std::string>> keyedParticipants;
    LogicalResult status = success();

    auto collectInit = [&](auto initOp) {
      if (failed(status))
        return;
      PipeInitInfo info;
      info.op = initOp.getOperation();
      info.funcOp = initOp->template getParentOfType<func::FuncOp>();
      info.dirMask = initOp.getDirMask();

      // Record one address into the keyed maps. Returns true when the
      // address comes from reserve_buffer / import_reserved_buffer.
      auto recordAddr = [&](Value addr, int8_t effectiveDirMask) -> bool {
        auto key = getPipePeerKey(addr, info.funcOp);
        if (!key)
          return false;
        key->dirMask = effectiveDirMask;
        keyedInits[*key].push_back(info);
        keyedParticipants[*key].insert(getFuncSymbol(info.funcOp));
        keyedParticipants[*key].insert(key->ownerFunc);
        return true;
      };

      if (info.dirMask == 3) {
        // DIR_BOTH: treat as two logical pipes keyed by direction.
        bool c2vOk = recordAddr(getLocalAddrOperand(initOp), /*c2v=*/1);
        Value peerAddr = initOp.getPeerLocalAddr();
        bool v2cOk = peerAddr && recordAddr(peerAddr, /*v2c=*/2);
        if (c2vOk || v2cOk)
          return;
      } else {
        Value localAddr = getLocalAddrOperand(initOp);
        if (recordAddr(localAddr, info.dirMask))
          return;
      }
      if (getFlagBaseAttr(initOp))
        return;
      status = initOp.emitOpError(
          "requires local_addr to come from pto.reserve_buffer or "
          "pto.import_reserved_buffer when 'flag_base' is not explicit");
    };

    moduleOp.walk([&](InitializeL2LPipeOp initOp) { collectInit(initOp); });
    moduleOp.walk([&](InitializeL2G2LPipeOp initOp) { collectInit(initOp); });
    if (failed(status))
      return failure();

    for (const auto &it : keyedInits) {
      if (hasCompletePeerInitPair(it.second, keyedParticipants[it.first]))
        continue;
      return it.second.front().op->emitOpError(
          "requires a complete peer init pair when local_addr comes from "
          "pto.reserve_buffer or pto.import_reserved_buffer");
    }

    OpBuilder builder(moduleOp.getContext());
    for (const auto &it : keyedInits) {
      const auto &inits = it.second;
      // flag_base is always 0: single-direction pipes use flag pair 0/1;
      // DIR_BOTH pipes internally manage 0/1 for C2V and 2/3 for V2C.
      int32_t desiredBase = 0;

      std::optional<int32_t> chosenBase;
      for (const PipeInitInfo &info : inits) {
        // Respect any explicit flag_base already present on one side, but make
        // sure all peers resolve to the same value.
        if (auto initOp = dyn_cast<InitializeL2LPipeOp>(info.op)) {
          if (auto flagBaseAttr = getFlagBaseAttr(initOp)) {
            if (chosenBase && *chosenBase != flagBaseAttr.getInt()) {
              return info.op->emitOpError(
                  "conflicting explicit flag_base across peer pipe inits");
            }
            chosenBase = flagBaseAttr.getInt();
          }
          continue;
        }

        auto initOp = cast<InitializeL2G2LPipeOp>(info.op);
        if (auto flagBaseAttr = getFlagBaseAttr(initOp)) {
          if (chosenBase && *chosenBase != flagBaseAttr.getInt()) {
            return info.op->emitOpError(
                "conflicting explicit flag_base across peer pipe inits");
          }
          chosenBase = flagBaseAttr.getInt();
        }
      }
      if (!chosenBase)
        chosenBase = desiredBase;

      auto flagBaseAttr = builder.getI32IntegerAttr(*chosenBase);
      for (const PipeInitInfo &info : inits) {
        if (auto initOp = dyn_cast<InitializeL2LPipeOp>(info.op)) {
          if (!getFlagBaseAttr(initOp))
            setFlagBaseAttr(initOp, flagBaseAttr);
          continue;
        }
        auto initOp = cast<InitializeL2G2LPipeOp>(info.op);
        if (!getFlagBaseAttr(initOp))
          setFlagBaseAttr(initOp, flagBaseAttr);
      }
    }

    return success();
  }

  LogicalResult materializeResolvedAddresses(ModuleOp moduleOp) {
    // Resolve frontend reserve/import ops to plain constant local addresses so
    // downstream lowering only sees ordinary SSA values.
    SmallVector<Operation *> eraseOps;

    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
      OpBuilder builder(funcOp.getContext());

      SmallVector<ReserveBufferOp> reserveOps;
      funcOp.walk(
          [&](ReserveBufferOp reserveOp) { reserveOps.push_back(reserveOp); });
      for (ReserveBufferOp reserveOp : reserveOps) {
        auto baseAttr = reserveOp.getBaseAttr();
        if (!baseAttr) {
          return reserveOp.emitOpError(
              "expects 'base' to be resolved before address materialization");
        }
        // After PlanMemory, reserve_buffer is only a frontend marker. Replace
        // its SSA result with the resolved constant base so later passes only
        // see plain local addresses.
        builder.setInsertionPoint(reserveOp);
        Value cst = builder.create<arith::ConstantIntOp>(reserveOp.getLoc(),
                                                         baseAttr.getInt(), 32);
        reserveOp.getAddr().replaceAllUsesWith(cst);
        eraseOps.push_back(reserveOp.getOperation());
      }

      SmallVector<ImportReservedBufferOp> importOps;
      funcOp.walk([&](ImportReservedBufferOp importOp) {
        importOps.push_back(importOp);
      });
      for (ImportReservedBufferOp importOp : importOps) {
        auto peerFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            importOp.getOperation(), importOp.getPeerFuncAttr());
        if (!peerFunc) {
          return importOp.emitOpError(
              "expects 'peer_func' to reference an existing func.func");
        }

        auto peerReserve =
            findReserveBufferByName(peerFunc, importOp.getName());
        if (!peerReserve)
          return importOp.emitOpError(
              "expects matching peer reserve_buffer to exist");

        auto baseAttr = peerReserve.getBaseAttr();
        if (!baseAttr) {
          return importOp.emitOpError(
              "expects peer reserve_buffer base to be resolved");
        }

        // import_reserved_buffer never allocates memory locally. It is just a
        // symbolic reference to the peer reserve_buffer and is materialized to
        // the same resolved constant base here.
        builder.setInsertionPoint(importOp);
        Value cst = builder.create<arith::ConstantIntOp>(importOp.getLoc(),
                                                         baseAttr.getInt(), 32);
        importOp.getAddr().replaceAllUsesWith(cst);
        eraseOps.push_back(importOp.getOperation());
      }
    }

    for (Operation *op : eraseOps)
      op->erase();

    return success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (failed(assignPeerAwareFlagBases(moduleOp)) ||
        failed(materializeResolvedAddresses(moduleOp))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOResolveReservedBuffersPass() {
  return std::make_unique<PTOResolveReservedBuffersPass>();
}
