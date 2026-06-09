// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTORESOLVERESERVEDBUFFERS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr int32_t kMaxHardwareFlagIds = 16;
constexpr size_t kPeerPipeInitOpCount = 2;
constexpr size_t kPeerPipeParticipantCount = 2;
constexpr int32_t kFlagAlignment = 2;
constexpr int8_t kC2VDirMask = 1;
constexpr int8_t kV2CDirMask = 2;
constexpr int8_t kBidirectionalDirMask = 3;
constexpr unsigned kSingleDirectionFlagWidth = 2;
constexpr unsigned kBidirectionalFlagWidth = 4;
constexpr unsigned kVisitedInitReserveSize = 16;
constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";

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
  int32_t slotSize = 0;
  int32_t slotNum = 0;
  std::optional<int32_t> localSlotNum;
  bool globalOnly = false;
};

struct PipeComponent {
  SmallVector<Operation *> ops;
  std::set<std::string> participants;
  int8_t dirMask = 0;
  int32_t slotSize = 0;
  int32_t slotNum = 0;
  std::optional<int32_t> localSlotNum;
  bool globalOnly = false;
  unsigned flagWidth = 0;
  std::optional<int32_t> explicitFlagBase;
};

struct FlagInterval {
  int32_t begin = 0;
  int32_t end = 0;
};

using PipeInitGroups = std::map<PipePeerKey, SmallVector<Operation *>>;
using PipeFlagUsage = std::map<std::string, SmallVector<FlagInterval>>;

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
  funcOp.walk([&found, name](ReserveBufferOp reserveOp) {
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

template <typename InitOpT>
static PipeInitInfo buildPipeInitInfo(InitOpT initOp) {
  PipeInitInfo info;
  info.op = initOp.getOperation();
  info.funcOp = initOp->template getParentOfType<func::FuncOp>();
  info.dirMask = initOp.getDirMask();
  info.slotSize = initOp.getSlotSize();
  info.slotNum = initOp.getSlotNum();
  if constexpr (std::is_same_v<InitOpT, InitializeL2G2LPipeOp>) {
    if (auto attr = initOp.getLocalSlotNumAttr())
      info.localSlotNum = attr.getInt();
    info.globalOnly = !initOp.getLocalAddr();
  }
  return info;
}

static PipePeerKey getGlobalTensorPipeKey(const PipeInitInfo &info) {
  std::string id = "unknown";
  if (auto idAttr =
          info.op->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName))
    id = std::to_string(idAttr.getInt());
  else
    id = std::to_string(static_cast<size_t>(
        llvm::hash_value(static_cast<const void *>(info.op))));
  return PipePeerKey{"__pto_globaltensor_pipe", "id_" + id, info.dirMask};
}

template <typename InitOpT>
static LogicalResult collectPeerAwareInit(InitOpT initOp,
                                          SmallVectorImpl<PipeInitInfo> &initInfos,
                                          PipeInitGroups &keyedInits) {
  PipeInitInfo info = buildPipeInitInfo(initOp);
  if (info.globalOnly) {
    keyedInits[getGlobalTensorPipeKey(info)].push_back(info.op);
    initInfos.push_back(info);
    return success();
  }

  auto recordAddr = [&info, &keyedInits](Value addr,
                                         int8_t effectiveDirMask) {
    if (!addr)
      return false;
    auto key = getPipePeerKey(addr, info.funcOp);
    if (!key)
      return false;
    key->dirMask = effectiveDirMask;
    keyedInits[*key].push_back(info.op);
    return true;
  };

  bool recorded = false;
  if (info.dirMask == kBidirectionalDirMask) {
    Value peerAddr = initOp.getPeerLocalAddr();
    recorded = recordAddr(getLocalAddrOperand(initOp), kC2VDirMask);
    recorded = ((peerAddr != nullptr) && recordAddr(peerAddr, kV2CDirMask)) ||
               recorded;
  } else {
    recorded = recordAddr(getLocalAddrOperand(initOp), info.dirMask);
  }

  if (recorded)
    initInfos.push_back(info);
  if (recorded || getFlagBaseAttr(initOp))
    return success();

  return initOp.emitOpError(
      "requires local_addr to come from pto.reserve_buffer or "
      "pto.import_reserved_buffer when 'flag_base' is not explicit");
}

static IntegerAttr getFlagBaseAttr(Operation *op) {
  if (auto initOp = dyn_cast<InitializeL2LPipeOp>(op))
    return initOp.getFlagBaseAttr();
  return cast<InitializeL2G2LPipeOp>(op).getFlagBaseAttr();
}

static void setFlagBaseAttr(Operation *op, IntegerAttr attr) {
  if (auto initOp = dyn_cast<InitializeL2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      setFlagBaseAttr(initOp, attr);
    return;
  }
  auto initOp = cast<InitializeL2G2LPipeOp>(op);
  if (!initOp.getFlagBaseAttr())
    setFlagBaseAttr(initOp, attr);
}

static bool samePipeInitSignature(const PipeInitInfo &lhs,
                                  const PipeInitInfo &rhs) {
  return std::tie(lhs.dirMask, lhs.slotSize, lhs.slotNum, lhs.localSlotNum,
                  lhs.globalOnly) ==
         std::tie(rhs.dirMask, rhs.slotSize, rhs.slotNum, rhs.localSlotNum,
                  rhs.globalOnly);
}

static SmallVector<Operation *>
collectUniqueInitOps(const SmallVector<Operation *> &ops) {
  SmallVector<Operation *> uniqueOps;
  for (Operation *op : ops) {
    if (std::find(uniqueOps.begin(), uniqueOps.end(), op) == uniqueOps.end())
      uniqueOps.push_back(op);
  }
  return uniqueOps;
}

static void addPeerAwareAdjacency(
    const SmallVector<Operation *> &uniqueOps,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &adjacency) {
  for (size_t i = 0; i < uniqueOps.size(); ++i) {
    for (size_t j = i + 1; j < uniqueOps.size(); ++j) {
      adjacency[uniqueOps[i]].push_back(uniqueOps[j]);
      adjacency[uniqueOps[j]].push_back(uniqueOps[i]);
    }
  }
}

static FailureOr<PipeComponent>
buildPipeComponent(const PipeInitInfo &rootInfo,
                   llvm::DenseMap<Operation *, SmallVector<Operation *>> &adjacency,
                   llvm::SmallPtrSet<Operation *, kVisitedInitReserveSize> &visited) {
  SmallVector<Operation *> stack{rootInfo.op};
  PipeComponent component;
  while (!stack.empty()) {
    Operation *current = stack.pop_back_val();
    component.ops.push_back(current);
    for (Operation *neighbor : adjacency[current]) {
      if (visited.insert(neighbor).second)
        stack.push_back(neighbor);
    }
  }
  if (!rootInfo.globalOnly && component.ops.size() != kPeerPipeInitOpCount) {
    return rootInfo.op->emitOpError(
        "requires a complete compatible peer init pair when local_addr comes "
        "from pto.reserve_buffer or pto.import_reserved_buffer");
  }
  return component;
}

static LogicalResult verifyPipeComponentShape(
    PipeComponent &component,
    const llvm::DenseMap<Operation *, const PipeInitInfo *> &infoByOp) {
  const PipeInitInfo &lhs = *infoByOp.lookup(component.ops[0]);
  for (Operation *op : ArrayRef<Operation *>(component.ops).drop_front()) {
    const PipeInitInfo &rhs = *infoByOp.lookup(op);
    if (!samePipeInitSignature(lhs, rhs)) {
      return component.ops.front()->emitOpError(
          "requires peer pipe init ops to agree on direction and pipe shape");
    }
  }
  component.dirMask = lhs.dirMask;
  component.slotSize = lhs.slotSize;
  component.slotNum = lhs.slotNum;
  component.localSlotNum = lhs.localSlotNum;
  component.globalOnly = lhs.globalOnly;
  component.flagWidth = component.dirMask == kBidirectionalDirMask
                            ? kBidirectionalFlagWidth
                            : kSingleDirectionFlagWidth;
  return success();
}

static LogicalResult collectPipeComponentParticipants(
    PipeComponent &component,
    const llvm::DenseMap<Operation *, const PipeInitInfo *> &infoByOp) {
  for (Operation *op : component.ops) {
    const PipeInitInfo &info = *infoByOp.lookup(op);
    component.participants.insert(getFuncSymbol(info.funcOp));
    if (auto flagBaseAttr = getFlagBaseAttr(op)) {
      if (component.explicitFlagBase &&
          *component.explicitFlagBase != flagBaseAttr.getInt()) {
        return op->emitOpError(
            "conflicting explicit flag_base across peer pipe inits");
      }
      component.explicitFlagBase = flagBaseAttr.getInt();
    }
  }
  if (!component.globalOnly &&
      component.participants.size() != kPeerPipeParticipantCount) {
    return component.ops.front()->emitOpError(
        "requires a complete compatible peer init pair when local_addr comes "
        "from pto.reserve_buffer or pto.import_reserved_buffer");
  }
  return success();
}

static FailureOr<SmallVector<PipeComponent>>
buildPeerAwareComponents(const SmallVectorImpl<PipeInitInfo> &initInfos,
                         const PipeInitGroups &keyedInits) {
  llvm::DenseMap<Operation *, SmallVector<Operation *>> adjacency;
  llvm::DenseMap<Operation *, const PipeInitInfo *> infoByOp;
  for (const PipeInitInfo &info : initInfos) {
    adjacency[info.op];
    infoByOp[info.op] = &info;
  }

  for (const auto &it : keyedInits) {
    addPeerAwareAdjacency(collectUniqueInitOps(it.second), adjacency);
  }

  SmallVector<PipeComponent> components;
  llvm::SmallPtrSet<Operation *, kVisitedInitReserveSize> visited;
  for (const PipeInitInfo &rootInfo : initInfos) {
    if (!visited.insert(rootInfo.op).second)
      continue;

    auto componentOr =
        buildPipeComponent(rootInfo, adjacency, visited);
    if (failed(componentOr))
      return failure();
    PipeComponent component = std::move(*componentOr);
    if (failed(verifyPipeComponentShape(component, infoByOp)) ||
        failed(collectPipeComponentParticipants(component, infoByOp)))
      return failure();

    components.push_back(std::move(component));
  }

  return components;
}

static bool overlaps(const FlagInterval &lhs, const FlagInterval &rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

static int32_t alignToEven(int32_t value) {
  return value % kFlagAlignment == 0 ? value : value + (kFlagAlignment - 1);
}

static LogicalResult reserveComponentFlagBase(const PipeComponent &component,
                                              int32_t base,
                                              PipeFlagUsage &usedByFunc) {
  FlagInterval interval{base, base + static_cast<int32_t>(component.flagWidth)};
  if (interval.end > kMaxHardwareFlagIds) {
    return component.ops.front()->emitOpError()
           << "requires all pipe components in a function to fit within "
           << kMaxHardwareFlagIds << " hardware flag ids";
  }
  for (const std::string &funcName : component.participants) {
    for (const FlagInterval &used : usedByFunc[funcName]) {
      if (!overlaps(interval, used))
        continue;
      return component.ops.front()->emitOpError(
          "conflicting flag_base across peer pipe init components in the same function");
    }
  }

  for (const std::string &funcName : component.participants)
    usedByFunc[funcName].push_back(interval);
  return success();
}

static FailureOr<int32_t> chooseFlagBaseForComponent(const PipeComponent &component,
                                                     PipeFlagUsage &usedByFunc) {
  if (component.explicitFlagBase) {
    if (failed(reserveComponentFlagBase(component, *component.explicitFlagBase,
                                        usedByFunc))) {
      return failure();
    }
    return *component.explicitFlagBase;
  }

  int32_t candidateBase = 0;
  while (true) {
    candidateBase = alignToEven(candidateBase);
    FlagInterval candidate{candidateBase,
                           candidateBase +
                               static_cast<int32_t>(component.flagWidth)};
    bool conflict = false;
    int32_t nextCandidate = candidateBase + 2;
    for (const std::string &funcName : component.participants) {
      for (const FlagInterval &used : usedByFunc[funcName]) {
        if (!overlaps(candidate, used))
          continue;
        conflict = true;
        nextCandidate = std::max(nextCandidate, alignToEven(used.end));
      }
    }
    if (!conflict)
      break;
    candidateBase = nextCandidate;
  }

  if (failed(reserveComponentFlagBase(component, candidateBase, usedByFunc)))
    return failure();
  return candidateBase;
}

static LogicalResult materializeReserveBufferBase(OpBuilder &builder,
                                                  ReserveBufferOp reserveOp,
                                                  SmallVectorImpl<Operation *> &eraseOps) {
  auto baseAttr = reserveOp.getBaseAttr();
  if (!baseAttr) {
    return reserveOp.emitOpError(
        "expects 'base' to be resolved before address materialization");
  }
  builder.setInsertionPoint(reserveOp);
  Value cst = builder.create<arith::ConstantIntOp>(reserveOp.getLoc(),
                                                   baseAttr.getInt(), 32);
  reserveOp.getAddr().replaceAllUsesWith(cst);
  eraseOps.push_back(reserveOp.getOperation());
  return success();
}

static LogicalResult materializeImportedReserveBufferBase(
    OpBuilder &builder, ImportReservedBufferOp importOp,
    SmallVectorImpl<Operation *> &eraseOps) {
  auto peerFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      importOp.getOperation(), importOp.getPeerFuncAttr());
  if (!peerFunc) {
    return importOp.emitOpError(
        "expects 'peer_func' to reference an existing func.func");
  }

  auto peerReserve = findReserveBufferByName(peerFunc, importOp.getName());
  if (!peerReserve) {
    return importOp.emitOpError(
        "expects matching peer reserve_buffer to exist");
  }

  auto baseAttr = peerReserve.getBaseAttr();
  if (!baseAttr) {
    return importOp.emitOpError(
        "expects peer reserve_buffer base to be resolved");
  }

  builder.setInsertionPoint(importOp);
  Value cst = builder.create<arith::ConstantIntOp>(importOp.getLoc(),
                                                   baseAttr.getInt(), 32);
  importOp.getAddr().replaceAllUsesWith(cst);
  eraseOps.push_back(importOp.getOperation());
  return success();
}

struct PTOResolveReservedBuffersPass
    : public mlir::pto::impl::PTOResolveReservedBuffersBase<
          PTOResolveReservedBuffersPass> {
  LogicalResult assignPeerAwareFlagBases(ModuleOp moduleOp) const {
    // Build peer-connected pipe-init components, assign one consistent
    // flag_base per component, and reserve non-overlapping flag ranges per
    // function so multiple frontend pipes can coexist safely.
    SmallVector<PipeInitInfo> initInfos;
    PipeInitGroups keyedInits;
    LogicalResult status = success();

    auto collectInit = [&initInfos, &keyedInits, &status](auto initOp) {
      if (failed(status))
        return;
      status = collectPeerAwareInit(initOp, initInfos, keyedInits);
    };

    moduleOp.walk([&collectInit](InitializeL2LPipeOp initOp) {
      collectInit(initOp);
    });
    moduleOp.walk([&collectInit](InitializeL2G2LPipeOp initOp) {
      collectInit(initOp);
    });
    if (failed(status))
      return failure();

    auto componentsOr = buildPeerAwareComponents(initInfos, keyedInits);
    if (failed(componentsOr))
      return failure();

    OpBuilder builder(moduleOp.getContext());
    PipeFlagUsage usedByFunc;
    for (const PipeComponent &component : *componentsOr) {
      auto chosenBaseOr = chooseFlagBaseForComponent(component, usedByFunc);
      if (failed(chosenBaseOr))
        return failure();
      auto flagBaseAttr = builder.getI32IntegerAttr(*chosenBaseOr);
      for (Operation *op : component.ops)
        setFlagBaseAttr(op, flagBaseAttr);
    }

    return success();
  }

  LogicalResult materializeResolvedAddresses(ModuleOp moduleOp) const {
    // Resolve frontend reserve/import ops to plain constant local addresses so
    // downstream lowering only sees ordinary SSA values.
    SmallVector<Operation *> eraseOps;

    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
      OpBuilder builder(funcOp.getContext());

      SmallVector<ReserveBufferOp> reserveOps;
      funcOp.walk([&reserveOps](ReserveBufferOp reserveOp) {
        reserveOps.push_back(reserveOp);
      });
      for (ReserveBufferOp reserveOp : reserveOps) {
        if (failed(materializeReserveBufferBase(builder, reserveOp, eraseOps)))
          return failure();
      }

      SmallVector<ImportReservedBufferOp> importOps;
      funcOp.walk([&importOps](ImportReservedBufferOp importOp) {
        importOps.push_back(importOp);
      });
      for (ImportReservedBufferOp importOp : importOps) {
        if (failed(
                materializeImportedReserveBufferBase(builder, importOp, eraseOps)))
          return failure();
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
