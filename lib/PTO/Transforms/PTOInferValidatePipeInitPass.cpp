// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tuple>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINFERVALIDATEPIPEINIT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr size_t kMinPeerPipeInitCount = 2;
constexpr int8_t kC2VDirMask = 1;
constexpr int8_t kV2CDirMask = 2;
constexpr int8_t kBidirectionalDirMask = 3;
constexpr unsigned kVisitedInitReserveSize = 16;
constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";

struct PipePeerKey {
  std::string ownerFunc;
  std::string reserveName;
  int8_t dirMask = 0;

  bool operator<(const PipePeerKey &other) const {
    return std::tie(ownerFunc, reserveName, dirMask) <
           std::tie(other.ownerFunc, other.reserveName, other.dirMask);
  }
};

enum class PipeSplitUsage {
  Unknown,
  SplitOnly,
  NoSplitOnly,
  Mixed,
};

struct PipeInitInfo {
  Operation *op = nullptr;
  func::FuncOp funcOp;
  int8_t dirMask = 0;
  PipeSplitUsage usage = PipeSplitUsage::Unknown;
  std::optional<bool> explicitNoSplit;
};

template <typename InitOpT> static Value getPipeResult(InitOpT op) {
  return op.getPipe();
}

template <typename InitOpT> static Value getLocalAddrOperand(InitOpT op) {
  return op.getLocalAddr();
}

template <typename InitOpT>
static std::optional<bool> getNoSplitAttr(InitOpT op) {
  if (auto attr = op.getNosplitAttr()) {
    return attr.getValue();
  }
  return std::nullopt;
}

template <typename InitOpT>
static void setNoSplitAttr(InitOpT op, BoolAttr attr) {
  op->setAttr("nosplit", attr);
}

static PipeSplitUsage classifyPipeUsage(Value pipe) {
  bool sawNoSplit = false;
  bool sawSplit = false;

  for (Operation *user : pipe.getUsers()) {
    int64_t split = -1;
    if (auto allocOp = dyn_cast<TAllocOp>(user)) {
      split = allocOp.getSplit();
    } else if (auto pushOp = dyn_cast<TPushOp>(user)) {
      split = pushOp.getSplit();
    } else if (auto popOp = dyn_cast<TPopOp>(user)) {
      split = popOp.getSplit();
    } else if (auto freeOp = dyn_cast<TFreeOp>(user)) {
      split = freeOp.getSplit();
    } else {
      continue;
    }

    if (split == 0) {
      sawNoSplit = true;
    } else {
      sawSplit = true;
    }

    if (sawNoSplit && sawSplit) {
      return PipeSplitUsage::Mixed;
    }
  }

  if (sawNoSplit) {
    return PipeSplitUsage::NoSplitOnly;
  }
  if (sawSplit) {
    return PipeSplitUsage::SplitOnly;
  }
  return PipeSplitUsage::Unknown;
}

static std::optional<bool> getUsageNoSplit(PipeSplitUsage usage) {
  switch (usage) {
  case PipeSplitUsage::Unknown:
    return std::nullopt;
  case PipeSplitUsage::SplitOnly:
    return false;
  case PipeSplitUsage::NoSplitOnly:
    return true;
  case PipeSplitUsage::Mixed:
    return std::nullopt;
  }
  return std::nullopt;
}

static std::string getFuncSymbol(func::FuncOp funcOp) {
  return funcOp.getSymName().str();
}

static PipePeerKey getGlobalTensorPipeKey(Operation *op, int8_t dirMask) {
  std::string id = "unknown";
  if (auto idAttr = op->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName)) {
    id = std::to_string(idAttr.getInt());
  } else {
    id = std::to_string(reinterpret_cast<uintptr_t>(op));
  }
  return PipePeerKey{"__pto_globaltensor_pipe", "id_" + id, dirMask};
}

static std::optional<PipePeerKey> getPipePeerKey(Value localAddr,
                                                 func::FuncOp currentFunc) {
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

static LogicalResult
resolveNoSplitComponent(ArrayRef<PipeInitInfo *> component, OpBuilder &builder) {
  std::optional<bool> explicitNoSplit;
  std::optional<bool> inferredNoSplit;

  for (PipeInitInfo *info : component) {
    if (info->usage == PipeSplitUsage::Mixed) {
      return info->op->emitOpError(
          "cannot mix 'split = 0' with 'split = 1', 'split = 2', "
          "'split = 3', or 'split = 4' on the "
          "same logical pipe");
    }

    if (!info->explicitNoSplit) {
      continue;
    }
    if (explicitNoSplit && *explicitNoSplit != *info->explicitNoSplit) {
      return info->op->emitOpError(
          "conflicting explicit 'nosplit' across peer pipe init ops");
    }
    explicitNoSplit = info->explicitNoSplit;
  }

  for (PipeInitInfo *info : component) {
    auto usageNoSplit = getUsageNoSplit(info->usage);
    if (!usageNoSplit) {
      continue;
    }
    if (inferredNoSplit && *inferredNoSplit != *usageNoSplit) {
      return info->op->emitOpError(
          "conflicting pipe split usage across peer pipe init ops");
    }
    inferredNoSplit = *usageNoSplit;
  }

  if (explicitNoSplit && inferredNoSplit && *explicitNoSplit != *inferredNoSplit) {
    for (PipeInitInfo *info : component) {
      if (!info->explicitNoSplit || *info->explicitNoSplit == *inferredNoSplit) {
        continue;
      }
      if (*info->explicitNoSplit) {
        return info->op->emitOpError(
            "explicit 'nosplit = true' conflicts with downstream users that "
            "require split = 1, 2, 3, or 4");
      }
      return info->op->emitOpError(
          "explicit 'nosplit = false' conflicts with downstream users that "
          "require split = 0");
    }
  }

  bool finalNoSplit =
      explicitNoSplit.value_or(inferredNoSplit.value_or(false));
  auto noSplitAttr = builder.getBoolAttr(finalNoSplit);
  for (PipeInitInfo *info : component) {
    if (auto initOp = dyn_cast<InitializeL2LPipeOp>(info->op)) {
      if (!initOp.getNosplitAttr()) {
        setNoSplitAttr(initOp, noSplitAttr);
      }
      continue;
    }

    auto initOp = cast<InitializeL2G2LPipeOp>(info->op);
    if (!initOp.getNosplitAttr()) {
      setNoSplitAttr(initOp, noSplitAttr);
    }
  }

  return success();
}

struct PTOInferValidatePipeInitPass
    : public mlir::pto::impl::PTOInferValidatePipeInitBase<
          PTOInferValidatePipeInitPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<PipeInitInfo> initInfos;
    llvm::DenseMap<Operation *, SmallVector<Operation *>> adjacency;
    std::map<PipePeerKey, SmallVector<Operation *>> keyedInits;

    auto collectInit = [&adjacency, &initInfos, &keyedInits](auto initOp) {
      PipeInitInfo &info = initInfos.emplace_back();
      info.op = initOp.getOperation();
      info.funcOp = initOp->template getParentOfType<func::FuncOp>();
      info.dirMask = initOp.getDirMask();
      info.usage = classifyPipeUsage(getPipeResult(initOp));
      info.explicitNoSplit = getNoSplitAttr(initOp);
      adjacency[info.op];

      auto recordAddr = [&info, &keyedInits](Value addr,
                                              int8_t effectiveDirMask) {
        if (!addr) {
          return;
        }
        auto key = getPipePeerKey(addr, info.funcOp);
        if (!key) {
          return;
        }
        key->dirMask = effectiveDirMask;
        keyedInits[*key].push_back(info.op);
      };

      auto recordGlobalTensor = [&info, &keyedInits](
                                   int8_t effectiveDirMask) {
        keyedInits[getGlobalTensorPipeKey(info.op, effectiveDirMask)].push_back(
            info.op);
      };

      if (auto l2g2l = dyn_cast<InitializeL2G2LPipeOp>(info.op)) {
        if (!l2g2l.getLocalAddr()) {
          if (info.dirMask == kBidirectionalDirMask) {
            recordGlobalTensor(kBidirectionalDirMask);
          } else {
            recordGlobalTensor(info.dirMask);
          }
          return;
        }
      }

      if (info.dirMask == kBidirectionalDirMask) {
        recordAddr(getLocalAddrOperand(initOp), kC2VDirMask);
        if (Value peerAddr = initOp.getPeerLocalAddr()) {
          recordAddr(peerAddr, kV2CDirMask);
        }
        return;
      }

      recordAddr(getLocalAddrOperand(initOp), info.dirMask);
    };

    moduleOp.walk([&](InitializeL2LPipeOp initOp) { collectInit(initOp); });
    moduleOp.walk([&](InitializeL2G2LPipeOp initOp) { collectInit(initOp); });

    for (const auto &it : keyedInits) {
      SmallVector<Operation *> uniqueOps;
      for (Operation *op : it.second) {
        if (std::find(uniqueOps.begin(), uniqueOps.end(), op) == uniqueOps.end()) {
          uniqueOps.push_back(op);
        }
      }
      if (uniqueOps.size() < kMinPeerPipeInitCount) {
        continue;
      }

      for (size_t i = 0; i < uniqueOps.size(); ++i) {
        for (size_t j = i + 1; j < uniqueOps.size(); ++j) {
          adjacency[uniqueOps[i]].push_back(uniqueOps[j]);
          adjacency[uniqueOps[j]].push_back(uniqueOps[i]);
        }
      }
    }

    llvm::DenseMap<Operation *, PipeInitInfo *> infoByOp;
    for (PipeInitInfo &info : initInfos) {
      infoByOp[info.op] = &info;
    }

    OpBuilder builder(moduleOp.getContext());
    llvm::SmallPtrSet<Operation *, kVisitedInitReserveSize> visited;
    for (PipeInitInfo &rootInfo : initInfos) {
      if (!visited.insert(rootInfo.op).second) {
        continue;
      }

      SmallVector<Operation *> stack{rootInfo.op};
      SmallVector<PipeInitInfo *> component;
      while (!stack.empty()) {
        Operation *current = stack.pop_back_val();
        component.push_back(infoByOp[current]);
        for (Operation *neighbor : adjacency[current]) {
          if (visited.insert(neighbor).second) {
            stack.push_back(neighbor);
          }
        }
      }

      if (failed(resolveNoSplitComponent(component, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInferValidatePipeInitPass() {
  return std::make_unique<PTOInferValidatePipeInitPass>();
}
