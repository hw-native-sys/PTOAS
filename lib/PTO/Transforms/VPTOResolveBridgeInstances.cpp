// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOResolveBridgeInstances.cpp - resolve bridge instances --------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_VPTORESOLVEBRIDGEINSTANCES
#define GEN_PASS_DEF_VPTORESOLVEBRIDGEINSTANCES
#include "PTO/Transforms/Passes.h.inc"

namespace {

static FailureOr<BridgeCoreKind> getBridgeCore(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto kind = func ? func->getAttrOfType<FunctionKernelKindAttr>(
                         FunctionKernelKindAttr::name)
                   : FunctionKernelKindAttr();
  if (!kind) {
    auto module = op->getParentOfType<ModuleOp>();
    kind = module ? module->getAttrOfType<FunctionKernelKindAttr>(
                        FunctionKernelKindAttr::name)
                  : FunctionKernelKindAttr();
  }
  if (!kind) {
    return failure();
  }
  switch (kind.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return BridgeCoreKind::Cube;
  case FunctionKernelKind::Vector:
    return BridgeCoreKind::Vector;
  default:
    return failure();
  }
}

static std::string buildInstanceKey(StringRef entry, Attribute spec,
                                    BridgeCoreKind core,
                                    ArrayRef<StringRef> lifecycle = {}) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << entry << "|core=" << stringifyBridgeCore(core) << "|";
  spec.print(os);
  for (StringRef lifecycleEntry : lifecycle) {
    os << "|" << lifecycleEntry;
  }
  return key;
}

static std::string instanceSymbol(const BridgeFunctionDesc &desc,
                                  unsigned instanceId) {
  return desc.symbolBase.str() + "__" + std::to_string(instanceId);
}

struct PipeSymbols {
  std::string init;
  std::string size;
  std::string push;
  std::string pop;
  std::string free;
};

static LogicalResult resolvePipeInstances(ModuleOp module) {
  llvm::StringMap<PipeSymbols> instances;
  unsigned nextId = 0;
  WalkResult result = module.walk([&](BridgeObjectCreateOp create) {
    bool isPipeInit = create.getEntry() == BridgeEntryId::PipeInit;
    if (!isPipeInit) {
      return WalkResult::advance();
    }
    auto core = getBridgeCore(create);
    Attribute spec = create.getSpecAttr();
    bool invalidSpec =
        failed(core) || !isa_and_nonnull<BridgePipeSpecAttr>(spec);
    if (invalidSpec) {
      create.emitError(
          "Pipe bridge object requires a core and structured spec");
      return WalkResult::interrupt();
    }
    SmallVector<BridgeCallOp> calls;
    SmallVector<StringRef> lifecycle;
    for (Operation *user : create.getResult().getUsers()) {
      auto call = dyn_cast<BridgeCallOp>(user);
      if (!call) {
        create.emitError("Pipe bridge object has a non-bridge lifecycle user");
        return WalkResult::interrupt();
      }
      const BridgeFunctionDesc *callDesc = findBridgeFunction(call.getEntry());
      if (!callDesc || callDesc->family != BridgeFamily::Pipe) {
        call.emitError("Pipe bridge object is used by another bridge family");
        return WalkResult::interrupt();
      }
      calls.push_back(call);
      lifecycle.push_back(stringifyBridgeEntryId(call.getEntry()));
    }
    llvm::sort(lifecycle);
    std::string key = buildInstanceKey(
        stringifyBridgeEntryId(create.getEntry()), spec, *core, lifecycle);
    auto found = instances.find(key);
    if (found == instances.end()) {
      unsigned id = nextId++;
      auto getSymbol = [id](BridgeEntryId entry) {
        return instanceSymbol(*findBridgeFunction(entry), id);
      };
      found =
          instances
              .try_emplace(key, PipeSymbols{getSymbol(BridgeEntryId::PipeInit),
                                            getSymbol(BridgeEntryId::PipeSize),
                                            getSymbol(BridgeEntryId::PipePush),
                                            getSymbol(BridgeEntryId::PipePop),
                                            getSymbol(BridgeEntryId::PipeFree)})
              .first;
    }
    const PipeSymbols &symbols = found->second;
    create.setInstanceKeyAttr(StringAttr::get(module.getContext(), key));
    create.setCalleeAttr(StringAttr::get(module.getContext(), symbols.init));
    create.setSizeCalleeAttr(
        StringAttr::get(module.getContext(), symbols.size));
    for (BridgeCallOp call : calls) {
      StringRef symbol;
      bool isPush = call.getEntry() == BridgeEntryId::PipePush;
      if (isPush) {
        symbol = symbols.push;
      } else if (call.getEntry() == BridgeEntryId::PipePop) {
        symbol = symbols.pop;
      } else if (call.getEntry() == BridgeEntryId::PipeFree) {
        symbol = symbols.free;
      } else {
        call.emitError("unsupported Pipe bridge lifecycle entry");
        return WalkResult::interrupt();
      }
      call.setSpecAttr(spec);
      call.setInstanceKeyAttr(StringAttr::get(module.getContext(), key));
      call.setCalleeAttr(StringAttr::get(module.getContext(), symbol));
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

static LogicalResult resolveCubeInstances(ModuleOp module) {
  llvm::StringMap<std::string> instances;
  unsigned nextId = 0;
  WalkResult result = module.walk([&](BridgeCallOp call) {
    const BridgeFunctionDesc *desc = findBridgeFunction(call.getEntry());
    if (!desc || desc->family != BridgeFamily::Cube) {
      return WalkResult::advance();
    }
    auto core = getBridgeCore(call);
    Attribute spec = call.getSpecAttr();
    bool invalidSpec = failed(core) ||
                       !isa_and_nonnull<BridgeCubeSpecAttr>(spec) ||
                       desc->core != *core;
    if (invalidSpec) {
      call.emitError(
          "Cube bridge call requires its registered core and structured spec");
      return WalkResult::interrupt();
    }
    std::string key =
        buildInstanceKey(stringifyBridgeEntryId(call.getEntry()), spec, *core);
    auto found = instances.find(key);
    if (found == instances.end()) {
      found = instances.try_emplace(key, instanceSymbol(*desc, nextId++)).first;
    }
    call.setInstanceKeyAttr(StringAttr::get(module.getContext(), key));
    call.setCalleeAttr(StringAttr::get(module.getContext(), found->second));
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

struct VPTOResolveBridgeInstancesPass final
    : public impl::VPTOResolveBridgeInstancesBase<
          VPTOResolveBridgeInstancesPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOResolveBridgeInstancesPass)

  void runOnOperation() override {
    LogicalResult pipeResult = resolvePipeInstances(getOperation());
    LogicalResult cubeResult = resolveCubeInstances(getOperation());
    bool failedResolution = failed(pipeResult) || failed(cubeResult);
    if (failedResolution) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVPTOResolveBridgeInstancesPass() {
  return std::make_unique<VPTOResolveBridgeInstancesPass>();
}

} // namespace pto
} // namespace mlir
