// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Resolve all families through one canonical instance table. Pipe lifecycle
// calls inherit their object's specialization before entry-specific resolution.
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/VPTOBridgeRegistry.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <tuple>

namespace mlir::pto {
#define GEN_PASS_DEF_VPTORESOLVEBRIDGEINSTANCES
#include "PTO/Transforms/Passes.h.inc"

namespace {
static FailureOr<std::string> getCompileUnitDigest(ModuleOp module)
{
    // Bytecode retains full regions and attribute payloads regardless of the
    // process-wide assembly printer flags. Strip locations on a private copy so
    // source paths do not enter the digest or disappear from later diagnostics.
    OwningOpRef<ModuleOp> snapshot(module.clone());
    auto unknownLoc = UnknownLoc::get(module.getContext());
    snapshot->walk([unknownLoc](Operation* op) {
        op->setLoc(unknownLoc);
        for (Region& region : op->getRegions()) {
            for (Block& block : region) {
                for (BlockArgument argument : block.getArguments()) {
                    argument.setLoc(unknownLoc);
                }
            }
        }
    });
    std::string bytecode;
    llvm::raw_string_ostream stream(bytecode);
    if (failed(writeBytecodeToFile(snapshot->getOperation(), stream))) {
        return failure();
    }
    llvm::SHA256 hash;
    hash.update(stream.str());
    // Encode the digest using lowercase hexadecimal digits.
    return llvm::toHex(hash.final(), true);
}

// Same order as CV split partitioning: nearest section, then an explicit
// pto.kernel_kind on an enclosing function or module.
std::optional<BridgeCoreKind> findContextCore(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<SectionCubeOp>(parent)) {
      return BridgeCoreKind::Cube;
    }
    if (isa<SectionVectorOp>(parent)) {
      return BridgeCoreKind::Vector;
    }
    auto kind = parent->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
    if (kind) {
      return kind.getKernelKind() == FunctionKernelKind::Cube
                 ? BridgeCoreKind::Cube : BridgeCoreKind::Vector;
    }
  }
  return std::nullopt;
}

FailureOr<BridgeCoreKind> resolveCore(Operation *op, const BridgeFunctionDesc &desc) {
  std::optional<BridgeCoreKind> context = findContextCore(op);
  if (desc.core == BridgeCoreKind::Any) {
    return context.value_or(BridgeCoreKind::Any);
  }
  if (context == desc.core) {
    return desc.core;
  }
  op->emitError() << "bridge entry '" << stringifyBridgeEntryId(desc.id)
                  << "' requires core '" << stringifyBridgeCoreKind(desc.core)
                  << "', but its context is '"
                  << (context ? stringifyBridgeCoreKind(*context) : StringRef("none"))
                  << "'";
  return failure();
}

LogicalResult inheritPipeSpec(ModuleOp module) {
  auto result = module.walk([](BridgeObjectCreateOp create) {
    auto spec = create.getSpecAttr();
    if (!spec) {
      create.emitError("bridge object requires a structured specialization");
      return WalkResult::interrupt();
    }
    for (Operation *user : create.getResult().getUsers()) {
      auto call = dyn_cast<BridgeCallOp>(user);
      const auto *desc = call ? findBridgeFunction(call.getEntry()) : nullptr;
      const bool validUser = desc && desc->family == BridgeFamily::Pipe &&
                            !call.getArgs().empty() && call.getArgs().front() == create.getResult();
      if (!validUser) {
        create.emitError("bridge object may only be used as the handle of a Pipe lifecycle call");
        return WalkResult::interrupt();
      }
      const bool mismatchedSpec = call.getSpecAttr() && call.getSpecAttr() != spec;
      if (mismatchedSpec) {
        call.emitError("Pipe lifecycle specialization differs from its object");
        return WalkResult::interrupt();
      }
      call.setSpecAttr(spec);
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

struct ResolveInstances final
    : impl::VPTOResolveBridgeInstancesBase<ResolveInstances> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveInstances)

  /// Interns one instance per (entry, core, spec). Uniqued attributes give the
  /// canonical identity of a specialization, so equal specs share a symbol.
  class InstanceTable {
  public:
    InstanceTable(ModuleOp module, std::string digest)
        : module(module), builder(OpBuilder::atBlockBegin(module.getBody())),
          unitDigest(std::move(digest)) {}

    FlatSymbolRefAttr intern(const BridgeFunctionDesc &desc, Attribute spec,
                             BridgeCoreKind core, Location loc) {
      Key key{static_cast<uint32_t>(desc.id), static_cast<uint32_t>(core), spec};
      auto found = instances.find(key);
      if (found != instances.end()) {
        return found->second;
      }
      std::string symbol =
          desc.symbolBase.str() + "__" + unitDigest + "_" +
          stringifyBridgeCoreKind(core).str() + "_" + std::to_string(instances.size());
      auto instance = builder.create<BridgeInstanceOp>(
          loc, builder.getStringAttr(symbol),
          BridgeEntryAttr::get(module.getContext(), desc.id),
          BridgeCoreKindAttr::get(module.getContext(), core), spec);
      builder.setInsertionPointAfter(instance);
      auto reference = FlatSymbolRefAttr::get(instance.getSymNameAttr());
      instances.try_emplace(key, reference);
      return reference;
    }

  private:
    using Key = std::tuple<uint32_t, uint32_t, Attribute>;
    ModuleOp module;
    OpBuilder builder;
    std::string unitDigest;
    llvm::DenseMap<Key, FlatSymbolRefAttr> instances;
  };

  /// Points one bridge op at its instance, interning the instance on first use.
  static LogicalResult resolveOne(Operation *op, InstanceTable &instances) {
    auto entry = op->getAttrOfType<BridgeEntryAttr>("entry");
    Attribute spec = op->getAttr("spec");
    const auto *desc = entry ? findBridgeFunction(entry.getValue()) : nullptr;
    if (!desc) {
      return op->emitError("bridge op requires a registered entry");
    }
    if (!spec && desc->family != BridgeFamily::Print) {
      return op->emitError("bridge instance requires a structured specialization");
    }
    FailureOr<BridgeCoreKind> core = resolveCore(op, *desc);
    if (failed(core)) {
      return failure();
    }
    op->setAttr("instance", instances.intern(*desc, spec, *core, op->getLoc()));
    return success();
  }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        if (failed(inheritPipeSpec(module))) {
            signalPassFailure();
            return;
        }
        if (!module.getOps<BridgeInstanceOp>().empty()) {
            module.emitError("bridge instances are already resolved");
            signalPassFailure();
            return;
        }
        FailureOr<std::string> digest = getCompileUnitDigest(module);
        if (failed(digest)) {
            module.emitError("failed to serialize bridge compile unit for its digest");
            signalPassFailure();
            return;
        }
        InstanceTable instances(module, std::move(*digest));
        auto result = module.walk([&instances](Operation* op) {
            if (!isa<BridgeCallOp, BridgeObjectCreateOp>(op)) {
                return WalkResult::advance();
            }
            return failed(resolveOne(op, instances)) ? WalkResult::interrupt() : WalkResult::advance();
        });
        if (result.wasInterrupted()) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> createVPTOResolveBridgeInstancesPass() {
  return std::make_unique<ResolveInstances>();
}
} // namespace mlir::pto
