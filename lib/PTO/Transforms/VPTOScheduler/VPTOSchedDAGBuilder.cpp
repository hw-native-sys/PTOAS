// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAGBuilder.cpp - VPTO scheduling DAG builder -------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"

#include "PTO/IR/PTO.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

#include "../Utils.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<std::unique_ptr<VPTOSchedDAG>>
VPTOSchedDAGBuilder::build(const VPTOSchedRegion &region) const {
  auto dag = std::make_unique<VPTOSchedDAG>(region);
  buildSSAEdges(*dag);
  buildMemoryEdges(*dag);
  buildImplicitAndSyncEdges(*dag);
  buildModelFallbackEdges(*dag);
  if (failed(dag->computeCriticalPaths()))
    return failure();
  dag->resetDependencyCounts();
  return std::move(dag);
}

namespace {
struct ResolvedMemoryAccess {
  VPTOMemoryAccess semantics;
  Value aliasRoot;
};

static Value getAliasRoot(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  while (Operation *definingOp = value.getDefiningOp()) {
    if (!visited.insert(definingOp).second)
      break;
    std::optional<std::pair<Value, Value>> alias =
        getOperationAliasInfo(definingOp);
    if (!alias || alias->first != value || !alias->second)
      break;
    value = alias->second;
  }
  return value;
}

static SmallVector<ResolvedMemoryAccess>
resolveMemoryAccesses(const VPTOSchedulingSemantics &semantics) {
  SmallVector<ResolvedMemoryAccess> accesses;
  accesses.reserve(semantics.memoryAccesses.size());
  for (const VPTOMemoryAccess &memoryAccess : semantics.memoryAccesses) {
    ResolvedMemoryAccess access{memoryAccess, {}};
    if (access.semantics.address) {
      access.aliasRoot = getAliasRoot(access.semantics.address);
      if (access.aliasRoot != access.semantics.address) {
        access.semantics.byteOffset.reset();
        access.semantics.byteSize.reset();
      }
    }
    accesses.push_back(std::move(access));
  }
  return accesses;
}

static bool mayAlias(const ResolvedMemoryAccess &lhs,
                     const ResolvedMemoryAccess &rhs) {
  if (lhs.semantics.addressSpace && rhs.semantics.addressSpace &&
      lhs.semantics.addressSpace != rhs.semantics.addressSpace)
    return false;
  if (!lhs.semantics.address || !rhs.semantics.address)
    return true;
  if (lhs.aliasRoot == rhs.aliasRoot && lhs.semantics.byteOffset &&
      lhs.semantics.byteSize && rhs.semantics.byteOffset &&
      rhs.semantics.byteSize) {
    int64_t lhsEnd;
    int64_t rhsEnd;
    if (!llvm::AddOverflow(*lhs.semantics.byteOffset, *lhs.semantics.byteSize,
                           lhsEnd) &&
        !llvm::AddOverflow(*rhs.semantics.byteOffset, *rhs.semantics.byteSize,
                           rhsEnd))
      return *lhs.semantics.byteOffset < rhsEnd &&
             *rhs.semantics.byteOffset < lhsEnd;
  }
  // Different roots in the same physical space remain conservative: memory
  // planning may have assigned overlapping ranges to distinct SSA roots.
  return true;
}

static bool needsMemoryOrder(const ResolvedMemoryAccess &lhs,
                             const ResolvedMemoryAccess &rhs) {
  if (!mayAlias(lhs, rhs))
    return false;
  if (lhs.semantics.ordered || rhs.semantics.ordered || lhs.semantics.unknown ||
      rhs.semantics.unknown)
    return true;
  return lhs.semantics.writes || rhs.semantics.writes;
}

static bool isPostUpdateAddress(const VPTOSUnit &producer, Value value) {
  return llvm::any_of(
      producer.getSemantics().effects, [&](const VPTOSchedulingEffect &effect) {
        return effect.kind == VPTOSchedulingEffectKind::PostUpdate &&
               effect.value == value;
      });
}

} // namespace

void VPTOSchedDAGBuilder::buildMemoryEdges(VPTOSchedDAG &dag) const {
  SmallVector<SmallVector<ResolvedMemoryAccess>> accesses;
  accesses.reserve(dag.getUnits().size());
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits())
    accesses.push_back(resolveMemoryAccesses(unit->getSemantics()));

  for (size_t successorIndex = 0; successorIndex < accesses.size();
       ++successorIndex) {
    if (accesses[successorIndex].empty())
      continue;
    for (size_t predecessorIndex = 0; predecessorIndex < successorIndex;
         ++predecessorIndex) {
      bool ordered =
          llvm::any_of(accesses[predecessorIndex],
                       [&](const ResolvedMemoryAccess &predecessor) {
                         return llvm::any_of(
                             accesses[successorIndex],
                             [&](const ResolvedMemoryAccess &successor) {
                               return needsMemoryOrder(predecessor, successor);
                             });
                       });
      if (!ordered)
        continue;
      dag.addEdge(*dag.getUnits()[predecessorIndex],
                  *dag.getUnits()[successorIndex], VPTOSchedEdgeKind::Memory,
                  VPTOSchedEdgeStrength::Must,
                  /*latency=*/0, "may-alias memory access in original order");
    }
  }
}

void VPTOSchedDAGBuilder::buildImplicitAndSyncEdges(VPTOSchedDAG &dag) const {
  llvm::StringMap<VPTOSUnit *> lastWrite;
  llvm::StringMap<SmallVector<VPTOSUnit *>> readsSinceWrite;
  VPTOSUnit *lastBarrier = nullptr;

  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    if (lastBarrier && lastBarrier != &unit)
      dag.addEdge(*lastBarrier, unit, VPTOSchedEdgeKind::Sync,
                  VPTOSchedEdgeStrength::Must, 0, "after scheduling barrier");

    for (const VPTOSchedulingEffect &effect : unit.getSemantics().effects) {
      if (effect.kind == VPTOSchedulingEffectKind::Barrier) {
        for (const std::unique_ptr<VPTOSUnit> &prior : dag.getUnits()) {
          if (prior->getOriginalIndex() >= unit.getOriginalIndex())
            break;
          dag.addEdge(*prior, unit, VPTOSchedEdgeKind::Sync,
                      VPTOSchedEdgeStrength::Must, 0,
                      "before scheduling barrier");
        }
        lastBarrier = &unit;
        continue;
      }
      if (effect.resource.empty())
        continue;
      if (effect.kind == VPTOSchedulingEffectKind::ImplicitRead) {
        if (VPTOSUnit *writer = lastWrite.lookup(effect.resource))
          dag.addEdge(*writer, unit, VPTOSchedEdgeKind::Data,
                      VPTOSchedEdgeStrength::Must, 1,
                      Twine("implicit read of ") + effect.resource);
        readsSinceWrite[effect.resource].push_back(&unit);
        continue;
      }
      if (effect.kind != VPTOSchedulingEffectKind::ImplicitWrite)
        continue;
      if (VPTOSUnit *writer = lastWrite.lookup(effect.resource))
        dag.addEdge(*writer, unit, VPTOSchedEdgeKind::Output,
                    VPTOSchedEdgeStrength::Must, 0,
                    Twine("implicit write of ") + effect.resource);
      for (VPTOSUnit *reader : readsSinceWrite[effect.resource])
        dag.addEdge(*reader, unit, VPTOSchedEdgeKind::Anti,
                    VPTOSchedEdgeStrength::Must, 0,
                    Twine("implicit anti-dependence on ") + effect.resource);
      readsSinceWrite[effect.resource].clear();
      lastWrite[effect.resource] = &unit;
    }
  }
}

void VPTOSchedDAGBuilder::buildSSAEdges(VPTOSchedDAG &dag) const {
  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    Operation *op = unit.getOperation();

    for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
      Operation *definingOp = operand.getDefiningOp();
      VPTOSUnit *predecessor = definingOp ? dag.lookup(definingOp) : nullptr;
      if (!predecessor) {
        dag.addLiveIn(operand);
        continue;
      }
      unsigned latency =
          model ? model->getSchedClass(predecessor->getOperation()).writeLatency
                : 1;
      std::string reason =
          (isPostUpdateAddress(*predecessor, operand)
               ? Twine("post-update address operand #") + Twine(operandIndex)
               : Twine("ssa operand #") + Twine(operandIndex))
              .str();
      dag.addEdge(*predecessor, unit, VPTOSchedEdgeKind::Data,
                  VPTOSchedEdgeStrength::Must, latency, reason);
    }

    for (Value result : op->getResults()) {
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !dag.lookup(user); }))
        dag.addLiveOut(result);
    }
  }
}

void VPTOSchedDAGBuilder::buildModelFallbackEdges(VPTOSchedDAG &dag) const {
  if (!model)
    return;
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  for (size_t index = 0; index < units.size(); ++index) {
    VPTOSUnit &unit = *units[index];
    if (model->getSchedClass(unit.getOperation()).known)
      continue;
    if (index != 0)
      dag.addEdge(*units[index - 1], unit, VPTOSchedEdgeKind::Artificial,
                  VPTOSchedEdgeStrength::Must, 0,
                  "unknown sched class preserves predecessor order");
    if (index + 1 != units.size())
      dag.addEdge(unit, *units[index + 1], VPTOSchedEdgeKind::Artificial,
                  VPTOSchedEdgeStrength::Must, 0,
                  "unknown sched class preserves successor order");
  }
}
