// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOSchedDAGBuilder.cpp - VPTO scheduling DAG builder -------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"

#include "PTO/IR/PTO.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
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
struct MemoryAccess {
  Value address;
  Value aliasRoot;
  Attribute addressSpace;
  std::optional<int64_t> byteOffset;
  std::optional<int64_t> byteSize;
  bool reads = false;
  bool writes = false;
  bool ordered = false;
  bool unknown = false;
};

static bool isMemoryAddress(Value value) {
  return value && isa<pto::PtrType, BaseMemRefType>(value.getType());
}

static Attribute getAddressSpace(Value value) {
  if (auto pointer = dyn_cast<pto::PtrType>(value.getType()))
    return pointer.getMemorySpace();
  if (auto memref = dyn_cast<BaseMemRefType>(value.getType()))
    return memref.getMemorySpace();
  return {};
}

static bool isStoreLikeName(StringRef name) {
  return name == "pto.store" || name == "pto.stg" || name == "pto.st_dev" ||
         name.starts_with("pto.vst") || name.starts_with("pto.pst");
}

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

static std::optional<int64_t> getElementByteSize(Value pointer) {
  Type elementType;
  if (auto pointerType = dyn_cast<pto::PtrType>(pointer.getType()))
    elementType = pointerType.getElementType();
  else if (auto memrefType = dyn_cast<BaseMemRefType>(pointer.getType()))
    elementType = memrefType.getElementType();
  if (!elementType)
    return std::nullopt;

  int64_t elementCount = 1;
  if (auto vectorType = dyn_cast<VectorType>(elementType)) {
    if (vectorType.isScalable())
      return std::nullopt;
    elementCount = vectorType.getNumElements();
    elementType = vectorType.getElementType();
  }
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;

  int64_t byteSize;
  if (llvm::MulOverflow(elementCount, static_cast<int64_t>(bitWidth / 8),
                        byteSize))
    return std::nullopt;
  return byteSize;
}

static std::optional<int64_t> getConstantOffset(Value offset) {
  APInt value;
  if (!matchPattern(offset, m_ConstantInt(&value)) ||
      !value.isSignedIntN(64))
    return std::nullopt;
  return value.getSExtValue();
}

template <typename OpTy>
static void setStaticIndexedRange(OpTy op, MemoryAccess &access) {
  if (access.address != op.getPtr() || access.aliasRoot != access.address)
    return;
  std::optional<int64_t> elementOffset = getConstantOffset(op.getOffset());
  std::optional<int64_t> elementByteSize =
      getElementByteSize(access.address);
  if (!elementOffset || !elementByteSize)
    return;
  int64_t byteOffset;
  if (llvm::MulOverflow(*elementOffset, *elementByteSize, byteOffset))
    return;
  access.byteOffset = byteOffset;
  access.byteSize = *elementByteSize;
}

static void setStaticAccessRange(Operation *op, MemoryAccess &access) {
  if (auto load = dyn_cast<PTOLoadOp>(op))
    return setStaticIndexedRange(load, access);
  if (auto store = dyn_cast<PTOStoreOp>(op))
    return setStaticIndexedRange(store, access);
  if (auto load = dyn_cast<PTOLdgOp>(op))
    return setStaticIndexedRange(load, access);
  if (auto store = dyn_cast<PTOStgOp>(op))
    return setStaticIndexedRange(store, access);
  if (auto load = dyn_cast<PTOLdDevOp>(op))
    return setStaticIndexedRange(load, access);
  if (auto store = dyn_cast<PTOStDevOp>(op))
    return setStaticIndexedRange(store, access);
}

static SmallVector<MemoryAccess> collectMemoryAccesses(Operation *op) {
  SmallVector<MemoryAccess> accesses;
  auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffects) {
    if (!isMemoryEffectFree(op)) {
      MemoryAccess access;
      access.writes = true;
      access.ordered = true;
      access.unknown = true;
      accesses.push_back(access);
    }
    return accesses;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffects.getEffects(effects);
  bool storeLike = isStoreLikeName(op->getName().getStringRef());
  for (const MemoryEffects::EffectInstance &effect : effects) {
    Value value = effect.getValue();
    if (value && !isMemoryAddress(value))
      continue;
    MemoryAccess access;
    access.address = value;
    access.aliasRoot = value ? getAliasRoot(value) : Value();
    access.addressSpace = value ? getAddressSpace(value) : Attribute();
    access.reads = isa<MemoryEffects::Read>(effect.getEffect());
    access.writes =
        isa<MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free>(
            effect.getEffect());
    if (value && storeLike)
      access.writes = true;
    access.unknown = !value || (!access.reads && !access.writes);
    setStaticAccessRange(op, access);
    accesses.push_back(access);
  }

  if (auto scheduling = dyn_cast<VPTOSchedulingOpInterface>(op)) {
    SmallVector<VPTOSchedulingEffect> effects;
    scheduling.getVPTOSchedulingEffects(effects);
    bool ordered =
        llvm::any_of(effects, [](const VPTOSchedulingEffect &effect) {
          return effect.kind == VPTOSchedulingEffectKind::AtomicMemory ||
                 effect.kind == VPTOSchedulingEffectKind::VolatileMemory;
        });
    if (ordered) {
      if (accesses.empty()) {
        MemoryAccess access;
        access.reads = true;
        access.writes = true;
        access.ordered = true;
        access.unknown = true;
        accesses.push_back(access);
      }
      for (MemoryAccess &access : accesses)
        access.ordered = true;
    }
  }
  return accesses;
}

static bool mayAlias(const MemoryAccess &lhs, const MemoryAccess &rhs) {
  if (lhs.addressSpace && rhs.addressSpace &&
      lhs.addressSpace != rhs.addressSpace)
    return false;
  if (!lhs.address || !rhs.address)
    return true;
  if (lhs.aliasRoot == rhs.aliasRoot && lhs.byteOffset && lhs.byteSize &&
      rhs.byteOffset && rhs.byteSize) {
    int64_t lhsEnd;
    int64_t rhsEnd;
    if (!llvm::AddOverflow(*lhs.byteOffset, *lhs.byteSize, lhsEnd) &&
        !llvm::AddOverflow(*rhs.byteOffset, *rhs.byteSize, rhsEnd))
      return *lhs.byteOffset < rhsEnd && *rhs.byteOffset < lhsEnd;
  }
  // Different roots in the same physical space remain conservative: memory
  // planning may have assigned overlapping ranges to distinct SSA roots.
  return true;
}

static bool needsMemoryOrder(const MemoryAccess &lhs, const MemoryAccess &rhs) {
  if (!mayAlias(lhs, rhs))
    return false;
  if (lhs.ordered || rhs.ordered || lhs.unknown || rhs.unknown)
    return true;
  return lhs.writes || rhs.writes;
}

static bool isPostUpdateAddress(Value value) {
  Operation *definingOp = value.getDefiningOp();
  auto scheduling = dyn_cast_or_null<VPTOSchedulingOpInterface>(definingOp);
  if (!scheduling)
    return false;
  SmallVector<VPTOSchedulingEffect> effects;
  scheduling.getVPTOSchedulingEffects(effects);
  return llvm::any_of(effects, [&](const VPTOSchedulingEffect &effect) {
    return effect.kind == VPTOSchedulingEffectKind::PostUpdate &&
           effect.value == value;
  });
}

static bool mayReferToSameEvent(const VPTOSchedulingEffect &lhs,
                                const VPTOSchedulingEffect &rhs) {
  auto lhsIdentity = dyn_cast_or_null<ArrayAttr>(lhs.attribute);
  auto rhsIdentity = dyn_cast_or_null<ArrayAttr>(rhs.attribute);
  if (!lhsIdentity || !rhsIdentity || lhsIdentity.size() < 2 ||
      rhsIdentity.size() < 2)
    return true;
  if (lhsIdentity[0] != rhsIdentity[0] ||
      lhsIdentity[1] != rhsIdentity[1])
    return false;
  if (lhsIdentity.size() == 3 && rhsIdentity.size() == 3)
    return lhsIdentity[2] == rhsIdentity[2];
  // A dynamic id may equal any static or dynamic id for the same pipe pair.
  return true;
}

static bool pipeEffectMatches(Attribute constraint, Attribute execution) {
  auto constraintPipe = dyn_cast_or_null<PipeAttr>(constraint);
  auto executionPipe = dyn_cast_or_null<PipeAttr>(execution);
  if (!constraintPipe || !executionPipe)
    return true;
  return constraintPipe.getPipe() == PIPE::PIPE_ALL ||
         executionPipe.getPipe() == PIPE::PIPE_ALL ||
         constraintPipe == executionPipe;
}

static bool mayReferToSameBuffer(const VPTOSchedulingEffect &lhs,
                                 const VPTOSchedulingEffect &rhs) {
  if (lhs.attribute && rhs.attribute)
    return lhs.attribute == rhs.attribute;
  // A dynamic id may equal any static or dynamic buffer id.
  return true;
}
} // namespace

void VPTOSchedDAGBuilder::buildMemoryEdges(VPTOSchedDAG &dag) const {
  SmallVector<SmallVector<MemoryAccess>> accesses;
  accesses.reserve(dag.getUnits().size());
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits())
    accesses.push_back(collectMemoryAccesses(unit->getOperation()));

  for (size_t successorIndex = 0; successorIndex < accesses.size();
       ++successorIndex) {
    if (accesses[successorIndex].empty())
      continue;
    for (size_t predecessorIndex = 0; predecessorIndex < successorIndex;
         ++predecessorIndex) {
      bool ordered = llvm::any_of(
          accesses[predecessorIndex], [&](const MemoryAccess &predecessor) {
            return llvm::any_of(
                accesses[successorIndex], [&](const MemoryAccess &successor) {
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
  struct EventSignal {
    VPTOSchedulingEffect effect;
    VPTOSUnit *unit;
  };
  struct PipeGate {
    Attribute pipe;
    VPTOSUnit *unit;
  };
  llvm::StringMap<VPTOSUnit *> lastWrite;
  llvm::StringMap<SmallVector<VPTOSUnit *>> readsSinceWrite;
  SmallVector<EventSignal> eventSignals;
  SmallVector<EventSignal> bufferAcquires;
  SmallVector<EventSignal> bufferReleases;
  SmallVector<PipeGate> pipeGates;
  SmallVector<PipeGate> pipeExecutions;
  VPTOSUnit *lastBarrier = nullptr;

  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    if (lastBarrier && lastBarrier != &unit)
      dag.addEdge(*lastBarrier, unit, VPTOSchedEdgeKind::Sync,
                  VPTOSchedEdgeStrength::Must, 0, "after scheduling barrier");

    auto scheduling = dyn_cast<VPTOSchedulingOpInterface>(unit.getOperation());
    if (!scheduling)
      continue;
    SmallVector<VPTOSchedulingEffect> effects;
    scheduling.getVPTOSchedulingEffects(effects);
    for (const VPTOSchedulingEffect &effect : effects) {
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
      if (effect.kind == VPTOSchedulingEffectKind::Event) {
        if (effect.resource == "signal") {
          eventSignals.push_back({effect, &unit});
          continue;
        }
        if (effect.resource == "wait") {
          for (const EventSignal &signal : eventSignals) {
            if (!mayReferToSameEvent(signal.effect, effect))
              continue;
            dag.addEdge(*signal.unit, unit, VPTOSchedEdgeKind::Sync,
                        VPTOSchedEdgeStrength::Must, 0,
                        "event signal before wait");
          }
          continue;
        }
      }
      if (effect.kind == VPTOSchedulingEffectKind::Pipe) {
        if (effect.resource == "execute") {
          for (const PipeGate &gate : pipeGates) {
            if (!pipeEffectMatches(gate.pipe, effect.attribute))
              continue;
            dag.addEdge(*gate.unit, unit, VPTOSchedEdgeKind::Sync,
                        VPTOSchedEdgeStrength::Must, 0,
                        "pipe synchronization before execution");
          }
          pipeExecutions.push_back({effect.attribute, &unit});
          continue;
        }
        if (effect.resource == "signal-source" ||
            effect.resource == "barrier") {
          for (const PipeGate &execution : pipeExecutions) {
            if (!pipeEffectMatches(effect.attribute, execution.pipe))
              continue;
            dag.addEdge(*execution.unit, unit, VPTOSchedEdgeKind::Sync,
                        VPTOSchedEdgeStrength::Must, 0,
                        effect.resource == "barrier"
                            ? "pipe execution before barrier"
                            : "source pipe execution before signal");
          }
        }
        if (effect.resource == "wait-destination" ||
            effect.resource == "barrier")
          pipeGates.push_back({effect.attribute, &unit});
        continue;
      }
      if (effect.kind == VPTOSchedulingEffectKind::BufferId) {
        if (effect.resource == "acquire") {
          for (const EventSignal &release : bufferReleases) {
            if (!mayReferToSameBuffer(release.effect, effect))
              continue;
            dag.addEdge(*release.unit, unit, VPTOSchedEdgeKind::Sync,
                        VPTOSchedEdgeStrength::Must, 0,
                        "buffer release before acquire");
          }
          bufferAcquires.push_back({effect, &unit});
          continue;
        }
        if (effect.resource == "release") {
          for (const EventSignal &acquire : bufferAcquires) {
            if (!mayReferToSameBuffer(acquire.effect, effect))
              continue;
            dag.addEdge(*acquire.unit, unit, VPTOSchedEdgeKind::Sync,
                        VPTOSchedEdgeStrength::Must, 0,
                        "buffer acquire before release");
          }
          bufferReleases.push_back({effect, &unit});
          continue;
        }
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
          (isPostUpdateAddress(operand)
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
