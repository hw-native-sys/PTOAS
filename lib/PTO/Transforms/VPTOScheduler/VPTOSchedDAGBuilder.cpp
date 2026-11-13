// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAGBuilder.cpp - VPTO scheduling DAG builder -------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/Analysis/PTOMemoryAliasAnalysis.h"
#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <climits>
#include <limits>

#include "Utils.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<std::unique_ptr<VPTOSchedDAG>>
VPTOSchedDAGBuilder::build(const VPTOSchedRegion &region) const {
  VPTOScheduleFailure failure;
  return build(region, failure);
}

FailureOr<std::unique_ptr<VPTOSchedDAG>> VPTOSchedDAGBuilder::build(
    const VPTOSchedRegion &region, VPTOScheduleFailure &failure) const {
  uint64_t nodeCount = region.operations.size();
  if (limits && nodeCount > limits->maxNodes) {
    failure = {VPTOScheduleFailureKind::Budget, "nodes", nodeCount,
               limits->maxNodes, {}};
    return mlir::failure();
  }

  auto dag = std::make_unique<VPTOSchedDAG>(region);
  if (failed(buildSSAEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildMemoryEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildImplicitAndSyncEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildRematerializationClusterEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildModelFallbackEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(consumeWork(failure, dag->getUnits().size()))) {
    return mlir::failure();
  }
  if (failed(consumeWork(failure, dag->getEdges().size()))) {
    return mlir::failure();
  }
  if (failed(dag->computeCriticalPaths())) {
    failure.kind = VPTOScheduleFailureKind::Scheduling;
    failure.detail = "dependency DAG contains a Must-edge cycle";
    return mlir::failure();
  }
  dag->resetDependencyCounts();
  return std::move(dag);
}

LogicalResult
VPTOSchedDAGBuilder::consumeWork(VPTOScheduleFailure &failure,
                                 uint64_t amount) const {
  if (!budget || budget->consume(amount)) {
    return success();
  }
  uint64_t limit = budget->getLimit();
  uint64_t actual = limit == std::numeric_limits<uint64_t>::max()
                        ? limit
                        : limit + 1;
  failure = {VPTOScheduleFailureKind::Budget, "work-units", actual, limit, {}};
  return mlir::failure();
}

LogicalResult VPTOSchedDAGBuilder::addEdge(
    VPTOSchedDAG &dag, VPTOSUnit &predecessor, VPTOSUnit &successor,
    VPTOSchedEdgeKind kind, VPTOSchedEdgeStrength strength, unsigned latency,
    Twine reason, VPTOScheduleFailure &failure) const {
  uint64_t edgeCount = dag.getEdges().size();
  if (limits && edgeCount >= limits->maxEdges) {
    uint64_t actual = edgeCount == std::numeric_limits<uint64_t>::max()
                          ? edgeCount
                          : edgeCount + 1;
    failure = {VPTOScheduleFailureKind::Budget, "edges", actual,
               limits->maxEdges, {}};
    return mlir::failure();
  }
  if (failed(consumeWork(failure))) {
    return mlir::failure();
  }
  dag.addEdge(predecessor, successor, kind, strength, latency, reason);
  return success();
}

namespace {
constexpr unsigned kVisitedOperationInlineCapacity = mlir::pto::kValue8;

struct FrontierAccess {
  const VPTOSUnit *unit = nullptr;
  ResolvedMemoryAccess access;
};

static bool needsMemoryOrder(const ResolvedMemoryAccess &lhs,
                             const ResolvedMemoryAccess &rhs) {
  if (!mayAlias(lhs, rhs))
    return false;
  if (lhs.semantics.ordered || rhs.semantics.ordered || lhs.semantics.unknown ||
      rhs.semantics.unknown)
    return true;
  return lhs.semantics.writes || rhs.semantics.writes;
}

struct MemoryRange {
  int64_t begin = 0;
  int64_t end = 0;
};

static std::optional<MemoryRange>
getMemoryRange(std::optional<int64_t> byteOffset,
               std::optional<int64_t> byteSize) {
  if (!byteOffset || !byteSize) {
    return std::nullopt;
  }
  int64_t end;
  bool overflow = llvm::AddOverflow(*byteOffset, *byteSize, end) != 0;
  if (overflow) {
    return std::nullopt;
  }
  return MemoryRange{*byteOffset, end};
}

static bool containsMemoryRange(const MemoryRange &outer,
                                const MemoryRange &inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
}

static bool coversAddressSpace(const ResolvedMemoryAccess &prior,
                               const ResolvedMemoryAccess &current) {
  if (!current.semantics.addressSpace) {
    return true;
  }
  return prior.semantics.addressSpace &&
         current.semantics.addressSpace == prior.semantics.addressSpace;
}

static bool containsMemoryRange(const ResolvedMemoryAccess &prior,
                                const ResolvedMemoryAccess &current) {
  if (prior.semantics.byteSize && *prior.semantics.byteSize > 0 &&
      current.semantics.byteSize && *current.semantics.byteSize > 0) {
    if (std::optional<int64_t> priorBegin =
            getExactByteDifference(current, prior)) {
      int64_t priorEnd;
      if (llvm::AddOverflow(*priorBegin, *prior.semantics.byteSize,
                            priorEnd) == 0 &&
          *priorBegin >= 0 && priorEnd <= *current.semantics.byteSize) {
        return true;
      }
    }
  }

  std::optional<MemoryRange> currentAbsolute = getMemoryRange(
      current.absoluteByteOffset, current.semantics.byteSize);
  std::optional<MemoryRange> priorAbsolute =
      getMemoryRange(prior.absoluteByteOffset, prior.semantics.byteSize);
  if (currentAbsolute && priorAbsolute &&
      containsMemoryRange(*currentAbsolute, *priorAbsolute)) {
    return true;
  }

  std::optional<MemoryRange> currentRelative = getMemoryRange(
      current.semantics.byteOffset, current.semantics.byteSize);
  std::optional<MemoryRange> priorRelative =
      getMemoryRange(prior.semantics.byteOffset, prior.semantics.byteSize);
  return current.aliasRoot && current.aliasRoot == prior.aliasRoot &&
         currentRelative && priorRelative &&
         containsMemoryRange(*currentRelative, *priorRelative);
}

static bool hasMemoryRange(const ResolvedMemoryAccess &access) {
  return (access.absoluteByteRange && access.semantics.byteSize) ||
         getMemoryRange(access.absoluteByteOffset, access.semantics.byteSize) ||
         getMemoryRange(access.semantics.byteOffset,
                        access.semantics.byteSize);
}

/// Return whether replacing `prior` with `current` preserves every future
/// ordering edge that `prior` could require. Merely overlapping ranges is not
/// sufficient: a later access may overlap the part of `prior` that lies
/// outside `current`, leaving no transitive path through `current`.
static bool subsumesMemoryFrontierAccess(
    const ResolvedMemoryAccess &prior,
    const ResolvedMemoryAccess &current) {
  bool closesFrontier = current.semantics.writes ||
                        current.semantics.ordered || current.semantics.unknown;
  if (!closesFrontier || !mayAlias(prior, current)) {
    return false;
  }

  if (!coversAddressSpace(prior, current)) {
    return false;
  }
  if (containsMemoryRange(prior, current)) {
    return true;
  }

  // Without any usable interval the alias model treats `current` as covering
  // its complete compatible address space, so it safely represents every
  // prior entry in that space. If it has a bounded interval, retain any prior
  // entry that was not proven to be contained above.
  return !hasMemoryRange(current);
}

static bool isPostUpdateAddress(const VPTOSUnit &producer, Value value) {
  return llvm::any_of(
      producer.getSemantics().effects, [&](const VPTOSchedulingEffect &effect) {
        return effect.kind == VPTOSchedulingEffectKind::PostUpdate &&
               effect.value == value;
      });
}

template <typename ConsumeWork>
static LogicalResult
collectMemoryPredecessors(ArrayRef<FrontierAccess> frontier,
                          ArrayRef<ResolvedMemoryAccess> currentAccesses,
                          ConsumeWork &&consumeWork,
                          SmallPtrSetImpl<const VPTOSUnit *> &predecessors) {
  for (const FrontierAccess &prior : frontier) {
    for (const ResolvedMemoryAccess &current : currentAccesses) {
      if (failed(consumeWork())) {
        return mlir::failure();
      }
      if (needsMemoryOrder(prior.access, current)) {
        predecessors.insert(prior.unit);
        break;
      }
    }
  }
  return success();
}

template <typename ConsumeWork>
static FailureOr<SmallVector<SmallVector<ResolvedMemoryAccess>>>
collectResolvedMemoryAccesses(const VPTOSchedDAG &dag,
                              PTOAddressAnalysis *addressAnalysis,
                              ConsumeWork &&consumeWork) {
  SmallVector<SmallVector<ResolvedMemoryAccess>> accesses;
  accesses.reserve(dag.getUnits().size());
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (failed(consumeWork(unit->getSemantics().memoryAccesses.size()))) {
      return failure();
    }
    accesses.push_back(resolveMemoryAccesses(
        unit->getOperation(), unit->getSemantics(), addressAnalysis));
  }
  return accesses;
}

static void
updateMemoryFrontier(SmallVectorImpl<FrontierAccess> &frontier,
                     const VPTOSUnit *unit,
                     ArrayRef<ResolvedMemoryAccess> currentAccesses) {
  llvm::erase_if(frontier, [&](const FrontierAccess &prior) {
    return llvm::any_of(
        currentAccesses, [&](const ResolvedMemoryAccess &current) {
          return subsumesMemoryFrontierAccess(prior.access, current);
        });
  });
  for (const ResolvedMemoryAccess &current : currentAccesses) {
    frontier.push_back({unit, current});
  }
}

template <typename ConsumeWork, typename AddEdge>
static LogicalResult processMemoryAccess(
    const VPTOSchedDAG &dag, SmallVectorImpl<FrontierAccess> &frontier,
    VPTOSUnit &unit, ArrayRef<ResolvedMemoryAccess> currentAccesses,
    ConsumeWork &&consumeWork, AddEdge &&addEdge) {
  SmallPtrSet<const VPTOSUnit *, kVisitedOperationInlineCapacity>
      predecessors;
  if (failed(collectMemoryPredecessors(frontier, currentAccesses, consumeWork,
                                       predecessors))) {
    return mlir::failure();
  }
  for (const VPTOSUnit *predecessor : predecessors) {
    VPTOSUnit *mutablePredecessor =
        dag.lookup(predecessor->getOperation());
    if (!mutablePredecessor ||
        failed(addEdge(*mutablePredecessor, unit))) {
      return mlir::failure();
    }
  }

  // A write (or ordered/unknown access) replaces an earlier entry only when
  // its may-alias coverage contains that entry. Partial overlap establishes
  // an edge but cannot remove the old entry: a future access may touch only
  // the uncovered portion. Read-only accesses remain side by side until a
  // later closing access safely subsumes them.
  updateMemoryFrontier(frontier, &unit, currentAccesses);
  return success();
}

template <typename AddEdge>
static LogicalResult addBarrierPredecessors(const VPTOSchedDAG &dag,
                                            VPTOSUnit &barrier,
                                            AddEdge &&addEdge) {
  for (const std::unique_ptr<VPTOSUnit> &prior : dag.getUnits()) {
    bool reachedBarrier =
        prior->getOriginalIndex() >= barrier.getOriginalIndex();
    if (reachedBarrier) {
      break;
    }
    if (failed(addEdge(*prior, barrier, VPTOSchedEdgeKind::Sync, 0,
                       "before scheduling barrier"))) {
      return mlir::failure();
    }
  }
  return success();
}

template <typename AddEdge>
static LogicalResult
addImplicitReadEdge(const VPTOSchedulingEffect &effect, VPTOSUnit &unit,
                    llvm::StringMap<VPTOSUnit *> &lastWrite,
                    llvm::StringMap<SmallVector<VPTOSUnit *>> &readsSinceWrite,
                    AddEdge &&addEdge) {
  if (VPTOSUnit *writer = lastWrite.lookup(effect.resource)) {
    if (failed(addEdge(*writer, unit, VPTOSchedEdgeKind::Data, 1,
                       Twine("implicit read of ") + effect.resource))) {
      return mlir::failure();
    }
  }
  readsSinceWrite[effect.resource].push_back(&unit);
  return success();
}

template <typename AddEdge>
static LogicalResult addImplicitWriteEdges(
    const VPTOSchedulingEffect &effect, VPTOSUnit &unit,
    llvm::StringMap<VPTOSUnit *> &lastWrite,
    llvm::StringMap<SmallVector<VPTOSUnit *>> &readsSinceWrite,
    AddEdge &&addEdge) {
  if (VPTOSUnit *writer = lastWrite.lookup(effect.resource)) {
    if (failed(addEdge(*writer, unit, VPTOSchedEdgeKind::Output, 0,
                       Twine("implicit write of ") + effect.resource))) {
      return mlir::failure();
    }
  }
  for (VPTOSUnit *reader : readsSinceWrite[effect.resource]) {
    if (failed(
            addEdge(*reader, unit, VPTOSchedEdgeKind::Anti, 0,
                    Twine("implicit anti-dependence on ") + effect.resource))) {
      return mlir::failure();
    }
  }
  readsSinceWrite[effect.resource].clear();
  lastWrite[effect.resource] = &unit;
  return success();
}

template <typename ConsumeWork>
static FailureOr<bool>
liveInEscapesRegion(const VPTOSchedDAG &dag, Operation &lastRegionOperation,
                    Value value, ConsumeWork &&consumeWork) {
  for (Operation *user : value.getUsers()) {
    if (failed(consumeWork())) {
      return failure();
    }
    if (dag.lookup(user)) {
      continue;
    }
    bool isInAnotherBlock = user->getBlock() != dag.getRegion().block;
    bool isAfterRegion =
        !isInAnotherBlock && lastRegionOperation.isBeforeInBlock(user);
    if (isInAnotherBlock || isAfterRegion) {
      return true;
    }
  }
  return false;
}

template <typename ConsumeWork, typename AddEdge>
static LogicalResult
addSSAOperandEdges(VPTOSchedDAG &dag, VPTOSUnit &unit,
                   const VPTOSchedModel *model, Operation &lastRegionOperation,
                   DenseSet<Value> &checkedLiveIns, ConsumeWork &&consumeWork,
                   AddEdge &&addEdge) {
  for (auto [operandIndex, operand] :
       llvm::enumerate(unit.getOperation()->getOperands())) {
    if (failed(consumeWork())) {
      return mlir::failure();
    }
    Operation *definingOp = operand.getDefiningOp();
    VPTOSUnit *predecessor = definingOp ? dag.lookup(definingOp) : nullptr;
    if (!predecessor) {
      dag.addLiveIn(operand);
      if (!checkedLiveIns.insert(operand).second) {
        continue;
      }
      FailureOr<bool> escapes =
          liveInEscapesRegion(dag, lastRegionOperation, operand, consumeWork);
      if (failed(escapes)) {
        return mlir::failure();
      }
      if (*escapes) {
        dag.addLiveOut(operand);
      }
      continue;
    }
    unsigned latency =
        model ? model->getSchedParameters(predecessor->getOperation())
                    .writeLatency
              : 1;
    std::string reason =
        (isPostUpdateAddress(*predecessor, operand)
             ? Twine("post-update address operand #") + Twine(operandIndex)
             : Twine("ssa operand #") + Twine(operandIndex))
            .str();
    if (failed(addEdge(*predecessor, unit, latency, reason))) {
      return mlir::failure();
    }
  }
  return success();
}

template <typename ConsumeWork>
static LogicalResult addSSAResultLiveOuts(VPTOSchedDAG &dag, VPTOSUnit &unit,
                                          ConsumeWork &&consumeWork) {
  for (Value result : unit.getOperation()->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (failed(consumeWork())) {
        return mlir::failure();
      }
      if (!dag.lookup(user)) {
        dag.addLiveOut(result);
        break;
      }
    }
  }
  return success();
}

} // namespace

LogicalResult VPTOSchedDAGBuilder::buildMemoryEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  std::unique_ptr<PTOValueEvolutionAnalysis> valueEvolution;
  std::unique_ptr<PTOAddressAnalysis> addressAnalysis;
  if (!dag.getUnits().empty()) {
    func::FuncOp function =
        dag.getUnits().front()->getOperation()->getParentOfType<func::FuncOp>();
    if (function) {
      valueEvolution =
          std::make_unique<PTOValueEvolutionAnalysis>(function);
      addressAnalysis =
          std::make_unique<PTOAddressAnalysis>(function, *valueEvolution);
    }
  }

  auto consume = [this, &failure](uint64_t amount) {
    return consumeWork(failure, amount);
  };
  auto resolved =
      collectResolvedMemoryAccesses(dag, addressAnalysis.get(), consume);
  if (failed(resolved)) {
    return mlir::failure();
  }

  SmallVector<FrontierAccess> frontier;
  auto addMemoryEdge = [this, &dag, &failure](VPTOSUnit &predecessor,
                                               VPTOSUnit &successor) {
    return addEdge(dag, predecessor, successor, VPTOSchedEdgeKind::Memory,
                   VPTOSchedEdgeStrength::Must, /*latency=*/0,
                   "may-alias memory frontier in original order", failure);
  };
  for (auto indexedAccesses : llvm::enumerate(*resolved)) {
    ArrayRef<ResolvedMemoryAccess> currentAccesses = indexedAccesses.value();
    if (currentAccesses.empty()) {
      continue;
    }
    auto consumeOne = [this, &failure]() { return consumeWork(failure); };
    VPTOSUnit *unit = dag.getUnits()[indexedAccesses.index()].get();
    if (failed(processMemoryAccess(dag, frontier, *unit, currentAccesses,
                                   consumeOne, addMemoryEdge))) {
      return mlir::failure();
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildImplicitAndSyncEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  llvm::StringMap<VPTOSUnit *> lastWrite;
  llvm::StringMap<SmallVector<VPTOSUnit *>> readsSinceWrite;
  VPTOSUnit *lastBarrier = nullptr;

  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    auto add = [this, &dag, &failure](VPTOSUnit &predecessor,
                                      VPTOSUnit &successor,
                                      VPTOSchedEdgeKind kind, unsigned latency,
                                      Twine reason) {
      return addEdge(dag, predecessor, successor, kind,
                     VPTOSchedEdgeStrength::Must, latency, reason, failure);
    };
    if (lastBarrier && lastBarrier != &unit &&
        failed(add(*lastBarrier, unit, VPTOSchedEdgeKind::Sync, 0,
                   "after scheduling barrier"))) {
      return mlir::failure();
    }

    for (const VPTOSchedulingEffect &effect : unit.getSemantics().effects) {
      if (failed(consumeWork(failure))) {
        return mlir::failure();
      }
      if (effect.kind == VPTOSchedulingEffectKind::Barrier) {
        if (failed(addBarrierPredecessors(dag, unit, add))) {
          return mlir::failure();
        }
        lastBarrier = &unit;
        continue;
      }
      if (effect.resource.empty()) {
        continue;
      }
      if (effect.kind == VPTOSchedulingEffectKind::ImplicitRead) {
        if (failed(addImplicitReadEdge(effect, unit, lastWrite, readsSinceWrite,
                                       add))) {
          return mlir::failure();
        }
        continue;
      }
      if (effect.kind != VPTOSchedulingEffectKind::ImplicitWrite) {
        continue;
      }
      if (failed(addImplicitWriteEdges(effect, unit, lastWrite, readsSinceWrite,
                                       add))) {
        return mlir::failure();
      }
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildRematerializationClusterEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  for (size_t index = 1; index < units.size(); ++index) {
    VPTOSUnit &current = *units[index];
    if (!rematerializationAnchors ||
        !rematerializationAnchors->contains(current.getOperation())) {
      continue;
    }
    if (failed(addEdge(dag, *units[index - 1], current,
                       VPTOSchedEdgeKind::Cluster,
                       VPTOSchedEdgeStrength::Must, /*latency=*/0,
                       "keep rematerialized chain near its use", failure))) {
      return mlir::failure();
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildSSAEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  for (Value value : dag.getRegion().liveThroughs) {
    if (failed(consumeWork(failure))) {
      return mlir::failure();
    }
    dag.addLiveIn(value);
    dag.addLiveOut(value);
  }

  DenseSet<Value> checkedLiveIns;
  Operation *lastRegionOperation = dag.getRegion().operations.back();
  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    auto consume = [this, &failure]() { return consumeWork(failure); };
    auto add = [this, &dag, &failure](VPTOSUnit &predecessor,
                                      VPTOSUnit &successor, unsigned latency,
                                      Twine reason) {
      return addEdge(dag, predecessor, successor, VPTOSchedEdgeKind::Data,
                     VPTOSchedEdgeStrength::Must, latency, reason, failure);
    };
    if (failed(addSSAOperandEdges(dag, unit, model, *lastRegionOperation,
                                  checkedLiveIns, consume, add)) ||
        failed(addSSAResultLiveOuts(dag, unit, consume))) {
      return mlir::failure();
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildModelFallbackEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  if (!model)
    return success();
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  for (size_t index = 0; index < units.size(); ++index) {
    if (failed(consumeWork(failure))) {
      return mlir::failure();
    }
    VPTOSUnit &unit = *units[index];
    if (model->getSchedClass(unit.getOperation()).known)
      continue;
    if (index != 0 &&
        failed(addEdge(dag, *units[index - 1], unit,
                       VPTOSchedEdgeKind::Artificial,
                       VPTOSchedEdgeStrength::Must, 0,
                       "unknown sched class preserves predecessor order",
                       failure))) {
      return mlir::failure();
    }
    if (index + 1 != units.size()) {
      if (failed(addEdge(dag, unit, *units[index + 1],
                         VPTOSchedEdgeKind::Artificial,
                         VPTOSchedEdgeStrength::Must, 0,
                         "unknown sched class preserves successor order",
                         failure))) {
        return mlir::failure();
      }
    }
  }
  return success();
}
