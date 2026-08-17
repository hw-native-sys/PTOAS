// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAG.h - VPTO scheduling DAG --------------------*- C++ -*-===//
//
// The DAG owns scheduling units and typed dependency edges.  Edges retain
// their reason for analyze-mode diagnostics.  Must edges control readiness;
// Weak edges are preferences only and must never be required for correctness.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAG_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAG_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedRegion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <string>

namespace mlir::pto {

enum class VPTOSchedEdgeKind {
  Data,
  Anti,
  Output,
  Memory,
  Control,
  Sync,
  Artificial,
  Cluster,
};

enum class VPTOSchedEdgeStrength { Must, Weak };

class VPTOSUnit;

class VPTOSchedEdge {
public:
  VPTOSchedEdge(VPTOSUnit *predecessor, VPTOSUnit *successor,
                VPTOSchedEdgeKind kind, VPTOSchedEdgeStrength strength,
                unsigned latency, std::string reason)
      : predecessor(predecessor), successor(successor), kind(kind),
        strength(strength), latency(latency), reason(std::move(reason)) {}

  VPTOSUnit *getPredecessor() const { return predecessor; }
  VPTOSUnit *getSuccessor() const { return successor; }
  VPTOSchedEdgeKind getKind() const { return kind; }
  VPTOSchedEdgeStrength getStrength() const { return strength; }
  unsigned getLatency() const { return latency; }
  StringRef getReason() const { return reason; }
  bool isMust() const { return strength == VPTOSchedEdgeStrength::Must; }

private:
  VPTOSUnit *predecessor;
  VPTOSUnit *successor;
  VPTOSchedEdgeKind kind;
  VPTOSchedEdgeStrength strength;
  unsigned latency;
  std::string reason;
};

class VPTOSUnit {
public:
  VPTOSUnit(unsigned id, unsigned originalIndex, Operation *op)
      : id(id), originalIndex(originalIndex), op(op),
        semantics(getVPTOSchedulingSemantics(op)) {}

  unsigned getId() const { return id; }
  unsigned getOriginalIndex() const { return originalIndex; }
  Operation *getOperation() const { return op; }
  VPTOSchedulingClass getSchedulingClass() const {
    return semantics.schedulingClass;
  }
  const VPTOSchedulingSemantics &getSemantics() const { return semantics; }
  ArrayRef<VPTOSchedEdge *> getPredecessors() const { return predecessors; }
  ArrayRef<VPTOSchedEdge *> getSuccessors() const { return successors; }

  unsigned getRemainingPredecessors() const { return remainingPredecessors; }
  unsigned getRemainingSuccessors() const { return remainingSuccessors; }
  void setRemainingPredecessors(unsigned value) {
    remainingPredecessors = value;
  }
  void setRemainingSuccessors(unsigned value) { remainingSuccessors = value; }

  unsigned getDepth() const { return depth; }
  unsigned getHeight() const { return height; }
  void setDepth(unsigned value) { depth = value; }
  void setHeight(unsigned value) { height = value; }

private:
  friend class VPTOSchedDAG;

  unsigned id;
  unsigned originalIndex;
  Operation *op;
  VPTOSchedulingSemantics semantics;
  SmallVector<VPTOSchedEdge *> predecessors;
  SmallVector<VPTOSchedEdge *> successors;
  unsigned remainingPredecessors = 0;
  unsigned remainingSuccessors = 0;
  unsigned depth = 0;
  unsigned height = 0;
};

class VPTOSchedDAG {
public:
  explicit VPTOSchedDAG(const VPTOSchedRegion &region);

  const VPTOSchedRegion &getRegion() const { return region; }
  ArrayRef<std::unique_ptr<VPTOSUnit>> getUnits() const { return units; }
  ArrayRef<std::unique_ptr<VPTOSchedEdge>> getEdges() const { return edges; }
  ArrayRef<Value> getLiveIns() const { return liveIns; }
  ArrayRef<Value> getLiveOuts() const { return liveOuts; }
  VPTOSUnit *lookup(Operation *op) const;

  VPTOSchedEdge &addEdge(VPTOSUnit &predecessor, VPTOSUnit &successor,
                         VPTOSchedEdgeKind kind,
                         VPTOSchedEdgeStrength strength, unsigned latency,
                         Twine reason);
  void resetDependencyCounts();
  LogicalResult computeCriticalPaths();
  void addLiveIn(Value value);
  void addLiveOut(Value value);

private:
  VPTOSchedRegion region;
  SmallVector<std::unique_ptr<VPTOSUnit>> units;
  SmallVector<std::unique_ptr<VPTOSchedEdge>> edges;
  DenseMap<Operation *, VPTOSUnit *> unitByOperation;
  SmallVector<Value> liveIns;
  SmallVector<Value> liveOuts;
};

StringRef stringifyVPTOSchedEdgeKind(VPTOSchedEdgeKind kind);
StringRef stringifyVPTOSchedEdgeStrength(VPTOSchedEdgeStrength strength);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAG_H
