// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARANALYSIS_H
#define PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARANALYSIS_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarIR.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemoryFootprint.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::pto::vecscopemembar {

enum class VecScopeHazardScope {
  SameIteration,
  InnerLoopCarried,
  OuterLoopCarried,
};

enum class DependenceStatus {
  NoDependence,
  ProvenDependence,
  Unknown,
};

enum class DependenceReason {
  ExactIntervalOverlap,
  PresburgerRelationNonEmpty,
  UnknownRoot,
  DynamicUnmodelledExpression,
  UnsupportedAccessShape,
};

struct IterationDistance {
  std::optional<int64_t> outer;
  std::optional<int64_t> inner;
  bool positive = false;
  bool exact = false;
};

struct MemoryHazard {
  unsigned id = 0;
  unsigned producer = 0;
  unsigned consumer = 0;
  MemBarKind kind = MemBarKind::VV_ALL;
  VecScopeHazardScope scope = VecScopeHazardScope::SameIteration;
  Operation *sameIterationAnchor = nullptr;
  std::optional<unsigned> carryingLoop;
  IterationDistance distance;
};

struct UnknownDependence {
  unsigned producer = 0;
  unsigned consumer = 0;
  MemBarKind kind = MemBarKind::VV_ALL;
  DependenceReason reason = DependenceReason::UnsupportedAccessShape;
};

struct ExistingBarrier {
  Operation *op = nullptr;
  MemBarKind kind = MemBarKind::VV_ALL;
  SmallVector<unsigned, 2> schedulePath;
  SmallVector<unsigned, 2> enclosingLoops;
  enum Phase { SameIteration, LoopLatch } phase = SameIteration;
  std::optional<unsigned> latchLoop;
};

struct VecScopeMemBarAnalysisResult {
  Operation *scope = nullptr;
  SmallVector<VecScopeScheduleNode> schedule;
  SmallVector<VecScopeLoopInfo> loops;
  SmallVector<AccessOccurrence> accesses;
  SmallVector<MemoryHazard> hazards;
  SmallVector<UnknownDependence> unknownDependences;
  SmallVector<ExistingBarrier> existingBarriers;
};

FailureOr<VecScopeMemBarAnalysisResult>
runVecScopeMemBarAnalysis(Operation *scope);

FailureOr<VecScopeMemBarAnalysisResult>
runCrossVecScopeMemBarAnalysis(Operation *scope);

Operation *findSameIterationAnchor(Operation *producer, Operation *consumer,
                                   Operation *scope);

} // namespace mlir::pto::vecscopemembar

#endif // PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARANALYSIS_H
