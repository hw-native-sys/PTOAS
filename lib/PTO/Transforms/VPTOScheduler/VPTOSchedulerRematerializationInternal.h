// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerRematerializationInternal.h - Remat internals -*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULERREMATERIALIZATIONINTERNAL_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULERREMATERIALIZATIONINTERNAL_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedRegion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace llvm {
class raw_ostream;
}

namespace mlir::pto {
class VPTOSchedModel;
}

namespace mlir::pto::remat {

constexpr unsigned kMaxCandidates = 16;
constexpr unsigned kMaxPredecessorDepth = 2;
constexpr unsigned kMaxGroupsPerCandidate = 4;
constexpr unsigned kMaxUsesPerGroup = 2;
constexpr unsigned kMaxUseGap = 64;
constexpr unsigned kMaxCloneOperations = 96;
constexpr uint64_t kMaxDynamicMicroOps = 2048;
constexpr int64_t kPressureHeadroom = 1;

struct PressureRegion {
    VPTOSchedRegion region;
    unsigned blockIndex = 0;
    bool needsRelief = false;
    int64_t peak = 0;
    int64_t limit = 0;
    int64_t target = 0;
    SmallVector<Value> liveIns;
    DenseMap<Operation*, unsigned> operationIndices;
};

struct TargetPosition {
    unsigned regionIndex = 0;
    unsigned operationIndex = 0;
};

struct UseGroup {
    unsigned regionIndex = 0;
    unsigned firstIndex = 0;
    unsigned lastIndex = 0;
    Operation* insertionPoint = nullptr;
    SmallVector<OpOperand*> uses;
    uint64_t dynamicMultiplier = 0;
};

struct RematCandidate {
    unsigned diagnosticId = 0;
    Value value;
    SmallVector<Operation*> recipeOperations;
    SmallVector<UseGroup> groups;
    SmallVector<unsigned> affectedRegions;
    uint64_t coverageBenefit = 0;
    uint64_t dynamicMicroOps = 0;
    unsigned cloneOperations = 0;
};

SmallVector<PressureRegion, 0> collectPressureRegions(
    func::FuncOp func, const VPTOSchedModel& model, llvm::raw_ostream& os, bool trace,
    SmallVectorImpl<int64_t>& initialMaxPressure);

SmallVector<RematCandidate, 0> collectCandidates(
    ArrayRef<PressureRegion> regions, const VPTOSchedModel& model, func::FuncOp func, llvm::raw_ostream& os,
    bool trace);

SmallVector<unsigned> selectCandidates(
    ArrayRef<RematCandidate> candidates, ArrayRef<PressureRegion> regions, llvm::raw_ostream& os, bool trace,
    unsigned& cloneOperations, uint64_t& dynamicMicroOps);

std::optional<SmallVector<int64_t>> evaluateMaxPressure(func::FuncOp func, const VPTOSchedModel& model);

} // namespace mlir::pto::remat

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULERREMATERIALIZATIONINTERNAL_H
