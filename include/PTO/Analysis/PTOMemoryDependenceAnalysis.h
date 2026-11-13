// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_ANALYSIS_PTOMEMORYDEPENDENCEANALYSIS_H
#define PTO_ANALYSIS_PTOMEMORYDEPENDENCEANALYSIS_H
#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::pto {
enum class PTOMemoryDependenceStatus { NoDependence, MayDependence, Unknown };
enum class PTOMemoryDependenceReason {
    InvalidContext, UnknownRoot,
    DynamicUnmodelledExpression,
    DynamicLoopBounds,
    UnsupportedAccessShape
};

struct PTOMemoryAccessOccurrence {
    Operation* op = nullptr;
    PTOMemoryFootprint footprint;
};

struct PTOMemoryCarriedDependence {
    scf::ForOp loop;
    PTOMemoryDependenceStatus status;
    PTOMemoryDependenceReason reason;
};

SmallVector<unsigned, 2> getCommonLoopPrefix(ArrayRef<unsigned> lhs, ArrayRef<unsigned> rhs);

// Scope must be a strict common ancestor of both accesses and must not be a
// loop: all loops below this container participate in the relation. Invalid
// context returns Unknown (a null-loop Unknown entry for carried queries).
// Empty conservative relations prove disjointness. Nonempty relations are
// possible dependence, never a claim that an actual execution must overlap.
PTOMemoryDependenceStatus querySequentialMemoryOverlap(
    const PTOMemoryAccessOccurrence& producer, const PTOMemoryAccessOccurrence& consumer, Operation* scope,
    bool useEnvelope = false);

SmallVector<PTOMemoryCarriedDependence> queryCarriedMemoryOverlap(
    const PTOMemoryAccessOccurrence& producer, const PTOMemoryAccessOccurrence& consumer, Operation* scope);
} // namespace mlir::pto
#endif
