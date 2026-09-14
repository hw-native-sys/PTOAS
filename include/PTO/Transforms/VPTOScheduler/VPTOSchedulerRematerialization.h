// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedulerRematerialization.h - Bounded VPTO remat ---*- C++ -*-===//
//
// This file defines the transactional rematerialization step used between two
// scheduler runs. Target-approved cheap producer DAGs are cloned within fixed
// depth and cost budgets. The transaction keeps original definitions until the
// second schedule lowers vector pressure without increasing another pressure
// set, so callers can restore the first schedule without rebuilding IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULERREMATERIALIZATION_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULERREMATERIALIZATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace llvm {
class raw_ostream;
}

namespace mlir {
class OpOperand;
class Operation;
} // namespace mlir

namespace mlir::pto {

class VPTOSchedModel;

struct VPTORematerializationStats {
    bool changed = false;
    int64_t initialMaxVectorPressure = 0;
    SmallVector<int64_t> initialMaxPressure;
    unsigned highPressureRegions = 0;
    unsigned selectedCandidates = 0;
    unsigned cloneGroups = 0;
    unsigned cloneOperations = 0;
    uint64_t estimatedDynamicMicroOps = 0;
};

class VPTORematerializationTransaction {
public:
    VPTORematerializationTransaction() = default;
    VPTORematerializationTransaction(const VPTORematerializationTransaction&) = delete;
    VPTORematerializationTransaction& operator=(const VPTORematerializationTransaction&) = delete;
    VPTORematerializationTransaction(VPTORematerializationTransaction&&) = default;
    VPTORematerializationTransaction& operator=(VPTORematerializationTransaction&&) = default;

    const VPTORematerializationStats& getStats() const { return stats; }
    bool changed() const { return stats.changed; }
    const llvm::DenseSet<Operation*>& getAnchors() const { return anchors; }

    void commit();
    void rollback();

private:
    friend VPTORematerializationTransaction prepareVPTORematerialization(
        func::FuncOp, const VPTOSchedModel&, llvm::raw_ostream&, bool);

    struct Builder;

    struct ReplacedUse {
        OpOperand* use = nullptr;
        Value original;
    };

    VPTORematerializationStats stats;
    SmallVector<ReplacedUse> replacedUses;
    SmallVector<Operation*> clones;
    llvm::DenseSet<Operation*> anchors;
    SmallVector<Operation*> originalOperations;
};

VPTORematerializationTransaction prepareVPTORematerialization(
    func::FuncOp func, const VPTOSchedModel& model, llvm::raw_ostream& os, bool trace);

std::optional<SmallVector<int64_t>> evaluateVPTOMaxPressure(func::FuncOp func, const VPTOSchedModel& model);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDULERREMATERIALIZATION_H
