// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Propagate single-C++-type constraints through TADD without mutating IR.
#include "PTO/Analysis/Linx/LinxTileRoleAnalysis.h"
#include "PTO/IR/Linx/LinxDialect.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
using namespace mlir::pto::linx;

LogicalResult LinxTileRoleAnalysis::require(Value value, TileRole role, Operation* origin)
{
    SmallVector<Value> pending{value};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
        Value current = pending.pop_back_val();
        if (!visited.insert(current).second) {
            continue;
        }
        auto found = requirements.find(current);
        if (found != requirements.end() && found->second.role != role) {
            auto diagnostic = origin->emitError("conflicting Linx C++ tile roles for the same allocation");
            diagnostic.attachNote(found->second.origin->getLoc()) << "previous role required here";
            return failure();
        }
        requirements.try_emplace(current, Requirement{role, origin});
        for (Operation* user : current.getUsers()) {
            if (auto add = dyn_cast<TAddOp>(user)) {
                pending.append({add.getLhs(), add.getRhs(), add.getDst()});
            }
            if (auto exp = dyn_cast<TExpOp>(user)) {
                pending.append({exp.getSrc(), exp.getDst()});
            }
        }
    }
    return success();
}

LogicalResult LinxTileRoleAnalysis::run(ModuleOp module)
{
    requirements.clear();
    WalkResult result = module.walk([&](TMatmulOp matmul) {
        if (failed(require(matmul.getLhs(), TileRole::Left, matmul)) ||
            failed(require(matmul.getRhs(), TileRole::Right, matmul))) {
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

TileRole LinxTileRoleAnalysis::getRole(Value value) const
{
    auto found = requirements.find(value);
    return found == requirements.end() ? TileRole::Vec : found->second.role;
}
