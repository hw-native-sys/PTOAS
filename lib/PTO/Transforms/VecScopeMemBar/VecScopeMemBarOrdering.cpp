// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarOrdering.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemoryFootprint.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;
namespace {
// Require a straight-line SSA relay. Region results and iter_args do not
// imply a must-dependence and cannot discharge a carried ordering obligation.
static bool dependsOn(Value value, Value target, llvm::DenseSet<Value>& visited, unsigned depth)
{
    if (value == target) {
        return true;
    }
    Operation* op = value.getDefiningOp();
    if (
        !op || depth == kValue32 || !visited.insert(value).second || op->getNumRegions() != 0 ||
        !isMemoryEffectFree(op)) {
        return false;
    }
    return llvm::any_of(op->getOperands(), [&](Value operand) {
        return dependsOn(operand, target, visited, depth + 1);
    });
}

static bool payloadDependsOnLoad(Operation* store, Operation* load)
{
    auto payloads = getStoredValues(store);
    if (auto shortStore = dyn_cast<VstusOp>(store)) {
        payloads.push_back(shortStore.getValue());
    }
    for (Value payload : payloads) {
        for (Value value : getLoadedValues(load)) {
            llvm::DenseSet<Value> visited;
            if (dependsOn(payload, value, visited, 0)) {
                return true;
            }
        }
    }
    return false;
}

// A carried WAR can already be ordered by a mandatory store-to-load cut:
// producer load -> dependent store -> VST_VLD -> next iteration load ->
// consumer store. Both SSA edges are required: a directed cut alone does
// not order independent stores after it. X(j) may be any same-iteration
// load feeding S(j): its next execution is after C(i) across the backedge.
// D(i) may be an intermediate store, not necessarily S(i). Keep physical
// load footprints.
static bool carriedWAROrdered(
    const VecScopeMemBarAnalysisResult& result, unsigned producer, unsigned consumer, scf::ForOp loop,
    ArrayRef<MemoryOrderingCut> cuts)
{
    const auto& accesses = result.accesses;
    const auto& load = accesses[producer];
    const auto& store = accesses[consumer];
    if (load.kind != PTOMemoryAccessKind::Load || store.kind != PTOMemoryAccessKind::Store ||
        load.unit != PTOMemoryAccessUnit::Vector || store.unit != PTOMemoryAccessUnit::Vector ||
        load.op->getBlock() != loop.getBody()) {
        return false;
    }
    Operation* payloadStore = store.op;
    for (const auto& stream : accesses) {
        if (stream.orderedStreamFlush == store.op && isa<VstusOp>(stream.op)) {
            payloadStore = stream.op;
        }
    }
    bool consumesLoad = llvm::any_of(accesses, [&](const auto& source) {
        return source.kind == PTOMemoryAccessKind::Load && source.unit == PTOMemoryAccessUnit::Vector &&
               source.op->getBlock() == payloadStore->getBlock() && source.op->isBeforeInBlock(payloadStore) &&
               payloadDependsOnLoad(payloadStore, source.op);
    });
    if (!consumesLoad) {
        return false;
    }
    auto ordersProducer = [&](Operation* anchor) {
        if (
            !anchor || anchor->getBlock() != loop.getBody() || !load.op->isBeforeInBlock(anchor)) {
            return false;
        }
        return llvm::any_of(accesses, [&](const auto& dependentStore) {
            return dependentStore.kind == PTOMemoryAccessKind::Store &&
                   dependentStore.unit == PTOMemoryAccessUnit::Vector &&
                   dependentStore.op->getBlock() == loop.getBody() &&
                   dependentStore.op->isBeforeInBlock(anchor) &&
                   payloadDependsOnLoad(dependentStore.op, load.op);
        });
    };
    return llvm::any_of(cuts, [&](const auto& barrier) {
        bool cut = barrier.kind == MemBarKind::VST_VLD || barrier.kind == MemBarKind::VV_ALL;
        return cut && ordersProducer(barrier.anchor);
    });
}

// A mandatory VST_VLD after a store drains it before every later vector
// load, including loads in subsequent executions of the carrying loop.
static bool carriedRAWOrdered(
    const VecScopeMemBarAnalysisResult& result, unsigned producer, unsigned consumer, scf::ForOp loop,
    ArrayRef<MemoryOrderingCut> cuts)
{
    const auto& store = result.accesses[producer];
    const auto& load = result.accesses[consumer];
    if (store.kind != PTOMemoryAccessKind::Store || load.kind != PTOMemoryAccessKind::Load ||
        store.unit != PTOMemoryAccessUnit::Vector || load.unit != PTOMemoryAccessUnit::Vector ||
        store.op->getBlock() != loop.getBody()) {
        return false;
    }
    auto followsStore = [&](Operation* anchor) {
        return anchor && anchor->getBlock() == loop.getBody() && store.op->isBeforeInBlock(anchor);
    };
    return llvm::any_of(cuts, [&](const auto& barrier) {
        bool cut = barrier.kind == MemBarKind::VST_VLD || barrier.kind == MemBarKind::VV_ALL;
        return cut && followsStore(barrier.anchor);
    });
}

} // namespace

bool vecscopemembar::carriedMemoryAlreadyOrdered(
    const VecScopeMemBarAnalysisResult& result, unsigned producer, unsigned consumer, scf::ForOp loop,
    ArrayRef<MemoryOrderingCut> cuts)
{
    return carriedWAROrdered(result, producer, consumer, loop, cuts) ||
           carriedRAWOrdered(result, producer, consumer, loop, cuts);
}
