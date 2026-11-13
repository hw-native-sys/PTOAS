// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Prove redundancy using actual surviving barrier cuts, including their ordering
// of non-aliasing accesses, and straight-line SSA dependencies. All original
// planned hazard obligations are rechecked after every proposed removal so one
// removed barrier cannot be used to justify another. Existing IR barriers and
// loop-latch cuts are preserved. Nested accesses may be endpoints, but cannot
// relay ordering through a possibly empty third loop or a conditional region.

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarPlacement.h"
#include "PTO/Analysis/PTOIntegerRange.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

using OrderGraph = SmallVector<SmallVector<unsigned, 4>, 16>;

static Operation* ancestorInBlock(Operation* op, Block* block)
{
    while (op && op->getBlock() != block) {
        op = op->getParentOp();
        if (!isa_and_nonnull<scf::ForOp>(op)) {
            return nullptr;
        }
    }
    return op;
}

// Four families (VV, VS, SV, SS), each with ALL, ST_LD, LD_ST, ST_ST.
// Consult the access descriptors, not generic memory-effect predicates, which
// alone do not distinguish scalar from vector accesses.
static bool ordersAccessPair(MemBarKind kind, const AccessOccurrence& producer, const AccessOccurrence& consumer)
{
    unsigned family = static_cast<unsigned>(kind) / 4;
    unsigned direction = static_cast<unsigned>(kind) % 4;
    bool producerScalar = family >= 2;
    bool consumerScalar = family == 1 || family == 3;
    bool unitMismatch = (producer.unit == PTOMemoryAccessUnit::Scalar) != producerScalar ||
        (consumer.unit == PTOMemoryAccessUnit::Scalar) != consumerScalar;
    if (unitMismatch) {
        return false;
    }
    bool storeBefore = producer.kind == PTOMemoryAccessKind::Store;
    bool storeAfter = consumer.kind == PTOMemoryAccessKind::Store;
    switch (direction) {
        case 0:
            return true;
        case 1:
            return storeBefore && !storeAfter;
        case 2:
            return !storeBefore && storeAfter;
        case 3:
            return storeBefore && storeAfter;
        default:
            return false;
    }
}

static void addBarrierEdges(OrderGraph& edges, ArrayRef<AccessOccurrence> accesses, Operation* anchor, MemBarKind kind)
{
    Block* block = anchor->getBlock();
    for (unsigned from = 0; from < accesses.size(); ++from) {
        Operation* producer = ancestorInBlock(accesses[from].op, block);
        if (!producer || producer == anchor || !producer->isBeforeInBlock(anchor)) {
            continue;
        }
        for (unsigned to = 0; to < accesses.size(); ++to) {
            Operation* consumer = ancestorInBlock(accesses[to].op, block);
            if (consumer && (consumer == anchor || anchor->isBeforeInBlock(consumer)) &&
                ordersAccessPair(kind, accesses[from], accesses[to])) {
                if (!llvm::is_contained(edges[from], to)) {
                    edges[from].push_back(to);
                }
            }
        }
    }
}

// Unlike a general use-def walk, do not traverse region results/iter_args: a
// load in a branch or a zero-trip loop need not execute to produce that result.
static bool straightLineDependsOn(Value sink, Value source, Block* block)
{
    SmallVector<Value, 8> worklist{sink};
    DenseSet<Value> visited;
    while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (value == source) {
            return true;
        }
        if (!visited.insert(value).second) {
            continue;
        }
        Operation* op = value.getDefiningOp();
        if (op && op->getBlock() == block && op->getNumRegions() == 0 && isMemoryEffectFree(op)) {
            llvm::append_range(worklist, op->getOperands());
        }
    }
    return false;
}

// Lift a load through a reduction only if every iteration contributes to
// the yielded result and each later iteration also consumes that result.
// Merely reaching an init operand is insufficient: the body may overwrite it.
static bool hasValueRelay(Operation* load, Operation* store, Block* block)
{
    SmallVector<Value, 4> values = getLoadedValues(load);
    Block* current = load->getBlock();
    while (current != block) {
        auto loop = dyn_cast_or_null<scf::ForOp>(current->getParentOp());
        if (!loop) {
            return false;
        }
        auto bounds = mlir::pto::integer_range::getConstBounds(loop);
        auto envelope = mlir::pto::integer_range::getLoopEnvelope(loop);
        auto trip = bounds.exactTripCount();
        if (!trip || *trip <= 0 || !envelope.valid) {
            return false;
        }
        SmallVector<Value, 4> lifted;
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        for (unsigned index = 0; index < loop.getNumResults(); ++index) {
            Value next = yield.getOperand(index);
            Value carry = loop.getRegionIterArgs()[index];
            if (!straightLineDependsOn(next, carry, current)) {
                continue;
            }
            for (Value value : values) {
                if (straightLineDependsOn(next, value, current)) {
                    lifted.push_back(loop.getResult(index));
                    break;
                }
            }
        }
        if (lifted.empty()) {
            return false;
        }
        values = std::move(lifted);
        current = loop->getBlock();
    }
    for (Value loaded : values) {
        for (Value stored : getStoredValues(store)) {
            if (straightLineDependsOn(stored, loaded, block)) {
                return true;
            }
        }
    }
    return false;
}

// A relay inside another loop is usable only when that entire enclosing
// path executes. Conditions and possibly empty loops never supply witnesses.
static bool guaranteedRelayInBlock(Operation* op, Block* block)
{
    bool nested = op->getBlock() != block;
    while (nested) {
        auto loop = dyn_cast_or_null<scf::ForOp>(op->getParentOp());
        if (!loop) {
            return false;
        }
        auto bounds = mlir::pto::integer_range::getConstBounds(loop);
        auto envelope = mlir::pto::integer_range::getLoopEnvelope(loop);
        auto trip = bounds.exactTripCount();
        if (!trip || *trip <= 0 || !envelope.valid) {
            return false;
        }
        op = loop;
        nested = op->getBlock() != block;
    }
    return true;
}

static OrderGraph buildValueEdges(ArrayRef<AccessOccurrence> accesses, Block* cutBlock, bool reductions)
{
    OrderGraph edges(accesses.size());
    for (unsigned from = 0; from < accesses.size(); ++from) {
        const auto& load = accesses[from];
        if (load.kind != PTOMemoryAccessKind::Load) {
            continue;
        }
        for (unsigned to = from + 1; to < accesses.size(); ++to) {
            const auto& store = accesses[to];
            if (store.kind == PTOMemoryAccessKind::Store &&
                (load.op->getBlock() == store.op->getBlock() ||
                 (reductions && guaranteedRelayInBlock(store.op, cutBlock))) &&
                hasValueRelay(load.op, store.op, store.op->getBlock())) {
                edges[from].push_back(to);
            }
        }
    }
    return edges;
}

static bool reaches(
    const OrderGraph& edges, ArrayRef<AccessOccurrence> accesses, Block* block, unsigned from, unsigned to,
    bool reductions)
{
    SmallVector<bool, 16> visited(edges.size(), false);
    SmallVector<unsigned, 16> worklist{from};
    visited[from] = true;
    while (!worklist.empty()) {
        unsigned current = worklist.pop_back_val();
        if (current == to) {
            return true;
        }
        Block* currentBlock = accesses[current].op->getBlock();
        bool endpointBlock = currentBlock == accesses[from].op->getBlock() ||
                             currentBlock == accesses[to].op->getBlock();
        if (current != from && currentBlock != block &&
            (reductions ? !guaranteedRelayInBlock(accesses[current].op, block) : !endpointBlock)) {
            continue;
        }
        for (unsigned next : edges[current]) {
            if (!visited[next]) {
                visited[next] = true;
                worklist.push_back(next);
            }
        }
    }
    return false;
}

static bool isOrdinaryCutInBlock(const BarrierPlacement& barrier, Block* block)
{
    return barrier.anchorKind == BarrierAnchorKind::BeforeOperation && barrier.anchor &&
           barrier.anchor->getBlock() == block;
}

static OrderGraph buildSurvivingEdges(
    const VecScopeMemBarAnalysisResult& result, const VecScopeMemBarPlan& plan, ArrayRef<bool> live, unsigned excluded,
    bool reductions = false)
{
    Block* block = plan.barriers[excluded].anchor->getBlock();
    OrderGraph edges = buildValueEdges(result.accesses, block, reductions);
    for (const auto& barrier : result.existingBarriers) {
        if (barrier.op->getBlock() == block) {
            addBarrierEdges(edges, result.accesses, barrier.op, barrier.kind);
        }
    }
    for (unsigned index = 0; index < plan.barriers.size(); ++index) {
        const auto& barrier = plan.barriers[index];
        if (index != excluded && live[index] && isOrdinaryCutInBlock(barrier, block)) {
            addBarrierEdges(edges, result.accesses, barrier.anchor, barrier.kind);
        }
    }
    return edges;
}

static bool preservesHazards(
    const VecScopeMemBarAnalysisResult& result, const OrderGraph& edges, Block* block, bool reductions = false)
{
    for (const auto& hazard : result.hazards) {
        if (hazard.scope != VecScopeHazardScope::SameIteration || hazard.producer >= result.accesses.size() ||
            hazard.consumer >= result.accesses.size()) {
            continue;
        }
        Operation* producer = ancestorInBlock(result.accesses[hazard.producer].op, block);
        Operation* consumer = ancestorInBlock(result.accesses[hazard.consumer].op, block);
        if (!producer || !consumer || producer == consumer) {
            continue;
        }
        if (!reaches(edges, result.accesses, block, hazard.producer, hazard.consumer, reductions)) {
            return false;
        }
    }
    return true;
}

static bool canModelCut(const BarrierPlacement& candidate, const VecScopeMemBarAnalysisResult& result)
{
    if (!candidate.anchor || !isOrdinaryCutInBlock(candidate, candidate.anchor->getBlock()) ||
        candidate.resolvedHazards.empty()) {
        return false;
    }
    Block* block = candidate.anchor->getBlock();
    for (unsigned id : candidate.resolvedHazards) {
        if (id >= result.hazards.size()) {
            return false;
        }
        const auto& hazard = result.hazards[id];
        if (hazard.scope != VecScopeHazardScope::SameIteration ||
            !ancestorInBlock(result.accesses[hazard.producer].op, block) ||
            !ancestorInBlock(result.accesses[hazard.consumer].op, block)) {
            return false;
        }
    }
    return true;
}

} // namespace

void vecscopemembar::eliminateTransitiveMemBars(const VecScopeMemBarAnalysisResult& result, VecScopeMemBarPlan& plan)
{
    SmallVector<bool, 8> live(plan.barriers.size(), true);
    // Each successful removal preserves every obligation, including those of
    // earlier removed candidates. One sweep is sufficient: removing more cuts
    // cannot create a new ordering proof for a previously rejected candidate.
    for (unsigned index = 0; index < plan.barriers.size(); ++index) {
        const auto& candidate = plan.barriers[index];
        if (!canModelCut(candidate, result)) {
            continue;
        }
        OrderGraph edges = buildSurvivingEdges(result, plan, live, index);
        if (preservesHazards(result, edges, candidate.anchor->getBlock())) {
            live[index] = false;
            continue;
        }
        edges = buildSurvivingEdges(result, plan, live, index, true);
        if (preservesHazards(result, edges, candidate.anchor->getBlock(), true)) {
            live[index] = false;
        }
    }
    SmallVector<BarrierPlacement, 8> kept;
    for (unsigned index = 0; index < plan.barriers.size(); ++index) {
        if (live[index]) {
            kept.push_back(plan.barriers[index]);
        }
    }
    plan.barriers = std::move(kept);
}
