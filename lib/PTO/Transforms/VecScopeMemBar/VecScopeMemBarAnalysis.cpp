// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VecScopeMemBarAnalysis.cpp --------------------------------------===//
//
// Builds the vecscope schedule, collects access occurrences and existing
// barriers, and enumerates RAW/WAW/WAR hazards: same-iteration, inner- and
// outer-loop-carried via Presburger dependence relations. The value walk here
// resolves pending WAR obligations, including loop yields; candidate removal
// belongs to Redundancy.cpp and uses its stricter straight-line proof. Does not
// mutate the IR.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarAnalysis.h"
#include "PTO/Analysis/PTOMemoryDependenceAnalysis.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;
using namespace mlir::pto::integer_range;

namespace {

// A defining op that writes UB memory is another store: the reverse walk
// must stop there, since relaying through it would mistake an unrelated
// store for the value relay.
static bool definesUBWrite(Operation* defOp)
{
    auto iface = dyn_cast<MemoryEffectOpInterface>(defOp);
    if (!iface) {
        return false;
    }
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> eff;
    iface.getEffects(eff);
    return llvm::any_of(eff, [](const auto& e) {
        return isa<MemoryEffects::Write>(e.getEffect()) && e.getValue() && pto::isUBBackedType(e.getValue().getType());
    });
}

// `sink` may be a result of an scf.for. Result i corresponds to region
// iter-arg i, whose value at each iteration is the i-th operand of the
// body's scf.yield. Return that operand so a dependence carried out of the
// loop (e.g. a reduction result feeding a later store) is followed back into
// the loop body, instead of stopping at the loop's input operands
// (lb/ub/step/init) which never see the body's loads. Nullopt when `sink` is
// not a for-result carrying a region iter-arg.
static std::optional<Value> forResultYieldedValue(Value sink)
{
    auto forOp = dyn_cast<scf::ForOp>(sink.getDefiningOp());
    if (!forOp) {
        return std::nullopt;
    }
    unsigned k = 0;
    for (Value r : forOp.getResults()) {
        if (r == sink) {
            break;
        }
        ++k;
    }
    if (k >= forOp.getNumRegionIterArgs()) {
        return std::nullopt;
    }
    return forOp.getBody()->getTerminator()->getOperand(k);
}

// Block argument: only scf.for iter_args are relayed, via the yield operand.
// Nullopt when `sink` is not a carried iter argument.
static std::optional<Value> iterArgYieldedValue(Value sink)
{
    auto ba = dyn_cast<BlockArgument>(sink);
    if (!ba) {
        return std::nullopt;
    }
    auto forOp = ba.getOwner() ? dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp()) : scf::ForOp();
    if (!forOp) {
        return std::nullopt;
    }
    unsigned arg = ba.getArgNumber();
    // arg 0 is the loop IV; iter_args start at 1.
    if (arg == 0 || arg > forOp.getRegion().getNumArguments() - 1) {
        return std::nullopt;
    }
    return forOp.getBody()->getTerminator()->getOperand(arg - 1);
}

static bool reverseReaches(Value sink, Value target, DenseSet<Value>& visited, unsigned depth, unsigned maxDepth)
{
    if (sink == target) {
        return true;
    }
    if (depth > maxDepth) {
        return false;
    }
    if (!visited.insert(sink).second) {
        return false;
    }

    if (Operation* defOp = sink.getDefiningOp()) {
        if (definesUBWrite(defOp)) {
            return false; // relay through another store: not a value relay
        }
        if (auto yielded = forResultYieldedValue(sink)) {
            if (reverseReaches(*yielded, target, visited, depth + 1, maxDepth)) {
                return true;
            }
        }
        for (Value op : defOp->getOperands()) {
            if (reverseReaches(op, target, visited, depth + 1, maxDepth)) {
                return true;
            }
        }
        return false;
    }

    auto yielded = iterArgYieldedValue(sink);
    if (!yielded) {
        return false;
    }
    return reverseReaches(*yielded, target, visited, depth + 1, maxDepth);
}

static bool valueDependsOn(Value sink, Value target)
{
    DenseSet<Value> visited;
    return reverseReaches(sink, target, visited, 0, 32);
}

// Whether any stored value depends on any loaded value through the value
// relay used to resolve pending WAR hazards.
static bool anyStoredValueDependsOnLoaded(ArrayRef<Value> storedVals, ArrayRef<Value> loadedVals)
{
    for (Value sv : storedVals) {
        for (Value lv : loadedVals) {
            if (valueDependsOn(sv, lv)) {
                return true;
            }
        }
    }
    return false;
}

static std::optional<MemBarKind> hazardKind(const AccessOccurrence& producer, const AccessOccurrence& consumer)
{
    bool producerVector = producer.unit == PTOMemoryAccessUnit::Vector;
    bool consumerVector = consumer.unit == PTOMemoryAccessUnit::Vector;
    // Scalar UB accesses are issued by the ordered scalar pipeline.  The
    // automatic vecscope pass only needs barriers when a vector pipeline is
    // involved; scalar-to-scalar ordering does not require a membar.
    if (!producerVector && !consumerVector) {
        return std::nullopt;
    }
    if (producer.kind == PTOMemoryAccessKind::Store && consumer.kind == PTOMemoryAccessKind::Load) {
        if (producerVector) {
            return consumerVector ? MemBarKind::VST_VLD : MemBarKind::VST_LD;
        }
        return MemBarKind::ST_VLD;
    }
    if (producer.kind == PTOMemoryAccessKind::Store && consumer.kind == PTOMemoryAccessKind::Store) {
        if (producerVector) {
            return consumerVector ? MemBarKind::VST_VST : MemBarKind::VST_ST;
        }
        return MemBarKind::ST_VST;
    }
    if (producer.kind == PTOMemoryAccessKind::Load && consumer.kind == PTOMemoryAccessKind::Store) {
        if (producerVector) {
            return consumerVector ? MemBarKind::VLD_VST : MemBarKind::VLD_ST;
        }
        return MemBarKind::LD_VST;
    }
    return std::nullopt; // RAR
}

// Collect enclosing loop IVs (outermost first) within the vecscope boundary.
// Memory-bar analysis is scoped to a single `pto.vecscope`: any loop outside
// that scope is invisible to it, so the walk stops at `scope` (exclusive).
// Stopping at the scope keeps the invariant that an op directly inside a
// vecscope with no inner loops has an empty IV set, which the same-iteration
// overlap model relies on to compare constant addresses directly.
static SmallVector<Value, 2> collectIVs(Operation* op, Operation* scope)
{
    SmallVector<Value, 2> ivs;
    for (Operation* cur = op->getParentOp(); cur && cur != scope; cur = cur->getParentOp()) {
        if (auto forOp = dyn_cast<scf::ForOp>(cur)) {
            ivs.push_back(forOp.getInductionVar());
        }
    }
    std::reverse(ivs.begin(), ivs.end());
    return ivs;
}

// Whether any loop enclosing `op` within the vecscope `scope` has non-constant
// bounds. Loops outside the scope are invisible to the analysis (see
// `collectIVs`).
static bool hasDynamicLoopDomain(Operation* op, Operation* scope)
{
    for (Operation* cursor = op->getParentOp(); cursor && cursor != scope; cursor = cursor->getParentOp()) {
        if (auto forOp = dyn_cast<scf::ForOp>(cursor)) {
            if (!getConstBounds(forOp).valid) {
                return true;
            }
        }
    }
    return false;
}

static void recordUnknown(
    VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, MemBarKind kind, DependenceReason reason,
    VecScopeHazardScope scope = VecScopeHazardScope::SameIteration, std::optional<unsigned> carryingLoop = {})
{
    result.unknownDependences.push_back({i, j, kind, reason, scope, carryingLoop});
}

static VecScopeHazardScope carriedScopeAt(unsigned level, unsigned numCommonLoops)
{
    return level + 1 == numCommonLoops ? VecScopeHazardScope::InnerLoopCarried : VecScopeHazardScope::OuterLoopCarried;
}

// Record an unknown carried dependence on every shared loop.
static void materializeUnknownDependences(VecScopeMemBarAnalysisResult& result, Operation* scope, unsigned begin)
{
    for (unsigned u = begin; u < result.unknownDependences.size(); ++u) {
        const UnknownDependence& d = result.unknownDependences[u];
        MemoryHazard h;
        h.id = result.hazards.size();
        h.producer = d.producer;
        h.consumer = d.consumer;
        h.kind = d.kind;
        h.scope = d.scope;
        if (d.scope == VecScopeHazardScope::SameIteration) {
            Operation* consumerOp = result.accesses[d.consumer].op;
            h.sameIterationAnchor = findSameIterationAnchor(result.accesses[d.producer].op, consumerOp, scope);
            if (!h.sameIterationAnchor) {
                h.sameIterationAnchor = consumerOp;
            }
        } else {
            h.carryingLoop = d.carryingLoop;
            h.distance.positive = true;
        }
        result.hazards.push_back(h);
    }
}

// Build a flat schedule tree: root Sequence, with loops as children carrying
// their own Sequence body. `schedulePath` is the child-index path from root.
static void buildSchedule(Operation* scope, VecScopeMemBarAnalysisResult& result)
{
    // Root sequence node.
    VecScopeScheduleNode root;
    root.kind = VecScopeNodeKind::Sequence;
    root.op = scope;
    result.schedule.push_back(root);
    unsigned loopId = 0;
    scope->walk([&](scf::ForOp forOp) {
        VecScopeLoopInfo info;
        info.id = loopId++;
        info.op = forOp;
        unsigned depth = 0;
        for (Operation* p = forOp->getParentOp(); p; p = p->getParentOp()) {
            if (isa<scf::ForOp>(p) || p == scope) {
                depth++;
            }
        }
        info.depth = depth;
        info.inductionVar = forOp.getInductionVar();
        info.lowerBound = forOp.getLowerBound();
        info.upperBound = forOp.getUpperBound();
        info.step = forOp.getStep();
        result.loops.push_back(info);
    });
}

// A mem_bar is a vector micro-op and must remain inside the consumer's
// vector scope. For a dependence crossing sibling vecscopes, anchor at the
// first operation in the consumer scope rather than before the scope
// container in the parent block.
static Operation* anchorInsideConsumerScope(Operation* consumer, Operation* c)
{
    Region& body = c->getRegion(0);
    if (!body.empty() && !body.front().empty()) {
        Operation* anchor = &body.front().front();
        while (isa<pto::MemBarOp>(anchor) && anchor->getNextNode()) {
            anchor = anchor->getNextNode();
        }
        return anchor;
    }
    return consumer;
}

// Record enclosing loop ids (outermost first).
static void recordEnclosingLoopNest(
    const VecScopeMemBarAnalysisResult& result, ArrayRef<Value> ivs, AccessOccurrence& occ)
{
    for (Value iv : ivs) {
        for (const auto& l : result.loops) {
            if (l.inductionVar == iv) {
                occ.loopNest.push_back(l.id);
                break;
            }
        }
    }
}

// Collect accesses in lexical order. Returns false when any access cannot be
// modelled.
static bool collectAccessOccurrences(Operation* scope, VecScopeMemBarAnalysisResult& result)
{
    bool anyFailed = false;
    unsigned order = 0;
    scope->walk([&](Operation* op) {
        if (!isUBMemoryOp(op)) {
            return;
        }
        AccessOccurrence occ;
        occ.op = op;
        occ.lexicalOrder = order++;
        auto ivs = collectIVs(op, scope);
        auto maybe = buildMemoryAccessDescriptor(op, ivs);
        if (failed(maybe)) {
            anyFailed = true;
            return;
        }
        occ.kind = (*maybe).kind;
        occ.unit = (*maybe).unit;
        recordEnclosingLoopNest(result, ivs, occ);
        auto footprints = getMemoryFootprints(*maybe, ivs);
        if (footprints.empty()) {
            anyFailed = true;
            return;
        }
        for (const auto& footprint : footprints) {
            occ.footprint = footprint;
            result.accesses.push_back(occ);
        }
    });
    return !anyFailed;
}

// The analysis-result loop id of the loop whose backedge the barrier sits
// on, if any.
static std::optional<unsigned> latchLoopOfBarrier(pto::MemBarOp mb, const VecScopeMemBarAnalysisResult& result)
{
    auto fl = mb->getParentOfType<scf::ForOp>();
    if (!fl) {
        return std::nullopt;
    }
    for (const auto& l : result.loops) {
        if (l.op == fl) {
            return l.id;
        }
    }
    return std::nullopt;
}

// Collect existing pto.mem_bar ops.
static void collectExistingBarriers(Operation* scope, VecScopeMemBarAnalysisResult& result)
{
    scope->walk([&](pto::MemBarOp mb) {
        ExistingBarrier b;
        b.op = mb.getOperation();
        b.kind = mb.getKind().getKind();
        b.phase = mb->getParentOfType<scf::ForOp>() ? ExistingBarrier::LoopLatch : ExistingBarrier::SameIteration;
        b.latchLoop = latchLoopOfBarrier(mb, result);
        result.existingBarriers.push_back(b);
    });
}

// --- Sequential-region RAW/WAW dependences ---
static void recordSequentialHazard(
    VecScopeMemBarAnalysisResult& result, const AccessOccurrence& producer, const AccessOccurrence& consumer,
    unsigned producerIndex, unsigned consumerIndex, MemBarKind kind, Operation* scope)
{
    MemoryHazard h;
    h.id = result.hazards.size();
    h.producer = producerIndex;
    h.consumer = consumerIndex;
    h.kind = kind;
    h.scope = VecScopeHazardScope::SameIteration;
    h.sameIterationAnchor = findSameIterationAnchor(producer.op, consumer.op, scope);
    if (!h.sameIterationAnchor) {
        h.sameIterationAnchor = consumer.op;
    }
    result.hazards.push_back(h);
}

static DependenceStatus sequentialOverlapStatus(
    const AccessOccurrence& producer, const AccessOccurrence& consumer, Operation* scope)
{
    auto status =
        querySequentialMemoryOverlap({producer.op, producer.footprint}, {consumer.op, consumer.footprint}, scope);
    if (status == PTOMemoryDependenceStatus::NoDependence) {
        return DependenceStatus::NoDependence;
    }
    return status == PTOMemoryDependenceStatus::Unknown ? DependenceStatus::Unknown :
                                                          DependenceStatus::ProvenDependence;
}

static void analyzeSequentialPair(
    VecScopeMemBarAnalysisResult& result, Operation* scope, unsigned i, unsigned j,
    SmallVectorImpl<std::pair<unsigned, unsigned>>& pendingWAR)
{
    const auto& A = result.accesses;
    if (A[i].op == A[j].op) {
        return;
    }
    auto kind = hazardKind(A[i], A[j]);
    if (!kind) {
        return;
    }
    const bool sameIterationNoAlias =
        aliasMemorySameIteration(A[i].footprint, A[j].footprint) == PTOMemoryAliasResult::NoAlias;
    if (sameIterationNoAlias) {
        return;
    }
    bool dynamicDomain = hasDynamicLoopDomain(A[i].op, scope) || hasDynamicLoopDomain(A[j].op, scope);
    bool envelopeDisjoint =
        querySequentialMemoryOverlap({A[i].op, A[i].footprint}, {A[j].op, A[j].footprint}, scope, true) ==
        PTOMemoryDependenceStatus::NoDependence;
    DependenceStatus status = sequentialOverlapStatus(A[i], A[j], scope);
    if (dynamicDomain) {
        status = envelopeDisjoint ? DependenceStatus::NoDependence : DependenceStatus::ProvenDependence;
    } else if (status == DependenceStatus::Unknown && envelopeDisjoint) {
        status = DependenceStatus::NoDependence;
    }
    if (status == DependenceStatus::NoDependence) {
        return;
    }
    if (status == DependenceStatus::Unknown) {
        recordUnknown(result, i, j, *kind, DependenceReason::DynamicUnmodelledExpression);
        return;
    }
    if (A[i].kind == PTOMemoryAccessKind::Load && A[j].kind == PTOMemoryAccessKind::Store) {
        pendingWAR.push_back({i, j});
        return;
    }
    recordSequentialHazard(result, A[i], A[j], i, j, *kind, scope);
}

static void enumerateSequentialHazards(
    VecScopeMemBarAnalysisResult& result, Operation* scope, SmallVectorImpl<std::pair<unsigned, unsigned>>& pendingWAR)
{
    const unsigned N = result.accesses.size();
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = i + 1; j < N; ++j) {
            analyzeSequentialPair(result, scope, i, j, pendingWAR);
        }
    }
}

// WAR-ordering graph: edge (u, v) means access u is already ordered before
// access v by SSA data flow or an existing/scheduled VST_VLD barrier.
using OrderingEdges = SmallVector<SmallVector<unsigned, 4>, 16>;

static void addOrderingEdge(OrderingEdges& edges, unsigned from, unsigned to)
{
    if (!llvm::is_contained(edges[from], to)) {
        edges[from].push_back(to);
    }
}

// Forward reachability over the ordering graph.
static bool reachesViaEdges(const OrderingEdges& edges, unsigned from, unsigned to)
{
    SmallVector<bool, 16> reachable(edges.size(), false);
    SmallVector<unsigned, 16> worklist;
    reachable[from] = true;
    worklist.push_back(from);
    while (!worklist.empty()) {
        unsigned current = worklist.pop_back_val();
        for (unsigned next : edges[current]) {
            if (reachable[next]) {
                continue;
            }
            reachable[next] = true;
            worklist.push_back(next);
        }
    }
    return reachable[to];
}

// SSA data dependence: a vector load is ordered before a later store if
// one of the store's payload values depends on one of the load results.
static void addSSADataEdges(const VecScopeMemBarAnalysisResult& result, OrderingEdges& edges)
{
    const auto& A = result.accesses;
    const unsigned N = A.size();
    for (unsigned load = 0; load < N; ++load) {
        if (A[load].kind != PTOMemoryAccessKind::Load) {
            continue;
        }
        SmallVector<Value, 2> loadedVals = getLoadedValues(A[load].op);
        if (loadedVals.empty()) {
            continue;
        }
        for (unsigned store = load + 1; store < N; ++store) {
            if (A[store].kind != PTOMemoryAccessKind::Store) {
                continue;
            }
            SmallVector<Value, 2> storedVals = getStoredValues(A[store].op);
            if (anyStoredValueDependsOnLoaded(storedVals, loadedVals)) {
                addOrderingEdge(edges, load, store);
            }
        }
    }
}

// A generated barrier executes immediately before its placement anchor.
// Only use endpoints in that block: lexical order across a conditional or
// zero-trip region does not prove that the barrier executes between them.
static void addCutOrderingEdges(
    ArrayRef<AccessOccurrence> A, Operation* anchor, PTOMemoryAccessUnit storeUnit, PTOMemoryAccessUnit loadUnit,
    OrderingEdges& edges)
{
    for (unsigned store = 0; store < A.size(); ++store) {
        if (A[store].kind != PTOMemoryAccessKind::Store || A[store].unit != storeUnit ||
            A[store].op->getBlock() != anchor->getBlock() || !A[store].op->isBeforeInBlock(anchor)) {
            continue;
        }
        for (unsigned load = 0; load < A.size(); ++load) {
            if (A[load].kind == PTOMemoryAccessKind::Load && A[load].unit == loadUnit &&
                A[load].op->getBlock() == anchor->getBlock() &&
                (A[load].op == anchor || anchor->isBeforeInBlock(A[load].op))) {
                addOrderingEdge(edges, store, load);
            }
        }
    }
}

// Use the same anchor as placement, including enclosing consumer loops.
// Its block and position, rather than the consumer access index, determine
// which store/load pairs may use the generated barrier as an ordering proof.
static void addHazardCutEdges(const VecScopeMemBarAnalysisResult& result, OrderingEdges& edges)
{
    const auto& A = result.accesses;
    const unsigned N = A.size();
    for (const auto& h : result.hazards) {
        if (h.scope != VecScopeHazardScope::SameIteration ||
            (h.kind != MemBarKind::VST_VLD && h.kind != MemBarKind::VST_LD && h.kind != MemBarKind::ST_VLD &&
             h.kind != MemBarKind::ST_LD) ||
            h.consumer >= N) {
            continue;
        }
        PTOMemoryAccessUnit storeUnit = h.kind == MemBarKind::VST_VLD || h.kind == MemBarKind::VST_LD ?
                                            PTOMemoryAccessUnit::Vector :
                                            PTOMemoryAccessUnit::Scalar;
        PTOMemoryAccessUnit loadUnit = h.kind == MemBarKind::VST_VLD || h.kind == MemBarKind::ST_VLD ?
                                           PTOMemoryAccessUnit::Vector :
                                           PTOMemoryAccessUnit::Scalar;
        Operation* anchor = h.sameIterationAnchor ? h.sameIterationAnchor : A[h.consumer].op;
        addCutOrderingEdges(A, anchor, storeUnit, loadUnit, edges);
    }
}

// Only use an existing barrier as an ordering witness within its own block.
// Lexical order across regions does not prove execution: an scf.if arm may
// not run, and even an scf.for body can execute zero times. Restrict both
// endpoints so a conditional fence cannot suppress an unconditional WAR cut.
// Cross-region proofs require control-flow reasoning and remain conservative.
// Both units are derived from the barrier kind's family: the family's
// first letter is the store unit, its second letter the load unit.
static std::optional<std::pair<PTOMemoryAccessUnit, PTOMemoryAccessUnit>> barrierStoreLoadUnits(MemBarKind kind)
{
    using Unit = PTOMemoryAccessUnit;
    switch (kind) {
        case MemBarKind::VST_VLD:
        case MemBarKind::VV_ALL:
            return std::make_pair(Unit::Vector, Unit::Vector);
        case MemBarKind::VST_LD:
        case MemBarKind::VS_ALL:
            return std::make_pair(Unit::Vector, Unit::Scalar);
        case MemBarKind::ST_VLD:
        case MemBarKind::SV_ALL:
            return std::make_pair(Unit::Scalar, Unit::Vector);
        case MemBarKind::ST_LD:
        case MemBarKind::SS_ALL:
            return std::make_pair(Unit::Scalar, Unit::Scalar);
        default:
            return std::nullopt;
    }
}

static void addBarrierSpanEdges(ArrayRef<AccessOccurrence> A, const ExistingBarrier& barrier, OrderingEdges& edges)
{
    auto units = barrierStoreLoadUnits(barrier.kind);
    if (!units) {
        return;
    }
    const unsigned N = A.size();
    for (unsigned store = 0; store < N; ++store) {
        if (A[store].kind != PTOMemoryAccessKind::Store ||
            A[store].unit != units->first ||
            A[store].op->getBlock() != barrier.op->getBlock() || !A[store].op->isBeforeInBlock(barrier.op)) {
            continue;
        }
        for (unsigned load = 0; load < N; ++load) {
            if (A[load].kind != PTOMemoryAccessKind::Load ||
                A[load].unit != units->second ||
                A[load].op->getBlock() != barrier.op->getBlock() || !barrier.op->isBeforeInBlock(A[load].op)) {
                continue;
            }
            addOrderingEdge(edges, store, load);
        }
    }
}

// Existing store-to-load barriers (directed or ALL) provide the same
// ordering edge as generated ones. This matters when the input already
// contains the first half of a chain: a later WAR pair must be recognized
// as ordered even though no new RAW hazard was materialized for that
// barrier in this run.
static void addExistingBarrierEdges(const VecScopeMemBarAnalysisResult& result, OrderingEdges& edges)
{
    const auto& A = result.accesses;
    for (const auto& barrier : result.existingBarriers) {
        if (barrier.kind != MemBarKind::VST_VLD && barrier.kind != MemBarKind::VST_LD &&
            barrier.kind != MemBarKind::ST_VLD && barrier.kind != MemBarKind::ST_LD &&
            barrier.kind != MemBarKind::VS_ALL && barrier.kind != MemBarKind::SV_ALL &&
            barrier.kind != MemBarKind::SS_ALL && barrier.kind != MemBarKind::VV_ALL) {
            continue;
        }
        addBarrierSpanEdges(A, barrier, edges);
    }
}

// Resolve WAR hazards over the combined SSA + VST_VLD ordering graph. The
// graph is deliberately built after the first pair pass so it contains all
// same-iteration RAW hazards, including hazards whose producer/consumer
// lexical order is later than a pending WAR pair.
static void resolvePendingWARHazards(
    VecScopeMemBarAnalysisResult& result, Operation* scope, SmallVectorImpl<std::pair<unsigned, unsigned>>& pendingWAR)
{
    if (pendingWAR.empty()) {
        return;
    }
    const auto& A = result.accesses;
    const unsigned N = A.size();
    OrderingEdges edges(N);
    addSSADataEdges(result, edges);
    addHazardCutEdges(result, edges);
    addExistingBarrierEdges(result, edges);

    for (const auto& [load, store] : pendingWAR) {
        if (reachesViaEdges(edges, load, store)) {
            continue;
        }
        MemoryHazard h;
        h.id = result.hazards.size();
        h.producer = load;
        h.consumer = store;
        h.kind = hazardKind(A[load], A[store]).value();
        h.scope = VecScopeHazardScope::SameIteration;
        h.sameIterationAnchor = findSameIterationAnchor(A[load].op, A[store].op, scope);
        if (!h.sameIterationAnchor) {
            h.sameIterationAnchor = A[store].op;
        }
        result.hazards.push_back(h);
    }
}

static DependenceReason dependenceReason(PTOMemoryDependenceReason reason)
{
    switch (reason) {
        case PTOMemoryDependenceReason::InvalidContext:
            return DependenceReason::DynamicUnmodelledExpression;
        case PTOMemoryDependenceReason::UnknownRoot:
            return DependenceReason::UnknownRoot;
        case PTOMemoryDependenceReason::DynamicUnmodelledExpression:
            return DependenceReason::DynamicUnmodelledExpression;
        case PTOMemoryDependenceReason::DynamicLoopBounds:
            return DependenceReason::DynamicLoopBounds;
        case PTOMemoryDependenceReason::UnsupportedAccessShape:
            return DependenceReason::UnsupportedAccessShape;
    }
    llvm_unreachable("invalid memory dependence reason");
}

static void analyzeLoopCarriedPair(VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, Operation* scope)
{
    const auto& producer = result.accesses[i];
    const auto& consumer = result.accesses[j];
    auto kind = hazardKind(producer, consumer);
    if (!kind) {
        return;
    }
    auto common = getCommonLoopPrefix(producer.loopNest, consumer.loopNest);
    auto answers =
        queryCarriedMemoryOverlap({producer.op, producer.footprint}, {consumer.op, consumer.footprint}, scope);
    for (const auto& answer : answers) {
        auto found = llvm::find_if(result.loops, [&](const auto& loop) { return loop.op == answer.loop; });
        if (found == result.loops.end()) {
            recordUnknown(result, i, j, *kind, DependenceReason::DynamicUnmodelledExpression);
            for (auto [level, id] : llvm::enumerate(common)) {
                recordUnknown(result, i, j, *kind, DependenceReason::DynamicUnmodelledExpression,
                              carriedScopeAt(level, common.size()), id);
            }
            continue;
        }
        unsigned level = llvm::find(common, found->id) - common.begin();
        auto hazardScope = carriedScopeAt(level, common.size());
        if (answer.status == PTOMemoryDependenceStatus::Unknown) {
            recordUnknown(result, i, j, *kind, dependenceReason(answer.reason), hazardScope, found->id);
            continue;
        }
        MemoryHazard hazard;
        hazard.id = result.hazards.size();
        hazard.producer = i;
        hazard.consumer = j;
        hazard.kind = *kind;
        hazard.scope = hazardScope;
        hazard.carryingLoop = found->id;
        hazard.distance.positive = true;
        result.hazards.push_back(hazard);
    }
}

// --- Loop-carried RAW/WAW hazards via Presburger relations ---
// For each ordered pair (i, j), including self pairs, query Rinner and
// Router whenever both accesses share an enclosing loop. A pair can be
// loop-carried even when its same-iteration instances are statically
// ordered in either direction.
static void enumerateLoopCarriedHazards(VecScopeMemBarAnalysisResult& result, Operation* scope)
{
    const unsigned N = result.accesses.size();
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            analyzeLoopCarriedPair(result, i, j, scope);
        }
    }
}

} // namespace

Operation* vecscopemembar::findSameIterationAnchor(Operation* producer, Operation* consumer, Operation* scope)
{
    for (Operation* p = producer; p && p != scope; p = p->getParentOp()) {
        for (Operation* c = consumer; c && c != scope; c = c->getParentOp()) {
            if (p->getBlock() != c->getBlock() || !p->isBeforeInBlock(c)) {
                continue;
            }
            if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(c)) {
                return anchorInsideConsumerScope(consumer, c);
            }
            return c;
        }
    }
    return nullptr;
}

FailureOr<VecScopeMemBarAnalysisResult> vecscopemembar::runVecScopeMemBarAnalysis(Operation* scope)
{
    VecScopeMemBarAnalysisResult result;
    result.scope = scope;
    buildSchedule(scope, result);

    if (!collectAccessOccurrences(scope, result)) {
        return failure();
    }
    collectExistingBarriers(scope, result);

    SmallVector<std::pair<unsigned, unsigned>, 16> pendingWAR;
    enumerateSequentialHazards(result, scope, pendingWAR);
    // Sequential unknowns become hazard cuts before WAR resolution so that a
    // conservative barrier can still discharge a pending WAR chain.
    materializeUnknownDependences(result, scope, 0);
    resolvePendingWARHazards(result, scope, pendingWAR);
    unsigned carriedUnknownBegin = result.unknownDependences.size();
    enumerateLoopCarriedHazards(result, scope);
    materializeUnknownDependences(result, scope, carriedUnknownBegin);
    return result;
}
