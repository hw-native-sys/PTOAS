// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Prove slot-event coverage before event allocation. Edges order operation
// completion before another operation starts and carry an exact iteration
// distance. An alternate path must have the SAME distance, not merely reach
// the same physical slot in a later rotation. Only straight, top-level,
// constant-bounded unit-step loops participate; their outside dependencies and
// all barriers remain intact. No implicit ordering between vector operations
// is assumed. Allocation still owns event IDs and the surviving primes/drains.

#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {

// Bound proof cost independently of loop trip count and user-provided IR size.
constexpr size_t kMaxProofNodes = 256;
constexpr size_t kMaxProofEdges = 512;

struct IterationEdge {
    unsigned source;
    unsigned target;
    unsigned distance;
    SyncOperation* sync;
    SyncOperation* wait{nullptr};
};

struct LoopProof {
    scf::ForOp loop;
    unsigned begin;
    unsigned end;
    int64_t lowerBound;
    int64_t upperBound;
    SmallVector<IterationEdge> edges;
    SmallVector<SmallVector<unsigned, 2>> outgoing;

    bool contains(unsigned index) const { return begin < index && index < end; }
};

std::optional<int64_t> getSignedConstant(Value value)
{
    IntegerAttr attr;
    if (!matchPattern(value, m_Constant(&attr)) || !attr.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return attr.getValue().getSExtValue();
}

std::optional<LoopProof> getLoopProof(const LoopInstanceElement& element, const SyncIRs& syncIR)
{
    auto loop = dyn_cast_if_present<scf::ForOp>(element.elementOp);
    if (!loop || !isa<func::FuncOp>(loop->getParentOp()) || !matchPattern(loop.getStep(), m_One()) ||
        element.beginId >= element.endId || element.endId >= syncIR.size() ||
        element.endId > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        element.endId - element.beginId > kMaxProofNodes) {
        return std::nullopt;
    }
    auto lower = getSignedConstant(loop.getLowerBound());
    auto upper = getSignedConstant(loop.getUpperBound());
    if (!lower || !upper || *lower < 0 || *upper <= *lower || !loop.getInductionVar().getType().isIndex()) {
        return std::nullopt;
    }
    for (unsigned i = element.beginId + 1; i < element.endId; ++i) {
        auto* compound = dyn_cast<CompoundInstanceElement>(syncIR[i].get());
        if (!compound || !compound->elementOp || compound->elementOp->getParentOp() != loop ||
            compound->macroOpInstanceId >= 0) {
            return std::nullopt;
        }
    }
    return LoopProof{loop, element.beginId, element.endId, *lower, *upper, {}, {}};
}

// Accept (iv + nonnegative constant) % count, with no integer wraparound.
// Other symbols, non-unit rotations and narrowing casts need a richer proof.
std::optional<unsigned> getRotationOffset(Value slot, const LoopProof& proof, unsigned count)
{
    auto rem = slot.getDefiningOp<arith::RemUIOp>();
    if (!rem || getSignedConstant(rem.getRhs()) != count) {
        return std::nullopt;
    }
    Value inner = rem.getLhs();
    scf::ForOp loop = proof.loop;
    Value induction = loop.getInductionVar();
    if (inner == induction) {
        return 0;
    }
    auto add = inner.getDefiningOp<arith::AddIOp>();
    if (!add || (add.getLhs() != induction && add.getRhs() != induction)) {
        return std::nullopt;
    }
    Value constant = add.getLhs() == induction ? add.getRhs() : add.getLhs();
    auto offset = getSignedConstant(constant);
    if (!offset || *offset < 0 || *offset > std::numeric_limits<int64_t>::max() - proof.upperBound) {
        return std::nullopt;
    }
    return static_cast<unsigned>(*offset % count);
}

std::optional<unsigned> getEventDistance(const SyncOperation& set, const SyncOperation& wait, const LoopProof& proof)
{
    bool carried = set.GetForEndIndex().has_value();
    if (set.GetForEndIndex() != wait.GetForEndIndex() ||
        (carried && *set.GetForEndIndex() != static_cast<int>(proof.end))) {
        return std::nullopt;
    }
    if (!isSlotKeyedSync(&set) && !isSlotKeyedSync(&wait)) {
        // Only ordinary, unprimed forward events provide static coverage here.
        if (!carried && set.eventIdNum == 1 && wait.eventIdNum == 1 && set.GetSyncIRIndex() < wait.GetSyncIRIndex()) {
            return 0;
        }
        return std::nullopt;
    }
    unsigned count = set.slotCount;
    if (!carried || !isSlotKeyedSync(&set) || !isSlotKeyedSync(&wait) || count < 2 || count > kMaxMultiBufferCount ||
        wait.slotCount != count || set.eventIdNum != static_cast<int>(count) || set.eventIdNum != wait.eventIdNum) {
        return std::nullopt;
    }
    auto producer = getRotationOffset(set.slotSSAExpr, proof, count);
    auto consumer = getRotationOffset(wait.slotSSAExpr, proof, count);
    if (!producer || !consumer) {
        return std::nullopt;
    }
    unsigned distance = (*producer + count - *consumer) % count;
    if (distance == 0 && set.GetSyncIRIndex() >= wait.GetSyncIRIndex()) {
        distance = count;
    }
    return distance;
}

void addBarrierEdge(SyncOperation& barrier, LoopProof& proof, const SyncIRs& syncIR)
{
    unsigned source = barrier.GetDepSyncIRIndex();
    unsigned target = barrier.GetSyncIRIndex();
    PipelineType pipe = barrier.GetSrcPipe();
    if (barrier.GetType() != SyncOperation::TYPE::PIPE_BARRIER || pipe != barrier.GetDstPipe() ||
        pipe == PipelineType::PIPE_ALL || !proof.contains(source) || !proof.contains(target)) {
        return;
    }
    auto func = proof.loop->getParentOfType<func::FuncOp>();
    if (pipe == PipelineType::PIPE_V && isTargetArchA5(func)) {
        return; // A5 codegen erases PIPE_V barriers.
    }
    auto* from = cast<CompoundInstanceElement>(syncIR[source].get());
    auto* to = cast<CompoundInstanceElement>(syncIR[target].get());
    if (from->kPipeValue == pipe && to->kPipeValue == pipe) {
        proof.edges.push_back({source, target, source < target ? 0U : 1U, &barrier});
    }
}

bool collectEdges(LoopProof& proof, const SyncIRs& syncIR, const SyncOperations& syncOperations)
{
    for (const auto& pair : syncOperations) {
        if (pair.empty() || pair[0]->uselessSync || pair[0]->isCompensation) {
            continue;
        }
        if (pair.size() == 1) {
            addBarrierEdge(*pair[0], proof, syncIR);
        } else if (pair.size() == 2) {
            const auto& set = *pair[0];
            const auto& wait = *pair[1];
            if (set.GetType() != SyncOperation::TYPE::SET_EVENT || wait.GetType() != SyncOperation::TYPE::WAIT_EVENT ||
                wait.uselessSync || wait.isCompensation || !proof.contains(set.GetSyncIRIndex()) ||
                !proof.contains(wait.GetSyncIRIndex())) {
                continue;
            }
            auto* from = cast<CompoundInstanceElement>(syncIR[set.GetSyncIRIndex()].get());
            auto* to = cast<CompoundInstanceElement>(syncIR[wait.GetSyncIRIndex()].get());
            if (set.GetSrcPipe() != from->kPipeValue || wait.GetDstPipe() != to->kPipeValue) {
                continue;
            }
            auto distance = getEventDistance(set, wait, proof);
            if (!distance && (isSlotKeyedSync(&set) || isSlotKeyedSync(&wait))) {
                return false;
            }
            if (distance) {
                proof.edges.push_back(
                    {set.GetSyncIRIndex(), wait.GetSyncIRIndex(), *distance, pair[0].get(), pair[1].get()});
            }
        }
        if (proof.edges.size() > kMaxProofEdges) {
            return false;
        }
    }
    proof.outgoing.resize(proof.end - proof.begin);
    for (unsigned i = 0; i < proof.edges.size(); ++i) {
        proof.outgoing[proof.edges[i].source - proof.begin].push_back(i);
    }
    return true;
}

bool hasAlternatePath(const IterationEdge& candidate, const LoopProof& proof)
{
    // Expanding at most one rotation makes each search finite. Nonnegative
    // distances keep the path within the candidate's real endpoint iterations,
    // so the proof also applies to short trips and the first/last iterations.
    using State = std::pair<unsigned, unsigned>;
    SmallVector<State> worklist{{candidate.source, 0}};
    llvm::DenseSet<State> visited;
    visited.insert(worklist.front());
    while (!worklist.empty()) {
        auto [node, elapsed] = worklist.pop_back_val();
        for (unsigned index : proof.outgoing[node - proof.begin]) {
            const auto& edge = proof.edges[index];
            if (edge.sync == candidate.sync || edge.sync->uselessSync || edge.distance > candidate.distance - elapsed) {
                continue;
            }
            unsigned nextDistance = elapsed + edge.distance;
            if (edge.target == candidate.target && nextDistance == candidate.distance) {
                return true;
            }
            State next{edge.target, nextDistance};
            if (visited.insert(next).second) {
                worklist.push_back(next);
            }
        }
    }
    return false;
}

uint32_t getPendingSlots(unsigned phase, unsigned producer, unsigned consumer, unsigned count, bool producerFirst)
{
    uint32_t mask = 0;
    for (unsigned lane = 0; lane < count; ++lane) {
        unsigned firstSet = (lane + 2 * count - phase - producer) % count;
        unsigned firstWait = (lane + 2 * count - phase - consumer) % count;
        bool pending = producerFirst ? firstWait < firstSet : firstWait <= firstSet;
        if (pending) {
            mask |= uint32_t{1} << lane;
        }
    }
    return mask;
}

void setBoundaryMasks(const IterationEdge& edge, const LoopProof& proof)
{
    if (!edge.wait || edge.sync->uselessSync || !isSlotKeyedSync(edge.sync)) {
        return;
    }
    unsigned count = edge.sync->slotCount;
    auto producer = getRotationOffset(edge.sync->slotSSAExpr, proof, count);
    auto consumer = getRotationOffset(edge.wait->slotSSAExpr, proof, count);
    if (!producer || !consumer) {
        return;
    }
    unsigned start = static_cast<unsigned>(proof.lowerBound % count);
    unsigned end = static_cast<unsigned>(proof.upperBound % count);
    bool producerFirst = edge.source < edge.target;
    SlotEventBoundaryMasks masks{
        getPendingSlots(start, *producer, *consumer, count, producerFirst),
        getPendingSlots(end, *producer, *consumer, count, producerFirst)};
    edge.sync->slotBoundaryMasks = masks;
    edge.wait->slotBoundaryMasks = masks;
}

} // namespace

void RemoveRedundantSync::RemoveSlotRedundantSync()
{
    if (syncAnalysisMode_ != SyncAnalysisMode::NORMALSYNC) {
        return;
    }
    for (const auto& element : syncIR_) {
        auto* loop = dyn_cast<LoopInstanceElement>(element.get());
        if (!loop || loop->getLoopKind() != KindOfLoop::LOOP_BEGIN) {
            continue;
        }
        auto proof = getLoopProof(*loop, syncIR_);
        if (!proof || !collectEdges(*proof, syncIR_, syncOperations_)) {
            continue;
        }
        bool removed = false;
        for (const auto& edge : proof->edges) {
            if (!edge.wait || !isSlotKeyedSync(edge.sync) || edge.sync->uselessSync ||
                !hasAlternatePath(edge, *proof)) {
                continue;
            }
            InstanceElement::RemoveSync(syncIR_[edge.source]->pipeAfter, edge.sync);
            InstanceElement::RemoveSync(syncIR_[edge.target]->pipeBefore, edge.wait);
            edge.sync->uselessSync = true;
            edge.wait->uselessSync = true;
            removed = true;
        }
        if (removed) {
            for (const auto& edge : proof->edges) {
                setBoundaryMasks(edge, *proof);
            }
        }
    }
}
