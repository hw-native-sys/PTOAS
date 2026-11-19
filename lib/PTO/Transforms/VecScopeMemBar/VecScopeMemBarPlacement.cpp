// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VecScopeMemBarPlacement.cpp ------------------------------------===//
//
// Solves barrier placement from the read-only analysis result. Picks a
// provably-mandatory anchor for each hazard relation, merges same-kind
// hazards at the same anchor, and recognizes existing-barrier coverage. The
// redundancy pass checks the complete plan before surviving kinds in each
// family are combined into ALL. Outputs a deterministic plan without IR edits.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarPlacement.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

// Hazards are identified by their stable index in the analysis result. Keep
// the complete set on every candidate barrier: a barrier at one anchor can
// cover several producer/consumer pairs of the same kind. Losing all but the
// last id here would hide obligations from the redundancy checker.
struct BarrierKindHazards {
    MemBarKind kind = MemBarKind::VV_ALL;
    SmallVector<unsigned, mlir::pto::kValue4> hazards;
};

// An insertion anchor keyed by the op it precedes, for de-duplication.
// The separate sentinel tag gives DenseMap its reserved keys without forging
// Operation pointers from integers.
struct AnchorKey {
    Operation* op;
    unsigned sentinel = 0;
    bool operator==(const AnchorKey& o) const { return sentinel == o.sentinel && op == o.op; }
};

struct AnchorKeyInfo {
    static AnchorKey getEmptyKey() { return {nullptr, 1}; }
    static AnchorKey getTombstoneKey() { return {nullptr, 2}; }
    static unsigned getHashValue(const AnchorKey& a) { return llvm::hash_combine(a.op, a.sentinel); }
    static bool isEqual(const AnchorKey& a, const AnchorKey& b) { return a == b; }
};

static void appendUniqueHazard(SmallVectorImpl<unsigned>& hazards, unsigned hazardId)
{
    if (!llvm::is_contained(hazards, hazardId)) {
        hazards.push_back(hazardId);
    }
}

static void sortUniqueHazards(SmallVectorImpl<unsigned>& hazards)
{
    llvm::sort(hazards);
    hazards.erase(std::unique(hazards.begin(), hazards.end()), hazards.end());
}

// Enum values encode four families, each with an ALL and three directed kinds.
static bool barrierCovers(MemBarKind covering, MemBarKind covered)
{
    const auto family = [](MemBarKind kind) { return static_cast<unsigned>(kind) / 4; };
    return covering == covered ||
           (family(covering) == family(covered) && static_cast<unsigned>(covering) % mlir::pto::kValue4 == 0);
}

static bool coveredByExisting(
    Operation* anchor, MemBarKind kind, bool isLoopLatch, scf::ForOp carryingLoop, ArrayRef<ExistingBarrier> existing)
{
    Operation* prev = anchor->getPrevNode();
    if (!prev) {
        return false;
    }
    for (const auto& b : existing) {
        if (b.op != prev) {
            continue;
        }
        if (!barrierCovers(b.kind, kind)) {
            continue;
        }
        if (isLoopLatch) {
            // Must be on the same carrying loop's backedge.
            if (b.latchLoop && carryingLoop) {
                // Match by op identity.
                if (b.op->getParentOfType<scf::ForOp>() != carryingLoop) {
                    continue;
                }
            } else if (carryingLoop) {
                if (b.op->getParentOfType<scf::ForOp>() != carryingLoop) {
                    continue;
                }
            }
        }
        return true;
    }
    return false;
}

//===----------------------------------------------------------------------===//
// Anchor scheduling
//===----------------------------------------------------------------------===//

// Per-anchor placement state. `kinds` collects the barrier kinds merged at
// each anchor (multiple kinds in the same family at one anchor -> that
// family's ALL barrier). `latchAnchors` marks genuine loop-latch cuts
// (carrying-loop terminator), distinguished from a plain
// last-op-in-vecscope anchor: vecscope blocks are NoTerminator, so their
// last op is `getBlock()->getTerminator()` without being a latch. Latch
// semantics differ (per-iteration backedge), so the redundant-barrier pass
// must not treat such anchors as latches. `order` records anchors in
// first-discovery order, which is deterministic lexical order; DenseMap
// iteration is intentionally not used for scheduling because anchors can
// belong to different blocks, so isBeforeInBlock would assert.
struct AnchorPlacements {
    DenseMap<AnchorKey, SmallVector<BarrierKindHazards>, AnchorKeyInfo> kinds;
    DenseSet<AnchorKey, AnchorKeyInfo> latchAnchors;
    SmallVector<Operation*, mlir::pto::kValue8> order;
};

// Register one hazard at an anchor: track the kinds merged at the anchor and
// the hazards each kind resolves.
static void addPlacement(
    const VecScopeMemBarAnalysisResult& result, AnchorPlacements& placements, Operation* anchor, MemBarKind kind,
    unsigned hazardId, bool isLoopLatch, scf::ForOp carryingLoop)
{
    if (coveredByExisting(anchor, kind, isLoopLatch, carryingLoop, result.existingBarriers)) {
        return;
    }
    auto& kinds = placements.kinds[{anchor}];
    if (kinds.empty()) {
        placements.order.push_back(anchor);
        if (isLoopLatch) {
            placements.latchAnchors.insert({anchor});
        }
    }

    for (auto& entry : kinds) {
        if (entry.kind == kind) {
            appendUniqueHazard(entry.hazards, hazardId);
            return;
        }
    }
    kinds.push_back({kind, {hazardId}});
}

// Schedule every hazard onto its mandatory anchor: the
// consumer-side cut for same-iteration hazards, or the carrying loop's
// terminator for loop-carried ones.
static void scheduleHazardsOntoAnchors(const VecScopeMemBarAnalysisResult& result, AnchorPlacements& placements)
{
    const auto& A = result.accesses;
    for (const auto& h : result.hazards) {
        Operation* anchor = nullptr;
        bool isLatch = false;
        scf::ForOp carrying = nullptr;
        if (h.scope == VecScopeHazardScope::SameIteration) {
            // Insert at the mandatory consumer-side cut. For cross-hierarchy pairs
            // this is the enclosing consumer loop, not an op inside its body.
            anchor = h.sameIterationAnchor ? h.sameIterationAnchor : A[h.consumer].op;
        } else {
            // Loop-carried: carrying loop terminator.
            if (!h.carryingLoop) {
                continue;
            }
            // Resolve the carrying loop op.
            for (const auto& l : result.loops) {
                if (l.id == *h.carryingLoop) {
                    carrying = l.op;
                    break;
                }
            }
            if (!carrying) {
                continue;
            }
            anchor = carrying.getBody()->getTerminator();
            isLatch = true;
        }
        if (!anchor) {
            continue;
        }
        addPlacement(result, placements, anchor, h.kind, h.id, isLatch, carrying);
    }
}

// Keep directed kinds separate until redundancy has been proved. Merging them
// into ALL first would prevent removing just one redundant direction.
static void materializeBarriers(const AnchorPlacements& placements, VecScopeMemBarPlan& plan)
{
    for (Operation* anchor : placements.order) {
        for (const auto& entry : placements.kinds.find({anchor})->second) {
            BarrierPlacement bp;
            bp.anchor = anchor;
            bp.kind = entry.kind;
            bp.resolvedHazards.assign(entry.hazards.begin(), entry.hazards.end());
            bp.anchorKind = placements.latchAnchors.count({anchor}) != 0 ? BarrierAnchorKind::BeforeLoopTerminator :
                                                                      BarrierAnchorKind::BeforeOperation;
            sortUniqueHazards(bp.resolvedHazards);
            plan.barriers.push_back(std::move(bp));
        }
    }
}

// Normalize only surviving cuts, including an adjacent existing directed bar.
// Combining kinds in one family strengthens ordering and preserves the proof.
static void normalizeBarriers(VecScopeMemBarPlan& plan)
{
    SmallVector<BarrierPlacement> merged;
    for (const auto& barrier : plan.barriers) {
        unsigned family = static_cast<unsigned>(barrier.kind) / 4;
        auto existing = llvm::find_if(merged, [&barrier, &family](const BarrierPlacement& other) {
            return other.anchor == barrier.anchor && static_cast<unsigned>(other.kind) / 4 == family;
        });
        if (existing != merged.end()) {
            existing->kind = static_cast<MemBarKind>(family * mlir::pto::kValue4);
            for (unsigned id : barrier.resolvedHazards) {
                appendUniqueHazard(existing->resolvedHazards, id);
            }
            sortUniqueHazards(existing->resolvedHazards);
            continue;
        }
        merged.push_back(barrier);
        auto adjacent = dyn_cast_or_null<pto::MemBarOp>(barrier.anchor->getPrevNode());
        if (adjacent) {
            MemBarKind kind = adjacent.getKind().getKind();
            if (static_cast<unsigned>(kind) / mlir::pto::kValue4 == family && kind != barrier.kind) {
                merged.back().kind = static_cast<MemBarKind>(family * mlir::pto::kValue4);
            }
        }
    }
    plan.barriers = std::move(merged);
}

} // namespace

FailureOr<VecScopeMemBarPlan> vecscopemembar::solveVecScopeMemBarPlacement(const VecScopeMemBarAnalysisResult& result)
{
    VecScopeMemBarPlan plan;
    AnchorPlacements placements;
    scheduleHazardsOntoAnchors(result, placements);
    materializeBarriers(placements, plan);
    eliminateTransitiveMemBars(result, plan);
    normalizeBarriers(plan);
    return plan;
}
