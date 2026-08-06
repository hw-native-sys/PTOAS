// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- UnifiedSyncModel.cpp - Unified sync allocator data model -----------===//
//
// Builds the hazard set the allocator colors. Decides nothing.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/InsertSync/UnifiedSyncModel.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include <limits>
#include <map>
#include <set>
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/Transforms/InsertSync/SyncOracleExtract.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::unified;

//===----------------------------------------------------------------------===//
// ResourceModel
//===----------------------------------------------------------------------===//

unsigned ResourceModel::eventPoolSize(PipelineType src, PipelineType dst) const {
  // Single source of truth, shared with the G3 gate: the allocator must not be
  // able to believe in a bigger pool than the gate will accept.
  uint64_t reserved = SyncEventIdAllocation::GetReservedEventIdNum(src, dst);
  if (reserved >= kTotalEventIdNum)
    return 0;
  return static_cast<unsigned>(kTotalEventIdNum - reserved);
}

//===----------------------------------------------------------------------===//
// Model construction
//===----------------------------------------------------------------------===//

namespace {

/// A direction key. Event flags are per-(src,dst) DISJOINT hardware, so every
/// per-direction quantity -- the pool, the counter, the coloring -- must
/// key on the ordered pair, never on the id alone.
using DirKey = std::pair<int, int>;

DirKey dirKeyOf(const Hazard &h) {
  return {static_cast<int>(h.srcPipe()), static_cast<int>(h.dstPipe())};
}

} // namespace

SyncModel mlir::pto::unified::buildSyncModel(
    const SyncOperations &syncOps, unsigned bufidCapacity,
    const MemInfoToClusters &memInfoToClusters, const SyncIRs &syncIR) {
  SyncModel model;
  model.resources.bufidCapacity = bufidCapacity;

  // --- 1. One Hazard per sync GROUP ---------------------------------------
  // The outer index of SyncOperations is already the hazard: InsertSyncAnalysis
  // emplaces one group per ordering, holding that ordering's set and wait (or a
  // lone barrier). Taking the group as the unit is what makes the both-halves
  // invariant structural rather than a convention.
  for (const auto &group : llvm::enumerate(syncOps)) {
    SyncOperation *setOp = nullptr;
    SyncOperation *waitOp = nullptr;
    for (const auto &owned : group.value()) {
      SyncOperation *s = owned.get();
      if (!s || s->uselessSync)
        continue;
      if (s->isSyncSetType() || s->isBarrierType()) {
        // A barrier group has no wait side; it occupies the `set` slot so that
        // `setOp` is never null and callers need no special case to read a pipe.
        if (!setOp)
          setOp = s;
      } else if (s->isSyncWaitType()) {
        if (!waitOp)
          waitOp = s;
      }
    }
    if (!setOp)
      continue; // fully-pruned group (every member uselessSync)

    Hazard h(static_cast<unsigned>(group.index()), setOp, waitOp);

    // --- 2. Interval, in SyncIR-index space --------------------------------
    // FORWARD hazard: the tight [set, wait) range -- the indices MoveSyncState
    // rewrites when it hoists. A barrier is a point (start == end).
    //
    // LOOP-CARRIED (backward) hazard: its interval is the CARRYING loop, not the
    // raw [set, wait) (which inverts, because the wait is satisfied by the
    // previous iteration). `setOp->GetForEndIndex()` indexes the carrying
    // `LoopInstanceElement`, whose begin/end is where the synthesised head-set /
    // tail-wait sit -- so that span is what the event id is held over.
    //
    // The old `if (end < start) end = start` clamp is DELETED, not patched (doc
    // build order): a backward hazard never reaches the raw-inversion branch,
    // because GetForEndIndex routes it to the (always-forward) carrying loop.
    //
    // `endId + 1`, and the +1 is load-bearing -- an interval must COVER every
    // op it owns. `Interval` is half-open, so [beginId, endId) would exclude
    // position endId, which is exactly where this hazard's own TAIL-WAIT sits.
    // With the short interval the colourer sees [begin,endId) and a following
    // hazard starting at endId as non-overlapping and reuses the id -- while in
    // program order the tail-wait and that hazard's set collide on one SyncIR
    // element, so whether the reuse is safe depends on emission order within the
    // element. The +1 is not a simplifiable off-by-one.
    unsigned start = setOp->GetSyncIRIndex();
    unsigned end = waitOp ? waitOp->GetSyncIRIndex() : start;
    if (setOp->GetForEndIndex().has_value()) {
      const auto *loopEl = llvm::dyn_cast<LoopInstanceElement>(
          syncIR[setOp->GetForEndIndex().value()].get());
      if (loopEl) {
        start = loopEl->beginId;
        end = loopEl->endId + 1;
      } else if (end < start) {
        // GetForEndIndex did not resolve to a loop element -- keep a well-formed
        // interval; synthesizeLoopCompensation will skip it and the per-kernel
        // pair-count check (fe == synthesised) will surface the inconsistency.
        end = start;
      }
    }
    h.setInterval(Interval{start, end});

    // --- 3. Join the hazard to its buffer-id cluster(s) ---------------------
    // Resolved by the caller via MemAlias -- the relation BufidSyncAnalysis
    // itself uses to decide two tiles are the same memory. See MemInfoToClusters
    // for the three plausible-looking keys that are all wrong.
    for (const BaseMemInfo *info : setOp->depMemInfos) {
      if (!info)
        continue;
      auto it = memInfoToClusters.find(info);
      if (it == memInfoToClusters.end())
        continue;
      for (int c : it->second)
        if (!llvm::is_contained(h.bufferClusters, c))
          h.bufferClusters.push_back(c);
    }
    llvm::sort(h.bufferClusters); // determinism

    model.hazards.push_back(std::move(h));
  }

  // --- 4. Is there a GENUINE reverse hazard on this buffer? ----------------
  //
  // A token's get/rel counters run both ways, so one token discharges the forward
  // RAW *and* the reverse WAR -- a saving only if the reverse edge is actually
  // wanted. Forward-only, the same token manufactures a back edge nobody asked for,
  // which costs more than the event it replaced. So this predicate decides whether
  // buffer-ID is cheap or wasteful, and a FALSE POSITIVE here is the expensive
  // direction: it prices the token low and routes a forward-only hazard onto it.
  //
  // "Reverse" = a mirror-direction hazard SHARING A CLUSTER. The cluster is the
  // right unit because one token covers one aliasing clique. Matching on
  // `depRootBuffers` instead -- the root ALLOCATION -- would call two hazards on
  // different tiles reverse partners merely because their tiles share a root,
  // which is the false positive above.
  for (Hazard &h : model.hazards) {
    if (h.isBarrier() || h.bufferClusters.empty())
      continue;
    for (const Hazard &other : model.hazards) {
      if (other.index() == h.index() || other.isBarrier())
        continue;
      if (other.srcPipe() != h.dstPipe() || other.dstPipe() != h.srcPipe())
        continue; // not the mirror direction
      for (int c : other.bufferClusters) {
        if (llvm::is_contained(h.bufferClusters, c)) {
          h.hasReverseWar = true;
          break;
        }
      }
      if (h.hasReverseWar)
        break;
    }
  }

  // --- 5. Buffers -- the unit routing decides on --------------------------
  //
  // Built AFTER hazards because a buffer's facts are derived from the hazards that
  // touch it. Nothing here assigns an id or routes anything; this is the data
  // model that Step 2's router will read.

  // Loop spans, from the SyncIR. A LoopInstanceElement's [beginId, endId] is the
  // span; the INNERMOST containing span is an access's loop context.
  struct LoopSpanRec {
    unsigned begin;
    unsigned end;
  };
  llvm::SmallVector<LoopSpanRec> loopSpans;
  for (const auto &element : syncIR) {
    if (auto *loop = dyn_cast_or_null<pto::LoopInstanceElement>(element.get()))
      if (loop->getLoopKind() == pto::KindOfLoop::LOOP_BEGIN)
        loopSpans.push_back({loop->beginId, loop->endId});
  }
  // Innermost span containing `idx`, or -1 for function level. Innermost == the
  // narrowest containing span.
  auto contextOf = [&](unsigned idx) -> int {
    int best = -1;
    unsigned bestWidth = std::numeric_limits<unsigned>::max();
    for (const auto &span : llvm::enumerate(loopSpans)) {
      if (idx < span.value().begin || idx > span.value().end)
        continue;
      unsigned width = span.value().end - span.value().begin;
      if (width < bestWidth) {
        bestWidth = width;
        best = int(span.index());
      }
    }
    return best;
  };

  std::map<int, Buffer> byLogicId;
  std::map<int, std::set<int>> contextsOf; // logicId -> distinct loop contexts
  for (const Hazard &h : model.hazards) {
    if (h.isBarrier())
      continue;
    for (int c : h.bufferClusters) {
      Buffer &b = byLogicId[c];
      bool first = b.logicId < 0;
      b.logicId = c;
      // accessRange: union of the hazards' own intervals -- the tight fact.
      unsigned lo = h.interval().start, hi = h.interval().end;
      b.accessRange.start = first ? lo : std::min(b.accessRange.start, lo);
      b.accessRange.end = first ? hi : std::max(b.accessRange.end, hi);

      // l_bufid, UNION SPAN, mirroring BufidSyncIdAlloc::computeLifeIntervals:
      // an access inside a loop contributes that whole loop, one outside
      // contributes itself.
      for (unsigned idx : {lo, hi}) {
        int ctx = contextOf(idx);
        contextsOf[c].insert(ctx);
        unsigned s = idx, e = idx;
        if (ctx >= 0) {
          s = loopSpans[ctx].begin;
          e = loopSpans[ctx].end;
          b.isLoopCarried = true;
        }
        if (!b.bufidLoopDefined) {
          b.bufidLoop = {s, e};
          b.bufidLoopDefined = true;
        } else {
          b.bufidLoop.start = std::min(b.bufidLoop.start, s);
          b.bufidLoop.end = std::max(b.bufidLoop.end, e);
        }
      }
    }
  }

  // isWrittenBack -- a fact about the BUFFER: does it carry a mirror-direction
  // pair at all? Deliberately NOT `Hazard::hasReverseWar` re-read; see the
  // divergence note on the struct.
  for (auto &entry : byLogicId) {
    int c = entry.first;
    bool mirrored = false;
    for (const Hazard &a : model.hazards) {
      if (a.isBarrier() || !llvm::is_contained(a.bufferClusters, c))
        continue;
      for (const Hazard &b2 : model.hazards) {
        if (b2.isBarrier() || b2.index() == a.index())
          continue;
        if (!llvm::is_contained(b2.bufferClusters, c))
          continue;
        if (b2.srcPipe() == a.dstPipe() && b2.dstPipe() == a.srcPipe()) {
          mirrored = true;
          break;
        }
      }
      if (mirrored)
        break;
    }
    entry.second.isWrittenBack = mirrored;
  }

  for (auto &entry : byLogicId) {
    auto &b = entry.second;
    const auto &ctxs = contextsOf[entry.first];
    b.loopContextCount = unsigned(ctxs.size());
    b.spansLoopBoundary = ctxs.size() > 1;
    model.buffers.push_back(b);
  }

  return model;
}

//===----------------------------------------------------------------------===//
// Loop-carried compensation synthesis
//===----------------------------------------------------------------------===//

namespace {

// Where a synthesised head-set may sit, given the loop that carries its hazard.
//
// A head-set only grants the credit the in-body wait consumes on iteration 1, so it
// has no producer to follow and can move as early as its region allows. That matters
// for cycles rather than correctness: the consumer pipe cannot start until the credit
// is up, so a head-set parked behind unrelated compute on the setting pipe delays the
// consumer by exactly that compute.
//
// THE BOUND IS THE CARRYING LOOP'S OWN REGION, NOT THE FUNCTION. Head and tail must
// execute the same number of times: the head primes one credit and the tail drains the
// last iteration's set. Lifting the head out of an ENCLOSING loop while the tail stays
// inside it would prime once and drain per outer iteration, so the second outer
// iteration would find no credit and the in-body wait would never be satisfied. Moving
// within the region cannot change either count.
//
// Nothing may be crossed that touches the primed buffers, on any pipe: the credit says
// the consumer may overwrite them, which must not be granted while a region-local
// reader or writer is still ahead of it.
unsigned hoistAnchorFor(const SyncIRs &syncIR, const LoopInstanceElement &loopEl,
                        const SyncOperation &headSet) {
  // Region depth per element, with loop/branch markers sitting at the depth of the
  // construct they delimit and bodies one deeper.
  llvm::SmallVector<unsigned> depth(syncIR.size(), 0);
  unsigned d = 0;
  for (unsigned i = 0; i < syncIR.size(); ++i) {
    const InstanceElement *el = syncIR[i].get();
    bool isEnd = false;
    if (const auto *lp = llvm::dyn_cast<LoopInstanceElement>(el))
      isEnd = lp->getLoopKind() == KindOfLoop::LOOP_END;
    else if (const auto *br = llvm::dyn_cast<BranchInstanceElement>(el))
      isEnd = br->getBranchKind() == KindOfBranch::IF_END;
    if (isEnd && d > 0)
      --d;
    depth[i] = d;
    bool isBegin = false;
    if (const auto *lp = llvm::dyn_cast<LoopInstanceElement>(el))
      isBegin = lp->getLoopKind() == KindOfLoop::LOOP_BEGIN;
    else if (const auto *br = llvm::dyn_cast<BranchInstanceElement>(el))
      isBegin = br->getBranchKind() == KindOfBranch::IF_BEGIN;
    if (isBegin)
      ++d;
  }

  auto touchesPrimedBuffer = [&](const CompoundInstanceElement &cp) {
    for (const BaseMemInfo *primed : headSet.depMemInfos) {
      if (!primed)
        continue;
      for (const BaseMemInfo *other : cp.defVec)
        if (other && *other == *primed)
          return true;
      for (const BaseMemInfo *other : cp.useVec)
        if (other && *other == *primed)
          return true;
    }
    return false;
  };

  const unsigned begin = loopEl.beginId;
  if (begin >= syncIR.size())
    return begin;
  const unsigned regionDepth = depth[begin];
  unsigned anchor = begin;
  for (unsigned i = begin; i-- > 0;) {
    if (depth[i] < regionDepth)
      break; // left the carrying loop's region
    const auto *cp = llvm::dyn_cast<CompoundInstanceElement>(syncIR[i].get());
    if (cp && touchesPrimedBuffer(*cp))
      break;
    // Only a real op at region depth is a legal parking spot. A marker's pipeBefore
    // belongs to the construct it delimits, and a deeper element is inside a sibling.
    if (cp && depth[i] == regionDepth)
      anchor = i;
  }
  return anchor;
}

} // namespace

unsigned mlir::pto::unified::synthesizeLoopCompensation(
    SyncModel &model, SyncOperations &syncOps, SyncIRs &syncIR) {
  unsigned synthesised = 0;

  // ONE pass over the hazards, ONE pair created per loop-carried hazard, and no
  // retry/reallocation anywhere in this allocator. That is the structural
  // one-of-each guarantee: ONE pass over the hazards, ONE pair created per
  // loop-carried hazard, and no retry or reallocation anywhere in this
  // allocator.
  for (Hazard &h : model.hazards) {
    if (h.isBarrier())
      continue;
    SyncOperation *inSet = h.setOp();
    if (!inSet->GetForEndIndex().has_value())
      continue; // forward hazard: its 2 ops are already sufficient

    const auto *loopEl = llvm::dyn_cast<LoopInstanceElement>(
        syncIR[inSet->GetForEndIndex().value()].get());
    if (!loopEl)
      continue; // unresolved carrying loop -- the fe/synthesised count check
                // in the caller reports it rather than silently priming nothing

    // The head-set is a clone of the in-body SET placed at the carrying loop's
    // begin (it raises the flag the in-body WAIT consumes on iteration 1); the
    // tail-wait is its GetMatchSync mirror at the loop's end (it drains the last
    // iteration's set). Same direction as the in-body pair, by construction.
    //
    // Both are created WITHOUT an event id: the id arrives from ONE coloring
    // decision on this hazard, via Hazard::SetEventId, which stamps set + wait +
    // head + tail together. They are never colored separately.
    unsigned groupIndex = static_cast<unsigned>(syncOps.size());
    // `compensationOf` is what lets G2 recognise this pair as the other half of
    // `inSet`'s hazard rather than a competing one.
    //
    // DELIBERATELY NOT `SyncOperation::isCompensation`, even though that is what the
    // field is named for. That flag DRIVES CODEGEN: `resolveSyncInsertAnchor` re-anchors
    // a compensation op to its block terminator and will even synthesise an `scf.if`
    // else-region for it, and `shouldInsertBefore` forces insert-before
    // (`SyncCodegen`'s anchor resolution). This pass does not pass `isComp=true` --
    // full grep: zero callers on this branch and on main -- so those four reads are
    // dormant, and setting it here would activate placement logic no test covers, to
    // buy an oracle detail. `compensationOf` is inert by construction: only the
    // extractor and G2 read it.
    // Where the head-set will actually be emitted. Computed BEFORE the op is built,
    // because the SyncOperation records its own SyncIR index and the hazard's interval
    // has to agree with it -- see the interval extension below.
    const unsigned headAnchor = hoistAnchorFor(syncIR, *loopEl, *inSet);

    auto headSet = std::make_unique<SyncOperation>(
        inSet->GetType(), inSet->GetSrcPipe(), inSet->GetDstPipe(), groupIndex,
        headAnchor, inSet->GetForEndIndex());
    headSet->compensationOf = static_cast<int>(inSet->GetSyncIndex());
    headSet->depRootBuffers = inSet->depRootBuffers;
    headSet->depMemInfos = inSet->depMemInfos;
    headSet->eventIdNum = inSet->eventIdNum;
    headSet->syncCoreType = inSet->syncCoreType;
    headSet->SetMechanism(inSet->GetMechanism());

    auto tailWait = headSet->GetMatchSync(loopEl->endId);

    SyncOperation *headPtr = headSet.get();
    SyncOperation *tailPtr = tailWait.get();

    llvm::SmallVector<std::unique_ptr<SyncOperation>> group;
    group.emplace_back(std::move(headSet));
    group.emplace_back(std::move(tailWait));
    syncOps.emplace_back(std::move(group));

    // --- Placement into syncIR ----------------------------------------------
    // `SyncCodegen` emits by walking syncIR's pipeBefore/pipeAfter lists, NOT
    // `SyncOperations`. Owning the pair in `syncOps` is therefore not enough:
    // without these two inserts the synthesised head/tail would be SILENTLY
    // DROPPED at emission, reintroducing exactly the iteration-1 deadlock Step 1
    // fixed -- and invisibly, since the syncops-view gates would still be green.
    // Mirrors the incumbent's carrying-loop placement
    // (`SyncEventIdAllocation`'s own compensation handling).
    //
    // THE TAIL USES push_FRONT, NOT push_back, AND THAT IS LOAD-BEARING.
    // The tail-wait must precede any existing loop-end SET at this boundary, so
    // the loop tail anchor does not emit a new set before consuming the previous
    // iteration's carried event.
    // push_back would produce a correct-LOOKING pipe list that emits a BROKEN
    // sequence -- the emission-side analogue of the interval collision the
    // carrying-loop `endId + 1` fixed on the allocation side.
    syncIR[headAnchor]->pipeBefore.push_back(headPtr);
    syncIR[loopEl->endId]->pipeAfter.push_front(tailPtr);

    // The interval must COVER every op the hazard owns, for the same reason the
    // `endId + 1` above covers the tail-wait. Hoisting moves the head-set below the
    // interval's start, so the id has to be held from the anchor instead: otherwise
    // the colourer sees an earlier hazard as non-overlapping and reuses the id, while
    // in emission the hoisted head-set sits inside that hazard's live range. Two
    // head-sets that then land on one element with one id are also indistinguishable
    // to `MergeSyncList`, which drops the duplicate and leaves a wait with no set.
    if (headAnchor < h.interval().start)
      h.setInterval(Interval{headAnchor, h.interval().end});

    h.setCompensation(headPtr, tailPtr);
    ++synthesised;
  }

  return synthesised;
}

//===----------------------------------------------------------------------===//
// Event interval-coloring allocator
//===----------------------------------------------------------------------===//

BufferRouteResult mlir::pto::unified::routeBuffers(SyncModel &model) {
  BufferRouteResult result;
  result.buffersConsidered = unsigned(model.buffers.size());

  // Peak per-direction overlap, over ALL hazards in each direction -- the same
  // quantity colorEventIds' left-edge sweep will hit. Computed by a sweep over
  // interval endpoints rather than pairwise, so it is exact rather than a bound.
  std::map<DirKey, unsigned> peakOf;
  std::map<DirKey, unsigned> poolOf;
  {
    std::map<DirKey, llvm::SmallVector<std::pair<unsigned, int>>> events;
    for (const Hazard &h : model.hazards) {
      if (h.isBarrier())
        continue;
      DirKey d = dirKeyOf(h);
      // WEIGHTED by demand, matching what the colourer does: a multi-buffered hazard
      // holds d ids live across its interval, not one. Unweighted, this predicate
      // under-predicts peak occupancy by a factor of d and declines to route exactly
      // the buffers the colourer then overflows and spills to barriers.
      int w = int(h.demand());
      events[d].push_back({h.interval().start, +w});
      events[d].push_back({h.interval().end, -w});
      poolOf[d] = model.resources.eventPoolSize(h.srcPipe(), h.dstPipe());
    }
    for (auto &entry : events) {
      // Sort by position; at equal positions process the releases first, because
      // touching endpoints do not overlap (Interval::overlaps uses strict `<`).
      llvm::sort(entry.second, [](const std::pair<unsigned, int> &a,
                                  const std::pair<unsigned, int> &b) {
        if (a.first != b.first)
          return a.first < b.first;
        return a.second < b.second;
      });
      int live = 0;
      unsigned peak = 0;
      for (const auto &ev : entry.second) {
        live += ev.second;
        peak = std::max(peak, unsigned(std::max(live, 0)));
      }
      peakOf[entry.first] = peak;
    }
  }

  const bool haveCapacity = model.resources.bufidCapacity > 0;

  for (Buffer &b : model.buffers) {
    // The directions this buffer's hazards participate in.
    unsigned worstPeak = 0;
    unsigned tightestPool = 0;
    bool any = false;
    for (const Hazard &h : model.hazards) {
      if (h.isBarrier() || !llvm::is_contained(h.bufferClusters, b.logicId))
        continue;
      DirKey d = dirKeyOf(h);
      unsigned peak = peakOf.count(d) ? peakOf[d] : 0;
      unsigned pool = poolOf.count(d) ? poolOf[d] : 0;
      if (!any || peak > worstPeak) {
        worstPeak = peak;
        tightestPool = pool;
      }
      any = true;
    }
    b.predictedPeakOverlap = worstPeak;
    b.smallestPool = tightestPool;

    if (!haveCapacity) {
      ++result.skippedNoCapacity;
      continue;
    }
    // STRICT: omega == pool fits exactly, so routing it would be pure waste.
    if (!(worstPeak > tightestPool)) {
      ++result.skippedNoOverflow;
      continue;
    }
    if (!b.isWrittenBack) {
      ++result.skippedNotWrittenBack;
      continue;
    }
    b.routedToBufid = true;
    ++result.routed;
  }

  // ALL-OR-NOTHING, asserted rather than assumed. A hazard touching both a routed
  // and an unrouted buffer cannot be wholly on one mechanism -- that is the split
  // that hangs, so it is counted and must be 0.
  for (const Hazard &h : model.hazards) {
    if (h.isBarrier() || h.bufferClusters.empty())
      continue;
    bool anyRouted = false, anyUnrouted = false;
    for (const Buffer &b : model.buffers) {
      if (!llvm::is_contained(h.bufferClusters, b.logicId))
        continue;
      if (b.routedToBufid)
        anyRouted = true;
      else
        anyUnrouted = true;
    }
    if (anyRouted && anyUnrouted)
      ++result.splitHazards;
    if (anyRouted && !anyUnrouted)
      ++result.hazardsCovered;
  }

  return result;
}

unsigned mlir::pto::unified::seedHiddenMacroEvents(SyncModel &model,
                                                   const SyncIRs &syncIR) {
  for (size_t i = 0; i < syncIR.size(); ++i) {
    auto *first = dyn_cast<pto::CompoundInstanceElement>(syncIR[i].get());
    if (!first || first->macroOpInstanceId != 0 || !first->elementOp)
      continue;
    auto macro = pto::getSyncMacroModel(first->elementOp);
    if (!macro || macro->hiddenEvents.empty())
      continue;

    unsigned end = first->GetIndex() + 1;
    for (size_t j = i + 1; j < syncIR.size(); ++j) {
      auto *other = dyn_cast<pto::CompoundInstanceElement>(syncIR[j].get());
      if (!other || other->elementOp != first->elementOp)
        continue;
      end = other->GetIndex();
    }
    unsigned begin = first->GetIndex();
    if (begin > 0)
      --begin;
    if (end + 1 < syncIR.size())
      ++end;
    if (begin >= end)
      continue;

    for (const auto &hidden : macro->hiddenEvents)
      for (unsigned id : hidden.eventIds)
        model.hiddenReservations.push_back(
            {static_cast<int>(hidden.srcPipe), static_cast<int>(hidden.dstPipe),
             static_cast<int>(id), Interval{begin, end},
             first->elementOp->getName().getStringRef()});
  }
  return model.hiddenReservations.size();
}

EventColorResult mlir::pto::unified::colorEventIds(SyncModel &model) {
  EventColorResult result;

  // Non-barrier hazards, ordered for a per-direction left-edge scan. The order
  // is (direction, interval start, hazard index) -- fully deterministic, and it
  // groups each direction into a contiguous run below.
  llvm::SmallVector<Hazard *> haz;
  for (Hazard &h : model.hazards) {
    if (h.isBarrier())
      continue;
    // Step 2 routed this hazard's buffer to the token protocol. It must NOT take
    // an event id -- a buffer with some hazards on events and some on get/rls is
    // a counter cycle missing a step, i.e. a hang. It is skipped, NOT counted as
    // overflow: overflow means "no id was available", and this is "no id was
    // wanted".
    if (h.mechanism() == SyncOperation::MECHANISM::BUFID) {
      ++result.skippedRouted;
      continue;
    }
    haz.push_back(&h);
  }
  llvm::stable_sort(haz, [](const Hazard *a, const Hazard *b) {
    DirKey da = dirKeyOf(*a), db = dirKeyOf(*b);
    if (da != db)
      return da < db;
    if (a->interval().start != b->interval().start)
      return a->interval().start < b->interval().start;
    return a->index() < b->index();
  });

  // Color one direction at a time; each is an independent interval graph, so a
  // left-edge sweep with the lowest free id attains its chromatic number.
  for (size_t i = 0; i < haz.size();) {
    DirKey dir = dirKeyOf(*haz[i]);
    unsigned pool =
        model.resources.eventPoolSize(haz[i]->srcPipe(), haz[i]->dstPipe());
    llvm::SmallVector<unsigned, 8> endOf(pool, 0);
    llvm::SmallVector<bool, 8> inUse(pool, false);
    llvm::SmallVector<bool, 8> everUsed(pool, false);

    for (; i < haz.size() && dirKeyOf(*haz[i]) == dir; ++i) {
      Hazard *h = haz[i];
      unsigned start = h->interval().start;

      // Left-edge release: an id whose interval ended at or before this start is
      // free again. Touching endpoints do not overlap (Interval::overlaps uses
      // strict `<`), so the test is `<=`.
      for (unsigned id = 0; id < pool; ++id)
        if (inUse[id] && endOf[id] <= start)
          inUse[id] = false;

      // The d LOWEST free ids -> optimal weighted coloring, deterministic.
      // d == 1 for an ordinary hazard, so this is the old single-id scan with the
      // loop bound generalised; d > 1 only for a multi-buffered (rotating) pair.
      //
      // ALL-OR-NOTHING: fewer than d ids is not a degraded assignment, it is an
      // out-of-bounds read -- `CreateSetWaitOpForMultiBuffer` indexes eventIds[i]
      // for i in [0, slotCount). So a short pool falls through to the spill.
      unsigned d = h->demand();
      // An id is unavailable if a macro's library implementation holds it over a
      // span overlapping this hazard. Checked per hazard rather than seeded into
      // `inUse`, because a reservation is an INTERVAL: seeding would also block
      // hazards that start before the call, which is conservative but wrong.
      auto reservedFor = [&](unsigned id) {
        for (const HiddenReservation &r : model.hiddenReservations)
          if (r.id == static_cast<int>(id) && r.srcPipe == dir.first &&
              r.dstPipe == dir.second && r.span.overlaps(h->interval()))
            return true;
        return false;
      };
      // DO NOT HAND BACK THE ID THAT WAS FREED MOST RECENTLY, while another is
      // free. Taking the lowest free id minimises the id COUNT, which is the wrong
      // objective: an event flag carries one outstanding signal, so two hazards
      // sharing an id cannot both be in flight, and the producing pipe can never
      // get more than one signal ahead of the consuming one. Alternating between
      // two ids restores that one step of run-ahead.
      //
      // TWO ids is the whole win. The run-ahead an event flag permits is one signal
      // deep, so a third id buys no further overlap and spends pool headroom that an
      // A3 overflow has no way to absorb. Hence: avoid the most recently freed id,
      // do not cycle through the pool.
      //
      // Peak usage is bounded by min(pool, omega + 1): the avoidance only applies
      // when a second id is free, so it can never turn an assignment that fit into
      // a spill.
      int avoid = -1;
      unsigned avoidEnd = 0;
      for (unsigned id = 0; id < pool; ++id)
        if (!inUse[id] && everUsed[id] && !reservedFor(id) &&
            (avoid < 0 || endOf[id] > avoidEnd)) {
          avoidEnd = endOf[id];
          avoid = static_cast<int>(id);
        }

      auto collect = [&](int skip) {
        llvm::SmallVector<int, 4> out;
        for (unsigned id = 0; id < pool && out.size() < d; ++id)
          if (!inUse[id] && !reservedFor(id) && static_cast<int>(id) != skip)
            out.push_back(static_cast<int>(id));
        return out;
      };
      llvm::SmallVector<int, 4> chosenIds = collect(avoid);
      if (chosenIds.size() < d)
        chosenIds = collect(-1); // not enough without it; the spread is a preference
      int chosen = chosenIds.size() == d ? chosenIds.front() : -1;
      if (chosen < 0) {
        // No id free at this instant in this direction. SPILL TO A BARRIER.
        //
        // This used to `continue` with no id assigned -- which kept G3 true by
        // construction but left codegen emitting the pair anyway, carrying ids
        // nobody handed out (`id=0` colliding with the legitimate holder, or the
        // unset sentinel 2147483647). "No id" was never a resting state: a hazard
        // must end on SOME mechanism, and barrier is the unbounded class that
        // makes the allocator total.
        //
        // That totality is what this path assumes and never re-checks: there is no
        // "spill failed" branch below it. Assert the assumption where it is relied
        // on, so a bounded barrier class cannot be introduced without this stopping
        // compiling until a failure mode is written here.
        static_assert(ResourceModel::barrierIsUnbounded,
                      "the spill path has no failure branch, so the barrier class "
                      "must be unbounded and always available");
        ++result.overflow;
        ++result.spilledToBarrier;
        h->SpillToBarrier();
        continue;
      }

      for (int id : chosenIds) {
        inUse[id] = true;
        endOf[id] = h->interval().end;
        if (everUsed[static_cast<unsigned>(id)])
          ++result.reused;
        everUsed[static_cast<unsigned>(id)] = true;
      }
      result.idsAssigned += d;
      if (d > 1)
        ++result.rotating;

      // Backward hazards need no special case here any more. Step 1 gave each of
      // them a synthesised head/tail pair and a real, forward CARRYING-LOOP
      // interval, so they colour exactly like any other interval -- and the
      // "burn" stopgap (reserve an id across a loop with no instruction priming
      // it) is gone, which is the failure this ordering exists to avoid.
      // Stamps set + wait + head + tail with the SAME id list, because
      // a compensation pair left unstamped emits with the wrong arity.
      h->SetEventIds(chosenIds);
      ++result.assigned;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

void mlir::pto::unified::printSyncModel(llvm::raw_ostream &os,
                                        llvm::StringRef funcName,
                                        const SyncModel &model) {
  os << "[unified-model] func=" << funcName
     << " hazards=" << model.hazards.size()
     << " K=" << model.resources.bufidCapacity << "\n";

  for (const Hazard &h : model.hazards) {
    Alpha a = Alpha::forHazard(h);
    os << "  h#" << h.index() << " "
       << (h.isBarrier() ? "barrier" : "pair")
       << " " << oracle::pipelineTypeName(h.srcPipe()) << "->"
       << oracle::pipelineTypeName(h.dstPipe())
       << " iv=[" << h.interval().start << "," << h.interval().end << ")"
       << " pool=" << model.resources.eventPoolSize(h.srcPipe(), h.dstPipe())
       << " mech=" << SyncOperation::MechanismName(h.mechanism())
       << " clusters=[";
    for (size_t i = 0; i < h.bufferClusters.size(); ++i) {
      if (i)
        os << ",";
      os << h.bufferClusters[i];
    }
    os << "]"
       << " revwar=" << (h.hasReverseWar ? "yes" : "no")
       << " alpha(e/b/bar)=" << a.event << "/" << a.bufid << "/" << a.barrier
       << " event_id=";
    // ALL ids, not just the first. A rotating hazard holds d of them, and
    // printing `front()` alone made the report claim a single id for a hazard
    // that had several -- in the very line used as evidence elsewhere.
    if (h.hasEventId()) {
      const auto &ids = h.setOp()->eventIds;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i)
          os << "/";
        os << ids[i];
      }
    } else {
      os << "-";
    }
    os << "\n";
  }

  // Buffers. TOTALITY IS PRINTED, on the `unmappable=0`
  // discipline: `lbufid=D/N` says how many of the N buffers got a resolved
  // reservation window. If D != N the difference is a real gap, not a default.
  // `span_boundary` counts buffers with accesses in more than one loop
  // context, so no single carrying loop. They are ROUTED CONSERVATIVELY (union
  // span, matching the shipping pass), never refused; see the struct's note.
  // `revwar_divergence` counts buffers where the per-BUFFER `isWrittenBack`
  // disagrees with at least one of its hazards' per-HAZARD `hasReverseWar` --
  // the two predicates are genuinely different questions and this makes the
  // difference visible instead of letting one silently stand in for the other.
  if (!model.buffers.empty()) {
    unsigned wb = 0, lc = 0, sb = 0, defined = 0, divergence = 0;
    for (const Buffer &b : model.buffers) {
      if (b.isWrittenBack)
        ++wb;
      if (b.isLoopCarried)
        ++lc;
      if (b.spansLoopBoundary)
        ++sb;
      if (b.bufidLoopDefined)
        ++defined;
      for (const Hazard &h : model.hazards) {
        if (h.isBarrier() || !llvm::is_contained(h.bufferClusters, b.logicId))
          continue;
        if (h.hasReverseWar != b.isWrittenBack) {
          ++divergence;
          break;
        }
      }
    }
    os << "  buffers=" << model.buffers.size() << " written_back=" << wb
       << " loop_carried=" << lc << " span_boundary=" << sb
       << " lbufid=" << defined << "/" << model.buffers.size()
       << " revwar_divergence=" << divergence << "\n";
    for (const Buffer &b : model.buffers)
      os << "    b#" << b.logicId << " access=[" << b.accessRange.start << ","
         << b.accessRange.end << ") lbufid="
         << (b.bufidLoopDefined ? "[" + std::to_string(b.bufidLoop.start) + "," +
                                      std::to_string(b.bufidLoop.end) + "]"
                                : std::string("UNRESOLVED"))
         << " wb=" << (b.isWrittenBack ? "yes" : "no")
         << " loop=" << (b.isLoopCarried ? "yes" : "no")
         << " ctx=" << b.loopContextCount
         << (b.spansLoopBoundary ? " SPANS-BOUNDARY" : "") << "\n";
  }
}
