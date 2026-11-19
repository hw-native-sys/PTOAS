// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- UnifiedSyncModel.cpp - Unified sync allocator data model -----------===//
//
// Constructs hazard and buffer facts, then routes buffers and colors event IDs.
// Construction helpers do not assign resources; the allocation stages consume
// their facts only after loop compensation has settled operation placement.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/InsertSync/UnifiedSyncModel.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/InsertSync/SyncOracleExtract.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include <limits>
#include <map>
#include <set>

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
  if (reserved >= kTotalEventIdNum) {
    return 0;
  }
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

DirKey dirKeyOf(const Hazard &h) { return {static_cast<int>(h.srcPipe()), static_cast<int>(h.dstPipe())}; }

static void joinHazardClusters(Hazard &h, const SyncOperation *setOp, const MemInfoToClusters &memInfoToClusters) {
  // --- 3. Join the hazard to its buffer-id cluster(s) ---------------------
  // Resolved by the caller via MemAlias -- the relation BufidSyncAnalysis
  // itself uses to decide two tiles are the same memory. See MemInfoToClusters
  // for the three plausible-looking keys that are all wrong.
  for (const BaseMemInfo *info : setOp->depMemInfos) {
    if (!info) {
      continue;
    }
    auto it = memInfoToClusters.find(info);
    if (it == memInfoToClusters.end()) {
      continue;
    }
    for (int c : it->second) {
      if (!llvm::is_contained(h.bufferClusters, c)) {
        h.bufferClusters.push_back(c);
      }
    }
  }
  llvm::sort(h.bufferClusters); // determinism
}

static void setHazardFacts(Hazard &h, const SyncOperation *setOp, const SyncOperation *waitOp, const SyncIRs &syncIR,
                           const MemInfoToClusters &memInfoToClusters) {
  // --- 2. Interval, in SyncIR-index space --------------------------------
  // Forward hazard: the tight [set, wait) range. A barrier is a point.
  //
  // Loop-carried hazard: the CARRYING loop, not the raw [set, wait), which
  // inverts because the wait is satisfied by the previous iteration.
  //
  // `endId + 1` is load-bearing. `Interval` is half-open, so [beginId, endId)
  // would exclude position endId, which is where this hazard's own tail-wait
  // sits; the colourer would then see a hazard starting at endId as disjoint
  // and reuse the id while the two collide on one SyncIR element.
  unsigned start = setOp->GetSyncIRIndex();
  unsigned end = waitOp ? waitOp->GetSyncIRIndex() : start;
  if (setOp->GetForEndIndex().has_value()) {
    const auto *loopEl = llvm::dyn_cast<LoopInstanceElement>(syncIR[setOp->GetForEndIndex().value()].get());
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

  joinHazardClusters(h, setOp, memInfoToClusters);
}

static void buildHazards(SyncModel &model, const SyncOperations &syncOps, const MemInfoToClusters &memInfoToClusters,
                         const SyncIRs &syncIR) {
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
      if (!s || s->uselessSync) {
        continue;
      }
      if (s->isSyncSetType() || s->isBarrierType()) {
        // A barrier group has no wait side; it occupies the `set` slot so that
        // `setOp` is never null and callers need no special case to read a pipe.
        if (!setOp) {
          setOp = s;
        }
      } else if (s->isSyncWaitType()) {
        if (!waitOp) {
          waitOp = s;
        }
      }
    }
    if (!setOp) {
      continue; // fully-pruned group (every member uselessSync)
    }

    Hazard h(static_cast<unsigned>(group.index()), setOp, waitOp);

    setHazardFacts(h, setOp, waitOp, syncIR, memInfoToClusters);

    model.hazards.push_back(std::move(h));
  }
}

static void markReverseHazards(SyncModel &model) {
  // --- 4. Is there a GENUINE reverse hazard on this buffer? ----------------
  //
  // "Reverse" = a mirror-direction hazard SHARING A CLUSTER. The cluster is the
  // unit because one token covers one aliasing clique. Matching on
  // `depRootBuffers` -- the root allocation -- would call two hazards on different
  // tiles reverse partners merely for sharing a root, and that false positive is
  // the expensive direction: it prices the token low and routes a forward-only
  // hazard onto it.
  for (Hazard &h : model.hazards) {
    if (h.isBarrier() || h.bufferClusters.empty()) {
      continue;
    }
    for (const Hazard &other : model.hazards) {
      if (other.index() == h.index() || other.isBarrier()) {
        continue;
      }
      if (other.srcPipe() != h.dstPipe() || other.dstPipe() != h.srcPipe()) {
        continue; // not the mirror direction
      }
      for (int c : other.bufferClusters) {
        if (llvm::is_contained(h.bufferClusters, c)) {
          h.hasReverseWar = true;
          break;
        }
      }
      if (h.hasReverseWar) {
        break;
      }
    }
  }
}

static bool encodablePipe(PipelineType p) {
  switch (p) {
  case PipelineType::PIPE_MTE1:
  case PipelineType::PIPE_MTE2:
  case PipelineType::PIPE_MTE3:
  case PipelineType::PIPE_FIX:
  case PipelineType::PIPE_V:
  case PipelineType::PIPE_M:
    return true;
  default:
    return false;
  }
}

static void markTokenExpressible(SyncModel &model) {
  // --- 4b. Is a token able to express this hazard at all? ------------------
  //
  // Mirrors the shapes `BufidSyncCodegen` and `BufidSyncAnalysis` exclude. Kept as a
  // hazard fact rather than checked at the routing site so the reason a buffer was
  // refused is visible in the model report.
  //
  // The encodable-pipe set is the one `BufidSyncCodegen::mapPipelineToSyncOpType`
  // names; a pipe outside it fails the compilation rather than degrading, so it must
  // be refused here instead.
  for (Hazard &h : model.hazards) {
    if (h.isBarrier() || h.bufferClusters.empty()) {
      continue;
    }
    if (h.srcPipe() == h.dstPipe()) {
      continue;
    }
    if (!encodablePipe(h.srcPipe()) || !encodablePipe(h.dstPipe())) {
      continue;
    }
    bool scopesOk = true;
    std::optional<pto::AddressSpace> scope;
    for (const BaseMemInfo *mi : h.setOp()->depMemInfos) {
      if (!mi) {
        continue;
      }
      if (mi->scope == pto::AddressSpace::GM) {
        scopesOk = false;
        break;
      }
      if (!scope) {
        scope = mi->scope;
      } else if (*scope != mi->scope) {
        scopesOk = false;
        break;
      }
    }
    if (scopesOk && scope) {
      h.tokenExpressible = true;
    }
  }
}

static pto::AllocTileOp addrAllocOf(const BaseMemInfo *mi) {
  if (!mi || !mi->rootBuffer) {
    return nullptr;
  }
  auto alloc = mi->rootBuffer.getDefiningOp<pto::AllocTileOp>();
  if (!alloc || !alloc.getAddr()) {
    return nullptr;
  }
  return alloc;
}

static std::optional<std::pair<pto::AddressSpace, uint64_t>> addrSlotOf(pto::AllocTileOp alloc) {
  mlir::IntegerAttr c;
  if (!mlir::matchPattern(alloc.getAddr(), mlir::m_Constant(&c))) {
    return std::nullopt;
  }
  auto tileType = dyn_cast<pto::TileBufType>(alloc.getResult().getType());
  if (!tileType) {
    return std::nullopt;
  }
  auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(tileType.getMemorySpace());
  if (!asAttr) {
    return std::nullopt;
  }
  return std::make_pair(asAttr.getAddressSpace(), static_cast<uint64_t>(c.getInt()));
}

static void markHazardAddressSharing(Hazard &h) {
  for (const BaseMemInfo *a : h.setOp()->depMemInfos) {
    pto::AllocTileOp allocA = addrAllocOf(a);
    if (!allocA) {
      continue;
    }
    auto slotA = addrSlotOf(allocA);
    if (!slotA) {
      continue;
    }
    for (const BaseMemInfo *b : h.waitOp()->depMemInfos) {
      pto::AllocTileOp allocB = addrAllocOf(b);
      if (!allocB || allocB == allocA) {
        continue;
      }
      auto slotB = addrSlotOf(allocB);
      if (!slotB || *slotA != *slotB) {
        continue;
      }

      // `depMemInfos` is unordered, so which endpoint matched on which side
      // means nothing. Order by program position where that is defined.
      Operation *first = allocA.getOperation();
      Operation *second = allocB.getOperation();
      if (first->getBlock() == second->getBlock() && second->isBeforeInBlock(first)) {
        std::swap(first, second);
      }

      h.addressSharing = allocA.getResult().getType() == allocB.getResult().getType()
                             ? Hazard::AddressSharing::Reuse
                             : Hazard::AddressSharing::AliasView;
      h.sharedScope = slotA->first;
      h.sharedAddress = slotA->second;
      h.sharedAllocFirst = first;
      h.sharedAllocSecond = second;
      return;
    }
  }
}

static void markAddressSharing(SyncModel &model) {
  // --- 4c. Address sharing between the endpoints' allocations ---------------
  //
  // Derived here so the model carries it with everything else it knows about a
  // hazard. Read by reporting only -- see `Hazard::addressSharing`.
  //
  // Covers every non-barrier hazard, same-pipe included.
  //
  // The address is read straight off the `pto.alloc_tile` operand. `MemAlias` reaches the
  // same addresses by a different route -- `BaseMemInfo::baseAddresses` guarded by
  // `hasKnownPhysicalAddresses`, which PTOIRTranslator sets for a local alloc_tile whose
  // addr is a non-negative constant -- so the two agree on level3. Reading the operand is
  // a directness choice, not a workaround for a missing flag.
  for (Hazard &h : model.hazards) {
    if (h.isBarrier()) {
      continue;
    }
    markHazardAddressSharing(h);
  }
}

class BufferModelBuilder {
public:
  explicit BufferModelBuilder(SyncModel &model) : model(model) {}
  void build(const SyncIRs &syncIR) {
    collectLoops(syncIR);
    accumulateHazards();
    markWrittenBack();
    finishBuffers();
  }

private:
  struct LoopSpanRec {
    unsigned begin;
    unsigned end;
  };
  SyncModel &model;
  llvm::SmallVector<LoopSpanRec> loopSpans;
  std::map<int, Buffer> byLogicId;
  std::map<int, std::set<int>> contextsOf;
  void collectLoops(const SyncIRs &syncIR) {
    for (const auto &element : syncIR) {
      if (auto *loop = dyn_cast_or_null<pto::LoopInstanceElement>(element.get())) {
        if (loop->getLoopKind() == pto::KindOfLoop::LOOP_BEGIN) {
          loopSpans.push_back({loop->beginId, loop->endId});
        }
      }
    }
  }

  int contextOf(unsigned idx) const {
    int best = -1;
    unsigned bestWidth = std::numeric_limits<unsigned>::max();
    for (const auto &span : llvm::enumerate(loopSpans)) {
      if (idx < span.value().begin || idx > span.value().end) {
        continue;
      }
      unsigned width = span.value().end - span.value().begin;
      if (width < bestWidth) {
        bestWidth = width;
        best = int(span.index());
      }
    }
    return best;
  }

  void extendBufferLoop(Buffer &b, int c, unsigned lo, unsigned hi) {
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

  void accumulateHazards() {
    for (const Hazard &h : model.hazards) {
      if (h.isBarrier()) {
        continue;
      }
      for (int c : h.bufferClusters) {
        Buffer &b = byLogicId[c];
        bool first = b.logicId < 0;
        b.logicId = c;
        // accessRange: union of the hazards' own intervals -- the tight fact.
        unsigned lo = h.interval().start, hi = h.interval().end;
        b.accessRange.start = first ? lo : std::min(b.accessRange.start, lo);
        b.accessRange.end = first ? hi : std::max(b.accessRange.end, hi);

        extendBufferLoop(b, c, lo, hi);
      }
    }
  }

  void markWrittenBack() {
    // isWrittenBack -- a fact about the BUFFER: does it carry a mirror-direction
    // pair at all? Deliberately NOT `Hazard::hasReverseWar` re-read; see the
    // divergence note on the struct.
    for (auto &entry : byLogicId) {
      int c = entry.first;
      bool mirrored = false;
      for (const Hazard &a : model.hazards) {
        if (a.isBarrier() || !llvm::is_contained(a.bufferClusters, c)) {
          continue;
        }
        for (const Hazard &b2 : model.hazards) {
          if (b2.isBarrier() || b2.index() == a.index()) {
            continue;
          }
          if (!llvm::is_contained(b2.bufferClusters, c)) {
            continue;
          }
          if (b2.srcPipe() == a.dstPipe() && b2.dstPipe() == a.srcPipe()) {
            mirrored = true;
            break;
          }
        }
        if (mirrored) {
          break;
        }
      }
      entry.second.isWrittenBack = mirrored;
    }
  }

  void finishBuffers() {
    for (auto &entry : byLogicId) {
      auto &b = entry.second;
      const auto &ctxs = contextsOf[entry.first];
      b.loopContextCount = unsigned(ctxs.size());
      b.spansLoopBoundary = ctxs.size() > 1;
      model.buffers.push_back(b);
    }
  }
};

static void buildBuffers(SyncModel &model, const SyncIRs &syncIR) { BufferModelBuilder(model).build(syncIR); }

} // namespace

SyncModel mlir::pto::unified::buildSyncModel(const SyncOperations &syncOps, unsigned bufidCapacity,
                                             const MemInfoToClusters &memInfoToClusters, const SyncIRs &syncIR) {
  SyncModel model;
  model.resources.bufidCapacity = bufidCapacity;

  buildHazards(model, syncOps, memInfoToClusters, syncIR);

  markReverseHazards(model);

  markTokenExpressible(model);

  markAddressSharing(model);

  buildBuffers(model, syncIR);

  return model;
}

//===----------------------------------------------------------------------===//
// Loop-carried compensation synthesis
//===----------------------------------------------------------------------===//

namespace {

llvm::SmallVector<unsigned> regionDepths(const SyncIRs &syncIR) {
  // Region depth per element, with loop/branch markers sitting at the depth of the
  // construct they delimit and bodies one deeper.
  llvm::SmallVector<unsigned> depth(syncIR.size(), 0);
  unsigned d = 0;
  for (unsigned i = 0; i < syncIR.size(); ++i) {
    const InstanceElement *el = syncIR[i].get();
    bool isEnd = false;
    if (const auto *lp = llvm::dyn_cast<LoopInstanceElement>(el)) {
      isEnd = lp->getLoopKind() == KindOfLoop::LOOP_END;
    } else if (const auto *br = llvm::dyn_cast<BranchInstanceElement>(el)) {
      isEnd = br->getBranchKind() == KindOfBranch::IF_END;
    }
    if (isEnd && d > 0) {
      --d;
    }
    depth[i] = d;
    bool isBegin = false;
    if (const auto *lp = llvm::dyn_cast<LoopInstanceElement>(el)) {
      isBegin = lp->getLoopKind() == KindOfLoop::LOOP_BEGIN;
    } else if (const auto *br = llvm::dyn_cast<BranchInstanceElement>(el)) {
      isBegin = br->getBranchKind() == KindOfBranch::IF_BEGIN;
    }
    if (isBegin) {
      ++d;
    }
  }

  return depth;
}

bool touchesPrimedBuffer(const CompoundInstanceElement &cp, const SyncOperation &headSet) {
  for (const BaseMemInfo *primed : headSet.depMemInfos) {
    if (!primed) {
      continue;
    }
    for (const BaseMemInfo *other : cp.defVec) {
      if (other && *other == *primed) {
        return true;
      }
    }
    for (const BaseMemInfo *other : cp.useVec) {
      if (other && *other == *primed) {
        return true;
      }
    }
  }
  return false;
}

// Where a synthesised head-set may sit, given the loop that carries its hazard.
// A head-set has no producer to follow, so it can move as early as its region
// allows. This is a cycle concern, not correctness: the consumer pipe cannot start
// until the credit is up, so a head-set parked behind unrelated compute on the
// setting pipe delays the consumer by that compute.
// THE BOUND IS THE CARRYING LOOP'S OWN REGION, NOT THE FUNCTION. Head and tail must
// execute the same number of times. Lifting the head out of an ENCLOSING loop while
// the tail stays inside would prime once and drain per outer iteration, so the
// second outer iteration finds no credit and the in-body wait never completes.
// Nothing that touches the primed buffers may be crossed, on any pipe: the credit
// permits the consumer to overwrite them, which must not be granted while a
// region-local reader or writer is still ahead of it.
unsigned hoistAnchorFor(const SyncIRs &syncIR, const LoopInstanceElement &loopEl, const SyncOperation &headSet) {
  auto depth = regionDepths(syncIR);

  const unsigned begin = loopEl.beginId;
  if (begin >= syncIR.size()) {
    return begin;
  }
  const unsigned regionDepth = depth[begin];
  unsigned anchor = begin;
  for (unsigned i = begin; i-- > 0;) {
    if (depth[i] < regionDepth) {
      break; // left the carrying loop's region
    }
    const auto *cp = llvm::dyn_cast<CompoundInstanceElement>(syncIR[i].get());
    if (cp && touchesPrimedBuffer(*cp, headSet)) {
      break;
    }
    // Only a real op at region depth is a legal parking spot. A marker's pipeBefore
    // belongs to the construct it delimits, and a deeper element is inside a sibling.
    if (cp && depth[i] == regionDepth) {
      anchor = i;
    }
  }
  return anchor;
}

} // namespace

unsigned mlir::pto::unified::synthesizeLoopCompensation(SyncModel &model, SyncOperations &syncOps, SyncIRs &syncIR) {
  unsigned synthesised = 0;

  // One pass, one pair per loop-carried hazard, no retry or reallocation.
  for (Hazard &h : model.hazards) {
    if (h.isBarrier()) {
      continue;
    }
    SyncOperation *inSet = h.setOp();
    if (!inSet->GetForEndIndex().has_value()) {
      continue; // forward hazard: its 2 ops are already sufficient
    }

    const auto *loopEl = llvm::dyn_cast<LoopInstanceElement>(syncIR[inSet->GetForEndIndex().value()].get());
    if (!loopEl) {
      continue; // unresolved carrying loop -- the fe/synthesised count check
                // in the caller reports it rather than silently priming nothing
    }

    // The head-set clones the in-body SET at the carrying loop's begin, raising the
    // flag the in-body WAIT consumes on iteration 1; the tail-wait is its
    // GetMatchSync mirror at the loop's end, draining the last iteration's set.
    // Both are created WITHOUT an event id -- one coloring decision stamps set,
    // wait, head and tail together, and they are never coloured separately.
    unsigned groupIndex = static_cast<unsigned>(syncOps.size());
    // `compensationOf` is what lets G2 recognise this pair as the other half of
    // `inSet`'s hazard rather than a competing one.
    //
    // DELIBERATELY NOT `SyncOperation::isCompensation`. That flag drives codegen:
    // `resolveSyncInsertAnchor` re-anchors a compensation op to its block
    // terminator and can synthesise an `scf.if` else-region for it, and
    // `shouldInsertBefore` forces insert-before. Nothing constructs a
    // SyncOperation with `isComp=true`, so those reads are dormant and setting it
    // here would activate placement logic no test covers. `compensationOf` is
    // inert: only the extractor and G2 read it.
    //
    // The anchor is computed BEFORE the op is built, because a SyncOperation
    // records its own SyncIR index and the hazard's interval must agree with it.
    const unsigned headAnchor = hoistAnchorFor(syncIR, *loopEl, *inSet);

    auto headSet = std::make_unique<SyncOperation>(inSet->GetType(), inSet->GetSrcPipe(), inSet->GetDstPipe(),
                                                   groupIndex, headAnchor, inSet->GetForEndIndex());
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
    // `SyncCodegen` emits by walking syncIR's pipeBefore/pipeAfter lists, not
    // `SyncOperations`, so owning the pair in `syncOps` is not enough: without
    // these inserts the head/tail are dropped silently at emission -- and
    // invisibly, since the syncops-view gates would still be green. Mirrors
    // `SyncEventIdAllocation`'s own carrying-loop placement.
    //
    // THE TAIL USES push_FRONT, AND THAT IS LOAD-BEARING. The tail-wait must
    // precede any existing loop-end SET at this boundary, so the loop tail anchor
    // does not emit a new set before consuming the previous iteration's carried
    // event. push_back yields a correct-looking pipe list that emits a broken
    // sequence.
    syncIR[headAnchor]->pipeBefore.push_back(headPtr);
    syncIR[loopEl->endId]->pipeAfter.push_front(tailPtr);

    // The interval must COVER every op the hazard owns. Hoisting moves the head-set
    // below the interval's start, so the id is held from the anchor instead;
    // otherwise the colourer reuses the id while the hoisted head-set sits inside
    // the other hazard's live range. Two head-sets on one element with one id are
    // also indistinguishable to `MergeSyncList`, which drops the duplicate and
    // leaves a wait with no set.
    if (headAnchor < h.interval().start) {
      h.setInterval(Interval{headAnchor, h.interval().end});
    }

    h.setCompensation(headPtr, tailPtr);
    ++synthesised;
  }

  return synthesised;
}

//===----------------------------------------------------------------------===//
// Event interval-coloring allocator
//===----------------------------------------------------------------------===//

namespace {
static void directionPeaks(const SyncModel &model, std::map<DirKey, unsigned> &peakOf,
                           std::map<DirKey, unsigned> &poolOf) {
  // Peak per-direction overlap, over ALL hazards in each direction -- the same
  // quantity colorEventIds' left-edge sweep will hit. Computed by a sweep over
  // interval endpoints rather than pairwise, so it is exact rather than a bound.
  std::map<DirKey, llvm::SmallVector<std::pair<unsigned, int>>> events;
  for (const Hazard &h : model.hazards) {
    if (h.isBarrier()) {
      continue;
    }
    DirKey d = dirKeyOf(h);
    // WEIGHTED by demand, matching the colourer: a multi-buffered hazard holds d
    // ids live, not one. Unweighted, this under-predicts peak occupancy by a
    // factor of d and declines to route exactly the buffers that then overflow.
    int w = int(h.demand());
    events[d].push_back({h.interval().start, +w});
    events[d].push_back({h.interval().end, -w});
    poolOf[d] = model.resources.eventPoolSize(h.srcPipe(), h.dstPipe());
  }
  for (auto &entry : events) {
    // Sort by position; at equal positions process the releases first, because
    // touching endpoints do not overlap (Interval::overlaps uses strict `<`).
    llvm::sort(entry.second, [](const std::pair<unsigned, int> &a, const std::pair<unsigned, int> &b) {
      if (a.first != b.first) {
        return a.first < b.first;
      }
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

static void predictBufferPeak(Buffer &b, const SyncModel &model, const std::map<DirKey, unsigned> &peakOf,
                              const std::map<DirKey, unsigned> &poolOf) {
  // The directions this buffer's hazards participate in.
  unsigned worstPeak = 0;
  unsigned tightestPool = 0;
  bool any = false;
  for (const Hazard &h : model.hazards) {
    if (h.isBarrier() || !llvm::is_contained(h.bufferClusters, b.logicId)) {
      continue;
    }
    DirKey d = dirKeyOf(h);
    unsigned peak = peakOf.count(d) != 0 ? peakOf.at(d) : 0;
    unsigned pool = poolOf.count(d) != 0 ? poolOf.at(d) : 0;
    if (!any || peak > worstPeak) {
      worstPeak = peak;
      tightestPool = pool;
    }
    any = true;
  }
  b.predictedPeakOverlap = worstPeak;
  b.smallestPool = tightestPool;
}

static bool bufferTokenExpressible(const Buffer &b, const SyncModel &model) {
  for (const Hazard &h : model.hazards) {
    if (h.isBarrier() || !llvm::is_contained(h.bufferClusters, b.logicId)) {
      continue;
    }
    if (!h.tokenExpressible) {
      return false;
    }
  }
  return true;
}

static void countRoutedHazards(const SyncModel &model, BufferRouteResult &result) {
  // ALL-OR-NOTHING. A hazard touching both a routed and an unrouted buffer cannot be
  // wholly on one mechanism -- that is the split that hangs. Counted here; the caller
  // refuses the whole plan when this is non-zero, before it mutates any hazard.
  for (const Hazard &h : model.hazards) {
    if (h.isBarrier() || h.bufferClusters.empty()) {
      continue;
    }
    bool anyRouted = false, anyUnrouted = false;
    for (const Buffer &b : model.buffers) {
      if (!llvm::is_contained(h.bufferClusters, b.logicId)) {
        continue;
      }
      if (b.routedToBufid) {
        anyRouted = true;
      } else {
        anyUnrouted = true;
      }
    }
    if (anyRouted && anyUnrouted) {
      ++result.splitHazards;
    }
    if (anyRouted && !anyUnrouted) {
      ++result.hazardsCovered;
    }
  }
}

} // namespace

BufferRouteResult mlir::pto::unified::routeBuffers(SyncModel &model) {
  BufferRouteResult result;
  result.buffersConsidered = unsigned(model.buffers.size());

  std::map<DirKey, unsigned> peakOf, poolOf;
  directionPeaks(model, peakOf, poolOf);

  const bool haveCapacity = model.resources.bufidCapacity > 0;

  for (Buffer &b : model.buffers) {
    predictBufferPeak(b, model, peakOf, poolOf);
    if (!haveCapacity) {
      ++result.skippedNoCapacity;
      continue;
    }

    // Second routing reason, off unless asked for: this buffer carries a hazard whose
    // consuming pipe lives in only one arm of a branch its wait was hoisted out of.
    //
    // THE UNIT IS STILL THE BUFFER, NOT THE HAZARD, and that is forced rather than
    // chosen: the token protocol is a counter cycle, so a buffer with some hazards on
    // events and some on get/rls deadlocks. One qualifying hazard therefore carries
    // its whole buffer across, including that buffer's other hazards.
    // STRICT: omega == pool fits exactly, so routing it would be pure waste.
    if (!(b.predictedPeakOverlap > b.smallestPool)) {
      ++result.skippedNoOverflow;
      continue;
    }
    if (!b.isWrittenBack) {
      ++result.skippedNotWrittenBack;
      continue;
    }

    // EVERY hazard on the buffer must be token-expressible, not just the one that
    // triggered routing. Routing is all-or-nothing, so one inexpressible hazard means
    // the whole buffer stays on events -- otherwise that hazard's event ops are
    // suppressed and nothing replaces them.
    bool allExpressible = bufferTokenExpressible(b, model);
    if (!allExpressible) {
      ++result.skippedNotExpressible;
      continue;
    }
    b.routedToBufid = true;
    ++result.routed;
  }

  countRoutedHazards(model, result);

  return result;
}

unsigned mlir::pto::unified::seedHiddenMacroEvents(SyncModel &model, const SyncIRs &syncIR) {
  for (size_t i = 0; i < syncIR.size(); ++i) {
    auto *first = dyn_cast<pto::CompoundInstanceElement>(syncIR[i].get());
    if (!first || first->macroOpInstanceId != 0 || !first->elementOp) {
      continue;
    }
    auto macro = pto::getSyncMacroModel(first->elementOp);
    if (!macro || macro->hiddenEvents.empty()) {
      continue;
    }

    auto span = getSyncMacroSpan(syncIR, i);
    if (!span) {
      continue;
    }
    auto [begin, end] = *span;

    for (const auto &hidden : macro->hiddenEvents) {
      for (unsigned id : hidden.eventIds) {
        model.hiddenReservations.push_back({static_cast<int>(hidden.srcPipe), static_cast<int>(hidden.dstPipe),
                                            static_cast<int>(id), Interval{begin, end},
                                            first->elementOp->getName().getStringRef()});
      }
    }
  }
  return model.hiddenReservations.size();
}

namespace {
class EventDirectionColorer {
public:
  EventDirectionColorer(SyncModel &model, EventColorResult &result, const Hazard &first)
      : model(model), result(result), dir(dirKeyOf(first)),
        pool(model.resources.eventPoolSize(first.srcPipe(), first.dstPipe())), endOf(pool, 0), inUse(pool, false),
        everUsed(pool, false) {}
  void color(Hazard *h) {
    unsigned start = h->interval().start;
    // Left-edge release: an id whose interval ended at or before this start is
    // free again. Touching endpoints do not overlap (Interval::overlaps uses
    // strict `<`), so the test is `<=`.
    for (unsigned id = 0; id < pool; ++id) {
      if (inUse[id] && endOf[id] <= start) {
        inUse[id] = false;
      }
    }

    unsigned d = h->demand();
    int avoid = recentlyFreed(h);
    llvm::SmallVector<int, mlir::pto::kValue4> chosenIds = collect(h, avoid);
    if (chosenIds.size() < d) {
      chosenIds = collect(h, -1); // not enough without it; the spread is a preference
    }
    int chosen = chosenIds.size() == d ? chosenIds.front() : -1;
    if (chosen < 0) {
      // No id free at this instant in this direction. SPILL TO A BARRIER.
      // "No id" is not a resting state: codegen emits the pair regardless, so an
      // unserved hazard goes out carrying id 0 or the unset sentinel.
      //
      // There is no "spill failed" branch below, so the assumption that the
      // barrier class is always available is asserted where it is relied on: a
      // bounded barrier class cannot be introduced without this failing to
      // compile until a failure mode is written here.
      static_assert(ResourceModel::barrierIsUnbounded, "the spill path has no failure branch, so the barrier class "
                                                       "must be unbounded and always available");
      ++result.overflow;
      ++result.spilledToBarrier;
      h->SpillToBarrier();
      return;
    }

    for (int id : chosenIds) {
      inUse[id] = true;
      endOf[id] = h->interval().end;
      if (everUsed[static_cast<unsigned>(id)]) {
        ++result.reused;
      }
      everUsed[static_cast<unsigned>(id)] = true;
    }
    result.idsAssigned += d;
    if (d > 1) {
      ++result.rotating;
    }

    // Backward hazards need no special case: compensation synthesis already gave
    // each a head/tail pair and a forward carrying-loop interval, so they colour
    // like any other. Stamps set, wait, head and tail with the SAME id list --
    // a compensation pair left unstamped emits with the wrong arity.
    h->SetEventIds(chosenIds);
    ++result.assigned;
  }

private:
  SyncModel &model;
  EventColorResult &result;
  DirKey dir;
  unsigned pool;
  llvm::SmallVector<unsigned, mlir::pto::kValue8> endOf;
  llvm::SmallVector<bool, mlir::pto::kValue8> inUse, everUsed;
  bool reservedFor(const Hazard *h, unsigned id) const {
    for (const HiddenReservation &r : model.hiddenReservations) {
      if (r.id == static_cast<int>(id) && r.srcPipe == dir.first && r.dstPipe == dir.second &&
          r.span.overlaps(h->interval())) {
        return true;
      }
    }
    return false;
  }

  int recentlyFreed(const Hazard *h) const {
    int avoid = -1;
    unsigned avoidEnd = 0;
    for (unsigned id = 0; id < pool; ++id) {
      if (!inUse[id] && everUsed[id] && !reservedFor(h, id) && (avoid < 0 || endOf[id] > avoidEnd)) {
        avoidEnd = endOf[id];
        avoid = static_cast<int>(id);
      }
    }

    return avoid;
  }

  llvm::SmallVector<int, 4> collect(const Hazard *h, int skip) const {
    unsigned d = h->demand();
    llvm::SmallVector<int, mlir::pto::kValue4> out;
    for (unsigned id = 0; id < pool && out.size() < d; ++id) {
      if (!inUse[id] && !reservedFor(h, id) && static_cast<int>(id) != skip) {
        out.push_back(static_cast<int>(id));
      }
    }
    return out;
  }
};

} // namespace

EventColorResult mlir::pto::unified::colorEventIds(SyncModel &model) {
  EventColorResult result;

  // Non-barrier hazards, ordered for a per-direction left-edge scan. The order
  // is (direction, interval start, hazard index) -- fully deterministic, and it
  // groups each direction into a contiguous run below.
  llvm::SmallVector<Hazard *> haz;
  for (Hazard &h : model.hazards) {
    if (h.isBarrier()) {
      continue;
    }
    // Routed to the token protocol, so it must NOT take an event id -- a buffer
    // with some hazards on events and some on get/rls is a counter cycle missing a
    // step, i.e. a hang. Skipped, not counted as overflow: overflow means no id was
    // available, this means none was wanted.
    if (h.mechanism() == SyncOperation::MECHANISM::BUFID) {
      ++result.skippedRouted;
      continue;
    }
    haz.push_back(&h);
  }
  llvm::stable_sort(haz, [](const Hazard *a, const Hazard *b) {
    DirKey da = dirKeyOf(*a), db = dirKeyOf(*b);
    if (da != db) {
      return da < db;
    }
    if (a->interval().start != b->interval().start) {
      return a->interval().start < b->interval().start;
    }
    return a->index() < b->index();
  });

  // Color one direction at a time; each is an independent interval graph, so a
  // left-edge sweep with the lowest free id attains its chromatic number.
  for (size_t i = 0; i < haz.size();) {
    DirKey dir = dirKeyOf(*haz[i]);
    EventDirectionColorer colorer(model, result, *haz[i]);

    for (; i < haz.size() && dirKeyOf(*haz[i]) == dir; ++i) {
      colorer.color(haz[i]);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

namespace {
static void printHazards(llvm::raw_ostream &os, const SyncModel &model) {
  for (const Hazard &h : model.hazards) {
    Alpha a = Alpha::forHazard(h);
    os << "  h#" << h.index() << " " << (h.isBarrier() ? "barrier" : "pair") << " "
       << oracle::pipelineTypeName(h.srcPipe()) << "->" << oracle::pipelineTypeName(h.dstPipe()) << " iv=["
       << h.interval().start << "," << h.interval().end << ")"
       << " pool=" << model.resources.eventPoolSize(h.srcPipe(), h.dstPipe())
       << " mech=" << SyncOperation::MechanismName(h.mechanism()) << " clusters=[";
    for (size_t i = 0; i < h.bufferClusters.size(); ++i) {
      if (i != 0) {
        os << ",";
      }
      os << h.bufferClusters[i];
    }
    os << "]"
       << " revwar=" << (h.hasReverseWar ? "yes" : "no") << " share="
       << (h.addressSharing == Hazard::AddressSharing::Reuse       ? "reuse"
           : h.addressSharing == Hazard::AddressSharing::AliasView ? "alias"
                                                                   : "none")
       << " alpha(e/b/bar)=" << a.event << "/" << a.bufid << "/" << a.barrier << " event_id=";
    // ALL ids, not just the first: a rotating hazard holds d of them.
    if (h.hasEventId()) {
      const auto &ids = h.setOp()->eventIds;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
          os << "/";
        }
        os << ids[i];
      }
    } else {
      os << "-";
    }
    os << "\n";
  }
}

static void printBuffers(llvm::raw_ostream &os, const SyncModel &model) {
  // `lbufid=D/N` says how many of the N buffers got a resolved reservation window;
  // D != N is a real gap, not a default. `span_boundary` counts buffers with no
  // single carrying loop, which are given a union span rather than refused.
  // `revwar_divergence` counts buffers where the per-BUFFER `isWrittenBack`
  // disagrees with a hazard's per-HAZARD `hasReverseWar`, so the two predicates
  // cannot silently stand in for one another.
  if (!model.buffers.empty()) {
    unsigned wb = 0, lc = 0, sb = 0, defined = 0, divergence = 0;
    for (const Buffer &b : model.buffers) {
      if (b.isWrittenBack) {
        ++wb;
      }
      if (b.isLoopCarried) {
        ++lc;
      }
      if (b.spansLoopBoundary) {
        ++sb;
      }
      if (b.bufidLoopDefined) {
        ++defined;
      }
      for (const Hazard &h : model.hazards) {
        if (h.isBarrier() || !llvm::is_contained(h.bufferClusters, b.logicId)) {
          continue;
        }
        if (h.hasReverseWar != b.isWrittenBack) {
          ++divergence;
          break;
        }
      }
    }
    os << "  buffers=" << model.buffers.size() << " written_back=" << wb << " loop_carried=" << lc
       << " span_boundary=" << sb << " lbufid=" << defined << "/" << model.buffers.size()
       << " revwar_divergence=" << divergence << "\n";
    for (const Buffer &b : model.buffers) {
      os << "    b#" << b.logicId << " access=[" << b.accessRange.start << "," << b.accessRange.end << ") lbufid="
         << (b.bufidLoopDefined ? "[" + std::to_string(b.bufidLoop.start) + "," + std::to_string(b.bufidLoop.end) + "]"
                                : std::string("UNRESOLVED"))
         << " wb=" << (b.isWrittenBack ? "yes" : "no") << " loop=" << (b.isLoopCarried ? "yes" : "no")
         << " ctx=" << b.loopContextCount << (b.spansLoopBoundary ? " SPANS-BOUNDARY" : "") << "\n";
    }
  }
}

} // namespace

void mlir::pto::unified::printSyncModel(llvm::raw_ostream &os, llvm::StringRef funcName, const SyncModel &model) {
  os << "[unified-model] func=" << funcName << " hazards=" << model.hazards.size()
     << " K=" << model.resources.bufidCapacity << "\n";

  printHazards(os, model);

  printBuffers(os, model);
}
