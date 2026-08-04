// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- UnifiedSyncModel.h - Unified sync allocator data model -------------===//
//
// The types the unified allocator colors: the hazard, its live
// interval, the per-class resource model, and the per-hazard mechanism cost.
//
// The organising idea of the whole allocator is that event ids, buffer ids and
// barriers are three RESOURCE CLASSES over one interval-coloring problem, not
// three algorithms. This file is that problem's statement; it decides nothing.
// Allocation, mechanism routing and codegen all live in the pass.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_INSERTSYNC_UNIFIEDSYNCMODEL_H
#define PTO_TRANSFORMS_INSERTSYNC_UNIFIEDSYNCMODEL_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
namespace unified {

//===----------------------------------------------------------------------===//
// Joining a hazard to its buffer-id cluster
//===----------------------------------------------------------------------===//

/// Which buffer-id cluster(s) a dependency's memory belongs to.
///
/// Keyed on the `BaseMemInfo *` itself, and built by the CALLER -- because
/// deciding "these two are the same buffer" requires `MemoryDependentAnalyzer`,
/// which this header must not depend on. `PTOUnifiedSync` owns the analyzer and
/// does the resolution.
///
/// THE IDENTITY IS `MemAlias`, NOT A TUPLE. Three keys look plausible and all
/// three are WRONG, each in an instructive way -- do not "simplify" this back to
/// any of them:
///
///   - `BaseMemInfo::rootBuffer` is the root ALLOCATION, which is coarser than a
///     buffer: several buffers can share one root, so this key conflates tiles that
///     do not alias.
///   - `TileInfo::rootBuffer` never matches at all: BufidSync rewrites it to the
///     tile value when the root is `ConstantLike`, so the two sides are looking
///     at different fields of the same object.
///   - `(scope, baseAddr, size)` LOOKS exact and is not. `baseAddresses` is EMPTY
///     until addresses are assigned, so `baseAddr` collapses to 0 and distinct
///     clusters degenerate onto one key. In
///     BufidSync's own code that tuple is only a BUCKET -- it is confirmed with
///     `memAnalyzer_.MemAlias(a.memInfo, b.memInfo)`, and the MemAlias call is
///     what actually decides identity.
using MemInfoToClusters =
    llvm::DenseMap<const BaseMemInfo *, llvm::SmallVector<int, 2>>;

//===----------------------------------------------------------------------===//
// Interval
//===----------------------------------------------------------------------===//

/// A hazard's live range in SyncIR-index space: the sync is live from where the
/// producer signals to where the consumer waits.
///
/// This is the coordinate system MoveSyncState works in, and it is deliberately
/// NOT op position. MoveSyncState hoists a sync out of a branch or loop by
/// rewriting these indices without moving a single operation -- which is exactly
/// why event and buffer-ID hazards can coexist: hoisting relocates the
/// interval, while a buffer-ID hazard's op anchor is untouched by it.
///
/// Overlap is the conflict test for interval coloring: two hazards may share a
/// physical id only if their intervals are disjoint. Touching endpoints do NOT
/// overlap -- an id freed at index i is reusable by a hazard starting at i.
struct Interval {
  unsigned start = 0; ///< SyncIR index of the set/acquire side.
  unsigned end = 0;   ///< SyncIR index of the wait/release side.

  bool overlaps(const Interval &o) const {
    return start < o.end && o.start < end;
  }
};

//===----------------------------------------------------------------------===//
// Hazard -- the unit of allocation, and the unit of MECHANISM
//===----------------------------------------------------------------------===//

/// One ordering that must be enforced: producer on `srcPipe` before consumer on
/// `dstPipe`. Owns BOTH halves of the sync pair.
///
/// THE BOTH-HALVES INVARIANT. The mechanism belongs to the HAZARD, never to an
/// individual SyncOperation, and `SetMechanism` below is the only writer. This
/// is not stylistic -- it is forced by the front-end:
///
///   `SyncOperation::GetMatchSync()` -- which copies the mechanism from a set to
///   its wait -- runs inside InsertSyncAnalysis, at the moment the pair is first
///   built, while BOTH halves still hold their TYPE-derived default. It never
///   runs again. So there is no path by which stamping one half propagates to
///   the other.
///
/// Route a hazard, and both halves move together. Stamp a lone SyncOperation and
/// you get a set on one mechanism whose wait is on another -- which is not a
/// pair at all, but two unrelated syncs, and no gate downstream would catch it
/// (G1 sees the ordering, G2 sees no id conflict, G4 sees the same op count).
class Hazard {
public:
  using MECHANISM = SyncOperation::MECHANISM;

  Hazard(unsigned index, SyncOperation *setOp, SyncOperation *waitOp)
      : index_(index), setOp_(setOp), waitOp_(waitOp) {}

  unsigned index() const { return index_; }
  SyncOperation *setOp() const { return setOp_; }
  SyncOperation *waitOp() const { return waitOp_; }

  /// Whether a wait side ever existed -- STRUCTURE, not mechanism. It stays false
  /// after `SpillToBarrier`, which converts a set/wait pair, so a spilled hazard
  /// reports `isBarrier() == false` alongside `mechanism() == BARRIER`.
  ///
  /// Both readings are needed and must not be merged: the colourer and the carried
  /// scan skip structurally-lone barriers because there is no pair to colour, while
  /// codegen and the gates need the spilled pair's mechanism. A reader asking "is
  /// this realised as a barrier" must use `mechanism()`.
  bool isBarrier() const { return waitOp_ == nullptr; }

  PipelineType srcPipe() const { return setOp_->GetActualSrcPipe(); }
  PipelineType dstPipe() const { return setOp_->GetActualDstPipe(); }

  const Interval &interval() const { return interval_; }
  void setInterval(Interval iv) { interval_ = iv; }

  /// The mechanism backing this hazard. Reads the set side; the invariant above
  /// guarantees the wait side agrees.
  /// MECHANISM, NOT STRUCTURE -- and it can disagree with `isBarrier()`.
  /// After `SpillToBarrier` this reports BARRIER while `isBarrier()` is still
  /// false, because the wait side object survives (marked `uselessSync`). This is
  /// the authoritative answer to "how is this hazard realised"; `isBarrier()`
  /// answers only "was it ever a lone barrier". See the note there.
  MECHANISM mechanism() const { return setOp_->GetMechanism(); }

  /// This hazard's buffer was routed to buffer-ID, so the
  /// token protocol backs it and it must NOT take an event id.
  ///
  /// SUPPRESSING THE EVENT OPS IS NOT OPTIONAL. `SyncCodegen` dispatches on the
  /// SyncOperation TYPE, never on `GetMechanism()` -- stamping BUFID alone changes
  /// nothing, and the set/wait pair would still be emitted, now with no event id.
  /// That re-creates the unrealised-hazard defect the barrier spill exists to prevent. So the ops are marked
  /// `uselessSync`, which is the same suppression `SpillToBarrier` relies on and
  /// which codegen honours (`SyncInsert`'s first line).
  ///
  /// All FOUR ops, because `SetEventId` stamps head/tail
  /// too, so a hazard that never takes an id leaves its synthesised compensation
  /// unstamped, and those are real ops in syncIR's pipe lists. A buffer-ID token
  /// needs no priming pair anyway -- the counter starts free.
  void RouteToBufid() {
    SetMechanism(MECHANISM::BUFID);
    setOp_->uselessSync = true;
    if (waitOp_)
      waitOp_->uselessSync = true;
    if (headOp_)
      headOp_->uselessSync = true;
    if (tailOp_)
      tailOp_->uselessSync = true;
  }

  /// The colourer could not serve this hazard, so realise it as a barrier
  /// rather than leaving it with no id.
  ///
  /// WHY THIS EXISTS AT ALL. `colorEventIds` deliberately assigns NOTHING on
  /// overflow, so that G3 id-legality stays true by construction -- but
  /// `SyncCodegen` emits the pair regardless, so the hazard went out carrying ids
  /// nobody handed it: `id=0` (colliding with the legitimate holder of 0) or the
  /// unset sentinel `2147483647`. "No id" was never a safe resting state; a
  /// hazard must end on SOME mechanism.
  ///
  /// ONE BARRIER PER HAZARD, PLAIN -- no interval stabbing. All four ops are
  /// converted so that `mechanism()` reports BARRIER consistently, then every one
  /// except the wait side is marked `uselessSync` so exactly one PIPE_ALL is
  /// emitted.
  ///
  /// THE WAIT SIDE IS THE ONE KEPT, and that choice is load-bearing for
  /// loop-carried hazards. A PIPE_ALL at the wait position sits before the
  /// consumer; for a backward hazard the in-body wait is at the TOP of the body,
  /// so the barrier drains the previous iteration -- ordering that iteration's
  /// producer before this one's consumer, which is exactly the carried edge. A
  /// barrier at the set side would sit after the producer and order nothing across
  /// the back edge.
  ///
  /// HEAD/TAIL MUST BE SUPPRESSED TOO, and missing this would have reintroduced
  /// the very defect being fixed. `SetEventId` stamps all FOUR ops, so a hazard
  /// that never gets an id leaves its synthesised compensation pair unstamped as
  /// well -- and those are real ops in `syncIR`'s pipe lists, so they
  /// would emit set/wait with the sentinel exactly like the main pair. An
  /// overflowing hazard can carry compensation, so this is reachable. Once the
  /// hazard is a barrier the compensation is redundant anyway -- a PIPE_ALL needs
  /// no priming.
  void SpillToBarrier() {
    if (!waitOp_)
      return; // already a barrier; the colourer never reaches these
    setOp_->SetPipeAll();
    waitOp_->SetPipeAll();
    setOp_->uselessSync = true;
    if (headOp_) {
      headOp_->SetPipeAll();
      headOp_->uselessSync = true;
    }
    if (tailOp_) {
      tailOp_->SetPipeAll();
      tailOp_->uselessSync = true;
    }
  }

  /// THE ONLY WAY a mechanism may be assigned. Stamps both halves.
  void SetMechanism(MECHANISM m) {
    setOp_->SetMechanism(m);
    if (waitOp_)
      waitOp_->SetMechanism(m);
  }

  /// Assign the physical id to EVERY op the hazard owns, from ONE coloring
  /// decision: set + wait, and -- for a loop-carried hazard -- the synthesised
  /// head-set and tail-wait too. Stamping them together (never re-coloring the
  /// head/tail separately) is exactly what makes the head prime and the tail
  /// drain the SAME flag the in-body pair uses. This is the id-sharing the
  /// a correct loop needs, and the reason a naive port double-syncs:
  /// there, head/tail are a separate coloring event; here they are not.
  /// d -- how many event ids this hazard needs live across its interval.
  ///
  /// 1 for an ordinary hazard. >1 for a MULTI-BUFFERED (rotating) one: the
  /// frontend's `pto.multi_tile_get %mb[%slot]` gives the pair a slot SSA and
  /// `eventIdNum = N`, and codegen emits `set_flag_dyn`/`wait_flag_dyn` selecting
  /// `eventIds[slot % N]`. Read from the set side; the both-halves invariant keeps
  /// the two in step.
  unsigned demand() const {
    return std::max(1u, static_cast<unsigned>(setOp_->eventIdNum));
  }

  /// THE ONLY WAY ids may be assigned when d > 1. Stamps ALL FOUR ops with the
  /// SAME id list.
  ///
  /// SPILLING MUST STAMP EVERY OP. `SetEventId` already stamps set/wait/head/tail
  /// because a hazard that misses its compensation emits with the wrong arity --
  /// and here "wrong arity" is worse than a wrong id: `CreateSetWaitOpForMultiBuffer`
  /// asserts `eventIds.size() == slotCount` and indexes `eventIds[i]` for
  /// i in [0, n), so a short list is an out-of-bounds read, not a mis-sync.
  void SetEventIds(llvm::ArrayRef<int> ids) {
    assert(!ids.empty() && "a hazard must get at least one id");
    setOp_->eventIds.assign(ids.begin(), ids.end());
    if (waitOp_)
      waitOp_->eventIds.assign(ids.begin(), ids.end());
    if (headOp_)
      headOp_->eventIds.assign(ids.begin(), ids.end());
    if (tailOp_)
      tailOp_->eventIds.assign(ids.begin(), ids.end());
  }

  void SetEventId(int id) {
    setOp_->eventIds.assign(1, id);
    if (waitOp_)
      waitOp_->eventIds.assign(1, id);
    if (headOp_)
      headOp_->eventIds.assign(1, id);
    if (tailOp_)
      tailOp_->eventIds.assign(1, id);
  }
  bool hasEventId() const { return !setOp_->eventIds.empty(); }

  //-- Loop-carried compensation -----------------------------------------------
  //
  // A loop-carried backward hazard's in-body pair (wait, then set) deadlocks on
  // iteration 1 with nothing priming the flag. The fix is a head-set before the
  // carrying loop and a tail-wait after it, on the SAME id (SetEventId above).
  // These are synthesised by `synthesizeLoopCompensation` before coloring; they
  // live in `SyncOperations` (so they are extracted and, later, emitted) and the
  // hazard holds raw pointers to them here. Null for a forward hazard.

  /// Attach the synthesised head-set / tail-wait. Set once, before coloring.
  void setCompensation(SyncOperation *head, SyncOperation *tail) {
    headOp_ = head;
    tailOp_ = tail;
  }
  SyncOperation *headOp() const { return headOp_; }
  SyncOperation *tailOp() const { return tailOp_; }
  bool hasCompensation() const { return headOp_ != nullptr; }

  //-- What makes buffer-ID cost REUSE-CONDITIONED ----------------------------
  //
  // A buffer-ID token's get/rel counters are bidirectional: ONE token discharges
  // the forward RAW *and* the reverse WAR. So buffer-ID is cheaper than an event
  // exactly when the buffer really is reused in both directions, and on a
  // forward-only hazard it costs about the same while adding a reverse edge
  // nobody asked for. The cost therefore cannot be read off the hazard alone --
  // it needs the buffer's whole hazard group. That is what these two fields are.

  /// Every buffer-id cluster this hazard's memory belongs to. Joined on TileKey;
  /// empty when none of the hazard's tiles is buffer-id-clusterable.
  llvm::SmallVector<int, 2> bufferClusters;

  /// Does a genuine reverse (WAR) hazard exist on the SAME buffer -- i.e. a
  /// mirror-direction hazard sharing a cluster with this one?
  ///
  /// THE KEY THIS IS COMPUTED ON IS LOAD-BEARING.
  /// A buffer-id token is only cheap if its reverse edge is really wanted. Get
  /// this predicate wrong in the FALSE-POSITIVE direction and the offload policy
  /// prices buffer-id at 1 instead of 2, routes a *forward-only* hazard onto a
  /// bidirectional token, and manufactures precisely the spurious reverse edge
  /// this predicate exists to avoid.
  ///
  /// Keying this on `depRootBuffers` -- as the first cut did -- does exactly
  /// that: the root is the root ALLOCATION, so two hazards on *different tiles*
  /// that merely share a root are reported as reverse partners when there is no
  /// reverse dependency between them at all. Clusters are the correct unit
  /// because one token covers one aliasing clique.
  bool hasReverseWar = false;

private:
  unsigned index_;
  SyncOperation *setOp_;  ///< never null
  SyncOperation *waitOp_; ///< null for a barrier
  SyncOperation *headOp_ = nullptr; ///< synthesised head-set; null unless loop-carried
  SyncOperation *tailOp_ = nullptr; ///< synthesised tail-wait; null unless loop-carried
  Interval interval_;
};

//===----------------------------------------------------------------------===//
// Per-class resource model
//===----------------------------------------------------------------------===//

/// The three id pools, in one place, with no arch branch anywhere.
///
/// K IS THE ARCHITECTURE. K=0 leaves the buffer-ID class empty, so the allocator
/// degenerates to event+barrier -- which is A3. K=32 opens it -- which is A5.
/// There is no `if (arch == a5)` in the allocator, and there must never be one.
struct ResourceModel {
  /// Buffer-ID pool: ONE pool, shared across all pipes and both directions.
  unsigned bufidCapacity = 0; // K

  /// Event pool for one direction. Event flags are per-(src,dst) DISJOINT
  /// hardware -- id 0 in MTE2->V is a different flag from id 0 in V->MTE3 -- so
  /// this is a pool PER DIRECTION, not a shared pool of 8. Some directions
  /// (V<->S) reserve their top id, so this is not always 8.
  ///
  /// Reads `kTotalEventIdNum` and `SyncEventIdAllocation::GetReservedEventIdNum`
  /// so the allocator and the G3 gate can never disagree about the bound.
  unsigned eventPoolSize(PipelineType src, PipelineType dst) const;

  /// Barrier is the spill class: unbounded, and always available. That is what
  /// makes the allocator total -- there is no "no id left" failure mode, only a
  /// worse schedule.
  static constexpr bool barrierIsUnbounded = true;
};

//===----------------------------------------------------------------------===//
// alpha -- the per-hazard mechanism cost
//===----------------------------------------------------------------------===//

/// What a hazard costs on each resource class.
///
/// SHAPE ONLY. The numbers here are an ordering, not a calibrated model,
/// and NOTHING reads them yet: the policy that consumes alpha is the offload seam
/// and a future flow policy may replace the scalar with a
/// length-aware term. What this fixes is the *interface* -- that cost is a
/// function of the hazard AND its buffer's hazard group, not of the hazard alone.
/// Getting this wrong here would bake a flat buffer-ID cost into
/// every later policy.
struct Alpha {
  unsigned event = 1;
  unsigned bufid = 1;
  unsigned barrier = 100; ///< spill: serializes the core, so it must lose to any id.

  /// In one line: a buffer-ID token is cheap only when its reverse WAR is
  /// really wanted. Forward-only, the same token adds a spurious back edge, so
  /// it is priced above an event rather than level with it.
  static Alpha forHazard(const Hazard &h) {
    Alpha a;
    a.bufid = h.hasReverseWar ? 1 : 2;
    return a;
  }
};

//===----------------------------------------------------------------------===//
// Model construction
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Buffer -- the unit buffer-ID routing decides on
//===----------------------------------------------------------------------===//
//
// THE BUFFER, NOT THE HAZARD, IS THE ROUTING UNIT, and that is a correctness
// requirement rather than a modelling preference. The buffer-ID protocol is a
// COUNTER CYCLE: every `get_buf` must be matched by an `rls_buf` on the same id.
// If some hazards on one buffer are realised as events while others take
// get/rls, the cycle is missing a step and the acquire blocks forever. So routing
// must be decided per buffer, BEFORE any hazard on it takes an event id. At K=0
// that was vacuous (the buffer-ID class was empty); at K=32 it is
// correctness-critical.
//
// l_bufid IS THE UNION SPAN, matching the shipping pass -- decided from its
// behaviour, not from this struct's shape. `BufidSyncIdAlloc::computeLifeIntervals`
// computes `inLoop` PER OP: an access outside a loop contributes its own
// `syncIRIndex`, an access inside contributes the whole `[loopBegin, loopEnd]`,
// and the result is min-start/max-end per logic id. That is the union span, and it
// is what production does on 40 real a5 buffers today.
//
// WHY UNION-SPAN AND NOT REFUSE, for the buffers that have no single loop scope:
// over-reserving an id is WASTEFUL, NOT WRONG. The `get_buf`/`rls_buf` ops are
// emitted at the access sites regardless; the interval only decides which PHYSICAL
// id a logic id maps to. A reservation that is too LONG cannot lose an ordering --
// it can only lose an id, and the shipping pass already degrades gracefully there
// (`while (maxPhysicalIdUsed_ >= physicalBufIdCount_)` coalesces logic ids rather
// than failing). A reservation can therefore span most of a function -- wide but
// correct. Refusing would be STRICTER THAN PRODUCTION: it would reject buffers the
// shipping pass handles, for no correctness property that can be named.
//
// SO `spansLoopBoundary` IS A ROUTING INPUT, NOT A REFUSAL. It marks the buffers
// whose reservation is deliberately conservative, so a later cost model can prefer
// events for them instead of paying a near-whole-function bufid reservation. It is
// COUNTED and printed, never silently absorbed. The shape it marks is common rather
// than hypothetical: a buffer touched both inside and outside a loop, or in two
// different loops, has no single carrying loop to scope its reservation to.
struct Buffer {
  /// The aliasing-clique id (`bufferClusters` entry). One token covers one clique.
  int logicId = -1;

  /// The routing predicate's other half: does anything write this buffer back?
  /// A token's counters run both ways, so it discharges the forward RAW *and* the
  /// reverse WAR -- a saving only when that reverse edge is genuinely wanted.
  ///
  /// THIS IS NOT THE SAME PREDICATE AS `Hazard::hasReverseWar`, and the
  /// difference is deliberate rather than an oversight -- see the divergence count
  /// in the model report. Per-hazard asks "does a mirror-direction hazard share a
  /// cluster with ME"; per-buffer asks "does this buffer carry a mirror pair at
  /// all". A third hazard touching the same buffer in an unmirrored direction
  /// inherits `true` here and `false` there. The per-buffer form is the correct one
  /// because the token is a property of the buffer, not of one hazard.
  bool isWrittenBack = false;

  /// Union of every access, in SyncIR-index space. Accesses only -- no loop
  /// widening. This is the tight fact; `bufidLoop` is the reservation.
  Interval accessRange{0, 0};

  /// Any access inside a loop.
  bool isLoopCarried = false;

  /// Item 3: accesses live in more than one loop context (inside AND outside, or
  /// in two different loops), so there is NO single well-defined carrying loop.
  bool spansLoopBoundary = false;

  /// How many distinct loop contexts the accesses occupy (function level counts
  /// as one). 1 means a single well-defined scope.
  unsigned loopContextCount = 0;

  /// l_bufid: the reservation window. Union span -- an access outside a loop
  /// contributes itself, an access inside contributes its whole loop.
  Interval bufidLoop{0, 0};

  /// Step 2 routing outcome. `true` means: every hazard touching this buffer is
  /// to be realised as a buffer-ID token, and NONE of them may take an event id.
  bool routedToBufid = false;

  /// Peak per-direction overlap predicted for the directions this buffer touches,
  /// and the smallest pool among them. Routing compares these; recorded so the
  /// decision is auditable rather than a bare bool.
  unsigned predictedPeakOverlap = 0;
  unsigned smallestPool = 0;

  /// TOTALITY, on the `unmappable=0` discipline: false means l_bufid could not be
  /// resolved, and such buffers are COUNTED in the report rather than silently
  /// defaulted to a plausible-looking interval. A wrong l_bufid is a wrong
  /// reservation window, which is a missing get/rls, which is a hang.
  bool bufidLoopDefined = false;
};

/// The hazard set H, plus the resources it must be colored into.
/// A (direction, id) pair the pto-isa library occupies INSIDE a macro call, over
/// that call's span. Pre-colouring: the id is not chosen here, it is already taken,
/// so the colourer must treat it as occupancy rather than as a candidate.
struct HiddenReservation {
  int srcPipe = -1;
  int dstPipe = -1;
  int id = -1;
  Interval span;                 ///< SyncIR indices, already padded
  llvm::StringRef macroName;     ///< for diagnostics only
};

struct SyncModel {
  llvm::SmallVector<Hazard> hazards;
  /// One entry per buffer-id cluster touched by any hazard, sorted by logicId.
  llvm::SmallVector<Buffer> buffers;
  ResourceModel resources;
  /// Ids already spoken for by a macro's library implementation. Empty on any
  /// kernel with no macro ops, which is why this costs nothing to consult.
  llvm::SmallVector<HiddenReservation> hiddenReservations;
};

/// Build the model from the reused front-end's output. Reads `SyncOperations`,
/// whose OUTER index is already the hazard: one group per ordering, holding its
/// set and its wait. That is the granularity the both-halves invariant needs, and
/// it is why `Hazard` can own the pair without re-deriving the pairing.
///
/// `memInfoToClusters` joins a hazard's memory to its buffer-id cluster(s); see
/// the type's note on why the caller must build it. An empty map is legal and
/// simply leaves every hazard unclustered, with `hasReverseWar` false.
///
/// Pure with respect to allocation: assigns no ids and emits no IR. (It does not
/// mutate the SyncOperations at all -- `Hazard` only holds pointers to them.)
///
/// `syncIR` is read (not mutated) only to resolve a loop-carried hazard's
/// CARRYING loop: `setOp->GetForEndIndex()` indexes a `LoopInstanceElement`
/// whose `[beginId, endId)` is that hazard's event interval.
/// This is why the `end < start` clamp is gone -- a backward hazard's interval is
/// read off its carrying loop, forward, never inverted.
SyncModel buildSyncModel(const SyncOperations &syncOps, unsigned bufidCapacity,
                         const MemInfoToClusters &memInfoToClusters,
                         const SyncIRs &syncIR);

/// Synthesise the head-set / tail-wait
/// compensation pair for every loop-carried (GetForEndIndex) hazard, so its
/// in-body backward pair is primed before the loop and drained after -- without
/// them the kernel deadlocks on iteration 1, and it is NOT optional.
///
/// For each such hazard it appends ONE new `SyncOperations` group holding a
/// head-set (at the carrying loop's begin) and a tail-wait (at its end), and
/// links both onto the hazard via `setCompensation`, so one `SetEventId` stamps
/// all four ops with one id. Exactly one head and one tail per hazard: the loop
/// runs once, no retry, no reallocation -- the structural one-of-each guarantee
/// a naive port lacks. Returns the number of pairs synthesised.
///
/// Mutates `syncOps` (appends groups) and `model` (links + no id yet). Runs
/// AFTER buildSyncModel (hazards must exist) and BEFORE colorEventIds (ids come
/// from coloring).
/// Also PLACES each pair into `syncIR`: head-set into the carrying
/// loop begin's `pipeBefore`, tail-wait into the loop end's `pipeAfter` --
/// `push_front` for the tail, so it precedes any existing loop-end set. That is
/// how `SyncCodegen` will find them; owning them in `syncOps` alone would let
/// emission drop them silently. Hence `syncIR` is mutable here.
unsigned synthesizeLoopCompensation(SyncModel &model, SyncOperations &syncOps,
                                    SyncIRs &syncIR);

/// Deterministic, diff-friendly rendering (one hazard per line).
void printSyncModel(llvm::raw_ostream &os, llvm::StringRef funcName,
                    const SyncModel &model);

//===----------------------------------------------------------------------===//
// Event interval-coloring allocator
//===----------------------------------------------------------------------===//

/// Result of the event-id coloring.
struct EventColorResult {
  /// Hazards the colourer could not serve, realised as PIPE_ALL
  /// barriers. Equal to `overflow` -- kept separate so a future partial spill is
  /// visible rather than silently folded into the overflow count.
  unsigned spilledToBarrier = 0;
  unsigned assigned = 0; ///< hazards that got an event id
  unsigned overflow = 0; ///< hazards whose direction had no free id at that instant
  unsigned skippedRouted = 0; ///< hazards on buffer-ID-routed buffers: not ours
  unsigned rotating = 0;      ///< hazards assigned d > 1 ids (multi-buffered)
  unsigned idsAssigned = 0;   ///< total ids handed out (sum of d), not hazards
  unsigned reused = 0;   ///< assignments that took an id freed earlier in the SAME
                         ///< direction -- the whole point of coloring over the naive
                         ///< stub, which never reused and so overflowed at peak.
};

/// Two reporting conventions that a reader can misread at d > 1: `reused` counts ID
/// ASSIGNMENTS, so one hazard of demand d can add up to d, and `assigned` is
/// per-HAZARD while `idsAssigned` is per-ID.
///
/// G2/G3's IR view sees the dyn ops, but a rotating id is a RUNTIME value recorded
/// as -1, so the static nesting rules cannot pair a rotating set with its wait
/// there. The SYNCOPS view checks those ids, and is the only place that does.
///
/// Per-(src,dst) greedy first-fit interval colouring, the classic left-edge
/// algorithm. Event flags are per-direction DISJOINT hardware, so each direction is
/// an independent interval graph.
///
/// A multi-buffered hazard needs d ids live across its interval, so this is WEIGHTED
/// colouring and the bound is `chi = omega_weighted`, the peak SUM of demands over a
/// point rather than the peak count of intervals. That holds only because the d ids
/// need NOT be contiguous: `CreateSetWaitOpForMultiBuffer` materialises each
/// `eventIds[i]` as an independent constant selected by `slotMod == i`, and the dyn
/// ops take the id as a runtime operand that lowering passes straight through, so
/// nothing constrains the id set. Were contiguity required this would be colouring
/// with BANDWIDTH, which is NP-hard, and the greedy core would not extend.
///
/// Taking the d lowest free ids at each interval start is then first-fit on a
/// resource of capacity `pool`, optimal against omega_weighted by the left-edge
/// exchange argument, in O(n log n + n*pool). Deterministic: hazards are ordered by
/// (direction, interval start, hazard index) and always take the lowest free ids.
///
/// A hazard must take EITHER all d ids OR none. Fewer than d emits a selector chain
/// indexing past `eventIds`, which is an out-of-bounds read rather than a mis-sync,
/// so the availability test is all-or-nothing and a short pool spills the whole
/// hazard to a barrier. Structurally-lone barriers are skipped: they hold no id.
/// Pre-colour the ids a macro's library implementation consumes internally.
///
/// AFTER buildSyncModel and BEFORE colorEventIds: the colourer must see these as
/// occupied, or it will hand one out to a compiler hazard whose interval spans the
/// call and the library's wait will be satisfied by the wrong producer.
///
/// The span runs from the macro's first phase to its last, padded one SyncIR index
/// each side. The pad is not cosmetic: without it a hazard ending exactly where the
/// call begins reads as disjoint and the collision survives.
///
/// DELIBERATELY NOT SHARED with the oracle's own reservation walk, which re-derives
/// the same spans independently. Sharing the code would make a bug in the span
/// arithmetic invisible to the gate that exists to catch exactly that.
/// Returns the number of reservations recorded.
unsigned seedHiddenMacroEvents(SyncModel &model, const SyncIRs &syncIR);

EventColorResult colorEventIds(SyncModel &model);

//===----------------------------------------------------------------------===//
// Step 2 -- buffer routing
//===----------------------------------------------------------------------===//
//
// THE ORDERING CONSTRAINT IS THE WHOLE DESIGN, and it is a correctness
// requirement, not a phase convention:
//     Step 1 (settle ops) -> Step 2 (route BUFFERS) -> Step 3 (colour the rest)
// Routing is per-BUFFER and ALL-OR-NOTHING. The buffer-ID protocol is a COUNTER
// CYCLE -- every `get_buf` matched by an `rls_buf` on the same id -- so if some
// hazards on one buffer are realised as events while others take get/rls, the
// cycle is missing a step and the acquire BLOCKS FOREVER. A split buffer is a
// HANG, not a worse schedule. At K=0 this was vacuous (empty buffer-ID class); at
// K=32 it is the first place a wrong answer hangs the core.
//
// THE PREDICATE: route b iff
//     K > 0  &&  b overflows its hazards' event pool  &&  b.isWrittenBack
//
//   (1) WHICH WRITE-BACK PREDICATE -- per-BUFFER `Buffer::isWrittenBack`, NOT
//       per-hazard `Hazard::hasReverseWar`. The token is a
//       property of the buffer, and there is a structural reason on top of that:
//       routing is all-or-nothing, so the predicate MUST be a buffer-level fact.
//       A per-hazard predicate lets two hazards on one buffer disagree -- which is
//       exactly the split that hangs. The two predicates genuinely disagree: a
//       buffer carrying an unmirrored third hazard is written back as a buffer while
//       that hazard alone is not. Under per-buffer such a buffer routes or does not
//       as a unit; under per-hazard it would be split. Per-buffer is the only one of
//       the two that CANNOT produce the deadlock shape.
//
//   (2) "OVERFLOWS ITS POOL" is a PREDICTION, because Step 3 has not run yet and
//       there is no overflow to observe. It is computed from the interval
//       structure: for each direction this buffer's hazards participate in, take
//       the peak simultaneous overlap omega over ALL hazards in that direction --
//       the same quantity the left-edge colourer will hit -- and compare against
//       that direction's pool. `predictedPeakOverlap`/`smallestPool` record it.
//       THE TEST IS STRICT (`omega > pool`), i.e. deliberately UNDER-predicting
//       at the boundary. Reason: the fallbacks are asymmetric. A buffer left on
//       events that then overflows spills to a barrier -- correct, just slower.
//       A buffer routed unnecessarily consumes K, which is ONE pool of 32 shared
//       across every pipe and direction, and whose own exhaustion path coalesces
//       logic ids and serialises co-consumers. At omega == pool the colouring fits
//       exactly, so routing it would be pure waste. Erring toward NOT routing is
//       therefore the cheap-mistake direction.
//
//   (3) `spansLoopBoundary` IS RECORDED, NOT CONSULTED. It OVER-FLAGS, because the
//       model derives loop context from hazard interval endpoints and a loop-carried
//       hazard's interval IS its loop. Using an over-flagging predicate to SKIP
//       buffer-ID would skip more than the evidence supports, cancelling the routing
//       the overflow predicate just asked for. It becomes a cost input when the
//       count is exact, which
//       needs per-access-site positions the model does not carry yet.
struct BufferRouteResult {
  unsigned buffersConsidered = 0;
  unsigned routed = 0;
  unsigned hazardsCovered = 0;   ///< hazards belonging to a routed buffer
  unsigned skippedNoOverflow = 0;
  unsigned skippedNotWrittenBack = 0;
  unsigned skippedNoCapacity = 0; ///< K == 0
  /// ALL-OR-NOTHING violations: a hazard that touches BOTH a routed and an
  /// unrouted buffer, so it cannot be wholly on one mechanism. Must be 0.
  unsigned splitHazards = 0;
};

/// Decide, per buffer, whether buffer-ID backs it. Sets `Buffer::routedToBufid`
/// and records the split check. Assigns NO ids and stamps NO mechanisms -- see
/// the staging note in PTOUnifiedSync.cpp.
BufferRouteResult routeBuffers(SyncModel &model);

} // namespace unified
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_INSERTSYNC_UNIFIEDSYNCMODEL_H
