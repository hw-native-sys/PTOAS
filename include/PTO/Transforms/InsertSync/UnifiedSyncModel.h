// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- UnifiedSyncModel.h - Unified sync allocator data model -------------===//
//
// Event ids, buffer ids and barriers are three resource classes over one
// interval-coloring problem. This file states that problem -- the hazard, its
// interval, the per-class resource model, the per-hazard cost -- and decides
// nothing. Allocation, routing and codegen live in the pass.
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
/// Built by the caller: deciding that two tiles are the same buffer needs
/// `MemoryDependentAnalyzer`, which this header must not depend on.
///
/// Cluster identity is `MemAlias`, not a tuple. Three near-misses this must not be
/// simplified back to: `BaseMemInfo::rootBuffer` is the root ALLOCATION, coarser
/// than a buffer, so it conflates tiles that do not alias; `TileInfo::rootBuffer`
/// never matches, because BufidSync rewrites it to the tile value for a
/// `ConstantLike` root; `(scope, baseAddr, size)` degenerates before addresses are
/// assigned, when `baseAddresses` is empty and every `baseAddr` collapses to 0.
using MemInfoToClusters =
    llvm::DenseMap<const BaseMemInfo *, llvm::SmallVector<int, 2>>;

//===----------------------------------------------------------------------===//
// Interval
//===----------------------------------------------------------------------===//

/// A hazard's live range in SyncIR-index space -- deliberately not op position.
/// MoveSyncState hoists a sync by rewriting these indices without moving an
/// operation, which is what lets event and buffer-ID hazards coexist: hoisting
/// relocates the interval while a buffer-ID hazard's op anchor is untouched.
///
/// Two hazards may share a physical id only if their intervals are disjoint.
/// Touching endpoints do NOT overlap: an id freed at index i is reusable at i.
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

/// One ordering to enforce: producer on `srcPipe` before consumer on `dstPipe`.
/// Owns both halves of the sync pair.
///
/// THE BOTH-HALVES INVARIANT. The mechanism belongs to the hazard, not to an
/// individual SyncOperation, and `SetMechanism` is the only writer. `GetMatchSync`
/// does copy the mechanism to the half it creates, but it built this pair inside
/// InsertSyncAnalysis, before any mechanism was chosen, so a later stamp on one
/// half leaves the other on its type-derived default. A set on one mechanism whose
/// wait is on another is not a pair but two unrelated syncs, and no gate catches
/// it: G1 sees the ordering, G2 sees no id conflict, G4 sees the same op count.
class Hazard {
public:
  using MECHANISM = SyncOperation::MECHANISM;

  Hazard(unsigned index, SyncOperation *setOp, SyncOperation *waitOp)
      : index_(index), setOp_(setOp), waitOp_(waitOp) {}

  unsigned index() const { return index_; }
  SyncOperation *setOp() const { return setOp_; }
  SyncOperation *waitOp() const { return waitOp_; }

  /// Whether a wait side ever existed -- STRUCTURE, not mechanism.
  /// `SpillToBarrier` converts a pair without dropping its wait object, so a
  /// spilled hazard reports `isBarrier() == false` and `mechanism() == BARRIER`.
  /// Both readings are needed: the colourer skips structurally-lone barriers
  /// because there is no pair to colour, while codegen and the gates need the
  /// spilled pair's mechanism. "Is this realised as a barrier" is `mechanism()`.
  bool isBarrier() const { return waitOp_ == nullptr; }

  PipelineType srcPipe() const { return setOp_->GetActualSrcPipe(); }
  PipelineType dstPipe() const { return setOp_->GetActualDstPipe(); }

  const Interval &interval() const { return interval_; }
  void setInterval(Interval iv) { interval_ = iv; }

  /// How this hazard is realised. Reads the set side; the both-halves invariant
  /// keeps the wait side in step. Can disagree with `isBarrier()` -- see there.
  MECHANISM mechanism() const { return setOp_->GetMechanism(); }

  /// Route to the buffer-ID token protocol, so this hazard must NOT take an event
  /// id.
  ///
  /// THE `uselessSync` MARKS ARE NOT OPTIONAL. `SyncCodegen` dispatches on the
  /// SyncOperation TYPE, never on the mechanism, so stamping BUFID alone would
  /// still emit the set/wait pair, now with no event id. All four ops, because
  /// `SetEventIds` stamps head/tail too and those are real ops in syncIR's pipe
  /// lists; a token needs no priming pair, its counter starts free.
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

  /// Realise as a PIPE_ALL barrier when the colourer cannot serve the hazard. "No
  /// id" is not a resting state: codegen emits the pair regardless, so an unserved
  /// hazard would go out carrying id 0 or the unset sentinel.
  ///
  /// THE WAIT SIDE IS THE ONE KEPT. A PIPE_ALL at the wait position sits before the
  /// consumer; for a backward hazard the in-body wait is at the top of the body, so
  /// the barrier drains the previous iteration, which is exactly the carried edge.
  /// At the set side it would sit after the producer and order nothing across the
  /// back edge. Head and tail are suppressed for the reason given in
  /// `RouteToBufid`, and a PIPE_ALL needs no priming anyway.
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

  /// How many event ids must be live across this hazard's interval. 1 ordinarily;
  /// N for a multi-buffered (rotating) pair, where the front end supplies a slot
  /// SSA via `pto.multi_tile_get %mb[%slot]` and `eventIdNum = N`, and codegen
  /// emits `set_flag_dyn`/`wait_flag_dyn` selecting `eventIds[slot % N]`. Read
  /// from the set side; the both-halves invariant keeps the two in step.
  unsigned demand() const {
    return std::max(1u, static_cast<unsigned>(setOp_->eventIdNum));
  }

  /// Assign ids to every op the hazard owns, from ONE coloring decision: set, wait,
  /// and for a loop-carried hazard the synthesised head-set and tail-wait too.
  /// Stamping them together is what makes the head prime and the tail drain the
  /// same flag the in-body pair uses.
  ///
  /// Arity matters more than identity here: `CreateSetWaitOpForMultiBuffer` asserts
  /// `eventIds.size() == slotCount` and indexes `eventIds[i]`, so a short list is
  /// an out-of-bounds read rather than a mis-sync.
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
  // A loop-carried hazard's in-body pair (wait, then set) deadlocks on iteration 1
  // with nothing priming the flag. `synthesizeLoopCompensation` adds a head-set
  // before the carrying loop and a tail-wait after it, on the same id. The hazard
  // holds raw pointers to them; both null for a forward hazard.

  /// Attach the synthesised head-set / tail-wait. Set once, before coloring.
  void setCompensation(SyncOperation *head, SyncOperation *tail) {
    headOp_ = head;
    tailOp_ = tail;
  }
  SyncOperation *headOp() const { return headOp_; }
  SyncOperation *tailOp() const { return tailOp_; }
  bool hasCompensation() const { return headOp_ != nullptr; }

  //-- Buffer-ID cost is reuse-conditioned -------------------------------------
  //
  // A token's get/rel counters are bidirectional: ONE token discharges the forward
  // RAW and the reverse WAR. So a token beats an event only where the buffer really
  // is reused in both directions; forward-only it adds a reverse edge nobody asked
  // for. The cost needs the buffer's whole hazard group, not the hazard alone --
  // hence these two fields.

  /// Every buffer-id cluster this hazard's memory belongs to. Resolved per
  /// `BaseMemInfo *`, by pointer identity with a `MemAlias` fallback; empty when
  /// none of the hazard's tiles is buffer-id-clusterable.
  llvm::SmallVector<int, 2> bufferClusters;

  /// Does a mirror-direction hazard share a cluster with this one?
  ///
  /// CLUSTERS, NOT `depRootBuffers`. A root is the root ALLOCATION, so keying on it
  /// reports two hazards on different tiles as reverse partners merely because they
  /// share a root. That false positive is the expensive direction: it prices a token
  /// low and routes a forward-only hazard onto it.
  bool hasReverseWar = false;

  //-- The cross-arm two-cycle shape -------------------------------------------

  /// Where this hazard's wait sits after `MoveSyncState`, which is what the motion
  /// did to it. A wait is normally attached to the consuming COMPOUND; finding it on
  /// a region delimiter means it was hoisted there.
  ///
  ///   - `OutOfArm`  -- on a branch's IF_BEGIN. `MoveOutBranchSync` moved it out of an
  ///     arm, so the ordering is now paid on every path while being needed on one. A
  ///     token brackets the access inside the arm and makes that cost conditional
  ///     again. This is the case routing looks for.
  ///   - `OutOfLoop` -- on a LOOP_BEGIN. `MoveForSync` moved it out of the loop, so the
  ///     event pays once where a token would pay per iteration. Routing must refuse.
  ///
  /// The two are mutually exclusive by construction and the precedence is automatic:
  /// `MoveForSync` walks every element in a loop body INCLUDING branch delimiters
  /// (`MoveSyncState.cpp:161-163`), so a wait parked on an IF_BEGIN inside a loop is
  /// re-examined and can be hoisted onward to LOOP_BEGIN. The final position therefore
  /// already encodes "loop-refusal beats branch-win", which is the correct ranking:
  /// losing N-fold amortisation outweighs saving one not-taken path.
  ///
  /// Measured over the 260-emission corpus dump: 1440 waits on a COMPOUND, 142 on an
  /// IF_BEGIN, 90 on a LOOP_BEGIN. The 90 independently matches the hoist count derived
  /// by diffing the mover's own before/after dumps.
  enum class WaitHoist {
    None,      ///< still on its consuming compound
    OutOfArm,  ///< hoisted to IF_BEGIN by MoveOutBranchSync
    OutOfLoop, ///< hoisted to LOOP_BEGIN by MoveForSync
  };
  WaitHoist waitHoist = WaitHoist::None;

  /// This hazard presents the cross-arm two-cycle shape, so a buffer-id token is a
  /// candidate for it. Corpus population of the shape is one kernel, and that one is
  /// the redundancy-merge case the coverage gate rejects, so nothing routes today.
  ///
  /// LOOP-CARRIEDNESS IS STRUCTURAL, NOT A PER-HAZARD FLAG. The saving is that
  /// iteration N's conditional consumer is ordered against iteration N+1's
  /// unconditional producer, and that holds once the unconditional cycle sits inside a
  /// loop -- a property of the BUFFER's accesses, not of any one hazard's
  /// `GetForEndIndex()`. Testing that flag per hazard selects nothing in either
  /// direction: on the probe that validated the shape the branch-hoisted hazards are
  /// not back-edge hazards, and the one back-edge hazard is not branch-hoisted.
  ///
  /// What IS unsound is a single cycle held ACROSS a branch (get in the arm, release
  /// outside). Two independently balanced cycles sharing one id are sound, because the
  /// token balance rule is per-pipeline -- measured: no deadlock on either path, and
  /// bit-identical to the event form on both.
  bool crossArmTokenFavoured = false;

  /// The observed hoist position and `moveForSyncHoistsWait`'s prediction disagreed
  /// for this hazard, so it was refused rather than routed. Non-zero anywhere means
  /// the shared helper no longer describes what `MoveSyncState` does -- which is the
  /// exact drift the helper was factored out to make visible.
  bool crossArmHoistDivergence = false;

  /// Can this hazard be realised as a buffer-id token AT ALL? Routing must refuse a
  /// buffer carrying any hazard for which this is false: suppressing the event ops
  /// of a hazard no token can express loses the ordering outright.
  ///
  /// Four ways a hazard is inexpressible, each because the token names something
  /// different from what the hazard needs:
  ///   - SAME-PIPE. A token orders accesses to one buffer; ordering a pipe against
  ///     itself needs a barrier, which is a different class.
  ///   - NO CLUSTER. With no aliasing clique there is no token to take.
  ///   - GM, or endpoints in DIFFERENT scopes. A token brackets a tile buffer;
  ///     `BufidSyncAnalysis` excludes both shapes for the same reason.
  ///   - A PIPE THE EMITTER CANNOT ENCODE. `BufidSyncCodegen` maps a pipe to a
  ///     `SyncOpType` and fails the compilation on the ones it has no name for.
  bool tokenExpressible = false;

  /// Why this hazard's two endpoints sit at one address, when they do.
  ///
  /// A hazard whose endpoints are DISTINCT allocations at the same address in the
  /// same scope is ordered because the addresses coincide: no SSA value flows
  /// between two distinct allocations, so the ordering is invisible as dataflow.
  /// Classified by whether those allocations also share a tile type:
  ///   - `Reuse`     -- same type, one buffer holding successive values;
  ///   - `AliasView` -- different types over the same bytes, a reshape or
  ///                    transpose view of live data.
  ///
  /// INFORMATIONAL ONLY. No routing, refusal, mechanism choice or emission may read
  /// this. The type test is a heuristic: two same-typed tiles can be aliased on
  /// purpose, and the ordering between them is then a real read-after-write through
  /// memory. A decision taken on a wrong classification would drop a live ordering.
  enum class AddressSharing {
    None,      ///< one allocation on both sides, or the addresses differ
    Reuse,     ///< distinct allocations, one address, SAME tile type
    AliasView, ///< distinct allocations, one address, DIFFERING tile types
  };
  AddressSharing addressSharing = AddressSharing::None;

  /// The shared slot and the two allocations in program order. Meaningful only when
  /// `addressSharing != None`.
  uint64_t sharedAddress = 0;
  pto::AddressSpace sharedScope = pto::AddressSpace::Zero;
  Operation *sharedAllocFirst = nullptr;
  Operation *sharedAllocSecond = nullptr;

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

/// The three id pools, with no arch branch anywhere.
///
/// K IS THE ARCHITECTURE. K=0 leaves the buffer-ID class empty and the allocator
/// degenerates to event+barrier, which is A3; K=32 opens it, which is A5. There is
/// no `if (arch == a5)` in the allocator and there must never be one.
struct ResourceModel {
  /// Buffer-ID pool: ONE pool, shared across all pipes and both directions.
  unsigned bufidCapacity = 0; // K

  /// Event pool for one direction. Event flags are per-(src,dst) DISJOINT hardware
  /// -- id 0 in MTE2->V is a different flag from id 0 in V->MTE3 -- so this is a
  /// pool per direction, not a shared pool of 8. Some directions (V<->S) reserve
  /// their top id, so it is not always 8.
  ///
  /// Reads the same `SyncEventIdAllocation::GetReservedEventIdNum` the G3 gate
  /// reads, so allocator and gate cannot disagree about the bound.
  unsigned eventPoolSize(PipelineType src, PipelineType dst) const;

  /// Barrier is the spill class: unbounded and always available. That is what makes
  /// the allocator total -- there is no "no id left" failure mode, only a worse
  /// schedule.
  static constexpr bool barrierIsUnbounded = true;

  /// Admit `Hazard::crossArmTokenFavoured` as a routing reason alongside pool
  /// overflow. OFF by default: the shape's natural population in the current corpus
  /// is zero, so enabling it can only change output on a kernel deliberately written
  /// to present it, and off-by-default keeps that provable rather than asserted.
  /// `--unified-sync-route-cross-arm` turns it on.
  bool routeCrossArm = false;
};

//===----------------------------------------------------------------------===//
// alpha -- the per-hazard mechanism cost
//===----------------------------------------------------------------------===//

/// What a hazard costs on each resource class.
///
/// SHAPE ONLY: an ordering, not a calibrated model. No allocation decision reads
/// alpha -- `routeBuffers` tests `Buffer::isWrittenBack` directly, and only
/// `printSyncModel` renders these numbers.
struct Alpha {
  unsigned event = 1;
  unsigned bufid = 1;
  unsigned barrier = 100; ///< spill: serializes the core, so it must lose to any id.

  /// A token is cheap only when its reverse WAR is really wanted; forward-only it
  /// adds a spurious back edge, so it is priced above an event.
  static Alpha forHazard(const Hazard &h) {
    Alpha a;
    a.bufid = h.hasReverseWar ? 1 : 2;
    return a;
  }
};

//===----------------------------------------------------------------------===//
// Buffer -- the unit buffer-ID routing decides on
//===----------------------------------------------------------------------===//
//
// THE BUFFER, NOT THE HAZARD, IS THE ROUTING UNIT, and that is correctness rather
// than a modelling preference. The protocol is a COUNTER CYCLE: every `get_buf` is
// matched by an `rls_buf` on the same id. If some hazards on one buffer are
// realised as events while others take get/rls, the cycle is missing a step and the
// acquire blocks forever. So routing is decided per buffer, before any hazard on it
// takes an event id.
//
// The reservation window is the UNION SPAN, matching
// `BufidSyncIdAlloc::computeLifeIntervals`: an access outside a loop contributes
// its own `syncIRIndex`, one inside contributes the whole `[loopBegin, loopEnd]`,
// and the result is min-start/max-end per logic id.
//
// Over-reserving is WASTEFUL, NOT WRONG, which is why a buffer with no single loop
// scope gets a wide window instead of being refused. The get_buf/rls_buf ops are
// emitted at the access sites regardless, and the window only decides which
// physical id a logic id maps to, so a window that is too long can lose an id but
// never an ordering -- and the exhaustion path coalesces logic ids rather than
// failing.
struct Buffer {
  /// The aliasing-clique id (`bufferClusters` entry). One token covers one clique.
  int logicId = -1;

  /// Does anything write this buffer back?
  ///
  /// NOT THE SAME PREDICATE AS `Hazard::hasReverseWar`, and the difference is
  /// deliberate. Per-hazard asks whether a mirror-direction hazard shares a cluster
  /// with one hazard; per-buffer asks whether the buffer carries a mirror pair at
  /// all, so a third hazard in an unmirrored direction is true here and false
  /// there. Per-buffer is the correct one: the token is a property of the buffer,
  /// and since routing is all-or-nothing a per-hazard predicate could let two
  /// hazards on one buffer disagree, which is the split that hangs.
  bool isWrittenBack = false;

  /// Union of the hazards' own intervals, in SyncIR-index space: a forward hazard
  /// contributes its tight [set, wait) range, a loop-carried one its whole carrying
  /// loop. `bufidLoop` is the reservation derived from these.
  Interval accessRange{0, 0};

  /// Any access inside a loop.
  bool isLoopCarried = false;

  /// Accesses live in more than one loop context (inside and outside, or in two
  /// different loops), so there is no single well-defined carrying loop.
  bool spansLoopBoundary = false;

  /// How many distinct loop contexts the accesses occupy (function level counts
  /// as one). 1 means a single well-defined scope.
  unsigned loopContextCount = 0;

  /// The reservation window. Union span -- an access outside a loop contributes
  /// itself, an access inside contributes its whole loop.
  Interval bufidLoop{0, 0};

  /// Routing outcome. `true` means every hazard touching this buffer is to be
  /// realised as a buffer-ID token, and NONE of them may take an event id.
  bool routedToBufid = false;

  /// Peak per-direction overlap predicted for the directions this buffer touches
  /// (the maximum over them), and the event pool of the direction that peak came
  /// from -- not the minimum pool. Routing compares these; recorded so the decision
  /// is auditable rather than a bare bool.
  unsigned predictedPeakOverlap = 0;
  unsigned smallestPool = 0;

  /// False means the reservation window could not be resolved. Such buffers are
  /// COUNTED in the report rather than defaulted to a plausible-looking interval: a
  /// wrong window is a wrong reservation, which is a missing get/rls, which is a
  /// hang.
  bool bufidLoopDefined = false;
};

/// A (direction, id) pair the pto-isa library occupies INSIDE a macro call, over
/// that call's span. The id is not chosen here, it is already taken, so the
/// colourer must treat it as occupancy rather than as a candidate.
struct HiddenReservation {
  int srcPipe = -1;
  int dstPipe = -1;
  int id = -1;
  Interval span;                 ///< SyncIR indices, already padded
  llvm::StringRef macroName;     ///< for diagnostics only
};

/// The hazard set, plus the resources it must be colored into.
struct SyncModel {
  llvm::SmallVector<Hazard> hazards;
  /// One entry per buffer-id cluster touched by any hazard, sorted by logicId.
  llvm::SmallVector<Buffer> buffers;
  ResourceModel resources;
  /// Ids already spoken for by a macro's library implementation. Empty on any
  /// kernel with no macro ops, which is why this costs nothing to consult.
  llvm::SmallVector<HiddenReservation> hiddenReservations;
};

/// Build the model from the reused front end's output. `SyncOperations`' OUTER
/// index is already the hazard -- one group per ordering, holding its set and its
/// wait -- which is the granularity the both-halves invariant needs, and why
/// `Hazard` can own the pair without re-deriving the pairing.
///
/// `memInfoToClusters` joins a hazard's memory to its cluster(s); an empty map is
/// legal and simply leaves every hazard unclustered with `hasReverseWar` false.
///
/// Assigns no ids, emits no IR, and does not mutate the SyncOperations. `syncIR` is
/// read only to resolve a loop-carried hazard's CARRYING loop:
/// `setOp->GetForEndIndex()` indexes a `LoopInstanceElement` whose
/// `[beginId, endId + 1)` is that hazard's interval -- the extra index covers the
/// tail-wait that sits at endId -- read forward, never inverted.
SyncModel buildSyncModel(const SyncOperations &syncOps, unsigned bufidCapacity,
                         const MemInfoToClusters &memInfoToClusters,
                         const SyncIRs &syncIR);

/// Synthesise the head-set / tail-wait pair for every loop-carried
/// (`GetForEndIndex`) hazard, so its in-body backward pair is primed before the
/// carrying loop and drained after. Without them the kernel deadlocks on iteration
/// 1; this is not an optimisation. Returns the number of pairs synthesised.
///
/// Appends ONE `SyncOperations` group per such hazard and links both ops onto the
/// hazard, so one `SetEventIds` stamps all four with one id. Exactly one head and
/// one tail per hazard.
///
/// Also PLACES each pair into `syncIR` -- head into the `pipeBefore` of the anchor
/// `hoistAnchorFor` picks (the carrying loop's begin, or an earlier op in that
/// loop's own enclosing region when one is legal), tail into the loop end's
/// `pipeAfter`, with `push_front` so the
/// tail precedes any existing loop-end set. `SyncCodegen` walks those lists, so
/// owning the pair in `syncOps` alone would let emission drop it silently. Hence
/// `syncIR` is mutable here.
///
/// Runs after `buildSyncModel` and before `colorEventIds`.
unsigned synthesizeLoopCompensation(SyncModel &model, SyncOperations &syncOps,
                                    SyncIRs &syncIR);

/// Deterministic, diff-friendly rendering (one hazard per line).
void printSyncModel(llvm::raw_ostream &os, llvm::StringRef funcName,
                    const SyncModel &model);

//===----------------------------------------------------------------------===//
// Event interval-coloring allocator
//===----------------------------------------------------------------------===//

/// Result of the event-id coloring.
///
/// Two conventions that read wrong at d > 1: `reused` counts ID ASSIGNMENTS, so one
/// hazard of demand d can add up to d, and `assigned` is per-HAZARD while
/// `idsAssigned` is per-ID.
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
                         ///< direction
};

/// Pre-colour the ids a macro's library implementation consumes internally.
/// Returns the number of reservations recorded.
///
/// Runs after `buildSyncModel` and BEFORE `colorEventIds`: the colourer must see
/// these as occupied, or it hands one to a compiler hazard whose interval spans the
/// call and the library's wait is satisfied by the wrong producer.
///
/// The span runs from the macro's first phase to its last, padded one SyncIR index
/// each side. The pad is load-bearing: without it a hazard ending exactly where the
/// call begins reads as disjoint and the collision survives.
///
/// DELIBERATELY NOT SHARED with the oracle's own reservation walk, which re-derives
/// the same spans. Sharing would hide a bug in the span arithmetic from the gate
/// that exists to catch it.
unsigned seedHiddenMacroEvents(SyncModel &model, const SyncIRs &syncIR);

/// Per-(src,dst) greedy first-fit interval colouring -- the classic left-edge
/// algorithm. Event flags are per-direction DISJOINT hardware, so each direction is
/// an independent interval graph.
///
/// A multi-buffered hazard needs d ids live across its interval, so this is WEIGHTED
/// colouring and the bound is the peak SUM of demands over a point rather than the
/// peak count of intervals. That holds only because the d ids need NOT be
/// contiguous: `CreateSetWaitOpForMultiBuffer` materialises each `eventIds[i]` as an
/// independent constant selected by `slotMod == i`, and the dyn ops take the id as a
/// runtime operand that lowering passes straight through. Were contiguity required
/// this would be colouring with BANDWIDTH, which is NP-hard.
///
/// A hazard takes EITHER all d ids OR none: fewer than d emits a selector chain
/// indexing past `eventIds`, which is an out-of-bounds read rather than a mis-sync,
/// so a short pool spills the whole hazard to a barrier. Structurally-lone barriers
/// are skipped -- they hold no id.
///
/// Deterministic: hazards are ordered by (direction, interval start, hazard index)
/// and take the lowest free ids, except that the most recently freed id is skipped
/// while another free id exists -- an event flag is one signal deep, so alternating
/// between two ids restores a step of producer run-ahead.
///
/// A rotating id is a RUNTIME value, recorded as -1 in the IR view, so G2/G3 cannot
/// pair a rotating set with its wait there; the syncops view is the only place those
/// ids are checked.
EventColorResult colorEventIds(SyncModel &model);

//===----------------------------------------------------------------------===//
// Buffer routing
//===----------------------------------------------------------------------===//
//
// The order is a correctness requirement, not a convention: settle the operations,
// then route buffers, then colour what remains. Routing is per-buffer and
// ALL-OR-NOTHING -- a split buffer is a HANG, not a worse schedule.
//
// Route b iff  K > 0  &&  b overflows its hazards' event pool  &&  b.isWrittenBack
//
//   (1) The write-back test is the per-BUFFER `Buffer::isWrittenBack`, never
//       per-hazard `Hazard::hasReverseWar`. Routing is all-or-nothing, so the
//       predicate must be a buffer-level fact; a per-hazard one lets two hazards on
//       one buffer disagree, which is exactly the split that hangs.
//
//   (2) "OVERFLOWS ITS POOL" is a PREDICTION -- colouring has not run, so there is
//       no overflow to observe. For each direction this buffer's hazards
//       participate in, take the peak simultaneous overlap over ALL hazards in that
//       direction -- the same quantity the left-edge colourer will hit -- and
//       compare against that direction's pool.
//       THE TEST IS STRICT (`omega > pool`), deliberately under-predicting at the
//       boundary, because the fallbacks are asymmetric. A buffer left on events
//       that then overflows spills to a barrier: correct, just slower. A buffer
//       routed unnecessarily consumes K, which is ONE pool shared across every pipe
//       and direction and whose own exhaustion path coalesces logic ids and
//       serialises co-consumers. At omega == pool the colouring fits exactly, so
//       routing would be pure waste.
//
//   (3) `spansLoopBoundary` IS RECORDED, NOT CONSULTED. It over-flags, because loop
//       context is derived from hazard interval endpoints and a loop-carried
//       hazard's interval IS its loop. Skipping buffer-ID on it would skip more
//       than the evidence supports and cancel the routing the overflow test just
//       asked for.
struct BufferRouteResult {
  unsigned buffersConsidered = 0;
  unsigned routed = 0;
  unsigned hazardsCovered = 0;   ///< hazards belonging to a routed buffer
  unsigned skippedNoOverflow = 0;
  unsigned skippedNotWrittenBack = 0;
  unsigned skippedNoCapacity = 0; ///< K == 0
  /// Buffers refused because a hazard on them is not `tokenExpressible`. Refusing
  /// is the whole point: routing such a buffer suppresses events no token replaces.
  unsigned skippedNotExpressible = 0;
  /// Buffers routed for the CROSS-ARM reason rather than because their direction
  /// overflows its pool. Zero unless a kernel actually presents the shape, which no
  /// corpus kernel currently does.
  unsigned routedCrossArm = 0;
  /// Buffers that presented the cross-arm shape but were refused because some hazard
  /// on them had its wait hoisted out of a LOOP. Counted rather than silently
  /// dropped: this is the loss-making direction of the same asymmetry, and a silent
  /// refusal here would look identical to "the shape was never there".
  unsigned skippedCrossArmLoopHoist = 0;
  /// Cross-arm candidates refused because the observed hoist position and the
  /// `moveForSyncHoistsWait` prediction disagreed. MUST BE 0: a non-zero count means
  /// `MoveSyncState` and the shared helper have drifted apart, and the routing
  /// predicate is no longer reasoning about the motion that actually happened.
  unsigned crossArmHoistDivergence = 0;
  /// ALL-OR-NOTHING violations: a hazard that touches BOTH a routed and an
  /// unrouted buffer, so it cannot be wholly on one mechanism. Must be 0.
  unsigned splitHazards = 0;
};

/// Decide, per buffer, whether buffer-ID backs it. Sets `Buffer::routedToBufid`
/// and records the split check. Assigns NO ids and stamps NO mechanisms.
BufferRouteResult routeBuffers(SyncModel &model);

} // namespace unified
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_INSERTSYNC_UNIFIEDSYNCMODEL_H
