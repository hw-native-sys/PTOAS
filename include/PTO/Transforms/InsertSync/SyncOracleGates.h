// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncOracleGates.h - Oracle correctness gates -------------*- C++ -*-===//
//
// Correctness gates over the sync a compilation emitted, read through
// SyncOracleExtract.h. Each judges one compilation on its own -- there is no
// reference file and nothing is compared against another run:
//
//   G3       device-id legality: every id is one the hardware and the compiler's
//            own reservations permit, for that op, direction and arch.
//   G2       non-interference: no two overlapping intervals share an id.
//   G1-self  every dependency `DepBetween` reports is ordered by emitted sync.
//
// G3's rules, and why each is not already covered elsewhere:
//
//   R1 event id range. Intra-core set/wait ids live in [0, kTotalEventIdNum).
//       The IR spelling cannot express a violation (`pto.set_flag`'s event_id is
//       an EVENT_ID0..EVENT_ID7 *enum*), but the in-memory `SyncOperation::
//       eventIds` is a plain SmallVector<int> that an allocator writes freely.
//       That is the surface this rule guards.
//
//   R2 per-direction reservation. `SyncEventIdAllocation::reservedEventIdNum`
//       reserves the tail of the pool for some directions (today: V<->S reserve
//       one id, so only [0, 7) is available there). Representable in real IR,
//       so this gate checks it.
//
//   R3 block-sync-all reserved ids. When a SYNC_BLOCK_ALL exists, ids
//       kBlockSyncAllCubeEventId (14) / kBlockSyncAllVectorEventId (15) belong
//       to it alone and the block set/wait pool shrinks to
//       [0, kBlockSyncSetWaitEventIdNum - kReservedBlockSyncEventIdNum).
//       NOTE: nothing in this build emits a block-sync op, so this rule is
//       unexercised. It is encoded because its id space is where a stomp appears.
//
//   R4 buffer id range. Buffer ids must lie in [0, 31]. Checked on the emitted
//       ops, a second surface after the op verifier: the pre-codegen record type
//       cannot represent a buffer id at all.
//
//   R5 buffer id arch capacity. Buffer-id sync is exposed only where K > 0
//       (K=0 on A3, K=32 on A5), so a buf-id op with buf_id >= K for the arch
//       being compiled is rejected. K reaches the gate from the caller.
//
// Barriers carry no id; an id on a barrier is an allocator bug, so it is
// reported too.
//
// Ops whose ids live in a *different* id space are deliberately out of scope:
// `pto.sync.set` / `pto.sync.wait` carry a cross-core FFTS/intra-block flag id
// (lowered through `createFFTSMsg` / `set_intra_block`), not an intra-core event
// id -- the in-tree corpus legitimately uses values such as 17 there.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCORACLEGATES_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCORACLEGATES_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/Transforms/InsertSync/SyncOracleExtract.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace pto {
namespace oracle {

/// Everything G3 needs to know about the target of one compilation.
struct DeviceIdLimits {
  /// Size of a per-(src,dst) intra-core event pool.
  unsigned intraCoreEventIdNum = kTotalEventIdNum;
  /// Size of the block-sync set/wait pool, before block-all reservation.
  unsigned blockSyncEventIdNum = kBlockSyncSetWaitEventIdNum;
  /// K: buffer ids this arch exposes. 0 on A3, 32 on A5.
  unsigned bufIdCapacity = 0;
  /// Width of the hardware buf_id field; mirrors `verifyBufSyncOp`'s [0, 31].
  unsigned bufIdHwMax = 32;
};

enum class IdViolationKind {
  EventIdOutOfRange,
  EventIdReservedForDirection,
  EventIdReservedForBlockAll,
  BlockAllEventIdWrong,
  BarrierCarriesEventId,
  BufIdOutOfRange,
  BufIdExceedsArchCapacity,
  EventIdCollidesWithMacroHidden,
  /// A (direction, event id) has a different number of `set_flag` and `wait_flag`
  /// ops in the emitted IR. Checked on the EMITTED ops because a pre-codegen count
  /// includes syncs the emitter may still drop: `MergeSyncList` discards a sync that
  /// `IsSyncExist` already finds in the destination list, so two indistinguishable
  /// ops merged onto one element leave a wait with no set -- a hang that counting
  /// before codegen cannot see.
  EventSetWaitUnbalanced,
};

llvm::StringRef idViolationKindName(IdViolationKind kind);

struct IdViolation {
  IdViolationKind kind;
  unsigned order; // program order (IR view) / sync index (SyncOperation view)
  std::string opName;
  int64_t id;
  std::string detail; // human-readable statement of the rule that was broken
};

/// G3 over the emitted IR.
llvm::SmallVector<IdViolation>
checkDeviceIdLegality(llvm::ArrayRef<IRSyncRecord> records,
                      const DeviceIdLimits &limits);

/// G3 over the pre-codegen SyncOperation set -- the ids an allocator just wrote.
llvm::SmallVector<IdViolation>
checkDeviceIdLegality(llvm::ArrayRef<SyncOpRecord> records,
                      const DeviceIdLimits &limits);

/// Deterministic rendering. `view` is "ir" or "syncops".
void printIdViolations(llvm::raw_ostream &os, llvm::StringRef funcName,
                       llvm::StringRef view, size_t recordCount,
                       llvm::ArrayRef<IdViolation> violations);

//===----------------------------------------------------------------------===//
// G2: non-interference
//===----------------------------------------------------------------------===//
//
// Coverage alone cannot see this class of bug. An allocation can order every
// hazard it is supposed to order (G1 green) and still be wrong, because two
// hazards whose lifetimes overlap were handed the SAME id: the first hazard's
// wait can be satisfied by the second hazard's set, and the ordering it was
// meant to establish silently evaporates, and the op count is unchanged, so
// nothing that counts ops can see it either.
//
// The invariant, which generalizes `BufidSyncIdAlloc::validateNoSamePhysicalIdNesting`
// from buffer ids to every id class:
//
//     For a given id key, the intervals that use that id must be pairwise
//     disjoint -- never nested, never overlapping.
//
// The key differs per id class, and getting it wrong makes the gate either
// blind or wrong:
//
//   * event ids: key = (srcPipe, dstPipe, eventId). Event flags are per-
//     (src,dst)-direction DISJOINT hardware, so EVENT_ID0 may be live in
//     MTE2->V and in V->MTE3 simultaneously. Keying on the id alone would raise
//     a false violation on correct code.
//   * buffer ids: key = bufId alone. The buffer-id pool is global across pipes,
//     matching validateNoSamePhysicalIdNesting, which keys on physicalId only.
//
// Implementation is a linear scan in program order keeping a stack of open intervals
// per key: a set/get pushes and reports any interval already open on that key, a
// wait/rls pops, a pop on an empty stack is a close without an open, and anything
// left on the stack at the end is a leaked id.
//
// A loop-carried pair -- a `wait` satisfied by the PREVIOUS iteration's `set`, so
// the wait precedes the set in program order -- surfaces as `wait-without-set`
// plus `unclosed`. The sync pass makes such a hazard work by priming the flag
// before the loop and draining it after, so on well-formed output neither kind
// fires. An unprimed loop-carried pair blocks forever on iteration 1, and these
// two kinds are the only thing that detects it. Do not weaken them: if cyclic
// matching is ever added it must DISTINGUISH a primed backward pair from an
// unprimed one, not accept both.

// --- Buffer-ID legality ---------------------------------------------------
//
// These are HARDWARE rules, not design choices: violating them desynchronises
// the id's global get/rel counters, and a get_buf that can never pop HANGS the
// core. They are therefore checked as part of G2, and they are immune to any
// allocator-design decision -- which is why this gate can be built before the
// allocator exists.
//
//   B1 no re-acquire of a live id. Covers BOTH the same-pipe case
//      same-id nesting: get,get,rel,rel double-increments the counter so a get
//      may never pop) AND the cross-pipe same-id interleave (
//      get(MTE2,#0) get(V,#0) rel(MTE2,#0) rel(V,#0) is illegal). Both surface
//      as "this id was opened twice before closing".
//   B2 no rls_buf without a live get_buf.
//   B3 no get_buf left unreleased.
//   B4 pipe-matched pairing -- a rls_buf closes a get_buf on the SAME pipe. The
//      token brackets one op on one pipe; a cross-pipe close is not a pair.
//   B5 an id must span exactly 2 distinct pipes: buffer-ID is
//      ONLY for sync between two different pipes; same-pipe ordering needs
//      bar.pipe. A single-pipe id synchronises nothing and burns an id.
//   B6 an id SHOULD NOT span more than 2 pipes ("not
//      recommended" -- soft). WARNING, not an error, and deliberately so: the
//      shipping BufidSync pass already emits one id across MTE2/V/MTE3 on real
//      kernels (measured). Making it an error would fail on known-good output.
//      It is reported because it is exactly the coalescing bound the unified
//      allocator has to decide about.
//   B7 a get_buf and its rls_buf must sit in the SAME block. A pair split
//      across if/else arms would be matched by any flat program-order walk, yet
//      at run time only one arm executes -- the counters desync (the ASC
//      alloc/free-balance invariant). Shipping BufidSync anchors both ops on the
//      same op, so this holds by construction there.
//
// MODE. get_buf/rls_buf carry a `mode` operand. Mode 0 is the bidirectional
// token -- the only mode PTOAS emits -- and the strict rules above apply to it.
// Non-zero mode is the directional variant (set_flagV2/wait_flagV2), which the
// spec explicitly frees from strict pairing; it is a deferred mechanism, so the
// pairing rules are NOT applied to it and its presence is reported instead. The
// gate keys on `mode` from day one so that mechanism is additive, not a rewrite.

enum class InterferenceKind {
  EventIdNested,
  EventIdWaitWithoutSet,
  EventIdUnclosed,
  /// A set_flag and the wait_flag closing it sit under DIFFERENT conditionals, so
  /// some path runs one without the other: a wait with no set hangs, a set with no
  /// wait leaks the id. Loop nesting is deliberately not compared -- a set outside a
  /// loop priming a wait inside it is the intended idiom, and the zero-trip case is
  /// checked on the loop spine instead.
  EventIdPairGuardMismatch,
  BufIdNested,                 // B1
  BufIdReleaseWithoutAcquire,  // B2
  BufIdUnclosed,               // B3
  BufIdPipeMismatch,           // B4
  BufIdSinglePipe,             // B5
  BufIdTooManyPipes,           // B6 (warning)
  BufIdPairSplitAcrossBlocks,  // B7
  BufIdDirectionalModeUnsupported, // mode != 0 (deferred mechanism)
};

/// A warning is reported but does not fail the gate. Used only where the spec
/// itself says "not recommended" rather than "illegal" -- see B6.
enum class Severity { Error, Warning };

llvm::StringRef interferenceKindName(InterferenceKind kind);
llvm::StringRef severityName(Severity severity);

struct InterferenceViolation {
  InterferenceKind kind;
  Severity severity = Severity::Error;
  unsigned order;     // program order of the offending op
  unsigned openedAt;  // program order of the still-live interval, when relevant
  std::string opName;
  std::string key; // rendered id key, e.g. "MTE2->V id=0" or "buf_id=0"
  std::string detail;
};

/// G2 over the emitted IR.
llvm::SmallVector<InterferenceViolation>
checkNonInterference(llvm::ArrayRef<IRSyncRecord> records);

/// KNOWN LIMITATION: G2 does not check ROTATING (multi-id) hazards at all.
/// `set_flag_dyn` / `wait_flag_dyn` carry a runtime id, recorded as -1, so those ops
/// never enter the id check and a collision between two rotating hazards would not be
/// reported. The ids are recoverable statically -- the selector is an N-way
/// `arith.select` chain over constants -- so closing this is mechanical.

/// Number of Error-severity violations (warnings do not fail the gate).
unsigned countErrors(llvm::ArrayRef<InterferenceViolation> violations);

void printInterferenceViolations(
    llvm::raw_ostream &os, llvm::StringRef funcName, llvm::StringRef view,
    size_t recordCount, llvm::ArrayRef<InterferenceViolation> violations);

//===----------------------------------------------------------------------===//
// The happens-before edge model
//===----------------------------------------------------------------------===//
//
// `computeCoverage` derives the orderings a function's emitted sync establishes.
// G1-self consumes it: a dependency is covered when the model orders its endpoints.
//
// Anchors are the ops carrying `OpPipeInterface` and a read/write memory effect,
// numbered in program order; sync ops are never anchors. An edge (i, j) means the
// emitted sync orders anchor i before anchor j, and edges come only from emitted
// sync:
//
//     barrier <PIPE_ALL> at slot s   =>  all i < s, all j >= s
//     barrier <X> at slot s          =>  i < s on pipe X, j >= s on pipe X
//     set_flag(P,Q,e)@s / wait@w     =>  i < s on pipe P, j >= w on pipe Q
//     rls_buf(b)@r on P, get_buf(b)@g>r on Q
//                                    =>  i < r on pipe P, j >= g on pipe Q
//
// The last rule is the buffer token's happens-before: `get_buf` acquires what the
// previous holder released. Without it a buffer-id allocation appears to order
// nothing and every dependency on it reads as uncovered.
//
// Program order is NOT an edge, not even within one pipe: `pto.barrier` is an
// intra-pipeline barrier, so crediting program order would credit orderings nothing
// established. Two exceptions, both narrow:
//   - two accesses to the SAME tile buffer are ordered by program order, so nothing
//     is owed for such a pair on any arch;
//   - A5 additionally guarantees PIPE_V intra-pipe ordering, which is why
//     `SyncCodegen::CreateBarrierOp` emits nothing for a V->V pair there and the
//     backend rejects one. A2 and A3 have no such guarantee.
//
//===----------------------------------------------------------------------===//
// THE ITERATION DIMENSION -- loop-carried (backward) orderings.
//===----------------------------------------------------------------------===//
//
// The forward model is a flat sequence, so "iteration k before iteration k+1" is
// inexpressible in it, not merely filtered by the `i < j` guard. A second,
// distance-annotated edge class is therefore carried alongside:
//
//   CarriedEdge{from=j, to=i, distance=1}
//       "anchor j in iteration k happens-before anchor i in iteration k+1"
//
// Three rules build it, kept strictly separate:
//
//   RULE 1 (detect). Within ONE loop body, match set/wait (and rls/get) forward
//   per key; whatever is LEFT OVER wraps, so an unmatched in-body set pairs with an
//   unmatched in-body wait at a lower slot. The head/tail compensation ops take no
//   part -- they sit outside the body, so a correct kernel and one missing its
//   compensation are detected identically. The forward pass already consumes the
//   compensation as two forward pairs, which is why bidirectional pairing alone
//   would establish nothing here.
//
//   RULE 2 (generate). For a carried pair (set@p, wait@q), emit (x, y, d=1) for
//   body anchors x < p on the src pipe and y >= q on the dst pipe. BOTH endpoints
//   are restricted to the carrying loop's body: a prologue anchor has no
//   iteration index, so a d=1 edge touching one would be meaningless.
//
//   RULE 3 (credit). An EVENT carried pair earns its edges only if PRIMED:
//     P1' a set_flag H on the same (srcPipe, dstPipe, id) lies in a block on L's
//         ANCESTOR SPINE -- the block holding L, or the block holding one of L's
//         ancestor ops. REACHABILITY, not nesting: a head under an `scf.if`, or in
//         a possibly-zero-trip sibling loop, may never run, and `scf.if` is not a
//         LoopLikeOpInterface so a nesting test cannot see it.
//     P2  H precedes L in program order.
//     P3  H is still live at loop entry -- no wait_flag on the same key lies
//         outside L between H and L.
//   An unprimed backward pair establishes nothing: iteration 1's wait blocks
//   forever. Rule 1 alone cannot tell a correct kernel from a deadlocking one,
//   since both carry the in-body pair; credit is what discriminates.
//
//   RULE 3b. The same reachability test on the in-body side: the carried producer,
//   consumer and any covering barrier must be DIRECT children of L's body, or
//   iterations 2..N have nothing to wait on.
//
//   BARRIER COVERAGE. A candidate may serialize instead of pairing, leaving Rule 1
//   no pair to find. A barrier at body slot `s` covers (x, y, d=1) iff
//   `x < s || y >= s`; the only uncovered case is the wrap gap `y < s <= x`.
//
//   BUFFER TOKENS ARE EXEMPT FROM RULE 3: the ticket lock starts free, so
//   iteration 1's `get_buf` needs no priming op.
//
// Only the HEAD set is required, not the tail wait. A missing drain is a leaked
// flag, which is G2's `event-id-unclosed`; demanding it here would duplicate G2.
//
// KNOWN GAPS. None can produce a false PASS -- each is either a false FAILURE on a
// shape no in-tree mode emits, or a weakening shared by both sides:
//   - PRIME ID MISMATCH: a compensation priming with a different id than the
//     in-body pair loses credit. Both share one id today; rotation must defuse it.
//   - PEELED FIRST ITERATION: needs no priming, but P1' still demands one. Nothing
//     in-tree peels. A guarded PRIME is a different shape and is correctly rejected.
//   - LIVENESS IS A LINEAR SCAN, not dataflow: a consuming wait inside a not-taken
//     `scf.if` still counts, so credit can be withheld from a primed pair.
//     Conservative by construction.
//   - NESTED-LOOP PRODUCER: a carried pair whose producer sits in a nested loop is
//     attributed to that loop, so an outer stranded consumer finds no partner. Both
//     sides lose it equally, so it weakens the gate rather than inverting a verdict.
//
// ROTATING PAIRS are modelled, in `computeCoverage`, on both sides: a dyn pair
// contributes a CARRIED edge at distance N recovered from the selector's own
// `arith.remui` modulus, and no forward edge (within one iteration the set and wait
// touch different ids). Demand is likewise at distance N, from
// `baseAddresses.size()`, because with N slots iteration k+1 touches a different
// slot and the real conflict is at reuse. WIDEN, NEVER DROP: the forward precedent
// `isForwardDepDroppableBySlotAffine` drops a slot-disjoint dep, which is wrong on
// the back edge -- it would excuse the dependency and let an unrotated kernel pass.
// The closure is capped at 32; above the cap a composition falls to UNCOVERED.

struct CarriedEdge {
  unsigned from = 0;
  unsigned to = 0;
  unsigned distance = 1;
  bool operator<(const CarriedEdge &o) const {
    return std::tie(from, to, distance) < std::tie(o.from, o.to, o.distance);
  }
  bool operator==(const CarriedEdge &o) const {
    return from == o.from && to == o.to && distance == o.distance;
  }
};

struct CoverageProfile {
  unsigned anchorCount = 0;
  /// Sorted, unique. `first < second`, both anchor indices.
  llvm::SmallVector<std::pair<unsigned, unsigned>> edges;
  /// Sorted, unique. Loop-carried orderings; `from`/`to` are the same global
  /// anchor indices as `edges`, and `from < to` is NOT implied.
  llvm::SmallVector<CarriedEdge> carriedEdges;
};

/// Walks `func` and derives the happens-before edges its emitted sync induces.
///
/// `anchorIndex`, when non-null, receives the Operation* -> anchor-index map
/// built by the same walk. G1-self needs it to map a dependency's endpoints into
/// this coordinate system, and taking it from here rather than rebuilding it is
/// what guarantees the two views cannot drift.
CoverageProfile
computeCoverage(func::FuncOp func,
                llvm::DenseMap<Operation *, unsigned> *anchorIndex = nullptr);

//===----------------------------------------------------------------------===//
// G1-self: ABSOLUTE coverage against the dependency analysis
//===----------------------------------------------------------------------===//
//
// G1 proper is DIFFERENTIAL: it compares the candidate against the reference's
// coverage, so a green result can only ever mean "matches the reference", and its
// strength is bounded by that reference. G1-self is ABSOLUTE: it asserts that
// every dependency the FRONT-END dependency analysis reports has an emitted sync
// ordering its source before its destination.
//
// WHERE THE EXPECTED SET COMES FROM, and why it is not circular. The expected set
// is built by running `PTOIRTranslator` (def/use extraction and pipe assignment)
// and then calling `MemoryDependentAnalyzer::DepBetween` over the resulting
// `CompoundInstanceElement::defVec` / `useVec` DIRECTLY. Nothing in it reads
// InsertSync's decisions: not which hazards it chose to sync, not the mechanism
// it picked, not the ids it allocated, not its emitted ops. In particular this
// check deliberately does NOT call `InsertSyncAnalysis::IsMemInfoHasDependency`,
// which layers InsertSync's POLICY on top of DepBetween (the tload->tload WAW
// exemption, the ACC read/read special case). Policy is what this gate audits,
// so it cannot be part of the oracle.
//
// HONEST LIMIT OF "ABSOLUTE". G1-self shares the front end -- def/use
// extraction and the alias analysis -- with InsertSync, because that IS the
// dependency side. So it cannot catch a dependency the front end itself fails to
// see (an alias `MemAlias` misses). It is independent of InsertSync's SYNC
// decisions, which is what makes it non-circular; it is not independent of the
// memory model. Claim accordingly.
//
// SELF-PAIRS ARE ENUMERATED SEPARATELY. The main pair loop is `for j = i + 1`, so
// it never pairs a compound with itself; a same-op loop-carried hazard -- op x in
// iteration k against the same op in k+1 -- is asked about by a second loop, which
// records the three hazard classes as `waw`/`raw`/`war/carried-self`. Only ops
// inside a loop are considered: without a back edge there is no second execution.
//
// Coverage answers these without special-casing. A same-pipe body barrier records a
// carried edge over every (x, y) anchor pair in its loop span subject to
// `x < slot || y >= slot`, which is trivially true when x == y, so the barrier
// contributes a carried SELF-edge at distance 1. Nothing else seeds the closure
// diagonal -- every cell starts at `kUnreached` -- so a self-pair is covered
// exactly when some emitted op orders the op against its own next execution.
//
// ONE LIMIT ON THAT, and it matters when reading a green result: coverage is keyed
// on the anchor pair alone, with no loop-nest attribution. A distance-1 self-edge
// contributed by an ENCLOSING loop's back edge therefore credits a dependence
// carried by an inner loop, so the diagonal can read covered whether or not
// anything orders the op one nest level down.
//
// COVERED, per mechanism. The covering relation is `c subset-of h`: the sync's
// interval must be CONTAINED in the dependency's. Reusing `computeCoverage`
// gives exactly that for all three mechanisms, which is the main reason to reuse
// it rather than re-derive:
//   event     set@s / wait@w  =>  edge (i, j) for i < s, j >= w. So the edge
//             exists iff [s, w) lies inside (i, j] -- containment, not mere
//             co-occurrence.
//   barrier   PIPE_ALL at s   =>  edge (i, j) for i < s <= j: a barrier strictly
//             between the two endpoints.
//   buffer-ID rls(b)@r, get(b)@g>r => edge (i, j) for i < r, j >= g. This is the
//             TOKEN PROTOCOL -- release before acquire ON THE SAME buf_id -- not
//             co-location. A check blind to the protocol would bless a
//             split-endpoint hazard that is silently unordered.
//
// TRANSITIVITY IS TAKEN, and it must be. Happens-before is transitive, but
// `computeCoverage`'s edge set is not closed: pipe filtering means a chain
// a(PIPE_A) -> b(PIPE_B) -> c(PIPE_C) is covered by two pair-wise syncs while the
// direct edge (a, c) is absent, since neither sync matches both endpoints' pipes.
// Checking against the raw set would report a false "uncovered" there. So the
// relation is closed before checking, with distances composing as
// 0+0=0, 0+1=1, 1+0=1, and 1+1 DROPPED (distance 2 is not represented, and
// dropping is the conservative direction).
//
// THE MAPPING, and what happens to what it cannot map. Dependencies live over
// `BaseMemInfo`, which carries no `Operation*`; coverage lives over anchor
// indices. The bridge is `CompoundInstanceElement::elementOp` -> the anchor index
// assigned by `computeCoverage`'s walk. A dependency whose endpoint op is not an
// anchor (no `OpPipeInterface`, or no read/write memory effect) CANNOT be
// expressed in the coverage coordinate system. Those are counted and reported as
// `unmappable`, never silently dropped -- silent drops are exactly how this check
// would become vacuous.

struct SelfCoverageViolation {
  unsigned srcAnchor;
  unsigned dstAnchor;
  unsigned distance; // 0 = same iteration, 1 = loop-carried
  std::string hazard;  // raw | war | waw
  std::string srcName;
  std::string dstName;
  std::string srcPipe;
  std::string dstPipe;
};

struct SelfCoverageReport {
  unsigned dependencies = 0;   // total requirements derived from DepBetween
  unsigned covered = 0;
  unsigned carriedDeps = 0;    // of those, how many are loop-carried (d=1)
  unsigned carriedCovered = 0;
  unsigned unmappable = 0;     // endpoint has no anchor representation
  unsigned armExcluded = 0;    // (a2) pairs on mutually exclusive scf.if arms
  /// (a1) A5 PIPE_V->PIPE_V: ordered by the target guarantee the codegen relies
  /// on when it declines to emit a barrier. NOT "A5 same-pipe".
  unsigned archGuaranteed = 0;
  /// (a3) PIPE_S->PIPE_S. Counted apart from `archGuaranteed` because no hardware
  /// guarantee is claimed for it -- see the exemption site for what it does rest on.
  unsigned pipeSelfOrdered = 0;
  unsigned uncoveredSamePipe = 0;   // diagnosis split: both endpoints one pipe
  unsigned uncoveredCrossPipe = 0;  // the interesting ones
  unsigned compounds = 0;      // SyncIR compounds seen (consistency check)
  llvm::SmallVector<SelfCoverageViolation> violations;
};

/// G1-self: every dependency `DepBetween` reports must be ordered by emitted sync.
SelfCoverageReport computeSelfCoverage(func::FuncOp func);

/// `cap` bounds the violations printed per function; 0 prints all. A capped report
/// HIDES CLASSES -- two cross-pipe carried outliers on `rope_kv_cache` sat past the
/// default cap and went undiagnosed through a whole round of analysis -- so raise it
/// when attributing causes.
void printSelfCoverage(llvm::raw_ostream &os, llvm::StringRef funcName,
                       const SelfCoverageReport &report, unsigned cap = 8);

} // namespace oracle
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCORACLEGATES_H
