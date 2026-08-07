// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnifiedSync.cpp - Unified intra-core sync allocator -------------===//
//
// Allocates intra-core synchronization across all three mechanisms from one
// model, rather than choosing a mechanism up front: event ids (a pool per
// (src, dst) pipe direction), buffer-id tokens (one pool of K, exposed only where
// K > 0), and PIPE_ALL barriers as the fallback that always succeeds.
//
// The front end is REUSED, not reimplemented: Translator -> InsertSyncAnalysis ->
// MoveSyncState -> RemoveRedundantSync produce the hazard set, the same
// producer-to-consumer orderings the incumbent pass works from. Only the
// assignment of mechanisms and ids to those hazards is new. Sharing the front end
// is what makes a differential comparison against the incumbent meaningful: both
// see the same program and the same hazards.
//
// Order of work, and why it is this order:
//   1. settle every operation (loop-carried compensation, mechanism routing)
//      before any id exists, since an id assigned to an op that later moves or
//      disappears is worse than no id;
//   2. route buffers whose direction would overflow its event pool to buffer-id
//      tokens, so the colouring below sees a smaller problem;
//   3. colour the remaining hazards' intervals into per-direction event pools,
//      spilling to a barrier where the pool cannot hold them;
//   4. emit, and check the result against the oracle gates.
//
// `K` is the buffer-id capacity: 0 on A3, which disables routing entirely, and 32
// on A5.
//===----------------------------------------------------------------------===//
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/InsertSyncAnalysis.h"
#include "PTO/Transforms/InsertSync/MoveSyncState.h"
#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/InsertSync/SyncOracleExtract.h"
#include "PTO/Transforms/InsertSync/SyncOracleGates.h"
#include "PTO/Transforms/InsertSync/UnifiedSyncModel.h"
#include "../BufidSync/BufidSyncAnalysis.h"
#include "../BufidSync/BufidSyncCodegen.h"
#include "../BufidSync/BufidSyncIdAlloc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOUNIFIEDSYNC
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Same gather/scatter skip condition the incumbent uses: this pass does not run
// RemoveRedundantSync on functions containing these ops.
static bool hasGatherScatterLikeOps(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::TGatherOp, pto::TGatherBOp, pto::TScatterOp, pto::MGatherOp,
            pto::MScatterOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// TEST-ONLY. Routes hazard group 0 off its default mechanism, so the BUFID value
/// is exercised end to end on kernels that would not otherwise route: set on both
/// halves of a pair, carried through the extractor, and printed by the dumper.
///
/// Marks BOTH halves rather than one. `GetMatchSync` does propagate the mechanism
/// to the half it creates, but it built this pair inside `InsertSyncAnalysis`,
/// before any mechanism was chosen, so a later stamp on one half leaves the other
/// on its constructed default. A whole hazard group is the granularity the
/// allocator itself routes at.
/// False when `mode` names nothing, which the caller must report. Only the two
/// values below may reach the stamp: any other string used to arrive here and be
/// treated as a request to route, so a misspelling silently stamped BUFID on
/// hazard group 0 and the run then failed for an unrealised mechanism instead of
/// for the bad argument.
static bool forceMechanismOnFirstHazard(SyncOperations &ops,
                                        llvm::StringRef mode) {
  if (mode != "bufid" && mode != "bufid-spill")
    return false;
  if (ops.empty())
    return true;
  for (auto &owned : ops.front()) {
    if (!owned)
      continue;
    owned->SetMechanism(SyncOperation::MECHANISM::BUFID);
    // "bufid-spill" additionally exercises the resource-exhaustion downgrade:
    // SetPipeAll must drop the mechanism back to BARRIER along with the type, or a
    // spilled barrier would keep claiming to hold a buffer-ID token it no longer
    // has.
    if (mode == "bufid-spill")
      owned->SetPipeAll();
  }
  return true;
}

struct PTOUnifiedSyncPass
    : public mlir::pto::impl::PTOUnifiedSyncBase<PTOUnifiedSyncPass> {
  // Buffer-ID capacity K (0 = A3, 32 = A5). Zero disables buffer routing.
  unsigned bufidCapacity = 0;
  // TEST-ONLY: "" | "bufid" | "bufid-spill". See forceMechanismOnFirstHazard.
  std::string forceMechanism;
  /// Print the allocator's model/routing/colouring reports. Off by default: the
  /// reports are verbose enough to bury a caller's own output. Gate violations are
  /// reported either way.
  bool debugEnabled = false;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;

    // Do not run on functions that already carry explicit synchronization.
    bool hasExplicitSync = false;
    func.walk([&](Operation *op) {
      if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::RecordEventOp,
              pto::WaitEventOp>(op)) {
        hasExplicitSync = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasExplicitSync)
      return;

    // Macro ops are supported. A macro op is one for which `getSyncMacroModel`
    // returns a model; every model declares `hiddenEvents`, the fixed
    // (direction, id) pairs the library's own `PtoSetWaitFlag<SRC, DST>()` calls
    // use. Those uses are invisible in PTO IR -- no op names them -- so without a
    // reservation the colourer would hand one to a compiler hazard whose interval
    // spans the call, and nothing in the emitted IR would show it.
    //
    // A reservation reduces effective pool capacity, so a kernel that fit in eight
    // ids can be pushed into the spill path, and at K=0 no buffer-ID routing can
    // absorb that. The spill path is load-bearing for macro kernels.

    // --- Reused front-end: build the hazard set H (no id assignment) --------
    MemoryDependentAnalyzer memAnalyzer;
    SyncIRs syncIR;
    SyncOperations syncOpsStorage;
    Buffer2MemInfoMap buffer2MemInfoMap;

    PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func,
                               SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    if (syncIR.size() <= 1)
      return;

    // --- Reused buffer clustering -------------------------------------------
    // One virtual buffer id per clique of aliasing tiles (buffer-keyed, not
    // per-hazard). Computed on the freshly-translated SyncIR -- exactly the IR
    // BufidSyncPass sees -- before any event-sync ops are inserted. Analysis
    // only: no physical ids assigned, no get_buf/rls_buf emitted.
    BufidSyncAnalysis bufAnalysis(syncIR, memAnalyzer, func,
                                  /*debugEnabled=*/false);
    bufAnalysis.collectDependencies();
    bufAnalysis.classifyTiles();
    bufAnalysis.allocateVirtualBufIds();
    const SmallVector<VirtualBufId> &clusters = bufAnalysis.getVirtualBufIds();

    InsertSyncAnalysis analyzer(syncIR, memAnalyzer, syncOpsStorage, func,
                                SyncAnalysisMode::NORMALSYNC);
    analyzer.Run(/*insertBarAllAtLast=*/true);

    MoveSyncState syncMove(syncIR, syncOpsStorage);
    syncMove.Run();

    if (!hasGatherScatterLikeOps(func)) {
      RemoveRedundantSync removeRedundant(syncIR, syncOpsStorage,
                                          SyncAnalysisMode::NORMALSYNC);
      removeRedundant.Run();
    }

    // Every SyncOperation carries a `mechanism` -- the resource class backing it --
    // derived from its type at construction. Only the test-only injector below
    // overrides that default on a kernel the router would otherwise leave alone.
    if (!forceMechanism.empty() &&
        !forceMechanismOnFirstHazard(syncOpsStorage, forceMechanism)) {
      func.emitError() << "unknown --unified-sync-force-mechanism value '"
                       << forceMechanism << "'; expected bufid or bufid-spill";
      signalPassFailure();
      return;
    }

    // --- The unified data model ---------------------------------------------
    // One Hazard per sync group: a mechanism must be stamped on the set and the wait
    // together or the pair is split across two resources and hangs.
    //
    // Cluster identity is `MemAlias`, the relation BufidSyncAnalysis itself uses.
    // Pointer identity alone is exact but incomplete, because the tile dedup drops
    // the losing BaseMemInfo objects and some hazards then retain no matching
    // pointer -- hence pointer fast path, MemAlias fallback. See
    // `MemInfoToClusters` for the keys this must not be simplified back to.
    unified::MemInfoToClusters memInfoToClusters;
    {
      llvm::SmallVector<const BaseMemInfo *> hazardMemInfos;
      llvm::DenseSet<const BaseMemInfo *> seen;
      for (const auto &group : syncOpsStorage)
        for (const auto &owned : group)
          if (owned)
            for (const BaseMemInfo *mi : owned->depMemInfos)
              if (mi && seen.insert(mi).second)
                hazardMemInfos.push_back(mi);

      for (const BaseMemInfo *mi : hazardMemInfos) {
        auto &out = memInfoToClusters[mi];
        for (const VirtualBufId &vb : clusters) {
          bool hit = false;
          for (const TileInfo &t : vb.tiles) {
            if (!t.memInfo)
              continue;
            if (t.memInfo == mi ||
                memAnalyzer.MemAlias(const_cast<BaseMemInfo *>(mi),
                                     const_cast<BaseMemInfo *>(t.memInfo))) {
              hit = true;
              break;
            }
          }
          if (hit && !llvm::is_contained(out, vb.logicId))
            out.push_back(vb.logicId);
        }
      }
    }

    unified::SyncModel model = unified::buildSyncModel(
        syncOpsStorage, bufidCapacity, memInfoToClusters, syncIR);

    // --- Settle the operations before any id is assigned --------------------
    // Synthesise the head/tail compensation pair for every loop-carried hazard
    // BEFORE coloring; without it the in-body backward pair waits on a flag nothing
    // raised and deadlocks on iteration 1.
    //
    // `fe` counts hazards flagged loop-carried by GetForEndIndex, `inv` counts
    // raw-index inversion (wait before set), `pairs` the head/tail groups appended.
    // fe == pairs is the invariant: one head and one tail per loop-carried hazard.
    unsigned feCount = 0, invCount = 0;
    for (const unified::Hazard &h : model.hazards) {
      if (h.isBarrier())
        continue;
      if (h.setOp()->GetForEndIndex().has_value())
        ++feCount;
      if (h.waitOp() &&
          h.waitOp()->GetSyncIRIndex() < h.setOp()->GetSyncIRIndex())
        ++invCount;
    }
    // `hazards=` on the model report means "orderings the front end found".
    // Capture it BEFORE compensation synthesis, which appends one group per
    // loop-carried hazard -- reading it afterwards silently redefines the number.
    unsigned nHazards = syncOpsStorage.size();

    unsigned synthPairs =
        unified::synthesizeLoopCompensation(model, syncOpsStorage, syncIR);
    unsigned compCount = 0;
    for (const unified::Hazard &h : model.hazards)
      if (h.hasCompensation())
        ++compCount;

    // Decide which buffers take tokens before anything takes an event id, so the
    // colourer sees the reduced demand.
    unified::BufferRouteResult route = unified::routeBuffers(model);

    // --- Buffer-ID emission ---------------------------------------------------
    //
    // Three things must land TOGETHER; any two without the third is the
    // split-mechanism shape that hangs:
    //   1. stamp BUFID on every hazard of a routed buffer, and suppress its event
    //      ops -- codegen dispatches on TYPE, so stamping alone changes nothing;
    //   2. the colourer skips them, so they take no event id;
    //   3. get_buf/rls_buf are emitted for them.
    // (1) and (2) are here; (3) is after the event codegen below.
    unsigned routedHazards = 0;
    llvm::SmallDenseSet<int, 8> routedLogicIds;
    for (const unified::Buffer &b : model.buffers)
      if (b.routedToBufid)
        routedLogicIds.insert(b.logicId);
    if (!routedLogicIds.empty()) {
      for (unified::Hazard &h : model.hazards) {
        if (h.isBarrier() || h.bufferClusters.empty())
          continue;
        bool onRouted = llvm::any_of(h.bufferClusters, [&](int c) {
          return routedLogicIds.contains(c);
        });
        if (!onRouted)
          continue;
        h.RouteToBufid();
        ++routedHazards;
      }
    }

    // Pre-colour the ids a macro's library implementation consumes internally,
    // before the greedy scan can hand one out.
    unsigned hiddenReserved = unified::seedHiddenMacroEvents(model, syncIR);
    unified::EventColorResult alloc = unified::colorEventIds(model);

    auto syncOpRecords = oracle::extractFromSyncOps(syncOpsStorage);

    oracle::DeviceIdLimits limits;
    limits.bufIdCapacity = bufidCapacity;
    auto idViolations = oracle::checkDeviceIdLegality(syncOpRecords, limits);
    // A macro's library implementation consumes event ids that no PTO op names, so
    // an id handed to a hazard spanning the call collides invisibly. `colorEventIds`
    // already refuses a reserved id; this re-checks the result independently, because
    // no other gate can see a use inside a library call. Folded into the same list as
    // G3 because it is the same question -- is this id legal here -- answered against
    // a different source.
    for (const auto &v :
         oracle::checkMacroHiddenEventCollisions(syncIR, syncOpRecords))
      idViolations.push_back(v);
    auto interference = oracle::checkNonInterference(syncOpRecords);

    // Gate violations are reported whether or not debug is on: they precede a
    // signalPassFailure, so suppressing them would leave a failing pass with no reason.
    // Nothing is printed when they are empty.
    if (!idViolations.empty() || oracle::countErrors(interference) > 0)
      oracle::emitReport([&](llvm::raw_ostream &os) {
        oracle::printIdViolations(os, func.getSymName(), "syncops",
                                  syncOpRecords.size(), idViolations);
        oracle::printInterferenceViolations(os, func.getSymName(), "syncops",
                                            syncOpRecords.size(), interference);
      });

    // One atomic report: nested func passes run in parallel (see emitReport).
    if (debugEnabled)
      oracle::emitReport([&](llvm::raw_ostream &os) {
        os << "[unified-sync model] kernel=" << func.getSymName()
           << " K=" << bufidCapacity << " hazards=" << nHazards << "\n";
        oracle::printSyncOpRecords(os, func.getSymName(), syncOpRecords);
        oracle::printIdViolations(os, func.getSymName(), "syncops",
                                  syncOpRecords.size(), idViolations);
        oracle::printInterferenceViolations(os, func.getSymName(), "syncops",
                                            syncOpRecords.size(), interference);
        os << "[unified-sync reachable] buffer clusters (virtual buf ids) = "
           << clusters.size() << "\n";
        unsigned shownC = 0;
        for (const VirtualBufId &vb : clusters) {
          os << "  cluster#" << vb.logicId
             << " scope=" << stringifyAddressSpace(vb.scope)
             << " aliasing_tiles=" << vb.tiles.size() << "\n";
          if (++shownC >= 6)
            break;
        }
        os << "[unified-sync reachable] bufid emission ops reachable: "
           << pto::GetBufOp::getOperationName() << ", "
           << pto::RlsBufOp::getOperationName() << "\n";
        unified::printSyncModel(os, func.getSymName(), model);
        os << "[unified-sync routing] func=" << func.getSymName()
           << " buffer routing: considered="
           << route.buffersConsidered << " routed=" << route.routed
           << " hazards_covered=" << route.hazardsCovered
           << " skipped(no_overflow=" << route.skippedNoOverflow
           << " not_written_back=" << route.skippedNotWrittenBack
           << " K=0:" << route.skippedNoCapacity << ")"
           << " not_expressible=" << route.skippedNotExpressible
           << " split=" << route.splitHazards
           << (route.splitHazards == 0 ? " OK" : " !! SPLIT-MECHANISM") << "\n";
        for (const unified::Buffer &b : model.buffers)
          if (b.routedToBufid)
            os << "    ROUTE " << func.getSymName() << " b#" << b.logicId
               << " -> bufid"
               << " omega=" << b.predictedPeakOverlap << ">pool="
               << b.smallestPool << " wb=yes lbufid=[" << b.bufidLoop.start << ","
               << b.bufidLoop.end << "]\n";
        // A divergence between `fe` and `inv` means compensation was synthesised for
        // a FORWARD hazard, as `prefetch_disjoint_slots` shows.
        //
        // WARNING, not error, and the incumbent is the reason: it primes forward
        // hazards too, so a divergence is not by itself wrong, and erroring would
        // refuse multi-tile kernels that `--enable-insert-sync` compiles.
        if (feCount != invCount)
          os << "[unified-sync compensation] !! predicate-divergence func="
             << func.getSymName() << " fe=" << feCount << " inv=" << invCount
             << " : compensation synthesised for " << (feCount - invCount)
             << " hazard(s) whose in-body set precedes its wait (forward, not "
                "index-inverted). Not an error -- the incumbent primes these too -- but "
                "the two predicates are not interchangeable.\n";
        os << "[unified-sync compensation] loop-compensation: fe=" << feCount
           << " inv=" << invCount << " pairs=" << synthPairs
           << " heads=" << synthPairs << " tails=" << synthPairs
           << " linked=" << compCount
           << (feCount == synthPairs && synthPairs == compCount ? " OK" : " MISMATCH")
           << "\n";
        os << "[unified-sync coloring] interval coloring: assigned=" << alloc.assigned
           << " overflow=" << alloc.overflow
           << " spilled_to_barrier=" << alloc.spilledToBarrier
           << " hidden_reserved=" << hiddenReserved
           << " reused=" << alloc.reused
           << " skipped_routed=" << alloc.skippedRouted
           << " rotating=" << alloc.rotating
           << " ids_assigned=" << alloc.idsAssigned
           << " (per-direction first-fit; ids reused across disjoint intervals)\n";

        // --- Placement of the synthesised compensation pairs ------------------
        // SyncCodegen walks syncIR's pipe lists, so the pair must BE in them or it
        // is dropped silently at emission. The load-bearing part is whether the tail
        // PRECEDES any existing loop-end SET there; a tail after a set is the
        // broken-sequence case, reported ORDER_BAD.
        unsigned placedHead = 0, placedTail = 0, orderOk = 0, orderBad = 0;
        os << "[unified-sync placement] placement into syncIR pipe lists:\n";
        for (const unified::Hazard &h : model.hazards) {
          if (!h.hasCompensation())
            continue;
          const SyncOperation *head = h.headOp();
          const SyncOperation *tail = h.tailOp();
          unsigned b = head->GetSyncIRIndex();
          unsigned e = tail->GetSyncIRIndex();
          const auto &before = syncIR[b]->pipeBefore;
          const auto &after = syncIR[e]->pipeAfter;
          int hPos = -1, tPos = -1, firstSet = -1;
          for (size_t i = 0; i < before.size(); ++i)
            if (before[i] == head) {
              hPos = static_cast<int>(i);
              break;
            }
          for (size_t i = 0; i < after.size(); ++i) {
            if (after[i] == tail && tPos < 0)
              tPos = static_cast<int>(i);
            if (firstSet < 0 && after[i] != tail &&
                after[i]->GetType() == SyncOperation::TYPE::SET_EVENT)
              firstSet = static_cast<int>(i);
          }
          if (hPos >= 0)
            ++placedHead;
          if (tPos >= 0)
            ++placedTail;
          bool ok = (tPos >= 0) && (firstSet < 0 || tPos < firstSet);
          ok ? ++orderOk : ++orderBad;
          os << "  pair h#" << h.index()
             << " id=" << (h.hasEventId() ? h.setOp()->eventIds.front() : -1)
             << "  head@syncIR[" << b << "].pipeBefore[" << hPos << "/"
             << before.size() << "]  tail@syncIR[" << e << "].pipeAfter[" << tPos
             << "/" << after.size() << "]  firstLoopEndSet=" << firstSet
             << (ok ? "  ORDER_OK" : "  ORDER_BAD(tail after set)") << "\n";
        }
        os << "[unified-sync placement] placed: heads=" << placedHead
           << " tails=" << placedTail << " of pairs=" << synthPairs
           << " order_ok=" << orderOk << " order_bad=" << orderBad
           << ((placedHead == synthPairs && placedTail == synthPairs &&
                orderBad == 0)
                   ? " OK"
                   : " MISMATCH")
           << "\n";

        os << "[unified-sync codegen] codegen WIRED: emitting set_flag/wait_flag/"
              "barrier from the colored model\n";
      });

    if (!idViolations.empty() || !interference.empty()) {
      func.emitError() << "unified allocator failed an oracle gate: "
                       << idViolations.size() << " illegal id(s), "
                       << interference.size() << " interfering id(s)";
      return signalPassFailure();
    }

    // --- Emission -----------------------------------------------------------
    // Driven exactly as the incumbent drives it (PTOInsertSync.cpp): construct
    // over the same `syncIR` and Run(). Codegen walks syncIR's pipeBefore/
    // pipeAfter lists -- which is why the step above had to place the synthesised
    // head/tail pairs there, or they would be dropped silently here.
    //
    // Runs AFTER the in-pass gates: an allocation the oracle rejects is not
    // emitted.
    SyncCodegen codegen(syncIR, func, SyncAnalysisMode::NORMALSYNC);
    codegen.Run();
    // Fail the pass on an unrealised hazard. Only the unified path signals failure:
    // insert-sync provably cannot reach the guard, so its shipping behaviour is
    // untouched (the diagnostic itself is emitted from SyncCodegen either way).
    if (codegen.sawUnrealisedHazard())
      signalPassFailure();

    // --- (3) Emit get_buf/rls_buf for the routed buffers ---------------------
    //
    // REUSE, NOT REBUILD. `BufidSyncAnalysis::insertSyncOperations` +
    // `BufidSyncIdAlloc` + `BufidSyncCodegen` are the production pipeline, and it
    // already runs over the same SyncIR and the same clusters routing ran on. The
    // only addition is a FILTER: BufidSync emits for every cluster, only the routed
    // ones are wanted, and `BufSyncOperation` carries `logicId`.
    //
    // PHYSICAL IDS ARE DELEGATED to `BufidSyncIdAlloc`, not re-colored here. It is
    // the same greedy left-edge scan but over ONE pool of K shared across all
    // directions, unlike `colorEventIds` which is per direction. It also owns the
    // exhaustion path (`needsReuse` -> `reuseIds` -> `compactPhysicalIds`, which
    // coalesces logic ids rather than failing) and
    // `validateNoSamePhysicalIdNesting`, and its interval model is
    // `computeLifeIntervals`' union span -- the same semantics the event colourer is
    // aligned to. Duplicating it would fork them.
    if (routedHazards > 0) {
      // --- Token sites come from the ALLOCATOR'S HAZARDS, not from
      // --- BufidSyncAnalysis's own dependence set ------------------------------
      //
      // `BufidSyncAnalysis::collectDependencies` enumerates FORWARD pairs only
      // (`for j = i + 1`), so a loop-carried hazard -- whose producer sits after its
      // consumer in program order and whose ordering crosses the back edge -- can
      // never appear in `depPairs_`. Feeding the emitter from that set while routing
      // from this one suppressed such a hazard's event ops and emitted no token for
      // it, losing the ordering with nothing to report it.
      //
      // A token DOES carry a loop-carried ordering, and needs no priming pair to do
      // it: the counter starts free, so iteration N+1's `get_buf` blocks on
      // iteration N's `rls_buf`. Only the emission was missing.
      //
      // So the sites are derived here, from each routed hazard, and BOTH endpoints
      // are bracketed -- four anchors where the producer and consumer are distinct
      // ops. Where several ops on the right pipe alias the cluster, ALL of them are
      // bracketed rather than one being chosen: over-bracketing spends a token,
      // under-bracketing loses an ordering, and those are not symmetric.
      auto &op2BufSync = bufAnalysis.getOp2BufSync();
      op2BufSync.clear();

      // Cluster -> its tiles' memInfos, for aliasing tests below.
      llvm::DenseMap<int, llvm::SmallVector<const BaseMemInfo *>> clusterMemInfos;
      for (const VirtualBufId &vb : clusters)
        if (routedLogicIds.contains(vb.logicId))
          for (const TileInfo &t : vb.tiles)
            if (t.memInfo)
              clusterMemInfos[vb.logicId].push_back(t.memInfo);

      auto aliasesCluster = [&](const CompoundInstanceElement *cp, int logicId) {
        auto it = clusterMemInfos.find(logicId);
        if (it == clusterMemInfos.end())
          return false;
        for (const auto *vec : {&cp->defVec, &cp->useVec})
          for (const BaseMemInfo *mi : *vec) {
            if (!mi)
              continue;
            for (const BaseMemInfo *tm : it->second)
              if (mi == tm ||
                  memAnalyzer.MemAlias(const_cast<BaseMemInfo *>(mi),
                                       const_cast<BaseMemInfo *>(tm)))
                return true;
          }
        return false;
      };

      // Bracket one op for one (pipe, cluster), de-duplicated: the emitter would
      // otherwise emit the same get/rls twice for an op touching the cluster twice.
      unsigned keptGet = 0, keptRls = 0;
      auto bracket = [&](Operation *op, PipelineType pipe, int logicId,
                         unsigned irIdx) {
        if (!op)
          return;
        auto &build = op2BufSync[op];
        auto present = [&](const llvm::SmallVector<BufSyncOperation> &v,
                           BufSyncType ty) {
          for (const BufSyncOperation &s : v)
            if (s.type == ty && s.logicId == logicId && s.pipe == pipe)
              return true;
          return false;
        };
        if (!present(build.pipeBefore, BufSyncType::GET_BUF)) {
          build.pipeBefore.push_back(
              {BufSyncType::GET_BUF, pipe, logicId, irIdx, irIdx});
          ++keptGet;
        }
        if (!present(build.pipeAfter, BufSyncType::RLS_BUF)) {
          build.pipeAfter.push_back(
              {BufSyncType::RLS_BUF, pipe, logicId, irIdx, irIdx});
          ++keptRls;
        }
      };

      unsigned bracketedHazards = 0, unbracketedHazards = 0;
      for (const unified::Hazard &h : model.hazards) {
        if (h.mechanism() != SyncOperation::MECHANISM::BUFID || h.isBarrier())
          continue;
        bool anySite = false;
        for (int c : h.bufferClusters) {
          if (!routedLogicIds.contains(c))
            continue;
          // Both ends: producer sites on srcPipe, consumer sites on dstPipe.
          for (const auto &element : syncIR) {
            auto *cp = dyn_cast_or_null<CompoundInstanceElement>(element.get());
            if (!cp || !cp->elementOp || !aliasesCluster(cp, c))
              continue;
            if (cp->kPipeValue == h.srcPipe() || cp->kPipeValue == h.dstPipe()) {
              bracket(cp->elementOp, cp->kPipeValue, c, cp->GetIndex());
              anySite = true;
            }
          }
        }
        anySite ? ++bracketedHazards : ++unbracketedHazards;
      }

      if (!op2BufSync.empty()) {
        BufidSyncIdAlloc idAlloc(bufAnalysis.getVirtualBufIds(), op2BufSync,
                                 syncIR, bufidCapacity, /*debugEnabled=*/false);
        idAlloc.computeLifeIntervals();
        idAlloc.linearScanAllocate();
        idAlloc.compactPhysicalIds();
        if (idAlloc.needsReuse()) {
          idAlloc.reuseIds();
          idAlloc.compactPhysicalIds();
        }
        std::string nestErr;
        bool nestOk = idAlloc.validateNoSamePhysicalIdNesting(&nestErr);
        bufAnalysis.setLogicToPhysicalId(idAlloc.getLogicToPhysical());
        bufAnalysis.mergeGetRls();

        BufidSyncCodegen bufCodegen(func, op2BufSync, idAlloc);
        bool emitOk = succeeded(bufCodegen.run());

        // ALL-OR-NOTHING, ASSERTED not assumed. Every event op belonging to a
        // routed hazard must have been suppressed, or codegen emitted a set/wait
        // for a buffer whose ordering is carried by the token -- the counter cycle
        // missing a step, i.e. a hang. `leaked_event_ops` must be 0.
        unsigned suppressed = 0, leaked = 0;
        for (const unified::Hazard &h : model.hazards) {
          if (h.mechanism() != SyncOperation::MECHANISM::BUFID)
            continue;
          for (SyncOperation *op :
               {h.setOp(), h.waitOp(), h.headOp(), h.tailOp()}) {
            if (!op)
              continue;
            op->uselessSync ? ++suppressed : ++leaked;
          }
        }

        if (debugEnabled)
          oracle::emitReport([&](llvm::raw_ostream &os) {
            os << "[unified-sync bufid] func=" << func.getSymName()
               << " bufid emission: routed_buffers=" << routedLogicIds.size()
               << " routed_hazards=" << routedHazards
               << " anchor_ops=" << op2BufSync.size() << " get=" << keptGet
               << " rls=" << keptRls
               << " physical_ids=" << idAlloc.getLogicToPhysical().size()
               << " nesting=" << (nestOk ? "OK" : "VIOLATION")
               << " emit=" << (emitOk ? "OK" : "FAILED")
               << " bracketed_hazards=" << bracketedHazards
               << " unbracketed_hazards=" << unbracketedHazards
               << (unbracketedHazards == 0 ? "" : " !! UNBRACKETED")
               << " suppressed_event_ops=" << suppressed
               << " leaked_event_ops=" << leaked
               << (leaked == 0 ? " ALL-OR-NOTHING-OK" : " !! SPLIT-MECHANISM")
               << "\n";
            if (!nestOk)
              os << "  !! " << nestErr << "\n";
          });
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOUnifiedSyncPass(unsigned bufidCapacity, bool debugEnabled,
                                    llvm::StringRef forceMechanism) {
  auto pass = std::make_unique<PTOUnifiedSyncPass>();
  pass->bufidCapacity = bufidCapacity;
  pass->debugEnabled = debugEnabled;
  pass->forceMechanism = forceMechanism.str();
  return pass;
}
