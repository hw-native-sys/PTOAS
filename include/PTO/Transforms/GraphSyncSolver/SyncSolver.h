// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- SyncSolver.h ---- Graph Sync Solver --------------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H

#include "PTO/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::pto::syncsolver {

class GraphSolver;

class Solver {
public:
  // Configuration options.
  const SyncSolverOptions options;

  // Original MLIR function being processed (may be null for test-only Solver).
  func::FuncOp funcOp;

  // In-memory hierarchical IR (Function -> Scopes -> Ops) used by the solver.
  std::unique_ptr<OperationBase> funcIr;

  // Linearized occurrence sequence (sync IR) built from funcIr, each Occurrence
  // represents one appearance of an operation in the sync-analysis order.
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // Set of RW operations that expose unit-flag feature and need special
  // handling.
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;

  // Collected conflict pairs chosen by the algorithm for insertion (and
  // persistent ones that survive multiple passes).
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs,
      persistentChosenConflictedPairs;

protected:
  using ExtraConflictPairs =
      llvm::SmallVector<std::pair<std::unique_ptr<ConflictPair>, Occurrence *>>;
  int64_t globalSetWaitIndex{0};
  int64_t maxReuseNum{20};
  int64_t maxRunNum{99};
  bool moveBackwardSyncPairsToOutmostLoop{false};
  bool dontMoveBackwardSyncPairsToOutmostLoop{false};

  llvm::DenseMap<std::tuple<pto::PIPE, pto::PIPE>,
                 std::unique_ptr<EventIdSolver>>
      eventIdSolver;

  // Map op -> list of occurrences in syncIr (quick lookup for an op's
  // occurrences).
  llvm::DenseMap<const OperationBase *, std::vector<Occurrence *>>
      opAllOccurrences;

  // Bookkeeping map used to record that a pair (scopeOp, op1, op2, setPipe,
  // waitPipe) has already been synchronized and which ConflictPair performed
  // it.
  SyncScopePairDenseMap<llvm::DenseSet<ConflictPair *>> syncedPairs;

  SyncScopePairDenseMap<ConflictPair *> replacedWithReusableSyncedPairs;

  // Chosen conflicts keyed by occurrence (scope occurrence) to allow retrieving
  // conflicts that affect a particular occurrence subtree.
  llvm::DenseMap<Occurrence *, llvm::DenseSet<ConflictPair *>>
      scopeOccChosenConflicts, persistentScopeOccChosenConflicts;

  // Chosen conflicts keyed by a pair of scope-occurrences, used when conflicts
  // are associated with a pair of sibling blocks (e.g., condition branches).
  llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOccPairChosenConflicts, persistentScopeOccPairChosenConflicts;

  // Processing order list created from syncIr that drives pairwise conflict
  // checks.
  std::vector<ProcessingOrder> processingOrders;

  // Set of processed occurrence pairs to avoid re-processing the same pair.
  llvm::DenseSet<std::pair<Occurrence *, Occurrence *>> processedOccPairs;

  // Occurrences marked skippable (exclusion set used during processing).
  llvm::DenseMap<uint8_t, llvm::DenseSet<Occurrence *>> skipOcc;

  // Accumulated backward-sync events for each operation (recorded instead of
  // inserting explicit conflict pairs). The outer map key is the scope op; the
  // inner map key is (setPipe, waitPipe) and value is the set of event ids
  // used.
  llvm::MapVector<OperationBase *,
                  CorePipePairDenseMap<llvm::DenseMap<int64_t, int64_t>>>
      backwardSyncEvents;

  llvm::MapVector<OperationBase *, CorePipePairDenseSet>
      backwardSyncEventsAfterMerge;

  // Memoization of memory-conflict discovery between specific RWOperation
  // pairs.
  llvm::DenseMap<
      std::pair<syncsolver::RWOperation *, syncsolver::RWOperation *>,
      llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>>>
      checkMemoryConflictsMem;

  // Set of pipe pairs that were forced to barrier-all (no event ids available).
  CorePipePairDenseSet barrierAllPairs;

  // Set of pipe pairs for which multi-event-id usage is disabled.
  CorePipePairDenseSet disabledMultiEventIdPairs;

  // Count-per-pipe-pair used to limit reuse of conflict pairs (reuse budget).
  CorePipePairDenseMap<int> reusePairs, reusedPairs;

  // Tracks inserted barrier-all markers before occurrences: op -> set of (occ,
  // isUseless).
  llvm::DenseMap<OperationBase *,
                 llvm::DenseSet<std::pair<Occurrence *, int32_t>>>
      insertedBarrierAllBefore;

  // Indices allocated during codegen walk: start/end and inclusive variants
  // used to evaluate ordering relationships between ops during merging checks.
  llvm::DenseMap<OperationBase *, int64_t> setWaitStartIndex, setWaitEndIndex,
      setWaitStartIndexInclusive, setWaitEndIndexInclusive;

  // Index of set/wait ops: key=(setPipe,waitPipe,eventId) -> ordered set of
  // (codegen-index, SetWaitOp*) for quick queries.
  llvm::DenseMap<std::tuple<pto::PIPE, pto::PIPE, int64_t>,
                 std::set<std::pair<int64_t, SetWaitOp *>>>
      setWaitFlagOpsIndex;

public:
  Solver() = delete;
  virtual ~Solver() = default;

  Solver(std::unique_ptr<IRTranslator> irTranslator)
      : options(irTranslator->options) {
    init(std::move(irTranslator));
  }

  // Orchestrate the solving process (entry point).
  void solve();

  // Build before/after maps of sync ops computed from chosen conflicts.
  SyncBeforeAfterMap getBeforeAfterSyncMaps();

protected:
  void init(std::unique_ptr<IRTranslator> irTranslator) {
    funcOp = irTranslator->funcOp;
    funcIr = std::move(irTranslator->funcIr);
    syncIr = std::move(irTranslator->syncIr);
    unitFlagFeaturedOps = std::move(irTranslator->unitFlagFeaturedOps);
    opAllOccurrences = std::move(irTranslator->opAllOccurrences);
    processingOrders = std::move(irTranslator->processingOrders);
  }

  // Reset solver internal bookkeeping prior to another pass.
  void reset(bool resetEventIdRanOutOpts = false);

  llvm::LogicalResult runSolver(bool enableOpts1 = true,
                                bool enableOpts2 = true);

  // Reset unit-flag related bookkeeping prior to another pass.
  void resetUnitFlag();

  // Walk and process the generated processingOrders to choose conflicts.
  void processOrders();

  virtual void processConflict(Occurrence *occ1, Occurrence *occ2,
                               RWOperation *rwOp1, RWOperation *rwOp2,
                               bool isUseless);

  std::optional<LoopLikeOpInterface>
  getMultiBufferLoop(RWOperation *rwOp1, RWOperation *rwOp2,
                     const llvm::SmallVector<MemInfo> &memInfoList1,
                     const llvm::SmallVector<MemInfo> &memInfoList2);
  std::optional<LoopLikeOpInterface> getMultiBufferLoop(RWOperation *rwOp1,
                                                        RWOperation *rwOp2);
  bool validateMultiBufferScope(Occurrence *occ1, Occurrence *occ2,
                                RWOperation *rwOp1, RWOperation *rwOp2,
                                LoopLikeOpInterface &multibufferLoop);
  void updateMultiBufferConflictStats(
      RWOperation *rwOp1, RWOperation *rwOp2,
      const llvm::SmallVector<MemInfo> &lhs,
      const llvm::SmallVector<MemInfo> &rhs, bool includeMemInfo1Min,
      bool includeMemInfo2Min,
      int64_t &lcm, int64_t &minWriteSize);
  std::optional<int64_t> findMultiBufferEventIdNum(
      RWOperation *rwOp1, RWOperation *rwOp2, int64_t lcm,
      int64_t minWriteSize);
  std::optional<EventIdInfo> getMultiBufferEventIdInfo(Occurrence *occ1,
                                                       Occurrence *occ2,
                                                       RWOperation *rwOp1,
                                                       RWOperation *rwOp2);

  // Determine how many event ids are needed for a particular occurrence pair.
  EventIdInfo getEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                             RWOperation *rwOp1, RWOperation *rwOp2,
                             CorePipeInfo, CorePipeInfo);

  std::optional<EventIdInfo>
  checkCVMultiBufferUnrollEventIdInfo(RWOperation *rwOp1, RWOperation *rwOp2);
  std::optional<EventIdInfo>
  checkCVMultiBufferPreloadEventIdInfo(RWOperation *rwOp1, RWOperation *rwOp2);

  std::optional<EventIdInfo> checkMultiBufferEventIdInfo(Occurrence *occ1,
                                                         Occurrence *occ2,
                                                         RWOperation *rwOp1,
                                                         RWOperation *rwOp2);

  // Graph-based conflict checking and memory conflict detection helpers.
  bool checkGraphConflict(
      Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
      CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
      std::optional<int> startIndex = {}, std::optional<int> endIndex = {},
      const llvm::SmallVector<ConflictPair *> &extraConflictPairs = {},
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs = {});
  void addGraphConflictPair(
      GraphSolver &graphSolver, llvm::DenseSet<ConflictPair *> &visited,
      ConflictPair *conflictPair, EventIdInfo eventIdInfo, int startIndex,
      int endIndex,
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) const;
  void addGraphConflictsFromScopes(
      const Occurrence *occ, GraphSolver &graphSolver,
      llvm::DenseSet<ConflictPair *> &visited, EventIdInfo eventIdInfo,
      int startIndex, int endIndex,
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs,
      bool persistent);
  void addGraphConflictsFromScopePairs(
      Occurrence *occ1, Occurrence *occ2, GraphSolver &graphSolver,
      llvm::DenseSet<ConflictPair *> &visited, EventIdInfo eventIdInfo,
      int startIndex, int endIndex,
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs);

  bool ignoreMemoryConflict(const RWOperation *rwOp1, const RWOperation *rwOp2,
                            const MemInfo &memInfo1, const MemInfo &memInfo2);

  bool checkMemInfoConflict(RWOperation *rwOp1, RWOperation *rwOp2,
                            const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});

  bool checkMemInfoConflict(RWOperation *rwOp1, RWOperation *rwOp2,
                            const llvm::SmallVector<MemInfo> &memInfoList1,
                            const llvm::SmallVector<MemInfo> &memInfoList2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});

  llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>>
  checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  bool checkMemoryConflictBetweenOccExclusive(
      Occurrence *occ1, Occurrence *occ2,
      std::function<bool(RWOperation *)> filter = [](RWOperation *) {
        return true;
      });

  // Feasibility checks and bookkeeping accessors used by the solver loop.
  bool checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2) const;

  bool checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2);

  bool checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2) const;

  bool checkAlreadySynced(Occurrence *occ1, Occurrence *occ2) const;

  bool checkAlreadySyncedWithUnitFlag(Occurrence *occ1,
                                      const Occurrence *occ2);

  bool skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2) const;

  bool checkSyncOpsConflicts(ConflictPair *conflictPair1,
                             ConflictPair *conflictPair2);
  bool checkSetSyncOpsConflict(ConflictPair *conflictPair1,
                               ConflictPair *conflictPair2);
  bool checkWaitSyncOpsConflict(ConflictPair *conflictPair1,
                                ConflictPair *conflictPair2);

  // Check whether two ConflictPair ranges/event mapping intersect (same
  // pipes/events).
  bool checkIntersect(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

  // Event-id allocation and reuse helpers.
  std::vector<ConflictPair *>
  getIntersectingConflictPairs(ConflictPair *conflictPair);

  // Visit tracking helpers for occurrence pairs.
  bool checkVisited(Occurrence *occ1, Occurrence *occ2);

  bool checkSkippable(bool reverseOrder, Occurrence *occ);

  // Bookkeeping for previously synchronized pairs within a scope to reuse their
  // event-ids.
  EventIdNode *getOldEventIdNodeIfExists(ConflictPair *conflictPair);

  void memorizeSyncedPair(ConflictPair *conflictPair);

  llvm::DenseSet<ConflictPair *>
  getMemorizedSyncedPairs(ConflictPair *conflictPair);

  void memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                ConflictPair *reusedConflictPair);

  void forgetSyncedPair(ConflictPair *conflictPair);

  // Utilities to map an occurrence pair to their set/wait occurrences.
  std::pair<Occurrence *, Occurrence *>
  getSetWaitLCAPairOcc(Occurrence *occ1, Occurrence *occ2) const;
  std::pair<Occurrence *, Occurrence *> getSetWaitOcc(Occurrence *occ1,
                                                      Occurrence *occ2);
  std::pair<Occurrence *, Occurrence *> getFixedSetWaitOcc(Occurrence *occ1,
                                                           Occurrence *occ2);
  void adjustSetWaitForLoopBoundary(Occurrence *occ1, Occurrence *&setOcc,
                                    Occurrence *&waitOcc);
  void adjustSetWaitForBackwardCondition(Occurrence *occ1, Occurrence *occ2,
                                         Occurrence *&setOcc,
                                         Occurrence *&waitOcc) const;
  void adjustSetWaitForCrossCoreLoops(Occurrence *occ1, Occurrence *occ2,
                                      Occurrence *&setOcc,
                                      Occurrence *&waitOcc);
  void adjustSetWaitForCrossCoreScopes(Occurrence *occ1, Occurrence *occ2,
                                       Occurrence *&setOcc,
                                       Occurrence *&waitOcc);
  void adjustSetWaitForLoopPlaceholders(Occurrence *&setOcc,
                                        Occurrence *&waitOcc);

  Occurrence *getBarrierWaitOcc(Occurrence *occ1, Occurrence *occ2);

  std::optional<std::pair<Occurrence *, Occurrence *>>
  getFunctionBlockSetWaitOcc(const Occurrence *occ1, Occurrence *occ2);

  std::optional<std::pair<Occurrence *, Occurrence *>>
  getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2);

  // Convenience to insert barrier-all before a given occurrence/op.
  void insertBarrierAllBeforeOcc(Occurrence *occ, bool isUseless,
                                 bool isPersistent = false);

  void insertBarrierAllBeforeOp(const OperationBase *op, bool isUseless,
                                bool isPersistent);

  // Determine the direction (backward) of a synchronization candidate.
  bool isBackwardSync(Occurrence *occ1, Occurrence *occ2) const;

  bool reuseCmp(const ConflictPair *conflictPair1,
                const ConflictPair *conflictPair2) const;

  // Reuse existing conflict pairs where possible to save event ids.
  ConflictPair *getReusableConflictPair(
      ConflictPair *conflictPair,
      const llvm::DenseSet<ConflictPair *> &conflictPairsSet);
  ConflictPair *findOldReusedConflictPair(ConflictPair *conflictPair);
  bool hasReusableConflictBudget(const ConflictPair *conflictPair,
                                 const ConflictPair *oldReusedConflictPair);
  ConflictPair *findBestReusableConflictPair(ConflictPair *conflictPair,
                                             Occurrence *scopeOcc1,
                                             Occurrence *scopeOcc2);
  void applyReusableConflictPair(ConflictPair *conflictPair,
                                 ConflictPair *reusableConflictPair) const;

  bool reuseConflictPair(ConflictPair *conflictPair, Occurrence *scopeOcc1,
                         Occurrence *scopeOcc2);

  std::unique_ptr<EventIdSolver> &getEventIdSolverRef(pto::PIPE pipeSrc,
                                                      pto::PIPE pipeDst);

  bool checkReuseMultiBufferFlagId(ConflictPair *conflictPair);

  // Primary handler invoked to register/record a found conflict.
  void handleConflict(Occurrence *occ1, Occurrence *occ2,
                      const RWOperation *rwOp1, const RWOperation *rwOp2,
                      CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
                      bool isUseless);

  void handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                             CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                             bool isUseless);

  void handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                             CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                             EventIdInfo eventIdInfo, bool isUseless);
  std::unique_ptr<ConflictPair> createSetWaitConflictPair(
      Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
      RWOperation *rwOp2, CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
      EventIdInfo eventIdInfo, bool isUseless, Occurrence *&setOcc,
      Occurrence *&waitOcc, Occurrence *&normScopeOcc1,
      Occurrence *&normScopeOcc2, OperationBase *&normScopeOp,
      Loop *&parentLCALoopOp, Occurrence *&parentLCALoopOcc,
      Occurrence *&parentLCALoopBeforePHOcc,
      Occurrence *&parentLCALoopAfterPHOcc);
  void normalizeSetWaitConflictPair(ConflictPair *conflictPair,
                                    Occurrence *occ1, Occurrence *occ2,
                                    Occurrence *&setOcc,
                                    Occurrence *&waitOcc,
                                    CorePipeInfo corePipeSrc,
                                    CorePipeInfo corePipeDst);
  bool prepareSetWaitConflictPairWithEventSolver(
      EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
      Occurrence *occ1, Occurrence *occ2, Occurrence *normScopeOcc1,
      Occurrence *normScopeOcc2, const OperationBase *normScopeOp,
      OperationBase *barrierOp, CorePipeInfo corePipeSrc,
      CorePipeInfo corePipeDst);
  bool checkColorableOrConvertToBarrierAll(EventIdSolver &curEventIdSolver,
                                           ConflictPair *conflictPair,
                                           OperationBase *barrierOp,
                                           CorePipeInfo corePipeSrc,
                                           CorePipeInfo corePipeDst);
  void initializeConflictEventIdNode(ConflictPair *conflictPair,
                                     EventIdSolver &curEventIdSolver,
                                     Occurrence *occ1, Occurrence *occ2,
                                     const OperationBase *normScopeOp);
  bool insertExtraConflictPair(
      EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
      Occurrence *setOcc, Occurrence *waitOcc, Occurrence *parentScope,
      OperationBase *barrierOp, CorePipeInfo corePipeSrc,
      CorePipeInfo corePipeDst, ExtraConflictPairs &extraConflictPairs,
      bool couldNotRun = false);
  bool insertOuterBackwardConflictPairIfNeeded(
      EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
      const Occurrence *setOcc, const Occurrence *waitOcc,
      const Loop *parentLCALoopOp,
      const Occurrence *parentLCALoopOcc, Occurrence *parentLCALoopBeforePHOcc,
      Occurrence *parentLCALoopAfterPHOcc, OperationBase *barrierOp,
      CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
      ExtraConflictPairs &extraConflictPairs);
  bool insertBoundaryBackwardConflictPairsIfNeeded(
      EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
      Occurrence *setOcc, Occurrence *waitOcc, Occurrence *normScopeOcc1,
      Occurrence *normScopeOcc2, Occurrence *parentLCALoopOcc,
      Occurrence *parentLCALoopBeforePHOcc,
      Occurrence *parentLCALoopAfterPHOcc, OperationBase *barrierOp,
      CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
      ExtraConflictPairs &extraConflictPairs);
  bool collectBackwardConflictExtras(
      EventIdSolver &curEventIdSolver, ConflictPair *conflictPair,
      Occurrence *setOcc, Occurrence *waitOcc, Occurrence *normScopeOcc1,
      Occurrence *normScopeOcc2, Loop *parentLCALoopOp,
      Occurrence *parentLCALoopOcc, Occurrence *parentLCALoopBeforePHOcc,
      Occurrence *parentLCALoopAfterPHOcc, OperationBase *barrierOp,
      CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
      ExtraConflictPairs &extraConflictPairs);
  void recordChosenConflictPair(ConflictPair *conflictPair,
                                Occurrence *normScopeOcc1,
                                Occurrence *normScopeOcc2,
                                const Occurrence *parentLCALoopOcc);
  void appendExtraConflictPairs(ExtraConflictPairs &extraConflictPairs);

  void handleUnitFlagConflict(Occurrence *occ1, Occurrence *occ2,
                              CorePipeInfo corePipeSrc,
                              CorePipeInfo corePipeDst,
                              UnitFlagInfo unitFlagInfo, bool isUseless);

  Occurrence *getFirstIterOcc(Occurrence *occ, Occurrence *parOcc);

  Occurrence *getLastIterOcc(Occurrence *occ, Occurrence *parOcc);

  std::optional<std::pair<Occurrence *, Occurrence *>>
  checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                             Occurrence *occ2, Occurrence *parOcc1,
                             Occurrence *parOcc2);

  // Unit-flag pattern checks used to transform sync into unit-flag modes.
  std::optional<UnitFlagInfo> checkUnitFlagPatterns(Occurrence *,
                                                    Occurrence *) const;

  void pickAndInsertABarrierAll();

  void calcAllEventIds();

  void collectBackwardSyncEventIds();

  void resetAndBuildSetWaitOpIndex(const SyncMap &syncMapBefore,
                                   const SyncMap &syncMapAfter);

  std::set<std::pair<int64_t, SetWaitOp *>> &
  getSetWaitOpsIndexRef(pto::PIPE pipeSrc, pto::PIPE pipeDst,
                        int64_t eventId);

  void collectSetWaitOpsIndexes(OperationBase *op, const SyncMap &syncMapBefore,
                                const SyncMap &syncMapAfter);

  bool checkBackwardSyncEventsContains(OperationBase *op,
                                       CorePipeInfo corePipeSrc,
                                       CorePipeInfo corePipeDst,
                                       int64_t eventId);

  bool checkBackwardSyncEventsContainsAfterMerge(OperationBase *op,
                                                 CorePipeInfo corePipeSrc,
                                                 CorePipeInfo corePipeDst);

  // Merge-related helpers for backward sync events and scope-level
  // optimizations.
  bool checkMergeable(Scope *scopeOp, CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, int64_t eventId,
                      bool shouldBeUsedAtleastOnce = true);

  bool checkMergeableConditionScopes(const Condition *conditionOp,
                                     CorePipeInfo corePipeSrc,
                                     CorePipeInfo corePipeDst, int64_t eventId);

  bool checkMergeableLoopScopes(const Loop *loopOp, CorePipeInfo corePipeSrc,
                                CorePipeInfo corePipeDst, int64_t eventId);

  bool checkMergeableScopeBody(
      const Scope *scopeOp,
      const std::set<std::pair<int64_t, SetWaitOp *>> &index,
      CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, int64_t eventId);

  void mergeBackwardSyncEventIds(OperationBase *op);

  bool shouldSkipBackwardSyncMerge(OperationBase *op) const;

  void collectMergeableBackwardSyncEvents(
      Scope *scopeOp,
      CorePipeEventDenseSet &toBeErased);

  void collectMergeableBackwardSyncEventsForEventId(
      Scope *scopeOp, ArrayRef<pto::TCoreType> coreTypes, int64_t eventId,
      CorePipeEventDenseSet &toBeErased);

  void collectMergeableBackwardSyncEventsForCorePair(
      Scope *scopeOp, pto::TCoreType coreSrc, pto::TCoreType coreDst,
      int64_t eventId, CorePipeEventDenseSet &toBeErased);

  bool tryCollectMergeableBackwardSyncEvent(
      Scope *scopeOp, CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
      int64_t eventId, CorePipeEventDenseSet &toBeErased);

  void eraseMergedBackwardSyncEventsFromChildScope(
      Scope *childScopeOp, const CorePipeEventDenseSet &toBeErased);

  void eraseMergedBackwardSyncEventsFromNestedScopes(
      Scope *scopeOp, const CorePipeEventDenseSet &toBeErased);

  void mergeBackwardSyncPairs(SyncMap &syncMapBefore, SyncMap &syncMapAfter);

  void appendSyncOpsForConflictPair(ConflictPair *conflictPair,
                                    SyncMap &syncMapBefore,
                                    SyncMap &syncMapAfter);

  void appendMergedBackwardSyncScopeOps(SyncMap &syncMapBefore,
                                        SyncMap &syncMapAfter);

  void insertMergedBackwardSyncPairs();

  bool pruneStaleMergedBackwardSyncPairs();

  SmallVector<OperationBase *> collectDeepestUnmergedBackwardSyncScopes();

  bool insertOuterBackwardSyncPairs(
      const SmallVectorImpl<OperationBase *> &chosenOps);

  llvm::LogicalResult considerOuterBackwardSyncPairs();

  llvm::LogicalResult reuseSyncPairToSaveEventIds();

  llvm::LogicalResult disableMultiEventIdForBarrierAllPairs();

  llvm::LogicalResult tryMovingOutBackwardSyncPairsToOuterLoops();

  Occurrence *getBeforePlaceHolderOcc(Occurrence *occ);
  Occurrence *getAfterPlaceHolderOcc(Occurrence *occ);
  Occurrence *getScopeBeginPlaceHolderOcc(Occurrence *occ);
  Occurrence *getScopeEndPlaceHolderOcc(Occurrence *occ);
};

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
