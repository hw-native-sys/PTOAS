// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- Utility.h ---- Graph Sync Solver -----------------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
#include <deque>
#include <memory>
#include <optional>
#include <pthread.h>
#include <string>

namespace mlir::pto::syncsolver {
inline constexpr size_t kDebugIndentWidth = 2;
inline constexpr int64_t kIntraCoreEventIdCount = 8;
inline constexpr int64_t kCrossCoreEventIdCount = 16;
inline constexpr int64_t kTestIntraCoreEventIdCount = 8;
inline constexpr int64_t kTestCrossCoreEventIdCount = 999;
inline constexpr int64_t kBlockAllIntraSyncFlagId1 = 15;
inline constexpr int64_t kBlockAllIntraSyncFlagId2 = 14;
inline constexpr int64_t kReservedCrossCoreEventIdCount = 2;
inline constexpr int64_t kReservedIntraCoreEventIdCount = 0;
} // namespace mlir::pto::syncsolver

using SyncMap = llvm::MapVector<
    mlir::pto::syncsolver::OperationBase *,
    std::deque<std::unique_ptr<mlir::pto::syncsolver::SyncOp>>>;
using SyncBeforeAfterMap = std::pair<SyncMap, SyncMap>;

namespace mlir::pto {
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     TCoreType coreType) {
  switch (coreType) {
  case TCoreType::VECTOR:
    return os << "VECTOR";
  case TCoreType::CUBE:
    return os << "CUBE";
  case TCoreType::CUBE_OR_VECTOR:
    return os << "CUBE_OR_VECTOR";
  case TCoreType::CUBE_AND_VECTOR:
    return os << "CUBE_AND_VECTOR";
  }
  return os << "UNKNOWN";
}
} // namespace mlir::pto

namespace mlir::pto::syncsolver {
struct CorePipeInfo {
  pto::TCoreType coreType{pto::TCoreType::CUBE_OR_VECTOR};
  pto::PIPE pipe{pto::PIPE::PIPE_UNASSIGNED};

  CorePipeInfo() = default;

  CorePipeInfo(pto::TCoreType coreType, pto::PIPE pipe)
      : coreType(coreType), pipe(pipe) {}

  CorePipeInfo(std::pair<pto::TCoreType, pto::PIPE> corePipePair)
      : mlir::pto::syncsolver::CorePipeInfo(corePipePair.first,
                                             corePipePair.second) {}

  bool operator==(const CorePipeInfo &other) const {
    return std::tie(coreType, pipe) == std::tie(other.coreType, other.pipe);
  }

  bool operator!=(const CorePipeInfo &other) const { return !(*this == other); }

  bool operator<(const CorePipeInfo &other) const {
    return std::tie(coreType, pipe) < std::tie(other.coreType, other.pipe);
  }
};
} // namespace mlir::pto::syncsolver

namespace llvm {
template <> struct DenseMapInfo<mlir::pto::syncsolver::CorePipeInfo> {
  using CorePipePairTy = std::pair<mlir::pto::TCoreType, mlir::pto::PIPE>;
  static inline mlir::pto::syncsolver::CorePipeInfo getEmptyKey() {
    // Use sentinel values that are guaranteed never to appear as valid keys
    return DenseMapInfo<CorePipePairTy>::getEmptyKey();
  }
  static inline mlir::pto::syncsolver::CorePipeInfo getTombstoneKey() {
    // Use a different set of sentinel values
    return DenseMapInfo<CorePipePairTy>::getTombstoneKey();
  }
  static unsigned
  getHashValue(const mlir::pto::syncsolver::CorePipeInfo &val) {
    // Combine hashes of members
    return DenseMapInfo<CorePipePairTy>::getHashValue({val.coreType, val.pipe});
  }
  static bool isEqual(const mlir::pto::syncsolver::CorePipeInfo &lhs,
                      const mlir::pto::syncsolver::CorePipeInfo &rhs) {
    // Use the defined operator==
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir::pto::syncsolver {

enum SyncMode {
  INTRA_CORE_SYNC,
  CROSS_CORE_SYNC,
  TEST_INTRA_CORE_MODE,
  TEST_CROSS_CORE_MODE,
};

struct SyncSolverOptions {
  // Synchronization mode.
  const SyncMode syncMode;

  // Architecture is memory based (A2/A3).
  const bool isMemBasedArch;

  // Architecture is register based (A5).
  const bool isRegBasedArch;

  // Decompose MMAD L1 ops into simpler ops for better sync handling.
  bool decomposeMmadl1Op{false};

  // Enable unit-flag feature handling.
  bool enableUnitFlagFeature{false};

  // Always use scalar pipe as waiting pipe in sync pairs.
  bool alwaysUsePipeSAsWaitingPipe{false};

  // Consider outer backward-sync pairs optimization.
  bool considerOuterBackwardSyncPairs{true};

  // Try merging backward sync pairs and moving them to an outer scope.
  bool moveOutAndMergeBackwardSyncPairs{true};

  // Disable multi-event-id usage for barrier-all pipe pairs.
  bool disableMultiEventIdForBarrierAllPairs{true};

  // Reuse existing sync pairs to save event ids.
  bool reuseSyncPairToSaveEventIds{false};

  // Use different flag-ids for multibuffer backward sync pairs.
  bool useDifferentMultiBufferFlagIds{false};

  // Ignore workspace function arguments.
  bool intraCoreIgnoreWorkSpaceFunctionArguments{false};

  // Optional driver-side cap for available event-id slots.
  std::optional<int64_t> eventIdNumMax;

  SyncSolverOptions(SyncMode syncMode, bool isMemBasedArch, bool isRegBasedArch)
      : syncMode(syncMode), isMemBasedArch(isMemBasedArch),
        isRegBasedArch(isRegBasedArch) {
    decomposeMmadl1Op = false;
    alwaysUsePipeSAsWaitingPipe =
        !isTestMode() && isCrossCoreMode() && isMemBasedArch;
    reuseSyncPairToSaveEventIds = isIntraCoreMode();
    useDifferentMultiBufferFlagIds = !isCrossCoreMode();
  }

  bool isCrossCoreMode() const {
    return syncMode == SyncMode::CROSS_CORE_SYNC ||
           syncMode == SyncMode::TEST_CROSS_CORE_MODE;
  }

  bool isIntraCoreMode() const {
    return syncMode == SyncMode::INTRA_CORE_SYNC ||
           syncMode == SyncMode::TEST_INTRA_CORE_MODE;
  }

  bool isTestMode() const {
    return syncMode == SyncMode::TEST_INTRA_CORE_MODE ||
           syncMode == SyncMode::TEST_CROSS_CORE_MODE;
  }
};

struct Occurrence;

struct ProcessingOrder {
  Occurrence *occ1{nullptr};
  Occurrence *occ2{nullptr};
  RWOperation *rwOp1{nullptr};
  RWOperation *rwOp2{nullptr};
  bool isUseless{false};
  ProcessingOrder(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                  RWOperation *rwOp2, bool isUseless)
      : occ1(occ1), occ2(occ2), rwOp1(rwOp1), rwOp2(rwOp2),
        isUseless(isUseless) {};
};

class UnitFlagInfo : public UnitFlagInfoBase {
public:
  Occurrence *linkedElementAsSet{nullptr};
  Occurrence *linkedElementAsWait{nullptr};

public:
  UnitFlagInfo() = default;
  ~UnitFlagInfo() override = default;

  explicit UnitFlagInfo(const UnitFlagInfoBase &other)
      : UnitFlagInfoBase(other) {}

  void reset() {
    UnitFlagInfoBase::reset();
    linkedElementAsSet = nullptr;
    linkedElementAsWait = nullptr;
  }

  void merge(const UnitFlagInfo &other, Occurrence *occ1, Occurrence *occ2,
             bool asSet = true, bool asWait = true) {
    UnitFlagInfoBase::merge(other, asSet, asWait);
    if (asSet && occ2 != nullptr) {
      linkedElementAsSet = occ2;
    }
    if (asWait && occ1 != nullptr) {
      linkedElementAsWait = occ1;
    }
  }
};

struct Occurrence {
  OperationBase *op{nullptr};
  Occurrence *parentOcc{nullptr};
  int depth{-1};
  int startIndex{-1};
  int endIndex{-1};
  int syncIrIndex{-1};
  int syncIrEndIndex{-1};
  int loopSplitIndex{-1};
  bool hasUnitFlagFeat{false};
  UnitFlagInfo unitFlagInfo;
  llvm::SmallVector<Occurrence *> childOccs;

  Occurrence(OperationBase *op, Occurrence *parentOcc, int depth,
             int startIndex, int endIdx)
      : op(op), parentOcc(parentOcc), depth(depth), startIndex(startIndex),
        endIndex(endIdx) {}

  // Return true if occ1 and occ2 have the same immediate parent occurrence.
  static bool sameScope(Occurrence *occ1, Occurrence *occ2);

  // Return depth (number of ancestors + 1) for the given occurrence.
  static int getDepth(Occurrence *occ);

  // Walk up parents to find the first ancestor occurrence associated with 'op'.
  Occurrence *getParentWithOp(Operation *op, bool assertExists = true);
  Occurrence *getParentWithOp(OperationBase *op, bool assertExists = true);

  // Return the ancestor that is `dist` levels above this occurrence.
  Occurrence *getNthParent(int dist);

  // Compute/return the pair of sibling occurrences just below their LCA.
  static std::pair<Occurrence *, Occurrence *> getLCAPair(Occurrence *occ1,
                                                          Occurrence *occ2);

  template <typename OpTy> Occurrence *getParentOfType() {
    Occurrence *cur = this->parentOcc;
    while (cur != nullptr && !isa_and_present<OpTy>(cur->op)) {
      cur = cur->parentOcc;
    }
    return cur;
  }

  // Find and return the nearest parent occurrence that is a loop.
  static Occurrence *getParentloop(Occurrence *occ);

  // Find and return the nearest parent occurrence that is a condition.
  static Occurrence *getParentCondition(Occurrence *occ);

  // Return true if this occurrence is a strict ancestor of `occ`.
  bool isProperAncestor(Occurrence *occ);

  // Collect and return all occurrence parents (in upward order).
  llvm::SmallVector<Occurrence *> getAllParents();

  static Occurrence *getUnlikelyParentCondition(Occurrence *occ);
};

struct EventIdNode;

struct ConflictPair {
  static int globalIdCounter;

  const int id;
  RWOperation *const op1;
  RWOperation *const op2;
  OperationBase *setOp{nullptr};
  OperationBase *waitOp{nullptr};
  Occurrence *setOcc{nullptr};
  Occurrence *waitOcc{nullptr};
  const CorePipeInfo setCorePipeInfo;
  const CorePipeInfo waitCorePipeInfo;
  int startIndex{-1};
  int endIndex{-1};
  bool isInnerBackward{false};
  bool isUseless{false};
  bool dontReuse{false};
  bool dontCheckForConflict{false};
  bool couldNotRun{false};
  bool setOnLastIterOnly{false};
  bool waitOnFirstIterOnly{false};
  bool replacedWithUnitFlag{false};
  bool movedToOuterLoop{false};
  Loop *backwardSyncLoopOp{nullptr};
  Occurrence *backwardSyncLoopOcc{nullptr};
  EventIdInfo eventIdInfo;
  EventIdNode *eventIdNode{nullptr};

  ConflictPair(RWOperation *op1, RWOperation *op2, OperationBase *setOp,
               OperationBase *waitOp, Occurrence *setOcc, Occurrence *waitOcc,
               CorePipeInfo setCorePipeInfo, CorePipeInfo waitCorePipeInfo,
               int startIndex, int endIndex)
      : id(globalIdCounter++), op1(op1), op2(op2), setOp(setOp), waitOp(waitOp),
        setOcc(setOcc), waitOcc(waitOcc), setCorePipeInfo(setCorePipeInfo),
        waitCorePipeInfo(waitCorePipeInfo), startIndex(startIndex),
        endIndex(endIndex) {};

  bool isBarrier() const { return setCorePipeInfo == waitCorePipeInfo; }

  // Human-readable description of the conflict pair for debug printing.
  std::string str() const;

  // Update the stored set/wait operation pointers and their indices from
  // occurrences.
  void updateSetWaitOccs(Occurrence *setOcc, Occurrence *waitOcc) {
    if (setOcc != nullptr) {
      this->setOcc = setOcc;
      this->setOp = setOcc->op;
      this->startIndex = setOcc->endIndex;
    }
    if (waitOcc != nullptr) {
      this->waitOcc = waitOcc;
      this->waitOp = waitOcc->op;
      this->endIndex = waitOcc->startIndex;
    }
  }

  std::unique_ptr<ConflictPair> clone() {
    auto clonedConflictPair = std::make_unique<ConflictPair>(
        op1, op2, setOp, waitOp, setOcc, waitOcc, setCorePipeInfo,
        waitCorePipeInfo, startIndex, endIndex);
    clonedConflictPair->isInnerBackward = isInnerBackward;
    clonedConflictPair->isUseless = isUseless;
    clonedConflictPair->dontReuse = dontReuse;
    clonedConflictPair->couldNotRun = couldNotRun;
    clonedConflictPair->setOnLastIterOnly = setOnLastIterOnly;
    clonedConflictPair->waitOnFirstIterOnly = waitOnFirstIterOnly;
    clonedConflictPair->replacedWithUnitFlag = replacedWithUnitFlag;
    clonedConflictPair->backwardSyncLoopOp = backwardSyncLoopOp;
    clonedConflictPair->backwardSyncLoopOcc = backwardSyncLoopOcc;
    clonedConflictPair->eventIdInfo = eventIdInfo;
    clonedConflictPair->eventIdNode = eventIdNode;
    return clonedConflictPair;
  }

  std::unique_ptr<ConflictPair> clone(Occurrence *setOcc, Occurrence *waitOcc) {
    auto clonedConflictPair = this->clone();
    clonedConflictPair->updateSetWaitOccs(setOcc, waitOcc);
    return clonedConflictPair;
  }
};

struct EventIdNode {
public:
  const int64_t id{-1};
  ConflictPair *const initConflictPair;
  const int64_t eventIdNum;
  const bool reversePriority;

private:
  static int globalIdCounter;
  llvm::SmallVector<int64_t> eventIds;
  llvm::DenseMap<ConflictPair *, int64_t> conflictPairs;

public:
  EventIdNode(ConflictPair *conflictPair, int64_t eventIdNum,
              bool reversePriority)
      : id(globalIdCounter++), initConflictPair(conflictPair),
        eventIdNum(eventIdNum), reversePriority(reversePriority) {
    insertConflictPair(conflictPair);
  }

  std::unique_ptr<EventIdNode> clone() {
    auto clonedNode = std::make_unique<EventIdNode>(
        initConflictPair, eventIdNum, reversePriority);
    clonedNode->eventIds = eventIds;
    clonedNode->conflictPairs = conflictPairs;
    return clonedNode;
  }

  void insertConflictPair(ConflictPair *conflictPair) {
    conflictPairs[conflictPair] += 1;
  }

  void eraseConflictPair(ConflictPair *conflictPair) {
    if (!(conflictPairs[conflictPair] -= 1)) {
      conflictPairs.erase(conflictPair);
    }
  }

  const llvm::SmallVector<int64_t> &getEventIds() { return eventIds; }

  void setEventIds(const llvm::SmallVector<int64_t> &newEventIds) {
    eventIds = newEventIds;
  }

  std::string str(bool printConflictPairs = true) {
    std::string ret = "EventIdNode" + std::to_string(id) + "[";
    ret += "eventIdNum(" + std::to_string(eventIdNum) + ")";
    ret += ", ";
    ret += "revPri(" + std::to_string(reversePriority) + ")";
    ret += ", ";
    ret += "eventIds(";
    for (auto eventId : eventIds) {
      ret += std::to_string(eventId) + ", ";
    }
    ret += ")";
    ret += "]\n";
    if (printConflictPairs) {
      for (auto [conflictPair, frq] : conflictPairs) {
        assert(frq > 0);
        ret += std::string(kDebugIndentWidth, ' ') + conflictPair->str() + "\n";
      }
    }
    ret.pop_back();
    return ret;
  }
};

struct MmadL1SyncArgs {
  MmadL1SyncArgs() = default;
  MmadL1SyncArgs(Value l0WaitL1AEvent, Value l0WaitL1BEvent,
                 Value l1AWaitL0Event, Value l1BWaitL0Event, Value kLoopDBCond,
                 Value bwdPipeMPipeMTE1Event0, Value bwdPipeMPipeMTE1Event1)
      : l0WaitL1AEvent(l0WaitL1AEvent), l0WaitL1BEvent(l0WaitL1BEvent),
        l1AWaitL0Event(l1AWaitL0Event), l1BWaitL0Event(l1BWaitL0Event),
        kLoopDBCond(kLoopDBCond),
        bwdPipeMPipeMTE1Event0(bwdPipeMPipeMTE1Event0),
        bwdPipeMPipeMTE1Event1(bwdPipeMPipeMTE1Event1) {}

  Value l0WaitL1AEvent;
  Value l0WaitL1BEvent;
  Value l1AWaitL0Event;
  Value l1BWaitL0Event;
  Value kLoopDBCond;
  Value bwdPipeMPipeMTE1Event0;
  Value bwdPipeMPipeMTE1Event1;
};

// Check if two integer ranges intersect.
bool checkRangesIntersect(int l1, int r1, int l2, int r2);

// Return explicit integer ranges covered by a conflict pair (empty for
// barrier).
std::vector<std::pair<int, int>> getRanges(ConflictPair *conflictPair);

// Return hardware-available EVENT ids for a given (setPipe, waitPipe) pair.
int64_t
getHWAvailableEventIdNum(SyncMode syncMode,
                         pto::PIPE setPipe = pto::PIPE::PIPE_UNASSIGNED,
                         pto::PIPE waitPipe = pto::PIPE::PIPE_UNASSIGNED);

llvm::SmallVector<int64_t>
getHWAvailableEventIds(SyncMode syncMode,
                       pto::PIPE setPipe = pto::PIPE::PIPE_UNASSIGNED,
                       pto::PIPE waitPipe = pto::PIPE::PIPE_UNASSIGNED);

// Create a boolean Value that is true for the first iteration of `forOp`.
Value getIsFirstIterationValue(scf::ForOp forOp, Location loc,
                               IRRewriter &rewriter);

// Create a boolean Value that is true for the last iteration of `forOp`.
Value getIsLastIterationValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter);

// Helper to stringify a Value to std::string for logging.
std::string op2str(Value val);

// Helper to stringify an Operation pointer to std::string for logging.
std::string op2str(Operation *op);

// Verify that all loop-like parents of `op` are SCF ForOps (returns true if
// so).
bool checkAllParentLoopsAreForLoops(Operation *op);

// Cast `val` to i64 type if it is not already.
Value getValueOrCreateCastToI64(IRRewriter &rewriter, Location loc, Value val);

pto::TCoreType getOppositeCoreType(pto::TCoreType coreType);

template <typename OpTy>
llvm::FailureOr<std::pair<OpTy, OpTy>> getFirstLastOp(Operation *parentOp) {
  OpTy firstOp{nullptr};
  OpTy lastOp{nullptr};
  parentOp->walk<WalkOrder::PreOrder, ForwardIterator>([&](OpTy op) {
    firstOp = op;
    return WalkResult::interrupt();
  });
  if (firstOp == nullptr) {
    return llvm::failure();
  }
  parentOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](OpTy op) {
    lastOp = op;
    return WalkResult::interrupt();
  });
  assert(lastOp != nullptr);
  return std::make_pair(firstOp, lastOp);
}

bool isEmptyScope(Scope *scope);

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H
