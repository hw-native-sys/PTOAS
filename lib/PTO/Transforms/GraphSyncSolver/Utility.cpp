// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- Utility.cpp ---- Graph Sync Solver ---------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace pto::syncsolver;

int ConflictPair::globalIdCounter = 0;
int EventIdNode::globalIdCounter = 0;

bool Occurrence::sameScope(const Occurrence *occ1, const Occurrence *occ2) {
  ASSERT(occ1 != nullptr && occ1->parentOcc != nullptr);
  ASSERT(occ2 != nullptr && occ2->parentOcc != nullptr);
  return occ1->parentOcc == occ2->parentOcc;
}

int Occurrence::getDepth(const Occurrence *occ) {
  int ret = 0;
  while (occ != nullptr) {
    occ = occ->parentOcc;
    ret++;
  }
  return ret;
}

Occurrence *Occurrence::getParentWithOp(Occurrence *occ, const OperationBase *op,
                                        bool assertExists) {
  ASSERT(op != nullptr);
  while (occ != nullptr) {
    if (occ->op == op) {
      return occ;
    }
    occ = occ->parentOcc;
  }
  ASSERT(!assertExists);
  return nullptr;
}

Occurrence *Occurrence::getParentWithOp(Occurrence *occ, const Operation *op,
                                        bool assertExists) {
  ASSERT(op != nullptr);
  while (occ != nullptr) {
    if (occ->op != nullptr && occ->op->op == op) {
      return occ;
    }
    occ = occ->parentOcc;
  }
  ASSERT(!assertExists);
  return nullptr;
}

Occurrence *Occurrence::getNthParent(Occurrence *occ, int dist) {
  while (dist > 0) {
    ASSERT(occ != nullptr);
    occ = occ->parentOcc;
    --dist;
  }
  ASSERT(occ != nullptr);
  return occ;
}

std::pair<Occurrence *, Occurrence *> Occurrence::getLCAPair(Occurrence *occ1,
                                                             Occurrence *occ2) {
  ASSERT(occ1 != nullptr && occ2 != nullptr);
  int depth1 = getDepth(occ1);
  int depth2 = getDepth(occ2);
  if (depth1 < depth2) {
    occ2 = getNthParent(occ2, depth2 - depth1);
  } else if (depth1 > depth2) {
    occ1 = getNthParent(occ1, depth1 - depth2);
  }
  while (occ1->parentOcc != occ2->parentOcc) {
    occ1 = occ1->parentOcc;
    occ2 = occ2->parentOcc;
  }
  ASSERT(occ1 != occ2);
  return std::make_pair(occ1, occ2);
}

Occurrence *Occurrence::getParentloop(Occurrence *occ) {
  ASSERT(occ != nullptr);
  Occurrence *cur = occ->parentOcc;
  while (cur != nullptr && !isa<Loop>(cur->op)) {
    cur = cur->parentOcc;
  }
  return cur;
}

Occurrence *Occurrence::getParentCondition(Occurrence *occ) {
  ASSERT(occ != nullptr);
  Occurrence *cur = occ->parentOcc;
  while (cur != nullptr && !isa<Condition>(cur->op)) {
    cur = cur->parentOcc;
  }
  return cur;
}

Occurrence *Occurrence::getUnlikelyParentCondition(Occurrence *occ) {
  ASSERT(occ != nullptr && occ->op != nullptr);
  if (auto *parentConditionOp =
          OperationBase::getUnlikelyParentCondition(occ->op)) {
    return getParentWithOp(occ, parentConditionOp, /*assertExists=*/true);
  }
  return nullptr;
}

bool Occurrence::isProperAncestor(const Occurrence *occ) const {
  ASSERT(occ != nullptr);
  int depth1 = getDepth(this);
  int depth2 = getDepth(occ);
  if (depth1 >= depth2) {
    return false;
  }
  const Occurrence *cursor = occ;
  for (int dist = depth2 - depth1; dist > 0; --dist) {
    ASSERT(cursor != nullptr);
    cursor = cursor->parentOcc;
  }
  return cursor == this;
}

llvm::SmallVector<Occurrence *> Occurrence::getAllParents() const {
  llvm::SmallVector<Occurrence *> collectedParents;
  Occurrence *occ = this->parentOcc;
  while (occ != nullptr) {
    collectedParents.push_back(occ);
    occ = occ->parentOcc;
  }
  return collectedParents;
}

llvm::SmallVector<OperationBase *> OperationBase::getAllParents() const {
  llvm::SmallVector<OperationBase *> collectedParents;
  OperationBase *op = this->parentOp;
  while (op != nullptr) {
    collectedParents.push_back(op);
    op = op->parentOp;
  }
  return collectedParents;
}

bool OperationBase::sameScope(const OperationBase *op1,
                              const OperationBase *op2) {
  ASSERT(op1->parentOp != nullptr);
  ASSERT(op2->parentOp != nullptr);
  return op1->parentOp == op2->parentOp;
}

int OperationBase::getDepth() const {
  int ret = 0;
  const OperationBase *op = this;
  while (op != nullptr) {
    op = op->parentOp;
    ret++;
  }
  return ret;
}

OperationBase *OperationBase::getNthParent(OperationBase *op, int dist) {
  while (dist > 0) {
    ASSERT(op != nullptr);
    op = op->parentOp;
    --dist;
  }
  return op;
}

std::pair<OperationBase *, OperationBase *>
OperationBase::getLCAPair(OperationBase *op1, OperationBase *op2) {
  ASSERT(op1 != nullptr && op2 != nullptr);
  int depth1 = op1->getDepth();
  int depth2 = op2->getDepth();
  if (depth1 < depth2) {
    op2 = getNthParent(op2, depth2 - depth1);
  } else if (depth1 > depth2) {
    op1 = getNthParent(op1, depth1 - depth2);
  }
  while (op1->parentOp != op2->parentOp) {
    op1 = op1->parentOp;
    op2 = op2->parentOp;
  }
  ASSERT(op1 != nullptr && op2 != nullptr);
  ASSERT(op1->parentOp == op2->parentOp);
  return std::make_pair(op1, op2);
}

OperationBase *OperationBase::getParentloop(OperationBase *op) {
  ASSERT(op != nullptr);
  OperationBase *cur = op->parentOp;
  while (cur != nullptr && !isa<Loop>(cur)) {
    cur = cur->parentOp;
  }
  return cur;
}

OperationBase *OperationBase::getParentCondition(OperationBase *op) {
  ASSERT(op != nullptr);
  OperationBase *cur = op->parentOp;
  while (cur != nullptr && !isa<Condition>(cur)) {
    cur = cur->parentOp;
  }
  return cur;
}

bool OperationBase::isProperAncestor(const OperationBase *op) const {
  ASSERT(op != nullptr);
  int depth1 = this->getDepth();
  int depth2 = op->getDepth();
  if (depth1 >= depth2) {
    return false;
  }
  const OperationBase *cursor = op;
  for (int dist = depth2 - depth1; dist > 0; --dist) {
    ASSERT(cursor != nullptr);
    cursor = cursor->parentOp;
  }
  return cursor == this;
}

OperationBase *OperationBase::getUnlikelyParentCondition(OperationBase *op) {
  ASSERT(op != nullptr);
  auto *cur = OperationBase::getParentCondition(op);
  while (cur != nullptr) {
    auto *conditionOp = dyn_cast<Condition>(cur);
    ASSERT(conditionOp != nullptr);
    if (conditionOp->isUnlikely &&
        conditionOp->getTrueScope()->isProperAncestor(op)) {
      return cur;
    }
    cur = OperationBase::getParentCondition(cur);
  }
  return nullptr;
}

namespace mlir::pto::syncsolver {

// Check if two integer ranges intersect (half-open semantics: [l, r) )
bool checkRangesIntersect(int l1, int r1, int l2, int r2) {
  return r1 > l2 && r2 > l1;
}

// Return explicit integer ranges covered by a conflict pair (barrier -> empty).
std::vector<std::pair<int, int>> getRanges(ConflictPair *conflictPair) {
  ASSERT(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  std::vector<std::pair<int, int>> ret;
  ret.emplace_back(conflictPair->startIndex, conflictPair->endIndex);
  return ret;
}

// Return the hardware-available EVENT ids for a given (setPipe, waitPipe) pair.
// Respects reserved ids for special pipe pairs and returns a vector of usable
// ids.
int64_t getHWAvailableEventIdNum(SyncMode syncMode, pto::PIPE setPipe,
                                 pto::PIPE waitPipe) {
  if (syncMode == SyncMode::INTRA_CORE_SYNC) {
    const llvm::DenseMap<std::tuple<PIPE, PIPE>, int64_t> reservedEventIdNum = {
        {{pto::PIPE::PIPE_V, pto::PIPE::PIPE_S}, 1},
        {{pto::PIPE::PIPE_S, pto::PIPE::PIPE_V}, 1},
        {{pto::PIPE::PIPE_M, pto::PIPE::PIPE_FIX}, 1},
        {{pto::PIPE::PIPE_FIX, pto::PIPE::PIPE_M}, 1},
    };
    int64_t eventIdNum = kIntraCoreEventIdNum;
    eventIdNum -= reservedIntraCoreEventIdNum;
    auto it = reservedEventIdNum.find({setPipe, waitPipe});
    if (it != reservedEventIdNum.end()) {
      eventIdNum -= it->second;
    }
    return eventIdNum;
  } else if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    int64_t eventIdNum = kCrossCoreEventIdNum;
    eventIdNum -= reservedCrossCoreEventIdNum;
    return eventIdNum;
  } else if (syncMode == SyncMode::TEST_INTRA_CORE_MODE) {
    int64_t eventIdNum = kTestIntraCoreEventIdNum;
    return eventIdNum;
  } else if (syncMode == SyncMode::TEST_CROSS_CORE_MODE) {
    int64_t eventIdNum = kTestCrossCoreEventIdNum;
    return eventIdNum;
  }
  llvm_unreachable("getHWAvailableEventIdNum: unhandled SyncMode");
}

SmallVector<int64_t> getHWAvailableEventIds(SyncMode syncMode,
                                            pto::PIPE setPipe,
                                            pto::PIPE waitPipe) {
  if (syncMode == SyncMode::INTRA_CORE_SYNC) {
    const llvm::DenseMap<std::tuple<PIPE, PIPE>, int64_t> reservedEventIdNum = {
        {{pto::PIPE::PIPE_V, pto::PIPE::PIPE_S}, 1},
        {{pto::PIPE::PIPE_S, pto::PIPE::PIPE_V}, 1},
        {{pto::PIPE::PIPE_M, pto::PIPE::PIPE_FIX}, 1},
        {{pto::PIPE::PIPE_FIX, pto::PIPE::PIPE_M}, 1},
    };
    int64_t eventIdNum = kIntraCoreEventIdNum;
    eventIdNum -= reservedIntraCoreEventIdNum;
    auto it = reservedEventIdNum.find({setPipe, waitPipe});
    if (it != reservedEventIdNum.end()) {
      eventIdNum -= it->second;
    }
    SmallVector<int64_t> hwAvailableEventIds(eventIdNum);
    std::iota(hwAvailableEventIds.begin(), hwAvailableEventIds.end(),
              static_cast<int64_t>(0));
    return hwAvailableEventIds;
  } else if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    int64_t eventIdNum = kCrossCoreEventIdNum;
    eventIdNum -= reservedCrossCoreEventIdNum;
    SmallVector<int64_t> hwAvailableEventIds(eventIdNum);
    std::iota(hwAvailableEventIds.begin(), hwAvailableEventIds.end(),
              static_cast<int64_t>(0));
    return hwAvailableEventIds;
  } else if (syncMode == SyncMode::TEST_INTRA_CORE_MODE) {
    int64_t eventIdNum = kTestIntraCoreEventIdNum;
    SmallVector<int64_t> availableEventIds(eventIdNum);
    std::iota(availableEventIds.begin(), availableEventIds.end(),
              static_cast<int64_t>(0));
    return availableEventIds;
  } else if (syncMode == SyncMode::TEST_CROSS_CORE_MODE) {
    int64_t eventIdNum = kTestCrossCoreEventIdNum;
    SmallVector<int64_t> availableEventIds(eventIdNum);
    std::iota(availableEventIds.begin(), availableEventIds.end(),
              static_cast<int64_t>(0));
    return availableEventIds;
  }
  llvm_unreachable("getHWAvailableEventIds: unhandled SyncMode");
}

// Build a Value that is true for the first iteration of the given scf::ForOp.
// Inserted at the start of the loop body and compares induction var with lower.
Value getIsFirstIterationValue(scf::ForOp forOp, Location loc,
                               IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value lowerBound = forOp.getLowerBound();
  Value currentInd = forOp.getInductionVar();
  Value isFirstIter = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lowerBound, currentInd);
  return isFirstIter;
}

// Build a Value that is true for the last iteration of the given scf::ForOp.
// Compares next induction value with the upper bound.
Value getIsLastIterationValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value currentInd = forOp.getInductionVar();
  Value nextInd = rewriter.create<arith::AddIOp>(loc, currentInd, step);
  Value isLastIter = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, nextInd, upperBound);
  return isLastIter;
}

// Convert a Value to its string representation for debugging/logging.
std::string op2str(Value val) {
  std::string printBuffer;
  llvm::raw_string_ostream os(printBuffer);
  val.print(os);
  return os.str();
}

// Convert an Operation pointer to its string representation.
std::string op2str(Operation *op) {
  std::string printBuffer;
  llvm::raw_string_ostream os(printBuffer);
  op->print(os);
  return os.str();
}

// Verify that all loop-like parents of `op` are SCF ForOps. Used to ensure
// certain multi-buffer/loop transformations are safe to apply.
bool checkAllParentLoopsAreForLoops(Operation *op) {
  while (op != nullptr) {
    auto parLoop = op->getParentOfType<LoopLikeOpInterface>();
    if (parLoop != nullptr && !isa<scf::ForOp>(parLoop)) {
      return false;
    }
    op = parLoop;
  }
  return true;
}

Value getValueOrCreateCastToI64(IRRewriter &rewriter, Location loc, Value val) {
  ASSERT(isa<OpResult>(val));
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(val);
  if (!val.getType().isInteger(kPTOI64BitWidth)) {
    if (val.getType().isIndex()) {
      val = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIntegerType(kPTOI64BitWidth), val);
    } else if (val.getType().isInteger()) {
      val = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(kPTOI64BitWidth),
                                            val);
    } else {
      llvm_unreachable("unhandled casting type");
    }
  }
  return val;
}

pto::TCoreType getOppositeCoreType(pto::TCoreType coreType) {
  switch (coreType) {
  case pto::TCoreType::CUBE:
    return pto::TCoreType::VECTOR;
  case pto::TCoreType::VECTOR:
    return pto::TCoreType::CUBE;
  case pto::TCoreType::CUBE_OR_VECTOR:
    return pto::TCoreType::CUBE_OR_VECTOR;
  case pto::TCoreType::CUBE_AND_VECTOR:
    return pto::TCoreType::CUBE_AND_VECTOR;
  }
  return pto::TCoreType::CUBE_OR_VECTOR;
}

bool isEmptyScope(const Scope *scope) {
  for (auto &childOp : scope->body) {
    if (isa<RWOperation>(childOp.get())) {
      return false;
    }
    if (auto *childScope = dyn_cast<Scope>(childOp.get())) {
      if (!isEmptyScope(childScope)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace mlir::pto::syncsolver
