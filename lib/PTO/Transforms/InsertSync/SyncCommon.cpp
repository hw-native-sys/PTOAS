// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <map>
#include <memory>
#include <utility>
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"


#define DEBUG_TYPE "pto-inject-sync"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr size_t kSingleUnitFlagCondition = 1;
constexpr size_t kDualUnitFlagConditions = 2;
constexpr size_t kFirstConditionIndex = 0;
constexpr size_t kSecondConditionIndex = 1;
} // namespace

SmallVector<const void *>
mlir::pto::canonicalizeSyncDepRoots(const SmallVector<Value> &roots) {
  SmallVector<const void *> result;
  result.reserve(roots.size());
  for (Value value : roots) {
    if (!value) {
      continue;
    }
    result.push_back(value.getAsOpaquePointer());
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool mlir::pto::hasSameSyncDepRoots(const SyncOperation *lhs,
                                    const SyncOperation *rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->depRootBuffers.empty() || rhs->depRootBuffers.empty()) {
    return false;
  }
  return canonicalizeSyncDepRoots(lhs->depRootBuffers) ==
         canonicalizeSyncDepRoots(rhs->depRootBuffers);
}

bool SyncOperation::operator==(const SyncOperation &other) const {
  if (this->GetDstPipe() == other.GetDstPipe()) {
    if (this->GetSrcPipe() == other.GetSrcPipe()) {
      if (this->GetType() == other.GetType()) {
        return this->GetEventIDs().empty() && other.GetEventIDs().empty();
      }
    }
  }
  return false;
}

std::string SyncOperation::TypeName(SyncOperation::TYPE t) {
  static std::map<TYPE, std::string> typeNameMap = {
      {TYPE::SET_EVENT, "set_flag"},
      {TYPE::WAIT_EVENT, "wait_flag"},
      {TYPE::PIPE_BARRIER, "pipe_barrier"},
      // 保留这些名称以防将来需要扩展
      {TYPE::PIPE_BARRIER_CUBE, "pipe_barrier_cube"},
      {TYPE::PIPE_BARRIER_VECTOR, "pipe_barrier_vector"},
      {TYPE::SYNC_BLOCK_SET, "sync_block_set"},
      {TYPE::SYNC_BLOCK_WAIT, "sync_block_wait"},
      {TYPE::SYNC_BLOCK_ALL, "sync_block_all"}};
  auto typeIt = typeNameMap.find(t);
  if (typeIt != typeNameMap.cend()) {
    return typeIt->second;
  }
  llvm_unreachable("Not supported sync type");
  return "";
}

std::string SyncOperation::GetCoreTypeName(TCoreType t) const {
  static std::map<TCoreType, std::string> coreTypeNameMap = {
      {TCoreType::CUBE, "CUBE"},
      {TCoreType::VECTOR, "VECTOR"},
      {TCoreType::CUBE_OR_VECTOR, "CUBE_OR_VECTOR"},
      {TCoreType::CUBE_AND_VECTOR, "CUBE_AND_VECTOR"}};
  auto typeIt = coreTypeNameMap.find(t);
  if (typeIt != coreTypeNameMap.cend()) {
    return typeIt->second;
  }
  llvm_unreachable("Not supported sync type");
  return "";
}

std::unique_ptr<SyncOperation>
SyncOperation::GetMatchSync(unsigned index) const {
  TYPE newType{TYPE::PIPE_BARRIER};
  static std::map<TYPE, TYPE> syncPair = {
      {TYPE::SET_EVENT, TYPE::WAIT_EVENT},
      {TYPE::WAIT_EVENT, TYPE::SET_EVENT},
      {TYPE::PIPE_BARRIER, TYPE::PIPE_BARRIER},
      {TYPE::SYNC_BLOCK_SET, TYPE::SYNC_BLOCK_WAIT},
      {TYPE::SYNC_BLOCK_WAIT, TYPE::SYNC_BLOCK_SET},
      {TYPE::SYNC_BLOCK_ALL, TYPE::SYNC_BLOCK_ALL},
  };
  auto syncIt = syncPair.find(this->type_);
  if (syncIt != syncPair.cend()) {
    newType = syncIt->second;
  }

  auto res =
      std::make_unique<SyncOperation>(newType, this->srcPipe_, this->dstPipe_,
                                      kSyncIndex_, index, this->forEndIndex_);
  res->eventIds = this->eventIds;
  res->depRootBuffers = this->depRootBuffers;
  res->eventIdNum = this->eventIdNum;
  res->isCompensation = this->isCompensation;
  res->autoSyncTailBarrier = this->autoSyncTailBarrier;
  res->SetDepSyncIRIndex(this->GetDepSyncIRIndex());
  // Slot info: propagate as a default; callers that know the matched side's
  // slot SSA (e.g. wait-side gets the consumer slot rather than the
  // producer slot) overwrite after GetMatchSync.
  res->slotSSAExpr = this->slotSSAExpr;
  res->slotCount = this->slotCount;
  return res;
}

void SyncOperation::SetPipeAll() {
  // set current sync to pipe_all
  this->type_ = TYPE::PIPE_BARRIER;
  // [修正] 使用 pto::PipelineType
  this->srcPipe_ = PipelineType::PIPE_ALL;
  this->dstPipe_ = PipelineType::PIPE_ALL;
}

bool SyncOperation::isSyncSetType() const {
  auto type = this->GetType();
  return type == TYPE::SET_EVENT || type == TYPE::SYNC_BLOCK_SET;
}

bool SyncOperation::isSyncWaitType() const {
  auto type = this->GetType();
  return type == TYPE::WAIT_EVENT || type == TYPE::SYNC_BLOCK_WAIT;
}

bool SyncOperation::isBarrierType() const {
  auto type = this->GetType();
  return type == TYPE::PIPE_BARRIER || type == TYPE::PIPE_BARRIER_CUBE ||
         type == TYPE::PIPE_BARRIER_VECTOR;
}

bool InstanceElement::RemoveSync(SyncOps &syncVector,
                                 const SyncOperation *sync) {
  auto it = std::find(syncVector.begin(), syncVector.end(), sync);
  if (it == syncVector.end()) {
    return false;
  }
  syncVector.erase(it);
  return true;
}

std::unique_ptr<InstanceElement>
LoopInstanceElement::CloneFor(KindOfLoop loopKind) const {
  unsigned index =
      loopKind == KindOfLoop::LOOP_BEGIN ? this->beginId : this->endId;
  checkCondition(this->beginId != this->endId,
                 "LoopInstanceElement clone failed.");
  auto res =
      std::make_unique<LoopInstanceElement>(index, beginId, endId, loopKind);
  res->elementOp = elementOp;
  return res;
}

std::unique_ptr<BranchInstanceElement>
BranchInstanceElement::CloneBranch(KindOfBranch branchKind) const {
  if (branchKind == KindOfBranch::ELSE_BEGIN) {
    auto res = std::make_unique<BranchInstanceElement>(
        branchId, beginId, branchId, endId, KindOfBranch::ELSE_BEGIN);
    res->elementOp = elementOp;
    return res;
  }
  if (branchKind == KindOfBranch::IF_END) {
    auto res = std::make_unique<BranchInstanceElement>(
        endId, beginId, branchId, endId, KindOfBranch::IF_END);
    res->elementOp = elementOp;
    return res;
  }
  checkCondition(branchKind == KindOfBranch::IF_BEGIN,
                 "element expected to be of kind IF_BEGIN");
  auto res = std::make_unique<BranchInstanceElement>(
      beginId, beginId, branchId, endId, KindOfBranch::IF_BEGIN);
  res->elementOp = elementOp;
  return res;
}

std::unique_ptr<PlaceHolderInstanceElement>
PlaceHolderInstanceElement::Clone() const {
  return std::make_unique<PlaceHolderInstanceElement>(this->kIndex,
                                                      this->parentScopeId);
}

bool LoopInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for LoopInstanceElement'sconst classof");
  return e->GetKind() == KindTy::LOOP;
}

bool CompoundInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for CompoundInstanceElement's classof");
  return e->GetKind() == KindTy::COMPOUND;
}

bool BranchInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for BranchInstanceElement's classof");
  return e->GetKind() == KindTy::BRANCH;
}

bool PlaceHolderInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for PlaceHolderInstanceElement's classof");
  return e->GetKind() == KindTy::PLACE_HOLDER;
}

UNIT_FLAG CompoundInstanceElement::getUnitFlagMode() const {
  static DenseMap<std::pair<UNIT_FLAG, UNIT_FLAG>, UNIT_FLAG> possibleStates = {
      {std::make_pair(UNIT_FLAG::DISABLED, UNIT_FLAG::DISABLED),
       UNIT_FLAG::DISABLED},
      {std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                      UNIT_FLAG::ENABLED_WITH_UPDATE),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE, UNIT_FLAG::DISABLED),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::DISABLED, UNIT_FLAG::ENABLED_WITH_UPDATE),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                      UNIT_FLAG::ENABLED_ONLY_FIRST_ITER),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::DISABLED, UNIT_FLAG::ENABLED_ONLY_FIRST_ITER),
       UNIT_FLAG::ENABLED_ONLY_FIRST_ITER},
      {std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                      UNIT_FLAG::ENABLED_WITH_UPDATE),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER, UNIT_FLAG::DISABLED),
       UNIT_FLAG::ENABLED_ONLY_LAST_ITER},
      {std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                      UNIT_FLAG::ENABLED_ONLY_FIRST_ITER),
       UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS},
  };
  auto it = possibleStates.find(
      std::make_pair(unitFlagModeAsSet, unitFlagModeAsWait));
  if (it == possibleStates.end()) {
    llvm_unreachable("unit-flag state not handled");
  }
  return it->second;
}

static Value getIsNotDeadLoopValue(scf::ForOp forOp, Location loc,
                                   OpBuilder &rewriter) {
  Value upperBound = forOp.getUpperBound();
  Value lowerBound = forOp.getLowerBound();
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                        lowerBound, upperBound);
}

std::optional<mlir::Value>
CompoundInstanceElement::getUnitFlagCond(Location loc, OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<Value> conditions;

  if (linkedUnitFlagCompAsWait &&
      (linkedUnitFlagCompAsWait->unitFlagModeAsSet ==
           UNIT_FLAG::ENABLED_ONLY_LAST_ITER ||
       linkedUnitFlagCompAsWait->unitFlagModeAsSet ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS)) {
    if (auto forOp = dyn_cast<scf::ForOp>(
            linkedUnitFlagCompAsWait->elementOp->getParentOp())) {
      rewriter.setInsertionPoint(forOp);
      Value cond = getIsNotDeadLoopValue(forOp, loc, rewriter);
      conditions.push_back(cond);
    }
  }
  if (linkedUnitFlagCompAsSet &&
      (linkedUnitFlagCompAsSet->unitFlagModeAsWait ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_ITER ||
       linkedUnitFlagCompAsSet->unitFlagModeAsWait ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS)) {
    if (auto forOp = dyn_cast<scf::ForOp>(
            linkedUnitFlagCompAsSet->elementOp->getParentOp())) {
      rewriter.setInsertionPoint(elementOp);
      Value cond = getIsNotDeadLoopValue(forOp, loc, rewriter);
      conditions.push_back(cond);
    }
  }

  if (conditions.empty()) {
    return nullptr;
  } else if (conditions.size() == kSingleUnitFlagCondition) {
    return conditions[kFirstConditionIndex];
  } else if (conditions.size() == kDualUnitFlagConditions) {
    rewriter.setInsertionPoint(elementOp);
    return rewriter.create<arith::OrIOp>(
        loc, conditions[kFirstConditionIndex], conditions[kSecondConditionIndex]);
  } else {
    llvm_unreachable("unexpected/unhandled number of unit-flag conditions.");
  }
}

namespace mlir::pto {

bool checkAllParentLoopsAreForLoops(Operation *op) {
  // This helper guards counted-loop-only unit-flag logic. Generic while
  // synchronization is handled by the loop translator and must not enter
  // this IV-based path.
  while ((op = op->getParentOfType<LoopLikeOpInterface>())) {
    if (!isa<scf::ForOp>(op)) {
      return false;
    }
  }
  return true;
}

void checkSyncIRIndex(const SyncIRs &syncIR, int index) {
  if (index < 0 || index >= static_cast<int>(syncIR.size())) {
    llvm_unreachable("index out of bounds when accessing syncIR");
  }
}

void checkCondition(bool condition, const std::string &message) {
  if (!condition) {
    llvm_unreachable(message.c_str());
  }
}

} // namespace mlir::pto
