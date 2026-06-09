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

#include "PTO/Transforms/InsertSync/InsertSyncAnalysis.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "pto-insert-sync-analysis"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {

namespace {

static constexpr uint64_t kVectorRegisterSizeInBytes = 256U;
static constexpr unsigned kPipeVPruneMinRepeat = 16U;
constexpr size_t kTileRank2D = 2;
constexpr size_t kNumber2 = 2;
constexpr unsigned kInlineCapacity2 = 2;

template <typename T>
using SmallVec2 = SmallVector<T, kInlineCapacity2>;

struct RepeatAccessShape {
  SmallVec2<int64_t> fullShape;
  SmallVec2<int64_t> validShape;
  Type elementType;
};

static std::optional<RepeatAccessShape> getKnownRepeatAccessShapeFromType(Type ty) {
  if (auto tileTy = dyn_cast<TileBufType>(ty)) {
    ArrayRef<int64_t> fullShape = tileTy.getShape();
    ArrayRef<int64_t> validShape = tileTy.getValidShape();
    if (fullShape.size() != kTileRank2D || validShape.size() != kTileRank2D) return std::nullopt;
    if (fullShape[0] < 0 || fullShape[1] < 0 || validShape[0] < 0 ||
        validShape[1] < 0)
      return std::nullopt;
    if (validShape[0] > fullShape[0] || validShape[1] > fullShape[1])
      return std::nullopt;
    return RepeatAccessShape{
        SmallVec2<int64_t>{fullShape[0], fullShape[1]},
        SmallVec2<int64_t>{validShape[0], validShape[1]},
        tileTy.getElementType()};
  }

  if (auto memRefTy = dyn_cast<MemRefType>(ty)) {
    if (!memRefTy.hasStaticShape() || memRefTy.getRank() != static_cast<int64_t>(kTileRank2D))
      return std::nullopt;
    auto shape = memRefTy.getShape();
    return RepeatAccessShape{SmallVec2<int64_t>{shape[0], shape[1]},
                             SmallVec2<int64_t>{shape[0], shape[1]},
                             memRefTy.getElementType()};
  }

  return std::nullopt;
}

static std::optional<int64_t> getConstantIndex(Value value) {
  if (!value) return std::nullopt;
  APInt intValue;
  if (!matchPattern(value, m_ConstantInt(&intValue))) return std::nullopt;
  return intValue.getSExtValue();
}

static std::optional<RepeatAccessShape> getKnownRepeatAccessShape(Value access) {
  if (!access) return std::nullopt;
  auto shape = getKnownRepeatAccessShapeFromType(access.getType());
  if (!shape) return std::nullopt;
  if (auto bind = access.getDefiningOp<BindTileOp>()) {
    auto row = getConstantIndex(bind.getValidRow());
    auto col = getConstantIndex(bind.getValidCol());
    if (row && col) {
      if (*row < 0 || *col < 0 || *row > shape->fullShape[0] ||
          *col > shape->fullShape[1])
        return std::nullopt;
      shape->validShape = SmallVec2<int64_t>{*row, *col};
    } else if (bind.getValidRow() || bind.getValidCol()) {
      return std::nullopt;
    }
  }

  return shape;
}

static std::optional<BLayout> getKnownBLayout(Type ty) {
  if (auto tileTy = dyn_cast<TileBufType>(ty)) {
    int32_t layout = tileTy.getBLayoutValueI32();
    if (layout == static_cast<int32_t>(BLayout::RowMajor))
      return BLayout::RowMajor;
    if (layout == static_cast<int32_t>(BLayout::ColMajor))
      return BLayout::ColMajor;
  }

  if (auto memRefTy = dyn_cast<MemRefType>(ty)) {
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(getStridesAndOffset(memRefTy, strides, offset)) ||
        strides.size() != kTileRank2D) {
      return std::nullopt;
    }
    ArrayRef<int64_t> shape = memRefTy.getShape();
    if (strides[1] == 1 && strides[0] == shape[1]) return BLayout::RowMajor;
    if (strides[0] == 1 && strides[1] == shape[0]) return BLayout::ColMajor;
  }

  return std::nullopt;
}

static bool isProvenContiguousAccess(Value access,
                                     const RepeatAccessShape &shape) {
  auto layout = getKnownBLayout(access.getType());
  if (!layout) return false;

  int64_t fullRow = shape.fullShape[0];
  int64_t fullCol = shape.fullShape[1];
  int64_t validRow = shape.validShape[0];
  int64_t validCol = shape.validShape[1];
  if (*layout == BLayout::RowMajor)
    return validCol == fullCol || validRow == 1;
  if (*layout == BLayout::ColMajor)
    return validRow == fullRow || validCol == 1;
  return false;
}

static std::optional<unsigned> getRepeatCountForAccess(Value access) {
  if (!access) return std::nullopt;
  auto shape = getKnownRepeatAccessShape(access);
  if (!shape || !isProvenContiguousAccess(access, *shape)) return std::nullopt;

  unsigned elemBytes = pto::getPTOStorageElemByteSize(shape->elementType);
  if (elemBytes == 0) return std::nullopt;

  uint64_t validElems = static_cast<uint64_t>(shape->validShape[0]) *
                        static_cast<uint64_t>(shape->validShape[1]);
  uint64_t elemsPerRepeat = kVectorRegisterSizeInBytes / elemBytes;
  if (elemsPerRepeat == 0) return std::nullopt;

  uint64_t repeat = (validElems + elemsPerRepeat - 1U) / elemsPerRepeat;
  if (repeat > std::numeric_limits<unsigned>::max()) return std::nullopt;
  return static_cast<unsigned>(repeat);
}

static bool isSameExactAccess(const BaseMemInfo *lhs, const BaseMemInfo *rhs) {
  return lhs && rhs && *lhs == *rhs;
}

static bool containsExactAccess(const SmallVector<const BaseMemInfo *> &infos,
                                const BaseMemInfo *access) {
  return llvm::any_of(infos, [&](const BaseMemInfo *info) {
    return isSameExactAccess(info, access);
  });
}

} // namespace

static constexpr unsigned kPipeStateSize =
    static_cast<unsigned>(PipelineType::PIPE_LAST) + 1U;

static bool isValidPipeIndex(PipelineType pipe) {
  return static_cast<unsigned>(pipe) < kPipeStateSize;
}

static bool isTLoadCompound(const CompoundInstanceElement *compound) {
  return compound && compound->elementOp && isa<pto::TLoadOp>(compound->elementOp);
}

static bool isTLoadToTLoadWAWExempt(const CompoundInstanceElement *nowCompound,
                                    const CompoundInstanceElement *frontCompound) {
  return isTLoadCompound(nowCompound) && isTLoadCompound(frontCompound) &&
         nowCompound->kPipeValue == PipelineType::PIPE_MTE2 &&
         frontCompound->kPipeValue == PipelineType::PIPE_MTE2;
}

// ==============================================================================
// 1. Entry Point
// ==============================================================================

void InsertSyncAnalysis::Run(bool insertBarAllAtLast) {
  syncIndex_ = syncOperations_.size();

  for (auto &nowElement : syncIR_) {
    if (auto *nowCompound =
            dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      DealWithCompoundSync(nowCompound);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(nowElement.get())) {
      DealWithLoopSync(loopElement);
    } else if (isa<BranchInstanceElement>(nowElement.get())) {
      continue;
    } else if (isa<PlaceHolderInstanceElement>(nowElement.get())) {
      continue;
    }
  }

  if (insertBarAllAtLast) {
    InsertLastPipeAll();
  }
}

// ==============================================================================
// 2. High-Level Traversal
// ==============================================================================

void InsertSyncAnalysis::DealWithCompoundSync(
    CompoundInstanceElement *nowCompound) {
  SyncRecordList syncRecordList;
  InsertSeqSync(nowCompound, syncIR_, 0, nowCompound->GetIndex(), syncRecordList,
                std::nullopt);
}

void InsertSyncAnalysis::DealWithLoopSync(LoopInstanceElement *nowElement) {
  // Insert backward sync by copying the loop body slice and running the same
  // sequential insertion on the copied structure.
  if (nowElement->getLoopKind() != KindOfLoop::LOOP_END) {
    return;
  }

  SyncIRs backSyncIr;
  assert(syncIR_.size() >= nowElement->endId);
  for (unsigned i = nowElement->beginId; i < nowElement->endId; i++) {
    if (auto *compound = dyn_cast<CompoundInstanceElement>(syncIR_[i].get())) {
      InsertBackForSync(compound, backSyncIr, nowElement);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(syncIR_[i].get())) {
      auto loopKind = loopElement->getLoopKind();
      backSyncIr.emplace_back(loopElement->CloneFor(loopKind));
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(syncIR_[i].get())) {
      backSyncIr.emplace_back(
          branchElement->CloneBranch(branchElement->getBranchKind()));
    } else if (auto *placeHolderElement =
                   dyn_cast<PlaceHolderInstanceElement>(syncIR_[i].get())) {
      backSyncIr.emplace_back(placeHolderElement->Clone());
    }
  }
}

void InsertSyncAnalysis::InsertBackForSync(
    CompoundInstanceElement *nowCompound, SyncIRs &backSyncIr,
    const LoopInstanceElement *loopElement) {
  SyncRecordList syncRecordList;

  auto backCompound = std::make_unique<CompoundInstanceElement>(
      nowCompound->GetIndex(), nowCompound->defVec, nowCompound->useVec,
      nowCompound->kPipeValue, nowCompound->opName);
  backCompound->compoundCoreType = nowCompound->compoundCoreType;
  backCompound->elementOp = nowCompound->elementOp;

  auto *backCompoundPtr = backCompound.get();
  backSyncIr.emplace_back(std::move(backCompound));

  // Insert sync between the copied commands (j+1 slice).
  InsertSeqSync(backCompoundPtr, backSyncIr, 0,
                static_cast<int>(backSyncIr.size()) - 1, syncRecordList,
                loopElement->endId);

  // Insert sync between original and copied commands to model loop-carried deps.
  InsertSeqSync(nowCompound, syncIR_, nowCompound->GetIndex(), loopElement->endId,
                syncRecordList, loopElement->endId);
}

// ==============================================================================
// 3. Sequential Sync Insertion (Core Logic)
// ==============================================================================

bool InsertSyncAnalysis::IsNoNeedToInsertSync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound, bool isBackwardDep) const {
  const PipelineType frontPipe = frontCompound->kPipeValue;
  const PipelineType nowPipe = nowCompound->kPipeValue;
  if (frontPipe == nowPipe && frontPipe == PipelineType::PIPE_S) {
    return true;
  }

  if (nowCompound->elementOp == frontCompound->elementOp && !isBackwardDep) {
    return true;
  }

  // Do not short-circuit same-pipe pairs here. If a real memory dependency is
  // present, MemAnalyze will insert a PIPE_BARRIER to serialize that pipe,
  // matching the "bar_v/bar_m" style intra-pipe synchronization expected by
  // higher-level frontends.

  return false;
}

void InsertSyncAnalysis::InsertSeqSync(
    CompoundInstanceElement *nowCompound, SyncIRs &syncElement, int begin,
    int end, SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  const PipelineType nowPipeValue = nowCompound->kPipeValue;

  checkSyncIRIndex(syncElement, begin);
  checkSyncIRIndex(syncElement, end);

  unsigned syncIRIndex = syncElement[end]->GetIndex();
  UpdateAlreadySync(syncIR_[syncIRIndex]->pipeBefore, syncRecordList, nowPipeValue);

  for (int i = end - 1; i >= begin; i--) {
    auto &frontPtr = syncElement[i];
    unsigned frontIndex = frontPtr->GetIndex();
    assert(frontIndex < syncIR_.size());
    assert(syncIR_[frontIndex] != nullptr);
    if (auto *frontCompound =
            dyn_cast<CompoundInstanceElement>(frontPtr.get())) {
      UpdateAlreadySync(syncIR_[frontIndex]->pipeAfter, syncRecordList,
                        nowPipeValue);
      InsertSync(nowCompound, frontCompound, syncRecordList, forEndIndex);
      UpdateAlreadySync(syncIR_[frontIndex]->pipeBefore, syncRecordList,
                        nowPipeValue);
    } else if (auto *loopInstance =
                   dyn_cast<LoopInstanceElement>(frontPtr.get())) {
      int skipLoop = static_cast<int>(InsertLoopSync(
          i, nowCompound, begin, loopInstance, syncElement, syncRecordList,
          forEndIndex));
      i -= skipLoop;
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(frontPtr.get())) {
      int skipBranch = static_cast<int>(InsertBranchSync(
          i, nowCompound, begin, branchElement, syncElement, syncRecordList,
          forEndIndex));
      i -= skipBranch;
    }
  }
}

unsigned InsertSyncAnalysis::InsertLoopSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    LoopInstanceElement *loopElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (loopElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncRecordList syncRecordForList = syncRecordList;
    unsigned newBegin =
        std::max(begin, index - (loopElement->endId - loopElement->beginId));
    unsigned newEnd = index;
    InsertSeqSync(nowCompound, syncElement, static_cast<int>(newBegin),
                  static_cast<int>(newEnd), syncRecordForList, forEndIndex);
    // A loop may execute zero iterations at runtime. Keep correctness for both
    // paths by not promoting alreadySync from the loop-body traversal into the
    // outer state. We only carry syncFinder updates, matching no-else branch
    // behavior in InsertBranchSync.
    for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++)
      syncRecordList[bufferIdx].syncFinder =
          syncRecordForList[bufferIdx].syncFinder;
    return (loopElement->endId - loopElement->beginId);
  }
  return 0;
}

unsigned InsertSyncAnalysis::InsertBranchSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    BranchInstanceElement *branchElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (branchElement->getBranchKind() == KindOfBranch::IF_END) {
    SyncRecordList syncRecordIfList = syncRecordList;

    // The indices here are positions in `syncElement` (which may be a slice
    // like backSyncIr), so compute ranges relative to `index`.
    unsigned branchIf =
        index - (branchElement->endId - branchElement->beginId);
    unsigned branchElse =
        index - (branchElement->endId - branchElement->branchId);
    unsigned branchEnd = index;

    InsertSeqSync(nowCompound, syncElement, static_cast<int>(branchIf),
                  static_cast<int>(branchElse), syncRecordIfList, forEndIndex);
    if (branchElement->branchId != branchElement->endId) {
      SyncRecordList syncRecordElseList = syncRecordList;
      InsertSeqSync(nowCompound, syncElement, static_cast<int>(branchElse),
                    static_cast<int>(branchEnd), syncRecordElseList, forEndIndex);
      MergeAlreadySync(syncRecordList, syncRecordIfList, syncRecordElseList);
    } else {
      // No else-branch: do not promote `alreadySync`, but keep syncFinder
      // updates from the then-branch.
      for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++)
        syncRecordList[bufferIdx].syncFinder = syncRecordIfList[bufferIdx].syncFinder;
    }
    return (branchElement->endId - branchElement->beginId);
  } else if (branchElement->getBranchKind() == KindOfBranch::ELSE_BEGIN &&
             index != begin) {
    assert(nowCompound->GetIndex() > branchElement->branchId);
    return (branchElement->branchId - branchElement->beginId);
  }
  return 0;
}

void InsertSyncAnalysis::MergeAlreadySync(
    SyncRecordList &syncRecordList, const SyncRecordList &syncRecordIfList,
    const SyncRecordList &syncRecordElseList) const {
  for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
    for (size_t pipeIdx = 0; pipeIdx < kPipeStateSize; pipeIdx++) {
      if (syncRecordIfList[bufferIdx].alreadySync[pipeIdx] &&
          syncRecordElseList[bufferIdx].alreadySync[pipeIdx]) {
        syncRecordList[bufferIdx].alreadySync[pipeIdx] = true;
      }
    }
  }
}

// ==============================================================================
// 4. Dependency Analysis & Operation Insertion
// ==============================================================================

void InsertSyncAnalysis::InsertSync(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (IsNoNeedToInsertSync(nowCompound, frontCompound, forEndIndex.has_value())) {
    return;
  }
  MemAnalyze(nowCompound, frontCompound, syncRecordList, forEndIndex);
}

void InsertSyncAnalysis::MemAnalyze(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (isAlreadySync(nowCompound, frontCompound, syncRecordList, 0)) {
    return;
  }

  DepBaseMemInfoPairVec depVec;
  if (!IsMemInfoHasDependency(nowCompound, frontCompound, depVec)) {
    return;
  }

  if (CanPrunePipeVBarrier(nowCompound, frontCompound, depVec, forEndIndex)) {
    return;
  }

  if (forEndIndex.has_value()) {
    int eventIdNum = GetEventIdNum(depVec);
    for (int i = 1; i < eventIdNum; i++) {
      if (isAlreadySync(nowCompound, frontCompound, syncRecordList,
                        static_cast<unsigned>(i))) {
        return;
      }
    }
  }

  InsertSyncOperation(nowCompound, frontCompound, depVec, forEndIndex);
  UpdateSyncRecordInfo(frontCompound, syncRecordList);
}

bool InsertSyncAnalysis::IsMemInfoHasDependency(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  const bool useDefDependency = memAnalyzer_.DepBetween(
      nowCompound->useVec, frontCompound->defVec, depBaseMemInfosVec);
  const bool defUseDependency = memAnalyzer_.DepBetween(
      nowCompound->defVec, frontCompound->useVec, depBaseMemInfosVec);
  bool defDefDependency = false;
  if (!isTLoadToTLoadWAWExempt(nowCompound, frontCompound)) {
    defDefDependency = memAnalyzer_.DepBetween(
        nowCompound->defVec, frontCompound->defVec, depBaseMemInfosVec);
  }
  bool hasDependency =
      useDefDependency || defUseDependency || defDefDependency;

  // Special hazard: ACC (L0C) read/read cross-pipe ordering.
  //
  // Some PTO-ISA sequences have semantically "read/read" patterns on ACC, but
  // executing them concurrently across pipelines can trigger device-side issues.
  if (nowCompound->kPipeValue != frontCompound->kPipeValue) {
    DepBaseMemInfoPairVec rrDepVec;
    if (memAnalyzer_.DepBetween(nowCompound->useVec, frontCompound->useVec,
                               rrDepVec)) {
      for (auto &pair : rrDepVec) {
        if (!pair.first) continue;
        if (pair.first->scope != pto::AddressSpace::ACC) continue;
        depBaseMemInfosVec.push_back(pair);
        hasDependency = true;
      }
    }
  }

  return hasDependency;
}

bool InsertSyncAnalysis::CanPrunePipeVBarrier(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound,
    const DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) const {
  if (forEndIndex.has_value()) return false;
  if (!nowCompound || !frontCompound) return false;
  if (nowCompound->kPipeValue != PipelineType::PIPE_V ||
      frontCompound->kPipeValue != PipelineType::PIPE_V) {
    return false;
  }

  // PIPE_V has a hardware-safe same-access chain case: exact same-access
  // dependencies from the producer result to the consumer source do not require
  // a vector-pipe barrier once the producer repeat is large enough. Keep the
  // check conservative: all dependency pairs for this candidate must describe
  // the exact same access.
  SmallVec2<const BaseMemInfo *> producerAccesses;
  for (const auto &pair : depBaseMemInfosVec) {
    if (!isSameExactAccess(pair.first, pair.second)) return false;
    if (containsExactAccess(nowCompound->useVec, pair.first) &&
        containsExactAccess(frontCompound->defVec, pair.second)) {
      if (!llvm::is_contained(producerAccesses, pair.second))
        producerAccesses.push_back(pair.second);
    } else {
      return false;
    }
  }
  if (producerAccesses.empty()) return false;

  // The caller is analyzing this specific front->now dependency. Do not look
  // for a later text-order writer here; it may belong to a different branch
  // path or a zero-trip loop body.
  for (const BaseMemInfo *producerAccess : producerAccesses) {
    auto repeat = getRepeatCountForAccess(producerAccess->baseBuffer);
    if (!repeat || *repeat < kPipeVPruneMinRepeat) return false;
  }

  return true;
}

void InsertSyncAnalysis::InsertSyncOperation(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  PipelineType nowPipe = nowCompound->kPipeValue;
  PipelineType frontPipe = frontCompound->kPipeValue;
  if (nowPipe == frontPipe) {
    unsigned insertBarrierId = nowCompound->GetIndex();
    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, frontPipe, nowPipe, syncIndex_,
        insertBarrierId, forEndIndex);
    barrierOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    syncIR_[insertBarrierId]->pipeBefore.push_back(barrierOp.get());
    barrierOp->SetSyncIRIndex(insertBarrierId);

    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(barrierOp));
    syncOperations_.emplace_back(std::move(newSync));
  } else {
    unsigned insertWaitId = nowCompound->GetIndex();
    unsigned insertSetId = frontCompound->GetIndex();
    auto setOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe, syncIndex_,
        insertSetId, forEndIndex);
    auto waitOp = setOp->GetMatchSync(insertWaitId);
    SmallVector<Value> depRoots = GetMemInfoBuffers(depBaseMemInfosVec);
    setOp->depRootBuffers = depRoots;
    waitOp->depRootBuffers = depRoots;
    setOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    waitOp->SetDepSyncIRIndex(frontCompound->GetIndex());

    // Back-edge dependencies may require multi-buffer event IDs.
    if (forEndIndex.has_value()) {
      int eventIdNum = GetEventIdNum(depBaseMemInfosVec);
      setOp->eventIdNum = eventIdNum;
      waitOp->eventIdNum = eventIdNum;
    }

    syncIR_[insertSetId]->pipeAfter.push_back(setOp.get());
    syncIR_[insertWaitId]->pipeBefore.push_back(waitOp.get());

    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(setOp));
    newSync.emplace_back(std::move(waitOp));
    syncOperations_.emplace_back(std::move(newSync));
  }

  syncIndex_++;
  assert(syncOperations_.size() == syncIndex_);
}

// ==============================================================================
// 5. Sync Record Maintenance
// ==============================================================================

bool InsertSyncAnalysis::isAlreadySync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList, unsigned recordListIndex) const {
  (void)nowCompound;
  const PipelineType frontPipe = frontCompound->kPipeValue;
  if (recordListIndex >= syncRecordList.size()) return false;
  if (!isValidPipeIndex(frontPipe)) return false;
  return syncRecordList[recordListIndex]
      .alreadySync[static_cast<unsigned>(frontPipe)];
}

void InsertSyncAnalysis::UpdateAlreadySync(const SyncOps &syncVector,
                                           SyncRecordList &syncRecordList,
                                           const PipelineType nowPipeValue) const {
  for (auto *sync : syncVector) {
    for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
      if (bufferIdx == 0 && sync->eventIdNum > 1 &&
          sync->GetForEndIndex().has_value()) {
        continue;
      }
      UpdateSyncRecord(sync, syncRecordList[bufferIdx], nowPipeValue);
    }
  }
}

void InsertSyncAnalysis::UpdateSyncRecord(const SyncOperation *sync,
                                          SyncRecord &syncRecord,
                                          PipelineType nowPipeValue) const {
  PipelineType setPipeValue = sync->GetSrcPipe();
  PipelineType waitPipeValue = sync->GetDstPipe();

  // Block-sync mode behaves like a global blocking pipe-s wait.
  if (syncAnalysisMode_ == SyncAnalysisMode::BLOCKSYNC) {
    nowPipeValue = PipelineType::PIPE_S;
    waitPipeValue = PipelineType::PIPE_S;
  }

  if (!isValidPipeIndex(nowPipeValue) || !isValidPipeIndex(waitPipeValue) ||
      !isValidPipeIndex(setPipeValue)) {
    return;
  }

  auto &recordAlready = syncRecord.alreadySync;
  auto &recordFinder = syncRecord.syncFinder;

  bool barrierFinder =
      (nowPipeValue == waitPipeValue) &&
      (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER);
  if (barrierFinder) {
    recordAlready[static_cast<unsigned>(nowPipeValue)] = true;
    return;
  }

  bool canTransitivelyEliminate =
      recordAlready[static_cast<unsigned>(waitPipeValue)] ||
      (nowPipeValue == waitPipeValue);
  if (!canTransitivelyEliminate) return;
  if (recordFinder[sync->GetSyncIndex()] &&
      (sync->GetType() == SyncOperation::TYPE::SET_EVENT ||
       sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET)) {
    recordAlready[static_cast<unsigned>(setPipeValue)] = true;
  }

  if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    recordFinder[sync->GetSyncIndex()] = true;
  }
}

void InsertSyncAnalysis::UpdateSyncRecordInfo(
    const CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList) const {
  (void)frontCompound;
  assert(!syncOperations_.empty());
  auto &syncPair = syncOperations_.back();
  assert(!syncPair.empty());

  auto *newSync = syncPair[0].get();
  for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
    if (bufferIdx == 0 && newSync->eventIdNum > 1) {
      continue;
    }
    if (!isValidPipeIndex(newSync->GetSrcPipe())) continue;
    syncRecordList[bufferIdx]
        .alreadySync[static_cast<unsigned>(newSync->GetSrcPipe())] = true;
  }
}

// ==============================================================================
// 6. Final Barrier
// ==============================================================================

void InsertSyncAnalysis::InsertLastPipeAll() {
  for (auto it = syncIR_.rbegin(); it != syncIR_.rend(); ++it) {
    auto *element = it->get();
    if (isa<PlaceHolderInstanceElement>(element)) continue;

    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, PipelineType::PIPE_ALL,
        PipelineType::PIPE_ALL, syncIndex_, element->GetIndex(), std::nullopt);
    barrierOp->MarkAutoSyncTailBarrier();

    SyncOperation *barrierRawPtr = barrierOp.get();
    SmallVector<std::unique_ptr<SyncOperation>> syncGroup;
    syncGroup.emplace_back(std::move(barrierOp));
    syncOperations_.emplace_back(std::move(syncGroup));
    syncIndex_++;

    element->pipeAfter.push_back(barrierRawPtr);
    return;
  }
}

// ==============================================================================
// 7. Helpers
// ==============================================================================

bool InsertSyncAnalysis::IsMemAllocOp(Operation *op) const {
  return isa<memref::AllocOp>(op) || isa<pto::PointerCastOp>(op);
}

SmallVector<Value> InsertSyncAnalysis::GetMemInfoBuffers(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) const {
  llvm::DenseSet<Value> touchedBuffer;
  SmallVector<Value> result;
  for (auto &pair : depBaseMemInfosVec) {
    if (pair.first && pair.first->rootBuffer)
      touchedBuffer.insert(pair.first->rootBuffer);
    if (pair.second && pair.second->rootBuffer)
      touchedBuffer.insert(pair.second->rootBuffer);
  }
  for (auto v : touchedBuffer)
    result.push_back(v);
  llvm::sort(result, [](Value lhs, Value rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  });
  return result;
}

int InsertSyncAnalysis::GetEventIdNum(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) const {
  for (const auto &pair : depBaseMemInfosVec) {
    bool isLocalA =
        pair.first && (pair.first->scope == pto::AddressSpace::MAT ||
                       pair.first->scope == pto::AddressSpace::VEC);
    bool isLocalB =
        pair.second && (pair.second->scope == pto::AddressSpace::MAT ||
                        pair.second->scope == pto::AddressSpace::VEC);
    if (isLocalA || isLocalB) return kNumber2;
  }
  return 1;
}

bool InsertSyncAnalysis::IsGMHazard(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound) const {
  auto hasGM = [](const SmallVector<const BaseMemInfo *> &vec) {
    for (const auto *info : vec) {
      if (info->scope == pto::AddressSpace::GM) return true;
    }
    return false;
  };

  bool frontWritesGM = hasGM(frontCompound->defVec);
  bool frontReadsGM = hasGM(frontCompound->useVec);

  bool nowWritesGM = hasGM(nowCompound->defVec);
  bool nowReadsGM = hasGM(nowCompound->useVec);
  if (frontWritesGM && nowReadsGM) return true;  // RAW
  if (frontReadsGM && nowWritesGM) return true;  // WAR
  if (frontWritesGM && nowWritesGM) return true; // WAW

  // RAR is considered safe for GM in this simplified model.
  return false;
}

} // namespace mlir::pto
