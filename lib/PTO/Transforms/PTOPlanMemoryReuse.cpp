// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPlanMemoryReuse.cpp ----Plan Buffer Memory Address --------------===//
//===----------------------------------------------------------------------===//

#include "PTOPlanMemory.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <utility>

#define DEBUG_TYPE "pto-plan-memory"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X)

using namespace mlir;
using namespace pto;

namespace {
constexpr int kSingleBufferCount = 1;
constexpr int kDoubleBufferCount = 2;
} // namespace

void MemPlan::ReportMemLifeDebugInfo(
    const StorageEntry *rootStorageEntry) const {
  LDBG("-------------------------- Buffer2Life --------------------------\n");
  MemLifeDebugInfo(rootStorageEntry);
  for (auto &StorageEntry : rootStorageEntry->mergedChildren) {
    MemLifeDebugInfo(StorageEntry);
  }
}

void MemPlan::MemLifeDebugInfo(const StorageEntry *storageEntry) const {
  for (auto &buffer : storageEntry->inplaceBuffers) {
    if (buffer.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
        LDBG("Buffer : " << allocOp.getResult() << "\n");
      }
    }
  }
  for (auto &bufferLife : storageEntry->bufferLifeVec) {
    (void)bufferLife;
    LDBG("bufferLife : "
         << "allocTime : " << bufferLife->allocTime
         << " , freeTime : " << bufferLife->freeTime << "\n");
  }
  LDBG("\n");
}

void MemPlan::ReportCurEntryDebugInfo(const StorageEntry *curEntry) const {
  for (auto &buffer : curEntry->inplaceBuffers) {
    if (buffer.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
        LDBG("buffer : ");
        LDBG(allocOp.getResult());
      }
    }
  }
}

StorageEntry *
MemPlan::GetReorderRootStorageEntry(StorageEntry *rootStorageEntry) {
  if (rootStorageEntry->bufInfo->bufferScope != pto::AddressSpace::VEC) {
    return rootStorageEntry;
  }
  SmallVector<StorageEntry *> origStorageEntryVec = {rootStorageEntry};
  origStorageEntryVec.insert(origStorageEntryVec.end(),
                             rootStorageEntry->mergedChildren.begin(),
                             rootStorageEntry->mergedChildren.end());

  // reorder storage entrys: dma touched buffers + other buffers + scalar
  // touched buffers
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  SmallVector<StorageEntry *> touchPipeScalarStorageEntryVec;
  for (auto &storageEntry : origStorageEntryVec) {
    for (auto &buffer : storageEntry->inplaceBuffers) {
      if (dmaFirstPipelineOpt.IsDmaBuffer(buffer)) {
        reorderedStorageEntryVec.push_back(storageEntry);
        break;
      }
      if (dmaFirstPipelineOpt.IsScalarBuffer(buffer)) {
        touchPipeScalarStorageEntryVec.push_back(storageEntry);
        break;
      }
    }
  }
  for (auto &storageEntry : origStorageEntryVec) {
    auto it1 = std::find(reorderedStorageEntryVec.begin(),
                         reorderedStorageEntryVec.end(), storageEntry);
    auto it2 = std::find(touchPipeScalarStorageEntryVec.begin(),
                         touchPipeScalarStorageEntryVec.end(), storageEntry);
    if (it1 == reorderedStorageEntryVec.end() &&
        it2 == touchPipeScalarStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
    }
  }

  reorderedStorageEntryVec.insert(reorderedStorageEntryVec.end(),
                                  touchPipeScalarStorageEntryVec.begin(),
                                  touchPipeScalarStorageEntryVec.end());

  // Ensure that ping pong is continuously plan mem in the multi buffer.
  ReorderContinuousPingPongEntry(reorderedStorageEntryVec);
  StorageEntry *reorderedRootStorageEntry = reorderedStorageEntryVec[0];
  reorderedRootStorageEntry->mergedChildren.clear();
  for (size_t j = 1; j < reorderedStorageEntryVec.size(); ++j) {
    reorderedRootStorageEntry->mergedChildren.push_back(
        reorderedStorageEntryVec[j]);
  }
  return reorderedRootStorageEntry;
}

void MemPlan::ReorderContinuousPingPongEntry(
    SmallVector<StorageEntry *> &storageEntryVec) const {
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  for (auto &storageEntry : storageEntryVec) {
    auto it = std::find(reorderedStorageEntryVec.begin(),
                        reorderedStorageEntryVec.end(), storageEntry);
    if (it == reorderedStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
      if (storageEntry->multiBufferNum == kDoubleBufferCount &&
          storageEntry->relationPongEntry) {
        // Ping Pong continuous save.
        reorderedStorageEntryVec.push_back(storageEntry->relationPongEntry);
      }
    }
  }
  reorderedStorageEntryVec.swap(storageEntryVec);
}

std::pair<size_t, size_t>
MemPlan::GetBufferSpaceInfo(const pto::AddressSpace &space) const {
  switch (space) {
  case pto::AddressSpace::VEC:
    return std::make_pair(ubAlignSize, ubSpaceSize);
  case pto::AddressSpace::MAT:
    return std::make_pair(l1AlignSize, l1SpaceSize);
  case pto::AddressSpace::ACC:
    return std::make_pair(l0cAlignSize, l0cSpaceSize);
  case pto::AddressSpace::LEFT:
    return std::make_pair(l0aAlignSize, l0aSpaceSize);
  case pto::AddressSpace::RIGHT:
    return std::make_pair(l0bAlignSize, l0bSpaceSize);
  case pto::AddressSpace::BIAS:
    return std::make_pair(biasAlignSize, biasSpaceSize);
  case pto::AddressSpace::SCALING:
    return std::make_pair(scalingAlignSize, scalingSpaceSize);
  case pto::AddressSpace::Zero:
  case pto::AddressSpace::GM:
    return std::make_pair(size_t{0}, size_t{0});
  }

  llvm_unreachable("Temporarily unsupported memory buffer space !");
}

LogicalResult MemPlan::MultiSpecPlan(SpecInfo &si, MemBoundList &outline,
                                     PlanRecHis &history, StorageEntry *entry) {
  LogicalResult planResult = failure();
  for (int i = si.specLevel; i >= si.minLevel; i--) {
    planResult = SpecAlloc(outline, history, entry, si, i);
    if (succeeded(planResult)) {
      if (si.childIdx == si.specStartIdx) {
        // In roll back plan, when the specified specStartIdx is reached,
        // the subsequent plan still adopts the maxLevel strategy.
        si.specLevel = si.maxLevel;
      }
      si.childIdx++;
      break;
    }
  }
  return planResult;
}

LogicalResult MemPlan::SpecAlloc(MemBoundList &outline, PlanRecHis &his,
                                 StorageEntry *e, const SpecInfo &si,
                                 int localLevel) {
  if (std::any_of(his.begin(), his.end(),
                  [e](PlanRecord &r) { return r.entry && r.entry == e; })) {
    // If the plan has already been completed, return success directly.
    return success();
  }
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    uint64_t allocOffset = (*start)->offset;
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      // if index & addr are as same as last rollback result,
      // continue to find next result
      if (IsSamePlanAsLastRollBack(allocOffset, e->childIdx, si) ||
          VerifyConflictStage0(e, last)) {
        start = end;
        break;
      }
      if (size < e->alignedConstBits) {
        continue;
      }
      uint64_t pongOffset{0};
      OutlineSectionInfo section(start, end, size, false);
      if (localLevel == SPEC_LEVEL_1 &&
          VerifyConflictStage1(outline, his, e, section, pongOffset)) {
        break;
      }
      if (VerifyConflictStage2(his, e, localLevel, start, outline)) {
        break;
      }
      e->bitsOffset = allocOffset;
      UpdateOutline(outline, his, e, section, localLevel);
      if (localLevel == SPEC_LEVEL_1) {
        PlanRelationPongEntryAddress(pongOffset, e);
        SpecAllocRelationPongEntry(outline, his, e, pongOffset);
      }
      LDBG("APPLY_SPEC_LEVEL:  " << localLevel << "\n");
      bool needRecord =
          allocatedEntry.end() ==
          std::find(allocatedEntry.begin(), allocatedEntry.end(), e);
      if (needRecord) {
        allocatedEntry.push_back(e);
      }
      return success();
    }
  }
  return failure();
}

LoopLikeOpInterface
MemPlan::GetBufferParentLoop(const SmallVector<Value> &buffers) const {
  llvm::SmallSet<LoopLikeOpInterface, 1> parentLoopVec;
  for (auto buffer : buffers) {
    if (!buffer.getDefiningOp()) {
      if (!isa<scf::ForOp>(buffer.getParentBlock()->getParentOp()))
        llvm::report_fatal_error("expected loop-carried block argument");
      // Init args and region iter arg are inplace, ignore Region Iter Arg
      // without DefineOp.
      continue;
    }
    LoopLikeOpInterface bufferParentLoop = getParentLoop(buffer);
    if (bufferParentLoop) {
      parentLoopVec.insert(bufferParentLoop);
    } else {
      return nullptr;
    }
  }
  if (parentLoopVec.size() == 1) {
    return *parentLoopVec.begin();
  }
  return nullptr;
}

bool MemPlan::VerifyConflictStage1(MemBoundList &, PlanRecHis &his,
                                   StorageEntry *e,
                                   const OutlineSectionInfo &outlineInfo,
                                   uint64_t &pongOffset) {
  if (outlineInfo.mem_start != outlineInfo.mem_end) {
    return true;
  }
  auto reuseBoundStorageEntry = (*outlineInfo.mem_start)->lastStorageEntry;
  if (!reuseBoundStorageEntry) {
    // This area has not been planed, so there is no need to consider it.
    return true;
  }

  StorageEntry *multiRelationPongEntry =
      GetMultiRelationPongEntry(reuseBoundStorageEntry);
  if (multiRelationPongEntry) {
    if (e->multiBufferNum == kSingleBufferCount ||
        (e->multiBufferNum == kDoubleBufferCount && e->relationPongEntry &&
         (e->relationPongEntry->bitsOffset != 0))) {
      auto parentLoop1 = GetBufferParentLoop(e->inplaceBuffers);
      auto parentLoop2 =
          GetBufferParentLoop(reuseBoundStorageEntry->inplaceBuffers);
      if (!(parentLoop1 != nullptr && parentLoop2 != nullptr &&
            parentLoop1 == parentLoop2)) {
        // Cannot be reused under the same for.
        return true;
      }
      // There are two situations
      // 1. Single reuse DB.
      // 2. DB reuse DB.
      pongOffset = multiRelationPongEntry->bitsOffset;
      bool conflict = std::any_of(
          his.begin(), his.end(), [pongOffset, e, this](PlanRecord &r) {
            return this->IsBufferLifeVecConflict(r, pongOffset, e);
          });
      if (!conflict) {
        return false;
      }
    }
  }
  return true;
}

StorageEntry *
MemPlan::GetMultiRelationPongEntry(const StorageEntry *reuseBoundStorageEntry) {
  if (reuseBoundStorageEntry->multiBufferNum == kDoubleBufferCount &&
      reuseBoundStorageEntry->relationPongEntry &&
      (reuseBoundStorageEntry->relationPongEntry->bitsOffset != 0)) {
    // If the reuseBoundStorageEntry itself requires db, directly match and
    // return relationPongEntry.
    return reuseBoundStorageEntry->relationPongEntry;
  }
  auto iter = pingEntry2RelationPongEntry.find(reuseBoundStorageEntry);
  if (iter != pingEntry2RelationPongEntry.end()) {
    // If the reuseBoundStorageEntry itself is single, but has already been
    // reused with db and has an extra pong StorageEntry is added.
    return iter->second.get();
  }
  return nullptr;
}

void MemPlan::SpecAllocRelationPongEntry(MemBoundList &outline, PlanRecHis &his,
                                         StorageEntry *e, uint64_t offset) {
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    // Find the MemBound corresponding to the Pong offset.
    if ((*start)->offset != offset) {
      continue;
    }
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      if (size < e->alignedConstBits) {
        continue;
      }
      StorageEntry *pongStorageEntry = nullptr;
      auto iter = pingEntry2RelationPongEntry.find(e);
      if (iter != pingEntry2RelationPongEntry.end()) {
        pongStorageEntry = iter->second.get();
      }
      if (e->multiBufferNum == kDoubleBufferCount && e->relationPongEntry) {
        pongStorageEntry = e->relationPongEntry;
      }
      if (!pongStorageEntry)
        llvm::report_fatal_error("pong storage entry not found");
      UpdateOutline(outline, his, pongStorageEntry,
                    OutlineSectionInfo(start, end, size, true), SPEC_LEVEL_1);
      return;
    }
  }
}

bool MemPlan::IsBufferLifeVecConflict(PlanRecord &r, uint64_t offset,
                                      const StorageEntry *e) const {
  if ((r.firstMemBound->offset + r.allExtent > offset) &&
      (r.firstMemBound->offset < offset + e->alignedConstBits)) {
    if (HasSemanticConflict(e, r.firstMemBound->bufferLifeVec))
      return true;
    DenseMap<ValuePair, BufferLife> intersection =
        GetOverlapBufferLife(r.entry->bufferLifeVec, e->bufferLifeVec);
    return !intersection.empty();
  }
  return false;
}

void MemPlan::PlanRelationPongEntryAddress(uint64_t offset, StorageEntry *e) {
  if (e->multiBufferNum == kSingleBufferCount) {
    std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
    entry->bufInfo = e->bufInfo;
    entry->bufferLifeVec = e->bufferLifeVec;
    entry->alignedConstBits = e->alignedConstBits;
    entry->inplaceBuffers = e->inplaceBuffers;
    entry->multiBufferNum = e->multiBufferNum;
    entry->bitsOffset = offset;
    pingEntry2RelationPongEntry[e] = std::move(entry);
  } else if (e->multiBufferNum == kDoubleBufferCount) {
    e->relationPongEntry->bitsOffset = offset;
  } else {
    llvm_unreachable("Does not support multi buffer num greater than 2 !");
  }
}

bool MemPlan::VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline) {
  if (specLevel != SPEC_LEVEL_2) {
    return false;
  }
  bool touchMemCanUse = false;
  MemBoundListConstIter foundMem;

  for (auto iter = start; iter != outline.end(); ++iter) {
    uint64_t offset = (*iter)->offset;
    bool conflict =
        std::any_of(his.begin(), his.end(), [offset, e, this](PlanRecord &r) {
          return (r.firstMemBound->offset + r.allExtent > offset) &&
                 (r.firstMemBound->offset < offset + e->alignedConstBits) &&
                 this->PipeConflict(r.entry, e, this->pipeDmaConflictMap);
        });
    // if conflict, continue finding the first bound that has no conflict
    // if last bound do not meet the size, continue
    if (conflict ||
        ((*iter == outline.back()) && (*iter)->extent < e->alignedConstBits)) {
      continue;
    }
    touchMemCanUse = true;
    foundMem = iter;
    break;
  }

  if (touchMemCanUse) {
    bool conflict = (foundMem != start);
    start = conflict ? --foundMem : start;
    return conflict;
  }
  // if cannot find a bound that has no conflict with current entry,
  return true;
}

bool MemPlan::PipeConflict(const StorageEntry *e1, const StorageEntry *e2,
                           DenseMap<StorageEntryPair, bool> &conflictMap) {
  if (e1 == nullptr || e2 == nullptr) {
    return false;
  }
  auto sePair = std::make_pair(e1, e2);
  auto [iter, isInserted] = conflictMap.try_emplace(sePair, false);
  if (!isInserted) {
    return iter->second;
  }

  for (const Value var1 : e1->inplaceBuffers) {
    for (const Value var2 : e2->inplaceBuffers) {
      bool conflict = dmaFirstPipelineOpt.BufferPipeConflict(var1, var2);
      if (conflict) {
        iter->second = true;
        return true;
      }
    }
  }
  return false;
}

void MemPlan::UpdateOutline(MemBoundList &outline, PlanRecHis &his,
                            StorageEntry *e,
                            const OutlineSectionInfo &outlineInfo,
                            int localLevel) const {
  auto start = outlineInfo.mem_start;
  MemBoundListConstIter end = outlineInfo.mem_end;
  // outline
  // |-------start+end-------------|
  // |--head--|--split e--|--tail--|
  uint64_t res = outlineInfo.size - e->alignedConstBits;
  std::shared_ptr<MemoryBound> last = *end;
  ++end;
  std::shared_ptr<MemoryBound> bound;
  SmallVector<std::shared_ptr<MemoryBound>> splitBound;
  // split e, to get Boundbound
  if (splitOutline) {
    // add splitBound by splitting e to section
    AddMemBoundInSectionalWay(e, start, end, splitBound);
  } else {
    // origin outline
    BufferLifeVec life(e->bufferLifeVec.begin(), e->bufferLifeVec.end());
    MergeBufferLife(start, end, life);
    splitBound.emplace_back(std::make_shared<MemoryBound>(
        life, e->bitsOffset, e->alignedConstBits, e));
  }

  // insert tail memory bound
  if (res > 0) {
    bound = std::make_shared<MemoryBound>(last->bufferLifeVec,
                                          last->offset + last->extent - res,
                                          res, last->lastStorageEntry);
    end = outline.insert(end, bound);
  }
  // insert split e memory bound
  for (int i = static_cast<int>(splitBound.size()) - 1; i >= 0; --i) {
    end = outline.insert(end, splitBound[i]);
  }
  // record the current plan of first split entry in his
  his.emplace_back(PlanRecord{localLevel,
                              e->childIdx,
                              res > 0,
                              false,
                              splitBound.size(),
                              e,
                              e->alignedConstBits,
                              splitBound[0],
                              {},
                              outlineInfo.isDirectlyRollback});
  PlanRecord &r = his.back();
  r.replaced.splice(r.replaced.begin(), outline, start, end);
}

void MemPlan::AddMemBoundInSectionalWay(
    StorageEntry *e, MemBoundListConstIter start, MemBoundListConstIter end,
    SmallVector<std::shared_ptr<MemoryBound>> &splitBound) const {
  // |--outline1--|--outline2--|--outline3--|
  // |---------e------------ |
  // |--split e1 -|-split e2-|
  for (auto iter = start; iter != end; ++iter) {
    BufferLifeVec life(e->bufferLifeVec.begin(), e->bufferLifeVec.end());
    life.insert(life.end(), (*iter)->bufferLifeVec.begin(),
                (*iter)->bufferLifeVec.end());
    // merge the buffer life
    MergeBufferVec(life);
    // get the extent
    uint64_t size = 0;
    if (std::distance(start, iter) == std::distance(start, end) - 1) {
      // deal with the last split e2
      size = e->bitsOffset + e->alignedConstBits - (*iter)->offset;
    } else {
      size = (*iter)->extent;
    }
    splitBound.emplace_back(
        std::make_shared<MemoryBound>(life, (*iter)->offset, size, e));
  }
}

inline void MemPlan::MergeBufferLife(MemBoundList::const_iterator start,
                                     MemBoundList::const_iterator end,
                                     BufferLifeVec &newLife) const {
  size_t size = 0;
  for (auto it = start; it != end; ++it) {
    size += (*it)->bufferLifeVec.size();
  }
  newLife.reserve(size);
  for (auto it = start; it != end; ++it) {
    newLife.insert(newLife.end(), (*it)->bufferLifeVec.begin(),
                   (*it)->bufferLifeVec.end());
  }
  MergeBufferVec(newLife);
}

void MemPlan::MergeBufferVec(BufferLifeVec &bufferLife) const {
  if (bufferLife.empty()) {
    return;
  }
  BufferLifeVec mergedLife;
  mergedLife.reserve(bufferLife.size());
  // sort life by alloc and free time
  std::sort(bufferLife.begin(), bufferLife.end(), CompareBufferLife());
  int start = bufferLife[0]->allocTime;
  int end = bufferLife[0]->freeTime;
  auto buffer = bufferLife[0]->buffer;
  // merge life
  for (size_t i = 1; i < bufferLife.size(); i++) {
    auto &life = bufferLife[i];
    if (life->allocTime <= end + 1) {
      end = end < life->freeTime ? life->freeTime : end;
    } else {
      mergedLife.emplace_back(std::make_unique<BufferLife>(buffer, start, end));
      buffer = life->buffer;
      start = life->allocTime;
      end = life->freeTime;
    }
  }
  mergedLife.emplace_back(std::make_unique<BufferLife>(buffer, start, end));
  bufferLife.swap(mergedLife);
}
