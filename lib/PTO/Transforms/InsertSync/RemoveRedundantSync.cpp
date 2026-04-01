// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "pto-inject-sync"

using namespace mlir;
using namespace mlir::pto;

namespace {

using SyncPair = std::pair<SyncOperation *, SyncOperation *>;

SmallVector<const void *> canonicalizeDepRoots(const SmallVector<Value> &roots) {
  SmallVector<const void *> result;
  result.reserve(roots.size());
  for (Value v : roots) {
    if (!v) {
      continue;
    }
    result.push_back(v.getAsOpaquePointer());
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool hasSameDepRoots(const SyncOperation *lhs, const SyncOperation *rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->depRootBuffers.empty() || rhs->depRootBuffers.empty()) {
    // Missing dependency signature => be conservative and keep sync.
    return false;
  }
  auto lhsRoots = canonicalizeDepRoots(lhs->depRootBuffers);
  auto rhsRoots = canonicalizeDepRoots(rhs->depRootBuffers);
  return lhsRoots == rhsRoots;
}

std::vector<SyncPair> collectPairedSyncOps(SyncOperations &syncOperations) {
  std::vector<SyncPair> syncOps;
  for (auto &syncPair : syncOperations) {
    if (syncPair.size() != 2)
      continue;
    syncOps.emplace_back(syncPair[0].get(), syncPair[1].get());
  }
  return syncOps;
}

bool compareSyncPairPriority(const SyncPair &lhs, const SyncPair &rhs) {
  auto *syncOp1 = lhs.first;
  auto *syncOp2 = rhs.first;

  bool hasLoop1 = syncOp1->GetForEndIndex().has_value();
  bool hasLoop2 = syncOp2->GetForEndIndex().has_value();
  if (hasLoop1 && hasLoop2) {
    if (syncOp1->GetForEndIndex().value() != syncOp2->GetForEndIndex().value())
      return syncOp1->GetForEndIndex().value() > syncOp2->GetForEndIndex().value();
    return syncOp1->GetSyncIndex() > syncOp2->GetSyncIndex();
  }
  if (hasLoop1 != hasLoop2)
    return hasLoop1;
  return syncOp1->GetSyncIndex() > syncOp2->GetSyncIndex();
}

bool shouldKeepSyncPair(SyncOperation *setFlag, SyncOperation *waitFlag) {
  if (setFlag->eventIdNum != 1 || waitFlag->eventIdNum != 1)
    return true;
  if (setFlag->isCompensation || waitFlag->isCompensation)
    return true;
  return !hasSameDepRoots(setFlag, waitFlag);
}

template <typename SyncContainer>
void eraseSyncFromList(SyncContainer &syncs, SyncOperation *target) {
  auto it = std::find(syncs.begin(), syncs.end(), target);
  if (it != syncs.end())
    syncs.erase(it);
}

void markSyncPairUseless(SyncIRs &syncIR, SyncOperation *setFlag,
                         SyncOperation *waitFlag) {
  eraseSyncFromList(syncIR[setFlag->GetSyncIRIndex()]->pipeAfter, setFlag);
  eraseSyncFromList(syncIR[waitFlag->GetSyncIRIndex()]->pipeBefore, waitFlag);
  setFlag->uselessSync = true;
  waitFlag->uselessSync = true;
}

} // namespace

void RemoveRedundantSync::Run() {
  std::vector<SyncPair> syncOps = collectPairedSyncOps(syncOperations_);
  std::sort(syncOps.begin(), syncOps.end(), compareSyncPairPriority);

  for (auto [setFlag, waitFlag] : syncOps) {
    if (shouldKeepSyncPair(setFlag, waitFlag))
      continue;
    if (CheckAllSync(setFlag, waitFlag))
      markSyncPairUseless(syncIR_, setFlag, waitFlag);
  }
}

bool RemoveRedundantSync::CheckAllSync(SyncOperation *setFlag,
                                       SyncOperation *waitFlag) {
  // syncFinder 用于跟踪在当前范围内，哪些 SyncIndex 的 Set 已经被看到了。
  // 如果随后看到了对应的 Wait，说明找到了一对完整的内部同步。
  SmallVector<bool> syncFinder(syncOperations_.size(), false);
  
  unsigned int begin = setFlag->GetSyncIRIndex();
  unsigned int end = waitFlag->GetSyncIRIndex();
  auto forEndIndex = setFlag->GetForEndIndex();

  if (begin < end) {
    // 普通的前向依赖
    return CheckRepeatSync(begin, end, syncFinder, setFlag);
  } else {
    // Back-edge pruning is intentionally disabled in correctness-first mode.
    (void)forEndIndex;
    return false;
  }
}

bool RemoveRedundantSync::CheckRepeatSync(unsigned int begin, unsigned int end,
                                          SmallVector<bool> &syncFinder,
                                          SyncOperation *setFlag) {
  checkCondition(begin <= end, "expected begin <= end");
  checkSyncIRIndex(syncIR_, end);
  
  bool res = false;
 
  // 1. 检查起始节点的 Post (Set)
  for (auto &relatedSync : syncIR_[begin]->pipeAfter) {
    res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
  }
 
  // 2. 遍历中间节点
  for (unsigned i = begin + 1; i <= end - 1; i++) {
    checkSyncIRIndex(syncIR_, i);
    
    // Check Pre (Wait)
    for (auto &relatedSync : syncIR_[i]->pipeBefore) {
      res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
    }
 
    // Recursion into Branch
    if (auto *branchElement = dyn_cast<BranchInstanceElement>(syncIR_[i].get())) {
      if (CheckBranchBetween(branchElement, syncFinder, setFlag, end, i)) {
        return true;
      }
    }
    
    // Recursion into Loop
    if (auto *forElement = dyn_cast<LoopInstanceElement>(syncIR_[i].get())) {
      if (CheckLoopBetween(forElement, setFlag, i)) {
        return true;
      }
    }
 
    // Check Post (Set)
    for (auto &relatedSync : syncIR_[i]->pipeAfter) {
      res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
    }
  }
 
  // 3. 检查结束节点的 Pre (Wait)
  for (auto &relatedSync : syncIR_[end]->pipeBefore) {
    res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
  }
  
  return res;
}

bool RemoveRedundantSync::CheckBranchBetween(
    BranchInstanceElement *branchElement, SmallVector<bool> syncFinder,
    SyncOperation *setFlag, unsigned endId, unsigned &i) {
  // 只处理 IF_BEGIN
  if (branchElement->getBranchKind() != KindOfBranch::IF_BEGIN) {
    i = branchElement->endId;
    return false;
  }

  bool hasElseBranch = branchElement->branchId < branchElement->endId;

  // 检查 waitFlag (endId) 是否在分支内部。如果是，我们不能简单跳过分支。
  // 这里逻辑是：如果当前的冗余检查范围跨越了整个分支（即 begin 在 if 前，end 在 if 后），
  // 那么我们需要检查是否在 THEN 和 ELSE 两个路径上都找到了内部同步。
  bool endIsInsideThenBranch =
      (!hasElseBranch && endId < branchElement->endId) ||
      (hasElseBranch && endId < branchElement->branchId);
  if (endIsInsideThenBranch)
    return false;

  bool endIsInsideElseBranch = hasElseBranch &&
                               endId >= branchElement->branchId &&
                               endId < branchElement->endId;
  if (endIsInsideElseBranch) {
    i = branchElement->branchId;
    return false;
  }
 
  // 核心：如果两个分支都存在内部覆盖，则整体覆盖
  if (hasElseBranch) {
    bool coveredInThen = CheckRepeatSync(
        branchElement->beginId, branchElement->branchId, syncFinder, setFlag);
    bool coveredInElse = CheckRepeatSync(
        branchElement->branchId, branchElement->endId, syncFinder, setFlag);

    if (coveredInThen && coveredInElse)
      return true;
  }
  // 如果只有 Then 分支 (Implicit Else)，除非我们在 Else (空路径) 上也能找到同步（不可能），
  // 否则无法断定冗余。所以单 If 分支通常无法帮助消除跨越它的外部同步。

  i = branchElement->endId; // 跳过整个分支块
  return false;
}

bool RemoveRedundantSync::CheckLoopBetween(LoopInstanceElement *loopElement,
                                           SyncOperation *setFlag,
                                           unsigned &i) {
  // 对于循环，保守起见暂时不深入检查内部是否覆盖外部。
  // 因为循环可能执行 0 次，如果循环内有同步，但循环不执行，外部依赖就没法满足。
  // 除非通过 Range Analysis 证明循环至少执行一次，否则这里返回 false 是安全的。
  i = loopElement->endId;
  return false;
}

bool RemoveRedundantSync::CanMatchedSync(SmallVector<bool> &syncFinder,
                                         SyncOperation *relatedSync,
                                         SyncOperation *setFlag) {
  // 1. 过滤不相关的同步
  // - 类型必须匹配 (Wait/Set)
  // - 不能是自己 (Index 相同)
  // - Pipe 必须完全一致 (Src->Dst)
  // - EventIdNum: 内部的同步能力必须强于外部 (related.eventIdNum >= set.eventIdNum ???) 
  //   这里暂时假设 Single Buffer (eventIdNum=1) 场景即可覆盖主流程
  
  bool isWait = (relatedSync->GetType() == SyncOperation::TYPE::WAIT_EVENT);
  bool isSet = (relatedSync->GetType() == SyncOperation::TYPE::SET_EVENT);
  
  // 支持 BlockSync 模式的检查
  if (syncAnalysisMode_ == SyncAnalysisMode::BLOCKSYNC) {
      isWait |= (relatedSync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT);
      isSet |= (relatedSync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET);
  }
 
  if (!isWait && !isSet) return false;
  if (relatedSync->GetSyncIndex() == setFlag->GetSyncIndex()) return false;
  if (relatedSync->eventIdNum != setFlag->eventIdNum) return false;
  if (relatedSync->GetForEndIndex() != setFlag->GetForEndIndex()) return false;
  if (relatedSync->isCompensation || setFlag->isCompensation) return false;
  if (!hasSameDepRoots(relatedSync, setFlag)) return false;
  
  // Pipe 检查：内部同步必须也是解决同样的 Src -> Dst 依赖
  if (relatedSync->GetSrcPipe() != setFlag->GetSrcPipe()) return false;
  if (relatedSync->GetDstPipe() != setFlag->GetDstPipe()) return false;
 
  // 2. 状态机逻辑
  // 如果遇到了 Set，记录下来
  if (isSet) {
    checkCondition(relatedSync->GetSyncIndex() < syncFinder.size(), "Index OOB");
    syncFinder[relatedSync->GetSyncIndex()] = true;
  }
  
  // 如果遇到了 Wait，且对应的 Set 之前已经遇到过 (syncFinder=true)
  // 说明我们在 [Begin, End] 范围内找到了一对完整的、同 Pipe 的同步 (Inner Pair)。
  // 这对 Inner Pair 保证了顺序，因此外部的 Pair 是冗余的。
  if (isWait) {
    checkCondition(relatedSync->GetSyncIndex() < syncFinder.size(), "Index OOB");
    if (syncFinder[relatedSync->GetSyncIndex()]) {
      return true; // Found redundant!
    }
  }
 
  return false;
}
