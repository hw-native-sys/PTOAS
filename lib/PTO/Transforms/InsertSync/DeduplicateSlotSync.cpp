// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/DeduplicateSlotSync.h"

#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/Support/CodeConstants.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

struct Attachment {
    unsigned count{0};
    unsigned index{0};
    bool before{false};
};

struct Family {
    SyncOperation* set;
    SyncOperation* wait;
    Value root;
};

bool hasUnprovedSlotRotation(const SyncOperations::value_type& pair)
{
    bool hasInvalidPairSize = pair.size() != mlir::pto::kValue2;
    if (hasInvalidPairSize) {
        return false;
    }
    const auto& set = *pair[0];
    const auto& wait = *pair[1];
    if (set.uselessSync || wait.uselessSync || set.GetType() != SyncOperation::TYPE::SET_EVENT ||
        wait.GetType() != SyncOperation::TYPE::WAIT_EVENT || set.GetMechanism() != SyncOperation::MECHANISM::EVENT ||
        wait.GetMechanism() != SyncOperation::MECHANISM::EVENT || !set.GetForEndIndex() || !isSlotKeyedSync(&set) ||
        !isSlotKeyedSync(&wait))
        return false;
    return set.slotSSAExpr != wait.slotSSAExpr && (!set.slotBoundaryMasks || !wait.slotBoundaryMasks);
}

bool sameMasks(const SyncOperation& lhs, const SyncOperation& rhs)
{
    if (lhs.slotBoundaryMasks.has_value() != rhs.slotBoundaryMasks.has_value())
        return false;
    return !lhs.slotBoundaryMasks || (lhs.slotBoundaryMasks->prime == rhs.slotBoundaryMasks->prime &&
                                      lhs.slotBoundaryMasks->drain == rhs.slotBoundaryMasks->drain);
}

bool isUnallocatedSlotEvent(const SyncOperation& sync)
{
    return !sync.uselessSync && !sync.isCompensation && sync.compensationOf == -1 && sync.eventIds.empty() &&
           sync.GetMechanism() == SyncOperation::MECHANISM::EVENT && isSlotKeyedSync(&sync) &&
           sync.slotCount == unsigned(sync.eventIdNum) && sync.slotCount <= kMaxMultiBufferCount &&
           !sync.reallocatedLoopHeadTailSync && sync.reuseCntForWiden == 0 && !sync.lowestCommonAncestorBuffer &&
           !sync.IsAutoSyncTailBarrier() && !sync.block_sync_event_value;
}

bool sameConfiguration(const SyncOperation& lhs, const SyncOperation& rhs)
{
    return lhs.GetMechanism() == rhs.GetMechanism() && lhs.GetSrcPipe() == rhs.GetSrcPipe() &&
           lhs.GetDstPipe() == rhs.GetDstPipe() && lhs.syncCoreType == rhs.syncCoreType &&
           lhs.GetForEndIndex() == rhs.GetForEndIndex() && lhs.eventIdNum == rhs.eventIdNum &&
           lhs.slotCount == rhs.slotCount && sameMasks(lhs, rhs);
}

// Matching node indices alone is insufficient when a node's deque contains
// SET A; WAIT C; SET B. Deleting either A or B could change ordering through C.
// Restrict this initial exact dedup to SET-only and WAIT-only event frontiers.
bool hasUniformFrontier(const SyncOps& list, SyncOperation::TYPE type)
{
    return llvm::all_of(list, [type](const SyncOperation* sync) {
        return sync->uselessSync ||
               (sync->GetType() == type && sync->GetMechanism() == SyncOperation::MECHANISM::EVENT &&
                !sync->isCompensation && sync->compensationOf == -1);
    });
}

bool hasUniqueAttachment(
    const SyncOperation& sync, bool before, const llvm::DenseMap<const SyncOperation*, Attachment>& attachments)
{
    auto found = attachments.find(&sync);
    return found != attachments.end() && found->second.count == 1 && found->second.index == sync.GetSyncIRIndex() &&
           found->second.before == before;
}

std::optional<Value> getClosedRoot(const SyncOperation& set, const SyncOperation& wait)
{
    if (set.depRootBuffers.empty() || wait.depRootBuffers.empty() || set.depMemInfos.empty() ||
        wait.depMemInfos.empty())
        return std::nullopt;
    Value root = set.depRootBuffers.front();
    if (!root)
        return std::nullopt;
    for (const auto* sync : {&set, &wait}) {
        if (!llvm::all_of(sync->depRootBuffers, [root](Value other) { return other == root; }) ||
            !llvm::all_of(
                sync->depMemInfos, [root](const BaseMemInfo* info) { return info && info->rootBuffer == root; }))
            return std::nullopt;
    }
    return root;
}

bool hasEligibleEndpoints(
    const SyncOperation& set, const SyncOperation& wait,
    const llvm::DenseMap<const SyncOperation*, Attachment>& attachments)
{
    return set.GetType() == SyncOperation::TYPE::SET_EVENT && wait.GetType() == SyncOperation::TYPE::WAIT_EVENT &&
           isUnallocatedSlotEvent(set) && isUnallocatedSlotEvent(wait) && sameConfiguration(set, wait) &&
           set.GetForEndIndex() && hasUniqueAttachment(set, false, attachments) &&
           hasUniqueAttachment(wait, true, attachments);
}

// Called only after endpoint validation establishes a shared loop identity.
bool hasValidLoopLifetime(const SyncOperation& set, const SyncOperation& wait, const SyncIRs& syncIR)
{
    unsigned end = *set.GetForEndIndex();
    if (end >= syncIR.size())
        return false;
    auto* loopEnd = dyn_cast<LoopInstanceElement>(syncIR[end].get());
    if (!loopEnd || loopEnd->getLoopKind() != KindOfLoop::LOOP_END || loopEnd->endId != end || loopEnd->beginId >= end)
        return false;
    auto* loopBegin = dyn_cast<LoopInstanceElement>(syncIR[loopEnd->beginId].get());
    return loopBegin && loopBegin->getLoopKind() == KindOfLoop::LOOP_BEGIN && loopBegin->beginId == loopEnd->beginId &&
           loopBegin->endId == end && wait.GetSyncIRIndex() > loopEnd->beginId && set.GetSyncIRIndex() < end &&
           wait.GetSyncIRIndex() < set.GetSyncIRIndex();
}

std::optional<Family> getFamily(
    SyncOperations::value_type& pair, const SyncIRs& syncIR,
    const llvm::DenseMap<const SyncOperation*, Attachment>& attachments)
{
    bool hasInvalidPairSize = pair.size() != mlir::pto::kValue2;
    if (hasInvalidPairSize) {
        return std::nullopt;
    }
    auto* set = pair[0].get();
    auto* wait = pair[1].get();
    if (!hasEligibleEndpoints(*set, *wait, attachments) || !hasValidLoopLifetime(*set, *wait, syncIR))
        return std::nullopt;
    if (!hasUniformFrontier(syncIR[set->GetSyncIRIndex()]->pipeAfter, SyncOperation::TYPE::SET_EVENT) ||
        !hasUniformFrontier(syncIR[wait->GetSyncIRIndex()]->pipeBefore, SyncOperation::TYPE::WAIT_EVENT))
        return std::nullopt;
    auto root = getClosedRoot(*set, *wait);
    if (!root)
        return std::nullopt;
    return Family{set, wait, *root};
}

bool sameProtocol(const Family& lhs, const Family& rhs)
{
    return lhs.root == rhs.root && sameConfiguration(*lhs.set, *rhs.set) &&
           lhs.set->GetSyncIRIndex() == rhs.set->GetSyncIRIndex() &&
           lhs.wait->GetSyncIRIndex() == rhs.wait->GetSyncIRIndex() && lhs.set->slotSSAExpr == rhs.set->slotSSAExpr &&
           lhs.wait->slotSSAExpr == rhs.wait->slotSSAExpr;
}

size_t protocolHash(const Family& family)
{
    const auto& set = *family.set;
    return llvm::hash_combine(
        family.root.getAsOpaquePointer(), *set.GetForEndIndex(), set.GetSyncIRIndex(), family.wait->GetSyncIRIndex(),
        set.slotSSAExpr.getAsOpaquePointer(), family.wait->slotSSAExpr.getAsOpaquePointer(), set.slotCount,
        set.eventIdNum, unsigned(set.GetSrcPipe()), unsigned(set.GetDstPipe()), unsigned(set.syncCoreType),
        set.slotBoundaryMasks.has_value(), set.slotBoundaryMasks ? set.slotBoundaryMasks->prime : 0,
        set.slotBoundaryMasks ? set.slotBoundaryMasks->drain : 0);
}

void mergeDependencies(const Family& keep, const Family& remove)
{
    SmallVector<Value> roots;
    SmallVector<const BaseMemInfo*> infos;
    for (const auto* sync : {keep.set, keep.wait, remove.set, remove.wait}) {
        for (Value root : sync->depRootBuffers)
            if (!llvm::is_contained(roots, root))
                roots.push_back(root);
        for (const auto* info : sync->depMemInfos)
            if (!llvm::is_contained(infos, info))
                infos.push_back(info);
    }
    keep.set->depRootBuffers = roots;
    keep.wait->depRootBuffers = roots;
    keep.set->depMemInfos = infos;
    keep.wait->depMemInfos = infos;
    // Keep the representative's original dependency-source index. It denotes
    // one original source, not all merged readers. No later redundancy proof is
    // run: allocation only clones it into synthetic boundary records.
}

} // namespace

unsigned mlir::pto::deduplicateSlotEventFamilies(SyncIRs& syncIR, SyncOperations& operations)
{
    // Unmasked legacy boundaries prime every lane. With different SET/WAIT
    // selectors, an entrance token may permit an overwrite before its reader
    // finishes. Dedup of ANY family can free IDs for such an unrelated pair
    // and remove its safe barrier fallback. Scan all pairs before any mutation,
    // including pairs in other roots/loops that are not dedup candidates.
    if (llvm::any_of(operations, hasUnprovedSlotRotation))
        return 0;

    llvm::DenseMap<const SyncOperation*, Attachment> attachments;
    for (const auto& node : syncIR) {
        for (bool before : {true, false}) {
            for (const auto* sync : before ? node->pipeBefore : node->pipeAfter) {
                auto& attachment = attachments[sync];
                ++attachment.count;
                attachment.index = node->GetIndex();
                attachment.before = before;
            }
        }
    }

    llvm::DenseMap<size_t, SmallVector<Family>> protocols;
    unsigned removed = 0;
    for (auto& pair : operations) {
        auto family = getFamily(pair, syncIR, attachments);
        if (!family)
            continue;
        auto& bucket = protocols[protocolHash(*family)];
        auto match = llvm::find_if(bucket, [&](const Family& other) { return sameProtocol(other, *family); });
        if (match == bucket.end()) {
            bucket.push_back(*family);
            continue;
        }
        // Corresponding endpoints have identical execution instances, per-side
        // lane selectors and boundary policy. Their per-lane token traces are
        // copies, so one whole channel preserves both dependency sets. In
        // particular, root or modulo equality ALONE is never coverage evidence.
        mergeDependencies(*match, *family);
        InstanceElement::RemoveSync(syncIR[family->set->GetSyncIRIndex()]->pipeAfter, family->set);
        InstanceElement::RemoveSync(syncIR[family->wait->GetSyncIRIndex()]->pipeBefore, family->wait);
        family->set->uselessSync = true;
        family->wait->uselessSync = true;
        ++removed;
    }
    return removed;
}
