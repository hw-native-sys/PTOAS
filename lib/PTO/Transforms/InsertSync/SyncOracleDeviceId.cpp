// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/SyncOracleGates.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"

#include "llvm/ADT/DenseMap.h"
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::oracle;

llvm::StringRef mlir::pto::oracle::idViolationKindName(IdViolationKind kind)
{
    switch (kind) {
        case IdViolationKind::EventIdOutOfRange:
            return "event-id-out-of-range";
        case IdViolationKind::EventIdReservedForDirection:
            return "event-id-reserved-for-direction";
        case IdViolationKind::EventIdReservedForBlockAll:
            return "event-id-reserved-for-block-all";
        case IdViolationKind::BlockAllEventIdWrong:
            return "block-all-event-id-wrong";
        case IdViolationKind::BarrierCarriesEventId:
            return "barrier-carries-event-id";
        case IdViolationKind::BufIdOutOfRange:
            return "buf-id-out-of-range";
        case IdViolationKind::BufIdExceedsArchCapacity:
            return "buf-id-exceeds-arch-capacity";
        case IdViolationKind::EventIdCollidesWithMacroHidden:
            return "event-id-collides-with-macro-hidden";
        case IdViolationKind::EventSetWaitUnbalanced:
            return "event-set-wait-unbalanced";
    }
    return "?";
}

namespace {

std::string pipeDir(PipelineType src, PipelineType dst)
{
    return (oracle::pipelineTypeName(src) + "->" + oracle::pipelineTypeName(dst)).str();
}

/// R1 + R2: an intra-core set/wait event id.
void checkIntraCoreEventId(
    int64_t id, PipelineType src, PipelineType dst, unsigned poolSize, unsigned order, llvm::StringRef opName,
    llvm::SmallVectorImpl<IdViolation>& out)
{
    if (id < 0 || id >= static_cast<int64_t>(poolSize)) {
        out.push_back(
            {IdViolationKind::EventIdOutOfRange, order, opName.str(), id,
             ("intra-core event pool for " + pipeDir(src, dst) + " is [0, " + std::to_string(poolSize) + ")")});
        return;
    }
    uint64_t reserved = SyncEventIdAllocation::GetReservedEventIdNum(src, dst);
    if (reserved > 0 && id >= static_cast<int64_t>(poolSize - reserved)) {
        out.push_back(
            {IdViolationKind::EventIdReservedForDirection, order, opName.str(), id,
             (pipeDir(src, dst) + " reserves its top " + std::to_string(reserved) + " id(s); available is [0, " +
              std::to_string(poolSize - reserved) + ")")});
    }
}

/// R4 + R5: a buffer id.
void checkBufId(
    int64_t id, unsigned order, llvm::StringRef opName, const DeviceIdLimits& limits,
    llvm::SmallVectorImpl<IdViolation>& out)
{
    if (id < 0 || id >= static_cast<int64_t>(limits.bufIdHwMax)) {
        out.push_back(
            {IdViolationKind::BufIdOutOfRange, order, opName.str(), id,
             ("hardware buf_id field is [0, " + std::to_string(limits.bufIdHwMax) + ")")});
        return;
    }
    if (id >= static_cast<int64_t>(limits.bufIdCapacity)) {
        out.push_back(
            {IdViolationKind::BufIdExceedsArchCapacity, order, opName.str(), id,
             ("this compilation exposes K=" + std::to_string(limits.bufIdCapacity) +
              " buffer id(s); buffer-id sync needs K > 0 (A5)")});
    }
}

} // namespace

llvm::SmallVector<IdViolation> mlir::pto::oracle::checkDeviceIdLegality(
    llvm::ArrayRef<IRSyncRecord> records, const DeviceIdLimits& limits)
{
    llvm::SmallVector<IdViolation> violations;

    for (const IRSyncRecord& rec : records) {
        if (rec.opName == "pto.set_flag" || rec.opName == "pto.wait_flag") {
            checkIntraCoreEventId(
                rec.eventId, rec.srcPipeType, rec.dstPipeType, limits.intraCoreEventIdNum, rec.order, rec.opName,
                violations);
        } else if (rec.opName == "pto.get_buf" || rec.opName == "pto.rls_buf") {
            checkBufId(rec.bufId, rec.order, rec.opName, limits, violations);
        } else if (rec.opName == "pto.barrier" && rec.eventId >= 0) {
            violations.push_back(
                {IdViolationKind::BarrierCarriesEventId, rec.order, rec.opName, rec.eventId,
                 "a pipe barrier takes no event id"});
        }
    }

    // Every (direction, id) must carry as many sets as waits. A shortfall either way
    // is a hang: a wait with no set never retires, and a set with no wait leaves the
    // flag raised for whoever takes the id next.
    //
    // Rotating pairs are excluded, not overlooked: `set_flag_dyn`/`wait_flag_dyn`
    // choose their id at runtime and it is recorded as -1, so a rotating set cannot be
    // paired with its wait. Nothing here checks a rotating id.
    llvm::DenseMap<std::tuple<int, int, int64_t>, std::pair<unsigned, unsigned>> tally;
    llvm::DenseMap<std::tuple<int, int, int64_t>, unsigned> firstOrder;
    for (const IRSyncRecord& rec : records) {
        bool isSet = rec.opName == "pto.set_flag";
        bool isWait = rec.opName == "pto.wait_flag";
        if ((!isSet && !isWait) || rec.eventId < 0) {
            continue;
        }
        auto key = std::make_tuple(static_cast<int>(rec.srcPipeType), static_cast<int>(rec.dstPipeType), rec.eventId);
        auto& slot = tally[key];
        (isSet ? slot.first : slot.second)++;
        if (firstOrder.count(key) == 0) {
            firstOrder[key] = rec.order;
        }
    }
    llvm::SmallVector<std::tuple<int, int, int64_t>> keys;
    for (const auto& kv : tally) {
        keys.push_back(kv.first);
    }
    llvm::sort(keys);
    for (const auto& key : keys) {
        auto [sets, waits] = tally[key];
        if (sets == waits) {
            continue;
        }
        violations.push_back(
            {IdViolationKind::EventSetWaitUnbalanced, firstOrder[key], "pto.set_flag",
             static_cast<int>(std::get<2>(key)),
             ("direction " + std::to_string(std::get<0>(key)) + "->" + std::to_string(std::get<1>(key)) + " carries " +
              std::to_string(sets) + " set_flag and " + std::to_string(waits) + " wait_flag")});
    }

    return violations;
}

namespace {
static void checkBlockEventIds(
    const SyncOpRecord& rec, bool blockAllPresent, unsigned blockPool, llvm::SmallVectorImpl<IdViolation>& violations)
{
    for (int id : rec.eventIds) {
        if (blockAllPresent &&
            (id == static_cast<int>(kBlockSyncAllCubeEventId) || id == static_cast<int>(kBlockSyncAllVectorEventId))) {
            violations.push_back(
                {IdViolationKind::EventIdReservedForBlockAll, rec.syncIndex, rec.type, id,
                 ("ids " + std::to_string(kBlockSyncAllCubeEventId) + "/" + std::to_string(kBlockSyncAllVectorEventId) +
                  " belong to sync_block_all while one is live")});
            continue;
        }
        if (id < 0 || id >= static_cast<int64_t>(blockPool)) {
            violations.push_back(
                {IdViolationKind::EventIdOutOfRange, rec.syncIndex, rec.type, id,
                 ("block-sync pool is [0, " + std::to_string(blockPool) + ")")});
        }
    }
}

} // namespace

llvm::SmallVector<IdViolation> mlir::pto::oracle::checkDeviceIdLegality(
    llvm::ArrayRef<SyncOpRecord> records, const DeviceIdLimits& limits)
{
    using TYPE = SyncOperation::TYPE;
    llvm::SmallVector<IdViolation> violations;

    // The block-sync pool only shrinks when a SYNC_BLOCK_ALL is actually present;
    // mirrors SyncEventIdAllocation::reserveBlockAllEventIds().
    bool blockAllPresent = llvm::any_of(
        records, [](const SyncOpRecord& rec) { return rec.typeCode == static_cast<int>(TYPE::SYNC_BLOCK_ALL); });
    const unsigned blockPool = limits.blockSyncEventIdNum - (blockAllPresent ? kReservedBlockSyncEventIdNum : 0u);

    for (const SyncOpRecord& rec : records) {
        auto src = static_cast<PipelineType>(rec.srcPipe);
        auto dst = static_cast<PipelineType>(rec.dstPipe);

        switch (static_cast<TYPE>(rec.typeCode)) {
            case TYPE::SET_EVENT:
            case TYPE::WAIT_EVENT:
                // EVERY id a hazard holds is checked, not just the first. A rotating hazard
                // carries several, and its emitted form (`set_flag_dyn`/`wait_flag_dyn`)
                // resolves the id at runtime, so once emitted the id cannot be checked at
                // all -- these pre-codegen records are the only place the full set is visible.
                // Weakening or narrowing this loop --
                // e.g. back to `eventIds[0]` -- would not fail any test: it would make that
                // blind spot LIVE SILENTLY, because no other view can catch an illegal id
                // on a rotating pair.
                //
                // Iterating the FULL set is the point: nothing else validates these ids, so
                // narrowing it to `eventIds[0]` would silently stop checking the rest.
                for (int id : rec.eventIds) {
                    checkIntraCoreEventId(
                        id, src, dst, limits.intraCoreEventIdNum, rec.syncIndex, rec.type, violations);
                }
                break;

            case TYPE::SYNC_BLOCK_SET:
            case TYPE::SYNC_BLOCK_WAIT:
                checkBlockEventIds(rec, blockAllPresent, blockPool, violations);
                break;

            case TYPE::SYNC_BLOCK_ALL:
                for (int id : rec.eventIds) {
                    const bool hasInvalidBlockAllEventId = id != static_cast<int>(kBlockSyncAllCubeEventId) &&
                                                           id != static_cast<int>(kBlockSyncAllVectorEventId);
                    if (hasInvalidBlockAllEventId) {
                        violations.push_back(
                            {IdViolationKind::BlockAllEventIdWrong, rec.syncIndex, rec.type, id,
                             ("sync_block_all must use " + std::to_string(kBlockSyncAllCubeEventId) + " (cube) or " +
                              std::to_string(kBlockSyncAllVectorEventId) + " (vector)")});
                    }
                }
                break;

            case TYPE::PIPE_BARRIER:
            case TYPE::PIPE_BARRIER_CUBE:
            case TYPE::PIPE_BARRIER_VECTOR:
                for (int id : rec.eventIds) {
                    violations.push_back(
                        {IdViolationKind::BarrierCarriesEventId, rec.syncIndex, rec.type, id,
                         "a pipe barrier takes no event id"});
                }
                break;
        }
    }

    return violations;
}
