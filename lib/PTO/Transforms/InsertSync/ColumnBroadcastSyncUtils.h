// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_COLUMN_BROADCAST_SYNC_UTILS_H
#define PTO_TRANSFORMS_INSERTSYNC_COLUMN_BROADCAST_SYNC_UTILS_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::pto::sync_detail {

inline Value getColumnBroadcastInput(Operation* op)
{
    return llvm::TypeSwitch<Operation*, Value>(op)
        .Case<TColExpandOp>([](auto colExpand) { return colExpand.getSrc(); })
        .Case<TColExpandAddOp, TColExpandMulOp, TColExpandDivOp, TColExpandSubOp, TColExpandExpdifOp,
              TColExpandMaxOp, TColExpandMinOp>([](auto colExpand) { return colExpand.getSrc1(); })
        .Default([](Operation*) { return Value{}; });
}

inline void collectColumnBroadcastReads(
    const CompoundInstanceElement* reader, SmallVectorImpl<const BaseMemInfo*>& reads)
{
    if (!reader->elementOp) {
        return;
    }
    auto collect = [&reads, reader](Operation* op) {
        Value broadcast = getColumnBroadcastInput(op);
        if (!broadcast) {
            return;
        }
        for (const BaseMemInfo* info : reader->useVec) {
            if (info->baseBuffer == broadcast) {
                reads.push_back(info);
            }
        }
    };
    if (isa<FusionRegionOp>(reader->elementOp)) {
        // Aggregation retains member access ranges but uses the region as its
        // placement anchor. Recover only the broadcast operands from its body.
        reader->elementOp->walk(collect);
    } else {
        collect(reader->elementOp);
    }
}

inline bool overwritesColumnBroadcastInput(
    const CompoundInstanceElement* writer, const CompoundInstanceElement* reader, MemoryDependentAnalyzer& memAnalyzer)
{
    SmallVector<const BaseMemInfo*> broadcastReads;
    collectColumnBroadcastReads(reader, broadcastReads);
    DepBaseMemInfoPairVec dependencies;
    return memAnalyzer.DepBetween(writer->defVec, broadcastReads, dependencies);
}

} // namespace mlir::pto::sync_detail

#endif
