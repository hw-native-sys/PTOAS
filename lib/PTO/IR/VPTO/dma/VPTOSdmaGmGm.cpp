// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOSdmaGmGm.cpp - pto.sdma_gm_gm methods --------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Support/AsyncSessionABI.h"
#include "VPTODmaInternal.h"

using namespace mlir;
using namespace mlir::pto;

// A session operand needs no check here. Its engine is carried by
// !pto.dma_session, so the ODS operand constraint already rejects another
// engine's session, and the physical field layout is built by the wrapper
// expansion out of PTO/Support/AsyncSessionABI.h rather than written by a
// caller. Restating that layout as a verifier would only describe how the
// expansion constructs it.

// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated verifier callback has a non-const signature.
LogicalResult SdmaGmGmOp::verify()
{
    if (failed(verifyAsyncTransferPtr(getOperation(), getDestination().getType(), "destination")))
        return failure();
    if (failed(verifyAsyncTransferPtr(getOperation(), getSource().getType(), "source")))
        return failure();
    if (failed(verifyAsyncTransferPtr(getOperation(), getCqRecord().getType(), "completion record")))
        return failure();

    // A waiter reads the record as 64-bit words, so the pointer has to say so.
    auto recordTy = cast<PtrType>(getCqRecord().getType());
    auto recordElem = dyn_cast<IntegerType>(recordTy.getElementType());
    if (!recordElem || recordElem.getWidth() != mlir::pto::kValue64) {
        return emitOpError() << "completion record must point at i64, but points at " << recordTy.getElementType();
    }

    // The expansion posts to the queue with scalar GM stores and, on A5, rings the
    // doorbell with pto.st_dev. None of that is legal under SIMT, so reject here
    // rather than letting the expansion fail later with a less obvious diagnostic.
    if (isInsideSimtExecutionScope(getOperation())) {
        return emitOpError() << "must be outside pto.simt_entry functions and pto.section.simt";
    }
    auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
    if (!funcOp || !pto::isPTOEntryFunction(funcOp))
        return emitOpError() << "requires an enclosing ordinary AICore entry function";

    if (auto blockBytes = getBlockBytes()) {
        if (*blockBytes == 0)
            return emitOpError() << "block_bytes must be positive";
        if (*blockBytes % comm::workspace::kMinTransferBytes != 0)
            return emitOpError() << "block_bytes must be a multiple of " << comm::workspace::kMinTransferBytes;
    }

    // Only the first kMaxChannels descriptors are filled in; a group past that
    // reads whatever the allocation happened to contain.
    if (auto channelIdx = getChannelIdx()) {
        if (*channelIdx >= comm::workspace::kMaxChannels)
            return emitOpError() << "channel_idx must be less than " << comm::workspace::kMaxChannels;
    }
    return success();
}

// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated side-effect callback has a non-const signature.
void SdmaGmGmOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
    effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
    // The queue tail in the channel descriptor is read and advanced, so two posts
    // on one session must not be reordered or dropped.
    effects.emplace_back(MemoryEffects::Read::get(), &getSessionMutable());
    effects.emplace_back(MemoryEffects::Write::get(), &getSessionMutable());
}
