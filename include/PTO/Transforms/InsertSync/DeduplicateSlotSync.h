// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_DEDUPLICATESLOTSYNC_H
#define PTO_TRANSFORMS_INSERTSYNC_DEDUPLICATESLOTSYNC_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"

namespace mlir::pto {

/// Delete duplicate, unallocated slot event channels without moving their
/// endpoints or changing their boundary-token policy. Run after sync motion and
/// redundancy removal, before event allocation. Returns the number of removed
/// whole event families. This does not establish new access coverage.
/// Skip the whole function if a live loop-carried slot EVENT pair has different
/// SET/WAIT selectors without boundary masks on both endpoints: freeing IDs for
/// any family could replace that pair's safe fallback with unsafe priming.
unsigned deduplicateSlotEventFamilies(SyncIRs& syncIR, SyncOperations& operations);

} // namespace mlir::pto

#endif
