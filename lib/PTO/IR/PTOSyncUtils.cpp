// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSyncUtils.cpp - Shared sync mapping helpers --------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOSyncUtils.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<SyncOpType> mlir::pto::parseSyncOpTypeLikeAttr(Attribute attr) {
  if (auto a = dyn_cast_or_null<PipeEventTypeAttr>(attr))
    return a.getOpType();
  if (auto a = dyn_cast_or_null<SyncOpTypeAttr>(attr))
    return a.getOpType();
  return failure();
}

PIPE mlir::pto::mapSyncOpTypeToPipe(SyncOpType opType) {
  switch (opType) {
  case SyncOpType::TLOAD:
    return PIPE::PIPE_MTE2;
  case SyncOpType::TSTORE_VEC:
    return PIPE::PIPE_MTE3;
  case SyncOpType::TSTORE_ACC:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2L:
  case SyncOpType::TMOV_M2B:
    return PIPE::PIPE_MTE1;
  case SyncOpType::TMOV_M2S:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2V:
    return PIPE::PIPE_V;
  case SyncOpType::TMOV_V2M:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMATMUL:
    return PIPE::PIPE_M;
  case SyncOpType::TVEC:
  case SyncOpType::TVECWAIT_EVENT:
    return PIPE::PIPE_V;
  default:
    return PIPE::PIPE_UNASSIGNED;
  }
}

bool mlir::pto::isConcreteSyncPipe(PIPE pipe) {
  return pipe != PIPE::PIPE_UNASSIGNED && pipe != PIPE::PIPE_ALL;
}
