// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSyncUtils.h - Shared sync mapping helpers ------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOSYNCUTILS_H_
#define MLIR_DIALECT_PTO_IR_PTOSYNCUTILS_H_

#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace pto {

/// Parse a sync endpoint-like attribute used by high-level sync operations.
/// Supported attribute kinds are:
///   - pto.pipe_event_type<...>
///   - pto.sync_op_type<...>
FailureOr<SyncOpType> parseSyncOpTypeLikeAttr(Attribute attr);

/// Map high-level sync operation type to concrete hardware PIPE.
PIPE mapSyncOpTypeToPipe(SyncOpType opType);

/// True if the pipe is a concrete endpoint pipe (not PIPE_ALL/UNASSIGNED).
bool isConcreteSyncPipe(PIPE pipe);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTOSYNCUTILS_H_
