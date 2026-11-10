// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <optional>
#include <utility>

namespace mlir {
namespace pto {

class MGatherOp;

struct SyncMacroPhase {
  unsigned phaseId{0};
  PipelineType pipe{PipelineType::PIPE_UNASSIGNED};
  SmallVector<Value> defValues;
  SmallVector<Value> useValues;
};

struct SyncMacroHiddenEvent {
  PipelineType srcPipe{PipelineType::PIPE_UNASSIGNED};
  PipelineType dstPipe{PipelineType::PIPE_UNASSIGNED};
  SmallVector<unsigned> eventIds;
};

struct SyncMacroModel {
  SmallVector<SyncMacroPhase> phases;
  SmallVector<SyncMacroHiddenEvent> hiddenEvents;

  explicit operator bool() const { return !phases.empty(); }
};

std::optional<SyncMacroModel> getSyncMacroModel(Operation *op);

/// Whether pto.mgather is eligible for the pto-isa A2/A3 async row-gather
/// overload (MGATHER<..., GatherSync::Async>): GM -> UB, Coalesce::Row, and
/// the GatherOOB::Undefined valid-index contract. Pure op-level predicate;
/// async additionally requires kMGatherAsyncRowAttrName (see below). The async
/// macro keeps only its leading index-readiness waits and internal S -> MTE2
/// handshake; dst completion is left to compiler-emitted sync at the actual
/// consumers (issue #1674 / pto-isa #380).
bool useAsyncA2A3RowMGather(MGatherOp op);

/// Mark every eligible pto.mgather in `func` with kMGatherAsyncRowAttrName.
/// Called by the automatic synchronization passes once they have decided to
/// process the function, so downstream (macro model and EmitC) can lower
/// exactly those ops to the async overload.
void markAsyncA2A3RowMGathers(func::FuncOp func);

/// Whether `op` carries kMGatherAsyncRowAttrName and is still eligible for
/// the async overload (the mark is re-verified against the op's arch, mode,
/// OOB, and destination space). True means an automatic sync pass took
/// responsibility for this gather's output completion.
bool hasAsyncA2A3RowMGatherMark(Operation *op);

/// Padded first-to-last phase span of a macro call in SyncIR index space.
std::optional<std::pair<unsigned, unsigned>> getSyncMacroSpan(const SyncIRs &syncIR, std::size_t firstIndex);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H
