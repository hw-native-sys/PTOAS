// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncOracleExtract.h - Shared sync-op extractor (oracle) ---*- C++ -*-===//
//
// The shared walk-surface the oracle gates ride on: the emitted `pto.set_flag` /
// `pto.wait_flag` / `pto.get_buf` / `pto.rls_buf` / `pto.barrier` ops of one
// function. G3 reads the ids, G2 reads their live ranges.
//
// `SyncOpRecord` describes the same synchronization before codegen, as an allocator
// holds it. Only G3 has an overload taking it, and only the gate self-test feeds it,
// because nothing else in this build produces pre-codegen records.
//
// Extraction is deterministic (program order), so a rendering is stable across runs.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCORACLEEXTRACT_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCORACLEEXTRACT_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace pto {
namespace oracle {

/// One emitted synchronization op, read out of the IR after codegen.
/// `eventId` is set only for set_flag/wait_flag; `bufId` only for
/// get_buf/rls_buf; both are -1 when not applicable.
struct IRSyncRecord {
  std::string opName;
  std::string srcPipe;
  std::string dstPipe;
  /// Typed pipes for the gates. The strings above are for rendering only;
  /// gates must never string-match a pipe name.
  PipelineType srcPipeType = PipelineType::PIPE_UNASSIGNED;
  PipelineType dstPipeType = PipelineType::PIPE_UNASSIGNED;
  int64_t eventId = -1;
  int64_t bufId = -1;
  /// get_buf/rls_buf `mode`. 0 = the bidirectional token (the only mode PTOAS
  /// emits today). Non-zero = the directional mode (set_flagV2/wait_flagV2),
  /// deferred -- see the buffer-ID legality notes in SyncOracleGates.h, which
  /// applies the strict pairing rules only to mode 0.
  int64_t mode = 0;
  /// Enclosing block. A get_buf and its rls_buf must sit in the SAME block, or
  /// the pair may not both execute (counter desync). Null for non-buf ops.
  mlir::Block *block = nullptr;
  /// The op itself, for gates that must ask where it sits in the control flow. An
  /// event pair is checked on its enclosing conditionals, which a block pointer
  /// alone cannot answer: two blocks differ for loop nesting too, and that case is
  /// legitimate.
  mlir::Operation *op = nullptr;
  unsigned order = 0; // program order within the function
};

/// One entry of the in-memory SyncOperation set, before codegen. `eventIds` is
/// empty until an allocator assigns them.
struct SyncOpRecord {
  std::string type;
  /// `static_cast<int>(SyncOperation::TYPE)`; -1 when unset. Gates switch on
  /// this, never on the `type` string.
  int typeCode = -1;
  /// `SyncOperation::MechanismName` of the resource class the allocator routed this
  /// sync to -- and so whether it is SyncIR-positioned (event/barrier) or op-anchored
  /// (bufid). Empty until the pre-codegen extractor fills it.
  ///
  /// RENDERING ONLY. Unlike `type`, this has no numeric companion, so nothing can
  /// switch on it without string-matching a name. A gate that needs the mechanism as
  /// a decision should get a `mechanismCode` beside this, the way `typeCode` sits
  /// beside `type`, rather than compare these strings.
  std::string mechanism;
  int srcPipe = -1;
  int dstPipe = -1;
  llvm::SmallVector<int, 2> eventIds;
  unsigned syncIndex = 0;
  unsigned irIndex = 0;
  /// Mirrors `SyncOperation::isCompensation` / `compensationOf`. G2's syncops scan
  /// needs both: a compensation record shares its ids with the in-body pair it primes
  /// BY DESIGN, so that pairing must not read as two hazards colliding -- while a
  /// collision with any OTHER hazard still must.
  bool isCompensation = false;
  int compensationOf = -1;
};

/// Human-readable name of an InsertSync PipelineType ("MTE2", "V", "ALL", ...).
llvm::StringRef pipelineTypeName(PipelineType pipe);

/// The pipe a `get_buf`/`rls_buf` `op_type` names.
///
/// `PTO_PipeLikeAttr` admits three spellings -- pipe_event_type, sync_op_type, and a
/// plain PipeAttr. A decoder handling only the first two returns PIPE_UNASSIGNED for
/// the third rather than failing, so its caller silently loses the op.
pto::PIPE bufSyncPipe(mlir::Attribute opTypeAttr);

/// Render a report and write it to stderr atomically.
///
/// STDERR, NEVER STDOUT: stdout carries the emitted kernel source, so a report
/// written there is spliced into the C++ and it stops compiling.
///
/// MLIR runs nested func::FuncOp passes in PARALLEL and ptoas does not disable
/// threading, so unsynchronized writes from several functions interleave mid-line.
/// Every oracle pass must print through here. Ordering *between* functions stays
/// nondeterministic, so no consumer may depend on it.
void emitReport(llvm::function_ref<void(llvm::raw_ostream &)> body);

/// Extract the emitted sync ops from `func`, in program order.
llvm::SmallVector<IRSyncRecord> extractFromIR(func::FuncOp func);

/// The same synchronization before codegen, as an allocator holds it: the ids it has
/// just written, before anything is emitted.
llvm::SmallVector<SyncOpRecord> extractFromSyncOps(const SyncOperations &ops);


/// Deterministic, diff-friendly rendering (one record per line).
void printIRRecords(llvm::raw_ostream &os, llvm::StringRef funcName,
                    llvm::ArrayRef<IRSyncRecord> records);
void printSyncOpRecords(llvm::raw_ostream &os, llvm::StringRef funcName,
                        llvm::ArrayRef<SyncOpRecord> records);

} // namespace oracle
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCORACLEEXTRACT_H
