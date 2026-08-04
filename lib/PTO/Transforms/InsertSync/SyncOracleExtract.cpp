// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncOracleExtract.cpp - Shared sync-op extractor (oracle) ----------===//
//
// Implements the two extraction views (see SyncOracleExtract.h) that the gates
// read a compilation's emitted sync through.
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/InsertSync/SyncOracleExtract.h"
#define GEN_PASS_DEF_PTODUMPSYNCEXTRACT
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "llvm/Support/Mutex.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#include "PTO/Transforms/Passes.h.inc"

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

void mlir::pto::oracle::emitReport(
    llvm::function_ref<void(llvm::raw_ostream &)> body) {
  std::string buffer;
  {
    llvm::raw_string_ostream os(buffer);
    body(os);
  }
  // stderr, NOT stdout: stdout carries ptoas's product (the emitted CCEC C++).
  // Reports written there are spliced into the kernel source, which then
  // stops compiling.
  static llvm::sys::SmartMutex<true> mutex;
  llvm::sys::SmartScopedLock<true> lock(mutex);
  llvm::errs() << buffer;
  llvm::errs().flush();
}

llvm::StringRef mlir::pto::oracle::pipelineTypeName(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
    return "S";
  case PipelineType::PIPE_V:
    return "V";
  case PipelineType::PIPE_M:
    return "M";
  case PipelineType::PIPE_MTE1:
    return "MTE1";
  case PipelineType::PIPE_MTE2:
    return "MTE2";
  case PipelineType::PIPE_MTE3:
    return "MTE3";
  case PipelineType::PIPE_ALL:
    return "ALL";
  default:
    return "?";
  }
}

/// IR pipe enum -> the InsertSync PipelineType the gates reason about. Written
/// as an explicit switch rather than a numeric cast: the two enums happen to
/// agree today, and nothing enforces that they keep agreeing.
static PipelineType toPipelineType(pto::PIPE pipe) {
  switch (pipe) {
  case pto::PIPE::PIPE_S:
    return PipelineType::PIPE_S;
  case pto::PIPE::PIPE_V:
    return PipelineType::PIPE_V;
  case pto::PIPE::PIPE_M:
    return PipelineType::PIPE_M;
  case pto::PIPE::PIPE_MTE1:
    return PipelineType::PIPE_MTE1;
  case pto::PIPE::PIPE_MTE2:
    return PipelineType::PIPE_MTE2;
  case pto::PIPE::PIPE_MTE3:
    return PipelineType::PIPE_MTE3;
  case pto::PIPE::PIPE_ALL:
    return PipelineType::PIPE_ALL;
  case pto::PIPE::PIPE_MTE4:
    return PipelineType::PIPE_MTE4;
  case pto::PIPE::PIPE_MTE5:
    return PipelineType::PIPE_MTE5;
  case pto::PIPE::PIPE_V2:
    return PipelineType::PIPE_V2;
  case pto::PIPE::PIPE_FIX:
    return PipelineType::PIPE_FIX;
  case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A:
    return PipelineType::VIRTUAL_PIPE_MTE2_L1A;
  case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return PipelineType::VIRTUAL_PIPE_MTE2_L1B;
  case pto::PIPE::PIPE_NUM:
  case pto::PIPE::PIPE_UNASSIGNED:
    return PipelineType::PIPE_UNASSIGNED;
  }
  return PipelineType::PIPE_UNASSIGNED;
}

/// Mirrors `verifyBufSyncOp` (PTO.cpp): the attribute may be a plain PipeAttr, or a
/// pipe_event_type/sync_op_type that must be mapped. Declared in the header because
/// G1's coverage walk decodes the same attribute and must agree with this.
pto::PIPE mlir::pto::oracle::bufSyncPipe(mlir::Attribute opTypeAttr) {
  if (!opTypeAttr)
    return pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<pto::PipeAttr>(opTypeAttr))
    return pipeAttr.getPipe();
  auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
  if (succeeded(opTypeOr))
    return mapSyncOpTypeToPipe(*opTypeOr);
  return pto::PIPE::PIPE_UNASSIGNED;
}

llvm::SmallVector<oracle::IRSyncRecord>
mlir::pto::oracle::extractFromIR(func::FuncOp func) {
  llvm::SmallVector<IRSyncRecord> records;

  func.walk([&](Operation *op) {
    IRSyncRecord rec;

    auto setPipes = [&](pto::PIPE src, pto::PIPE dst) {
      rec.srcPipe = stringifyPIPE(src).str();
      rec.dstPipe = stringifyPIPE(dst).str();
      rec.srcPipeType = toPipelineType(src);
      rec.dstPipeType = toPipelineType(dst);
    };

    if (auto setFlag = dyn_cast<pto::SetFlagOp>(op)) {
      rec.opName = "pto.set_flag";
      setPipes(setFlag.getSrcPipeAttr().getPipe(),
               setFlag.getDstPipeAttr().getPipe());
      rec.eventId = static_cast<int64_t>(setFlag.getEventIdAttr().getEvent());
    } else if (auto waitFlag = dyn_cast<pto::WaitFlagOp>(op)) {
      rec.opName = "pto.wait_flag";
      setPipes(waitFlag.getSrcPipeAttr().getPipe(),
               waitFlag.getDstPipeAttr().getPipe());
      rec.eventId = static_cast<int64_t>(waitFlag.getEventIdAttr().getEvent());
    } else if (auto getBuf = dyn_cast<pto::GetBufOp>(op)) {
      rec.opName = "pto.get_buf";
      rec.bufId = static_cast<int64_t>(getBuf.getBufId());
      rec.mode = static_cast<int64_t>(getBuf.getMode());
      rec.block = op->getBlock();
      pto::PIPE pipe = bufSyncPipe(getBuf.getOpTypeAttr());
      setPipes(pipe, pipe);
    } else if (auto rlsBuf = dyn_cast<pto::RlsBufOp>(op)) {
      rec.opName = "pto.rls_buf";
      rec.bufId = static_cast<int64_t>(rlsBuf.getBufId());
      rec.mode = static_cast<int64_t>(rlsBuf.getMode());
      rec.block = op->getBlock();
      pto::PIPE pipe = bufSyncPipe(rlsBuf.getOpTypeAttr());
      setPipes(pipe, pipe);
    } else if (auto setDyn = dyn_cast<pto::SetFlagDynOp>(op)) {
      // Rotating hazards emit these. They must be recorded, or a gate reading the
      // emitted ops sees fewer sync ops than the kernel has.
      //
      // `eventId` stays -1 because the id is a RUNTIME value chosen by the selector
      // chain rather than an attribute. THE ID ITSELF IS THEREFORE UNCHECKED: no gate
      // here validates a rotating id, so a collision between two rotating hazards
      // would not be reported. Recovering the ids statically is possible -- the
      // selector is a chain of `arith.select` over constants -- and is not attempted.
      rec.opName = "pto.set_flag_dyn";
      setPipes(setDyn.getSrcPipeAttr().getPipe(),
               setDyn.getDstPipeAttr().getPipe());
    } else if (auto waitDyn = dyn_cast<pto::WaitFlagDynOp>(op)) {
      rec.opName = "pto.wait_flag_dyn";
      setPipes(waitDyn.getSrcPipeAttr().getPipe(),
               waitDyn.getDstPipeAttr().getPipe());
    } else if (auto barrier = dyn_cast<pto::BarrierOp>(op)) {
      rec.opName = "pto.barrier";
      pto::PIPE pipe = barrier.getPipeAttr().getPipe();
      setPipes(pipe, pipe);
    } else {
      return WalkResult::advance();
    }

    rec.op = op;
    rec.order = records.size();
    records.push_back(std::move(rec));
    return WalkResult::advance();
  });

  return records;
}

llvm::SmallVector<oracle::SyncOpRecord>
mlir::pto::oracle::extractFromSyncOps(const SyncOperations &ops) {
  llvm::SmallVector<SyncOpRecord> records;

  for (const auto &group : ops) {
    for (const auto &owned : group) {
      const SyncOperation *sync = owned.get();
      if (!sync)
        continue;
      SyncOpRecord rec;
      rec.type = SyncOperation::TypeName(sync->GetType());
      rec.typeCode = static_cast<int>(sync->GetType());
      rec.mechanism = SyncOperation::MechanismName(sync->GetMechanism());
      rec.mechanismCode = static_cast<int>(sync->GetMechanism());
      rec.srcPipe = static_cast<int>(sync->GetActualSrcPipe());
      rec.dstPipe = static_cast<int>(sync->GetActualDstPipe());
      for (int id : sync->eventIds)
        rec.eventIds.push_back(id);
      rec.syncIndex = sync->GetSyncIndex();
      rec.irIndex = sync->GetSyncIRIndex();
      rec.compensationOf = sync->compensationOf;
      // DERIVED, not copied from `SyncOperation::isCompensation`: that flag also
      // steers codegen placement and is deliberately left unset (see
      // synthesizeLoopCompensation). `compensationOf >= 0` is the oracle-only marker.
      rec.isCompensation = sync->compensationOf >= 0;
      records.push_back(std::move(rec));
    }
  }

  return records;
}


void mlir::pto::oracle::printIRRecords(llvm::raw_ostream &os,
                                       llvm::StringRef funcName,
                                       llvm::ArrayRef<IRSyncRecord> records) {
  os << "[sync-extract:ir] func=" << funcName << " records=" << records.size()
     << "\n";
  for (const IRSyncRecord &rec : records) {
    os << "  #" << rec.order << " " << rec.opName
       << " src=" << (rec.srcPipe.empty() ? "-" : rec.srcPipe)
       << " dst=" << (rec.dstPipe.empty() ? "-" : rec.dstPipe);
    if (rec.eventId >= 0)
      os << " event_id=" << rec.eventId;
    if (rec.bufId >= 0)
      os << " buf_id=" << rec.bufId;
    os << "\n";
  }
}

void mlir::pto::oracle::printSyncOpRecords(
    llvm::raw_ostream &os, llvm::StringRef funcName,
    llvm::ArrayRef<SyncOpRecord> records) {
  os << "[sync-extract:syncops] func=" << funcName
     << " records=" << records.size() << "\n";
  for (const SyncOpRecord &rec : records) {
    os << "  #" << rec.syncIndex << " " << rec.type
       << " mech=" << rec.mechanism
       << " src="
       << pipelineTypeName(static_cast<PipelineType>(rec.srcPipe))
       << " dst=" << pipelineTypeName(static_cast<PipelineType>(rec.dstPipe))
       << " irIdx=" << rec.irIndex << " event_ids=[";
    for (size_t i = 0; i < rec.eventIds.size(); ++i) {
      if (i)
        os << ",";
      os << rec.eventIds[i];
    }
    os << "]\n";
  }
}

namespace {

/// Renders the IR view of a function's emitted sync ops. Read-only.
struct PTODumpSyncExtractPass
    : public mlir::pto::impl::PTODumpSyncExtractBase<PTODumpSyncExtractPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;
    auto records = oracle::extractFromIR(func);
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printIRRecords(os, func.getSymName(), records);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTODumpSyncExtractPass() {
  return std::make_unique<PTODumpSyncExtractPass>();
}
