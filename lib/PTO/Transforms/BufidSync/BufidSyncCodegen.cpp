// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <tuple>
#include "BufidSyncCodegen.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"


#define DEBUG_TYPE "pto-bufid-sync"

using namespace mlir;
using namespace mlir::pto;

std::optional<pto::SyncOpType>
BufidSyncCodegen::mapPipelineToSyncOpType(PipelineType pipe) const {
  switch (pipe) {
  case PipelineType::PIPE_MTE2:
    return pto::SyncOpType::TLOAD;
  case PipelineType::PIPE_MTE3:
    return pto::SyncOpType::TSTORE_VEC;
  case PipelineType::PIPE_FIX:
    return pto::SyncOpType::TSTORE_ACC;
  case PipelineType::PIPE_MTE1:
    return pto::SyncOpType::TMOV_M2L;
  case PipelineType::PIPE_V:
    return pto::SyncOpType::TVEC;
  case PipelineType::PIPE_M:
    return pto::SyncOpType::TMATMUL;
  default:
    return std::nullopt;
  }
}

pto::PipeEventTypeAttr
BufidSyncCodegen::getOpTypeAttr(Builder &builder,
                                pto::SyncOpType opType) const {
  return pto::PipeEventTypeAttr::get(builder.getContext(), opType);
}

LogicalResult BufidSyncCodegen::run() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);

  WalkResult walkResult = func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto it = op2BufSync_.find(op);
    if (it == op2BufSync_.end()) {
      return WalkResult::advance();
    }

    auto &build = it->second;
    SmallVector<BufSyncOperation> pipeBefore(build.pipeBefore.begin(),
                                             build.pipeBefore.end());
    SmallVector<BufSyncOperation> pipeAfter(build.pipeAfter.begin(),
                                            build.pipeAfter.end());
    auto physicalIdFor = [&](const BufSyncOperation &sync) {
      return idAlloc_.getLogicToPhysical().lookup(sync.logicId);
    };
    std::sort(pipeBefore.begin(), pipeBefore.end(),
              [&](const BufSyncOperation &a, const BufSyncOperation &b) {
                return std::make_tuple(physicalIdFor(a),
                                       static_cast<int>(a.pipe), a.logicId) <
                       std::make_tuple(physicalIdFor(b),
                                       static_cast<int>(b.pipe), b.logicId);
              });
    std::sort(pipeAfter.begin(), pipeAfter.end(),
              [&](const BufSyncOperation &a, const BufSyncOperation &b) {
                return std::make_tuple(physicalIdFor(a),
                                       static_cast<int>(a.pipe), a.logicId) >
                       std::make_tuple(physicalIdFor(b),
                                       static_cast<int>(b.pipe), b.logicId);
              });

    for (auto &syncBefore : pipeBefore) {
      int physicalId = idAlloc_.getLogicToPhysical().lookup(syncBefore.logicId);
      auto syncOpType = mapPipelineToSyncOpType(syncBefore.pipe);
      if (!syncOpType) {
        op->emitError("bufid_sync cannot encode get_buf for unsupported pipe ")
            << static_cast<int>(syncBefore.pipe);
        return WalkResult::interrupt();
      }

      rewriter.setInsertionPoint(op);
      auto opTypeAttr = getOpTypeAttr(rewriter, *syncOpType);
      rewriter.create<pto::GetBufOp>(op->getLoc(), opTypeAttr,
                                     static_cast<uint32_t>(physicalId), 0);
    }

    for (auto &syncAfter : pipeAfter) {
      int physicalId = idAlloc_.getLogicToPhysical().lookup(syncAfter.logicId);
      auto syncOpType = mapPipelineToSyncOpType(syncAfter.pipe);
      if (!syncOpType) {
        op->emitError("bufid_sync cannot encode rls_buf for unsupported pipe ")
            << static_cast<int>(syncAfter.pipe);
        return WalkResult::interrupt();
      }

      if (op->hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(op);
      } else {
        rewriter.setInsertionPointAfter(op);
      }
      Attribute opTypeAttr = getOpTypeAttr(rewriter, *syncOpType);
      rewriter.create<pto::RlsBufOp>(op->getLoc(), opTypeAttr,
                                     static_cast<uint32_t>(physicalId), 0);
    }

    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}
