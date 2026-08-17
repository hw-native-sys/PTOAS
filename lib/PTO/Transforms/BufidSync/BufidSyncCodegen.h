// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCCODEGEN_H
#define MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCCODEGEN_H

#include <optional>
#include "BufidSyncAnalysis.h"
#include "BufidSyncIdAlloc.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"


namespace mlir {
namespace pto {

class BufidSyncCodegen {
public:
  BufidSyncCodegen(func::FuncOp func,
                   const DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync,
                   const BufidSyncIdAlloc &idAlloc)
      : func_(func), op2BufSync_(op2BufSync), idAlloc_(idAlloc) {}

  LogicalResult run();

private:
  std::optional<pto::SyncOpType>
  mapPipelineToSyncOpType(PipelineType pipe) const;
  pto::PipeEventTypeAttr getOpTypeAttr(Builder &builder,
                                       pto::SyncOpType opType) const;

  func::FuncOp func_;
  const DenseMap<Operation *, BufSyncPipeBuild> &op2BufSync_;
  const BufidSyncIdAlloc &idAlloc_;
};

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_BUFIDSYNC_BUFIDSYNCCODEGEN_H
