// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- OptMemPlanForPipeline.h --Pipeline optimization for plan memory------==//
#ifndef OPT_MEM_PLAN_FOR_PIPELINE_H
#define OPT_MEM_PLAN_FOR_PIPELINE_H
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "Utils.h"

namespace mlir {
namespace pto {

class OptMemPlanForDma {
public:
  OptMemPlanForDma(){};

  /// Main interface for OptMemPlanForDma.
  void build(func::FuncOp func);

  /// Check if buf1 and buf2 is dma and scalar pipe conflict.
  bool BufferPipeConflict(const Value buf1, const Value buf2) const;

  /// Is the current buffer used by DMA instructions.
  bool IsDmaBuffer(const Value buf) const;

  bool IsScalarBuffer(const Value buf) const;

private:
  /// Update the buffers for MTE2 and MTE3.
  void UpdateDmaBuffers(SmallVector<Value> dpsOperand);

  template <typename OP>
  typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                              std::is_same_v<OP, memref::StoreOp>,
                          void>::type
  UpdateScalarBuffers(OP op);

  void UpdateScalarBuffersForLowerToLoops(Operation *operands);

  /// Buffer in MTE2 and MTE3.
  DenseSet<Value> DmaBuffers;

  /// Buffer in Scalar.
  DenseSet<Value> ScalarBuffers;
};
} // namespace pto
} // namespace mlir

#endif // OPT_MEM_PLAN_FOR_PIPELINE_H
