// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- InferPTOMemScope.h --Infer Memory Scope for PTO Ops ----*- C++ -*-===//
//===----------------------------------------------------------------------===//
#ifndef PTO_INFERPTOMEMSCOPE_H
#define PTO_INFERPTOMEMSCOPE_H

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "Utils.h"

namespace mlir {
namespace pto {

class MemScopeInferAndPropagateHelper {
public:
  LogicalResult Run(Value operand, const AddressSpaceAttr &targetMemScope);

private:
  /// Propagate the memory scope change to users of the value.
  LogicalResult propagateMemScopeToUsers(Value val);

  /// Set memory scope for the root alloc op.
  void setMemRefAllocScope(memref::AllocOp op,
                           const AddressSpaceAttr &newScope);
  /// Set memory scope for the block argument.
  void setBlockArgumentScope(BlockArgument operand,
                             const AddressSpaceAttr &targetMemScope);
};

/// Infer, propagate, and set memory scope information to MmadL1Op.
/// \note MmadL1Op should be bufferized beforehand.

LogicalResult inferAndPropagateMemScopeForMatmulDps(TMatmulOp op);
LogicalResult inferAndPropagateMemScopeForMatmulAccDps(TMatmulAccOp op);
LogicalResult inferAndPropagateMemScopeForMatmulBiasDps(TMatmulBiasOp op);
LogicalResult inferAndPropagateMemScopeForMovDps(TMovOp op);
/// Infer, propagate, and set memory scope information to FuncOp.
/// \note FuncOp should be bufferized beforehand.
LogicalResult inferAndPropagateMemScopeForFunc(func::FuncOp op);

/// Infer, propagate, and set memory scope information to AllocOp.
/// \note Set alloc memory scope to ub.
LogicalResult inferAndPropagateUbufMemScope(memref::AllocOp allocOp);

/// Infer, propagate, and set memory scope information to GPUFuncOp.
LogicalResult inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op);

} // namespace pto
} // namespace mlir

#endif // PTO_INFERPTOMEMSCOPE_H
