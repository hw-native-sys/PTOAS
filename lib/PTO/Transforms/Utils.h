// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_UTILS_UTILS_H
#define MLIR_DIALECT_PTO_UTILS_UTILS_H
#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <queue>
#include <set>
#include <type_traits>

namespace mlir {
namespace pto {
  const std::set<pto::AddressSpace> LocalBufferSpace{
    pto::AddressSpace::VEC, pto::AddressSpace::MAT, pto::AddressSpace::ACC, pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT, pto::AddressSpace::BIAS, pto::AddressSpace::SCALING};
  constexpr const uint8_t kBitsToByte = 8;
  func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp);
  std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation *op);
  std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand);
  bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr);
  Value tracebackMemRef(Value memrefVal);
  std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes);
  uint64_t AlignUp(uint64_t lhs, uint64_t rhs);
  LoopLikeOpInterface getParentLoop(Value val);
  ModuleOp getTopLevelModuleOp(Operation *op);
  void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope);
  BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             AddressSpaceAttr targetMemScope);
  std::optional<memref::AllocOp> tracebackMemRefToAlloc(Value memrefVal);
  bool isFromFunctionArg(mlir::Value v);
  bool isOpTouchLocalBuffer(Operation *op);
}
}
#endif
