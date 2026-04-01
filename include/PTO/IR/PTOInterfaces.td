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

// mlir/include/mlir/Dialect/PTO/IR/PTOInterfaces.td
 
#ifndef MLIR_DIALECT_PTO_IR_PTOINTERFACES
#define MLIR_DIALECT_PTO_IR_PTOINTERFACES
 
include "mlir/IR/OpBase.td"
 
// 定义 OpPipeInterface
def OpPipeInterface : OpInterface<"OpPipeInterface"> {
  let description = [{
    Interface for operations to expose their execution pipeline (Pipe) on Ascend architecture.
    This is used by the InjectSync pass to insert synchronization barriers between
    different hardware units (e.g., MTE2 -> Vector).
  }];
  
  let cppNamespace = "::mlir::pto";
 
  let methods = [
    InterfaceMethod<
      "Get the hardware pipe where this operation executes.",
      "::mlir::pto::PIPE",  // 返回类型对应 PTOAttrs.td 中的定义
      "getPipe"
    >
  ];
}

// PTO destination-style ops: have "init" operands (e.g. $dst) for plan-memory.
// No tensor/memref type requirement (unlike MLIR DestinationStyleOpInterface).
// Each op implements getDpsInitsMutable(); this interface provides getDpsInits().
def PTO_DpsInitOpInterface : OpInterface<"PTO_DpsInitOpInterface"> {
  let description = [{
    PTO ops that have destination (init) operands for memory planning. Init
    operands may be tile_buf, partition_tensor_view, tensor, or memref. Used by
    PTOPlanMemory to get written-to buffers without MLIR's tensor/memref-only
    verification.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<
      "Return the mutable range of destination (init) operands.",
      "::mlir::MutableOperandRange",
      "getDpsInitsMutable",
      (ins)
    >
  ];
  let extraSharedClassDeclaration = [{
    ::mlir::OperandRange getDpsInits() {
      return $_op.getDpsInitsMutable();
    }
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOINTERFACES
