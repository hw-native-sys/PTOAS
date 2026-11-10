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

// Explicit phase-bearing operation contracts keep token visibility in sync
// with ODS rather than a hand-maintained list in the synchronization analysis.
def UFAccProducerOpInterface : OpInterface<"UFAccProducerOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let methods = [InterfaceMethod<"Return the accumulator unit-flag phase.",
                                 "::mlir::pto::AccPhase", "getAccPhase">];
}

def UFAccConsumerOpInterface : OpInterface<"UFAccConsumerOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let methods = [InterfaceMethod<"Return the drain unit-flag phase.",
                                 "::mlir::pto::STPhase", "getStPhase">,
                 InterfaceMethod<"Return the drain source.",
                                 "::mlir::Value", "getSrc">];
}

// Instruction-class marker interfaces. These categories are independent from
// OpPipeInterface: a pipe describes where an operation executes, while these
// markers describe the operation's abstraction level and instruction family.
def TileOpInterface : OpInterface<"TileOpInterface"> {
  let description = [{
    Marks a high-level PTO Tile operation with tile-granular semantics.
  }];
  let cppNamespace = "::mlir::pto";
}

def MicroOpInterface : OpInterface<"MicroOpInterface"> {
  let description = [{
    Marks a low-level PTO micro-instruction that directly models a hardware
    instruction rather than a structural, view, or high-level Tile operation.
  }];
  let cppNamespace = "::mlir::pto";
}

def VectorMicroOpInterface : OpInterface<"VectorMicroOpInterface"> {
  let description = [{
    Marks a vector micro-instruction operating on vector registers, masks, or
    vector memory state.
  }];
  let cppNamespace = "::mlir::pto";
}

def CubeMicroOpInterface : OpInterface<"CubeMicroOpInterface"> {
  let description = [{
    Marks a Cube micro-instruction operating on the matrix execution unit.
  }];
  let cppNamespace = "::mlir::pto";
}

def MteOpInterface : OpInterface<"MteOpInterface"> {
  let description = [{
    Marks an operation that performs MTE/DMA data movement. High-level Tile
    operations and raw micro-instructions may both carry this marker.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the source buffer.",
      "::mlir::Value", "getSource", (ins), [{}],
      /*defaultImplementation=*/[{
        auto operation = $_op.getOperation();
        if (auto dps = ::llvm::dyn_cast<::mlir::DestinationStyleOpInterface>(
                operation)) {
          auto inputs = dps.getDpsInputOperands();
          if (!inputs.empty())
            return inputs.front()->get();
        }
        return operation->getNumOperands() ? operation->getOperand(0)
                                           : ::mlir::Value();
      }]>,
    InterfaceMethod<"Return the destination buffer.",
      "::mlir::Value", "getDestination", (ins), [{}],
      /*defaultImplementation=*/[{
        auto operation = $_op.getOperation();
        if (auto dps = ::llvm::dyn_cast<::mlir::DestinationStyleOpInterface>(
                operation)) {
          auto inits = dps.getDpsInitOperands();
          if (!inits.empty())
            return inits.front()->get();
        }
        return operation->getNumOperands() > 1 ? operation->getOperand(1)
                                              : ::mlir::Value();
      }]>
  ];
}

def SimtOpInterface : OpInterface<"SimtOpInterface"> {
  let description = [{
    Marks an instruction executed directly by a SIMT entry. This excludes the
    pto.simt_launch boundary operation.
  }];
  let cppNamespace = "::mlir::pto";
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
