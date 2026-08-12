// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOOpCommon.td - Shared PTO operation definitions -----*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// This file owns constraints and operation base classes shared across PTO
// operation families. Concrete operations belong in a family ODS file.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOOPCOMMON
#define MLIR_DIALECT_PTO_IR_PTOOPCOMMON

include "PTO/IR/PTODialect.td"
include "PTO/IR/PTOAttrs.td"
include "PTO/IR/PTOTypeDefs.td"
include "PTO/IR/PTOInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def PTODpsType :
  AnyTypeOf<[AnyRankedTensor, PartitionTensorViewType, TileBufType]>;

def PTOPipeEntryType :
  AnyTypeOf<[AnyRankedTensor, TensorViewType, TileBufType],
            "TensorView, TileBuf, or Tensor">;

def PTOCommType :
  AnyTypeOf<[AnyRankedTensor, TensorViewType, PartitionTensorViewType,
             TileBufType], "TensorView, PartitionTensorView, TileBuf, or Tensor">;

def PtrOrMemRef :
  AnyTypeOf<[PtrType, AnyMemRef], "Ptr or MemRef">;

def ScalarType :
  AnyTypeOf<[AnySignlessInteger, AnyFloat], "numeric (integer/float)">;

def PrintScalarType :
  AnyTypeOf<[AnyInteger, AnyFloat], "integer/float printable scalar">;

def TensorViewLike :
  AnyTypeOf<[TensorViewType, PartitionTensorViewType],
            "TensorView or PartitionTensorView">;

def PTO_BufferType : Type<
  CPred<"::llvm::isa<::mlir::pto::PtrType>($_self)">,
  "pointer-like buffer type">;

def PTO_PipeEventTypeLikeAttr :
  AnyAttrOf<[PTO_PipeEventTypeAttr, PTO_SyncOpTypeAttr]>;

def PTO_PipeLikeAttr :
  AnyAttrOf<[PTO_PipeEventTypeAttr, PTO_SyncOpTypeAttr, PTO_PipeAttr]>;

class PTO_TOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic,
         !listconcat(traits, [TileOpInterface])>;

class PTO_DpsOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, traits # [DestinationStyleOpInterface]> {
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

class PTO_Op<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [ ])>;

class PTO_SectionOp<string mnemonic>
    : PTO_Op<mnemonic, [SingleBlock, NoTerminator]> {
  let summary = "Container for core-specific code guarded by macros";
  let description = [{
    During conversion to EmitC, this op is lowered to:
    emitc.verbatim("#if defined(MACRO)")
    ... inlined body ...
    emitc.verbatim("#endif")
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

#endif // MLIR_DIALECT_PTO_IR_PTOOPCOMMON
