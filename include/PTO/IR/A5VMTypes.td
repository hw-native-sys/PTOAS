//===- A5VMTypes.td - A5VM type definitions -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_A5VMTYPES
#define MLIR_DIALECT_PTO_IR_A5VMTYPES

include "PTO/IR/A5VMDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def A5VM_VecType : TypeDef<A5VM_Dialect, "Vec"> {
  let mnemonic = "vec";
  let summary = "A 256-byte A5VM vector";

  let parameters = (ins
    "int64_t":$elementCount,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def A5VM_MaskType : TypeDef<A5VM_Dialect, "Mask"> {
  let mnemonic = "mask";
  let summary = "An A5VM predicate/mask register";
}

def A5VM_AlignType : TypeDef<A5VM_Dialect, "Align"> {
  let mnemonic = "align";
  let summary = "An A5VM vector_align carrier";
}

#endif // MLIR_DIALECT_PTO_IR_A5VMTYPES
