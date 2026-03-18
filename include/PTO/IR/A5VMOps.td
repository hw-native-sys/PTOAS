//===- A5VMOps.td - A5VM operations -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_A5VMOPS
#define MLIR_DIALECT_PTO_IR_A5VMOPS

include "PTO/IR/A5VMDialect.td"
include "PTO/IR/A5VMTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def A5VM_VectorType : Type<CPred<"::llvm::isa<::mlir::pto::a5vm::VecType>($_self)">,
                           "A5VM vector type">;

class A5VM_Op<string mnemonic, list<Trait> traits = []>
    : Op<A5VM_Dialect, mnemonic, traits>;

def A5VM_LoadOp : A5VM_Op<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    AnyMemRef:$base,
    Index:$offset,
    StrAttr:$layout,
    StrAttr:$domain,
    I64Attr:$valid_rows,
    I64Attr:$valid_cols
  );

  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $base `[` $offset `]` attr-dict `:` type($base) `->` type($result)
  }];
}

def A5VM_AbsOp : A5VM_Op<"abs", [Pure]> {
  let arguments = (ins A5VM_VectorType:$input);
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_StoreOp : A5VM_Op<"store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    AnyMemRef:$base,
    Index:$offset,
    StrAttr:$layout,
    StrAttr:$domain
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $base `[` $offset `]` attr-dict `:` type($value) `,` type($base)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_A5VMOPS
