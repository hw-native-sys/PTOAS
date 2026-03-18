//===- A5VMDialect.td - A5VM dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_A5VMDIALECT
#define MLIR_DIALECT_PTO_IR_A5VMDIALECT

include "mlir/IR/OpBase.td"

def A5VM_Dialect : Dialect {
  let name = "a5vm";
  let cppNamespace = "::mlir::pto::a5vm";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // MLIR_DIALECT_PTO_IR_A5VMDIALECT
