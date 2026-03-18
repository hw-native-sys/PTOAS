//===- A5VM.h - A5VM dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_A5VM_H_
#define MLIR_DIALECT_PTO_IR_A5VM_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "PTO/IR/A5VMDialect.h"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/A5VMTypes.h.inc"

#define GET_OP_CLASSES
#include "PTO/IR/A5VMOps.h.inc"

#endif // MLIR_DIALECT_PTO_IR_A5VM_H_
