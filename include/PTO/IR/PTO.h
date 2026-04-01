// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.h - PTO Dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This file defines the dialect for the PTO Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTO_H_
#define MLIR_DIALECT_PTO_IR_PTO_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// PTO Dialect
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTODialect.h"

//===----------------------------------------------------------------------===//
// PTO Enums
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOEnums.h.inc"

//===----------------------------------------------------------------------===//
// PTO Interfaces
//===----------------------------------------------------------------------===//
 
#include "PTO/IR/PTOInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// PTO Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.h.inc"

//===----------------------------------------------------------------------===//
// PTO Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.h.inc"

//===----------------------------------------------------------------------===//
// PTO Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.h.inc"

namespace mlir {
class TypeConverter;

namespace pto {

inline constexpr char kPTOTargetArchAttrName[] = "pto.target_arch";

/// Get PTO Address Space Attr from input type.
AddressSpaceAttr getPTOAddressSpaceAttr(Type type);

/// Return true if type is a ptr/memref in GM address space (or default).
bool isScalarPtrOrMemRef(Type type);

enum class PTOArch {
  A3,
  A5,
};

/// Resolve the effective PTO target architecture from module-level IR state.
PTOArch getTargetArch(ModuleOp module);
PTOArch getTargetArch(Operation *op);
bool isTargetArchA3(ModuleOp module);
bool isTargetArchA5(ModuleOp module);
bool isTargetArchA3(Operation *op);
bool isTargetArchA5(Operation *op);

enum class PTOParserTargetArch {
  Unspecified,
  A3,
  A5,
};

void setPTOParserTargetArch(PTOParserTargetArch arch);
PTOParserTargetArch getPTOParserTargetArch();

class ScopedPTOParserTargetArch {
public:
  explicit ScopedPTOParserTargetArch(PTOParserTargetArch arch);
  ~ScopedPTOParserTargetArch();

private:
  PTOParserTargetArch previousArch;
};


/// Function attribute that marks an explicit PTO kernel entry.
inline constexpr llvm::StringLiteral kPTOEntryAttrName = "pto.entry";
inline constexpr llvm::StringLiteral kLegacyHACCEntryAttrName = "hacc.entry";

/// Return true if the function carries an explicit entry marker.
bool hasExplicitPTOEntryAttr(func::FuncOp func);

/// Return true if the function should be emitted as an AICORE entry.
bool isPTOEntryFunction(func::FuncOp func);

/// Validate module-level PTO entry configuration before EmitC lowering.
LogicalResult validatePTOEntryFunctions(ModuleOp module);

/// Materialize the effective PTO entry selection onto function attributes.
void annotatePTOEntryFunctions(ModuleOp module);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTO_H_
