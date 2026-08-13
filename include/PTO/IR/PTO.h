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
// This compatibility header aggregates the common PTO IR declarations and all
// PTO operation classes. Internal components should include their narrow owner
// header when they do not require the complete operation surface.
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

#include "PTO/IR/PTOBase.h"
#include "PTO/IR/PTOTile.h"
#include "PTO/IR/VMI.h"
#include "PTO/IR/VPTO.h"

namespace mlir {
class MLIRContext;
class TypeConverter;

namespace pto {

/// The semantic form selected by the optional third tile of pto.tmov.  The
/// public operand remains named `fp` for API compatibility; address space is
/// the sole discriminator between legacy FP and exponent X-to-ZZ lowering.
enum class TMovForm {
  NoTileAux,
  Fp,
  XToZz,
};

TMovForm classifyTMovForm(Value fp);

/// Resolve the effective PTO target architecture from module-level IR state.
PTOArch getTargetArch(ModuleOp module);
PTOArch getTargetArch(Operation *op);
bool isTargetArchA3(ModuleOp module);
bool isTargetArchA5(ModuleOp module);
bool isTargetArchA3(Operation *op);
bool isTargetArchA5(Operation *op);

/// Return the target-specific alignment size in bytes for a supported
/// load/store vector op. Unsupported operations, modes, and targets return
/// std::nullopt.
std::optional<int64_t> getLoadStoreVecAlignmentSize(Operation *op);

/// Return the PTODSL logical function name when present, otherwise fall back to
/// the current symbol name. PTODSL uses this to mark ABI-specialized helper and
/// kernel-module symbols without relying on symbol-name parsing.
inline StringRef getPTODSLLogicalNameOrSymbolName(func::FuncOp func) {
  if (!func)
    return {};
  if (auto attr = func->getAttrOfType<StringAttr>(kPTODSLLogicalNameAttrName))
    return attr.getValue();
  return func.getSymName();
}

/// Return true if the function carries an explicit entry marker. PTO accepts
/// both the EmitC naming (`pto.entry`) and VPTO naming (`pto.kernel`) as entry
/// aliases; `hacc.entry` and `pto.aicore` are legacy aliases.
bool hasExplicitPTOEntryAttr(func::FuncOp func);
bool hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func);

/// Return true if the function should be emitted as an AICORE entry.
bool isPTOEntryFunction(func::FuncOp func);
bool isPTOEntryFunction(LLVM::LLVMFuncOp func);

/// Return true if the function should remain externally visible in backend
/// artifacts. PTO entries are always treated as externally visible. Non-entry
/// functions default to internal visibility unless they carry
/// `pto.visibility = "external"`.
bool hasExternalArtifactVisibility(func::FuncOp func);

/// Set explicit artifact visibility on one function definition.
void setExternalArtifactVisibility(func::FuncOp func, bool isExternal);

/// Validate module-level PTO entry configuration before EmitC lowering.
LogicalResult validatePTOEntryFunctions(ModuleOp module);

/// Reject !pto.struct function arguments/results and operation results other
/// than pto.declare_struct, so aliases cannot hide stack-storage provenance.
LogicalResult validateStructProvenance(ModuleOp module);

/// Compatibility hook kept for existing pass pipelines. This is now a no-op
/// because PTO entry state is expressed directly through explicit entry attrs
/// such as ``pto.entry``.
void annotatePTOEntryFunctions(ModuleOp module);

/// Look up a peer function for import_reserved_buffer-style cross-kernel links.
/// This first honors ordinary nearest symbol lookup, then falls back to the
/// outer backend-partitioned container and PTODSL ABI-specialized public
/// helper symbols when needed.
func::FuncOp lookupPeerFuncAcrossContainer(Operation *op,
                                           FlatSymbolRefAttr peerAttr);

/// Find one reserve_buffer by logical name inside a function.
ReserveBufferOp findReserveBufferByName(func::FuncOp funcOp, StringRef name);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTO_H_
