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

#include "PTO/Support/CodeConstants.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// PTO Dialect
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTODialect.h"
#include "PTO/IR/VPTOAddressSemantics.h"
#include "PTO/IR/VPTOScheduling.h"

//===----------------------------------------------------------------------===//
// PTO Enums
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOEnums.h.inc"

//===----------------------------------------------------------------------===//
// PTO Interfaces
//===----------------------------------------------------------------------===//
 
#include "PTO/IR/PTOInterfaces.h.inc"
#include "PTO/IR/VPTOInterfaces.h.inc"

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

namespace mlir {
namespace pto {

//===----------------------------------------------------------------------===//
// S Fractal Size Constants
//===----------------------------------------------------------------------===//

/// Fractal size for mxBox layout (16x2 inner block, 32 bytes total).
inline constexpr int32_t kFractalMxSize = 32;

/// Fractal size for AB matrices in matmul (16xN inner block, 512 bytes).
inline constexpr int32_t kFractalABSize = 512;

/// Fractal size for C matrix in matmul (16x16 inner block, 1024 bytes).
inline constexpr int32_t kFractalCSize = 1024;

struct DmaLoopConfig {
  Value count;
  Value srcStride;
  Value dstStride;
};

struct DmaPadConfig {
  Value value;
  Value leftCount;
  Value rightCount;
};

struct AccStoreModeConfig {
  AccStoreMode mode;
  std::optional<Value> split;
  std::optional<Value> loop0SrcStride;
};

struct CubeLoadFracShapeConfig {
  Value nValue;
  Value dValue;
};

struct CubeLoadFracSrcLayoutConfig {
  Value srcInnerStride;
  std::optional<Value> srcOuterStride;
};

struct CubeLoadFracDstGroupConfig {
  Value groupCount;
  Value dstLoop2Stride;
  Value dstLoop3Stride;
  Value dstLoop4Stride;
};

struct CubeLoadFracCtrlConfig {
  Value l2CacheCtrl;
  Value smallc0En;
};

} // namespace pto
} // namespace mlir

#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.h.inc"

namespace mlir {

namespace pto {

inline constexpr char kPTOTargetArchAttrName[] = "pto.target_arch";

/// Get PTO Address Space Attr from input type.
AddressSpaceAttr getPTOAddressSpaceAttr(Type type);

/// Return true if type is a ptr/memref in GM address space (or default).

enum class PTOArch {
  A3,
  A5,
};

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

enum class PTOParserTargetArch {
  Unspecified,
  A3,
  A5,
};

void setPTOParserTargetArch(MLIRContext *context, PTOParserTargetArch arch);
PTOParserTargetArch getPTOParserTargetArch(MLIRContext *context);

class ScopedPTOParserTargetArch {
public:
  explicit ScopedPTOParserTargetArch(MLIRContext *context,
                                     PTOParserTargetArch arch);
  ~ScopedPTOParserTargetArch();

private:
  MLIRContext *context;
  PTOParserTargetArch previousArch;
};

/// Return the target-specific alignment size in bytes for a supported
/// load/store vector op. Unsupported operations, modes, and targets return
/// std::nullopt.
std::optional<int64_t> getLoadStoreVecAlignmentSize(Operation *op);

/// Function attributes that mark an explicit PTO kernel entry.
inline constexpr llvm::StringLiteral kPTOEntryAttrName = "pto.entry";
inline constexpr llvm::StringLiteral kLegacyHACCEntryAttrName = "hacc.entry";
inline constexpr llvm::StringLiteral kPTOKernelAttrName = "pto.kernel";
inline constexpr llvm::StringLiteral kLegacyPTOAICoreAttrName = "pto.aicore";
inline constexpr llvm::StringLiteral kPTOSimtEntryAttrName = "pto.simt_entry";
inline constexpr llvm::StringLiteral kPTOSimtMaxThreadsAttrName =
    "pto.simt_max_threads";
inline constexpr llvm::StringLiteral kPTOSimtMaxRegistersAttrName =
    "pto.simt_max_regs";
inline constexpr llvm::StringLiteral kPTOVisibilityAttrName = "pto.visibility";
inline constexpr llvm::StringLiteral kPTOVisibilityInternalValue = "internal";
inline constexpr llvm::StringLiteral kPTOVisibilityExternalValue = "external";
inline constexpr llvm::StringLiteral kPTODSLLogicalNameAttrName =
    "pto.ptodsl.logical_name";

/// Loop-unroll hint attributes carried on `scf.for` as discardable attrs.
///
/// `pto.unroll` is a string attribute; "full" and "enable" are supported.
/// `pto.unroll_factor` is an integer attribute holding a positive unroll
/// factor.  The two attributes are mutually exclusive on one loop.
///
/// Consumption contract:
/// - "full": `pto-unroll-loops` unrolls natively when the trip count is a
///   positive constant; otherwise the hint is dropped with a remark and the
///   loop is kept.
/// - "enable": never unrolled natively.  `pto-convert-scf-to-cf-with-loop-hints` translates
///   it into an llvm.loop_annotation that becomes !llvm.loop.unroll.enable
///   metadata, delegating the unroll decision to the compiler's cost model
///   (LLVM's ForceEnable semantics).
/// - `pto.unroll_factor`: unrolled natively by `pto-unroll-loops` when the
///   value satisfies `isValidUnrollFactorAttr`, the step is a positive
///   constant, and the factor does not exceed the pass's max-unroll-factor
///   cap; otherwise the hint is dropped with a remark.  Malformed hints
///   (unknown pto.unroll value, both attributes on one loop, out-of-contract
///   factor) are hard errors reported by `pto-unroll-loops`.
inline constexpr llvm::StringLiteral kUnrollAttrName = "pto.unroll";
inline constexpr llvm::StringLiteral kUnrollEnableValue = "enable";
inline constexpr llvm::StringLiteral kUnrollFullValue = "full";
inline constexpr llvm::StringLiteral kUnrollFactorAttrName =
    "pto.unroll_factor";

/// Unit attribute marking an `llvm.alloca` as a persistent SIMT fragment
/// buffer.  Shared by the persistent-fragment analysis and the loop
/// promotion pass (used as the discovery entry point).
inline constexpr llvm::StringLiteral kPersistentAttrName = "pto.persistent";

/// Internal marker set by pto-promote-persistent-fragment-loops on loops it
/// promoted to {pto.unroll = "full"}: pto-unroll-loops turns its
/// drop-with-remark fallback into a hard error for marked loops, because a
/// persistent loop that survives unrolling breaks fragment materialization.
inline constexpr llvm::StringLiteral kPersistentUnrollMarkerAttrName =
    "pto.persistent_unroll";

/// Check whether a `pto.unroll_factor` attribute value satisfies the
/// contract: a signless i32 holding a positive factor.  The factor is read
/// back as a signed value, so anything wider or non-positive would silently
/// truncate (e.g. an i64 2**31 becomes a negative factor).
inline bool isValidUnrollFactorAttr(IntegerAttr attr) {
  return attr && attr.getType().isSignlessInteger(kValue32) && attr.getInt() >= 1;
}

/// Return the PTODSL logical function name when present, otherwise fall back to
/// the current symbol name. PTODSL uses this to mark ABI-specialized helper and
/// kernel-module symbols without relying on symbol-name parsing.
inline StringRef getPTODSLLogicalNameOrSymbolName(func::FuncOp func) {
  if (!func) {
    return {};
  }
  if (auto attr = func->getAttrOfType<StringAttr>(kPTODSLLogicalNameAttrName)) {
    return attr.getValue();
  }
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
