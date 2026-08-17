// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIUtils.h - PTO VMI shared helpers ----------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VMIUTILS_H
#define PTO_IR_VMIUTILS_H

#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::pto {

inline constexpr StringLiteral kVMIDiagUnsupported = "VMI-UNSUPPORTED";
inline constexpr StringLiteral kVMIDiagLayoutContract =
    "VMI-LAYOUT-CONTRACT";
inline constexpr StringLiteral kVMIDiagPassInvariant = "VMI-PASS-INVARIANT";
inline constexpr StringLiteral kVMIDiagResidualOp = "VMI-RESIDUAL-OP";

inline constexpr StringLiteral kVMIDiagUnsupportedPrefix =
    "VMI-UNSUPPORTED: ";
inline constexpr StringLiteral kVMIDiagLayoutContractPrefix =
    "VMI-LAYOUT-CONTRACT: ";
inline constexpr StringLiteral kVMIDiagPassInvariantPrefix =
    "VMI-PASS-INVARIANT: ";
inline constexpr StringLiteral kVMIDiagResidualOpPrefix = "VMI-RESIDUAL-OP: ";

struct VMIPhysicalLane {
  int64_t part = 0;
  int64_t chunk = 0;
  int64_t lane = 0;
};

// VMI lane layouts never change the data element type of a physical part.
// Wider values used by pack/unpack or memory distributions are
// instruction-local carriers rather than the type of the VMI SSA value.
Type getVMIPhysicalDataElementType(VMIVRegType type);
FailureOr<int64_t> getDataLanesPerPart(Type elementType);
FailureOr<int64_t> getMaskLanesPerPart(StringRef granularity);
FailureOr<int64_t> getVMILayoutBlockElems(Type type);
FailureOr<int64_t> getVMIPhysicalArity(Type type);
FailureOr<VMIPhysicalLane> mapLogicalLaneToPhysical(Type type,
                                                     int64_t logicalLane);
FailureOr<int64_t> mapPhysicalLaneToLogical(Type type, int64_t part,
                                             int64_t chunk, int64_t lane);
FailureOr<bool> isPaddingLane(Type type, int64_t part, int64_t chunk,
                              int64_t lane);

// ---------------------------------------------------------------------------
// VMI FpToSi hardware contract (mirrored from VPTO lookupVcvtContract).
// Used by VMI verifiers and VMIToVPTO lowering to agree on requiresSat /
// requiresPart without duplicating the logic.
// ---------------------------------------------------------------------------

struct VMIFpToSiContract {
  bool requiresSat = false;
  bool requiresPart = false;
};

/// Returns the FpToSi contract for the given src→dst element type pair,
/// or nullopt if this float→signed-int path is not supported by the hardware.
std::optional<VMIFpToSiContract>
lookupVMIFpToSiContract(Type srcElem, Type dstElem);

// ---------------------------------------------------------------------------
// VMI FpToUi hardware contract (mirrors VPTO lookupVcvtContract).
// Symmetric to FpToSi but for float→unsigned-int paths.
// ---------------------------------------------------------------------------

struct VMIFpToUiContract {
  bool requiresSat = false;
  bool requiresPart = false;
};

/// Returns the FpToUi contract for the given src→dst element type pair,
/// or nullopt if this float→unsigned-int path is not supported by the hardware.
std::optional<VMIFpToUiContract>
lookupVMIFpToUIContract(Type srcElem, Type dstElem);

// ---------------------------------------------------------------------------
// VMI FpToFp hardware contract (VMI-owned; may diverge from VPTO).
// Enumerates same-width fp->fp whitelist pairs plus the fp->fp narrow paths
// whose sat semantics differ from the truncf default (e.g. bf16x2->f4x2).
// ---------------------------------------------------------------------------

struct VMIFpToFpContract {
  bool requiresRnd = false;
  bool requiresSat = false;
  bool requiresPart = false;
  StringRef allowedRndModes = StringRef();
};

/// Returns the FpToFp contract for the given src->dst element type pair,
/// or nullopt if this fp-to-fp path is not supported by the VMI contract.
std::optional<VMIFpToFpContract>
lookupVMIFpToFpContract(Type srcElem, Type dstElem);

} // namespace mlir::pto

#endif // PTO_IR_VMIUTILS_H
