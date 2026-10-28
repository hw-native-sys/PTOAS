// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIFastmathState.h - VMI fastmath pipeline state ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// The VMI conversion state is pipeline configuration, not an IR property: it
// says whether the NaN-relaxed fast paths (which may change NaN results) may be
// emitted for this compilation.  It is recorded once on the module as an
// attribute and read back by every pass that has to qualify a layout relation by
// the lowering state, so two passes can never disagree about which lowering is
// legal.
//
//===----------------------------------------------------------------------===//
#ifndef PTO_TRANSFORMS_VMIFASTMATHSTATE_H
#define PTO_TRANSFORMS_VMIFASTMATHSTATE_H

#include "PTO/IR/PTO.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::pto {

/// Module attribute recording the state.  The driver records --vmi-fastmath
/// here once; every VMI pass reads it back from this single place.
inline constexpr llvm::StringLiteral kVMIFastmathStateAttrName = "pto.vmi_fastmath";

/// The module that owns the compilation-wide state: the OUTERMOST module that
/// encloses \p scope, or null when \p scope is not nested in a module.
///
/// The state belongs to the compilation, not to a kernel module.  Resolving the
/// owner from the outermost module is what makes a nested kernel module's
/// \c pto.vmi_fastmath inert: no part of the input IR can shadow the driver's
/// --vmi-fastmath, and because the attribute is scrubbed before emission, such a
/// shadowing attempt cannot leave a trace either.
ModuleOp getVMIFastmathOwner(Operation* scope);

/// The state recorded by the owning module (see getVMIFastmathOwner): the value
/// of the outermost module's \c pto.vmi_fastmath.  A nested module's attribute
/// is inert.  An absent attribute means enabled - the historical default - so IR
/// that never states the pipeline configuration keeps its old meaning.
bool getVMIFastmathState(Operation* scope);

/// Record \p enable on \p module, replacing any previous value.  Callers that
/// want the compilation-wide state pass its owning module (the outermost one);
/// the -vmi-set-fastmath pass does that on the caller's behalf.
void setVMIFastmathState(ModuleOp module, bool enable);

/// Remove the recorded state from \p scope and every module nested in it, so the
/// pipeline configuration never leaks into emitted IR.
void clearVMIFastmathState(Operation* scope);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMIFASTMATHSTATE_H
