// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSpineAnalysis.h - VMI direction-spine analysis -*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Pure IR analysis over VMI layout equivalence classes, shared by the layout
// assignment pass:
//
//   * the direction-spine recognition (closed widening/narrowing cast chains),
//   * the spine-scoped casts derived from it,
//   * the narrow-side-compute classification of width-changing casts.
//
// Every entry point only reads the op structure, so it can run before the
// solver decides any layout seed.
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTSPINEANALYSIS_H
#define PTO_TRANSFORMS_VMILAYOUTSPINEANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::pto {

/// Inline capacity of the per-module op sets the analysis fills, so that a
/// consumer declares its members with the same bound.
constexpr unsigned kDirectionSpineSetInlineCapacity = 8;

/// Collect the cast ops that are legs of a closed nested round trip (see the
/// direction-spine recognition in VMILayoutAssignment).  The result is a set of
/// ops, not of values: a leg is identified by the cast that implements it.
void collectDirectionSpineLegs(ModuleOp module,
                               llvm::SmallPtrSetImpl<Operation *> &spineLegs);

/// Subset of \p spineLegs plus the widening legs that consume them: the cast
/// ops whose reconciliation must use the spine-scoped cast layout table.
void collectSpineScopedCasts(
    const llvm::SmallPtrSetImpl<Operation *> &spineLegs,
    llvm::SmallPtrSetImpl<Operation *> &scopedCasts);

/// Width-changing casts whose sub-32-bit side carries elementwise compute in
/// its layout equivalence class: those sides are kept contiguous instead of
/// paying the lane-stride carrier inflation once per physical part.
void collectNarrowSideCompute(
    ModuleOp module, llvm::SmallPtrSetImpl<Operation *> &narrowSideCompute);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTSPINEANALYSIS_H
