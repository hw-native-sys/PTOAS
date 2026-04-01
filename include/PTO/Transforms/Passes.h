// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// TODO.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"
#include "PTO/IR/PTODialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL
#include "PTO/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createLoweringSyncToPipePass();
std::unique_ptr<Pass> createPTOLowerFrontendPipeOpsPass();
std::unique_ptr<Pass> createPTOResolveReservedBuffersPass();
std::unique_ptr<Pass> createPTOWrapFunctionsInSectionsPass();
std::unique_ptr<Pass> createPTOVerifyTFreePass();

// Creates a pass for ...
std::unique_ptr<Pass> createPTOInsertSyncPass();
// Default arch is A3 unless overridden by callers.
std::unique_ptr<Pass> createEmitPTOManualPass();
// Explicitly select target arch for codegen.
std::unique_ptr<Pass> createEmitPTOManualPass(PTOArch arch);


/// Create a pass to convert ops from other dialects to PTO Ops.
std::unique_ptr<Pass> createConvertToPTOOpPass();

/// Create a pass to infer, propagate, and add memory scope information to
/// PTO Ops.
std::unique_ptr<Pass> createInferPTOMemScopePass();

/// Create a pass to plan memory.
std::unique_ptr<Pass>
createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption = {});

std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass();
std::unique_ptr<Pass> createPTOViewToMemrefPass();
std::unique_ptr<Pass> createInferPTOLayoutPass();
// Declare register function
void registerPTOPasses();

} // namespace pto
} // namespace mlir

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#undef GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "PTO/Transforms/Passes.h.inc"

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
