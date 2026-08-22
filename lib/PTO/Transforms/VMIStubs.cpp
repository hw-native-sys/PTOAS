// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Stub implementations for passes declared in Passes.td but whose
// full implementations reference rebuild-only IR ops not present in main.
#include "PTO/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::pto {

namespace {
struct StubPTOViewToMemrefPass
    : public PassWrapper<StubPTOViewToMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StubPTOViewToMemrefPass)

  void runOnOperation() override {}
  StringRef getArgument() const final { return "pto-view-to-memref"; }
  StringRef getDescription() const final { return "Stub"; }
};
struct StubPTOMaterializeTileHandlesPass
    : public PassWrapper<StubPTOMaterializeTileHandlesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StubPTOMaterializeTileHandlesPass)

  void runOnOperation() override {}
  StringRef getArgument() const final { return "pto-materialize-tile-handles"; }
  StringRef getDescription() const final { return "Stub"; }
};
struct StubVPTONormalizeEquivalentVcvtPass
    : public PassWrapper<StubVPTONormalizeEquivalentVcvtPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      StubVPTONormalizeEquivalentVcvtPass)

  void runOnOperation() override {}
  StringRef getArgument() const final { return "pto-vpto-normalize-equiv-vcvt"; }
  StringRef getDescription() const final { return "Stub"; }
};
} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<StubPTOViewToMemrefPass>();
}
std::unique_ptr<Pass> createPTOMaterializeTileHandlesPass() {
  return std::make_unique<StubPTOMaterializeTileHandlesPass>();
}
std::unique_ptr<Pass> createVPTONormalizeEquivalentVcvtPass() {
  return std::make_unique<StubVPTONormalizeEquivalentVcvtPass>();
}

} // namespace mlir::pto
