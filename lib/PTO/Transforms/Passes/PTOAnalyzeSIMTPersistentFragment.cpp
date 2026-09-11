// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/Passes.h"
#include "SIMTPersistentFragmentAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOANALYZESIMTPERSISTENTFRAGMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PTOAnalyzeSIMTPersistentFragmentPass
    : public pto::impl::PTOAnalyzeSIMTPersistentFragmentBase<
          PTOAnalyzeSIMTPersistentFragmentPass> {
  using pto::impl::PTOAnalyzeSIMTPersistentFragmentBase<
      PTOAnalyzeSIMTPersistentFragmentPass>::
      PTOAnalyzeSIMTPersistentFragmentBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    const auto &analysis = getAnalysis<pto::SIMTPersistentFragmentAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOAnalyzeSIMTPersistentFragmentPass() {
  return std::make_unique<PTOAnalyzeSIMTPersistentFragmentPass>();
}
