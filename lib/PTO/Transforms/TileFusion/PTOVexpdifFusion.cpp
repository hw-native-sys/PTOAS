// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVEXPDIFFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasMatchingF32Types(VsubOp sub, VexpOp exp) {
  auto lhsType = dyn_cast<VRegType>(sub.getLhs().getType());
  auto rhsType = dyn_cast<VRegType>(sub.getRhs().getType());
  auto subResultType = dyn_cast<VRegType>(sub.getResult().getType());
  auto expSourceType = dyn_cast<VRegType>(exp.getInput().getType());
  auto expResultType = dyn_cast<VRegType>(exp.getResult().getType());
  if (!lhsType || !rhsType || !subResultType || !expSourceType ||
      !expResultType) {
    return false;
  }

  return lhsType.getElementType().isF32() &&
         rhsType.getElementType().isF32() &&
         subResultType.getElementType().isF32() &&
         expSourceType.getElementType().isF32() &&
         expResultType.getElementType().isF32() && lhsType == rhsType &&
         rhsType == subResultType && subResultType == expSourceType &&
         expSourceType == expResultType;
}

static bool canFuse(VsubOp sub, VexpOp exp) {
  // Fuse conservatively only when vexp is the sole user.
  if (!sub.getResult().hasOneUse()) {
    return false;
  }
  if (sub.getMask() != exp.getMask()) {
    return false;
  }
  return hasMatchingF32Types(sub, exp);
}

static void fuseVsubVexp(func::FuncOp func) {
  SmallVector<VexpOp> candidates;
  func.walk([&](VexpOp exp) {
    if (!exp->getParentOfType<FusionRegionOp>()) {
      return;
    }
    auto sub = exp.getInput().getDefiningOp<VsubOp>();
    if (sub && canFuse(sub, exp)) {
      candidates.push_back(exp);
    }
  });

  OpBuilder builder(func.getContext());
  for (VexpOp exp : candidates) {
    auto sub = exp.getInput().getDefiningOp<VsubOp>();
    if (!sub || !canFuse(sub, exp)) {
      continue;
    }

    builder.setInsertionPoint(exp);
    Location fusedLoc = builder.getFusedLoc({sub.getLoc(), exp.getLoc()});
    auto fused = builder.create<VexpdifOp>(
        fusedLoc, exp.getResult().getType(), sub.getLhs(), sub.getRhs(),
        exp.getMask(), builder.getStringAttr("ODD"));
    exp.getResult().replaceAllUsesWith(fused.getResult());
    exp.erase();
    sub.erase();
  }
}

struct PTOVexpdifFusionPass
    : pto::impl::PTOVexpdifFusionBase<PTOVexpdifFusionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    fuseVsubVexp(func);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVexpdifFusionPass() {
  return std::make_unique<PTOVexpdifFusionPass>();
}
