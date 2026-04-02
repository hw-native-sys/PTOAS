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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOREMOVEIDENTITYTMOV
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isA5Target(func::FuncOp funcOp) {
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
  return arch && arch.getValue() == "a5";
}

static bool canEraseIdentityTMov(TMovOp op) {
  if (op.getSrc() != op.getDst())
    return false;

  Value result = op.getResult();
  if (!result || result.use_empty())
    return true;

  return result.getType() == op.getDst().getType();
}

struct PTORemoveIdentityTMovPass
    : public mlir::pto::impl::PTORemoveIdentityTMovBase<
          PTORemoveIdentityTMovPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!isA5Target(funcOp))
      return;

    SmallVector<TMovOp> toErase;
    funcOp.walk([&](TMovOp op) {
      if (canEraseIdentityTMov(op))
        toErase.push_back(op);
    });

    for (TMovOp op : toErase) {
      Value result = op.getResult();
      if (result && !result.use_empty())
        result.replaceAllUsesWith(op.getDst());
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTORemoveIdentityTMovPass() {
  return std::make_unique<PTORemoveIdentityTMovPass>();
}
