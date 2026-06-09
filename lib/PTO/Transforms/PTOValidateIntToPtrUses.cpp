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
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOVALIDATEINTTOPTRUSES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

static bool isAllowedIntToPtrUse(Value ptr, OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<LoadScalarOp, StoreScalarOp>(user))
    return use.getOperandNumber() == 0 && user->getOperand(0) == ptr;
  return false;
}

LogicalResult mlir::pto::validateIntToPtrUses(func::FuncOp func) {
  WalkResult walkResult = func.walk([](IntToPtrOp op) -> WalkResult {
    Value ptr = op.getResult();
    for (OpOperand &use : ptr.getUses()) {
      if (isAllowedIntToPtrUse(ptr, use))
        continue;

      Operation *user = use.getOwner();
      InFlightDiagnostic diag =
          op.emitOpError()
          << "result may only be used as the pointer operand of "
             "pto.load_scalar or pto.store_scalar; found use by '"
          << user->getName().getStringRef() << "'";
      diag.attachNote(user->getLoc()) << "disallowed pto.inttoptr use here";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

namespace {
struct PTOValidateIntToPtrUsesPass
    : public mlir::pto::impl::PTOValidateIntToPtrUsesBase<
          PTOValidateIntToPtrUsesPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateIntToPtrUsesPass)

  void runOnOperation() override {
    if (failed(validateIntToPtrUses(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOValidateIntToPtrUsesPass() {
  return std::make_unique<PTOValidateIntToPtrUsesPass>();
}
