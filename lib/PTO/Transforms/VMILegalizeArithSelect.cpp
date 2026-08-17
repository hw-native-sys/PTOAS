// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILegalizeArithSelect.cpp - Legalize VMI arith.select ------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILEGALIZEARITHSELECT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isVMIValueType(Type type) {
  return isa<VMIVRegType, VMIMaskType>(type);
}

static bool hasScalarI1Condition(arith::SelectOp select) {
  return select.getCondition().getType().isSignlessInteger(1);
}

static void rewriteSelectToIf(arith::SelectOp select) {
  OpBuilder builder(select);
  auto ifOp = builder.create<scf::IfOp>(
      select.getLoc(), TypeRange{select.getResult().getType()},
      select.getCondition(), /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    builder.create<scf::YieldOp>(select.getLoc(), select.getTrueValue());
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<scf::YieldOp>(select.getLoc(), select.getFalseValue());
  }

  select.getResult().replaceAllUsesWith(ifOp.getResult(0));
  select.erase();
}

struct VMILegalizeArithSelectPass
    : public mlir::pto::impl::VMILegalizeArithSelectBase<
          VMILegalizeArithSelectPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILegalizeArithSelectPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<arith::SelectOp> selects;
    module.walk([&](arith::SelectOp select) {
      if (isVMIValueType(select.getResult().getType()) &&
          hasScalarI1Condition(select)) {
        selects.push_back(select);
      }
    });

    for (arith::SelectOp select : llvm::reverse(selects)) {
      if (select->getBlock() != nullptr) {
        rewriteSelectToIf(select);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILegalizeArithSelectPass() {
  return std::make_unique<VMILegalizeArithSelectPass>();
}
