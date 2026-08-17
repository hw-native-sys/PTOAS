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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINDIRECTPTRNORMALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct IndirectLoad {
  pto::LoadScalarOp op;
  BlockArgument table;
  pto::PtrType pointee;
};

static LogicalResult normalizeFunction(func::FuncOp func) {
  SmallVector<IndirectLoad> loads;
  SmallVector<std::pair<BlockArgument, pto::PtrType>> tables;

  for (BlockArgument arg : func.getArguments()) {
    auto tableTy = dyn_cast<pto::PtrType>(arg.getType());
    if (!tableTy) {
      continue;
    }
    auto pointee = dyn_cast<pto::PtrType>(tableTy.getElementType());
    if (!pointee) {
      continue;
    }
    if (isa<pto::PtrType>(pointee.getElementType())) {
      return func.emitError()
             << "pointer tables support one level of indirection only";
    }

    for (OpOperand &use : arg.getUses()) {
      auto load = dyn_cast<pto::LoadScalarOp>(use.getOwner());
      bool isPointerOperand = use.getOperandNumber() == 0;
      if (!load || !isPointerOperand) {
        return func.emitError()
               << "indirect pointer-table argument must only be used as the "
                  "pointer operand of pto.load_scalar";
      }
      loads.push_back({load, arg, pointee});
    }
    tables.push_back({arg, tableTy});
  }

  if (tables.empty()) {
    return success();
  }

  MLIRContext *ctx = func.getContext();
  auto ui64 = IntegerType::get(ctx, 64, IntegerType::Unsigned);
  SmallVector<Type> inputs(func.getFunctionType().getInputs());
  for (auto [arg, oldTy] : tables) {
    auto addressTableTy = pto::PtrType::get(ctx, ui64, oldTy.getMemorySpace());
    inputs[arg.getArgNumber()] = addressTableTy;
    arg.setType(addressTableTy);
  }
  func.setFunctionType(FunctionType::get(
      ctx, inputs, func.getFunctionType().getResults()));

  for (IndirectLoad candidate : loads) {
    auto op = candidate.op;
    bool isDirectTableLoad = op.getPtr() == candidate.table;
    if (!isDirectTableLoad) {
      return op.emitError(
          "indirect pointer-table load must use the table argument directly");
    }

    OpBuilder builder(op);
    auto raw = builder.create<pto::LoadScalarOp>(
        op.getLoc(), ui64, candidate.table, op.getOffset());
    auto ptr = builder.create<pto::IntToPtrOp>(op.getLoc(), candidate.pointee,
                                               raw.getValue());
    op.getValue().replaceAllUsesWith(ptr.getResult());
    op.erase();
  }
  return success();
}

struct PTOIndirectPtrNormalizePass
    : public pto::impl::PTOIndirectPtrNormalizeBase<
          PTOIndirectPtrNormalizePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOIndirectPtrNormalizePass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(normalizeFunction(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOIndirectPtrNormalizePass() {
  return std::make_unique<PTOIndirectPtrNormalizePass>();
}
