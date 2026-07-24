// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOMATERIALIZEPIPESTATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr llvm::StringLiteral kPipeTileLibOwnedAttr =
    "pto.pipe_tilelib_owned";

static StructType getPipeStateType(MLIRContext *context) {
  Builder builder(context);
  return StructType::get(context, {builder.getI32Type(), builder.getI32Type()});
}

static bool isStatefulPipeOperation(Operation *op) {
  return isa<TAllocOp, TPushOp, TPopOp, TFreeOp>(op);
}

static Value getPipeHandle(Operation *op) {
  if (auto alloc = dyn_cast<TAllocOp>(op))
    return alloc.getPipeHandle();
  if (auto push = dyn_cast<TPushOp>(op))
    return push.getPipeHandle();
  if (auto pop = dyn_cast<TPopOp>(op))
    return pop.getPipeHandle();
  if (auto free = dyn_cast<TFreeOp>(op))
    return free.getPipeHandle();
  return {};
}

static Value getPipeState(Operation *op) {
  if (auto alloc = dyn_cast<TAllocOp>(op))
    return alloc.getPipeState();
  if (auto push = dyn_cast<TPushOp>(op))
    return push.getPipeState();
  if (auto pop = dyn_cast<TPopOp>(op))
    return pop.getPipeState();
  if (auto free = dyn_cast<TFreeOp>(op))
    return free.getPipeState();
  return {};
}

static void setPipeState(Operation *op, Value state) {
  if (auto alloc = dyn_cast<TAllocOp>(op)) {
    alloc.getPipeStateMutable().assign(state);
    return;
  }
  if (auto push = dyn_cast<TPushOp>(op)) {
    push.getPipeStateMutable().assign(state);
    return;
  }
  if (auto pop = dyn_cast<TPopOp>(op)) {
    pop.getPipeStateMutable().assign(state);
    return;
  }
  cast<TFreeOp>(op).getPipeStateMutable().assign(state);
}

static bool isInternalPipeInitializer(Operation *op) {
  return isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(op);
}

static FailureOr<uint8_t> getTerminalDrainSplit(Operation *initializer) {
  BoolAttr noSplit;
  if (auto l2l = dyn_cast<InitializeL2LPipeOp>(initializer))
    noSplit = l2l.getNosplitAttr();
  else if (auto l2g2l = dyn_cast<InitializeL2G2LPipeOp>(initializer))
    noSplit = l2g2l.getNosplitAttr();

  if (!noSplit) {
    initializer->emitOpError()
        << "requires resolved 'nosplit' before terminal pipe drain "
           "materialization";
    return failure();
  }

  // TPipe::~TPipe uses the pipe-level IsNoSplit configuration for cleanup,
  // rather than the split value of any individual producer operation.
  return noSplit.getValue() ? 0 : 1;
}

struct PTOMaterializePipeStatePass
    : public mlir::pto::impl::PTOMaterializePipeStateBase<
          PTOMaterializePipeStatePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    DominanceInfo dominance(funcOp);
    SmallVector<Operation *> initializers;
    SmallVector<func::ReturnOp> returns;

    funcOp.walk([&](Operation *op) {
      if (isInternalPipeInitializer(op))
        initializers.push_back(op);
      if (auto returnOp = dyn_cast<func::ReturnOp>(op))
        returns.push_back(returnOp);
    });

    for (Operation *initializer : initializers) {
      Value pipe = initializer->getResult(0);
      SmallVector<Operation *> pipeUsers;
      bool hasProducer = false;
      for (Operation *user : pipe.getUsers()) {
        if (user->getParentOfType<func::FuncOp>() != funcOp ||
            !isStatefulPipeOperation(user))
          continue;
        if (getPipeHandle(user) != pipe)
          continue;
        if (Value existingState = getPipeState(user)) {
          user->emitOpError()
              << "already has a pipe_state; "
                 "pto-materialize-pipe-state expects unmaterialized pipe IR";
          return signalPassFailure();
        }
        pipeUsers.push_back(user);
        hasProducer |= isa<TPushOp>(user);
      }

      if (pipeUsers.empty())
        continue;

      OpBuilder builder(initializer);
      builder.setInsertionPointAfter(initializer);
      StructType stateType = getPipeStateType(context);
      auto state = builder.create<DeclareStructOp>(initializer->getLoc(), stateType).getS();
      Value zero = builder.create<arith::ConstantIntOp>(initializer->getLoc(), 0, 32);
      auto path0 = DenseI64ArrayAttr::get(context, {0});
      auto path1 = DenseI64ArrayAttr::get(context, {1});
      builder.create<StructSetOp>(initializer->getLoc(), state, path0, zero);
      builder.create<StructSetOp>(initializer->getLoc(), state, path1, zero);

      for (Operation *user : pipeUsers) {
        setPipeState(user, state);
        user->setAttr(kPipeTileLibOwnedAttr, UnitAttr::get(context));
      }
      initializer->setAttr(kPipeTileLibOwnedAttr, UnitAttr::get(context));

      if (!hasProducer)
        continue;
      FailureOr<uint8_t> drainSplit = getTerminalDrainSplit(initializer);
      if (failed(drainSplit))
        return signalPassFailure();
      for (func::ReturnOp returnOp : returns) {
        if (!dominance.dominates(initializer, returnOp)) {
          initializer->emitOpError()
              << "does not dominate a return that requires terminal pipe drain";
          return signalPassFailure();
        }
        builder.setInsertionPoint(returnOp);
        auto drain =
            builder.create<TDrainOp>(returnOp.getLoc(), pipe, state, *drainSplit);
        drain->setAttr(kPipeTileLibOwnedAttr, UnitAttr::get(context));
      }
    }
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createPTOMaterializePipeStatePass() {
  return std::make_unique<PTOMaterializePipeStatePass>();
}

} // namespace pto
} // namespace mlir
