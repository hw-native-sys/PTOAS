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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOVERIFYSUBKERNELPIPECONTRACT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static StringRef getSubkernelRole(func::FuncOp funcOp) {
  return pto::getPTODSLSubkernelHelperRole(funcOp);
}

static std::optional<PIPE> getExpectedPipeForRole(StringRef role) {
  return llvm::StringSwitch<std::optional<PIPE>>(role)
      .Case("simd", PIPE::PIPE_V)
      .Case("cube", PIPE::PIPE_M)
      .Default(std::nullopt);
}

static bool verifyPTOOpAgainstRole(Operation *op, PIPE expectedPipe,
                                   func::FuncOp rootHelper) {
  if (!isa<OpPipeInterface>(op))
    return true;

  PIPE actualPipe = cast<OpPipeInterface>(op).getPipe();
  if (actualPipe == expectedPipe)
    return true;

  InFlightDiagnostic diag =
      op->emitError("violates PTODSL subkernel pipe contract for root helper @");
  diag << rootHelper.getSymName() << ": expected PTO ops on "
       << stringifyPIPE(expectedPipe) << " only, but found "
       << stringifyPIPE(actualPipe);
  return false;
}

static LogicalResult verifySubkernelCallee(func::CallOp callOp,
                                           func::FuncOp callee,
                                           func::FuncOp rootHelper,
                                           PIPE expectedPipe,
                                           DenseSet<Operation *> &visited);

static LogicalResult verifySubkernelClosure(func::FuncOp currentFunc,
                                            func::FuncOp rootHelper,
                                            PIPE expectedPipe,
                                            DenseSet<Operation *> &visited) {
  if (!visited.insert(currentFunc.getOperation()).second)
    return success();

  bool sawFailure = false;
  ModuleOp module = currentFunc->getParentOfType<ModuleOp>();
  WalkResult result = currentFunc.walk([&](Operation *op) {
    if (op == currentFunc.getOperation())
      return WalkResult::advance();

    if (isa<func::ReturnOp>(op))
      return WalkResult::advance();

    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      if (!module || callOp.getCallee().empty()) {
        callOp.emitOpError("requires a resolvable same-module callee inside "
                           "PTODSL subkernel helper call closure");
        sawFailure = true;
        return WalkResult::interrupt();
      }
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (!callee) {
        callOp.emitOpError("requires a resolvable same-module callee inside "
                           "PTODSL subkernel helper call closure");
        sawFailure = true;
        return WalkResult::interrupt();
      }
      if (mlir::failed(verifySubkernelCallee(callOp, callee, rootHelper,
                                             expectedPipe, visited))) {
        sawFailure = true;
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    }

    if (op->getName().getDialectNamespace() != PTODialect::getDialectNamespace())
      return WalkResult::advance();

    if (!verifyPTOOpAgainstRole(op, expectedPipe, rootHelper)) {
      sawFailure = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted() || sawFailure)
    return failure();
  return success();
}

static LogicalResult verifySubkernelCallee(func::CallOp callOp,
                                           func::FuncOp callee,
                                           func::FuncOp rootHelper,
                                           PIPE expectedPipe,
                                           DenseSet<Operation *> &visited) {
  StringRef calleeRole = getSubkernelRole(callee);
  if (!calleeRole.empty()) {
    std::optional<PIPE> calleeExpectedPipe = getExpectedPipeForRole(calleeRole);
    if (!calleeExpectedPipe || *calleeExpectedPipe != expectedPipe) {
      callOp.emitOpError("cannot call PTODSL subkernel helper @")
          << callee.getSymName()
          << " with a different pipe role inside root helper @"
          << rootHelper.getSymName();
      return failure();
    }
  }

  return verifySubkernelClosure(callee, rootHelper, expectedPipe, visited);
}

struct PTOVerifySubkernelPipeContractPass
    : public mlir::pto::impl::PTOVerifySubkernelPipeContractBase<
          PTOVerifySubkernelPipeContractPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    StringRef role = getSubkernelRole(funcOp);
    if (role.empty())
      return;

    std::optional<PIPE> expectedPipe = getExpectedPipeForRole(role);
    if (!expectedPipe)
      return;

    DenseSet<Operation *> visited;
    if (mlir::failed(
            verifySubkernelClosure(funcOp, funcOp, *expectedPipe, visited))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVerifySubkernelPipeContractPass() {
  return std::make_unique<PTOVerifySubkernelPipeContractPass>();
}
