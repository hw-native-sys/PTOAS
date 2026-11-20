// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- pto-test-opt.cpp - PTO lit pass runner -----------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/TypeSwitch.h"

namespace {

template <typename Effect>
bool hasEffectOn(llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effects,
                 mlir::Value value) {
  return llvm::any_of(effects, [value](const auto &effect) {
    return llvm::isa<Effect>(effect.getEffect()) && effect.getValue() == value &&
           effect.getResource() == mlir::SideEffects::DefaultResource::get();
  });
}

struct VerifyAsyncEffectsPass
    : mlir::PassWrapper<VerifyAsyncEffectsPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAsyncEffectsPass)

  llvm::StringRef getArgument() const final { return "pto-test-async-effects"; }

  void runOnOperation() override {
    auto result = getOperation().walk([](mlir::Operation *op) {
      mlir::Value session = llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
          .Case<mlir::pto::TPutAsyncGatherOp, mlir::pto::TGetAsyncScatterOp,
                mlir::pto::TPutAsyncOp,
                mlir::pto::TGetAsyncOp, mlir::pto::WaitAsyncEventOp,
                mlir::pto::TestAsyncEventOp>([](auto async) {
            return async.getSession();
          })
          .Default([](mlir::Operation *) { return mlir::Value(); });
      if (!session) {
        return mlir::WalkResult::advance();
      }
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
      llvm::cast<mlir::MemoryEffectOpInterface>(op).getEffects(effects);
      const auto hasReadWrite = [&effects](mlir::Value value) {
        return hasEffectOn<mlir::MemoryEffects::Read>(effects, value) &&
               hasEffectOn<mlir::MemoryEffects::Write>(effects, value);
      };
      const bool hasSessionEffects = hasReadWrite(session);
      const bool hasDefaultEffects = hasReadWrite({});
      if (!hasSessionEffects || !hasDefaultEffects) {
        op->emitError("expected read/write effects on the session and default memory");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::pto::PTODialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();

  mlir::registerAllPasses();
  mlir::pto::registerPTOPasses();
  mlir::PassRegistration<VerifyAsyncEffectsPass>();

  return failed(mlir::MlirOptMain(argc, argv, "PTO lit pass runner\n",
                                  registry));
}
