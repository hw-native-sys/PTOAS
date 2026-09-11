// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOValidatePhysicalSectionBoundaries.cpp ---------------------------===//
//
// Verify the lexical SSA boundary of physical Cube/Vector sections.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOVALIDATEPHYSICALSECTIONBOUNDARIES
#include "PTO/Transforms/Passes.h.inc"

namespace {

static Operation *getNearestPhysicalSection(Operation *op) {
  for (Operation *current = op; current;
       current = current->getParentOp()) {
    if (isa<SectionCubeOp, SectionVectorOp>(current)) {
      return current;
    }
  }
  return nullptr;
}

static Operation *getNearestPhysicalSection(BlockArgument argument) {
  if (!argument) {
    return nullptr;
  }
  return getNearestPhysicalSection(argument.getOwner()->getParentOp());
}

static Operation *getDefiningPhysicalSection(Value value) {
  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    return getNearestPhysicalSection(blockArgument);
  }
  if (Operation *definingOp = value.getDefiningOp()) {
    return getNearestPhysicalSection(definingOp);
  }
  return nullptr;
}

static StringRef getPhysicalSectionKind(Operation *section) {
  if (!section) {
    return "outside a physical section";
  }
  return isa<SectionCubeOp>(section) ? "pto.section.cube"
                                     : "pto.section.vector";
}

static LogicalResult verifyOperandBoundary(Operation *user, OpOperand &operand) {
  Operation *definingSection = getDefiningPhysicalSection(operand.get());
  if (!definingSection) {
    return success();
  }

  Operation *usingSection = getNearestPhysicalSection(user);
  if (definingSection == usingSection) {
    return success();
  }

  return user->emitOpError()
         << "value defined in " << getPhysicalSectionKind(definingSection)
         << " cannot be used " << (usingSection ? "inside " : "outside ")
         << (usingSection ? getPhysicalSectionKind(usingSection)
                          : "the defining physical section")
         << "; physical sections have lexical SSA scope. Pass shared data "
            "through a caller-owned GM/UB buffer with explicit "
            "synchronization, or recompute it in the consumer section";
}

static LogicalResult verifyFunction(func::FuncOp function) {
  LogicalResult result = success();
  function.walk([&](Operation *op) {
    if (failed(result)) {
      return WalkResult::interrupt();
    }
    for (OpOperand &operand : op->getOpOperands()) {
      if (failed(verifyOperandBoundary(op, operand))) {
        result = failure();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

struct PTOValidatePhysicalSectionBoundariesPass
    : public impl::PTOValidatePhysicalSectionBoundariesBase<
          PTOValidatePhysicalSectionBoundariesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LogicalResult result = success();
    module.walk([&](func::FuncOp function) {
      if (failed(result)) {
        return WalkResult::interrupt();
      }
      result = verifyFunction(function);
      return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace pto
} // namespace mlir

std::unique_ptr<mlir::Pass>
mlir::pto::createPTOValidatePhysicalSectionBoundariesPass() {
  return std::make_unique<mlir::pto::PTOValidatePhysicalSectionBoundariesPass>();
}
