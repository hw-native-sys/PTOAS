// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOWRAPFUNCTIONSINSECTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasExistingSection(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<SectionCubeOp, SectionVectorOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

template <typename SectionOpT>
static void wrapSingleBlockFuncBody(func::FuncOp funcOp) {
  Block &entryBlock = funcOp.getBody().front();
  Operation *terminator = entryBlock.getTerminator();

  OpBuilder builder(terminator);
  auto sectionOp = builder.create<SectionOpT>(funcOp.getLoc());
  sectionOp.getBody().push_back(new Block());
  Block &sectionBlock = sectionOp.getBody().front();

  auto sectionIt = Block::iterator(sectionOp.getOperation());
  sectionBlock.getOperations().splice(sectionBlock.end(),
                                      entryBlock.getOperations(),
                                      entryBlock.begin(), sectionIt);
}

static LogicalResult rewriteFunction(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return success();

  if (!funcOp.getBody().hasOneBlock())
    return funcOp.emitOpError(
        "requires a single-block body for kernel_kind wrapping");

  if (hasExistingSection(funcOp)) {
    return funcOp.emitOpError(
        "already contains pto.section.cube or pto.section.vector");
  }

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    wrapSingleBlockFuncBody<SectionCubeOp>(funcOp);
    return success();
  case FunctionKernelKind::Vector:
    wrapSingleBlockFuncBody<SectionVectorOp>(funcOp);
    return success();
  }

  llvm_unreachable("unexpected kernel kind");
}

struct PTOWrapFunctionsInSectionsPass
    : public mlir::pto::impl::PTOWrapFunctionsInSectionsBase<
          PTOWrapFunctionsInSectionsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (failed(rewriteFunction(funcOp)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOWrapFunctionsInSectionsPass() {
  return std::make_unique<PTOWrapFunctionsInSectionsPass>();
}
