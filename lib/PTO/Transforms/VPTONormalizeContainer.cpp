// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTONORMALIZECONTAINER
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isVPTOKernelSubmodule(ModuleOp module) {
  return module->hasAttr(FunctionKernelKindAttr::name);
}

static LogicalResult verifyNormalizedVPTOContainer(ModuleOp module) {
  bool hasChildModules = false;
  for (Operation &op : module.getBodyRegion().front().getOperations()) {
    auto child = dyn_cast<ModuleOp>(op);
    if (!child) {
      return op.emitError()
             << "expected VPTO container top level to contain only kernel "
                "submodules";
    }
    hasChildModules = true;
    if (!isVPTOKernelSubmodule(child)) {
      return child.emitError()
             << "expected VPTO kernel submodule to carry 'pto.kernel_kind'";
    }
  }

  if (hasChildModules) {
    return success();
  }

  return module.emitError()
         << "expected VPTO input to be a kernel submodule with "
            "'pto.kernel_kind' or a container of kernel submodules";
}

struct VPTONormalizeContainerPass
    : public mlir::pto::impl::VPTONormalizeContainerBase<
          VPTONormalizeContainerPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (isVPTOKernelSubmodule(module)) {
      MLIRContext *context = module.getContext();
      SmallVector<NamedAttribute> outerAttrs;
      for (NamedAttribute attr : module->getAttrs()) {
        if (attr.getName() != SymbolTable::getSymbolAttrName() &&
            attr.getName() != FunctionKernelKindAttr::name) {
          outerAttrs.push_back(attr);
        }
      }

      auto child = ModuleOp::create(module.getLoc());
      child->setAttrs(module->getAttrDictionary());
      child.getBodyRegion().takeBody(module.getBodyRegion());

      module->setAttrs(DictionaryAttr::get(context, outerAttrs));
      module.getBodyRegion().push_back(new Block);
      module.getBodyRegion().front().push_back(child.getOperation());
    }

    if (failed(verifyNormalizedVPTOContainer(module))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::pto {

std::unique_ptr<Pass> createVPTONormalizeContainerPass() {
  return std::make_unique<VPTONormalizeContainerPass>();
}

} // namespace mlir::pto
