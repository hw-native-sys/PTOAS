// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIFastmathState.cpp - Record and read the VMI fastmath state --------===//
//===----------------------------------------------------------------------===//
//
// Defines the single source of truth for the VMI conversion state and the pass
// that records it.  The pass is the only writer: the driver creates it with the
// value of --vmi-fastmath, and test pipelines pin the state with
// -vmi-set-fastmath=vmi-fastmath=false.  Readers never take the state as a pass
// option, so they cannot disagree.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMIFastmathState.h"

#include "PTO/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMISETFASTMATHSTATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

ModuleOp mlir::pto::getVMIFastmathOwner(Operation* scope)
{
    // Only the outermost module speaks for the compilation.  Reading the nearest
    // enclosing module would let a nested kernel module carry a pto.vmi_fastmath
    // that silently overrides the driver's --vmi-fastmath.  Walking to the
    // outermost module keeps the state a pipeline-wide property that no part of
    // the input IR can shadow.
    ModuleOp outermost;
    for (Operation* op = scope; op; op = op->getParentOp()) {
        if (auto module = dyn_cast<ModuleOp>(op)) {
            outermost = module;
        }
    }
    return outermost;
}

bool mlir::pto::getVMIFastmathState(Operation* scope)
{
    ModuleOp owner = getVMIFastmathOwner(scope);
    if (!owner) {
        return true;
    }
    if (auto state = owner->getAttrOfType<BoolAttr>(kVMIFastmathStateAttrName)) {
        return state.getValue();
    }
    return true;
}

void mlir::pto::setVMIFastmathState(ModuleOp module, bool enable)
{
    module->setAttr(kVMIFastmathStateAttrName, BoolAttr::get(module.getContext(), enable));
}

void mlir::pto::clearVMIFastmathState(Operation* scope)
{
    if (!scope) {
        return;
    }
    scope->walk([](ModuleOp module) { module->removeAttr(kVMIFastmathStateAttrName); });
}

namespace {
struct VMISetFastmathStatePass : public mlir::pto::impl::VMISetFastmathStateBase<VMISetFastmathStatePass> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMISetFastmathStatePass)

    void runOnOperation() override
    {
        // The state is compilation-wide: record it where the readers look, on the
        // owning (outermost) module.  A pass scheduled on a nested module
        // therefore sets the state instead of silently no-op-ing.
        setVMIFastmathState(getVMIFastmathOwner(getOperation()), enableVmiFastmath);
    }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createVMISetFastmathStatePass()
{
    return std::make_unique<VMISetFastmathStatePass>();
}
