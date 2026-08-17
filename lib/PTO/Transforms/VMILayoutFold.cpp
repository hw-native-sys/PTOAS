// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutFold.cpp - Fold VMI layout materializations --------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTFOLD
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasSameDataShapeAndElementType(VMIVRegType lhs, VMIVRegType rhs) {
  return lhs && rhs && lhs.getElementCount() == rhs.getElementCount() &&
         lhs.getElementType() == rhs.getElementType();
}

static bool isUnitContiguousLayout(VMILayoutAttr layout) {
  return layout && layout.isContiguous() && layout.getLaneStride() == 1;
}

static bool isLoadProducerLayout(VMIVRegType type) {
  if (!type) {
    return false;
  }
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout) {
    return false;
  }
  if (layout.isContiguous() && layout.getLaneStride() == 1) {
    return true;
  }
  if (layout.isContiguous() && layout.getLaneStride() == mlir::pto::kValue2) {
    unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
    return elementBits == mlir::pto::kValue8 || elementBits == 16 || elementBits == 32;
  }
  if (layout.isContiguous() && layout.getLaneStride() == mlir::pto::kValue4) {
    unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
    return elementBits == mlir::pto::kValue8;
  }
  if (!layout.isDeinterleaved() || layout.getLaneStride() != 1 ||
      (layout.getFactor() != mlir::pto::kValue2 &&
       layout.getFactor() != mlir::pto::kValue4)) {
    return false;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  return elementBits == mlir::pto::kValue8 || elementBits == 16 || elementBits == 32;
}

static bool isFoldableLoadEnsure(VMIEnsureLayoutOp ensure) {
  auto load = ensure.getSource().getDefiningOp<VMILoadOp>();
  if (!load) {
    return false;
  }

  auto sourceType = dyn_cast<VMIVRegType>(ensure.getSource().getType());
  auto resultType = dyn_cast<VMIVRegType>(ensure.getResult().getType());
  if (!hasSameDataShapeAndElementType(sourceType, resultType)) {
    return false;
  }

  return isLoadProducerLayout(resultType);
}

static void tryFoldLoadEnsures(
    VMILoadOp load, SmallVectorImpl<VMIEnsureLayoutOp> &maybeDeadEnsures) {
  auto sourceType = dyn_cast<VMIVRegType>(load.getResult().getType());
  if (!sourceType) {
    return;
  }

  VMIVRegType targetType;
  SmallVector<VMIEnsureLayoutOp> ensures;
  for (OpOperand &use : load.getResult().getUses()) {
    auto ensure = dyn_cast<VMIEnsureLayoutOp>(use.getOwner());
    if (!ensure || use.getOperandNumber() != 0 || !isFoldableLoadEnsure(ensure)) {
      return;
    }

    auto resultType = cast<VMIVRegType>(ensure.getResult().getType());
    if (!targetType) {
      targetType = resultType;
    } else if (targetType != resultType) {
      return;
    }
    ensures.push_back(ensure);
  }

  if (ensures.empty() || targetType == sourceType) {
    return;
  }

  load.getResult().setType(targetType);
  for (VMIEnsureLayoutOp ensure : ensures) {
    ensure.getResult().replaceAllUsesWith(load.getResult());
    maybeDeadEnsures.push_back(ensure);
  }
}

static void
tryFoldNestedEnsureLayout(VMIEnsureLayoutOp ensure,
                          SmallVectorImpl<VMIEnsureLayoutOp> &maybeDeadEnsures) {
  auto inner = ensure.getSource().getDefiningOp<VMIEnsureLayoutOp>();
  if (!inner) {
    return;
  }

  if (inner.getSource().getType() != ensure.getResult().getType()) {
    return;
  }

  ensure.getResult().replaceAllUsesWith(inner.getSource());
  maybeDeadEnsures.push_back(ensure);
  maybeDeadEnsures.push_back(inner);
}

static bool isFoldableStoreEnsure(VMIEnsureLayoutOp ensure) {
  auto sourceType = dyn_cast<VMIVRegType>(ensure.getSource().getType());
  auto resultType = dyn_cast<VMIVRegType>(ensure.getResult().getType());
  if (!sourceType || !resultType) {
    return false;
  }

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!isUnitContiguousLayout(resultLayout) ||
      isUnitContiguousLayout(sourceLayout)) {
    return false;
  }

  VMILayoutSupport supports;
  return succeeded(supports.getStoreLayoutFact(sourceType));
}

static void tryFoldEnsureLayoutIntoOperand(
    OpOperand &operand, SmallVectorImpl<VMIEnsureLayoutOp> &maybeDeadEnsures) {
  auto ensure = operand.get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!ensure || !isFoldableStoreEnsure(ensure)) {
    return;
  }

  operand.set(ensure.getSource());
  maybeDeadEnsures.push_back(ensure);
}

static void tryFoldEnsureLayoutIntoMaskedStore(
    VMIMaskedStoreOp store,
    SmallVectorImpl<VMIEnsureLayoutOp> &maybeDeadEnsures,
    SmallVectorImpl<VMIEnsureMaskLayoutOp> &maybeDeadMaskEnsures) {
  auto ensure = store.getValue().getDefiningOp<VMIEnsureLayoutOp>();
  if (!ensure || !isFoldableStoreEnsure(ensure)) {
    return;
  }
  auto maskEnsure = store.getMask().getDefiningOp<VMIEnsureMaskLayoutOp>();
  if (!maskEnsure) {
    return;
  }

  auto sourceType = dyn_cast<VMIVRegType>(ensure.getSource().getType());
  auto maskSourceType = dyn_cast<VMIMaskType>(maskEnsure.getSource().getType());
  auto maskResultType = dyn_cast<VMIMaskType>(maskEnsure.getResult().getType());
  if (!sourceType || !maskSourceType || !maskResultType) {
    return;
  }

  VMILayoutAttr maskResultLayout = maskResultType.getLayoutAttr();
  if (!isUnitContiguousLayout(maskResultLayout)) {
    return;
  }

  VMILayoutSupport supports;
  if (failed(supports.getMaskedStoreLayoutFact(sourceType, maskSourceType))) {
    return;
  }

  store.getValueMutable().set(ensure.getSource());
  store.getMaskMutable().set(maskEnsure.getSource());
  maybeDeadEnsures.push_back(ensure);
  maybeDeadMaskEnsures.push_back(maskEnsure);
}

struct VMILayoutFoldPass
    : public mlir::pto::impl::VMILayoutFoldBase<VMILayoutFoldPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutFoldPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<VMIEnsureLayoutOp> maybeDeadEnsures;
    SmallVector<VMIEnsureMaskLayoutOp> maybeDeadMaskEnsures;

    module.walk([&](VMILoadOp load) {
      tryFoldLoadEnsures(load, maybeDeadEnsures);
    });

    module.walk([&](VMIEnsureLayoutOp ensure) {
      tryFoldNestedEnsureLayout(ensure, maybeDeadEnsures);
    });

    module.walk([&](Operation *op) {
      if (auto store = dyn_cast<VMIStoreOp>(op)) {
        tryFoldEnsureLayoutIntoOperand(store.getValueMutable(),
                                       maybeDeadEnsures);
      }
      if (auto maskedStore = dyn_cast<VMIMaskedStoreOp>(op)) {
        tryFoldEnsureLayoutIntoMaskedStore(maskedStore, maybeDeadEnsures,
                                           maybeDeadMaskEnsures);
      }
    });

    for (VMIEnsureMaskLayoutOp ensure : llvm::reverse(maybeDeadMaskEnsures)) {
      if (ensure->use_empty()) {
        ensure.erase();
      }
    }
    for (VMIEnsureLayoutOp ensure : llvm::reverse(maybeDeadEnsures)) {
      if (ensure->use_empty()) {
        ensure.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutFoldPass() {
  return std::make_unique<VMILayoutFoldPass>();
}
