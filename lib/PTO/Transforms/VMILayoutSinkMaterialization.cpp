// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSinkMaterialization.cpp - Sink VMI layout helpers --------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTSINKMATERIALIZATION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct BinaryVRegOperands {
  OpOperand *lhs = nullptr;
  OpOperand *rhs = nullptr;
};

struct SelectOperands {
  OpOperand *mask = nullptr;
  OpOperand *trueValue = nullptr;
  OpOperand *falseValue = nullptr;
};

struct BinaryMaskOperands {
  OpOperand *lhs = nullptr;
  OpOperand *rhs = nullptr;
};

struct UnaryMaskOperand {
  OpOperand *source = nullptr;
};

static bool isSinkableElementwiseOp(Operation *op) {
  return isa<VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp, VMIVmaxOp,
             VMIVandOp, VMIVorOp, VMIVxorOp, VMIVshlOp, VMIVshrOp, VMIVnegOp,
             VMIVabsOp, VMIVsqrtOp, VMIVexpOp, VMIVlnOp, VMIVreluOp, VMIVnotOp,
             VMIVmulaOp>(op);
}

static std::optional<BinaryVRegOperands>
getSinkableCompareOperands(Operation *op) {
  if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
    return BinaryVRegOperands{&cmpf.getLhsMutable(), &cmpf.getRhsMutable()};
  }
  if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
    return BinaryVRegOperands{&cmpi.getLhsMutable(), &cmpi.getRhsMutable()};
  }
  return std::nullopt;
}

static std::optional<SelectOperands> getSinkableSelectOperands(Operation *op) {
  if (auto select = dyn_cast<VMISelectOp>(op)) {
    return SelectOperands{&select.getMaskMutable(),
                          &select.getTrueValueMutable(),
                          &select.getFalseValueMutable()};
  }
  return std::nullopt;
}

static std::optional<BinaryMaskOperands>
getSinkableBinaryMaskOperands(Operation *op) {
  if (auto maskAnd = dyn_cast<VMIMaskAndOp>(op)) {
    return BinaryMaskOperands{&maskAnd.getLhsMutable(),
                              &maskAnd.getRhsMutable()};
  }
  if (auto maskOr = dyn_cast<VMIMaskOrOp>(op)) {
    return BinaryMaskOperands{&maskOr.getLhsMutable(), &maskOr.getRhsMutable()};
  }
  if (auto maskXor = dyn_cast<VMIMaskXOrOp>(op)) {
    return BinaryMaskOperands{&maskXor.getLhsMutable(),
                              &maskXor.getRhsMutable()};
  }
  return std::nullopt;
}

static std::optional<UnaryMaskOperand>
getSinkableUnaryMaskOperand(Operation *op) {
  if (auto maskNot = dyn_cast<VMIMaskNotOp>(op)) {
    return UnaryMaskOperand{&maskNot.getSourceMutable()};
  }
  return std::nullopt;
}

static bool isSameMaterialization(VMIEnsureLayoutOp ensure,
                                  VMIVRegType resultType) {
  if (!ensure || !resultType) {
    return false;
  }

  auto sourceType = dyn_cast<VMIVRegType>(ensure.getSource().getType());
  auto ensureResultType = dyn_cast<VMIVRegType>(ensure.getResult().getType());
  if (!sourceType || !ensureResultType) {
    return false;
  }

  return ensureResultType == resultType && sourceType != resultType;
}

static bool hasEnsureLayoutSupport(VMIVRegType sourceType,
                                   VMIVRegType resultType) {
  VMILayoutSupport supports;
  return succeeded(supports.getEnsureLayoutFact(sourceType, resultType));
}

template <typename EnsureOp>
static bool isSameMaskMaterialization(EnsureOp ensure, VMIMaskType resultType) {
  if (!ensure || !resultType) {
    return false;
  }

  auto sourceType = dyn_cast<VMIMaskType>(ensure.getSource().getType());
  auto ensureResultType = dyn_cast<VMIMaskType>(ensure.getResult().getType());
  if (!sourceType || !ensureResultType) {
    return false;
  }

  return ensureResultType == resultType && sourceType != resultType;
}

template <typename EnsureOp>
static bool isSameMaskMaterialization(EnsureOp lhsEnsure, EnsureOp rhsEnsure,
                                      VMIMaskType resultType) {
  if (!lhsEnsure || !rhsEnsure || !resultType) {
    return false;
  }

  auto lhsSourceType = dyn_cast<VMIMaskType>(lhsEnsure.getSource().getType());
  auto rhsSourceType = dyn_cast<VMIMaskType>(rhsEnsure.getSource().getType());
  auto lhsResultType = dyn_cast<VMIMaskType>(lhsEnsure.getResult().getType());
  auto rhsResultType = dyn_cast<VMIMaskType>(rhsEnsure.getResult().getType());
  if (!lhsSourceType || !rhsSourceType || !lhsResultType || !rhsResultType) {
    return false;
  }

  return lhsSourceType == rhsSourceType && lhsResultType == rhsResultType &&
         lhsResultType == resultType && lhsSourceType != resultType;
}

static bool hasEnsureMaskSupport(VMIEnsureMaskLayoutOp, VMIMaskType sourceType,
                                 VMIMaskType resultType) {
  VMILayoutSupport supports;
  return succeeded(supports.getEnsureMaskLayoutFact(sourceType, resultType));
}

static bool hasEnsureMaskSupport(VMIEnsureMaskGranularityOp,
                                 VMIMaskType sourceType,
                                 VMIMaskType resultType) {
  return sourceType.getElementCount() == resultType.getElementCount() &&
         sourceType.getLayoutAttr() == resultType.getLayoutAttr() &&
         !sourceType.isPred() && !resultType.isPred();
}

static Value rematerializeMask(Value mask, VMILayoutAttr layout,
                               OpBuilder &builder) {
  auto maskType = dyn_cast<VMIMaskType>(mask.getType());
  if (!maskType || !layout) {
    return {};
  }
  auto resultType = VMIMaskType::get(
      maskType.getContext(), maskType.getElementCount(),
      maskType.getGranularity(), layout);
  if (auto createMask = mask.getDefiningOp<VMICreateMaskOp>()) {
    return builder
        .create<VMICreateMaskOp>(createMask.getLoc(), resultType,
                                 createMask.getActiveLanes())
        .getResult();
  }
  if (auto createGroupMask = mask.getDefiningOp<VMICreateGroupMaskOp>()) {
    return builder
        .create<VMICreateGroupMaskOp>(
            createGroupMask.getLoc(), resultType,
            createGroupMask.getActiveElemsPerGroup(),
            createGroupMask.getNumGroupsAttr(),
            createGroupMask.getGroupSizeAttr())
        .getResult();
  }
  if (auto constantMask = mask.getDefiningOp<VMIConstantMaskOp>()) {
    return builder
        .create<VMIConstantMaskOp>(constantMask.getLoc(), resultType,
                                   constantMask.getValueAttr())
        .getResult();
  }
  return {};
}

static bool trySinkElementwiseMaterialization(Operation *op) {
  if (!isSinkableElementwiseOp(op) || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  SmallVector<VMIEnsureLayoutOp> dataEnsures;
  VMIEnsureMaskLayoutOp maskEnsure;
  Value directMask;
  for (Value operand : op->getOperands()) {
    if (isa<VMIVRegType>(operand.getType())) {
      auto ensure = operand.getDefiningOp<VMIEnsureLayoutOp>();
      if (!ensure || !isSameMaterialization(ensure, resultType)) {
        return false;
      }
      if (llvm::none_of(dataEnsures, [ensure](VMIEnsureLayoutOp existing) {
            return existing == ensure;
          })) {
        dataEnsures.push_back(ensure);
      }
      continue;
    }
    if (isa<VMIMaskType>(operand.getType())) {
      maskEnsure = operand.getDefiningOp<VMIEnsureMaskLayoutOp>();
      if (!maskEnsure) {
        directMask = operand;
      }
    }
  }
  if (dataEnsures.empty()) {
    return false;
  }

  auto sourceType =
      cast<VMIVRegType>(dataEnsures.front().getSource().getType());
  if (llvm::any_of(dataEnsures, [&](VMIEnsureLayoutOp ensure) {
        return ensure.getSource().getType() != sourceType;
      })) {
    return false;
  }
  if (!hasEnsureLayoutSupport(sourceType, resultType)) {
    return false;
  }

  if (maskEnsure) {
    auto maskSourceType =
        dyn_cast<VMIMaskType>(maskEnsure.getSource().getType());
    auto maskResultType =
        dyn_cast<VMIMaskType>(maskEnsure.getResult().getType());
    if (!maskSourceType || !maskResultType ||
        maskSourceType.getLayoutAttr() != sourceType.getLayoutAttr() ||
        maskResultType.getLayoutAttr() != resultType.getLayoutAttr() ||
        !hasEnsureMaskSupport(maskEnsure, maskSourceType, maskResultType)) {
      return false;
    }
  } else if (directMask &&
             !isa<VMICreateMaskOp, VMICreateGroupMaskOp, VMIConstantMaskOp>(
                 directMask.getDefiningOp())) {
    return false;
  }

  OpBuilder builder(op);
  SmallVector<Value> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (auto ensure = operand.getDefiningOp<VMIEnsureLayoutOp>()) {
      operands.push_back(ensure.getSource());
    } else if (auto ensure = operand.getDefiningOp<VMIEnsureMaskLayoutOp>()) {
      operands.push_back(ensure.getSource());
    } else if (isa<VMIMaskType>(operand.getType())) {
      Value rematerialized =
          rematerializeMask(operand, sourceType.getLayoutAttr(), builder);
      if (!rematerialized) {
        return false;
      }
      operands.push_back(rematerialized);
    } else {
      operands.push_back(operand);
    }
  }

  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureLayoutOp>(
      op->getLoc(), resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  for (VMIEnsureLayoutOp ensure : llvm::reverse(dataEnsures)) {
    if (ensure->use_empty()) {
      ensure.erase();
    }
  }
  if (maskEnsure && maskEnsure->use_empty()) {
    maskEnsure.erase();
  }
  return true;
}

static bool trySinkSelectMaterialization(Operation *op) {
  std::optional<SelectOperands> operands = getSinkableSelectOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto maskEnsure =
      operands->mask->get().getDefiningOp<VMIEnsureMaskLayoutOp>();
  auto trueEnsure =
      operands->trueValue->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto falseEnsure =
      operands->falseValue->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!maskEnsure || !trueEnsure || !falseEnsure) {
    return false;
  }

  auto trueSourceType = dyn_cast<VMIVRegType>(trueEnsure.getSource().getType());
  auto falseSourceType =
      dyn_cast<VMIVRegType>(falseEnsure.getSource().getType());
  auto trueResultType = dyn_cast<VMIVRegType>(trueEnsure.getResult().getType());
  auto falseResultType =
      dyn_cast<VMIVRegType>(falseEnsure.getResult().getType());
  auto maskSourceType = dyn_cast<VMIMaskType>(maskEnsure.getSource().getType());
  auto maskResultType = dyn_cast<VMIMaskType>(maskEnsure.getResult().getType());
  if (!trueSourceType || !falseSourceType || !trueResultType ||
      !falseResultType || !maskSourceType || !maskResultType) {
    return false;
  }

  if (trueSourceType != falseSourceType || trueResultType != falseResultType ||
      trueResultType != resultType || trueSourceType == resultType) {
    return false;
  }
  if (maskResultType != operands->mask->get().getType()) {
    return false;
  }
  if (maskResultType.getLayoutAttr() != resultType.getLayoutAttr() ||
      maskSourceType.getLayoutAttr() != trueSourceType.getLayoutAttr()) {
    return false;
  }
  if (maskSourceType.getElementCount() != trueSourceType.getElementCount() ||
      maskResultType.getElementCount() != resultType.getElementCount() ||
      maskSourceType.getGranularity() != maskResultType.getGranularity()) {
    return false;
  }
  if (!hasEnsureLayoutSupport(trueSourceType, resultType) ||
      !hasEnsureMaskSupport(maskEnsure, maskSourceType, maskResultType)) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({maskEnsure.getSource(), trueEnsure.getSource(),
                     falseEnsure.getSource()});
  state.addTypes(trueSourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureLayoutOp>(
      op->getLoc(), resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (maskEnsure->use_empty()) {
    maskEnsure.erase();
  }
  if (trueEnsure->use_empty()) {
    trueEnsure.erase();
  }
  if (falseEnsure != trueEnsure && falseEnsure->use_empty()) {
    falseEnsure.erase();
  }
  return true;
}

static bool trySinkCompareMaterialization(Operation *op) {
  std::optional<BinaryVRegOperands> operands = getSinkableCompareOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return false;
  }

  auto resultMaskType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
  if (!resultMaskType) {
    return false;
  }

  auto lhsEnsure = operands->lhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto rhsEnsure = operands->rhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!lhsEnsure || !rhsEnsure) {
    return false;
  }

  auto lhsSourceType = dyn_cast<VMIVRegType>(lhsEnsure.getSource().getType());
  auto rhsSourceType = dyn_cast<VMIVRegType>(rhsEnsure.getSource().getType());
  auto lhsResultType = dyn_cast<VMIVRegType>(lhsEnsure.getResult().getType());
  auto rhsResultType = dyn_cast<VMIVRegType>(rhsEnsure.getResult().getType());
  if (!lhsSourceType || !rhsSourceType || !lhsResultType || !rhsResultType) {
    return false;
  }
  if (lhsSourceType != rhsSourceType || lhsResultType != rhsResultType ||
      lhsSourceType == lhsResultType) {
    return false;
  }
  if (lhsResultType.getElementCount() != resultMaskType.getElementCount() ||
      lhsResultType.getLayoutAttr() != resultMaskType.getLayoutAttr()) {
    return false;
  }

  auto sourceMaskType = VMIMaskType::get(
      op->getContext(), resultMaskType.getElementCount(),
      resultMaskType.getGranularity(), lhsSourceType.getLayoutAttr());
  VMILayoutSupport supports;
  if (failed(supports.getEnsureMaskLayoutFact(sourceMaskType, resultMaskType))) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({lhsEnsure.getSource(), rhsEnsure.getSource()});
  state.addTypes(sourceMaskType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureMaskLayoutOp>(
      op->getLoc(), resultMaskType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (lhsEnsure->use_empty()) {
    lhsEnsure.erase();
  }
  if (rhsEnsure != lhsEnsure && rhsEnsure->use_empty()) {
    rhsEnsure.erase();
  }
  return true;
}

template <typename EnsureOp>
static bool trySinkBinaryMaskMaterialization(Operation *op) {
  std::optional<BinaryMaskOperands> operands =
      getSinkableBinaryMaskOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto lhsEnsure = operands->lhs->get().getDefiningOp<EnsureOp>();
  auto rhsEnsure = operands->rhs->get().getDefiningOp<EnsureOp>();
  if (!isSameMaskMaterialization(lhsEnsure, rhsEnsure, resultType)) {
    return false;
  }

  auto sourceType = cast<VMIMaskType>(lhsEnsure.getSource().getType());
  if (!hasEnsureMaskSupport(lhsEnsure, sourceType, resultType)) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({lhsEnsure.getSource(), rhsEnsure.getSource()});
  state.addTypes(sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure =
      builder.create<EnsureOp>(op->getLoc(), resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (lhsEnsure->use_empty()) {
    lhsEnsure.erase();
  }
  if (rhsEnsure != lhsEnsure && rhsEnsure->use_empty()) {
    rhsEnsure.erase();
  }
  return true;
}

template <typename EnsureOp>
static bool trySinkUnaryMaskMaterialization(Operation *op) {
  std::optional<UnaryMaskOperand> operand = getSinkableUnaryMaskOperand(op);
  if (!operand || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto sourceEnsure = operand->source->get().getDefiningOp<EnsureOp>();
  if (!isSameMaskMaterialization(sourceEnsure, resultType)) {
    return false;
  }

  auto sourceType = cast<VMIMaskType>(sourceEnsure.getSource().getType());
  if (!hasEnsureMaskSupport(sourceEnsure, sourceType, resultType)) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(sourceEnsure.getSource());
  state.addTypes(sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure =
      builder.create<EnsureOp>(op->getLoc(), resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (sourceEnsure->use_empty()) {
    sourceEnsure.erase();
  }
  return true;
}

static bool trySinkMaskMaterialization(Operation *op) {
  return trySinkBinaryMaskMaterialization<VMIEnsureMaskLayoutOp>(op) ||
         trySinkBinaryMaskMaterialization<VMIEnsureMaskGranularityOp>(op) ||
         trySinkUnaryMaskMaterialization<VMIEnsureMaskLayoutOp>(op) ||
         trySinkUnaryMaskMaterialization<VMIEnsureMaskGranularityOp>(op);
}

struct VMILayoutSinkMaterializationPass
    : public mlir::pto::impl::VMILayoutSinkMaterializationBase<
          VMILayoutSinkMaterializationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutSinkMaterializationPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isSinkableElementwiseOp(op) || getSinkableCompareOperands(op) ||
          getSinkableSelectOperands(op) || getSinkableBinaryMaskOperands(op) ||
          getSinkableUnaryMaskOperand(op)) {
        candidates.push_back(op);
      }
    });

    for (Operation *op : candidates) {
      if (op->getBlock() == nullptr) {
        continue;
      }
      if (trySinkElementwiseMaterialization(op) ||
          trySinkCompareMaterialization(op) ||
          trySinkSelectMaterialization(op)) {
        continue;
      }
      trySinkMaskMaterialization(op);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutSinkMaterializationPass() {
  return std::make_unique<VMILayoutSinkMaterializationPass>();
}
