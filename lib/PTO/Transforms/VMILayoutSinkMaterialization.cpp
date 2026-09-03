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
#include "llvm/ADT/TypeSwitch.h"

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

struct TernaryVRegOperands {
  OpOperand *lhs = nullptr;
  OpOperand *rhs = nullptr;
  OpOperand *acc = nullptr;
};

struct SelectOperands {
  OpOperand *mask = nullptr;
  OpOperand *trueValue = nullptr;
  OpOperand *falseValue = nullptr;
};

struct UnaryVRegOperand {
  OpOperand *source = nullptr;
};

struct BinaryMaskOperands {
  OpOperand *lhs = nullptr;
  OpOperand *rhs = nullptr;
};

struct UnaryMaskOperand {
  OpOperand *source = nullptr;
};

static std::optional<BinaryVRegOperands>
getSinkableBinaryOperands(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<BinaryVRegOperands>>(op)
      .Case<VMIAddFOp, VMIAddIOp, VMISubFOp, VMISubIOp, VMIMulFOp, VMIMulIOp,
            VMIDivFOp, VMIMinFOp, VMIMinIOp, VMIMaxFOp, VMIMaxIOp, VMIAndIOp,
            VMIOrIOp, VMIXOrIOp, VMIShLIOp, VMIShRUIOp, VMIShRSIOp>(
          [](auto typedOp) {
            return BinaryVRegOperands{&typedOp.getLhsMutable(),
                                      &typedOp.getRhsMutable()};
          })
      .Default([](Operation *) { return std::nullopt; });
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

static std::optional<TernaryVRegOperands>
getSinkableTernaryOperands(Operation *op) {
  if (auto fma = dyn_cast<VMIFmaOp>(op)) {
    return TernaryVRegOperands{&fma.getLhsMutable(), &fma.getRhsMutable(),
                               &fma.getAccMutable()};
  }
  return std::nullopt;
}

static std::optional<UnaryVRegOperand> getSinkableUnaryOperand(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<UnaryVRegOperand>>(op)
      .Case<VMINegFOp, VMINegIOp, VMIAbsFOp, VMIAbsIOp, VMISqrtOp, VMIExpOp,
            VMILnOp, VMIReluOp, VMINotOp>([](auto typedOp) {
        return UnaryVRegOperand{&typedOp.getSourceMutable()};
      }).Default([](Operation *) { return std::nullopt; });
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

static bool isSameMaterialization(VMIEnsureLayoutOp lhsEnsure,
                                  VMIEnsureLayoutOp rhsEnsure,
                                  VMIVRegType resultType) {
  if (!lhsEnsure || !rhsEnsure || !resultType) {
    return false;
  }

  auto lhsSourceType = dyn_cast<VMIVRegType>(lhsEnsure.getSource().getType());
  auto rhsSourceType = dyn_cast<VMIVRegType>(rhsEnsure.getSource().getType());
  auto lhsResultType = dyn_cast<VMIVRegType>(lhsEnsure.getResult().getType());
  auto rhsResultType = dyn_cast<VMIVRegType>(rhsEnsure.getResult().getType());
  if (!lhsSourceType || !rhsSourceType || !lhsResultType || !rhsResultType) {
    return false;
  }

  return lhsSourceType == rhsSourceType && lhsResultType == rhsResultType &&
         lhsResultType == resultType && lhsSourceType != resultType;
}

static bool isSameMaterialization(VMIEnsureLayoutOp lhsEnsure,
                                  VMIEnsureLayoutOp rhsEnsure,
                                  VMIEnsureLayoutOp accEnsure,
                                  VMIVRegType resultType) {
  if (!lhsEnsure || !rhsEnsure || !accEnsure || !resultType) {
    return false;
  }

  auto lhsSourceType = dyn_cast<VMIVRegType>(lhsEnsure.getSource().getType());
  auto rhsSourceType = dyn_cast<VMIVRegType>(rhsEnsure.getSource().getType());
  auto accSourceType = dyn_cast<VMIVRegType>(accEnsure.getSource().getType());
  auto lhsResultType = dyn_cast<VMIVRegType>(lhsEnsure.getResult().getType());
  auto rhsResultType = dyn_cast<VMIVRegType>(rhsEnsure.getResult().getType());
  auto accResultType = dyn_cast<VMIVRegType>(accEnsure.getResult().getType());
  if (!lhsSourceType || !rhsSourceType || !accSourceType || !lhsResultType ||
      !rhsResultType || !accResultType) {
    return false;
  }

  return lhsSourceType == rhsSourceType && lhsSourceType == accSourceType &&
         lhsResultType == rhsResultType && lhsResultType == accResultType &&
         lhsResultType == resultType && lhsSourceType != resultType;
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

static bool trySinkBinaryMaterialization(Operation *op) {
  std::optional<BinaryVRegOperands> operands = getSinkableBinaryOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto lhsEnsure = operands->lhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto rhsEnsure = operands->rhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!isSameMaterialization(lhsEnsure, rhsEnsure, resultType)) {
    return false;
  }

  auto sourceType = cast<VMIVRegType>(lhsEnsure.getSource().getType());
  if (!hasEnsureLayoutSupport(sourceType, resultType)) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({lhsEnsure.getSource(), rhsEnsure.getSource()});
  state.addTypes(sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureLayoutOp>(
      op->getLoc(), resultType, newOp->getResult(0));
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

struct SelectMaterialization {
  VMIEnsureMaskLayoutOp maskEnsure;
  VMIEnsureLayoutOp trueEnsure;
  VMIEnsureLayoutOp falseEnsure;
  VMIVRegType sourceType;
  VMIVRegType resultType;
};

static bool isCompatibleSelectMask(VMIEnsureMaskLayoutOp maskEnsure,
                                   Type operandType, VMIVRegType sourceType,
                                   VMIVRegType resultType) {
  auto maskSourceType = dyn_cast<VMIMaskType>(maskEnsure.getSource().getType());
  auto maskResultType = dyn_cast<VMIMaskType>(maskEnsure.getResult().getType());
  if (!maskSourceType || !maskResultType || maskResultType != operandType) {
    return false;
  }
  bool hasMismatchedLayout =
      maskResultType.getLayoutAttr() != resultType.getLayoutAttr() ||
      maskSourceType.getLayoutAttr() != sourceType.getLayoutAttr();
  if (hasMismatchedLayout) {
    return false;
  }
  bool hasMismatchedShape =
      maskSourceType.getElementCount() != sourceType.getElementCount() ||
      maskResultType.getElementCount() != resultType.getElementCount() ||
      maskSourceType.getGranularity() != maskResultType.getGranularity();
  if (hasMismatchedShape) {
    return false;
  }
  return hasEnsureMaskSupport(maskEnsure, maskSourceType, maskResultType);
}

static std::optional<SelectMaterialization>
getSelectMaterialization(Operation *op) {
  std::optional<SelectOperands> operands = getSinkableSelectOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return std::nullopt;
  }

  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!resultType) {
    return std::nullopt;
  }

  auto maskEnsure =
      operands->mask->get().getDefiningOp<VMIEnsureMaskLayoutOp>();
  auto trueEnsure =
      operands->trueValue->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto falseEnsure =
      operands->falseValue->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!maskEnsure || !trueEnsure || !falseEnsure) {
    return std::nullopt;
  }

  auto trueSourceType = dyn_cast<VMIVRegType>(trueEnsure.getSource().getType());
  if (!trueSourceType ||
      !isSameMaterialization(trueEnsure, falseEnsure, resultType)) {
    return std::nullopt;
  }
  if (!hasEnsureLayoutSupport(trueSourceType, resultType) ||
      !isCompatibleSelectMask(maskEnsure, operands->mask->get().getType(),
                              trueSourceType, resultType)) {
    return std::nullopt;
  }
  return SelectMaterialization{maskEnsure, trueEnsure, falseEnsure,
                               trueSourceType, resultType};
}

static bool trySinkSelectMaterialization(Operation *op) {
  std::optional<SelectMaterialization> materialization =
      getSelectMaterialization(op);
  if (!materialization) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({materialization->maskEnsure.getSource(),
                     materialization->trueEnsure.getSource(),
                     materialization->falseEnsure.getSource()});
  state.addTypes(materialization->sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureLayoutOp>(
      op->getLoc(), materialization->resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (materialization->maskEnsure->use_empty()) {
    materialization->maskEnsure.erase();
  }
  if (materialization->trueEnsure->use_empty()) {
    materialization->trueEnsure.erase();
  }
  if (materialization->falseEnsure != materialization->trueEnsure &&
      materialization->falseEnsure->use_empty()) {
    materialization->falseEnsure.erase();
  }
  return true;
}

struct CompareMaterialization {
  VMIEnsureLayoutOp lhsEnsure;
  VMIEnsureLayoutOp rhsEnsure;
  VMIMaskType sourceType;
  VMIMaskType resultType;
};

static std::optional<CompareMaterialization>
getCompareMaterialization(Operation *op) {
  std::optional<BinaryVRegOperands> operands = getSinkableCompareOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return std::nullopt;
  }

  auto resultMaskType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
  if (!resultMaskType) {
    return std::nullopt;
  }

  auto lhsEnsure = operands->lhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto rhsEnsure = operands->rhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!lhsEnsure || !rhsEnsure) {
    return std::nullopt;
  }

  auto lhsSourceType = dyn_cast<VMIVRegType>(lhsEnsure.getSource().getType());
  auto lhsResultType = dyn_cast<VMIVRegType>(lhsEnsure.getResult().getType());
  if (!lhsSourceType || !lhsResultType ||
      !isSameMaterialization(lhsEnsure, rhsEnsure, lhsResultType)) {
    return std::nullopt;
  }
  if (lhsResultType.getElementCount() != resultMaskType.getElementCount() ||
      lhsResultType.getLayoutAttr() != resultMaskType.getLayoutAttr()) {
    return std::nullopt;
  }

  auto sourceMaskType = VMIMaskType::get(
      op->getContext(), resultMaskType.getElementCount(),
      resultMaskType.getGranularity(), lhsSourceType.getLayoutAttr());
  VMILayoutSupport supports;
  if (failed(supports.getEnsureMaskLayoutFact(sourceMaskType, resultMaskType))) {
    return std::nullopt;
  }
  return CompareMaterialization{lhsEnsure, rhsEnsure, sourceMaskType,
                                resultMaskType};
}

static bool trySinkCompareMaterialization(Operation *op) {
  std::optional<CompareMaterialization> materialization =
      getCompareMaterialization(op);
  if (!materialization) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({materialization->lhsEnsure.getSource(),
                     materialization->rhsEnsure.getSource()});
  state.addTypes(materialization->sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureMaskLayoutOp>(
      op->getLoc(), materialization->resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (materialization->lhsEnsure->use_empty()) {
    materialization->lhsEnsure.erase();
  }
  if (materialization->rhsEnsure != materialization->lhsEnsure &&
      materialization->rhsEnsure->use_empty()) {
    materialization->rhsEnsure.erase();
  }
  return true;
}

static bool trySinkTernaryMaterialization(Operation *op) {
  std::optional<TernaryVRegOperands> operands = getSinkableTernaryOperands(op);
  if (!operands || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto lhsEnsure = operands->lhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto rhsEnsure = operands->rhs->get().getDefiningOp<VMIEnsureLayoutOp>();
  auto accEnsure = operands->acc->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!isSameMaterialization(lhsEnsure, rhsEnsure, accEnsure, resultType)) {
    return false;
  }

  auto sourceType = cast<VMIVRegType>(lhsEnsure.getSource().getType());
  if (!hasEnsureLayoutSupport(sourceType, resultType)) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(
      {lhsEnsure.getSource(), rhsEnsure.getSource(), accEnsure.getSource()});
  state.addTypes(sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureLayoutOp>(
      op->getLoc(), resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (lhsEnsure->use_empty()) {
    lhsEnsure.erase();
  }
  if (rhsEnsure != lhsEnsure && rhsEnsure->use_empty()) {
    rhsEnsure.erase();
  }
  if (accEnsure != lhsEnsure && accEnsure != rhsEnsure &&
      accEnsure->use_empty()) {
    accEnsure.erase();
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

static bool trySinkUnaryMaterialization(Operation *op) {
  std::optional<UnaryVRegOperand> operand = getSinkableUnaryOperand(op);
  if (!operand || op->getNumResults() != 1) {
    return false;
  }

  auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }

  auto sourceEnsure = operand->source->get().getDefiningOp<VMIEnsureLayoutOp>();
  if (!isSameMaterialization(sourceEnsure, resultType)) {
    return false;
  }

  auto sourceType = cast<VMIVRegType>(sourceEnsure.getSource().getType());
  if (!hasEnsureLayoutSupport(sourceType, resultType)) {
    return false;
  }

  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(sourceEnsure.getSource());
  state.addTypes(sourceType);
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  builder.setInsertionPointAfter(newOp);
  auto resultEnsure = builder.create<VMIEnsureLayoutOp>(
      op->getLoc(), resultType, newOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(resultEnsure.getResult());
  op->erase();

  if (sourceEnsure->use_empty()) {
    sourceEnsure.erase();
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

static bool isSinkCandidate(Operation *op) {
  return getSinkableBinaryOperands(op) || getSinkableCompareOperands(op) ||
         getSinkableSelectOperands(op) || getSinkableTernaryOperands(op) ||
         getSinkableUnaryOperand(op) || getSinkableBinaryMaskOperands(op) ||
         getSinkableUnaryMaskOperand(op);
}

static void trySinkMaterialization(Operation *op) {
  bool materialized =
      trySinkBinaryMaterialization(op) || trySinkCompareMaterialization(op) ||
      trySinkSelectMaterialization(op) || trySinkTernaryMaterialization(op) ||
      trySinkUnaryMaterialization(op);
  if (materialized) {
    return;
  }
  (void)trySinkMaskMaterialization(op);
}

struct VMILayoutSinkMaterializationPass
    : public mlir::pto::impl::VMILayoutSinkMaterializationBase<
          VMILayoutSinkMaterializationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutSinkMaterializationPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isSinkCandidate(op)) {
        candidates.push_back(op);
      }
    });

    for (Operation *op : candidates) {
      if (op->getBlock() == nullptr) {
        continue;
      }
      trySinkMaterialization(op);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutSinkMaterializationPass() {
  return std::make_unique<VMILayoutSinkMaterializationPass>();
}
