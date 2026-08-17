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
  if (auto addf = dyn_cast<VMIAddFOp>(op)) {
    return BinaryVRegOperands{&addf.getLhsMutable(), &addf.getRhsMutable()};
  }
  if (auto addi = dyn_cast<VMIAddIOp>(op)) {
    return BinaryVRegOperands{&addi.getLhsMutable(), &addi.getRhsMutable()};
  }
  if (auto subf = dyn_cast<VMISubFOp>(op)) {
    return BinaryVRegOperands{&subf.getLhsMutable(), &subf.getRhsMutable()};
  }
  if (auto subi = dyn_cast<VMISubIOp>(op)) {
    return BinaryVRegOperands{&subi.getLhsMutable(), &subi.getRhsMutable()};
  }
  if (auto mulf = dyn_cast<VMIMulFOp>(op)) {
    return BinaryVRegOperands{&mulf.getLhsMutable(), &mulf.getRhsMutable()};
  }
  if (auto muli = dyn_cast<VMIMulIOp>(op)) {
    return BinaryVRegOperands{&muli.getLhsMutable(), &muli.getRhsMutable()};
  }
  if (auto divf = dyn_cast<VMIDivFOp>(op)) {
    return BinaryVRegOperands{&divf.getLhsMutable(), &divf.getRhsMutable()};
  }
  if (auto minf = dyn_cast<VMIMinFOp>(op)) {
    return BinaryVRegOperands{&minf.getLhsMutable(), &minf.getRhsMutable()};
  }
  if (auto mini = dyn_cast<VMIMinIOp>(op)) {
    return BinaryVRegOperands{&mini.getLhsMutable(), &mini.getRhsMutable()};
  }
  if (auto maxf = dyn_cast<VMIMaxFOp>(op)) {
    return BinaryVRegOperands{&maxf.getLhsMutable(), &maxf.getRhsMutable()};
  }
  if (auto maxi = dyn_cast<VMIMaxIOp>(op)) {
    return BinaryVRegOperands{&maxi.getLhsMutable(), &maxi.getRhsMutable()};
  }
  if (auto andi = dyn_cast<VMIAndIOp>(op)) {
    return BinaryVRegOperands{&andi.getLhsMutable(), &andi.getRhsMutable()};
  }
  if (auto ori = dyn_cast<VMIOrIOp>(op)) {
    return BinaryVRegOperands{&ori.getLhsMutable(), &ori.getRhsMutable()};
  }
  if (auto xori = dyn_cast<VMIXOrIOp>(op)) {
    return BinaryVRegOperands{&xori.getLhsMutable(), &xori.getRhsMutable()};
  }
  if (auto shli = dyn_cast<VMIShLIOp>(op)) {
    return BinaryVRegOperands{&shli.getLhsMutable(), &shli.getRhsMutable()};
  }
  if (auto shrui = dyn_cast<VMIShRUIOp>(op)) {
    return BinaryVRegOperands{&shrui.getLhsMutable(), &shrui.getRhsMutable()};
  }
  if (auto shrsi = dyn_cast<VMIShRSIOp>(op)) {
    return BinaryVRegOperands{&shrsi.getLhsMutable(), &shrsi.getRhsMutable()};
  }
  return std::nullopt;
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
  if (auto negf = dyn_cast<VMINegFOp>(op)) {
    return UnaryVRegOperand{&negf.getSourceMutable()};
  if (auto negi = dyn_cast<VMINegIOp>(op)) {
    return UnaryVRegOperand{&negi.getSourceMutable()};
  }
  if (auto absf = dyn_cast<VMIAbsFOp>(op))
    return UnaryVRegOperand{&absf.getSourceMutable()};
  }
  if (auto absi = dyn_cast<VMIAbsIOp>(op)) {
    return UnaryVRegOperand{&absi.getSourceMutable()};
  }
  if (auto sqrt = dyn_cast<VMISqrtOp>(op)) {
    return UnaryVRegOperand{&sqrt.getSourceMutable()};
  }
  if (auto exp = dyn_cast<VMIExpOp>(op)) {
    return UnaryVRegOperand{&exp.getSourceMutable()};
  }
  if (auto ln = dyn_cast<VMILnOp>(op)) {
    return UnaryVRegOperand{&ln.getSourceMutable()};
  }
  if (auto relu = dyn_cast<VMIReluOp>(op)) {
    return UnaryVRegOperand{&relu.getSourceMutable()};
  }
  if (auto notOp = dyn_cast<VMINotOp>(op)) {
    return UnaryVRegOperand{&notOp.getSourceMutable()};
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

struct VMILayoutSinkMaterializationPass
    : public mlir::pto::impl::VMILayoutSinkMaterializationBase<
          VMILayoutSinkMaterializationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutSinkMaterializationPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (getSinkableBinaryOperands(op) || getSinkableCompareOperands(op) ||
          getSinkableSelectOperands(op) || getSinkableTernaryOperands(op) ||
          getSinkableUnaryOperand(op) || getSinkableBinaryMaskOperands(op) ||
          getSinkableUnaryMaskOperand(op)) {
        candidates.push_back(op);
      }
    });

    for (Operation *op : candidates) {
      if (op->getBlock() == nullptr) {
        continue;
      }
      if (!trySinkBinaryMaterialization(op)) {
        if (!trySinkCompareMaterialization(op)) {
          if (!trySinkSelectMaterialization(op)) {
            if (!trySinkTernaryMaterialization(op)) {
              if (!trySinkUnaryMaterialization(op)) {
                trySinkMaskMaterialization(op);
              }
            }
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutSinkMaterializationPass() {
  return std::make_unique<VMILayoutSinkMaterializationPass>();
}
