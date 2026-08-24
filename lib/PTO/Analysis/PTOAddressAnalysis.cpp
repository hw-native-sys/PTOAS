// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAddressAnalysis.cpp - Typed VPTO address analysis --------------===//

#include "PTO/Analysis/PTOAddressAnalysis.h"

#include "PTO/IR/PTO.h"
#include "mlir/Pass/AnalysisManager.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static PTOAnalysisResult<PTOTypedExprRef>
scaleExpression(const PTOTypedExprRef &expression, int64_t numerator,
                int64_t denominator) {
  if (numerator <= 0 || denominator <= 0) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::UnknownUnitSize);
  }
  PTOTypedExprRef scaled = expression;
  if (numerator != 1) {
    scaled = makePTOMulExpr(
        scaled, makePTOConstantExpr(numerator, scaled ? scaled->type : Type()),
        scaled ? scaled->type : Type());
  }
  if (denominator == 1) {
    return PTOAnalysisResult<PTOTypedExprRef>::known(std::move(scaled));
  }
  auto linear = normalizePTOLinearExpr(scaled);
  if (!linear || !dividePTOLinearExprExact(*linear, denominator)) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::InexactUnitConversion);
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(
      buildPTOTypedExpr(*linear, scaled ? scaled->type : Type()));
}

static bool isZero(const PTOTypedExprRef &expression) {
  if (auto linear = normalizePTOLinearExpr(expression)) {
    return isZeroPTOLinearExpr(*linear);
  }
  auto constant = foldPTOConstant(expression);
  return constant && *constant == 0;
}

} // namespace

PTOAddressAnalysis::PTOAddressAnalysis(func::FuncOp func,
                                       AnalysisManager &analysisManager)
    : func(func),
      valueEvolution(
          analysisManager
              .getAnalysis<PTOValueEvolutionAnalysis, func::FuncOp>()) {}

PTOAddressAnalysis::PTOAddressAnalysis(Operation *operation,
                                       AnalysisManager &analysisManager)
    : PTOAddressAnalysis(cast<func::FuncOp>(operation), analysisManager) {}

PTOAddressAnalysis::PTOAddressAnalysis(
    func::FuncOp func, PTOValueEvolutionAnalysis &valueEvolution)
    : func(func), valueEvolution(valueEvolution) {}

PTOAnalysisResult<SmallVector<PTOAddressExpr>>
PTOAddressAnalysis::getAddresses(Operation *operation) {
  auto semantics = dyn_cast<VPTOAddressSemanticsOpInterface>(operation);
  if (!semantics) {
    return PTOAnalysisResult<SmallVector<PTOAddressExpr>>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }

  SmallVector<PTOAddressExpr> addresses;
  VPTOAddressSemantics contract = semantics.getVPTOAddressSemantics();
  for (const VPTOAddressAccess &access : contract.currentAccesses) {
    Value base = access.baseOperand->get();
    auto elementBytes = getVPTOAddressUnitBytes(
        operation, VPTOAddressUnit::Element, base);
    if (!elementBytes) {
      return PTOAnalysisResult<SmallVector<PTOAddressExpr>>::unknown(
          PTOAnalysisUnknownReason::UnknownElementSize);
    }

    PTOAddressExpr address;
    address.currentBase = base;
    address.rootOrBase = base;
    address.elementBytes = *elementBytes;
    address.elementOffset = makePTOConstantExpr(0);

    while (auto addPointer =
               address.rootOrBase.getDefiningOp<AddPtrOp>()) {
      auto parentElementBytes = getVPTOAddressUnitBytes(
          operation, VPTOAddressUnit::Element, addPointer.getPtr());
      if (!parentElementBytes || *parentElementBytes != *elementBytes) {
        break;
      }
      address.elementOffset = makePTOAddExpr(
          address.elementOffset,
          valueEvolution.getExpr(addPointer.getOffset()),
          addPointer.getOffset().getType());
      address.rootOrBase = addPointer.getPtr();
    }

    if (access.offset) {
      auto unitBytes = getVPTOAddressUnitBytes(
          operation, access.offset->unit, access.offset->elementTypeSource);
      Value offset = access.offset->operand->get();
      address.offset = PTOTypedAddressOffset{
          offset, valueEvolution.getExpr(offset), access.offset->unit,
          unitBytes};
    }
    addresses.push_back(std::move(address));
  }
  return PTOAnalysisResult<SmallVector<PTOAddressExpr>>::known(
      std::move(addresses));
}

PTOAnalysisResult<PTOTypedExprRef>
PTOAddressAnalysis::getPointerDelta(Value pointer, scf::ForOp loop) {
  if (loop.isDefinedOutsideOfLoop(pointer)) {
    return PTOAnalysisResult<PTOTypedExprRef>::known(
        makePTOConstantExpr(0));
  }

  if (auto addPointer = pointer.getDefiningOp<AddPtrOp>()) {
    auto baseDelta = getPointerDelta(addPointer.getPtr(), loop);
    auto offsetEvolution =
        valueEvolution.getEvolution(addPointer.getOffset(), loop);
    if (!baseDelta || !offsetEvolution) {
      return PTOAnalysisResult<PTOTypedExprRef>::unknown(
          !baseDelta ? baseDelta.reason : offsetEvolution.reason);
    }
    return PTOAnalysisResult<PTOTypedExprRef>::known(makePTOAddExpr(
        *baseDelta.value, offsetEvolution.value->step,
        offsetEvolution.value->step->type));
  }

  auto iterArg = dyn_cast<BlockArgument>(pointer);
  if (
      !iterArg || iterArg.getOwner() != loop.getBody() ||
      iterArg.getArgNumber() == 0) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }

  unsigned index = iterArg.getArgNumber() - 1;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value yielded = yield.getOperand(index);
  PTOTypedExprRef increment = makePTOConstantExpr(0);
  while (auto addPointer = yielded.getDefiningOp<AddPtrOp>()) {
    increment = makePTOAddExpr(
        increment, valueEvolution.getExpr(addPointer.getOffset()),
        addPointer.getOffset().getType());
    yielded = addPointer.getPtr();
  }
  if (yielded != iterArg) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::NonAffineRecurrence);
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(std::move(increment));
}

PTOAnalysisResult<PTOTypedExprRef>
PTOAddressAnalysis::getDeltaBytes(const PTOAddressExpr &address,
                                  scf::ForOp loop) {
  auto rootDelta = getPointerDelta(address.rootOrBase, loop);
  PTOTypedExprRef elementDelta;
  if (isZero(address.elementOffset)) {
    elementDelta = makePTOConstantExpr(0);
  } else {
    auto evolution =
        valueEvolution.getEvolution(address.elementOffset, loop);
    if (!evolution) {
      return PTOAnalysisResult<PTOTypedExprRef>::unknown(evolution.reason);
    }
    elementDelta = evolution.value->step;
  }

  if (!rootDelta) {
    return rootDelta;
  }
  PTOTypedExprRef pointerElementDelta =
      makePTOAddExpr(*rootDelta.value, elementDelta,
                     elementDelta ? elementDelta->type : Type());
  auto pointerBytes =
      scaleExpression(pointerElementDelta, address.elementBytes, 1);
  if (!pointerBytes) {
    return pointerBytes;
  }

  PTOTypedExprRef offsetBytes = makePTOConstantExpr(0);
  if (address.offset) {
    if (!address.offset->unitBytes) {
      return PTOAnalysisResult<PTOTypedExprRef>::unknown(
          PTOAnalysisUnknownReason::UnknownUnitSize);
    }
    auto offsetEvolution =
        valueEvolution.getEvolution(address.offset->sourceValue, loop);
    if (!offsetEvolution) {
      return PTOAnalysisResult<PTOTypedExprRef>::unknown(
          offsetEvolution.reason);
    }
    auto scaled = scaleExpression(offsetEvolution.value->step,
                                  *address.offset->unitBytes, 1);
    if (!scaled) {
      return scaled;
    }
    offsetBytes = *scaled.value;
  }

  PTOTypedExprRef result =
      makePTOAddExpr(*pointerBytes.value, offsetBytes,
                     offsetBytes ? offsetBytes->type : Type());
  if (isZero(result)) {
    return PTOAnalysisResult<PTOTypedExprRef>::known(
        makePTOConstantExpr(0, result ? result->type : Type()));
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(std::move(result));
}

PTOAnalysisResult<PTOTypedExprRef>
PTOAddressAnalysis::convertDeltaToUnit(const PTOTypedExprRef &deltaBytes,
                                       int64_t targetUnitBytes) {
  return scaleExpression(deltaBytes, 1, targetUnitBytes);
}

PTOAnalysisResult<PTOTypedExprRef>
PTOAddressAnalysis::getDeltaInUnit(const PTOAddressExpr &address,
                                   scf::ForOp loop,
                                   int64_t targetUnitBytes) {
  auto bytes = getDeltaBytes(address, loop);
  if (!bytes) {
    return bytes;
  }
  return convertDeltaToUnit(*bytes.value, targetUnitBytes);
}

PTOAnalysisResult<PTOTypedExprRef>
PTOAddressAnalysis::getPointerDifference(const PTOAddressExpr &from,
                                         const PTOAddressExpr &to) {
  if (from.rootOrBase != to.rootOrBase ||
      from.elementBytes != to.elementBytes) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::DifferentBase);
  }
  auto fromOffset = valueEvolution.getPointExpression(from.elementOffset);
  auto toOffset = valueEvolution.getPointExpression(to.elementOffset);
  if (!fromOffset || !toOffset) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        !fromOffset ? fromOffset.reason : toOffset.reason);
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(
      makePTOSubExpr(*toOffset.value, *fromOffset.value,
                     (*toOffset.value)->type));
}

PTOAnalysisResult<PTOTypedExprRef>
PTOAddressAnalysis::getDifferenceBytes(const PTOAddressExpr &from,
                                       const PTOAddressExpr &to) {
  auto pointerDifference = getPointerDifference(from, to);
  if (!pointerDifference) {
    return pointerDifference;
  }
  auto pointerBytes =
      scaleExpression(*pointerDifference.value, to.elementBytes, 1);
  if (!pointerBytes) {
    return pointerBytes;
  }

  PTOTypedExprRef fromOffsetBytes = makePTOConstantExpr(0);
  if (from.offset) {
    if (!from.offset->unitBytes) {
      return PTOAnalysisResult<PTOTypedExprRef>::unknown(
          PTOAnalysisUnknownReason::UnknownUnitSize);
    }
    auto pointValue =
        valueEvolution.getPointExpression(from.offset->value);
    if (!pointValue) {
      return pointValue;
    }
    auto scaled = scaleExpression(*pointValue.value,
                                  *from.offset->unitBytes, 1);
    if (!scaled) {
      return scaled;
    }
    fromOffsetBytes = *scaled.value;
  }

  PTOTypedExprRef toOffsetBytes = makePTOConstantExpr(0);
  if (to.offset) {
    if (!to.offset->unitBytes) {
      return PTOAnalysisResult<PTOTypedExprRef>::unknown(
          PTOAnalysisUnknownReason::UnknownUnitSize);
    }
    auto pointValue = valueEvolution.getPointExpression(to.offset->value);
    if (!pointValue) {
      return pointValue;
    }
    auto scaled =
        scaleExpression(*pointValue.value, *to.offset->unitBytes, 1);
    if (!scaled) {
      return scaled;
    }
    toOffsetBytes = *scaled.value;
  }

  PTOTypedExprRef offsetDifference =
      makePTOSubExpr(toOffsetBytes, fromOffsetBytes,
                     toOffsetBytes ? toOffsetBytes->type : Type());
  PTOTypedExprRef result =
      makePTOAddExpr(*pointerBytes.value, offsetDifference,
                     offsetDifference ? offsetDifference->type : Type());
  if (isZero(result)) {
    return PTOAnalysisResult<PTOTypedExprRef>::known(
        makePTOConstantExpr(0, result ? result->type : Type()));
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(std::move(result));
}
