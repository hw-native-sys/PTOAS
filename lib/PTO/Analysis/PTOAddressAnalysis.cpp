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
#include "PTO/Support/CodeConstants.h"
#include "mlir/Pass/AnalysisManager.h"

#include <functional>

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr int64_t kBlockSizeBytes = 32;

static std::optional<int64_t> getElementBytes(Value pointer) {
  Type elementType;
  if (auto pointerType = dyn_cast<PtrType>(pointer.getType())) {
    elementType = pointerType.getElementType();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(pointer.getType())) {
    elementType = memrefType.getElementType();
  } else {
    return std::nullopt;
  }
  if (!elementType || !elementType.isIntOrFloat()) {
    return std::nullopt;
  }
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % mlir::pto::kValue8 != 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(bitWidth / mlir::pto::kValue8);
}

static std::optional<int64_t> getUnitBytes(Operation *operation,
                                           VPTOAddressUnit unit,
                                           int64_t elementBytes) {
  switch (unit) {
  case VPTOAddressUnit::Element:
    return elementBytes;
  case VPTOAddressUnit::Block:
    return kBlockSizeBytes;
  case VPTOAddressUnit::Byte:
    return 1;
  case VPTOAddressUnit::Alignment:
    return getLoadStoreVecAlignmentSize(operation);
  }
  return std::nullopt;
}

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
  for (const VPTOAddressAccess &access : semantics.getVPTOAddressAccesses()) {
    auto elementBytes = getElementBytes(access.base);
    if (!elementBytes) {
      return PTOAnalysisResult<SmallVector<PTOAddressExpr>>::unknown(
          PTOAnalysisUnknownReason::UnknownElementSize);
    }

    PTOAddressExpr address;
    address.currentBase = access.base;
    address.rootOrBase = access.base;
    address.elementBytes = *elementBytes;
    address.elementOffset = makePTOConstantExpr(0);

    while (auto addPointer =
               address.rootOrBase.getDefiningOp<AddPtrOp>()) {
      auto parentElementBytes = getElementBytes(addPointer.getPtr());
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
      auto unitBytes =
          getUnitBytes(operation, access.offset->unit, *elementBytes);
      address.offset = PTOTypedAddressOffset{
          access.offset->value, valueEvolution.getExpr(access.offset->value),
          access.offset->unit, unitBytes};
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
    // AddressExpr offsets can be composite expressions without a backing SSA
    // value. Their loop delta is obtained by recursively evaluating leaves.
    std::function<PTOAnalysisResult<PTOTypedExprRef>(const PTOTypedExprRef &)>
        getExprDelta = [&](const PTOTypedExprRef &expr)
        -> PTOAnalysisResult<PTOTypedExprRef> {
      if (!expr) {
        return PTOAnalysisResult<PTOTypedExprRef>::unknown(
            PTOAnalysisUnknownReason::UnsupportedOperation);
      }
      if (expr->kind == PTOTypedExpr::Kind::Constant) {
        return PTOAnalysisResult<PTOTypedExprRef>::known(
            makePTOConstantExpr(0, expr->type));
      }
      if (expr->kind == PTOTypedExpr::Kind::Opaque) {
        auto evolution = valueEvolution.getEvolution(expr->opaque, loop);
        if (!evolution) {
          return PTOAnalysisResult<PTOTypedExprRef>::unknown(evolution.reason);
        }
        return PTOAnalysisResult<PTOTypedExprRef>::known(
            evolution.value->step);
      }
      if (expr->kind == PTOTypedExpr::Kind::Cast) {
        if (expr->sourceOperation &&
            expr->sourceOperation->getNumResults() == 1) {
          auto evolution = valueEvolution.getEvolution(
              expr->sourceOperation->getResult(0), loop);
          if (!evolution) {
            return PTOAnalysisResult<PTOTypedExprRef>::unknown(
                evolution.reason);
          }
          // A value-preserving cast keeps the mathematical delta computed for
          // the cast result. Recasting the input delta would turn a descending
          // unsigned step such as -1 into its positive source-width bit
          // pattern.
          return PTOAnalysisResult<PTOTypedExprRef>::known(
              evolution.value->step);
        }
        auto input = getExprDelta(expr->lhs);
        if (!input) {
          return input;
        }
        return PTOAnalysisResult<PTOTypedExprRef>::known(
            makePTOCastExpr(expr->castKind, *input.value, expr->type,
                            expr->sourceOperation));
      }
      auto lhs = getExprDelta(expr->lhs);
      auto rhs = getExprDelta(expr->rhs);
      if (!lhs || !rhs) {
        return PTOAnalysisResult<PTOTypedExprRef>::unknown(
            !lhs ? lhs.reason : rhs.reason);
      }
      if (expr->kind == PTOTypedExpr::Kind::Add) {
        return PTOAnalysisResult<PTOTypedExprRef>::known(
            makePTOAddExpr(*lhs.value, *rhs.value, expr->type));
      }
      if (expr->kind == PTOTypedExpr::Kind::Sub) {
        return PTOAnalysisResult<PTOTypedExprRef>::known(
            makePTOSubExpr(*lhs.value, *rhs.value, expr->type));
      }
      bool lhsInvariant = isZero(*lhs.value);
      bool rhsInvariant = isZero(*rhs.value);
      if (lhsInvariant == rhsInvariant) {
        return PTOAnalysisResult<PTOTypedExprRef>::unknown(
            PTOAnalysisUnknownReason::NonAffineRecurrence);
      }
      return PTOAnalysisResult<PTOTypedExprRef>::known(
          lhsInvariant ? makePTOMulExpr(expr->lhs, *rhs.value, expr->type)
                       : makePTOMulExpr(*lhs.value, expr->rhs, expr->type));
    };
    auto delta = getExprDelta(address.elementOffset);
    if (!delta) {
      return delta;
    }
    elementDelta = *delta.value;
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
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::ZeroDelta);
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
  return PTOAnalysisResult<PTOTypedExprRef>::known(makePTOSubExpr(
      to.elementOffset, from.elementOffset, to.elementOffset->type));
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
    auto scaled = scaleExpression(from.offset->value,
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
    auto scaled =
        scaleExpression(to.offset->value, *to.offset->unitBytes, 1);
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
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::ZeroDelta);
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(std::move(result));
}
