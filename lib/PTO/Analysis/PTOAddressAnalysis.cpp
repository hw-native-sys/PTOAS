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
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<int64_t> getElementBytes(Value pointer) {
  Type elementType;
  if (auto pointerType = dyn_cast<PtrType>(pointer.getType())) {
    elementType = pointerType.getElementType();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(pointer.getType())) {
    elementType = memrefType.getElementType();
  } else {
    return std::nullopt;
  }
  unsigned bitWidth = getPTOStorageElemBitWidth(elementType);
  if (bitWidth == 0 || bitWidth % mlir::pto::kValue8 != 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(bitWidth / mlir::pto::kValue8);
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

static std::optional<int64_t> getConstantIndexValue(Value value) {
  IntegerAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  if (!attr.getValue().isSignedIntN(64)) {
    return std::nullopt;
  }
  return attr.getValue().getSExtValue();
}

static int64_t normalizeRemainder(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

static std::optional<int64_t>
getKnownIndexRemainder(Value value, int64_t modulus, unsigned depth = 0) {
  if (modulus <= 1) {
    return 0;
  }
  if (depth > 8) {
    return std::nullopt;
  }
  if (auto constant = getConstantIndexValue(value)) {
    return normalizeRemainder(*constant, modulus);
  }

  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    auto forOp = dyn_cast_or_null<scf::ForOp>(
        blockArgument.getOwner()->getParentOp());
    if (forOp) {
      bool isInductionVariable = forOp.getInductionVar() == value;
      if (!isInductionVariable) {
        return std::nullopt;
      }
      auto lower =
          getKnownIndexRemainder(forOp.getLowerBound(), modulus, depth + 1);
      auto step =
          getKnownIndexRemainder(forOp.getStep(), modulus, depth + 1);
      if (lower && step && *step == 0) {
        return lower;
      }
    }
  }

  if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  }
  if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>()) {
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  }
  if (auto cast = value.getDefiningOp<arith::ExtSIOp>()) {
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  }
  if (auto cast = value.getDefiningOp<arith::ExtUIOp>()) {
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  }
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    bool isSingleCast = cast->getNumOperands() == 1 &&
                        cast->getNumResults() == 1;
    if (isSingleCast) {
      return getKnownIndexRemainder(cast.getOperand(0), modulus, depth + 1);
    }
  }

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getKnownIndexRemainder(add.getLhs(), modulus, depth + 1);
    auto rhs = getKnownIndexRemainder(add.getRhs(), modulus, depth + 1);
    return lhs && rhs ? std::optional<int64_t>(normalizeRemainder(
                              *lhs + *rhs, modulus))
                      : std::nullopt;
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = getKnownIndexRemainder(sub.getLhs(), modulus, depth + 1);
    auto rhs = getKnownIndexRemainder(sub.getRhs(), modulus, depth + 1);
    return lhs && rhs ? std::optional<int64_t>(normalizeRemainder(
                              *lhs - *rhs, modulus))
                      : std::nullopt;
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = getKnownIndexRemainder(mul.getLhs(), modulus, depth + 1);
    auto rhs = getKnownIndexRemainder(mul.getRhs(), modulus, depth + 1);
    if (lhs && *lhs == 0) {
      return 0;
    }
    if (rhs && *rhs == 0) {
      return 0;
    }
    return lhs && rhs ? std::optional<int64_t>(normalizeRemainder(
                              *lhs * *rhs, modulus))
                      : std::nullopt;
  }
  return std::nullopt;
}

static std::optional<int64_t>
getKnownPointerRemainder(Value pointer, int64_t alignmentBytes,
                         unsigned depth = 0) {
  if (alignmentBytes <= 1) {
    return 0;
  }
  if (depth > 8) {
    return std::nullopt;
  }
  if (isa<BlockArgument>(pointer)) {
    return 0;
  }
  if (auto cast = pointer.getDefiningOp<CastPtrOp>()) {
    Value input = cast.getInput();
    if (isa<PtrType>(input.getType())) {
      return getKnownPointerRemainder(input, alignmentBytes, depth + 1);
    }
    if (isa<IntegerType>(input.getType())) {
      return getKnownIndexRemainder(input, alignmentBytes);
    }
    return std::nullopt;
  }
  if (auto cast = pointer.getDefiningOp<UnrealizedConversionCastOp>()) {
    bool isSingleCast = cast->getNumOperands() == 1 &&
                        cast->getNumResults() == 1;
    if (isSingleCast) {
      return getKnownPointerRemainder(cast.getOperand(0), alignmentBytes,
                                      depth + 1);
    }
    return std::nullopt;
  }
  auto add = pointer.getDefiningOp<AddPtrOp>();
  if (!add) {
    return std::nullopt;
  }
  auto base = getKnownPointerRemainder(add.getPtr(), alignmentBytes,
                                       depth + 1);
  auto pointerType = dyn_cast<PtrType>(add.getPtr().getType());
  if (!base || !pointerType) {
    return std::nullopt;
  }
  unsigned elementBits = getPTOStorageElemBitWidth(pointerType.getElementType());
  if (elementBits == 0 || elementBits % 8 != 0) {
    return std::nullopt;
  }
  int64_t elementBytes = static_cast<int64_t>(elementBits / 8);
  int64_t offsetModulus =
      alignmentBytes / std::gcd(alignmentBytes, elementBytes);
  auto offset = getKnownIndexRemainder(add.getOffset(), offsetModulus);
  if (!offset) {
    return std::nullopt;
  }
  return normalizeRemainder(*base + *offset * elementBytes, alignmentBytes);
}

struct NormalizedPointer {
  Value root;
  PTOTypedExprRef offset;
  int64_t elementBytes = 0;
};

static std::optional<NormalizedPointer>
normalizePointer(Value pointer, PTOValueEvolutionAnalysis &valueEvolution) {
  auto elementBytes = getElementBytes(pointer);
  if (!elementBytes) {
    return std::nullopt;
  }
  NormalizedPointer result{pointer,
                           makePTOConstantExpr(0, pointer.getType()),
                           *elementBytes};
  while (true) {
    if (auto add = result.root.getDefiningOp<AddPtrOp>()) {
      auto parentBytes = getElementBytes(add.getPtr());
      if (!parentBytes || *parentBytes != result.elementBytes) {
        break;
      }
      result.offset = makePTOAddExpr(
          result.offset, valueEvolution.getExpr(add.getOffset()),
          add.getOffset().getType());
      result.root = add.getPtr();
      continue;
    }
    if (auto cast = result.root.getDefiningOp<CastPtrOp>()) {
      auto inputBytes = getElementBytes(cast.getInput());
      if (!inputBytes || *inputBytes != result.elementBytes) {
        break;
      }
      result.root = cast.getInput();
      continue;
    }
    break;
  }
  return result;
}

} // namespace

std::optional<int64_t>
mlir::pto::getKnownAddressRemainderBytes(Value pointer, Value elementOffset,
                                         Type elementType,
                                         int64_t alignmentBytes) {
  if (alignmentBytes <= 0) {
    return std::nullopt;
  }
  auto base = getKnownPointerRemainder(pointer, alignmentBytes);
  unsigned elementBits = getPTOStorageElemBitWidth(elementType);
  if (!base || elementBits == 0 || elementBits % 8 != 0) {
    return std::nullopt;
  }
  int64_t elementBytes = static_cast<int64_t>(elementBits / 8);
  int64_t offsetModulus =
      alignmentBytes / std::gcd(alignmentBytes, elementBytes);
  auto offset = getKnownIndexRemainder(elementOffset, offsetModulus);
  if (!offset) {
    return std::nullopt;
  }
  return normalizeRemainder(*base + *offset * elementBytes, alignmentBytes);
}

bool mlir::pto::isKnownAddressAligned(Value pointer, Value elementOffset,
                                      Type elementType,
                                      int64_t alignmentBytes) {
  std::optional<int64_t> remainder = getKnownAddressRemainderBytes(
      pointer, elementOffset, elementType, alignmentBytes);
  return remainder && *remainder == 0;
}

std::optional<int64_t>
mlir::pto::getKnownAddressDifferenceBytes(Value from, Value to) {
  func::FuncOp func = from ? from.getParentRegion()->getParentOfType<func::FuncOp>()
                           : func::FuncOp();
  if (!func || !to ||
      to.getParentRegion()->getParentOfType<func::FuncOp>() != func) {
    return std::nullopt;
  }
  PTOValueEvolutionAnalysis valueEvolution(func);
  auto fromInfo = normalizePointer(from, valueEvolution);
  auto toInfo = normalizePointer(to, valueEvolution);
  if (!fromInfo || !toInfo || fromInfo->root != toInfo->root ||
      fromInfo->elementBytes != toInfo->elementBytes) {
    return std::nullopt;
  }
  auto fromPoint = valueEvolution.getPointExpression(fromInfo->offset);
  auto toPoint = valueEvolution.getPointExpression(toInfo->offset);
  if (!fromPoint || !toPoint) {
    return std::nullopt;
  }
  auto differenceExpr = makePTOSubExpr(*toPoint.value, *fromPoint.value);
  auto difference = normalizePTOLinearExpr(differenceExpr);
  if (!difference) {
    return std::nullopt;
  }
  if (!difference->terms.empty()) {
    return std::nullopt;
  }
  int64_t result;
  if (llvm::MulOverflow(difference->constant, fromInfo->elementBytes, result)) {
    return std::nullopt;
  }
  return result;
}

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
