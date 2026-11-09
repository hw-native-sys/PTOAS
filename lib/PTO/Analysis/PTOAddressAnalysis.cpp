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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/DenseSet.h"

#include <numeric>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr int64_t kBitsPerByte = 8;

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

// Each top-level address query has its own budget and memo table. Cycles and
// budget exhaustion never publish a provisional result.
class AlignmentQueryContext {
public:
  std::optional<int64_t> query(Value value, int64_t modulus) {
    if (modulus <= 1)
      return 0;
    auto key = std::make_pair(value, modulus);
    auto found = cache.find(key);
    if (found != cache.end())
      return found->second;
    if (active.contains(key) || remaining == 0)
      return std::nullopt;
    --remaining;
    active.insert(key);
    auto result = evaluate(value, modulus);
    active.erase(key);
    cache[key] = result;
    return result;
  }

private:
  unsigned remaining = 256;
  llvm::DenseMap<std::pair<Value, int64_t>, std::optional<int64_t>> cache;
  llvm::DenseSet<std::pair<Value, int64_t>> active;

  // Integer wrapping and signed/unsigned casts preserve congruence only if
  // the modulus divides the wrap boundary of every relevant width.
  static bool survivesWidth(Type type, int64_t modulus) {
    unsigned width = 64;
    if (auto integer = dyn_cast<IntegerType>(type))
      width = integer.getWidth();
    else if (!isa<IndexType>(type))
      return false;
    return llvm::isPowerOf2_64(static_cast<uint64_t>(modulus)) &&
           (width >= 63 || static_cast<uint64_t>(modulus) <= (uint64_t(1) << width));
  }

  std::optional<int64_t> explicitAlignment(Value value) {
    IntegerAttr attr;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto func = dyn_cast_or_null<func::FuncOp>(arg.getOwner()->getParentOp());
      if (func && !func.isDeclaration() && arg.getOwner() == &func.getBody().front())
        attr = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "pto.alignment");
    } else if (auto op = value.getDefiningOp<CastPtrOp>()) {
      if (isa<IntegerType>(op.getInput().getType()) && isa<PtrType>(value.getType()))
        attr = op->getAttrOfType<IntegerAttr>("pto.alignment");
    }
    if (!attr || !attr.getValue().isStrictlyPositive() ||
        attr.getValue().getActiveBits() > 63 || !attr.getValue().isPowerOf2())
      return std::nullopt;
    return static_cast<int64_t>(attr.getValue().getZExtValue());
  }

  std::optional<int64_t> evaluate(Value value, int64_t modulus) {
    if (auto type = dyn_cast<MemRefType>(value.getType())) {
      // The query has no use-site parameter. An assumption is usable only if
      // it dominates every use, including all uses in nested control flow.
      // Restrict this to identity layouts whose logical offset is zero.
      if (!type.hasStaticShape() || !type.getLayout().isIdentity())
        return std::nullopt;
      auto func = value.getParentRegion()->getParentOfType<func::FuncOp>();
      if (!func)
        return std::nullopt;
      DominanceInfo dominance(func);
      for (Operation *user : value.getUsers()) {
        auto assume = dyn_cast<memref::AssumeAlignmentOp>(user);
        if (!assume || assume.getAlignment() == 0 ||
            assume.getAlignment() % modulus != 0)
          continue;
        bool dominatesAll = llvm::all_of(value.getUsers(), [&](Operation *use) {
          return use == user || dominance.dominates(user, use);
        });
        if (dominatesAll)
          return 0;
      }
      return std::nullopt;
    }
    if (isa<PtrType>(value.getType())) {
      if (auto alignment = explicitAlignment(value))
        if (*alignment % modulus == 0)
          return 0;
      if (auto cast = value.getDefiningOp<CastPtrOp>()) {
        Value input = cast.getInput();
        if (isa<PtrType>(input.getType()) ||
            (isa<IntegerType>(input.getType()) && survivesWidth(input.getType(), modulus)))
          return query(input, modulus);
        return std::nullopt;
      }
      if (auto add = value.getDefiningOp<AddPtrOp>()) {
        auto base = query(add.getPtr(), modulus);
        auto bytes = getElementBytes(add.getPtr());
        if (!base || !bytes)
          return std::nullopt;
        auto offset = query(add.getOffset(), modulus / std::gcd(modulus, *bytes));
        if (!offset)
          return std::nullopt;
        return static_cast<int64_t>((static_cast<unsigned __int128>(*base) +
               static_cast<unsigned __int128>(*offset) * *bytes) % modulus);
      }
    } else {
      IntegerAttr constant;
      if (matchPattern(value, m_Constant(&constant))) {
        // Sign extension changes the represented integer by 2^width.
        if (!survivesWidth(value.getType(), modulus)) {
          // For non-power-of-two queries, retain exact signed constants only.
          if (!constant.getValue().isSignedIntN(64))
            return std::nullopt;
          int64_t remainder = constant.getValue().getSExtValue() % modulus;
          return remainder < 0 ? remainder + modulus : remainder;
        }
        return static_cast<int64_t>(constant.getValue().urem(static_cast<uint64_t>(modulus)));
      }
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        if (loop && loop.getInductionVar() == value && survivesWidth(value.getType(), modulus)) {
          auto lower = query(loop.getLowerBound(), modulus);
          auto step = query(loop.getStep(), modulus);
          if (lower && step && *step == 0)
            return lower;
        }
      }
      if (!survivesWidth(value.getType(), modulus))
        return std::nullopt;
      Value input;
      if (auto op = value.getDefiningOp<arith::IndexCastOp>()) input = op.getIn();
      else if (auto op = value.getDefiningOp<arith::IndexCastUIOp>()) input = op.getIn();
      else if (auto op = value.getDefiningOp<arith::ExtSIOp>()) input = op.getIn();
      else if (auto op = value.getDefiningOp<arith::ExtUIOp>()) input = op.getIn();
      else if (auto op = value.getDefiningOp<arith::TruncIOp>()) input = op.getIn();
      if (input)
        return survivesWidth(input.getType(), modulus) ? query(input, modulus) : std::nullopt;
      Value lhs, rhs;
      enum { None, Add, Sub, Mul } kind = None;
      if (auto op = value.getDefiningOp<arith::AddIOp>()) { lhs = op.getLhs(); rhs = op.getRhs(); kind = Add; }
      else if (auto op = value.getDefiningOp<arith::SubIOp>()) { lhs = op.getLhs(); rhs = op.getRhs(); kind = Sub; }
      else if (auto op = value.getDefiningOp<arith::MulIOp>()) { lhs = op.getLhs(); rhs = op.getRhs(); kind = Mul; }
      if (kind != None) {
        auto a = query(lhs, modulus), b = query(rhs, modulus);
        if (kind == Mul && ((a && *a == 0) || (b && *b == 0))) return 0;
        if (!a || !b) return std::nullopt;
        unsigned __int128 x = *a, y = *b, m = modulus;
        return static_cast<int64_t>((kind == Add ? x + y : kind == Sub ? x + m - y : x * y) % m);
      }
    }
    if (auto select = value.getDefiningOp<arith::SelectOp>()) {
      auto a = query(select.getTrueValue(), modulus);
      auto b = query(select.getFalseValue(), modulus);
      return a && b && *a == *b ? a : std::nullopt;
    }
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
          isa<PtrType>(value.getType()) && isa<PtrType>(cast.getOperand(0).getType()))
        return query(cast.getOperand(0), modulus);
    }
    return std::nullopt;
  }
};

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
  AlignmentQueryContext context;
  auto base = context.query(pointer, alignmentBytes);
  unsigned elementBits = getPTOStorageElemBitWidth(elementType);
  if (!base || elementBits == 0 || elementBits % kBitsPerByte != 0) {
    return std::nullopt;
  }
  int64_t elementBytes = static_cast<int64_t>(elementBits / 8);
  int64_t offsetModulus =
      alignmentBytes / std::gcd(alignmentBytes, elementBytes);
  auto offset = context.query(elementOffset, offsetModulus);
  if (!offset) {
    return std::nullopt;
  }
  return static_cast<int64_t>((static_cast<unsigned __int128>(*base) +
         static_cast<unsigned __int128>(*offset) * elementBytes) % alignmentBytes);
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
  if (llvm::MulOverflow(difference->constant, fromInfo->elementBytes, result) != 0) {
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
PTOAddressAnalysis::getOffsetEvolutionDeltaBytes(const PTOAddressExpr &address,
                                                 scf::ForOp loop) {
  if (!address.offset) {
    return PTOAnalysisResult<PTOTypedExprRef>::known(makePTOConstantExpr(0));
  }
  if (!address.offset->unitBytes) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::UnknownUnitSize);
  }
  auto offsetEvolution =
      valueEvolution.getEvolution(address.offset->sourceValue, loop);
  if (!offsetEvolution) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(offsetEvolution.reason);
  }
  return scaleExpression(offsetEvolution.value->step,
                         *address.offset->unitBytes, 1);
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

  auto offsetBytesResult = getOffsetEvolutionDeltaBytes(address, loop);
  if (!offsetBytesResult) {
    return offsetBytesResult;
  }
  PTOTypedExprRef offsetBytes = *offsetBytesResult.value;

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
                                       int64_t targetUnitBytes) const {
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
PTOAddressAnalysis::getOffsetPointBytes(
    const std::optional<PTOTypedAddressOffset> &offset) {
  if (!offset) {
    return PTOAnalysisResult<PTOTypedExprRef>::known(makePTOConstantExpr(0));
  }
  if (!offset->unitBytes) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(
        PTOAnalysisUnknownReason::UnknownUnitSize);
  }
  auto pointValue = valueEvolution.getPointExpression(offset->value);
  if (!pointValue) {
    return pointValue;
  }
  return scaleExpression(*pointValue.value, *offset->unitBytes, 1);
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

  auto fromOffsetResult = getOffsetPointBytes(from.offset);
  if (!fromOffsetResult) {
    return fromOffsetResult;
  }
  PTOTypedExprRef fromOffsetBytes = *fromOffsetResult.value;

  auto toOffsetResult = getOffsetPointBytes(to.offset);
  if (!toOffsetResult) {
    return toOffsetResult;
  }
  PTOTypedExprRef toOffsetBytes = *toOffsetResult.value;

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
