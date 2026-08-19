// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOValueEvolutionAnalysis.cpp - Typed scalar evolution -----------===//

#include "PTO/Analysis/PTOValueEvolutionAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr unsigned kIndexBitWidth = 64;
// Exact state propagation is the conservative fallback for loop-varying
// recurrences.  Keep it bounded so compile time cannot scale with an
// arbitrarily large constant trip count; constant-step recurrences continue
// to use the closed-form proof below and are not subject to this limit.
static constexpr uint64_t kMaxExactDynamicTripCount = 65536;

static unsigned getIntegerLikeWidth(Type type) {
  if (!type) {
    return 0;
  }
  if (type.isIndex()) {
    return kIndexBitWidth;
  }
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }
  return 0;
}

static bool operationProvesNoWrap(Operation *operation, bool isUnsigned) {
  auto overflow =
      dyn_cast_or_null<arith::ArithIntegerOverflowFlagsInterface>(operation);
  return overflow && (isUnsigned ? overflow.hasNoUnsignedWrap()
                                 : overflow.hasNoSignedWrap());
}

static bool staticallyPreservesCastValue(PTOCastKind kind, Type sourceType,
                                         Type resultType) {
  unsigned sourceWidth = getIntegerLikeWidth(sourceType);
  unsigned resultWidth = getIntegerLikeWidth(resultType);
  if (sourceWidth == 0 || resultWidth == 0) {
    return false;
  }
  switch (kind) {
  case PTOCastKind::IndexCast:
  case PTOCastKind::ExtSI:
    return resultWidth >= sourceWidth;
  case PTOCastKind::IndexCastUI:
  case PTOCastKind::ExtUI:
    return resultWidth > sourceWidth;
  case PTOCastKind::TruncI:
    return false;
  }
  return false;
}

static Type chooseType(Type requested, const PTOTypedExprRef &lhs,
                       const PTOTypedExprRef &rhs = {}) {
  if (requested) {
    return requested;
  }
  if (lhs && lhs->type) {
    return lhs->type;
  }
  return rhs ? rhs->type : Type();
}

static std::optional<APInt> getConstantAPInt(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant))) {
    return std::nullopt;
  }
  return constant;
}

static std::optional<uint64_t> getConstantTripCount(scf::ForOp loop) {
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0) {
    return std::nullopt;
  }
  if (*lower >= *upper) {
    return 0;
  }

  __int128 distance = static_cast<__int128>(*upper) - *lower;
  __int128 count =
      (distance + static_cast<__int128>(*step) - 1) / *step;
  if (count > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(count);
}

static bool sameAtom(const PTOTypedExprRef &lhs,
                     const PTOTypedExprRef &rhs) {
  if (!lhs || !rhs || lhs->type != rhs->type) {
    return false;
  }
  if (lhs->sourceValue || rhs->sourceValue) {
    return lhs->sourceValue && lhs->sourceValue == rhs->sourceValue;
  }
  if (lhs->kind != rhs->kind) {
    return false;
  }
  if (lhs->kind == PTOTypedExpr::Kind::Opaque) {
    return lhs->opaque == rhs->opaque;
  }
  if (lhs->kind != PTOTypedExpr::Kind::Cast) {
    return false;
  }
  return lhs->castKind == rhs->castKind && sameAtom(lhs->lhs, rhs->lhs);
}

static bool addLinearConstant(PTOLinearExpr &linear, int64_t value) {
  int64_t result;
  if (llvm::AddOverflow(linear.constant, value, result)) {
    return false;
  }
  linear.constant = result;
  return true;
}

static bool addLinearTerm(PTOLinearExpr &linear, PTOTypedExprRef atom,
                          int64_t coefficient) {
  if (coefficient == 0) {
    return true;
  }
  for (PTOLinearTerm &term : linear.terms) {
    if (!sameAtom(term.atom, atom)) {
      continue;
    }
    int64_t result;
    if (llvm::AddOverflow(term.coefficient, coefficient, result)) {
      return false;
    }
    term.coefficient = result;
    llvm::erase_if(linear.terms,
                   [](const PTOLinearTerm &item) {
                     return item.coefficient == 0;
                   });
    return true;
  }
  linear.terms.push_back({std::move(atom), coefficient});
  return true;
}

static bool accumulateLinear(const PTOTypedExprRef &expr, int64_t scale,
                             PTOLinearExpr &linear) {
  if (!expr) {
    return false;
  }
  if (expr->sourceValue) {
    if (auto constant = foldPTOConstant(expr)) {
      int64_t scaled;
      return !llvm::MulOverflow(*constant, scale, scaled) &&
             addLinearConstant(linear, scaled);
    }
    return addLinearTerm(linear, expr, scale);
  }
  switch (expr->kind) {
  case PTOTypedExpr::Kind::Constant: {
    auto constant = foldPTOConstant(expr);
    int64_t scaled;
    return constant && !llvm::MulOverflow(*constant, scale, scaled) &&
           addLinearConstant(linear, scaled);
  }
  case PTOTypedExpr::Kind::Opaque:
  case PTOTypedExpr::Kind::Cast:
    if (auto constant = foldPTOConstant(expr)) {
      int64_t scaled;
      return !llvm::MulOverflow(*constant, scale, scaled) &&
             addLinearConstant(linear, scaled);
    }
    return addLinearTerm(linear, expr, scale);
  case PTOTypedExpr::Kind::Add:
    return accumulateLinear(expr->lhs, scale, linear) &&
           accumulateLinear(expr->rhs, scale, linear);
  case PTOTypedExpr::Kind::Sub: {
    int64_t negativeScale;
    return !llvm::SubOverflow(int64_t{0}, scale, negativeScale) &&
           accumulateLinear(expr->lhs, scale, linear) &&
           accumulateLinear(expr->rhs, negativeScale, linear);
  }
  case PTOTypedExpr::Kind::Mul: {
    if (auto lhsConstant = foldPTOConstant(expr->lhs)) {
      int64_t nextScale;
      return !llvm::MulOverflow(scale, *lhsConstant, nextScale) &&
             accumulateLinear(expr->rhs, nextScale, linear);
    }
    if (auto rhsConstant = foldPTOConstant(expr->rhs)) {
      int64_t nextScale;
      return !llvm::MulOverflow(scale, *rhsConstant, nextScale) &&
             accumulateLinear(expr->lhs, nextScale, linear);
    }
    return false;
  }
  }
  return false;
}

static bool fitsSigned(__int128 value, unsigned width) {
  if (width == 0 || width > 64) {
    return false;
  }
  __int128 minimum = -(__int128{1} << (width - 1));
  __int128 maximum = (__int128{1} << (width - 1)) - 1;
  return value >= minimum && value <= maximum;
}

static bool fitsUnsigned(__int128 value, unsigned width) {
  if (width == 0 || width > 64 || value < 0) {
    return false;
  }
  __int128 maximum = width == 64
                         ? static_cast<__int128>(
                               std::numeric_limits<uint64_t>::max())
                         : (__int128{1} << width) - 1;
  return value <= maximum;
}

static APInt makeRangeEndpoint(__int128 value, unsigned width) {
  return APInt(width, static_cast<uint64_t>(value), true);
}

static PTOFiniteRange makeRange(__int128 minimum, __int128 maximum,
                                unsigned width, bool isUnsigned) {
  return {makeRangeEndpoint(minimum, width),
          makeRangeEndpoint(maximum, width), isUnsigned};
}

static std::optional<__int128> getMathematicalConstant(Value value,
                                                       bool isUnsigned) {
  auto bits = getConstantAPInt(value);
  if (
      !bits || bits->getBitWidth() > 64) {
    return std::nullopt;
  }
  return isUnsigned ? static_cast<__int128>(bits->getZExtValue())
                    : static_cast<__int128>(bits->getSExtValue());
}

static std::optional<std::pair<Value, bool>>
getDirectRecurrenceStep(BlockArgument iterArg, scf::ForOp loop) {
  unsigned index = iterArg.getArgNumber() - 1;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value yielded = yield.getOperand(index);
  if (yielded == iterArg) {
    return std::pair<Value, bool>{Value(), false};
  }
  if (auto add = yielded.getDefiningOp<arith::AddIOp>()) {
    if (
        add.getLhs() == iterArg && loop.isDefinedOutsideOfLoop(add.getRhs())) {
      return std::pair<Value, bool>{add.getRhs(), false};
    }
    if (
        add.getRhs() == iterArg && loop.isDefinedOutsideOfLoop(add.getLhs())) {
      return std::pair<Value, bool>{add.getLhs(), false};
    }
  }
  if (auto sub = yielded.getDefiningOp<arith::SubIOp>()) {
    if (
        sub.getLhs() == iterArg &&
        loop.isDefinedOutsideOfLoop(sub.getRhs())) {
      return std::pair<Value, bool>{sub.getRhs(), true};
    }
  }
  return std::nullopt;
}

static PTOCastKind getCastKind(Operation *operation) {
  if (isa<arith::IndexCastUIOp>(operation)) {
    return PTOCastKind::IndexCastUI;
  }
  if (isa<arith::TruncIOp>(operation)) {
    return PTOCastKind::TruncI;
  }
  if (isa<arith::ExtSIOp>(operation)) {
    return PTOCastKind::ExtSI;
  }
  if (isa<arith::ExtUIOp>(operation)) {
    return PTOCastKind::ExtUI;
  }
  return PTOCastKind::IndexCast;
}

struct RecurrenceDecomposition {
  int64_t coefficient = 0;
  PTOTypedExprRef increment;
};

using RecurrenceDecompositionCache =
    DenseMap<Value, std::optional<RecurrenceDecomposition>>;

static std::optional<RecurrenceDecomposition> decomposeRecurrence(
    Value value, BlockArgument iterArg, scf::ForOp loop,
    PTOValueEvolutionAnalysis &analysis,
    RecurrenceDecompositionCache &cache) {
  if (value == iterArg) {
    return RecurrenceDecomposition{
        1, makePTOConstantExpr(0, iterArg.getType())};
  }
  if (auto cached = cache.find(value); cached != cache.end()) {
    return cached->second;
  }
  auto record = [&](std::optional<RecurrenceDecomposition> result) {
    cache.try_emplace(value, result);
    return result;
  };

  Operation *definition = value.getDefiningOp();
  if (
      !definition || loop.isDefinedOutsideOfLoop(value) ||
      definition->hasTrait<OpTrait::ConstantLike>()) {
    return record(RecurrenceDecomposition{0, makePTOOpaqueExpr(value)});
  }

  if (isa<arith::AddIOp, arith::SubIOp>(definition)) {
    auto lhs = decomposeRecurrence(definition->getOperand(0), iterArg, loop,
                                   analysis, cache);
    auto rhs = decomposeRecurrence(definition->getOperand(1), iterArg, loop,
                                   analysis, cache);
    if (!lhs || !rhs) {
      return record(std::nullopt);
    }
    if (lhs->coefficient == 0 && rhs->coefficient == 0) {
      return record(RecurrenceDecomposition{0, makePTOOpaqueExpr(value)});
    }
    bool isSubtraction = isa<arith::SubIOp>(definition);
    int64_t coefficient;
    bool overflow =
        isSubtraction
            ? llvm::SubOverflow(lhs->coefficient, rhs->coefficient,
                                coefficient)
            : llvm::AddOverflow(lhs->coefficient, rhs->coefficient,
                                coefficient);
    if (overflow) {
      return record(std::nullopt);
    }
    PTOTypedExprRef increment =
        isSubtraction
            ? makePTOSubExpr(lhs->increment, rhs->increment, value.getType())
            : makePTOAddExpr(lhs->increment, rhs->increment, value.getType());
    return record(RecurrenceDecomposition{coefficient,
                                           std::move(increment)});
  }

  if (auto multiplication = dyn_cast<arith::MulIOp>(definition)) {
    auto lhs = decomposeRecurrence(multiplication.getLhs(), iterArg, loop,
                                   analysis, cache);
    auto rhs = decomposeRecurrence(multiplication.getRhs(), iterArg, loop,
                                   analysis, cache);
    if (!lhs || !rhs ||
        (lhs->coefficient != 0 && rhs->coefficient != 0)) {
      return record(std::nullopt);
    }
    if (lhs->coefficient == 0 && rhs->coefficient == 0) {
      return record(RecurrenceDecomposition{0, makePTOOpaqueExpr(value)});
    }
    const RecurrenceDecomposition &variant =
        lhs->coefficient != 0 ? *lhs : *rhs;
    Value multiplierValue = lhs->coefficient != 0
                                ? multiplication.getRhs()
                                : multiplication.getLhs();
    auto multiplier = getConstantIntValue(multiplierValue);
    int64_t coefficient;
    if (!multiplier || llvm::MulOverflow(variant.coefficient, *multiplier,
                                         coefficient)) {
      return record(std::nullopt);
    }
    return record(RecurrenceDecomposition{
        coefficient,
        makePTOMulExpr(variant.increment, analysis.getExpr(multiplierValue),
                       value.getType())});
  }

  // A cast wholly inside the increment is retained by getExpr.  A cast of the
  // accumulator itself needs a stronger modular-affine proof and is rejected
  // conservatively for now.
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(definition)) {
    auto input = decomposeRecurrence(definition->getOperand(0), iterArg, loop,
                                     analysis, cache);
    if (input && input->coefficient == 0) {
      return record(RecurrenceDecomposition{0, makePTOOpaqueExpr(value)});
    }
  }
  return record(std::nullopt);
}

static bool checkedAdd(__int128 lhs, __int128 rhs, __int128 &result) {
  return !__builtin_add_overflow(lhs, rhs, &result);
}

static bool checkedSub(__int128 lhs, __int128 rhs, __int128 &result) {
  return !__builtin_sub_overflow(lhs, rhs, &result);
}

static bool checkedMul(__int128 lhs, __int128 rhs, __int128 &result) {
  return !__builtin_mul_overflow(lhs, rhs, &result);
}

struct MathematicalRange {
  __int128 minimum;
  __int128 maximum;
};

static std::optional<MathematicalRange>
getMathematicalRange(const PTOFiniteRange &range, bool isUnsigned) {
  if (range.unsignedInterpretation != isUnsigned ||
      range.lowerInclusive.getBitWidth() > 64 ||
      range.upperInclusive.getBitWidth() > 64) {
    return std::nullopt;
  }
  if (isUnsigned) {
    return MathematicalRange{
        static_cast<__int128>(range.lowerInclusive.getZExtValue()),
        static_cast<__int128>(range.upperInclusive.getZExtValue())};
  }
  return MathematicalRange{
      static_cast<__int128>(range.lowerInclusive.getSExtValue()),
      static_cast<__int128>(range.upperInclusive.getSExtValue())};
}

static std::optional<PTOFiniteRange>
makeProvenRange(__int128 minimum, __int128 maximum, Type type,
                bool isUnsigned) {
  unsigned width = getIntegerLikeWidth(type);
  bool fits = isUnsigned ? fitsUnsigned(minimum, width) &&
                               fitsUnsigned(maximum, width)
                         : fitsSigned(minimum, width) &&
                               fitsSigned(maximum, width);
  if (minimum > maximum || !fits) {
    return std::nullopt;
  }
  return makeRange(minimum, maximum, width, isUnsigned);
}

static std::optional<PTOFiniteRange>
computeAddRange(const PTOFiniteRange &lhs, const PTOFiniteRange &rhs,
                Type type, bool isUnsigned) {
  auto lhsRange = getMathematicalRange(lhs, isUnsigned);
  auto rhsRange = getMathematicalRange(rhs, isUnsigned);
  __int128 minimum;
  __int128 maximum;
  if (!lhsRange || !rhsRange ||
      !checkedAdd(lhsRange->minimum, rhsRange->minimum, minimum) ||
      !checkedAdd(lhsRange->maximum, rhsRange->maximum, maximum)) {
    return std::nullopt;
  }
  return makeProvenRange(minimum, maximum, type, isUnsigned);
}

static std::optional<PTOFiniteRange>
computeSubRange(const PTOFiniteRange &lhs, const PTOFiniteRange &rhs,
                Type type, bool isUnsigned) {
  auto lhsRange = getMathematicalRange(lhs, isUnsigned);
  auto rhsRange = getMathematicalRange(rhs, isUnsigned);
  __int128 minimum;
  __int128 maximum;
  if (!lhsRange || !rhsRange ||
      !checkedSub(lhsRange->minimum, rhsRange->maximum, minimum) ||
      !checkedSub(lhsRange->maximum, rhsRange->minimum, maximum)) {
    return std::nullopt;
  }
  return makeProvenRange(minimum, maximum, type, isUnsigned);
}

static std::optional<PTOFiniteRange>
computeMulRange(const PTOFiniteRange &valueRange, __int128 multiplier,
                Type type, bool isUnsigned) {
  auto range = getMathematicalRange(valueRange, isUnsigned);
  if (!range) {
    return std::nullopt;
  }
  __int128 lowerProduct;
  __int128 upperProduct;
  bool lowerValid = checkedMul(range->minimum, multiplier, lowerProduct);
  bool upperValid = checkedMul(range->maximum, multiplier, upperProduct);
  if (!lowerValid || !upperValid) {
    return std::nullopt;
  }
  return makeProvenRange(std::min(lowerProduct, upperProduct),
                         std::max(lowerProduct, upperProduct), type,
                         isUnsigned);
}

static std::optional<PTOFiniteRange>
computeMulRange(const PTOFiniteRange &lhs, const PTOFiniteRange &rhs,
                Type type, bool isUnsigned) {
  auto lhsRange = getMathematicalRange(lhs, isUnsigned);
  auto rhsRange = getMathematicalRange(rhs, isUnsigned);
  if (!lhsRange || !rhsRange) {
    return std::nullopt;
  }
  __int128 products[4];
  bool allProductsValid =
      checkedMul(lhsRange->minimum, rhsRange->minimum, products[0]) &&
      checkedMul(lhsRange->minimum, rhsRange->maximum, products[1]) &&
      checkedMul(lhsRange->maximum, rhsRange->minimum, products[2]) &&
      checkedMul(lhsRange->maximum, rhsRange->maximum, products[3]);
  if (!allProductsValid) {
    return std::nullopt;
  }
  auto bounds = std::minmax_element(std::begin(products), std::end(products));
  return makeProvenRange(*bounds.first, *bounds.second, type, isUnsigned);
}

static bool fitsInterpretation(__int128 value, Type type, bool isUnsigned) {
  unsigned width = getIntegerLikeWidth(type);
  return isUnsigned ? fitsUnsigned(value, width) : fitsSigned(value, width);
}

static std::optional<__int128> evaluateLoopValue(
    Value value, scf::ForOp loop, __int128 inductionValue,
    ArrayRef<std::optional<__int128>> iterArgValues, bool isUnsigned);

static std::optional<__int128> evaluateTypedExpr(
    const PTOTypedExprRef &expression, scf::ForOp loop,
    __int128 inductionValue,
    ArrayRef<std::optional<__int128>> iterArgValues, bool isUnsigned) {
  if (!expression) {
    return std::nullopt;
  }
  if (expression->kind == PTOTypedExpr::Kind::Opaque) {
    return evaluateLoopValue(expression->opaque, loop, inductionValue,
                             iterArgValues, isUnsigned);
  }
  if (expression->kind == PTOTypedExpr::Kind::Constant) {
    if (
        expression->constant.getBitWidth() > 64) {
      return std::nullopt;
    }
    return isUnsigned
               ? static_cast<__int128>(expression->constant.getZExtValue())
               : static_cast<__int128>(expression->constant.getSExtValue());
  }
  if (expression->kind == PTOTypedExpr::Kind::Cast) {
    if (expression->castKind != PTOCastKind::IndexCast &&
        expression->castKind != PTOCastKind::IndexCastUI) {
      return std::nullopt;
    }
    bool sourceIsUnsigned =
        expression->castKind == PTOCastKind::IndexCastUI;
    auto input = evaluateTypedExpr(expression->lhs, loop, inductionValue,
                                   iterArgValues, sourceIsUnsigned);
    if (!input || !fitsInterpretation(*input, expression->type,
                                      sourceIsUnsigned)) {
      return std::nullopt;
    }
    return input;
  }

  auto lhs = evaluateTypedExpr(expression->lhs, loop, inductionValue,
                               iterArgValues, isUnsigned);
  auto rhs = evaluateTypedExpr(expression->rhs, loop, inductionValue,
                               iterArgValues, isUnsigned);
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  __int128 result;
  bool valid = expression->kind == PTOTypedExpr::Kind::Add
                   ? checkedAdd(*lhs, *rhs, result)
               : expression->kind == PTOTypedExpr::Kind::Sub
                   ? checkedSub(*lhs, *rhs, result)
                   : checkedMul(*lhs, *rhs, result);
  if (!valid || !fitsInterpretation(result, expression->type, isUnsigned)) {
    return std::nullopt;
  }
  return result;
}

static std::optional<__int128> evaluateLoopValue(
    Value value, scf::ForOp loop, __int128 inductionValue,
    ArrayRef<std::optional<__int128>> iterArgValues, bool isUnsigned) {
  if (value == loop.getInductionVar()) {
    return fitsInterpretation(inductionValue, value.getType(), isUnsigned)
               ? std::optional<__int128>(inductionValue)
               : std::nullopt;
  }
  if (auto blockArgument = dyn_cast<BlockArgument>(value);
      blockArgument && blockArgument.getOwner() == loop.getBody() &&
      blockArgument.getArgNumber() > 0) {
    unsigned index = blockArgument.getArgNumber() - 1;
    return index < iterArgValues.size() ? iterArgValues[index]
                                        : std::nullopt;
  }
  if (loop.isDefinedOutsideOfLoop(value)) {
    return getMathematicalConstant(value, isUnsigned);
  }

  Operation *definition = value.getDefiningOp();
  if (!definition) {
    return std::nullopt;
  }
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(definition)) {
    bool sourceIsUnsigned = isa<arith::IndexCastUIOp>(definition);
    auto input = evaluateLoopValue(definition->getOperand(0), loop,
                                   inductionValue, iterArgValues,
                                   sourceIsUnsigned);
    if (!input || !fitsInterpretation(*input, value.getType(),
                                      sourceIsUnsigned)) {
      return std::nullopt;
    }
    return input;
  }
  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(definition)) {
    return std::nullopt;
  }

  auto lhs = evaluateLoopValue(definition->getOperand(0), loop,
                               inductionValue, iterArgValues, isUnsigned);
  auto rhs = evaluateLoopValue(definition->getOperand(1), loop,
                               inductionValue, iterArgValues, isUnsigned);
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  __int128 result;
  bool valid = isa<arith::AddIOp>(definition)
                   ? checkedAdd(*lhs, *rhs, result)
               : isa<arith::SubIOp>(definition)
                   ? checkedSub(*lhs, *rhs, result)
                   : checkedMul(*lhs, *rhs, result);
  if (!valid || !fitsInterpretation(result, value.getType(), isUnsigned)) {
    return std::nullopt;
  }
  return result;
}

static PTOAnalysisResult<PTOLoopEvolution> analyzeDynamicRecurrence(
    BlockArgument iterArg, scf::ForOp loop, uint64_t tripCount,
    bool isUnsigned, PTOValueEvolutionAnalysis &analysis) {
  if (tripCount > kMaxExactDynamicTripCount) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::RangeUnavailable);
  }
  unsigned index = iterArg.getArgNumber() - 1;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  RecurrenceDecompositionCache decompositionCache;
  auto decomposition = decomposeRecurrence(
      yield.getOperand(index), iterArg, loop, analysis, decompositionCache);
  if (!decomposition || decomposition->coefficient != 1) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::NonAffineRecurrence);
  }

  unsigned width = getIntegerLikeWidth(iterArg.getType());
  if (width == 0 || width > 64) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::TypeMismatch);
  }

  std::vector<std::optional<__int128>> states;
  states.reserve(loop.getInitArgs().size());
  for (Value initial : loop.getInitArgs()) {
    states.push_back(getMathematicalConstant(initial, isUnsigned));
  }
  if (!states[index]) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::RangeUnavailable);
  }

  auto induction = getMathematicalConstant(loop.getLowerBound(), false);
  auto inductionStep = getMathematicalConstant(loop.getStep(), false);
  if (!induction || !inductionStep) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::UnknownIterationDomain);
  }

  __int128 minimum = *states[index];
  __int128 maximum = *states[index];
  for (uint64_t iteration = 0; iteration < tripCount; ++iteration) {
    auto step = evaluateTypedExpr(decomposition->increment, loop, *induction,
                                  states, isUnsigned);
    std::vector<std::optional<__int128>> nextStates;
    nextStates.reserve(states.size());
    for (Value yielded : yield.getOperands()) {
      nextStates.push_back(evaluateLoopValue(
          yielded, loop, *induction, states, isUnsigned));
    }
    if (!step || !nextStates[index]) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::PossibleWrap);
    }
    __int128 observedStep;
    if (
        !checkedSub(*nextStates[index], *states[index], observedStep) ||
        observedStep != *step) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::NonAffineRecurrence);
    }
    minimum = std::min(minimum, *nextStates[index]);
    maximum = std::max(maximum, *nextStates[index]);
    states = std::move(nextStates);
    if (!checkedAdd(*induction, *inductionStep, *induction)) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::PossibleWrap);
    }
  }

  auto constantStep = foldPTOConstant(decomposition->increment);
  return PTOAnalysisResult<PTOLoopEvolution>::known(
      {analysis.getExpr(loop.getInitArgs()[index]), decomposition->increment,
       tripCount, makeRange(minimum, maximum, width, isUnsigned), true, true,
       constantStep});
}

} // namespace

StringRef mlir::pto::stringifyPTOAnalysisUnknownReason(
    PTOAnalysisUnknownReason reason) {
  switch (reason) {
  case PTOAnalysisUnknownReason::None:
    return "none";
  case PTOAnalysisUnknownReason::UnsupportedOperation:
    return "unsupported-op";
  case PTOAnalysisUnknownReason::UnsupportedCast:
    return "unsupported-cast";
  case PTOAnalysisUnknownReason::UnknownIterationDomain:
    return "unknown-loop-domain";
  case PTOAnalysisUnknownReason::RangeUnavailable:
    return "range-unavailable";
  case PTOAnalysisUnknownReason::PossibleWrap:
    return "possible-wrap";
  case PTOAnalysisUnknownReason::NonAffineRecurrence:
    return "non-affine-recurrence";
  case PTOAnalysisUnknownReason::UnknownElementSize:
    return "unknown-element-size";
  case PTOAnalysisUnknownReason::UnknownUnitSize:
    return "unknown-unit-size";
  case PTOAnalysisUnknownReason::InexactUnitConversion:
    return "inexact-unit-conversion";
  case PTOAnalysisUnknownReason::DifferentBase:
    return "different-base";
  case PTOAnalysisUnknownReason::TypeMismatch:
    return "type-mismatch";
  }
  return "unknown";
}

PTOTypedExprRef mlir::pto::makePTOConstantExpr(int64_t value, Type type) {
  auto expr = std::make_shared<PTOTypedExpr>();
  expr->kind = PTOTypedExpr::Kind::Constant;
  expr->type = type;
  unsigned width = std::max(1U, getIntegerLikeWidth(type));
  expr->constant = APInt(width, static_cast<uint64_t>(value), true);
  return expr;
}

PTOTypedExprRef mlir::pto::makePTOOpaqueExpr(Value value) {
  auto expr = std::make_shared<PTOTypedExpr>();
  expr->kind = PTOTypedExpr::Kind::Opaque;
  expr->type = value.getType();
  expr->opaque = value;
  expr->sourceValue = value;
  return expr;
}

static PTOTypedExprRef makeBinary(PTOTypedExpr::Kind kind,
                                  PTOTypedExprRef lhs, PTOTypedExprRef rhs,
                                  Type type, Value sourceValue) {
  auto expr = std::make_shared<PTOTypedExpr>();
  expr->kind = kind;
  expr->type = chooseType(type, lhs, rhs);
  expr->sourceValue = sourceValue;
  expr->lhs = std::move(lhs);
  expr->rhs = std::move(rhs);
  return expr;
}

PTOTypedExprRef mlir::pto::makePTOAddExpr(PTOTypedExprRef lhs,
                                          PTOTypedExprRef rhs, Type type,
                                          Value sourceValue) {
  if (sourceValue) {
    return makeBinary(PTOTypedExpr::Kind::Add, std::move(lhs),
                      std::move(rhs), type, sourceValue);
  }
  auto lhsConstant = foldPTOConstant(lhs);
  auto rhsConstant = foldPTOConstant(rhs);
  if (lhsConstant && rhsConstant) {
    int64_t result;
    if (!llvm::AddOverflow(*lhsConstant, *rhsConstant, result)) {
      return makePTOConstantExpr(result, chooseType(type, lhs, rhs));
    }
  }
  if (lhsConstant && *lhsConstant == 0) {
    return rhs;
  }
  if (rhsConstant && *rhsConstant == 0) {
    return lhs;
  }
  return makeBinary(PTOTypedExpr::Kind::Add, std::move(lhs), std::move(rhs),
                    type, {});
}

PTOTypedExprRef mlir::pto::makePTOSubExpr(PTOTypedExprRef lhs,
                                          PTOTypedExprRef rhs, Type type,
                                          Value sourceValue) {
  if (sourceValue) {
    return makeBinary(PTOTypedExpr::Kind::Sub, std::move(lhs),
                      std::move(rhs), type, sourceValue);
  }
  auto lhsConstant = foldPTOConstant(lhs);
  auto rhsConstant = foldPTOConstant(rhs);
  if (lhsConstant && rhsConstant) {
    int64_t result;
    if (!llvm::SubOverflow(*lhsConstant, *rhsConstant, result)) {
      return makePTOConstantExpr(result, chooseType(type, lhs, rhs));
    }
  }
  if (rhsConstant && *rhsConstant == 0) {
    return lhs;
  }
  return makeBinary(PTOTypedExpr::Kind::Sub, std::move(lhs), std::move(rhs),
                    type, {});
}

PTOTypedExprRef mlir::pto::makePTOMulExpr(PTOTypedExprRef lhs,
                                          PTOTypedExprRef rhs, Type type,
                                          Value sourceValue) {
  if (sourceValue) {
    return makeBinary(PTOTypedExpr::Kind::Mul, std::move(lhs),
                      std::move(rhs), type, sourceValue);
  }
  auto lhsConstant = foldPTOConstant(lhs);
  auto rhsConstant = foldPTOConstant(rhs);
  if (lhsConstant && rhsConstant) {
    int64_t result;
    if (!llvm::MulOverflow(*lhsConstant, *rhsConstant, result)) {
      return makePTOConstantExpr(result, chooseType(type, lhs, rhs));
    }
  }
  if (
      (lhsConstant && *lhsConstant == 0) ||
      (rhsConstant && *rhsConstant == 0)) {
    return makePTOConstantExpr(0, chooseType(type, lhs, rhs));
  }
  if (lhsConstant && *lhsConstant == 1) {
    return rhs;
  }
  if (rhsConstant && *rhsConstant == 1) {
    return lhs;
  }
  return makeBinary(PTOTypedExpr::Kind::Mul, std::move(lhs), std::move(rhs),
                    type, {});
}

PTOTypedExprRef mlir::pto::makePTOCastExpr(PTOCastKind kind,
                                           PTOTypedExprRef input,
                                           Type resultType,
                                           Operation *sourceOperation,
                                           Value sourceValue) {
  if (!sourceValue && input && input->kind == PTOTypedExpr::Kind::Constant &&
      (kind == PTOCastKind::IndexCast ||
       kind == PTOCastKind::IndexCastUI)) {
    unsigned resultWidth = getIntegerLikeWidth(resultType);
    if (resultWidth != 0) {
      auto expr = std::make_shared<PTOTypedExpr>();
      expr->kind = PTOTypedExpr::Kind::Constant;
      expr->type = resultType;
      expr->constant = kind == PTOCastKind::IndexCastUI
                           ? input->constant.zextOrTrunc(resultWidth)
                           : input->constant.sextOrTrunc(resultWidth);
      return expr;
    }
  }
  auto expr = std::make_shared<PTOTypedExpr>();
  expr->kind = PTOTypedExpr::Kind::Cast;
  expr->type = resultType;
  expr->castKind = kind;
  expr->sourceValue = sourceValue;
  expr->sourceOperation = sourceOperation;
  expr->lhs = std::move(input);
  return expr;
}

static std::optional<APInt>
getFoldedConstantBits(const PTOTypedExprRef &expr) {
  if (!expr) {
    return std::nullopt;
  }
  if (expr->kind == PTOTypedExpr::Kind::Constant) {
    return expr->constant;
  }
  if (expr->kind == PTOTypedExpr::Kind::Opaque) {
    return getConstantAPInt(expr->opaque);
  }

  unsigned width = getIntegerLikeWidth(expr->type);
  auto value = foldPTOConstant(expr);
  if (width == 0 || !value) {
    return std::nullopt;
  }
  return APInt(width, static_cast<uint64_t>(*value), true);
}

static std::optional<__int128>
getMathematicalConstant(const PTOTypedExprRef &expression,
                        bool isUnsigned) {
  auto bits = getFoldedConstantBits(expression);
  if (!bits) {
    return std::nullopt;
  }
  bool hasUnsupportedWidth = bits->getBitWidth() > 64;
  if (hasUnsupportedWidth) {
    return std::nullopt;
  }
  return isUnsigned ? static_cast<__int128>(bits->getZExtValue())
                    : static_cast<__int128>(bits->getSExtValue());
}

static std::optional<APInt> applyPTOCastToBits(PTOCastKind kind,
                                               APInt input,
                                               unsigned resultWidth) {
  if (resultWidth == 0) {
    return std::nullopt;
  }
  switch (kind) {
  case PTOCastKind::IndexCast:
    return input.sextOrTrunc(resultWidth);
  case PTOCastKind::IndexCastUI:
    return input.zextOrTrunc(resultWidth);
  case PTOCastKind::TruncI:
    if (resultWidth >= input.getBitWidth()) {
      return std::nullopt;
    }
    return input.trunc(resultWidth);
  case PTOCastKind::ExtSI:
    if (resultWidth <= input.getBitWidth()) {
      return std::nullopt;
    }
    return input.sext(resultWidth);
  case PTOCastKind::ExtUI:
    if (resultWidth <= input.getBitWidth()) {
      return std::nullopt;
    }
    return input.zext(resultWidth);
  }
  return std::nullopt;
}

static std::optional<int64_t>
foldPTOCastConstant(const PTOTypedExprRef &expr) {
  auto input = getFoldedConstantBits(expr->lhs);
  unsigned resultWidth = getIntegerLikeWidth(expr->type);
  if (!input) {
    return std::nullopt;
  }

  auto result = applyPTOCastToBits(expr->castKind, *input, resultWidth);
  if (!result) {
    return std::nullopt;
  }
  if (expr->castKind == PTOCastKind::IndexCastUI ||
      expr->castKind == PTOCastKind::ExtUI) {
    unsigned activeBits = result->getActiveBits();
    if (activeBits > 63) {
      return std::nullopt;
    }
    return static_cast<int64_t>(result->getZExtValue());
  }
  if (!result->isSignedIntN(64)) {
    return std::nullopt;
  }
  return result->sextOrTrunc(64).getSExtValue();
}

static std::optional<int64_t>
foldSourceBinaryConstant(const PTOTypedExprRef &expr) {
  auto lhs = getFoldedConstantBits(expr->lhs);
  auto rhs = getFoldedConstantBits(expr->rhs);
  unsigned resultWidth = getIntegerLikeWidth(expr->type);
  if (!lhs || !rhs || resultWidth == 0 || resultWidth > 64) {
    return std::nullopt;
  }
  APInt lhsBits = lhs->sextOrTrunc(resultWidth);
  APInt rhsBits = rhs->sextOrTrunc(resultWidth);
  APInt result = expr->kind == PTOTypedExpr::Kind::Add
                     ? lhsBits + rhsBits
                 : expr->kind == PTOTypedExpr::Kind::Sub
                     ? lhsBits - rhsBits
                     : lhsBits * rhsBits;
  return result.sextOrTrunc(64).getSExtValue();
}

std::optional<int64_t>
mlir::pto::foldPTOConstant(const PTOTypedExprRef &expr) {
  if (!expr) {
    return std::nullopt;
  }
  switch (expr->kind) {
  case PTOTypedExpr::Kind::Constant:
    if (
        expr->constant.getBitWidth() > 64) {
      return std::nullopt;
    }
    return expr->constant.getSExtValue();
  case PTOTypedExpr::Kind::Opaque:
    return getConstantIntValue(expr->opaque);
  case PTOTypedExpr::Kind::Cast:
    return foldPTOCastConstant(expr);
  case PTOTypedExpr::Kind::Add:
  case PTOTypedExpr::Kind::Sub:
  case PTOTypedExpr::Kind::Mul: {
    if (expr->sourceValue) {
      return foldSourceBinaryConstant(expr);
    }
    auto lhs = foldPTOConstant(expr->lhs);
    auto rhs = foldPTOConstant(expr->rhs);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    int64_t result;
    bool overflow = expr->kind == PTOTypedExpr::Kind::Add
                        ? llvm::AddOverflow(*lhs, *rhs, result)
                    : expr->kind == PTOTypedExpr::Kind::Sub
                        ? llvm::SubOverflow(*lhs, *rhs, result)
                        : llvm::MulOverflow(*lhs, *rhs, result);
    return overflow ? std::nullopt : std::optional<int64_t>(result);
  }
  }
  return std::nullopt;
}

std::optional<PTOLinearExpr>
mlir::pto::normalizePTOLinearExpr(const PTOTypedExprRef &expr) {
  PTOLinearExpr linear;
  if (!accumulateLinear(expr, 1, linear)) {
    return std::nullopt;
  }
  return linear;
}

bool mlir::pto::equalPTOLinearExprs(const PTOLinearExpr &lhs,
                                    const PTOLinearExpr &rhs) {
  if (
      lhs.constant != rhs.constant || lhs.terms.size() != rhs.terms.size()) {
    return false;
  }
  return llvm::all_of(lhs.terms, [&rhs](const PTOLinearTerm &leftTerm) {
    return llvm::any_of(rhs.terms, [&leftTerm](const PTOLinearTerm &rightTerm) {
      return leftTerm.coefficient == rightTerm.coefficient &&
             sameAtom(leftTerm.atom, rightTerm.atom);
    });
  });
}

bool mlir::pto::dividePTOLinearExprExact(PTOLinearExpr &expr,
                                         int64_t divisor) {
  if (divisor <= 0 || expr.constant % divisor != 0) {
    return false;
  }
  if (llvm::any_of(expr.terms, [divisor](const PTOLinearTerm &term) {
        return term.coefficient % divisor != 0;
      })) {
    return false;
  }
  expr.constant /= divisor;
  for (PTOLinearTerm &term : expr.terms) {
    term.coefficient /= divisor;
  }
  return true;
}

bool mlir::pto::isZeroPTOLinearExpr(const PTOLinearExpr &expr) {
  return expr.constant == 0 && expr.terms.empty();
}

PTOTypedExprRef mlir::pto::buildPTOTypedExpr(const PTOLinearExpr &linear,
                                             Type type) {
  PTOTypedExprRef result;
  if (linear.constant != 0) {
    result = makePTOConstantExpr(linear.constant, type);
  }
  for (const PTOLinearTerm &term : linear.terms) {
    PTOTypedExprRef value = term.atom;
    if (term.coefficient != 1) {
      value = makePTOMulExpr(
          value, makePTOConstantExpr(term.coefficient, value->type), type);
    }
    result = result ? makePTOAddExpr(result, value, type) : value;
  }
  return result ? result : makePTOConstantExpr(0, type);
}

void mlir::pto::collectPTOExprLeaves(const PTOTypedExprRef &expr,
                                     SmallVectorImpl<Value> &leaves) {
  if (!expr) {
    return;
  }
  if (expr->sourceValue &&
      expr->kind != PTOTypedExpr::Kind::Constant) {
    leaves.push_back(expr->sourceValue);
    return;
  }
  if (expr->kind == PTOTypedExpr::Kind::Opaque) {
    leaves.push_back(expr->opaque);
    return;
  }
  collectPTOExprLeaves(expr->lhs, leaves);
  collectPTOExprLeaves(expr->rhs, leaves);
}

bool mlir::pto::getPTOExprType(const PTOTypedExprRef &expr, Type &type) {
  if (!expr) {
    return false;
  }
  if (expr->kind == PTOTypedExpr::Kind::Constant) {
    if (expr->type) {
      type = expr->type;
    }
    return true;
  }
  if (expr->kind == PTOTypedExpr::Kind::Opaque ||
      expr->kind == PTOTypedExpr::Kind::Cast) {
    type = expr->type;
    return true;
  }
  Type lhsType;
  Type rhsType;
  if (
      !getPTOExprType(expr->lhs, lhsType) ||
      !getPTOExprType(expr->rhs, rhsType)) {
    return false;
  }
  if (lhsType && rhsType && lhsType != rhsType) {
    return false;
  }
  type = expr->type ? expr->type : (lhsType ? lhsType : rhsType);
  return true;
}

void mlir::pto::printPTOTypedExpr(const PTOTypedExprRef &expr,
                                  llvm::raw_ostream &os) {
  if (!expr) {
    os << "<unknown>";
    return;
  }
  switch (expr->kind) {
  case PTOTypedExpr::Kind::Constant:
    os << expr->constant.getSExtValue();
    return;
  case PTOTypedExpr::Kind::Opaque:
    expr->opaque.printAsOperand(os, OpPrintingFlags());
    return;
  case PTOTypedExpr::Kind::Cast:
    os << (expr->castKind == PTOCastKind::IndexCastUI ? "index_castui(" :
                                                       "index_cast(");
    printPTOTypedExpr(expr->lhs, os);
    os << ")";
    return;
  case PTOTypedExpr::Kind::Add:
  case PTOTypedExpr::Kind::Sub:
  case PTOTypedExpr::Kind::Mul:
    os << "(";
    printPTOTypedExpr(expr->lhs, os);
    os << (expr->kind == PTOTypedExpr::Kind::Add
               ? " + "
               : expr->kind == PTOTypedExpr::Kind::Sub ? " - " : " * ");
    printPTOTypedExpr(expr->rhs, os);
    os << ")";
    return;
  }
}

namespace {

struct PointExpressionProof {
  PTOTypedExprRef expression;
  std::optional<PTOFiniteRange> range;
};

static std::optional<PTOFiniteRange> getFullPointRange(Type type,
                                                       bool isUnsigned) {
  unsigned width = getIntegerLikeWidth(type);
  if (width == 0 || width > 64) {
    return std::nullopt;
  }
  APInt minimum = isUnsigned ? APInt::getZero(width)
                             : APInt::getSignedMinValue(width);
  APInt maximum = isUnsigned ? APInt::getMaxValue(width)
                             : APInt::getSignedMaxValue(width);
  return PTOFiniteRange{minimum, maximum, isUnsigned};
}

static std::optional<PTOFiniteRange>
computePointBinaryRange(const PTOTypedExprRef &expression,
                        const PointExpressionProof &lhs,
                        const PointExpressionProof &rhs, bool isUnsigned) {
  if (!lhs.range || !rhs.range) {
    return std::nullopt;
  }
  switch (expression->kind) {
  case PTOTypedExpr::Kind::Add:
    return computeAddRange(*lhs.range, *rhs.range, expression->type,
                           isUnsigned);
  case PTOTypedExpr::Kind::Sub:
    return computeSubRange(*lhs.range, *rhs.range, expression->type,
                           isUnsigned);
  case PTOTypedExpr::Kind::Mul:
    return computeMulRange(*lhs.range, *rhs.range, expression->type,
                           isUnsigned);
  default:
    return std::nullopt;
  }
}

static PTOAnalysisResult<PointExpressionProof>
analyzePointExpression(const PTOTypedExprRef &expression, bool isUnsigned) {
  if (!expression) {
    return PTOAnalysisResult<PointExpressionProof>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }
  if (auto constant = foldPTOConstant(expression)) {
    PTOTypedExprRef folded = makePTOConstantExpr(*constant, expression->type);
    return PTOAnalysisResult<PointExpressionProof>::known(
        {folded, PTOFiniteRange{folded->constant, folded->constant,
                                isUnsigned}});
  }
  if (expression->kind == PTOTypedExpr::Kind::Opaque) {
    return PTOAnalysisResult<PointExpressionProof>::known(
        {expression, getFullPointRange(expression->type, isUnsigned)});
  }

  if (expression->kind == PTOTypedExpr::Kind::Cast) {
    bool sourceIsUnsigned =
        expression->castKind == PTOCastKind::IndexCastUI ||
        expression->castKind == PTOCastKind::ExtUI;
    auto input = analyzePointExpression(expression->lhs, sourceIsUnsigned);
    if (!input) {
      return input;
    }
    std::optional<PTOFiniteRange> resultRange;
    if (input.value->range) {
      auto sourceRange =
          getMathematicalRange(*input.value->range, sourceIsUnsigned);
      if (sourceRange) {
        resultRange = makeProvenRange(sourceRange->minimum,
                                      sourceRange->maximum, expression->type,
                                      isUnsigned);
      }
    }
    bool proven = staticallyPreservesCastValue(
                      expression->castKind, expression->lhs->type,
                      expression->type) ||
                  resultRange.has_value();
    if (expression->sourceValue && !proven) {
      return PTOAnalysisResult<PointExpressionProof>::known(
          {makePTOOpaqueExpr(expression->sourceValue),
           getFullPointRange(expression->type, isUnsigned)});
    }
    if (!proven) {
      return PTOAnalysisResult<PointExpressionProof>::unknown(
          PTOAnalysisUnknownReason::RangeUnavailable);
    }
    PTOTypedExprRef result = makePTOCastExpr(
        expression->castKind, input.value->expression, expression->type,
        expression->sourceOperation);
    if (!resultRange) {
      resultRange = getFullPointRange(expression->type, isUnsigned);
    }
    return PTOAnalysisResult<PointExpressionProof>::known(
        {std::move(result), std::move(resultRange)});
  }

  Operation *source = expression->sourceValue
                          ? expression->sourceValue.getDefiningOp()
                          : nullptr;
  bool sourceProvesNoWrap =
      operationProvesNoWrap(source, isUnsigned);

  auto lhs = analyzePointExpression(expression->lhs, isUnsigned);
  auto rhs = analyzePointExpression(expression->rhs, isUnsigned);
  if (!lhs || !rhs) {
    return PTOAnalysisResult<PointExpressionProof>::unknown(
        !lhs ? lhs.reason : rhs.reason);
  }
  std::optional<PTOFiniteRange> provenRange;
  if (!sourceProvesNoWrap) {
    provenRange = computePointBinaryRange(
        expression, *lhs.value, *rhs.value, isUnsigned);
  }
  if (expression->sourceValue && !sourceProvesNoWrap && !provenRange) {
    return PTOAnalysisResult<PointExpressionProof>::known(
        {makePTOOpaqueExpr(expression->sourceValue),
         getFullPointRange(expression->type, isUnsigned)});
  }

  PTOTypedExprRef result;
  if (expression->kind == PTOTypedExpr::Kind::Add) {
    result = makePTOAddExpr(lhs.value->expression, rhs.value->expression,
                            expression->type);
  } else if (expression->kind == PTOTypedExpr::Kind::Sub) {
    result = makePTOSubExpr(lhs.value->expression, rhs.value->expression,
                            expression->type);
  } else if (expression->kind == PTOTypedExpr::Kind::Mul) {
    result = makePTOMulExpr(lhs.value->expression, rhs.value->expression,
                            expression->type);
  } else {
    return PTOAnalysisResult<PointExpressionProof>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }
  if (sourceProvesNoWrap) {
    provenRange = getFullPointRange(expression->type, isUnsigned);
  }
  return PTOAnalysisResult<PointExpressionProof>::known(
      {std::move(result), std::move(provenRange)});
}

} // namespace

PTOValueEvolutionAnalysis::PTOValueEvolutionAnalysis(func::FuncOp func)
    : func(func) {}

PTOValueEvolutionAnalysis::PTOValueEvolutionAnalysis(Operation *operation)
    : PTOValueEvolutionAnalysis(cast<func::FuncOp>(operation)) {}

PTOValueEvolutionAnalysis::PTOValueEvolutionAnalysis(func::FuncOp func,
                                                     AnalysisManager &)
    : PTOValueEvolutionAnalysis(func) {}

PTOTypedExprRef PTOValueEvolutionAnalysis::getExpr(Value value) {
  if (!value) {
    return makePTOConstantExpr(0);
  }
  if (auto cached = expressionCache.find(value);
      cached != expressionCache.end()) {
    return cached->second;
  }

  PTOTypedExprRef expression;
  if (auto constant = getConstantAPInt(value)) {
    auto node = std::make_shared<PTOTypedExpr>();
    node->kind = PTOTypedExpr::Kind::Constant;
    node->type = value.getType();
    node->constant = *constant;
    node->sourceValue = value;
    expression = node;
  } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    expression = makePTOAddExpr(getExpr(add.getLhs()), getExpr(add.getRhs()),
                                value.getType(), value);
  } else if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    expression = makePTOSubExpr(getExpr(sub.getLhs()), getExpr(sub.getRhs()),
                                value.getType(), value);
  } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    expression = makePTOMulExpr(getExpr(mul.getLhs()), getExpr(mul.getRhs()),
                                value.getType(), value);
  } else if (Operation *definition = value.getDefiningOp();
             isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp,
                             arith::TruncIOp, arith::ExtSIOp,
                             arith::ExtUIOp>(definition)) {
    expression = makePTOCastExpr(getCastKind(definition),
                                 getExpr(definition->getOperand(0)),
                                 value.getType(), definition, value);
  } else {
    expression = makePTOOpaqueExpr(value);
  }
  expressionCache.try_emplace(value, expression);
  return expression;
}

PTOAnalysisResult<PTOLoopEvolution>
PTOValueEvolutionAnalysis::getEvolution(Value value, scf::ForOp loop) {
  return getEvolutionImpl(value, loop, Interpretation::Signed);
}

PTOAnalysisResult<PTOLoopEvolution>
PTOValueEvolutionAnalysis::getEvolution(const PTOTypedExprRef &expression,
                                        scf::ForOp loop) {
  return getSyntheticEvolutionImpl(expression, loop, Interpretation::Signed);
}

PTOAnalysisResult<PTOTypedExprRef>
PTOValueEvolutionAnalysis::getPointExpression(
    const PTOTypedExprRef &expression) {
  return getPointExpressionImpl(expression);
}

PTOAnalysisResult<PTOTypedExprRef>
PTOValueEvolutionAnalysis::getPointExpressionImpl(
    const PTOTypedExprRef &expression) {
  auto proof = analyzePointExpression(expression, false);
  if (!proof) {
    return PTOAnalysisResult<PTOTypedExprRef>::unknown(proof.reason);
  }
  return PTOAnalysisResult<PTOTypedExprRef>::known(
      std::move(proof.value->expression));
}

PTOAnalysisResult<PTOLoopEvolution>
PTOValueEvolutionAnalysis::getSyntheticEvolutionImpl(
    const PTOTypedExprRef &expression, scf::ForOp loop,
    Interpretation interpretation, bool sourceProvesNoWrap) {
  if (!expression) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }
  if (expression->sourceValue) {
    return getEvolutionImpl(expression->sourceValue, loop, interpretation);
  }

  bool isUnsigned = interpretation == Interpretation::Unsigned;
  unsigned width = getIntegerLikeWidth(expression->type);
  if (expression->kind == PTOTypedExpr::Kind::Constant) {
    PTOFiniteRange range{expression->constant, expression->constant,
                         isUnsigned};
    return PTOAnalysisResult<PTOLoopEvolution>::known(
        {expression, makePTOConstantExpr(0, expression->type), 0, range,
         width != 0, true, int64_t{0}});
  }
  if (expression->kind == PTOTypedExpr::Kind::Opaque) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }

  if (expression->kind == PTOTypedExpr::Kind::Cast) {
    if (expression->castKind == PTOCastKind::TruncI) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::UnsupportedCast);
    }
    bool sourceIsUnsigned =
        expression->castKind == PTOCastKind::IndexCastUI ||
        expression->castKind == PTOCastKind::ExtUI;
    Interpretation sourceInterpretation =
        sourceIsUnsigned ? Interpretation::Unsigned : Interpretation::Signed;
    auto source = getSyntheticEvolutionImpl(expression->lhs, loop,
                                            sourceInterpretation);
    if (!source) {
      return source;
    }
    if (!source.value->noWrap) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::PossibleWrap);
    }
    unsigned sourceWidth =
        getIntegerLikeWidth(expression->lhs ? expression->lhs->type : Type());
    if (width == 0 || sourceWidth == 0) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::TypeMismatch);
    }
    bool staticallyPreserves = staticallyPreservesCastValue(
        expression->castKind, expression->lhs->type, expression->type);
    if (!source.value->rangeKnown && !staticallyPreserves) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::RangeUnavailable);
    }

    const PTOFiniteRange &sourceRange = source.value->range;
    __int128 minimum = 0;
    __int128 maximum = 0;
    if (source.value->rangeKnown) {
      minimum = sourceRange.unsignedInterpretation
                    ? static_cast<__int128>(
                          sourceRange.lowerInclusive.getZExtValue())
                    : static_cast<__int128>(
                          sourceRange.lowerInclusive.getSExtValue());
      maximum = sourceRange.unsignedInterpretation
                    ? static_cast<__int128>(
                          sourceRange.upperInclusive.getZExtValue())
                    : static_cast<__int128>(
                          sourceRange.upperInclusive.getSExtValue());
    }
    bool preserves = staticallyPreserves ||
                     (sourceInterpretation == Interpretation::Unsigned
                          ? fitsUnsigned(minimum, width) &&
                                fitsUnsigned(maximum, width)
                          : fitsSigned(minimum, width) &&
                                fitsSigned(maximum, width));
    if (!preserves) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::RangeUnavailable);
    }

    PTOLoopEvolution result = *source.value;
    result.initial = makePTOCastExpr(
        expression->castKind, result.initial, expression->type,
        expression->sourceOperation);
    result.step =
        result.constantStep
            ? makePTOConstantExpr(*result.constantStep, expression->type)
            : makePTOCastExpr(expression->castKind, result.step,
                              expression->type,
                              expression->sourceOperation);
    if (source.value->rangeKnown) {
      result.range = makeRange(
          minimum, maximum, width,
          sourceInterpretation == Interpretation::Unsigned);
    }
    result.rangeKnown = source.value->rangeKnown;
    result.noWrap = true;
    return PTOAnalysisResult<PTOLoopEvolution>::known(std::move(result));
  }

  auto lhs = getSyntheticEvolutionImpl(expression->lhs, loop,
                                       interpretation);
  auto rhs = getSyntheticEvolutionImpl(expression->rhs, loop,
                                       interpretation);
  if (!lhs || !rhs) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        !lhs ? lhs.reason : rhs.reason);
  }

  if (expression->kind == PTOTypedExpr::Kind::Add ||
      expression->kind == PTOTypedExpr::Kind::Sub) {
    std::optional<PTOFiniteRange> provenRange;
    bool canComputeRange = !sourceProvesNoWrap &&
                           expression->type.isIndex() &&
                           lhs.value->noWrap && rhs.value->noWrap &&
                           lhs.value->rangeKnown && rhs.value->rangeKnown;
    if (canComputeRange) {
      provenRange = expression->kind == PTOTypedExpr::Kind::Add
                        ? computeAddRange(lhs.value->range, rhs.value->range,
                                          expression->type, isUnsigned)
                        : computeSubRange(lhs.value->range, rhs.value->range,
                                          expression->type, isUnsigned);
    }
    if (!provenRange && !sourceProvesNoWrap) {
      return PTOAnalysisResult<PTOLoopEvolution>::unknown(
          PTOAnalysisUnknownReason::PossibleWrap);
    }
    PTOLoopEvolution result = *lhs.value;
    result.initial =
        expression->kind == PTOTypedExpr::Kind::Add
            ? makePTOAddExpr(lhs.value->initial, rhs.value->initial,
                             expression->type)
            : makePTOSubExpr(lhs.value->initial, rhs.value->initial,
                             expression->type);
    result.step = expression->kind == PTOTypedExpr::Kind::Add
                      ? makePTOAddExpr(lhs.value->step, rhs.value->step,
                                       expression->type)
                      : makePTOSubExpr(lhs.value->step, rhs.value->step,
                                       expression->type);
    if (lhs.value->constantStep && rhs.value->constantStep) {
      int64_t combined;
      bool overflow = expression->kind == PTOTypedExpr::Kind::Add
                          ? llvm::AddOverflow(*lhs.value->constantStep,
                                              *rhs.value->constantStep,
                                              combined)
                          : llvm::SubOverflow(*lhs.value->constantStep,
                                              *rhs.value->constantStep,
                                              combined);
      result.constantStep =
          overflow ? std::nullopt : std::optional<int64_t>(combined);
    } else {
      result.constantStep = std::nullopt;
    }
    if (provenRange) {
      result.range = *provenRange;
    }
    result.rangeKnown = provenRange.has_value();
    result.noWrap = true;
    return PTOAnalysisResult<PTOLoopEvolution>::known(std::move(result));
  }

  auto lhsConstant = getMathematicalConstant(expression->lhs, isUnsigned);
  auto rhsConstant = getMathematicalConstant(expression->rhs, isUnsigned);
  const PTOTypedExprRef *invariantExpression = nullptr;
  const PTOAnalysisResult<PTOLoopEvolution> *variantEvolution = nullptr;
  __int128 multiplier = 0;
  if (lhsConstant) {
    invariantExpression = &expression->lhs;
    variantEvolution = &rhs;
    multiplier = *lhsConstant;
  } else if (rhsConstant) {
    invariantExpression = &expression->rhs;
    variantEvolution = &lhs;
    multiplier = *rhsConstant;
  } else {
    bool hasInvariantOperand = lhs.value->constantStep == int64_t{0} ||
                               rhs.value->constantStep == int64_t{0};
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        hasInvariantOperand ? PTOAnalysisUnknownReason::RangeUnavailable
                            : PTOAnalysisUnknownReason::NonAffineRecurrence);
  }

  std::optional<PTOFiniteRange> provenRange;
  if (!sourceProvesNoWrap && variantEvolution->value->noWrap &&
      variantEvolution->value->rangeKnown) {
    provenRange = computeMulRange(variantEvolution->value->range, multiplier,
                                  expression->type, isUnsigned);
  }
  bool hasIndexType = expression->type.isIndex();
  if (!hasIndexType || (!provenRange && !sourceProvesNoWrap)) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::PossibleWrap);
  }

  PTOLoopEvolution result = *variantEvolution->value;
  result.initial = makePTOMulExpr(*invariantExpression, result.initial,
                                  expression->type);
  result.step = makePTOMulExpr(*invariantExpression, result.step,
                               expression->type);
  bool multiplierFitsInt64 =
      multiplier >= std::numeric_limits<int64_t>::min() &&
      multiplier <= std::numeric_limits<int64_t>::max();
  if (result.constantStep && multiplierFitsInt64) {
    int64_t combined;
    result.constantStep =
        llvm::MulOverflow(*result.constantStep,
                          static_cast<int64_t>(multiplier), combined)
            ? std::nullopt
            : std::optional<int64_t>(combined);
  } else {
    result.constantStep = std::nullopt;
  }
  if (provenRange) {
    result.range = *provenRange;
  }
  result.rangeKnown = provenRange.has_value();
  result.noWrap = true;
  return PTOAnalysisResult<PTOLoopEvolution>::known(std::move(result));
}

PTOAnalysisResult<PTOLoopEvolution>
PTOValueEvolutionAnalysis::getEvolutionImpl(Value value, scf::ForOp loop,
                                            Interpretation interpretation) {
  if (loop.isDefinedOutsideOfLoop(value)) {
    unsigned width = getIntegerLikeWidth(value.getType());
    bool isUnsigned = interpretation == Interpretation::Unsigned;
    PTOFiniteRange range{APInt(std::max(1U, width), 0),
                         APInt(std::max(1U, width), 0),
                         isUnsigned};
    bool rangeKnown = false;
    if (auto constant = getConstantAPInt(value)) {
      range = {*constant, *constant, isUnsigned};
      rangeKnown = true;
    }
    return PTOAnalysisResult<PTOLoopEvolution>::known(
        {getExpr(value), makePTOConstantExpr(0, value.getType()), 0, range,
         rangeKnown, true, int64_t{0}});
  }

  if (Operation *cast = value.getDefiningOp();
      isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp,
                      arith::TruncIOp, arith::ExtSIOp,
                      arith::ExtUIOp>(cast)) {
    return getSyntheticEvolutionImpl(
        makePTOCastExpr(getCastKind(cast), getExpr(cast->getOperand(0)),
                        value.getType(), cast),
        loop, interpretation);
  }

  std::optional<uint64_t> tripCount = getConstantTripCount(loop);
  if (!tripCount) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::UnknownIterationDomain);
  }

  Value initialValue;
  Value stepValue;
  bool negateStep = false;
  if (value == loop.getInductionVar()) {
    initialValue = loop.getLowerBound();
    stepValue = loop.getStep();
  } else if (auto iterArg = dyn_cast<BlockArgument>(value);
             iterArg && iterArg.getOwner() == loop.getBody() &&
             iterArg.getArgNumber() > 0) {
    unsigned index = iterArg.getArgNumber() - 1;
    initialValue = loop.getInitArgs()[index];
    auto recurrence = getDirectRecurrenceStep(iterArg, loop);
    if (!recurrence) {
      return analyzeDynamicRecurrence(
          iterArg, loop, *tripCount,
          interpretation == Interpretation::Unsigned, *this);
    }
    stepValue = recurrence->first;
    negateStep = recurrence->second;
  } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    return getSyntheticEvolutionImpl(
        makePTOAddExpr(getExpr(add.getLhs()), getExpr(add.getRhs()),
                       value.getType()),
        loop, interpretation,
        operationProvesNoWrap(add, interpretation == Interpretation::Unsigned));
  } else if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    return getSyntheticEvolutionImpl(
        makePTOSubExpr(getExpr(sub.getLhs()), getExpr(sub.getRhs()),
                       value.getType()),
        loop, interpretation,
        operationProvesNoWrap(sub, interpretation == Interpretation::Unsigned));
  } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    return getSyntheticEvolutionImpl(
        makePTOMulExpr(getExpr(mul.getLhs()), getExpr(mul.getRhs()),
                       value.getType()),
        loop, interpretation,
        operationProvesNoWrap(mul, interpretation == Interpretation::Unsigned));
  } else {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::UnsupportedOperation);
  }

  unsigned width = getIntegerLikeWidth(value.getType());
  if (width == 0 || width > 64) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::TypeMismatch);
  }
  bool isUnsigned = interpretation == Interpretation::Unsigned;
  auto initial = getMathematicalConstant(initialValue, isUnsigned);
  auto step = stepValue ? getMathematicalConstant(stepValue, isUnsigned)
                        : std::optional<__int128>(0);
  if (!initial || !step) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::RangeUnavailable);
  }
  if (negateStep) {
    *step = -*step;
  }

  __int128 finalBackedge =
      *initial + *step * static_cast<__int128>(*tripCount);
  __int128 finalAccess = *tripCount == 0
                             ? *initial
                             : *initial +
                                   *step * static_cast<__int128>(*tripCount - 1);
  __int128 minimum = std::min(*initial, std::min(finalAccess, finalBackedge));
  __int128 maximum = std::max(*initial, std::max(finalAccess, finalBackedge));
  bool noWrap = isUnsigned ? fitsUnsigned(minimum, width) &&
                                 fitsUnsigned(maximum, width)
                           : fitsSigned(minimum, width) &&
                                 fitsSigned(maximum, width);
  if (!noWrap) {
    return PTOAnalysisResult<PTOLoopEvolution>::unknown(
        PTOAnalysisUnknownReason::PossibleWrap);
  }

  Type expressionType = value.getType();
  PTOTypedExprRef stepExpression =
      stepValue ? getExpr(stepValue) : makePTOConstantExpr(0, expressionType);
  if (negateStep) {
    stepExpression = makePTOSubExpr(makePTOConstantExpr(0, expressionType),
                                    stepExpression, expressionType);
  }
  return PTOAnalysisResult<PTOLoopEvolution>::known(
      {getExpr(initialValue), stepExpression, *tripCount,
       makeRange(minimum, maximum, width, isUnsigned), true, true,
       *step >= std::numeric_limits<int64_t>::min() &&
               *step <= std::numeric_limits<int64_t>::max()
           ? std::optional<int64_t>(static_cast<int64_t>(*step))
           : std::nullopt});
}

PTOAnalysisResult<PTOFiniteRange>
PTOValueEvolutionAnalysis::getRange(Value value, scf::ForOp loop) {
  auto evolution = getEvolution(value, loop);
  if (!evolution) {
    return PTOAnalysisResult<PTOFiniteRange>::unknown(evolution.reason);
  }
  if (!evolution.value->rangeKnown) {
    return PTOAnalysisResult<PTOFiniteRange>::unknown(
        PTOAnalysisUnknownReason::RangeUnavailable);
  }
  return PTOAnalysisResult<PTOFiniteRange>::known(evolution.value->range);
}

PTOAnalysisResult<bool>
PTOValueEvolutionAnalysis::preservesValue(Operation *castOp,
                                          scf::ForOp loop) {
  if (!isa<arith::IndexCastOp, arith::IndexCastUIOp>(castOp)) {
    return PTOAnalysisResult<bool>::unknown(
        PTOAnalysisUnknownReason::UnsupportedCast);
  }
  Value input = castOp->getOperand(0);
  if (
      loop.isDefinedOutsideOfLoop(input) && !getConstantAPInt(input)) {
    return PTOAnalysisResult<bool>::unknown(
        PTOAnalysisUnknownReason::RangeUnavailable);
  }
  auto evolution = getEvolution(castOp->getResult(0), loop);
  if (!evolution) {
    return PTOAnalysisResult<bool>::unknown(evolution.reason);
  }
  return PTOAnalysisResult<bool>::known(true);
}

PTOAnalysisResult<bool>
PTOValueEvolutionAnalysis::doesNotWrap(Value value, scf::ForOp loop) {
  auto evolution = getEvolution(value, loop);
  if (!evolution) {
    return PTOAnalysisResult<bool>::unknown(evolution.reason);
  }
  return PTOAnalysisResult<bool>::known(evolution.value->noWrap);
}
