// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOValueEvolutionAnalysis.h - Typed scalar evolution ----*- C++ -*-===//
//
// This file owns the reusable, read-only mathematical model used by VPTO
// address consumers.  It deliberately knows nothing about memory operations
// or post-update instruction legality.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_ANALYSIS_PTOVALUEEVOLUTIONANALYSIS_H
#define PTO_ANALYSIS_PTOVALUEEVOLUTIONANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace mlir {

namespace pto {

enum class PTOAnalysisUnknownReason {
  None,
  UnsupportedOperation,
  UnsupportedCast,
  UnknownIterationDomain,
  RangeUnavailable,
  PossibleWrap,
  NonAffineRecurrence,
  UnknownElementSize,
  UnknownUnitSize,
  InexactUnitConversion,
  DifferentBase,
  TypeMismatch,
};

llvm::StringRef stringifyPTOAnalysisUnknownReason(
    PTOAnalysisUnknownReason reason);

template <typename T>
struct PTOAnalysisResult {
  std::optional<T> value;
  PTOAnalysisUnknownReason reason = PTOAnalysisUnknownReason::None;

  static PTOAnalysisResult known(T knownValue) {
    return {std::move(knownValue), PTOAnalysisUnknownReason::None};
  }

  static PTOAnalysisResult unknown(PTOAnalysisUnknownReason unknownReason) {
    return {std::nullopt, unknownReason};
  }

  explicit operator bool() const { return value.has_value(); }
};

enum class PTOCastKind {
  IndexCast,
  IndexCastUI,
  TruncI,
  ExtSI,
  ExtUI,
};

struct PTOTypedExpr;
using PTOTypedExprRef = std::shared_ptr<const PTOTypedExpr>;

/// A typed finite-width scalar expression. Arithmetic nodes keep the result
/// type from the source IR; synthetic scale constants may have a null type and
/// adapt only when a consumer proves an exact target representation.
///
/// A non-null `sourceValue` marks a node as the expression for that exact SSA
/// value. Consumers must preserve that finite-width operation boundary rather
/// than reassociate through the node. Nodes without a source value are
/// synthetic expressions assembled from already-proven components.
struct PTOTypedExpr {
  enum class Kind { Constant, Opaque, Add, Sub, Mul, Cast };

  Kind kind = Kind::Constant;
  Type type;
  APInt constant = APInt(1, 0);
  Value opaque;
  PTOCastKind castKind = PTOCastKind::IndexCast;
  Value sourceValue;
  Operation *sourceOperation = nullptr;
  PTOTypedExprRef lhs;
  PTOTypedExprRef rhs;
};

struct PTOLinearTerm {
  PTOTypedExprRef atom;
  int64_t coefficient = 0;
};

struct PTOLinearExpr {
  int64_t constant = 0;
  SmallVector<PTOLinearTerm> terms;
};

PTOTypedExprRef makePTOConstantExpr(int64_t value, Type type = {});
PTOTypedExprRef makePTOOpaqueExpr(Value value);
PTOTypedExprRef makePTOAddExpr(PTOTypedExprRef lhs, PTOTypedExprRef rhs,
                               Type type = {}, Value sourceValue = {});
PTOTypedExprRef makePTOSubExpr(PTOTypedExprRef lhs, PTOTypedExprRef rhs,
                               Type type = {}, Value sourceValue = {});
PTOTypedExprRef makePTOMulExpr(PTOTypedExprRef lhs, PTOTypedExprRef rhs,
                               Type type = {}, Value sourceValue = {});
PTOTypedExprRef makePTOCastExpr(PTOCastKind kind, PTOTypedExprRef input,
                                Type resultType,
                                Operation *sourceOperation = nullptr,
                                Value sourceValue = {});

std::optional<int64_t> foldPTOConstant(const PTOTypedExprRef &expr);
std::optional<PTOLinearExpr> normalizePTOLinearExpr(
    const PTOTypedExprRef &expr);
bool equalPTOLinearExprs(const PTOLinearExpr &lhs,
                         const PTOLinearExpr &rhs);
bool dividePTOLinearExprExact(PTOLinearExpr &expr, int64_t divisor);
bool isZeroPTOLinearExpr(const PTOLinearExpr &expr);
PTOTypedExprRef buildPTOTypedExpr(const PTOLinearExpr &expr, Type type = {});
void collectPTOExprLeaves(const PTOTypedExprRef &expr,
                          SmallVectorImpl<Value> &leaves);
bool getPTOExprType(const PTOTypedExprRef &expr, Type &type);
void printPTOTypedExpr(const PTOTypedExprRef &expr, llvm::raw_ostream &os);

struct PTOFiniteRange {
  APInt lowerInclusive;
  APInt upperInclusive;
  bool unsignedInterpretation = false;
};

struct PTOLoopEvolution {
  PTOTypedExprRef initial;
  /// Mathematical delta from the current iteration to the next.  The
  /// expression may reference the induction variable or other iter_args and
  /// is evaluated in the current loop iteration.
  PTOTypedExprRef step;
  uint64_t tripCount = 0;
  PTOFiniteRange range;
  bool rangeKnown = false;
  bool noWrap = false;
  /// Mathematical per-iteration delta when it is representable as int64_t.
  /// This is distinct from the source bit pattern for unsigned recurrences.
  std::optional<int64_t> constantStep;
};

/// Function-scoped AnalysisManager analysis for typed scalar expressions,
/// ranges, cast preservation, and affine loop-carried scf.for evolution.
class PTOValueEvolutionAnalysis {
public:
  explicit PTOValueEvolutionAnalysis(Operation *operation);
  explicit PTOValueEvolutionAnalysis(func::FuncOp func);
  PTOValueEvolutionAnalysis(func::FuncOp func, AnalysisManager &);

  PTOTypedExprRef getExpr(Value value);

  /// Returns the evolution of a source SSA value under its actual typed,
  /// finite-width semantics, including range, cast, and no-wrap proofs.
  PTOAnalysisResult<PTOLoopEvolution> getEvolution(Value value,
                                                   scf::ForOp loop);

  /// Returns the evolution of a synthetic expression by composing proven
  /// child evolutions. If `expression` is source-backed, this is exactly the
  /// corresponding getEvolution(Value, loop) query and never a stronger
  /// recursive fallback.
  PTOAnalysisResult<PTOLoopEvolution>
  getEvolution(const PTOTypedExprRef &expression, scf::ForOp loop);

  /// Returns a point-value expression suitable for exact address-difference
  /// algebra. A matching no-wrap flag on the original SSA operation is the
  /// primary proof for source-backed arithmetic; without one, operand ranges
  /// must prove that the result fits. Casts are expanded only when their
  /// proven source range fits the result type. Unproven source operations
  /// remain opaque SSA atoms.
  PTOAnalysisResult<PTOTypedExprRef>
  getPointExpression(const PTOTypedExprRef &expression);

  PTOAnalysisResult<PTOFiniteRange> getRange(Value value, scf::ForOp loop);
  PTOAnalysisResult<bool> preservesValue(Operation *castOp,
                                         scf::ForOp loop);
  PTOAnalysisResult<bool> doesNotWrap(Value value, scf::ForOp loop);

private:
  enum class Interpretation { Signed, Unsigned };

  PTOAnalysisResult<PTOLoopEvolution>
  getEvolutionImpl(Value value, scf::ForOp loop, Interpretation interpretation);
  PTOAnalysisResult<PTOLoopEvolution>
  getSyntheticEvolutionImpl(const PTOTypedExprRef &expression,
                            scf::ForOp loop,
                            Interpretation interpretation,
                            bool sourceProvesNoWrap = false);
  PTOAnalysisResult<PTOTypedExprRef>
  getPointExpressionImpl(const PTOTypedExprRef &expression);

  func::FuncOp func;
  DenseMap<Value, PTOTypedExprRef> expressionCache;
};

} // namespace pto
} // namespace mlir

#endif // PTO_ANALYSIS_PTOVALUEEVOLUTIONANALYSIS_H
