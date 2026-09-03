// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- LoopUnrollUtils.h - shared loop-unroll hint utilities ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_LOOPUNROLLUTILS_H_
#define MLIR_DIALECT_PTO_TRANSFORMS_LOOPUNROLLUTILS_H_

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {
namespace pto {

/// Compute the constant trip count of *forOp*, or std::nullopt when any of
/// the bounds/step is not a compile-time constant, the step is not positive,
/// the loop never executes (ub <= lb), or the true count does not fit in
/// int64_t (such a loop cannot be unrolled natively anyway).  The count is
/// computed in uint64_t: the difference of two in-range int64_t bounds with
/// ub > lb is exact, so extreme bounds (e.g. lb = INT64_MIN, ub = INT64_MAX)
/// cannot trigger signed-overflow UB.
inline std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) {
    return std::nullopt;
  }
  uint64_t span = static_cast<uint64_t>(*ub) - static_cast<uint64_t>(*lb);
  uint64_t ustep = static_cast<uint64_t>(*step);
  // Ceiling division without an overflowing `span + step - 1`: the quotient
  // only increases by one when there is a remainder, and span / ustep is at
  // most span, so the addition itself cannot overflow.
  uint64_t count = span / ustep + ((span % ustep) != 0 ? 1 : 0);
  if (count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<int64_t>(count);
}

/// Validate the loop-unroll hint attributes on *forOp*.  Emits a hard error
/// for anything malformed: a wrongly typed attribute, an unknown `pto.unroll`
/// value, both attributes on one loop, or an out-of-contract factor.  Shared
/// by pto-unroll-loops and pto-promote-persistent-fragment-loops (which must
/// not silently overwrite a malformed hint with "full").
inline LogicalResult validateLoopUnrollHint(scf::ForOp forOp) {
  Attribute unrollRaw = forOp->getAttr(pto::kUnrollAttrName);
  Attribute factorRaw = forOp->getAttr(pto::kUnrollFactorAttrName);

  // Wrong attribute *types* must not slip through as "no hint": the typed
  // getters below would return null and the loop would silently keep a
  // malformed annotation all the way down the pipeline.
  auto unrollAttr = dyn_cast_if_present<StringAttr>(unrollRaw);
  if (unrollRaw && !unrollAttr) {
    forOp.emitError() << "'" << pto::kUnrollAttrName
                      << "' must be a string attribute, got " << unrollRaw;
    return failure();
  }
  auto factorAttr = dyn_cast_if_present<IntegerAttr>(factorRaw);
  if (factorRaw && !factorAttr) {
    forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                      << "' must be a signless i32 attribute, got "
                      << factorRaw;
    return failure();
  }

  if (unrollAttr && factorAttr) {
    forOp.emitError() << "'" << pto::kUnrollAttrName << "' and '"
                      << pto::kUnrollFactorAttrName
                      << "' are mutually exclusive on one loop";
    return failure();
  }

  StringRef unrollValue = unrollAttr ? unrollAttr.getValue() : "";
  if (unrollAttr && unrollValue != pto::kUnrollFullValue &&
      unrollValue != pto::kUnrollEnableValue) {
    forOp.emitError() << "unknown '" << pto::kUnrollAttrName << "' value '"
                      << unrollAttr.getValue()
                      << "'; expected \"full\" (native full unroll) or "
                         "\"enable\" (forwarded to the compiler's cost "
                         "model by pto-convert-scf-to-cf-with-loop-hints)";
    return failure();
  }

  if (factorAttr && !pto::isValidUnrollFactorAttr(factorAttr)) {
    if (!factorAttr.getType().isSignlessInteger(32)) {
      forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                        << "' must be a signless i32 attribute, got "
                        << factorAttr.getType();
    } else {
      forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                        << "' must be a positive integer, got "
                        << factorAttr.getInt();
    }
    return failure();
  }

  return success();
}

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_LOOPUNROLLUTILS_H_
