// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAddressAnalysis.h - Typed VPTO address analysis ------*- C++ -*-===//
//
// Address analysis composes PTOValueEvolutionAnalysis with op-provided current
// access semantics.  It preserves pointer provenance but deliberately makes no
// alias, disjointness, or memory-dependence claims.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_ANALYSIS_PTOADDRESSANALYSIS_H
#define PTO_ANALYSIS_PTOADDRESSANALYSIS_H

#include "PTO/Analysis/PTOValueEvolutionAnalysis.h"
#include "PTO/IR/VPTOAddressSemantics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
class AnalysisManager;

namespace pto {

struct PTOTypedAddressOffset {
  Value sourceValue;
  PTOTypedExprRef value;
  VPTOAddressUnit unit = VPTOAddressUnit::Element;
  std::optional<int64_t> unitBytes;
};

struct PTOAddressExpr {
  Value currentBase;
  Value rootOrBase;
  PTOTypedExprRef elementOffset;
  std::optional<PTOTypedAddressOffset> offset;
  int64_t elementBytes = 0;
};

/// Function-scoped AnalysisManager analysis that builds typed AddressExprs and
/// provides exact byte/unit deltas without materializing IR.
class PTOAddressAnalysis {
public:
  PTOAddressAnalysis(Operation *operation,
                     AnalysisManager &analysisManager);
  PTOAddressAnalysis(func::FuncOp func, AnalysisManager &analysisManager);
  PTOAddressAnalysis(func::FuncOp func,
                     PTOValueEvolutionAnalysis &valueEvolution);

  PTOAnalysisResult<SmallVector<PTOAddressExpr>>
  getAddresses(Operation *operation);

  PTOAnalysisResult<PTOTypedExprRef>
  getDeltaBytes(const PTOAddressExpr &address, scf::ForOp loop);

  PTOAnalysisResult<PTOTypedExprRef>
  convertDeltaToUnit(const PTOTypedExprRef &deltaBytes,
                     int64_t targetUnitBytes);

  PTOAnalysisResult<PTOTypedExprRef>
  getDeltaInUnit(const PTOAddressExpr &address, scf::ForOp loop,
                 int64_t targetUnitBytes);

  /// Returns the point-value byte difference between two addresses. This
  /// query has no loop domain, so source-backed finite-width arithmetic is
  /// reassociated only when ValueEvolution finds an operation-local no-wrap or
  /// value-preservation proof. Unproven source operations stay opaque.
  PTOAnalysisResult<PTOTypedExprRef>
  getDifferenceBytes(const PTOAddressExpr &from,
                     const PTOAddressExpr &to);

  PTOValueEvolutionAnalysis &getValueEvolution() { return valueEvolution; }

private:
  PTOAnalysisResult<PTOTypedExprRef> getPointerDelta(Value pointer,
                                                     scf::ForOp loop);
  PTOAnalysisResult<PTOTypedExprRef>
  getPointerDifference(const PTOAddressExpr &from,
                       const PTOAddressExpr &to);

  func::FuncOp func;
  PTOValueEvolutionAnalysis &valueEvolution;
};

} // namespace pto
} // namespace mlir

#endif // PTO_ANALYSIS_PTOADDRESSANALYSIS_H
