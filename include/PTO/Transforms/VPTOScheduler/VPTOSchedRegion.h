// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedRegion.h - VPTO scheduling regions ------------*- C++ -*-===//
//
// A scheduling region is a contiguous, block-local sequence which can be
// analyzed independently. Boundary operations are never members of a region.
// Structural operations are retained alongside schedulable instructions so
// their SSA dependencies remain visible to later DAG construction. Values
// live across a region without being referenced inside it are recorded so
// pressure tracking includes the surrounding block and loop context.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDREGION_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDREGION_H

#include "PTO/IR/VPTOScheduling.h"

#include "mlir/IR/Block.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <array>
#include <string>

namespace mlir {
class Liveness;
}

namespace mlir::pto {

struct VPTOSchedulingCoverage {
  std::array<unsigned, kNumVPTOSchedulingClasses> classCounts{};
  llvm::StringMap<unsigned> boundaryReasons;
  llvm::StringMap<unsigned> unsupportedOps;
  llvm::StringMap<unsigned> unclassifiedOps;

  void record(Operation *op, const VPTOSchedulingSemantics &semantics);
  unsigned getCount(VPTOSchedulingClass schedulingClass) const;
  unsigned getUnclassifiedCount() const;
};

struct VPTOSchedRegion {
  Block *block = nullptr;
  unsigned index = 0;
  SmallVector<Operation *> operations;
  Operation *precedingBoundary = nullptr;
  Operation *followingBoundary = nullptr;
  std::string precedingBoundaryReason;
  std::string followingBoundaryReason;
  SmallVector<Value> liveThroughs;
};

class VPTOSchedRegionBuilder {
public:
  explicit VPTOSchedRegionBuilder(VPTOSchedulingCoverage *coverage = nullptr,
                                  const Liveness *liveness = nullptr)
      : coverage(coverage), liveness(liveness) {}

  SmallVector<VPTOSchedRegion> build(Block &block) const;

private:
  VPTOSchedulingCoverage *coverage;
  const Liveness *liveness;
};

std::string getVPTOSchedulingBoundaryReason(Operation *op);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDREGION_H
