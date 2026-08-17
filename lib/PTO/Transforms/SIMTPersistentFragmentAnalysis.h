// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_LIB_PTO_TRANSFORMS_SIMTPERSISTENTFRAGMENTANALYSIS_H
#define PTO_LIB_PTO_TRANSFORMS_SIMTPERSISTENTFRAGMENTANALYSIS_H

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

struct AccessLane {
  Operation *op;
  // Index of this scalar component in the original vector operation. Scalar
  // accesses use lane 0. The owning ResidentElementPlan supplies the element
  // offset.
  unsigned laneIndex;
};

struct ResidentElementPlan {
  int64_t elementOffset;
  // Assigned by function-wide slot allocation before the plan is published.
  unsigned slot = 0;
  SmallVector<AccessLane> accesses;
};

struct PersistentFragmentAnalysis {
  explicit PersistentFragmentAnalysis(LLVM::AllocaOp allocaOp)
      : allocaOp(allocaOp) {}

  LLVM::AllocaOp allocaOp;
  // The unique section that initializes all resident elements.
  SectionSimtOp initSection;
  // All sections dominated by initSection that must carry the complete
  // resident set. The sections are kept in function walk order and do not
  // include initSection.
  SmallVector<SectionSimtOp> carrySections;
  // Elements initialized by the init section, sorted by element offset.
  SmallVector<ResidentElementPlan> residentElements;

  ResidentElementPlan *findResidentElement(int64_t elementOffset) {
    for (ResidentElementPlan &element : residentElements) {
      if (element.elementOffset == elementOffset) {
        return &element;
      }
    }
    return nullptr;
  }
};

struct PersistentMaterializationPlan {
  // All inline SIMT sections in function walk order.
  SmallVector<SectionSimtOp> sections;
  // Persistent fragments in alloca walk order.
  SmallVector<PersistentFragmentAnalysis, 1> fragments;
};

/// Cached read-only analysis consumed by persistent fragment materialization.
class SIMTPersistentFragmentAnalysis {
public:
  explicit SIMTPersistentFragmentAnalysis(func::FuncOp func);

  bool isValid() const { return plan.has_value(); }

  const PersistentMaterializationPlan &getPlan() const {
    assert(plan && "expected valid persistent fragment analysis");
    return *plan;
  }

private:
  std::optional<PersistentMaterializationPlan> plan;
};

} // namespace pto
} // namespace mlir

#endif // PTO_LIB_PTO_TRANSFORMS_SIMTPERSISTENTFRAGMENTANALYSIS_H
