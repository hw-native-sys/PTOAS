// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARPLACEMENT_H
#define PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARPLACEMENT_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarAnalysis.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::pto::vecscopemembar {

enum class BarrierAnchorKind {
  BeforeOperation,
  BeforeLoop,
  AfterLoop,
  BeforeLoopTerminator,
};

struct BarrierPlacement {
  BarrierAnchorKind anchorKind = BarrierAnchorKind::BeforeOperation;
  Operation *anchor = nullptr;
  MemBarKind kind = MemBarKind::VV_ALL;
  SmallVector<unsigned> resolvedHazards;
};

struct VecScopeMemBarPlan {
  SmallVector<BarrierPlacement> barriers;
};

FailureOr<VecScopeMemBarPlan>
solveVecScopeMemBarPlacement(const VecScopeMemBarAnalysisResult &result);

} // namespace mlir::pto::vecscopemembar

#endif // PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARPLACEMENT_H
