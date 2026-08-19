// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARIR_H
#define PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARIR_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::pto::vecscopemembar {

enum class VecScopeNodeKind {
  Sequence,
  Loop,
  Access,
  ExistingBarrier,
};

struct VecScopeLoopInfo {
  unsigned id = 0;
  scf::ForOp op;
  unsigned depth = 0;
  Value inductionVar;
  Value lowerBound;
  Value upperBound;
  Value step;
};

struct VecScopeScheduleNode {
  VecScopeNodeKind kind = VecScopeNodeKind::Sequence;
  Operation *op = nullptr;
  SmallVector<unsigned> children;
  std::optional<unsigned> parent;
  SmallVector<unsigned, 2> schedulePath;
};

bool isUBBackedType(Type type);

bool isUBVectorStore(Operation *op);

bool isUBVectorLoad(Operation *op);

bool isUBVectorMemoryOp(Operation *op);

SmallVector<Value, 2> getStoredValues(Operation *storeOp);

SmallVector<Value, 2> getLoadedValues(Operation *loadOp);

bool valueDependsOn(Value sink, Value target);

} // namespace mlir::pto::vecscopemembar

#endif // PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMBARIR_H
