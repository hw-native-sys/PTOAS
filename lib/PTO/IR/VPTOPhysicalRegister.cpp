// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOPhysicalRegister.cpp - VPTO physical register views -----------===//

#include "PTO/IR/VPTOPhysicalRegister.h"

#include "PTO/IR/PTO.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::pto;

bool mlir::pto::isPhysicalRegisterView(Operation *op) {
  return isa_and_nonnull<VbitcastOp, PbitcastOp>(op);
}

Value mlir::pto::getPhysicalRegisterViewRoot(Value value) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (auto result = dyn_cast<OpResult>(value)) {
    Operation *owner = result.getOwner();
    if (!visited.insert(owner).second) {
      break;
    }
    if (auto view = dyn_cast<VbitcastOp>(owner)) {
      value = view.getInput();
      continue;
    }
    if (auto view = dyn_cast<PbitcastOp>(owner)) {
      value = view.getInput();
      continue;
    }
    break;
  }
  return value;
}
