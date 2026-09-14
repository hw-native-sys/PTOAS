// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOPstu.cpp - pto.pstu align-chain role ---------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOAlignInternal.h"

using namespace mlir;
using namespace mlir::pto;

// pto.pstu advances a store align chain: backward through align_in, forward
// through align_out; it is also an accepted store-chain root.
bool isPstuStoreRoot(Operation *def) {
  return isa<PstuOp>(def);
}

std::optional<Value> pstuStoreStateIn(Operation *def) {
  if (auto stateOp = dyn_cast<PstuOp>(def)) {
    return stateOp.getAlignIn();
  }
  return std::nullopt;
}

std::optional<Value> pstuStoreStateOut(Operation *def) {
  if (auto stateOp = dyn_cast<PstuOp>(def)) {
    return stateOp.getAlignOut();
  }
  return std::nullopt;
}
