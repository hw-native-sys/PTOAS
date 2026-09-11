// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOVstus.cpp - pto.vstus align-chain role -------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOAlignInternal.h"

using namespace mlir;
using namespace mlir::pto;

// pto.vstus advances a store align chain: backward through align_in, forward
// through align_out; it is also an accepted store-chain root.
bool isVstusStoreRoot(Operation *def) {
  return isa<VstusOp>(def);
}

std::optional<Value> vstusStoreStateIn(Operation *def) {
  if (auto stateOp = dyn_cast<VstusOp>(def)) {
    return stateOp.getAlignIn();
  }
  return std::nullopt;
}

std::optional<Value> vstusStoreStateOut(Operation *def) {
  if (auto stateOp = dyn_cast<VstusOp>(def)) {
    return stateOp.getAlignOut();
  }
  return std::nullopt;
}
