// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOVldus.cpp - pto.vldus align-chain role -------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOAlignInternal.h"

using namespace mlir;
using namespace mlir::pto;

// pto.vldus advances a load align chain: backward through align, forward
// through updated_align; it is also an accepted load-chain root.
bool isVldusLoadRoot(Operation *def) {
  return isa<VldusOp>(def);
}

std::optional<Value> vldusLoadStateIn(Operation *def) {
  if (auto stateOp = dyn_cast<VldusOp>(def)) {
    return stateOp.getAlign();
  }
  return std::nullopt;
}

std::optional<Value> vldusLoadStateOut(Operation *def) {
  if (auto stateOp = dyn_cast<VldusOp>(def)) {
    return stateOp.getUpdatedAlign();
  }
  return std::nullopt;
}
