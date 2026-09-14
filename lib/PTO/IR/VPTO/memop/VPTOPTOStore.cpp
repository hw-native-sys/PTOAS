// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOPTOStore.cpp - pto.PTOStore methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

LogicalResult PTOStoreOp::verify() {
  if (failed(verifyVPTOScalarAccessTypes(getOperation(), getPtr().getType(),
                                         getValue().getType(), "store"))) {
    return failure();
  }
  return success();
}

void PTOStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
}
