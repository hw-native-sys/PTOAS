// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVstas.cpp - pto.Vstas methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void VstasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstasOp::verify() {
  if (failed(verifyStoreAlignChain(getValue(), *this, "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}
