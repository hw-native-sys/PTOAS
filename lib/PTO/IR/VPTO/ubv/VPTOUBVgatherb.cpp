// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOUBVgatherb.cpp - pto.UBVgatherb methods ------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOUBVInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::ubv_detail;

void UBVgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVgatherbOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getOffset().getType()) ||
      !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getOffset().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}
