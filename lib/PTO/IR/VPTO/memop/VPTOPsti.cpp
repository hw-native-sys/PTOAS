// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOPsti.cpp - pto.Psti methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void PstiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult PstiOp::verify() {
  return verifyPredicateStoreOp(
      *this, [&]() -> LogicalResult {
        if (!matchPattern(getOffset(), m_Constant())) {
          return emitOpError("requires offset to be a constant index immediate");
        }
        return success();
      });
}
