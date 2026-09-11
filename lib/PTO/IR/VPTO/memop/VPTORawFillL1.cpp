// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTORawFillL1.cpp - pto.RawFillL1 methods --------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

LogicalResult RawFillL1Op::verify() {
  if (failed(
          verifyRawFillDestination(getOperation(), getDst().getType(), "dst"))) {
    return failure();
  }
  if (failed(verifyRawFillWordBits(getOperation(), getFillWordBits()))) {
    return failure();
  }
  return verifyRawFillGeometry(getOperation(), getByteOffset(),
                               getRepeatTimes(), getBlockNum_32b(),
                               getDstGap_32b());
}

void RawFillL1Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}
