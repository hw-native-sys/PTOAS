// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMadMxRawOp.cpp - pto.mad_mx_raw verification -------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMadInternal.h"

using namespace mlir;
using namespace mlir::pto;

void MadMxRawOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult MadMxRawOp::verify() {
  return verifyMadMxCommon(*this, getLhs().getType(), getRhs().getType(),
                           getDst().getType());
}

bool MadMxRawOp::isMadMxFamily() { return true; }
bool MadMxRawOp::hasBiasOperand() { return false; }
Value MadMxRawOp::getBiasOrNull() { return {}; }
