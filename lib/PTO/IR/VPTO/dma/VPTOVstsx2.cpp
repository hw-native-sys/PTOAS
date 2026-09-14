// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVstsx2.cpp - pto.vstsx2 verification ---------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

static bool isSupportedVstsx2DistToken(StringRef dist) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::StoreX2, dist);
}

void Vstsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLowMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getHighMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vstsx2Op::verify() {
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low value type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (getLow().getType() != getHigh().getType()) {
    return emitOpError("requires low/high values to share one vector type");
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (!isSupportedVstsx2DistToken(getDist())) {
    return emitOpError("requires a supported x2 store distribution token");
  }
  return success();
}
