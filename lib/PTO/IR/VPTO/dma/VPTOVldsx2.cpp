// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVldsx2.cpp - pto.vldsx2 verification ---------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"
#include "VPTODmaInternal.h"

using namespace mlir;
using namespace mlir::pto;

static bool isSupportedVldx2DistToken(StringRef dist, Type elementType) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::LoadX2, dist,
                              getDmaDistElementWidth(elementType));
}

void Vldsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vldsx2Op::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (!getOffset().getType().isIndex()) {
    return emitOpError("requires index offset");
  }
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high result type"))) {
    return failure();
  }
  if (getLow().getType() != getHigh().getType()) {
    return emitOpError("requires low/high results to share one vector type");
  }
  Type elementType = cast<VRegType>(getLow().getType()).getElementType();
  if (!isSupportedVldx2DistToken(getDist(), elementType)) {
    return emitOpError("requires a supported x2 load distribution token");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}
