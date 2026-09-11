// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVlds.cpp - pto.vlds verification -------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"
#include "VPTODmaInternal.h"

using namespace mlir;
using namespace mlir::pto;

static bool isSupportedVldsDistToken(StringRef dist, Type elementType) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::Load, dist,
                              getDmaDistElementWidth(elementType));
}

void VldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

template <typename LoadOp>
static LogicalResult verifyVldsCommon(LoadOp op) {
  if (!isBufferLike(op.getSource().getType())) {
    return op.emitOpError("requires a pointer-like source");
  }

  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed source");
  }

  if (op.getDistAttr()) {
    StringRef dist = *op.getDist();
    Type elementType =
        cast<VRegType>(op.getResult().getType()).getElementType();
    if (!isSupportedVldsDistToken(dist, elementType)) {
      return op.emitOpError(
          "supports only NORM, BRC_B8/B16/B32, US_B8/B16, DS_B8/B16, "
          "UNPK_B8/B16/B32, BRC_BLK, E2B_B16/B32, UNPK4, SPLT4CHN, and "
          "SPLT2CHN_B8/B16 load distributions");
    }
  }

  return success();
}

LogicalResult VldsOp::verify() {
  if (failed(verifyVldsCommon(*this))) {
    return failure();
  }
  if (Value updatedBase = getUpdatedBase()) {
    if (updatedBase.getType() != getSource().getType()) {
      return emitOpError("requires updated base result to match base type");
    }
  }
  return success();
}
