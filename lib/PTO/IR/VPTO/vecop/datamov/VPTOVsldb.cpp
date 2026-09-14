// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVsldb.cpp - pto.Vsldb methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

void VsldbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VsldbOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!getBlockStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires block_stride to be i16");
  }
  if (!getRepeatStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires repeat_stride to be i16");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}
