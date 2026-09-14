// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOUvld.cpp - pto.Uvld methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void UvldOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult UvldOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a buffer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }

  auto sourceMemRef = dyn_cast<BaseMemRefType>(getSource().getType());
  if (!sourceMemRef) {
    return success();
  }

  Type sourceElementType = sourceMemRef.getElementType();
  Type vectorElementType = cast<VRegType>(getResult().getType()).getElementType();
  if (sourceElementType != vectorElementType) {
    return emitOpError(
        "requires source element type to match vector element type");
  }
  return success();
}
