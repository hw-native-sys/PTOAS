// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVgatherb.cpp - pto.Vgatherb methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void VgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VgatherbOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }

  if (failed(verifyMaskTypeWithGranularityLike(getOperation(), getMask().getType(),
                                               "mask type", "b32"))) {
    return failure();
  }

  VRegType offsetsType, resultType;
  IntegerType offsetsElemType;
  if (failed(verifyGatherOffsetTypesAndWidth(*this, offsetsType, resultType,
                                             offsetsElemType))) {
    return failure();
  }
  // vgatherb is a 32-byte block gather: each offset addresses one 32-byte block.
  // The offset vector holds VL/32 block addresses (always ui32), while the
  // result vector holds VL/sizeof(T) elements of the data type.  These counts
  // only coincide when sizeof(T)==4 (e.g. f32/i32/ui32).  For smaller types
  // the result has more elements than the offset, which is correct because the
  // hardware interprets the low VL/32 bytes of the offset register as block
  // addresses and gathers VL bytes of data per invocation.
  return success();
}
