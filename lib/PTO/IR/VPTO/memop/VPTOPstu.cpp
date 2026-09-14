// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOPstu.cpp - pto.Pstu methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void PstuOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult PstuOp::verify() {
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type"))) {
    return failure();
  }
  if (!isBufferLike(getBase().getType()) || !isBufferLike(getBaseOut().getType())) {
    return emitOpError("requires pointer-like base and base_out");
  }
  if (getBase().getType() != getBaseOut().getType()) {
    return emitOpError("requires base and base_out to have identical types");
  }
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed base");
  }
  auto baseType = cast<pto::PtrType>(getBase().getType());
  auto maskType = cast<pto::MaskType>(getValue().getType());
  auto elemType = dyn_cast<IntegerType>(baseType.getElementType());
  if (!elemType || elemType.isSigned() || (elemType.getWidth() != mlir::pto::kValue16 && elemType.getWidth() != 32)) {
    return emitOpError("requires ui16/ui32 UB base type");
  }
  if (maskType.isB16() && elemType.getWidth() != mlir::pto::kValue16) {
    return emitOpError("requires !pto.mask<b16> to pair with !pto.ptr<ui16, ub>");
  }
  if (maskType.isB32() && elemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires !pto.mask<b32> to pair with !pto.ptr<ui32, ub>");
  }
  return success();
}
