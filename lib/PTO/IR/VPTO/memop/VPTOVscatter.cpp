// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVscatter.cpp - pto.Vscatter methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void VscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VscatterOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto valueType = dyn_cast<VRegType>(getValue().getType());
  if (!offsetsType || !valueType) {
    return emitOpError("value and offsets must be !pto.vreg<...>");
  }
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType) {
    return emitOpError("offset vector must use integer element type");
  }
  unsigned valueElemWidth = getPTOStorageElemBitWidth(valueType.getElementType());
  if (valueElemWidth != mlir::pto::kValue8 && valueElemWidth != 16 && valueElemWidth != 32) {
    return emitOpError("requires 8-, 16-, or 32-bit value elements");
  }
  unsigned expectedOffsetWidth = valueElemWidth == 32 ? 32 : 16;
  if (offsetsElemType.getWidth() != expectedOffsetWidth) {
    return emitOpError() << "requires " << expectedOffsetWidth
                         << "-bit offset vector elements for "
                         << valueElemWidth << "-bit values";
  }
  int64_t expectedOffsetCount = valueElemWidth == 8
                                    ? valueType.getElementCount() / 2
                                    : valueType.getElementCount();
  if (offsetsType.getElementCount() != expectedOffsetCount) {
    return emitOpError() << "requires " << expectedOffsetCount
                         << " offsets for " << valueType.getElementCount()
                         << "x" << valueElemWidth << "-bit values";
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          *this, getMask().getType(), "mask type",
          valueElemWidth == mlir::pto::kValue32 ? "b32" : "b16"))) {
    return failure();
  }
  auto destinationType = cast<PtrType>(getDestination().getType());
  if (destinationType.getElementType() != valueType.getElementType()) {
    return emitOpError(
        "requires destination element type to match value element type");
  }
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  return success();
}
