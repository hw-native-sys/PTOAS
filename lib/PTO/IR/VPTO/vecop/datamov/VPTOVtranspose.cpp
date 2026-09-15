// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVtranspose.cpp - pto.Vtranspose methods ------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VtransposeOp::verify() {
  const bool pointerLike = isBufferLike(getDestination().getType()) &&
                           isBufferLike(getSource().getType());
  if (!pointerLike) {
    return emitOpError("requires pointer-like destination and source");
  }

  const bool ubBacked =
      classifyMemoryRole(getDestination().getType()) == MemoryRole::UB &&
      classifyMemoryRole(getSource().getType()) == MemoryRole::UB;
  if (!ubBacked) {
    return emitOpError("requires UB-backed destination and source");
  }

  const bool samePointer = getDestination() == getSource();
  if (samePointer) {
    return emitOpError("requires distinct destination and source pointers");
  }

  auto destinationType = dyn_cast<pto::PtrType>(getDestination().getType());
  auto sourceType = dyn_cast<pto::PtrType>(getSource().getType());
  if (!destinationType || !sourceType) {
    return emitOpError("requires ptr-backed destination and source");
  }
  const bool matchingElements =
      destinationType.getElementType() == sourceType.getElementType();
  if (!matchingElements) {
    return emitOpError("requires destination and source element types to match");
  }

  Type elementType = sourceType.getElementType();
  // VTRANSPOSE moves 16-bit values bitwise. Signed and signless i16 use the
  // .s16 intrinsic; ui16 uses the .u16 intrinsic.
  const bool supportedElement = elementType.isInteger(16);
  if (!supportedElement) {
    return emitOpError("requires a 16-bit integer element type");
  }
  return verifyNotNestedInVecScope(*this, "pto.vtranspose");
}

void VtransposeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}
