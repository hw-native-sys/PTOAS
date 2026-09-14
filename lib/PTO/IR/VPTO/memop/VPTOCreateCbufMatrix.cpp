// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOCreateCbufMatrix.cpp - pto.CreateCbufMatrix methods ------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

LogicalResult CreateCbufMatrixOp::verify() {
  auto ptrType = dyn_cast<pto::PtrType>(getDst().getType());
  if (!ptrType) {
    return emitOpError("requires a typed !pto.ptr destination");
  }
  const bool matDestination =
      ptrType.getMemorySpace().getAddressSpace() == pto::AddressSpace::MAT;
  if (!matDestination) {
    return emitOpError()
           << "requires a mat/l1 destination, got "
           << getAddressSpaceDiagnosticName(
                  ptrType.getMemorySpace().getAddressSpace());
  }
  Type elementType = ptrType.getElementType();
  auto integerType = dyn_cast<IntegerType>(elementType);
  const bool canonicalView =
      integerType && integerType.isUnsigned() &&
      (integerType.getWidth() == 16 || integerType.getWidth() == 32);
  if (!canonicalView) {
    return emitOpError()
           << "requires a ui16 or ui32 destination view, got "
              "element type "
           << elementType;
  }
  if (failed(verifyRawFillWordBits(getOperation(), getFillWordBits()))) {
    return failure();
  }
  const bool wordWidthMatches =
      static_cast<unsigned>(getFillWordBits()) == integerType.getWidth();
  if (!wordWidthMatches) {
    return emitOpError()
           << "fill_word_bits " << getFillWordBits()
           << " does not match the " << integerType.getWidth()
           << "-bit destination view";
  }
  return verifyRawFillGeometry(getOperation(), /*byteOffset=*/{},
                               getRepeatTimes(), getBlockNum_32b(),
                               getDstGap_32b());
}

void CreateCbufMatrixOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}
