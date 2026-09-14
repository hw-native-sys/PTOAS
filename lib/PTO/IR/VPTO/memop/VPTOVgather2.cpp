// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVgather2.cpp - pto.Vgather2 methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

void Vgather2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2Op::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  VRegType offsetsType, resultType;
  IntegerType offsetsElemType;
  if (failed(verifyGatherOffsetTypes(*this, offsetsType, resultType,
                                     offsetsElemType))) {
    return failure();
  }
  if (offsetsType.getElementCount() != resultType.getElementCount()) {
    return emitOpError("offset and result vectors must have the same element count");
  }
  Type sourceElemType = getBufferElementType(getSource().getType());
  Type resultElemType = resultType.getElementType();
  unsigned resultElemWidth = getPTOStorageElemBitWidth(resultElemType);
  unsigned expectedOffsetWidth = 0;
  StringRef expectedMaskGranularity;
  int64_t expectedLanes = 0;
  if (failed(resolveVgather2WidthInfo(*this, sourceElemType, resultElemType,
                                      expectedOffsetWidth,
                                      expectedMaskGranularity, expectedLanes))) {
    return failure();
  }
  if (resultElemWidth != mlir::pto::kValue16 && resultElemWidth != 32) {
    return emitOpError("result element type must be 16-bit or 32-bit");
  }
  if (resultType.getElementCount() != expectedLanes) {
    return emitOpError() << "expects result type "
                         << formatVRegType(expectedLanes, resultElemType);
  }
  if (!isUnsignedOrSignlessIntegerOfWidth(offsetsElemType, expectedOffsetWidth)) {
    return emitOpError() << "requires ui" << expectedOffsetWidth << "/i"
                         << expectedOffsetWidth << " offset vector elements";
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          getOperation(), getMask().getType(), "mask type",
          expectedMaskGranularity))) {
    return failure();
  }
  return success();
}
