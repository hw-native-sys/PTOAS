// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVbitcast.cpp - pto.Vbitcast methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VbitcastOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }

  auto getStorageBits = [](VRegType type) -> std::optional<int64_t> {
    Type elementType = type.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      return type.getElementCount() * static_cast<int64_t>(intType.getWidth());
    }
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      return type.getElementCount() *
             static_cast<int64_t>(floatType.getWidth());
}
    // Packed PTO element types (f8/hif8/f4x2/bf16x2/...) have a known storage
    // width even though they are not IntegerType/FloatType.
    unsigned packedBits = pto::getPTOStorageElemBitWidth(elementType);
    if (packedBits != 0) {
      return type.getElementCount() * static_cast<int64_t>(packedBits);
}
    return std::nullopt;
  };

  auto inputBits = getStorageBits(inputType);
  auto resultBits = getStorageBits(resultType);
  if (!inputBits || !resultBits) {
    return emitOpError("requires integer or floating-point vreg element type");
  }
  if (*inputBits != *resultBits) {
    return emitOpError("requires source and result vectors to carry the same "
                       "total number of bits");
  }

  return success();
}
