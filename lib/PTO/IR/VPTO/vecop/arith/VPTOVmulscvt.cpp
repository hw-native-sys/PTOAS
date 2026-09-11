// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVmulscvt.cpp - pto.Vmulscvt methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VmulscvtOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }

  auto inputType = cast<VRegType>(getInput().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (!inputType.getElementType().isF32()) {
    return emitOpError("requires f32 input vector element type");
  }
  if (!resultType.getElementType().isF16()) {
    return emitOpError("requires f16 result vector element type");
  }

  auto scalarType = getScalar().getType();
  if (!scalarType.isF32()) {
    return emitOpError("requires f32 scalar operand");
  }

  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type", "b32"))) {
    return failure();
  }

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits) {
    return emitOpError(
        "requires source and result to preserve total vector storage width");
  }

  auto normalizedRnd = normalizeRoundModeToken(getRnd());
  if (!normalizedRnd) {
    return emitOpError("rnd must be one of R/A/F/C/Z/O");
  }
  if (*normalizedRnd != "A") {
    return emitOpError("currently only supports rnd A");
  }

  auto normalizedPart = normalizeEvenOddPartToken(getPart());
  if (!normalizedPart) {
    return emitOpError("part must be EVEN or ODD");
  }
  return success();
}
