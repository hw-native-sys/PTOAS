// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVexpdif.cpp - pto.Vexpdif methods ------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VexpdifOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyVRegTypeLike(*this, getMax().getType(), "max type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }

  auto inputType = cast<VRegType>(getInput().getType());
  auto maxType = cast<VRegType>(getMax().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (inputType != maxType) {
    return emitOpError("requires input and max to share one vector type");
  }

  Type inputElemType = inputType.getElementType();
  if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return emitOpError("requires f16 or f32 input vector element type");
  }
  auto expectedGranularity = getVdupMaskGranularity(inputElemType);
  if (!expectedGranularity) {
    return emitOpError("requires input element type with supported predicate granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type",
                                               *expectedGranularity))) {
    return failure();
  }
  if (!resultType.getElementType().isF32()) {
    return emitOpError("requires f32 result vector element type");
  }

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits) {
    return emitOpError(
        "requires source and result to preserve total vector storage width");
  }

  if (auto part = getPart()) {
    if (*part != "EVEN" && *part != "ODD") {
      return emitOpError("part must be EVEN or ODD");
    }
    return success();
  }

  // An f16 source packs two elements per 32-bit lane, so one instruction only
  // consumes the half selected by `part`; require an explicit choice. An f32
  // source covers the whole vector with a single instruction and the hardware
  // contract value is not observable in the result, so `part` may be omitted.
  if (inputElemType.isF16()) {
    return emitOpError(
        "requires part (EVEN or ODD) for f16 input: one instruction consumes "
        "a single 16-bit half of every 32-bit lane");
  }
  return success();
}
