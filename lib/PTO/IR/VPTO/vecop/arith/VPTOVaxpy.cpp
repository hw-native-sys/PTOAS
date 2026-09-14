// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVaxpy.cpp - pto.Vaxpy methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VaxpyOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  auto src0Type = cast<VRegType>(getSrc0().getType());
  auto src1Type = cast<VRegType>(getSrc1().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (src0Type != src1Type || src0Type != resultType) {
    return emitOpError("requires src0, src1, and result to share one vector type");
  }
  Type elemType = src0Type.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires f16 or f32 vector element type");
  }
  if (failed(verifyVdupMaskGranularityLike(*this, elemType))) {
    return failure();
  }
  if (getAlpha().getType() != elemType) {
    return emitOpError("requires alpha type to match vector element type");
  }
  return success();
}
