// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVusqz.cpp - pto.Vusqz methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VusqzOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getSrc().getType() != getResult().getType()) {
    return emitOpError("requires src and result to share one vector type");
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  auto elemType = dyn_cast<IntegerType>(srcType.getElementType());
  if (!elemType) {
    return emitOpError("requires signed integer vector element type");
  }
  if (elemType.isUnsigned()) {
    return emitOpError("requires signed integer vector element type");
  }
  unsigned width = elemType.getWidth();
  if (width != mlir::pto::kValue8 && width != mlir::pto::kValue16 && width != mlir::pto::kValue32) {
    return emitOpError("requires s8/s16/s32 vector element type");
  }
  return success();
}
