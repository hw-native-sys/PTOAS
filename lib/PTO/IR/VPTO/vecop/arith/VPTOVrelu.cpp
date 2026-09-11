// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVrelu.cpp - pto.Vrelu methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VreluOp::verify() {
  if (failed(verifyUnaryVecOp(*this))) {
    return failure();
  }
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() != mlir::pto::kValue32 || intType.isUnsigned()) {
      return emitOpError("requires si32/i32/f16/f32 vector element type");
    }
    return success();
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires si32/i32/f16/f32 vector element type");
  }
  return success();
}
