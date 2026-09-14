// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVbr.cpp - pto.Vbr methods --------------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VbrOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }

  auto resultVecType = cast<VRegType>(getResult().getType());
  Type elementType = getValue().getType();
  if (isa<ShapedType, VectorType>(elementType)) {
    return emitOpError("value must be a scalar matching the result element type");
  }
  Type resultElementType = resultVecType.getElementType();
  if (!isCompatibleScalarForSemanticType(resultElementType, elementType)) {
    return emitOpError("value type must match result element type");
  }
  return success();
}
