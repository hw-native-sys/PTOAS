// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVci.cpp - pto.Vci methods --------------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VciOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !pto.vreg<...>");
  }
  Type resultElemType = resultType.getElementType();
  bool supportedInteger = false;
  if (auto intType = dyn_cast<IntegerType>(resultElemType)) {
    supportedInteger = intType.getWidth() == mlir::pto::kValue8 ||
                       intType.getWidth() == mlir::pto::kValue16 ||
                       intType.getWidth() == mlir::pto::kValue32;
  }
  bool supportedFloat = resultElemType.isF16() || resultElemType.isF32();
  if (!supportedInteger && !supportedFloat) {
    return emitOpError("result element type must be integer or f16/f32");
  }
  if (!isCompatibleScalarForSemanticType(resultElemType, getIndex().getType())) {
    return emitOpError("index type must match result element type");
  }
  return success();
}
