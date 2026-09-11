// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVdup.cpp - pto.Vdup methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VdupOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !pto.vreg<...>");
  }

  std::optional<StringRef> granularity =
      getVdupMaskGranularity(resultType.getElementType());
  if (!granularity) {
    return emitOpError("result element type must use b8, b16, or b32 mask granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          getOperation(), getMask().getType(), "mask type", *granularity))) {
    return failure();
  }

  if (!isSupportedVdupPosition(getPosition())) {
    return emitOpError("position must be LOWEST or HIGHEST");
  }

  Type inputType = getInput().getType();
  if (auto inputVecType = dyn_cast<VRegType>(inputType)) {
    if (inputVecType != resultType) {
      return emitOpError("vector input must match result vector type");
    }
    return success();
  }

  if (getPosition()) {
    return emitOpError("position is only supported for vector input");
  }

  Type resultElementType = resultType.getElementType();
  if (!isCompatibleScalarForSemanticType(resultElementType, inputType)) {
    return emitOpError("scalar input must match result element type");
  }

  return success();
}
