// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVpack.cpp - pto.Vpack methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VpackOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isSupportedPartToken(getPart())) {
    return emitOpError("requires part to be LOWER or HIGHER");
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType)) {
    return emitOpError("currently requires integer source and result element types");
  }
  if (resultType.getElementCount() != srcType.getElementCount() * mlir::pto::kValue2) {
    return emitOpError(
        "requires result element count to be twice the source element count");
  }
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (srcWidth == 0 || resultWidth * mlir::pto::kValue2 != srcWidth) {
    return emitOpError(
        "requires result element width to be half the source element width");
  }
  auto srcIntType = cast<IntegerType>(srcElemType);
  auto resultIntType = cast<IntegerType>(resultElemType);
  if (!resultIntType.isUnsigned()) {
    return emitOpError("requires unsigned result element type");
  }
  if (!((srcIntType.getWidth() == mlir::pto::kValue32 &&
         resultIntType.getWidth() == mlir::pto::kValue16) ||
        (srcIntType.getWidth() == mlir::pto::kValue16 &&
         resultIntType.getWidth() == mlir::pto::kValue8))) {
    return emitOpError(
        "currently supports only s32/u32 -> u16 and s16/u16 -> u8");
  }
  return success();
}
