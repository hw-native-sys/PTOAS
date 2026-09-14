// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVcmps.cpp - pto.Vcmps methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VcmpsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  Type srcElementType = srcType.getElementType();
  Type scalarType = getScalar().getType();
  if (!isCompatibleScalarForSemanticType(srcElementType, scalarType)) {
    return emitOpError("requires scalar type to match source element type");
  }
  if (!isSupportedCmpMode(getCmpMode())) {
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  }
  return success();
}
