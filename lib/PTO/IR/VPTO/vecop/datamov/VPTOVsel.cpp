// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVsel.cpp - pto.Vsel methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VselOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          getOperation(), getSrc0().getType(), "src0 type"))) {
    return failure();
  }
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType()) {
    return emitOpError("requires src0, src1, and result to have identical vector types");
  }
  return success();
}
