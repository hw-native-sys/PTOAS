// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"

#include "PTOLowerToOpLibCalls.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

FailureOr<bool> mlir::pto::tryCloneOpLibInlineBridgeOp(OpBuilder &builder,
                                                       Operation &op,
                                                       IRMapping &mapping) {
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(&op)) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
      return failure();
    }

    Value mappedSrc = mapping.lookupOrNull(cast.getOperand(0));
    if (!mappedSrc) {
      return failure();
    }

    Type dstTy = cast.getResult(0).getType();
    if (mappedSrc.getType() == dstTy) {
      mapping.map(cast.getResult(0), mappedSrc);
      return true;
    }

    auto clonedCast = builder.create<UnrealizedConversionCastOp>(
        cast.getLoc(), TypeRange{dstTy}, ValueRange{mappedSrc});
    mapping.map(cast.getResult(0), clonedCast.getResult(0));
    return true;
  }

  return false;
}
