// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecScope.cpp - pto.VecScope methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOVecScopeInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscope_detail;

LogicalResult VecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0) {
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();
  }

  if (Operation *boundaryOp = findForbiddenSyncInRegion(bodyRegion)) {
    return boundaryOp->emitOpError()
           << "must be outside 'pto.vecscope'; synchronization operations "
              "delimit vector scopes";
  }

  return success();
}
