// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOLoadCbufToCbMx.cpp - pto.load_cbuf_to_cb_mx verification -------===//
//===----------------------------------------------------------------------===//

#include "VPTOCubeBridgeInternal.h"

using namespace mlir;
using namespace mlir::pto;

// pto.load_cbuf_to_cb_mx: CBUF -> CB (RIGHT) matrix cube bridge load.
LogicalResult LoadCbufToCbMxOp::verify() {
  if (mlir::failed(verifyCubeBridgeLoadLikeOp(*this, AddressSpace::RIGHT, "RIGHT"))) {
    return failure();
  }
  return verifyCubeBridgeLoadStart(getOperation(), getXStartPosition(),
                                   "x_start_position", getYStartPosition(),
                                   "y_start_position");
}
