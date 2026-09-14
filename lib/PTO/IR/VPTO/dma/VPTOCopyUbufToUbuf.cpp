// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOCopyUbufToUbuf.cpp - pto.copy_ubuf_to_ubuf verification --------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult CopyUbufToUbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType())) {
    return emitOpError("requires pointer-like source and destination");
  }
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed source and destination");
  }
  return success();
}
