// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOTileBufAddr.cpp - pto.TileBufAddr methods ----------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

LogicalResult TileBufAddrOp::verify() {
  Type dstType = getDst().getType();
  Type elementType;
  Attribute srcMemorySpace;
  int64_t srcRank = 0;

  if (auto srcTileType = dyn_cast<pto::TileBufType>(getSrc().getType())) {
    elementType = srcTileType.getElementType();
    srcMemorySpace = srcTileType.getMemorySpace();
    srcRank = static_cast<int64_t>(srcTileType.getShape().size());
  } else if (auto srcMemRefType = dyn_cast<BaseMemRefType>(getSrc().getType())) {
    elementType = srcMemRefType.getElementType();
    srcMemorySpace = srcMemRefType.getMemorySpace();
    srcRank = srcMemRefType.getRank();
  } else {
    return emitOpError("source must be a !pto.tile_buf<...> or memref");
  }

  auto srcSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMemorySpace);

  if (auto dstMemRefType = dyn_cast<BaseMemRefType>(dstType)) {
    if (dstMemRefType.getElementType() != elementType) {
      return emitOpError(
          "memref result element type must match tile element type");
    }
    if (dstMemRefType.getRank() != srcRank) {
      return emitOpError("memref result rank must match tile rank");
    }
    auto dstSpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(dstMemRefType.getMemorySpace());
    if (srcSpace && dstSpace && srcSpace != dstSpace) {
      return emitOpError("memref result must stay within the tile memory space");
    }
    return success();
  }

  auto dstPtrType = dyn_cast<pto::PtrType>(dstType);
  if (!dstPtrType) {
    return emitOpError("result must be a memref or !pto.ptr<...>");
  }
  if (dstPtrType.getElementType() != elementType) {
    return emitOpError(
        "pointer result element type must match tile element type");
  }
  if (srcSpace && dstPtrType.getMemorySpace() != srcSpace) {
    return emitOpError("pointer result must stay within the tile memory space");
  }
  return success();
}
