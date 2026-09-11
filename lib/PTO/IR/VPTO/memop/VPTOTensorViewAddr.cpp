// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOTensorViewAddr.cpp - pto.TensorViewAddr methods ----------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

LogicalResult TensorViewAddrOp::verify() {
  Type srcType = getSrc().getType(); Type dstType = getDst().getType();
  Type elementType; int64_t expectedRank = -1;
  auto gmSpace = pto::AddressSpaceAttr::get(getContext(), pto::AddressSpace::GM);
  if (auto tvType = dyn_cast<pto::TensorViewType>(srcType)) {
    elementType = tvType.getElementType();
    expectedRank = tvType.getRank();
  } else if (auto partType = dyn_cast<pto::PartitionTensorViewType>(srcType)) {
    elementType = partType.getElementType();
    expectedRank = partType.getRank();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(srcType)) {
    elementType = memrefType.getElementType();
    expectedRank = memrefType.getRank();
    auto srcSpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
    if (srcSpace && srcSpace != gmSpace) {
      return emitOpError("memref source must stay in gm memory space");
    }
  } else {
    return emitOpError(
        "source must be a tensor_view, partition_tensor_view, or memref");
  }
  if (auto dstMemRefType = dyn_cast<BaseMemRefType>(dstType)) {
    if (dstMemRefType.getElementType() != elementType) {
      return emitOpError(
          "memref result element type must match source element type");
    }
    if (dstMemRefType.getRank() != expectedRank) {
      return emitOpError("memref result rank must match source rank");
    }
    auto dstSpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(dstMemRefType.getMemorySpace());
    if (dstSpace && dstSpace != gmSpace) {
      return emitOpError("memref result must stay in gm memory space");
    }
    return success();
  }
  auto dstPtrType = dyn_cast<pto::PtrType>(dstType);
  if (!dstPtrType) {
    return emitOpError("result must be a memref or !pto.ptr<...>");
  }
  if (dstPtrType.getElementType() != elementType) {
    return emitOpError(
        "pointer result element type must match source element type");
  }
  if (dstPtrType.getMemorySpace() != gmSpace) {
    return emitOpError("pointer result must stay in gm memory space");
  }
  return success();
}
