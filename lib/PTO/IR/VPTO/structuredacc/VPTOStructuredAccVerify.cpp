// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOStructuredAccVerify.cpp - structured acc-store verify entry point ===//
//===----------------------------------------------------------------------===//

#include "VPTOStructuredAccInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::structuredacc_detail;

LogicalResult verifyStructuredAccStoreLike(
    Operation *op, Type srcType, Type dstType, Value preQuant,
    Value preRelu, Value clipValue, Value split, Value loop0SrcStride,
    Value loop3Count, Value loop3SrcStride, Value loop3DstStride,
    std::optional<AccStoreUnitFlagCtrl> unitFlag, std::optional<AccStoreQuantPreMode> preQuantMode,
    std::optional<ReluPreMode> preReluMode, std::optional<AccStoreMode> mode,
    std::optional<AccStoreAtomicType> atomicType,
    std::optional<AccStoreAtomicOp> atomicOp, bool allowAtomic) {
  auto getBufferElementType = [](Type type) -> Type {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      return ptrType.getElementType();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      return memrefType.getElementType();
    }
    return {};
  };
  Type sourceElementType = getBufferElementType(srcType);
  Type destinationElementType = getBufferElementType(dstType);
  if (failed(verifyStructuredPreQuant(op, preQuant, sourceElementType,
                                     destinationElementType, preQuantMode))) {
    return failure();
  }
  if (clipValue &&
      !isStructuredAccStoreClipSupportedElementType(destinationElementType)) {
    return op->emitOpError()
           << "clip requires destination element type to be f16, ui8, or signed 4/8/16-bit integer, got "
           << destinationElementType;
  }
  if (failed(verifyStructuredAccStoreClipPayload(op, destinationElementType,
                                                 clipValue))) {
    return failure();
  }
  if (failed(verifyStructuredPreRelu(op, preRelu, clipValue, preReluMode))) {
    return failure();
  }
  bool hasLoop3 = static_cast<bool>(loop3Count) ||
                   static_cast<bool>(loop3SrcStride) ||
                   static_cast<bool>(loop3DstStride);
  if (hasLoop3 && !(loop3Count && loop3SrcStride && loop3DstStride)) {
    return op->emitOpError("loop3 requires count, src stride, and dst stride together");
  }
  if (failed(verifyStructuredAccStoreMode(op, split, loop0SrcStride, loop3Count,
                                          destinationElementType, unitFlag,
                                          mode))) {
    return failure();
  }
  if (static_cast<bool>(atomicType) != static_cast<bool>(atomicOp)) {
    return op->emitOpError("atomic requires type and op together");
  }
  if ((atomicType || atomicOp) && !allowAtomic) {
    return op->emitOpError("atomic is only supported for mte_l0c_gm");
  }
  return success();
}
