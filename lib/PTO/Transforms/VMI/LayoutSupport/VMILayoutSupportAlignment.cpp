// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupportAlignment.cpp - E2B broadcast address legality -----===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutSupport.h"

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Support/CodeConstants.h"

namespace mlir::pto {

bool isGroupBroadcastLoadDirectAddressLegal(
    VMIGroupBroadcastLoadOp op, const VMIGroupBroadcastLoadDirectFact &fact) {
  if (fact.kind != VMIGroupBroadcastLoadDirectKind::E2B) {
    return true;
  }
  unsigned bits = fact.layout.elementBits;
  if (bits != kValue16 && bits != kValue32) {
    return false;
  }
  llvm::StringRef dist = bits == kValue16 ? "E2B_B16" : "E2B_B32";
  const VPTOMemoryDistContract *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::Load, dist,
                           std::optional<unsigned>(bits));
  if (!contract) {
    return false;
  }
  Type elementType =
      cast<VMIVRegType>(op.getResult().getType()).getElementType();
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(elementType);
  unsigned elementBits = getPTOStorageElemBitWidth(elementType);
  bool validPartShape = succeeded(lanesPerPart) && elementBits != 0;
  if (!validPartShape) {
    return false;
  }
  constexpr int64_t kBitsPerByte = kValue8;
  int64_t vectorBytes = (*lanesPerPart * elementBits) / kBitsPerByte;
  std::optional<int64_t> requiredAlignment =
      contract->getRequiredAlignmentBytes(vectorBytes);
  if (!requiredAlignment) {
    return false;
  }
  return isKnownAddressAligned(op.getSource(), op.getOffset(), elementType,
                               *requiredAlignment);
}

VMILayoutAttr getUnalignedE2BFallbackLayout(
    VMIGroupBroadcastLoadOp op, VMIVRegType type,
    const VMIGroupBroadcastLoadDirectFact &fact,
    const VMILayoutSupport &supports) {
  if (fact.kind != VMIGroupBroadcastLoadDirectKind::E2B ||
      isGroupBroadcastLoadDirectAddressLegal(op, fact)) {
    return {};
  }
  VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op.getContext());
  auto contiguousType = VMIVRegType::get(
      op.getContext(), type.getElementCount(), type.getElementType(), contiguous);
  FailureOr<VMIGroupBroadcastLoadDirectFact> contiguousDirect =
      supports.getGroupBroadcastLoadDirectFact(
          contiguousType, op.getSource().getType(),
          op.getSourceGroupStride(), op.getNumGroupsAttr().getInt());
  bool contiguousUsesBRC =
      succeeded(contiguousDirect) &&
      contiguousDirect->kind == VMIGroupBroadcastLoadDirectKind::BRC;
  if (!contiguousUsesBRC) {
    return {};
  }
  return contiguous;
}

} // namespace mlir::pto
