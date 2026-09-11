// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVsts.cpp - pto.vsts verification -------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

static bool isSupportedVstsDistToken(StringRef dist) {
  return lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
}

static std::optional<StringRef>
getVstsMaskGranularityOverride(StringRef dist) {
  const auto *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
  if (!contract || contract->maskGranularity.empty()) {
    return std::nullopt;
  }
  return contract->maskGranularity;
}

void VstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

template <typename StoreOp>
static LogicalResult verifyVstsCommon(StoreOp op) {
  if (failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type"))) {
    return failure();
  }

  if (!isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires a pointer-like destination");
  }

  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  if (destinationRole == MemoryRole::GM) {
    return op.emitOpError("requires a UB-backed destination");
  }

  if (std::optional<StringRef> dist = op.getDist();
      dist && !isSupportedVstsDistToken(*dist)) {
    return op.emitOpError("requires a supported store distribution token");
  }
  if (std::optional<StringRef> dist = op.getDist()) {
    if (std::optional<StringRef> granularity =
            getVstsMaskGranularityOverride(*dist)) {
      if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                                   "mask type", *granularity))) {
        return failure();
      }
    } else if (failed(verifyMaskTypeLike(op, op.getMask().getType(),
                                         "mask type"))) {
      return failure();
    }
  } else if (failed(verifyMaskTypeLike(op, op.getMask().getType(),
                                       "mask type"))) {
    return failure();
  }

  return success();
}

LogicalResult VstsOp::verify() {
  if (failed(verifyVstsCommon(*this))) {
    return failure();
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getDestination().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}
