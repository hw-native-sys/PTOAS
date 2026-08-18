// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAddressSemantics.cpp - VPTO current-access contract -----------===//

#include "PTO/IR/VPTOAddressSemantics.h"

#include "PTO/IR/PTO.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static SmallVector<VPTOAddressAccess>
oneAccess(Value base, Value offset, VPTOAddressUnit unit) {
  return {{base, VPTOAddressOffset{offset, unit}}};
}

static SmallVector<VPTOAddressAccess> baseOnly(Value base) {
  return {{base, std::nullopt}};
}

} // namespace

SmallVector<VPTOAddressAccess>
mlir::pto::getDefaultVPTOAddressAccesses(Operation *operation) {
  return llvm::TypeSwitch<Operation *, SmallVector<VPTOAddressAccess>>(
             operation)
      .Case<VldsOp, Vldsx2Op>([](auto op) {
        return oneAccess(op.getSource(), op.getOffset(),
                         VPTOAddressUnit::Element);
      })
      .Case<VldusOp>([](VldusOp op) { return baseOnly(op.getSource()); })
      .Case<PldsOp>([](PldsOp op) {
        return oneAccess(op.getSource(), op.getOffset(),
                         VPTOAddressUnit::Byte);
      })
      .Case<PldiOp>([](PldiOp op) {
        return oneAccess(op.getSource(), op.getOffset(),
                         VPTOAddressUnit::Alignment);
      })
      .Case<VstsOp>([](VstsOp op) {
        return oneAccess(op.getDestination(), op.getOffset(),
                         VPTOAddressUnit::Element);
      })
      .Case<VstusOp>([](VstusOp op) { return baseOnly(op.getBase()); })
      .Case<PstsOp>([](PstsOp op) {
        return oneAccess(op.getDestination(), op.getOffset(),
                         VPTOAddressUnit::Byte);
      })
      .Case<PstiOp>([](PstiOp op) {
        return oneAccess(op.getDestination(), op.getOffset(),
                         VPTOAddressUnit::Alignment);
      })
      .Case<SprstsOp>([](SprstsOp op) {
        return oneAccess(op.getDestination(), op.getOffset(),
                         VPTOAddressUnit::Byte);
      })
      .Case<SprstiOp>([](SprstiOp op) {
        return oneAccess(op.getDestination(), op.getOffset(),
                         VPTOAddressUnit::Alignment);
      })
      .Case<VstasOp>([](VstasOp op) {
        return oneAccess(op.getDestination(), op.getOffset(),
                         VPTOAddressUnit::Element);
      })
      .Case<VsldbOp>([](VsldbOp op) {
        return oneAccess(op.getSource(), op.getRepeatStride(),
                         VPTOAddressUnit::Block);
      })
      .Case<VsstbOp>([](VsstbOp op) {
        return oneAccess(op.getDestination(), op.getRepeatStride(),
                         VPTOAddressUnit::Block);
      })
      .Default([](Operation *) { return SmallVector<VPTOAddressAccess>(); });
}

StringRef mlir::pto::stringifyVPTOAddressUnit(VPTOAddressUnit unit) {
  switch (unit) {
  case VPTOAddressUnit::Element:
    return "element";
  case VPTOAddressUnit::Block:
    return "block";
  case VPTOAddressUnit::Byte:
    return "byte";
  case VPTOAddressUnit::Alignment:
    return "alignment";
  }
  return "unknown";
}
