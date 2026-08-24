// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAddressSemantics.cpp - VPTO addressing contract ---------------===//

#include "PTO/IR/VPTOAddressSemantics.h"

#include "PTO/IR/PTO.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static VPTOAddressAccess oneAccess(OpOperand &base, OpOperand &offset,
                                   VPTOAddressUnit unit) {
  return {&base, VPTOAddressOffset{&offset, unit}};
}

static VPTOAddressAccess baseOnly(OpOperand &base) {
  return {&base, std::nullopt};
}

static VPTOAddressAccess currentAccess(OpOperand &base, OpOperand &offset,
                                       VPTOAddressUnit unit,
                                       Value updatedBase) {
  // In post-update form the offset is the after-access advance. The current
  // address has already been materialized in base by the producer/transform.
  return updatedBase ? baseOnly(base) : oneAccess(base, offset, unit);
}

static VPTOPostUpdateSemantics postUpdate(
    OpOperand &base, OpOperand *advance, VPTOAddressUnit unit,
    Value updatedBase,
    VPTOAdvanceConstraint constraint = VPTOAdvanceConstraint::Dynamic) {
  return {&base, advance, unit, constraint, updatedBase};
}

static OpOperand *getOptionalOperand(MutableOperandRange operands) {
  return operands.empty() ? nullptr : &*operands.begin();
}

} // namespace

VPTOAddressSemantics
mlir::pto::getDefaultVPTOAddressSemantics(Operation *operation) {
  return llvm::TypeSwitch<Operation *, VPTOAddressSemantics>(operation)
      .Case<VldsOp, Vldsx2Op>([](auto op) {
        OpOperand &base = op.getSourceMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Element,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Element,
                       op.getUpdatedBase())};
      })
      .Case<VldusOp>([](VldusOp op) {
        OpOperand &base = op.getSourceMutable();
        return VPTOAddressSemantics{
            {baseOnly(base)},
            postUpdate(base, getOptionalOperand(op.getIncrementMutable()),
                       VPTOAddressUnit::Element, op.getUpdatedBase())};
      })
      .Case<PldsOp>([](PldsOp op) {
        OpOperand &base = op.getSourceMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Byte,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Byte,
                       op.getUpdatedBase())};
      })
      .Case<PldiOp>([](PldiOp op) {
        OpOperand &base = op.getSourceMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Alignment,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Alignment,
                       op.getUpdatedBase(),
                       VPTOAdvanceConstraint::Constant)};
      })
      .Case<VstsOp>([](VstsOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Element,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Element,
                       op.getUpdatedBase())};
      })
      .Case<VstusOp>([](VstusOp op) {
        OpOperand &base = op.getBaseMutable();
        return VPTOAddressSemantics{
            {baseOnly(base)},
            postUpdate(base, &op.getOffsetMutable(),
                       VPTOAddressUnit::Element, op.getBaseOut())};
      })
      .Case<PstsOp>([](PstsOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Byte,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Byte,
                       op.getUpdatedBase())};
      })
      .Case<PstiOp>([](PstiOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Alignment,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Alignment,
                       op.getUpdatedBase(),
                       VPTOAdvanceConstraint::Constant)};
      })
      .Case<SprstsOp>([](SprstsOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Byte,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Byte,
                       op.getUpdatedBase())};
      })
      .Case<SprstiOp>([](SprstiOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Alignment,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Alignment,
                       op.getUpdatedBase(),
                       VPTOAdvanceConstraint::SignedI8)};
      })
      .Case<VstasOp>([](VstasOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &offset = op.getOffsetMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Element,
                           op.getUpdatedBase())},
            postUpdate(base, &offset, VPTOAddressUnit::Element,
                       op.getUpdatedBase())};
      })
      .Case<VsldbOp>([](VsldbOp op) {
        OpOperand &base = op.getSourceMutable();
        OpOperand &stride = op.getRepeatStrideMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, stride, VPTOAddressUnit::Block,
                           op.getUpdatedBase())},
            postUpdate(base, &stride, VPTOAddressUnit::Block,
                       op.getUpdatedBase())};
      })
      .Case<VsstbOp>([](VsstbOp op) {
        OpOperand &base = op.getDestinationMutable();
        OpOperand &stride = op.getRepeatStrideMutable();
        return VPTOAddressSemantics{
            {currentAccess(base, stride, VPTOAddressUnit::Block,
                           op.getUpdatedBase())},
            postUpdate(base, &stride, VPTOAddressUnit::Block,
                       op.getUpdatedBase())};
      })
      .Default([](Operation *) { return VPTOAddressSemantics{}; });
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

StringRef mlir::pto::stringifyVPTOAdvanceConstraint(
    VPTOAdvanceConstraint value) {
  switch (value) {
  case VPTOAdvanceConstraint::Dynamic:
    return "dynamic";
  case VPTOAdvanceConstraint::Constant:
    return "constant";
  case VPTOAdvanceConstraint::SignedI8:
    return "signed-i8";
  }
  return "unknown";
}
