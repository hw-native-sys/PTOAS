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
#include "PTO/IR/PTOTypeUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr int64_t kBlockSizeBytes = 32;

static VPTOAddressAccess oneAccess(OpOperand &base, OpOperand &offset,
                                   VPTOAddressUnit unit,
                                   Value elementTypeSource = {}) {
  if (!elementTypeSource) {
    elementTypeSource = base.get();
  }
  return {&base, VPTOAddressOffset{&offset, unit, elementTypeSource}};
}

static VPTOAddressAccess baseOnly(OpOperand &base) {
  return {&base, std::nullopt};
}

static VPTOAddressAccess currentAccess(OpOperand &base, OpOperand &offset,
                                       VPTOAddressUnit unit,
                                       Value updatedBase,
                                       Value elementTypeSource = {}) {
  // In post-update form the offset is the after-access advance. The current
  // address has already been materialized in base by the producer/transform.
  return updatedBase ? baseOnly(base)
                     : oneAccess(base, offset, unit, elementTypeSource);
}

static VPTOPostUpdateSemantics postUpdate(
    OpOperand &base, OpOperand *advance, VPTOAddressUnit unit,
    Value updatedBase,
    VPTOAdvanceConstraint constraint = VPTOAdvanceConstraint::Dynamic,
    Value elementTypeSource = {}) {
  if (!elementTypeSource) {
    elementTypeSource = base.get();
  }
  return {&base, advance, unit, constraint, updatedBase, elementTypeSource};
}

static OpOperand *getOptionalOperand(MutableOperandRange operands) {
  return operands.empty() ? nullptr : &*operands.begin();
}

static std::optional<int64_t> getElementBytes(Value source) {
  if (!source) {
    return std::nullopt;
  }

  Type elementType;
  Type sourceType = source.getType();
  if (auto pointerType = dyn_cast<PtrType>(sourceType)) {
    elementType = pointerType.getElementType();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(sourceType)) {
    elementType = memrefType.getElementType();
  } else if (auto vectorType = dyn_cast<VRegType>(sourceType)) {
    elementType = vectorType.getElementType();
  } else {
    return std::nullopt;
  }

  unsigned bytes = getPTOStorageElemByteSize(elementType);
  return bytes == 0 ? std::nullopt
                    : std::optional<int64_t>(static_cast<int64_t>(bytes));
}

} // namespace

VPTOAddressSemantics
mlir::pto::getDefaultVPTOAddressSemantics(Operation *operation) {
  return llvm::TypeSwitch<Operation *, VPTOAddressSemantics>(operation)
      .Case<VldsOp, Vldsx2Op>([](auto op) {
        OpOperand &base = op.getSourceMutable();
        OpOperand &offset = op.getOffsetMutable();
        Value payload = op.getOperation()->getResult(0);
        return VPTOAddressSemantics{
            {currentAccess(base, offset, VPTOAddressUnit::Element,
                           op.getUpdatedBase(), payload)},
            postUpdate(base, &offset, VPTOAddressUnit::Element,
                       op.getUpdatedBase(), VPTOAdvanceConstraint::Dynamic,
                       payload)};
      })
      .Case<VldusOp>([](VldusOp op) {
        OpOperand &base = op.getSourceMutable();
        return VPTOAddressSemantics{
            {baseOnly(base)},
            postUpdate(base, getOptionalOperand(op.getIncrementMutable()),
                       VPTOAddressUnit::Element, op.getUpdatedBase(),
                       VPTOAdvanceConstraint::Dynamic, op.getResult())};
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
                       VPTOAddressUnit::Element, op.getBaseOut(),
                       VPTOAdvanceConstraint::Dynamic, op.getValue())};
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

std::optional<int64_t>
mlir::pto::getVPTOAddressUnitBytes(Operation *operation, VPTOAddressUnit unit,
                                   Value elementTypeSource) {
  switch (unit) {
  case VPTOAddressUnit::Element:
    return getElementBytes(elementTypeSource);
  case VPTOAddressUnit::Block:
    return kBlockSizeBytes;
  case VPTOAddressUnit::Byte:
    return 1;
  case VPTOAddressUnit::Alignment:
    if (!operation) {
      return std::nullopt;
    }
    return getLoadStoreVecAlignmentSize(operation);
  }
  return std::nullopt;
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
