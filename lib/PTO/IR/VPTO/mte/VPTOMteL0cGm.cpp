// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL0cGm.cpp - pto.MteL0cGm methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

ParseResult MteL0cGmOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride, sid, l2CacheCtrl;
  if (parseRequiredOperandWithComma(parser, source) ||
      parseRequiredOperandWithComma(parser, destination) ||
      parseRequiredOperandWithComma(parser, m) ||
      parseRequiredOperandWithComma(parser, n) ||
      parseRequiredOperandWithComma(parser, srcStride) ||
      parseRequiredOperandWithComma(parser, dstStride) ||
      parseRequiredOperandWithComma(parser, sid) ||
      parseRequiredOperandWithComma(parser, l2CacheCtrl) ||
      parseStructuredAccStoreClauses(parser, state) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, mType, nType, srcStrideType, dstStrideType,
      sidType, l2CacheCtrlType;
  if (failed(parseMteL0cGmTypes(parser, sourceType, destinationType, mType,
                                 nType, srcStrideType, dstStrideType,
                                 sidType, l2CacheCtrlType, state))) {
    return failure();
  }
  setMteL0cGmSegmentSizes(result, state);
  addStructuredAccStoreAttrs<MteL0cGmOp>(result, builder, state);
  if (failed(resolveMteL0cGmOperands(
          parser, result, source, sourceType, destination, destinationType,
          m, mType, n, nType, srcStride, srcStrideType, dstStride,
          dstStrideType, sid, sidType, l2CacheCtrl, l2CacheCtrlType, state))) {
    return failure();
  }
  return success();
}

void MteL0cGmOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", "
          << getDstStride() << ", " << getSid() << ", " << getL2CacheCtrl();
  printStructuredAccStoreClausesAndAttrs(*this, p);
  p << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType()
          << ", " << getSid().getType() << ", " << getL2CacheCtrl().getType();
  printStructuredAccStoreOptionalTypes(
      p, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

LogicalResult MteL0cGmOp::verify() {
  if (!isBufferLike(getSource().getType()) ||
      !isBufferLike(getDestination().getType())) {
    return emitOpError("requires buffer-like source and destination");
  }
  std::optional<AddressSpace> sourceSpace =
      getBufferAddressSpace(getSource().getType());
  std::optional<AddressSpace> destinationSpace =
      getBufferAddressSpace(getDestination().getType());
  if (sourceSpace != AddressSpace::ACC || destinationSpace != AddressSpace::GM) {
    return emitOpError("requires ACC source and GM destination");
  }
  return verifyStructuredAccStoreLike(
      *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
      getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
      getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
      getPreQuantMode(), getPreReluMode(), getMode(), getAtomicType(),
      getAtomicOp(), /*allowAtomic=*/true);
}

void MteL0cGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}
