// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteL0cUb.cpp - pto.MteL0cUb methods ----------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

ParseResult MteL0cUbOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(parser.getContext());
  StructuredAccStoreAsmState state;
  OpAsmParser::UnresolvedOperand source, destination, m, n, srcStride,
      dstStride, subBlockId;
  bool hasSubBlockId = false;
  AccStoreUbDstMode dstMode = AccStoreUbDstMode::Single;
  if (failed(parseMteL0cUbBasicOperands(parser, source, destination, m,
                                        n, srcStride, dstStride))) {
    return failure();
  }
  if (failed(parseMteL0cUbDstMode(parser, dstMode, subBlockId, hasSubBlockId))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma()) &&
      parseStructuredAccStoreClauses(parser, state)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType, mType, nType, srcStrideType,
      dstStrideType, subBlockIdType;
  if (failed(parseMteL0cUbTypes(parser, sourceType, destinationType, mType,
                                nType, srcStrideType, dstStrideType,
                                hasSubBlockId, subBlockIdType, state))) {
    return failure();
  }
  setStructuredAccStoreSegmentSizes<MteL0cUbOp>(
      result, {1, 1, 1, 1, 1, 1, !state.preQuantOperands.empty() ? 1 : 0,
               !state.preReluOperands.empty() ? 1 : 0,
               !state.clipValueOperands.empty() ? 1 : 0,
               hasSubBlockId ? 1 : 0,
               !state.splitOperands.empty() ? 1 : 0,
               !state.loop0SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3CountOperands.empty() ? 1 : 0,
               !state.loop3SrcStrideOperands.empty() ? 1 : 0,
               !state.loop3DstStrideOperands.empty() ? 1 : 0});
  if (state.atomicType || state.atomicOp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "atomic is only supported for mte_l0c_gm");
  }
  addStructuredAccStoreAttrs<MteL0cUbOp>(result, builder, state);
  result.addAttribute("dst_mode", AccStoreUbDstModeAttr::get(builder.getContext(), dstMode));
  return resolveMteL0cUbOperands(parser, result, source, sourceType,
                                 destination, destinationType, m, mType, n,
                                 nType, srcStride, srcStrideType, dstStride,
                                 dstStrideType, hasSubBlockId, subBlockId,
                                 subBlockIdType, state);
}

void MteL0cUbOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << ", " << getDestination() << ", " << getM()
          << ", " << getN() << ", " << getSrcStride() << ", "
          << getDstStride() << ", dst_mode(";
  switch (getDstMode()) {
  case AccStoreUbDstMode::Single:
    p << getSubBlockid();
    break;
  case AccStoreUbDstMode::SplitM:
    p << "split_m";
    break;
  case AccStoreUbDstMode::SplitN:
    p << "split_n";
    break;
  }
  p << ")";
  printStructuredAccStoreClauses(p, getUnitFlag(), getPreQuant(),
                                 getPreQuantMode(), getPreRelu(),
                                 getPreReluMode(), getClipValue(), getMode(),
                                 getSplit(), getLoop0SrcStride(),
                                 getLoop3Count(), getLoop3SrcStride(),
                                 getLoop3DstStride(), getSatMode(),
                                 std::nullopt, std::nullopt);
  p.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode",
                                                 "unit_flag",
                                                 "pre_quant_mode",
                                                 "pre_relu_mode",
                                                 "dst_mode",
                                                 "sat_mode"});
  p << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << getM().getType() << ", " << getN().getType() << ", "
          << getSrcStride().getType() << ", " << getDstStride().getType();
  if (getSubBlockid()) {
    p << ", " << getSubBlockid().getType();
  }
  printStructuredAccStoreOptionalTypes(
      p, getPreQuant(), getPreRelu(), getClipValue(), getSplit(),
      getLoop0SrcStride(), getLoop3Count(), getLoop3SrcStride(),
      getLoop3DstStride());
}

LogicalResult MteL0cUbOp::verify() {
  if (failed(verifyMteL0cUbBufferSpaces(*this)) ||
      failed(verifyStructuredAccStoreLike(
          *this, getSource().getType(), getDestination().getType(), getPreQuant(), getPreRelu(),
          getClipValue(), getSplit(), getLoop0SrcStride(), getLoop3Count(),
          getLoop3SrcStride(), getLoop3DstStride(), getUnitFlag(),
          getPreQuantMode(), getPreReluMode(), getMode(), std::nullopt,
          std::nullopt, /*allowAtomic=*/false))) {
    return failure();
  }
  if (getDstMode() == AccStoreUbDstMode::Single) {
    return verifyMteL0cUbSubBlockId(*this);
  }
  if (getSubBlockid()) {
    return emitOpError("split destination modes do not accept sub_blockid");
  }
  return verifyMteL0cUbSplitRestrictions(*this);
}

void MteL0cUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}
