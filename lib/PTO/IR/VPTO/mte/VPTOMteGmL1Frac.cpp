// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOMteGmL1Frac.cpp - pto.MteGmL1Frac methods ----------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMteInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::mte_detail;

void MteGmL1FracOp::build(OpBuilder &odsBuilder, OperationState &state,
                           Value source, Value destination,
                           pto::CubeLoadFracMode mode,
                           pto::CubeLoadFracShapeConfig shape,
                           pto::CubeLoadFracSrcLayoutConfig srcLayout,
                           pto::CubeLoadFracDstGroupConfig dstGroup,
                           pto::CubeLoadFracCtrlConfig ctrl) {
  state.addOperands({source, destination, shape.nValue, shape.dValue,
                     srcLayout.srcInnerStride});
  state.addOperands({dstGroup.groupCount, dstGroup.dstLoop2Stride,
                     dstGroup.dstLoop3Stride, dstGroup.dstLoop4Stride,
                     ctrl.l2CacheCtrl, ctrl.smallc0En});
  bool hasSrcOuterStride = srcLayout.srcOuterStride.has_value();
  if (hasSrcOuterStride) {
    state.addOperands(*srcLayout.srcOuterStride);
  }

  state.addAttribute(getModeAttrName(state.name),
                     CubeLoadFracModeAttr::get(odsBuilder.getContext(), mode));
}

ParseResult MteGmL1FracOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, destination;
  StringRef modeKeyword;
  SmallVector<OpAsmParser::UnresolvedOperand> shapeOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> srcLayoutOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> dstGroupOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> ctrlOperands;
  if (failed(parseMteGmL1FracBasicOperands(parser, source, destination,
                                           modeKeyword, shapeOperands,
                                           srcLayoutOperands,
                                           dstGroupOperands, ctrlOperands))) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType, destinationType;
  SmallVector<Type> shapeTypes, srcLayoutTypes, dstGroupTypes, ctrlTypes;
  if (failed(parseMteGmL1FracBasicTypes(parser, sourceType, destinationType,
                                        modeKeyword, shapeTypes,
                                        srcLayoutTypes, dstGroupTypes,
                                        ctrlTypes))) {
    return failure();
  }
  auto modeOr = parseCubeLoadFracModeKeyword(modeKeyword);
  if (failed(modeOr)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one of 'nd2nz' or 'dn2nz'");
  }
  if (failed(validateMteGmL1FracOperands(
          parser, shapeOperands.size(), shapeTypes.size(),
          srcLayoutOperands.size(), srcLayoutTypes.size(),
          dstGroupOperands.size(), dstGroupTypes.size(),
          ctrlOperands.size(), ctrlTypes.size()))) {
    return failure();
  }
  result.addAttribute(getModeAttrName(result.name),
                      CubeLoadFracModeAttr::get(parser.getContext(), *modeOr));
  if (failed(resolveMteGmL1FracOperands(parser, result, source, sourceType,
                                        destination, destinationType,
                                        shapeOperands, shapeTypes,
                                        srcLayoutOperands, srcLayoutTypes,
                                        dstGroupOperands, dstGroupTypes,
                                        ctrlOperands, ctrlTypes))) {
    return failure();
  }
  return success();
}

void MteGmL1FracOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << ", " << getDestination() << ", "
          << pto::stringifyCubeLoadFracMode(getMode());
  p << ", shape(" << getNValue() << ", " << getDValue() << ")";
  printCubeLoadFracSrcLayoutGroup(p, getSrcInnerStride(),
                                  getSrcOuterStride());
  p << ", dst_group(" << getGroupCount() << ", " << getDstLoop2Stride()
          << ", " << getDstLoop3Stride() << ", " << getDstLoop4Stride()
          << ")";
  p << ", ctrl(" << getL2CacheCtrl() << ", " << getSmallc0En() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "mode"});
  p << " : " << getSource().getType() << ", " << getDestination().getType()
          << ", " << pto::stringifyCubeLoadFracMode(getMode())
          << ", shape " << getNValue().getType() << ", " << getDValue().getType();
  printCubeLoadFracSrcLayoutTypes(
      p, getSrcInnerStride().getType(),
      getSrcOuterStride() ? getSrcOuterStride().getType() : Type());
  p << ", dst_group " << getGroupCount().getType() << ", "
          << getDstLoop2Stride().getType() << ", "
          << getDstLoop3Stride().getType() << ", "
          << getDstLoop4Stride().getType() << ", ctrl "
          << getL2CacheCtrl().getType() << ", " << getSmallc0En().getType();
}

LogicalResult MteGmL1FracOp::verify() {
  if (failed(verifyCopyGmToUbufOp(*this, true))) {
    return failure();
  }

  auto checkNonNegativeConst = [this](Value value, StringRef name) -> LogicalResult {
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return emitOpError() << name << " must be non-negative";
    }
    return success();
  };
  if (failed(checkNonNegativeConst(getGroupCount(), "group_count")) ||
      failed(checkNonNegativeConst(getSrcInnerStride(), "src_inner_stride")) ||
      failed(checkNonNegativeConst(getDstLoop2Stride(), "dst_loop2_stride")) ||
      failed(checkNonNegativeConst(getDstLoop3Stride(), "dst_loop3_stride")) ||
      failed(checkNonNegativeConst(getDstLoop4Stride(), "dst_loop4_stride")) ||
      (getSrcOuterStride() &&
       failed(checkNonNegativeConst(getSrcOuterStride(), "src_outer_stride")))) {
    return failure();
  }

  APInt groupCount;
  if (matchPattern(getGroupCount(), m_ConstantInt(&groupCount)) &&
      groupCount.isZero()) {
    return emitOpError("group_count must be greater than zero");
  }

  APInt smallc0En;
  APInt dValue;
  if (matchPattern(getSmallc0En(), m_ConstantInt(&smallc0En)) &&
      smallc0En.getBoolValue() && matchPattern(getDValue(), m_ConstantInt(&dValue)) &&
      dValue.ugt(mlir::pto::kValue4)) {
    return emitOpError("smallc0_en requires d_value <= 4");
  }

  return success();
}

void MteGmL1FracOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}
