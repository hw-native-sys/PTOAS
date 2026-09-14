// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVtrc.cpp - pto.Vtrc methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

ParseResult VtrcOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  OpAsmParser::UnresolvedOperand mask;
  std::string roundModeToken;
  NamedAttrList attrs;
  Type inputType, maskType, resultType;

  if (parser.parseOperand(input) || parser.parseComma() ||
      parser.parseOperand(mask) || parser.parseComma() ||
      parser.parseKeywordOrString(&roundModeToken) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseComma() ||
      parser.parseType(maskType) || parser.parseArrow() ||
      parser.parseType(resultType)) {
    return failure();
  }

  auto normalized = normalizeRoundModeToken(roundModeToken);
  if (!normalized || !isSupportedVtrcRoundMode(*normalized)) {
    return parser.emitError(parser.getCurrentLocation())
           << "round mode must be one of R/A/F/C/Z or "
              "ROUND_R/ROUND_A/ROUND_F/ROUND_C/ROUND_Z";
  }

  attrs.set("round_mode", parser.getBuilder().getStringAttr(*normalized));
  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands) ||
      parser.resolveOperand(mask, maskType, result.operands)) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void VtrcOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput() << ", " << getMask() << ", ";
  Builder builder(getContext());
  auto normalized = normalizeRoundModeToken(getRoundMode());
  p.printAttributeWithoutType(
      builder.getStringAttr(normalized.value_or(getRoundMode())));
  p.printOptionalAttrDict((*this)->getAttrs(), {"round_mode"});
  p << " : " << getInput().getType() << ", " << getMask().getType()
          << " -> " << getResult().getType();
}

LogicalResult VtrcOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (inputType != resultType) {
    return emitOpError("requires input and result to have identical vreg type");
  }
  auto elemType = inputType.getElementType();
  if (!(elemType.isF16() || elemType.isF32() || elemType.isBF16())) {
    return emitOpError("requires f16/f32/bf16 vector element type");
  }
  if (failed(verifyVdupMaskGranularityLike(*this, elemType))) {
    return failure();
  }
  auto normalized = normalizeRoundModeToken(getRoundMode());
  if (!normalized || !isSupportedVtrcRoundMode(*normalized)) {
    return emitOpError("round mode must be one of R/A/F/C/Z");
  }
  return success();
}
