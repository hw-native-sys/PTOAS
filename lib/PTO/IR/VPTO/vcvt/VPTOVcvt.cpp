// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVcvt.cpp - pto.Vcvt methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOVcvtInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vcvt_detail;

ParseResult VcvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  OpAsmParser::UnresolvedOperand mask;
  NamedAttrList attrs;
  Type inputType, maskType, resultType;

  if (parser.parseOperand(input) || parser.parseComma() ||
      parser.parseOperand(mask) || parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseComma() ||
      parser.parseType(maskType) || parser.parseArrow() ||
      parser.parseType(resultType)) {
    return failure();
  }

  Attribute legacyRndAttr = attrs.get("round_mode");
  Attribute rndAttr = attrs.get("rnd");
  if (legacyRndAttr && rndAttr) {
    return parser.emitError(parser.getCurrentLocation())
           << "rnd and round_mode cannot be specified together";
  }

  if (failed(normalizeNamedStringAttr(parser, attrs, "round_mode", "rnd",
                                      normalizeRoundModeToken)) ||
      failed(normalizeNamedStringAttr(parser, attrs, "rnd", "rnd",
                                      normalizeRoundModeToken)) ||
      failed(normalizeNamedStringAttr(parser, attrs, "sat", "sat",
                                      normalizeSaturationToken)) ||
      failed(normalizeNamedStringAttr(parser, attrs, "part", "part",
                                      normalizeVcvtPartToken))) {
    return failure();
  }

  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands) ||
      parser.resolveOperand(mask, maskType, result.operands)) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void VcvtOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput() << ", " << getMask();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getInput().getType() << ", " << getMask().getType()
          << " -> " << getResult().getType();
}

LogicalResult VcvtOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }

  VcvtElemKind inputElemKind = classifyVcvtElemType(inputType.getElementType());
  VcvtElemKind resultElemKind = classifyVcvtElemType(resultType.getElementType());
  auto contract = lookupVcvtContract(inputElemKind, resultElemKind);
  if (!contract) {
    return emitOpError("unsupported vcvt source/result element type pair");
  }

  if (failed(verifyVcvtMaskGranularity(*this, getMask().getType(), inputElemKind,
                                       resultElemKind)) ||
      failed(verifyVcvtTotalElementBits(*this, getInput().getType(),
                                        getResult().getType(), inputElemKind,
                                        resultElemKind))) {
    return failure();
  }
  if (failed(verifyVcvtRndAttr(*this, *contract)) ||
      failed(verifyVcvtSatAttr(*this, *contract)) ||
      failed(verifyVcvtPartAttr(*this, *contract, inputElemKind, resultElemKind))) {
    return failure();
  }
  return success();
}
