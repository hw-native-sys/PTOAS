// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTRowExpandBroadcastOperand(
    Operation *op, Type operandTy, ArrayRef<int64_t> operandValid,
    ArrayRef<int64_t> dstValid, Type elem, StringRef name,
    bool requireNonRowMajor) {
  if (!hasCompatibleKnownExtent(operandValid[0], dstValid[0]))
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to equal dst valid_shape[0]";
  bool rowMajor = isRowMajorTileBuf(operandTy);
  if (requireNonRowMajor && rowMajor)
    return op->emitOpError() << "expects " << name
                             << " to use a non-row-major layout when tmp is present";
  int64_t expectedCol = elem.isInteger(8)
                            ? 32
                            : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t column = operandValid[1];
  if (rowMajor && column != ShapedType::kDynamic && column != expectedCol)
    return op->emitOpError() << "expects row-major " << name
                             << " valid_shape[1] to be 32/sizeof(dtype)";
  if (!rowMajor && column != ShapedType::kDynamic && column != 1)
    return op->emitOpError() << "expects non-row-major " << name
                             << " valid_shape[1] to be 1";
  return success();
}

static LogicalResult verifyTRowExpandFullAndBroadcast(
    Operation *op, Type fullTy, ArrayRef<int64_t> fullValid,
    StringRef fullName, Type broadcastTy, ArrayRef<int64_t> broadcastValid,
    StringRef broadcastName, ArrayRef<int64_t> dstValid, Type elem,
    bool requireNonRowMajor) {
  if (!isRowMajorTileBuf(fullTy))
    return op->emitOpError() << "expects " << fullName
                             << " to use row-major layout when it matches dst";
  if (!rowExpandValidShapesMatch(fullValid, dstValid))
    return op->emitOpError() << "expects " << fullName
                             << " valid_shape to equal dst valid_shape";
  return verifyTRowExpandBroadcastOperand(
      op, broadcastTy, broadcastValid, dstValid, elem, broadcastName,
      requireNonRowMajor);
}

static LogicalResult verifyTRowExpandReduceValidBasics(
    Operation *op, ArrayRef<int64_t> src0Valid, ArrayRef<int64_t> src1Valid,
    ArrayRef<int64_t> dstValid) {
    if (src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2 ||
        dstValid.size() != mlir::pto::kValue2)
        return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
        return op->emitOpError("expects dst valid_shape[0] to be non-zero");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
        return op->emitOpError("expects dst valid_shape[1] to be non-zero");
    return success();
}

static LogicalResult verifyTRowExpandReduceLikeOp(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch, bool enforceTmpContract, StringRef opName,
    bool allowIntegerTypes) {
  auto elemResult = verifyTRowExpandReduceTypes(
      op, src0Ty, src1Ty, dstTy, tmpTy, hasTmp, targetArch, opName,
      allowIntegerTypes);
  if (failed(elemResult))
    return failure();
  Type elem = *elemResult;

  if (!isRowMajorTileBuf(dstTy)) {
    return op->emitOpError("expects dst to use row-major layout");
  }

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2 ||
      dstValid.size() != mlir::pto::kValue2)
      return verifyTRowExpandReduceValidBasics(op, src0Valid, src1Valid, dstValid);

  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. Element
  // type/layout were already checked above; the op writes no elements, so accept
  // and skip the non-empty broadcast/width constraints. One-sided empties still
  // fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }

  if (failed(verifyTRowExpandReduceValidBasics(op, src0Valid, src1Valid,
                                               dstValid)))
    return failure();

  const bool src0MatchesDst = rowExpandValidShapesMatch(src0Valid, dstValid);
  const bool src1MatchesDst = rowExpandValidShapesMatch(src1Valid, dstValid);

  // (A5 tmp-form invariant is checked earlier, before the empty-marker accept.)

  auto verifyTmpContract = [&]() {
    return enforceTmpContract
               ? verifyTRowExpandImplicitTmpContract(
                     op, src0Ty, src1Ty, dstTy, tmpTy, hasTmp, targetArch)
               : success();
  };
  bool requireNonRowMajor = hasTmp && targetArch == PTOArch::A3;

  if (src0MatchesDst) {
    if (succeeded(verifyTRowExpandFullAndBroadcast(
            op, src0Ty, src0Valid, "src0", src1Ty, src1Valid, "src1",
            dstValid, elem, requireNonRowMajor)) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }
  if (src1MatchesDst) {
    if (succeeded(verifyTRowExpandFullAndBroadcast(
            op, src1Ty, src1Valid, "src1", src0Ty, src0Valid, "src0",
            dstValid, elem, requireNonRowMajor)) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }

  return op->emitOpError() << "expects one of src0/src1 to match dst valid_shape"
                           << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


struct OptionalTmpUnaryParseState {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;
};

static ParseResult parseOptionalTmpUnarySyntax(
    OpAsmParser &parser, OperationState &result,
    OptionalTmpUnaryParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(state.tmp)) {
      return failure();
    }
    state.hasTmp = true;
  }
  if (parser.parseColonType(state.srcTy)) {
    return failure();
  }
  if (state.hasTmp &&
      (parser.parseComma() || parser.parseType(state.tmpTy))) {
    return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(state.dst) ||
      parser.parseColonType(state.dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

static ParseResult parseOptionalTmpRowReductionOp(OpAsmParser &parser,
                                                  OperationState &result) {
  OptionalTmpUnaryParseState state;
  if (failed(parseOptionalTmpUnarySyntax(parser, result, state))) {
    return failure();
  }
  return resolveOptionalUnaryAndAddSegments(
      parser, result, state.src, state.srcTy, state.hasTmp, state.tmp,
      state.tmpTy, state.dst, state.dstTy);
}

static void printOptionalTmpRowReductionOp(OpAsmPrinter &p, Operation *op,
                                           Value src, Value tmp, Value dst) {
  p << " ins(" << src;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << src.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseFixedDpsInputs(
    OpAsmParser &parser, unsigned minInputs, unsigned maxInputs,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
    SmallVectorImpl<Type> &inputTypes) {
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  do {
    inputs.emplace_back();
    if (parser.parseOperand(inputs.back())) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  if (inputs.size() < minInputs || inputs.size() > maxInputs ||
      parser.parseColon()) {
    return failure();
  }
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if ((i && parser.parseComma()) || parser.parseType(inputTypes.emplace_back())) {
      return failure();
    }
  }
  return parser.parseRParen();
}
