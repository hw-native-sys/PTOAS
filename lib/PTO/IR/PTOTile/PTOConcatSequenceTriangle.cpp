// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTConcatidxElementTypes(TConcatidxOp op,
                                                  Type dataElem,
                                                  Type idxElem) {
    // Data element type: f16, f32, bf16, i8, i16, i32 (signless).
    if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
      auto it = mlir::dyn_cast<IntegerType>(dataElem);
      if (!it || !it.isSignless() ||
          (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
           it.getWidth() != mlir::pto::kValue32)) {
          return op.emitOpError() << "expects data element type to be i8, i16, i32, f16, f32, or bf16";
      }
    }

    // Index element type: i8, i16, i32 (signless).
    auto it = mlir::dyn_cast<IntegerType>(idxElem);
    if (!it || !it.isSignless() ||
        (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
         it.getWidth() != mlir::pto::kValue32)) {
        return op.emitOpError() << "expects index element type to be i8, i16, or i32";
    }
    return success();
}

static LogicalResult verifyTConcatidxLocVec(TConcatidxOp op, Type ty,
                                            StringRef name) {
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC) {
      return op.emitOpError() << "expects " << name << " to use loc=vec";
    }
    return success();
}

static LogicalResult verifyTConcatidxArch(TConcatidxOp op,
                                          bool requireRowMajor) {
    auto elemOr = verifyTConcatidxCommon(op);
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyTConcatidxLocVec(op, op.getSrc0().getType(), "src0")) ||
        failed(verifyTConcatidxLocVec(op, op.getSrc1().getType(), "src1")) ||
        failed(verifyTConcatidxLocVec(op, op.getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyTConcatidxLocVec(op, op.getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyTConcatidxLocVec(op, op.getDst().getType(), "dst"))) {
      return failure();
    }
    if (requireRowMajor &&
        (!isRowMajorTileBuf(op.getSrc0().getType()) ||
         !isRowMajorTileBuf(op.getSrc1().getType()) ||
         !isRowMajorTileBuf(op.getSrc0Idx().getType()) ||
         !isRowMajorTileBuf(op.getSrc1Idx().getType()) ||
         !isRowMajorTileBuf(op.getDst().getType()))) {
      return op.emitOpError(
          "expects all operands to use row-major layout");
    }
    return verifyTConcatidxElementTypes(op, elemOr->first, elemOr->second);
}

LogicalResult pto::TConcatidxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTConcatidxArch(*this, /*requireRowMajor=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTConcatidxArch(*this, /*requireRowMajor=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  return verifyBitwiseScalarOp(getOperation(), getSrc(), getDst(), "tands");
}

struct TCILikeParseState {
  OpAsmParser::UnresolvedOperand source;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type sourceType;
  Type tmpType;
  Type dstType;
  bool hasTmp = false;
};

static ParseResult parseTCILikeSyntax(OpAsmParser &parser,
                                     OperationState &result,
                                     TCILikeParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.source)) {
    return failure();
  }
  state.hasTmp = succeeded(parser.parseOptionalComma());
  if (state.hasTmp && parser.parseOperand(state.tmp)) {
    return failure();
  }
  if (parser.parseColonType(state.sourceType)) {
    return failure();
  }
  if (state.hasTmp) {
    if (parser.parseComma() || parser.parseType(state.tmpType)) {
      return failure();
    }
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(state.dst) || parser.parseColonType(state.dstType) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

static ParseResult resolveOptionalUnaryAndAddSegments(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand source, Type sourceType, bool hasTmp,
    OpAsmParser::UnresolvedOperand tmp, Type tmpType,
    OpAsmParser::UnresolvedOperand dst, Type dstType) {
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpType, result.operands)) ||
      parser.resolveOperand(dst, dstType, result.operands))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

static ParseResult parseTCILikeOp(OpAsmParser &parser, OperationState &result) {
  TCILikeParseState state;
  if (failed(parseTCILikeSyntax(parser, result, state)))
    return failure();

  return resolveOptionalUnaryAndAddSegments(
      parser, result, state.source, state.sourceType, state.hasTmp, state.tmp,
      state.tmpType, state.dst, state.dstType);
}

static void printTCILikeOp(OpAsmPrinter &p, Operation *op, Value s, Value tmp,
                           Value dst) {
  p << " ins(" << s;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << s.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TCIOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTCILikeOp(parser, result);
}

void mlir::pto::TCIOp::print(OpAsmPrinter &p) {
  printTCILikeOp(p, getOperation(), getOperand(0), getTmp(), getDst());
}

static LogicalResult verifyTCITmp(TCIOp op, unsigned bitWidth) {
    auto tmpTy = mlir::dyn_cast<TileBufType>(op.getTmp().getType());
    if (!tmpTy) {
      return op.emitOpError("expects tmp to be a tile buffer");
    }
    auto tmpSpace =
        mlir::dyn_cast_or_null<AddressSpaceAttr>(tmpTy.getMemorySpace());
    if (!tmpSpace || tmpSpace.getAddressSpace() != AddressSpace::VEC) {
      return op.emitOpError("expects tmp to be in vec address space");
    }
    Type tmpElemTy = tmpTy.getElementType();
    if (!(tmpElemTy.isF32() || tmpElemTy.isInteger(mlir::pto::kValue32))) {
        return op.emitOpError("expects A2/A3 tmp element type to be a 4-byte type");
    }
    if (tmpTy.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor)) {
      return op.emitOpError("expects tmp blayout to be row_major");
    }
    if (tmpTy.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox)) {
      return op.emitOpError("expects tmp slayout to be none_box");
    }
    if (tmpTy.getSFractalSizeI32() != 512) {
      return op.emitOpError("expects tmp fractal size to be 512");
    }
    auto tmpBytes = getStaticByteSize(tmpTy);
    if (!tmpBytes) {
      return op.emitOpError("expects tmp to have static byte size");
    }
    uint64_t minTmpBytes = bitWidth == 32 ? 768 : 1792;
    if (*tmpBytes < minTmpBytes) {
      return op.emitOpError("expects A2/A3 tmp capacity to be at least ")
             << minTmpBytes << " bytes for " << bitWidth
             << "-bit dst element type";
    }
    return success();
}

LogicalResult pto::TCIOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
      (getTmp() && failed(verifyTileBufCommon(*this, getTmp().getType(), "tmp")))) {
    return failure();
  }
  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy) {
    return emitOpError("expects dst element type to be integer");
  }
  unsigned bw = elemTy.getWidth();
  if (bw != mlir::pto::kValue16 && bw != mlir::pto::kValue32) {
      return emitOpError("expects dst element type to be i16/i32");
  }
  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5 &&
      failed(verifyTCITmp(*this, bw))) {
    return failure();
  }

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy) {
    return emitOpError("expects S to be integer");
  }

  if (sTy != elemTy) {
    return emitOpError("expects S and dst element type to be exactly the same type");
  }
  auto shape = getShapeVec(dstTy);
  if (shape.size() != mlir::pto::kValue2) {
      return emitOpError("expects dst to be rank-2");
  }
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1) {
    return emitOpError("expects dst cols to be different from 1");
  }

  return success();
}

LogicalResult pto::TTriOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy) {
    return emitOpError("expects diagonal to be an integer operand");
  }

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1) {
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");
  }

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false)) {
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        }
        return success();
      },
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true)) {
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        }
        return success();
      });
}
