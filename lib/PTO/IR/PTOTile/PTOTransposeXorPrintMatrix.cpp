// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTTransA5(TTransOp op) {
  auto types = verifyTTransCommon(
      op, "expects src, tmp, and dst to have the same element type");
  if (failed(types))
    return failure();
  Type srcTy = types->src;
  Type tmpTy = types->tmp;
  Type dstTy = types->dst;
  unsigned elemBytes = types->elemBytes;
  if (tmpTy && failed(verifyTmpCapacityAtLeast(op, tmpTy, mlir::pto::kValue32))) {
      return failure();
  }
  if (failed(verifyTTransAlignedMajor(op, srcTy, "src", elemBytes)) ||
      failed(verifyTTransAlignedMajor(op, dstTy, "dst", elemBytes))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TTransOp::verify() {
    // The runtime conv-layout TTRANS (TTransImplConvTile in pto-isa) operates
    // on UB-resident plain tiles -- on both A5 and A2A3 every conversion
    // helper takes __ubuf__ pointers and budgets src+dst+tmp against the UB
    // (256KB on A5, 192KB on A2A3) -- while conv tiles live in L1 (mat).
    // Feeding a conv tile into it produces an illegal __cbuf__ -> __ubuf__
    // cast. The A2A3 rewrite that made conv TTRANS valid-shape aware
    // (pto-isa 4db7ef42) kept the UB-only pointer domain, and the A5 path
    // still iterates the declared shape and requires exact src/dst capacity
    // equality, so a conv tile's valid_shape would be silently ignored there
    // anyway. Reject conv tiles outright until the runtime grows an
    // L1-native conv TTRANS.
    if (isa<pto::ConvTileBufType>(getSrc().getType()) || (getTmp() && isa<pto::ConvTileBufType>(getTmp().getType())) ||
        isa<pto::ConvTileBufType>(getDst().getType())) {
        return emitOpError(
            "ttrans does not support conv_tile operands: the runtime conv TTRANS "
            "works on UB-resident plain tiles, while conv tiles live in L1 (mat)");
    }
  auto verifyA2A3 = [this]() -> LogicalResult { return verifyTTransA2A3(*this); };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTTransA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TXorOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/true,
                                     /*addSegmentSizes=*/true);
}

void mlir::pto::TXorOp::print(OpAsmPrinter &p) {
  printOptionalTmpBinaryDpsOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TXorSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseOptionalTmpBinaryDpsOp(parser, result,
                                     /*tmpBeforeDst=*/true,
                                     /*addSegmentSizes=*/true);
}

void mlir::pto::TXorSOp::print(OpAsmPrinter &p) {
  printOptionalTmpBinaryDpsOp(p, getOperation(), getSrc(), getScalar(),
                              getTmp(), getDst());
}

static LogicalResult verifyTXorA2A3(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError(
          "expects tmp to have the same element type as src0, src1, and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects tmp to use row-major layout");
    }
    if (failed(verifyTileBufSameValidShape(
            op, tmpTy, op.getDst().getType(), "tmp", "dst"))) {
      return failure();
    }
    auto requiredBytes = getStaticByteSize(op.getDst().getType());
    if (!requiredBytes) {
      return op.emitOpError(
          "expects A2/A3 txor dst shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
      return failure();
    }
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
              it.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16/i32");
  }
  return success();
}

static LogicalResult verifyTXorA5(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  auto it = mlir::dyn_cast<IntegerType>(*elemOr);
  if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
              it.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
    auto verifyA2A3 = [this]() -> LogicalResult { return verifyTXorA2A3(*this); };
    auto verifyA5 = [this]() -> LogicalResult { return verifyTXorA5(*this); };
    return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTXorSTmp(TXorSOp op, Type elem) {
  if (!op.getTmp())
    return success();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (getElemTy(tmpTy) != elem)
    return op.emitOpError(
            "expects tmp to have the same element type as src and dst");
  if (!isRowMajorTileBuf(tmpTy))
    return op.emitOpError("expects tmp to use row-major layout");
  auto requiredBytes = getStaticByteSize(op.getDst().getType());
  if (!requiredBytes)
    return op.emitOpError(
            "expects A2/A3 txors dst shape to be static when tmp is provided");
  return verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes);
}

static LogicalResult verifyTXorSArch(TXorSOp op, bool isA5) {
  auto elemResult = verifyDistinctRowMajorUnaryTileOpCommon(
      op, op.getSrc(), op.getDst(), "src", "dst");
  if (failed(elemResult))
    return failure();
  Type elem = *elemResult;
  if (!isA5 && failed(verifyTXorSTmp(op, elem)))
    return failure();
  const bool supported =
      hasBitwiseElementWidth(elem, isA5 ? ArrayRef<unsigned>{8, 16, 32}
                                        : ArrayRef<unsigned>{8, 16});
  if (!supported)
    return op.emitOpError(
        isA5 ? "expects A5 txors src and dst element type to be a 1/2/4-byte "
               "integer"
             : "expects A2/A3 txors src and dst element type to be a 1/2-byte "
               "integer");
  return success();
}

mlir::LogicalResult mlir::pto::TXorSOp::verify() {
    auto verifyA2A3 = [this]() { return verifyTXorSArch(*this, false); };
    auto verifyA5 = [this]() { return verifyTXorSArch(*this, true); };
    return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TPrintOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  Type srcTy, tmpTy;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

  if (failed(parseOptionalTmpIns(parser, src, tmp, srcTy, tmpTy, hasTmp))) {
    return failure();
  }
  if (failed(parsePTOInherentAttrs<TPrintOp>(
          parser, result, parsedAttrs, {"printFormat"}))) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPrintOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (Value tmp = getTPrintTmpIfPresent(*this)) {
    p << ", " << tmp << " : " << getSrc().getType() << ", "
      << tmp.getType();
  } else {
    p << " : " << getSrc().getType();
  }
  p << ")";
  NamedAttrList attrs = getNonInherentAttrs(getOperation(), {"printFormat"});
  if (auto printFormatAttr =
          dyn_cast_or_null<pto::PrintFormatAttr>(getProperties().printFormat)) {
    attrs.append("printFormat", printFormatAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  auto srcType = getSrc().getType();
  Value tmp = getTPrintTmpIfPresent(*this);
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(mlir::pto::kValue8) || elem.isInteger(mlir::pto::kValue16) ||
          elem.isInteger(mlir::pto::kValue32))) {
        return emitOpError() << "expects printable tile element type";
    }
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!tmp) {
      if (!space || *space != pto::AddressSpace::VEC) {
        return emitOpError() << "expects printable tile_buf without tmp to be in vec address space";
}
      return success();
    }

    if (!space) {
      return emitOpError() << "expects printable tile_buf with tmp to use a supported address space";
}
    if (*space == pto::AddressSpace::MAT && isTargetArchA5(getOperation())) {
      return emitOpError() << "expects mat tile printing with tmp only on A2/A3 targets";
}
    if (*space != pto::AddressSpace::VEC && *space != pto::AddressSpace::MAT &&
        *space != pto::AddressSpace::ACC) {
      return emitOpError() << "expects printable tile_buf with tmp to be in vec/mat/acc address space";
}
    if (failed(verifyMGatherMScatterMemOperand(getOperation(), tmp, elem, "tmp"))) {
      return failure();
}
    return success();
  }
  if (tmp) {
    return emitOpError() << "expects tmp only when src is a tile_buf";
}
  if (mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType)) {
    return mlir::success();
  }
  return emitOpError() << "expects tile_buf or partition_tensor_view for src";
}

static LogicalResult verifyMatmulOrGemv(Operation *op, Type lhs, Type rhs,
                                        Type dst, bool isGemv,
                                        bool allowLowPrecision = false) {
  LogicalResult operands =
      isGemv ? verifyGemvTileOperands(op, lhs, rhs, dst, allowLowPrecision)
             : verifyMatTileOperands(op, lhs, rhs, dst, allowLowPrecision);
  if (failed(operands) ||
      failed(verifyMatmulTypeTriple(op, getElemTy(lhs), getElemTy(rhs),
                                    getElemTy(dst))))
    return failure();
  return verifyMatmulLike(op, lhs, rhs, dst);
}

LogicalResult mlir::pto::TMatmulOp::verify() {
    auto verifyA2A3 = [this]() -> LogicalResult {
        return verifyMatmulOrGemv(getOperation(), getLhs().getType(), getRhs().getType(), getDst().getType(), false);
    };
    auto verifyA5 = [this]() -> LogicalResult {
        return verifyMatmulOrGemv(
            getOperation(), getLhs().getType(), getRhs().getType(), getDst().getType(), false,
            /*allowLowPrecision=*/true);
    };
    if (failed(dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5)))
        return failure();
    return verifyMatmulTf32Mode(getOperation(), getTf32Mode(), getLhs().getType(),
                                getRhs().getType(), getDst().getType());
}

LogicalResult mlir::pto::TGemvOp::verify() {
    auto verifyA2A3 = [this]() -> LogicalResult {
        return verifyMatmulOrGemv(getOperation(), getLhs().getType(), getRhs().getType(), getDst().getType(), true);
    };
    auto verifyA5 = [this]() -> LogicalResult {
        return verifyMatmulOrGemv(
            getOperation(), getLhs().getType(), getRhs().getType(), getDst().getType(), true,
            /*allowLowPrecision=*/true);
    };
    return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}