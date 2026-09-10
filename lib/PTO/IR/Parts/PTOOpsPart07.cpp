// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

void RlsBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

// ---- GetBufDynOp / RlsBufDynOp parse/print ----
static ParseResult parseBufDynSyncOp(OpAsmParser &parser,
                                     OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr modeAttr;
  auto loc = parser.getCurrentLocation();
  std::string token;
  bool bracketed = false;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc)
             << "invalid get_buf_dyn/rls_buf_dyn token: " << token;
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    bracketed = true;
    if (parser.parseAttribute(opTypeAttr)) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }
  OpAsmParser::UnresolvedOperand bufOperand;
  if (parser.parseComma() || parser.parseOperand(bufOperand) ||
      parser.resolveOperand(bufOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parseI32LiteralAttr(parser, modeAttr)) {
      return failure();
    }
  } else {
    modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
  }
  if ((bracketed && parser.parseRSquare()) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufDynSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                              Value bufId, IntegerAttr modeAttr,
                              ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", " << bufId << ", "
      << modeAttr.getInt();
  } else {
    p << "[" << opTypeAttr << ", " << bufId << ", " << modeAttr.getInt()
      << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "mode"});
}

ParseResult GetBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void GetBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}

ParseResult RlsBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void RlsBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}
// ---- TOp ----
static LogicalResult verifyMatBiasCommon(Operation *op, Type a, Type b,
                                         Type bias, Type dst, bool isGemv,
                                         bool allowLowPrecision = false) {
  LogicalResult operands =
      isGemv ? verifyGemvTileOperands(op, a, b, dst)
             : verifyMatTileOperands(op, a, b, dst, allowLowPrecision);
  if (failed(operands) || failed(verifyMatBiasTile(op, bias, dst)) ||
      failed(verifyMatmulTypeTriple(op, getElemTy(a), getElemTy(b),
                                    getElemTy(dst))))
    return failure();
  return verifyMatmulLike(op, a, b, dst);
}

LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatBiasCommon(getOperation(), getA().getType(),
                               getB().getType(), getBias().getType(),
                               getDst().getType(), true);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyA5MxGemvOperands(Operation *op, Type a, Type b,
                                            Type dst, Type aScale,
                                            Type bScale) {
  if (failed(verifyA5MxGemvTileOperands(op, a, b, dst)) ||
      failed(verifyA5MxGemvScaleTile(op, aScale, a, b, "a_scale",
                                     /*isLeftScale=*/true)) ||
      failed(verifyA5MxGemvScaleTile(op, bScale, a, b, "b_scale",
                                     /*isLeftScale=*/false)))
    return failure();
  return success();
}

static LogicalResult verifyA5Only(
    Operation *op, StringRef opName,
    llvm::function_ref<LogicalResult()> verifyA5Body) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return op->emitOpError() << opName << " is only supported on A5 targets";
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5Body);
}

static LogicalResult verifyA5MxTypeAndMatmulShape(Operation *op, Type a,
                                                  Type b, Type dst) {
  if (failed(verifyA5MxTypeTriple(op, a, b, dst, "lhs", "rhs", "dst")))
    return failure();
  return verifyMatmulLike(op, a, b, dst);
}

static LogicalResult verifyA5MxAccumulator(Operation *op, Type cIn,
                                           Type dst) {
  if (failed(verifyTileBufSameElemType(op, cIn, dst, "c_in", "dst")) ||
      failed(verifyTileBufSameValidShape(op, cIn, dst, "c_in", "dst")))
    return failure();
  return success();
}

static LogicalResult verifyA5MxAccCommon(Operation *op, Type a, Type b,
                                         Type cIn, Type dst, Type aScale,
                                         Type bScale, bool isGemv);
static LogicalResult verifyA5MxBiasBase(Operation *op, Type a, Type b,
                                        Type bias, Type dst, Type aScale,
                                        Type bScale, bool isGemv);

LogicalResult TGemvMxOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvOperands(
            getOperation(), getA().getType(), getB().getType(),
            getDst().getType(), getAScale().getType(),
            getBScale().getType())))
      return failure();
    return verifyA5MxTypeAndMatmulShape(
        getOperation(), getA().getType(), getB().getType(), getDst().getType());
  };
  return verifyA5Only(getOperation(), "tgemv.mx", verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyA5MxAccCommon(
        getOperation(), getA().getType(), getB().getType(), getCIn().getType(),
        getDst().getType(), getAScale().getType(), getBScale().getType(), true);
  };
  return verifyA5Only(getOperation(), "tgemv.mx.acc", verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxBiasBase(
            getOperation(), getA().getType(), getB().getType(),
            getBias().getType(), getDst().getType(), getAScale().getType(),
            getBScale().getType(), true)))
      return failure();
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2) {
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    }
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1]) {
      return emitOpError("expects bias and dst to have the same column shape");
    }
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return verifyA5Only(getOperation(), "tgemv.mx.bias", verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatBiasCommon(getOperation(), getA().getType(),
                               getB().getType(), getBias().getType(),
                               getDst().getType(), false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyMatBiasCommon(getOperation(), getA().getType(),
                               getB().getType(), getBias().getType(),
                               getDst().getType(), false,
                               /*allowLowPrecision=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyA5MxMatOperands(Operation *op, Type a, Type b,
                                           Type dst, Type aScale,
                                           Type bScale) {
  if (failed(verifyA5MxMatTileOperands(op, a, b, dst)) ||
      failed(verifyA5MxMatScaleTiles(op, aScale, bScale, a, b)))
    return failure();
  return success();
}

static LogicalResult verifyA5MxAccCommon(Operation *op, Type a, Type b,
                                         Type cIn, Type dst, Type aScale,
                                         Type bScale, bool isGemv) {
  LogicalResult operands =
      isGemv ? verifyA5MxGemvOperands(op, a, b, dst, aScale, bScale)
             : verifyA5MxMatOperands(op, a, b, dst, aScale, bScale);
  if (failed(verifyAccTileCommon(op, cIn, "c_in")) || failed(operands) ||
      failed(verifyA5MxTypeTriple(op, a, b, dst, "lhs", "rhs", "dst")) ||
      failed(verifyA5MxAccumulator(op, cIn, dst)))
    return failure();
  return verifyMatmulLike(op, a, b, dst);
}

static LogicalResult verifyA5MxBiasBase(Operation *op, Type a, Type b,
                                        Type bias, Type dst, Type aScale,
                                        Type bScale, bool isGemv) {
  LogicalResult operands =
      isGemv ? verifyA5MxGemvOperands(op, a, b, dst, aScale, bScale)
             : verifyA5MxMatOperands(op, a, b, dst, aScale, bScale);
  if (failed(operands) ||
      failed(verifyMatBiasTile(op, bias, dst, /*requireFloatBias=*/true)) ||
      failed(verifyA5MxTypeTriple(op, a, b, dst, "lhs", "rhs", "dst")))
    return failure();
  return success();
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatOperands(
            getOperation(), getA().getType(), getB().getType(),
            getDst().getType(), getAScale().getType(),
            getBScale().getType())))
      return failure();
    return verifyA5MxTypeAndMatmulShape(
        getOperation(), getA().getType(), getB().getType(), getDst().getType());
  };
  return verifyA5Only(getOperation(), "tmatmul.mx", verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyA5MxAccCommon(
        getOperation(), getA().getType(), getB().getType(), getCIn().getType(),
        getDst().getType(), getAScale().getType(), getBScale().getType(), false);
  };
  return verifyA5Only(getOperation(), "tmatmul.mx.acc", verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxBiasBase(
            getOperation(), getA().getType(), getB().getType(),
            getBias().getType(), getDst().getType(), getAScale().getType(),
            getBScale().getType(), false)))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return verifyA5Only(getOperation(), "tmatmul.mx.bias", verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType()) {
      return emitOpError("expects val type to match dst element type");
    }
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType>(srcTy)) {
    return emitOpError("expects src to be tile_buf type");
  }

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace = cast<pto::TileBufType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT) {
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    }
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType()) {
    return emitOpError("expects dst type to match src element type");
  }
  return success();
}

static bool isIntegerWidth(Type ty, unsigned width) {
  auto integer = dyn_cast<IntegerType>(ty);
  return integer && integer.getWidth() == width;
}

static FailureOr<int64_t> getTHistogramByte(THistogramOp op) {
  int64_t byte = 1;
  auto byteAttr = op.getByteAttr();
  if (byteAttr)
    byte = byteAttr.getInt();
  if (auto legacyIsMSB = op->getAttrOfType<BoolAttr>("isMSB")) {
    int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
    if (byteAttr && byte != legacyByte) {
      op.emitOpError(
          "does not allow conflicting 'byte' and legacy 'isMSB' attributes");
      return failure();
    }
    byte = legacyByte;
  }
  if (byte < 0 || byte > 3) {
    op.emitOpError("expects byte to be in range [0, 3]");
    return failure();
  }
  return byte;
}

struct THistogramState {
  pto::TileBufType src;
  pto::TileBufType idx;
  pto::TileBufType dst;
  bool srcIsUi16;
};

static FailureOr<THistogramState> verifyTHistogramTypes(THistogramOp op) {
  Type srcTy = op.getSrc().getType();
  Type idxTy = op.getIdx().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, idxTy, "idx")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type types[] = {srcTy, idxTy, dstTy};
  StringRef names[] = {"src", "idx", "dst"};
  for (auto [type, name] : llvm::zip_equal(types, names)) {
    auto space = getPTOMemorySpaceEnum(type);
    if (!space || *space != pto::AddressSpace::VEC) {
      op.emitOpError() << "expects " << name
                       << " to be in the vec address space";
      return failure();
    }
  }
  auto src = dyn_cast<pto::TileBufType>(srcTy);
  auto idx = dyn_cast<pto::TileBufType>(idxTy);
  auto dst = dyn_cast<pto::TileBufType>(dstTy);
  if (!src || !idx || !dst) {
    op.emitOpError("expects src, idx, and dst to be tile_buf types");
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy)) {
    op.emitOpError("expects src to use row_major + none_box layout");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op.emitOpError("expects dst to use row_major + none_box layout");
    return failure();
  }
  bool srcIsUi16 = isIntegerWidth(getElemTy(srcTy), 16);
  if (!srcIsUi16 && !isIntegerWidth(getElemTy(srcTy), 32)) {
    op.emitOpError("expects src element type to be ui16 or ui32");
    return failure();
  }
  if (!isIntegerWidth(getElemTy(idxTy), 8) ||
      !isIntegerWidth(getElemTy(dstTy), 32)) {
    op.emitOpError(!isIntegerWidth(getElemTy(idxTy), 8)
                       ? "expects idx element type to be ui8"
                       : "expects dst element type to be ui32");
    return failure();
  }
  return THistogramState{src, idx, dst, srcIsUi16};
}

static LogicalResult verifyTHistogramUi16Idx(THistogramOp op,
                                             const THistogramState &state,
                                             int64_t byte) {
  if (byte > 1)
    return op.emitOpError(
        "expects byte to be 0 or 1 when src element type is ui16");
  if (state.idx.getBLayoutValueI32() !=
          static_cast<int32_t>(pto::BLayout::ColMajor) ||
      state.idx.getSLayoutValueI32() !=
          static_cast<int32_t>(pto::SLayout::NoneBox))
    return op.emitOpError(
        "expects idx to use DN layout (col_major + none_box) when src element type is ui16");
  auto srcShape = getShapeVec(state.src);
  auto idxShape = getShapeVec(state.idx);
  auto srcValid = getValidShapeVec(state.src);
  auto idxValid = getValidShapeVec(state.idx);
  if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], idxValid[0]))
    return op.emitOpError(
        "expects idx rows and valid rows to match src when src element type is ui16");
  if (!isKnownUnitExtent(idxShape[1]) ||
      !isKnownZeroOrUnitExtent(idxValid[1]))
    return op.emitOpError(
        "expects idx to have exactly one physical column and 0 or 1 valid column when src element type is ui16");
  return success();
}

static LogicalResult verifyTHistogramUi32Idx(THistogramOp op,
                                             const THistogramState &state,
                                             int64_t byte) {
  if (byte == 3)
    return success();
  if (!isRowMajorTileBuf(state.idx))
    return op.emitOpError(
        "expects idx to use row_major + none_box layout when src element type is ui32 and byte is 0, 1, or 2");
  auto srcShape = getShapeVec(state.src);
  auto idxShape = getShapeVec(state.idx);
  auto srcValid = getValidShapeVec(state.src);
  auto idxValid = getValidShapeVec(state.idx);
  if (!hasCompatibleKnownExtent(srcShape[1], idxShape[1]) ||
      !hasCompatibleKnownExtent(srcValid[1], idxValid[1]))
    return op.emitOpError(
        "expects idx cols and valid cols to match src when src element type is ui32 and byte is 0, 1, or 2");
  int64_t expectedRows = byte == 1 ? 2 : (byte == 0 ? 3 : 1);
  if (!hasCompatibleKnownExtent(idxShape[0], expectedRows) ||
      !hasCompatibleKnownExtentOrZero(idxValid[0], expectedRows))
    return op.emitOpError(
        "expects idx rows to match the byte-selected filter depth and idx valid rows to be 0 or match it when src element type is ui32 and byte is 0, 1, or 2");
  return success();
}

static LogicalResult verifyTHistogramShapes(THistogramOp op,
                                            const THistogramState &state,
                                            int64_t byte) {
  auto srcShape = getShapeVec(state.src);
  auto idxShape = getShapeVec(state.idx);
  auto dstShape = getShapeVec(state.dst);
  auto srcValid = getValidShapeVec(state.src);
  auto idxValid = getValidShapeVec(state.idx);
  auto dstValid = getValidShapeVec(state.dst);
  if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
      srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2)
    return op.emitOpError(
        "expects src, idx, and dst to have rank-2 shape and valid_shape");
  if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], dstValid[0]))
    return op.emitOpError("expects dst rows and valid rows to match src");
  LogicalResult idxResult = state.srcIsUi16
                                ? verifyTHistogramUi16Idx(op, state, byte)
                                : verifyTHistogramUi32Idx(op, state, byte);
  if (failed(idxResult))
    return failure();
  if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256)
    return op.emitOpError("expects dst shape[1] to be at least 256");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 0 &&
      dstValid[1] < 256)
    return op.emitOpError(
        "expects dst valid_shape[1] to be 0 or at least 256");
  return success();
}

static LogicalResult verifyTHistogramA5(THistogramOp op, int64_t byte) {
  auto state = verifyTHistogramTypes(op);
  if (failed(state))
    return failure();
  return verifyTHistogramShapes(op, *state, byte);
}

LogicalResult THistogramOp::verify() {
  auto byte = getTHistogramByte(*this);
  if (failed(byte))
    return failure();

  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() { return verifyTHistogramA5(*this, *byte); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTGetScaleAddrShape(TGetScaleAddrOp op,
                                              pto::AddressSpace srcSpace,
                                              ArrayRef<int64_t> srcShape,
                                              ArrayRef<int64_t> srcValid,
                                              ArrayRef<int64_t> dstShape,
                                              ArrayRef<int64_t> dstValid) {
  if (srcSpace == pto::AddressSpace::LEFT) {
    int64_t scaleK = ceilDivKnown(srcValid[1], 32);
    if (!hasCompatibleKnownExtent(dstShape[0], srcShape[0]) ||
        !hasCompatibleKnownExtent(dstShape[1], scaleK) ||
        !hasCompatibleKnownExtent(dstValid[0], srcValid[0]) ||
        !hasCompatibleKnownExtent(dstValid[1], scaleK))
      return op.emitOpError(
          "expects dst shape/valid_shape to be [M, ceil(K/32)]");
    return success();
  }
  int64_t scaleK = ceilDivKnown(srcValid[0], 32);
  if (!hasCompatibleKnownExtent(dstShape[0], scaleK) ||
      !hasCompatibleKnownExtent(dstShape[1], srcShape[1]) ||
      !hasCompatibleKnownExtent(dstValid[0], scaleK) ||
      !hasCompatibleKnownExtent(dstValid[1], srcValid[1]))
    return op.emitOpError(
        "expects dst shape/valid_shape to be [ceil(K/32), N]");
  return success();
}

static LogicalResult verifyTGetScaleAddrA5(TGetScaleAddrOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", true)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", true)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || (*srcSpace != pto::AddressSpace::LEFT &&
                    *srcSpace != pto::AddressSpace::RIGHT))
    return op.emitOpError(
        "expects src to be in the left or right address space");
  if (!dstSpace || *dstSpace != pto::AddressSpace::SCALING)
    return op.emitOpError("expects dst to be in the scaling address space");
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2 || srcValid.size() != 2 ||
      dstValid.size() != 2)
    return op.emitOpError(
        "expects src/dst to have rank-2 shape and valid_shape");
  return verifyTGetScaleAddrShape(op, *srcSpace, srcShape, srcValid, dstShape,
                                  dstValid);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() { return verifyTGetScaleAddrA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
ParseResult mlir::pto::MScatterOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand idx;
  OpAsmParser::UnresolvedOperand mem;
  Type srcTy, idxTy, memTy;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(idx) || parser.parseColonType(srcTy) ||
      parser.parseComma() || parser.parseType(idxTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(mem) || parser.parseColonType(memTy) ||
      parser.parseRParen() ||
      parsePTOInherentAttrs<MScatterOp>(
          parser, result, parsedAttrs,
          {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"})) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands) ||
      parser.resolveOperand(mem, memTy, result.operands)) {
    return failure();
  }
  return success();
}

void mlir::pto::MScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx() << " : "
    << getSrc().getType() << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  p << ") outs(" << getMem() << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ")";

  NamedAttrList attrs = getNonInherentAttrs(
      getOperation(),
      {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"});
  if (auto coalesceAttr = getMScatterCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto scatterAtomicAttr = getMScatterScatterAtomicOpAttrIfPresent(*this);
      scatterAtomicAttr &&
      scatterAtomicAttr.getValue() != pto::ScatterAtomicOp::None) {
    attrs.append("scatterAtomicOp", scatterAtomicAttr);
  }
  if (auto scatterOobAttr = getMScatterScatterOobAttrIfPresent(*this);
      scatterOobAttr &&
      scatterOobAttr.getValue() != pto::ScatterOOB::Undefined) {
    attrs.append("scatterOob", scatterOobAttr);
  }
  if (auto scatterConflictAttr =
          getMScatterScatterConflictAttrIfPresent(*this)) {
    attrs.append("scatterConflict", scatterConflictAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}
static LogicalResult verifyMScatterAttrs(
    MScatterOp op, Type srcElem, std::optional<pto::Coalesce> coalesce) {
  pto::ScatterAtomicOp atomic = getScatterAtomicOpOrDefault(op);
  pto::ScatterOOB oob = getScatterOobOrDefault(op);
  if (!coalesce &&
      (atomic != pto::ScatterAtomicOp::None ||
       oob != pto::ScatterOOB::Undefined ||
       getScatterConflictAttrIfPresent(op)))
    return op.emitOpError(
        "expects coalesce when scatterAtomicOp/scatterOob/scatterConflict is specified");
  if (getScatterConflictAttrIfPresent(op) &&
      !isTargetArchA5(op.getOperation()))
    return op.emitOpError("expects scatterConflict only on A5 targets");
  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, atomic))
    return op.emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");
  return success();
}

LogicalResult MScatterOp::verify() {
  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1) {
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");
  }

  if (failed(verifyNDStyleVecTile(
          *this, srcTy, "src",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx"))) {
    return failure();
  }

  auto coalesce = getCoalesceIfPresent(*this);

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  if (!srcElem || !idxElem) {
    return emitOpError("failed to resolve element types for src or idx");
  }

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem)) {
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  }

  if (!isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return emitOpError("expects idx element type to be signless i32");
  }

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src"))) {
    return failure();
  }

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src",
                                            coalesce))) {
    return failure();
  }

  return verifyMScatterAttrs(*this, srcElem, coalesce);
}

// ---- MGatherOp ----
// GM -> L1 (cube Mat) gather verifier. The destination is an L1 (loc=mat) tile
// in NZ layout; the index is a GM tensor (the cube core cannot read UB on A5),
// and Coalesce::Elem carries a contiguous GM scratch workspace. Mirrors the
// pto-isa MGATHER GM -> L1 overloads / MGatherCheckGm2L1.
static FailureOr<Type> verifyMGatherGm2L1Dst(Operation *op, Value dst) {
  Type dstTy = dst.getType();
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!dstTb)
    return op->emitOpError("expects GM->L1 mgather dst to be a tile_buf");
  if (!isColMajorRowMajorNZTileBuf(dstTb))
    return op->emitOpError("expects GM->L1 mgather dst (loc=mat) to use "
                           "blayout=col_major and slayout=row_major (NZ)");
  if (dstTb.getSFractalSizeI32() != 512)
    return op->emitOpError("expects GM->L1 mgather dst fractal size to be 512");
  Type dstElem = getElemTy(dstTy);
  if (!dstElem)
    return op->emitOpError("failed to resolve GM->L1 mgather dst element type");
  if (!isSupportedMGatherMScatterPayloadElemType(op, dstElem))
    return op->emitOpError(
        "expects GM->L1 mgather dst element type to be "
        "i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 (and on A5 targets also "
        "float8_e4m3/float8_e5m2 family types)");
  unsigned elemBytes =
      std::max<unsigned>(1u, dstElem.getIntOrFloatBitWidth() / 8u);
  int64_t kC0 = 32 / static_cast<int64_t>(elemBytes);
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == 2) {
    if (kC0 > 0 && dstShape[1] != ShapedType::kDynamic &&
        dstShape[1] % kC0 != 0) {
      return op->emitOpError()
             << "expects GM->L1 mgather dst padded cols to be a multiple of "
             << kC0 << " (C0 = 32 / sizeof(elem))";
    }
    if (dstShape[0] != ShapedType::kDynamic && dstShape[0] % 16 != 0) {
      return op->emitOpError("expects GM->L1 mgather dst padded rows to be a "
                             "multiple of 16 (FRACTAL_NZ_ROW)");
    }
  }

  return dstElem;
}

static LogicalResult verifyMGatherGm2L1Idx(Operation *op, Value idx) {
  Type idxTy = idx.getType();
  if (isa<pto::TileBufType>(idxTy))
    return op->emitOpError("expects GM->L1 mgather idx to be a GM tensor "
                           "partition_tensor_view, not a tile_buf");
  if (!isa<pto::PartitionTensorViewType>(idxTy))
    return op->emitOpError(
        "expects GM->L1 mgather idx to be a partition_tensor_view");
  Type idxElem = getElemTy(idxTy);
  if (!idxElem || !isSupportedMGatherMScatterIndexElemType(idxElem))
    return op->emitOpError("expects GM->L1 mgather idx element type to be i32");
  return success();
}

static LogicalResult verifyMGatherGm2L1Scratch(
    Operation *op, Value scratch, Type dstElem,
    std::optional<pto::Coalesce> coalesce) {
  if (!coalesce)
    return op->emitOpError("expects GM->L1 mgather to specify an explicit "
                           "coalesce attribute (row or elem)");
  if (*coalesce == pto::Coalesce::Elem) {
    if (!scratch)
      return op->emitOpError("expects GM->L1 mgather with coalesce=elem to "
                             "provide a GM scratch operand");
    Type scTy = scratch.getType();
    if (!isa<pto::PartitionTensorViewType>(scTy))
      return op->emitOpError(
          "expects GM->L1 mgather scratch to be a partition_tensor_view");
    Type scElem = getElemTy(scTy);
    if (!scElem || scElem != dstElem)
      return op->emitOpError("expects GM->L1 mgather scratch element type to "
                             "match dst element type");
    return success();
  }
  if (scratch)
    return op->emitOpError("expects GM->L1 mgather with coalesce=row to omit "
                           "the scratch operand");
  return success();
}

static LogicalResult verifyMGatherGm2L1(Operation *op, Value mem, Value idx,
                                        Value dst, Value scratch,
                                        std::optional<pto::Coalesce> coalesce) {
  auto dstElem = verifyMGatherGm2L1Dst(op, dst);
  if (failed(dstElem) ||
      failed(verifyMGatherMScatterMemOperand(op, mem, *dstElem, "dst")) ||
      failed(verifyMGatherGm2L1Idx(op, idx)))
    return failure();
  return verifyMGatherGm2L1Scratch(op, scratch, *dstElem, coalesce);
}
static ParseResult parseMGatherInputs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) {
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));
  if (operands.size() < 2 || operands.size() > 3) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects mgather ins(mem, idx[, scratch])");
  }
  if (parser.parseColon()) {
    return failure();
  }
  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));
  if (operands.size() != types.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "expects the number of ins operands to match the number of ins types");
  }
  return success();
}

static ParseResult resolveMGatherOperands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> inputs, ArrayRef<Type> inputTypes,
    OpAsmParser::UnresolvedOperand dst, Type dstTy) {
  if (parser.resolveOperand(inputs[0], inputTypes[0], result.operands) ||
      parser.resolveOperand(inputs[1], inputTypes[1], result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (inputs.size() == 3 &&
      parser.resolveOperand(inputs[2], inputTypes[2], result.operands)) {
    return failure();
  }
  return success();
}

ParseResult mlir::pto::MGatherOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> insOperands;
  SmallVector<Type, 3> insTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstTy;
  NamedAttrList parsedAttrs;

  if (failed(parseMGatherInputs(parser, insOperands, insTypes))) {
    return failure();
  }

  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parsePTOInherentAttrs<MGatherOp>(
          parser, result, parsedAttrs, {"coalesce", "gatherOob"})) {
    return failure();
  }

  if (failed(resolveMGatherOperands(parser, result, insOperands, insTypes, dst,
                                    dstTy))) {
    return failure();
  }
  return success();
}

void mlir::pto::MGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getMem() << ", " << getIdx();
  if (auto scratch = getScratch()) {
    p << ", " << scratch;
  }
  p << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  if (auto scratch = getScratch()) {
    p << ", ";
    p.printStrippedAttrOrType(scratch.getType());
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"coalesce", "gatherOob"});
  if (auto coalesceAttr = getMGatherCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto gatherOobAttr = getMGatherGatherOobAttrIfPresent(*this);
      gatherOobAttr &&
      gatherOobAttr.getValue() != pto::GatherOOB::Undefined) {
    attrs.append("gatherOob", gatherOobAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

static LogicalResult verifyMGatherGm2Ub(MGatherOp op) {
  Type idxTy = op.getIdx().getType();
  Type dstTy = op.getDst().getType();
  if (op.getScratch())
    return op.emitOpError(
        "expects scratch operand only on GM->L1 (loc=mat) mgather");
  if (failed(verifyNDStyleVecTile(
          op, dstTy, "dst",
          /*allowLowPrecision=*/isTargetArchA5(op.getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(op, idxTy, "idx")))
    return failure();
  auto coalesce = getCoalesceIfPresent(op);
  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  if (!dstElem || !idxElem)
    return op.emitOpError("failed to resolve element types for dst or idx");
  if (!isSupportedMGatherMScatterPayloadElemType(op, dstElem))
    return op.emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return op.emitOpError("expects idx element type to be signless i32");
  if (failed(verifyMGatherMScatterMemOperand(op, op.getMem(), dstElem, "dst")) ||
      failed(verifyMGatherMScatterTileShape(op, dstTy, idxTy, "dst", coalesce)))
    return failure();
  if (getGatherOobOrDefault(op) != pto::GatherOOB::Undefined && !coalesce)
    return op.emitOpError("expects coalesce when gatherOob is specified");
  return success();
}

LogicalResult MGatherOp::verify() {
  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();
  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1)
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");
  auto space = getPTOMemorySpaceEnum(dstTy);
  if (isa<pto::TileBufType>(dstTy) && space &&
      *space == pto::AddressSpace::MAT) {
    std::optional<pto::Coalesce> coalesce;
    if (auto coalesceAttr = getCoalesceAttr()) {
      coalesce = coalesceAttr.getValue();
    }
    return verifyMGatherGm2L1(getOperation(), getMem(), getIdx(), getDst(),
                              getScratch(), coalesce);
  }
  return verifyMGatherGm2Ub(*this);
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

static ParseResult resolveTCvtOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand src, Type srcTy,
    OpAsmParser::UnresolvedOperand tmp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dst, Type dstTy, bool hasTmp) {
  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }
  return parser.resolveOperand(dst, dstTy, result.operands);
}

struct TCvtParseState {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy;
  Type tmpTy;
  Type dstTy;
  bool hasTmp = false;
};

static ParseResult parseTCvtSyntax(OpAsmParser &parser, OperationState &result,
                                  TCvtParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src))
    return failure();
  state.hasTmp = succeeded(parser.parseOptionalComma());
  if (state.hasTmp && parser.parseOperand(state.tmp))
    return failure();
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(state.srcTy))
    return failure();
  if (state.hasTmp && (parser.parseComma() || parser.parseType(state.tmpTy)))
    return failure();
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode"))
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(state.dst) ||
      parser.parseColonType(state.dstTy) || parser.parseRParen())
    return failure();
  return success();
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  TCvtParseState state;
  if (failed(parseTCvtSyntax(parser, result, state)) ||
      failed(resolveTCvtOperands(parser, result, state.src, state.srcTy,
                                 state.tmp, state.tmpTy, state.dst, state.dstTy,
                                 state.hasTmp)))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, state.hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TDeInterleaveOp::print(OpAsmPrinter &p) {
  p << " ins(";
  llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
  p << " : ";
  llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type type) { p << type; });
  p << ") outs(" << getDst0() << ", " << getDst1() << " : "
    << getDst0().getType() << ", " << getDst1().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseTDeInterleaveSources(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcs,
    SmallVectorImpl<Type> &srcTypes) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand src;
  if (parser.parseOperand(src))
    return failure();
  srcs.push_back(src);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(src))
      return failure();
    srcs.push_back(src);
  }
  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();
  srcTypes.push_back(srcType);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(srcType))
      return failure();
    srcTypes.push_back(srcType);
  }
  if (srcs.size() < 1 || srcs.size() > 2)
    return parser.emitError(parser.getCurrentLocation(),
                            "tdeinterleave expects one or two source operands");
  return success();
}

static ParseResult parseTDeInterleaveOutputs(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &dst0,
    OpAsmParser::UnresolvedOperand &dst1, Type &dst0Ty, Type &dst1Ty) {
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst0) || parser.parseComma() ||
      parser.parseOperand(dst1) || parser.parseColonType(dst0Ty) ||
      parser.parseComma() || parser.parseType(dst1Ty) || parser.parseRParen())
    return failure();
  return success();
}

ParseResult mlir::pto::TDeInterleaveOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> srcs;
  SmallVector<Type, 2> srcTypes;
  OpAsmParser::UnresolvedOperand dst0, dst1;
  Type dst0Ty, dst1Ty;
  if (failed(parseTDeInterleaveSources(parser, srcs, srcTypes)) ||
      failed(parseTDeInterleaveOutputs(parser, dst0, dst1, dst0Ty, dst1Ty)))
    return failure();
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dst0, dst0Ty, result.operands) ||
      parser.resolveOperand(dst1, dst1Ty, result.operands))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(srcs.size()), 2}));
  return parser.parseOptionalAttrDict(result.attributes);
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2() || isFormat2WithoutTmp()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    if (getTmp()) {
      p << ", " << getTmp();
    } else {
      p << " no_tmp";
}
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    if (getTmp()) {
      p << ", " << getTmp().getType();
    }
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

struct TMrgSortFormat2State {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs;
  SmallVector<Type, 4> srcTypes;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  OpAsmParser::UnresolvedOperand executed;
  Type tmpTy;
  Type dstTy;
  Type executedTy;
  bool noTmp = false;
  bool exhausted = false;
};

static ParseResult parseTMrgSortFormat1(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  Type srcTy, blockLenTy, dstTy;
  OpAsmParser::UnresolvedOperand dst;
  if (parser.parseType(srcTy) || parser.parseComma() ||
      parser.parseType(blockLenTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0, 0}));
  if (parser.resolveOperand(first, srcTy, result.operands) ||
      parser.resolveOperand(second, blockLenTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
  return success();
}

static ParseResult parseTMrgSortFormat2Inputs(
    OpAsmParser &parser, TMrgSortFormat2State &state,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  state.srcs = {first, second};
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next))
      return failure();
    state.srcs.push_back(next);
  }
  state.noTmp = succeeded(parser.parseOptionalKeyword("no_tmp"));
  size_t min = state.noTmp ? 2 : 3;
  size_t max = state.noTmp ? 4 : 5;
  if (state.srcs.size() < min || state.srcs.size() > max)
    return parser.emitError(
        parser.getCurrentLocation(),
        "tmrgsort format2 expects 2 to 4 src operands and optional no_tmp marker");
  if (!state.noTmp)
    state.tmp = state.srcs.pop_back_val();
  if (succeeded(parser.parseOptionalLBrace())) {
    StringRef keyword;
    if (parser.parseKeyword("exhausted") || parser.parseEqual() ||
        parser.parseKeyword(&keyword) || parser.parseRBrace())
      return failure();
    state.exhausted = keyword == "true";
  }
  return success();
}

static ParseResult parseTMrgSortFormat2TypesAndOutputs(
    OpAsmParser &parser, TMrgSortFormat2State &state) {
  if (parser.parseColon())
    return failure();
  Type type;
  if (parser.parseType(type))
    return failure();
  state.srcTypes.push_back(type);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(type))
      return failure();
    state.srcTypes.push_back(type);
  }
  size_t expectedTypes = state.srcs.size() + (state.noTmp ? 0 : 1);
  if (state.srcTypes.size() != expectedTypes || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  if (!state.noTmp)
    state.tmpTy = state.srcTypes.pop_back_val();
  if (parser.parseOperand(state.dst) || parser.parseComma() ||
      parser.parseOperand(state.executed) || parser.parseColon() ||
      parser.parseType(state.dstTy) || parser.parseComma() ||
      parser.parseType(state.executedTy) || parser.parseRParen())
    return failure();
  return success();
}

static ParseResult resolveTMrgSortFormat2(OpAsmParser &parser,
                                          OperationState &result,
                                          TMrgSortFormat2State &state) {
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(state.srcs.size()), 0, 1,
           state.noTmp ? 0 : 1, 1}));
  if (parser.resolveOperands(state.srcs, state.srcTypes,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(state.dst, state.dstTy, result.operands) ||
      (!state.noTmp && parser.resolveOperand(state.tmp, state.tmpTy,
                                             result.operands)) ||
      parser.resolveOperand(state.executed, state.executedTy,
                            result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted",
                        parser.getBuilder().getBoolAttr(state.exhausted));
  return success();
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(first) || parser.parseComma() ||
      parser.parseOperand(second))
    return failure();
  if (succeeded(parser.parseOptionalColon()))
    return parseTMrgSortFormat1(parser, result, first, second);
  TMrgSortFormat2State state;
  if (failed(parseTMrgSortFormat2Inputs(parser, state, first, second)) ||
      failed(parseTMrgSortFormat2TypesAndOutputs(parser, state)))
    return failure();
  return resolveTMrgSortFormat2(parser, result, state);
}

static LogicalResult verifyTMrgSortFormat1(TMrgSortOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "format1 expects PTO shaped-like types for src/dst";
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src/dst to have the same element type";
  }
  if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects element type to be f16 or f32";
  }
  auto ss = getShapeVec(srcTy);
  auto ds = getShapeVec(dstTy);
  if (ss.size() != 2 || ds.size() != 2) {
    return op.emitOpError() << "expects src/dst to be rank-2 tile-shaped";
  }
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1) {
    return op.emitOpError() << "expects src rows == 1";
  }
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1) {
    return op.emitOpError() << "expects dst rows == 1";
  }
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1]) {
    return op.emitOpError() << "expects src/dst cols to match";
  }
  if (op.getBlockLen()) {
    if (auto cstOp = op.getBlockLen().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
        int64_t v = intAttr.getValue().getSExtValue();
        if (v <= 0 || (v % 64) != 0) {
          return op.emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
  }
  return mlir::success();
}

static LogicalResult verifyTMrgSortOutputShapes(TMrgSortOp op, Type dstTy,
                                                Type tmpTy) {
  auto dstShape = getShapeVec(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  if (dstShape.size() != 2 || (tmpTy && tmpShape.size() != 2))
    return op.emitOpError(
        "format2 expects dst/tmp to be rank-2 tile-shaped");
  if (dstShape[0] != ShapedType::kDynamic && dstShape[0] != 1)
    return op.emitOpError("format2 expects dst/tmp rows == 1");
  if (tmpTy && tmpShape[0] != ShapedType::kDynamic && tmpShape[0] != 1)
    return op.emitOpError("format2 expects dst/tmp rows == 1");
  if (tmpTy && dstShape[1] != ShapedType::kDynamic &&
      tmpShape[1] != ShapedType::kDynamic && tmpShape[1] < dstShape[1])
    return op.emitOpError("format2 expects tmp.cols >= dst.cols");
  return success();
}

static LogicalResult verifyTMrgSortFormat2Outputs(TMrgSortOp op, Type dstTy,
                                                  Type tmpTy) {
  if (!isPTOShapedLike(dstTy) || (tmpTy && !isPTOShapedLike(tmpTy)))
    return op.emitOpError("format2 dst/tmp must be PTO shaped-like");
  auto executedTy = dyn_cast<mlir::VectorType>(op.getExcuted().getType());
  if (!executedTy || executedTy.getRank() != 1 ||
      executedTy.getNumElements() != 4 ||
      !executedTy.getElementType().isInteger(16))
    return op.emitOpError("format2 excuted must be vector<4xi16>");
  Type elemTy = getElemTy(dstTy);
  if (tmpTy && elemTy != getElemTy(tmpTy))
    return op.emitOpError(
        "format2 expects dst/tmp element types to match");
  return verifyTMrgSortOutputShapes(op, dstTy, tmpTy);
}

static LogicalResult verifyTMrgSortFormat2Basics(TMrgSortOp op) {
  for (Value v : op.getSrcs()) {
    if (!isPTOShapedLike(v.getType())) {
      return op.emitOpError() << "format2 expects PTO shaped-like type for each src";
    }
  }
  if (op.getSrcs().size() < 2u || op.getSrcs().size() > 4u) {
    return op.emitOpError() << "format2 expects 2 to 4 srcs";
  }
  if (op.getDsts().size() != 1u || !op.getExcuted()) {
    return op.emitOpError()
           << "format2 expects 2 to 4 srcs, one dst, and excuted=vector";
  }
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  return verifyTMrgSortFormat2Outputs(op, dstTy, tmpTy);
}

static LogicalResult verifyTMrgSortFormat2Srcs(TMrgSortOp op) {
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type elemTy = getElemTy(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  int64_t requiredTmpCols = 0;
  for (Value src : op.getSrcs()) {
    Type srcTy = src.getType();
    auto srcShape = getShapeVec(srcTy);
    auto srcValidShape = getValidShapeVec(src);
    if (srcShape.size() != 2 || srcValidShape.size() != 2) {
      return op.emitOpError() << "format2 expects src to be rank-2 tile-shaped";
    }
    if (srcShape[0] != mlir::ShapedType::kDynamic && srcShape[0] != 1) {
      return op.emitOpError() << "format2 expects src rows == 1";
    }
    if (getElemTy(srcTy) != elemTy) {
      return op.emitOpError() << "format2 expects src/dst/tmp element types to match";
    }
    if (srcValidShape[1] == mlir::ShapedType::kDynamic) {
      requiredTmpCols = mlir::ShapedType::kDynamic;
    } else if (requiredTmpCols != mlir::ShapedType::kDynamic) {
      requiredTmpCols += srcValidShape[1];
    }
  }
  if (tmpTy && requiredTmpCols != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < requiredTmpCols) {
    return op.emitOpError()
           << "format2 expects tmp.cols >= sum(src.cols) = "
           << requiredTmpCols;
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (isFormat1()) {
    return verifyTMrgSortFormat1(*this);
  }
  if (isFormat2() || isFormat2WithoutTmp()) {
    if (failed(verifyTMrgSortFormat2Basics(*this))) {
      return failure();
    }
    return verifyTMrgSortFormat2Srcs(*this);
  }
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return emitOpError() << "failed to get element type for src/dst";
  }
  if (srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!mlir::isa<IntegerType>(srcElem)) {
    return emitOpError() << "expects integral element types";
  }
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0) {
    return emitOpError("expects tshls scalar to be non-negative");
  }
  return mlir::success();
}

static FailureOr<Type> verifyMatchingVecUnaryTiles(Operation *op, Type srcTy,
                                                   Type dstTy) {
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  return verifyMatchingElementTypes(op, srcTy, dstTy);
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingVecUnaryTiles(getOperation(), getSrc().getType(),
                                       getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTNegArch(TNegOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (isA5) {
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return op.emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1])
      return op.emitOpError(
          "expects src and dst to have the same valid_shape[1]");
  } else if (failed(
                 verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(srcTy);
  bool supported = isA5
                       ? isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                                /*allowInt8=*/true)
                       : (elemTy.isInteger(16) || elemTy.isInteger(32) ||
                          elemTy.isF16() || elemTy.isF32());
  if (!supported)
    return op.emitOpError(isA5
        ? "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16"
        : "expects A2/A3 tneg element type to be i16/i32/f16/f32");
  return success();
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() { return verifyTNegArch(*this, false); };
  auto verifyA5 = [&]() { return verifyTNegArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyCommon = [&]() {
    return verifyMatchingVecUnaryTiles(getOperation(), getSrc().getType(),
                                       getDst().getType());
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto elemTy = verifyCommon();
    return failed(elemTy)
               ? failure()
               : verifyIntegerWidths(getOperation(), *elemTy, {16},
                                     "expects A2/A3 tnot element type to be i16");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto elemTy = verifyCommon();
    return failed(elemTy)
               ? failure()
               : verifyIntegerWidths(
                     getOperation(), *elemTy, {8, 16, 32},
                     "expects A5 tnot element type to be i8/i16/i32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  return verifyBitwiseBinaryOp(getOperation(), getSrc0().getType(),
                               getSrc1().getType(), getDst().getType(), "tor");
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  // ORS has the same operand contract as ANDS; diagnostics intentionally omit
  // the scalar noun for compatibility with the existing verifier messages.
  auto verifyFor = [&](bool isA5) -> LogicalResult {
    auto elem = verifyDistinctRowMajorUnaryTileOpCommon(
        getOperation(), getSrc(), getDst(), "src", "dst");
    if (failed(elem))
      return failure();
    return verifyIntegerWidths(
        getOperation(), *elem,
        isA5 ? ArrayRef<unsigned>{8, 16, 32} : ArrayRef<unsigned>{8, 16},
        isA5 ? "expects A5 tors src and dst element type to be i8/i16/i32"
             : "expects A2/A3 tors src and dst element type to be i8/i16");
  };
  auto verifyA2A3 = [&]() { return verifyFor(false); };
  auto verifyA5 = [&]() { return verifyFor(true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op->emitOpError(
               "expects src0/src1/dst to be tensor/tile_buf/tile_view types"),
           failure();
  }
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for operands"), failure();
  }
  if (e0 != e1 || e0 != ed) {
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  }
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd) {
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  }
  return e0;
}

static LogicalResult verifyTPartBinaryA2A3(Operation *op, Type src0Ty,
                                           Type src1Ty, Type dstTy,
                                           StringRef opName) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op->emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op->emitOpError()
           << "expects src0/src1/dst to have the same element type";
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op->emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op->emitOpError()
           << "expects A2/A3 " << opName
           << " element type to be i32/i16/f16/f32";
  }
  return mlir::success();
}

static LogicalResult verifyTPartBinaryA5(Operation *op, Type src0Ty,
                                         Type src1Ty, Type dstTy,
                                         StringRef opName) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy))
    return op->emitOpError() << "expects PTO shaped-like src0/src1/dst";
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy))
    return op->emitOpError()
           << "expects src0/src1/dst to have the same element type";
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32()))
    return op->emitOpError()
           << "expects A5 " << opName
           << " element type to be i32/i16/i8/f16/bf16/f32";
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
    return op->emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  return verifyPartialValidPatternLoose(op, src0Ty, src1Ty, dstTy);
}

static LogicalResult verifyTPartAddA2A3(TPartAddOp op) {
  return verifyTPartBinaryA2A3(op, op.getSrc0().getType(),
                               op.getSrc1().getType(), op.getDst().getType(),
                               "tpartadd");
}
