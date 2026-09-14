// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
