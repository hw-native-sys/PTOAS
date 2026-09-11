// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTGatherArch(TGatherOp op, bool isA5) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue()) {
      return op.emitOpError("mask-pattern tgather only allows src and dst operands");
    }
    return verifyTGatherMaskForm(op, /*allowA5MaskTypes=*/isA5);
  }
  if (op.getAxisAttr()) {
    return op.emitOpError("axis attribute must not be provided without maskPattern");
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp()) {
      return op.emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
    }
    if (op.getIndices()) {
      return op.emitOpError("compare-form tgather does not take indices");
    }
    return verifyTGatherCompareForm(op, /*allowA5SrcTypes=*/isA5);
  }
  if (!op.getIndices())
    return op.emitOpError(isA5 ? "index-form tgather expects indices"
                               : "index-form tgather expects both indices and tmp");
  if (!isA5 && !op.getTmp())
    return op.emitOpError("index-form tgather expects both indices and tmp");
  return verifyTGatherIndexForm(op, /*allow16BitIndices=*/isA5,
                                /*allowA5ElemTypes=*/isA5);
}

llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto verifyA2A3 = [&]() { return verifyTGatherArch(*this, false); };
  auto verifyA5 = [&]() { return verifyTGatherArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static std::optional<unsigned> tgatherbElemBytes(Type ty) {
  unsigned elemBytes = getPTOStorageElemByteSize(ty);
  if (elemBytes == 0) {
    return std::nullopt;
  }
  return elemBytes;
}

static FailureOr<std::pair<Type, Type>> verifyTGatherBCommon(TGatherBOp op) {
  Type srcTy = op.getSrc().getType();
  Type offTy = op.getOffsets().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, offTy, "offsets")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto srcElemTy = getElemTy(srcTy);
  auto dstElemTy = getElemTy(dstTy);
  if (!srcElemTy || !dstElemTy) {
    return op.emitOpError() << "failed to get element type for src/dst";
  }
  return std::make_pair(srcElemTy, dstElemTy);
}

static LogicalResult verifyTGatherBA2A3(TGatherBOp op) {
  FailureOr<std::pair<Type, Type>> elems = verifyTGatherBCommon(op);
  if (failed(elems)) {
    return failure();
  }
  Type offTy = op.getOffsets().getType();
  Type dstTy = op.getDst().getType();
  Type dstElemTy = elems->second;
  if (!isRowMajorTileBuf(dstTy) || !isRowMajorTileBuf(offTy)) {
    return op.emitOpError()
           << "expects dst and offsets to use row-major layout";
  }
  auto dstBytes = tgatherbElemBytes(dstElemTy);
  if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4)) {
    return op.emitOpError()
           << "expects A2/A3 dst element size to be 1, 2, or 4 bytes";
  }
  Type offElemTy = getElemTy(offTy);
  if (!offElemTy.isInteger(32)) {
    return op.emitOpError() << "expects offsets element type to be i32";
  }

  auto dstValid = getValidShapeVec(dstTy);
  auto offValid = getValidShapeVec(offTy);
  if (dstValid.size() != 2 || offValid.size() != 2) {
    return op.emitOpError() << "expects rank-2 src/offsets/dst tile buffers";
  }
  if (dstValid[0] != ShapedType::kDynamic &&
      offValid[0] != ShapedType::kDynamic && dstValid[0] != offValid[0]) {
    return op.emitOpError() << "expects offsets valid rows to match dst valid rows";
  }
  if (dstValid[1] != ShapedType::kDynamic &&
      offValid[1] != ShapedType::kDynamic) {
    int64_t blockElems = 32 / *dstBytes;
    int64_t blocks = (dstValid[1] + blockElems - 1) / blockElems;
    int64_t expectedOffsetCols = ((blocks + 7) / 8) * 8;
    if (offValid[1] != expectedOffsetCols) {
      return op.emitOpError()
             << "expects offsets valid cols to be compact 32B block address "
                "count padded to 8 entries; expected "
             << expectedOffsetCols << ", got " << offValid[1];
    }
  }
  return mlir::success();
}

static LogicalResult verifyTGatherBA5(TGatherBOp op) {
  FailureOr<std::pair<Type, Type>> elems = verifyTGatherBCommon(op);
  if (failed(elems)) {
    return failure();
  }
  Type dstElemTy = elems->second;
  auto dstBytes = tgatherbElemBytes(dstElemTy);
  if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4)) {
    return op.emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTGatherBA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTGatherBA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  return verifyF16F32VecUnary(getOperation(), getSrc().getType(),
                              getDst().getType());
}

static LogicalResult verifyTLReluCommon(TLReluOp op, StringRef typeError) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return op.emitOpError() << typeError;
  return success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTLReluCommon(
            *this, "expects A2/A3 tlrelu element type to be f16 or f32")))
      return failure();
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2) {
      return emitOpError("expects src to have rank-2 valid_shape");
    }
    if (valid[0] != ShapedType::kDynamic && valid[0] < 0) {
      return emitOpError("expects src valid_shape[0] to be non-negative");
    }
    if (valid[1] != ShapedType::kDynamic && valid[1] < 0) {
      return emitOpError("expects src valid_shape[1] to be non-negative");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyTLReluCommon(
            *this, "expects A5 tlrelu element type to be f16 or f32")))
      return failure();
    if (!getSlope().getType().isF32()) {
      return emitOpError() << "expects slope to have type f32";
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmax element type to be i32/i16/f16/f32",
      "expects A5 tmax element type to be i32/i16/i8/f16/f32");
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmaxs element type to be i32/i16/f16/f32",
      "expects A5 tmaxs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmin element type to be i32/i16/f16/f32",
      "expects A5 tmin element type to be i32/i16/i8/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmins element type to be i32/i16/f16/f32",
      "expects A5 tmins element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

static std::optional<int64_t> tmovCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> tmovCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> tmovAlign16(int64_t value) {
  auto biased = tmovCheckedAdd(value, 15);
  return biased ? tmovCheckedMul(*biased / 16, 16) : std::nullopt;
}

static std::optional<int64_t> tmovCheckedElements(ArrayRef<int64_t> shape) {
  if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0 ||
      (shape[1] != 0 &&
       shape[0] > std::numeric_limits<int64_t>::max() / shape[1])) {
    return std::nullopt;
  }
  return shape[0] * shape[1];
}

static LogicalResult verifyTMovXToZzElemLayout(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  auto srcTb = cast<pto::TileBufType>(srcTy);
  auto dstTb = cast<pto::TileBufType>(dstTy);
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type tmpElem = getElemTy(fp.getType());
  if (srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError("expects src, dst, and tmp to share one element type");
  }
  bool validElem = isPTOHiFloat8Type(srcElem) || isPTOF8E8M0Type(srcElem);
  if (auto integer = dyn_cast<IntegerType>(srcElem)) {
    validElem = integer.getWidth() == 8 &&
                integer.getSignedness() == IntegerType::Unsigned;
  }
  if (!validElem) {
    return op.emitOpError("expects element type to be one of ui8, !pto.hif8, !pto.f8E8M0 (i8 lowers to int8_t, which PTO-ISA CommonCheckZZ rejects)");
  }
  if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op.emitOpError("expects src to use blayout=row_major, slayout=none_box");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects dst to use blayout=row_major, slayout=row_major (ZZ box)");
  }
  return success();
}

static bool isStaticTMovShape(Type type, bool checkValid) {
  if (llvm::is_contained(getShapeVec(type), ShapedType::kDynamic))
    return false;
  return !checkValid ||
         !llvm::is_contained(getValidShapeVec(type), ShapedType::kDynamic);
}
