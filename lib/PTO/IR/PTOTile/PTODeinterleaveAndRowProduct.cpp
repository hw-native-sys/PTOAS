// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

    bool mismatchedElementTypes =
        failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedElementTypes) {
      return failure();
    }
    if (!isSupportedVecElemType(getElemTy(src0Ty), /*allowBf16=*/true,
                                /*allowInt8=*/true)) {
      return emitOpError("expects vec tile element types to be supported");
    }

    bool mismatchedValidShapes =
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedValidShapes) {
      return failure();
    }

    auto validShape = getValidShapeVec(dst0Ty);
    bool hasInvalidRank = validShape.size() != 2;
    if (hasInvalidRank) {
      return emitOpError("expects src0, src1, dst0, and dst1 to have rank-2 valid_shape");
    }
    bool hasOddValidColumns =
        validShape[1] != ShapedType::kDynamic && (validShape[1] & 1) != 0;
    if (hasOddValidColumns) {
      return emitOpError("expects valid_shape[1] to be even");
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTDeInterleaveCommon(TDeInterleaveOp op) {
  Type srcTy = op.getSrc0().getType();
  Type dst0Ty = op.getDst0().getType();
  Type dst1Ty = op.getDst1().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src0")) ||
      failed(verifyVecTileCommon(op, dst0Ty, "dst0")) ||
      failed(verifyVecTileCommon(op, dst1Ty, "dst1")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dst0Ty, "src0", "dst0")) ||
      failed(verifyTileBufSameElemType(op, srcTy, dst1Ty, "src0", "dst1")))
    return failure();
  if (!isSupportedVecElemType(getElemTy(srcTy), true, true))
    return op.emitOpError("expects vec tile element types to be supported");
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dst0Ty) ||
      !isRowMajorTileBuf(dst1Ty))
    return op.emitOpError(
        "expects src and dst tiles to use row-major layout");
  if (getValidShapeVec(srcTy).size() != mlir::pto::kValue2 || getValidShapeVec(dst0Ty).size() != mlir::pto::kValue2 ||
      getValidShapeVec(dst1Ty).size() != mlir::pto::kValue2)
      return op.emitOpError("expects src and dst tiles to have rank-2 valid_shape");
  return success();
}

static LogicalResult verifyTDeInterleaveTwoSources(TDeInterleaveOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  if (failed(verifyVecTileCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, op.getDst0().getType(),
                                         "src0", "dst0")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, op.getDst1().getType(),
                                         "src0", "dst1")))
    return failure();
  if (!isRowMajorTileBuf(src1Ty))
    return op.emitOpError("expects src1 to use row-major layout");
  int64_t columns = getValidShapeVec(src0Ty)[1];
  if (columns != ShapedType::kDynamic && (columns & 1) != 0)
    return op.emitOpError("expects two-source valid_shape[1] to be even");
  return success();
}

static LogicalResult verifyTDeInterleaveSingleOutput(
    TDeInterleaveOp op, ArrayRef<int64_t> srcValid,
    ArrayRef<int64_t> dstValid, StringRef name) {
  if (!hasCompatibleKnownExtent(srcValid[0], dstValid[0]))
    return op.emitOpError() << "expects src0 and " << name
                            << " to have the same valid_shape[0]";
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      dstValid[1] != srcValid[1] / mlir::pto::kValue2)
      return op.emitOpError() << "expects " << name << " valid_shape[1] to be half of src0 valid_shape[1]";
  return success();
}

static LogicalResult verifyTDeInterleaveA5(TDeInterleaveOp op) {
  if (failed(verifyTDeInterleaveCommon(op)))
    return failure();
  if (op.getSrcs().size() == mlir::pto::kValue2)
      return verifyTDeInterleaveTwoSources(op);
  auto srcValid = getValidShapeVec(op.getSrc0());
  if (srcValid[1] != ShapedType::kDynamic && (srcValid[1] & 1) != 0)
    return op.emitOpError("expects single-source valid_shape[1] to be even");
  if (failed(verifyTDeInterleaveSingleOutput(
          op, srcValid, getValidShapeVec(op.getDst0()), "dst0")))
    return failure();
  return verifyTDeInterleaveSingleOutput(
      op, srcValid, getValidShapeVec(op.getDst1()), "dst1");
}

mlir::LogicalResult mlir::pto::TDeInterleaveOp::verify() {
  auto verifyA2A3 = [&]() {
    return emitOpError("tdeinterleave is only supported on A5 targets");
  };
  auto verifyA5 = [&]() { return verifyTDeInterleaveA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A5 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32())) {
    return emitOpError("expects element type to be f16 or f32");
  }
  if (getPrecisionType() == pto::RsqrtPrecision::HighPrecision && !getTmp()) {
    return emitOpError("expects tmp when precisionType is high_precision");
  }
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp"))) {
      return failure();
    }

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value()) {
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    }
    if (tmpElemBytes.value() * tmpNumel.value() < mlir::pto::kValue32) {
        return emitOpError("expects tmp to be at least 32 bytes when provided");
    }
  }
  return mlir::success();
}


static bool isTScatterAllowedDataElem(mlir::Type t) {
  if (t.isF16() || t.isF32() || t.isBF16()) {
    return true;
  }
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      return (
          it.getWidth() == mlir::pto::kValue8 || it.getWidth() == mlir::pto::kValue16 ||
          it.getWidth() == mlir::pto::kValue32);
  }
  return false;
}

static bool isTScatterAllowedIndexElem(mlir::Type t) {
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      return (it.getWidth() == mlir::pto::kValue16 || it.getWidth() == mlir::pto::kValue32);
  }
  return false;
}

static LogicalResult verifyTScatterIndexedElemTypes(TScatterOp op, Type ts,
                                                    Type ti, Type td) {
  Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
  if (!srcElem || !dstElem || !idxElem) {
    return op.emitOpError("failed to get element type for operands");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src/dst to have the same element type");
  }

  if (!isTScatterAllowedDataElem(srcElem)) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  if (!isTScatterAllowedIndexElem(idxElem)) {
    return op.emitOpError("expects indexes element type to be i16/i32");
  }

  auto bwData = getPTOStorageElemBitWidth(srcElem);
  auto bwIdx  = getPTOStorageElemBitWidth(idxElem);
  if (bwData != mlir::pto::kValue8 && bwData != mlir::pto::kValue16 && bwData != mlir::pto::kValue32) {
      return op.emitOpError("unexpected src/dst element bitwidth");
  }

  unsigned dataBytes = bwData / 8;
  unsigned idxBytes  = bwIdx / 8;
  unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
  if (idxBytes != expectedIdxBytes) {
    return op.emitOpError("expects indexes element size to match the documented scatter rule");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedShapes(TScatterOp op, Type ts, Type ti,
                                                 Type td) {
  auto srcValid = getValidShapeVec(ts);
  auto idxValid = getValidShapeVec(ti);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != mlir::pto::kValue2 || idxValid.size() != mlir::pto::kValue2 ||
      dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects src, indexes and dst to have rank-2 valid_shape");
  }

  for (unsigned d = 0; d < mlir::pto::kValue2; ++d) {
      if (srcValid[d] != ShapedType::kDynamic && idxValid[d] != ShapedType::kDynamic && srcValid[d] != idxValid[d]) {
          return op.emitOpError("expects src and indexes to have the same valid_shape");
      }
      if (srcValid[d] != ShapedType::kDynamic && dstValid[d] != ShapedType::kDynamic && dstValid[d] < srcValid[d]) {
          return op.emitOpError("expects dst valid_shape to be >= src valid_shape");
      }
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type ti = op.getIndexes().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, ti, "indexes")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }
  if (failed(verifyTScatterIndexedElemTypes(op, ts, ti, td))) {
    return failure();
  }
  return verifyTScatterIndexedShapes(op, ts, ti, td);
}
