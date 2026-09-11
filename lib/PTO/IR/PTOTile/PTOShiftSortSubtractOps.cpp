// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyShiftOp(Operation *op, Type src0Ty, Type src1Ty,
                                   Type dstTy, StringRef name) {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyShiftLikeBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
                it.getWidth() != mlir::pto::kValue32)) {
        return op->emitOpError() << "expects " << name << " src0 and src1 element type to be i8/i16/i32";
    }
    return success();
  };
  return dispatchVerifierByArch(op, verify, verify);
}

mlir::LogicalResult mlir::pto::TShlOp::verify() {
  return verifyShiftOp(getOperation(), getSrc0().getType(), getSrc1().getType(),
                       getDst().getType(), "tshl");
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  return verifyShiftOp(getOperation(), getSrc0().getType(), getSrc1().getType(),
                       getDst().getType(), "tshr");
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx"))) {
    return failure();
  }
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5) {
    auto requiredBytes = getStaticByteSize(srcTy);
    if (!requiredBytes) {
      return emitOpError(
          "expects A2/A3 tsort32 src shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(*this, getTmp().getType(),
                                        *requiredBytes))) {
      return failure();
    }
  }

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!(srcElem.isF16() || srcElem.isF32())) {
    return emitOpError() << "expects src and dst element type to be f16 or f32";
  }

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != mlir::pto::kValue32) {
      return emitOpError() << "expects idx element type to be i32/u32";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem))) {
    return emitOpError() << "expects src and dst element type to be float or half";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size()) {
    return emitOpError() << "expects all tensors to have the same rank";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size()) {
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  }
  return mlir::success();
}
static bool ttransUsesTmp(Type srcTy, Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  unsigned elemBytes = getPTOStorageElemByteSize(getElemTy(srcTy));
  if (srcShape.size() != mlir::pto::kValue2 || dstShape.size() != mlir::pto::kValue2 || elemBytes == 0 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) || llvm::is_contained(dstShape, ShapedType::kDynamic)) {
      return true;
  }
  int64_t rowStride = elemBytes == 1 ? 32 : 16;
  int64_t elemPerBlock = 32 / elemBytes;
  int64_t srcStride = srcShape[1];
  int64_t dstStride = dstShape[1];
  return dstStride % rowStride == 0 && srcStride % elemPerBlock == 0 &&
         srcStride / elemPerBlock <= mlir::pto::kValue255;
}

static bool ttransIsAllowedWidthType(Type ty, unsigned elemBytes) {
    if (elemBytes == mlir::pto::kValue4) {
        return ty.isInteger(mlir::pto::kValue32) || ty.isF32();
    }
    if (elemBytes == mlir::pto::kValue2) {
        return ty.isInteger(mlir::pto::kValue16) || ty.isF16() || ty.isBF16();
    }
    return ty.isInteger(mlir::pto::kValue8);
}

static LogicalResult verifyTTransElemWidth(TTransOp op, Type srcElem,
                                           unsigned &elemBytes) {
  elemBytes = getPTOStorageElemByteSize(srcElem);
  if (elemBytes == 0) {
    return op.emitOpError() << "failed to get transpose element size";
  }
  if (elemBytes != 1 && elemBytes != mlir::pto::kValue2 && elemBytes != mlir::pto::kValue4) {
      return op.emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
  }
  if (!ttransIsAllowedWidthType(srcElem, elemBytes)) {
    return op.emitOpError() << "expects transpose element type to match the supported set for its width";
  }
  return success();
}

static LogicalResult verifyTTransAlignedMajor(TTransOp op, Type ty, StringRef name,
                                              unsigned elemBytes) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return success();
  }
  auto shape = getShapeVec(ty);
  if (shape.size() != mlir::pto::kValue2) {
      return success();
  }
  bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[1] : shape[0];
  if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % mlir::pto::kValue32 != 0) {
      return op.emitOpError() << "expects " << name
                              << " major dimension times element size to be 32-byte aligned on A5";
  }
  return success();
}

struct TTransTypes {
  Type src;
  Type tmp;
  Type dst;
  Type elem;
  unsigned elemBytes;
};

static FailureOr<TTransTypes> verifyTTransCommon(TTransOp op,
                                                 StringRef typeError) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (tmpTy && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = tmpTy ? getElemTy(tmpTy) : srcElem;
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem) {
    op.emitOpError() << typeError;
    return failure();
  }
  unsigned elemBytes = 0;
  if (failed(verifyTTransElemWidth(op, srcElem, elemBytes))) {
    return failure();
  }
  return TTransTypes{srcTy, tmpTy, dstTy, srcElem, elemBytes};
}

static LogicalResult verifyTTransA2A3(TTransOp op) {
  auto types = verifyTTransCommon(
      op, "expects src and dst to have the same element type");
  if (failed(types))
    return failure();
  Type srcTy = types->src;
  Type tmpTy = types->tmp;
  Type dstTy = types->dst;
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op.emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
  }
  if (tmpTy) {
    uint64_t requiredBytes = 32;
    if (ttransUsesTmp(srcTy, dstTy)) {
      auto srcBytes = getStaticByteSize(srcTy);
      if (!srcBytes) {
        return op.emitOpError(
            "expects A2/A3 transpose src shape to be static when tmp is used");
      }
      requiredBytes = *srcBytes;
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, requiredBytes))) {
      return failure();
    }
  }
  return mlir::success();
}
