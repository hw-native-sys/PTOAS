// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTColSumArch(TColSumOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
    return failure();
  }
  if (op.getTmp() && !op.getIsBinaryAttr()) {
    return op.emitOpError("tmp operand requires isBinary attribute");
  }
  if (failed(verifyTColSumTmp(op, srcTy, dstTy))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects src/dst element types to match");
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/isA5))) {
    return failure();
  }
  Type elem = getElemTy(srcTy);
  bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                   elem.isInteger(32) ||
                   (isA5 && (elem.isBF16() || elem.isInteger(8)));
  if (!supported) {
    return op.emitOpError(isA5
        ? "expects A5 tcolsum element type to be i8/i16/i32/f16/bf16/f32"
        : "expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
  }
  return success();
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTColSumArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTColSumArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColProdOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/false,
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolprod element type to be f16/f32/i16/i32",
      "expects A5 tcolprod element type to be i16/ui16/i32/ui32/f16/bf16/f32");
}

static bool tcvtIsResolvedSubview(Value value) {
  auto alloc = value.getDefiningOp<pto::AllocTileOp>();
  auto semantics =
      alloc ? alloc->getAttrOfType<StringAttr>("pto.view_semantics")
            : StringAttr();
  return semantics && semantics.getValue() == "subview";
}

static bool tcvtNeedsTmp(TCvtOp op, Type srcElem, Type dstElem) {
  if (op.getSatMode() != pto::SaturationMode::OFF) {
    return false;
  }
  return (srcElem.isF32() && dstElem.isInteger(mlir::pto::kValue16)) ||
         (srcElem.isF16() && (dstElem.isInteger(mlir::pto::kValue16) || dstElem.isInteger(mlir::pto::kValue8)));
}

static int64_t computeTCvtTmpRequiredBytes(Type srcElem, Type dstElem,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> dstValid) {
  int64_t rows = dstValid[0], cols = dstValid[1];
  int64_t requiredBytes = 0;
  if (rows > 0 && cols > 0 && srcElem.isF32()) {
    int64_t head = 4 * 64 * std::min<int64_t>(cols / 64, 255);
    int64_t remainder = cols % 64;
    int64_t tail = remainder == 0
                       ? 0
                       : 32 * ((std::min<int64_t>(rows, 255) - 1) *
                                   (srcShape[1] / 8) +
                               llvm::divideCeil(remainder, int64_t{8}));
    requiredBytes = std::max(head, tail);
  } else if (cols > 0 && srcElem.isF16()) {
    int64_t width = std::min<int64_t>(cols, 64);
    int64_t halfToI16 = 32 * llvm::divideCeil(width, int64_t{8});
    int64_t halfToI8 = std::max<int64_t>(
        halfToI16,
        128 + 32 * static_cast<int64_t>(
                       llvm::divideCeil(width, int64_t{16})));
    requiredBytes = dstElem.isInteger(mlir::pto::kValue8) ? halfToI8 : halfToI16;
  }
  return requiredBytes;
}

static LogicalResult verifyTCvtTmp(TCvtOp op, Type srcTy, Type dstTy,
                                   Type srcElem, Type dstElem) {
  if (!op.getTmp()) {
    return success();
  }
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!tcvtNeedsTmp(op, srcElem, dstElem)) {
    return success();
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) || llvm::is_contained(dstValid, ShapedType::kDynamic)) {
      return op.emitOpError("expects static src shape and dst valid_shape to verify tcvt tmp");
  }
  int64_t requiredBytes =
      computeTCvtTmpRequiredBytes(srcElem, dstElem, srcShape, dstValid);
  auto tmpBytes = getStaticByteSize(tmpTy);
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(requiredBytes)) {
    return op.emitOpError()
           << "expects tcvt tmp capacity to be at least " << requiredBytes
           << " bytes";
  }
  return success();
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true))) {
    return failure();
  }
  // A resolved subview keeps its parent's physical shape so the generated
  // Tile retains the parent stride. Its valid shape is the logical tcvt
  // extent, so comparing physical shapes would reject a valid sliced tile.
  if (!tcvtIsResolvedSubview(getSrc()) && !tcvtIsResolvedSubview(getDst()) &&
      failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    }
    return verifyTCvtTmp(*this, srcTy, dstTy, srcElem, dstElem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem)) {
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    }
    if (getTmp() && failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy)) {
      return emitOpError("expects dst to use row-major layout");
    }

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32)) {
      return emitOpError("expects dst element type to be i32 or ui32");
    }

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32) {
        return emitOpError() << "expects " << name << " to be i32/ui32";
      }
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3"))) {
      return failure();
    }

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10) {
      return emitOpError("expects rounds to be 7 or 10");
    }

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32())) {
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32))) {
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);
    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile)) {
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");
    }
