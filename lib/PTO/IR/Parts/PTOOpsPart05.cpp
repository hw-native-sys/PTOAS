// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto isSupportedGatherElemTypeA5Index = [&](Type ty) -> bool {
    if (isPTOFloat8Type(ty))
      return true;
    if (ty.isF16() || ty.isF32())
      return true;
    if (auto it = dyn_cast<IntegerType>(ty)) {
      unsigned width = it.getWidth();
      return width == 8 || width == 16 || width == 32;
    }
    return false;
  };

  auto verifyMaskForm = [&](bool allowA5MaskTypes) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src", allowA5MaskTypes)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", allowA5MaskTypes)))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem)
      return emitOpError("failed to get element type for src/dst");
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src and dst to use row-major layout");
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src and dst to be in the vec address space");
    unsigned srcElemBytes = getPTOStorageElemByteSize(srcElem);
    unsigned dstElemBytes = getPTOStorageElemByteSize(dstElem);
    if (srcElemBytes == 0 || dstElemBytes == 0)
      return emitOpError("failed to get element size for src/dst");
    if (srcElemBytes != dstElemBytes)
      return emitOpError("expects src and dst element sizes to match");

    auto dstValid = getValidShapeVec(dstTy);
    auto dstShape = getShapeVec(dstTy);
    if (dstValid.size() == 2 && dstShape.size() == 2 &&
        dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        dstValid[1] != dstShape[1]) {
      return emitOpError("expects dst valid_shape[1] to equal dst cols");
    }

    if (allowA5MaskTypes) {
      if (!(srcElemBytes == 1 || srcElemBytes == 2 || srcElemBytes == 4))
        return emitOpError("expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
      if (!isSupportedGatherElemTypeA5(srcElem) || !isSupportedGatherElemTypeA5(dstElem))
        return emitOpError(
            "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
    } else {
      if (!(srcElemBytes == 2 || srcElemBytes == 4))
        return emitOpError("expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
    }
    return success();
  };

  auto verifyIndexForm = [&](bool allow16BitIndices, bool allowA5ElemTypes) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Type idxTy = getIndices().getType();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src", allowA5ElemTypes)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", allowA5ElemTypes)) ||
        failed(verifyTileBufCommon(*this, idxTy, "indices")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem)
      return emitOpError("failed to get element type for src/dst");
    if (srcElem != dstElem)
      return emitOpError("expects src and dst to have the same element type");
    if (allowA5ElemTypes) {
      if (!isSupportedGatherElemTypeA5Index(srcElem) ||
          !isSupportedGatherElemTypeA5Index(dstElem))
        return emitOpError(
            "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    } else if (!isSupportedGatherElemTypeA2A3(srcElem) ||
               !isSupportedGatherElemTypeA2A3(dstElem)) {
      return emitOpError("expects gather src/dst element type to be i16/i32/f16/f32");
    }

    auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
    if (!idxElem)
      return emitOpError("indices element type must be integer");
    unsigned width = idxElem.getWidth();
    if (!(width == 32 || (allow16BitIndices && width == 16))) {
      return emitOpError() << "expects indices element type to be i32"
                           << (allow16BitIndices ? " or i16" : "");
    }

    auto dstValid = getValidShapeVec(dstTy);
    auto dstShape = getShapeVec(dstTy);
    if (dstValid.size() == 2 && dstShape.size() == 2 &&
        dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        dstValid[1] != dstShape[1]) {
      return emitOpError("expects dst valid_shape[1] to equal dst cols");
    }

    auto idxValid = getValidShapeVec(idxTy);
    auto idxShape = getShapeVec(idxTy);
    if (idxValid.size() == 2 && idxShape.size() == 2 &&
        idxValid[1] != ShapedType::kDynamic && idxShape[1] != ShapedType::kDynamic &&
        idxValid[1] != idxShape[1]) {
      return emitOpError("expects indices valid_shape[1] to equal indices cols");
    }

    if (!allowA5ElemTypes) {
      Type tmpElem = getElemTy(tmpTy);
      if (tmpElem != idxElem)
        return emitOpError("expects tmp and indices to have the same element type");
      if (failed(verifyTileBufSameValidShape(*this, idxTy, tmpTy, "indices", "tmp")))
        return failure();
    }
    return success();
  };

  auto verifyCompareForm = [&](bool allowA5SrcTypes) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Type cdstTy = getCdst().getType();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyTileBufCommon(*this, cdstTy, "cdst")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    Type cdstElem = getElemTy(cdstTy);
    if (!srcElem || !dstElem || !cdstElem)
      return emitOpError("failed to get element type for src/dst/cdst");
    auto dstInt = dyn_cast<IntegerType>(dstElem);
    if (!dstInt || dstInt.getWidth() != 32)
      return emitOpError("expects dst element type to be i32");
    if (cdstElem != dstElem)
      return emitOpError("expects cdst to have the same element type as dst");
    if (getKValue().getType() != srcElem)
      return emitOpError("expects kValue to have the same type as src element type");

    auto cmpAttr = getCmpModeAttr();
    auto cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
    if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT)
      return emitOpError("expects compare-form tgather cmpMode to be eq or gt");

    if (allowA5SrcTypes) {
      if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isInteger(16) ||
            srcElem.isInteger(32))) {
        return emitOpError(
            "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
      }
    } else {
      if (!(srcElem.isF16() || srcElem.isF32() ||
            (srcElem.isInteger(32) && cmpMode == pto::CmpMode::EQ))) {
        return emitOpError(
            "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
      }
    }

    if (failed(verifyVecTileCommonA2A3(*this, srcTy, "src")) ||
        failed(verifyVecTileCommonA2A3(*this, dstTy, "dst")) ||
        failed(verifyVecTileCommonA2A3(*this, cdstTy, "cdst")) ||
        failed(verifyVecTileCommonA2A3(*this, tmpTy, "tmp")))
      return failure();
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (getMaskPatternAttr()) {
      if (getCdst() || getIndices() || getTmp() || getKValue())
        return emitOpError("mask-pattern tgather only allows src and dst operands");
      return verifyMaskForm(/*allowA5MaskTypes=*/false);
    }
    if (getCdst() || getKValue()) {
      if (!getCdst() || !getKValue() || !getTmp())
        return emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
      if (getIndices())
        return emitOpError("compare-form tgather does not take indices");
      return verifyCompareForm(/*allowA5SrcTypes=*/false);
    }
    if (!getIndices() || !getTmp())
      return emitOpError("index-form tgather expects both indices and tmp");
    return verifyIndexForm(/*allow16BitIndices=*/false, /*allowA5ElemTypes=*/false);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    if (getMaskPatternAttr()) {
      if (getCdst() || getIndices() || getTmp() || getKValue())
        return emitOpError("mask-pattern tgather only allows src and dst operands");
      return verifyMaskForm(/*allowA5MaskTypes=*/true);
    }
    if (getCdst() || getKValue()) {
      if (!getCdst() || !getKValue() || !getTmp())
        return emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
      if (getIndices())
        return emitOpError("compare-form tgather does not take indices");
      return verifyCompareForm(/*allowA5SrcTypes=*/true);
    }
    if (!getIndices() || !getTmp())
      return emitOpError("index-form tgather expects both indices and tmp");
    return verifyIndexForm(/*allow16BitIndices=*/true, /*allowA5ElemTypes=*/true);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type srcTy = getSrc().getType();
    Type offTy = getOffsets().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, offTy, "offsets")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto dstElemTy = getElemTy(dstTy);
    if (!srcElemTy || !dstElemTy)
      return emitOpError() << "failed to get element type for src/dst";
    return std::make_pair(srcElemTy, dstElemTy);
  };

  auto getElemBytes = [](Type ty) -> std::optional<unsigned> {
    unsigned elemBytes = getPTOStorageElemByteSize(ty);
    if (elemBytes == 0)
      return std::nullopt;
    return elemBytes;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstTy = getDst().getType();
    Type dstElemTy = elems->second;
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError() << "expects dst to use row-major layout";
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstElemTy = elems->second;
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  return success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2)
      return emitOpError("expects src to have rank-2 valid_shape");
    if (valid[0] != ShapedType::kDynamic && valid[0] < 0)
      return emitOpError("expects src valid_shape[0] to be non-negative");
    if (valid[1] != ShapedType::kDynamic && valid[1] < 0)
      return emitOpError("expects src valid_shape[1] to be non-negative");
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    if (!getSlope().getType().isF32())
      return emitOpError() << "expects slope to have type f32";
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

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyImpl = [&](bool isA5) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Value fp = getFp();
    Value preQuantScalar = getPreQuantScalar();
    auto accToVecModeAttr = getAccToVecModeAttr();
    auto reluMode = getReluPreMode();
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);

    if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/isA5)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/isA5)))
      return failure();
    if (hasFp && failed(verifyTileBufCommon(*this, fp.getType(), "fp",
                                            /*allowLowPrecision=*/isA5)))
      return failure();
    if (hasFp && hasPreQuantScalar)
      return emitOpError() << "expects fp and preQuantScalar forms to be mutually exclusive";

    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace)
      return emitOpError() << "expects src and dst to have explicit address spaces";

    auto srcShape = getShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape)
      return emitOpError() << "expects mat-source tmov to use matching src/dst shapes";
    if (!isA5 && *srcSpace != pto::AddressSpace::MAT && srcShape != dstShape)
      return emitOpError() << "expects A2/A3 non-mat tmov to use matching src/dst shapes";

    const bool isMatToTile =
        *srcSpace == pto::AddressSpace::MAT &&
        (*dstSpace == pto::AddressSpace::LEFT ||
         *dstSpace == pto::AddressSpace::RIGHT ||
         *dstSpace == pto::AddressSpace::BIAS ||
         *dstSpace == pto::AddressSpace::SCALING);
    const bool isVecToVec =
        *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::VEC;
    const bool isVecToMat =
        *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::MAT;
    const bool isAccToMat =
        *srcSpace == pto::AddressSpace::ACC &&
        *dstSpace == pto::AddressSpace::MAT;
    const bool isAccToVec =
        *srcSpace == pto::AddressSpace::ACC &&
        *dstSpace == pto::AddressSpace::VEC;

    bool okPair = isMatToTile || isVecToVec || isAccToMat || isAccToVec;
    if (isA5)
      okPair = okPair || isVecToMat;
    if (!okPair)
      return emitOpError()
             << "expects a supported tmov address-space pair for this target";

    if (accToVecModeAttr && !isAccToVec)
      return emitOpError()
             << "expects accToVecMode to be used only for acc-to-vec tmov";

    if (reluMode != pto::ReluPreMode::NoRelu && !(isAccToMat || isAccToVec))
      return emitOpError()
             << "expects reluPreMode form to use loc=acc src";

    if (hasPreQuantScalar && !(isAccToMat || isAccToVec))
      return emitOpError()
             << "expects preQuantScalar form to use loc=acc src";

    if (hasFp) {
      auto fpTy = fp.getType();
      auto fpSpace = getPTOMemorySpaceEnum(fpTy);
      if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
        return emitOpError() << "expects fp to be in the scaling address space";
      auto srcElemTy = getElemTy(srcTy);
      auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
      if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32)))
        return emitOpError()
               << "expects fp form src to have element type f32, i32";
      if (!(isAccToMat || isAccToVec))
        return emitOpError() << "expects fp form to use loc=acc src";
    }

    if ((hasFp || hasPreQuantScalar) && accToVecModeAttr) {
      switch (accToVecModeAttr.getValue()) {
      case pto::AccToVecMode::SingleModeVec0:
      case pto::AccToVecMode::SingleModeVec1:
        break;
      case pto::AccToVecMode::DualModeSplitM:
      case pto::AccToVecMode::DualModeSplitN:
        return emitOpError()
               << "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode";
      }
    }

    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (srcTb && *srcSpace == pto::AddressSpace::ACC &&
        (hasFp || reluMode != pto::ReluPreMode::NoRelu)) {
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError()
               << "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major";
    }
    if (srcTb && dstTb && isAccToMat && !isA5 &&
        dstTb.getSFractalSizeI32() != 512)
      return emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";

    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyImpl(/*isA5=*/false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyImpl(/*isA5=*/true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMovFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != mlir::pto::AddressSpace::SCALING)
      return emitOpError() << "expects fp to be in the scaling address space";
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != mlir::pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || *dstSpace != mlir::pto::AddressSpace::MAT)
      return emitOpError() << "expects dst to be in the mat address space";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (srcTb &&
        (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects src to use blayout=col_major and slayout=row_major";
    if (dstTb &&
        (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects dst to use blayout=col_major and slayout=row_major";
    if (dstTb && dstTb.getSFractalSizeI32() != 512)
      return emitOpError() << "expects dst to use fractal size 512";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp", /*allowLowPrecision=*/true)) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true)))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != mlir::pto::AddressSpace::SCALING)
      return emitOpError() << "expects fp to be in the scaling address space";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    if (srcTb &&
        (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects src to use blayout=col_major and slayout=row_major";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<ShapedType>(t)) return s.getRank();
  if (auto tile = dyn_cast<pto::TileBufType>(t)) return tile.getRank();
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) return view.getRank();
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (ShapedType 或 Tile 类型)
  bool aValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid)
    return op->emitOpError("expects inputs/outputs to be shaped types or PTO tile types");

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank)
      return op->emitOpError("expects a and dst to have the same rank");
    if (bRank != -1 && dRank != -1 && bRank != dRank)
      return op->emitOpError("expects b and dst to have the same rank");
  }

  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar load only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects result type to match ptr element type");

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar store only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects value type to match ptr element type");

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr, IntegerAttr modeAttr) {
  if (!opTypeAttr)
    return op->emitOpError("expects 'op_type' attribute");

  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opTypeOr)) {
      auto diag = op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ");
      diag << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opTypeOr);
  }
  if (!isConcreteSyncPipe(pipe))
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");

  if (!bufIdAttr)
    return op->emitOpError("expects 'buf_id' attribute");
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31)
    return op->emitOpError("expects 'buf_id' in range [0, 31]");

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0)
      return op->emitOpError("expects 'mode' to be non-negative");
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

static ParseResult parseLegacyOrAttrMemBar(OpAsmParser &parser,
                                           MemBarAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeMemBarKind(token);
    if (!kind)
      return parser.emitError(loc) << "invalid membar token: " << token;
    attr = MemBarAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed)))
    return failure();
  auto memBarAttr = dyn_cast<MemBarAttr>(parsed);
  if (!memBarAttr)
    return parser.emitError(loc, "expected membar attribute");
  attr = memBarAttr;
  return success();
}

static void printLegacyOrAttrMemBar(OpAsmPrinter &p, MemBarAttr kind,
                                    ArrayRef<NamedAttribute> attrs) {
  p << ' ' << '"' << stringifyMemBarKind(kind.getKind()) << '"';
  p.printOptionalAttrDict(attrs, {"kind"});
}

static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto pipe = symbolizePIPE(token);
    if (!pipe)
      return parser.emitError(loc) << "invalid pipe token: " << token;
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater())
      return failure();
    auto pipe = symbolizePIPE(keyword);
    if (!pipe)
      return parser.emitError(loc) << "invalid pipe token: " << keyword;
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed)))
    return failure();
  auto pipeAttr = dyn_cast<PipeAttr>(parsed);
  if (!pipeAttr)
    return parser.emitError(loc, "expected pipe attribute");
  attr = pipeAttr;
  return success();
}

static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto event = symbolizeEVENT(token);
    if (!event)
      return parser.emitError(loc) << "invalid event token: " << token;
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater())
      return failure();
    auto event = symbolizeEVENT(keyword);
    if (!event)
      return parser.emitError(loc) << "invalid event token: " << keyword;
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed)))
    return failure();
  auto eventAttr = dyn_cast<EventAttr>(parsed);
  if (!eventAttr)
    return parser.emitError(loc, "expected event attribute");
  attr = eventAttr;
  return success();
}

static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  int64_t value = 0;
  if (failed(parser.parseInteger(value)))
    return failure();
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max())
    return parser.emitError(loc, "expected 32-bit integer literal");
  attr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), value);
  return success();
}

static void printLegacySyncTriplet(OpAsmPrinter &p, PipeAttr srcPipe,
                                   PipeAttr dstPipe, EventAttr eventId,
                                   ArrayRef<NamedAttribute> attrs) {
  p << "[<" << stringifyPIPE(srcPipe.getPipe()) << ">, <"
    << stringifyPIPE(dstPipe.getPipe()) << ">, <"
    << stringifyEVENT(eventId.getEvent()) << ">]";
  p.printOptionalAttrDict(attrs, {"src_pipe", "dst_pipe", "event_id"});
}

ParseResult SetFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void SetFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult WaitFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void WaitFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult MemBarOp::parse(OpAsmParser &parser, OperationState &result) {
  MemBarAttr kind;
  if (parseLegacyOrAttrMemBar(parser, kind))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("kind", kind);
  return success();
}

void MemBarOp::print(OpAsmPrinter &p) {
  printLegacyOrAttrMemBar(p, getKind(), (*this)->getAttrs());
}

static ParseResult parseBufSyncOp(OpAsmParser &parser, OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr bufIdAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token))
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    else if (auto opType = symbolizeSyncOpType(token))
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    else
      return parser.emitError(loc) << "invalid get_buf/rls_buf token: " << token;

    if (parser.parseComma() || parseI32LiteralAttr(parser, bufIdAttr))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr))
        return failure();
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma() ||
        parseI32LiteralAttr(parser, bufIdAttr))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr))
        return failure();
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
    if (parser.parseRSquare())
      return failure();
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("buf_id", bufIdAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                           IntegerAttr bufIdAttr, IntegerAttr modeAttr,
                           ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", "
      << bufIdAttr.getInt() << ", " << modeAttr.getInt();
  } else if (auto pipeEventType = dyn_cast<PipeEventTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else if (auto syncOpType = dyn_cast<SyncOpTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "buf_id", "mode"});
}

ParseResult GetBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void GetBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

ParseResult RlsBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void RlsBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}
// ---- TOp ----
LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false)) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2)
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1])
      return emitOpError("expects bias and dst to have the same column shape");
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true)) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                              /*requireFloatBias=*/true)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType())
      return emitOpError("expects val type to match dst element type");
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType, MemRefType>(srcTy))
    return emitOpError("expects src to be tile_buf or memref type");

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace =
      isa<pto::TileBufType>(srcTy)
          ? cast<pto::TileBufType>(srcTy).getMemorySpace()
          : cast<MemRefType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT)
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType())
    return emitOpError("expects dst type to match src element type");
  return success();
}

LogicalResult THistogramOp::verify() {
  auto isIntegerWidth = [](Type ty, unsigned width) {
    auto it = dyn_cast<IntegerType>(ty);
    return it && it.getWidth() == width;
  };
  int64_t byte = 1;
  auto byteAttr = getByteAttr();
  if (byteAttr)
    byte = byteAttr.getInt();
  if (auto legacyIsMSB = (*this)->getAttrOfType<BoolAttr>("isMSB")) {
    int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
    if (byteAttr && byte != legacyByte)
      return emitOpError("does not allow conflicting 'byte' and legacy 'isMSB' attributes");
    byte = legacyByte;
  }
  if (byte < 0 || byte > 3)
    return emitOpError("expects byte to be in range [0, 3]");

  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type idxTy = getIdx().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, idxTy, "idx")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto idxSpace = getPTOMemorySpaceEnum(idxTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src to be in the vec address space");
    if (!idxSpace || *idxSpace != pto::AddressSpace::VEC)
      return emitOpError("expects idx to be in the vec address space");
    if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects dst to be in the vec address space");

    auto srcTB = dyn_cast<pto::TileBufType>(srcTy);
    auto idxTB = dyn_cast<pto::TileBufType>(idxTy);
    auto dstTB = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTB || !idxTB || !dstTB)
      return emitOpError("expects src, idx, and dst to be tile_buf types");

    if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        srcTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return emitOpError("expects src to use row_major + none_box layout");
    if (dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        dstTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return emitOpError("expects dst to use row_major + none_box layout");

    bool srcIsUi16 = isIntegerWidth(getElemTy(srcTy), 16);
    bool srcIsUi32 = isIntegerWidth(getElemTy(srcTy), 32);
    if (!srcIsUi16 && !srcIsUi32)
      return emitOpError("expects src element type to be ui16 or ui32");
    if (!isIntegerWidth(getElemTy(idxTy), 8))
      return emitOpError("expects idx element type to be ui8");
    if (!isIntegerWidth(getElemTy(dstTy), 32))
      return emitOpError("expects dst element type to be ui32");

    auto srcShape = getShapeVec(srcTy);
    auto idxShape = getShapeVec(idxTy);
    auto dstShape = getShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    auto idxValid = getValidShapeVec(idxTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
        srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2)
      return emitOpError(
          "expects src, idx, and dst to have rank-2 shape and valid_shape");

    if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
        !hasCompatibleKnownExtent(srcValid[0], dstValid[0]))
      return emitOpError("expects dst rows and valid rows to match src");

    if (srcIsUi16) {
      if (byte > 1)
        return emitOpError("expects byte to be 0 or 1 when src element type is ui16");
      if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return emitOpError(
            "expects idx to use DN layout (col_major + none_box) when src element type is ui16");
      if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
          !hasCompatibleKnownExtent(srcValid[0], idxValid[0]))
        return emitOpError("expects idx rows and valid rows to match src when src element type is ui16");
      if (!isKnownUnitExtent(idxShape[1]) || !isKnownZeroOrUnitExtent(idxValid[1]))
        return emitOpError("expects idx to have exactly one physical column and 0 or 1 valid column when src element type is ui16");
    } else {
      if (byte != 3) {
        if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
          return emitOpError(
              "expects idx to use row_major + none_box layout when src element type is ui32 and byte is 0, 1, or 2");
        if (!hasCompatibleKnownExtent(srcShape[1], idxShape[1]) ||
            !hasCompatibleKnownExtent(srcValid[1], idxValid[1]))
          return emitOpError(
              "expects idx cols and valid cols to match src when src element type is ui32 and byte is 0, 1, or 2");

        int64_t expectedIdxRows = 1;
        if (byte == 1)
          expectedIdxRows = 2;
        else if (byte == 0)
          expectedIdxRows = 3;
        if (!hasCompatibleKnownExtent(idxShape[0], expectedIdxRows) ||
            !hasCompatibleKnownExtentOrZero(idxValid[0], expectedIdxRows))
          return emitOpError(
              "expects idx rows to match the byte-selected filter depth and idx valid rows to be 0 or match it when src element type is ui32 and byte is 0, 1, or 2");
      }
    }
    if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256)
      return emitOpError("expects dst shape[1] to be at least 256");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 0 &&
        dstValid[1] < 256)
      return emitOpError("expects dst valid_shape[1] to be 0 or at least 256");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)))
      return failure();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true)))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace ||
        (*srcSpace != pto::AddressSpace::LEFT && *srcSpace != pto::AddressSpace::RIGHT))
      return emitOpError("expects src to be in the left or right address space");
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || *dstSpace != pto::AddressSpace::SCALING)
      return emitOpError("expects dst to be in the scaling address space");
    auto dstShape = getShapeVec(dstTy);
    auto srcShape = getShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    if (dstShape.size() != 2 || srcShape.size() != 2 || dstValid.size() != 2 ||
        srcValid.size() != 2)
      return emitOpError(
          "expects src/dst to have rank-2 shape and valid_shape");
    if (*srcSpace == pto::AddressSpace::LEFT) {
      int64_t mShape = srcShape[0];
      int64_t vk = srcValid[1];
      int64_t expectedScaleK = ceilDivKnown(vk, 32);
      if (!hasCompatibleKnownExtent(dstShape[0], mShape) ||
          !hasCompatibleKnownExtent(dstShape[1], expectedScaleK) ||
          !hasCompatibleKnownExtent(dstValid[0], srcValid[0]) ||
          !hasCompatibleKnownExtent(dstValid[1], expectedScaleK))
        return emitOpError("expects dst shape/valid_shape to be [M, ceil(K/32)]");
    } else {
      int64_t k = srcValid[0];
      int64_t n = srcShape[1];
      int64_t vk = srcValid[0];
      int64_t vn = srcValid[1];
      if (!hasCompatibleKnownExtent(dstShape[0], ceilDivKnown(k, 32)) ||
          !hasCompatibleKnownExtent(dstShape[1], n) ||
          !hasCompatibleKnownExtent(dstValid[0], ceilDivKnown(vk, 32)) ||
          !hasCompatibleKnownExtent(dstValid[1], vn))
        return emitOpError("expects dst shape/valid_shape to be [ceil(K/32), N]");
    }
    return success();
  };
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
          {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"}))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands) ||
      parser.resolveOperand(mem, memTy, result.operands))
    return failure();
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
  if (auto coalesceAttr = getMScatterCoalesceAttrIfPresent(*this))
    attrs.append("coalesce", coalesceAttr);
  if (auto scatterAtomicAttr = getMScatterScatterAtomicOpAttrIfPresent(*this);
      scatterAtomicAttr &&
      scatterAtomicAttr.getValue() != pto::ScatterAtomicOp::None)
    attrs.append("scatterAtomicOp", scatterAtomicAttr);
  if (auto scatterOobAttr = getMScatterScatterOobAttrIfPresent(*this);
      scatterOobAttr &&
      scatterOobAttr.getValue() != pto::ScatterOOB::Undefined)
    attrs.append("scatterOob", scatterOobAttr);
  if (auto scatterConflictAttr =
          getMScatterScatterConflictAttrIfPresent(*this))
    attrs.append("scatterConflict", scatterConflictAttr);
  p.printOptionalAttrDict(attrs.getAttrs());
}

LogicalResult MScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1)
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(
          *this, srcTy, "src",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  auto coalesce = getCoalesceIfPresent(*this);

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  pto::ScatterAtomicOp scatterAtomicOp = getScatterAtomicOpOrDefault(*this);
  pto::ScatterOOB scatterOob = getScatterOobOrDefault(*this);
  if (!srcElem || !idxElem)
    return emitOpError("failed to resolve element types for src or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem))
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src")))
    return failure();

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src",
                                            coalesce)))
    return failure();

  if (!coalesce &&
      (scatterAtomicOp != pto::ScatterAtomicOp::None ||
       scatterOob != pto::ScatterOOB::Undefined ||
       getScatterConflictAttrIfPresent(*this)))
    return emitOpError(
        "expects coalesce when scatterAtomicOp/scatterOob/scatterConflict is specified");

  if (getScatterConflictAttrIfPresent(*this) && !isTargetArchA5(getOperation()))
    return emitOpError("expects scatterConflict only on A5 targets");

  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, scatterAtomicOp))
    return emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");

  return success();
}

// ---- MGatherOp ----
// GM -> L1 (cube Mat) gather verifier. The destination is an L1 (loc=mat) tile
// in NZ layout; the index is a GM tensor (the cube core cannot read UB on A5),
// and Coalesce::Elem carries a contiguous GM scratch workspace. Mirrors the
// pto-isa MGATHER GM -> L1 overloads / MGatherCheckGm2L1.
static LogicalResult verifyMGatherGm2L1(Operation *op, Value mem, Value idx,
                                        Value dst, Value scratch,
                                        std::optional<pto::Coalesce> coalesce) {
  Type dstTy = dst.getType();
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!dstTb)
    return op->emitOpError("expects GM->L1 mgather dst to be a tile_buf");

  // dst must be an L1 / cube Mat tile in NZ layout (col_major + row_major sub +
  // fractal 512), matching the matmul A/NZ operand a TLOAD would produce.
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

  // NZ tile shape: padded Cols a multiple of C0 (= 32 / sizeof(elem)) and padded
  // Rows a multiple of FRACTAL_NZ_ROW (= 16).
  unsigned elemBytes =
      std::max<unsigned>(1u, dstElem.getIntOrFloatBitWidth() / 8u);
  int64_t kC0 = 32 / static_cast<int64_t>(elemBytes);
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == 2) {
    if (kC0 > 0 && dstShape[1] != ShapedType::kDynamic &&
        dstShape[1] % kC0 != 0)
      return op->emitOpError()
             << "expects GM->L1 mgather dst padded cols to be a multiple of "
             << kC0 << " (C0 = 32 / sizeof(elem))";
    if (dstShape[0] != ShapedType::kDynamic && dstShape[0] % 16 != 0)
      return op->emitOpError("expects GM->L1 mgather dst padded rows to be a "
                             "multiple of 16 (FRACTAL_NZ_ROW)");
  }

  // mem table: GM, element type matches dst.
  if (failed(verifyMGatherMScatterMemOperand(op, mem, dstElem, "dst")))
    return failure();

  // idx: GM tensor (memref / partition_tensor_view) of i32 -- NOT a UB tile.
  Type idxTy = idx.getType();
  if (isa<pto::TileBufType>(idxTy))
    return op->emitOpError("expects GM->L1 mgather idx to be a GM tensor "
                           "(memref / partition_tensor_view), not a tile_buf");
  if (auto idxMr = dyn_cast<MemRefType>(idxTy)) {
    auto as = getPTOMemorySpaceEnum(idxMr);
    if (!as || (*as != pto::AddressSpace::GM && *as != pto::AddressSpace::Zero))
      return op->emitOpError(
          "expects GM->L1 mgather idx memref to use GM or zero address space");
  } else if (!isa<pto::PartitionTensorViewType>(idxTy)) {
    return op->emitOpError("expects GM->L1 mgather idx to be a GM memref or "
                           "partition_tensor_view");
  }
  Type idxElem = getElemTy(idxTy);
  if (!idxElem || !isSupportedMGatherMScatterIndexElemType(idxElem))
    return op->emitOpError("expects GM->L1 mgather idx element type to be i32");

  // Coalesce must be explicit: the GM index has no UB tile shape to infer from.
  if (!coalesce)
    return op->emitOpError("expects GM->L1 mgather to specify an explicit "
                           "coalesce attribute (row or elem)");

  if (*coalesce == pto::Coalesce::Elem) {
    // Elem mode stages discrete elements into NZ layout through a GM scratch
    // workspace before the bulk GM -> L1 copy.
    if (!scratch)
      return op->emitOpError("expects GM->L1 mgather with coalesce=elem to "
                             "provide a GM scratch operand");
    Type scTy = scratch.getType();
    if (auto scMr = dyn_cast<MemRefType>(scTy)) {
      auto as = getPTOMemorySpaceEnum(scMr);
      if (!as ||
          (*as != pto::AddressSpace::GM && *as != pto::AddressSpace::Zero))
        return op->emitOpError("expects GM->L1 mgather scratch memref to use "
                               "GM or zero address space");
    } else if (!isa<pto::PartitionTensorViewType>(scTy)) {
      return op->emitOpError("expects GM->L1 mgather scratch to be a GM memref "
                             "or partition_tensor_view");
    }
    Type scElem = getElemTy(scTy);
    if (!scElem || scElem != dstElem)
      return op->emitOpError("expects GM->L1 mgather scratch element type to "
                             "match dst element type");
  } else { // Row
    if (scratch)
      return op->emitOpError("expects GM->L1 mgather with coalesce=row to omit "
                             "the scratch operand");
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

  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();

  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();
    insOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));

  if (insOperands.size() < 2 || insOperands.size() > 3)
    return parser.emitError(parser.getCurrentLocation(),
                            "expects mgather ins(mem, idx[, scratch])");

  if (parser.parseColon())
    return failure();

  do {
    Type type;
    if (parser.parseType(type))
      return failure();
    insTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (insOperands.size() != insTypes.size())
    return parser.emitError(parser.getCurrentLocation(),
                            "expects the number of ins operands to match the number of ins types");

  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parsePTOInherentAttrs<MGatherOp>(
          parser, result, parsedAttrs, {"coalesce", "gatherOob"}))
    return failure();

  if (parser.resolveOperand(insOperands[0], insTypes[0], result.operands) ||
      parser.resolveOperand(insOperands[1], insTypes[1], result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (insOperands.size() == 3 &&
      parser.resolveOperand(insOperands[2], insTypes[2], result.operands))
    return failure();
  return success();
}

void mlir::pto::MGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getMem() << ", " << getIdx();
  if (auto scratch = getScratch())
    p << ", " << scratch;
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
  if (auto coalesceAttr = getMGatherCoalesceAttrIfPresent(*this))
    attrs.append("coalesce", coalesceAttr);
  if (auto gatherOobAttr = getMGatherGatherOobAttrIfPresent(*this);
      gatherOobAttr &&
      gatherOobAttr.getValue() != pto::GatherOOB::Undefined)
    attrs.append("gatherOob", gatherOobAttr);
  p.printOptionalAttrDict(attrs.getAttrs());
}

LogicalResult MGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();

  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1)
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");

  // GM -> L1 (cube Mat) gather: dst is an L1 (loc=mat) tile; idx comes from GM
  // and Coalesce::Elem carries a GM scratch operand.
  if (isa<pto::TileBufType>(dstTy)) {
    if (auto as = getPTOMemorySpaceEnum(dstTy);
        as && *as == pto::AddressSpace::MAT) {
      std::optional<pto::Coalesce> coalesce;
      if (auto coalesceAttr = getCoalesceAttr())
        coalesce = coalesceAttr.getValue();
      return verifyMGatherGm2L1(getOperation(), getMem(), getIdx(), getDst(),
                                getScratch(), coalesce);
    }
  }

  // GM -> UB (VEC) gather: the default path. A GM scratch operand is only valid
  // for the GM -> L1 path above.
  if (getScratch())
    return emitOpError("expects scratch operand only on GM->L1 (loc=mat) "
                       "mgather");

  if (failed(verifyNDStyleVecTile(
          *this, dstTy, "dst",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  auto coalesce = getCoalesceIfPresent(*this);

  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  pto::GatherOOB gatherOob = getGatherOobOrDefault(*this);
  if (!dstElem || !idxElem)
    return emitOpError("failed to resolve element types for dst or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), dstElem))
    return emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), dstElem,
                                             "dst")))
    return failure();

  if (failed(verifyMGatherMScatterTileShape(getOperation(), dstTy, idxTy, "dst",
                                            coalesce)))
    return failure();

  if (gatherOob != pto::GatherOOB::Undefined && !coalesce)
    return emitOpError("expects coalesce when gatherOob is specified");

  return success();
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs());
  p << " : " << getSrc().getType();
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(srcTy))
    return failure();
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode"))
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  return success();
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    p << ", " << getTmp();
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    p << ", " << getTmp().getType();
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}
