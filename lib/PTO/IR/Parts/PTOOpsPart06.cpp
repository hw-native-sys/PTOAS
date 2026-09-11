// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyTFillPadElementTypes(Operation *op, Type srcTy,
                                                Type dstTy) {
  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  int64_t srcB = getPTOStorageElemByteSize(srcElem);
  int64_t dstB = getPTOStorageElemByteSize(dstElem);
  if (srcB == 0 || dstB == 0) {
    return op->emitError("unsupported element type (expects int/float element types)");
  }
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");
  return success();
}

static LogicalResult verifyTFillPadShapeExpansion(Operation *op, Type srcTy,
                                                  Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  bool expanded = false;
  for (auto [srcDim, dstDim] : llvm::zip_equal(srcShape, dstShape)) {
    if (srcDim == dstDim)
      continue;
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim))
      return op->emitError("cannot infer TFILLPAD lowering from mismatched "
                           "dynamic physical shapes");
    if (srcDim > dstDim)
      return op->emitError(
          "expects each dst physical shape dimension to be >= src");
    expanded = true;
  }
  if (expanded &&
      (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
       *dstSpace != pto::AddressSpace::VEC))
    return op->emitError("expects expanded TFILLPAD only for loc=vec");
  return success();
}

static LogicalResult verifyTFillPadMatTypes(Operation *op, Type srcTy,
                                            Type dstTy) {
  auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
  auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (srcTb && dstTb && srcSpace && dstSpace &&
      *srcSpace == mlir::pto::AddressSpace::MAT &&
      *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb) {
    auto dimToStr = [](int64_t dim) -> std::string {
      return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
    };
    SmallVector<std::string, 4> mismatchFields;
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() == 2 && dstValid.size() == 2) {
      if (srcValid[0] != dstValid[0]) {
        mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                                 dimToStr(dstValid[0]) + ")");
      }
      if (srcValid[1] != dstValid[1]) {
        mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                                 dimToStr(dstValid[1]) + ")");
      }
    }
    if (srcTb.getPadValueI32() != dstTb.getPadValueI32()) {
      mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                               " vs " + std::to_string(dstTb.getPadValueI32()) +
                               ")");
    }

    auto diag = op->emitError()
                << "expects src/dst tile types to be lowerable to TFILLPAD "
                   "for loc=mat";
    if (!mismatchFields.empty()) {
      diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
    }
    diag << "\n  src: " << srcTy;
    diag << "\n  dst: " << dstTy;
    diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
    return failure();
  }
  return success();
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy,
                                              Type dstTy) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");
  if (getShapeVec(srcTy).size() != 2 || getShapeVec(dstTy).size() != 2)
    return op->emitError("expects rank-2 shaped types for src/dst");
  if (failed(verifyTFillPadElementTypes(op, srcTy, dstTy)) ||
      failed(verifyTFillPadShapeExpansion(op, srcTy, dstTy)) ||
      failed(verifyTFillPadMatTypes(op, srcTy, dstTy)))
    return failure();
  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError("expects dst PadVal != Null for tfillpad");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  if (getOperation()->getAttr("mode")) {
    return emitOpError("does not accept 'mode'; PTOAS infers TFILLPAD lowering "
                       "from physical shape and planned addresses");
  }

  if (failed(verifyTFillPadLike(getOperation(), getSrc().getType(),
                                getDst().getType()))) {
    return failure();
  }

  if (auto padValueAttr = getPadValueAttr()) {
    auto dstSpace = getPTOMemorySpaceEnum(getDst().getType());
    if (!dstSpace || *dstSpace != pto::AddressSpace::MAT) {
      return emitOpError("expects padValue attribute only for loc=mat tfillpad");
    }
    auto dstTileTy = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!dstTileTy) {
      return emitOpError("expects dst to be tile_buf when padValue is specified");
    }
    if (dstTileTy.getPadValueI32() != static_cast<int32_t>(padValueAttr.getValue())) {
      return emitOpError("expects padValue attribute to match dst tile pad configuration");
    }
  }

  return success();
}


static bool isSupportedGatherElemTypeA5Index(Type ty) {
  if (isPTOFloat8Type(ty)) {
    return true;
  }
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32;
  }
  return false;
}

static unsigned getMaskGatherTimes(mlir::pto::MaskPatternAttr mp) {
  switch (mp.getValue()) {
  case mlir::pto::MaskPattern::P1111:
    return 1;
  case mlir::pto::MaskPattern::P0101:
  case mlir::pto::MaskPattern::P1010:
    return 2;
  default:
    return 4;
  }
}

static LogicalResult verifyTGatherMaskShapes(TGatherOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto axisAttr = op.getAxisAttr();
  if (!axisAttr) {
    return op.emitOpError("expects mask-pattern tgather to provide axis attribute");
  }
  StringRef axisVal = axisAttr.getValue();
  auto mp = op.getMaskPatternAttr();
  if (!mp) {
    return op.emitOpError("expects mask-pattern tgather to provide maskPattern");
  }
  const unsigned times = getMaskGatherTimes(mp);
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (axisVal == "row") {
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != srcValid[0]) {
      return op.emitOpError("expects dst valid rows to equal src valid rows for row direction");
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != static_cast<int64_t>(dstValid[1] * times)) {
      return op.emitOpError("expects src valid cols to equal dst valid cols times the mask expansion factor for row direction");
    }
  } else if (axisVal == "col") {
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != srcValid[1]) {
      return op.emitOpError("expects dst valid cols to equal src valid cols for col direction");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != static_cast<int64_t>(dstValid[0] * times)) {
      return op.emitOpError("expects src valid rows to equal dst valid rows times the mask expansion factor for col direction");
    }
  } else {
    return op.emitOpError("Invalid axis value, expected \"row\" or \"col\"");
  }
  return success();
}

static LogicalResult verifyTGatherMaskArchTypes(TGatherOp op, Type srcElem,
                                                Type dstElem,
                                                unsigned elemBytes,
                                                bool allowA5MaskTypes) {
  if (!allowA5MaskTypes) {
    if (elemBytes == 2 || elemBytes == 4)
      return success();
    return op.emitOpError(
        "expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
  }
  if (!(elemBytes == 1 || elemBytes == 2 || elemBytes == 4))
    return op.emitOpError(
        "expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
  if (!isSupportedGatherElemTypeA5(srcElem) ||
      !isSupportedGatherElemTypeA5(dstElem))
    return op.emitOpError(
        "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
  return success();
}

static LogicalResult verifyTGatherMaskForm(TGatherOp op, bool allowA5MaskTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowA5MaskTypes)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowA5MaskTypes))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("failed to get element type for src/dst");
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError("expects src and dst to use row-major layout");
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src and dst to be in the vec address space");
  }
  unsigned srcElemBytes = getPTOStorageElemByteSize(srcElem);
  unsigned dstElemBytes = getPTOStorageElemByteSize(dstElem);
  if (srcElemBytes == 0 || dstElemBytes == 0) {
    return op.emitOpError("failed to get element size for src/dst");
  }
  if (srcElemBytes != dstElemBytes) {
    return op.emitOpError("expects src and dst element sizes to match");
  }
  auto dstValid = getValidShapeVec(dstTy);
  auto dstShape = getShapeVec(dstTy);
  if (dstValid.size() == 2 && dstShape.size() == 2 &&
      dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
      dstValid[1] != dstShape[1]) {
    return op.emitOpError("expects dst valid_shape[1] to equal dst cols");
  }
  if (failed(verifyTGatherMaskShapes(op)))
    return failure();
  return verifyTGatherMaskArchTypes(op, srcElem, dstElem, srcElemBytes,
                                    allowA5MaskTypes);
}

static LogicalResult verifyTGatherIndexTypes(TGatherOp op, bool allow16BitIndices,
                                             bool allowA5ElemTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type idxTy = op.getIndices().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowA5ElemTypes)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowA5ElemTypes)) ||
      failed(verifyTileBufCommon(op, idxTy, "indices"))) {
    return failure();
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("failed to get element type for src/dst");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (allowA5ElemTypes) {
    if (!isSupportedGatherElemTypeA5Index(srcElem) ||
        !isSupportedGatherElemTypeA5Index(dstElem)) {
      return op.emitOpError(
          "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    }
  } else if (!isSupportedGatherElemTypeA2A3(srcElem) ||
             !isSupportedGatherElemTypeA2A3(dstElem)) {
    return op.emitOpError("expects gather src/dst element type to be i16/i32/f16/f32");
  }
  auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
  if (!idxElem) {
    return op.emitOpError("indices element type must be integer");
  }
  unsigned width = idxElem.getWidth();
  if (!(width == 32 || (allow16BitIndices && width == 16))) {
    return op.emitOpError() << "expects indices element type to be i32"
                            << (allow16BitIndices ? " or i16" : "");
  }
  return success();
}

static LogicalResult verifyTGatherIndexShapes(TGatherOp op, bool allowA5ElemTypes) {
  Type dstTy = op.getDst().getType();
  Type idxTy = op.getIndices().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto dstShape = getShapeVec(dstTy);
  if (dstValid.size() == 2 && dstShape.size() == 2 &&
      dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
      dstValid[1] != dstShape[1]) {
    return op.emitOpError("expects dst valid_shape[1] to equal dst cols");
  }
  auto idxValid = getValidShapeVec(idxTy);
  auto idxShape = getShapeVec(idxTy);
  if (idxValid.size() == 2 && idxShape.size() == 2 &&
      idxValid[1] != ShapedType::kDynamic && idxShape[1] != ShapedType::kDynamic &&
      idxValid[1] != idxShape[1]) {
    return op.emitOpError("expects indices valid_shape[1] to equal indices cols");
  }
  if (!allowA5ElemTypes) {
    if (failed(verifyTileBufSameValidShape(op, dstTy, idxTy, "dst", "indices"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy) || !isRowMajorTileBuf(idxTy) ||
        !isRowMajorTileBuf(op.getTmp().getType())) {
      return op.emitOpError(
          "expects A2/A3 index-form dst, indices, and tmp to use row-major layout");
    }
    auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
    Type tmpElem = getElemTy(op.getTmp().getType());
    if (tmpElem != idxElem) {
      return op.emitOpError("expects tmp and indices to have the same element type");
    }
    if (failed(verifyTileBufSameValidShape(op, idxTy, op.getTmp().getType(), "indices", "tmp"))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTGatherIndexForm(TGatherOp op, bool allow16BitIndices,
                                            bool allowA5ElemTypes) {
  if (failed(verifyTGatherIndexTypes(op, allow16BitIndices, allowA5ElemTypes))) {
    return failure();
  }
  return verifyTGatherIndexShapes(op, allowA5ElemTypes);
}

static LogicalResult verifyTGatherCompareSrcType(TGatherOp op, Type srcElem,
                                                 pto::CmpMode cmpMode,
                                                 bool allowA5SrcTypes) {
  if (allowA5SrcTypes) {
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isInteger(16) ||
          srcElem.isInteger(32))) {
      return op.emitOpError(
          "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
    }
  } else {
    if (!(srcElem.isF16() || srcElem.isF32() ||
          (srcElem.isInteger(32) && cmpMode == pto::CmpMode::EQ))) {
      return op.emitOpError(
          "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
    }
  }
  return success();
}

static LogicalResult verifyTGatherCompareForm(TGatherOp op, bool allowA5SrcTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type cdstTy = op.getCdst().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufCommon(op, cdstTy, "cdst")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type cdstElem = getElemTy(cdstTy);
  if (!srcElem || !dstElem || !cdstElem) {
    return op.emitOpError("failed to get element type for src/dst/cdst");
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || dstInt.getWidth() != 32) {
    return op.emitOpError("expects dst element type to be i32");
  }
  if (cdstElem != dstElem) {
    return op.emitOpError("expects cdst to have the same element type as dst");
  }
  if (op.getKValue().getType() != srcElem) {
    return op.emitOpError("expects kValue to have the same type as src element type");
  }
  auto cmpAttr = op.getCmpModeAttr();
  auto cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
  if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT) {
    return op.emitOpError("expects compare-form tgather cmpMode to be eq or gt");
  }
  if (failed(verifyTGatherCompareSrcType(op, srcElem, cmpMode, allowA5SrcTypes))) {
    return failure();
  }
  if (failed(verifyVecTileCommonA2A3(op, srcTy, "src")) ||
      failed(verifyVecTileCommonA2A3(op, dstTy, "dst")) ||
      failed(verifyVecTileCommonA2A3(op, cdstTy, "cdst")) ||
      failed(verifyVecTileCommonA2A3(op, tmpTy, "tmp"))) {
    return failure();
  }
  return success();
}

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

static bool isVecTile(Type type) {
  auto space = getPTOMemorySpaceEnum(type);
  return isa<pto::TileBufType>(type) && space &&
         *space == pto::AddressSpace::VEC;
}

static LogicalResult verifyTMovXToZzForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (!isA5) {
    return op.emitOpError("X-to-ZZ tmov is only supported on A5");
  }
  if (op.getNumResults() != 0) {
    return op.emitOpError("expects X-to-ZZ tmov not to have results");
  }
  if (op.getPreQuantScalar() || op.getAccToVecModeAttr() ||
      op.getReluPreMode() != pto::ReluPreMode::NoRelu) {
    return op.emitOpError("expects the X-to-ZZ tmov form not to use preQuantScalar, accToVecMode, or reluPreMode");
  }

  if (!isVecTile(srcTy) || !isVecTile(dstTy) || !isVecTile(fp.getType())) {
    return op.emitOpError("expects X-to-ZZ src/dst/tmp to be vec tiles");
  }
  if (op.getSrc() == op.getDst() || op.getSrc() == fp || op.getDst() == fp) {
    return op.emitOpError("expects X-to-ZZ src, dst, and tmp to be distinct tile values");
  }
  if (cast<pto::TileBufType>(srcTy).getRank() != 2 ||
      cast<pto::TileBufType>(dstTy).getRank() != 2 ||
      cast<pto::TileBufType>(fp.getType()).getRank() != 2) {
    return op.emitOpError("expects rank-2 valid_shape for src/dst/tmp");
  }
  if (!isStaticTMovShape(srcTy, true) || !isStaticTMovShape(dstTy, true) ||
      !isStaticTMovShape(fp.getType(), false)) {
    return op.emitOpError("expects static valid and physical shapes for src/dst and a static tmp physical shape for X-to-ZZ");
  }
  return verifyTMovXToZzElemLayout(op);
}

static LogicalResult verifyTMovXToZzAxis1(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> dstValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (dstValid[1] % 2 != 0) {
    return op.emitOpError("expects ND-to-ZZ dst valid_shape[1] (the grouped exponent column count) to be even");
  }
  if (srcValid[0] != 1 && srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects ND-to-ZZ src valid elements to form a compact prefix (single-row legacy flat or physical row stride equal to valid cols)");
  }
  auto paddedRows = tmovAlign16(dstValid[0]);
  auto required =
      paddedRows ? tmovCheckedMul(*paddedRows, dstValid[1]) : std::nullopt;
  if (!required) {
    return op.emitOpError("cannot compute ND-to-ZZ padded capacity without overflow");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  if (!srcBytes || *srcBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ src physical capacity to cover align16(dst rows) * dst cols because source padding is zeroed in place");
  }
  if (!dstBytes || *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ dst physical capacity to cover align16(dst rows) * dst cols");
  }
  auto rowBlocksBias = tmovCheckedAdd(dstValid[0], 15);
  auto offsetBytes = rowBlocksBias
                         ? tmovCheckedMul(*rowBlocksBias / 16, dstValid[1])
                         : std::nullopt;
  auto tmpRequired =
      offsetBytes ? tmovCheckedAdd(64, *offsetBytes) : std::nullopt;
  if (!tmpRequired) {
    return op.emitOpError("cannot compute ND-to-ZZ tmp capacity without overflow");
  }
  auto tmpBytes = getStaticByteSize(fp.getType());
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(*tmpRequired)) {
    return op.emitOpError() << "expects tmp to provide at least " << *tmpRequired
                            << " bytes for ND-to-ZZ (64 + ceil(dst rows / 16) * dst cols)";
  }
  return success();
}

static LogicalResult verifyTMovXToZzAxis0(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (srcValid[0] < 2 || srcValid[0] % 2 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[0] to be an even count >= 2; a single row-group produces no output in PTO-ISA");
  }
  if (srcValid[1] % 16 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[1] to be a multiple of 16");
  }
  if (srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects DN-to-ZZ src physical row stride to equal src valid_shape[1]");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  auto required = tmovCheckedMul(srcValid[0], srcValid[1]);
  if (!required || !srcBytes || !dstBytes ||
      *srcBytes < static_cast<uint64_t>(*required) ||
      *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects DN-to-ZZ src/dst physical capacity to cover src valid rows * src valid cols");
  }
  return success();
}

static LogicalResult verifyTMovXToZzCapacity(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  auto srcPhysical = getShapeVec(srcTy);
  auto srcElements = tmovCheckedElements(srcValid);
  auto dstElements = tmovCheckedElements(dstValid);
  if (!srcElements || !dstElements || *srcElements != *dstElements) {
    return op.emitOpError("expects src and dst to hold the same exponent count");
  }
  const MxGroupAxis axis =
      op.getGrpAxisAttr() ? op.getGrpAxisAttr().getValue() : MxGroupAxis::Axis1;
  if (axis == MxGroupAxis::Axis1) {
    return verifyTMovXToZzAxis1(op, srcValid, dstValid, srcPhysical);
  }
  return verifyTMovXToZzAxis0(op, srcValid, srcPhysical);
}

static LogicalResult verifyTMovXToZz(TMovOp op, bool isA5) {
  if (failed(verifyTMovXToZzForm(op, isA5))) {
    return failure();
  }
  return verifyTMovXToZzCapacity(op);
}

static LogicalResult verifyTMovGenericPreconditions(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  if (op.getGrpAxisAttr()) {
    return op.emitOpError("expects grpAxis only on the X-to-ZZ form with a non-scaling third tile");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", /*allowLowPrecision=*/isA5)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && failed(verifyTileBufCommon(op, fp.getType(), "fp",
                                          /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && op.getPreQuantScalar()) {
    return op.emitOpError() << "expects fp and preQuantScalar forms to be mutually exclusive";
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace) {
    return op.emitOpError() << "expects src and dst to have explicit address spaces";
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects mat-source tmov to use matching src/dst shapes";
  }
  if (!isA5 && *srcSpace != pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects A2/A3 non-mat tmov to use matching src/dst shapes";
  }
  return success();
}

static LogicalResult verifyTMovGenericPairing(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  using AddressPair = std::pair<pto::AddressSpace, pto::AddressSpace>;
  SmallVector<AddressPair> supported = {
      {pto::AddressSpace::MAT, pto::AddressSpace::LEFT},
      {pto::AddressSpace::MAT, pto::AddressSpace::RIGHT},
      {pto::AddressSpace::MAT, pto::AddressSpace::BIAS},
      {pto::AddressSpace::MAT, pto::AddressSpace::SCALING},
      {pto::AddressSpace::VEC, pto::AddressSpace::VEC},
      {pto::AddressSpace::ACC, pto::AddressSpace::MAT},
      {pto::AddressSpace::ACC, pto::AddressSpace::VEC}};
  if (isA5)
    supported.push_back({pto::AddressSpace::VEC, pto::AddressSpace::MAT});
  bool okPair = llvm::is_contained(supported, AddressPair(*srcSpace, *dstSpace));
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  if (!okPair) {
    return op.emitOpError() << "expects a supported tmov address-space pair for this target";
  }
  if (op.getAccToVecModeAttr() && !isAccToVec) {
    return op.emitOpError() << "expects accToVecMode to be used only for acc-to-vec tmov";
  }
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects reluPreMode form to use loc=acc src";
  }
  if (op.getPreQuantScalar() && !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects preQuantScalar form to use loc=acc src";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpLayout(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  const bool hasFp = static_cast<bool>(op.getFp());
  auto reluMode = op.getReluPreMode();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (srcTb && *srcSpace == pto::AddressSpace::ACC &&
      (hasFp || reluMode != pto::ReluPreMode::NoRelu)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError() << "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major";
    }
  }
  if (hasFp && !isA5 && dstTb && isAccToMat &&
      (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
       dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))) {
    return op.emitOpError() << "expects fp tmov dst to use blayout=col_major and slayout=row_major";
  }
  if (srcTb && dstTb && isAccToMat && !isA5 &&
      dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  auto accToVecModeAttr = op.getAccToVecModeAttr();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  if (hasFp) {
    auto fpSpace = getPTOMemorySpaceEnum(fp.getType());
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError() << "expects fp to be in the scaling address space";
    }
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32))) {
      return op.emitOpError() << "expects fp form src to have element type f32, i32";
    }
    if (!(isAccToMat || isAccToVec)) {
      return op.emitOpError() << "expects fp form to use loc=acc src";
    }
  }
  if ((hasFp || hasPreQuantScalar) && accToVecModeAttr) {
    switch (accToVecModeAttr.getValue()) {
    case pto::AccToVecMode::SingleModeVec0:
    case pto::AccToVecMode::SingleModeVec1:
      break;
    case pto::AccToVecMode::DualModeSplitM:
    case pto::AccToVecMode::DualModeSplitN:
      return op.emitOpError() << "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode";
    }
  }
  return verifyTMovGenericFpLayout(op, isA5);
}

static LogicalResult verifyTMovGeneric(TMovOp op, bool isA5) {
  if (failed(verifyTMovGenericPreconditions(op, isA5)) ||
      failed(verifyTMovGenericPairing(op, isA5)) ||
      failed(verifyTMovGenericFpForm(op, isA5))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTMovImpl(TMovOp op, bool isA5) {
  Value fp = op.getFp();
  if (fp && !getPTOMemorySpaceEnum(fp.getType())) {
    return op.emitOpError("expects the third tile to have an explicit address space");
  }
  if (classifyTMovForm(fp) == TMovForm::XToZz) {
    return verifyTMovXToZz(op, isA5);
  }
  return verifyTMovGeneric(op, isA5);
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<RankedTensorType>(t)) {
    return s.getRank();
  }
  if (auto tile = dyn_cast<pto::TileBufType>(t)) {
    return tile.getRank();
  }
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) {
    return view.getRank();
  }
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (Tensor 或 Tile 类型)
  bool aValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid) {
    return op->emitOpError("expects inputs/outputs to be tensors or PTO tile types");
  }

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank) {
      return op->emitOpError("expects a and dst to have the same rank");
    }
    if (bRank != -1 && dRank != -1 && bRank != dRank) {
      return op->emitOpError("expects b and dst to have the same rank");
    }
  }

  return success();
}

static LogicalResult verifyScalarPointerAccess(Operation *op, Value ptr,
                                               Type valueType,
                                               StringRef valueName) {
  auto ptrType = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrType)
    return op->emitOpError("expects ptr to be !pto.ptr type");
  if (valueType != ptrType.getElementType())
    return op->emitOpError()
           << "expects " << valueName << " type to match ptr element type";
  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  return verifyScalarPointerAccess(getOperation(), getPtr(),
                                   getValue().getType(), "result");
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  return verifyScalarPointerAccess(getOperation(), getPtr(),
                                   getValue().getType(), "value");
}

// ---- CmoCacheInvalidOp ----
static bool isGmOrDefaultAddressSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}

static bool isGmOrDefaultCmoAddressType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return isGmOrDefaultAddressSpace(ptrTy.getMemorySpace().getAddressSpace());
  }
  if (isa<mlir::pto::TensorViewType, mlir::pto::PartitionTensorViewType>(type)) {
    return true;
  }
  return false;
}

ParseResult CmoCacheInvalidOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  if (succeeded(parser.parseOptionalKeyword("all"))) {
    AddressSpaceAttr spaceAttr;
    if (parser.parseAttribute(spaceAttr, "space", result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    return success();
  }

  OpAsmParser::UnresolvedOperand addr;
  Type addrTy;
  if (parser.parseOperand(addr) ||
      parser.parseKeyword("single_cache_line") ||
      parser.parseColonType(addrTy) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(addr, addrTy, result.operands)) {
    return failure();
  }

  if (!result.attributes.get("space")) {
    result.addAttribute(
        "space", AddressSpaceAttr::get(parser.getContext(), AddressSpace::GM));
  }
  return success();
}

void CmoCacheInvalidOp::print(OpAsmPrinter &p) {
  if (Value addr = getAddr()) {
    p << " " << addr << " single_cache_line";
    p << " : " << addr.getType();
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"space"});
    return;
  }

  p << " all " << getSpace();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"space"});
}

LogicalResult CmoCacheInvalidOp::verify() {
  if (!isGmOrDefaultAddressSpace(getSpace().getAddressSpace())) {
    return emitOpError("only supports GM cache maintenance");
  }

  if (Value addr = getAddr()) {
    if (!isGmOrDefaultCmoAddressType(addr.getType())) {
      return emitOpError("single_cache_line address expects a GM pointer or GM tensor view");
    }
  }

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static FailureOr<pto::PIPE> getConcreteSyncPipe(Operation *op,
                                                Attribute opTypeAttr) {
  if (!opTypeAttr) {
    op->emitOpError("expects 'op_type' attribute");
    return failure();
  }
  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opType = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opType)) {
      op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ")
          << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opType);
  }
  if (!isConcreteSyncPipe(pipe)) {
    op->emitOpError(
        "expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");
    return failure();
  }
  return pipe;
}

static LogicalResult verifyOptionalSyncMode(Operation *op,
                                            IntegerAttr modeAttr) {
  if (modeAttr && modeAttr.getInt() < 0)
    return op->emitOpError("expects 'mode' to be non-negative");
  return success();
}

static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr,
                                     IntegerAttr modeAttr) {
  if (failed(getConcreteSyncPipe(op, opTypeAttr)))
    return failure();

  if (!bufIdAttr) {
    return op->emitOpError("expects 'buf_id' attribute");
  }
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31) {
    return op->emitOpError("expects 'buf_id' in range [0, 31]");
  }

  return verifyOptionalSyncMode(op, modeAttr);
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

// ---- GetBufDynOp / RlsBufDynOp ----
static LogicalResult verifyBufDynSyncOp(Operation *op, Attribute opTypeAttr,
                                        Value bufId, IntegerAttr modeAttr) {
  if (failed(getConcreteSyncPipe(op, opTypeAttr)))
    return failure();
  if (!bufId) {
    return op->emitOpError("expects 'buf_id' operand");
  }
  return verifyOptionalSyncMode(op, modeAttr);
}

LogicalResult GetBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

LogicalResult RlsBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

static ParseResult parseLegacyOrAttrMemBar(OpAsmParser &parser,
                                           MemBarAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeMemBarKind(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid membar token: " << token;
    }
    attr = MemBarAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto memBarAttr = dyn_cast<MemBarAttr>(parsed);
  if (!memBarAttr) {
    return parser.emitError(loc, "expected membar attribute");
  }
  attr = memBarAttr;
  return success();
}

static void printLegacyOrAttrMemBar(OpAsmPrinter &p, MemBarAttr kind,
                                    ArrayRef<NamedAttribute> attrs) {
  p << ' ' << '"' << stringifyMemBarKind(kind.getKind()) << '"';
  p.printOptionalAttrDict(attrs, {"kind"});
}

static ParseResult parseLegacyOrAttrDsbMem(OpAsmParser &parser,
                                           DsbMemAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDsbMem(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dsb memory token: " << token;
    }
    attr = DsbMemAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dsbMemAttr = dyn_cast<DsbMemAttr>(parsed);
  if (!dsbMemAttr) {
    return parser.emitError(loc, "expected dsb_mem attribute");
  }
  attr = dsbMemAttr;
  return success();
}

static void printLegacyOrAttrDsbMem(OpAsmPrinter &printer, Operation *op,
                                    DsbMemAttr mem) {
  (void)op;
  printer << ' ' << '"' << stringifyDsbMem(mem.getKind()) << '"';
}

static ParseResult parseLegacyOrAttrDcciCacheLine(OpAsmParser &parser,
                                                  DcciCacheLineAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciCacheLine(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci cache token: " << token;
    }
    attr = DcciCacheLineAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto cacheAttr = dyn_cast<DcciCacheLineAttr>(parsed);
  if (!cacheAttr) {
    return parser.emitError(loc, "expected dcci_cache_line attribute");
  }
  attr = cacheAttr;
  return success();
}

static void printLegacyOrAttrDcciCacheLine(OpAsmPrinter &printer, Operation *op,
                                           DcciCacheLineAttr cache) {
  (void)op;
  printer << ' ' << '"' << stringifyDcciCacheLine(cache.getKind()) << '"';
}

static ParseResult parseOptionalDcciDst(OpAsmParser &parser,
                                        DcciDstAttr &attr) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciDst(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci dst token: " << token;
    }
    attr = DcciDstAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dstAttr = dyn_cast<DcciDstAttr>(parsed);
  if (!dstAttr) {
    return parser.emitError(loc, "expected dcci_dst attribute");
  }
  attr = dstAttr;
  return success();
}

static void printOptionalDcciDst(OpAsmPrinter &printer, Operation *op,
                                 DcciDstAttr dst) {
  (void)op;
  if (!dst) {
    return;
  }
  printer << ", \"" << stringifyDcciDst(dst.getKind()) << '"';
}

LogicalResult DcciOp::verify() {
  auto space = getPTOMemorySpaceEnum(getPtr().getType());
  if (!space) {
    return emitOpError("expects ptr to have a PTO memory space");
  }
  if (*space != pto::AddressSpace::GM && *space != pto::AddressSpace::VEC) {
    return emitOpError("expects ptr memory space to be gm or ub/vec");
  }

  return success();
}

static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto pipe = symbolizePIPE(token);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << token;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto pipe = symbolizePIPE(keyword);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << keyword;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto pipeAttr = dyn_cast<PipeAttr>(parsed);
  if (!pipeAttr) {
    return parser.emitError(loc, "expected pipe attribute");
  }
  attr = pipeAttr;
  return success();
}

static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto event = symbolizeEVENT(token);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << token;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto event = symbolizeEVENT(keyword);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << keyword;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto eventAttr = dyn_cast<EventAttr>(parsed);
  if (!eventAttr) {
    return parser.emitError(loc, "expected event attribute");
  }
  attr = eventAttr;
  return success();
}

static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  int64_t value = 0;
  if (failed(parser.parseInteger(value))) {
    return failure();
  }
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    return parser.emitError(loc, "expected 32-bit integer literal");
  }
  attr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), value);
  return success();
}

static ParseResult parseOptionalSyncMode(OpAsmParser &parser,
                                         IntegerAttr &modeAttr) {
  if (succeeded(parser.parseOptionalComma()))
    return parseI32LiteralAttr(parser, modeAttr);
  modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
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

static ParseResult parseLegacySyncTriplet(OpAsmParser &parser,
                                          OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

ParseResult SetFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLegacySyncTriplet(parser, result);
}

void SetFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult WaitFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLegacySyncTriplet(parser, result);
}

void WaitFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult MemBarOp::parse(OpAsmParser &parser, OperationState &result) {
  MemBarAttr kind;
  if (parseLegacyOrAttrMemBar(parser, kind)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
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
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc) << "invalid get_buf/rls_buf token: " << token;
}

    if (parser.parseComma() || parseI32LiteralAttr(parser, bufIdAttr)) {
      return failure();
    }
    if (failed(parseOptionalSyncMode(parser, modeAttr)))
      return failure();
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma() ||
        parseI32LiteralAttr(parser, bufIdAttr)) {
      return failure();
    }
    if (failed(parseOptionalSyncMode(parser, modeAttr)))
      return failure();
    if (parser.parseRSquare()) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
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
