// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
