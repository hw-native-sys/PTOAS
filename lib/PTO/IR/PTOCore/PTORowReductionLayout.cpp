// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyRowReductionDstLayout(Operation* op, Type ty, StringRef name)
{
    if (failed(verifyTileBufInVec(op, ty, name)))
        return failure();
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
            return op->emitOpError() << "expects " << name << " to use the none_box slayout";
        }
    }
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
            return op->emitOpError() << "expects " << name << " to use the row_major or col_major blayout";
        }
    }
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        auto layout = getTileBufLogicalLayout(tb);
        if (layout && *layout == pto::Layout::DN) {
            auto shape = getShapeVec(ty);
            if (shape.size() == 2 && shape[1] != ShapedType::kDynamic && shape[1] != 1) {
                return op->emitOpError() << "expects DN-style " << name << " to have shape[1] == 1";
            }
            return success();
        }
        if (layout && *layout == pto::Layout::ND) {
            return success();
        }
        if (layout) {
            return op->emitOpError() << "expects " << name
                                     << " to use a DN-style column vector tile or legacy ND-style tile";
        }
    }
    // The dst valid_shape[1] == 1 constraint for row reductions is enforced in
    // verifyRowReductionValidRegion (it must be conditional on the no-op-marker
    // path), so it is intentionally not duplicated here. A previous unreachable
    // copy of that check lived after this return and has been removed.
    return success();
}

static LogicalResult verifyRowReductionValidRegion(Operation* op, Type srcTy, Type dstTy, bool allowEmptyMarker)
{
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
        return op->emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    // A fully-empty dst valid region (0x0) is PyPTO's dual-AIV no-op replay
    // marker: the op writes no elements, so accept it and skip the non-empty
    // structural constraints. Only plain reductions opt in (allowEmptyMarker);
    // arg reductions (trowargmax/trowargmin) still produce a real per-row index,
    // so they stay strict. One-sided empties (only one dim 0) still fall through
    // and are rejected below. Hardware Rv=0 no-op is tracked in pto-isa#143;
    // PTOAS only guarantees the IR is legal here.
    if (allowEmptyMarker && dstValid[0] == 0 && dstValid[1] == 0) {
        return success();
    }
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
        return op->emitOpError("expects src valid_shape[0] to be non-zero");
    }
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
        return op->emitOpError("expects src valid_shape[1] to be non-zero");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic && srcValid[0] != dstValid[0]) {
        return op->emitOpError("expects src and dst to have the same valid_shape[0]");
    }
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1) {
        return op->emitOpError("expects dst valid_shape[1] to be 1");
    }
    return success();
}

static bool isSupportedRowReductionElemType(Type elem)
{
    return elem.isInteger(16) || elem.isInteger(32) || elem.isF16() || elem.isF32();
}

static LogicalResult verifyTRowReductionTail(Operation *op, Type srcTy,
                                             Type dstTy,
                                             StringRef elemTypeError) {
  if (failed(verifyRowReductionValidRegion(
          op, srcTy, dstTy, /*allowEmptyMarker=*/true)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

[[maybe_unused]] static LogicalResult verifyTRowReductionNoTmpCommon(
    Operation* op, Type srcTy, Type dstTy, StringRef elemTypeError)
{
    if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
        failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
        return op->emitOpError("expects src and dst to have the same element type");
    }
    return verifyTRowReductionTail(op, srcTy, dstTy, elemTypeError);
}

static LogicalResult verifyTRowReductionWithTmpCommon(
    Operation* op, Type srcTy, Type tmpTy, Type dstTy, StringRef elemTypeError)
{
    if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) || failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
        failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
        return op->emitOpError("expects src and dst to have the same element type");
    }
    if (getTargetArch(op) != PTOArch::A5 && getElemTy(srcTy) != getElemTy(tmpTy)) {
        return op->emitOpError("expects A2/A3 tmp to have the same element type as src and dst");
    }
    if (failed(verifyTRowReductionTail(op, srcTy, dstTy, elemTypeError)))
        return failure();
    if (getTargetArch(op) != PTOArch::A5 && failed(verifyTmpCapacityAtLeast(op, tmpTy, 32))) {
        return failure();
    }
    return success();
}

static std::optional<int64_t> getVectorRepeatElements(Type elemTy)
{
    unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
    if (elemBits == 0 || 2048 % elemBits != 0) {
        return std::nullopt;
    }
    return static_cast<int64_t>(2048 / elemBits);
}

static std::optional<int64_t> getVectorBlockElements(Type elemTy)
{
    unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
    if (elemBits == 0 || 256 % elemBits != 0) {
        return std::nullopt;
    }
    return static_cast<int64_t>(256 / elemBits);
}

static int64_t ceilDivInt64(int64_t numerator, int64_t denominator)
{
    if (denominator == 0 || denominator < 0 || numerator < 0) {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}
static std::optional<int64_t> getArgReductionTmpMinStride(Type elemTy, int64_t srcValidCols)
{
    if (srcValidCols == ShapedType::kDynamic || srcValidCols < 0) {
        return std::nullopt;
    }
    auto repeatElems = getVectorRepeatElements(elemTy);
    auto blockElems = getVectorBlockElements(elemTy);
    if (!repeatElems || !blockElems) {
        return std::nullopt;
    }
    int64_t repeats = ceilDivInt64(srcValidCols, *repeatElems);
    return (ceilDivInt64(repeats * 2, *blockElems) + ceilDivInt64(repeats, *blockElems)) * *blockElems;
}

static bool hasExactKnownValidShape(Type lhsTy, Type rhsTy)
{
    return getValidShapeVec(lhsTy) == getValidShapeVec(rhsTy);
}

static LogicalResult verifyArgTmpMinStride(Operation *op, Type srcTy,
                                           Type tmpTy,
                                           ArrayRef<int64_t> srcValid,
                                           ArrayRef<int64_t> tmpValid) {
  if (srcValid[1] == ShapedType::kDynamic)
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
  if (!minStride)
    return op->emitOpError(
        "failed to infer A2/A3 tmp stride from src element type");
  if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride)
    return op->emitOpError()
           << "expects A2/A3 tmp valid_shape[1] to be at least " << *minStride
           << " for src valid_shape[1] = " << srcValid[1];
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTColArgTmpA2A3(Operation* op, Type srcTy, Type tmpTy)
{
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
        failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
        return failure();
    }

    if (hasExactKnownValidShape(srcTy, tmpTy)) {
        return verifyTmpCapacityAtLeast(op, tmpTy, 32);
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto tmpValid = getValidShapeVec(tmpTy);
    if (srcValid.size() != 2 || tmpValid.size() != 2) {
        return op->emitOpError("expects src and tmp to have rank-2 valid_shape");
    }
    if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
        return op->emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
    }
    return verifyArgTmpMinStride(op, srcTy, tmpTy, srcValid, tmpValid);
}

static LogicalResult verifyColArgReductionTypes(Operation *op, Type srcTy,
                                                Type dstTy,
                                                StringRef srcTypeError) {
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits =
      srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32)))
    return op->emitOpError(srcTypeError);
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyColArgReductionTail(Operation *op, Type srcTy,
                                               Type dstTy,
                                               StringRef srcTypeError) {
  if (failed(verifyColReductionValidRegion(
          op, srcTy, dstTy, /*requireNonZeroSrc=*/true)))
    return failure();
  return verifyColArgReductionTypes(op, srcTy, dstTy, srcTypeError);
}

static LogicalResult verifyTColArgReductionOpA2A3(Operation* op, Type srcTy, Type tmpTy, Type dstTy)
{
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) || failed(verifyTColArgTmpA2A3(op, srcTy, tmpTy)) ||
        failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    return verifyColArgReductionTail(
        op, srcTy, dstTy,
        "expects src/tmp element type to be 1, 2, or 4 bytes wide");
}

static LogicalResult verifyTColArgReductionNoTmp(Operation* op, Type srcTy, Type dstTy)
{
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
        failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    return verifyColArgReductionTail(
        op, srcTy, dstTy,
        "expects src element type to be 1, 2, or 4 bytes wide");
}

static LogicalResult verifyTColArgReductionOpA5(Operation* op, Type srcTy, Type tmpTy, Type dstTy)
{
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) || failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
        failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    return verifyColArgReductionTail(
        op, srcTy, dstTy,
        "expects src element type to be 1, 2, or 4 bytes wide");
}
