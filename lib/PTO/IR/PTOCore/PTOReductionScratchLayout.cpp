// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTColSumTmpStride(Operation* op, Type srcTy, Type tmpTy, bool isBinary)
{
    if (!isBinary) {
        return success();
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto tmpShape = getShapeVec(tmpTy);
    if (srcValid.size() != 2 || tmpShape.size() != 2) {
        return op->emitOpError("expects src and tmp to be rank-2 tiles");
    }

    int64_t srcValidCols = srcValid[1];
    int64_t tmpStride = tmpShape[1];
    if (srcValidCols != ShapedType::kDynamic && tmpStride != ShapedType::kDynamic && tmpStride < srcValidCols) {
        return op->emitOpError() << "expects tmp shape[1] to be at least src valid_shape[1] when "
                                    "isBinary is true; got "
                                 << tmpStride << " vs " << srcValidCols;
    }
    return success();
}

static LogicalResult verifyTRowArgTmpSmallDn(Operation* op, Type tmpTy, ArrayRef<int64_t> srcValid,
                                             ArrayRef<int64_t> tmpShape, ArrayRef<int64_t> tmpValid)
{
    if (tmpShape[1] != ShapedType::kDynamic && tmpShape[1] != 1) {
        return op->emitOpError("expects A2/A3 tmp DN layout to have shape[1] == 1");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] != 1) {
        return op->emitOpError("expects A2/A3 tmp DN layout to have valid_shape[1] == 1");
    }
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
        tmpValid[0] < srcValid[0] * 2) {
        return op->emitOpError() << "expects A2/A3 tmp DN layout to have valid_shape[0] >= "
                                 << (srcValid[0] * 2);
    }
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTRowArgTmpSmallNd(Operation* op, Type tmpTy, ArrayRef<int64_t> srcValid,
                                             ArrayRef<int64_t> tmpValid)
{
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
        return failure();
    }
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < srcValid[0]) {
        return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < 2) {
        return op->emitOpError("expects A2/A3 tmp valid_shape[1] to be at least 2 in the small-col ND path");
    }
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTRowArgTmpSmall(Operation* op, Type tmpTy, ArrayRef<int64_t> srcValid,
                                           ArrayRef<int64_t> tmpShape, ArrayRef<int64_t> tmpValid)
{
    auto tmpTile = dyn_cast<pto::TileBufType>(tmpTy);
    auto layout = tmpTile ? getTileBufLogicalLayout(tmpTile) : std::nullopt;
    if (layout && *layout == pto::Layout::DN) {
        return verifyTRowArgTmpSmallDn(op, tmpTy, srcValid, tmpShape, tmpValid);
    }
    if (!layout || *layout != pto::Layout::ND) {
        return op->emitOpError(
            "expects A2/A3 tmp to use DN 1-col or ND 2-col layout when src valid_shape[1] fits in one repeat");
    }
    return verifyTRowArgTmpSmallNd(op, tmpTy, srcValid, tmpValid);
}

static LogicalResult verifyTRowArgTmpWide(Operation* op, Type srcTy, Type tmpTy, ArrayRef<int64_t> srcShape,
                                          ArrayRef<int64_t> tmpShape, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> tmpValid)
{
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
        return failure();
    }
    if (srcShape[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic && tmpShape[0] != srcShape[0]) {
        return op->emitOpError("expects A2/A3 tmp shape[0] to match src shape[0]");
    }
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < srcValid[0]) {
        return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
    }
    return verifyArgTmpMinStride(op, srcTy, tmpTy, srcValid, tmpValid);
}

static LogicalResult verifyTRowArgTmpA2A3(Operation* op, Type srcTy, Type tmpTy)
{
    if (failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
        failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
        return failure();
    }

    if (hasExactKnownValidShape(srcTy, tmpTy)) {
        return verifyTmpCapacityAtLeast(op, tmpTy, 32);
    }

    auto srcShape = getShapeVec(srcTy);
    auto tmpShape = getShapeVec(tmpTy);
    auto srcValid = getValidShapeVec(srcTy);
    auto tmpValid = getValidShapeVec(tmpTy);
    if (srcShape.size() != 2 || tmpShape.size() != 2 || srcValid.size() != 2 || tmpValid.size() != 2) {
        return op->emitOpError("expects src and tmp to be rank-2 tiles");
    }

    auto repeatElems = getVectorRepeatElements(getElemTy(srcTy));
    if (!repeatElems) {
        return op->emitOpError("failed to infer A2/A3 tmp contract from src element type");
    }

    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] <= *repeatElems) {
        return verifyTRowArgTmpSmall(op, tmpTy, srcValid, tmpShape, tmpValid);
    }
    return verifyTRowArgTmpWide(op, srcTy, tmpTy, srcShape, tmpShape, srcValid, tmpValid);
}

static LogicalResult verifyRowArgReductionTypes(Operation *op, Type srcTy,
                                                Type dstTy) {
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(
        "expects src element type to be i16/i32/f16/f32");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyRowArgReductionTail(Operation *op, Type srcTy,
                                               Type dstTy) {
  if (failed(verifyRowReductionValidRegion(
          op, srcTy, dstTy, /*allowEmptyMarker=*/false)))
    return failure();
  return verifyRowArgReductionTypes(op, srcTy, dstTy);
}

static LogicalResult verifyTRowArgReductionOpA2A3(Operation* op, Type srcTy, Type tmpTy, Type dstTy)
{
    if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) || failed(verifyTRowArgTmpA2A3(op, srcTy, tmpTy)) ||
        failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    return verifyRowArgReductionTail(op, srcTy, dstTy);
}

static LogicalResult verifyTRowArgReductionNoTmp(Operation* op, Type srcTy, Type dstTy)
{
    if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
        failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    return verifyRowArgReductionTail(op, srcTy, dstTy);
}

static LogicalResult verifyTRowArgReductionOpA5(Operation* op, Type srcTy, Type tmpTy, Type dstTy)
{
    if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) || failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
        failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
        return failure();
    }
    return verifyRowArgReductionTail(op, srcTy, dstTy);
}

static LogicalResult verifyNDStyleVecTile(Operation* op, Type ty, StringRef name, bool allowLowPrecision)
{
    if (failed(verifyTileBufInVec(op, ty, name, allowLowPrecision))) {
        return failure();
    }
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
            return op->emitOpError() << "expects " << name << " to use the row_major blayout";
        }
        if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
            return op->emitOpError() << "expects " << name << " to use the none_box slayout";
        }
    }
    return success();
}

static LogicalResult verifyColReductionValidRegion(Operation* op, Type srcTy, Type dstTy, bool requireNonZeroSrc)
{
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
        return op->emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
    // writes no elements; accept and skip the non-empty constraints. One-sided
    // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
    // Col arg reductions (tcolargmax/tcolargmin) never reach this point with a
    // 0x0 dst: verifyColArgReductionDstLayout enforces dst valid_shape[0] == 1
    // first, so they stay strict without needing a flag here (unlike the row
    // path, whose dst-layout check does not constrain valid).
    if (dstValid[0] == 0 && dstValid[1] == 0) {
        return success();
    }
    if (requireNonZeroSrc) {
        if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
            return op->emitOpError("expects src valid_shape[0] to be non-zero");
        }
        if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
            return op->emitOpError("expects src valid_shape[1] to be non-zero");
        }
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic && srcValid[1] != dstValid[1]) {
        return op->emitOpError("expects src and dst to have the same valid_shape[1]");
    }
    return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation* op, Type ty, StringRef name)
{
    if (failed(verifyNDStyleVecTile(op, ty, name))) {
        return failure();
    }
    auto valid = getValidShapeVec(ty);
    if (valid.size() != 2) {
        return op->emitOpError() << "expects " << name << " to have rank-2 valid_shape";
    }
    if (valid[0] != ShapedType::kDynamic && valid[0] != 1) {
        return op->emitOpError() << "expects " << name << " valid_shape[0] to be 1";
    }
    return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value)
{
    if (!value) {
        return std::nullopt;
    }
    if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue())) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

LogicalResult mlir::pto::SectionSimtOp::verify()
{
    func::FuncOp func = getOperation()->getParentOfType<func::FuncOp>();
    if (!func) {
        return emitOpError("must be nested under a func.func");
    }

    if (getDimXAttr().getInt() < 0 || getDimYAttr().getInt() < 0 || getDimZAttr().getInt() < 0) {
        return emitOpError("requires non-negative i32 launch dimensions");
    }

    if (func->hasAttr(pto::kPTOSimtEntryAttrName)) {
        return emitOpError("must not appear inside a function marked with '") << pto::kPTOSimtEntryAttrName << "'";
    }

    WalkResult nested = getBody().walk([&](SectionSimtOp nestedOp) {
        nestedOp.emitOpError("nested pto.section.simt is not allowed");
        return WalkResult::interrupt();
    });
    if (nested.wasInterrupted()) {
        return failure();
    }

    return success();
}

static LogicalResult verifyYieldedValueTypes(Operation *op, ValueRange yielded,
                                             ValueRange outputs) {
  for (auto [idx, pair] : llvm::enumerate(llvm::zip(yielded, outputs))) {
    Value value = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (value.getType() != output.getType())
      return op->emitOpError()
             << "expects yielded value #" << idx << " to have type "
             << output.getType() << ", got " << value.getType();
  }
  return success();
}
