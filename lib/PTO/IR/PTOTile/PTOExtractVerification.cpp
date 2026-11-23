// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTExtractA2A3Acc(TExtractOp op, const TExtractCommon& c)
{
    if (*c.dstSpace != pto::AddressSpace::MAT) {
        return op.emitOpError("expects A2/A3 acc-source textract dst to use loc=mat");
    }
    const bool hasInvalidSrcLayout = c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
                                     c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor);
    if (hasInvalidSrcLayout) {
        return op.emitOpError("expects A2/A3 acc-source textract src to use blayout=col_major and slayout=row_major");
    }
    const bool hasInvalidDstLayout = c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
                                     c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor);
    if (hasInvalidDstLayout) {
        return op.emitOpError("expects A2/A3 acc-source textract dst to use blayout=col_major and slayout=row_major");
    }
    if (c.dstTb.getSFractalSizeI32() != mlir::pto::kValue512) {
        return op.emitOpError("expects A2/A3 acc-source textract dst fractal size to be 512");
    }
    const bool hasFp = static_cast<bool>(op.getFp());
    const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
    if (hasFp || hasPreQuantScalar) {
        if (!isA2A3AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
            return op.emitOpError(
                "expects A2/A3 acc preQuantScalar textract element types to be "
                "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
        }
    } else if (!isA2A3AccCastExtractTypePair(c.srcElem, c.dstElem)) {
        return op.emitOpError("expects A2/A3 acc textract element types to be src=f32, dst=f16/bf16");
    }
    return success();
}

static LogicalResult verifyTExtractA2A3Mat(TExtractOp op, const TExtractCommon& c)
{
    if (*c.srcSpace != pto::AddressSpace::MAT) {
        return op.emitOpError("expects A2/A3 textract src to use loc=mat, loc=acc, or loc=vec");
    }
    if (*c.dstSpace != pto::AddressSpace::LEFT && *c.dstSpace != pto::AddressSpace::RIGHT) {
        return op.emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, loc=mat, or loc=vec");
    }
    if (!hasMatExtractSourceLayoutA2A3(c.srcTb)) {
        return op.emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
    }
    if (*c.dstSpace == pto::AddressSpace::LEFT) {
        const bool hasInvalidLeftLayout =
            c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor);
        if (hasInvalidLeftLayout) {
            return op.emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
        }
    } else {
        const bool hasInvalidRightLayout =
            c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor);
        if (hasInvalidRightLayout) {
            return op.emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
        }
    }
    return success();
}

static bool isConvTileExtractElem(Type ty)
{
    if (ty.isF16() || ty.isBF16() || ty.isF32()) {
        return true;
    }
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
        unsigned width = intTy.getWidth();
        return width == mlir::pto::kValue8 || width == mlir::pto::kValue16 || width == mlir::pto::kValue32;
    }
    return false;
}

static LogicalResult verifyConvTileTExtractLayouts(TExtractOp op)
{
    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    auto srcCT = dyn_cast<pto::ConvTileBufType>(srcTy);
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::MAT) {
        return op.emitOpError("expects convtile textract src to use loc=mat");
    }
    if (!dstSpace || *dstSpace != pto::AddressSpace::RIGHT) {
        return op.emitOpError("expects convtile textract dst to use loc=right");
    }
    auto srcLayout = dyn_cast_or_null<pto::ConvLayoutAttr>(srcCT.getLayout());
    if (!srcLayout ||
        (srcLayout.getValue() != pto::ConvLayout::FRACTAL_Z && srcLayout.getValue() != pto::ConvLayout::FRACTAL_Z_3D)) {
        return op.emitOpError("expects convtile textract src to use layout=FRACTAL_Z or FRACTAL_Z_3D");
    }
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    const bool hasInvalidDstLayout = !dstTb ||
                                     dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
                                     dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor);
    if (hasInvalidDstLayout) {
        return op.emitOpError("expects convtile textract dst to use blayout=row_major and slayout=col_major");
    }
    return success();
}

static LogicalResult verifyConvTileTExtractElems(TExtractOp op)
{
    Type srcElem = getElemTy(op.getSrc().getType());
    Type dstElem = getElemTy(op.getDst().getType());
    if (!srcElem || !dstElem || srcElem != dstElem) {
        return op.emitOpError("expects convtile textract src and dst to have the same element type");
    }
    // A2/A3's TEXTRACT_CONVTILE_IMPL only accepts int8/half/bfloat16/float;
    // the wider i16/i32 forms are A5-only (a5/TExtractCommon.hpp).
    const bool isA5 = getVerifierTargetArch(op.getOperation()) == VerifierTargetArch::A5;
    const bool hasUnsupportedElem =
        !isConvTileExtractElem(srcElem) || (!isA5 && (srcElem.isInteger(mlir::pto::kValue16) || srcElem.isInteger(32)));
    if (hasUnsupportedElem) {
        return op.emitOpError("expects convtile textract element type to be i8/i16/i32/f16/bf16/f32");
    }
    return success();
}

static LogicalResult verifyConvTileTExtractShape(TExtractOp op)
{
    auto srcCT = dyn_cast<pto::ConvTileBufType>(op.getSrc().getType());
    auto srcLayout = dyn_cast_or_null<pto::ConvLayoutAttr>(srcCT.getLayout());
    const bool isFractalZ = srcLayout && srcLayout.getValue() == pto::ConvLayout::FRACTAL_Z;
    const bool isFractalZ3D = srcLayout && srcLayout.getValue() == pto::ConvLayout::FRACTAL_Z_3D;
    if (!isFractalZ && !isFractalZ3D) {
        return success();
    }
    // Mirrors the runtime TEXTRACT_CONVTILE_IMPL: a rank-4 FRACTAL_Z or
    // FRACTAL_Z_3D conv tile is [C1HW, N/16, 16, C0] and must end in the
    // static [16, c0] minor dims, a rank-5 FRACTAL_Z one is [C1, H, W, N, C0]
    // and only the trailing c0 is fixed, where c0 = 32 / sizeof(element); see
    // convC0ForElemBytes for why the c0 lookup is not a literal division.
    // Rank is checked before indexing.
    ArrayRef<int64_t> shape = srcCT.getShape();
    if (shape.size() != mlir::pto::kValue4 && !(isFractalZ && shape.size() == mlir::pto::kValue5)) {
        return op.emitOpError() << "expects convtile textract " << stringifyConvLayout(srcLayout.getValue())
                                << " src rank to be 4" << (isFractalZ ? " or 5" : "") << ", got " << shape.size();
    }
    const int64_t elemBytes = getPTOStorageElemByteSize(srcCT.getElementType());
    const std::optional<int64_t> c0 = mlir::pto::convC0ForElemBytes(elemBytes);
    if (!c0) {
        return op.emitOpError() << "expects convtile textract src element type to "
                                   "have a known storage size";
    }
    if (shape.size() == mlir::pto::kValue4) {
        if (shape[mlir::pto::kValue2] != mlir::pto::kValue16 || shape[mlir::pto::kValue3] != *c0) {
            return op.emitOpError() << "expects convtile textract " << stringifyConvLayout(srcLayout.getValue())
                                    << " src trailing dims to be [16, " << *c0 << "]";
        }
    } else if (shape.back() != *c0) {
        return op.emitOpError() << "expects convtile textract rank-5 FRACTAL_Z src trailing dim to "
                                   "be "
                                << *c0;
    }
    return success();
}

static LogicalResult verifyConvTileTExtractCoverage(TExtractOp op);
static LogicalResult verifyConvTileTExtract(TExtractOp op, bool allowLowPrecision)
{
    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    const bool hasFp = static_cast<bool>(op.getFp());
    const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
    const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    const bool hasInvalidCommonForm =
        failed(verifyTileLikeCommon(op, srcTy, "src", allowLowPrecision)) ||
        failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
        failed(verifyNonNegativeIndexRowCol(*op.getOperation(), op.getIndexRow(), op.getIndexCol(), hasFp)) ||
        failed(verifyConvTileTExtractShape(op)) || failed(verifyConvTileTExtractCoverage(op));
    if (hasInvalidCommonForm) {
        return failure();
    }
    if (hasFp || hasPreQuantScalar || hasRelu || op.getAccToVecModeAttr()) {
        return op.emitOpError("expects convtile textract to use the base form");
    }
    if (failed(verifyConvTileTExtractLayouts(op)) || failed(verifyConvTileTExtractElems(op))) {
        return failure();
    }
    return success();
}

// Verify the textract window stays within the source capacity and that
// constant indices are aligned. The capacity formulas were validated on
// real hardware (probe P-a: indexRow=32, K=64 → bit-exact golden match).
// Misalignment does not corrupt data — the runtime truncates sub-fractal
// offsets via >>4 / >>5 (probe P-c: indexCol=8 behaved identically to 0) —
// but silently dropping the offset is a semantic trap worth catching.
// K/N totals of a FRACTAL_Z / FRACTAL_Z_3D conv tile, derived from the
// runtime's address math (probe-validated). Returns false when the shape
// is not one of the supported fractal loadings.
static bool convTileExtractCapacity(
    pto::ConvTileBufType srcCT, pto::ConvLayout layoutVal, int64_t& kTotal, int64_t& nTotal)
{
    ArrayRef<int64_t> shape = srcCT.getShape();
    if (shape.size() == mlir::pto::kValue4) {
        nTotal = shape[1] * shape[2];
        kTotal = shape[0] * shape[3];
        return true;
    }
    if (shape.size() == mlir::pto::kValue5 && layoutVal == pto::ConvLayout::FRACTAL_Z) {
        nTotal = shape[3];
        kTotal = shape[0] * shape[1] * shape[2] * shape[4];
        return true;
    }
    return false;
}

static LogicalResult verifyConvTileExtractColumn(TExtractOp op, int64_t dstCols, int64_t nTotal)
{
    auto idxCol = mlir::getConstantIntValue(op.getIndexCol());
    if (!idxCol) {
        return success();
    }
    if (*idxCol % mlir::pto::kValue16 != 0) {
        return op.emitOpError() << "textract indexCol " << *idxCol
                                << " is not aligned to 16; the runtime silently truncates the sub-fractal offset";
    }
    if (*idxCol + dstCols > nTotal) {
        return op.emitOpError() << "textract indexCol " << *idxCol << " + dstCols " << dstCols << " exceeds N capacity "
                                << nTotal;
    }
    return success();
}

static LogicalResult verifyConvTileExtractRow(TExtractOp op, int64_t dstRows, int64_t kTotal, int64_t c0)
{
    auto idxRow = mlir::getConstantIntValue(op.getIndexRow());
    if (!idxRow) {
        return success();
    }
    if (c0 == 0) {
        return op.emitOpError() << "textract has invalid zero c0 alignment";
    }
    if (*idxRow % c0 != 0) {
        return op.emitOpError() << "textract indexRow " << *idxRow << " is not aligned to c0=" << c0
                                << "; the runtime silently truncates the offset";
    }
    if (*idxRow + dstRows > kTotal) {
        return op.emitOpError() << "textract indexRow " << *idxRow << " + dstRows " << dstRows << " exceeds K capacity "
                                << kTotal;
    }
    return success();
}

static LogicalResult verifyConvTileTExtractCoverage(TExtractOp op)
{
    auto srcCT = dyn_cast<pto::ConvTileBufType>(op.getSrc().getType());
    auto srcLayout = dyn_cast_or_null<pto::ConvLayoutAttr>(srcCT.getLayout());
    if (!srcLayout) {
        return success();
    }
    auto layoutVal = srcLayout.getValue();
    if (layoutVal != pto::ConvLayout::FRACTAL_Z && layoutVal != pto::ConvLayout::FRACTAL_Z_3D) {
        return success();
    }
    auto dstTB = dyn_cast<pto::TileBufType>(op.getDst().getType());
    if (!dstTB) {
        return success();
    }
    auto valid = dstTB.getValidShape();
    if (valid.size() != mlir::pto::kValue2 || valid[0] < 0 || valid[1] < 0) {
        return success();
    }
    const int64_t dstRows = valid[0];
    const int64_t dstCols = valid[1];

    const int64_t elemBytes = getPTOStorageElemByteSize(srcCT.getElementType());
    const auto c0 = mlir::pto::convC0ForElemBytes(elemBytes);
    if (!c0) {
        return success();
    }

    int64_t kTotal = 0;
    int64_t nTotal = 0;
    if (!convTileExtractCapacity(srcCT, layoutVal, kTotal, nTotal)) {
        return success();
    }

    if (failed(verifyConvTileExtractColumn(op, dstCols, nTotal))) {
        return failure();
    }
    return verifyConvTileExtractRow(op, dstRows, kTotal, *c0);
}

static LogicalResult verifyTExtractA2A3(TExtractOp op)
{
    auto common = verifyTExtractCommon(op, false);
    if (failed(common)) {
        return failure();
    }
    const TExtractCommon& c = *common;
    const bool hasFp = static_cast<bool>(op.getFp());
    const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
    const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (!isA2A3ExtractElemType(c.dstElem) && !(hasFp && c.dstElem.isInteger(mlir::pto::kValue16))) {
        return op.emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
    }
    if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
        return failure();
    }
    if (op.getAccToVecModeAttr()) {
        return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
    }
    if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::VEC) {
        if (hasPreQuantScalar || hasRelu) {
            return op.emitOpError(
                "expects vec->vec textract to use the base form without preQuantScalar or reluPreMode");
        }
        return success();
    }
    if (!c.srcSpace || !c.dstSpace) {
        return op.emitOpError("expects src and dst to have explicit loc");
    }
    if (*c.srcSpace == pto::AddressSpace::ACC) {
        return verifyTExtractA2A3Acc(op, c);
    }
    return verifyTExtractA2A3Mat(op, c);
}

static bool isTExtractA5SupportedPair(pto::AddressSpace srcSpace, pto::AddressSpace dstSpace)
{
    return (srcSpace == pto::AddressSpace::MAT &&
            (dstSpace == pto::AddressSpace::LEFT || dstSpace == pto::AddressSpace::RIGHT ||
             dstSpace == pto::AddressSpace::SCALING)) ||
           (srcSpace == pto::AddressSpace::VEC &&
            (dstSpace == pto::AddressSpace::MAT || dstSpace == pto::AddressSpace::VEC)) ||
           (srcSpace == pto::AddressSpace::ACC &&
            (dstSpace == pto::AddressSpace::MAT || dstSpace == pto::AddressSpace::VEC));
}

// The MX transfer encodes the outer axis in blocks of 16 and the K axis
// in pairs of E8M0 elements. Its step fields are eight bits.
template <int64_t Alignment>
static LogicalResult verifyMxExtractAxis(TExtractOp op, Value index, StringRef name, int64_t source,
                                         int64_t capacity, int64_t extent)
{
    static_assert(Alignment > 0);
    constexpr int64_t maxSteps = std::numeric_limits<uint8_t>::max();
    constexpr int64_t maxIndex = std::numeric_limits<uint16_t>::max();
    const auto offset = mlir::getConstantIntValue(index);
    if (offset && (*offset < 0 || *offset > maxIndex || *offset % Alignment != 0)) {
        return op.emitOpError() << "expects E8M0 scale textract " << name
                                << " to fit in uint16_t and be aligned to " << Alignment;
    }
    if (extent <= 0 || extent > maxSteps * Alignment || extent > capacity) {
        return op.emitOpError("expects E8M0 scale destination physical capacity and step to cover the copy footprint");
    }
    if (extent > source || (offset && *offset > source - extent)) {
        return op.emitOpError("expects E8M0 scale source physical capacity to cover the copy footprint");
    }
    return success();
}

static int64_t getMxExtractExtent(int64_t physical, int64_t valid, bool useValid)
{
    return useValid && !ShapedType::isDynamic(valid) ? valid : physical;
}

static LogicalResult verifyMxExtractSingleRow(TExtractOp op, const TExtractCommon& c)
{
    const auto src = c.srcTb.getShape();
    const auto dst = c.dstTb.getShape();
    const auto cols = getMxExtractExtent(dst[1], c.dstTb.getValidShape()[1], true);
    constexpr int64_t maxSingleRow = std::numeric_limits<uint8_t>::max() / pto::kValue2 * pto::kValue32;
    if (cols <= 0 || cols > maxSingleRow) {
        return op.emitOpError("expects a positive single-row E8M0 scale K extent fitting in the copy step");
    }
    const auto row = mlir::getConstantIntValue(op.getIndexRow());
    const auto col = mlir::getConstantIntValue(op.getIndexCol());
    if (src[0] != 1 || row != 0 || col != 0) {
        return op.emitOpError("expects single-row E8M0 ScaleLeft textract to use a single-row source "
                              "and constant zero offsets");
    }
    // The single-row intrinsic transfers whole 32-element blocks.
    const int64_t roundedCols = ((cols - 1) / pto::kValue32 + 1) * pto::kValue32;
    return verifyMxExtractAxis<pto::kValue32>(op, op.getIndexCol(), "indexCol", src[1], dst[1], roundedCols);
}

static LogicalResult verifyMxExtractMatrix(TExtractOp op, const TExtractCommon& c, bool left)
{
    const unsigned kAxis = left ? 1 : 0;
    const unsigned outerAxis = left ? 0 : 1;
    const auto src = c.srcTb.getShape();
    const auto dst = c.dstTb.getShape();
    const auto valid = c.dstTb.getValidShape();
    const bool compact = c.dstTb.getCompactModeI32() == static_cast<int32_t>(pto::CompactMode::Normal);
    const int64_t k = getMxExtractExtent(dst[kAxis], valid[kAxis], compact);
    if (k % pto::kValue2 != 0) {
        return op.emitOpError("expects E8M0 scale textract K extents to be aligned to 2");
    }
    if (src[outerAxis] % pto::kValue16 != 0 || dst[outerAxis] % pto::kValue16 != 0 ||
        src[kAxis] / pto::kValue2 > std::numeric_limits<uint16_t>::max()) {
        return op.emitOpError("expects E8M0 scale outer extents aligned to 16 and source K stride fitting in uint16_t");
    }
    const int64_t outer = getMxExtractExtent(dst[outerAxis], valid[outerAxis], compact);
    if (outer <= 0) {
        return op.emitOpError("expects a positive E8M0 scale textract outer extent");
    }
    // The aligned physical extent bounds the rounded compact footprint.
    const int64_t roundedOuter = ((outer - 1) / pto::kValue16 + 1) * pto::kValue16;
    const Value kIndex = left ? op.getIndexCol() : op.getIndexRow();
    const Value outerIndex = left ? op.getIndexRow() : op.getIndexCol();
    const auto kBounds = verifyMxExtractAxis<pto::kValue2>(
        op, kIndex, left ? "indexCol" : "indexRow", src[kAxis], dst[kAxis], k);
    if (failed(kBounds)) {
        return failure();
    }
    return verifyMxExtractAxis<pto::kValue16>(
        op, outerIndex, left ? "indexRow" : "indexCol", src[outerAxis], dst[outerAxis], roundedOuter);
}

static LogicalResult verifyMxExtractShape(TExtractOp op, const TExtractCommon& c, bool left)
{
    const auto src = c.srcTb.getShape();
    const auto dst = c.dstTb.getShape();
    const bool hasDynamicShape = llvm::any_of(src, ShapedType::isDynamic) || llvm::any_of(dst, ShapedType::isDynamic);
    if (hasDynamicShape) {
        return op.emitOpError("expects E8M0 scale textract to use static physical shapes");
    }
    const unsigned kAxis = left ? 1 : 0;
    if (src[kAxis] % pto::kValue2 != 0 || dst[kAxis] % pto::kValue2 != 0) {
        return op.emitOpError("expects E8M0 scale textract K extents to be aligned to 2");
    }
    if (left && dst[0] == 1) {
        return verifyMxExtractSingleRow(op, c);
    }
    return verifyMxExtractMatrix(op, c, left);
}

static LogicalResult verifyTExtractA5MxScale(TExtractOp op, const TExtractCommon& c)
{
    if (!isPTOF8E8M0Type(c.dstElem)) {
        return op.emitOpError("expects A5 mat->scaling textract to use !pto.f8E8M0");
    }
    const int32_t bl = c.dstTb.getBLayoutValueI32();
    const int32_t sl = c.dstTb.getSLayoutValueI32();
    const bool left = bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                      sl == static_cast<int32_t>(pto::SLayout::RowMajor);
    const bool right = bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                       sl == static_cast<int32_t>(pto::SLayout::ColMajor);
    const bool matchingLayout = c.srcTb.getBLayoutValueI32() == bl && c.srcTb.getSLayoutValueI32() == sl;
    const bool supportedLayout = left || right;
    if (!supportedLayout || !matchingLayout) {
        return op.emitOpError("expects E8M0 scale textract src and dst to use matching ZZ or NN layouts");
    }
    const bool mxFractal = c.srcTb.getSFractalSizeI32() == pto::kValue32 &&
                           c.dstTb.getSFractalSizeI32() == pto::kValue32;
    if (!mxFractal) {
        return op.emitOpError("expects E8M0 scale textract to use fractal=32");
    }
    const auto srcCompact = c.srcTb.getCompactModeI32();
    const auto dstCompact = c.dstTb.getCompactModeI32();
    const auto noCompact = static_cast<int32_t>(pto::CompactMode::Null);
    const auto normalCompact = static_cast<int32_t>(pto::CompactMode::Normal);
    const bool supportedCompact = srcCompact == noCompact && (dstCompact == noCompact || dstCompact == normalCompact);
    if (!supportedCompact) {
        return op.emitOpError("expects E8M0 scale textract compact mode Null on src and Null or Normal on dst");
    }
    return verifyMxExtractShape(op, c, left);
}

static LogicalResult verifyTExtractA5Mat(TExtractOp op, const TExtractCommon& c)
{
    const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
    const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (hasPreQuantScalar || hasRelu) {
        return op.emitOpError("expects mat-source textract to use the base form without preQuantScalar or reluPreMode");
    }
    if (*c.dstSpace == pto::AddressSpace::SCALING) {
        return verifyTExtractA5MxScale(op, c);
    }
    if (!hasMatExtractSourceLayoutA5(c.srcTb, *c.dstSpace)) {
        return op.emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
    }
    if (*c.dstSpace == pto::AddressSpace::LEFT) {
        const bool hasInvalidLeftLayout =
            c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
            c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor);
        if (hasInvalidLeftLayout) {
            return op.emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
        }
    } else if (*c.dstSpace == pto::AddressSpace::RIGHT) {
        const bool hasInvalidRightLayout =
            c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor);
        if (hasInvalidRightLayout) {
            return op.emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
        }
    }
    return success();
}

static LogicalResult verifyTExtractA5Acc(TExtractOp op, const TExtractCommon& c)
{
    const bool hasInvalidSrcLayout = c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
                                     c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor);
    if (hasInvalidSrcLayout) {
        return op.emitOpError("expects A5 acc-source textract src to use blayout=col_major and slayout=row_major");
    }
    if (*c.dstSpace == pto::AddressSpace::MAT) {
        const bool hasInvalidDstLayout = c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
                                         c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor);
        if (hasInvalidDstLayout) {
            return op.emitOpError("expects A5 acc-source textract dst to use blayout=col_major and slayout=row_major");
        }
    } else {
        if (!isRowMajorNoneBoxND(c.dstTb)) {
            return op.emitOpError(
                "expects A5 acc->vec textract dst to use ND layout (blayout=row_major, slayout=none_box)");
        }
    }
    const bool hasFp = static_cast<bool>(op.getFp());
    const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
    if (hasFp || hasPreQuantScalar) {
        if (!isA5AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
            return op.emitOpError(
                "expects A5 acc preQuantScalar textract element types to be "
                "(src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)");
        }
    } else if (!isA5AccCastExtractTypePair(c.srcElem, c.dstElem)) {
        return op.emitOpError(
            "expects A5 acc textract element types to be "
            "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
    }
    return success();
}

static LogicalResult verifyTExtractA5(TExtractOp op)
{
    auto common = verifyTExtractCommon(op, true);
    if (failed(common)) {
        return failure();
    }
    const TExtractCommon& c = *common;
    const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
    const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    const bool isScaleExtract = c.srcSpace == pto::AddressSpace::MAT && c.dstSpace == pto::AddressSpace::SCALING;
    if (!isScaleExtract && !isA5ExtractElemType(c.dstElem)) {
        return op.emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
    }
    if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
        return failure();
    }
    const bool hasInvalidAccToVecMode =
        op.getAccToVecModeAttr() &&
        (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC || *c.dstSpace != pto::AddressSpace::VEC);
    if (hasInvalidAccToVecMode) {
        return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
    }
    if (!c.srcSpace || !c.dstSpace) {
        return op.emitOpError("expects src and dst to have explicit loc");
    }
    if (!isTExtractA5SupportedPair(*c.srcSpace, *c.dstSpace)) {
        return op.emitOpError("expects A5 textract to use a supported src/dst loc pair");
    }
    if (*c.srcSpace == pto::AddressSpace::MAT) {
        return verifyTExtractA5Mat(op, c);
    }
    if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::VEC) {
        if (hasPreQuantScalar || hasRelu) {
            return op.emitOpError(
                "expects vec-source textract to use the base form without preQuantScalar or reluPreMode");
        }
        if (!isRowMajorNoneBoxND(c.srcTb) || !isRowMajorNoneBoxND(c.dstTb)) {
            return op.emitOpError(
                "expects A5 vec->vec textract src/dst to use ND layout "
                "(blayout=row_major, slayout=none_box)");
        }
        return success();
    }
    if (*c.srcSpace == pto::AddressSpace::ACC) {
        return verifyTExtractA5Acc(op, c);
    }
    return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify()
{
    if (isa<pto::ConvTileBufType>(getSrc().getType())) {
        const bool allowLowPrecision = getVerifierTargetArch(getOperation()) == VerifierTargetArch::A5;
        return verifyConvTileTExtract(*this, allowLowPrecision);
    }
    auto verifyA2A3 = [this]() -> LogicalResult { return verifyTExtractA2A3(*this); };
    auto verifyA5 = [this]() -> LogicalResult { return verifyTExtractA5(*this); };
    return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
static bool isA2A3AccCastInsertTypePair(Type srcElem, Type dstElem)
{
    return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantInsertTypePair(Type srcElem, Type dstElem)
{
    return isA2A3AccQuantTypePair(srcElem, dstElem);
}

static bool isColMajorRowMajorNZ(pto::TileBufType ty)
{
    return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5SupportedVecElemType(Type ty)
{
    if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
        return true;
    }
    if (auto it = dyn_cast<IntegerType>(ty)) {
        return it.getWidth() == mlir::pto::kValue8 || it.getWidth() == mlir::pto::kValue32;
    }
    if (auto ft = dyn_cast<FloatType>(ty)) {
        return ft.getWidth() == mlir::pto::kValue8 || ft.isF16() || ft.isBF16() || ft.isF32();
    }
    return false;
}

static bool isA2A3VecInsertElemType(Type ty)
{
    return ty.isInteger(mlir::pto::kValue8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

using TInsertCommon = TileTransferCommon;

static FailureOr<TInsertCommon> verifyTInsertCommon(TInsertOp op, bool allowLowPrecision)
{
    return verifyTileTransferCommon(
        op, op.getSrc(), op.getDst(), op.getIndexRow(), op.getIndexCol(), allowLowPrecision, true, true);
}
