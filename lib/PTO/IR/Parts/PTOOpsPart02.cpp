// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

namespace {
struct PartitionViewParseState {
    OpAsmParser::UnresolvedOperand source;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> sizes;
    Type sourceTy;
    Type resultTy;
    bool hasExplicitResultTy = false;
};

static ParseResult parseOptionalResultType(OpAsmParser &parser, Type &resultTy,
                                           bool &hasExplicitResultTy) {
  if (failed(parser.parseOptionalArrow()))
    return success();
  if (parser.parseType(resultTy))
    return failure();
  hasExplicitResultTy = true;
  return success();
}

static ParseResult parsePartitionViewSyntax(OpAsmParser& parser, OperationState& result, PartitionViewParseState& state)
{
    if (parser.parseOperand(state.source) || parser.parseComma() || parser.parseKeyword("offsets") ||
        parser.parseEqual() || parser.parseLSquare() || parser.parseOperandList(state.offsets) ||
        parser.parseRSquare() || parser.parseComma() || parser.parseKeyword("sizes") || parser.parseEqual() ||
        parser.parseLSquare() || parser.parseOperandList(state.sizes) || parser.parseRSquare() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(state.sourceTy)) {
        return failure();
    }
    return parseOptionalResultType(parser, state.resultTy,
                                   state.hasExplicitResultTy);
}

static ParseResult resolvePartitionViewOperands(
    OpAsmParser& parser, OperationState& result, PartitionViewParseState& state)
{
    if (parser.resolveOperand(state.source, state.sourceTy, result.operands)) {
        return failure();
    }
    Type indexTy = parser.getBuilder().getIndexType();
    if (parser.resolveOperands(state.offsets, indexTy, result.operands) ||
        parser.resolveOperands(state.sizes, indexTy, result.operands)) {
        return failure();
    }
    auto& properties = result.getOrAddProperties<PartitionViewOp::Properties>();
    llvm::copy(
        ArrayRef<int32_t>({1, static_cast<int32_t>(state.offsets.size()), static_cast<int32_t>(state.sizes.size())}),
        properties.operandSegmentSizes.begin());
    return success();
}

static ParseResult inferPartitionViewParseResult(
    OpAsmParser& parser, OperationState& result, PartitionViewParseState& state)
{
    if (state.hasExplicitResultTy) {
        result.addTypes(state.resultTy);
        return success();
    }
    ValueRange allOperands(result.operands);
    ValueRange sizeOperands = allOperands.slice(1 + state.offsets.size(), state.sizes.size());
    auto inferredResultType =
        inferPartitionViewResultTypeFromSizes(state.sourceTy, sizeOperands);
    if (failed(inferredResultType)) {
        return parser.emitError(parser.getCurrentLocation(), "failed to infer pto.partition_view result type");
    }

    result.addTypes(*inferredResultType);
    return success();
}
} // namespace

ParseResult mlir::pto::PartitionViewOp::parse(OpAsmParser& parser, OperationState& result)
{
    PartitionViewParseState state;
    if (failed(parsePartitionViewSyntax(parser, result, state)) ||
        failed(resolvePartitionViewOperands(parser, result, state))) {
        return failure();
    }
    return inferPartitionViewParseResult(parser, result, state);
}

void mlir::pto::PartitionViewOp::print(OpAsmPrinter& printer)
{
    printer << " " << getSource() << ", offsets = [";
    printer.printOperands(getOffsets());
    printer << "], sizes = [";
    printer.printOperands(getSizes());
    printer << "]";
    printer.printOptionalAttrDict(
        (*this)->getAttrs(),
        /*elidedAttrs=*/{"operandSegmentSizes"});
    printer << " : " << getSource().getType();

    auto inferredResultType =
        inferPartitionViewResultTypeFromSizes(getSource().getType(), getSizes());
    if (succeeded(inferredResultType) && *inferredResultType == getResult().getType()) {
        return;
    }

    printer << " -> " << getResult().getType();
}

static std::optional<int64_t> getConstantIntegerValueEx(Value v, bool includeIndexAndIntOpsInConstFold)
{
    if (includeIndexAndIntOpsInConstFold) {
        if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
            return c.value();
        }
        if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
            return c.value();
        }
    }
    if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
            return ia.getInt();
        }
    }
    return std::nullopt;
}

static LogicalResult verifyNonNegativeIndexRowCol(
    Operation& op, Value indexRow, Value indexCol, bool includeIndexAndIntOpsInConstFold)
{
    if (!indexRow.getType().isIndex() || !indexCol.getType().isIndex()) {
        return op.emitOpError("expects indexRow and indexCol to be index type");
    }
    auto row = getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
    auto col = getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
    if (row && *row < 0) {
        return op.emitOpError("expects indexRow to be non-negative");
    }
    if (col && *col < 0) {
        return op.emitOpError("expects indexCol to be non-negative");
    }
    return success();
}

static LogicalResult verifyExtractStaticBoundsCommon(
    Operation& op, Value indexRow, Value indexCol, Type srcTy, Type dstTy, bool includeIndexAndIntOpsInConstFold)
{
    auto row = getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
    auto col = getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
    auto srcShape = getShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (srcShape.size() != 2 || dstShape.size() != 2) {
        return op.emitOpError("expects src and dst to be rank-2 tile_buf");
    }
    if (row && srcShape[0] != ShapedType::kDynamic && dstShape[0] != ShapedType::kDynamic &&
        *row + dstShape[0] > srcShape[0]) {
        return op.emitOpError("expects indexRow + dst.rows <= src.rows");
    }
    if (col && srcShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        *col + dstShape[1] > srcShape[1]) {
        return op.emitOpError("expects indexCol + dst.cols <= src.cols");
    }
    return success();
}

static LogicalResult verifyInsertStaticBoundsCommon(
    Operation& op, Value indexRow, Value indexCol, Type srcTy, Type dstTy, bool includeIndexAndIntOpsInConstFold)
{
    auto row = getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
    auto col = getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
    auto srcShape = getValidShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (srcShape.size() != 2 || dstShape.size() != 2) {
        return op.emitOpError("expects src and dst to be rank-2 tile_buf");
    }
    if (row && srcShape[0] != ShapedType::kDynamic && dstShape[0] != ShapedType::kDynamic &&
        *row + srcShape[0] > dstShape[0]) {
        return op.emitOpError("expects indexRow + src.rows <= dst.rows");
    }
    if (col && srcShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        *col + srcShape[1] > dstShape[1]) {
        return op.emitOpError("expects indexCol + src.cols <= dst.cols");
    }
    return success();
}

static unsigned getElemByteSize(Type ty) { return getPTOStorageElemByteSize(ty); }

static LogicalResult verifyTileBufPositiveShape(Operation* op, ArrayRef<int64_t> shape, StringRef name)
{
    if (shape.size() != 2) {
        return op->emitOpError() << "expects " << name << " to be rank-2";
    }
    if (shape[0] != ShapedType::kDynamic && shape[0] <= 0) {
        return op->emitOpError() << "expects " << name << " rows to be positive";
    }
    if (shape[1] != ShapedType::kDynamic && shape[1] <= 0) {
        return op->emitOpError() << "expects " << name << " cols to be positive";
    }
    return success();
}

template <typename LayoutAttrT>
static bool readLayoutValue(Attribute attr, int32_t &out) {
  if (auto layout = dyn_cast_or_null<LayoutAttrT>(attr)) {
    out = static_cast<int32_t>(layout.getValue());
    return true;
  }
  if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(value.getInt());
    return true;
  }
  return false;
}

static bool readBLayoutValue(Attribute attr, int32_t &out) {
  return readLayoutValue<BLayoutAttr>(attr, out);
}

static bool readSLayoutValue(Attribute attr, int32_t &out) {
  return readLayoutValue<SLayoutAttr>(attr, out);
}

static LogicalResult verifyTileByteAlignment(
    Operation* op, StringRef name, int64_t dim, StringRef layoutName, StringRef byteExpr, unsigned elemBytes,
    int64_t requiredAlignment)
{
    if (requiredAlignment <= 0 || !llvm::isPowerOf2_64(requiredAlignment)) {
        return op->emitOpError() << "expects tile byte alignment to be a positive power of two";
    }
    if (dim == ShapedType::kDynamic) {
        return success();
    }
    int64_t bytes = dim * static_cast<int64_t>(elemBytes);
    auto alignmentMask = static_cast<uint64_t>(requiredAlignment) - 1;
    if ((static_cast<uint64_t>(bytes) & alignmentMask) == 0) {
        return success();
    }
    return op->emitOpError() << "expects " << name << " " << layoutName << " none_box tile " << byteExpr << " to be "
                             << requiredAlignment << "-byte aligned, but got " << bytes << " bytes";
}

static LogicalResult verifyNoneBoxTileBufLayout(
    Operation* op, StringRef name, int32_t blayout, int64_t rows, int64_t cols, unsigned elemBytes, bool packedFp4)
{
    constexpr int64_t kAlignedBytes = 32;
    constexpr int64_t kPackedFp4AlignedBytes = 16;
    int64_t alignment = packedFp4 ? kPackedFp4AlignedBytes : kAlignedBytes;
    if (blayout == static_cast<int32_t>(BLayout::RowMajor)) {
        return verifyTileByteAlignment(
            op, name, cols, "row-major", "row byte size (cols * sizeof(dtype))", elemBytes, alignment);
    }
    return verifyTileByteAlignment(
        op, name, rows, "col-major", "column byte size (rows * sizeof(dtype))", elemBytes, alignment);
}

static LogicalResult getBoxedTileInnerShape(
    Operation* op, StringRef name, int32_t slayout, int32_t fractal, unsigned elemBytes, int64_t& innerRows,
    int64_t& innerCols)
{
    constexpr int64_t kAlignedBytes = 32;
    innerRows = 0;
    innerCols = 0;
    if (elemBytes == 0) {
        return op->emitOpError() << "expects " << name
                                 << " element byte size to be non-zero";
    }
    switch (fractal) {
        case 1024:
            innerRows = 16;
            innerCols = 16;
            break;
        case 32:
            innerRows = 16;
            innerCols = 2;
            break;
        case 512:
            if (kAlignedBytes % elemBytes != 0) {
                return op->emitOpError() << "expects " << name
                                         << " element byte size to divide 32 for boxed "
                                            "fractal-512 tile layout";
            }
            if (slayout == static_cast<int32_t>(SLayout::RowMajor)) {
                innerRows = 16;
                innerCols = kAlignedBytes / static_cast<int64_t>(elemBytes);
            } else if (slayout == static_cast<int32_t>(SLayout::ColMajor)) {
                innerRows = kAlignedBytes / static_cast<int64_t>(elemBytes);
                innerCols = 16;
            }
            break;
        default:
            break;
    }
    if (innerRows <= 0 || innerCols <= 0) {
        return op->emitOpError() << "expects " << name << " to use a supported boxed tile layout";
    }
    return success();
}

static LogicalResult verifyBoxedTileBufLayout(
    Operation* op, pto::TileBufType tb, StringRef name, int64_t rows, int64_t cols, unsigned elemBytes, int32_t slayout,
    int32_t fractal)
{
    int64_t innerRows = 0;
    int64_t innerCols = 0;
    if (failed(getBoxedTileInnerShape(op, name, slayout, fractal, elemBytes, innerRows, innerCols))) {
        return failure();
    }
    auto loc = getPTOMemorySpaceEnum(tb);
    bool allowUnalignedRows = (loc && *loc == pto::AddressSpace::VEC) || fractal == 32 || rows == 1;
    if (!allowUnalignedRows && rows != ShapedType::kDynamic && rows % innerRows != 0) {
        return op->emitOpError() << "expects " << name << " boxed tile rows to be a multiple of innerRows ("
                                 << innerRows << "), but got " << rows;
    }
    if (cols != ShapedType::kDynamic && cols % innerCols != 0) {
        return op->emitOpError() << "expects " << name << " boxed tile cols to be a multiple of innerCols ("
                                 << innerCols << "), but got " << cols;
    }

    return success();
}

static LogicalResult verifyTileBufLayoutConstraints(Operation* op, pto::TileBufType tb, StringRef name)
{
    auto shape = tb.getShape();
    if (failed(verifyTileBufPositiveShape(op, shape, name))) {
        return failure();
    }
    unsigned elemBytes = getElemByteSize(tb.getElementType());
    if (elemBytes == 0) {
        return op->emitOpError() << "expects " << name << " element type to have a byte size";
    }
    auto cfg = tb.getConfigAttr();
    if (!cfg) {
        cfg = TileBufConfigAttr::getDefault(tb.getContext());
    }
    int32_t blayout = 0;
    int32_t slayout = 0;
    if (!readBLayoutValue(cfg.getBLayout(), blayout) || !readSLayoutValue(cfg.getSLayout(), slayout)) {
        return op->emitOpError() << "expects " << name << " to have concrete tile layout attributes";
    }
    if (slayout == static_cast<int32_t>(SLayout::NoneBox)) {
        return verifyNoneBoxTileBufLayout(
            op, name, blayout, shape[0], shape[1], elemBytes, isPTOFloat4PackedType(tb.getElementType()));
    }
    int32_t fractal = static_cast<int32_t>(cfg.getSFractalSize().getInt());
    return verifyBoxedTileBufLayout(op, tb, name, shape[0], shape[1], elemBytes, slayout, fractal);
}

[[maybe_unused]] static bool isSupportedLoadStoreElemTypeA2A3(Type ty)
{
    if (ty.isF16() || ty.isBF16() || ty.isF32()) {
        return true;
    }
    if (auto it = dyn_cast<IntegerType>(ty)) {
        unsigned width = it.getWidth();
        return width == 8 || width == 16 || width == 32 || width == 64;
    }
    return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty)
{
    if (ty.isF16() || ty.isF32()) {
        return true;
    }
    if (auto it = dyn_cast<IntegerType>(ty)) {
        unsigned width = it.getWidth();
        return width == 16 || width == 32;
    }
    return false;
}

static bool isSupportedGatherElemTypeA5(Type ty)
{
    if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16()) {
        return true;
    }
    if (isPTOHiFloat8Type(ty)) {
        return true;
    }
    if (auto ft = dyn_cast<FloatType>(ty)) {
        unsigned width = ft.getWidth();
        return width == 8;
    }
    if (auto it = dyn_cast<IntegerType>(ty)) {
        return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
    }
    return false;
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value)
{
    if (!value) {
        return std::nullopt;
    }
    if (auto type = dyn_cast<pto::TensorViewType>(value.getType())) {
        if (auto layout = type.getLayoutAttr()) {
            return layout.getLayout();
        }
    } else if (auto type = dyn_cast<pto::PartitionTensorViewType>(value.getType())) {
        if (auto layout = type.getLayoutAttr()) {
            return layout.getLayout();
        }
    }
    if (auto part = value.getDefiningOp<pto::PartitionViewOp>()) {
        return getLogicalViewLayout(part.getSource());
    }
    if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
        // Prefer the explicit layout attribute when available.  After rank-2 →
        // rank-5 canonicalization, the padded leading strides satisfy the ND
        // (row-major) recurrence even for DN (col-major) data, so inferLayout
        // alone would misclassify DN as ND (the col-major recurrence breaks at
        // the boundary between padded unit-extent dims and real dims).  The
        // layout attribute carries the *intended* memory layout and is the
        // authoritative source — inferLayout is only a fallback for views that
        // lack an explicit layout.
        if (auto layoutAttr = make.getLayoutAttr()) {
            return layoutAttr.getLayout();
        }
        auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
        if (!tvTy) {
            return std::nullopt;
        }
        SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
        SmallVector<int64_t> strides;
        strides.reserve(make.getStrides().size());
        for (Value stride : make.getStrides()) {
            auto cst = getConstIndexValue(stride);
            if (!cst) {
                return std::nullopt;
            }
            strides.push_back(*cst);
        }
        return pto::inferLayout5D(shape, strides, getElemByteSize(tvTy.getElementType()));
    }
    return std::nullopt;
}

static bool getLogicalViewShape(Value value, SmallVectorImpl<int64_t>& shape)
{
    if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
        auto type = dyn_cast<pto::TensorViewType>(make.getResult().getType());
        if (!type) {
            return false;
        }
        shape.assign(type.getShape().begin(), type.getShape().end());
        for (auto [index, operand] : llvm::enumerate(make.getShape())) {
            if (shape[index] == ShapedType::kDynamic) {
                if (auto constant = getConstIndexValue(operand)) {
                    shape[index] = *constant;
                }
            }
        }
        return true;
    }

    if (auto partition = value.getDefiningOp<pto::PartitionViewOp>()) {
        auto type = dyn_cast<pto::PartitionTensorViewType>(partition.getResult().getType());
        if (!type) {
            return false;
        }
        shape.assign(type.getShape().begin(), type.getShape().end());
        for (auto [index, operand] : llvm::enumerate(partition.getSizes())) {
            if (shape[index] == ShapedType::kDynamic) {
                if (auto constant = getConstIndexValue(operand)) {
                    shape[index] = *constant;
                }
            }
        }
        return true;
    }

    return false;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type)
{
    if (!type) {
        return std::nullopt;
    }
    int32_t sl = type.getSLayoutValueI32();
    int32_t bl = type.getBLayoutValueI32();
    if (sl != static_cast<int32_t>(pto::SLayout::NoneBox)) {
        return pto::Layout::NZ;
    }
    if (bl == static_cast<int32_t>(pto::BLayout::RowMajor)) {
        return pto::Layout::ND;
    }
    if (bl == static_cast<int32_t>(pto::BLayout::ColMajor)) {
        return pto::Layout::DN;
    }
    return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty)
{
    auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
    return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static bool isColMajorTileBuf(Type ty)
{
    auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
    return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::ColMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation* op, Type ty, StringRef name)
{
    if (failed(verifyTileBufCommon(op, ty, name))) {
        return failure();
    }
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC) {
        return op->emitOpError() << "expects " << name << " to be in the vec address space";
    }
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
            return op->emitOpError() << "expects " << name << " to use the row_major blayout";
        }
    }
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
            return op->emitOpError() << "expects " << name << " to use the none_box slayout";
        }
    }
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
        auto layout = getTileBufLogicalLayout(tb);
        if (layout && *layout != pto::Layout::ND) {
            return op->emitOpError() << "expects " << name << " to use an ND-style tile layout";
        }
    }
    return success();
}

static LogicalResult verifyTileBufInVec(Operation *op, Type ty, StringRef name,
                                        bool allowLowPrecision = false) {
    if (failed(verifyTileBufCommon(op, ty, name, allowLowPrecision)))
        return failure();
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC)
        return op->emitOpError() << "expects " << name << " to be in the vec address space";
    return success();
}

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

LogicalResult mlir::pto::FusionRegionOp::verify()
{
    Region& bodyRegion = getBody();
    if (bodyRegion.empty()) {
        return emitOpError("expects a non-empty body region");
    }

    Block& body = bodyRegion.front();
    if (body.getNumArguments() != 0) {
        return emitOpError() << "expects body block to have no arguments, got " << body.getNumArguments();
    }

    if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>()) {
        return emitOpError("expects body to terminate with pto.yield");
    }

    auto yield = dyn_cast<YieldOp>(&body.back());
    if (!yield) {
        return emitOpError("expects body to terminate with pto.yield");
    }

    if (yield.getValues().size() != getOutputs().size()) {
        return emitOpError() << "expects pto.yield to return " << getOutputs().size() << " values, got "
                             << yield.getValues().size();
    }

    if (failed(verifyYieldedValueTypes(getOperation(), yield.getValues(),
                                       getOutputs())))
        return failure();

    return success();
}

LogicalResult mlir::pto::YieldOp::verify()
{
    auto parent = dyn_cast_or_null<FusionRegionOp>(getOperation()->getParentOp());
    if (!parent) {
        return emitOpError("expects parent op to be pto.fusion_region");
    }

    if (getValues().size() != parent.getOutputs().size()) {
        return emitOpError() << "expects " << parent.getOutputs().size()
                             << " yielded values to match parent results, got " << getValues().size();
    }

    return verifyYieldedValueTypes(getOperation(), getValues(),
                                   parent.getOutputs());
}

static SmallVector<int64_t> getConstantOrDynamicValues(ValueRange values)
{
    SmallVector<int64_t> result;
    result.reserve(values.size());
    for (Value value : values) {
        result.push_back(getConstIndexValue(value).value_or(ShapedType::kDynamic));
    }
    return result;
}

static SmallVector<int64_t> getResolvedMakeTensorViewShape(
    mlir::pto::MakeTensorViewOp op, mlir::pto::TensorViewType resultType)
{
    SmallVector<int64_t> shape(resultType.getShape());
    for (auto [index, dim] : llvm::enumerate(shape)) {
        if (dim != ShapedType::kDynamic) {
            continue;
        }
        if (auto constant = getConstIndexValue(op.getShape()[index])) {
            shape[index] = *constant;
        }
    }
    return shape;
}

static LogicalResult verifyMakeTensorViewLayout(
    mlir::pto::MakeTensorViewOp op, mlir::pto::TensorViewType resultType)
{
    auto opLayoutAttr = op.getLayoutAttr();
    auto typeLayoutAttr = resultType.getLayoutAttr();
    if (opLayoutAttr && typeLayoutAttr && opLayoutAttr != typeLayoutAttr) {
        return op.emitOpError() << "layout attribute " << opLayoutAttr
                                << " does not match result type layout " << typeLayoutAttr;
    }

    auto layoutAttr = opLayoutAttr ? opLayoutAttr : typeLayoutAttr;
    if (!layoutAttr) {
        return success();
    }

    SmallVector<int64_t> shape = getResolvedMakeTensorViewShape(op, resultType);
    SmallVector<int64_t> strides = getConstantOrDynamicValues(op.getStrides());
    Layout layout = layoutAttr.getLayout();
    unsigned storageElemBytes = getElemByteSize(resultType.getElementType());
    if (pto::isLayoutCompatible5D(layout, shape, strides, storageElemBytes)) {
        return success();
    }
    if (layout != Layout::NZ) {
        return op.emitOpError() << "user-specified layout=" << stringifyLayout(layout)
                                << " is incompatible with the view shape/stride";
    }
    if (resultType.getRank() != static_cast<int64_t>(pto::kPTOLayoutRank)) {
        return op.emitOpError() << "user-specified layout=nz requires a rank-5 view, got rank "
                                << resultType.getRank();
    }

    auto error = pto::getNZViewCompatibilityError(shape, strides, storageElemBytes);
    return op.emitOpError() << "user-specified layout=nz is incompatible with shape/stride: "
                            << error.value_or("unknown NZ layout mismatch");
}

LogicalResult mlir::pto::MakeTensorViewOp::verify()
{
    auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
    if (!tvTy) {
        return emitOpError("result must be pto.tensor_view<...>");
    }

    auto ptrTy = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
    if (!ptrTy) {
        return emitOpError("ptr operand must be !pto.ptr<...>");
    }
    Type ptrElemTy = ptrTy.getElementType();

    if (ptrElemTy != tvTy.getElementType()) {
        return emitOpError() << "ptr element type must match tensor_view element "
                                "type, but got ptr="
                             << ptrElemTy << " view=" << tvTy.getElementType();
    }

    int64_t rank = tvTy.getRank();
    if (static_cast<int64_t>(getShape().size()) != rank ||
        static_cast<int64_t>(getStrides().size()) != rank) {
        return emitOpError() << "shape/strides operand counts must match tensor_view rank=" << rank;
    }
    return verifyMakeTensorViewLayout(*this, tvTy);
}

struct PartitionSourceInfo {
    Type elementType;
    int64_t rank;
    SmallVector<int64_t> shape;
};

static FailureOr<PartitionSourceInfo> getPartitionSourceInfo(mlir::pto::PartitionViewOp op)
{
    PartitionSourceInfo info;
    if (auto tensorView = dyn_cast<mlir::pto::TensorViewType>(op.getSource().getType())) {
        info.elementType = tensorView.getElementType();
        info.rank = tensorView.getRank();
        info.shape.assign(tensorView.getShape().begin(), tensorView.getShape().end());
    } else if (auto partitionView = dyn_cast<mlir::pto::PartitionTensorViewType>(op.getSource().getType())) {
        info.elementType = partitionView.getElementType();
        info.rank = partitionView.getRank();
        info.shape.assign(partitionView.getShape().begin(), partitionView.getShape().end());
    } else {
        op.emitOpError("expects tensor_view or partition_tensor_view source");
        return failure();
    }

    SmallVector<int64_t> logicalShape;
    if (getLogicalViewShape(op.getSource(), logicalShape)) {
        info.shape = std::move(logicalShape);
    }
    return info;
}

static LogicalResult verifyPartitionSignature(mlir::pto::PartitionViewOp op,
                                              mlir::pto::PartitionTensorViewType resultType,
                                              const PartitionSourceInfo& source)
{
    if (source.elementType != resultType.getElementType()) {
        return op.emitOpError() << "element type mismatch between source and result: src=" << source.elementType
                                << " result=" << resultType.getElementType();
    }
    if (static_cast<int64_t>(op.getOffsets().size()) != source.rank) {
        return op.emitOpError() << "offset count (" << op.getOffsets().size() << ") must match source rank ("
                                << source.rank << ")";
    }
    if (static_cast<int64_t>(op.getSizes().size()) != source.rank) {
        return op.emitOpError() << "size count (" << op.getSizes().size() << ") must match source rank ("
                                << source.rank << ")";
    }
    return success();
}

static LogicalResult verifyPartitionDimension(mlir::pto::PartitionViewOp op, int64_t index, int64_t sourceDim,
                                              std::optional<int64_t> resultDim)
{
    auto offset = getConstIndexValue(op.getOffsets()[index]);
    auto size = getConstIndexValue(op.getSizes()[index]);
    if (offset && *offset < 0) {
        return op.emitOpError() << "offset at dim " << index << " must be non-negative, got " << *offset;
    }
    if (size && *size <= 0) {
        return op.emitOpError() << "size at dim " << index << " must be positive, got " << *size;
    }
    if (resultDim && size && *resultDim != ShapedType::kDynamic && *size != *resultDim) {
        return op.emitOpError() << "size/result mismatch at dim " << index << ": size operand=" << *size
                                << " result type dim=" << *resultDim;
    }
    if (sourceDim == ShapedType::kDynamic) {
        return success();
    }
    if (size && *size > sourceDim) {
        return op.emitOpError() << "size at dim " << index << " (" << *size << ") exceeds static source dim ("
                                << sourceDim << ")";
    }
    if (!offset || !size) {
        return success();
    }
    int64_t end = 0;
    if (llvm::AddOverflow(*offset, *size, end)) {
        return op.emitOpError() << "offset+size at dim " << index << " overflows";
    }
    if (end > sourceDim) {
        return op.emitOpError() << "offset+size at dim " << index << " (" << end
                                << ") exceeds static source dim (" << sourceDim << ")";
    }
    return success();
}

static LogicalResult verifyPartitionBounds(mlir::pto::PartitionViewOp op,
                                           mlir::pto::PartitionTensorViewType resultType,
                                           const PartitionSourceInfo& source)
{
    bool sameRank = resultType.getRank() == source.rank;
    for (int64_t index = 0; index < source.rank; ++index) {
        std::optional<int64_t> resultDim;
        if (sameRank) {
            resultDim = resultType.getShape()[index];
        }
        if (failed(verifyPartitionDimension(op, index, source.shape[index], resultDim))) {
            return failure();
        }
    }
    return success();
}

static LogicalResult verifyNZPartition(mlir::pto::PartitionViewOp op, const PartitionSourceInfo& source)
{
    if (getLogicalViewLayout(op.getSource()) != Layout::NZ) {
        return success();
    }
    SmallVector<int64_t> offsets = getConstantOrDynamicValues(op.getOffsets());
    SmallVector<int64_t> sizes = getConstantOrDynamicValues(op.getSizes());
    if (auto error = pto::getNZSubviewCompatibilityError(source.shape, offsets, sizes)) {
        return op.emitOpError(*error);
    }
    return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify()
{
    auto resultType = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
    if (!resultType) {
        return emitOpError("expects partition_tensor_view result");
    }
    FailureOr<PartitionSourceInfo> source = getPartitionSourceInfo(*this);
    if (failed(source)) {
        return failure();
    }
    if (failed(verifyPartitionSignature(*this, resultType, *source)) ||
        failed(verifyPartitionBounds(*this, resultType, *source))) {
        return failure();
    }
    return verifyNZPartition(*this, *source);
}

LogicalResult mlir::pto::AddPtrOp::verify()
{
    Value ptr = getOperation()->getOperand(0);
    Value result = getOperation()->getResult(0);

    auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
    if (!ptrTy) {
        return emitOpError("ptr operand must be !pto.ptr<...>");
    }

    auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
    if (!resTy) {
        return emitOpError("result must be !pto.ptr<...>");
    }

    if (ptrTy != resTy) {
        return emitOpError("result type must match ptr operand type");
    }

    return success();
}

static Type getPointerLikeElementType(Type type)
{
    if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
        return ptrTy.getElementType();
    }
    return Type();
}

static bool isEmitCSupportedScalarType(Type type)
{
    if (!type) {
        return false;
    }
    if (type.isF16() || type.isBF16() || type.isF32() || type.isF64()) {
        return true;
    }
    if (auto intTy = dyn_cast<IntegerType>(type)) {
        return intTy.getWidth() == 8 || intTy.getWidth() == 16 || intTy.getWidth() == 32 || intTy.getWidth() == 64;
    }
    if (mlir::pto::isPTOFloat8Type(type)) {
        return true;
    }
    if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type, mlir::pto::F4E2M1x2Type>(type)) {
        return true;
    }
    return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify()
{
    Type resultTy = getResult().getType();
    auto intTy = dyn_cast<IntegerType>(resultTy);
    if (!intTy || intTy.getWidth() != 64) {
        return emitOpError("result must be i64");
    }

    if (!isa<mlir::pto::PtrType>(getPtr().getType())) {
        return emitOpError("ptr operand must be !pto.ptr<...>");
    }
    return success();
}

LogicalResult mlir::pto::IntToPtrOp::verify()
{
    auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
    if (!addrTy || addrTy.getWidth() != 64) {
        return emitOpError("address operand must be i64");
    }

    if (!isa<mlir::pto::PtrType>(getResult().getType())) {
        return emitOpError("result must be !pto.ptr<...>");
    }

    Type dstElem = getPointerLikeElementType(getResult().getType());
    if (!isEmitCSupportedScalarType(dstElem)) {
        return emitOpError("result element type is not supported by EmitC: ") << dstElem;
    }

    return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify()
{
    auto arrayTy = getArray().getType();
    int64_t rank = arrayTy.getRank();
    int64_t numIdx = static_cast<int64_t>(getIndices().size());
    if (numIdx != rank) {
        return emitOpError() << "expects " << rank << " indices for !pto.local_array of rank " << rank << ", got "
                             << numIdx;
    }
    if (getResult().getType() != arrayTy.getElementType()) {
        return emitOpError() << "result type " << getResult().getType() << " does not match array element type "
                             << arrayTy.getElementType();
    }
    return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify()
{
    auto arrayTy = getArray().getType();
    int64_t rank = arrayTy.getRank();
    int64_t numIdx = static_cast<int64_t>(getIndices().size());
    if (numIdx != rank) {
        return emitOpError() << "expects " << rank << " indices for !pto.local_array of rank " << rank << ", got "
                             << numIdx;
    }
    if (getValue().getType() != arrayTy.getElementType()) {
        return emitOpError() << "value type " << getValue().getType() << " does not match array element type "
                             << arrayTy.getElementType();
    }
    return success();
}

// Resolve the field type reached by following a constant `path` of field
// indices from `root`, descending through nested structs. Emits an actionable
// op error and returns failure on an empty path, an out-of-range index, or a
// descent into a non-struct field. On success writes the terminal field type to
// `fieldTyOut`.
static LogicalResult walkStructPath(
    Operation* op, mlir::pto::StructType root, llvm::ArrayRef<int64_t> path, Type& fieldTyOut)
{
    if (path.empty()) {
        return op->emitOpError() << "struct path must have at least one index";
    }
    Type cur = root;
    for (auto [depth, idx] : llvm::enumerate(path)) {
        auto st = dyn_cast<mlir::pto::StructType>(cur);
        if (!st) {
            return op->emitOpError() << "struct path index " << depth << " descends into non-struct field of type "
                                     << cur;
        }
        if (idx < 0 || idx >= static_cast<int64_t>(st.getNumFields())) {
            return op->emitOpError() << "struct path index " << depth << " (" << idx << ") is out of range for " << st
                                     << " with " << st.getNumFields() << " field(s)";
        }
        cur = st.getFieldType(static_cast<unsigned>(idx));
    }
    fieldTyOut = cur;
    return success();
}

// The declared struct is stack storage owned by the enclosing scope, and the
// value lowers to a pointer to that storage. Letting it reach a terminator
// would publish that address outside the owning scope: `return %s` hands the
// caller a pointer into a dead frame, and `scf.yield %s` carries it out of the
// region that owns it. Both are rejected here rather than emitted as C++ that
// looks fine and is undefined at run time.
LogicalResult mlir::pto::DeclareStructOp::verify()
{
    for (Operation* user : getResult().getUsers()) {
        if (!user->hasTrait<mlir::OpTrait::IsTerminator>()) {
            continue;
        }
        return emitOpError() << "stack-local struct must not escape the scope that declares it, "
                                "but its value is passed to '"
                             << user->getName()
                             << "', which would expose the address of storage that is about to "
                                "die; declare the struct in the outer scope and mutate it from "
                                "the nested region instead (pto.struct_set mutates in place, "
                                "so a struct never needs to be returned or yielded)";
    }
    return success();
}

// Both accessors bottom out at a scalar. A path ending on a nested !pto.struct
// is rejected: the member chain lowers to `emitc.member`, which yields an
// lvalue, and handing a whole aggregate back as an SSA value would mean copying
// it out of the struct — so reaching into a nested struct is spelled as a longer
// path instead.
static LogicalResult verifyStructLeafIsScalar(Operation* op, Type fieldTy)
{
    if (!fieldTy.isIntOrFloat()) {
        return op->emitOpError() << "struct path must end at a scalar field, but ends at " << fieldTy
                                 << "; extend the path to reach a scalar inside it";
    }
    return success();
}

template <typename PathRange>
static LogicalResult verifyStructAccess(Operation *op, PathRange path,
                                        Type valueType, StringRef valueLabel) {
  Type fieldTy;
  if (failed(walkStructPath(
          op, cast<mlir::pto::StructType>(op->getOperand(0).getType()), path,
          fieldTy)) ||
      failed(verifyStructLeafIsScalar(op, fieldTy)))
    return failure();
  if (valueType != fieldTy)
    return op->emitOpError()
           << valueLabel << " type " << valueType << " does not match field type "
           << fieldTy << " at the given path";
  return success();
}

LogicalResult mlir::pto::StructGetOp::verify()
{
    return verifyStructAccess(getOperation(), getPath(), getValue().getType(),
                              "result");
}

LogicalResult mlir::pto::StructSetOp::verify()
{
    return verifyStructAccess(getOperation(), getPath(), getValue().getType(),
                              "value");
}

LogicalResult mlir::pto::CastPtrOp::verify()
{
    Type inputType = getInput().getType();
    Type resultType = getResult().getType();

    auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
    auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
    auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
    bool inputIsInteger = isa<IntegerType>(inputType);
    bool resultIsInteger = isa<IntegerType>(resultType);

    if (!inputPtrType && !inputMemRefType && !inputIsInteger) {
        return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
    }
    if (!resultPtrType && !resultIsInteger) {
        return emitOpError("result must be an integer or !pto.ptr<...>");
    }

    if (inputIsInteger && resultIsInteger) {
        return emitOpError("integer-to-integer cast is not a ptr cast");
    }

    if (inputMemRefType && resultIsInteger) {
        return emitOpError("memref-to-integer cast is unsupported");
    }

    if (inputMemRefType && resultPtrType) {
        auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(inputMemRefType.getMemorySpace());
        auto resultSpace = resultPtrType.getMemorySpace();
        if (memrefSpace && memrefSpace != resultSpace) {
            return emitOpError("memref-to-ptr cast must stay within the same PTO memory space");
        }
    }

    if (inputPtrType && resultPtrType && inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
        return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
    }

    return success();
}

void PTODialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
        >();

    addInterfaces<PTOInlinerInterface>();
}

AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type)
{
    if (auto ptrType = dyn_cast<PtrType>(type)) {
        return ptrType.getMemorySpace();
    }
    return {};
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func)
{
    return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func)
{
    return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::isPTOEntryFunction(LLVM::LLVMFuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    if (isPTOEntryFunction(func)) {
        return true;
    }
    auto attr = func->getAttrOfType<StringAttr>(kPTOVisibilityAttrName);
    if (!attr) {
        return false;
    }
    return attr.getValue() == kPTOVisibilityExternalValue;
}

void mlir::pto::setExternalArtifactVisibility(func::FuncOp func, bool isExternal)
{
    if (!func) {
        return;
    }
    if (isExternal) {
        func->setAttr(kPTOVisibilityAttrName, StringAttr::get(func.getContext(), kPTOVisibilityExternalValue));
        return;
    }
    func->removeAttr(kPTOVisibilityAttrName);
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module)
{
    if (!module) {
        return success();
    }

    for (auto func : module.getOps<func::FuncOp>()) {
        if (!hasExplicitPTOEntryAttr(func)) {
            continue;
        }
        if (func.isDeclaration()) {
            return func.emitOpError() << "`" << kPTOEntryAttrName << "` is only valid on function definitions";
        }
    }

    for (auto func : module.getOps<func::FuncOp>()) {
        if (!isPTOEntryFunction(func)) {
            continue;
        }
        if (func.getFunctionType().getNumResults() != 0) {
            return func.emitOpError() << "PTO entry functions must return void";
        }
    }
    return success();
}

// A !pto.struct is represented as a pointer to stack storage. Its provenance
// must therefore remain explicit: the value comes directly from
// pto.declare_struct in the owning function. Function arguments/results and
// operations such as arith.select and scf.if must not manufacture or relay a
// struct-typed SSA value, because that alias hides the declaration from
// DeclareStructOp's direct-use escape check. CFG block arguments cannot make a
// declaration safe to forward either: the branch is a terminator and is
// rejected by DeclareStructOp::verify.
LogicalResult mlir::pto::validateStructProvenance(ModuleOp module)
{
    if (!module) {
        return success();
    }

    WalkResult result = module.walk([&](Operation* op) -> WalkResult {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
            for (auto [i, inputTy] : llvm::enumerate(func.getFunctionType().getInputs())) {
                if (!isa<StructType>(inputTy)) {
                    continue;
                }
                func.emitOpError() << "argument " << i << " has type " << inputTy
                                   << ", but a stack-local struct must not be a function argument; "
                                      "structs must originate from 'pto.declare_struct' in the same "
                                      "function";
                return WalkResult::interrupt();
            }
            for (auto [i, resultTy] : llvm::enumerate(func.getFunctionType().getResults())) {
                if (!isa<StructType>(resultTy)) {
                    continue;
                }
                func.emitOpError() << "result " << i << " has type " << resultTy
                                   << ", but a stack-local struct must not be returned: the value is "
                                      "a pointer into the callee's frame, and returning it (even "
                                      "when it merely passes an argument back through) launders its "
                                      "provenance; keep the struct in its declaring function "
                                      "(pto.struct_set mutates in place, so a result is never needed)";
                return WalkResult::interrupt();
            }
        }

        if (!isa<DeclareStructOp>(op)) {
            for (auto [i, opResult] : llvm::enumerate(op->getResults())) {
                if (!isa<StructType>(opResult.getType())) {
                    continue;
                }
                op->emitOpError() << "result " << i << " has type " << opResult.getType()
                                  << ", but only 'pto.declare_struct' may produce a !pto.struct "
                                     "result; derived results hide the stack-storage lifetime and "
                                     "can escape their declaring scope";
                return WalkResult::interrupt();
            }
        }

        return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) { (void)module; }

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//===----------------------------------------------------------------------===//
