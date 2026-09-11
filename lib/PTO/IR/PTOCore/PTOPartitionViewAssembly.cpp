// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
