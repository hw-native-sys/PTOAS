// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

void mlir::pto::PartitionViewOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << ", offsets = [";
  p.printOperands(getOffsets());
  p << "], sizes = [";
  p.printOperands(getSizes());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  p << " : " << getSource().getType();

  auto inferredResultType = inferPartitionViewResultTypeFromSizes(
      dyn_cast<mlir::pto::TensorViewType>(getSource().getType()), getSizes());
  if (succeeded(inferredResultType) && *inferredResultType == getResult().getType())
    return;

  p << " -> " << getResult().getType();
}

static std::optional<int64_t> getConstantIntegerValueEx(
    Value v, bool includeIndexAndIntOpsInConstFold) {
  if (includeIndexAndIntOpsInConstFold) {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
      return c.value();
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
      return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static LogicalResult verifyNonNegativeIndexRowCol(
    Operation &op, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold) {
  if (!indexRow.getType().isIndex() || !indexCol.getType().isIndex())
    return op.emitOpError("expects indexRow and indexCol to be index type");
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  if (row && *row < 0)
    return op.emitOpError("expects indexRow to be non-negative");
  if (col && *col < 0)
    return op.emitOpError("expects indexCol to be non-negative");
  return success();
}

static LogicalResult verifyExtractStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != kPTORowColRank || dstShape.size() != kPTORowColRank)
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + dstShape[0] > srcShape[0])
    return op.emitOpError("expects indexRow + dst.rows <= src.rows");
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + dstShape[1] > srcShape[1])
    return op.emitOpError("expects indexCol + dst.cols <= src.cols");
  return success();
}

static LogicalResult verifyInsertStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getValidShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != kPTORowColRank || dstShape.size() != kPTORowColRank)
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + srcShape[0] > dstShape[0])
    return op.emitOpError("expects indexRow + src.rows <= dst.rows");
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + srcShape[1] > dstShape[1])
    return op.emitOpError("expects indexCol + src.cols <= dst.cols");
  return success();
}

static unsigned getElemByteSize(Type ty) {
  return getPTOStorageElemByteSize(ty);
}

static bool readBLayoutValue(Attribute attr, int32_t &out) {
  if (auto layout = dyn_cast_or_null<BLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layout.getValue());
    return true;
  }
  if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(value.getInt());
    return true;
  }
  return false;
}

static bool readSLayoutValue(Attribute attr, int32_t &out) {
  if (auto layout = dyn_cast_or_null<SLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layout.getValue());
    return true;
  }
  if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(value.getInt());
    return true;
  }
  return false;
}

static LogicalResult verifyTileBufPositiveShape(Operation *op,
                                                ArrayRef<int64_t> shape,
                                                StringRef name) {
  if (shape.size() != kPTORowColRank)
    return op->emitOpError() << "expects " << name << " to be rank-2";
  if (shape[0] != ShapedType::kDynamic && shape[0] <= 0)
    return op->emitOpError() << "expects " << name << " rows to be positive";
  if (shape[1] != ShapedType::kDynamic && shape[1] <= 0)
    return op->emitOpError() << "expects " << name << " cols to be positive";
  return success();
}

static LogicalResult verifyNoneBoxTileBufLayout(Operation *op, StringRef name,
                                                int32_t blayout,
                                                int64_t rows, int64_t cols,
                                                unsigned elemBytes) {
  constexpr int64_t kAlignedBytes = 32;
  auto checkByteAlignment = [op, name, elemBytes](
                                int64_t dim, StringRef layoutName,
                                StringRef byteExpr) -> LogicalResult {
    if (dim == ShapedType::kDynamic)
      return success();
    int64_t bytes = dim * static_cast<int64_t>(elemBytes);
    if (bytes % kAlignedBytes == 0)
      return success();
    return op->emitOpError()
           << "expects " << name << " " << layoutName
           << " none_box tile " << byteExpr
           << " to be 32-byte aligned, but got " << bytes << " bytes";
  };
  if (blayout == static_cast<int32_t>(BLayout::RowMajor))
    return checkByteAlignment(cols, "row-major",
                              "row byte size (cols * sizeof(dtype))");
  return checkByteAlignment(rows, "col-major",
                            "column byte size (rows * sizeof(dtype))");
}

static LogicalResult getBoxedTileInnerShape(Operation *op, StringRef name,
                                            int32_t slayout, int32_t fractal,
                                            unsigned elemBytes,
                                            int64_t &innerRows,
                                            int64_t &innerCols) {
  constexpr int64_t kAlignedBytes = 32;
  if (elemBytes == 0)
    return op->emitOpError() << "expects " << name
                             << " to have a non-zero element byte size";
  switch (fractal) {
  case kFractalSize1024:
    innerRows = kFractalSize16;
    innerCols = kFractalSize16;
    return success();
  case kFractalSize32:
    innerRows = kFractalSize16;
    innerCols = kFractalSize32 / kFractalSize16;
    return success();
  case kFractalSize512:
    if (kAlignedBytes % elemBytes != 0) {
      return op->emitOpError() << "expects " << name
                               << " element byte size to divide 32 for boxed "
                                  "fractal-512 tile layout";
    }
    if (slayout == static_cast<int32_t>(SLayout::RowMajor)) {
      innerRows = kFractalSize16;
      innerCols = kAlignedBytes / static_cast<int64_t>(elemBytes);
      return success();
    }
    if (slayout == static_cast<int32_t>(SLayout::ColMajor)) {
      innerRows = kAlignedBytes / static_cast<int64_t>(elemBytes);
      innerCols = kFractalSize16;
      return success();
    }
    break;
  default:
    break;
  }
  return op->emitOpError() << "expects " << name
                           << " to use a supported boxed tile layout";
}
