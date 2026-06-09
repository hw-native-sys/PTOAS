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

static LogicalResult verifyBoxedTileBufLayout(Operation *op, pto::TileBufType tb,
                                              StringRef name, int64_t rows,
                                              int64_t cols, unsigned elemBytes,
                                              int32_t slayout,
                                              int32_t fractal) {
  int64_t innerRows = 0;
  int64_t innerCols = 0;
  if (failed(getBoxedTileInnerShape(op, name, slayout, fractal, elemBytes,
                                    innerRows, innerCols)))
    return failure();

  auto loc = getPTOMemorySpaceEnum(tb);
  bool allowUnalignedRows =
      (loc && *loc == pto::AddressSpace::VEC) || fractal == kFractalSize32 || rows == 1;
  if (!allowUnalignedRows && rows != ShapedType::kDynamic &&
      rows % innerRows != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile rows to be a multiple of innerRows (" << innerRows
           << "), but got " << rows;
  }
  if (cols != ShapedType::kDynamic && cols % innerCols != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile cols to be a multiple of innerCols (" << innerCols
           << "), but got " << cols;
  }
  return success();
}

static LogicalResult verifyTileBufLayoutConstraints(Operation *op,
                                                    pto::TileBufType tb,
                                                    StringRef name) {
  auto shape = tb.getShape();
  if (failed(verifyTileBufPositiveShape(op, shape, name)))
    return failure();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  unsigned elemBytes = getElemByteSize(tb.getElementType());
  if (elemBytes == 0)
    return op->emitOpError() << "expects " << name
                             << " element type to have a byte size";

  auto cfg = tb.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(tb.getContext());
  int32_t blayout = 0;
  int32_t slayout = 0;
  if (!readBLayoutValue(cfg.getBLayout(), blayout) ||
      !readSLayoutValue(cfg.getSLayout(), slayout)) {
    return op->emitOpError() << "expects " << name
                             << " to have concrete tile layout attributes";
  }

  if (slayout == static_cast<int32_t>(SLayout::NoneBox))
    return verifyNoneBoxTileBufLayout(op, name, blayout, rows, cols, elemBytes);
  int32_t fractal = static_cast<int32_t>(cfg.getSFractalSize().getInt());
  return verifyBoxedTileBufLayout(op, tb, name, rows, cols, elemBytes, slayout,
                                  fractal);
}

[[maybe_unused]] static bool isSupportedLoadStoreElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isBF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == kPTOI8BitWidth || width == kPTOI16BitWidth ||
           width == kPTOI32BitWidth || width == kPTOI64BitWidth;
  }
  return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == kPTOI16BitWidth || width == kPTOI32BitWidth;
  }
  return false;
}

static bool isSupportedGatherElemTypeA5(Type ty) {
  if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16())
    return true;
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == kPTOI8BitWidth;
  }
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == kPTOI8BitWidth || it.getWidth() == kPTOI16BitWidth || it.getWidth() == kPTOI32BitWidth;
  return false;
}

static std::optional<mlir::pto::Layout>
inferLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
            unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;

  // NZ / fractal: rank>=5, check middle dims (sh3/sh4/sh5 per spec)
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == kFractalSize512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return mlir::pto::Layout::NZ;
  }

  // ND: row-major contiguous
  bool isRowMajor = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i) {
    if (strides[i] != strides[i + 1] * shape[i + 1]) {
      isRowMajor = false;
      break;
    }
  }
  if (isRowMajor && strides.back() == 1)
    return mlir::pto::Layout::ND;

  // DN: col-major
  bool isColMajor = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i) {
    if (strides[i + 1] != strides[i] * shape[i]) {
      isColMajor = false;
      break;
    }
  }
  if (isColMajor && strides.front() == 1)
    return mlir::pto::Layout::DN;

  return mlir::pto::Layout::ND; // fallback
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value) {
  if (!value)
    return std::nullopt;
  if (auto part = value.getDefiningOp<pto::PartitionViewOp>())
    return getLogicalViewLayout(part.getSource());
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!tvTy)
      return std::nullopt;
    SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
    SmallVector<int64_t> strides;
    strides.reserve(make.getStrides().size());
    for (Value stride : make.getStrides()) {
      auto cst = getConstIndexValue(stride);
      if (!cst)
        return std::nullopt;
      strides.push_back(*cst);
    }
    return inferLayout(shape, strides, getElemByteSize(tvTy.getElementType()));
  }
  return std::nullopt;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type) {
  if (!type)
    return std::nullopt;
  int32_t sl = type.getSLayoutValueI32();
  int32_t bl = type.getBLayoutValueI32();
  if (sl != static_cast<int32_t>(pto::SLayout::NoneBox))
    return pto::Layout::NZ;
  if (bl == static_cast<int32_t>(pto::BLayout::RowMajor))
    return pto::Layout::ND;
  if (bl == static_cast<int32_t>(pto::BLayout::ColMajor))
    return pto::Layout::DN;
  return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  if (auto mr = dyn_cast<MemRefType>(ty))
    (void)mr;
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout != pto::Layout::ND)
      return op->emitOpError() << "expects " << name
                               << " to use an ND-style tile layout";
  }
  return success();
}
