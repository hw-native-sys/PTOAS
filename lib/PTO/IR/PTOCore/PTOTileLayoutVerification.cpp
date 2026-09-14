// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
