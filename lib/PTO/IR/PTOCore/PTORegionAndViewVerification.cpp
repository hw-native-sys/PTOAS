// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

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
