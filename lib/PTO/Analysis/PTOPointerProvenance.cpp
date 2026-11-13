// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOMemoryAliasAnalysis.h"
#include "PTO/IR/TileProvenance.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::pto;

static std::optional<std::pair<Value, Value>> getPTOOperationAliasInfo(Operation* op)
{
    Value result = op->getNumResults() == 1 ? op->getResult(0) : Value{};
    Value source = result ? getTileViewSource(result) : Value{};
    if (source) {
        return std::make_pair(result, source);
    }
    if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
        return std::make_pair(makeViewOp.getResult(), makeViewOp.getPtr());
    }
    if (auto partViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
        return std::make_pair(partViewOp.getResult(), partViewOp.getSource());
    }
    if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(op)) {
        return std::make_pair(addPtrOp.getResult(), addPtrOp.getPtr());
    }
    if (auto castPtrOp = dyn_cast<pto::CastPtrOp>(op)) {
        return std::make_pair(castPtrOp.getResult(), castPtrOp.getInput());
    }
    if (auto bitcastOp = dyn_cast<pto::BitcastOp>(op)) {
        return std::make_pair(bitcastOp.getResult(), bitcastOp.getSrc());
    }
    if (auto multiGetOp = dyn_cast<pto::MultiTileGetOp>(op)) {
        return std::make_pair(multiGetOp.getResult(), multiGetOp.getSource());
    }
    return std::nullopt;
}

static std::optional<std::pair<Value, Value>> getMemRefOperationAliasInfo(Operation* op)
{
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
        return std::make_pair(subViewOp.getResult(), subViewOp.getViewSource());
    }
    if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {
        return std::make_pair(collapseShapeOp.getResult(), collapseShapeOp.getViewSource());
    }
    if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
        return std::make_pair(expandShapeOp.getResult(), expandShapeOp.getViewSource());
    }
    if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
        return std::make_pair(viewOp.getResult(), viewOp.getViewSource());
    }
    if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
        return std::make_pair(reinterpretCastOp.getResult(), reinterpretCastOp.getViewSource());
    }
    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(op)) {
        return std::make_pair(reshapeOp.getResult(), reshapeOp.getViewSource());
    }
    if (auto castOp = dyn_cast<memref::CastOp>(op)) {
        return std::make_pair(castOp.getResult(), castOp.getViewSource());
    }
    if (auto extractOp = dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
        return std::make_pair(extractOp.getBaseBuffer(), extractOp.getViewSource());
    }
    return std::nullopt;
}

std::optional<std::pair<Value, Value>> pto::getOperationAliasInfo(Operation* op)
{
    if (auto alias = getPTOOperationAliasInfo(op)) {
        return alias;
    }
    if (auto alias = getMemRefOperationAliasInfo(op)) {
        return alias;
    }
    if (auto extSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
        return std::make_pair(extSliceOp.getResult(), extSliceOp.getSource());
    }
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
        bool singleValueCast = castOp.getNumOperands() == 1 && castOp.getNumResults() == 1;
        if (singleValueCast) {
            return std::make_pair(castOp.getResult(0), castOp.getOperand(0));
        }
    }
    if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op)) {
        return std::make_pair(toMemrefOp.getResult(), toMemrefOp.getOperand());
    }
    if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
        return std::make_pair(toTensorOp.getResult(), toTensorOp.getOperand());
    }
    return std::nullopt;
}
