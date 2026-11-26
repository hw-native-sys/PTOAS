// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Static local/shared f32 tile types and two-dimensional GM views.
#include "PTO/IR/Linx/LinxDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "PTO/IR/Linx/LinxDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PTO/IR/Linx/LinxOps.cpp.inc"

namespace mlir::pto::linx {

void PTODialect::initialize()
{
    initializeTypes();
    addOperations<
#define GET_OP_LIST
#include "PTO/IR/Linx/LinxOps.cpp.inc"
        >();
}

static LogicalResult verifyTransfer(Operation* op, Type tileType, Type gmType)
{
    auto tile = dyn_cast<TileBufType>(tileType);
    auto gm = dyn_cast<GMViewType>(gmType);
    if (!tile || !gm || tile.getRows() != gm.getRows() || tile.getCols() != gm.getCols()) {
        return op->emitOpError("requires matching Linx tile and GM view shapes");
    }
    return success();
}
LogicalResult TLoadOp::verify() { return verifyTransfer(*this, getDst().getType(), getSrc().getType()); }
LogicalResult TStoreOp::verify() { return verifyTransfer(*this, getSrc().getType(), getDst().getType()); }
LogicalResult TAddOp::verify()
{
    Type type = getDst().getType();
    if (!isa<TileBufType>(type) || getLhs().getType() != type || getRhs().getType() != type) {
        return emitOpError("requires identical Linx tile types for inputs and output");
    }
    if (getDst() == getLhs() || getDst() == getRhs()) {
        return emitOpError("in-place TADD is not supported by this Linx profile");
    }
    return success();
}
LogicalResult TMatmulOp::verify()
{
    auto lhs = cast<TileBufType>(getLhs().getType());
    auto rhs = cast<TileBufType>(getRhs().getType());
    auto dst = cast<TileBufType>(getDst().getType());
    if (lhs.getLocation() != TileLocation::Local || rhs.getLocation() != TileLocation::Local ||
        dst.getLocation() != TileLocation::Local) {
        return emitOpError("currently requires local matrix operands and destination");
    }
    if ((lhs.getLayout() != TileLayout::M16 && lhs.getLayout() != TileLayout::M32) ||
        rhs.getLayout() != TileLayout::N8 || dst.getLayout() != lhs.getLayout()) {
        return emitOpError("requires lhs/dst with matching m16 or m32 layout and rhs with n8 layout");
    }
    if (lhs.getCols() != rhs.getRows() || dst.getRows() != lhs.getRows() || dst.getCols() != rhs.getCols()) {
        return emitOpError("requires lhs MxK, rhs KxN and dst MxN shapes");
    }
    if (getDst() == getLhs() || getDst() == getRhs()) {
        return emitOpError("destination must not alias either matrix input");
    }
    return success();
}
static LogicalResult verifyLocalTiles(Operation* op)
{
    for (Value operand : op->getOperands()) {
        if (cast<TileBufType>(operand.getType()).getLocation() != TileLocation::Local) {
            return op->emitOpError("requires local tiles");
        }
    }
    Value dst = op->getOperands().back();
    for (Value input : op->getOperands().drop_back()) {
        if (input == dst) {
            return op->emitOpError("in-place operation is not supported by this Linx profile");
        }
    }
    return success();
}

static LogicalResult verifyRowPair(Operation* op, Type fullType, Type rowType)
{
    auto full = cast<TileBufType>(fullType);
    auto row = cast<TileBufType>(rowType);
    if (full.getRows() != row.getRows() || row.getCols() != 1 || full.getLayout() != row.getLayout()) {
        return op->emitOpError("requires matching layouts and rows, with a one-column row tile");
    }
    return verifyLocalTiles(op);
}

LogicalResult TExpOp::verify()
{
    if (getSrc().getType() != getDst().getType()) {
        return emitOpError("requires identical input and output tile types");
    }
    return verifyLocalTiles(*this);
}
LogicalResult TRowMaxOp::verify() { return verifyRowPair(*this, getSrc().getType(), getDst().getType()); }
LogicalResult TRowSumOp::verify() { return verifyRowPair(*this, getSrc().getType(), getDst().getType()); }

static LogicalResult verifyRowExpand(Operation* op, Value src, Value row, Value dst)
{
    if (src.getType() != dst.getType()) {
        return op->emitOpError("requires identical full input and output tile types");
    }
    return verifyRowPair(op, dst.getType(), row.getType());
}
LogicalResult TRowExpandSubOp::verify() { return verifyRowExpand(*this, getSrc(), getRow(), getDst()); }
LogicalResult TRowExpandDivOp::verify() { return verifyRowExpand(*this, getSrc(), getRow(), getDst()); }
} // namespace mlir::pto::linx
