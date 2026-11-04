// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Preserve descriptor identity and capture time until final SCF conversion.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::pto {
#define GEN_PASS_DEF_PTOMATERIALIZETILEIFRESULTS
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir::pto

using namespace mlir;

namespace {
bool needsCapture(Value value)
{
    return isa<pto::TileBufType>(value.getType()) && !value.use_empty();
}

void captureYield(scf::YieldOp yield, unsigned index, IRRewriter &rewriter)
{
    Value source = yield.getOperand(index);
    auto copy = source.getDefiningOp<pto::CopyTileDescriptorOp>();
    if (copy && copy->hasOneUse() && copy->getBlock() == yield->getBlock()) {
        return;
    }
    rewriter.setInsertionPoint(yield);
    auto captured = rewriter.create<pto::CopyTileDescriptorOp>(yield.getLoc(), source.getType(), source);
    yield->setOperand(index, captured.getResult());
}

void captureResults(scf::IfOp selection, IRRewriter &rewriter)
{
    if (!selection.elseBlock()) {
        return;
    }
    for (OpResult result : selection.getResults()) {
        if (!needsCapture(result)) {
            continue;
        }
        unsigned index = result.getResultNumber();
        captureYield(cast<scf::YieldOp>(selection.thenBlock()->getTerminator()), index, rewriter);
        captureYield(cast<scf::YieldOp>(selection.elseBlock()->getTerminator()), index, rewriter);
    }
}

void captureSelect(arith::SelectOp select, IRRewriter &rewriter)
{
    if (!needsCapture(select.getResult())) {
        return;
    }
    Type type = select.getType();
    rewriter.setInsertionPoint(select);
    auto selection = rewriter.create<scf::IfOp>(
        select.getLoc(), TypeRange{type}, select.getCondition(), /*withElseRegion=*/true);
    rewriter.setInsertionPointToEnd(selection.thenBlock());
    rewriter.create<scf::YieldOp>(select.getLoc(), select.getTrueValue());
    rewriter.setInsertionPointToEnd(selection.elseBlock());
    rewriter.create<scf::YieldOp>(select.getLoc(), select.getFalseValue());
    rewriter.replaceOp(select, selection.getResults());
}

struct PTOMaterializeTileIfResultsPass final
    : pto::impl::PTOMaterializeTileIfResultsBase<PTOMaterializeTileIfResultsPass> {
    void runOnOperation() override
    {
        // Tile is an opaque mutable descriptor in the C++ ABI. Preserve
        // value selection at the yield for every Tile, including static
        // valid shapes and library-internal metadata/address updates.
        IRRewriter rewriter(&getContext());
        getOperation().walk([&](arith::SelectOp select) {
            captureSelect(select, rewriter);
        });
        getOperation().walk<WalkOrder::PostOrder>([&](scf::IfOp selection) {
            captureResults(selection, rewriter);
        });
    }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMaterializeTileIfResultsPass()
{
    return std::make_unique<PTOMaterializeTileIfResultsPass>();
}
