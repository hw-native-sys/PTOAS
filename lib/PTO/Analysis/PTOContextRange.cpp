// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOValueEvolutionAnalysis.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

PTOAnalysisResult<PTOFiniteRange> PTOValueEvolutionAnalysis::getRangeAt(Value value, Operation* context)
{
    auto contextFunction = dyn_cast_or_null<func::FuncOp>(context);
    if (!contextFunction && context) {
        contextFunction = context->getParentOfType<func::FuncOp>();
    }
    bool outsideFunction = !value || !context || contextFunction != func ||
                           value.getParentRegion()->getParentOfType<func::FuncOp>() != func;
    if (outsideFunction) {
        return PTOAnalysisResult<PTOFiniteRange>::unknown(PTOAnalysisUnknownReason::RangeUnavailable);
    }
    DominanceInfo dominance(func);
    bool available = value.getDefiningOp() == context || dominance.dominates(value, context);
    if (!available) {
        return PTOAnalysisResult<PTOFiniteRange>::unknown(PTOAnalysisUnknownReason::RangeUnavailable);
    }
    auto range = integer_range::getSignedRangeAt(value, context);
    if (!range) {
        return PTOAnalysisResult<PTOFiniteRange>::unknown(PTOAnalysisUnknownReason::RangeUnavailable);
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    return PTOAnalysisResult<PTOFiniteRange>::known(
        {APInt(width, range->lo, true), APInt(width, range->hi, true), false});
}

PTOAnalysisResult<bool> PTOValueEvolutionAnalysis::preservesValueAt(Operation* castOp, Operation* context)
{
    if (!castOp ||
        !isa<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp, arith::IndexCastUIOp>(castOp)) {
        return PTOAnalysisResult<bool>::unknown(PTOAnalysisUnknownReason::UnsupportedCast);
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(castOp->getResult(0).getType());
    if (!width || width > 64) {
        return PTOAnalysisResult<bool>::unknown(PTOAnalysisUnknownReason::TypeMismatch);
    }
    auto input = getRangeAt(castOp->getOperand(0), context);
    if (!input) {
        return PTOAnalysisResult<bool>::unknown(PTOAnalysisUnknownReason::RangeUnavailable);
    }
    int64_t lower = input.value->lowerInclusive.getSExtValue();
    int64_t upper = input.value->upperInclusive.getSExtValue();
    bool zeroExtend = isa<arith::ExtUIOp, arith::IndexCastUIOp>(castOp);
    __int128 limit = static_cast<__int128>(1) << (width - 1);
    bool notRepresentable = (zeroExtend && lower < 0) || lower < -limit || upper >= limit;
    if (notRepresentable) {
        return PTOAnalysisResult<bool>::unknown(PTOAnalysisUnknownReason::RangeUnavailable);
    }
    return PTOAnalysisResult<bool>::known(true);
}

namespace {
static bool isProvenNoSignedWrap(Value value)
{
    Operation* operation = value.getDefiningOp();
    auto overflow = dyn_cast_or_null<arith::ArithIntegerOverflowFlagsInterface>(operation);
    if (overflow && overflow.hasNoSignedWrap()) {
        return true;
    }
    bool arithmetic = isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation);
    if (!arithmetic) {
        return false;
    }
    auto lhs = integer_range::inferPointIntegerRange(operation->getOperand(0));
    auto rhs = integer_range::inferPointIntegerRange(operation->getOperand(1));
    bool unsupportedRange = !lhs || !rhs || !lhs->smin().isSignedIntN(64) || !lhs->smax().isSignedIntN(64) ||
                            !rhs->smin().isSignedIntN(64) || !rhs->smax().isSignedIntN(64);
    if (unsupportedRange) {
        return false;
    }
    __int128 lo = lhs->smin().getSExtValue(), hi = lhs->smax().getSExtValue();
    __int128 rlo = rhs->smin().getSExtValue(), rhi = rhs->smax().getSExtValue();
    if (isa<arith::AddIOp>(operation)) {
        lo += rlo;
        hi += rhi;
    } else if (isa<arith::SubIOp>(operation)) {
        lo -= rhi;
        hi -= rlo;
    } else {
        __int128 products[] = {lo*rlo, lo*rhi, hi*rlo, hi*rhi};
        lo = *std::min_element(std::begin(products), std::end(products));
        hi = *std::max_element(std::begin(products), std::end(products));
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    if (!width || width > 64) {
        return false;
    }
    __int128 limit = __int128{1} << (width - 1);
    return lo >= -limit && hi < limit;
}

/// Remove finite-width source boundaries only when range inference proves
/// that the source operation cannot wrap. Remaining source values are kept as
/// opaque atoms, so equal dynamic terms can cancel without reassociating
/// unproven arithmetic.
static PTOTypedExprRef expandProvenPointExpression(const PTOTypedExprRef& expression, unsigned depth = 0)
{
    if (!expression) {
        return {};
    }
    if (depth >= 32) {
        return expression->sourceValue ? makePTOOpaqueExpr(expression->sourceValue) : expression;
    }
    if (expression->kind == PTOTypedExpr::Kind::Constant || expression->kind == PTOTypedExpr::Kind::Opaque ||
        expression->kind == PTOTypedExpr::Kind::Cast) {
        return expression;
    }
    if (expression->sourceValue && !isProvenNoSignedWrap(expression->sourceValue)) {
        return makePTOOpaqueExpr(expression->sourceValue);
    }

    PTOTypedExprRef lhs = expandProvenPointExpression(expression->lhs, depth + 1);
    PTOTypedExprRef rhs = expandProvenPointExpression(expression->rhs, depth + 1);
    switch (expression->kind) {
        case PTOTypedExpr::Kind::Add:
            return makePTOAddExpr(lhs, rhs, expression->type);
        case PTOTypedExpr::Kind::Sub:
            return makePTOSubExpr(lhs, rhs, expression->type);
        case PTOTypedExpr::Kind::Mul:
            return makePTOMulExpr(lhs, rhs, expression->type);
        default:
            return expression;
    }
}

} // namespace

PTOTypedExprRef PTOValueEvolutionAnalysis::getConservativePointExpression(const PTOTypedExprRef& expression) const
{
    return expandProvenPointExpression(expression);
}
