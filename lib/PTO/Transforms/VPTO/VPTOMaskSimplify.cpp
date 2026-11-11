// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <optional>
#include <string>

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOMASKSIMPLIFY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kDecimalRadix = 10;

template <typename OpTy>
static bool isAllTrueMaskFrom(Value mask)
{
    auto op = mask.getDefiningOp<OpTy>();
    return op && op.getPattern() == "PAT_ALL";
}

static bool isAllTrueMask(Value mask)
{
    return isAllTrueMaskFrom<PsetB8Op>(mask) || isAllTrueMaskFrom<PsetB16Op>(mask) ||
           isAllTrueMaskFrom<PsetB32Op>(mask) || isAllTrueMaskFrom<PgeB8Op>(mask) ||
           isAllTrueMaskFrom<PgeB16Op>(mask) || isAllTrueMaskFrom<PgeB32Op>(mask);
}

// Number of predicate lanes carried by one physical mask part.
static int64_t getMaskLaneCount(Type type)
{
    auto maskType = dyn_cast<MaskType>(type);
    if (!maskType) {
        return 0;
    }
    if (maskType.isB8()) {
        return 256;
    }
    if (maskType.isB16()) {
        return 128;
    }
    if (maskType.isB32()) {
        return 64;
    }
    return 0;
}

// Active lane count of a static prefix predicate (PAT_ALL / PAT_ALLF / PAT_VL),
// or nullopt when the value is not a static prefix pattern.
static std::optional<int64_t> getStaticPrefixLanes(Value mask)
{
    Operation* patternOp = mask.getDefiningOp();
    bool isStaticPattern = patternOp && isa<PsetB8Op, PsetB16Op, PsetB32Op, PgeB8Op, PgeB16Op, PgeB32Op>(patternOp);
    if (!isStaticPattern) {
        return std::nullopt;
    }
    auto pattern = patternOp->getAttrOfType<StringAttr>("pattern");
    if (!pattern) {
        return std::nullopt;
    }
    StringRef name = pattern.getValue();
    if (name == "PAT_ALL") {
        return getMaskLaneCount(mask.getType());
    }
    if (name == "PAT_ALLF") {
        return 0;
    }
    StringRef digits = name;
    if (!digits.consume_front("PAT_VL")) {
        return std::nullopt;
    }
    if (digits.empty()) {
        return std::nullopt;
    }
    int64_t lanes = 0;
    if (digits.getAsInteger(kDecimalRadix, lanes)) {
        return std::nullopt;
    }
    return lanes;
}

// An all-false predicate is a static prefix with zero active lanes (PAT_ALLF).
static bool isAllFalseMask(Value mask)
{
    std::optional<int64_t> lanes = getStaticPrefixLanes(mask);
    return lanes && *lanes == 0;
}

// Replacing a gated mask op by one of its operands is only valid when that
// operand already carries the result granularity. The mask verifiers
// (verifyBinaryMaskOp and PnotOp::verify) only require MaskType, so a
// mixed-granularity op can reach the pass; folding it blindly would emit a
// value whose type no longer matches the users. The static prefix folds need
// the same precondition: PAT_VL<k> lane counts are only comparable within one
// mask granularity.
static bool canReplaceWith(Value candidate, Value result) { return candidate.getType() == result.getType(); }

static bool isSupportedPrefixLanes(int64_t lanes)
{
    switch (lanes) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
            return true;
        default:
            return false;
    }
}

static Value createPatternMask(Location loc, Type type, StringRef pattern, PatternRewriter& rewriter)
{
    auto maskType = dyn_cast<MaskType>(type);
    if (!maskType) {
        return {};
    }
    StringAttr patternAttr = rewriter.getStringAttr(pattern);
    if (maskType.isB8()) {
        return rewriter.create<PsetB8Op>(loc, type, patternAttr).getResult();
    }
    if (maskType.isB16()) {
        return rewriter.create<PsetB16Op>(loc, type, patternAttr).getResult();
    }
    if (maskType.isB32()) {
        return rewriter.create<PsetB32Op>(loc, type, patternAttr).getResult();
    }
    return {};
}

static Value createPrefixMaskForLanes(Location loc, Type type, int64_t lanes, PatternRewriter& rewriter)
{
    int64_t laneCount = getMaskLaneCount(type);
    if (laneCount <= 0) {
        return {};
    }
    if (lanes <= 0) {
        return createPatternMask(loc, type, "PAT_ALLF", rewriter);
    }
    if (lanes >= laneCount) {
        return createPatternMask(loc, type, "PAT_ALL", rewriter);
    }
    if (!isSupportedPrefixLanes(lanes)) {
        return {};
    }
    std::string pattern = "PAT_VL" + std::to_string(lanes);
    return createPatternMask(loc, type, pattern, rewriter);
}

// Replace a matched gated mask op by the all-false predicate of its own result
// granularity.
template <typename OpTy>
static LogicalResult replaceWithAllFalse(OpTy op, PatternRewriter& rewriter)
{
    Value allFalse = createPatternMask(op.getLoc(), op.getResult().getType(), "PAT_ALLF", rewriter);
    if (!allFalse) {
        return failure();
    }
    rewriter.replaceOp(op, allFalse);
    return success();
}

template <typename OpTy>
struct SimplifyAllTruePredicateReorder : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override
    {
        if (!isAllTrueMask(op.getLhs()) || !isAllTrueMask(op.getRhs())) {
            return failure();
        }

        rewriter.replaceOp(op, {op.getLhs(), op.getRhs()});
        return success();
    }
};

// Folds a deinterleave of static prefix predicates back into the prefix masks it
// produces.  pdintlv(a, b) yields low = [even(a); even(b)] and
// high = [odd(a); odd(b)], each image occupying one half of the result; the two
// halves form one prefix lane run only when the first half is full or the second
// half is empty.  The sub-chunk form pdintlv(m, PAT_ALLF) is the useful case: it
// produces even(m) / odd(m), which are plain prefixes.
template <typename OpTy>
struct SimplifyStaticPredicateDeinterleave : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override
    {
        // All-true pairs keep the existing identity rewrite (and their uses).
        if (isAllTrueMask(op.getLhs()) && isAllTrueMask(op.getRhs())) {
            return failure();
        }
        std::optional<int64_t> lhsLanes = getStaticPrefixLanes(op.getLhs());
        std::optional<int64_t> rhsLanes = getStaticPrefixLanes(op.getRhs());
        if (!lhsLanes || !rhsLanes) {
            return failure();
        }
        int64_t laneCount = getMaskLaneCount(op.getLow().getType());
        if (laneCount <= 0) {
            return failure();
        }
        int64_t half = laneCount / 2;
        int64_t lowEven = (*lhsLanes + 1) / 2;
        int64_t lowOdd = *lhsLanes / 2;
        int64_t highEven = (*rhsLanes + 1) / 2;
        int64_t highOdd = *rhsLanes / 2;
        bool lowIsPrefix = lowEven == half || highEven == 0;
        bool highIsPrefix = lowOdd == half || highOdd == 0;
        if (!lowIsPrefix || !highIsPrefix) {
            return failure();
        }
        Value low = createPrefixMaskForLanes(op.getLoc(), op.getLow().getType(), lowEven + highEven, rewriter);
        Value high = createPrefixMaskForLanes(op.getLoc(), op.getHigh().getType(), lowOdd + highOdd, rewriter);
        if (!low || !high) {
            return failure();
        }
        rewriter.replaceOp(op, {low, high});
        return success();
    }
};

struct SimplifyVselAllTrueMask : OpRewritePattern<VselOp> {
    using OpRewritePattern<VselOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(VselOp op, PatternRewriter& rewriter) const override
    {
        // vsel(src0, src1, mask) selects src0 for all-true masks.
        if (!isAllTrueMask(op.getMask())) {
            return failure();
        }

        rewriter.replaceOp(op, op.getSrc0());
        return success();
    }
};

// pand(src0, src1, mask) = mask ? (src0 & src1) : 0. An all-true governing mask
// together with an all-true operand leaves only the other operand.
struct SimplifyPandAllTrueOperand : OpRewritePattern<PandOp> {
    using OpRewritePattern<PandOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PandOp op, PatternRewriter& rewriter) const override
    {
        if (!isAllTrueMask(op.getMask())) {
            return failure();
        }
        if (isAllTrueMask(op.getSrc0()) && canReplaceWith(op.getSrc1(), op.getResult())) {
            rewriter.replaceOp(op, op.getSrc1());
            return success();
        }
        if (isAllTrueMask(op.getSrc1()) && canReplaceWith(op.getSrc0(), op.getResult())) {
            rewriter.replaceOp(op, op.getSrc0());
            return success();
        }
        return failure();
    }
};

// (a XOR b) AND m can use PXOR's governing predicate directly. Keep
// both original governing masks unless they are known identities.
struct FuseXorAndPredicate : OpRewritePattern<PandOp> {
    using OpRewritePattern<PandOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PandOp op, PatternRewriter& rewriter) const override
    {
        // Predicate ops accept mixed granularities. Do not move a predicate
        // across either op unless every operand and result has one mask type.
        Type maskType = op.getResult().getType();
        if (op.getSrc0().getType() != maskType || op.getSrc1().getType() != maskType ||
            op.getMask().getType() != maskType || !isAllTrueMask(op.getMask())) {
            return failure();
        }
        for (unsigned side = 0; side < 2; ++side) {
            Value value = side == 0 ? op.getSrc0() : op.getSrc1();
            Value predicate = side == 0 ? op.getSrc1() : op.getSrc0();
            auto xorOp = value.getDefiningOp<PxorOp>();
            if (!xorOp || !xorOp.getResult().hasOneUse() || xorOp.getResult().getType() != maskType ||
                xorOp.getSrc0().getType() != maskType || xorOp.getSrc1().getType() != maskType ||
                xorOp.getMask().getType() != maskType || !isAllTrueMask(xorOp.getMask())) {
                continue;
            }
            rewriter.replaceOpWithNewOp<PxorOp>(
                op, op.getResult().getType(), xorOp.getSrc0(), xorOp.getSrc1(), predicate);
            return success();
        }
        return failure();
    }
};

// por(src0, src1, mask) = mask ? (src0 | src1) : 0. An all-true operand already
// saturates the result under an all-true governing mask.
struct SimplifyPorAllTrueOperand : OpRewritePattern<PorOp> {
    using OpRewritePattern<PorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PorOp op, PatternRewriter& rewriter) const override
    {
        if (!isAllTrueMask(op.getMask())) {
            return failure();
        }
        if (isAllTrueMask(op.getSrc0()) && canReplaceWith(op.getSrc0(), op.getResult())) {
            rewriter.replaceOp(op, op.getSrc0());
            return success();
        }
        if (isAllTrueMask(op.getSrc1()) && canReplaceWith(op.getSrc1(), op.getResult())) {
            rewriter.replaceOp(op, op.getSrc1());
            return success();
        }
        return failure();
    }
};

// pnot(input, mask) = mask ? ~input : 0. An all-true input is all-false under
// any governing mask, and an all-false input is the governing mask itself.
struct SimplifyPnotAllTrueInput : OpRewritePattern<PnotOp> {
    using OpRewritePattern<PnotOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PnotOp op, PatternRewriter& rewriter) const override
    {
        if (!isAllTrueMask(op.getInput())) {
            return failure();
        }
        return replaceWithAllFalse(op, rewriter);
    }
};

struct SimplifyPnotAllFalseInput : OpRewritePattern<PnotOp> {
    using OpRewritePattern<PnotOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PnotOp op, PatternRewriter& rewriter) const override
    {
        if (!isAllFalseMask(op.getInput()) || !canReplaceWith(op.getMask(), op.getResult())) {
            return failure();
        }
        rewriter.replaceOp(op, op.getMask());
        return success();
    }
};

// pand(x, x, all) = x and por(x, x, all) = x.
template <typename OpTy>
struct SimplifyIdempotentPredicate : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override
    {
        if (!isAllTrueMask(op.getMask()) || op.getSrc0() != op.getSrc1() ||
            !canReplaceWith(op.getSrc0(), op.getResult())) {
            return failure();
        }
        rewriter.replaceOp(op, op.getSrc0());
        return success();
    }
};

// pxor(x, x, mask) = 0 for any governing mask.
struct SimplifyPxorSameOperand : OpRewritePattern<PxorOp> {
    using OpRewritePattern<PxorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PxorOp op, PatternRewriter& rewriter) const override
    {
        if (op.getSrc0() != op.getSrc1()) {
            return failure();
        }
        return replaceWithAllFalse(op, rewriter);
    }
};

// A gated predicate op whose governing mask is all-false is all-false.
template <typename OpTy>
struct SimplifyAllFalseGoverningMask : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override
    {
        if (!isAllFalseMask(op.getMask())) {
            return failure();
        }
        return replaceWithAllFalse(op, rewriter);
    }
};

// pand(src0, src1, mask) = mask ? (src0 & src1) : 0, so an all-false operand
// makes the result all-false under any governing mask.
struct SimplifyPandAllFalseOperand : OpRewritePattern<PandOp> {
    using OpRewritePattern<PandOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PandOp op, PatternRewriter& rewriter) const override
    {
        if (!isAllFalseMask(op.getSrc0()) && !isAllFalseMask(op.getSrc1())) {
            return failure();
        }
        return replaceWithAllFalse(op, rewriter);
    }
};

// Static prefix predicates compose through the gated binary ops: the AND keeps
// lanes below every bound, the OR keeps lanes below the mask and the wider
// operand.
struct SimplifyStaticPrefixAnd : OpRewritePattern<PandOp> {
    using OpRewritePattern<PandOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PandOp op, PatternRewriter& rewriter) const override
    {
        std::optional<int64_t> lhs = getStaticPrefixLanes(op.getSrc0());
        std::optional<int64_t> rhs = getStaticPrefixLanes(op.getSrc1());
        std::optional<int64_t> mask = getStaticPrefixLanes(op.getMask());
        if (!lhs || !rhs || !mask || !canReplaceWith(op.getSrc0(), op.getResult()) ||
            !canReplaceWith(op.getSrc1(), op.getResult()) || !canReplaceWith(op.getMask(), op.getResult())) {
            return failure();
        }
        Value result =
            createPrefixMaskForLanes(op.getLoc(), op.getResult().getType(), std::min({*lhs, *rhs, *mask}), rewriter);
        if (!result) {
            return failure();
        }
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct SimplifyStaticPrefixOr : OpRewritePattern<PorOp> {
    using OpRewritePattern<PorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PorOp op, PatternRewriter& rewriter) const override
    {
        std::optional<int64_t> lhs = getStaticPrefixLanes(op.getSrc0());
        std::optional<int64_t> rhs = getStaticPrefixLanes(op.getSrc1());
        std::optional<int64_t> mask = getStaticPrefixLanes(op.getMask());
        if (!lhs || !rhs || !mask || !canReplaceWith(op.getSrc0(), op.getResult()) ||
            !canReplaceWith(op.getSrc1(), op.getResult()) || !canReplaceWith(op.getMask(), op.getResult())) {
            return failure();
        }
        Value result = createPrefixMaskForLanes(
            op.getLoc(), op.getResult().getType(), std::min(*mask, std::max(*lhs, *rhs)), rewriter);
        if (!result) {
            return failure();
        }
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct VPTOMaskSimplifyPass : public pto::impl::VPTOMaskSimplifyBase<VPTOMaskSimplifyPass> {
    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<
            SimplifyVselAllTrueMask, SimplifyAllFalseGoverningMask<PandOp>, SimplifyAllFalseGoverningMask<PorOp>,
            SimplifyAllFalseGoverningMask<PxorOp>, SimplifyAllFalseGoverningMask<PnotOp>, SimplifyPandAllFalseOperand,
            FuseXorAndPredicate, SimplifyStaticPrefixAnd, SimplifyStaticPrefixOr, SimplifyPandAllTrueOperand, SimplifyPorAllTrueOperand,
            SimplifyPnotAllTrueInput, SimplifyPnotAllFalseInput, SimplifyIdempotentPredicate<PandOp>,
            SimplifyIdempotentPredicate<PorOp>, SimplifyPxorSameOperand, SimplifyAllTruePredicateReorder<PintlvB8Op>,
            SimplifyAllTruePredicateReorder<PintlvB16Op>, SimplifyAllTruePredicateReorder<PintlvB32Op>,
            SimplifyAllTruePredicateReorder<PdintlvB8Op>, SimplifyAllTruePredicateReorder<PdintlvB16Op>,
            SimplifyAllTruePredicateReorder<PdintlvB32Op>, SimplifyStaticPredicateDeinterleave<PdintlvB8Op>,
            SimplifyStaticPredicateDeinterleave<PdintlvB16Op>, SimplifyStaticPredicateDeinterleave<PdintlvB32Op>>(
            &getContext());

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

namespace mlir::pto {

std::unique_ptr<Pass> createVPTOMaskSimplifyPass() { return std::make_unique<VPTOMaskSimplifyPass>(); }

} // namespace mlir::pto
