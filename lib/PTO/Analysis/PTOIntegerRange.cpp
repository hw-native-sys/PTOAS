// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOIntegerRange.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::pto;
using mlir::presburger::BoundType;
using mlir::presburger::IntegerPolyhedron;

namespace mlir::pto::integer_range {

// Try to evaluate `v` as a constant int64.
std::optional<int64_t> getConstantInt(Value v)
{
    APInt c;
    bool isConstant = matchPattern(v, m_ConstantInt(&c));
    bool fitsInt64 = isConstant && c.isSignedIntN(64);
    if (isConstant && fitsInt64) {
        return c.getSExtValue();
    }
    return std::nullopt;
}

static Value stripSameWidthCast(Value value)
{
    auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
    bool sameInteger = cast && cast.getInputs().size() == 1 && isa<IntegerType>(value.getType()) &&
                       isa<IntegerType>(cast.getInputs().front().getType()) &&
                       ConstantIntRanges::getStorageBitwidth(value.getType()) ==
                           ConstantIntRanges::getStorageBitwidth(cast.getInputs().front().getType());
    if (sameInteger) {
        return cast.getInputs().front();
    }
    return value;
}

static std::optional<AddressRange> recurrenceDelta(Value value, Value argument, unsigned depth)
{
    value = stripSameWidthCast(value);
    if (value == stripSameWidthCast(argument)) {
        return AddressRange{0, 0};
    }
    if (depth >= 8) {
        return std::nullopt;
    }
    if (auto branch = value.getDefiningOp<scf::IfOp>()) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        if (branch.getElseRegion().empty()) {
            return std::nullopt;
        }
        auto yes = recurrenceDelta(branch.thenYield().getOperand(index), argument, depth + 1);
        auto no = recurrenceDelta(branch.elseYield().getOperand(index), argument, depth + 1);
        if (yes && no) {
            return AddressRange{std::min(yes->lo, no->lo), std::max(yes->hi, no->hi)};
        }
    }
    auto add = value.getDefiningOp<arith::AddIOp>();
    bool advancesArgument = add && stripSameWidthCast(add.getLhs()) == stripSameWidthCast(argument);
    if (advancesArgument) {
        if (auto step = getConstantInt(add.getRhs())) {
            return AddressRange{*step, *step};
        }
    }
    return std::nullopt;
}

static std::optional<ConstantIntRanges> inferIterArgumentRange(BlockArgument arg, unsigned width)
{
    auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
    bool unsupported = !loop || !arg.getArgNumber() || width > 64;
    if (unsupported) {
        return std::nullopt;
    }
    auto trip = getLoopEnvelope(loop).maxTripCount();
    unsigned index = arg.getArgNumber() - 1;
    auto init = getConstantInt(stripSameWidthCast(loop.getInits()[index]));
    auto delta = recurrenceDelta(loop.getBody()->getTerminator()->getOperand(index), arg, 0);
    if (!trip || !init || !delta) {
        return std::nullopt;
    }
    __int128 lo = static_cast<__int128>(*init) + static_cast<__int128>(*trip) * std::min<int64_t>(0, delta->lo);
    __int128 hi = static_cast<__int128>(*init) + static_cast<__int128>(*trip) * std::max<int64_t>(0, delta->hi);
    __int128 limit = static_cast<__int128>(1) << (width - 1);
    if (lo < -limit || hi >= limit) {
        return std::nullopt;
    }
    return ConstantIntRanges::fromSigned(APInt(width, int64_t(lo), true), APInt(width, int64_t(hi), true));
}

static std::optional<ConstantIntRanges> inferBlockArgumentRange(BlockArgument arg, unsigned width)
{
    if (auto recurrence = inferIterArgumentRange(arg, width)) {
        return *recurrence;
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
    if (loop && arg == loop.getInductionVar()) {
        auto bounds = getLoopEnvelope(loop);
        auto trip = bounds.maxTripCount();
        if (trip && *trip > 0) {
            __int128 last = static_cast<__int128>(bounds.lb) + static_cast<__int128>(*trip - 1) * bounds.step;
            return ConstantIntRanges::fromSigned(APInt(width, bounds.lb, true), APInt(width, int64_t(last), true));
        }
    }
    return std::nullopt;
}

static std::optional<ConstantIntRanges> inferIfResultRange(Value value, unsigned depth)
{
    auto branch = value.getDefiningOp<scf::IfOp>();
    if (!branch || branch.getElseRegion().empty()) {
        return std::nullopt;
    }
    unsigned index = cast<OpResult>(value).getResultNumber();
    auto yes = inferIntegerRange(branch.thenYield().getOperand(index), depth + 1);
    auto no = inferIntegerRange(branch.elseYield().getOperand(index), depth + 1);
    return yes.rangeUnion(no);
}

// Infer a conservative range through side-effect-free integer expressions.
// Unknown leaves and a depth limit widen to the full range, never to a guess.
ConstantIntRanges inferIntegerRange(Value value, unsigned depth)
{
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    if (depth >= 16) {
        return ConstantIntRanges::maxRange(width);
    }
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) {
        return ConstantIntRanges::constant(constant);
    }
    auto range = ConstantIntRanges::maxRange(width);
    if (auto arg = dyn_cast<BlockArgument>(value)) {
        if (auto inferred = inferBlockArgumentRange(arg, width)) {
            return *inferred;
        }
    }
    Value storage = stripSameWidthCast(value);
    if (storage != value) {
        return inferIntegerRange(storage, depth + 1);
    }
    Operation* op = value.getDefiningOp();
    if (auto branchRange = inferIfResultRange(value, depth)) {
        return *branchRange;
    }
    bool hasRegions = op && op->getNumRegions() != 0;
    if (!op || depth == 16 || hasRegions) {
        return range;
    }
    auto interface = dyn_cast<InferIntRangeInterface>(op);
    if (!interface) {
        return range;
    }
    SmallVector<ConstantIntRanges> operands;
    for (Value operand : op->getOperands()) {
        if (!ConstantIntRanges::getStorageBitwidth(operand.getType())) {
            return range;
        }
        operands.push_back(inferIntegerRange(operand, depth + 1));
    }
    interface.inferResultRanges(operands, [&](Value result, const ConstantIntRanges& inferred) {
        if (result == value) {
            range = inferred;
        }
    });
    return range;
}

// Constant lower/upper/step for an scf.for, else nullopt (dynamic).
PTOLoopDomain getConstBounds(scf::ForOp forOp)
{
    PTOLoopDomain b;
    auto lb = getConstantInt(forOp.getLowerBound());
    auto ub = getConstantInt(forOp.getUpperBound());
    auto step = getConstantInt(forOp.getStep());
    if (lb && ub && step && *step != 0) {
        b.lb = *lb;
        b.ub = *ub;
        b.step = *step;
        b.valid = true;
        b.exact = true;
    }
    return b;
}

// Opaque leaves retain their reliable value range; expanded address arithmetic
// must separately be certified not to wrap.
std::optional<AddressRange> inferredLeafRange(Value value)
{
    // The sign-corrected remainder sum is a leaf whose correlated bound is
    // tighter than independent inference over its parts.
    if (auto remBound = matchSignCorrectedRemSum(value)) {
        return AddressRange{0, *remBound};
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    if (!width || width > 64) {
        return std::nullopt;
    }
    ConstantIntRanges inferred = inferIntegerRange(value);
    return AddressRange{inferred.smin().getSExtValue(), inferred.smax().getSExtValue()};
}

// Guard-constrained affine forms forward only value-preserving arithmetic.
// Recover a truncated clamp only if its entire guarded range is representable.
static bool int64Representable(__int128 v) { return v >= INT64_MIN && v <= INT64_MAX; }

// A Presburger constant bound is only usable when it fits int64.
static std::optional<int64_t> checkedConstantBound(const IntegerPolyhedron& poly, BoundType type, unsigned pos)
{
    auto bound = poly.getConstantBound(type, pos);
    if (!bound) {
        return std::nullopt;
    }
    if (*bound < INT64_MIN || *bound > INT64_MAX) {
        return std::nullopt;
    }
    return static_cast<int64_t>(*bound);
}

struct CertifiedAffineForm {
    SmallVector<std::pair<Value, int64_t>, 4> terms;
    int64_t constant = 0;
    int64_t lo = 0;
    int64_t hi = 0;

    bool addTerm(Value leaf, __int128 coefficient)
    {
        if (coefficient == 0) {
            return true;
        }
        if (!int64Representable(coefficient)) {
            return false;
        }
        for (auto& [value, coeff] : terms) {
            if (value == leaf) {
                __int128 merged = static_cast<__int128>(coeff) + coefficient;
                if (!int64Representable(merged)) {
                    return false;
                }
                coeff = static_cast<int64_t>(merged);
                return true;
            }
        }
        terms.push_back({leaf, static_cast<int64_t>(coefficient)});
        return true;
    }
};

// Leaf ranges for the certified extraction: the refined map (polyhedron
// bounds under guards) takes precedence; everything else uses inference.
struct CertifiedLeafRanges {
    const DenseMap<Value, AddressRange>* refined = nullptr;

    std::optional<AddressRange> lookup(Value value) const
    {
        if (refined) {
            auto it = refined->find(value);
            if (it != refined->end()) {
                return it->second;
            }
        }
        return inferredLeafRange(value);
    }
};

static std::optional<unsigned> storageBitwidth(Type type)
{
    if (!type.isIntOrIndex()) {
        return std::nullopt;
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(type);
    if (!width || width > 64) {
        return std::nullopt;
    }
    return width;
}

// Certified affine form of `value`. Fails only when the top-level value ends
// up an opaque leaf without a representable range; inner failures just stop
// the forwarding there, which loses precision but never faithfulness.
static bool formFitsWidth(const CertifiedAffineForm& form, unsigned width)
{
    __int128 limit = static_cast<__int128>(1) << (width - 1);
    return form.lo >= -limit && form.hi < limit;
}

static std::optional<CertifiedAffineForm> combineCertifiedForms(
    const CertifiedAffineForm& lhs, const CertifiedAffineForm& rhs, unsigned width, bool subtract)
{
    int64_t sign = subtract ? -1 : 1;
    __int128 constant = static_cast<__int128>(lhs.constant) + static_cast<__int128>(sign) * rhs.constant;
    __int128 lo = static_cast<__int128>(lhs.lo) + static_cast<__int128>(sign) * (subtract ? rhs.hi : rhs.lo);
    __int128 hi = static_cast<__int128>(lhs.hi) + static_cast<__int128>(sign) * (subtract ? rhs.lo : rhs.hi);
    bool fits = int64Representable(constant) && int64Representable(lo) && int64Representable(hi);
    if (!fits) {
        return std::nullopt;
    }
    CertifiedAffineForm form = lhs;
    for (auto& [leaf, coeff] : rhs.terms) {
        bool termAdded = form.addTerm(leaf, static_cast<__int128>(sign) * coeff);
        if (!termAdded) {
            return std::nullopt;
        }
    }
    form.constant = static_cast<int64_t>(constant);
    form.lo = static_cast<int64_t>(lo);
    form.hi = static_cast<int64_t>(hi);
    return formFitsWidth(form, width) ? std::optional<CertifiedAffineForm>(form) : std::nullopt;
}

static std::optional<CertifiedAffineForm> certifiedAffineForm(
    Value value, const CertifiedLeafRanges& ranges, unsigned depth);

static std::optional<CertifiedAffineForm> scaleCertifiedForm(
    const CertifiedAffineForm& inner, int64_t factor, unsigned width)
{
    __int128 constant = static_cast<__int128>(inner.constant) * factor;
    __int128 lo = static_cast<__int128>(inner.lo) * factor;
    __int128 hi = static_cast<__int128>(inner.hi) * factor;
    bool fits = int64Representable(constant) && int64Representable(lo) && int64Representable(hi);
    if (!fits) {
        return std::nullopt;
    }
    CertifiedAffineForm form;
    for (auto& [leaf, coeff] : inner.terms) {
        bool termAdded = form.addTerm(leaf, static_cast<__int128>(coeff) * factor);
        if (!termAdded) {
            return std::nullopt;
        }
    }
    form.constant = static_cast<int64_t>(constant);
    form.lo = static_cast<int64_t>(std::min(lo, hi));
    form.hi = static_cast<int64_t>(std::max(lo, hi));
    return formFitsWidth(form, width) ? std::optional<CertifiedAffineForm>(form) : std::nullopt;
}

static std::optional<CertifiedAffineForm> certifiedSum(Operation* op, const CertifiedLeafRanges& ranges, unsigned depth)
{
    auto lhs = certifiedAffineForm(op->getOperand(0), ranges, depth + 1);
    auto rhs = certifiedAffineForm(op->getOperand(1), ranges, depth + 1);
    auto width = storageBitwidth(op->getResult(0).getType());
    if (!lhs || !rhs || !width) {
        return std::nullopt;
    }
    return combineCertifiedForms(*lhs, *rhs, *width, isa<arith::SubIOp>(op));
}

static std::optional<CertifiedAffineForm> certifiedProduct(
    arith::MulIOp op, const CertifiedLeafRanges& ranges, unsigned depth)
{
    std::optional<int64_t> factor = getConstantInt(op.getLhs());
    Value operand = op.getRhs();
    if (!factor) {
        factor = getConstantInt(op.getRhs());
        operand = op.getLhs();
    }
    if (!factor) {
        return std::nullopt;
    }
    auto inner = certifiedAffineForm(operand, ranges, depth + 1);
    auto width = storageBitwidth(op.getType());
    if (!inner || !width) {
        return std::nullopt;
    }
    return scaleCertifiedForm(*inner, *factor, *width);
}

static std::optional<CertifiedAffineForm> certifiedCast(
    Operation* op, const CertifiedLeafRanges& ranges, unsigned depth)
{
    Value operand = op->getOperand(0);
    auto inner = certifiedAffineForm(operand, ranges, depth + 1);
    if (!inner) {
        return std::nullopt;
    }
    if (isa<arith::ExtSIOp>(op)) {
        return inner; // Sign extension is value-exact.
    }
    if (isa<arith::ExtUIOp>(op)) {
        return inner->lo >= 0 ? inner : std::nullopt;
    }
    auto outWidth = storageBitwidth(op->getResult(0).getType());
    if (isa<arith::TruncIOp>(op)) {
        return outWidth && formFitsWidth(*inner, *outWidth) ? inner : std::nullopt;
    }
    auto inWidth = storageBitwidth(operand.getType());
    if (!inWidth || !outWidth) {
        return std::nullopt;
    }
    if (isa<arith::IndexCastOp>(op)) {
        return *inWidth <= *outWidth || formFitsWidth(*inner, *outWidth) ? inner : std::nullopt;
    }
    // Unsigned index casts are forwarded only when widening non-negative values.
    return *inWidth <= *outWidth && inner->lo >= 0 ? inner : std::nullopt;
}

static std::optional<CertifiedAffineForm> certifiedOperationForm(
    Operation* op, const CertifiedLeafRanges& ranges, unsigned depth)
{
    if (isa<arith::AddIOp, arith::SubIOp>(op)) {
        return certifiedSum(op, ranges, depth);
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        return certifiedProduct(mulOp, ranges, depth);
    }
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
        return certifiedCast(op, ranges, depth);
    }
    return std::nullopt;
}

static std::optional<CertifiedAffineForm> certifiedAffineForm(
    Value value, const CertifiedLeafRanges& ranges, unsigned depth)
{
    if (depth < 16) {
        if (auto constant = getConstantInt(value)) {
            CertifiedAffineForm form;
            form.constant = *constant;
            form.lo = *constant;
            form.hi = *constant;
            return form;
        }
        if (Operation* op = value.getDefiningOp()) {
            if (auto form = certifiedOperationForm(op, ranges, depth)) {
                return form;
            }
        }
    }
    // Opaque leaf: failed certification loses precision, never faithfulness.
    auto range = ranges.lookup(value);
    if (!range) {
        return std::nullopt;
    }
    CertifiedAffineForm form;
    form.terms.push_back({value, 1});
    form.lo = range->lo;
    form.hi = range->hi;
    return form;
}

struct GuardOperands {
    Value low;
    Value high;
    int64_t rhs = 0;
};

static std::optional<GuardOperands> signedGuardOperands(arith::CmpIOp cmp)
{
    GuardOperands operands;
    switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::slt:
            operands.low = cmp.getLhs();
            operands.high = cmp.getRhs();
            operands.rhs = 1;
            break;
        case arith::CmpIPredicate::sle:
            operands.low = cmp.getLhs();
            operands.high = cmp.getRhs();
            operands.rhs = 0;
            break;
        case arith::CmpIPredicate::sgt:
            operands.low = cmp.getRhs();
            operands.high = cmp.getLhs();
            operands.rhs = 1;
            break;
        case arith::CmpIPredicate::sge:
            operands.low = cmp.getRhs();
            operands.high = cmp.getLhs();
            operands.rhs = 0;
            break;
        default:
            return std::nullopt;
    }
    return operands;
}

static std::optional<std::pair<CertifiedAffineForm, int64_t>> certifiedGuardConstraint(
    const GuardOperands& operands, const CertifiedLeafRanges& freeRanges)
{
    Value low = operands.low;
    Value high = operands.high;
    int64_t rhs = operands.rhs;
    auto lowForm = certifiedAffineForm(low, freeRanges, 0);
    auto highForm = certifiedAffineForm(high, freeRanges, 0);
    if (!lowForm || !highForm) {
        return std::nullopt;
    }
    CertifiedAffineForm diff = *highForm;
    bool exact = true;
    for (auto& [leaf, coeff] : lowForm->terms) {
        if (!diff.addTerm(leaf, -static_cast<__int128>(coeff))) {
            exact = false;
            break;
        }
    }
    if (exact) {
        __int128 constant = static_cast<__int128>(diff.constant) - lowForm->constant;
        if (int64Representable(constant)) {
            diff.constant = static_cast<int64_t>(constant);
            // A constant-only constraint says nothing about any leaf; the
            // polyhedron cannot use it.
            if (!diff.terms.empty()) {
                return std::make_pair(std::move(diff), rhs);
            }
        }
    }
    return std::nullopt;
}

// A dominating scf.if guard as a certified inequality
// `diff >= rhs` over leaf values, where diff = high - low per the predicate.
static void collectGuardConstraint(
    Operation* context, Operation* parent, SmallVectorImpl<std::pair<CertifiedAffineForm, int64_t>>& out)
{
    CertifiedLeafRanges freeRanges;
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp || ifOp.getElseRegion().isAncestor(context->getParentRegion())) {
        return;
    }
    auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp) {
        return;
    }
    auto operands = signedGuardOperands(cmp);
    if (!operands) {
        return;
    }
    if (auto constraint = certifiedGuardConstraint(*operands, freeRanges)) {
        out.push_back(std::move(*constraint));
    }
}

static void collectGuardConstraints(Operation* context, SmallVectorImpl<std::pair<CertifiedAffineForm, int64_t>>& out)
{
    for (Operation* parent = context->getParentOp(); parent; parent = parent->getParentOp()) {
        collectGuardConstraint(context, parent, out);
    }
}

// Add `form >= rhs` as an inequality row over the leaf dimensions of `poly`.
static bool addFormConstraint(
    IntegerPolyhedron& poly, unsigned rowWidth, const DenseMap<Value, unsigned>& dimOf, const CertifiedAffineForm& form,
    int64_t rhs)
{
    SmallVector<int64_t, 8> row(rowWidth, 0);
    for (auto& [leaf, coeff] : form.terms) {
        if (coeff == 0) {
            continue;
        }
        auto it = dimOf.find(leaf);
        if (it == dimOf.end()) {
            return false;
        }
        __int128 acc = static_cast<__int128>(row[it->second]) + coeff;
        if (!int64Representable(acc)) {
            return false;
        }
        row[it->second] = static_cast<int64_t>(acc);
    }
    __int128 constant = -(static_cast<__int128>(rhs) - form.constant);
    if (!int64Representable(constant)) {
        return false;
    }
    row.back() = static_cast<int64_t>(constant);
    poly.addInequality(row);
    return true;
}

using GuardConstraints = SmallVector<std::pair<CertifiedAffineForm, int64_t>, 2>;

struct GuardLeafDimensions {
    SmallVector<Value, 4> leaves;
    DenseMap<Value, unsigned> dimOf;

    void registerForm(const CertifiedAffineForm& form)
    {
        for (auto& [leaf, coeff] : form.terms) {
            if (coeff == 0) {
                continue;
            }
            auto [it, inserted] = dimOf.try_emplace(leaf, leaves.size());
            if (inserted) {
                leaves.push_back(leaf);
            }
        }
    }
};

// An unrepresentable lower-bound constant drops only a restriction, so the
// feasible set is widened. In particular, never negate INT64_MIN in int64.
static void addLeafRange(IntegerPolyhedron& poly, unsigned dim, unsigned rowWidth, AddressRange range)
{
    __int128 negLo = -static_cast<__int128>(range.lo);
    if (int64Representable(negLo)) {
        SmallVector<int64_t> row(rowWidth, 0);
        row[dim] = 1;
        row.back() = static_cast<int64_t>(negLo);
        poly.addInequality(row);
    }
    SmallVector<int64_t> row(rowWidth, 0);
    row[dim] = -1;
    row.back() = range.hi;
    poly.addInequality(row);
}

static std::optional<AddressRange> intersectConstantBounds(
    const IntegerPolyhedron& poly, unsigned dim, AddressRange range)
{
    if (auto lb = checkedConstantBound(poly, BoundType::LB, dim)) {
        range.lo = std::max(range.lo, *lb);
    }
    if (auto ub = checkedConstantBound(poly, BoundType::UB, dim)) {
        range.hi = std::min(range.hi, *ub);
    }
    return range.lo <= range.hi ? std::optional<AddressRange>(range) : std::nullopt;
}

static IntegerPolyhedron buildGuardPolyhedron(
    const GuardLeafDimensions& dims, const GuardConstraints& guards, const CertifiedLeafRanges& ranges,
    unsigned extraDims = 0)
{
    unsigned numDims = dims.leaves.size() + extraDims;
    IntegerPolyhedron poly(presburger::PresburgerSpace::getSetSpace(numDims));
    for (unsigned d = 0; d < dims.leaves.size(); ++d) {
        if (auto range = ranges.lookup(dims.leaves[d])) {
            addLeafRange(poly, d, numDims + 1, *range);
        }
    }
    for (auto& [diff, rhs] : guards) {
        addFormConstraint(poly, numDims + 1, dims.dimOf, diff, rhs);
    }
    return poly;
}

static std::optional<DenseMap<Value, AddressRange>> refineGuardLeaves(
    const GuardLeafDimensions& dims, const IntegerPolyhedron& poly)
{
    DenseMap<Value, AddressRange> refined;
    for (unsigned d = 0; d < dims.leaves.size(); ++d) {
        if (auto base = inferredLeafRange(dims.leaves[d])) {
            auto range = intersectConstantBounds(poly, d, *base);
            if (!range) {
                return std::nullopt;
            }
            refined[dims.leaves[d]] = *range;
        }
    }
    return refined;
}

static bool pinTargetForm(IntegerPolyhedron& poly, const GuardLeafDimensions& dims, const CertifiedAffineForm& form)
{
    unsigned targetDim = dims.leaves.size();
    SmallVector<int64_t> eq(targetDim + 2, 0);
    eq[targetDim] = -1;
    for (auto& [leaf, coeff] : form.terms) {
        if (coeff == 0) {
            continue;
        }
        auto it = dims.dimOf.find(leaf);
        if (it == dims.dimOf.end()) {
            return false;
        }
        __int128 acc = static_cast<__int128>(eq[it->second]) + coeff;
        if (!int64Representable(acc)) {
            return false;
        }
        eq[it->second] = static_cast<int64_t>(acc);
    }
    eq.back() = form.constant;
    poly.addEquality(eq);
    return true;
}

// Refine guard leaves first, then certify the target over those refined
// ranges and pin it as an extra dimension of the same guarded domain.
static std::optional<AddressRange> guardedValueRange(Operation* context, Value target)
{
    GuardConstraints guards;
    collectGuardConstraints(context, guards);
    if (guards.empty()) {
        return std::nullopt;
    }
    GuardLeafDimensions dims;
    for (auto& [diff, rhs] : guards) {
        dims.registerForm(diff);
    }
    CertifiedLeafRanges ranges;
    auto refine = buildGuardPolyhedron(dims, guards, ranges);
    auto refined = refineGuardLeaves(dims, refine);
    if (!refined) {
        return std::nullopt;
    }
    ranges.refined = &*refined;
    auto form = certifiedAffineForm(target, ranges, 0);
    if (!form) {
        return std::nullopt;
    }
    if (form->terms.empty()) {
        return AddressRange{form->lo, form->hi};
    }
    dims.registerForm(*form);
    auto bound = buildGuardPolyhedron(dims, guards, ranges, /*extraDims=*/1);
    if (!pinTargetForm(bound, dims, *form)) {
        return std::nullopt;
    }
    return intersectConstantBounds(bound, dims.leaves.size(), AddressRange{form->lo, form->hi});
}

// A truncation-narrowed `minsi(x, K)` bound keeps the clamp only when the truncated value is the faithful
// minsi result: its guarded range must be representable in the bound's type.
static std::optional<AddressRange> getSignedRangeAtImpl(Value value, Operation* context, unsigned depth)
{
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    if (!width || width > 64 || !context || depth >= 16) {
        return std::nullopt;
    }
    if (auto min = value.getDefiningOp<arith::MinSIOp>()) {
        auto lhs = getSignedRangeAtImpl(min.getLhs(), context, depth + 1);
        auto rhs = getSignedRangeAtImpl(min.getRhs(), context, depth + 1);
        if (lhs && rhs) {
            return AddressRange{std::min(lhs->lo, rhs->lo), std::min(lhs->hi, rhs->hi)};
        }
    }
    Operation* definition = value.getDefiningOp();
    if (isa_and_nonnull<arith::TruncIOp, arith::IndexCastOp>(definition)) {
        auto input = getSignedRangeAtImpl(definition->getOperand(0), context, depth + 1);
        __int128 limit = static_cast<__int128>(1) << (width - 1);
        if (input && input->lo >= -limit && input->hi < limit) {
            return input;
        }
    }
    if (auto guarded = guardedValueRange(context, value)) {
        return guarded;
    }
    return inferredLeafRange(value);
}

std::optional<AddressRange> getSignedRangeAt(Value value, Operation* context)
{
    return getSignedRangeAtImpl(value, context, 0);
}

static Value unwrapNarrowingBound(Value value)
{
    for (unsigned hop = 0; hop < 4; ++hop) {
        Operation* op = value.getDefiningOp();
        if (!op) {
            break;
        }
        if (auto truncOp = dyn_cast<arith::TruncIOp>(op)) {
            value = truncOp.getIn();
            continue;
        }
        if (auto castOp = dyn_cast<arith::IndexCastOp>(op)) {
            auto inWidth = storageBitwidth(castOp.getIn().getType());
            auto outWidth = storageBitwidth(value.getType());
            if (inWidth && outWidth && *inWidth > *outWidth) {
                value = castOp.getIn();
                continue;
            }
        }
        break;
    }
    return value;
}

static std::optional<std::pair<Value, int64_t>> constantSignedClamp(Value value)
{
    auto minOp = value.getDefiningOp<arith::MinSIOp>();
    if (!minOp) {
        return std::nullopt;
    }
    if (auto k = getConstantInt(minOp.getRhs())) {
        return std::make_pair(minOp.getLhs(), *k);
    }
    if (auto k = getConstantInt(minOp.getLhs())) {
        return std::make_pair(minOp.getRhs(), *k);
    }
    return std::nullopt;
}

static std::optional<int64_t> guardedClampUpperBound(scf::ForOp forOp)
{
    auto boundWidth = storageBitwidth(forOp.getUpperBound().getType());
    if (!boundWidth) {
        return std::nullopt;
    }
    auto clamp = constantSignedClamp(unwrapNarrowingBound(forOp.getUpperBound()));
    if (!clamp || clamp->second < 0) {
        return std::nullopt;
    }
    auto range = guardedValueRange(forOp, clamp->first);
    if (!range) {
        return std::nullopt;
    }
    // V = minsi(X, K) over X in [lo, hi] lies in [min(lo, K), min(hi, K)].
    __int128 vLo = std::min<__int128>(range->lo, clamp->second);
    __int128 vHi = std::min<__int128>(range->hi, clamp->second);
    __int128 limit = static_cast<__int128>(1) << (*boundWidth - 1);
    if (vLo < -limit || vHi >= limit) {
        return std::nullopt; // not representable: keep the wide envelope
    }
    return static_cast<int64_t>(std::max<__int128>(vHi, 0));
}

static PTOLoopDomain inferLoopEnvelope(scf::ForOp forOp)
{
    auto bounds = getConstBounds(forOp);
    if (bounds.valid) {
        return bounds;
    }
    auto lb = getConstantInt(forOp.getLowerBound());
    auto step = getConstantInt(forOp.getStep());
    if (!lb || !step || *step <= 0) {
        return bounds;
    }
    std::optional<int64_t> upper;
    auto direct = inferIntegerRange(forOp.getUpperBound()).smax();
    // Full narrow ranges are finite and cheap to model. Keep full-width
    // i64/index leaves conservative; their huge domains do not refine access
    // proofs and can overflow downstream signed-distance calculations.
    bool directBound = direct.isSignedIntN(64) && (direct.getBitWidth() < 64 || !direct.isMaxSignedValue());
    if (directBound) {
        upper = direct.getSExtValue();
    }
    // A truncation-narrowed clamp bound loses the minsi clamp to the full
    // narrow range; the guards of the loop may certify it back.
    if (auto clamped = guardedClampUpperBound(forOp)) {
        upper = upper ? std::min(*upper, *clamped) : clamped;
    }
    if (!upper) {
        return bounds;
    }
    bounds.lb = *lb;
    bounds.ub = *upper;
    bounds.step = *step;
    bounds.valid = true;
    return bounds;
}

// Every consumer of an envelope, including at-most-once and cast proofs,
// needs the final increment to remain representable before the exit test.
PTOLoopDomain getLoopEnvelope(scf::ForOp forOp)
{
    auto bounds = inferLoopEnvelope(forOp);
    auto trip = bounds.maxTripCount();
    unsigned width = ConstantIntRanges::getStorageBitwidth(forOp.getInductionVar().getType());
    if (!bounds.valid || !trip || !width || width > 64 || bounds.step <= 0) {
        return {};
    }
    __int128 limit = static_cast<__int128>(1) << (width - 1);
    __int128 after = static_cast<__int128>(bounds.lb) + static_cast<__int128>(*trip) * bounds.step;
    if (bounds.lb < -limit || after >= limit) {
        return {};
    }
    return bounds;
}

static std::optional<AddressRange> inductionRange(Value value, ArrayRef<Value> ivs, ArrayRef<PTOLoopDomain> bounds)
{
    for (auto [iv, bound] : llvm::zip(ivs, bounds)) {
        if (value != iv) {
            continue;
        }
        auto trip = bound.maxTripCount();
        if (!trip || *trip == 0) {
            return std::nullopt;
        }
        __int128 last = static_cast<__int128>(bound.lb) + static_cast<__int128>(*trip - 1) * bound.step;
        if (last > INT64_MAX || last < INT64_MIN) {
            return std::nullopt;
        }
        return AddressRange{bound.lb, static_cast<int64_t>(last)};
    }
    return std::nullopt;
}

static std::optional<AddressRange> checkedArithmeticRange(
    Operation* op, AddressRange lhs, std::optional<AddressRange> rhs, unsigned width)
{
    __int128 lo = lhs.lo;
    __int128 hi = lhs.hi;
    if (isa<arith::AddIOp>(op)) {
        lo += rhs->lo;
        hi += rhs->hi;
    } else if (isa<arith::SubIOp>(op)) {
        lo -= rhs->hi;
        hi -= rhs->lo;
    } else if (isa<arith::MulIOp>(op)) {
        __int128 products[] = {lo * rhs->lo, lo * rhs->hi, hi * rhs->lo, hi * rhs->hi};
        lo = *std::min_element(std::begin(products), std::end(products));
        hi = *std::max_element(std::begin(products), std::end(products));
    }
    bool unsignedCast = isa<arith::ExtUIOp, arith::IndexCastUIOp>(op);
    if (unsignedCast && lhs.lo < 0) {
        return std::nullopt;
    }
    __int128 limit = static_cast<__int128>(1) << (width - 1);
    if (lo < -limit || hi >= limit) {
        return std::nullopt;
    }
    return AddressRange{static_cast<int64_t>(lo), static_cast<int64_t>(hi)};
}

std::optional<AddressRange> getAddressRange(
    Value value, ArrayRef<Value> ivs, ArrayRef<PTOLoopDomain> bounds, unsigned depth)
{
    if (depth == 32) {
        return std::nullopt;
    }
    if (auto constant = getConstantInt(value)) {
        return AddressRange{*constant, *constant};
    }
    if (auto remBound = matchSignCorrectedRemSum(value)) {
        return AddressRange{0, *remBound};
    }
    if (llvm::is_contained(ivs, value)) {
        return inductionRange(value, ivs, bounds);
    }
    Operation* op = value.getDefiningOp();
    bool noOperands = op && op->getNumOperands() == 0;
    if (!op || noOperands) {
        return inferredLeafRange(value);
    }
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    if (!width || width > 64) {
        return std::nullopt;
    }
    bool arithmetic = isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op);
    if (!arithmetic &&
        !isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
        // Decomposed affine.apply cannot be certified as an opaque leaf.
        return isa<affine::AffineApplyOp>(op) ? std::nullopt : inferredLeafRange(value);
    }
    auto lhs = getAddressRange(op->getOperand(0), ivs, bounds, depth + 1);
    if (!lhs) {
        return std::nullopt;
    }
    std::optional<AddressRange> rhs;
    if (arithmetic) {
        rhs = getAddressRange(op->getOperand(1), ivs, bounds, depth + 1);
        if (!rhs) {
            return std::nullopt;
        }
    }
    return checkedArithmeticRange(op, *lhs, rhs, width);
}

// Domain-free point queries retain the scheduling consumer's original leaf
// policy. In particular, loop bounds and branch guards do not tighten them.


static std::optional<ConstantIntRanges>
getLoopInductionRange(BlockArgument argument, unsigned bitWidth) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
  bool isLoopInduction = forOp && forOp.getInductionVar() == argument;
  if (!isLoopInduction) {
    return std::nullopt;
  }
  std::optional<int64_t> lower = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lower || !step || *lower < 0 || *step <= 0) {
    return std::nullopt;
  }
  APInt lowerBound(bitWidth, static_cast<uint64_t>(*lower),
                   /*isSigned=*/true);
  return ConstantIntRanges::fromSigned(lowerBound,
                                       APInt::getSignedMaxValue(bitWidth));
}

static std::optional<SmallVector<ConstantIntRanges>>
getOperandRanges(Operation &operation, unsigned depth) {
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(operation.getNumOperands());
  for (Value operand : operation.getOperands()) {
    std::optional<ConstantIntRanges> range =
        inferPointIntegerRange(operand, depth + 1);
    if (!range) {
      return std::nullopt;
    }
    ranges.push_back(std::move(*range));
  }
  return ranges;
}

std::optional<ConstantIntRanges> inferPointIntegerRange(Value value,
                                                         unsigned depth) {
  constexpr unsigned maxDepth = 32;
  if (!value || depth > maxDepth) {
    return std::nullopt;
  }
  IntegerAttr constant;
  if (matchPattern(value, m_Constant(&constant))) {
    return ConstantIntRanges::constant(constant.getValue());
  }
  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(value.getType());
  if (bitWidth == 0) {
    return std::nullopt;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    return getLoopInductionRange(argument, bitWidth);
  }
  Operation *definingOp = value.getDefiningOp();
  auto rangeInterface = dyn_cast_or_null<InferIntRangeInterface>(definingOp);
  if (!rangeInterface) {
    return std::nullopt;
  }
  std::optional<SmallVector<ConstantIntRanges>> ranges =
      getOperandRanges(*definingOp, depth);
  if (!ranges) {
    return std::nullopt;
  }
  std::optional<ConstantIntRanges> result;
  rangeInterface.inferResultRanges(
      *ranges, [&](Value resultValue, const ConstantIntRanges &range) {
        if (resultValue == value) {
          result = range;
        }
      });
  return result;
}

} // namespace mlir::pto::integer_range
