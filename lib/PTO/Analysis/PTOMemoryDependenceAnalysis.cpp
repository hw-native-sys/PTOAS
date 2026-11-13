// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOMemoryDependenceAnalysis.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::integer_range;
using mlir::presburger::IntegerPolyhedron;

SmallVector<unsigned, 2> pto::getCommonLoopPrefix(ArrayRef<unsigned> lhs, ArrayRef<unsigned> rhs)
{
    SmallVector<unsigned, 2> common;
    for (auto [left, right] : llvm::zip(lhs, rhs)) {
        if (left != right) {
            break;
        }
        common.push_back(left);
    }
    return common;
}

namespace {
struct MemoryQueryLoop {
    unsigned id;
    scf::ForOp op;
    Value inductionVar;
};
struct MemoryQueryAccess : PTOMemoryAccessOccurrence {
    SmallVector<unsigned, 2> loopNest;
};
struct MemoryQueryContext {
    Operation* scope;
    SmallVector<MemoryQueryLoop> loops;
    SmallVector<MemoryQueryAccess> accesses;
};

static SmallVector<Value, 2> collectIVs(Operation* op, Operation* scope)
{
    SmallVector<Value, 2> ivs;
    for (Operation* parent = op->getParentOp(); parent && parent != scope; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            ivs.push_back(loop.getInductionVar());
        }
    }
    std::reverse(ivs.begin(), ivs.end());
    return ivs;
}

static std::optional<unsigned> loopIdForIV(const MemoryQueryContext& context, Value iv)
{
    for (const auto& loop : context.loops) {
        if (loop.inductionVar == iv) {
            return loop.id;
        }
    }
    return std::nullopt;
}

static MemoryQueryContext buildContext(
    const PTOMemoryAccessOccurrence& producer, const PTOMemoryAccessOccurrence& consumer, Operation* scope)
{
    MemoryQueryContext context{scope, {}, {{producer, {}}, {consumer, {}}}};
    for (auto& access : context.accesses) {
        access.loopNest.clear();
        for (Value iv : collectIVs(access.op, scope)) {
            auto id = loopIdForIV(context, iv);
            if (!id) {
                id = context.loops.size();
                auto argument = cast<BlockArgument>(iv);
                context.loops.push_back({*id, cast<scf::ForOp>(argument.getOwner()->getParentOp()), iv});
            }
            access.loopNest.push_back(*id);
        }
    }
    return context;
}
// Presburger loop-carried dependence relation

static bool addIterationDomain(IntegerPolyhedron& p, unsigned pos, const PTOLoopDomain& b)
{
    auto trip = b.maxTripCount();
    if (!trip) {
        return false;
    }
    // Convention: coeffs[var...] + constant >= 0.
    SmallVector<int64_t> lo(p.getNumVars() + 1, 0);
    lo[pos] = 1;
    p.addInequality(lo);
    SmallVector<int64_t> hi(p.getNumVars() + 1, 0);
    hi[pos] = -1;
    hi.back() = *trip - 1;
    p.addInequality(hi);
    return true;
}

// Resolve the loop carrying an IV. IVs are scf.for block arguments (no
// defining op), so the loop is found via the analysis result's recorded
// inductionVar.
static scf::ForOp resolveLoopForIV(const MemoryQueryContext& r, Value iv)
{
    if (auto id = loopIdForIV(r, iv)) {
        for (const auto& l : r.loops) {
            if (l.id == *id) {
                return l.op;
            }
        }
    }
    return scf::ForOp();
}

// Gather the constant bounds of every IV in `ivs` (outermost first). Returns
// false when an IV has no recorded loop or a non-constant domain.
static bool gatherConstBounds(
    const MemoryQueryContext& result, ArrayRef<Value> ivs, SmallVectorImpl<PTOLoopDomain>& bounds)
{
    for (Value iv : ivs) {
        scf::ForOp forOp = resolveLoopForIV(result, iv);
        if (!forOp) {
            return false;
        }
        if (!getConstBounds(forOp).valid) {
            return false;
        }
        bounds.push_back(getLoopEnvelope(forOp));
        if (!bounds.back().valid) {
            return false;
        }
    }
    return true;
}

// Rewrite byte-offset coefficients from loop-IV coordinates to iteration-
// counter coordinates (iv = lb + step * iter), folding the lb products into
// the constant. Fails when any scaled coefficient or constant overflows.
static bool convertToIterationCounters(
    SmallVectorImpl<int64_t>& coeffs, ArrayRef<PTOLoopDomain> bounds, int64_t& constant)
{
    __int128 adjusted = constant;
    for (unsigned k = 0; k < coeffs.size(); ++k) {
        adjusted += static_cast<__int128>(coeffs[k]) * bounds[k].lb;
        __int128 scaled = static_cast<__int128>(coeffs[k]) * bounds[k].step;
        if (scaled < INT64_MIN || scaled > INT64_MAX) {
            return false;
        }
        coeffs[k] = static_cast<int64_t>(scaled);
    }
    if (adjusted < INT64_MIN || adjusted > INT64_MAX) {
        return false;
    }
    constant = static_cast<int64_t>(adjusted);
    return true;
}

// Whether an absolute base is present and representable as int64.
static bool absoluteBaseComparable(const PTOMemoryFootprint& fp)
{
    return fp.absoluteBase && *fp.absoluteBase <= static_cast<uint64_t>(INT64_MAX);
}

static bool comparableMemoryRoots(const PTOMemoryFootprint& P, const PTOMemoryFootprint& C)
{
    bool bothAbsolute = P.rootKind == PTOMemoryRootKind::Absolute && C.rootKind == PTOMemoryRootKind::Absolute;
    bool sameNonAbsoluteRoot =
        (P.rootKind == PTOMemoryRootKind::Symbolic || P.rootKind == PTOMemoryRootKind::ProvenAllocation) &&
        (C.rootKind == PTOMemoryRootKind::Symbolic || C.rootKind == PTOMemoryRootKind::ProvenAllocation) &&
        P.root == C.root;
    if (!bothAbsolute && !sameNonAbsoluteRoot) {
        return false;
    }
    bool basesComparable = absoluteBaseComparable(P) && absoluteBaseComparable(C);
    if (bothAbsolute && !basesComparable) {
        return false;
    }
    // Distinct proven allocations are already NoAlias at `aliasMemorySameIteration`;
    // reaching here means roots are comparable, so continue.
    return true;
}

// Early feasibility checks for the sequential-overlap model: closed byte
// ranges on both sides, no forced-MayAlias footprint, and comparable roots.
static bool sequentialRootsComparable(const PTOMemoryFootprint& P, const PTOMemoryFootprint& C)
{
    // Need closed byte ranges on both sides. forcesMayAlias footprints (e.g.
    // vsstb) with an open upper bound cannot be disproven here because their
    // upper bound is unknown.
    if (!P.byteSize || !C.byteSize || !P.byteOffset.exact || !C.byteOffset.exact ||
        P.byteOffset.lossy || C.byteOffset.lossy) {
        return false;
    }
    if (P.forcesMayAlias || C.forcesMayAlias) {
        return false;
    }
    // Both must resolve to a comparable root: same symbolic/allocation root, or
    // both absolute (absolute bases are comparable regardless of SSA identity).
    return comparableMemoryRoots(P, C);
}

// A relation currently has dimensions for IVs only. Never silently discard
// opaque index terms that were retained for same-iteration cancellation.
static bool hasOnlyIVTerms(const PTOMemoryFootprint& footprint, ArrayRef<Value> ivs)
{
    return llvm::all_of(footprint.byteOffset.coefficients, [&](const auto& term) {
        return term.second == 0 || llvm::is_contained(ivs, term.first);
    });
}

static bool addressArithmeticDoesNotWrap(
    Value value, ArrayRef<Value> ivs, ArrayRef<PTOLoopDomain> bounds, unsigned depth = 0)
{
    if (depth == 32) {
        return false;
    }
    if (value.getType().isIntOrIndex()) {
        return getAddressRange(value, ivs, bounds).has_value();
    }
    Operation* op = value.getDefiningOp();
    if (!op) {
        return true; // A symbolic pointer root; root comparability is checked separately.
    }
    if (!isa<pto::AddPtrOp, pto::CastPtrOp>(op)) {
        return false;
    }
    return llvm::all_of(op->getOperands(), [&](Value operand) {
        return addressArithmeticDoesNotWrap(operand, ivs, bounds, depth + 1);
    });
}

// Whether the whole address computation of an access is verified not to wrap: both the base-pointer chain
// (pto.add_ptr/pto.cast_ptr layers, which carry stage-style symbolic offsets) and the access's own offset
// operand.
static bool accessArithmeticDoesNotWrap(
    const PTOMemoryFootprint& fp, ArrayRef<Value> ivs, ArrayRef<PTOLoopDomain> bounds)
{
    if (!fp.accessBase || (!fp.accessOffset && !fp.offsetDomainCovered)) {
        return false;
    }
    return addressArithmeticDoesNotWrap(fp.accessBase, ivs, bounds) &&
           (fp.offsetDomainCovered || addressArithmeticDoesNotWrap(fp.accessOffset, ivs, bounds));
}

// The old exact IV-only path keeps its admission rule. Newly admitted
// symbolic addresses also need a no-wrap proof, even when marked exact.
static bool carriedAddressModelIsFaithful(
    const PTOMemoryFootprint& fp, ArrayRef<Value> ivs, ArrayRef<PTOLoopDomain> bounds)
{
    return (fp.byteOffset.exact && hasOnlyIVTerms(fp, ivs)) || accessArithmeticDoesNotWrap(fp, ivs, bounds);
}

// Keep accumulated symbolic endpoints in the int64 proof domain. Compute
// each candidate in 128 bits and reject before any subsequent combination
// can overflow; losing a bound only loses a proof.
static bool addSymbolSpan(__int128& lo, __int128& hi, __int128 termLo, __int128 termHi)
{
    __int128 nextLo;
    __int128 nextHi;
    bool overflow = __builtin_add_overflow(lo, termLo, &nextLo) || __builtin_add_overflow(hi, termHi, &nextHi);
    if (overflow || nextLo < INT64_MIN || nextHi > INT64_MAX) {
        return false;
    }
    lo = nextLo;
    hi = nextHi;
    return true;
}

// Bound the contribution `coefficient * symbol` and accumulate it into [lo, hi].
static bool addSymbolTermRange(__int128& lo, __int128& hi, Value symbol, __int128 coefficient)
{
    if (coefficient == 0) {
        return true;
    }
    auto range = inferredLeafRange(symbol);
    if (!range) {
        return false;
    }
    __int128 termLo = coefficient * range->lo;
    __int128 termHi = coefficient * range->hi;
    return addSymbolSpan(lo, hi, std::min(termLo, termHi), std::max(termLo, termHi));
}

// Reassociate equal-coefficient remainder/correction terms without losing
// their correlation; unmatched terms retain independent ranges.
static bool addSymbolRanges(__int128& lo, __int128& hi, ArrayRef<std::pair<Value, __int128>> terms)
{
    SmallVector<bool, 8> consumed(terms.size(), false);
    for (unsigned i = 0; i < terms.size(); ++i) {
        auto [symbol, coefficient] = terms[i];
        if (consumed[i] || coefficient == 0) {
            continue;
        }
        bool paired = false;
        for (unsigned j = i + 1; j < terms.size(); ++j) {
            if (consumed[j] || terms[j].second != coefficient) {
                continue;
            }
            auto upper = matchSignCorrectedRemPair(symbol, terms[j].first);
            if (!upper) {
                continue;
            }
            __int128 span = coefficient * *upper;
            if (!addSymbolSpan(lo, hi, std::min<__int128>(0, span), std::max<__int128>(0, span))) {
                return false;
            }
            consumed[j] = true;
            paired = true;
            break;
        }
        if (!paired && !addSymbolTermRange(lo, hi, symbol, coefficient)) {
            return false;
        }
    }
    return true;
}

// Loop-IV terms are modelled as relation dimensions, not symbolic terms:
// every symbolic-range computation below must skip them.
static bool isModelledIV(Value value, ArrayRef<Value> ivs) { return llvm::is_contained(ivs, value); }

// Range of the symbolic (non-IV) part of one byte-offset expression. The
// expression must not be lossy: `exact == false && !lossy` is a faithful sum
// of constants and shared symbolic terms, which this bounds.
static bool symbolicPartRange(const PTOAffineByteExpr& expr, ArrayRef<Value> ivs, __int128& lo, __int128& hi)
{
    lo = 0;
    hi = 0;
    if (expr.lossy) {
        return false;
    }
    SmallVector<std::pair<Value, __int128>, 4> terms;
    for (auto& [symbol, coefficient] : expr.coefficients) {
        if (!isModelledIV(symbol, ivs)) {
            terms.push_back({symbol, coefficient});
        }
    }
    return addSymbolRanges(lo, hi, terms);
}

// Range of the symbolic-part difference `lhs - rhs` over the shared symbol values (constants excluded;
// callers fold them separately).
static bool symbolicDifferenceRange(
    const PTOAffineByteExpr& lhs, const PTOAffineByteExpr& rhs, ArrayRef<Value> ivs, __int128& lo, __int128& hi)
{
    lo = 0;
    hi = 0;
    if (lhs.lossy || rhs.lossy) {
        return false;
    }
    SmallVector<Value, 4> symbols;
    auto collectSymbols = [&](const PTOAffineByteExpr& expr) {
        for (auto& [symbol, coefficient] : expr.coefficients) {
            bool candidate = coefficient != 0 && !isModelledIV(symbol, ivs);
            bool alreadyPresent = llvm::is_contained(symbols, symbol);
            if (candidate && !alreadyPresent) {
                symbols.push_back(symbol);
            }
        }
    };
    collectSymbols(lhs);
    collectSymbols(rhs);
    SmallVector<std::pair<Value, __int128>, 4> terms;
    for (Value symbol : symbols) {
        __int128 difference = static_cast<__int128>(lhs.getCoeff(symbol)) - static_cast<__int128>(rhs.getCoeff(symbol));
        terms.push_back({symbol, difference});
    }
    return addSymbolRanges(lo, hi, terms);
}

// Resolved loop-nest layout of a sequential-region access pair. Dual-set
// dims are [p0.., c0..]; single-set dims are [v0..] with v shared.
struct SequentialOverlapLayout {
    SmallVector<Value, 2> pi;        // producer IVs (outermost first)
    SmallVector<Value, 2> pj;        // consumer IVs (outermost first)
    SmallVector<unsigned, 2> common; // shared loop-id prefix
    SmallVector<PTOLoopDomain> boundsP;
    SmallVector<PTOLoopDomain> boundsC;
    // single-nest iff one IV set is a prefix of the other (incl. equal/empty).
    bool singleNest = false;
    unsigned numDims = 0;
};

// Gather the envelope bounds of every IV in `ivs` (outermost first).
static bool gatherEnvelopeBounds(
    const MemoryQueryContext& result, ArrayRef<Value> ivs, SmallVectorImpl<PTOLoopDomain>& bounds)
{
    for (Value iv : ivs) {
        scf::ForOp forOp = resolveLoopForIV(result, iv);
        if (!forOp) {
            return false;
        }
        auto envelope = getLoopEnvelope(forOp);
        auto trip = envelope.maxTripCount();
        if (!envelope.valid || !trip) {
            return false;
        }
        unsigned width = ConstantIntRanges::getStorageBitwidth(iv.getType());
        if (width == 0 || width >= 128) {
            return false;
        }
        __int128 limit = static_cast<__int128>(1) << (width - 1);
        __int128 after = static_cast<__int128>(envelope.lb) + static_cast<__int128>(*trip) * envelope.step;
        if (envelope.step <= 0 || envelope.lb < -limit || after >= limit) {
            return false;
        }
        bounds.push_back(envelope);
    }
    return true;
}

// Resolve the IV layout for the pair. Handled shapes:
//  (a) pi == pj (same loop nest): single IV set, shared IVs identical.
//  (b) one IV set is a proper prefix of the other (one access further out,
//      e.g. a top-level store feeding a load inside a loop, or the reverse):
//      single IV set = the longer of pi/pj, the shorter side's coefficients
//      read against that set (0 for IVs it does not carry).
//  (c) pi and pj share a common prefix but diverge (producer and consumer in
//      sibling loops): dual IV sets [pi..., pj...], shared prefix constrained
//      equal, non-shared IVs independent.
// Returns false when a loop has an unmodelled domain; arbitrary nest depths are supported.
// With `useEnvelope`, the bounded envelope replaces the constant domain: a
// proof over the superset stays sound, but only emptiness is actionable.
static bool resolveSequentialIVLayout(
    const MemoryQueryContext& result, const MemoryQueryAccess& prod, const MemoryQueryAccess& cons,
    SequentialOverlapLayout& layout, bool useEnvelope = false)
{
    layout.pi = collectIVs(prod.op, result.scope);
    layout.pj = collectIVs(cons.op, result.scope);
    layout.common = getCommonLoopPrefix(prod.loopNest, cons.loopNest);
    bool prefixP = layout.pi.size() <= layout.pj.size() && layout.common.size() == layout.pi.size();
    bool prefixC = layout.pj.size() <= layout.pi.size() && layout.common.size() == layout.pj.size();
    layout.singleNest = prefixP || prefixC;
    bool boundsP = useEnvelope ? gatherEnvelopeBounds(result, layout.pi, layout.boundsP) :
                                 gatherConstBounds(result, layout.pi, layout.boundsP);
    bool boundsC = useEnvelope ? gatherEnvelopeBounds(result, layout.pj, layout.boundsC) :
                                 gatherConstBounds(result, layout.pj, layout.boundsC);
    if (!boundsP || !boundsC) {
        return false;
    }
    unsigned depthP = layout.pi.size();
    unsigned depthC = layout.pj.size();
    layout.numDims = layout.singleNest ? std::max(depthP, depthC) : (depthP + depthC);
    return true;
}

// Consumer coordinate k: shared with the producer in the single-nest layout,
// offset by the producer nest in the dual-set layout.
static unsigned consumerDim(const SequentialOverlapLayout& layout, unsigned k)
{
    return layout.singleNest ? k : (unsigned(layout.pi.size()) + k);
}

// Build the base polyhedron of the pair: iteration domains for every used IV
// (pi then pj, dedup shared by position), and in the dual-set case the
// shared-prefix same-iteration equality. The domains come from the layout's
// bounds: constant for the exact model, envelopes for the dynamic disproof.
// Returns null when a domain cannot be modelled.
static std::unique_ptr<IntegerPolyhedron> buildSequentialPolyhedron(const SequentialOverlapLayout& layout)
{
    unsigned depthP = layout.pi.size();
    unsigned depthC = layout.pj.size();
    auto space = presburger::PresburgerSpace::getSetSpace(layout.numDims);
    auto poly = std::make_unique<IntegerPolyhedron>(space);
    if (layout.singleNest) {
        // Coefficients are read against the longer IV set; use whichever side is
        // longer as the coordinate basis.
        const auto& singleBounds = (depthP >= depthC) ? layout.boundsP : layout.boundsC;
        for (unsigned k = 0; k < singleBounds.size(); ++k) {
            if (!addIterationDomain(*poly, k, singleBounds[k])) {
                return nullptr;
            }
        }
    } else {
        for (unsigned k = 0; k < depthP; ++k) {
            if (!addIterationDomain(*poly, k, layout.boundsP[k])) {
                return nullptr;
            }
        }
        for (unsigned k = 0; k < depthC; ++k) {
            if (!addIterationDomain(*poly, depthP + k, layout.boundsC[k])) {
                return nullptr;
            }
        }
        // Constrain the shared prefix equal (same iteration for the common loops).
        for (unsigned k = 0; k < layout.common.size(); ++k) {
            SmallVector<int64_t> eq(layout.numDims + 1, 0);
            eq[k] -= 1;
            eq[depthP + k] += 1;
            poly->addInequality(eq);
            SmallVector<int64_t> eq2(layout.numDims + 1, 0);
            eq2[k] += 1;
            eq2[depthP + k] -= 1;
            poly->addInequality(eq2);
        }
    }
    return poly;
}

// Base bytes. For absolute roots combine the absolute base with the constant
// part of the byte offset; for symbolic/allocation roots use the constant
// part only (relative to the shared root).
static bool modelRootBase(const PTOMemoryFootprint& fp, int64_t& base)
{
    if (fp.rootKind == PTOMemoryRootKind::Symbolic || fp.rootKind == PTOMemoryRootKind::ProvenAllocation) {
        base = fp.byteOffset.constant;
        return true;
    }
    if (fp.rootKind != PTOMemoryRootKind::Absolute || !fp.absoluteBase ||
        *fp.absoluteBase > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    __int128 v = static_cast<__int128>(*fp.absoluteBase) + static_cast<__int128>(fp.byteOffset.constant);
    if (v < INT64_MIN || v > INT64_MAX) {
        return false;
    }
    base = static_cast<int64_t>(v);
    return true;
}

struct SequentialAddressModel {
    int64_t constP = 0;
    int64_t constC = 0;
    int64_t sizeP = 0;
    int64_t sizeC = 0;
    SmallVector<int64_t, 2> coeffP;
    SmallVector<int64_t, 2> coeffC;
};

static void appendLoopCoefficients(
    const PTOMemoryFootprint& footprint, ArrayRef<Value> ivs, SmallVectorImpl<int64_t>& coefficients)
{
    for (Value iv : ivs) {
        coefficients.push_back(footprint.byteOffset.getCoeff(iv));
    }
}

static bool modelSequentialAddresses(
    const PTOMemoryFootprint& P, const PTOMemoryFootprint& C, const SequentialOverlapLayout& layout,
    SequentialAddressModel& model)
{
    bool rootsKnown = modelRootBase(P, model.constP) && modelRootBase(C, model.constC);
    if (!rootsKnown || !P.byteSize || !C.byteSize || *P.byteSize > INT64_MAX || *C.byteSize > INT64_MAX) {
        return false;
    }
    model.sizeP = static_cast<int64_t>(*P.byteSize);
    model.sizeC = static_cast<int64_t>(*C.byteSize);
    appendLoopCoefficients(P, layout.pi, model.coeffP);
    appendLoopCoefficients(C, layout.pj, model.coeffC);
    return convertToIterationCounters(model.coeffP, layout.boundsP, model.constP) &&
           convertToIterationCounters(model.coeffC, layout.boundsC, model.constC);
}

// Keep invariant integer symbols as integer coordinates instead of eliminating
// them to a convex difference interval (which fills gaps between stage slots).
using SymbolTerms = SmallVector<std::pair<Value, int64_t>, 4>;

static SymbolTerms boundedSymbolDifference(
    const PTOAffineByteExpr& lhs, const PTOAffineByteExpr& rhs, ArrayRef<Value> ivs)
{
    SymbolTerms terms;
    SmallVector<Value, 4> symbols;
    for (const auto* expr : {&lhs, &rhs}) {
        for (auto [value, coefficient] : expr->coefficients) {
            bool collect = coefficient && !llvm::is_contained(ivs, value) && !llvm::is_contained(symbols, value);
            if (collect) {
                symbols.push_back(value);
            }
        }
    }
    for (Value value : symbols) {
        __int128 coefficient = static_cast<__int128>(lhs.getCoeff(value)) - rhs.getCoeff(value);
        if (coefficient <= INT64_MIN || coefficient > INT64_MAX || !inferredLeafRange(value)) {
            return {};
        }
        if (coefficient) {
            terms.push_back({value, static_cast<int64_t>(coefficient)});
        }
    }
    return terms;
}

static void appendSymbolCoordinates(IntegerPolyhedron& poly, ArrayRef<std::pair<Value, int64_t>> terms)
{
    for (auto [value, coefficient] : terms) {
        auto range = *inferredLeafRange(value);
        unsigned position = poly.appendVar(presburger::VarKind::Local);
        SmallVector<int64_t> row(poly.getNumVars() + 1, 0);
        row[position] = 1;
        if (range.lo != INT64_MIN) {
            row.back() = -range.lo;
            poly.addInequality(row);
        }
        row[position] = -1;
        row.back() = range.hi;
        poly.addInequality(row);
    }
}

// Overlap rows eliminate D=symP-symC with relaxed endpoints -D.lo/+D.hi.
// Only an empty overapproximation proves disjointness.
static bool addIntervalIntersectionRows(
    IntegerPolyhedron& poly, unsigned numDims, ArrayRef<int64_t> coeffP, ArrayRef<unsigned> pPositions,
    ArrayRef<int64_t> coeffC, ArrayRef<unsigned> cPositions, int64_t constP, int64_t constC, int64_t sizeP,
    int64_t sizeC, __int128 symDiffLo = 0, __int128 symDiffHi = 0, ArrayRef<std::pair<Value, int64_t>> symbols = {})
{
    if (!symbols.empty()) {
        bool intervalAdded = addIntervalIntersectionRows(
            poly, numDims, coeffP, pPositions, coeffC, cPositions, constP, constC, sizeP, sizeC, symDiffLo, symDiffHi);
        if (!intervalAdded) {
            return false;
        }
    }
    unsigned symbolStart = poly.getNumVars();
    appendSymbolCoordinates(poly, symbols);
    if (!symbols.empty()) {
        symDiffLo = 0;
        symDiffHi = 0;
    }
    SmallVector<int64_t> row(poly.getNumVars() + 1, 0);
    for (auto [coefficient, position] : llvm::zip(coeffP, pPositions)) {
        row[position] -= coefficient;
    }
    for (auto [coefficient, position] : llvm::zip(coeffC, cPositions)) {
        row[position] += coefficient;
    }
    for (auto [index, term] : llvm::enumerate(symbols)) {
        row[symbolStart + index] = -term.second;
    }
    __int128 rowConstant = static_cast<__int128>(constC) - constP + sizeC - 1 - symDiffLo;
    if (rowConstant < INT64_MIN || rowConstant > INT64_MAX) {
        return false;
    }
    row.back() = static_cast<int64_t>(rowConstant);
    poly.addInequality(row);

    SmallVector<int64_t> row2(poly.getNumVars() + 1, 0);
    for (auto [coefficient, position] : llvm::zip(coeffP, pPositions)) {
        row2[position] += coefficient;
    }
    for (auto [coefficient, position] : llvm::zip(coeffC, cPositions)) {
        row2[position] -= coefficient;
    }
    for (auto [index, term] : llvm::enumerate(symbols)) {
        row2[symbolStart + index] = term.second;
    }
    rowConstant = static_cast<__int128>(constP) - constC + sizeP - 1 + symDiffHi;
    if (rowConstant < INT64_MIN || rowConstant > INT64_MAX) {
        return false;
    }
    row2.back() = static_cast<int64_t>(rowConstant);
    poly.addInequality(row2);
    return true;
}

// Closed intervals overlap when C-P+sizeC-1 >= 0 and P-C+sizeP-1 >= 0.
// Eliminate D=symP-symC using -D.lo/+D.hi, never the tightening endpoints.
static bool addSequentialIntersection(
    IntegerPolyhedron& poly, const SequentialOverlapLayout& layout, ArrayRef<int64_t> coeffP, ArrayRef<int64_t> coeffC,
    int64_t constP, int64_t constC, int64_t sizeP, int64_t sizeC, __int128 symDiffLo = 0, __int128 symDiffHi = 0,
    ArrayRef<std::pair<Value, int64_t>> symbols = {})
{
    SmallVector<unsigned> pPositions;
    SmallVector<unsigned> cPositions;
    for (unsigned k = 0; k < layout.pi.size(); ++k) {
        pPositions.push_back(k);
    }
    for (unsigned k = 0; k < layout.pj.size(); ++k) {
        cPositions.push_back(consumerDim(layout, k));
    }
    return addIntervalIntersectionRows(
        poly, layout.numDims, coeffP, pPositions, coeffC, cPositions, constP, constC, sizeP, sizeC, symDiffLo,
        symDiffHi, symbols);
}

// Analyze interval overlap across two sequential execution regions. A
// non-empty exact Presburger relation is ProvenDependence, an empty one is
// NoDependence, and any incomplete root/domain/footprint model is Unknown.
// Modelled dimensions: the union of producer and consumer IVs. Shared loops
// (common prefix of the two loop nests) are constrained equal (same
// iteration); IVs that only one side carries are left as free dims of that
// side. The address-intersection constraints mirror `addIntersection`: a pair
// of intervals is disjoint iff NOT (addrP < addrC + sizeC AND addrC < addrP +
// sizeP), so proving the polyhedron of the conjunction empty proves disjoint.
static PTOMemoryDependenceStatus analyzeSequentialOverlap(
    const MemoryQueryContext& result, const MemoryQueryAccess& prod, const MemoryQueryAccess& cons)
{
    const auto& P = prod.footprint;
    const auto& C = cons.footprint;
    if (!sequentialRootsComparable(P, C)) {
        return PTOMemoryDependenceStatus::Unknown;
    }

    SequentialOverlapLayout layout;
    if (!resolveSequentialIVLayout(result, prod, cons, layout)) {
        return PTOMemoryDependenceStatus::Unknown;
    }
    bool unmodelledSymbols = !hasOnlyIVTerms(P, layout.pi) || !hasOnlyIVTerms(C, layout.pj);
    if (unmodelledSymbols) {
        return PTOMemoryDependenceStatus::Unknown;
    }
    auto poly = buildSequentialPolyhedron(layout);
    if (!poly) {
        return PTOMemoryDependenceStatus::Unknown;
    }

    SequentialAddressModel model;
    if (!modelSequentialAddresses(P, C, layout, model)) {
        return PTOMemoryDependenceStatus::Unknown;
    }

    if (!addSequentialIntersection(
            *poly, layout, model.coeffP, model.coeffC, model.constP, model.constC, model.sizeP, model.sizeC)) {
        return PTOMemoryDependenceStatus::Unknown;
    }

    return poly->isIntegerEmpty() ? PTOMemoryDependenceStatus::NoDependence : PTOMemoryDependenceStatus::MayDependence;
}

// Certify full address arithmetic, then solve over bounded supersets.
// Nonempty or unmodelled intersections retain the conservative hazard.
static bool prepareBoundedSequentialModel(
    const MemoryQueryContext& result, unsigned i, unsigned j, SequentialOverlapLayout& layout,
    SequentialAddressModel& model, __int128& symDiffLo, __int128& symDiffHi)
{
    const auto& A = result.accesses;
    const auto& P = A[i].footprint;
    const auto& C = A[j].footprint;
    // Closed byte ranges on both sides; forcesMayAlias footprints with an
    // open upper bound cannot be disproven here.
    bool closedAccesses = P.byteSize && C.byteSize && !P.forcesMayAlias && !C.forcesMayAlias;
    if (!closedAccesses) {
        return false;
    }
    if (!comparableMemoryRoots(P, C)) {
        return false;
    }

    if (!resolveSequentialIVLayout(result, A[i], A[j], layout, /*useEnvelope=*/true)) {
        return false;
    }
    bool noWrap = accessArithmeticDoesNotWrap(P, layout.pi, layout.boundsP) &&
                  accessArithmeticDoesNotWrap(C, layout.pj, layout.boundsC);
    if (!noWrap) {
        return false;
    }

    if (!modelSequentialAddresses(P, C, layout, model)) {
        return false;
    }

    SmallVector<Value, 4> modelledIVs;
    llvm::append_range(modelledIVs, layout.pi);
    llvm::append_range(modelledIVs, layout.pj);
    if (!symbolicDifferenceRange(P.byteOffset, C.byteOffset, modelledIVs, symDiffLo, symDiffHi)) {
        return false;
    }

    return true;
}

static bool disproveBoundedSequentialPair(const MemoryQueryContext& result, unsigned i, unsigned j)
{
    SequentialOverlapLayout layout;
    SequentialAddressModel model;
    __int128 symDiffLo = 0;
    __int128 symDiffHi = 0;
    if (!prepareBoundedSequentialModel(result, i, j, layout, model, symDiffLo, symDiffHi)) {
        return false;
    }
    auto poly = buildSequentialPolyhedron(layout);
    if (!poly) {
        return false;
    }
    SmallVector<Value, 4> ivs(layout.pi.begin(), layout.pi.end());
    llvm::append_range(ivs, layout.pj);
    auto symbols =
        boundedSymbolDifference(result.accesses[i].footprint.byteOffset, result.accesses[j].footprint.byteOffset, ivs);
    if (!addSequentialIntersection(
            *poly, layout, model.coeffP, model.coeffC, model.constP, model.constC, model.sizeP, model.sizeC, symDiffLo,
            symDiffHi, symbols)) {
        return false;
    }
    return poly->isIntegerEmpty();
}

// Per-pair model for the loop-carried Presburger relations: the resolved
// loop depths, iteration domains, and modelled byte coefficients of one
// producer/consumer pair.
struct CarriedPairModel {
    SymbolTerms invariantSymbols;
    unsigned depthP = 0;
    unsigned depthC = 0;
    unsigned numDims = 0;
    SmallVector<PTOLoopDomain> boundsP;
    SmallVector<PTOLoopDomain> boundsC;
    SmallVector<int64_t, 2> coeffP;
    SmallVector<int64_t, 2> coeffC;
    int64_t constP = 0;
    int64_t constC = 0;
    int64_t sizeP = 0;
    int64_t sizeC = 0;
    __int128 symDiffLo = 0;
    __int128 symDiffHi = 0;
    __int128 symSpanLoP = 0;
    __int128 symSpanHiP = 0;
    __int128 symSpanLoC = 0;
    __int128 symSpanHiC = 0;
};

// Dimensions: all producer coordinates, followed by all consumer coordinates.
static unsigned carriedDimPos(const CarriedPairModel& m, bool producer, unsigned k)
{
    return producer ? k : (m.depthP + k);
}

// Build the base relation polyhedron: the iteration domains of both sides.
// Returns null when a domain cannot be modelled.
static std::unique_ptr<IntegerPolyhedron> buildCarriedBase(const CarriedPairModel& m)
{
    auto space = presburger::PresburgerSpace::getSetSpace(m.numDims);
    auto poly = std::make_unique<IntegerPolyhedron>(space);
    for (unsigned k = 0; k < m.depthP; ++k) {
        if (!addIterationDomain(*poly, carriedDimPos(m, true, k), m.boundsP[k])) {
            return std::unique_ptr<IntegerPolyhedron>(nullptr);
        }
    }
    for (unsigned k = 0; k < m.depthC; ++k) {
        if (!addIterationDomain(*poly, carriedDimPos(m, false, k), m.boundsC[k])) {
            return std::unique_ptr<IntegerPolyhedron>(nullptr);
        }
    }
    return poly;
}

// Carried overlap uses independent iteration coordinates and the same
// relaxed symbolic endpoints as the sequential intersection.
static bool addCarriedIntersection(IntegerPolyhedron& poly, const CarriedPairModel& m)
{
    // addrP - addrC - sizeC < 0 => -(addrP - addrC - sizeC) > 0
    // => addrC - addrP + sizeC > 0 => addrC - addrP + sizeC - 1 >= 0
    // Symbolic part of (addrC - addrP) is (symC - symP) = -symDiff.
    SmallVector<unsigned> pPositions;
    SmallVector<unsigned> cPositions;
    for (unsigned k = 0; k < m.depthP; ++k) {
        pPositions.push_back(carriedDimPos(m, true, k));
    }
    for (unsigned k = 0; k < m.depthC; ++k) {
        cPositions.push_back(carriedDimPos(m, false, k));
    }
    return addIntervalIntersectionRows(
        poly, m.numDims, m.coeffP, pPositions, m.coeffC, cPositions, m.constP, m.constC, m.sizeP, m.sizeC, m.symDiffLo,
        m.symDiffHi, m.invariantSymbols);
}

// Model a footprint's base byte: symbolic/allocation roots use the constant
// byte offset; absolute roots combine the absolute base with it.
static bool modelCarriedRoot(const PTOMemoryFootprint& fp, int64_t& base)
{
    if (!fp.byteSize || *fp.byteSize > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    return modelRootBase(fp, base);
}

// Relationship between the two access roots: same symbolic/allocation root,
// both absolute, or distinct proven allocations (definitely disjoint).
struct RootRelation {
    bool sameRoot = false;
    bool bothAbsolute = false;
    bool distinctProvenAllocations = false;
};

static RootRelation classifyRootRelation(const PTOMemoryFootprint& producer, const PTOMemoryFootprint& consumer)
{
    RootRelation rel;
    rel.sameRoot = (producer.rootKind == PTOMemoryRootKind::Symbolic ||
                    producer.rootKind == PTOMemoryRootKind::ProvenAllocation) &&
                   (consumer.rootKind == PTOMemoryRootKind::Symbolic ||
                    consumer.rootKind == PTOMemoryRootKind::ProvenAllocation) &&
                   producer.root == consumer.root;
    rel.distinctProvenAllocations = producer.rootKind == PTOMemoryRootKind::ProvenAllocation &&
                                    consumer.rootKind == PTOMemoryRootKind::ProvenAllocation &&
                                    producer.root != consumer.root;
    rel.bothAbsolute =
        producer.rootKind == PTOMemoryRootKind::Absolute && consumer.rootKind == PTOMemoryRootKind::Absolute;
    return rel;
}

// A symbolic byte-offset term may cancel across a loop-carried pair only when it holds the same value in
// every iteration of the carrying loops: its defining operation (or, for a block argument, the op
// introducing the region) must not be enclosed by any of them, or it must be pure arithmetic over operands
// that each satisfy the same condition — recomputing such a value in every iteration reproduces the result.
static bool valueInvariantAcrossLoops(Value value, ArrayRef<scf::ForOp> carryingLoops, unsigned depth)
{
    if (carryingLoops.empty()) {
        return true;
    }
    if (depth == 16) {
        return false;
    }
    Operation* defOp = value.getDefiningOp();
    if (!defOp) {
        // A block argument: invariant unless the op introducing its region
        // (e.g. the scf.for owning an iter_arg) is itself a carrying loop.
        Operation* introducedBy = value.getParentBlock()->getParentOp();
        if (!introducedBy) {
            return false;
        }
        return llvm::none_of(carryingLoops, [&](scf::ForOp loop) { return loop->isAncestor(introducedBy); });
    }
    if (llvm::none_of(carryingLoops, [&](scf::ForOp loop) { return loop->isAncestor(defOp); })) {
        return true;
    }
    // Defined inside a carrying loop: still value-stable when it is pure
    // arithmetic over stable operands.
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
        bool singleInputOutput = cast.getInputs().size() == 1 && cast.getOutputs().size() == 1;
        if (!singleInputOutput) {
            return false;
        }
        auto source = dyn_cast<IntegerType>(cast.getInputs().front().getType());
        auto target = dyn_cast<IntegerType>(cast.getOutputs().front().getType());
        bool sameIntegerWidth = source && target && source.getWidth() == target.getWidth();
        if (!sameIntegerWidth) {
            return false;
        }
        return valueInvariantAcrossLoops(cast.getInputs().front(), carryingLoops, depth + 1);
    }
    if (!isa<
            arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
            arith::RemSIOp, arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::TruncIOp, arith::IndexCastOp, arith::IndexCastUIOp, arith::SelectOp>(defOp)) {
        return false;
    }
    return llvm::all_of(defOp->getOperands(), [&](Value operand) {
        return valueInvariantAcrossLoops(operand, carryingLoops, depth + 1);
    });
}

static bool symbolInvariantAcrossLoops(Value symbol, ArrayRef<scf::ForOp> carryingLoops)
{
    return valueInvariantAcrossLoops(symbol, carryingLoops, 0);
}

// Resolve the loop ops of a common-loop id prefix.
static SmallVector<scf::ForOp, 2> resolveCommonLoops(const MemoryQueryContext& result, ArrayRef<unsigned> ids)
{
    SmallVector<scf::ForOp, 2> loops;
    for (unsigned id : ids) {
        for (const auto& l : result.loops) {
            if (l.id == id) {
                loops.push_back(l.op);
                break;
            }
        }
    }
    return loops;
}

static bool modelCarriedSymbols(
    const MemoryQueryContext& result, const PTOMemoryFootprint& P, const PTOMemoryFootprint& C, ArrayRef<Value> pi,
    ArrayRef<Value> pj, ArrayRef<unsigned> commonLoops, CarriedPairModel& m)
{
    SmallVector<scf::ForOp, 2> carrying = resolveCommonLoops(result, commonLoops);
    SmallVector<Value, 4> modelledIVs;
    llvm::append_range(modelledIVs, pi);
    llvm::append_range(modelledIVs, pj);
    bool symbolsValid = !P.byteOffset.lossy && !C.byteOffset.lossy;
    bool spansValid = symbolicPartRange(P.byteOffset, modelledIVs, m.symSpanLoP, m.symSpanHiP) &&
                      symbolicPartRange(C.byteOffset, modelledIVs, m.symSpanLoC, m.symSpanHiC);
    bool invariant =
        llvm::all_of(
            P.byteOffset.coefficients,
            [&](const auto& term) {
                return isModelledIV(term.first, modelledIVs) || symbolInvariantAcrossLoops(term.first, carrying);
            }) &&
        llvm::all_of(C.byteOffset.coefficients, [&](const auto& term) {
            return isModelledIV(term.first, modelledIVs) || symbolInvariantAcrossLoops(term.first, carrying);
        });
    // A bounded opaque term may vary between iterations. Its two executions
    // are independent: cancellation is only valid for invariant symbols.
    bool differenceValid = spansValid;
    if (invariant) {
        differenceValid = symbolicDifferenceRange(P.byteOffset, C.byteOffset, modelledIVs, m.symDiffLo, m.symDiffHi);
        m.invariantSymbols = boundedSymbolDifference(P.byteOffset, C.byteOffset, modelledIVs);
    } else {
        m.symDiffLo = m.symSpanLoP - m.symSpanHiC;
        m.symDiffHi = m.symSpanHiP - m.symSpanLoC;
    }
    if (!symbolsValid || !spansValid || !differenceValid) {
        return false;
    }
    return true;
}

static bool modelCarriedStorage(
    const MemoryQueryContext& result, unsigned i, unsigned j, ArrayRef<Value> pi, ArrayRef<Value> pj,
    ArrayRef<unsigned> commonLoops, CarriedPairModel& model, PTOMemoryDependenceReason& failure)
{
    const auto& accesses = result.accesses;
    const auto& producer = accesses[i].footprint;
    const auto& consumer = accesses[j].footprint;
    bool sizesAndRootsKnown = modelCarriedRoot(producer, model.constP) && modelCarriedRoot(consumer, model.constC);
    if (sizesAndRootsKnown) {
        model.sizeP = static_cast<int64_t>(*producer.byteSize);
        model.sizeC = static_cast<int64_t>(*consumer.byteSize);
    }
    RootRelation relation = classifyRootRelation(producer, consumer);
    auto carrying = resolveCommonLoops(result, commonLoops);
    const bool stableRoot = relation.sameRoot && symbolInvariantAcrossLoops(producer.root, carrying);
    const bool comparable = stableRoot || relation.bothAbsolute;
    if (!comparable || !sizesAndRootsKnown) {
        failure = !relation.sameRoot && !relation.bothAbsolute ? PTOMemoryDependenceReason::UnknownRoot :
                                                                 PTOMemoryDependenceReason::UnsupportedAccessShape;
        return false;
    }
    appendLoopCoefficients(producer, pi, model.coeffP);
    appendLoopCoefficients(consumer, pj, model.coeffC);
    const bool countersValid = convertToIterationCounters(model.coeffP, model.boundsP, model.constP) &&
                               convertToIterationCounters(model.coeffC, model.boundsC, model.constC);
    if (!countersValid) {
        failure = PTOMemoryDependenceReason::DynamicUnmodelledExpression;
        return false;
    }
    return true;
}

static bool modelCarriedExpressions(
    const MemoryQueryContext& result, unsigned i, unsigned j, ArrayRef<Value> pi, ArrayRef<Value> pj,
    ArrayRef<unsigned> commonLoops, CarriedPairModel& model, PTOMemoryDependenceReason& failure)
{
    const auto& accesses = result.accesses;
    const auto& producer = accesses[i].footprint;
    const auto& consumer = accesses[j].footprint;
    const bool expressionsValid = carriedAddressModelIsFaithful(producer, pi, model.boundsP) &&
                                  carriedAddressModelIsFaithful(consumer, pj, model.boundsC) &&
                                  modelCarriedSymbols(result, producer, consumer, pi, pj, commonLoops, model);
    if (!expressionsValid) {
        failure = PTOMemoryDependenceReason::DynamicUnmodelledExpression;
        return false;
    }
    return true;
}

// Model the roots, byte sizes, IV coefficients, and symbolic-term ranges of a carried pair.
static bool modelCarriedPairRoots(
    const MemoryQueryContext& result, unsigned i, unsigned j, ArrayRef<Value> pi, ArrayRef<Value> pj,
    ArrayRef<unsigned> commonLoops, CarriedPairModel& m, PTOMemoryDependenceReason& failure)
{
    if (!modelCarriedStorage(result, i, j, pi, pj, commonLoops, m, failure)) {
        return false;
    }
    return modelCarriedExpressions(result, i, j, pi, pj, commonLoops, m, failure);
}

static void addCarryingOrder(IntegerPolyhedron& poly, const CarriedPairModel& m, unsigned level)
{
    for (unsigned k = 0; k < level; ++k) {
        SmallVector<int64_t> eq(poly.getNumVars() + 1, 0);
        eq[carriedDimPos(m, true, k)] = -1;
        eq[carriedDimPos(m, false, k)] = 1;
        poly.addEquality(eq);
    }
    SmallVector<int64_t> gt(poly.getNumVars() + 1, 0);
    gt[carriedDimPos(m, false, level)] = 1;
    gt[carriedDimPos(m, true, level)] = -1;
    gt.back() = -1;
    poly.addInequality(gt);
}

static bool carriedAddressRangeFits(
    __int128 base, ArrayRef<int64_t> coeffs, ArrayRef<PTOLoopDomain> bounds, __int128 size)
{
    __int128 lo = base;
    __int128 hi = base;
    for (auto [coefficient, bound] : llvm::zip(coeffs, bounds)) {
        auto count = bound.maxTripCount();
        if (!count || bound.step <= 0 || coefficient == INT64_MIN) {
            return false;
        }
        __int128 delta = static_cast<__int128>(coefficient) * std::max<int64_t>(*count - 1, 0);
        lo += std::min<__int128>(delta, 0);
        hi += std::max<__int128>(delta, 0);
        if (lo < INT64_MIN || hi > INT64_MAX) {
            return false;
        }
    }
    return lo >= INT64_MIN && hi + size <= INT64_MAX;
}

static bool carriedEnvelopeFits(const CarriedPairModel& model)
{
    bool producerFits = carriedAddressRangeFits(
        static_cast<__int128>(model.constP) + model.symSpanLoP, model.coeffP, model.boundsP,
        static_cast<__int128>(model.sizeP) + (model.symSpanHiP - model.symSpanLoP));
    bool consumerFits = carriedAddressRangeFits(
        static_cast<__int128>(model.constC) + model.symSpanLoC, model.coeffC, model.boundsC,
        static_cast<__int128>(model.sizeC) + (model.symSpanHiC - model.symSpanLoC));
    return producerFits && consumerFits;
}

static bool queryCarriedEnvelope(const CarriedPairModel& model, ArrayRef<unsigned> commonLoops)
{
    for (unsigned level = 0; level < commonLoops.size(); ++level) {
        auto poly = buildCarriedBase(model);
        if (!poly || !addCarriedIntersection(*poly, model)) {
            return false;
        }
        addCarryingOrder(*poly, model, level);
        if (!poly->isIntegerEmpty()) {
            return false;
        }
    }
    return true;
}

// Only emptiness of the widened carried relation disproves a hazard.
// Varying symbols have independent execution ranges; only invariants cancel.
static bool disproveDynamicCarriedPair(
    MemoryQueryContext& result, unsigned i, unsigned j, ArrayRef<Value> pi, ArrayRef<Value> pj,
    ArrayRef<unsigned> commonLoops)
{
    CarriedPairModel model;
    model.depthP = pi.size();
    model.depthC = pj.size();
    model.numDims = model.depthP + model.depthC;
    bool boundsKnown =
        gatherEnvelopeBounds(result, pi, model.boundsP) && gatherEnvelopeBounds(result, pj, model.boundsC);
    if (!boundsKnown) {
        return false;
    }
    bool noWrap = accessArithmeticDoesNotWrap(result.accesses[i].footprint, pi, model.boundsP) &&
                  accessArithmeticDoesNotWrap(result.accesses[j].footprint, pj, model.boundsC);
    if (!noWrap) {
        return false;
    }
    // A failed probe retains the conservative Unknown carrying hazards.
    PTOMemoryDependenceReason failure = PTOMemoryDependenceReason::UnsupportedAccessShape;
    if (!modelCarriedPairRoots(result, i, j, pi, pj, commonLoops, model, failure)) {
        return false;
    }
    if (!carriedEnvelopeFits(model)) {
        return false;
    }
    return queryCarriedEnvelope(model, commonLoops);
}

// A same-iteration NoAlias is also valid across iterations only when neither
// footprint varies with any enclosing IV. Distinct allocations need no domain.
static bool disjointAcrossIterations(
    const PTOMemoryAccessOccurrence& producer, const PTOMemoryAccessOccurrence& consumer,
    ArrayRef<scf::ForOp> carrying)
{
    const auto& a = producer.footprint;
    const auto& b = consumer.footprint;
    if (classifyRootRelation(a, b).distinctProvenAllocations) {
        return true;
    }
    // Equal SSA pointer roots may denote different addresses on each iteration.
    if (!classifyRootRelation(a, b).bothAbsolute &&
        ((a.root && !symbolInvariantAcrossLoops(a.root, carrying)) ||
         (b.root && !symbolInvariantAcrossLoops(b.root, carrying)))) {
        return false;
    }
    return a.byteOffset.isConstant() && b.byteOffset.isConstant() && accessArithmeticDoesNotWrap(a, {}, {}) &&
           accessArithmeticDoesNotWrap(b, {}, {}) && aliasMemorySameIteration(a, b) == PTOMemoryAliasResult::NoAlias;
}

} // namespace

static bool validQueryContext(const PTOMemoryAccessOccurrence& producer,
                              const PTOMemoryAccessOccurrence& consumer, Operation* scope)
{
    return scope && producer.op && consumer.op && !isa<scf::ForOp>(scope) &&
           scope->isProperAncestor(producer.op) && scope->isProperAncestor(consumer.op);
}

PTOMemoryDependenceStatus pto::querySequentialMemoryOverlap(
    const PTOMemoryAccessOccurrence& producer, const PTOMemoryAccessOccurrence& consumer, Operation* scope,
    bool useEnvelope)
{
    if (!validQueryContext(producer, consumer, scope)) {
        return PTOMemoryDependenceStatus::Unknown;
    }
    auto context = buildContext(producer, consumer, scope);
    if (useEnvelope) {
        return disproveBoundedSequentialPair(context, 0, 1) ? PTOMemoryDependenceStatus::NoDependence :
                                                              PTOMemoryDependenceStatus::Unknown;
    }
    return analyzeSequentialOverlap(context, context.accesses[0], context.accesses[1]);
}

static PTOMemoryDependenceStatus queryConstantCarried(const CarriedPairModel& model, unsigned level)
{
    auto poly = buildCarriedBase(model);
    if (!poly) {
        return PTOMemoryDependenceStatus::Unknown;
    }
    addCarryingOrder(*poly, model, level);
    if (!addCarriedIntersection(*poly, model)) {
        return PTOMemoryDependenceStatus::Unknown;
    }
    return poly->isIntegerEmpty() ? PTOMemoryDependenceStatus::NoDependence : PTOMemoryDependenceStatus::MayDependence;
}

SmallVector<PTOMemoryCarriedDependence> pto::queryCarriedMemoryOverlap(
    const PTOMemoryAccessOccurrence& producer, const PTOMemoryAccessOccurrence& consumer, Operation* scope)
{
    if (!validQueryContext(producer, consumer, scope)) {
        return {{scf::ForOp(), PTOMemoryDependenceStatus::Unknown, PTOMemoryDependenceReason::InvalidContext}};
    }
    auto context = buildContext(producer, consumer, scope);
    auto common = getCommonLoopPrefix(context.accesses[0].loopNest, context.accesses[1].loopNest);
    SmallVector<PTOMemoryCarriedDependence> answers;
    auto carrying = resolveCommonLoops(context, common);
    bool noCommonDomain = common.empty() || disjointAcrossIterations(producer, consumer, carrying);
    if (noCommonDomain) {
        return answers;
    }
    auto pi = collectIVs(producer.op, scope);
    auto pj = collectIVs(consumer.op, scope);
    CarriedPairModel model;
    model.depthP = pi.size();
    model.depthC = pj.size();
    model.numDims = model.depthP + model.depthC;
    bool producerConstant = gatherConstBounds(context, pi, model.boundsP);
    bool constant = gatherConstBounds(context, pj, model.boundsC) && producerConstant;
    PTOMemoryDependenceReason failure = PTOMemoryDependenceReason::UnsupportedAccessShape;
    bool modelled = constant && modelCarriedPairRoots(context, 0, 1, pi, pj, common, model, failure);
    bool disjoint = !constant && disproveDynamicCarriedPair(context, 0, 1, pi, pj, common);
    for (auto [level, id] : llvm::enumerate(common)) {
        scf::ForOp loop = context.loops[id].op;
        auto status = PTOMemoryDependenceStatus::Unknown;
        auto reason = constant ? failure : PTOMemoryDependenceReason::DynamicLoopBounds;
        if (modelled) {
            status = queryConstantCarried(model, level);
            reason = PTOMemoryDependenceReason::DynamicUnmodelledExpression;
        } else if (!constant) {
            auto bound = getLoopEnvelope(loop);
            auto trip = bound.maxTripCount();
            if (disjoint || (trip && *trip <= 1)) {
                status = PTOMemoryDependenceStatus::NoDependence;
            }
        }
        if (status != PTOMemoryDependenceStatus::NoDependence) {
            answers.push_back({loop, status, reason});
        }
    }
    return answers;
}
