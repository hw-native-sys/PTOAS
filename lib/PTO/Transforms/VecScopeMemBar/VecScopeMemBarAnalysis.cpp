// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VecScopeMemBarAnalysis.cpp --------------------------------------===//
//
// Builds the vecscope schedule, collects access occurrences and existing
// barriers, and enumerates RAW/WAW/WAR hazards: same-iteration, inner- and
// outer-loop-carried via Presburger dependence relations. The value walk here
// resolves pending WAR obligations, including loop yields; candidate removal
// belongs to Redundancy.cpp and uses its stricter straight-line proof. Does not
// mutate the IR.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarAnalysis.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;
using mlir::presburger::IntegerPolyhedron;

namespace {

// A defining op that writes UB memory is another store: the reverse walk
// must stop there, since relaying through it would mistake an unrelated
// store for the value relay.
static bool definesUBWrite(Operation* defOp)
{
    auto iface = dyn_cast<MemoryEffectOpInterface>(defOp);
    if (!iface) {
        return false;
    }
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> eff;
    iface.getEffects(eff);
    return llvm::any_of(eff, [](const auto& e) {
        return isa<MemoryEffects::Write>(e.getEffect()) && e.getValue() &&
               vecscopemembar::isUBBackedType(e.getValue().getType());
    });
}

// `sink` may be a result of an scf.for. Result i corresponds to region
// iter-arg i, whose value at each iteration is the i-th operand of the
// body's scf.yield. Return that operand so a dependence carried out of the
// loop (e.g. a reduction result feeding a later store) is followed back into
// the loop body, instead of stopping at the loop's input operands
// (lb/ub/step/init) which never see the body's loads. Nullopt when `sink` is
// not a for-result carrying a region iter-arg.
static std::optional<Value> forResultYieldedValue(Value sink)
{
    auto forOp = dyn_cast<scf::ForOp>(sink.getDefiningOp());
    if (!forOp) {
        return std::nullopt;
    }
    unsigned k = 0;
    for (Value r : forOp.getResults()) {
        if (r == sink) {
            break;
        }
        ++k;
    }
    if (k >= forOp.getNumRegionIterArgs()) {
        return std::nullopt;
    }
    return forOp.getBody()->getTerminator()->getOperand(k);
}

// Block argument: only scf.for iter_args are relayed, via the yield operand.
// Nullopt when `sink` is not a carried iter argument.
static std::optional<Value> iterArgYieldedValue(Value sink)
{
    auto ba = dyn_cast<BlockArgument>(sink);
    if (!ba) {
        return std::nullopt;
    }
    auto forOp = ba.getOwner() ? dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp()) : scf::ForOp();
    if (!forOp) {
        return std::nullopt;
    }
    unsigned arg = ba.getArgNumber();
    // arg 0 is the loop IV; iter_args start at 1.
    if (arg == 0 || arg > forOp.getRegion().getNumArguments() - 1) {
        return std::nullopt;
    }
    return forOp.getBody()->getTerminator()->getOperand(arg - 1);
}

static bool reverseReaches(Value sink, Value target, DenseSet<Value>& visited, unsigned depth, unsigned maxDepth)
{
    if (sink == target) {
        return true;
    }
    if (depth > maxDepth) {
        return false;
    }
    if (!visited.insert(sink).second) {
        return false;
    }

    if (Operation* defOp = sink.getDefiningOp()) {
        if (definesUBWrite(defOp)) {
            return false; // relay through another store: not a value relay
        }
        if (auto yielded = forResultYieldedValue(sink)) {
            if (reverseReaches(*yielded, target, visited, depth + 1, maxDepth)) {
                return true;
            }
        }
        for (Value op : defOp->getOperands()) {
            if (reverseReaches(op, target, visited, depth + 1, maxDepth)) {
                return true;
            }
        }
        return false;
    }

    auto yielded = iterArgYieldedValue(sink);
    if (!yielded) {
        return false;
    }
    return reverseReaches(*yielded, target, visited, depth + 1, maxDepth);
}

static bool valueDependsOn(Value sink, Value target)
{
    DenseSet<Value> visited;
    return reverseReaches(sink, target, visited, 0, 32);
}

// Whether any stored value depends on any loaded value through the value
// relay used to resolve pending WAR hazards.
static bool anyStoredValueDependsOnLoaded(ArrayRef<Value> storedVals, ArrayRef<Value> loadedVals)
{
    for (Value sv : storedVals) {
        for (Value lv : loadedVals) {
            if (valueDependsOn(sv, lv)) {
                return true;
            }
        }
    }
    return false;
}

static std::optional<MemBarKind> hazardKind(const AccessOccurrence& producer, const AccessOccurrence& consumer)
{
    bool producerVector = producer.unit == VecScopeAccessUnit::Vector;
    bool consumerVector = consumer.unit == VecScopeAccessUnit::Vector;
    // Scalar UB accesses are issued by the ordered scalar pipeline.  The
    // automatic vecscope pass only needs barriers when a vector pipeline is
    // involved; scalar-to-scalar ordering does not require a membar.
    if (!producerVector && !consumerVector) {
        return std::nullopt;
    }
    if (producer.kind == VecScopeAccessKind::Store && consumer.kind == VecScopeAccessKind::Load) {
        if (producerVector) {
            return consumerVector ? MemBarKind::VST_VLD : MemBarKind::VST_LD;
        }
        return MemBarKind::ST_VLD;
    }
    if (producer.kind == VecScopeAccessKind::Store && consumer.kind == VecScopeAccessKind::Store) {
        if (producerVector) {
            return consumerVector ? MemBarKind::VST_VST : MemBarKind::VST_ST;
        }
        return MemBarKind::ST_VST;
    }
    if (producer.kind == VecScopeAccessKind::Load && consumer.kind == VecScopeAccessKind::Store) {
        if (producerVector) {
            return consumerVector ? MemBarKind::VLD_VST : MemBarKind::VLD_ST;
        }
        return MemBarKind::LD_VST;
    }
    return std::nullopt; // RAR
}

// Compare two operations in the lexical order of the vecscope, including
// operations nested in loops. `Operation::isBeforeInBlock` only accepts
// siblings, while the dependency graph also needs to reason about a barrier
// in one region ordering an access in a nested or sibling region.
static bool isLexicallyBefore(Operation* lhs, Operation* rhs, Operation* scope)
{
    if (!lhs || !rhs || lhs == rhs) {
        return false;
    }

    SmallVector<Operation*, 8> lhsAncestors;
    for (Operation* op = lhs; op; op = op->getParentOp()) {
        lhsAncestors.push_back(op);
        if (op == scope) {
            break;
        }
    }
    SmallVector<Operation*, 8> rhsAncestors;
    for (Operation* op = rhs; op; op = op->getParentOp()) {
        rhsAncestors.push_back(op);
        if (op == scope) {
            break;
        }
    }

    Operation* common = nullptr;
    for (Operation* op : lhsAncestors) {
        if (llvm::is_contained(rhsAncestors, op)) {
            common = op;
            break;
        }
    }
    if (!common) {
        return false;
    }

    auto childUnder = [&](Operation* op) {
        while (op && op->getParentOp() != common) {
            op = op->getParentOp();
        }
        return op;
    };
    Operation* lhsChild = childUnder(lhs);
    Operation* rhsChild = childUnder(rhs);
    if (!lhsChild || !rhsChild || lhsChild->getBlock() != rhsChild->getBlock()) {
        return false;
    }
    return lhsChild->isBeforeInBlock(rhsChild);
}

// Collect enclosing loop IVs (outermost first) within the vecscope boundary.
// Memory-bar analysis is scoped to a single `pto.vecscope`: any loop outside
// that scope is invisible to it, so the walk stops at `scope` (exclusive).
// Stopping at the scope keeps the invariant that an op directly inside a
// vecscope with no inner loops has an empty IV set, which the same-iteration
// overlap model relies on to compare constant addresses directly.
static SmallVector<Value, 2> collectIVs(Operation* op, Operation* scope)
{
    SmallVector<Value, 2> ivs;
    for (Operation* cur = op->getParentOp(); cur && cur != scope; cur = cur->getParentOp()) {
        if (auto forOp = dyn_cast<scf::ForOp>(cur)) {
            ivs.push_back(forOp.getInductionVar());
        }
    }
    std::reverse(ivs.begin(), ivs.end());
    return ivs;
}

// Try to evaluate `v` as a constant int64.
static std::optional<int64_t> getConstantInt(Value v)
{
    APInt c;
    if (matchPattern(v, m_ConstantInt(&c))) {
        return c.getSExtValue();
    }
    return std::nullopt;
}

// Constant lower/upper/step for an scf.for, else nullopt (dynamic).
struct ConstLoopBounds {
    int64_t lb = 0;
    int64_t ub = 0;
    int64_t step = 1;
    bool valid = false;
    // Number of iterations = (ub - lb + step - 1) / step (>=0).
    std::optional<int64_t> tripCount() const
    {
        if (!valid || step == 0) {
            return std::nullopt;
        }
        int64_t diff = ub - lb;
        if (diff <= 0) {
            return 0;
        }
        return (diff + (step > 0 ? step : -step) - 1) / (step > 0 ? step : -step);
    }
};
static ConstLoopBounds getConstBounds(scf::ForOp forOp)
{
    ConstLoopBounds b;
    auto lb = getConstantInt(forOp.getLowerBound());
    auto ub = getConstantInt(forOp.getUpperBound());
    auto step = getConstantInt(forOp.getStep());
    if (lb && ub && step && *step != 0) {
        b.lb = *lb;
        b.ub = *ub;
        b.step = *step;
        b.valid = true;
    }
    return b;
}

// Whether any loop enclosing `op` within the vecscope `scope` has non-constant
// bounds. Loops outside the scope are invisible to the analysis (see
// `collectIVs`).
static bool hasDynamicLoopDomain(Operation* op, Operation* scope)
{
    for (Operation* cursor = op->getParentOp(); cursor && cursor != scope; cursor = cursor->getParentOp()) {
        if (auto forOp = dyn_cast<scf::ForOp>(cursor)) {
            if (!getConstBounds(forOp).valid) {
                return true;
            }
        }
    }
    return false;
}

static std::optional<unsigned> loopIdForIV(const VecScopeMemBarAnalysisResult& r, Value iv)
{
    for (const auto& l : r.loops) {
        if (l.inductionVar == iv) {
            return l.id;
        }
    }
    return std::nullopt;
}

static void recordUnknown(
    VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, MemBarKind kind, DependenceReason reason)
{
    result.unknownDependences.push_back({i, j, kind, reason});
}

// Dynamic loop bounds cannot currently be represented by the constant-domain
// model below. Preserve correctness by covering every shared carrying loop;
// other modelling failures use the same conservative carrying-loop coverage.
static void emitDynamicBoundsHazards(
    VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, MemBarKind kind, ArrayRef<unsigned> loopIds)
{
    for (auto [index, loopId] : llvm::enumerate(loopIds)) {
        MemoryHazard h;
        h.id = result.hazards.size();
        h.producer = i;
        h.consumer = j;
        h.kind = kind;
        h.scope =
            index + 1 == loopIds.size() ? VecScopeHazardScope::InnerLoopCarried : VecScopeHazardScope::OuterLoopCarried;
        h.carryingLoop = loopId;
        h.distance.positive = true;
        result.hazards.push_back(h);
    }
}

static std::optional<unsigned> innermostLoopId(const VecScopeMemBarAnalysisResult& r, Value iv)
{
    return loopIdForIV(r, iv);
}
static std::optional<unsigned> outermostLoopId(const VecScopeMemBarAnalysisResult& r, Value iv)
{
    return loopIdForIV(r, iv);
}

// Build a flat schedule tree: root Sequence, with loops as children carrying
// their own Sequence body. `schedulePath` is the child-index path from root.
static void buildSchedule(Operation* scope, VecScopeMemBarAnalysisResult& result)
{
    // Root sequence node.
    VecScopeScheduleNode root;
    root.kind = VecScopeNodeKind::Sequence;
    root.op = scope;
    result.schedule.push_back(root);
    unsigned loopId = 0;
    scope->walk([&](scf::ForOp forOp) {
        VecScopeLoopInfo info;
        info.id = loopId++;
        info.op = forOp;
        unsigned depth = 0;
        for (Operation* p = forOp->getParentOp(); p; p = p->getParentOp()) {
            if (isa<scf::ForOp>(p) || p == scope) {
                depth++;
            }
        }
        info.depth = depth;
        info.inductionVar = forOp.getInductionVar();
        info.lowerBound = forOp.getLowerBound();
        info.upperBound = forOp.getUpperBound();
        info.step = forOp.getStep();
        result.loops.push_back(info);
    });
}

static SmallVector<unsigned, 2> commonLoopPrefix(const AccessOccurrence& a, const AccessOccurrence& b)
{
    SmallVector<unsigned, 2> common;
    for (auto [lhs, rhs] : llvm::zip(a.loopNest, b.loopNest)) {
        if (lhs != rhs) {
            break;
        }
        common.push_back(lhs);
    }
    return common;
}

// Presburger loop-carried dependence relation

static bool addIterationDomain(IntegerPolyhedron& p, unsigned pos, const ConstLoopBounds& b)
{
    auto trip = b.tripCount();
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
static scf::ForOp resolveLoopForIV(const VecScopeMemBarAnalysisResult& r, Value iv)
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
    const VecScopeMemBarAnalysisResult& result, ArrayRef<Value> ivs, SmallVectorImpl<ConstLoopBounds>& bounds)
{
    for (Value iv : ivs) {
        scf::ForOp forOp = resolveLoopForIV(result, iv);
        if (!forOp) {
            return false;
        }
        bounds.push_back(getConstBounds(forOp));
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
    SmallVectorImpl<int64_t>& coeffs, ArrayRef<ConstLoopBounds> bounds, int64_t& constant)
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
static bool absoluteBaseComparable(const VecScopeMemoryFootprint& fp)
{
    return fp.absoluteBase && *fp.absoluteBase <= static_cast<uint64_t>(INT64_MAX);
}

// Early feasibility checks for the sequential-overlap model: closed byte
// ranges on both sides, no forced-MayAlias footprint, and comparable roots.
static bool sequentialRootsComparable(const VecScopeMemoryFootprint& P, const VecScopeMemoryFootprint& C)
{
    // Need closed byte ranges on both sides. forcesMayAlias footprints (e.g.
    // vsstb) with an open upper bound cannot be disproven here because their
    // upper bound is unknown.
    if (!P.byteSize || !C.byteSize || !P.byteOffset.exact || !C.byteOffset.exact) {
        return false;
    }
    if (P.forcesMayAlias || C.forcesMayAlias) {
        return false;
    }
    // Both must resolve to a comparable root: same symbolic/allocation root, or
    // both absolute (absolute bases are comparable regardless of SSA identity).
    bool bothAbsolute = P.rootKind == MemoryRootKind::Absolute && C.rootKind == MemoryRootKind::Absolute;
    bool sameNonAbsoluteRoot =
        (P.rootKind == MemoryRootKind::Symbolic || P.rootKind == MemoryRootKind::ProvenAllocation) &&
        (C.rootKind == MemoryRootKind::Symbolic || C.rootKind == MemoryRootKind::ProvenAllocation) && P.root == C.root;
    if (!bothAbsolute && !sameNonAbsoluteRoot) {
        return false;
    }
    if (bothAbsolute && (!absoluteBaseComparable(P) || !absoluteBaseComparable(C))) {
        return false;
    }
    // Distinct proven allocations are already NoAlias at `aliasSameIteration`;
    // reaching here means roots are comparable, so continue.
    return true;
}

// Resolved loop-nest layout of a sequential-region access pair. Dual-set
// dims are [p0.., c0..]; single-set dims are [v0..] with v shared.
struct SequentialOverlapLayout {
    SmallVector<Value, 2> pi;        // producer IVs (outermost first)
    SmallVector<Value, 2> pj;        // consumer IVs (outermost first)
    SmallVector<unsigned, 2> common; // shared loop-id prefix
    SmallVector<ConstLoopBounds> boundsP;
    SmallVector<ConstLoopBounds> boundsC;
    // single-nest iff one IV set is a prefix of the other (incl. equal/empty).
    bool singleNest = false;
    unsigned numDims = 0;
};

// Resolve the IV layout for the pair. Handled shapes:
//  (a) pi == pj (same loop nest): single IV set, shared IVs identical.
//  (b) one IV set is a proper prefix of the other (one access further out,
//      e.g. a top-level store feeding a load inside a loop, or the reverse):
//      single IV set = the longer of pi/pj, the shorter side's coefficients
//      read against that set (0 for IVs it does not carry).
//  (c) pi and pj share a common prefix but diverge (producer and consumer in
//      sibling loops): dual IV sets [pi..., pj...], shared prefix constrained
//      equal, non-shared IVs independent.
// Returns false when a nest is too deep or a loop has a non-constant domain.
static bool resolveSequentialIVLayout(
    const VecScopeMemBarAnalysisResult& result, const AccessOccurrence& prod, const AccessOccurrence& cons,
    SequentialOverlapLayout& layout)
{
    layout.pi = collectIVs(prod.op, result.scope);
    layout.pj = collectIVs(cons.op, result.scope);
    if (layout.pi.size() > 2 || layout.pj.size() > 2) {
        return false;
    }
    layout.common = commonLoopPrefix(prod, cons);
    bool prefixP = layout.pi.size() <= layout.pj.size() && layout.common.size() == layout.pi.size();
    bool prefixC = layout.pj.size() <= layout.pi.size() && layout.common.size() == layout.pj.size();
    layout.singleNest = prefixP || prefixC;
    if (!gatherConstBounds(result, layout.pi, layout.boundsP)) {
        return false;
    }
    if (!gatherConstBounds(result, layout.pj, layout.boundsC)) {
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
// shared-prefix same-iteration equality. Returns null when a domain cannot
// be modelled.
static std::unique_ptr<IntegerPolyhedron> buildSequentialPolyhedron(
    const VecScopeMemBarAnalysisResult& result, const SequentialOverlapLayout& layout)
{
    unsigned depthP = layout.pi.size();
    unsigned depthC = layout.pj.size();
    auto space = presburger::PresburgerSpace::getSetSpace(layout.numDims);
    auto poly = std::make_unique<IntegerPolyhedron>(space);
    if (layout.singleNest) {
        // Coefficients are read against the longer IV set; use whichever side is
        // longer as the coordinate basis.
        const auto& singleIVs = (depthP >= depthC) ? layout.pi : layout.pj;
        for (unsigned k = 0; k < singleIVs.size(); ++k) {
            auto forOp = resolveLoopForIV(result, singleIVs[k]);
            if (!addIterationDomain(*poly, k, getConstBounds(forOp))) {
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
static bool sequentialRootBase(const VecScopeMemoryFootprint& fp, int64_t& base)
{
    if (fp.rootKind == MemoryRootKind::Symbolic || fp.rootKind == MemoryRootKind::ProvenAllocation) {
        base = fp.byteOffset.constant;
        return true;
    }
    if (fp.rootKind != MemoryRootKind::Absolute || !fp.absoluteBase ||
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

// Intersection constraints (same shape as the loop-carried addIntersection):
//   addrP < addrC + sizeC  =>  addrC - addrP + sizeC - 1 >= 0
//   addrC < addrP + sizeP  =>  addrP - addrC + sizeP - 1 >= 0
// If the conjunction is empty, the intervals never overlap -> NoAlias.
static bool addSequentialIntersection(
    IntegerPolyhedron& poly, const SequentialOverlapLayout& layout, ArrayRef<int64_t> coeffP, ArrayRef<int64_t> coeffC,
    int64_t constP, int64_t constC, int64_t sizeP, int64_t sizeC)
{
    unsigned depthP = layout.pi.size();
    unsigned depthC = layout.pj.size();
    SmallVector<int64_t> row(layout.numDims + 1, 0);
    for (unsigned k = 0; k < depthP; ++k) {
        row[k] -= coeffP[k];
    }
    for (unsigned k = 0; k < depthC; ++k) {
        row[consumerDim(layout, k)] += coeffC[k];
    }
    __int128 rc = static_cast<__int128>(constC) - constP + sizeC - 1;
    if (rc < INT64_MIN || rc > INT64_MAX) {
        return false;
    }
    row.back() = static_cast<int64_t>(rc);
    poly.addInequality(row);

    SmallVector<int64_t> row2(layout.numDims + 1, 0);
    for (unsigned k = 0; k < depthP; ++k) {
        row2[k] += coeffP[k];
    }
    for (unsigned k = 0; k < depthC; ++k) {
        row2[consumerDim(layout, k)] -= coeffC[k];
    }
    rc = static_cast<__int128>(constP) - constC + sizeP - 1;
    if (rc < INT64_MIN || rc > INT64_MAX) {
        return false;
    }
    row2.back() = static_cast<int64_t>(rc);
    poly.addInequality(row2);
    return true;
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
static DependenceStatus analyzeSequentialOverlap(
    const VecScopeMemBarAnalysisResult& result, const AccessOccurrence& prod, const AccessOccurrence& cons)
{
    const auto& P = prod.footprint;
    const auto& C = cons.footprint;
    if (!sequentialRootsComparable(P, C)) {
        return DependenceStatus::Unknown;
    }

    SequentialOverlapLayout layout;
    if (!resolveSequentialIVLayout(result, prod, cons, layout)) {
        return DependenceStatus::Unknown;
    }
    auto poly = buildSequentialPolyhedron(result, layout);
    if (!poly) {
        return DependenceStatus::Unknown;
    }

    int64_t constP = 0;
    int64_t constC = 0;
    if (!sequentialRootBase(P, constP) || !sequentialRootBase(C, constC)) {
        return DependenceStatus::Unknown;
    }
    int64_t sizeP = static_cast<int64_t>(*P.byteSize);
    int64_t sizeC = static_cast<int64_t>(*C.byteSize);

    // Producer coefficients in the producer's IV coordinates, consumer
    // coefficients in the consumer's. In the single-nest layout the producer's
    // IVs are a prefix of the consumer's, so producer coefficients for IVs it
    // does not carry are 0 (getCoeff returns 0 for an absent IV).
    SmallVector<int64_t, 2> coeffP;
    SmallVector<int64_t, 2> coeffC;
    for (unsigned k = 0; k < layout.pi.size(); ++k) {
        coeffP.push_back(P.byteOffset.getCoeff(layout.pi[k]));
    }
    for (unsigned k = 0; k < layout.pj.size(); ++k) {
        coeffC.push_back(C.byteOffset.getCoeff(layout.pj[k]));
    }
    if (!convertToIterationCounters(coeffP, layout.boundsP, constP) ||
        !convertToIterationCounters(coeffC, layout.boundsC, constC)) {
        return DependenceStatus::Unknown;
    }

    if (!addSequentialIntersection(*poly, layout, coeffP, coeffC, constP, constC, sizeP, sizeC)) {
        return DependenceStatus::Unknown;
    }

    return poly->isIntegerEmpty() ? DependenceStatus::NoDependence : DependenceStatus::ProvenDependence;
}

// A mem_bar is a vector micro-op and must remain inside the consumer's
// vector scope. For a dependence crossing sibling vecscopes, anchor at the
// first operation in the consumer scope rather than before the scope
// container in the parent block.
static Operation* anchorInsideConsumerScope(Operation* consumer, Operation* c)
{
    Region& body = c->getRegion(0);
    if (!body.empty() && !body.front().empty()) {
        Operation* anchor = &body.front().front();
        while (isa<pto::MemBarOp>(anchor) && anchor->getNextNode()) {
            anchor = anchor->getNextNode();
        }
        return anchor;
    }
    return consumer;
}

// Record enclosing loop ids (outermost first).
static void recordEnclosingLoopNest(
    const VecScopeMemBarAnalysisResult& result, ArrayRef<Value> ivs, AccessOccurrence& occ)
{
    for (Value iv : ivs) {
        for (const auto& l : result.loops) {
            if (l.inductionVar == iv) {
                occ.loopNest.push_back(l.id);
                break;
            }
        }
    }
}

// Collect accesses in lexical order. Returns false when any access cannot be
// modelled.
static bool collectAccessOccurrences(Operation* scope, VecScopeMemBarAnalysisResult& result)
{
    bool anyFailed = false;
    unsigned order = 0;
    scope->walk([&](Operation* op) {
        if (!isUBMemoryOp(op)) {
            return;
        }
        AccessOccurrence occ;
        occ.op = op;
        occ.lexicalOrder = order++;
        auto ivs = collectIVs(op, scope);
        auto maybe = buildAccessDescriptor(op, ivs);
        if (failed(maybe)) {
            anyFailed = true;
            return;
        }
        occ.kind = (*maybe).kind;
        occ.unit = (*maybe).unit;
        occ.footprint = footprintFromDescriptor(*maybe, ivs);
        recordEnclosingLoopNest(result, ivs, occ);
        result.accesses.push_back(occ);
    });
    return !anyFailed;
}

// The analysis-result loop id of the loop whose backedge the barrier sits
// on, if any.
static std::optional<unsigned> latchLoopOfBarrier(pto::MemBarOp mb, const VecScopeMemBarAnalysisResult& result)
{
    auto fl = mb->getParentOfType<scf::ForOp>();
    if (!fl) {
        return std::nullopt;
    }
    for (const auto& l : result.loops) {
        if (l.op == fl) {
            return l.id;
        }
    }
    return std::nullopt;
}

// Collect existing pto.mem_bar ops.
static void collectExistingBarriers(Operation* scope, VecScopeMemBarAnalysisResult& result)
{
    scope->walk([&](pto::MemBarOp mb) {
        ExistingBarrier b;
        b.op = mb.getOperation();
        b.kind = mb.getKind().getKind();
        b.phase = mb->getParentOfType<scf::ForOp>() ? ExistingBarrier::LoopLatch : ExistingBarrier::SameIteration;
        b.latchLoop = latchLoopOfBarrier(mb, result);
        result.existingBarriers.push_back(b);
    });
}

// --- Sequential-region RAW/WAW dependences ---
static void enumerateSequentialHazards(
    VecScopeMemBarAnalysisResult& result, Operation* scope, SmallVectorImpl<std::pair<unsigned, unsigned>>& pendingWAR)
{
    const auto& A = result.accesses;
    const unsigned N = A.size();
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = i + 1; j < N; ++j) {
            auto kind = hazardKind(A[i], A[j]);
            if (!kind) {
                continue;
            }

            bool dynamicDomain = hasDynamicLoopDomain(A[i].op, scope) || hasDynamicLoopDomain(A[j].op, scope);
            if (aliasSameIteration(A[i].footprint, A[j].footprint) == VecScopeAliasResult::NoAlias) {
                continue;
            }
            DependenceStatus status =
                dynamicDomain ? DependenceStatus::ProvenDependence : analyzeSequentialOverlap(result, A[i], A[j]);
            if (status == DependenceStatus::NoDependence) {
                continue;
            }
            if (status == DependenceStatus::Unknown) {
                recordUnknown(result, i, j, *kind, DependenceReason::DynamicUnmodelledExpression);
                // Missing footprint information is not a proof of independence.
                // In particular, gather and unaligned loads must observe earlier
                // UB stores even when their exact byte range cannot be modelled.
            }
            // Delay load-to-store materialization until all RAW barriers are known. A WAR
            // chain may be indirect:
            //
            //   load A -> value -> store B --VST_VLD--> load C -> value -> store A
            //
            // The first store-to-load edge is supplied by SSA, while VST_VLD orders
            // every store before the later load. Checking only load A's direct
            // value flow into the final store A would incorrectly add VLD_VST.
            if (A[i].kind == VecScopeAccessKind::Load && A[j].kind == VecScopeAccessKind::Store) {
                pendingWAR.push_back({i, j});
                continue;
            }
            MemoryHazard h;
            h.id = result.hazards.size();
            h.producer = i;
            h.consumer = j;
            h.kind = *kind;
            h.scope = VecScopeHazardScope::SameIteration;
            h.sameIterationAnchor = findSameIterationAnchor(A[i].op, A[j].op, scope);
            if (!h.sameIterationAnchor) {
                h.sameIterationAnchor = A[j].op;
            }
            result.hazards.push_back(h);
        }
    }
}

// WAR-ordering graph: edge (u, v) means access u is already ordered before
// access v by SSA data flow or an existing/scheduled VST_VLD barrier.
using OrderingEdges = SmallVector<SmallVector<unsigned, 4>, 16>;

static void addOrderingEdge(OrderingEdges& edges, unsigned from, unsigned to)
{
    if (!llvm::is_contained(edges[from], to)) {
        edges[from].push_back(to);
    }
}

// Forward reachability over the ordering graph.
static bool reachesViaEdges(const OrderingEdges& edges, unsigned from, unsigned to)
{
    SmallVector<bool, 16> reachable(edges.size(), false);
    SmallVector<unsigned, 16> worklist;
    reachable[from] = true;
    worklist.push_back(from);
    while (!worklist.empty()) {
        unsigned current = worklist.pop_back_val();
        for (unsigned next : edges[current]) {
            if (reachable[next]) {
                continue;
            }
            reachable[next] = true;
            worklist.push_back(next);
        }
    }
    return reachable[to];
}

// SSA data dependence: a vector load is ordered before a later store if
// one of the store's payload values depends on one of the load results.
static void addSSADataEdges(const VecScopeMemBarAnalysisResult& result, OrderingEdges& edges)
{
    const auto& A = result.accesses;
    const unsigned N = A.size();
    for (unsigned load = 0; load < N; ++load) {
        if (A[load].kind != VecScopeAccessKind::Load) {
            continue;
        }
        SmallVector<Value, 2> loadedVals = getLoadedValues(A[load].op);
        if (loadedVals.empty()) {
            continue;
        }
        for (unsigned store = load + 1; store < N; ++store) {
            if (A[store].kind != VecScopeAccessKind::Store) {
                continue;
            }
            SmallVector<Value, 2> storedVals = getStoredValues(A[store].op);
            if (anyStoredValueDependsOnLoaded(storedVals, loadedVals)) {
                addOrderingEdge(edges, load, store);
            }
        }
    }
}

// Every store of `storeUnit` before the cut is ordered before every load of
// `loadUnit` after it: the barrier's producer unit selects which stores the
// cut orders, and its consumer unit selects which loads.
static void addCutOrderingEdges(
    ArrayRef<AccessOccurrence> A, unsigned cut, VecScopeAccessUnit storeUnit, VecScopeAccessUnit loadUnit,
    OrderingEdges& edges)
{
    for (unsigned store = 0; store < cut; ++store) {
        if (A[store].kind != VecScopeAccessKind::Store || A[store].unit != storeUnit) {
            continue;
        }
        for (unsigned load = cut; load < A.size(); ++load) {
            if (A[load].kind == VecScopeAccessKind::Load && A[load].unit == loadUnit) {
                addOrderingEdge(edges, store, load);
            }
        }
    }
}

// A store-to-load barrier is placed before its consumer load. It orders
// all stores of the producer's unit preceding that cut before later loads
// of the consumer's unit, not merely the particular store/load pair that
// caused the barrier.
static void addHazardCutEdges(const VecScopeMemBarAnalysisResult& result, OrderingEdges& edges)
{
    const auto& A = result.accesses;
    const unsigned N = A.size();
    for (const auto& h : result.hazards) {
        if (h.scope != VecScopeHazardScope::SameIteration ||
            (h.kind != MemBarKind::VST_VLD && h.kind != MemBarKind::VST_LD && h.kind != MemBarKind::ST_VLD &&
             h.kind != MemBarKind::ST_LD) ||
            h.consumer >= N) {
            continue;
        }
        VecScopeAccessUnit storeUnit = h.kind == MemBarKind::VST_VLD || h.kind == MemBarKind::VST_LD ?
                                           VecScopeAccessUnit::Vector :
                                           VecScopeAccessUnit::Scalar;
        VecScopeAccessUnit loadUnit = h.kind == MemBarKind::VST_VLD || h.kind == MemBarKind::ST_VLD ?
                                          VecScopeAccessUnit::Vector :
                                          VecScopeAccessUnit::Scalar;
        addCutOrderingEdges(A, h.consumer, storeUnit, loadUnit, edges);
    }
}

// Every store before the barrier is ordered before every load after it.
// Both units are derived from the barrier kind's family: the family's
// first letter is the store unit, its second letter the load unit.
static void addBarrierSpanEdges(
    ArrayRef<AccessOccurrence> A, Operation* scope, const ExistingBarrier& barrier, OrderingEdges& edges)
{
    const unsigned N = A.size();
    for (unsigned store = 0; store < N; ++store) {
        if (A[store].kind != VecScopeAccessKind::Store ||
            A[store].unit != (static_cast<unsigned>(barrier.kind) / 4 < 2 ? VecScopeAccessUnit::Vector :
                                                                            VecScopeAccessUnit::Scalar) ||
            !isLexicallyBefore(A[store].op, barrier.op, scope)) {
            continue;
        }
        for (unsigned load = 0; load < N; ++load) {
            if (A[load].kind != VecScopeAccessKind::Load ||
                A[load].unit != (static_cast<unsigned>(barrier.kind) / 4 % 2 == 0 ? VecScopeAccessUnit::Vector :
                                                                                    VecScopeAccessUnit::Scalar) ||
                !isLexicallyBefore(barrier.op, A[load].op, scope)) {
                continue;
            }
            addOrderingEdge(edges, store, load);
        }
    }
}

// Existing store-to-load barriers (directed or ALL) provide the same
// ordering edge as generated ones. This matters when the input already
// contains the first half of a chain: a later WAR pair must be recognized
// as ordered even though no new RAW hazard was materialized for that
// barrier in this run.
static void addExistingBarrierEdges(const VecScopeMemBarAnalysisResult& result, Operation* scope, OrderingEdges& edges)
{
    const auto& A = result.accesses;
    for (const auto& barrier : result.existingBarriers) {
        if (barrier.kind != MemBarKind::VST_VLD && barrier.kind != MemBarKind::VST_LD &&
            barrier.kind != MemBarKind::ST_VLD && barrier.kind != MemBarKind::ST_LD &&
            barrier.kind != MemBarKind::VS_ALL && barrier.kind != MemBarKind::SV_ALL &&
            barrier.kind != MemBarKind::SS_ALL && barrier.kind != MemBarKind::VV_ALL) {
            continue;
        }
        addBarrierSpanEdges(A, scope, barrier, edges);
    }
}

// Resolve WAR hazards over the combined SSA + VST_VLD ordering graph. The
// graph is deliberately built after the first pair pass so it contains all
// same-iteration RAW hazards, including hazards whose producer/consumer
// lexical order is later than a pending WAR pair.
static void resolvePendingWARHazards(
    VecScopeMemBarAnalysisResult& result, Operation* scope, SmallVectorImpl<std::pair<unsigned, unsigned>>& pendingWAR)
{
    if (pendingWAR.empty()) {
        return;
    }
    const auto& A = result.accesses;
    const unsigned N = A.size();
    OrderingEdges edges(N);
    addSSADataEdges(result, edges);
    addHazardCutEdges(result, edges);
    addExistingBarrierEdges(result, scope, edges);

    for (const auto& [load, store] : pendingWAR) {
        if (reachesViaEdges(edges, load, store)) {
            continue;
        }
        MemoryHazard h;
        h.id = result.hazards.size();
        h.producer = load;
        h.consumer = store;
        h.kind = hazardKind(A[load], A[store]).value();
        h.scope = VecScopeHazardScope::SameIteration;
        h.sameIterationAnchor = findSameIterationAnchor(A[load].op, A[store].op, scope);
        if (!h.sameIterationAnchor) {
            h.sameIterationAnchor = A[store].op;
        }
        result.hazards.push_back(h);
    }
}

// Per-pair model for the loop-carried Presburger relations: the resolved
// loop depths, iteration domains, and modelled byte coefficients of one
// producer/consumer pair.
struct CarriedPairModel {
    unsigned depthP = 0;
    unsigned depthC = 0;
    unsigned numDims = 0;
    SmallVector<ConstLoopBounds> boundsP;
    SmallVector<ConstLoopBounds> boundsC;
    SmallVector<int64_t, 2> coeffP;
    SmallVector<int64_t, 2> coeffC;
    int64_t constP = 0;
    int64_t constC = 0;
    int64_t sizeP = 0;
    int64_t sizeC = 0;
    bool offsetsExact = true;
};

// Dimensions: [ip0, ip1?, ic0, ic1?]: producer coordinates first, then
// consumer coordinates (both depths are <= 2 by the time a model is built).
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

// addrP(x) < addrC(y) + sizeC => addrP - addrC - sizeC < 0
// addrC(y) < addrP(x) + sizeP => addrC - addrP - sizeP < 0
// addrP = producerConstByte + sum(ivCoeff_k * ivP_k)
// addrC = consumerConstByte + sum(ivCoeff_k * ivC_k)
// Only modelable when both byte offsets are exact affine in the IVs with
// known coefficients and constant sizes.
static bool addCarriedIntersection(IntegerPolyhedron& poly, const CarriedPairModel& m)
{
    if (!m.offsetsExact) {
        return false;
    }
    // addrP - addrC - sizeC < 0 => -(addrP - addrC - sizeC) > 0
    // => addrC - addrP + sizeC > 0 => addrC - addrP + sizeC - 1 >= 0
    SmallVector<int64_t> row(m.numDims + 1, 0);
    for (unsigned k = 0; k < m.depthP; ++k) {
        row[carriedDimPos(m, true, k)] -= m.coeffP[k];
    }
    for (unsigned k = 0; k < m.depthC; ++k) {
        row[carriedDimPos(m, false, k)] += m.coeffC[k];
    }
    __int128 rowConstant = static_cast<__int128>(m.constC) - m.constP + m.sizeC - 1;
    if (rowConstant < INT64_MIN || rowConstant > INT64_MAX) {
        return false;
    }
    row.back() = static_cast<int64_t>(rowConstant);
    poly.addInequality(row);
    // addrC - addrP - sizeP < 0 => addrP - addrP + sizeP - 1 >= 0
    SmallVector<int64_t> row2(m.numDims + 1, 0);
    for (unsigned k = 0; k < m.depthP; ++k) {
        row2[carriedDimPos(m, true, k)] += m.coeffP[k];
    }
    for (unsigned k = 0; k < m.depthC; ++k) {
        row2[carriedDimPos(m, false, k)] -= m.coeffC[k];
    }
    rowConstant = static_cast<__int128>(m.constP) - m.constC + m.sizeP - 1;
    if (rowConstant < INT64_MIN || rowConstant > INT64_MAX) {
        return false;
    }
    row2.back() = static_cast<int64_t>(rowConstant);
    poly.addInequality(row2);
    return true;
}

// Model a footprint's base byte: symbolic/allocation roots use the constant
// byte offset; absolute roots combine the absolute base with it. Fails when
// the offset is inexact, the size is unknown, or either overflows int64.
static bool modelCarriedRoot(const VecScopeMemoryFootprint& fp, int64_t& base)
{
    if (!fp.byteOffset.exact || !fp.byteSize || *fp.byteSize > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    if (fp.rootKind == MemoryRootKind::Symbolic || fp.rootKind == MemoryRootKind::ProvenAllocation) {
        base = fp.byteOffset.constant;
        return true;
    }
    if (fp.rootKind != MemoryRootKind::Absolute || !fp.absoluteBase ||
        *fp.absoluteBase > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    __int128 value = static_cast<__int128>(*fp.absoluteBase) + static_cast<__int128>(fp.byteOffset.constant);
    if (value < INT64_MIN || value > INT64_MAX) {
        return false;
    }
    base = static_cast<int64_t>(value);
    return true;
}

// Relationship between the two access roots: same symbolic/allocation root,
// both absolute, or distinct proven allocations (definitely disjoint).
struct RootRelation {
    bool sameRoot = false;
    bool bothAbsolute = false;
    bool distinctProvenAllocations = false;
};

static RootRelation classifyRootRelation(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    RootRelation rel;
    rel.sameRoot =
        (producer.rootKind == MemoryRootKind::Symbolic || producer.rootKind == MemoryRootKind::ProvenAllocation) &&
        (consumer.rootKind == MemoryRootKind::Symbolic || consumer.rootKind == MemoryRootKind::ProvenAllocation) &&
        producer.root == consumer.root;
    rel.distinctProvenAllocations = producer.rootKind == MemoryRootKind::ProvenAllocation &&
                                    consumer.rootKind == MemoryRootKind::ProvenAllocation &&
                                    producer.root != consumer.root;
    rel.bothAbsolute = producer.rootKind == MemoryRootKind::Absolute && consumer.rootKind == MemoryRootKind::Absolute;
    return rel;
}

// Model the roots, byte sizes, and IV coefficients of a carried pair. Every
// modelling failure is recorded as Unknown before returning false.
static bool modelCarriedPairRoots(
    VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, MemBarKind kind, ArrayRef<Value> pi,
    ArrayRef<Value> pj, CarriedPairModel& m)
{
    const auto& A = result.accesses;
    bool sizesAndRootsKnown = modelCarriedRoot(A[i].footprint, m.constP) &&
                              modelCarriedRoot(A[j].footprint, m.constC) && A[i].footprint.byteSize &&
                              A[j].footprint.byteSize;
    if (sizesAndRootsKnown) {
        m.sizeP = static_cast<int64_t>(*A[i].footprint.byteSize);
        m.sizeC = static_cast<int64_t>(*A[j].footprint.byteSize);
    }
    RootRelation rel = classifyRootRelation(A[i].footprint, A[j].footprint);
    if (rel.distinctProvenAllocations) {
        return false;
    }
    if ((!rel.sameRoot && !rel.bothAbsolute) || !sizesAndRootsKnown) {
        recordUnknown(
            result, i, j, kind,
            !rel.sameRoot && !rel.bothAbsolute ? DependenceReason::UnknownRoot :
                                                 DependenceReason::UnsupportedAccessShape);
        return false;
    }
    for (unsigned k = 0; k < m.depthP; ++k) {
        m.coeffP.push_back(A[i].footprint.byteOffset.getCoeff(pi[k]));
    }
    for (unsigned k = 0; k < m.depthC; ++k) {
        m.coeffC.push_back(A[j].footprint.byteOffset.getCoeff(pj[k]));
    }
    if (!convertToIterationCounters(m.coeffP, m.boundsP, m.constP) ||
        !convertToIterationCounters(m.coeffC, m.boundsC, m.constC)) {
        recordUnknown(result, i, j, kind, DependenceReason::DynamicUnmodelledExpression);
        return false;
    }
    return true;
}

// Evaluate a fully-constrained carried relation: an unmodelable address
// intersection stays Unknown; a satisfiable relation proves a carried hazard
// on `carryingLoop`.
static void recordCarriedQuery(
    IntegerPolyhedron& poly, const CarriedPairModel& m, VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j,
    MemBarKind kind, VecScopeHazardScope scope, std::optional<unsigned> carryingLoop)
{
    if (!addCarriedIntersection(poly, m)) {
        recordUnknown(result, i, j, kind, DependenceReason::DynamicUnmodelledExpression);
    } else if (poly.isIntegerEmpty()) {
        return;
    }
    MemoryHazard h;
    h.id = result.hazards.size();
    h.producer = i;
    h.consumer = j;
    h.kind = kind;
    h.scope = scope;
    h.carryingLoop = carryingLoop;
    h.distance.positive = true;
    result.hazards.push_back(h);
}

// ic0 > ip0.
static void addCommonOuterGreater(IntegerPolyhedron& poly, const CarriedPairModel& m)
{
    SmallVector<int64_t> gt(m.numDims + 1, 0);
    gt[carriedDimPos(m, false, 0)] += 1;
    gt[carriedDimPos(m, true, 0)] -= 1;
    gt.back() = -1;
    poly.addInequality(gt);
}

// ---- Same-nest Rinner ----
// depth==2: ic0 == ip0 && ic1 > ip1 => (ic0-ip0 == 0) && (ic1-ip1 >= 1)
// depth==1: ic0 > ip0 => (ic0 - ip0 >= 1)
static void queryInnerCarriedRelation(
    const CarriedPairModel& m, VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, MemBarKind kind,
    std::optional<unsigned> carryingLoop)
{
    const unsigned depth = m.depthP;
    auto poly = buildCarriedBase(m);
    if (!poly) {
        return;
    }
    if (depth == 2) {
        // ic0 == ip0
        SmallVector<int64_t> eq(m.numDims + 1, 0);
        eq[carriedDimPos(m, true, 0)] -= 1;
        eq[carriedDimPos(m, false, 0)] += 1;
        poly->addInequality(eq); // ==0 via >=0 and <=0
        SmallVector<int64_t> eq2(m.numDims + 1, 0);
        eq2[carriedDimPos(m, true, 0)] += 1;
        eq2[carriedDimPos(m, false, 0)] -= 1;
        poly->addInequality(eq2);
        // ic1 > ip1
        SmallVector<int64_t> gt(m.numDims + 1, 0);
        gt[carriedDimPos(m, false, 1)] += 1;
        gt[carriedDimPos(m, true, 1)] -= 1;
        gt.back() = -1;
        poly->addInequality(gt);
    } else {
        addCommonOuterGreater(*poly, m);
    }
    recordCarriedQuery(*poly, m, result, i, j, kind, VecScopeHazardScope::InnerLoopCarried, carryingLoop);
}

// ---- Outer-carried relation (same-nest Router and cross-hierarchy) ----
// ic0 > ip0: the consumer runs in a strictly later iteration of the common
// outer loop.
static void queryOuterCarriedRelation(
    const CarriedPairModel& m, VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, MemBarKind kind,
    std::optional<unsigned> carryingLoop)
{
    auto poly = buildCarriedBase(m);
    if (!poly) {
        return;
    }
    addCommonOuterGreater(*poly, m);
    recordCarriedQuery(*poly, m, result, i, j, kind, VecScopeHazardScope::OuterLoopCarried, carryingLoop);
}

// Loop-carried RAW/WAW relations for one ordered pair (i, j) (including
// self pairs). A relation is "empty" => no carried hazard; satisfiable =>
// proven carried hazard; unmodelable => conservative carrying-loop placement.
static void analyzeLoopCarriedPair(VecScopeMemBarAnalysisResult& result, unsigned i, unsigned j, Operation* scope)
{
    const auto& A = result.accesses;
    auto kind = hazardKind(A[i], A[j]);
    if (!kind) {
        return;
    }
    auto pi = collectIVs(A[i].op, scope);
    auto pj = collectIVs(A[j].op, scope);
    auto commonLoops = commonLoopPrefix(A[i], A[j]);
    if (commonLoops.empty()) {
        return;
    }

    // Different address spaces are definitely disjoint for both same- and
    // cross-iteration relations. Do not use same-iteration NoAlias for any
    // other case: an IV shift may make a relation overlap.
    if (A[i].footprint.addressSpace && A[j].footprint.addressSpace &&
        *A[i].footprint.addressSpace != *A[j].footprint.addressSpace) {
        return;
    }

    // Gather loop bounds for each access independently. A cross-hierarchy
    // pair has two iteration domains: the producer may carry an inner loop
    // that the consumer does not, or the two accesses may be in sibling
    // inner loops.
    CarriedPairModel model;
    model.depthP = pi.size();
    model.depthC = pj.size();
    model.numDims = model.depthP + model.depthC;
    bool allConst = gatherConstBounds(result, pi, model.boundsP);
    allConst = gatherConstBounds(result, pj, model.boundsC) && allConst;
    if (!allConst) {
        emitDynamicBoundsHazards(result, i, j, *kind, commonLoops);
        return;
    }
    if (model.depthP > 2 || model.depthC > 2) {
        recordUnknown(result, i, j, *kind, DependenceReason::UnsupportedAccessShape);
        emitDynamicBoundsHazards(result, i, j, *kind, commonLoops);
        return;
    }
    size_t unknownBefore = result.unknownDependences.size();
    if (!modelCarriedPairRoots(result, i, j, *kind, pi, pj, model)) {
        if (result.unknownDependences.size() != unknownBefore) {
            emitDynamicBoundsHazards(result, i, j, *kind, commonLoops);
        }
        return;
    }

    // ---- Same-nest Rinner/Router ----
    if (pi == pj) {
        queryInnerCarriedRelation(model, result, i, j, *kind, innermostLoopId(result, pi.back()));
    }
    // ---- Same-nest Router (only depth==2) ----
    if (pi == pj && model.depthP == 2) {
        queryOuterCarriedRelation(model, result, i, j, *kind, outermostLoopId(result, pi.front()));
    }
    // ---- Cross-hierarchy outer-carried relation ----
    // The common outer loop is the execution-order carrier. Independent
    // inner-loop coordinates remain free in their respective domains; a
    // dependence exists when a consumer instance in a later common-outer
    // iteration overlaps a producer instance from an earlier one.
    if (pi != pj) {
        queryOuterCarriedRelation(model, result, i, j, *kind, commonLoops.front());
    }
}

// --- Loop-carried RAW/WAW hazards via Presburger relations ---
// For each ordered pair (i, j), including self pairs, query Rinner and
// Router whenever both accesses share an enclosing loop. A pair can be
// loop-carried even when its same-iteration instances are statically
// ordered in either direction.
static void enumerateLoopCarriedHazards(VecScopeMemBarAnalysisResult& result, Operation* scope)
{
    const unsigned N = result.accesses.size();
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            analyzeLoopCarriedPair(result, i, j, scope);
        }
    }
}

} // namespace

Operation* vecscopemembar::findSameIterationAnchor(Operation* producer, Operation* consumer, Operation* scope)
{
    for (Operation* p = producer; p && p != scope; p = p->getParentOp()) {
        for (Operation* c = consumer; c && c != scope; c = c->getParentOp()) {
            if (p->getBlock() != c->getBlock() || !p->isBeforeInBlock(c)) {
                continue;
            }
            if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(c)) {
                return anchorInsideConsumerScope(consumer, c);
            }
            return c;
        }
    }
    return nullptr;
}

FailureOr<VecScopeMemBarAnalysisResult> vecscopemembar::runVecScopeMemBarAnalysis(Operation* scope)
{
    VecScopeMemBarAnalysisResult result;
    result.scope = scope;
    buildSchedule(scope, result);

    if (!collectAccessOccurrences(scope, result)) {
        return failure();
    }
    collectExistingBarriers(scope, result);

    SmallVector<std::pair<unsigned, unsigned>, 16> pendingWAR;
    enumerateSequentialHazards(result, scope, pendingWAR);
    resolvePendingWARHazards(result, scope, pendingWAR);
    enumerateLoopCarriedHazards(result, scope);
    return result;
}
