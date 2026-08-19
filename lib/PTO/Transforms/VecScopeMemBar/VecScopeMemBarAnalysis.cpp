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
// barriers, and enumerates RAW/WAW hazards: same-iteration, inner- and
// outer-loop-carried via Presburger dependence relations. Does not mutate
// the IR.
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
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;
using mlir::presburger::IntegerPolyhedron;

namespace {

static std::optional<MemBarKind> hazardKind(VecScopeAccessKind producer,
                                            VecScopeAccessKind consumer) {
  if (producer == VecScopeAccessKind::Store &&
      consumer == VecScopeAccessKind::Load)
    return MemBarKind::VST_VLD; // RAW
  if (producer == VecScopeAccessKind::Store &&
      consumer == VecScopeAccessKind::Store)
    return MemBarKind::VST_VST; // WAW
  if (producer == VecScopeAccessKind::Load &&
      consumer == VecScopeAccessKind::Store)
    return MemBarKind::VLD_VST; // WAR
  return std::nullopt;          // RAR
}

// Compare two operations in the lexical order of the vecscope, including
// operations nested in loops. `Operation::isBeforeInBlock` only accepts
// siblings, while the dependency graph also needs to reason about a barrier
// in one region ordering an access in a nested or sibling region.
static bool isLexicallyBefore(Operation *lhs, Operation *rhs,
                              Operation *scope) {
  if (!lhs || !rhs || lhs == rhs)
    return false;

  SmallVector<Operation *, 8> lhsAncestors;
  for (Operation *op = lhs; op; op = op->getParentOp()) {
    lhsAncestors.push_back(op);
    if (op == scope)
      break;
  }
  SmallVector<Operation *, 8> rhsAncestors;
  for (Operation *op = rhs; op; op = op->getParentOp()) {
    rhsAncestors.push_back(op);
    if (op == scope)
      break;
  }

  Operation *common = nullptr;
  for (Operation *op : lhsAncestors) {
    if (llvm::is_contained(rhsAncestors, op)) {
      common = op;
      break;
    }
  }
  if (!common)
    return false;

  auto childUnder = [&](Operation *op) {
    while (op && op->getParentOp() != common)
      op = op->getParentOp();
    return op;
  };
  Operation *lhsChild = childUnder(lhs);
  Operation *rhsChild = childUnder(rhs);
  if (!lhsChild || !rhsChild || lhsChild->getBlock() != rhsChild->getBlock())
    return false;
  return lhsChild->isBeforeInBlock(rhsChild);
}

// Collect enclosing loop IVs (outermost first) within the vecscope boundary.
// Memory-bar analysis is scoped to a single `pto.vecscope`: any loop outside
// that scope is invisible to it, so the walk stops at `scope` (exclusive).
// Stopping at the scope keeps the invariant that an op directly inside a
// vecscope with no inner loops has an empty IV set, which the same-iteration
// overlap model relies on to compare constant addresses directly.
static SmallVector<Value, 2> collectIVs(Operation *op, Operation *scope) {
  SmallVector<Value, 2> ivs;
  for (Operation *cur = op->getParentOp(); cur && cur != scope;
       cur = cur->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(cur))
      ivs.push_back(forOp.getInductionVar());
  }
  std::reverse(ivs.begin(), ivs.end());
  return ivs;
}

// Try to evaluate `v` as a constant int64.
static std::optional<int64_t> getConstantInt(Value v) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue();
  return std::nullopt;
}

// Constant lower/upper/step for an scf.for, else nullopt (dynamic).
struct ConstLoopBounds {
  int64_t lb = 0;
  int64_t ub = 0;
  int64_t step = 1;
  bool valid = false;
  // Number of iterations = (ub - lb + step - 1) / step (>=0).
  std::optional<int64_t> tripCount() const {
    if (!valid || step == 0)
      return std::nullopt;
    int64_t diff = ub - lb;
    if (diff <= 0)
      return 0;
    return (diff + (step > 0 ? step : -step) - 1) / (step > 0 ? step : -step);
  }
};
static ConstLoopBounds getConstBounds(scf::ForOp forOp) {
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
static bool hasDynamicLoopDomain(Operation *op, Operation *scope) {
  for (Operation *cursor = op->getParentOp(); cursor && cursor != scope;
       cursor = cursor->getParentOp())
    if (auto forOp = dyn_cast<scf::ForOp>(cursor))
      if (!getConstBounds(forOp).valid)
        return true;
  return false;
}

static std::optional<unsigned>
loopIdForIV(const VecScopeMemBarAnalysisResult &r, Value iv) {
  for (const auto &l : r.loops)
    if (l.inductionVar == iv)
      return l.id;
  return std::nullopt;
}

static void recordUnknown(VecScopeMemBarAnalysisResult &result, unsigned i,
                          unsigned j, MemBarKind kind,
                          DependenceReason reason) {
  result.unknownDependences.push_back({i, j, kind, reason});
}

// Dynamic loop bounds cannot currently be represented by the constant-domain
// model below. Preserve correctness by covering every shared carrying loop;
// other modelling failures remain Unknown and do not drive placement.
static void emitDynamicBoundsHazards(VecScopeMemBarAnalysisResult &result,
                                     unsigned i, unsigned j, MemBarKind kind,
                                     ArrayRef<unsigned> loopIds) {
  for (auto [index, loopId] : llvm::enumerate(loopIds)) {
    MemoryHazard h;
    h.id = result.hazards.size();
    h.producer = i;
    h.consumer = j;
    h.kind = kind;
    h.scope = index + 1 == loopIds.size()
                  ? VecScopeHazardScope::InnerLoopCarried
                  : VecScopeHazardScope::OuterLoopCarried;
    h.carryingLoop = loopId;
    h.distance.positive = true;
    result.hazards.push_back(h);
  }
}

static std::optional<unsigned>
innermostLoopId(const VecScopeMemBarAnalysisResult &r, Value iv) {
  return loopIdForIV(r, iv);
}
static std::optional<unsigned>
outermostLoopId(const VecScopeMemBarAnalysisResult &r, Value iv) {
  return loopIdForIV(r, iv);
}

// Build a flat schedule tree: root Sequence, with loops as children carrying
// their own Sequence body. `schedulePath` is the child-index path from root.
static void buildSchedule(Operation *scope,
                          VecScopeMemBarAnalysisResult &result) {
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
    for (Operation *p = forOp->getParentOp(); p; p = p->getParentOp())
      if (isa<scf::ForOp>(p) || p == scope)
        depth++;
    info.depth = depth;
    info.inductionVar = forOp.getInductionVar();
    info.lowerBound = forOp.getLowerBound();
    info.upperBound = forOp.getUpperBound();
    info.step = forOp.getStep();
    result.loops.push_back(info);
  });
}

static SmallVector<unsigned, 2> commonLoopPrefix(const AccessOccurrence &a,
                                                 const AccessOccurrence &b) {
  SmallVector<unsigned, 2> common;
  for (auto [lhs, rhs] : llvm::zip(a.loopNest, b.loopNest)) {
    if (lhs != rhs)
      break;
    common.push_back(lhs);
  }
  return common;
}

// Presburger loop-carried dependence relation

static bool addIterationDomain(IntegerPolyhedron &p, unsigned pos,
                               const ConstLoopBounds &b) {
  auto trip = b.tripCount();
  if (!trip)
    return false;
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

// Analyze interval overlap across two sequential execution regions. A
// non-empty exact Presburger relation is ProvenDependence, an empty one is
// NoDependence, and any incomplete root/domain/footprint model is Unknown.
//
// Modelled dimensions: the union of producer and consumer IVs. Shared loops
// (common prefix of the two loop nests) are constrained equal (same
// iteration); IVs that only one side carries are left as free dims of that
// side. The address-intersection constraints mirror `addIntersection`: a pair
// of intervals is disjoint iff NOT (addrP < addrC + sizeC AND addrC < addrP +
// sizeP), so proving the polyhedron of the conjunction empty proves disjoint.
static DependenceStatus
analyzeSequentialOverlap(const VecScopeMemBarAnalysisResult &result,
                         const AccessOccurrence &prod,
                         const AccessOccurrence &cons) {
  const auto &P = prod.footprint;
  const auto &C = cons.footprint;
  // Need closed byte ranges on both sides. forcesMayAlias footprints (e.g.
  // vsstb) with an open upper bound cannot be disproven here because their
  // upper bound is unknown.
  if (!P.byteSize || !C.byteSize || !P.byteOffset.exact || !C.byteOffset.exact)
    return DependenceStatus::Unknown;
  if (P.forcesMayAlias || C.forcesMayAlias)
    return DependenceStatus::Unknown;

  // Both must resolve to a comparable root: same symbolic/allocation root, or
  // both absolute (absolute bases are comparable regardless of SSA identity).
  auto bothAbsolute = [&]() {
    return P.rootKind == MemoryRootKind::Absolute &&
           C.rootKind == MemoryRootKind::Absolute;
  };
  auto sameNonAbsoluteRoot = [&]() {
    return (P.rootKind == MemoryRootKind::Symbolic ||
            P.rootKind == MemoryRootKind::ProvenAllocation) &&
           (C.rootKind == MemoryRootKind::Symbolic ||
            C.rootKind == MemoryRootKind::ProvenAllocation) &&
           P.root == C.root;
  };
  if (!bothAbsolute() && !sameNonAbsoluteRoot())
    return DependenceStatus::Unknown;
  if (bothAbsolute() &&
      (!P.absoluteBase || !C.absoluteBase ||
       *P.absoluteBase > static_cast<uint64_t>(INT64_MAX) ||
       *C.absoluteBase > static_cast<uint64_t>(INT64_MAX)))
    return DependenceStatus::Unknown;

  // Distinct proven allocations are already NoAlias at `aliasSameIteration`;
  // reaching here means roots are comparable, so continue.

  // Resolve the IV layout for the pair. Handled shapes:
  //  (a) pi == pj (same loop nest): single IV set, shared IVs identical.
  //  (b) one IV set is a proper prefix of the other (one access further out,
  //      e.g. a top-level store feeding a load inside a loop, or the reverse):
  //      single IV set = the longer of pi/pj, the shorter side's coefficients
  //      read against that set (0 for IVs it does not carry).
  //  (c) pi and pj share a common prefix but diverge (producer and consumer in
  //      sibling loops): dual IV sets [pi..., pj...], shared prefix constrained
  //      equal, non-shared IVs independent.
  auto pi = collectIVs(prod.op, result.scope);
  auto pj = collectIVs(cons.op, result.scope);
  if (pi.size() > 2 || pj.size() > 2)
    return DependenceStatus::Unknown;
  auto common = commonLoopPrefix(prod, cons);
  // single-nest iff one IV set is a prefix of the other (incl. equal/empty).
  bool prefixP = pi.size() <= pj.size() && common.size() == pi.size();
  bool prefixC = pj.size() <= pi.size() && common.size() == pj.size();
  bool singleNest = prefixP || prefixC;
  // Cross-loop pair (sibling loops, no shared prefix): model with dual IV sets
  // and no same-iteration equality — the two IVs are independent coordinates,
  // each ranging over its own loop domain.

  // Bounds for every IV we will use (pi then pj, dedup shared by position).
  auto resolveLoop = [&](Value iv) -> scf::ForOp {
    if (auto id = loopIdForIV(result, iv))
      for (const auto &l : result.loops)
        if (l.id == *id)
          return l.op;
    return scf::ForOp();
  };
  SmallVector<ConstLoopBounds> boundsP, boundsC;
  for (Value iv : pi) {
    auto forOp = resolveLoop(iv);
    if (!forOp)
      return DependenceStatus::Unknown;
    boundsP.push_back(getConstBounds(forOp));
    if (!boundsP.back().valid)
      return DependenceStatus::Unknown;
  }
  for (Value iv : pj) {
    auto forOp = resolveLoop(iv);
    if (!forOp)
      return DependenceStatus::Unknown;
    boundsC.push_back(getConstBounds(forOp));
    if (!boundsC.back().valid)
      return DependenceStatus::Unknown;
  }

  unsigned depthP = pi.size();
  unsigned depthC = pj.size();
  // Dual-set dims: [p0.., c0..]; single-set dims: [v0..] with v shared. In the
  // single-nest case the shared coordinate is the longer IV set.
  unsigned numDims = singleNest ? std::max(depthP, depthC) : (depthP + depthC);
  auto dimP = [&](unsigned k) -> unsigned { return k; };
  auto dimC = [&](unsigned k) -> unsigned {
    return singleNest ? k : (depthP + k);
  };
  // In single-nest layout, coefficients are read against the longer IV set; use
  // whichever side is longer as the coordinate basis.
  const auto &singleIVs = (depthP >= depthC) ? pi : pj;
  auto space = presburger::PresburgerSpace::getSetSpace(numDims);
  auto poly = std::make_unique<IntegerPolyhedron>(space);
  if (singleNest) {
    for (unsigned k = 0; k < singleIVs.size(); ++k) {
      auto forOp = resolveLoop(singleIVs[k]);
      if (!addIterationDomain(*poly, k, getConstBounds(forOp)))
        return DependenceStatus::Unknown;
    }
  } else {
    for (unsigned k = 0; k < depthP; ++k)
      if (!addIterationDomain(*poly, dimP(k), boundsP[k]))
        return DependenceStatus::Unknown;
    for (unsigned k = 0; k < depthC; ++k)
      if (!addIterationDomain(*poly, dimC(k), boundsC[k]))
        return DependenceStatus::Unknown;
    // Constrain the shared prefix equal (same iteration for the common loops).
    for (unsigned k = 0; k < common.size(); ++k) {
      SmallVector<int64_t> eq(numDims + 1, 0);
      eq[dimP(k)] -= 1;
      eq[dimC(k)] += 1;
      poly->addInequality(eq);
      SmallVector<int64_t> eq2(numDims + 1, 0);
      eq2[dimP(k)] += 1;
      eq2[dimC(k)] -= 1;
      poly->addInequality(eq2);
    }
  }

  // Base bytes. For absolute roots combine the absolute base with the constant
  // part of the byte offset; for symbolic/allocation roots use the constant
  // part only (relative to the shared root).
  auto rootBase = [](const VecScopeMemoryFootprint &fp,
                     int64_t &base) -> bool {
    if (fp.rootKind == MemoryRootKind::Symbolic ||
        fp.rootKind == MemoryRootKind::ProvenAllocation) {
      base = fp.byteOffset.constant;
      return true;
    }
    if (fp.rootKind != MemoryRootKind::Absolute || !fp.absoluteBase ||
        *fp.absoluteBase > static_cast<uint64_t>(INT64_MAX))
      return false;
    __int128 v = static_cast<__int128>(*fp.absoluteBase) +
                 static_cast<__int128>(fp.byteOffset.constant);
    if (v < INT64_MIN || v > INT64_MAX)
      return false;
    base = static_cast<int64_t>(v);
    return true;
  };
  int64_t constP = 0, constC = 0;
  if (!rootBase(P, constP) || !rootBase(C, constC))
    return DependenceStatus::Unknown;
  int64_t sizeP = static_cast<int64_t>(*P.byteSize);
  int64_t sizeC = static_cast<int64_t>(*C.byteSize);

  // Producer coefficients in the producer's IV coordinates, consumer
  // coefficients in the consumer's. In the single-nest layout the producer's
  // IVs are a prefix of the consumer's, so producer coefficients for IVs it
  // does not carry are 0 (getCoeff returns 0 for an absent IV).
  SmallVector<int64_t, 2> coeffP, coeffC;
  for (unsigned k = 0; k < depthP; ++k)
    coeffP.push_back(P.byteOffset.getCoeff(pi[k]));
  for (unsigned k = 0; k < depthC; ++k)
    coeffC.push_back(C.byteOffset.getCoeff(pj[k]));

  auto convertToIterationCounters = [](SmallVectorImpl<int64_t> &coeffs,
                                       ArrayRef<ConstLoopBounds> bounds,
                                       int64_t &constant) {
    __int128 adjusted = constant;
    for (unsigned k = 0; k < coeffs.size(); ++k) {
      adjusted += static_cast<__int128>(coeffs[k]) * bounds[k].lb;
      __int128 scaled = static_cast<__int128>(coeffs[k]) * bounds[k].step;
      if (scaled < INT64_MIN || scaled > INT64_MAX)
        return false;
      coeffs[k] = static_cast<int64_t>(scaled);
    }
    if (adjusted < INT64_MIN || adjusted > INT64_MAX)
      return false;
    constant = static_cast<int64_t>(adjusted);
    return true;
  };
  if (!convertToIterationCounters(coeffP, boundsP, constP) ||
      !convertToIterationCounters(coeffC, boundsC, constC))
    return DependenceStatus::Unknown;

  // Intersection constraints (same shape as the loop-carried addIntersection):
  //   addrP < addrC + sizeC  =>  addrC - addrP + sizeC - 1 >= 0
  //   addrC < addrP + sizeP  =>  addrP - addrC + sizeP - 1 >= 0
  // If the conjunction is empty, the intervals never overlap -> NoAlias.
  SmallVector<int64_t> row(numDims + 1, 0);
  for (unsigned k = 0; k < depthP; ++k)
    row[dimP(k)] -= coeffP[k];
  for (unsigned k = 0; k < depthC; ++k)
    row[dimC(k)] += coeffC[k];
  __int128 rc = static_cast<__int128>(constC) - constP + sizeC - 1;
  if (rc < INT64_MIN || rc > INT64_MAX)
    return DependenceStatus::Unknown;
  row.back() = static_cast<int64_t>(rc);
  poly->addInequality(row);

  SmallVector<int64_t> row2(numDims + 1, 0);
  for (unsigned k = 0; k < depthP; ++k)
    row2[dimP(k)] += coeffP[k];
  for (unsigned k = 0; k < depthC; ++k)
    row2[dimC(k)] -= coeffC[k];
  rc = static_cast<__int128>(constP) - constC + sizeP - 1;
  if (rc < INT64_MIN || rc > INT64_MAX)
    return DependenceStatus::Unknown;
  row2.back() = static_cast<int64_t>(rc);
  poly->addInequality(row2);

  return poly->isIntegerEmpty() ? DependenceStatus::NoDependence
                                : DependenceStatus::ProvenDependence;
}

} // namespace

Operation *vecscopemembar::findSameIterationAnchor(Operation *producer,
                                                   Operation *consumer,
                                                   Operation *scope) {
  for (Operation *p = producer; p && p != scope; p = p->getParentOp()) {
    for (Operation *c = consumer; c && c != scope; c = c->getParentOp()) {
      if (p->getBlock() != c->getBlock() || !p->isBeforeInBlock(c))
        continue;
      // A mem_bar is a vector micro-op and must remain inside the consumer's
      // vector scope. For a dependence crossing sibling vecscopes, anchor at
      // the first operation in the consumer scope rather than before the scope
      // container in the parent block.
      if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(c)) {
        Region &body = c->getRegion(0);
        if (!body.empty() && !body.front().empty()) {
          Operation *anchor = &body.front().front();
          while (isa<pto::MemBarOp>(anchor) && anchor->getNextNode())
            anchor = anchor->getNextNode();
          return anchor;
        }
        return consumer;
      }
      return c;
    }
  }
  return nullptr;
}

FailureOr<VecScopeMemBarAnalysisResult>
vecscopemembar::runVecScopeMemBarAnalysis(Operation *scope) {
  VecScopeMemBarAnalysisResult result;
  result.scope = scope;
  buildSchedule(scope, result);

  // Collect accesses in lexical order.
  bool anyFailed = false;
  unsigned order = 0;
  scope->walk([&](Operation *op) {
    if (!isUBVectorMemoryOp(op))
      return;
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
    occ.footprint = footprintFromDescriptor(*maybe, ivs);
    // Record enclosing loop ids (outermost first).
    for (Value iv : ivs) {
      for (auto &l : result.loops)
        if (l.inductionVar == iv) {
          occ.loopNest.push_back(l.id);
          break;
        }
    }
    result.accesses.push_back(occ);
  });
  if (anyFailed)
    return failure();

  // Collect existing pto.mem_bar ops.
  scope->walk([&](pto::MemBarOp mb) {
    ExistingBarrier b;
    b.op = mb.getOperation();
    b.kind = mb.getKind().getKind();
    b.phase = mb->getParentOfType<scf::ForOp>()
                  ? ExistingBarrier::LoopLatch
                  : ExistingBarrier::SameIteration;
    if (auto fl = mb->getParentOfType<scf::ForOp>()) {
      for (auto &l : result.loops)
        if (l.op == fl) {
          b.latchLoop = l.id;
          break;
        }
    }
    result.existingBarriers.push_back(b);
  });

  auto &A = result.accesses;
  const unsigned N = A.size();
  SmallVector<std::pair<unsigned, unsigned>, 16> pendingWAR;

  // --- Sequential-region RAW/WAW dependences ---
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = i + 1; j < N; ++j) {
      auto kind = hazardKind(A[i].kind, A[j].kind);
      if (!kind)
        continue;

      bool dynamicDomain = hasDynamicLoopDomain(A[i].op, scope) ||
                           hasDynamicLoopDomain(A[j].op, scope);
      if (dynamicDomain &&
          aliasSameIteration(A[i].footprint, A[j].footprint) ==
              VecScopeAliasResult::NoAlias)
        continue;
      DependenceStatus status =
          dynamicDomain ? DependenceStatus::ProvenDependence
                        : analyzeSequentialOverlap(result, A[i], A[j]);
      if (status == DependenceStatus::NoDependence)
        continue;
      if (status == DependenceStatus::Unknown) {
        recordUnknown(result, i, j, *kind,
                      DependenceReason::DynamicUnmodelledExpression);
        continue;
      }
      // Delay WAR materialization until all RAW barriers are known. A WAR
      // chain may be indirect:
      //
      //   load A -> value -> store B --VST_VLD--> load C -> value -> store A
      //
      // The first store-to-load edge is supplied by SSA, while VST_VLD orders
      // every store before the later load. Checking only load A's direct
      // value flow into the final store A would incorrectly add VLD_VST.
      if (*kind == MemBarKind::VLD_VST) {
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
      if (!h.sameIterationAnchor)
        h.sameIterationAnchor = A[j].op;
      result.hazards.push_back(h);
    }
  }

  // Resolve WAR hazards over the combined SSA + VST_VLD ordering graph. The
  // graph is deliberately built after the first pair pass so it contains all
  // same-iteration RAW hazards, including hazards whose producer/consumer
  // lexical order is later than a pending WAR pair.
  if (!pendingWAR.empty()) {
    SmallVector<SmallVector<unsigned, 4>, 16> edges(N);
    auto addEdge = [&](unsigned from, unsigned to) {
      if (!llvm::is_contained(edges[from], to))
        edges[from].push_back(to);
    };

    // SSA data dependence: a vector load is ordered before a later store if
    // one of the store's payload values depends on one of the load results.
    for (unsigned load = 0; load < N; ++load) {
      if (A[load].kind != VecScopeAccessKind::Load)
        continue;
      SmallVector<Value, 2> loadedVals = getLoadedValues(A[load].op);
      if (loadedVals.empty())
        continue;
      for (unsigned store = load + 1; store < N; ++store) {
        if (A[store].kind != VecScopeAccessKind::Store)
          continue;
        SmallVector<Value, 2> storedVals = getStoredValues(A[store].op);
        bool depends = false;
        for (Value sv : storedVals) {
          for (Value lv : loadedVals) {
            if (valueDependsOn(sv, lv)) {
              depends = true;
              break;
            }
          }
          if (depends)
            break;
        }
        if (depends)
          addEdge(load, store);
      }
    }

    // A VST_VLD barrier is placed before its consumer load. It orders all
    // stores preceding that cut before all later vector loads, not merely the
    // particular store/load pair that caused the barrier. This is the
    // transitive edge that was missing for the repro in a.pto.
    for (const auto &h : result.hazards) {
      if (h.scope != VecScopeHazardScope::SameIteration ||
          h.kind != MemBarKind::VST_VLD || h.consumer >= N)
        continue;
      unsigned cut = h.consumer;
      for (unsigned store = 0; store < cut; ++store) {
        if (A[store].kind != VecScopeAccessKind::Store)
          continue;
        for (unsigned load = cut; load < N; ++load)
          if (A[load].kind == VecScopeAccessKind::Load)
            addEdge(store, load);
      }
    }

    // Existing VST_VLD barriers provide the same ordering edge as generated
    // ones. This matters when the input already contains the first half of a
    // chain: a later WAR pair must be recognized as ordered even though no
    // new RAW hazard was materialized for that barrier in this run.
    for (const auto &barrier : result.existingBarriers) {
      if (barrier.kind != MemBarKind::VST_VLD &&
          barrier.kind != MemBarKind::VV_ALL)
        continue;
      for (unsigned store = 0; store < N; ++store) {
        if (A[store].kind != VecScopeAccessKind::Store ||
            !isLexicallyBefore(A[store].op, barrier.op, scope))
          continue;
        for (unsigned load = 0; load < N; ++load) {
          if (A[load].kind != VecScopeAccessKind::Load ||
              !isLexicallyBefore(barrier.op, A[load].op, scope))
            continue;
          addEdge(store, load);
        }
      }
    }

    for (const auto &[load, store] : pendingWAR) {
      SmallVector<bool, 16> reachable(N, false);
      SmallVector<unsigned, 16> worklist;
      reachable[load] = true;
      worklist.push_back(load);
      while (!worklist.empty()) {
        unsigned current = worklist.pop_back_val();
        for (unsigned next : edges[current]) {
          if (reachable[next])
            continue;
          reachable[next] = true;
          worklist.push_back(next);
        }
      }
      if (reachable[store])
        continue;

      MemoryHazard h;
      h.id = result.hazards.size();
      h.producer = load;
      h.consumer = store;
      h.kind = MemBarKind::VLD_VST;
      h.scope = VecScopeHazardScope::SameIteration;
      h.sameIterationAnchor =
          findSameIterationAnchor(A[load].op, A[store].op, scope);
      if (!h.sameIterationAnchor)
        h.sameIterationAnchor = A[store].op;
      result.hazards.push_back(h);
    }
  }

  // --- Loop-carried RAW/WAW hazards via Presburger relations ---
  // For each ordered pair (i, j), including self pairs, query Rinner and
  // Router whenever both accesses share an enclosing loop. A pair can be
  // loop-carried even when its same-iteration instances are statically
  // ordered in either direction.
  // A relation is "empty" => no carried hazard; satisfiable => proven carried
  // hazard; unmodelable => Unknown and no placement.
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      auto kind = hazardKind(A[i].kind, A[j].kind);
      if (!kind)
        continue;
      auto pi = collectIVs(A[i].op, scope);
      auto pj = collectIVs(A[j].op, scope);
      auto commonLoops = commonLoopPrefix(A[i], A[j]);
      if (commonLoops.empty())
        continue;

      // Different address spaces are definitely disjoint for both same- and
      // cross-iteration relations. Do not use same-iteration NoAlias for any
      // other case: an IV shift may make a relation overlap.
      if (A[i].footprint.addressSpace && A[j].footprint.addressSpace &&
          *A[i].footprint.addressSpace != *A[j].footprint.addressSpace)
        continue;

      const bool sameLoopNest = pi == pj;
      const unsigned depthP = pi.size();
      const unsigned depthC = pj.size();
      // Gather loop bounds for each access independently. A cross-hierarchy
      // pair has two iteration domains: the producer may carry an inner loop
      // that the consumer does not, or the two accesses may be in sibling
      // inner loops.
      SmallVector<ConstLoopBounds> boundsP;
      SmallVector<ConstLoopBounds> boundsC;
      bool allConst = true;
      for (Value iv : pi) {
        // IVs are scf.for block arguments (no defining op). Resolve the
        // carrying loop via the analysis result's recorded inductionVar.
        scf::ForOp forOp;
        if (auto id = loopIdForIV(result, iv)) {
          for (const auto &l : result.loops)
            if (l.id == *id) {
              forOp = l.op;
              break;
            }
        }
        if (!forOp) {
          allConst = false;
          break;
        }
        boundsP.push_back(getConstBounds(forOp));
        if (!boundsP.back().valid) {
          allConst = false;
          break;
        }
      }
      for (Value iv : pj) {
        scf::ForOp forOp;
        if (auto id = loopIdForIV(result, iv)) {
          for (const auto &l : result.loops)
            if (l.id == *id) {
              forOp = l.op;
              break;
            }
        }
        if (!forOp) {
          allConst = false;
          break;
        }
        boundsC.push_back(getConstBounds(forOp));
        if (!boundsC.back().valid) {
          allConst = false;
          break;
        }
      }
      if (!allConst) {
        emitDynamicBoundsHazards(result, i, j, *kind, commonLoops);
        continue;
      }
      if (depthP > 2 || depthC > 2) {
        recordUnknown(result, i, j, *kind,
                      DependenceReason::UnsupportedAccessShape);
        continue;
      }

      // Dimensions: [ip0, ip1?, ic0, ic1?] for the same nest, and
      // [ip0, ip1?, ic0, ic1?] with independent producer/consumer domains for
      // a cross-hierarchy pair. (Both depths are <= 2.)
      // Rinner (depth==2): ic0 == ip0 && ic1 > ip1
      // Rinner (depth==1): ic0 > ip0
      // Router (depth==2): ic0 > ip0
      // (depth==1 has only an inner loop; Router does not apply.)
      auto dimPos = [depthP](bool producer, unsigned k) -> unsigned {
        return producer ? k : (depthP + k);
      };

      unsigned numDims = depthP + depthC;
      auto space = presburger::PresburgerSpace::getSetSpace(numDims);

      auto buildBase = [&](presburger::PresburgerSpace &sp,
                           bool withIntersection) {
        auto poly = std::make_unique<IntegerPolyhedron>(sp);
        // Iteration domains.
        for (unsigned k = 0; k < depthP; ++k)
          if (!addIterationDomain(*poly, dimPos(true, k), boundsP[k]))
            return std::unique_ptr<IntegerPolyhedron>(nullptr);
        for (unsigned k = 0; k < depthC; ++k)
          if (!addIterationDomain(*poly, dimPos(false, k), boundsC[k]))
            return std::unique_ptr<IntegerPolyhedron>(nullptr);
        return poly;
      };

      // Address intersection constraints:
      // addrP(x) < addrC(y) + sizeC => addrP - addrC - sizeC < 0
      // addrC(y) < addrP(x) + sizeP => addrC - addrP - sizeP < 0
      // addrP = producerConstByte + sum(ivCoeff_k * ivP_k)
      // addrC = consumerConstByte + sum(ivCoeff_k * ivC_k)
      // Only modelable when both byte offsets are exact affine in the IVs with
      // known coefficients and constant sizes.
      auto modelRoot = [](const VecScopeMemoryFootprint &fp,
                          int64_t &base) -> bool {
        if (!fp.byteOffset.exact || !fp.byteSize ||
            *fp.byteSize > static_cast<uint64_t>(INT64_MAX))
          return false;
        if (fp.rootKind == MemoryRootKind::Symbolic ||
            fp.rootKind == MemoryRootKind::ProvenAllocation) {
          base = fp.byteOffset.constant;
          return true;
        }
        if (fp.rootKind != MemoryRootKind::Absolute || !fp.absoluteBase ||
            *fp.absoluteBase > static_cast<uint64_t>(INT64_MAX))
          return false;
        __int128 value = static_cast<__int128>(*fp.absoluteBase) +
                         static_cast<__int128>(fp.byteOffset.constant);
        if (value < INT64_MIN || value > INT64_MAX)
          return false;
        base = static_cast<int64_t>(value);
        return true;
      };
      int64_t sizeP = 0;
      int64_t sizeC = 0;
      int64_t constP = 0;
      int64_t constC = 0;
      bool sizesAndRootsKnown = modelRoot(A[i].footprint, constP) &&
                                modelRoot(A[j].footprint, constC) &&
                                A[i].footprint.byteSize &&
                                A[j].footprint.byteSize;
      if (sizesAndRootsKnown) {
        sizeP = static_cast<int64_t>(*A[i].footprint.byteSize);
        sizeC = static_cast<int64_t>(*A[j].footprint.byteSize);
      }
      bool sameRoot =
          (A[i].footprint.rootKind == MemoryRootKind::Symbolic ||
           A[i].footprint.rootKind == MemoryRootKind::ProvenAllocation) &&
          (A[j].footprint.rootKind == MemoryRootKind::Symbolic ||
           A[j].footprint.rootKind == MemoryRootKind::ProvenAllocation) &&
          A[i].footprint.root == A[j].footprint.root;
      bool distinctProvenAllocations =
          A[i].footprint.rootKind == MemoryRootKind::ProvenAllocation &&
          A[j].footprint.rootKind == MemoryRootKind::ProvenAllocation &&
          A[i].footprint.root != A[j].footprint.root;
      if (distinctProvenAllocations)
        continue;
      bool bothAbsolute = A[i].footprint.rootKind == MemoryRootKind::Absolute &&
                          A[j].footprint.rootKind == MemoryRootKind::Absolute;
      if ((!sameRoot && !bothAbsolute) || !sizesAndRootsKnown) {
        recordUnknown(result, i, j, *kind,
                      !sameRoot && !bothAbsolute
                          ? DependenceReason::UnknownRoot
                          : DependenceReason::UnsupportedAccessShape);
        continue;
      }
      SmallVector<int64_t, 2> coeffP, coeffC;
      bool offsetsExact = sizesAndRootsKnown;
      for (unsigned k = 0; k < depthP && offsetsExact; ++k)
        coeffP.push_back(A[i].footprint.byteOffset.getCoeff(pi[k]));
      for (unsigned k = 0; k < depthC && offsetsExact; ++k)
        coeffC.push_back(A[j].footprint.byteOffset.getCoeff(pj[k]));
      auto convertToIterationCounters = [&](SmallVectorImpl<int64_t> &coeffs,
                                            ArrayRef<ConstLoopBounds> loopBounds,
                                            int64_t &constant) {
        __int128 adjusted = constant;
        for (unsigned k = 0; k < coeffs.size(); ++k) {
          adjusted +=
              static_cast<__int128>(coeffs[k]) * loopBounds[k].lb;
          __int128 scaled =
              static_cast<__int128>(coeffs[k]) * loopBounds[k].step;
          if (scaled < INT64_MIN || scaled > INT64_MAX)
            return false;
          coeffs[k] = static_cast<int64_t>(scaled);
        }
        if (adjusted < INT64_MIN || adjusted > INT64_MAX)
          return false;
        constant = static_cast<int64_t>(adjusted);
        return true;
      };
      offsetsExact = convertToIterationCounters(coeffP, boundsP, constP) &&
                     convertToIterationCounters(coeffC, boundsC, constC);
      if (!offsetsExact) {
        recordUnknown(result, i, j, *kind,
                      DependenceReason::DynamicUnmodelledExpression);
        continue;
      }

      auto addIntersection = [&](IntegerPolyhedron &poly) {
        if (!offsetsExact)
          return false;
        // addrP - addrC - sizeC < 0 => -(addrP - addrC - sizeC) > 0
        // => addrC - addrP + sizeC > 0 => addrC - addrP + sizeC - 1 >= 0
        SmallVector<int64_t> row(numDims + 1, 0);
        for (unsigned k = 0; k < depthP; ++k)
          row[dimPos(true, k)] -= coeffP[k];
        for (unsigned k = 0; k < depthC; ++k)
          row[dimPos(false, k)] += coeffC[k];
        __int128 rowConstant =
            static_cast<__int128>(constC) - constP + sizeC - 1;
        if (rowConstant < INT64_MIN || rowConstant > INT64_MAX)
          return false;
        row.back() = static_cast<int64_t>(rowConstant);
        poly.addInequality(row);
        // addrC - addrP - sizeP < 0 => addrP - addrP + sizeP - 1 >= 0
        SmallVector<int64_t> row2(numDims + 1, 0);
        for (unsigned k = 0; k < depthP; ++k)
          row2[dimPos(true, k)] += coeffP[k];
        for (unsigned k = 0; k < depthC; ++k)
          row2[dimPos(false, k)] -= coeffC[k];
        rowConstant = static_cast<__int128>(constP) - constC + sizeP - 1;
        if (rowConstant < INT64_MIN || rowConstant > INT64_MAX)
          return false;
        row2.back() = static_cast<int64_t>(rowConstant);
        poly.addInequality(row2);
        return true;
      };

      // ---- Same-nest Rinner/Router ----
      // depth==2: ic0 == ip0 && ic1 > ip1 => (ic0-ip0 == 0) && (ic1-ip1 >= 1)
      // depth==1: ic0 > ip0 => (ic0 - ip0 >= 1)
      if (sameLoopNest) {
        const unsigned depth = depthP;
        auto sp = presburger::PresburgerSpace::getSetSpace(numDims);
        auto poly = buildBase(sp, true);
        if (poly) {
          if (depth == 2) {
            // ic0 == ip0
            SmallVector<int64_t> eq(numDims + 1, 0);
            eq[dimPos(true, 0)] -= 1;
            eq[dimPos(false, 0)] += 1;
            poly->addInequality(eq); // ==0 via >=0 and <=0
            SmallVector<int64_t> eq2(numDims + 1, 0);
            eq2[dimPos(true, 0)] += 1;
            eq2[dimPos(false, 0)] -= 1;
            poly->addInequality(eq2);
            // ic1 > ip1
            SmallVector<int64_t> gt(numDims + 1, 0);
            gt[dimPos(false, 1)] += 1;
            gt[dimPos(true, 1)] -= 1;
            gt.back() = -1;
            poly->addInequality(gt);
          } else {
            // ic0 > ip0
            SmallVector<int64_t> gt(numDims + 1, 0);
            gt[dimPos(false, 0)] += 1;
            gt[dimPos(true, 0)] -= 1;
            gt.back() = -1;
            poly->addInequality(gt);
          }
          bool modelable = addIntersection(*poly);
          if (!modelable) {
            recordUnknown(result, i, j, *kind,
                          DependenceReason::DynamicUnmodelledExpression);
          } else if (!poly->isIntegerEmpty()) {
            // Inner-carried hazard on innermost loop.
            MemoryHazard h;
            h.id = result.hazards.size();
            h.producer = i;
            h.consumer = j;
            h.kind = *kind;
            h.scope = VecScopeHazardScope::InnerLoopCarried;
            h.carryingLoop = innermostLoopId(result, pi.back());
            h.distance.positive = true;
            result.hazards.push_back(h);
          }
        }
      }

      // ---- Same-nest Router (only depth==2) ----
      if (sameLoopNest && depthP == 2) {
        auto sp = presburger::PresburgerSpace::getSetSpace(numDims);
        auto poly = buildBase(sp, true);
        if (poly) {
          // ic0 > ip0
          SmallVector<int64_t> gt(numDims + 1, 0);
          gt[dimPos(false, 0)] += 1;
          gt[dimPos(true, 0)] -= 1;
          gt.back() = -1;
          poly->addInequality(gt);
          bool modelable = addIntersection(*poly);
          if (!modelable) {
            recordUnknown(result, i, j, *kind,
                          DependenceReason::DynamicUnmodelledExpression);
          } else if (!poly->isIntegerEmpty()) {
            MemoryHazard h;
            h.id = result.hazards.size();
            h.producer = i;
            h.consumer = j;
            h.kind = *kind;
            h.scope = VecScopeHazardScope::OuterLoopCarried;
            h.carryingLoop = outermostLoopId(result, pi.front());
            h.distance.positive = true;
            result.hazards.push_back(h);
          }
        }
      }

      // ---- Cross-hierarchy outer-carried relation ----
      // The common outer loop is the execution-order carrier. Independent
      // inner-loop coordinates remain free in their respective domains; a
      // dependence exists when a consumer instance in a later common-outer
      // iteration overlaps a producer instance from an earlier one.
      if (!sameLoopNest) {
        auto sp = presburger::PresburgerSpace::getSetSpace(numDims);
        auto poly = buildBase(sp, true);
        if (poly) {
          SmallVector<int64_t> gt(numDims + 1, 0);
          gt[dimPos(false, 0)] += 1;
          gt[dimPos(true, 0)] -= 1;
          gt.back() = -1;
          poly->addInequality(gt);
          bool modelable = addIntersection(*poly);
          if (!modelable) {
            recordUnknown(result, i, j, *kind,
                          DependenceReason::DynamicUnmodelledExpression);
          } else if (!poly->isIntegerEmpty()) {
            MemoryHazard h;
            h.id = result.hazards.size();
            h.producer = i;
            h.consumer = j;
            h.kind = *kind;
            h.scope = VecScopeHazardScope::OuterLoopCarried;
            h.carryingLoop = commonLoops.front();
            h.distance.positive = true;
            result.hazards.push_back(h);
          }
        }
      }
    }
  }

  return result;
}

FailureOr<VecScopeMemBarAnalysisResult>
vecscopemembar::runCrossVecScopeMemBarAnalysis(Operation *scope) {
  VecScopeMemBarAnalysisResult result;
  result.scope = scope;

  auto enclosingVecScope = [](Operation *op) -> Operation * {
    for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
         parent = parent->getParentOp())
      if (isa<pto::VecScopeOp, pto::StrictVecScopeOp>(parent))
        return parent;
    return nullptr;
  };
  auto isVectorMemoryOp = [](Operation *op) {
    if (!isa<pto::VectorMicroOpInterface>(op))
      return false;
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      return false;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    iface.getEffects(effects);
    return llvm::any_of(effects, [](const auto &effect) {
      Value value = effect.getValue();
      return value && isUBBackedType(value.getType()) &&
             (isa<MemoryEffects::Read>(effect.getEffect()) ||
              isa<MemoryEffects::Write>(effect.getEffect()));
    });
  };
  auto crossScopeKind =
      [](VecScopeAccessKind producer,
         VecScopeAccessKind consumer) -> std::optional<MemBarKind> {
    if (producer == VecScopeAccessKind::Store &&
        consumer == VecScopeAccessKind::Load)
      return MemBarKind::VST_VLD;
    if (producer == VecScopeAccessKind::Store &&
        consumer == VecScopeAccessKind::Store)
      return MemBarKind::VST_VST;
    if (producer == VecScopeAccessKind::Load &&
        consumer == VecScopeAccessKind::Store)
      return MemBarKind::VLD_VST;
    return std::nullopt;
  };

  unsigned lexicalOrder = 0;
  bool anyFailed = false;
  scope->walk([&](Operation *op) {
    if (!isVectorMemoryOp(op) || !enclosingVecScope(op))
      return;
    auto descriptor = buildAccessDescriptor(op, /*ivs=*/{});
    if (failed(descriptor)) {
      anyFailed = true;
      return;
    }
    AccessOccurrence access;
    access.id = result.accesses.size();
    access.op = op;
    access.kind = descriptor->kind;
    access.footprint = footprintFromDescriptor(*descriptor, /*ivs=*/{});
    access.lexicalOrder = lexicalOrder++;
    result.accesses.push_back(access);
  });
  if (anyFailed)
    return failure();

  scope->walk([&](pto::MemBarOp op) {
    ExistingBarrier barrier;
    barrier.op = op;
    barrier.kind = op.getKind().getKind();
    barrier.phase = op->getParentOfType<scf::ForOp>()
                        ? ExistingBarrier::LoopLatch
                        : ExistingBarrier::SameIteration;
    result.existingBarriers.push_back(barrier);
  });

  auto isOrderedSiblingPair = [&](unsigned producer, unsigned consumer) {
    Operation *producerScope = enclosingVecScope(result.accesses[producer].op);
    Operation *consumerScope = enclosingVecScope(result.accesses[consumer].op);
    return producerScope && consumerScope && producerScope != consumerScope &&
           producerScope->getBlock() == consumerScope->getBlock() &&
           producerScope->isBeforeInBlock(consumerScope);
  };

  for (unsigned producer = 0; producer < result.accesses.size(); ++producer) {
    for (unsigned consumer = producer + 1; consumer < result.accesses.size();
         ++consumer) {
      auto kind = crossScopeKind(result.accesses[producer].kind,
                                 result.accesses[consumer].kind);
      if (!kind || !isOrderedSiblingPair(producer, consumer))
        continue;
      if (aliasSameIteration(result.accesses[producer].footprint,
                             result.accesses[consumer].footprint) ==
          VecScopeAliasResult::NoAlias)
        continue;

      MemoryHazard hazard;
      hazard.id = result.hazards.size();
      hazard.producer = producer;
      hazard.consumer = consumer;
      hazard.kind = *kind;
      hazard.scope = VecScopeHazardScope::SameIteration;
      hazard.sameIterationAnchor = findSameIterationAnchor(
          result.accesses[producer].op, result.accesses[consumer].op, scope);
      if (!hazard.sameIterationAnchor)
        hazard.sameIterationAnchor = result.accesses[consumer].op;
      result.hazards.push_back(hazard);
    }
  }

  return result;
}
