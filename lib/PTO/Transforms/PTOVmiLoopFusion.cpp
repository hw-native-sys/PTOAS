// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//
// PTOVmiLoopFusion.cpp - fuse same-header scf.for inside pto.fusion_region
//===----------------------------------------------------------------------===//
//
// VMI tile-library compute is always a single scf.for layer (the inner VL
// loop). This pass fuses adjacent same-header scf.for ops inside each
// pto.fusion_region into one fused scf.for. Two for's can be fused only if the
// ops sitting between them can be legally relocated after fusion:
//   - hoisted above the fused for (loop-invariant: no SSA/UB input produced by
//     any member's result or UB write),
//   - sunk below the fused for (its SSA results / UB writes not read inside
//     any member).
// Between-ops form dependency-connected components (SSA def-use, or same-UB
// store->load); each component must move as a whole. A component that can
// neither hoist nor sink — e.g. the tmuls(scale ColMax) chain, which reads
// the ColMax final UB (cannot hoist, the reduce is complete only after the
// loop) and whose store is read by the ColExpand-sub loop (cannot sink) —
// blocks fusion: the run stops there, so a reduce and the loop that consumes
// its final result stay separate for's.
//
// The fused scf.for's init args concatenate each member's init args (reduce
// carry); the fused body clones each member's body (without scf.yield) in
// source order; the fused scf.yield concatenates each member's yield operands
// mapped through the fused iter-args. Between-components hoisted above /
// sunk below the fused for are moved there (not cloned). The fused loop is
// built with a body-builder callback so the yield is created in place (no
// post-hoc setOperands on iter-arg/result linkage).
//
// The pass only touches scf.for ops directly nested inside a pto.fusion_region
// body. It does not perform mem2reg (UB roundtrip elimination) and does not
// build pto.vecscope.
//
// CROSS-ITERATION UB GUARD (first-version legality): a candidate loop joins a
// run only if every UB it exchanges with the run is a SAME-iteration transfer.
// Producer writes UB W at offset f(i) and consumer reads W at offset g(i);
// fusing into one body makes the consumer, in iteration i, read whatever the
// producer wrote in iteration i. That equals the original (where the consumer
// read the producer's FINAL value across iterations) ONLY when:
//   - BOTH offsets depend on the IV (a per-iteration transfer; a fixed-offset
//     transfer is cross-iteration — the consumer reads the producer's final
//     value, not the current-iteration value — and is blocked);
//   - BOTH offsets are restricted INJECTIVE AFFINE forms (IV, IV*positive_const,
//     +const). Non-injective forms like i%2 collide (f(0)==f(2)) so the consumer
//     would read the producer's final write, not the current — blocked even if
//     the two offsets are structurally equivalent;
//   - the two offsets are structurally equivalent (all run IVs map to the single
//     fused IV).
// Stencils (A[i+1]), fixed-offset loops (UB[0] every iteration), and i%2 are
// all blocked. A loop containing any other memory-effecting op or an unmodeled
// vload/vstore shape is also blocked: the first version cannot prove its
// accesses are same-iteration transfers.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VmiMemoryLocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVMILOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Structural equivalence for loop bounds and steps. Preserve every operation,
// operand and attribute: in particular, N+1 and N+2 are different headers.
static bool areEquivalentHeaderValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs || lhs.getType() != rhs.getType())
    return false;
  Operation *ld = lhs.getDefiningOp();
  Operation *rd = rhs.getDefiningOp();
  if (!ld || !rd || ld == rd)
    return ld == rd;
  if (ld->getName() != rd->getName() || ld->getNumRegions() != 0 ||
      rd->getNumRegions() != 0 ||
      ld->getNumOperands() != rd->getNumOperands() ||
      ld->getAttrDictionary() != rd->getAttrDictionary() ||
      !llvm::equal(ld->getResultTypes(), rd->getResultTypes()))
    return false;
  for (auto [a, b] : llvm::zip(ld->getOperands(), rd->getOperands()))
    if (!areEquivalentHeaderValues(a, b))
      return false;
  return true;
}

static bool isFusionProvenanceAttr(NamedAttribute attr) {
  StringRef name = attr.getName().strref();
  return name.starts_with("pto.tilelib.") ||
         name.starts_with("pto.vmi.fusion.");
}

static SmallVector<NamedAttribute, 4> getSemanticLoopAttrs(scf::ForOp loop) {
  SmallVector<NamedAttribute, 4> attrs;
  for (NamedAttribute attr : loop->getAttrs())
    if (!isFusionProvenanceAttr(attr))
      attrs.push_back(attr);
  return attrs;
}

static bool sameHeader(scf::ForOp a, scf::ForOp b) {
  if (!areEquivalentHeaderValues(a.getStep(), b.getStep()))
    return false;
  if (!areEquivalentHeaderValues(a.getLowerBound(), b.getLowerBound()))
    return false;
  if (!areEquivalentHeaderValues(a.getUpperBound(), b.getUpperBound()))
    return false;
  if (getSemanticLoopAttrs(a) != getSemanticLoopAttrs(b))
    return false;
  return true;
}

static bool isTileLibVmiPrincipalLoop(scf::ForOp loop) {
  auto impl = loop->getAttrOfType<StringAttr>("pto.tilelib.impl");
  auto source = loop->getAttrOfType<StringAttr>("pto.vmi.fusion.source");
  if (!impl || impl.getValue() != "vmi" || !source ||
      source.getValue() != "tilelib")
    return false;
  return loop->hasAttr("pto.vmi.fusion.principal_loop") &&
         !loop->hasAttr("pto.vmi.fusion.boundary");
}

// --- UB (tile buffer) identity: which compile-time address+type a vmi
// load/store accesses. Two ops touch the same UB iff they resolve to the same
// (addr-constant, memref-type) pair. Traced through pto.castptr -> memref ->
// pto.pointer_cast(addr-const). Returns std::nullopt if the base is not a
// compile-time constant address (then we conservatively cannot reason).
struct UBId {
  int64_t address = 0;  // numeric address, independent of SSA identity
  Type memrefType;      // the pointer_cast view type (shape+dtype)
  std::optional<int64_t> storageBytes;
  bool operator==(const UBId &o) const {
    return address == o.address && memrefType == o.memrefType;
  }
};

static std::optional<UBId> resolvePtrUB(Value base) {
  auto root = pto::resolveVmiStorageRoot(base);
  if (!root)
    return std::nullopt;
  return UBId{root->address, root->viewType, root->storageBytes};
}

static bool mayAlias(const UBId &lhs, const UBId &rhs) {
  return pto::mayAliasVmiStorageRoot(
      pto::VmiStorageRoot{lhs.address, lhs.storageBytes, lhs.memrefType},
      pto::VmiStorageRoot{rhs.address, rhs.storageBytes, rhs.memrefType});
}

static std::optional<UBId> getVLoadUB(pto::VMIvLoadOp op) {
  return resolvePtrUB(op.getSource());
}
static std::optional<UBId> getVStoreUB(pto::VMIvStoreOp op) {
  return resolvePtrUB(op.getDestination());
}

// Collect every UB a loop's body loads (reads) and stores (writes).
static void collectLoopUBs(scf::ForOp loop, SmallVectorImpl<UBId> &reads,
                           SmallVectorImpl<UBId> &writes) {
  loop.getBody()->walk([&](Operation *op) {
    if (auto v = dyn_cast<pto::VMIvLoadOp>(op)) {
      if (auto id = getVLoadUB(v))
        reads.push_back(*id);
    } else if (auto v = dyn_cast<pto::VMIvStoreOp>(op)) {
      if (auto id = getVStoreUB(v))
        writes.push_back(*id);
    }
    return WalkResult::advance();
  });
}

static bool ubListContains(ArrayRef<UBId> list, const UBId &x) {
  return llvm::any_of(list, [&](const UBId &u) { return mayAlias(u, x); });
}

// A UB access with its offset value, so cross-iteration dependencies can be
// detected by checking whether the offset depends on a loop IV. A5 vload/
// vstore offsets are `index`-typed.
struct UBAccess {
  UBId ub;
  Value offset;
  bool isLoad;
  int64_t accessBytes = 0;
};

static std::optional<int64_t> getElementBytes(Type type) {
  if (auto memref = dyn_cast<MemRefType>(type))
    type = memref.getElementType();
  if (auto ptr = dyn_cast<pto::PtrType>(type))
    type = ptr.getElementType();
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16() || type.isBF16() || type.isInteger(16))
    return 2;
  if (type.isInteger(8) || type.isInteger(1))
    return 1;
  if (type.isInteger(64))
    return 8;
  return std::nullopt;
}

static int64_t getVRegAccessBytes(Type type) {
  auto vreg = dyn_cast<pto::VMIVRegType>(type);
  if (!vreg)
    return 0;
  auto elementBytes = getElementBytes(vreg.getElementType());
  if (!elementBytes)
    return 0;
  return static_cast<int64_t>(vreg.getElementCount()) * *elementBytes;
}

static std::optional<int64_t> getStaticInteger(Value value) {
  if (auto c = value.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
      return attr.getInt();
  if (auto c = value.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = value.getDefiningOp<arith::ConstantIntOp>())
    return c.value();
  return std::nullopt;
}

static bool accessRangesMayOverlap(const UBAccess &lhs,
                                   const UBAccess &rhs) {
  if (!mayAlias(lhs.ub, rhs.ub))
    return false;
  if (lhs.ub.memrefType != rhs.ub.memrefType || lhs.accessBytes <= 0 ||
      rhs.accessBytes <= 0)
    return true;
  auto lhsOffset = getStaticInteger(lhs.offset);
  auto rhsOffset = getStaticInteger(rhs.offset);
  auto elementBytes = getElementBytes(lhs.ub.memrefType);
  if (!lhsOffset || !rhsOffset || !elementBytes)
    return true;
  int64_t lhsDelta = 0;
  int64_t rhsDelta = 0;
  if (__builtin_mul_overflow(*lhsOffset, *elementBytes, &lhsDelta) ||
      __builtin_mul_overflow(*rhsOffset, *elementBytes, &rhsDelta))
    return true;
  int64_t lhsBegin = 0;
  int64_t rhsBegin = 0;
  if (__builtin_add_overflow(lhs.ub.address, lhsDelta, &lhsBegin) ||
      __builtin_add_overflow(rhs.ub.address, rhsDelta, &rhsBegin))
    return true;
  int64_t lhsEnd = 0;
  int64_t rhsEnd = 0;
  if (__builtin_add_overflow(lhsBegin, lhs.accessBytes, &lhsEnd) ||
      __builtin_add_overflow(rhsBegin, rhs.accessBytes, &rhsEnd))
    return true;
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

// Indirect/non-vload-vstore memory ops are not modeled by the UB dependency
// analysis and therefore block fusion. Ordinary vload/vstore accesses must
// have a compile-time-resolvable base, except post-update stores: their moving
// destination is an explicit loop-carried SSA chain and they cannot establish
// a UB exchange with another loop through the compile-time UB table.
static bool hasUnmodeledMemoryAccess(scf::ForOp loop) {
  bool unmodeled = false;
  loop.getBody()->walk([&](Operation *op) {
    if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
      if (!getVLoadUB(load)) {
        unmodeled = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<pto::VMIvStoreOp>(op)) {
      if (!store.getUpdatedBase() && !getVStoreUB(store)) {
        unmodeled = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (isa<func::CallOp>(op)) {
      unmodeled = true;
      return WalkResult::interrupt();
    }
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (iface.hasEffect<MemoryEffects::Read>() ||
          iface.hasEffect<MemoryEffects::Write>()) {
        unmodeled = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return unmodeled;
}

// Collect every modeled (UB, offset, load|store) access in a loop body. The
// caller first rejects loops containing unmodeled memory accesses.
static void collectLoopUBAccesses(scf::ForOp loop,
                                  SmallVectorImpl<UBAccess> &out) {
  loop.getBody()->walk([&](Operation *op) {
    if (auto v = dyn_cast<pto::VMIvLoadOp>(op)) {
      if (auto id = getVLoadUB(v))
        out.push_back({*id, v.getOffset(), /*isLoad=*/true,
                       getVRegAccessBytes(v.getResult(0).getType())});
    } else if (auto v = dyn_cast<pto::VMIvStoreOp>(op)) {
      if (auto id = getVStoreUB(v)) {
        int64_t bytes = v.getValues().empty()
                            ? 0
                            : getVRegAccessBytes(v.getValues().front().getType());
        out.push_back({*id, v.getOffset(), /*isLoad=*/false, bytes});
      }
    }
    return WalkResult::advance();
  });
}

// Does `offset` depend (via SSA def-use) on any induction variable in `ivs`?
// A bounded backward walk over `offset`'s defining ops: if any operand chain
// reaches an IV in `ivs`, return true (the offset is a function of the loop
// index -> a cross-iteration UB dependency). Constants and non-IV block
// arguments (function params, other-region iter args) return false. Region-
// bearing ops and walks past the depth cap return true conservatively (we
// cannot prove the offset is loop-invariant, so fusion is blocked).
static bool offsetDependsOnIV(Value offset, ArrayRef<Value> ivs) {
  if (!offset)
    return false;
  for (Value iv : ivs)
    if (offset == iv)
      return true;
  // Bounded worklist backward walk.
  SmallVector<Value, 8> work = {offset};
  SmallPtrSet<Value, 32> seen;
  unsigned depth = 0;
  constexpr unsigned kMaxDepth = 16;
  while (!work.empty()) {
    if (++depth > kMaxDepth)
      return true; // too deep to prove invariant -> conservatively block
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (llvm::is_contained(ivs, v))
      return true;
    // Block arguments that are not one of the IVs are loop-invariant inputs
    // (function params / outer-region iter args) -> not an IV dependency.
    if (isa<BlockArgument>(v))
      continue;
    Operation *def = v.getDefiningOp();
    if (!def)
      continue; // unreachable; treated as invariant
    if (def->getNumRegions() != 0)
      return true; // region-bearing producer (e.g. another scf.for result)
    // a function call / unknown op: conservatively block.
    if (isa<func::CallOp>(def))
      return true;
    for (Value opnd : def->getOperands())
      work.push_back(opnd);
  }
  return false; // walked to roots (constants / non-IV args) without hitting an IV
}

// Structural equivalence of two index-typed offset values with IV
// normalization: two values are equivalent iff they share the same SSA def
// tree shape (same op name, attrs, operand types) and, recursively, equivalent
// operands. All induction variables in `ivs` are treated as the SAME value —
// after fusion every member's IV maps to the fused loop's single IV, so a
// producer offset `arith.muli %iv_member, %c64` and a consumer offset
// `arith.muli %iv_cand, %c64` (distinct BlockArguments, same constant) are
// equivalent. Constants are compared by attr dict.
static bool areEquivalentOffsetValues(Value lhs, Value rhs,
                                      ArrayRef<Value> ivs) {
  if (lhs == rhs)
    return true;
  bool lhsIV = llvm::is_contained(ivs, lhs);
  bool rhsIV = llvm::is_contained(ivs, rhs);
  if (lhsIV || rhsIV)
    return lhsIV && rhsIV; // both induction vars -> same fused IV
  if (!lhs || !rhs)
    return false;
  Operation *ld = lhs.getDefiningOp();
  Operation *rd = rhs.getDefiningOp();
  if (!ld || !rd)
    return false;
  if (ld == rd)
    return true;
  if (ld->getName() != rd->getName() || ld->getNumRegions() != 0 ||
      rd->getNumRegions() != 0 ||
      ld->getNumOperands() != rd->getNumOperands() ||
      ld->getAttrDictionary() != rd->getAttrDictionary())
    return false;
  for (auto [a, b] : llvm::zip(ld->getOperands(), rd->getOperands()))
    if (!areEquivalentOffsetValues(a, b, ivs))
      return false;
  return true;
}

// Is `offset` a RESTRICTED INJECTIVE AFFINE form in the IV? First version only
// accepts:
//   IV                                  (bare induction variable)
//   IV * positive_constant              (mul by a positive integer)
//   <injective affine> + constant       (add a loop-invariant constant)
// These forms are injective in the IV across the loop's iteration domain for
// any positive step, so a producer write at f(i) and a consumer read at f(i)
// hit the SAME address each iteration (true same-iteration transfer).
// Non-injective forms (i % 2, i & mask, dynamic gather/scatter indices,
// select-on-IV, ...) are NOT accepted: f(0)==f(2) would make the consumer, in
// the original program, read the producer's FINAL write while fusion makes it
// read the current-iteration write. A constant offset (no IV) is also not
// injective-affine here — it is the fixed-offset case (cross-iteration) and is
// blocked by the caller.
static bool isInjectiveAffineOffset(Value offset, ArrayRef<Value> ivs) {
  if (!offset)
    return false;
  if (llvm::is_contained(ivs, offset))
    return true; // bare IV
  Operation *def = offset.getDefiningOp();
  if (!def || def->getNumRegions() != 0)
    return false;
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    // IV * positive_constant. Either operand may be the IV; the other must be a
    // positive integer constant.
    Value lhs = mul.getLhs(), rhs = mul.getRhs();
    bool lhsIV = llvm::is_contained(ivs, lhs);
    bool rhsIV = llvm::is_contained(ivs, rhs);
    if (lhsIV == rhsIV)
      return false; // both IV or both non-IV -> not the accepted form
    Value constSide = lhsIV ? rhs : lhs;
    auto multiplier = getStaticInteger(constSide);
    return multiplier && *multiplier > 0;
  }
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    // <injective affine> + constant: one side must be injective affine, the
    // other a (loop-invariant) constant. The constant side may itself be any
    // loop-invariant value; we only require the affine side to be injective.
    bool lhsAffine = isInjectiveAffineOffset(add.getLhs(), ivs);
    bool rhsAffine = isInjectiveAffineOffset(add.getRhs(), ivs);
    if (lhsAffine == rhsAffine)
      return false; // require exactly one affine side + one constant side
    Value invariantSide = lhsAffine ? add.getRhs() : add.getLhs();
    return !offsetDependsOnIV(invariantSide, ivs);
  }
  return false;
}

static std::optional<int64_t>
getInjectiveAffineCoefficient(Value offset, ArrayRef<Value> ivs) {
  if (llvm::is_contained(ivs, offset))
    return 1;
  Operation *def = offset.getDefiningOp();
  if (!def || def->getNumRegions() != 0)
    return std::nullopt;
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    Value lhs = mul.getLhs(), rhs = mul.getRhs();
    bool lhsIV = llvm::is_contained(ivs, lhs);
    bool rhsIV = llvm::is_contained(ivs, rhs);
    if (lhsIV == rhsIV)
      return std::nullopt;
    auto multiplier = getStaticInteger(lhsIV ? rhs : lhs);
    if (!multiplier || *multiplier <= 0)
      return std::nullopt;
    return *multiplier;
  }
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = getInjectiveAffineCoefficient(add.getLhs(), ivs);
    auto rhs = getInjectiveAffineCoefficient(add.getRhs(), ivs);
    if (lhs && !offsetDependsOnIV(add.getRhs(), ivs))
      return lhs;
    if (rhs && !offsetDependsOnIV(add.getLhs(), ivs))
      return rhs;
  }
  return std::nullopt;
}

// A member of a fusion run: the scf.for plus the ops sitting between the
// previous member's for and this one, split by where they can legally land
// after fusion:
//   hoisted -> move before the fused for (loop-invariant: inputs available
//              before the run; UB reads not produced by any member)
//   sunk    -> move after the fused for (outputs not read inside any member)
struct Member {
  scf::ForOp loop;
  SmallVector<Operation *, 8> hoisted; // before fused for
  SmallVector<Operation *, 8> sunk;   // after fused for
};

// A cross-iteration UB dependency between the candidate and the existing run:
// the run writes UB W at offset f(i) and cand reads W at offset g(i) (or the
// symmetric write-in-cand / read-in-run case). Fusing into one body executed
// per iteration in source order makes cand, in iteration i, read whatever the
// run wrote in iteration i. That is correct ONLY when f(i)==g(i) for all i AND
// f is injective (no two iterations write the same address).
//
// Same-iteration fusion is allowed ONLY when BOTH offsets depend on the IV,
// BOTH are restricted injective affine forms (IV, IV*positive_const, +const),
// and the two are structurally equivalent (with all run IVs mapped to the fused
// IV). Everything else is blocked:
//   - fixed-offset transfer (neither side carries the IV): cross-iteration;
//     the consumer reads the producer's FINAL value, not the current-iteration
//     value. (reduce-final fixed-offset round trips handled by the between-op
//     stuck mechanism are not across loop bodies.)
//   - one side IV, one side not: misaligned stencil.
//   - non-injective affine (i % 2, dynamic gather/scatter indices): f(0)==f(2)
//     collides; consumer reads producer's final write, not current — block.
static bool hasCrossIterationUBDependency(ArrayRef<Member> members,
                                          scf::ForOp cand) {
  if (hasUnmodeledMemoryAccess(cand))
    return true;
  for (const Member &m : members) {
    scf::ForOp loop = m.loop;
    if (hasUnmodeledMemoryAccess(loop))
      return true;
  }
  SmallVector<Value, 8> runIVs;
  for (const Member &m : members) {
    scf::ForOp loop = m.loop; // copy to drop const (ForOp is a value wrapper)
    runIVs.push_back(loop.getInductionVar());
  }
  runIVs.push_back(cand.getInductionVar());

  SmallVector<UBAccess, 8> runAcc, candAcc;
  for (const Member &m : members) {
    scf::ForOp loop = m.loop;
    collectLoopUBAccesses(loop, runAcc);
  }
  collectLoopUBAccesses(cand, candAcc);
  scf::ForOp firstMemberLoop = members.front().loop;
  auto iterationStep = getStaticInteger(firstMemberLoop.getStep());

  auto isCrossIter = [&](const UBAccess &w, const UBAccess &r) -> bool {
    if (!accessRangesMayOverlap(w, r))
      return false;
    // A same numeric UB address exposed through different element types is a
    // byte-range alias, but the first fusion legality proof cannot normalize
    // the two element-index domains into one byte affine expression.  Keep
    // the original loop ordering rather than guessing.
    if (w.ub.address != r.ub.address || w.ub.memrefType != r.ub.memrefType)
      return true;
    bool wIV = offsetDependsOnIV(w.offset, runIVs);
    bool rIV = offsetDependsOnIV(r.offset, runIVs);
    // Only a per-iteration transfer where BOTH offsets carry the IV is a
    // candidate for same-iteration fusion. A fixed-offset transfer (neither
    // side carries the IV) is cross-iteration: the consumer reads the
    // producer's FINAL value, not the current-iteration value, so fusion
    // changes semantics — block it. A mix of IV and non-IV offsets is a
    // misaligned stencil — block it.
    if (!(wIV && rIV))
      return true;
    // Both offsets carry the IV. Require BOTH to be restricted injective
    // affine forms (IV, IV*positive_const, + const). Non-injective forms like
    // i % 2 collide across iterations (f(0)==f(2)) and are NOT same-iteration
    // even when structurally equivalent — block them.
    if (!isInjectiveAffineOffset(w.offset, runIVs) ||
        !isInjectiveAffineOffset(r.offset, runIVs))
      return true;
    // Both injective affine in the IV: same-iteration iff structurally
    // equivalent (all run IVs map to the single fused IV).
    if (!areEquivalentOffsetValues(w.offset, r.offset, runIVs))
      return true;

    // Injectivity of the scalar start address is not sufficient for a wide
    // access: offset=i with a 64-lane f32 vreg overlaps the next 63 logical
    // iterations.  Require the byte distance between adjacent iterations to
    // cover both accesses before changing loop-by-loop execution into
    // interleaved execution.
    auto coefficient = getInjectiveAffineCoefficient(w.offset, runIVs);
    auto elementBytes = getElementBytes(w.ub.memrefType);
    if (!coefficient || !iterationStep || *iterationStep <= 0 ||
        !elementBytes || w.accessBytes <= 0 || r.accessBytes <= 0)
      return true;
    if (*coefficient > INT64_MAX / *iterationStep ||
        *coefficient * *iterationStep > INT64_MAX / *elementBytes)
      return true;
    int64_t iterationDistanceBytes =
        *coefficient * *iterationStep * *elementBytes;
    return iterationDistanceBytes < std::max(w.accessBytes, r.accessBytes);
  };

  // run writes that cand reads:
  for (const auto &w : runAcc) {
    if (w.isLoad)
      continue;
    for (const auto &r : candAcc) {
      if (!r.isLoad || !mayAlias(w.ub, r.ub))
        continue;
      if (isCrossIter(w, r))
        return true;
    }
  }
  // cand writes that run reads:
  for (const auto &w : candAcc) {
    if (w.isLoad)
      continue;
    for (const auto &r : runAcc) {
      if (!r.isLoad || !mayAlias(w.ub, r.ub))
        continue;
      if (isCrossIter(w, r))
        return true;
    }
  }
  // Writes are order-sensitive too. Only the same injective location in the
  // same logical iteration preserves the original loop-by-loop WAW order.
  for (const auto &runWrite : runAcc) {
    if (runWrite.isLoad)
      continue;
    for (const auto &candWrite : candAcc) {
      if (candWrite.isLoad || !mayAlias(runWrite.ub, candWrite.ub))
        continue;
      if (isCrossIter(runWrite, candWrite))
        return true;
    }
  }
  return false;
};

// First-version iter-arg handling only concatenates independent loop-carried
// state. Reject a candidate that consumes any result of an earlier member,
// whether as an init arg or as a value captured in its body.
static bool hasMemberResultDependency(ArrayRef<Member> members,
                                      scf::ForOp cand) {
  SmallPtrSet<Value, 16> memberResults;
  for (const Member &member : members) {
    scf::ForOp loop = member.loop;
    for (Value result : loop.getResults())
      memberResults.insert(result);
  }
  bool dependent = false;
  cand->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!memberResults.count(operand))
        continue;
      dependent = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return dependent;
}

static SmallVector<scf::ForOp, 8> membersAsLoops(ArrayRef<Member> members) {
  SmallVector<scf::ForOp, 8> loops;
  for (const Member &m : members)
    loops.push_back(m.loop);
  return loops;
}

// PTO address/mask materializations predate consistent Pure traits but are
// side-effect-free and safe to relocate. Keep this exception list closed.
static bool isKnownRelocatablePure(Operation *op) {
  return isPure(op) ||
         isa<pto::CastPtrOp, pto::VMICreateMaskOp>(op);
}

// Can `op` be hoisted above the fused for (run before any member executes)?
// Inputs must be available before the run: no SSA use of any member's result,
// and no UB read of an address that some member writes (that would read a
// loop-produced value).
static bool canHoistAboveRun(Operation *op, ArrayRef<scf::ForOp> runLoops,
                             ArrayRef<UBId> runReads,
                             ArrayRef<UBId> runWrites) {
  // Closed-set relocation: pure regionless ops and the two explicitly modeled
  // unified memory ops are the only operations movable across a fusion run.
  if (op->getNumRegions() != 0 ||
      (!isKnownRelocatablePure(op) &&
       !isa<pto::VMIvLoadOp, pto::VMIvStoreOp>(op)))
    return false;
  SmallPtrSet<Value, 16> loopResults;
  for (scf::ForOp l : runLoops)
    for (Value r : l.getResults())
      loopResults.insert(r);
  for (Value opnd : op->getOperands())
    if (loopResults.count(opnd))
      return false;
  if (auto v = dyn_cast<pto::VMIvLoadOp>(op)) {
    auto id = getVLoadUB(v);
    if (!id || ubListContains(runWrites, *id))
      return false;
  }
  if (auto v = dyn_cast<pto::VMIvStoreOp>(op)) {
    auto id = getVStoreUB(v);
    if (!id || ubListContains(runReads, *id) ||
        ubListContains(runWrites, *id))
      return false;
  }
  return true;
}

// Can `op` be sunk below the fused for (run after all members execute)? Its
// UB writes must not be read inside any member (members run per iteration and
// would need the value). SSA outputs consumed inside members also block sink.
static bool canSinkBelowRun(Operation *op, ArrayRef<scf::ForOp> runLoops,
                            ArrayRef<UBId> runReads,
                            ArrayRef<UBId> runWrites) {
  if (op->getNumRegions() != 0 ||
      (!isKnownRelocatablePure(op) &&
       !isa<pto::VMIvLoadOp, pto::VMIvStoreOp>(op)))
    return false;
  for (Value res : op->getResults())
    for (OpOperand &use : res.getUses())
      for (scf::ForOp l : runLoops)
        if (l->isAncestor(use.getOwner()))
          return false;
  if (auto v = dyn_cast<pto::VMIvStoreOp>(op)) {
    auto id = getVStoreUB(v);
    if (!id || ubListContains(runReads, *id) ||
        ubListContains(runWrites, *id))
      return false;
  }
  if (auto v = dyn_cast<pto::VMIvLoadOp>(op)) {
    auto id = getVLoadUB(v);
    if (!id || ubListContains(runWrites, *id))
      return false;
  }
  return true;
}

// Partition between-ops into dependency-connected components. Two between-ops
// are in the same component if data flows between them within the between
// region: either SSA (one's result is used by another), or UB (a vstore's
// written UB is read by a later vload). Each component must be placed AS A
// WHOLE after fusion (all hoisted above the fused for, or all sunk below it).
// Components are returned in source order; each component's ops are in source
// order.
static SmallVector<SmallVector<Operation *, 4>, 8>
partitionBetween(ArrayRef<Operation *> between) {
  unsigned n = between.size();
  SmallVector<unsigned, 8> parent(n);
  for (unsigned i = 0; i < n; ++i)
    parent[i] = i;
  auto find = [&](unsigned x) -> unsigned {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](unsigned a, unsigned b) {
    unsigned ra = find(a), rb = find(b);
    if (ra != rb)
      parent[ra] = rb;
  };
  DenseMap<Operation *, unsigned> idx;
  for (unsigned i = 0; i < n; ++i)
    idx[between[i]] = i;
  SmallVector<std::optional<UBId>, 8> writes(n);
  for (unsigned i = 0; i < n; ++i)
    if (auto v = dyn_cast<pto::VMIvStoreOp>(between[i]))
      writes[i] = getVStoreUB(v);
  for (unsigned j = 0; j < n; ++j) {
    Operation *opj = between[j];
    for (Value opnd : opj->getOperands()) {
      Operation *def = opnd.getDefiningOp();
      if (!def)
        continue;
      auto it = idx.find(def);
      if (it != idx.end() && it->second < j)
        unite(it->second, j);
    }
    if (auto v = dyn_cast<pto::VMIvLoadOp>(opj)) {
      if (auto id = getVLoadUB(v)) {
        for (unsigned i = 0; i < j; ++i)
          if (writes[i] && mayAlias(*writes[i], *id))
            unite(i, j);
      }
    }
  }
  SmallVector<SmallVector<unsigned>, 8> byRoot(n);
  for (unsigned i = 0; i < n; ++i)
    byRoot[find(i)].push_back(i);
  SmallVector<unsigned, 8> roots;
  for (unsigned i = 0; i < n; ++i)
    if (find(i) == i)
      roots.push_back(i);
  llvm::sort(roots, [&](unsigned a, unsigned b) {
    return byRoot[a].front() < byRoot[b].front();
  });
  SmallVector<SmallVector<Operation *, 4>, 8> comps;
  for (unsigned r : roots) {
    SmallVector<Operation *, 4> comp;
    for (unsigned i : byRoot[r])
      comp.push_back(between[i]);
    comps.push_back(std::move(comp));
  }
  return comps;
}

// Split the region body's op list into members. A run grows by adding the
// next same-header for ONLY IF every op between the previous member's for and
// the candidate for can be legally placed after fusion — hoisted above the
// fused for, or sunk below it. Between-ops sit outside any for body
// originally, so they are not per-iteration and cannot be cloned into the
// fused body (that would change their execution count). If any between-op is
// stuck (can hoist neither above nor below — e.g. it reads a preceding
// reduce's final UB result AND its output is read inside a following member),
// the run stops: the stuck op and the following for start a separate run, so
// two for's separated by a stuck op are not fused into one iteration.
static SmallVector<Member, 8> collectRun(Block &body,
                                         SmallVectorImpl<scf::ForOp> &loops,
                                         unsigned firstLoopIdx) {
  SmallVector<Member, 8> members;
  scf::ForOp first = loops[firstLoopIdx];
  if (!isTileLibVmiPrincipalLoop(first))
    return members;
  members.push_back(Member{first, {}, {}});

  Operation *betweenStart = first->getNextNode();
  for (unsigned i = firstLoopIdx + 1; i < loops.size(); ++i) {
    scf::ForOp cand = loops[i];
    if (!isTileLibVmiPrincipalLoop(cand)) {
      break;
    }
    if (!sameHeader(first, cand)) {
      break;
    }

    if (hasMemberResultDependency(members, cand)) {
      break;
    }

    // Between-ops: [betweenStart, cand).
    SmallVector<Operation *, 8> between;
    for (Operation *op = betweenStart; op && op != cand;
         op = op->getNextNode())
      between.push_back(op);

    // UB read/written by the full run if cand joins (members + cand).
    SmallVector<UBId, 8> runReads, runWrites;
    for (Member &m : members)
      collectLoopUBs(m.loop, runReads, runWrites);
    SmallVector<UBId, 8> candReads, candWrites;
    collectLoopUBs(cand, candReads, candWrites);
    runReads.append(candReads.begin(), candReads.end());
    runWrites.append(candWrites.begin(), candWrites.end());

    SmallVector<scf::ForOp, 8> fullLoops = membersAsLoops(members);
    fullLoops.push_back(cand);

    // Reject unmodeled memory accesses and every UB exchange that is not a
    // proven injective same-iteration transfer before relocating between-ops.
    if (hasCrossIterationUBDependency(members, cand)) {
      break;
    }

    // Partition between-ops into dependency components (SSA def-use or same-UB
    // store->load). Each component must be placed AS A WHOLE: all hoisted above
    // the fused for, or all sunk below it (splitting a component would break its
    // internal dataflow). A component is stuck if it can neither hoist (some op
    // reads a member-produced UB / result) nor sink (some op's UB write / result
    // is used inside a member). If any component is stuck, the run stops here.
    SmallVector<SmallVector<Operation *, 4>, 8> comps =
        partitionBetween(between);
    bool stuck = false;
    for (const SmallVector<Operation *, 4> &comp : comps) {
      bool compHoist = true, compSink = true;
      for (Operation *op : comp) {
        if (!canHoistAboveRun(op, fullLoops, runReads, runWrites)) {
          compHoist = false;
        }
        if (!canSinkBelowRun(op, fullLoops, runReads, runWrites)) {
          compSink = false;
        }
      }
      if (!compHoist && !compSink) {
        stuck = true;
        break;
      }
    }
    if (stuck)
      break;

    // Commit cand. Each component goes to the bucket it can: hoist if
    // compHoist, else sink (compSink must hold here).
    Member &last = members.back();
    for (const SmallVector<Operation *, 4> &comp : comps) {
      bool compHoist = true;
      for (Operation *op : comp)
        if (!canHoistAboveRun(op, fullLoops, runReads, runWrites)) {
          compHoist = false;
          break;
        }
      if (compHoist) {
        for (Operation *op : comp)
          last.hoisted.push_back(op);
      } else {
        for (Operation *op : comp)
          last.sunk.push_back(op);
      }
    }
    members.push_back(Member{cand, {}, {}});
    betweenStart = cand->getNextNode();
  }
  return members;
}

// Build the fused scf.for for a run of members. Members are erased by caller.
static scf::ForOp buildFusedLoop(OpBuilder &builder,
                                 MutableArrayRef<Member> members) {
  scf::ForOp firstLoop = members.front().loop;
  Location loc = firstLoop.getLoc();

  // Fused init args = concatenation of each member's init args.
  SmallVector<Value, 8> fusedInitArgs;
  for (Member &m : members)
    fusedInitArgs.append(m.loop.getInitArgs().begin(),
                         m.loop.getInitArgs().end());

  SmallVector<IRMapping, 8> mappings(members.size());

  auto bodyBuilder = [&](OpBuilder &b, Location bl, Value iv,
                         ValueRange iterArgs) {
    unsigned iterOffset = 0;
    for (auto [idx, m] : llvm::enumerate(members)) {
      mappings[idx].map(m.loop.getInductionVar(), iv);
      unsigned nArgs = m.loop.getRegionIterArgs().size();
      for (unsigned k = 0; k < nArgs; ++k)
        mappings[idx].map(m.loop.getRegionIterArgs()[k],
                          iterArgs[iterOffset + k]);
      iterOffset += nArgs;
    }

    // Per member: clone only its body (without scf.yield) in source order.
    // Between-ops that were loop-invariant (hoisted bucket) are moved before
    // the fused for below; those whose output no member reads (sunk bucket)
    // are moved after it. Body uses of hoisted values resolve to the top-level
    // originals via lookupOrDefault.
    for (auto [idx, m] : llvm::enumerate(members)) {
      Block &mbody = *m.loop.getBody();
      for (Operation &op : mbody.without_terminator())
        b.clone(op, mappings[idx]);
    }

    // Fused yield = concatenation of each member's yield operands, mapped.
    SmallVector<Value, 8> fusedYield;
    for (auto [idx, m] : llvm::enumerate(members)) {
      auto y = cast<scf::YieldOp>(m.loop.getBody()->getTerminator());
      for (Value v : y.getOperands())
        fusedYield.push_back(mappings[idx].lookupOrDefault(v));
    }
    b.create<scf::YieldOp>(bl, fusedYield);
  };

  auto fused = builder.create<scf::ForOp>(
      loc, firstLoop.getLowerBound(), firstLoop.getUpperBound(),
      firstLoop.getStep(), fusedInitArgs, bodyBuilder);
  fused->setAttrs(
      DictionaryAttr::get(fused.getContext(), getSemanticLoopAttrs(firstLoop)));
  fused->setAttr("pto.tilelib.impl", builder.getStringAttr("vmi"));
  fused->setAttr("pto.vmi.fusion.source", builder.getStringAttr("tilelib"));
  fused->setAttr("pto.vmi.fusion.principal_loop", builder.getUnitAttr());

  // Map each member's results to the corresponding slice of the fused loop's
  // results so external (top-level) users can be rewired.
  unsigned resOffset = 0;
  for (auto [idx, m] : llvm::enumerate(members)) {
    for (Value r : m.loop.getResults())
      mappings[idx].map(r, fused.getResults()[resOffset++]);
  }

  // Rewire external uses of each member's results to the fused results.
  resOffset = 0;
  for (auto [idx, m] : llvm::enumerate(members)) {
    for (auto [res, fusedRes] :
         llvm::zip(m.loop.getResults(),
                   fused.getResults().slice(
                       resOffset, m.loop.getNumResults()))) {
      res.replaceAllUsesWith(fusedRes);
    }
    resOffset += m.loop.getNumResults();
  }

  // Place between-ops and init-arg producers:
  //  - hoisted bucket (loop-invariant) and init-arg producers -> move before
  //    the fused for so they dominate the body / init args.
  //  - sunk bucket (outputs not read by any member) -> move after the fused
  //    for.
  // These ops are NOT cloned, so each UB materialization stays materialized
  // once. A later CSE dedups remaining duplicates.
  SmallVector<Operation *, 16> hoistOrder, sinkOrder;
  SmallPtrSet<Operation *, 32> seen;
  auto gather = [&](Operation *op) {
    if (!op || op == fused || op->getParentOp() != fused->getParentOp())
      return;
    if (seen.insert(op).second)
      hoistOrder.push_back(op);
  };
  auto gatherSink = [&](Operation *op) {
    if (!op || op == fused || op->getParentOp() != fused->getParentOp())
      return;
    if (seen.insert(op).second)
      sinkOrder.push_back(op);
  };
  for (Member &m : members) {
    for (Operation *pre : m.hoisted)
      gather(pre);
    for (Operation *sop : m.sunk)
      gatherSink(sop);
    for (Value ia : m.loop.getInitArgs())
      if (Operation *def = ia.getDefiningOp())
        gather(def);
  }
  for (Operation *op : hoistOrder)
    if (!op->isBeforeInBlock(fused))
      op->moveBefore(fused);
  for (Operation *op : sinkOrder)
    if (op->isBeforeInBlock(fused))
      op->moveAfter(fused);

  // Erase the member for ops (between-ops are kept, only the for ops go away).
  for (Member &m : llvm::reverse(members))
    m.loop.erase();

  return fused;
}

// Fuse one maximal run of same-header scf.for starting at firstLoopIdx.
// collectRun stops the run at a between-op that can neither hoist above nor
// sink below the run (a reduce-final UB dependency), so the fused run only
// spans for's whose between-ops are all placeable. Returns true if a fusion
// happened (>=2 members).
static bool fuseRun(Block &body, SmallVectorImpl<scf::ForOp> &loops,
                    unsigned firstLoopIdx) {
  SmallVector<Member, 8> members = collectRun(body, loops, firstLoopIdx);
  if (members.size() < 2)
    return false;
  OpBuilder builder(members.front().loop);
  buildFusedLoop(builder, members);
  return true;
}

struct PTOVmiLoopFusionPass
    : public mlir::pto::impl::PTOVmiLoopFusionBase<PTOVmiLoopFusionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](pto::FusionRegionOp region) {
      bool progressed = true;
      while (progressed) {
        progressed = false;
        Block &body = region.getBody().front();
        SmallVector<scf::ForOp, 16> loops;
        for (Operation &op : body.getOperations())
          if (auto f = dyn_cast<scf::ForOp>(op))
            loops.push_back(f);

        for (unsigned i = 0; i < loops.size();) {
          if (fuseRun(body, loops, i)) {
            progressed = true;
            break; // re-collect after mutation
          }
          ++i;
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVmiLoopFusionPass() {
  return std::make_unique<PTOVmiLoopFusionPass>();
}
