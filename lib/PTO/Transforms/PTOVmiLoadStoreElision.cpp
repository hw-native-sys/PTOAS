// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//
// PTOVmiLoadStoreElision.cpp - forward and elide vmi loads/stores
//===----------------------------------------------------------------------===//
//
// Adapted from PTOFusionLoadStoreElision for the unified VMI path. Inside each
// pto.fusion_region, a TWO-PASS scan over the single-layer scf.for leaf body
// (fused by PTOVmiLoopFusion) and the top-level straight-line segments between
// for's builds a content-version table of every vmi.vload / vmi.vstore, then
// eliminates redundant ones in reverse order.
//
// FIRST-VERSION LEGALITY SCOPE (conservative; "correct first, broaden later"):
//   - Only continuous, single-result (load) / single-value (store) vmi.vload/
//     vmi.vstore with no stride / block_stride / repeat_stride / group /
//     dist_mode and (for stores) no updated_base (post_update) result are
//     modeled. Any other shape (dintlv dual load, group load/store, block-
//     strided load/store, multi-value store, post_update store) is left alone:
//     loads become non-matchable entries that flush the table; stores flush the
//     table and are not registered as forward targets.
//   - A vload/vstore base must resolve to a COMPILE-TIME or AFFINE UB identity: a
//     pto.castptr -> memref -> pto.pointer_cast of a constant address, or of an
//     affine address `addi(muli(%iv, c), b)` from a loop induction variable. Two
//     bases with the same affine key (iv, baseOffset, coeff) are the same UB every
//     iteration, so a store->load on the same affine base may forward. A base that
//     is a runtime pointer (block argument / untraceable / non-affine value)
//     CANNOT be soundly matched: two such values comparing equal under SSA is not
//     a proof of identity, so a store->load on the same runtime base must NOT
//     forward. Such a base never acts as a forward source/target.
//   - A vload is a PURE READ: it never changes memory, so it must NOT invalidate
//     the CONTENT of any tracked store (a later constant/affine load may still
//     forward to a preceding store). But a vload MAY OBSERVE a preceding store's
//     UB, so any store whose base MAY alias the load's base is marked non-erasable
//     (a later overwrite-DSE must not delete it). A vstore is a WRITE: an
//     untrackable store flushes tracked content, while a trackable store marks
//     every may-alias entry stale (an affine store may alias any tracked UB).
//     Only must-alias, same-location writes may additionally prove an earlier
//     store dead.
//   - Transparency is decided by a CLOSED policy, not dialect prefixes:
//       * region-bearing op, func.call, vload/vstore, and the explicit sync/DMA
//         name set (mte_*/set_flag/mem_bar/...) are NEVER transparent;
//       * an op implementing MemoryEffectOpInterface is transparent ONLY if it
//         declares no Read/Write effect (catches vgather/vscatter/masked_load/
//         group_store/...);
//       * an op WITHOUT the interface is transparent ONLY if it is explicitly
//         Pure (mlir::isPure) — admits the VMI compute ops (vmuls/vcvt/...),
//         pointer_cast/castptr/create_mask/broadcast/arith/...; any UNKNOWN op
//         that forgot to declare effects is treated as impure and flushes.
//   - A vload's read-lane set is inferred from its consumers ONLY for consumers
//     in a closed whitelist of SEMANTICALLY-KNOWN ops:
//       * masked elementwise/reduce (vmuls/vadd/vmax/...): mask predicates the
//         data lanes -> read set = mask prefix [0,N);
//       * mask-free pure compute (vcvt/vselr/vinterpret_cast): read set = full
//         vreg;
//       * a vmi.vstore (as a LOAD consumer): reads its value operand on ALL
//         lanes -> full vreg.
//     Any OTHER consumer — vsel (mask routes output but BOTH values are read on
//     all lanes), select, compress_store, region-bearing, unknown — forces no
//     forward. A mix of masked + mask-free whitelisted consumers also forces no
//     forward (a partial merge store covering only masked lanes would be partly
//     read by the mask-free consumer).
//
// Canonical base resolution traces pto.castptr -> memref -> pto.pointer_cast
// -> addr, decomposing the addr into a constant or an affine (iv, baseOffset,
// coeff) key so that distinct castptr chains to the same compile-time or same
// affine UB compare equal. A vmi.vload has no mask operand, so its read lane set
// is inferred from its consuming op: if all consumers share one mask, that mask
// bounds the read set; if all are mask-free (e.g. vcvt) the read set is the full
// vreg; otherwise (mixed, or an unresolvable mask) the vload is left alone. A
// vmi.vstore carries its own mask and a pmode ("zero" default | "merge"): the
// store's write lane set is the mask's prefix [0,N); under pmode=merge only those
// lanes are written (inactive lanes keep the prior UB content), under pmode=zero
// the whole region is defined (inactive lanes store 0).
//
// Lane sets are modeled as prefix intervals [0,N) (create_mask %N is a prefix
// predicate); masks that cannot be statically resolved (constant_mask, masked
// combinations, non-constant active_lanes) are treated as "unknown" and the
// elision conservatively skips any vload/vstore whose lane set is unknown.
//
// Two passes:
//   Pass 1 (build, forward scan): record each load/store with its (base,
//     offset, lane-set, source value) and mark forward targets:
//       - a vload whose read set is fully covered by a preceding store's write
//         set, with no intervening intersecting write, forwards to that store's
//         value (store->load elision, the store is erased only if dead);
//       - a vload whose read set equals a preceding vload's read set, with no
//         intervening intersecting write, forwards to that load's result
//         (vload->vload dedup);
//       - a store fully overwritten by a later same-base/offset store whose
//         write set covers it is marked dead-store-erase.
//     A merge store invalidates only the lane interval it writes among the
//     preceding entries (a preceding entry fully covered by the merge write
//     set is dead; a partially intersecting one is marked stale so it no
//     longer participates in matching, but is retained so a later vload of the
//     mixed content correctly does NOT forward). A store whose UB is read by a
//     region-escaping op (mte_ub_gm/mte_gm_ub) is marked non-erasable.
//   Pass 2 (eliminate, reverse): for each marked entry, replace the load's
//     uses with the recorded source value and erase the dead loads/stores in
//     reverse order (so a value consumed by a later-forwarded op is replaced
//     before that op is erased). erase is guarded by use_empty().
//
// Runs in the VMI semantic pipeline AFTER PTOVmiLoopFusion + CSE (so cross-for
// UB round trips have become same-block straight-line pairs inside the fused
// loop) and before VMILowerUnifiedToLegacy.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VmiMemoryLocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVMILOADSTOREELISION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isTileLibVmiPrincipalLoop(scf::ForOp loop) {
  auto impl = loop->getAttrOfType<StringAttr>("pto.tilelib.impl");
  auto source = loop->getAttrOfType<StringAttr>("pto.vmi.fusion.source");
  return impl && impl.getValue() == "vmi" && source &&
         source.getValue() == "tilelib" &&
         loop->hasAttr("pto.vmi.fusion.principal_loop") &&
         !loop->hasAttr("pto.vmi.fusion.boundary");
}

// Trace a value through the vmi UB alias chain to its canonical root: a
// pto.castptr (memref->ptr) whose memref is a pto.pointer_cast of a constant
// address. For values not on this chain, return the value itself.
static Value getCanonicalTrackedValue(Value value) {
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def)
      break;
    if (auto cp = dyn_cast<pto::CastPtrOp>(def)) {
      value = cp->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

// ----------------------------------------------------------------------------
// Affine UB address model.
//
// A vload/vstore base may resolve to a COMPILE-TIME-DERIVABLE affine address
// instead of a bare constant: `pointer_cast(addi(muli(%iv, cK), cB))` from a
// loop-carried induction variable. Such an address is a deterministic function
// of the iteration, so two accesses built from the SAME induction variable and
// the SAME (coeff, base) compare equal — they must alias. This lets a
// store->load round trip on the same affine UB be folded, and lets us reason
// about whether a dynamic store can alias a tracked constant store.
//
// The affine key is (iv, baseOffset, coeff); two affine addresses alias iff
// they share the same iv and (baseOffset, coeff) differ by a multiple of the
// vector-interval width (so the intervals on this iteration overlap). Because
// the model is conservative (an affine address may alias ANY constant UB on
// some iteration), affine keys are used for MUST-ALIAS (same iv+offset) but
// never for must-not-alias against a constant.
// ----------------------------------------------------------------------------
struct AffineAddr {
  Value iv;                 // loop induction variable (null => pure constant)
  int64_t baseOffset = 0;   // constant part
  int64_t coeff = 0;        // iv multiplier
  bool isAffine = false;    // valid affine form (not a bare runtime ptr)
};

// Peel type-preserving/unary casts that wrap an affine expression or its IV:
// arith.index_cast is the common wrapper in the VMI path (loop IV is index,
// the address is i64). The peeled value is the affine operand.
static Value peelAffineWrappers(Value v) {
  // Bound the peel to avoid pathological cycles in the def chain.
  unsigned steps = 0;
  while (v && steps++ < 16) {
    Operation *def = v.getDefiningOp();
    if (!def)
      break;
    if (auto ic = dyn_cast<arith::IndexCastOp>(def)) {
      Value in = ic.getIn();
      if (in == v)
        break;
      v = in;
      continue;
    }
    break;
  }
  return v;
}

// Parse an address Value into an AffineAddr. Handles:
//   constant              -> {iv=null, baseOffset=c}
//   addi(muli(%iv, c), b) -> {iv, baseOffset=b, coeff=c}
//   addi(%iv, b)          -> {iv, baseOffset=b, coeff=1}
//   muli(%iv, c)          -> {iv, baseOffset=0, coeff=c}
// The IV and the addr may be wrapped in index_cast (peeled). The IV is stored
// as its peeled value; equality is decided by areEquivalentValues (two casts of
// the same induction value are equivalent).
// Anything else (block arg, untraceable) -> isAffine=false.
static AffineAddr parseAffineAddr(Value addr) {
  AffineAddr result;
  if (!addr)
    return result;
  addr = peelAffineWrappers(addr);
  if (auto c = addr.getDefiningOp<arith::ConstantOp>()) {
    if (auto iv = dyn_cast<IntegerAttr>(c.getValue())) {
      result.isAffine = true;
      result.coeff = 0;
      result.baseOffset = iv.getInt();
      return result;
    }
    return result;
  }
  if (auto addi = dyn_cast<arith::AddIOp>(addr.getDefiningOp())) {
    Value lhs = addi.getLhs();
    Value rhs = addi.getRhs();
    // addi(muli(%iv, c), b): affine in RHS.
    if (auto muli = dyn_cast<arith::MulIOp>(rhs.getDefiningOp())) {
      Value mulLhs = peelAffineWrappers(muli.getLhs());
      Value mulRhs = peelAffineWrappers(muli.getRhs());
      // The multiplier operand must be a constant; the other is the IV.
      arith::ConstantOp cst = mulRhs.getDefiningOp<arith::ConstantOp>();
      Value iv = mulLhs;
      if (!cst) {
        cst = mulLhs.getDefiningOp<arith::ConstantOp>();
        iv = mulRhs;
      }
      arith::ConstantOp b = lhs.getDefiningOp<arith::ConstantOp>();
      if (cst && b) {
        if (auto cval = dyn_cast<IntegerAttr>(cst.getValue())) {
          if (auto bval = dyn_cast<IntegerAttr>(b.getValue())) {
            result.isAffine = true;
            result.iv = iv;
            result.coeff = cval.getInt();
            result.baseOffset = bval.getInt();
            return result;
          }
        }
      }
    }
    // addi(%iv, b): affine coeff 1.
    if (auto b = rhs.getDefiningOp<arith::ConstantOp>()) {
      if (auto bval = dyn_cast<IntegerAttr>(b.getValue())) {
        result.isAffine = true;
        result.iv = peelAffineWrappers(lhs);
        result.coeff = 1;
        result.baseOffset = bval.getInt();
        return result;
      }
    }
    return result;
  }
  if (auto muli = dyn_cast<arith::MulIOp>(addr.getDefiningOp())) {
    // muli(%iv, c): affine coeff c, base 0.
    Value mulLhs = peelAffineWrappers(muli.getLhs());
    Value mulRhs = peelAffineWrappers(muli.getRhs());
    // The multiplier operand must be a constant; the other is the IV.
    arith::ConstantOp cst = mulRhs.getDefiningOp<arith::ConstantOp>();
    Value iv = mulLhs;
    if (!cst) {
      cst = mulLhs.getDefiningOp<arith::ConstantOp>();
      iv = mulRhs;
    }
    if (cst) {
      if (auto cval = dyn_cast<IntegerAttr>(cst.getValue())) {
        result.isAffine = true;
        result.iv = iv;
        result.coeff = cval.getInt();
        result.baseOffset = 0;
        return result;
      }
    }
    return result;
  }
  return result;
}

// A canonical, comparable identity for a vload/vstore base: either a constant
// pointer_cast address, or an affine (iv, baseOffset, coeff) key. Two bases
// compare equal (must-alias) iff their constants are equal, or their affine key
// is equal (same iv, same baseOffset, same coeff). This is the compile-time
// identity used for forwarding matches.
struct BaseIdentity {
  Value base;          // canonical base (pointer_cast result), or null if untrackable
  AffineAddr affine;   // affine decomposition (isAffine==false => untrackable)
  bool isConstantAddr; // resolved to a bare constant pointer_cast
  std::optional<pto::VmiStorageRoot> storageRoot;
};

// Resolve a vload/vstore base to a BaseIdentity. Returns an untrackable identity
// (base==null) for block args / untraceable / non-affine bases.
static BaseIdentity resolveBaseIdentity(Value base) {
  if (auto root = pto::resolveVmiStorageRoot(base))
    return {base, {}, true, root};
  if (auto cast = base.getDefiningOp<pto::CastPtrOp>()) {
    Value input = cast.getInput();
    if (isa<IntegerType>(input.getType()) || input.getType().isIndex()) {
      AffineAddr affine = parseAffineAddr(input);
      if (affine.isAffine)
        return {base, affine, affine.iv == nullptr, std::nullopt};
    }
    // CastPtrOp(memref) -> PointerCastOp(addr): trace through the
    // pointer_cast to its integer address operand for affine parsing.
    if (auto pc = input.getDefiningOp<pto::PointerCastOp>()) {
      auto addrs = pc.getAddrs();
      if (!addrs.empty()) {
        AffineAddr affine = parseAffineAddr(addrs[0]);
        if (affine.isAffine)
          return {base, affine, affine.iv == nullptr, std::nullopt};
      }
    }
  }
  Value canon = getCanonicalTrackedValue(base);
  if (!canon)
    return {};
  if (isa<BlockArgument>(canon))
    return {};
  return {};
}
static bool isTrackableIdentity(const BaseIdentity &id) {
  return id.base != nullptr;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  Value cl = getCanonicalTrackedValue(lhs);
  Value cr = getCanonicalTrackedValue(rhs);
  if (cl == cr)
    return true;
  if (!cl || !cr)
    return false;
  if (cl.getType() != cr.getType())
    return false;
  Operation *ld = cl.getDefiningOp();
  Operation *rd = cr.getDefiningOp();
  if (!ld || !rd)
    return false;
  // Two identical pure ops (same name, operands, attrs, result type) — e.g.
  // two pto.vmi.create_mask with the same active_lanes constant. This is what
  // makes masks produced by distinct create_mask ops compare equal.
  if (ld->getName() == rd->getName() && ld->getNumRegions() == 0 &&
      rd->getNumRegions() == 0 &&
      ld->getNumOperands() == rd->getNumOperands() &&
      ld->getAttrDictionary() == rd->getAttrDictionary() &&
      llvm::equal(ld->getOperandTypes(), rd->getOperandTypes())) {
    for (auto [a, b] :
         llvm::zip(ld->getOperands(), rd->getOperands())) {
      if (!areEquivalentValues(a, b))
        return false;
    }
    return true;
  }
  // Two identical pure constants.
  if (isa<arith::ConstantOp>(ld) && isa<arith::ConstantOp>(rd))
    return ld->getAttrDictionary() == rd->getAttrDictionary();
  return cl == cr;
}

static bool areEquivalentMaskValues(Value lhs, Value rhs) {
  return areEquivalentValues(lhs, rhs);
}

// Whether two base identities reference the SAME UB on this iteration (must
// alias). This is the identity used for content forwarding: two accesses with
// must-alias bases read/write the same bytes, so a store->load forward is safe.
//   - two constant pointer_casts: must-alias iff the same address;
//   - two affine keys: must-alias iff the same iv and the same (baseOffset,
//     coeff) — they compute the identical address on every iteration;
//   - constant vs affine, or any untrackable: NOT must-alias (may differ).
static bool basesMustAlias(const BaseIdentity &lhs, const BaseIdentity &rhs) {
  if (!isTrackableIdentity(lhs) || !isTrackableIdentity(rhs))
    return false;
  if (lhs.storageRoot && rhs.storageRoot)
    return *lhs.storageRoot == *rhs.storageRoot;
  if (areEquivalentValues(lhs.base, rhs.base))
    return true;
  if (lhs.affine.isAffine && rhs.affine.isAffine)
    return areEquivalentValues(lhs.affine.iv, rhs.affine.iv) &&
           lhs.affine.baseOffset == rhs.affine.baseOffset &&
           lhs.affine.coeff == rhs.affine.coeff;
  return false;
}

// Whether two base identities MAY alias on some iteration. Conservative:
//   - two constant pointer_casts: alias iff the same address (a constant never
//     sweeps a range). Note `isAffine` is also true for a bare constant (iv is
//     null), so the constant/constant case must be decided by `isConstantAddr`
//     BEFORE the affine sweep rule, or two distinct constants would be
//     misclassified as may-alias.
//   - an affine base may alias any constant UB on some iteration (its address
//     sweeps a range as the loop runs), so it is treated as may-alias with a
//     constant;
//   - any untrackable base (block arg / non-affine) may alias anything.
// This drives the nonErasable marking: a store whose UB a later load MAY read
// must not be deleted by overwrite-DSE even after its value is forwarded.
static bool basesMayAlias(const BaseIdentity &lhs, const BaseIdentity &rhs) {
  if (!isTrackableIdentity(lhs) || !isTrackableIdentity(rhs))
    return true; // unknown may alias anything
  if (lhs.storageRoot && rhs.storageRoot)
    return pto::mayAliasVmiStorageRoot(*lhs.storageRoot, *rhs.storageRoot);
  // Both bare constants: alias iff the same address. This must be checked
  // first, because a bare constant carries isAffine==true (iv==null) and would
  // otherwise fall through to the affine-sweep rule below.
  if (lhs.isConstantAddr && rhs.isConstantAddr)
    return areEquivalentValues(lhs.base, rhs.base);
  if (lhs.affine.isAffine || rhs.affine.isAffine)
    return true; // affine sweeps a range -> may hit a constant
  return false;
}

// ----------------------------------------------------------------------------
// Lane-set modeling (prefix interval [0,N)).
//
// create_mask %N is a prefix predicate: lanes [0,N) are active. Lane sets are
// therefore representable as a prefix interval [0,N). A vload reads the full
// vreg [0,VL) (VL = vreg element count) unless its inferred consumer mask bounds
// the read to [0,N) <= [0,VL). A vstore writes its mask's prefix [0,N); under
// pmode=merge only those lanes are written (inactive lanes keep prior UB
// content), under pmode=zero the whole region is defined (inactive lanes store
// 0). Masks that cannot be statically resolved (constant_mask, masked
// combinations, non-constant active_lanes) yield Unknown and the elision
// conservatively skips the affected vload/vstore.
// ----------------------------------------------------------------------------
struct LaneRange {
  // Inclusive upper bound of the prefix [0, upperBound). std::nullopt means the
  // full set [0, VL) (i.e. a mask-free / full vreg read). isUnknown marks a
  // mask we cannot reason about — nothing involving it is forwardable.
  std::optional<unsigned> upperBound;
  bool isUnknown = false;

  static LaneRange full() { return {std::nullopt, false}; }
  static LaneRange unknown() { return {std::nullopt, true}; }
  static LaneRange prefix(unsigned n) { return {n, false}; }

  bool isFull() const { return !isUnknown && !upperBound.has_value(); }
  bool isUnknownSet() const { return isUnknown; }

  // Does this lane set contain (cover) `other`? Unknown never covers or is
  // covered (conservatively not a subset/superset).
  bool contains(const LaneRange &other) const {
    if (isUnknown || other.isUnknown)
      return false;
    if (isFull())
      return true;
    if (other.isFull())
      return false;
    return *upperBound >= *other.upperBound;
  }
  // Do the two lane sets intersect? Unknown => conservatively intersects.
  // Two non-empty prefix intervals [0,A) and [0,B) always share lane 0.
  bool intersects(const LaneRange &other) const {
    if (isUnknown || other.isUnknown)
      return true;
    if (isFull() || other.isFull())
      return true;
    return *upperBound > 0 && *other.upperBound > 0;
  }
};

// Resolve a mask Value to a prefix LaneRange. create_mask %constN -> [0,N).
// Anything else (constant_mask, mask_and, non-const active_lanes) -> unknown.
static LaneRange resolveMaskLanes(Value mask) {
  if (!mask)
    return LaneRange::full(); // no mask operand => full predicate
  Operation *def = mask.getDefiningOp();
  if (auto cm = dyn_cast<pto::VMICreateMaskOp>(def)) {
    if (auto c =
            cm.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      if (auto iv = dyn_cast<IntegerAttr>(c.getValue())) {
        int64_t n = iv.getInt();
        if (n >= 0)
          return LaneRange::prefix(static_cast<unsigned>(n));
      }
    }
  }
  return LaneRange::unknown();
}

// The vreg width (VL) of a vload result, for bounding a full read. Returns 0
// if not a vmi.vreg type.
static unsigned getVRegWidth(Type t) {
  if (auto vt = dyn_cast<pto::VMIVRegType>(t))
    return static_cast<unsigned>(vt.getElementCount());
  return 0;
}

// Classification of a vload's consumer for read-lane inference:
//   MaskInferable   — a masked elementwise/reduce compute op whose mask is a
//                     TRUE predicate on the data lanes it reads/writes. The
//                     vload is read only on the mask's prefix [0,N).
//   MaskFreeFullRead — a pure, mask-free compute op (vcvt/vselr/vinterpret_cast)
//                     that reads the FULL vreg on every lane.
//   NotKnown         — anything else (vsel, where the mask routes the output but
//                     BOTH values are read on all lanes; select; compress_store;
//                     region-bearing ops; unknown ops). The vload's read set
//                     cannot be bounded — forwarding is disabled for the load.
//
// vmuls/vadds/... come from the VMI_VecScalarOp template; vadd/vmul/... are the
// direct VMI_Op<"v*"> elementwise/reduce ops. Both classes take (vreg, [scalar,]
// mask) and the mask directly predicates the data lanes.
//
// vsel is EXCLUDED on purpose: its mask selects between true/false_value, but
// BOTH values are read on ALL lanes (mask only routes the output), so a vsel
// consumer reads the full vreg — its mask must NOT bound the vload's read set.
// Treating it as NotKnown disables forwarding for any load feeding a vsel.
enum class ConsumerKind { NotKnown, MaskInferable, MaskFreeFullRead };
static ConsumerKind classifyLoadConsumer(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return ConsumerKind::NotKnown;
  StringRef name = op->getName().getStringRef();
  // A vmi.vstore reads its value operand(s) on ALL lanes (the mask only
  // governs which lanes are written OUT; the value vreg is consumed in full
  // to produce the written data, incl. under pmode=merge where inactive lanes
  // retain prior UB content but the value is still read). So as a LOAD
  // consumer a vstore is a full-vreg read — regardless of the store's own
  // shape (continuous / group / block-stride). (The store's UB write is
  // handled separately in the store branch; here we only classify its read
  // of the vreg operand.)
  if (isa<pto::VMIvStoreOp>(op))
    return ConsumerKind::MaskFreeFullRead;
  // Mask-free pure compute ops that read the full vreg.
  static const llvm::StringLiteral kMaskFree[] = {
      "pto.vmi.vcvt", "pto.vmi.vselr", "pto.vmi.vinterpret_cast",
      "pto.vmi.vshuffle", "pto.vmi.vbrc", "pto.vmi.vci"};
  for (auto n : kMaskFree)
    if (name == n)
      return ConsumerKind::MaskFreeFullRead;
  // Masked elementwise/reduce ops whose mask predicates the data lanes.
  static const llvm::StringLiteral kInferable[] = {
      // vec-scalar elementwise (VMI_VecScalarOp template)
      "pto.vmi.vadds", "pto.vmi.vmuls", "pto.vmi.vmaxs", "pto.vmi.vmins",
      "pto.vmi.vshls", "pto.vmi.vshrs",
      // direct elementwise / reduce
      "pto.vmi.vadd",  "pto.vmi.vsub",  "pto.vmi.vmul",  "pto.vmi.vdiv",
      "pto.vmi.vmin",  "pto.vmi.vmax",  "pto.vmi.vneg",  "pto.vmi.vabs",
      "pto.vmi.vsqrt", "pto.vmi.vexp",  "pto.vmi.vln",   "pto.vmi.vrelu",
      "pto.vmi.vshl",  "pto.vmi.vshr",  "pto.vmi.vcmp",  "pto.vmi.vcmps",
      "pto.vmi.vcadd", "pto.vmi.vcmax", "pto.vmi.vcmin", "pto.vmi.vexpdif",
      "pto.vmi.vaxpy", "pto.vmi.vlrelu", "pto.vmi.vprelu", "pto.vmi.vmull",
      "pto.vmi.vmula"};
  for (auto n : kInferable)
    if (name == n)
      return ConsumerKind::MaskInferable;
  return ConsumerKind::NotKnown;
}

// A vmi.vload has no mask operand. Infer the mask constraint from its
// consuming op(s), but ONLY for consumers whose semantics are known (closed
// whitelist): masked elementwise/reduce ops (mask predicates data lanes ->
// read set = mask prefix) and mask-free pure compute ops (read full vreg).
// Any OTHER consumer (vsel, where the mask routes output but both values are
// read on all lanes; select; compress_store; region-bearing; unknown ops)
// forces std::nullopt — the load's read set cannot be bounded and forwarding
// is disabled for it.
//
// Result:
//   std::nullopt             -> cannot infer (a non-whitelisted consumer, a
//                              region-bearing consumer, OR a mix of masked and
//                              mask-free whitelisted consumers): do not forward.
//   some(Value{}) [empty]    -> all consumers are mask-free whitelisted: any
//                              tracked store matches (forward is safe regardless
//                              of store mask).
//   some(Value{nonEmpty})    -> every consumer shares this one mask: only a
//                              tracked store with an equivalent mask matches.
static std::optional<Value>
inferVMILoadUserMask(pto::VMIvLoadOp load) {
  // Whether at least one consuming op has been seen (vs. a load with no users,
  // which is conservatively not forwardable).
  bool seenConsumer = false;
  // The inferred mask, if any consumer carries one. Empty Value means
  // "no mask constraint so far".
  Value inferred;
  bool hasMaskConstraint = false;
  bool hasMaskFreeConsumer = false;
  for (OpOperand &use : load->getResult(0).getUses()) {
    Operation *owner = use.getOwner();
    ConsumerKind kind = classifyLoadConsumer(owner);
    if (kind == ConsumerKind::NotKnown)
      return std::nullopt; // vsel/unknown/etc: cannot bound the read set.
    seenConsumer = true;
    if (kind == ConsumerKind::MaskFreeFullRead) {
      // reads the full vreg; contributes no mask constraint but is
      // incompatible with a masked consumer.
      hasMaskFreeConsumer = true;
      continue;
    }
    // MaskInferable: extract its (single) mask operand.
    Value opMask;
    for (Value operand : owner->getOperands()) {
      if (!isa<pto::VMIMaskType>(operand.getType()))
        continue;
      if (!opMask)
        opMask = operand;
      else if (!areEquivalentMaskValues(opMask, operand))
        return std::nullopt; // conflicting masks within one consumer
    }
    if (!hasMaskConstraint) {
      inferred = opMask;
      hasMaskConstraint = true;
    } else if (!areEquivalentMaskValues(inferred, opMask)) {
      return std::nullopt; // two consumers with different masks
    }
  }
  if (!seenConsumer)
    return std::nullopt;
  // A masked consumer bounds the read to [0,N); a mask-free consumer reads the
  // full vreg. Both at once means the load reads the FULL vreg (the union of
  // all lanes the masked consumer reads and the full-vreg read of the
  // mask-free consumer). Return an empty Value to signal "full read" so the
  // caller treats readLanes as the full prefix — this still allows forwarding
  // from a store that writes ALL lanes (zero-pmode stores, which normalize to
  // writeLanes=full), which is the common case. A partial (merge) store whose
  // writeLanes do not cover full will simply not match, so correctness holds.
  if (hasMaskConstraint && hasMaskFreeConsumer)
    return Value(); // full-vreg read: forwardable only from full-lane stores
  return inferred; // empty if all consumers mask-free, else the shared mask
}

// First-version shape guard: this pass models ONLY the continuous, single
// result (load) / single value (store) vmi.vload/vmi.vstore, with no stride,
// block_stride, group or dist_mode, and (for stores) no updated_base
// (post_update) result. Every other shape — dintlv dual load (2 results),
// unpack/brc load, grouped load/store, block-strided load/store, multi-value
// store, post_update store — is left untouched: loads are recorded as
// non-matchable (unknown read set) and flush the table; stores flush the
// table (we cannot soundly model which lanes they define).
static bool isContinuousSingleVLoad(pto::VMIvLoadOp op) {
  if (op.getStride() || op.getBlockStride())
    return false;
  if (op.getDistMode() || op.getGroup())
    return false;
  return op.getResults().size() == 1;
}
static bool isContinuousSingleVStore(pto::VMIvStoreOp op) {
  if (op.getStride() || op.getBlockStride())
    return false;
  if (op.getDistMode() || op.getGroup())
    return false;
  if (op.getValues().size() != 1)
    return false;
  // updated_base result marks a post_update block-stride store; unmodeled.
  if (op.getUpdatedBase())
    return false;
  return true;
}

// Whether `op` is safe to step over without invalidating tracked UB content.
// Conservative closed-set policy (no dialect-prefix wildcards):
//   1. region-bearing op, pto.vmi.vload/vstore, func.call -> NEVER transparent.
//   2. the explicit escape/sync name set (mte_*/set_flag/mem_bar/...) -> never
//      transparent (handled by the escape/invalidate branches in the loop).
//   3. an op that implements MemoryEffectOpInterface is transparent ONLY if it
//      reports no Read/Write effect (catches vgather/vscatter/masked_load/
//      group_store/... and forwards them to the invalidate-all path).
//   4. an op WITHOUT MemoryEffectOpInterface is transparent ONLY if it is
//      explicitly Pure (mlir::isPure, the C++ equivalent of the TableGen
//      `Pure` trait) — i.e. the dialect declared it side-effect-free. This
//      admits the VMI compute ops (vmuls/vcvt/...),
//      pointer_cast/castptr/create_mask/broadcast/iota/arith.constant/muli/...
//      and rejects any UNKNOWN op that forgot to declare effects: such an op
//      is conservatively treated as impure and flushes the table. We
//      deliberately do NOT use a dialect prefix like "arith."/"func." here,
//      because a future op added under such a prefix would be auto-admitted.
static bool isTransparentToTrackedStores(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (isa<pto::VMIvLoadOp, pto::VMIvStoreOp>(op))
    return false;
  if (isa<func::CallOp>(op))
    return false;
  StringRef name = op->getName().getStringRef();
  static const llvm::StringLiteral kImpure[] = {
      "pto.mte_gm_ub",  "pto.mte_ub_gm",   "pto.set_flag",
      "pto.wait_flag",  "pto.mem_bar",      "pto.pipe_barrier",
      "pto.vecscope",   "pto.strict_vecscope"};
  for (auto n : kImpure)
    if (name == n)
      return false;
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (iface.hasEffect<MemoryEffects::Read>() ||
        iface.hasEffect<MemoryEffects::Write>())
      return false;
    return true; // implements the interface, declared no Read/Write -> safe
  }
  // No MemoryEffectOpInterface: require the op to be explicitly Pure (the
  // TableGen `Pure` trait, exposed as mlir::isPure). This admits the VMI
  // compute ops (vmuls/vcvt/...), pointer_cast/castptr/create_mask/broadcast/
  // iota/arith.constant/muli/... and rejects any UNKNOWN op that forgot to
  // declare effects — such an op is conservatively treated as impure and
  // flushes the table.
  return isPure(op);
}

// A region-escaping op reads a UB and exports it out of the region (mte_ub_gm
// writes UB->GM, mte_gm_ub writes GM->UB). For elision correctness: an escape
// READ of a store's UB means the store is observable and must NOT be erased
// even after its value is forwarded to a load (the escape re-reads the UB).
// mte_gm_ub is an escape WRITE: it redefines the UB from GM, so any prior
// tracked content of that UB is stale.
static bool isEscapeReadOfUB(Operation *op, Value &ubRead) {
  if (auto mte = dyn_cast<pto::MteUbGmOp>(op)) {
    ubRead = mte.getSource();
    return true;
  }
  return false;
}
static bool isEscapeWriteToUB(Operation *op, Value &ubWritten) {
  if (auto mte = dyn_cast<pto::MteGmUbOp>(op)) {
    ubWritten = mte.getDestination();
    return true;
  }
  return false;
}

// A content-version table entry for one vload or vstore. Built in Pass 1 and
// consumed (mutated by marking) in Pass 2.
struct ContentEntry {
  Operation *op = nullptr;
  Value base;          // canonical UB (pointer_cast result, traced from dest/src)
  Value offset;
  LaneRange lanes;     // read set (load) / write set (store)
  bool isLoad = false;
  Value sourceValue;    // store.value or load.result (forward target value)
  Value storeMask;      // original store mask; null for loads
  StringAttr storePmode; // original store pmode; null for loads

  // Pass 1 marks:
  int forwardToIdx = -1; // >=0: this load forwards to entries[forwardToIdx].sourceValue
  bool eraseMark = false;     // this op should be erased in Pass 2 (dead load/store)
  bool escapeMark = false;    // a store whose UB is read by a region-escaping op: keep
  bool stale = false;         // content no longer usable as a forward target
  //    (only a WRITE invalidates content; a read never does)
  bool nonErasable = false;   // store may be observed by an unknown/affine read or
  //    escape: must NOT be erased by overwrite-DSE even after forwarding
};

// Two-pass elision over a straight-line range (a fused scf.for body, or the
// top-level ops of a fusion_region between two for's). Pass 1 builds a content
// table and marks forward targets / dead stores; Pass 2 applies replacements
// and erases in reverse order. A scf.for in the range (only at the top level)
// is not transparent (it has a region), so it flushes the table — correct, as
// a for body may read/write tracked UBs.
template <typename OpRange>
static bool elideOpRange(OpRange ops) {
  SmallVector<ContentEntry, 8> entries;
  bool changed = false;

  // ---- Pass 1: build + mark (forward scan, no IR mutation) ----
  // Match helpers operating on the live entry set (stale entries skipped).
  // Two accesses locate the same UB iff their bases must-alias (constant or
  // affine identity) and their offsets are equivalent.
  auto sameLoc = [&](const ContentEntry &e, Value base, Value offset) {
    return basesMustAlias(resolveBaseIdentity(e.base),
                          resolveBaseIdentity(base)) &&
           e.base.getType() == base.getType() &&
           areEquivalentValues(e.offset, offset);
  };

  for (Operation &op : ops) {
    if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
      if (!isContinuousSingleVLoad(load)) {
        // Non-continuous / multi-result / grouped / block-strided load: its
        // read set cannot be bounded as a prefix interval, and the load may
        // touch UBs we don't track. It is still a PURE READ, so it must not
        // invalidate tracked content (a later constant-base load may still
        // forward). But it may observe any tracked store -> mark those
        // non-erasable. It never acts as a forward target.
        for (auto &e : entries)
          if (!e.isLoad)
            e.nonErasable = true;
        entries.push_back({load, load.getSource(), load.getOffset(),
                           LaneRange::unknown(), true, load->getResult(0), {}, {},
                           -1, false, false, /*stale=*/true,
                           /*nonErasable=*/false});
        continue;
      }
      // Resolve the vload read lane set from its consumer mask.
      std::optional<Value> inferredMask = inferVMILoadUserMask(load);
      LaneRange readLanes;
      if (!inferredMask) {
        readLanes = LaneRange::unknown();
      } else if (!*inferredMask) {
        // all consumers mask-free: full vreg read
        unsigned vl = getVRegWidth(load->getResult(0).getType());
        readLanes = vl ? LaneRange::prefix(vl) : LaneRange::full();
      } else {
        // Consumers share one mask: the read set is bounded by its prefix
        // [0,N). An unresolvable mask yields unknown (skip).
        readLanes = resolveMaskLanes(*inferredMask);
      }
      Value base = load.getSource();
      Value offset = load.getOffset();
      BaseIdentity id = resolveBaseIdentity(base);

      if (!isTrackableIdentity(id) || readLanes.isUnknownSet()) {
        // The load base is a runtime pointer / untrackable / non-affine base,
        // OR its read lanes cannot be bounded. We cannot soundly match it for
        // forwarding. But a vload is a PURE READ: it does not change memory, so
        // it must NOT invalidate the content of any tracked store (a later
        // constant-base load may still forward to a preceding store). What it
        // MAY do is observe a preceding store's UB — so if this load may alias
        // a tracked store, that store becomes non-erasable (a later overwrite
        // must not delete it, since this load could still read it).
        for (auto &e : entries)
          if (!e.isLoad && basesMayAlias(id, resolveBaseIdentity(e.base)))
            e.nonErasable = true;
        // Record a non-matchable entry (it never acts as a forward target).
        entries.push_back({load, base, offset, LaneRange::unknown(), true,
                           load->getResult(0), {}, {}, -1, false, false,
                           /*stale=*/true,
                           /*nonErasable=*/false});
        continue;
      }

      // Look for a preceding entry that fully covers readLanes with no
      // intervening intersecting write. Scan from nearest backwards.
      int matchIdx = -1;
      for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
        ContentEntry &e = entries[i];
        if (e.stale || e.eraseMark)
          continue;
        if (!sameLoc(e, base, offset))
          continue;
        if (e.sourceValue.getType() != load->getResult(0).getType())
          continue;
        // Need e.lanes to fully cover readLanes.
        if (!e.lanes.contains(readLanes))
          continue;
        // For a store match: any intervening write to the same loc between e
        // and this load would have invalidated e (it would be stale/erased or
        // a newer entry). Because stale entries are skipped and a later write
        // to intersecting lanes marks prior entries stale, reaching here means
        // no intervening write touched readLanes -> safe to forward.
        matchIdx = i;
        break;
      }
      if (matchIdx >= 0) {
        entries.push_back({load, base, offset, readLanes, true,
                           load->getResult(0), {}, {}, matchIdx, true, false, false,
                           false});
        changed = true; // load will be forwarded + erased in Pass 2
      } else {
        // Trackable load that did NOT forward (no covering preceding entry).
        // It is retained and still observes the UB (and its preceding stores'
        // content) at its base. Even though base/lanes are trackable, the same
        // overwrite-DSE hazard as the untrackable branch applies: a preceding
        // store this load MAY alias must not be deleted by a later full
        // overwrite, or this retained load would read different content than
        // the original program. This is the tracked analog of the untrackable
        // branch above (where may-alias is trivially true because the base is
        // unknown).
        for (auto &e : entries)
          if (!e.isLoad && basesMayAlias(id, resolveBaseIdentity(e.base)))
            e.nonErasable = true;
        entries.push_back({load, base, offset, readLanes, true,
                           load->getResult(0), {}, {}, -1, false, false, false,
                           false});
      }
      continue;
    }

    if (auto store = dyn_cast<pto::VMIvStoreOp>(op)) {
      if (!isContinuousSingleVStore(store)) {
        // Unmodeled store shape (dintlv/group/block-stride/multi-value/
        // post_update): conservatively invalidate all tracked content and do
        // not register it as a forward target.
        for (auto &e : entries)
          e.stale = true;
        continue;
      }
      Value base = store.getDestination();
      Value offset = store.getOffset();
      BaseIdentity id = resolveBaseIdentity(base);
      if (!isTrackableIdentity(id)) {
        // The store base is a runtime pointer / block argument / untraceable /
        // non-affine base. It may alias any tracked UB at runtime; we cannot
        // prove it does not, so conservatively invalidate all tracked content
        // and do not register this store as a forward target.
        for (auto &e : entries)
          e.stale = true;
        continue;
      }
      Value mask = store.getMask().empty() ? Value() : store.getMask().front();
      LaneRange sourceLanes = resolveMaskLanes(mask);
      LaneRange writeLanes = sourceLanes;
      // pmode: "merge" => only writeLanes written; "zero"(default)/absent =>
      // whole region defined (inactive lanes store 0 -> treat as full cover).
      bool pmodeMerge = false;
      StringAttr pmode = store.getPmodeAttr();
      if (pmode)
        pmodeMerge = pmode.getValue().equals_insensitive("merge");
      if (!pmodeMerge)
        writeLanes = LaneRange::full(); // zero: entire UB defined

      // Redundant-store elision (strict). If a preceding, still-live store at
      // the same location writes the SAME effective lane set and the SAME SSA
      // value, this store writes nothing new to memory (the earlier store
      // already established that content), so it is redundant and dead. We
      // require the SAME SSA value (not a structural equivalence), equivalent
      // original masks, and the same pmode. Comparing only normalized lanes is
      // insufficient: zero-pmode stores with different masks both normalize to
      // full, but write different active-source/inactive-zero lane content. The
      // earlier store must be live (not stale/erased), guaranteeing no
      // intervening write touched these lanes.
      Value curValue = store.getValues().front();
      bool redundant = false;
      for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
        ContentEntry &e = entries[i];
        if (e.isLoad || e.stale || e.eraseMark)
          continue;
        if (!sameLoc(e, base, offset))
          continue;
        if (!areEquivalentMaskValues(e.storeMask, mask))
          continue;
        if (e.storePmode != pmode)
          continue;
        if (e.sourceValue != curValue)
          continue;
        redundant = true;
        break;
      }
      if (redundant) {
        // Record the redundant store as dead (eraseMark + stale) so Pass 2
        // erases it, but do NOT let it invalidate the earlier store: it wrote
        // the same content, so the earlier store stays the canonical forward
        // target / content source.
        entries.push_back({store, base, offset, sourceLanes, false, curValue,
                           mask, pmode, -1, /*eraseMark=*/true, false,
                           /*stale=*/true,
                           /*nonErasable=*/false});
        changed = true;
        continue;
      }

      // Mark preceding may-alias entries by how this write touches them:
      //  - may-alias but not must-alias -> the entry is stale. The write may
      //    redefine its content at runtime, but cannot prove the earlier store
      //    dead, so overwrite-DSE is not allowed.
      //  - fully covered -> a store is dead (eraseMark), unless it escapes or
      //    is non-erasable (may be observed by an unknown/affine read); the
      //    entry stops matching (stale).
      //  - partial overlap (merge) -> the entry no longer fully represents the
      //    current UB content, so it must not be a forward target anymore
      //    (stale), but it is neither dead nor erasable (other lanes may still
      //    be read / escape).
      for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
        ContentEntry &e = entries[i];
        BaseIdentity entryId = resolveBaseIdentity(e.base);
        if (!basesMayAlias(entryId, id))
          continue;
        if (!basesMustAlias(entryId, id)) {
          e.stale = true;
          continue;
        }
        // Without a byte-range alias model, a different offset or view on the
        // same storage root may partially overlap this write.
        if (!sameLoc(e, base, offset)) {
          e.stale = true;
          continue;
        }
        if (e.sourceValue.getType() != curValue.getType()) {
          e.stale = true;
          continue;
        }
        if (writeLanes.contains(e.lanes)) {
          if (!e.isLoad && !e.escapeMark && !e.nonErasable) {
            e.eraseMark = true;
            changed = true; // a dead store will be erased in Pass 2
          }
          e.stale = true;
        } else if (writeLanes.intersects(e.lanes)) {
          e.stale = true;
        }
      }
      // `writeLanes` describes memory invalidation.  The source SSA value is
      // forwardable only on active lanes: zero-pmode inactive lanes are
      // materialized as zero in memory and need not be zero in the source vreg.
      entries.push_back({store, base, offset, sourceLanes, false,
                         store.getValues().front(), mask, pmode, -1, false,
                         false, false, false});
      continue;
    }

    // Non-load/store ops.
    if (!isTransparentToTrackedStores(&op)) {
      // Region-escaping or aliasing op. mte_ub_gm reads a UB (escape: keep its
      // store); mte_gm_ub writes a UB (redefines: invalidate prior entries);
      // other impure ops conservatively invalidate everything.
      Value esc;
      if (isEscapeReadOfUB(&op, esc)) {
        // mte_ub_gm reads a UB out of the region: its store is observable and
        // must survive even after forwarding. The read does not redefine the
        // UB, so entries keep matching (content stays available).
        BaseIdentity escId = resolveBaseIdentity(esc);
        for (auto &e : entries)
          if (!e.isLoad && basesMayAlias(escId, resolveBaseIdentity(e.base)))
            e.escapeMark = true;
        continue;
      }
      if (isEscapeWriteToUB(&op, esc)) {
        // mte_gm_ub redefines the UB from GM: prior tracked content is invalid.
        // If the rewritten UB may alias a tracked base (const or affine), that
        // entry's content is stale.
        BaseIdentity escId = resolveBaseIdentity(esc);
        for (auto &e : entries)
          if (basesMayAlias(escId, resolveBaseIdentity(e.base)))
            e.stale = true;
        continue;
      }
      // Other impure (set_flag/mem_bar/scf.for body that may alias tracked
      // UBs): mark every existing entry stale. In a two-pass design entries
      // cannot be dropped mid-scan — stale preserves any already-recorded
      // forward marks for Pass 2 while preventing further matching against
      // these (possibly-aliased) entries. This is the two-pass analog of the
      // old single-pass `trackedStores.clear()`.
      for (auto &e : entries)
        e.stale = true;
    }
  }

  // ---- Pass 2: eliminate (reverse order) ----
  // Replace forwarded loads' uses first (reverse so a value consumed by a
  // later-forwarded op is replaced before that op is erased), then erase dead
  // loads/stores guarded by use_empty.
  for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
    ContentEntry &e = entries[i];
    if (e.forwardToIdx >= 0 && e.isLoad) {
      Value target = entries[e.forwardToIdx].sourceValue;
      e.op->getResult(0).replaceAllUsesWith(target);
    }
  }
  for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
    ContentEntry &e = entries[i];
    if (e.eraseMark && e.op->use_empty())
      e.op->erase();
    else if (e.forwardToIdx >= 0 && e.isLoad && e.op->use_empty())
      e.op->erase();
  }
  return changed;
}

// Run the two-pass elision over each fusion_region in three scopes:
//  1. the top-level ops of the region body (the prologue/between/epilogue
//     straight-line segments separated by scf.for's);
//  2. the straight-line body of each vecscope nested in the region; and
//  3. each scf.for body nested in the region (the fused leaf body).
//
// VecScope inference may wrap the whole fusion body in a region. The
// fusion-region scan must remain for pre-existing unscoped VMI, but it cannot
// see the direct vload/vstore operations inside that wrapper. Scanning each
// vecscope body explicitly preserves the fusion-local legality assumptions
// while allowing round trips between fused loops to be eliminated.
static bool elideInRegion(pto::FusionRegionOp region) {
  bool changed = false;
  Block &body = region.getBody().front();
  // Top-level: walk all ops except the region's pto.yield terminator.
  changed |= elideOpRange(body.without_terminator());

  // VecScope bodies are independent straight-line optimization ranges. Do
  // not treat the vecscope operation itself as transparent: enter its body
  // explicitly so the range scan can see VMI loads and stores.
  region.getBody().walk([&](pto::VecScopeOp vecscope) {
    if (vecscope->getParentOfType<pto::FusionRegionOp>() == region) {
      Block &scopeBody = vecscope.getBody().front();
      changed |= elideOpRange(
          llvm::make_range(scopeBody.begin(), scopeBody.end()));
    }
    return WalkResult::advance();
  });
  region.getBody().walk([&](pto::StrictVecScopeOp vecscope) {
    if (vecscope->getParentOfType<pto::FusionRegionOp>() == region) {
      Block &scopeBody = vecscope.getBody().front();
      changed |= elideOpRange(
          llvm::make_range(scopeBody.begin(), scopeBody.end()));
    }
    return WalkResult::advance();
  });

  // Each nested scf.for body.
  region.getBody().walk([&](scf::ForOp loop) {
    if (loop->getParentOfType<pto::FusionRegionOp>() == region &&
        isTileLibVmiPrincipalLoop(loop))
      changed |= elideOpRange(loop.getBody()->without_terminator());
    return WalkResult::advance();
  });
  return changed;
}

struct PTOVmiLoadStoreElisionPass
    : public mlir::pto::impl::PTOVmiLoadStoreElisionBase<
          PTOVmiLoadStoreElisionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    bool changed = false;
    func.walk([&](pto::FusionRegionOp region) {
      changed |= elideInRegion(region);
    });
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVmiLoadStoreElisionPass() {
  return std::make_unique<PTOVmiLoadStoreElisionPass>();
}
