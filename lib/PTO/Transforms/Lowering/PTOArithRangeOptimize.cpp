// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOArithRangeOptimize.cpp -----------------------------------------===//
//
// Integer-range-driven arith optimizations on PTO IR. Hardware SIMT ID
// queries implement InferIntRangeInterface with non-negative ranges, so the
// upstream arith passes can prove signed-vs-unsigned equivalence (e.g.
// floordivsi/remsi by a power of two on get_tid_x results) and rewrite them
// into unsigned forms the LLVM backend folds into shifts/masks.
//
// A follow-up narrowing step rewrites element-offset chains feeding
// pto.load/pto.store into i32 when a local recursive range
// evaluation proves the chain values fit in [0, 2^32). The default lowering
// widens these chains to index/i64 (loop induction variables are index-typed
// and arith.index_cast extends out of them), so each mul/add is lowered as a
// 64-bit operation although A5 scalar units compute in 32 bits. Two rewrite
// flavors fall out of one traversal:
//
//   * full chain: every value is provably in range, so the offset operand is
//     replaced by the i32 chain converted back with arith.index_castui
//     (zero-extend, exact for the proven range);
//   * subchain: the chain as a whole is unprovable (typically because a
//     block index with a runtime-count range feeds it), but an operand of an
//     interior add/mul is provable (e.g. `w * 2048` under
//     `+ block_idx`). The provable operand is swapped for its i32 chain
//     widened back with index_castui/extui, so the wide operation shrinks to
//     a wide add over a narrow product.
//
// All other users of the original chain are left untouched. The evaluation
// is local rather than a dataflow-framework analysis on purpose: the rewrite
// needs operand-level subchain decisions (narrow only the provable operand of
// an unprovable parent) that a global sparse range analysis does not drive on
// its own, and the chain grammar is closed — constants, scf.for induction
// variables with constant bounds, nullary ops implementing
// InferIntRangeInterface (the hardware ID queries), muli/addi, integer
// division/remainder by a positive constant, and index_cast — so a recursive
// evaluator is both sound and precise for it. Mirrored i32 operations are
// inserted immediately before their originals to preserve
// dominance for every possible consumer.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOARITHRANGEOPTIMIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

// Kill switch mirroring vpto-disable-align-chain-verification: the offset
// narrowing rewrites address arithmetic, so keep a way to turn just that
// step off for field triage and A/B measurements without dropping the
// range-driven signed-to-unsigned rewrites above it.
static llvm::cl::opt<bool> disableOffsetNarrowing(
    "pto-disable-offset-narrowing",
    llvm::cl::desc("Disable the range-driven i32 narrowing of pto.load/store "
                   "offset chains"),
    llvm::cl::init(false), llvm::cl::Hidden);

namespace {

constexpr unsigned kI32RangeBitWidth = 32;
constexpr unsigned kI64RangeBitWidth = 64;
constexpr unsigned kCollectedRangesInlineCapacity = 4;
constexpr unsigned kAnchorInlineCapacity = 32;

// Unsigned 64-bit interval used by the local evaluator; `valid` is false
// once a value escapes the grammar or its range cannot be represented.
struct URange {
  APInt lo = APInt(kI64RangeBitWidth, 0); // 64-bit
  APInt hi = APInt(kI64RangeBitWidth, 0); // 64-bit
  bool valid = false;

  URange() = default;
  URange(const APInt &lo, const APInt &hi, bool valid)
      : lo(lo), hi(hi), valid(valid) {}

  static URange invalid() { return URange{}; }
  static URange exact(const APInt &v) { return {v, v, true}; }
  static URange range(const APInt &lo, const APInt &hi) {
    return {lo, hi, true};
  }
};

static bool fitsU32(const URange &r) {
  return r.valid && r.lo.isNonNegative() && r.hi.isNonNegative() &&
         r.hi.getActiveBits() <= kI32RangeBitWidth;
}

// Result of proving one chain value: `range.valid` means the value is
// provably inside `range`; `i32Value` is additionally set when the value is
// already available in i32 (an i32-typed chain node); `binOps` counts the
// arithmetic operations of the mirrored subchain so cast-only chains can be
// skipped.
struct ChainInfo {
  URange range;
  Value i32Value;
  unsigned binOps = 0;

  bool provable() const { return range.valid; }
};

class OffsetChainNarrower {
public:
  OffsetChainNarrower() = default;

  bool changed() const { return anyChange; }

  // Attempts the full-chain rewrite of one offset operand. `offset` is the
  // index-typed operand of pto.load/pto.store.
  void narrowAnchorOperand(Operation *anchor, unsigned operandIdx) {
    Value offset = anchor->getOperand(operandIdx);
    ChainInfo top = visit(offset);
    if (!top.provable() || !fitsU32(top.range) || top.binOps == 0) {
      return;
    }
    OpBuilder b(anchor);
    Value i32Chain = asI32(offset, top, b, anchor);
    Value back = widen(i32Chain, offset.getType(), b, anchor);
    if (!back) {
      return;
    }
    anchor->setOperand(operandIdx, back);
    anyChange = true;
  }

private:
  static ChainInfo fail() { return ChainInfo{}; }

  static std::optional<APInt> constantOf(Operation *op) {
    if (!op) {
      return std::nullopt; // block arguments and absent defs
    }
    auto cst = dyn_cast<arith::ConstantOp>(op);
    if (!cst) {
      return std::nullopt;
    }
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!attr) {
      return std::nullopt;
    }
    return attr.getValue();
  }

  // Range of an scf.for induction variable with constant bounds. The last
  // iteration value is lb + (ceil((ub - lb) / step) - 1) * step; computed
  // in 128 bits so extreme bounds cannot wrap before the range check.
  static URange loopIVRange(BlockArgument arg) {
    auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!forOp || forOp.getInductionVar() != arg) {
      return URange::invalid();
    }
    auto lb = constantOf(forOp.getLowerBound().getDefiningOp());
    auto ub = constantOf(forOp.getUpperBound().getDefiningOp());
    auto step = constantOf(forOp.getStep().getDefiningOp());
    if (!lb || !ub || !step || !step->isStrictlyPositive() ||
        !lb->isNonNegative() || !ub->sgt(*lb)) {
      return URange::invalid();
    }
    APInt lb128 = lb->zext(128);
    APInt ub128 = ub->zext(128);
    APInt step128 = step->zext(128);
    APInt iters = (ub128 - lb128 + step128 - 1).udiv(step128);
    APInt last = lb128 + step128 * (iters - 1);
    if (last.getActiveBits() > kI32RangeBitWidth) {
      return URange::invalid();
    }
    return URange::range(lb->zext(kI64RangeBitWidth), last.trunc(kI64RangeBitWidth).zext(kI64RangeBitWidth));
  }

  // Range of a nullary op implementing InferIntRangeInterface (the hardware
  // ID queries). Only called on operand-less ops: operand-dependent impls
  // index argRanges and would read out of bounds on the empty argument.
  static URange interfaceRange(Operation *op) {
    auto iface = cast<InferIntRangeInterface>(op);
    SmallVector<std::pair<Value, ConstantIntRanges>, kCollectedRangesInlineCapacity> collected;
    iface.inferResultRanges(
        {}, [&](Value result, const ConstantIntRanges &ranges) {
          collected.emplace_back(result, ranges);
        });
    if (collected.size() != 1 || collected[0].first != op->getResult(0)) {
      return URange::invalid();
    }
    const ConstantIntRanges &r = collected[0].second;
    if (!r.umin().isNonNegative()) {
      return URange::invalid();
    }
    return URange::range(r.umin().zext(kI64RangeBitWidth), r.umax().zext(kI64RangeBitWidth));
  }

  // Produces the i32 value of a proven chain node. i32-typed nodes are used
  // directly; wider boundary values (index/i64) get a narrowing cast
  // inserted before `pos` (trunc/index_cast are exact for the proven
  // non-negative u32 range).
  Value asI32(Value v, const ChainInfo &ci, OpBuilder &b, Operation *pos) const {
    // Runtime guard instead of an assertion: callers only pass proven chains,
    // and if a future caller forgets, no narrowing cast may be emitted.
    if (!ci.provable()) {
      return Value();
    }
    if (ci.i32Value) {
      return ci.i32Value;
    }
    Type ty = v.getType();
    if (ty.isSignlessInteger(kI64RangeBitWidth)) {
      return b.create<arith::TruncIOp>(pos->getLoc(), b.getI32Type(), v);
    }
    if (isa<IndexType>(ty)) {
      return b.create<arith::IndexCastOp>(pos->getLoc(), b.getI32Type(), v);
    }
    return Value();
  }

  // Zero-extends an i32 chain value back to `ty` (index or i64). Exact for
  // proven non-negative u32 ranges.
  Value widen(Value i32Value, Type ty, OpBuilder &b, Operation *pos) const {
    if (ty.isSignlessInteger(kI32RangeBitWidth)) {
      return i32Value;
    }
    if (ty.isSignlessInteger(kI64RangeBitWidth)) {
      return b.create<arith::ExtUIOp>(pos->getLoc(), ty, i32Value);
    }
    if (isa<IndexType>(ty)) {
      return b.create<arith::IndexCastUIOp>(pos->getLoc(), ty, i32Value);
    }
    return Value();
  }

  // Subchain rewrite: `user` stays in its original width, but a provable
  // arithmetic operand is replaced by its i32 mirror widened back, shrinking
  // e.g. `w*2048 + block_idx` to `zext(w32*2048) + block_idx`.
  void swapInNarrowedOperand(Operation *user, unsigned idx, Value operand,
                             const ChainInfo &ci) {
    if (!ci.provable() || ci.binOps == 0) {
      return;
    }
    Operation *operandDef = operand.getDefiningOp();
    if (operandDef &&
        isa<arith::IndexCastUIOp, arith::ExtUIOp>(operandDef)) {
      return; // already narrowed by a previous anchor
    }
    Type operandTy = operand.getType();
    if (operandTy.isSignlessInteger(kI32RangeBitWidth)) {
      return; // nothing to shrink; the operand already is i32
    }
    OpBuilder b(user);
    Value i32Chain = asI32(operand, ci, b, user);
    if (!i32Chain) {
      return;
    }
    Value widened = widen(i32Chain, operandTy, b, user);
    if (!widened) {
      return;
    }
    user->setOperand(idx, widened);
    anyChange = true;
  }

  ChainInfo visit(Value v) {
    auto cacheHit = cache.find(v);
    if (cacheHit != cache.end()) {
      return cacheHit->second;
    }
    cache[v] = fail(); // cycle guard
    ChainInfo result = visitImpl(v);
    cache[v] = result;
    return result;
  }

  // Only index/i32/i64 participate: narrower integers wrap inside the
  // proven range window (mirroring an i16 mul in i32 would change
  // semantics), and nothing wider is worth narrowing.
  static bool supportedChainType(Type ty) {
    if (isa<IndexType>(ty)) {
      return true;
    }
    auto intTy = dyn_cast<IntegerType>(ty);
    return intTy && (intTy.getWidth() == kI32RangeBitWidth || intTy.getWidth() == kI64RangeBitWidth);
  }

  ChainInfo visitImpl(Value v) {
    if (!supportedChainType(v.getType())) {
      return fail();
    }
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      return visitLoopIV(blockArg);
    }
    Operation *def = v.getDefiningOp();
    if (!def) {
      return fail();
    }
    return visitDefinedValue(v, def);
  }

  ChainInfo visitLoopIV(BlockArgument blockArg) const {
    URange range = loopIVRange(blockArg);
    if (!fitsU32(range)) {
      return fail();
    }
    return ChainInfo{range, Value(), 0};
  }

  ChainInfo visitDefinedValue(Value v, Operation *def) {
    if (auto cst = constantOf(def)) {
      // Constant operands are re-materialized in i32 where needed, so no
      // eager i32Value here.
      if (!cst->isNonNegative() || cst->getActiveBits() > kI32RangeBitWidth) {
        return fail();
      }
      return ChainInfo{URange::exact(cst->zext(kI64RangeBitWidth)), Value(), 0};
    }
    if (isa<arith::MulIOp, arith::AddIOp>(def)) {
      return visitMulAdd(v, def);
    }
    // Integer division by a positive constant keeps non-negative ranges
    // non-negative and bounded ([lo/d, hi/d] is exact for every div flavor
    // over a non-negative numerator interval).
    if (isa<arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp>(def)) {
      return visitDivRem(v, def, /*isRem=*/false);
    }
    // rem x, d with d > 0 and x non-negative stays in [0, d-1].
    if (isa<arith::RemSIOp, arith::RemUIOp>(def)) {
      return visitDivRem(v, def, /*isRem=*/true);
    }
    // andi is deliberately not narrowed: the chains it feeds (the ping-pong
    // UB buffer addresses) sit on the MTE address path, where rewriting the
    // chain perturbs the helper ABI and measurably regresses the kernel.
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(def)) {
      return visitIndexCast(def);
    }
    return visitInterfaceQuery(v, def);
  }

  // True when a cast widens a sub-index integer into index (the direction in
  // which a signed cast performs sign extension on this target).
  static bool isWideningToIndex(Type from, Type to) {
    auto fromInt = dyn_cast<IntegerType>(from);
    return fromInt && fromInt.getWidth() < kI64RangeBitWidth && isa<IndexType>(to);
  }

  // index_cast / index_castui between index and an integer type preserve
  // the value within the proven range, so the input's chain info carries
  // over. A signed index_cast sign-extends on widening, so it is only
  // value-preserving when the sign bit is provably clear (hi <= INT32_MAX);
  // the unsigned index_castui is exact across the whole proven u32 window.
  ChainInfo visitIndexCast(Operation *def) {
    Value in = def->getOperand(0);
    if (!in.getType().isIntOrIndex()) {
      return fail();
    }
    ChainInfo inner = visit(in);
    if (!inner.provable() || !fitsU32(inner.range)) {
      return fail();
    }
    // A signed index_cast sign-extends when it widens (i32 -> index), so a
    // set sign bit would turn into a negative index while the range says
    // otherwise. Narrowing or same-width casts (index -> i32, i64 -> index)
    // keep the low bits and stay exact for the proven window. Only guard the
    // widening direction.
    if (isa<arith::IndexCastOp>(def) &&
        isWideningToIndex(def->getOperand(0).getType(),
                          def->getResult(0).getType()) &&
        inner.range.hi.sgt(APInt(kI64RangeBitWidth, INT32_MAX))) {
      return fail();
    }
    return inner;
  }

  // Nullary hardware ID queries and other operand-independent range
  // providers.
  ChainInfo visitInterfaceQuery(Value v, Operation *def) const {
    if (!isa<InferIntRangeInterface>(def) || def->getNumOperands() != 0) {
      return fail();
    }
    URange range = interfaceRange(def);
    if (!fitsU32(range)) {
      return fail();
    }
    return ChainInfo{range, v.getType().isSignlessInteger(kI32RangeBitWidth) ? v : Value(), 0};
  }

  ChainInfo visitMulAdd(Value v, Operation *def) {
    Value lhs = def->getOperand(0);
    Value rhs = def->getOperand(1);
    ChainInfo lhsCI = visit(lhs);
    ChainInfo rhsCI = visit(rhs);
    bool lhsOK = lhsCI.provable() && fitsU32(lhsCI.range);
    bool rhsOK = rhsCI.provable() && fitsU32(rhsCI.range);

    URange combined = URange::invalid();
    if (lhsOK && rhsOK) {
      // Non-negative bounded intervals: combine the extrema in 64 bits and
      // reject anything that could leave the representable window (the
      // sign-bit check also catches 64-bit product wraparound, which is
      // only possible near (2^32-1)^2).
      bool isMul = isa<arith::MulIOp>(def);
      APInt lo = isMul ? lhsCI.range.lo * rhsCI.range.lo
                       : lhsCI.range.lo + rhsCI.range.lo;
      APInt hi = isMul ? lhsCI.range.hi * rhsCI.range.hi
                       : lhsCI.range.hi + rhsCI.range.hi;
      if (!lo.isNegative() && !hi.isNegative() && hi.getActiveBits() <= kI32RangeBitWidth) {
        combined = URange::range(lo, hi);
      }
    }

    if (combined.valid) {
      unsigned binOps = std::max(lhsCI.binOps, rhsCI.binOps) + 1;
      if (v.getType().isSignlessInteger(kI32RangeBitWidth)) {
        return ChainInfo{combined, v, binOps};
      }
      // Mirror the operation in i32 right before its original so any later
      // consumer (this def, an ancestor, or the anchor) sees dominating
      // definitions.
      OpBuilder b(def);
      Value lhs32 = asI32(lhs, lhsCI, b, def);
      Value rhs32 = asI32(rhs, rhsCI, b, def);
      if (!lhs32 || !rhs32) {
        return fail();
      }
      Value mirror =
          isa<arith::MulIOp>(def)
              ? Value(b.create<arith::MulIOp>(def->getLoc(), lhs32, rhs32)
                          ->getResult(0))
              : b.create<arith::AddIOp>(def->getLoc(), lhs32, rhs32)
                    ->getResult(0);
      return ChainInfo{combined, mirror, binOps};
    }

    // Unprovable combination (or sibling): still shrink the provable side.
    swapInNarrowedOperand(def, 0, lhs, lhsCI);
    swapInNarrowedOperand(def, 1, rhs, rhsCI);
    return fail();
  }

  ChainInfo visitDivRem(Value v, Operation *def, bool isRem) {
    Value lhs = def->getOperand(0);
    ChainInfo lhsCI = visit(lhs);
    auto rhsCst = constantOf(def->getOperand(1).getDefiningOp());
    // The mirror rematerializes the divisor in i32, so it must fit u32:
    // wider divisors would truncate (2^32 -> 0, 2^32+1 -> 1) and silently
    // change the result or introduce a division by zero. Keep those chains
    // in their original i64 form.
    if (!rhsCst || !rhsCst->isStrictlyPositive() ||
        rhsCst->getActiveBits() > kI32RangeBitWidth) {
      return fail();
    }
    if (!lhsCI.provable() || !fitsU32(lhsCI.range)) {
      return fail();
    }
    // Signed div/rem flavors treat a set sign bit as a negative operand;
    // the unsigned interval formulas only hold when the whole proven window
    // keeps the sign bit clear.
    if (isa<arith::DivSIOp, arith::RemSIOp, arith::FloorDivSIOp>(def) &&
        (lhsCI.range.hi.sgt(APInt(kI64RangeBitWidth, INT32_MAX)) ||
         rhsCst->sgt(APInt(kI64RangeBitWidth, INT32_MAX)))) {
      return fail();
    }
    APInt rhs64 = rhsCst->zext(kI64RangeBitWidth);
    APInt remHi =
        lhsCI.range.hi.ult(rhs64 - 1) ? lhsCI.range.hi : rhs64 - 1;
    URange range = isRem
                       ? URange::range(APInt(kI64RangeBitWidth, 0), remHi)
                       : URange::range(lhsCI.range.lo.udiv(rhs64),
                                       lhsCI.range.hi.udiv(rhs64));
    if (!fitsU32(range)) {
      return fail();
    }
    if (v.getType().isSignlessInteger(kI32RangeBitWidth)) {
      return ChainInfo{range, v, lhsCI.binOps + 1};
    }
    auto ops = i32BinaryMirrorOperands(def, lhs, lhsCI, rhs64);
    if (!ops) {
      return fail();
    }
    OpBuilder b(def);
    // The proven range is non-negative, so the unsigned mirror is exact for
    // every original div/rem flavor — and keeps the shift-friendly udiv/urem
    // form the int-range phase above established (an sdiv mirror would undo
    // that and force the sign-correction sequence in the LLVM backend).
    Value mirror =
        isRem
            ? Value(b.create<arith::RemUIOp>(def->getLoc(), ops->first,
                                             ops->second)
                        ->getResult(0))
            : b.create<arith::DivUIOp>(def->getLoc(), ops->first, ops->second)
                  ->getResult(0);
    return ChainInfo{range, mirror, lhsCI.binOps + 1};
  }

  // Builds the i32 operand pair for a binary mirror of `def`: the proven lhs
  // narrowed via asI32, the constant rhs rematerialized in i32.
  std::optional<std::pair<Value, Value>>
  i32BinaryMirrorOperands(Operation *def, Value lhs, const ChainInfo &lhsCI,
                          const APInt &rhsCst) {
    OpBuilder b(def);
    Value lhs32 = asI32(lhs, lhsCI, b, def);
    if (!lhs32) {
      return std::nullopt;
    }
    Value rhs32 = b.create<arith::ConstantOp>(
        def->getLoc(), b.getI32Type(),
        b.getIntegerAttr(b.getI32Type(), rhsCst.zext(kI64RangeBitWidth).trunc(32)));
    return std::pair{lhs32, rhs32};
  }

  DenseMap<Value, ChainInfo> cache;
  bool anyChange = false;
};

// Rewrites the offset operand (index 1 of pto.load/pto.store)
// and any provable subchains under it. Returns true on change.
static bool narrowOffsetOperand(Operation *anchor, unsigned operandIdx) {
  if (!isa<IndexType>(anchor->getOperand(operandIdx).getType())) {
    return false;
  }
  OffsetChainNarrower narrower;
  narrower.narrowAnchorOperand(anchor, operandIdx);
  return narrower.changed();
}

struct PTOArithRangeOptimizePass
    : public pto::impl::PTOArithRangeOptimizeBase<PTOArithRangeOptimizePass> {
  using pto::impl::PTOArithRangeOptimizeBase<
      PTOArithRangeOptimizePass>::PTOArithRangeOptimizeBase;

  void runOnOperation() override {
    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
    pm.addPass(mlir::arith::createArithUnsignedWhenEquivalentPass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
      return;
    }
    if (disableOffsetNarrowing) {
      return;
    }
    if (failed(narrowOffsetChains())) {
      signalPassFailure();
    }
  }

  LogicalResult narrowOffsetChains() {
    // Collect anchors first: rewriting the offset detaches the walker from
    // the original chain, so gather every candidate before touching any.
    // Only SIMT load/store offsets are narrowed: pto.addptr chains feed the
    // MTE copy address path in the orchestrator, where the rewrite changes
    // the helper ABI and measurably regresses the kernel. pto.ldg/pto.stg are
    // excluded for the same reason — they address GM directly and carry cache
    // controls, so they belong to that copy path rather than to the UB-side
    // scalar accesses this rewrite targets. Both narrowed ops carry the
    // element offset at operand index 1.
    SmallVector<std::pair<Operation *, unsigned>, kAnchorInlineCapacity> anchors;
    getOperation()->walk([&](Operation *op) {
      if (isa<pto::PTOStoreOp, pto::PTOLoadOp>(op)) {
        anchors.emplace_back(op, 1);
      }
    });
    if (anchors.empty()) {
      return success();
    }

    bool changed = false;
    for (auto [op, idx] : anchors) {
      if (op && narrowOffsetOperand(op, idx)) {
        changed = true;
      }
    }
    if (changed) {
      // Chains that fail the range proof partway leave materialized i32
      // mirrors behind without users. The canonicalizer's greedy driver
      // erases them by the generic isOpTriviallyDead rule, so the rewrite
      // never has to enumerate the ops it may have created.
      OpPassManager cleanup(ModuleOp::getOperationName());
      cleanup.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      cleanup.addNestedPass<func::FuncOp>(createCSEPass());
      if (failed(runPipeline(cleanup, getOperation()))) {
        return failure();
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOArithRangeOptimizePass() {
  return std::make_unique<PTOArithRangeOptimizePass>();
}
