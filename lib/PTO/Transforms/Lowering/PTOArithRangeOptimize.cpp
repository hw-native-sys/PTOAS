// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://console.huawei.com/cann/license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- PTOArithRangeOptimize.cpp -----------------------------------------===//
//
// Integer-range-driven arith optimizations on PTO IR. Hardware SIMT ID
// queries implement InferIntRangeInterface with non-negative ranges, so the
// upstream arith passes can prove signed-vs-unsigned equivalence (e.g.
// floordivsi/remsi by a power of two on get_tid_x results) and rewrite them
// into unsigned forms the LLVM backend folds into shifts/masks.
//
// A follow-up narrowing step rewrites element-offset chains feeding
// pto.load/pto.store/pto.addptr into i32 when a local recursive range
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
// is local rather than a dataflow-framework analysis on purpose: values
// inside pto.section.simt regions are unreachable to DeadCodeAnalysis (the
// op has no RegionBranchOpInterface and its body has no terminator), which
// would leave exactly the chains this rewrite targets without range facts.
// The chain grammar is closed — constants, scf.for induction variables with
// constant bounds, nullary ops implementing InferIntRangeInterface (the
// hardware ID queries), muli/addi, integer division/remainder by a positive
// constant, and index_cast — so a
// recursive evaluator is both sound and precise for it. Mirrored i32
// operations are inserted immediately before their originals to preserve
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

namespace {

// Unsigned 64-bit interval used by the local evaluator; `valid` is false
// once a value escapes the grammar or its range cannot be represented.
struct URange {
  APInt lo = APInt(64, 0); // 64-bit
  APInt hi = APInt(64, 0); // 64-bit
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
         r.hi.getActiveBits() <= 32;
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
  // index-typed operand of pto.load/pto.store/pto.addptr.
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
    if (last.getActiveBits() > 32) {
      return URange::invalid();
    }
    return URange::range(lb->zext(64), last.trunc(64).zext(64));
  }

  // Range of a nullary op implementing InferIntRangeInterface (the hardware
  // ID queries). Only called on operand-less ops: operand-dependent impls
  // index argRanges and would read out of bounds on the empty argument.
  static URange interfaceRange(Operation *op) {
    auto iface = cast<InferIntRangeInterface>(op);
    SmallVector<std::pair<Value, ConstantIntRanges>, 4> collected;
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
    return URange::range(r.umin().zext(64), r.umax().zext(64));
  }

  // Produces the i32 value of a proven chain node. i32-typed nodes are used
  // directly; wider boundary values (index/i64) get a narrowing cast
  // inserted before `pos` (trunc/index_cast are exact for the proven
  // non-negative u32 range).
  Value asI32(Value v, const ChainInfo &ci, OpBuilder &b, Operation *pos) {
    // Runtime guard instead of an assertion: callers only pass proven chains,
    // and if a future caller forgets, no narrowing cast may be emitted.
    if (!ci.provable()) {
      return Value();
    }
    if (ci.i32Value) {
      return ci.i32Value;
    }
    Type ty = v.getType();
    if (ty.isSignlessInteger(64)) {
      return b.create<arith::TruncIOp>(pos->getLoc(), b.getI32Type(), v);
    }
    if (isa<IndexType>(ty)) {
      return b.create<arith::IndexCastOp>(pos->getLoc(), b.getI32Type(), v);
    }
    return Value();
  }

  // Zero-extends an i32 chain value back to `ty` (index or i64). Exact for
  // proven non-negative u32 ranges.
  Value widen(Value i32Value, Type ty, OpBuilder &b, Operation *pos) {
    if (ty.isSignlessInteger(32)) {
      return i32Value;
    }
    if (ty.isSignlessInteger(64)) {
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
    if (operandTy.isSignlessInteger(32)) {
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
    return intTy && (intTy.getWidth() == 32 || intTy.getWidth() == 64);
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

  ChainInfo visitLoopIV(BlockArgument blockArg) {
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
      if (!cst->isNonNegative() || cst->getActiveBits() > 32) {
        return fail();
      }
      return ChainInfo{URange::exact(cst->zext(64)), Value(), 0};
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
    // remsi x, d with d > 0 and x non-negative stays in [0, d-1].
    if (isa<arith::RemSIOp>(def)) {
      return visitDivRem(v, def, /*isRem=*/true);
    }
    // andi is deliberately not narrowed: the chains it feeds (e.g. the
    // ping-pong UB buffer address `(w & 1) * 4096`) sit on the MTE address
    // path, where the extra trunc/zext pair costs more than the i64 mul.
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(def)) {
      return visitIndexCast(def);
    }
    return visitInterfaceQuery(v, def);
  }

  // index_cast / index_castui between index and an integer type preserve
  // the value within the proven range, so the input's chain info carries
  // over.
  ChainInfo visitIndexCast(Operation *def) {
    Value in = def->getOperand(0);
    if (!in.getType().isIntOrIndex()) {
      return fail();
    }
    ChainInfo inner = visit(in);
    if (!inner.provable() || !fitsU32(inner.range)) {
      return fail();
    }
    return inner;
  }

  // Nullary hardware ID queries and other operand-independent range
  // providers.
  ChainInfo visitInterfaceQuery(Value v, Operation *def) {
    if (!isa<InferIntRangeInterface>(def) || def->getNumOperands() != 0) {
      return fail();
    }
    URange range = interfaceRange(def);
    if (!fitsU32(range)) {
      return fail();
    }
    return ChainInfo{range, v.getType().isSignlessInteger(32) ? v : Value(), 0};
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
      APInt lo = lhsCI.range.lo * rhsCI.range.lo;
      APInt hi = isa<arith::MulIOp>(def) ? lhsCI.range.hi * rhsCI.range.hi
                                          : lhsCI.range.hi + rhsCI.range.hi;
      if (!lo.isNegative() && !hi.isNegative() && hi.getActiveBits() <= 32) {
        combined = URange::range(lo, hi);
      }
    }

    if (combined.valid) {
      unsigned binOps = std::max(lhsCI.binOps, rhsCI.binOps) + 1;
      if (v.getType().isSignlessInteger(32)) {
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
    if (!rhsCst || !rhsCst->isStrictlyPositive()) {
      return fail();
    }
    // Signed division/remainder semantics: the whole interval must stay in
    // the positive i32 window for the range claim and the i32 mirror to be
    // exact.
    APInt i32Max = APInt(64, INT32_MAX);
    if (!lhsCI.provable() || !fitsU32(lhsCI.range) ||
        lhsCI.range.hi.sgt(i32Max) || rhsCst->zext(64).sgt(i32Max)) {
      return fail();
    }
    APInt rhs64 = rhsCst->zext(64);
    APInt remHi =
        lhsCI.range.hi.ult(rhs64 - 1) ? lhsCI.range.hi : rhs64 - 1;
    URange range = isRem
                       ? URange::range(APInt(64, 0), remHi)
                       : URange::range(lhsCI.range.lo.sdiv(rhs64),
                                       lhsCI.range.hi.sdiv(rhs64));
    if (!fitsU32(range)) {
      return fail();
    }
    if (v.getType().isSignlessInteger(32)) {
      return ChainInfo{range, v, lhsCI.binOps + 1};
    }
    auto ops = i32BinaryMirrorOperands(def, lhs, lhsCI, rhs64);
    if (!ops) {
      return fail();
    }
    OpBuilder b(def);
    Value mirror =
        isRem
            ? Value(b.create<arith::RemSIOp>(def->getLoc(), ops->first,
                                             ops->second)
                        ->getResult(0))
            : b.create<arith::DivSIOp>(def->getLoc(), ops->first, ops->second)
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
        b.getIntegerAttr(b.getI32Type(), rhsCst.zext(64).trunc(32)));
    return std::pair{lhs32, rhs32};
  }

  DenseMap<Value, ChainInfo> cache;
  bool anyChange = false;
};

// Rewrites the offset operand (index 1 of pto.load/pto.store/pto.addptr)
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
    if (failed(narrowOffsetChains())) {
      signalPassFailure();
    }
  }

  LogicalResult narrowOffsetChains() {
    // Collect anchors first: rewriting the offset detaches the walker from
    // the original chain, so gather every candidate before touching any.
    // All three ops carry the element offset at operand index 1.
    SmallVector<std::pair<Operation *, unsigned>, 32> anchors;
    getOperation()->walk([&](Operation *op) {
      if (isa<pto::PTOStoreOp, pto::PTOLoadOp, pto::AddPtrOp>(op)) {
        anchors.emplace_back(op, 1);
      }
    });
    if (anchors.empty()) {
      return success();
    }

    bool changed = false;
    for (auto [op, idx] : anchors) {
      if (op) {
        changed |= narrowOffsetOperand(op, idx);
      }
    }
    if (changed) {
      eraseDeadArith();
      OpPassManager cleanup(ModuleOp::getOperationName());
      cleanup.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      cleanup.addNestedPass<func::FuncOp>(createCSEPass());
      if (failed(runPipeline(cleanup, getOperation()))) {
        return failure();
      }
    }
    return success();
  }

  // Chains that fail range proof partway may leave materialized i32 ops
  // without users; drop them before the cleanup pipeline.
  void eraseDeadArith() {
    SmallVector<Operation *, 64> dead;
    getOperation()->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->use_empty() &&
          isa<arith::ConstantOp, arith::MulIOp, arith::AddIOp, arith::DivSIOp,
              arith::DivUIOp, arith::FloorDivSIOp, arith::RemSIOp,
              arith::IndexCastOp, arith::IndexCastUIOp,
              arith::ExtUIOp, arith::TruncIOp>(op)) {
        dead.push_back(op);
      }
    });
    for (Operation *op : llvm::reverse(dead)) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOArithRangeOptimizePass() {
  return std::make_unique<PTOArithRangeOptimizePass>();
}
