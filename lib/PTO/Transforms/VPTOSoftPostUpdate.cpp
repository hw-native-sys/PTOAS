// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"


namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSOFTPOSTUPDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {
static constexpr int64_t kSignedI8Min = -128;
static constexpr int64_t kSignedI8Max = 127;

// What one unit of an op's strideOperand means, in address terms.  This is a
// property of the op's lowering, not of the pass: `Element` ops lower their
// logical index offset to bytes at the target ABI boundary, `Block` ops pass a
// packed control word straight to the intrinsic, `Alignment` ops use an
// op-specific hardware alignment table, and `Byte` ops pass a raw byte offset.
// Byte widths are resolved by the operation's VPTO address semantics.
using StrideUnit = pto::VPTOAddressUnit;

static std::optional<pto::VPTOPostUpdateSemantics>
getPostUpdateSemantics(Operation *op) {
  auto interface = dyn_cast<pto::VPTOAddressSemanticsOpInterface>(op);
  if (!interface) {
    return std::nullopt;
  }
  return interface.getVPTOAddressSemantics().postUpdate;
}

// Check if op is directly inside the scf.for body (not nested in scf.if etc).
static bool isDirectlyInForBody(Operation *op, scf::ForOp forOp) {
  return op->getParentOp() == forOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Accumulator Analysis: Linear Decomposition
//===----------------------------------------------------------------------===//

// Stride analysis is purely symbolic: it never creates IR.  A candidate is
// analyzed, checked for viability, and only then materialized at a single
// insertion point.  Two properties follow from that split:
//   - Every Value the analysis inspects is a pre-existing loop-body value, so
//     "defined before insertPt" transitively implies its operands are too.
//     Availability can be decided by looking at the expression's leaves alone.
//   - The recursion is side-effect free, so it can be memoized; decomposition
//     stays linear in the size of the def-chain DAG instead of exponential.
struct StrideExpr;
using StrideExprRef = std::shared_ptr<const StrideExpr>;

struct StrideExpr {
  enum class Kind { Const, Leaf, Add, Sub, Mul, Cast };
  Kind kind;
  int64_t constant = 0;        // Kind::Const
  Value leaf;                  // Kind::Leaf
  Operation *castOp = nullptr; // Kind::Cast — template op to clone
  StrideExprRef lhs, rhs;
};

static StrideExprRef makeConst(int64_t c) {
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Const;
  e->constant = c;
  return e;
}

static StrideExprRef makeLeaf(Value v) {
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Leaf;
  e->leaf = v;
  return e;
}

// Compile-time value of `e`, if it has one.
static std::optional<int64_t> foldConst(const StrideExprRef &e) {
  if (!e) {
    return std::nullopt;
  }
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return e->constant;
  case StrideExpr::Kind::Leaf:
    return getConstantIntValue(e->leaf);
  case StrideExpr::Kind::Cast:
    // index_cast/index_castui preserve the numeric value; only the type
    // changes, and the type is chosen at materialization time.
    return foldConst(e->lhs);
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    auto a = foldConst(e->lhs);
    auto b = foldConst(e->rhs);
    if (!a || !b) {
      return std::nullopt;
    }
    if (e->kind == StrideExpr::Kind::Add) {
      return *a + *b;
    }
    if (e->kind == StrideExpr::Kind::Sub) {
      return *a - *b;
    }
    return *a * *b;
  }
  }
  return std::nullopt;
}

static StrideExprRef makeBinary(StrideExpr::Kind kind, StrideExprRef a,
                                StrideExprRef b) {
  auto e = std::make_shared<StrideExpr>();
  e->kind = kind;
  e->lhs = std::move(a);
  e->rhs = std::move(b);
  return e;
}

static StrideExprRef makeAdd(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb) {
    return makeConst(*ca + *cb);
  }
  if (ca && *ca == 0) {
    return b;
  }
  if (cb && *cb == 0) {
    return a;
  }
  return makeBinary(StrideExpr::Kind::Add, a, b);
}

static StrideExprRef makeSub(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb) {
    return makeConst(*ca - *cb);
  }
  if (cb && *cb == 0) {
    return a;
  }
  return makeBinary(StrideExpr::Kind::Sub, a, b);
}

static StrideExprRef makeMul(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb) {
    return makeConst(*ca * *cb);
  }
  if ((ca && *ca == 0) || (cb && *cb == 0)) {
    return makeConst(0);
  }
  if (ca && *ca == 1) {
    return b;
  }
  if (cb && *cb == 1) {
    return a;
  }
  return makeBinary(StrideExpr::Kind::Mul, a, b);
}

static StrideExprRef makeCast(Operation *castOp, StrideExprRef a) {
  if (auto c = foldConst(a)) {
    return makeConst(*c);
  }
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Cast;
  e->castOp = castOp;
  e->lhs = std::move(a);
  return e;
}

// Convert the public typed analysis expression into the consumer's
// materialization tree. This is a representation-only handoff: all recurrence,
// range, cast, and unit proofs have already been completed by the analyses.
static StrideExprRef importTypedExpr(const pto::PTOTypedExprRef &expression) {
  if (!expression) {
    return nullptr;
  }
  if (expression->sourceValue) {
    auto constant = pto::foldPTOConstant(expression);
    return constant ? makeConst(*constant)
                    : makeLeaf(expression->sourceValue);
  }
  switch (expression->kind) {
  case pto::PTOTypedExpr::Kind::Constant: {
    auto constant = pto::foldPTOConstant(expression);
    return constant ? makeConst(*constant) : nullptr;
  }
  case pto::PTOTypedExpr::Kind::Opaque:
    return makeLeaf(expression->opaque);
  case pto::PTOTypedExpr::Kind::Add:
    return makeAdd(importTypedExpr(expression->lhs),
                   importTypedExpr(expression->rhs));
  case pto::PTOTypedExpr::Kind::Sub:
    return makeSub(importTypedExpr(expression->lhs),
                   importTypedExpr(expression->rhs));
  case pto::PTOTypedExpr::Kind::Mul:
    return makeMul(importTypedExpr(expression->lhs),
                   importTypedExpr(expression->rhs));
  case pto::PTOTypedExpr::Kind::Cast:
    if (!expression->sourceOperation) {
      return nullptr;
    }
    return makeCast(expression->sourceOperation,
                    importTypedExpr(expression->lhs));
  }
  return nullptr;
}

static void collectLeaves(const StrideExprRef &e, SmallVectorImpl<Value> &out) {
  if (!e) {
    return;
  }
  if (e->kind == StrideExpr::Kind::Leaf) {
    out.push_back(e->leaf);
    return;
  }
  collectLeaves(e->lhs, out);
  collectLeaves(e->rhs, out);
}

// Determine the concrete type `e` will materialize to.  Returns false when two
// subexpressions demand different types (an expression we must not build).
// `out` is left null when `e` is fully constant and can adapt to any type.
static bool exprType(const StrideExprRef &e, Type &out) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return true; // adapts to context
  case StrideExpr::Kind::Leaf:
    out = e->leaf.getType();
    return true;
  case StrideExpr::Kind::Cast:
    out = e->castOp->getResult(0).getType();
    return true;
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    Type ta, tb;
    if (!exprType(e->lhs, ta) || !exprType(e->rhs, tb)) {
      return false;
    }
    if (ta && tb && ta != tb) {
      return false;
    }
    out = ta ? ta : tb;
    return true;
  }
  }
  return false;
}

// Rewrite: create new ForOp with additional iter_arg
//===----------------------------------------------------------------------===//

// Compute the value of `v` at the first iteration (IV = lower bound) by
// cloning the def-chain with IV replaced by the lower bound.  Returns nullptr
// if `v` cannot be materialized outside the loop.
static Value materializeAtLoopEntry(Value v, scf::ForOp forOp,
                                    OpBuilder &builder) {
  // IV → lower bound
  if (v == forOp.getInductionVar()) {
    return forOp.getLowerBound();
  }

  // Already defined outside the loop — use directly.
  if (forOp.isDefinedOutsideOfLoop(v)) {
    return v;
  }

  // iter_arg → its init value
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() && blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      return forOp.getInitArgs()[idx];
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp)) {
    return nullptr;
  }

  // Cloning duplicates the op, so it must be safe to execute an extra time and
  // its result must not depend on anything but its operands.
  if (!isPure(defOp)) {
    return nullptr;
  }

  // Clone the defining op with operands materialized at loop entry.
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    Value materialized = materializeAtLoopEntry(operand, forOp, builder);
    if (!materialized) {
      return nullptr;
    }
    newOperands.push_back(materialized);
  }
  builder.setInsertionPoint(forOp);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands)) {
    cloned->setOperand(i, operand);
  }
  // Preserve which result was asked for; `v` need not be result 0.
  return cloned->getResult(cast<OpResult>(v).getResultNumber());
}

// Whether strideOperand can be restated in pto.addptr element units without
// creating IR. Element and block units are always exact for supported element
// types; finer byte units require a suitably aligned compile-time constant.
static bool canScaleInitialOffset(Value strideOperand, int64_t elemBytes,
                                  int64_t unitBytes) {
  if (!strideOperand || unitBytes == elemBytes || unitBytes % elemBytes == 0) {
    return true;
  }
  if (elemBytes % unitBytes != 0) {
    return false;
  }
  auto constant = getConstantIntValue(strideOperand);
  return constant && *constant % (elemBytes / unitBytes) == 0;
}

// Loop-varying byte offsets may still have an exactly representable entry
// value.  For example, a byte-denominated IV that starts at zero can initialize
// an element pointer even though arbitrary values of that IV are not divisible
// by the element size.  Query the public evolution model here so the legality
// plan remains read-only and the eventual materialization cannot fail after
// rewriting has started.
static bool canScaleInitialOffsetAtLoopEntry(
    Value offset, int64_t elemBytes, int64_t unitBytes, scf::ForOp forOp,
    pto::PTOValueEvolutionAnalysis &valueEvolution) {
  if (canScaleInitialOffset(offset, elemBytes, unitBytes)) {
    return true;
  }
  if (!offset || elemBytes % unitBytes != 0) {
    return false;
  }
  auto evolution = valueEvolution.getEvolution(offset, forOp);
  if (!evolution) {
    return false;
  }
  auto initial = pto::foldPTOConstant(evolution.value->initial);
  return initial && *initial % (elemBytes / unitBytes) == 0;
}

// pto.addptr always consumes an index offset. Block offsets retain their
// existing unsigned interpretation; every other supported address unit is
// signed, including sprsti's signed 8-bit word offset.
static Value normalizeAddPtrOffsetToIndex(Value offset, StrideUnit strideUnit,
                                          Location loc, OpBuilder &builder) {
  if (offset.getType().isIndex()) {
    return offset;
  }
  if (strideUnit == StrideUnit::Block) {
    return builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(),
                                                offset);
  }
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            offset);
}

// Create the address reached by one memory op before post-update rewriting.
// The builder must already point at the desired insertion location.
static Value createInitialPtr(Value base, Value strideOperand,
                              StrideUnit strideUnit, int64_t elemBytes,
                              int64_t unitBytes, Location loc,
                              OpBuilder &builder) {
  if (!strideOperand) {
    return base;
  }
  auto constSo = getConstantIntValue(strideOperand);
  if (constSo && *constSo == 0) {
    return base;
  }
  if (!canScaleInitialOffset(strideOperand, elemBytes, unitBytes)) {
    return nullptr;
  }

  Value scaledOffset = strideOperand;
  if (unitBytes != elemBytes) {
    if (unitBytes % elemBytes == 0) {
      Value soIndex = strideOperand;
      if (strideOperand.getType() != builder.getIndexType()) {
        soIndex = builder.create<arith::IndexCastUIOp>(
            loc, builder.getIndexType(), strideOperand);
      }
      Value factor =
          builder.create<arith::ConstantIndexOp>(loc, unitBytes / elemBytes);
      scaledOffset = builder.create<arith::MulIOp>(loc, soIndex, factor);
    } else {
      int64_t divisor = elemBytes / unitBytes;
      scaledOffset =
          builder.create<arith::ConstantIndexOp>(loc, *constSo / divisor);
    }
  }
  scaledOffset =
      normalizeAddPtrOffsetToIndex(scaledOffset, strideUnit, loc, builder);
  return builder.create<pto::AddPtrOp>(loc, base, scaledOffset);
}

// Compute the initial pointer for a loop candidate, i.e. the address reached on
// the first iteration. Values defined in the loop are first materialized at the
// loop entry, then the shared unit conversion above is applied.
static Value computeInitialPtr(Value base, Value strideOperand,
                               StrideUnit strideUnit, int64_t elemBytes,
                               int64_t unitBytes, scf::ForOp forOp,
                               OpBuilder &builder) {
  Value baseAtEntry = materializeAtLoopEntry(base, forOp, builder);
  if (!baseAtEntry) {
    return nullptr;
  }

  if (!strideOperand) {
    return baseAtEntry;
  }

  Value soAtEntry = materializeAtLoopEntry(strideOperand, forOp, builder);
  if (!soAtEntry) {
    return nullptr;
  }

  builder.setInsertionPoint(forOp);
  return createInitialPtr(baseAtEntry, soAtEntry, strideUnit, elemBytes,
                          unitBytes, forOp.getLoc(), builder);
}
// def-chain if needed?  Pure: inspects only, never mutates the IR.
static bool canHoistBefore(Value v, Operation *insertPt, scf::ForOp forOp,
                           DenseMap<Value, bool> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v) || isa<BlockArgument>(v)) {
    return true;
  }
  Operation *defOp = v.getDefiningOp();
  if (!defOp) {
    return true;
  }
  if (defOp->getBlock() != insertPt->getBlock()) {
    return false;
  }
  // Already earlier in the block, so usable as-is: SSA guarantees its operands
  // are defined even earlier.  This reasoning is only sound because analysis
  // creates no IR — every value examined here predates the transform.
  if (defOp->isBeforeInBlock(insertPt)) {
    return true;
  }
  auto it = memo.find(v);
  if (it != memo.end()) {
    return it->second;
  }
  memo[v] = false;
  if (!isPure(defOp)) {
    return false;
  }
  for (Value operand : defOp->getOperands()) {
    if (!canHoistBefore(operand, insertPt, forOp, memo)) {
      return false;
    }
  }
  memo[v] = true;
  return true;
}

// Clone `v`'s def-chain before `insertPt` as needed.  Only valid after
// canHoistBefore has approved `v`.
static Value hoistBefore(Value v, Operation *insertPt, scf::ForOp forOp,
                         OpBuilder &builder, DenseMap<Value, Value> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v) || isa<BlockArgument>(v)) {
    return v;
  }
  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getBlock() != insertPt->getBlock() ||
      defOp->isBeforeInBlock(insertPt)) {
    return v;
  }
  auto it = memo.find(v);
  if (it != memo.end()) {
    return it->second;
  }

  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    newOperands.push_back(hoistBefore(operand, insertPt, forOp, builder, memo));
  }

  builder.setInsertionPoint(insertPt);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands)) {
    cloned->setOperand(i, operand);
  }
  // Preserve which result was asked for; `v` need not be result 0.
  Value res = cloned->getResult(cast<OpResult>(v).getResultNumber());
  memo[v] = res;
  return res;
}

// Rewrite `e` so that all of its leaves are available at `insertPt`.
static StrideExprRef makeAvailableAt(const StrideExprRef &e,
                                     Operation *insertPt, scf::ForOp forOp,
                                     OpBuilder &builder,
                                     DenseMap<Value, Value> &memo) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return e;
  case StrideExpr::Kind::Leaf: {
    Value hv = hoistBefore(e->leaf, insertPt, forOp, builder, memo);
    return hv == e->leaf ? e : makeLeaf(hv);
  }
  case StrideExpr::Kind::Cast:
    return makeCast(e->castOp,
                    makeAvailableAt(e->lhs, insertPt, forOp, builder, memo));
  case StrideExpr::Kind::Add:
    return makeAdd(makeAvailableAt(e->lhs, insertPt, forOp, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, forOp, builder, memo));
  case StrideExpr::Kind::Sub:
    return makeSub(makeAvailableAt(e->lhs, insertPt, forOp, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, forOp, builder, memo));
  case StrideExpr::Kind::Mul:
    return makeMul(makeAvailableAt(e->lhs, insertPt, forOp, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, forOp, builder, memo));
  }
  llvm_unreachable("unhandled StrideExpr kind");
}

// Constants are loop-invariant, so they are always emitted before the loop and
// shared across every candidate in it.  Sharing matters beyond tidiness: the
// rewrite groups ops by base/offset/stride Value identity and effective byte
// unit, so two compatible candidates with the same numeric stride must end up
// with the *same* Value to share an iter_arg.
using ConstCache = DenseMap<std::pair<int64_t, Type>, Value>;

static Value materializeConst(int64_t c, Type ty, Location loc,
                              scf::ForOp forOp, ConstCache &cache,
                              OpBuilder &builder) {
  if (!ty) {
    ty = builder.getIndexType();
  }
  auto key = std::make_pair(c, ty);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Value v;
  if (ty.isIndex()) {
    v = builder.create<arith::ConstantIndexOp>(loc, c);
  }
  else {
    v = builder.create<arith::ConstantIntOp>(loc, c,
                                             ty.getIntOrFloatBitWidth());
}
  cache[key] = v;
  return v;
}

// Check that every constant in `e` is representable in the type it would be
// materialized as.  `stride_new` for block-stride ops is a narrow integer
// (i16), and a stride that does not fit would otherwise become an out-of-range
// arith.constant.  Pure, so an out-of-range candidate is rejected before any
// IR is created.
static bool constantsFitType(const StrideExprRef &e, Type wantType) {
  switch (e->kind) {
  case StrideExpr::Kind::Const: {
    if (!wantType || wantType.isIndex()) {
      return true; // index is 64-bit, always holds an int64_t
    }
    unsigned bitWidth = wantType.getIntOrFloatBitWidth();
    return bitWidth >= mlir::pto::kValue64 || llvm::isIntN(bitWidth, e->constant);
  }
  case StrideExpr::Kind::Leaf:
    return true;
  case StrideExpr::Kind::Cast:
    return constantsFitType(e->lhs, e->castOp->getOperand(0).getType());
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul:
    return constantsFitType(e->lhs, wantType) &&
           constantsFitType(e->rhs, wantType);
  }
  return false;
}

static bool satisfiesStrideConstraint(const StrideExprRef &stride,
                                      pto::VPTOAdvanceConstraint constraint) {
  if (constraint == pto::VPTOAdvanceConstraint::Dynamic) {
    return true;
  }
  std::optional<int64_t> constant = foldConst(stride);
  if (!constant) {
    return false;
  }
  return constraint == pto::VPTOAdvanceConstraint::Constant ||
         (*constant >= kSignedI8Min && *constant <= kSignedI8Max);
}

// Emit `e` at the builder's current insertion point.  Sub-expressions are
// emitted bottom-up, so every operand is created before its user and the
// result dominates the insertion point by construction.
static Value materialize(const StrideExprRef &e, Type wantType, Location loc,
                         scf::ForOp forOp, ConstCache &cache,
                         OpBuilder &builder) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return materializeConst(e->constant, wantType, loc, forOp, cache, builder);
  case StrideExpr::Kind::Leaf:
    return e->leaf;
  case StrideExpr::Kind::Cast: {
    Value in = materialize(e->lhs, e->castOp->getOperand(0).getType(), loc,
                           forOp, cache, builder);
    Operation *cloned = builder.clone(*e->castOp);
    cloned->setOperand(0, in);
    return cloned->getResult(0);
  }
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    Value a = materialize(e->lhs, wantType, loc, forOp, cache, builder);
    Value b = materialize(e->rhs, a.getType(), loc, forOp, cache, builder);
    if (e->kind == StrideExpr::Kind::Add) {
      return builder.create<arith::AddIOp>(loc, a, b);
    }
    if (e->kind == StrideExpr::Kind::Sub) {
      return builder.create<arith::SubIOp>(loc, a, b);
    }
    return builder.create<arith::MulIOp>(loc, a, b);
  }
  }
  llvm_unreachable("unhandled StrideExpr kind");
}

// Sequential runs use the operand type fixed by the op definition. Preserve
// every cast in the analyzed expression: dropping a widening cast and
// materializing the surrounding arithmetic in its input type can introduce
// overflow that was absent from the original address calculation. Without a
// range proof, a dynamic expression whose result type differs from the stride
// operand type is conservatively rejected.
static bool canMaterializeAs(const StrideExprRef &e, Type wantType) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return true;
  case StrideExpr::Kind::Leaf:
    return e->leaf.getType() == wantType;
  case StrideExpr::Kind::Cast: {
    Type inputType = e->castOp->getOperand(0).getType();
    Type resultType = e->castOp->getResult(0).getType();
    return wantType == resultType && canMaterializeAs(e->lhs, inputType);
  }
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul:
    return canMaterializeAs(e->lhs, wantType) &&
           canMaterializeAs(e->rhs, wantType);
  }
  return false;
}

// Dominance-aware counterpart of the loop-specific availability check above.
// Values already dominating the run head are reused; later pure definitions in
// the same block may be cloned before it.
static bool canHoistBefore(Value v, Operation *insertPt,
                           DominanceInfo &dominance,
                           DenseMap<Value, bool> &memo) {
  if (dominance.dominates(v, insertPt)) {
    return true;
  }
  auto it = memo.find(v);
  if (it != memo.end()) {
    return it->second;
  }
  memo[v] = false;

  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getBlock() != insertPt->getBlock() || !isPure(defOp)) {
    return false;
  }
  for (Value operand : defOp->getOperands()) {
    if (!canHoistBefore(operand, insertPt, dominance, memo)) {
      return false;
    }
  }
  memo[v] = true;
  return true;
}

static Value hoistBefore(Value v, Operation *insertPt, DominanceInfo &dominance,
                         OpBuilder &builder, DenseMap<Value, Value> &memo) {
  if (dominance.dominates(v, insertPt)) {
    return v;
  }
  if (auto it = memo.find(v); it != memo.end()) {
    return it->second;
  }

  Operation *defOp = v.getDefiningOp();
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    newOperands.push_back(
        hoistBefore(operand, insertPt, dominance, builder, memo));
  }

  builder.setInsertionPoint(insertPt);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands)) {
    cloned->setOperand(i, operand);
  }
  Value result = cloned->getResult(cast<OpResult>(v).getResultNumber());
  memo[v] = result;
  return result;
}

static StrideExprRef makeAvailableAt(const StrideExprRef &e,
                                     Operation *insertPt,
                                     DominanceInfo &dominance,
                                     OpBuilder &builder,
                                     DenseMap<Value, Value> &memo) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return e;
  case StrideExpr::Kind::Leaf: {
    Value available = hoistBefore(e->leaf, insertPt, dominance, builder, memo);
    return available == e->leaf ? e : makeLeaf(available);
  }
  case StrideExpr::Kind::Cast:
    return makeCast(
        e->castOp, makeAvailableAt(e->lhs, insertPt, dominance, builder, memo));
  case StrideExpr::Kind::Add:
    return makeAdd(makeAvailableAt(e->lhs, insertPt, dominance, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, dominance, builder, memo));
  case StrideExpr::Kind::Sub:
    return makeSub(makeAvailableAt(e->lhs, insertPt, dominance, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, dominance, builder, memo));
  case StrideExpr::Kind::Mul:
    return makeMul(makeAvailableAt(e->lhs, insertPt, dominance, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, dominance, builder, memo));
  }
  llvm_unreachable("unhandled StrideExpr kind");
}

static Value materializeSequential(const StrideExprRef &e, Type wantType,
                                   Location loc, OpBuilder &builder) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    if (wantType.isIndex()) {
      return builder.create<arith::ConstantIndexOp>(loc, e->constant);
    }
    return builder.create<arith::ConstantIntOp>(
        loc, e->constant, wantType.getIntOrFloatBitWidth());
  case StrideExpr::Kind::Leaf:
    return e->leaf;
  case StrideExpr::Kind::Cast: {
    Type inputType = e->castOp->getOperand(0).getType();
    Value input = materializeSequential(e->lhs, inputType, loc, builder);
    Operation *cloned = builder.clone(*e->castOp);
    cloned->setOperand(0, input);
    return cloned->getResult(0);
  }
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    Value lhs = materializeSequential(e->lhs, wantType, loc, builder);
    Value rhs = materializeSequential(e->rhs, wantType, loc, builder);
    if (e->kind == StrideExpr::Kind::Add) {
      return builder.create<arith::AddIOp>(loc, lhs, rhs);
    }
    if (e->kind == StrideExpr::Kind::Sub) {
      return builder.create<arith::SubIOp>(loc, lhs, rhs);
    }
    return builder.create<arith::MulIOp>(loc, lhs, rhs);
  }
  }
  llvm_unreachable("unhandled StrideExpr kind");
}

// Information about a post-update transformation to apply.
struct PostUpdateRewrite {
  Operation *op;
  Value base;
  Value strideOperand; // original offset / repeat_stride operand
  Value stride;        // stride value (stride_new for block-stride ops)
  Value initPtr;       // base + strideOperand_at_iter0, in addptr units
  int64_t unitBytes;   // bytes advanced by one unit of stride
};

struct PostUpdateCandidatePlan {
  Operation *op;
  Value base;
  Value currentOffset;
  StrideUnit currentUnit;
  int64_t elementBytes;
  int64_t currentUnitBytes;
  int64_t advanceUnitBytes;
  Type strideType;
  StrideExprRef stride;
};

struct LoopPostUpdatePlan {
  scf::ForOp loop;
  SmallVector<PostUpdateCandidatePlan> candidates;
};

// A unique key for grouping rewrites that can share an iter_arg.
//
// Two ops may share an iter_arg only if they walk the same address sequence,
// i.e. they start at the same address and advance by the same stride.  The
// start address is `initPtr`, which is derived from base *and* strideOperand,
// so strideOperand has to be part of the key: same base and same stride but
// different offsets (e.g. %ub[%iv] and %ub[%iv + 64]) are distinct sequences,
// and merging them would make the second op start at the first one's address.
//
// Keying on the original operands rather than on `initPtr` itself keeps the
// comparison by Value identity meaningful: computeInitialPtr may materialize a
// fresh pto.addptr per candidate, so equal start addresses do not necessarily
// share a Value. The effective byte unit is also part of the address sequence:
// equal numeric strides in element and byte ops need not advance equally.
// This is conservative — it can split groups that could have been merged —
// but never merges groups that must stay apart.
using IterArgGroupKey = std::tuple<Value, Value, Value, int64_t>;

static IterArgGroupKey getGroupKey(const PostUpdateRewrite &rw) {
  return {rw.base, rw.strideOperand, rw.stride, rw.unitBytes};
}

// Build the post-update form of an op while preserving every operand,
// attribute, and original result. The updated base is always appended last.
static Operation *createPostUpdateOp(Operation *op,
                                     const pto::VPTOPostUpdateSemantics &info,
                                     Value base, Value stride,
                                     OpBuilder &builder) {
  OperationState state(op->getLoc(), op->getName());
  for (OpOperand &operand : op->getOpOperands()) {
    if (&operand == info.baseOperand) {
      state.addOperands(base);
    }
    else if (&operand == info.advanceOperand) {
      state.addOperands(stride);
    }
    else {
      state.addOperands(operand.get());
    }
  }
  if (!info.advanceOperand) {
    state.addOperands(stride);
  }
  state.addTypes(op->getResultTypes());
  state.addTypes(base.getType());
  state.addAttributes(op->getAttrs());
  return builder.create(state);
}

// Build the normal form of an op while preserving every operand, attribute,
// and original result. Unlike createPostUpdateOp, no updated base is appended.
static Operation *createNormalOp(Operation *op,
                                 const pto::VPTOPostUpdateSemantics &info,
                                 Value base, Value zeroStride,
                                 OpBuilder &builder) {
  OperationState state(op->getLoc(), op->getName());
  for (OpOperand &operand : op->getOpOperands()) {
    if (&operand == info.baseOperand) {
      state.addOperands(base);
    }
    else if (&operand == info.advanceOperand) {
      state.addOperands(zeroStride);
    }
    else {
      state.addOperands(operand.get());
    }
  }
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  return builder.create(state);
}

// Remove loop-carried recurrences that became dead after post-update rewrites.
// Ordinary DCE cannot break a recurrence such as
//   %next = arith.addi %iter_arg, %step
//   scf.yield %next
// even when neither the iter_arg nor the loop result has a real user: the
// block argument, update, and yield form a use cycle. Compute liveness from
// side-effecting operations and externally used loop results, close it across
// the loop backedge, then rebuild the loop with only live iter_args and pure
// operations that feed live values.
static scf::ForOp pruneDeadLoopCarriedValues(scf::ForOp forOp,
                                             OpBuilder &builder) {
  unsigned numIterArgs = forOp.getInitArgs().size();
  if (numIterArgs == 0) {
    return forOp;
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  DenseSet<Value> liveValues;
  SmallVector<Value> worklist;
  SmallVector<bool> keepIterArg(numIterArgs, false);

  auto markLive = [&liveValues, &worklist](Value value) {
    if (value && liveValues.insert(value).second) {
      worklist.push_back(value);
    }
  };

  // A loop result used outside the loop keeps its corresponding backedge live.
  for (auto [idx, result] : llvm::enumerate(forOp.getResults())) {
    if (result.use_empty()) {
      continue;
    }
    keepIterArg[idx] = true;
    markLive(yieldOp.getOperand(idx));
  }

  // Side-effecting operations are liveness roots. Region-bearing operations
  // are handled conservatively: keep their own operands and every value
  // captured by a nested region, even when the nested computation is pure,
  // rather than attempting a second control-flow-sensitive liveness analysis
  // here.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isPure(&op)) {
      for (Value operand : op.getOperands()) {
        markLive(operand);
      }
    }
    if (op.getNumRegions() != 0) {
      op.walk([&markLive](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          markLive(operand);
        }
      });
    }
  }

  // Propagate liveness through pure def chains and across loop backedges.
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      if (blockArg.getOwner() != forOp.getBody() ||
          blockArg.getArgNumber() == 0) {
        continue;
      }
      unsigned idx = blockArg.getArgNumber() - 1;
      if (!keepIterArg[idx]) {
        keepIterArg[idx] = true;
        markLive(yieldOp.getOperand(idx));
      }
      continue;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || !forOp->isAncestor(defOp)) {
      continue;
    }
    for (Value operand : defOp->getOperands()) {
      markLive(operand);
    }
  }

  if (llvm::all_of(keepIterArg, [](bool keep) { return keep; })) {
    return forOp;
  }

  SmallVector<Value> newInitArgs;
  newInitArgs.reserve(numIterArgs);
  for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
    if (keepIterArg[idx]) {
      newInitArgs.push_back(init);
    }
  }

  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  unsigned newArgIdx = 0;
  for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (keepIterArg[idx]) {
      mapping.map(oldArg, newForOp.getRegionIterArgs()[newArgIdx++]);
    }
  }

  builder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    bool hasLiveResult = llvm::any_of(op.getResults(), [&liveValues](Value result) {
      return liveValues.contains(result);
    });
    if (!isPure(&op) || op.getNumRegions() != 0 || hasLiveResult) {
      builder.clone(op, mapping);
    }
  }

  SmallVector<Value> newYields;
  newYields.reserve(newInitArgs.size());
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (keepIterArg[idx]) {
      newYields.push_back(mapping.lookupOrDefault(yielded));
    }
  }
  builder.setInsertionPointToEnd(newForOp.getBody());
  builder.create<scf::YieldOp>(yieldOp.getLoc(), newYields);

  unsigned newResultIdx = 0;
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    if (keepIterArg[idx]) {
      oldResult.replaceAllUsesWith(newForOp.getResult(newResultIdx++));
      continue;
    }
    assert(oldResult.use_empty() && "cannot drop a used scf.for result");
  }

  forOp.erase();
  return newForOp;
}

// Apply post-update rewrites to a single scf.for.
// Returns the new ForOp if any rewrites were applied, null otherwise.
static scf::ForOp applyPostUpdateRewrites(scf::ForOp forOp,
                                          ArrayRef<PostUpdateRewrite> rewrites,
                                          OpBuilder &builder) {
  if (rewrites.empty()) {
    return nullptr;
  }

  // Group rewrites by start-address operands, stride, and effective byte unit.
  // Ops in the same group share one iter_arg and all use the pre-update
  // pointer. Only one updated_base per group is yielded. This avoids redundant
  // iter_args for same-address ops (e.g. vlds + vsts both accessing
  // %base[%iv]) without merging byte- and element-scaled recurrences.
  DenseMap<IterArgGroupKey, unsigned> groupToIdx; // group key -> iter_arg index
  SmallVector<unsigned> rwGroupIdx(rewrites.size()); // rewrite -> group index
  SmallVector<Value>
      groupInitPtrs; // initial pointer per group (base + offset_at_iter0)

  for (auto [i, rw] : llvm::enumerate(rewrites)) {
    auto key = getGroupKey(rw);
    auto [it, inserted] = groupToIdx.try_emplace(key, groupInitPtrs.size());
    if (inserted) {
      groupInitPtrs.push_back(rw.initPtr);
    }
    rwGroupIdx[i] = it->second;
  }

  unsigned numGroups = groupInitPtrs.size();

  // Build new init args: original + one new pointer per group.
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  for (Value ptr : groupInitPtrs) {
    newInitArgs.push_back(ptr);
  }

  unsigned origIterArgCount = forOp.getInitArgs().size();

  // Create new ForOp.
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());

  // Map old block args to new: IV + original iter_args.
  IRMapping mapping;
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < origIterArgCount; ++i) {
    mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));
  }

  // Clone the body, tracking old->new op correspondence.
  DenseMap<Operation *, Operation *> opMapping;
  builder.setInsertionPointToStart(newBody);
  for (auto &op : oldBody->without_terminator()) {
    Operation *cloned = builder.clone(op, mapping);
    opMapping[&op] = cloned;
  }

  // Apply rewrites. All ops in a group use the same pre-update pointer (block
  // arg). Track the last updated_base per group for yielding.
  SmallVector<Value> groupYieldPtrs(numGroups);
  for (unsigned g = 0; g < numGroups; ++g) {
    groupYieldPtrs[g] = newBody->getArgument(origIterArgCount + 1 + g);
  }

  for (auto [rwIdx, rw] : llvm::enumerate(rewrites)) {
    auto it = opMapping.find(rw.op);
    if (it == opMapping.end()) {
      continue;
    }
    Operation *clonedOp = it->second;
    unsigned gIdx = rwGroupIdx[rwIdx];
    Value ptr = newBody->getArgument(origIterArgCount + 1 + gIdx);
    Value strideNew = mapping.lookupOrDefault(rw.stride);

    builder.setInsertionPoint(clonedOp);

    auto info = getPostUpdateSemantics(clonedOp);
    if (!info) {
      continue;
    }

    Operation *newOp =
        createPostUpdateOp(clonedOp, *info, ptr, strideNew, builder);

    // Replace old results with new and update the mapping so that later
    // yield construction via mapping.lookupOrDefault sees the new results
    // instead of dangling pointers to the erased clonedOp.
    for (unsigned r = 0; r < clonedOp->getNumResults(); ++r) {
      clonedOp->getResult(r).replaceAllUsesWith(newOp->getResult(r));
      mapping.map(rw.op->getResult(r), newOp->getResult(r));
    }

    groupYieldPtrs[gIdx] =
        getPostUpdateSemantics(newOp)->updatedBase;
    clonedOp->erase();
  }

  // Build yield: original yields + one pointer per group.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYields;
  for (Value v : oldYield.getOperands()) {
    newYields.push_back(mapping.lookupOrDefault(v));
  }
  for (Value ptr : groupYieldPtrs) {
    newYields.push_back(ptr);
  }

  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYields);

  // Replace original ForOp results (only the original ones).
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  }

  forOp.erase();
  return pruneDeadLoopCarriedValues(newForOp, builder);
}

//===----------------------------------------------------------------------===//
// Sequential Path
//===----------------------------------------------------------------------===//

struct SequentialCandidate {
  Operation *op;
  OpOperand *advanceOperand;
  pto::VPTOAdvanceConstraint constraint;
  Value base;
  Value currentOffset;
  Value advanceValue;
  pto::PTOAddressExpr address;
  Value rootBase;
  int64_t elemBytes;
  StrideUnit currentUnit;
  int64_t currentUnitBytes;
  int64_t unitBytes; // post-access advance unit bytes
};

struct SequentialBucket {
  OperationName opName;
  Value rootBase;
  SmallVector<SequentialCandidate> candidates;
};

struct SequentialStep {
  StrideExprRef expr;
  pto::PTOLinearExpr form;
};

static std::optional<SequentialStep>
analyzeSequentialStep(const SequentialCandidate &previous,
                      const SequentialCandidate &current,
                      pto::PTOAddressAnalysis &addressAnalysis) {
  if (previous.elemBytes != current.elemBytes ||
      previous.unitBytes != current.unitBytes) {
    return std::nullopt;
  }

  auto deltaBytes =
      addressAnalysis.getDifferenceBytes(previous.address, current.address);
  if (!deltaBytes) {
    return std::nullopt;
  }
  auto deltaInUnit =
      addressAnalysis.convertDeltaToUnit(*deltaBytes.value,
                                         current.unitBytes);
  if (!deltaInUnit) {
    return std::nullopt;
  }
  auto form = pto::normalizePTOLinearExpr(*deltaInUnit.value);
  if (!form || pto::isZeroPTOLinearExpr(*form)) {
    return std::nullopt;
  }
  StrideExprRef step = importTypedExpr(
      pto::buildPTOTypedExpr(*form, (*deltaInUnit.value)->type));
  if (!step) {
    return std::nullopt;
  }
  return SequentialStep{step, std::move(*form)};
}

struct SequentialRun {
  SmallVector<SequentialCandidate *> candidates;
  StrideExprRef step;
  pto::PTOLinearExpr stepForm;
  Type strideType;
  Value strideValue;
  Value zeroStride;
  Value currentPtr;
};

static bool validateSequentialRun(SequentialRun &run,
                                  DominanceInfo &dominance) {
  if (run.candidates.size() < mlir::pto::kValue3) {
    return false;
  }

  SequentialCandidate *first = run.candidates.front();
  run.strideType = first->advanceValue
                       ? first->advanceValue.getType()
                       : IndexType::get(first->op->getContext());
  if (!canMaterializeAs(run.step, run.strideType) ||
      !constantsFitType(run.step, run.strideType) ||
      !satisfiesStrideConstraint(run.step,
                                 first->constraint) ||
      !canScaleInitialOffset(first->currentOffset, first->elemBytes,
                             first->currentUnitBytes)) {
    return false;
  }

  for (SequentialCandidate *candidate : run.candidates) {
    Type candidateStrideType =
        candidate->advanceValue
            ? candidate->advanceValue.getType()
            : IndexType::get(candidate->op->getContext());
    if (candidateStrideType != run.strideType) {
      return false;
    }
  }

  SmallVector<Value> leaves;
  collectLeaves(run.step, leaves);
  DenseMap<Value, bool> canCache;
  return llvm::all_of(leaves, [&](Value leaf) {
    return canHoistBefore(leaf, first->op, dominance, canCache);
  });
}

static bool hasOnlyExpectedUser(Value value, Operation *expectedUser) {
  return value.hasOneUse() && *value.getUsers().begin() == expectedUser;
}

static bool isDynamicSequentialValue(
    Value value, pto::PTOValueEvolutionAnalysis &valueEvolution) {
  auto form = pto::normalizePTOLinearExpr(valueEvolution.getExpr(value));
  return form && !form->terms.empty();
}

// Count only addptrs that are guaranteed to disappear after the candidates
// following the run head are rewritten. The first candidate's base chain is
// retained to construct the initial pointer and therefore is not a saving.
static unsigned countDeadDynamicAddPtrs(
    const SequentialRun &run,
    pto::PTOValueEvolutionAnalysis &valueEvolution) {
  DenseSet<Operation *> counted;
  for (SequentialCandidate *candidate : llvm::drop_begin(run.candidates)) {
    Value value = candidate->base;
    Operation *expectedUser = candidate->op;
    while (auto addPtr = value.getDefiningOp<pto::AddPtrOp>()) {
      if (!hasOnlyExpectedUser(value, expectedUser)) {
        break;
      }
      if (isDynamicSequentialValue(addPtr.getOffset(), valueEvolution)) {
        counted.insert(addPtr);
      }
      expectedUser = addPtr;
      value = addPtr.getPtr();
    }
  }
  return counted.size();
}

static unsigned initialPointerCost(const SequentialRun &run) {
  SequentialCandidate *first = run.candidates.front();
  if (!first->currentOffset) {
    return 0;
  }
  auto initialOffset = getConstantIntValue(first->currentOffset);
  return initialOffset && *initialOffset == 0 ? 0 : 1;
}

static bool isRunStrideUse(OpOperand &use, const SequentialRun &run) {
  return llvm::any_of(run.candidates, [&use](SequentialCandidate *candidate) {
    return candidate->advanceOperand == &use;
  });
}

// Collect the cumulative add/sub chain used to form the third and later
// offsets of a direct symbolic-leaf run. Unsupported producers are the
// symbolic leaves at which this slice stops.
static void collectCumulativeOffsetOps(Value value,
                                       DenseSet<Operation *> &ops) {
  Operation *defOp = value.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp>(defOp) ||
      !ops.insert(defOp).second) {
    return;
  }
  for (Value operand : defOp->getOperands()) {
    collectCumulativeOffsetOps(operand, ops);
  }
}

static bool allUsesDisappearAfterRewrite(Operation *op,
                                         const DenseSet<Operation *> &deadOps,
                                         const SequentialRun &run) {
  return llvm::all_of(op->getResults(), [&](Value result) {
    return llvm::all_of(result.getUses(), [&](OpOperand &use) {
      return deadOps.contains(use.getOwner()) || isRunStrideUse(use, run);
    });
  });
}

static bool cumulativeOffsetChainDefinitelyDies(
    const SequentialRun &run, DenseSet<Operation *> &deadOps) {
  for (SequentialCandidate *candidate :
       llvm::drop_begin(run.candidates, mlir::pto::kValue2)) {
    if (candidate->currentOffset) {
      collectCumulativeOffsetOps(candidate->currentOffset, deadOps);
    }
  }
  return !deadOps.empty() &&
llvm::all_of(deadOps, [&deadOps, &run](Operation *op) {
            return allUsesDisappearAfterRewrite(op, deadOps, run);
         });
}

static bool collectLatePureDefinitions(Value value, Operation *runHead,
                                       DominanceInfo &dominance,
                                       DenseSet<Operation *> &clonedOps) {
  if (dominance.dominates(value, runHead)) {
    return true;
  }
  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != runHead->getBlock() || !isPure(defOp)) {
    return false;
  }
  if (!clonedOps.insert(defOp).second) {
    return true;
  }
  return llvm::all_of(defOp->getOperands(), [&](Value operand) {
    return collectLatePureDefinitions(operand, runHead, dominance, clonedOps);
  });
}

// Direct symbolic steps are either reused at the run head or cloned there.
// Cloning is cost-neutral only when the original pure definition chain becomes
// dead after the address operands are replaced.
static bool isStepMaterializationCostNeutral(
    const SequentialRun &run, const DenseSet<Operation *> &deadOffsetOps,
    DominanceInfo &dominance) {
  SequentialCandidate *first = run.candidates.front();
  DenseSet<Operation *> clonedOps;
  pto::PTOTypedExprRef atom = run.stepForm.terms.front().atom;
  if (atom->sourceValue) {
    if (!collectLatePureDefinitions(atom->sourceValue, first->op, dominance,
                                    clonedOps)) {
      return false;
    }
  } else if (atom->kind == pto::PTOTypedExpr::Kind::Cast) {
    if (!atom->sourceOperation) {
      return false;
    }
    clonedOps.insert(atom->sourceOperation);
    SmallVector<Value> leaves;
    pto::collectPTOExprLeaves(atom, leaves);
    for (Value leaf : leaves) {
      if (!collectLatePureDefinitions(leaf, first->op, dominance, clonedOps)) {
        return false;
      }
    }
  } else if (atom->kind == pto::PTOTypedExpr::Kind::Opaque &&
             !collectLatePureDefinitions(atom->opaque, first->op, dominance,
                                         clonedOps)) {
    return false;
  }

  DenseSet<Operation *> disappearing = deadOffsetOps;
  disappearing.insert(clonedOps.begin(), clonedOps.end());
  return llvm::all_of(clonedOps, [&disappearing, &run](Operation *op) {
    return allUsesDisappearAfterRewrite(op, disappearing, run);
  });
}

static bool isProfitableDynamicBaseRun(
    const SequentialRun &run,
    pto::PTOValueEvolutionAnalysis &valueEvolution) {
  if (!run.stepForm.terms.empty()) {
    return false;
  }
  unsigned pointerCost = run.candidates.size() - 1;
  return countDeadDynamicAddPtrs(run, valueEvolution) >
         pointerCost + initialPointerCost(run);
}

static bool isProfitableDirectSymbolicLeafRun(
    const SequentialRun &run, DominanceInfo &dominance) {
  // validateSequentialRun already enforces N >= 3. This class intentionally
  // has no higher length threshold, so an N3 run may be accepted.
  if (run.stepForm.constant != 0 || run.stepForm.terms.size() != 1 ||
      run.stepForm.terms.front().coefficient != 1) {
    return false;
  }

  // This class covers a direct fixed base with offsets 0, step, 2*step, ...
  // Dynamic base chains are handled independently above.
  if (!llvm::all_of(run.candidates, [](SequentialCandidate *candidate) {
        return candidate->base == candidate->rootBase;
      })) {
    return false;
  }
  Value firstStrideOperand = run.candidates.front()->currentOffset;
  auto firstOffset = firstStrideOperand
                         ? getConstantIntValue(firstStrideOperand)
                         : std::optional<int64_t>(0);
  if (!firstOffset || *firstOffset != 0) {
    return false;
  }

  DenseSet<Operation *> deadOffsetOps;
  if (!cumulativeOffsetChainDefinitelyDies(run, deadOffsetOps)) {
    return false;
  }
  return isStepMaterializationCostNeutral(run, deadOffsetOps, dominance);
}

// Profitability is intentionally a structural whitelist rather than a
// weighted MLIR-op cost model. It applies uniformly to every supported
// post-update op: either later candidates delete enough dynamic addptr work, or
// a direct symbolic step replaces a cumulative address chain.
static bool isProfitableSequentialRun(const SequentialRun &run,
                                      DominanceInfo &dominance,
                                      pto::PTOValueEvolutionAnalysis
                                          &valueEvolution) {
  return isProfitableDynamicBaseRun(run, valueEvolution) ||
         isProfitableDirectSymbolicLeafRun(run, dominance);
}

static void collectNestedBlocks(Operation *op, pto::VecScopeOp owner,
                                SmallVectorImpl<Block *> &blocks) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      blocks.push_back(&block);
      for (Operation &nested : block) {
        if (auto nestedScope = dyn_cast<pto::VecScopeOp>(nested);
            nestedScope && nestedScope != owner) {
          continue;
        }
        collectNestedBlocks(&nested, owner, blocks);
      }
    }
  }
}

static void processSequentialBlock(Block *block, DominanceInfo &dominance,
                                   pto::PTOAddressAnalysis &addressAnalysis,
                                   OpBuilder &builder) {
  SmallVector<Operation *> originalOps;
  SmallVector<SequentialBucket> buckets;

  for (Operation &op : *block) {
    originalOps.push_back(&op);
    auto postUpdate = getPostUpdateSemantics(&op);
    if (!postUpdate || postUpdate->updatedBase) {
      continue;
    }

    auto addresses = addressAnalysis.getAddresses(&op);
    if (
        !addresses || addresses.value->size() != 1) {
      continue;
    }
    pto::PTOAddressExpr address = addresses.value->front();
    int64_t elemBytes = address.elementBytes;
    auto unitBytes = pto::getVPTOAddressUnitBytes(
        &op, postUpdate->advanceUnit, postUpdate->elementTypeSource);
    if (!unitBytes) {
      continue;
    }
    Value currentOffset =
        address.offset ? address.offset->sourceValue : Value();
    StrideUnit currentUnit = address.offset ? address.offset->unit
                                            : StrideUnit::Element;
    int64_t currentUnitBytes =
        address.offset && address.offset->unitBytes
            ? *address.offset->unitBytes
            : elemBytes;
    auto bucketIt = llvm::find_if(
        buckets, [&op, &address](const SequentialBucket &bucket) {
          return bucket.opName == op.getName() &&
                 bucket.rootBase == address.rootOrBase;
        });
    if (bucketIt == buckets.end()) {
      buckets.push_back({op.getName(), address.rootOrBase, {}});
      bucketIt = std::prev(buckets.end());
    }
    bucketIt->candidates.push_back({
        &op,
        postUpdate->advanceOperand,
        postUpdate->constraint,
        address.currentBase,
        currentOffset,
        postUpdate->advanceOperand ? postUpdate->advanceOperand->get()
                                   : Value(),
        std::move(address),
        addresses.value->front().rootOrBase,
        elemBytes,
        currentUnit,
        currentUnitBytes,
        *unitBytes,
    });
  }

  SmallVector<SequentialRun> runs;
  for (SequentialBucket &bucket : buckets) {
    auto &candidates = bucket.candidates;
    size_t start = 0;
    while (start + 1 < candidates.size()) {
      auto firstStep =
          analyzeSequentialStep(candidates[start], candidates[start + 1],
                                addressAnalysis);
      if (!firstStep) {
        ++start;
        continue;
      }

      size_t end = start + 2;
      while (end < candidates.size()) {
        auto nextStep =
            analyzeSequentialStep(candidates[end - 1], candidates[end],
                                  addressAnalysis);
        if (!nextStep ||
            !pto::equalPTOLinearExprs(firstStep->form, nextStep->form)) {
          break;
        }
        ++end;
      }

      SequentialRun run;
      run.step = firstStep->expr;
      run.stepForm = firstStep->form;
      for (size_t i = start; i < end; ++i) {
        run.candidates.push_back(&candidates[i]);
      }
      if (validateSequentialRun(run, dominance) &&
          isProfitableSequentialRun(
              run, dominance, addressAnalysis.getValueEvolution())) {
        runs.push_back(std::move(run));
        // Accepted runs are deliberately non-overlapping. The candidate that
        // broke the current stride becomes the start of the next run.
        start = end;
      } else {
        // Reuse the rejected run's last candidate as the next head so it can
        // form a new run with the candidate that broke the current stride.
        start = end - 1;
      }
    }
  }

  if (runs.empty()) {
    return;
  }

  // Materialize every accepted run before erasing any candidate op.
  for (SequentialRun &run : runs) {
    SequentialCandidate *first = run.candidates.front();
    DenseMap<Value, Value> hoistMemo;
    StrideExprRef available =
        makeAvailableAt(run.step, first->op, dominance, builder, hoistMemo);
    builder.setInsertionPoint(first->op);
    run.strideValue = materializeSequential(available, run.strideType,
                                            first->op->getLoc(), builder);
    run.zeroStride = materializeSequential(makeConst(0), run.strideType,
                                           first->op->getLoc(), builder);
    builder.setInsertionPoint(first->op);
    run.currentPtr = createInitialPtr(
        first->base, first->currentOffset, first->currentUnit,
        first->elemBytes, first->currentUnitBytes, first->op->getLoc(),
        builder);
  }

  DenseMap<Operation *, unsigned> opToRun;
  for (auto [runIdx, run] : llvm::enumerate(runs)) {
    for (SequentialCandidate *candidate : run.candidates) {
      opToRun[candidate->op] = runIdx;
    }
  }

  // Rewrite in original program order so interleaved buckets maintain separate
  // pointer chains without invalidating one another.
  for (Operation *op : originalOps) {
    auto it = opToRun.find(op);
    if (it == opToRun.end()) {
      continue;
    }
    SequentialRun &run = runs[it->second];
    auto postUpdate = getPostUpdateSemantics(op);
    if (!postUpdate) {
      continue;
    }
    builder.setInsertionPoint(op);
    bool isLast = op == run.candidates.back()->op;
    Operation *newOp =
        isLast ? createNormalOp(op, *postUpdate, run.currentPtr,
                                run.zeroStride, builder)
               : createPostUpdateOp(op, *postUpdate, run.currentPtr,
                                    run.strideValue, builder);
    for (unsigned result = 0; result < op->getNumResults(); ++result) {
      op->getResult(result).replaceAllUsesWith(newOp->getResult(result));
    }
    if (!isLast) {
      run.currentPtr = getPostUpdateSemantics(newOp)->updatedBase;
    }
    op->erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct VPTOSoftPostUpdatePass
    : public pto::impl::VPTOSoftPostUpdateBase<VPTOSoftPostUpdatePass> {
  using pto::impl::VPTOSoftPostUpdateBase<
      VPTOSoftPostUpdatePass>::VPTOSoftPostUpdateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());
    module.walk([&](func::FuncOp function) {
      auto &addressAnalysis =
          getAnalysisManager()
              .getChildAnalysis<pto::PTOAddressAnalysis, func::FuncOp>(
                  function);

      SmallVector<pto::VecScopeOp> vecscopes;
      function.walk(
          [&vecscopes](pto::VecScopeOp vecscope) { vecscopes.push_back(vecscope); });

      // Build every loop rewrite plan in the function before the first
      // mutation. Reusing a function analysis after rewriting an earlier
      // vecscope would otherwise observe stale cached expressions.
      SmallVector<LoopPostUpdatePlan> plans;
      function.walk([&](scf::ForOp forOp) {
        if (!forOp->getParentOfType<pto::VecScopeOp>()) {
          return;
        }
        LoopPostUpdatePlan plan =
            analyzeForOp(forOp, addressAnalysis, builder);
        if (!plan.candidates.empty()) {
          plans.push_back(std::move(plan));
        }
      });
      for (LoopPostUpdatePlan &plan : plans) {
        applyLoopPlan(plan, builder);
      }

      // Sequential analysis is block-local. Construct a fresh public analysis
      // pair for each block so no cached fact survives an earlier block's
      // mutation.
      for (pto::VecScopeOp vecscope : vecscopes) {
        SmallVector<Block *> blocks;
        collectNestedBlocks(vecscope, vecscope, blocks);
        for (Block *block : blocks) {
          DominanceInfo dominance(vecscope->getParentOp());
          pto::PTOValueEvolutionAnalysis freshValueAnalysis(function);
          pto::PTOAddressAnalysis freshAddressAnalysis(function,
                                                       freshValueAnalysis);
          processSequentialBlock(block, dominance, freshAddressAnalysis,
                                 builder);
        }
      }
    });
  }

private:
  LoopPostUpdatePlan analyzeForOp(
      scf::ForOp forOp, pto::PTOAddressAnalysis &addressAnalysis,
      OpBuilder &builder) {
    LoopPostUpdatePlan plan{forOp, {}};
    for (Operation &op : *forOp.getBody()) {
      auto postUpdate = getPostUpdateSemantics(&op);
      if (!postUpdate || postUpdate->updatedBase ||
          !isDirectlyInForBody(&op, forOp)) {
        continue;
      }

      auto addresses = addressAnalysis.getAddresses(&op);
      if (
          !addresses || addresses.value->size() != 1) {
        continue;
      }
      const pto::PTOAddressExpr &address = addresses.value->front();
      auto advanceUnitBytes = pto::getVPTOAddressUnitBytes(
          &op, postUpdate->advanceUnit, postUpdate->elementTypeSource);
      if (!advanceUnitBytes) {
        continue;
      }
      auto delta = addressAnalysis.getDeltaInUnit(
          address, forOp, *advanceUnitBytes);
      if (!delta) {
        continue;
      }
      if (auto linear = pto::normalizePTOLinearExpr(*delta.value);
          linear && pto::isZeroPTOLinearExpr(*linear)) {
        continue;
      }
      StrideExprRef total = importTypedExpr(*delta.value);
      if (!total) {
        continue;
      }

      // Reject expressions whose subterms demand conflicting types, or whose
      // dynamic result cannot be materialized exactly as the op's declared
      // stride operand type.
      Type exprResultType;
      if (!exprType(total, exprResultType)) {
        continue;
      }
      Value advanceOperand = postUpdate->advanceOperand
                                 ? postUpdate->advanceOperand->get()
                                 : Value();
      Type strideType = advanceOperand ? advanceOperand.getType()
                                       : builder.getIndexType();
      if (exprResultType && exprResultType != strideType) {
        continue;
      }

      // Reject strides whose constants do not fit the target operand type.
      if (!constantsFitType(total, strideType)) {
        continue;
      }
      if (!satisfiesStrideConstraint(total, postUpdate->constraint)) {
        continue;
      }

      SmallVector<Value> leaves;
      collectLeaves(total, leaves);
      DenseMap<Value, bool> canCache;
      if (!llvm::all_of(leaves, [&](Value leaf) {
            return canHoistBefore(leaf, &op, forOp, canCache);
          })) {
        continue;
      }

      Value currentOffset =
          address.offset ? address.offset->sourceValue : Value();
      StrideUnit currentUnit = address.offset ? address.offset->unit
                                              : StrideUnit::Element;
      int64_t currentUnitBytes =
          address.offset && address.offset->unitBytes
              ? *address.offset->unitBytes
              : address.elementBytes;
      if (currentOffset &&
          !canScaleInitialOffsetAtLoopEntry(
              currentOffset, address.elementBytes, currentUnitBytes, forOp,
              addressAnalysis.getValueEvolution())) {
        continue;
      }
      plan.candidates.push_back(
          {&op, address.currentBase, currentOffset, currentUnit,
           address.elementBytes, currentUnitBytes, *advanceUnitBytes,
           strideType, total});
    }
    return plan;
  }

  void applyLoopPlan(LoopPostUpdatePlan &plan, OpBuilder &builder) {
    SmallVector<PostUpdateRewrite> rewrites;
    ConstCache constCache;
    for (PostUpdateCandidatePlan &candidate : plan.candidates) {
      SmallVector<Value> leaves;
      collectLeaves(candidate.stride, leaves);
      bool allInvariant = llvm::all_of(leaves, [&](Value leaf) {
        return plan.loop.isDefinedOutsideOfLoop(leaf);
      });
      StrideExprRef finalExpression = candidate.stride;
      if (!allInvariant) {
        DenseMap<Value, Value> hoistMemo;
        finalExpression =
            makeAvailableAt(candidate.stride, candidate.op, plan.loop,
                            builder, hoistMemo);
      }
      builder.setInsertionPoint(allInvariant ? plan.loop.getOperation()
                                             : candidate.op);
      Value stride = materialize(finalExpression, candidate.strideType,
                                 candidate.op->getLoc(), plan.loop, constCache,
                                 builder);
      Value initialPointer = computeInitialPtr(
          candidate.base, candidate.currentOffset, candidate.currentUnit,
          candidate.elementBytes, candidate.currentUnitBytes, plan.loop,
          builder);
      if (!initialPointer) {
        continue;
      }
      rewrites.push_back({candidate.op, candidate.base,
                          candidate.currentOffset, stride, initialPointer,
                          candidate.advanceUnitBytes});
    }
    if (!rewrites.empty()) {
      applyPostUpdateRewrites(plan.loop, rewrites, builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSoftPostUpdatePass() {
  return std::make_unique<VPTOSoftPostUpdatePass>();
}
