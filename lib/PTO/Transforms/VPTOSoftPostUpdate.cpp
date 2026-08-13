// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOPostUpdateUtils.h"
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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSOFTPOSTUPDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

// Narrow loop-varying integer addresses use a certified recurrence of this
// width. Pure index-domain analysis remains local to soft post-update.
static constexpr unsigned kCanonicalAddressWidth = 16;

using StrideUnit = pto::PostUpdateAddressUnit;
using StrideConstraint = pto::PostUpdateStrideConstraint;
using AddressDomain = pto::PostUpdateAddressDomain;
using PostUpdateOpInfo = pto::PostUpdateOpInfo;

static const PostUpdateOpInfo *getPostUpdateInfo(Operation *op) {
  return pto::getPostUpdateOpInfo(op);
}

static std::optional<int64_t> addPtrUnitBytes(Value base) {
  return pto::getPostUpdateBaseUnitBytes(base);
}

static std::optional<int64_t>
strideUnitBytes(Operation *op, const PostUpdateOpInfo &info,
                int64_t elemBytes) {
  return pto::getPostUpdateAddressUnitBytes(op, info, elemBytes);
}

// Extract base and stride operand from a candidate op using table info.
static void extractBaseAndStrideOperand(Operation *op,
                                        const PostUpdateOpInfo &info,
                                        Value &base, Value &strideOperand) {
  base = op->getOperand(info.baseOperandIdx);
  strideOperand =
      info.strideOperandIdx ? op->getOperand(*info.strideOperandIdx) : Value();
}

// Check if op already has an updated_base result.
static bool isAlreadyPostUpdate(Operation *op, const PostUpdateOpInfo &info) {
  return op->getNumResults() > info.minResultsForPost;
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
  if (!e)
    return std::nullopt;
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
    if (!a || !b)
      return std::nullopt;
    if (e->kind == StrideExpr::Kind::Add)
      return *a + *b;
    if (e->kind == StrideExpr::Kind::Sub)
      return *a - *b;
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
  if (ca && cb)
    return makeConst(*ca + *cb);
  if (ca && *ca == 0)
    return b;
  if (cb && *cb == 0)
    return a;
  return makeBinary(StrideExpr::Kind::Add, a, b);
}

static StrideExprRef makeSub(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb)
    return makeConst(*ca - *cb);
  if (cb && *cb == 0)
    return a;
  return makeBinary(StrideExpr::Kind::Sub, a, b);
}

static StrideExprRef makeMul(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb)
    return makeConst(*ca * *cb);
  if ((ca && *ca == 0) || (cb && *cb == 0))
    return makeConst(0);
  if (ca && *ca == 1)
    return b;
  if (cb && *cb == 1)
    return a;
  return makeBinary(StrideExpr::Kind::Mul, a, b);
}

static StrideExprRef makeCast(Operation *castOp, StrideExprRef a) {
  if (auto c = foldConst(a))
    return makeConst(*c);
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Cast;
  e->castOp = castOp;
  e->lhs = std::move(a);
  return e;
}

// A canonical affine view of StrideExpr used for exact division and symbolic
// equality. Cast expressions are kept as typed atoms so materialization can
// preserve them and reject a final operand type that would require narrowing.
struct AffineTerm {
  StrideExprRef atom;
  int64_t coeff;
};

struct AffineForm {
  int64_t constant = 0;
  SmallVector<AffineTerm> terms;
};

static bool sameAffineAtom(const StrideExprRef &a, const StrideExprRef &b) {
  if (!a || !b || a->kind != b->kind)
    return false;
  if (a->kind == StrideExpr::Kind::Leaf)
    return a->leaf == b->leaf;
  if (a->kind != StrideExpr::Kind::Cast)
    return false;
  return a->castOp->getName() == b->castOp->getName() &&
         a->castOp->getOperand(0).getType() ==
             b->castOp->getOperand(0).getType() &&
         a->castOp->getResult(0).getType() ==
             b->castOp->getResult(0).getType() &&
         sameAffineAtom(a->lhs, b->lhs);
}

static bool addAffineConstant(AffineForm &form, int64_t value) {
  int64_t result;
  if (llvm::AddOverflow(form.constant, value, result))
    return false;
  form.constant = result;
  return true;
}

static bool addAffineTerm(AffineForm &form, StrideExprRef atom, int64_t coeff) {
  if (coeff == 0)
    return true;
  for (unsigned i = 0; i < form.terms.size(); ++i) {
    if (!sameAffineAtom(form.terms[i].atom, atom))
      continue;
    int64_t result;
    if (llvm::AddOverflow(form.terms[i].coeff, coeff, result))
      return false;
    if (result == 0)
      form.terms.erase(form.terms.begin() + i);
    else
      form.terms[i].coeff = result;
    return true;
  }
  form.terms.push_back({std::move(atom), coeff});
  return true;
}

static bool accumulateAffine(const StrideExprRef &e, int64_t scale,
                             AffineForm &form) {
  if (!e)
    return false;
  switch (e->kind) {
  case StrideExpr::Kind::Const: {
    int64_t scaled;
    return !llvm::MulOverflow(e->constant, scale, scaled) &&
           addAffineConstant(form, scaled);
  }
  case StrideExpr::Kind::Leaf: {
    if (auto c = getConstantIntValue(e->leaf)) {
      int64_t scaled;
      return !llvm::MulOverflow(*c, scale, scaled) &&
             addAffineConstant(form, scaled);
    }
    return addAffineTerm(form, e, scale);
  }
  case StrideExpr::Kind::Cast:
    if (auto c = foldConst(e)) {
      int64_t scaled;
      return !llvm::MulOverflow(*c, scale, scaled) &&
             addAffineConstant(form, scaled);
    }
    return addAffineTerm(form, e, scale);
  case StrideExpr::Kind::Add:
    return accumulateAffine(e->lhs, scale, form) &&
           accumulateAffine(e->rhs, scale, form);
  case StrideExpr::Kind::Sub: {
    int64_t negScale;
    return !llvm::SubOverflow(int64_t{0}, scale, negScale) &&
           accumulateAffine(e->lhs, scale, form) &&
           accumulateAffine(e->rhs, negScale, form);
  }
  case StrideExpr::Kind::Mul: {
    if (auto lhsConst = foldConst(e->lhs)) {
      int64_t newScale;
      return !llvm::MulOverflow(scale, *lhsConst, newScale) &&
             accumulateAffine(e->rhs, newScale, form);
    }
    if (auto rhsConst = foldConst(e->rhs)) {
      int64_t newScale;
      return !llvm::MulOverflow(scale, *rhsConst, newScale) &&
             accumulateAffine(e->lhs, newScale, form);
    }
    return false;
  }
  }
  return false;
}

static std::optional<AffineForm> normalizeAffine(const StrideExprRef &e) {
  AffineForm form;
  if (!accumulateAffine(e, 1, form))
    return std::nullopt;
  return form;
}

static bool equalAffineForms(const AffineForm &a, const AffineForm &b) {
  if (a.constant != b.constant || a.terms.size() != b.terms.size())
    return false;
  for (const AffineTerm &termA : a.terms) {
    bool found = llvm::any_of(b.terms, [&](const AffineTerm &termB) {
      return termA.coeff == termB.coeff &&
             sameAffineAtom(termA.atom, termB.atom);
    });
    if (!found)
      return false;
  }
  return true;
}

static Value stripSignedAddressCasts(Value value) {
  while (Operation *defOp = value.getDefiningOp()) {
    if (!isa<arith::IndexCastOp, arith::ExtSIOp>(defOp))
      break;
    value = defOp->getOperand(0);
  }
  return value;
}

static bool sameAddressAdvanceAtom(const StrideExprRef &a,
                                   const StrideExprRef &b) {
  if (!a || !b || a->kind != StrideExpr::Kind::Leaf ||
      b->kind != StrideExpr::Kind::Leaf)
    return sameAffineAtom(a, b);
  return stripSignedAddressCasts(a->leaf) == stripSignedAddressCasts(b->leaf);
}

static bool equalAddressAdvanceExprs(const StrideExprRef &a,
                                     const StrideExprRef &b) {
  auto formA = normalizeAffine(a);
  auto formB = normalizeAffine(b);
  if (!formA || !formB || formA->constant != formB->constant ||
      formA->terms.size() != formB->terms.size())
    return false;
  for (const AffineTerm &termA : formA->terms) {
    if (llvm::none_of(formB->terms, [&](const AffineTerm &termB) {
          return termA.coeff == termB.coeff &&
                 sameAddressAdvanceAtom(termA.atom, termB.atom);
        }))
      return false;
  }
  return true;
}

static bool isZeroAffineForm(const AffineForm &form) {
  return form.constant == 0 && form.terms.empty();
}

static bool divideAffineForm(AffineForm &form, int64_t divisor) {
  if (divisor <= 0 || form.constant % divisor != 0)
    return false;
  for (const AffineTerm &term : form.terms)
    if (term.coeff % divisor != 0)
      return false;
  form.constant /= divisor;
  for (AffineTerm &term : form.terms)
    term.coeff /= divisor;
  return true;
}

static StrideExprRef affineFormToExpr(const AffineForm &form) {
  StrideExprRef result;
  if (form.constant != 0)
    result = makeConst(form.constant);
  for (const AffineTerm &term : form.terms) {
    StrideExprRef value = term.atom;
    if (term.coeff != 1)
      value = makeMul(value, makeConst(term.coeff));
    result = result ? makeAdd(result, value) : value;
  }
  return result ? result : makeConst(0);
}

static void collectLeaves(const StrideExprRef &e, SmallVectorImpl<Value> &out) {
  if (!e)
    return;
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
    if (!exprType(e->lhs, ta) || !exprType(e->rhs, tb))
      return false;
    if (ta && tb && ta != tb)
      return false;
    out = ta ? ta : tb;
    return true;
  }
  }
  return false;
}

// Result of decomposing a value into blockArg * coeff + increment.
struct LinearDecomp {
  int64_t coeff;
  StrideExprRef increment; // never null; zero is makeConst(0)
};

using DecompCache = DenseMap<Value, std::optional<LinearDecomp>>;

static bool castPreservesLoopDelta(Operation *castOp, scf::ForOp forOp);

// Decompose `v` into blockArg * coeff + increment by recursing through
// addi/subi/muli/index_cast/addptr chains.  Pure: builds only StrideExprs.
// `cache` is scoped to one decomposition (a single `blockArg`).
static std::optional<LinearDecomp> decomposeLinear(Value v,
                                                   BlockArgument blockArg,
                                                   scf::ForOp forOp,
                                                   DecompCache &cache) {
  // v == blockArg → {1, 0}
  if (v == blockArg)
    return LinearDecomp{1, makeConst(0)};

  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;
  auto record = [&](std::optional<LinearDecomp> r) {
    cache[v] = r;
    return r;
  };

  // v is other block arg (IV, different iter_arg, func arg) → {0, v}
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return record(LinearDecomp{0, makeLeaf(v)});

  // v is loop-invariant or constant → {0, v}
  if (forOp.isDefinedOutsideOfLoop(v) ||
      defOp->hasTrait<OpTrait::ConstantLike>())
    return record(LinearDecomp{0, makeLeaf(v)});

  // v = addi(a, b) → {ca + cb, ia + ib}
  // v = subi(a, b) → {ca - cb, ia - ib}
  if (isa<arith::AddIOp, arith::SubIOp>(defOp)) {
    auto da = decomposeLinear(defOp->getOperand(0), blockArg, forOp, cache);
    auto db = decomposeLinear(defOp->getOperand(1), blockArg, forOp, cache);
    if (!da || !db)
      return record(std::nullopt);
    if (da->coeff == 0 && db->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    bool isSub = isa<arith::SubIOp>(defOp);
    return record(
        LinearDecomp{isSub ? da->coeff - db->coeff : da->coeff + db->coeff,
                     isSub ? makeSub(da->increment, db->increment)
                           : makeAdd(da->increment, db->increment)});
  }

  // v = muli(a, b), one side blockArg-free with constant k → {c * k, i * k}
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto da = decomposeLinear(mulOp.getLhs(), blockArg, forOp, cache);
    auto db = decomposeLinear(mulOp.getRhs(), blockArg, forOp, cache);
    if (!da || !db)
      return record(std::nullopt);
    if (da->coeff == 0 && db->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    if (da->coeff != 0 && db->coeff != 0)
      return record(std::nullopt);
    const LinearDecomp &withBA = (da->coeff != 0) ? *da : *db;
    Value multiplier = (da->coeff != 0) ? mulOp.getRhs() : mulOp.getLhs();
    auto constMul = getConstantIntValue(multiplier);
    if (!constMul)
      return record(std::nullopt);
    return record(
        LinearDecomp{withBA.coeff * *constMul,
                     makeMul(withBA.increment, makeConst(*constMul))});
  }

  // v = address_cast(a) → {ca, cast(ia)} when the cast preserves loop delta
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp, arith::ExtSIOp,
          arith::ExtUIOp>(defOp)) {
    auto d = decomposeLinear(defOp->getOperand(0), blockArg, forOp, cache);
    if (!d)
      return record(std::nullopt);
    if (d->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    if (!castPreservesLoopDelta(defOp, forOp))
      return record(std::nullopt);
    return record(LinearDecomp{d->coeff, makeCast(defOp, d->increment)});
  }

  // v = addptr(ptr, offset) → {c_ptr, i_ptr + offset}
  if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(defOp)) {
    auto dp = decomposeLinear(addPtrOp.getPtr(), blockArg, forOp, cache);
    if (!dp)
      return record(std::nullopt);
    if (dp->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    return record(LinearDecomp{
        dp->coeff, makeAdd(dp->increment, makeLeaf(addPtrOp.getOffset()))});
  }

  // Unrecognized op → unknown
  return record(std::nullopt);
}

// Outcome of accumulator analysis.  Distinguishing "not an iter_arg" from
// "is an iter_arg but could not be decomposed" matters: the former falls back
// to delta analysis, the latter must give up (treating an unknown increment as
// zero would silently miscompile).
enum class StrideStatus { NotIterArg, Failed, Ok };

struct StrideResult {
  StrideStatus status;
  StrideExprRef expr; // valid only when status == Ok
};

// Trace `v` back to an iter_arg BlockArgument of `forOp`, then decompose
// the yield expression to extract the per-iteration increment. Walks through
// delta-preserving index_cast (type-changing), addi/subi with loop-invariant
// offset, and addptr with loop-invariant offset. Type casts along the path are
// applied to the increment so its type matches v's context. Pure: creates no
// IR.
static StrideResult
getIterArgIncrement(Value v, scf::ForOp forOp,
                    std::optional<AddressDomain> expectedDomain) {
  SmallVector<Operation *> casts;
  Value current = v;

  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (blockArg.getOwner() != forOp.getBody() ||
          blockArg.getArgNumber() == 0)
        return {StrideStatus::NotIterArg, nullptr};

      if (isa<IntegerType>(blockArg.getType())) {
        if (!expectedDomain)
          return {StrideStatus::Failed, nullptr};
        auto step = pto::getCanonicalAddressRecurrenceStep(blockArg, forOp,
                                                           *expectedDomain);
        if (!step)
          return {StrideStatus::Failed, nullptr};
        StrideExprRef inc = makeConst(*step);
        for (Operation *op : llvm::reverse(casts))
          inc = makeCast(op, inc);
        return {StrideStatus::Ok, inc};
      }

      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      DecompCache cache;
      auto decomp =
          decomposeLinear(yieldOp.getOperand(idx), blockArg, forOp, cache);
      if (!decomp || decomp->coeff != 1)
        return {StrideStatus::Failed, nullptr};

      StrideExprRef inc = decomp->increment;
      for (Operation *op : llvm::reverse(casts))
        inc = makeCast(op, inc);
      return {StrideStatus::Ok, inc};
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp || forOp.isDefinedOutsideOfLoop(current) ||
        defOp->hasTrait<OpTrait::ConstantLike>())
      return {StrideStatus::NotIterArg, nullptr};

    if (isa<arith::IndexCastUIOp, arith::IndexCastOp, arith::ExtSIOp,
            arith::ExtUIOp>(defOp)) {
      if (!castPreservesLoopDelta(defOp, forOp))
        return {StrideStatus::Failed, nullptr};
      casts.push_back(defOp);
      current = defOp->getOperand(0);
      continue;
    }

    if (isa<arith::AddIOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(1))) {
        current = defOp->getOperand(0);
        continue;
      }
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(0))) {
        current = defOp->getOperand(1);
        continue;
      }
    }

    if (isa<arith::SubIOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(1))) {
        current = defOp->getOperand(0);
        continue;
      }
    }

    if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(addPtrOp.getOffset())) {
        current = addPtrOp.getPtr();
        continue;
      }
    }

    return {StrideStatus::NotIterArg, nullptr};
  }
}

//===----------------------------------------------------------------------===//
// Delta Analysis
//===----------------------------------------------------------------------===//

// Cached delta results.  A cached null expr means "delta is unknown"; absence
// from the map means "not computed yet".
using DeltaCache = DenseMap<Value, StrideExprRef>;

// Loop-varying casts commute with delta only when they consume the explicit
// canonical recurrence contract emitted by address normalization. The nsw/nuw
// flag on that recurrence is the proof certificate; soft post-update does not
// repeat trip-count and endpoint analysis.
static bool castPreservesLoopDelta(Operation *castOp, scf::ForOp forOp) {
  Value input = castOp->getOperand(0);
  if (forOp.isDefinedOutsideOfLoop(input))
    return true; // Both the cast value and its delta (zero) are invariant.

  auto inputInteger = dyn_cast<IntegerType>(input.getType());
  if (!inputInteger)
    return false;

  AddressDomain domain;
  if (isa<arith::IndexCastOp, arith::ExtSIOp>(castOp))
    domain = AddressDomain::Signed;
  else if (isa<arith::IndexCastUIOp, arith::ExtUIOp>(castOp))
    domain = AddressDomain::Unsigned;
  else
    return false;

  if (inputInteger.getWidth() == kCanonicalAddressWidth)
    return pto::getCanonicalAddressRecurrenceStep(input, forOp, domain)
        .has_value();

  Operation *extension = input.getDefiningOp();
  if (domain == AddressDomain::Signed &&
      !isa_and_nonnull<arith::ExtSIOp>(extension))
    return false;
  if (domain == AddressDomain::Unsigned &&
      !isa_and_nonnull<arith::ExtUIOp>(extension))
    return false;
  Value canonicalInput = extension->getOperand(0);
  return pto::getCanonicalAddressRecurrenceStep(canonicalInput, forOp, domain)
      .has_value();
}

// Compute the per-iteration delta of value `v` within `forOp`.
// Returns a loop-invariant symbolic delta, or null if unknown.  Pure.
static StrideExprRef computeDelta(Value v, scf::ForOp forOp,
                                  DeltaCache &cache) {
  // IV: delta = step
  if (v == forOp.getInductionVar())
    return makeLeaf(forOp.getStep());

  // Constant or loop-invariant: delta = 0
  if (forOp.isDefinedOutsideOfLoop(v))
    return makeConst(0);

  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;
  auto record = [&](StrideExprRef r) {
    cache[v] = r;
    return r;
  };

  // Block argument from iter_args: check yield = arg + c
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() && blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value yieldVal = yieldOp.getOperand(idx);
      if (auto addOp = yieldVal.getDefiningOp<arith::AddIOp>()) {
        Value other;
        if (addOp.getLhs() == blockArg)
          other = addOp.getRhs();
        else if (addOp.getRhs() == blockArg)
          other = addOp.getLhs();
        if (other && forOp.isDefinedOutsideOfLoop(other))
          return record(makeLeaf(other));
      }
      return record(nullptr);
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return record(nullptr);

  // arith.addi(a, b): delta = delta(a) + delta(b)
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto da = computeDelta(addOp.getLhs(), forOp, cache);
    auto db = computeDelta(addOp.getRhs(), forOp, cache);
    if (!da || !db)
      return record(nullptr);
    return record(makeAdd(da, db));
  }

  // arith.subi(a, b): delta = delta(a) - delta(b)
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto da = computeDelta(subOp.getLhs(), forOp, cache);
    auto db = computeDelta(subOp.getRhs(), forOp, cache);
    if (!da || !db)
      return record(nullptr);
    return record(makeSub(da, db));
  }

  // arith.muli(a, b) where one is loop-invariant:
  //   delta = invariant * delta(other)
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();
    for (auto [invariant, variant] :
         {std::pair{rhs, lhs}, std::pair{lhs, rhs}}) {
      if (forOp.isDefinedOutsideOfLoop(invariant)) {
        auto dv = computeDelta(variant, forOp, cache);
        if (!dv)
          continue;
        return record(makeMul(makeLeaf(invariant), dv));
      }
    }
    return record(nullptr);
  }

  // pto.addptr(ptr, offset): both the pointer and integer offset contribute
  // in addptr element units. This is the canonical base form emitted by the
  // address-recurrence normalization pass for a loop-varying base.
  if (auto addPtr = dyn_cast<pto::AddPtrOp>(defOp)) {
    auto pointerDelta = computeDelta(addPtr.getPtr(), forOp, cache);
    auto offsetDelta = computeDelta(addPtr.getOffset(), forOp, cache);
    if (!pointerDelta || !offsetDelta)
      return record(nullptr);
    return record(makeAdd(pointerDelta, offsetDelta));
  }

  // Preserve value-preserving casts in the symbolic delta. Narrowing casts are
  // accepted only when castPreservesLoopDelta proves they cannot truncate.
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp, arith::ExtSIOp,
          arith::ExtUIOp>(defOp)) {
    if (!castPreservesLoopDelta(defOp, forOp))
      return record(nullptr);
    StrideExprRef inputDelta = computeDelta(defOp->getOperand(0), forOp, cache);
    return record(inputDelta ? makeCast(defOp, inputDelta) : nullptr);
  }

  return record(nullptr);
}

// Get the per-iteration stride of `v`: tries accumulator analysis first
// (for iter_arg-derived values with possibly loop-varying increment),
// falls back to delta analysis (for IV-derived values, loop-invariant result).
// Returns null if the stride cannot be determined.
static StrideExprRef
getStride(Value v, scf::ForOp forOp, DeltaCache &cache,
          std::optional<AddressDomain> expectedDomain = std::nullopt) {
  StrideResult r = getIterArgIncrement(v, forOp, expectedDomain);
  if (r.status == StrideStatus::Ok)
    return r.expr;
  if (r.status == StrideStatus::Failed)
    return nullptr;
  return computeDelta(v, forOp, cache);
}

// Narrow values must trace through canonical signed/unsigned widening casts to
// an i16 iter_arg carrying the corresponding nsw/nuw backedge certificate.
// Pure index-domain IVs and recurrences retain the existing soft analysis.
static bool isCanonicalLoopInteger(Value value, scf::ForOp forOp) {
  if (forOp.isDefinedOutsideOfLoop(value))
    return true;

  std::optional<AddressDomain> domain;
  while (Operation *defOp = value.getDefiningOp()) {
    AddressDomain castDomain;
    if (isa<arith::IndexCastOp, arith::ExtSIOp>(defOp))
      castDomain = AddressDomain::Signed;
    else if (isa<arith::IndexCastUIOp, arith::ExtUIOp>(defOp))
      castDomain = AddressDomain::Unsigned;
    else
      break;
    if (domain && *domain != castDomain)
      return false;
    domain = castDomain;
    value = defOp->getOperand(0);
  }

  if (domain)
    return pto::getCanonicalAddressRecurrenceStep(value, forOp, *domain)
        .has_value();
  return pto::getCanonicalAddressRecurrenceStep(value, forOp,
                                                AddressDomain::Signed)
             .has_value() ||
         pto::getCanonicalAddressRecurrenceStep(value, forOp,
                                                AddressDomain::Unsigned)
             .has_value();
}

// Pure index-domain recurrences retain the existing soft analysis. The
// canonical recurrence requirement applies when a loop-varying value is
// narrow, or when an index value was obtained by widening a narrow integer.
static bool isSafeLoopInteger(Value value, scf::ForOp forOp) {
  if (forOp.isDefinedOutsideOfLoop(value))
    return true;
  if (!value.getType().isIndex())
    return isCanonicalLoopInteger(value, forOp);
  Operation *defOp = value.getDefiningOp();
  if (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp>(defOp))
    return castPreservesLoopDelta(defOp, forOp);
  return true;
}

static bool hasOnlySafeLoopIntegerLeaves(const StrideExprRef &expr,
                                         scf::ForOp forOp) {
  SmallVector<Value> leaves;
  collectLeaves(expr, leaves);
  return llvm::all_of(leaves, [&](Value leaf) {
    return !isa<IntegerType, IndexType>(leaf.getType()) ||
           isSafeLoopInteger(leaf, forOp);
  });
}

static bool isCanonicalLoopBase(Value base, scf::ForOp forOp) {
  if (forOp.isDefinedOutsideOfLoop(base))
    return true;

  if (auto addPtr = base.getDefiningOp<pto::AddPtrOp>())
    return forOp.isDefinedOutsideOfLoop(addPtr.getPtr()) &&
           isSafeLoopInteger(addPtr.getOffset(), forOp);

  auto iterArg = dyn_cast<BlockArgument>(base);
  if (!iterArg || iterArg.getOwner() != forOp.getBody() ||
      iterArg.getArgNumber() == 0)
    return false;
  unsigned index = iterArg.getArgNumber() - 1;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  auto addPtr = yieldOp.getOperand(index).getDefiningOp<pto::AddPtrOp>();
  return addPtr && addPtr.getPtr() == base &&
         isSafeLoopInteger(addPtr.getOffset(), forOp);
}

//===----------------------------------------------------------------------===//
// Rewrite: create new ForOp with additional iter_arg
//===----------------------------------------------------------------------===//

// Compute the value of `v` at the first iteration (IV = lower bound) by
// cloning the def-chain with IV replaced by the lower bound.  Returns nullptr
// if `v` cannot be materialized outside the loop.
static Value materializeAtLoopEntry(Value v, scf::ForOp forOp,
                                    OpBuilder &builder) {
  // IV → lower bound
  if (v == forOp.getInductionVar())
    return forOp.getLowerBound();

  // Already defined outside the loop — use directly.
  if (forOp.isDefinedOutsideOfLoop(v))
    return v;

  // iter_arg → its init value
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() && blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      return forOp.getInitArgs()[idx];
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return nullptr;

  // Cloning duplicates the op, so it must be safe to execute an extra time and
  // its result must not depend on anything but its operands.
  if (!isPure(defOp))
    return nullptr;

  // Clone the defining op with operands materialized at loop entry.
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    Value materialized = materializeAtLoopEntry(operand, forOp, builder);
    if (!materialized)
      return nullptr;
    newOperands.push_back(materialized);
  }
  builder.setInsertionPoint(forOp);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
  // Preserve which result was asked for; `v` need not be result 0.
  return cloned->getResult(cast<OpResult>(v).getResultNumber());
}

// Whether strideOperand can be restated in pto.addptr element units without
// creating IR. Element and block units are always exact for supported element
// types; finer byte units require a suitably aligned compile-time constant.
static bool canScaleInitialOffset(Value strideOperand, int64_t elemBytes,
                                  int64_t unitBytes) {
  if (!strideOperand || unitBytes == elemBytes || unitBytes % elemBytes == 0)
    return true;
  if (elemBytes % unitBytes != 0)
    return false;
  auto getConstantThroughSignedCasts = [&](Value value) {
    while (Operation *defOp = value.getDefiningOp()) {
      if (!isa<arith::IndexCastOp, arith::ExtSIOp>(defOp))
        break;
      value = defOp->getOperand(0);
    }
    return getConstantIntValue(value);
  };
  auto constant = getConstantThroughSignedCasts(strideOperand);
  return constant && *constant % (elemBytes / unitBytes) == 0;
}

// Reproduce the index-to-i32 truncation performed by vlds/vsts lowering before
// using an Element-class offset in pto.addptr.  A direct index -> i32 -> index
// round trip is canonicalized away by arith, so route through i64 + trunci to
// keep the narrowing explicit.  The final signed index_cast preserves negative
// offsets in the same way as the existing lowering.
static Value truncateElementOffsetToI32(Value offset, Location loc,
                                        OpBuilder &builder) {
  if (!offset.getType().isIndex())
    return offset;
  Value offsetI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), offset);
  Value offsetI32 =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), offsetI64);
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            offsetI32);
}

// pto.addptr always consumes an index offset. Block offsets retain their
// existing unsigned interpretation; every other supported address unit is
// signed, including sprsti's signed 8-bit word offset.
static Value normalizeAddPtrOffsetToIndex(Value offset, StrideUnit strideUnit,
                                          Location loc, OpBuilder &builder) {
  if (offset.getType().isIndex())
    return offset;
  if (strideUnit == StrideUnit::Block)
    return builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(),
                                                offset);
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            offset);
}

// Create the address reached by one memory op before post-update rewriting.
// The builder must already point at the desired insertion location.
static Value createInitialPtr(Value base, Value strideOperand,
                              bool strideParticipatesInCurrentAddress,
                              StrideUnit strideUnit, int64_t elemBytes,
                              int64_t unitBytes, Location loc,
                              OpBuilder &builder) {
  if (!strideParticipatesInCurrentAddress || !strideOperand)
    return base;
  Value constantSource = strideOperand;
  while (Operation *defOp = constantSource.getDefiningOp()) {
    if (!isa<arith::IndexCastOp, arith::ExtSIOp>(defOp))
      break;
    constantSource = defOp->getOperand(0);
  }
  auto constSo = getConstantIntValue(constantSource);
  if (constSo && *constSo == 0)
    return base;
  if (!canScaleInitialOffset(strideOperand, elemBytes, unitBytes))
    return nullptr;

  Value scaledOffset =
      strideUnit == StrideUnit::Element
          ? truncateElementOffsetToI32(strideOperand, loc, builder)
          : strideOperand;
  if (unitBytes != elemBytes) {
    if (unitBytes % elemBytes == 0) {
      Value soIndex = strideOperand;
      if (strideOperand.getType() != builder.getIndexType())
        soIndex = builder.create<arith::IndexCastUIOp>(
            loc, builder.getIndexType(), strideOperand);
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
                               bool strideParticipatesInCurrentAddress,
                               StrideUnit strideUnit, int64_t elemBytes,
                               int64_t unitBytes, scf::ForOp forOp,
                               OpBuilder &builder) {
  Value baseAtEntry = materializeAtLoopEntry(base, forOp, builder);
  if (!baseAtEntry)
    return nullptr;

  if (!strideParticipatesInCurrentAddress || !strideOperand)
    return baseAtEntry;

  Value soAtEntry = materializeAtLoopEntry(strideOperand, forOp, builder);
  if (!soAtEntry)
    return nullptr;

  builder.setInsertionPoint(forOp);
  return createInitialPtr(baseAtEntry, soAtEntry,
                          strideParticipatesInCurrentAddress, strideUnit,
                          elemBytes, unitBytes, forOp.getLoc(), builder);
}

// Rescale a per-iteration base delta from `pto.addptr` units (elements) into
// the op's strideOperand units.  Returns null when the conversion is not exact.
//
// Expanding the byte-denominated form
//     stride_new = (E*delta(base) + W*delta(strideOperand)) / W
//                = (E/W)*delta(base) + delta(strideOperand)
// shows that only delta(base) is ever rescaled; delta(strideOperand) passes
// through untouched.  That matters: it keeps the stride symbolic, so a
// loop-varying increment stays supported for every op class.  When E == W the
// factor is 1 and nothing is emitted at all, so Element-class ops (vlds/vsts)
// behave exactly as before this scaling existed.
static StrideExprRef scaleBaseDelta(StrideExprRef deltaBase, int64_t elemBytes,
                                    int64_t unitBytes) {
  if (unitBytes == elemBytes)
    return deltaBase;

  if (unitBytes % elemBytes == 0) {
    // Coarser stride unit (e.g. 32-byte blocks over 4-byte elements): the base
    // delta's affine coefficients must all land on a whole unit.
    int64_t divisor = unitBytes / elemBytes;
    auto form = normalizeAffine(deltaBase);
    if (!form || !divideAffineForm(*form, divisor))
      return nullptr;
    return affineFormToExpr(*form);
  }

  if (elemBytes % unitBytes == 0)
    return makeMul(deltaBase, makeConst(elemBytes / unitBytes));

  return nullptr;
}

// Combine per-operand strides into the final stride_new for the post-update op.
// stride_new = (E/W) * deltaBase + deltaOffset, where E is the byte size of one
// addptr unit and W the byte size of one strideOperand unit.  Returns null if
// the stride is zero or the rescaling is inexact.  Purely symbolic: a rejected
// candidate leaves no IR behind, and the scaling is folded rather than emitted,
// so no `index` factor is ever multiplied against a narrower stride operand.
static StrideExprRef combineStride(StrideExprRef deltaBase,
                                   StrideExprRef deltaOffset, int64_t elemBytes,
                                   int64_t unitBytes) {
  StrideExprRef scaledBase = scaleBaseDelta(deltaBase, elemBytes, unitBytes);
  if (!scaledBase)
    return nullptr;

  StrideExprRef total = makeAdd(scaledBase, deltaOffset);
  if (auto form = normalizeAffine(total)) {
    if (isZeroAffineForm(*form))
      return nullptr;
    return affineFormToExpr(*form);
  }
  if (auto constTotal = foldConst(total); constTotal && *constTotal == 0)
    return nullptr;
  return total;
}

//===----------------------------------------------------------------------===//
// Materialization
//===----------------------------------------------------------------------===//

// Can `v` be made available immediately before `insertPt`, cloning a pure
// def-chain if needed?  Pure: inspects only, never mutates the IR.
static bool canHoistBefore(Value v, Operation *insertPt, scf::ForOp forOp,
                           DenseMap<Value, bool> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v) || isa<BlockArgument>(v))
    return true;
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return true;
  if (defOp->getBlock() != insertPt->getBlock())
    return false;
  // Already earlier in the block, so usable as-is: SSA guarantees its operands
  // are defined even earlier.  This reasoning is only sound because analysis
  // creates no IR — every value examined here predates the transform.
  if (defOp->isBeforeInBlock(insertPt))
    return true;
  auto it = memo.find(v);
  if (it != memo.end())
    return it->second;
  memo[v] = false;
  if (!isPure(defOp))
    return false;
  for (Value operand : defOp->getOperands())
    if (!canHoistBefore(operand, insertPt, forOp, memo))
      return false;
  memo[v] = true;
  return true;
}

// Clone `v`'s def-chain before `insertPt` as needed.  Only valid after
// canHoistBefore has approved `v`.
static Value hoistBefore(Value v, Operation *insertPt, scf::ForOp forOp,
                         OpBuilder &builder, DenseMap<Value, Value> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v) || isa<BlockArgument>(v))
    return v;
  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getBlock() != insertPt->getBlock() ||
      defOp->isBeforeInBlock(insertPt))
    return v;
  auto it = memo.find(v);
  if (it != memo.end())
    return it->second;

  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands())
    newOperands.push_back(hoistBefore(operand, insertPt, forOp, builder, memo));

  builder.setInsertionPoint(insertPt);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
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
  if (!ty)
    ty = builder.getIndexType();
  auto key = std::make_pair(c, ty);
  if (auto it = cache.find(key); it != cache.end())
    return it->second;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Value v;
  if (ty.isIndex())
    v = builder.create<arith::ConstantIndexOp>(loc, c);
  else
    v = builder.create<arith::ConstantIntOp>(loc, c,
                                             ty.getIntOrFloatBitWidth());
  cache[key] = v;
  return v;
}

// Check that every constant in `e` is representable in the type it would be
// materialized as.  `stride_new` for block-stride ops is a narrow integer
// (i16), and a stride that does not fit would otherwise become an out-of-range
// arith.constant.  Pure, so an out-of-range candidate is rejected before any
// IR is created.
static bool constantsFitType(const StrideExprRef &e, Type wantType,
                             AddressDomain domain) {
  switch (e->kind) {
  case StrideExpr::Kind::Const: {
    if (!wantType || wantType.isIndex())
      return true; // index is 64-bit, always holds an int64_t
    unsigned bitWidth = wantType.getIntOrFloatBitWidth();
    return bitWidth >= 64 ||
           (domain == AddressDomain::Signed
                ? llvm::isIntN(bitWidth, e->constant)
                : e->constant >= 0 && llvm::isUIntN(bitWidth, e->constant));
  }
  case StrideExpr::Kind::Leaf:
    return true;
  case StrideExpr::Kind::Cast:
    return constantsFitType(e->lhs, e->castOp->getOperand(0).getType(),
                            isa<arith::IndexCastUIOp, arith::ExtUIOp>(e->castOp)
                                ? AddressDomain::Unsigned
                                : AddressDomain::Signed);
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul:
    return constantsFitType(e->lhs, wantType, domain) &&
           constantsFitType(e->rhs, wantType, domain);
  }
  return false;
}

static bool satisfiesStrideConstraint(const StrideExprRef &stride,
                                      StrideConstraint constraint) {
  return pto::satisfiesPostUpdateStrideConstraint(constraint,
                                                  foldConst(stride));
}

// Address normalization may restore an i32 operand and then cast it to index
// for pto.addptr. When the resulting base delta becomes an i32-stride op's
// post increment, drop only that outer signed index cast; the inner i16->i32
// canonical extension and its no-wrap proof remain intact.
static StrideExprRef adaptCanonicalAddressExprType(const StrideExprRef &expr,
                                                   Type wantedType) {
  if (!expr || expr->kind != StrideExpr::Kind::Cast)
    return expr;
  if (expr->castOp->getResult(0).getType() == wantedType)
    return expr;
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(expr->castOp) &&
      expr->castOp->getOperand(0).getType() == wantedType)
    return expr->lhs;
  return expr;
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
    if (e->kind == StrideExpr::Kind::Add)
      return builder.create<arith::AddIOp>(loc, a, b);
    if (e->kind == StrideExpr::Kind::Sub)
      return builder.create<arith::SubIOp>(loc, a, b);
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
  if (dominance.dominates(v, insertPt))
    return true;
  auto it = memo.find(v);
  if (it != memo.end())
    return it->second;
  memo[v] = false;

  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getBlock() != insertPt->getBlock() || !isPure(defOp))
    return false;
  for (Value operand : defOp->getOperands())
    if (!canHoistBefore(operand, insertPt, dominance, memo))
      return false;
  memo[v] = true;
  return true;
}

static Value hoistBefore(Value v, Operation *insertPt, DominanceInfo &dominance,
                         OpBuilder &builder, DenseMap<Value, Value> &memo) {
  if (dominance.dominates(v, insertPt))
    return v;
  if (auto it = memo.find(v); it != memo.end())
    return it->second;

  Operation *defOp = v.getDefiningOp();
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands())
    newOperands.push_back(
        hoistBefore(operand, insertPt, dominance, builder, memo));

  builder.setInsertionPoint(insertPt);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
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
    if (wantType.isIndex())
      return builder.create<arith::ConstantIndexOp>(loc, e->constant);
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
    if (e->kind == StrideExpr::Kind::Add)
      return builder.create<arith::AddIOp>(loc, lhs, rhs);
    if (e->kind == StrideExpr::Kind::Sub)
      return builder.create<arith::SubIOp>(loc, lhs, rhs);
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
                                     const PostUpdateOpInfo &info, Value base,
                                     Value stride, OpBuilder &builder) {
  OperationState state(op->getLoc(), op->getName());
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    if (i == info.baseOperandIdx)
      state.addOperands(base);
    else if (info.strideOperandIdx && i == *info.strideOperandIdx)
      state.addOperands(info.strideParticipatesInCurrentAddress ? stride
                                                                : operand);
    else
      state.addOperands(operand);
  }
  if (!info.strideOperandIdx)
    state.addOperands(stride);
  state.addTypes(op->getResultTypes());
  state.addTypes(base.getType());
  state.addAttributes(op->getAttrs());
  return builder.create(state);
}

// Build the normal form of an op while preserving every operand, attribute,
// and original result. Unlike createPostUpdateOp, no updated base is appended.
static Operation *createNormalOp(Operation *op, const PostUpdateOpInfo &info,
                                 Value base, Value zeroStride,
                                 OpBuilder &builder) {
  OperationState state(op->getLoc(), op->getName());
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    if (i == info.baseOperandIdx)
      state.addOperands(base);
    else if (info.strideOperandIdx && i == *info.strideOperandIdx)
      state.addOperands(info.strideParticipatesInCurrentAddress ? zeroStride
                                                                : operand);
    else
      state.addOperands(operand);
  }
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  return builder.create(state);
}

// Initial-address planning can leave pure constants/casts outside the rebuilt
// loop. Remove those def chains before the sequential path observes the block.
// Region-bearing operations are deliberately excluded.
static void eraseDeadPureOps(Operation *root) {
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> dead;
    root->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op != root && op->getNumRegions() == 0 && op->getNumResults() != 0 &&
          isPure(op) && llvm::all_of(op->getResults(), [](Value result) {
            return result.use_empty();
          }))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  } while (changed);
}

// Apply post-update rewrites to a single scf.for.
// Returns the new ForOp if any rewrites were applied, null otherwise.
static scf::ForOp applyPostUpdateRewrites(scf::ForOp forOp,
                                          ArrayRef<PostUpdateRewrite> rewrites,
                                          OpBuilder &builder) {
  if (rewrites.empty())
    return nullptr;

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
    if (inserted)
      groupInitPtrs.push_back(rw.initPtr);
    rwGroupIdx[i] = it->second;
  }

  unsigned numGroups = groupInitPtrs.size();

  // Build new init args: original + one new pointer per group.
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  for (Value ptr : groupInitPtrs)
    newInitArgs.push_back(ptr);

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
  for (unsigned i = 0; i < origIterArgCount; ++i)
    mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));

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
  for (unsigned g = 0; g < numGroups; ++g)
    groupYieldPtrs[g] = newBody->getArgument(origIterArgCount + 1 + g);

  for (auto [rwIdx, rw] : llvm::enumerate(rewrites)) {
    auto it = opMapping.find(rw.op);
    if (it == opMapping.end())
      continue;
    Operation *clonedOp = it->second;
    unsigned gIdx = rwGroupIdx[rwIdx];
    Value ptr = newBody->getArgument(origIterArgCount + 1 + gIdx);
    Value strideNew = mapping.lookupOrDefault(rw.stride);

    builder.setInsertionPoint(clonedOp);

    const PostUpdateOpInfo *info = getPostUpdateInfo(clonedOp);
    if (!info)
      continue;

    Operation *newOp =
        createPostUpdateOp(clonedOp, *info, ptr, strideNew, builder);

    // Replace old results with new and update the mapping so that later
    // yield construction via mapping.lookupOrDefault sees the new results
    // instead of dangling pointers to the erased clonedOp.
    for (unsigned r = 0; r < clonedOp->getNumResults(); ++r) {
      clonedOp->getResult(r).replaceAllUsesWith(newOp->getResult(r));
      mapping.map(rw.op->getResult(r), newOp->getResult(r));
    }

    // updated_base is the last result.
    groupYieldPtrs[gIdx] = newOp->getResult(newOp->getNumResults() - 1);
    clonedOp->erase();
  }

  // Build yield: original yields + one pointer per group.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYields;
  for (Value v : oldYield.getOperands())
    newYields.push_back(mapping.lookupOrDefault(v));
  for (Value ptr : groupYieldPtrs)
    newYields.push_back(ptr);

  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYields);

  // Replace original ForOp results (only the original ones).
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

  forOp.erase();
  return pto::pruneDeadLoopCarriedValues(newForOp, builder);
}

//===----------------------------------------------------------------------===//
// Sequential Path
//===----------------------------------------------------------------------===//

using SequentialExprCache = DenseMap<Value, StrideExprRef>;

// Build an affine StrideExpr for one scalar address operand. Unsupported
// arithmetic remains an opaque leaf, so it may still cancel when the exact same
// SSA value is reused without guessing at its semantics.
static StrideExprRef buildSequentialExpr(Value value,
                                         SequentialExprCache &cache) {
  if (!value)
    return makeConst(0);
  if (auto constant = getConstantIntValue(value))
    return makeConst(*constant);
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  Operation *defOp = value.getDefiningOp();
  StrideExprRef result;
  if (auto add = dyn_cast_or_null<arith::AddIOp>(defOp)) {
    result = makeAdd(buildSequentialExpr(add.getLhs(), cache),
                     buildSequentialExpr(add.getRhs(), cache));
  } else if (auto sub = dyn_cast_or_null<arith::SubIOp>(defOp)) {
    result = makeSub(buildSequentialExpr(sub.getLhs(), cache),
                     buildSequentialExpr(sub.getRhs(), cache));
  } else if (auto mul = dyn_cast_or_null<arith::MulIOp>(defOp)) {
    if (auto lhsConst = getConstantIntValue(mul.getLhs()))
      result = makeMul(makeConst(*lhsConst),
                       buildSequentialExpr(mul.getRhs(), cache));
    else if (auto rhsConst = getConstantIntValue(mul.getRhs()))
      result = makeMul(buildSequentialExpr(mul.getLhs(), cache),
                       makeConst(*rhsConst));
    else
      result = makeLeaf(value);
  } else if (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp,
                             arith::ExtSIOp, arith::ExtUIOp>(defOp)) {
    result = makeCast(defOp, buildSequentialExpr(defOp->getOperand(0), cache));
  } else {
    result = makeLeaf(value);
  }
  cache[value] = result;
  return result;
}

struct NormalizedBase {
  Value root;
  StrideExprRef offset; // in pto.addptr element units
};

// Strip a same-element-unit pto.addptr chain into (root, accumulated offset).
static NormalizedBase normalizeSequentialBase(Value base, int64_t elemBytes,
                                              SequentialExprCache &cache) {
  Value root = base;
  StrideExprRef offset = makeConst(0);
  while (auto addPtr = root.getDefiningOp<pto::AddPtrOp>()) {
    auto parentElemBytes = addPtrUnitBytes(addPtr.getPtr());
    if (!parentElemBytes || *parentElemBytes != elemBytes)
      break;
    offset = makeAdd(offset, buildSequentialExpr(addPtr.getOffset(), cache));
    root = addPtr.getPtr();
  }
  return {root, offset};
}

struct SequentialCandidate {
  Operation *op;
  const PostUpdateOpInfo *info;
  Value base;
  Value strideOperand;
  Value rootBase;
  StrideExprRef baseOffset;
  StrideExprRef strideExpr;
  int64_t elemBytes;
  int64_t unitBytes;
};

struct SequentialBucket {
  StringRef opName;
  Value rootBase;
  SmallVector<SequentialCandidate> candidates;
};

struct SequentialStep {
  StrideExprRef expr;
  AffineForm form;
};

static std::optional<SequentialStep>
analyzeSequentialStep(const SequentialCandidate &previous,
                      const SequentialCandidate &current) {
  if (previous.elemBytes != current.elemBytes ||
      previous.unitBytes != current.unitBytes)
    return std::nullopt;

  StrideExprRef deltaBase = makeSub(current.baseOffset, previous.baseOffset);
  StrideExprRef deltaStride =
      current.info->strideParticipatesInCurrentAddress
          ? makeSub(current.strideExpr, previous.strideExpr)
          : makeConst(0);
  StrideExprRef step = combineStride(deltaBase, deltaStride, current.elemBytes,
                                     current.unitBytes);
  if (!step)
    return std::nullopt;
  auto form = normalizeAffine(step);
  if (!form || isZeroAffineForm(*form))
    return std::nullopt;
  return SequentialStep{affineFormToExpr(*form), std::move(*form)};
}

struct SequentialRun {
  SmallVector<SequentialCandidate *> candidates;
  StrideExprRef step;
  AffineForm stepForm;
  Type strideType;
  Value strideValue;
  Value zeroStride;
  Value currentPtr;
};

static bool validateSequentialRun(SequentialRun &run,
                                  DominanceInfo &dominance) {
  if (run.candidates.size() < 3)
    return false;

  SequentialCandidate *first = run.candidates.front();
  run.strideType = first->strideOperand
                       ? first->strideOperand.getType()
                       : IndexType::get(first->op->getContext());
  if (!canMaterializeAs(run.step, run.strideType) ||
      !constantsFitType(run.step, run.strideType, first->info->strideDomain) ||
      !satisfiesStrideConstraint(run.step, first->info->strideConstraint) ||
      (first->info->strideParticipatesInCurrentAddress &&
       !canScaleInitialOffset(first->strideOperand, first->elemBytes,
                              first->unitBytes)))
    return false;

  if (first->info->strideOperandIdx &&
      !first->info->strideParticipatesInCurrentAddress) {
    auto stepForm = normalizeAffine(run.step);
    if (!stepForm)
      return false;
    for (SequentialCandidate *candidate : llvm::drop_end(run.candidates)) {
      auto strideForm = normalizeAffine(candidate->strideExpr);
      if (!strideForm || !equalAffineForms(*stepForm, *strideForm))
        return false;
    }
  }

  for (SequentialCandidate *candidate : run.candidates) {
    Type candidateStrideType =
        candidate->strideOperand ? candidate->strideOperand.getType()
                                 : IndexType::get(candidate->op->getContext());
    if (candidateStrideType != run.strideType)
      return false;
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

static bool isDynamicSequentialValue(Value value, SequentialExprCache &cache) {
  auto form = normalizeAffine(buildSequentialExpr(value, cache));
  return form && !form->terms.empty();
}

// Count only addptrs that are guaranteed to disappear after the candidates
// following the run head are rewritten. The first candidate's base chain is
// retained to construct the initial pointer and therefore is not a saving.
static unsigned countDeadDynamicAddPtrs(const SequentialRun &run) {
  DenseSet<Operation *> counted;
  SequentialExprCache cache;
  for (SequentialCandidate *candidate : llvm::drop_begin(run.candidates)) {
    Value value = candidate->base;
    Operation *expectedUser = candidate->op;
    while (auto addPtr = value.getDefiningOp<pto::AddPtrOp>()) {
      if (!hasOnlyExpectedUser(value, expectedUser))
        break;
      if (isDynamicSequentialValue(addPtr.getOffset(), cache))
        counted.insert(addPtr);
      expectedUser = addPtr;
      value = addPtr.getPtr();
    }
  }
  return counted.size();
}

static unsigned initialPointerCost(const SequentialRun &run) {
  if (!run.candidates.front()->info->strideParticipatesInCurrentAddress)
    return 0;
  if (!run.candidates.front()->strideOperand)
    return 0;
  auto initialOffset =
      getConstantIntValue(run.candidates.front()->strideOperand);
  return initialOffset && *initialOffset == 0 ? 0 : 1;
}

static bool isRunStrideUse(OpOperand &use, const SequentialRun &run) {
  return llvm::any_of(run.candidates, [&](SequentialCandidate *candidate) {
    return candidate->info->strideParticipatesInCurrentAddress &&
           candidate->info->strideOperandIdx &&
           use.getOwner() == candidate->op &&
           use.getOperandNumber() == *candidate->info->strideOperandIdx;
  });
}

// Collect the cumulative add/sub chain used to form the third and later
// offsets of a direct symbolic-leaf run. Unsupported producers are the
// symbolic leaves at which this slice stops.
static void collectCumulativeOffsetOps(Value value,
                                       DenseSet<Operation *> &ops) {
  Operation *defOp = value.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp>(defOp) ||
      !ops.insert(defOp).second)
    return;
  for (Value operand : defOp->getOperands())
    collectCumulativeOffsetOps(operand, ops);
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

static bool
cumulativeOffsetChainDefinitelyDies(const SequentialRun &run,
                                    DenseSet<Operation *> &deadOps) {
  for (SequentialCandidate *candidate : llvm::drop_begin(run.candidates, 2)) {
    if (candidate->strideOperand)
      collectCumulativeOffsetOps(candidate->strideOperand, deadOps);
  }
  return !deadOps.empty() && llvm::all_of(deadOps, [&](Operation *op) {
    return allUsesDisappearAfterRewrite(op, deadOps, run);
  });
}

static bool collectLatePureDefinitions(Value value, Operation *runHead,
                                       DominanceInfo &dominance,
                                       DenseSet<Operation *> &clonedOps) {
  if (dominance.dominates(value, runHead))
    return true;
  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != runHead->getBlock() || !isPure(defOp))
    return false;
  if (!clonedOps.insert(defOp).second)
    return true;
  return llvm::all_of(defOp->getOperands(), [&](Value operand) {
    return collectLatePureDefinitions(operand, runHead, dominance, clonedOps);
  });
}

// Direct symbolic steps are either reused at the run head or cloned there.
// Cloning is cost-neutral only when the original pure definition chain becomes
// dead after the address operands are replaced.
static bool
isStepMaterializationCostNeutral(const SequentialRun &run,
                                 const DenseSet<Operation *> &deadOffsetOps,
                                 DominanceInfo &dominance) {
  SequentialCandidate *first = run.candidates.front();
  DenseSet<Operation *> clonedOps;
  StrideExprRef atom = run.stepForm.terms.front().atom;
  if (atom->kind == StrideExpr::Kind::Cast) {
    clonedOps.insert(atom->castOp);
    SmallVector<Value> leaves;
    collectLeaves(atom, leaves);
    for (Value leaf : leaves)
      if (!collectLatePureDefinitions(leaf, first->op, dominance, clonedOps))
        return false;
  } else if (atom->kind == StrideExpr::Kind::Leaf &&
             !collectLatePureDefinitions(atom->leaf, first->op, dominance,
                                         clonedOps)) {
    return false;
  }

  DenseSet<Operation *> disappearing = deadOffsetOps;
  disappearing.insert(clonedOps.begin(), clonedOps.end());
  return llvm::all_of(clonedOps, [&](Operation *op) {
    return allUsesDisappearAfterRewrite(op, disappearing, run);
  });
}

static bool isProfitableDynamicBaseRun(const SequentialRun &run) {
  if (!run.stepForm.terms.empty())
    return false;
  unsigned pointerCost = run.candidates.size() - 1;
  return countDeadDynamicAddPtrs(run) > pointerCost + initialPointerCost(run);
}

static bool isProfitableDirectSymbolicLeafRun(const SequentialRun &run,
                                              DominanceInfo &dominance) {
  // validateSequentialRun already enforces N >= 3. This class intentionally
  // has no higher length threshold, so an N3 run may be accepted.
  if (run.stepForm.constant != 0 || run.stepForm.terms.size() != 1 ||
      run.stepForm.terms.front().coeff != 1)
    return false;

  // This class covers a direct fixed base with offsets 0, step, 2*step, ...
  // Dynamic base chains are handled independently above.
  if (!llvm::all_of(run.candidates, [](SequentialCandidate *candidate) {
        return candidate->base == candidate->rootBase;
      }))
    return false;
  Value firstStrideOperand = run.candidates.front()->strideOperand;
  auto firstOffset = firstStrideOperand
                         ? getConstantIntValue(firstStrideOperand)
                         : std::optional<int64_t>(0);
  if (!firstOffset || *firstOffset != 0)
    return false;

  DenseSet<Operation *> deadOffsetOps;
  if (!cumulativeOffsetChainDefinitelyDies(run, deadOffsetOps))
    return false;
  return isStepMaterializationCostNeutral(run, deadOffsetOps, dominance);
}

// Profitability is intentionally a structural whitelist rather than a
// weighted MLIR-op cost model. It applies uniformly to every supported
// post-update op: either later candidates delete enough dynamic addptr work, or
// a direct symbolic step replaces a cumulative address chain.
static bool isProfitableSequentialRun(const SequentialRun &run,
                                      DominanceInfo &dominance) {
  return isProfitableDynamicBaseRun(run) ||
         isProfitableDirectSymbolicLeafRun(run, dominance);
}

static void collectNestedBlocks(Operation *op, pto::VecScopeOp owner,
                                SmallVectorImpl<Block *> &blocks) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      blocks.push_back(&block);
      for (Operation &nested : block) {
        if (auto nestedScope = dyn_cast<pto::VecScopeOp>(nested);
            nestedScope && nestedScope != owner)
          continue;
        collectNestedBlocks(&nested, owner, blocks);
      }
    }
  }
}

static void processSequentialBlock(Block *block, DominanceInfo &dominance,
                                   OpBuilder &builder) {
  SmallVector<Operation *> originalOps;
  SmallVector<SequentialBucket> buckets;
  SequentialExprCache exprCache;

  for (Operation &op : *block) {
    originalOps.push_back(&op);
    const PostUpdateOpInfo *info = getPostUpdateInfo(&op);
    if (!info || isAlreadyPostUpdate(&op, *info))
      continue;

    Value base, strideOperand;
    extractBaseAndStrideOperand(&op, *info, base, strideOperand);
    auto elemBytes = addPtrUnitBytes(base);
    if (!elemBytes)
      continue;
    auto unitBytes = strideUnitBytes(&op, *info, *elemBytes);
    if (!unitBytes)
      continue;
    NormalizedBase normalized =
        normalizeSequentialBase(base, *elemBytes, exprCache);

    auto bucketIt = llvm::find_if(buckets, [&](const SequentialBucket &bucket) {
      return bucket.opName == op.getName().getStringRef() &&
             bucket.rootBase == normalized.root;
    });
    if (bucketIt == buckets.end()) {
      buckets.push_back({op.getName().getStringRef(), normalized.root, {}});
      bucketIt = std::prev(buckets.end());
    }
    bucketIt->candidates.push_back(
        {&op, info, base, strideOperand, normalized.root, normalized.offset,
         buildSequentialExpr(strideOperand, exprCache), *elemBytes,
         *unitBytes});
  }

  SmallVector<SequentialRun> runs;
  for (SequentialBucket &bucket : buckets) {
    auto &candidates = bucket.candidates;
    size_t start = 0;
    while (start + 1 < candidates.size()) {
      auto firstStep =
          analyzeSequentialStep(candidates[start], candidates[start + 1]);
      if (!firstStep) {
        ++start;
        continue;
      }

      size_t end = start + 2;
      while (end < candidates.size()) {
        auto nextStep =
            analyzeSequentialStep(candidates[end - 1], candidates[end]);
        if (!nextStep || !equalAffineForms(firstStep->form, nextStep->form))
          break;
        ++end;
      }

      SequentialRun run;
      run.step = firstStep->expr;
      run.stepForm = firstStep->form;
      for (size_t i = start; i < end; ++i)
        run.candidates.push_back(&candidates[i]);
      if (validateSequentialRun(run, dominance) &&
          isProfitableSequentialRun(run, dominance)) {
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

  if (runs.empty())
    return;

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
    run.currentPtr =
        createInitialPtr(first->base, first->strideOperand,
                         first->info->strideParticipatesInCurrentAddress,
                         first->info->addressUnit, first->elemBytes,
                         first->unitBytes, first->op->getLoc(), builder);
  }

  DenseMap<Operation *, unsigned> opToRun;
  for (auto [runIdx, run] : llvm::enumerate(runs))
    for (SequentialCandidate *candidate : run.candidates)
      opToRun[candidate->op] = runIdx;

  // Rewrite in original program order so interleaved buckets maintain separate
  // pointer chains without invalidating one another.
  for (Operation *op : originalOps) {
    auto it = opToRun.find(op);
    if (it == opToRun.end())
      continue;
    SequentialRun &run = runs[it->second];
    const PostUpdateOpInfo *info = getPostUpdateInfo(op);
    builder.setInsertionPoint(op);
    bool isLast = op == run.candidates.back()->op;
    Operation *newOp = isLast ? createNormalOp(op, *info, run.currentPtr,
                                               run.zeroStride, builder)
                              : createPostUpdateOp(op, *info, run.currentPtr,
                                                   run.strideValue, builder);
    for (unsigned result = 0; result < op->getNumResults(); ++result)
      op->getResult(result).replaceAllUsesWith(newOp->getResult(result));
    if (!isLast)
      run.currentPtr = newOp->getResult(newOp->getNumResults() - 1);
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

    module.walk(
        [&](pto::VecScopeOp vecscope) { processVecScope(vecscope, builder); });
  }

private:
  void processVecScope(pto::VecScopeOp vecscope, OpBuilder &builder) {
    // Collect scf.for ops inside this vecscope.  Operation::walk defaults to
    // post-order, so nested loops already come before the loops enclosing
    // them.
    SmallVector<scf::ForOp> forOps;
    vecscope.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    // Process inner-to-outer, i.e. in collection order.  The order is load
    // bearing: rewriting a loop erases it, which also destroys every loop
    // nested inside it.  Visiting an enclosing loop first would leave the
    // already-collected inner ForOp handles dangling.
    for (scf::ForOp forOp : forOps)
      processForOp(forOp, builder);

    eraseDeadPureOps(vecscope);

    // Loop rewriting rebuilds ForOps, so collect blocks only after every loop
    // handle has been consumed. This second phase includes loop bodies and
    // handles only candidates that remain in non-post-update form.
    SmallVector<Block *> blocks;
    collectNestedBlocks(vecscope, vecscope, blocks);
    DominanceInfo dominance(vecscope->getParentOp());
    for (Block *block : blocks)
      processSequentialBlock(block, dominance, builder);
  }

  void processForOp(scf::ForOp forOp, OpBuilder &builder) {
    SmallVector<PostUpdateRewrite> rewrites;
    // Shared across all candidates in this loop so equal strides map to one
    // Value, which is what lets same-address ops share an iter_arg.
    ConstCache constCache;

    for (Operation &op : *forOp.getBody()) {
      const PostUpdateOpInfo *info = getPostUpdateInfo(&op);
      if (!info)
        continue;
      if (isAlreadyPostUpdate(&op, *info))
        continue;
      if (!isDirectlyInForBody(&op, forOp))
        continue;

      Value base, strideOperand;
      extractBaseAndStrideOperand(&op, *info, base, strideOperand);

      if (!isCanonicalLoopBase(base, forOp) ||
          (strideOperand && !forOp.isDefinedOutsideOfLoop(strideOperand) &&
           !isSafeLoopInteger(strideOperand, forOp)))
        continue;

      // Both address terms have to be expressed in the same currency before
      // they can be combined: delta(base) is measured in pto.addptr units
      // (elements), while the strideOperand is counted in whatever unit the
      // op's lowering expects.  Bail on pointers whose addptr unit we cannot
      // pin down rather than guess at the scale.
      std::optional<int64_t> elemBytes = addPtrUnitBytes(base);
      if (!elemBytes)
        continue;
      auto unitBytes = strideUnitBytes(&op, *info, *elemBytes);
      if (!unitBytes)
        continue;

      // Analyze each operand independently: accumulator (iter_arg) first,
      // delta (IV/affine) fallback. Both return a symbolic per-iteration
      // stride; no IR is created until the candidate is known to be viable.
      DeltaCache deltaCache;
      StrideExprRef deltaBase = getStride(base, forOp, deltaCache);
      StrideExprRef deltaOffset =
          strideOperand && info->strideParticipatesInCurrentAddress
              ? getStride(strideOperand, forOp, deltaCache, info->strideDomain)
              : makeConst(0);

      if (!deltaBase || !deltaOffset)
        continue;

      StrideExprRef total;
      if (!strideOperand || info->strideParticipatesInCurrentAddress) {
        total = combineStride(deltaBase, deltaOffset, *elemBytes, *unitBytes);
      } else {
        // Stateful forms such as vstus access the current base directly and
        // use their explicit stride only to advance the returned base. The
        // post chain is valid exactly when that advancement matches the
        // original base recurrence; the operand itself must be preserved.
        StrideExprRef baseAdvance =
            scaleBaseDelta(deltaBase, *elemBytes, *unitBytes);
        auto explicitConstant = getConstantIntValue(strideOperand);
        StrideExprRef explicitAdvance =
            explicitConstant ? makeConst(*explicitConstant)
                             : makeLeaf(strideOperand);
        if (!baseAdvance ||
            !equalAddressAdvanceExprs(baseAdvance, explicitAdvance))
          continue;
        total = explicitAdvance;
      }
      if (!total)
        continue;

      if (!hasOnlySafeLoopIntegerLeaves(total, forOp))
        continue;

      // Reject expressions whose subterms demand conflicting types, or whose
      // dynamic result cannot be materialized exactly as the op's declared
      // stride operand type.
      Type exprResultType;
      Type strideType =
          strideOperand ? strideOperand.getType() : builder.getIndexType();
      total = adaptCanonicalAddressExprType(total, strideType);
      if (!exprType(total, exprResultType))
        continue;
      if (exprResultType && exprResultType != strideType)
        continue;

      // Reject strides whose constants do not fit the target operand type.
      if (!constantsFitType(total, strideType, info->strideDomain))
        continue;
      if (!satisfiesStrideConstraint(total, info->strideConstraint))
        continue;

      // A stride built only from loop-invariant leaves is materialized before
      // the loop; otherwise it goes immediately before the candidate op.
      SmallVector<Value> leaves;
      collectLeaves(total, leaves);
      bool allInvariant = llvm::all_of(
          leaves, [&](Value l) { return forOp.isDefinedOutsideOfLoop(l); });

      StrideExprRef finalExpr = total;
      if (!allInvariant) {
        // Every leaf must be usable at the candidate op.  Checked before any
        // IR is created, so a rejected candidate leaves nothing behind.
        DenseMap<Value, bool> canCache;
        if (!llvm::all_of(leaves, [&](Value l) {
              return canHoistBefore(l, &op, forOp, canCache);
            }))
          continue;
        DenseMap<Value, Value> hoistMemo;
        finalExpr = makeAvailableAt(total, &op, forOp, builder, hoistMemo);
      }

      builder.setInsertionPoint(allInvariant ? forOp.getOperation() : &op);
      Value strideNew = materialize(finalExpr, strideType, op.getLoc(), forOp,
                                    constCache, builder);

      Value initPtr = computeInitialPtr(
          base, strideOperand, info->strideParticipatesInCurrentAddress,
          info->addressUnit, *elemBytes, *unitBytes, forOp, builder);
      if (!initPtr)
        continue;
      rewrites.push_back(
          {&op, base, strideOperand, strideNew, initPtr, *unitBytes});
    }

    if (!rewrites.empty()) {
      applyPostUpdateRewrites(forOp, rewrites, builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSoftPostUpdatePass() {
  return std::make_unique<VPTOSoftPostUpdatePass>();
}
