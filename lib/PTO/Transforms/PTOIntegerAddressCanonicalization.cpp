// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOIntegerAddressCanonicalization.cpp -----------------------------===//
//
// Canonicalizes integer-backed `pto.castptr` into the canonical pointer form
//
//   castptr(byte_address)  ->  addptr(castptr(R), Q)
//
// where `R` is the canonical root (constant 0, or a single non-divisible atom
// leaf such as a runtime base-address kernel parameter) and `Q` is the exact
// element quotient `(B - R) / sizeof(element)`. The rewrite only matches
// zero-origin integral address spaces (A5 UB) and requires a non-trivial
// quotient so it converges to the normal form in one pass (constant-zero and
// pure-atom inputs stay untouched).
//
// This is the "integer address canonicalization" rule of
// docs/designs/vpto-integer-address-canonicalization-design-zh.md. The addptr
// absorption fold into memory-op offsets lives in PTOAbsorbAddPtr.cpp.
//
//===----------------------------------------------------------------------===//

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/Analysis/PTOValueEvolutionAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINTEGERADDRESSCANONICALIZATION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Current PTO targets fix index at 64 bits (PTOValueEvolutionAnalysis uses the
// same assumption); the round-trip proof below requires the input width to
// equal this width so CastIndex is a same-width conversion.
static constexpr unsigned kIndexBitWidth = 64;

/// Convert a signedness-carrying integer to the equivalent signless carrier
/// required by arith operations. Reuse the source of an existing no-op bridge
/// when possible so canonicalization does not grow redundant casts.
static Value getSignlessIntegerCarrier(PatternRewriter &rewriter, Location loc,
                                       Value value) {
  auto integerType = dyn_cast<IntegerType>(value.getType());
  if (!integerType || integerType.isSignless()) {
    return value;
  }
  auto carrierType = rewriter.getIntegerType(integerType.getWidth());
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1 &&
        cast.getInputs().front().getType() == carrierType) {
      return cast.getInputs().front();
    }
  }
  return rewriter
      .create<UnrealizedConversionCastOp>(loc, carrierType, value)
      .getResult(0);
}

/// Rebuild a cast with `kind` into `targetType`. Same-type casts are skipped
/// so absorbed index extensions do not leave a redundant index_cast behind.
/// arith's ext/si/trunc ops require fixed-width targets, so an index target
/// is reached through index_cast/index_castui (which preserve the same
/// zero/sign extension semantics).
static Value createCastToTarget(PatternRewriter &rewriter, Location loc,
                                PTOCastKind kind, Value input,
                                Type targetType) {
  if (input.getType() == targetType) {
    return input;
  }
  if (targetType.isIndex()) {
    switch (kind) {
    case PTOCastKind::IndexCast:
    case PTOCastKind::ExtSI:
      return rewriter.create<arith::IndexCastOp>(loc, targetType, input);
    case PTOCastKind::IndexCastUI:
    case PTOCastKind::ExtUI:
      return rewriter.create<arith::IndexCastUIOp>(loc, targetType, input);
    case PTOCastKind::TruncI:
      // Truncation into index is ill-defined; refuse.
      return {};
    }
  }
  switch (kind) {
  case PTOCastKind::IndexCast:
    return rewriter.create<arith::IndexCastOp>(loc, targetType, input);
  case PTOCastKind::IndexCastUI:
    return rewriter.create<arith::IndexCastUIOp>(loc, targetType, input);
  case PTOCastKind::ExtSI:
    return rewriter.create<arith::ExtSIOp>(loc, targetType, input);
  case PTOCastKind::ExtUI:
    return rewriter.create<arith::ExtUIOp>(loc, targetType, input);
  case PTOCastKind::TruncI:
    return rewriter.create<arith::TruncIOp>(loc, targetType, input);
  }
  return {};
}

/// Materialize a synthetic typed expression as SSA values directly in
/// `targetType` (the index domain). Leaves (nodes carrying a sourceValue) are
/// reused verbatim when their type already matches; an index<->int extension
/// cast leaf is rebuilt from its cast input so the emitted offset stays a
/// direct affine index expression (the post-update consumer needs that shape).
/// Only synthetic constant/add/sub/mul nodes built by buildPTOTypedExpr are
/// reconstructed.
static Value materializeQuotient(PatternRewriter &rewriter, Location loc,
                                 const PTOTypedExprRef &expr,
                                 Type targetType) {
  if (!expr) {
    return {};
  }
  if (expr->sourceValue) {
    Value src = expr->sourceValue;
    if (src.getType() == targetType) {
      return src;
    }
    if (expr->kind == PTOTypedExpr::Kind::Cast) {
      // Rebuild the cast into the target type from its input, materializing
      // the input in its own type first (preserves zero/sign semantics).
      Type innerType = expr->lhs ? expr->lhs->type : Type();
      Value inner = innerType ? materializeQuotient(rewriter, loc, expr->lhs,
                                                    innerType)
                              : Value();
      if (!inner) {
        return {};
      }
      return createCastToTarget(rewriter, loc, expr->castKind, inner,
                                targetType);
    }
    // Plain integer leaf (e.g. an i64 atom inside the quotient): same-width
    // conversion into index keeps the bit pattern.
    if (src.getType().isIntOrIndex() && targetType.isIndex() &&
        src.getType().getIntOrFloatBitWidth() == kIndexBitWidth) {
      src = getSignlessIntegerCarrier(rewriter, loc, src);
      return rewriter.create<arith::IndexCastOp>(loc, targetType, src);
    }
    return {};
  }
  switch (expr->kind) {
  case PTOTypedExpr::Kind::Constant:
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(targetType, expr->constant));
  case PTOTypedExpr::Kind::Add: {
    Value lhs = materializeQuotient(rewriter, loc, expr->lhs, targetType);
    Value rhs = materializeQuotient(rewriter, loc, expr->rhs, targetType);
    if (!lhs || !rhs) {
      return {};
    }
    return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
  }
  case PTOTypedExpr::Kind::Sub: {
    Value lhs = materializeQuotient(rewriter, loc, expr->lhs, targetType);
    Value rhs = materializeQuotient(rewriter, loc, expr->rhs, targetType);
    if (!lhs || !rhs) {
      return {};
    }
    return rewriter.create<arith::SubIOp>(loc, lhs, rhs);
  }
  case PTOTypedExpr::Kind::Mul: {
    Value lhs = materializeQuotient(rewriter, loc, expr->lhs, targetType);
    Value rhs = materializeQuotient(rewriter, loc, expr->rhs, targetType);
    if (!lhs || !rhs) {
      return {};
    }
    return rewriter.create<arith::MulIOp>(loc, lhs, rhs);
  }
  case PTOTypedExpr::Kind::Cast:
  case PTOTypedExpr::Kind::Opaque:
    // buildPTOTypedExpr never emits cast/opaque nodes; a source-backed node
    // would have been returned above.
    return {};
  }
  return {};
}

/// Strip the sourceValue from structural (add/sub/mul) nodes so
/// normalizePTOLinearExpr expands them instead of treating the whole
/// expression as one opaque atom. Cast/opaque/constant leaves keep their
/// sourceValue and stay atomic, matching the design's cast-as-atom rule.
static PTOTypedExprRef stripStructuralSourceValues(const PTOTypedExprRef &expr) {
  if (!expr) {
    return expr;
  }
  switch (expr->kind) {
  case PTOTypedExpr::Kind::Add:
    return makePTOAddExpr(stripStructuralSourceValues(expr->lhs),
                          stripStructuralSourceValues(expr->rhs), expr->type);
  case PTOTypedExpr::Kind::Sub:
    return makePTOSubExpr(stripStructuralSourceValues(expr->lhs),
                          stripStructuralSourceValues(expr->rhs), expr->type);
  case PTOTypedExpr::Kind::Mul:
    return makePTOMulExpr(stripStructuralSourceValues(expr->lhs),
                          stripStructuralSourceValues(expr->rhs), expr->type);
  default:
    // Constant / Cast / Opaque stay atomic (and keep their sourceValue).
    return expr;
  }
}

struct CanonicalizeIntegerCastPtr final
    : public OpRewritePattern<pto::CastPtrOp> {
  CanonicalizeIntegerCastPtr(MLIRContext *context,
                             PTOValueEvolutionAnalysis &valueEvolution)
      : OpRewritePattern<pto::CastPtrOp>(context),
        valueEvolution(valueEvolution) {}

  LogicalResult matchAndRewrite(pto::CastPtrOp castOp,
                                PatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<pto::PtrType>(castOp.getResult().getType());
    if (!ptrType) {
      return rewriter.notifyMatchFailure(castOp, "result is not a ptr type");
    }
    // Only zero-origin integral address spaces are eligible (A5 UB prints as
    // "ub" and maps to AddressSpace::VEC).
    if (ptrType.getMemorySpace().getAddressSpace() != pto::AddressSpace::VEC) {
      return rewriter.notifyMatchFailure(castOp, "not a zero-origin space");
    }
    Value input = castOp.getInput();
    Type inputType = input.getType();
    if (!inputType.isIntOrIndex()) {
      return rewriter.notifyMatchFailure(castOp, "input is not an integer");
    }
    unsigned inputWidth =
        inputType.isIndex() ? kIndexBitWidth
                            : inputType.getIntOrFloatBitWidth();
    // Design C14 requires the *quotient* to round-trip losslessly into index.
    // The first implementation approximates this with the sufficient condition
    // inputWidth == index width (same-width CastIndex is trivially lossless).
    // Narrower inputs (e.g. i32) are conservatively rejected until PTO defines
    // the zero/sign extension semantics of castptr into the 64-bit address
    // space; the full quotient round-trip proof is future work.
    if (inputWidth != kIndexBitWidth) {
      return rewriter.notifyMatchFailure(
          castOp, "only 64-bit integer inputs are supported (first version)");
    }

    Type elementType = ptrType.getElementType();
    std::optional<int64_t> elementBytes = std::nullopt;
    if (elementType && elementType.isIntOrFloat()) {
      unsigned bitWidth = elementType.getIntOrFloatBitWidth();
      if (bitWidth != 0 && bitWidth % 8 == 0) {
        elementBytes = static_cast<int64_t>(bitWidth / 8);
      }
    }
    if (!elementBytes || *elementBytes <= 0) {
      return rewriter.notifyMatchFailure(castOp, "unknown element size");
    }

    auto linear = normalizePTOLinearExpr(
        stripStructuralSourceValues(valueEvolution.getExpr(input)));
    if (!linear) {
      return rewriter.notifyMatchFailure(castOp, "input is not linear");
    }

    // Pointer-derived integer leaves (ptr-to-int castptr / ptrtoint results)
    // need a separate provenance contract and stay untouched (design C13).
    for (const PTOLinearTerm &term : linear->terms) {
      if (!term.atom || !term.atom->sourceValue) {
        continue;
      }
      Operation *leafDef = term.atom->sourceValue.getDefiningOp();
      if (isa_and_nonnull<pto::PtrToIntOp>(leafDef)) {
        return rewriter.notifyMatchFailure(castOp,
                                           "pointer-derived integer leaf");
      }
      if (auto ptrCast = dyn_cast_or_null<pto::CastPtrOp>(leafDef)) {
        if (!isa<pto::PtrType>(ptrCast.getResult().getType())) {
          return rewriter.notifyMatchFailure(
              castOp, "pointer-derived integer leaf");
        }
      }
    }

    // Canonical-root selection: a coefficient not divisible by the element
    // size cannot move into the offset. Only a single unit-coefficient atom
    // leaf is supported as the root; everything else is rejected.
    Value rootInteger = nullptr;
    PTOLinearExpr quotient = *linear;
    SmallVector<PTOLinearTerm> nonDivisible;
    for (const PTOLinearTerm &term : quotient.terms) {
      if (term.coefficient % *elementBytes != 0) {
        nonDivisible.push_back(term);
      }
    }
    if (!nonDivisible.empty()) {
      if (nonDivisible.size() > 1) {
        return rewriter.notifyMatchFailure(
            castOp, "multiple non-divisible atoms have no unique root");
      }
      const PTOLinearTerm &rootTerm = nonDivisible.front();
      // Design contract: only a unit-coefficient (+1) atom is the canonical
      // root; negative or scaled atoms have no unique root/offset split.
      if (rootTerm.coefficient != 1) {
        return rewriter.notifyMatchFailure(
            castOp, "non-unit-coefficient atom has no root/offset split");
      }
      if (!rootTerm.atom || !rootTerm.atom->sourceValue ||
          !rootTerm.atom->sourceValue.getType().isIntOrIndex()) {
        return rewriter.notifyMatchFailure(
            castOp, "atom leaf is not a materializable integer");
      }
      rootInteger = rootTerm.atom->sourceValue;
      Value rootLeaf = rootInteger;
      llvm::erase_if(quotient.terms, [&](const PTOLinearTerm &term) {
        return term.coefficient == rootTerm.coefficient &&
               term.atom && term.atom->sourceValue == rootLeaf;
      });
    }

    if (!dividePTOLinearExprExact(quotient, *elementBytes)) {
      return rewriter.notifyMatchFailure(castOp, "no exact element quotient");
    }
    if (isZeroPTOLinearExpr(quotient)) {
      return rewriter.notifyMatchFailure(
          castOp, "quotient is trivial (already canonical)");
    }

    // Round-trip proof: inputWidth == index width makes CastIndex a
    // same-width conversion, and every quotient leaf comes from the input
    // expression, so `Q -> index -> element scaling` preserves the address
    // bit pattern (mod 2^Waddr) because Q * E == B - R is an exact identity.
    // The quotient is materialized directly in the index domain so the
    // emitted offset is a direct affine index expression that the post-update
    // consumer can analyze.
    PTOTypedExprRef quotientExpr = buildPTOTypedExpr(quotient, inputType);
    rewriter.setInsertionPoint(castOp);
    Value indexValue = materializeQuotient(
        rewriter, castOp.getLoc(), quotientExpr, rewriter.getIndexType());
    if (!indexValue) {
      return rewriter.notifyMatchFailure(castOp, "cannot materialize quotient");
    }

    // The canonical root (castptr(0) or castptr(%atom)) is hoisted above any
    // enclosing scf.for so the post-update consumer sees a base defined
    // outside the loop (VPTOSoftPostUpdate requires that). Hoisting is only
    // legal when the root value itself is loop-invariant; an atom-root defined
    // inside the loop must stay in place or the hoisted castptr would violate
    // SSA dominance.
    Operation *anchor = castOp.getOperation();
    bool rootIsLoopInvariant = true;
    while (scf::ForOp forOp = anchor->getParentOfType<scf::ForOp>()) {
      if (rootInteger && !forOp.isDefinedOutsideOfLoop(rootInteger)) {
        rootIsLoopInvariant = false;
      }
      anchor = forOp.getOperation();
    }
    rewriter.setInsertionPoint(rootIsLoopInvariant ? anchor
                                                   : castOp.getOperation());
    Value rootValue = rootInteger;
    if (!rootValue) {
      Type rootType = inputType;
      if (auto integerType = dyn_cast<IntegerType>(inputType)) {
        rootType = rewriter.getIntegerType(integerType.getWidth());
      }
      rootValue = rewriter.create<arith::ConstantOp>(
          castOp.getLoc(), rewriter.getIntegerAttr(rootType, 0));
    }
    Value base =
        rewriter.create<pto::CastPtrOp>(castOp.getLoc(), ptrType, rootValue);

    rewriter.setInsertionPoint(castOp);
    Value canonical = rewriter.create<pto::AddPtrOp>(castOp.getLoc(), ptrType,
                                                     base, indexValue);
    rewriter.replaceOp(castOp, canonical);
    return success();
  }

  PTOValueEvolutionAnalysis &valueEvolution;
};

struct PTOIntegerAddressCanonicalizationPass final
    : public pto::impl::PTOIntegerAddressCanonicalizationBase<
          PTOIntegerAddressCanonicalizationPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto &valueEvolution = getAnalysis<pto::PTOValueEvolutionAnalysis>();
    RewritePatternSet patterns(&getContext());
    patterns.add<CanonicalizeIntegerCastPtr>(&getContext(), valueEvolution);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOIntegerAddressCanonicalizationPass() {
  return std::make_unique<PTOIntegerAddressCanonicalizationPass>();
}
