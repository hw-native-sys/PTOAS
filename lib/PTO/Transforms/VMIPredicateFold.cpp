// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIPredicateFold.cpp - Fold statically proven VMI predicates -------===//
//
 // Predicate-algebra simplifier for unified VMI IR:
 //  * prove AllTrue / AllFalse masks (create_mask, vcmp ranges, mask algebra)
 //  * fold vsel / select
 //  * demask AllTrue consumers (Variadic-mask compute)
 //  * fold AllFalse pure consumers (vdhist → acc, merge → passthru)
 //  * fold first-iter neutral splat peeps (vmax(-inf,x) / vadd(0,x))
 //  * materialize proven masks to create_mask(VL|0)
 //  * DCE pure unused defs
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIMaskUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIPREDICATEFOLD
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<int64_t> matchConstantInt(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

static StringRef getPmodeOrDefault(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("pmode"))
    return attr.getValue();
  return "merge";
}

static bool isZeroPmode(Operation *op) { return getPmodeOrDefault(op) == "zero"; }

/// Match a splat constant vector: vbrc/broadcast of Imm, or dense constant.
enum class SplatKind { None, Zero, NegInf, PosInf, Other };

static SplatKind classifySplat(Value v, Type *elemTyOut = nullptr) {
  auto vty = dyn_cast<VMIVRegType>(v.getType());
  if (!vty)
    return SplatKind::None;
  if (elemTyOut)
    *elemTyOut = vty.getElementType();

  auto fromAPFloat = [&](const APFloat &f) -> SplatKind {
    if (f.isZero())
      return SplatKind::Zero;
    if (f.isInfinity())
      return f.isNegative() ? SplatKind::NegInf : SplatKind::PosInf;
    return SplatKind::Other;
  };
  auto fromAPInt = [&](const APInt &i) -> SplatKind {
    if (i.isZero())
      return SplatKind::Zero;
    return SplatKind::Other;
  };

  auto fromScalar = [&](Value s) -> SplatKind {
    if (auto c = matchConstantInt(s))
      return *c == 0 ? SplatKind::Zero : SplatKind::Other;
    Attribute attr;
    if (!matchPattern(s, m_Constant(&attr)))
      return SplatKind::None;
    if (auto fa = dyn_cast<FloatAttr>(attr))
      return fromAPFloat(fa.getValue());
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return fromAPInt(ia.getValue());
    return SplatKind::None;
  };

  if (auto brc = v.getDefiningOp<VMIVbrcOp>())
    return fromScalar(brc.getValue());
  if (auto brc = v.getDefiningOp<VMIBroadcastOp>())
    return fromScalar(brc.getValue());
  if (auto cst = v.getDefiningOp<VMIConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense || dense.getNumElements() == 0)
      return SplatKind::None;
    auto fvals = dense.tryGetValues<APFloat>();
    if (succeeded(fvals)) {
      auto it = fvals->begin();
      APFloat first = *it;
      for (APFloat x : *fvals)
        if (!x.bitwiseIsEqual(first))
          return SplatKind::None;
      return fromAPFloat(first);
    }
    auto ivals = dense.tryGetValues<APInt>();
    if (succeeded(ivals)) {
      auto it = ivals->begin();
      APInt first = *it;
      for (APInt x : *ivals)
        if (x != first)
          return SplatKind::None;
      return fromAPInt(first);
    }
  }
  return SplatKind::None;
}

static Value createSplatConstant(OpBuilder &builder, Location loc,
                                 VMIVRegType vty, SplatKind kind) {
  Type elem = vty.getElementType();
  int64_t lanes = vty.getElementCount();
  auto shaped = RankedTensorType::get({lanes}, elem);
  DenseElementsAttr attr;
  if (auto floatTy = dyn_cast<FloatType>(elem)) {
    APFloat val = APFloat::getZero(floatTy.getFloatSemantics());
    if (kind == SplatKind::NegInf)
      val = APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/true);
    else if (kind == SplatKind::PosInf)
      val = APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/false);
    attr = DenseElementsAttr::get(shaped, val);
  } else {
    auto intTy = cast<IntegerType>(elem);
    APInt val = APInt::getZero(intTy.getWidth());
    if (kind == SplatKind::NegInf)
      val = APInt::getSignedMinValue(intTy.getWidth());
    else if (kind == SplatKind::PosInf)
      val = APInt::getSignedMaxValue(intTy.getWidth());
    attr = DenseElementsAttr::get(shaped, val);
  }
  return builder.create<VMIConstantOp>(loc, vty, attr).getResult();
}

static Value materializeCanonicalMask(OpBuilder &builder, Location loc,
                                      VMIMaskType maskTy, MaskLattice lat) {
  int64_t lanes = maskTy.getElementCount();
  int64_t active = lat == MaskLattice::AllTrue ? lanes : 0;
  Value n = builder.create<arith::ConstantIndexOp>(loc, active);
  return builder.create<VMICreateMaskOp>(loc, maskTy, n).getResult();
}

static bool isTriviallyDeadPureOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<func::FuncOp, ModuleOp, scf::ForOp, scf::IfOp, scf::WhileOp,
          scf::YieldOp, scf::ConditionOp>(op))
    return false;
  if (!isMemoryEffectFree(op))
    return false;
  return llvm::all_of(op->getResults(),
                      [](Value r) { return r.use_empty(); });
}

static void dcePureUnusedOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (isTriviallyDeadPureOp(op))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewrites
//===----------------------------------------------------------------------===//

static bool foldSelectLike(Value mask, Value t, Value f, Value result,
                           Operation *op) {
  if (t == f) {
    result.replaceAllUsesWith(t);
    op->erase();
    return true;
  }
  MaskLattice lat = classifyMaskValue(mask);
  if (lat == MaskLattice::Unknown)
    return false;
  // Default vsel pmode is merge-like (false arm). Explicit zero → 0 splat.
  if (lat == MaskLattice::AllTrue) {
    result.replaceAllUsesWith(t);
  } else if (isZeroPmode(op)) {
    OpBuilder b(op);
    auto vty = cast<VMIVRegType>(result.getType());
    result.replaceAllUsesWith(
        createSplatConstant(b, op->getLoc(), vty, SplatKind::Zero));
  } else {
    result.replaceAllUsesWith(f);
  }
  op->erase();
  return true;
}

static bool foldNeutralBinary(Value lhs, Value rhs, Value result, Operation *op,
                              SplatKind identityOnLhs, SplatKind identityOnRhs) {
  // vmax(neg_inf, x) / vmax(x, neg_inf) → x; vadd(0, x) → x; etc.
  if (classifySplat(lhs) == identityOnLhs) {
    result.replaceAllUsesWith(rhs);
    op->erase();
    return true;
  }
  if (classifySplat(rhs) == identityOnRhs) {
    result.replaceAllUsesWith(lhs);
    op->erase();
    return true;
  }
  return false;
}

template <typename OpTy>
static bool demaskBinaryAllTrue(OpTy op) {
  if (op.getMask().empty())
    return false;
  if (classifyMaskValue(op.getMask().front()) != MaskLattice::AllTrue)
    return false;
  OpBuilder b(op);
  auto neu =
      b.create<OpTy>(op.getLoc(), op.getResult().getType(), op.getLhs(),
                     op.getRhs(), ValueRange{}, op.getPmodeAttr());
  op.getResult().replaceAllUsesWith(neu.getResult());
  op.erase();
  return true;
}

template <typename OpTy>
static bool demaskUnaryAllTrue(OpTy op) {
  if (op.getMask().empty())
    return false;
  if (classifyMaskValue(op.getMask().front()) != MaskLattice::AllTrue)
    return false;
  OpBuilder b(op);
  auto neu =
      b.create<OpTy>(op.getLoc(), op.getResult().getType(), op.getSource(),
                     ValueRange{}, op.getPmodeAttr());
  op.getResult().replaceAllUsesWith(neu.getResult());
  op.erase();
  return true;
}

template <typename OpTy>
static bool foldBinaryAllFalse(OpTy op) {
  if (op.getMask().empty())
    return false;
  if (classifyMaskValue(op.getMask().front()) != MaskLattice::AllFalse)
    return false;
  OpBuilder b(op);
  auto vty = cast<VMIVRegType>(op.getResult().getType());
  if (isZeroPmode(op)) {
    op.getResult().replaceAllUsesWith(
        createSplatConstant(b, op.getLoc(), vty, SplatKind::Zero));
  } else {
    // merge (default): inactive lanes pass lhs / source convention → lhs
    op.getResult().replaceAllUsesWith(op.getLhs());
  }
  op.erase();
  return true;
}

template <typename OpTy>
static bool foldUnaryAllFalse(OpTy op) {
  if (op.getMask().empty())
    return false;
  if (classifyMaskValue(op.getMask().front()) != MaskLattice::AllFalse)
    return false;
  OpBuilder b(op);
  auto vty = cast<VMIVRegType>(op.getResult().getType());
  if (isZeroPmode(op)) {
    op.getResult().replaceAllUsesWith(
        createSplatConstant(b, op.getLoc(), vty, SplatKind::Zero));
  } else {
    op.getResult().replaceAllUsesWith(op.getSource());
  }
  op.erase();
  return true;
}

static bool foldMaskAlgebra(Value result, Operation *op) {
  MaskLattice lat = classifyMaskValue(result);
  if (lat == MaskLattice::Unknown)
    return false;
  // Only rewrite pure mask producers that are not already canonical.
  if (isa<VMICreateMaskOp, VMIPsetOp, VMIConstantMaskOp>(op))
    return false;
  OpBuilder b(op);
  auto maskTy = cast<VMIMaskType>(result.getType());
  Value canon = materializeCanonicalMask(b, op->getLoc(), maskTy, lat);
  result.replaceAllUsesWith(canon);
  op->erase();
  return true;
}

static bool foldHistAllFalse(Value acc, Value mask, Value result,
                             Operation *op) {
  if (classifyMaskValue(mask) != MaskLattice::AllFalse)
    return false;
  result.replaceAllUsesWith(acc);
  op->erase();
  return true;
}

static bool foldReduceAllFalse(Value mask, Value result, Operation *op,
                               SplatKind nilKind) {
  if (classifyMaskValue(mask) != MaskLattice::AllFalse)
    return false;
  OpBuilder b(op);
  auto vty = cast<VMIVRegType>(result.getType());
  result.replaceAllUsesWith(createSplatConstant(b, op->getLoc(), vty, nilKind));
  op->erase();
  return true;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct VMIPredicateFoldPass
    : public mlir::pto::impl::VMIPredicateFoldBase<VMIPredicateFoldPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIPredicateFoldPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = true;
    while (changed) {
      changed = false;

      // 1) vsel / select
      SmallVector<Operation *> sels;
      module.walk([&](Operation *op) {
        if (isa<VMIvSelOp, VMISelectOp>(op))
          sels.push_back(op);
      });
      for (Operation *op : llvm::reverse(sels)) {
        if (!op->getBlock())
          continue;
        if (auto sel = dyn_cast<VMIvSelOp>(op)) {
          changed |= foldSelectLike(sel.getMask(), sel.getTrueValue(),
                                    sel.getFalseValue(), sel.getResult(), op);
        } else if (auto sel = dyn_cast<VMISelectOp>(op)) {
          changed |= foldSelectLike(sel.getMask(), sel.getTrueValue(),
                                    sel.getFalseValue(), sel.getResult(), op);
        }
      }

      // 2) Neutral splat peeps on unrolled accumulators
      SmallVector<Operation *> binOps;
      module.walk([&](Operation *op) {
        if (isa<VMIVmaxOp, VMIVminOp, VMIVaddOp>(op))
          binOps.push_back(op);
      });
      for (Operation *op : llvm::reverse(binOps)) {
        if (!op->getBlock())
          continue;
        if (auto vmax = dyn_cast<VMIVmaxOp>(op)) {
          // Only fold when mask absent or AllTrue (identity under full lanes).
          if (!vmax.getMask().empty() &&
              classifyMaskValue(vmax.getMask().front()) != MaskLattice::AllTrue)
            continue;
          changed |= foldNeutralBinary(vmax.getLhs(), vmax.getRhs(),
                                       vmax.getResult(), op, SplatKind::NegInf,
                                       SplatKind::NegInf);
        } else if (auto vmin = dyn_cast<VMIVminOp>(op)) {
          if (!vmin.getMask().empty() &&
              classifyMaskValue(vmin.getMask().front()) != MaskLattice::AllTrue)
            continue;
          changed |= foldNeutralBinary(vmin.getLhs(), vmin.getRhs(),
                                       vmin.getResult(), op, SplatKind::PosInf,
                                       SplatKind::PosInf);
        } else if (auto vadd = dyn_cast<VMIVaddOp>(op)) {
          if (!vadd.getMask().empty() &&
              classifyMaskValue(vadd.getMask().front()) != MaskLattice::AllTrue)
            continue;
          changed |= foldNeutralBinary(vadd.getLhs(), vadd.getRhs(),
                                       vadd.getResult(), op, SplatKind::Zero,
                                       SplatKind::Zero);
        }
      }

      // 3) AllTrue demask / AllFalse fold on compute
      SmallVector<Operation *> compute;
      module.walk([&](Operation *op) {
        if (isa<VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp, VMIVmaxOp,
                VMIVnegOp, VMIVabsOp, VMIVdhistOp, VMIVchistOp, VMIvcaddOp,
                VMIvcmaxOp, VMIvcminOp>(op))
          compute.push_back(op);
      });
      for (Operation *op : llvm::reverse(compute)) {
        if (!op->getBlock())
          continue;
        if (auto o = dyn_cast<VMIVaddOp>(op)) {
          changed |= demaskBinaryAllTrue(o) || foldBinaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVsubOp>(op)) {
          changed |= demaskBinaryAllTrue(o) || foldBinaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVmulOp>(op)) {
          changed |= demaskBinaryAllTrue(o) || foldBinaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVdivOp>(op)) {
          changed |= demaskBinaryAllTrue(o) || foldBinaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVminOp>(op)) {
          changed |= demaskBinaryAllTrue(o) || foldBinaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVmaxOp>(op)) {
          changed |= demaskBinaryAllTrue(o) || foldBinaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVnegOp>(op)) {
          changed |= demaskUnaryAllTrue(o) || foldUnaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVabsOp>(op)) {
          changed |= demaskUnaryAllTrue(o) || foldUnaryAllFalse(o);
        } else if (auto o = dyn_cast<VMIVdhistOp>(op)) {
          changed |=
              foldHistAllFalse(o.getAcc(), o.getMask(), o.getResult(), op);
        } else if (auto o = dyn_cast<VMIVchistOp>(op)) {
          changed |=
              foldHistAllFalse(o.getAcc(), o.getMask(), o.getResult(), op);
        } else if (auto o = dyn_cast<VMIvcaddOp>(op)) {
          changed |=
              foldReduceAllFalse(o.getMask(), o.getResult(), op, SplatKind::Zero);
        } else if (auto o = dyn_cast<VMIvcmaxOp>(op)) {
          changed |= foldReduceAllFalse(o.getMask(), o.getResult(), op,
                                        SplatKind::NegInf);
        } else if (auto o = dyn_cast<VMIvcminOp>(op)) {
          changed |= foldReduceAllFalse(o.getMask(), o.getResult(), op,
                                        SplatKind::PosInf);
        }
      }

      // 4) Materialize proven mask algebra / vcmp results to create_mask
      SmallVector<Operation *> masks;
      module.walk([&](Operation *op) {
        if (isa<VMIMaskAndOp, VMIMaskOrOp, VMIMaskXOrOp, VMIMaskNotOp, VMIVcmpOp,
                VMIVcmpsOp>(op))
          masks.push_back(op);
      });
      for (Operation *op : llvm::reverse(masks)) {
        if (!op->getBlock() || op->use_empty())
          continue;
        changed |= foldMaskAlgebra(op->getResult(0), op);
      }

      if (changed)
        dcePureUnusedOps(module);
    }

    dcePureUnusedOps(module);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIPredicateFoldPass() {
  return std::make_unique<VMIPredicateFoldPass>();
}
