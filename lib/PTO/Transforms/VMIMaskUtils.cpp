// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskUtils.cpp - Shared VMI predicate / seed helpers -------------===//

#include "PTO/Transforms/VMIMaskUtils.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<int64_t> matchConstantInt(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

static MaskLattice classifyCompare(StringRef cmp, const IntRange &lhs,
                                   const IntRange &rhs) {
  if (cmp == "lt" || cmp == "olt") {
    if (lhs.hi < rhs.lo)
      return MaskLattice::AllTrue;
    if (lhs.lo >= rhs.hi)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "le" || cmp == "ole") {
    if (lhs.hi <= rhs.lo)
      return MaskLattice::AllTrue;
    if (lhs.lo > rhs.hi)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "gt" || cmp == "ogt") {
    if (lhs.lo > rhs.hi)
      return MaskLattice::AllTrue;
    if (lhs.hi <= rhs.lo)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "ge" || cmp == "oge") {
    if (lhs.lo >= rhs.hi)
      return MaskLattice::AllTrue;
    if (lhs.hi < rhs.lo)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "eq" || cmp == "oeq") {
    if (lhs.lo == lhs.hi && rhs.lo == rhs.hi && lhs.lo == rhs.lo)
      return MaskLattice::AllTrue;
    if (lhs.hi < rhs.lo || lhs.lo > rhs.hi)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  if (cmp == "ne" || cmp == "one") {
    if (lhs.hi < rhs.lo || lhs.lo > rhs.hi)
      return MaskLattice::AllTrue;
    if (lhs.lo == lhs.hi && rhs.lo == rhs.hi && lhs.lo == rhs.lo)
      return MaskLattice::AllFalse;
    return MaskLattice::Unknown;
  }
  return MaskLattice::Unknown;
}

} // namespace

bool mlir::pto::isAllActiveSeed(Value seed) {
  Operation *def = seed.getDefiningOp();
  if (!def)
    return false;
  if (isa<VMIPsetOp>(def))
    return true;
  if (auto cm = dyn_cast<VMICreateMaskOp>(def)) {
    auto maskTy = cast<VMIMaskType>(cm.getResult().getType());
    if (auto cst = cm.getActiveLanes().getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
        return ia.getInt() >= maskTy.getElementCount();
  }
  return false;
}

bool mlir::pto::isAllInactiveSeed(Value seed) {
  Operation *def = seed.getDefiningOp();
  if (!def)
    return false;
  if (auto cm = dyn_cast<VMICreateMaskOp>(def)) {
    if (auto cst = cm.getActiveLanes().getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
        return ia.getInt() <= 0;
  }
  return false;
}

std::optional<IntRange> mlir::pto::matchAffineIntRange(Value v) {
  if (auto c = matchConstantInt(v))
    return IntRange::splat(*c);

  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::IndexCastUIOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::ExtSIOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::ExtUIOp>())
    return matchAffineIntRange(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::TruncIOp>())
    return matchAffineIntRange(cast.getIn());

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto lhs = matchAffineIntRange(add.getLhs());
    auto rhs = matchAffineIntRange(add.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t lo, hi;
    if (llvm::AddOverflow(lhs->lo, rhs->lo, lo) ||
        llvm::AddOverflow(lhs->hi, rhs->hi, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  }

  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    auto lhs = matchAffineIntRange(sub.getLhs());
    auto rhs = matchAffineIntRange(sub.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t lo, hi;
    if (llvm::SubOverflow(lhs->lo, rhs->hi, lo) ||
        llvm::SubOverflow(lhs->hi, rhs->lo, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  }

  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto lhsC = matchConstantInt(mul.getLhs());
    auto rhsC = matchConstantInt(mul.getRhs());
    if (lhsC && rhsC) {
      int64_t prod;
      if (llvm::MulOverflow(*lhsC, *rhsC, prod))
        return std::nullopt;
      return IntRange::splat(prod);
    }

    Value dyn = lhsC ? mul.getRhs() : mul.getLhs();
    auto factorOpt = lhsC ? lhsC : rhsC;
    if (!factorOpt)
      return std::nullopt;
    int64_t factor = *factorOpt;
    auto dynR = matchAffineIntRange(dyn);
    if (!dynR)
      return std::nullopt;
    int64_t a, b;
    if (llvm::MulOverflow(dynR->lo, factor, a) ||
        llvm::MulOverflow(dynR->hi, factor, b))
      return std::nullopt;
    return IntRange{std::min(a, b), std::max(a, b)};
  }

  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg != forOp.getInductionVar())
        return std::nullopt;
      auto lb = matchConstantInt(forOp.getLowerBound());
      auto ub = matchConstantInt(forOp.getUpperBound());
      auto step = matchConstantInt(forOp.getStep());
      if (!lb || !ub || !step || *step <= 0 || *lb >= *ub)
        return std::nullopt;
      int64_t last = *lb + ((*ub - 1 - *lb) / *step) * *step;
      return IntRange{*lb, last};
    }
  }

  return std::nullopt;
}

std::optional<IntRange> mlir::pto::matchVectorLaneRange(Value v) {
  auto vty = dyn_cast<VMIVRegType>(v.getType());
  if (!vty)
    return std::nullopt;
  int64_t vl = vty.getElementCount();

  if (auto brc = v.getDefiningOp<VMIVbrcOp>()) {
    if (auto c = matchConstantInt(brc.getValue()))
      return IntRange::splat(*c);
    return std::nullopt;
  }
  if (auto brc = v.getDefiningOp<VMIBroadcastOp>()) {
    if (auto c = matchConstantInt(brc.getValue()))
      return IntRange::splat(*c);
    return std::nullopt;
  }

  auto matchIotaLike = [&](Value base, std::optional<int64_t> group,
                           StringRef order) -> std::optional<IntRange> {
    if (!order.empty() && order != "ASC")
      return std::nullopt;
    auto baseR = matchAffineIntRange(base);
    if (!baseR)
      return std::nullopt;
    int64_t span = vl;
    if (group && *group > 0) {
      if (vl % *group != 0)
        return std::nullopt;
      span = vl / *group;
    }
    int64_t lo = baseR->lo;
    int64_t hi;
    if (llvm::AddOverflow(baseR->hi, span - 1, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  };

  if (auto vci = v.getDefiningOp<VMIVciOp>()) {
    std::optional<int64_t> group;
    if (auto g = vci->getAttrOfType<IntegerAttr>("group"))
      group = g.getInt();
    StringRef order = vci.getOrder() ? *vci.getOrder() : StringRef("ASC");
    return matchIotaLike(vci.getBase(), group, order);
  }
  if (auto iota = v.getDefiningOp<VMIIotaOp>()) {
    std::optional<int64_t> group;
    if (auto g = iota->getAttrOfType<IntegerAttr>("group"))
      group = g.getInt();
    StringRef order = iota.getOrder() ? *iota.getOrder() : StringRef("ASC");
    return matchIotaLike(iota.getBase(), group, order);
  }

  if (auto vadds = v.getDefiningOp<VMIAddSOp>()) {
    if (!isAllActiveSeed(vadds.getMask()))
      return std::nullopt;
    auto srcR = matchVectorLaneRange(vadds.getSrc());
    auto sc = matchConstantInt(vadds.getScalar());
    if (!srcR || !sc)
      return std::nullopt;
    int64_t lo, hi;
    if (llvm::AddOverflow(srcR->lo, *sc, lo) ||
        llvm::AddOverflow(srcR->hi, *sc, hi))
      return std::nullopt;
    return IntRange{lo, hi};
  }

  // vadd(v, vbrc(C)) / vadd(vbrc(C), v) with all-active (or absent) mask.
  if (auto vadd = v.getDefiningOp<VMIVaddOp>()) {
    if (!vadd.getMask().empty() && !isAllActiveSeed(vadd.getMask().front()))
      return std::nullopt;
    Value lhs = vadd.getLhs();
    Value rhs = vadd.getRhs();
    auto tryShift = [&](Value src, Value brcCand) -> std::optional<IntRange> {
      auto srcR = matchVectorLaneRange(src);
      if (!srcR)
        return std::nullopt;
      std::optional<int64_t> sc;
      if (auto vb = brcCand.getDefiningOp<VMIVbrcOp>())
        sc = matchConstantInt(vb.getValue());
      else if (auto vb = brcCand.getDefiningOp<VMIBroadcastOp>())
        sc = matchConstantInt(vb.getValue());
      if (!sc)
        return std::nullopt;
      int64_t lo, hi;
      if (llvm::AddOverflow(srcR->lo, *sc, lo) ||
          llvm::AddOverflow(srcR->hi, *sc, hi))
        return std::nullopt;
      return IntRange{lo, hi};
    };
    if (auto r = tryShift(lhs, rhs))
      return r;
    if (auto r = tryShift(rhs, lhs))
      return r;
  }

  return std::nullopt;
}

MaskLattice mlir::pto::classifyMaskValue(Value mask) {
  if (isAllActiveSeed(mask))
    return MaskLattice::AllTrue;
  if (isAllInactiveSeed(mask))
    return MaskLattice::AllFalse;

  if (auto mand = mask.getDefiningOp<VMIMaskAndOp>()) {
    MaskLattice lhs = classifyMaskValue(mand.getLhs());
    MaskLattice rhs = classifyMaskValue(mand.getRhs());
    if (lhs == MaskLattice::AllFalse || rhs == MaskLattice::AllFalse)
      return MaskLattice::AllFalse;
    if (lhs == MaskLattice::AllTrue && rhs == MaskLattice::AllTrue)
      return MaskLattice::AllTrue;
    if (lhs == MaskLattice::AllTrue)
      return rhs;
    if (rhs == MaskLattice::AllTrue)
      return lhs;
    return MaskLattice::Unknown;
  }
  if (auto mor = mask.getDefiningOp<VMIMaskOrOp>()) {
    MaskLattice lhs = classifyMaskValue(mor.getLhs());
    MaskLattice rhs = classifyMaskValue(mor.getRhs());
    if (lhs == MaskLattice::AllTrue || rhs == MaskLattice::AllTrue)
      return MaskLattice::AllTrue;
    if (lhs == MaskLattice::AllFalse && rhs == MaskLattice::AllFalse)
      return MaskLattice::AllFalse;
    if (lhs == MaskLattice::AllFalse)
      return rhs;
    if (rhs == MaskLattice::AllFalse)
      return lhs;
    return MaskLattice::Unknown;
  }
  if (auto mxor = mask.getDefiningOp<VMIMaskXOrOp>()) {
    MaskLattice lhs = classifyMaskValue(mxor.getLhs());
    MaskLattice rhs = classifyMaskValue(mxor.getRhs());
    if (lhs == MaskLattice::Unknown || rhs == MaskLattice::Unknown)
      return MaskLattice::Unknown;
    if (lhs == rhs)
      return MaskLattice::AllFalse;
    return MaskLattice::AllTrue;
  }
  if (auto mnot = mask.getDefiningOp<VMIMaskNotOp>()) {
    MaskLattice src = classifyMaskValue(mnot.getSource());
    if (src == MaskLattice::AllTrue)
      return MaskLattice::AllFalse;
    if (src == MaskLattice::AllFalse)
      return MaskLattice::AllTrue;
    return MaskLattice::Unknown;
  }

  if (auto vcmp = mask.getDefiningOp<VMIVcmpOp>()) {
    auto lhs = matchVectorLaneRange(vcmp.getLhs());
    auto rhs = matchVectorLaneRange(vcmp.getRhs());
    if (!lhs || !rhs)
      return MaskLattice::Unknown;
    MaskLattice raw = classifyCompare(vcmp.getCmp(), *lhs, *rhs);
    MaskLattice seedLat = classifyMaskValue(vcmp.getSeed());
    if (raw == MaskLattice::AllFalse || seedLat == MaskLattice::AllFalse)
      return MaskLattice::AllFalse;
    if (raw == MaskLattice::AllTrue && seedLat == MaskLattice::AllTrue)
      return MaskLattice::AllTrue;
    return MaskLattice::Unknown;
  }

  if (auto vcmps = mask.getDefiningOp<VMIVcmpsOp>()) {
    auto lhs = matchVectorLaneRange(vcmps.getSrc());
    auto sc = matchConstantInt(vcmps.getScalar());
    if (!lhs || !sc)
      return MaskLattice::Unknown;
    MaskLattice raw =
        classifyCompare(vcmps.getCmp(), *lhs, IntRange::splat(*sc));
    MaskLattice seedLat = classifyMaskValue(vcmps.getSeed());
    if (raw == MaskLattice::AllFalse || seedLat == MaskLattice::AllFalse)
      return MaskLattice::AllFalse;
    if (raw == MaskLattice::AllTrue && seedLat == MaskLattice::AllTrue)
      return MaskLattice::AllTrue;
    return MaskLattice::Unknown;
  }

  return MaskLattice::Unknown;
}
