// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SlotAffineAnalysis.cpp ----------------------------------*- C++ -*-===//

#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir {
namespace pto {

Value findMultiTileSlotExpr(Value v) {
  int hops = 0;
  while (v && hops++ < mlir::pto::kValue32) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return {};
    }
    if (auto get = dyn_cast<pto::MultiTileGetOp>(op)) {
      return get.getSlot();
    }
    if (auto subview = dyn_cast<pto::SubViewOp>(op)) {
      v = subview.getSource();
      continue;
    }
    if (auto reshape = dyn_cast<pto::TReshapeOp>(op)) {
      v = reshape.getSrc();
      continue;
    }
    if (auto bitcast = dyn_cast<pto::BitcastOp>(op)) {
      v = bitcast.getSrc();
      continue;
    }
    return {};
  }
  return {};
}

namespace {

constexpr int kMaxAddSubPeelCount = 4;

// Canonical form `(innerSym + innerOffset) mod N`. `innerSym` may be null
// when the input is a pure constant -- then the canonical form is just
// `innerOffset mod N` with `innerSym == nullptr`.
struct SlotForm {
  Value innerSym;        // null = constant-only form
  int64_t innerOffset{0};
  uint32_t N{0};
};

static bool tryGetConstantInt(Value v, int64_t &out) {
  IntegerAttr attr;
  if (!matchPattern(v, m_Constant(&attr))) {
    return false;
  }
  out = attr.getValue().getSExtValue();
  return true;
}

// Peel `arith.addi`/`arith.subi` with one constant side off `v` into
// `(remaining, offsetDelta)`. Returns false if `v` is not such an op or
// neither side is a constant.
static bool peelAddSubConst(Value v, Value &remaining, int64_t &offset) {
  Operation *op = v.getDefiningOp();
  if (!op) {
    return false;
  }
  Value lhs, rhs;
  bool isSub = false;
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    lhs = add.getLhs();
    rhs = add.getRhs();
  } else if (auto sub = dyn_cast<arith::SubIOp>(op)) {
    lhs = sub.getLhs();
    rhs = sub.getRhs();
    isSub = true;
  } else {
    return false;
  }

  int64_t c;
  if (tryGetConstantInt(rhs, c)) {
    remaining = lhs;
    offset += isSub ? -c : c;
    return true;
  }
  if (!isSub && tryGetConstantInt(lhs, c)) {
    // commutativity only for add
    remaining = rhs;
    offset += c;
    return true;
  }
  return false;
}

// Express `slot` as `(innerSym + innerOffset) mod N` where N is taken from a
// surrounding `arith.remui slot, %const_N`. If `slot` is a pure constant
// without a `remui`, treat N as the caller-supplied `expectN` and reduce.
// Returns false if the form is not representable.
static bool extractSlotForm(Value slot, uint32_t expectN, SlotForm &out) {
  if (!slot) {
    return false;
  }

  out.innerSym = Value();
  out.innerOffset = 0;
  out.N = expectN;

  Operation *def = slot.getDefiningOp();

  // Case 1: `arith.remui inner, %const_N`.
  if (auto remOp = dyn_cast_if_present<arith::RemUIOp>(def)) {
    int64_t n;
    if (!tryGetConstantInt(remOp.getRhs(), n) || n <= 0) {
      return false;
    }
    out.N = static_cast<uint32_t>(n);
    Value inner = remOp.getLhs();
    int64_t offset = 0;
    // Peel at most one add/sub of a constant.
    Value rem = inner;
    int peeled = 0;
    while (peeled++ < kMaxAddSubPeelCount) {
      Value next;
      if (!peelAddSubConst(rem, next, offset)) {
        break;
      }
      rem = next;
    }
    int64_t cst;
    if (tryGetConstantInt(rem, cst)) {
      out.innerSym = Value();
      out.innerOffset = cst + offset;
    } else {
      out.innerSym = rem;
      out.innerOffset = offset;
    }
    return true;
  }

  // Case 2: pure constant (no remui wrapper).
  int64_t cst;
  if (tryGetConstantInt(slot, cst)) {
    out.innerSym = Value();
    out.innerOffset = cst;
    return true;
  }

  // Case 3: bare symbol (`iv` with no remui). Compare equality of bare
  // symbols still works (kEqual when same SSA), but we cannot guarantee
  // the underlying value is in `[0, N)` so disjointness is unsafe.
  out.innerSym = slot;
  out.innerOffset = 0;
  return true;
}

static int64_t pyMod(int64_t a, int64_t n) {
  int64_t r = a % n;
  if (r < 0) {
    r += n;
  }
  return r;
}

} // namespace

SlotRelation compareSlotSSA(Value a, Value b, uint32_t N) {
  if (!a || !b || N == 0) {
    return SlotRelation::kUnknown;
  }

  // Shortcut: same SSA value -> always equal regardless of N.
  if (a == b) {
    return SlotRelation::kEqual;
  }

  SlotForm fa, fb;
  if (!extractSlotForm(a, N, fa) || !extractSlotForm(b, N, fb)) {
    return SlotRelation::kUnknown;
  }

  // Only compare slot forms that share the canonical `mod N` window the
  // caller asked about. The shared utility leaves slots that were not
  // wrapped in `arith.remui` (or were wrapped with a different modulus)
  // to fall through as kUnknown for symbolic forms.
  if (fa.N != N || fb.N != N) {
    // Both pure-constant fallthrough still works: project both onto N.
    if (!fa.innerSym && !fb.innerSym) {
      int64_t da = pyMod(fa.innerOffset, N);
      int64_t db = pyMod(fb.innerOffset, N);
      return da == db ? SlotRelation::kEqual : SlotRelation::kDisjoint;
    }
    return SlotRelation::kUnknown;
  }

  // Pure constants on both sides: project mod N.
  if (!fa.innerSym && !fb.innerSym) {
    int64_t da = pyMod(fa.innerOffset, N);
    int64_t db = pyMod(fb.innerOffset, N);
    return da == db ? SlotRelation::kEqual : SlotRelation::kDisjoint;
  }

  // One side const, other symbolic: cannot prove disjoint without
  // assuming a value range on the symbol. Equality also unprovable.
  if (!fa.innerSym || !fb.innerSym) {
    return SlotRelation::kUnknown;
  }

  // Both symbolic. Need same symbol to reason about (a - b) mod N.
  if (fa.innerSym != fb.innerSym) {
    return SlotRelation::kUnknown;
  }

  int64_t diff = pyMod(fa.innerOffset - fb.innerOffset, N);
  return diff == 0 ? SlotRelation::kEqual : SlotRelation::kDisjoint;
}

} // namespace pto
} // namespace mlir
