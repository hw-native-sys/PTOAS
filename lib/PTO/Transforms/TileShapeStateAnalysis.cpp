// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileShapeStateAnalysis.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool getStaticInt(Value value, int64_t &out) {
  if (auto c = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = c.value();
    return true;
  }
  if (auto c = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = c.value();
    return true;
  }
  auto foldBinary = [&](auto op) {
    int64_t lhs = 0, rhs = 0;
    if (!getStaticInt(op.getLhs(), lhs) || !getStaticInt(op.getRhs(), rhs))
      return false;
    if constexpr (std::is_same_v<decltype(op), arith::AddIOp>)
      return !__builtin_add_overflow(lhs, rhs, &out);
    else if constexpr (std::is_same_v<decltype(op), arith::SubIOp>)
      return !__builtin_sub_overflow(lhs, rhs, &out);
    else
      return !__builtin_mul_overflow(lhs, rhs, &out);
  };
  if (auto op = value.getDefiningOp<arith::AddIOp>())
    return foldBinary(op);
  if (auto op = value.getDefiningOp<arith::SubIOp>())
    return foldBinary(op);
  if (auto op = value.getDefiningOp<arith::MulIOp>())
    return foldBinary(op);
  return false;
}

static bool validShapeFromSetValidShape(
    pto::SetValidShapeOp op, SmallVectorImpl<int64_t> &result) {
  int64_t row = 0, col = 0;
  if (!getStaticInt(op.getValidRow(), row) ||
      !getStaticInt(op.getValidCol(), col) || row < 0 || col < 0)
    return false;
  result.assign({row, col});
  return true;
}

enum class UpdateResolution { None, Found, Ambiguous };

static UpdateResolution
resolveDominatingUpdate(Value value, Operation *useOp,
                        SmallVectorImpl<int64_t> &result) {
  SmallVector<pto::SetValidShapeOp, 4> updates;
  for (Operation *user : value.getUsers()) {
    auto update = dyn_cast<pto::SetValidShapeOp>(user);
    if (update && update.getSource() == value)
      updates.push_back(update);
  }
  if (updates.empty())
    return UpdateResolution::None;

  if (!useOp) {
    std::optional<SmallVector<int64_t, 2>> only;
    for (auto update : updates) {
      SmallVector<int64_t, 2> candidate;
      if (!validShapeFromSetValidShape(update, candidate))
        return UpdateResolution::Ambiguous;
      if (only && *only != candidate)
        return UpdateResolution::Ambiguous;
      only = std::move(candidate);
    }
    if (!only)
      return UpdateResolution::None;
    result.assign(only->begin(), only->end());
    return UpdateResolution::Found;
  }

  Operation *root = useOp->getParentOfType<ModuleOp>();
  if (!root)
    root = useOp->getParentOp();
  DominanceInfo dominance(root);
  std::optional<pto::SetValidShapeOp> nearest;
  bool hasPotentiallyReachingNonDominator = false;
  for (auto update : updates) {
    if (!dominance.dominates(update.getOperation(), useOp)) {
      // An update textually after the use in the same block cannot affect the
      // current use.  An update in another block may reach a join without
      // dominating it, so the shape is path-dependent and therefore unknown.
      if (update->getBlock() != useOp->getBlock() ||
          !useOp->isBeforeInBlock(update))
        hasPotentiallyReachingNonDominator = true;
      continue;
    }
    if (!nearest) {
      nearest = update;
      continue;
    }
    // A later update is more precise only when it dominates the current use
    // and the previous update.  Incomparable updates mean the value is not
    // uniquely known at this point.
    if (dominance.dominates(nearest->getOperation(), update.getOperation()))
      nearest = update;
    else if (!dominance.dominates(update.getOperation(),
                                  nearest->getOperation()))
      return UpdateResolution::Ambiguous;
  }
  if (hasPotentiallyReachingNonDominator)
    return UpdateResolution::Ambiguous;
  if (!nearest)
    return UpdateResolution::None;
  return validShapeFromSetValidShape(*nearest, result)
             ? UpdateResolution::Found
             : UpdateResolution::Ambiguous;
}

static bool resolveDeclaredTpop(Value value,
                                SmallVectorImpl<int64_t> &result) {
  auto decl = value.getDefiningOp<pto::DeclareTileOp>();
  if (!decl)
    return false;
  auto type = dyn_cast<pto::TileBufType>(value.getType());
  if (!type || llvm::any_of(type.getShape(), ShapedType::isDynamic))
    return false;
  for (Operation *user : value.getUsers()) {
    if (auto setValidShape = dyn_cast<pto::SetValidShapeOp>(user))
      if (setValidShape.getSource() == value)
        return false;
    if (isa<pto::TPopOp>(user)) {
      result.assign(type.getShape().begin(), type.getShape().end());
      return true;
    }
  }
  return false;
}

} // namespace

bool mlir::pto::resolveStaticTileValidShape(
    Value value, SmallVectorImpl<int64_t> &validShape, Operation *useOp) {
  UpdateResolution update =
      resolveDominatingUpdate(value, useOp, validShape);
  if (update == UpdateResolution::Found)
    return true;
  if (update == UpdateResolution::Ambiguous)
    // An explicit dynamic or path-dependent update overrides any
    // declaration-level shape fact. Never treat it as a full tile.
    return false;

  if (auto type = dyn_cast<pto::TileBufType>(value.getType())) {
    ArrayRef<int64_t> declared = type.getValidShape();
    if (declared.size() == type.getShape().size() &&
        !llvm::any_of(declared, ShapedType::isDynamic)) {
      validShape.assign(declared.begin(), declared.end());
      return true;
    }
  }

  if (resolveDeclaredTpop(value, validShape))
    return true;

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  Value row, col;
  if (auto alloc = dyn_cast<pto::AllocTileOp>(def)) {
    row = alloc.getValidRow();
    col = alloc.getValidCol();
  } else if (auto subview = dyn_cast<pto::SubViewOp>(def)) {
    row = subview.getValidRow();
    col = subview.getValidCol();
  } else if (auto aic = dyn_cast<pto::TPopFromAicOp>(def)) {
    row = aic.getValidRow();
    col = aic.getValidCol();
  } else if (auto aiv = dyn_cast<pto::TPopFromAivOp>(def)) {
    row = aiv.getValidRow();
    col = aiv.getValidCol();
  } else if (auto region = dyn_cast<pto::FusionRegionOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    auto yield = dyn_cast<pto::YieldOp>(region.getBody().front().getTerminator());
    if (!yield || result.getResultNumber() >= yield.getNumOperands())
      return false;
    return resolveStaticTileValidShape(yield.getOperand(result.getResultNumber()),
                                       validShape, region);
  }
  if (!row || !col)
    return false;
  int64_t r = 0, c = 0;
  if (!getStaticInt(row, r) || !getStaticInt(col, c) || r < 0 || c < 0)
    return false;
  validShape.assign({r, c});
  return true;
}

TileShapeState mlir::pto::analyzeTileShape(Value value, Operation *useOp) {
  TileShapeState state;
  auto type = dyn_cast<pto::TileBufType>(value.getType());
  if (!type)
    return state;
  state.shape.assign(type.getShape().begin(), type.getShape().end());
  if (!resolveStaticTileValidShape(value, state.validShape, useOp))
    return state;
  if (state.shape.size() != state.validShape.size() ||
      llvm::any_of(state.shape, ShapedType::isDynamic) ||
      llvm::any_of(state.validShape, ShapedType::isDynamic)) {
    state.kind = TileShapeState::Kind::Unknown;
    return state;
  }
  state.kind = state.shape == state.validShape ? TileShapeState::Kind::Full
                                                : TileShapeState::Kind::Partial;
  return state;
}

bool mlir::pto::hasStaticFullTileValidShape(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (!isa<pto::TileBufType>(operand.getType()))
      continue;
    if (!analyzeTileShape(operand, op).isFull())
      return false;
  }
  // A scalar-only operation has no tile shape obligation.  Candidate-specific
  // operand-form checks remain responsible for rejecting forms that cannot be
  // vectorized; this helper only answers the valid-shape question.
  return true;
}
