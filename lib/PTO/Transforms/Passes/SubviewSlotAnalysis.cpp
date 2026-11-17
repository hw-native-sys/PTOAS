// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Disjointness for dense full-row tile subviews. Keep this separate from
// general memory aliasing: rotating rows can reuse storage in later iterations.
// Cross-iteration pruning requires its own non-reuse proof. Unsupported forms
// fail closed.

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::pto {
namespace {

constexpr unsigned kMaxAnalysisDepth = 16;
constexpr unsigned kTileRank = 2;

static std::optional<uint64_t> getNonnegativeConstant(Value value) {
  IntegerAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  bool valid = attr.getValue().isSignedIntN(64) && !attr.getValue().isNegative();
  if (!valid) {
    return std::nullopt;
  }
  return attr.getValue().getZExtValue();
}

struct IndexRange {
  uint64_t min;
  uint64_t max;
};

// Stay within the positive signed range as well as the unsigned range. For
// index, use the portable 32-bit range rather than assuming a host pointer size.
static uint64_t getIndexLimit(Type type) {
  unsigned width = type.isIndex() ? 32 : type.getIntOrFloatBitWidth();
  if (width == 0 || width > mlir::pto::kValue64) {
      return 0;
  }
  return (uint64_t{1} << (width - 1)) - 1;
}

static std::optional<IndexRange> getIndexRange(Value value, unsigned depth);

static std::optional<IndexRange> getArithmeticRange(Value value, Value lhs,
                                                  Value rhs, bool subtract,
                                                  unsigned depth) {
  auto left = getIndexRange(lhs, depth + 1);
  auto right = getIndexRange(rhs, depth + 1);
  if (!left || !right) {
    return std::nullopt;
  }
  if (subtract) {
    if (left->min < right->max) {
      return std::nullopt;
    }
    return IndexRange{left->min - right->max, left->max - right->min};
  }
  uint64_t limit = getIndexLimit(value.getType());
  if (right->max > limit || left->max > limit - right->max) {
    return std::nullopt;
  }
  return IndexRange{left->min + right->min, left->max + right->max};
}

static std::optional<IndexRange> getIndexRange(Value value, unsigned depth) {
  if (depth >= kMaxAnalysisDepth) {
    return std::nullopt;
  }
  if (auto constant = getNonnegativeConstant(value)) {
    return IndexRange{*constant, *constant};
  }
  if (auto rem = value.getDefiningOp<arith::RemUIOp>()) {
    auto modulus = getNonnegativeConstant(rem.getRhs());
    if (modulus && *modulus > 0) {
      return IndexRange{0, *modulus - 1};
    }
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    return getArithmeticRange(value, add.getLhs(), add.getRhs(), false, depth);
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    return getArithmeticRange(value, sub.getLhs(), sub.getRhs(), true, depth);
  }
  return std::nullopt;
}

struct RowSlotForm {
  Value symbol;
  uint64_t offset = 0;
};

// Only peel arithmetic after proving that it cannot wrap. Reducing offsets
// after every step also keeps the compiler's own arithmetic bounded by 2*N.
static bool peelSlotOffset(Value &inner, uint64_t &offset, uint64_t count) {
  if (count == 0) {
    return false;
  }
  Value lhs;
  Value rhs;
  bool subtract = false;
  if (auto add = inner.getDefiningOp<arith::AddIOp>()) {
    lhs = add.getLhs();
    rhs = add.getRhs();
  } else if (auto sub = inner.getDefiningOp<arith::SubIOp>()) {
    lhs = sub.getLhs();
    rhs = sub.getRhs();
    subtract = true;
  } else {
    return false;
  }
  auto constant = getNonnegativeConstant(rhs);
  if (!constant && !subtract) {
    constant = getNonnegativeConstant(lhs);
    std::swap(lhs, rhs);
  }
  if (!constant || !getIndexRange(inner, 0)) {
    return false;
  }
  uint64_t delta = *constant % count;
  offset = (offset + (subtract ? count - delta : delta)) % count;
  inner = lhs;
  return true;
}

static std::optional<RowSlotForm> getRowSlotForm(Value slot, uint64_t count) {
  if (count == 0) {
    return std::nullopt;
  }
  if (auto constant = getNonnegativeConstant(slot)) {
    if (*constant < count) {
      return RowSlotForm{Value(), *constant};
    }
    return std::nullopt;
  }
  auto rem = slot.getDefiningOp<arith::RemUIOp>();
  if (!rem) {
    return std::nullopt;
  }
  auto modulus = getNonnegativeConstant(rem.getRhs());
  if (modulus != count) {
    return std::nullopt;
  }
  Value inner = rem.getLhs();
  uint64_t offset = 0;
  for (unsigned depth = 0; depth < kMaxAnalysisDepth; ++depth) {
    if (!peelSlotOffset(inner, offset, count)) {
      break;
    }
  }
  if (auto constant = getNonnegativeConstant(inner)) {
    return RowSlotForm{Value(), (offset + *constant % count) % count};
  }
  return RowSlotForm{inner, offset};
}

static std::optional<uint64_t> getDenseTileBytes(Type type) {
  auto tile = dyn_cast<TileBufType>(type);
  auto space = tile ? dyn_cast_if_present<AddressSpaceAttr>(tile.getMemorySpace())
                    : AddressSpaceAttr();
  bool dense = tile && space && space.getAddressSpace() == AddressSpace::VEC &&
               tile.getShape().size() == kTileRank &&
               tile.getBLayoutValueI32() == static_cast<int32_t>(BLayout::RowMajor) &&
               tile.getSLayoutValueI32() == static_cast<int32_t>(SLayout::NoneBox) &&
               tile.getCompactModeI32() != static_cast<int32_t>(CompactMode::RowPlusOne);
  if (!dense) {
    return std::nullopt;
  }
  uint64_t bytes = getPTOStorageElemByteSize(tile.getElementType());
  bool byteAligned = getPTOStorageElemBitWidth(tile.getElementType()) % 8 == 0;
  if (bytes == 0 || !byteAligned) {
    return std::nullopt;
  }
  for (int64_t dim : tile.getShape()) {
    bool validDim = dim > 0 && static_cast<uint64_t>(dim) <=
                                  std::numeric_limits<uint64_t>::max() / bytes;
    if (!validDim) {
      return std::nullopt;
    }
    bytes *= static_cast<uint64_t>(dim);
  }
  return bytes;
}

static SubViewOp findRowSubview(Value value) {
  Value current = value;
  for (unsigned depth = 0; depth < kMaxAnalysisDepth; ++depth) {
    if (auto subview = current.getDefiningOp<SubViewOp>()) {
      return subview;
    }
    Value source;
    if (auto reshape = current.getDefiningOp<TReshapeOp>()) {
      source = reshape.getSrc();
    } else if (auto bitcast = current.getDefiningOp<BitcastOp>()) {
      source = bitcast.getSrc();
    } else {
      return {};
    }
    auto bytes = getDenseTileBytes(current.getType());
    if (!bytes || bytes != getDenseTileBytes(source.getType())) {
      return {};
    }
    current = source;
  }
  return {};
}

struct DenseRowAccess {
    Value allocation;
    uint64_t count;
    Value row;
};

static std::optional<DenseRowAccess> getDenseRowAccess(Value value)
{
    auto view = findRowSubview(value);
    bool denseView = view && view.getSource().getDefiningOp<AllocTileOp>() &&
                     getDenseTileBytes(view.getSource().getType()) && getDenseTileBytes(view.getType()) &&
                     view.getOffsets().size() == kTileRank && view.getSizes().size() == kTileRank;
    if (!denseView) {
        return std::nullopt;
    }
    auto parent = cast<TileBufType>(view.getSource().getType());
    auto result = cast<TileBufType>(view.getType());
    int64_t rows = parent.getShape()[0];
    int64_t cols = parent.getShape()[1];
    bool fullRow = rows >= 2 && rows <= std::numeric_limits<uint32_t>::max() &&
                   getNonnegativeConstant(view.getOffsets()[1]) == 0 &&
                   cast<IntegerAttr>(view.getSizes()[0]).getInt() == 1 &&
                   cast<IntegerAttr>(view.getSizes()[1]).getInt() == cols && result.getShape()[0] == 1 &&
                   result.getShape()[1] == cols && result.getElementType() == parent.getElementType();
    if (!fullRow) {
        return std::nullopt;
    }
    return DenseRowAccess{view.getSource(), static_cast<uint64_t>(rows), view.getOffsets()[0]};
}

struct RowSlotAccess {
    Value allocation;
    uint64_t count;
    RowSlotForm slot;
};

static std::optional<RowSlotAccess> getRowSlotAccess(Value value)
{
    auto access = getDenseRowAccess(value);
    if (!access) {
        return std::nullopt;
    }
    auto slot = getRowSlotForm(access->row, access->count);
    if (!slot) {
        return std::nullopt;
    }
    return RowSlotAccess{access->allocation, access->count, *slot};
}

} // namespace

bool areDisjointSubviewSlots(Value a, Value b) {
  auto left = getRowSlotAccess(a);
  auto right = getRowSlotAccess(b);
  return left && right && left->allocation == right->allocation &&
         left->count == right->count && left->slot.symbol == right->slot.symbol &&
         left->slot.offset != right->slot.offset;
}

bool areSubviewIterationsDisjoint(Value a, Value b, scf::ForOp forOp)
{
    if (!a || !b || !forOp) {
        return false;
    }
    auto left = getDenseRowAccess(a);
    auto right = getDenseRowAccess(b);
    if (!left || !right || left->allocation != right->allocation || left->count != right->count ||
        left->row != forOp.getInductionVar() || right->row != forOp.getInductionVar() ||
        !forOp.isDefinedOutsideOfLoop(left->allocation)) {
        return false;
    }

    auto lower = getNonnegativeConstant(forOp.getLowerBound());
    auto upper = getNonnegativeConstant(forOp.getUpperBound());
    auto step = getNonnegativeConstant(forOp.getStep());
    if (!lower || !upper || step != 1 || *lower > *upper || *upper > left->count ||
        *upper > getIndexLimit(forOp.getInductionVar().getType())) {
        return false;
    }

    // Every execution selects exactly its induction-variable row. Unit steps
    // within the parent and the positive signed index range cannot wrap, so
    // ANY two distinct iterations select disjoint byte ranges. Proving only
    // adjacent rows would be insufficient for rotating or strided aliases.
    return true;
}

} // namespace mlir::pto
