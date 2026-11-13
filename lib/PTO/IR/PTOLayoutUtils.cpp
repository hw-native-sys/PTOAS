// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTOLayoutUtils.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr int64_t kNZBlockBytes = 32;
constexpr int64_t kNZInnerRows = 16;

struct ShapeStride5D {
  SmallVector<int64_t, kPTOLayoutRank> shape;
  SmallVector<int64_t, kPTOLayoutRank> stride;
};

static bool isDynamic(int64_t value) { return value == ShapedType::kDynamic; }

static bool isKnownInvalid(int64_t value) {
  return value < 0 && !isDynamic(value);
}

static bool matchesKnown(int64_t actual, int64_t expected) {
  return isDynamic(actual) || actual == expected;
}

static std::optional<int64_t> checkedMultiply(int64_t lhs, int64_t rhs) {
  if (isDynamic(lhs) || isDynamic(rhs)) {
    return ShapedType::kDynamic;
  }
  int64_t product = 0;
  if (llvm::MulOverflow(lhs, rhs, product) != 0) {
    return std::nullopt;
  }
  return product;
}

static std::optional<ShapeStride5D> rightAlignTo5D(ArrayRef<int64_t> shape,
                                                   ArrayRef<int64_t> stride) {
  if (shape.empty() || shape.size() != stride.size() ||
      shape.size() > kPTOLayoutRank) {
    return std::nullopt;
  }
  if (llvm::any_of(shape, isKnownInvalid) ||
      llvm::any_of(stride, isKnownInvalid)) {
    return std::nullopt;
  }

  ShapeStride5D result;
  result.shape.assign(kPTOLayoutRank, 1);
  result.stride.assign(kPTOLayoutRank, 1);

  const int64_t shift = kPTOLayoutRank - shape.size();
  for (auto [index, value] : llvm::enumerate(shape)) {
    result.shape[shift + index] = value;
  }
  for (auto [index, value] : llvm::enumerate(stride)) {
    result.stride[shift + index] = value;
  }

  for (int64_t index = shift - 1; index >= 0; --index) {
    auto value =
        checkedMultiply(result.shape[index + 1], result.stride[index + 1]);
    if (!value) {
      return std::nullopt;
    }
    result.stride[index] = *value;
  }
  return result;
}

static bool allStatic(ArrayRef<int64_t> values) {
  return llvm::none_of(values, isDynamic);
}

static bool matchesNDMinor2D(int64_t rows, int64_t cols, int64_t rowStride,
                             int64_t colStride) {
  if (!isDynamic(cols) && cols != 1 && !isDynamic(colStride) && colStride != 1) {
    return false;
  }
  if (!isDynamic(rows) && rows == 1) {
    return true;
  }
  if (!isDynamic(cols) && cols == 1) {
    return isDynamic(rowStride) || rowStride == 1;
  }
  if (isDynamic(rowStride) || isDynamic(cols)) {
    return true;
  }
  return rowStride == cols;
}

static bool matchesDNMinor2D(int64_t rows, int64_t cols, int64_t rowStride,
                             int64_t colStride) {
  if (!isDynamic(rows) && rows != 1 && !isDynamic(rowStride) && rowStride != 1) {
    return false;
  }
  if (!isDynamic(cols) && cols == 1) {
    return true;
  }
  if (!isDynamic(rows) && rows == 1) {
    return isDynamic(colStride) || colStride == 1;
  }
  if (isDynamic(colStride) || isDynamic(rows)) {
    return true;
  }
  return colStride == rows;
}

static std::optional<Layout>
inferMinor2DLayout(int64_t rows, int64_t cols, int64_t rowStride,
                   int64_t colStride, std::optional<Layout> preferredMinor2D,
                   bool *isMinor2DAmbiguous) {
  const bool nd = matchesNDMinor2D(rows, cols, rowStride, colStride);
  const bool dn = matchesDNMinor2D(rows, cols, rowStride, colStride);
  if (!nd && !dn) {
    return Layout::ND;
  }
  if (nd && dn) {
    if (isMinor2DAmbiguous) {
      *isMinor2DAmbiguous = true;
    }
    if (preferredMinor2D &&
        (*preferredMinor2D == Layout::ND || *preferredMinor2D == Layout::DN)) {
      return *preferredMinor2D;
    }
    // Both interpretations address the same elements when a minor extent is
    // one. Keep the historical row-major default unless a consumer supplies
    // an explicit preference.
    return Layout::ND;
  }
  return dn ? Layout::DN : Layout::ND;
}

static std::string expectedActual(StringRef name, int64_t expected,
                                  int64_t actual) {
  return ("expected " + name + "==" + Twine(expected) + " (got " +
          Twine(actual) + ")")
      .str();
}

} // namespace

bool mlir::pto::isMxLayout(Layout layout)
{
  switch (layout) {
  case Layout::MX_A_ND:
  case Layout::MX_A_DN:
  case Layout::MX_A_ZZ:
  case Layout::MX_B_ND:
  case Layout::MX_B_DN:
  case Layout::MX_B_NN:
    return true;
  default:
    return false;
  }
}

static std::optional<int64_t>
getNZC0StorageElems(unsigned storageElemBytes) {
  if (storageElemBytes == 0 || kNZBlockBytes % storageElemBytes != 0) {
    return std::nullopt;
  }
  return kNZBlockBytes / storageElemBytes;
}

static std::optional<std::string>
getNZInnerStructureError(ArrayRef<int64_t> shape5D,
                         ArrayRef<int64_t> stride5D, int64_t c0) {
  if (!isDynamic(shape5D[0]) && shape5D[0] <= 0) {
    return "NZ layout requires shape[0] to be positive";
  }
  if (!matchesKnown(shape5D[3], kNZInnerRows)) {
    return expectedActual("shape[3]", kNZInnerRows, shape5D[3]);
  }
  if (!matchesKnown(shape5D[4], c0)) {
    return expectedActual("shape[4]", c0, shape5D[4]);
  }
  if (!matchesKnown(stride5D[4], 1)) {
    return expectedActual("stride[4]", 1, stride5D[4]);
  }
  if (!matchesKnown(stride5D[3], c0)) {
    return expectedActual("stride[3]", c0, stride5D[3]);
  }
  if (!matchesKnown(stride5D[2], kNZInnerRows * c0)) {
    return expectedActual("stride[2]", kNZInnerRows * c0, stride5D[2]);
  }
  return std::nullopt;
}

static std::optional<std::string>
getNZOuterStrideError(ArrayRef<int64_t> shape5D,
                      ArrayRef<int64_t> stride5D, int64_t index, int64_t c0) {
  if (c0 == 0) {
    return "NZ layout requires a non-zero C0";
  }
  if (isDynamic(stride5D[index])) {
    return std::nullopt;
  }
  if (stride5D[index] % c0 != 0) {
    return ("expected stride[" + Twine(index) + "] to be C0-aligned (" +
            Twine(c0) + "), got " + Twine(stride5D[index]))
        .str();
  }

  auto minimum = checkedMultiply(shape5D[index + 1], stride5D[index + 1]);
  if (!minimum) {
    return ("shape[" + Twine(index + 1) + "] * stride[" +
            Twine(index + 1) + "] overflows")
        .str();
  }
  if (!isDynamic(*minimum) && stride5D[index] < *minimum) {
    return ("expected stride[" + Twine(index) + "] >= shape[" +
            Twine(index + 1) + "] * stride[" + Twine(index + 1) + "] (" +
            Twine(*minimum) + "), got " + Twine(stride5D[index]))
        .str();
  }
  return std::nullopt;
}

std::optional<std::string>
mlir::pto::getNZViewCompatibilityError(ArrayRef<int64_t> shape5D,
                                       ArrayRef<int64_t> stride5D,
                                       unsigned storageElemBytes) {
  if (shape5D.size() != kPTOLayoutRank || stride5D.size() != kPTOLayoutRank) {
    return "NZ layout requires rank-5 shape and stride";
  }

  auto c0 = getNZC0StorageElems(storageElemBytes);
  if (!c0) {
    return ("NZ layout requires a non-zero storage element byte size that "
            "divides 32, got " +
            Twine(storageElemBytes))
        .str();
  }

  // The leading dimension is the outer batch/expert axis in the pto-isa
  // canonical form [E, K/C0, N/16, 16, C0]. It is independent of the fractal.
  if (auto error = getNZInnerStructureError(shape5D, stride5D, *c0)) {
    return error;
  }
  if (auto error = getNZOuterStrideError(shape5D, stride5D, 1, *c0)) {
    return error;
  }
  return getNZOuterStrideError(shape5D, stride5D, 0, *c0);
}

static bool isNZViewCompatible5D(ArrayRef<int64_t> shape5D,
                                 ArrayRef<int64_t> stride5D,
                                 unsigned storageElemBytes) {
  return !getNZViewCompatibilityError(shape5D, stride5D, storageElemBytes);
}

std::optional<std::string>
mlir::pto::getNZSubviewCompatibilityError(ArrayRef<int64_t> sourceShape5D,
                                          ArrayRef<int64_t> offsets5D,
                                          ArrayRef<int64_t> sizes5D) {
  if (sourceShape5D.size() != kPTOLayoutRank ||
      offsets5D.size() != kPTOLayoutRank || sizes5D.size() != kPTOLayoutRank) {
    return "NZ partition requires rank-5 source, offsets, and sizes";
  }

  // d0 is the outer batch/expert axis in the canonical descriptor and can be
  // sliced independently.  Only the two dimensions that make up the NZ
  // fractal must remain complete in a subview.
  for (int64_t index : {3, 4}) {
    int64_t expectedSize = sourceShape5D[index];
    if (isDynamic(offsets5D[index]) || isDynamic(sizes5D[index]) ||
        isDynamic(expectedSize)) {
      return ("NZ partition must prove that d" + Twine(index) +
              " remains complete at compile time")
          .str();
    }
    if (offsets5D[index] != 0 || sizes5D[index] != expectedSize) {
      return ("NZ view cannot be partitioned inside a fractal: d" +
              Twine(index) + " must keep offset=0 and size=" +
              Twine(expectedSize) + " (got offset=" + Twine(offsets5D[index]) +
              ", size=" + Twine(sizes5D[index]) + ")")
          .str();
    }
  }
  return std::nullopt;
}

static bool isCanonicalNZRoot5D(ArrayRef<int64_t> shape5D,
                                ArrayRef<int64_t> stride5D,
                                unsigned storageElemBytes) {
  if (!allStatic(shape5D) || !allStatic(stride5D) ||
      !isNZViewCompatible5D(shape5D, stride5D, storageElemBytes)) {
    return false;
  }
  auto stride1 = checkedMultiply(shape5D[2], stride5D[2]);
  auto stride0 = checkedMultiply(shape5D[1], stride5D[1]);
  return stride1 && stride0 && stride5D[1] == *stride1 &&
         stride5D[0] == *stride0;
}

bool mlir::pto::isLayoutCompatible5D(Layout layout, ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> stride,
                                     unsigned storageElemBytes) {
  switch (layout) {
  case Layout::ND:
  case Layout::DN:
    // ND/DN are logical GlobalTensor interpretations, not compactness
    // promises. Existing inputs legitimately use interleaved and padded
    // outer views whose minor strides do not satisfy a dense 2-D recurrence.
    // Their explicit layout remains authoritative.
    return true;
  case Layout::MX_A_ZZ:
  case Layout::MX_B_NN:
  case Layout::MX_A_ND:
  case Layout::MX_A_DN:
  case Layout::MX_B_ND:
  case Layout::MX_B_DN:
    return true;
  case Layout::NZ: {
    auto padded = rightAlignTo5D(shape, stride);
    if (!padded || shape.size() != kPTOLayoutRank) {
      return false;
    }
    return isNZViewCompatible5D(padded->shape, padded->stride,
                                storageElemBytes);
  }
  }
  return false;
}

std::optional<Layout>
mlir::pto::inferLayout5D(ArrayRef<int64_t> shape, ArrayRef<int64_t> stride,
                         unsigned storageElemBytes,
                         std::optional<Layout> preferredMinor2D,
                         bool *isMinor2DAmbiguous) {
  if (isMinor2DAmbiguous) {
    *isMinor2DAmbiguous = false;
  }
  if (storageElemBytes == 0) {
    return std::nullopt;
  }
  auto padded = rightAlignTo5D(shape, stride);
  if (!padded || !allStatic(padded->shape) || !allStatic(padded->stride)) {
    return std::nullopt;
  }

  if (shape.size() == kPTOLayoutRank &&
      isCanonicalNZRoot5D(padded->shape, padded->stride, storageElemBytes) &&
      // Keep inference conservative for E > 1: blocked ND and NZ have the
      // same shape/stride structure, so only an explicit layout can disambiguate
      // a batched/expert view.
      padded->shape[0] == 1 && (padded->shape[1] > 1 || padded->shape[2] > 1)) {
    return Layout::NZ;
  }

  return inferMinor2DLayout(padded->shape[3], padded->shape[4],
                            padded->stride[3], padded->stride[4],
                            preferredMinor2D, isMinor2DAmbiguous);
}

// Follow the descriptor of a local allocation within its block. An earlier
// opaque use can change or expose the descriptor, so its extents stay unknown.
static std::pair<Value, Value> getLocalTileValidOperands(Value tile,
                                                        Operation *at) {
  auto alloc = tile.getDefiningOp<AllocTileOp>();
  Block *block = at->getBlock();
  if (!alloc || !block) {
    return {};
  }
  Block *allocBlock = alloc->getBlock();
  if (allocBlock != block) {
    return {};
  }
  Operation *lastUpdate = alloc.getOperation();
  std::pair<Value, Value> valid{alloc.getValidRow(), alloc.getValidCol()};
  for (Operation *user : tile.getUsers()) {
    Operation *anchor = block->findAncestorOpInBlock(*user);
    if (!anchor) {
      return {};
    }
    if (anchor == at || !anchor->isBeforeInBlock(at)) {
      continue;
    }
    if (isa<GetValidShapeOp, TRemOp, TRemSOp>(user)) {
      continue;
    }
    auto update = dyn_cast<SetValidShapeOp>(user);
    if (!update || anchor != user) {
      return {};
    }
    if (lastUpdate->isBeforeInBlock(user)) {
      lastUpdate = user;
      valid = {update.getValidRow(), update.getValidCol()};
    }
  }
  return valid;
}

std::array<std::optional<int64_t>, 2>
mlir::pto::getKnownTileValidShapeAt(Value tile, Operation *at) {
  std::array<std::optional<int64_t>, 2> result{};
  auto type = dyn_cast<TileBufType>(tile.getType());
  if (!type) {
    return result;
  }
  auto valid = type.getValidShape();
  if (valid.size() != 2) {
    return result;
  }
  for (size_t i = 0; i < result.size(); ++i) {
    if (!ShapedType::isDynamic(valid[i])) {
      result[i] = valid[i];
    }
  }
  if (result[0] && result[1]) {
    return result;
  }
  auto [row, col] = getLocalTileValidOperands(tile, at);
  if (!result[0] && row) {
    result[0] = getConstantIntValue(row);
  }
  if (!result[1] && col) {
    result[1] = getConstantIntValue(col);
  }
  return result;
}

std::optional<int64_t>
mlir::pto::getA2A3RemainderScratchCols(Value dst, Operation *at,
                                      RemPrecision precision) {
  constexpr int64_t kMaxTileCols = std::numeric_limits<int32_t>::max();
  auto type = dyn_cast<TileBufType>(dst.getType());
  if (!type) {
    return std::nullopt;
  }
  const auto shape = type.getShape();
  const auto validShape = type.getValidShape();
  const bool rankTwo = shape.size() == 2 && validShape.size() == 2;
  if (!rankTwo) {
    return std::nullopt;
  }
  // ISA Tile::RowStride is the physical column count for row-major tiles.
  // Bound it to the signed width of the ISA Tile template before arithmetic.
  const int64_t stride = shape[1];
  if (stride <= 0 || stride > kMaxTileCols) {
    return std::nullopt;
  }

  constexpr int64_t kVectorElements = 64; // One 256-byte repeat of i32 or f32.
  constexpr int64_t kBlockElements = 8;   // One 32-byte block of i32 or f32.
  constexpr int64_t kMaskBitsPerBlock = 256;
  const int64_t candidateCols =
      static_cast<int64_t>(llvm::alignTo(stride, kVectorElements));
  const int64_t baseCols =
      candidateCols + kBlockElements * llvm::divideCeil(stride, kMaskBitsPerBlock) +
      kBlockElements;
  int64_t requiredCols = baseCols;
  if (precision == RemPrecision::HighPrecision && type.getElementType().isF32()) {
    constexpr int64_t kMasksPerBlock = 4;
    // The pinned PTO-ISA (ece75ec8) needs no compensation window. Reserve
    // extra space for the final high-precision iteration in 7db8bb0a.
    constexpr int64_t kCompensationVectorsPerInput = 8;
    constexpr int64_t kFusedChunkVectors = 8;
    const int64_t width = getKnownTileValidShapeAt(dst, at)[1].value_or(stride);
    if (width < 0 || width > stride) {
      return std::nullopt;
    }
    const int64_t capacity = std::max<int64_t>(width / kVectorElements, 1);
    const int64_t maskCols =
        kBlockElements * llvm::divideCeil(capacity, kMasksPerBlock);
    const int64_t stateCols = capacity == 1 ? kVectorElements : 0;
    const int64_t compensationCols =
        kCompensationVectorsPerInput * kVectorElements *
        std::min(capacity, kFusedChunkVectors);
    const int64_t highPrecisionCols =
        3 * kVectorElements * capacity + maskCols + kBlockElements + stateCols +
        compensationCols;
    requiredCols = std::max(baseCols, highPrecisionCols);
  }
  if (requiredCols > kMaxTileCols) {
    return std::nullopt;
  }
  return requiredCols;
}
