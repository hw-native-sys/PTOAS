// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
// [P0 新增] 引入副作用接口和 PTO 接口
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <limits>
#include <optional>

#define DEBUG_TYPE "pto-ir-translator"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr size_t kTileRank2D = 2;
constexpr unsigned kMemoryEffectInlineCapacity = 4;
constexpr llvm::StringLiteral kTileOpEffectsAttr = "pto.tileop.effects";

using MemoryEffectVector =
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
                kMemoryEffectInlineCapacity>;

static uint64_t getStaticBufferSizeInBytes(ArrayRef<int64_t> shape,
                                           Type elementType) {
  uint64_t size = pto::getPTOStorageElemByteSize(elementType);
  if (size == 0) {
    return 0;
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return 0;
    }
    size *= static_cast<uint64_t>(dim);
  }
  return size;
}

// A conv tile reserves buffer_size elements of storage in L1 (the type's
// declared capacity, not the logical shape product).
static uint64_t getConvTileFootprintBytes(pto::ConvTileBufType type) {
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(type.getElementType());
  int64_t bufferSize = type.getBufferSizeValue();
  if (elemBytes == 0 || bufferSize <= 0 ||
      static_cast<uint64_t>(bufferSize) >
          std::numeric_limits<uint64_t>::max() / elemBytes) {
    return 0;
  }
  return static_cast<uint64_t>(bufferSize) * elemBytes;
}

static uint64_t getTileBufferFootprintBytes(pto::TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0) {
    return 0;
  }
  if (type.getCompactModeI32() !=
      static_cast<int32_t>(pto::CompactMode::RowPlusOne)) {
    return getStaticBufferSizeInBytes(shape, type.getElementType());
  }
  if (shape.size() != kTileRank2D ||
      llvm::is_contained(shape, ShapedType::kDynamic)) {
    return 0;
  }

  bool rowMajor =
      type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  uint64_t major = static_cast<uint64_t>(rowMajor ? shape[0] : shape[1]);
  uint64_t minor = static_cast<uint64_t>(rowMajor ? shape[1] : shape[0]);
  if (major == 0 || minor == 0) {
    return 0;
  }
  return ((major - 1) * (minor + 1) + minor) * elemBytes;
}

static pto::AddressSpace getTileAddressSpace(pto::TileBufType type) {
  if (auto attr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace())) {
    return attr.getAddressSpace();
  }
  return pto::AddressSpace::MAT;
}

static void
appendUniqueMemInfo(SmallVectorImpl<std::unique_ptr<BaseMemInfo>> &infos,
                    std::unique_ptr<BaseMemInfo> candidate) {
  if (!candidate) {
    return;
  }
  if (llvm::any_of(infos, [&](const std::unique_ptr<BaseMemInfo> &info) {
        return info && *info == *candidate;
      })) {
    return;
  }
  infos.emplace_back(std::move(candidate));
}

} // namespace

// 按 arith cast 语义重算整数值：trunci 截断、extui 零扩展、extsi 和
// index_cast 符号扩展或截断（index 按 64 位处理）。
static llvm::APInt applyIntegerCast(Operation *castOp, const llvm::APInt &in) {
  Type resultType = castOp->getResult(0).getType();
  unsigned width = resultType.isIndex() ? IndexType::kInternalStorageBitWidth
                                        : resultType.getIntOrFloatBitWidth();
  if (isa<arith::TruncIOp>(castOp)) {
    return in.trunc(width);
  }
  if (isa<arith::ExtUIOp>(castOp)) {
    return in.zext(width);
  }
  return in.sextOrTrunc(width);
}

// 只穿透整数 cast，并按每一步的语义精确计算结果；直接剥掉 cast 会把
// 例如 256 -> trunci i8 的 0 误认成 256，进而错判地址或窗口不重叠。
static bool getConstIndexValue(Value value, int64_t &out) {
  SmallVector<Operation *, 4> casts;
  while (Operation *def = value.getDefiningOp()) {
    if (!isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::TruncIOp>(def)) {
      break;
    }
    casts.push_back(def);
    value = def->getOperand(0);
  }
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  auto intAttr =
      constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr();
  if (!intAttr) {
    return false;
  }
  llvm::APInt bits = intAttr.getValue();
  for (Operation *castOp : llvm::reverse(casts)) {
    bits = applyIntegerCast(castOp, bits);
  }
  if (!bits.isSignedIntN(64)) {
    return false;
  }
  out = bits.getSExtValue();
  return true;
}

static std::optional<uint64_t> getKnownPhysicalAddress(Value value) {
  int64_t address = 0;
  if (!getConstIndexValue(value, address) || address < 0) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(address);
}

static bool isLocalAddressSpace(pto::AddressSpace space) {
  return space != pto::AddressSpace::GM &&
         space != pto::AddressSpace::Zero;
}

static pto::AddressSpace getPointerLikeAddressSpace(Type type) {
  if (auto space = pto::getPTOAddressSpaceAttr(type)) {
    return space.getAddressSpace();
  }
  return pto::AddressSpace::GM;
}

static bool isStaticRank2Shape(ArrayRef<int64_t> shape) {
  return shape.size() == kTileRank2D &&
         llvm::none_of(shape, [](int64_t dim) {
           return dim == ShapedType::kDynamic;
         });
}

static int64_t getTileMajorStride(pto::TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  bool rowMajor =
      type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t stride = rowMajor ? cols : rows;
  if (type.getCompactModeI32() == static_cast<int32_t>(pto::CompactMode::RowPlusOne)) {
    ++stride;
  }
  return stride;
}

// A full-height, block-aligned ACC column window is one contiguous byte
// range. Other boxed rectangles must retain the conservative parent range.
struct AccColumnRange {
  uint64_t offset;
  uint64_t size;
  uint64_t sourceSize;
};

static bool hasContiguousAccColumnLayout(pto::TileBufType type) {
  Type elementType = type.getElementType();
  return getTileAddressSpace(type) == pto::AddressSpace::ACC &&
         (elementType.isF32() || elementType.isInteger(32)) &&
         type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::ColMajor) &&
         type.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor) &&
         type.getSFractalSizeI32() == 1024 &&
         type.getCompactModeI32() == static_cast<int32_t>(pto::CompactMode::Null);
}

static bool isAlignedAccColumnWindow(pto::SubViewOp op, int64_t rowOffset,
                                     int64_t colOffset) {
  constexpr int64_t blockExtent = 16;
  auto type = op.getSource().getType();
  int64_t rows = type.getShape()[0];
  int64_t cols = type.getShape()[1];
  int64_t rowSize = cast<IntegerAttr>(op.getSizes()[0]).getInt();
  int64_t colSize = cast<IntegerAttr>(op.getSizes()[1]).getInt();
  if (rows <= 0 || cols <= 0 || rows % blockExtent != 0 ||
      cols % blockExtent != 0 || rowOffset != 0 || rowSize != rows ||
      colOffset < 0 || colOffset > cols || colSize <= 0 ||
      colSize > cols - colOffset || colOffset % blockExtent != 0 ||
      colSize % blockExtent != 0) {
    return false;
  }
  return true;
}

static std::optional<AccColumnRange>
getContiguousAccColumnRange(pto::SubViewOp op) {
  constexpr uint64_t elemBytes = 4;
  auto type = op.getSource().getType();
  if (!hasContiguousAccColumnLayout(type) ||
      !isStaticRank2Shape(type.getShape()) ||
      op.getOffsets().size() != kTileRank2D ||
      !op.getSizes() || op.getSizes().size() != kTileRank2D) {
    return std::nullopt;
  }
  IntegerAttr rowAttr, colAttr;
  if (!matchPattern(op.getOffsets()[0], m_Constant(&rowAttr)) ||
      !matchPattern(op.getOffsets()[1], m_Constant(&colAttr)) ||
      !isAlignedAccColumnWindow(op, rowAttr.getInt(), colAttr.getInt())) {
    return std::nullopt;
  }
  int64_t rows = type.getShape()[0];
  int64_t cols = type.getShape()[1];
  int64_t colOffset = colAttr.getInt();
  int64_t colSize = cast<IntegerAttr>(op.getSizes()[1]).getInt();
  uint64_t maxBytes = std::numeric_limits<uint64_t>::max();
  if (static_cast<uint64_t>(rows) > maxBytes / elemBytes) {
    return std::nullopt;
  }
  uint64_t columnBytes = static_cast<uint64_t>(rows) * elemBytes;
  if (static_cast<uint64_t>(cols) > maxBytes / columnBytes) {
    return std::nullopt;
  }
  return AccColumnRange{static_cast<uint64_t>(colOffset) * columnBytes,
                        static_cast<uint64_t>(colSize) * columnBytes,
                        static_cast<uint64_t>(cols) * columnBytes};
}

// A single inherited base can also be a conservative envelope. Only refine
// sources rooted at an allocation through already-proven column windows;
// notably, a static child of a dynamic/partial-row view is not an exact view.
static Value getExactAccColumnRoot(Value source) {
  while (auto parent = source.getDefiningOp<pto::SubViewOp>()) {
    if (!getContiguousAccColumnRange(parent)) {
      return {};
    }
    source = parent.getSource();
  }
  return source.getDefiningOp<pto::AllocTileOp>() ? source : Value();
}

static bool canRefineAccColumnParents(
    const SmallVector<std::unique_ptr<BaseMemInfo>> &parents, Value root,
    const AccColumnRange &range) {
  if (!root || parents.empty()) {
    return false;
  }
  for (const auto &parent : parents) {
    if (!parent || parent->rootBuffer != root ||
        parent->baseAddresses.size() != 1 ||
        parent->allocateSize != range.sourceSize ||
        range.sourceSize > std::numeric_limits<uint64_t>::max() -
                               parent->baseAddresses.front()) {
      return false;
    }
  }
  return true;
}

static bool updateAccColumnAliasBufferInfo(pto::SubViewOp op,
                                           const AccColumnRange &range,
                                           Buffer2MemInfoMap &buffer2MemInfoMap) {
  Value source = op.getSource();
  auto &parents = buffer2MemInfoMap[source];
  if (!canRefineAccColumnParents(parents, getExactAccColumnRoot(source), range)) {
    return false;
  }
  Value result = op.getResult();
  auto &resultInfos = buffer2MemInfoMap[result];
  for (const auto &parent : buffer2MemInfoMap[source]) {
    auto info = parent->clone(result);
    info->baseAddresses = {parent->baseAddresses.front() + range.offset};
    info->allocateSize = range.size;
    resultInfos.emplace_back(std::move(info));
  }
  return true;
}

// boxed Mat Tile 中一个矩形窗口的物理字节分段（相对 Tile 基址）。
struct BoxedTileRegion {
  SmallVector<uint64_t> offsets;
  uint64_t segmentBytes = 0;
};

// boxed INT8 Tile 的外层分形块尺寸；nz 表示 Nz 布局，否则为 Zn 布局。
// 分形内层尺寸均为 2 的幂，以 log2 记录，用移位和掩码代替除法和取余。
struct BoxedTileLayout {
  bool nz = false;
  int64_t rows = 0;
  int64_t cols = 0;
  unsigned innerRowsLog2 = 0;
  unsigned innerColsLog2 = 0;
};

// 仅接受 Mat 上普通 INT8、512B 分形、非 compact 的 Nz/Zn 静态 Tile。
static std::optional<BoxedTileLayout> getBoxedI8Layout(pto::TileBufType type) {
  if (!type || getTileAddressSpace(type) != pto::AddressSpace::MAT ||
      !type.getElementType().isInteger(8) ||
      !isStaticRank2Shape(type.getShape()) ||
      type.getSFractalSizeI32() != 512 ||
      type.getCompactModeI32() !=
          static_cast<int32_t>(pto::CompactMode::Null)) {
    return std::nullopt;
  }
  int32_t bl = type.getBLayoutValueI32();
  int32_t sl = type.getSLayoutValueI32();
  bool nz = bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor);
  bool zn = bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor);
  if (!nz && !zn) {
    return std::nullopt;
  }
  constexpr unsigned kInnerShortLog2 = 4; // 16
  constexpr unsigned kInnerLongLog2 = 5;  // 32
  BoxedTileLayout layout;
  layout.nz = nz;
  layout.rows = type.getShape()[0];
  layout.cols = type.getShape()[1];
  layout.innerRowsLog2 = nz ? kInnerShortLog2 : kInnerLongLog2;
  layout.innerColsLog2 = nz ? kInnerLongLog2 : kInnerShortLog2;
  return layout;
}

// 一维窗口 [offset, offset + size) 落在 [0, extent) 内且按 2^innerLog2 对齐。
static bool isAlignedWindow(int64_t offset, int64_t size, int64_t extent,
                            unsigned innerLog2) {
  const int64_t mask = (int64_t{1} << innerLog2) - 1;
  return extent > 0 && offset >= 0 && size > 0 && offset <= extent - size &&
         (extent & mask) == 0 && (offset & mask) == 0 && (size & mask) == 0;
}

// 首尾相接的分段合并为一个连续范围，避免等价访问因表示不同而影响
// 后续同步去重。
static void mergeContiguousSegments(BoxedTileRegion &region) {
  for (size_t i = 1; i < region.offsets.size(); ++i) {
    if (region.offsets[i] != region.offsets[i - 1] + region.segmentBytes) {
      return;
    }
  }
  if (region.offsets.size() > 1) {
    region.segmentBytes *= region.offsets.size();
    region.offsets.resize(1);
  }
}

// 仅证明 Mat 上普通 INT8、512B 分形的 Nz/Zn 布局，且窗口沿完整分形边界切分：
//   Nz: 每个外层块列（32 列）内各行连续存放，窗口每块列贡献一段；
//   Zn: 每个外层块行（32 行）内各列连续存放，窗口每块行贡献一段。
// 偏移与 pto-isa GetTileOffset 一致。其它 dtype/compact/未对齐窗口返回 nullopt，
// 由调用方保守回退。
static std::optional<BoxedTileRegion>
getBoxedTileRegion(pto::TileBufType type, int64_t rowOffset, int64_t colOffset,
                   int64_t rowSize, int64_t colSize) {
  auto layout = getBoxedI8Layout(type);
  if (!layout ||
      !isAlignedWindow(rowOffset, rowSize, layout->rows,
                       layout->innerRowsLog2) ||
      !isAlignedWindow(colOffset, colSize, layout->cols,
                       layout->innerColsLog2)) {
    return std::nullopt;
  }
  // 段数上限只用于约束编译期开销。
  constexpr int64_t kMaxSegments = 4096;
  // 以下各量已由 isAlignedWindow 保证非负且按分形对齐，移位即精确除法。
  bool nz = layout->nz;
  unsigned rowLog2 = layout->innerRowsLog2;
  unsigned colLog2 = layout->innerColsLog2;
  int64_t innerRows = int64_t{1} << rowLog2;
  int64_t innerCols = int64_t{1} << colLog2;
  int64_t count = nz ? (colSize >> colLog2) : (rowSize >> rowLog2);
  if (count > kMaxSegments) {
    return std::nullopt;
  }

  BoxedTileRegion region;
  region.segmentBytes =
      static_cast<uint64_t>(nz ? rowSize * innerCols : colSize * innerRows);
  region.offsets.reserve(static_cast<size_t>(count));
  for (int64_t block = 0; block < count; ++block) {
    int64_t offset =
        nz ? ((colOffset >> colLog2) + block) * layout->rows * innerCols +
                 rowOffset * innerCols
           : ((rowOffset >> rowLog2) + block) * layout->cols * innerRows +
                 colOffset * innerRows;
    region.offsets.push_back(static_cast<uint64_t>(offset));
  }
  mergeContiguousSegments(region);
  return region;
}

// 嵌套 subview 可能继承更大父 Tile 的物理 stride，只有直接来自 alloc_tile
// 的源 Tile 才能用自身 shape 作为物理布局。
static std::optional<BoxedTileRegion>
getBoxedSubViewRegion(pto::SubViewOp op, pto::TileBufType sourceType) {
  if (!op.getSource().getDefiningOp<pto::AllocTileOp>() ||
      op.getOffsets().size() != kTileRank2D) {
    return std::nullopt;
  }
  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != kTileRank2D) {
    return std::nullopt;
  }
  int64_t rowOffset = 0;
  int64_t colOffset = 0;
  if (!getConstIndexValue(op.getOffsets()[0], rowOffset) ||
      !getConstIndexValue(op.getOffsets()[1], colOffset)) {
    return std::nullopt;
  }
  return getBoxedTileRegion(sourceType, rowOffset, colOffset,
                            cast<IntegerAttr>(sizesAttr[0]).getInt(),
                            cast<IntegerAttr>(sizesAttr[1]).getInt());
}

static std::optional<SmallVector<uint64_t>>
getPtoSubViewBaseAddresses(pto::SubViewOp op, pto::TileBufType sourceType,
                           int64_t elemBytes) {
  if (!isStaticRank2Shape(sourceType.getShape())) {
    return std::nullopt;
  }
  if (auto region = getBoxedSubViewRegion(op, sourceType)) {
    return region->offsets;
  }
  if (sourceType.getSLayoutValueI32() !=
      static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return std::nullopt;
  }
  if (op.getOffsets().size() != kTileRank2D) {
    return std::nullopt;
  }

  int64_t rowOffset = 0;
  int64_t colOffset = 0;
  if (!getConstIndexValue(op.getOffsets()[0], rowOffset) ||
      !getConstIndexValue(op.getOffsets()[1], colOffset)) {
    return std::nullopt;
  }
  if (rowOffset < 0 || colOffset < 0) {
    return std::nullopt;
  }

  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != kTileRank2D) {
    return std::nullopt;
  }
  int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (rowSize <= 0 || colSize <= 0) {
    return std::nullopt;
  }

  bool rowMajor =
      sourceType.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t majorStride = getTileMajorStride(sourceType);

  SmallVector<uint64_t> addresses;
  if (rowMajor) {
    addresses.reserve(static_cast<size_t>(rowSize));
    for (int64_t row = 0; row < rowSize; ++row) {
      int64_t elemOffset = (rowOffset + row) * majorStride + colOffset;
      addresses.push_back(static_cast<uint64_t>(elemOffset * elemBytes));
    }
  } else {
    addresses.reserve(static_cast<size_t>(colSize));
    for (int64_t col = 0; col < colSize; ++col) {
      int64_t elemOffset = (colOffset + col) * majorStride + rowOffset;
      addresses.push_back(static_cast<uint64_t>(elemOffset * elemBytes));
    }
  }

  return addresses;
}

// Keeps the window arithmetic below inside 64 bits without per-expression
// overflow checks. GM views are far smaller than this in every real kernel.
static constexpr int64_t kMaxModelledViewExtent = 1LL << 31;

static uint64_t getGMViewElemBytes(Type type) {
  if (auto viewType = dyn_cast<pto::TensorViewType>(type)) {
    return pto::getPTOStorageElemByteSize(viewType.getElementType());
  }
  if (auto viewType = dyn_cast<pto::PartitionTensorViewType>(type)) {
    return pto::getPTOStorageElemByteSize(viewType.getElementType());
  }
  return 0;
}

static bool getModelledConstIndex(Value value, int64_t &out, int64_t lowerBound) {
  return getConstIndexValue(value, out) && out >= lowerBound &&
         out < kMaxModelledViewExtent;
}

/// Seeds a GM window from `pto.make_tensor_view`: the whole tensor, with the
/// innermost dimension as the contiguous run. Returns nullopt whenever any
/// extent is dynamic or the innermost dimension is not unit-strided, which
/// leaves the view on the existing conservative path.
static std::optional<GMWindow>
buildMakeTensorViewWindow(pto::MakeTensorViewOp op) {
  uint64_t elemBytes = getGMViewElemBytes(op.getResult().getType());
  if (elemBytes == 0) {
    return std::nullopt;
  }
  ValueRange shape = op.getShape();
  ValueRange strides = op.getStrides();
  if (shape.empty() || shape.size() != strides.size()) {
    return std::nullopt;
  }

  SmallVector<int64_t> shapeConst(shape.size());
  SmallVector<int64_t> strideConst(strides.size());
  for (auto [i, dim] : llvm::enumerate(shape)) {
    if (!getModelledConstIndex(dim, shapeConst[i], /*lowerBound=*/1)) {
      return std::nullopt;
    }
  }
  for (auto [i, stride] : llvm::enumerate(strides)) {
    if (!getModelledConstIndex(stride, strideConst[i], /*lowerBound=*/1)) {
      return std::nullopt;
    }
  }
  if (strideConst.back() != 1) {
    return std::nullopt;
  }

  GMWindow window;
  window.contigBytes = static_cast<uint64_t>(shapeConst.back()) * elemBytes;
  window.outerDims.reserve(shapeConst.size() - 1);
  for (size_t i = 0; i + 1 < shapeConst.size(); ++i) {
    window.outerDims.emplace_back(
        static_cast<uint64_t>(strideConst[i]) * elemBytes,
        static_cast<uint64_t>(shapeConst[i]));
  }
  return window;
}

/// Composes a `pto.partition_view` onto its source window. Composition rather
/// than replacement is what makes nested partition views land on the right
/// bytes.
static std::optional<GMWindow>
composePartitionViewWindow(const GMWindow &parent, pto::PartitionViewOp op,
                           uint64_t elemBytes) {
  ValueRange offsets = op.getOffsets();
  ValueRange sizes = op.getSizes();
  size_t rank = parent.outerDims.size() + 1;
  if (offsets.size() != rank || sizes.size() != rank) {
    return std::nullopt;
  }

  SmallVector<int64_t> offsetConst(rank);
  SmallVector<int64_t> sizeConst(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (!getModelledConstIndex(offsets[i], offsetConst[i], /*lowerBound=*/0) ||
        !getModelledConstIndex(sizes[i], sizeConst[i], /*lowerBound=*/1)) {
      return std::nullopt;
    }
  }

  GMWindow window;
  window.contigBytes = static_cast<uint64_t>(sizeConst.back()) * elemBytes;
  uint64_t innerOffsetBytes =
      static_cast<uint64_t>(offsetConst.back()) * elemBytes;
  // A slice reaching past its parent's run would mean the modelled strides do
  // not describe the access; stay conservative instead of guessing.
  if (innerOffsetBytes + window.contigBytes > parent.contigBytes) {
    return std::nullopt;
  }
  window.byteOffset = parent.byteOffset + innerOffsetBytes;

  window.outerDims.reserve(parent.outerDims.size());
  for (size_t i = 0; i + 1 < rank; ++i) {
    auto [stride, count] = parent.outerDims[i];
    if (static_cast<uint64_t>(offsetConst[i]) +
            static_cast<uint64_t>(sizeConst[i]) >
        count) {
      return std::nullopt;
    }
    window.byteOffset += static_cast<uint64_t>(offsetConst[i]) * stride;
    window.outerDims.emplace_back(stride, static_cast<uint64_t>(sizeConst[i]));
  }
  return window;
}

namespace {

static func::FuncOp lookupSyncHelper(func::CallOp callOp) {
  auto module = callOp->getParentOfType<ModuleOp>();
  if (!module || callOp.getCallee().empty()) {
    return {};
  }
  auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
  if (!callee) {
    return {};
  }
  if (!callee->hasAttr("pto.tileop.helper") &&
      !callee->hasAttr("pto.ptodsl.subkernel_helper")) {
    return {};
  }
  return callee;
}

static std::optional<pto::PipelineType> getSyncHelperPipe(func::FuncOp callee) {
  if (auto roleAttr = callee->getAttrOfType<mlir::StringAttr>(
          "pto.ptodsl.subkernel_helper")) {
    return llvm::StringSwitch<std::optional<pto::PipelineType>>(
               roleAttr.getValue())
        .Case("cube", pto::PipelineType::PIPE_M)
        .Case("simd", pto::PipelineType::PIPE_V)
        .Default(std::nullopt);
  }

  auto kindAttr = callee->getAttrOfType<mlir::StringAttr>("pto.tileop.kind");
  if (!kindAttr) {
    return std::nullopt;
  }
  return llvm::StringSwitch<std::optional<pto::PipelineType>>(
             kindAttr.getValue())
      .Case("cube", pto::PipelineType::PIPE_M)
      .Case("vector", pto::PipelineType::PIPE_V)
      .Default(std::nullopt);
}

static bool isSyncHelperMemoryOperand(Type type) {
  return isa<pto::PtrType, pto::TileBufType, pto::ConvTileBufType,
             pto::TensorViewType, pto::PartitionTensorViewType>(type);
}

static pto::TCoreType getSyncHelperCoreType(pto::PipelineType pipe) {
  return pipe == pto::PipelineType::PIPE_M ? pto::TCoreType::CUBE
                                           : pto::TCoreType::VECTOR;
}

} // namespace

// ============================================================================
// 1. 构建入口
// ============================================================================
void PTOIRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  RecursionIR(&funcRegion);
}

// ============================================================================
// 2. 更新 Kernel 参数内存信息
// ============================================================================
void PTOIRTranslator::UpdateKernelArgMemInfo() {
  for (Value funcArg : func_.getArguments()) {
    Type argType = funcArg.getType();
    pto::AddressSpace space = pto::AddressSpace::Zero;
    uint64_t sizeInBytes = 0;
    SmallVector<uint64_t> baseAddresses{0};

    if (auto ptrType = dyn_cast<pto::PtrType>(argType)) {
      space = ptrType.getMemorySpace().getAddressSpace();
    } else if (auto tileType = dyn_cast<pto::TileBufType>(argType)) {
      space = getTileAddressSpace(tileType);
      sizeInBytes = getTileBufferFootprintBytes(tileType);
    } else if (auto convType = dyn_cast<pto::ConvTileBufType>(argType)) {
      // A conv tile handed in as a kernel argument participates in the same
      // dependency tracking as a locally allocated one: without registering
      // it here, a TLOAD (MTE2) -> TIMG2COL (MTE1) pair through the argument
      // gets no barrier at all.
      space = pto::AddressSpace::MAT;
      if (auto attr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(convType.getMemorySpace())) {
        space = attr.getAddressSpace();
      }
      sizeInBytes = getConvTileFootprintBytes(convType);
    } else if (auto multiType = dyn_cast<pto::MultiTileBufType>(argType)) {
      pto::TileBufType slotType = multiType.getSlotType();
      space = getTileAddressSpace(slotType);
      sizeInBytes = getTileBufferFootprintBytes(slotType);
      baseAddresses.clear();
      for (uint32_t slot = 0; slot < multiType.getCount(); ++slot) {
        baseAddresses.push_back(sizeInBytes == 0 ? 0 : slot * sizeInBytes);
      }
    } else if (auto viewType = dyn_cast<pto::TensorViewType>(argType)) {
      space = pto::AddressSpace::GM;
      sizeInBytes = getStaticBufferSizeInBytes(viewType.getShape(),
                                               viewType.getElementType());
    } else if (auto viewType =
                   dyn_cast<pto::PartitionTensorViewType>(argType)) {
      space = pto::AddressSpace::GM;
      sizeInBytes = getStaticBufferSizeInBytes(viewType.getShape(),
                                               viewType.getElementType());
    } else {
      continue;
    }

    buffer2MemInfoMap_[funcArg].emplace_back(std::make_unique<BaseMemInfo>(
        funcArg, funcArg, space, std::move(baseAddresses), sizeInBytes));
  }
}

// ============================================================================
// 3. 递归遍历 IR (核心分发逻辑)
// ============================================================================

// 各分发函数返回 std::nullopt 表示该 op 不属于本类别，需继续尝试后续类别；
// 返回 WalkResult 表示已匹配并处理完毕（保持原 if/else-if 链的互斥语义）。

// --- Case A: 内存分配 (AllocTile / AllocMultiTile / Declare*) ---
std::optional<WalkResult>
PTOIRTranslator::dispatchAllocOp(Operation *op) {
  if (auto allocOp = dyn_cast<pto::AllocTileOp>(op)) {
    if (failed(UpdateAllocTileOpMemInfo(allocOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }
  if (auto allocMultiOp = dyn_cast<pto::AllocMultiTileOp>(op)) {
    if (failed(UpdateAllocMultiTileOpMemInfo(allocMultiOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }
  if (auto declareOp = dyn_cast<pto::DeclareTileOp>(op)) {
    if (failed(UpdateDeclareTileOpMemInfo(declareOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }
  if (auto declareGlobalOp = dyn_cast<pto::DeclareGlobalOp>(op)) {
    if (failed(UpdateDeclareGlobalOpMemInfo(declareGlobalOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }
  return std::nullopt;
}

// --- Case B: 别名/视图操作 ---
std::optional<WalkResult>
PTOIRTranslator::dispatchAliasViewOp(Operation *op) {
  if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
    UpdateMakeTensorViewAliasBufferInfo(makeViewOp);
  } else if (auto subViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
    UpdatePartitionViewAliasBufferInfo(subViewOp);
  } else if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(op)) {
    UpdateAliasBufferInfo(addPtrOp.getResult(), addPtrOp.getPtr());
  } else if (auto castPtrOp = dyn_cast<pto::CastPtrOp>(op)) {
    if (isa<pto::PtrType>(castPtrOp.getInput().getType()) &&
        isa<pto::PtrType>(castPtrOp.getResult().getType())) {
      UpdateAliasBufferInfo(castPtrOp.getResult(), castPtrOp.getInput());
    } else if (isa<pto::PtrType>(castPtrOp.getInput().getType()) &&
               isa<IntegerType>(castPtrOp.getResult().getType())) {
      UpdateAliasBufferInfo(castPtrOp.getResult(), castPtrOp.getInput());
    } else if (isa<IntegerType>(castPtrOp.getInput().getType()) &&
               isa<pto::PtrType>(castPtrOp.getResult().getType()) &&
               failed(UpdateIntegerToPtrCastMemInfo(castPtrOp))) {
      return WalkResult::interrupt();
    }
  } else if (auto subViewOp = dyn_cast<pto::SubViewOp>(op)) {
    UpdateTileSubViewAliasBufferInfo(subViewOp);
  } else if (auto multiGet = dyn_cast<pto::MultiTileGetOp>(op)) {
    UpdateMultiTileGetAliasBufferInfo(multiGet);
  } else if (auto reshape = dyn_cast<pto::TReshapeOp>(op)) {
    UpdateAliasBufferInfo(reshape.getResult(), reshape.getSrc());
  } else if (auto bitcast = dyn_cast<pto::BitcastOp>(op)) {
    UpdateAliasBufferInfo(bitcast.getResult(), bitcast.getSrc());
  } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
    UpdateAliasBufferInfo(select.getResult(), select.getTrueValue());
    UpdateAliasBufferInfo(select.getResult(), select.getFalseValue());
  } else {
    return std::nullopt;
  }
  return WalkResult::advance();
}

// --- Case C/D: 控制流 (SCF)、同步宏指令、调用与计算指令 ---
std::optional<WalkResult>
PTOIRTranslator::dispatchControlAndComputeOp(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    UpdateForOpInfo(forOp);
    return WalkResult::skip();
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    UpdateWhileOpInfo(whileOp);
    return WalkResult::skip();
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    UpdateIfOpInfo(ifOp);
    return WalkResult::skip();
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    UpdateYieldOpInfo(yieldOp);
  } else if (getSyncMacroModel(op)) {
    UpdateMacroOpInfo(op);
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    UpdateHelperCallInfo(callOp);
  } else if (isa<pto::OpPipeInterface>(op)) {
    // --- Case D: 带有 OpPipeInterface 的计算/搬运指令 ---
    UpdatePTOOpInfo(op);
  } else {
    return std::nullopt;
  }
  return WalkResult::advance();
}

void PTOIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([this](Operation *op) {
    // 保持原有 if/else-if 链的互斥匹配顺序：A 内存分配 → B 别名/视图 →
    // C/D 控制流与计算指令，任一类别命中后不再尝试后续类别。
    if (auto allocResult = dispatchAllocOp(op)) {
      return *allocResult;
    }
    if (auto aliasResult = dispatchAliasViewOp(op)) {
      return *aliasResult;
    }
    if (auto controlResult = dispatchControlAndComputeOp(op)) {
      return *controlResult;
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PTO InjectSync Traverse IR Failed!");
  }
}

// ============================================================================
// 4. 处理本地 tile 分配
// ============================================================================
LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
  Value res = op.getResult();

  auto tileType = dyn_cast<pto::TileBufType>(res.getType());
  uint64_t sizeInBytes = 0;
  uint64_t baseAddr = 0;
  std::optional<uint64_t> knownPhysicalAddress;

  // If alloc_tile carries an explicit address, record it when it's a constant.
  if (Value addr = op.getAddr()) {
    knownPhysicalAddress = getKnownPhysicalAddress(addr);
    if (knownPhysicalAddress) {
      baseAddr = *knownPhysicalAddress;
    }
  }

  // 1. 计算大小
  if (tileType) {
    sizeInBytes = getTileBufferFootprintBytes(tileType);
    if (sizeInBytes == 0) {
      return failure();
    }
  } else if (auto convType = dyn_cast<pto::ConvTileBufType>(res.getType())) {
    // Without a footprint the overlap analysis treats every conv tile as
    // size 0 and conservatively syncs non-overlapping buffers.
    sizeInBytes = getConvTileFootprintBytes(convType);
    if (sizeInBytes == 0) {
      return failure();
    }
  }

  // 2. 解析地址空间
  // 默认设为 MAT (Matrix Buffer)，但优先读取 Type 中的属性
  pto::AddressSpace space = pto::AddressSpace::MAT;

  if (tileType) {
      if (auto attr = tileType.getMemorySpace()) {
          // 尝试转换为 PTO 的 AddressSpaceAttr
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
              space = ptoAttr.getAddressSpace();
          }
      }
  }

  // An unresolved explicit local address may overlap any allocation in the
  // same address space. Unlike an address-free allocation, its SSA root does
  // not identify independent storage (issue #1620).
  const bool aliasesUnknownRange = op.getAddr() && !knownPhysicalAddress && isLocalAddressSpace(space);

  // 3. 注册 Buffer 信息
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res, res, space, SmallVector<uint64_t>{baseAddr}, sizeInBytes,
      knownPhysicalAddress.has_value() && isLocalAddressSpace(space), aliasesUnknownRange);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateAllocMultiTileOpMemInfo(pto::AllocMultiTileOp op) {
  Value result = op.getResult();
  auto multiType = op.getResult().getType();
  pto::TileBufType slotType = multiType.getSlotType();

  uint64_t slotBytes = getTileBufferFootprintBytes(slotType);
  if (slotBytes == 0) {
    return failure();
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(
          slotType.getMemorySpace())) {
    space = attr.getAddressSpace();
  }

  SmallVector<uint64_t> addresses;
  bool hasKnownAddresses = false;
  if (auto planned = op->getAttrOfType<DenseI64ArrayAttr>(
          pto::kPtoMultiBufferAddrsAttrName)) {
    if (planned.size() != multiType.getCount()) {
      return op.emitError("planned address count does not match slot count");
    }
    for (int64_t address : planned.asArrayRef()) {
      addresses.push_back(static_cast<uint64_t>(address));
    }
    hasKnownAddresses = true;
  } else if (Value base = op.getAddr()) {
    if (std::optional<uint64_t> knownBase = getKnownPhysicalAddress(base)) {
      for (uint32_t slot = 0; slot < multiType.getCount(); ++slot) {
        addresses.push_back(*knownBase + slot * slotBytes);
      }
      hasKnownAddresses = true;
    }
  }

  if (addresses.empty()) {
    return op.emitError(
        "requires planner-assigned slot addresses or a constant level3 base");
  }

  auto info = std::make_unique<BaseMemInfo>(
      result, result, space, std::move(addresses), slotBytes,
      hasKnownAddresses && isLocalAddressSpace(space));
  buffer2MemInfoMap_[result].emplace_back(info->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareTileOpMemInfo(pto::DeclareTileOp op) {
  Value result = op.getTile();
  auto tileType = dyn_cast<pto::TileBufType>(result.getType());
  if (!tileType) {
    return op.emitError("requires a tile_buf result for sync analysis");
  }

  uint64_t sizeInBytes = getTileBufferFootprintBytes(tileType);
  if (sizeInBytes == 0) {
    return failure();
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(
          tileType.getMemorySpace())) {
    space = attr.getAddressSpace();
  }

  auto info = std::make_unique<BaseMemInfo>(
      result, result, space, SmallVector<uint64_t>{0}, sizeInBytes);
  buffer2MemInfoMap_[result].emplace_back(info->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareGlobalOpMemInfo(pto::DeclareGlobalOp op) {
  Value res = op.getEntry();
  auto tensorViewType = dyn_cast<pto::TensorViewType>(res.getType());
  if (!tensorViewType) {
    return failure();
  }

  uint64_t sizeInBytes = 0;
  ArrayRef<int64_t> shape = tensorViewType.getShape();
  bool isStatic = llvm::all_of(shape, [](int64_t dim) {
    return dim != ShapedType::kDynamic;
  });
  if (isStatic) {
    int64_t elemSize = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(tensorViewType.getElementType()));
    if (elemSize == 0) {
      return failure();
    }
    int64_t numElements = 1;
    for (int64_t dim : shape) {
      numElements *= dim;
    }
    sizeInBytes = static_cast<uint64_t>(numElements * elemSize);
  }

  // declare_global is a symbolic GM entry placeholder whose address gets bound
  // later by talloc/tpop. Using the SSA result as both base/root lets
  // partition_view/tstore/tload and tpush/tpop share one alias chain so
  // InsertSync can recover the GM-slot handshake ordering.
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      pto::AddressSpace::GM,
      SmallVector<uint64_t>{0},
      sizeInBytes);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

// ============================================================================
// 5. [P0 修改] 更新 PTO Op 信息 (通用接口版)
// ============================================================================
void PTOIRTranslator::UpdatePTOOpInfo(Operation *op) {
  // 1. 获取流水线类型 (现在通过 Interface)
  UpdatePTOOpInfoWithPipeline(op, getOpPipeline(op));
}

void PTOIRTranslator::UpdatePTOOpInfoWithPipeline(Operation *op,
                                                  pto::PipelineType pipe,
                                                  bool skipIfNoMemInfo) {
  // 如果 Op 不属于任何关心的流水线，直接跳过，不建立 Sync 节点
  if (pipe == pto::PipelineType::PIPE_UNASSIGNED) {
    return;
  }

  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  // 2. [关键] 使用 MemoryEffects 接口自动获取读写依赖
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    MemoryEffectVector effects;
    memEffect.getEffects(effects);
    for (auto &effect : effects) {
      Value val = effect.getValue();
      if (!val) {
        continue;
      }

       // 只有当 Value 在我们的 BufferMap 中有记录时，才视为有效依赖
       // (过滤掉比如 Loop Iterator 或其他标量)
       if (isa<MemoryEffects::Read>(effect.getEffect())) {
          if (!UpdateTExtractReadRegion(op, val, useVec)) {
            UpdateDefUseVec({val}, useVec);
          }
       } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          UpdateDefUseVec({val}, defVec);
       }
     }
  } else {
    // 如果算子有 Pipe 属性但没实现 MemoryEffects，这是一个定义错误
    // 我们可以打印个 Warning 或者保持为空 (认为无副作用)
    LLVM_DEBUG(llvm::dbgs() << "Warning: Op " << op->getName()
                            << " has Pipe but no MemoryEffects interface.\n");
  }

  if (skipIfNoMemInfo && defVec.empty() && useVec.empty()) {
    return;
  }

  // 3. 构建 Compound Node
  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, op->getName());
  compoundElement->elementOp = op;

  // 4. 设置 Core Type (用于区分 Cube/Vector 资源)
  // Matmul (M) 和 L1->L0 搬运 (MTE1) 通常涉及 Cube 资源
  if (pipe == pto::PipelineType::PIPE_M || pipe == pto::PipelineType::PIPE_MTE1) {
    compoundElement->compoundCoreType = pto::TCoreType::CUBE;
  } else {
    // MTE2, MTE3, Vector 归类为 Vector Core (或者对应 MTE 资源)
    compoundElement->compoundCoreType = pto::TCoreType::VECTOR;
  }

  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}

void PTOIRTranslator::MakeMacroCompound(Operation *op, PipelineType pipe,
                                        ValueRange defValues,
                                        ValueRange useValues,
                                        int macroPhaseId) {
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(defValues, defVec);
  UpdateDefUseVec(useValues, useVec);

  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, std::move(defVec), std::move(useVec), pipe, op->getName());
  compoundElement->elementOp = op;
  compoundElement->macroOpInstanceId = macroPhaseId;
  compoundElement->compoundCoreType =
      pipe == PipelineType::PIPE_M ? pto::TCoreType::CUBE
                                   : pto::TCoreType::VECTOR;
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}

void PTOIRTranslator::UpdateMacroOpInfo(Operation *op) {
  auto model = getSyncMacroModel(op);
  if (!model) {
    return;
  }
  for (const auto &phase : model->phases) {
    MakeMacroCompound(op, phase.pipe, ValueRange(phase.defValues),
                      ValueRange(phase.useValues), phase.phaseId);
  }
}

void PTOIRTranslator::UpdateHelperCallInfo(func::CallOp callOp) {
  func::FuncOp callee = lookupSyncHelper(callOp);
  if (!callee) {
    return;
  }

  std::optional<pto::PipelineType> pipe = getSyncHelperPipe(callee);
  if (!pipe || *pipe == pto::PipelineType::PIPE_UNASSIGNED) {
    return;
  }

  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  auto effects = callee->getAttrOfType<ArrayAttr>(kTileOpEffectsAttr);
  bool hasPreciseEffects = effects && effects.size() == callOp.getNumOperands();
  for (auto [operandIndex, operand] : llvm::enumerate(callOp.getOperands())) {
    if (!isSyncHelperMemoryOperand(operand.getType())) {
      continue;
    }

    StringRef effect = "readwrite";
    if (hasPreciseEffects) {
      if (auto effectAttr = dyn_cast<StringAttr>(effects[operandIndex])) {
        effect = effectAttr.getValue();
      }
    }
    if (effect == "read" || effect == "readwrite") {
      UpdateDefUseVec({operand}, useVec);
    }
    if (effect == "write" || effect == "readwrite") {
      UpdateDefUseVec({operand}, defVec);
    }
  }

  if (defVec.empty() && useVec.empty()) {
    return;
  }

  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, *pipe, callOp->getName());
  compoundElement->elementOp = callOp;
  compoundElement->compoundCoreType = getSyncHelperCoreType(*pipe);
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}

// ============================================================================
// 6. [P0 修改] 获取 Op 的 Pipeline 类型
// ============================================================================
pto::PipelineType PTOIRTranslator::getOpPipeline(Operation *op) const {
  // 1. 优先尝试通过接口获取
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
    // 注意：假设 pto::Pipe (ODS Enum) 和 pto::PipelineType (C++ Enum) 的数值定义是一致的
    // 或者在这里做一个 switch-case 映射
    // 目前假设直接 cast 是安全的 (0=S, 1=V, 2=M ...)
    return static_cast<pto::PipelineType>(pipeOp.getPipe());
  }
  // 2. 如果没实现接口，返回 Unassigned
  return pto::PipelineType::PIPE_UNASSIGNED;
}

// ============================================================================
// 7. 控制流处理 (SCF Support)
// ============================================================================

void PTOIRTranslator::UpdateForOpInfo(scf::ForOp forOp) {
  auto forBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  forBeginElement->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forBeginElement));

  std::unique_ptr<InstanceElement> &forElement = syncIR_[index];
  index++;

  auto *forBeginPtr = dyn_cast<LoopInstanceElement>(forElement.get());
  assert(forBeginPtr != nullptr && "Sync IR Construction failed.");

  if (!forOp.getInitArgs().empty()) {
    assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
    for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
      UpdateAliasBufferInfo(forOp.getRegionIterArgs()[i], arg);
      // An iter arg is only ever the init value on the first trip; later trips
      // carry whatever the yield produced. A window describing the init would
      // then claim disjointness for slices the loop actually revisits, so no
      // GMWindow is allowed to cross a loop boundary.
      DropCarriedGMWindows(forOp.getRegionIterArgs()[i]);
    }
  }

  RecursionIR(&forOp.getRegion());

  forBeginPtr->endId = index;
  auto forEnd = forBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}

void PTOIRTranslator::UpdateWhileOpInfo(scf::WhileOp whileOp) {
  auto loopBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  loopBeginElement->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(loopBeginElement));

  auto *loopBeginPtr = dyn_cast<LoopInstanceElement>(syncIR_.back().get());
  index++;

  if (!whileOp.getInits().empty()) {
    for (auto [initArg, blockArg] : llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
      UpdateAliasBufferInfo(blockArg, initArg);
      DropCarriedGMWindows(blockArg);
    }
    auto conditionOp = whileOp.getConditionOp();
    for (auto [yieldedArg, blockArg] : llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
      UpdateAliasBufferInfo(blockArg, yieldedArg);
      DropCarriedGMWindows(blockArg);
    }
  }

  RecursionIR(&whileOp.getBefore());
  RecursionIR(&whileOp.getAfter());

  loopBeginPtr->endId = index;
  auto forEnd = loopBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}

void PTOIRTranslator::UpdateIfOpInfo(scf::IfOp ifOp) {
  auto ifBeginElement = std::make_unique<BranchInstanceElement>(index, index, KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  auto *ifPtr = ifBeginElement.get();

  syncIR_.emplace_back(std::move(ifBeginElement));
  index++;

  // 1. 处理 Then 区域
  RecursionIR(&ifOp.getThenRegion());

  // Then 的结束占位符
  auto placeHolder = std::make_unique<PlaceHolderInstanceElement>(index, ifPtr->GetIndex());

  // 直接指向Then Block的yieldop
  placeHolder->elementOp = ifOp.getThenRegion().front().getTerminator();

  syncIR_.emplace_back(std::move(placeHolder));
  index++;

  ifPtr->branchId = index;

  // 2. 处理 Else 区域 (总是创建 SyncIR 节点，即使 IR 中没有 Else)
  auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
  ifElseElement->elementOp = ifOp.getOperation();
  auto *elsePtr = ifElseElement.get();

  syncIR_.emplace_back(std::move(ifElseElement));
  index++;

  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion());
  }

  // Else 的结束占位符
  auto elsePlaceHolder = std::make_unique<PlaceHolderInstanceElement>(index, elsePtr->GetIndex());

  if (ifOp.elseBlock()) {
      // 如果有真实的 Else Block，映射到 ifOp (CodeGen 需定位到 Else Yield 前)
      elsePlaceHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      elsePlaceHolder->isVirtualElse = false;
  } else {
      // 如果没有 Else Block，标记为虚拟，映射到 ifOp
      elsePlaceHolder->elementOp = ifOp.getOperation();
      elsePlaceHolder->isVirtualElse = true;
      elsePlaceHolder->parentIfOp = ifOp.getOperation();
  }

  syncIR_.emplace_back(std::move(elsePlaceHolder));
  index++;

  elsePtr->endId = index;
  ifPtr->endId = index;

  // 3. If End
  auto ifEndElement = ifPtr->CloneBranch(KindOfBranch::IF_END);
  ifEndElement->elementOp = ifOp.getOperation();
  syncIR_.emplace_back(std::move(ifEndElement));
  index++;
}

void PTOIRTranslator::UpdateYieldOpInfo(scf::YieldOp yieldOp) {
  auto *parentOp = yieldOp->getParentOp();
  if (!parentOp || isa<scf::WhileOp>(parentOp)) {
    return;
  }

  assert(parentOp->getResults().size() == yieldOp->getOpOperands().size());
  // A loop result is whatever the last trip yielded, but the window modelled
  // inside the body describes the first. An scf.if result needs no such care:
  // both arms are recorded, and neither repeats.
  bool parentIsLoop = isa<LoopLikeOpInterface>(parentOp);
  for (auto [yieldVal, resultVal] : llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
    if (parentIsLoop) {
      DropCarriedGMWindows(resultVal);
    }
  }
}

// ============================================================================
// 8. 辅助函数
// ============================================================================
void PTOIRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  if (!result || !source) {
    return;
  }
  if (!buffer2MemInfoMap_.contains(source)) {
    return;
  }

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    appendUniqueMemInfo(resultMemInfoVec, parentInfo->clone(result));
  }
}

void PTOIRTranslator::UpdateConservativeAliasBufferInfo(Value result,
                                                        Value source) {
  UpdateAliasBufferInfo(result, source);
}

// Same aliasing as UpdateAliasBufferInfo, plus the strided extent the view
// selects. The window is layered on top of the existing clone: dropping it
// only costs precision, so every bail-out below is safe.
void PTOIRTranslator::UpdateMakeTensorViewAliasBufferInfo(
    pto::MakeTensorViewOp op) {
  Value result = op.getResult();
  Value source = op.getPtr();
  if (!result || !source || !buffer2MemInfoMap_.contains(source)) {
    return;
  }

  std::optional<GMWindow> window = buildMakeTensorViewWindow(op);
  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto cloned = parentInfo->clone(result);
    // A window's offsets are relative to the root buffer, and the view starts
    // at `source`. Only when the two coincide is a zero base correct: an
    // intervening `pto.add_ptr` shifts the pointer without being modelled, so
    // the window would describe the wrong bytes.
    if (parentInfo->scope == pto::AddressSpace::GM &&
        !parentInfo->aliasesUnknownRange && source == parentInfo->rootBuffer) {
      cloned->gmWindow = window;
    }
    appendUniqueMemInfo(resultMemInfoVec, std::move(cloned));
  }
}

void PTOIRTranslator::UpdatePartitionViewAliasBufferInfo(
    pto::PartitionViewOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  if (!result || !source || !buffer2MemInfoMap_.contains(source)) {
    return;
  }

  uint64_t elemBytes = getGMViewElemBytes(source.getType());
  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto cloned = parentInfo->clone(result);
    // Composing onto the parent window, rather than rebuilding from the root,
    // is what makes a nested partition_view land on the right bytes.
    cloned->gmWindow =
        (elemBytes != 0 && parentInfo->gmWindow)
            ? composePartitionViewWindow(*parentInfo->gmWindow, op, elemBytes)
            : std::nullopt;
    appendUniqueMemInfo(resultMemInfoVec, std::move(cloned));
  }
}

// A value that re-enters a loop body names a different slice on every trip,
// while a window records the first trip only. Dropping it restores the
// conservative answer; the rest of the alias record stays intact.
void PTOIRTranslator::DropCarriedGMWindows(Value value) {
  if (!value || !buffer2MemInfoMap_.contains(value)) {
    return;
  }
  for (auto &info : buffer2MemInfoMap_[value]) {
    info->gmWindow = std::nullopt;
  }
}

LogicalResult
PTOIRTranslator::UpdateIntegerToPtrCastMemInfo(pto::CastPtrOp op) {
  Value result = op.getResult();
  if (!result) {
    return failure();
  }

  // Preserve provenance across the explicit byte-address round trip:
  //   %addr = pto.castptr %ptr : !pto.ptr<T> -> i64
  //   %ptr2 = pto.castptr %addr : i64 -> !pto.ptr<U>
  // `ptr2` is the same address as `ptr`, possibly with a different element
  // type, so scalar memory ops must participate in the same GM dependency
  // chain as tile stores/loads through the original pointer.
  if (auto ptrToInt = op.getInput().getDefiningOp<pto::CastPtrOp>();
      ptrToInt && isa<pto::PtrType>(ptrToInt.getInput().getType()) &&
      isa<IntegerType>(ptrToInt.getResult().getType())) {
    UpdateConservativeAliasBufferInfo(result, ptrToInt.getInput());
    if (buffer2MemInfoMap_.contains(result)) {
      return success();
    }
  }

  // A raw integer address may still point at any object in its address space.
  // Keep it in the sync IR instead of letting scalar ops be dropped by
  // skipIfNoMemInfo=true.
  auto space = getPointerLikeAddressSpace(result.getType());
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      result, result, space, SmallVector<uint64_t>{0},
      /*allocateSize=*/0, /*hasKnownPhysicalAddresses=*/false,
      /*aliasesUnknownRange=*/true);
  buffer2MemInfoMap_[result].emplace_back(newMemInfo->clone());
  return success();
}

void PTOIRTranslator::UpdateMultiTileGetAliasBufferInfo(
    pto::MultiTileGetOp op) {
  UpdateSlotSelectedAliasBufferInfo(op.getResult(), op.getSource(),
                                    op.getSlot());
}

void PTOIRTranslator::UpdateSlotSelectedAliasBufferInfo(Value result,
                                                        Value source,
                                                        Value slot) {
  if (!result || !source) {
    return;
  }
  if (!buffer2MemInfoMap_.contains(source)) {
    return;
  }

  IntegerAttr constAttr;
  bool isConstSlot = matchPattern(slot, m_Constant(&constAttr));
  int64_t constSlotIdx = isConstSlot ? constAttr.getValue().getSExtValue() : -1;

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
    // Multi-buffer parent: `baseAddresses` lists every physical slot's
    // planner-assigned offset. For a constant slot index, narrow it to just
    // that one slot so MemAlias's range-overlap check returns false when
    // two const-slot uses pick different slots. For a dynamic slot index,
    // keep all addresses -- the runtime SSA could resolve to any slot, so
    // sync must conservatively treat the use as touching all of them.
    if (isConstSlot && constSlotIdx >= 0 &&
        constSlotIdx < static_cast<int64_t>(newInfo->baseAddresses.size()) &&
        newInfo->baseAddresses.size() > 1) {
      uint64_t pickAddr =
          newInfo->baseAddresses[static_cast<size_t>(constSlotIdx)];
      newInfo->baseAddresses.clear();
      newInfo->baseAddresses.push_back(pickAddr);
    }
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

// 计算 SubView 结果 segment 的大小（按主序维度取连续方向的大小）。
static uint64_t getSubViewSegmentSize(pto::SubViewOp op,
                                      pto::TileBufType sourceType,
                                      unsigned elemBytes) {
  if (auto region = getBoxedSubViewRegion(op, sourceType)) {
    return region->segmentBytes;
  }
  auto sizesAttr = op.getSizes();
  int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();
  bool rowMajor =
      sourceType.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  return static_cast<uint64_t>((rowMajor ? colSize : rowSize) *
                               static_cast<int64_t>(elemBytes));
}

// 校验所有父 buffer 的 MemInfo 均为单基地址且已分配，方可精确推导别名。
static bool hasSingleBaseAllocatedParents(
    const llvm::SmallVector<std::unique_ptr<BaseMemInfo>> &parentInfos) {
  for (auto &parentInfo : parentInfos) {
    if (!parentInfo || parentInfo->baseAddresses.size() != 1 ||
        parentInfo->allocateSize == 0) {
      return false;
    }
  }
  return true;
}

// 父 Tile 地址已知、不与未知范围别名，且 [0, end) 落在其分配范围内。
static bool fitsKnownParentRange(
    const llvm::SmallVector<std::unique_ptr<BaseMemInfo>> &parentInfos,
    uint64_t end) {
  for (auto &parentInfo : parentInfos) {
    if (!parentInfo || parentInfo->aliasesUnknownRange ||
        !parentInfo->hasKnownPhysicalAddresses ||
        parentInfo->baseAddresses.size() != 1 ||
        end > parentInfo->allocateSize ||
        parentInfo->baseAddresses[0] >
            std::numeric_limits<uint64_t>::max() - end) {
      return false;
    }
  }
  return true;
}

// 为 result 克隆 source 的每个父 MemInfo，基址按 offsets 偏移，大小为 segmentSize。
static void appendOffsetAliasMemInfos(Buffer2MemInfoMap &buffer2MemInfoMap,
                                      Value result, Value source,
                                      ArrayRef<uint64_t> offsets,
                                      uint64_t segmentSize) {
  auto &resultMemInfoVec = buffer2MemInfoMap[result];
  for (auto &parentInfo : buffer2MemInfoMap[source]) {
    auto newInfo = parentInfo->clone(result);
    SmallVector<uint64_t> addresses;
    addresses.reserve(offsets.size());
    uint64_t parentBase = parentInfo->baseAddresses[0];
    for (uint64_t offset : offsets) {
      addresses.push_back(parentBase + offset);
    }
    newInfo->baseAddresses = std::move(addresses);
    newInfo->allocateSize = segmentSize;
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

void PTOIRTranslator::UpdateTileSubViewAliasBufferInfo(pto::SubViewOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  if (!result || !source) {
    return;
  }
  if (!buffer2MemInfoMap_.contains(source)) {
    return;
  }

  auto sourceType = dyn_cast<pto::TileBufType>(source.getType());
  if (!sourceType) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  if (auto range = getContiguousAccColumnRange(op)) {
    if (!updateAccColumnAliasBufferInfo(op, *range, buffer2MemInfoMap_)) {
      UpdateConservativeAliasBufferInfo(result, source);
    }
    return;
  }

  unsigned elemBytes = pto::getPTOStorageElemByteSize(sourceType.getElementType());
  auto subViewAddresses =
      elemBytes == 0 ? std::nullopt
                     : getPtoSubViewBaseAddresses(
                           op, sourceType, static_cast<int64_t>(elemBytes));
  if (!subViewAddresses || subViewAddresses->empty()) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  uint64_t segmentSize = getSubViewSegmentSize(op, sourceType, elemBytes);

  if (!hasSingleBaseAllocatedParents(buffer2MemInfoMap_[source])) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }
  // boxed 分段只在父 Tile 的物理范围已知、且分段不越界时使用。
  if (getBoxedSubViewRegion(op, sourceType) &&
      !fitsKnownParentRange(buffer2MemInfoMap_[source],
                            subViewAddresses->back() + segmentSize)) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  appendOffsetAliasMemInfos(buffer2MemInfoMap_, result, source,
                            *subViewAddresses, segmentSize);
}

void PTOIRTranslator::UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec) {
  for (Value v : values) {
    if (buffer2MemInfoMap_.contains(v)) {
      for (auto &memInfo : buffer2MemInfoMap_[v]) {
        vec.push_back(memInfo.get());
      }
    }
  }
}

// TEXTRACT 源与目标为同 dtype/布局的 MAT->LEFT/RIGHT 搬运，目标为非 compact
// 的静态 Tile 时，读窗口才能由目标物理 shape 推出。
static bool isRefinableTExtract(pto::TileBufType sourceType,
                                pto::TileBufType destType) {
  if (!sourceType || !destType || !isStaticRank2Shape(destType.getShape()) ||
      sourceType.getElementType() != destType.getElementType() ||
      sourceType.getBLayoutValueI32() != destType.getBLayoutValueI32() ||
      sourceType.getSLayoutValueI32() != destType.getSLayoutValueI32() ||
      sourceType.getSFractalSizeI32() != destType.getSFractalSizeI32() ||
      destType.getCompactModeI32() !=
          static_cast<int32_t>(pto::CompactMode::Null)) {
    return false;
  }
  pto::AddressSpace destSpace = getTileAddressSpace(destType);
  return destSpace == pto::AddressSpace::LEFT ||
         destSpace == pto::AddressSpace::RIGHT;
}

// 用目标物理 shape 覆盖最大读取范围，而不是可能动态变化的 valid shape。
static std::optional<BoxedTileRegion>
getTExtractReadRegion(pto::TExtractOp extract) {
  auto sourceType = dyn_cast<pto::TileBufType>(extract.getSrc().getType());
  auto destType = dyn_cast<pto::TileBufType>(extract.getDst().getType());
  if (!isRefinableTExtract(sourceType, destType)) {
    return std::nullopt;
  }
  int64_t rowOffset = 0;
  int64_t colOffset = 0;
  if (!getConstIndexValue(extract.getIndexRow(), rowOffset) ||
      !getConstIndexValue(extract.getIndexCol(), colOffset)) {
    return std::nullopt;
  }
  return getBoxedTileRegion(sourceType, rowOffset, colOffset,
                            destType.getShape()[0], destType.getShape()[1]);
}

bool PTOIRTranslator::UpdateTExtractReadRegion(
    Operation *op, Value source, SmallVector<const BaseMemInfo *> &vec) {
  auto extract = dyn_cast<pto::TExtractOp>(op);
  if (!extract || source != extract.getSrc() ||
      !source.getDefiningOp<pto::AllocTileOp>()) {
    return false;
  }
  auto region = getTExtractReadRegion(extract);
  auto found = buffer2MemInfoMap_.find(source);
  if (!region || region->offsets.empty() || found == buffer2MemInfoMap_.end() ||
      found->second.empty() ||
      !fitsKnownParentRange(found->second,
                            region->offsets.back() + region->segmentBytes)) {
    return false;
  }
  // 读窗口只属于这条指令，不能写回源 Tile 的全局别名表。
  for (auto &parentInfo : found->second) {
    auto info = parentInfo->clone(source);
    info->baseAddresses.clear();
    for (uint64_t offset : region->offsets) {
      info->baseAddresses.push_back(parentInfo->baseAddresses[0] + offset);
    }
    info->allocateSize = region->segmentBytes;
    vec.push_back(info.get());
    opAccessMemInfos_.push_back(std::move(info));
  }
  return true;
}

// ============================================================================
// 9. 调试与打印支持
// ============================================================================

std::string PTOIRTranslator::getPipelineName(pto::PipelineType pipe) const {
  switch (pipe) {
  case pto::PipelineType::PIPE_MTE1: return "MTE1";
  case pto::PipelineType::PIPE_MTE2: return "MTE2";
  case pto::PipelineType::PIPE_MTE3: return "MTE3";
  case pto::PipelineType::PIPE_M:    return "CUBE";
  case pto::PipelineType::PIPE_V:    return "VECTOR";
  case pto::PipelineType::PIPE_S:    return "SCALAR";
  case pto::PipelineType::PIPE_ALL:  return "BARRIER";
  default: return "UNKNOWN";
  }
}

void PTOIRTranslator::printMemInfoList(llvm::raw_ostream &os,
                                       const SmallVector<const BaseMemInfo *> &list,
                                       AsmState &state) const {
  os << "[";
  bool first = true;
  for (const auto *info : list) {
    if (!first) {
      os << ", ";
    }
    info->rootBuffer.printAsOperand(os, state);
    // [Fix] 打印 MAT 或 VEC 或 GM
    if (info->scope == pto::AddressSpace::GM) {
      os << "(GM)";
    } else if (info->scope == pto::AddressSpace::MAT) {
      os << "(MAT)";
    } else if (info->scope == pto::AddressSpace::VEC) {
      os << "(VEC)";
    } else {
      os << "(Other)"; // 处理 LEFT/RIGHT/ACC 等其他情况
    }
    if (info->gmWindow) {
      const GMWindow &window = *info->gmWindow;
      os << "{off=" << window.byteOffset << ",contig=" << window.contigBytes;
      for (auto [stride, count] : window.outerDims) {
        os << ",[" << stride << "x" << count << "]";
      }
      os << "}";
    }
    first = false;
  }
  os << "]";
}

void PTOIRTranslator::print() {
  llvm::errs() << "\n=== PTO IR Translator Dump ===\n";

  AsmState state(func_);

  llvm::errs() << "--- Buffer Analysis (Value -> Root) ---\n";
  for (auto &it : buffer2MemInfoMap_) {
    Value v = it.first;
    auto &infoList = it.second;

    llvm::errs() << "  ";
    v.printAsOperand(llvm::errs(), state);
    llvm::errs() << " -> ";

    for (auto &mem : infoList) {
        mem->rootBuffer.printAsOperand(llvm::errs(), state);
        llvm::errs() << " ";
    }
    llvm::errs() << "\n";
  }

  llvm::errs() << "\n--- SyncIR Structure ---\n";
  for (const auto &element : syncIR_) {
    unsigned id = element->GetIndex();
    llvm::errs() << llvm::formatv("{0,4}: ", id);

    switch (element->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = dyn_cast<CompoundInstanceElement>(element.get());
      llvm::errs() << "COMPOUND [" << getPipelineName(comp->kPipeValue) << "] ";
      llvm::errs() << comp->opName.getStringRef() << "\n";

      llvm::errs() << "      DEF: ";
      printMemInfoList(llvm::errs(), comp->defVec, state);
      llvm::errs() << "\n      USE: ";
      printMemInfoList(llvm::errs(), comp->useVec, state);
      llvm::errs() << "\n";
      break;
    }
    case InstanceElement::KindTy::LOOP:
        llvm::errs() << "LOOP\n"; break;
    case InstanceElement::KindTy::BRANCH:
        llvm::errs() << "BRANCH\n"; break;
    case InstanceElement::KindTy::PLACE_HOLDER:
        llvm::errs() << "PLACE_HOLDER\n"; break;
    }
  }
  llvm::errs() << "==============================\n\n";
}
