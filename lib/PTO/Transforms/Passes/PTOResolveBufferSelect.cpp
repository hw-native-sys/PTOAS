// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOResolveBufferSelect.cpp -----------------------------------------===//
//
// Lowering for tile-native buffer views and multi-buffer slot selection.
//
// Consumes tile-native `pto.subview` and `pto.multi_tile_get` operations after
// memory planning. Subviews rooted at `pto.alloc_tile` reuse the planned
// address, while subviews rooted at `pto.declare_tile` materialize the runtime
// address assigned by a preceding pipe operation. Constant multi-buffer slots
// become addressed `pto.alloc_tile` handles directly; dynamic slots select an
// address through an N-way `arith.select` chain. The user SSA remains the slot
// selector; this pass does not synthesize `iv mod N`.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTORESOLVEBUFFERSELECT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#define DEBUG_TYPE "pto-resolve-buffer-select"

using namespace mlir;

namespace {

constexpr int64_t kSFractal1024 = 1024;
constexpr int64_t kSFractal512 = 512;
constexpr int64_t kSFractal32 = 32;
constexpr int64_t kFractalInnerDimension = 16;
constexpr int64_t kSFractal32InnerColumnCount = 2;
constexpr int32_t kBLayoutColMajor = static_cast<int32_t>(pto::BLayout::ColMajor);
constexpr int32_t kBLayoutRowMajor =
    static_cast<int32_t>(pto::BLayout::RowMajor);
constexpr int32_t kSlayoutNoneBox = static_cast<int32_t>(pto::SLayout::NoneBox);
constexpr int32_t kSlayoutRowMajor = static_cast<int32_t>(pto::SLayout::RowMajor);
constexpr int32_t kSlayoutColMajor = static_cast<int32_t>(pto::SLayout::ColMajor);
// Tile shape / valid-shape dimension indices: dim0 = row, dim1 = col.
constexpr unsigned kDim0 = 0;
constexpr unsigned kDim1 = 1;
constexpr uint64_t kCubeTileAddressAlignmentBytes = 512;
constexpr uint64_t kVectorTileAddressAlignmentBytes = 32;
constexpr unsigned kI64BitWidth = 64;

static uint64_t alignUp(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  return ((value + align - 1) / align) * align;
}

static Value ensureI64(Value value, IRRewriter &rewriter, Location loc) {
  if (!value) {
    return {};
  }
  if (value.getType().isInteger(kI64BitWidth)) {
    return value;
  }
  if (value.getType().isIndex()) {
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), value);
  }
  if (isa<IntegerType>(value.getType())) {
    return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
  }
  return {};
}

// Strides for the non-boxed (plain row/col major) layout family: the unit
// stride follows the storage order and the other stride spans the full
// matrix, optionally extended by one row for compact RowPlusOne mode.
static bool getNoneBoxPointerStrides(pto::TileBufType type,
                                     ArrayRef<int64_t> shape, int32_t bl,
                                     int64_t &rowStride, int64_t &colStride) {
  bool rowPlusOne =
      type.getCompactModeI32() ==
      static_cast<int32_t>(pto::CompactMode::RowPlusOne);
  rowStride = bl == kBLayoutColMajor ? 1 : shape[kDim1] + (rowPlusOne ? 1 : 0);
  colStride =
      bl == kBLayoutColMajor ? shape[kDim0] + (rowPlusOne ? 1 : 0) : 1;
  return true;
}

// Fractal inner-matrix dimensions by fractal size and sub-layout. Returns
// false for combinations the pointer arithmetic below does not model, or a
// zero element size.
static bool getFractalInnerDims(pto::TileBufConfigAttr config, int32_t sl,
                                unsigned elemBytes, int64_t &innerRows,
                                int64_t &innerCols) {
  if (elemBytes == 0) {
    return false;
  }
  int32_t fractal = config.getSFractalSize().getInt();
  if (fractal == kSFractal1024) {
    innerRows = kFractalInnerDimension;
    innerCols = kFractalInnerDimension;
  } else if (fractal == kSFractal32) {
    innerRows = kFractalInnerDimension;
    innerCols = kSFractal32InnerColumnCount;
  } else if (fractal == kSFractal512 && sl == kSlayoutRowMajor) {
    innerRows = kFractalInnerDimension;
    innerCols = kSFractal32 / elemBytes;
  } else if (fractal == kSFractal512 && sl == kSlayoutColMajor) {
    innerRows = kSFractal32 / elemBytes;
    innerCols = kFractalInnerDimension;
  } else {
    return false;
  }
  return true;
}

// Strides for the boxed (fractal) layout family: the stride across fractal
// blocks spans the full matrix row/column; the in-block stride follows the
// fractal sub-layout.
static bool getBoxedPointerStrides(pto::TileBufType type,
                                   ArrayRef<int64_t> shape, int32_t bl,
                                   int32_t sl, int64_t &rowStride,
                                   int64_t &colStride) {
  unsigned elemBytes = pto::getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0) {
    return false;
  }
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (!getFractalInnerDims(type.getConfigAttr(), sl, elemBytes, innerRows,
                           innerCols)) {
    return false;
  }

  if (bl == kBLayoutColMajor) {
    if (sl != kSlayoutRowMajor) {
      return false;
    }
    rowStride = innerCols;
    colStride =
        shape[kDim0] +
        (type.getCompactModeI32() ==
                 static_cast<int32_t>(pto::CompactMode::RowPlusOne)
             ? 1
             : 0);
  } else {
    rowStride =
        shape[kDim1] +
        (type.getCompactModeI32() ==
                 static_cast<int32_t>(pto::CompactMode::RowPlusOne)
             ? 1
             : 0);
    colStride = innerRows;
  }
  return true;
}

static bool getTilePointerStrides(pto::TileBufType type, int64_t &rowStride,
                                  int64_t &colStride) {
  auto shape = type.getShape();
  if (shape.size() != mlir::pto::kValue2 ||
      llvm::is_contained(shape, ShapedType::kDynamic)) {
    return false;
  }

  auto config = type.getConfigAttr();
  int32_t bl = static_cast<int32_t>(config.getBLayout().getValue());
  int32_t sl = static_cast<int32_t>(config.getSLayout().getValue());
  if (sl == kSlayoutNoneBox) {
    return getNoneBoxPointerStrides(type, shape, bl, rowStride, colStride);
  }
  return getBoxedPointerStrides(type, shape, bl, sl, rowStride, colStride);
}

static uint64_t getTileAddressAlignmentBytes(pto::TileBufType type) {
  auto addrSpace =
      dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace());
  if (!addrSpace) {
    return 1;
  }

  switch (addrSpace.getAddressSpace()) {
  case pto::AddressSpace::LEFT:
  case pto::AddressSpace::RIGHT:
  case pto::AddressSpace::ACC:
    return kCubeTileAddressAlignmentBytes;
  case pto::AddressSpace::VEC:
  case pto::AddressSpace::MAT:
  case pto::AddressSpace::BIAS:
  case pto::AddressSpace::SCALING:
    return kVectorTileAddressAlignmentBytes;
  case pto::AddressSpace::GM:
  case pto::AddressSpace::Zero:
    return 1;
  }
  return 1;
}

static Value computeTileAddress(Value value, IRRewriter &rewriter,
                                Location loc) {
  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>()) {
    return ensureI64(alloc.getAddr(), rewriter, loc);
  }
  if (value.getDefiningOp<pto::DeclareTileOp>()) {
    auto tileType = dyn_cast<pto::TileBufType>(value.getType());
    if (!tileType) {
      return {};
    }
    auto memorySpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(tileType.getMemorySpace());
    if (!memorySpace) {
      return {};
    }
    auto ptrType = pto::PtrType::get(rewriter.getContext(),
                                     tileType.getElementType(), memorySpace);
    Value ptr =
        rewriter.create<pto::TileBufAddrOp>(loc, ptrType, value).getDst();
    return rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), ptr)
        .getResult();
  }
  if (auto subview = value.getDefiningOp<pto::SubViewOp>()) {
    Value base = computeTileAddress(subview.getSource(), rewriter, loc);
    auto sourceType = subview.getSource().getType();
    int64_t rowStride = 0;
    int64_t colStride = 0;
    if (!base || !getTilePointerStrides(sourceType, rowStride, colStride) ||
        subview.getOffsets().size() != mlir::pto::kValue2) {
      return {};
    }
    Value row = ensureI64(subview.getOffsets()[0], rewriter, loc);
    Value col = ensureI64(subview.getOffsets()[1], rewriter, loc);
    if (!row || !col) {
      return {};
    }
    Value rowScale = rewriter.create<arith::ConstantIntOp>(loc, rowStride, 64);
    Value colScale = rewriter.create<arith::ConstantIntOp>(loc, colStride, 64);
    row = rewriter.create<arith::MulIOp>(loc, row, rowScale);
    col = rewriter.create<arith::MulIOp>(loc, col, colScale);
    Value elements = rewriter.create<arith::AddIOp>(loc, row, col);
    int64_t elemBytes = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(sourceType.getElementType()));
    if (elemBytes == 0) {
      return {};
    }
    Value byteScale = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    Value bytes = rewriter.create<arith::MulIOp>(loc, elements, byteScale);
    return rewriter.create<arith::AddIOp>(loc, base, bytes);
  }
  return {};
}

// A stride difference along a dimension the child window does not span is
// immaterial: with a unit extent no element address ever uses that stride, so
// a single-row (or single-column) window keeps contiguous addressing. This
// only holds for the plain none_box layout family, whose element addresses
// are base + row * rowStride + col * colStride; boxed layouts interleave
// fractal blocks, so their block strides stay load-bearing in both dims.
static bool windowStrideMatches(int64_t inheritedStride, int64_t childStride, int64_t childExtent,
                                bool plainLayout)
{
    return inheritedStride == childStride || (plainLayout && childExtent == 1);
}

static bool isNoneBoxLayout(pto::TileBufType type)
{
    return static_cast<int32_t>(type.getConfigAttr().getSLayout().getValue()) == kSlayoutNoneBox;
}

static pto::TileBufType getSubviewPhysicalType(pto::SubViewOp op) {
  pto::TileBufType sourceType = op.getSource().getType();
  pto::TileBufType resultType = op.getResult().getType();
  ArrayRef<int64_t> physicalShape = sourceType.getShape();

  int64_t inheritedRowStride = 0;
  int64_t inheritedColStride = 0;
  int64_t childRowStride = 0;
  int64_t childColStride = 0;
  auto compactChildType = pto::TileBufType::get(
      op.getContext(), resultType.getShape(), resultType.getElementType(),
      resultType.getMemorySpace(), resultType.getValidShape(),
      resultType.getConfigAttr());
  bool plainLayout = isNoneBoxLayout(sourceType);
  // If the child tile's compact layout keeps the inherited addressing, the
  // addressed handle can use the child physical shape. Otherwise keep the
  // parent shape and express the logical slice via valid.
  if (getTilePointerStrides(sourceType, inheritedRowStride,
                            inheritedColStride) &&
      getTilePointerStrides(compactChildType, childRowStride, childColStride) &&
      windowStrideMatches(inheritedRowStride, childRowStride,
                          resultType.getShape()[kDim0], plainLayout) &&
      windowStrideMatches(inheritedColStride, childColStride,
                          resultType.getShape()[kDim1], plainLayout)) {
    physicalShape = resultType.getShape();
  }

  return pto::TileBufType::get(
      op.getContext(), physicalShape, resultType.getElementType(),
      resultType.getMemorySpace(), resultType.getValidShape(),
      resultType.getConfigAttr());
}

static Value getSubviewValidOperand(pto::SubViewOp op,
                                    pto::TileBufType physicalType,
                                    unsigned dim, IRRewriter &rewriter) {
  Value operand = dim == kDim0 ? op.getValidRow() : op.getValidCol();
  ArrayRef<int64_t> validShape = physicalType.getValidShape();
  if (validShape.size() <= dim || validShape[dim] >= 0) {
    return {};
  }
  if (operand) {
    return operand;
  }
  ArrayRef<int64_t> shape = physicalType.getShape();
  if (shape.size() > dim && shape[dim] != ShapedType::kDynamic) {
    return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), shape[dim]);
  }
  return {};
}

static std::optional<pto::AddressSpace> getTileBufSpace(Type type) {
  auto tileType = dyn_cast<pto::TileBufType>(type);
  if (!tileType) {
    return std::nullopt;
  }
  auto spaceAttr =
      dyn_cast_or_null<pto::AddressSpaceAttr>(tileType.getMemorySpace());
  if (!spaceAttr) {
    return std::nullopt;
  }
  return spaceAttr.getAddressSpace();
}

// A parent-shaped subview keeps the parent's physical strides, so a
// mat-source tmov from it can no longer pair with a compact left/right
// destination: the tmov verifier compares physical shapes and fails. Such a
// move is a window transfer, which TEXTRACT_M2LR expresses directly from the
// parent tile. Mirror the textract verifier's layout and element tables so
// only rewrites that verify are made; anything else keeps the tmov and its
// existing diagnostic.
static bool isTMovExtractCompatible(pto::TMovOp mov, bool isA5) {
  auto srcTb = dyn_cast<pto::TileBufType>(mov.getSrc().getType());
  auto dstTb = dyn_cast<pto::TileBufType>(mov.getDst().getType());
  if (!srcTb || !dstTb) {
    return false;
  }
  Type elem = srcTb.getElementType();
  const bool elemPairOk =
      elem == dstTb.getElementType() &&
      (elem.isF16() || elem.isBF16() || elem.isF32() ||
       elem.isInteger(static_cast<unsigned>(mlir::pto::kValue8)));
  if (!elemPairOk) {
    return false;
  }
  auto dstSpace = getTileBufSpace(mov.getDst().getType());
  if (!dstSpace || (*dstSpace != pto::AddressSpace::LEFT &&
                   *dstSpace != pto::AddressSpace::RIGHT)) {
    return false;
  }
  int32_t srcBl = srcTb.getBLayoutValueI32();
  int32_t srcSl = srcTb.getSLayoutValueI32();
  int32_t dstBl = dstTb.getBLayoutValueI32();
  int32_t dstSl = dstTb.getSLayoutValueI32();
  const bool dstIsRight = *dstSpace == pto::AddressSpace::RIGHT;
  const bool dstLayoutOk =
      dstIsRight
          ? (dstBl == kBLayoutRowMajor && dstSl == kSlayoutColMajor)
          : (isA5 ? (dstBl == kBLayoutColMajor && dstSl == kSlayoutRowMajor)
                  : (dstBl == kBLayoutRowMajor && dstSl == kSlayoutRowMajor));
  if (!dstLayoutOk) {
    return false;
  }
  if (isA5) {
    return (srcBl == kBLayoutRowMajor && srcSl == kSlayoutColMajor) ||
           (srcBl != kBLayoutRowMajor && srcSl == kSlayoutRowMajor) ||
           (!dstIsRight && srcBl == kBLayoutRowMajor);
  }
  return srcBl == kBLayoutRowMajor || srcSl == kSlayoutRowMajor;
}

// pto-isa TExtractToA/TExtractToB assert window-origin alignment only in
// debug builds; release kernels silently truncate sub-aligned offsets, and
// PTOAS's own alignment verification covers conv-tile sources only. Accept
// statically known, branch-aligned offsets so a misaligned or dynamic window
// keeps the original tmov and its compile-time diagnostic instead of turning
// into a silent wrong-window device read. Runtime branches
// (textract_common.hpp): the transpose variants run when the src/dst
// slayouts differ; c0 is 32 / element bytes and the fractal span is 16
// (32 for one-byte elements).
//   dst right, non-transpose: row % c0 == 0, col % 16 == 0;
//   dst right, transpose:     row % fractal == 0, col % fractal == 0;
//   dst left,  non-transpose: row % 16 == 0, col % c0 == 0;
//   dst left,  transpose:     row % fractal == 0, col % fractal == 0.
static bool isSubviewWindowExtractAligned(pto::SubViewOp op, pto::TMovOp mov) {
  APInt row;
  APInt col;
  const bool rowIsConstant =
      matchPattern(op.getOffsets()[kDim0], m_ConstantInt(&row));
  const bool colIsConstant =
      matchPattern(op.getOffsets()[kDim1], m_ConstantInt(&col));
  if (!rowIsConstant || !colIsConstant) {
    return false;
  }
  auto srcTb = dyn_cast<pto::TileBufType>(op.getSource().getType());
  auto dstTb = dyn_cast<pto::TileBufType>(mov.getDst().getType());
  if (!srcTb || !dstTb) {
    return false;
  }
  // The Compact dst variants take their own valid-shape-aware paths and a
  // single-row row-major source uses the vector extract path; neither is
  // modeled by the alignment rules above.
  const bool compactDst =
      dstTb.getCompactModeI32() != static_cast<int32_t>(pto::CompactMode::Null);
  if (compactDst) {
    return false;
  }
  auto dstSpace = getTileBufSpace(mov.getDst().getType());
  if (!dstSpace) {
    return false;
  }
  const bool dstIsLeftOrRight =
      *dstSpace == pto::AddressSpace::LEFT ||
      *dstSpace == pto::AddressSpace::RIGHT;
  ArrayRef<int64_t> srcShape = srcTb.getShape();
  const bool rank2Source = srcShape.size() > kDim0;
  if (!rank2Source || !dstIsLeftOrRight || srcShape[kDim0] == 1) {
    return false;
  }
  const int64_t elemBytes = static_cast<int64_t>(
      pto::getPTOStorageElemByteSize(srcTb.getElementType()));
  if (elemBytes == 0) {
    return false;
  }
  const int64_t c0Size = kSFractal32 / elemBytes;
  const int64_t fractalSize =
      elemBytes == 1 ? kSFractal32 : kFractalInnerDimension;
  const bool transpose =
      srcTb.getSLayoutValueI32() != dstTb.getSLayoutValueI32();
  const bool dstIsRight = *dstSpace == pto::AddressSpace::RIGHT;
  const int64_t rowModulus =
      transpose ? fractalSize : (dstIsRight ? c0Size : kFractalInnerDimension);
  const int64_t colModulus =
      transpose ? fractalSize : (dstIsRight ? kFractalInnerDimension : c0Size);
  return row.getSExtValue() % rowModulus == 0 &&
         col.getSExtValue() % colModulus == 0;
}

// Rewrite mat-source tmov users of a parent-shaped subview into textract
// windows from the subview's source at the subview offsets.
static void rewriteParentShapedTMovUsers(pto::SubViewOp op, MLIRContext *ctx) {
  const bool rank2Offsets = op.getOffsets().size() == mlir::pto::kValue2;
  auto srcSpace = getTileBufSpace(op.getResult().getType());
  if (!rank2Offsets || !srcSpace || *srcSpace != pto::AddressSpace::MAT) {
    return;
  }
  const bool isA5 = pto::isTargetArchA5(op.getOperation());
  IRRewriter rewriter(ctx);
  for (OpOperand &use : llvm::make_early_inc_range(op.getResult().getUses())) {
    auto mov = dyn_cast<pto::TMovOp>(use.getOwner());
    if (!mov) {
      continue;
    }
    const bool isMoveSource = mov.getSrc() == op.getResult();
    if (!isMoveSource) {
      continue;
    }
    const bool plainMove =
        !mov.getFp() && !mov.getPreQuantScalar() &&
        !mov.getAccToVecModeAttr() && !mov.getGrpAxisAttr() &&
        mov.getReluPreMode() == pto::ReluPreMode::NoRelu;
    if (!plainMove) {
      continue;
    }
    if (!isTMovExtractCompatible(mov, isA5)) {
      continue;
    }
    if (!isSubviewWindowExtractAligned(op, mov)) {
      continue;
    }
    rewriter.setInsertionPoint(mov);
    rewriter.create<pto::TExtractOp>(
        mov.getLoc(), op.getSource(), op.getOffsets()[kDim0],
        op.getOffsets()[kDim1], mov.getDst(), Value(), Value(),
        pto::AccToVecModeAttr(), pto::ReluPreMode::NoRelu);
    rewriter.eraseOp(mov);
  }
}

static LogicalResult resolveTileNativeSubviews(ModuleOp module,
                                               MLIRContext *ctx) {
  SmallVector<pto::SubViewOp, mlir::pto::kValue16> subviews;
  module.walk([&subviews](pto::SubViewOp op) { subviews.push_back(op); });
  for (pto::SubViewOp op : subviews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Value addr = computeTileAddress(op.getResult(), rewriter, op.getLoc());
    // A tile function argument is a symbolic runtime-bound handle. Keep its
    // subview tile-native; only planned local roots and declare_tile handles
    // assigned by preceding pipe operations can be normalized here.
    if (!addr) {
      continue;
    }
    pto::TileBufType physicalType = getSubviewPhysicalType(op);
    const bool keepsParentShape =
        physicalType.getShape() != op.getResult().getType().getShape();
    if (keepsParentShape) {
      // The resolved handle keeps the parent's physical shape and strides,
      // so mat-source tmov users would compare the parent shape against the
      // compact destination and fail verification. Move them onto textract
      // windows from the parent tile before the handle replaces the subview.
      rewriteParentShapedTMovUsers(op, ctx);
    }
    auto alloc = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), physicalType, addr,
        getSubviewValidOperand(op, physicalType, kDim0, rewriter),
        getSubviewValidOperand(op, physicalType, kDim1, rewriter),
        pto::ConvTileConfigAttr());
    alloc->setAttr("pto.view_semantics", rewriter.getStringAttr("subview"));
    rewriter.replaceOp(op, alloc.getResult());
  }
  return success();
}

static FailureOr<uint64_t> getStaticSlotBytes(pto::TileBufType slotType) {
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(slotType.getElementType());
  if (elemBytes == 0) {
    return failure();
  }
  uint64_t bytes = elemBytes;
  for (int64_t dim : slotType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      return failure();
    }
    bytes *= static_cast<uint64_t>(dim);
  }
  return bytes;
}

static LogicalResult getMultiTileAddresses(pto::AllocMultiTileOp alloc,
                                           IRRewriter &rewriter,
                                           SmallVectorImpl<Value> &addrs) {
  uint32_t count = alloc.getResult().getType().getCount();
  if (auto planned = alloc->getAttrOfType<DenseI64ArrayAttr>(
          pto::kPtoMultiBufferAddrsAttrName)) {
    if (planned.size() != count) {
      return alloc.emitError("planned address count does not match slot count");
    }
    for (int64_t address : planned.asArrayRef()) {
      addrs.push_back(rewriter.create<arith::ConstantIntOp>(
          alloc.getLoc(), address, kI64BitWidth));
    }
    return success();
  }

  Value base = alloc.getAddr();
  if (!base) {
    return alloc.emitError(
        "has neither a level3 base address nor planner-assigned slot addresses");
  }
  auto slotBytes = getStaticSlotBytes(alloc.getResult().getType().getSlotType());
  if (failed(slotBytes)) {
    return alloc.emitError(
        "requires a static slot shape and known element byte size");
  }

  uint64_t slotStride =
      alignUp(*slotBytes,
              getTileAddressAlignmentBytes(
                  alloc.getResult().getType().getSlotType()));

  addrs.push_back(base);
  for (uint32_t slot = 1; slot < count; ++slot) {
    Value offset = rewriter.create<arith::ConstantIntOp>(
        alloc.getLoc(), static_cast<int64_t>(slot * slotStride), 64);
    addrs.push_back(
        rewriter.create<arith::AddIOp>(alloc.getLoc(), base, offset));
  }
  return success();
}

static LogicalResult resolveTileNativeMultiGets(ModuleOp module,
                                                MLIRContext *ctx) {
  SmallVector<pto::MultiTileGetOp, mlir::pto::kValue8> getOps;
  module.walk([&getOps](pto::MultiTileGetOp op) { getOps.push_back(op); });

  for (pto::MultiTileGetOp op : getOps) {
    auto alloc = op.getSource().getDefiningOp<pto::AllocMultiTileOp>();
    if (!alloc) {
      return op.emitError(
          "currently requires a direct pto.alloc_multi_tile source");
    }

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, mlir::pto::kValue8> addrs;
    if (failed(getMultiTileAddresses(alloc, rewriter, addrs))) {
      return failure();
    }

    Value selectedAddr;
    IntegerAttr constSlotAttr;
    if (matchPattern(op.getSlot(), m_Constant(&constSlotAttr))) {
      int64_t slot = constSlotAttr.getValue().getSExtValue();
      if (slot < 0 || slot >= static_cast<int64_t>(addrs.size())) {
        return op.emitError("constant slot is outside planned address range");
      }
      selectedAddr = addrs[static_cast<size_t>(slot)];
    } else {
      selectedAddr = addrs.front();
      for (uint32_t slot = 1; slot < addrs.size(); ++slot) {
        Value slotValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), slot);
        Value matches = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::eq, op.getSlot(), slotValue);
        selectedAddr = rewriter.create<arith::SelectOp>(
            op.getLoc(), matches, addrs[slot], selectedAddr);
      }
    }

    auto slotHandle = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), op.getResult().getType(), selectedAddr,
        alloc.getValidRow() ? alloc.getValidRow() : Value(),
        alloc.getValidCol() ? alloc.getValidCol() : Value(),
        pto::ConvTileConfigAttr());
    rewriter.replaceOp(op, slotHandle.getResult());
  }

  SmallVector<pto::AllocMultiTileOp, mlir::pto::kValue8> allocs;
  module.walk([&allocs](pto::AllocMultiTileOp op) { allocs.push_back(op); });
  for (pto::AllocMultiTileOp alloc : allocs) {
    if (!alloc.getResult().use_empty()) {
      return alloc.emitError(
          "has unsupported uses after resolving pto.multi_tile_get");
    }
    alloc.erase();
  }
  return success();
}

struct PTOResolveBufferSelectPass
    : public mlir::pto::impl::PTOResolveBufferSelectBase<
          PTOResolveBufferSelectPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOResolveBufferSelectPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    if (failed(resolveTileNativeMultiGets(mod, ctx))) {
      signalPassFailure();
      return;
    }
    if (failed(resolveTileNativeSubviews(mod, ctx))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOResolveBufferSelectPass() {
  return std::make_unique<PTOResolveBufferSelectPass>();
}
