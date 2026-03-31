// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- InferPTOLayout.cpp - Infer layout for global tensor views -----------===//
//
// The pto-isa GlobalTensor ABI expects shape/stride to be represented in a 5D
// right-aligned form (pad leading dims with 1). We infer ND/DN/NZ with the same
// 5D view here and attach an optional `layout` attribute to:
//   - memref.reinterpret_cast (lowered from pto.make_tensor_view)
//   - memref.subview          (lowered from pto.partition_view)
//   - pto.tload / pto.tstore  (for fully-static GM memrefs)
//
// EmitC lowering should consume this attribute and avoid re-inferring layout
// when it is available.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr llvm::StringLiteral kLayoutAttrName = "layout";
static constexpr llvm::StringLiteral kInferredLayoutAttrName =
    "pto.inferred_layout";

static std::optional<int64_t> getConstInt(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static std::optional<int64_t> getConstInt(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt();
    return std::nullopt;
  }
  return getConstInt(ofr.get<Value>());
}

static unsigned elemByteSize(Type ty) {
  if (auto f = dyn_cast<FloatType>(ty))
    return f.getWidth() / 8;
  if (auto i = dyn_cast<IntegerType>(ty))
    return i.getWidth() / 8;
  return 0;
}

static bool isGlobalMemRef(MemRefType ty) {
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(ty.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    return (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  // Treat missing memory_space as GM.
  return true;
}

struct ShapeStride5D {
  SmallVector<int64_t, 5> shape;
  SmallVector<int64_t, 5> stride;
};

static bool isMinor2DLayout(Layout layout) {
  return layout == Layout::ND || layout == Layout::DN;
}

static std::optional<ShapeStride5D> rightAlignTo5D(ArrayRef<int64_t> shape,
                                                   ArrayRef<int64_t> stride) {
  if (shape.size() != stride.size())
    return std::nullopt;
  if (shape.size() > 5)
    return std::nullopt;

  ShapeStride5D out;
  out.shape.assign(5, 1);
  out.stride.assign(5, 1);

  const int rank = static_cast<int>(shape.size());
  const int shift = 5 - rank;
  for (int i = 0; i < rank; ++i) {
    out.shape[shift + i] = shape[i];
    out.stride[shift + i] = stride[i];
  }

  // Derive the padded leading strides with the same rule used in EmitC:
  // stride[i] = shape[i+1] * stride[i+1].
  for (int i = shift - 1; i >= 0; --i)
    out.stride[i] = out.shape[i + 1] * out.stride[i + 1];

  return out;
}

static bool matchesNDMinor2D(int64_t rows, int64_t cols, int64_t rowStride,
                             int64_t colStride) {
  if (cols != 1 && colStride != 1)
    return false;
  if (rows == 1)
    return true;
  return cols == 1 ? rowStride == 1 : rowStride == cols;
}

static bool matchesDNMinor2D(int64_t rows, int64_t cols, int64_t rowStride,
                             int64_t colStride) {
  if (rows != 1 && rowStride != 1)
    return false;
  if (cols == 1)
    return true;
  return rows == 1 ? colStride == 1 : colStride == rows;
}

static std::optional<Layout> inferMinor2DLayout(
    int64_t rows, int64_t cols, int64_t rowStride, int64_t colStride,
    std::optional<Layout> preferredMinor2D, bool *isMinor2DAmbiguous) {
  const bool nd = matchesNDMinor2D(rows, cols, rowStride, colStride);
  const bool dn = matchesDNMinor2D(rows, cols, rowStride, colStride);
  if (!nd && !dn)
    return Layout::ND;
  if (nd && dn) {
    if (isMinor2DAmbiguous)
      *isMinor2DAmbiguous = true;
    if (preferredMinor2D &&
        (*preferredMinor2D == Layout::ND || *preferredMinor2D == Layout::DN)) {
      return *preferredMinor2D;
    }
    return (cols == 1 && rows != 1) ? Layout::DN : Layout::ND;
  }
  return dn ? Layout::DN : Layout::ND;
}

static std::optional<Layout> inferLayout5D(ArrayRef<int64_t> shape,
                                           ArrayRef<int64_t> strides,
                                           unsigned elemBytes,
                                           std::optional<Layout> preferredMinor2D =
                                               std::nullopt,
                                           bool *isMinor2DAmbiguous = nullptr) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;
  if (isMinor2DAmbiguous)
    *isMinor2DAmbiguous = false;
  if (auto padded = rightAlignTo5D(shape, strides)) {
    auto &sh = padded->shape;
    auto &st = padded->stride;

    // NZ: 5D right-aligned, check middle dims (sh3/sh4/sh5 per spec)
    int64_t sh3 = sh[2], sh4 = sh[3], sh5 = sh[4];
    int64_t st4 = st[3], st5 = st[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return Layout::NZ;

    // ND/DN are minor-2D layout hints for the last two dimensions
    // (DIM_3 = rows, DIM_4 = cols). They are not a full 5D row/col-major tag.
    //
    // For vector-like shapes where one minor dim is 1, multiple stride patterns
    // are semantically equivalent. Prefer:
    //   - DN when cols == 1 (column vector)
    //   - ND when rows == 1 (row vector)
    const int64_t rows = sh[3];
    const int64_t cols = sh[4];
    const int64_t rowStride = st[3];
    const int64_t colStride = st[4];
    return inferMinor2DLayout(rows, cols, rowStride, colStride,
                              preferredMinor2D, isMinor2DAmbiguous);
  }
  return std::nullopt;
}

static std::optional<Layout> tileBLayoutToGlobalLayout(Type tileLikeTy) {
  auto tbTy = dyn_cast<TileBufType>(tileLikeTy);
  if (!tbTy)
    return std::nullopt;
  auto bl = dyn_cast_or_null<BLayoutAttr>(tbTy.getBLayoutAttr());
  if (!bl)
    return std::nullopt;
  switch (bl.getValue()) {
  case BLayout::RowMajor:
    return Layout::ND;
  case BLayout::ColMajor:
    return Layout::DN;
  }
  return std::nullopt;
}

static bool isVectorTileType(Type tileLikeTy) {
  auto tbTy = dyn_cast<TileBufType>(tileLikeTy);
  if (!tbTy)
    return false;
  auto ms = dyn_cast_or_null<AddressSpaceAttr>(tbTy.getMemorySpace());
  return ms && ms.getAddressSpace() == AddressSpace::VEC;
}

static bool isMinorColsOne(ArrayRef<int64_t> shape) {
  return !shape.empty() && shape.back() == 1;
}

struct ResolvedLayoutInfo {
  Operation *owner = nullptr;
  std::optional<Layout> layout;
  bool inferred = false;
};

static bool getStaticShapeAndStride(MakeTensorViewOp op,
                                    SmallVectorImpl<int64_t> &shape,
                                    SmallVectorImpl<int64_t> &strides);
static ResolvedLayoutInfo resolveLayoutFromViewValue(Value v);

static void setLayoutAttr(Operation *op, Layout layout, bool inferred) {
  op->setAttr(kLayoutAttrName, LayoutAttr::get(op->getContext(), layout));
  if (inferred)
    op->setAttr(kInferredLayoutAttrName, BoolAttr::get(op->getContext(), true));
  else
    op->removeAttr(kInferredLayoutAttrName);
}

template <typename SignalFailureFn>
static void verifyOrSetLayoutAttr(Operation *op,
                                  std::optional<Layout> inferred,
                                  SignalFailureFn signalFailure,
                                  bool isMinor2DAmbiguous = false) {
  auto existing = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
  if (existing) {
    if (inferred && existing.getLayout() != *inferred) {
      if (isMinor2DAmbiguous && isMinor2DLayout(existing.getLayout()) &&
          isMinor2DLayout(*inferred)) {
        return;
      }
      op->emitError() << "layout mismatch: user-specified layout="
                      << stringifyLayout(existing.getLayout())
                      << " but inferred=" << stringifyLayout(*inferred);
      signalFailure();
    }
    return;
  }
  setLayoutAttr(op, inferred.value_or(Layout::ND), /*inferred=*/true);
}

static std::optional<Layout> inferFromStaticMemRefTy(MemRefType mrTy) {
  if (!mrTy.hasStaticShape() || mrTy.getRank() == 0 || mrTy.getRank() > 5)
    return std::nullopt;
  SmallVector<int64_t> strideInts;
  int64_t offset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(mrTy, strideInts, offset)))
    return std::nullopt;
  if (offset == ShapedType::kDynamic ||
      llvm::any_of(strideInts,
                   [](int64_t s) { return s == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  return inferLayout5D(mrTy.getShape(), strideInts,
                       elemByteSize(mrTy.getElementType()));
}

template <typename LoadStoreOp, typename ViewGetter, typename TileGetter>
static void maybeRepairMinor2DLoadStoreLayout(LoadStoreOp op, ViewGetter getView,
                                              TileGetter getTile) {
  auto tilePref = isVectorTileType(getTile(op).getType())
                      ? tileBLayoutToGlobalLayout(getTile(op).getType())
                      : std::nullopt;
  if (!tilePref || (*tilePref != Layout::ND && *tilePref != Layout::DN))
    return;

  auto viewInfo = resolveLayoutFromViewValue(getView(op));
  if (!viewInfo.owner || !viewInfo.layout || !viewInfo.inferred ||
      *viewInfo.layout == *tilePref) {
    return;
  }
  auto tv = dyn_cast<MakeTensorViewOp>(viewInfo.owner);
  if (!tv)
    return;

  SmallVector<int64_t> shape, strides;
  bool ambiguous = false;
  if (!getStaticShapeAndStride(tv, shape, strides))
    return;
  (void)inferLayout5D(
      shape, strides,
      elemByteSize(cast<TensorViewType>(tv.getResult().getType()).getElementType()),
      std::nullopt, &ambiguous);
  if (ambiguous && isMinorColsOne(shape)) {
    setLayoutAttr(viewInfo.owner, *tilePref, /*inferred=*/true);
    setLayoutAttr(op.getOperation(), *tilePref, /*inferred=*/true);
  }
}

template <typename LoadStoreOp, typename ViewGetter, typename TileGetter>
static void attachLoadStoreLayout(LoadStoreOp op, ViewGetter getView,
                                  TileGetter getTile) {
  if (op->template getAttrOfType<LayoutAttr>(kLayoutAttrName)) {
    maybeRepairMinor2DLoadStoreLayout(op, getView, getTile);
    return;
  }

  auto viewInfo = resolveLayoutFromViewValue(getView(op));
  if (viewInfo.layout) {
    setLayoutAttr(op.getOperation(), *viewInfo.layout, viewInfo.inferred);
  } else if (auto memTy = dyn_cast<MemRefType>(getView(op).getType());
             memTy && isGlobalMemRef(memTy)) {
    setLayoutAttr(op.getOperation(), inferFromStaticMemRefTy(memTy).value_or(Layout::ND),
                  /*inferred=*/true);
  }

  maybeRepairMinor2DLoadStoreLayout(op, getView, getTile);
}

struct LayoutPreference {
  std::optional<Layout> preferred;
  bool conflict = false;
};

static LayoutPreference collectPreferredLayoutFromConsumers(Value tensorView) {
  LayoutPreference result;
  auto mergePref = [&](std::optional<Layout> candidate) {
    if (!candidate || (*candidate != Layout::ND && *candidate != Layout::DN))
      return;
    if (!result.preferred) {
      result.preferred = candidate;
      return;
    }
    if (*result.preferred != *candidate) {
      result.preferred = std::nullopt;
      result.conflict = true;
    }
  };

  auto walkUses = [&](auto &&self, Value v) -> void {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      unsigned operandIndex = use.getOperandNumber();

      if (auto part = dyn_cast<PartitionViewOp>(owner)) {
        if (operandIndex == 0)
          self(self, part.getResult());
        continue;
      }

      if (auto load = dyn_cast<pto::TLoadOp>(owner)) {
        if (operandIndex == 0 && isVectorTileType(load.getDst().getType()))
          mergePref(tileBLayoutToGlobalLayout(load.getDst().getType()));
        continue;
      }

      if (auto store = dyn_cast<pto::TStoreOp>(owner)) {
        if (operandIndex == 1 && isVectorTileType(store.getSrc().getType()))
          mergePref(tileBLayoutToGlobalLayout(store.getSrc().getType()));
        continue;
      }
    }
  };

  walkUses(walkUses, tensorView);
  return result;
}

static bool getStaticShapeAndStride(MakeTensorViewOp op,
                                    SmallVectorImpl<int64_t> &shape,
                                    SmallVectorImpl<int64_t> &strides) {
  auto tvTy = dyn_cast<TensorViewType>(op.getResult().getType());
  if (!tvTy)
    return false;

  const size_t rank = op.getShape().size();
  if (rank == 0 || rank > 5)
    return false;

  shape.clear();
  shape.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    int64_t dim = tvTy.getShape()[i];
    if (dim == ShapedType::kDynamic) {
      auto v = getConstInt(op.getShape()[i]);
      if (!v)
        return false;
      dim = *v;
    }
    shape.push_back(dim);
  }

  strides.clear();
  strides.reserve(rank);
  for (Value s : op.getStrides()) {
    auto v = getConstInt(s);
    if (!v)
      return false;
    strides.push_back(*v);
  }
  return true;
}

static ResolvedLayoutInfo resolveLayoutFromViewValue(Value v) {
  ResolvedLayoutInfo info;
  Operation *def = v.getDefiningOp();
  while (def) {
    if (auto layoutAttr = def->getAttrOfType<LayoutAttr>(kLayoutAttrName)) {
      info.owner = def;
      info.layout = layoutAttr.getLayout();
      if (auto inferred =
              def->getAttrOfType<BoolAttr>(kInferredLayoutAttrName))
        info.inferred = inferred.getValue();
      return info;
    }
    if (auto part = dyn_cast<PartitionViewOp>(def)) {
      v = part.getSource();
      def = v.getDefiningOp();
      continue;
    }
    break;
  }
  return info;
}

struct InferPTOLayoutPass
    : public PassWrapper<InferPTOLayoutPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferPTOLayoutPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // ------------------------------------------------------------------
    // 1) pto.make_tensor_view (only if it still exists in the pipeline)
    // ------------------------------------------------------------------
    func.walk([&](MakeTensorViewOp op) {
      SmallVector<int64_t> shape, strides;
      if (!getStaticShapeAndStride(op, shape, strides)) {
        verifyOrSetLayoutAttr(op.getOperation(), std::nullopt,
                              [this] { signalPassFailure(); });
        return;
      }

      auto pref = collectPreferredLayoutFromConsumers(op.getResult());
      // Guard rail: only use consumer preference for minor-2D ambiguous
      // "column-vector-like" outputs (cols == 1). This is the row-reduction
      // case we need to repair; applying it more broadly can violate pto-isa
      // static layout constraints (e.g. some GEMV/GEMM outputs).
      auto preferredForAmbiguous =
          (!pref.conflict && isMinorColsOne(shape)) ? pref.preferred
                                                    : std::nullopt;
      bool isAmbiguous = false;
      auto inferred = inferLayout5D(
          shape, strides,
          elemByteSize(cast<TensorViewType>(op.getResult().getType())
                           .getElementType()),
          preferredForAmbiguous, &isAmbiguous);
      verifyOrSetLayoutAttr(op.getOperation(), inferred,
                            [this] { signalPassFailure(); }, isAmbiguous);

      // If this make_tensor_view layout was inferred in an ambiguous ND/DN
      // shape and a downstream tile has a clear BLayout preference, force-align
      // to that preference to avoid GlobalTensor/Tile mismatch.
      if (isAmbiguous && isMinorColsOne(shape) &&
          op->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
        auto cur = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
        if (cur && pref.preferred && *pref.preferred != cur.getLayout())
          setLayoutAttr(op.getOperation(), *pref.preferred, /*inferred=*/true);
      }
    });

    // ------------------------------------------------------------------
    // 2) memref.reinterpret_cast (lowered from make_tensor_view)
    // ------------------------------------------------------------------
    func.walk([&](memref::ReinterpretCastOp op) {
      auto mrTy = dyn_cast<MemRefType>(op.getType());
      if (!mrTy || !isGlobalMemRef(mrTy))
        return;

      const size_t rank = op.getMixedSizes().size();
      if (rank == 0 || rank > 5) {
        verifyOrSetLayoutAttr(op.getOperation(), std::nullopt,
                              [this] { signalPassFailure(); });
        return;
      }

      SmallVector<int64_t> shape;
      shape.reserve(rank);
      for (OpFoldResult s : op.getMixedSizes()) {
        auto v = getConstInt(s);
        if (!v) {
          verifyOrSetLayoutAttr(op.getOperation(), std::nullopt,
                                [this] { signalPassFailure(); });
          return;
        }
        shape.push_back(*v);
      }

      SmallVector<int64_t> strides;
      strides.reserve(rank);
      for (OpFoldResult s : op.getMixedStrides()) {
        auto v = getConstInt(s);
        if (!v) {
          verifyOrSetLayoutAttr(op.getOperation(), std::nullopt,
                                [this] { signalPassFailure(); });
          return;
        }
        strides.push_back(*v);
      }

      bool isMinor2DAmbiguous = false;
      auto inferred =
          inferLayout5D(shape, strides, elemByteSize(mrTy.getElementType()),
                        std::nullopt, &isMinor2DAmbiguous);
      verifyOrSetLayoutAttr(op.getOperation(), inferred,
                            [this] { signalPassFailure(); },
                            isMinor2DAmbiguous);
    });

    // ------------------------------------------------------------------
    // 3) memref.subview: layout is preserved from the source view
    // ------------------------------------------------------------------
    func.walk([&](memref::SubViewOp op) {
      auto resTy = dyn_cast<MemRefType>(op.getType());
      if (!resTy || !isGlobalMemRef(resTy))
        return;

      if (op->getAttrOfType<LayoutAttr>(kLayoutAttrName))
        return;

      if (Operation *def = op.getSource().getDefiningOp()) {
        if (auto srcLayout = def->getAttrOfType<LayoutAttr>(kLayoutAttrName)) {
          op->setAttr(kLayoutAttrName, srcLayout);
          if (auto inferred =
                  def->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
            op->setAttr(kInferredLayoutAttrName, inferred);
          }
          return;
        }
      }

      // Fallback: if source memref type is fully static, infer from it.
      auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
      if (!srcTy || !srcTy.hasStaticShape()) {
        setLayoutAttr(op.getOperation(), Layout::ND, /*inferred=*/true);
        return;
      }

      SmallVector<int64_t> strideInts;
      int64_t offset = ShapedType::kDynamic;
      if (failed(getStridesAndOffset(srcTy, strideInts, offset)) ||
          offset == ShapedType::kDynamic ||
          llvm::any_of(strideInts,
                       [](int64_t s) { return s == ShapedType::kDynamic; })) {
        setLayoutAttr(op.getOperation(), Layout::ND, /*inferred=*/true);
        return;
      }

      auto inferred = inferLayout5D(srcTy.getShape(), strideInts,
                                    elemByteSize(srcTy.getElementType()));
      setLayoutAttr(op.getOperation(), inferred.value_or(Layout::ND),
                    /*inferred=*/true);
    });

    // ------------------------------------------------------------------
    // 4) pto.tload / pto.tstore: attach layout for static GM memrefs so EmitC
    //    doesn't need to infer again in buildGlobalTensorFromMemref().
    // ------------------------------------------------------------------
    func.walk([&](pto::TLoadOp op) {
      attachLoadStoreLayout(op, [](auto load) { return load.getSrc(); },
                            [](auto load) { return load.getDst(); });
    });

    func.walk([&](pto::TStoreOp op) {
      attachLoadStoreLayout(op, [](auto store) { return store.getDst(); },
                            [](auto store) { return store.getSrc(); });
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOLayoutPass() {
  return std::make_unique<InferPTOLayoutPass>();
}
