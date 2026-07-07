// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOViewToMemref.cpp ------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Lower PTO tile/view operations to memref-based IR while preserving tile
// metadata through binding ops and SSA backtracking.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVIEWTOMEMREF
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "Utils.h" // 假设包含一些通用的工具函数

#include <algorithm>
#include <functional>
#include <limits>

#define DEBUG_TYPE "pto-view-to-memref"

using namespace mlir;

namespace mlir {
namespace pto {

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

namespace {

static void markForceDynamicValidShape(Operation *op, bool force,
                                       MLIRContext *ctx);

static Type convertPTOTypeToMemRef(Type t);

constexpr size_t kTileRank2D = 2;
constexpr size_t kRowDimensionIndex = 0;
constexpr size_t kColumnDimensionIndex = 1;
constexpr unsigned kShapeVectorInlineCapacity = 4;
constexpr unsigned kOperationVectorInlineCapacity = 8;

constexpr int64_t kElementBytes1 = 1;
constexpr int64_t kElementBytes2 = 2;
constexpr int64_t kElementBytes4 = 4;
constexpr int64_t kElementBytes8 = 8;
constexpr int64_t kElementBytes16 = 16;
constexpr int64_t kElementBytes32 = 32;

constexpr int64_t kInnerExtent1 = 1;
constexpr int64_t kInnerExtent2 = 2;
constexpr int64_t kInnerExtent4 = 4;
constexpr int64_t kInnerExtent8 = 8;
constexpr int64_t kInnerExtent16 = 16;
constexpr int64_t kInnerExtent32 = 32;

constexpr int32_t kFractalSize32 = 32;
constexpr int32_t kFractalSize512 = 512;
constexpr int32_t kFractalSize1024 = 1024;

constexpr int32_t kBLayoutColMajor =
    static_cast<int32_t>(BLayout::ColMajor);
constexpr int32_t kSLayoutNoneBox =
    static_cast<int32_t>(SLayout::NoneBox);
constexpr int32_t kSLayoutRowMajor =
    static_cast<int32_t>(SLayout::RowMajor);
constexpr int32_t kSLayoutColMajor =
    static_cast<int32_t>(SLayout::ColMajor);
constexpr int32_t kCompactModeRowPlusOne =
    static_cast<int32_t>(CompactMode::RowPlusOne);

constexpr unsigned kThirdOperandIndex = 2;
constexpr unsigned kFourthOperandIndex = 3;
constexpr unsigned kFifthOperandIndex = 4;
constexpr unsigned kSixthOperandIndex = 5;

template <typename T>
using SmallInlineVector = SmallVector<T, kShapeVectorInlineCapacity>;

template <typename T>
using DefaultInlineVector = SmallVector<T, kOperationVectorInlineCapacity>;

// =============================================================================
// Helper: Metadata Backtracking (核心机制)
// =============================================================================
// 从一个 MemRef Value 向上回溯，找到它绑定的 TileBufConfig。
// 这解决了 "Type Erasure" 问题：memref 类型本身不包含 config，但 SSA 定义链包含。
static mlir::pto::TileBufConfigAttr lookupConfig(Value v) {
  // 1. 最直接的情况：它就是 bind_tile 的结果
  if (auto bind = v.getDefiningOp<mlir::pto::BindTileOp>()) {
    return bind.getConfig();
  }
  // PointerCastOp can also carry tile metadata (used when alloc_tile specifies
  // an explicit address).
  if (auto pc = v.getDefiningOp<mlir::pto::PointerCastOp>()) {
    if (auto cfg = pc.getConfig())
      return *cfg;
    return {};
  }
  
  // 2. 穿透 View 操作 (SubView, Cast 等) 向上查找
  if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
    return lookupConfig(subview.getSource());
  }
  if (auto cast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    return lookupConfig(cast.getSource());
  }
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    return lookupConfig(cast.getSource());
  }
  if (auto slot = v.getDefiningOp<mlir::pto::SlotMarkerOp>()) {
    return lookupConfig(slot.getSource());
  }

  // 3. pto.fusion_region result 本身不携带 config；作为兜底情况，沿着
  //    pto.yield 回溯到 region 内真正的 tile handle/memref 定义链继续查找。
  if (auto regionResult = dyn_cast<OpResult>(v)) {
    if (auto fusionRegion =
            dyn_cast<mlir::pto::FusionRegionOp>(regionResult.getOwner())) {
      auto yieldOp = dyn_cast<mlir::pto::YieldOp>(
          fusionRegion.getBody().front().getTerminator());
      if (!yieldOp)
        return {};
      unsigned resultIndex = regionResult.getResultNumber();
      if (resultIndex >= yieldOp.getNumOperands())
        return {};
      return lookupConfig(yieldOp.getOperand(resultIndex));
    }
  }

  // 如果追溯到 BlockArgument (函数参数) 或其他无法穿透的 Op，则返回空
  return {};
}

// =============================================================================
// Helper: Valid dims backtracking (v_row / v_col)
// =============================================================================
static void lookupValidDims(Value v, Value &vRow, Value &vCol) {
  if (auto alloc = v.getDefiningOp<mlir::pto::AllocTileOp>()) {
    vRow = alloc.getValidRow();
    vCol = alloc.getValidCol();
    return;
  }
  if (auto bind = v.getDefiningOp<mlir::pto::BindTileOp>()) {
    vRow = bind.getValidRow();
    vCol = bind.getValidCol();
    return;
  }
  if (auto pc = v.getDefiningOp<mlir::pto::PointerCastOp>()) {
    vRow = pc.getValidRow();
    vCol = pc.getValidCol();
    return;
  }
  if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
    lookupValidDims(subview.getSource(), vRow, vCol);
    return;
  }
  if (auto cast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    lookupValidDims(cast.getSource(), vRow, vCol);
    return;
  }
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    lookupValidDims(cast.getSource(), vRow, vCol);
    return;
  }
  if (auto slot = v.getDefiningOp<mlir::pto::SlotMarkerOp>()) {
    lookupValidDims(slot.getSource(), vRow, vCol);
    return;
  }

  // pto.fusion_region result 不直接携带 valid dims；作为兜底情况，沿着
  // pto.yield 回溯到 region 内真正的 tile handle/memref 定义链继续查找。
  if (auto regionResult = dyn_cast<OpResult>(v)) {
    if (auto fusionRegion =
            dyn_cast<mlir::pto::FusionRegionOp>(regionResult.getOwner())) {
      auto yieldOp = dyn_cast<mlir::pto::YieldOp>(
          fusionRegion.getBody().front().getTerminator());
      if (!yieldOp) {
        vRow = Value();
        vCol = Value();
        return;
      }
      unsigned resultIndex = regionResult.getResultNumber();
      if (resultIndex >= yieldOp.getNumOperands()) {
        vRow = Value();
        vCol = Value();
        return;
      }
      lookupValidDims(yieldOp.getOperand(resultIndex), vRow, vCol);
      return;
    }
  }

  vRow = Value();
  vCol = Value();
}

template <typename OpTy, typename... Args>
static OpTy replaceOpWithClonedAttrs(IRRewriter &rewriter, Operation *op,
                                     Args &&...args) {
  auto newOp =
      rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  newOp->setAttrs(op->getAttrs());
  rewriter.replaceOp(op, newOp->getResults());
  return newOp;
}

static Value resolveTileBufViewLikeSource(Value src) {
  if (isa<MemRefType>(src.getType()))
    return src;

  if (auto regionResult = dyn_cast<OpResult>(src)) {
    if (auto fusionRegion =
            dyn_cast<mlir::pto::FusionRegionOp>(regionResult.getOwner())) {
      auto yieldOp = dyn_cast<mlir::pto::YieldOp>(
          fusionRegion.getBody().front().getTerminator());
      if (!yieldOp)
        return Value();
      unsigned resultIndex = regionResult.getResultNumber();
      if (resultIndex >= yieldOp.getNumOperands())
        return Value();
      return resolveTileBufViewLikeSource(yieldOp.getOperand(resultIndex));
    }
  }

  return Value();
}

static LogicalResult synthesizeMissingTQuantTmpOps(func::FuncOp func,
                                                   MLIRContext *ctx) {
  if (mlir::pto::getTargetArch(func.getOperation()) == mlir::pto::PTOArch::A5)
    return success();

  DefaultInlineVector<mlir::pto::TQuantOp> quantOps;
  func.walk([&](mlir::pto::TQuantOp op) { quantOps.push_back(op); });

  for (auto op : quantOps) {
    if (op.getTmp())
      continue;

    auto srcTy = dyn_cast<mlir::pto::TileBufType>(op.getSrc().getType());
    if (!srcTy)
      continue;

    if (srcTy.getRank() != static_cast<int64_t>(kTileRank2D)) {
      op.emitError("expects rank-2 src tile to synthesize tquant tmp");
      return failure();
    }

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    SmallInlineVector<int64_t> shape(srcTy.getShape().begin(),
                                     srcTy.getShape().end());
    SmallInlineVector<int64_t> validShape(srcTy.getValidShape().begin(),
                                          srcTy.getValidShape().end());
    if (validShape.empty())
      validShape.assign(shape.begin(), shape.end());

    auto tmpTy = mlir::pto::TileBufType::get(
        ctx, shape, srcTy.getElementType(), srcTy.getMemorySpace(), validShape,
        mlir::pto::TileBufConfigAttr::getDefault(ctx));
    Value validRow;
    Value validCol;
    lookupValidDims(op.getSrc(), validRow, validCol);
    Value tmp = rewriter
                    .create<mlir::pto::AllocTileOp>(loc, tmpTy, Value(),
                                                    validRow, validCol)
                    .getResult();

    rewriter.create<mlir::pto::TQuantOp>(
        loc, TypeRange{}, op.getSrc(), op.getFp(), op.getOffset(), tmp,
        op.getDst(), op.getQuantTypeAttr());
    rewriter.eraseOp(op);
  }

  return success();
}

// =============================================================================
// Helper Functions for Layout Normalization
// =============================================================================

struct TileLayoutInfo {
  int64_t rowStride = 1;
  int64_t colStride = 1;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  bool boxed = false; // slayout != NoneBox
};

struct TileLayoutConfig {
  int32_t bLayout = 0;
  int32_t sLayout = 0;
  int32_t fractalSize = kFractalSize512;
  int32_t compactMode = 0;
};

static int64_t getElemBytes(Type elemTy) {
  unsigned bytes = getPTOStorageElemByteSize(elemTy);
  return bytes == 0 ? -1 : static_cast<int64_t>(bytes);
}

template <typename EnumAttrTy>
static bool readEnumAttrOrIntegerI32(Attribute attr, int32_t &out) {
  if (auto enumAttr = dyn_cast<EnumAttrTy>(attr)) {
    out = static_cast<int32_t>(enumAttr.getValue());
    return true;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
  }
  return false;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  return readEnumAttrOrIntegerI32<BLayoutAttr>(attr, out);
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  return readEnumAttrOrIntegerI32<SLayoutAttr>(attr, out);
}

static bool readCompactModeI32(Attribute attr, int32_t &out) {
  return readEnumAttrOrIntegerI32<CompactModeAttr>(attr, out);
}

static Value peelIndexLikeCast(Value value) {
  while (true) {
    if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
      value = castOp.getIn();
      continue;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtSIOp>()) {
      value = extOp.getIn();
      continue;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
      value = extOp.getIn();
      continue;
    }
    if (auto truncOp = value.getDefiningOp<arith::TruncIOp>()) {
      value = truncOp.getIn();
      continue;
    }
    return value;
  }
}

static bool getConstIndexValue(Value value, int64_t &out) {
  value = peelIndexLikeCast(value);
  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = constIndex.value();
    return true;
  }
  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = constInt.value();
    return true;
  }
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  auto intAttr =
      constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr();
  if (!intAttr)
    return false;
  out = intAttr.getInt();
  return true;
}

static TileLayoutConfig getTileLayoutConfig(mlir::pto::TileBufConfigAttr cfg) {
  TileLayoutConfig config;
  (void)readBLayoutI32(cfg.getBLayout(), config.bLayout);
  (void)readSLayoutI32(cfg.getSLayout(), config.sLayout);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    config.fractalSize = static_cast<int32_t>(attr.getInt());
  (void)readCompactModeI32(cfg.getCompactMode(), config.compactMode);
  return config;
}

static bool getFractal512InnerExtent(int64_t elemBytes, int64_t &extent) {
  switch (elemBytes) {
  case kElementBytes1:
    extent = kInnerExtent32;
    return true;
  case kElementBytes2:
    extent = kInnerExtent16;
    return true;
  case kElementBytes4:
    extent = kInnerExtent8;
    return true;
  case kElementBytes8:
    extent = kInnerExtent4;
    return true;
  case kElementBytes16:
    extent = kInnerExtent2;
    return true;
  case kElementBytes32:
    extent = kInnerExtent1;
    return true;
  default:
    return false;
  }
}

static bool computeBoxInnerShape(const TileLayoutConfig &config, Type elemTy,
                                 TileLayoutInfo &info) {
  info.boxed = config.sLayout != kSLayoutNoneBox;
  if (!info.boxed) {
    info.innerRows = kInnerExtent1;
    info.innerCols = kInnerExtent1;
    return true;
  }

  int64_t elemBytes = getElemBytes(elemTy);
  if (elemBytes <= 0)
    return false;

  switch (config.fractalSize) {
  case kFractalSize1024:
    info.innerRows = kInnerExtent16;
    info.innerCols = kInnerExtent16;
    return true;
  case kFractalSize32:
    info.innerRows = kInnerExtent16;
    info.innerCols = kInnerExtent2;
    return true;
  case kFractalSize512:
    if (config.sLayout == kSLayoutRowMajor) {
      info.innerRows = kInnerExtent16;
      return getFractal512InnerExtent(elemBytes, info.innerCols);
    }
    if (config.sLayout == kSLayoutColMajor) {
      if (!getFractal512InnerExtent(elemBytes, info.innerRows))
        return false;
      info.innerCols = kInnerExtent16;
      return true;
    }
    return false;
  default:
    return false;
  }
}

static bool computeTilePointerStrides(const TileLayoutConfig &config,
                                      ArrayRef<int64_t> shape,
                                      TileLayoutInfo &info) {
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  auto applyCompactToMajorStride = [&](int64_t majorStride) -> int64_t {
    if (config.compactMode == kCompactModeRowPlusOne)
      return majorStride + kInnerExtent1;
    return majorStride;
  };
  if (!info.boxed) {
    if (config.bLayout == kBLayoutColMajor) {
      info.rowStride = kInnerExtent1;
      info.colStride = applyCompactToMajorStride(rows);
      return true;
    }
    info.rowStride = applyCompactToMajorStride(cols);
    info.colStride = kInnerExtent1;
    return true;
  }

  if (config.bLayout == kBLayoutColMajor) {
    if (config.sLayout != kSLayoutRowMajor)
      return false;
    info.rowStride = info.innerCols;
    info.colStride = applyCompactToMajorStride(rows);
    return true;
  }

  info.rowStride = applyCompactToMajorStride(cols);
  info.colStride = info.innerRows;
  return true;
}

static bool computeTileLayoutInfo(mlir::pto::TileBufConfigAttr cfg, Type elemTy,
                                  ArrayRef<int64_t> shape,
                                  TileLayoutInfo &info) {
  if (shape.size() != kTileRank2D ||
      llvm::is_contained(shape, ShapedType::kDynamic))
    return false;

  TileLayoutConfig config = getTileLayoutConfig(cfg);
  return computeBoxInnerShape(config, elemTy, info) &&
         computeTilePointerStrides(config, shape, info);
}

static void collectAffineAddTerms(AffineExpr root,
                                  SmallVectorImpl<AffineExpr> &terms) {
  SmallInlineVector<AffineExpr> pending{root};
  while (!pending.empty()) {
    AffineExpr current = pending.pop_back_val();
    auto addExpr = llvm::dyn_cast<AffineBinaryOpExpr>(current);
    if (!addExpr || addExpr.getKind() != AffineExprKind::Add) {
      terms.push_back(current);
      continue;
    }
    pending.push_back(addExpr.getRHS());
    pending.push_back(addExpr.getLHS());
  }
}

static bool tryAssignAffineStride(AffineExpr expr,
                                  MutableArrayRef<int64_t> strides) {
  if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
    strides[dim.getPosition()] = 1;
    return true;
  }

  auto mulExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;

  auto assignStride = [&](AffineExpr dimExpr,
                          AffineExpr constantExpr) -> bool {
    auto dim = llvm::dyn_cast<AffineDimExpr>(dimExpr);
    auto constant = llvm::dyn_cast<AffineConstantExpr>(constantExpr);
    if (!dim || !constant)
      return false;
    strides[dim.getPosition()] = constant.getValue();
    return true;
  };
  return assignStride(mulExpr.getLHS(), mulExpr.getRHS()) ||
         assignStride(mulExpr.getRHS(), mulExpr.getLHS());
}

[[maybe_unused]] static void decomposeStridedLayout(AffineMap map,
                                   SmallVectorImpl<int64_t> &strides) {
  strides.assign(map.getNumDims(), 0);
  if (map.getNumResults() != 1)
    return;

  SmallInlineVector<AffineExpr> terms;
  collectAffineAddTerms(map.getResult(0), terms);
  for (AffineExpr term : terms)
    (void)tryAssignAffineStride(term, strides);
}

static Value makeIndexConstant(IRRewriter &rewriter, Location loc,
                               int64_t value) {
  return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                            rewriter.getIndexAttr(value));
}

static Value buildFlattenedPtrLikeMemRefView(IRRewriter &rewriter, Location loc,
                                             Value sourceMemref,
                                             MemRefType targetType,
                                             Operation *anchorOp) {
  auto sourceType = dyn_cast<BaseMemRefType>(sourceMemref.getType());
  if (!sourceType) {
    anchorOp->emitError(
        "tile_buf_addr ptr-like lowering expects the source to be memref-backed");
    return Value();
  }

  Value totalElems = makeIndexConstant(rewriter, loc, 1);
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    Value extent;
    if (sourceType.isDynamicDim(dim))
      extent = rewriter.create<memref::DimOp>(loc, sourceMemref, dim);
    else
      extent = makeIndexConstant(rewriter, loc, sourceType.getDimSize(dim));
    totalElems = rewriter.create<arith::MulIOp>(loc, totalElems, extent);
  }

  SmallVector<OpFoldResult, 1> sizes{totalElems};
  SmallVector<OpFoldResult, 1> strides{rewriter.getIndexAttr(1)};
  return rewriter
      .create<memref::ReinterpretCastOp>(loc, targetType, sourceMemref,
                                         rewriter.getIndexAttr(0), sizes,
                                         strides)
      .getResult();
}

static SmallVector<int64_t> computeCompactStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    if (shape[i] != ShapedType::kDynamic)
      stride *= shape[i];
  }
  return strides;
}

static void materializeStaticValidDims(IRRewriter &rewriter, Location loc,
                                       mlir::pto::TileBufType tbTy, Value &vRow,
                                       Value &vCol) {
  ArrayRef<int64_t> validShape = tbTy.getValidShape();
  if (tbTy.hasDynamicValid())
    return;
  if (!validShape.empty() && validShape[kRowDimensionIndex] >= 0)
    vRow = makeIndexConstant(rewriter, loc, validShape[kRowDimensionIndex]);
  if (validShape.size() >= kTileRank2D &&
      validShape[kColumnDimensionIndex] >= 0)
    vCol = makeIndexConstant(rewriter, loc, validShape[kColumnDimensionIndex]);
}

static bool checkMultipleOf(Operation *op, int64_t value, int64_t divisor,
                            StringRef label) {
  if (divisor <= 0) {
    op->emitError("boxed layout requires positive divisor for ") << label;
    return false;
  }
  if (value % divisor == 0)
    return true;
  op->emitError("boxed layout requires ")
      << label << " multiple of " << divisor << ", got " << value;
  return false;
}

// 确保 Value 是 Index 类型
static Value ensureIndex(IRRewriter &rewriter, Location loc, Value v,
                         Operation *anchorOp) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), v);
  if (anchorOp)
    anchorOp->emitError() << "expected index or integer, but got " << v.getType();
  return Value();
}

static bool tryGetIndexAttrFromValue(IRRewriter &rewriter, Value v,
                                     IntegerAttr &constAttr) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    constAttr = rewriter.getIndexAttr(cOp.value());
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    constAttr = rewriter.getIndexAttr(cInt.value());
    return true;
  }
  return false;
}

static void appendMixedIndex(IRRewriter &rewriter, Location loc, Value v,
                             Operation *anchorOp,
                             SmallVectorImpl<OpFoldResult> &mixedVals) {
  IntegerAttr constAttr;
  if (tryGetIndexAttrFromValue(rewriter, v, constAttr)) {
    mixedVals.push_back(constAttr);
    return;
  }
  mixedVals.push_back(ensureIndex(rewriter, loc, v, anchorOp));
}

static bool foldAddPtrChainIntoOffset(IRRewriter &rewriter, Location loc,
                                      Value &base, Value &totalOffset) {
  bool folded = false;
  while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
    folded = true;
    Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
    totalOffset =
        totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off) : off;
    base = add.getOperand(0);
  }
  return folded;
}

static Value clampSubViewValidDim(IRRewriter &rewriter, Location loc,
                                  Value explicitValid, int64_t size,
                                  int64_t inferredValid, Operation *anchorOp) {
  if (!explicitValid) {
    // No explicit valid operand: take the valid extent the result type
    // declares. For an ordinary subview this equals `size`; for an empty
    // tail/no-op-replay tile the type carries 0, which must survive to
    // bind_tile rather than being widened back to `size`. A dynamic declared
    // extent is materialized via markForceDynamicValidShape, so fall back to
    // `size` here.
    int64_t fallback = inferredValid >= 0 ? inferredValid : size;
    return rewriter.create<arith::ConstantIndexOp>(loc, fallback);
  }

  Value sizeVal = rewriter.create<arith::ConstantIndexOp>(loc, size);
  int64_t cst = 0;
  if (getConstIndexValue(explicitValid, cst))
    return rewriter.create<arith::ConstantIndexOp>(loc, std::min(cst, size));

  Value v = ensureIndex(rewriter, loc, explicitValid, anchorOp);
  Value lt = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, v,
                                            sizeVal);
  return rewriter.create<arith::SelectOp>(loc, lt, v, sizeVal);
}

[[maybe_unused]] static void dumpPretty(Operation *op, llvm::raw_ostream &os) {
  OpPrintingFlags flags;
  flags.useLocalScope();            
  AsmState state(op, flags);
  op->print(os, state);
  os << "\n";
  os.flush();
}

// =============================================================================
// Type Converter Logic
// =============================================================================

static SmallVector<int64_t> buildTileMemRefStrides(mlir::pto::TileBufType tbTy) {
  TileLayoutInfo info;
  if (computeTileLayoutInfo(tbTy.getConfigAttr(), tbTy.getElementType(),
                            tbTy.getShape(), info)) {
    return {info.rowStride, info.colStride};
  }
  return computeCompactStrides(tbTy.getShape());
}

static Type convertTileBufTypeToMemRef(mlir::pto::TileBufType tbTy) {
  auto layoutAttr = StridedLayoutAttr::get(tbTy.getContext(),
                                           ShapedType::kDynamic,
                                           buildTileMemRefStrides(tbTy));
  return MemRefType::get(tbTy.getShape(), tbTy.getElementType(), layoutAttr,
                         tbTy.getMemorySpace());
}

static Type convertTensorViewLikeTypeToMemRef(ArrayRef<int64_t> shape,
                                              Type elementType,
                                              MLIRContext *ctx,
                                              bool forceDynamicShape) {
  SmallVector<int64_t, 4> resultShape;
  if (forceDynamicShape) {
    resultShape.assign(shape.size(), ShapedType::kDynamic);
  } else {
    resultShape.assign(shape.begin(), shape.end());
  }

  SmallVector<int64_t, 4> dynStrides(shape.size(), ShapedType::kDynamic);
  auto layoutAttr =
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dynStrides);
  auto gmSpace = AddressSpaceAttr::get(ctx, AddressSpace::GM);
  return MemRefType::get(resultShape, elementType, layoutAttr, gmSpace);
}

static Type convertPTOTypeToMemRef(Type t) {
  // 1. 处理 !pto.ptr<T>
  if (auto pty = dyn_cast<mlir::pto::PtrType>(t)) {
    return MemRefType::get({ShapedType::kDynamic}, pty.getElementType(),
                           MemRefLayoutAttrInterface(), pty.getMemorySpace());
  }

  // 2. 处理 !pto.tile_buf<...>
  if (auto tbTy = dyn_cast<mlir::pto::TileBufType>(t))
    return convertTileBufTypeToMemRef(tbTy);
  // 3. !pto.multi_tile_buf<S, count=N>: collapses to the slot memref shape;
  //    the N-slot fan-out lives on the `pto.multi_buffer` attr written by
  //    the alloc lowering, not in the type. This branch is defensive --
  //    by design multi_tile_buf does not appear on function boundaries.
  if (auto mtbTy = dyn_cast<mlir::pto::MultiTileBufType>(t))
    return convertTileBufTypeToMemRef(mtbTy.getSlotType());
  if (auto tvTy = dyn_cast<mlir::pto::TensorViewType>(t))
    return convertTensorViewLikeTypeToMemRef(tvTy.getShape(),
                                             tvTy.getElementType(),
                                             tvTy.getContext(),
                                             /*forceDynamicShape=*/true);
  if (auto partTy = dyn_cast<mlir::pto::PartitionTensorViewType>(t))
    return convertTensorViewLikeTypeToMemRef(partTy.getShape(),
                                             partTy.getElementType(),
                                             partTy.getContext(),
                                             /*forceDynamicShape=*/false);
  // 其他类型透传
  return t;
}

// Ensure scf.if result types follow the rewritten yield operand types.
// PTOViewToMemref rewrites tile values to memref in branch bodies, but scf.if
// result types are not auto-updated by those op-local rewrites.
static LogicalResult reconcileSCFIfResultTypes(func::FuncOp func) {
  DefaultInlineVector<scf::IfOp> ifOps;
  func.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  for (scf::IfOp ifOp : ifOps) {
    if (ifOp.getNumResults() == 0)
      continue;

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (!thenYield || !elseYield) {
      ifOp.emitError("result-bearing scf.if must end with scf.yield in both "
                     "then/else regions");
      return failure();
    }

    if (thenYield.getNumOperands() != ifOp.getNumResults() ||
        elseYield.getNumOperands() != ifOp.getNumResults()) {
      ifOp.emitError("scf.if result count does not match yielded values");
      return failure();
    }

    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Type thenTy = thenYield.getOperand(i).getType();
      Type elseTy = elseYield.getOperand(i).getType();
      if (thenTy != elseTy) {
        ifOp.emitError() << "scf.if branch yield type mismatch at result #" << i
                         << ": then=" << thenTy << ", else=" << elseTy;
        return failure();
      }

      if (ifOp.getResult(i).getType() != thenTy)
        ifOp.getResult(i).setType(thenTy);
    }
  }

  return success();
}

static LogicalResult reconcileSCFForResultTypes(func::FuncOp func) {
  DefaultInlineVector<scf::ForOp> forOps;
  func.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  for (scf::ForOp forOp : forOps) {
    if (forOp.getNumResults() == 0)
      continue;

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield) {
      forOp.emitError("result-bearing scf.for must end with scf.yield");
      return failure();
    }

    if (yield.getNumOperands() != forOp.getNumResults() ||
        forOp.getInitArgs().size() != forOp.getNumResults()) {
      forOp.emitError("scf.for result count does not match iter/yield values");
      return failure();
    }

    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      Type initTy = forOp.getInitArgs()[i].getType();
      Type yieldTy = yield.getOperand(i).getType();
      if (initTy != yieldTy) {
        forOp.emitError() << "scf.for init/yield type mismatch at result #" << i
                          << ": init=" << initTy << ", yield=" << yieldTy;
        return failure();
      }

      BlockArgument iterArg = forOp.getRegionIterArg(i);
      if (iterArg.getType() != initTy)
        iterArg.setType(initTy);
      if (forOp.getResult(i).getType() != initTy)
        forOp.getResult(i).setType(initTy);
    }
  }

  return success();
}

// Ensure pto.fusion_region result types follow the rewritten pto.yield operand
// types. PTOViewToMemref rewrites region-local tile values to memref, but the
// region result types are not auto-updated by those op-local rewrites.
static LogicalResult reconcileFusionRegionResultTypes(func::FuncOp func) {
  SmallVector<pto::FusionRegionOp, 8> fusionRegions;
  func.walk([&](pto::FusionRegionOp fusionRegion) {
    fusionRegions.push_back(fusionRegion);
  });

  for (pto::FusionRegionOp fusionRegion : fusionRegions) {
    if (fusionRegion.getNumResults() == 0)
      continue;

    auto yieldOp =
        dyn_cast<pto::YieldOp>(fusionRegion.getBody().front().getTerminator());
    if (!yieldOp) {
      fusionRegion.emitError("result-bearing pto.fusion_region must end with "
                             "pto.yield");
      return failure();
    }

    if (yieldOp.getNumOperands() != fusionRegion.getNumResults()) {
      fusionRegion.emitError(
          "pto.fusion_region result count does not match yielded values");
      return failure();
    }

    for (unsigned i = 0; i < fusionRegion.getNumResults(); ++i) {
      Type yieldedTy = yieldOp.getOperand(i).getType();
      if (fusionRegion.getResult(i).getType() != yieldedTy)
        fusionRegion.getResult(i).setType(yieldedTy);
    }
  }

  return success();
}

static LogicalResult markLoweredSetValidShapeOps(func::FuncOp func,
                                                 MLIRContext *ctx) {
  WalkResult result = func.walk([&](mlir::pto::SetValidShapeOp op) {
    if (isa<MemRefType>(op.getSource().getType())) {
      if (!lookupConfig(op.getSource())) {
        op.emitError(
            "set_validshape requires a locally bound tile source; function "
            "arguments/results are unsupported");
        return WalkResult::interrupt();
      }
      op->setAttr(kLoweredSetValidShapeAttrName, UnitAttr::get(ctx));
      return WalkResult::advance();
    }
    op->removeAttr(kLoweredSetValidShapeAttrName);
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static void markForceDynamicValidShape(Operation *op, bool force,
                                       MLIRContext *ctx) {
  if (force) {
    op->setAttr(kForceDynamicValidShapeAttrName, UnitAttr::get(ctx));
    return;
  }
  op->removeAttr(kForceDynamicValidShapeAttrName);
}

[[maybe_unused]] static void rewriteFunctionSignature(func::FuncOp func, MLIRContext *ctx) {
  Block &entry = func.front();
  auto fnTy = func.getFunctionType();

  SmallVector<Type> newInputs;
  for (Type type : fnTy.getInputs())
    newInputs.push_back(convertPTOTypeToMemRef(type));

  SmallVector<Type> newResults;
  for (Type type : fnTy.getResults())
    newResults.push_back(convertPTOTypeToMemRef(type));

  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    if (entry.getArgument(i).getType() != newInputs[i])
      entry.getArgument(i).setType(newInputs[i]);
  }
  func.setFunctionType(FunctionType::get(ctx, newInputs, newResults));
}

[[maybe_unused]] static LogicalResult lowerAllocTileOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::AllocTileOp> allocTiles;
  func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

  for (auto op : allocTiles) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy)
      continue;

    SmallInlineVector<int64_t> shape(tbTy.getShape().begin(),
                                  tbTy.getShape().end());
    Type elemTy = tbTy.getElementType();
    SmallVector<int64_t> strides = buildTileMemRefStrides(tbTy);

    auto targetLayout =
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
    auto targetType =
        MemRefType::get(shape, elemTy, targetLayout, tbTy.getMemorySpace());

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();
    materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

    auto configAttr = tbTy.getConfigAttr();
    if (!configAttr)
      configAttr = pto::TileBufConfigAttr::getDefault(ctx);

    if (Value addr = op.getAddr()) {
      auto pc = rewriter.create<pto::PointerCastOp>(
          loc, targetType, ValueRange{addr}, vRow ? vRow : Value(),
          vCol ? vCol : Value(), configAttr);
      markForceDynamicValidShape(pc, tbTy.hasDynamicValid(), ctx);
      auto bindOp = rewriter.create<pto::BindTileOp>(
          loc, targetType, pc.getResult(), vRow ? vRow : Value(),
          vCol ? vCol : Value(), configAttr);
      markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
      rewriter.replaceOp(op, bindOp.getResult());
      continue;
    }

    auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides);
    auto allocType =
        MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());
    Value alloc = rewriter.create<memref::AllocOp>(loc, allocType);
    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, targetType, alloc, vRow ? vRow : Value(), vCol ? vCol : Value(),
        configAttr);
    markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
    rewriter.replaceOp(op, bindOp.getResult());
  }
  return success();
}

[[maybe_unused]] static LogicalResult lowerDeclareTileOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::DeclareTileOp> declaredTiles;
  func.walk([&](mlir::pto::DeclareTileOp op) { declaredTiles.push_back(op); });

  for (auto op : declaredTiles) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getTile().getType());
    if (!tbTy) {
      op.emitError("declare_tile result must be tile_buf type");
      return failure();
    }

    auto targetType = dyn_cast<MemRefType>(convertPTOTypeToMemRef(tbTy));
    if (!targetType) {
      op.emitError("failed to convert declare_tile result to memref type");
      return failure();
    }

    auto configAttr = tbTy.getConfigAttr();
    if (!configAttr)
      configAttr = pto::TileBufConfigAttr::getDefault(ctx);

    Value vRow;
    Value vCol;
    materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

    auto declaredMemRef =
        rewriter.create<pto::DeclareTileMemRefOp>(loc, targetType);
    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, targetType, declaredMemRef.getResult(), vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
    rewriter.replaceOp(op, bindOp.getResult());
  }
  return success();
}

static Value castIndexToI64(IRRewriter &rewriter, Location loc, Value value) {
  Type i64Ty = rewriter.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  return rewriter.create<arith::IndexCastOp>(loc, i64Ty, value).getResult();
}

static FailureOr<Value>
materializePtrToIntAddPtrAddress(IRRewriter &rewriter, Location loc,
                                 mlir::pto::PtrToIntOp anchor, Value source) {
  SmallVector<mlir::pto::AddPtrOp, 4> addPtrChain;
  Value base = source;
  while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
    addPtrChain.push_back(add);
    base = add.getOperand(0);
  }

  if (addPtrChain.empty())
    return failure();

  auto baseMemTy = dyn_cast<MemRefType>(base.getType());
  if (!baseMemTy) {
    anchor.emitOpError(
        "pto.addptr source base could not be lowered to a GM memref");
    return failure();
  }

  Value byteAddress = rewriter.create<mlir::pto::PtrToIntOp>(
      loc, rewriter.getI64Type(), base);
  for (auto add : addPtrChain) {
    auto addPtrTy = dyn_cast<mlir::pto::PtrType>(add.getResult().getType());
    if (!addPtrTy) {
      anchor.emitOpError("requires pto.addptr source to have !pto.ptr result "
                         "type before byte-address lowering");
      return failure();
    }

    unsigned elemBytes =
        mlir::pto::getPTOStorageElemByteSize(addPtrTy.getElementType());
    if (elemBytes == 0) {
      anchor.emitOpError("cannot lower pto.addptr source with unknown element "
                         "byte size to a byte address");
      return failure();
    }

    Value byteOffset = castIndexToI64(rewriter, loc, add.getOffset());
    if (elemBytes != 1) {
      Value elemBytesValue =
          rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
      byteOffset =
          rewriter.create<arith::MulIOp>(loc, byteOffset, elemBytesValue)
              .getResult();
    }
    byteAddress =
        rewriter.create<arith::AddIOp>(loc, byteAddress, byteOffset).getResult();
  }

  return byteAddress;
}

static LogicalResult lowerIntToPtrOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::IntToPtrOp> intToPtrs;
  func.walk([&](mlir::pto::IntToPtrOp op) { intToPtrs.push_back(op); });

  for (auto op : intToPtrs) {
    if (!isa<mlir::pto::PtrType>(op.getResult().getType()))
      continue;

    auto targetTy =
        dyn_cast<MemRefType>(convertPTOTypeToMemRef(op.getResult().getType()));
    if (!targetTy) {
      op.emitError("failed to convert inttoptr result to memref type");
      return failure();
    }

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    auto lowered =
        rewriter.create<mlir::pto::IntToPtrOp>(op.getLoc(), targetTy,
                                               op.getAddr());
    lowered->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, lowered.getResult());
  }

  return success();
}

static LogicalResult lowerPtrToIntOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::PtrToIntOp> ptrToInts;
  func.walk([&](mlir::pto::PtrToIntOp op) { ptrToInts.push_back(op); });

  for (auto op : ptrToInts) {
    Value source = op.getPtr();
    if (source.getDefiningOp<mlir::pto::AddPtrOp>()) {
      IRRewriter rewriter(ctx);
      rewriter.setInsertionPoint(op);
      FailureOr<Value> byteAddress =
          materializePtrToIntAddPtrAddress(rewriter, op.getLoc(), op, source);
      if (failed(byteAddress))
        return failure();
      rewriter.replaceOp(op, *byteAddress);
      continue;
    }

    if (isa<mlir::pto::PtrType>(source.getType()))
      continue;
  }

  DefaultInlineVector<mlir::pto::PtrToIntOp> remaining;
  func.walk([&](mlir::pto::PtrToIntOp op) {
    if (isa<mlir::pto::PtrType>(op.getPtr().getType()))
      remaining.push_back(op);
  });
  for (auto op : remaining) {
    op.emitError("ptrtoint source could not be lowered to a GM memref");
    return failure();
  }

  return success();
}

[[maybe_unused]] static LogicalResult lowerMakeTensorViewOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::MakeTensorViewOp> makeViews;
  func.walk([&](mlir::pto::MakeTensorViewOp op) { makeViews.push_back(op); });

  for (auto op : makeViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value baseBuf = op.getOperand(0);
    OpFoldResult off0 = rewriter.getIndexAttr(0);
    bool foldedAddPtr = false;
    {
      Value cur = baseBuf;
      Value totalOffset;
      while (auto add = cur.getDefiningOp<mlir::pto::AddPtrOp>()) {
        foldedAddPtr = true;
        Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
        totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                                  : off;
        cur = add.getOperand(0);
      }
      if (cur != baseBuf) {
        baseBuf = cur;
        off0 = totalOffset ? OpFoldResult(totalOffset) : off0;
      }
    }

    auto baseMr = dyn_cast<BaseMemRefType>(baseBuf.getType());
    if (!baseMr) {
      op.emitError("make_tensor_view base must be memref");
      return failure();
    }

    size_t rank = op.getShape().size();
    int64_t dyn = ShapedType::kDynamic;
    SmallVector<int64_t> dynStrides(rank, dyn);
    auto layout =
        StridedLayoutAttr::get(ctx, /*offset=*/dyn, /*strides=*/dynStrides);
    SmallVector<int64_t> dynShape(rank, dyn);
    auto mrTy = MemRefType::get(dynShape, baseMr.getElementType(), layout,
                                baseMr.getMemorySpace());

    SmallInlineVector<OpFoldResult> sizes;
    for (Value value : op.getShape())
      sizes.push_back(ensureIndex(rewriter, loc, value, op));
    SmallInlineVector<OpFoldResult> strides;
    for (Value value : op.getStrides())
      strides.push_back(ensureIndex(rewriter, loc, value, op));

    auto rc = rewriter.create<memref::ReinterpretCastOp>(loc, mrTy, baseBuf, off0,
                                                         sizes, strides);
    if (foldedAddPtr)
      rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
    if (auto layoutAttr = op.getLayoutAttr())
      rc->setAttr("layout", layoutAttr);
    rewriter.replaceOp(op, rc.getResult());
  }
  return success();
}

static LogicalResult lowerPtrLikeTileBufAddrOps(func::FuncOp func,
                                                MLIRContext *ctx) {
  SmallVector<mlir::pto::TileBufAddrOp, 8> addrOps;
  func.walk([&](mlir::pto::TileBufAddrOp op) { addrOps.push_back(op); });

  for (auto op : addrOps) {
    if (!isa<mlir::pto::PtrType>(op.getDst().getType()))
      continue;

    auto targetType =
        dyn_cast<MemRefType>(convertPTOTypeToMemRef(op.getDst().getType()));
    if (!targetType) {
      op.emitError("failed to convert tile_buf_addr pointer result to memref");
      return failure();
    }

    Value source = op.getSrc();
    if (!isa<BaseMemRefType>(source.getType()))
      continue;

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Value replacement =
        buildFlattenedPtrLikeMemRefView(rewriter, op.getLoc(), source,
                                        targetType, op.getOperation());
    if (!replacement)
      return failure();
    rewriter.replaceOp(op, replacement);
  }
  return success();
}

[[maybe_unused]] static LogicalResult lowerTensorViewDimOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::GetTensorViewDimOp> tvDims;
  func.walk([&](mlir::pto::GetTensorViewDimOp op) { tvDims.push_back(op); });

  for (auto op : tvDims) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Value view = op.getTensorView();
    auto mrTy = dyn_cast<BaseMemRefType>(view.getType());
    if (!mrTy)
      continue;
    Value dim = rewriter.create<memref::DimOp>(op.getLoc(), view, op.getDimIndex());
    rewriter.replaceOp(op, dim);
  }
  return success();
}

[[maybe_unused]] static LogicalResult foldAddPtrIntoScalarOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::LoadScalarOp> loadScalars;
  func.walk([&](mlir::pto::LoadScalarOp op) { loadScalars.push_back(op); });
  for (auto op : loadScalars) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value base = op.getPtr();
    Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);
    bool foldedAddPtr = foldAddPtrChainIntoOffset(rewriter, loc, base, totalOffset);
    if (foldedAddPtr) {
      auto newOp =
          rewriter.create<pto::LoadScalarOp>(loc, op.getValue().getType(), base,
                                             totalOffset);
      rewriter.replaceOp(op, newOp.getValue());
    }
  }

  DefaultInlineVector<mlir::pto::StoreScalarOp> storeScalars;
  func.walk([&](mlir::pto::StoreScalarOp op) { storeScalars.push_back(op); });
  for (auto op : storeScalars) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value base = op.getPtr();
    Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);
    bool foldedAddPtr = foldAddPtrChainIntoOffset(rewriter, loc, base, totalOffset);
    if (foldedAddPtr) {
      rewriter.create<pto::StoreScalarOp>(loc, base, totalOffset, op.getValue());
      rewriter.eraseOp(op);
    }
  }

  DefaultInlineVector<Operation *> addPtrs;
  func.walk([&](mlir::pto::AddPtrOp op) { addPtrs.push_back(op.getOperation()); });
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &op : addPtrs) {
      if (!op)
        continue;
      if (op->use_empty()) {
        op->erase();
        op = nullptr;
        changed = true;
      }
    }
  }
  for (Operation *op : addPtrs) {
    if (!op)
      continue;
    op->emitError(
        "addptr must feed make_tensor_view or load/store_scalar for lowering");
    return failure();
  }
  return success();
}

static LogicalResult lowerPartitionViewOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::PartitionViewOp> partitionViews;
  func.walk([&](mlir::pto::PartitionViewOp op) { partitionViews.push_back(op); });

  for (auto op : partitionViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value src = op.getOperand(0);
    auto srcMrTy = dyn_cast<MemRefType>(src.getType());
    if (!srcMrTy)
      continue;
    int64_t rank = srcMrTy.getRank();

    SmallVector<int64_t> staticSizes;
    SmallVector<OpFoldResult> mixedSizes;
    for (Value size : op.getSizes()) {
      IntegerAttr constAttr;
      bool isStatic = false;
      if (auto cOp = size.getDefiningOp<arith::ConstantIndexOp>()) {
        constAttr = rewriter.getIndexAttr(cOp.value());
        isStatic = true;
      } else if (auto cInt = size.getDefiningOp<arith::ConstantIntOp>()) {
        constAttr = rewriter.getIndexAttr(cInt.value());
        isStatic = true;
      }

      if (isStatic) {
        mixedSizes.push_back(constAttr);
        staticSizes.push_back(constAttr.getInt());
      } else {
        mixedSizes.push_back(ensureIndex(rewriter, loc, size, op));
        staticSizes.push_back(ShapedType::kDynamic);
      }
    }

    SmallVector<OpFoldResult> mixedOffsets;
    for (Value offset : op.getOffsets()) {
      appendMixedIndex(rewriter, loc, offset, op, mixedOffsets);
    }

    int64_t dyn = ShapedType::kDynamic;
    SmallVector<int64_t> dynStrides(rank, dyn);
    auto layout = StridedLayoutAttr::get(ctx, dyn, dynStrides);
    auto resTy = MemRefType::get(staticSizes, srcMrTy.getElementType(), layout,
                                 srcMrTy.getMemorySpace());

    SmallVector<OpFoldResult> mixedStrides(rank, rewriter.getIndexAttr(1));
    auto sv = rewriter.create<memref::SubViewOp>(loc, resTy, src, mixedOffsets,
                                                 mixedSizes, mixedStrides);
    if (Operation *srcDef = src.getDefiningOp()) {
      if (auto layoutAttr = srcDef->getAttrOfType<pto::LayoutAttr>("layout"))
        sv->setAttr("layout", layoutAttr);
    }
    rewriter.replaceOp(op, sv.getResult());
  }
  return success();
}

static LogicalResult lowerSubViewOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::SubViewOp> subViews;
  func.walk([&](mlir::pto::SubViewOp op) { subViews.push_back(op); });

  for (auto op : subViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto resultTileTy =
        dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    Value src = op->getOperand(0);
    auto srcMrTy = dyn_cast<MemRefType>(src.getType());
    if (!srcMrTy) {
      op.emitError("pto.subview source must be lowered to memref first");
      return failure();
    }

    ArrayAttr sizeAttr = op.getSizes();
    SmallVector<int64_t> staticSizes;
    SmallVector<OpFoldResult> mixedSizes;
    for (Attribute attr : sizeAttr) {
      int64_t size = cast<IntegerAttr>(attr).getInt();
      staticSizes.push_back(size);
      mixedSizes.push_back(rewriter.getIndexAttr(size));
    }

    SmallVector<OpFoldResult> mixedOffsets;
    for (Value offset : op.getOffsets()) {
      appendMixedIndex(rewriter, loc, offset, op, mixedOffsets);
    }

    auto configAttr = lookupConfig(src);
    if (!configAttr)
      configAttr = pto::TileBufConfigAttr::getDefault(ctx);

    TileLayoutInfo layoutInfo;
    if (!computeTileLayoutInfo(configAttr, srcMrTy.getElementType(),
                               srcMrTy.getShape(), layoutInfo)) {
      op.emitError("unsupported tile layout for pto.subview");
      return failure();
    }

    if (layoutInfo.boxed) {
      if (staticSizes.size() != kTileRank2D ||
          op.getOffsets().size() != kTileRank2D) {
        op.emitError("boxed layout subview expects 2D sizes/offsets");
        return failure();
      }
      if (!checkMultipleOf(op, staticSizes[0], layoutInfo.innerRows, "row size") ||
          !checkMultipleOf(op, staticSizes[1], layoutInfo.innerCols, "col size")) {
        return failure();
      }

      int64_t off0 = 0;
      int64_t off1 = 0;
      bool off0Const = getConstIndexValue(op.getOffsets()[0], off0);
      bool off1Const = getConstIndexValue(op.getOffsets()[1], off1);
      if (off0Const &&
          !checkMultipleOf(op, off0, layoutInfo.innerRows, "row offset")) {
        return failure();
      }
      if (off1Const &&
          !checkMultipleOf(op, off1, layoutInfo.innerCols, "col offset")) {
        return failure();
      }

    }

    SmallVector<int64_t> srcStrides;
    int64_t srcOffset = ShapedType::kDynamic;
    if (failed(mlir::pto::getPTOMemRefStridesAndOffset(srcMrTy, srcStrides,
                                                       srcOffset)))
      srcStrides = computeCompactStrides(srcMrTy.getShape());

    // Keep parent physical shape + strides for bound tile semantics.
    auto resultLayout =
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, srcStrides);
    auto parentShape = srcMrTy.getShape();
    auto resultMemRefType =
        MemRefType::get(parentShape, srcMrTy.getElementType(), resultLayout,
                        srcMrTy.getMemorySpace());

    // Intermediate memref.subview keeps logical subview size.
    auto subViewMemRefType =
        MemRefType::get(staticSizes, srcMrTy.getElementType(), resultLayout,
                        srcMrTy.getMemorySpace());

    SmallVector<OpFoldResult> mixedStrides(staticSizes.size(),
                                           rewriter.getIndexAttr(1));
    auto sv = rewriter.create<memref::SubViewOp>(loc, subViewMemRefType, src,
                                                 mixedOffsets, mixedSizes,
                                                 mixedStrides);

    // When a valid operand is omitted, fall back to the extent the result type
    // declares (which the verifier pins to either `sizes` or an empty 0 marker)
    // rather than the physical subview size, so a no-op-replay v_row/v_col=0
    // survives lowering.
    ArrayRef<int64_t> resultValid =
        resultTileTy ? resultTileTy.getValidShape() : ArrayRef<int64_t>{};
    auto inferredValidDim = [&](unsigned d) -> int64_t {
      return d < resultValid.size() ? resultValid[d] : ShapedType::kDynamic;
    };

    Value vRow;
    Value vCol;
    if (!staticSizes.empty())
      vRow = clampSubViewValidDim(rewriter, loc, op.getValidRow(),
                                  staticSizes[0], inferredValidDim(0), op);
    if (staticSizes.size() > 1)
      vCol = clampSubViewValidDim(rewriter, loc, op.getValidCol(),
                                  staticSizes[1], inferredValidDim(1), op);

    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, resultMemRefType, sv.getResult(), vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(bindOp,
                               resultTileTy && resultTileTy.hasDynamicValid(),
                               ctx);
    bindOp->setAttr("pto.view_semantics", rewriter.getStringAttr("subview"));
    rewriter.replaceOp(op, bindOp.getResult());
  }
  return success();
}

static Value buildTileBufViewLikeValue(Operation *anchorOp, Value src,
                                       mlir::pto::TileBufType tbTy,
                                       StringRef viewSemantics,
                                       MLIRContext *ctx) {
  Location loc = anchorOp->getLoc();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(anchorOp);

  src = resolveTileBufViewLikeSource(src);
  auto srcMrTy = dyn_cast_or_null<MemRefType>(src.getType());
  if (!srcMrTy) {
    anchorOp->emitError("tile_buf view op src must be lowered to memref first");
    return Value();
  }

  auto targetType = dyn_cast<MemRefType>(convertPTOTypeToMemRef(tbTy));
  if (!targetType) {
    anchorOp->emitError("failed to convert tile_buf type to memref type");
    return Value();
  }
  for (int64_t dim : targetType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      anchorOp->emitError("dynamic shapes are not supported for tile_buf view ops");
      return Value();
    }
  }

  Value parentVRow;
  Value parentVCol;
  lookupValidDims(src, parentVRow, parentVCol);
  Value vRow = parentVRow;
  Value vCol = parentVCol;
  materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

  auto configAttr = tbTy.getConfigAttr();
  if (!configAttr)
    configAttr = pto::TileBufConfigAttr::getDefault(ctx);

  auto bindOp = rewriter.create<pto::BindTileOp>(
      loc, targetType, src, vRow ? vRow : Value(), vCol ? vCol : Value(),
      configAttr);
  markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
  if (!viewSemantics.empty())
    bindOp->setAttr("pto.view_semantics", rewriter.getStringAttr(viewSemantics));
  return bindOp.getResult();
}

static LogicalResult lowerTileBufViewLikeOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::TReshapeOp> reshapes;
  func.walk([&](mlir::pto::TReshapeOp op) { reshapes.push_back(op); });
  for (auto op : reshapes) {
    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy) {
      op.emitError("treshape result must be tile_buf type");
      return failure();
    }
    Value lowered = buildTileBufViewLikeValue(op, op->getOperand(0), tbTy,
                                              "treshape", ctx);
    if (!lowered)
      return failure();
    IRRewriter rewriter(ctx);
    rewriter.replaceOp(op, lowered);
  }

  DefaultInlineVector<mlir::pto::BitcastOp> bitcasts;
  func.walk([&](mlir::pto::BitcastOp op) { bitcasts.push_back(op); });
  for (auto op : bitcasts) {
    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy) {
      op.emitError("bitcast result must be tile_buf type");
      return failure();
    }
    Value lowered = buildTileBufViewLikeValue(op, op->getOperand(0), tbTy,
                                              "bitcast", ctx);
    if (!lowered)
      return failure();
    IRRewriter rewriter(ctx);
    rewriter.replaceOp(op, lowered);
  }
  return success();
}

// =============================================================================
// The Pass Implementation
// =============================================================================

struct PTOViewToMemrefPass
    : public mlir::pto::impl::PTOViewToMemrefBase<PTOViewToMemrefPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOViewToMemrefPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    for (auto func : mod.getOps<func::FuncOp>()) {
      // ------------------------------------------------------------------
      // Stage 0: ensure inttoptr values remain scalar-load/store only.
      // ------------------------------------------------------------------
      if (failed(validateIntToPtrUses(func))) {
        signalPassFailure();
        return;
      }

      auto fnTy = func.getFunctionType();

      // ------------------------------------------------------------------
      // Stage 0.10: Rewrite Function Signature
      // ------------------------------------------------------------------
      SmallVector<Type> newInputs;
      for (Type t : fnTy.getInputs()) newInputs.push_back(convertPTOTypeToMemRef(t));

      SmallVector<Type> newResults;
      for (Type t : fnTy.getResults()) newResults.push_back(convertPTOTypeToMemRef(t));

      func.setFunctionType(FunctionType::get(ctx, newInputs, newResults));
      if (func.isExternal()) continue;

      Block &entry = func.front();

      // Update entry block arguments
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        if (entry.getArgument(i).getType() != newInputs[i]) {
            entry.getArgument(i).setType(newInputs[i]);
        }
      }

      // ------------------------------------------------------------------
      // Stage 0.20: lower pto.inttoptr result types to GM memrefs.
      // ------------------------------------------------------------------
      if (failed(lowerIntToPtrOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 0.25: synthesize missing A2/A3 tquant tmp tiles before type lowering.
      // ------------------------------------------------------------------
      if (failed(synthesizeMissingTQuantTmpOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 0.30: materialize pto.ptrtoint(addptr ...) byte offsets.
      // ------------------------------------------------------------------
      if (failed(lowerPtrToIntOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // Stage 0.40 Insert pto.bind_tile for function args that were tile_buf.
      // ------------------------------------------------------------------
      // Later materialization and intrinsic folding use BindTileOp as the
      // anchor to recover tile metadata after the Stage-0 type rewrite.
      {
        IRRewriter rewriter(ctx);
        // Insert after existing block args, before any existing ops.
        rewriter.setInsertionPointToStart(&entry);
        for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
          Type origTy = fnTy.getInputs()[i];
          auto tbTy = dyn_cast<mlir::pto::TileBufType>(origTy);
          if (!tbTy)
            continue;

          auto configAttr = tbTy.getConfigAttr();
          if (!configAttr) configAttr = pto::TileBufConfigAttr::getDefault(ctx);

          Value vRow, vCol;
          auto vs = tbTy.getValidShape();
          if (vs.size() == 2) {
            if (vs[0] != ShapedType::kDynamic)
              vRow = rewriter.create<arith::ConstantIndexOp>(func.getLoc(), vs[0]);
            if (vs[1] != ShapedType::kDynamic)
              vCol = rewriter.create<arith::ConstantIndexOp>(func.getLoc(), vs[1]);
          }

          auto bindOp = rewriter.create<pto::BindTileOp>(
              func.getLoc(), newInputs[i], entry.getArgument(i),
              vRow ? vRow : Value(), vCol ? vCol : Value(), configAttr);
          markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

          entry.getArgument(i).replaceAllUsesExcept(bindOp.getResult(), bindOp);
        }
      }

      // ------------------------------------------------------------------
      // Stage 0.5: lower pto.alloc_tile -> memref.alloc + pto.bind_tile
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::AllocTileOp> allocTiles;
      func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

      for (auto op : allocTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
        if (!tbTy) continue;

        // 1. 获取 Shape 和 ElementType
        SmallInlineVector<int64_t> shape(tbTy.getShape().begin(), tbTy.getShape().end());
        Type elemTy = tbTy.getElementType();

        // 2. 计算 Strides (layout-aware when possible)
        SmallVector<int64_t> strides;
        TileLayoutInfo info;
        if (computeTileLayoutInfo(tbTy.getConfigAttr(), elemTy, shape, info)) {
          strides = {info.rowStride, info.colStride};
        } else {
          strides.resize(shape.size());
          int64_t s = 1;
          for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            if (shape[i] != ShapedType::kDynamic) s *= shape[i];
          }
        }

        // 3. 构造 [BindTile 输出] 的动态类型 (Offset: ?)
        // 这必须与 convertPTOTypeToMemRef 返回的类型一致，以便与 Subview 兼容
        auto targetLayout =
            StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides); // offset = ?
        auto targetType =
            MemRefType::get(shape, elemTy, targetLayout, tbTy.getMemorySpace());

        // 4. Preserve tile valid dims (v_row / v_col).
        //
        // `pto.alloc_tile` encodes the valid shape in the result TileBufType
        // (e.g. acc tile may be rows=16 but v_row=1). The alloc op itself does
        // not necessarily carry explicit operands for static valid dims, so we
        // must materialize them from the type to keep them through
        // tile_buf -> memref lowering.
        //
        // For dynamically valid tiles (validShape == [-1, -1]), preserve the
        // runtime operands if present.
        Value vRow = op.getValidRow();
        Value vCol = op.getValidCol();
        // TileBuf valid dims use a negative sentinel (e.g. '?' / -1), which is
        // distinct from MLIR's ShapedType::kDynamic (INT64_MIN). Treat any
        // negative value as dynamic here.
        materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

        // 5. 获取 Config (保持不变)
        auto configAttr = tbTy.getConfigAttr();
        if (!configAttr) configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        // 6. If alloc_tile provides an explicit address, keep the original
        // pointer_cast lowering intact and additionally rebind through
        // pto.bind_tile. PointerCastOp continues to carry the tile metadata
        // used by existing lowering paths, while BindTileOp provides the
        // unified anchor EmitC uses to recover tile_buf information.
        if (Value addr = op.getAddr()) {
          auto pc = rewriter.create<pto::PointerCastOp>(
              loc, targetType, ValueRange{addr}, vRow ? vRow : Value(),
              vCol ? vCol : Value(), configAttr);
          markForceDynamicValidShape(pc, tbTy.hasDynamicValid(), ctx);
          auto bindOp = rewriter.create<pto::BindTileOp>(
              loc, targetType, pc.getResult(), vRow ? vRow : Value(),
              vCol ? vCol : Value(), configAttr);
          markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
          rewriter.replaceOp(op, bindOp.getResult());
          continue;
        }

        // 7. Otherwise allocate a concrete memref buffer and bind tile.
        // memref.alloc 要求明确的 layout，不能是动态 offset。
        auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides); // offset = 0
        auto allocType = MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());
        Value alloc = rewriter.create<memref::AllocOp>(loc, allocType);

        // BindTileOp 的 Builder 会自动处理空的 Value，将其视为静态维度
        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, targetType, alloc, vRow ? vRow : Value(), vCol ? vCol : Value(),
            configAttr);
        markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 0.6: lower pto.alloc_multi_tile / pto.multi_tile_get
      //
      // alloc_multi_tile produces a `multi_tile_buf<S, count=N>`. We lower
      // it into:
      //   %a = memref.alloc() {pto.multi_buffer = N : i32} : memref<S>
      //   %m = pto.bind_tile %a, ... : memref<S> -> memref<dyn-offset S>
      // and replace all uses of the multi_tile_buf SSA with %m. The N-way
      // physical fan-out lives on the `pto.multi_buffer` attr and is
      // materialized later by PTOPlanMemory.
      //
      // multi_tile_get consumes that memref and wraps it in
      //   %slot = pto.slot_marker %m[%k] : memref -> memref
      //   %t    = pto.bind_tile %slot, ... : memref -> memref<dyn-offset>
      // The slot_marker carries the user-supplied slot SSA forward so that
      // PlanMemory / sync / EnableBufferSelect can identify which physical
      // slot this use refers to.
      //
      // Ordering: alloc_multi_tile must be lowered before multi_tile_get so
      // that the get's source SSA has already been rewired to a memref.
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::AllocMultiTileOp, 8> allocMultiTiles;
      func.walk(
          [&](mlir::pto::AllocMultiTileOp op) { allocMultiTiles.push_back(op); });

      for (auto op : allocMultiTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto mtbTy = op.getResult().getType();
        if (!mtbTy)
          continue;
        auto tbTy = mtbTy.getSlotType();
        if (!tbTy)
          continue;

        SmallVector<int64_t, 4> shape(tbTy.getShape().begin(),
                                      tbTy.getShape().end());
        Type elemTy = tbTy.getElementType();

        // Stride / layout reuse the same logic as alloc_tile.
        SmallVector<int64_t> strides;
        TileLayoutInfo info;
        if (computeTileLayoutInfo(tbTy.getConfigAttr(), elemTy, shape, info)) {
          strides = {info.rowStride, info.colStride};
        } else {
          strides.resize(shape.size());
          int64_t s = 1;
          for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            if (shape[i] != ShapedType::kDynamic)
              s *= shape[i];
          }
        }

        auto targetLayout =
            StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
        auto targetType =
            MemRefType::get(shape, elemTy, targetLayout, tbTy.getMemorySpace());

        Value vRow = op.getValidRow();
        Value vCol = op.getValidCol();
        ArrayRef<int64_t> validShape = tbTy.getValidShape();
        if (!tbTy.hasDynamicValid()) {
          if (validShape.size() >= 1 && validShape[0] >= 0) {
            vRow = rewriter
                       .create<arith::ConstantOp>(
                           loc, rewriter.getIndexType(),
                           rewriter.getIndexAttr(validShape[0]))
                       .getResult();
          }
          if (validShape.size() >= 2 && validShape[1] >= 0) {
            vCol = rewriter
                       .create<arith::ConstantOp>(
                           loc, rewriter.getIndexType(),
                           rewriter.getIndexAttr(validShape[1]))
                       .getResult();
          }
        }

        auto configAttr = tbTy.getConfigAttr();
        if (!configAttr)
          configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        // Level3 (caller-owned memory): an explicit base address is given and
        // PlanMemory is skipped. Lay the N slots out contiguously as
        // [addr, addr + slotBytes, ..., addr + (N-1)*slotBytes] and emit the
        // multi-address `pto.pointer_cast` alloc anchor directly -- the same
        // shape PlanMemory produces in the default pipeline (addrs are kept in
        // slot order so PTOResolveBufferSelect can pick addrs[slot]). Sync and
        // buffer-select then run unchanged.
        if (Value addr = op.getAddr()) {
          uint32_t n = mtbTy.getCount();
          int64_t elemBytes = getElemBytes(elemTy);
          int64_t numElems = 1;
          bool staticShape = true;
          for (int64_t d : shape) {
            if (d == ShapedType::kDynamic) {
              staticShape = false;
              break;
            }
            numElems *= d;
          }
          if (!staticShape || elemBytes <= 0) {
            op.emitError("pto.alloc_multi_tile with explicit addr requires a "
                         "static slot shape and known element byte size");
            signalPassFailure();
            return;
          }
          int64_t slotBytes = numElems * elemBytes;

          SmallVector<Value, 8> slotAddrs;
          slotAddrs.reserve(n);
          slotAddrs.push_back(addr);
          for (uint32_t slot = 1; slot < n; ++slot) {
            Value off = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(static_cast<int64_t>(slot) *
                                           slotBytes));
            slotAddrs.push_back(
                rewriter.create<arith::AddIOp>(loc, addr, off).getResult());
          }

          auto pc = rewriter.create<pto::PointerCastOp>(
              loc, targetType, slotAddrs, vRow ? vRow : Value(),
              vCol ? vCol : Value(), configAttr);
          markForceDynamicValidShape(pc, tbTy.hasDynamicValid(), ctx);

          auto bindOp = rewriter.create<pto::BindTileOp>(
              loc, targetType, pc.getResult(), vRow ? vRow : Value(),
              vCol ? vCol : Value(), configAttr);
          markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

          rewriter.replaceOp(op, bindOp.getResult());
          continue;
        }

        // memref.alloc with N-slot annotation. The actual N-way address
        // expansion happens in PlanMemory.
        auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides);
        auto allocType =
            MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());
        auto allocOp = rewriter.create<memref::AllocOp>(loc, allocType);
        auto i32Ty = IntegerType::get(ctx, 32);
        allocOp->setAttr(
            mlir::pto::kPtoMultiBufferAttrName,
            IntegerAttr::get(i32Ty, static_cast<int64_t>(mtbTy.getCount())));

        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, targetType, allocOp.getResult(), vRow ? vRow : Value(),
            vCol ? vCol : Value(), configAttr);
        markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      SmallVector<mlir::pto::MultiTileGetOp, 8> multiTileGets;
      func.walk(
          [&](mlir::pto::MultiTileGetOp op) { multiTileGets.push_back(op); });

      for (auto op : multiTileGets) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        // After alloc_multi_tile has been replaced, the source SSA value
        // is a memref (no longer multi_tile_buf). We accept the raw Value.
        Value srcMem = op->getOperand(0);
        if (!llvm::isa<MemRefType>(srcMem.getType())) {
          op.emitError(
              "multi_tile_get expects its source to have been lowered to "
              "memref by alloc_multi_tile rewriting; got ")
              << srcMem.getType();
          signalPassFailure();
          return;
        }
        auto srcMemTy = llvm::cast<MemRefType>(srcMem.getType());

        // Recover per-slot tile_buf info from the get op's result type
        // (which still describes the per-slot tile shape, valid dims,
        // config, etc.).
        auto tbTy = llvm::dyn_cast<pto::TileBufType>(op.getResult().getType());
        if (!tbTy) {
          op.emitError("multi_tile_get result must be `!pto.tile_buf<...>`");
          signalPassFailure();
          return;
        }

        // The slot view aliases the source memref byte-for-byte.
        auto slotMarker = rewriter.create<pto::SlotMarkerOp>(
            loc, srcMemTy, srcMem, op.getSlot());

        // Recover valid_row / valid_col from the per-slot tile type so
        // downstream BindTile carries the same metadata as the alloc.
        Value vRow;
        Value vCol;
        ArrayRef<int64_t> validShape = tbTy.getValidShape();
        if (!tbTy.hasDynamicValid()) {
          if (validShape.size() >= 1 && validShape[0] >= 0) {
            vRow = rewriter
                       .create<arith::ConstantOp>(
                           loc, rewriter.getIndexType(),
                           rewriter.getIndexAttr(validShape[0]))
                       .getResult();
          }
          if (validShape.size() >= 2 && validShape[1] >= 0) {
            vCol = rewriter
                       .create<arith::ConstantOp>(
                           loc, rewriter.getIndexType(),
                           rewriter.getIndexAttr(validShape[1]))
                       .getResult();
          }
        }

        auto configAttr = tbTy.getConfigAttr();
        if (!configAttr)
          configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        // BindTile result type uses the same layout signature as in the
        // alloc lowering, so subsequent subview / DMA ops see a consistent
        // memref view.
        SmallVector<int64_t> strides = buildTileMemRefStrides(tbTy);
        auto targetLayout =
            StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
        auto targetType =
            MemRefType::get(tbTy.getShape(), tbTy.getElementType(), targetLayout,
                            tbTy.getMemorySpace());

        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, targetType, slotMarker.getResult(), vRow ? vRow : Value(),
            vCol ? vCol : Value(), configAttr);
        markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 0.75: lower pto.declare_tile -> pto.declare_tile_memref +
      //             pto.bind_tile
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::DeclareTileOp> declaredTiles;
      func.walk([&](mlir::pto::DeclareTileOp op) { declaredTiles.push_back(op); });

      for (auto op : declaredTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getTile().getType());
        if (!tbTy) {
          op.emitError("declare_tile result must be tile_buf type");
          signalPassFailure();
          return;
        }

        auto targetType = dyn_cast<MemRefType>(convertPTOTypeToMemRef(tbTy));
        if (!targetType) {
          op.emitError("failed to convert declare_tile result to memref type");
          signalPassFailure();
          return;
        }

        auto configAttr = tbTy.getConfigAttr();
        if (!configAttr)
          configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        Value vRow;
        Value vCol;
        materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

        auto declaredMemRef =
            rewriter.create<pto::DeclareTileMemRefOp>(loc, targetType);
        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, targetType, declaredMemRef.getResult(),
            vRow ? vRow : Value(), vCol ? vCol : Value(), configAttr);
        markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 0.8: normalize pto.tassign result type to match tile operand
      // after tile_buf -> memref lowering (required for verifier consistency).
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::TAssignOp> tassignOps;
      func.walk([&](mlir::pto::TAssignOp op) { tassignOps.push_back(op); });
      for (auto op : tassignOps) {
        Type targetTy = op.getTile().getType();
        if (op.getResult().getType() == targetTy)
          continue;
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        auto normalized =
            rewriter.create<pto::TAssignOp>(op.getLoc(), targetTy, op.getTile(),
                                            op.getAddr());
        rewriter.replaceOp(op, normalized.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 0.9: Lower ptr-like pto.tile_buf_addr -> memref<?xT>.
      // The original authoring surface is pointer-like, so shape information
      // is intentionally discarded here and only the linear address view is
      // preserved for later memref-based rewriting.
      // ------------------------------------------------------------------
      if (failed(lowerPtrLikeTileBufAddrOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1: Lower pto.make_tensor_view -> memref.reinterpret_cast
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::MakeTensorViewOp> makeViews;
      func.walk([&](mlir::pto::MakeTensorViewOp op) { makeViews.push_back(op); });

      for (auto op : makeViews) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value baseBuf = op.getOperand(0);
        OpFoldResult off0 = rewriter.getIndexAttr(0);

        // Fold pto.addptr chains into the view base to avoid nested reinterpret_cast.
        bool foldedAddPtr = false;
        {
          Value cur = baseBuf;
          Value totalOffset;
          while (auto add = cur.getDefiningOp<mlir::pto::AddPtrOp>()) {
            foldedAddPtr = true;
            Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
            if (totalOffset)
              totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
            else
              totalOffset = off;
            cur = add.getOperand(0);
          }
          if (cur != baseBuf) {
            baseBuf = cur;
            off0 = totalOffset ? OpFoldResult(totalOffset) : off0;
          }
        }

        auto baseMr = dyn_cast<BaseMemRefType>(baseBuf.getType());
        if (!baseMr) {
             op.emitError("make_tensor_view base must be memref"); signalPassFailure(); return;
        }

        // [修复] 获取动态 Rank (根据 shape 输入的数量)
        size_t rank = op.getShape().size(); 

        // Construct target type with dynamic offset/strides
        Type elemTy = baseMr.getElementType();
        int64_t dyn = ShapedType::kDynamic;
        
        // [修复] 构建 N 维 Strided Layout
        // strides 数组长度必须等于 rank
        SmallVector<int64_t> dynStrides(rank, dyn);
        auto layout = StridedLayoutAttr::get(ctx, /*offset=*/dyn, /*strides=*/dynStrides);
        
        // [修复] 构建 N 维 Shape
        SmallVector<int64_t> dynShape(rank, dyn);
        auto mrTy = MemRefType::get(dynShape, elemTy, layout, baseMr.getMemorySpace());

        SmallInlineVector<OpFoldResult> sizes;
        for (Value v : op.getShape()) sizes.push_back(ensureIndex(rewriter, loc, v, op));

        SmallInlineVector<OpFoldResult> strides;
        for (Value v : op.getStrides()) strides.push_back(ensureIndex(rewriter, loc, v, op));

        auto rc = rewriter.create<memref::ReinterpretCastOp>(
            loc, mrTy, baseBuf, off0, sizes, strides);
        if (foldedAddPtr) {
          rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
        }
        if (auto layoutAttr = op.getLayoutAttr()) {
          rc->setAttr("layout", layoutAttr);
        }

        rewriter.replaceOp(op, rc.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 1.25: Lower pto.get_tensor_view_dim -> memref.dim
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::GetTensorViewDimOp> tvDims;
      func.walk([&](mlir::pto::GetTensorViewDimOp op) { tvDims.push_back(op); });

      for (auto op : tvDims) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value view = op.getTensorView();
        auto mrTy = dyn_cast<BaseMemRefType>(view.getType());
        if (!mrTy)
          continue; // leave it to later passes if it hasn't been lowered yet

        Value dimIdx = op.getDimIndex();
        Value dim = rewriter.create<memref::DimOp>(loc, view, dimIdx);
        rewriter.replaceOp(op, dim);
      }

      // ------------------------------------------------------------------
      // Stage 1.3: Lower pto.partition_view -> memref.subview
      // ------------------------------------------------------------------
      if (failed(lowerPartitionViewOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1.35: Lower pto.subview -> memref.subview + pto.bind_tile
      // ------------------------------------------------------------------
      if (failed(lowerSubViewOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1.4: Lower tile_buf view-like ops (treshape/bitcast)
      // ------------------------------------------------------------------
      if (failed(lowerTileBufViewLikeOps(func, ctx))) {
        signalPassFailure();
        return;
      }
      if (failed(reconcileFusionRegionResultTypes(func))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1.5: Lower pto.get_tensor_view_stride -> strided memref metadata
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::GetTensorViewStrideOp, 8> tvStrides;
      func.walk([&](mlir::pto::GetTensorViewStrideOp op) { tvStrides.push_back(op); });

      for (auto op : tvStrides) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value view = op.getTensorView();
        auto mrTy = dyn_cast<MemRefType>(view.getType());
        if (!mrTy)
          continue; // leave it to later passes if it hasn't been lowered yet

        int64_t dimIndex = 0;
        if (!getConstIndexValue(op.getDimIndex(), dimIndex)) {
          op.emitError("get_tensor_view_stride currently expects a constant dim index");
          signalPassFailure();
          return;
        }
        if (dimIndex < 0 || dimIndex >= mrTy.getRank()) {
          op.emitError("get_tensor_view_stride dim index is out of bounds");
          signalPassFailure();
          return;
        }

        SmallVector<int64_t> staticStrides;
        int64_t offset = ShapedType::kDynamic;
        if (succeeded(mlir::pto::getPTOMemRefStridesAndOffset(
                mrTy, staticStrides, offset)) &&
            dimIndex < (int64_t)staticStrides.size() &&
            staticStrides[dimIndex] != ShapedType::kDynamic) {
          rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
              op, staticStrides[dimIndex]);
          continue;
        }

        auto metadata =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc, view);
        rewriter.replaceOp(op, metadata.getStrides()[dimIndex]);
      }

      // ------------------------------------------------------------------
      // Stage 1.6: Fold pto.addptr chains into load/store_scalar.
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::LoadScalarOp> loadScalars;
      func.walk([&](mlir::pto::LoadScalarOp op) { loadScalars.push_back(op); });

      for (auto op : loadScalars) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value base = op.getPtr();
        Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);

        bool foldedAddPtr = false;
        while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
          foldedAddPtr = true;
          Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
          if (totalOffset)
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
          else
            totalOffset = off;
          base = add.getOperand(0);
        }

        if (foldedAddPtr) {
          auto newOp = rewriter.create<pto::LoadScalarOp>(
              loc, op.getValue().getType(), base, totalOffset);
          rewriter.replaceOp(op, newOp.getValue());
        }
      }

      DefaultInlineVector<mlir::pto::StoreScalarOp> storeScalars;
      func.walk([&](mlir::pto::StoreScalarOp op) { storeScalars.push_back(op); });

      for (auto op : storeScalars) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value base = op.getPtr();
        Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);

        bool foldedAddPtr = false;
        while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
          foldedAddPtr = true;
          Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
          if (totalOffset)
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
          else
            totalOffset = off;
          base = add.getOperand(0);
        }

        if (foldedAddPtr) {
          rewriter.create<pto::StoreScalarOp>(
              loc, base, totalOffset, op.getValue());
          rewriter.eraseOp(op);
        }
      }

      // ------------------------------------------------------------------
      // Stage 1.75: Fold addptr used by initialize_l2g2l_pipe(gm_addr).
      // This keeps IR well-typed after function arguments are rewritten from
      // !pto.ptr<T> to memref<?xT>.
      // ------------------------------------------------------------------
      bool foldedPipeInitAddPtr = true;
      while (foldedPipeInitAddPtr) {
        foldedPipeInitAddPtr = false;
        DefaultInlineVector<mlir::pto::AddPtrOp> addPtrsForPipeInit;
        func.walk([&](mlir::pto::AddPtrOp op) {
          bool eligible = !op->use_empty();
          for (Operation *user : op->getUsers()) {
            auto init = dyn_cast<mlir::pto::InitializeL2G2LPipeOp>(user);
            if (!init || init.getGmAddr() != op->getResult(0)) {
              eligible = false;
              break;
            }
          }
          if (eligible)
            addPtrsForPipeInit.push_back(op);
        });

        for (auto op : addPtrsForPipeInit) {
          IRRewriter rewriter(ctx);
          rewriter.setInsertionPoint(op);
          Location loc = op.getLoc();

          Value base = op->getOperand(0);
          Value totalOffset = ensureIndex(rewriter, loc, op->getOperand(1), op);
          while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
            Value off = ensureIndex(rewriter, loc, add->getOperand(1), add);
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
            base = add->getOperand(0);
          }

          auto baseMrTy = dyn_cast<MemRefType>(base.getType());
          if (!baseMrTy || baseMrTy.getRank() != 1)
            continue;

          int64_t dyn = ShapedType::kDynamic;
          auto layout = StridedLayoutAttr::get(ctx, dyn, {dyn});
          auto targetTy = MemRefType::get({dyn}, baseMrTy.getElementType(), layout,
                                          baseMrTy.getMemorySpace());
          SmallVector<OpFoldResult, 1> sizes{rewriter.getIndexAttr(1)};
          SmallVector<OpFoldResult, 1> strides{rewriter.getIndexAttr(1)};
          auto rc = rewriter.create<memref::ReinterpretCastOp>(
              loc, targetTy, base, OpFoldResult(totalOffset), sizes, strides);
          rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
          rewriter.replaceOp(op, rc.getResult());
          foldedPipeInitAddPtr = true;
        }
      }

      // Clean up: addptr should be folded into make_tensor_view.
      DefaultInlineVector<Operation *> addPtrs;
      func.walk([&](mlir::pto::AddPtrOp op) { addPtrs.push_back(op.getOperation()); });
      bool changed = true;
      while (changed) {
        changed = false;
        for (auto &op : addPtrs) {
          if (!op)
            continue;
          if (op->use_empty()) {
            op->erase();
            op = nullptr;
            changed = true;
          }
        }
      }
      for (auto *op : addPtrs) {
        if (!op)
          continue;
        op->emitError("addptr must feed make_tensor_view,  initialize_l2g2l_pipe(gm_addr) or load/store_scalar for lowering");
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 3: Rewrite Compute Ops
      // [关键] 全面使用 op->getOperand(i) 避免 Typed Accessor Crash
      // ------------------------------------------------------------------
      
      // --- TLoadOp [Src, Dst] ---
      DefaultInlineVector<mlir::pto::TLoadOp> loads;
      func.walk([&](mlir::pto::TLoadOp op) { loads.push_back(op); });
      for (auto op : loads) {
          IRRewriter rewriter(ctx);
          rewriter.setInsertionPoint(op);
          
          Value src = op->getOperand(0); 
          Value dst = op->getOperand(1);

          auto newOp =
              rewriter.create<pto::TLoadOp>(op.getLoc(), TypeRange{}, src, dst);
          newOp->setAttrs(op->getAttrs());
          rewriter.replaceOp(op, newOp->getResults());
      }

      // --- TStoreOp [Src, Dst] ---
      DefaultInlineVector<mlir::pto::TStoreOp> storeops;
      func.walk([&](mlir::pto::TStoreOp op) { storeops.push_back(op); });
      for (auto op : storeops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op->getOperand(0); 
        Value dst = op->getOperand(1);
        Value preQuant = op.getPreQuantScalar();

        pto::TStoreOp newOp;
        if (preQuant) {
          newOp = rewriter.create<pto::TStoreOp>(op.getLoc(), TypeRange{},
                                                 src, dst, preQuant);
        } else {
          newOp = rewriter.create<pto::TStoreOp>(op.getLoc(), TypeRange{},
                                                 src, dst, Value{});
        }
        newOp->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newOp->getResults());
      }

       // --- TTransOp [Src, Tmp, Dst] ---
      DefaultInlineVector<mlir::pto::TTransOp> trans;
      func.walk([&](mlir::pto::TTransOp op) { trans.push_back(op); });
      for (auto op : trans) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        replaceOpWithClonedAttrs<pto::TTransOp>(
            rewriter, op, TypeRange{}, op->getOperand(0), op->getOperand(1),
            op->getOperand(kThirdOperandIndex));
      }

      // --- TExpOp [Src, Dst] ---
      DefaultInlineVector<mlir::pto::TExpOp> exp;
      func.walk([&](mlir::pto::TExpOp op) { exp.push_back(op); });
      for (auto op : exp) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        replaceOpWithClonedAttrs<pto::TExpOp>(rewriter, op, TypeRange{},
                                              op->getOperand(0),
                                              op->getOperand(1), op.getPrecisionTypeAttr());
      }

      // --- TMulOp [Src, Scalar, Dst] ---
      DefaultInlineVector<mlir::pto::TMulOp> mul;
      func.walk([&](mlir::pto::TMulOp op) { mul.push_back(op); });
      for (auto op : mul) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        replaceOpWithClonedAttrs<pto::TMulOp>(
            rewriter, op, op->getOperand(0), op.getOperand(1),
            op->getOperand(kThirdOperandIndex));
      }

      // --- TMulSOp [Src, Scalar, Dst] ---
      DefaultInlineVector<mlir::pto::TMulSOp> muls;
      func.walk([&](mlir::pto::TMulSOp op) { muls.push_back(op); });
      for (auto op : muls) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        replaceOpWithClonedAttrs<pto::TMulSOp>(
            rewriter, op, op->getOperand(0), op.getScalar(),
            op->getOperand(kThirdOperandIndex));
      }

      // --- TAddOp [Src0, Src1, Dst] ---
      DefaultInlineVector<mlir::pto::TAddOp> addops;
      func.walk([&](mlir::pto::TAddOp op) { addops.push_back(op); });
      for (auto op : addops) {
          IRRewriter rewriter(ctx);
          rewriter.setInsertionPoint(op);
          
          replaceOpWithClonedAttrs<pto::TAddOp>(
              rewriter, op, TypeRange{}, op->getOperand(0), op->getOperand(1),
              op->getOperand(kThirdOperandIndex));
      }

      // --- TMatmulOp [Lhs, Rhs, Dst] (no optional bias in ODS) ---
      DefaultInlineVector<mlir::pto::TMatmulOp > matmuls;
      func.walk([&](mlir::pto::TMatmulOp  op) { matmuls.push_back(op); });
      for (auto op : matmuls) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value dst = op->getOperand(kThirdOperandIndex);

        rewriter.replaceOpWithNewOp<pto::TMatmulOp>(
            op, TypeRange{}, lhs, rhs, dst, op.getAccPhaseAttr());
      }

      // --- TMatmulAccOp [Acc, Lhs, Rhs, Dst] ---
      DefaultInlineVector<mlir::pto::TMatmulAccOp > matmulAccs;
      func.walk([&](mlir::pto::TMatmulAccOp  op) { matmulAccs.push_back(op); });
      for (auto op : matmulAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulAccOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TMatmulBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
      DefaultInlineVector<mlir::pto::TMatmulBiasOp > matmulBiass;
      func.walk([&](mlir::pto::TMatmulBiasOp  op) { matmulBiass.push_back(op); });
      for (auto op : matmulBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulBiasOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TMatmulMxOp---
      DefaultInlineVector<mlir::pto::TMatmulMxOp > matmulMxs;
      func.walk([&](mlir::pto::TMatmulMxOp  op) { matmulMxs.push_back(op); });
      for (auto op : matmulMxs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulMxOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex),
          op->getOperand(kFifthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TMatmulMxAccOp  ---
      DefaultInlineVector<mlir::pto::TMatmulMxAccOp > matmulMxAccs;
      func.walk([&](mlir::pto::TMatmulMxAccOp  op) { matmulMxAccs.push_back(op); });
      for (auto op : matmulMxAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulMxAccOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex),
          op->getOperand(kFifthOperandIndex),
          op->getOperand(kSixthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TMatmulMxBiasOp ---
      DefaultInlineVector<mlir::pto::TMatmulMxBiasOp > matmulMxBiass;
      func.walk([&](mlir::pto::TMatmulMxBiasOp  op) { matmulMxBiass.push_back(op); });
      for (auto op : matmulMxBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulMxBiasOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex),
          op->getOperand(kFifthOperandIndex),
          op->getOperand(kSixthOperandIndex));
      }

      // --- TGemvOp [Lhs, Rhs, Dst] ---
      DefaultInlineVector<mlir::pto::TGemvOp > gemvs;
      func.walk([&](mlir::pto::TGemvOp  op) { gemvs.push_back(op); });
      for (auto op : gemvs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value dst = op->getOperand(kThirdOperandIndex);

        rewriter.replaceOpWithNewOp<pto::TGemvOp>(
          op, TypeRange{}, lhs, rhs, dst, op.getAccPhaseAttr());
      }

      // --- TGemvAccOp [Acc, Lhs, Rhs, Dst] ---
      DefaultInlineVector<mlir::pto::TGemvAccOp > gemvAccs;
      func.walk([&](mlir::pto::TGemvAccOp  op) { gemvAccs.push_back(op); });
      for (auto op : gemvAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvAccOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TGemvBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
      DefaultInlineVector<mlir::pto::TGemvBiasOp > gemvBiass;
      func.walk([&](mlir::pto::TGemvBiasOp  op) { gemvBiass.push_back(op); });
      for (auto op : gemvBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvBiasOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TGemvMxOp [A, AScale, B, BScale, Dst] ---
      DefaultInlineVector<mlir::pto::TGemvMxOp > gemvMxs;
      func.walk([&](mlir::pto::TGemvMxOp  op) { gemvMxs.push_back(op); });
      for (auto op : gemvMxs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvMxOp>(
          op, TypeRange{},
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex),
          op->getOperand(kFifthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TGemvMxAccOp [CIn, A, AScale, B, BScale, Dst] ---
      DefaultInlineVector<mlir::pto::TGemvMxAccOp > gemvMxAccs;
      func.walk([&](mlir::pto::TGemvMxAccOp  op) { gemvMxAccs.push_back(op); });
      for (auto op : gemvMxAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvMxAccOp>(
          op, TypeRange{},
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex),
          op->getOperand(kFifthOperandIndex),
          op->getOperand(kSixthOperandIndex), op.getAccPhaseAttr());
      }

      // --- TGemvMxBiasOp [A, AScale, B, BScale, Bias, Dst] ---
      DefaultInlineVector<mlir::pto::TGemvMxBiasOp > gemvMxBiass;
      func.walk([&](mlir::pto::TGemvMxBiasOp  op) { gemvMxBiass.push_back(op); });
      for (auto op : gemvMxBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvMxBiasOp>(
          op, TypeRange{},
          op->getOperand(0), op->getOperand(1),
          op->getOperand(kThirdOperandIndex),
          op->getOperand(kFourthOperandIndex),
          op->getOperand(kFifthOperandIndex),
          op->getOperand(kSixthOperandIndex));
      }

      // --- TMovOp [Src, Dst] ---
      DefaultInlineVector<mlir::pto::TMovOp > movs;
      func.walk([&](mlir::pto::TMovOp  op) { movs.push_back(op); });
      for (auto op : movs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMovOp>(
            op, TypeRange{}, op.getSrc(), op.getDst(), op.getFp(),
            op.getPreQuantScalar(), op.getAccToVecModeAttr(),
            op.getReluPreModeAttr());
      }

      DefaultInlineVector<mlir::pto::TAbsOp> abseops;
      func.walk([&](mlir::pto::TAbsOp op) { abseops.push_back(op); });

      for (auto op : abseops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TAbsOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      DefaultInlineVector<mlir::pto::TAddCOp> addcops;
      func.walk([&](mlir::pto::TAddCOp op) { addcops.push_back(op); });

      for (auto op : addcops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value src2 = op.getSrc2();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto src2Ty = dyn_cast<MemRefType>(src2.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !src2Ty ||!dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TAddCOp>(
            op,
            TypeRange{},
            src0,
            src1,
            src2,
            dst);
      }

      DefaultInlineVector<mlir::pto::TAddSOp> addsops;
      func.walk([&](mlir::pto::TAddSOp op) { addsops.push_back(op); });

      for (auto op : addsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TAddSOp>(rewriter, op, TypeRange{}, src,
                                               scalar, dst);
      }

      DefaultInlineVector<mlir::pto::TAddSCOp> addscops;
      func.walk([&](mlir::pto::TAddSCOp op) { addscops.push_back(op); });

      for (auto op : addscops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value scalar = op.getScalar();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TAddSCOp>(
            op,
            TypeRange{},
            src0,
            scalar,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TAndOp> andops;
      func.walk([&](mlir::pto::TAndOp op) { andops.push_back(op); });

      for (auto op : andops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TAndOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TConcatOp> concats;
      func.walk([&](mlir::pto::TConcatOp op) { concats.push_back(op); });

      for (auto op : concats) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TConcatOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TConcatidxOp> concatIdxs;
      func.walk([&](mlir::pto::TConcatidxOp op) { concatIdxs.push_back(op); });

      IRRewriter rewriter(ctx);
      for (auto op : concatIdxs) {
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value src0Idx = op.getSrc0Idx();
        Value src1Idx = op.getSrc1Idx();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto src0IdxTy = dyn_cast<MemRefType>(src0Idx.getType());
        auto src1IdxTy = dyn_cast<MemRefType>(src1Idx.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !src0IdxTy || !src1IdxTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TConcatidxOp>(
            op,
            TypeRange{},
            src0,
            src1,
            src0Idx,
            src1Idx,
            dst);
      }

      DefaultInlineVector<mlir::pto::TAndSOp> andsops;
      func.walk([&](mlir::pto::TAndSOp op) { andsops.push_back(op); });

      for (auto op : andsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TAndSOp>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      DefaultInlineVector<mlir::pto::TCIOp> ciops;
      func.walk([&](mlir::pto::TCIOp op) { ciops.push_back(op); });

      for (auto op : ciops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value s = op->getOperand(0);
        Value tmp = op.getTmp();
        Value dst = op.getDst();
        bool descending = op.getDescending();

        auto sTy = dyn_cast<IntegerType>(s.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto tmpTy = tmp ? dyn_cast<MemRefType>(tmp.getType()) : MemRefType{};
        if (!sTy || !dstTy || (tmp && !tmpTy)) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TCIOp>(
            op,
            TypeRange{},
            s,
            tmp,
            dst,
            descending);
      }

      DefaultInlineVector<mlir::pto::TCmpOp> cmpops;
      func.walk([&](mlir::pto::TCmpOp op) { cmpops.push_back(op); });

      for (auto op : cmpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

         auto newOp = rewriter.create<pto::TCmpOp>(
            op.getLoc(),
            TypeRange{},
            src0,
            src1,
            dst);
         
          if (auto a = op.getCmpModeAttr())
            newOp->setAttr("cmpMode", a);

        rewriter.replaceOp(op, newOp->getResults()); // 0 results -> OK
      }

      DefaultInlineVector<mlir::pto::TCmpSOp> cmpsops;
      func.walk([&](mlir::pto::TCmpSOp op) { cmpsops.push_back(op); });

      for (auto op : cmpsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto scalarTy = scalar.getType();
        bool scalarOk =
            isa<IntegerType, FloatType>(scalarTy); // ScalarType in ODS: int/float
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }
        if (!scalarOk) {
          op.emitError("expects scalar to be an integer or float type");
          signalPassFailure();
          return;
        }

        auto cmpMode = op.getCmpModeAttr();
        auto newOp = rewriter.create<pto::TCmpSOp>(
            op.getLoc(),
            TypeRange{},
            src,
            scalar,
            cmpMode,
            dst);

        rewriter.replaceOp(op, newOp->getResults()); // 0 results -> OK
      }

      DefaultInlineVector<mlir::pto::TColExpandOp> colexpand;
      func.walk([&](mlir::pto::TColExpandOp op) { colexpand.push_back(op); });

      for (auto op : colexpand) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if ( !srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TColExpandOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      DefaultInlineVector<mlir::pto::TColMaxOp> colmaxops;
      func.walk([&](mlir::pto::TColMaxOp op) { colmaxops.push_back(op); });

      for (auto op : colmaxops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if ( !srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TColMaxOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      DefaultInlineVector<mlir::pto::TColMinOp> colminops;
      func.walk([&](mlir::pto::TColMinOp op) { colminops.push_back(op); });

      for (auto op : colminops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if ( !srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TColMinOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      DefaultInlineVector<mlir::pto::TColExpandMulOp> colexpandmulops;
      func.walk([&](mlir::pto::TColExpandMulOp op) {
        colexpandmulops.push_back(op);
      });

      for (auto op : colexpandmulops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TColExpandMulOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TColExpandMaxOp> colexpandmaxops;
      func.walk([&](mlir::pto::TColExpandMaxOp op) {
        colexpandmaxops.push_back(op);
      });

      for (auto op : colexpandmaxops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TColExpandMaxOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TColExpandMinOp> colexpandminops;
      func.walk([&](mlir::pto::TColExpandMinOp op) {
        colexpandminops.push_back(op);
      });

      for (auto op : colexpandminops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TColExpandMinOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TColSumOp> colsumops;
      func.walk([&](mlir::pto::TColSumOp op) { colsumops.push_back(op); });

      for (auto op : colsumops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value tmp = op.getTmp();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("src/dst are not memref yet");
          signalPassFailure();
          return;
        }

        // If tmp exists, it must have isBinary attribute
        if (tmp) {
          auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
          if (!tmpTy) {
            op.emitError("tmp is not memref yet");
            signalPassFailure();
            return;
          }

          // Get isBinary attribute (should exist if tmp exists)
          BoolAttr isBinaryAttr = op.getIsBinaryAttr();
          if (!isBinaryAttr) {
            isBinaryAttr = BoolAttr::get(ctx, false);
          }

          rewriter.replaceOpWithNewOp<pto::TColSumOp>(
              op,
              TypeRange{},
              src,
              tmp,
              dst,
              isBinaryAttr);
        } else {
          // Format 1: no tmp, no isBinary
          // Use generic builder to avoid adding default isBinary attribute
          SmallVector<Value> operands = {src, dst};
          SmallVector<NamedAttribute> attrs;
          // Copy all attributes except isBinary
          for (auto attr : op->getAttrs()) {
            if (attr.getName() != "isBinary") {
              attrs.push_back(attr);
            }
          }
          rewriter.replaceOpWithNewOp<pto::TColSumOp>(
              op,
              TypeRange{},
              operands,
              attrs);
        }
      }

      DefaultInlineVector<mlir::pto::TCvtOp> cvtops;
      func.walk([&](mlir::pto::TCvtOp op) { cvtops.push_back(op); });

      for (auto op : cvtops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        auto rmodeAttr = op.getRmodeAttr(); // PTO_RoundModeAttr
        auto satModeAttr = op.getSatModeAttr();

        auto newOp = rewriter.create<pto::TCvtOp>(
            op.getLoc(),
            TypeRange{},
            src,
            dst,
            rmodeAttr,
            satModeAttr);

        rewriter.replaceOp(op, newOp->getResults());
      }

      DefaultInlineVector<mlir::pto::TDivOp> divops;
      func.walk([&](mlir::pto::TDivOp op) { divops.push_back(op); });

      for (auto op : divops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TDivOp>(rewriter,
            op,
            TypeRange{},
            src0,
            src1,
            dst,
            op.getPrecisionTypeAttr());
      }

      DefaultInlineVector<mlir::pto::TDivSOp> divsops;
      func.walk([&](mlir::pto::TDivSOp op) { divsops.push_back(op); });

      for (auto op : divsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scale = op.getScalar();
        Value dst = op.getDst();

        // Check types - they might still be TileBufType or already converted to MemRefType
        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto srcTileTy = dyn_cast<mlir::pto::TileBufType>(src.getType());
        auto scaleTileTy = dyn_cast<mlir::pto::TileBufType>(scale.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto dstTileTy = dyn_cast<mlir::pto::TileBufType>(dst.getType());
        
        // Determine which operand is tile-like and which is scalar-like.
        // Keep the original operand order (set by parser textual form).
        // Check if src is memref/tensor/tile (not scalar)
        bool srcIsMemref = (srcTy != nullptr || srcTileTy != nullptr || 
                            isa<RankedTensorType>(src.getType()) ||
                            isa<mlir::pto::PartitionTensorViewType>(src.getType()));
        // Check if scale is memref/tensor/tile (not scalar)
        bool scaleIsMemref = (isa<MemRefType>(scale.getType()) || 
                              scaleTileTy != nullptr ||
                              isa<RankedTensorType>(scale.getType()) ||
                              isa<mlir::pto::PartitionTensorViewType>(scale.getType()));

        // Type validation - ensure we have the right types
        if (!srcIsMemref && !scaleIsMemref) {
          op.emitError("at least one operand (src or scale) must be tile_buf or memref");
          signalPassFailure();
          return;
        }
        if (srcIsMemref && scaleIsMemref) {
          op.emitError("exactly one operand (src or scale) must be tile_buf or memref, the other must be scalar");
          signalPassFailure();
          return;
        }

        if (!dstTy && !dstTileTy) {
          op.emitError("dst operand must be tile_buf or memref");
          signalPassFailure();
          return;
        }
        replaceOpWithClonedAttrs<pto::TDivSOp>(rewriter,
            op,
            TypeRange{},
            src,
            scale,
            dst,
            op.getPrecisionTypeAttr());
      }

      DefaultInlineVector<mlir::pto::TExpandsOp> expandsops;
      func.walk([&](mlir::pto::TExpandsOp op) { expandsops.push_back(op); });

      for (auto op : expandsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TExpandsOp>(rewriter, op, TypeRange{},
                                                  scalar, dst);
      }

      DefaultInlineVector<mlir::pto::TExtractOp> extractops;
      func.walk([&](mlir::pto::TExtractOp op) { extractops.push_back(op); });

      for (auto op : extractops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value indexRow = op.getIndexRow();
        Value indexCol = op.getIndexCol();
        Value preQuantScalar = op.getPreQuantScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto indexRowTy = dyn_cast<IndexType>(indexRow.getType());
        auto indexColTy = dyn_cast<IndexType>(indexCol.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !indexRowTy || !indexColTy || !dstTy) {
          op.emitError("ins/outs are not correct yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TExtractOp>(
            rewriter, op, TypeRange{}, src, indexRow, indexCol,
            preQuantScalar, dst, op.getAccToVecModeAttr(),
            op.getReluPreModeAttr());
      }

      DefaultInlineVector<mlir::pto::TExtractFPOp> extractfpops;
      func.walk([&](mlir::pto::TExtractFPOp op) { extractfpops.push_back(op); });
      for (auto op : extractfpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
        Value indexRow = op.getIndexRow();
        Value indexCol = op.getIndexCol();
        Value dst = op.getDst();

        if (!dyn_cast<MemRefType>(src.getType()) ||
            !dyn_cast<MemRefType>(fp.getType()) ||
            !dyn_cast<IndexType>(indexRow.getType()) ||
            !dyn_cast<IndexType>(indexCol.getType()) ||
            !dyn_cast<MemRefType>(dst.getType())) {
          op.emitError("ins/outs are not correct yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TExtractFPOp>(
            rewriter, op, TypeRange{}, src, fp, indexRow, indexCol, dst,
            op.getAccToVecModeAttr(), op.getReluPreModeAttr());
      }

      DefaultInlineVector<mlir::pto::TInsertOp> insertops;
      func.walk([&](mlir::pto::TInsertOp op) { insertops.push_back(op); });
      for (auto op : insertops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value indexRow = op.getIndexRow();
        Value indexCol = op.getIndexCol();
        Value fp = op.getFp();
        Value preQuantScalar = op.getPreQuantScalar();
        Value dst = op.getDst();

        if (!dyn_cast<MemRefType>(src.getType()) ||
            !dyn_cast<IndexType>(indexRow.getType()) ||
            !dyn_cast<IndexType>(indexCol.getType()) ||
            !dyn_cast<MemRefType>(dst.getType()) ||
            (fp && !dyn_cast<MemRefType>(fp.getType()))) {
          op.emitError("ins/outs are not correct yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TInsertOp>(
            rewriter, op, TypeRange{}, src, indexRow, indexCol, dst, fp,
            preQuantScalar, op.getAccToVecModeAttr(), op.getReluPreModeAttr(),
            op.getTinsertModeAttr());
      }

      DefaultInlineVector<mlir::pto::TInsertFPOp> insertfpops;
      func.walk([&](mlir::pto::TInsertFPOp op) { insertfpops.push_back(op); });
      for (auto op : insertfpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
        Value indexRow = op.getIndexRow();
        Value indexCol = op.getIndexCol();
        Value dst = op.getDst();

        if (!dyn_cast<MemRefType>(src.getType()) ||
            !dyn_cast<MemRefType>(fp.getType()) ||
            !dyn_cast<IndexType>(indexRow.getType()) ||
            !dyn_cast<IndexType>(indexCol.getType()) ||
            !dyn_cast<MemRefType>(dst.getType())) {
          op.emitError("ins/outs are not correct yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TInsertFPOp>(
            rewriter, op, TypeRange{}, src, fp, indexRow, indexCol, dst,
            op.getAccToVecModeAttr(), op.getReluPreModeAttr());
      }

      DefaultInlineVector<mlir::pto::TFillPadOp> fillpadops;
      func.walk([&](mlir::pto::TFillPadOp op) { fillpadops.push_back(op); });

      for (auto op : fillpadops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TFillPadOp>(
            op,
            TypeRange{},
            src,
            dst,
            op.getPadValueAttr());
      }

      DefaultInlineVector<mlir::pto::TFillPadInplaceOp> fillpadInplaceOps;
      func.walk(
          [&](mlir::pto::TFillPadInplaceOp op) { fillpadInplaceOps.push_back(op); });

      for (auto op : fillpadInplaceOps) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TFillPadInplaceOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      // --- TSetValOp [Dst, Offset, Val] ---
      // Lower tile-world scalar write to memref-world SETVAL DPS op.
      DefaultInlineVector<mlir::pto::TSetValOp> tsetvalops;
      func.walk([&](mlir::pto::TSetValOp op) { tsetvalops.push_back(op); });

      for (auto op : tsetvalops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value dst = op.getDst();
        Value offset = op.getOffset();
        Value val = op.getVal();

        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!dstTy) {
          op.emitError("dst is not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TSetValOp>(
            op,
            TypeRange{},
            dst,
            offset,
            val);
      }

      // --- TGetValOp [Src, Offset] -> Scalar ---
      // Lower tile-world scalar read to memref-world GETVAL DPS op.
      DefaultInlineVector<mlir::pto::TGetValOp> tgetvalops;
      func.walk([&](mlir::pto::TGetValOp op) { tgetvalops.push_back(op); });

      for (auto op : tgetvalops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value offset = op.getOffset();
        Type dstType = op.getDst().getType();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        if (!srcTy) {
          op.emitError("src is not memref yet");
          signalPassFailure();
          return;
        }

        auto newOp = rewriter.create<pto::TGetValOp>(
            op.getLoc(),
            dstType,
            src,
            offset);
        rewriter.replaceOp(op, newOp.getDst());
      }

      DefaultInlineVector<mlir::pto::TGatherOp> gatherops;
      func.walk([&](mlir::pto::TGatherOp op) { gatherops.push_back(op); });

      for (auto op : gatherops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value cdst = op.getCdst();
        Value indices = op.getIndices();
        Value tmp = op.getTmp();
        Value kValue = op.getKValue();
        auto maskPattern = op.getMaskPatternAttr();
        auto cmpMode = op.getCmpModeAttr();
        auto offset = op.getOffsetAttr();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());

        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        if (maskPattern) {
          rewriter.replaceOpWithNewOp<pto::TGatherOp>(
              op,
              TypeRange{},
              src,
              dst,
              /*cdst=*/Value(),
              /*indices=*/Value(),
              /*tmp=*/Value(),
              /*kValue=*/Value(),
              /*maskPattern=*/maskPattern,
              /*cmpMode=*/pto::CmpModeAttr(),
              /*offset=*/IntegerAttr());
          continue;
        }

        if (cdst || kValue) {
          auto cdstTy = dyn_cast<MemRefType>(cdst.getType());
          auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
          if (!cdstTy || !tmpTy) {
            op.emitError("compare-form tgather expects cdst/tmp to be memref yet");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::TGatherOp>(
              op,
              TypeRange{},
              src,
              dst,
              cdst,
              /*indices=*/Value(),
              tmp,
              kValue,
              /*maskPattern=*/pto::MaskPatternAttr(),
              cmpMode,
              offset);
          continue;
        }

        if (indices || tmp) {
          auto indicesTy = dyn_cast<MemRefType>(indices.getType());
          auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
          if (!indicesTy || !tmpTy) {
            op.emitError("index-form tgather expects indices/tmp to be memref yet");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::TGatherOp>(
              op,
              TypeRange{},
              src,
              dst,
              /*cdst=*/Value(),
              indices,
              tmp,
              /*kValue=*/Value(),
              /*maskPattern=*/pto::MaskPatternAttr(),
              /*cmpMode=*/pto::CmpModeAttr(),
              /*offset=*/IntegerAttr());
          continue;
        }

        op.emitError("expects tgather to be in mask, index+tmp, or compare+tmp form");
        signalPassFailure();
        return;
      }

      DefaultInlineVector<mlir::pto::TGatherBOp> gatherbops;
      func.walk([&](mlir::pto::TGatherBOp op) { gatherbops.push_back(op); });

      for (auto op : gatherbops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value offsets = op.getOffsets();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto offsetsTy = dyn_cast<MemRefType>(offsets.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !offsetsTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TGatherBOp>(
            op,
            TypeRange{},
            src,
            offsets,
            dst);
      }

      DefaultInlineVector<mlir::pto::TLogOp> logops;
      func.walk([&](mlir::pto::TLogOp op) { logops.push_back(op); });

      for (auto op : logops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TLogOp>(
            op,
            TypeRange{},
            src,
            dst,
            op.getPrecisionTypeAttr());
      }

      DefaultInlineVector<mlir::pto::TLReluOp> lreluops;
      func.walk([&](mlir::pto::TLReluOp op) { lreluops.push_back(op); });

      for (auto op : lreluops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value slope = op.getSlope();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto slopeTy = dyn_cast<FloatType>(slope.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !slopeTy || !dstTy) {
          op.emitError("ins/outs are not correct type yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TLReluOp>(
            op,
            TypeRange{},
            src,
            slope,
            dst);
      }

      DefaultInlineVector<mlir::pto::TMaxOp> maxops;
      func.walk([&](mlir::pto::TMaxOp op) { maxops.push_back(op); });

      for (auto op : maxops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TMaxOp>(rewriter,
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TMaxSOp> maxsops;
      func.walk([&](mlir::pto::TMaxSOp op) { maxsops.push_back(op); });

      for (auto op : maxsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        bool scalarIsScalar = isa<IntegerType, FloatType>(scalar.getType());
        if (!srcTy || !scalarIsScalar || !dstTy) {
          op.emitError("expects src/dst to be memref and scalar to be integer/float");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TMaxSOp>(rewriter,
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      DefaultInlineVector<mlir::pto::TMinOp> minops;
      func.walk([&](mlir::pto::TMinOp op) { minops.push_back(op); });

      for (auto op : minops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TMinOp>(rewriter,
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TMinSOp> minsops;
      func.walk([&](mlir::pto::TMinSOp op) { minsops.push_back(op); });

      for (auto op : minsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        bool scalarIsScalar = isa<IntegerType, FloatType>(scalar.getType());
        if (!srcTy || !scalarIsScalar || !dstTy) {
          op.emitError("expects src/dst to be memref and scalar to be integer/float");
          signalPassFailure();
          return;
        }

        replaceOpWithClonedAttrs<pto::TMinSOp>(rewriter,
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      DefaultInlineVector<mlir::pto::TMovFPOp> movfpops;
      func.walk([&](mlir::pto::TMovFPOp op) { movfpops.push_back(op); });

      for (auto op : movfpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto fpTy = dyn_cast<MemRefType>(fp.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !fpTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TMovFPOp>(
            op,
            TypeRange{},
            src,
            fp,
            dst);
      }

      DefaultInlineVector<mlir::pto::TQuantOp> quantops;
      func.walk([&](mlir::pto::TQuantOp op) { quantops.push_back(op); });

      for (auto op : quantops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
        Value tmp = op.getTmp();
        Value offset = op.getOffset();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto fpTy = dyn_cast<MemRefType>(fp.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !fpTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }
        if (tmp && !dyn_cast<MemRefType>(tmp.getType())) {
          op.emitError("tmp is not memref yet");
          signalPassFailure();
          return;
        }
        if (offset && !dyn_cast<MemRefType>(offset.getType())) {
          op.emitError("offset is not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TQuantOp>(
            op,
            TypeRange{},
            src,
            fp,
            offset,
            tmp,
            dst,
            op.getQuantTypeAttr());
      }

      DefaultInlineVector<mlir::pto::TMrgSortOp> mrgsortops;
      func.walk([&](mlir::pto::TMrgSortOp op) { mrgsortops.push_back(op); });

      for (auto op : mrgsortops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        if (op.isFormat1()) {
          Value src = op.getSrc();
          Value dst = op.getDst();
          Value blockLenVal = op.getBlockLen();

          auto srcTy = dyn_cast<MemRefType>(src.getType());
          auto dstTy = dyn_cast<MemRefType>(dst.getType());
          if (!srcTy || !dstTy) {
            op.emitError("ins/outs are not memref yet");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::TMrgSortOp>(
              op,
              TypeRange{},
              ValueRange{src},
              blockLenVal,
              ValueRange{dst},
              Value() /*tmp*/,
              Value() /*excuted*/,
              op.getExhaustedAttr());
        } else if (op.isFormat2()) {
          bool allMemRef = true;
          for (Value v : op.getSrcs())
            if (!dyn_cast<MemRefType>(v.getType())) { allMemRef = false; break; }
          if (!allMemRef) {
            op.emitError("format2 ins/outs are not memref yet");
            signalPassFailure();
            return;
          }
          if (op.getDsts().size() != 1u || !op.getTmp()) {
            op.emitError("format2 expects outs(dst) and ins(tmp)");
            signalPassFailure();
            return;
          }

          Value dst = op.getDst();
          Value tmp = op.getTmp();
          Value excuted = op.getExcuted();
          if (!dyn_cast<MemRefType>(dst.getType()) || !dyn_cast<MemRefType>(tmp.getType())) {
            op.emitError("format2 dst/tmp must be memref");
            signalPassFailure();
            return;
          }
          if (!dyn_cast<VectorType>(excuted.getType())) {
            op.emitError("format2 outs(excuted) must be vector");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::TMrgSortOp>(
              op,
              TypeRange{},
              op.getSrcs(),
              Value() /*blockLen*/,
              ValueRange{dst},
              tmp,
              excuted,
              op.getExhaustedAttr());
        } else {
          op.emitError("tmrgsort must be format1 or format2");
          signalPassFailure();
          return;
        }
      }

      DefaultInlineVector<mlir::pto::TNegOp> negops;
      func.walk([&](mlir::pto::TNegOp op) { negops.push_back(op); });

      for (auto op : negops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TNegOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      DefaultInlineVector<mlir::pto::TNotOp> notops;
      func.walk([&](mlir::pto::TNotOp op) { notops.push_back(op); });

      for (auto op : notops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TNotOp>(
            op,
            TypeRange{},
            src,
            dst);
      }

      DefaultInlineVector<mlir::pto::TOrOp> orops;
      func.walk([&](mlir::pto::TOrOp op) { orops.push_back(op); });

      for (auto op : orops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TOrOp>(
            op,
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TOrSOp> orsops;
      func.walk([&](mlir::pto::TOrSOp op) { orsops.push_back(op); });

      for (auto op : orsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto scalarTy = dyn_cast<IntegerType>(scalar.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !scalarTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TOrSOp>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      DefaultInlineVector<mlir::pto::TPartAddOp> partaddops;
      func.walk([&](mlir::pto::TPartAddOp op) { partaddops.push_back(op); });

      for (auto op : partaddops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TPartAddOp>(
            op,
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::TPartMulOp> partmulops;
      func.walk([&](mlir::pto::TPartMulOp op) { partmulops.push_back(op); });

      for (auto op : partmulops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TPartMulOp>(
            op,
            src0,
            src1,
            dst);
      }

      DefaultInlineVector<mlir::pto::MGatherOp> mgatherops;
      func.walk([&](mlir::pto::MGatherOp op) { mgatherops.push_back(op); });

      for (auto op : mgatherops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value dst = op.getDst();
        Value idx = op.getIdx();
        Value mem = op.getMem();

        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto idxTy = dyn_cast<MemRefType>(idx.getType());
        auto memTy = dyn_cast<MemRefType>(mem.getType());
        if (!dstTy || !idxTy || !memTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MGatherOp>(
            op,
            TypeRange{},
            mem,
            idx,
            dst,
            /*scratch=*/op.getScratch(),
            op.getCoalesceAttr(),
            op.getGatherOobAttr());
      }

      DefaultInlineVector<mlir::pto::MScatterOp> mascatterops;
      func.walk([&](mlir::pto::MScatterOp op) { mascatterops.push_back(op); });

      for (auto op : mascatterops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value idx = op.getIdx();
        Value mem = op.getMem();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto idxTy = dyn_cast<MemRefType>(idx.getType());
        auto memTy = dyn_cast<MemRefType>(mem.getType());
        if (!srcTy || !idxTy || !memTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MScatterOp>(
            op,
            TypeRange{},
            src,
            idx,
            mem,
            op.getCoalesceAttr(),
            op.getScatterAtomicOpAttr(),
            op.getScatterOobAttr(),
            op.getScatterConflictAttr());
      }
      DefaultInlineVector<mlir::pto::TPrintOp> printops;
      func.walk([&](mlir::pto::TPrintOp op) { printops.push_back(op); });

      for (auto op : printops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value tmp = op->getNumOperands() > 1 ? op->getOperand(1) : Value();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto tmpTy = tmp ? dyn_cast<MemRefType>(tmp.getType()) : MemRefType();
        if (!srcTy || (tmp && !tmpTy)) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TPrintOp>(
            op,
            TypeRange{},
            src,
            tmp,
            dyn_cast_or_null<pto::PrintFormatAttr>(
                op.getProperties().printFormat));
      }

      // ------------------------------------------------------------------
      // Stage 4: Reconcile control-flow result types
      // ------------------------------------------------------------------
      if (failed(reconcileSCFIfResultTypes(func))) {
        signalPassFailure();
        return;
      }
      if (failed(reconcileSCFForResultTypes(func))) {
        signalPassFailure();
        return;
      }
      // Mark memref-form set_validshape only after control-flow result-type
      // reconciliation. Values such as scf.if results can stay tile_buf until
      // this late stage.
      if (failed(markLoweredSetValidShapeOps(func, ctx))) {
        signalPassFailure();
        return;
      }
    }
    
    // Debug Output
    LLVM_DEBUG(llvm::dbgs() << mod.getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<PTOViewToMemrefPass>();
}

} // namespace pto
} // namespace mlir
