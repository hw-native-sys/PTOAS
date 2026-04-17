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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "Utils.h" // 假设包含一些通用的工具函数

#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOVIEWTOMEMREF

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

namespace {

static void markForceDynamicValidShape(Operation *op, bool force,
                                       MLIRContext *ctx);

static Type convertPTOTypeToMemRef(Type t);

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
  
  // 如果追溯到 BlockArgument (函数参数) 或其他无法穿透的 Op，则返回空
  return {}; 
}

// =============================================================================
// Helper: Valid dims backtracking (v_row / v_col)
// =============================================================================
static void lookupValidDims(Value v, Value &vRow, Value &vCol) {
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
  vRow = Value();
  vCol = Value();
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
  int32_t fractalSize = 512;
  int32_t compactMode = 0;
};

static int64_t getElemBytes(Type elemTy) {
  if (auto ft = elemTy.dyn_cast<FloatType>()) {
    if (ft.isF16() || ft.isBF16()) return 2;
    if (ft.isF32()) return 4;
    if (ft.isF64()) return 8;
  }
  if (auto it = elemTy.dyn_cast<IntegerType>()) {
    int64_t bytes = it.getWidth() / 8;
    return bytes > 0 ? bytes : 1;
  }
  return -1;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<BLayoutAttr>(attr)) {
    out = (int32_t)a.getValue();
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = (int32_t)a.getInt();
    return true;
  }
  return false;
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<SLayoutAttr>(attr)) {
    out = (int32_t)a.getValue();
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = (int32_t)a.getInt();
    return true;
  }
  return false;
}

static bool readCompactModeI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<CompactModeAttr>(attr)) {
    out = (int32_t)a.getValue();
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = (int32_t)a.getInt();
    return true;
  }
  return false;
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

static bool computeBoxInnerShape(const TileLayoutConfig &config, Type elemTy,
                                 TileLayoutInfo &info) {
  info.boxed = config.sLayout != 0;
  if (!info.boxed) {
    info.innerRows = 1;
    info.innerCols = 1;
    return true;
  }

  int64_t elemBytes = getElemBytes(elemTy);
  if (elemBytes <= 0)
    return false;

  switch (config.fractalSize) {
  case 1024:
    info.innerRows = 16;
    info.innerCols = 16;
    return true;
  case 32:
    info.innerRows = 16;
    info.innerCols = 2;
    return true;
  case 512:
    if (config.sLayout == 1) {
      info.innerRows = 16;
      info.innerCols = 32 / elemBytes;
      return true;
    }
    if (config.sLayout == 2) {
      info.innerRows = 32 / elemBytes;
      info.innerCols = 16;
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
    if (config.compactMode == 2)
      return majorStride + 1;
    return majorStride;
  };
  if (!info.boxed) {
    if (config.bLayout == 1) {
      info.rowStride = 1;
      info.colStride = applyCompactToMajorStride(rows);
      return true;
    }
    info.rowStride = applyCompactToMajorStride(cols);
    info.colStride = 1;
    return true;
  }

  if (config.bLayout == 1) {
    if (config.sLayout != 1)
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
  if (shape.size() != 2 || llvm::is_contained(shape, ShapedType::kDynamic))
    return false;

  TileLayoutConfig config = getTileLayoutConfig(cfg);
  return computeBoxInnerShape(config, elemTy, info) &&
         computeTilePointerStrides(config, shape, info);
}

static void collectAffineAddTerms(AffineExpr root,
                                  SmallVectorImpl<AffineExpr> &terms) {
  SmallVector<AffineExpr, 4> pending{root};
  while (!pending.empty()) {
    AffineExpr current = pending.pop_back_val();
    auto addExpr = current.dyn_cast<AffineBinaryOpExpr>();
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
  if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
    strides[dim.getPosition()] = 1;
    return true;
  }

  auto mulExpr = expr.dyn_cast<AffineBinaryOpExpr>();
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;

  auto assignStride = [&](AffineExpr dimExpr,
                          AffineExpr constantExpr) -> bool {
    auto dim = dimExpr.dyn_cast<AffineDimExpr>();
    auto constant = constantExpr.dyn_cast<AffineConstantExpr>();
    if (!dim || !constant)
      return false;
    strides[dim.getPosition()] = constant.getValue();
    return true;
  };
  return assignStride(mulExpr.getLHS(), mulExpr.getRHS()) ||
         assignStride(mulExpr.getRHS(), mulExpr.getLHS());
}

static void decomposeStridedLayout(AffineMap map,
                                   SmallVectorImpl<int64_t> &strides) {
  strides.assign(map.getNumDims(), 0);
  if (map.getNumResults() != 1)
    return;

  SmallVector<AffineExpr, 4> terms;
  collectAffineAddTerms(map.getResult(0), terms);
  for (AffineExpr term : terms)
    (void)tryAssignAffineStride(term, strides);
}

static Value makeIndexConstant(IRRewriter &rewriter, Location loc,
                               int64_t value) {
  return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                            rewriter.getIndexAttr(value));
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
  if (validShape.size() >= 1 && validShape[0] >= 0)
    vRow = makeIndexConstant(rewriter, loc, validShape[0]);
  if (validShape.size() >= 2 && validShape[1] >= 0)
    vCol = makeIndexConstant(rewriter, loc, validShape[1]);
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

static Value computeSubsetValidDim(IRRewriter &rewriter, Location loc,
                                   Value parentValid, Value offset,
                                   int64_t size, Operation *anchorOp) {
  Value sizeVal = rewriter.create<arith::ConstantIndexOp>(loc, size);
  if (!parentValid)
    return sizeVal;

  int64_t pvConst = 0, offConst = 0;
  if (getConstIndexValue(parentValid, pvConst) &&
      getConstIndexValue(offset, offConst)) {
    int64_t diff = 0;
    if (pvConst > 0) {
      int64_t offMod = offConst % pvConst;
      if (offMod < 0)
        offMod += pvConst;
      diff = pvConst - offMod; // in [1, pvConst] when pvConst>0
    }
    if (diff < 0)
      diff = 0;
    int64_t clipped = std::min<int64_t>(size, diff);
    return rewriter.create<arith::ConstantIndexOp>(loc, clipped);
  }

  Value pv = ensureIndex(rewriter, loc, parentValid, anchorOp);
  Value off = ensureIndex(rewriter, loc, offset, anchorOp);

  // Use the same "periodic valid dims" rule as SubsetOp::inferReturnTypes:
  // diff = pv - (off % pv), so offsets that land on the next tile (off == pv)
  // still produce a full valid dim (diff == pv), instead of 0.
  Type i64Ty = rewriter.getI64Type();
  Value pvI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pv);
  Value offI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, off);
  Value remI64 = rewriter.create<arith::RemUIOp>(loc, offI64, pvI64);
  Value diffI64 = rewriter.create<arith::SubIOp>(loc, pvI64, remI64);
  Value diff = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   diffI64);

  Value lt = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, diff,
                                            sizeVal);
  return rewriter.create<arith::SelectOp>(loc, lt, diff, sizeVal);
}

static void dumpPretty(Operation *op, llvm::raw_ostream &os) {
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

static Type convertPTOTypeToMemRef(Type t) {
  // 1. 处理 !pto.ptr<T>
  if (auto pty = dyn_cast<mlir::pto::PtrType>(t)) {
    return MemRefType::get({ShapedType::kDynamic}, pty.getElementType(),
                           MemRefLayoutAttrInterface(), Attribute());
  }
  
  // 2. 处理 !pto.tile_buf<...>
  if (auto tbTy = dyn_cast<mlir::pto::TileBufType>(t)) {
    SmallVector<int64_t> strides;
    
    // Try layout-aware strides first (BLayout/SLayout-aware).
    auto shape = tbTy.getShape();
    TileLayoutInfo info;
    bool gotLayout = false;
    if (computeTileLayoutInfo(tbTy.getConfigAttr(), tbTy.getElementType(), shape,
                              info)) {
      strides = {info.rowStride, info.colStride};
      gotLayout = true;
    }

    // Fallback: Row-Major contiguous strides.
    if (!gotLayout) {
      strides.resize(shape.size());
      int64_t s = 1;
      for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        if (shape[i] != ShapedType::kDynamic)
          s *= shape[i];
        else
          s = ShapedType::kDynamic;
      }
    }

    // 构造归一化的 Strided Layout
    // 【关键】Offset 设为 Dynamic (?)。
    // 这对于 Subview 出来的 MemRef 和 Alloc 出来的 MemRef 都必须一致，
    // 否则 TAdd 的两个输入类型不匹配会报错。
    auto layoutAttr = StridedLayoutAttr::get(t.getContext(), 
                                             ShapedType::kDynamic, // offset: ?
                                             strides);

    return MemRefType::get(
        tbTy.getShape(), 
        tbTy.getElementType(), 
        layoutAttr,
        tbTy.getMemorySpace()
    );
  }
  // 其他类型透传
  return t;
}

// Ensure scf.if result types follow the rewritten yield operand types.
// PTOViewToMemref rewrites tile values to memref in branch bodies, but scf.if
// result types are not auto-updated by those op-local rewrites.
static LogicalResult reconcileSCFIfResultTypes(func::FuncOp func) {
  SmallVector<scf::IfOp, 8> ifOps;
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

static void rewriteFunctionSignature(func::FuncOp func, MLIRContext *ctx) {
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

static LogicalResult lowerAllocTileOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::AllocTileOp, 8> allocTiles;
  func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

  for (auto op : allocTiles) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy)
      continue;

    SmallVector<int64_t, 4> shape(tbTy.getShape().begin(), tbTy.getShape().end());
    Type elemTy = tbTy.getElementType();

    SmallVector<int64_t> strides;
    TileLayoutInfo info;
    if (computeTileLayoutInfo(tbTy.getConfigAttr(), elemTy, shape, info))
      strides = {info.rowStride, info.colStride};
    else
      strides = computeCompactStrides(shape);

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

static LogicalResult lowerDeclareTileOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::DeclareTileOp, 8> declaredTiles;
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

static LogicalResult lowerMakeTensorViewOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::MakeTensorViewOp, 8> makeViews;
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

    SmallVector<OpFoldResult, 4> sizes;
    for (Value value : op.getShape())
      sizes.push_back(ensureIndex(rewriter, loc, value, op));
    SmallVector<OpFoldResult, 4> strides;
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

static LogicalResult lowerTensorViewDimOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::GetTensorViewDimOp, 8> tvDims;
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

static LogicalResult foldAddPtrIntoScalarOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::LoadScalarOp, 8> loadScalars;
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
      totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                                : off;
      base = add.getOperand(0);
    }
    if (foldedAddPtr) {
      auto newOp =
          rewriter.create<pto::LoadScalarOp>(loc, op.getValue().getType(), base,
                                             totalOffset);
      rewriter.replaceOp(op, newOp.getValue());
    }
  }

  SmallVector<mlir::pto::StoreScalarOp, 8> storeScalars;
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
      totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                                : off;
      base = add.getOperand(0);
    }
    if (foldedAddPtr) {
      rewriter.create<pto::StoreScalarOp>(loc, base, totalOffset, op.getValue());
      rewriter.eraseOp(op);
    }
  }

  SmallVector<Operation *, 8> addPtrs;
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
  SmallVector<mlir::pto::PartitionViewOp, 8> partitionViews;
  func.walk([&](mlir::pto::PartitionViewOp op) { partitionViews.push_back(op); });

  for (auto op : partitionViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value src = op.getOperand(0);
    auto srcMrTy = dyn_cast<MemRefType>(src.getType());
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
      IntegerAttr constAttr;
      bool isStatic = false;
      if (auto cOp = offset.getDefiningOp<arith::ConstantIndexOp>()) {
        constAttr = rewriter.getIndexAttr(cOp.value());
        isStatic = true;
      } else if (auto cInt = offset.getDefiningOp<arith::ConstantIntOp>()) {
        constAttr = rewriter.getIndexAttr(cInt.value());
        isStatic = true;
      }
      mixedOffsets.push_back(isStatic ? OpFoldResult(constAttr)
                                      : OpFoldResult(ensureIndex(rewriter, loc,
                                                                 offset, op)));
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

static LogicalResult lowerSubsetOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::SubsetOp, 8> subsets;
  func.walk([&](mlir::pto::SubsetOp op) { subsets.push_back(op); });

  for (auto op : subsets) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto resultTileTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    Value src = op->getOperand(0);
    auto srcMrTy = dyn_cast<MemRefType>(src.getType());
    if (!srcMrTy) {
      op.emitError("pto.subset source must be lowered to memref first");
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
      IntegerAttr constAttr;
      bool isStatic = false;
      if (auto cOp = offset.getDefiningOp<arith::ConstantIndexOp>()) {
        constAttr = rewriter.getIndexAttr(cOp.value());
        isStatic = true;
      } else if (auto cInt = offset.getDefiningOp<arith::ConstantIntOp>()) {
        constAttr = rewriter.getIndexAttr(cInt.value());
        isStatic = true;
      }
      mixedOffsets.push_back(isStatic ? OpFoldResult(constAttr)
                                      : OpFoldResult(ensureIndex(rewriter, loc,
                                                                 offset, op)));
    }

    auto configAttr = lookupConfig(src);
    if (!configAttr)
      configAttr = pto::TileBufConfigAttr::getDefault(ctx);

    TileLayoutInfo layoutInfo;
    if (!computeTileLayoutInfo(configAttr, srcMrTy.getElementType(),
                               srcMrTy.getShape(), layoutInfo)) {
      op.emitError("unsupported tile layout for pto.subset");
      return failure();
    }

    if (layoutInfo.boxed) {
      if (staticSizes.size() != 2 || op.getOffsets().size() != 2) {
        op.emitError("boxed layout subset expects 2D sizes/offsets");
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

      int32_t bl = 0;
      (void)readBLayoutI32(configAttr.getBLayout(), bl);
      auto srcShape = srcMrTy.getShape();
      if (srcShape.size() == 2) {
        if (bl == 0) {
          if (staticSizes[1] != srcShape[1]) {
            op.emitError("boxed RowMajor subset must keep full cols");
            return failure();
          }
          if (!off1Const || off1 != 0) {
            op.emitError("boxed RowMajor subset requires static col offset = 0");
            return failure();
          }
        } else {
          if (staticSizes[0] != srcShape[0]) {
            op.emitError("boxed ColMajor subset must keep full rows");
            return failure();
          }
          if (!off0Const || off0 != 0) {
            op.emitError("boxed ColMajor subset requires static row offset = 0");
            return failure();
          }
        }
      }
    }

    SmallVector<int64_t> srcStrides;
    int64_t srcOffset = ShapedType::kDynamic;
    if (failed(getStridesAndOffset(srcMrTy, srcStrides, srcOffset)))
      srcStrides = computeCompactStrides(srcMrTy.getShape());

    auto resultLayout =
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, srcStrides);
    auto resultMemRefType =
        MemRefType::get(staticSizes, srcMrTy.getElementType(), resultLayout,
                        srcMrTy.getMemorySpace());
    SmallVector<OpFoldResult> mixedStrides(staticSizes.size(),
                                           rewriter.getIndexAttr(1));
    auto sv = rewriter.create<memref::SubViewOp>(loc, resultMemRefType, src,
                                                 mixedOffsets, mixedSizes,
                                                 mixedStrides);

    Value parentVRow;
    Value parentVCol;
    lookupValidDims(src, parentVRow, parentVCol);
    Value vRow;
    Value vCol;
    if (!staticSizes.empty())
      vRow = computeSubsetValidDim(rewriter, loc, parentVRow, op.getOffsets()[0],
                                   staticSizes[0], op);
    if (staticSizes.size() > 1)
      vCol = computeSubsetValidDim(rewriter, loc, parentVCol, op.getOffsets()[1],
                                   staticSizes[1], op);

    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, resultMemRefType, sv.getResult(), vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(bindOp,
                               resultTileTy && resultTileTy.hasDynamicValid(),
                               ctx);
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

  auto srcMrTy = dyn_cast<MemRefType>(src.getType());
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
  SmallVector<mlir::pto::TReshapeOp, 8> reshapes;
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

  SmallVector<mlir::pto::BitcastOp, 8> bitcasts;
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
    : public PassWrapper<PTOViewToMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOViewToMemrefPass)

  StringRef getArgument() const final { return "pto-view-to-memref"; }
  StringRef getDescription() const final {
    return "Lower PTO views to memref with Metadata Binding";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::pto::PTODialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;

      Block &entry = func.front();
      auto fnTy = func.getFunctionType();

      // ------------------------------------------------------------------
      // Stage 0: Rewrite Function Signature
      // ------------------------------------------------------------------
      SmallVector<Type> newInputs;
      for (Type t : fnTy.getInputs()) newInputs.push_back(convertPTOTypeToMemRef(t));

      SmallVector<Type> newResults;
      for (Type t : fnTy.getResults()) newResults.push_back(convertPTOTypeToMemRef(t));

      // Update entry block arguments
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        if (entry.getArgument(i).getType() != newInputs[i]) {
            entry.getArgument(i).setType(newInputs[i]);
        }
      }

      // Update function type
      func.setFunctionType(FunctionType::get(ctx, newInputs, newResults));

      // ------------------------------------------------------------------
      // Stage 0.5: lower pto.alloc_tile -> memref.alloc + pto.bind_tile
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::AllocTileOp, 8> allocTiles;
      func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

      for (auto op : allocTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
        if (!tbTy) continue;

        // 1. 获取 Shape 和 ElementType
        SmallVector<int64_t, 4> shape(tbTy.getShape().begin(), tbTy.getShape().end());
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
        ArrayRef<int64_t> validShape = tbTy.getValidShape();
        if (!tbTy.hasDynamicValid()) {
          // TileBuf valid dims use a negative sentinel (e.g. '?' / -1), which is
          // distinct from MLIR's ShapedType::kDynamic (INT64_MIN). Treat any
          // negative value as dynamic here.
          if (validShape.size() >= 1 && validShape[0] >= 0) {
            vRow = rewriter
                       .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(validShape[0]))
                       .getResult();
          }
          if (validShape.size() >= 2 && validShape[1] >= 0) {
            vCol = rewriter
                       .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(validShape[1]))
                       .getResult();
          }
        }

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
      // Stage 0.75: lower pto.declare_tile -> pto.declare_tile_memref +
      //             pto.bind_tile
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::DeclareTileOp, 8> declaredTiles;
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
        ArrayRef<int64_t> validShape = tbTy.getValidShape();
        if (!tbTy.hasDynamicValid()) {
          if (validShape.size() >= 1 && validShape[0] >= 0) {
            vRow = rewriter
                       .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(validShape[0]))
                       .getResult();
          }
          if (validShape.size() >= 2 && validShape[1] >= 0) {
            vCol = rewriter
                       .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(validShape[1]))
                       .getResult();
          }
        }

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
      SmallVector<mlir::pto::TAssignOp, 8> tassignOps;
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
      // Stage 1: Lower pto.make_tensor_view -> memref.reinterpret_cast
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::MakeTensorViewOp, 8> makeViews;
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

        SmallVector<OpFoldResult, 4> sizes;
        for (Value v : op.getShape()) sizes.push_back(ensureIndex(rewriter, loc, v, op));

        SmallVector<OpFoldResult, 4> strides;
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
      SmallVector<mlir::pto::GetTensorViewDimOp, 8> tvDims;
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
      // Stage 1.35: Lower pto.subset -> memref.subview + pto.bind_tile
      // ------------------------------------------------------------------
      if (failed(lowerSubsetOps(func, ctx))) {
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

      // ------------------------------------------------------------------
      // Stage 1.5: Fold pto.addptr chains into load/store_scalar.
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::LoadScalarOp, 8> loadScalars;
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

      SmallVector<mlir::pto::StoreScalarOp, 8> storeScalars;
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
        SmallVector<mlir::pto::AddPtrOp, 8> addPtrsForPipeInit;
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
      SmallVector<Operation *, 8> addPtrs;
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
      SmallVector<mlir::pto::TLoadOp, 8> loads;
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
      SmallVector<mlir::pto::TStoreOp, 8> storeops;
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
      SmallVector<mlir::pto::TTransOp, 8> trans;
      func.walk([&](mlir::pto::TTransOp op) { trans.push_back(op); });
      for (auto op : trans) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TTransOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1), op->getOperand(2));
      }

      // --- TExpOp [Src, Dst] ---
      SmallVector<mlir::pto::TExpOp, 8> exp;
      func.walk([&](mlir::pto::TExpOp op) { exp.push_back(op); });
      for (auto op : exp) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TExpOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1));
      }

      // --- TMulOp [Src, Scalar, Dst] ---
      SmallVector<mlir::pto::TMulOp, 8> mul;
      func.walk([&](mlir::pto::TMulOp op) { mul.push_back(op); });
      for (auto op : mul) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMulOp>(
            op, op->getOperand(0), op.getOperand(1), op->getOperand(2));
      }

      // --- TMulSOp [Src, Scalar, Dst] ---
      SmallVector<mlir::pto::TMulSOp, 8> muls;
      func.walk([&](mlir::pto::TMulSOp op) { muls.push_back(op); });
      for (auto op : muls) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMulSOp>(
            op, op->getOperand(0), op.getScalar(), op->getOperand(2));
      }

      // --- TAddOp [Src0, Src1, Dst] ---
      SmallVector<mlir::pto::TAddOp, 8> addops;
      func.walk([&](mlir::pto::TAddOp op) { addops.push_back(op); });
      for (auto op : addops) {
          IRRewriter rewriter(ctx);
          rewriter.setInsertionPoint(op);
          
          Value src0 = op->getOperand(0);
          auto config = lookupConfig(src0);
          
          rewriter.replaceOpWithNewOp<pto::TAddOp>(
              op, TypeRange{}, 
              op->getOperand(0), op->getOperand(1), op->getOperand(2));
      }

      // --- TMatmulOp [Lhs, Rhs, Dst] (no optional bias in ODS) ---
      SmallVector<mlir::pto::TMatmulOp , 8> matmuls;
      func.walk([&](mlir::pto::TMatmulOp  op) { matmuls.push_back(op); });
      for (auto op : matmuls) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value dst = op->getOperand(2);

        auto config = lookupConfig(lhs);

        rewriter.replaceOpWithNewOp<pto::TMatmulOp>(op, TypeRange{}, lhs, rhs, dst);
      }

      // --- TMatmulAccOp [Acc, Lhs, Rhs, Dst] ---
      SmallVector<mlir::pto::TMatmulAccOp , 8> matmulAccs;
      func.walk([&](mlir::pto::TMatmulAccOp  op) { matmulAccs.push_back(op); });
      for (auto op : matmulAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulAccOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TMatmulBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
      SmallVector<mlir::pto::TMatmulBiasOp , 8> matmulBiass;
      func.walk([&](mlir::pto::TMatmulBiasOp  op) { matmulBiass.push_back(op); });
      for (auto op : matmulBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulBiasOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TMatmulMxOp---
      SmallVector<mlir::pto::TMatmulMxOp , 8> matmulMxs;
      func.walk([&](mlir::pto::TMatmulMxOp  op) { matmulMxs.push_back(op); });
      for (auto op : matmulMxs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulMxOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4));
      }

      // --- TMatmulMxAccOp  ---
      SmallVector<mlir::pto::TMatmulMxAccOp , 8> matmulMxAccs;
      func.walk([&](mlir::pto::TMatmulMxAccOp  op) { matmulMxAccs.push_back(op); });
      for (auto op : matmulMxAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulMxAccOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4), op->getOperand(5));
      }

      // --- TMatmulMxBiasOp ---
      SmallVector<mlir::pto::TMatmulMxBiasOp , 8> matmulMxBiass;
      func.walk([&](mlir::pto::TMatmulMxBiasOp  op) { matmulMxBiass.push_back(op); });
      for (auto op : matmulMxBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMatmulMxBiasOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4), op->getOperand(5));
      }

      // --- TGemvOp [Lhs, Rhs, Dst] ---
      SmallVector<mlir::pto::TGemvOp , 8> gemvs;
      func.walk([&](mlir::pto::TGemvOp  op) { gemvs.push_back(op); });
      for (auto op : gemvs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value dst = op->getOperand(2);

        auto config = lookupConfig(lhs);

        rewriter.replaceOpWithNewOp<pto::TGemvOp>(
          op, TypeRange{}, lhs, rhs, dst);
      }

      // --- TGemvAccOp [Acc, Lhs, Rhs, Dst] ---
      SmallVector<mlir::pto::TGemvAccOp , 8> gemvAccs;
      func.walk([&](mlir::pto::TGemvAccOp  op) { gemvAccs.push_back(op); });
      for (auto op : gemvAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvAccOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TGemvBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
      SmallVector<mlir::pto::TGemvBiasOp , 8> gemvBiass;
      func.walk([&](mlir::pto::TGemvBiasOp  op) { gemvBiass.push_back(op); });
      for (auto op : gemvBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvBiasOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TGemvMxOp [A, AScale, B, BScale, Dst] ---
      SmallVector<mlir::pto::TGemvMxOp , 8> gemvMxs;
      func.walk([&](mlir::pto::TGemvMxOp  op) { gemvMxs.push_back(op); });
      for (auto op : gemvMxs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvMxOp>(
          op, TypeRange{},
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4));
      }

      // --- TGemvMxAccOp [CIn, A, AScale, B, BScale, Dst] ---
      SmallVector<mlir::pto::TGemvMxAccOp , 8> gemvMxAccs;
      func.walk([&](mlir::pto::TGemvMxAccOp  op) { gemvMxAccs.push_back(op); });
      for (auto op : gemvMxAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvMxAccOp>(
          op, TypeRange{},
          op->getOperand(0), op->getOperand(1), op->getOperand(2),
          op->getOperand(3), op->getOperand(4), op->getOperand(5));
      }

      // --- TGemvMxBiasOp [A, AScale, B, BScale, Bias, Dst] ---
      SmallVector<mlir::pto::TGemvMxBiasOp , 8> gemvMxBiass;
      func.walk([&](mlir::pto::TGemvMxBiasOp  op) { gemvMxBiass.push_back(op); });
      for (auto op : gemvMxBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TGemvMxBiasOp>(
          op, TypeRange{},
          op->getOperand(0), op->getOperand(1), op->getOperand(2),
          op->getOperand(3), op->getOperand(4), op->getOperand(5));
      }

      // --- TMovOp [Src, Dst] ---
      SmallVector<mlir::pto::TMovOp , 8> movs;
      func.walk([&](mlir::pto::TMovOp  op) { movs.push_back(op); });
      for (auto op : movs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TMovOp>(
            op, TypeRange{}, op.getSrc(), op.getDst(), op.getFp(),
            op.getPreQuantScalar(), op.getAccToVecModeAttr(),
            op.getReluPreModeAttr());
      }

      SmallVector<mlir::pto::TAbsOp, 8> abseops;
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

      SmallVector<mlir::pto::TAddCOp, 8> addcops;
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

      SmallVector<mlir::pto::TAddSOp, 8> addsops;
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

        rewriter.replaceOpWithNewOp<pto::TAddSOp>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TAddSCOp, 8> addscops;
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

      SmallVector<mlir::pto::TAndOp, 8> andops;
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

      SmallVector<mlir::pto::TConcatOp, 8> concats;
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

      SmallVector<mlir::pto::TAndSOp, 8> andsops;
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

      SmallVector<mlir::pto::TCIOp, 8> ciops;
      func.walk([&](mlir::pto::TCIOp op) { ciops.push_back(op); });

      for (auto op : ciops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value s = op->getOperand(0);
        Value dst = op.getDst();
        bool descending = op.getDescending();

        auto sTy = dyn_cast<IntegerType>(s.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!sTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TCIOp>(
            op,
            TypeRange{},
            s,
            dst,
            descending);
      }

      SmallVector<mlir::pto::TCmpOp, 8> cmpops;
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

      SmallVector<mlir::pto::TCmpSOp, 8> cmpsops;
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

      SmallVector<mlir::pto::TColExpandOp, 8> colexpand;
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

      SmallVector<mlir::pto::TColMaxOp, 8> colmaxops;
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

      SmallVector<mlir::pto::TColMinOp, 8> colminops;
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

      SmallVector<mlir::pto::TColExpandMulOp, 8> colexpandmulops;
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

      SmallVector<mlir::pto::TColExpandMaxOp, 8> colexpandmaxops;
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

      SmallVector<mlir::pto::TColExpandMinOp, 8> colexpandminops;
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

      SmallVector<mlir::pto::TColSumOp, 8> colsumops;
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

      SmallVector<mlir::pto::TCvtOp, 8> cvtops;
      func.walk([&](mlir::pto::TCvtOp op) { cvtops.push_back(op); });

      for (auto op : cvtops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value tmp = op.getTmp();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }
        if (tmp && !dyn_cast<MemRefType>(tmp.getType())) {
          op.emitError("tmp is not memref yet");
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
            tmp,
            rmodeAttr,
            satModeAttr);

        rewriter.replaceOp(op, newOp->getResults());
      }

      SmallVector<mlir::pto::TDivOp, 8> divops;
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

        rewriter.replaceOpWithNewOp<pto::TDivOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TDivSOp, 8> divsops;
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
        rewriter.replaceOpWithNewOp<pto::TDivSOp>(
            op,
            TypeRange{},
            src,
            scale,
            dst);
      }

      SmallVector<mlir::pto::TExpandsOp, 8> expandsops;
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

        rewriter.replaceOpWithNewOp<pto::TExpandsOp>(
            op,
            TypeRange{},
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TExtractOp, 8> extractops;
      func.walk([&](mlir::pto::TExtractOp op) { extractops.push_back(op); });

      for (auto op : extractops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value indexRow = op.getIndexRow();
        Value indexCol = op.getIndexCol();
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

        rewriter.replaceOpWithNewOp<pto::TExtractOp>(
            op,
            TypeRange{},
            src,
            indexRow,
            indexCol,
            dst);
      }

      SmallVector<mlir::pto::TFillPadOp, 8> fillpadops;
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
            dst);
      }

      SmallVector<mlir::pto::TFillPadInplaceOp, 8> fillpadInplaceOps;
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
      SmallVector<mlir::pto::TSetValOp, 8> tsetvalops;
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
      SmallVector<mlir::pto::TGetValOp, 8> tgetvalops;
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

      SmallVector<mlir::pto::TGatherOp, 8> gatherops;
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

      SmallVector<mlir::pto::TGatherBOp, 8> gatherbops;
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

      SmallVector<mlir::pto::TLogOp, 8> logops;
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
            dst);
      }

      SmallVector<mlir::pto::TLReluOp, 8> lreluops;
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

      SmallVector<mlir::pto::TMaxOp, 8> maxops;
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

        rewriter.replaceOpWithNewOp<pto::TMaxOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TMaxSOp, 8> maxsops;
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

        rewriter.replaceOpWithNewOp<pto::TMaxSOp>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TMinOp, 8> minops;
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

        rewriter.replaceOpWithNewOp<pto::TMinOp>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TMinSOp, 8> minsops;
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

        rewriter.replaceOpWithNewOp<pto::TMinSOp>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TMovFPOp, 8> movfpops;
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

      SmallVector<mlir::pto::TQuantOp, 8> quantops;
      func.walk([&](mlir::pto::TQuantOp op) { quantops.push_back(op); });

      for (auto op : quantops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
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
            dst,
            op.getQuantTypeAttr());
      }

      SmallVector<mlir::pto::TMrgSortOp, 8> mrgsortops;
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

      SmallVector<mlir::pto::TNegOp, 8> negops;
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

      SmallVector<mlir::pto::TNotOp, 8> notops;
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

      SmallVector<mlir::pto::TOrOp, 8> orops;
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

      SmallVector<mlir::pto::TOrSOp, 8> orsops;
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

      SmallVector<mlir::pto::TPartAddOp, 8> partaddops;
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

      SmallVector<mlir::pto::TPartMulOp, 8> partmulops;
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

      SmallVector<mlir::pto::MGatherOp, 8> mgatherops;
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
            dst);
      }

      SmallVector<mlir::pto::MScatterOp, 8> mascatterops;
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
            mem);
      }
      SmallVector<mlir::pto::TPrintOp, 8> printops;
      func.walk([&](mlir::pto::TPrintOp op) { printops.push_back(op); });

      for (auto op : printops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        if (!srcTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::TPrintOp>(
            op,
            TypeRange{},
            src);
      }

      // ------------------------------------------------------------------
      // Stage 4: Reconcile control-flow result types
      // ------------------------------------------------------------------
      if (failed(reconcileSCFIfResultTypes(func))) {
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
    dumpPretty(mod.getOperation(), llvm::errs());
  }
};

} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<PTOViewToMemrefPass>();
}

} // namespace pto
} // namespace mlir
