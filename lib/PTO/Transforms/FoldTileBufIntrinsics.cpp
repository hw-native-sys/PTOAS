// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- FoldTileBufIntrinsics.cpp ------------------------------------------===//
//
// After TileLang DSL template functions are inlined, the IR contains
// structured-view intrinsics that reference template parameters:
//
// tile_buf family:
//   - pto.tile_buf_addr   → extract memref address from tile_buf
//   - pto.tile_valid_rows → extract valid row count
//   - pto.tile_valid_cols → extract valid column count
//
// tensor_view family:
//   - pto.tensor_view_addr       → extract memref/ptr from tensor_view
//   - pto.get_tensor_view_dim    → extract dimension size
//   - pto.get_tensor_view_stride → extract dimension stride
//
// This pass resolves them against the concrete values at the call site.
// For tile_buf intrinsics, the active VPTO path folds against materialized tile
// handles produced by the shared tile-handle bridge (`pto.alloc_tile` or
// `pto.materialize_tile`).
// For tensor_view intrinsics, the pass traces through the full
// unrealized_conversion_cast → memref.subview → memref.reinterpret_cast
// chain to fold directly to constants or SSA operands from the
// reinterpret_cast, without generating intermediate memref.dim /
// memref.extract_strided_metadata ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_FOLDTILEBUFINTRINSICS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

static void eraseDeadAllocTileOps(func::FuncOp func) {
  SmallVector<pto::AllocTileOp> deadAllocs;
  func.walk([&](pto::AllocTileOp alloc) {
    if (alloc.getResult().use_empty())
      deadAllocs.push_back(alloc);
  });

  for (pto::AllocTileOp alloc : llvm::reverse(deadAllocs))
    alloc.erase();
}

struct TileHandleInfo {
  Value sourceMemref;
  Value addr;
  Value validRow;
  Value validCol;
  pto::TileBufConfigAttr config;
};

// Walk through value-conversion casts that preserve the underlying producer.
// ExpandTileOp / PTOInlineLibCall and other passes may wrap tile_buf values in
// UnrealizedConversionCastOp / arith.index_cast / memref.cast when bridging
// tile_buf parameter types. These wrappers must not defeat anchor recovery.
static Value unwrapBridgingCasts(Value v) {
  while (v) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      break;
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      if (cast.getNumOperands() == 1 && cast.getNumResults() == 1) {
        v = cast.getOperand(0);
        continue;
      }
      break;
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(defOp)) {
      v = cast.getIn();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      v = cast.getSource();
      continue;
    }
    break;
  }
  return v;
}

static std::optional<TileHandleInfo> resolveTileHandle(Value tileBuf,
                                                       Operation *user) {
  tileBuf = unwrapBridgingCasts(tileBuf);
  if (auto alloc = tileBuf.getDefiningOp<pto::AllocTileOp>()) {
    auto tileTy = dyn_cast<pto::TileBufType>(alloc.getResult().getType());
    if (!tileTy) {
      user->emitError(
          "FoldTileBufIntrinsics: pto.alloc_tile must produce !pto.tile_buf");
      return std::nullopt;
    }
    return TileHandleInfo{Value(), alloc.getAddr(), alloc.getValidRow(),
                          alloc.getValidCol(), tileTy.getConfigAttr()};
  }

  if (auto materialize = tileBuf.getDefiningOp<pto::MaterializeTileOp>()) {
    return TileHandleInfo{materialize.getSource(), Value(),
                          materialize.getValidRow(), materialize.getValidCol(),
                          materialize.getConfig()};
  }

  user->emitError("FoldTileBufIntrinsics: expected tile_buf to be defined by "
                  "the active materialized tile-handle bridge "
                  "(pto.alloc_tile or pto.materialize_tile)");
  return std::nullopt;
}

static MemRefType getCanonicalMemRefTypeForTileBuf(pto::TileBufType tileTy) {
  return MemRefType::get(tileTy.getShape(), tileTy.getElementType(),
                         AffineMap(), tileTy.getMemorySpace());
}

struct ViewChain {
  UnrealizedConversionCastOp cast;
  memref::SubViewOp subview;
  memref::ReinterpretCastOp reinterpretCast;
  Value baseMemref;
};

static std::optional<ViewChain> traceViewChain(Value tensorView,
                                               Operation *user) {
  Value memrefVal;
  UnrealizedConversionCastOp castOp;

  if (isa<MemRefType>(tensorView.getType())) {
    memrefVal = tensorView;
  } else {
    castOp = tensorView.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1) {
      user->emitError(
          "FoldTileBufIntrinsics: expected tensor_view to be defined by a "
          "single-operand builtin.unrealized_conversion_cast");
      return std::nullopt;
    }
    memrefVal = castOp.getOperand(0);
    if (!isa<MemRefType>(memrefVal.getType())) {
      user->emitError(
          "FoldTileBufIntrinsics: expected cast operand to be a memref, got ")
          << memrefVal.getType();
      return std::nullopt;
    }
  }

  auto subviewOp = memrefVal.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp) {
    user->emitError("FoldTileBufIntrinsics: expected memref to be defined by "
                    "memref.subview, got ")
        << (memrefVal.getDefiningOp()
                ? memrefVal.getDefiningOp()->getName().getStringRef()
                : StringRef("block argument"));
    return std::nullopt;
  }

  auto rcOp = subviewOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!rcOp) {
    user->emitError(
        "FoldTileBufIntrinsics: expected subview source to be defined by "
        "memref.reinterpret_cast, got ")
        << (subviewOp.getSource().getDefiningOp()
                ? subviewOp.getSource().getDefiningOp()->getName().getStringRef()
                : StringRef("block argument"));
    return std::nullopt;
  }

  return ViewChain{castOp, subviewOp, rcOp, rcOp.getSource()};
}

static bool getConstIndexValue(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexValue(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexValue(truncOp.getIn(), out);
  return false;
}

static Value getValueOrCreateConstant(OpBuilder &builder, Location loc,
                                      OpFoldResult ofr) {
  if (auto val = dyn_cast<Value>(ofr))
    return val;
  auto intAttr = dyn_cast<IntegerAttr>(cast<Attribute>(ofr));
  assert(intAttr && "expected integer attribute in OpFoldResult");
  return builder.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
}

static bool isAllStaticZero(ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    auto attr = dyn_cast<Attribute>(ofr);
    if (!attr)
      return false;
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr || intAttr.getInt() != 0)
      return false;
  }
  return true;
}

static Value computeResultStride(OpBuilder &builder, Location loc,
                                 OpFoldResult rcStride,
                                 OpFoldResult svStride) {
  if (auto attr = dyn_cast<Attribute>(svStride)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (intAttr && intAttr.getInt() == 1)
      return getValueOrCreateConstant(builder, loc, rcStride);
  }

  Value lhs = getValueOrCreateConstant(builder, loc, rcStride);
  Value rhs = getValueOrCreateConstant(builder, loc, svStride);
  return builder.create<arith::MulIOp>(loc, lhs, rhs);
}

static Value computeLinearOffset(OpBuilder &builder, Location loc,
                                 ArrayRef<OpFoldResult> rcOffsets,
                                 ArrayRef<OpFoldResult> svOffsets,
                                 ArrayRef<OpFoldResult> rcStrides) {
  bool rcAllZero = isAllStaticZero(rcOffsets);
  bool svAllZero = isAllStaticZero(svOffsets);

  if (rcAllZero && svAllZero)
    return Value();

  Value svPart;
  if (!svAllZero) {
    for (auto [svOffset, rcStride] : llvm::zip(svOffsets, rcStrides)) {
      if (auto attr = dyn_cast<Attribute>(svOffset)) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (intAttr && intAttr.getInt() == 0)
          continue;
      }

      Value off = getValueOrCreateConstant(builder, loc, svOffset);
      Value stride = getValueOrCreateConstant(builder, loc, rcStride);
      Value term = builder.create<arith::MulIOp>(loc, off, stride);
      svPart = svPart ? builder.create<arith::AddIOp>(loc, svPart, term) : term;
    }
  }

  Value rcPart;
  if (!rcAllZero) {
    if (rcOffsets.empty())
      return Value();
    rcPart = getValueOrCreateConstant(builder, loc, rcOffsets.front());
  }

  if (rcPart && svPart)
    return builder.create<arith::AddIOp>(loc, rcPart, svPart);
  return rcPart ? rcPart : svPart;
}

struct FoldTileBufIntrinsicsPass
    : public pto::impl::FoldTileBufIntrinsicsBase<FoldTileBufIntrinsicsPass> {
  using FoldTileBufIntrinsicsBase::FoldTileBufIntrinsicsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Leftover TileLang template instances (private, uncalled after
    // PTOInlineLibCall) still contain pto.tile_buf_addr / tile_valid_*
    // ops on tile_buf function arguments — they have no materialized tile
    // handle anchor to fold against and will be removed by later DCE. Skip
    // them.
    if (func->hasAttr("pto.tilelang.instance"))
      return;

    SmallVector<pto::TileBufAddrOp, 8> addrOps;
    SmallVector<pto::TileValidRowsOp, 8> rowsOps;
    SmallVector<pto::TileValidColsOp, 8> colsOps;
    SmallVector<pto::TensorViewAddrOp, 8> tvAddrOps;
    SmallVector<pto::GetTensorViewDimOp, 8> tvDimOps;
    SmallVector<pto::GetTensorViewStrideOp, 8> tvStrideOps;

    func.walk([&](Operation *op) {
      if (auto addr = dyn_cast<pto::TileBufAddrOp>(op))
        addrOps.push_back(addr);
      else if (auto rows = dyn_cast<pto::TileValidRowsOp>(op))
        rowsOps.push_back(rows);
      else if (auto cols = dyn_cast<pto::TileValidColsOp>(op))
        colsOps.push_back(cols);
      else if (auto tvAddr = dyn_cast<pto::TensorViewAddrOp>(op))
        tvAddrOps.push_back(tvAddr);
      else if (auto tvDim = dyn_cast<pto::GetTensorViewDimOp>(op))
        tvDimOps.push_back(tvDim);
      else if (auto tvStride = dyn_cast<pto::GetTensorViewStrideOp>(op))
        tvStrideOps.push_back(tvStride);
    });

    // Fold pto.tile_buf_addr by recovering the active materialized tile
    // handle contract:
    //   - pto.materialize_tile → use the source memref directly
    //   - pto.alloc_tile       → rebuild a memref from the explicit addr
    // When the requested result type is already !pto.ptr<...>, cast from the
    // recovered memref instead of leaving tile_buf_addr in the IR.
    for (auto addrOp : addrOps) {
      auto handleInfo = resolveTileHandle(addrOp.getSrc(), addrOp);
      if (!handleInfo)
        return signalPassFailure();

      auto tileTy = dyn_cast<pto::TileBufType>(addrOp.getSrc().getType());
      if (!tileTy) {
        addrOp.emitError("FoldTileBufIntrinsics: tile_buf_addr source must be "
                         "!pto.tile_buf");
        return signalPassFailure();
      }

      if (auto resultMemrefType = dyn_cast<MemRefType>(addrOp.getDst().getType())) {
        if (handleInfo->sourceMemref) {
          Value srcMemref = handleInfo->sourceMemref;
          if (!isa<MemRefType>(srcMemref.getType())) {
            addrOp.emitError(
                "FoldTileBufIntrinsics: pto.materialize_tile source is not a memref");
            return signalPassFailure();
          }

          // The declared tile_buf_addr result type may differ from the actual
          // materialized source layout (e.g. plain shape vs. strided layout).
          if (srcMemref.getType() != resultMemrefType)
            addrOp.getDst().setType(cast<MemRefType>(srcMemref.getType()));
          addrOp.getDst().replaceAllUsesWith(srcMemref);
          addrOp.erase();
          continue;
        }

        if (!handleInfo->addr) {
          addrOp.emitError("FoldTileBufIntrinsics: pto.alloc_tile used by "
                           "tile_buf_addr must carry an addr operand on the "
                           "VPTO path");
          return signalPassFailure();
        }

        builder.setInsertionPoint(addrOp);
        Value replacement = builder.create<pto::PointerCastOp>(
            addrOp.getLoc(), resultMemrefType, ValueRange{handleInfo->addr},
            handleInfo->validRow ? handleInfo->validRow : Value(),
            handleInfo->validCol ? handleInfo->validCol : Value(),
            handleInfo->config);
        addrOp.getDst().replaceAllUsesWith(replacement);
        addrOp.erase();
        continue;
      }

      auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
      if (!resultPtrType) {
        addrOp.emitError(
            "FoldTileBufIntrinsics: tile_buf_addr result must be memref or !pto.ptr");
        return signalPassFailure();
      }

      Value memrefValue;
      if (handleInfo->sourceMemref) {
        memrefValue = handleInfo->sourceMemref;
        if (!isa<MemRefType>(memrefValue.getType())) {
          addrOp.emitError(
              "FoldTileBufIntrinsics: pto.materialize_tile source is not a memref");
          return signalPassFailure();
        }
      } else {
        if (!handleInfo->addr) {
          addrOp.emitError("FoldTileBufIntrinsics: pto.alloc_tile used by "
                           "tile_buf_addr must carry an addr operand on the "
                           "VPTO path");
          return signalPassFailure();
        }

        builder.setInsertionPoint(addrOp);
        auto canonicalMemrefType = getCanonicalMemRefTypeForTileBuf(tileTy);
        memrefValue = builder.create<pto::PointerCastOp>(
            addrOp.getLoc(), canonicalMemrefType, ValueRange{handleInfo->addr},
            handleInfo->validRow ? handleInfo->validRow : Value(),
            handleInfo->validCol ? handleInfo->validCol : Value(),
            handleInfo->config);
      }

      builder.setInsertionPoint(addrOp);
      Value replacement =
          builder.create<pto::CastPtrOp>(addrOp.getLoc(), resultPtrType,
                                         memrefValue);
      addrOp.getDst().replaceAllUsesWith(replacement);
      addrOp.erase();
    }

    // Fold pto.tile_valid_rows → arith.constant (static) or the dynamic
    // valid_row operand carried by the new tile handle bridge.
    for (auto rowsOp : rowsOps) {
      builder.setInsertionPoint(rowsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(rowsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().empty()) {
        rowsOp.emitError("tile_valid_rows: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vRow = tbTy.getValidShape()[0];
      Value replacement;
      if (vRow != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(rowsOp.getLoc(), vRow);
      } else {
        auto handleInfo = resolveTileHandle(rowsOp.getSrc(), rowsOp);
        if (!handleInfo)
          return signalPassFailure();
        replacement = handleInfo->validRow;
        if (!replacement) {
          rowsOp.emitError(
              "tile_valid_rows: dynamic v_row but the materialized tile "
              "handle has no valid_row operand");
          return signalPassFailure();
        }
        assert(replacement.getType() == rowsOp.getResult().getType() &&
               "tile_valid_rows fold: type mismatch with handle valid_row");
      }
      rowsOp.getResult().replaceAllUsesWith(replacement);
      rowsOp.erase();
    }

    // Fold pto.tile_valid_cols → arith.constant (static) or the dynamic
    // valid_col operand carried by the new tile handle bridge.
    for (auto colsOp : colsOps) {
      builder.setInsertionPoint(colsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(colsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().size() < 2) {
        colsOp.emitError("tile_valid_cols: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vCol = tbTy.getValidShape()[1];
      Value replacement;
      if (vCol != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(colsOp.getLoc(), vCol);
      } else {
        auto handleInfo = resolveTileHandle(colsOp.getSrc(), colsOp);
        if (!handleInfo)
          return signalPassFailure();
        replacement = handleInfo->validCol;
        if (!replacement) {
          colsOp.emitError(
              "tile_valid_cols: dynamic v_col but the materialized tile "
              "handle has no valid_col operand");
          return signalPassFailure();
        }
        assert(replacement.getType() == colsOp.getResult().getType() &&
               "tile_valid_cols fold: type mismatch with handle valid_col");
      }
      colsOp.getResult().replaceAllUsesWith(replacement);
      colsOp.erase();
    }

    for (auto addrOp : tvAddrOps) {
      auto chain = traceViewChain(addrOp.getSrc(), addrOp);
      if (!chain)
        return signalPassFailure();

      builder.setInsertionPoint(addrOp);

      auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
      if (!resultPtrType) {
        if (auto resultMemrefType =
                dyn_cast<MemRefType>(addrOp.getDst().getType())) {
          Value base = chain->baseMemref;
          if (base.getType() != resultMemrefType)
            addrOp.getDst().setType(cast<MemRefType>(base.getType()));
          addrOp.getDst().replaceAllUsesWith(base);
          addrOp.erase();
          continue;
        }
        addrOp.emitError(
            "FoldTileBufIntrinsics: tensor_view_addr result must be memref or "
            "!pto.ptr");
        return signalPassFailure();
      }

      Value linearOffset =
          computeLinearOffset(builder, addrOp.getLoc(),
                              chain->reinterpretCast.getMixedOffsets(),
                              chain->subview.getMixedOffsets(),
                              chain->reinterpretCast.getMixedStrides());

      Value basePtr = builder.create<pto::CastPtrOp>(
          addrOp.getLoc(), resultPtrType, chain->baseMemref);
      Value replacement =
          linearOffset
              ? builder.create<pto::AddPtrOp>(addrOp.getLoc(), resultPtrType,
                                              basePtr, linearOffset)
              : basePtr;

      addrOp.getDst().replaceAllUsesWith(replacement);
      addrOp.erase();
    }

    for (auto dimOp : tvDimOps) {
      auto chain = traceViewChain(dimOp.getTensorView(), dimOp);
      if (!chain)
        return signalPassFailure();

      int64_t dimIdx = 0;
      if (!getConstIndexValue(dimOp.getDimIndex(), dimIdx)) {
        dimOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_dim requires a constant "
            "dim index");
        return signalPassFailure();
      }

      auto svTy = cast<MemRefType>(chain->subview.getType());
      if (dimIdx < 0 || dimIdx >= svTy.getRank()) {
        dimOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_dim dim index out of "
            "bounds");
        return signalPassFailure();
      }

      builder.setInsertionPoint(dimOp);
      Value replacement;
      if (!svTy.isDynamicDim(dimIdx)) {
        replacement =
            builder.create<arith::ConstantIndexOp>(dimOp.getLoc(),
                                                   svTy.getDimSize(dimIdx));
      } else {
        replacement = getValueOrCreateConstant(
            builder, dimOp.getLoc(), chain->subview.getMixedSizes()[dimIdx]);
      }

      dimOp.getResult().replaceAllUsesWith(replacement);
      dimOp.erase();
    }

    for (auto strideOp : tvStrideOps) {
      auto chain = traceViewChain(strideOp.getTensorView(), strideOp);
      if (!chain)
        return signalPassFailure();

      int64_t dimIdx = 0;
      if (!getConstIndexValue(strideOp.getDimIndex(), dimIdx)) {
        strideOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_stride requires a "
            "constant dim index");
        return signalPassFailure();
      }

      auto svTy = cast<MemRefType>(chain->subview.getType());
      if (dimIdx < 0 || dimIdx >= svTy.getRank()) {
        strideOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_stride dim index out of "
            "bounds");
        return signalPassFailure();
      }

      builder.setInsertionPoint(strideOp);
      Value replacement = computeResultStride(
          builder, strideOp.getLoc(),
          chain->reinterpretCast.getMixedStrides()[dimIdx],
          chain->subview.getMixedStrides()[dimIdx]);

      strideOp.getResult().replaceAllUsesWith(replacement);
      strideOp.erase();
    }

    // Clean up dead unrealized_conversion_cast ops that bridged
    // memref -> partition_tensor_view / tile_buf and are now unused
    // after folding.
    SmallVector<UnrealizedConversionCastOp, 8> deadCasts;
    func.walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp.use_empty() && castOp.getNumOperands() == 1 &&
          isa<MemRefType>(castOp.getOperand(0).getType()) &&
          isa<pto::PartitionTensorViewType, pto::TileBufType>(
              castOp.getResult(0).getType()))
        deadCasts.push_back(castOp);
    });
    for (auto castOp : llvm::reverse(deadCasts))
      castOp.erase();

    while (true) {
      SmallVector<Operation *, 8> deadMemrefOps;
      func.walk([&](Operation *op) {
        if ((isa<memref::SubViewOp>(op) ||
             isa<memref::ReinterpretCastOp>(op)) &&
            op->use_empty())
          deadMemrefOps.push_back(op);
      });
      if (deadMemrefOps.empty())
        break;
      for (auto *op : llvm::reverse(deadMemrefOps))
        op->erase();
    }

    eraseDeadAllocTileOps(func);
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass() {
  return std::make_unique<FoldTileBufIntrinsicsPass>();
}

} // namespace pto
} // namespace mlir
