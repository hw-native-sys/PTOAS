// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- LowerPTOToUBufOps.cpp - Lower pto.tadd/tsub/tmul/tdiv on a2a3 -----===//
//===----------------------------------------------------------------------===//
//
// Lowers pto.tadd/tsub/tmul/tdiv to pto.ub.vadd/vsub/vmul/vdiv on a3.
// Uses the full CCE dispatch tree from TBinOp.hpp with all modes.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_LOWERPTOTOUBUFOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {
static constexpr int64_t kRepeatMax = 255;
static constexpr int64_t kRepeatStrideMax = 255;
static constexpr int64_t kSmallRptBinOp = 4;
static constexpr int64_t kDefaultRepeatStride = 8;
static constexpr unsigned kMaskLen = 64;
static constexpr unsigned kHalfElementSizeBytes = 2;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static unsigned getElementSize(Type elemTy) {
  if (elemTy.isF16() || elemTy.isBF16()) {
    return kHalfElementSizeBytes;
  }
  if (elemTy.isF32()) {
    return mlir::pto::kValue4;
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    if (width == mlir::pto::kValue16 || width == mlir::pto::kValue32) {
      return width / mlir::pto::kValue8;
    }
  }
  return 0;
}

static Type getStoredElemType(Type ty) {
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    return tbTy.getElementType();
  }
  if (auto mrTy = dyn_cast<MemRefType>(ty)) {
    return mrTy.getElementType();
  }
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    return ptrTy.getElementType();
  }
  return Type();
}

/// Returns true if the type lives in UB (VEC) address space.
static std::optional<bool> isUBMemorySpaceImpl(Type ty) {
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    auto msAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
    if (!msAttr) {
      return false;
    }
    return msAttr.getAddressSpace() == pto::AddressSpace::VEC;
  }
  if (auto mrTy = dyn_cast<MemRefType>(ty)) {
    auto msAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace());
    if (!msAttr) {
      return false;
    }
    return msAttr.getAddressSpace() == pto::AddressSpace::VEC;
  }
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    return ptrTy.getMemorySpace().getAddressSpace() == pto::AddressSpace::VEC;
  }
  return std::nullopt;
}

/// Returns true if the given type is confirmed UB memory space.
static bool isUBMemorySpace(Type ty) {
  auto result = isUBMemorySpaceImpl(ty);
  return result.has_value() && result.value();
}

static bool isRowMajor(pto::TileBufType tbTy) {
  auto config = tbTy.getConfigAttr();
  if (!config) {
    return true;
  }
  return config.getBLayout().getValue() != pto::BLayout::ColMajor;
}

static pto::PtrType getUBPtrType(MLIRContext *ctx, Type elemTy) {
  auto msAttr = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
  return pto::PtrType::get(ctx, elemTy, msAttr);
}

// Gather/pointer lowerings accept either a UB pointer (already materialized)
// or a tile buffer (take its address). Passes `tile` through when it is
// already a pointer.
static Value emitTileOrPtrAddress(OpBuilder &builder, Location loc,
                                  Value tile, pto::PtrType ptrType) {
  if (isa<pto::PtrType>(tile.getType())) {
    return tile;
  }
  return builder.create<pto::TileBufAddrOp>(loc, ptrType, tile).getDst();
}

static std::pair<int64_t, int64_t>
computeContMaskValues(unsigned nElements) {
  int64_t mask0 = (nElements >= kMaskLen)
      ? static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)
      : static_cast<int64_t>((1ULL << nElements) - 1ULL);
  int64_t mask1 = (nElements > kMaskLen)
      ? static_cast<int64_t>((1ULL << (nElements - kMaskLen)) - 1ULL)
      : 0LL;
  return {mask0, mask1};
}

struct TileShapeInfo {
  int64_t vRows;
  int64_t vCols;
  int64_t cols;
  int64_t rows;
  unsigned elemSize;
  unsigned elementsPerRepeat;
  unsigned blockSizeElem;
};

struct TileShapeMetadata {
  SmallVector<int64_t, mlir::pto::kValue2> shape;
  SmallVector<int64_t, mlir::pto::kValue2> validShape;
};

using TileShapeMap = DenseMap<Value, TileShapeMetadata>;

// Bundled (loc, builder, dst, src0, src1, ptrTy) context threaded through
// the binary-op lowering tree below; keeps the mode* family signatures
// short and the data clumps out of the checkers' way.
struct TileOpContext {
  Location loc;
  OpBuilder &b;
  Value dst;
  Value s0;
  Value s1;
  pto::PtrType ptrTy;
};

// Extract (elementType, shape, validShape) from a tile value. A tile_buf
// carries its own shape; a raw pto.ptr is looked up in the planned
// tileShapes map (recorded by collectTileShapes).
struct RawShapeDesc {
  Type elemTy;
  ArrayRef<int64_t> shape;
  ArrayRef<int64_t> validShape;
};

static std::optional<RawShapeDesc> extractRawShape(Value opDst,
                                                   const TileShapeMap &tileShapes) {
  Type dstTy = opDst.getType();
  if (!isUBMemorySpace(dstTy)) {
    return std::nullopt;
  }

  if (auto tbTy = dyn_cast<pto::TileBufType>(dstTy)) {
    if (!isRowMajor(tbTy)) {
      return std::nullopt;
    }
    return RawShapeDesc{tbTy.getElementType(), tbTy.getShape(),
                        tbTy.getValidShape()};
  }
  if (auto mrTy = dyn_cast<MemRefType>(dstTy)) {
    return RawShapeDesc{mrTy.getElementType(), mrTy.getShape(), {}};
  }
  if (isa<pto::PtrType>(dstTy)) {
    auto it = tileShapes.find(opDst);
    if (it == tileShapes.end()) {
      return std::nullopt;
    }
    return RawShapeDesc{cast<pto::PtrType>(dstTy).getElementType(),
                        llvm::ArrayRef(it->second.shape),
                        llvm::ArrayRef(it->second.validShape)};
  }
  return std::nullopt;
}

// Build the final TileShapeInfo from a raw shape description, rejecting
// unsupported element sizes and dynamic dimensions.
static std::optional<TileShapeInfo> finalizeTileShapeInfo(
    const RawShapeDesc &raw) {
  unsigned elemSize = getElementSize(raw.elemTy);
  if (elemSize == 0) {
    return std::nullopt;
  }

  if (raw.shape.size() < mlir::pto::kValue2) {
    return std::nullopt;
  }

  int64_t rows = raw.shape[0];
  int64_t cols = raw.shape[1];
  int64_t vRows = (!raw.validShape.empty() &&
                   raw.validShape[0] != ShapedType::kDynamic)
                      ? raw.validShape[0] : rows;
  int64_t vCols = (raw.validShape.size() >= 2 &&
                   raw.validShape[1] != ShapedType::kDynamic)
                      ? raw.validShape[1] : cols;
  if (vRows == ShapedType::kDynamic || vCols == ShapedType::kDynamic ||
      rows == ShapedType::kDynamic || cols == ShapedType::kDynamic) {
    return std::nullopt;
  }

  TileShapeInfo info;
  info.vRows = vRows;
  info.vCols = vCols;
  info.cols = cols;
  info.rows = rows;
  info.elemSize = elemSize;
  info.elementsPerRepeat = mlir::pto::kValue256 / elemSize;
  info.blockSizeElem = mlir::pto::kValue32 / elemSize;
  return info;
}

static std::optional<TileShapeInfo> extractTileShapeInfoFromValue(
    Value opDst, const TileShapeMap &tileShapes) {
  auto raw = extractRawShape(opDst, tileShapes);
  if (!raw) {
    return std::nullopt;
  }
  return finalizeTileShapeInfo(*raw);
}

static std::optional<TileShapeInfo> extractTileShapeInfo(
    Operation *op, const TileShapeMap &tileShapes) {
  return extractTileShapeInfoFromValue(op->getOperand(mlir::pto::kValue2), tileShapes);
}

static bool canLower(Operation *op, const TileShapeMap &tileShapes) {
  return extractTileShapeInfo(op, tileShapes).has_value();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerPTOToUBufOpsPass
    : public pto::impl::LowerPTOToUBufOpsBase<LowerPTOToUBufOpsPass> {
  using LowerPTOToUBufOpsBase::LowerPTOToUBufOpsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    auto mod = func->getParentOfType<ModuleOp>();
    if (!mod) {
      return;
    }
    auto archAttr = mod->getAttrOfType<StringAttr>("pto.target_arch");
    if (!archAttr ||
        (archAttr.getValue() != "a2" && archAttr.getValue() != "a3")) {
      return;
    }

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // A2/A3: consume planned addresses from PTOPlanMemory /
    // PTOMaterializeTileHandles. Each alloc_tile must carry a planned addr
    // operand.
    TileShapeMap tileShapes;
    if (failed(collectTileShapes(func, ctx, builder, tileShapes))) {
      return;
    }

    // Elementwise tile ops on two tile operands (plus txor's De Morgan form).
    lowerBinaryTileOps(func, ctx, builder, tileShapes);
    lowerXorTileOps(func, ctx, builder, tileShapes);
    // Unary elementwise tile ops.
    lowerUnaryTileOps(func, ctx, builder, tileShapes);
    // Scalar-operand tile ops (tadds/tmuls/tmaxs/tmins/tshls/tshrs).
    lowerScalarTileOps(func, ctx, builder, tileShapes);
    // GM<->UB data movement (tload/tstore).
    lowerMemTileOps(func, builder, tileShapes);
    // Gather ops (block-form and index-form).
    if (failed(lowerGatherTileOps(func, ctx, builder, tileShapes))) {
      return;
    }

    // ---- cleanup dead PTO ops ----
    cleanupDeadPTOOps(func);
  }

private:
  //===--------------------------------------------------------------------===//
  // runOnOperation sub-drivers
  //===--------------------------------------------------------------------===//

  // Consume planned alloc_tile addresses: replace every alloc_tile with a
  // VEC-space CastPtr of its planned address and record the tile shape.
  LogicalResult collectTileShapes(func::FuncOp func, MLIRContext *ctx,
                                  OpBuilder &builder,
                                  TileShapeMap &tileShapes) {
    SmallVector<pto::AllocTileOp> allocOps;
    func.walk([&](pto::AllocTileOp op) { allocOps.push_back(op); });
    for (auto op : allocOps) {
      auto tbTy = cast<pto::TileBufType>(op.getResult().getType());
      if (llvm::any_of(tbTy.getValidShape(), ShapedType::isDynamic)) {
    return op.emitError("A2/A3 UB lowering requires static valid_row and "
                        "valid_col");
      }
    }
    for (auto op : allocOps) {
      auto tbTy = cast<pto::TileBufType>(op.getResult().getType());
      auto shape = tbTy.getShape();
      Value addr = op.getAddr();
      if (!addr) {
    return op.emitError("A3 VPTO UB lowering requires planned alloc_tile "
                        "addresses; run PTOViewToMemref, PTOPlanMemory, "
                        "PTOResolveReservedBuffers, and "
                        "PTOMaterializeTileHandles before LowerPTOToUBufOps");
      }
      builder.setInsertionPoint(op);
      auto ptrTy = pto::PtrType::get(
          ctx, tbTy.getElementType(),
          pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC));
      auto pc = builder.create<pto::CastPtrOp>(op.getLoc(), ptrTy, addr);
      tileShapes[pc.getResult()] = {
          SmallVector<int64_t, 2>(shape),
          SmallVector<int64_t, 2>(tbTy.getValidShape())};
      op.getResult().replaceAllUsesWith(pc.getResult());
      op.erase();
    }

    return success();
  }

  // tadd/taddrelu/tsub/tmul/tdiv/tmax/tmin/tand/tor -> pto.ub.<op>. They all
  // lower through the same shape-driven dispatch.
  template <typename TileOp, typename UBop>
  void lowerBinaryTileFamily(func::FuncOp func, MLIRContext *ctx,
                             OpBuilder &builder,
                             const TileShapeMap &tileShapes) {
    SmallVector<TileOp> ops;
    func.walk([&](TileOp op) { ops.push_back(op); });
    for (auto op : ops) {
      if (!canLower(op, tileShapes)) {
        continue;
      }
      auto info = extractTileShapeInfo(op, tileShapes);
      if (!info) {
        continue;
      }
      auto [ptrs, ptrType] = lowerOpPtrs(builder, ctx, op, op.getDst(),
                                         {op.getSrc0(), op.getSrc1()});
      if (ptrs.empty()) {
        continue;
      }
      Value dstPtr = ptrs[0];
      Value src0Ptr = ptrs[1];
      Value src1Ptr = ptrs[2];
      TileOpContext c{op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                      ptrType};
      dispatch<UBop>(c, *info);
      op.erase();
    }
  }

  void lowerBinaryTileOps(func::FuncOp func, MLIRContext *ctx,
                          OpBuilder &builder, const TileShapeMap &tileShapes) {
    // ---- tadd -> pto.ub.vadd ----
    lowerBinaryTileFamily<pto::TAddOp, pto::UBVaddOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- taddrelu -> pto.ub.vaddrelu ----
    lowerBinaryTileFamily<pto::TAddReluOp, pto::UBVaddReluOp>(func, ctx,
                                                              builder,
                                                              tileShapes);
    // ---- tsub -> pto.ub.vsub ----
    lowerBinaryTileFamily<pto::TSubOp, pto::UBVsubOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- tmul -> pto.ub.vmul ----
    lowerBinaryTileFamily<pto::TMulOp, pto::UBVmulOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- tdiv -> pto.ub.vdiv ----
    lowerBinaryTileFamily<pto::TDivOp, pto::UBVdivOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- tmax -> pto.ub.vmax ----
    lowerBinaryTileFamily<pto::TMaxOp, pto::UBVmaxOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- tmin -> pto.ub.vmin ----
    lowerBinaryTileFamily<pto::TMinOp, pto::UBVminOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- tand -> pto.ub.vand ----
    lowerBinaryTileFamily<pto::TAndOp, pto::UBVandOp>(func, ctx, builder,
                                                      tileShapes);
    // ---- tor -> pto.ub.vor ----
    lowerBinaryTileFamily<pto::TOrOp, pto::UBVorOp>(func, ctx, builder,
                                                    tileShapes);
  }

  // txor -> vor(tmp) + vand(dst) + vnot(dst) + vand(dst,tmp).
  // De Morgan: src0 ^ src1 = ~(src0 & src1) & (src0 | src1).
  void lowerXorTileOps(func::FuncOp func, MLIRContext *ctx,
                       OpBuilder &builder, const TileShapeMap &tileShapes) {
    // ---- txor → vor(tmp) + vand(dst) + vnot(dst) + vand(dst,tmp) ----
    // De Morgan: src0 ^ src1 = ~(src0 & src1) & (src0 | src1)
    {
      SmallVector<pto::TXorOp> ops;
      func.walk([&](pto::TXorOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [ptrs, ptrType] =
            lowerOpPtrs(builder, ctx, op, op.getDst(),
                        {op.getSrc0(), op.getSrc1(), op.getTmp()});
        if (ptrs.empty()) {
          continue;
        }
        Value dstPtr = ptrs[0];
        Value src0Ptr = ptrs[1];
        Value src1Ptr = ptrs[2];
        Value tmpPtr = ptrs[3];
        auto pipeV = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
        // tmp = src0 | src1
        TileOpContext v{op.getLoc(), builder, tmpPtr, src0Ptr, src1Ptr,
                        ptrType};
        dispatch<pto::UBVorOp>(v, *info);
        builder.create<pto::BarrierOp>(op.getLoc(), pipeV);
        // dst = src0 & src1
        TileOpContext a{op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                        ptrType};
        dispatch<pto::UBVandOp>(a, *info);
        builder.create<pto::BarrierOp>(op.getLoc(), pipeV);
        // dst = ~dst
        dispatchUnary<pto::UBVnotOp>(op.getLoc(), builder, dstPtr, dstPtr,
                                     ptrType, *info);
        builder.create<pto::BarrierOp>(op.getLoc(), pipeV);
        // dst = dst & tmp
        TileOpContext t{op.getLoc(), builder, dstPtr, dstPtr, tmpPtr,
                        ptrType};
        dispatch<pto::UBVandOp>(t, *info);
        op.erase();
      }
    }
  }

  // tnot/tabs/trelu/tneg/trecip/texp/tlog/tsqrt/trsqrt -> pto.ub.<op>.

  // Common skeleton for per-op family lowerings: walk one source op type,
  // validate the dst tile shape, lower the dst/src pointers, then hand the
  // prepared operands to `lower`, which emits the family-specific UB ops.
  template <typename SrcOp, typename LowerFn>
  void forEachTileOpWithPtrs(func::FuncOp func, MLIRContext *ctx,
                             OpBuilder &builder,
                             const TileShapeMap &tileShapes, LowerFn lower) {
    SmallVector<SrcOp> ops;
    func.walk([&](SrcOp op) { ops.push_back(op); });
    for (auto op : ops) {
      auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
      if (!info) {
        continue;
      }
      auto [ptrs, ptrType] =
          lowerOpPtrs(builder, ctx, op, op.getDst(), {op.getSrc()});
      if (ptrs.empty()) {
        continue;
      }
      lower(op, ptrs, ptrType, *info);
      op.erase();
    }
  }

  // tnot → pto.ub.vnot, tabs → pto.ub.vabs, trelu → pto.ub.vrelu,
  // texp → pto.ub.vexp, tlog → pto.ub.vln, tsqrt → pto.ub.vsqrt,
  // trsqrt → pto.ub.vrsqrt: a plain dispatch on the prepared operands.
  template <typename SrcOp, typename UBop>
  void lowerUnaryFamily(func::FuncOp func, MLIRContext *ctx,
                        OpBuilder &builder, const TileShapeMap &tileShapes) {
    forEachTileOpWithPtrs<SrcOp>(
        func, ctx, builder, tileShapes,
        [&](SrcOp op, ValueRange ptrs, pto::PtrType ptrType,
            const TileShapeInfo &info) {
          dispatchUnary<UBop>(op.getLoc(), builder, ptrs[0], ptrs[1],
                              ptrType, info);
        });
  }

  // tneg → pto.ub.vmuls(dst, src, -1): needs a typed -1 constant, so it
  // cannot reuse the plain unary dispatch.
  void lowerTNegFamily(func::FuncOp func, MLIRContext *ctx,
                       OpBuilder &builder, const TileShapeMap &tileShapes) {
    forEachTileOpWithPtrs<pto::TNegOp>(
        func, ctx, builder, tileShapes,
        [&](pto::TNegOp op, ValueRange ptrs, pto::PtrType ptrType,
            const TileShapeInfo &info) {
          Type elemTy = ptrType.getElementType();
          Value minusOneScalar;
          if (elemTy.isF32() || elemTy.isF16()) {
            minusOneScalar = builder.create<arith::ConstantOp>(
                op.getLoc(), builder.getFloatAttr(elemTy, -1.0));
          } else {
            minusOneScalar = builder.create<arith::ConstantOp>(
                op.getLoc(), builder.getIntegerAttr(elemTy, -1));
          }
          Value minusOne =
              convertScalarToI64(builder, op.getLoc(), minusOneScalar);
          dispatchShift<pto::UBVmulSOp>(op.getLoc(), builder, ptrs[0],
                                        ptrs[1], minusOne, ptrType, info);
        });
  }

  // trecip → vector_dup(dst, 1) + vdiv(dst, dst, src): a two-step sequence,
  // so it cannot reuse the plain unary dispatch either.
  void lowerTRecipFamily(func::FuncOp func, MLIRContext *ctx,
                         OpBuilder &builder, const TileShapeMap &tileShapes) {
    forEachTileOpWithPtrs<pto::TRecipOp>(
        func, ctx, builder, tileShapes,
        [&](pto::TRecipOp op, ValueRange ptrs, pto::PtrType ptrType,
            const TileShapeInfo &info) {
          Type elemTy = ptrType.getElementType();
          Value oneScalar = builder.create<arith::ConstantOp>(
              op.getLoc(), builder.getFloatAttr(elemTy, 1.0));
          Value one = convertScalarToI64(builder, op.getLoc(), oneScalar);
          dispatchDup(op.getLoc(), builder, ptrs[0], one, ptrType, info);
          builder.create<pto::BarrierOp>(
              op.getLoc(), pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V));
          TileOpContext d{op.getLoc(), builder, ptrs[0], ptrs[0], ptrs[1],
                          ptrType};
          dispatch<pto::UBVdivOp>(d, info);
        });
  }

  void lowerUnaryTileOps(func::FuncOp func, MLIRContext *ctx,
                         OpBuilder &builder, const TileShapeMap &tileShapes) {
    // tnot → pto.ub.vnot, tabs → pto.ub.vabs, trelu → pto.ub.vrelu,
    // texp → pto.ub.vexp, tlog → pto.ub.vln, tsqrt → pto.ub.vsqrt,
    // trsqrt → pto.ub.vrsqrt.
    lowerUnaryFamily<pto::TNotOp, pto::UBVnotOp>(func, ctx, builder,
                                                 tileShapes);
    lowerUnaryFamily<pto::TAbsOp, pto::UBVabsOp>(func, ctx, builder,
                                                 tileShapes);
    lowerUnaryFamily<pto::TReluOp, pto::UBVreluOp>(func, ctx, builder,
                                                   tileShapes);
    lowerUnaryFamily<pto::TExpOp, pto::UBVexpOp>(func, ctx, builder,
                                                 tileShapes);
    lowerUnaryFamily<pto::TLogOp, pto::UBVlnOp>(func, ctx, builder,
                                                tileShapes);
    lowerUnaryFamily<pto::TSqrtOp, pto::UBVsqrtOp>(func, ctx, builder,
                                                   tileShapes);
    lowerUnaryFamily<pto::TRsqrtOp, pto::UBVrsqrtOp>(func, ctx, builder,
                                                     tileShapes);
    // tneg → pto.ub.vmuls(dst, src, -1).
    lowerTNegFamily(func, ctx, builder, tileShapes);
    // trecip → vector_dup(dst, 1) + vdiv(dst, dst, src).
    lowerTRecipFamily(func, ctx, builder, tileShapes);
  }

  // tadds/tmuls/tmaxs/tmins/tshls/tshrs -> pto.ub.<op> (scalar operand).

  // Common skeleton for scalar-operand tile lowerings: walk one source op
  // type, validate the tile shape, lower dst/src pointers (from `srcOf`),
  // convert the scalar operand and emit the matching UB shift-style op.
  template <typename SrcOp, typename UBop>
  void lowerScalarShiftFamily(
      func::FuncOp func, MLIRContext *ctx, OpBuilder &builder,
      const TileShapeMap &tileShapes,
      Value (SrcOp::*srcOf)() = &SrcOp::getSrc) {
    SmallVector<SrcOp> ops;
    func.walk([&](SrcOp op) { ops.push_back(op); });
    for (auto op : ops) {
      auto info = extractTileShapeInfo(op, tileShapes);
      if (!info) {
        continue;
      }
      auto [ptrs, ptrType] =
          lowerOpPtrs(builder, ctx, op, op.getDst(), {(op.*srcOf)()});
      if (ptrs.empty()) {
        continue;
      }
      Value scalarI64 = convertScalarToI64(builder, op.getLoc(), op.getScalar());
      dispatchShift<UBop>(op.getLoc(), builder, ptrs[0], ptrs[1], scalarI64,
                          ptrType, *info);
      op.erase();
    }
  }

  void lowerScalarTileOps(func::FuncOp func, MLIRContext *ctx,
                          OpBuilder &builder, const TileShapeMap &tileShapes) {
    // tadds → pto.ub.vadds, tmaxs → pto.ub.vmaxs, tmins → pto.ub.vmins.
    lowerScalarShiftFamily<pto::TAddSOp, pto::UBVaddSOp>(func, ctx, builder,
                                                         tileShapes);
    lowerScalarShiftFamily<pto::TMaxSOp, pto::UBVmaxSOp>(func, ctx, builder,
                                                         tileShapes);
    lowerScalarShiftFamily<pto::TMinSOp, pto::UBVminSOp>(func, ctx, builder,
                                                         tileShapes);

    // tmuls → pto.ub.vmuls: src0 is the tile operand.
    lowerScalarShiftFamily<pto::TMulSOp, pto::UBVmulSOp>(
        func, ctx, builder, tileShapes, &pto::TMulSOp::getSrc0);

    // tshls → pto.ub.vshl, tshrs → pto.ub.vshr: the shift amount stays a
    // raw scalar (no i64 conversion).
    forEachTileOpWithPtrs<pto::TShlSOp>(
        func, ctx, builder, tileShapes,
        [&](pto::TShlSOp op, ValueRange ptrs, pto::PtrType ptrType,
            const TileShapeInfo &info) {
          dispatchShift<pto::UBVshlOp>(op.getLoc(), builder, ptrs[0], ptrs[1],
                                       op.getScalar(), ptrType, info);
        });
    forEachTileOpWithPtrs<pto::TShrSOp>(
        func, ctx, builder, tileShapes,
        [&](pto::TShrSOp op, ValueRange ptrs, pto::PtrType ptrType,
            const TileShapeInfo &info) {
          dispatchShift<pto::UBVshrOp>(op.getLoc(), builder, ptrs[0], ptrs[1],
                                       op.getScalar(), ptrType, info);
        });
  }

  // tload -> mte_gm_ub, tstore -> mte_ub_gm.
  void lowerMemTileOps(func::FuncOp func, OpBuilder &builder,
                       const TileShapeMap &tileShapes) {
    // ---- tload → mte_gm_ub ----
    SmallVector<pto::TLoadOp> tloadOps;
    func.walk([&](pto::TLoadOp op) { tloadOps.push_back(op); });
    for (auto op : tloadOps) {
      builder.setInsertionPoint(op);
      if (succeeded(lowerTLoad(op, builder, tileShapes))) {
        op.erase();
      }
    }

    // ---- tstore → mte_ub_gm ----
    SmallVector<pto::TStoreOp> tstoreOps;
    func.walk([&](pto::TStoreOp op) { tstoreOps.push_back(op); });
    for (auto op : tstoreOps) {
      builder.setInsertionPoint(op);
      if (succeeded(lowerTStore(op, builder, tileShapes))) {
        op.erase();
      }
    }
  }

  // tgatherb -> pto.ub.vgatherb (GatherBlockHead/Tail tiling) and
  // tgather (index form) -> ub.vmuls + ub.vgather.
  LogicalResult lowerGatherTileOps(func::FuncOp func, MLIRContext *ctx,
                                   OpBuilder &builder,
                                   const TileShapeMap &tileShapes) {
    if (failed(lowerTGatherBOps(func, ctx, builder, tileShapes))) {
      return failure();
    }
    return lowerTGatherIndexOps(func, ctx, builder, tileShapes);
  }

  // GatherBlockHead/Tail tiling geometry shared by the loop nests below.
  struct GatherBlockTiling {
    unsigned blockSizeElem;
    unsigned elementsPerRepeat;
    int64_t validRow;
    int64_t numRepeatPerLine;
    int64_t numRemainPerLine;
    int64_t dstRowStride;
    int64_t offsetRowStride;
  };

  // GatherBlockHead/Tail tiling parameters (from pto-isa a2a3/TGatherB.hpp):
  // vgatherb reads 8 u32 block addresses per repeat, each pointing at a 32B
  // source block. Derived from the dst/offset tile shapes; nullopt when the
  // shapes leave a zero divisor.
  std::optional<GatherBlockTiling>
  computeGatherBlockTiling(const TileShapeInfo &dstInfo,
                           const TileShapeInfo &offsetInfo) {
    GatherBlockTiling t;
    t.blockSizeElem = dstInfo.blockSizeElem;
    t.elementsPerRepeat = dstInfo.elementsPerRepeat;
    t.validRow = dstInfo.vRows;
    int64_t validCol = dstInfo.vCols;
    t.dstRowStride = dstInfo.cols;
    t.offsetRowStride = offsetInfo.cols;
    if (t.blockSizeElem == 0 || t.elementsPerRepeat == 0 ||
        validCol == 0) {
      return std::nullopt;
    }
    t.numRepeatPerLine = validCol / t.elementsPerRepeat;
    t.numRemainPerLine = validCol % t.elementsPerRepeat;
    return t;
  }

  // tgatherb → pto.ub.vgatherb (GatherBlockHead/Tail tiling).
  // Mirrors the pto-isa a2a3/TGatherB.hpp GatherBlockHead/Tail driver.
  // The CCE vgatherb hardware requires specific repeat counts and pointer
  // advancement per call.
  LogicalResult lowerTGatherBOps(func::FuncOp func, MLIRContext *ctx,
                                 OpBuilder &builder,
                                 const TileShapeMap &tileShapes) {
    SmallVector<pto::TGatherBOp> ops;
    func.walk([&](pto::TGatherBOp op) { ops.push_back(op); });
    for (auto op : ops) {
      auto dstInfo = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
      auto offsetInfo =
          extractTileShapeInfoFromValue(op.getOffsets(), tileShapes);
      if (!dstInfo || !offsetInfo) {
        continue;
      }
      auto t = computeGatherBlockTiling(*dstInfo, *offsetInfo);
      if (!t) {
        continue;
      }

      Location loc = op.getLoc();
      builder.setInsertionPoint(op);

      Type dstElemTy = getStoredElemType(op.getDst().getType());
      if (!dstElemTy) {
        continue;
      }
      auto dstPtrType = getUBPtrType(ctx, dstElemTy);
      auto offPtrType = getUBPtrType(ctx, builder.getI32Type());

      Value dstBase = emitTileOrPtrAddress(builder, loc, op.getDst(),
                                           dstPtrType);
      Value offBase = emitTileOrPtrAddress(builder, loc, op.getOffsets(),
                                           offPtrType);
      Value srcBase = emitTileOrPtrAddress(builder, loc, op.getSrc(),
                                           dstPtrType);

      auto emitGatherb = [&](Value dst, Value off, int64_t repStride,
                             int64_t repeat) {
        builder.create<pto::UBVgatherbOp>(
            loc, dst, off, srcBase, i64c(repStride, loc, builder),
            i64c(1, loc, builder), i64c(repeat, loc, builder));
      };

      emitGatherBlockHead(loc, builder, *t, dstBase, offBase, dstPtrType,
                          offPtrType, emitGatherb);
      emitGatherBlockTail(loc, builder, *t, dstBase, offBase, dstPtrType,
                          offPtrType, emitGatherb);

      op.erase();
    }
    return success();
  }

  // GatherBlockHead: process full epr-element blocks.
  template <typename EmitFn>
  void emitGatherBlockHead(Location loc, OpBuilder &builder,
                           const GatherBlockTiling &t, Value dstBase,
                           Value offBase, pto::PtrType dstPtrType,
                           pto::PtrType offPtrType, EmitFn &&emitGatherb) {
    constexpr int64_t REPEAT_MAX = kRepeatMax;
    constexpr int64_t ADDRS_PER_REPEAT = 8;
    if (t.numRepeatPerLine <= 0) {
      return;
    }
    int64_t numLoop = t.numRepeatPerLine / REPEAT_MAX;
    int64_t remainAfterLoop = t.numRepeatPerLine % REPEAT_MAX;

    for (int64_t i = 0; i < t.validRow; ++i) {
      int64_t rowOff = i * t.dstRowStride;
      int64_t offRowOff = i * t.offsetRowStride;

      for (int64_t j = 0; j < numLoop; ++j) {
        int64_t elemOff = rowOff + j * t.elementsPerRepeat * REPEAT_MAX;
        int64_t offOff = offRowOff + j * ADDRS_PER_REPEAT * REPEAT_MAX;
        Value dstAdv = addPtr(loc, builder, dstBase, dstPtrType,
                              idxc(elemOff, loc, builder));
        Value offAdv = addPtr(loc, builder, offBase, offPtrType,
                              idxc(offOff, loc, builder));
        emitGatherb(dstAdv, offAdv, ADDRS_PER_REPEAT, REPEAT_MAX);
      }
      if (remainAfterLoop > 0) {
        int64_t elemOff = rowOff + numLoop * t.elementsPerRepeat * REPEAT_MAX;
        int64_t offOff =
            offRowOff + numLoop * ADDRS_PER_REPEAT * REPEAT_MAX;
        Value dstAdv = addPtr(loc, builder, dstBase, dstPtrType,
                              idxc(elemOff, loc, builder));
        Value offAdv = addPtr(loc, builder, offBase, offPtrType,
                              idxc(offOff, loc, builder));
        emitGatherb(dstAdv, offAdv, ADDRS_PER_REPEAT, remainAfterLoop);
      }
    }
  }

  // GatherBlockTail: process remaining elements.
  template <typename EmitFn>
  void emitGatherBlockTail(Location loc, OpBuilder &builder,
                           const GatherBlockTiling &t, Value dstBase,
                           Value offBase, pto::PtrType dstPtrType,
                           pto::PtrType offPtrType, EmitFn &&emitGatherb) {
    if (t.numRemainPerLine <= 0) {
      return;
    }
    int64_t tailElemOff = t.numRepeatPerLine * t.elementsPerRepeat;
    int64_t tailRepStride = t.dstRowStride / t.blockSizeElem;
    if (tailRepStride == 0) {
      tailRepStride = 1;
    }

    for (int64_t i = 0; i < t.validRow; ++i) {
      int64_t elemOff = i * t.dstRowStride + tailElemOff;
      int64_t offOff = i * t.offsetRowStride + tailElemOff / t.blockSizeElem;
      Value dstAdv = addPtr(loc, builder, dstBase, dstPtrType,
                            idxc(elemOff, loc, builder));
      Value offAdv = addPtr(loc, builder, offBase, offPtrType,
                            idxc(offOff, loc, builder));
      emitGatherb(dstAdv, offAdv, tailRepStride, 1);
    }
  }

  // tgather (index form) → ub.vmuls + ub.vgather.
  // Decomposes element-index gather into byte offsets (vmuls), then passes
  // the source tile address as vgather's offsetAddr config. This mirrors
  // pto-isa a2a3/TGather.hpp.
  // Emits the per-(row, epr-chunk) sequence for one index-form gather:
  // count-mode mask, tmp = indices * elemSize (byte offsets), barrier, then
  // vgather from the source base address. Restores the normal mask
  // afterwards. Mirrors pto-isa a2a3/TGather.hpp.
  void emitGatherIndexSequence(Location loc, OpBuilder &builder,
                               MLIRContext *ctx, Value indicesPtr,
                               Value tmpPtr, Value dstPtr,
                               pto::PtrType i32PtrType,
                               pto::PtrType dstPtrType, Value srcAddr,
                               const TileShapeInfo &dstInfo,
                               const TileShapeInfo &indexInfo,
                               const TileShapeInfo &tmpInfo,
                               unsigned elemSize) {
    auto pipeV = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
    int64_t epr = dstInfo.elementsPerRepeat;
    for (int64_t i = 0; i < dstInfo.vRows; ++i) {
      for (int64_t col = 0; col < dstInfo.vCols; col += epr) {
        int64_t chunkElems =
            std::min<int64_t>(epr, dstInfo.vCols - col);
        Value tmpOff =
            idxc(i * tmpInfo.cols + col, loc, builder);
        Value indexOff =
            idxc(i * indexInfo.cols + col, loc, builder);
        Value dstOff = idxc(i * dstInfo.cols + col, loc, builder);
        Value tmpRow = addPtr(loc, builder, tmpPtr, i32PtrType, tmpOff);
        Value idxRow =
            addPtr(loc, builder, indicesPtr, i32PtrType, indexOff);
        Value dstRow = addPtr(loc, builder, dstPtr, dstPtrType, dstOff);

        builder.create<pto::UBSetMaskCountOp>(loc);
        builder.create<pto::UBSetMaskOp>(loc,
                                         i64c(chunkElems, loc, builder),
                                         i64c0(loc, builder));
        // tmp = indices * elemSize (byte offsets). Keep count-mode mask
        // active for vgather, matching pto-isa a2a3/TGather.hpp.
        builder.create<pto::UBVmulSOp>(
            loc, tmpRow, idxRow, i64c(elemSize, loc, builder),
            i64c1(loc, builder), i64c1(loc, builder), i64c1(loc, builder),
            i64c8(loc, builder), i64c8(loc, builder));
        builder.create<pto::BarrierOp>(loc, pipeV);
        builder.create<pto::UBVgatherOp>(
            loc, dstRow, tmpRow, srcAddr,
            i64c(mlir::pto::kValue8, loc, builder),
            i64c1(loc, builder));
      }
    }
    builder.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, builder);
  }

  // Resolves one index-form gather's source base address (i64) from the
  // CastPtrOp that defines the src tile pointer. Only handles
  // alloc_tile-backed src.
  FailureOr<Value> getGatherIndexSrcAddr(pto::TGatherOp op) {
    auto srcCast = op.getSrc().getDefiningOp<pto::CastPtrOp>();
    if (!srcCast) {
      op.emitOpError(
          "requires an alloc_tile-backed src with a planned address");
      signalPassFailure();
      return failure();
    }
    return srcCast.getInput();
  }

  // Materializes the i32 index/tmp and typed dst pointers for one
  // index-form gather. Returns the element size, or nullopt when the dst
  // element type has no representable size.
  std::optional<unsigned>
  emitGatherIndexPointers(pto::TGatherOp op, MLIRContext *ctx,
                          OpBuilder &builder, pto::PtrType &i32PtrType,
                          pto::PtrType &dstPtrType, Value &indicesPtr,
                          Value &tmpPtr, Value &dstPtr) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    i32PtrType = getUBPtrType(ctx, builder.getI32Type());
    indicesPtr =
        emitTileOrPtrAddress(builder, loc, op.getIndices(), i32PtrType);
    tmpPtr = emitTileOrPtrAddress(builder, loc, op.getTmp(), i32PtrType);

    Type dstElemTy = getStoredElemType(op.getDst().getType());
    if (!dstElemTy) {
      return std::nullopt;
    }
    unsigned elemSize = getElementSize(dstElemTy);
    if (elemSize == 0) {
      return std::nullopt;
    }
    dstPtrType = getUBPtrType(ctx, dstElemTy);
    dstPtr = emitTileOrPtrAddress(builder, loc, op.getDst(), dstPtrType);
    return elemSize;
  }

  LogicalResult lowerTGatherIndexOps(func::FuncOp func, MLIRContext *ctx,
                                     OpBuilder &builder,
                                     const TileShapeMap &tileShapes) {
    SmallVector<pto::TGatherOp> ops;
    func.walk([&](pto::TGatherOp op) { ops.push_back(op); });
    for (auto op : ops) {
      if (!op.hasIndexForm()) {
        continue;
      }
      auto dstInfo = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
      auto indexInfo =
          extractTileShapeInfoFromValue(op.getIndices(), tileShapes);
      auto tmpInfo = extractTileShapeInfoFromValue(op.getTmp(), tileShapes);
      if (!dstInfo || !indexInfo || !tmpInfo) {
        continue;
      }
      if (dstInfo->elementsPerRepeat == 0) {
        continue;
      }

      FailureOr<Value> srcAddr = getGatherIndexSrcAddr(op);
      if (failed(srcAddr)) {
        return failure();
      }

      pto::PtrType i32PtrType;
      pto::PtrType dstPtrType;
      Value indicesPtr;
      Value tmpPtr;
      Value dstPtr;
      auto elemSize = emitGatherIndexPointers(op, ctx, builder, i32PtrType,
                                              dstPtrType, indicesPtr, tmpPtr,
                                              dstPtr);
      if (!elemSize) {
        continue;
      }

      emitGatherIndexSequence(op.getLoc(), builder, ctx, indicesPtr, tmpPtr,
                              dstPtr, i32PtrType, dstPtrType, *srcAddr,
                              *dstInfo, *indexInfo, *tmpInfo, *elemSize);
      op.erase();
    }
    return success();
  }

  // Erase dead view/cast ops left behind after the tile lowerings.
  void cleanupDeadPTOOps(func::FuncOp func) {
    SmallVector<Operation *> toErase;
    func.walk([&](Operation *op) {
      if (isa<pto::PartitionViewOp, pto::MakeTensorViewOp,
              memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp>(op)) {
        toErase.push_back(op);
      }
    });
    for (auto *op : llvm::reverse(toErase)) {
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

private:
  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  Value i64c(int64_t val, Location loc, OpBuilder &b) {
    return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(val));
  }
  Value idxc(int64_t val, Location loc, OpBuilder &b) {
    return b.create<arith::ConstantOp>(
               loc, b.getIntegerAttr(b.getIndexType(), val))
        .getResult();
  }
  Value i64c0(Location loc, OpBuilder &b) { return i64c(0, loc, b); }
  Value i64c1(Location loc, OpBuilder &b) { return i64c(1, loc, b); }
  Value i64cM1(Location loc, OpBuilder &b) { return i64c(-1, loc, b); }
  Value i64c8(Location loc, OpBuilder &b) { return i64c(kDefaultRepeatStride, loc, b); }
  Value idxc0(Location loc, OpBuilder &b) { return idxc(0, loc, b); }
  Value idxc1(Location loc, OpBuilder &b) { return idxc(1, loc, b); }

  template <typename UBop>
  void emitUBBinOp(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   Value repeat, Value repStride) {
    b.create<UBop>(loc, dst, s0, s1, repeat,
                   i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                   repStride, repStride, i64c0(loc, b));
  }

  // Shared lowering prologue: compute the element pointer type of the
  // destination tile and materialize address values for the destination
  // followed by each tile operand (a raw pto.ptr passes through, a tile_buf
  // gets a TileBufAddrOp). The destination address is always ptrs.front().
  SmallVector<Value> lowerTilePtrs(OpBuilder &builder, MLIRContext *ctx,
                                   Operation *op, Value dstVal,
                                   ArrayRef<Value> tiles) {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);
    Type elemTy = getStoredElemType(dstVal.getType());
    auto ptrType = getUBPtrType(ctx, elemTy);

    auto emitAddr = [&](Value tile) -> Value {
      if (isa<pto::PtrType>(tile.getType())) {
        return tile;
      }
      auto addrOp = builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
      return addrOp.getDst();
    };

    SmallVector<Value> ptrs;
    ptrs.reserve(tiles.size() + 1);
    ptrs.push_back(emitAddr(dstVal));
    for (Value tile : tiles) {
      ptrs.push_back(emitAddr(tile));
    }
    return ptrs;
  }

  // Lowering prologue shared by every tile op family: materialize the
  // destination pointer followed by one pointer per source tile. Returns
  // the pointers (ptrs[0] is the destination) plus their element type.
  std::pair<SmallVector<Value>, pto::PtrType>
  lowerOpPtrs(OpBuilder &builder, MLIRContext *ctx, Operation *op,
              Value dstVal, ArrayRef<Value> srcs) {
    Type elemTy = getStoredElemType(dstVal.getType());
    auto ptrType = getUBPtrType(ctx, elemTy);
    SmallVector<Value> ptrs = lowerTilePtrs(builder, ctx, op, dstVal, srcs);
    return {std::move(ptrs), ptrType};
  }

  Value convertScalarToI64(OpBuilder &builder, Location loc, Value scalar) {
    if (scalar.getType().isF32() || scalar.getType().isF16()) {
      unsigned width = scalar.getType().isF32() ? 32 : 16;
      auto intTy = builder.getIntegerType(width);
      Value asInt = builder.create<arith::BitcastOp>(loc, intTy, scalar);
      return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), asInt);
    }
    if (scalar.getType().isInteger(mlir::pto::kValue64)) {
      return scalar;
    }
    return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), scalar);
  }

  // Row-splitting prologue shared by the tile dispatchers: when the valid
  // region is not contiguous across rows (multiple valid rows and vCols !=
  // cols), recurse per row with a single-row shape. Returns true if the
  // region was split (the caller must not emit anything else).
  template <typename Fn>
  bool splitIntoRows(Location loc, OpBuilder &b, Value dst, Value src,
                     pto::PtrType ptrTy, const TileShapeInfo &info,
                     Fn &&recurse) {
    if (!(info.vRows > 1 && info.vCols != info.cols)) {
      return false;
    }
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value off = b.create<arith::MulIOp>(
        loc, forOp.getInductionVar(), idxc(info.cols, loc, b));
    TileShapeInfo rowInfo = info;
    rowInfo.rows = 1;
    rowInfo.vRows = 1;
    recurse(addPtr(loc, b, dst, ptrTy, off),
            src ? addPtr(loc, b, src, ptrTy, off) : Value{}, ptrTy, rowInfo);
    b.setInsertionPointAfter(forOp);
    return true;
  }

  // Emit an elementwise op over a possibly non-repeat-aligned region using
  // count-mode masks: full repeats in a loop, a masked tail, or a single
  // masked op when the whole region fits into one repeat. `src` may be null
  // for dst-only ops (vdup).
  template <typename EmitFn>
  void emitMaskedRepeats(Location loc, OpBuilder &b, Value dst, Value src,
                         pto::PtrType ptrTy, const TileShapeInfo &info,
                         EmitFn &&emit) {
    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 1 || tailElements > 0) {
      if (headRepeats > 0) {
        auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                          idxc(headRepeats, loc, b), idxc1(loc, b));
        b.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value off = b.create<arith::MulIOp>(loc, iv, idxc(epr, loc, b)).getResult();
        Value rd = addPtr(loc, b, dst, ptrTy, off);
        Value rs = src ? addPtr(loc, b, src, ptrTy, off) : Value{};
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(epr, loc, b), i64c0(loc, b));
        emit(rd, rs);
        b.create<pto::UBSetMaskNormOp>(loc);
        b.setInsertionPointAfter(forOp);
      }
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, loc, b);
        Value td = addPtr(loc, b, dst, ptrTy, offT);
        Value ts = src ? addPtr(loc, b, src, ptrTy, offT) : Value{};
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(tailElements, loc, b), i64c0(loc, b));
        emit(td, ts);
        b.create<pto::UBSetMaskNormOp>(loc);
      }
      fullMask(loc, b);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emit(dst, src);
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  template <typename UBop>
  void dispatchShift(Location loc, OpBuilder &b, Value dst, Value src,
                     Value scalar, pto::PtrType ptrTy,
                     const TileShapeInfo &info) {
    if (splitIntoRows(loc, b, dst, src, ptrTy, info,
                      [&](Value rd, Value rs, pto::PtrType pty,
                          const TileShapeInfo &rowInfo) {
                        dispatchShift<UBop>(loc, b, rd, rs, scalar, pty,
                                            rowInfo);
                      })) {
      return;
    }
    auto emitShift = [this, loc, &b, scalar](Value d, Value s) {
      Value scalarI64 = scalar;
      if (scalarI64.getType() != b.getI64Type()) {
        scalarI64 = b.create<arith::ExtSIOp>(
            loc, b.getI64Type(), scalar);
      }
      b.create<UBop>(loc, d, s, scalarI64,
                     i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                     i64c8(loc, b), i64c8(loc, b));
    };
    emitMaskedRepeats(loc, b, dst, src, ptrTy, info, emitShift);
  }

  template <typename UBop>
  void dispatchUnary(Location loc, OpBuilder &b, Value dst, Value src,
                     pto::PtrType ptrTy, const TileShapeInfo &info) {
    if (splitIntoRows(loc, b, dst, src, ptrTy, info,
                      [&](Value rd, Value rs, pto::PtrType pty,
                          const TileShapeInfo &rowInfo) {
                        dispatchUnary<UBop>(loc, b, rd, rs, pty, rowInfo);
                      })) {
      return;
    }
    auto emit = [this, loc, &b](Value rd, Value rs) {
      b.create<UBop>(loc, rd, rs,
                     i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                     i64c8(loc, b), i64c8(loc, b));
    };
    emitMaskedRepeats(loc, b, dst, src, ptrTy, info, emit);
  }

  void dispatchDup(Location loc, OpBuilder &b, Value dst, Value scalar,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    if (splitIntoRows(loc, b, dst, Value{}, ptrTy, info,
                      [&](Value rd, Value, pto::PtrType pty,
                          const TileShapeInfo &rowInfo) {
                        dispatchDup(loc, b, rd, scalar, pty, rowInfo);
                      })) {
      return;
    }
    auto emit = [this, loc, &b, scalar](Value rd, Value) {
      Value scalarI64 = scalar;
      if (scalarI64.getType() != b.getI64Type()) {
        scalarI64 = b.create<arith::ExtSIOp>(loc, b.getI64Type(), scalar);
      }
      b.create<pto::UBVdupOp>(loc, rd, scalarI64,
                              i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                              i64c8(loc, b), i64c0(loc, b));
    };
    emitMaskedRepeats(loc, b, dst, Value{}, ptrTy, info, emit);
  }

  template <typename UBop>
  void modeNorm1L(const TileOpContext &c, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 1 || tailElements > 0) {
      if (headRepeats > 0) {
        auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                          idxc(headRepeats, c.loc, c.b), idxc1(c.loc, c.b));
        c.b.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value off = c.b.create<arith::MulIOp>(c.loc, iv, idxc(epr, c.loc, c.b)).getResult();
        Value rd = addPtr(c.loc, c.b, c.dst, c.ptrTy, off);
        Value r0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, off);
        Value r1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, off);
        c.b.create<pto::UBSetMaskCountOp>(c.loc);
        c.b.create<pto::UBSetMaskOp>(c.loc, i64c(epr, c.loc, c.b), i64c0(c.loc, c.b));
        emitUBBinOp<UBop>(c.loc, c.b, rd, r0, r1, i64c1(c.loc, c.b), i64c8(c.loc, c.b));
        c.b.create<pto::UBSetMaskNormOp>(c.loc);
        c.b.setInsertionPointAfter(forOp);
      }
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, c.loc, c.b);
        Value td = addPtr(c.loc, c.b, c.dst, c.ptrTy, offT);
        Value ts0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, offT);
        Value ts1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, offT);
        c.b.create<pto::UBSetMaskCountOp>(c.loc);
        c.b.create<pto::UBSetMaskOp>(c.loc, i64c(tailElements, c.loc, c.b), i64c0(c.loc, c.b));
        emitUBBinOp<UBop>(c.loc, c.b, td, ts0, ts1, i64c1(c.loc, c.b), i64c8(c.loc, c.b));
        c.b.create<pto::UBSetMaskNormOp>(c.loc);
      }
      fullMask(c.loc, c.b);
      return;
    }

    c.b.create<pto::UBSetMaskCountOp>(c.loc);
    c.b.create<pto::UBSetMaskOp>(c.loc, i64c(totalV, c.loc, c.b), i64c0(c.loc, c.b));
    emitUBBinOp<UBop>(c.loc, c.b, c.dst, c.s0, c.s1, i64c1(c.loc, c.b), i64c8(c.loc, c.b));
    c.b.create<pto::UBSetMaskNormOp>(c.loc);
    fullMask(c.loc, c.b);
  }

  void setMask(Location loc, OpBuilder &b, unsigned n) {
    auto [m0, m1] = computeContMaskValues(n);
    b.create<pto::UBSetMaskOp>(loc, i64c(m0, loc, b), i64c(m1, loc, b));
  }

  void fullMask(Location loc, OpBuilder &b) {
    b.create<pto::UBSetMaskOp>(loc, i64cM1(loc, b), i64cM1(loc, b));
  }

  Value addPtr(Location loc, OpBuilder &b, Value base, pto::PtrType ptrTy,
                Value off) {
    return b.create<pto::AddPtrOp>(loc, ptrTy, base, off);
  }

  //===--------------------------------------------------------------------===//
  // tload → mte_gm_ub / tstore → mte_ub_gm
  //===--------------------------------------------------------------------===//

  struct DmaViewInfo {
    Value gmPtr;
    Value linearOffset;
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    SmallVector<Value> offsets;
  };

  static bool hasUnitInnermostStride(Value view) {
    while (Operation *def = view.getDefiningOp()) {
      if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
        auto strides = subview.getStaticStrides();
        if (strides.empty() || strides.back() != 1) {
          return false;
        }
        view = subview.getSource();
        continue;
      }
      if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
        auto strides = reinterpret.getConstifiedMixedStrides();
        if (strides.empty()) {
          return false;
        }
        auto stride = getConstantIntValue(strides.back());
        return stride && *stride == 1;
      }
      if (auto cast = dyn_cast<memref::CastOp>(def)) {
        view = cast.getSource();
        continue;
      }
      break;
    }
    auto memTy = dyn_cast<MemRefType>(view.getType());
    if (!memTy) {
      return false;
    }
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(pto::getPTOMemRefStridesAndOffset(memTy, strides, offset))) {
      return false;
    }
    return !strides.empty() && strides.back() == 1;
  }

  static FailureOr<DmaViewInfo>
  extractDirectDmaViewInfo(Value view, Operation *consumer) {
    SmallVector<pto::PartitionViewOp> partitions;
    Value root = view;
    while (auto partition = root.getDefiningOp<pto::PartitionViewOp>()) {
      partitions.push_back(partition);
      root = partition.getSource();
    }
    if (partitions.empty()) {
      return failure();
    }

    auto makeView = root.getDefiningOp<pto::MakeTensorViewOp>();
    if (!makeView) {
      return failure();
    }

    DmaViewInfo info;
    info.gmPtr = makeView.getPtr();
    info.sizes.assign(partitions.front().getSizes().begin(),
                      partitions.front().getSizes().end());
    info.strides.assign(makeView.getStrides().begin(),
                        makeView.getStrides().end());
    if (info.strides.empty() ||
        !matchPattern(info.strides.back(), m_One())) {
      consumer->emitError(
          "A2/A3 DMA lowering requires a unit innermost stride");
      return failure();
    }

    info.offsets.assign(partitions.front().getOffsets().begin(),
                        partitions.front().getOffsets().end());
    OpBuilder builder(consumer);
    for (pto::PartitionViewOp partition :
         llvm::drop_begin(partitions)) {
      if (partition.getOffsets().size() != info.offsets.size() ||
          partition.getSizes().size() != info.sizes.size()) {
        consumer->emitError(
            "A2/A3 DMA lowering requires rank-preserving nested partitions");
        return failure();
      }
      for (auto [index, offset] :
           llvm::enumerate(partition.getOffsets())) {
        info.offsets[index] = builder.create<arith::AddIOp>(
            consumer->getLoc(), info.offsets[index], offset);
      }
    }
    return info;
  }

  static FailureOr<DmaViewInfo> extractDmaViewInfo(pto::TLoadOp op) {
    if (op.getSrc().getDefiningOp<pto::PartitionViewOp>()) {
      return extractDirectDmaViewInfo(op.getSrc(), op.getOperation());
    }
    return extractDmaMemRefViewInfo(op.getLoc(), op.getSrc(), op.getContext());
  }

  static FailureOr<DmaViewInfo> extractDmaViewInfo(pto::TStoreOp op) {
    if (op.getDst().getDefiningOp<pto::PartitionViewOp>()) {
      return extractDirectDmaViewInfo(op.getDst(), op.getOperation());
    }
    return extractDmaMemRefViewInfo(op.getLoc(), op.getDst(), op.getContext());
  }

  static FailureOr<DmaViewInfo> extractDmaMemRefViewInfo(Location loc, Value view,
                                                         MLIRContext *ctx) {
    auto memTy = dyn_cast<MemRefType>(view.getType());
    if (!memTy) {
      return failure();
    }
    auto msAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memTy.getMemorySpace());
    if (!msAttr || msAttr.getAddressSpace() != pto::AddressSpace::GM) {
      return failure();
    }
    ArrayRef<int64_t> shape = memTy.getShape();
    if (shape.size() < mlir::pto::kValue2) {
      return failure();
    }
    if (!hasUnitInnermostStride(view)) {
      emitError(loc) << "A2/A3 DMA lowering requires a unit innermost stride";
      return failure();
    }

    OpBuilder b(ctx);
    b.setInsertionPointAfterValue(view);
    DmaViewInfo info;
    auto ptrTy = pto::PtrType::get(ctx, memTy.getElementType(), msAttr);
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, view);
    info.gmPtr = b.create<pto::CastPtrOp>(loc, ptrTy, traceRootMemRef(view));
    info.linearOffset = metadata.getOffset();
    info.sizes.assign(metadata.getSizes().begin(), metadata.getSizes().end());
    info.strides.assign(metadata.getStrides().begin(),
                        metadata.getStrides().end());
    return info;
  }

  static Value traceRootMemRef(Value value) {
    while (Operation *def = value.getDefiningOp()) {
      if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
        value = subview.getSource();
        continue;
      }
      if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
        value = reinterpret.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(def)) {
        value = cast.getSource();
        continue;
      }
      break;
    }
    return value;
  }

  Value computeGMByteOffset(Location loc, OpBuilder &b,
                            const DmaViewInfo &viewInfo, unsigned elemSize) {
    if (viewInfo.linearOffset) {
      Value offset = b.create<arith::IndexCastOp>(
          loc, b.getI64Type(), viewInfo.linearOffset);
      return b.create<arith::MulIOp>(loc, offset,
                                     i64c(elemSize, loc, b));
    }
    Value totalOff = idxc0(loc, b);
    for (size_t i = 0; i < viewInfo.offsets.size() &&
                        i < viewInfo.strides.size(); ++i) {
      APInt constOff;
      if (matchPattern(viewInfo.offsets[i], m_ConstantInt(&constOff)) &&
          constOff.isZero()) {
        continue;
      }
      Value dimOff = b.create<arith::MulIOp>(loc, viewInfo.offsets[i],
                                             viewInfo.strides[i]).getResult();
      totalOff = b.create<arith::AddIOp>(loc, totalOff, dimOff).getResult();
    }
    if (elemSize > 1) {
      totalOff = b.create<arith::MulIOp>(loc, totalOff,
                                         idxc(elemSize, loc, b)).getResult();
    }
    return totalOff;
  }

  Value offsetGMPtrByBytes(Location loc, OpBuilder &b, Value gmPtr,
                           Value byteOff) {
    APInt constOff;
    if (matchPattern(byteOff, m_ConstantInt(&constOff)) && constOff.isZero()) {
      return gmPtr;
    }
    auto origPtrTy = cast<pto::PtrType>(gmPtr.getType());
    auto bytePtrTy = pto::PtrType::get(b.getContext(), b.getI8Type(),
                                       origPtrTy.getMemorySpace());
    Value bytePtr = b.create<pto::CastPtrOp>(loc, bytePtrTy, gmPtr);
    Value offIdx = byteOff;
    if (!offIdx.getType().isIndex()) {
      offIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), byteOff)
                   .getResult();
    }
    Value offsetBytePtr =
        b.create<pto::AddPtrOp>(loc, bytePtrTy, bytePtr, offIdx);
    return b.create<pto::CastPtrOp>(loc, origPtrTy, offsetBytePtr);
  }

  Value i64Cast(Location loc, OpBuilder &b, Value indexVal) {
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), indexVal)
        .getResult();
  }

  // i64 byte-stride of a GM view dimension (index-typed stride x elemSize).
  Value strideBytes(Location loc, OpBuilder &b, Value idxStride,
                    unsigned elemSize) {
    return b.create<arith::MulIOp>(loc, i64Cast(loc, b, idxStride),
                                   i64c(elemSize, loc, b)).getResult();
  }

  // i64 byte-length of the innermost (contiguous) burst dimension.
  Value burstBytes(Location loc, OpBuilder &b, Value lenBurstElts,
                   unsigned elemSize) {
    return b.create<arith::MulIOp>(loc, i64Cast(loc, b, lenBurstElts),
                                   i64c(elemSize, loc, b)).getResult();
  }

  // Product (in elements) of all dims below `i`, used as the UB-side stride
  // of loop level `i` (the UB tile is dense).
  Value innerDimElems(Location loc, OpBuilder &b, const DmaViewInfo &viewInfo,
                      int i) {
    Value innerElems = i64c1(loc, b);
    for (int j = i + 1; j < static_cast<int>(viewInfo.sizes.size()); ++j) {
      innerElems = b.create<arith::MulIOp>(loc, innerElems,
          i64Cast(loc, b, viewInfo.sizes[j])).getResult();
    }
    return innerElems;
  }

  // Shared burst-transfer geometry validated by both GM<->UB emitters:
  // UB tile row span in bytes plus the view's rank, with the same shape and
  // rank guards the emitters historically applied inline.
  struct BurstGeometry {
    int64_t ubCols;
    unsigned elemSize;
    unsigned nd;
  };

  static std::optional<BurstGeometry>
  getBurstGeometry(const DmaViewInfo &viewInfo, Type elemTy,
                   ArrayRef<int64_t> tileShape) {
    if (tileShape.size() < mlir::pto::kValue2) {
      return std::nullopt;
    }
    BurstGeometry g;
    g.ubCols = tileShape[1];
    g.elemSize = getElementSize(elemTy);
    g.nd = viewInfo.sizes.size();
    if (g.nd < mlir::pto::kValue2) {
      return std::nullopt;
    }
    return g;
  }

  // The innermost-burst operands shared by both GM<->UB directions: burst
  // length and the dense UB row stride.
  struct BurstPlan {
    Value lenBurst;
    Value ubRowStride;
  };

  BurstPlan buildBurstPlan(Location loc, OpBuilder &b,
                           const DmaViewInfo &viewInfo,
                           const BurstGeometry &g) {
    // The burst length covers one innermost view dim; the UB row stride is
    // the full UB tile row.
    Value lenBurst = burstBytes(loc, b, viewInfo.sizes[g.nd - 1], g.elemSize);
    Value ubRowStride = b.create<arith::MulIOp>(loc, i64c(g.ubCols, loc, b),
                                                i64c(g.elemSize, loc, b)).getResult();
    return {lenBurst, ubRowStride};
  }

  LogicalResult emitMteGmUb(Location loc, OpBuilder &b, Value gmPtr,
                             Value ubPtr, const DmaViewInfo &viewInfo,
                             Type elemTy, ArrayRef<int64_t> tileShape) {
    auto geometry = getBurstGeometry(viewInfo, elemTy, tileShape);
    if (!geometry) {
      return failure();
    }
    BurstPlan plan = buildBurstPlan(loc, b, viewInfo, *geometry);

    // GM -> UB: the burst reads a contiguous GM run and writes a dense UB row.
    pto::DmaLoopConfig nburst{
        i64Cast(loc, b, viewInfo.sizes[geometry->nd - 2]),
        strideBytes(loc, b, viewInfo.strides[geometry->nd - 2],
                    geometry->elemSize),
        plan.ubRowStride};

    SmallVector<pto::DmaLoopConfig> loops;
    for (int i = geometry->nd - 3; i >= 0; --i) {
      Value count = i64Cast(loc, b, viewInfo.sizes[i]);
      Value srcStride = strideBytes(loc, b, viewInfo.strides[i],
                                    geometry->elemSize);
      Value dstStride = b.create<arith::MulIOp>(
          loc, innerDimElems(loc, b, viewInfo, i),
          i64c(geometry->elemSize, loc, b)).getResult();
      loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteGmUbOp>(loc, gmPtr, ubPtr, i64c0(loc, b),
                             plan.lenBurst, nburst,
                             llvm::ArrayRef(loops), std::nullopt);
    return success();
  }

  LogicalResult emitMteUbGm(Location loc, OpBuilder &b, Value ubPtr,
                             Value gmPtr, const DmaViewInfo &viewInfo,
                             Type elemTy, ArrayRef<int64_t> tileShape) {
    auto geometry = getBurstGeometry(viewInfo, elemTy, tileShape);
    if (!geometry) {
      return failure();
    }
    BurstPlan plan = buildBurstPlan(loc, b, viewInfo, *geometry);

    // UB -> GM: the burst reads a dense UB row and writes a strided GM run.
    pto::DmaLoopConfig nburst{
        i64Cast(loc, b, viewInfo.sizes[geometry->nd - 2]), plan.ubRowStride,
        strideBytes(loc, b, viewInfo.strides[geometry->nd - 2],
                    geometry->elemSize)};

    SmallVector<pto::DmaLoopConfig> loops;
    for (int i = geometry->nd - 3; i >= 0; --i) {
      Value count = i64Cast(loc, b, viewInfo.sizes[i]);
      Value srcStride = b.create<arith::MulIOp>(
          loc, innerDimElems(loc, b, viewInfo, i),
          i64c(geometry->elemSize, loc, b)).getResult();
      Value dstStride = strideBytes(loc, b, viewInfo.strides[i],
                                    geometry->elemSize);
      loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteUbGmOp>(loc, ubPtr, gmPtr, plan.lenBurst, nburst,
                             Value{}, llvm::ArrayRef(loops));
    return success();
  }

  // Shared prologue of lowerTLoad/lowerTStore: validates that the UB-side
  // tile is well-formed (memory space, element type/size, known tile shape)
  // and returns its element type and size. `tile` is the UB operand whose
  // shape drives the DMA.
  struct TLoadStorePrologue {
    Type elemTy;
    unsigned elemSize;
    llvm::ArrayRef<int64_t> shape;
  };

  std::optional<TLoadStorePrologue>
  checkLoadStoreTile(Value tile, const TileShapeMap &tileShapes) {
    Type tileType = tile.getType();
    if (!isUBMemorySpace(tileType)) {
      return std::nullopt;
    }
    Type elemTy = getStoredElemType(tileType);
    if (!elemTy) {
      return std::nullopt;
    }
    unsigned elemSize = getElementSize(elemTy);
    if (elemSize == 0) {
      return std::nullopt;
    }
    auto it = tileShapes.find(tile);
    if (it == tileShapes.end()) {
      return std::nullopt;
    }
    return TLoadStorePrologue{elemTy, elemSize,
                              llvm::ArrayRef(it->second.shape)};
  }

  LogicalResult lowerTLoad(pto::TLoadOp op, OpBuilder &b,
                           const TileShapeMap &tileShapes) {
    Location loc = op.getLoc();
    auto viewInfo = extractDmaViewInfo(op);
    if (failed(viewInfo)) {
      return failure();
    }
    auto prologue = checkLoadStoreTile(op.getDst(), tileShapes);
    if (!prologue) {
      return failure();
    }

    Value byteOff = computeGMByteOffset(loc, b, *viewInfo, prologue->elemSize);
    Value gmPtr = offsetGMPtrByBytes(loc, b, viewInfo->gmPtr, byteOff);
    return emitMteGmUb(loc, b, gmPtr, op.getDst(), *viewInfo,
                       prologue->elemTy, prologue->shape);
  }

  LogicalResult lowerTStore(pto::TStoreOp op, OpBuilder &b,
                            const TileShapeMap &tileShapes) {
    Location loc = op.getLoc();
    auto viewInfo = extractDmaViewInfo(op);
    if (failed(viewInfo)) {
      return failure();
    }
    auto prologue = checkLoadStoreTile(op.getSrc(), tileShapes);
    if (!prologue) {
      return failure();
    }

    Value byteOff = computeGMByteOffset(loc, b, *viewInfo, prologue->elemSize);
    Value gmPtr = offsetGMPtrByBytes(loc, b, viewInfo->gmPtr, byteOff);
    return emitMteUbGm(loc, b, op.getSrc(), gmPtr, *viewInfo,
                       prologue->elemTy, prologue->shape);
  }

  //===--------------------------------------------------------------------===//
  // CCE dispatch tree — mirrors TBinOp.hpp BinaryInstr
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void dispatch(const TileOpContext &c, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t cols = info.cols;
    int64_t rows = info.rows;
    int64_t vRows = info.vRows;
    int64_t vCols = info.vCols;

    // 1. Small tile
    if (rows <= kRepeatMax && cols < static_cast<int64_t>(epr)) {
      modeSmall<UBop>(c, info);
      return;
    }

    // 2. Continuous at compile time
    if (vCols == cols || vRows == 1) {
      int64_t totalV = vRows * vCols;
      int64_t totalRpts = (totalV + epr - 1) / epr;

      if (totalRpts > kRepeatMax) {
        modeNorm1L<UBop>(c, info);
      } else {
        modeNorm1L<UBop>(c, info);
      }
      return;
    }

    // 3. Non-continuous
    int64_t normColRepeat = cols / epr;
    if (normColRepeat > 1 && vRows * normColRepeat < kSmallRptBinOp) {
      modeCount2L<UBop>(c, info);
    } else if (vRows < normColRepeat + 1) {
      if (vCols % epr > 0) {
        modeCount2L<UBop>(c, info);
      } else {
        modeColVLAlign<UBop>(c, info);
      }
    } else {
      modeRowRpt<UBop>(c, info);
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin1LNormModeSmall
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeSmall(const TileOpContext &c, const TileShapeInfo &info) {
    int64_t rs = info.cols / static_cast<int64_t>(info.blockSizeElem);

    if (info.vRows > 1) {
      auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                         idxc(info.vRows, c.loc, c.b), idxc1(c.loc, c.b));
      c.b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value off = c.b.create<arith::MulIOp>(c.loc, iv, idxc(info.cols, c.loc, c.b)).getResult();
      Value rd = addPtr(c.loc, c.b, c.dst, c.ptrTy, off);
      Value r0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, off);
      Value r1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, off);
      c.b.create<pto::UBSetMaskCountOp>(c.loc);
      c.b.create<pto::UBSetMaskOp>(c.loc, i64c(info.vCols, c.loc, c.b), i64c0(c.loc, c.b));
      emitUBBinOp<UBop>(c.loc, c.b, rd, r0, r1, i64c1(c.loc, c.b), i64c(rs, c.loc, c.b));
      c.b.create<pto::UBSetMaskNormOp>(c.loc);
      c.b.setInsertionPointAfter(forOp);
      fullMask(c.loc, c.b);
      return;
    }

    c.b.create<pto::UBSetMaskCountOp>(c.loc);
    c.b.create<pto::UBSetMaskOp>(c.loc, i64c(info.vCols, c.loc, c.b), i64c0(c.loc, c.b));
    emitUBBinOp<UBop>(c.loc, c.b, c.dst, c.s0, c.s1, i64c1(c.loc, c.b), i64c(rs, c.loc, c.b));
    c.b.create<pto::UBSetMaskNormOp>(c.loc);
    fullMask(c.loc, c.b);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeColVLAlign
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeColVLAlign(const TileOpContext &c, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t headRepeats = info.vCols / epr;
    int64_t rowStride = info.cols;

    if (headRepeats > kRepeatMax) {
      modeCount2L<UBop>(c, info);
      return;
    }

    auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                      idxc(info.vRows, c.loc, c.b), idxc1(c.loc, c.b));
    c.b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = c.b.create<arith::MulIOp>(c.loc, iv, idxc(rowStride, c.loc, c.b))
                    .getResult();
    Value rd = addPtr(c.loc, c.b, c.dst, c.ptrTy, off);
    Value rs0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, off);
    Value rs1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, off);
    emitUBBinOp<UBop>(c.loc, c.b, rd, rs0, rs1, i64c(headRepeats, c.loc, c.b), i64c8(c.loc, c.b));
    c.b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LCountMode – row-by-row count mode
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeCount2L(const TileOpContext &c, const TileShapeInfo &info) {
    int64_t rowStride = info.cols;

    auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                      idxc(info.vRows, c.loc, c.b), idxc1(c.loc, c.b));
    c.b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = c.b.create<arith::MulIOp>(c.loc, iv, idxc(rowStride, c.loc, c.b))
                    .getResult();
    Value rd = addPtr(c.loc, c.b, c.dst, c.ptrTy, off);
    Value rs0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, off);
    Value rs1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, off);
    TileShapeInfo rowInfo = info;
    rowInfo.rows = 1;
    rowInfo.vRows = 1;
    TileOpContext row{c.loc, c.b, rd, rs0, rs1, c.ptrTy};
    modeNorm1L<UBop>(row, rowInfo);
    c.b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeRowRpt
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeRowRpt(const TileOpContext &c, const TileShapeInfo &info) {
    int64_t be = info.blockSizeElem;
    int64_t rowStride = info.cols;
    int64_t rs = rowStride / be;
    bool condRowRpt = (info.vRows <= kRepeatMax) && (rs <= kRepeatStrideMax);

    if (condRowRpt) {
      rowRptFast<UBop>(c, info, rs);
    } else {
      rowRptChunked<UBop>(c, info, rowStride, rs);
    }
  }

  template <typename UBop>
  void rowRptFast(const TileOpContext &c, const TileShapeInfo &info,
                  int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = info.vCols / epr;
    int64_t tailElements = info.vCols % epr;

    for (int64_t i = 0; i < numLoop; i++) {
      Value rd = addPtr(c.loc, c.b, c.dst, c.ptrTy, idxc(i * epr, c.loc, c.b));
      Value r0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, idxc(i * epr, c.loc, c.b));
      Value r1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, idxc(i * epr, c.loc, c.b));
      emitUBBinOp<UBop>(c.loc, c.b, rd, r0, r1, i64c(info.vRows, c.loc, c.b), i64c(rs, c.loc, c.b));
    }

    if (tailElements > 0) {
      Value off = idxc(numLoop * epr, c.loc, c.b);
      Value rd = addPtr(c.loc, c.b, c.dst, c.ptrTy, off);
      Value r0 = addPtr(c.loc, c.b, c.s0, c.ptrTy, off);
      Value r1 = addPtr(c.loc, c.b, c.s1, c.ptrTy, off);
      setMask(c.loc, c.b, tailElements);
      emitUBBinOp<UBop>(c.loc, c.b, rd, r0, r1, i64c(info.vRows, c.loc, c.b), i64c(rs, c.loc, c.b));
      fullMask(c.loc, c.b);
    }
  }

  template <typename UBop>
  void rowRptChunked(const TileOpContext &c, const TileShapeInfo &info,
                     int64_t rowStride, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t rptPerLine = info.vCols / epr;
    int64_t remainElem = info.vCols % epr;

    if (info.vRows > static_cast<int64_t>(epr)) {
      if (rptPerLine > 0) {
        headRows<UBop>(c, info, rowStride, rptPerLine);
      }
      if (remainElem > 0) {
        Value off = idxc(rptPerLine * epr, c.loc, c.b);
        TileOpContext tail{c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
                           addPtr(c.loc, c.b, c.s0, c.ptrTy, off),
                           addPtr(c.loc, c.b, c.s1, c.ptrTy, off), c.ptrTy};
        tailRows<UBop>(tail, info, rowStride, rs, remainElem);
      }
    } else {
      if (remainElem == 0) {
        headRows<UBop>(c, info, rowStride, info.vCols / epr);
      } else if (rptPerLine > 0) {
        headRows<UBop>(c, info, rowStride, rptPerLine);
        Value off = idxc(rptPerLine * epr, c.loc, c.b);
        TileOpContext tail{c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
                           addPtr(c.loc, c.b, c.s0, c.ptrTy, off),
                           addPtr(c.loc, c.b, c.s1, c.ptrTy, off), c.ptrTy};
        tailRows<UBop>(tail, info, rowStride, rs, remainElem);
      } else {
        tailRows<UBop>(c, info, rowStride, rs, remainElem);
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeHead – chunked per-row head
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void headRows(const TileOpContext &c, const TileShapeInfo &info,
                int64_t rowStride, int64_t rptPerLine) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = rptPerLine / kRepeatMax;
    int64_t remain = rptPerLine % kRepeatMax;
    int64_t chunkElems = kRepeatMax * epr;

    auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                      idxc(info.vRows, c.loc, c.b),
                                      idxc1(c.loc, c.b));
    c.b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value rowBase =
        c.b.create<arith::MulIOp>(c.loc, iv, idxc(rowStride, c.loc, c.b)).getResult();

    if (numLoop > 0) {
      auto inner = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                        idxc(numLoop, c.loc, c.b), idxc1(c.loc, c.b));
      c.b.setInsertionPointToStart(inner.getBody());
      Value jv = inner.getInductionVar();
      Value co = c.b.create<arith::MulIOp>(c.loc, jv, idxc(chunkElems, c.loc, c.b))
                     .getResult();
      Value off = c.b.create<arith::AddIOp>(c.loc, rowBase, co).getResult();
      emitUBBinOp<UBop>(c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
           addPtr(c.loc, c.b, c.s0, c.ptrTy, off), addPtr(c.loc, c.b, c.s1, c.ptrTy, off),
           i64c(kRepeatMax, c.loc, c.b), i64c8(c.loc, c.b));
      c.b.setInsertionPointAfter(inner);
    }

    if (remain > 0) {
      Value co = idxc(numLoop * chunkElems, c.loc, c.b);
      Value off = c.b.create<arith::AddIOp>(c.loc, rowBase, co).getResult();
      emitUBBinOp<UBop>(c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
           addPtr(c.loc, c.b, c.s0, c.ptrTy, off), addPtr(c.loc, c.b, c.s1, c.ptrTy, off),
           i64c(remain, c.loc, c.b), i64c8(c.loc, c.b));
    }
    c.b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeTail – masked per-row tail
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void tailRows(const TileOpContext &c, const TileShapeInfo &info,
                int64_t rowStride, int64_t rs, unsigned remainPerLine) {
    bool strideOver =
        (rowStride / info.blockSizeElem > kRepeatStrideMax);
    setMask(c.loc, c.b, remainPerLine);

    int64_t numLoop = 0;
    int64_t remainAfterLoop = info.vRows;
    if (info.vRows > kRepeatMax) {
      numLoop = info.vRows / kRepeatMax;
      remainAfterLoop = info.vRows % kRepeatMax;

      auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                        idxc(numLoop, c.loc, c.b), idxc1(c.loc, c.b));
      c.b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      if (strideOver) {
        tailStrideOverChunk<UBop>(c, iv, rowStride);
      } else {
        tailStrideOkChunk<UBop>(c, iv, rowStride, rs);
      }
      c.b.setInsertionPointAfter(forOp);
    }

    if (remainAfterLoop > 0) {
      if (strideOver) {
        tailStrideOverRemain<UBop>(c, rowStride, numLoop, remainAfterLoop);
      } else {
        tailStrideOkRemain<UBop>(c, rowStride, rs, numLoop, remainAfterLoop);
      }
    }

    fullMask(c.loc, c.b);
  }

  template <typename UBop>
  void tailStrideOverChunk(const TileOpContext &c, Value iv,
                           int64_t rowStride) {
    auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b),
                                      idxc(kRepeatMax, c.loc, c.b), idxc1(c.loc, c.b));
    c.b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = c.b.create<arith::MulIOp>(
        c.loc, iv, idxc(kRepeatMax * rowStride, c.loc, c.b)).getResult();
    Value rowOff =
        c.b.create<arith::MulIOp>(c.loc, jv, idxc(rowStride, c.loc, c.b)).getResult();
    Value off = c.b.create<arith::AddIOp>(c.loc, baseOff, rowOff).getResult();
    emitUBBinOp<UBop>(c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
         addPtr(c.loc, c.b, c.s0, c.ptrTy, off), addPtr(c.loc, c.b, c.s1, c.ptrTy, off),
         i64c1(c.loc, c.b), i64c1(c.loc, c.b));
    c.b.setInsertionPointAfter(forOp);
  }

  template <typename UBop>
  void tailStrideOkChunk(const TileOpContext &c, Value iv,
                         int64_t rowStride, int64_t rs) {
    Value off = c.b.create<arith::MulIOp>(
        c.loc, iv, idxc(kRepeatMax * rowStride, c.loc, c.b)).getResult();
    emitUBBinOp<UBop>(c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
         addPtr(c.loc, c.b, c.s0, c.ptrTy, off), addPtr(c.loc, c.b, c.s1, c.ptrTy, off),
         i64c(kRepeatMax, c.loc, c.b), i64c(rs, c.loc, c.b));
  }

  template <typename UBop>
  void tailStrideOverRemain(const TileOpContext &c, int64_t rowStride,
                            int64_t numLoop, int64_t remain) {
    auto forOp = c.b.create<scf::ForOp>(c.loc, idxc0(c.loc, c.b), idxc(remain, c.loc, c.b),
                                      idxc1(c.loc, c.b));
    c.b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = idxc(numLoop * kRepeatMax * rowStride, c.loc, c.b);
    Value rowOff =
        c.b.create<arith::MulIOp>(c.loc, jv, idxc(rowStride, c.loc, c.b)).getResult();
    Value off = c.b.create<arith::AddIOp>(c.loc, baseOff, rowOff).getResult();
    emitUBBinOp<UBop>(c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
         addPtr(c.loc, c.b, c.s0, c.ptrTy, off), addPtr(c.loc, c.b, c.s1, c.ptrTy, off),
         i64c1(c.loc, c.b), i64c1(c.loc, c.b));
    c.b.setInsertionPointAfter(forOp);
  }

  template <typename UBop>
  void tailStrideOkRemain(const TileOpContext &c, int64_t rowStride,
                          int64_t rs, int64_t numLoop, int64_t remain) {
    Value off = idxc(numLoop * kRepeatMax * rowStride, c.loc, c.b);
    emitUBBinOp<UBop>(c.loc, c.b, addPtr(c.loc, c.b, c.dst, c.ptrTy, off),
         addPtr(c.loc, c.b, c.s0, c.ptrTy, off), addPtr(c.loc, c.b, c.s1, c.ptrTy, off),
         i64c(remain, c.loc, c.b), i64c(rs, c.loc, c.b));
  }
};
} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createLowerPTOToUBufOpsPass() {
  return std::make_unique<LowerPTOToUBufOpsPass>();
}
} // namespace pto
} // namespace mlir
