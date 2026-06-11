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

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_LOWERPTOTOUBUFOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

static constexpr int64_t kRepeatMax = 255;
static constexpr int64_t kRepeatStrideMax = 255;
static constexpr int64_t kSmallRptBinOp = 4;
static constexpr int64_t kDefaultRepeatStride = 8;
static constexpr unsigned kMaskLen = 64;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static unsigned getElementSize(Type elemTy) {
  if (elemTy.isF16() || elemTy.isBF16())
    return 2;
  if (elemTy.isF32())
    return 4;
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    if (width == 16 || width == 32)
      return width / 8;
  }
  return 0;
}

static Type getStoredElemType(Type ty) {
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty))
    return tbTy.getElementType();
  if (auto mrTy = dyn_cast<MemRefType>(ty))
    return mrTy.getElementType();
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty))
    return ptrTy.getElementType();
  return Type();
}

/// Returns true if the type lives in UB (VEC) address space.
static std::optional<bool> isUBMemorySpaceImpl(Type ty) {
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    auto msAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
    if (!msAttr)
      return false;
    return msAttr.getAddressSpace() == pto::AddressSpace::VEC;
  }
  if (auto mrTy = dyn_cast<MemRefType>(ty)) {
    auto msAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace());
    if (!msAttr)
      return false;
    return msAttr.getAddressSpace() == pto::AddressSpace::VEC;
  }
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty))
    return ptrTy.getMemorySpace().getAddressSpace() == pto::AddressSpace::VEC;
  return std::nullopt;
}

/// Returns true if the given type is confirmed UB memory space.
static bool isUBMemorySpace(Type ty) {
  auto result = isUBMemorySpaceImpl(ty);
  return result.has_value() && result.value();
}

static bool isRowMajor(pto::TileBufType tbTy) {
  auto config = tbTy.getConfigAttr();
  if (!config)
    return true;
  return config.getBLayout().getValue() != pto::BLayout::ColMajor;
}

static pto::PtrType getUBPtrType(MLIRContext *ctx, Type elemTy) {
  auto msAttr = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
  return pto::PtrType::get(ctx, elemTy, msAttr);
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

static std::optional<TileShapeInfo> extractTileShapeInfo(
    Operation *op,
    const DenseMap<Value, SmallVector<int64_t, 2>> &tileShapes) {
  auto opDst = op->getOperand(2);
  Type dstTy = opDst.getType();
  if (!isUBMemorySpace(dstTy))
    return std::nullopt;

  Type elemTy;
  ArrayRef<int64_t> shape;
  ArrayRef<int64_t> validShape;
  if (auto tbTy = dyn_cast<pto::TileBufType>(dstTy)) {
    elemTy = tbTy.getElementType();
    shape = tbTy.getShape();
    validShape = tbTy.getValidShape();
    if (!isRowMajor(tbTy))
      return std::nullopt;
  } else if (auto mrTy = dyn_cast<MemRefType>(dstTy)) {
    elemTy = mrTy.getElementType();
    shape = mrTy.getShape();
  } else if (isa<pto::PtrType>(dstTy)) {
    auto it = tileShapes.find(opDst);
    if (it == tileShapes.end())
      return std::nullopt;
    elemTy = cast<pto::PtrType>(dstTy).getElementType();
    shape = llvm::ArrayRef(it->second);
  } else {
    return std::nullopt;
  }
  unsigned elemSize = getElementSize(elemTy);
  if (elemSize == 0)
    return std::nullopt;

  if (shape.size() < 2)
    return std::nullopt;

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  int64_t vRows = (!validShape.empty() &&
                   validShape[0] != ShapedType::kDynamic)
                      ? validShape[0] : rows;
  int64_t vCols = (validShape.size() >= 2 &&
                   validShape[1] != ShapedType::kDynamic)
                      ? validShape[1] : cols;
  if (vRows == ShapedType::kDynamic || vCols == ShapedType::kDynamic ||
      rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
    return std::nullopt;

  TileShapeInfo info;
  info.vRows = vRows;
  info.vCols = vCols;
  info.cols = cols;
  info.rows = rows;
  info.elemSize = elemSize;
  info.elementsPerRepeat = 256 / elemSize;
  info.blockSizeElem = 32 / elemSize;
  return info;
}

static bool canLower(Operation *op,
                     const DenseMap<Value, SmallVector<int64_t, 2>> &tileShapes) {
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
    if (func.isExternal())
      return;
    auto mod = func->getParentOfType<ModuleOp>();
    if (!mod)
      return;
    auto archAttr = mod->getAttrOfType<StringAttr>("pto.target_arch");
    if (!archAttr || archAttr.getValue() != "a3")
      return;

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // A3: sequential memory planning for alloc_tile buffers.  Must run first
    // so lowering sees PtrType from CastPtrOp instead of tile_buf.
    DenseMap<Value, SmallVector<int64_t, 2>> tileShapes;
    {
      SmallVector<pto::AllocTileOp> allocOps;
      func.walk([&](pto::AllocTileOp op) { allocOps.push_back(op); });
      uint64_t offset = 0;
      constexpr uint64_t ubBankAlignment = 65536;
      for (auto op : allocOps) {
        auto tbTy = cast<pto::TileBufType>(op.getResult().getType());
        auto shape = tbTy.getShape();
        uint64_t numElems = 1;
        for (auto d : shape)
          numElems *= d;
        uint64_t elemSize = getElementSize(tbTy.getElementType());
        uint64_t totalBytes = numElems * elemSize;
        builder.setInsertionPoint(op);
        auto addrConst = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getI64IntegerAttr(offset));
        auto ptrTy = pto::PtrType::get(
            ctx, tbTy.getElementType(),
            pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC));
        auto pc = builder.create<pto::CastPtrOp>(
            op.getLoc(), ptrTy, addrConst.getResult());
        tileShapes[pc.getResult()] = SmallVector<int64_t, 2>(shape);
        op.getResult().replaceAllUsesWith(pc.getResult());
        op.erase();
        uint64_t paddedBytes = (totalBytes + ubBankAlignment - 1) / ubBankAlignment * ubBankAlignment;
        offset += paddedBytes;
      }
    }

    // ---- tadd → pto.ub.vadd ----
    {
      SmallVector<pto::TAddOp> ops;
      func.walk([&](pto::TAddOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes))
          continue;
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info)
          continue;
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr)
          continue;
        dispatch<pto::UBVaddOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tsub → pto.ub.vsub ----
    {
      SmallVector<pto::TSubOp> ops;
      func.walk([&](pto::TSubOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes))
          continue;
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info)
          continue;
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr)
          continue;
        dispatch<pto::UBVsubOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tmul → pto.ub.vmul ----
    {
      SmallVector<pto::TMulOp> ops;
      func.walk([&](pto::TMulOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes))
          continue;
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info)
          continue;
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr)
          continue;
        dispatch<pto::UBVmulOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tdiv → pto.ub.vdiv ----
    {
      SmallVector<pto::TDivOp> ops;
      func.walk([&](pto::TDivOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes))
          continue;
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info)
          continue;
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr)
          continue;
        dispatch<pto::UBVdivOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tload → mte_gm_ub ----
    SmallVector<pto::TLoadOp> tloadOps;
    func.walk([&](pto::TLoadOp op) { tloadOps.push_back(op); });
    for (auto op : tloadOps) {
      builder.setInsertionPoint(op);
      if (succeeded(lowerTLoad(op, builder, tileShapes)))
        op.erase();
    }

    // ---- tstore → mte_ub_gm ----
    SmallVector<pto::TStoreOp> tstoreOps;
    func.walk([&](pto::TStoreOp op) { tstoreOps.push_back(op); });
    for (auto op : tstoreOps) {
      builder.setInsertionPoint(op);
      if (succeeded(lowerTStore(op, builder, tileShapes)))
        op.erase();
    }

    // ---- cleanup dead PTO ops ----
    SmallVector<Operation *> toErase;
    func.walk([&](Operation *op) {
      if (isa<pto::PartitionViewOp, pto::MakeTensorViewOp>(op))
        toErase.push_back(op);
    });
    for (auto *op : llvm::reverse(toErase)) {
      if (op->use_empty())
        op->erase();
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

  void vadd(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVaddOp>(loc, b, dst, s0, s1, repeat, repStride);
  }
  void vsub(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVsubOp>(loc, b, dst, s0, s1, repeat, repStride);
  }
  void vmul(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVmulOp>(loc, b, dst, s0, s1, repeat, repStride);
  }
  void vdiv(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVdivOp>(loc, b, dst, s0, s1, repeat, repStride);
  }

  std::tuple<Value, Value, Value, pto::PtrType>
  lowerBinaryOpCommon(OpBuilder &builder, MLIRContext *ctx, Operation *op,
                      Value dstVal, Value src0Val, Value src1Val,
                      const DenseMap<Value, SmallVector<int64_t, 2>> &tileShapes) {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);
    Type elemTy = getStoredElemType(dstVal.getType());
    auto ptrType = getUBPtrType(ctx, elemTy);

    auto emitAddr = [&](Value tile) -> Value {
      if (isa<pto::PtrType>(tile.getType()))
        return tile;
      auto addrOp = builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
      return addrOp.getDst();
    };

    Value dstPtr = emitAddr(dstVal);
    Value src0Ptr = emitAddr(src0Val);
    Value src1Ptr = emitAddr(src1Val);
    return {dstPtr, src0Ptr, src1Ptr, ptrType};
  }

  template <typename UBop>
  void modeNorm1L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 1) {
      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                          idxc(headRepeats, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value off = b.create<arith::MulIOp>(loc, iv, idxc(epr, loc, b)).getResult();
      Value rd = addPtr(loc, b, dst, ptrTy, off);
      Value r0 = addPtr(loc, b, s0, ptrTy, off);
      Value r1 = addPtr(loc, b, s1, ptrTy, off);
      b.create<pto::UBSetMaskCountOp>(loc);
      b.create<pto::UBSetMaskOp>(loc, i64c(epr, loc, b), i64c0(loc, b));
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c1(loc, b), i64c8(loc, b));
      b.create<pto::UBSetMaskNormOp>(loc);
      b.setInsertionPointAfter(forOp);
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, loc, b);
        Value td = addPtr(loc, b, dst, ptrTy, offT);
        Value ts0 = addPtr(loc, b, s0, ptrTy, offT);
        Value ts1 = addPtr(loc, b, s1, ptrTy, offT);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(tailElements, loc, b), i64c0(loc, b));
        emitUBBinOp<UBop>(loc, b, td, ts0, ts1, i64c1(loc, b), i64c8(loc, b));
        b.create<pto::UBSetMaskNormOp>(loc);
      }
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emitUBBinOp<UBop>(loc, b, dst, s0, s1, i64c1(loc, b), i64c8(loc, b));
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  void setMask(Location loc, OpBuilder &b, unsigned n) {
    auto [m0, m1] = computeContMaskValues(n);
    b.create<pto::UBSetMaskOp>(loc, i64c(m0, loc, b), i64c(m1, loc, b));
  }

  void fullMask(Location loc, OpBuilder &b) {
    b.create<pto::UBSetMaskOp>(loc, i64cM1(loc, b), i64c0(loc, b));
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
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    SmallVector<Value> offsets;
  };

  static FailureOr<DmaViewInfo> extractDmaViewInfo(pto::TLoadOp op) {
    auto pvOp = op.getSrc().getDefiningOp<pto::PartitionViewOp>();
    if (!pvOp)
      return failure();
    auto mtvOp = pvOp.getSource().getDefiningOp<pto::MakeTensorViewOp>();
    if (!mtvOp)
      return failure();
    DmaViewInfo info;
    info.gmPtr = mtvOp.getPtr();
    info.sizes.assign(pvOp.getSizes().begin(), pvOp.getSizes().end());
    info.strides.assign(mtvOp.getStrides().begin(),
                        mtvOp.getStrides().end());
    info.offsets.assign(pvOp.getOffsets().begin(), pvOp.getOffsets().end());
    return info;
  }

  Value computeGMByteOffset(Location loc, OpBuilder &b,
                            const DmaViewInfo &viewInfo, unsigned elemSize) {
    Value totalOff = idxc0(loc, b);
    for (size_t i = 0; i < viewInfo.offsets.size() &&
                        i < viewInfo.strides.size(); ++i) {
      APInt constOff;
      if (matchPattern(viewInfo.offsets[i], m_ConstantInt(&constOff)) &&
          constOff.isZero())
        continue;
      Value dimOff = b.create<arith::MulIOp>(loc, viewInfo.offsets[i],
                                             viewInfo.strides[i]).getResult();
      totalOff = b.create<arith::AddIOp>(loc, totalOff, dimOff).getResult();
    }
    if (elemSize > 1)
      totalOff = b.create<arith::MulIOp>(loc, totalOff,
                                         idxc(elemSize, loc, b)).getResult();
    return totalOff;
  }

  Value offsetGMPtrByBytes(Location loc, OpBuilder &b, Value gmPtr,
                           Value byteOff) {
    APInt constOff;
    if (matchPattern(byteOff, m_ConstantInt(&constOff)) && constOff.isZero())
      return gmPtr;
    auto origPtrTy = cast<pto::PtrType>(gmPtr.getType());
    auto bytePtrTy = pto::PtrType::get(b.getContext(), b.getI8Type(),
                                       origPtrTy.getMemorySpace());
    Value bytePtr = b.create<pto::CastPtrOp>(loc, bytePtrTy, gmPtr);
    Value offIdx = byteOff;
    if (!offIdx.getType().isIndex())
      offIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), byteOff)
                   .getResult();
    Value offsetBytePtr =
        b.create<pto::AddPtrOp>(loc, bytePtrTy, bytePtr, offIdx);
    return b.create<pto::CastPtrOp>(loc, origPtrTy, offsetBytePtr);
  }

  Value i64Cast(Location loc, OpBuilder &b, Value indexVal) {
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), indexVal)
        .getResult();
  }

  LogicalResult emitMteGmUb(Location loc, OpBuilder &b, Value gmPtr,
                             Value ubPtr, const DmaViewInfo &viewInfo,
                             Type elemTy, ArrayRef<int64_t> tileShape) {
    if (tileShape.size() < 2) return failure();
    int64_t ubCols = tileShape[1];
    unsigned elemSize = getElementSize(elemTy);
    unsigned nd = viewInfo.sizes.size();
    if (nd < 2) return failure();

    Value nburstCount = viewInfo.sizes[nd - 2];
    Value lenBurstElts = viewInfo.sizes[nd - 1];
    Value lenBurst = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(), lenBurstElts)
            .getResult(), i64c(elemSize, loc, b)).getResult();
    Value nburstSrcStride = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.strides[nd - 2]).getResult(),
        i64c(elemSize, loc, b)).getResult();
    Value ubRowStride = b.create<arith::MulIOp>(loc, i64c(ubCols, loc, b),
                                                i64c(elemSize, loc, b)).getResult();
    pto::DmaLoopConfig nburst{i64Cast(loc, b, nburstCount),
                              nburstSrcStride, ubRowStride};

    SmallVector<pto::DmaLoopConfig> loops;
    for (int i = nd - 3; i >= 0; --i) {
      Value count = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
          viewInfo.sizes[i]).getResult();
      Value srcStride = b.create<arith::MulIOp>(loc,
          b.create<arith::IndexCastOp>(loc, b.getI64Type(),
              viewInfo.strides[i]).getResult(),
          i64c(elemSize, loc, b)).getResult();
      Value innerElems = i64c1(loc, b);
      for (int j = i + 1; j < (int)nd; ++j) {
        Value dimSize = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.sizes[j]).getResult();
        innerElems = b.create<arith::MulIOp>(loc, innerElems, dimSize).getResult();
      }
      Value dstStride = b.create<arith::MulIOp>(loc, innerElems,
          i64c(elemSize, loc, b)).getResult();
      loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteGmUbOp>(loc, gmPtr, ubPtr, i64c0(loc, b), lenBurst,
        nburst, llvm::ArrayRef(loops), std::nullopt);
    return success();
  }

  LogicalResult emitMteUbGm(Location loc, OpBuilder &b, Value ubPtr,
                             Value gmPtr, const DmaViewInfo &viewInfo,
                             Type elemTy, ArrayRef<int64_t> tileShape) {
    if (tileShape.size() < 2) return failure();
    int64_t ubCols = tileShape[1];
    unsigned elemSize = getElementSize(elemTy);
    unsigned nd = viewInfo.sizes.size();
    if (nd < 2) return failure();

    Value nburstCount = viewInfo.sizes[nd - 2];
    Value lenBurstElts = viewInfo.sizes[nd - 1];
    Value lenBurst = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(), lenBurstElts)
            .getResult(), i64c(elemSize, loc, b)).getResult();
    Value nburstSrcStride = b.create<arith::MulIOp>(loc,
        i64c(ubCols, loc, b), i64c(elemSize, loc, b)).getResult();
    Value nburstDstStride = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.strides[nd - 2]).getResult(),
        i64c(elemSize, loc, b)).getResult();
    pto::DmaLoopConfig nburst{i64Cast(loc, b, nburstCount),
                              nburstSrcStride, nburstDstStride};

    SmallVector<pto::DmaLoopConfig> loops;
    for (int i = nd - 3; i >= 0; --i) {
      Value count = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
          viewInfo.sizes[i]).getResult();
      Value innerElems = i64c1(loc, b);
      for (int j = i + 1; j < (int)nd; ++j) {
        Value dimSize = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.sizes[j]).getResult();
        innerElems = b.create<arith::MulIOp>(loc, innerElems, dimSize).getResult();
      }
      Value srcStride = b.create<arith::MulIOp>(loc, innerElems,
          i64c(elemSize, loc, b)).getResult();
      Value dstStride = b.create<arith::MulIOp>(loc,
          b.create<arith::IndexCastOp>(loc, b.getI64Type(),
              viewInfo.strides[i]).getResult(),
          i64c(elemSize, loc, b)).getResult();
      loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteUbGmOp>(loc, ubPtr, gmPtr, lenBurst, nburst,
                             llvm::ArrayRef(loops));
    return success();
  }

  LogicalResult lowerTLoad(pto::TLoadOp op, OpBuilder &b,
                            const DenseMap<Value, SmallVector<int64_t, 2>> &tileShapes) {
    Location loc = op.getLoc();
    auto viewInfo = extractDmaViewInfo(op);
    if (failed(viewInfo)) return failure();
    Type dstType = op.getDst().getType();
    if (!isUBMemorySpace(dstType)) return failure();
    Type elemTy = getStoredElemType(dstType);
    if (!elemTy) return failure();
    unsigned elemSize = getElementSize(elemTy);
    if (elemSize == 0) return failure();

    auto it = tileShapes.find(op.getDst());
    if (it == tileShapes.end()) return failure();

    Value byteOff = computeGMByteOffset(loc, b, *viewInfo, elemSize);
    Value gmPtr = offsetGMPtrByBytes(loc, b, viewInfo->gmPtr, byteOff);
    return emitMteGmUb(loc, b, gmPtr, op.getDst(), *viewInfo, elemTy,
                       llvm::ArrayRef(it->second));
  }

  LogicalResult lowerTStore(pto::TStoreOp op, OpBuilder &b,
                             const DenseMap<Value, SmallVector<int64_t, 2>> &tileShapes) {
    Location loc = op.getLoc();
    Type srcType = op.getSrc().getType();
    if (!isUBMemorySpace(srcType)) return failure();
    Type elemTy = getStoredElemType(srcType);
    if (!elemTy) return failure();
    unsigned elemSize = getElementSize(elemTy);
    if (elemSize == 0) return failure();

    auto it = tileShapes.find(op.getSrc());
    if (it == tileShapes.end()) return failure();

    auto pvOp = op.getDst().getDefiningOp<pto::PartitionViewOp>();
    if (!pvOp) return failure();
    auto mtvOp = pvOp.getSource().getDefiningOp<pto::MakeTensorViewOp>();
    if (!mtvOp) return failure();

    DmaViewInfo viewInfo;
    viewInfo.gmPtr = mtvOp.getPtr();
    viewInfo.sizes.assign(pvOp.getSizes().begin(), pvOp.getSizes().end());
    viewInfo.strides.assign(mtvOp.getStrides().begin(), mtvOp.getStrides().end());
    viewInfo.offsets.assign(pvOp.getOffsets().begin(), pvOp.getOffsets().end());

    Value byteOff = computeGMByteOffset(loc, b, viewInfo, elemSize);
    Value gmPtr = offsetGMPtrByBytes(loc, b, viewInfo.gmPtr, byteOff);
    return emitMteUbGm(loc, b, op.getSrc(), gmPtr, viewInfo, elemTy,
                       llvm::ArrayRef(it->second));
  }

  //===--------------------------------------------------------------------===//
  // CCE dispatch tree — mirrors TBinOp.hpp BinaryInstr
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void dispatch(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t cols = info.cols;
    int64_t rows = info.rows;
    int64_t vRows = info.vRows;
    int64_t vCols = info.vCols;

    // 1. Small tile
    if (rows <= kRepeatMax && cols < static_cast<int64_t>(epr)) {
      modeSmall<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      return;
    }

    // 2. Continuous at compile time
    if (vCols == cols || vRows == 1) {
      int64_t totalV = vRows * vCols;
      int64_t totalRpts = (totalV + epr - 1) / epr;
      bool nonVLAligned =
          (vCols > static_cast<int64_t>(epr)) && ((vCols % epr) != 0);

      if (nonVLAligned || totalRpts > kRepeatMax)
        modeCount1L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      else
        modeNorm1L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      return;
    }

    // 3. Non-continuous
    int64_t normColRepeat = cols / epr;
    if (normColRepeat > 1 && vRows * normColRepeat < kSmallRptBinOp) {
      modeCount2L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
    } else if (vRows < normColRepeat + 1) {
      if (vCols % epr > 0)
        modeCount2L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      else
        modeColVLAlign<UBop>(loc, b, dst, s0, s1, ptrTy, info);
    } else {
      modeRowRpt<UBop>(loc, b, dst, s0, s1, ptrTy, info);
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin1LNormModeSmall
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeSmall(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                 pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t rs = info.cols / static_cast<int64_t>(info.blockSizeElem);

    if (info.vRows > 1) {
      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                         idxc(info.vRows, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value off = b.create<arith::MulIOp>(loc, iv, idxc(info.cols, loc, b)).getResult();
      Value rd = addPtr(loc, b, dst, ptrTy, off);
      Value r0 = addPtr(loc, b, s0, ptrTy, off);
      Value r1 = addPtr(loc, b, s1, ptrTy, off);
      b.create<pto::UBSetMaskCountOp>(loc);
      b.create<pto::UBSetMaskOp>(loc, i64c(info.vCols, loc, b), i64c0(loc, b));
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c1(loc, b), i64c(rs, loc, b));
      b.create<pto::UBSetMaskNormOp>(loc);
      b.setInsertionPointAfter(forOp);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(info.vCols, loc, b), i64c0(loc, b));
    emitUBBinOp<UBop>(loc, b, dst, s0, s1, i64c1(loc, b), i64c(rs, loc, b));
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin1LCountMode
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeCount1L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t totalV = info.vRows * info.vCols;
    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emitUBBinOp<UBop>(loc, b, dst, s0, s1, i64c0(loc, b), i64c8(loc, b));
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeColVLAlign
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeColVLAlign(Location loc, OpBuilder &b, Value dst, Value s0,
                      Value s1, pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t headRepeats = info.vCols / epr;
    int64_t rowStride = info.cols;

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b))
                    .getResult();
    Value rd = addPtr(loc, b, dst, ptrTy, off);
    Value rs0 = addPtr(loc, b, s0, ptrTy, off);
    Value rs1 = addPtr(loc, b, s1, ptrTy, off);
    emitUBBinOp<UBop>(loc, b, rd, rs0, rs1, i64c(headRepeats, loc, b), i64c8(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LCountMode – row-by-row count mode
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeCount2L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t rowStride = info.cols;
    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(info.vCols, loc, b),
                               i64c0(loc, b));

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b))
                    .getResult();
    Value rd = addPtr(loc, b, dst, ptrTy, off);
    Value rs0 = addPtr(loc, b, s0, ptrTy, off);
    Value rs1 = addPtr(loc, b, s1, ptrTy, off);
    emitUBBinOp<UBop>(loc, b, rd, rs0, rs1, i64c0(loc, b), i64c8(loc, b));
    b.setInsertionPointAfter(forOp);

    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeRowRpt
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeRowRpt(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t be = info.blockSizeElem;
    int64_t rowStride = info.cols;
    int64_t rs = rowStride / be;
    bool condRowRpt = (info.vRows <= kRepeatMax) && (rs <= kRepeatStrideMax);

    if (condRowRpt)
      rowRptFast<UBop>(loc, b, dst, s0, s1, ptrTy, info, rs);
    else
      rowRptChunked<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rs);
  }

  template <typename UBop>
  void rowRptFast(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = info.vCols / epr;
    int64_t tailElements = info.vCols % epr;

    for (int64_t i = 0; i < numLoop; i++) {
      Value rd = addPtr(loc, b, dst, ptrTy, idxc(i * epr, loc, b));
      Value r0 = addPtr(loc, b, s0, ptrTy, idxc(i * epr, loc, b));
      Value r1 = addPtr(loc, b, s1, ptrTy, idxc(i * epr, loc, b));
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
    }

    if (tailElements > 0) {
      Value off = idxc(numLoop * epr, loc, b);
      Value rd = addPtr(loc, b, dst, ptrTy, off);
      Value r0 = addPtr(loc, b, s0, ptrTy, off);
      Value r1 = addPtr(loc, b, s1, ptrTy, off);
      setMask(loc, b, tailElements);
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
      fullMask(loc, b);
    }
  }

  template <typename UBop>
  void rowRptChunked(Location loc, OpBuilder &b, Value dst, Value s0,
                     Value s1, pto::PtrType ptrTy, const TileShapeInfo &info,
                     int64_t rowStride, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t rptPerLine = info.vCols / epr;
    int64_t remainElem = info.vCols % epr;

    if (info.vRows > static_cast<int64_t>(epr)) {
      if (rptPerLine > 0)
        headRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rptPerLine);
      if (remainElem > 0) {
        Value off = idxc(rptPerLine * epr, loc, b);
        tailRows<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
                 addPtr(loc, b, s0, ptrTy, off),
                 addPtr(loc, b, s1, ptrTy, off), ptrTy, info, rowStride, rs,
                 remainElem);
      }
    } else {
      if (remainElem == 0) {
        headRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride,
                 info.vCols / epr);
      } else if (rptPerLine > 0) {
        headRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rptPerLine);
        Value off = idxc(rptPerLine * epr, loc, b);
        tailRows<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
                 addPtr(loc, b, s0, ptrTy, off),
                 addPtr(loc, b, s1, ptrTy, off), ptrTy, info, rowStride, rs,
                 remainElem);
      } else {
        tailRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rs, remainElem);
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeHead – chunked per-row head
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void headRows(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info,
                int64_t rowStride, int64_t rptPerLine) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = rptPerLine / kRepeatMax;
    int64_t remain = rptPerLine % kRepeatMax;
    int64_t chunkElems = kRepeatMax * epr;

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b),
                                      idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value rowBase =
        b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b)).getResult();

    if (numLoop > 0) {
      auto inner = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(numLoop, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(inner.getBody());
      Value jv = inner.getInductionVar();
      Value co = b.create<arith::MulIOp>(loc, jv, idxc(chunkElems, loc, b))
                     .getResult();
      Value off = b.create<arith::AddIOp>(loc, rowBase, co).getResult();
      emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
           addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
           i64c(kRepeatMax, loc, b), i64c8(loc, b));
      b.setInsertionPointAfter(inner);
    }

    if (remain > 0) {
      Value co = idxc(numLoop * chunkElems, loc, b);
      Value off = b.create<arith::AddIOp>(loc, rowBase, co).getResult();
      emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
           addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
           i64c(remain, loc, b), i64c8(loc, b));
    }
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeTail – masked per-row tail
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void tailRows(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info,
                int64_t rowStride, int64_t rs, unsigned remainPerLine) {
    bool strideOver =
        (rowStride / info.blockSizeElem > kRepeatStrideMax);
    setMask(loc, b, remainPerLine);

    int64_t numLoop = 0;
    int64_t remainAfterLoop = info.vRows;
    if (info.vRows > kRepeatMax) {
      numLoop = info.vRows / kRepeatMax;
      remainAfterLoop = info.vRows % kRepeatMax;

      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(numLoop, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      if (strideOver)
        tailStrideOverChunk<UBop>(loc, b, iv, dst, s0, s1, ptrTy, rowStride);
      else
        tailStrideOkChunk<UBop>(loc, b, iv, dst, s0, s1, ptrTy, rowStride, rs);
      b.setInsertionPointAfter(forOp);
    }

    if (remainAfterLoop > 0) {
      if (strideOver)
        tailStrideOverRemain<UBop>(loc, b, dst, s0, s1, ptrTy, rowStride, numLoop,
                             remainAfterLoop);
      else
        tailStrideOkRemain<UBop>(loc, b, dst, s0, s1, ptrTy, rowStride, rs, numLoop,
                           remainAfterLoop);
    }

    fullMask(loc, b);
  }

  template <typename UBop>
  void tailStrideOverChunk(Location loc, OpBuilder &b, Value iv, Value dst,
                           Value s0, Value s1, pto::PtrType ptrTy,
                           int64_t rowStride) {
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(kRepeatMax, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = b.create<arith::MulIOp>(
        loc, iv, idxc(kRepeatMax * rowStride, loc, b)).getResult();
    Value rowOff =
        b.create<arith::MulIOp>(loc, jv, idxc(rowStride, loc, b)).getResult();
    Value off = b.create<arith::AddIOp>(loc, baseOff, rowOff).getResult();
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c1(loc, b), i64c1(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  template <typename UBop>
  void tailStrideOkChunk(Location loc, OpBuilder &b, Value iv, Value dst,
                         Value s0, Value s1, pto::PtrType ptrTy,
                         int64_t rowStride, int64_t rs) {
    Value off = b.create<arith::MulIOp>(
        loc, iv, idxc(kRepeatMax * rowStride, loc, b)).getResult();
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c(kRepeatMax, loc, b), i64c(rs, loc, b));
  }

  template <typename UBop>
  void tailStrideOverRemain(Location loc, OpBuilder &b, Value dst, Value s0,
                            Value s1, pto::PtrType ptrTy, int64_t rowStride,
                            int64_t numLoop, int64_t remain) {
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b), idxc(remain, loc, b),
                                      idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = idxc(numLoop * kRepeatMax * rowStride, loc, b);
    Value rowOff =
        b.create<arith::MulIOp>(loc, jv, idxc(rowStride, loc, b)).getResult();
    Value off = b.create<arith::AddIOp>(loc, baseOff, rowOff).getResult();
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c1(loc, b), i64c1(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  template <typename UBop>
  void tailStrideOkRemain(Location loc, OpBuilder &b, Value dst, Value s0,
                          Value s1, pto::PtrType ptrTy, int64_t rowStride,
                          int64_t rs, int64_t numLoop, int64_t remain) {
    Value off = idxc(numLoop * kRepeatMax * rowStride, loc, b);
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c(remain, loc, b), i64c(rs, loc, b));
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
