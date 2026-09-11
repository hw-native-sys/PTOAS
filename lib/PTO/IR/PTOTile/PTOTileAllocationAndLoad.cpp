// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

//===----------------------------------------------------------------------===//
// AllocMultiTileOp / MultiTileGetOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyMultiTileSlot(AllocMultiTileOp op,
                                         pto::MultiTileBufType mtbTy) {
  TileBufType slotTy = mtbTy.getSlotType();
  if (!slotTy)
    return op.emitOpError("multi_tile_buf slot type must be non-null");
  Type elemTy = slotTy.getElementType();
  if (isPTOLowPrecisionType(elemTy))
    return op.emitOpError() << "slot dtype " << elemTy
                            << " is not supported by pto.alloc_multi_tile yet";
  if (failed(verifyTileBufLayoutConstraints(op, slotTy, "slot")) ||
      failed(verifyConstantLocalAddress(op, op.getAddr(),
                                        slotTy.getMemorySpace())))
    return failure();
  if (slotTy.getCompactModeI32() ==
      static_cast<int32_t>(mlir::pto::CompactMode::RowPlusOne))
    return op.emitOpError()
           << "multi_tile_buf slot uses row_plus_one compaction, whose padded "
              "storage footprint exceeds product(shape) and would overlap "
              "adjacent multi-buffer slots; use a compact (non-row_plus_one) "
              "slot layout or a single pto.alloc_tile";
  return success();
}

static LogicalResult verifyMultiTileValidShape(AllocMultiTileOp op,
                                               TileBufType slotTy) {
  auto vs = slotTy.getValidShape();
  if (vs.size() != 2)
    return op.emitOpError("slot tile_buf must have rank-2 validShape");
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);
  if (static_cast<bool>(op.getValidRow()) != needVR)
    return op.emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because slot v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  if (static_cast<bool>(op.getValidCol()) != needVC)
    return op.emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because slot v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  return success();
}

static LogicalResult verifyMultiTilePlannedAddresses(AllocMultiTileOp op,
                                                     pto::MultiTileBufType mtbTy) {
  Attribute rawAddrs = op->getAttr(pto::kPtoMultiBufferAddrsAttrName);
  if (!rawAddrs)
    return success();
  uint32_t count = mtbTy.getCount();
  auto addrs = dyn_cast<DenseI64ArrayAttr>(rawAddrs);
  if (!addrs)
    return op.emitOpError() << "expects internal '" << pto::kPtoMultiBufferAddrsAttrName
                            << "' to be a dense i64 array";
  if (op.getAddr())
    return op.emitOpError() << "cannot carry both base 'addr' and internal '"
                            << pto::kPtoMultiBufferAddrsAttrName << "'";
  if (addrs.size() != count)
    return op.emitOpError() << "expects " << count
                            << " planned slot addresses, got " << addrs.size();
  TileBufType slotTy = mtbTy.getSlotType();
  uint64_t slotBytes = getPTOStorageElemByteSize(slotTy.getElementType());
  for (int64_t dim : slotTy.getShape()) {
    if (dim == ShapedType::kDynamic)
      return op.emitOpError(
          "planned multi-buffer addresses require a static slot shape");
    slotBytes *= static_cast<uint64_t>(dim);
  }
  for (auto [lhsIdx, lhs] : llvm::enumerate(addrs.asArrayRef())) {
    if (lhs < 0)
      return op.emitOpError("planned slot addresses must be non-negative");
    for (size_t rhsIdx = lhsIdx + 1;
         rhsIdx < static_cast<size_t>(addrs.size()); ++rhsIdx) {
      uint64_t lhsBegin = lhs;
      uint64_t rhsBegin = addrs[rhsIdx];
      if (std::max(lhsBegin, rhsBegin) <
          std::min(lhsBegin + slotBytes, rhsBegin + slotBytes))
        return op.emitOpError() << "planned slots " << lhsIdx << " and "
                                << rhsIdx << " overlap";
    }
  }
  return success();
}

LogicalResult AllocMultiTileOp::verify() {
  auto mtbTy = getResult().getType();
  if (!mtbTy)
    return emitOpError("result must be `!pto.multi_tile_buf`");
  if (failed(verifyMultiTileSlot(*this, mtbTy)) ||
      failed(verifyMultiTileValidShape(*this, mtbTy.getSlotType())))
    return failure();
  uint32_t count = mtbTy.getCount();
  if (count < kPtoMultiBufferMinNum || count > kPtoMultiBufferMaxNum)
    return emitOpError() << "multi_tile_buf count must be in ["
                         << kPtoMultiBufferMinNum << ", "
                         << kPtoMultiBufferMaxNum << "] (got " << count << ")";
  return verifyMultiTilePlannedAddresses(*this, mtbTy);
}

LogicalResult MultiTileGetOp::verify() {
  auto srcTy = getSource().getType();
  auto resultTy = getResult().getType();
  if (!srcTy || !resultTy) {
    return emitOpError("source and result types must be non-null");
  }

  if (srcTy.getSlotType() != resultTy) {
    return emitOpError()
           << "result tile_buf must match the multi_tile_buf slot type: "
           << "expected " << srcTy.getSlotType() << ", got " << resultTy;
  }

  // If slot is an `arith.constant`, check it is in range.
  if (auto slotDef = getSlot().getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<IntegerAttr>(slotDef.getValue())) {
      int64_t slotVal = attr.getValue().getSExtValue();
      int64_t count = static_cast<int64_t>(srcTy.getCount());
      if (slotVal < 0 || slotVal >= count) {
        return emitOpError()
               << "constant slot " << slotVal
               << " is out of range for multi_tile_buf count=" << count;
      }
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }

  auto tileTy = dyn_cast<TileBufType>(getTile().getType());
  if (!tileTy) {
    return emitOpError("expects tile operand and result to be !pto.tile_buf");
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        tileTy.getMemorySpace()))) {
    return failure();
  }

  return success();
}

using LoadTypes =
    std::pair<pto::PartitionTensorViewType, pto::TileBufType>;

static LogicalResult verifyShapeSign(Operation *op, ArrayRef<int64_t> shape,
                                     StringRef name, bool positive) {
  for (auto [index, dim] : llvm::enumerate(shape)) {
    bool invalid = positive ? dim <= 0 : dim < 0;
    if (dim != ShapedType::kDynamic && invalid)
      return op->emitOpError() << "expects " << name << "[" << index
                               << "] to be "
                               << (positive ? "positive" : "non-negative");
  }
  return success();
}

static FailureOr<LoadTypes> verifyTLoadCommon(TLoadOp op,
                                              bool allowLowPrecision) {
  auto srcPart = dyn_cast<pto::PartitionTensorViewType>(op.getSrc().getType());
  auto dstTile = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!srcPart || !dstTile) {
    op.emitOpError(
        "expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision)) ||
      failed(verifyShapeSign(op, srcPart.getShape(), "src shape", true)) ||
      failed(verifyShapeSign(op, dstTile.getValidShape(), "dst valid_shape",
                             false)))
    return failure();
  return LoadTypes(srcPart, dstTile);
}

static LogicalResult verifyTLoadA2A3(TLoadOp op) {
  auto common = verifyTLoadCommon(op, /*allowLowPrecision=*/false);
  if (failed(common))
    return failure();
  auto [srcPart, dstTile] = *common;
  Type srcElem = srcPart.getElementType();
  Type dstElem = dstTile.getElementType();
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
    return op.emitOpError(
        "expects A2/A3 tload low-precision element types to be unsupported");
  if (!isSupportedLoadStoreElemTypeA2A3(dstElem))
    return op.emitOpError(
        "expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                    *dstSpace != pto::AddressSpace::MAT))
    return op.emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
    return op.emitOpError(
        "expects src and dst element types to have the same bitwidth");
  return success();
}

static bool hasA5LoadStoreLayout(pto::TileBufType tile) {
  int32_t bl = tile.getBLayoutValueI32();
  int32_t sl = tile.getSLayoutValueI32();
  bool isND = bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::NoneBox);
  bool isDN = bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::NoneBox);
  bool isNZ = bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::RowMajor);
  return isND || isDN || isNZ;
}

static LogicalResult verifyTLoadA5Types(TLoadOp op, Type srcElem,
                                        pto::TileBufType dstTile) {
  Type dstElem = dstTile.getElementType();
  unsigned srcBytes = getElemByteSize(srcElem);
  unsigned dstBytes = getElemByteSize(dstElem);
  if (srcBytes != dstBytes)
    return op.emitOpError(
        "expects src and dst element types to have the same element size");
  if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8))
    return op.emitOpError(
        "expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
  if (!isA5TLoadStoreTransferElemType(srcElem))
    return op.emitOpError(
        "expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  if (!isA5TLoadStoreTransferElemType(dstElem))
    return op.emitOpError(
        "expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  auto pad = dstTile.getPadValueI32();
  if (dstElem.isInteger(64) &&
      pad != static_cast<int32_t>(pto::PadValue::Null) &&
      pad != static_cast<int32_t>(pto::PadValue::Zero))
    return op.emitOpError(
        "expects A5 i64/u64 tload dst pad to be null or zero");
  return success();
}

static LogicalResult verifyTLoadA5(TLoadOp op) {
  auto common = verifyTLoadCommon(op, /*allowLowPrecision=*/true);
  if (failed(common))
    return failure();
  auto [srcPart, dstTile] = *common;
  if (failed(verifyTLoadA5Types(op, srcPart.getElementType(), dstTile)))
    return failure();
  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (dstSpace && *dstSpace == pto::AddressSpace::VEC &&
      !hasA5LoadStoreLayout(dstTile))
    return op.emitOpError(
        "expects A5 tload vec dst layout to be ND, DN, or NZ");
  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyA2A3 = [&]() { return verifyTLoadA2A3(*this); };
  auto verifyA5 = [&]() { return verifyTLoadA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyStaticShapeSign(Operation *op, ArrayRef<int64_t> shape,
                                           StringRef name, bool requirePositive) {
  return verifyShapeSign(op, shape, name, requirePositive);
}
