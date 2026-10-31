// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

#include "PTO/IR/PTOMultiBuffer.h"

//===----------------------------------------------------------------------===//
// AllocMultiTileOp / MultiTileGetOp
//===----------------------------------------------------------------------===//

constexpr unsigned kValidShapeRank = 2;
// Referenced by PTOExtentCompatibility.cpp, which is part of the same PTO.cpp
// aggregate translation unit.
constexpr unsigned kI64ElemBitWidth = 64;

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
  if (vs.size() != kValidShapeRank)
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

  auto tileTy = getTile().getType();
  if (!isa<TileBufType, ConvTileBufType>(tileTy)) {
    return emitOpError("expects tile operand and result to be !pto.tile_buf or !pto.conv_tile");
  }

  Attribute memorySpace;
  if (auto tb = dyn_cast<TileBufType>(tileTy)) {
    memorySpace = tb.getMemorySpace();
  } else if (auto ct = dyn_cast<ConvTileBufType>(tileTy)) {
    memorySpace = ct.getMemorySpace();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        memorySpace))) {
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

using TLoadOperandTypes =
    std::pair<pto::PartitionTensorViewType, Type>;

// A rank-4 FRACTAL_Z conv tile is [C1HW, N/16, 16, C0]: the last two dims
// are fixed by the hardware fractal layout. The textract apply the same
// verify rule.
static LogicalResult verifyTLoadConvFractalZTrailingDims(
    TLoadOp op, pto::ConvTileBufType dstCT, pto::ConvLayoutAttr dstLayout) {
  if (dstLayout.getValue() != pto::ConvLayout::FRACTAL_Z ||
      dstCT.getShape().size() != 4) {
    return success();
  }
  const int64_t elemBytes = getPTOStorageElemByteSize(dstCT.getElementType());
  const std::optional<int64_t> c0 = mlir::pto::convC0ForElemBytes(elemBytes);
  if (!c0) {
    return op.emitOpError() << "expects tload conv dst element type to "
                               "have a known storage size";
  }
  ArrayRef<int64_t> shape = dstCT.getShape();
  if (shape[2] != 16 || shape[3] != *c0) {
    return op.emitOpError()
           << "expects tload conv FRACTAL_Z dst trailing dims to be [16, "
           << *c0 << "]";
  }
  return success();
}

// A2/A3 restricts conv tile destinations to i8/i16/i32/f16/bf16/f32;
// A5 additionally accepts the two standard FP8 encodings. Neither accepts
// i64 (or A5's hif8/fp4) on the conv path, which the generic transfer
// whitelists wrongly admitted.
static LogicalResult verifyTLoadConvDstElemType(TLoadOp op, Type dstTile,
                                                Type dstElem) {
  if (!isa<pto::ConvTileBufType>(dstTile) ||
      dstElem.isInteger(8) || dstElem.isInteger(16) ||
      dstElem.isInteger(32) || dstElem.isF16() || dstElem.isBF16() ||
      dstElem.isF32() ||
      (getVerifierTargetArch(op.getOperation()) == VerifierTargetArch::A5 &&
       (pto::isPTOFloat8E4M3LikeType(dstElem) ||
        pto::isPTOFloat8E5M2LikeType(dstElem)))) {
    return success();
  }
  return op.emitOpError(
      "expects tload conv dst element type to be i8/i16/i32/f16/bf16/f32"
      " (A5 also accepts f8e4m3/f8e5m2)");
}

// The GM view layout paired with the conv tile layout must be one the
// runtime TLOAD_CONVTILE_IMPL accepts. The two runtime generations differ:
// A2/A3 only accepts identity pairs (a2a3/TLoad.hpp CheckConvTileData),
// A5 additionally converts raw feature-map layouts (a5/TLoad.hpp).
static LogicalResult
verifyTLoadConvA5ConvertShapes(TLoadOp op,
                              pto::PartitionTensorViewType srcPart,
                              pto::ConvTileBufType dstCT, pto::ConvLayout src);

static LogicalResult
verifyTLoadConvTileLayoutPair(TLoadOp op, PTOArch arch,
                              pto::PartitionTensorViewType srcPart,
                              Type dstTile) {
  auto dstCT = dyn_cast<pto::ConvTileBufType>(dstTile);
  if (!dstCT) {
    return success();
  }
  auto srcLayout =
      dyn_cast_or_null<pto::ConvLayoutAttr>(srcPart.getLayout());
  auto dstLayout = dyn_cast_or_null<pto::ConvLayoutAttr>(dstCT.getLayout());
  if (!srcLayout || !dstLayout) {
    return op.emitOpError(
        "expects tload into a conv_tile to have conv layouts on both the "
        "source view and the destination tile");
  }
  using CL = pto::ConvLayout;
  auto same = [&](CL gm, CL tile) {
    return srcLayout.getValue() == gm && dstLayout.getValue() == tile;
  };
  const bool legal =
      same(CL::NC1HWC0, CL::NC1HWC0) || same(CL::FRACTAL_Z, CL::FRACTAL_Z) ||
      (arch == PTOArch::A5 &&
       (same(CL::NHWC, CL::NC1HWC0) || same(CL::NCHW, CL::NC1HWC0) ||
        same(CL::NCHW, CL::FRACTAL_Z) || same(CL::NCDHW, CL::FRACTAL_Z_3D) ||
        same(CL::NCDHW, CL::NDC1HWC0))) ||
      (arch != PTOArch::A5 &&
       (same(CL::FRACTAL_Z_3D, CL::FRACTAL_Z_3D) ||
        same(CL::NDC1HWC0, CL::NDC1HWC0)));
  if (!legal) {
    return op.emitOpError()
           << "tload conv layout pair "
           << stringifyConvLayout(srcLayout.getValue()) << " -> "
           << stringifyConvLayout(dstLayout.getValue())
           << " is not supported on this arch";
  }
  if (arch == PTOArch::A5 &&
      failed(verifyTLoadConvA5ConvertShapes(op, srcPart, dstCT,
                                            srcLayout.getValue()))) {
    return failure();
  }
  return verifyTLoadConvFractalZTrailingDims(op, dstCT, dstLayout);
}

// The A5 conversion pairs (raw GM layouts -> fractal tile) assert shape
// and stride contracts at device-compile time that PTOAS can check
// statically from the view and tile types:
//   - NCHW->NC1HWC0 / NCHW->FRACTAL_Z / NCDHW->FRACTAL_Z_3D /
//     NCDHW->NDC1HWC0: the GM layout's trailing H*W dims must be
//     memory-contiguous (runtime: gStride_C == H*W).
//   - NHWC->NC1HWC0 / NCHW->NC1HWC0: the source spatial dims must match
//     the destination tile's.
static bool convShapeDimEq(ArrayRef<int64_t> srcShape,
                            ArrayRef<int64_t> dstShape, size_t srcIdx,
                            size_t dstIdx) {
  return srcShape.size() > srcIdx && dstShape.size() > dstIdx &&
         srcShape[srcIdx] == dstShape[dstIdx];
}

static bool convShapeDimDynamic(ArrayRef<int64_t> srcShape, size_t idx) {
  return srcShape.size() <= idx || srcShape[idx] == ShapedType::kDynamic;
}

// N/H/W dims must agree between the GM view and the tile for each layout
// family; the trailing dims are checked pairwise.
static LogicalResult checkConvConvertDims(TLoadOp op,
                                          ArrayRef<int64_t> srcShape,
                                          ArrayRef<int64_t> dstShape,
                                          pto::ConvLayout src,
                                          pto::ConvLayout dst) {
  using CL = pto::ConvLayout;
  SmallVector<std::pair<size_t, size_t>> dims;
  if (src == CL::NHWC && dst == CL::NC1HWC0) {
    dims = {{0, 0}, {1, 2}, {2, 3}};
  } else if (src == CL::NCHW && dst == CL::NC1HWC0) {
    dims = {{0, 0}, {2, 2}, {3, 3}};
  } else if (src == CL::NCDHW && dst == CL::NDC1HWC0) {
    dims = {{0, 0}, {3, 3}, {4, 4}};
  } else {
    return success();
  }
  for (auto [srcIdx, dstIdx] : dims) {
    if (convShapeDimDynamic(srcShape, srcIdx)) {
      continue;
    }
    if (!convShapeDimEq(srcShape, dstShape, srcIdx, dstIdx)) {
      return op.emitOpError()
             << "tload conversion requires matching dims at src[" << srcIdx
             << "]/dst[" << dstIdx << "]: src [" << srcShape << "] vs dst ["
             << dstShape << "]";
    }
  }
  return success();
}

// N/H/W-style pairs: the src dim must not exceed its valid counterpart.
static LogicalResult convValidFitDims(TLoadOp op, ArrayRef<int64_t> srcShape, ArrayRef<int64_t> valid,
                                      ArrayRef<std::pair<size_t, size_t>> fits)
{
    for (auto [srcIdx, validIdx] : fits) {
        if (convShapeDimDynamic(srcShape, srcIdx) || validIdx >= valid.size()) {
            continue;
        }
        if (srcShape[srcIdx] > valid[validIdx]) {
            return op.emitOpError() << "tload source dim " << srcIdx << " (" << srcShape[srcIdx]
                                    << ") exceeds the tile valid dim " << valid[validIdx]
                                    << "; data outside valid_shape is dead";
        }
    }
    return success();
}

// Channel-capacity pairs: the src channel count must fit valid C1 * C0.
static LogicalResult convValidFitChannelCaps(TLoadOp op, ArrayRef<int64_t> srcShape, pto::ConvTileBufType dstCT,
                                             ArrayRef<int64_t> valid, ArrayRef<std::pair<size_t, size_t>> cCaps)
{
    const int64_t elemBytes = getPTOStorageElemByteSize(dstCT.getElementType());
    const std::optional<int64_t> c0Opt = pto::convC0ForElemBytes(elemBytes);
    if (!c0Opt) {
        return success();
    }
    for (auto [srcIdx, c1Idx] : cCaps) {
        if (convShapeDimDynamic(srcShape, srcIdx) || c1Idx >= valid.size()) {
            continue;
        }
        if (srcShape[srcIdx] > valid[c1Idx] * *c0Opt) {
            return op.emitOpError() << "tload source dim " << srcIdx << " (" << srcShape[srcIdx]
                                    << ") exceeds the tile valid channel capacity " << valid[c1Idx] * *c0Opt
                                    << " (valid_shape)";
        }
    }
    return success();
}

// With a declared valid shape, everything the GM source contributes must
// fit inside the valid box: extents beyond it are dead for valid-aware
// consumers. For the N/H/W dims the transfer itself pins src == declared
// (checkConvConvertDims above), so a shrunk valid dim there contradicts
// the load; for the C capacity the bound is src C <= valid C1 * C0.
static LogicalResult checkConvConvertValidFit(
    TLoadOp op, ArrayRef<int64_t> srcShape, pto::ConvTileBufType dstCT, pto::ConvLayout src)
{
    using CL = pto::ConvLayout;
    if (!dstCT.hasValidShape()) {
        return success();
    }
    const auto dstCL = dyn_cast_or_null<pto::ConvLayoutAttr>(dstCT.getLayout());
    if (!dstCL) {
        return success();
    }
    ArrayRef<int64_t> valid = dstCT.getValidOrShape();
    // (srcIdx, validIdx) pairs that must satisfy src <= valid.
    SmallVector<std::pair<size_t, size_t>> fits;
    // (srcIdx, c1Idx) pairs that must satisfy src C <= valid[c1Idx] * c0.
    SmallVector<std::pair<size_t, size_t>> cCaps;
    if (src == CL::NHWC && dstCL.getValue() == CL::NC1HWC0) {
        fits = {{0, 0}, {1, 2}, {2, 3}};
        cCaps = {{3, 1}};
    } else if (src == CL::NCHW && dstCL.getValue() == CL::NC1HWC0) {
        fits = {{0, 0}, {2, 2}, {3, 3}};
        cCaps = {{1, 1}};
    } else if (src == CL::NCDHW && dstCL.getValue() == CL::NDC1HWC0) {
        fits = {{0, 0}, {2, 1}, {3, 3}, {4, 4}};
        cCaps = {{1, 2}};
    } else {
        return success();
    }
    if (failed(convValidFitDims(op, srcShape, valid, fits))) {
        return failure();
    }
    return convValidFitChannelCaps(op, srcShape, dstCT, valid, cCaps);
}

static LogicalResult verifyTLoadConvA5ConvertShapes(
    TLoadOp op, pto::PartitionTensorViewType srcPart, pto::ConvTileBufType dstCT, pto::ConvLayout src)
{
    auto dstCL = dyn_cast_or_null<pto::ConvLayoutAttr>(dstCT.getLayout());
    if (!dstCL) {
        return success();
    }
    if (failed(checkConvConvertDims(op, srcPart.getShape(), dstCT.getShape(), src, dstCL.getValue()))) {
        return failure();
    }
    return checkConvConvertValidFit(op, srcPart.getShape(), dstCT, src);
}

static FailureOr<TLoadOperandTypes> verifyTLoadOperandTypes(TLoadOp op, bool allowLowPrecision)
{
    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(op.getSrc().getType());
    Type dstTy = op.getDst().getType();
    if (!srcPart || !isa<pto::TileBufType, pto::ConvTileBufType>(dstTy)) {
        op.emitOpError(
            "expects src to be !pto.partition_tensor_view and dst to be "
            "!pto.tile_buf or !pto.conv_tile");
        return failure();
    }
    if (failed(verifyTileLikeCommon(op, dstTy, "dst", allowLowPrecision))) {
        return failure();
    }

    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
        if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
            op.emitOpError() << "expects src shape[" << i << "] to be positive";
            return failure();
        }
    }
    auto dstValid = getValidShapeVec(dstTy);
    for (unsigned i = 0; i < dstValid.size(); ++i) {
        if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
            op.emitOpError() << "expects dst valid_shape[" << i << "] to be non-negative";
            return failure();
        }
    }
    return TLoadOperandTypes(srcPart, dstTy);
}

static LogicalResult verifyTLoadA2A3Types(TLoadOp op) {
  auto common = verifyTLoadOperandTypes(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  auto [srcPart, dstTile] = *common;

  Type srcElem = srcPart.getElementType();
  Type dstElem = getElemTy(dstTile);
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
    return op.emitOpError(
        "expects A2/A3 tload low-precision element types to be unsupported");
  }
  if (!(dstElem.isInteger(8) || dstElem.isInteger(16) ||
        dstElem.isInteger(32) || dstElem.isInteger(64) || dstElem.isF16() ||
        dstElem.isBF16() || dstElem.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
  }

  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                    *dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
  }

  if (failed(verifyTLoadConvDstElemType(op, dstTile, dstElem))) {
    return failure();
  }
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
    return op.emitOpError(
        "expects src and dst element types to have the same bitwidth");
  }
  return verifyTLoadConvTileLayoutPair(op, PTOArch::A3, srcPart, dstTile);
}

static LogicalResult verifyTLoadA5PadAndVecLayout(TLoadOp op, Type dstElem,
                                                  Type dstTile) {
  if (dstElem.isInteger(64)) {
    if (auto dstTB = dyn_cast<pto::TileBufType>(dstTile)) {
      auto pad = dstTB.getPadValueI32();
      if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
          pad != static_cast<int32_t>(pto::PadValue::Zero)) {
        return op.emitOpError(
            "expects A5 i64/u64 tload dst pad to be null or zero");
      }
    }
  }

  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (dstSpace && *dstSpace == pto::AddressSpace::VEC) {
    auto dstTB = dyn_cast<pto::TileBufType>(dstTile);
    if (!dstTB) {
      return op.emitOpError("expects A5 tload vec dst to be a tile_buf");
    }
    int32_t bl = dstTB.getBLayoutValueI32();
    int32_t sl = dstTB.getSLayoutValueI32();
    bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                 sl == static_cast<int32_t>(pto::SLayout::NoneBox));
    bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                 sl == static_cast<int32_t>(pto::SLayout::NoneBox));
    bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                 sl == static_cast<int32_t>(pto::SLayout::RowMajor));
    if (!isND && !isDN && !isNZ) {
      return op.emitOpError("expects A5 tload vec dst layout to be ND, DN, or NZ");
    }
  }
  return success();
}

static LogicalResult verifyTLoadA5Types(TLoadOp op) {
  auto common = verifyTLoadOperandTypes(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  auto [srcPart, dstTile] = *common;

  Type srcElem = srcPart.getElementType();
  Type dstElem = getElemTy(dstTile);
  unsigned srcBytes = getElemByteSize(srcElem);
  unsigned dstBytes = getElemByteSize(dstElem);
  if (srcBytes != dstBytes) {
    return op.emitOpError(
        "expects src and dst element types to have the same element size");
  }
  if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8)) {
    return op.emitOpError(
        "expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
  }
  if (!isA5TLoadStoreTransferElemType(srcElem)) {
    return op.emitOpError(
        "expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  if (!isA5TLoadStoreTransferElemType(dstElem)) {
    return op.emitOpError(
        "expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  if (failed(verifyTLoadConvDstElemType(op, dstTile, dstElem))) {
    return failure();
  }
  if (failed(verifyTLoadConvTileLayoutPair(op, PTOArch::A5, srcPart,
                                           dstTile))) {
    return failure();
  }
  return verifyTLoadA5PadAndVecLayout(op, dstElem, dstTile);
}

LogicalResult TLoadOp::verify() {
  if (getL2HintAttr()) {
    if (failed(verifyA5L2HintTarget(getOperation()))) {
      return failure();
    }
    const auto aliasPolicy = getCachePolicyAttr();
    const bool hasBypass = aliasPolicy && aliasPolicy.getValue() == pto::LoadCachePolicy::L2Bypass;
    const bool hasOffset = static_cast<bool>(getOffset());
    if (hasOffset || hasBypass) {
      return emitOpError("l2_hint cannot be combined with a cache-policy bypass or offset");
    }
    if (!isa<pto::TileBufType>(getDst().getType())) {
      return emitOpError("l2_hint requires a regular tile_buf destination");
    }
  }
  auto verifyA2A3 = [this]() -> LogicalResult { return verifyTLoadA2A3Types(*this); };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTLoadA5Types(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyStaticShapeSign(Operation *op, ArrayRef<int64_t> shape,
                                           StringRef name, bool requirePositive) {
  return verifyShapeSign(op, shape, name, requirePositive);
}
