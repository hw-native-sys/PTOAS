// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyRowReductionDstLayout(Operation *op, Type ty,
                                                 StringRef name) {
  auto verifyBaseLayout = [op, ty, name]() -> LogicalResult {
    if (failed(verifyTileBufCommon(op, ty, name)))
      return failure();
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC)
      return op->emitOpError()
             << "expects " << name << " to be in the vec address space";
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
      if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return op->emitOpError()
               << "expects " << name << " to use the none_box slayout";
      if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
        return op->emitOpError()
               << "expects " << name
               << " to use the row_major or col_major blayout";
      }
    }
    return success();
  };
  auto verifyTileLayout = [op, ty, name](pto::TileBufType tb) -> LogicalResult {
    auto layout = getTileBufLogicalLayout(tb);
    if (!layout || *layout == pto::Layout::ND)
      return success();
    if (*layout != pto::Layout::DN) {
      return op->emitOpError()
             << "expects " << name
             << " to use a DN-style column vector tile or legacy ND-style tile";
    }
    auto shape = getShapeVec(ty);
    if (shape.size() == kPTORowColRank && shape[1] != ShapedType::kDynamic &&
        shape[1] != 1) {
      return op->emitOpError()
             << "expects DN-style " << name << " to have shape[1] == 1";
    }
    return success();
  };

  if (failed(verifyBaseLayout()))
    return failure();
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return verifyTileLayout(tb);
  return success();
}

static LogicalResult verifyPositiveStaticDims(Operation *op,
                                             ArrayRef<int64_t> dims,
                                             StringRef name,
                                             StringRef kind) {
  for (auto [idx, dim] : llvm::enumerate(dims)) {
    if (dim != ShapedType::kDynamic && dim <= 0) {
      return op->emitOpError()
             << "expects " << name << " " << kind << "[" << idx
             << "] to be positive";
    }
  }
  return success();
}

static LogicalResult verifyPositiveRankedMemrefShape(Operation *op, MemRefType mr,
                                                     StringRef name) {
  if (!mr.hasRank())
    return op->emitOpError() << "expects " << name << " memref to be ranked";
  for (int64_t dim : mr.getShape()) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " memref shape to be positive";
  }
  return success();
}

struct TLoadVerifyInfo {
  pto::PartitionTensorViewType srcPart;
  pto::TileBufType dstTile;
  Type srcElem;
  Type dstElem;
};

static FailureOr<TLoadVerifyInfo> verifyTLoadCommon(Operation *op, Value src,
                                                    Value dst,
                                                    bool allowLowPrecision) {
  auto srcPart = dyn_cast<pto::PartitionTensorViewType>(src.getType());
  auto dstTile = dyn_cast<pto::TileBufType>(dst.getType());
  if (!srcPart || !dstTile) {
    op->emitOpError(
        "expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision)) ||
      failed(verifyPositiveStaticDims(op, srcPart.getShape(), "src", "shape")) ||
      failed(
          verifyPositiveStaticDims(op, dstTile.getValidShape(), "dst", "valid_shape"))) {
    return failure();
  }
  return TLoadVerifyInfo{srcPart, dstTile, srcPart.getElementType(),
                         dstTile.getElementType()};
}

static bool isA2A3TLoadDstElemType(Type elem) {
  return elem.isInteger(kPTOI8BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth) ||
         elem.isInteger(kPTOI64BitWidth) || elem.isF16() || elem.isBF16() || elem.isF32();
}

static LogicalResult verifyTLoadA2A3(Operation *op, const TLoadVerifyInfo &info) {
  if (isPTOLowPrecisionType(info.srcElem) || isPTOLowPrecisionType(info.dstElem))
    return op->emitOpError(
        "expects A2/A3 tload low-precision element types to be unsupported");
  if (!isA2A3TLoadDstElemType(info.dstElem)) {
    return op->emitOpError(
        "expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
  }

  auto dstSpace = getPTOMemorySpaceEnum(info.dstTile);
  if (!dstSpace ||
      (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
    return op->emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
  }
  if (getElemByteSize(info.srcElem) != getElemByteSize(info.dstElem)) {
    return op->emitOpError(
        "expects src and dst element types to have the same bitwidth");
  }
  return success();
}

static LogicalResult verifyTLoadA5(Operation *op, const TLoadVerifyInfo &info) {
  unsigned srcBytes = getElemByteSize(info.srcElem);
  unsigned dstBytes = getElemByteSize(info.dstElem);
  if (srcBytes != dstBytes) {
    return op->emitOpError(
        "expects src and dst element types to have the same element size");
  }
  if (!(dstBytes == kPTOByteSize || dstBytes == kPTOHalfWordBytes ||
        dstBytes == kPTOWordBytes || dstBytes == kPTODoubleWordBytes)) {
    return op->emitOpError(
        "expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
  }
  if (!isA5TLoadStoreTransferElemType(info.srcElem)) {
    return op->emitOpError(
        "expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  if (!isA5TLoadStoreTransferElemType(info.dstElem)) {
    return op->emitOpError(
        "expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  if (info.dstElem.isInteger(kPTOI64BitWidth)) {
    auto pad = info.dstTile.getPadValueI32();
    if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
        pad != static_cast<int32_t>(pto::PadValue::Zero)) {
      return op->emitOpError(
          "expects A5 i64/u64 tload dst pad to be null or zero");
    }
  }
  return success();
}

static FailureOr<Type> verifyTPrefetchSrcElemType(Operation *op, Type srcTy) {
  if (auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy)) {
    if (failed(verifyPositiveStaticDims(op, srcPart.getShape(), "src", "shape")))
      return failure();
    return srcPart.getElementType();
  }
  if (auto srcMr = dyn_cast<MemRefType>(srcTy)) {
    if (failed(verifyPositiveRankedMemrefShape(op, srcMr, "src")))
      return failure();
    return srcMr.getElementType();
  }
  op->emitOpError("expects src to be !pto.partition_tensor_view or memref");
  return failure();
}

static FailureOr<Type> verifyTPrefetchDstElemType(Operation *op, Type dstTy,
                                                  bool allowLowPrecision) {
  if (auto dstTile = dyn_cast<pto::TileBufType>(dstTy)) {
    if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision)) ||
        failed(
            verifyPositiveStaticDims(op, dstTile.getValidShape(), "dst", "valid_shape"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace ||
        (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
      return op->emitOpError("expects dst to use loc=vec or loc=mat"), failure();
    }
    return dstTile.getElementType();
  }
  if (auto dstMr = dyn_cast<MemRefType>(dstTy)) {
    auto dstSpace = getPTOMemorySpaceEnum(dstMr);
    if (!dstSpace ||
        (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
      return op->emitOpError("expects dst memref to use loc=vec or loc=mat"),
             failure();
    }
    if (failed(verifyPositiveRankedMemrefShape(op, dstMr, "dst")) ||
        failed(verifyTileBufCommon(op, dstMr, "dst", allowLowPrecision))) {
      return failure();
    }
    return dstMr.getElementType();
  }
  op->emitOpError("expects dst to be !pto.tile_buf or memref");
  return failure();
}
