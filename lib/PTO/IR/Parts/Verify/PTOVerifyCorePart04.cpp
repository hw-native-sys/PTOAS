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

static LogicalResult verifyTPrefetchElemTypes(Operation *op, Type srcElem,
                                              Type dstElem,
                                              bool allowLowPrecision) {
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
    return op->emitOpError(
        "expects src and dst element types to have the same element size");
  }
  if (!allowLowPrecision &&
      (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))) {
    return op->emitOpError(
        "expects A2/A3 tprefetch low-precision element types to be unsupported");
  }
  if (allowLowPrecision &&
      (!isA5TLoadStoreTransferElemType(srcElem) ||
       !isA5TLoadStoreTransferElemType(dstElem))) {
    return op->emitOpError(
        "expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  return success();
}

static std::optional<int64_t> getStaticElemCount(ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return std::nullopt;
    if (total > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    total *= dim;
  }
  return total;
}

struct TStoreVerifyInfo {
  pto::TileBufType srcTile;
  pto::PartitionTensorViewType dstPart;
  Type srcElem;
  Type dstElem;
};

static FailureOr<TStoreVerifyInfo> verifyTStoreCommon(Operation *op, Value src,
                                                      Value dst,
                                                      bool allowLowPrecision) {
  auto srcTile = dyn_cast<pto::TileBufType>(src.getType());
  auto dstPart = dyn_cast<pto::PartitionTensorViewType>(dst.getType());
  if (!srcTile || !dstPart) {
    op->emitOpError(
        "expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, srcTile, "src", allowLowPrecision)) ||
      failed(verifyPositiveStaticDims(op, dstPart.getShape(), "dst", "shape")) ||
      failed(
          verifyPositiveStaticDims(op, srcTile.getValidShape(), "src", "valid_shape"))) {
    return failure();
  }

  auto dstElemCount = getStaticElemCount(dstPart.getShape());
  auto srcValidElemCount = getStaticElemCount(srcTile.getValidShape());
  if (dstElemCount && srcValidElemCount && *dstElemCount != *srcValidElemCount) {
    op->emitOpError() << "expects dst static element count (" << *dstElemCount
                      << ") to match src valid_shape static element count ("
                      << *srcValidElemCount << ")";
    return failure();
  }
  return TStoreVerifyInfo{srcTile, dstPart, srcTile.getElementType(),
                          dstPart.getElementType()};
}

static bool isLoadStoreElemType(Type ty) {
  return ty.isInteger(kPTOI8BitWidth) || ty.isInteger(kPTOI16BitWidth) || ty.isInteger(kPTOI32BitWidth) ||
         ty.isInteger(kPTOI64BitWidth) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isI8Like(Type ty) { return ty.isInteger(kPTOI8BitWidth); }

static LogicalResult verifyTStoreModeSource(Operation *op,
                                            pto::AddressSpace srcSpace,
                                            bool hasPreQuant,
                                            pto::ReluPreMode reluMode) {
  if (hasPreQuant && srcSpace != pto::AddressSpace::ACC)
    return op->emitOpError("expects preQuantScalar form to use loc=acc src");
  if (reluMode != pto::ReluPreMode::NoRelu && srcSpace != pto::AddressSpace::ACC) {
    return op->emitOpError("expects reluPreMode form to use loc=acc src");
  }
  return success();
}

static LogicalResult verifyTStoreA2A3AccDstType(Operation *op, Type srcElem,
                                                Type dstElem, bool hasPreQuant) {
  if (hasPreQuant) {
    if (srcElem.isInteger(kPTOI32BitWidth)) {
      if (!(isI8Like(dstElem) || dstElem.isF16())) {
        return op->emitOpError(
            "expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
      }
    } else if (srcElem.isF32() && !isI8Like(dstElem)) {
      return op->emitOpError(
          "expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
    }
    return success();
  }
  if (!(dstElem.isInteger(kPTOI32BitWidth) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16())) {
    return op->emitOpError(
        "expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
  }
  return success();
}

static LogicalResult verifyTStoreA2A3(TStoreOp op, const TStoreVerifyInfo &info,
                                      bool hasPreQuant,
                                      pto::ReluPreMode reluMode) {
  auto srcSpace = getPTOMemorySpaceEnum(info.srcTile);
  if (!srcSpace ||
      (*srcSpace != pto::AddressSpace::VEC && *srcSpace != pto::AddressSpace::MAT &&
       *srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError(
        "expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
  }
  if (failed(verifyTStoreModeSource(op.getOperation(), *srcSpace, hasPreQuant,
                                    reluMode))) {
    return failure();
  }

  if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
    if (isPTOLowPrecisionType(info.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
    }
    if (!isLoadStoreElemType(info.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
    }
    if (getElemByteSize(info.srcElem) != getElemByteSize(info.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
    }
    return success();
  }

  if (!(info.srcElem.isInteger(kPTOI32BitWidth) || info.srcElem.isF32()))
    return op.emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
  if (failed(verifyTStoreA2A3AccDstType(op.getOperation(), info.srcElem,
                                        info.dstElem, hasPreQuant))) {
    return failure();
  }

  auto srcShape = info.srcTile.getShape();
  if (srcShape[kPTOColumnDim] != ShapedType::kDynamic &&
      (srcShape[kPTOColumnDim] < kPTOMatmulDimMin || srcShape[kPTOColumnDim] > kPTOMatmulDimMax)) {
    return op.emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
  }
  auto srcValid = info.srcTile.getValidShape();
  if (srcValid[kPTOColumnDim] != ShapedType::kDynamic &&
      (srcValid[kPTOColumnDim] < kPTOMatmulDimMin || srcValid[kPTOColumnDim] > kPTOMatmulDimMax)) {
    return op.emitOpError(
        "expects A2/A3 acc tstore src valid_shape[1] to be in [1, 4095]");
  }
  return success();
}

static LogicalResult verifyTStoreA5AccDstType(Operation *op, Type srcElem,
                                              Type dstElem, bool hasPreQuant) {
  if (hasPreQuant) {
    if (!isA5AccStorePreQuantDstType(srcElem, dstElem)) {
      return op->emitOpError(
          "expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
    }
    return success();
  }
  if (!(dstElem.isInteger(kPTOI32BitWidth) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16())) {
    return op->emitOpError(
        "expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
  }
  return success();
}

static LogicalResult verifyTStoreA5(TStoreOp op, const TStoreVerifyInfo &info,
                                    bool hasPreQuant,
                                    pto::ReluPreMode reluMode) {
  auto srcSpace = getPTOMemorySpaceEnum(info.srcTile);
  if (!srcSpace ||
      (*srcSpace != pto::AddressSpace::VEC && *srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
  }
  if (failed(verifyTStoreModeSource(op.getOperation(), *srcSpace, hasPreQuant,
                                    reluMode))) {
    return failure();
  }

  if (*srcSpace == pto::AddressSpace::VEC) {
    if (!isA5TLoadStoreTransferElemType(info.srcElem)) {
      return op.emitOpError(
          "expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    if (getElemByteSize(info.srcElem) != getElemByteSize(info.dstElem)) {
      return op.emitOpError(
          "expects A5 vec tstore src and dst element types to have the same bitwidth");
    }
    return success();
  }

  if (!(info.srcElem.isInteger(kPTOI32BitWidth) || info.srcElem.isF32()))
    return op.emitOpError("expects A5 acc tstore src element type to be i32 or f32");
  return verifyTStoreA5AccDstType(op.getOperation(), info.srcElem, info.dstElem,
                                  hasPreQuant);
}

