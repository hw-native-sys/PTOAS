// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyMisc.cpp; kept as a fragment included by PTOVerifyMisc.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyMatmulBiasLikeOp(*this, getA().getType(), getB().getType(),
                                  getBias().getType(), getDst().getType(),
                                  /*useGemvOperands=*/false);
  };
  auto verifyA5 = [&verifyA2A3]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyMatmulMxA2A3LikeOp(
        *this, getAScale().getType(), getBScale().getType(), getA().getType(),
        getB().getType(), getDst().getType(),
        []() -> LogicalResult { return success(); });
  };
  auto verifyA5 = [this, &verifyA2A3]() -> LogicalResult {
    return verifyMatmulMxA5LikeOp(*this, getA().getType(), getB().getType(),
                                  getDst().getType(), verifyA2A3);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")))
      return failure();
    return success();
  };
  auto verifyA5 = [this, &verifyA2A3]() -> LogicalResult {
    if (failed(verifyA2A3()))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyOperands = [this]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getA().getType(),
                                     getB().getType(), getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true)))
      return failure();
    return success();
  };
  auto verifyA2A3 = [this, &verifyOperands]() -> LogicalResult {
    return verifyMatmulMxA2A3LikeOp(
        *this, getAScale().getType(), getBScale().getType(), getA().getType(),
        getB().getType(), getDst().getType(), verifyOperands);
  };
  auto verifyA5 = [this, &verifyA2A3]() -> LogicalResult {
    return verifyMatmulMxA5LikeOp(*this, getA().getType(), getB().getType(),
                                  getDst().getType(), verifyA2A3);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType())
      return emitOpError("expects val type to match dst element type");
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType, MemRefType>(srcTy))
    return emitOpError("expects src to be tile_buf or memref type");

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace =
      isa<pto::TileBufType>(srcTy)
          ? cast<pto::TileBufType>(srcTy).getMemorySpace()
          : cast<MemRefType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT)
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType())
    return emitOpError("expects dst type to match src element type");
  return success();
}

static bool isIntegerTypeWidth(Type ty, unsigned width) {
  auto it = dyn_cast<IntegerType>(ty);
  return it && it.getWidth() == width;
}

static LogicalResult verifyTHistogramShapes(THistogramOp op, Type srcTy,
                                            Type idxTy, Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto idxShape = getShapeVec(idxTy);
  auto dstShape = getShapeVec(dstTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto idxValid = getValidShapeVec(idxTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != kPTORowColRank || idxShape.size() != kPTORowColRank || dstShape.size() != kPTORowColRank ||
      srcValid.size() != kPTORowColRank || idxValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank) {
    return op.emitOpError(
        "expects src, idx, and dst to have rank-2 shape and valid_shape");
  }
  if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], idxValid[0])) {
    return op.emitOpError("expects idx rows and valid rows to match src");
  }
  if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], dstValid[0])) {
    return op.emitOpError("expects dst rows and valid rows to match src");
  }
  if (!isKnownUnitExtent(idxShape[kPTOColumnDim]) || !isKnownUnitExtent(idxValid[kPTOColumnDim]))
    return op.emitOpError("expects idx to have exactly one column");
  if (dstShape[kPTOColumnDim] != ShapedType::kDynamic && dstShape[kPTOColumnDim] < kPTOMinGatherDstColumns)
    return op.emitOpError("expects dst shape[1] to be at least 256");
  if (dstValid[kPTOColumnDim] != ShapedType::kDynamic && dstValid[kPTOColumnDim] < kPTOMinGatherDstColumns)
    return op.emitOpError("expects dst valid_shape[1] to be at least 256");
  return success();
}

static LogicalResult verifyTHistogramA5(THistogramOp op) {
  Type srcTy = op.getSrc().getType();
  Type idxTy = op.getIdx().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, idxTy, "idx")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto idxSpace = getPTOMemorySpaceEnum(idxTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects src to be in the vec address space");
  if (!idxSpace || *idxSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects idx to be in the vec address space");
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects dst to be in the vec address space");

  auto srcTB = dyn_cast<pto::TileBufType>(srcTy);
  auto idxTB = dyn_cast<pto::TileBufType>(idxTy);
  auto dstTB = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTB || !idxTB || !dstTB)
    return op.emitOpError("expects src, idx, and dst to be tile_buf types");
  if (!hasTileBufLayout(srcTB, pto::BLayout::RowMajor, pto::SLayout::NoneBox))
    return op.emitOpError("expects src to use row_major + none_box layout");
  if (!hasTileBufLayout(dstTB, pto::BLayout::RowMajor, pto::SLayout::NoneBox))
    return op.emitOpError("expects dst to use row_major + none_box layout");
  if (!hasTileBufLayout(idxTB, pto::BLayout::ColMajor, pto::SLayout::NoneBox)) {
    return op.emitOpError(
        "expects idx to use DN layout (col_major + none_box)");
  }

  if (!isIntegerTypeWidth(getElemTy(srcTy), kPTOI16BitWidth))
    return op.emitOpError("expects src element type to be ui16");
  if (!isIntegerTypeWidth(getElemTy(idxTy), kPTOI8BitWidth))
    return op.emitOpError("expects idx element type to be ui8");
  if (!isIntegerTypeWidth(getElemTy(dstTy), kPTOI32BitWidth))
    return op.emitOpError("expects dst element type to be ui32");
  return verifyTHistogramShapes(op, srcTy, idxTy, dstTy);
}

LogicalResult THistogramOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTHistogramA5(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [this]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")))
      return failure();
    if (failed(verifyScaleTileMatchesOperand(*this, dstTy, srcTy, "dst", "src")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
