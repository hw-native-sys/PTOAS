// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticC.cpp; kept as a fragment included by PTOVerifyArithmeticC.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyTileValidWidthMatchesCols(Operation *op, Type ty,
                                                     StringRef operandName) {
  auto validShape = getValidShapeVec(ty);
  auto shape = getShapeVec(ty);
  if (validShape.size() == kPTORowColRank && shape.size() == kPTORowColRank &&
      validShape[1] != ShapedType::kDynamic &&
      shape[1] != ShapedType::kDynamic && validShape[1] != shape[1]) {
    return op->emitOpError() << "expects " << operandName
                             << " valid_shape[1] to equal " << operandName
                             << " cols";
  }
  return success();
}

static FailureOr<GatherSrcDstCommon> verifyGatherSrcDstCommon(TGatherOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return op.emitOpError("failed to get element type for src/dst"), failure();
  return GatherSrcDstCommon{srcTy, dstTy, srcElem, dstElem};
}

static LogicalResult verifyGatherMaskForm(TGatherOp op,
                                          bool allowA5MaskTypes) {
  auto common = verifyGatherSrcDstCommon(op);
  if (failed(common))
    return failure();
  if (!isRowMajorTileBuf(common->srcTy) || !isRowMajorTileBuf(common->dstTy))
    return op.emitOpError("expects src and dst to use row-major layout");

  auto srcSpace = getPTOMemorySpaceEnum(common->srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(common->dstTy);
  if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src and dst to be in the vec address space");
  }

  unsigned srcElemBytes = getPTOStorageElemByteSize(common->srcElem);
  unsigned dstElemBytes = getPTOStorageElemByteSize(common->dstElem);
  if (srcElemBytes == 0 || dstElemBytes == 0)
    return op.emitOpError("failed to get element size for src/dst");
  if (srcElemBytes != dstElemBytes)
    return op.emitOpError("expects src and dst element sizes to match");
  if (failed(verifyTileValidWidthMatchesCols(op.getOperation(), common->dstTy,
                                             "dst"))) {
    return failure();
  }

  if (allowA5MaskTypes) {
    if (!(srcElemBytes == kPTOByteSize ||
          srcElemBytes == kPTOHalfWordBytes ||
          srcElemBytes == kPTOWordBytes)) {
      return op.emitOpError(
          "expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
    }
    if (!isSupportedGatherElemTypeA5(common->srcElem) ||
        !isSupportedGatherElemTypeA5(common->dstElem)) {
      return op.emitOpError(
          "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
    }
    return success();
  }

  if (!(srcElemBytes == kPTOHalfWordBytes || srcElemBytes == kPTOWordBytes)) {
    return op.emitOpError(
        "expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
  }
  return success();
}

static FailureOr<GatherIndexCommon> verifyGatherIndexCommon(TGatherOp op) {
  auto base = verifyGatherSrcDstCommon(op);
  if (failed(base))
    return failure();
  Type idxTy = op.getIndices().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, idxTy, "indices")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (base->srcElem != base->dstElem)
    return op.emitOpError("expects src and dst to have the same element type"),
           failure();
  auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
  if (!idxElem)
    return op.emitOpError("indices element type must be integer"), failure();
  return GatherIndexCommon{*base, idxTy, tmpTy, idxElem};
}

static LogicalResult verifyGatherIndexForm(TGatherOp op,
                                           bool allow16BitIndices,
                                           bool allowA5ElemTypes) {
  auto common = verifyGatherIndexCommon(op);
  if (failed(common))
    return failure();
  if (allowA5ElemTypes) {
    if (!isSupportedGatherElemTypeA5Index(common->base.srcElem) ||
        !isSupportedGatherElemTypeA5Index(common->base.dstElem)) {
      return op.emitOpError(
          "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    }
  } else if (!isSupportedGatherElemTypeA2A3(common->base.srcElem) ||
             !isSupportedGatherElemTypeA2A3(common->base.dstElem)) {
    return op.emitOpError(
        "expects gather src/dst element type to be i16/i32/f16/f32");
  }

  unsigned width = common->idxElem.getWidth();
  if (!(width == kPTOI32BitWidth ||
        (allow16BitIndices && width == kPTOI16BitWidth))) {
    return op.emitOpError() << "expects indices element type to be i32"
                            << (allow16BitIndices ? " or i16" : "");
  }
  if (failed(verifyTileValidWidthMatchesCols(op.getOperation(), common->base.dstTy,
                                             "dst")) ||
      failed(verifyTileValidWidthMatchesCols(op.getOperation(), common->idxTy,
                                             "indices"))) {
    return failure();
  }
  if (!allowA5ElemTypes) {
    if (getElemTy(common->tmpTy) != common->idxElem)
      return op.emitOpError(
          "expects tmp and indices to have the same element type");
    if (failed(verifyTileBufSameValidShape(op, common->idxTy, common->tmpTy,
                                           "indices", "tmp"))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<GatherCompareCommon> verifyGatherCompareCommon(TGatherOp op) {
  auto base = verifyGatherSrcDstCommon(op);
  if (failed(base))
    return failure();
  Type cdstTy = op.getCdst().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, cdstTy, "cdst")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type cdstElem = getElemTy(cdstTy);
  if (!cdstElem)
    return op.emitOpError("failed to get element type for src/dst/cdst"),
           failure();
  auto dstInt = dyn_cast<IntegerType>(base->dstElem);
  if (!dstInt || dstInt.getWidth() != kPTOI32BitWidth)
    return op.emitOpError("expects dst element type to be i32"), failure();
  if (cdstElem != base->dstElem)
    return op.emitOpError("expects cdst to have the same element type as dst"),
           failure();
  if (op.getKValue().getType() != base->srcElem) {
    return op.emitOpError(
               "expects kValue to have the same type as src element type"),
           failure();
  }
  auto cmpAttr = op.getCmpModeAttr();
  pto::CmpMode cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
  if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT) {
    return op.emitOpError(
               "expects compare-form tgather cmpMode to be eq or gt"),
           failure();
  }
  return GatherCompareCommon{*base, cdstTy, tmpTy, cdstElem, cmpMode};
}

static LogicalResult verifyGatherCompareForm(TGatherOp op,
                                             bool allowA5SrcTypes) {
  auto common = verifyGatherCompareCommon(op);
  if (failed(common))
    return failure();
  if (allowA5SrcTypes) {
    if (!(common->base.srcElem.isF16() || common->base.srcElem.isF32() ||
          common->base.srcElem.isInteger(kPTOI16BitWidth) ||
          common->base.srcElem.isInteger(kPTOI32BitWidth))) {
      return op.emitOpError(
          "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
    }
  } else if (!(common->base.srcElem.isF16() || common->base.srcElem.isF32() ||
               (common->base.srcElem.isInteger(kPTOI32BitWidth) &&
                common->cmpMode == pto::CmpMode::EQ))) {
    return op.emitOpError(
        "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
  }

  if (failed(verifyVecTileCommonA2A3(op, common->base.srcTy, "src")) ||
      failed(verifyVecTileCommonA2A3(op, common->base.dstTy, "dst")) ||
      failed(verifyVecTileCommonA2A3(op, common->cdstTy, "cdst")) ||
      failed(verifyVecTileCommonA2A3(op, common->tmpTy, "tmp"))) {
    return failure();
  }
  return success();
}
