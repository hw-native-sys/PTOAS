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

static bool isA2A3ExtractElemType(Type ty) {
  return ty.isInteger(kPTOI8BitWidth) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5ExtractElemType(Type ty) {
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == kPTOI8BitWidth;
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == kPTOI8BitWidth || ft.isF16() || ft.isBF16() || ft.isF32();
  return false;
}

static bool isA2A3VecInsertElemType(Type ty) {
  return ty.isInteger(kPTOI8BitWidth) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5SupportedVecInsertElemType(Type ty) {
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == kPTOI8BitWidth || it.getWidth() == kPTOI32BitWidth;
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == kPTOI8BitWidth || ft.isF16() || ft.isBF16() || ft.isF32();
  return false;
}

static FailureOr<IndexedTileTransferCommon> verifyIndexedTileTransferCommon(
    Operation *op, Value src, Value dst, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold, bool isInsertOp,
    bool requireSameElementType) {
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb)
    return op->emitOpError("expects src and dst to be !pto.tile_buf"), failure();

  auto verifyBounds =
      isInsertOp ? verifyInsertStaticBoundsCommon : verifyExtractStaticBoundsCommon;
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyNonNegativeIndexRowCol(
          *op, indexRow, indexCol, includeIndexAndIntOpsInConstFold)) ||
      failed(verifyBounds(*op, indexRow, indexCol, srcTy, dstTy,
                          includeIndexAndIntOpsInConstFold))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (requireSameElementType && (!srcElem || !dstElem || srcElem != dstElem))
    return op->emitOpError("expects src and dst to have the same element type"),
           failure();

  return IndexedTileTransferCommon{
      srcTy, dstTy, srcTb, dstTb, srcElem, dstElem,
      getPTOMemorySpaceEnum(srcTy), getPTOMemorySpaceEnum(dstTy)};
}

static LogicalResult verifyTExtractA2A3(const IndexedTileTransferCommon &common,
                                        TExtractOp op) {
  if (!isA2A3ExtractElemType(common.dstElem))
    return op.emitOpError(
        "expects A2/A3 textract element type to be i8/f16/bf16/f32");
  if (common.srcSpace && common.dstSpace &&
      *common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC) {
    return success();
  }
  if (!common.srcSpace || *common.srcSpace != pto::AddressSpace::MAT)
    return op.emitOpError("expects A2/A3 textract src to use loc=mat or vec");
  if (!common.dstSpace || (*common.dstSpace != pto::AddressSpace::LEFT &&
                           *common.dstSpace != pto::AddressSpace::RIGHT)) {
    return op.emitOpError(
        "expects A2/A3 textract dst to use loc=left, loc=right, or loc=vec");
  }
  if (!hasMatExtractSourceLayoutA2A3(common.srcTb))
    return op.emitOpError(
        "expects A2/A3 textract src to use a supported mat blayout/slayout combination");
  if (*common.dstSpace == pto::AddressSpace::LEFT &&
      !hasTileBufLayout(common.dstTb, pto::BLayout::RowMajor,
                        pto::SLayout::RowMajor)) {
    return op.emitOpError(
        "expects A2/A3 left dst to use row_major blayout and row_major slayout");
  }
  if (*common.dstSpace == pto::AddressSpace::RIGHT &&
      !hasTileBufLayout(common.dstTb, pto::BLayout::RowMajor,
                        pto::SLayout::ColMajor)) {
    return op.emitOpError(
        "expects A2/A3 right dst to use row_major blayout and col_major slayout");
  }
  return success();
}

static LogicalResult verifyTExtractA5Pair(const IndexedTileTransferCommon &common,
                                          TExtractOp op) {
  const bool okPair =
      (*common.srcSpace == pto::AddressSpace::MAT &&
       (*common.dstSpace == pto::AddressSpace::LEFT ||
        *common.dstSpace == pto::AddressSpace::RIGHT ||
        *common.dstSpace == pto::AddressSpace::SCALING)) ||
      (*common.srcSpace == pto::AddressSpace::VEC &&
       (*common.dstSpace == pto::AddressSpace::MAT ||
        *common.dstSpace == pto::AddressSpace::VEC));
  if (!okPair) {
    return op.emitOpError(
        "expects A5 textract to use a supported src/dst loc pair");
  }
  return success();
}

static LogicalResult verifyTExtractA5Layouts(const IndexedTileTransferCommon &common,
                                             TExtractOp op) {
  if (*common.srcSpace == pto::AddressSpace::MAT) {
    if (!hasMatExtractSourceLayoutA5(common.srcTb, *common.dstSpace)) {
      return op.emitOpError(
          "expects A5 textract src to use a supported mat blayout/slayout combination");
    }
    if (*common.dstSpace == pto::AddressSpace::LEFT &&
        !hasTileBufLayout(common.dstTb, pto::BLayout::ColMajor,
                          pto::SLayout::RowMajor)) {
      return op.emitOpError(
          "expects A5 left dst to use col_major blayout and row_major slayout");
    }
    if (*common.dstSpace == pto::AddressSpace::RIGHT &&
        !hasTileBufLayout(common.dstTb, pto::BLayout::RowMajor,
                          pto::SLayout::ColMajor)) {
      return op.emitOpError(
          "expects A5 right dst to use row_major blayout and col_major slayout");
    }
    return success();
  }
  if (*common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC &&
      (!isRowMajorNoneBoxNDTileBuf(common.srcTb) ||
       !isRowMajorNoneBoxNDTileBuf(common.dstTb))) {
    return op.emitOpError(
        "expects A5 vec->vec textract src/dst to use ND layout "
        "(blayout=row_major, slayout=none_box)");
  }
  return success();
}

static LogicalResult verifyTExtractA5(const IndexedTileTransferCommon &common,
                                      TExtractOp op) {
  if (!isA5ExtractElemType(common.dstElem))
    return op.emitOpError(
        "expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
  if (!common.srcSpace || !common.dstSpace)
    return op.emitOpError("expects src and dst to have explicit loc");
  if (failed(verifyTExtractA5Pair(common, op)) ||
      failed(verifyTExtractA5Layouts(common, op)))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto common = verifyIndexedTileTransferCommon(
      getOperation(), getSrc(), getDst(), getIndexRow(), getIndexCol(),
      /*includeIndexAndIntOpsInConstFold=*/false, /*isInsertOp=*/false,
      /*requireSameElementType=*/true);
  if (failed(common))
    return failure();

  auto verifyA2A3 = [this, &common]() -> LogicalResult {
    return verifyTExtractA2A3(*common, *this);
  };
  auto verifyA5 = [this, &common]() -> LogicalResult {
    return verifyTExtractA5(*common, *this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTInsertA2A3(const IndexedTileTransferCommon &common,
                                       TInsertOp op) {
  if (common.srcSpace && common.dstSpace &&
      *common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC) {
    if (common.srcElem != common.dstElem ||
        !isA2A3VecInsertElemType(common.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
          "(i8/f16/bf16/f32)");
    }
    return success();
  }
  if (!common.srcSpace || !common.dstSpace ||
      *common.srcSpace != pto::AddressSpace::ACC ||
      *common.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");
  }
  if (!isColMajorRowMajorNZTileBuf(common.srcTb))
    return op.emitOpError(
        "expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
  if (!isColMajorRowMajorNZTileBuf(common.dstTb))
    return op.emitOpError(
        "expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
  if (common.dstTb.getSFractalSizeI32() != kFractalSize512)
    return op.emitOpError("expects A2/A3 tinsert dst fractal size to be 512");
  if (!(common.srcElem.isF32() &&
        (common.dstElem.isF16() || common.dstElem.isBF16()))) {
    return op.emitOpError(
        "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
  }
  return success();
}
