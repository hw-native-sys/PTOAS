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

static LogicalResult verifyTInsertA5AccToMat(const IndexedTileTransferCommon &common,
                                             TInsertOp op) {
  if (!isColMajorRowMajorNZTileBuf(common.srcTb))
    return op.emitOpError(
        "expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
  if (!isColMajorRowMajorNZTileBuf(common.dstTb))
    return op.emitOpError(
        "expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
  const bool okTypes =
      (common.srcElem.isF32() &&
       (common.dstElem.isF16() || common.dstElem.isBF16() ||
        common.dstElem.isF32())) ||
      (common.srcElem.isInteger(kPTOI32BitWidth) && common.dstElem.isInteger(kPTOI32BitWidth));
  if (!okTypes) {
    return op.emitOpError(
        "expects A5 acc->mat tinsert element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecToMat(const IndexedTileTransferCommon &common,
                                             TInsertOp op) {
  if (!isColMajorRowMajorNZTileBuf(common.dstTb)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
  }
  const bool srcIsND = isRowMajorNoneBoxNDTileBuf(common.srcTb);
  const bool srcIsNZ = isColMajorRowMajorNZTileBuf(common.srcTb);
  if (!srcIsND && !srcIsNZ) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
  }
  if (common.srcElem != common.dstElem ||
      !isA5SupportedVecInsertElemType(common.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecToVec(const IndexedTileTransferCommon &common,
                                             TInsertOp op) {
  if (!isRowMajorNoneBoxNDTileBuf(common.srcTb) ||
      !isRowMajorNoneBoxNDTileBuf(common.dstTb)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to use ND layout "
        "(blayout=row_major, slayout=none_box)");
  }
  if (common.srcElem != common.dstElem ||
      !isA5SupportedVecInsertElemType(common.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5(const IndexedTileTransferCommon &common,
                                     TInsertOp op) {
  if (!common.srcSpace || !common.dstSpace)
    return op.emitOpError("expects A5 tinsert src/dst to have explicit loc");

  if (*common.srcSpace == pto::AddressSpace::ACC &&
      *common.dstSpace == pto::AddressSpace::MAT)
    return verifyTInsertA5AccToMat(common, op);

  if (*common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::MAT)
    return verifyTInsertA5VecToMat(common, op);

  if (*common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC)
    return verifyTInsertA5VecToVec(common, op);

  return op.emitOpError(
      "expects A5 tinsert to use a supported src/dst loc pair: "
      "acc->mat, vec->mat, or vec->vec");
}

mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto common = verifyIndexedTileTransferCommon(
      getOperation(), getSrc(), getDst(), getIndexRow(), getIndexCol(),
      /*includeIndexAndIntOpsInConstFold=*/true, /*isInsertOp=*/true,
      /*requireSameElementType=*/false);
  if (failed(common))
    return failure();

  auto verifyA2A3 = [this, &common]() -> LogicalResult {
    return verifyTInsertA2A3(*common, *this);
  };
  auto verifyA5 = [this, &common]() -> LogicalResult {
    return verifyTInsertA5(*common, *this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isA2A3VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(kPTOI8BitWidth);
  if (srcElem.isInteger(kPTOI32BitWidth))
    return dstElem.isInteger(kPTOI8BitWidth) || dstElem.isF16() || dstElem.isInteger(kPTOI16BitWidth);
  return false;
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == kPTOI8BitWidth;
  return false;
}

static bool isA5MxInputType(Type ty) {
  return isA5Fp8LikeType(ty);
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputType(lhsElem) || !isA5MxInputType(rhsElem))
    return op->emitOpError()
           << "expects A5 mx operands " << lhsName << " and " << rhsName
           << " to use fp8 element types";

  if (!dstElem.isF32())
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(kPTOI8BitWidth) || isA5Fp8LikeType(dstElem) || dstElem.isF16() ||
           dstElem.isBF16() || dstElem.isF32();
  if (srcElem.isInteger(kPTOI32BitWidth))
    return dstElem.isInteger(kPTOI8BitWidth) || dstElem.isF16() || dstElem.isBF16();
  return false;
}

static FailureOr<std::tuple<Type, Type, Type, pto::TileBufType, pto::TileBufType,
                            pto::TileBufType, pto::AddressSpace,
                            pto::AddressSpace, pto::AddressSpace>>
verifyVectorPreQuantTransferCommon(Operation *op, Value src, Value fp, Value dst,
                                   Value indexRow, Value indexCol,
                                   bool isInsertOp) {
  Type srcTy = src.getType();
  Type fpTy = fp.getType();
  Type dstTy = dst.getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !fpTb || !dstTb)
    return op->emitOpError("expects src, fp, and dst to be !pto.tile_buf"),
           failure();
  auto verifyBounds = isInsertOp ? verifyInsertStaticBoundsCommon
                                 : verifyExtractStaticBoundsCommon;
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyNonNegativeIndexRowCol(
          *op, indexRow, indexCol,
          /*includeIndexAndIntOpsInConstFold=*/true)) ||
      failed(verifyBounds(*op, indexRow, indexCol, srcTy, dstTy,
                          /*includeIndexAndIntOpsInConstFold=*/true)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto fpSpace = getPTOMemorySpaceEnum(fpTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !fpSpace || !dstSpace)
    return op->emitOpError("expects src, fp, and dst to have explicit loc"),
           failure();
  if (*srcSpace != pto::AddressSpace::ACC)
    return op->emitOpError("expects src to use loc=acc"), failure();
  if (*fpSpace != pto::AddressSpace::SCALING)
    return op->emitOpError("expects fp to use loc=scaling"), failure();
  if (*dstSpace != pto::AddressSpace::MAT)
    return op->emitOpError("expects dst to use loc=mat"), failure();
  if (!isColMajorRowMajorNZTileBuf(srcTb))
    return op->emitOpError(
               "expects src to use blayout=col_major and slayout=row_major"),
           failure();
  if (!isColMajorRowMajorNZTileBuf(dstTb))
    return op->emitOpError(
               "expects dst to use blayout=col_major and slayout=row_major"),
           failure();
  return std::make_tuple(srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, *srcSpace,
                         *fpSpace, *dstSpace);
}

using VectorPreQuantTypePairFn = bool (*)(Type, Type);

static LogicalResult verifyVectorPreQuantTransferOp(
    Operation *op, Value src, Value fp, Value dst, Value indexRow,
    Value indexCol, bool isInsertOp, bool requireDstFractal512,
    VectorPreQuantTypePairFn verifyTypePair, llvm::StringRef message) {
  auto common = verifyVectorPreQuantTransferCommon(op, src, fp, dst, indexRow,
                                                   indexCol, isInsertOp);
  if (failed(common))
    return failure();
  auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
      *common;
  (void)fpTy;
  (void)srcTb;
  (void)fpTb;
  (void)srcSpace;
  (void)fpSpace;
  (void)dstSpace;
  if (requireDstFractal512 && dstTb.getSFractalSizeI32() != kFractalSize512)
    return op->emitOpError("expects dst fractal size to be 512");
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!verifyTypePair(srcElem, dstElem))
    return op->emitOpError(message);
  return success();
}
