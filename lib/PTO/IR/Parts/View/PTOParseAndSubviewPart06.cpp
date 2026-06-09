// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem)))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}

static bool shouldBypassTStoreFPVerifier(TStoreFPOp op) {
  Value src = op.getSrc();
  Value fp = op.getFp();
  return isa<MemRefType>(src.getType()) || isa<MemRefType>(fp.getType()) ||
         src.getDefiningOp<pto::BindTileOp>() ||
         fp.getDefiningOp<pto::BindTileOp>();
}

static LogicalResult verifyTStoreFPDstType(TStoreFPOp op) {
  Type dstTy = op.getDst().getType();
  if (!isa<MemRefType, pto::PartitionTensorViewType>(dstTy))
    return op.emitOpError()
           << "expects dst to be a memref or !pto.partition_tensor_view";
  if (auto dstPart = dyn_cast<pto::PartitionTensorViewType>(dstTy)) {
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        return op.emitOpError()
               << "expects dst shape[" << idx << "] to be positive";
      }
    }
  }
  return success();
}

static LogicalResult verifyTStoreFPTileOperands(TStoreFPOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  if (!isa<pto::TileBufType>(srcTy))
    return op.emitOpError() << "expects src to be a !pto.tile_buf";
  if (!isa<pto::TileBufType>(fpTy))
    return op.emitOpError() << "expects fp to be a !pto.tile_buf";
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")))
    return failure();
  if (failed(verifyTStoreFPDstType(op)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
    return op.emitOpError() << "expects src to be in the acc address space";
  return success();
}

static LogicalResult verifyTStoreFPA2A3Constraints(TStoreFPOp op) {
  Type srcTy = op.getSrc().getType();
  auto srcElemTy = getElemTy(srcTy);
  auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
  if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == kPTOI32BitWidth)))
    return op.emitOpError() << "expects src to have element type f32, i32";
  auto srcShape = getShapeVec(srcTy);
  if (srcShape.size() != kPTORowColRank)
    return op.emitOpError() << "expects src to have rank 2";
  if (srcShape[kPTOColumnDim] != ShapedType::kDynamic &&
      (srcShape[kPTOColumnDim] < kPTOMatmulDimMin || srcShape[kPTOColumnDim] > kPTOMatmulDimMax))
    return op.emitOpError() << "expects src.cols to be in the range [1, 4095]";
  auto srcValid = getValidShapeVec(srcTy);
  if (srcValid.size() != kPTORowColRank)
    return op.emitOpError() << "expects src to have a rank-2 valid_shape";
  if (srcValid[kPTOColumnDim] != ShapedType::kDynamic &&
      (srcValid[kPTOColumnDim] < kPTOMatmulDimMin || srcValid[kPTOColumnDim] > kPTOMatmulDimMax)) {
    return op.emitOpError()
           << "expects src.valid_shape[1] to be in the range [1, 4095]";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TStoreFPOp::verify() {
  if (shouldBypassTStoreFPVerifier(*this))
    return success();
  auto verifyA2A3 = [this]() -> LogicalResult {
    if (failed(verifyTStoreFPTileOperands(*this)))
      return failure();
    return verifyTStoreFPA2A3Constraints(*this);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTStoreFPTileOperands(*this);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}


mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size())
    return emitOpError() << "expects all tensors to have the same rank";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  return mlir::success();
}

struct TTransVerifyState {
  Type srcTy;
  Type dstTy;
  unsigned elemBytes;
};

static bool isSupportedTransposeElemType(Type type, unsigned elemBytes) {
  if (elemBytes == kPTOWordBytes)
    return type.isInteger(kPTOI32BitWidth) || type.isF32();
  if (elemBytes == kPTOHalfWordBytes)
    return type.isInteger(kPTOI16BitWidth) || type.isF16() || type.isBF16();
  return type.isInteger(kPTOI8BitWidth);
}

static FailureOr<TTransVerifyState>
verifyTTransCommon(TTransOp op, StringRef mismatchMessage) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = getElemTy(tmpTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem ||
      srcElem != tmpElem) {
    op.emitOpError() << mismatchMessage;
    return failure();
  }
  unsigned elemBytes = getPTOStorageElemByteSize(srcElem);
  if (elemBytes == 0) {
    op.emitOpError() << "failed to get transpose element size";
    return failure();
  }
  if (elemBytes != kPTOByteSize && elemBytes != kPTOHalfWordBytes &&
      elemBytes != kPTOWordBytes) {
    op.emitOpError()
        << "expects transpose element size to be 1, 2, or 4 bytes";
    return failure();
  }
  if (!isSupportedTransposeElemType(srcElem, elemBytes)) {
    op.emitOpError()
        << "expects transpose element type to match the supported set for its width";
    return failure();
  }
  return TTransVerifyState{srcTy, dstTy, elemBytes};
}
