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

static LogicalResult verifyTGatherForArch(TGatherOp op, bool allowA5Forms) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue())
      return op.emitOpError(
          "mask-pattern tgather only allows src and dst operands");
    return verifyGatherMaskForm(op, /*allowA5MaskTypes=*/allowA5Forms);
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp())
      return op.emitOpError(
          "compare-form tgather expects dst, cdst, kValue, and tmp");
    if (op.getIndices())
      return op.emitOpError("compare-form tgather does not take indices");
    return verifyGatherCompareForm(op, /*allowA5SrcTypes=*/allowA5Forms);
  }
  if (!op.getIndices() || !op.getTmp())
    return op.emitOpError("index-form tgather expects both indices and tmp");
  return verifyGatherIndexForm(op, /*allow16BitIndices=*/allowA5Forms,
                               /*allowA5ElemTypes=*/allowA5Forms);
}

llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    return verifyTGatherForArch(*this, false);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    return verifyTGatherForArch(*this, true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyCommon = [this]() -> FailureOr<std::pair<Type, Type>> {
    Type srcTy = getSrc().getType();
    Type offTy = getOffsets().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, offTy, "offsets")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto dstElemTy = getElemTy(dstTy);
    if (!srcElemTy || !dstElemTy)
      return emitOpError() << "failed to get element type for src/dst";
    return std::make_pair(srcElemTy, dstElemTy);
  };

  auto getElemBytes = [](Type ty) -> std::optional<unsigned> {
    unsigned elemBytes = getPTOStorageElemByteSize(ty);
    if (elemBytes == 0)
      return std::nullopt;
    return elemBytes;
  };

  auto verifyA2A3 = [this, &getElemBytes, &verifyCommon]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstTy = getDst().getType();
    Type dstElemTy = elems->second;
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError() << "expects dst to use row-major layout";
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  auto verifyA5 = [this, &getElemBytes, &verifyCommon]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstElemTy = elems->second;
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [this, srcTy, dstTy]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != kPTORowColRank)
      return emitOpError("expects src to have rank-2 valid_shape");
    if (valid[0] != ShapedType::kDynamic && valid[0] <= 0)
      return emitOpError("expects src valid_shape[0] to be positive");
    if (valid[1] != ShapedType::kDynamic && valid[1] <= 0)
      return emitOpError("expects src valid_shape[1] to be positive");
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    return success();
  };
  auto verifyA5 = [this, srcTy, dstTy]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    if (!getSlope().getType().isF32())
      return emitOpError() << "expects slope to have type f32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmax element type to be i32/i16/f16/f32",
      "expects A5 tmax element type to be i32/i16/i8/f16/f32");
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmaxs element type to be i32/i16/f16/f32",
      "expects A5 tmaxs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmin element type to be i32/i16/f16/f32",
      "expects A5 tmin element type to be i32/i16/i8/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmins element type to be i32/i16/f16/f32",
      "expects A5 tmins element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}

struct TMovCommonInfo {
  Type srcTy;
  Type dstTy;
  Value fp;
  TileBufType srcTb;
  TileBufType dstTb;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
  bool hasFp = false;
  bool hasPreQuantScalar = false;
  bool isMatToTile = false;
  bool isVecToVec = false;
  bool isVecToMat = false;
  bool isAccToMat = false;
  bool isAccToVec = false;
};

struct TMovFpCommonInfo {
  Type srcTy;
  Type fpTy;
  Type dstTy;
  Type srcElemTy;
  TileBufType srcTb;
  TileBufType dstTb;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> fpSpace;
  std::optional<pto::AddressSpace> dstSpace;
};
