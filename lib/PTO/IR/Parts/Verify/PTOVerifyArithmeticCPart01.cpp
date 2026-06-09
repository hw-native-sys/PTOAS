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

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32()))
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    return success();
  };
  auto verifyA5 = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(kPTOI16BitWidth) || elem0.isInteger(kPTOI32BitWidth)))
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, MemRefType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch =
      [this, &isTileLike, &isScalarLike](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);
    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile))
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    if (!mlir::isa<IntegerType, FloatType>(scalarTy))
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isF16() ||
          elem.isF32()))
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(kPTOI32BitWidth) || elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI8BitWidth) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    return success();
  };
  auto verifyA2A3 = [&verifyByArch]() -> LogicalResult {
    return verifyByArch(PTOArch::A3);
  };
  auto verifyA5 = [&verifyByArch]() -> LogicalResult {
    return verifyByArch(PTOArch::A5);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false)))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32())
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&verifyA2A3]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<std::pair<Type, std::optional<pto::AddressSpace>>>
verifyTExpandsCommon(TExpandsOp op) {
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!dstSpace ||
      (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects dst to be in the vec or mat address space"),
           failure();
  }
  Type dstElem = getElemTy(dstTy);
  if (op.getScalar().getType() != dstElem)
    return op.emitOpError("expects scalar type == dst element type"), failure();
  return std::make_pair(dstElem, dstSpace);
}

static LogicalResult verifyTExpandsElemType(Operation *op, Type dstElem,
                                            StringRef error, bool allowI8) {
  if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())
    return success();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
    unsigned w = it.getWidth();
    if (w == kPTOI16BitWidth || w == kPTOI32BitWidth ||
        (allowI8 && w == kPTOI8BitWidth))
      return success();
  }
  return op->emitOpError(error);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    auto common = verifyTExpandsCommon(*this);
    if (failed(common))
      return failure();
    Type dstTy = getDst().getType();
    auto [dstElem, dstSpace] = *common;
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy))
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    return verifyTExpandsElemType(
        getOperation(), dstElem,
        "expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32",
        /*allowI8=*/false);
  };
  auto verifyA5 = [this]() -> LogicalResult {
    auto common = verifyTExpandsCommon(*this);
    if (failed(common))
      return failure();
    auto [dstElem, dstSpace] = *common;
    (void)dstSpace;
    return verifyTExpandsElemType(
        getOperation(), dstElem,
        "expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32",
        /*allowI8=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

struct IndexedTileTransferCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static bool hasTileBufLayout(pto::TileBufType ty, pto::BLayout bl,
                             pto::SLayout sl) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(bl) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(sl);
}

static bool hasMatExtractSourceLayoutA2A3(pto::TileBufType srcTy) {
  return srcTy.getBLayoutValueI32() ==
             static_cast<int32_t>(pto::BLayout::RowMajor) ||
         (srcTy.getBLayoutValueI32() !=
              static_cast<int32_t>(pto::BLayout::RowMajor) &&
          srcTy.getSLayoutValueI32() ==
              static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool hasMatExtractSourceLayoutA5(pto::TileBufType srcTy,
                                        pto::AddressSpace dstSpace) {
  const bool rowMajorSrc = srcTy.getBLayoutValueI32() ==
                           static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool colMajorView = srcTy.getSLayoutValueI32() ==
                            static_cast<int32_t>(pto::SLayout::ColMajor);
  const bool rowMajorView = srcTy.getSLayoutValueI32() ==
                            static_cast<int32_t>(pto::SLayout::RowMajor);
  if (dstSpace == pto::AddressSpace::LEFT)
    return (rowMajorSrc && colMajorView) || (!rowMajorSrc && rowMajorView) ||
           rowMajorSrc;
  return (rowMajorSrc && colMajorView) || (!rowMajorSrc && rowMajorView);
}

static bool isRowMajorNoneBoxNDTileBuf(pto::TileBufType ty) {
  return hasTileBufLayout(ty, pto::BLayout::RowMajor, pto::SLayout::NoneBox);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}
