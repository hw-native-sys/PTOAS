// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true))) {
      return failure();
    }
    if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
      return emitOpError("scalar must be a scalar type (integer/float)");
    }
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    }
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyF16F32VecUnary(Operation *op, Type srcTy,
                                          Type dstTy) {
  if (failed(verifyVecTileUnaryOp(op, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false,
                                  /*allowInt8=*/false)) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elem = getElemTy(srcTy);
  if (!elem.isF16() && !elem.isF32())
    return op->emitOpError("expects element type to be f16 or f32");
  return success();
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() {
    return verifyF16F32VecUnary(getOperation(), getSrc().getType(),
                                getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTExpandsArch(TExpandsOp op, bool isA5) {
    Type dstTy = op.getDst().getType();
    if (failed(verifyTileBufCommon(op, dstTy, "dst"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return op.emitOpError("expects dst to be in the vec or mat address space");
    }
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = op.getScalar().getType();
    if (scalarTy != dstElem) {
      return op.emitOpError("expects scalar type == dst element type");
    }
    if (!isA5 && *dstSpace == pto::AddressSpace::VEC &&
        !isRowMajorTileBuf(dstTy)) {
      return op.emitOpError("expects vec dst to use row-major layout on A2/A3");
    }
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()) {
      return mlir::success();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if ((isA5 && w == mlir::pto::kValue8) || w == mlir::pto::kValue16 || w == mlir::pto::kValue32) {
          return mlir::success();
      }
    }
    return op.emitOpError(isA5
        ? "expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32"
        : "expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTExpandsArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTExpandsArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isA2A3AccCastExtractTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
      return dstElem.isInteger(mlir::pto::kValue8);
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isInteger(mlir::pto::kValue16);
  }
  return false;
}

static bool isA2A3AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  return isA2A3AccQuantTypePair(srcElem, dstElem);
}

static bool isA5AccCastExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue32);
  }
  return false;
}

static bool isA5AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32() ||
             (llvm::isa<FloatType>(dstElem) && llvm::cast<FloatType>(dstElem).getWidth() == mlir::pto::kValue8);
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static bool hasMatExtractSourceLayoutA2A3(pto::TileBufType srcTy) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool hasMatExtractSourceLayoutA5(pto::TileBufType srcTy,
                                        pto::AddressSpace dstSpace) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  if (dstSpace == pto::AddressSpace::LEFT) {
    return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor)) ||
           bl == static_cast<int32_t>(pto::BLayout::RowMajor);
  }
  return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool isA2A3ExtractElemType(Type ty) {
    return ty.isInteger(mlir::pto::kValue8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5ExtractElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
      return it.getWidth() == mlir::pto::kValue8;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
      return ft.getWidth() == mlir::pto::kValue8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isRowMajorNoneBoxND(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
}

struct TileTransferCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TileTransferCommon> verifyTileTransferCommon(
    Operation *op, Value src, Value dst, Value indexRow, Value indexCol,
    bool allowLowPrecision, bool includeIndexAndIntOpsInConstFold,
    bool insert) {
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb)
    return op->emitOpError("expects src and dst to be !pto.tile_buf");
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
      failed(verifyNonNegativeIndexRowCol(
          *op, indexRow, indexCol, includeIndexAndIntOpsInConstFold)))
    return failure();
  LogicalResult bounds =
      insert ? verifyInsertStaticBoundsCommon(
                   *op, indexRow, indexCol, srcTy, dstTy,
                   includeIndexAndIntOpsInConstFold)
             : verifyExtractStaticBoundsCommon(
                   *op, indexRow, indexCol, srcTy, dstTy,
                   includeIndexAndIntOpsInConstFold);
  if (failed(bounds))
    return failure();
  return TileTransferCommon{srcTy,
                            dstTy,
                            srcTb,
                            dstTb,
                            getElemTy(srcTy),
                            getElemTy(dstTy),
                            getPTOMemorySpaceEnum(srcTy),
                            getPTOMemorySpaceEnum(dstTy)};
}

using TExtractCommon = TileTransferCommon;

static FailureOr<TExtractCommon> verifyTExtractCommon(TExtractOp op,
                                                      bool allowLowPrecision) {
  const bool hasFp = static_cast<bool>(op.getFp());
  auto common = verifyTileTransferCommon(
      op, op.getSrc(), op.getDst(), op.getIndexRow(), op.getIndexCol(),
      allowLowPrecision, /*includeIndexAndIntOpsInConstFold=*/hasFp,
      /*insert=*/false);
  if (failed(common))
    return failure();
  if (hasFp) {
    Type fpTy = op.getFp().getType();
    if (failed(verifyTileBufCommon(op, fpTy, "fp", allowLowPrecision))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to use loc=scaling");
    }
  }
  if (!common->srcElem || !common->dstElem) {
    return op.emitOpError("expects src and dst to have element types");
  }
  if ((!common->srcSpace || *common->srcSpace != pto::AddressSpace::ACC) &&
      common->srcElem != common->dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  return *common;
}

static LogicalResult
verifyTExtractFpFormLoc(TExtractOp op,
                        std::optional<pto::AddressSpace> srcSpace) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("expects fp and preQuantScalar to be mutually exclusive");
  }
  if (hasFp && !srcIsAcc) {
    return op.emitOpError("expects fp form to use loc=acc src");
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && !srcIsAcc) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  return success();
}
