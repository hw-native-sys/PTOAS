// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", allowLowPrecision)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }
  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] || lhsShape[1] != rhsShape[0])) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }
  return verifyMatmulValidSizes(op, lhsTy, rhsTy, 0);
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                       allowLowPrecision))) {
    return failure();
  }

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb) {
    return success();
  }

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  }
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects dst to use the col_major blayout on A5");
  }

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  }
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  }
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  }
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           bool allowLowPrecision) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                     allowLowPrecision);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                   allowLowPrecision);
  }
  return failure();
}

static LogicalResult verifyGemvValidShapes(Operation *op, Type lhsTy,
                                           Type rhsTy, Type dstTy,
                                           bool dstMayBeNonTile) {
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1)
    return op->emitOpError(
        "expects lhs valid_shape[0] to be 1 for tgemv");
  if ((!dstMayBeNonTile || isa<pto::TileBufType>(dstTy)) &&
      dstValid[0] != ShapedType::kDynamic && dstValid[0] != 1)
    return op->emitOpError(
        "expects dst valid_shape[0] to be 1 for tgemv");
  if (lhsValid[1] != ShapedType::kDynamic &&
      rhsValid[0] != ShapedType::kDynamic && lhsValid[1] != rhsValid[0])
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  if (rhsValid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic && rhsValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  return success();
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace) {
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT) {
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");
  }

  return verifyGemvValidShapes(op, lhsTy, rhsTy, dstTy,
                               /*dstMayBeNonTile=*/true);
}
static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy))) {
    return failure();
  }
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyA5MxMatTileOperands(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true))) {
    return failure();
  }

  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  if (lhsShape.size() == mlir::pto::kValue2 && rhsShape.size() == mlir::pto::kValue2) {
      int64_t lhsK = lhsShape[1];
      int64_t rhsK = rhsShape[0];
      auto checkPhysicalK = [&](int64_t value, StringRef name) -> LogicalResult {
          if (value != ShapedType::kDynamic && (value < 1 || (value % mlir::pto::kValue64) != 0)) {
              return op->emitOpError() << "expects " << name
                                       << " physical K shape to be a positive multiple of 64 on A5";
          }
          return success();
      };
      if (failed(checkPhysicalK(lhsK, "lhs"))) {
          return failure();
      }
      if (failed(checkPhysicalK(rhsK, "rhs"))) {
          return failure();
      }
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > mlir::pto::kValue4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > mlir::pto::kValue4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > mlir::pto::kValue4095))) {
        return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
    }
  }
  return success();
}

static int64_t ceilDivKnown(int64_t value, int64_t divisor) {
  if (divisor == 0 || divisor < 0) {
    return ShapedType::kDynamic;
  }
  if (value == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }
  return (value + divisor - 1) / divisor;
}

static LogicalResult verifyA5MxMatScaleDims(
    Operation *op, StringRef scaleName, ArrayRef<int64_t> scaleDims,
    ArrayRef<int64_t> lhsDims, ArrayRef<int64_t> rhsDims, StringRef dimsName,
    bool isLeftScale) {
    if (scaleDims.size() != mlir::pto::kValue2 || lhsDims.size() != mlir::pto::kValue2 ||
        rhsDims.size() != mlir::pto::kValue2) {
        return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 " << dimsName;
    }
  int64_t scaleK = ceilDivKnown(lhsDims[1], 32);
  int64_t expectedRows = isLeftScale ? lhsDims[0] : scaleK;
  int64_t expectedCols = isLeftScale ? scaleK : rhsDims[1];
  if (!hasCompatibleKnownExtent(scaleDims[0], expectedRows) ||
      !hasCompatibleKnownExtent(scaleDims[1], expectedCols)) {
    return op->emitOpError()
           << "expects " << scaleName << " " << dimsName << " to be "
           << (isLeftScale ? "[M, ceil(K/32)]" : "[ceil(K/32), N]");
  }
  return success();
}

static LogicalResult verifyA5MxMatScaleLayout(Operation *op,
                                              pto::TileBufType scaleTb,
                                              StringRef scaleName,
                                              bool isLeftScale) {
  if (scaleTb.getBLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::BLayout::RowMajor
                                       : pto::BLayout::ColMajor)) {
    return op->emitOpError() << "expects " << scaleName << " to use the "
                             << (isLeftScale ? "row_major" : "col_major")
                             << " blayout on A5";
  }
  if (scaleTb.getSLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::SLayout::RowMajor
                                       : pto::SLayout::ColMajor)) {
    return op->emitOpError() << "expects " << scaleName << " to use the "
                             << (isLeftScale ? "row_major" : "col_major")
                             << " slayout on A5";
  }
  if (scaleTb.getSFractalSizeI32() != mlir::pto::kValue32) {
      return op->emitOpError() << "expects " << scaleName << " to use fractal=32 on A5";
  }
  return success();
}

static LogicalResult verifyA5MxScaleStorage(Operation *op, Type scaleTy,
                                            StringRef scaleName) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError()
           << "expects " << scaleName << " to be in the scaling address space";
  return success();
}
