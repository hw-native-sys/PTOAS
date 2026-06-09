// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticA.cpp; kept as a fragment included by PTOVerifyArithmeticA.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != kPTORowColRank || idxValid.size() != kPTORowColRank)
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile)
    return op->emitOpError("expects idx to be a tile_buf type");

  const bool idxRowMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool idxColMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::ColMajor);

  const bool rowCoalesce1xR =
      idxRowMajor && isKnownUnitExtent(idxValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[0]);
  const bool rowCoalesceRx1 =
      idxColMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownUnitExtent(idxValid[1]);
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);
  if (!(rowCoalesce1xR || rowCoalesceRx1 || elemCoalesce))
    return op->emitOpError()
           << "expects idx valid_shape to be [1, " << dataName
           << ".valid_row], [" << dataName
           << ".valid_row, 1], or match " << dataName << " valid_shape";

  return success();
}

static LogicalResult verifyMGatherMScatterIdxTile(Operation *op, Type ty,
                                                  StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in the vec address space";
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb)
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(kPTOI8BitWidth) || ty.isInteger(kPTOI16BitWidth) || ty.isInteger(kPTOI32BitWidth) ||
         ty.isInteger(kPTOI64BitWidth) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(kPTOI32BitWidth))
    return dstElem.isInteger(kPTOI8BitWidth) || dstElem.isF16() || dstElem.isBF16();
  if (!srcElem.isF32())
    return false;
  return dstElem.isInteger(kPTOI8BitWidth) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         dstElem.isFloat8E4M3() || dstElem.isFloat8E4M3FN() ||
         dstElem.isFloat8E4M3FNUZ() || dstElem.isFloat8E4M3B11FNUZ();
}

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  if (srcElem.isF16())
    return isPTOHiFloat8Type(dstElem);
  if (srcElem.isBF16())
    return isPTOFloat4PackedType(dstElem);
  if (isPTOFloat4PackedType(srcElem))
    return dstElem.isBF16();
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem))
    return dstElem.isF32();
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  return true;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != kPTORowColRank)
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
    Type elemTy = tb.getElementType();
    if (!allowLowPrecision && isPTOLowPrecisionType(elemTy))
      return op->emitOpError() << name << ": dtype " << elemTy
                               << " is not supported by this op yet";
  } else if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (mr.getRank() != kPTORowColRank)
      return op->emitOpError() << "expects " << name << " to be a rank-2 memref";
    if (!allowLowPrecision && isPTOLowPrecisionType(mr.getElementType()))
      return op->emitOpError() << name << ": dtype " << mr.getElementType()
                               << " is not supported by this op yet";
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf or rank-2 memref";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != kPTORowColRank)
    return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < kPTORowColRank; ++i) {
    if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic &&
        validShape[i] > shape[i])
      return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i
                               << "] <= shape[" << i << "]";
  }
  return success();
}

static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf or memref";
  if (getElemTy(lhs) != getElemTy(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i])
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
  }
  if (lhsValid.size() != rhsValid.size())
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  return success();
}

static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();

  auto lhsExtent = getLogicalTileExtentVec(lhs, compareValidShape);
  auto rhsExtent = getLogicalTileExtentVec(rhs, compareValidShape);
  auto emitMismatch = [compareValidShape, lhsName, op,
                       rhsName]() -> LogicalResult {
    if (compareValidShape)
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have compatible shapes";
  };
  if (lhsExtent.size() != rhsExtent.size())
    return emitMismatch();

  for (size_t i = 0, e = lhsExtent.size(); i < e; ++i) {
    if (lhsExtent[i] != ShapedType::kDynamic &&
        rhsExtent[i] != ShapedType::kDynamic && lhsExtent[i] != rhsExtent[i])
      return emitMismatch();
  }
  return success();
}

static LogicalResult verifyScaleTileMatchesOperand(Operation *op, Type scaleTy,
                                                   Type operandTy,
                                                   StringRef scaleName,
                                                   StringRef operandName) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";

  auto scaleShape = getShapeVec(scaleTy);
  auto operandShape = getShapeVec(operandTy);
  if (scaleShape.size() != operandShape.size())
    return op->emitOpError() << "expects " << scaleName << " and " << operandName
                             << " to have the same rank";
  for (size_t i = 0; i < scaleShape.size(); ++i) {
    if (scaleShape[i] != ShapedType::kDynamic &&
        operandShape[i] != ShapedType::kDynamic &&
        scaleShape[i] != operandShape[i])
      return op->emitOpError() << "expects " << scaleName << " and " << operandName
                               << " to have the same shape";
  }

  auto scaleValid = getValidShapeVec(scaleTy);
  auto operandValid = getValidShapeVec(operandTy);
  if (scaleValid.size() != operandValid.size())
    return op->emitOpError() << "expects " << scaleName << " and " << operandName
                             << " to have the same valid_shape";
  for (size_t i = 0; i < scaleValid.size(); ++i) {
    if (scaleValid[i] != ShapedType::kDynamic &&
        operandValid[i] != ShapedType::kDynamic &&
        scaleValid[i] != operandValid[i])
      return op->emitOpError() << "expects " << scaleName << " and " << operandName
                               << " to have the same valid_shape";
  }
  return success();
}
