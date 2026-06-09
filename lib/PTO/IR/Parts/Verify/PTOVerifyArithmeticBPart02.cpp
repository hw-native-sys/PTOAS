// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticB.cpp; kept as a fragment included by PTOVerifyArithmeticB.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyTConcatDstCols(TConcatOp op, Type dstTy,
                                          ArrayRef<int64_t> src0Valid,
                                          ArrayRef<int64_t> src1Valid) {
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == kPTORowColRank && dstShape[1] != ShapedType::kDynamic &&
      src0Valid[1] != ShapedType::kDynamic &&
      src1Valid[1] != ShapedType::kDynamic &&
      src0Valid[1] + src1Valid[1] > dstShape[1]) {
    return op.emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
  }
  return success();
}

struct BinaryTileTypeInfo {
  Type src0Ty;
  Type src1Ty;
  Type dstTy;
  Type src0Elem;
  Type src1Elem;
  Type dstElem;
};

template <typename VerifyFn>
static FailureOr<BinaryTileTypeInfo>
verifyBinaryTileTypeInfo(Operation *op, Value src0, Value src1, Value dst,
                         VerifyFn verifyOperand) {
  Type src0Ty = src0.getType();
  Type src1Ty = src1.getType();
  Type dstTy = dst.getType();
  if (failed(verifyOperand(op, src0Ty, "src0")) ||
      failed(verifyOperand(op, src1Ty, "src1")) ||
      failed(verifyOperand(op, dstTy, "dst"))) {
    return failure();
  }
  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !dstElem)
    return op->emitOpError("failed to get element type for operands"), failure();
  return BinaryTileTypeInfo{src0Ty, src1Ty, dstTy, src0Elem, src1Elem, dstElem};
}

static FailureOr<Type> verifyTConcatCommon(TConcatOp op) {
  auto verifyTileBufOperand = [](Operation *op, Type ty, StringRef name) {
    return verifyTileBufCommon(op, ty, name);
  };
  auto infoOr = verifyBinaryTileTypeInfo(op, op.getSrc0(), op.getSrc1(),
                                         op.getDst(), verifyTileBufOperand);
  if (failed(infoOr))
    return failure();
  const auto &info = *infoOr;
  if (info.src0Elem != info.src1Elem || info.src0Elem != info.dstElem) {
    return op.emitOpError(
               "expects src0, src1, and dst to have the same element type"),
           failure();
  }

  auto src0Valid = getValidShapeVec(op.getSrc0());
  auto src1Valid = getValidShapeVec(op.getSrc1());
  auto dstValid = getValidShapeVec(op.getDst());
  if (failed(verifyTConcatValidRows(op, src0Valid, src1Valid, dstValid)) ||
      failed(verifyTConcatDstCols(op, info.dstTy, src0Valid, src1Valid)))
    return failure();
  return info.src0Elem;
}

static LogicalResult verifyTColExpandRowMajorLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (auto tileTy = dyn_cast<TileBufType>(ty); tileTy &&
      tileTy.getBLayoutValueI32() != 0) {
    return op->emitOpError() << "expects " << name << " to use row-major layout";
  }
  return success();
}

static LogicalResult verifyTColExpandSrc1ValidCols(Operation *op, Type t1,
                                                   Type td) {
  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == kPTORowColRank && dstValid.size() == kPTORowColRank &&
      src1Valid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic &&
      src1Valid[1] != dstValid[1]) {
    return op->emitOpError(
        "expects src1 valid_shape[1] to equal dst valid_shape[1]");
  }
  return success();
}

static LogicalResult verifyTColExpandShapeAndLayout(Operation *op, Type t0,
                                                    Type t1, Type td) {
  if (getShapeVec(t0) != getShapeVec(td))
    return op->emitOpError("expects src0/dst to have same shape");
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")) ||
      failed(verifyTColExpandRowMajorLayout(op, t0, "src0")) ||
      failed(verifyTColExpandRowMajorLayout(op, t1, "src1")) ||
      failed(verifyTColExpandRowMajorLayout(op, td, "dst")) ||
      failed(verifyTColExpandSrc1ValidCols(op, t1, td)))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto elemOr = verifyTConcatCommon(*this);
  if (failed(elemOr))
    return failure();
  auto verifyA2A3 = [this, &elemOr]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    return verifyConcatElemType(getOperation(), *elemOr);
  };
  auto verifyA5 = [this, &elemOr]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    }
    return verifyConcatElemType(getOperation(), *elemOr);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyConcatidxElementTypes(Operation *op, Type dataElem,
                                                 Type idxElem) {
  if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
    auto dataInt = dyn_cast<IntegerType>(dataElem);
    if (!dataInt || !dataInt.isSignless() ||
        (dataInt.getWidth() != kPTOI8BitWidth && dataInt.getWidth() != kPTOI16BitWidth &&
         dataInt.getWidth() != kPTOI32BitWidth)) {
      return op->emitOpError(
          "expects data element type to be i8, i16, i32, f16, f32, or bf16");
    }
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || !idxInt.isSignless() ||
      (idxInt.getWidth() != kPTOI8BitWidth && idxInt.getWidth() != kPTOI16BitWidth &&
       idxInt.getWidth() != kPTOI32BitWidth)) {
    return op->emitOpError(
        "expects index element type to be i8, i16, or i32");
  }
  return success();
}

static LogicalResult verifyTConcatidxValidShapes(TConcatidxOp op) {
  auto src0Valid = getValidShapeVec(op.getSrc0());
  auto src1Valid = getValidShapeVec(op.getSrc1());
  auto src0IdxValid = getValidShapeVec(op.getSrc0Idx());
  auto src1IdxValid = getValidShapeVec(op.getSrc1Idx());
  auto dstValid = getValidShapeVec(op.getDst());
  if (src0Valid.size() != kPTORowColRank || src1Valid.size() != kPTORowColRank ||
      src0IdxValid.size() != kPTORowColRank || src1IdxValid.size() != kPTORowColRank ||
      dstValid.size() != kPTORowColRank) {
    return op.emitOpError("expects all operands to have rank-2 valid_shape");
  }

  Operation *opBase = op.getOperation();
  auto checkValidRow = [opBase, &dstValid](const auto &validShape,
                                           StringRef name) -> LogicalResult {
    if (validShape[0] != ShapedType::kDynamic &&
        dstValid[0] != ShapedType::kDynamic && validShape[0] != dstValid[0]) {
      opBase->emitOpError("expects ")
          << name << " valid row to match dst valid row";
      return failure();
    }
    return success();
  };
  if (failed(checkValidRow(src0Valid, "src0")) ||
      failed(checkValidRow(src1Valid, "src1")) ||
      failed(checkValidRow(src0IdxValid, "src0Idx")) ||
      failed(checkValidRow(src1IdxValid, "src1Idx"))) {
    return failure();
  }
  if (src0IdxValid[1] != ShapedType::kDynamic && src0IdxValid[1] < 1)
    return op.emitOpError("expects src0Idx valid_col >= 1");
  if (src1IdxValid[1] != ShapedType::kDynamic && src1IdxValid[1] < 1)
    return op.emitOpError("expects src1Idx valid_col >= 1");
  return success();
}

static FailureOr<std::pair<Type, Type>> verifyTConcatidxCommon(TConcatidxOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type src0IdxTy = op.getSrc0Idx().getType();
  Type src1IdxTy = op.getSrc1Idx().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, src0IdxTy, "src0Idx")) ||
      failed(verifyTileBufCommon(op, src1IdxTy, "src1Idx")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }

  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !dstElem)
    return op.emitOpError("failed to get element type for data operands"),
           failure();
  if (src0Elem != src1Elem || src0Elem != dstElem) {
    return op.emitOpError(
               "expects src0, src1, and dst to have the same element type"),
           failure();
  }

  Type src0IdxElem = getElemTy(src0IdxTy);
  Type src1IdxElem = getElemTy(src1IdxTy);
  if (!src0IdxElem || !src1IdxElem) {
    return op.emitOpError("failed to get element type for index operands"),
           failure();
  }
  if (src0IdxElem != src1IdxElem) {
    return op.emitOpError(
               "expects src0Idx and src1Idx to have the same element type"),
           failure();
  }

  if (failed(verifyTConcatidxValidShapes(op)))
    return failure();
  return std::make_pair(src0Elem, src0IdxElem);
}
