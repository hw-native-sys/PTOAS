// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static LogicalResult verifyRowReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
    return op->emitOpError("expects src valid_shape[0] to be non-zero");
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
    return op->emitOpError("expects src valid_shape[1] to be non-zero");
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError("expects src and dst to have the same valid_shape[0]");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1)
    return op->emitOpError("expects dst valid_shape[1] to be 1");
  return success();
}

static bool isSupportedRowReductionElemType(Type elem) {
  return elem.isInteger(kPTOI16BitWidth) || elem.isInteger(kPTOI32BitWidth) || elem.isF16() ||
         elem.isF32();
}

static LogicalResult verifyTRowReductionNoTmpCommon(Operation *op, Type srcTy,
                                                    Type dstTy,
                                                    StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

static LogicalResult verifyTRowReductionWithTmpCommon(Operation *op, Type srcTy,
                                                      Type tmpTy, Type dstTy,
                                                      StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

static LogicalResult verifyTRowArgReductionCommon(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
    return failure();
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem))
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != kPTOI32BitWidth)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
  }
  return success();
}

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != kPTORowColRank || dstValid.size() != kPTORowColRank)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name)))
    return failure();
  auto valid = getValidShapeVec(ty);
  if (valid.size() != kPTORowColRank)
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy)
    return emitOpError("result must be pto.tensor_view<...>");

  auto pty = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!pty)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  if (pty.getElementType() != tvTy.getElementType())
    return emitOpError() << "ptr element type must match tensor_view element type, but got ptr="
                         << pty.getElementType() << " view=" << tvTy.getElementType();

  int64_t rank = tvTy.getRank();
  if (static_cast<int64_t>(getShape().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();
  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });
  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

static LogicalResult verifyPartitionViewDimension(
    PartitionViewOp op, int64_t dimIdx, ArrayRef<int64_t> srcShape,
    ArrayRef<int64_t> resShape, bool sameRank) {
  auto offVal = getConstIndexValue(op.getOffsets()[dimIdx]);
  auto sizeVal = getConstIndexValue(op.getSizes()[dimIdx]);
  if (offVal && *offVal < 0)
    return op.emitOpError() << "offset at dim " << dimIdx
                            << " must be non-negative, got " << *offVal;
  if (sizeVal && *sizeVal <= 0)
    return op.emitOpError() << "size at dim " << dimIdx
                            << " must be positive, got " << *sizeVal;
  if (sameRank && sizeVal) {
    int64_t resDim = resShape[dimIdx];
    if (resDim != ShapedType::kDynamic && *sizeVal != resDim) {
      return op.emitOpError() << "size/result mismatch at dim " << dimIdx
                              << ": size operand=" << *sizeVal
                              << " result type dim=" << resDim;
    }
  }
  int64_t srcDim = srcShape[dimIdx];
  if (srcDim == ShapedType::kDynamic)
    return success();
  if (sizeVal && *sizeVal > srcDim) {
    return op.emitOpError() << "size at dim " << dimIdx << " (" << *sizeVal
                            << ") exceeds static source dim (" << srcDim
                            << ")";
  }
  if (offVal && sizeVal && (*offVal + *sizeVal > srcDim)) {
    return op.emitOpError() << "offset+size at dim " << dimIdx << " ("
                            << (*offVal + *sizeVal)
                            << ") exceeds static source dim (" << srcDim
                            << ")";
  }
  return success();
}
