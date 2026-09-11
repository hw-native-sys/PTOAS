// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

LogicalResult pto::TAxpyOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTAxpyArch(*this, /*allowBf16=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTAxpyArch(*this, /*allowBf16=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd) {
    return emitOpError("expects src0/src1/dst to have the same shape");
  }
  return success();
}

static LogicalResult verifyIntegerWidths(Operation *op, Type type,
                                         ArrayRef<unsigned> widths,
                                         StringRef diagnostic) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || !llvm::is_contained(widths, intType.getWidth()))
    return op->emitOpError(diagnostic);
  return success();
}

static LogicalResult verifyBitwiseBinaryOp(Operation *op, Type src0,
                                           Type src1, Type dst,
                                           StringRef opName) {
  auto verifyCommon = [&]() {
    return verifyMatchingRowMajorBinaryTileOpCommon(op, src0, src1, dst);
  };
  auto verifyFor = [&](StringRef arch) -> LogicalResult {
    auto elem = verifyCommon();
    if (failed(elem))
      return failure();
    std::string diagnostic =
        (Twine("expects ") + arch + " " + opName +
         " src0, src1, and dst element type to be i8/i16/i32")
            .str();
    return verifyIntegerWidths(op, *elem, {8, 16, 32}, diagnostic);
  };
  auto verifyA2A3 = [&]() { return verifyFor("A2/A3"); };
  auto verifyA5 = [&]() { return verifyFor("A5"); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyBitwiseScalarOp(Operation *op, Value src,
                                           Value dst, StringRef opName) {
  auto verifyFor = [&](bool isA5) -> LogicalResult {
    auto elem = verifyDistinctRowMajorUnaryTileOpCommon(op, src, dst, "src",
                                                        "dst");
    if (failed(elem))
      return failure();
    std::string diagnostic =
        (Twine("expects ") + (isA5 ? "A5 " : "A2/A3 ") + opName +
         " src, scalar, and dst element type to be " +
         (isA5 ? "i8/i16/i32" : "i8/i16"))
            .str();
    return verifyIntegerWidths(op, *elem, isA5 ? ArrayRef<unsigned>{8, 16, 32}
                                               : ArrayRef<unsigned>{8, 16},
                               diagnostic);
  };
  auto verifyA2A3 = [&]() { return verifyFor(false); };
  auto verifyA5 = [&]() { return verifyFor(true); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

LogicalResult pto::TAndOp::verify() {
  return verifyBitwiseBinaryOp(getOperation(), getSrc0().getType(),
                               getSrc1().getType(), getDst().getType(), "tand");
}

static LogicalResult verifyTConcatValidShapes(TConcatOp op, ArrayRef<int64_t> v0,
                                              ArrayRef<int64_t> v1,
                                              ArrayRef<int64_t> vd, Type dstTy) {
    if (v0.size() != mlir::pto::kValue2 || v1.size() != mlir::pto::kValue2 || vd.size() != mlir::pto::kValue2) {
        return op.emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    }
  if (v0[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
      v0[0] != vd[0]) {
    return op.emitOpError("expects src0 valid row to match dst valid row");
  }
  if (v1[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
      v1[0] != vd[0]) {
    return op.emitOpError("expects src1 valid row to match dst valid row");
  }
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == mlir::pto::kValue2 && dstShape[1] != ShapedType::kDynamic && v0[1] != ShapedType::kDynamic &&
      v1[1] != ShapedType::kDynamic && v0[1] + v1[1] > dstShape[1]) {
      return op.emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
  }
  return success();
}

static FailureOr<Type> verifyThreeMatchingTiles(Operation *op, Type t0,
                                                Type t1, Type td,
                                                Type optionalTmp = {}) {
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst")) ||
      (optionalTmp && failed(verifyVecTileCommon(op, optionalTmp, "tmp")))) {
    return failure();
  }
  return verifyThreeMatchingElementTypes(op, t0, t1, td);
}

static FailureOr<Type> verifyTConcatCommon(TConcatOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  FailureOr<Type> elem = verifyThreeMatchingTiles(op, t0, t1, td);
  if (failed(elem))
    return failure();

  auto v0 = getValidShapeVec(op.getSrc0());
  auto v1 = getValidShapeVec(op.getSrc1());
  auto vd = getValidShapeVec(op.getDst());
  if (failed(verifyTConcatValidShapes(op, v0, v1, vd, td))) {
    return failure();
  }

  return *elem;
}

static LogicalResult verifyTConcatElemType(TConcatOp op, Type elem) {
  if (elem.isF16() || elem.isF32() || elem.isBF16()) {
    return success();
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != mlir::pto::kValue8 && it.getWidth() != mlir::pto::kValue16 &&
              it.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("expects element type to be i8, i16, i32, f16, f32, or bf16");
  }
  return success();
}

static LogicalResult verifyTConcatLocVec(TConcatOp op, Type ty, StringRef name) {
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op.emitOpError() << "expects " << name << " to use loc=vec";
  }
  return success();
}

static FailureOr<Type> verifyTConcatBase(TConcatOp op) {
  auto elem = verifyTConcatCommon(op);
  if (failed(elem))
    return failure();
  if (failed(verifyTConcatLocVec(op, op.getSrc0().getType(), "src0")) ||
      failed(verifyTConcatLocVec(op, op.getSrc1().getType(), "src1")) ||
      failed(verifyTConcatLocVec(op, op.getDst().getType(), "dst")))
    return failure();
  return *elem;
}

static LogicalResult verifyTConcatA2A3(TConcatOp op) {
  auto elem = verifyTConcatBase(op);
  return failed(elem) ? failure() : verifyTConcatElemType(op, *elem);
}

static LogicalResult verifyTConcatA5(TConcatOp op) {
  auto elem = verifyTConcatBase(op);
  if (failed(elem))
    return failure();
  if (!isRowMajorTileBuf(op.getSrc0().getType()) || !isRowMajorTileBuf(op.getSrc1().getType()) ||
      !isRowMajorTileBuf(op.getDst().getType())) {
    return op.emitOpError("expects src0, src1, and dst to use row-major layout");
  }
  return verifyTConcatElemType(op, *elem);
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTConcatA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTConcatA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTConcatidxValidRows(
    TConcatidxOp op, ArrayRef<int64_t> dstValid) {
  SmallVector<Value, 4> values = {op.getSrc0(), op.getSrc1(), op.getSrc0Idx(),
                                  op.getSrc1Idx()};
  SmallVector<StringRef, mlir::pto::kValue4> names = {"src0", "src1", "src0Idx", "src1Idx"};
  for (auto [value, name] : llvm::zip_equal(values, names)) {
    auto valid = getValidShapeVec(value);
    if (valid.size() != mlir::pto::kValue2) {
        return op.emitOpError("expects all operands to have rank-2 valid_shape");
    }
    if (valid[0] != ShapedType::kDynamic &&
        dstValid[0] != ShapedType::kDynamic && valid[0] != dstValid[0]) {
      return op.emitOpError("expects ") << name
                                         << " valid row to match dst valid row";
    }
  }
  return success();
}

static FailureOr<std::pair<Type, Type>> verifyTConcatidxElementAgreement(
    TConcatidxOp op, ArrayRef<Type> types) {
  Type dataElem = getElemTy(types[0]);
  Type secondDataElem = getElemTy(types[1]);
  Type dstElem = getElemTy(types[4]);
  if (!dataElem || !secondDataElem || !dstElem) {
    op.emitOpError("failed to get element type for data operands");
    return failure();
  }
  if (dataElem != secondDataElem || dataElem != dstElem) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  Type indexElem = getElemTy(types[2]);
  Type secondIndexElem = getElemTy(types[3]);
  if (!indexElem || !secondIndexElem) {
    op.emitOpError("failed to get element type for index operands");
    return failure();
  }
  if (indexElem != secondIndexElem) {
    op.emitOpError("expects src0Idx and src1Idx to have the same element type");
    return failure();
  }
  return std::make_pair(dataElem, indexElem);
}

static LogicalResult verifyTConcatidxIndexColumns(TConcatidxOp op) {
  for (Value index : {op.getSrc0Idx(), op.getSrc1Idx()}) {
    auto valid = getValidShapeVec(index);
    if (valid[1] != ShapedType::kDynamic && valid[1] < 1)
      return op.emitOpError() << "expects "
                              << (index == op.getSrc0Idx() ? "src0Idx"
                                                           : "src1Idx")
                              << " valid_col >= 1";
  }
  return success();
}

static FailureOr<std::pair<Type, Type>> verifyTConcatidxCommon(
    TConcatidxOp op) {
    SmallVector<Type, mlir::pto::kValue5> types = {
        op.getSrc0().getType(), op.getSrc1().getType(), op.getSrc0Idx().getType(), op.getSrc1Idx().getType(),
        op.getDst().getType()};
    SmallVector<StringRef, mlir::pto::kValue5> names = {"src0", "src1", "src0Idx", "src1Idx", "dst"};
    for (auto [type, name] : llvm::zip_equal(types, names)) {
        if (failed(verifyTileBufCommon(op, type, name))) {
            return failure();
        }
    }
  auto elementTypes = verifyTConcatidxElementAgreement(op, types);
  if (failed(elementTypes))
    return failure();
  auto dstValid = getValidShapeVec(op.getDst());
  if (dstValid.size() != mlir::pto::kValue2) {
      return op.emitOpError("expects all operands to have rank-2 valid_shape");
  }
  if (failed(verifyTConcatidxValidRows(op, dstValid))) {
    return failure();
  }
  if (failed(verifyTConcatidxIndexColumns(op)))
    return failure();
  return *elementTypes;
}
