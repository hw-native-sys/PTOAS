// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult
verifyTInsertOptionalFp(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                        bool isA5) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("fp and preQuantScalar are mutually exclusive");
  }
  if (hasFp) {
    if (!srcIsAcc) {
      return op.emitOpError("fp is only valid with src loc=acc");
    }
    auto fpTy = op.getFp().getType();
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    if (!fpTb) {
      return op.emitOpError("expects fp to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(op, fpTy, "fp", /*allowLowPrecision=*/isA5))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to be loc=scaling");
    }
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("preQuantScalar is only valid with src loc=acc");
  }
  if (reluNonDefault && !srcIsAcc) {
    return op.emitOpError("reluPreMode is only valid with src loc=acc");
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalAttrs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                           std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasAccToVecMode) {
    if (!isA5) {
      return op.emitOpError("accToVecMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
        *dstSpace != pto::AddressSpace::VEC)
      return op.emitOpError("accToVecMode is only valid with src=acc, dst=vec");
  }
  if (hasInsertMode) {
    if (!isA5) {
      return op.emitOpError("tinsertMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::MAT) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) is only valid with src=vec, dst=mat");
    }
    auto srcTb = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!srcTb || !isColMajorRowMajorNZ(srcTb)) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) requires src NZ layout "
          "(blayout=col_major, slayout=row_major)");
    }
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalArgs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                          std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  if (failed(verifyTInsertOptionalFp(op, srcSpace, isA5))) {
    return failure();
  }
  return verifyTInsertOptionalAttrs(op, srcSpace, dstSpace, isA5);
}

static LogicalResult verifyTInsertA2A3AccMat(TInsertOp op,
                                             const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != mlir::pto::kValue512) {
      return op.emitOpError("expects A2/A3 tinsert dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantInsertTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc fp/preQuantScalar tinsert element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastInsertTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTInsertA2A3(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  if (failed(verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/false))) {
    return failure();
  }
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError(
          "expects vec->vec tinsert to use the base form without "
          "preQuantScalar or reluPreMode");
    }
    if (c.srcElem != c.dstElem || !isA2A3VecInsertElemType(c.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
          "(i8/f16/bf16/f32)");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
      *c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");
  }
  return verifyTInsertA2A3AccMat(op, c);
}

static LogicalResult verifyTInsertA5Acc(TInsertOp op, const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (!isColMajorRowMajorNZ(c.dstTb)) {
      return op.emitOpError("expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
    bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
    if (!dstIsND && !dstIsNZ) {
      return op.emitOpError(
          "expects A5 acc->vec tinsert dst to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
    }
  }
  const bool hasQuant =
      static_cast<bool>(op.getFp()) || static_cast<bool>(op.getPreQuantScalar());
  bool okTypes;
  if (hasQuant) {
    okTypes = isA5VectorPreQuantTypePair(c.srcElem, c.dstElem);
  } else {
      okTypes = (c.srcElem.isF32() && (c.dstElem.isF16() || c.dstElem.isBF16() || c.dstElem.isF32())) ||
                (c.srcElem.isInteger(mlir::pto::kValue32) && c.dstElem.isInteger(mlir::pto::kValue32));
  }
  if (!okTypes) {
    return op.emitOpError(
        "expects A5 acc-source tinsert element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)" +
        (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
  }
  return success();
}

static LogicalResult verifyTInsertA5VecMat(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->mat tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  if (!srcIsND && !srcIsNZ) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
  }
  if (hasTInsertMode && !srcIsNZ) {
    return op.emitOpError("expects tinsertMode vec->mat tinsert src to use NZ(col_major/row_major) layout");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecVec(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->vec tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
  if (srcIsND && dstIsND) {
    // ND->ND path
  } else if (srcIsNZ && dstIsNZ) {
    // NZ->NZ path
  } else {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst layouts to match: "
        "both ND(row_major/none_box) or both NZ(col_major/row_major)");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5Modes(TInsertOp op,
                                          const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  if (hasAccToVecMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec tinsert forms");
  }
  if (hasTInsertMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::VEC ||
       *c.dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects tinsertMode only on A5 vec->mat tinsert forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects A5 tinsert src/dst to have explicit loc");
  }
  return verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/true);
}
