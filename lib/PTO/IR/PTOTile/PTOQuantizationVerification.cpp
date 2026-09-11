// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifyTQuantA2A3(TQuantOp op) {
  if (failed(verifyTQuantInt8Common(op))) {
    return failure();
  }
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError()
           << "expects A2/A3 src and dst to use row-major layout";
  }
  if (op.getTmp() &&
      failed(verifyTQuantA2A3Tmp(op, srcTy, op.getTmp().getType()))) {
    return failure();
  }
  if (failed(verifyTQuantA2A3Param(op, fpTy, dstTy, "fp"))) {
    return failure();
  }
  if (op.getOffset() &&
      failed(verifyTQuantA2A3Param(op, op.getOffset().getType(), dstTy, "offset"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  if (failed(verifyTQuantStructural(*this))) {
    return failure();
  }
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTQuantA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantInt8Common(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static std::optional<int64_t> mxCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> mxCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> mxCeilDiv(int64_t value, int64_t divisor) {
  if (value < 0 || divisor == 0 || divisor < 0) {
    return std::nullopt;
  }
  auto plus = mxCheckedAdd(value, divisor - 1);
  return plus ? std::optional<int64_t>(*plus / divisor) : std::nullopt;
}

static std::optional<int64_t> mxAlignTo(int64_t value, int64_t alignment) {
  if (alignment == 0 || alignment < 0) {
    return std::nullopt;
  }
  auto quotient = mxCeilDiv(value, alignment);
  if (!quotient || *quotient > std::numeric_limits<int64_t>::max() / alignment) {
    return std::nullopt;
  }
  return *quotient * alignment;
}

static std::optional<int64_t> mxCapacityElems(Type type) {
  auto shape = getShapeVec(type);
  return mxCheckedMul(shape[0], shape[1]);
}

static std::optional<int64_t> mxCapacityBytes(Type type) {
  auto elems = mxCapacityElems(type);
  unsigned bytes = getElemByteSize(getElemTy(type));
  if (!elems || bytes == 0 ||
      *elems >
          std::numeric_limits<int64_t>::max() / static_cast<int64_t>(bytes)) {
    return std::nullopt;
  }
  return *elems * static_cast<int64_t>(bytes);
}

static LogicalResult mxRequireCapacity(TQuantMxOp op, StringRef name, Type type,
                                       int64_t required) {
  auto actual = mxCapacityElems(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " elements";
  }
  return success();
}

static LogicalResult mxRequireCapacityBytes(TQuantMxOp op, StringRef name,
                                            Type type, int64_t required) {
  auto actual = mxCapacityBytes(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " bytes";
  }
  return success();
}

static LogicalResult mxRequireCompact(TQuantMxOp op, StringRef name, Type type) {
  auto valid = getValidShapeVec(type);
  auto physical = getShapeVec(type);
  if (valid[0] != 1 && physical[1] != valid[1]) {
    return op.emitOpError() << "expects " << name
                            << " valid elements to form a compact physical prefix";
  }
  return success();
}

static LogicalResult mxRequireStaticShape(TQuantMxOp op, StringRef name,
                                          Type type) {
  for (int64_t dim : getValidShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  for (int64_t dim : getShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxTilesAndForm(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst", /*allowLowPrecision=*/true)) ||
      failed(verifyNDStyleVecTile(op, expTy, "exp")) ||
      failed(verifyNDStyleVecTile(op, maxTy, "max")) ||
      failed(verifyNDStyleVecTile(op, scalingTy, "scaling"))) {
    return failure();
  }
  if (op.getExpZz() &&
      failed(verifyNDStyleVecTile(op, op.getExpZz().getType(), "exp_zz"))) {
    return failure();
  }
  const bool isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  if (op.getInterleave() && !isDn) {
    return op.emitOpError("expects interleave to be used only with grpAxis=axis0");
  }
  if (op.getExpZz() && isDn) {
    return op.emitOpError("expects the deprecated exp_zz form to use grpAxis=axis1; use pto.tmov with a non-scaling tmp for axis0 exponents");
  }
  auto quantType = op.getQuantType();
  if (quantType != mlir::pto::QuantType::MXFP8 &&
      quantType != mlir::pto::QuantType::MXFP4_E2M1) {
    return op.emitOpError("expects quant_type to be MXFP8 or MXFP4_E2M1");
  }
  if (op.getExpZz() && !op.getStoreMode()) {
    return op.emitOpError("expects storeMode when exp_zz is present");
  }
  if (op.getStoreMode() && !op.getExpZz()) {
    return op.emitOpError("expects exp_zz when storeMode is present");
  }
  if (op.getStoreMode() && op.getQuantScaleAlg() != mlir::pto::QuantScaleAlg::OCP) {
    return op.emitOpError("storeMode form must not override quantScaleAlg");
  }
  return success();
}

static LogicalResult verifyTQuantMxElemTypes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(op.getDst().getType());
  Type expElem = getElemTy(op.getExp().getType());
  Type maxElem = getElemTy(op.getMax().getType());
  Type scalingElem = getElemTy(op.getScaling().getType());
  if (!(srcElem.isF32() || srcElem.isF16() || srcElem.isBF16())) {
    return op.emitOpError("expects src element type to be f32/f16/bf16");
  }
  if (!expElem.isInteger(8)) {
    return op.emitOpError("expects exp element type to be i8/ui8");
  }
  if (op.getExpZz() && !getElemTy(op.getExpZz().getType()).isInteger(8)) {
    return op.emitOpError("expects exp_zz element type to be i8/ui8");
  }
  if (maxElem != srcElem) {
    return op.emitOpError("expects max element type to match src element type");
  }
  if (scalingElem != srcElem) {
    return op.emitOpError("expects scaling element type to match src element type");
  }
  if (op.getQuantType() == mlir::pto::QuantType::MXFP8) {
    if (!dstElem.isInteger(8)) {
      return op.emitOpError("expects MXFP8 dst element type to be i8/ui8");
    }
  } else {
    if (!isa<pto::F4E2M1x2Type>(dstElem)) {
      return op.emitOpError("expects MXFP4_E2M1 dst element type to be !pto.f4E2M1x2");
    }
    if (!(srcElem.isF16() || srcElem.isBF16())) {
      return op.emitOpError("expects MXFP4_E2M1 src element type to be f16/bf16");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxStaticShapes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  for (auto [name, type] : {std::pair<StringRef, Type>("src", srcTy),
                            {"dst", dstTy}, {"exp", expTy}, {"max", maxTy},
                            {"scaling", scalingTy}}) {
    if (failed(mxRequireStaticShape(op, name, type))) {
      return failure();
    }
  }
  if (op.getExpZz()) {
    Type expZzTy = op.getExpZz().getType();
    if (llvm::is_contained(getValidShapeVec(expZzTy), ShapedType::kDynamic) ||
        llvm::is_contained(getShapeVec(expZzTy), ShapedType::kDynamic)) {
      return op.emitOpError("expects static valid and physical shapes for exp_zz in the deprecated fused MX quantization form");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxShapes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  for (Type type : {srcTy, dstTy, expTy, maxTy, scalingTy}) {
    if (getValidShapeVec(type).size() != 2 || getShapeVec(type).size() != 2)
      return op.emitOpError(
          "expects rank-2 valid and physical shapes for MX quantization");
  }
  if (op.getExpZz() &&
      (getValidShapeVec(op.getExpZz().getType()).size() != 2 ||
       getShapeVec(op.getExpZz().getType()).size() != 2))
    return op.emitOpError(
        "expects rank-2 valid and physical shapes for exp_zz");
  if (failed(verifyTQuantMxStaticShapes(op)))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, maxTy, "src", "max")) ||
      failed(verifyTileBufSameElemType(op, srcTy, scalingTy, "src", "scaling")) ||
      failed(verifyTileBufSameLogicalExtent(op, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  return success();
}

struct TQuantMxA5 {
  Type srcTy, dstTy, expTy, maxTy, scalingTy;
  Type srcElem;
  SmallVector<int64_t, 4> dstValid, expValid, expPhysical;
  SmallVector<int64_t, 4> dstPhysical, maxPhysical, scalingPhysical;
  bool isDn, isMxFp4;
  int64_t srcRows, srcCols, srcPhysicalRows, srcPhysicalCols;
  int64_t pack, dstValidCols, groups;
};
