// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult mxRequireGroupedShape(TQuantMxOp op, const TQuantMxA5 &s,
                                           StringRef name, Type type,
                                           bool allowLegacy) {
  auto valid = getValidShapeVec(type);
  SmallVector<int64_t, 2> canonical = {s.isDn ? s.srcRows / 32 : s.srcRows,
                                       s.isDn ? s.srcCols : s.srcCols / 32};
  if (llvm::equal(valid, canonical)) {
    return success();
  }
  if (!s.isDn && allowLegacy && valid[0] == 1 && valid[1] == s.groups) {
    return success();
  }
  return op.emitOpError()
         << "expects " << name << " valid_shape to match "
         << (s.isDn ? "canonical [M/32, N] for grpAxis=axis0"
                    : "canonical [M, N/32] or legacy flat [1, M*N/32] for grpAxis=axis1");
}

static FailureOr<TQuantMxA5> buildTQuantMxA5State(TQuantMxOp op) {
  TQuantMxA5 s;
  s.srcTy = op.getSrc().getType();
  s.dstTy = op.getDst().getType();
  s.expTy = op.getExp().getType();
  s.maxTy = op.getMax().getType();
  s.scalingTy = op.getScaling().getType();
  s.srcElem = getElemTy(s.srcTy);
  auto srcValid = getValidShapeVec(s.srcTy);
  s.dstValid = getValidShapeVec(s.dstTy);
  s.expValid = getValidShapeVec(s.expTy);
  auto srcPhysical = getShapeVec(s.srcTy);
  s.dstPhysical = getShapeVec(s.dstTy);
  s.expPhysical = getShapeVec(s.expTy);
  s.maxPhysical = getShapeVec(s.maxTy);
  s.scalingPhysical = getShapeVec(s.scalingTy);
  s.isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  s.srcRows = srcValid[0];
  s.srcCols = srcValid[1];
  s.srcPhysicalRows = srcPhysical[0];
  s.srcPhysicalCols = srcPhysical[1];
  if (s.srcRows <= 0 || s.srcCols <= 0 || s.srcPhysicalRows < s.srcRows ||
      s.srcPhysicalCols < s.srcCols) {
    return op.emitOpError("expects positive source valid shape within physical shape");
  }
  if ((s.isDn ? s.srcRows : s.srcCols) % 32 != 0) {
    return op.emitOpError() << "expects src valid_shape[" << (s.isDn ? 0 : 1)
                            << "] to be a multiple of 32 when grpAxis is "
                            << (s.isDn ? "axis0" : "axis1");
  }
  auto groupsOr = mxCheckedMul(s.srcRows, s.srcCols / 32);
  if (!groupsOr) {
    return op.emitOpError("cannot compute MX quantization group count without overflow");
  }
  s.groups = *groupsOr;
  s.isMxFp4 = op.getQuantType() == mlir::pto::QuantType::MXFP4_E2M1;
  s.pack = s.isMxFp4 ? 2 : 1;
  s.dstValidCols = s.isMxFp4 ? s.srcCols / 2 : s.srcCols;
  return s;
}

static LogicalResult verifyTQuantMxGrouping(TQuantMxOp op, const TQuantMxA5 &s) {
  if (op.getExpZz()) {
    auto expZzElements =
        mxCheckedMul(getValidShapeVec(op.getExpZz().getType())[0],
                     getValidShapeVec(op.getExpZz().getType())[1]);
    if (!expZzElements || *expZzElements != s.groups) {
      return op.emitOpError("expects exp_zz valid element count to equal MX group count");
    }
  }
  if (failed(mxRequireGroupedShape(op, s, "max", s.maxTy, /*allowLegacy=*/true)) ||
      failed(mxRequireGroupedShape(op, s, "scaling", s.scalingTy,
                                   /*allowLegacy=*/true))) {
    return failure();
  }
  if (!op.getInterleave()) {
    return mxRequireGroupedShape(op, s, "exp", s.expTy, /*allowLegacy=*/true);
  }
  if (s.srcRows % 64 != 0) {
    return op.emitOpError("expects src valid rows to be a multiple of 64 when interleave is true");
  }
  if (s.srcPhysicalRows % 64 != 0) {
    return op.emitOpError("expects src physical rows to be a multiple of 64 when interleave is true");
  }
  auto doubledValidCols = mxCheckedMul(s.srcCols, 2);
  if (!doubledValidCols) {
    return op.emitOpError("cannot compute interleaved exp valid shape without overflow");
  }
  if (s.expValid[0] != s.srcRows / 64 || s.expValid[1] != *doubledValidCols) {
    return op.emitOpError("expects exp valid_shape to match [M/64, 2N] for grpAxis=axis0 with interleave=true");
  }
  return success();
}

static LogicalResult verifyTQuantMxDstShape(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.isMxFp4 && s.srcPhysicalCols % 2 != 0) {
    return op.emitOpError("expects MXFP4 src physical cols to be even for packed destination addressing");
  }
  if (s.dstValid[0] != s.srcRows || s.dstValid[1] != s.dstValidCols) {
    return op.emitOpError() << "expects dst valid_shape to be [" << s.srcRows
                            << ", " << s.dstValidCols << "] for MX quantization";
  }
  if (s.dstPhysical[0] < s.srcRows) {
    return op.emitOpError("expects dst physical rows to cover src valid rows");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0Dst(TQuantMxOp op,
                                            const TQuantMxA5 &s) {
  if (s.isMxFp4) {
    if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
      return op.emitOpError("expects MXFP4 axis0 dst physical cols to equal src physical cols / 2");
    }
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
    if ((s.srcPhysicalCols / s.pack) % 32 != 0 && s.srcElem.isF16()) {
      return op.emitOpError("does not support FP16 MXFP4 axis0 when packed source stride is not a multiple of 32 bytes");
    }
  } else {
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.dstPhysical[1]);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0Aux(TQuantMxOp op,
                                            const TQuantMxA5 &s) {
  if (s.maxPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects max physical cols to equal src physical cols for grpAxis=axis0");
  }
  if (s.scalingPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects scaling physical cols to equal src physical cols for grpAxis=axis0");
  }
  auto auxRequired = mxCheckedMul(s.srcRows / 32, s.srcPhysicalCols);
  if (!auxRequired || failed(mxRequireCapacity(op, "max", s.maxTy, *auxRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, *auxRequired))) {
    return failure();
  }
  if (!op.getInterleave()) {
    if (s.expPhysical[1] != s.srcPhysicalCols) {
      return op.emitOpError("expects exp physical cols to equal src physical cols for grpAxis=axis0");
    }
    if (failed(mxRequireCapacity(op, "exp", s.expTy, *auxRequired))) {
      return failure();
    }
  } else {
    auto doubledPhysicalCols = mxCheckedMul(s.srcPhysicalCols, 2);
    auto alignedPhysicalCols =
        doubledPhysicalCols ? mxAlignTo(*doubledPhysicalCols, 32) : std::nullopt;
    if (!alignedPhysicalCols) {
      return op.emitOpError("cannot compute interleaved exp physical cols without overflow");
    }
    if (s.expPhysical[0] != s.srcPhysicalRows / 64) {
      return op.emitOpError("expects interleaved exp physical rows to be src physical rows / 64");
    }
    if (s.expPhysical[1] != *alignedPhysicalCols) {
      return op.emitOpError("expects interleaved exp physical cols to be align32(2 * src physical cols)");
    }
  }
  if (s.srcElem.isF32() && op.getInterleave()) {
    return op.emitOpError("does not support FP32 interleave with the pinned pto-isa revision");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0(TQuantMxOp op,
                                         const TQuantMxA5 &s) {
  if (failed(verifyTQuantMxAxis0Dst(op, s)))
    return failure();
  return verifyTQuantMxAxis0Aux(op, s);
}

static LogicalResult verifyTQuantMxAxis1Flat(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.srcPhysicalCols != s.srcCols) {
    return op.emitOpError("expects axis1 flat exp to use a tight source with physical cols equal to valid cols");
  }
  if (s.expValid[0] != 1 || s.expValid[1] != s.groups) {
    return op.emitOpError("expects axis1 flat exp valid_shape to match legacy flat [1, M*N/32]");
  }
  if (failed(mxRequireCapacity(op, "exp", s.expTy, s.groups))) {
    return failure();
  }
  auto srcElems = mxCheckedMul(s.srcPhysicalRows, s.srcPhysicalCols);
  auto validElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
  bool unroll = srcElems && validElems && *srcElems > 1024 &&
                *srcElems % 256 == 0 && *validElems % 256 == 0;
  if (s.srcElem.isF32()) {
    auto scaleGroups =
        unroll ? mxCheckedMul(s.groups, 2) : std::optional<int64_t>(s.groups);
    auto aligned = scaleGroups ? mxAlignTo(*scaleGroups, 64) : std::nullopt;
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 4) : std::nullopt;
    StringRef scalingName = unroll ? "axis1 flat unrolled f32 scaling"
                                   : "axis1 flat f32 scaling";
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, scalingName, s.scalingTy,
                                      *requiredBytes))) {
      return failure();
    }
  } else if (op.getQuantScaleAlg() == pto::QuantScaleAlg::OCP) {
    auto aligned = mxAlignTo(s.groups, 128);
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 2) : std::nullopt;
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, "axis1 flat B16 OCP scaling",
                                      s.scalingTy, *requiredBytes))) {
      return failure();
    }
  } else {
    return op.emitOpError("does not support axis1 flat B16 NV quantization with the pinned pto-isa revisions");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1Canonical(TQuantMxOp op,
                                                  const TQuantMxA5 &s) {
  if (s.expValid[0] == 1) {
    return op.emitOpError("expects legacy flat exp to use physical rows == 1");
  }
  if (s.srcElem.isF16() || s.srcElem.isBF16()) {
    return op.emitOpError("does not support axis1 canonical 2D B16 quantization with the pinned pto-isa revision");
  }
  SmallVector<int64_t, 2> canonicalShape = {s.srcRows, s.srcCols / 32};
  if (!llvm::equal(s.expValid, canonicalShape)) {
    return op.emitOpError("expects exp valid_shape to match canonical [M, N/32] for grpAxis=axis1");
  }
  if (s.expPhysical[0] < s.srcRows || s.expPhysical[1] < s.srcCols / 32) {
    return op.emitOpError("expects axis1 canonical exp physical shape to cover [M, N/32]");
  }
  auto expPrefix = mxCheckedMul(s.srcRows - 1, s.expPhysical[1]);
  auto expRequired =
      expPrefix ? mxCheckedAdd(*expPrefix, s.srcCols / 32) : std::nullopt;
  if (!expRequired || failed(mxRequireCapacity(op, "exp", s.expTy, *expRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, s.groups))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
    if (s.isMxFp4) {
      return op.emitOpError("expects MXFP4 axis1 dst physical cols to equal src physical cols / 2");
    }
    return op.emitOpError("expects MXFP8 axis1 dst physical cols to equal src physical cols");
  }
  auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
  auto dstRequired =
      dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
  if (!dstRequired || failed(mxRequireCapacity(op, "dst", s.dstTy, *dstRequired))) {
    return failure();
  }
  if (failed(mxRequireCompact(op, "max", s.maxTy)) ||
      failed(mxRequireCompact(op, "scaling", s.scalingTy)) ||
      failed(mxRequireCapacity(op, "max", s.maxTy, s.groups))) {
    return failure();
  }
  const bool flat = s.expPhysical[0] == 1;
  if (flat) {
    return verifyTQuantMxAxis1Flat(op, s);
  }
  return verifyTQuantMxAxis1Canonical(op, s);
}
