// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOParseAndSubview.cpp; kept as a fragment included by PTOParseAndSubview.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }

  auto verifyA2A3 = [this, hasMaskPattern]() -> LogicalResult {
    if (hasMaskPattern)
      return verifyTScatterMaskForm(*this);
    return verifyTScatterIndexedForm(*this);
  };
  auto verifyA5 = [this, hasMaskPattern]() -> LogicalResult {
    if (hasMaskPattern)
      return emitOpError("mask-pattern tscatter is not supported on A5 yet");
    return verifyTScatterIndexedForm(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifySelectElementType(Operation *op, Type elem,
                                             PTOArch targetArch,
                                             bool allowBf16,
                                             StringRef a2a3Message,
                                             StringRef a5Message) {
  bool ok = elem.isF16() || elem.isF32() || (allowBf16 && elem.isBF16());
  if (auto intTy = dyn_cast<IntegerType>(elem))
    ok = intTy.getWidth() == kPTOI16BitWidth || intTy.getWidth() == kPTOI32BitWidth ||
         (targetArch == PTOArch::A5 && intTy.getWidth() == kPTOI8BitWidth);
  if (ok)
    return success();
  if (targetArch == PTOArch::A5)
    return op->emitOpError(a5Message);
  return op->emitOpError(a2a3Message);
}

static FailureOr<Type> verifyTSelCommon(TSelOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !src1Elem || !dstElem) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (srcElem != src1Elem || srcElem != dstElem) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op.emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return srcElem;
}


mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A3, /*allowBf16=*/true,
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32",
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  };

  auto verifyA5 = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A5, /*allowBf16=*/true,
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32",
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static FailureOr<Type> verifyTSelSCommon(TSelSOp op) {
  Type maskTy = op.getMask().getType();
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, maskTy, "mask")) ||
      failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type maskElem = getElemTy(maskTy);
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = getElemTy(tmpTy);
  Type dstElem = getElemTy(dstTy);
  if (!maskElem || !srcElem || !tmpElem || !dstElem) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (srcElem != dstElem) {
    op.emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op.emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  return dstElem;
}


mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyA2A3 = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelSCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A3, /*allowBf16=*/false,
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32",
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  };

  auto verifyA5 = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelSCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A5, /*allowBf16=*/false,
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32",
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verify = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth &&
                it.getWidth() != kPTOI32BitWidth))
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/i16/i32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verify = [this]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != kPTOI8BitWidth && it.getWidth() != kPTOI16BitWidth &&
                it.getWidth() != kPTOI32BitWidth))
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/i16/i32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx")))
    return failure();
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!(srcElem.isF16() || srcElem.isF32()))
    return emitOpError() << "expects src and dst element type to be f16 or f32";

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != kPTOI32BitWidth)
    return emitOpError() << "expects idx element type to be i32/u32";
  return mlir::success();
}
