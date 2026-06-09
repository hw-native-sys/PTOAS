// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyArithmeticC.cpp; kept as a fragment included by PTOVerifyArithmeticC.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

static FailureOr<TMovCommonInfo> verifyTMovCommon(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasFp && failed(verifyTileBufCommon(op, fp.getType(), "fp")))
    return failure();
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError(
               "expects fp and preQuantScalar forms to be mutually exclusive"),
           failure();
  }

  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace)
    return op.emitOpError("expects src and dst to have explicit address spaces"),
           failure();

  TMovCommonInfo info{
      srcTy, dstTy, fp, dyn_cast<pto::TileBufType>(srcTy),
      dyn_cast<pto::TileBufType>(dstTy), srcSpace, dstSpace, hasFp,
      hasPreQuantScalar};
  info.isMatToTile =
      *srcSpace == pto::AddressSpace::MAT &&
      (*dstSpace == pto::AddressSpace::LEFT ||
       *dstSpace == pto::AddressSpace::RIGHT ||
       *dstSpace == pto::AddressSpace::BIAS ||
       *dstSpace == pto::AddressSpace::SCALING);
  info.isVecToVec = *srcSpace == pto::AddressSpace::VEC &&
                    *dstSpace == pto::AddressSpace::VEC;
  info.isVecToMat = *srcSpace == pto::AddressSpace::VEC &&
                    *dstSpace == pto::AddressSpace::MAT;
  info.isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                    *dstSpace == pto::AddressSpace::MAT;
  info.isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                    *dstSpace == pto::AddressSpace::VEC;
  return info;
}

static LogicalResult verifyTMovShapes(TMovOp op, const TMovCommonInfo &info,
                                      bool isA5) {
  auto srcShape = getShapeVec(info.srcTy);
  auto dstShape = getShapeVec(info.dstTy);
  if (*info.srcSpace == pto::AddressSpace::MAT && srcShape != dstShape)
    return op.emitOpError(
        "expects mat-source tmov to use matching src/dst shapes");
  if (!isA5 && *info.srcSpace != pto::AddressSpace::MAT && srcShape != dstShape)
    return op.emitOpError(
        "expects A2/A3 non-mat tmov to use matching src/dst shapes");
  return success();
}

static LogicalResult verifyTMovAddressSpacePair(TMovOp op,
                                                const TMovCommonInfo &info,
                                                bool isA5) {
  bool okPair = info.isMatToTile || info.isVecToVec || info.isAccToMat ||
                info.isAccToVec || (isA5 && info.isVecToMat);
  if (!okPair)
    return op.emitOpError(
        "expects a supported tmov address-space pair for this target");
  if (op.getAccToVecModeAttr() && !info.isAccToVec) {
    return op.emitOpError(
        "expects accToVecMode to be used only for acc-to-vec tmov");
  }
  return success();
}

static LogicalResult verifyTMovDerivedForms(TMovOp op,
                                            const TMovCommonInfo &info) {
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      !(info.isAccToMat || info.isAccToVec)) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  if (info.hasPreQuantScalar && !(info.isAccToMat || info.isAccToVec)) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (!info.hasFp)
    return success();

  auto fpSpace = getPTOMemorySpaceEnum(info.fp.getType());
  if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
    return op.emitOpError("expects fp to be in the scaling address space");
  auto srcElemTy = getElemTy(info.srcTy);
  auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
  if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == kPTOI32BitWidth))) {
    return op.emitOpError("expects fp form src to have element type f32, i32");
  }
  if (!(info.isAccToMat || info.isAccToVec))
    return op.emitOpError("expects fp form to use loc=acc src");
  return success();
}

static LogicalResult verifyTMovAccToVecMode(TMovOp op,
                                            const TMovCommonInfo &info) {
  auto accToVecModeAttr = op.getAccToVecModeAttr();
  if (!(info.hasFp || info.hasPreQuantScalar) || !accToVecModeAttr)
    return success();
  switch (accToVecModeAttr.getValue()) {
  case pto::AccToVecMode::SingleModeVec0:
  case pto::AccToVecMode::SingleModeVec1:
    return success();
  case pto::AccToVecMode::DualModeSplitM:
  case pto::AccToVecMode::DualModeSplitN:
    return op.emitOpError(
        "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode");
  }
  return success();
}

static LogicalResult verifyTMovLayouts(TMovOp op, const TMovCommonInfo &info,
                                       bool isA5) {
  if (info.srcTb && *info.srcSpace == pto::AddressSpace::ACC &&
      (info.hasFp || op.getReluPreMode() != pto::ReluPreMode::NoRelu) &&
      !isColMajorRowMajorNZTileBuf(info.srcTb)) {
    return op.emitOpError(
        "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major");
  }
  if (info.srcTb && info.dstTb && info.isAccToMat && !isA5 &&
      info.dstTb.getSFractalSizeI32() != kFractalSize512) {
    return op.emitOpError(
        "expects A2/A3 acc-to-mat tmov destination fractal to be 512");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyImpl = [this](bool isA5) -> LogicalResult {
    auto common = verifyTMovCommon(*this);
    if (failed(common))
      return failure();
    if (failed(verifyTMovShapes(*this, *common, isA5)) ||
        failed(verifyTMovAddressSpacePair(*this, *common, isA5)) ||
        failed(verifyTMovDerivedForms(*this, *common)) ||
        failed(verifyTMovAccToVecMode(*this, *common)) ||
        failed(verifyTMovLayouts(*this, *common, isA5))) {
      return failure();
    }
    return success();
  };
  auto verifyA2A3 = [&verifyImpl]() -> LogicalResult {
    return verifyImpl(/*isA5=*/false);
  };
  auto verifyA5 = [&verifyImpl]() -> LogicalResult {
    return verifyImpl(/*isA5=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<TMovFpCommonInfo> verifyTMovFpCommon(TMovFPOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
  if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == kPTOI32BitWidth))) {
    return op.emitOpError("expects src to have element type f32, i32"),
           failure();
  }
  auto fpSpace = getPTOMemorySpaceEnum(fpTy);
  if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
    return op.emitOpError("expects fp to be in the scaling address space"),
           failure();
  return TMovFpCommonInfo{
      srcTy, fpTy, dstTy, srcElemTy, dyn_cast<pto::TileBufType>(srcTy),
      dyn_cast<pto::TileBufType>(dstTy), getPTOMemorySpaceEnum(srcTy), fpSpace,
      getPTOMemorySpaceEnum(dstTy)};
}

static LogicalResult verifyTMovFpA2A3(const TMovFpCommonInfo &info,
                                      TMovFPOp op) {
  if (!info.srcSpace || *info.srcSpace != pto::AddressSpace::ACC)
    return op.emitOpError("expects src to be in the acc address space");
  if (!info.dstSpace || *info.dstSpace != pto::AddressSpace::MAT)
    return op.emitOpError("expects dst to be in the mat address space");
  if (info.srcTb && !isColMajorRowMajorNZTileBuf(info.srcTb))
    return op.emitOpError(
        "expects src to use blayout=col_major and slayout=row_major");
  if (info.dstTb && !isColMajorRowMajorNZTileBuf(info.dstTb))
    return op.emitOpError(
        "expects dst to use blayout=col_major and slayout=row_major");
  if (info.dstTb && info.dstTb.getSFractalSizeI32() != kFractalSize512)
    return op.emitOpError("expects dst to use fractal size 512");
  return success();
}
