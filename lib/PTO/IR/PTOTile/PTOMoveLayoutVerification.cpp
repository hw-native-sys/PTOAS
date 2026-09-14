// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static bool isVecTile(Type type) {
  auto space = getPTOMemorySpaceEnum(type);
  return isa<pto::TileBufType>(type) && space &&
         *space == pto::AddressSpace::VEC;
}

static LogicalResult verifyTMovXToZzForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (!isA5) {
    return op.emitOpError("X-to-ZZ tmov is only supported on A5");
  }
  if (op.getNumResults() != 0) {
    return op.emitOpError("expects X-to-ZZ tmov not to have results");
  }
  if (op.getPreQuantScalar() || op.getAccToVecModeAttr() ||
      op.getReluPreMode() != pto::ReluPreMode::NoRelu) {
    return op.emitOpError("expects the X-to-ZZ tmov form not to use preQuantScalar, accToVecMode, or reluPreMode");
  }

  if (!isVecTile(srcTy) || !isVecTile(dstTy) || !isVecTile(fp.getType())) {
    return op.emitOpError("expects X-to-ZZ src/dst/tmp to be vec tiles");
  }
  if (op.getSrc() == op.getDst() || op.getSrc() == fp || op.getDst() == fp) {
    return op.emitOpError("expects X-to-ZZ src, dst, and tmp to be distinct tile values");
  }
  if (cast<pto::TileBufType>(srcTy).getRank() != 2 ||
      cast<pto::TileBufType>(dstTy).getRank() != 2 ||
      cast<pto::TileBufType>(fp.getType()).getRank() != 2) {
    return op.emitOpError("expects rank-2 valid_shape for src/dst/tmp");
  }
  if (!isStaticTMovShape(srcTy, true) || !isStaticTMovShape(dstTy, true) ||
      !isStaticTMovShape(fp.getType(), false)) {
    return op.emitOpError("expects static valid and physical shapes for src/dst and a static tmp physical shape for X-to-ZZ");
  }
  return verifyTMovXToZzElemLayout(op);
}

static LogicalResult verifyTMovXToZzAxis1(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> dstValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (dstValid[1] % 2 != 0) {
    return op.emitOpError("expects ND-to-ZZ dst valid_shape[1] (the grouped exponent column count) to be even");
  }
  if (srcValid[0] != 1 && srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects ND-to-ZZ src valid elements to form a compact prefix (single-row legacy flat or physical row stride equal to valid cols)");
  }
  auto paddedRows = tmovAlign16(dstValid[0]);
  auto required =
      paddedRows ? tmovCheckedMul(*paddedRows, dstValid[1]) : std::nullopt;
  if (!required) {
    return op.emitOpError("cannot compute ND-to-ZZ padded capacity without overflow");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  if (!srcBytes || *srcBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ src physical capacity to cover align16(dst rows) * dst cols because source padding is zeroed in place");
  }
  if (!dstBytes || *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ dst physical capacity to cover align16(dst rows) * dst cols");
  }
  auto rowBlocksBias = tmovCheckedAdd(dstValid[0], 15);
  auto offsetBytes = rowBlocksBias
                         ? tmovCheckedMul(*rowBlocksBias / 16, dstValid[1])
                         : std::nullopt;
  auto tmpRequired =
      offsetBytes ? tmovCheckedAdd(64, *offsetBytes) : std::nullopt;
  if (!tmpRequired) {
    return op.emitOpError("cannot compute ND-to-ZZ tmp capacity without overflow");
  }
  auto tmpBytes = getStaticByteSize(fp.getType());
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(*tmpRequired)) {
    return op.emitOpError() << "expects tmp to provide at least " << *tmpRequired
                            << " bytes for ND-to-ZZ (64 + ceil(dst rows / 16) * dst cols)";
  }
  return success();
}

static LogicalResult verifyTMovXToZzAxis0(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (srcValid[0] < 2 || srcValid[0] % 2 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[0] to be an even count >= 2; a single row-group produces no output in PTO-ISA");
  }
  if (srcValid[1] % 16 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[1] to be a multiple of 16");
  }
  if (srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects DN-to-ZZ src physical row stride to equal src valid_shape[1]");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  auto required = tmovCheckedMul(srcValid[0], srcValid[1]);
  if (!required || !srcBytes || !dstBytes ||
      *srcBytes < static_cast<uint64_t>(*required) ||
      *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects DN-to-ZZ src/dst physical capacity to cover src valid rows * src valid cols");
  }
  return success();
}

static LogicalResult verifyTMovXToZzCapacity(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  auto srcPhysical = getShapeVec(srcTy);
  auto srcElements = tmovCheckedElements(srcValid);
  auto dstElements = tmovCheckedElements(dstValid);
  if (!srcElements || !dstElements || *srcElements != *dstElements) {
    return op.emitOpError("expects src and dst to hold the same exponent count");
  }
  const MxGroupAxis axis =
      op.getGrpAxisAttr() ? op.getGrpAxisAttr().getValue() : MxGroupAxis::Axis1;
  if (axis == MxGroupAxis::Axis1) {
    return verifyTMovXToZzAxis1(op, srcValid, dstValid, srcPhysical);
  }
  return verifyTMovXToZzAxis0(op, srcValid, srcPhysical);
}

static LogicalResult verifyTMovXToZz(TMovOp op, bool isA5) {
  if (failed(verifyTMovXToZzForm(op, isA5))) {
    return failure();
  }
  return verifyTMovXToZzCapacity(op);
}

static LogicalResult verifyTMovGenericPreconditions(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  if (op.getGrpAxisAttr()) {
    return op.emitOpError("expects grpAxis only on the X-to-ZZ form with a non-scaling third tile");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", /*allowLowPrecision=*/isA5)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && failed(verifyTileBufCommon(op, fp.getType(), "fp",
                                          /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && op.getPreQuantScalar()) {
    return op.emitOpError() << "expects fp and preQuantScalar forms to be mutually exclusive";
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace) {
    return op.emitOpError() << "expects src and dst to have explicit address spaces";
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects mat-source tmov to use matching src/dst shapes";
  }
  if (!isA5 && *srcSpace != pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects A2/A3 non-mat tmov to use matching src/dst shapes";
  }
  return success();
}

static LogicalResult verifyTMovGenericPairing(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  using AddressPair = std::pair<pto::AddressSpace, pto::AddressSpace>;
  SmallVector<AddressPair> supported = {
      {pto::AddressSpace::MAT, pto::AddressSpace::LEFT},
      {pto::AddressSpace::MAT, pto::AddressSpace::RIGHT},
      {pto::AddressSpace::MAT, pto::AddressSpace::BIAS},
      {pto::AddressSpace::MAT, pto::AddressSpace::SCALING},
      {pto::AddressSpace::VEC, pto::AddressSpace::VEC},
      {pto::AddressSpace::ACC, pto::AddressSpace::MAT},
      {pto::AddressSpace::ACC, pto::AddressSpace::VEC}};
  if (isA5)
    supported.push_back({pto::AddressSpace::VEC, pto::AddressSpace::MAT});
  bool okPair = llvm::is_contained(supported, AddressPair(*srcSpace, *dstSpace));
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  if (!okPair) {
    return op.emitOpError() << "expects a supported tmov address-space pair for this target";
  }
  if (op.getAccToVecModeAttr() && !isAccToVec) {
    return op.emitOpError() << "expects accToVecMode to be used only for acc-to-vec tmov";
  }
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects reluPreMode form to use loc=acc src";
  }
  if (op.getPreQuantScalar() && !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects preQuantScalar form to use loc=acc src";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpLayout(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  const bool hasFp = static_cast<bool>(op.getFp());
  auto reluMode = op.getReluPreMode();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (srcTb && *srcSpace == pto::AddressSpace::ACC &&
      (hasFp || reluMode != pto::ReluPreMode::NoRelu)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError() << "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major";
    }
  }
  if (hasFp && !isA5 && dstTb && isAccToMat &&
      (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
       dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))) {
    return op.emitOpError() << "expects fp tmov dst to use blayout=col_major and slayout=row_major";
  }
  if (srcTb && dstTb && isAccToMat && !isA5 &&
      dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  auto accToVecModeAttr = op.getAccToVecModeAttr();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  if (hasFp) {
    auto fpSpace = getPTOMemorySpaceEnum(fp.getType());
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError() << "expects fp to be in the scaling address space";
    }
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32))) {
      return op.emitOpError() << "expects fp form src to have element type f32, i32";
    }
    if (!(isAccToMat || isAccToVec)) {
      return op.emitOpError() << "expects fp form to use loc=acc src";
    }
  }
  if ((hasFp || hasPreQuantScalar) && accToVecModeAttr) {
    switch (accToVecModeAttr.getValue()) {
    case pto::AccToVecMode::SingleModeVec0:
    case pto::AccToVecMode::SingleModeVec1:
      break;
    case pto::AccToVecMode::DualModeSplitM:
    case pto::AccToVecMode::DualModeSplitN:
      return op.emitOpError() << "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode";
    }
  }
  return verifyTMovGenericFpLayout(op, isA5);
}
