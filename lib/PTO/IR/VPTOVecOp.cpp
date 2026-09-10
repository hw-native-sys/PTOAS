// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOVecOp.cpp - VPTO vector compute ops -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

template <typename CarryOp>
static LogicalResult verifyCarryVecOp(CarryOp op) {
  if (failed(verifyIntegerVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getResult().getType(),
                                       "result type")) ||
      failed(verifyMaskTypeLike(op, op.getCarry().getType(), "carry type"))) {
    return failure();
  }

  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  auto lhsElemType = cast<IntegerType>(lhsType.getElementType());
  if (lhsType != rhsType || lhsType != resultType) {
    return op.emitOpError("requires lhs, rhs, and result to have matching vector types");
  }
  if (lhsElemType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("currently requires 32-bit integer vector elements");
  }
  return success();
}

template <typename PltOp>
static LogicalResult verifyPredicateLaneCountOp(PltOp op,
                                                StringRef granularity) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", granularity))) {
    return failure();
  }
  Type scalarType = op.getScalar().getType();
  auto scalarIntType = dyn_cast<IntegerType>(scalarType);
  if (!scalarIntType || scalarIntType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("requires scalar to be i32");
  }
  if (op.getScalarOut().getType() != scalarType) {
    return op.emitOpError("requires scalar_out to match scalar type");
  }
  return success();
}

template <typename PltmOp>
static LogicalResult verifyPredicateLoopBoundOp(PltmOp op,
                                                StringRef granularity) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", granularity))) {
    return failure();
  }
  if (!op.getLoop().getType().isInteger(mlir::pto::kValue16)) {
    return op.emitOpError("requires loop operand to be i16");
  }
  if (!op.getBound().getType().isInteger(mlir::pto::kValue32)) {
    return op.emitOpError("requires bound operand to be i32");
  }
  return success();
}

static bool isMaskGranularityAdjacentWidening(StringRef inputGranularity,
                                              StringRef resultGranularity) {
  return (inputGranularity == "b8" && resultGranularity == "b16") ||
         (inputGranularity == "b16" && resultGranularity == "b32");
}

static bool isMaskGranularityAdjacentNarrowing(StringRef inputGranularity,
                                               StringRef resultGranularity) {
  return (inputGranularity == "b16" && resultGranularity == "b8") ||
         (inputGranularity == "b32" && resultGranularity == "b16");
}

static bool isSupportedShuffleValueType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32 || intType.getWidth() == 64;
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return vecType.getRank() == 1 && vecType.getDimSize(0) == mlir::pto::kValue2 &&
           vecType.getElementType().isF16();
  }
  return type.isF16() || type.isF32();
}

static bool isSupportedReduxValueType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue32;
  }
  return type.isF16() || type.isF32();
}

static LogicalResult verifyShuffleSemanticControl(Operation *op,
                                                  Type controlType,
                                                  IntegerAttr widthAttr,
                                                  StringRef ctrlName) {
  if (!isSupportedShuffleValueType(op->getResultTypes().front())) {
    return op->emitOpError()
           << "requires i32, i64, f16, f32 or vector<2xf16> value/result type";
  }
  if (!controlType.isInteger(32)) {
    return op->emitOpError() << "requires " << ctrlName
                             << " operand to be i32";
  }

  int64_t width = widthAttr.getInt();
  if (width != mlir::pto::kValue16 && width != 32) {
    return op->emitOpError() << "requires width to be 16 or 32";
  }
  return success();
}

static LogicalResult verifyReduxSemanticType(Operation *op, Type valueType,
                                             Attribute signednessAttr,
                                             bool requireSignedness) {
  if (!isSupportedReduxValueType(valueType)) {
    return op->emitOpError()
           << "requires i32, f16 or f32 value/result type";
  }

  auto intType = dyn_cast<IntegerType>(valueType);
  if (!intType) {
    if (signednessAttr) {
      return op->emitOpError()
             << "does not accept signedness for floating-point redux";
    }
    return success();
  }

  if (!signednessAttr && requireSignedness) {
    return op->emitOpError()
           << "requires explicit signedness for integer redux";
  }

  if (!signednessAttr) {
    return success();
  }

  auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
  (void)signedness;
  return success();
}

LogicalResult ShuffleIdxOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getIndex().getType(),
                                      getWidthAttr(), "index");
}

LogicalResult ShuffleUpOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getOffset().getType(),
                                      getWidthAttr(), "offset");
}

LogicalResult ShuffleDownOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getOffset().getType(),
                                      getWidthAttr(), "offset");
}

LogicalResult ShuffleBflyOp::verify() {
  return verifyShuffleSemanticControl(getOperation(), getMask().getType(),
                                      getWidthAttr(), "mask");
}

LogicalResult ReduxAddOp::verify() {
  return verifyReduxSemanticType(getOperation(), getValue().getType(),
                                  getSignednessAttr(), /*requireSignedness=*/false);
}

LogicalResult ReduxMaxOp::verify() {
  return verifyReduxSemanticType(getOperation(), getValue().getType(),
                                  getSignednessAttr(), /*requireSignedness=*/true);
}

LogicalResult ReduxMinOp::verify() {
  return verifyReduxSemanticType(getOperation(), getValue().getType(),
                                  getSignednessAttr(), /*requireSignedness=*/true);
}

LogicalResult MulhiOp::verify() {
  if (!getResult().getType().isInteger(mlir::pto::kValue32) &&
      !getResult().getType().isInteger(mlir::pto::kValue64)) {
    return emitOpError() << "requires i32 or i64 result type";
  }
  return success();
}

LogicalResult MulI32ToI64Op::verify() { return success(); }

static bool isSupportedVdupPosition(std::optional<StringRef> position) {
  return !position || *position == "LOWEST" || *position == "HIGHEST";
}

bool isMxElementType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
         isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(type);
}

static std::optional<StringRef> getVdupMaskGranularity(Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case mlir::pto::kValue8:
      return StringRef("b8");
    case mlir::pto::kValue16:
      return StringRef("b16");
    case mlir::pto::kValue32:
      return StringRef("b32");
    default:
      return std::nullopt;
    }
  }
  if (elementType.isF16() || elementType.isBF16()) {
    return StringRef("b16");
  }
  if (elementType.isF32()) {
    return StringRef("b32");
  }
  return std::nullopt;
}

static bool isSupportedVtrcRoundMode(StringRef mode) {
  return mode == "R" || mode == "A" || mode == "F" || mode == "C" ||
         mode == "Z";
}

static bool isSupportedPartToken(StringRef part) {
  return part == "LOWER" || part == "HIGHER";
}

bool isSupportedPostMode(StringRef mode) {
  return mode == "NO_POST_UPDATE" || mode == "POST_UPDATE";
}

bool isCompatibleScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt || semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }

  return isSemanticSignCompatible(semanticInt, scalarInt);
}

LogicalResult VbrOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }

  auto resultVecType = cast<VRegType>(getResult().getType());
  Type elementType = getValue().getType();
  if (isa<ShapedType, VectorType>(elementType)) {
    return emitOpError("value must be a scalar matching the result element type");
  }
  Type resultElementType = resultVecType.getElementType();
  if (!isCompatibleScalarForSemanticType(resultElementType, elementType)) {
    return emitOpError("value type must match result element type");
  }
  return success();
}

template <typename ReductionOp>
static LogicalResult verifyWideningReductionVecOp(ReductionOp op,
                                                  StringRef opName) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result"))) {
    return failure();
  }

  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType) {
    return failure();
  }

  Type inputElemType = inputType.getElementType();
  Type expectedResultElemType = inputElemType;
  int64_t expectedResultLanes = inputType.getElementCount();
  if (auto inputInt = dyn_cast<IntegerType>(inputElemType)) {
    if (inputInt.getWidth() < mlir::pto::kValue8 || inputInt.getWidth() > 32) {
      return op.emitOpError(
          "requires 8-bit, 16-bit, or 32-bit integer vector element type");
    }
    if (inputInt.getWidth() == mlir::pto::kValue8) {
      expectedResultElemType =
          IntegerType::get(op.getContext(), mlir::pto::kValue16, inputInt.getSignedness());
      expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
    }
    if (inputInt.getWidth() == mlir::pto::kValue16) {
      expectedResultElemType =
          IntegerType::get(op.getContext(), mlir::pto::kValue32, inputInt.getSignedness());
      expectedResultLanes = inputType.getElementCount() / mlir::pto::kValue2;
    }
  } else if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return op.emitOpError("requires i16/i32/f16/f32 vector element type");
  }

  if (resultType.getElementCount() == expectedResultLanes &&
      resultType.getElementType() == expectedResultElemType) {
    return success();
  }

  return op.emitOpError() << opName << " expects result type !pto.vreg<"
                          << expectedResultLanes << "x"
                          << expectedResultElemType
                          << " for input element type " << inputElemType;
}

LogicalResult VcaddOp::verify() {
  return verifyWideningReductionVecOp(*this, "vcadd");
}

LogicalResult VcmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result must have the same vector type");
  }
  return success();
}

LogicalResult VcminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result must have the same vector type");
  }
  return success();
}

LogicalResult VciOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !pto.vreg<...>");
  }
  Type resultElemType = resultType.getElementType();
  bool supportedInteger = false;
  if (auto intType = dyn_cast<IntegerType>(resultElemType)) {
    supportedInteger = intType.getWidth() == mlir::pto::kValue8 || intType.getWidth() == 16 ||
                       intType.getWidth() == mlir::pto::kValue32;
  }
  bool supportedFloat = resultElemType.isF16() || resultElemType.isF32();
  if (!supportedInteger && !supportedFloat) {
    return emitOpError("result element type must be integer or f16/f32");
  }
  if (!isCompatibleScalarForSemanticType(resultElemType, getIndex().getType())) {
    return emitOpError("index type must match result element type");
  }
  return success();
}

LogicalResult VbitsortOp::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource().getType()) ||
      !isBufferLike(getIndices().getType())) {
    return emitOpError("requires pointer-like destination/source/indices");
  }
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getIndices().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed destination/source/indices");
  }
  if (!getRepeatTimes().getType().isIndex()) {
    return emitOpError("repeat_times must be index");
  }
  if (failed(verifyNotNestedInVecScope(*this, "pto.vbitsort"))) {
    return failure();
  }
  return success();
}

void VbitsortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndicesMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

// Batch7: Vmax/Vmin 共用三操作数同型校验
template <typename OpTy>
static LogicalResult verifySameVecTripleOp(OpTy op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("lhs, rhs, and result must have the same vector type");
  }
  return success();
}

LogicalResult VmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result"))) {
    return failure();
  }
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType()) {
    return emitOpError("lhs, rhs, and result must have the same vector type");
  }
  return success();
}

LogicalResult VminOp::verify() { return verifySameVecTripleOp(*this); }

LogicalResult VdupOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !pto.vreg<...>");
  }

  std::optional<StringRef> granularity =
      getVdupMaskGranularity(resultType.getElementType());
  if (!granularity) {
    return emitOpError("result element type must use b8, b16, or b32 mask granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(
          getOperation(), getMask().getType(), "mask type", *granularity))) {
    return failure();
  }

  if (!isSupportedVdupPosition(getPosition())) {
    return emitOpError("position must be LOWEST or HIGHEST");
  }

  Type inputType = getInput().getType();
  if (auto inputVecType = dyn_cast<VRegType>(inputType)) {
    if (inputVecType != resultType) {
      return emitOpError("vector input must match result vector type");
    }
    return success();
  }

  if (getPosition()) {
    return emitOpError("position is only supported for vector input");
  }

  Type resultElementType = resultType.getElementType();
  if (!isCompatibleScalarForSemanticType(resultElementType, inputType)) {
    return emitOpError("scalar input must match result element type");
  }

  return success();
}

LogicalResult PsetB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8"))) {
    return failure();
  }

  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PsetB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16"))) {
    return failure();
  }

  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PsetB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PgeB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PgeB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PgeB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32"))) {
    return failure();
  }
  if (!isSupportedPredicatePattern(getPattern())) {
    return emitOpError("requires a supported PAT_* predicate pattern");
  }
  return success();
}

LogicalResult PltB8Op::verify() { return verifyPredicateLaneCountOp(*this, "b8"); }
LogicalResult PltB16Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b16");
}
LogicalResult PltB32Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b32");
}

LogicalResult PltmB8Op::verify() {
  return verifyPredicateLoopBoundOp(*this, "b8");
}
LogicalResult PltmB16Op::verify() {
  return verifyPredicateLoopBoundOp(*this, "b16");
}
LogicalResult PltmB32Op::verify() {
  return verifyPredicateLoopBoundOp(*this, "b32");
}

// Batch12: Ppack/Punpack shared gran direction
template <typename OpTy>
static LogicalResult verifyPartGranularityOp(OpTy op, bool widening) {
  if (failed(verifyMaskTypeLike(op, op.getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isSupportedPartToken(op.getPart())) {
    return op.emitOpError("requires part to be LOWER or HIGHER");
  }
  auto inputMaskType = cast<MaskType>(op.getInput().getType());
  auto resultMaskType = cast<MaskType>(op.getResult().getType());
  StringRef inputGranularity = inputMaskType.getGranularity();
  StringRef resultGranularity = resultMaskType.getGranularity();
  bool adjacent = widening
                      ? isMaskGranularityAdjacentWidening(inputGranularity, resultGranularity)
                      : isMaskGranularityAdjacentNarrowing(inputGranularity, resultGranularity);
  if (inputGranularity != resultGranularity && !adjacent) {
    return op.emitOpError("requires result mask granularity to match the input or ")
           << (widening ? "widen" : "narrow") << " by one step";
  }
  return success();
}

LogicalResult PpackOp::verify() { return verifyPartGranularityOp(*this, /*widening=*/false); }

LogicalResult PunpackOp::verify() { return verifyPartGranularityOp(*this, /*widening=*/true); }

LogicalResult PbitcastOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

LogicalResult PnotOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

LogicalResult PselOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

template <typename BinaryMaskOp>
static LogicalResult verifyBinaryMaskOp(BinaryMaskOp op) {
  if (failed(verifyMaskTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  return success();
}

LogicalResult PandOp::verify() { return verifyBinaryMaskOp(*this); }
LogicalResult PorOp::verify() { return verifyBinaryMaskOp(*this); }
LogicalResult PxorOp::verify() { return verifyBinaryMaskOp(*this); }

template <typename OpTy>
static LogicalResult verifyElementwiseVecScalarOpLike(OpTy op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType) {
    return op.emitOpError("input and result must be !pto.vreg<...>");
  }
  if (inputType != resultType) {
    return op.emitOpError("input and result vector types must match");
  }

  Type elemType = inputType.getElementType();
  Type scalarType = op.getScalar().getType();
  if (scalarType == elemType) {
    return success();
  }

  auto elemInt = dyn_cast<IntegerType>(elemType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!elemInt || !scalarInt || elemInt.getWidth() != scalarInt.getWidth()) {
    return op.emitOpError("scalar type must match vector element type");
  }

  if (elemInt.isSigned() && (scalarInt.isSigned() || scalarInt.isSignless())) {
    return success();
  }
  if (elemInt.isUnsigned() &&
      (scalarInt.isUnsigned() || scalarInt.isSignless())) {
    return success();
  }
  if (elemInt.isSignless() && scalarInt.isSignless()) {
    return success();
  }

  return op.emitOpError(
      "integer scalar type must match vector element width and use matching signedness or signless i<width>");
}

template <typename OpTy>
[[maybe_unused]] static LogicalResult verifyVecScalarOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyVecScalarMaskedOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getInput().getType(), "input type"))) {
    return failure();
  }
  return success();
}

template <typename CarryWithInputOp>
static LogicalResult verifyCarryVecOpWithInput(CarryWithInputOp op) {
  if (failed(verifyCarryVecOp(op)) ||
      failed(verifyMaskTypeLike(op, op.getCarryIn().getType(),
                                "carry_in type"))) {
    return failure();
  }
  return success();
}

LogicalResult VmulsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VaddsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VmaxsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VminsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VlreluOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
// Batch7: Vshls/Vshrs 共用移位标量校验
template <typename OpTy>
static LogicalResult verifyShiftScalarVecOp(OpTy op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType) {
    return op.emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (inputType != resultType) {
    return op.emitOpError("input and result vector types must match");
  }
  if (!isa<IntegerType>(inputType.getElementType())) {
    return op.emitOpError("requires integer vector and integer scalar");
  }
  auto scalarType = dyn_cast<IntegerType>(op.getScalar().getType());
  if (!scalarType || !scalarType.isSignlessInteger(mlir::pto::kValue16)) {
    return op.emitOpError("requires signless i16 scalar");
  }
  return success();
}

LogicalResult VshlsOp::verify() { return verifyShiftScalarVecOp(*this); }
LogicalResult VshrsOp::verify() { return verifyShiftScalarVecOp(*this); }

LogicalResult VabsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "operand type"))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("requires matching register vector shape");
  }
  return success();
}

template <typename UnaryOp>
static LogicalResult verifyUnaryVecOp(UnaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "operand type"))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getInput().getType(), "operand type"))) {
    return failure();
  }
  if (op.getInput().getType() != op.getResult().getType()) {
    return op.emitOpError("requires matching register vector shape");
  }
  return success();
}

LogicalResult VexpOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VlnOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VsqrtOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VnegOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VreluOp::verify() {
  if (failed(verifyUnaryVecOp(*this))) {
    return failure();
  }
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() != mlir::pto::kValue32 || intType.isUnsigned()) {
      return emitOpError("requires si32/i32/f16/f32 vector element type");
    }
    return success();
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires si32/i32/f16/f32 vector element type");
  }
  return success();
}
LogicalResult VnotOp::verify() { return verifyUnaryVecOp(*this); }

template <typename BinaryOp>
static LogicalResult verifyBinaryVecOp(BinaryOp op,
                                       bool allowLowPrecision = false) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type"))) {
    return failure();
  }
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type"))) {
    return failure();
  }
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (!allowLowPrecision &&
      failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getLhs().getType(), "lhs type"))) {
    return failure();
}
  if (allowLowPrecision) {
    auto lhsType = cast<VRegType>(op.getLhs().getType());
    if (pto::isPTOBF16x2Type(lhsType.getElementType())) {
      return op.emitOpError(
          "does not support bf16x2 vector elements; low-precision bitwise "
          "operations require an 8-bit payload type");
}
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires matching register vector shapes");
}
  return success();
}

LogicalResult VaddOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VsubOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VmulOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VdivOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VandOp::verify() {
  return verifyBinaryVecOp(*this, /*allowLowPrecision=*/true);
}
LogicalResult VorOp::verify() {
  return verifyBinaryVecOp(*this, /*allowLowPrecision=*/true);
}
LogicalResult VxorOp::verify() {
  return verifyBinaryVecOp(*this, /*allowLowPrecision=*/true);
}

template <typename TernaryOp>
static LogicalResult verifyTernaryVecOp(TernaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getAcc().getType() != op.getLhs().getType() ||
      op.getAcc().getType() != op.getRhs().getType() ||
      op.getAcc().getType() != op.getResult().getType()) {
    return op.emitOpError(
        "requires acc, lhs, rhs, and result to share one vector type");
  }
  return success();
}

LogicalResult VmaddOp::verify() {
  if (failed(verifyTernaryVecOp(*this))) {
    return failure();
  }
  Type elemType = cast<VRegType>(getAcc().getType()).getElementType();
  if (!elemType.isF16() && !elemType.isBF16() && !elemType.isF32()) {
    return emitOpError("requires f16/bf16/f32 vector element type");
  }
  return success();
}
// Shared verifier for vshl/vshr. Shift counts use a signed integer vreg.
// DSL frontends must bitcast signless or unsigned count vectors to this
// canonical form.
template <typename BinaryOp>
static LogicalResult verifyShiftVecOp(BinaryOp op) {
  const bool hasInvalidOperandType =
      failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"));
  if (hasInvalidOperandType) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          op.getOperation(), op.getLhs().getType(), "lhs type"))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  // Shifting is only meaningful for integer vectors.
  if (!isa<IntegerType>(lhsType.getElementType())) {
    return op.emitOpError("requires integer vector element type");
  }
  // Result type must match lhs exactly.
  if (lhsType != resultType) {
    return op.emitOpError("requires matching result register vector shape");
  }
  // Shift count must have the same lane count and element bitwidth as the
  // shifted data.
  const bool hasMismatchedLaneCount =
      lhsType.getElementCount() != rhsType.getElementCount();
  if (hasMismatchedLaneCount) {
    return op.emitOpError("requires matching lane count for shift count");
  }
  auto lhsElem = cast<IntegerType>(lhsType.getElementType());
  auto rhsElem = dyn_cast<IntegerType>(rhsType.getElementType());
  if (!rhsElem) {
    return op.emitOpError(
        "requires integer vector element type for shift count");
  }
  const bool hasMismatchedElementBitwidth =
      rhsElem.getWidth() != lhsElem.getWidth();
  if (hasMismatchedElementBitwidth) {
    return op.emitOpError(
        "requires shift count with matching element bitwidth");
  }
  if (!rhsElem.isSigned()) {
    return op.emitOpError(
        "requires shift count to use a signed integer element type");
  }
  return success();
}

LogicalResult VshlOp::verify() { return verifyShiftVecOp(*this); }
LogicalResult VshrOp::verify() { return verifyShiftVecOp(*this); }
LogicalResult VaddcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VsubcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VaddcsOp::verify() { return verifyCarryVecOpWithInput(*this); }
LogicalResult VsubcsOp::verify() { return verifyCarryVecOpWithInput(*this); }

template <typename ReductionOp>
static LogicalResult verifyReductionVecOp(ReductionOp op) {
  return verifyUnaryVecOp(op);
}

template <typename ReductionOp>
static LogicalResult verifyGroupReductionVecOp(ReductionOp op) {
  if (failed(verifyReductionVecOp(op))) {
    return failure();
  }
  auto inputType = cast<VRegType>(op.getInput().getType());
  Type elemType = inputType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() != mlir::pto::kValue8 && intType.getWidth() != 16 &&
        intType.getWidth() != mlir::pto::kValue32) {
      return op.emitOpError(
          "requires 8-bit, 16-bit, or 32-bit integer vector element type");
    }
    return success();
  }
  if (!elemType.isF16() && !elemType.isF32()) {
    return op.emitOpError("requires i16/i32/f16/f32 vector element type");
  }
  return success();
}

LogicalResult VcgaddOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcgmaxOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcgminOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcpaddOp::verify() {
  if (failed(verifyReductionVecOp(*this))) {
    return failure();
  }
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires f16 or f32 vector element type");
  }
  return success();
}

template <typename ExtremaOp>
static LogicalResult verifyExtremaPredicateOp(ExtremaOp op) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type")) ||
      failed(verifyMaskTypeLike(op, op.getPredicate().getType(),
                                "predicate type"))) {
    return failure();
  }
  if (op.getInput().getType() != op.getValue().getType()) {
    return op.emitOpError(
        "requires input and value result to share one vector type");
  }
  if (op.getMask().getType() != op.getPredicate().getType()) {
    return op.emitOpError(
        "requires mask and predicate result to share one mask type");
  }

  Type elemType = cast<VRegType>(op.getInput().getType()).getElementType();
  if (elemType.isF16() || elemType.isF32()) {
    return success();
  }
  auto intType = dyn_cast<IntegerType>(elemType);
  if (!intType || (intType.getWidth() != mlir::pto::kValue8 && intType.getWidth() != 16 &&
                   intType.getWidth() != mlir::pto::kValue32)) {
    return op.emitOpError("requires i8/i16/i32/f16/f32 vector element type");
  }
  return success();
}

LogicalResult VcbmaxOp::verify() { return verifyExtremaPredicateOp(*this); }
LogicalResult VcbminOp::verify() { return verifyExtremaPredicateOp(*this); }

template <typename SelectOp>
static LogicalResult verifyLaneSelectOp(SelectOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }

  auto src0Type = cast<VRegType>(op.getSrc0().getType());
  auto src1Type = cast<VRegType>(op.getSrc1().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (src0Type != resultType) {
    return op.emitOpError("requires src0 and result to have identical vector types");
  }
  if (src1Type.getElementCount() != src0Type.getElementCount()) {
    return op.emitOpError("requires src0/src1 to have identical element counts");
  }
  auto src1ElemType = dyn_cast<IntegerType>(src1Type.getElementType());
  if (!src1ElemType) {
    return op.emitOpError("requires src1 to use integer vector elements");
  }
  if (src1ElemType.getWidth() != getIntOrFloatBitWidth(src0Type.getElementType())) {
    return op.emitOpError("requires src1 integer element width to match src0 element width");
  }
  return success();
}

template <typename PairOp>
static LogicalResult verifyPairVecResults(PairOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(op, op.getHigh().getType(), "high result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getLow().getType() ||
      op.getLhs().getType() != op.getHigh().getType()) {
    return op.emitOpError("requires operands and results to share one vector type");
  }
  return success();
}

template <typename PartOp>
static LogicalResult verifyPartVecOp(PartOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires operands and result to share one vector type");
  }
  if (!isSupportedPartToken(op.getPart())) {
    return op.emitOpError("requires part to be LOWER or HIGHER");
  }
  return success();
}

LogicalResult VselOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (failed(verifyNonLowPrecisionVRegElementTypeLike(
          getOperation(), getSrc0().getType(), "src0 type"))) {
    return failure();
  }
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType()) {
    return emitOpError("requires src0, src1, and result to have identical vector types");
  }
  return success();
}

LogicalResult VselrOp::verify() { return verifyLaneSelectOp(*this); }
LogicalResult Vselrv2Op::verify() { return verifyLaneSelectOp(*this); }

LogicalResult VsqzOp::verify() { return verifyUnaryVecOp(*this); }

LogicalResult VusqzOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getSrc().getType() != getResult().getType()) {
    return emitOpError("requires src and result to share one vector type");
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  auto elemType = dyn_cast<IntegerType>(srcType.getElementType());
  if (!elemType) {
    return emitOpError("requires signed integer vector element type");
  }
  if (elemType.isUnsigned()) {
    return emitOpError("requires signed integer vector element type");
  }
  unsigned width = elemType.getWidth();
  if (width != mlir::pto::kValue8 && width != 16 && width != mlir::pto::kValue32) {
    return emitOpError("requires s8/s16/s32 vector element type");
  }
  return success();
}

LogicalResult VpackOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!isSupportedPartToken(getPart())) {
    return emitOpError("requires part to be LOWER or HIGHER");
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType)) {
    return emitOpError("currently requires integer source and result element types");
  }
  if (resultType.getElementCount() != srcType.getElementCount() * mlir::pto::kValue2) {
    return emitOpError(
        "requires result element count to be twice the source element count");
  }
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (!srcWidth || resultWidth * mlir::pto::kValue2 != srcWidth) {
    return emitOpError(
        "requires result element width to be half the source element width");
  }
  auto srcIntType = cast<IntegerType>(srcElemType);
  auto resultIntType = cast<IntegerType>(resultElemType);
  if (!resultIntType.isUnsigned()) {
    return emitOpError("requires unsigned result element type");
  }
  if (!((srcIntType.getWidth() == mlir::pto::kValue32 &&
         resultIntType.getWidth() == mlir::pto::kValue16) ||
        (srcIntType.getWidth() == mlir::pto::kValue16 &&
         resultIntType.getWidth() == mlir::pto::kValue8))) {
    return emitOpError(
        "currently supports only s32/u32 -> u16 and s16/u16 -> u8");
  }
  return success();
}

template <typename UnpackOp>
static LogicalResult verifyUnpackVecOp(UnpackOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  auto srcType = cast<VRegType>(op.getSrc().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType)) {
    return op.emitOpError(
        "currently requires integer source and result element types");
  }
  if (srcType.getElementCount() != resultType.getElementCount() * 2) {
    return op.emitOpError(
        "requires source element count to be twice the result element count");
  }
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (!srcWidth || srcWidth * mlir::pto::kValue2 != resultWidth) {
    return op.emitOpError(
        "requires result element width to be twice the source element width");
  }
  return success();
}

LogicalResult VsunpackOp::verify() { return verifyUnpackVecOp(*this); }
LogicalResult VzunpackOp::verify() { return verifyUnpackVecOp(*this); }

static bool isSupportedCmpMode(StringRef mode) {
  return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
         mode == "gt" || mode == "ge";
}

LogicalResult VcmpOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getSrc0().getType() != getSrc1().getType()) {
    return emitOpError("requires src0 and src1 to have identical vector types");
  }
  if (!isSupportedCmpMode(getCmpMode())) {
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  }
  return success();
}

LogicalResult VcmpsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  auto srcType = cast<VRegType>(getSrc().getType());
  Type srcElementType = srcType.getElementType();
  Type scalarType = getScalar().getType();
  if (!isCompatibleScalarForSemanticType(srcElementType, scalarType)) {
    return emitOpError("requires scalar type to match source element type");
  }
  if (!isSupportedCmpMode(getCmpMode())) {
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  }
  return success();
}

ParseResult VtrcOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  OpAsmParser::UnresolvedOperand mask;
  std::string roundModeToken;
  NamedAttrList attrs;
  Type inputType, maskType, resultType;

  if (parser.parseOperand(input) || parser.parseComma() ||
      parser.parseOperand(mask) || parser.parseComma() ||
      parser.parseKeywordOrString(&roundModeToken) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseComma() ||
      parser.parseType(maskType) || parser.parseArrow() ||
      parser.parseType(resultType)) {
    return failure();
  }

  auto normalized = normalizeRoundModeToken(roundModeToken);
  if (!normalized || !isSupportedVtrcRoundMode(*normalized)) {
    return parser.emitError(parser.getCurrentLocation())
           << "round mode must be one of R/A/F/C/Z or "
              "ROUND_R/ROUND_A/ROUND_F/ROUND_C/ROUND_Z";
  }

  attrs.set("round_mode", parser.getBuilder().getStringAttr(*normalized));
  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands) ||
      parser.resolveOperand(mask, maskType, result.operands)) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void VtrcOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << ", " << getMask() << ", ";
  Builder builder(getContext());
  auto normalized = normalizeRoundModeToken(getRoundMode());
  printer.printAttributeWithoutType(
      builder.getStringAttr(normalized.value_or(getRoundMode())));
  printer.printOptionalAttrDict((*this)->getAttrs(), {"round_mode"});
  printer << " : " << getInput().getType() << ", " << getMask().getType()
          << " -> " << getResult().getType();
}

// Shared granularity-derivation + mask-type check for ops whose mask
// granularity follows the vector element type.
template <typename OpTy>
static LogicalResult verifyVdupMaskGranularityLike(OpTy op, Type elemType) {
  auto expectedGranularity = getVdupMaskGranularity(elemType);
  if (!expectedGranularity) {
    return op.emitOpError(
        "requires element type with supported predicate granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type",
                                               *expectedGranularity))) {
    return failure();
  }
  return success();
}

LogicalResult VtrcOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  if (inputType != resultType) {
    return emitOpError("requires input and result to have identical vreg type");
  }
  auto elemType = inputType.getElementType();
  if (!(elemType.isF16() || elemType.isF32() || elemType.isBF16())) {
    return emitOpError("requires f16/f32/bf16 vector element type");
  }
  if (failed(verifyVdupMaskGranularityLike(*this, elemType))) {
    return failure();
  }
  auto normalized = normalizeRoundModeToken(getRoundMode());
  if (!normalized || !isSupportedVtrcRoundMode(*normalized)) {
    return emitOpError("round mode must be one of R/A/F/C/Z");
  }
  return success();
}

LogicalResult VmulscvtOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }

  auto inputType = cast<VRegType>(getInput().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (!inputType.getElementType().isF32()) {
    return emitOpError("requires f32 input vector element type");
  }
  if (!resultType.getElementType().isF16()) {
    return emitOpError("requires f16 result vector element type");
  }

  auto scalarType = getScalar().getType();
  if (!scalarType.isF32()) {
    return emitOpError("requires f32 scalar operand");
  }

  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type", "b32"))) {
    return failure();
  }

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits) {
    return emitOpError(
        "requires source and result to preserve total vector storage width");
  }

  auto normalizedRnd = normalizeRoundModeToken(getRnd());
  if (!normalizedRnd) {
    return emitOpError("rnd must be one of R/A/F/C/Z/O");
  }
  if (*normalizedRnd != "A") {
    return emitOpError("currently only supports rnd A");
  }

  auto normalizedPart = normalizeEvenOddPartToken(getPart());
  if (!normalizedPart) {
    return emitOpError("part must be EVEN or ODD");
  }
  return success();
}

LogicalResult VbitcastOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType) {
    return emitOpError("input and result must be !pto.vreg<...>");
  }

  auto getStorageBits = [](VRegType type) -> std::optional<int64_t> {
    Type elementType = type.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      return type.getElementCount() * static_cast<int64_t>(intType.getWidth());
    }
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      return type.getElementCount() *
             static_cast<int64_t>(floatType.getWidth());
}
    // Packed PTO element types (f8/hif8/f4x2/bf16x2/...) have a known storage
    // width even though they are not IntegerType/FloatType.
    unsigned packedBits = pto::getPTOStorageElemBitWidth(elementType);
    if (packedBits != 0) {
      return type.getElementCount() * static_cast<int64_t>(packedBits);
}
    return std::nullopt;
  };

  auto inputBits = getStorageBits(inputType);
  auto resultBits = getStorageBits(resultType);
  if (!inputBits || !resultBits) {
    return emitOpError("requires integer or floating-point vreg element type");
  }
  if (*inputBits != *resultBits) {
    return emitOpError("requires source and result vectors to carry the same "
                       "total number of bits");
  }

  return success();
}

// Batch6: Pdintlv/Pintlv 双胞胎 op 共用 gran 校验
template <typename OpTy>
static LogicalResult verifyPackedInterleaveGranularity(OpTy op,
                                                       llvm::StringRef gran) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getLhs().getType(),
                                               "lhs type", gran)) ||
      failed(verifyMaskTypeWithGranularityLike(op, op.getRhs().getType(),
                                               "rhs type", gran)) ||
      failed(verifyMaskTypeWithGranularityLike(op, op.getLow().getType(),
                                               "low type", gran)) ||
      failed(verifyMaskTypeWithGranularityLike(op, op.getHigh().getType(),
                                               "high type", gran))) {
    return failure();
  }
  return success();
}

LogicalResult PdintlvB8Op::verify() {
  return verifyPackedInterleaveGranularity(*this, "b8");
}

LogicalResult PdintlvB16Op::verify() {
  return verifyPackedInterleaveGranularity(*this, "b16");
}

LogicalResult PdintlvB32Op::verify() {
  return verifyPackedInterleaveGranularity(*this, "b32");
}

LogicalResult PintlvB8Op::verify() {
  return verifyPackedInterleaveGranularity(*this, "b8");
}

LogicalResult PintlvB16Op::verify() {
  return verifyPackedInterleaveGranularity(*this, "b16");
}

LogicalResult PintlvB32Op::verify() {
  return verifyPackedInterleaveGranularity(*this, "b32");
}


LogicalResult VintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult VdintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult Vintlvv2Op::verify() { return verifyPartVecOp(*this); }
LogicalResult Vdintlvv2Op::verify() { return verifyPartVecOp(*this); }

LogicalResult VmullOp::verify() {
  if (failed(verifyPairVecResults(*this)) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(getLhs().getType());
  auto lhsElemType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!lhsElemType) {
    return emitOpError("requires integer vector element type");
  }
  if (lhsElemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("currently requires 32-bit integer vector elements");
  }
  return success();
}

LogicalResult VmulaOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (getAcc().getType() != getLhs().getType() ||
      getAcc().getType() != getRhs().getType() ||
      getAcc().getType() != getResult().getType()) {
    return emitOpError("requires acc, lhs, rhs, and result to share one vector type");
  }
  return success();
}

template <typename BinaryVecNoMaskOp>
static LogicalResult verifyBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires lhs, rhs, and result to share one vector type");
  }
  return success();
}

template <typename BinaryVecNoMaskOp>
[[maybe_unused]] static LogicalResult verifyFloatBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
  if (failed(verifyBinaryVecNoMaskOp(op))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  Type elemType = lhsType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return op.emitOpError("requires f16 or f32 vector element type");
  }
  return success();
}

template <typename BinaryVecMaskOp>
static LogicalResult verifyFloatBinaryVecMaskOp(BinaryVecMaskOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType()) {
    return op.emitOpError("requires lhs, rhs, and result to share one vector type");
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  Type elemType = lhsType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return op.emitOpError("requires f16 or f32 vector element type");
  }
  return success();
}

LogicalResult VpreluOp::verify() { return verifyFloatBinaryVecMaskOp(*this); }
LogicalResult VexpdifOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyVRegTypeLike(*this, getMax().getType(), "max type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }

  auto inputType = cast<VRegType>(getInput().getType());
  auto maxType = cast<VRegType>(getMax().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (inputType != maxType) {
    return emitOpError("requires input and max to share one vector type");
  }

  Type inputElemType = inputType.getElementType();
  if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return emitOpError("requires f16 or f32 input vector element type");
  }
  auto expectedGranularity = getVdupMaskGranularity(inputElemType);
  if (!expectedGranularity) {
    return emitOpError("requires input element type with supported predicate granularity");
  }
  if (failed(verifyMaskTypeWithGranularityLike(*this, getMask().getType(),
                                               "mask type",
                                               *expectedGranularity))) {
    return failure();
  }
  if (!resultType.getElementType().isF32()) {
    return emitOpError("requires f32 result vector element type");
  }

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits) {
    return emitOpError(
        "requires source and result to preserve total vector storage width");
  }

  StringRef part = getPart();
  if (part != "EVEN" && part != "ODD") {
    return emitOpError("part must be EVEN or ODD");
  }
  return success();
}

LogicalResult VaxpyOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type"))) {
    return failure();
  }
  auto src0Type = cast<VRegType>(getSrc0().getType());
  auto src1Type = cast<VRegType>(getSrc1().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (src0Type != src1Type || src0Type != resultType) {
    return emitOpError("requires src0, src1, and result to share one vector type");
  }
  Type elemType = src0Type.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return emitOpError("requires f16 or f32 vector element type");
  }
  if (failed(verifyVdupMaskGranularityLike(*this, elemType))) {
    return failure();
  }
  if (getAlpha().getType() != elemType) {
    return emitOpError("requires alpha type to match vector element type");
  }
  return success();
}

template <typename ConvOp>
static LogicalResult verifyFusedConvVecOp(ConvOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type"))) {
    return failure();
  }
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (lhsType != rhsType) {
    return op.emitOpError("requires lhs and rhs to share one vector type");
  }
  if (!isIntegerOrFloatLike(lhsType.getElementType()) ||
      !isIntegerOrFloatLike(resultType.getElementType())) {
    return op.emitOpError(
        "requires integer or floating-point vector element types");
  }
  auto lhsBits = getVRegStorageBitWidth(lhsType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!lhsBits || !resultBits || *lhsBits != *resultBits) {
    return op.emitOpError(
        "requires source and result to preserve total vector storage width");
  }
  return success();
}

LogicalResult VaddreluconvOp::verify() {
  return verifyFusedConvVecOp(*this);
}
LogicalResult VmulconvOp::verify() { return verifyFusedConvVecOp(*this); }

void VsldbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VsldbOp::verify() {
  if (!isBufferLike(getSource().getType())) {
    return emitOpError("requires a pointer-like source");
  }
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed source");
  }
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type"))) {
    return failure();
  }
  if (!getBlockStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires block_stride to be i16");
  }
  if (!getRepeatStride().getType().isSignlessInteger(mlir::pto::kValue16)) {
    return emitOpError("requires repeat_stride to be i16");
  }
  if (getUpdatedBase() &&
      getUpdatedBase().getType() != getSource().getType()) {
    return emitOpError("requires updated base result to match base type");
  }
  return success();
}

void VstarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstarOp::verify() {
  if (failed(verifyStoreAlignChain(getValue(), *this, "value type"))) {
    return failure();
  }
  if (!isBufferLike(getDestination().getType())) {
    return emitOpError("requires a pointer-like destination");
  }
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM) {
    return emitOpError("requires a UB-backed destination");
  }
  return success();
}
