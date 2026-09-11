// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOpInternal.h - shared VPTOVecOp helpers ---------------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
// Internal to lib/PTO/IR/VPTO/vecop; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VPTOVECOP_INTERNAL_H
#define PTO_IR_VPTO_VPTOVECOP_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

  template <typename CarryOp>
  [[maybe_unused]] static LogicalResult verifyCarryVecOp(CarryOp op) {
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
  [[maybe_unused]] static LogicalResult verifyPredicateLaneCountOp(PltOp op,
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
  [[maybe_unused]] static LogicalResult verifyPredicateLoopBoundOp(PltmOp op,
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

  [[maybe_unused]] static bool isMaskGranularityAdjacentWidening(StringRef inputGranularity,
                                                StringRef resultGranularity) {
    return (inputGranularity == "b8" && resultGranularity == "b16") ||
           (inputGranularity == "b16" && resultGranularity == "b32");
  }

  [[maybe_unused]] static bool isMaskGranularityAdjacentNarrowing(StringRef inputGranularity,
                                                 StringRef resultGranularity) {
    return (inputGranularity == "b16" && resultGranularity == "b8") ||
           (inputGranularity == "b32" && resultGranularity == "b16");
  }

  [[maybe_unused]] static bool isSupportedShuffleValueType(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue32 ||
             intType.getWidth() == mlir::pto::kValue64;
    }
    if (auto vecType = dyn_cast<VectorType>(type)) {
      return vecType.getRank() == 1 && vecType.getDimSize(0) == mlir::pto::kValue2 &&
             vecType.getElementType().isF16();
    }
    return type.isF16() || type.isF32();
  }

  [[maybe_unused]] static bool isSupportedReduxValueType(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return intType.getWidth() == mlir::pto::kValue32;
    }
    return type.isF16() || type.isF32();
  }

  [[maybe_unused]] static LogicalResult verifyShuffleSemanticControl(Operation *op,
                                                    Type controlType,
                                                    IntegerAttr widthAttr,
                                                    StringRef ctrlName) {
    if (!isSupportedShuffleValueType(op->getResultTypes().front())) {
      return op->emitOpError()
             << "requires i32, i64, f16, f32 or vector<2xf16> value/result type";
    }
    if (!controlType.isInteger(mlir::pto::kValue32)) {
      return op->emitOpError() << "requires " << ctrlName
                               << " operand to be i32";
    }

    int64_t width = widthAttr.getInt();
    if (width != mlir::pto::kValue16 && width != mlir::pto::kValue32) {
      return op->emitOpError() << "requires width to be 16 or 32";
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyReduxSemanticType(Operation *op, Type valueType,
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

  [[maybe_unused]] static bool isSupportedVdupPosition(std::optional<StringRef> position) {
    return !position || *position == "LOWEST" || *position == "HIGHEST";
  }

  [[maybe_unused]] static std::optional<StringRef> getVdupMaskGranularity(Type elementType) {
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

  [[maybe_unused]] static bool isSupportedVtrcRoundMode(StringRef mode) {
    return mode == "R" || mode == "A" || mode == "F" || mode == "C" ||
           mode == "Z";
  }

  [[maybe_unused]] static bool isSupportedPartToken(StringRef part) {
    return part == "LOWER" || part == "HIGHER";
  }

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyWideningReductionVecOp(ReductionOp op,
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
      if (inputInt.getWidth() < mlir::pto::kValue8 ||
          inputInt.getWidth() > mlir::pto::kValue32) {
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

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifySameVecTripleOp(OpTy op) {
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

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyPartGranularityOp(OpTy op, bool widening) {
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

  template <typename BinaryMaskOp>
  [[maybe_unused]] static LogicalResult verifyBinaryMaskOp(BinaryMaskOp op) {
    if (failed(verifyMaskTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
        failed(verifyMaskTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
        failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
        failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type"))) {
      return failure();
    }
    return success();
  }

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyElementwiseVecScalarOpLike(OpTy op) {
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
  [[maybe_unused]] static LogicalResult verifyVecScalarMaskedOpLike(OpTy op) {
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
  [[maybe_unused]] static LogicalResult verifyCarryVecOpWithInput(CarryWithInputOp op) {
    if (failed(verifyCarryVecOp(op)) ||
        failed(verifyMaskTypeLike(op, op.getCarryIn().getType(),
                                  "carry_in type"))) {
      return failure();
    }
    return success();
  }

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyShiftScalarVecOp(OpTy op) {
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

  template <typename UnaryOp>
  [[maybe_unused]] static LogicalResult verifyUnaryVecOp(UnaryOp op) {
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

  template <typename BinaryOp>
  [[maybe_unused]] static LogicalResult verifyBinaryVecOp(BinaryOp op,
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

  template <typename TernaryOp>
  [[maybe_unused]] static LogicalResult verifyTernaryVecOp(TernaryOp op) {
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

  template <typename BinaryOp>
  [[maybe_unused]] static LogicalResult verifyShiftVecOp(BinaryOp op) {
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

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyReductionVecOp(ReductionOp op) {
    return verifyUnaryVecOp(op);
  }

  template <typename ReductionOp>
  [[maybe_unused]] static LogicalResult verifyGroupReductionVecOp(ReductionOp op) {
    if (failed(verifyReductionVecOp(op))) {
      return failure();
    }
    auto inputType = cast<VRegType>(op.getInput().getType());
    Type elemType = inputType.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      if (intType.getWidth() != mlir::pto::kValue8 &&
          intType.getWidth() != mlir::pto::kValue16 &&
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

  template <typename ExtremaOp>
  [[maybe_unused]] static LogicalResult verifyExtremaPredicateOp(ExtremaOp op) {
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
    if (!intType || (intType.getWidth() != mlir::pto::kValue8 &&
                     intType.getWidth() != mlir::pto::kValue16 &&
                     intType.getWidth() != mlir::pto::kValue32)) {
      return op.emitOpError("requires i8/i16/i32/f16/f32 vector element type");
    }
    return success();
  }

  template <typename SelectOp>
  [[maybe_unused]] static LogicalResult verifyLaneSelectOp(SelectOp op) {
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
  [[maybe_unused]] static LogicalResult verifyPairVecResults(PairOp op) {
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
  [[maybe_unused]] static LogicalResult verifyPartVecOp(PartOp op) {
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

  template <typename UnpackOp>
  [[maybe_unused]] static LogicalResult verifyUnpackVecOp(UnpackOp op) {
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
    if (srcType.getElementCount() != resultType.getElementCount() * mlir::pto::kValue2) {
      return op.emitOpError(
          "requires source element count to be twice the result element count");
    }
    unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
    unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
    if (srcWidth == 0 || srcWidth * mlir::pto::kValue2 != resultWidth) {
      return op.emitOpError(
          "requires result element width to be twice the source element width");
    }
    return success();
  }

  [[maybe_unused]] static bool isSupportedCmpMode(StringRef mode) {
    return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
           mode == "gt" || mode == "ge";
  }

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyVdupMaskGranularityLike(OpTy op, Type elemType) {
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

  template <typename OpTy>
  [[maybe_unused]] static LogicalResult verifyPackedInterleaveGranularity(OpTy op,
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

  template <typename BinaryVecNoMaskOp>
  [[maybe_unused]] static LogicalResult verifyBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
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
  [[maybe_unused]] static LogicalResult verifyFloatBinaryVecMaskOp(BinaryVecMaskOp op) {
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

  template <typename ConvOp>
  [[maybe_unused]] static LogicalResult verifyFusedConvVecOp(ConvOp op) {
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


} // namespace mlir::pto::vecop_detail

#endif // PTO_IR_VPTO_VPTOVECOP_INTERNAL_H
