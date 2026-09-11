// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVecOpArith.h - shared vecop Arith helpers ----------------------===//
//===----------------------------------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
//
// Arith helper group of the vecop per-instruction TUs.
// Internal to lib/PTO/IR/VPTO/vecop/detail; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VECOP_DETAIL_ARITH_H
#define PTO_IR_VPTO_VECOP_DETAIL_ARITH_H

#include "VPTOInternal.h"

namespace mlir::pto::vecop_detail {

using namespace mlir;
using namespace mlir::pto;

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

#endif // PTO_IR_VPTO_VECOP_DETAIL_ARITH_H
