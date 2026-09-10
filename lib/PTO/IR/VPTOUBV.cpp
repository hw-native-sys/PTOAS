// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOUBV.cpp - VPTO UBV op implementations ---------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

//===----------------------------------------------------------------------===//
// UBVaddOp
//===----------------------------------------------------------------------===//

void UBVaddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

static LogicalResult verifyUBBinaryOperands(Operation *op, Value dst,
                                                Value src0, Value src1) {
  if (!isBufferLike(dst.getType()) || !isBufferLike(src0.getType()) ||
      !isBufferLike(src1.getType())) {
    return op->emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(dst.getType()) != MemoryRole::UB ||
      classifyMemoryRole(src0.getType()) != MemoryRole::UB ||
      classifyMemoryRole(src1.getType()) != MemoryRole::UB) {
    return op->emitOpError("requires UB-backed operands");
  }
  return success();
}

static LogicalResult verifyUBUnaryOperands(Operation *op, Value dst, Value src) {
  if (!isBufferLike(dst.getType()) || !isBufferLike(src.getType())) {
    return op->emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(dst.getType()) != MemoryRole::UB ||
      classifyMemoryRole(src.getType()) != MemoryRole::UB) {
    return op->emitOpError("requires UB-backed operands");
  }
  return success();
}

LogicalResult UBVaddOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVsubOp
//===----------------------------------------------------------------------===//

void UBVsubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVsubOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVmulOp
//===----------------------------------------------------------------------===//

void UBVmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmulOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVdivOp
//===----------------------------------------------------------------------===//

void UBVdivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVdivOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVmaxOp
//===----------------------------------------------------------------------===//

void UBVmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmaxOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVminOp
//===----------------------------------------------------------------------===//

void UBVminOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVminOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVandOp
//===----------------------------------------------------------------------===//

void UBVandOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVandOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

//===----------------------------------------------------------------------===//
// UBVorOp
//===----------------------------------------------------------------------===//

void UBVorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVorOp::verify() {
  return verifyUBBinaryOperands(getOperation(), getDst(), getSrc0(), getSrc1());
}

void UBVaddReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVaddReluOp::verify() {
  if (!isBufferLike(getDst().getType()) ||
      !isBufferLike(getSrc0().getType()) ||
      !isBufferLike(getSrc1().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc1().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVnotOp
//===----------------------------------------------------------------------===//

void UBVnotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVnotOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

//===----------------------------------------------------------------------===//
// UBVabsOp
//===----------------------------------------------------------------------===//

void UBVabsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVabsOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

//===----------------------------------------------------------------------===//
// UBVreluOp
//===----------------------------------------------------------------------===//

void UBVreluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVreluOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVexpOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVexpOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVlnOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVlnOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVsqrtOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVrsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVrsqrtOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVaddSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVaddSOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVmaxSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmaxSOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVminSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVminSOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

void UBVdupOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVdupOp::verify() {
  if (!isBufferLike(getDst().getType())) {
    return emitOpError("requires pointer-like dst operand");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed dst operand");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVgatherbOp
//===----------------------------------------------------------------------===//

void UBVgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVgatherbOp::verify() {
  if (!isBufferLike(getDst().getType()) || !isBufferLike(getOffset().getType()) ||
      !isBufferLike(getSrc().getType())) {
    return emitOpError("requires pointer-like operands");
  }
  if (classifyMemoryRole(getDst().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getOffset().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSrc().getType()) != MemoryRole::UB) {
    return emitOpError("requires UB-backed operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UBVgatherOp
//===----------------------------------------------------------------------===//

void UBVgatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVgatherOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

//===----------------------------------------------------------------------===//
// UBVshlOp
//===----------------------------------------------------------------------===//

void UBVshlOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVshlOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

//===----------------------------------------------------------------------===//
// UBVshrOp
//===----------------------------------------------------------------------===//

void UBVshrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVshrOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}

//===----------------------------------------------------------------------===//
// UBVmulSOp
//===----------------------------------------------------------------------===//

void UBVmulSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable());
}

LogicalResult UBVmulSOp::verify() {
  return verifyUBUnaryOperands(getOperation(), getDst(), getSrc());
}
