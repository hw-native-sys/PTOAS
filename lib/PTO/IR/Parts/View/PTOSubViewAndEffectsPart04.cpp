// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOSubViewAndEffects.cpp; kept as a fragment included by PTOSubViewAndEffects.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

void TColArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TColArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(effects, tmp[0]);
  }
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TCvtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(effects, getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_READ(effects, getScalarMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TEXTRACT_FP: Read(src), Read(fp) -> Write(dst)
void TExtractFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_READ(effects, getFpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TINSERT_FP: Read(src), Read(fp) -> Write(dst)
void TInsertFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_READ(effects, getFpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadInplaceOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty())
    PTO_ADD_WRITE(effects, cdst[0]);
  if (auto indices = getIndicesMutable(); !indices.empty())
    PTO_ADD_READ(effects, indices[0]);
  if (auto tmp = getTmpMutable(); !tmp.empty())
    PTO_ADD_READ(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMovFPOp, getSrcMutable(), getFpMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(effects, opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(effects, opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(effects, executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TPartArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  PTO_ADD_READ(effects, getSrc0IdxMutable());
  PTO_ADD_READ(effects, getSrc1IdxMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
  PTO_ADD_WRITE(effects, getDstIdxMutable());
}
void TPartArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  PTO_ADD_READ(effects, getSrc0IdxMutable());
  PTO_ADD_READ(effects, getSrc1IdxMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
  PTO_ADD_WRITE(effects, getDstIdxMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_READ(effects, getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty())
    PTO_ADD_READ(effects, offsetRange[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
void TRemOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRemSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

void TRowExpandDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowExpandMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

