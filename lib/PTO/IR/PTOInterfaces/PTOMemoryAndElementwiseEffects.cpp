// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

void THistogramOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TGetScaleAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// SET_VALIDSHAPE: update runtime valid row/col metadata on source tile in-place.
void SetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getSourceMutable());
}

// GET_VALIDSHAPE: read runtime valid row/col metadata from source tile.
void GetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSourceMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddReluOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TAxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TConcatOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_QUATERNARY_EFFECTS(TConcatidxOp, getSrc0Mutable(), getSrc1Mutable(), getSrc0IdxMutable(), getSrc1IdxMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (auto tmp = getTmpMutable();
      !tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TTRI: Write(dst) (generates triangular mask)
void TTriOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandExpdifOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColProdOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TColArgMaxOp, getSrcMutable(),
                                 getTmpMutable(), getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TColArgMinOp, getSrcMutable(),
                                 getTmpMutable(), getDstMutable())

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  addOptionalEffects(effects, getTmpMutable(), /*read=*/true, /*write=*/true);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TCvtOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addOptionalEffects(effects, getFpMutable(), /*read=*/true, /*write=*/false);
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addOptionalEffects(effects, getFpMutable(), /*read=*/true, /*write=*/false);
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty()) {
    PTO_ADD_WRITE(cdst[0]);
  }
  if (auto indices = getIndicesMutable(); !indices.empty()) {
    PTO_ADD_READ(indices[0]);
  }
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
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
void TInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getDst0Mutable());
  PTO_ADD_WRITE(getDst1Mutable());
}
void TDeInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &operand : getSrcsMutable()) {
    PTO_ADD_READ(operand);
  }
  for (auto &operand : getDstsMutable()) {
    PTO_ADD_WRITE(operand);
  }
}
#define PTO_DEFINE_PART_ARG_EFFECTS(OpClass)                                       \
  void OpClass::getEffects(PTOEffectList &effects) {                               \
    PTO_ADD_READ(getSrc0Mutable());                                                \
    PTO_ADD_READ(getSrc1Mutable());                                                \
    PTO_ADD_READ(getSrc0IdxMutable());                                             \
    PTO_ADD_READ(getSrc1IdxMutable());                                             \
    PTO_ADD_WRITE(getDstMutable());                                                \
    PTO_ADD_WRITE(getDstIdxMutable());                                             \
  }
PTO_DEFINE_PART_ARG_EFFECTS(TPartArgMaxOp)
PTO_DEFINE_PART_ARG_EFFECTS(TPartArgMinOp)
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty()) {
    PTO_ADD_READ(offsetRange[0]);
  }
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}
