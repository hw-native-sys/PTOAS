// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

void TQuantMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  Type srcTy = getSrc().getType();
  auto valid = getValidShapeVec(srcTy);
  auto physical = getShapeVec(srcTy);
  Type elem = getElemTy(srcTy);
  if ((elem.isF16() || elem.isBF16()) && valid.size() == mlir::pto::kValue2 && physical.size() == mlir::pto::kValue2 &&
      valid[1] < physical[1]) {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
      addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
  } else {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  }
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getExpMutable());
  PTO_ADD_WRITE(getMaxMutable());
  PTO_ADD_WRITE(getScalingMutable());
  auto expZzRange = getExpZzMutable();
  if (!expZzRange.empty()) {
    PTO_ADD_WRITE(expZzRange[0]);
  }
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRemOp, getSrc0Mutable(), getSrc1Mutable(),
                                  getTmpMutable(), getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRemSOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TPowOp, getBaseMutable(), getExpMutable(),
                                  getTmpMutable(), getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TPowSOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandDivOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandMulOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandSubOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandAddOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandMaxOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())
PTO_DEFINE_BINARY_SCRATCH_EFFECTS(TRowExpandMinOp, getSrc0Mutable(),
                                  getSrc1Mutable(), getTmpMutable(),
                                  getDstMutable())

// Row reductions use tmp scratch tile.
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowMaxOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowMinOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowSumOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
PTO_DEFINE_UNARY_SCRATCH_EFFECTS(TRowProdOp, getSrcMutable(), getTmpMutable(),
                                 getDstMutable())
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  addOptionalEffects(effects, getTmpMutable(), /*read=*/true, /*write=*/true);
  PTO_ADD_WRITE(getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty()) {
      PTO_ADD_READ(idx[0]);
    }
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp on A2/A3, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TSEL; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(mask, src) -> Write(tmp on A2/A3, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TSELS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp on A2/A3, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TXORS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp on A2/A3, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TXOR; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  addA2A3ScratchEffects(effects, getOperation(), getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && ttransUsesTmp(getSrc().getType(), getDst().getType())) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (!getTmpMutable().empty()) {
    PTO_ADD_WRITE(getTmpMutable()[0]);
  }
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
