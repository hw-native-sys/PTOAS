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

void TRowExpandSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TRowExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowExpandMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowExpandMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TRowProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty())
      PTO_ADD_READ(effects, idx[0]);
  }
  PTO_ADD_WRITE(effects, getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getMaskMutable());
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TSELS: Read(src0, src1) -> Write(tmp, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getMaskMutable());
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_READ(effects, getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(effects, tmp[0]);
  PTO_ADD_WRITE(effects, getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp?, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrc0Mutable());
  PTO_ADD_READ(effects, getSrc1Mutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getTmpMutable());
  PTO_ADD_WRITE(effects, getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(effects, getSrcMutable());
  PTO_ADD_WRITE(effects, getSrcMutable());
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
