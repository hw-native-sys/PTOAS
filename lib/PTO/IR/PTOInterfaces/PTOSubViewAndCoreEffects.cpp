// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static LogicalResult verifySubViewBoxed(SubViewOp op, TileBufType srcTy,
                                        const SubViewInfo &info) {
  auto cfg = srcTy.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(op.getContext());
  }

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl))) {
    return op.emitOpError("unsupported tile layout for subview");
  }

  if (!boxed) {
    return success();
  }

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  if (info.sizeR % innerRows != 0 || info.sizeC % innerCols != 0) {
    return op.emitOpError("boxed layout subview sizes must be multiples of inner shape");
  }

  if (info.offRConst) {
    if (info.offR % innerRows != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }
  if (info.offCConst) {
    if (info.offC % innerCols != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }

  (void)bl;
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != mlir::pto::kValue2 || srcShape[0] == ShapedType::kDynamic ||
      srcShape[1] == ShapedType::kDynamic) {
      return op.emitOpError("boxed layout subview requires static source shape");
  }

  return success();
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }
  if (srcTy.getRank() != mlir::pto::kValue2 || dstTy.getRank() != mlir::pto::kValue2) {
      return emitOpError("expects rank-2 tilebuf for src/dst");
  }

  SubViewInfo info;
  if (failed(verifySubViewSizesAndOffsets(*this, info))) {
    return failure();
  }
  if (failed(verifySubViewValidBounds(*this, info.sizeR, info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewShapeAndConfig(*this, srcTy, dstTy, info.sizeR,
                                         info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewValidShape(*this, dstTy, info.sizeR, info.sizeC))) {
    return failure();
  }
  return verifySubViewBoxed(*this, srcTy, info);
}

} // namespace pto
} // namespace mlir

// =============================================================================
// Helper Functions
// =============================================================================

// =============================================================================
// Side Effects Implementation
// =============================================================================

// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value

// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand) {
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
  }
}

using PTOEffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

static void addOptionalEffects(PTOEffectList &effects,
                               mlir::MutableOperandRange operands,
                               bool read, bool write) {
  if (operands.empty())
    return;
  OpOperand &operand = *operands.begin();
  if (read)
    addEffect(effects, &operand, MemoryEffects::Read::get());
  if (write)
    addEffect(effects, &operand, MemoryEffects::Write::get());
}

static void addA2A3ScratchEffects(PTOEffectList &effects, Operation *op,
                                  mlir::MutableOperandRange scratch) {
  if (getTargetArch(op) != PTOArch::A5)
    addOptionalEffects(effects, scratch, /*read=*/true, /*write=*/true);
}

static void addStoreLikeEffects(PTOEffectList &effects, OpOperand &src,
                                mlir::MutableOperandRange fp,
                                mlir::MutableOperandRange preQuant,
                                OpOperand &dst) {
  addEffect(effects, &src, MemoryEffects::Read::get());
  addOptionalEffects(effects, fp, /*read=*/true, /*write=*/false);
  addOptionalEffects(effects, preQuant, /*read=*/true, /*write=*/false);
  addEffect(effects, &dst, MemoryEffects::Write::get());
}

// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result) {
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
  }
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

void TPrefetchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addStoreLikeEffects(effects, getSrcMutable(), getFpMutable(),
                      getPreQuantScalarMutable(), getDstMutable());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (classifyTMovForm(getFp()) == TMovForm::XToZz) {
    const MxGroupAxis axis = getGrpAxisAttr()
                                 ? getGrpAxisAttr().getValue()
                                 : MxGroupAxis::Axis1;
    if (axis == MxGroupAxis::Axis1) {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
      addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
    } else {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
    }
    addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
    auto fp = getFpMutable();
    if (axis == MxGroupAxis::Axis1 && !fp.empty()) {
      addEffect(effects, &fp[0], MemoryEffects::Read::get());
      addEffect(effects, &fp[0], MemoryEffects::Write::get());
    }
    return;
  }
  addStoreLikeEffects(effects, getSrcMutable(), getFpMutable(),
                      getPreQuantScalarMutable(), getDstMutable());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_QUATERNARY_EFFECTS(OpClass, op0, op1, op2, op3, dstOperand)      \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_READ(op3);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_UNARY_SCRATCH_EFFECTS(OpClass, srcOperand, tmpOperand,          \
                                         dstOperand)                               \
  void OpClass::getEffects(PTOEffectList &effects) {                               \
    PTO_ADD_READ(srcOperand);                                                      \
    addA2A3ScratchEffects(effects, getOperation(), tmpOperand);                    \
    PTO_ADD_WRITE(dstOperand);                                                     \
  }

#define PTO_DEFINE_BINARY_SCRATCH_EFFECTS(OpClass, lhsOperand, rhsOperand,         \
                                          tmpOperand, dstOperand)                  \
  void OpClass::getEffects(PTOEffectList &effects) {                               \
    PTO_ADD_READ(lhsOperand);                                                      \
    PTO_ADD_READ(rhsOperand);                                                      \
    addA2A3ScratchEffects(effects, getOperation(), tmpOperand);                    \
    PTO_ADD_WRITE(dstOperand);                                                     \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  // GM -> L1 Elem mode stages the gathered elements into the GM scratch buffer
  // before the bulk copy: the op clobbers scratch, so model it as a write.
  auto scratchRange = getScratchMutable();
  if (!scratchRange.empty()) {
    addEffect(effects, &*scratchRange.begin(), MemoryEffects::Write::get());
  }
}
