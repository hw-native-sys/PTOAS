// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static bool isInsideSimtExecutionScope(Operation *op) {
  func::FuncOp func = getParentFunc(op);
  return (func && func->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static LogicalResult verifyInsideSimtExecutionScope(Operation *op) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName
           << "' or inside pto.section.simt";
  }
  return success();
}

static LogicalResult verifySimtKeepResumeCommon(Operation *op, int64_t slot) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "' or inside pto.section.simt";
  }
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op->emitOpError("requires slot in range [0, ")
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  return success();
}

LogicalResult SyncthreadsOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceBlockOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

void SyncthreadsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceBlockOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

static Operation *getLastSimtPayloadOp(KeepOp op) {
  Block *block = op->getBlock();
  bool insideSection =
      op->getParentOfType<SectionSimtOp>() != nullptr;
  if (insideSection)
    return block->empty() ? nullptr : &block->back();
  if (Operation *terminator = block->getTerminator()) {
    if (isa<func::ReturnOp>(terminator))
      return terminator->getPrevNode();
  }
  return nullptr;
}

static LogicalResult verifyKeepEpiloguePosition(KeepOp op) {
  Operation *lastPayloadOp = getLastSimtPayloadOp(op);
  if (!lastPayloadOp) {
    return op.emitOpError(
        "must be placed in the SIMT epilogue before func.return or the end "
        "of pto.section.simt");
  }

  Operation *cur = lastPayloadOp;
  while (cur && isa<SyncthreadsOp>(cur)) {
    cur = cur->getPrevNode();
  }
  Operation *lastKeep = cur;
  if (!lastKeep || !isa<KeepOp>(lastKeep)) {
    return op.emitOpError()
           << "must be placed in the SIMT epilogue before func.return; only "
              "'pto.syncthreads' may appear between the final 'pto.keep' group "
              "and func.return or the end of pto.section.simt";
  }

  Operation *firstKeep = lastKeep;
  while (Operation *prev = firstKeep->getPrevNode()) {
    if (!isa<KeepOp>(prev)) {
      break;
    }
    firstKeep = prev;
  }
  if (!isOpInRange(op, firstKeep, lastKeep)) {
    return op.emitOpError()
           << "must be in the contiguous SIMT keep epilogue group immediately "
              "before optional 'pto.syncthreads' and func.return or the end "
              "of pto.section.simt";
  }
  return verifyUniqueKeepGroupSlots(op, firstKeep, lastKeep);
}

LogicalResult KeepOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot())))
    return failure();
  if (!isSupportedSimtKeepResumeType(getPayload().getType()))
    return emitOpError()
           << "supports integer scalar payloads up to 64 bits and "
              "f16/bf16/f32 payloads";
  if (failed(verifySimtKeepResumeSlotRange(*this)))
    return failure();
  return verifyKeepEpiloguePosition(*this);
}

LogicalResult ResumeOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot()))) {
    return failure();
  }
  if (!isSupportedSimtKeepResumeType(getResult().getType())) {
    return emitOpError()
           << "supports integer scalar results up to 64 bits and "
              "f16/bf16/f32 results";
  }
  if (failed(verifySimtKeepResumeSlotRange(*this))) {
    return failure();
  }
  Block *block = getOperation()->getBlock();
  Operation *first = getFirstNonConstantLikeOp(block);
  if (!first || !isa<ResumeOp>(first)) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }

  bool found = false;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (!isa<ResumeOp>(cur)) {
      break;
    }
    if (cur == getOperation()) {
      found = true;
      break;
    }
  }
  if (!found) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }
  if (failed(verifyUniqueResumeGroupSlots(*this, first))) {
    return failure();
  }
  return success();
}

using MemoryEffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

static void addReadWriteEffects(MemoryEffectList &effects,
                                OpOperand &operand) {
  addEffect(effects, &operand, MemoryEffects::Read::get());
  addEffect(effects, &operand, MemoryEffects::Write::get());
}

static void addOptionalReadWriteEffects(MemoryEffectList &effects,
                                        mlir::MutableOperandRange operands) {
  if (auto it = operands.begin(); it != operands.end())
    addReadWriteEffects(effects, *it);
}

static void addPingPongEffects(MemoryEffectList &effects, OpOperand &ping,
                               mlir::MutableOperandRange pong) {
  addReadWriteEffects(effects, ping);
  addOptionalReadWriteEffects(effects, pong);
}

static void addGroupEffects(MemoryEffectList &effects,
                            mlir::MutableOperandRange group,
                            MemoryEffects::Effect *effect) {
  for (OpOperand &operand : group)
    addEffect(effects, &operand, effect);
}

static void addAsyncTransferEffects(MemoryEffectList &effects,
                                    OpOperand &dst, OpOperand &src,
                                    OpOperand &session, OpResult event) {
  addEffect(effects, &dst, MemoryEffects::Write::get());
  addEffect(effects, &src, MemoryEffects::Read::get());
  addEffect(effects, &session, MemoryEffects::Read::get());
  addEffect(effects, event, MemoryEffects::Write::get());
}

static void addSyncTransferEffects(MemoryEffectList &effects,
                                   OpOperand &dst, OpOperand &src,
                                   OpOperand &ping,
                                   mlir::MutableOperandRange pong) {
  addEffect(effects, &dst, MemoryEffects::Write::get());
  addEffect(effects, &src, MemoryEffects::Read::get());
  addPingPongEffects(effects, ping, pong);
}

void BuildAsyncSessionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScratchMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getWorkspaceMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addAsyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                          getSessionMutable(), getOperation()->getOpResult(0));
}

void TGetAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addAsyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                          getSessionMutable(), getOperation()->getOpResult(0));
}

void TPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addSyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                         getPingMutable(), getPongMutable());
}

void TGetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addSyncTransferEffects(effects, getDstMutable(), getSrcMutable(),
                         getPingMutable(), getPongMutable());
}

void TNotifyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getValueMutable(), MemoryEffects::Read::get());
}

void TWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
}

void TTestOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}
