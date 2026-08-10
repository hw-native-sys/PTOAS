// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOScheduling.cpp - VPTO scheduling semantics --------------------===//

#include "PTO/IR/VPTOScheduling.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
static std::optional<PIPE> getExecutionPipe(Operation *op) {
  if (auto pipeOp = dyn_cast<OpPipeInterface>(op))
    return pipeOp.getPipe();
  if (isa<VectorMicroOpInterface>(op))
    return PIPE::PIPE_V;
  if (isa<CubeMicroOpInterface>(op))
    return PIPE::PIPE_M;
  if (isa<SimtOpInterface>(op))
    return PIPE::PIPE_S;
  // Raw MTE micro-ops do not all expose a precise OpPipeInterface yet.  Use
  // PIPE_ALL as an unknown-pipe effect so synchronization remains conservative.
  if (isa<MteOpInterface>(op))
    return PIPE::PIPE_ALL;
  return std::nullopt;
}

static PipeAttr getBufferSyncPipe(Operation *op, Attribute opType) {
  if (auto pipe = dyn_cast_or_null<PipeAttr>(opType))
    return pipe;
  FailureOr<SyncOpType> syncType = parseSyncOpTypeLikeAttr(opType);
  if (failed(syncType))
    return {};
  PIPE pipe = mapSyncOpTypeToPipe(*syncType);
  if (!isConcreteSyncPipe(pipe))
    return {};
  return PipeAttr::get(op->getContext(), pipe);
}
} // namespace

VPTOSchedulingClass mlir::pto::getDefaultVPTOSchedulingClass(Operation *op) {
  if (getExecutionPipe(op))
    return VPTOSchedulingClass::Schedulable;
  SmallVector<VPTOSchedulingEffect> effects;
  getDefaultVPTOSchedulingEffects(op, effects);
  return effects.empty() ? VPTOSchedulingClass::SchedulingBoundary
                         : VPTOSchedulingClass::Schedulable;
}

void mlir::pto::getDefaultVPTOSchedulingEffects(
    Operation *op, SmallVectorImpl<VPTOSchedulingEffect> &effects) {
  if (isa<MemBarOp, DsbOp, FenceBarrierAllOp, SyncthreadsOp, ThreadfenceOp,
          ThreadfenceBlockOp>(op)) {
    effects.push_back(
        {VPTOSchedulingEffectKind::Barrier, "memory-order", Value()});
  }
  auto addStaticEvent = [&](Attribute srcPipe, Attribute dstPipe,
                            Attribute eventId, StringRef access) {
    effects.push_back({VPTOSchedulingEffectKind::Event, access, Value(),
                       ArrayAttr::get(op->getContext(),
                                      {srcPipe, dstPipe, eventId})});
  };
  auto addDynamicEvent = [&](Attribute srcPipe, Attribute dstPipe,
                             Value eventId, StringRef access) {
    effects.push_back({VPTOSchedulingEffectKind::Event, access, eventId,
                       ArrayAttr::get(op->getContext(), {srcPipe, dstPipe})});
  };
  if (auto set = dyn_cast<SetFlagOp>(op))
    addStaticEvent(set.getSrcPipeAttr(), set.getDstPipeAttr(),
                   set.getEventIdAttr(), "signal");
  if (auto wait = dyn_cast<WaitFlagOp>(op))
    addStaticEvent(wait.getSrcPipeAttr(), wait.getDstPipeAttr(),
                   wait.getEventIdAttr(), "wait");
  if (auto set = dyn_cast<SetFlagDynOp>(op))
    addDynamicEvent(set.getSrcPipeAttr(), set.getDstPipeAttr(),
                    set.getEventId(), "signal");
  if (auto wait = dyn_cast<WaitFlagDynOp>(op))
    addDynamicEvent(wait.getSrcPipeAttr(), wait.getDstPipeAttr(),
                    wait.getEventId(), "wait");
  if (auto set = dyn_cast<SetFlagOp>(op))
    effects.emplace_back(VPTOSchedulingEffectKind::Pipe, "signal-source",
                         Value(), set.getSrcPipeAttr());
  if (auto set = dyn_cast<SetFlagDynOp>(op))
    effects.emplace_back(VPTOSchedulingEffectKind::Pipe, "signal-source",
                         Value(), set.getSrcPipeAttr());
  if (auto wait = dyn_cast<WaitFlagOp>(op))
    effects.emplace_back(VPTOSchedulingEffectKind::Pipe, "wait-destination",
                         Value(), wait.getDstPipeAttr());
  if (auto wait = dyn_cast<WaitFlagDynOp>(op))
    effects.emplace_back(VPTOSchedulingEffectKind::Pipe, "wait-destination",
                         Value(), wait.getDstPipeAttr());
  if (auto barrier = dyn_cast<BarrierOp>(op))
    effects.emplace_back(VPTOSchedulingEffectKind::Pipe, "barrier", Value(),
                         barrier.getPipeAttr());
  if (std::optional<PIPE> pipe = getExecutionPipe(op))
    effects.emplace_back(VPTOSchedulingEffectKind::Pipe, "execute", Value(),
                         PipeAttr::get(op->getContext(), *pipe));
  auto addStaticBuffer = [&](Attribute opType, IntegerAttr bufferId,
                             uint32_t mode, StringRef access) {
    effects.emplace_back(VPTOSchedulingEffectKind::BufferId, access, Value(),
                         bufferId);
    PipeAttr pipe = getBufferSyncPipe(op, opType);
    if (!pipe)
      return;
    if (access == "acquire" && mode == 0)
      effects.emplace_back(VPTOSchedulingEffectKind::Pipe,
                           "wait-destination", Value(), pipe);
    if (access == "release")
      effects.emplace_back(
          VPTOSchedulingEffectKind::Pipe, "signal-source", Value(),
          mode == 0 ? Attribute(pipe)
                    : Attribute(PipeAttr::get(op->getContext(), PIPE::PIPE_ALL)));
  };
  auto addDynamicBuffer = [&](Attribute opType, Value bufferId, uint32_t mode,
                              StringRef access) {
    effects.emplace_back(VPTOSchedulingEffectKind::BufferId, access, bufferId);
    PipeAttr pipe = getBufferSyncPipe(op, opType);
    if (!pipe)
      return;
    if (access == "acquire" && mode == 0)
      effects.emplace_back(VPTOSchedulingEffectKind::Pipe,
                           "wait-destination", Value(), pipe);
    if (access == "release")
      effects.emplace_back(
          VPTOSchedulingEffectKind::Pipe, "signal-source", Value(),
          mode == 0 ? Attribute(pipe)
                    : Attribute(PipeAttr::get(op->getContext(), PIPE::PIPE_ALL)));
  };
  if (auto acquire = dyn_cast<GetBufOp>(op))
    addStaticBuffer(acquire.getOpTypeAttr(), acquire.getBufIdAttr(),
                    acquire.getMode(), "acquire");
  if (auto acquire = dyn_cast<GetBufDynOp>(op))
    addDynamicBuffer(acquire.getOpTypeAttr(), acquire.getBufId(),
                     acquire.getMode(), "acquire");
  if (auto release = dyn_cast<RlsBufOp>(op))
    addStaticBuffer(release.getOpTypeAttr(), release.getBufIdAttr(),
                    release.getMode(), "release");
  if (auto release = dyn_cast<RlsBufDynOp>(op))
    addDynamicBuffer(release.getOpTypeAttr(), release.getBufId(),
                     release.getMode(), "release");
  if (isa<AtomicCasOp, AtomicExchOp, AtomicAddOp, AtomicSubOp, AtomicMinOp,
          AtomicMaxOp, AtomicAndOp, AtomicOrOp, AtomicXorOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::AtomicMemory, "memory", Value()});
  if (op->hasAttr("volatile") || op->hasAttr("is_volatile"))
    effects.push_back(
        {VPTOSchedulingEffectKind::VolatileMemory, "memory", Value()});
  auto addPostUpdate = [&](Value updatedBase) {
    if (updatedBase)
      effects.push_back({VPTOSchedulingEffectKind::PostUpdate,
                         "updated-address", updatedBase});
  };
  if (auto typedOp = dyn_cast<VldsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<Vldsx2Op>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<SprstiOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<SprstsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VldusOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PldsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PldiOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PstiOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VstsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<PstsOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VsldbOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VsstbOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto typedOp = dyn_cast<VstasOp>(op))
    addPostUpdate(typedOp.getUpdatedBase());
  if (auto sprclr = dyn_cast<SprclrOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitWrite, sprclr.getSpr(), Value()});
  if (auto sprsti = dyn_cast<SprstiOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, sprsti.getSpr(), Value()});
  if (auto sprsts = dyn_cast<SprstsOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, sprsts.getSpr(), Value()});
  if (isa<GetCtrlOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitRead, "ctrl", Value()});
  if (isa<SetCtrlOp>(op))
    effects.push_back(
        {VPTOSchedulingEffectKind::ImplicitWrite, "ctrl", Value()});
}

VPTOSchedulingClass mlir::pto::classifyVPTOSchedulingOp(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() != 0)
    return VPTOSchedulingClass::SchedulingBoundary;

  if (auto scheduling = dyn_cast<VPTOSchedulingOpInterface>(op))
    return scheduling.getVPTOSchedulingClass();

  if (isMemoryEffectFree(op))
    return VPTOSchedulingClass::Structural;

  if (op->getDialect() &&
      op->getDialect()->getNamespace() == PTODialect::getDialectNamespace())
    return VPTOSchedulingClass::Unsupported;

  return VPTOSchedulingClass::SchedulingBoundary;
}

StringRef mlir::pto::stringifyVPTOSchedulingClass(VPTOSchedulingClass value) {
  switch (value) {
  case VPTOSchedulingClass::Schedulable:
    return "schedulable";
  case VPTOSchedulingClass::Structural:
    return "structural";
  case VPTOSchedulingClass::SchedulingBoundary:
    return "boundary";
  case VPTOSchedulingClass::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown VPTO scheduling class");
}
